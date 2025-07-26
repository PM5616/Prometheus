"""Authentication Module

认证管理模块，负责API网关的认证和授权。

主要功能：
- 多种认证方式支持
- 用户权限管理
- Token管理
- 会话管理
"""

import time
import hashlib
import hmac
import base64
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from aiohttp import web
import aioredis

from ..common.logging.logger import get_logger
from ..common.exceptions.gateway_exceptions import AuthenticationError, AuthorizationError


class AuthType(Enum):
    """认证类型"""
    NONE = "none"                    # 无认证
    BASIC = "basic"                  # Basic认证
    BEARER = "bearer"                # Bearer Token
    JWT = "jwt"                      # JWT Token
    API_KEY = "api_key"              # API Key
    OAUTH2 = "oauth2"                # OAuth2
    CUSTOM = "custom"                # 自定义认证


class Permission(Enum):
    """权限类型"""
    READ = "read"                    # 读权限
    WRITE = "write"                  # 写权限
    DELETE = "delete"                # 删除权限
    ADMIN = "admin"                  # 管理员权限
    EXECUTE = "execute"              # 执行权限


@dataclass
class User:
    """用户信息"""
    user_id: str
    username: str
    email: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[Permission] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    is_active: bool = True
    
    def has_permission(self, permission: Permission) -> bool:
        """检查是否有指定权限
        
        Args:
            permission: 权限类型
            
        Returns:
            是否有权限
        """
        return permission in self.permissions or Permission.ADMIN in self.permissions
    
    def has_role(self, role: str) -> bool:
        """检查是否有指定角色
        
        Args:
            role: 角色名称
            
        Returns:
            是否有角色
        """
        return role in self.roles


@dataclass
class ApiKey:
    """API密钥"""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    permissions: List[Permission] = field(default_factory=list)
    rate_limit: Optional[int] = None
    expires_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """检查是否过期
        
        Returns:
            是否过期
        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def verify_key(self, key: str) -> bool:
        """验证密钥
        
        Args:
            key: 原始密钥
            
        Returns:
            是否匹配
        """
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return hmac.compare_digest(self.key_hash, key_hash)


@dataclass
class AuthConfig:
    """认证配置"""
    # 基本配置
    default_auth_type: AuthType = AuthType.JWT
    require_auth: bool = True
    
    # JWT配置
    jwt_secret: str = "your-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # 1小时
    jwt_refresh_expiration: int = 86400  # 24小时
    
    # API Key配置
    api_key_header: str = "X-API-Key"
    api_key_query_param: str = "api_key"
    
    # OAuth2配置
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None
    oauth2_authorize_url: Optional[str] = None
    oauth2_token_url: Optional[str] = None
    oauth2_userinfo_url: Optional[str] = None
    
    # 会话配置
    session_timeout: int = 1800  # 30分钟
    max_sessions_per_user: int = 5
    
    # 安全配置
    password_min_length: int = 8
    password_require_special: bool = True
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15分钟
    
    # Redis配置（用于会话存储）
    redis_url: Optional[str] = None
    redis_key_prefix: str = "auth:"


@dataclass
class AuthResult:
    """认证结果"""
    success: bool
    user: Optional[User] = None
    api_key: Optional[ApiKey] = None
    token: Optional[str] = None
    error_message: Optional[str] = None
    permissions: List[Permission] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuthManager:
    """认证管理器
    
    负责用户认证、授权和会话管理。
    """
    
    def __init__(self, config: AuthConfig):
        """初始化认证管理器
        
        Args:
            config: 认证配置
        """
        self.config = config
        self.logger = get_logger("auth_manager")
        
        # 存储
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, ApiKey] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.login_attempts: Dict[str, List[float]] = {}
        
        # Redis连接
        self.redis: Optional[aioredis.Redis] = None
        
        # 自定义认证器
        self.custom_authenticators: Dict[str, Callable] = {}
        
        # 统计信息
        self.stats = {
            'total_authentications': 0,
            'successful_authentications': 0,
            'failed_authentications': 0,
            'active_sessions': 0,
            'api_key_usage': 0,
            'jwt_issued': 0
        }
        
        self.logger.info("认证管理器初始化完成")
    
    async def initialize(self) -> None:
        """初始化异步组件"""
        # 初始化Redis连接
        if self.config.redis_url:
            try:
                self.redis = aioredis.from_url(self.config.redis_url)
                await self.redis.ping()
                self.logger.info("Redis连接初始化成功")
            except Exception as e:
                self.logger.error(f"Redis连接初始化失败: {e}")
                self.redis = None
    
    async def authenticate(self, request: web.Request) -> AuthResult:
        """认证请求
        
        Args:
            request: HTTP请求
            
        Returns:
            认证结果
        """
        self.stats['total_authentications'] += 1
        
        try:
            # 检测认证类型
            auth_type = self._detect_auth_type(request)
            
            # 执行认证
            if auth_type == AuthType.NONE:
                result = AuthResult(success=True)
            elif auth_type == AuthType.BASIC:
                result = await self._authenticate_basic(request)
            elif auth_type == AuthType.BEARER:
                result = await self._authenticate_bearer(request)
            elif auth_type == AuthType.JWT:
                result = await self._authenticate_jwt(request)
            elif auth_type == AuthType.API_KEY:
                result = await self._authenticate_api_key(request)
            elif auth_type == AuthType.OAUTH2:
                result = await self._authenticate_oauth2(request)
            elif auth_type == AuthType.CUSTOM:
                result = await self._authenticate_custom(request)
            else:
                result = AuthResult(
                    success=False,
                    error_message=f"不支持的认证类型: {auth_type}"
                )
            
            # 更新统计
            if result.success:
                self.stats['successful_authentications'] += 1
                
                # 更新用户最后登录时间
                if result.user:
                    result.user.last_login = time.time()
                
                # 更新API Key使用时间
                if result.api_key:
                    result.api_key.last_used = time.time()
                    self.stats['api_key_usage'] += 1
            else:
                self.stats['failed_authentications'] += 1
                
                # 记录失败尝试
                client_ip = request.remote or 'unknown'
                await self._record_failed_attempt(client_ip)
            
            return result
            
        except Exception as e:
            self.logger.error(f"认证过程出错: {e}")
            self.stats['failed_authentications'] += 1
            return AuthResult(
                success=False,
                error_message=f"认证失败: {str(e)}"
            )
    
    def _detect_auth_type(self, request: web.Request) -> AuthType:
        """检测认证类型
        
        Args:
            request: HTTP请求
            
        Returns:
            认证类型
        """
        # 检查Authorization头
        auth_header = request.headers.get('Authorization', '')
        
        if auth_header.startswith('Basic '):
            return AuthType.BASIC
        elif auth_header.startswith('Bearer '):
            # 进一步检查是否为JWT
            token = auth_header[7:]
            if self._is_jwt_token(token):
                return AuthType.JWT
            else:
                return AuthType.BEARER
        
        # 检查API Key
        api_key_header = request.headers.get(self.config.api_key_header)
        api_key_query = request.query.get(self.config.api_key_query_param)
        
        if api_key_header or api_key_query:
            return AuthType.API_KEY
        
        # 检查自定义认证
        if self.custom_authenticators:
            return AuthType.CUSTOM
        
        # 默认认证类型
        return self.config.default_auth_type
    
    def _is_jwt_token(self, token: str) -> bool:
        """检查是否为JWT Token
        
        Args:
            token: Token字符串
            
        Returns:
            是否为JWT
        """
        try:
            parts = token.split('.')
            return len(parts) == 3
        except:
            return False
    
    async def _authenticate_basic(self, request: web.Request) -> AuthResult:
        """Basic认证
        
        Args:
            request: HTTP请求
            
        Returns:
            认证结果
        """
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Basic '):
            return AuthResult(
                success=False,
                error_message="缺少Basic认证头"
            )
        
        try:
            # 解码Basic认证
            encoded_credentials = auth_header[6:]
            decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8')
            username, password = decoded_credentials.split(':', 1)
            
            # 验证用户
            user = await self._verify_user_credentials(username, password)
            if user:
                return AuthResult(
                    success=True,
                    user=user,
                    permissions=user.permissions
                )
            else:
                return AuthResult(
                    success=False,
                    error_message="用户名或密码错误"
                )
                
        except Exception as e:
            return AuthResult(
                success=False,
                error_message=f"Basic认证失败: {str(e)}"
            )
    
    async def _authenticate_bearer(self, request: web.Request) -> AuthResult:
        """Bearer Token认证
        
        Args:
            request: HTTP请求
            
        Returns:
            认证结果
        """
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return AuthResult(
                success=False,
                error_message="缺少Bearer Token"
            )
        
        token = auth_header[7:]
        
        # 验证Token（这里需要实现具体的Token验证逻辑）
        # 例如：查询数据库、调用外部服务等
        
        return AuthResult(
            success=False,
            error_message="Bearer Token认证未实现"
        )
    
    async def _authenticate_jwt(self, request: web.Request) -> AuthResult:
        """JWT认证
        
        Args:
            request: HTTP请求
            
        Returns:
            认证结果
        """
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return AuthResult(
                success=False,
                error_message="缺少JWT Token"
            )
        
        token = auth_header[7:]
        
        try:
            # 验证JWT
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # 检查过期时间
            if 'exp' in payload and payload['exp'] < time.time():
                return AuthResult(
                    success=False,
                    error_message="JWT Token已过期"
                )
            
            # 获取用户信息
            user_id = payload.get('user_id')
            if user_id and user_id in self.users:
                user = self.users[user_id]
                return AuthResult(
                    success=True,
                    user=user,
                    token=token,
                    permissions=user.permissions
                )
            else:
                return AuthResult(
                    success=False,
                    error_message="用户不存在"
                )
                
        except jwt.ExpiredSignatureError:
            return AuthResult(
                success=False,
                error_message="JWT Token已过期"
            )
        except jwt.InvalidTokenError as e:
            return AuthResult(
                success=False,
                error_message=f"无效的JWT Token: {str(e)}"
            )
    
    async def _authenticate_api_key(self, request: web.Request) -> AuthResult:
        """API Key认证
        
        Args:
            request: HTTP请求
            
        Returns:
            认证结果
        """
        # 从头部或查询参数获取API Key
        api_key = (
            request.headers.get(self.config.api_key_header) or
            request.query.get(self.config.api_key_query_param)
        )
        
        if not api_key:
            return AuthResult(
                success=False,
                error_message="缺少API Key"
            )
        
        # 查找API Key
        for key_id, api_key_obj in self.api_keys.items():
            if api_key_obj.verify_key(api_key):
                # 检查API Key状态
                if not api_key_obj.is_active:
                    return AuthResult(
                        success=False,
                        error_message="API Key已禁用"
                    )
                
                if api_key_obj.is_expired():
                    return AuthResult(
                        success=False,
                        error_message="API Key已过期"
                    )
                
                # 获取用户信息
                user = self.users.get(api_key_obj.user_id)
                if not user:
                    return AuthResult(
                        success=False,
                        error_message="关联用户不存在"
                    )
                
                return AuthResult(
                    success=True,
                    user=user,
                    api_key=api_key_obj,
                    permissions=api_key_obj.permissions or user.permissions
                )
        
        return AuthResult(
            success=False,
            error_message="无效的API Key"
        )
    
    async def _authenticate_oauth2(self, request: web.Request) -> AuthResult:
        """OAuth2认证
        
        Args:
            request: HTTP请求
            
        Returns:
            认证结果
        """
        # OAuth2认证实现（需要根据具体的OAuth2提供商实现）
        return AuthResult(
            success=False,
            error_message="OAuth2认证未实现"
        )
    
    async def _authenticate_custom(self, request: web.Request) -> AuthResult:
        """自定义认证
        
        Args:
            request: HTTP请求
            
        Returns:
            认证结果
        """
        # 尝试所有自定义认证器
        for name, authenticator in self.custom_authenticators.items():
            try:
                result = await authenticator(request)
                if result and result.success:
                    return result
            except Exception as e:
                self.logger.error(f"自定义认证器 {name} 执行失败: {e}")
        
        return AuthResult(
            success=False,
            error_message="自定义认证失败"
        )
    
    async def _verify_user_credentials(self, username: str, password: str) -> Optional[User]:
        """验证用户凭据
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            用户对象，如果验证失败返回None
        """
        # 这里需要实现具体的用户验证逻辑
        # 例如：查询数据库、LDAP等
        
        # 示例实现（实际应用中需要替换）
        for user in self.users.values():
            if user.username == username and user.is_active:
                # 这里应该验证密码哈希
                # 示例中直接返回用户
                return user
        
        return None
    
    async def _record_failed_attempt(self, client_ip: str) -> None:
        """记录失败尝试
        
        Args:
            client_ip: 客户端IP
        """
        current_time = time.time()
        
        if client_ip not in self.login_attempts:
            self.login_attempts[client_ip] = []
        
        # 添加当前尝试
        self.login_attempts[client_ip].append(current_time)
        
        # 清理过期的尝试记录
        cutoff_time = current_time - self.config.lockout_duration
        self.login_attempts[client_ip] = [
            attempt_time for attempt_time in self.login_attempts[client_ip]
            if attempt_time > cutoff_time
        ]
        
        # 检查是否需要锁定
        if len(self.login_attempts[client_ip]) >= self.config.max_login_attempts:
            self.logger.warning(f"IP {client_ip} 登录失败次数过多，已锁定")
    
    def is_ip_locked(self, client_ip: str) -> bool:
        """检查IP是否被锁定
        
        Args:
            client_ip: 客户端IP
            
        Returns:
            是否被锁定
        """
        if client_ip not in self.login_attempts:
            return False
        
        current_time = time.time()
        cutoff_time = current_time - self.config.lockout_duration
        
        # 清理过期的尝试记录
        self.login_attempts[client_ip] = [
            attempt_time for attempt_time in self.login_attempts[client_ip]
            if attempt_time > cutoff_time
        ]
        
        return len(self.login_attempts[client_ip]) >= self.config.max_login_attempts
    
    def generate_jwt_token(self, user: User, expires_in: Optional[int] = None) -> str:
        """生成JWT Token
        
        Args:
            user: 用户对象
            expires_in: 过期时间（秒），如果为None则使用默认值
            
        Returns:
            JWT Token
        """
        if expires_in is None:
            expires_in = self.config.jwt_expiration
        
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': user.roles,
            'permissions': [p.value for p in user.permissions],
            'iat': int(time.time()),
            'exp': int(time.time()) + expires_in
        }
        
        token = jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm
        )
        
        self.stats['jwt_issued'] += 1
        return token
    
    def create_api_key(self, user_id: str, name: str, permissions: List[Permission] = None, expires_in: Optional[int] = None) -> tuple[str, ApiKey]:
        """创建API Key
        
        Args:
            user_id: 用户ID
            name: API Key名称
            permissions: 权限列表
            expires_in: 过期时间（秒）
            
        Returns:
            (原始密钥, API Key对象)
        """
        import secrets
        
        # 生成随机密钥
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # 创建API Key对象
        key_id = f"ak_{int(time.time() * 1000000)}"
        expires_at = time.time() + expires_in if expires_in else None
        
        api_key = ApiKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions or [],
            expires_at=expires_at
        )
        
        # 存储API Key
        self.api_keys[key_id] = api_key
        
        self.logger.info(f"为用户 {user_id} 创建API Key: {name}")
        return raw_key, api_key
    
    def revoke_api_key(self, key_id: str) -> bool:
        """撤销API Key
        
        Args:
            key_id: API Key ID
            
        Returns:
            是否成功撤销
        """
        if key_id in self.api_keys:
            self.api_keys[key_id].is_active = False
            self.logger.info(f"撤销API Key: {key_id}")
            return True
        return False
    
    def add_user(self, user: User) -> None:
        """添加用户
        
        Args:
            user: 用户对象
        """
        self.users[user.user_id] = user
        self.logger.info(f"添加用户: {user.username} ({user.user_id})")
    
    def remove_user(self, user_id: str) -> bool:
        """移除用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否成功移除
        """
        if user_id in self.users:
            del self.users[user_id]
            
            # 同时移除相关的API Key
            keys_to_remove = [key_id for key_id, api_key in self.api_keys.items() if api_key.user_id == user_id]
            for key_id in keys_to_remove:
                del self.api_keys[key_id]
            
            self.logger.info(f"移除用户: {user_id}")
            return True
        return False
    
    def add_custom_authenticator(self, name: str, authenticator: Callable) -> None:
        """添加自定义认证器
        
        Args:
            name: 认证器名称
            authenticator: 认证器函数
        """
        self.custom_authenticators[name] = authenticator
        self.logger.info(f"添加自定义认证器: {name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息
        """
        return {
            'auth_stats': self.stats,
            'users_count': len(self.users),
            'api_keys_count': len(self.api_keys),
            'active_sessions': self.stats['active_sessions'],
            'locked_ips': len([ip for ip, attempts in self.login_attempts.items() if len(attempts) >= self.config.max_login_attempts])
        }