"""API Gateway Module

API网关主服务模块，提供统一的接口访问和管理功能。

主要功能：
- 请求路由和转发
- 认证和授权
- 限流和熔断
- 监控和日志
- 协议转换
- 负载均衡
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
from aiohttp import web, WSMsgType
import ssl

from .router import Router, Route
from .auth import AuthManager, AuthConfig
from .limiter import RateLimiter, LimitConfig
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .load_balancer import LoadBalancer, LoadBalancerConfig
from .converter import ProtocolConverter
from .middleware import MiddlewareChain
from ..common.logging.logger import get_logger
from ..common.exceptions.gateway_exceptions import GatewayError, AuthenticationError, RateLimitError


class GatewayStatus(Enum):
    """网关状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class GatewayConfig:
    """网关配置"""
    # 基本配置
    host: str = "0.0.0.0"
    port: int = 8080
    ssl_context: Optional[ssl.SSLContext] = None
    
    # 性能配置
    max_connections: int = 1000
    connection_timeout: float = 30.0
    request_timeout: float = 60.0
    keepalive_timeout: float = 75.0
    
    # 安全配置
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # 监控配置
    enable_metrics: bool = True
    metrics_path: str = "/metrics"
    health_check_path: str = "/health"
    
    # 日志配置
    enable_access_log: bool = True
    log_level: str = "INFO"
    
    # 中间件配置
    enable_compression: bool = True
    compression_level: int = 6
    
    # WebSocket配置
    enable_websocket: bool = True
    websocket_timeout: float = 300.0
    websocket_heartbeat: float = 30.0


@dataclass
class RequestContext:
    """请求上下文"""
    request_id: str
    start_time: float
    client_ip: str
    user_agent: str
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, str]
    auth_info: Optional[Dict[str, Any]] = None
    route_info: Optional[Route] = None
    upstream_url: Optional[str] = None
    response_time: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None


class APIGateway:
    """API网关
    
    提供统一的接口访问和管理功能。
    """
    
    def __init__(self, config: GatewayConfig):
        """初始化API网关
        
        Args:
            config: 网关配置
        """
        self.config = config
        self.logger = get_logger("api_gateway")
        
        # 状态管理
        self.status = GatewayStatus.STOPPED
        self.start_time: Optional[datetime] = None
        
        # 核心组件
        self.router = Router()
        self.auth_manager: Optional[AuthManager] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.circuit_breaker: Optional[CircuitBreaker] = None
        self.load_balancer: Optional[LoadBalancer] = None
        self.protocol_converter = ProtocolConverter()
        self.middleware_chain = MiddlewareChain()
        
        # Web应用
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        
        # 连接管理
        self.active_connections: Dict[str, Any] = {}
        self.websocket_connections: Dict[str, web.WebSocketResponse] = {}
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'avg_response_time': 0.0,
            'active_connections': 0,
            'websocket_connections': 0
        }
        
        self.logger.info(f"API网关初始化完成，监听地址: {config.host}:{config.port}")
    
    def set_auth_manager(self, auth_manager: AuthManager) -> None:
        """设置认证管理器
        
        Args:
            auth_manager: 认证管理器
        """
        self.auth_manager = auth_manager
        self.logger.info("认证管理器已设置")
    
    def set_rate_limiter(self, rate_limiter: RateLimiter) -> None:
        """设置限流器
        
        Args:
            rate_limiter: 限流器
        """
        self.rate_limiter = rate_limiter
        self.logger.info("限流器已设置")
    
    def set_circuit_breaker(self, circuit_breaker: CircuitBreaker) -> None:
        """设置熔断器
        
        Args:
            circuit_breaker: 熔断器
        """
        self.circuit_breaker = circuit_breaker
        self.logger.info("熔断器已设置")
    
    def set_load_balancer(self, load_balancer: LoadBalancer) -> None:
        """设置负载均衡器
        
        Args:
            load_balancer: 负载均衡器
        """
        self.load_balancer = load_balancer
        self.logger.info("负载均衡器已设置")
    
    def add_route(self, route: Route) -> None:
        """添加路由
        
        Args:
            route: 路由配置
        """
        self.router.add_route(route)
        self.logger.info(f"添加路由: {route.method} {route.path} -> {route.upstream}")
    
    def add_middleware(self, middleware: Callable) -> None:
        """添加中间件
        
        Args:
            middleware: 中间件函数
        """
        self.middleware_chain.add_middleware(middleware)
        self.logger.info("添加中间件")
    
    async def start(self) -> None:
        """启动网关"""
        if self.status != GatewayStatus.STOPPED:
            self.logger.warning(f"网关当前状态: {self.status.value}，无法启动")
            return
        
        try:
            self.status = GatewayStatus.STARTING
            self.logger.info("启动API网关")
            
            # 创建Web应用
            self.app = web.Application(
                middlewares=self._create_middlewares(),
                client_max_size=1024**2 * 10  # 10MB
            )
            
            # 设置路由
            self._setup_routes()
            
            # 启动应用
            self.runner = web.AppRunner(
                self.app,
                access_log=self.logger if self.config.enable_access_log else None,
                keepalive_timeout=self.config.keepalive_timeout
            )
            
            await self.runner.setup()
            
            self.site = web.TCPSite(
                self.runner,
                self.config.host,
                self.config.port,
                ssl_context=self.config.ssl_context
            )
            
            await self.site.start()
            
            self.status = GatewayStatus.RUNNING
            self.start_time = datetime.now()
            
            self.logger.info(f"API网关启动成功，监听地址: {self.config.host}:{self.config.port}")
            
        except Exception as e:
            self.status = GatewayStatus.ERROR
            self.logger.error(f"启动API网关失败: {e}")
            raise GatewayError(f"启动网关失败: {e}")
    
    async def stop(self) -> None:
        """停止网关"""
        if self.status != GatewayStatus.RUNNING:
            self.logger.warning(f"网关当前状态: {self.status.value}，无法停止")
            return
        
        try:
            self.status = GatewayStatus.STOPPING
            self.logger.info("停止API网关")
            
            # 关闭WebSocket连接
            for ws in self.websocket_connections.values():
                if not ws.closed:
                    await ws.close()
            self.websocket_connections.clear()
            
            # 停止站点
            if self.site:
                await self.site.stop()
            
            # 清理运行器
            if self.runner:
                await self.runner.cleanup()
            
            self.status = GatewayStatus.STOPPED
            self.logger.info("API网关已停止")
            
        except Exception as e:
            self.status = GatewayStatus.ERROR
            self.logger.error(f"停止API网关失败: {e}")
            raise GatewayError(f"停止网关失败: {e}")
    
    def _create_middlewares(self) -> List[Callable]:
        """创建中间件列表"""
        middlewares = []
        
        # CORS中间件
        if self.config.enable_cors:
            middlewares.append(self._cors_middleware)
        
        # 请求上下文中间件
        middlewares.append(self._context_middleware)
        
        # 认证中间件
        if self.auth_manager:
            middlewares.append(self._auth_middleware)
        
        # 限流中间件
        if self.rate_limiter:
            middlewares.append(self._rate_limit_middleware)
        
        # 熔断中间件
        if self.circuit_breaker:
            middlewares.append(self._circuit_breaker_middleware)
        
        # 压缩中间件
        if self.config.enable_compression:
            middlewares.append(self._compression_middleware)
        
        # 统计中间件
        middlewares.append(self._stats_middleware)
        
        # 错误处理中间件
        middlewares.append(self._error_middleware)
        
        return middlewares
    
    def _setup_routes(self) -> None:
        """设置路由"""
        # 健康检查
        self.app.router.add_get(self.config.health_check_path, self._health_check_handler)
        
        # 指标接口
        if self.config.enable_metrics:
            self.app.router.add_get(self.config.metrics_path, self._metrics_handler)
        
        # WebSocket处理
        if self.config.enable_websocket:
            self.app.router.add_get('/ws', self._websocket_handler)
        
        # 通用路由处理
        self.app.router.add_route('*', '/{path:.*}', self._route_handler)
    
    @web.middleware
    async def _cors_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """CORS中间件"""
        # 处理预检请求
        if request.method == 'OPTIONS':
            response = web.Response()
        else:
            response = await handler(request)
        
        # 设置CORS头
        response.headers['Access-Control-Allow-Origin'] = ', '.join(self.config.cors_origins)
        response.headers['Access-Control-Allow-Methods'] = ', '.join(self.config.cors_methods)
        response.headers['Access-Control-Allow-Headers'] = ', '.join(self.config.cors_headers)
        response.headers['Access-Control-Max-Age'] = '86400'
        
        return response
    
    @web.middleware
    async def _context_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """请求上下文中间件"""
        # 创建请求上下文
        context = RequestContext(
            request_id=f"{int(time.time() * 1000000)}",
            start_time=time.time(),
            client_ip=request.remote or 'unknown',
            user_agent=request.headers.get('User-Agent', 'unknown'),
            method=request.method,
            path=request.path,
            headers=dict(request.headers),
            query_params=dict(request.query)
        )
        
        # 将上下文附加到请求
        request['context'] = context
        
        try:
            response = await handler(request)
            context.status_code = response.status
            return response
        except Exception as e:
            context.error = str(e)
            raise
        finally:
            context.response_time = time.time() - context.start_time
    
    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """认证中间件"""
        # 跳过健康检查和指标接口
        if request.path in [self.config.health_check_path, self.config.metrics_path]:
            return await handler(request)
        
        try:
            # 执行认证
            auth_info = await self.auth_manager.authenticate(request)
            request['context'].auth_info = auth_info
            
            return await handler(request)
            
        except AuthenticationError as e:
            return web.json_response(
                {'error': 'Authentication failed', 'message': str(e)},
                status=401
            )
    
    @web.middleware
    async def _rate_limit_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """限流中间件"""
        # 跳过健康检查和指标接口
        if request.path in [self.config.health_check_path, self.config.metrics_path]:
            return await handler(request)
        
        try:
            # 检查限流
            client_id = request['context'].client_ip
            if request['context'].auth_info:
                client_id = request['context'].auth_info.get('user_id', client_id)
            
            await self.rate_limiter.check_limit(client_id)
            
            return await handler(request)
            
        except RateLimitError as e:
            return web.json_response(
                {'error': 'Rate limit exceeded', 'message': str(e)},
                status=429
            )
    
    @web.middleware
    async def _circuit_breaker_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """熔断中间件"""
        # 跳过健康检查和指标接口
        if request.path in [self.config.health_check_path, self.config.metrics_path]:
            return await handler(request)
        
        try:
            # 检查熔断状态
            service_name = request.path.split('/')[1] if len(request.path.split('/')) > 1 else 'default'
            
            if not await self.circuit_breaker.can_execute(service_name):
                return web.json_response(
                    {'error': 'Service unavailable', 'message': 'Circuit breaker is open'},
                    status=503
                )
            
            try:
                response = await handler(request)
                await self.circuit_breaker.record_success(service_name)
                return response
            except Exception as e:
                await self.circuit_breaker.record_failure(service_name)
                raise
            
        except Exception as e:
            return web.json_response(
                {'error': 'Internal server error', 'message': str(e)},
                status=500
            )
    
    @web.middleware
    async def _compression_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """压缩中间件"""
        response = await handler(request)
        
        # 检查是否支持压缩
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if 'gzip' in accept_encoding and len(response.body) > 1024:
            # 这里可以添加gzip压缩逻辑
            pass
        
        return response
    
    @web.middleware
    async def _stats_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """统计中间件"""
        self.stats['total_requests'] += 1
        
        try:
            response = await handler(request)
            
            if 200 <= response.status < 400:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
            
            # 更新响应时间
            response_time = request['context'].response_time
            if response_time:
                self.stats['avg_response_time'] = (
                    (self.stats['avg_response_time'] * 0.9) + (response_time * 0.1)
                )
            
            return response
            
        except Exception:
            self.stats['failed_requests'] += 1
            raise
    
    @web.middleware
    async def _error_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """错误处理中间件"""
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"请求处理错误: {e}", exc_info=True)
            return web.json_response(
                {'error': 'Internal server error', 'message': 'An unexpected error occurred'},
                status=500
            )
    
    async def _health_check_handler(self, request: web.Request) -> web.Response:
        """健康检查处理器"""
        health_info = {
            'status': self.status.value,
            'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
        
        return web.json_response(health_info)
    
    async def _metrics_handler(self, request: web.Request) -> web.Response:
        """指标处理器"""
        metrics = {
            'gateway_stats': self.stats,
            'router_stats': self.router.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        if self.auth_manager:
            metrics['auth_stats'] = self.auth_manager.get_stats()
        
        if self.rate_limiter:
            metrics['rate_limit_stats'] = self.rate_limiter.get_stats()
        
        if self.circuit_breaker:
            metrics['circuit_breaker_stats'] = self.circuit_breaker.get_stats()
        
        return web.json_response(metrics)
    
    async def _websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket处理器"""
        ws = web.WebSocketResponse(
            timeout=self.config.websocket_timeout,
            heartbeat=self.config.websocket_heartbeat
        )
        
        await ws.prepare(request)
        
        # 添加到连接池
        connection_id = request['context'].request_id
        self.websocket_connections[connection_id] = ws
        self.stats['websocket_connections'] = len(self.websocket_connections)
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        # 处理WebSocket消息
                        response = await self._handle_websocket_message(data)
                        await ws.send_text(json.dumps(response))
                    except json.JSONDecodeError:
                        await ws.send_text(json.dumps({'error': 'Invalid JSON'}))
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f'WebSocket错误: {ws.exception()}')
                    break
        
        except Exception as e:
            self.logger.error(f"WebSocket处理错误: {e}")
        
        finally:
            # 从连接池移除
            if connection_id in self.websocket_connections:
                del self.websocket_connections[connection_id]
            self.stats['websocket_connections'] = len(self.websocket_connections)
        
        return ws
    
    async def _handle_websocket_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理WebSocket消息
        
        Args:
            data: 消息数据
            
        Returns:
            响应数据
        """
        message_type = data.get('type')
        
        if message_type == 'ping':
            return {'type': 'pong', 'timestamp': datetime.now().isoformat()}
        elif message_type == 'subscribe':
            # 处理订阅请求
            return {'type': 'subscribed', 'channel': data.get('channel')}
        elif message_type == 'unsubscribe':
            # 处理取消订阅请求
            return {'type': 'unsubscribed', 'channel': data.get('channel')}
        else:
            return {'type': 'error', 'message': f'Unknown message type: {message_type}'}
    
    async def _route_handler(self, request: web.Request) -> web.Response:
        """路由处理器"""
        context = request['context']
        
        # 查找路由
        route = self.router.match_route(request.method, request.path)
        if not route:
            return web.json_response(
                {'error': 'Not found', 'message': f'Route not found: {request.method} {request.path}'},
                status=404
            )
        
        context.route_info = route
        
        try:
            # 选择上游服务
            if self.load_balancer:
                upstream_url = await self.load_balancer.select_upstream(route.upstream)
            else:
                upstream_url = route.upstream[0] if isinstance(route.upstream, list) else route.upstream
            
            context.upstream_url = upstream_url
            
            # 转发请求
            response = await self._forward_request(request, upstream_url, route)
            
            return response
            
        except Exception as e:
            self.logger.error(f"路由处理错误: {e}")
            return web.json_response(
                {'error': 'Gateway error', 'message': str(e)},
                status=502
            )
    
    async def _forward_request(self, request: web.Request, upstream_url: str, route: Route) -> web.Response:
        """转发请求
        
        Args:
            request: 原始请求
            upstream_url: 上游URL
            route: 路由配置
            
        Returns:
            响应
        """
        # 构建目标URL
        target_url = upstream_url.rstrip('/') + request.path_qs
        
        # 准备请求头
        headers = dict(request.headers)
        headers.pop('Host', None)  # 移除Host头
        
        # 添加代理头
        headers['X-Forwarded-For'] = request['context'].client_ip
        headers['X-Forwarded-Proto'] = 'https' if request.scheme == 'https' else 'http'
        headers['X-Request-ID'] = request['context'].request_id
        
        # 读取请求体
        body = None
        if request.method in ['POST', 'PUT', 'PATCH']:
            body = await request.read()
        
        # 发送请求
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body
            ) as response:
                # 读取响应
                response_body = await response.read()
                
                # 创建响应
                web_response = web.Response(
                    body=response_body,
                    status=response.status,
                    headers=response.headers
                )
                
                return web_response
    
    def get_stats(self) -> Dict[str, Any]:
        """获取网关统计信息"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'status': self.status.value,
            'uptime_seconds': uptime,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'config': {
                'host': self.config.host,
                'port': self.config.port,
                'max_connections': self.config.max_connections,
                'enable_cors': self.config.enable_cors,
                'enable_metrics': self.config.enable_metrics,
                'enable_websocket': self.config.enable_websocket
            },
            'stats': self.stats,
            'active_connections': len(self.active_connections),
            'websocket_connections': len(self.websocket_connections)
        }
    
    async def broadcast_websocket_message(self, message: Dict[str, Any]) -> int:
        """广播WebSocket消息
        
        Args:
            message: 消息内容
            
        Returns:
            发送成功的连接数
        """
        if not self.websocket_connections:
            return 0
        
        message_text = json.dumps(message)
        success_count = 0
        
        # 复制连接列表以避免并发修改
        connections = list(self.websocket_connections.values())
        
        for ws in connections:
            try:
                if not ws.closed:
                    await ws.send_text(message_text)
                    success_count += 1
            except Exception as e:
                self.logger.warning(f"WebSocket消息发送失败: {e}")
        
        return success_count
    
    def __del__(self):
        """析构函数"""
        if self.status == GatewayStatus.RUNNING:
            asyncio.create_task(self.stop())