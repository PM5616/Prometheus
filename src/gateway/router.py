"""Router Module

路由管理模块，负责请求路由和转发。

主要功能：
- 路由规则管理
- 路径匹配
- 参数提取
- 路由统计
"""

import re
import time
from typing import Dict, List, Optional, Any, Pattern, Union
from dataclasses import dataclass, field
from enum import Enum
import fnmatch

from ..common.logging.logger import get_logger


class RouteType(Enum):
    """路由类型"""
    EXACT = "exact"          # 精确匹配
    PREFIX = "prefix"        # 前缀匹配
    REGEX = "regex"          # 正则匹配
    WILDCARD = "wildcard"    # 通配符匹配


class LoadBalanceMethod(Enum):
    """负载均衡方法"""
    ROUND_ROBIN = "round_robin"    # 轮询
    WEIGHTED = "weighted"          # 加权
    LEAST_CONN = "least_conn"      # 最少连接
    IP_HASH = "ip_hash"            # IP哈希
    RANDOM = "random"              # 随机


@dataclass
class RouteConfig:
    """路由配置"""
    # 超时配置
    timeout: float = 30.0
    connect_timeout: float = 5.0
    read_timeout: float = 25.0
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    # 健康检查
    health_check_path: str = "/health"
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    
    # 负载均衡
    load_balance_method: LoadBalanceMethod = LoadBalanceMethod.ROUND_ROBIN
    
    # 缓存配置
    enable_cache: bool = False
    cache_ttl: int = 300
    cache_key_headers: List[str] = field(default_factory=list)
    
    # 请求转换
    strip_path_prefix: bool = False
    add_path_prefix: str = ""
    
    # 请求头处理
    add_headers: Dict[str, str] = field(default_factory=dict)
    remove_headers: List[str] = field(default_factory=list)
    
    # 响应处理
    response_headers: Dict[str, str] = field(default_factory=dict)
    
    # 安全配置
    require_auth: bool = True
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    rate_limit: Optional[int] = None


@dataclass
class Route:
    """路由定义"""
    # 基本信息
    name: str
    method: str  # HTTP方法，支持 * 表示所有方法
    path: str    # 路径模式
    upstream: Union[str, List[str]]  # 上游服务地址
    
    # 路由类型和配置
    route_type: RouteType = RouteType.PREFIX
    config: RouteConfig = field(default_factory=RouteConfig)
    
    # 权重和优先级
    weight: int = 100
    priority: int = 0  # 数字越大优先级越高
    
    # 状态
    enabled: bool = True
    
    # 元数据
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    # 编译后的模式（内部使用）
    _compiled_pattern: Optional[Pattern] = field(default=None, init=False, repr=False)
    _path_params: List[str] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        """初始化后处理"""
        self._compile_pattern()
    
    def _compile_pattern(self) -> None:
        """编译路径模式"""
        if self.route_type == RouteType.EXACT:
            # 精确匹配
            self._compiled_pattern = re.compile(f"^{re.escape(self.path)}$")
        
        elif self.route_type == RouteType.PREFIX:
            # 前缀匹配
            escaped_path = re.escape(self.path.rstrip('/'))
            self._compiled_pattern = re.compile(f"^{escaped_path}(/.*)?$")
        
        elif self.route_type == RouteType.REGEX:
            # 正则匹配
            self._compiled_pattern = re.compile(self.path)
            # 提取命名组作为路径参数
            self._path_params = list(self._compiled_pattern.groupindex.keys())
        
        elif self.route_type == RouteType.WILDCARD:
            # 通配符匹配，转换为正则表达式
            pattern = self.path.replace('*', '([^/]*)')
            pattern = pattern.replace('**', '(.*)')
            self._compiled_pattern = re.compile(f"^{pattern}$")
    
    def match(self, path: str) -> Optional[Dict[str, str]]:
        """匹配路径
        
        Args:
            path: 请求路径
            
        Returns:
            匹配结果，包含路径参数，如果不匹配返回None
        """
        if not self._compiled_pattern:
            return None
        
        match = self._compiled_pattern.match(path)
        if not match:
            return None
        
        # 提取路径参数
        params = {}
        
        if self.route_type == RouteType.REGEX:
            # 正则匹配的命名组
            params.update(match.groupdict())
        
        elif self.route_type == RouteType.WILDCARD:
            # 通配符匹配的位置参数
            groups = match.groups()
            for i, value in enumerate(groups):
                params[f"param{i}"] = value
        
        return params
    
    def is_method_allowed(self, method: str) -> bool:
        """检查HTTP方法是否允许
        
        Args:
            method: HTTP方法
            
        Returns:
            是否允许
        """
        if self.method == "*":
            return method in self.config.allowed_methods
        
        return self.method.upper() == method.upper()


@dataclass
class RouteStats:
    """路由统计信息"""
    route_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_request_time: Optional[float] = None
    error_rate: float = 0.0
    
    def update_request(self, success: bool, response_time: float) -> None:
        """更新请求统计
        
        Args:
            success: 是否成功
            response_time: 响应时间
        """
        self.total_requests += 1
        self.last_request_time = time.time()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # 更新平均响应时间（指数移动平均）
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * 0.9) + (response_time * 0.1)
        
        # 更新错误率
        self.error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0.0


class Router:
    """路由器
    
    负责管理路由规则和请求匹配。
    """
    
    def __init__(self):
        """初始化路由器"""
        self.logger = get_logger("router")
        
        # 路由存储
        self.routes: List[Route] = []
        self.route_map: Dict[str, Route] = {}  # name -> route
        
        # 统计信息
        self.stats: Dict[str, RouteStats] = {}
        self.global_stats = {
            'total_routes': 0,
            'enabled_routes': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
        
        self.logger.info("路由器初始化完成")
    
    def add_route(self, route: Route) -> None:
        """添加路由
        
        Args:
            route: 路由配置
        """
        # 检查路由名称是否重复
        if route.name in self.route_map:
            raise ValueError(f"路由名称已存在: {route.name}")
        
        # 添加路由
        self.routes.append(route)
        self.route_map[route.name] = route
        
        # 按优先级排序
        self.routes.sort(key=lambda r: r.priority, reverse=True)
        
        # 初始化统计信息
        self.stats[route.name] = RouteStats(route.name)
        
        # 更新全局统计
        self.global_stats['total_routes'] = len(self.routes)
        self.global_stats['enabled_routes'] = len([r for r in self.routes if r.enabled])
        
        self.logger.info(f"添加路由: {route.name} ({route.method} {route.path})")
    
    def remove_route(self, name: str) -> bool:
        """移除路由
        
        Args:
            name: 路由名称
            
        Returns:
            是否成功移除
        """
        if name not in self.route_map:
            return False
        
        route = self.route_map[name]
        
        # 从列表中移除
        self.routes.remove(route)
        del self.route_map[name]
        
        # 移除统计信息
        if name in self.stats:
            del self.stats[name]
        
        # 更新全局统计
        self.global_stats['total_routes'] = len(self.routes)
        self.global_stats['enabled_routes'] = len([r for r in self.routes if r.enabled])
        
        self.logger.info(f"移除路由: {name}")
        return True
    
    def update_route(self, name: str, route: Route) -> bool:
        """更新路由
        
        Args:
            name: 原路由名称
            route: 新路由配置
            
        Returns:
            是否成功更新
        """
        if name not in self.route_map:
            return False
        
        # 如果名称改变，需要检查新名称是否重复
        if route.name != name and route.name in self.route_map:
            raise ValueError(f"路由名称已存在: {route.name}")
        
        old_route = self.route_map[name]
        
        # 更新路由
        index = self.routes.index(old_route)
        self.routes[index] = route
        
        # 更新映射
        del self.route_map[name]
        self.route_map[route.name] = route
        
        # 重新排序
        self.routes.sort(key=lambda r: r.priority, reverse=True)
        
        # 更新统计信息
        if name != route.name:
            if name in self.stats:
                self.stats[route.name] = self.stats[name]
                self.stats[route.name].route_name = route.name
                del self.stats[name]
        
        self.logger.info(f"更新路由: {name} -> {route.name}")
        return True
    
    def enable_route(self, name: str) -> bool:
        """启用路由
        
        Args:
            name: 路由名称
            
        Returns:
            是否成功启用
        """
        if name not in self.route_map:
            return False
        
        route = self.route_map[name]
        if not route.enabled:
            route.enabled = True
            self.global_stats['enabled_routes'] = len([r for r in self.routes if r.enabled])
            self.logger.info(f"启用路由: {name}")
        
        return True
    
    def disable_route(self, name: str) -> bool:
        """禁用路由
        
        Args:
            name: 路由名称
            
        Returns:
            是否成功禁用
        """
        if name not in self.route_map:
            return False
        
        route = self.route_map[name]
        if route.enabled:
            route.enabled = False
            self.global_stats['enabled_routes'] = len([r for r in self.routes if r.enabled])
            self.logger.info(f"禁用路由: {name}")
        
        return True
    
    def match_route(self, method: str, path: str) -> Optional[Route]:
        """匹配路由
        
        Args:
            method: HTTP方法
            path: 请求路径
            
        Returns:
            匹配的路由，如果没有匹配返回None
        """
        for route in self.routes:
            # 检查路由是否启用
            if not route.enabled:
                continue
            
            # 检查HTTP方法
            if not route.is_method_allowed(method):
                continue
            
            # 检查路径匹配
            params = route.match(path)
            if params is not None:
                # 记录匹配统计
                self._record_route_match(route.name)
                return route
        
        return None
    
    def get_route(self, name: str) -> Optional[Route]:
        """获取路由
        
        Args:
            name: 路由名称
            
        Returns:
            路由配置
        """
        return self.route_map.get(name)
    
    def list_routes(self, enabled_only: bool = False) -> List[Route]:
        """列出所有路由
        
        Args:
            enabled_only: 是否只返回启用的路由
            
        Returns:
            路由列表
        """
        if enabled_only:
            return [route for route in self.routes if route.enabled]
        return self.routes.copy()
    
    def find_routes_by_tag(self, tag: str) -> List[Route]:
        """根据标签查找路由
        
        Args:
            tag: 标签
            
        Returns:
            匹配的路由列表
        """
        return [route for route in self.routes if tag in route.tags]
    
    def find_routes_by_upstream(self, upstream: str) -> List[Route]:
        """根据上游服务查找路由
        
        Args:
            upstream: 上游服务地址
            
        Returns:
            匹配的路由列表
        """
        result = []
        for route in self.routes:
            if isinstance(route.upstream, str):
                if route.upstream == upstream:
                    result.append(route)
            elif isinstance(route.upstream, list):
                if upstream in route.upstream:
                    result.append(route)
        return result
    
    def record_request(self, route_name: str, success: bool, response_time: float) -> None:
        """记录请求统计
        
        Args:
            route_name: 路由名称
            success: 是否成功
            response_time: 响应时间
        """
        # 更新路由统计
        if route_name in self.stats:
            self.stats[route_name].update_request(success, response_time)
        
        # 更新全局统计
        self.global_stats['total_requests'] += 1
        
        if success:
            self.global_stats['successful_requests'] += 1
        else:
            self.global_stats['failed_requests'] += 1
        
        # 更新全局平均响应时间
        if self.global_stats['avg_response_time'] == 0:
            self.global_stats['avg_response_time'] = response_time
        else:
            self.global_stats['avg_response_time'] = (
                (self.global_stats['avg_response_time'] * 0.9) + (response_time * 0.1)
            )
    
    def _record_route_match(self, route_name: str) -> None:
        """记录路由匹配
        
        Args:
            route_name: 路由名称
        """
        # 这里可以添加路由匹配的统计逻辑
        pass
    
    def get_route_stats(self, name: str) -> Optional[RouteStats]:
        """获取路由统计信息
        
        Args:
            name: 路由名称
            
        Returns:
            统计信息
        """
        return self.stats.get(name)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取路由器统计信息
        
        Returns:
            统计信息
        """
        return {
            'global_stats': self.global_stats,
            'route_stats': {name: {
                'total_requests': stats.total_requests,
                'successful_requests': stats.successful_requests,
                'failed_requests': stats.failed_requests,
                'error_rate': stats.error_rate,
                'avg_response_time': stats.avg_response_time,
                'last_request_time': stats.last_request_time
            } for name, stats in self.stats.items()}
        }
    
    def clear_stats(self) -> None:
        """清除统计信息"""
        for stats in self.stats.values():
            stats.total_requests = 0
            stats.successful_requests = 0
            stats.failed_requests = 0
            stats.avg_response_time = 0.0
            stats.last_request_time = None
            stats.error_rate = 0.0
        
        self.global_stats.update({
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        })
        
        self.logger.info("清除路由统计信息")
    
    def export_routes(self) -> List[Dict[str, Any]]:
        """导出路由配置
        
        Returns:
            路由配置列表
        """
        return [
            {
                'name': route.name,
                'method': route.method,
                'path': route.path,
                'upstream': route.upstream,
                'route_type': route.route_type.value,
                'weight': route.weight,
                'priority': route.priority,
                'enabled': route.enabled,
                'description': route.description,
                'tags': route.tags,
                'created_at': route.created_at,
                'config': {
                    'timeout': route.config.timeout,
                    'max_retries': route.config.max_retries,
                    'load_balance_method': route.config.load_balance_method.value,
                    'require_auth': route.config.require_auth,
                    'allowed_methods': route.config.allowed_methods,
                    'rate_limit': route.config.rate_limit
                }
            }
            for route in self.routes
        ]
    
    def import_routes(self, routes_data: List[Dict[str, Any]]) -> int:
        """导入路由配置
        
        Args:
            routes_data: 路由配置数据
            
        Returns:
            成功导入的路由数量
        """
        success_count = 0
        
        for route_data in routes_data:
            try:
                # 创建路由配置
                config_data = route_data.get('config', {})
                config = RouteConfig(
                    timeout=config_data.get('timeout', 30.0),
                    max_retries=config_data.get('max_retries', 3),
                    load_balance_method=LoadBalanceMethod(config_data.get('load_balance_method', 'round_robin')),
                    require_auth=config_data.get('require_auth', True),
                    allowed_methods=config_data.get('allowed_methods', ["GET", "POST", "PUT", "DELETE"]),
                    rate_limit=config_data.get('rate_limit')
                )
                
                # 创建路由
                route = Route(
                    name=route_data['name'],
                    method=route_data['method'],
                    path=route_data['path'],
                    upstream=route_data['upstream'],
                    route_type=RouteType(route_data.get('route_type', 'prefix')),
                    config=config,
                    weight=route_data.get('weight', 100),
                    priority=route_data.get('priority', 0),
                    enabled=route_data.get('enabled', True),
                    description=route_data.get('description', ''),
                    tags=route_data.get('tags', []),
                    created_at=route_data.get('created_at', time.time())
                )
                
                # 添加路由
                self.add_route(route)
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"导入路由失败: {route_data.get('name', 'unknown')}, 错误: {e}")
        
        self.logger.info(f"成功导入 {success_count}/{len(routes_data)} 个路由")
        return success_count