"""Gateway Module

API网关模块，提供统一的接口访问和管理功能。

主要功能：
- API路由和转发
- 请求认证和授权
- 限流和熔断
- 监控和日志
- 协议转换
- 负载均衡

核心组件：
- APIGateway: 主网关服务
- Router: 路由管理器
- AuthManager: 认证管理器
- RateLimiter: 限流器
- CircuitBreaker: 熔断器
- LoadBalancer: 负载均衡器
- ProtocolConverter: 协议转换器

支持的协议：
- HTTP/HTTPS
- WebSocket
- gRPC
- TCP/UDP

支持的认证方式：
- API Key
- JWT Token
- OAuth 2.0
- Basic Auth

支持的限流策略：
- 令牌桶
- 漏桶
- 滑动窗口
- 固定窗口
"""

from .gateway import APIGateway
from .router import Router, Route
from .auth import AuthManager, AuthConfig
from .limiter import RateLimiter, LimitConfig
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .load_balancer import LoadBalancer, LoadBalancerConfig
from .converter import ProtocolConverter
from .middleware import Middleware, MiddlewareChain

__all__ = [
    'APIGateway',
    'Router',
    'Route',
    'AuthManager',
    'AuthConfig',
    'RateLimiter',
    'LimitConfig',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'LoadBalancer',
    'LoadBalancerConfig',
    'ProtocolConverter',
    'Middleware',
    'MiddlewareChain'
]

__version__ = '1.0.0'
__author__ = 'Prometheus Team'
__description__ = 'API Gateway for Prometheus Trading System'