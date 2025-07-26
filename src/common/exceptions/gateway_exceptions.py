"""网关异常模块

定义网关相关的异常类。
"""

from .base import PrometheusException


class GatewayException(PrometheusException):
    """网关基础异常"""
    pass


class GatewayError(GatewayException):
    """网关错误
    
    通用网关错误类，用于向后兼容。
    """
    pass


class AuthenticationError(GatewayException):
    """认证错误
    
    当用户认证失败时抛出此异常。
    """
    
    def __init__(self, message: str = "Authentication failed", auth_type: str = None, user_id: str = None):
        super().__init__(message)
        self.auth_type = auth_type
        self.user_id = user_id


class AuthorizationError(GatewayException):
    """授权错误
    
    当用户没有足够权限访问资源时抛出此异常。
    """
    
    def __init__(self, message: str = "Authorization failed", required_permission: str = None, user_id: str = None):
        super().__init__(message)
        self.required_permission = required_permission
        self.user_id = user_id


class RateLimitError(GatewayException):
    """速率限制错误
    
    当请求超过速率限制时抛出此异常。
    """
    
    def __init__(self, message: str = "Rate limit exceeded", limit: int = None, window: int = None):
        super().__init__(message)
        self.limit = limit
        self.window = window


class CircuitBreakerError(GatewayException):
    """熔断器错误
    
    当熔断器开启时抛出此异常。
    """
    
    def __init__(self, message: str = "Circuit breaker is open", service_name: str = None):
        super().__init__(message)
        self.service_name = service_name


class LoadBalancerError(GatewayException):
    """负载均衡器错误
    
    当负载均衡器无法找到可用服务时抛出此异常。
    """
    
    def __init__(self, message: str = "No available services", service_name: str = None):
        super().__init__(message)
        self.service_name = service_name


class ProtocolError(GatewayException):
    """协议错误
    
    当协议转换失败时抛出此异常。
    """
    
    def __init__(self, message: str = "Protocol conversion failed", source_protocol: str = None, target_protocol: str = None):
        super().__init__(message)
        self.source_protocol = source_protocol
        self.target_protocol = target_protocol


class MiddlewareError(GatewayException):
    """中间件错误
    
    当中间件处理失败时抛出此异常。
    """
    
    def __init__(self, message: str = "Middleware processing failed", middleware_name: str = None):
        super().__init__(message)
        self.middleware_name = middleware_name


class GatewayTimeoutError(GatewayException):
    """网关超时错误
    
    当网关操作超时时抛出此异常。
    """
    
    def __init__(self, message: str = "Gateway operation timeout", timeout: float = None, operation: str = None):
        super().__init__(message)
        self.timeout = timeout
        self.operation = operation


class ConfigurationError(GatewayException):
    """配置错误
    
    当网关配置无效时抛出此异常。
    """
    
    def __init__(self, message: str = "Invalid gateway configuration", config_key: str = None):
        super().__init__(message)
        self.config_key = config_key