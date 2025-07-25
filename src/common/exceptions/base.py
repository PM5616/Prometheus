"""Base Exception Classes

基础异常类定义。
"""

from typing import Optional, Dict, Any
from datetime import datetime


class PrometheusException(Exception):
    """Prometheus系统基础异常类"""
    
    def __init__(self, 
                 message: str, 
                 error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None):
        """初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 错误详情
            cause: 原始异常
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            Dict[str, Any]: 异常信息字典
        """
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'cause': str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """字符串表示
        
        Returns:
            str: 异常字符串
        """
        base_msg = f"[{self.error_code}] {self.message}"
        if self.details:
            base_msg += f" | Details: {self.details}"
        if self.cause:
            base_msg += f" | Caused by: {self.cause}"
        return base_msg


class ConfigurationError(PrometheusException):
    """配置错误异常"""
    
    def __init__(self, 
                 message: str = "Configuration error occurred",
                 config_key: Optional[str] = None,
                 **kwargs):
        """初始化配置错误
        
        Args:
            message: 错误消息
            config_key: 配置键名
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class ValidationError(PrometheusException):
    """数据验证错误异常"""
    
    def __init__(self, 
                 message: str = "Validation error occurred",
                 field_name: Optional[str] = None,
                 field_value: Optional[Any] = None,
                 **kwargs):
        """初始化验证错误
        
        Args:
            message: 错误消息
            field_name: 字段名
            field_value: 字段值
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if field_name:
            details['field_name'] = field_name
        if field_value is not None:
            details['field_value'] = str(field_value)
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class AuthenticationError(PrometheusException):
    """认证错误异常"""
    
    def __init__(self, 
                 message: str = "Authentication failed",
                 user_id: Optional[str] = None,
                 **kwargs):
        """初始化认证错误
        
        Args:
            message: 错误消息
            user_id: 用户ID
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if user_id:
            details['user_id'] = user_id
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class AuthorizationError(PrometheusException):
    """授权错误异常"""
    
    def __init__(self, 
                 message: str = "Authorization failed",
                 user_id: Optional[str] = None,
                 resource: Optional[str] = None,
                 action: Optional[str] = None,
                 **kwargs):
        """初始化授权错误
        
        Args:
            message: 错误消息
            user_id: 用户ID
            resource: 资源名
            action: 操作名
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if user_id:
            details['user_id'] = user_id
        if resource:
            details['resource'] = resource
        if action:
            details['action'] = action
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class NetworkError(PrometheusException):
    """网络错误异常"""
    
    def __init__(self, 
                 message: str = "Network error occurred",
                 url: Optional[str] = None,
                 status_code: Optional[int] = None,
                 **kwargs):
        """初始化网络错误
        
        Args:
            message: 错误消息
            url: 请求URL
            status_code: HTTP状态码
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if url:
            details['url'] = url
        if status_code:
            details['status_code'] = status_code
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class TimeoutError(PrometheusException):
    """超时错误异常"""
    
    def __init__(self, 
                 message: str = "Operation timed out",
                 timeout_seconds: Optional[float] = None,
                 operation: Optional[str] = None,
                 **kwargs):
        """初始化超时错误
        
        Args:
            message: 错误消息
            timeout_seconds: 超时时间（秒）
            operation: 操作名称
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        if operation:
            details['operation'] = operation
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class RateLimitError(PrometheusException):
    """频率限制错误异常"""
    
    def __init__(self, 
                 message: str = "Rate limit exceeded",
                 limit: Optional[int] = None,
                 window_seconds: Optional[int] = None,
                 retry_after: Optional[int] = None,
                 **kwargs):
        """初始化频率限制错误
        
        Args:
            message: 错误消息
            limit: 限制次数
            window_seconds: 时间窗口（秒）
            retry_after: 重试等待时间（秒）
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if limit:
            details['limit'] = limit
        if window_seconds:
            details['window_seconds'] = window_seconds
        if retry_after:
            details['retry_after'] = retry_after
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class DataError(PrometheusException):
    """数据错误异常"""
    
    def __init__(self, 
                 message: str = "Data error occurred",
                 data_type: Optional[str] = None,
                 data_source: Optional[str] = None,
                 **kwargs):
        """初始化数据错误
        
        Args:
            message: 错误消息
            data_type: 数据类型
            data_source: 数据源
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if data_type:
            details['data_type'] = data_type
        if data_source:
            details['data_source'] = data_source
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class ProcessingError(PrometheusException):
    """处理错误异常"""
    
    def __init__(self, 
                 message: str = "Processing error occurred",
                 process_name: Optional[str] = None,
                 stage: Optional[str] = None,
                 **kwargs):
        """初始化处理错误
        
        Args:
            message: 错误消息
            process_name: 处理过程名称
            stage: 处理阶段
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if process_name:
            details['process_name'] = process_name
        if stage:
            details['stage'] = stage
        kwargs['details'] = details
        super().__init__(message, **kwargs)