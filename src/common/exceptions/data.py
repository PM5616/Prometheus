"""Data Exception Classes

数据相关异常类定义。
"""

from typing import Optional, Dict, Any, List
from .base import PrometheusException


class DataException(PrometheusException):
    """数据异常基类"""
    
    def __init__(self, 
                 message: str = "Data error occurred",
                 data_source: Optional[str] = None,
                 data_type: Optional[str] = None,
                 **kwargs):
        """初始化数据异常
        
        Args:
            message: 错误消息
            data_source: 数据源
            data_type: 数据类型
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if data_source:
            details['data_source'] = data_source
        if data_type:
            details['data_type'] = data_type
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class DataNotFoundError(DataException):
    """数据未找到异常"""
    
    def __init__(self, 
                 message: str = "Data not found",
                 query_params: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """初始化数据未找到异常
        
        Args:
            message: 错误消息
            query_params: 查询参数
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if query_params:
            details['query_params'] = query_params
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class DataValidationError(DataException):
    """数据验证异常"""
    
    def __init__(self, 
                 message: str = "Data validation failed",
                 validation_errors: Optional[List[str]] = None,
                 invalid_fields: Optional[List[str]] = None,
                 **kwargs):
        """初始化数据验证异常
        
        Args:
            message: 错误消息
            validation_errors: 验证错误列表
            invalid_fields: 无效字段列表
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if validation_errors:
            details['validation_errors'] = validation_errors
        if invalid_fields:
            details['invalid_fields'] = invalid_fields
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class DataFormatError(DataException):
    """数据格式异常"""
    
    def __init__(self, 
                 message: str = "Invalid data format",
                 expected_format: Optional[str] = None,
                 actual_format: Optional[str] = None,
                 **kwargs):
        """初始化数据格式异常
        
        Args:
            message: 错误消息
            expected_format: 期望格式
            actual_format: 实际格式
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if expected_format:
            details['expected_format'] = expected_format
        if actual_format:
            details['actual_format'] = actual_format
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class DataCorruptionError(DataException):
    """数据损坏异常"""
    
    def __init__(self, 
                 message: str = "Data corruption detected",
                 corruption_type: Optional[str] = None,
                 affected_records: Optional[int] = None,
                 **kwargs):
        """初始化数据损坏异常
        
        Args:
            message: 错误消息
            corruption_type: 损坏类型
            affected_records: 受影响记录数
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if corruption_type:
            details['corruption_type'] = corruption_type
        if affected_records is not None:
            details['affected_records'] = affected_records
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class DataInconsistencyError(DataException):
    """数据不一致异常"""
    
    def __init__(self, 
                 message: str = "Data inconsistency detected",
                 inconsistency_type: Optional[str] = None,
                 conflicting_sources: Optional[List[str]] = None,
                 **kwargs):
        """初始化数据不一致异常
        
        Args:
            message: 错误消息
            inconsistency_type: 不一致类型
            conflicting_sources: 冲突数据源
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if inconsistency_type:
            details['inconsistency_type'] = inconsistency_type
        if conflicting_sources:
            details['conflicting_sources'] = conflicting_sources
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class DataStaleError(DataException):
    """数据过期异常"""
    
    def __init__(self, 
                 message: str = "Data is stale",
                 data_timestamp: Optional[str] = None,
                 max_age_seconds: Optional[int] = None,
                 **kwargs):
        """初始化数据过期异常
        
        Args:
            message: 错误消息
            data_timestamp: 数据时间戳
            max_age_seconds: 最大年龄（秒）
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if data_timestamp:
            details['data_timestamp'] = data_timestamp
        if max_age_seconds is not None:
            details['max_age_seconds'] = max_age_seconds
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class DataConnectionError(DataException):
    """数据连接异常"""
    
    def __init__(self, 
                 message: str = "Data connection failed",
                 connection_string: Optional[str] = None,
                 retry_count: Optional[int] = None,
                 **kwargs):
        """初始化数据连接异常
        
        Args:
            message: 错误消息
            connection_string: 连接字符串（敏感信息会被掩码）
            retry_count: 重试次数
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if connection_string:
            # 掩码敏感信息
            masked_connection = self._mask_connection_string(connection_string)
            details['connection_string'] = masked_connection
        if retry_count is not None:
            details['retry_count'] = retry_count
        kwargs['details'] = details
        super().__init__(message, **kwargs)
    
    @staticmethod
    def _mask_connection_string(connection_string: str) -> str:
        """掩码连接字符串中的敏感信息"""
        import re
        # 掩码密码
        masked = re.sub(r'(password=)[^;\s]+', r'\1***', connection_string, flags=re.IGNORECASE)
        # 掩码API密钥
        masked = re.sub(r'(api[_-]?key=)[^;\s]+', r'\1***', masked, flags=re.IGNORECASE)
        # 掩码token
        masked = re.sub(r'(token=)[^;\s]+', r'\1***', masked, flags=re.IGNORECASE)
        return masked


class DatabaseError(DataException):
    """数据库异常"""
    
    def __init__(self, 
                 message: str = "Database error occurred",
                 operation: Optional[str] = None,
                 table_name: Optional[str] = None,
                 **kwargs):
        """初始化数据库异常
        
        Args:
            message: 错误消息
            operation: 数据库操作
            table_name: 表名
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        if table_name:
            details['table_name'] = table_name
        kwargs['details'] = details
        super().__init__(message, data_source="database", **kwargs)


class CacheError(DataException):
    """缓存异常"""
    
    def __init__(self, 
                 message: str = "Cache error occurred",
                 cache_key: Optional[str] = None,
                 cache_operation: Optional[str] = None,
                 **kwargs):
        """初始化缓存异常
        
        Args:
            message: 错误消息
            cache_key: 缓存键
            cache_operation: 缓存操作
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if cache_key:
            details['cache_key'] = cache_key
        if cache_operation:
            details['cache_operation'] = cache_operation
        kwargs['details'] = details
        super().__init__(message, data_source="cache", **kwargs)


class APIDataError(DataException):
    """API数据异常"""
    
    def __init__(self, 
                 message: str = "API data error occurred",
                 api_endpoint: Optional[str] = None,
                 status_code: Optional[int] = None,
                 response_data: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """初始化API数据异常
        
        Args:
            message: 错误消息
            api_endpoint: API端点
            status_code: HTTP状态码
            response_data: 响应数据
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if api_endpoint:
            details['api_endpoint'] = api_endpoint
        if status_code is not None:
            details['status_code'] = status_code
        if response_data:
            # 限制响应数据大小
            details['response_data'] = str(response_data)[:1000]
        kwargs['details'] = details
        super().__init__(message, data_source="api", **kwargs)


class FileDataError(DataException):
    """文件数据异常"""
    
    def __init__(self, 
                 message: str = "File data error occurred",
                 file_path: Optional[str] = None,
                 file_operation: Optional[str] = None,
                 **kwargs):
        """初始化文件数据异常
        
        Args:
            message: 错误消息
            file_path: 文件路径
            file_operation: 文件操作
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = file_path
        if file_operation:
            details['file_operation'] = file_operation
        kwargs['details'] = details
        super().__init__(message, data_source="file", **kwargs)


class DataTransformationError(DataException):
    """数据转换异常"""
    
    def __init__(self, 
                 message: str = "Data transformation failed",
                 transformation_type: Optional[str] = None,
                 input_format: Optional[str] = None,
                 output_format: Optional[str] = None,
                 **kwargs):
        """初始化数据转换异常
        
        Args:
            message: 错误消息
            transformation_type: 转换类型
            input_format: 输入格式
            output_format: 输出格式
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if transformation_type:
            details['transformation_type'] = transformation_type
        if input_format:
            details['input_format'] = input_format
        if output_format:
            details['output_format'] = output_format
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class DataSizeError(DataException):
    """数据大小异常"""
    
    def __init__(self, 
                 message: str = "Data size error",
                 data_size: Optional[int] = None,
                 size_limit: Optional[int] = None,
                 size_unit: Optional[str] = "bytes",
                 **kwargs):
        """初始化数据大小异常
        
        Args:
            message: 错误消息
            data_size: 数据大小
            size_limit: 大小限制
            size_unit: 大小单位
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if data_size is not None:
            details['data_size'] = data_size
        if size_limit is not None:
            details['size_limit'] = size_limit
        if size_unit:
            details['size_unit'] = size_unit
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class DataQualityError(DataException):
    """数据质量异常"""
    
    def __init__(self, 
                 message: str = "Data quality issue detected",
                 quality_issues: Optional[List[str]] = None,
                 quality_score: Optional[float] = None,
                 **kwargs):
        """初始化数据质量异常
        
        Args:
            message: 错误消息
            quality_issues: 质量问题列表
            quality_score: 质量评分
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if quality_issues:
            details['quality_issues'] = quality_issues
        if quality_score is not None:
            details['quality_score'] = quality_score
        kwargs['details'] = details
        super().__init__(message, **kwargs)