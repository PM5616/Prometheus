"""Logging Filters

自定义日志过滤器，用于过滤敏感数据、控制日志级别、限制频率等。
"""

import logging
import re
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Pattern, Set, Any
from datetime import datetime, timedelta


class SensitiveDataFilter(logging.Filter):
    """敏感数据过滤器"""
    
    def __init__(self, sensitive_patterns: Optional[List[str]] = None,
                 sensitive_fields: Optional[List[str]] = None,
                 mask_char: str = '*'):
        super().__init__()
        
        # 默认敏感字段
        self.sensitive_fields = sensitive_fields or [
            'password', 'token', 'key', 'secret', 'api_key', 'access_token',
            'refresh_token', 'private_key', 'credit_card', 'ssn', 'pin'
        ]
        
        # 默认敏感模式
        default_patterns = [
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # 信用卡号
            r'\b\d{3}-\d{2}-\d{4}\b',  # 社会安全号
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 邮箱
            r'(?i)(?:password|token|key|secret)\s*[:=]\s*["\']?([^\s"\',}]+)',  # 密码等
            r'\b(?:sk_|pk_|rk_)[a-zA-Z0-9]{20,}\b',  # API密钥
            r'\b[A-Za-z0-9+/]{40,}={0,2}\b',  # Base64编码的密钥
        ]
        
        patterns = (sensitive_patterns or []) + default_patterns
        self.sensitive_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        self.mask_char = mask_char
    
    def filter(self, record: logging.LogRecord) -> bool:
        """过滤敏感数据"""
        # 过滤消息
        if hasattr(record, 'msg'):
            record.msg = self._mask_sensitive_data(str(record.msg))
        
        # 过滤额外字段
        for key, value in record.__dict__.items():
            if self._is_sensitive_field(key):
                if isinstance(value, str):
                    record.__dict__[key] = self._mask_string(value)
                elif isinstance(value, dict):
                    record.__dict__[key] = self._mask_dict(value)
                elif isinstance(value, list):
                    record.__dict__[key] = self._mask_list(value)
        
        return True
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """检查字段是否为敏感字段"""
        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in self.sensitive_fields)
    
    def _mask_sensitive_data(self, text: str) -> str:
        """掩码文本中的敏感数据"""
        masked_text = text
        for pattern in self.sensitive_patterns:
            masked_text = pattern.sub('[MASKED]', masked_text)
        return masked_text
    
    def _mask_string(self, value: str) -> str:
        """掩码字符串值"""
        if len(value) <= 4:
            return self.mask_char * len(value)
        else:
            return value[:2] + self.mask_char * (len(value) - 4) + value[-2:]
    
    def _mask_dict(self, data: dict) -> dict:
        """掩码字典中的敏感数据"""
        masked_data = {}
        for key, value in data.items():
            if self._is_sensitive_field(key):
                if isinstance(value, str):
                    masked_data[key] = self._mask_string(value)
                else:
                    masked_data[key] = '[MASKED]'
            else:
                masked_data[key] = value
        return masked_data
    
    def _mask_list(self, data: list) -> list:
        """掩码列表中的敏感数据"""
        return ['[MASKED]' if isinstance(item, str) else item for item in data]


class LevelFilter(logging.Filter):
    """日志级别过滤器"""
    
    def __init__(self, min_level: int = logging.INFO, max_level: int = logging.CRITICAL,
                 allowed_levels: Optional[Set[int]] = None):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.allowed_levels = allowed_levels
    
    def filter(self, record: logging.LogRecord) -> bool:
        """根据级别过滤日志"""
        if self.allowed_levels:
            return record.levelno in self.allowed_levels
        
        return self.min_level <= record.levelno <= self.max_level


class RateLimitFilter(logging.Filter):
    """频率限制过滤器"""
    
    def __init__(self, max_rate: int = 100, time_window: int = 60,
                 burst_limit: int = 10, key_func: Optional[callable] = None):
        super().__init__()
        self.max_rate = max_rate  # 时间窗口内最大日志数
        self.time_window = time_window  # 时间窗口（秒）
        self.burst_limit = burst_limit  # 突发限制
        self.key_func = key_func or self._default_key_func
        
        # 存储每个键的日志时间戳
        self.log_times: Dict[str, deque] = defaultdict(lambda: deque())
        self.burst_counts: Dict[str, int] = defaultdict(int)
        self.last_reset: Dict[str, float] = defaultdict(float)
    
    def _default_key_func(self, record: logging.LogRecord) -> str:
        """默认键函数：使用日志器名称和消息"""
        return f"{record.name}:{hash(record.getMessage()) % 10000}"
    
    def filter(self, record: logging.LogRecord) -> bool:
        """根据频率限制过滤日志"""
        current_time = time.time()
        key = self.key_func(record)
        
        # 清理过期的时间戳
        self._cleanup_old_timestamps(key, current_time)
        
        # 检查突发限制
        if self._check_burst_limit(key, current_time):
            return False
        
        # 检查频率限制
        if len(self.log_times[key]) >= self.max_rate:
            # 记录被限制的日志
            self._log_rate_limit_warning(record, key)
            return False
        
        # 记录当前时间戳
        self.log_times[key].append(current_time)
        return True
    
    def _cleanup_old_timestamps(self, key: str, current_time: float):
        """清理过期的时间戳"""
        cutoff_time = current_time - self.time_window
        while self.log_times[key] and self.log_times[key][0] < cutoff_time:
            self.log_times[key].popleft()
    
    def _check_burst_limit(self, key: str, current_time: float) -> bool:
        """检查突发限制"""
        # 重置突发计数器（每秒重置）
        if current_time - self.last_reset[key] >= 1.0:
            self.burst_counts[key] = 0
            self.last_reset[key] = current_time
        
        # 检查突发限制
        if self.burst_counts[key] >= self.burst_limit:
            return True
        
        self.burst_counts[key] += 1
        return False
    
    def _log_rate_limit_warning(self, record: logging.LogRecord, key: str):
        """记录频率限制警告"""
        # 避免无限递归，只在特定条件下记录警告
        if not hasattr(self, '_warning_logged'):
            self._warning_logged = set()
        
        if key not in self._warning_logged:
            logger = logging.getLogger('prometheus.rate_limit')
            logger.warning(f"Rate limit exceeded for key: {key}")
            self._warning_logged.add(key)


class DuplicateFilter(logging.Filter):
    """重复日志过滤器"""
    
    def __init__(self, max_duplicates: int = 5, time_window: int = 300,
                 key_func: Optional[callable] = None):
        super().__init__()
        self.max_duplicates = max_duplicates
        self.time_window = time_window
        self.key_func = key_func or self._default_key_func
        
        # 存储消息计数和时间戳
        self.message_counts: Dict[str, int] = defaultdict(int)
        self.first_seen: Dict[str, float] = {}
        self.last_logged: Dict[str, float] = {}
    
    def _default_key_func(self, record: logging.LogRecord) -> str:
        """默认键函数：使用消息内容"""
        return f"{record.levelname}:{record.name}:{record.getMessage()}"
    
    def filter(self, record: logging.LogRecord) -> bool:
        """过滤重复日志"""
        current_time = time.time()
        key = self.key_func(record)
        
        # 清理过期的记录
        self._cleanup_expired_records(current_time)
        
        # 检查是否是新消息
        if key not in self.message_counts:
            self.message_counts[key] = 1
            self.first_seen[key] = current_time
            self.last_logged[key] = current_time
            return True
        
        # 增加计数
        self.message_counts[key] += 1
        
        # 检查是否超过重复限制
        if self.message_counts[key] <= self.max_duplicates:
            self.last_logged[key] = current_time
            return True
        
        # 定期记录重复统计
        if current_time - self.last_logged[key] >= 60:  # 每分钟记录一次统计
            self._log_duplicate_summary(record, key)
            self.last_logged[key] = current_time
        
        return False
    
    def _cleanup_expired_records(self, current_time: float):
        """清理过期的记录"""
        expired_keys = []
        for key, first_time in self.first_seen.items():
            if current_time - first_time > self.time_window:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.message_counts[key]
            del self.first_seen[key]
            del self.last_logged[key]
    
    def _log_duplicate_summary(self, record: logging.LogRecord, key: str):
        """记录重复日志统计"""
        count = self.message_counts[key]
        logger = logging.getLogger('prometheus.duplicate')
        logger.info(f"Suppressed {count - self.max_duplicates} duplicate messages: {record.getMessage()[:100]}")


class ModuleFilter(logging.Filter):
    """模块过滤器"""
    
    def __init__(self, allowed_modules: Optional[List[str]] = None,
                 blocked_modules: Optional[List[str]] = None,
                 use_regex: bool = False):
        super().__init__()
        self.allowed_modules = allowed_modules or []
        self.blocked_modules = blocked_modules or []
        self.use_regex = use_regex
        
        if use_regex:
            self.allowed_patterns = [re.compile(pattern) for pattern in self.allowed_modules]
            self.blocked_patterns = [re.compile(pattern) for pattern in self.blocked_modules]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """根据模块名过滤日志"""
        module_name = record.name
        
        # 检查阻止列表
        if self.blocked_modules:
            if self.use_regex:
                for pattern in self.blocked_patterns:
                    if pattern.search(module_name):
                        return False
            else:
                for blocked in self.blocked_modules:
                    if module_name.startswith(blocked):
                        return False
        
        # 检查允许列表
        if self.allowed_modules:
            if self.use_regex:
                for pattern in self.allowed_patterns:
                    if pattern.search(module_name):
                        return True
                return False
            else:
                for allowed in self.allowed_modules:
                    if module_name.startswith(allowed):
                        return True
                return False
        
        return True


class TimeRangeFilter(logging.Filter):
    """时间范围过滤器"""
    
    def __init__(self, start_time: Optional[str] = None, end_time: Optional[str] = None,
                 allowed_hours: Optional[List[int]] = None,
                 blocked_hours: Optional[List[int]] = None):
        super().__init__()
        
        # 解析时间范围
        self.start_time = self._parse_time(start_time) if start_time else None
        self.end_time = self._parse_time(end_time) if end_time else None
        
        self.allowed_hours = set(allowed_hours) if allowed_hours else None
        self.blocked_hours = set(blocked_hours) if blocked_hours else None
    
    def _parse_time(self, time_str: str) -> datetime:
        """解析时间字符串"""
        try:
            return datetime.strptime(time_str, '%H:%M:%S')
        except ValueError:
            try:
                return datetime.strptime(time_str, '%H:%M')
            except ValueError:
                raise ValueError(f"Invalid time format: {time_str}. Use HH:MM:SS or HH:MM")
    
    def filter(self, record: logging.LogRecord) -> bool:
        """根据时间范围过滤日志"""
        current_time = datetime.fromtimestamp(record.created)
        current_hour = current_time.hour
        
        # 检查阻止小时
        if self.blocked_hours and current_hour in self.blocked_hours:
            return False
        
        # 检查允许小时
        if self.allowed_hours and current_hour not in self.allowed_hours:
            return False
        
        # 检查时间范围
        if self.start_time and self.end_time:
            current_time_only = current_time.time()
            start_time_only = self.start_time.time()
            end_time_only = self.end_time.time()
            
            if start_time_only <= end_time_only:
                # 同一天内的时间范围
                return start_time_only <= current_time_only <= end_time_only
            else:
                # 跨天的时间范围
                return current_time_only >= start_time_only or current_time_only <= end_time_only
        
        return True


class ContextFilter(logging.Filter):
    """上下文过滤器"""
    
    def __init__(self, required_fields: Optional[List[str]] = None,
                 field_values: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.required_fields = required_fields or []
        self.field_values = field_values or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """根据上下文字段过滤日志"""
        # 检查必需字段
        for field in self.required_fields:
            if not hasattr(record, field):
                return False
        
        # 检查字段值
        for field, expected_value in self.field_values.items():
            if not hasattr(record, field):
                return False
            
            actual_value = getattr(record, field)
            if isinstance(expected_value, (list, tuple, set)):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False
        
        return True


class SizeFilter(logging.Filter):
    """消息大小过滤器"""
    
    def __init__(self, max_size: int = 10000, truncate: bool = True):
        super().__init__()
        self.max_size = max_size
        self.truncate = truncate
    
    def filter(self, record: logging.LogRecord) -> bool:
        """根据消息大小过滤日志"""
        message = record.getMessage()
        message_size = len(message.encode('utf-8'))
        
        if message_size > self.max_size:
            if self.truncate:
                # 截断消息
                truncated_message = message[:self.max_size - 50] + '... [TRUNCATED]'
                record.msg = truncated_message
                record.args = ()
            else:
                # 拒绝日志
                return False
        
        return True


class ExceptionFilter(logging.Filter):
    """异常过滤器"""
    
    def __init__(self, allowed_exceptions: Optional[List[str]] = None,
                 blocked_exceptions: Optional[List[str]] = None,
                 include_traceback: bool = True):
        super().__init__()
        self.allowed_exceptions = allowed_exceptions or []
        self.blocked_exceptions = blocked_exceptions or []
        self.include_traceback = include_traceback
    
    def filter(self, record: logging.LogRecord) -> bool:
        """根据异常类型过滤日志"""
        if not record.exc_info:
            return True
        
        exception_type = record.exc_info[0].__name__ if record.exc_info[0] else None
        
        if not exception_type:
            return True
        
        # 检查阻止列表
        if self.blocked_exceptions:
            for blocked in self.blocked_exceptions:
                if exception_type == blocked or exception_type.endswith(blocked):
                    return False
        
        # 检查允许列表
        if self.allowed_exceptions:
            for allowed in self.allowed_exceptions:
                if exception_type == allowed or exception_type.endswith(allowed):
                    break
            else:
                return False
        
        # 控制是否包含堆栈跟踪
        if not self.include_traceback:
            record.exc_info = None
            record.exc_text = None
        
        return True