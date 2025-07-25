"""Logging Formatters

自定义日志格式化器，支持JSON、彩色、结构化等多种格式。
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None,
                 include_extra: bool = True, exclude_fields: Optional[list] = None):
        super().__init__()
        self.include_extra = include_extra
        self.exclude_fields = exclude_fields or []
        self.datefmt = datefmt or '%Y-%m-%dT%H:%M:%S.%fZ'
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为JSON"""
        # 基础字段
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).strftime(self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'pathname': record.pathname,
            'filename': record.filename,
            'process': record.process,
            'thread': record.thread,
            'thread_name': record.threadName if hasattr(record, 'threadName') else None
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if self.include_extra:
            for key, value in record.__dict__.items():
                if (key not in log_data and 
                    key not in self.exclude_fields and 
                    not key.startswith('_') and
                    key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                               'pathname', 'filename', 'module', 'lineno', 
                               'funcName', 'created', 'msecs', 'relativeCreated',
                               'thread', 'threadName', 'processName', 'process',
                               'exc_info', 'exc_text', 'stack_info']):
                    try:
                        # 确保值可以JSON序列化
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)
        
        # 移除排除的字段
        for field in self.exclude_fields:
            log_data.pop(field, None)
        
        return json.dumps(log_data, ensure_ascii=False, separators=(',', ':'))


class ColoredFormatter(logging.Formatter):
    """彩色格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    
    RESET = '\033[0m'  # 重置颜色
    BOLD = '\033[1m'   # 粗体
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None,
                 use_colors: bool = True):
        if fmt is None:
            fmt = ('%(asctime)s - %(name)s - %(levelname)s - '
                   '%(module)s:%(funcName)s:%(lineno)d - %(message)s')
        
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and self._supports_color()
    
    def _supports_color(self) -> bool:
        """检查终端是否支持颜色"""
        return (
            hasattr(sys.stderr, 'isatty') and sys.stderr.isatty() and
            'TERM' in os.environ and os.environ['TERM'] != 'dumb'
        )
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录并添加颜色"""
        if not self.use_colors:
            return super().format(record)
        
        # 获取颜色
        color = self.COLORS.get(record.levelname, '')
        
        # 保存原始级别名称
        original_levelname = record.levelname
        
        # 添加颜色到级别名称
        if color:
            record.levelname = f"{color}{self.BOLD}{record.levelname}{self.RESET}"
        
        # 格式化消息
        formatted = super().format(record)
        
        # 恢复原始级别名称
        record.levelname = original_levelname
        
        return formatted


class StructuredFormatter(logging.Formatter):
    """结构化格式化器"""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None,
                 include_context: bool = True, max_message_length: int = 1000):
        super().__init__(fmt, datefmt)
        self.include_context = include_context
        self.max_message_length = max_message_length
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为结构化格式"""
        # 基础信息
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # 截断过长的消息
        message = record.getMessage()
        if len(message) > self.max_message_length:
            message = message[:self.max_message_length] + '...'
        
        # 构建格式化字符串
        parts = [
            f"[{timestamp}]",
            f"[{record.levelname:8}]",
            f"[{record.name}]",
            f"[{record.module}:{record.funcName}:{record.lineno}]",
            message
        ]
        
        formatted = ' '.join(parts)
        
        # 添加上下文信息
        if self.include_context:
            context_parts = []
            
            # 进程和线程信息
            if record.process:
                context_parts.append(f"pid={record.process}")
            if record.thread:
                context_parts.append(f"tid={record.thread}")
            
            # 额外字段
            extra_fields = self._extract_extra_fields(record)
            if extra_fields:
                context_parts.extend([f"{k}={v}" for k, v in extra_fields.items()])
            
            if context_parts:
                formatted += f" [{', '.join(context_parts)}]"
        
        # 添加异常信息
        if record.exc_info:
            formatted += '\n' + self.formatException(record.exc_info)
        
        return formatted
    
    def _extract_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """提取额外字段"""
        extra_fields = {}
        
        # 标准字段列表
        standard_fields = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'exc_info', 'exc_text',
            'stack_info', 'getMessage', 'message'
        }
        
        for key, value in record.__dict__.items():
            if (key not in standard_fields and 
                not key.startswith('_') and
                not callable(value)):
                # 简化复杂对象
                if isinstance(value, (dict, list)):
                    extra_fields[key] = str(value)[:100] + '...' if len(str(value)) > 100 else value
                else:
                    extra_fields[key] = value
        
        return extra_fields


class CompactFormatter(logging.Formatter):
    """紧凑格式化器"""
    
    def __init__(self, include_timestamp: bool = True, include_level: bool = True,
                 include_logger: bool = False, include_location: bool = False):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_location = include_location
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为紧凑格式"""
        parts = []
        
        # 时间戳
        if self.include_timestamp:
            timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            parts.append(timestamp)
        
        # 级别
        if self.include_level:
            level_short = {
                'DEBUG': 'D',
                'INFO': 'I',
                'WARNING': 'W',
                'ERROR': 'E',
                'CRITICAL': 'C'
            }.get(record.levelname, record.levelname[0])
            parts.append(level_short)
        
        # 日志器名称
        if self.include_logger:
            logger_short = record.name.split('.')[-1]  # 只取最后一部分
            parts.append(logger_short)
        
        # 位置信息
        if self.include_location:
            location = f"{record.module}:{record.lineno}"
            parts.append(location)
        
        # 消息
        message = record.getMessage()
        
        # 组合所有部分
        if parts:
            prefix = '[' + '|'.join(parts) + ']'
            return f"{prefix} {message}"
        else:
            return message


class MultilineFormatter(logging.Formatter):
    """多行格式化器"""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None,
                 indent: str = '  ', max_width: int = 120):
        super().__init__(fmt, datefmt)
        self.indent = indent
        self.max_width = max_width
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为多行格式"""
        # 基础格式化
        formatted = super().format(record)
        
        # 分割长行
        lines = []
        for line in formatted.split('\n'):
            if len(line) <= self.max_width:
                lines.append(line)
            else:
                # 分割长行
                words = line.split(' ')
                current_line = ''
                
                for word in words:
                    if len(current_line + ' ' + word) <= self.max_width:
                        current_line += (' ' + word) if current_line else word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
        
        # 添加缩进（除了第一行）
        if len(lines) > 1:
            for i in range(1, len(lines)):
                lines[i] = self.indent + lines[i]
        
        return '\n'.join(lines)


class SensitiveDataFormatter(logging.Formatter):
    """敏感数据格式化器"""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None,
                 sensitive_fields: Optional[list] = None, mask_char: str = '*'):
        super().__init__(fmt, datefmt)
        self.sensitive_fields = sensitive_fields or [
            'password', 'token', 'key', 'secret', 'api_key', 'access_token',
            'refresh_token', 'private_key', 'credit_card', 'ssn'
        ]
        self.mask_char = mask_char
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录并掩码敏感数据"""
        # 复制记录以避免修改原始记录
        record_copy = logging.makeLogRecord(record.__dict__)
        
        # 掩码消息中的敏感数据
        if hasattr(record_copy, 'msg'):
            record_copy.msg = self._mask_sensitive_data(str(record_copy.msg))
        
        # 掩码额外字段中的敏感数据
        for key, value in record_copy.__dict__.items():
            if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                if isinstance(value, str):
                    record_copy.__dict__[key] = self._mask_string(value)
                elif isinstance(value, dict):
                    record_copy.__dict__[key] = self._mask_dict(value)
        
        return super().format(record_copy)
    
    def _mask_sensitive_data(self, text: str) -> str:
        """掩码文本中的敏感数据"""
        import re
        
        # 掩码常见的敏感数据模式
        patterns = [
            (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', 'CARD'),  # 信用卡号
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),  # 社会安全号
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL'),  # 邮箱
            (r'\b(?:password|token|key|secret)\s*[:=]\s*["\']?([^\s"\',}]+)', 'SENSITIVE'),  # 密码等
        ]
        
        masked_text = text
        for pattern, replacement in patterns:
            masked_text = re.sub(pattern, f'[{replacement}_MASKED]', masked_text, flags=re.IGNORECASE)
        
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
            if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                if isinstance(value, str):
                    masked_data[key] = self._mask_string(value)
                else:
                    masked_data[key] = '[MASKED]'
            else:
                masked_data[key] = value
        return masked_data


class PerformanceFormatter(logging.Formatter):
    """性能格式化器"""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None,
                 include_memory: bool = True, include_timing: bool = True):
        super().__init__(fmt, datefmt)
        self.include_memory = include_memory
        self.include_timing = include_timing
        self._start_time = datetime.now()
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录并添加性能信息"""
        # 基础格式化
        formatted = super().format(record)
        
        performance_info = []
        
        # 添加时间信息
        if self.include_timing:
            elapsed = datetime.now() - self._start_time
            performance_info.append(f"elapsed={elapsed.total_seconds():.3f}s")
        
        # 添加内存信息
        if self.include_memory:
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                performance_info.append(f"memory={memory_mb:.1f}MB")
            except ImportError:
                pass
        
        # 添加性能信息到日志
        if performance_info:
            formatted += f" [{', '.join(performance_info)}]"
        
        return formatted