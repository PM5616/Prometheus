"""Prometheus Logger

统一的日志器实现，提供结构化日志记录功能。
"""

import logging
import logging.config
import sys
import os
from enum import Enum
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from ..config.settings import Settings


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """日志格式枚举"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    COLORED = "colored"


class PrometheusLogger:
    """Prometheus 日志器类"""
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    def __init__(self, name: str, settings: Optional[Settings] = None):
        """初始化日志器
        
        Args:
            name: 日志器名称
            settings: 系统设置
        """
        self.name = name
        self.settings = settings or Settings()
        self.logger = self._get_or_create_logger(name)
    
    def _get_or_create_logger(self, name: str) -> logging.Logger:
        """获取或创建日志器"""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
            
            # 如果还没有配置过日志系统，则进行配置
            if not self._configured:
                self._setup_logging()
        
        return self._loggers[name]
    
    def _setup_logging(self):
        """设置日志配置"""
        if self._configured:
            return
        
        # 创建日志目录
        log_dir = Path(self.settings.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建日志配置
        config = self._build_logging_config()
        
        # 应用配置
        logging.config.dictConfig(config)
        
        self._configured = True
    
    def _build_logging_config(self) -> Dict[str, Any]:
        """构建日志配置字典"""
        log_dir = self.settings.logging.log_dir
        
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'simple': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
                },
                'json': {
                    'class': 'src.common.logging.formatters.JSONFormatter'
                },
                'colored': {
                    'class': 'src.common.logging.formatters.ColoredFormatter'
                }
            },
            'filters': {
                'sensitive_data': {
                    'class': 'src.common.logging.filters.SensitiveDataFilter'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': self.settings.logging.console_level,
                    'formatter': 'colored' if self.settings.logging.colored_output else 'simple',
                    'stream': sys.stdout,
                    'filters': ['sensitive_data']
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': self.settings.logging.file_level,
                    'formatter': 'detailed',
                    'filename': os.path.join(log_dir, 'prometheus.log'),
                    'maxBytes': self.settings.logging.max_file_size,
                    'backupCount': self.settings.logging.backup_count,
                    'encoding': 'utf-8',
                    'filters': ['sensitive_data']
                },
                'error_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'ERROR',
                    'formatter': 'detailed',
                    'filename': os.path.join(log_dir, 'error.log'),
                    'maxBytes': self.settings.logging.max_file_size,
                    'backupCount': self.settings.logging.backup_count,
                    'encoding': 'utf-8',
                    'filters': ['sensitive_data']
                }
            },
            'loggers': {
                '': {  # root logger
                    'level': self.settings.logging.level,
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False
                },
                'prometheus': {
                    'level': self.settings.logging.level,
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False
                }
            }
        }
        
        # 添加数据库处理器（如果启用）
        if self.settings.logging.enable_database_logging:
            config['handlers']['database'] = {
                'class': 'src.common.logging.handlers.DatabaseHandler',
                'level': 'INFO',
                'formatter': 'json'
            }
            config['loggers']['']['handlers'].append('database')
            config['loggers']['prometheus']['handlers'].append('database')
        
        return config
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录调试日志"""
        self._log(logging.DEBUG, message, extra, **kwargs)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录信息日志"""
        self._log(logging.INFO, message, extra, **kwargs)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录警告日志"""
        self._log(logging.WARNING, message, extra, **kwargs)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = True, **kwargs):
        """记录错误日志"""
        self._log(logging.ERROR, message, extra, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = True, **kwargs):
        """记录严重错误日志"""
        self._log(logging.CRITICAL, message, extra, exc_info=exc_info, **kwargs)
    
    def exception(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录异常日志"""
        self._log(logging.ERROR, message, extra, exc_info=True, **kwargs)
    
    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """内部日志记录方法"""
        if extra is None:
            extra = {}
        
        # 添加上下文信息
        extra.update({
            'logger_name': self.name,
            'timestamp': datetime.utcnow().isoformat(),
            'process_id': os.getpid(),
            'thread_id': threading.get_ident() if 'threading' in sys.modules else None
        })
        
        # 添加自定义字段
        extra.update(kwargs)
        
        self.logger.log(level, message, extra=extra, **{k: v for k, v in kwargs.items() if k not in extra})
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """记录交易日志"""
        self.info(
            "Trade executed",
            extra={
                'event_type': 'trade',
                'trade_data': trade_data
            }
        )
    
    def log_order(self, order_data: Dict[str, Any]):
        """记录订单日志"""
        self.info(
            "Order processed",
            extra={
                'event_type': 'order',
                'order_data': order_data
            }
        )
    
    def log_strategy(self, strategy_name: str, action: str, data: Dict[str, Any]):
        """记录策略日志"""
        self.info(
            f"Strategy {action}: {strategy_name}",
            extra={
                'event_type': 'strategy',
                'strategy_name': strategy_name,
                'action': action,
                'strategy_data': data
            }
        )
    
    def log_risk(self, risk_type: str, risk_data: Dict[str, Any]):
        """记录风险日志"""
        level = logging.WARNING if risk_data.get('severity') == 'high' else logging.INFO
        self._log(
            level,
            f"Risk event: {risk_type}",
            extra={
                'event_type': 'risk',
                'risk_type': risk_type,
                'risk_data': risk_data
            }
        )
    
    def log_performance(self, metric_name: str, value: Union[int, float], unit: str = None):
        """记录性能指标日志"""
        self.info(
            f"Performance metric: {metric_name} = {value}{unit or ''}",
            extra={
                'event_type': 'performance',
                'metric_name': metric_name,
                'metric_value': value,
                'metric_unit': unit
            }
        )
    
    def log_api_call(self, endpoint: str, method: str, status_code: int, 
                     response_time: float, request_data: Dict[str, Any] = None):
        """记录API调用日志"""
        self.info(
            f"API call: {method} {endpoint} - {status_code} ({response_time:.3f}s)",
            extra={
                'event_type': 'api_call',
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'response_time': response_time,
                'request_data': request_data
            }
        )


def get_logger(name: str, settings: Optional[Settings] = None) -> PrometheusLogger:
    """获取日志器实例
    
    Args:
        name: 日志器名称
        settings: 系统设置
    
    Returns:
        PrometheusLogger: 日志器实例
    """
    return PrometheusLogger(name, settings)


def setup_logging(settings: Optional[Settings] = None, config_file: Optional[str] = None):
    """设置日志系统
    
    Args:
        settings: 系统设置
        config_file: 日志配置文件路径
    """
    if config_file and os.path.exists(config_file):
        # 从配置文件加载
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.json'):
                config = json.load(f)
            else:
                import yaml
                config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # 使用默认配置
        logger = PrometheusLogger('setup', settings)
        logger._setup_logging()


# 导入threading模块（如果可用）
try:
    import threading
except ImportError:
    threading = None