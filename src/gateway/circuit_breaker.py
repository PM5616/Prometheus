"""Circuit Breaker Module

熔断器模块，提供服务熔断和故障恢复功能。
"""

import time
import threading
from enum import Enum
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from src.common.logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 关闭状态，正常工作
    OPEN = "open"          # 开启状态，熔断中
    HALF_OPEN = "half_open" # 半开状态，尝试恢复


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5          # 失败阈值
    recovery_timeout: float = 60.0      # 恢复超时时间（秒）
    expected_exception: type = Exception # 期望的异常类型
    name: str = "default"               # 熔断器名称


class CircuitBreaker:
    """熔断器
    
    实现熔断器模式，防止级联故障。
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        """初始化熔断器
        
        Args:
            config: 熔断器配置
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = threading.RLock()
        
        logger.info(f"Circuit breaker '{config.name}' initialized")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """调用函数并应用熔断逻辑
        
        Args:
            func: 要调用的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数返回值
            
        Raises:
            Exception: 当熔断器开启时抛出异常
        """
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.config.name}' entering half-open state")
                else:
                    raise Exception(f"Circuit breaker '{self.config.name}' is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置
        
        Returns:
            bool: 是否应该尝试重置
        """
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _on_success(self):
        """成功时的处理"""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info(f"Circuit breaker '{self.config.name}' reset to closed state")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """失败时的处理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.config.name}' opened due to {self.failure_count} failures")
    
    def get_state(self) -> CircuitState:
        """获取当前状态
        
        Returns:
            CircuitState: 当前状态
        """
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标
        
        Returns:
            Dict[str, Any]: 指标数据
        """
        return {
            'name': self.config.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'failure_threshold': self.config.failure_threshold,
            'recovery_timeout': self.config.recovery_timeout
        }
    
    def reset(self):
        """手动重置熔断器"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit breaker '{self.config.name}' manually reset")