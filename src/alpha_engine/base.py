"""Base Strategy Compatibility Module

为了保持向后兼容性，提供原strategy模块的核心类和接口。
这个模块将原strategy/base.py的关键功能整合到alpha_engine中。

主要功能：
- 策略抽象基类（兼容原接口）
- 策略信号定义（兼容原接口）
- 策略状态管理
- 策略参数管理
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import time
import json

from src.common.logging import get_logger
from src.common.models.market import Kline, Ticker, OrderBook
from src.common.exceptions.strategy import StrategyError, StrategyInitializationError

# 重新导出alpha_engine的信号类型，保持兼容性
from .signal import SignalType, SignalStrength, Signal as AlphaSignal
from .base_strategy import BaseStrategy as AlphaBaseStrategy, StrategyConfig, StrategyState as AlphaStrategyState


# 为了兼容性，保留原来的枚举定义
class StrategyState(Enum):
    """策略状态枚举（兼容性）"""
    INITIALIZED = "initialized"  # 已初始化
    RUNNING = "running"          # 运行中
    PAUSED = "paused"            # 已暂停
    STOPPED = "stopped"          # 已停止
    ERROR = "error"              # 错误状态


class StrategySignal:
    """策略信号类（兼容性）
    
    封装策略生成的交易信号信息，保持与原strategy模块的兼容性。
    """
    
    def __init__(self,
                 symbol: str,
                 signal_type: SignalType,
                 strength: SignalStrength = SignalStrength.MODERATE,
                 price: float = None,
                 quantity: float = None,
                 timestamp: datetime = None,
                 metadata: Dict = None):
        """初始化策略信号
        
        Args:
            symbol: 交易对符号
            signal_type: 信号类型
            strength: 信号强度
            price: 建议价格
            quantity: 建议数量
            timestamp: 信号时间戳
            metadata: 附加元数据
        """
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        
        # 信号唯一标识
        self.signal_id = f"{symbol}_{signal_type.value}_{int(self.timestamp.timestamp())}"
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 信号字典
        """
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'price': self.price,
            'quantity': self.quantity,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategySignal':
        """从字典创建信号
        
        Args:
            data: 信号字典
            
        Returns:
            StrategySignal: 策略信号实例
        """
        return cls(
            symbol=data['symbol'],
            signal_type=SignalType(data['signal_type']),
            strength=SignalStrength(data['strength']),
            price=data.get('price'),
            quantity=data.get('quantity'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )
    
    def to_alpha_signal(self) -> AlphaSignal:
        """转换为alpha_engine的Signal对象
        
        Returns:
            AlphaSignal: alpha_engine信号对象
        """
        return AlphaSignal(
            symbol=self.symbol,
            signal_type=self.signal_type,
            strength=self.strength,
            price=self.price,
            quantity=self.quantity,
            timestamp=self.timestamp,
            metadata=self.metadata
        )
    
    def __str__(self) -> str:
        return f"Signal({self.symbol}, {self.signal_type.value}, {self.strength.value})"
    
    def __repr__(self) -> str:
        return self.__str__()


class StrategyParameters:
    """策略参数管理类（兼容性）"""
    
    def __init__(self, parameters: Dict = None):
        """初始化策略参数
        
        Args:
            parameters: 参数字典
        """
        self._parameters = parameters or {}
        self._parameter_types = {}
        self._parameter_ranges = {}
        self._parameter_descriptions = {}
    
    def add_parameter(self, 
                     name: str, 
                     value: Any, 
                     param_type: type = None,
                     valid_range: tuple = None,
                     description: str = None):
        """添加参数
        
        Args:
            name: 参数名称
            value: 参数值
            param_type: 参数类型
            valid_range: 有效范围 (min, max)
            description: 参数描述
        """
        self._parameters[name] = value
        
        if param_type:
            self._parameter_types[name] = param_type
        
        if valid_range:
            self._parameter_ranges[name] = valid_range
        
        if description:
            self._parameter_descriptions[name] = description
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """获取参数值
        
        Args:
            name: 参数名称
            default: 默认值
            
        Returns:
            Any: 参数值
        """
        return self._parameters.get(name, default)
    
    def set_parameter(self, name: str, value: Any) -> bool:
        """设置参数值
        
        Args:
            name: 参数名称
            value: 参数值
            
        Returns:
            bool: 设置是否成功
        """
        try:
            # 类型检查
            if name in self._parameter_types:
                expected_type = self._parameter_types[name]
                if not isinstance(value, expected_type):
                    try:
                        value = expected_type(value)
                    except (ValueError, TypeError):
                        return False
            
            # 范围检查
            if name in self._parameter_ranges:
                min_val, max_val = self._parameter_ranges[name]
                if not (min_val <= value <= max_val):
                    return False
            
            self._parameters[name] = value
            return True
            
        except Exception:
            return False
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """获取所有参数
        
        Returns:
            Dict[str, Any]: 所有参数字典
        """
        return self._parameters.copy()
    
    def validate_parameters(self) -> bool:
        """验证所有参数
        
        Returns:
            bool: 验证是否通过
        """
        for name, value in self._parameters.items():
            if not self.set_parameter(name, value):
                return False
        return True


class BaseStrategy(ABC):
    """策略基类（兼容性）
    
    保持与原strategy模块的兼容性，同时集成alpha_engine的功能。
    """
    
    def __init__(self, name: str, parameters: Dict = None):
        """初始化策略
        
        Args:
            name: 策略名称
            parameters: 策略参数
        """
        self.name = name
        self.logger = get_logger(f"strategy.{name}")
        
        # 参数管理
        self.parameters = StrategyParameters(parameters)
        
        # 状态管理
        self.state = StrategyState.INITIALIZED
        self.is_running = False
        
        # 数据缓存
        self._market_data_cache = {}
        self._signal_history = []
        
        # 性能统计
        self.total_signals = 0
        self.successful_signals = 0
        self.start_time = None
        self.last_signal_time = None
        
        # 初始化策略
        self.initialize()
    
    @abstractmethod
    def initialize(self):
        """初始化策略（子类实现）"""
        pass
    
    @abstractmethod
    def generate_signal(self, market_data: Dict) -> Optional[StrategySignal]:
        """生成交易信号（子类实现）
        
        Args:
            market_data: 市场数据
            
        Returns:
            Optional[StrategySignal]: 生成的信号
        """
        pass
    
    def start(self):
        """启动策略"""
        if self.is_running:
            self.logger.warning(f"策略 {self.name} 已在运行中")
            return
        
        self.logger.info(f"启动策略: {self.name}")
        self.is_running = True
        self.state = StrategyState.RUNNING
        self.start_time = datetime.now()
        
        self.on_start()
    
    def stop(self):
        """停止策略"""
        if not self.is_running:
            self.logger.warning(f"策略 {self.name} 未在运行")
            return
        
        self.logger.info(f"停止策略: {self.name}")
        self.is_running = False
        self.state = StrategyState.STOPPED
        
        self.on_stop()
    
    def pause(self):
        """暂停策略"""
        if not self.is_running:
            self.logger.warning(f"策略 {self.name} 未在运行")
            return
        
        self.logger.info(f"暂停策略: {self.name}")
        self.is_running = False
        self.state = StrategyState.PAUSED
        
        self.on_pause()
    
    def resume(self):
        """恢复策略"""
        if self.state != StrategyState.PAUSED:
            self.logger.warning(f"策略 {self.name} 未处于暂停状态")
            return
        
        self.logger.info(f"恢复策略: {self.name}")
        self.is_running = True
        self.state = StrategyState.RUNNING
        
        self.on_resume()
    
    def update_market_data(self, symbol: str, data: Union[Kline, Ticker, OrderBook]):
        """更新市场数据
        
        Args:
            symbol: 交易对
            data: 市场数据
        """
        if symbol not in self._market_data_cache:
            self._market_data_cache[symbol] = []
        
        self._market_data_cache[symbol].append(data)
        
        # 保持缓存大小
        max_cache_size = self.parameters.get_parameter('max_cache_size', 1000)
        if len(self._market_data_cache[symbol]) > max_cache_size:
            self._market_data_cache[symbol] = self._market_data_cache[symbol][-max_cache_size:]
    
    def process_signal(self, signal: StrategySignal):
        """处理生成的信号
        
        Args:
            signal: 策略信号
        """
        if signal:
            self._signal_history.append(signal)
            self.total_signals += 1
            self.last_signal_time = signal.timestamp
            
            self.logger.info(f"生成信号: {signal}")
            self.on_signal_generated(signal)
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要
        
        Returns:
            Dict: 性能摘要
        """
        runtime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        success_rate = self.successful_signals / self.total_signals if self.total_signals > 0 else 0
        
        return {
            'strategy_name': self.name,
            'state': self.state.value,
            'runtime_seconds': runtime,
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'success_rate': success_rate,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None
        }
    
    # 生命周期回调方法（子类可选择性重写）
    def on_start(self):
        """策略启动回调"""
        pass
    
    def on_stop(self):
        """策略停止回调"""
        pass
    
    def on_pause(self):
        """策略暂停回调"""
        pass
    
    def on_resume(self):
        """策略恢复回调"""
        pass
    
    def on_signal_generated(self, signal: StrategySignal):
        """信号生成回调
        
        Args:
            signal: 生成的信号
        """
        pass
    
    def on_error(self, error: Exception):
        """错误处理回调
        
        Args:
            error: 发生的错误
        """
        self.logger.error(f"策略错误: {error}")
        self.state = StrategyState.ERROR