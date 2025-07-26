"""Base Strategy Module

策略基础框架，定义策略的抽象基类和核心数据结构。

主要功能：
- 策略抽象基类
- 策略信号定义
- 策略状态管理
- 策略参数管理
- 策略生命周期管理
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import time
import json

from ..common.logging import get_logger
from ..common.models.market_data import Kline, Ticker, OrderBook
from ..common.exceptions.strategy import StrategyError, StrategyInitializationError


class SignalType(Enum):
    """信号类型枚举"""
    BUY = "buy"              # 买入信号
    SELL = "sell"            # 卖出信号
    HOLD = "hold"            # 持有信号
    CLOSE_LONG = "close_long"    # 平多信号
    CLOSE_SHORT = "close_short"  # 平空信号
    STOP_LOSS = "stop_loss"      # 止损信号
    TAKE_PROFIT = "take_profit"  # 止盈信号


class SignalStrength(Enum):
    """信号强度枚举"""
    WEAK = 1      # 弱信号
    MEDIUM = 2    # 中等信号
    STRONG = 3    # 强信号
    URGENT = 4    # 紧急信号


class StrategyState(Enum):
    """策略状态枚举"""
    INITIALIZED = "initialized"  # 已初始化
    RUNNING = "running"          # 运行中
    PAUSED = "paused"            # 已暂停
    STOPPED = "stopped"          # 已停止
    ERROR = "error"              # 错误状态


class StrategySignal:
    """策略信号类
    
    封装策略生成的交易信号信息。
    """
    
    def __init__(self,
                 symbol: str,
                 signal_type: SignalType,
                 strength: SignalStrength = SignalStrength.MEDIUM,
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
    
    def __str__(self) -> str:
        return f"Signal({self.symbol}, {self.signal_type.value}, {self.strength.value})"
    
    def __repr__(self) -> str:
        return self.__str__()


class StrategyParameters:
    """策略参数管理类"""
    
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
    
    def validate_parameters(self) -> List[str]:
        """验证所有参数
        
        Returns:
            List[str]: 验证错误列表
        """
        errors = []
        
        for name, value in self._parameters.items():
            # 类型检查
            if name in self._parameter_types:
                expected_type = self._parameter_types[name]
                if not isinstance(value, expected_type):
                    errors.append(f"参数 {name} 类型错误，期望 {expected_type.__name__}")
            
            # 范围检查
            if name in self._parameter_ranges:
                min_val, max_val = self._parameter_ranges[name]
                if not (min_val <= value <= max_val):
                    errors.append(f"参数 {name} 超出范围 [{min_val}, {max_val}]")
        
        return errors
    
    def to_dict(self) -> Dict:
        """转换为字典
        
        Returns:
            Dict: 参数字典
        """
        return {
            'parameters': self._parameters.copy(),
            'types': {k: v.__name__ for k, v in self._parameter_types.items()},
            'ranges': self._parameter_ranges.copy(),
            'descriptions': self._parameter_descriptions.copy()
        }
    
    def update_from_dict(self, data: Dict):
        """从字典更新参数
        
        Args:
            data: 参数字典
        """
        if 'parameters' in data:
            self._parameters.update(data['parameters'])
    
    def __getitem__(self, key: str) -> Any:
        return self._parameters[key]
    
    def __setitem__(self, key: str, value: Any):
        self._parameters[key] = value
    
    def __contains__(self, key: str) -> bool:
        return key in self._parameters


class BaseStrategy(ABC):
    """策略抽象基类
    
    定义策略的基本接口和生命周期管理。
    所有具体策略都应该继承此类。
    """
    
    def __init__(self, 
                 name: str,
                 symbols: List[str],
                 parameters: Dict = None,
                 config: Dict = None):
        """初始化策略
        
        Args:
            name: 策略名称
            symbols: 交易对列表
            parameters: 策略参数
            config: 策略配置
        """
        self.name = name
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.config = config or {}
        
        # 策略参数
        self.parameters = StrategyParameters(parameters)
        self._setup_default_parameters()
        
        # 策略状态
        self.state = StrategyState.INITIALIZED
        self.created_at = datetime.now()
        self.started_at = None
        self.stopped_at = None
        
        # 日志记录
        self.logger = get_logger(f"Strategy.{self.name}")
        
        # 数据存储
        self.market_data = {}  # 存储市场数据
        self.indicators = {}   # 存储技术指标
        self.positions = {}    # 存储持仓信息
        self.signals = []      # 存储生成的信号
        
        # 性能统计
        self.stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'processing_time': 0.0,
            'last_signal_time': None,
            'error_count': 0
        }
        
        # 回调函数
        self.on_signal_callbacks = []
        self.on_error_callbacks = []
        
        # 风险控制
        self.risk_limits = self.config.get('risk_limits', {})
        
        self.logger.info(f"策略 {self.name} 初始化完成")
    
    @abstractmethod
    def _setup_default_parameters(self):
        """设置默认参数
        
        子类必须实现此方法来定义策略的默认参数。
        """
        pass
    
    @abstractmethod
    def on_market_data(self, symbol: str, data: Union[Kline, Ticker, OrderBook]) -> Optional[StrategySignal]:
        """处理市场数据
        
        Args:
            symbol: 交易对符号
            data: 市场数据
            
        Returns:
            Optional[StrategySignal]: 生成的信号
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """计算技术指标
        
        Args:
            symbol: 交易对符号
            data: 历史数据
            
        Returns:
            Dict[str, Any]: 计算的指标
        """
        pass
    
    @abstractmethod
    def generate_signal(self, symbol: str) -> Optional[StrategySignal]:
        """生成交易信号
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[StrategySignal]: 生成的信号
        """
        pass
    
    def initialize(self) -> bool:
        """初始化策略
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info(f"开始初始化策略 {self.name}")
            
            # 验证参数
            param_errors = self.parameters.validate_parameters()
            if param_errors:
                for error in param_errors:
                    self.logger.error(error)
                return False
            
            # 初始化数据结构
            for symbol in self.symbols:
                self.market_data[symbol] = []
                self.indicators[symbol] = {}
                self.positions[symbol] = {
                    'quantity': 0.0,
                    'avg_price': 0.0,
                    'unrealized_pnl': 0.0
                }
            
            # 调用子类初始化
            if not self._initialize_strategy():
                return False
            
            self.state = StrategyState.INITIALIZED
            self.logger.info(f"策略 {self.name} 初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"策略初始化失败: {e}")
            self.state = StrategyState.ERROR
            return False
    
    def _initialize_strategy(self) -> bool:
        """子类可重写的初始化方法
        
        Returns:
            bool: 初始化是否成功
        """
        return True
    
    def start(self) -> bool:
        """启动策略
        
        Returns:
            bool: 启动是否成功
        """
        try:
            if self.state != StrategyState.INITIALIZED:
                self.logger.error(f"策略状态错误，当前状态: {self.state.value}")
                return False
            
            self.started_at = datetime.now()
            self.state = StrategyState.RUNNING
            
            self.logger.info(f"策略 {self.name} 已启动")
            return True
            
        except Exception as e:
            self.logger.error(f"策略启动失败: {e}")
            self.state = StrategyState.ERROR
            return False
    
    def stop(self) -> bool:
        """停止策略
        
        Returns:
            bool: 停止是否成功
        """
        try:
            if self.state not in [StrategyState.RUNNING, StrategyState.PAUSED]:
                self.logger.warning(f"策略未在运行状态，当前状态: {self.state.value}")
                return True
            
            self.stopped_at = datetime.now()
            self.state = StrategyState.STOPPED
            
            self.logger.info(f"策略 {self.name} 已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"策略停止失败: {e}")
            return False
    
    def pause(self) -> bool:
        """暂停策略
        
        Returns:
            bool: 暂停是否成功
        """
        try:
            if self.state != StrategyState.RUNNING:
                self.logger.warning(f"策略未在运行状态，当前状态: {self.state.value}")
                return False
            
            self.state = StrategyState.PAUSED
            self.logger.info(f"策略 {self.name} 已暂停")
            return True
            
        except Exception as e:
            self.logger.error(f"策略暂停失败: {e}")
            return False
    
    def resume(self) -> bool:
        """恢复策略
        
        Returns:
            bool: 恢复是否成功
        """
        try:
            if self.state != StrategyState.PAUSED:
                self.logger.warning(f"策略未在暂停状态，当前状态: {self.state.value}")
                return False
            
            self.state = StrategyState.RUNNING
            self.logger.info(f"策略 {self.name} 已恢复")
            return True
            
        except Exception as e:
            self.logger.error(f"策略恢复失败: {e}")
            return False
    
    def process_market_data(self, symbol: str, data: Union[Kline, Ticker, OrderBook]) -> Optional[StrategySignal]:
        """处理市场数据的入口方法
        
        Args:
            symbol: 交易对符号
            data: 市场数据
            
        Returns:
            Optional[StrategySignal]: 生成的信号
        """
        if self.state != StrategyState.RUNNING:
            return None
        
        start_time = time.time()
        
        try:
            # 存储市场数据
            if symbol not in self.market_data:
                self.market_data[symbol] = []
            
            self.market_data[symbol].append(data)
            
            # 限制数据长度
            max_data_length = self.config.get('max_data_length', 1000)
            if len(self.market_data[symbol]) > max_data_length:
                self.market_data[symbol] = self.market_data[symbol][-max_data_length:]
            
            # 调用子类处理方法
            signal = self.on_market_data(symbol, data)
            
            # 处理生成的信号
            if signal:
                self._process_signal(signal)
            
            # 更新统计信息
            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
            
            return signal
            
        except Exception as e:
            self.logger.error(f"处理市场数据时发生错误: {e}")
            self.stats['error_count'] += 1
            self._handle_error(e)
            return None
    
    def _process_signal(self, signal: StrategySignal):
        """处理生成的信号
        
        Args:
            signal: 策略信号
        """
        # 风险检查
        if not self._check_risk_limits(signal):
            self.logger.warning(f"信号 {signal} 被风险控制拒绝")
            return
        
        # 存储信号
        self.signals.append(signal)
        
        # 更新统计信息
        self.stats['total_signals'] += 1
        self.stats['last_signal_time'] = signal.timestamp
        
        if signal.signal_type == SignalType.BUY:
            self.stats['buy_signals'] += 1
        elif signal.signal_type == SignalType.SELL:
            self.stats['sell_signals'] += 1
        
        # 调用回调函数
        for callback in self.on_signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                self.logger.error(f"信号回调函数执行失败: {e}")
        
        self.logger.info(f"生成信号: {signal}")
    
    def _check_risk_limits(self, signal: StrategySignal) -> bool:
        """检查风险限制
        
        Args:
            signal: 策略信号
            
        Returns:
            bool: 是否通过风险检查
        """
        try:
            # 检查最大持仓
            max_position = self.risk_limits.get('max_position')
            if max_position and signal.quantity:
                current_position = self.positions.get(signal.symbol, {}).get('quantity', 0)
                if signal.signal_type == SignalType.BUY:
                    new_position = current_position + signal.quantity
                elif signal.signal_type == SignalType.SELL:
                    new_position = current_position - signal.quantity
                else:
                    new_position = current_position
                
                if abs(new_position) > max_position:
                    return False
            
            # 检查信号频率
            max_signals_per_hour = self.risk_limits.get('max_signals_per_hour')
            if max_signals_per_hour:
                recent_signals = [
                    s for s in self.signals 
                    if s.symbol == signal.symbol and 
                    (datetime.now() - s.timestamp).total_seconds() < 3600
                ]
                if len(recent_signals) >= max_signals_per_hour:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"风险检查失败: {e}")
            return False
    
    def _handle_error(self, error: Exception):
        """处理错误
        
        Args:
            error: 异常对象
        """
        # 调用错误回调函数
        for callback in self.on_error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"错误回调函数执行失败: {e}")
    
    def add_signal_callback(self, callback: Callable[[StrategySignal], None]):
        """添加信号回调函数
        
        Args:
            callback: 回调函数
        """
        self.on_signal_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """添加错误回调函数
        
        Args:
            callback: 回调函数
        """
        self.on_error_callbacks.append(callback)
    
    def get_latest_data(self, symbol: str, count: int = 1) -> List[Union[Kline, Ticker, OrderBook]]:
        """获取最新的市场数据
        
        Args:
            symbol: 交易对符号
            count: 数据数量
            
        Returns:
            List: 最新的市场数据
        """
        if symbol not in self.market_data:
            return []
        
        return self.market_data[symbol][-count:]
    
    def get_position(self, symbol: str) -> Dict:
        """获取持仓信息
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Dict: 持仓信息
        """
        return self.positions.get(symbol, {
            'quantity': 0.0,
            'avg_price': 0.0,
            'unrealized_pnl': 0.0
        })
    
    def update_position(self, symbol: str, quantity: float, price: float):
        """更新持仓信息
        
        Args:
            symbol: 交易对符号
            quantity: 数量变化
            price: 成交价格
        """
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0.0,
                'avg_price': 0.0,
                'unrealized_pnl': 0.0
            }
        
        position = self.positions[symbol]
        old_quantity = position['quantity']
        old_avg_price = position['avg_price']
        
        # 计算新的持仓数量和平均价格
        new_quantity = old_quantity + quantity
        
        if new_quantity != 0:
            if old_quantity == 0:
                new_avg_price = price
            elif (old_quantity > 0 and quantity > 0) or (old_quantity < 0 and quantity < 0):
                # 同向加仓
                total_cost = old_quantity * old_avg_price + quantity * price
                new_avg_price = total_cost / new_quantity
            else:
                # 反向减仓或平仓
                new_avg_price = old_avg_price
        else:
            new_avg_price = 0.0
        
        position['quantity'] = new_quantity
        position['avg_price'] = new_avg_price
        
        self.logger.info(f"更新持仓 {symbol}: 数量={new_quantity}, 均价={new_avg_price}")
    
    def get_signals(self, symbol: str = None, limit: int = None) -> List[StrategySignal]:
        """获取信号历史
        
        Args:
            symbol: 交易对符号（可选）
            limit: 限制数量（可选）
            
        Returns:
            List[StrategySignal]: 信号列表
        """
        signals = self.signals
        
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        if limit:
            signals = signals[-limit:]
        
        return signals
    
    def get_stats(self) -> Dict:
        """获取策略统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = self.stats.copy()
        
        # 添加运行时间
        if self.started_at:
            if self.stopped_at:
                runtime = (self.stopped_at - self.started_at).total_seconds()
            else:
                runtime = (datetime.now() - self.started_at).total_seconds()
            stats['runtime_seconds'] = runtime
        
        # 添加状态信息
        stats.update({
            'state': self.state.value,
            'symbols': self.symbols,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'stopped_at': self.stopped_at.isoformat() if self.stopped_at else None
        })
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'processing_time': 0.0,
            'last_signal_time': None,
            'error_count': 0
        }
        self.logger.info("策略统计信息已重置")
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 策略字典
        """
        return {
            'name': self.name,
            'symbols': self.symbols,
            'state': self.state.value,
            'parameters': self.parameters.to_dict(),
            'config': self.config,
            'stats': self.get_stats(),
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'stopped_at': self.stopped_at.isoformat() if self.stopped_at else None
        }
    
    def __str__(self) -> str:
        return f"Strategy({self.name}, {self.state.value}, {len(self.symbols)} symbols)"
    
    def __repr__(self) -> str:
        return self.__str__()