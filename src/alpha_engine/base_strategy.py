"""Base Strategy Module

策略基类模块，为所有策略提供统一的接口和基础功能。

主要功能：
- 策略基类定义
- 策略生命周期管理
- 数据接口标准化
- 信号生成接口
- 性能监控接口
- 参数管理接口
"""

import abc
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from decimal import Decimal
import pandas as pd
import numpy as np

from .signal import Signal, SignalType, SignalStrength, SignalSource
from ..common.models.market_data import MarketData, KlineData
from ..common.exceptions.strategy_exceptions import StrategyError, StrategyConfigError


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    enabled: bool = True
    
    # 基础参数
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m'])
    lookback_period: int = 100  # 回看周期
    
    # 风险参数
    max_position_size: float = 0.1  # 最大仓位比例
    stop_loss_pct: float = 0.02     # 止损百分比
    take_profit_pct: float = 0.04   # 止盈百分比
    max_drawdown: float = 0.05      # 最大回撤
    
    # 信号参数
    min_confidence: float = 0.6     # 最小置信度
    signal_cooldown: int = 300      # 信号冷却时间（秒）
    
    # 自定义参数
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """验证配置"""
        if not self.name:
            raise StrategyConfigError("策略名称不能为空")
        
        if not self.symbols:
            raise StrategyConfigError("交易对列表不能为空")
        
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise StrategyConfigError("最大仓位比例必须在(0, 1]范围内")
        
        if self.min_confidence < 0 or self.min_confidence > 1:
            raise StrategyConfigError("最小置信度必须在[0, 1]范围内")
        
        return True


@dataclass
class StrategyState:
    """策略状态"""
    is_running: bool = False
    last_signal_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    current_positions: Dict[str, Decimal] = field(default_factory=dict)
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    max_drawdown: float = 0.0
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_signals == 0:
            return 0.0
        return self.successful_signals / self.total_signals
    
    def get_total_pnl(self) -> Decimal:
        """获取总盈亏"""
        return self.unrealized_pnl + self.realized_pnl


class BaseStrategy(abc.ABC):
    """策略基类
    
    所有策略都必须继承此基类并实现抽象方法。
    """
    
    def __init__(self, config: StrategyConfig):
        """初始化策略
        
        Args:
            config: 策略配置
        """
        self.config = config
        self.config.validate()
        
        self.logger = logging.getLogger(f"strategy.{config.name}")
        self.state = StrategyState()
        
        # 数据缓存
        self._market_data_cache: Dict[str, List[MarketData]] = {}
        self._kline_data_cache: Dict[str, pd.DataFrame] = {}
        
        # 信号历史
        self._signal_history: List[Signal] = []
        
        # 性能指标
        self._performance_metrics: Dict[str, Any] = {}
        
        # 初始化策略
        self._initialize()
    
    @abc.abstractmethod
    def _initialize(self) -> None:
        """初始化策略（子类实现）"""
        pass
    
    @abc.abstractmethod
    def generate_signals(self, market_data: Dict[str, MarketData]) -> List[Signal]:
        """生成交易信号（子类实现）
        
        Args:
            market_data: 市场数据字典，key为交易对，value为市场数据
            
        Returns:
            生成的信号列表
        """
        pass
    
    @abc.abstractmethod
    def update_parameters(self, new_params: Dict[str, Any]) -> bool:
        """更新策略参数（子类实现）
        
        Args:
            new_params: 新参数字典
            
        Returns:
            是否更新成功
        """
        pass
    
    def start(self) -> None:
        """启动策略"""
        if self.state.is_running:
            self.logger.warning(f"策略 {self.config.name} 已在运行中")
            return
        
        self.logger.info(f"启动策略: {self.config.name}")
        self.state.is_running = True
        self.state.last_update_time = datetime.now()
        
        self._on_start()
    
    def stop(self) -> None:
        """停止策略"""
        if not self.state.is_running:
            self.logger.warning(f"策略 {self.config.name} 未在运行")
            return
        
        self.logger.info(f"停止策略: {self.config.name}")
        self.state.is_running = False
        
        self._on_stop()
    
    def update_market_data(self, symbol: str, data: MarketData) -> None:
        """更新市场数据
        
        Args:
            symbol: 交易对
            data: 市场数据
        """
        if symbol not in self._market_data_cache:
            self._market_data_cache[symbol] = []
        
        self._market_data_cache[symbol].append(data)
        
        # 保持缓存大小
        max_cache_size = self.config.lookback_period * 2
        if len(self._market_data_cache[symbol]) > max_cache_size:
            self._market_data_cache[symbol] = self._market_data_cache[symbol][-max_cache_size:]
        
        self.state.last_update_time = datetime.now()
    
    def update_kline_data(self, symbol: str, timeframe: str, klines: List[KlineData]) -> None:
        """更新K线数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            klines: K线数据列表
        """
        key = f"{symbol}_{timeframe}"
        
        # 转换为DataFrame
        df_data = []
        for kline in klines:
            df_data.append({
                'timestamp': kline.timestamp,
                'open': float(kline.open_price),
                'high': float(kline.high_price),
                'low': float(kline.low_price),
                'close': float(kline.close_price),
                'volume': float(kline.volume),
                'quote_volume': float(kline.quote_volume),
                'trades': kline.trades_count
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        self._kline_data_cache[key] = df
    
    def get_market_data(self, symbol: str, limit: Optional[int] = None) -> List[MarketData]:
        """获取市场数据
        
        Args:
            symbol: 交易对
            limit: 数据条数限制
            
        Returns:
            市场数据列表
        """
        if symbol not in self._market_data_cache:
            return []
        
        data = self._market_data_cache[symbol]
        if limit:
            return data[-limit:]
        return data
    
    def get_kline_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """获取K线数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            
        Returns:
            K线数据DataFrame
        """
        key = f"{symbol}_{timeframe}"
        return self._kline_data_cache.get(key)
    
    def process_signal(self, signal: Signal) -> bool:
        """处理信号
        
        Args:
            signal: 交易信号
            
        Returns:
            是否处理成功
        """
        try:
            # 检查信号冷却时间
            if self._is_in_cooldown():
                self.logger.debug(f"策略 {self.config.name} 处于冷却期，忽略信号")
                return False
            
            # 验证信号
            if not self._validate_signal(signal):
                self.logger.warning(f"信号验证失败: {signal}")
                return False
            
            # 记录信号
            self._signal_history.append(signal)
            self.state.total_signals += 1
            self.state.last_signal_time = datetime.now()
            
            # 处理信号
            success = self._process_signal_internal(signal)
            
            if success:
                self.state.successful_signals += 1
            else:
                self.state.failed_signals += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"处理信号时发生错误: {e}")
            self.state.failed_signals += 1
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            'strategy_name': self.config.name,
            'is_running': self.state.is_running,
            'total_signals': self.state.total_signals,
            'success_rate': self.state.get_success_rate(),
            'total_pnl': float(self.state.get_total_pnl()),
            'unrealized_pnl': float(self.state.unrealized_pnl),
            'realized_pnl': float(self.state.realized_pnl),
            'max_drawdown': self.state.max_drawdown,
            'current_positions': {k: float(v) for k, v in self.state.current_positions.items()},
            'last_signal_time': self.state.last_signal_time.isoformat() if self.state.last_signal_time else None,
            'last_update_time': self.state.last_update_time.isoformat() if self.state.last_update_time else None,
            **self._performance_metrics
        }
    
    def get_signal_history(self, limit: Optional[int] = None) -> List[Signal]:
        """获取信号历史
        
        Args:
            limit: 返回数量限制
            
        Returns:
            信号历史列表
        """
        if limit:
            return self._signal_history[-limit:]
        return self._signal_history.copy()
    
    def reset_state(self) -> None:
        """重置策略状态"""
        self.logger.info(f"重置策略状态: {self.config.name}")
        self.state = StrategyState()
        self._signal_history.clear()
        self._market_data_cache.clear()
        self._kline_data_cache.clear()
        self._performance_metrics.clear()
    
    def _is_in_cooldown(self) -> bool:
        """检查是否在冷却期"""
        if not self.state.last_signal_time:
            return False
        
        cooldown_end = self.state.last_signal_time + timedelta(seconds=self.config.signal_cooldown)
        return datetime.now() < cooldown_end
    
    def _validate_signal(self, signal: Signal) -> bool:
        """验证信号
        
        Args:
            signal: 交易信号
            
        Returns:
            是否有效
        """
        # 基础验证
        if not signal.is_valid():
            return False
        
        # 置信度检查
        if signal.confidence < self.config.min_confidence:
            return False
        
        # 交易对检查
        if signal.symbol not in self.config.symbols:
            return False
        
        # 策略特定验证
        return self._validate_signal_custom(signal)
    
    def _validate_signal_custom(self, signal: Signal) -> bool:
        """自定义信号验证（子类可重写）
        
        Args:
            signal: 交易信号
            
        Returns:
            是否有效
        """
        return True
    
    def _process_signal_internal(self, signal: Signal) -> bool:
        """内部信号处理（子类可重写）
        
        Args:
            signal: 交易信号
            
        Returns:
            是否处理成功
        """
        return True
    
    def _on_start(self) -> None:
        """启动时回调（子类可重写）"""
        pass
    
    def _on_stop(self) -> None:
        """停止时回调（子类可重写）"""
        pass
    
    def _calculate_position_size(self, signal: Signal, account_balance: Decimal) -> Decimal:
        """计算仓位大小
        
        Args:
            signal: 交易信号
            account_balance: 账户余额
            
        Returns:
            仓位大小
        """
        if signal.percentage:
            return account_balance * Decimal(str(signal.percentage))
        
        max_position_value = account_balance * Decimal(str(self.config.max_position_size))
        
        if signal.quantity:
            position_value = signal.quantity * (signal.price or Decimal('0'))
            return min(position_value, max_position_value)
        
        return max_position_value
    
    def _update_performance_metrics(self, metric_name: str, value: Any) -> None:
        """更新性能指标
        
        Args:
            metric_name: 指标名称
            value: 指标值
        """
        self._performance_metrics[metric_name] = value
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Strategy({self.config.name}, running={self.state.is_running})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()