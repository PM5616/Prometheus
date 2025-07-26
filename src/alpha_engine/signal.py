"""Signal Module

交易信号模块，定义交易信号的数据结构、类型和强度。

主要功能：
- 信号类型定义
- 信号强度分级
- 信号数据结构
- 信号验证和序列化
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from decimal import Decimal


class SignalType(Enum):
    """信号类型枚举"""
    BUY = "buy"                    # 买入信号
    SELL = "sell"                  # 卖出信号
    HOLD = "hold"                  # 持有信号
    CLOSE_LONG = "close_long"      # 平多信号
    CLOSE_SHORT = "close_short"    # 平空信号
    REDUCE_LONG = "reduce_long"    # 减多信号
    REDUCE_SHORT = "reduce_short"  # 减空信号
    STOP_LOSS = "stop_loss"        # 止损信号
    TAKE_PROFIT = "take_profit"    # 止盈信号


class SignalStrength(Enum):
    """信号强度枚举"""
    WEAK = 1        # 弱信号
    MODERATE = 2    # 中等信号
    STRONG = 3      # 强信号
    VERY_STRONG = 4 # 极强信号


class SignalSource(Enum):
    """信号来源枚举"""
    MEAN_REVERSION = "mean_reversion"      # 震荡套利
    TREND_FOLLOWING = "trend_following"    # 趋势跟踪
    REGIME_DETECTION = "regime_detection"  # 市场状态识别
    COINTEGRATION = "cointegration"        # 协整配对
    KALMAN_FILTER = "kalman_filter"        # 卡尔曼滤波
    HMM = "hmm"                            # 隐马尔可夫模型
    TECHNICAL = "technical"                # 技术分析
    FUNDAMENTAL = "fundamental"            # 基本面分析
    SENTIMENT = "sentiment"                # 情绪分析
    ARBITRAGE = "arbitrage"                # 套利
    MANUAL = "manual"                      # 手动信号


@dataclass
class Signal:
    """交易信号数据结构"""
    
    # 基本信息
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    signal_type: SignalType = SignalType.HOLD
    strength: SignalStrength = SignalStrength.MODERATE
    source: SignalSource = SignalSource.TECHNICAL
    
    # 价格信息
    price: Optional[Decimal] = None
    target_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    
    # 数量信息
    quantity: Optional[Decimal] = None
    percentage: Optional[float] = None  # 仓位百分比
    
    # 策略信息
    strategy_name: str = ""
    strategy_version: str = "1.0.0"
    confidence: float = 0.5  # 信号置信度 [0, 1]
    
    # 风险信息
    risk_level: float = 0.5  # 风险等级 [0, 1]
    max_loss: Optional[Decimal] = None
    expected_return: Optional[Decimal] = None
    
    # 时间信息
    valid_until: Optional[datetime] = None
    execution_delay: int = 0  # 执行延迟（秒）
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: list = field(default_factory=list)
    
    # 状态信息
    is_active: bool = True
    is_executed: bool = False
    execution_time: Optional[datetime] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.price, (int, float)):
            self.price = Decimal(str(self.price))
        if isinstance(self.target_price, (int, float)):
            self.target_price = Decimal(str(self.target_price))
        if isinstance(self.stop_loss_price, (int, float)):
            self.stop_loss_price = Decimal(str(self.stop_loss_price))
        if isinstance(self.take_profit_price, (int, float)):
            self.take_profit_price = Decimal(str(self.take_profit_price))
        if isinstance(self.quantity, (int, float)):
            self.quantity = Decimal(str(self.quantity))
    
    def is_valid(self) -> bool:
        """检查信号是否有效"""
        if not self.is_active:
            return False
        
        if self.valid_until and datetime.now() > self.valid_until:
            return False
        
        if not self.symbol:
            return False
        
        if self.confidence < 0 or self.confidence > 1:
            return False
        
        if self.risk_level < 0 or self.risk_level > 1:
            return False
        
        return True
    
    def is_buy_signal(self) -> bool:
        """是否为买入信号"""
        return self.signal_type in [SignalType.BUY]
    
    def is_sell_signal(self) -> bool:
        """是否为卖出信号"""
        return self.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]
    
    def is_close_signal(self) -> bool:
        """是否为平仓信号"""
        return self.signal_type in [
            SignalType.CLOSE_LONG, 
            SignalType.CLOSE_SHORT,
            SignalType.STOP_LOSS,
            SignalType.TAKE_PROFIT
        ]
    
    def is_risk_signal(self) -> bool:
        """是否为风险信号"""
        return self.signal_type in [SignalType.STOP_LOSS, SignalType.TAKE_PROFIT]
    
    def get_direction(self) -> int:
        """获取信号方向
        
        Returns:
            1: 多头方向
            -1: 空头方向
            0: 中性或平仓
        """
        if self.signal_type in [SignalType.BUY]:
            return 1
        elif self.signal_type in [SignalType.SELL]:
            return -1
        else:
            return 0
    
    def get_urgency_score(self) -> float:
        """获取信号紧急度评分
        
        Returns:
            紧急度评分 [0, 1]
        """
        base_score = self.strength.value / 4.0  # 基础评分
        confidence_bonus = self.confidence * 0.3  # 置信度加成
        risk_penalty = self.risk_level * 0.2  # 风险惩罚
        
        # 风险信号具有更高优先级
        if self.is_risk_signal():
            base_score += 0.3
        
        return min(1.0, max(0.0, base_score + confidence_bonus - risk_penalty))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'source': self.source.value,
            'price': str(self.price) if self.price else None,
            'target_price': str(self.target_price) if self.target_price else None,
            'stop_loss_price': str(self.stop_loss_price) if self.stop_loss_price else None,
            'take_profit_price': str(self.take_profit_price) if self.take_profit_price else None,
            'quantity': str(self.quantity) if self.quantity else None,
            'percentage': self.percentage,
            'strategy_name': self.strategy_name,
            'strategy_version': self.strategy_version,
            'confidence': self.confidence,
            'risk_level': self.risk_level,
            'max_loss': str(self.max_loss) if self.max_loss else None,
            'expected_return': str(self.expected_return) if self.expected_return else None,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'execution_delay': self.execution_delay,
            'metadata': self.metadata,
            'tags': self.tags,
            'is_active': self.is_active,
            'is_executed': self.is_executed,
            'execution_time': self.execution_time.isoformat() if self.execution_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """从字典创建信号"""
        # 处理枚举类型
        if 'signal_type' in data:
            data['signal_type'] = SignalType(data['signal_type'])
        if 'strength' in data:
            data['strength'] = SignalStrength(data['strength'])
        if 'source' in data:
            data['source'] = SignalSource(data['source'])
        
        # 处理时间类型
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'valid_until' in data and isinstance(data['valid_until'], str):
            data['valid_until'] = datetime.fromisoformat(data['valid_until'])
        if 'execution_time' in data and isinstance(data['execution_time'], str):
            data['execution_time'] = datetime.fromisoformat(data['execution_time'])
        
        # 处理Decimal类型
        decimal_fields = ['price', 'target_price', 'stop_loss_price', 'take_profit_price', 
                         'quantity', 'max_loss', 'expected_return']
        for field in decimal_fields:
            if field in data and data[field] is not None:
                data[field] = Decimal(str(data[field]))
        
        return cls(**data)
    
    def copy(self) -> 'Signal':
        """复制信号"""
        return Signal.from_dict(self.to_dict())
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"Signal({self.signal_type.value}, {self.symbol}, "
                f"strength={self.strength.value}, confidence={self.confidence:.2f})")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()