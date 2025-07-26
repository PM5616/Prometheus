"""Strategy Data Models

策略相关的数据模型，包括信号、策略配置等。
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import Field, validator

from .base import BaseModel, TimestampMixin, IdentifiableMixin
from .trading import OrderSide


from .enums import SignalType, SignalStrength


class Signal(TimestampMixin, IdentifiableMixin, BaseModel):
    """交易信号模型"""
    
    # 基本信息
    symbol: str = Field(..., description="交易对")
    signal_type: SignalType = Field(..., description="信号类型")
    strength: SignalStrength = Field(SignalStrength.MODERATE, description="信号强度")
    
    # 价格信息
    price: Decimal = Field(..., description="信号价格")
    target_price: Optional[Decimal] = Field(None, description="目标价格")
    stop_loss_price: Optional[Decimal] = Field(None, description="止损价格")
    
    # 数量信息
    quantity: Optional[Decimal] = Field(None, description="建议数量")
    position_size: Optional[Decimal] = Field(None, description="建议仓位大小")
    
    # 策略信息
    strategy_id: str = Field(..., description="策略ID")
    strategy_name: str = Field(..., description="策略名称")
    
    # 信号来源
    source: str = Field(..., description="信号来源")
    indicators: Dict[str, Any] = Field(default_factory=dict, description="技术指标值")
    
    # 置信度
    confidence: float = Field(..., description="信号置信度", ge=0.0, le=1.0)
    
    # 有效期
    expires_at: Optional[datetime] = Field(None, description="信号过期时间")
    
    # 状态
    is_active: bool = Field(True, description="是否活跃")
    is_executed: bool = Field(False, description="是否已执行")
    
    # 执行信息
    executed_at: Optional[datetime] = Field(None, description="执行时间")
    executed_price: Optional[Decimal] = Field(None, description="执行价格")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """验证置信度"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v
    
    def is_expired(self) -> bool:
        """信号是否已过期
        
        Returns:
            bool: 是否过期
        """
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """信号是否有效
        
        Returns:
            bool: 是否有效
        """
        return self.is_active and not self.is_expired() and not self.is_executed
    
    def execute(self, price: Decimal) -> None:
        """标记信号为已执行
        
        Args:
            price: 执行价格
        """
        self.is_executed = True
        self.executed_at = datetime.utcnow()
        self.executed_price = price
    
    def get_order_side(self) -> Optional[OrderSide]:
        """获取对应的订单方向
        
        Returns:
            Optional[OrderSide]: 订单方向
        """
        if self.signal_type in [SignalType.BUY]:
            return OrderSide.BUY
        elif self.signal_type in [SignalType.SELL, SignalType.CLOSE, SignalType.STOP_LOSS, SignalType.TAKE_PROFIT]:
            return OrderSide.SELL
        return None


class StrategyConfig(BaseModel):
    """策略配置模型"""
    
    # 基本信息
    name: str = Field(..., description="策略名称")
    description: str = Field("", description="策略描述")
    version: str = Field("1.0.0", description="策略版本")
    
    # 交易参数
    symbols: List[str] = Field(..., description="支持的交易对")
    timeframes: List[str] = Field(..., description="支持的时间周期")
    
    # 风险参数
    max_position_size: Decimal = Field(..., description="最大仓位大小")
    stop_loss_percent: Decimal = Field(..., description="止损百分比")
    take_profit_percent: Decimal = Field(..., description="止盈百分比")
    
    # 技术指标参数
    indicators: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="技术指标参数")
    
    # 信号过滤参数
    min_confidence: float = Field(0.6, description="最小信号置信度")
    signal_cooldown: int = Field(300, description="信号冷却时间（秒）")
    
    # 其他参数
    parameters: Dict[str, Any] = Field(default_factory=dict, description="其他策略参数")
    
    @validator('min_confidence')
    def validate_min_confidence(cls, v):
        """验证最小置信度"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Min confidence must be between 0.0 and 1.0')
        return v


from .enums import StrategyStatus


class Strategy(TimestampMixin, IdentifiableMixin, BaseModel):
    """策略模型"""
    
    # 基本信息
    name: str = Field(..., description="策略名称")
    description: str = Field("", description="策略描述")
    version: str = Field("1.0.0", description="策略版本")
    
    # 配置
    config: StrategyConfig = Field(..., description="策略配置")
    
    # 状态
    status: StrategyStatus = Field(StrategyStatus.INACTIVE, description="策略状态")
    
    # 性能统计
    total_signals: int = Field(0, description="总信号数")
    successful_signals: int = Field(0, description="成功信号数")
    total_trades: int = Field(0, description="总交易数")
    winning_trades: int = Field(0, description="盈利交易数")
    
    # 盈亏统计
    total_pnl: Decimal = Field(Decimal('0'), description="总盈亏")
    max_drawdown: Decimal = Field(Decimal('0'), description="最大回撤")
    
    # 时间信息
    started_at: Optional[datetime] = Field(None, description="启动时间")
    stopped_at: Optional[datetime] = Field(None, description="停止时间")
    last_signal_at: Optional[datetime] = Field(None, description="最后信号时间")
    
    # 错误信息
    last_error: Optional[str] = Field(None, description="最后错误信息")
    error_count: int = Field(0, description="错误次数")
    
    @property
    def success_rate(self) -> float:
        """信号成功率
        
        Returns:
            float: 成功率
        """
        if self.total_signals == 0:
            return 0.0
        return self.successful_signals / self.total_signals
    
    @property
    def win_rate(self) -> float:
        """交易胜率
        
        Returns:
            float: 胜率
        """
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def avg_pnl_per_trade(self) -> Decimal:
        """平均每笔交易盈亏
        
        Returns:
            Decimal: 平均盈亏
        """
        if self.total_trades == 0:
            return Decimal('0')
        return self.total_pnl / self.total_trades
    
    def is_active(self) -> bool:
        """策略是否活跃
        
        Returns:
            bool: 是否活跃
        """
        return self.status == StrategyStatus.ACTIVE
    
    def start(self) -> None:
        """启动策略"""
        self.status = StrategyStatus.ACTIVE
        self.started_at = datetime.utcnow()
        self.stopped_at = None
    
    def stop(self) -> None:
        """停止策略"""
        self.status = StrategyStatus.STOPPED
        self.stopped_at = datetime.utcnow()
    
    def pause(self) -> None:
        """暂停策略"""
        self.status = StrategyStatus.PAUSED
    
    def resume(self) -> None:
        """恢复策略"""
        if self.status == StrategyStatus.PAUSED:
            self.status = StrategyStatus.ACTIVE
    
    def record_signal(self, successful: bool = True) -> None:
        """记录信号
        
        Args:
            successful: 是否成功
        """
        self.total_signals += 1
        if successful:
            self.successful_signals += 1
        self.last_signal_at = datetime.utcnow()
    
    def record_trade(self, pnl: Decimal, winning: bool = True) -> None:
        """记录交易
        
        Args:
            pnl: 盈亏
            winning: 是否盈利
        """
        self.total_trades += 1
        if winning:
            self.winning_trades += 1
        self.total_pnl += pnl
        
        # 更新最大回撤
        if pnl < 0 and abs(pnl) > self.max_drawdown:
            self.max_drawdown = abs(pnl)
    
    def record_error(self, error_message: str) -> None:
        """记录错误
        
        Args:
            error_message: 错误信息
        """
        self.error_count += 1
        self.last_error = error_message
        self.status = StrategyStatus.ERROR
        self.update_timestamp()