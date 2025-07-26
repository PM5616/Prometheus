"""Portfolio Manager Models

投资组合管理模块的数据模型定义
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class SignalStrength(Enum):
    """信号强度"""
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class PositionType(Enum):
    """持仓类型"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class PositionStatus(Enum):
    """持仓状态"""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    PENDING = "pending"


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Signal:
    """交易信号"""
    id: UUID = field(default_factory=uuid4)
    strategy_id: str = ""
    symbol: str = ""
    signal_type: SignalType = SignalType.HOLD
    strength: SignalStrength = SignalStrength.MEDIUM
    confidence: float = 0.0  # 0-1之间
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    position_size: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=datetime.now)
    expiry: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查信号是否过期"""
        if self.expiry is None:
            return False
        return datetime.now() > self.expiry
    
    def get_weight(self) -> float:
        """获取信号权重"""
        strength_weights = {
            SignalStrength.WEAK: 0.25,
            SignalStrength.MEDIUM: 0.5,
            SignalStrength.STRONG: 0.75,
            SignalStrength.VERY_STRONG: 1.0
        }
        return strength_weights[self.strength] * self.confidence


@dataclass
class Position:
    """持仓信息"""
    id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    position_type: PositionType = PositionType.NEUTRAL
    status: PositionStatus = PositionStatus.PENDING
    quantity: Decimal = Decimal('0')
    entry_price: Optional[Decimal] = None
    current_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    strategy_id: str = ""
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    commission: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_price(self, current_price: Decimal) -> None:
        """更新当前价格和未实现盈亏"""
        self.current_price = current_price
        if self.entry_price is not None:
            if self.position_type == PositionType.LONG:
                self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            elif self.position_type == PositionType.SHORT:
                self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def get_market_value(self) -> Decimal:
        """获取市值"""
        if self.current_price is None:
            return Decimal('0')
        return abs(self.quantity) * self.current_price
    
    def get_total_pnl(self) -> Decimal:
        """获取总盈亏"""
        return self.realized_pnl + self.unrealized_pnl - self.commission


@dataclass
class Portfolio:
    """投资组合"""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    base_currency: str = "USDT"
    total_value: Decimal = Decimal('0')
    available_cash: Decimal = Decimal('0')
    positions: Dict[str, Position] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_position(self, position: Position) -> None:
        """添加持仓"""
        self.positions[position.symbol] = position
        self.updated_at = datetime.now()
    
    def remove_position(self, symbol: str) -> Optional[Position]:
        """移除持仓"""
        position = self.positions.pop(symbol, None)
        if position:
            self.updated_at = datetime.now()
        return position
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(symbol)
    
    def get_total_market_value(self) -> Decimal:
        """获取总市值"""
        return sum(pos.get_market_value() for pos in self.positions.values())
    
    def get_total_pnl(self) -> Decimal:
        """获取总盈亏"""
        return sum(pos.get_total_pnl() for pos in self.positions.values())
    
    def get_exposure(self) -> Decimal:
        """获取总敞口"""
        return sum(abs(pos.quantity * (pos.current_price or Decimal('0'))) 
                  for pos in self.positions.values())


@dataclass
class RiskMetrics:
    """风险指标"""
    portfolio_id: UUID
    timestamp: datetime = field(default_factory=datetime.now)
    total_exposure: Decimal = Decimal('0')
    max_drawdown: Decimal = Decimal('0')
    var_95: Decimal = Decimal('0')  # 95% VaR
    var_99: Decimal = Decimal('0')  # 99% VaR
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    volatility: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    concentration_risk: float = 0.0  # 集中度风险
    leverage: float = 0.0
    margin_usage: float = 0.0
    
    def is_risk_acceptable(self, max_var: Decimal, max_drawdown: Decimal) -> bool:
        """检查风险是否可接受"""
        return (self.var_95 <= max_var and 
                self.max_drawdown <= max_drawdown)


@dataclass
class PerformanceMetrics:
    """绩效指标"""
    portfolio_id: UUID
    period_start: datetime
    period_end: datetime
    total_return: Decimal = Decimal('0')
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: Decimal = Decimal('0')
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: Decimal = Decimal('0')
    average_loss: Decimal = Decimal('0')
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    
    def calculate_metrics(self) -> None:
        """计算绩效指标"""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        if self.losing_trades > 0 and self.average_loss != 0:
            self.profit_factor = float(self.average_win * self.winning_trades / 
                                     abs(self.average_loss * self.losing_trades))


@dataclass
class OptimizationConfig:
    """优化配置"""
    method: str = "mean_variance"  # mean_variance, black_litterman, risk_parity
    objective: str = "max_sharpe"  # max_sharpe, min_variance, max_return
    constraints: Dict[str, Any] = field(default_factory=dict)
    risk_aversion: float = 1.0
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    max_weight: float = 0.3  # 单个资产最大权重
    min_weight: float = 0.0  # 单个资产最小权重
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    lookback_period: int = 252  # 回望期（交易日）
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if self.max_weight <= self.min_weight:
            return False
        if self.max_weight > 1.0 or self.min_weight < 0.0:
            return False
        return True


@dataclass
class RiskConfig:
    """风险配置"""
    max_portfolio_var: Decimal = Decimal('0.05')  # 最大组合VaR
    max_drawdown: Decimal = Decimal('0.15')  # 最大回撤
    max_leverage: float = 2.0  # 最大杠杆
    max_concentration: float = 0.3  # 最大集中度
    max_correlation: float = 0.8  # 最大相关性
    stop_loss_threshold: Decimal = Decimal('0.02')  # 止损阈值
    position_size_limit: Decimal = Decimal('0.1')  # 单笔仓位限制
    daily_loss_limit: Decimal = Decimal('0.03')  # 日损失限制
    margin_call_threshold: float = 0.8  # 保证金预警阈值
    force_liquidation_threshold: float = 0.9  # 强制平仓阈值
    
    def validate(self) -> bool:
        """验证风险配置"""
        return (self.max_drawdown > 0 and 
                self.max_leverage > 0 and
                self.max_concentration > 0 and
                self.max_concentration <= 1.0)