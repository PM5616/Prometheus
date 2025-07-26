"""Risk Management Data Models

风险管理相关的数据模型，包括风险限制、风险指标、风险告警等。
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import Field, validator

from .base import BaseModel, TimestampMixin, IdentifiableMixin
from .enums import AlertLevel, AlertType


class RiskLimitType(str, Enum):
    """风险限制类型"""
    POSITION_SIZE = "POSITION_SIZE"  # 仓位大小限制
    DAILY_LOSS = "DAILY_LOSS"  # 日损失限制
    TOTAL_EXPOSURE = "TOTAL_EXPOSURE"  # 总敞口限制
    DRAWDOWN = "DRAWDOWN"  # 回撤限制
    LEVERAGE = "LEVERAGE"  # 杠杆限制
    CONCENTRATION = "CONCENTRATION"  # 集中度限制


class RiskLimitStatus(str, Enum):
    """风险限制状态"""
    NORMAL = "NORMAL"  # 正常
    WARNING = "WARNING"  # 警告
    BREACH = "BREACH"  # 违规
    CRITICAL = "CRITICAL"  # 严重


class RiskLimit(TimestampMixin, IdentifiableMixin, BaseModel):
    """风险限制模型"""
    
    # 基本信息
    name: str = Field(..., description="限制名称")
    type: RiskLimitType = Field(..., description="限制类型")
    description: str = Field("", description="限制描述")
    
    # 限制参数
    limit_value: Decimal = Field(..., description="限制值")
    warning_threshold: Decimal = Field(..., description="警告阈值")
    
    # 适用范围
    symbol: Optional[str] = Field(None, description="适用交易对")
    strategy_id: Optional[str] = Field(None, description="适用策略")
    
    # 状态
    is_active: bool = Field(True, description="是否启用")
    status: RiskLimitStatus = Field(RiskLimitStatus.NORMAL, description="当前状态")
    
    # 当前值
    current_value: Decimal = Field(Decimal('0'), description="当前值")
    
    # 违规信息
    breach_count: int = Field(0, description="违规次数")
    last_breach_at: Optional[datetime] = Field(None, description="最后违规时间")
    
    @validator('warning_threshold')
    def validate_warning_threshold(cls, v, values):
        """验证警告阈值"""
        if 'limit_value' in values and v > values['limit_value']:
            raise ValueError('Warning threshold cannot exceed limit value')
        return v
    
    def check_limit(self, value: Decimal) -> RiskLimitStatus:
        """检查风险限制
        
        Args:
            value: 当前值
            
        Returns:
            RiskLimitStatus: 风险状态
        """
        self.current_value = value
        
        if value >= self.limit_value:
            self.status = RiskLimitStatus.BREACH
            self.breach_count += 1
            self.last_breach_at = datetime.utcnow()
        elif value >= self.warning_threshold:
            self.status = RiskLimitStatus.WARNING
        else:
            self.status = RiskLimitStatus.NORMAL
        
        self.update_timestamp()
        return self.status
    
    def get_utilization_rate(self) -> float:
        """获取利用率
        
        Returns:
            float: 利用率（0-1）
        """
        if self.limit_value == 0:
            return 0.0
        return float(self.current_value / self.limit_value)
    
    def is_breached(self) -> bool:
        """是否违规
        
        Returns:
            bool: 是否违规
        """
        return self.status in [RiskLimitStatus.BREACH, RiskLimitStatus.CRITICAL]


class RiskMetrics(TimestampMixin, BaseModel):
    """风险指标模型"""
    
    # 基本信息
    symbol: Optional[str] = Field(None, description="交易对")
    strategy_id: Optional[str] = Field(None, description="策略ID")
    
    # 仓位风险
    total_position_value: Decimal = Field(Decimal('0'), description="总仓位价值")
    max_position_value: Decimal = Field(Decimal('0'), description="最大仓位价值")
    position_concentration: Decimal = Field(Decimal('0'), description="仓位集中度")
    
    # 盈亏风险
    unrealized_pnl: Decimal = Field(Decimal('0'), description="未实现盈亏")
    realized_pnl: Decimal = Field(Decimal('0'), description="已实现盈亏")
    daily_pnl: Decimal = Field(Decimal('0'), description="日盈亏")
    
    # 回撤风险
    current_drawdown: Decimal = Field(Decimal('0'), description="当前回撤")
    max_drawdown: Decimal = Field(Decimal('0'), description="最大回撤")
    
    # 波动率风险
    volatility: Decimal = Field(Decimal('0'), description="波动率")
    var_95: Decimal = Field(Decimal('0'), description="95% VaR")
    var_99: Decimal = Field(Decimal('0'), description="99% VaR")
    
    # 流动性风险
    liquidity_score: Decimal = Field(Decimal('0'), description="流动性评分")
    
    # 杠杆风险
    leverage_ratio: Decimal = Field(Decimal('1'), description="杠杆比率")
    margin_ratio: Decimal = Field(Decimal('0'), description="保证金比率")
    
    # 相关性风险
    correlation_risk: Decimal = Field(Decimal('0'), description="相关性风险")
    
    @property
    def total_pnl(self) -> Decimal:
        """总盈亏
        
        Returns:
            Decimal: 总盈亏
        """
        return self.unrealized_pnl + self.realized_pnl
    
    @property
    def risk_score(self) -> Decimal:
        """综合风险评分
        
        Returns:
            Decimal: 风险评分（0-100）
        """
        # 简单的风险评分计算
        score = Decimal('0')
        
        # 回撤风险权重 30%
        if self.max_drawdown > 0:
            drawdown_score = min(self.current_drawdown / self.max_drawdown * 30, 30)
            score += drawdown_score
        
        # 仓位集中度权重 25%
        concentration_score = min(self.position_concentration * 25, 25)
        score += concentration_score
        
        # 杠杆风险权重 20%
        leverage_score = min((self.leverage_ratio - 1) * 10, 20)
        score += leverage_score
        
        # 波动率风险权重 15%
        volatility_score = min(self.volatility * 15, 15)
        score += volatility_score
        
        # 流动性风险权重 10%
        liquidity_score = max(10 - self.liquidity_score, 0)
        score += liquidity_score
        
        return min(score, 100)


class RiskAlert(TimestampMixin, IdentifiableMixin, BaseModel):
    """风险告警模型"""
    
    # 基本信息
    title: str = Field(..., description="告警标题")
    message: str = Field(..., description="告警消息")
    level: AlertLevel = Field(..., description="告警级别")
    type: AlertType = Field(..., description="告警类型")
    
    # 关联信息
    symbol: Optional[str] = Field(None, description="相关交易对")
    strategy_id: Optional[str] = Field(None, description="相关策略")
    risk_limit_id: Optional[str] = Field(None, description="相关风险限制")
    
    # 告警数据
    current_value: Optional[Decimal] = Field(None, description="当前值")
    threshold_value: Optional[Decimal] = Field(None, description="阈值")
    
    # 状态
    is_active: bool = Field(True, description="是否活跃")
    is_acknowledged: bool = Field(False, description="是否已确认")
    is_resolved: bool = Field(False, description="是否已解决")
    
    # 处理信息
    acknowledged_at: Optional[datetime] = Field(None, description="确认时间")
    acknowledged_by: Optional[str] = Field(None, description="确认人")
    resolved_at: Optional[datetime] = Field(None, description="解决时间")
    resolved_by: Optional[str] = Field(None, description="解决人")
    
    # 附加数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加数据")
    
    def acknowledge(self, user: str) -> None:
        """确认告警
        
        Args:
            user: 确认用户
        """
        self.is_acknowledged = True
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = user
        self.update_timestamp()
    
    def resolve(self, user: str) -> None:
        """解决告警
        
        Args:
            user: 解决用户
        """
        self.is_resolved = True
        self.resolved_at = datetime.utcnow()
        self.resolved_by = user
        self.is_active = False
        self.update_timestamp()
    
    def is_critical(self) -> bool:
        """是否为严重告警
        
        Returns:
            bool: 是否严重
        """
        return self.level == AlertLevel.CRITICAL
    
    def get_duration(self) -> Optional[int]:
        """获取告警持续时间（秒）
        
        Returns:
            Optional[int]: 持续时间
        """
        if self.resolved_at:
            return int((self.resolved_at - self.created_at).total_seconds())
        return int((datetime.utcnow() - self.created_at).total_seconds())


class RiskReport(TimestampMixin, BaseModel):
    """风险报告模型"""
    
    # 报告信息
    report_date: datetime = Field(default_factory=datetime.utcnow, description="报告日期")
    report_type: str = Field("daily", description="报告类型")
    
    # 风险指标
    metrics: RiskMetrics = Field(..., description="风险指标")
    
    # 风险限制状态
    limit_statuses: List[RiskLimit] = Field(default_factory=list, description="风险限制状态")
    
    # 活跃告警
    active_alerts: List[RiskAlert] = Field(default_factory=list, description="活跃告警")
    
    # 风险评估
    overall_risk_score: Decimal = Field(Decimal('0'), description="整体风险评分")
    risk_level: str = Field("LOW", description="风险等级")
    
    # 建议
    recommendations: List[str] = Field(default_factory=list, description="风险建议")
    
    def get_risk_level(self) -> str:
        """获取风险等级
        
        Returns:
            str: 风险等级
        """
        score = float(self.overall_risk_score)
        
        if score >= 80:
            return "CRITICAL"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        elif score >= 20:
            return "LOW"
        else:
            return "MINIMAL"
    
    def add_recommendation(self, recommendation: str) -> None:
        """添加风险建议
        
        Args:
            recommendation: 建议内容
        """
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)