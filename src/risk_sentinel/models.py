"""Risk Sentinel Models

风险哨兵模块的数据模型定义
"""

import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from ..common.models import RiskLevel, AlertType


class RiskType(Enum):
    """风险类型"""
    MARKET_RISK = "market_risk"  # 市场风险
    CREDIT_RISK = "credit_risk"  # 信用风险
    LIQUIDITY_RISK = "liquidity_risk"  # 流动性风险
    OPERATIONAL_RISK = "operational_risk"  # 操作风险
    CONCENTRATION_RISK = "concentration_risk"  # 集中度风险
    LEVERAGE_RISK = "leverage_risk"  # 杠杆风险
    VOLATILITY_RISK = "volatility_risk"  # 波动率风险
    DRAWDOWN_RISK = "drawdown_risk"  # 回撤风险
    VAR_RISK = "var_risk"  # VaR风险
    COMPLIANCE_RISK = "compliance_risk"  # 合规风险



from ..common.models import AlertStatus


class ControlAction(Enum):
    """控制动作"""
    NONE = "none"  # 无动作
    WARNING = "warning"  # 警告
    REDUCE_POSITION = "reduce_position"  # 减仓
    CLOSE_POSITION = "close_position"  # 平仓
    STOP_TRADING = "stop_trading"  # 停止交易
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止
    INCREASE_MARGIN = "increase_margin"  # 增加保证金
    LIMIT_EXPOSURE = "limit_exposure"  # 限制敞口


@dataclass
class RiskMetric:
    """风险指标"""
    name: str
    value: float
    threshold: float
    risk_type: RiskType
    risk_level: RiskLevel
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    unit: str = ""
    confidence: float = 1.0  # 置信度
    
    @property
    def is_breach(self) -> bool:
        """是否突破阈值"""
        return abs(self.value) > abs(self.threshold)
    
    @property
    def breach_ratio(self) -> float:
        """突破比例"""
        if self.threshold == 0:
            return 0.0
        return self.value / self.threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'threshold': self.threshold,
            'risk_type': self.risk_type.value,
            'risk_level': self.risk_level.value,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'unit': self.unit,
            'confidence': self.confidence,
            'is_breach': self.is_breach,
            'breach_ratio': self.breach_ratio
        }


@dataclass
class RiskAlert:
    """风险预警"""
    alert_id: str
    alert_type: AlertType
    risk_type: RiskType
    risk_level: RiskLevel
    title: str
    message: str
    status: AlertStatus = AlertStatus.ACTIVE
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    affected_symbols: List[str] = field(default_factory=list)
    metrics: List[RiskMetric] = field(default_factory=list)
    recommended_actions: List[ControlAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    def acknowledge(self, user: str) -> None:
        """确认预警"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = user
        self.acknowledged_at = datetime.now()
    
    def resolve(self) -> None:
        """解决预警"""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()
    
    def dismiss(self) -> None:
        """忽略预警"""
        self.status = AlertStatus.DISMISSED
        self.resolved_at = datetime.now()
    
    def escalate(self) -> None:
        """升级预警"""
        self.status = AlertStatus.ESCALATED
        if self.risk_level != RiskLevel.EMERGENCY:
            # 升级风险等级
            levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.EMERGENCY]
            current_index = levels.index(self.risk_level)
            if current_index < len(levels) - 1:
                self.risk_level = levels[current_index + 1]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'risk_type': self.risk_type.value,
            'risk_level': self.risk_level.value,
            'title': self.title,
            'message': self.message,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'affected_symbols': self.affected_symbols,
            'metrics': [metric.to_dict() for metric in self.metrics],
            'recommended_actions': [action.value for action in self.recommended_actions],
            'metadata': self.metadata,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class RiskLimit:
    """风险限制"""
    name: str
    risk_type: RiskType
    limit_value: float
    current_value: float = 0.0
    warning_threshold: float = 0.8  # 警告阈值（限制的80%）
    critical_threshold: float = 0.95  # 严重阈值（限制的95%）
    enabled: bool = True
    description: str = ""
    unit: str = ""
    applicable_symbols: List[str] = field(default_factory=list)
    applicable_strategies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def utilization_ratio(self) -> float:
        """使用率"""
        if self.limit_value == 0:
            return 0.0
        return abs(self.current_value) / abs(self.limit_value)
    
    @property
    def remaining_capacity(self) -> float:
        """剩余容量"""
        return self.limit_value - abs(self.current_value)
    
    @property
    def is_warning(self) -> bool:
        """是否达到警告阈值"""
        return self.utilization_ratio >= self.warning_threshold
    
    @property
    def is_critical(self) -> bool:
        """是否达到严重阈值"""
        return self.utilization_ratio >= self.critical_threshold
    
    @property
    def is_breached(self) -> bool:
        """是否突破限制"""
        return abs(self.current_value) > abs(self.limit_value)
    
    def update_current_value(self, value: float) -> None:
        """更新当前值"""
        self.current_value = value
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'risk_type': self.risk_type.value,
            'limit_value': self.limit_value,
            'current_value': self.current_value,
            'warning_threshold': self.warning_threshold,
            'critical_threshold': self.critical_threshold,
            'enabled': self.enabled,
            'description': self.description,
            'unit': self.unit,
            'applicable_symbols': self.applicable_symbols,
            'applicable_strategies': self.applicable_strategies,
            'utilization_ratio': self.utilization_ratio,
            'remaining_capacity': self.remaining_capacity,
            'is_warning': self.is_warning,
            'is_critical': self.is_critical,
            'is_breached': self.is_breached,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class RiskEvent:
    """风险事件"""
    event_id: str
    event_type: str
    risk_type: RiskType
    risk_level: RiskLevel
    title: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    affected_symbols: List[str] = field(default_factory=list)
    affected_strategies: List[str] = field(default_factory=list)
    impact_assessment: str = ""
    mitigation_actions: List[str] = field(default_factory=list)
    status: str = "open"  # open, investigating, resolved, closed
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'risk_type': self.risk_type.value,
            'risk_level': self.risk_level.value,
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'affected_symbols': self.affected_symbols,
            'affected_strategies': self.affected_strategies,
            'impact_assessment': self.impact_assessment,
            'mitigation_actions': self.mitigation_actions,
            'status': self.status,
            'metadata': self.metadata
        }


@dataclass
class ComplianceRule:
    """合规规则"""
    rule_id: str
    name: str
    description: str
    rule_type: str  # position_limit, concentration_limit, leverage_limit, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    severity: RiskLevel = RiskLevel.MEDIUM
    applicable_symbols: List[str] = field(default_factory=list)
    applicable_strategies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'description': self.description,
            'rule_type': self.rule_type,
            'parameters': self.parameters,
            'enabled': self.enabled,
            'severity': self.severity.value,
            'applicable_symbols': self.applicable_symbols,
            'applicable_strategies': self.applicable_strategies,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class StressTestScenario:
    """压力测试场景"""
    scenario_id: str
    name: str
    description: str
    scenario_type: str  # market_crash, volatility_spike, liquidity_crisis, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    shock_factors: Dict[str, float] = field(default_factory=dict)  # symbol -> shock_factor
    duration: timedelta = timedelta(days=1)
    confidence_level: float = 0.95
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scenario_id': self.scenario_id,
            'name': self.name,
            'description': self.description,
            'scenario_type': self.scenario_type,
            'parameters': self.parameters,
            'shock_factors': self.shock_factors,
            'duration': self.duration.total_seconds(),
            'confidence_level': self.confidence_level,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class RiskConfig:
    """风险配置"""
    # VaR配置
    var_confidence_level: float = 0.95
    var_holding_period: int = 1  # 天
    var_method: str = "historical"  # historical, parametric, monte_carlo
    
    # 监控频率
    monitoring_interval: int = 60  # 秒
    alert_cooldown: int = 300  # 预警冷却时间（秒）
    
    # 风险限制
    max_portfolio_var: float = 0.05  # 5%
    max_position_concentration: float = 0.1  # 10%
    max_sector_concentration: float = 0.3  # 30%
    max_leverage: float = 3.0  # 3倍
    max_drawdown: float = 0.2  # 20%
    
    # 流动性要求
    min_liquidity_ratio: float = 0.1  # 10%
    max_illiquid_exposure: float = 0.2  # 20%
    
    # 预警设置
    enable_email_alerts: bool = True
    enable_sms_alerts: bool = False
    enable_webhook_alerts: bool = True
    
    # 自动控制
    enable_auto_risk_control: bool = True
    auto_reduce_position_threshold: float = 0.9  # 90%限制使用率时自动减仓
    emergency_stop_threshold: float = 0.98  # 98%限制使用率时紧急停止
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'var_confidence_level': self.var_confidence_level,
            'var_holding_period': self.var_holding_period,
            'var_method': self.var_method,
            'monitoring_interval': self.monitoring_interval,
            'alert_cooldown': self.alert_cooldown,
            'max_portfolio_var': self.max_portfolio_var,
            'max_position_concentration': self.max_position_concentration,
            'max_sector_concentration': self.max_sector_concentration,
            'max_leverage': self.max_leverage,
            'max_drawdown': self.max_drawdown,
            'min_liquidity_ratio': self.min_liquidity_ratio,
            'max_illiquid_exposure': self.max_illiquid_exposure,
            'enable_email_alerts': self.enable_email_alerts,
            'enable_sms_alerts': self.enable_sms_alerts,
            'enable_webhook_alerts': self.enable_webhook_alerts,
            'enable_auto_risk_control': self.enable_auto_risk_control,
            'auto_reduce_position_threshold': self.auto_reduce_position_threshold,
            'emergency_stop_threshold': self.emergency_stop_threshold
        }