"""Common Enums

通用枚举定义，用于统一管理系统中的枚举类型。
"""

from enum import Enum


# ============================================================================
# 告警相关枚举
# ============================================================================

class AlertLevel(str, Enum):
    """告警级别"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertStatus(str, Enum):
    """告警状态"""
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"
    ACKNOWLEDGED = "ACKNOWLEDGED"


class AlertType(str, Enum):
    """告警类型"""
    RISK_LIMIT_BREACH = "RISK_LIMIT_BREACH"  # 风险限制违规
    HIGH_DRAWDOWN = "HIGH_DRAWDOWN"  # 高回撤
    POSITION_SIZE_EXCEEDED = "POSITION_SIZE_EXCEEDED"  # 仓位超限
    DAILY_LOSS_EXCEEDED = "DAILY_LOSS_EXCEEDED"  # 日损失超限
    SYSTEM_ERROR = "SYSTEM_ERROR"  # 系统错误
    MARKET_ANOMALY = "MARKET_ANOMALY"  # 市场异常
    STRATEGY_ERROR = "STRATEGY_ERROR"  # 策略错误
    THRESHOLD_BREACH = "THRESHOLD_BREACH"  # 阈值突破
    TREND_ALERT = "TREND_ALERT"  # 趋势预警
    ANOMALY_DETECTION = "ANOMALY_DETECTION"  # 异常检测
    MODEL_ALERT = "MODEL_ALERT"  # 模型预警
    COMPLIANCE_VIOLATION = "COMPLIANCE_VIOLATION"  # 合规违规


# ============================================================================
# 风险相关枚举
# ============================================================================

class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "LOW"  # 低风险
    MEDIUM = "MEDIUM"  # 中等风险
    HIGH = "HIGH"  # 高风险
    CRITICAL = "CRITICAL"  # 严重风险
    EMERGENCY = "EMERGENCY"  # 紧急风险


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


class RiskViolationType(str, Enum):
    """风险违规类型"""
    POSITION_SIZE = "POSITION_SIZE"  # 仓位大小超限
    CONCENTRATION = "CONCENTRATION"  # 集中度超限
    LEVERAGE = "LEVERAGE"  # 杠杆超限
    DRAWDOWN = "DRAWDOWN"  # 回撤超限
    VAR = "VAR"  # VaR超限
    CORRELATION = "CORRELATION"  # 相关性超限
    VOLATILITY = "VOLATILITY"  # 波动率超限
    SECTOR_EXPOSURE = "SECTOR_EXPOSURE"  # 行业暴露超限


# ============================================================================
# 交易相关枚举
# ============================================================================

class OrderType(str, Enum):
    """订单类型"""
    MARKET = "MARKET"  # 市价单
    LIMIT = "LIMIT"  # 限价单
    STOP = "STOP"  # 止损单
    STOP_LIMIT = "STOP_LIMIT"  # 止损限价单
    TRAILING_STOP = "TRAILING_STOP"  # 跟踪止损单
    IOC = "IOC"  # 立即成交或取消
    FOK = "FOK"  # 全部成交或取消


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "BUY"  # 买入
    SELL = "SELL"  # 卖出


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "PENDING"  # 待处理
    SUBMITTED = "SUBMITTED"  # 已提交
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # 部分成交
    FILLED = "FILLED"  # 已成交
    CANCELLED = "CANCELLED"  # 已取消
    REJECTED = "REJECTED"  # 已拒绝
    EXPIRED = "EXPIRED"  # 已过期


class TimeInForce(str, Enum):
    """订单有效期"""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"  # Day Order


# ============================================================================
# 数据质量相关枚举
# ============================================================================

class QualityLevel(str, Enum):
    """质量等级"""
    EXCELLENT = "EXCELLENT"  # 优秀
    GOOD = "GOOD"  # 良好
    FAIR = "FAIR"  # 一般
    POOR = "POOR"  # 较差
    CRITICAL = "CRITICAL"  # 严重


class CheckType(str, Enum):
    """检查类型"""
    COMPLETENESS = "COMPLETENESS"  # 完整性
    ACCURACY = "ACCURACY"  # 准确性
    CONSISTENCY = "CONSISTENCY"  # 一致性
    VALIDITY = "VALIDITY"  # 有效性
    TIMELINESS = "TIMELINESS"  # 及时性
    UNIQUENESS = "UNIQUENESS"  # 唯一性


# ============================================================================
# 系统相关枚举
# ============================================================================

class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class HealthStatus(str, Enum):
    """健康状态"""
    HEALTHY = "HEALTHY"  # 健康
    DEGRADED = "DEGRADED"  # 降级
    UNHEALTHY = "UNHEALTHY"  # 不健康
    UNKNOWN = "UNKNOWN"  # 未知


class NotificationChannel(str, Enum):
    """通知渠道"""
    EMAIL = "EMAIL"
    SMS = "SMS"
    WEBHOOK = "WEBHOOK"
    SLACK = "SLACK"
    DINGTALK = "DINGTALK"
    WECHAT = "WECHAT"


# ============================================================================
# 指标相关枚举
# ============================================================================

class MetricType(str, Enum):
    """指标类型"""
    COUNTER = "COUNTER"  # 计数器
    GAUGE = "GAUGE"  # 仪表盘
    HISTOGRAM = "HISTOGRAM"  # 直方图
    SUMMARY = "SUMMARY"  # 摘要


class MetricCategory(str, Enum):
    """指标分类"""
    RETURN = "RETURN"  # 收益指标
    RISK = "RISK"  # 风险指标
    RISK_ADJUSTED = "RISK_ADJUSTED"  # 风险调整指标
    DRAWDOWN = "DRAWDOWN"  # 回撤指标
    TRADING = "TRADING"  # 交易指标
    BENCHMARK = "BENCHMARK"  # 基准比较指标


# ============================================================================
# 策略相关枚举
# ============================================================================

class StrategyStatus(str, Enum):
    """策略状态"""
    INACTIVE = "INACTIVE"  # 未激活
    ACTIVE = "ACTIVE"  # 激活
    PAUSED = "PAUSED"  # 暂停
    STOPPED = "STOPPED"  # 停止
    ERROR = "ERROR"  # 错误


class SignalType(str, Enum):
    """信号类型"""
    BUY = "BUY"  # 买入信号
    SELL = "SELL"  # 卖出信号
    HOLD = "HOLD"  # 持有信号
    CLOSE = "CLOSE"  # 平仓信号
    REDUCE = "REDUCE"  # 减仓信号
    INCREASE = "INCREASE"  # 加仓信号


class SignalStrength(str, Enum):
    """信号强度"""
    WEAK = "WEAK"  # 弱信号
    MODERATE = "MODERATE"  # 中等信号
    STRONG = "STRONG"  # 强信号
    VERY_STRONG = "VERY_STRONG"  # 非常强信号