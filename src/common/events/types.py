"""Event Types

事件类型定义。
"""

from enum import Enum


class EventType(Enum):
    """事件类型枚举"""
    
    # 系统事件
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    
    # 交易事件
    ORDER_CREATED = "order.created"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    
    # 市场数据事件
    MARKET_DATA_RECEIVED = "market_data.received"
    PRICE_UPDATED = "price.updated"
    
    # 策略事件
    SIGNAL_GENERATED = "signal.generated"
    STRATEGY_STARTED = "strategy.started"
    STRATEGY_STOPPED = "strategy.stopped"
    
    # 风险事件
    RISK_LIMIT_EXCEEDED = "risk.limit_exceeded"
    POSITION_SIZE_EXCEEDED = "risk.position_size_exceeded"
    DRAWDOWN_EXCEEDED = "risk.drawdown_exceeded"
    
    # 组合管理事件
    PORTFOLIO_UPDATED = "portfolio.updated"
    REBALANCE_TRIGGERED = "portfolio.rebalance_triggered"
    
    # 监控事件
    ALERT_TRIGGERED = "monitor.alert_triggered"
    HEALTH_CHECK_FAILED = "monitor.health_check_failed"
    
    # 自定义事件
    CUSTOM = "custom"