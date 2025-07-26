"""Monitor Exception Classes

监控相关的异常类定义。
"""

from .base import PrometheusException


class MonitorException(PrometheusException):
    """监控异常基类"""
    pass


class MonitorInitializationError(MonitorException):
    """监控初始化错误"""
    pass


class MonitorConfigurationError(MonitorException):
    """监控配置错误"""
    pass


class MetricCollectionError(MonitorException):
    """指标收集错误"""
    pass


class MetricValidationError(MonitorException):
    """指标验证错误"""
    pass


class AlertGenerationError(MonitorException):
    """告警生成错误"""
    pass


class AlertDeliveryError(MonitorException):
    """告警发送错误"""
    pass


class DashboardError(MonitorException):
    """仪表板错误"""
    pass


class ReportGenerationError(MonitorException):
    """报告生成错误"""
    pass


class PerformanceMonitorError(MonitorException):
    """性能监控错误"""
    pass


class SystemMonitorError(MonitorException):
    """系统监控错误"""
    pass


class HealthCheckError(MonitorException):
    """健康检查错误"""
    pass


class LoggingError(MonitorException):
    """日志记录错误"""
    pass


class AuditError(MonitorException):
    """审计错误"""
    pass


class MonitoringServiceError(MonitorException):
    """监控服务错误"""
    pass


class ThresholdExceededError(MonitorException):
    """阈值超限错误"""
    pass


class MonitorTimeoutError(MonitorException):
    """监控超时错误"""
    pass


class DataCollectionError(MonitorException):
    """数据收集错误"""
    pass


class MonitorResourceError(MonitorException):
    """监控资源错误"""
    pass


class NotificationError(MonitorException):
    """通知错误"""
    pass


# 为了兼容性，添加别名
MonitorError = MonitorException