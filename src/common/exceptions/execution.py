"""Execution Exception Classes

执行相关的异常类定义。
"""

from .base import PrometheusException


class ExecutionException(PrometheusException):
    """执行异常基类"""
    pass


class OrderExecutionError(ExecutionException):
    """订单执行错误"""
    pass


class OrderValidationError(ExecutionException):
    """订单验证错误"""
    pass


class OrderTimeoutError(ExecutionException):
    """订单超时错误"""
    pass


class OrderCancellationError(ExecutionException):
    """订单取消错误"""
    pass


class OrderModificationError(ExecutionException):
    """订单修改错误"""
    pass


class InsufficientBalanceError(ExecutionException):
    """余额不足错误"""
    pass


class InsufficientLiquidityError(ExecutionException):
    """流动性不足错误"""
    pass


class PriceDeviationError(ExecutionException):
    """价格偏差错误"""
    pass


class SlippageExceededError(ExecutionException):
    """滑点超限错误"""
    pass


class ExecutionEngineError(ExecutionException):
    """执行引擎错误"""
    pass


class BrokerConnectionError(ExecutionException):
    """券商连接错误"""
    pass


class BrokerAPIError(ExecutionException):
    """券商API错误"""
    pass


class TradeExecutionError(ExecutionException):
    """交易执行错误"""
    pass


class PositionManagementError(ExecutionException):
    """仓位管理错误"""
    pass


class RiskCheckError(ExecutionException):
    """风险检查错误"""
    pass


class ExecutionLatencyError(ExecutionException):
    """执行延迟错误"""
    pass


class OrderBookError(ExecutionException):
    """订单簿错误"""
    pass


class FillReportError(ExecutionException):
    """成交报告错误"""
    pass


class ExecutionReportError(ExecutionException):
    """执行报告错误"""
    pass


class OrderManagerError(ExecutionException):
    """订单管理器错误"""
    pass


class TradeRouterError(ExecutionException):
    """交易路由器错误"""
    pass


class RoutingError(ExecutionException):
    """路由错误"""
    pass


class MarketConnectorError(ExecutionException):
    """市场连接器错误"""
    pass


class ConnectionError(ExecutionException):
    """连接错误"""
    pass