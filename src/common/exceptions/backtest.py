"""Backtest Exception Classes

回测相关的异常类定义。
"""

from .base import PrometheusException


class BacktestException(PrometheusException):
    """回测异常基类"""
    pass


class BacktestInitializationError(BacktestException):
    """回测初始化错误"""
    pass


class BacktestConfigurationError(BacktestException):
    """回测配置错误"""
    pass


class BacktestDataError(BacktestException):
    """回测数据错误"""
    pass


class BacktestExecutionError(BacktestException):
    """回测执行错误"""
    pass


class BacktestValidationError(BacktestException):
    """回测验证错误"""
    pass


class BacktestResultError(BacktestException):
    """回测结果错误"""
    pass


class BacktestMetricsError(BacktestException):
    """回测指标错误"""
    pass


class BacktestReportError(BacktestException):
    """回测报告错误"""
    pass


class BacktestEngineError(BacktestException):
    """回测引擎错误"""
    pass


class HistoricalDataError(BacktestException):
    """历史数据错误"""
    pass


class SimulationError(BacktestException):
    """模拟错误"""
    pass


class PerformanceCalculationError(BacktestException):
    """性能计算错误"""
    pass


class BenchmarkError(BacktestException):
    """基准错误"""
    pass


class BacktestTimeoutError(BacktestException):
    """回测超时错误"""
    pass


class BacktestResourceError(BacktestException):
    """回测资源错误"""
    pass


class StrategyBacktestError(BacktestException):
    """策略回测错误"""
    pass


class PortfolioBacktestError(BacktestException):
    """组合回测错误"""
    pass


class RiskBacktestError(BacktestException):
    """风险回测错误"""
    pass


class OptimizationError(BacktestException):
    """优化错误"""
    pass


class BacktestOptimizationError(BacktestException):
    """回测优化错误"""
    pass


class WalkForwardError(BacktestException):
    """滚动前进错误"""
    pass


# 为了兼容性，添加别名
BacktestError = BacktestException