"""Strategy Exception Classes

策略相关的异常类定义。
"""

from .base import PrometheusException


class StrategyException(PrometheusException):
    """策略异常基类"""
    pass


class StrategyInitializationError(StrategyException):
    """策略初始化错误"""
    pass


class StrategyConfigurationError(StrategyException):
    """策略配置错误"""
    pass


class StrategyExecutionError(StrategyException):
    """策略执行错误"""
    pass


class StrategyParameterError(StrategyException):
    """策略参数错误"""
    pass


class StrategySignalError(StrategyException):
    """策略信号错误"""
    pass


class StrategyBacktestError(StrategyException):
    """策略回测错误"""
    pass


class StrategyOptimizationError(StrategyException):
    """策略优化错误"""
    pass


class StrategyValidationError(StrategyException):
    """策略验证错误"""
    pass


class StrategyTimeoutError(StrategyException):
    """策略超时错误"""
    pass


class StrategyResourceError(StrategyException):
    """策略资源错误"""
    pass


class StrategyDataError(StrategyException):
    """策略数据错误"""
    pass


class StrategyModelError(StrategyException):
    """策略模型错误"""
    pass


class StrategyIndicatorError(StrategyException):
    """策略指标错误"""
    pass


class StrategyRiskError(StrategyException):
    """策略风险错误"""
    pass


class StrategyManagerError(StrategyException):
    """策略管理器错误"""
    pass


class StrategyEngineError(StrategyException):
    """策略引擎错误"""
    pass


# 为了兼容性，添加别名
StrategyError = StrategyException
StrategyConfigError = StrategyConfigurationError
StrategyLoadError = StrategyInitializationError