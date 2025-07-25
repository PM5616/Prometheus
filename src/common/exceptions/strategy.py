"""Strategy Exception Classes

策略相关异常类定义。
"""

from typing import Optional, Dict, Any
from .base import PrometheusException


class StrategyException(PrometheusException):
    """策略异常基类"""
    
    def __init__(self, 
                 message: str = "Strategy error occurred",
                 strategy_id: Optional[str] = None,
                 strategy_name: Optional[str] = None,
                 **kwargs):
        """初始化策略异常
        
        Args:
            message: 错误消息
            strategy_id: 策略ID
            strategy_name: 策略名称
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if strategy_id:
            details['strategy_id'] = strategy_id
        if strategy_name:
            details['strategy_name'] = strategy_name
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class StrategyNotFoundError(StrategyException):
    """策略未找到异常"""
    
    def __init__(self, 
                 message: str = "Strategy not found",
                 available_strategies: Optional[list] = None,
                 **kwargs):
        """初始化策略未找到异常
        
        Args:
            message: 错误消息
            available_strategies: 可用策略列表
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if available_strategies:
            details['available_strategies'] = available_strategies[:10]  # 限制数量
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class StrategyConfigurationError(StrategyException):
    """策略配置异常"""
    
    def __init__(self, 
                 message: str = "Strategy configuration error",
                 config_errors: Optional[list] = None,
                 **kwargs):
        """初始化策略配置异常
        
        Args:
            message: 错误消息
            config_errors: 配置错误列表
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if config_errors:
            details['config_errors'] = config_errors
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class StrategyInitializationError(StrategyException):
    """策略初始化异常"""
    
    def __init__(self, 
                 message: str = "Strategy initialization failed",
                 initialization_stage: Optional[str] = None,
                 **kwargs):
        """初始化策略初始化异常
        
        Args:
            message: 错误消息
            initialization_stage: 初始化阶段
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if initialization_stage:
            details['initialization_stage'] = initialization_stage
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class StrategyExecutionError(StrategyException):
    """策略执行异常"""
    
    def __init__(self, 
                 message: str = "Strategy execution failed",
                 execution_stage: Optional[str] = None,
                 signal_data: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """初始化策略执行异常
        
        Args:
            message: 错误消息
            execution_stage: 执行阶段
            signal_data: 信号数据
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if execution_stage:
            details['execution_stage'] = execution_stage
        if signal_data:
            details['signal_data'] = signal_data
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class StrategyValidationError(StrategyException):
    """策略验证异常"""
    
    def __init__(self, 
                 message: str = "Strategy validation failed",
                 validation_errors: Optional[list] = None,
                 **kwargs):
        """初始化策略验证异常
        
        Args:
            message: 错误消息
            validation_errors: 验证错误列表
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if validation_errors:
            details['validation_errors'] = validation_errors
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class StrategyStateError(StrategyException):
    """策略状态异常"""
    
    def __init__(self, 
                 message: str = "Invalid strategy state",
                 current_state: Optional[str] = None,
                 expected_state: Optional[str] = None,
                 **kwargs):
        """初始化策略状态异常
        
        Args:
            message: 错误消息
            current_state: 当前状态
            expected_state: 期望状态
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if current_state:
            details['current_state'] = current_state
        if expected_state:
            details['expected_state'] = expected_state
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class StrategyTimeoutError(StrategyException):
    """策略超时异常"""
    
    def __init__(self, 
                 message: str = "Strategy operation timeout",
                 timeout_seconds: Optional[float] = None,
                 operation: Optional[str] = None,
                 **kwargs):
        """初始化策略超时异常
        
        Args:
            message: 错误消息
            timeout_seconds: 超时秒数
            operation: 操作名称
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if timeout_seconds is not None:
            details['timeout_seconds'] = timeout_seconds
        if operation:
            details['operation'] = operation
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class StrategyResourceError(StrategyException):
    """策略资源异常"""
    
    def __init__(self, 
                 message: str = "Strategy resource error",
                 resource_type: Optional[str] = None,
                 resource_limit: Optional[str] = None,
                 **kwargs):
        """初始化策略资源异常
        
        Args:
            message: 错误消息
            resource_type: 资源类型
            resource_limit: 资源限制
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if resource_type:
            details['resource_type'] = resource_type
        if resource_limit:
            details['resource_limit'] = resource_limit
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class StrategyDependencyError(StrategyException):
    """策略依赖异常"""
    
    def __init__(self, 
                 message: str = "Strategy dependency error",
                 missing_dependencies: Optional[list] = None,
                 **kwargs):
        """初始化策略依赖异常
        
        Args:
            message: 错误消息
            missing_dependencies: 缺失依赖列表
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if missing_dependencies:
            details['missing_dependencies'] = missing_dependencies
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class SignalGenerationError(StrategyException):
    """信号生成异常"""
    
    def __init__(self, 
                 message: str = "Signal generation failed",
                 signal_type: Optional[str] = None,
                 market_data: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """初始化信号生成异常
        
        Args:
            message: 错误消息
            signal_type: 信号类型
            market_data: 市场数据
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if signal_type:
            details['signal_type'] = signal_type
        if market_data:
            # 只保留关键信息，避免数据过大
            details['market_data_keys'] = list(market_data.keys())
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class BacktestError(StrategyException):
    """回测异常"""
    
    def __init__(self, 
                 message: str = "Backtest failed",
                 backtest_period: Optional[str] = None,
                 data_issues: Optional[list] = None,
                 **kwargs):
        """初始化回测异常
        
        Args:
            message: 错误消息
            backtest_period: 回测周期
            data_issues: 数据问题列表
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if backtest_period:
            details['backtest_period'] = backtest_period
        if data_issues:
            details['data_issues'] = data_issues
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class StrategyOptimizationError(StrategyException):
    """策略优化异常"""
    
    def __init__(self, 
                 message: str = "Strategy optimization failed",
                 optimization_method: Optional[str] = None,
                 parameter_space: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """初始化策略优化异常
        
        Args:
            message: 错误消息
            optimization_method: 优化方法
            parameter_space: 参数空间
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if optimization_method:
            details['optimization_method'] = optimization_method
        if parameter_space:
            details['parameter_count'] = len(parameter_space)
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class StrategyPerformanceError(StrategyException):
    """策略性能异常"""
    
    def __init__(self, 
                 message: str = "Strategy performance issue",
                 performance_metric: Optional[str] = None,
                 threshold_value: Optional[float] = None,
                 actual_value: Optional[float] = None,
                 **kwargs):
        """初始化策略性能异常
        
        Args:
            message: 错误消息
            performance_metric: 性能指标
            threshold_value: 阈值
            actual_value: 实际值
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if performance_metric:
            details['performance_metric'] = performance_metric
        if threshold_value is not None:
            details['threshold_value'] = threshold_value
        if actual_value is not None:
            details['actual_value'] = actual_value
        kwargs['details'] = details
        super().__init__(message, **kwargs)