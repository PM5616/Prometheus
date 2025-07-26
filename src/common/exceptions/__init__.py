"""Common Exceptions Module

系统中使用的异常定义。
"""

from .base import (
    PrometheusException,
    ConfigurationError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NetworkError,
    PrometheusTimeoutError,
    RateLimitError,
    DataError,
    ProcessingError
)

from .trading import (
    TradingException,
    OrderException,
    InsufficientBalanceError,
    InvalidOrderError,
    OrderNotFoundError,
    MarketClosedError,
    SymbolNotFoundError,
    PriceOutOfRangeError
)

from .strategy import (
    StrategyException,
    StrategyNotFoundError,
    StrategyConfigError,
    StrategyExecutionError,
    SignalGenerationError,
    BacktestError
)

from .risk import (
    RiskException,
    RiskLimitExceededError,
    PositionSizeError,
    DrawdownLimitError,
    VolatilityLimitError,
    ExposureLimitError
)

from .data import (
    DataException,
    DataSourceError,
    DataQualityError,
    DataNotFoundError,
    DataFormatError,
    DataSyncError
)

from .strategy_exceptions import (
    StrategyException as StrategyExceptionNew,
    StrategyInitializationError,
    StrategyConfigurationError,
    StrategyExecutionError as StrategyExecutionErrorNew,
    StrategyParameterError,
    StrategySignalError,
    StrategyBacktestError,
    StrategyOptimizationError,
    StrategyValidationError,
    StrategyTimeoutError,
    StrategyResourceError,
    StrategyDataError,
    StrategyModelError,
    StrategyIndicatorError,
    StrategyRiskError,
    StrategyManagerError,
    StrategyError,
    StrategyConfigError,
    StrategyLoadError
)

from .execution import (
    ExecutionException,
    OrderExecutionError,
    OrderValidationError,
    OrderTimeoutError,
    OrderCancellationError,
    OrderModificationError,
    InsufficientBalanceError as InsufficientBalanceErrorExec,
    InsufficientLiquidityError,
    PriceDeviationError,
    SlippageExceededError,
    ExecutionEngineError,
    BrokerConnectionError,
    BrokerAPIError,
    TradeExecutionError,
    PositionManagementError,
    RiskCheckError,
    ExecutionLatencyError,
    OrderBookError,
    FillReportError,
    ExecutionReportError
)

from .monitor_exceptions import (
    MonitorException,
    MonitorInitializationError,
    MonitorConfigurationError,
    MetricCollectionError,
    MetricValidationError,
    AlertGenerationError,
    AlertDeliveryError,
    DashboardError,
    ReportGenerationError,
    PerformanceMonitorError,
    SystemMonitorError,
    HealthCheckError,
    LoggingError,
    AuditError,
    MonitoringServiceError,
    ThresholdExceededError,
    MonitorTimeoutError,
    DataCollectionError,
    MonitorResourceError,
    NotificationError
)

from .backtest import (
    BacktestException,
    BacktestInitializationError,
    BacktestConfigurationError,
    BacktestDataError,
    BacktestExecutionError,
    BacktestValidationError,
    BacktestResultError,
    BacktestReportError,
    BacktestEngineError,
    HistoricalDataError,
    SimulationError,
    PerformanceCalculationError,
    BenchmarkError,
    BacktestTimeoutError,
    BacktestResourceError,
    StrategyBacktestError,
    PortfolioBacktestError,
    RiskBacktestError,
    OptimizationError,
    WalkForwardError
)

__all__ = [
    # Base exceptions
    'PrometheusException',
    'ConfigurationError',
    'ValidationError',
    'AuthenticationError',
    'AuthorizationError',
    'NetworkError',
    'PrometheusTimeoutError',
    'RateLimitError',
    'DataError',
    'ProcessingError',
    
    # Trading exceptions
    'TradingException',
    'OrderException',
    'InsufficientBalanceError',
    'InvalidOrderError',
    'OrderNotFoundError',
    'MarketClosedError',
    'SymbolNotFoundError',
    'PriceOutOfRangeError',
    
    # Strategy exceptions
    'StrategyException',
    'StrategyNotFoundError',
    'StrategyConfigError',
    'StrategyExecutionError',
    'SignalGenerationError',
    'BacktestError',
    'StrategyError',
    
    # Risk exceptions
    'RiskException',
    'RiskLimitExceededError',
    'PositionSizeError',
    'DrawdownLimitError',
    'VolatilityLimitError',
    'ExposureLimitError',
    
    # Data exceptions
    'DataException',
    'DataSourceError',
    'DataQualityError',
    'DataNotFoundError',
    'DataFormatError',
    'DataSyncError',
    
    # Strategy exceptions (new)
    'StrategyExceptionNew',
    'StrategyInitializationError',
    'StrategyConfigurationError',
    'StrategyExecutionErrorNew',
    'StrategyParameterError',
    'StrategySignalError',
    'StrategyBacktestError',
    'StrategyOptimizationError',
    'StrategyValidationError',
    'StrategyTimeoutError',
    'StrategyResourceError',
    'StrategyDataError',
    'StrategyModelError',
    'StrategyIndicatorError',
    'StrategyRiskError',
    'StrategyManagerError',
    'StrategyError',
    'StrategyConfigError',
    'StrategyLoadError',
    
    # Execution exceptions
    'ExecutionException',
    'OrderExecutionError',
    'OrderValidationError',
    'OrderTimeoutError',
    'OrderCancellationError',
    'OrderModificationError',
    'InsufficientBalanceErrorExec',
    'InsufficientLiquidityError',
    'PriceDeviationError',
    'SlippageExceededError',
    'ExecutionEngineError',
    'BrokerConnectionError',
    'BrokerAPIError',
    'TradeExecutionError',
    'PositionManagementError',
    'RiskCheckError',
    'ExecutionLatencyError',
    'OrderBookError',
    'FillReportError',
    'ExecutionReportError',
    
    # Monitor exceptions
    'MonitorException',
    'MonitorInitializationError',
    'MonitorConfigurationError',
    'MetricCollectionError',
    'MetricValidationError',
    'AlertGenerationError',
    'AlertDeliveryError',
    'DashboardError',
    'ReportGenerationError',
    'PerformanceMonitorError',
    'SystemMonitorError',
    'HealthCheckError',
    'LoggingError',
    'AuditError',
    'MonitoringServiceError',
    'ThresholdExceededError',
    'MonitorTimeoutError',
    'DataCollectionError',
    'MonitorResourceError',
    'NotificationError',
    
    # Backtest exceptions
    'BacktestException',
    'BacktestInitializationError',
    'BacktestConfigurationError',
    'BacktestDataError',
    'BacktestExecutionError',
    'BacktestValidationError',
    'BacktestResultError',
    'BacktestReportError',
    'BacktestEngineError',
    'HistoricalDataError',
    'SimulationError',
    'PerformanceCalculationError',
    'BenchmarkError',
    'BacktestTimeoutError',
    'BacktestResourceError',
    'StrategyBacktestError',
    'PortfolioBacktestError',
    'RiskBacktestError',
    'OptimizationError',
    'WalkForwardError'
]

# 为了向后兼容，保留一些常用的别名
GatewayError = PrometheusException
DataHubError = DataError
TimeoutError = PrometheusTimeoutError  # 向后兼容别名

__version__ = "1.0.0"
__author__ = "Prometheus Trading System"