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
    TimeoutError,
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

__all__ = [
    # Base exceptions
    'PrometheusException',
    'ConfigurationError',
    'ValidationError',
    'AuthenticationError',
    'AuthorizationError',
    'NetworkError',
    'TimeoutError',
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
    'DataSyncError'
]

__version__ = "1.0.0"
__author__ = "Prometheus Trading System"