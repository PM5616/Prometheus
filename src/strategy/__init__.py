"""Strategy Engine Module

策略引擎模块，提供量化交易策略的开发、回测和执行框架。

主要功能：
- 策略基础框架
- 策略执行引擎
- 信号生成与管理
- 策略性能分析
- 策略组合管理

支持的策略类型：
- 趋势跟踪策略
- 均值回归策略
- 套利策略
- 机器学习策略
- 多因子策略

作者: Prometheus Team
版本: 1.0.0
"""

from .base import BaseStrategy, StrategySignal, StrategyState
from .engine import StrategyEngine
from .signal_manager import SignalManager
from .performance_analyzer import PerformanceAnalyzer
from .portfolio_manager import PortfolioManager

__version__ = "1.0.0"
__author__ = "Prometheus Team"

__all__ = [
    'BaseStrategy',
    'StrategySignal', 
    'StrategyState',
    'StrategyEngine',
    'SignalManager',
    'PerformanceAnalyzer',
    'PortfolioManager'
]