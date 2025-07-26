"""Alpha Engine Module

策略引擎模块，系统的"大脑"，负责策略的插件化加载和交易信号生成。

主要功能：
- 策略插件化管理
- 多策略并行运行
- 交易信号生成和分发
- 策略性能监控
- 策略参数动态调整
- 策略风险控制

核心组件：
- StrategyEngine: 策略引擎核心
- StrategyLoader: 策略加载器
- SignalAggregator: 信号聚合器
- PerformanceTracker: 性能跟踪器
- StrategyManager: 策略管理器

支持的策略类型：
- 震荡套利策略 (Mean Reversion)
- 趋势跟踪策略 (Trend Following)
- 市场状态识别策略 (Market Regime Detection)
- 协整配对交易策略 (Cointegration)
- 卡尔曼滤波趋势策略 (Kalman Filter)
- 隐马尔可夫模型策略 (HMM)
"""

from .engine import StrategyEngine
from .loader import StrategyLoader
from .aggregator import SignalAggregator
from .tracker import PerformanceTracker
from .manager import StrategyManager
from .base_strategy import BaseStrategy
from .signal import Signal, SignalType, SignalStrength

__all__ = [
    'StrategyEngine',
    'StrategyLoader', 
    'SignalAggregator',
    'PerformanceTracker',
    'StrategyManager',
    'BaseStrategy',
    'Signal',
    'SignalType',
    'SignalStrength'
]

__version__ = '1.0.0'
__author__ = 'Prometheus Team'
__description__ = 'Alpha Engine - 策略引擎模块'