"""Alpha Engine Module

Alpha策略引擎模块，提供策略开发、回测和实盘交易的核心功能。
整合了原strategy模块的功能，提供向后兼容性。

主要功能：
- 策略基类和接口
- 信号生成和管理
- 策略加载和管理
- 策略引擎核心
- 兼容原strategy模块接口
"""

from .base_strategy import BaseStrategy as AlphaBaseStrategy, StrategyConfig, StrategyState as AlphaStrategyState
from .signal import Signal, SignalType, SignalStrength, SignalSource
from .strategy_loader import StrategyLoader
from .strategy_manager import StrategyManager
from .engine import StrategyEngine as AlphaEngine

# 兼容性导入 - 保持与原strategy模块的兼容性
from .base import BaseStrategy, StrategySignal, StrategyParameters, StrategyState

__all__ = [
    # Alpha Engine 核心类
    'AlphaBaseStrategy',
    'StrategyConfig', 
    'AlphaStrategyState',
    'Signal',
    'SignalType',
    'SignalStrength', 
    'SignalSource',
    'StrategyLoader',
    'StrategyManager',
    'AlphaEngine',
    
    # 兼容性类（原strategy模块）
    'BaseStrategy',
    'StrategySignal',
    'StrategyParameters',
    'StrategyState'
]

__version__ = '1.0.0'
__author__ = 'Prometheus Team'
__description__ = 'Alpha Engine - 策略引擎模块'