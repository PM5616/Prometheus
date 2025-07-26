"""Backtest Engine Module

回测引擎模块，提供历史数据回测功能。

主要功能：
- 历史数据回测
- 策略性能评估
- 风险分析
- 回测报告生成
- 参数优化
- 多策略对比

支持的回测类型：
- 单策略回测
- 多策略组合回测
- 参数扫描回测
- 蒙特卡洛回测
- 压力测试回测

核心组件：
- BacktestEngine: 回测引擎核心
- BacktestData: 回测数据管理
- BacktestMetrics: 回测指标计算
- BacktestReporter: 回测报告生成
- ParameterOptimizer: 参数优化器
"""

from .engine import BacktestEngine
from .data import BacktestData
from .metrics import BacktestMetrics
from .reporter import BacktestReporter
from .optimizer import ParameterOptimizer

__all__ = [
    'BacktestEngine',
    'BacktestData', 
    'BacktestMetrics',
    'BacktestReporter',
    'ParameterOptimizer'
]

__version__ = "1.0.0"
__author__ = "Prometheus Team"
__description__ = "Professional backtesting engine for quantitative trading strategies"