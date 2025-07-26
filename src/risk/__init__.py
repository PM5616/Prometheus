"""Risk Management Module

风险管理模块，负责风险评估、控制和监控。

主要功能：
- 风险度量和评估
- 风险限制和控制
- 风险监控和报警
- VaR和压力测试
- 风险归因分析

支持的风险类型：
- 市场风险
- 信用风险
- 流动性风险
- 操作风险
- 模型风险

核心组件：
- RiskManager: 风险管理器
- RiskMetrics: 风险指标计算
- RiskMonitor: 风险监控器
- StressTest: 压力测试
- RiskReporter: 风险报告
"""

from .manager import RiskManager
from .metrics import RiskMetrics, RiskCalculator
from .monitor import RiskMonitor, RiskAlert
from .stress_test import StressTest, StressScenario
from .reporter import RiskReporter, RiskReport

__all__ = [
    'RiskManager',
    'RiskMetrics',
    'RiskCalculator', 
    'RiskMonitor',
    'RiskAlert',
    'StressTest',
    'StressScenario',
    'RiskReporter',
    'RiskReport'
]

__version__ = '1.0.0'
__author__ = 'Prometheus Team'
__description__ = 'Risk Management Module for Prometheus Trading System'