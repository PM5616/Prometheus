"""Portfolio Manager Module

投资组合管理模块，负责整合策略信号、优化投资组合配置、管理风险

主要职责：
1. 信号整合：收集和处理来自不同策略的交易信号
2. 风险管理：监控和控制投资组合风险
3. 投资组合优化：根据风险收益目标优化资产配置
4. 执行管理：生成和管理交易指令
5. 绩效监控：跟踪和分析投资组合表现
6. 绩效分析：计算和分析投资组合绩效指标
7. 投资组合再平衡：根据优化结果进行再平衡
"""

from .models import (
    SignalType,
    SignalStrength,
    PositionType,
    PositionStatus,
    RiskLevel,
    Signal,
    Position,
    Portfolio,
    RiskMetrics,
    PerformanceMetrics,
    OptimizationConfig,
    RiskConfig
)

from .manager import PortfolioManager
from .optimizer import PortfolioOptimizer
from .risk_manager import RiskManager, RiskViolationType, RiskViolation
from .signal_processor import SignalProcessor, SignalConflictResolution, SignalAggregationMethod, ProcessedSignal
from .performance_analyzer import PerformanceAnalyzer, PerformancePeriod, BenchmarkType, PerformanceRecord
from .rebalancer import PortfolioRebalancer, RebalanceFrequency, RebalanceMethod, RebalanceConfig, RebalanceOrder

__all__ = [
    # 枚举类型
    'SignalType',
    'SignalStrength', 
    'PositionType',
    'PositionStatus',
    'RiskLevel',
    'RiskViolationType',
    'SignalConflictResolution',
    'SignalAggregationMethod',
    'PerformancePeriod',
    'BenchmarkType',
    'RebalanceFrequency',
    'RebalanceMethod',
    
    # 数据模型
    'Signal',
    'Position',
    'Portfolio',
    'RiskMetrics',
    'PerformanceMetrics',
    'OptimizationConfig',
    'RiskConfig',
    'RiskViolation',
    'ProcessedSignal',
    'PerformanceRecord',
    'RebalanceConfig',
    'RebalanceOrder',
    
    # 核心类
    'PortfolioManager',
    'PortfolioOptimizer',
    'RiskManager',
    'SignalProcessor',
    'PerformanceAnalyzer',
    'PortfolioRebalancer'
]