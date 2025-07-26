"""Backtest Metrics Module

回测指标计算模块，负责计算各种回测性能指标。

主要功能：
- 收益率计算
- 风险指标计算
- 回撤分析
- 基准比较
- 统计分析
- 指标可视化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import math
from scipy import stats
from collections import defaultdict

from ..common.logging import get_logger
from ..common.exceptions.backtest import BacktestMetricsError


class MetricCategory(Enum):
    """指标分类枚举"""
    RETURN = "return"              # 收益指标
    RISK = "risk"                  # 风险指标
    RISK_ADJUSTED = "risk_adjusted" # 风险调整指标
    DRAWDOWN = "drawdown"          # 回撤指标
    TRADING = "trading"            # 交易指标
    BENCHMARK = "benchmark"        # 基准比较指标


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str
    description: str
    category: MetricCategory
    calculation_function: callable
    format_function: Optional[callable] = None
    higher_is_better: bool = True
    unit: str = ""
    
    def format_value(self, value: float) -> str:
        """格式化指标值
        
        Args:
            value: 指标值
            
        Returns:
            str: 格式化后的值
        """
        if self.format_function:
            return self.format_function(value)
        
        if self.unit == "%":
            return f"{value:.2%}"
        elif self.unit == "ratio":
            return f"{value:.4f}"
        elif self.unit == "days":
            return f"{value:.0f} days"
        else:
            return f"{value:.4f}"


@dataclass
class DrawdownPeriod:
    """回撤期间信息"""
    start_date: datetime
    end_date: datetime
    recovery_date: Optional[datetime]
    peak_value: float
    trough_value: float
    drawdown: float
    duration_days: int
    recovery_days: Optional[int] = None
    
    @property
    def is_recovered(self) -> bool:
        """是否已恢复"""
        return self.recovery_date is not None
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 回撤期间字典
        """
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'recovery_date': self.recovery_date.isoformat() if self.recovery_date else None,
            'peak_value': self.peak_value,
            'trough_value': self.trough_value,
            'drawdown': self.drawdown,
            'duration_days': self.duration_days,
            'recovery_days': self.recovery_days,
            'is_recovered': self.is_recovered
        }


@dataclass
class TradingStatistics:
    """交易统计信息"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    total_profit: float = 0.0
    total_loss: float = 0.0
    
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    @property
    def win_rate(self) -> float:
        """胜率"""
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
    
    @property
    def loss_rate(self) -> float:
        """败率"""
        return self.losing_trades / self.total_trades if self.total_trades > 0 else 0.0
    
    @property
    def profit_factor(self) -> float:
        """盈利因子"""
        return abs(self.total_profit / self.total_loss) if self.total_loss != 0 else float('inf')
    
    @property
    def avg_trade(self) -> float:
        """平均每笔交易收益"""
        return (self.total_profit + self.total_loss) / self.total_trades if self.total_trades > 0 else 0.0
    
    @property
    def expectancy(self) -> float:
        """期望收益"""
        if self.total_trades == 0:
            return 0.0
        return (self.win_rate * self.avg_win) + (self.loss_rate * self.avg_loss)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 交易统计字典
        """
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'loss_rate': self.loss_rate,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'avg_trade': self.avg_trade,
            'expectancy': self.expectancy,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses
        }


@dataclass
class BacktestMetrics:
    """回测指标结果"""
    # 基本信息
    start_date: datetime
    end_date: datetime
    total_days: int
    trading_days: int
    
    # 收益指标
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_return: float = 0.0
    daily_return: float = 0.0
    
    # 风险指标
    volatility: float = 0.0
    downside_volatility: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    # 风险调整指标
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # 回撤指标
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_drawdown: float = 0.0
    drawdown_periods: List[DrawdownPeriod] = field(default_factory=list)
    
    # 交易指标
    trading_stats: TradingStatistics = field(default_factory=TradingStatistics)
    
    # 基准比较指标
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    correlation: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    
    # 其他指标
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 指标字典
        """
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_days': self.total_days,
            'trading_days': self.trading_days,
            
            # 收益指标
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'monthly_return': self.monthly_return,
            'daily_return': self.daily_return,
            
            # 风险指标
            'volatility': self.volatility,
            'downside_volatility': self.downside_volatility,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            
            # 风险调整指标
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'omega_ratio': self.omega_ratio,
            
            # 回撤指标
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'avg_drawdown': self.avg_drawdown,
            'drawdown_periods': [dp.to_dict() for dp in self.drawdown_periods],
            
            # 交易指标
            'trading_stats': self.trading_stats.to_dict(),
            
            # 基准比较指标
            'benchmark_return': self.benchmark_return,
            'alpha': self.alpha,
            'beta': self.beta,
            'correlation': self.correlation,
            'tracking_error': self.tracking_error,
            'information_ratio': self.information_ratio,
            
            # 其他指标
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'tail_ratio': self.tail_ratio
        }


class MetricsCalculator:
    """指标计算器
    
    负责计算各种回测性能指标。
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """初始化指标计算器
        
        Args:
            risk_free_rate: 无风险利率
        """
        self.risk_free_rate = risk_free_rate
        self.logger = get_logger("MetricsCalculator")
        
        # 注册指标定义
        self.metric_definitions = self._register_metric_definitions()
    
    def _register_metric_definitions(self) -> Dict[str, MetricDefinition]:
        """注册指标定义
        
        Returns:
            Dict[str, MetricDefinition]: 指标定义字典
        """
        definitions = {}
        
        # 收益指标
        definitions['total_return'] = MetricDefinition(
            name="总收益率",
            description="整个回测期间的总收益率",
            category=MetricCategory.RETURN,
            calculation_function=self._calculate_total_return,
            unit="%"
        )
        
        definitions['annual_return'] = MetricDefinition(
            name="年化收益率",
            description="年化收益率",
            category=MetricCategory.RETURN,
            calculation_function=self._calculate_annual_return,
            unit="%"
        )
        
        # 风险指标
        definitions['volatility'] = MetricDefinition(
            name="波动率",
            description="年化波动率",
            category=MetricCategory.RISK,
            calculation_function=self._calculate_volatility,
            higher_is_better=False,
            unit="%"
        )
        
        definitions['max_drawdown'] = MetricDefinition(
            name="最大回撤",
            description="最大回撤幅度",
            category=MetricCategory.DRAWDOWN,
            calculation_function=self._calculate_max_drawdown,
            higher_is_better=False,
            unit="%"
        )
        
        # 风险调整指标
        definitions['sharpe_ratio'] = MetricDefinition(
            name="夏普比率",
            description="风险调整后收益率",
            category=MetricCategory.RISK_ADJUSTED,
            calculation_function=self._calculate_sharpe_ratio,
            unit="ratio"
        )
        
        definitions['sortino_ratio'] = MetricDefinition(
            name="索提诺比率",
            description="下行风险调整后收益率",
            category=MetricCategory.RISK_ADJUSTED,
            calculation_function=self._calculate_sortino_ratio,
            unit="ratio"
        )
        
        return definitions
    
    def calculate_metrics(self, equity_curve: pd.Series, 
                         benchmark_curve: Optional[pd.Series] = None,
                         trades: Optional[List[Dict]] = None) -> BacktestMetrics:
        """计算回测指标
        
        Args:
            equity_curve: 资产曲线
            benchmark_curve: 基准曲线
            trades: 交易记录
            
        Returns:
            BacktestMetrics: 回测指标
        """
        if equity_curve.empty:
            raise BacktestMetricsError("资产曲线为空")
        
        self.logger.info("开始计算回测指标...")
        
        # 基本信息
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        total_days = (end_date - start_date).days
        trading_days = len(equity_curve)
        
        # 计算收益率序列
        returns = equity_curve.pct_change().dropna()
        
        # 初始化指标对象
        metrics = BacktestMetrics(
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            trading_days=trading_days
        )
        
        try:
            # 计算收益指标
            metrics.total_return = self._calculate_total_return(equity_curve)
            metrics.annual_return = self._calculate_annual_return(equity_curve)
            metrics.monthly_return = self._calculate_monthly_return(equity_curve)
            metrics.daily_return = self._calculate_daily_return(returns)
            
            # 计算风险指标
            metrics.volatility = self._calculate_volatility(returns)
            metrics.downside_volatility = self._calculate_downside_volatility(returns)
            metrics.var_95 = self._calculate_var(returns, 0.05)
            metrics.var_99 = self._calculate_var(returns, 0.01)
            metrics.cvar_95 = self._calculate_cvar(returns, 0.05)
            metrics.cvar_99 = self._calculate_cvar(returns, 0.01)
            
            # 计算风险调整指标
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
            metrics.calmar_ratio = self._calculate_calmar_ratio(equity_curve)
            metrics.omega_ratio = self._calculate_omega_ratio(returns)
            
            # 计算回撤指标
            drawdown_info = self._calculate_drawdown_analysis(equity_curve)
            metrics.max_drawdown = drawdown_info['max_drawdown']
            metrics.max_drawdown_duration = drawdown_info['max_duration']
            metrics.avg_drawdown = drawdown_info['avg_drawdown']
            metrics.drawdown_periods = drawdown_info['periods']
            
            # 计算交易指标
            if trades:
                metrics.trading_stats = self._calculate_trading_statistics(trades)
            
            # 计算基准比较指标
            if benchmark_curve is not None:
                benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_curve)
                metrics.benchmark_return = benchmark_metrics['benchmark_return']
                metrics.alpha = benchmark_metrics['alpha']
                metrics.beta = benchmark_metrics['beta']
                metrics.correlation = benchmark_metrics['correlation']
                metrics.tracking_error = benchmark_metrics['tracking_error']
                metrics.information_ratio = benchmark_metrics['information_ratio']
            
            # 计算其他指标
            metrics.skewness = self._calculate_skewness(returns)
            metrics.kurtosis = self._calculate_kurtosis(returns)
            metrics.tail_ratio = self._calculate_tail_ratio(returns)
            
            self.logger.info("回测指标计算完成")
            
        except Exception as e:
            self.logger.error(f"指标计算失败: {e}")
            raise BacktestMetricsError(f"指标计算失败: {e}")
        
        return metrics
    
    def _calculate_total_return(self, equity_curve: pd.Series) -> float:
        """计算总收益率
        
        Args:
            equity_curve: 资产曲线
            
        Returns:
            float: 总收益率
        """
        if len(equity_curve) < 2:
            return 0.0
        
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
    
    def _calculate_annual_return(self, equity_curve: pd.Series) -> float:
        """计算年化收益率
        
        Args:
            equity_curve: 资产曲线
            
        Returns:
            float: 年化收益率
        """
        if len(equity_curve) < 2:
            return 0.0
        
        total_return = self._calculate_total_return(equity_curve)
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        
        if days <= 0:
            return 0.0
        
        return (1 + total_return) ** (365.25 / days) - 1.0
    
    def _calculate_monthly_return(self, equity_curve: pd.Series) -> float:
        """计算月化收益率
        
        Args:
            equity_curve: 资产曲线
            
        Returns:
            float: 月化收益率
        """
        annual_return = self._calculate_annual_return(equity_curve)
        return (1 + annual_return) ** (1/12) - 1.0
    
    def _calculate_daily_return(self, returns: pd.Series) -> float:
        """计算平均日收益率
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 平均日收益率
        """
        return returns.mean()
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """计算年化波动率
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 年化波动率
        """
        if len(returns) < 2:
            return 0.0
        
        return returns.std() * np.sqrt(252)  # 假设252个交易日
    
    def _calculate_downside_volatility(self, returns: pd.Series, target_return: float = 0.0) -> float:
        """计算下行波动率
        
        Args:
            returns: 收益率序列
            target_return: 目标收益率
            
        Returns:
            float: 下行波动率
        """
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) < 2:
            return 0.0
        
        return downside_returns.std() * np.sqrt(252)
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """计算风险价值(VaR)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            
        Returns:
            float: VaR值
        """
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """计算条件风险价值(CVaR)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            
        Returns:
            float: CVaR值
        """
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        
        return tail_returns.mean() if len(tail_returns) > 0 else var
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """计算夏普比率
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 夏普比率
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252  # 日化无风险利率
        
        if excess_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """计算索提诺比率
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 索提诺比率
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252
        downside_std = self._calculate_downside_volatility(returns) / np.sqrt(252)
        
        if downside_std == 0:
            return 0.0
        
        return excess_returns.mean() / downside_std * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, equity_curve: pd.Series) -> float:
        """计算卡玛比率
        
        Args:
            equity_curve: 资产曲线
            
        Returns:
            float: 卡玛比率
        """
        annual_return = self._calculate_annual_return(equity_curve)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        if max_drawdown == 0:
            return 0.0
        
        return annual_return / abs(max_drawdown)
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """计算欧米茄比率
        
        Args:
            returns: 收益率序列
            threshold: 阈值收益率
            
        Returns:
            float: 欧米茄比率
        """
        if len(returns) == 0:
            return 0.0
        
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        total_gains = gains.sum()
        total_losses = losses.sum()
        
        if total_losses == 0:
            return float('inf') if total_gains > 0 else 0.0
        
        return total_gains / total_losses
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """计算最大回撤
        
        Args:
            equity_curve: 资产曲线
            
        Returns:
            float: 最大回撤
        """
        if len(equity_curve) < 2:
            return 0.0
        
        # 计算累计最高点
        peak = equity_curve.expanding().max()
        
        # 计算回撤
        drawdown = (equity_curve - peak) / peak
        
        return drawdown.min()
    
    def _calculate_drawdown_analysis(self, equity_curve: pd.Series) -> Dict:
        """计算回撤分析
        
        Args:
            equity_curve: 资产曲线
            
        Returns:
            Dict: 回撤分析结果
        """
        if len(equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'max_duration': 0,
                'avg_drawdown': 0.0,
                'periods': []
            }
        
        # 计算累计最高点和回撤
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        # 识别回撤期间
        periods = []
        in_drawdown = False
        start_idx = None
        peak_value = None
        
        for i, (date, dd) in enumerate(drawdown.items()):
            if dd < 0 and not in_drawdown:
                # 开始回撤
                in_drawdown = True
                start_idx = i
                peak_value = peak.iloc[i]
                
            elif dd >= 0 and in_drawdown:
                # 结束回撤
                in_drawdown = False
                end_idx = i - 1
                
                if start_idx is not None:
                    period_drawdown = drawdown.iloc[start_idx:end_idx+1]
                    min_idx = period_drawdown.idxmin()
                    
                    period = DrawdownPeriod(
                        start_date=drawdown.index[start_idx],
                        end_date=drawdown.index[end_idx],
                        recovery_date=date,
                        peak_value=peak_value,
                        trough_value=equity_curve.loc[min_idx],
                        drawdown=period_drawdown.min(),
                        duration_days=(drawdown.index[end_idx] - drawdown.index[start_idx]).days,
                        recovery_days=(date - drawdown.index[end_idx]).days
                    )
                    periods.append(period)
        
        # 处理未恢复的回撤
        if in_drawdown and start_idx is not None:
            period_drawdown = drawdown.iloc[start_idx:]
            min_idx = period_drawdown.idxmin()
            
            period = DrawdownPeriod(
                start_date=drawdown.index[start_idx],
                end_date=drawdown.index[-1],
                recovery_date=None,
                peak_value=peak_value,
                trough_value=equity_curve.loc[min_idx],
                drawdown=period_drawdown.min(),
                duration_days=(drawdown.index[-1] - drawdown.index[start_idx]).days
            )
            periods.append(period)
        
        # 计算统计信息
        max_drawdown = drawdown.min()
        max_duration = max([p.duration_days for p in periods]) if periods else 0
        avg_drawdown = np.mean([p.drawdown for p in periods]) if periods else 0.0
        
        return {
            'max_drawdown': max_drawdown,
            'max_duration': max_duration,
            'avg_drawdown': avg_drawdown,
            'periods': periods
        }
    
    def _calculate_trading_statistics(self, trades: List[Dict]) -> TradingStatistics:
        """计算交易统计
        
        Args:
            trades: 交易记录
            
        Returns:
            TradingStatistics: 交易统计
        """
        stats = TradingStatistics()
        
        if not trades:
            return stats
        
        # 计算每笔交易的盈亏
        trade_pnls = []
        for trade in trades:
            pnl = trade.get('pnl', 0.0)
            trade_pnls.append(pnl)
            
            if pnl > 0:
                stats.winning_trades += 1
                stats.total_profit += pnl
                stats.largest_win = max(stats.largest_win, pnl)
            elif pnl < 0:
                stats.losing_trades += 1
                stats.total_loss += pnl
                stats.largest_loss = min(stats.largest_loss, pnl)
        
        stats.total_trades = len(trades)
        
        # 计算平均值
        if stats.winning_trades > 0:
            stats.avg_win = stats.total_profit / stats.winning_trades
        
        if stats.losing_trades > 0:
            stats.avg_loss = stats.total_loss / stats.losing_trades
        
        # 计算连续胜负次数
        current_streak = 0
        current_type = None  # 'win' or 'loss'
        
        for pnl in trade_pnls:
            if pnl > 0:
                if current_type == 'win':
                    current_streak += 1
                else:
                    stats.max_consecutive_wins = max(stats.max_consecutive_wins, current_streak)
                    current_streak = 1
                    current_type = 'win'
            elif pnl < 0:
                if current_type == 'loss':
                    current_streak += 1
                else:
                    stats.max_consecutive_losses = max(stats.max_consecutive_losses, current_streak)
                    current_streak = 1
                    current_type = 'loss'
            else:
                # 平局，重置
                if current_type == 'win':
                    stats.max_consecutive_wins = max(stats.max_consecutive_wins, current_streak)
                elif current_type == 'loss':
                    stats.max_consecutive_losses = max(stats.max_consecutive_losses, current_streak)
                current_streak = 0
                current_type = None
        
        # 处理最后的连续记录
        if current_type == 'win':
            stats.max_consecutive_wins = max(stats.max_consecutive_wins, current_streak)
        elif current_type == 'loss':
            stats.max_consecutive_losses = max(stats.max_consecutive_losses, current_streak)
        
        return stats
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, 
                                   benchmark_curve: pd.Series) -> Dict:
        """计算基准比较指标
        
        Args:
            returns: 策略收益率序列
            benchmark_curve: 基准曲线
            
        Returns:
            Dict: 基准比较指标
        """
        # 计算基准收益率
        benchmark_returns = benchmark_curve.pct_change().dropna()
        
        # 对齐时间序列
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) < 2:
            return {
                'benchmark_return': 0.0,
                'alpha': 0.0,
                'beta': 0.0,
                'correlation': 0.0,
                'tracking_error': 0.0,
                'information_ratio': 0.0
            }
        
        # 基准总收益率
        benchmark_return = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0]) - 1.0
        
        # 计算Beta和Alpha
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0.0
        
        # Alpha = 策略收益率 - (无风险利率 + Beta * (基准收益率 - 无风险利率))
        strategy_return = aligned_returns.mean() * 252  # 年化
        benchmark_annual = aligned_benchmark.mean() * 252  # 年化
        alpha = strategy_return - (self.risk_free_rate + beta * (benchmark_annual - self.risk_free_rate))
        
        # 相关系数
        correlation = np.corrcoef(aligned_returns, aligned_benchmark)[0, 1]
        
        # 跟踪误差
        tracking_error = (aligned_returns - aligned_benchmark).std() * np.sqrt(252)
        
        # 信息比率
        excess_return = aligned_returns.mean() - aligned_benchmark.mean()
        information_ratio = (excess_return * 252) / tracking_error if tracking_error != 0 else 0.0
        
        return {
            'benchmark_return': benchmark_return,
            'alpha': alpha,
            'beta': beta,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
    
    def _calculate_skewness(self, returns: pd.Series) -> float:
        """计算偏度
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 偏度
        """
        if len(returns) < 3:
            return 0.0
        
        return stats.skew(returns)
    
    def _calculate_kurtosis(self, returns: pd.Series) -> float:
        """计算峰度
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 峰度
        """
        if len(returns) < 4:
            return 0.0
        
        return stats.kurtosis(returns)
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """计算尾部比率
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 尾部比率
        """
        if len(returns) < 20:  # 需要足够的数据点
            return 0.0
        
        # 计算95%分位数和5%分位数的比率
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if p5 == 0:
            return 0.0
        
        return abs(p95 / p5)
    
    def calculate_rolling_metrics(self, equity_curve: pd.Series, 
                                window_days: int = 252) -> pd.DataFrame:
        """计算滚动指标
        
        Args:
            equity_curve: 资产曲线
            window_days: 滚动窗口天数
            
        Returns:
            pd.DataFrame: 滚动指标DataFrame
        """
        if len(equity_curve) < window_days:
            raise BacktestMetricsError(f"数据长度不足，需要至少{window_days}个数据点")
        
        returns = equity_curve.pct_change().dropna()
        
        rolling_metrics = pd.DataFrame(index=equity_curve.index[window_days:])
        
        # 滚动收益率
        rolling_metrics['return'] = equity_curve.rolling(window_days).apply(
            lambda x: (x.iloc[-1] / x.iloc[0]) - 1.0
        )[window_days:]
        
        # 滚动波动率
        rolling_metrics['volatility'] = returns.rolling(window_days).std() * np.sqrt(252)
        
        # 滚动夏普比率
        rolling_metrics['sharpe_ratio'] = returns.rolling(window_days).apply(
            lambda x: (x.mean() - self.risk_free_rate/252) / x.std() * np.sqrt(252)
        )
        
        # 滚动最大回撤
        rolling_metrics['max_drawdown'] = equity_curve.rolling(window_days).apply(
            lambda x: self._calculate_max_drawdown(x)
        )
        
        return rolling_metrics.dropna()
    
    def compare_strategies(self, strategy_metrics: Dict[str, BacktestMetrics]) -> pd.DataFrame:
        """比较多个策略的指标
        
        Args:
            strategy_metrics: 策略指标字典
            
        Returns:
            pd.DataFrame: 策略比较表
        """
        if not strategy_metrics:
            return pd.DataFrame()
        
        comparison_data = []
        
        for strategy_name, metrics in strategy_metrics.items():
            row = {
                '策略': strategy_name,
                '总收益率': f"{metrics.total_return:.2%}",
                '年化收益率': f"{metrics.annual_return:.2%}",
                '波动率': f"{metrics.volatility:.2%}",
                '夏普比率': f"{metrics.sharpe_ratio:.4f}",
                '最大回撤': f"{metrics.max_drawdown:.2%}",
                '卡玛比率': f"{metrics.calmar_ratio:.4f}",
                '胜率': f"{metrics.trading_stats.win_rate:.2%}",
                '盈利因子': f"{metrics.trading_stats.profit_factor:.4f}"
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_metric_definition(self, metric_name: str) -> Optional[MetricDefinition]:
        """获取指标定义
        
        Args:
            metric_name: 指标名称
            
        Returns:
            Optional[MetricDefinition]: 指标定义
        """
        return self.metric_definitions.get(metric_name)
    
    def list_available_metrics(self) -> List[str]:
        """列出可用的指标
        
        Returns:
            List[str]: 指标名称列表
        """
        return list(self.metric_definitions.keys())
    
    def __str__(self) -> str:
        return f"MetricsCalculator(risk_free_rate={self.risk_free_rate}, metrics={len(self.metric_definitions)})"
    
    def __repr__(self) -> str:
        return self.__str__()