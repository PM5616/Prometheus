"""Performance Analyzer Module

性能分析模块，负责分析和计算投资组合的各种性能指标。
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger

from ..common.exceptions import (
    ValidationError,
    DataError,
    PerformanceCalculationError
)


class PerformanceThreshold(Enum):
    """性能阈值枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    var_95: float
    cvar_95: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: 无风险利率（年化）
        """
        self.risk_free_rate = risk_free_rate
        self._returns_cache: Dict[str, pd.Series] = {}
        
    def calculate_returns(self, prices: Union[pd.Series, List[float]], method: str = 'simple') -> pd.Series:
        """计算收益率
        
        Args:
            prices: 价格序列
            method: 计算方法 ('simple' 或 'log')
            
        Returns:
            pd.Series: 收益率序列
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)
        
        if len(prices) < 2:
            raise ValidationError("Price series must have at least 2 data points")
        
        if method == 'simple':
            returns = prices.pct_change().dropna()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValidationError(f"Unknown return calculation method: {method}")
        
        return returns
    
    def calculate_total_return(self, returns: pd.Series) -> float:
        """计算总收益率"""
        return float((1 + returns).prod() - 1)
    
    def calculate_annualized_return(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """计算年化收益率"""
        if len(returns) == 0:
            return 0.0
        
        total_return = self.calculate_total_return(returns)
        years = len(returns) / periods_per_year
        
        if years <= 0:
            return 0.0
        
        return float((1 + total_return) ** (1 / years) - 1)
    
    def calculate_volatility(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """计算年化波动率"""
        if len(returns) == 0:
            return 0.0
        
        return float(returns.std() * np.sqrt(periods_per_year))
    
    def calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """计算夏普比率"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / periods_per_year
        
        if excess_returns.std() == 0:
            return 0.0
        
        return float(excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year))
    
    def calculate_max_drawdown(self, returns: pd.Series) -> Tuple[float, int, int]:
        """计算最大回撤
        
        Returns:
            Tuple[float, int, int]: (最大回撤, 开始位置, 结束位置)
        """
        if len(returns) == 0:
            return 0.0, 0, 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_end = drawdown.idxmin()
        max_dd_start = cumulative.loc[:max_dd_end].idxmax()
        
        return float(abs(max_dd)), max_dd_start, max_dd_end
    
    def calculate_calmar_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """计算卡尔马比率"""
        annualized_return = self.calculate_annualized_return(returns, periods_per_year)
        max_drawdown, _, _ = self.calculate_max_drawdown(returns)
        
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / max_drawdown
    
    def calculate_sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """计算索提诺比率"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
        
        if downside_deviation == 0:
            return 0.0
        
        return float(excess_returns.mean() * np.sqrt(periods_per_year) / downside_deviation)
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """计算胜率"""
        if len(returns) == 0:
            return 0.0
        
        winning_trades = (returns > 0).sum()
        return float(winning_trades / len(returns))
    
    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """计算盈利因子"""
        if len(returns) == 0:
            return 0.0
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return float(gross_profit / gross_loss)
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算风险价值（VaR）"""
        if len(returns) == 0:
            return 0.0
        
        return float(np.percentile(returns, (1 - confidence_level) * 100))
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算条件风险价值（CVaR）"""
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        return float(returns[returns <= var].mean())
    
    def calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算贝塔系数"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # 对齐数据
        aligned_data = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        covariance = aligned_data['portfolio'].cov(aligned_data['benchmark'])
        benchmark_variance = aligned_data['benchmark'].var()
        
        if benchmark_variance == 0:
            return 0.0
        
        return float(covariance / benchmark_variance)
    
    def calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series, 
                      periods_per_year: int = 252) -> float:
        """计算阿尔法系数"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        portfolio_return = self.calculate_annualized_return(returns, periods_per_year)
        benchmark_return = self.calculate_annualized_return(benchmark_returns, periods_per_year)
        beta = self.calculate_beta(returns, benchmark_returns)
        
        return portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
    
    def calculate_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series,
                               periods_per_year: int = 252) -> float:
        """计算跟踪误差"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # 对齐数据
        aligned_data = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        return float(excess_returns.std() * np.sqrt(periods_per_year))
    
    def calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series,
                                  periods_per_year: int = 252) -> float:
        """计算信息比率"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        alpha = self.calculate_alpha(returns, benchmark_returns, periods_per_year)
        tracking_error = self.calculate_tracking_error(returns, benchmark_returns, periods_per_year)
        
        if tracking_error == 0:
            return 0.0
        
        return alpha / tracking_error
    
    def analyze_performance(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                          periods_per_year: int = 252) -> PerformanceMetrics:
        """综合性能分析
        
        Args:
            returns: 投资组合收益率序列
            benchmark_returns: 基准收益率序列（可选）
            periods_per_year: 每年的交易周期数
            
        Returns:
            PerformanceMetrics: 性能指标对象
        """
        try:
            # 基础指标
            total_return = self.calculate_total_return(returns)
            annualized_return = self.calculate_annualized_return(returns, periods_per_year)
            volatility = self.calculate_volatility(returns, periods_per_year)
            sharpe_ratio = self.calculate_sharpe_ratio(returns, periods_per_year)
            max_drawdown, _, _ = self.calculate_max_drawdown(returns)
            calmar_ratio = self.calculate_calmar_ratio(returns, periods_per_year)
            sortino_ratio = self.calculate_sortino_ratio(returns, periods_per_year)
            win_rate = self.calculate_win_rate(returns)
            profit_factor = self.calculate_profit_factor(returns)
            
            # 计算平均盈亏
            winning_returns = returns[returns > 0]
            losing_returns = returns[returns < 0]
            avg_win = float(winning_returns.mean()) if len(winning_returns) > 0 else 0.0
            avg_loss = float(losing_returns.mean()) if len(losing_returns) > 0 else 0.0
            
            # 计算连续盈亏
            max_consecutive_wins = self._calculate_max_consecutive(returns > 0)
            max_consecutive_losses = self._calculate_max_consecutive(returns < 0)
            
            # 风险指标
            var_95 = self.calculate_var(returns, 0.95)
            cvar_95 = self.calculate_cvar(returns, 0.95)
            
            # 相对指标（如果有基准）
            beta = None
            alpha = None
            information_ratio = None
            tracking_error = None
            
            if benchmark_returns is not None:
                beta = self.calculate_beta(returns, benchmark_returns)
                alpha = self.calculate_alpha(returns, benchmark_returns, periods_per_year)
                information_ratio = self.calculate_information_ratio(returns, benchmark_returns, periods_per_year)
                tracking_error = self.calculate_tracking_error(returns, benchmark_returns, periods_per_year)
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                var_95=var_95,
                cvar_95=cvar_95,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                tracking_error=tracking_error
            )
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            raise PerformanceCalculationError(f"Failed to analyze performance: {e}")
    
    def _calculate_max_consecutive(self, condition_series: pd.Series) -> int:
        """计算最大连续次数"""
        if len(condition_series) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in condition_series:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def generate_performance_report(self, metrics: PerformanceMetrics) -> Dict:
        """生成性能报告"""
        report = {
            'return_metrics': {
                'total_return': f"{metrics.total_return:.2%}",
                'annualized_return': f"{metrics.annualized_return:.2%}",
                'volatility': f"{metrics.volatility:.2%}"
            },
            'risk_adjusted_metrics': {
                'sharpe_ratio': f"{metrics.sharpe_ratio:.3f}",
                'sortino_ratio': f"{metrics.sortino_ratio:.3f}",
                'calmar_ratio': f"{metrics.calmar_ratio:.3f}"
            },
            'risk_metrics': {
                'max_drawdown': f"{metrics.max_drawdown:.2%}",
                'var_95': f"{metrics.var_95:.2%}",
                'cvar_95': f"{metrics.cvar_95:.2%}"
            },
            'trading_metrics': {
                'win_rate': f"{metrics.win_rate:.2%}",
                'profit_factor': f"{metrics.profit_factor:.3f}",
                'avg_win': f"{metrics.avg_win:.2%}",
                'avg_loss': f"{metrics.avg_loss:.2%}",
                'max_consecutive_wins': metrics.max_consecutive_wins,
                'max_consecutive_losses': metrics.max_consecutive_losses
            }
        }
        
        # 添加相对指标（如果可用）
        if metrics.beta is not None:
            report['relative_metrics'] = {
                'beta': f"{metrics.beta:.3f}",
                'alpha': f"{metrics.alpha:.2%}" if metrics.alpha else "N/A",
                'information_ratio': f"{metrics.information_ratio:.3f}" if metrics.information_ratio else "N/A",
                'tracking_error': f"{metrics.tracking_error:.2%}" if metrics.tracking_error else "N/A"
            }
        
        return report


class AnomalyDetection:
    """异常检测器"""
    
    def __init__(self, window_size: int = 30, threshold: float = 2.0):
        """
        Args:
            window_size: 滑动窗口大小
            threshold: 异常检测阈值（标准差倍数）
        """
        self.window_size = window_size
        self.threshold = threshold
    
    def detect_return_anomalies(self, returns: pd.Series) -> pd.Series:
        """检测收益率异常
        
        Args:
            returns: 收益率序列
            
        Returns:
            pd.Series: 异常标记（True为异常）
        """
        if len(returns) < self.window_size:
            return pd.Series([False] * len(returns), index=returns.index)
        
        # 计算滚动均值和标准差
        rolling_mean = returns.rolling(window=self.window_size).mean()
        rolling_std = returns.rolling(window=self.window_size).std()
        
        # 计算Z分数
        z_scores = (returns - rolling_mean) / rolling_std
        
        # 标记异常值
        anomalies = abs(z_scores) > self.threshold
        
        return anomalies
    
    def detect_volatility_anomalies(self, returns: pd.Series) -> pd.Series:
        """检测波动率异常
        
        Args:
            returns: 收益率序列
            
        Returns:
            pd.Series: 异常标记（True为异常）
        """
        if len(returns) < self.window_size:
            return pd.Series([False] * len(returns), index=returns.index)
        
        # 计算滚动波动率
        rolling_vol = returns.rolling(window=self.window_size).std()
        
        # 计算波动率的滚动均值和标准差
        vol_mean = rolling_vol.rolling(window=self.window_size).mean()
        vol_std = rolling_vol.rolling(window=self.window_size).std()
        
        # 计算Z分数
        z_scores = (rolling_vol - vol_mean) / vol_std
        
        # 标记异常值
        anomalies = abs(z_scores) > self.threshold
        
        return anomalies.fillna(False)
    
    def detect_drawdown_anomalies(self, prices: pd.Series) -> pd.Series:
        """检测回撤异常
        
        Args:
            prices: 价格序列
            
        Returns:
            pd.Series: 异常标记（True为异常）
        """
        if len(prices) < self.window_size:
            return pd.Series([False] * len(prices), index=prices.index)
        
        # 计算回撤
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max
        
        # 计算回撤的滚动统计
        dd_mean = drawdown.rolling(window=self.window_size).mean()
        dd_std = drawdown.rolling(window=self.window_size).std()
        
        # 计算Z分数
        z_scores = (drawdown - dd_mean) / dd_std
        
        # 标记异常值（主要关注异常大的回撤）
        anomalies = z_scores < -self.threshold
        
        return anomalies.fillna(False)