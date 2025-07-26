"""Performance Analyzer Module

性能分析器模块，负责策略和投资组合的性能评估和分析。

主要功能：
- 收益率计算
- 风险指标分析
- 回撤分析
- 夏普比率等风险调整收益指标
- 基准比较
- 性能报告生成
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import math
import json

from ..common.logging import get_logger
from ..common.exceptions.strategy import PerformanceAnalysisError


class PerformanceMetric(Enum):
    """性能指标枚举"""
    TOTAL_RETURN = "total_return"                    # 总收益率
    ANNUALIZED_RETURN = "annualized_return"          # 年化收益率
    VOLATILITY = "volatility"                        # 波动率
    SHARPE_RATIO = "sharpe_ratio"                    # 夏普比率
    SORTINO_RATIO = "sortino_ratio"                  # 索提诺比率
    CALMAR_RATIO = "calmar_ratio"                    # 卡玛比率
    MAX_DRAWDOWN = "max_drawdown"                    # 最大回撤
    MAX_DRAWDOWN_DURATION = "max_drawdown_duration"  # 最大回撤持续时间
    WIN_RATE = "win_rate"                            # 胜率
    PROFIT_LOSS_RATIO = "profit_loss_ratio"          # 盈亏比
    VAR = "var"                                      # 风险价值
    CVAR = "cvar"                                    # 条件风险价值
    BETA = "beta"                                    # 贝塔系数
    ALPHA = "alpha"                                  # 阿尔法系数
    INFORMATION_RATIO = "information_ratio"          # 信息比率
    TRACKING_ERROR = "tracking_error"                # 跟踪误差


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 指标字典
        """
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'win_rate': self.win_rate,
            'profit_loss_ratio': self.profit_loss_ratio,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'beta': self.beta,
            'alpha': self.alpha,
            'information_ratio': self.information_ratio,
            'tracking_error': self.tracking_error
        }


@dataclass
class DrawdownInfo:
    """回撤信息数据类"""
    start_date: datetime
    end_date: datetime
    duration: int  # 天数
    max_drawdown: float
    recovery_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 回撤信息字典
        """
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'duration': self.duration,
            'max_drawdown': self.max_drawdown,
            'recovery_date': self.recovery_date.isoformat() if self.recovery_date else None
        }


@dataclass
class PerformanceReport:
    """性能报告数据类"""
    start_date: datetime
    end_date: datetime
    total_days: int
    trading_days: int
    metrics: PerformanceMetrics
    drawdowns: List[DrawdownInfo]
    monthly_returns: Dict[str, float]
    yearly_returns: Dict[str, float]
    benchmark_comparison: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 报告字典
        """
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_days': self.total_days,
            'trading_days': self.trading_days,
            'metrics': self.metrics.to_dict(),
            'drawdowns': [dd.to_dict() for dd in self.drawdowns],
            'monthly_returns': self.monthly_returns,
            'yearly_returns': self.yearly_returns,
            'benchmark_comparison': self.benchmark_comparison
        }


class PerformanceAnalyzer:
    """性能分析器
    
    负责计算和分析策略及投资组合的各种性能指标。
    """
    
    def __init__(self, config: Dict = None):
        """初始化性能分析器
        
        Args:
            config: 分析器配置
        """
        self.config = config or {}
        
        # 配置参数
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 无风险利率
        self.trading_days_per_year = self.config.get('trading_days_per_year', 252)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        
        # 日志记录
        self.logger = get_logger("PerformanceAnalyzer")
        
        # 缓存
        self._cache = {}
        self._cache_enabled = self.config.get('cache_enabled', True)
        
        self.logger.info("性能分析器初始化完成")
    
    def analyze_returns(self, returns: Union[pd.Series, List[float], np.ndarray],
                       benchmark_returns: Optional[Union[pd.Series, List[float], np.ndarray]] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> PerformanceReport:
        """分析收益率序列
        
        Args:
            returns: 收益率序列
            benchmark_returns: 基准收益率序列
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            PerformanceReport: 性能报告
        """
        try:
            # 转换为pandas Series
            if not isinstance(returns, pd.Series):
                returns = pd.Series(returns)
            
            if benchmark_returns is not None and not isinstance(benchmark_returns, pd.Series):
                benchmark_returns = pd.Series(benchmark_returns)
            
            # 设置日期索引
            if start_date and end_date:
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                if len(date_range) == len(returns):
                    returns.index = date_range
                    if benchmark_returns is not None:
                        benchmark_returns.index = date_range
            
            # 计算性能指标
            metrics = self._calculate_metrics(returns, benchmark_returns)
            
            # 计算回撤信息
            drawdowns = self._calculate_drawdowns(returns)
            
            # 计算月度和年度收益
            monthly_returns = self._calculate_monthly_returns(returns)
            yearly_returns = self._calculate_yearly_returns(returns)
            
            # 基准比较
            benchmark_comparison = None
            if benchmark_returns is not None:
                benchmark_comparison = self._compare_with_benchmark(returns, benchmark_returns)
            
            # 创建报告
            report = PerformanceReport(
                start_date=returns.index[0] if hasattr(returns.index, '__getitem__') else datetime.now(),
                end_date=returns.index[-1] if hasattr(returns.index, '__getitem__') else datetime.now(),
                total_days=len(returns),
                trading_days=len(returns.dropna()),
                metrics=metrics,
                drawdowns=drawdowns,
                monthly_returns=monthly_returns,
                yearly_returns=yearly_returns,
                benchmark_comparison=benchmark_comparison
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"分析收益率失败: {e}")
            raise PerformanceAnalysisError(f"分析收益率失败: {e}")
    
    def analyze_portfolio_value(self, portfolio_values: Union[pd.Series, List[float], np.ndarray],
                               benchmark_values: Optional[Union[pd.Series, List[float], np.ndarray]] = None) -> PerformanceReport:
        """分析投资组合价值序列
        
        Args:
            portfolio_values: 投资组合价值序列
            benchmark_values: 基准价值序列
            
        Returns:
            PerformanceReport: 性能报告
        """
        try:
            # 转换为pandas Series
            if not isinstance(portfolio_values, pd.Series):
                portfolio_values = pd.Series(portfolio_values)
            
            if benchmark_values is not None and not isinstance(benchmark_values, pd.Series):
                benchmark_values = pd.Series(benchmark_values)
            
            # 计算收益率
            returns = portfolio_values.pct_change().dropna()
            
            benchmark_returns = None
            if benchmark_values is not None:
                benchmark_returns = benchmark_values.pct_change().dropna()
            
            return self.analyze_returns(returns, benchmark_returns)
            
        except Exception as e:
            self.logger.error(f"分析投资组合价值失败: {e}")
            raise PerformanceAnalysisError(f"分析投资组合价值失败: {e}")
    
    def _calculate_metrics(self, returns: pd.Series, 
                          benchmark_returns: Optional[pd.Series] = None) -> PerformanceMetrics:
        """计算性能指标
        
        Args:
            returns: 收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            PerformanceMetrics: 性能指标
        """
        # 基本统计
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(returns)) - 1
        volatility = returns.std() * np.sqrt(self.trading_days_per_year)
        
        # 夏普比率
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days_per_year) if excess_returns.std() > 0 else 0
        
        # 索提诺比率
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.trading_days_per_year)
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 卡玛比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 最大回撤持续时间
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
        
        # 胜率
        win_rate = (returns > 0).sum() / len(returns)
        
        # 盈亏比
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = abs(losing_returns.mean()) if len(losing_returns) > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # VaR和CVaR
        var_95 = returns.quantile(1 - self.confidence_level)
        cvar_95 = returns[returns <= var_95].mean()
        
        # 与基准相关的指标
        beta = None
        alpha = None
        information_ratio = None
        tracking_error = None
        
        if benchmark_returns is not None:
            # 贝塔系数
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # 阿尔法系数
            benchmark_annualized_return = (1 + benchmark_returns).prod() ** (self.trading_days_per_year / len(benchmark_returns)) - 1
            alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_annualized_return - self.risk_free_rate))
            
            # 跟踪误差
            tracking_error = (returns - benchmark_returns).std() * np.sqrt(self.trading_days_per_year)
            
            # 信息比率
            excess_return = annualized_return - benchmark_annualized_return
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error
        )
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """计算最大回撤持续时间
        
        Args:
            drawdown: 回撤序列
            
        Returns:
            int: 最大回撤持续时间（天数）
        """
        is_drawdown = drawdown < 0
        drawdown_periods = []
        
        start = None
        for i, in_drawdown in enumerate(is_drawdown):
            if in_drawdown and start is None:
                start = i
            elif not in_drawdown and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        # 如果序列结束时仍在回撤中
        if start is not None:
            drawdown_periods.append(len(is_drawdown) - start)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_drawdowns(self, returns: pd.Series) -> List[DrawdownInfo]:
        """计算回撤信息
        
        Args:
            returns: 收益率序列
            
        Returns:
            List[DrawdownInfo]: 回撤信息列表
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        drawdowns = []
        is_drawdown = drawdown < 0
        
        start = None
        max_dd = 0
        
        for i, (date, in_drawdown) in enumerate(zip(returns.index, is_drawdown)):
            if in_drawdown and start is None:
                start = i
                max_dd = drawdown.iloc[i]
            elif in_drawdown and start is not None:
                max_dd = min(max_dd, drawdown.iloc[i])
            elif not in_drawdown and start is not None:
                # 回撤结束
                start_date = returns.index[start]
                end_date = returns.index[i-1] if i > 0 else date
                duration = i - start
                
                # 查找恢复日期
                recovery_date = date if cumulative_returns.iloc[i] >= running_max.iloc[start] else None
                
                drawdowns.append(DrawdownInfo(
                    start_date=start_date,
                    end_date=end_date,
                    duration=duration,
                    max_drawdown=max_dd,
                    recovery_date=recovery_date
                ))
                
                start = None
                max_dd = 0
        
        # 如果序列结束时仍在回撤中
        if start is not None:
            start_date = returns.index[start]
            end_date = returns.index[-1]
            duration = len(returns) - start
            
            drawdowns.append(DrawdownInfo(
                start_date=start_date,
                end_date=end_date,
                duration=duration,
                max_drawdown=max_dd
            ))
        
        return drawdowns
    
    def _calculate_monthly_returns(self, returns: pd.Series) -> Dict[str, float]:
        """计算月度收益率
        
        Args:
            returns: 收益率序列
            
        Returns:
            Dict[str, float]: 月度收益率字典
        """
        if not hasattr(returns.index, 'to_period'):
            return {}
        
        try:
            monthly_returns = returns.groupby(returns.index.to_period('M')).apply(
                lambda x: (1 + x).prod() - 1
            )
            return {str(period): ret for period, ret in monthly_returns.items()}
        except:
            return {}
    
    def _calculate_yearly_returns(self, returns: pd.Series) -> Dict[str, float]:
        """计算年度收益率
        
        Args:
            returns: 收益率序列
            
        Returns:
            Dict[str, float]: 年度收益率字典
        """
        if not hasattr(returns.index, 'to_period'):
            return {}
        
        try:
            yearly_returns = returns.groupby(returns.index.to_period('Y')).apply(
                lambda x: (1 + x).prod() - 1
            )
            return {str(period): ret for period, ret in yearly_returns.items()}
        except:
            return {}
    
    def _compare_with_benchmark(self, returns: pd.Series, 
                               benchmark_returns: pd.Series) -> Dict:
        """与基准进行比较
        
        Args:
            returns: 策略收益率
            benchmark_returns: 基准收益率
            
        Returns:
            Dict: 比较结果
        """
        # 计算基准指标
        benchmark_metrics = self._calculate_metrics(benchmark_returns)
        
        # 计算超额收益
        excess_returns = returns - benchmark_returns
        excess_total_return = (1 + excess_returns).prod() - 1
        excess_annualized_return = (1 + excess_total_return) ** (self.trading_days_per_year / len(excess_returns)) - 1
        
        # 胜率（相对基准）
        outperformance_rate = (excess_returns > 0).sum() / len(excess_returns)
        
        return {
            'benchmark_metrics': benchmark_metrics.to_dict(),
            'excess_total_return': excess_total_return,
            'excess_annualized_return': excess_annualized_return,
            'outperformance_rate': outperformance_rate,
            'correlation': returns.corr(benchmark_returns)
        }
    
    def calculate_rolling_metrics(self, returns: pd.Series, 
                                 window: int = 252,
                                 metrics: List[PerformanceMetric] = None) -> pd.DataFrame:
        """计算滚动性能指标
        
        Args:
            returns: 收益率序列
            window: 滚动窗口大小
            metrics: 要计算的指标列表
            
        Returns:
            pd.DataFrame: 滚动指标数据框
        """
        if metrics is None:
            metrics = [PerformanceMetric.ANNUALIZED_RETURN, PerformanceMetric.VOLATILITY, 
                      PerformanceMetric.SHARPE_RATIO, PerformanceMetric.MAX_DRAWDOWN]
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        for metric in metrics:
            if metric == PerformanceMetric.ANNUALIZED_RETURN:
                rolling_returns = returns.rolling(window).apply(
                    lambda x: (1 + x).prod() ** (self.trading_days_per_year / len(x)) - 1
                )
                rolling_metrics[metric.value] = rolling_returns
            
            elif metric == PerformanceMetric.VOLATILITY:
                rolling_vol = returns.rolling(window).std() * np.sqrt(self.trading_days_per_year)
                rolling_metrics[metric.value] = rolling_vol
            
            elif metric == PerformanceMetric.SHARPE_RATIO:
                rolling_sharpe = returns.rolling(window).apply(
                    lambda x: (x.mean() - self.risk_free_rate / self.trading_days_per_year) / x.std() * np.sqrt(self.trading_days_per_year)
                )
                rolling_metrics[metric.value] = rolling_sharpe
            
            elif metric == PerformanceMetric.MAX_DRAWDOWN:
                rolling_dd = returns.rolling(window).apply(
                    lambda x: ((1 + x).cumprod() / (1 + x).cumprod().expanding().max() - 1).min()
                )
                rolling_metrics[metric.value] = rolling_dd
        
        return rolling_metrics
    
    def compare_strategies(self, strategy_returns: Dict[str, pd.Series],
                          benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """比较多个策略的性能
        
        Args:
            strategy_returns: 策略收益率字典
            benchmark_returns: 基准收益率
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        comparison_results = {}
        
        # 分析每个策略
        for strategy_name, returns in strategy_returns.items():
            report = self.analyze_returns(returns, benchmark_returns)
            comparison_results[strategy_name] = report.to_dict()
        
        # 创建比较表
        metrics_comparison = pd.DataFrame({
            strategy_name: result['metrics'] 
            for strategy_name, result in comparison_results.items()
        }).T
        
        # 排名
        rankings = {}
        for metric in metrics_comparison.columns:
            if metric in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 
                         'annualized_return', 'win_rate', 'profit_loss_ratio']:
                # 越大越好
                rankings[metric] = metrics_comparison[metric].rank(ascending=False)
            else:
                # 越小越好
                rankings[metric] = metrics_comparison[metric].rank(ascending=True)
        
        rankings_df = pd.DataFrame(rankings)
        
        return {
            'individual_results': comparison_results,
            'metrics_comparison': metrics_comparison.to_dict(),
            'rankings': rankings_df.to_dict(),
            'summary': {
                'best_sharpe': metrics_comparison['sharpe_ratio'].idxmax(),
                'best_return': metrics_comparison['annualized_return'].idxmax(),
                'lowest_drawdown': metrics_comparison['max_drawdown'].idxmax(),  # 最接近0
                'best_win_rate': metrics_comparison['win_rate'].idxmax()
            }
        }
    
    def generate_performance_summary(self, report: PerformanceReport) -> str:
        """生成性能摘要文本
        
        Args:
            report: 性能报告
            
        Returns:
            str: 性能摘要
        """
        summary = f"""
性能分析报告
================

分析期间: {report.start_date.strftime('%Y-%m-%d')} 至 {report.end_date.strftime('%Y-%m-%d')}
总天数: {report.total_days} 天
交易天数: {report.trading_days} 天

收益指标:
--------
总收益率: {report.metrics.total_return:.2%}
年化收益率: {report.metrics.annualized_return:.2%}
波动率: {report.metrics.volatility:.2%}

风险调整收益:
-----------
夏普比率: {report.metrics.sharpe_ratio:.3f}
索提诺比率: {report.metrics.sortino_ratio:.3f}
卡玛比率: {report.metrics.calmar_ratio:.3f}

风险指标:
--------
最大回撤: {report.metrics.max_drawdown:.2%}
最大回撤持续时间: {report.metrics.max_drawdown_duration} 天
VaR (95%): {report.metrics.var_95:.2%}
CVaR (95%): {report.metrics.cvar_95:.2%}

交易统计:
--------
胜率: {report.metrics.win_rate:.2%}
盈亏比: {report.metrics.profit_loss_ratio:.3f}
"""
        
        if report.metrics.beta is not None:
            summary += f"""
基准比较:
--------
贝塔系数: {report.metrics.beta:.3f}
阿尔法系数: {report.metrics.alpha:.2%}
信息比率: {report.metrics.information_ratio:.3f}
跟踪误差: {report.metrics.tracking_error:.2%}
"""
        
        if report.drawdowns:
            summary += f"""
主要回撤期间:
-----------
"""
            for i, dd in enumerate(report.drawdowns[:5]):  # 显示前5个回撤
                summary += f"{i+1}. {dd.start_date.strftime('%Y-%m-%d')} - {dd.end_date.strftime('%Y-%m-%d')}: {dd.max_drawdown:.2%} ({dd.duration}天)\n"
        
        return summary
    
    def export_report(self, report: PerformanceReport, 
                     file_path: str, format: str = 'json'):
        """导出性能报告
        
        Args:
            report: 性能报告
            file_path: 文件路径
            format: 导出格式 ('json', 'csv', 'txt')
        """
        try:
            if format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                # 导出指标为CSV
                metrics_df = pd.DataFrame([report.metrics.to_dict()])
                metrics_df.to_csv(file_path, index=False)
            
            elif format.lower() == 'txt':
                summary = self.generate_performance_summary(report)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
            
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"性能报告已导出到: {file_path}")
            
        except Exception as e:
            self.logger.error(f"导出性能报告失败: {e}")
            raise PerformanceAnalysisError(f"导出性能报告失败: {e}")
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self.logger.info("性能分析器缓存已清空")
    
    def get_config(self) -> Dict:
        """获取配置信息
        
        Returns:
            Dict: 配置字典
        """
        return self.config.copy()
    
    def update_config(self, config: Dict):
        """更新配置
        
        Args:
            config: 新配置
        """
        self.config.update(config)
        
        # 更新相关参数
        self.risk_free_rate = self.config.get('risk_free_rate', self.risk_free_rate)
        self.trading_days_per_year = self.config.get('trading_days_per_year', self.trading_days_per_year)
        self.confidence_level = self.config.get('confidence_level', self.confidence_level)
        
        self.logger.info("性能分析器配置已更新")
    
    def __str__(self) -> str:
        return f"PerformanceAnalyzer(risk_free_rate={self.risk_free_rate}, trading_days={self.trading_days_per_year})"
    
    def __repr__(self) -> str:
        return self.__str__()