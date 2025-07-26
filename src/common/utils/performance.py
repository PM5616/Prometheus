"""Performance Utilities Module

性能计算和分析工具模块。

主要功能：
- 收益率计算
- 风险指标计算
- 性能评估
- 基准比较
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List
from decimal import Decimal
import warnings
import time
from contextlib import contextmanager


def calculate_returns(prices: Union[pd.Series, np.ndarray], 
                     method: str = 'simple') -> Union[pd.Series, np.ndarray]:
    """计算收益率
    
    Args:
        prices: 价格序列
        method: 计算方法，'simple'或'log'
        
    Returns:
        收益率序列
    """
    if isinstance(prices, pd.Series):
        if method == 'simple':
            return prices.pct_change().dropna()
        elif method == 'log':
            return np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError(f"不支持的计算方法: {method}")
    else:
        prices = np.array(prices)
        if method == 'simple':
            return (prices[1:] - prices[:-1]) / prices[:-1]
        elif method == 'log':
            return np.log(prices[1:] / prices[:-1])
        else:
            raise ValueError(f"不支持的计算方法: {method}")


def calculate_volatility(returns: Union[pd.Series, np.ndarray], 
                        annualize: bool = True,
                        periods_per_year: int = 252) -> float:
    """计算波动率
    
    Args:
        returns: 收益率序列
        annualize: 是否年化
        periods_per_year: 每年的期数
        
    Returns:
        波动率
    """
    if isinstance(returns, pd.Series):
        vol = returns.std()
    else:
        vol = np.std(returns, ddof=1)
    
    if annualize:
        vol *= np.sqrt(periods_per_year)
    
    return float(vol)


def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray],
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年的期数
        
    Returns:
        夏普比率
    """
    if isinstance(returns, pd.Series):
        mean_return = returns.mean()
        std_return = returns.std()
    else:
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
    
    # 年化收益率
    annualized_return = mean_return * periods_per_year
    
    # 年化波动率
    annualized_vol = std_return * np.sqrt(periods_per_year)
    
    if annualized_vol == 0:
        return 0.0
    
    return (annualized_return - risk_free_rate) / annualized_vol


def calculate_max_drawdown(prices: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """计算最大回撤
    
    Args:
        prices: 价格序列
        
    Returns:
        包含最大回撤信息的字典
    """
    if isinstance(prices, pd.Series):
        # 计算累计最高价
        peak = prices.expanding().max()
        # 计算回撤
        drawdown = (prices - peak) / peak
        # 找到最大回撤
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # 找到最大回撤开始的峰值
        peak_idx = prices.loc[:max_dd_idx].idxmax()
        
        return {
            'max_drawdown': float(max_dd),
            'peak_date': peak_idx,
            'trough_date': max_dd_idx,
            'peak_value': float(prices.loc[peak_idx]),
            'trough_value': float(prices.loc[max_dd_idx]),
            'duration': max_dd_idx - peak_idx if hasattr(max_dd_idx - peak_idx, 'days') else None
        }
    else:
        prices = np.array(prices)
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        peak_idx = np.argmax(prices[:max_dd_idx + 1])
        
        return {
            'max_drawdown': float(max_dd),
            'peak_index': int(peak_idx),
            'trough_index': int(max_dd_idx),
            'peak_value': float(prices[peak_idx]),
            'trough_value': float(prices[max_dd_idx]),
            'duration': int(max_dd_idx - peak_idx)
        }


def calculate_calmar_ratio(returns: Union[pd.Series, np.ndarray],
                          prices: Union[pd.Series, np.ndarray],
                          periods_per_year: int = 252) -> float:
    """计算卡尔马比率
    
    Args:
        returns: 收益率序列
        prices: 价格序列
        periods_per_year: 每年的期数
        
    Returns:
        卡尔马比率
    """
    # 计算年化收益率
    if isinstance(returns, pd.Series):
        annualized_return = returns.mean() * periods_per_year
    else:
        annualized_return = np.mean(returns) * periods_per_year
    
    # 计算最大回撤
    max_dd_info = calculate_max_drawdown(prices)
    max_dd = abs(max_dd_info['max_drawdown'])
    
    if max_dd == 0:
        return float('inf') if annualized_return > 0 else 0.0
    
    return annualized_return / max_dd


def calculate_sortino_ratio(returns: Union[pd.Series, np.ndarray],
                           target_return: float = 0.0,
                           periods_per_year: int = 252) -> float:
    """计算索提诺比率
    
    Args:
        returns: 收益率序列
        target_return: 目标收益率（年化）
        periods_per_year: 每年的期数
        
    Returns:
        索提诺比率
    """
    if isinstance(returns, pd.Series):
        mean_return = returns.mean()
        # 只考虑负收益的标准差
        downside_returns = returns[returns < target_return / periods_per_year]
        if len(downside_returns) == 0:
            downside_deviation = 0
        else:
            downside_deviation = downside_returns.std()
    else:
        mean_return = np.mean(returns)
        downside_returns = returns[returns < target_return / periods_per_year]
        if len(downside_returns) == 0:
            downside_deviation = 0
        else:
            downside_deviation = np.std(downside_returns, ddof=1)
    
    # 年化收益率
    annualized_return = mean_return * periods_per_year
    
    # 年化下行偏差
    annualized_downside_deviation = downside_deviation * np.sqrt(periods_per_year)
    
    if annualized_downside_deviation == 0:
        return float('inf') if annualized_return > target_return else 0.0
    
    return (annualized_return - target_return) / annualized_downside_deviation


def calculate_var(returns: Union[pd.Series, np.ndarray],
                  confidence_level: float = 0.05) -> float:
    """计算风险价值(VaR)
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平
        
    Returns:
        VaR值
    """
    if isinstance(returns, pd.Series):
        return float(returns.quantile(confidence_level))
    else:
        return float(np.percentile(returns, confidence_level * 100))


def calculate_cvar(returns: Union[pd.Series, np.ndarray],
                   confidence_level: float = 0.05) -> float:
    """计算条件风险价值(CVaR)
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平
        
    Returns:
        CVaR值
    """
    var = calculate_var(returns, confidence_level)
    
    if isinstance(returns, pd.Series):
        tail_returns = returns[returns <= var]
    else:
        tail_returns = returns[returns <= var]
    
    if len(tail_returns) == 0:
        return var
    
    return float(np.mean(tail_returns))


def calculate_beta(returns: Union[pd.Series, np.ndarray],
                   benchmark_returns: Union[pd.Series, np.ndarray]) -> float:
    """计算贝塔系数
    
    Args:
        returns: 资产收益率序列
        benchmark_returns: 基准收益率序列
        
    Returns:
        贝塔系数
    """
    if isinstance(returns, pd.Series) and isinstance(benchmark_returns, pd.Series):
        # 确保索引对齐
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 0.0
        
        asset_returns = aligned_data.iloc[:, 0]
        bench_returns = aligned_data.iloc[:, 1]
        
        covariance = asset_returns.cov(bench_returns)
        benchmark_variance = bench_returns.var()
    else:
        returns = np.array(returns)
        benchmark_returns = np.array(benchmark_returns)
        
        # 确保长度一致
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        if min_len < 2:
            return 0.0
        
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)
    
    if benchmark_variance == 0:
        return 0.0
    
    return covariance / benchmark_variance


def calculate_alpha(returns: Union[pd.Series, np.ndarray],
                   benchmark_returns: Union[pd.Series, np.ndarray],
                   risk_free_rate: float = 0.0,
                   periods_per_year: int = 252) -> float:
    """计算阿尔法系数
    
    Args:
        returns: 资产收益率序列
        benchmark_returns: 基准收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年的期数
        
    Returns:
        阿尔法系数（年化）
    """
    beta = calculate_beta(returns, benchmark_returns)
    
    if isinstance(returns, pd.Series):
        asset_return = returns.mean() * periods_per_year
    else:
        asset_return = np.mean(returns) * periods_per_year
    
    if isinstance(benchmark_returns, pd.Series):
        benchmark_return = benchmark_returns.mean() * periods_per_year
    else:
        benchmark_return = np.mean(benchmark_returns) * periods_per_year
    
    # Alpha = 资产收益率 - (无风险利率 + Beta * (基准收益率 - 无风险利率))
    alpha = asset_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    
    return alpha


def calculate_information_ratio(returns: Union[pd.Series, np.ndarray],
                               benchmark_returns: Union[pd.Series, np.ndarray],
                               periods_per_year: int = 252) -> float:
    """计算信息比率
    
    Args:
        returns: 资产收益率序列
        benchmark_returns: 基准收益率序列
        periods_per_year: 每年的期数
        
    Returns:
        信息比率
    """
    if isinstance(returns, pd.Series) and isinstance(benchmark_returns, pd.Series):
        # 确保索引对齐
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 0.0
        
        asset_returns = aligned_data.iloc[:, 0]
        bench_returns = aligned_data.iloc[:, 1]
        
        excess_returns = asset_returns - bench_returns
        mean_excess_return = excess_returns.mean() * periods_per_year
        tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
    else:
        returns = np.array(returns)
        benchmark_returns = np.array(benchmark_returns)
        
        # 确保长度一致
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        if min_len < 2:
            return 0.0
        
        excess_returns = returns - benchmark_returns
        mean_excess_return = np.mean(excess_returns) * periods_per_year
        tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    return mean_excess_return / tracking_error


def calculate_performance_metrics(prices: Union[pd.Series, np.ndarray],
                                 benchmark_prices: Optional[Union[pd.Series, np.ndarray]] = None,
                                 risk_free_rate: float = 0.0,
                                 periods_per_year: int = 252) -> Dict[str, float]:
    """计算综合性能指标
    
    Args:
        prices: 价格序列
        benchmark_prices: 基准价格序列（可选）
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年的期数
        
    Returns:
        包含各种性能指标的字典
    """
    # 计算收益率
    returns = calculate_returns(prices)
    
    # 基本指标
    metrics = {
        'total_return': float((prices[-1] / prices[0] - 1) if isinstance(prices, (pd.Series, np.ndarray)) and len(prices) > 0 else 0),
        'annualized_return': float(np.mean(returns) * periods_per_year if len(returns) > 0 else 0),
        'volatility': calculate_volatility(returns, periods_per_year=periods_per_year),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'max_drawdown': calculate_max_drawdown(prices)['max_drawdown'],
        'calmar_ratio': calculate_calmar_ratio(returns, prices, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'var_5': calculate_var(returns, 0.05),
        'cvar_5': calculate_cvar(returns, 0.05)
    }
    
    # 如果有基准，计算相对指标
    if benchmark_prices is not None:
        benchmark_returns = calculate_returns(benchmark_prices)
        metrics.update({
            'beta': calculate_beta(returns, benchmark_returns),
            'alpha': calculate_alpha(returns, benchmark_returns, risk_free_rate, periods_per_year),
            'information_ratio': calculate_information_ratio(returns, benchmark_returns, periods_per_year)
        })
    
    return metrics


class PerformanceTimer:
    """性能计时器
    
    用于测量代码执行时间和性能分析。
    """
    
    def __init__(self, name: str = "Timer"):
        """初始化性能计时器
        
        Args:
            name: 计时器名称
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.lap_times = []
    
    def start(self) -> 'PerformanceTimer':
        """开始计时
        
        Returns:
            自身实例，支持链式调用
        """
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed_time = None
        self.lap_times = []
        return self
    
    def stop(self) -> float:
        """停止计时
        
        Returns:
            经过的时间（秒）
        """
        if self.start_time is None:
            raise ValueError("Timer has not been started")
        
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        return self.elapsed_time
    
    def lap(self) -> float:
        """记录分段时间
        
        Returns:
            从开始到现在的时间（秒）
        """
        if self.start_time is None:
            raise ValueError("Timer has not been started")
        
        current_time = time.perf_counter()
        lap_time = current_time - self.start_time
        self.lap_times.append(lap_time)
        return lap_time
    
    def reset(self) -> 'PerformanceTimer':
        """重置计时器
        
        Returns:
            自身实例，支持链式调用
        """
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.lap_times = []
        return self
    
    def get_elapsed_time(self) -> Optional[float]:
        """获取经过的时间
        
        Returns:
            经过的时间（秒），如果未停止则返回None
        """
        return self.elapsed_time
    
    def get_current_time(self) -> Optional[float]:
        """获取当前经过的时间
        
        Returns:
            从开始到现在的时间（秒），如果未开始则返回None
        """
        if self.start_time is None:
            return None
        
        current_time = time.perf_counter()
        return current_time - self.start_time
    
    def get_lap_times(self) -> List[float]:
        """获取所有分段时间
        
        Returns:
            分段时间列表
        """
        return self.lap_times.copy()
    
    def __enter__(self) -> 'PerformanceTimer':
        """上下文管理器入口"""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.stop()
    
    def __str__(self) -> str:
        """字符串表示"""
        if self.elapsed_time is not None:
            return f"{self.name}: {self.elapsed_time:.6f}s"
        elif self.start_time is not None:
            current = self.get_current_time()
            return f"{self.name}: {current:.6f}s (running)"
        else:
            return f"{self.name}: not started"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"PerformanceTimer(name='{self.name}', elapsed={self.elapsed_time})"


@contextmanager
def timer(name: str = "Operation"):
    """计时器上下文管理器
    
    Args:
        name: 操作名称
        
    Yields:
        PerformanceTimer实例
        
    Example:
        with timer("Database Query") as t:
            # 执行数据库查询
            pass
        print(f"Query took {t.get_elapsed_time():.3f}s")
    """
    perf_timer = PerformanceTimer(name)
    try:
        yield perf_timer.start()
    finally:
        perf_timer.stop()


def measure_time(func):
    """装饰器：测量函数执行时间
    
    Args:
        func: 要测量的函数
        
    Returns:
        装饰后的函数
        
    Example:
        @measure_time
        def slow_function():
            time.sleep(1)
            return "done"
    """
    def wrapper(*args, **kwargs):
        timer_name = f"{func.__module__}.{func.__name__}"
        with timer(timer_name) as t:
            result = func(*args, **kwargs)
        print(f"Function {timer_name} took {t.get_elapsed_time():.6f}s")
        return result
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper