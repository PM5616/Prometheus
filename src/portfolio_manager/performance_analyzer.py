"""Performance Analyzer

绩效分析器，负责计算和分析投资组合的绩效指标
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
from enum import Enum

from src.common.exceptions.base import PrometheusException
from src.common.models.base import BaseModel
from .models import Portfolio, Position, PerformanceMetrics


class PerformancePeriod(Enum):
    """绩效分析周期"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    INCEPTION = "inception"


class BenchmarkType(Enum):
    """基准类型"""
    MARKET_INDEX = "market_index"  # 市场指数
    RISK_FREE = "risk_free"  # 无风险利率
    CUSTOM = "custom"  # 自定义基准


class PerformanceRecord:
    """绩效记录"""
    
    def __init__(
        self,
        timestamp: datetime,
        portfolio_value: float,
        cash: float,
        positions_value: float,
        unrealized_pnl: float,
        realized_pnl: float,
        daily_return: float = 0.0,
        cumulative_return: float = 0.0,
        benchmark_return: float = 0.0
    ):
        self.timestamp = timestamp
        self.portfolio_value = portfolio_value
        self.cash = cash
        self.positions_value = positions_value
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        self.daily_return = daily_return
        self.cumulative_return = cumulative_return
        self.benchmark_return = benchmark_return
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions_value': self.positions_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'daily_return': self.daily_return,
            'cumulative_return': self.cumulative_return,
            'benchmark_return': self.benchmark_return
        }


class PerformanceAnalyzerException(PrometheusException):
    """绩效分析异常"""
    pass


class PerformanceAnalyzer(BaseModel):
    """绩效分析器
    
    负责计算和分析投资组合的绩效指标，包括：
    1. 收益率计算（日收益率、累计收益率、年化收益率）
    2. 风险指标（波动率、最大回撤、VaR）
    3. 风险调整收益指标（夏普比率、索提诺比率、卡尔马比率）
    4. 基准比较分析
    5. 归因分析
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        benchmark_return: float = 0.03,  # 3%年化基准收益率
        risk_free_rate: float = 0.02,  # 2%无风险利率
        trading_days_per_year: int = 252
    ):
        super().__init__()
        self.initial_capital = initial_capital
        self.benchmark_return = benchmark_return
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 绩效历史记录
        self.performance_history: List[PerformanceRecord] = []
        
        # 缓存的绩效指标
        self.cached_metrics: Optional[PerformanceMetrics] = None
        self.cache_timestamp: Optional[datetime] = None
        
        # 基准数据
        self.benchmark_data: Dict[datetime, float] = {}
        
        self.logger.info("Performance Analyzer initialized")
    
    async def update_performance(
        self, 
        portfolio: Portfolio,
        timestamp: Optional[datetime] = None
    ) -> None:
        """更新绩效记录
        
        Args:
            portfolio: 当前投资组合
            timestamp: 时间戳
        """
        try:
            timestamp = timestamp or datetime.now()
            
            # 计算当前绩效数据
            portfolio_value = float(portfolio.total_value)
            cash = float(portfolio.cash)
            positions_value = sum(float(pos.market_value) for pos in portfolio.positions.values())
            unrealized_pnl = float(portfolio.unrealized_pnl)
            realized_pnl = float(portfolio.realized_pnl)
            
            # 计算收益率
            daily_return = 0.0
            cumulative_return = 0.0
            
            if self.performance_history:
                previous_value = self.performance_history[-1].portfolio_value
                if previous_value > 0:
                    daily_return = (portfolio_value - previous_value) / previous_value
            
            if self.initial_capital > 0:
                cumulative_return = (portfolio_value - self.initial_capital) / self.initial_capital
            
            # 获取基准收益率
            benchmark_return = self._get_benchmark_return(timestamp)
            
            # 创建绩效记录
            record = PerformanceRecord(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                cash=cash,
                positions_value=positions_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                benchmark_return=benchmark_return
            )
            
            # 添加到历史记录
            self.performance_history.append(record)
            
            # 限制历史记录长度
            if len(self.performance_history) > 10000:
                self.performance_history = self.performance_history[-10000:]
            
            # 清空缓存
            self.cached_metrics = None
            self.cache_timestamp = None
            
            self.logger.debug(f"Performance updated: value={portfolio_value:.2f}, return={daily_return:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")
            raise PerformanceAnalyzerException(f"Failed to update performance: {e}")
    
    async def calculate_performance_metrics(
        self, 
        period: PerformancePeriod = PerformancePeriod.INCEPTION
    ) -> PerformanceMetrics:
        """计算绩效指标
        
        Args:
            period: 分析周期
            
        Returns:
            绩效指标
        """
        try:
            # 检查缓存
            if (self.cached_metrics and 
                self.cache_timestamp and 
                datetime.now() - self.cache_timestamp < timedelta(minutes=5)):
                return self.cached_metrics
            
            if not self.performance_history:
                return self._create_empty_metrics()
            
            # 获取指定周期的数据
            period_data = self._get_period_data(period)
            
            if len(period_data) < 2:
                return self._create_empty_metrics()
            
            # 计算基础指标
            returns = [record.daily_return for record in period_data]
            portfolio_values = [record.portfolio_value for record in period_data]
            
            # 总收益率
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] if portfolio_values[0] > 0 else 0.0
            
            # 年化收益率
            days = len(period_data)
            annualized_return = (1 + total_return) ** (self.trading_days_per_year / days) - 1 if days > 0 else 0.0
            
            # 波动率
            volatility = np.std(returns) * np.sqrt(self.trading_days_per_year) if returns else 0.0
            
            # 最大回撤
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            # 夏普比率
            excess_returns = [r - self.risk_free_rate / self.trading_days_per_year for r in returns]
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days_per_year) if np.std(excess_returns) > 0 else 0.0
            
            # 索提诺比率
            downside_returns = [r for r in excess_returns if r < 0]
            downside_deviation = np.std(downside_returns) if downside_returns else 0.0
            sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(self.trading_days_per_year) if downside_deviation > 0 else 0.0
            
            # 卡尔马比率
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
            
            # 信息比率
            benchmark_returns = [record.benchmark_return for record in period_data]
            active_returns = [r - b for r, b in zip(returns, benchmark_returns)]
            tracking_error = np.std(active_returns) * np.sqrt(self.trading_days_per_year) if active_returns else 0.0
            information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(self.trading_days_per_year) if np.std(active_returns) > 0 else 0.0
            
            # 胜率
            positive_returns = [r for r in returns if r > 0]
            win_rate = len(positive_returns) / len(returns) if returns else 0.0
            
            # 盈亏比
            avg_win = np.mean(positive_returns) if positive_returns else 0.0
            negative_returns = [r for r in returns if r < 0]
            avg_loss = abs(np.mean(negative_returns)) if negative_returns else 0.0
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
            
            # VaR和CVaR
            var_95 = np.percentile(returns, 5) if returns else 0.0
            cvar_95 = np.mean([r for r in returns if r <= var_95]) if returns else 0.0
            
            # 贝塔值
            beta = self._calculate_beta(returns, benchmark_returns)
            
            # 阿尔法值
            alpha = annualized_return - (self.risk_free_rate + beta * (self.benchmark_return - self.risk_free_rate))
            
            # 创建绩效指标对象
            metrics = PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                information_ratio=information_ratio,
                win_rate=win_rate,
                profit_loss_ratio=profit_loss_ratio,
                var_95=abs(var_95),
                cvar_95=abs(cvar_95),
                beta=beta,
                alpha=alpha,
                tracking_error=tracking_error
            )
            
            # 缓存结果
            self.cached_metrics = metrics
            self.cache_timestamp = datetime.now()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return self._create_empty_metrics()
    
    async def get_performance_summary(
        self, 
        periods: List[PerformancePeriod] = None
    ) -> Dict[str, Dict[str, float]]:
        """获取绩效摘要
        
        Args:
            periods: 分析周期列表
            
        Returns:
            各周期的绩效指标字典
        """
        if periods is None:
            periods = [PerformancePeriod.DAILY, PerformancePeriod.MONTHLY, PerformancePeriod.INCEPTION]
        
        summary = {}
        
        for period in periods:
            try:
                metrics = await self.calculate_performance_metrics(period)
                summary[period.value] = asdict(metrics)
            except Exception as e:
                self.logger.error(f"Error calculating metrics for period {period.value}: {e}")
                summary[period.value] = asdict(self._create_empty_metrics())
        
        return summary
    
    async def get_returns_analysis(
        self, 
        period: PerformancePeriod = PerformancePeriod.INCEPTION
    ) -> Dict[str, Any]:
        """获取收益率分析"""
        try:
            period_data = self._get_period_data(period)
            
            if not period_data:
                return {}
            
            returns = [record.daily_return for record in period_data]
            
            analysis = {
                'mean_return': float(np.mean(returns)),
                'median_return': float(np.median(returns)),
                'std_return': float(np.std(returns)),
                'min_return': float(np.min(returns)),
                'max_return': float(np.max(returns)),
                'skewness': float(self._calculate_skewness(returns)),
                'kurtosis': float(self._calculate_kurtosis(returns)),
                'positive_days': len([r for r in returns if r > 0]),
                'negative_days': len([r for r in returns if r < 0]),
                'zero_days': len([r for r in returns if r == 0]),
                'total_days': len(returns)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in returns analysis: {e}")
            return {}
    
    async def get_drawdown_analysis(self) -> Dict[str, Any]:
        """获取回撤分析"""
        try:
            if not self.performance_history:
                return {}
            
            values = [record.portfolio_value for record in self.performance_history]
            
            # 计算回撤序列
            peak = np.maximum.accumulate(values)
            drawdown = (np.array(values) - peak) / peak
            
            # 找出回撤期间
            drawdown_periods = []
            in_drawdown = False
            start_idx = 0
            
            for i, dd in enumerate(drawdown):
                if dd < 0 and not in_drawdown:
                    # 开始回撤
                    in_drawdown = True
                    start_idx = i
                elif dd >= 0 and in_drawdown:
                    # 结束回撤
                    in_drawdown = False
                    end_idx = i - 1
                    
                    if end_idx > start_idx:
                        period_drawdown = drawdown[start_idx:end_idx+1]
                        drawdown_periods.append({
                            'start_date': self.performance_history[start_idx].timestamp,
                            'end_date': self.performance_history[end_idx].timestamp,
                            'duration_days': end_idx - start_idx + 1,
                            'max_drawdown': float(np.min(period_drawdown)),
                            'recovery_days': 0  # 需要额外计算
                        })
            
            # 处理最后一个未结束的回撤
            if in_drawdown:
                period_drawdown = drawdown[start_idx:]
                drawdown_periods.append({
                    'start_date': self.performance_history[start_idx].timestamp,
                    'end_date': self.performance_history[-1].timestamp,
                    'duration_days': len(period_drawdown),
                    'max_drawdown': float(np.min(period_drawdown)),
                    'recovery_days': -1  # 未恢复
                })
            
            analysis = {
                'max_drawdown': float(np.min(drawdown)),
                'current_drawdown': float(drawdown[-1]),
                'drawdown_periods': len(drawdown_periods),
                'avg_drawdown_duration': np.mean([p['duration_days'] for p in drawdown_periods]) if drawdown_periods else 0,
                'longest_drawdown_duration': max([p['duration_days'] for p in drawdown_periods]) if drawdown_periods else 0,
                'periods': drawdown_periods
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in drawdown analysis: {e}")
            return {}
    
    def _get_period_data(self, period: PerformancePeriod) -> List[PerformanceRecord]:
        """获取指定周期的数据"""
        if not self.performance_history:
            return []
        
        if period == PerformancePeriod.INCEPTION:
            return self.performance_history
        
        now = datetime.now()
        
        if period == PerformancePeriod.DAILY:
            cutoff = now - timedelta(days=1)
        elif period == PerformancePeriod.WEEKLY:
            cutoff = now - timedelta(weeks=1)
        elif period == PerformancePeriod.MONTHLY:
            cutoff = now - timedelta(days=30)
        elif period == PerformancePeriod.QUARTERLY:
            cutoff = now - timedelta(days=90)
        elif period == PerformancePeriod.YEARLY:
            cutoff = now - timedelta(days=365)
        else:
            return self.performance_history
        
        return [record for record in self.performance_history if record.timestamp >= cutoff]
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """计算最大回撤"""
        if len(values) < 2:
            return 0.0
        
        peak = np.maximum.accumulate(values)
        drawdown = (np.array(values) - peak) / peak
        return float(np.min(drawdown))
    
    def _calculate_beta(self, returns: List[float], benchmark_returns: List[float]) -> float:
        """计算贝塔值"""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 1.0
        
        try:
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            if benchmark_variance == 0:
                return 1.0
            
            return covariance / benchmark_variance
            
        except Exception:
            return 1.0
    
    def _calculate_skewness(self, returns: List[float]) -> float:
        """计算偏度"""
        if len(returns) < 3:
            return 0.0
        
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            skewness = np.mean([((r - mean_return) / std_return) ** 3 for r in returns])
            return skewness
            
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, returns: List[float]) -> float:
        """计算峰度"""
        if len(returns) < 4:
            return 0.0
        
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            kurtosis = np.mean([((r - mean_return) / std_return) ** 4 for r in returns]) - 3
            return kurtosis
            
        except Exception:
            return 0.0
    
    def _get_benchmark_return(self, timestamp: datetime) -> float:
        """获取基准收益率"""
        # 简化实现：使用固定的日基准收益率
        daily_benchmark = self.benchmark_return / self.trading_days_per_year
        return daily_benchmark
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """创建空的绩效指标"""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            win_rate=0.0,
            profit_loss_ratio=0.0,
            var_95=0.0,
            cvar_95=0.0,
            beta=1.0,
            alpha=0.0,
            tracking_error=0.0
        )
    
    def set_benchmark_data(self, benchmark_data: Dict[datetime, float]) -> None:
        """设置基准数据"""
        self.benchmark_data = benchmark_data.copy()
        self.logger.info(f"Benchmark data updated: {len(benchmark_data)} records")
    
    def get_performance_history(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """获取绩效历史记录"""
        filtered_history = self.performance_history
        
        if start_date:
            filtered_history = [r for r in filtered_history if r.timestamp >= start_date]
        
        if end_date:
            filtered_history = [r for r in filtered_history if r.timestamp <= end_date]
        
        return [record.to_dict() for record in filtered_history]
    
    def clear_history(self) -> None:
        """清空历史记录"""
        self.performance_history.clear()
        self.cached_metrics = None
        self.cache_timestamp = None
        self.logger.info("Performance history cleared")
    
    def export_performance_data(self) -> pd.DataFrame:
        """导出绩效数据为DataFrame"""
        if not self.performance_history:
            return pd.DataFrame()
        
        data = [record.to_dict() for record in self.performance_history]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df