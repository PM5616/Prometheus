"""Performance Tracker Module

投资组合性能跟踪模块，负责实时跟踪和记录投资组合的性能表现。
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
from collections import defaultdict
from loguru import logger

from ..common.exceptions import (
    ValidationError,
    DataError
)


@dataclass
class PerformanceSnapshot:
    """性能快照"""
    timestamp: datetime
    portfolio_value: Decimal
    cash_balance: Decimal
    total_return: Decimal
    daily_return: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    positions_count: int
    max_drawdown: Decimal = Decimal('0')
    sharpe_ratio: Optional[Decimal] = None
    volatility: Optional[Decimal] = None


@dataclass
class TradeRecord:
    """交易记录"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    price: Decimal
    commission: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    trade_id: Optional[str] = None


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self, initial_capital: Decimal):
        self.initial_capital = initial_capital
        self._snapshots: List[PerformanceSnapshot] = []
        self._trade_records: List[TradeRecord] = []
        self._daily_returns: List[Decimal] = []
        self._peak_value = initial_capital
        self._max_drawdown = Decimal('0')
        
        # 缓存计算结果
        self._cache: Dict = {}
        self._cache_timestamp: Optional[datetime] = None
        
    def add_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """添加性能快照"""
        # 计算最大回撤
        if snapshot.portfolio_value > self._peak_value:
            self._peak_value = snapshot.portfolio_value
        
        current_drawdown = (self._peak_value - snapshot.portfolio_value) / self._peak_value
        if current_drawdown > self._max_drawdown:
            self._max_drawdown = current_drawdown
        
        snapshot.max_drawdown = self._max_drawdown
        
        # 计算日收益率
        if len(self._snapshots) > 0:
            prev_value = self._snapshots[-1].portfolio_value
            if prev_value > 0:
                daily_return = (snapshot.portfolio_value - prev_value) / prev_value
                snapshot.daily_return = daily_return
                self._daily_returns.append(daily_return)
        
        self._snapshots.append(snapshot)
        
        # 清除缓存
        self._cache.clear()
        self._cache_timestamp = None
        
        logger.debug(f"Added performance snapshot: value={snapshot.portfolio_value}, return={snapshot.total_return}")
    
    def add_trade_record(self, trade: TradeRecord) -> None:
        """添加交易记录"""
        self._trade_records.append(trade)
        logger.debug(f"Added trade record: {trade.symbol} {trade.side} {trade.quantity} @ {trade.price}")
    
    def get_current_performance(self) -> Optional[PerformanceSnapshot]:
        """获取当前性能快照"""
        return self._snapshots[-1] if self._snapshots else None
    
    def get_performance_history(self, start_date: Optional[datetime] = None, 
                              end_date: Optional[datetime] = None) -> List[PerformanceSnapshot]:
        """获取性能历史"""
        filtered_snapshots = self._snapshots
        
        if start_date:
            filtered_snapshots = [s for s in filtered_snapshots if s.timestamp >= start_date]
        
        if end_date:
            filtered_snapshots = [s for s in filtered_snapshots if s.timestamp <= end_date]
        
        return filtered_snapshots
    
    def get_trade_history(self, symbol: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[TradeRecord]:
        """获取交易历史"""
        filtered_trades = self._trade_records
        
        if symbol:
            filtered_trades = [t for t in filtered_trades if t.symbol == symbol]
        
        if start_date:
            filtered_trades = [t for t in filtered_trades if t.timestamp >= start_date]
        
        if end_date:
            filtered_trades = [t for t in filtered_trades if t.timestamp <= end_date]
        
        return filtered_trades
    
    def calculate_total_return(self) -> Decimal:
        """计算总收益率"""
        if not self._snapshots:
            return Decimal('0')
        
        current_value = self._snapshots[-1].portfolio_value
        return (current_value - self.initial_capital) / self.initial_capital
    
    def calculate_annualized_return(self, periods_per_year: int = 252) -> Decimal:
        """计算年化收益率"""
        if len(self._snapshots) < 2:
            return Decimal('0')
        
        total_return = self.calculate_total_return()
        days = (self._snapshots[-1].timestamp - self._snapshots[0].timestamp).days
        
        if days <= 0:
            return Decimal('0')
        
        years = Decimal(days) / Decimal(365)
        return (1 + total_return) ** (1 / years) - 1
    
    def calculate_volatility(self, periods_per_year: int = 252) -> Decimal:
        """计算年化波动率"""
        if len(self._daily_returns) < 2:
            return Decimal('0')
        
        returns_series = pd.Series([float(r) for r in self._daily_returns])
        daily_vol = returns_series.std()
        
        return Decimal(str(daily_vol * (periods_per_year ** 0.5)))
    
    def calculate_sharpe_ratio(self, risk_free_rate: Decimal = Decimal('0.02'),
                             periods_per_year: int = 252) -> Decimal:
        """计算夏普比率"""
        if len(self._daily_returns) < 2:
            return Decimal('0')
        
        annualized_return = self.calculate_annualized_return(periods_per_year)
        volatility = self.calculate_volatility(periods_per_year)
        
        if volatility == 0:
            return Decimal('0')
        
        return (annualized_return - risk_free_rate) / volatility
    
    def calculate_max_drawdown(self) -> Tuple[Decimal, Optional[datetime], Optional[datetime]]:
        """计算最大回撤
        
        Returns:
            Tuple[Decimal, datetime, datetime]: (最大回撤, 开始时间, 结束时间)
        """
        if len(self._snapshots) < 2:
            return Decimal('0'), None, None
        
        peak_value = self._snapshots[0].portfolio_value
        max_dd = Decimal('0')
        peak_time = self._snapshots[0].timestamp
        trough_time = None
        
        for snapshot in self._snapshots[1:]:
            if snapshot.portfolio_value > peak_value:
                peak_value = snapshot.portfolio_value
                peak_time = snapshot.timestamp
            else:
                drawdown = (peak_value - snapshot.portfolio_value) / peak_value
                if drawdown > max_dd:
                    max_dd = drawdown
                    trough_time = snapshot.timestamp
        
        return max_dd, peak_time, trough_time
    
    def calculate_win_rate(self) -> Decimal:
        """计算胜率"""
        if not self._trade_records:
            return Decimal('0')
        
        profitable_trades = sum(1 for trade in self._trade_records if trade.realized_pnl > 0)
        return Decimal(profitable_trades) / Decimal(len(self._trade_records))
    
    def calculate_profit_factor(self) -> Decimal:
        """计算盈利因子"""
        if not self._trade_records:
            return Decimal('0')
        
        gross_profit = sum(trade.realized_pnl for trade in self._trade_records if trade.realized_pnl > 0)
        gross_loss = abs(sum(trade.realized_pnl for trade in self._trade_records if trade.realized_pnl < 0))
        
        if gross_loss == 0:
            return Decimal('inf') if gross_profit > 0 else Decimal('0')
        
        return gross_profit / gross_loss
    
    def get_monthly_returns(self) -> Dict[str, Decimal]:
        """获取月度收益率"""
        if len(self._snapshots) < 2:
            return {}
        
        monthly_returns = {}
        current_month = None
        month_start_value = None
        
        for snapshot in self._snapshots:
            month_key = snapshot.timestamp.strftime('%Y-%m')
            
            if current_month != month_key:
                if current_month and month_start_value:
                    # 计算上个月的收益率
                    prev_snapshot = self._snapshots[self._snapshots.index(snapshot) - 1]
                    monthly_return = (prev_snapshot.portfolio_value - month_start_value) / month_start_value
                    monthly_returns[current_month] = monthly_return
                
                current_month = month_key
                month_start_value = snapshot.portfolio_value
        
        # 处理最后一个月
        if current_month and month_start_value and self._snapshots:
            last_value = self._snapshots[-1].portfolio_value
            monthly_return = (last_value - month_start_value) / month_start_value
            monthly_returns[current_month] = monthly_return
        
        return monthly_returns
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        if not self._snapshots:
            return {}
        
        current_snapshot = self._snapshots[-1]
        max_dd, dd_start, dd_end = self.calculate_max_drawdown()
        
        return {
            'initial_capital': float(self.initial_capital),
            'current_value': float(current_snapshot.portfolio_value),
            'total_return': float(self.calculate_total_return()),
            'annualized_return': float(self.calculate_annualized_return()),
            'volatility': float(self.calculate_volatility()),
            'sharpe_ratio': float(self.calculate_sharpe_ratio()),
            'max_drawdown': float(max_dd),
            'win_rate': float(self.calculate_win_rate()),
            'profit_factor': float(self.calculate_profit_factor()),
            'total_trades': len(self._trade_records),
            'tracking_period_days': (current_snapshot.timestamp - self._snapshots[0].timestamp).days if len(self._snapshots) > 1 else 0
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """导出为DataFrame"""
        if not self._snapshots:
            return pd.DataFrame()
        
        data = []
        for snapshot in self._snapshots:
            data.append({
                'timestamp': snapshot.timestamp,
                'portfolio_value': float(snapshot.portfolio_value),
                'cash_balance': float(snapshot.cash_balance),
                'total_return': float(snapshot.total_return),
                'daily_return': float(snapshot.daily_return),
                'unrealized_pnl': float(snapshot.unrealized_pnl),
                'realized_pnl': float(snapshot.realized_pnl),
                'positions_count': snapshot.positions_count,
                'max_drawdown': float(snapshot.max_drawdown)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def reset(self) -> None:
        """重置跟踪器"""
        self._snapshots.clear()
        self._trade_records.clear()
        self._daily_returns.clear()
        self._peak_value = self.initial_capital
        self._max_drawdown = Decimal('0')
        self._cache.clear()
        self._cache_timestamp = None
        
        logger.info("Performance tracker reset")