"""Backtest Engine Module

回测引擎核心模块，负责执行历史数据回测。

主要功能：
- 回测环境管理
- 策略执行控制
- 数据回放
- 订单模拟
- 性能统计
- 结果分析
"""

import asyncio
import threading
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import uuid
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.common.logging import get_logger
from src.common.exceptions.backtest import BacktestError
from src.alpha_engine.base import BaseStrategy, StrategySignal
from src.execution.engine import ExecutionOrder, Trade, OrderStatus, OrderType, OrderSide
from src.risk_sentinel.manager import RiskManager
from src.datahub.data_manager import DataManager


class BacktestMode(Enum):
    """回测模式枚举"""
    SINGLE_STRATEGY = "single_strategy"      # 单策略回测
    MULTI_STRATEGY = "multi_strategy"        # 多策略回测
    PARAMETER_SCAN = "parameter_scan"        # 参数扫描
    MONTE_CARLO = "monte_carlo"              # 蒙特卡洛
    STRESS_TEST = "stress_test"              # 压力测试
    WALK_FORWARD = "walk_forward"            # 滚动回测


class BacktestStatus(Enum):
    """回测状态枚举"""
    PENDING = "pending"        # 待执行
    RUNNING = "running"        # 运行中
    PAUSED = "paused"          # 暂停
    COMPLETED = "completed"    # 完成
    CANCELLED = "cancelled"    # 取消
    ERROR = "error"            # 错误


@dataclass
class BacktestConfig:
    """回测配置"""
    # 基本配置
    backtest_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Backtest"
    description: str = ""
    
    # 时间配置
    start_date: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=365))
    end_date: datetime = field(default_factory=lambda: datetime.now())
    
    # 数据配置
    symbols: List[str] = field(default_factory=list)
    data_frequency: str = "1d"  # 数据频率: 1m, 5m, 15m, 1h, 1d
    
    # 资金配置
    initial_capital: float = 1000000.0  # 初始资金
    commission_rate: float = 0.001      # 手续费率
    slippage_rate: float = 0.001        # 滑点率
    
    # 执行配置
    execution_delay: int = 0            # 执行延迟(bar数)
    price_type: str = "close"           # 价格类型: open, high, low, close
    
    # 风险配置
    max_position_size: Optional[float] = None
    max_leverage: float = 1.0
    margin_requirement: float = 0.0
    
    # 回测模式配置
    mode: BacktestMode = BacktestMode.SINGLE_STRATEGY
    benchmark_symbol: Optional[str] = None
    
    # 性能配置
    enable_parallel: bool = False
    max_workers: int = 4
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 配置字典
        """
        return {
            'backtest_id': self.backtest_id,
            'name': self.name,
            'description': self.description,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'symbols': self.symbols,
            'data_frequency': self.data_frequency,
            'initial_capital': self.initial_capital,
            'commission_rate': self.commission_rate,
            'slippage_rate': self.slippage_rate,
            'execution_delay': self.execution_delay,
            'price_type': self.price_type,
            'max_position_size': self.max_position_size,
            'max_leverage': self.max_leverage,
            'margin_requirement': self.margin_requirement,
            'mode': self.mode.value,
            'benchmark_symbol': self.benchmark_symbol,
            'enable_parallel': self.enable_parallel,
            'max_workers': self.max_workers
        }


@dataclass
class BacktestPosition:
    """回测持仓信息"""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_position(self, trade_quantity: float, trade_price: float):
        """更新持仓
        
        Args:
            trade_quantity: 交易数量(正数买入，负数卖出)
            trade_price: 交易价格
        """
        if self.quantity == 0:
            # 新建仓位
            self.quantity = trade_quantity
            self.avg_price = trade_price
        elif (self.quantity > 0 and trade_quantity > 0) or (self.quantity < 0 and trade_quantity < 0):
            # 加仓
            total_value = self.quantity * self.avg_price + trade_quantity * trade_price
            self.quantity += trade_quantity
            self.avg_price = total_value / self.quantity if self.quantity != 0 else 0
        else:
            # 减仓或反向开仓
            if abs(trade_quantity) <= abs(self.quantity):
                # 部分平仓
                self.realized_pnl += (trade_price - self.avg_price) * abs(trade_quantity) * (1 if self.quantity > 0 else -1)
                self.quantity += trade_quantity
            else:
                # 完全平仓并反向开仓
                close_quantity = -self.quantity
                self.realized_pnl += (trade_price - self.avg_price) * abs(close_quantity) * (1 if self.quantity > 0 else -1)
                
                # 反向开仓
                remaining_quantity = trade_quantity + close_quantity
                self.quantity = remaining_quantity
                self.avg_price = trade_price if remaining_quantity != 0 else 0
    
    def update_market_value(self, current_price: float):
        """更新市值和未实现盈亏
        
        Args:
            current_price: 当前价格
        """
        self.market_value = self.quantity * current_price
        if self.quantity != 0:
            self.unrealized_pnl = (current_price - self.avg_price) * self.quantity
        else:
            self.unrealized_pnl = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 持仓字典
        """
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }


@dataclass
class BacktestAccount:
    """回测账户信息"""
    initial_capital: float
    cash: float
    positions: Dict[str, BacktestPosition] = field(default_factory=dict)
    
    # 统计信息
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # 历史记录
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    
    def get_position(self, symbol: str) -> BacktestPosition:
        """获取持仓
        
        Args:
            symbol: 交易品种
            
        Returns:
            BacktestPosition: 持仓信息
        """
        if symbol not in self.positions:
            self.positions[symbol] = BacktestPosition(symbol=symbol)
        return self.positions[symbol]
    
    def calculate_total_value(self) -> float:
        """计算总资产
        
        Returns:
            float: 总资产
        """
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + total_market_value
    
    def calculate_total_pnl(self) -> float:
        """计算总盈亏
        
        Returns:
            float: 总盈亏
        """
        realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return realized_pnl + unrealized_pnl
    
    def update_equity_curve(self, timestamp: datetime):
        """更新资产曲线
        
        Args:
            timestamp: 时间戳
        """
        total_value = self.calculate_total_value()
        self.equity_curve.append((timestamp, total_value))
    
    def record_trade(self, trade_info: Dict):
        """记录交易
        
        Args:
            trade_info: 交易信息
        """
        self.trade_history.append(trade_info)
        self.total_trades += 1
        
        if trade_info.get('pnl', 0) > 0:
            self.winning_trades += 1
        elif trade_info.get('pnl', 0) < 0:
            self.losing_trades += 1
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 账户字典
        """
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'total_value': self.calculate_total_value(),
            'total_pnl': self.calculate_total_pnl(),
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        }


@dataclass
class BacktestResult:
    """回测结果"""
    backtest_id: str
    config: BacktestConfig
    account: BacktestAccount
    
    # 执行信息
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: BacktestStatus = BacktestStatus.PENDING
    
    # 性能指标
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # 错误信息
    error_message: str = ""
    
    def calculate_metrics(self):
        """计算性能指标"""
        if len(self.account.equity_curve) < 2:
            return
        
        # 提取资产曲线
        equity_values = [value for _, value in self.account.equity_curve]
        
        # 计算收益率
        self.total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
        
        # 计算年化收益率
        days = (self.config.end_date - self.config.start_date).days
        if days > 0:
            self.annual_return = (1 + self.total_return) ** (365 / days) - 1
        
        # 计算波动率
        returns = np.diff(equity_values) / equity_values[:-1]
        self.volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
        
        # 计算夏普比率
        if self.volatility > 0:
            self.sharpe_ratio = self.annual_return / self.volatility
        
        # 计算最大回撤
        peak = equity_values[0]
        max_dd = 0
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        self.max_drawdown = max_dd
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 结果字典
        """
        return {
            'backtest_id': self.backtest_id,
            'config': self.config.to_dict(),
            'account': self.account.to_dict(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status.value,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'error_message': self.error_message
        }


class BacktestEngine:
    """回测引擎
    
    负责执行历史数据回测。
    """
    
    def __init__(self, data_manager: DataManager, risk_manager: Optional[RiskManager] = None):
        """初始化回测引擎
        
        Args:
            data_manager: 数据管理器
            risk_manager: 风险管理器
        """
        self.data_manager = data_manager
        self.risk_manager = risk_manager
        
        # 回测状态
        self.current_backtest: Optional[BacktestResult] = None
        self.strategies: Dict[str, BaseStrategy] = {}
        
        # 数据状态
        self.current_data: Dict[str, pd.DataFrame] = {}
        self.current_timestamp: Optional[datetime] = None
        self.current_bar_index: int = 0
        
        # 订单管理
        self.pending_orders: List[ExecutionOrder] = []
        self.order_history: List[Dict] = []
        
        # 回调函数
        self.on_bar_callbacks: List[Callable] = []
        self.on_trade_callbacks: List[Callable] = []
        self.on_signal_callbacks: List[Callable] = []
        
        # 线程控制
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # 日志记录
        self.logger = get_logger("BacktestEngine")
        
        self.logger.info("回测引擎初始化完成")
    
    def add_strategy(self, strategy: BaseStrategy, strategy_id: str = None) -> str:
        """添加策略
        
        Args:
            strategy: 策略实例
            strategy_id: 策略ID
            
        Returns:
            str: 策略ID
        """
        if strategy_id is None:
            strategy_id = str(uuid.uuid4())
        
        self.strategies[strategy_id] = strategy
        self.logger.info(f"添加策略: {strategy_id} - {strategy.__class__.__name__}")
        
        return strategy_id
    
    def remove_strategy(self, strategy_id: str):
        """移除策略
        
        Args:
            strategy_id: 策略ID
        """
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            self.logger.info(f"移除策略: {strategy_id}")
    
    async def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """运行回测
        
        Args:
            config: 回测配置
            
        Returns:
            BacktestResult: 回测结果
            
        Raises:
            BacktestError: 回测失败
        """
        if not self.strategies:
            raise BacktestError("没有添加策略")
        
        # 初始化回测结果
        account = BacktestAccount(
            initial_capital=config.initial_capital,
            cash=config.initial_capital
        )
        
        self.current_backtest = BacktestResult(
            backtest_id=config.backtest_id,
            config=config,
            account=account,
            start_time=datetime.now(),
            status=BacktestStatus.RUNNING
        )
        
        try:
            self._running = True
            
            # 加载数据
            await self._load_backtest_data(config)
            
            # 初始化策略
            await self._initialize_strategies(config)
            
            # 执行回测
            await self._execute_backtest(config)
            
            # 计算性能指标
            self.current_backtest.calculate_metrics()
            
            # 完成回测
            self.current_backtest.status = BacktestStatus.COMPLETED
            self.current_backtest.end_time = datetime.now()
            
            self.logger.info(f"回测完成: {config.backtest_id}")
            
        except Exception as e:
            self.current_backtest.status = BacktestStatus.ERROR
            self.current_backtest.error_message = str(e)
            self.current_backtest.end_time = datetime.now()
            
            self.logger.error(f"回测失败: {e}")
            raise BacktestError(f"回测执行失败: {e}")
        
        finally:
            self._running = False
        
        return self.current_backtest
    
    async def _load_backtest_data(self, config: BacktestConfig):
        """加载回测数据
        
        Args:
            config: 回测配置
        """
        self.logger.info("加载回测数据...")
        
        for symbol in config.symbols:
            try:
                # 获取历史数据
                data = await self.data_manager.get_historical_data(
                    symbol=symbol,
                    start_date=config.start_date,
                    end_date=config.end_date,
                    frequency=config.data_frequency
                )
                
                if data is not None and not data.empty:
                    self.current_data[symbol] = data
                    self.logger.info(f"加载数据: {symbol} - {len(data)}条记录")
                else:
                    self.logger.warning(f"无法获取数据: {symbol}")
                    
            except Exception as e:
                self.logger.error(f"加载数据失败 {symbol}: {e}")
                raise BacktestError(f"数据加载失败: {symbol} - {e}")
        
        if not self.current_data:
            raise BacktestError("没有可用的回测数据")
    
    async def _initialize_strategies(self, config: BacktestConfig):
        """初始化策略
        
        Args:
            config: 回测配置
        """
        self.logger.info("初始化策略...")
        
        for strategy_id, strategy in self.strategies.items():
            try:
                # 设置策略参数
                strategy.set_symbols(config.symbols)
                
                # 初始化策略
                await strategy.initialize()
                
                self.logger.info(f"策略初始化完成: {strategy_id}")
                
            except Exception as e:
                self.logger.error(f"策略初始化失败 {strategy_id}: {e}")
                raise BacktestError(f"策略初始化失败: {strategy_id} - {e}")
    
    async def _execute_backtest(self, config: BacktestConfig):
        """执行回测
        
        Args:
            config: 回测配置
        """
        self.logger.info("开始执行回测...")
        
        # 获取所有时间戳
        all_timestamps = set()
        for data in self.current_data.values():
            all_timestamps.update(data.index)
        
        timestamps = sorted(all_timestamps)
        
        if not timestamps:
            raise BacktestError("没有可用的时间戳")
        
        # 逐个时间点执行
        for i, timestamp in enumerate(timestamps):
            if not self._running:
                break
            
            self.current_timestamp = timestamp
            self.current_bar_index = i
            
            # 更新市场数据
            await self._update_market_data(timestamp)
            
            # 处理待执行订单
            await self._process_pending_orders(config)
            
            # 更新持仓市值
            self._update_positions_market_value()
            
            # 更新资产曲线
            self.current_backtest.account.update_equity_curve(timestamp)
            
            # 执行策略
            await self._execute_strategies(timestamp)
            
            # 通知Bar回调
            self._notify_bar_callbacks(timestamp)
        
        self.logger.info(f"回测执行完成，共处理 {len(timestamps)} 个时间点")
    
    async def _update_market_data(self, timestamp: datetime):
        """更新市场数据
        
        Args:
            timestamp: 时间戳
        """
        for symbol, data in self.current_data.items():
            if timestamp in data.index:
                bar_data = data.loc[timestamp]
                
                # 更新策略市场数据
                for strategy in self.strategies.values():
                    await strategy.on_market_data(symbol, bar_data.to_dict(), timestamp)
    
    async def _process_pending_orders(self, config: BacktestConfig):
        """处理待执行订单
        
        Args:
            config: 回测配置
        """
        executed_orders = []
        
        for order in self.pending_orders:
            try:
                # 检查执行条件
                if self._should_execute_order(order, config):
                    # 执行订单
                    trade = await self._execute_order(order, config)
                    if trade:
                        executed_orders.append(order)
                        
                        # 更新账户
                        self._update_account_with_trade(trade, config)
                        
                        # 通知交易回调
                        self._notify_trade_callbacks(trade)
                        
            except Exception as e:
                self.logger.error(f"订单执行失败 {order.order_id}: {e}")
                executed_orders.append(order)  # 移除失败的订单
        
        # 移除已执行的订单
        for order in executed_orders:
            self.pending_orders.remove(order)
    
    def _should_execute_order(self, order: ExecutionOrder, config: BacktestConfig) -> bool:
        """检查订单是否应该执行
        
        Args:
            order: 订单
            config: 回测配置
            
        Returns:
            bool: 是否应该执行
        """
        # 检查执行延迟
        if config.execution_delay > 0:
            order_bar_index = getattr(order, 'bar_index', self.current_bar_index)
            if self.current_bar_index - order_bar_index < config.execution_delay:
                return False
        
        # 检查价格条件
        if order.order_type == OrderType.LIMIT:
            current_price = self._get_current_price(order.symbol, config.price_type)
            if current_price is None:
                return False
            
            if order.side == OrderSide.BUY and current_price > order.price:
                return False
            elif order.side == OrderSide.SELL and current_price < order.price:
                return False
        
        return True
    
    async def _execute_order(self, order: ExecutionOrder, config: BacktestConfig) -> Optional[Trade]:
        """执行订单
        
        Args:
            order: 订单
            config: 回测配置
            
        Returns:
            Optional[Trade]: 成交信息
        """
        # 获取执行价格
        execution_price = self._get_execution_price(order, config)
        if execution_price is None:
            return None
        
        # 计算滑点
        slippage = execution_price * config.slippage_rate
        if order.side == OrderSide.BUY:
            execution_price += slippage
        else:
            execution_price -= slippage
        
        # 计算手续费
        commission = order.quantity * execution_price * config.commission_rate
        
        # 创建成交记录
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=commission,
            timestamp=self.current_timestamp
        )
        
        return trade
    
    def _get_execution_price(self, order: ExecutionOrder, config: BacktestConfig) -> Optional[float]:
        """获取执行价格
        
        Args:
            order: 订单
            config: 回测配置
            
        Returns:
            Optional[float]: 执行价格
        """
        if order.order_type == OrderType.MARKET:
            return self._get_current_price(order.symbol, config.price_type)
        elif order.order_type == OrderType.LIMIT:
            return order.price
        else:
            return self._get_current_price(order.symbol, config.price_type)
    
    def _get_current_price(self, symbol: str, price_type: str) -> Optional[float]:
        """获取当前价格
        
        Args:
            symbol: 交易品种
            price_type: 价格类型
            
        Returns:
            Optional[float]: 当前价格
        """
        if symbol not in self.current_data:
            return None
        
        data = self.current_data[symbol]
        if self.current_timestamp not in data.index:
            return None
        
        bar_data = data.loc[self.current_timestamp]
        return bar_data.get(price_type)
    
    def _update_account_with_trade(self, trade: Trade, config: BacktestConfig):
        """用成交更新账户
        
        Args:
            trade: 成交信息
            config: 回测配置
        """
        account = self.current_backtest.account
        
        # 更新持仓
        position = account.get_position(trade.symbol)
        
        # 计算交易数量（买入为正，卖出为负）
        trade_quantity = trade.quantity if trade.side == OrderSide.BUY else -trade.quantity
        
        # 更新持仓
        position.update_position(trade_quantity, trade.price)
        
        # 更新现金
        trade_value = trade.quantity * trade.price
        if trade.side == OrderSide.BUY:
            account.cash -= (trade_value + trade.commission)
        else:
            account.cash += (trade_value - trade.commission)
        
        # 更新统计
        account.total_commission += trade.commission
        account.total_slippage += abs(trade.price * config.slippage_rate * trade.quantity)
        
        # 记录交易
        trade_info = {
            'timestamp': trade.timestamp,
            'symbol': trade.symbol,
            'side': trade.side.value,
            'quantity': trade.quantity,
            'price': trade.price,
            'commission': trade.commission,
            'pnl': 0.0  # 将在平仓时计算
        }
        account.record_trade(trade_info)
        
        self.logger.debug(f"账户更新: {trade.symbol} {trade.side.value} {trade.quantity}@{trade.price}")
    
    def _update_positions_market_value(self):
        """更新持仓市值"""
        account = self.current_backtest.account
        
        for symbol, position in account.positions.items():
            current_price = self._get_current_price(symbol, "close")
            if current_price is not None:
                position.update_market_value(current_price)
    
    async def _execute_strategies(self, timestamp: datetime):
        """执行策略
        
        Args:
            timestamp: 时间戳
        """
        for strategy_id, strategy in self.strategies.items():
            try:
                # 执行策略逻辑
                signals = await strategy.on_bar(timestamp)
                
                # 处理策略信号
                if signals:
                    for signal in signals:
                        await self._process_strategy_signal(signal, strategy_id)
                        
            except Exception as e:
                self.logger.error(f"策略执行失败 {strategy_id}: {e}")
    
    async def _process_strategy_signal(self, signal: StrategySignal, strategy_id: str):
        """处理策略信号
        
        Args:
            signal: 策略信号
            strategy_id: 策略ID
        """
        try:
            # 风险检查
            if self.risk_manager:
                risk_check = await self.risk_manager.check_signal_risk(signal)
                if not risk_check.approved:
                    self.logger.warning(f"信号被风险管理器拒绝: {risk_check.reason}")
                    return
            
            # 创建订单
            order = self._create_order_from_signal(signal, strategy_id)
            if order:
                # 添加到待执行队列
                order.bar_index = self.current_bar_index  # 记录订单创建时的bar索引
                self.pending_orders.append(order)
                
                # 通知信号回调
                self._notify_signal_callbacks(signal)
                
                self.logger.debug(f"处理策略信号: {signal.symbol} {signal.signal_type.value}")
                
        except Exception as e:
            self.logger.error(f"处理策略信号失败: {e}")
    
    def _create_order_from_signal(self, signal: StrategySignal, strategy_id: str) -> Optional[ExecutionOrder]:
        """从策略信号创建订单
        
        Args:
            signal: 策略信号
            strategy_id: 策略ID
            
        Returns:
            Optional[ExecutionOrder]: 订单
        """
        from src.alpha_engine.signal import SignalType
        
        # 确定订单方向
        if signal.signal_type == SignalType.BUY:
            side = OrderSide.BUY
        elif signal.signal_type == SignalType.SELL:
            side = OrderSide.SELL
        else:
            return None
        
        # 确定订单类型和价格
        order_type = OrderType.MARKET
        price = None
        
        if hasattr(signal, 'price') and signal.price:
            order_type = OrderType.LIMIT
            price = signal.price
        
        # 确定订单数量
        quantity = getattr(signal, 'quantity', 100)  # 默认数量
        
        # 创建订单
        order = ExecutionOrder(
            order_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            strategy_id=strategy_id,
            timestamp=self.current_timestamp
        )
        
        return order
    
    def _notify_bar_callbacks(self, timestamp: datetime):
        """通知Bar回调
        
        Args:
            timestamp: 时间戳
        """
        for callback in self.on_bar_callbacks:
            try:
                callback(timestamp, self.current_backtest)
            except Exception as e:
                self.logger.error(f"Bar回调执行失败: {e}")
    
    def _notify_trade_callbacks(self, trade: Trade):
        """通知交易回调
        
        Args:
            trade: 成交信息
        """
        for callback in self.on_trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                self.logger.error(f"交易回调执行失败: {e}")
    
    def _notify_signal_callbacks(self, signal: StrategySignal):
        """通知信号回调
        
        Args:
            signal: 策略信号
        """
        for callback in self.on_signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                self.logger.error(f"信号回调执行失败: {e}")
    
    def add_bar_callback(self, callback: Callable):
        """添加Bar回调
        
        Args:
            callback: 回调函数
        """
        self.on_bar_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable):
        """添加交易回调
        
        Args:
            callback: 回调函数
        """
        self.on_trade_callbacks.append(callback)
    
    def add_signal_callback(self, callback: Callable):
        """添加信号回调
        
        Args:
            callback: 回调函数
        """
        self.on_signal_callbacks.append(callback)
    
    def stop_backtest(self):
        """停止回测"""
        self._running = False
        if self.current_backtest:
            self.current_backtest.status = BacktestStatus.CANCELLED
            self.current_backtest.end_time = datetime.now()
        
        self.logger.info("回测已停止")
    
    def get_current_result(self) -> Optional[BacktestResult]:
        """获取当前回测结果
        
        Returns:
            Optional[BacktestResult]: 回测结果
        """
        return self.current_backtest
    
    def cleanup(self):
        """清理资源"""
        self._running = False
        self._executor.shutdown(wait=True)
        
        self.current_backtest = None
        self.current_data.clear()
        self.pending_orders.clear()
        self.order_history.clear()
        
        self.logger.info("回测引擎资源清理完成")
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 引擎字典
        """
        return {
            'strategies': list(self.strategies.keys()),
            'current_backtest': self.current_backtest.to_dict() if self.current_backtest else None,
            'current_timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
            'current_bar_index': self.current_bar_index,
            'pending_orders': len(self.pending_orders),
            'running': self._running
        }
    
    def __str__(self) -> str:
        return f"BacktestEngine(strategies={len(self.strategies)}, running={self._running})"
    
    def __repr__(self) -> str:
        return self.__str__()