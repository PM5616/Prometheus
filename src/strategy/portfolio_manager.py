"""Portfolio Manager Module

投资组合管理器模块，负责资产配置、风险管理和持仓管理。

主要功能：
- 投资组合构建和管理
- 资产配置优化
- 风险控制和管理
- 持仓跟踪和调整
- 绩效归因分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
import json
import copy

from ..common.logging import get_logger
from ..common.exceptions.strategy import PortfolioManagerError, RiskLimitExceededError
from .base import StrategySignal, SignalType


class AssetType(Enum):
    """资产类型枚举"""
    STOCK = "stock"              # 股票
    BOND = "bond"                # 债券
    COMMODITY = "commodity"      # 商品
    CURRENCY = "currency"        # 货币
    CRYPTO = "crypto"            # 加密货币
    DERIVATIVE = "derivative"    # 衍生品
    CASH = "cash"                # 现金
    OTHER = "other"              # 其他


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"            # 市价单
    LIMIT = "limit"              # 限价单
    STOP = "stop"                # 止损单
    STOP_LIMIT = "stop_limit"    # 止损限价单


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"          # 待处理
    SUBMITTED = "submitted"      # 已提交
    FILLED = "filled"            # 已成交
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    CANCELLED = "cancelled"      # 已取消
    REJECTED = "rejected"        # 已拒绝
    EXPIRED = "expired"          # 已过期


@dataclass
class Asset:
    """资产信息"""
    symbol: str
    name: str = ""
    asset_type: AssetType = AssetType.OTHER
    sector: str = ""
    currency: str = "USD"
    multiplier: float = 1.0
    min_tick: float = 0.01
    lot_size: float = 1.0
    margin_rate: float = 1.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 资产字典
        """
        return {
            'symbol': self.symbol,
            'name': self.name,
            'asset_type': self.asset_type.value,
            'sector': self.sector,
            'currency': self.currency,
            'multiplier': self.multiplier,
            'min_tick': self.min_tick,
            'lot_size': self.lot_size,
            'margin_rate': self.margin_rate,
            'metadata': self.metadata
        }


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    cost_basis: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        self.cost_basis = abs(self.quantity) * self.avg_price
        self.update_market_value()
    
    def update_market_value(self, price: Optional[float] = None):
        """更新市值
        
        Args:
            price: 当前价格
        """
        if price is not None:
            self.current_price = price
        
        self.market_value = self.quantity * self.current_price
        self.unrealized_pnl = self.market_value - (self.quantity * self.avg_price)
        self.last_updated = datetime.now()
    
    def add_trade(self, quantity: float, price: float) -> float:
        """添加交易
        
        Args:
            quantity: 交易数量（正数买入，负数卖出）
            price: 交易价格
            
        Returns:
            float: 实现盈亏
        """
        realized_pnl = 0.0
        
        # 如果是反向交易（平仓）
        if (self.quantity > 0 and quantity < 0) or (self.quantity < 0 and quantity > 0):
            close_quantity = min(abs(quantity), abs(self.quantity))
            if self.quantity > 0:
                close_quantity = min(quantity * -1, self.quantity)
                realized_pnl = close_quantity * (price - self.avg_price)
            else:
                close_quantity = min(quantity, abs(self.quantity))
                realized_pnl = close_quantity * (self.avg_price - price)
            
            self.realized_pnl += realized_pnl
        
        # 更新持仓
        old_quantity = self.quantity
        old_cost = old_quantity * self.avg_price
        new_cost = quantity * price
        
        self.quantity += quantity
        
        if self.quantity != 0:
            self.avg_price = (old_cost + new_cost) / self.quantity
        else:
            self.avg_price = 0.0
        
        self.update_market_value()
        
        return realized_pnl
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 持仓字典
        """
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'cost_basis': self.cost_basis,
            'last_updated': self.last_updated.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class Order:
    """订单信息"""
    order_id: str
    symbol: str
    quantity: float
    price: Optional[float]
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    strategy_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def update_fill(self, quantity: float, price: float):
        """更新成交信息
        
        Args:
            quantity: 成交数量
            price: 成交价格
        """
        total_filled_value = self.filled_quantity * self.avg_fill_price
        new_filled_value = quantity * price
        
        self.filled_quantity += quantity
        self.avg_fill_price = (total_filled_value + new_filled_value) / self.filled_quantity
        
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_time = datetime.now()
    
    def cancel(self):
        """取消订单"""
        if self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            self.status = OrderStatus.CANCELLED
            self.updated_time = datetime.now()
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 订单字典
        """
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'order_type': self.order_type.value,
            'side': self.side,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'created_time': self.created_time.isoformat(),
            'updated_time': self.updated_time.isoformat(),
            'strategy_id': self.strategy_id,
            'metadata': self.metadata
        }


@dataclass
class RiskLimits:
    """风险限制"""
    max_position_size: Optional[float] = None      # 最大持仓规模
    max_portfolio_value: Optional[float] = None    # 最大组合价值
    max_leverage: Optional[float] = None           # 最大杠杆
    max_drawdown: Optional[float] = None           # 最大回撤
    max_var: Optional[float] = None                # 最大VaR
    max_concentration: Optional[float] = None      # 最大集中度
    stop_loss_pct: Optional[float] = None          # 止损百分比
    take_profit_pct: Optional[float] = None        # 止盈百分比
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 风险限制字典
        """
        return {
            'max_position_size': self.max_position_size,
            'max_portfolio_value': self.max_portfolio_value,
            'max_leverage': self.max_leverage,
            'max_drawdown': self.max_drawdown,
            'max_var': self.max_var,
            'max_concentration': self.max_concentration,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }


@dataclass
class PortfolioMetrics:
    """投资组合指标"""
    total_value: float = 0.0
    cash: float = 0.0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 0.0
    concentration: Dict[str, float] = field(default_factory=dict)
    sector_allocation: Dict[str, float] = field(default_factory=dict)
    asset_allocation: Dict[str, float] = field(default_factory=dict)
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 指标字典
        """
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'leverage': self.leverage,
            'concentration': self.concentration,
            'sector_allocation': self.sector_allocation,
            'asset_allocation': self.asset_allocation,
            'var_95': self.var_95,
            'expected_shortfall': self.expected_shortfall,
            'beta': self.beta,
            'alpha': self.alpha,
            'sharpe_ratio': self.sharpe_ratio,
            'last_updated': self.last_updated.isoformat()
        }


class PortfolioManager:
    """投资组合管理器
    
    负责管理投资组合的资产配置、风险控制和持仓跟踪。
    """
    
    def __init__(self, initial_cash: float = 1000000.0, config: Dict = None):
        """初始化投资组合管理器
        
        Args:
            initial_cash: 初始现金
            config: 配置参数
        """
        self.config = config or {}
        
        # 投资组合状态
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.assets: Dict[str, Asset] = {}
        
        # 风险管理
        self.risk_limits = RiskLimits(**self.config.get('risk_limits', {}))
        
        # 历史记录
        self.trade_history: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        self.metrics_history: List[PortfolioMetrics] = []
        
        # 价格数据
        self.current_prices: Dict[str, float] = {}
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 回调函数
        self.on_order_filled_callbacks: List[Callable] = []
        self.on_position_changed_callbacks: List[Callable] = []
        self.on_risk_limit_exceeded_callbacks: List[Callable] = []
        
        # 日志记录
        self.logger = get_logger("PortfolioManager")
        
        # 统计信息
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_commission': 0.0,
            'max_portfolio_value': initial_cash,
            'min_portfolio_value': initial_cash,
            'max_drawdown': 0.0
        }
        
        self.logger.info(f"投资组合管理器初始化完成，初始资金: {initial_cash:,.2f}")
    
    def add_asset(self, asset: Asset):
        """添加资产信息
        
        Args:
            asset: 资产信息
        """
        with self._lock:
            self.assets[asset.symbol] = asset
            self.logger.info(f"添加资产: {asset.symbol} ({asset.name})")
    
    def update_price(self, symbol: str, price: float):
        """更新资产价格
        
        Args:
            symbol: 资产符号
            price: 当前价格
        """
        with self._lock:
            self.current_prices[symbol] = price
            
            # 更新持仓市值
            if symbol in self.positions:
                self.positions[symbol].update_market_value(price)
    
    def update_prices(self, prices: Dict[str, float]):
        """批量更新价格
        
        Args:
            prices: 价格字典
        """
        with self._lock:
            for symbol, price in prices.items():
                self.update_price(symbol, price)
    
    def place_order(self, symbol: str, quantity: float, 
                   order_type: OrderType = OrderType.MARKET,
                   price: Optional[float] = None,
                   strategy_id: Optional[str] = None) -> str:
        """下单
        
        Args:
            symbol: 资产符号
            quantity: 数量（正数买入，负数卖出）
            order_type: 订单类型
            price: 价格（限价单需要）
            strategy_id: 策略ID
            
        Returns:
            str: 订单ID
        """
        with self._lock:
            # 生成订单ID
            order_id = f"order_{len(self.orders)}_{int(datetime.now().timestamp())}"
            
            # 确定买卖方向
            side = 'buy' if quantity > 0 else 'sell'
            
            # 创建订单
            order = Order(
                order_id=order_id,
                symbol=symbol,
                quantity=abs(quantity),
                price=price,
                order_type=order_type,
                side=side,
                strategy_id=strategy_id
            )
            
            # 风险检查
            if not self._check_risk_limits(order):
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"订单被风险控制拒绝: {order_id}")
                return order_id
            
            # 资金检查
            if not self._check_sufficient_funds(order):
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"资金不足，订单被拒绝: {order_id}")
                return order_id
            
            # 保存订单
            self.orders[order_id] = order
            order.status = OrderStatus.SUBMITTED
            
            # 如果是市价单，立即执行
            if order_type == OrderType.MARKET:
                self._execute_market_order(order)
            
            self.logger.info(f"订单已提交: {order_id} {side} {quantity} {symbol}")
            return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            bool: 取消是否成功
        """
        with self._lock:
            if order_id not in self.orders:
                return False
            
            order = self.orders[order_id]
            order.cancel()
            
            self.logger.info(f"订单已取消: {order_id}")
            return True
    
    def _execute_market_order(self, order: Order):
        """执行市价单
        
        Args:
            order: 订单对象
        """
        # 获取当前价格
        current_price = self.current_prices.get(order.symbol)
        if current_price is None:
            order.status = OrderStatus.REJECTED
            self.logger.error(f"无法获取 {order.symbol} 的当前价格")
            return
        
        # 执行交易
        quantity = order.quantity if order.side == 'buy' else -order.quantity
        self._execute_trade(order.symbol, quantity, current_price, order.order_id, order.strategy_id)
        
        # 更新订单状态
        order.update_fill(order.quantity, current_price)
    
    def _execute_trade(self, symbol: str, quantity: float, price: float, 
                      order_id: str, strategy_id: Optional[str] = None):
        """执行交易
        
        Args:
            symbol: 资产符号
            quantity: 数量（正数买入，负数卖出）
            price: 价格
            order_id: 订单ID
            strategy_id: 策略ID
        """
        # 计算交易金额
        trade_value = abs(quantity) * price
        commission = self._calculate_commission(trade_value)
        
        # 更新现金
        if quantity > 0:  # 买入
            self.cash -= (trade_value + commission)
        else:  # 卖出
            self.cash += (trade_value - commission)
        
        # 更新持仓
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, 0, 0)
        
        realized_pnl = self.positions[symbol].add_trade(quantity, price)
        
        # 如果持仓为0，删除持仓记录
        if self.positions[symbol].quantity == 0:
            del self.positions[symbol]
        
        # 记录交易历史
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'trade_value': trade_value,
            'commission': commission,
            'realized_pnl': realized_pnl,
            'order_id': order_id,
            'strategy_id': strategy_id
        }
        self.trade_history.append(trade_record)
        
        # 更新统计
        self.stats['total_trades'] += 1
        self.stats['total_commission'] += commission
        
        if realized_pnl > 0:
            self.stats['winning_trades'] += 1
        elif realized_pnl < 0:
            self.stats['losing_trades'] += 1
        
        # 调用回调函数
        for callback in self.on_order_filled_callbacks:
            try:
                callback(trade_record)
            except Exception as e:
                self.logger.error(f"订单成交回调函数执行失败: {e}")
        
        for callback in self.on_position_changed_callbacks:
            try:
                callback(symbol, self.positions.get(symbol))
            except Exception as e:
                self.logger.error(f"持仓变化回调函数执行失败: {e}")
        
        self.logger.info(f"交易执行: {quantity} {symbol} @ {price}, 实现盈亏: {realized_pnl:.2f}")
    
    def _calculate_commission(self, trade_value: float) -> float:
        """计算手续费
        
        Args:
            trade_value: 交易金额
            
        Returns:
            float: 手续费
        """
        commission_rate = self.config.get('commission_rate', 0.001)  # 默认0.1%
        min_commission = self.config.get('min_commission', 1.0)
        
        commission = max(trade_value * commission_rate, min_commission)
        return commission
    
    def _check_risk_limits(self, order: Order) -> bool:
        """检查风险限制
        
        Args:
            order: 订单对象
            
        Returns:
            bool: 是否通过风险检查
        """
        try:
            # 检查最大持仓规模
            if self.risk_limits.max_position_size:
                current_position = self.positions.get(order.symbol)
                if current_position:
                    new_quantity = current_position.quantity
                    if order.side == 'buy':
                        new_quantity += order.quantity
                    else:
                        new_quantity -= order.quantity
                    
                    if abs(new_quantity) > self.risk_limits.max_position_size:
                        self.logger.warning(f"超过最大持仓限制: {order.symbol}")
                        return False
            
            # 检查最大组合价值
            if self.risk_limits.max_portfolio_value:
                current_value = self.get_portfolio_value()
                if current_value > self.risk_limits.max_portfolio_value:
                    self.logger.warning(f"超过最大组合价值限制: {current_value}")
                    return False
            
            # 检查最大杠杆
            if self.risk_limits.max_leverage:
                leverage = self.calculate_leverage()
                if leverage > self.risk_limits.max_leverage:
                    self.logger.warning(f"超过最大杠杆限制: {leverage}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"风险检查失败: {e}")
            return False
    
    def _check_sufficient_funds(self, order: Order) -> bool:
        """检查资金是否充足
        
        Args:
            order: 订单对象
            
        Returns:
            bool: 资金是否充足
        """
        if order.side == 'sell':
            # 卖出检查持仓是否充足
            position = self.positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                return False
        else:
            # 买入检查现金是否充足
            price = order.price or self.current_prices.get(order.symbol, 0)
            required_cash = order.quantity * price
            commission = self._calculate_commission(required_cash)
            
            if self.cash < (required_cash + commission):
                return False
        
        return True
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓信息
        
        Args:
            symbol: 资产符号
            
        Returns:
            Optional[Position]: 持仓信息
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """获取所有持仓
        
        Returns:
            Dict[str, Position]: 所有持仓
        """
        return self.positions.copy()
    
    def get_portfolio_value(self) -> float:
        """获取投资组合总价值
        
        Returns:
            float: 投资组合总价值
        """
        total_value = self.cash
        
        for position in self.positions.values():
            total_value += position.market_value
        
        return total_value
    
    def calculate_metrics(self) -> PortfolioMetrics:
        """计算投资组合指标
        
        Returns:
            PortfolioMetrics: 投资组合指标
        """
        with self._lock:
            total_value = self.get_portfolio_value()
            total_pnl = total_value - self.initial_cash
            
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            
            # 计算杠杆
            leverage = self.calculate_leverage()
            
            # 计算集中度
            concentration = {}
            if total_value > 0:
                for symbol, position in self.positions.items():
                    concentration[symbol] = abs(position.market_value) / total_value
            
            # 计算行业配置
            sector_allocation = defaultdict(float)
            asset_allocation = defaultdict(float)
            
            for symbol, position in self.positions.items():
                asset_info = self.assets.get(symbol)
                if asset_info and total_value > 0:
                    weight = abs(position.market_value) / total_value
                    
                    if asset_info.sector:
                        sector_allocation[asset_info.sector] += weight
                    
                    asset_allocation[asset_info.asset_type.value] += weight
            
            metrics = PortfolioMetrics(
                total_value=total_value,
                cash=self.cash,
                total_pnl=total_pnl,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                leverage=leverage,
                concentration=dict(concentration),
                sector_allocation=dict(sector_allocation),
                asset_allocation=dict(asset_allocation)
            )
            
            # 更新统计
            self.stats['max_portfolio_value'] = max(self.stats['max_portfolio_value'], total_value)
            self.stats['min_portfolio_value'] = min(self.stats['min_portfolio_value'], total_value)
            
            # 计算最大回撤
            if self.metrics_history:
                peak_value = max(m.total_value for m in self.metrics_history)
                current_drawdown = (peak_value - total_value) / peak_value if peak_value > 0 else 0
                self.stats['max_drawdown'] = max(self.stats['max_drawdown'], current_drawdown)
            
            return metrics
    
    def calculate_leverage(self) -> float:
        """计算杠杆比率
        
        Returns:
            float: 杠杆比率
        """
        total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        total_value = self.get_portfolio_value()
        
        return total_exposure / total_value if total_value > 0 else 0
    
    def process_signal(self, signal: StrategySignal) -> Optional[str]:
        """处理策略信号
        
        Args:
            signal: 策略信号
            
        Returns:
            Optional[str]: 订单ID（如果下单）
        """
        try:
            if signal.signal_type == SignalType.BUY:
                quantity = signal.quantity or self._calculate_position_size(signal)
                return self.place_order(
                    symbol=signal.symbol,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    strategy_id=signal.strategy_id
                )
            
            elif signal.signal_type == SignalType.SELL:
                quantity = signal.quantity or self._calculate_position_size(signal)
                return self.place_order(
                    symbol=signal.symbol,
                    quantity=-quantity,
                    order_type=OrderType.MARKET,
                    strategy_id=signal.strategy_id
                )
            
            elif signal.signal_type == SignalType.CLOSE:
                position = self.get_position(signal.symbol)
                if position and position.quantity != 0:
                    return self.place_order(
                        symbol=signal.symbol,
                        quantity=-position.quantity,
                        order_type=OrderType.MARKET,
                        strategy_id=signal.strategy_id
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"处理信号失败: {e}")
            return None
    
    def _calculate_position_size(self, signal: StrategySignal) -> float:
        """计算持仓规模
        
        Args:
            signal: 策略信号
            
        Returns:
            float: 持仓规模
        """
        # 简单的固定比例仓位管理
        position_ratio = self.config.get('default_position_ratio', 0.1)  # 默认10%
        
        current_price = self.current_prices.get(signal.symbol, signal.price or 1.0)
        portfolio_value = self.get_portfolio_value()
        
        position_value = portfolio_value * position_ratio
        quantity = position_value / current_price
        
        # 考虑最小交易单位
        asset_info = self.assets.get(signal.symbol)
        if asset_info:
            lot_size = asset_info.lot_size
            quantity = round(quantity / lot_size) * lot_size
        
        return quantity
    
    def rebalance_portfolio(self, target_weights: Dict[str, float]) -> List[str]:
        """重新平衡投资组合
        
        Args:
            target_weights: 目标权重字典
            
        Returns:
            List[str]: 生成的订单ID列表
        """
        order_ids = []
        
        try:
            total_value = self.get_portfolio_value()
            
            for symbol, target_weight in target_weights.items():
                target_value = total_value * target_weight
                current_position = self.get_position(symbol)
                current_value = current_position.market_value if current_position else 0
                
                diff_value = target_value - current_value
                
                if abs(diff_value) > total_value * 0.01:  # 1%的阈值
                    current_price = self.current_prices.get(symbol)
                    if current_price:
                        quantity = diff_value / current_price
                        
                        order_id = self.place_order(
                            symbol=symbol,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        )
                        
                        if order_id:
                            order_ids.append(order_id)
            
            self.logger.info(f"投资组合重新平衡完成，生成 {len(order_ids)} 个订单")
            return order_ids
            
        except Exception as e:
            self.logger.error(f"投资组合重新平衡失败: {e}")
            return order_ids
    
    def add_order_filled_callback(self, callback: Callable):
        """添加订单成交回调
        
        Args:
            callback: 回调函数
        """
        self.on_order_filled_callbacks.append(callback)
    
    def add_position_changed_callback(self, callback: Callable):
        """添加持仓变化回调
        
        Args:
            callback: 回调函数
        """
        self.on_position_changed_callbacks.append(callback)
    
    def add_risk_limit_exceeded_callback(self, callback: Callable):
        """添加风险限制超出回调
        
        Args:
            callback: 回调函数
        """
        self.on_risk_limit_exceeded_callbacks.append(callback)
    
    def get_trade_history(self, symbol: Optional[str] = None, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict]:
        """获取交易历史
        
        Args:
            symbol: 资产符号（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            List[Dict]: 交易历史列表
        """
        filtered_history = self.trade_history
        
        if symbol:
            filtered_history = [t for t in filtered_history if t['symbol'] == symbol]
        
        if start_date:
            filtered_history = [
                t for t in filtered_history 
                if datetime.fromisoformat(t['timestamp']) >= start_date
            ]
        
        if end_date:
            filtered_history = [
                t for t in filtered_history 
                if datetime.fromisoformat(t['timestamp']) <= end_date
            ]
        
        return filtered_history
    
    def get_performance_summary(self) -> Dict:
        """获取绩效摘要
        
        Returns:
            Dict: 绩效摘要
        """
        metrics = self.calculate_metrics()
        
        total_return = (metrics.total_value - self.initial_cash) / self.initial_cash
        win_rate = self.stats['winning_trades'] / max(self.stats['total_trades'], 1)
        
        return {
            'initial_cash': self.initial_cash,
            'current_value': metrics.total_value,
            'total_return': total_return,
            'total_pnl': metrics.total_pnl,
            'unrealized_pnl': metrics.unrealized_pnl,
            'realized_pnl': metrics.realized_pnl,
            'total_trades': self.stats['total_trades'],
            'win_rate': win_rate,
            'max_drawdown': self.stats['max_drawdown'],
            'leverage': metrics.leverage,
            'cash_ratio': self.cash / metrics.total_value if metrics.total_value > 0 else 1.0
        }
    
    def save_snapshot(self) -> Dict:
        """保存投资组合快照
        
        Returns:
            Dict: 投资组合快照
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'cash': self.cash,
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'metrics': self.calculate_metrics().to_dict(),
            'stats': self.stats.copy(),
            'current_prices': self.current_prices.copy()
        }
        
        self.portfolio_history.append(snapshot)
        return snapshot
    
    def reset(self, initial_cash: Optional[float] = None):
        """重置投资组合
        
        Args:
            initial_cash: 新的初始资金（可选）
        """
        with self._lock:
            if initial_cash is not None:
                self.initial_cash = initial_cash
            
            self.cash = self.initial_cash
            self.positions.clear()
            self.orders.clear()
            self.trade_history.clear()
            self.portfolio_history.clear()
            self.metrics_history.clear()
            self.current_prices.clear()
            
            # 重置统计
            self.stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_commission': 0.0,
                'max_portfolio_value': self.initial_cash,
                'min_portfolio_value': self.initial_cash,
                'max_drawdown': 0.0
            }
            
            self.logger.info(f"投资组合已重置，初始资金: {self.initial_cash:,.2f}")
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 投资组合字典
        """
        return {
            'cash': self.cash,
            'initial_cash': self.initial_cash,
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'orders': {order_id: order.to_dict() for order_id, order in self.orders.items()},
            'assets': {symbol: asset.to_dict() for symbol, asset in self.assets.items()},
            'risk_limits': self.risk_limits.to_dict(),
            'metrics': self.calculate_metrics().to_dict(),
            'stats': self.stats.copy(),
            'current_prices': self.current_prices.copy()
        }
    
    def __str__(self) -> str:
        value = self.get_portfolio_value()
        pnl = value - self.initial_cash
        return f"Portfolio(value={value:,.2f}, pnl={pnl:,.2f}, positions={len(self.positions)})"
    
    def __repr__(self) -> str:
        return self.__str__()