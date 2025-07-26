"""Execution Engine Module

执行引擎核心模块，负责整体执行流程的协调和管理。

主要功能：
- 订单生命周期管理
- 执行策略选择
- 市场连接管理
- 执行监控和报告
- 风险控制集成
"""

import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import json

from ..common.logging import get_logger
from ..common.exceptions.execution import ExecutionEngineError, OrderExecutionError
from ..alpha_engine.base import StrategySignal
from ..risk_sentinel.manager import RiskManager


class ExecutionMode(Enum):
    """执行模式枚举"""
    SYNC = "sync"          # 同步执行
    ASYNC = "async"        # 异步执行
    BATCH = "batch"        # 批量执行
    STREAMING = "streaming" # 流式执行


class ExecutionStatus(Enum):
    """执行状态枚举"""
    IDLE = "idle"                    # 空闲
    RUNNING = "running"              # 运行中
    PAUSED = "paused"                # 暂停
    STOPPED = "stopped"              # 停止
    ERROR = "error"                  # 错误
    EMERGENCY_STOP = "emergency_stop" # 紧急停止


from ..common.models import OrderSide, OrderType, OrderStatus


@dataclass
class ExecutionOrder:
    """执行订单"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancel
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    remaining_quantity: float = 0.0
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    strategy_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    execution_algorithm: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        self.remaining_quantity = self.quantity
    
    def update_fill(self, quantity: float, price: float, trade_id: str = None):
        """更新成交信息
        
        Args:
            quantity: 成交数量
            price: 成交价格
            trade_id: 交易ID
        """
        # 更新成交信息
        total_filled_value = self.filled_quantity * self.avg_fill_price
        new_filled_value = quantity * price
        
        self.filled_quantity += quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        
        if self.filled_quantity > 0:
            self.avg_fill_price = (total_filled_value + new_filled_value) / self.filled_quantity
        
        # 更新状态
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_time = datetime.now()
        
        # 记录成交信息
        if 'fills' not in self.metadata:
            self.metadata['fills'] = []
        
        self.metadata['fills'].append({
            'quantity': quantity,
            'price': price,
            'trade_id': trade_id,
            'timestamp': datetime.now().isoformat()
        })
    
    def cancel(self, reason: str = ""):
        """取消订单
        
        Args:
            reason: 取消原因
        """
        if self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, 
                          OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]:
            self.status = OrderStatus.CANCELLED
            self.updated_time = datetime.now()
            self.metadata['cancel_reason'] = reason
    
    def reject(self, reason: str = ""):
        """拒绝订单
        
        Args:
            reason: 拒绝原因
        """
        self.status = OrderStatus.REJECTED
        self.updated_time = datetime.now()
        self.metadata['reject_reason'] = reason
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 订单字典
        """
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'remaining_quantity': self.remaining_quantity,
            'created_time': self.created_time.isoformat(),
            'updated_time': self.updated_time.isoformat(),
            'strategy_id': self.strategy_id,
            'parent_order_id': self.parent_order_id,
            'child_order_ids': self.child_order_ids,
            'execution_algorithm': self.execution_algorithm,
            'metadata': self.metadata
        }


@dataclass
class Trade:
    """交易记录"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    commission: float = 0.0
    fees: float = 0.0
    market: str = ""
    counterparty: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 交易字典
        """
        return {
            'trade_id': self.trade_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'commission': self.commission,
            'fees': self.fees,
            'market': self.market,
            'counterparty': self.counterparty,
            'metadata': self.metadata
        }


@dataclass
class ExecutionMetrics:
    """执行指标"""
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    total_volume: float = 0.0
    total_value: float = 0.0
    avg_fill_time: float = 0.0
    fill_rate: float = 0.0
    slippage: float = 0.0
    implementation_shortfall: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 指标字典
        """
        return {
            'total_orders': self.total_orders,
            'filled_orders': self.filled_orders,
            'cancelled_orders': self.cancelled_orders,
            'rejected_orders': self.rejected_orders,
            'total_volume': self.total_volume,
            'total_value': self.total_value,
            'avg_fill_time': self.avg_fill_time,
            'fill_rate': self.fill_rate,
            'slippage': self.slippage,
            'implementation_shortfall': self.implementation_shortfall,
            'last_updated': self.last_updated.isoformat()
        }


class ExecutionEngine:
    """执行引擎
    
    负责整体执行流程的协调和管理。
    """
    
    def __init__(self, config: Dict = None, risk_manager: RiskManager = None):
        """初始化执行引擎
        
        Args:
            config: 配置参数
            risk_manager: 风险管理器
        """
        self.config = config or {}
        self.risk_manager = risk_manager
        
        # 执行状态
        self.status = ExecutionStatus.IDLE
        self.mode = ExecutionMode.SYNC
        
        # 订单管理
        self.orders: Dict[str, ExecutionOrder] = {}
        self.trades: List[Trade] = []
        self.pending_orders: deque = deque()
        
        # 市场连接器（将在后续实现中注入）
        self.market_connectors: Dict[str, Any] = {}
        
        # 执行算法（将在后续实现中注入）
        self.execution_algorithms: Dict[str, Any] = {}
        
        # 路由配置
        self.routing_rules: Dict[str, str] = {}  # symbol -> market
        
        # 线程管理
        self._lock = threading.RLock()
        self._execution_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 异步事件循环
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 回调函数
        self.on_order_update_callbacks: List[Callable] = []
        self.on_trade_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
        
        # 日志记录
        self.logger = get_logger("ExecutionEngine")
        
        # 统计信息
        self.stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'total_volume': 0.0,
            'total_value': 0.0,
            'avg_execution_time': 0.0,
            'last_execution_time': None
        }
        
        # 性能监控
        self.execution_times: deque = deque(maxlen=1000)
        self.slippage_history: deque = deque(maxlen=1000)
        
        self.logger.info("执行引擎初始化完成")
    
    def start(self, mode: ExecutionMode = ExecutionMode.SYNC):
        """启动执行引擎
        
        Args:
            mode: 执行模式
        """
        with self._lock:
            if self.status != ExecutionStatus.IDLE:
                raise ExecutionEngineError("执行引擎已在运行")
            
            self.mode = mode
            self.status = ExecutionStatus.RUNNING
            self._stop_event.clear()
            
            # 启动执行线程
            if mode in [ExecutionMode.ASYNC, ExecutionMode.STREAMING]:
                self._execution_thread = threading.Thread(
                    target=self._execution_loop,
                    name="ExecutionEngine"
                )
                self._execution_thread.start()
            
            self.logger.info(f"执行引擎已启动，模式: {mode.value}")
    
    def stop(self):
        """停止执行引擎"""
        with self._lock:
            if self.status == ExecutionStatus.IDLE:
                return
            
            self.status = ExecutionStatus.STOPPED
            self._stop_event.set()
            
            # 等待执行线程结束
            if self._execution_thread and self._execution_thread.is_alive():
                self._execution_thread.join(timeout=5.0)
            
            # 取消所有待处理订单
            self._cancel_pending_orders("引擎停止")
            
            self.logger.info("执行引擎已停止")
    
    def pause(self):
        """暂停执行引擎"""
        with self._lock:
            if self.status == ExecutionStatus.RUNNING:
                self.status = ExecutionStatus.PAUSED
                self.logger.info("执行引擎已暂停")
    
    def resume(self):
        """恢复执行引擎"""
        with self._lock:
            if self.status == ExecutionStatus.PAUSED:
                self.status = ExecutionStatus.RUNNING
                self.logger.info("执行引擎已恢复")
    
    def emergency_stop(self):
        """紧急停止"""
        with self._lock:
            self.status = ExecutionStatus.EMERGENCY_STOP
            self._stop_event.set()
            
            # 取消所有订单
            self._cancel_all_orders("紧急停止")
            
            self.logger.critical("执行引擎紧急停止")
    
    def submit_order(self, order: ExecutionOrder) -> str:
        """提交订单
        
        Args:
            order: 执行订单
            
        Returns:
            str: 订单ID
        """
        with self._lock:
            if self.status not in [ExecutionStatus.RUNNING, ExecutionStatus.PAUSED]:
                raise ExecutionEngineError(f"执行引擎状态不允许提交订单: {self.status.value}")
            
            # 风险检查
            if self.risk_manager and not self._check_order_risk(order):
                order.reject("风险控制拒绝")
                self.orders[order.order_id] = order
                self.stats['orders_rejected'] += 1
                return order.order_id
            
            # 保存订单
            self.orders[order.order_id] = order
            order.status = OrderStatus.SUBMITTED
            order.updated_time = datetime.now()
            
            # 添加到执行队列
            if self.mode == ExecutionMode.SYNC:
                self._execute_order_sync(order)
            else:
                self.pending_orders.append(order.order_id)
            
            self.stats['orders_submitted'] += 1
            
            # 调用回调
            self._notify_order_update(order)
            
            self.logger.info(f"订单已提交: {order.order_id} {order.side.value} {order.quantity} {order.symbol}")
            return order.order_id
    
    def cancel_order(self, order_id: str, reason: str = "") -> bool:
        """取消订单
        
        Args:
            order_id: 订单ID
            reason: 取消原因
            
        Returns:
            bool: 是否成功取消
        """
        with self._lock:
            if order_id not in self.orders:
                return False
            
            order = self.orders[order_id]
            
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, 
                               OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]:
                order.cancel(reason)
                self.stats['orders_cancelled'] += 1
                
                # 从待处理队列中移除
                try:
                    pending_list = list(self.pending_orders)
                    if order_id in pending_list:
                        pending_list.remove(order_id)
                        self.pending_orders = deque(pending_list)
                except ValueError:
                    pass
                
                # 调用回调
                self._notify_order_update(order)
                
                self.logger.info(f"订单已取消: {order_id} - {reason}")
                return True
            
            return False
    
    def modify_order(self, order_id: str, **kwargs) -> bool:
        """修改订单
        
        Args:
            order_id: 订单ID
            **kwargs: 修改参数
            
        Returns:
            bool: 是否成功修改
        """
        with self._lock:
            if order_id not in self.orders:
                return False
            
            order = self.orders[order_id]
            
            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED]:
                return False
            
            # 更新订单参数
            for key, value in kwargs.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            
            order.updated_time = datetime.now()
            
            # 调用回调
            self._notify_order_update(order)
            
            self.logger.info(f"订单已修改: {order_id}")
            return True
    
    def process_signal(self, signal: StrategySignal) -> Optional[str]:
        """处理策略信号
        
        Args:
            signal: 策略信号
            
        Returns:
            Optional[str]: 订单ID（如果生成）
        """
        try:
            # 将信号转换为订单
            order = self._signal_to_order(signal)
            if order:
                return self.submit_order(order)
            
            return None
            
        except Exception as e:
            self.logger.error(f"处理信号失败: {e}")
            return None
    
    def _signal_to_order(self, signal: StrategySignal) -> Optional[ExecutionOrder]:
        """将策略信号转换为执行订单
        
        Args:
            signal: 策略信号
            
        Returns:
            Optional[ExecutionOrder]: 执行订单
        """
        from ..alpha_engine.signal import SignalType
        
        if signal.signal_type == SignalType.BUY:
            side = OrderSide.BUY
        elif signal.signal_type == SignalType.SELL:
            side = OrderSide.SELL
        else:
            return None
        
        # 生成订单ID
        order_id = f"order_{uuid.uuid4().hex[:8]}"
        
        # 确定订单类型
        order_type = OrderType.MARKET
        if signal.price:
            order_type = OrderType.LIMIT
        
        # 创建订单
        order = ExecutionOrder(
            order_id=order_id,
            symbol=signal.symbol,
            side=side,
            order_type=order_type,
            quantity=signal.quantity or 0,
            price=signal.price,
            strategy_id=signal.strategy_id,
            metadata={
                'signal_id': signal.signal_id,
                'signal_strength': signal.strength.value if signal.strength else None,
                'signal_metadata': signal.metadata
            }
        )
        
        return order
    
    def _check_order_risk(self, order: ExecutionOrder) -> bool:
        """检查订单风险
        
        Args:
            order: 执行订单
            
        Returns:
            bool: 是否通过风险检查
        """
        if not self.risk_manager:
            return True
        
        try:
            # 这里可以实现更详细的风险检查逻辑
            # 例如检查持仓限制、资金充足性等
            return True
            
        except Exception as e:
            self.logger.error(f"订单风险检查失败: {e}")
            return False
    
    def _execution_loop(self):
        """执行循环（异步模式）"""
        self.logger.info("执行循环已启动")
        
        while not self._stop_event.is_set():
            try:
                if self.status == ExecutionStatus.RUNNING and self.pending_orders:
                    order_id = self.pending_orders.popleft()
                    
                    if order_id in self.orders:
                        order = self.orders[order_id]
                        if order.status == OrderStatus.SUBMITTED:
                            self._execute_order_async(order)
                
                # 短暂休眠
                self._stop_event.wait(0.01)
                
            except Exception as e:
                self.logger.error(f"执行循环错误: {e}")
                self._stop_event.wait(1.0)
        
        self.logger.info("执行循环已结束")
    
    def _execute_order_sync(self, order: ExecutionOrder):
        """同步执行订单
        
        Args:
            order: 执行订单
        """
        start_time = datetime.now()
        
        try:
            # 获取市场连接器
            market = self._get_market_for_symbol(order.symbol)
            connector = self.market_connectors.get(market)
            
            if not connector:
                order.reject(f"未找到市场连接器: {market}")
                return
            
            # 执行订单
            if order.order_type == OrderType.MARKET:
                self._execute_market_order(order, connector)
            elif order.order_type == OrderType.LIMIT:
                self._execute_limit_order(order, connector)
            elif order.order_type in [OrderType.TWAP, OrderType.VWAP]:
                self._execute_algorithmic_order(order)
            else:
                order.reject(f"不支持的订单类型: {order.order_type.value}")
            
            # 记录执行时间
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_times.append(execution_time)
            
            # 更新统计
            if order.status == OrderStatus.FILLED:
                self.stats['orders_filled'] += 1
                self.stats['total_volume'] += order.filled_quantity
                self.stats['total_value'] += order.filled_quantity * order.avg_fill_price
            
            self.stats['last_execution_time'] = datetime.now()
            
        except Exception as e:
            order.reject(f"执行失败: {e}")
            self.logger.error(f"订单执行失败: {order.order_id} - {e}")
            
            # 调用错误回调
            for callback in self.on_error_callbacks:
                try:
                    callback(order, e)
                except Exception as cb_error:
                    self.logger.error(f"错误回调执行失败: {cb_error}")
    
    def _execute_order_async(self, order: ExecutionOrder):
        """异步执行订单
        
        Args:
            order: 执行订单
        """
        # 在实际实现中，这里会使用异步方式执行
        # 目前简化为调用同步执行
        self._execute_order_sync(order)
    
    def _execute_market_order(self, order: ExecutionOrder, connector):
        """执行市价单
        
        Args:
            order: 执行订单
            connector: 市场连接器
        """
        # 模拟市价单执行
        # 在实际实现中，这里会调用市场连接器的相应方法
        
        # 获取当前市价
        current_price = self._get_current_price(order.symbol)
        if current_price is None:
            order.reject("无法获取当前价格")
            return
        
        # 模拟滑点
        slippage = self._calculate_slippage(order, current_price)
        fill_price = current_price + slippage
        
        # 生成交易记录
        trade = self._create_trade(order, order.quantity, fill_price)
        
        # 更新订单
        order.update_fill(order.quantity, fill_price, trade.trade_id)
        order.status = OrderStatus.FILLED
        
        # 记录交易
        self.trades.append(trade)
        
        # 调用回调
        self._notify_order_update(order)
        self._notify_trade(trade)
        
        self.logger.info(f"市价单已成交: {order.order_id} @ {fill_price}")
    
    def _execute_limit_order(self, order: ExecutionOrder, connector):
        """执行限价单
        
        Args:
            order: 执行订单
            connector: 市场连接器
        """
        # 模拟限价单执行
        # 在实际实现中，这里会将订单发送到交易所
        
        current_price = self._get_current_price(order.symbol)
        if current_price is None:
            order.reject("无法获取当前价格")
            return
        
        # 检查是否可以立即成交
        can_fill = False
        if order.side == OrderSide.BUY and current_price <= order.price:
            can_fill = True
        elif order.side == OrderSide.SELL and current_price >= order.price:
            can_fill = True
        
        if can_fill:
            # 立即成交
            trade = self._create_trade(order, order.quantity, order.price)
            order.update_fill(order.quantity, order.price, trade.trade_id)
            order.status = OrderStatus.FILLED
            
            self.trades.append(trade)
            self._notify_order_update(order)
            self._notify_trade(trade)
            
            self.logger.info(f"限价单已成交: {order.order_id} @ {order.price}")
        else:
            # 等待成交
            order.status = OrderStatus.ACCEPTED
            self._notify_order_update(order)
            
            self.logger.info(f"限价单已接受: {order.order_id} @ {order.price}")
    
    def _execute_algorithmic_order(self, order: ExecutionOrder):
        """执行算法订单
        
        Args:
            order: 执行订单
        """
        algorithm_name = order.order_type.value
        algorithm = self.execution_algorithms.get(algorithm_name)
        
        if not algorithm:
            order.reject(f"未找到执行算法: {algorithm_name}")
            return
        
        try:
            # 启动算法执行
            algorithm.execute(order)
            order.status = OrderStatus.ACCEPTED
            
            self._notify_order_update(order)
            
            self.logger.info(f"算法订单已启动: {order.order_id} ({algorithm_name})")
            
        except Exception as e:
            order.reject(f"算法执行失败: {e}")
            self.logger.error(f"算法订单执行失败: {order.order_id} - {e}")
    
    def _get_market_for_symbol(self, symbol: str) -> str:
        """获取交易品种对应的市场
        
        Args:
            symbol: 交易品种
            
        Returns:
            str: 市场名称
        """
        return self.routing_rules.get(symbol, "default")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格
        
        Args:
            symbol: 交易品种
            
        Returns:
            Optional[float]: 当前价格
        """
        # 在实际实现中，这里会从市场数据中获取价格
        # 目前返回模拟价格
        return 100.0  # 模拟价格
    
    def _calculate_slippage(self, order: ExecutionOrder, current_price: float) -> float:
        """计算滑点
        
        Args:
            order: 执行订单
            current_price: 当前价格
            
        Returns:
            float: 滑点
        """
        # 简单的滑点模型
        base_slippage = current_price * 0.0001  # 0.01%
        
        # 根据订单大小调整滑点
        size_factor = min(order.quantity / 1000, 1.0)
        slippage = base_slippage * (1 + size_factor)
        
        # 买入为正滑点，卖出为负滑点
        if order.side == OrderSide.SELL:
            slippage = -slippage
        
        self.slippage_history.append(abs(slippage) / current_price)
        
        return slippage
    
    def _create_trade(self, order: ExecutionOrder, quantity: float, price: float) -> Trade:
        """创建交易记录
        
        Args:
            order: 执行订单
            quantity: 成交数量
            price: 成交价格
            
        Returns:
            Trade: 交易记录
        """
        trade_id = f"trade_{uuid.uuid4().hex[:8]}"
        
        trade = Trade(
            trade_id=trade_id,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=self._calculate_commission(quantity * price),
            market=self._get_market_for_symbol(order.symbol)
        )
        
        return trade
    
    def _calculate_commission(self, trade_value: float) -> float:
        """计算手续费
        
        Args:
            trade_value: 交易金额
            
        Returns:
            float: 手续费
        """
        commission_rate = self.config.get('commission_rate', 0.001)
        min_commission = self.config.get('min_commission', 1.0)
        
        return max(trade_value * commission_rate, min_commission)
    
    def _cancel_pending_orders(self, reason: str):
        """取消所有待处理订单
        
        Args:
            reason: 取消原因
        """
        pending_order_ids = list(self.pending_orders)
        for order_id in pending_order_ids:
            self.cancel_order(order_id, reason)
        
        self.pending_orders.clear()
    
    def _cancel_all_orders(self, reason: str):
        """取消所有订单
        
        Args:
            reason: 取消原因
        """
        for order_id in list(self.orders.keys()):
            self.cancel_order(order_id, reason)
    
    def _notify_order_update(self, order: ExecutionOrder):
        """通知订单更新
        
        Args:
            order: 执行订单
        """
        for callback in self.on_order_update_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"订单更新回调执行失败: {e}")
    
    def _notify_trade(self, trade: Trade):
        """通知交易
        
        Args:
            trade: 交易记录
        """
        for callback in self.on_trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                self.logger.error(f"交易回调执行失败: {e}")
    
    def get_order(self, order_id: str) -> Optional[ExecutionOrder]:
        """获取订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            Optional[ExecutionOrder]: 执行订单
        """
        return self.orders.get(order_id)
    
    def get_orders(self, symbol: Optional[str] = None, 
                  status: Optional[OrderStatus] = None) -> List[ExecutionOrder]:
        """获取订单列表
        
        Args:
            symbol: 交易品种（可选）
            status: 订单状态（可选）
            
        Returns:
            List[ExecutionOrder]: 订单列表
        """
        orders = list(self.orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        return orders
    
    def get_trades(self, symbol: Optional[str] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None) -> List[Trade]:
        """获取交易列表
        
        Args:
            symbol: 交易品种（可选）
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
            
        Returns:
            List[Trade]: 交易列表
        """
        trades = self.trades
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        if start_time:
            trades = [t for t in trades if t.timestamp >= start_time]
        
        if end_time:
            trades = [t for t in trades if t.timestamp <= end_time]
        
        return trades
    
    def calculate_metrics(self) -> ExecutionMetrics:
        """计算执行指标
        
        Returns:
            ExecutionMetrics: 执行指标
        """
        metrics = ExecutionMetrics()
        
        # 基本统计
        metrics.total_orders = len(self.orders)
        metrics.filled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        metrics.cancelled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED])
        metrics.rejected_orders = len([o for o in self.orders.values() if o.status == OrderStatus.REJECTED])
        
        # 成交率
        if metrics.total_orders > 0:
            metrics.fill_rate = metrics.filled_orders / metrics.total_orders
        
        # 交易量和金额
        for trade in self.trades:
            metrics.total_volume += trade.quantity
            metrics.total_value += trade.quantity * trade.price
        
        # 平均执行时间
        if self.execution_times:
            metrics.avg_fill_time = sum(self.execution_times) / len(self.execution_times)
        
        # 平均滑点
        if self.slippage_history:
            metrics.slippage = sum(self.slippage_history) / len(self.slippage_history)
        
        return metrics
    
    def add_order_update_callback(self, callback: Callable):
        """添加订单更新回调
        
        Args:
            callback: 回调函数
        """
        self.on_order_update_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable):
        """添加交易回调
        
        Args:
            callback: 回调函数
        """
        self.on_trade_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """添加错误回调
        
        Args:
            callback: 回调函数
        """
        self.on_error_callbacks.append(callback)
    
    def add_market_connector(self, name: str, connector):
        """添加市场连接器
        
        Args:
            name: 连接器名称
            connector: 连接器实例
        """
        self.market_connectors[name] = connector
        self.logger.info(f"添加市场连接器: {name}")
    
    def add_execution_algorithm(self, name: str, algorithm):
        """添加执行算法
        
        Args:
            name: 算法名称
            algorithm: 算法实例
        """
        self.execution_algorithms[name] = algorithm
        self.logger.info(f"添加执行算法: {name}")
    
    def set_routing_rule(self, symbol: str, market: str):
        """设置路由规则
        
        Args:
            symbol: 交易品种
            market: 市场名称
        """
        self.routing_rules[symbol] = market
        self.logger.info(f"设置路由规则: {symbol} -> {market}")
    
    def get_status_summary(self) -> Dict:
        """获取状态摘要
        
        Returns:
            Dict: 状态摘要
        """
        metrics = self.calculate_metrics()
        
        return {
            'status': self.status.value,
            'mode': self.mode.value,
            'pending_orders': len(self.pending_orders),
            'active_orders': len([o for o in self.orders.values() 
                                if o.status in [OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]]),
            'metrics': metrics.to_dict(),
            'stats': self.stats.copy(),
            'connectors': list(self.market_connectors.keys()),
            'algorithms': list(self.execution_algorithms.keys())
        }
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 执行引擎字典
        """
        return {
            'status': self.status.value,
            'mode': self.mode.value,
            'orders': {oid: order.to_dict() for oid, order in self.orders.items()},
            'trades': [trade.to_dict() for trade in self.trades],
            'metrics': self.calculate_metrics().to_dict(),
            'stats': self.stats.copy(),
            'routing_rules': self.routing_rules.copy()
        }
    
    def __str__(self) -> str:
        return f"ExecutionEngine(status={self.status.value}, orders={len(self.orders)}, trades={len(self.trades)})"
    
    def __repr__(self) -> str:
        return self.__str__()