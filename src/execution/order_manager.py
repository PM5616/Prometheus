"""Order Manager Module

订单管理器模块，负责订单的生命周期管理和状态跟踪。

主要功能：
- 订单创建和验证
- 订单状态管理
- 订单修改和取消
- 订单历史记录
- 订单查询和过滤
"""

import threading
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import json

from src.common.logging import get_logger
from src.common.exceptions.execution import OrderManagerError, OrderValidationError
from .engine import ExecutionOrder, OrderStatus, OrderType, OrderSide, Trade


class OrderValidationRule(Enum):
    """订单验证规则枚举"""
    MIN_QUANTITY = "min_quantity"        # 最小数量
    MAX_QUANTITY = "max_quantity"        # 最大数量
    MIN_PRICE = "min_price"              # 最小价格
    MAX_PRICE = "max_price"              # 最大价格
    PRICE_PRECISION = "price_precision"  # 价格精度
    QUANTITY_PRECISION = "quantity_precision"  # 数量精度
    TRADING_HOURS = "trading_hours"      # 交易时间
    SYMBOL_WHITELIST = "symbol_whitelist"  # 品种白名单
    SYMBOL_BLACKLIST = "symbol_blacklist"  # 品种黑名单


@dataclass
class OrderValidationConfig:
    """订单验证配置"""
    min_quantity: float = 0.0
    max_quantity: float = float('inf')
    min_price: float = 0.0
    max_price: float = float('inf')
    price_precision: int = 2
    quantity_precision: int = 4
    trading_start: str = "09:00:00"
    trading_end: str = "15:00:00"
    symbol_whitelist: List[str] = field(default_factory=list)
    symbol_blacklist: List[str] = field(default_factory=list)
    allow_weekend_trading: bool = False
    allow_holiday_trading: bool = False
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 配置字典
        """
        return {
            'min_quantity': self.min_quantity,
            'max_quantity': self.max_quantity,
            'min_price': self.min_price,
            'max_price': self.max_price,
            'price_precision': self.price_precision,
            'quantity_precision': self.quantity_precision,
            'trading_start': self.trading_start,
            'trading_end': self.trading_end,
            'symbol_whitelist': self.symbol_whitelist,
            'symbol_blacklist': self.symbol_blacklist,
            'allow_weekend_trading': self.allow_weekend_trading,
            'allow_holiday_trading': self.allow_holiday_trading
        }


@dataclass
class OrderFilter:
    """订单过滤器"""
    symbol: Optional[str] = None
    side: Optional[OrderSide] = None
    order_type: Optional[OrderType] = None
    status: Optional[OrderStatus] = None
    strategy_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_quantity: Optional[float] = None
    max_quantity: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    
    def matches(self, order: ExecutionOrder) -> bool:
        """检查订单是否匹配过滤条件
        
        Args:
            order: 执行订单
            
        Returns:
            bool: 是否匹配
        """
        if self.symbol and order.symbol != self.symbol:
            return False
        
        if self.side and order.side != self.side:
            return False
        
        if self.order_type and order.order_type != self.order_type:
            return False
        
        if self.status and order.status != self.status:
            return False
        
        if self.strategy_id and order.strategy_id != self.strategy_id:
            return False
        
        if self.start_time and order.created_time < self.start_time:
            return False
        
        if self.end_time and order.created_time > self.end_time:
            return False
        
        if self.min_quantity and order.quantity < self.min_quantity:
            return False
        
        if self.max_quantity and order.quantity > self.max_quantity:
            return False
        
        if self.min_price and order.price and order.price < self.min_price:
            return False
        
        if self.max_price and order.price and order.price > self.max_price:
            return False
        
        return True


@dataclass
class OrderStatistics:
    """订单统计信息"""
    total_orders: int = 0
    pending_orders: int = 0
    submitted_orders: int = 0
    accepted_orders: int = 0
    partially_filled_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    expired_orders: int = 0
    suspended_orders: int = 0
    
    total_quantity: float = 0.0
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    
    total_value: float = 0.0
    filled_value: float = 0.0
    
    avg_order_size: float = 0.0
    avg_fill_price: float = 0.0
    fill_rate: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 统计字典
        """
        return {
            'total_orders': self.total_orders,
            'pending_orders': self.pending_orders,
            'submitted_orders': self.submitted_orders,
            'accepted_orders': self.accepted_orders,
            'partially_filled_orders': self.partially_filled_orders,
            'filled_orders': self.filled_orders,
            'cancelled_orders': self.cancelled_orders,
            'rejected_orders': self.rejected_orders,
            'expired_orders': self.expired_orders,
            'suspended_orders': self.suspended_orders,
            'total_quantity': self.total_quantity,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'total_value': self.total_value,
            'filled_value': self.filled_value,
            'avg_order_size': self.avg_order_size,
            'avg_fill_price': self.avg_fill_price,
            'fill_rate': self.fill_rate,
            'last_updated': self.last_updated.isoformat()
        }


class OrderManager:
    """订单管理器
    
    负责订单的生命周期管理和状态跟踪。
    """
    
    def __init__(self, config: Dict = None):
        """初始化订单管理器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 订单存储
        self.orders: Dict[str, ExecutionOrder] = {}
        self.order_history: List[ExecutionOrder] = []
        
        # 索引
        self.orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_strategy: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_status: Dict[OrderStatus, List[str]] = defaultdict(list)
        
        # 验证配置
        self.validation_config = OrderValidationConfig()
        self._load_validation_config()
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 回调函数
        self.on_order_created_callbacks: List[Callable] = []
        self.on_order_updated_callbacks: List[Callable] = []
        self.on_order_cancelled_callbacks: List[Callable] = []
        self.on_order_filled_callbacks: List[Callable] = []
        
        # 日志记录
        self.logger = get_logger("OrderManager")
        
        # 统计信息
        self.stats = {
            'orders_created': 0,
            'orders_validated': 0,
            'orders_rejected': 0,
            'validation_errors': 0,
            'last_order_time': None
        }
        
        self.logger.info("订单管理器初始化完成")
    
    def _load_validation_config(self):
        """加载验证配置"""
        validation_config = self.config.get('validation', {})
        
        for key, value in validation_config.items():
            if hasattr(self.validation_config, key):
                setattr(self.validation_config, key, value)
    
    def create_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                    quantity: float, price: Optional[float] = None,
                    stop_price: Optional[float] = None, time_in_force: str = "GTC",
                    strategy_id: Optional[str] = None, metadata: Dict = None) -> ExecutionOrder:
        """创建订单
        
        Args:
            symbol: 交易品种
            side: 订单方向
            order_type: 订单类型
            quantity: 数量
            price: 价格（限价单必需）
            stop_price: 止损价格（止损单必需）
            time_in_force: 有效期类型
            strategy_id: 策略ID
            metadata: 元数据
            
        Returns:
            ExecutionOrder: 执行订单
            
        Raises:
            OrderValidationError: 订单验证失败
        """
        with self._lock:
            # 生成订单ID
            order_id = self._generate_order_id()
            
            # 创建订单对象
            order = ExecutionOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                strategy_id=strategy_id,
                metadata=metadata or {}
            )
            
            # 验证订单
            self._validate_order(order)
            
            # 保存订单
            self._store_order(order)
            
            # 更新统计
            self.stats['orders_created'] += 1
            self.stats['last_order_time'] = datetime.now()
            
            # 调用回调
            self._notify_order_created(order)
            
            self.logger.info(f"订单已创建: {order_id} {side.value} {quantity} {symbol}")
            return order
    
    def _generate_order_id(self) -> str:
        """生成订单ID
        
        Returns:
            str: 订单ID
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_part = uuid.uuid4().hex[:6]
        return f"ORD_{timestamp}_{random_part}"
    
    def _validate_order(self, order: ExecutionOrder):
        """验证订单
        
        Args:
            order: 执行订单
            
        Raises:
            OrderValidationError: 验证失败
        """
        errors = []
        
        try:
            # 基本验证
            if order.quantity <= 0:
                errors.append("订单数量必须大于0")
            
            if order.order_type == OrderType.LIMIT and order.price is None:
                errors.append("限价单必须指定价格")
            
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
                errors.append("止损单必须指定止损价格")
            
            # 数量验证
            if order.quantity < self.validation_config.min_quantity:
                errors.append(f"订单数量不能小于 {self.validation_config.min_quantity}")
            
            if order.quantity > self.validation_config.max_quantity:
                errors.append(f"订单数量不能大于 {self.validation_config.max_quantity}")
            
            # 价格验证
            if order.price is not None:
                if order.price < self.validation_config.min_price:
                    errors.append(f"订单价格不能小于 {self.validation_config.min_price}")
                
                if order.price > self.validation_config.max_price:
                    errors.append(f"订单价格不能大于 {self.validation_config.max_price}")
            
            # 精度验证
            if not self._check_price_precision(order.price):
                errors.append(f"价格精度不符合要求（{self.validation_config.price_precision}位小数）")
            
            if not self._check_quantity_precision(order.quantity):
                errors.append(f"数量精度不符合要求（{self.validation_config.quantity_precision}位小数）")
            
            # 品种验证
            if not self._validate_symbol(order.symbol):
                errors.append(f"不支持的交易品种: {order.symbol}")
            
            # 交易时间验证
            if not self._validate_trading_hours():
                errors.append("当前不在交易时间内")
            
            if errors:
                self.stats['validation_errors'] += 1
                raise OrderValidationError(f"订单验证失败: {'; '.join(errors)}")
            
            self.stats['orders_validated'] += 1
            
        except OrderValidationError:
            raise
        except Exception as e:
            self.stats['validation_errors'] += 1
            raise OrderValidationError(f"订单验证异常: {e}")
    
    def _check_price_precision(self, price: Optional[float]) -> bool:
        """检查价格精度
        
        Args:
            price: 价格
            
        Returns:
            bool: 是否符合精度要求
        """
        if price is None:
            return True
        
        decimal_places = len(str(price).split('.')[-1]) if '.' in str(price) else 0
        return decimal_places <= self.validation_config.price_precision
    
    def _check_quantity_precision(self, quantity: float) -> bool:
        """检查数量精度
        
        Args:
            quantity: 数量
            
        Returns:
            bool: 是否符合精度要求
        """
        decimal_places = len(str(quantity).split('.')[-1]) if '.' in str(quantity) else 0
        return decimal_places <= self.validation_config.quantity_precision
    
    def _validate_symbol(self, symbol: str) -> bool:
        """验证交易品种
        
        Args:
            symbol: 交易品种
            
        Returns:
            bool: 是否有效
        """
        # 检查黑名单
        if self.validation_config.symbol_blacklist and symbol in self.validation_config.symbol_blacklist:
            return False
        
        # 检查白名单
        if self.validation_config.symbol_whitelist and symbol not in self.validation_config.symbol_whitelist:
            return False
        
        return True
    
    def _validate_trading_hours(self) -> bool:
        """验证交易时间
        
        Returns:
            bool: 是否在交易时间内
        """
        now = datetime.now()
        
        # 检查周末
        if not self.validation_config.allow_weekend_trading and now.weekday() >= 5:
            return False
        
        # 检查交易时间
        current_time = now.time()
        start_time = datetime.strptime(self.validation_config.trading_start, "%H:%M:%S").time()
        end_time = datetime.strptime(self.validation_config.trading_end, "%H:%M:%S").time()
        
        return start_time <= current_time <= end_time
    
    def _store_order(self, order: ExecutionOrder):
        """存储订单
        
        Args:
            order: 执行订单
        """
        # 主存储
        self.orders[order.order_id] = order
        
        # 更新索引
        self.orders_by_symbol[order.symbol].append(order.order_id)
        
        if order.strategy_id:
            self.orders_by_strategy[order.strategy_id].append(order.order_id)
        
        self.orders_by_status[order.status].append(order.order_id)
    
    def update_order(self, order_id: str, **kwargs) -> bool:
        """更新订单
        
        Args:
            order_id: 订单ID
            **kwargs: 更新参数
            
        Returns:
            bool: 是否成功更新
        """
        with self._lock:
            if order_id not in self.orders:
                return False
            
            order = self.orders[order_id]
            old_status = order.status
            
            # 更新订单属性
            for key, value in kwargs.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            
            order.updated_time = datetime.now()
            
            # 更新状态索引
            if order.status != old_status:
                self._update_status_index(order_id, old_status, order.status)
            
            # 调用回调
            self._notify_order_updated(order)
            
            # 特殊状态回调
            if order.status == OrderStatus.CANCELLED:
                self._notify_order_cancelled(order)
            elif order.status == OrderStatus.FILLED:
                self._notify_order_filled(order)
            
            self.logger.info(f"订单已更新: {order_id}")
            return True
    
    def _update_status_index(self, order_id: str, old_status: OrderStatus, new_status: OrderStatus):
        """更新状态索引
        
        Args:
            order_id: 订单ID
            old_status: 旧状态
            new_status: 新状态
        """
        # 从旧状态索引中移除
        if order_id in self.orders_by_status[old_status]:
            self.orders_by_status[old_status].remove(order_id)
        
        # 添加到新状态索引
        self.orders_by_status[new_status].append(order_id)
    
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
            
            # 检查是否可以取消
            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, 
                                   OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]:
                return False
            
            # 取消订单
            order.cancel(reason)
            
            # 更新索引
            self._update_status_index(order_id, OrderStatus.ACCEPTED, OrderStatus.CANCELLED)
            
            # 调用回调
            self._notify_order_updated(order)
            self._notify_order_cancelled(order)
            
            self.logger.info(f"订单已取消: {order_id} - {reason}")
            return True
    
    def cancel_orders_by_symbol(self, symbol: str, reason: str = "") -> int:
        """按品种取消订单
        
        Args:
            symbol: 交易品种
            reason: 取消原因
            
        Returns:
            int: 取消的订单数量
        """
        cancelled_count = 0
        
        order_ids = self.orders_by_symbol.get(symbol, [])
        for order_id in order_ids.copy():
            if self.cancel_order(order_id, reason):
                cancelled_count += 1
        
        return cancelled_count
    
    def cancel_orders_by_strategy(self, strategy_id: str, reason: str = "") -> int:
        """按策略取消订单
        
        Args:
            strategy_id: 策略ID
            reason: 取消原因
            
        Returns:
            int: 取消的订单数量
        """
        cancelled_count = 0
        
        order_ids = self.orders_by_strategy.get(strategy_id, [])
        for order_id in order_ids.copy():
            if self.cancel_order(order_id, reason):
                cancelled_count += 1
        
        return cancelled_count
    
    def get_order(self, order_id: str) -> Optional[ExecutionOrder]:
        """获取订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            Optional[ExecutionOrder]: 执行订单
        """
        return self.orders.get(order_id)
    
    def get_orders(self, filter_obj: Optional[OrderFilter] = None) -> List[ExecutionOrder]:
        """获取订单列表
        
        Args:
            filter_obj: 过滤器
            
        Returns:
            List[ExecutionOrder]: 订单列表
        """
        orders = list(self.orders.values())
        
        if filter_obj:
            orders = [order for order in orders if filter_obj.matches(order)]
        
        return orders
    
    def get_orders_by_symbol(self, symbol: str) -> List[ExecutionOrder]:
        """按品种获取订单
        
        Args:
            symbol: 交易品种
            
        Returns:
            List[ExecutionOrder]: 订单列表
        """
        order_ids = self.orders_by_symbol.get(symbol, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]
    
    def get_orders_by_strategy(self, strategy_id: str) -> List[ExecutionOrder]:
        """按策略获取订单
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            List[ExecutionOrder]: 订单列表
        """
        order_ids = self.orders_by_strategy.get(strategy_id, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[ExecutionOrder]:
        """按状态获取订单
        
        Args:
            status: 订单状态
            
        Returns:
            List[ExecutionOrder]: 订单列表
        """
        order_ids = self.orders_by_status.get(status, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]
    
    def get_active_orders(self) -> List[ExecutionOrder]:
        """获取活跃订单
        
        Returns:
            List[ExecutionOrder]: 活跃订单列表
        """
        active_statuses = [OrderStatus.PENDING, OrderStatus.SUBMITTED, 
                          OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]
        
        active_orders = []
        for status in active_statuses:
            active_orders.extend(self.get_orders_by_status(status))
        
        return active_orders
    
    def calculate_statistics(self, symbol: Optional[str] = None,
                           strategy_id: Optional[str] = None) -> OrderStatistics:
        """计算订单统计信息
        
        Args:
            symbol: 交易品种（可选）
            strategy_id: 策略ID（可选）
            
        Returns:
            OrderStatistics: 统计信息
        """
        # 获取订单列表
        if symbol:
            orders = self.get_orders_by_symbol(symbol)
        elif strategy_id:
            orders = self.get_orders_by_strategy(strategy_id)
        else:
            orders = list(self.orders.values())
        
        stats = OrderStatistics()
        
        # 基本统计
        stats.total_orders = len(orders)
        
        # 按状态统计
        for order in orders:
            if order.status == OrderStatus.PENDING:
                stats.pending_orders += 1
            elif order.status == OrderStatus.SUBMITTED:
                stats.submitted_orders += 1
            elif order.status == OrderStatus.ACCEPTED:
                stats.accepted_orders += 1
            elif order.status == OrderStatus.PARTIALLY_FILLED:
                stats.partially_filled_orders += 1
            elif order.status == OrderStatus.FILLED:
                stats.filled_orders += 1
            elif order.status == OrderStatus.CANCELLED:
                stats.cancelled_orders += 1
            elif order.status == OrderStatus.REJECTED:
                stats.rejected_orders += 1
            elif order.status == OrderStatus.EXPIRED:
                stats.expired_orders += 1
            elif order.status == OrderStatus.SUSPENDED:
                stats.suspended_orders += 1
            
            # 数量和金额统计
            stats.total_quantity += order.quantity
            stats.filled_quantity += order.filled_quantity
            stats.remaining_quantity += order.remaining_quantity
            
            if order.price:
                stats.total_value += order.quantity * order.price
            
            if order.filled_quantity > 0 and order.avg_fill_price > 0:
                stats.filled_value += order.filled_quantity * order.avg_fill_price
        
        # 计算平均值
        if stats.total_orders > 0:
            stats.avg_order_size = stats.total_quantity / stats.total_orders
            stats.fill_rate = stats.filled_orders / stats.total_orders
        
        if stats.filled_quantity > 0:
            stats.avg_fill_price = stats.filled_value / stats.filled_quantity
        
        stats.last_updated = datetime.now()
        
        return stats
    
    def archive_completed_orders(self, days: int = 30) -> int:
        """归档已完成的订单
        
        Args:
            days: 保留天数
            
        Returns:
            int: 归档的订单数量
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        archived_count = 0
        
        completed_statuses = [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                             OrderStatus.REJECTED, OrderStatus.EXPIRED]
        
        with self._lock:
            orders_to_archive = []
            
            for order in self.orders.values():
                if (order.status in completed_statuses and 
                    order.updated_time < cutoff_time):
                    orders_to_archive.append(order)
            
            for order in orders_to_archive:
                # 移动到历史记录
                self.order_history.append(order)
                
                # 从主存储中删除
                del self.orders[order.order_id]
                
                # 更新索引
                self._remove_from_indexes(order)
                
                archived_count += 1
        
        self.logger.info(f"已归档 {archived_count} 个订单")
        return archived_count
    
    def _remove_from_indexes(self, order: ExecutionOrder):
        """从索引中移除订单
        
        Args:
            order: 执行订单
        """
        # 从品种索引中移除
        if order.order_id in self.orders_by_symbol[order.symbol]:
            self.orders_by_symbol[order.symbol].remove(order.order_id)
        
        # 从策略索引中移除
        if order.strategy_id and order.order_id in self.orders_by_strategy[order.strategy_id]:
            self.orders_by_strategy[order.strategy_id].remove(order.order_id)
        
        # 从状态索引中移除
        if order.order_id in self.orders_by_status[order.status]:
            self.orders_by_status[order.status].remove(order.order_id)
    
    def _notify_order_created(self, order: ExecutionOrder):
        """通知订单创建
        
        Args:
            order: 执行订单
        """
        for callback in self.on_order_created_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"订单创建回调执行失败: {e}")
    
    def _notify_order_updated(self, order: ExecutionOrder):
        """通知订单更新
        
        Args:
            order: 执行订单
        """
        for callback in self.on_order_updated_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"订单更新回调执行失败: {e}")
    
    def _notify_order_cancelled(self, order: ExecutionOrder):
        """通知订单取消
        
        Args:
            order: 执行订单
        """
        for callback in self.on_order_cancelled_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"订单取消回调执行失败: {e}")
    
    def _notify_order_filled(self, order: ExecutionOrder):
        """通知订单成交
        
        Args:
            order: 执行订单
        """
        for callback in self.on_order_filled_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"订单成交回调执行失败: {e}")
    
    def add_order_created_callback(self, callback: Callable):
        """添加订单创建回调
        
        Args:
            callback: 回调函数
        """
        self.on_order_created_callbacks.append(callback)
    
    def add_order_updated_callback(self, callback: Callable):
        """添加订单更新回调
        
        Args:
            callback: 回调函数
        """
        self.on_order_updated_callbacks.append(callback)
    
    def add_order_cancelled_callback(self, callback: Callable):
        """添加订单取消回调
        
        Args:
            callback: 回调函数
        """
        self.on_order_cancelled_callbacks.append(callback)
    
    def add_order_filled_callback(self, callback: Callable):
        """添加订单成交回调
        
        Args:
            callback: 回调函数
        """
        self.on_order_filled_callbacks.append(callback)
    
    def get_validation_config(self) -> OrderValidationConfig:
        """获取验证配置
        
        Returns:
            OrderValidationConfig: 验证配置
        """
        return self.validation_config
    
    def update_validation_config(self, **kwargs):
        """更新验证配置
        
        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.validation_config, key):
                setattr(self.validation_config, key, value)
        
        self.logger.info("验证配置已更新")
    
    def get_summary(self) -> Dict:
        """获取摘要信息
        
        Returns:
            Dict: 摘要信息
        """
        stats = self.calculate_statistics()
        
        return {
            'total_orders': len(self.orders),
            'active_orders': len(self.get_active_orders()),
            'archived_orders': len(self.order_history),
            'statistics': stats.to_dict(),
            'validation_config': self.validation_config.to_dict(),
            'stats': self.stats.copy(),
            'symbols': list(self.orders_by_symbol.keys()),
            'strategies': list(self.orders_by_strategy.keys())
        }
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 订单管理器字典
        """
        return {
            'orders': {oid: order.to_dict() for oid, order in self.orders.items()},
            'order_history': [order.to_dict() for order in self.order_history],
            'statistics': self.calculate_statistics().to_dict(),
            'validation_config': self.validation_config.to_dict(),
            'stats': self.stats.copy()
        }
    
    def __str__(self) -> str:
        return f"OrderManager(orders={len(self.orders)}, active={len(self.get_active_orders())})"
    
    def __repr__(self) -> str:
        return self.__str__()