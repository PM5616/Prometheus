"""Execution Algorithm Module

执行算法模块，实现各种订单执行算法。

主要功能：
- TWAP (时间加权平均价格)
- VWAP (成交量加权平均价格)
- Implementation Shortfall
- Iceberg Orders
- Sniper Algorithm
- 自定义算法框架
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
import math
import numpy as np

from src.common.logging import get_logger
from src.common.exceptions.execution import ExecutionAlgorithmError
from .engine import ExecutionOrder, Trade, OrderStatus, OrderType, OrderSide


class AlgorithmType(Enum):
    """算法类型枚举"""
    TWAP = "twap"                    # 时间加权平均价格
    VWAP = "vwap"                    # 成交量加权平均价格
    IMPLEMENTATION_SHORTFALL = "is"   # 实施缺口
    ICEBERG = "iceberg"              # 冰山订单
    SNIPER = "sniper"                # 狙击算法
    MARKET_MAKING = "market_making"  # 做市算法
    CUSTOM = "custom"                # 自定义算法


class AlgorithmStatus(Enum):
    """算法状态枚举"""
    PENDING = "pending"        # 待执行
    RUNNING = "running"        # 运行中
    PAUSED = "paused"          # 暂停
    COMPLETED = "completed"    # 完成
    CANCELLED = "cancelled"    # 取消
    ERROR = "error"            # 错误


@dataclass
class AlgorithmParameters:
    """算法参数"""
    # 通用参数
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    participation_rate: float = 0.1  # 参与率
    urgency: float = 0.5            # 紧急程度 (0-1)
    
    # TWAP参数
    twap_intervals: int = 10        # TWAP分割数量
    twap_randomize: bool = True     # 是否随机化
    
    # VWAP参数
    vwap_lookback_period: int = 20  # VWAP回看期间
    vwap_volume_profile: List[float] = field(default_factory=list)  # 成交量分布
    
    # 冰山订单参数
    iceberg_slice_size: float = 0.0  # 冰山切片大小
    iceberg_variance: float = 0.1    # 切片大小变化
    
    # 实施缺口参数
    is_risk_aversion: float = 0.5    # 风险厌恶系数
    is_market_impact: float = 0.1    # 市场冲击系数
    
    # 价格限制
    limit_price: Optional[float] = None
    price_tolerance: float = 0.01    # 价格容忍度
    
    # 风险控制
    max_position_size: Optional[float] = None
    max_order_size: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 参数字典
        """
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'participation_rate': self.participation_rate,
            'urgency': self.urgency,
            'twap_intervals': self.twap_intervals,
            'twap_randomize': self.twap_randomize,
            'vwap_lookback_period': self.vwap_lookback_period,
            'vwap_volume_profile': self.vwap_volume_profile,
            'iceberg_slice_size': self.iceberg_slice_size,
            'iceberg_variance': self.iceberg_variance,
            'is_risk_aversion': self.is_risk_aversion,
            'is_market_impact': self.is_market_impact,
            'limit_price': self.limit_price,
            'price_tolerance': self.price_tolerance,
            'max_position_size': self.max_position_size,
            'max_order_size': self.max_order_size,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price
        }


@dataclass
class AlgorithmState:
    """算法状态"""
    algorithm_id: str
    status: AlgorithmStatus = AlgorithmStatus.PENDING
    
    # 执行进度
    total_quantity: float = 0.0
    executed_quantity: float = 0.0
    remaining_quantity: float = 0.0
    
    # 价格信息
    avg_execution_price: float = 0.0
    best_price: float = 0.0
    worst_price: float = 0.0
    
    # 时间信息
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    
    # 子订单
    child_orders: List[str] = field(default_factory=list)
    completed_orders: List[str] = field(default_factory=list)
    
    # 统计信息
    total_trades: int = 0
    total_commission: float = 0.0
    slippage: float = 0.0
    
    # 错误信息
    error_message: str = ""
    
    def update_execution(self, quantity: float, price: float, commission: float = 0.0):
        """更新执行信息
        
        Args:
            quantity: 执行数量
            price: 执行价格
            commission: 手续费
        """
        if self.executed_quantity == 0:
            self.avg_execution_price = price
            self.best_price = price
            self.worst_price = price
        else:
            # 更新平均价格
            total_value = self.avg_execution_price * self.executed_quantity + price * quantity
            self.executed_quantity += quantity
            self.avg_execution_price = total_value / self.executed_quantity
            
            # 更新最优/最差价格
            self.best_price = min(self.best_price, price)
            self.worst_price = max(self.worst_price, price)
        
        self.remaining_quantity = self.total_quantity - self.executed_quantity
        self.total_trades += 1
        self.total_commission += commission
        self.last_update_time = datetime.now()
    
    def calculate_completion_rate(self) -> float:
        """计算完成率
        
        Returns:
            float: 完成率 (0-1)
        """
        if self.total_quantity <= 0:
            return 0.0
        
        return self.executed_quantity / self.total_quantity
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 状态字典
        """
        return {
            'algorithm_id': self.algorithm_id,
            'status': self.status.value,
            'total_quantity': self.total_quantity,
            'executed_quantity': self.executed_quantity,
            'remaining_quantity': self.remaining_quantity,
            'completion_rate': self.calculate_completion_rate(),
            'avg_execution_price': self.avg_execution_price,
            'best_price': self.best_price,
            'worst_price': self.worst_price,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'child_orders': self.child_orders,
            'completed_orders': self.completed_orders,
            'total_trades': self.total_trades,
            'total_commission': self.total_commission,
            'slippage': self.slippage,
            'error_message': self.error_message
        }


class BaseExecutionAlgorithm(ABC):
    """执行算法基类
    
    定义执行算法的基本接口。
    """
    
    def __init__(self, algorithm_id: str, algorithm_type: AlgorithmType,
                 order: ExecutionOrder, parameters: AlgorithmParameters):
        """初始化执行算法
        
        Args:
            algorithm_id: 算法ID
            algorithm_type: 算法类型
            order: 父订单
            parameters: 算法参数
        """
        self.algorithm_id = algorithm_id
        self.algorithm_type = algorithm_type
        self.parent_order = order
        self.parameters = parameters
        
        # 算法状态
        self.state = AlgorithmState(
            algorithm_id=algorithm_id,
            total_quantity=order.quantity
        )
        
        # 市场数据
        self.current_price: Optional[float] = None
        self.bid_price: Optional[float] = None
        self.ask_price: Optional[float] = None
        self.volume: Optional[float] = None
        
        # 回调函数
        self.on_child_order_callbacks: List[Callable] = []
        self.on_trade_callbacks: List[Callable] = []
        self.on_completion_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 运行控制
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # 日志记录
        self.logger = get_logger(f"ExecutionAlgorithm-{algorithm_id}")
        
        self.logger.info(f"执行算法初始化: {algorithm_type.value} - {algorithm_id}")
    
    @abstractmethod
    async def execute(self):
        """执行算法"""
        pass
    
    @abstractmethod
    def calculate_next_order(self) -> Optional[ExecutionOrder]:
        """计算下一个子订单
        
        Returns:
            Optional[ExecutionOrder]: 下一个子订单
        """
        pass
    
    async def start(self):
        """启动算法"""
        if self._running:
            return
        
        with self._lock:
            self._running = True
            self.state.status = AlgorithmStatus.RUNNING
            self.state.start_time = datetime.now()
            
            # 启动执行任务
            self._task = asyncio.create_task(self.execute())
            
            self.logger.info(f"算法启动: {self.algorithm_id}")
    
    async def stop(self):
        """停止算法"""
        if not self._running:
            return
        
        with self._lock:
            self._running = False
            
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            
            if self.state.status == AlgorithmStatus.RUNNING:
                self.state.status = AlgorithmStatus.CANCELLED
            
            self.state.end_time = datetime.now()
            
            self.logger.info(f"算法停止: {self.algorithm_id}")
    
    async def pause(self):
        """暂停算法"""
        with self._lock:
            if self.state.status == AlgorithmStatus.RUNNING:
                self.state.status = AlgorithmStatus.PAUSED
                self.logger.info(f"算法暂停: {self.algorithm_id}")
    
    async def resume(self):
        """恢复算法"""
        with self._lock:
            if self.state.status == AlgorithmStatus.PAUSED:
                self.state.status = AlgorithmStatus.RUNNING
                self.logger.info(f"算法恢复: {self.algorithm_id}")
    
    def update_market_data(self, price: float, bid: float = None, ask: float = None, volume: float = None):
        """更新市场数据
        
        Args:
            price: 当前价格
            bid: 买价
            ask: 卖价
            volume: 成交量
        """
        with self._lock:
            self.current_price = price
            if bid is not None:
                self.bid_price = bid
            if ask is not None:
                self.ask_price = ask
            if volume is not None:
                self.volume = volume
    
    def handle_trade(self, trade: Trade):
        """处理成交回报
        
        Args:
            trade: 成交信息
        """
        with self._lock:
            # 更新执行状态
            self.state.update_execution(
                quantity=trade.quantity,
                price=trade.price,
                commission=trade.commission
            )
            
            # 检查是否完成
            if self.state.remaining_quantity <= 0:
                self.state.status = AlgorithmStatus.COMPLETED
                self.state.end_time = datetime.now()
                self._notify_completion()
            
            # 通知成交
            self._notify_trade(trade)
            
            self.logger.info(f"处理成交: {trade.trade_id} - {trade.quantity}@{trade.price}")
    
    def handle_order_status(self, order_id: str, status: OrderStatus):
        """处理订单状态更新
        
        Args:
            order_id: 订单ID
            status: 订单状态
        """
        with self._lock:
            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                if order_id in self.state.child_orders and order_id not in self.state.completed_orders:
                    self.state.completed_orders.append(order_id)
    
    def _create_child_order(self, quantity: float, price: float = None, 
                           order_type: OrderType = OrderType.LIMIT) -> ExecutionOrder:
        """创建子订单
        
        Args:
            quantity: 订单数量
            price: 订单价格
            order_type: 订单类型
            
        Returns:
            ExecutionOrder: 子订单
        """
        child_order = ExecutionOrder(
            order_id=str(uuid.uuid4()),
            symbol=self.parent_order.symbol,
            side=self.parent_order.side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            strategy_id=self.parent_order.strategy_id,
            parent_order_id=self.parent_order.order_id,
            algorithm_id=self.algorithm_id
        )
        
        self.state.child_orders.append(child_order.order_id)
        return child_order
    
    def _notify_child_order(self, order: ExecutionOrder):
        """通知子订单创建
        
        Args:
            order: 子订单
        """
        for callback in self.on_child_order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"子订单回调执行失败: {e}")
    
    def _notify_trade(self, trade: Trade):
        """通知成交
        
        Args:
            trade: 成交信息
        """
        for callback in self.on_trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                self.logger.error(f"成交回调执行失败: {e}")
    
    def _notify_completion(self):
        """通知算法完成"""
        for callback in self.on_completion_callbacks:
            try:
                callback(self.algorithm_id, self.state)
            except Exception as e:
                self.logger.error(f"完成回调执行失败: {e}")
    
    def _notify_error(self, error: Exception):
        """通知错误
        
        Args:
            error: 错误信息
        """
        self.state.status = AlgorithmStatus.ERROR
        self.state.error_message = str(error)
        
        for callback in self.on_error_callbacks:
            try:
                callback(self.algorithm_id, error)
            except Exception as e:
                self.logger.error(f"错误回调执行失败: {e}")
    
    def add_child_order_callback(self, callback: Callable):
        """添加子订单回调
        
        Args:
            callback: 回调函数
        """
        self.on_child_order_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable):
        """添加成交回调
        
        Args:
            callback: 回调函数
        """
        self.on_trade_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable):
        """添加完成回调
        
        Args:
            callback: 回调函数
        """
        self.on_completion_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """添加错误回调
        
        Args:
            callback: 回调函数
        """
        self.on_error_callbacks.append(callback)
    
    def get_state(self) -> AlgorithmState:
        """获取算法状态
        
        Returns:
            AlgorithmState: 算法状态
        """
        return self.state
    
    def is_running(self) -> bool:
        """检查是否运行中
        
        Returns:
            bool: 是否运行中
        """
        return self._running and self.state.status == AlgorithmStatus.RUNNING
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 算法字典
        """
        return {
            'algorithm_id': self.algorithm_id,
            'algorithm_type': self.algorithm_type.value,
            'parent_order': self.parent_order.to_dict(),
            'parameters': self.parameters.to_dict(),
            'state': self.state.to_dict(),
            'current_price': self.current_price,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'volume': self.volume,
            'running': self._running
        }
    
    def __str__(self) -> str:
        return f"ExecutionAlgorithm(id={self.algorithm_id}, type={self.algorithm_type.value}, status={self.state.status.value})"
    
    def __repr__(self) -> str:
        return self.__str__()


class TWAPAlgorithm(BaseExecutionAlgorithm):
    """TWAP (时间加权平均价格) 算法
    
    将大订单分割成多个小订单，在指定时间内均匀执行。
    """
    
    def __init__(self, algorithm_id: str, order: ExecutionOrder, parameters: AlgorithmParameters):
        """初始化TWAP算法
        
        Args:
            algorithm_id: 算法ID
            order: 父订单
            parameters: 算法参数
        """
        super().__init__(algorithm_id, AlgorithmType.TWAP, order, parameters)
        
        # TWAP特定状态
        self.slice_intervals: List[Tuple[datetime, datetime, float]] = []
        self.current_slice_index = 0
        
        # 计算时间切片
        self._calculate_time_slices()
    
    def _calculate_time_slices(self):
        """计算时间切片"""
        start_time = self.parameters.start_time or datetime.now()
        end_time = self.parameters.end_time or (start_time + timedelta(hours=1))
        
        total_duration = (end_time - start_time).total_seconds()
        slice_duration = total_duration / self.parameters.twap_intervals
        
        remaining_quantity = self.state.total_quantity
        
        for i in range(self.parameters.twap_intervals):
            slice_start = start_time + timedelta(seconds=i * slice_duration)
            slice_end = start_time + timedelta(seconds=(i + 1) * slice_duration)
            
            # 计算切片数量
            if i == self.parameters.twap_intervals - 1:
                # 最后一个切片包含所有剩余数量
                slice_quantity = remaining_quantity
            else:
                base_quantity = self.state.total_quantity / self.parameters.twap_intervals
                
                if self.parameters.twap_randomize:
                    # 添加随机变化 (±10%)
                    variance = base_quantity * 0.1
                    import random
                    slice_quantity = base_quantity + random.uniform(-variance, variance)
                    slice_quantity = max(0, min(slice_quantity, remaining_quantity))
                else:
                    slice_quantity = base_quantity
            
            self.slice_intervals.append((slice_start, slice_end, slice_quantity))
            remaining_quantity -= slice_quantity
        
        self.logger.info(f"TWAP时间切片计算完成: {len(self.slice_intervals)}个切片")
    
    async def execute(self):
        """执行TWAP算法"""
        try:
            while self._running and self.current_slice_index < len(self.slice_intervals):
                if self.state.status != AlgorithmStatus.RUNNING:
                    await asyncio.sleep(0.1)
                    continue
                
                current_time = datetime.now()
                slice_start, slice_end, slice_quantity = self.slice_intervals[self.current_slice_index]
                
                # 检查是否到达执行时间
                if current_time < slice_start:
                    await asyncio.sleep(0.1)
                    continue
                
                # 检查是否超过结束时间
                if current_time > slice_end:
                    self.current_slice_index += 1
                    continue
                
                # 创建并提交子订单
                child_order = self.calculate_next_order()
                if child_order:
                    self._notify_child_order(child_order)
                    self.logger.info(f"提交TWAP子订单: {child_order.order_id} - {child_order.quantity}")
                
                # 等待到下一个切片
                self.current_slice_index += 1
                
                if self.current_slice_index < len(self.slice_intervals):
                    next_slice_start = self.slice_intervals[self.current_slice_index][0]
                    wait_time = (next_slice_start - current_time).total_seconds()
                    if wait_time > 0:
                        await asyncio.sleep(min(wait_time, 1.0))
            
            # 检查是否完成
            if self.state.remaining_quantity <= 0 or self.current_slice_index >= len(self.slice_intervals):
                self.state.status = AlgorithmStatus.COMPLETED
                self.state.end_time = datetime.now()
                self._notify_completion()
                
        except Exception as e:
            self.logger.error(f"TWAP算法执行异常: {e}")
            self._notify_error(e)
    
    def calculate_next_order(self) -> Optional[ExecutionOrder]:
        """计算下一个TWAP子订单
        
        Returns:
            Optional[ExecutionOrder]: 下一个子订单
        """
        if self.current_slice_index >= len(self.slice_intervals):
            return None
        
        _, _, slice_quantity = self.slice_intervals[self.current_slice_index]
        
        # 调整数量以不超过剩余数量
        order_quantity = min(slice_quantity, self.state.remaining_quantity)
        
        if order_quantity <= 0:
            return None
        
        # 计算订单价格
        order_price = self._calculate_order_price()
        
        return self._create_child_order(
            quantity=order_quantity,
            price=order_price,
            order_type=OrderType.LIMIT if order_price else OrderType.MARKET
        )
    
    def _calculate_order_price(self) -> Optional[float]:
        """计算订单价格
        
        Returns:
            Optional[float]: 订单价格
        """
        if self.parameters.limit_price:
            return self.parameters.limit_price
        
        if not self.current_price:
            return None
        
        # 根据紧急程度调整价格
        if self.parent_order.side == OrderSide.BUY:
            if self.ask_price:
                # 买单：在ask价格基础上根据紧急程度调整
                price_adjustment = (self.ask_price - self.current_price) * self.parameters.urgency
                return self.current_price + price_adjustment
            else:
                return self.current_price * (1 + self.parameters.price_tolerance)
        else:
            if self.bid_price:
                # 卖单：在bid价格基础上根据紧急程度调整
                price_adjustment = (self.current_price - self.bid_price) * self.parameters.urgency
                return self.current_price - price_adjustment
            else:
                return self.current_price * (1 - self.parameters.price_tolerance)


class VWAPAlgorithm(BaseExecutionAlgorithm):
    """VWAP (成交量加权平均价格) 算法
    
    根据历史成交量分布来分配订单执行。
    """
    
    def __init__(self, algorithm_id: str, order: ExecutionOrder, parameters: AlgorithmParameters):
        """初始化VWAP算法
        
        Args:
            algorithm_id: 算法ID
            order: 父订单
            parameters: 算法参数
        """
        super().__init__(algorithm_id, AlgorithmType.VWAP, order, parameters)
        
        # VWAP特定状态
        self.volume_profile = parameters.vwap_volume_profile or self._generate_default_volume_profile()
        self.target_participation_rate = parameters.participation_rate
        self.current_market_volume = 0.0
        self.executed_volume = 0.0
    
    def _generate_default_volume_profile(self) -> List[float]:
        """生成默认成交量分布
        
        Returns:
            List[float]: 成交量分布
        """
        # 简化的U型分布（开盘和收盘成交量较大）
        profile = []
        intervals = 24  # 24小时
        
        for i in range(intervals):
            # U型分布
            if i < 2 or i > 20:  # 开盘和收盘时段
                volume_weight = 0.8
            elif 10 <= i <= 14:  # 午间时段
                volume_weight = 0.3
            else:
                volume_weight = 0.5
            
            profile.append(volume_weight)
        
        # 归一化
        total_weight = sum(profile)
        return [w / total_weight for w in profile]
    
    async def execute(self):
        """执行VWAP算法"""
        try:
            while self._running and self.state.remaining_quantity > 0:
                if self.state.status != AlgorithmStatus.RUNNING:
                    await asyncio.sleep(0.1)
                    continue
                
                # 计算目标执行量
                target_volume = self._calculate_target_volume()
                
                if target_volume > 0:
                    child_order = self.calculate_next_order()
                    if child_order:
                        self._notify_child_order(child_order)
                        self.logger.info(f"提交VWAP子订单: {child_order.order_id} - {child_order.quantity}")
                
                await asyncio.sleep(1.0)  # 每秒检查一次
            
            if self.state.remaining_quantity <= 0:
                self.state.status = AlgorithmStatus.COMPLETED
                self.state.end_time = datetime.now()
                self._notify_completion()
                
        except Exception as e:
            self.logger.error(f"VWAP算法执行异常: {e}")
            self._notify_error(e)
    
    def _calculate_target_volume(self) -> float:
        """计算目标执行量
        
        Returns:
            float: 目标执行量
        """
        if not self.volume:
            return 0.0
        
        # 根据当前时间获取成交量权重
        current_hour = datetime.now().hour
        volume_weight = self.volume_profile[current_hour] if current_hour < len(self.volume_profile) else 0.5
        
        # 计算目标参与量
        target_participation = self.volume * self.target_participation_rate * volume_weight
        
        # 考虑已执行量
        return max(0, target_participation - self.executed_volume)
    
    def calculate_next_order(self) -> Optional[ExecutionOrder]:
        """计算下一个VWAP子订单
        
        Returns:
            Optional[ExecutionOrder]: 下一个子订单
        """
        target_volume = self._calculate_target_volume()
        
        if target_volume <= 0:
            return None
        
        # 限制订单大小
        max_order_size = self.parameters.max_order_size or (self.state.remaining_quantity * 0.1)
        order_quantity = min(target_volume, max_order_size, self.state.remaining_quantity)
        
        if order_quantity <= 0:
            return None
        
        # 计算订单价格（接近VWAP）
        order_price = self._calculate_vwap_price()
        
        return self._create_child_order(
            quantity=order_quantity,
            price=order_price,
            order_type=OrderType.LIMIT if order_price else OrderType.MARKET
        )
    
    def _calculate_vwap_price(self) -> Optional[float]:
        """计算VWAP价格
        
        Returns:
            Optional[float]: VWAP价格
        """
        if not self.current_price:
            return None
        
        # 简化的VWAP计算（实际应该使用历史数据）
        if self.state.avg_execution_price > 0:
            # 使用已执行的平均价格作为VWAP参考
            vwap_price = self.state.avg_execution_price
        else:
            vwap_price = self.current_price
        
        # 根据市场方向调整
        if self.parent_order.side == OrderSide.BUY:
            return min(vwap_price * (1 + self.parameters.price_tolerance), 
                      self.parameters.limit_price or float('inf'))
        else:
            return max(vwap_price * (1 - self.parameters.price_tolerance),
                      self.parameters.limit_price or 0)
    
    def update_market_data(self, price: float, bid: float = None, ask: float = None, volume: float = None):
        """更新市场数据
        
        Args:
            price: 当前价格
            bid: 买价
            ask: 卖价
            volume: 成交量
        """
        super().update_market_data(price, bid, ask, volume)
        
        # 更新市场成交量
        if volume:
            self.current_market_volume = volume


class IcebergAlgorithm(BaseExecutionAlgorithm):
    """冰山订单算法
    
    将大订单分割成多个小订单，只显示一小部分，避免暴露真实意图。
    """
    
    def __init__(self, algorithm_id: str, order: ExecutionOrder, parameters: AlgorithmParameters):
        """初始化冰山算法
        
        Args:
            algorithm_id: 算法ID
            order: 父订单
            parameters: 算法参数
        """
        super().__init__(algorithm_id, AlgorithmType.ICEBERG, order, parameters)
        
        # 冰山特定参数
        self.slice_size = parameters.iceberg_slice_size or (order.quantity * 0.1)
        self.slice_variance = parameters.iceberg_variance
        self.current_visible_order: Optional[str] = None
    
    async def execute(self):
        """执行冰山算法"""
        try:
            while self._running and self.state.remaining_quantity > 0:
                if self.state.status != AlgorithmStatus.RUNNING:
                    await asyncio.sleep(0.1)
                    continue
                
                # 检查是否需要提交新的可见订单
                if not self.current_visible_order or self._is_current_order_completed():
                    child_order = self.calculate_next_order()
                    if child_order:
                        self.current_visible_order = child_order.order_id
                        self._notify_child_order(child_order)
                        self.logger.info(f"提交冰山子订单: {child_order.order_id} - {child_order.quantity}")
                
                await asyncio.sleep(1.0)
            
            if self.state.remaining_quantity <= 0:
                self.state.status = AlgorithmStatus.COMPLETED
                self.state.end_time = datetime.now()
                self._notify_completion()
                
        except Exception as e:
            self.logger.error(f"冰山算法执行异常: {e}")
            self._notify_error(e)
    
    def _is_current_order_completed(self) -> bool:
        """检查当前可见订单是否完成
        
        Returns:
            bool: 是否完成
        """
        return (self.current_visible_order and 
                self.current_visible_order in self.state.completed_orders)
    
    def calculate_next_order(self) -> Optional[ExecutionOrder]:
        """计算下一个冰山子订单
        
        Returns:
            Optional[ExecutionOrder]: 下一个子订单
        """
        if self.state.remaining_quantity <= 0:
            return None
        
        # 计算切片大小（添加随机变化）
        base_size = self.slice_size
        if self.slice_variance > 0:
            import random
            variance = base_size * self.slice_variance
            slice_size = base_size + random.uniform(-variance, variance)
        else:
            slice_size = base_size
        
        # 确保不超过剩余数量
        order_quantity = min(slice_size, self.state.remaining_quantity)
        
        if order_quantity <= 0:
            return None
        
        # 计算订单价格
        order_price = self._calculate_order_price()
        
        return self._create_child_order(
            quantity=order_quantity,
            price=order_price,
            order_type=OrderType.LIMIT if order_price else OrderType.MARKET
        )
    
    def _calculate_order_price(self) -> Optional[float]:
        """计算订单价格
        
        Returns:
            Optional[float]: 订单价格
        """
        if self.parameters.limit_price:
            return self.parameters.limit_price
        
        if not self.current_price:
            return None
        
        # 冰山订单通常使用限价单，价格略优于市价
        if self.parent_order.side == OrderSide.BUY:
            return self.current_price * (1 - self.parameters.price_tolerance * 0.5)
        else:
            return self.current_price * (1 + self.parameters.price_tolerance * 0.5)
    
    def handle_order_status(self, order_id: str, status: OrderStatus):
        """处理订单状态更新
        
        Args:
            order_id: 订单ID
            status: 订单状态
        """
        super().handle_order_status(order_id, status)
        
        # 如果当前可见订单完成，清除引用
        if order_id == self.current_visible_order and status in [
            OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED
        ]:
            self.current_visible_order = None


class ExecutionAlgorithmManager:
    """执行算法管理器
    
    管理多个执行算法的生命周期。
    """
    
    def __init__(self):
        """初始化算法管理器"""
        self.algorithms: Dict[str, BaseExecutionAlgorithm] = {}
        
        # 回调函数
        self.on_child_order_callbacks: List[Callable] = []
        self.on_trade_callbacks: List[Callable] = []
        self.on_completion_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 日志记录
        self.logger = get_logger("ExecutionAlgorithmManager")
        
        self.logger.info("执行算法管理器初始化完成")
    
    def create_algorithm(self, algorithm_type: AlgorithmType, order: ExecutionOrder,
                        parameters: AlgorithmParameters) -> str:
        """创建执行算法
        
        Args:
            algorithm_type: 算法类型
            order: 父订单
            parameters: 算法参数
            
        Returns:
            str: 算法ID
            
        Raises:
            ExecutionAlgorithmError: 创建失败
        """
        algorithm_id = str(uuid.uuid4())
        
        try:
            # 根据类型创建算法
            if algorithm_type == AlgorithmType.TWAP:
                algorithm = TWAPAlgorithm(algorithm_id, order, parameters)
            elif algorithm_type == AlgorithmType.VWAP:
                algorithm = VWAPAlgorithm(algorithm_id, order, parameters)
            elif algorithm_type == AlgorithmType.ICEBERG:
                algorithm = IcebergAlgorithm(algorithm_id, order, parameters)
            else:
                raise ExecutionAlgorithmError(f"不支持的算法类型: {algorithm_type}")
            
            # 添加回调
            algorithm.add_child_order_callback(self._on_child_order)
            algorithm.add_trade_callback(self._on_trade)
            algorithm.add_completion_callback(self._on_completion)
            algorithm.add_error_callback(self._on_error)
            
            # 保存算法
            with self._lock:
                self.algorithms[algorithm_id] = algorithm
            
            self.logger.info(f"创建执行算法: {algorithm_type.value} - {algorithm_id}")
            return algorithm_id
            
        except Exception as e:
            self.logger.error(f"创建执行算法失败: {e}")
            raise ExecutionAlgorithmError(f"创建算法失败: {e}")
    
    async def start_algorithm(self, algorithm_id: str):
        """启动算法
        
        Args:
            algorithm_id: 算法ID
            
        Raises:
            ExecutionAlgorithmError: 启动失败
        """
        with self._lock:
            if algorithm_id not in self.algorithms:
                raise ExecutionAlgorithmError(f"算法不存在: {algorithm_id}")
            
            algorithm = self.algorithms[algorithm_id]
            await algorithm.start()
    
    async def stop_algorithm(self, algorithm_id: str):
        """停止算法
        
        Args:
            algorithm_id: 算法ID
            
        Raises:
            ExecutionAlgorithmError: 停止失败
        """
        with self._lock:
            if algorithm_id not in self.algorithms:
                raise ExecutionAlgorithmError(f"算法不存在: {algorithm_id}")
            
            algorithm = self.algorithms[algorithm_id]
            await algorithm.stop()
    
    async def pause_algorithm(self, algorithm_id: str):
        """暂停算法
        
        Args:
            algorithm_id: 算法ID
        """
        with self._lock:
            if algorithm_id in self.algorithms:
                await self.algorithms[algorithm_id].pause()
    
    async def resume_algorithm(self, algorithm_id: str):
        """恢复算法
        
        Args:
            algorithm_id: 算法ID
        """
        with self._lock:
            if algorithm_id in self.algorithms:
                await self.algorithms[algorithm_id].resume()
    
    def remove_algorithm(self, algorithm_id: str):
        """移除算法
        
        Args:
            algorithm_id: 算法ID
        """
        with self._lock:
            if algorithm_id in self.algorithms:
                algorithm = self.algorithms[algorithm_id]
                
                # 停止算法
                if algorithm.is_running():
                    asyncio.create_task(algorithm.stop())
                
                del self.algorithms[algorithm_id]
                self.logger.info(f"移除执行算法: {algorithm_id}")
    
    def update_market_data(self, symbol: str, price: float, bid: float = None,
                          ask: float = None, volume: float = None):
        """更新市场数据
        
        Args:
            symbol: 交易品种
            price: 当前价格
            bid: 买价
            ask: 卖价
            volume: 成交量
        """
        with self._lock:
            for algorithm in self.algorithms.values():
                if algorithm.parent_order.symbol == symbol:
                    algorithm.update_market_data(price, bid, ask, volume)
    
    def handle_trade(self, trade: Trade):
        """处理成交回报
        
        Args:
            trade: 成交信息
        """
        with self._lock:
            # 查找对应的算法
            for algorithm in self.algorithms.values():
                if trade.order_id in algorithm.state.child_orders:
                    algorithm.handle_trade(trade)
                    break
    
    def handle_order_status(self, order_id: str, status: OrderStatus):
        """处理订单状态更新
        
        Args:
            order_id: 订单ID
            status: 订单状态
        """
        with self._lock:
            # 查找对应的算法
            for algorithm in self.algorithms.values():
                if order_id in algorithm.state.child_orders:
                    algorithm.handle_order_status(order_id, status)
                    break
    
    def get_algorithm(self, algorithm_id: str) -> Optional[BaseExecutionAlgorithm]:
        """获取算法
        
        Args:
            algorithm_id: 算法ID
            
        Returns:
            Optional[BaseExecutionAlgorithm]: 算法实例
        """
        return self.algorithms.get(algorithm_id)
    
    def get_all_algorithms(self) -> Dict[str, BaseExecutionAlgorithm]:
        """获取所有算法
        
        Returns:
            Dict[str, BaseExecutionAlgorithm]: 算法字典
        """
        return self.algorithms.copy()
    
    def _on_child_order(self, order: ExecutionOrder):
        """子订单回调
        
        Args:
            order: 子订单
        """
        for callback in self.on_child_order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"子订单回调执行失败: {e}")
    
    def _on_trade(self, trade: Trade):
        """成交回调
        
        Args:
            trade: 成交信息
        """
        for callback in self.on_trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                self.logger.error(f"成交回调执行失败: {e}")
    
    def _on_completion(self, algorithm_id: str, state: AlgorithmState):
        """完成回调
        
        Args:
            algorithm_id: 算法ID
            state: 算法状态
        """
        for callback in self.on_completion_callbacks:
            try:
                callback(algorithm_id, state)
            except Exception as e:
                self.logger.error(f"完成回调执行失败: {e}")
    
    def _on_error(self, algorithm_id: str, error: Exception):
        """错误回调
        
        Args:
            algorithm_id: 算法ID
            error: 错误信息
        """
        for callback in self.on_error_callbacks:
            try:
                callback(algorithm_id, error)
            except Exception as e:
                self.logger.error(f"错误回调执行失败: {e}")
    
    def add_child_order_callback(self, callback: Callable):
        """添加子订单回调
        
        Args:
            callback: 回调函数
        """
        self.on_child_order_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable):
        """添加成交回调
        
        Args:
            callback: 回调函数
        """
        self.on_trade_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable):
        """添加完成回调
        
        Args:
            callback: 回调函数
        """
        self.on_completion_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """添加错误回调
        
        Args:
            callback: 回调函数
        """
        self.on_error_callbacks.append(callback)
    
    async def stop_all(self):
        """停止所有算法"""
        tasks = []
        for algorithm in self.algorithms.values():
            if algorithm.is_running():
                tasks.append(algorithm.stop())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info("所有执行算法已停止")
    
    def get_statistics(self) -> Dict:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self._lock:
            algorithm_stats = {}
            total_algorithms = len(self.algorithms)
            running_algorithms = 0
            completed_algorithms = 0
            
            for algorithm_id, algorithm in self.algorithms.items():
                state = algorithm.get_state()
                algorithm_stats[algorithm_id] = {
                    'type': algorithm.algorithm_type.value,
                    'status': state.status.value,
                    'completion_rate': state.calculate_completion_rate(),
                    'total_quantity': state.total_quantity,
                    'executed_quantity': state.executed_quantity,
                    'avg_execution_price': state.avg_execution_price,
                    'total_trades': state.total_trades
                }
                
                if state.status == AlgorithmStatus.RUNNING:
                    running_algorithms += 1
                elif state.status == AlgorithmStatus.COMPLETED:
                    completed_algorithms += 1
            
            return {
                'total_algorithms': total_algorithms,
                'running_algorithms': running_algorithms,
                'completed_algorithms': completed_algorithms,
                'algorithm_stats': algorithm_stats
            }
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 管理器字典
        """
        return {
            'algorithms': {aid: algorithm.to_dict() for aid, algorithm in self.algorithms.items()},
            'statistics': self.get_statistics()
        }
    
    def __str__(self) -> str:
        return f"ExecutionAlgorithmManager(algorithms={len(self.algorithms)})"
    
    def __repr__(self) -> str:
        return self.__str__()