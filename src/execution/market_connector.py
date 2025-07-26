"""Market Connector Module

市场连接器模块，负责与不同交易所和市场的连接和通信。

主要功能：
- 市场连接管理
- 订单提交和管理
- 市场数据接收
- 连接状态监控
- 错误处理和重连
"""

import asyncio
import threading
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import uuid
import json
import time

from src.common.logging import get_logger
from src.common.exceptions.execution import MarketConnectorError, ConnectionError
from .engine import ExecutionOrder, Trade, OrderStatus, OrderType, OrderSide


class ConnectionStatus(Enum):
    """连接状态枚举"""
    DISCONNECTED = "disconnected"  # 断开连接
    CONNECTING = "connecting"      # 连接中
    CONNECTED = "connected"        # 已连接
    AUTHENTICATED = "authenticated"  # 已认证
    ERROR = "error"                # 错误状态
    RECONNECTING = "reconnecting"  # 重连中


class MessageType(Enum):
    """消息类型枚举"""
    ORDER_SUBMIT = "order_submit"      # 订单提交
    ORDER_CANCEL = "order_cancel"      # 订单取消
    ORDER_MODIFY = "order_modify"      # 订单修改
    ORDER_STATUS = "order_status"      # 订单状态
    TRADE_REPORT = "trade_report"      # 成交回报
    MARKET_DATA = "market_data"        # 市场数据
    HEARTBEAT = "heartbeat"            # 心跳
    ERROR = "error"                    # 错误
    SYSTEM = "system"                  # 系统消息


@dataclass
class ConnectionConfig:
    """连接配置"""
    host: str
    port: int
    username: str = ""
    password: str = ""
    api_key: str = ""
    secret_key: str = ""
    
    # 连接参数
    timeout: float = 30.0
    heartbeat_interval: float = 30.0
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 10
    
    # SSL配置
    use_ssl: bool = True
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    
    # 其他配置
    encoding: str = "utf-8"
    buffer_size: int = 8192
    compression: bool = False
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 配置字典
        """
        return {
            'host': self.host,
            'port': self.port,
            'username': self.username,
            'password': '***' if self.password else '',
            'api_key': self.api_key[:8] + '***' if self.api_key else '',
            'secret_key': '***' if self.secret_key else '',
            'timeout': self.timeout,
            'heartbeat_interval': self.heartbeat_interval,
            'reconnect_interval': self.reconnect_interval,
            'max_reconnect_attempts': self.max_reconnect_attempts,
            'use_ssl': self.use_ssl,
            'ssl_cert_path': self.ssl_cert_path,
            'ssl_key_path': self.ssl_key_path,
            'encoding': self.encoding,
            'buffer_size': self.buffer_size,
            'compression': self.compression
        }


@dataclass
class MarketMessage:
    """市场消息"""
    message_id: str
    message_type: MessageType
    timestamp: datetime
    data: Dict
    source: str = ""
    target: str = ""
    correlation_id: str = ""
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 消息字典
        """
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'source': self.source,
            'target': self.target,
            'correlation_id': self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketMessage':
        """从字典创建消息
        
        Args:
            data: 消息字典
            
        Returns:
            MarketMessage: 市场消息
        """
        return cls(
            message_id=data['message_id'],
            message_type=MessageType(data['message_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            data=data['data'],
            source=data.get('source', ''),
            target=data.get('target', ''),
            correlation_id=data.get('correlation_id', '')
        )


@dataclass
class ConnectionMetrics:
    """连接指标"""
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    
    connection_count: int = 0
    disconnection_count: int = 0
    reconnection_count: int = 0
    error_count: int = 0
    
    avg_latency: float = 0.0
    max_latency: float = 0.0
    min_latency: float = float('inf')
    
    last_message_time: Optional[datetime] = None
    last_heartbeat_time: Optional[datetime] = None
    connection_start_time: Optional[datetime] = None
    total_uptime: float = 0.0
    
    def update_latency(self, latency: float):
        """更新延迟统计
        
        Args:
            latency: 延迟时间（秒）
        """
        self.avg_latency = (self.avg_latency * self.total_messages_received + latency) / (self.total_messages_received + 1)
        self.max_latency = max(self.max_latency, latency)
        self.min_latency = min(self.min_latency, latency)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 指标字典
        """
        return {
            'total_messages_sent': self.total_messages_sent,
            'total_messages_received': self.total_messages_received,
            'total_bytes_sent': self.total_bytes_sent,
            'total_bytes_received': self.total_bytes_received,
            'connection_count': self.connection_count,
            'disconnection_count': self.disconnection_count,
            'reconnection_count': self.reconnection_count,
            'error_count': self.error_count,
            'avg_latency': self.avg_latency,
            'max_latency': self.max_latency,
            'min_latency': self.min_latency if self.min_latency != float('inf') else 0.0,
            'last_message_time': self.last_message_time.isoformat() if self.last_message_time else None,
            'last_heartbeat_time': self.last_heartbeat_time.isoformat() if self.last_heartbeat_time else None,
            'connection_start_time': self.connection_start_time.isoformat() if self.connection_start_time else None,
            'total_uptime': self.total_uptime
        }


class BaseMarketConnector(ABC):
    """市场连接器基类
    
    定义市场连接器的基本接口。
    """
    
    def __init__(self, connector_id: str, config: ConnectionConfig):
        """初始化连接器
        
        Args:
            connector_id: 连接器ID
            config: 连接配置
        """
        self.connector_id = connector_id
        self.config = config
        
        # 连接状态
        self.status = ConnectionStatus.DISCONNECTED
        self.last_error: Optional[Exception] = None
        
        # 消息队列
        self.outbound_queue: asyncio.Queue = asyncio.Queue()
        self.inbound_queue: asyncio.Queue = asyncio.Queue()
        
        # 回调函数
        self.on_connected_callbacks: List[Callable] = []
        self.on_disconnected_callbacks: List[Callable] = []
        self.on_message_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
        
        # 指标统计
        self.metrics = ConnectionMetrics()
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 日志记录
        self.logger = get_logger(f"MarketConnector-{connector_id}")
        
        # 运行控制
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        self.logger.info(f"市场连接器初始化: {connector_id}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接到市场
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """认证
        
        Returns:
            bool: 认证是否成功
        """
        pass
    
    @abstractmethod
    async def send_message(self, message: MarketMessage) -> bool:
        """发送消息
        
        Args:
            message: 市场消息
            
        Returns:
            bool: 发送是否成功
        """
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[MarketMessage]:
        """接收消息
        
        Returns:
            Optional[MarketMessage]: 接收到的消息
        """
        pass
    
    async def submit_order(self, order: ExecutionOrder) -> bool:
        """提交订单
        
        Args:
            order: 执行订单
            
        Returns:
            bool: 提交是否成功
        """
        message = MarketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ORDER_SUBMIT,
            timestamp=datetime.now(),
            data=self._order_to_dict(order),
            correlation_id=order.order_id
        )
        
        return await self.send_message(message)
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            bool: 取消是否成功
        """
        message = MarketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ORDER_CANCEL,
            timestamp=datetime.now(),
            data={'order_id': order_id},
            correlation_id=order_id
        )
        
        return await self.send_message(message)
    
    async def modify_order(self, order_id: str, new_quantity: float = None,
                          new_price: float = None) -> bool:
        """修改订单
        
        Args:
            order_id: 订单ID
            new_quantity: 新数量
            new_price: 新价格
            
        Returns:
            bool: 修改是否成功
        """
        data = {'order_id': order_id}
        if new_quantity is not None:
            data['new_quantity'] = new_quantity
        if new_price is not None:
            data['new_price'] = new_price
        
        message = MarketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ORDER_MODIFY,
            timestamp=datetime.now(),
            data=data,
            correlation_id=order_id
        )
        
        return await self.send_message(message)
    
    async def start(self):
        """启动连接器"""
        if self._running:
            return
        
        self._running = True
        
        # 启动异步任务
        self._tasks = [
            asyncio.create_task(self._connection_manager()),
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._heartbeat_sender())
        ]
        
        self.logger.info(f"连接器启动: {self.connector_id}")
    
    async def stop(self):
        """停止连接器"""
        if not self._running:
            return
        
        self._running = False
        
        # 断开连接
        await self.disconnect()
        
        # 取消任务
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # 等待任务完成
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        
        self.logger.info(f"连接器停止: {self.connector_id}")
    
    async def _connection_manager(self):
        """连接管理器"""
        reconnect_attempts = 0
        
        while self._running:
            try:
                if self.status == ConnectionStatus.DISCONNECTED:
                    self.logger.info(f"尝试连接: {self.connector_id}")
                    
                    if await self.connect():
                        reconnect_attempts = 0
                        
                        # 认证
                        if await self.authenticate():
                            self.status = ConnectionStatus.AUTHENTICATED
                            self._notify_connected()
                        else:
                            self.logger.error(f"认证失败: {self.connector_id}")
                            await self.disconnect()
                    else:
                        reconnect_attempts += 1
                        if reconnect_attempts >= self.config.max_reconnect_attempts:
                            self.logger.error(f"达到最大重连次数: {self.connector_id}")
                            break
                        
                        await asyncio.sleep(self.config.reconnect_interval)
                
                elif self.status in [ConnectionStatus.CONNECTED, ConnectionStatus.AUTHENTICATED]:
                    # 检查连接状态
                    await asyncio.sleep(1.0)
                
                else:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"连接管理器异常: {self.connector_id} - {e}")
                self.last_error = e
                self.status = ConnectionStatus.ERROR
                self._notify_error(e)
                await asyncio.sleep(self.config.reconnect_interval)
    
    async def _message_processor(self):
        """消息处理器"""
        while self._running:
            try:
                # 处理入站消息
                message = await self.receive_message()
                if message:
                    self.metrics.total_messages_received += 1
                    self.metrics.last_message_time = datetime.now()
                    self._notify_message(message)
                
                # 处理出站消息
                try:
                    message = self.outbound_queue.get_nowait()
                    if await self.send_message(message):
                        self.metrics.total_messages_sent += 1
                except asyncio.QueueEmpty:
                    pass
                
                await asyncio.sleep(0.001)  # 避免CPU占用过高
                
            except Exception as e:
                self.logger.error(f"消息处理器异常: {self.connector_id} - {e}")
                self.last_error = e
                self._notify_error(e)
                await asyncio.sleep(0.1)
    
    async def _heartbeat_sender(self):
        """心跳发送器"""
        while self._running:
            try:
                if self.status == ConnectionStatus.AUTHENTICATED:
                    heartbeat_message = MarketMessage(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.HEARTBEAT,
                        timestamp=datetime.now(),
                        data={'timestamp': datetime.now().isoformat()}
                    )
                    
                    if await self.send_message(heartbeat_message):
                        self.metrics.last_heartbeat_time = datetime.now()
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"心跳发送器异常: {self.connector_id} - {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    def _order_to_dict(self, order: ExecutionOrder) -> Dict:
        """将订单转换为字典
        
        Args:
            order: 执行订单
            
        Returns:
            Dict: 订单字典
        """
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': order.quantity,
            'price': order.price,
            'strategy_id': order.strategy_id,
            'timestamp': order.timestamp.isoformat()
        }
    
    def _notify_connected(self):
        """通知连接成功"""
        self.metrics.connection_count += 1
        self.metrics.connection_start_time = datetime.now()
        
        for callback in self.on_connected_callbacks:
            try:
                callback(self.connector_id)
            except Exception as e:
                self.logger.error(f"连接回调执行失败: {e}")
    
    def _notify_disconnected(self):
        """通知连接断开"""
        self.metrics.disconnection_count += 1
        
        if self.metrics.connection_start_time:
            uptime = (datetime.now() - self.metrics.connection_start_time).total_seconds()
            self.metrics.total_uptime += uptime
        
        for callback in self.on_disconnected_callbacks:
            try:
                callback(self.connector_id)
            except Exception as e:
                self.logger.error(f"断开连接回调执行失败: {e}")
    
    def _notify_message(self, message: MarketMessage):
        """通知消息接收
        
        Args:
            message: 市场消息
        """
        for callback in self.on_message_callbacks:
            try:
                callback(self.connector_id, message)
            except Exception as e:
                self.logger.error(f"消息回调执行失败: {e}")
    
    def _notify_error(self, error: Exception):
        """通知错误
        
        Args:
            error: 错误信息
        """
        self.metrics.error_count += 1
        
        for callback in self.on_error_callbacks:
            try:
                callback(self.connector_id, error)
            except Exception as e:
                self.logger.error(f"错误回调执行失败: {e}")
    
    def add_connected_callback(self, callback: Callable):
        """添加连接回调
        
        Args:
            callback: 回调函数
        """
        self.on_connected_callbacks.append(callback)
    
    def add_disconnected_callback(self, callback: Callable):
        """添加断开连接回调
        
        Args:
            callback: 回调函数
        """
        self.on_disconnected_callbacks.append(callback)
    
    def add_message_callback(self, callback: Callable):
        """添加消息回调
        
        Args:
            callback: 回调函数
        """
        self.on_message_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """添加错误回调
        
        Args:
            callback: 回调函数
        """
        self.on_error_callbacks.append(callback)
    
    def get_status(self) -> ConnectionStatus:
        """获取连接状态
        
        Returns:
            ConnectionStatus: 连接状态
        """
        return self.status
    
    def get_metrics(self) -> ConnectionMetrics:
        """获取连接指标
        
        Returns:
            ConnectionMetrics: 连接指标
        """
        return self.metrics
    
    def is_connected(self) -> bool:
        """检查是否已连接
        
        Returns:
            bool: 是否已连接
        """
        return self.status in [ConnectionStatus.CONNECTED, ConnectionStatus.AUTHENTICATED]
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 连接器字典
        """
        return {
            'connector_id': self.connector_id,
            'status': self.status.value,
            'config': self.config.to_dict(),
            'metrics': self.metrics.to_dict(),
            'last_error': str(self.last_error) if self.last_error else None,
            'running': self._running
        }
    
    def __str__(self) -> str:
        return f"MarketConnector(id={self.connector_id}, status={self.status.value})"
    
    def __repr__(self) -> str:
        return self.__str__()


class MockMarketConnector(BaseMarketConnector):
    """模拟市场连接器
    
    用于测试和开发。
    """
    
    def __init__(self, connector_id: str, config: ConnectionConfig = None):
        """初始化模拟连接器
        
        Args:
            connector_id: 连接器ID
            config: 连接配置
        """
        if config is None:
            config = ConnectionConfig(host="localhost", port=8080)
        
        super().__init__(connector_id, config)
        
        # 模拟数据
        self.mock_orders: Dict[str, Dict] = {}
        self.mock_trades: List[Dict] = []
        
        # 模拟延迟
        self.mock_latency = 0.01  # 10ms
        
        self.logger.info(f"模拟市场连接器初始化: {connector_id}")
    
    async def connect(self) -> bool:
        """模拟连接
        
        Returns:
            bool: 连接是否成功
        """
        await asyncio.sleep(self.mock_latency)
        
        self.status = ConnectionStatus.CONNECTED
        self.logger.info(f"模拟连接成功: {self.connector_id}")
        return True
    
    async def disconnect(self):
        """模拟断开连接"""
        await asyncio.sleep(self.mock_latency)
        
        self.status = ConnectionStatus.DISCONNECTED
        self._notify_disconnected()
        self.logger.info(f"模拟断开连接: {self.connector_id}")
    
    async def authenticate(self) -> bool:
        """模拟认证
        
        Returns:
            bool: 认证是否成功
        """
        await asyncio.sleep(self.mock_latency)
        
        self.logger.info(f"模拟认证成功: {self.connector_id}")
        return True
    
    async def send_message(self, message: MarketMessage) -> bool:
        """模拟发送消息
        
        Args:
            message: 市场消息
            
        Returns:
            bool: 发送是否成功
        """
        await asyncio.sleep(self.mock_latency)
        
        # 处理不同类型的消息
        if message.message_type == MessageType.ORDER_SUBMIT:
            await self._handle_order_submit(message)
        elif message.message_type == MessageType.ORDER_CANCEL:
            await self._handle_order_cancel(message)
        elif message.message_type == MessageType.ORDER_MODIFY:
            await self._handle_order_modify(message)
        
        self.metrics.total_messages_sent += 1
        self.metrics.total_bytes_sent += len(json.dumps(message.to_dict()))
        
        return True
    
    async def receive_message(self) -> Optional[MarketMessage]:
        """模拟接收消息
        
        Returns:
            Optional[MarketMessage]: 接收到的消息
        """
        try:
            message = await asyncio.wait_for(self.inbound_queue.get(), timeout=0.1)
            
            self.metrics.total_messages_received += 1
            self.metrics.total_bytes_received += len(json.dumps(message.to_dict()))
            
            return message
        except asyncio.TimeoutError:
            return None
    
    async def _handle_order_submit(self, message: MarketMessage):
        """处理订单提交
        
        Args:
            message: 订单消息
        """
        order_data = message.data
        order_id = order_data['order_id']
        
        # 保存订单
        self.mock_orders[order_id] = {
            **order_data,
            'status': OrderStatus.PENDING.value,
            'filled_quantity': 0.0,
            'remaining_quantity': order_data['quantity'],
            'submit_time': datetime.now().isoformat()
        }
        
        # 发送订单确认
        confirm_message = MarketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ORDER_STATUS,
            timestamp=datetime.now(),
            data={
                'order_id': order_id,
                'status': OrderStatus.PENDING.value,
                'message': '订单已接收'
            },
            correlation_id=order_id
        )
        
        await self.inbound_queue.put(confirm_message)
        
        # 模拟订单执行
        asyncio.create_task(self._simulate_order_execution(order_id))
    
    async def _handle_order_cancel(self, message: MarketMessage):
        """处理订单取消
        
        Args:
            message: 取消消息
        """
        order_id = message.data['order_id']
        
        if order_id in self.mock_orders:
            order = self.mock_orders[order_id]
            
            if order['status'] in [OrderStatus.PENDING.value, OrderStatus.PARTIALLY_FILLED.value]:
                order['status'] = OrderStatus.CANCELLED.value
                
                # 发送取消确认
                cancel_message = MarketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.ORDER_STATUS,
                    timestamp=datetime.now(),
                    data={
                        'order_id': order_id,
                        'status': OrderStatus.CANCELLED.value,
                        'message': '订单已取消'
                    },
                    correlation_id=order_id
                )
                
                await self.inbound_queue.put(cancel_message)
    
    async def _handle_order_modify(self, message: MarketMessage):
        """处理订单修改
        
        Args:
            message: 修改消息
        """
        data = message.data
        order_id = data['order_id']
        
        if order_id in self.mock_orders:
            order = self.mock_orders[order_id]
            
            if order['status'] in [OrderStatus.PENDING.value, OrderStatus.PARTIALLY_FILLED.value]:
                # 更新订单
                if 'new_quantity' in data:
                    order['quantity'] = data['new_quantity']
                    order['remaining_quantity'] = data['new_quantity'] - order['filled_quantity']
                
                if 'new_price' in data:
                    order['price'] = data['new_price']
                
                # 发送修改确认
                modify_message = MarketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.ORDER_STATUS,
                    timestamp=datetime.now(),
                    data={
                        'order_id': order_id,
                        'status': order['status'],
                        'message': '订单已修改'
                    },
                    correlation_id=order_id
                )
                
                await self.inbound_queue.put(modify_message)
    
    async def _simulate_order_execution(self, order_id: str):
        """模拟订单执行
        
        Args:
            order_id: 订单ID
        """
        await asyncio.sleep(1.0)  # 模拟执行延迟
        
        if order_id not in self.mock_orders:
            return
        
        order = self.mock_orders[order_id]
        
        if order['status'] != OrderStatus.PENDING.value:
            return
        
        # 模拟部分成交
        fill_quantity = order['remaining_quantity'] * 0.5
        fill_price = order['price'] if order['price'] else 100.0  # 模拟价格
        
        order['filled_quantity'] += fill_quantity
        order['remaining_quantity'] -= fill_quantity
        
        # 创建成交记录
        trade = {
            'trade_id': str(uuid.uuid4()),
            'order_id': order_id,
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': fill_quantity,
            'price': fill_price,
            'timestamp': datetime.now().isoformat()
        }
        
        self.mock_trades.append(trade)
        
        # 发送成交回报
        trade_message = MarketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TRADE_REPORT,
            timestamp=datetime.now(),
            data=trade,
            correlation_id=order_id
        )
        
        await self.inbound_queue.put(trade_message)
        
        # 更新订单状态
        if order['remaining_quantity'] <= 0:
            order['status'] = OrderStatus.FILLED.value
        else:
            order['status'] = OrderStatus.PARTIALLY_FILLED.value
        
        # 发送状态更新
        status_message = MarketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ORDER_STATUS,
            timestamp=datetime.now(),
            data={
                'order_id': order_id,
                'status': order['status'],
                'filled_quantity': order['filled_quantity'],
                'remaining_quantity': order['remaining_quantity']
            },
            correlation_id=order_id
        )
        
        await self.inbound_queue.put(status_message)
        
        # 如果还有剩余数量，继续执行
        if order['remaining_quantity'] > 0:
            asyncio.create_task(self._simulate_order_execution(order_id))
    
    def get_mock_orders(self) -> Dict[str, Dict]:
        """获取模拟订单
        
        Returns:
            Dict[str, Dict]: 模拟订单字典
        """
        return self.mock_orders.copy()
    
    def get_mock_trades(self) -> List[Dict]:
        """获取模拟成交
        
        Returns:
            List[Dict]: 模拟成交列表
        """
        return self.mock_trades.copy()


class MarketConnectorManager:
    """市场连接器管理器
    
    管理多个市场连接器。
    """
    
    def __init__(self):
        """初始化连接器管理器"""
        self.connectors: Dict[str, BaseMarketConnector] = {}
        
        # 回调函数
        self.on_connector_connected_callbacks: List[Callable] = []
        self.on_connector_disconnected_callbacks: List[Callable] = []
        self.on_message_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 日志记录
        self.logger = get_logger("MarketConnectorManager")
        
        self.logger.info("市场连接器管理器初始化完成")
    
    def add_connector(self, connector: BaseMarketConnector):
        """添加连接器
        
        Args:
            connector: 市场连接器
        """
        with self._lock:
            self.connectors[connector.connector_id] = connector
            
            # 添加回调
            connector.add_connected_callback(self._on_connector_connected)
            connector.add_disconnected_callback(self._on_connector_disconnected)
            connector.add_message_callback(self._on_message)
            connector.add_error_callback(self._on_error)
            
            self.logger.info(f"添加连接器: {connector.connector_id}")
    
    def remove_connector(self, connector_id: str):
        """移除连接器
        
        Args:
            connector_id: 连接器ID
        """
        with self._lock:
            if connector_id in self.connectors:
                connector = self.connectors[connector_id]
                
                # 停止连接器
                asyncio.create_task(connector.stop())
                
                del self.connectors[connector_id]
                
                self.logger.info(f"移除连接器: {connector_id}")
    
    def get_connector(self, connector_id: str) -> Optional[BaseMarketConnector]:
        """获取连接器
        
        Args:
            connector_id: 连接器ID
            
        Returns:
            Optional[BaseMarketConnector]: 连接器
        """
        return self.connectors.get(connector_id)
    
    def get_all_connectors(self) -> Dict[str, BaseMarketConnector]:
        """获取所有连接器
        
        Returns:
            Dict[str, BaseMarketConnector]: 连接器字典
        """
        return self.connectors.copy()
    
    async def start_all(self):
        """启动所有连接器"""
        tasks = []
        for connector in self.connectors.values():
            tasks.append(connector.start())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info("所有连接器已启动")
    
    async def stop_all(self):
        """停止所有连接器"""
        tasks = []
        for connector in self.connectors.values():
            tasks.append(connector.stop())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info("所有连接器已停止")
    
    def _on_connector_connected(self, connector_id: str):
        """连接器连接回调
        
        Args:
            connector_id: 连接器ID
        """
        for callback in self.on_connector_connected_callbacks:
            try:
                callback(connector_id)
            except Exception as e:
                self.logger.error(f"连接器连接回调执行失败: {e}")
    
    def _on_connector_disconnected(self, connector_id: str):
        """连接器断开回调
        
        Args:
            connector_id: 连接器ID
        """
        for callback in self.on_connector_disconnected_callbacks:
            try:
                callback(connector_id)
            except Exception as e:
                self.logger.error(f"连接器断开回调执行失败: {e}")
    
    def _on_message(self, connector_id: str, message: MarketMessage):
        """消息回调
        
        Args:
            connector_id: 连接器ID
            message: 市场消息
        """
        for callback in self.on_message_callbacks:
            try:
                callback(connector_id, message)
            except Exception as e:
                self.logger.error(f"消息回调执行失败: {e}")
    
    def _on_error(self, connector_id: str, error: Exception):
        """错误回调
        
        Args:
            connector_id: 连接器ID
            error: 错误信息
        """
        for callback in self.on_error_callbacks:
            try:
                callback(connector_id, error)
            except Exception as e:
                self.logger.error(f"错误回调执行失败: {e}")
    
    def add_connector_connected_callback(self, callback: Callable):
        """添加连接器连接回调
        
        Args:
            callback: 回调函数
        """
        self.on_connector_connected_callbacks.append(callback)
    
    def add_connector_disconnected_callback(self, callback: Callable):
        """添加连接器断开回调
        
        Args:
            callback: 回调函数
        """
        self.on_connector_disconnected_callbacks.append(callback)
    
    def add_message_callback(self, callback: Callable):
        """添加消息回调
        
        Args:
            callback: 回调函数
        """
        self.on_message_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """添加错误回调
        
        Args:
            callback: 回调函数
        """
        self.on_error_callbacks.append(callback)
    
    def get_statistics(self) -> Dict:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self._lock:
            connector_stats = {}
            total_messages_sent = 0
            total_messages_received = 0
            connected_count = 0
            
            for connector_id, connector in self.connectors.items():
                metrics = connector.get_metrics()
                connector_stats[connector_id] = {
                    'status': connector.get_status().value,
                    'metrics': metrics.to_dict()
                }
                
                total_messages_sent += metrics.total_messages_sent
                total_messages_received += metrics.total_messages_received
                
                if connector.is_connected():
                    connected_count += 1
            
            return {
                'total_connectors': len(self.connectors),
                'connected_connectors': connected_count,
                'total_messages_sent': total_messages_sent,
                'total_messages_received': total_messages_received,
                'connector_stats': connector_stats
            }
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 管理器字典
        """
        return {
            'connectors': {cid: connector.to_dict() for cid, connector in self.connectors.items()},
            'statistics': self.get_statistics()
        }
    
    def __str__(self) -> str:
        return f"MarketConnectorManager(connectors={len(self.connectors)})"
    
    def __repr__(self) -> str:
        return self.__str__()