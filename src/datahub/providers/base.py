"""Base Data Provider

数据提供商基类，定义统一的数据接口规范。
所有数据提供商都应继承此基类并实现相应方法。
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator, Any, Union
from datetime import datetime, timedelta
from enum import Enum

from src.common.models.market import Ticker, Kline, OrderBook, MarketData
from src.common.models.trading import Symbol
from src.common.logging import get_logger
from src.common.exceptions.data import (
    DataConnectionError, DataNotFoundError, DataValidationError
)


class DataType(Enum):
    """数据类型枚举"""
    TICKER = "ticker"  # 行情数据
    KLINE = "kline"    # K线数据
    ORDER_BOOK = "order_book"  # 订单簿数据
    TRADE = "trade"    # 成交数据
    FUNDING_RATE = "funding_rate"  # 资金费率
    OPEN_INTEREST = "open_interest"  # 持仓量


class SubscriptionStatus(Enum):
    """订阅状态枚举"""
    PENDING = "pending"      # 待订阅
    ACTIVE = "active"        # 活跃订阅
    PAUSED = "paused"        # 暂停订阅
    STOPPED = "stopped"      # 停止订阅
    ERROR = "error"          # 订阅错误


class BaseDataProvider(ABC):
    """数据提供商基类
    
    定义了数据提供商的标准接口，包括：
    - 连接管理
    - 数据订阅
    - 历史数据获取
    - 错误处理
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化数据提供商
        
        Args:
            name: 提供商名称
            config: 配置参数
        """
        self.name = name
        self.config = config
        self.logger = get_logger(f"datahub.providers.{name}")
        
        # 连接状态
        self.is_connected = False
        self.last_heartbeat = None
        
        # 订阅管理
        self.subscriptions: Dict[str, SubscriptionStatus] = {}
        self.subscription_callbacks: Dict[str, List[callable]] = {}
        
        # 统计信息
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'errors_count': 0,
            'last_message_time': None,
            'connection_time': None
        }
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        建立连接
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        断开连接
        
        Returns:
            bool: 断开是否成功
        """
        pass
    
    @abstractmethod
    async def subscribe_ticker(self, symbol: str, callback: callable) -> bool:
        """
        订阅行情数据
        
        Args:
            symbol: 交易对符号
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        pass
    
    @abstractmethod
    async def subscribe_kline(self, symbol: str, interval: str, callback: callable) -> bool:
        """
        订阅K线数据
        
        Args:
            symbol: 交易对符号
            interval: K线间隔（1m, 5m, 1h等）
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        pass
    
    @abstractmethod
    async def subscribe_order_book(self, symbol: str, depth: int, callback: callable) -> bool:
        """
        订阅订单簿数据
        
        Args:
            symbol: 交易对符号
            depth: 订单簿深度
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_key: str) -> bool:
        """
        取消订阅
        
        Args:
            subscription_key: 订阅键
            
        Returns:
            bool: 取消订阅是否成功
        """
        pass
    
    @abstractmethod
    async def get_historical_klines(
        self, 
        symbol: str, 
        interval: str, 
        start_time: datetime, 
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Kline]:
        """
        获取历史K线数据
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            start_time: 开始时间
            end_time: 结束时间
            limit: 数据条数限制
            
        Returns:
            List[Kline]: K线数据列表
        """
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """
        获取当前行情数据
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[Ticker]: 行情数据
        """
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 20) -> Optional[OrderBook]:
        """
        获取当前订单簿数据
        
        Args:
            symbol: 交易对符号
            depth: 订单簿深度
            
        Returns:
            Optional[OrderBook]: 订单簿数据
        """
        pass
    
    @abstractmethod
    async def get_symbols(self) -> List[Symbol]:
        """
        获取支持的交易对列表
        
        Returns:
            List[Symbol]: 交易对列表
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        return {
            'provider': self.name,
            'connected': self.is_connected,
            'last_heartbeat': self.last_heartbeat,
            'active_subscriptions': len([s for s in self.subscriptions.values() 
                                       if s == SubscriptionStatus.ACTIVE]),
            'stats': self.stats.copy()
        }
    
    def add_subscription_callback(self, subscription_key: str, callback: callable):
        """
        添加订阅回调函数
        
        Args:
            subscription_key: 订阅键
            callback: 回调函数
        """
        if subscription_key not in self.subscription_callbacks:
            self.subscription_callbacks[subscription_key] = []
        self.subscription_callbacks[subscription_key].append(callback)
    
    def remove_subscription_callback(self, subscription_key: str, callback: callable):
        """
        移除订阅回调函数
        
        Args:
            subscription_key: 订阅键
            callback: 回调函数
        """
        if subscription_key in self.subscription_callbacks:
            try:
                self.subscription_callbacks[subscription_key].remove(callback)
                if not self.subscription_callbacks[subscription_key]:
                    del self.subscription_callbacks[subscription_key]
            except ValueError:
                pass
    
    async def _notify_callbacks(self, subscription_key: str, data: Any):
        """
        通知订阅回调函数
        
        Args:
            subscription_key: 订阅键
            data: 数据
        """
        if subscription_key in self.subscription_callbacks:
            for callback in self.subscription_callbacks[subscription_key]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"回调函数执行失败: {e}", exc_info=True)
    
    def _update_stats(self, message_received: bool = False, message_processed: bool = False, error: bool = False):
        """
        更新统计信息
        
        Args:
            message_received: 是否接收到消息
            message_processed: 是否处理了消息
            error: 是否发生错误
        """
        now = datetime.now()
        
        if message_received:
            self.stats['messages_received'] += 1
            self.stats['last_message_time'] = now
        
        if message_processed:
            self.stats['messages_processed'] += 1
        
        if error:
            self.stats['errors_count'] += 1
    
    def _generate_subscription_key(self, data_type: DataType, symbol: str, **kwargs) -> str:
        """
        生成订阅键
        
        Args:
            data_type: 数据类型
            symbol: 交易对符号
            **kwargs: 其他参数
            
        Returns:
            str: 订阅键
        """
        key_parts = [data_type.value, symbol.upper()]
        
        # 添加额外参数
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}={value}")
        
        return ":".join(key_parts)
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.disconnect()
        return False