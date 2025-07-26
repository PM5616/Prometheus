"""Base Storage Interface

存储接口基类，定义统一的数据存储规范。

功能特性：
- 统一的存储接口
- 连接管理
- 数据CRUD操作
- 批量操作支持
- 查询优化
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import logging

from src.common.models.market import Ticker, Kline, OrderBook
from src.common.models.trading import Symbol, Order, Trade
from src.common.exceptions.data import (
    DataConnectionError, DataNotFoundError, DataValidationError
)


class StorageType(Enum):
    """存储类型枚举"""
    INFLUXDB = "influxdb"  # 时序数据库
    REDIS = "redis"  # 内存数据库
    POSTGRESQL = "postgresql"  # 关系型数据库
    FILE = "file"  # 文件存储
    MONGODB = "mongodb"  # 文档数据库


class BaseStorage(ABC):
    """存储接口基类
    
    定义统一的数据存储接口，所有存储实现都应继承此类。
    """
    
    def __init__(self, storage_type: str, config: Dict[str, Any]):
        """
        初始化存储基类
        
        Args:
            storage_type: 存储类型
            config: 配置参数
        """
        self.storage_type = storage_type
        self.config = config
        self.is_connected = False
        self.logger = logging.getLogger(f"storage.{storage_type}")
        
        # 连接参数
        self.host = config.get('host', 'localhost')
        self.port = config.get('port')
        self.username = config.get('username')
        self.password = config.get('password')
        self.database = config.get('database')
        
        # 性能统计
        self.stats = {
            'connection_time': None,
            'total_reads': 0,
            'total_writes': 0,
            'total_errors': 0,
            'last_operation_time': None
        }
        
        self.logger.info(f"初始化{storage_type}存储")
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        建立存储连接
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        断开存储连接
        
        Returns:
            bool: 断开是否成功
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            bool: 存储是否健康
        """
        pass
    
    # ==================== 行情数据操作 ====================
    
    @abstractmethod
    async def save_ticker(self, ticker: Ticker) -> bool:
        """
        保存行情数据
        
        Args:
            ticker: 行情数据
            
        Returns:
            bool: 保存是否成功
        """
        pass
    
    @abstractmethod
    async def save_tickers(self, tickers: List[Ticker]) -> bool:
        """
        批量保存行情数据
        
        Args:
            tickers: 行情数据列表
            
        Returns:
            bool: 保存是否成功
        """
        pass
    
    @abstractmethod
    async def get_latest_ticker(self, symbol: str) -> Optional[Ticker]:
        """
        获取最新行情数据
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[Ticker]: 行情数据
        """
        pass
    
    @abstractmethod
    async def get_tickers(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Ticker]:
        """
        获取历史行情数据
        
        Args:
            symbol: 交易对符号
            start_time: 开始时间
            end_time: 结束时间
            limit: 数据条数限制
            
        Returns:
            List[Ticker]: 行情数据列表
        """
        pass
    
    # ==================== K线数据操作 ====================
    
    @abstractmethod
    async def save_kline(self, kline: Kline) -> bool:
        """
        保存K线数据
        
        Args:
            kline: K线数据
            
        Returns:
            bool: 保存是否成功
        """
        pass
    
    @abstractmethod
    async def save_klines(self, klines: List[Kline]) -> bool:
        """
        批量保存K线数据
        
        Args:
            klines: K线数据列表
            
        Returns:
            bool: 保存是否成功
        """
        pass
    
    @abstractmethod
    async def get_latest_kline(self, symbol: str, interval: str) -> Optional[Kline]:
        """
        获取最新K线数据
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            
        Returns:
            Optional[Kline]: K线数据
        """
        pass
    
    @abstractmethod
    async def get_klines(
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
    
    # ==================== 订单簿数据操作 ====================
    
    @abstractmethod
    async def save_order_book(self, order_book: OrderBook) -> bool:
        """
        保存订单簿数据
        
        Args:
            order_book: 订单簿数据
            
        Returns:
            bool: 保存是否成功
        """
        pass
    
    @abstractmethod
    async def get_latest_order_book(self, symbol: str) -> Optional[OrderBook]:
        """
        获取最新订单簿数据
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[OrderBook]: 订单簿数据
        """
        pass
    
    # ==================== 交易数据操作 ====================
    
    @abstractmethod
    async def save_order(self, order: Order) -> bool:
        """
        保存订单数据
        
        Args:
            order: 订单数据
            
        Returns:
            bool: 保存是否成功
        """
        pass
    
    @abstractmethod
    async def save_trade(self, trade: Trade) -> bool:
        """
        保存成交数据
        
        Args:
            trade: 成交数据
            
        Returns:
            bool: 保存是否成功
        """
        pass
    
    @abstractmethod
    async def get_orders(
        self, 
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Order]:
        """
        获取订单数据
        
        Args:
            symbol: 交易对符号（可选）
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
            limit: 数据条数限制（可选）
            
        Returns:
            List[Order]: 订单数据列表
        """
        pass
    
    @abstractmethod
    async def get_trades(
        self, 
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Trade]:
        """
        获取成交数据
        
        Args:
            symbol: 交易对符号（可选）
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
            limit: 数据条数限制（可选）
            
        Returns:
            List[Trade]: 成交数据列表
        """
        pass
    
    # ==================== 通用数据操作 ====================
    
    @abstractmethod
    async def save_data(self, table: str, data: Dict[str, Any]) -> bool:
        """
        保存通用数据
        
        Args:
            table: 表名/集合名
            data: 数据字典
            
        Returns:
            bool: 保存是否成功
        """
        pass
    
    @abstractmethod
    async def get_data(
        self, 
        table: str, 
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取通用数据
        
        Args:
            table: 表名/集合名
            filters: 过滤条件（可选）
            limit: 数据条数限制（可选）
            
        Returns:
            List[Dict[str, Any]]: 数据列表
        """
        pass
    
    @abstractmethod
    async def update_data(
        self, 
        table: str, 
        filters: Dict[str, Any], 
        updates: Dict[str, Any]
    ) -> bool:
        """
        更新数据
        
        Args:
            table: 表名/集合名
            filters: 过滤条件
            updates: 更新数据
            
        Returns:
            bool: 更新是否成功
        """
        pass
    
    @abstractmethod
    async def delete_data(self, table: str, filters: Dict[str, Any]) -> bool:
        """
        删除数据
        
        Args:
            table: 表名/集合名
            filters: 过滤条件
            
        Returns:
            bool: 删除是否成功
        """
        pass
    
    # ==================== 数据管理操作 ====================
    
    @abstractmethod
    async def create_table(self, table: str, schema: Dict[str, Any]) -> bool:
        """
        创建表/集合
        
        Args:
            table: 表名/集合名
            schema: 表结构/模式
            
        Returns:
            bool: 创建是否成功
        """
        pass
    
    @abstractmethod
    async def drop_table(self, table: str) -> bool:
        """
        删除表/集合
        
        Args:
            table: 表名/集合名
            
        Returns:
            bool: 删除是否成功
        """
        pass
    
    @abstractmethod
    async def list_tables(self) -> List[str]:
        """
        列出所有表/集合
        
        Returns:
            List[str]: 表名/集合名列表
        """
        pass
    
    # ==================== 统计和监控 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'storage_type': self.storage_type,
            'is_connected': self.is_connected,
            'host': self.host,
            'port': self.port,
            'database': self.database,
            **self.stats
        }
    
    def _update_stats(self, read: bool = False, write: bool = False, error: bool = False):
        """
        更新统计信息
        
        Args:
            read: 是否为读操作
            write: 是否为写操作
            error: 是否为错误
        """
        if read:
            self.stats['total_reads'] += 1
        if write:
            self.stats['total_writes'] += 1
        if error:
            self.stats['total_errors'] += 1
        
        self.stats['last_operation_time'] = datetime.now()
    
    async def cleanup(self):
        """
        清理资源
        """
        try:
            await self.disconnect()
            self.logger.info(f"{self.storage_type}存储资源清理完成")
        except Exception as e:
            self.logger.error(f"{self.storage_type}存储资源清理失败: {e}")
    
    def __str__(self) -> str:
        return f"{self.storage_type}Storage({self.host}:{self.port}/{self.database})"
    
    def __repr__(self) -> str:
        return self.__str__()