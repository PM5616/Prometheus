"""MongoDB Storage Implementation

MongoDB存储实现，用于文档型数据存储。

功能特性：
- 文档型数据存储
- 灵活的数据结构
- 高性能查询
- 水平扩展支持
- 聚合分析功能
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging

try:
    import motor.motor_asyncio
    from pymongo import ASCENDING, DESCENDING
    from pymongo.errors import DuplicateKeyError, ConnectionFailure
except ImportError:
    motor = None
    ASCENDING = DESCENDING = None
    DuplicateKeyError = ConnectionFailure = None

from .base import BaseStorage, StorageType
from src.common.models.market import Ticker, Kline, OrderBook
from src.common.models.trading import Symbol, Order, Trade
from src.common.exceptions.data import (
    DataConnectionError, DataNotFoundError, DataValidationError
)


class MongoDBStorage(BaseStorage):
    """MongoDB存储实现
    
    用于文档型数据的灵活存储。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化MongoDB存储
        
        Args:
            config: MongoDB配置
                - host: 数据库主机地址
                - port: 数据库端口
                - username: 用户名
                - password: 密码
                - database: 数据库名
                - auth_source: 认证数据库
                - replica_set: 副本集名称
                - ssl: 是否使用SSL
                - max_pool_size: 连接池最大连接数
                - min_pool_size: 连接池最小连接数
                - max_idle_time_ms: 最大空闲时间
        """
        super().__init__(StorageType.MONGODB.value, config)
        
        if not motor:
            raise ImportError("motor package is required for MongoDBStorage")
        
        # MongoDB配置
        self.auth_source = config.get('auth_source', 'admin')
        self.replica_set = config.get('replica_set')
        self.ssl = config.get('ssl', False)
        self.max_pool_size = config.get('max_pool_size', 100)
        self.min_pool_size = config.get('min_pool_size', 0)
        self.max_idle_time_ms = config.get('max_idle_time_ms', 30000)
        
        # 连接对象
        self.client = None
        self.db = None
        
        # 集合名称配置
        self.collection_prefix = config.get('collection_prefix', 'prometheus_')
        
        self.logger.info("MongoDB存储初始化完成")
    
    def _get_collection_name(self, collection_type: str) -> str:
        """
        获取集合名称
        
        Args:
            collection_type: 集合类型
            
        Returns:
            str: 完整集合名称
        """
        return f"{self.collection_prefix}{collection_type}"
    
    def _build_connection_string(self) -> str:
        """
        构建MongoDB连接字符串
        
        Returns:
            str: 连接字符串
        """
        # 基础连接字符串
        if self.username and self.password:
            auth_part = f"{self.username}:{self.password}@"
        else:
            auth_part = ""
        
        host_part = f"{self.host}:{self.port or 27017}"
        
        # 构建参数
        params = []
        
        if self.auth_source:
            params.append(f"authSource={self.auth_source}")
        
        if self.replica_set:
            params.append(f"replicaSet={self.replica_set}")
        
        if self.ssl:
            params.append("ssl=true")
        
        params.append(f"maxPoolSize={self.max_pool_size}")
        params.append(f"minPoolSize={self.min_pool_size}")
        params.append(f"maxIdleTimeMS={self.max_idle_time_ms}")
        
        param_string = "&".join(params) if params else ""
        
        return f"mongodb://{auth_part}{host_part}/{self.database}?{param_string}"
    
    async def connect(self) -> bool:
        """
        建立MongoDB连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            start_time = datetime.now()
            
            # 构建连接字符串
            connection_string = self._build_connection_string()
            
            # 创建客户端
            self.client = motor.motor_asyncio.AsyncIOMotorClient(connection_string)
            
            # 获取数据库
            self.db = self.client[self.database]
            
            # 测试连接
            await self.client.admin.command('ping')
            
            # 创建索引
            await self._create_indexes()
            
            self.is_connected = True
            self.stats['connection_time'] = datetime.now() - start_time
            
            self.logger.info(f"MongoDB连接成功: {self.host}:{self.port}/{self.database}")
            return True
            
        except Exception as e:
            self.logger.error(f"MongoDB连接失败: {e}")
            self.stats['total_errors'] += 1
            raise DataConnectionError(f"MongoDB连接失败: {e}")
    
    async def disconnect(self) -> bool:
        """
        断开MongoDB连接
        
        Returns:
            bool: 断开是否成功
        """
        try:
            if self.client:
                self.client.close()
            
            self.is_connected = False
            self.logger.info("MongoDB连接已断开")
            return True
            
        except Exception as e:
            self.logger.error(f"MongoDB断开连接失败: {e}")
            return False
    
    async def health_check(self) -> bool:
        """
        MongoDB健康检查
        
        Returns:
            bool: 是否健康
        """
        try:
            if not self.is_connected or not self.client:
                return False
            
            result = await self.client.admin.command('ping')
            return result.get('ok') == 1
            
        except Exception as e:
            self.logger.error(f"MongoDB健康检查失败: {e}")
            return False
    
    async def _create_indexes(self):
        """
        创建数据库索引
        """
        try:
            # 行情数据索引
            tickers_collection = self.db[self._get_collection_name('tickers')]
            await tickers_collection.create_index([
                ('symbol', ASCENDING),
                ('timestamp', DESCENDING)
            ])
            await tickers_collection.create_index([('timestamp', DESCENDING)])
            
            # K线数据索引
            klines_collection = self.db[self._get_collection_name('klines')]
            await klines_collection.create_index([
                ('symbol', ASCENDING),
                ('interval', ASCENDING),
                ('open_time', DESCENDING)
            ], unique=True)
            await klines_collection.create_index([('open_time', DESCENDING)])
            
            # 订单索引
            orders_collection = self.db[self._get_collection_name('orders')]
            await orders_collection.create_index([('order_id', ASCENDING)], unique=True)
            await orders_collection.create_index([
                ('symbol', ASCENDING),
                ('status', ASCENDING)
            ])
            await orders_collection.create_index([('created_time', DESCENDING)])
            
            # 交易记录索引
            trades_collection = self.db[self._get_collection_name('trades')]
            await trades_collection.create_index([('trade_id', ASCENDING)], unique=True)
            await trades_collection.create_index([
                ('symbol', ASCENDING),
                ('trade_time', DESCENDING)
            ])
            await trades_collection.create_index([('order_id', ASCENDING)])
            
            # 策略配置索引
            strategies_collection = self.db[self._get_collection_name('strategies')]
            await strategies_collection.create_index([('name', ASCENDING)], unique=True)
            await strategies_collection.create_index([('strategy_type', ASCENDING)])
            
            # 系统日志索引
            logs_collection = self.db[self._get_collection_name('system_logs')]
            await logs_collection.create_index([
                ('level', ASCENDING),
                ('created_at', DESCENDING)
            ])
            await logs_collection.create_index([
                ('logger_name', ASCENDING),
                ('created_at', DESCENDING)
            ])
            
            # 创建TTL索引（自动删除过期日志）
            await logs_collection.create_index(
                [('created_at', ASCENDING)],
                expireAfterSeconds=30*24*3600  # 30天后自动删除
            )
            
            self.logger.info("MongoDB索引创建完成")
            
        except Exception as e:
            self.logger.warning(f"创建MongoDB索引失败: {e}")
    
    # ==================== 行情数据操作 ====================
    
    async def save_ticker(self, ticker: Ticker) -> bool:
        """
        保存行情数据
        
        Args:
            ticker: 行情数据
            
        Returns:
            bool: 保存是否成功
        """
        try:
            collection = self.db[self._get_collection_name('tickers')]
            
            document = {
                'symbol': ticker.symbol,
                'price': float(ticker.price),
                'volume': float(ticker.volume),
                'change': float(ticker.change) if ticker.change else None,
                'change_percent': float(ticker.change_percent) if ticker.change_percent else None,
                'high': float(ticker.high) if ticker.high else None,
                'low': float(ticker.low) if ticker.low else None,
                'timestamp': ticker.timestamp,
                'created_at': datetime.now()
            }
            
            await collection.insert_one(document)
            
            self.stats['total_writes'] += 1
            self.stats['last_operation_time'] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存行情数据失败: {e}")
            self.stats['total_errors'] += 1
            return False
    
    async def save_tickers(self, tickers: List[Ticker]) -> bool:
        """
        批量保存行情数据
        
        Args:
            tickers: 行情数据列表
            
        Returns:
            bool: 保存是否成功
        """
        try:
            collection = self.db[self._get_collection_name('tickers')]
            
            documents = [
                {
                    'symbol': ticker.symbol,
                    'price': float(ticker.price),
                    'volume': float(ticker.volume),
                    'change': float(ticker.change) if ticker.change else None,
                    'change_percent': float(ticker.change_percent) if ticker.change_percent else None,
                    'high': float(ticker.high) if ticker.high else None,
                    'low': float(ticker.low) if ticker.low else None,
                    'timestamp': ticker.timestamp,
                    'created_at': datetime.now()
                }
                for ticker in tickers
            ]
            
            await collection.insert_many(documents)
            
            self.stats['total_writes'] += len(tickers)
            self.stats['last_operation_time'] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"批量保存行情数据失败: {e}")
            self.stats['total_errors'] += 1
            return False
    
    async def get_latest_ticker(self, symbol: str) -> Optional[Ticker]:
        """
        获取最新行情数据
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[Ticker]: 行情数据
        """
        try:
            collection = self.db[self._get_collection_name('tickers')]
            
            document = await collection.find_one(
                {'symbol': symbol},
                sort=[('timestamp', DESCENDING)]
            )
            
            if document:
                ticker = Ticker(
                    symbol=document['symbol'],
                    price=document['price'],
                    volume=document['volume'],
                    change=document.get('change'),
                    change_percent=document.get('change_percent'),
                    high=document.get('high'),
                    low=document.get('low'),
                    timestamp=document['timestamp']
                )
                
                self.stats['total_reads'] += 1
                return ticker
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取最新行情数据失败: {e}")
            self.stats['total_errors'] += 1
            return None
    
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
        try:
            collection = self.db[self._get_collection_name('tickers')]
            
            # 构建查询条件
            query = {
                'symbol': symbol,
                'timestamp': {'$gte': start_time}
            }
            
            if end_time:
                query['timestamp']['$lte'] = end_time
            
            # 执行查询
            cursor = collection.find(query).sort('timestamp', DESCENDING)
            
            if limit:
                cursor = cursor.limit(limit)
            
            documents = await cursor.to_list(length=None)
            
            tickers = [
                Ticker(
                    symbol=doc['symbol'],
                    price=doc['price'],
                    volume=doc['volume'],
                    change=doc.get('change'),
                    change_percent=doc.get('change_percent'),
                    high=doc.get('high'),
                    low=doc.get('low'),
                    timestamp=doc['timestamp']
                )
                for doc in documents
            ]
            
            self.stats['total_reads'] += len(tickers)
            return tickers
            
        except Exception as e:
            self.logger.error(f"获取历史行情数据失败: {e}")
            self.stats['total_errors'] += 1
            return []
    
    # ==================== K线数据操作 ====================
    
    async def save_kline(self, kline: Kline) -> bool:
        """
        保存K线数据
        
        Args:
            kline: K线数据
            
        Returns:
            bool: 保存是否成功
        """
        try:
            collection = self.db[self._get_collection_name('klines')]
            
            document = {
                'symbol': kline.symbol,
                'interval': kline.interval,
                'open_time': kline.open_time,
                'close_time': kline.close_time,
                'open': float(kline.open),
                'high': float(kline.high),
                'low': float(kline.low),
                'close': float(kline.close),
                'volume': float(kline.volume),
                'quote_volume': float(kline.quote_volume) if kline.quote_volume else None,
                'trades_count': kline.trades_count,
                'created_at': datetime.now()
            }
            
            # 使用upsert避免重复数据
            await collection.replace_one(
                {
                    'symbol': kline.symbol,
                    'interval': kline.interval,
                    'open_time': kline.open_time
                },
                document,
                upsert=True
            )
            
            self.stats['total_writes'] += 1
            self.stats['last_operation_time'] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存K线数据失败: {e}")
            self.stats['total_errors'] += 1
            return False
    
    async def save_klines(self, klines: List[Kline]) -> bool:
        """
        批量保存K线数据
        
        Args:
            klines: K线数据列表
            
        Returns:
            bool: 保存是否成功
        """
        try:
            collection = self.db[self._get_collection_name('klines')]
            
            # 使用批量操作
            operations = []
            
            for kline in klines:
                document = {
                    'symbol': kline.symbol,
                    'interval': kline.interval,
                    'open_time': kline.open_time,
                    'close_time': kline.close_time,
                    'open': float(kline.open),
                    'high': float(kline.high),
                    'low': float(kline.low),
                    'close': float(kline.close),
                    'volume': float(kline.volume),
                    'quote_volume': float(kline.quote_volume) if kline.quote_volume else None,
                    'trades_count': kline.trades_count,
                    'created_at': datetime.now()
                }
                
                operations.append(
                    motor.motor_asyncio.ReplaceOne(
                        {
                            'symbol': kline.symbol,
                            'interval': kline.interval,
                            'open_time': kline.open_time
                        },
                        document,
                        upsert=True
                    )
                )
            
            if operations:
                await collection.bulk_write(operations)
            
            self.stats['total_writes'] += len(klines)
            self.stats['last_operation_time'] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"批量保存K线数据失败: {e}")
            self.stats['total_errors'] += 1
            return False
    
    async def get_latest_kline(self, symbol: str, interval: str) -> Optional[Kline]:
        """
        获取最新K线数据
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            
        Returns:
            Optional[Kline]: K线数据
        """
        try:
            collection = self.db[self._get_collection_name('klines')]
            
            document = await collection.find_one(
                {'symbol': symbol, 'interval': interval},
                sort=[('open_time', DESCENDING)]
            )
            
            if document:
                kline = Kline(
                    symbol=document['symbol'],
                    interval=document['interval'],
                    open_time=document['open_time'],
                    close_time=document['close_time'],
                    open=document['open'],
                    high=document['high'],
                    low=document['low'],
                    close=document['close'],
                    volume=document['volume'],
                    quote_volume=document.get('quote_volume'),
                    trades_count=document.get('trades_count')
                )
                
                self.stats['total_reads'] += 1
                return kline
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取最新K线数据失败: {e}")
            self.stats['total_errors'] += 1
            return None
    
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
        try:
            collection = self.db[self._get_collection_name('klines')]
            
            # 构建查询条件
            query = {
                'symbol': symbol,
                'interval': interval,
                'open_time': {'$gte': start_time}
            }
            
            if end_time:
                query['open_time']['$lte'] = end_time
            
            # 执行查询
            cursor = collection.find(query).sort('open_time', DESCENDING)
            
            if limit:
                cursor = cursor.limit(limit)
            
            documents = await cursor.to_list(length=None)
            
            klines = [
                Kline(
                    symbol=doc['symbol'],
                    interval=doc['interval'],
                    open_time=doc['open_time'],
                    close_time=doc['close_time'],
                    open=doc['open'],
                    high=doc['high'],
                    low=doc['low'],
                    close=doc['close'],
                    volume=doc['volume'],
                    quote_volume=doc.get('quote_volume'),
                    trades_count=doc.get('trades_count')
                )
                for doc in documents
            ]
            
            self.stats['total_reads'] += len(klines)
            return klines
            
        except Exception as e:
            self.logger.error(f"获取历史K线数据失败: {e}")
            self.stats['total_errors'] += 1
            return []
    
    # ==================== 订单数据操作 ====================
    
    async def save_order(self, order: Order) -> bool:
        """
        保存订单数据
        
        Args:
            order: 订单数据
            
        Returns:
            bool: 保存是否成功
        """
        try:
            collection = self.db[self._get_collection_name('orders')]
            
            document = {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side,
                'type': order.type,
                'quantity': float(order.quantity),
                'price': float(order.price) if order.price else None,
                'status': order.status,
                'filled_quantity': float(order.filled_quantity) if order.filled_quantity else 0,
                'avg_price': float(order.avg_price) if order.avg_price else None,
                'commission': float(order.commission) if order.commission else None,
                'commission_asset': order.commission_asset,
                'created_time': order.created_time,
                'updated_time': order.updated_time,
                'created_at': datetime.now()
            }
            
            # 使用upsert更新或插入
            await collection.replace_one(
                {'order_id': order.order_id},
                document,
                upsert=True
            )
            
            self.stats['total_writes'] += 1
            self.stats['last_operation_time'] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存订单数据失败: {e}")
            self.stats['total_errors'] += 1
            return False
    
    async def save_trade(self, trade: Trade) -> bool:
        """
        保存交易记录
        
        Args:
            trade: 交易记录
            
        Returns:
            bool: 保存是否成功
        """
        try:
            collection = self.db[self._get_collection_name('trades')]
            
            document = {
                'trade_id': trade.trade_id,
                'order_id': trade.order_id,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': float(trade.quantity),
                'price': float(trade.price),
                'commission': float(trade.commission) if trade.commission else None,
                'commission_asset': trade.commission_asset,
                'trade_time': trade.trade_time,
                'created_at': datetime.now()
            }
            
            # 避免重复插入
            try:
                await collection.insert_one(document)
            except DuplicateKeyError:
                # 交易记录已存在，忽略
                pass
            
            self.stats['total_writes'] += 1
            self.stats['last_operation_time'] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存交易记录失败: {e}")
            self.stats['total_errors'] += 1
            return False
    
    # ==================== 配置数据操作 ====================
    
    async def save_strategy_config(self, name: str, config: Dict[str, Any]) -> bool:
        """
        保存策略配置
        
        Args:
            name: 策略名称
            config: 策略配置
            
        Returns:
            bool: 保存是否成功
        """
        try:
            collection = self.db[self._get_collection_name('strategies')]
            
            document = {
                'name': name,
                'description': config.get('description', ''),
                'strategy_type': config.get('strategy_type', ''),
                'parameters': config.get('parameters', {}),
                'is_active': config.get('is_active', True),
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # 使用upsert更新或插入
            await collection.replace_one(
                {'name': name},
                document,
                upsert=True
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存策略配置失败: {e}")
            return False
    
    async def get_strategy_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取策略配置
        
        Args:
            name: 策略名称
            
        Returns:
            Optional[Dict[str, Any]]: 策略配置
        """
        try:
            collection = self.db[self._get_collection_name('strategies')]
            
            document = await collection.find_one({'name': name})
            
            if document:
                # 移除MongoDB的_id字段
                document.pop('_id', None)
                return document
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取策略配置失败: {e}")
            return None
    
    # ==================== 日志操作 ====================
    
    async def save_log(self, log_record: Dict[str, Any]) -> bool:
        """
        保存系统日志
        
        Args:
            log_record: 日志记录
            
        Returns:
            bool: 保存是否成功
        """
        try:
            collection = self.db[self._get_collection_name('system_logs')]
            
            document = {
                'level': log_record.get('level'),
                'logger_name': log_record.get('logger_name'),
                'message': log_record.get('message'),
                'module': log_record.get('module'),
                'function': log_record.get('function'),
                'line_number': log_record.get('line_number'),
                'exception': log_record.get('exception'),
                'extra_data': log_record.get('extra_data', {}),
                'created_at': datetime.now()
            }
            
            await collection.insert_one(document)
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存系统日志失败: {e}")
            return False
    
    # ==================== 聚合查询 ====================
    
    async def aggregate_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        aggregation_type: str = 'ohlcv'
    ) -> Dict[str, Any]:
        """
        聚合K线数据
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            start_time: 开始时间
            end_time: 结束时间
            aggregation_type: 聚合类型
            
        Returns:
            Dict[str, Any]: 聚合结果
        """
        try:
            collection = self.db[self._get_collection_name('klines')]
            
            # 构建匹配条件
            match_stage = {
                'symbol': symbol,
                'interval': interval,
                'open_time': {'$gte': start_time}
            }
            
            if end_time:
                match_stage['open_time']['$lte'] = end_time
            
            # 构建聚合管道
            pipeline = [
                {'$match': match_stage}
            ]
            
            if aggregation_type == 'ohlcv':
                pipeline.append({
                    '$group': {
                        '_id': None,
                        'count': {'$sum': 1},
                        'first_open': {'$first': '$open'},
                        'last_close': {'$last': '$close'},
                        'max_high': {'$max': '$high'},
                        'min_low': {'$min': '$low'},
                        'total_volume': {'$sum': '$volume'},
                        'avg_volume': {'$avg': '$volume'},
                        'first_time': {'$min': '$open_time'},
                        'last_time': {'$max': '$open_time'}
                    }
                })
            elif aggregation_type == 'volume_profile':
                pipeline.extend([
                    {
                        '$group': {
                            '_id': {
                                '$round': [{'$divide': ['$close', 10]}, 0]
                            },
                            'volume': {'$sum': '$volume'},
                            'count': {'$sum': 1}
                        }
                    },
                    {'$sort': {'volume': -1}}
                ])
            
            # 执行聚合查询
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            return {
                'aggregation_type': aggregation_type,
                'symbol': symbol,
                'interval': interval,
                'start_time': start_time,
                'end_time': end_time,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"聚合K线数据失败: {e}")
            return {}
    
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
            'connection_info': {
                'host': self.host,
                'port': self.port,
                'database': self.database,
                'username': self.username
            },
            'performance': self.stats.copy(),
            'config': {
                'auth_source': self.auth_source,
                'replica_set': self.replica_set,
                'ssl': self.ssl,
                'max_pool_size': self.max_pool_size,
                'min_pool_size': self.min_pool_size,
                'max_idle_time_ms': self.max_idle_time_ms,
                'collection_prefix': self.collection_prefix
            }
        }
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            Dict[str, Any]: 集合统计信息
        """
        try:
            collections = ['tickers', 'klines', 'orders', 'trades', 'strategies', 'system_logs']
            stats = {}
            
            for collection_type in collections:
                collection_name = self._get_collection_name(collection_type)
                collection = self.db[collection_name]
                
                # 获取文档数量
                count = await collection.count_documents({})
                
                # 获取集合统计信息
                try:
                    collection_stats = await self.db.command('collStats', collection_name)
                    stats[collection_type] = {
                        'count': count,
                        'size': collection_stats.get('size', 0),
                        'storage_size': collection_stats.get('storageSize', 0),
                        'avg_obj_size': collection_stats.get('avgObjSize', 0),
                        'indexes': collection_stats.get('nindexes', 0)
                    }
                except Exception:
                    stats[collection_type] = {'count': count}
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取集合统计信息失败: {e}")
            return {}