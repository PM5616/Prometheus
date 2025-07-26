"""Redis Storage Implementation

Redis存储实现，用于缓存和实时数据存储。

功能特性：
- 高性能内存存储
- 实时数据缓存
- 发布订阅功能
- 数据过期管理
- 集群支持
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging

try:
    import redis.asyncio as redis
    import redis.exceptions
except ImportError:
    redis = None

from .base import BaseStorage, StorageType
from src.common.models.market import Ticker, Kline, OrderBook
from src.common.models.trading import Symbol, Order, Trade
from src.common.exceptions.data import (
    DataConnectionError, DataNotFoundError, DataValidationError
)


class RedisStorage(BaseStorage):
    """Redis存储实现
    
    用于高性能缓存和实时数据存储。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Redis存储
        
        Args:
            config: Redis配置
                - host: Redis主机地址
                - port: Redis端口
                - password: Redis密码
                - db: 数据库编号
                - max_connections: 最大连接数
                - decode_responses: 是否解码响应
                - socket_timeout: 套接字超时
                - socket_connect_timeout: 连接超时
        """
        super().__init__(StorageType.REDIS.value, config)
        
        if not redis:
            raise ImportError("redis package is required for RedisStorage")
        
        # Redis配置
        self.db = config.get('db', 0)
        self.max_connections = config.get('max_connections', 10)
        self.decode_responses = config.get('decode_responses', True)
        self.socket_timeout = config.get('socket_timeout', 5)
        self.socket_connect_timeout = config.get('socket_connect_timeout', 5)
        
        # Redis连接池
        self.pool = None
        self.client = None
        
        # 缓存配置
        self.default_ttl = config.get('default_ttl', 3600)  # 默认1小时过期
        self.ticker_ttl = config.get('ticker_ttl', 60)  # 行情数据1分钟过期
        self.kline_ttl = config.get('kline_ttl', 300)  # K线数据5分钟过期
        
        # 键前缀
        self.key_prefix = config.get('key_prefix', 'prometheus:')
        
        self.logger.info("Redis存储初始化完成")
    
    def _get_key(self, key_type: str, *args) -> str:
        """
        生成Redis键名
        
        Args:
            key_type: 键类型
            *args: 键参数
            
        Returns:
            str: 完整的键名
        """
        parts = [self.key_prefix, key_type] + [str(arg) for arg in args]
        return ':'.join(parts)
    
    async def connect(self) -> bool:
        """
        建立Redis连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            start_time = datetime.now()
            
            # 创建连接池
            self.pool = redis.ConnectionPool(
                host=self.host,
                port=self.port or 6379,
                password=self.password,
                db=self.db,
                max_connections=self.max_connections,
                decode_responses=self.decode_responses,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout
            )
            
            # 创建Redis客户端
            self.client = redis.Redis(connection_pool=self.pool)
            
            # 测试连接
            await self.client.ping()
            
            self.is_connected = True
            self.stats['connection_time'] = datetime.now() - start_time
            
            self.logger.info(f"Redis连接成功: {self.host}:{self.port}/{self.db}")
            return True
            
        except Exception as e:
            self.logger.error(f"Redis连接失败: {e}")
            self.stats['total_errors'] += 1
            raise DataConnectionError(f"Redis连接失败: {e}")
    
    async def disconnect(self) -> bool:
        """
        断开Redis连接
        
        Returns:
            bool: 断开是否成功
        """
        try:
            if self.client:
                await self.client.close()
            if self.pool:
                await self.pool.disconnect()
            
            self.is_connected = False
            self.logger.info("Redis连接已断开")
            return True
            
        except Exception as e:
            self.logger.error(f"Redis断开连接失败: {e}")
            return False
    
    async def health_check(self) -> bool:
        """
        Redis健康检查
        
        Returns:
            bool: 是否健康
        """
        try:
            if not self.is_connected or not self.client:
                return False
            
            # 执行ping命令
            result = await self.client.ping()
            return result is True
            
        except Exception as e:
            self.logger.error(f"Redis健康检查失败: {e}")
            return False
    
    # ==================== 行情数据操作 ====================
    
    async def save_ticker(self, ticker: Ticker) -> bool:
        """
        保存行情数据到Redis
        
        Args:
            ticker: 行情数据
            
        Returns:
            bool: 保存是否成功
        """
        try:
            key = self._get_key('ticker', ticker.symbol)
            data = {
                'symbol': ticker.symbol,
                'price': float(ticker.price),
                'volume': float(ticker.volume),
                'change': float(ticker.change) if ticker.change else None,
                'change_percent': float(ticker.change_percent) if ticker.change_percent else None,
                'high': float(ticker.high) if ticker.high else None,
                'low': float(ticker.low) if ticker.low else None,
                'timestamp': ticker.timestamp.isoformat()
            }
            
            # 保存到Redis，设置过期时间
            await self.client.setex(
                key, 
                self.ticker_ttl, 
                json.dumps(data)
            )
            
            # 同时保存到最新行情列表
            latest_key = self._get_key('latest_tickers')
            await self.client.hset(latest_key, ticker.symbol, json.dumps(data))
            
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
            # 使用管道批量操作
            pipe = self.client.pipeline()
            
            latest_data = {}
            
            for ticker in tickers:
                key = self._get_key('ticker', ticker.symbol)
                data = {
                    'symbol': ticker.symbol,
                    'price': float(ticker.price),
                    'volume': float(ticker.volume),
                    'change': float(ticker.change) if ticker.change else None,
                    'change_percent': float(ticker.change_percent) if ticker.change_percent else None,
                    'high': float(ticker.high) if ticker.high else None,
                    'low': float(ticker.low) if ticker.low else None,
                    'timestamp': ticker.timestamp.isoformat()
                }
                
                # 添加到管道
                pipe.setex(key, self.ticker_ttl, json.dumps(data))
                latest_data[ticker.symbol] = json.dumps(data)
            
            # 批量更新最新行情
            if latest_data:
                latest_key = self._get_key('latest_tickers')
                pipe.hmset(latest_key, latest_data)
            
            # 执行管道
            await pipe.execute()
            
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
            key = self._get_key('ticker', symbol)
            data = await self.client.get(key)
            
            if not data:
                # 尝试从最新行情列表获取
                latest_key = self._get_key('latest_tickers')
                data = await self.client.hget(latest_key, symbol)
            
            if data:
                ticker_data = json.loads(data)
                ticker = Ticker(
                    symbol=ticker_data['symbol'],
                    price=ticker_data['price'],
                    volume=ticker_data['volume'],
                    change=ticker_data.get('change'),
                    change_percent=ticker_data.get('change_percent'),
                    high=ticker_data.get('high'),
                    low=ticker_data.get('low'),
                    timestamp=datetime.fromisoformat(ticker_data['timestamp'])
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
        获取历史行情数据（Redis主要用于缓存，历史数据有限）
        
        Args:
            symbol: 交易对符号
            start_time: 开始时间
            end_time: 结束时间
            limit: 数据条数限制
            
        Returns:
            List[Ticker]: 行情数据列表
        """
        # Redis主要用于缓存最新数据，历史数据查询返回空列表
        # 实际应用中应该从时序数据库查询历史数据
        self.logger.warning("Redis存储不支持历史行情数据查询，请使用时序数据库")
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
            key = self._get_key('kline', kline.symbol, kline.interval)
            data = {
                'symbol': kline.symbol,
                'interval': kline.interval,
                'open_time': kline.open_time.isoformat(),
                'close_time': kline.close_time.isoformat(),
                'open': float(kline.open),
                'high': float(kline.high),
                'low': float(kline.low),
                'close': float(kline.close),
                'volume': float(kline.volume),
                'quote_volume': float(kline.quote_volume) if kline.quote_volume else None,
                'trades_count': kline.trades_count if kline.trades_count else None
            }
            
            # 保存最新K线
            await self.client.setex(
                key, 
                self.kline_ttl, 
                json.dumps(data)
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
            pipe = self.client.pipeline()
            
            for kline in klines:
                key = self._get_key('kline', kline.symbol, kline.interval)
                data = {
                    'symbol': kline.symbol,
                    'interval': kline.interval,
                    'open_time': kline.open_time.isoformat(),
                    'close_time': kline.close_time.isoformat(),
                    'open': float(kline.open),
                    'high': float(kline.high),
                    'low': float(kline.low),
                    'close': float(kline.close),
                    'volume': float(kline.volume),
                    'quote_volume': float(kline.quote_volume) if kline.quote_volume else None,
                    'trades_count': kline.trades_count if kline.trades_count else None
                }
                
                pipe.setex(key, self.kline_ttl, json.dumps(data))
            
            await pipe.execute()
            
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
            key = self._get_key('kline', symbol, interval)
            data = await self.client.get(key)
            
            if data:
                kline_data = json.loads(data)
                kline = Kline(
                    symbol=kline_data['symbol'],
                    interval=kline_data['interval'],
                    open_time=datetime.fromisoformat(kline_data['open_time']),
                    close_time=datetime.fromisoformat(kline_data['close_time']),
                    open=kline_data['open'],
                    high=kline_data['high'],
                    low=kline_data['low'],
                    close=kline_data['close'],
                    volume=kline_data['volume'],
                    quote_volume=kline_data.get('quote_volume'),
                    trades_count=kline_data.get('trades_count')
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
        获取历史K线数据（Redis主要用于缓存，历史数据有限）
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            start_time: 开始时间
            end_time: 结束时间
            limit: 数据条数限制
            
        Returns:
            List[Kline]: K线数据列表
        """
        # Redis主要用于缓存最新数据，历史数据查询返回空列表
        self.logger.warning("Redis存储不支持历史K线数据查询，请使用时序数据库")
        return []
    
    # ==================== 订单簿数据操作 ====================
    
    async def save_orderbook(self, orderbook: OrderBook) -> bool:
        """
        保存订单簿数据
        
        Args:
            orderbook: 订单簿数据
            
        Returns:
            bool: 保存是否成功
        """
        try:
            key = self._get_key('orderbook', orderbook.symbol)
            data = {
                'symbol': orderbook.symbol,
                'bids': [[float(price), float(qty)] for price, qty in orderbook.bids],
                'asks': [[float(price), float(qty)] for price, qty in orderbook.asks],
                'timestamp': orderbook.timestamp.isoformat()
            }
            
            # 订单簿数据实时性要求高，设置较短过期时间
            await self.client.setex(
                key, 
                30,  # 30秒过期
                json.dumps(data)
            )
            
            self.stats['total_writes'] += 1
            self.stats['last_operation_time'] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存订单簿数据失败: {e}")
            self.stats['total_errors'] += 1
            return False
    
    async def get_latest_orderbook(self, symbol: str) -> Optional[OrderBook]:
        """
        获取最新订单簿数据
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[OrderBook]: 订单簿数据
        """
        try:
            key = self._get_key('orderbook', symbol)
            data = await self.client.get(key)
            
            if data:
                orderbook_data = json.loads(data)
                orderbook = OrderBook(
                    symbol=orderbook_data['symbol'],
                    bids=[(price, qty) for price, qty in orderbook_data['bids']],
                    asks=[(price, qty) for price, qty in orderbook_data['asks']],
                    timestamp=datetime.fromisoformat(orderbook_data['timestamp'])
                )
                
                self.stats['total_reads'] += 1
                return orderbook
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取最新订单簿数据失败: {e}")
            self.stats['total_errors'] += 1
            return None
    
    # ==================== 缓存操作 ====================
    
    async def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
            
        Returns:
            bool: 设置是否成功
        """
        try:
            cache_key = self._get_key('cache', key)
            data = json.dumps(value) if not isinstance(value, str) else value
            
            if ttl:
                await self.client.setex(cache_key, ttl, data)
            else:
                await self.client.setex(cache_key, self.default_ttl, data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"设置缓存失败: {e}")
            return False
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存值
        """
        try:
            cache_key = self._get_key('cache', key)
            data = await self.client.get(cache_key)
            
            if data:
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取缓存失败: {e}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 删除是否成功
        """
        try:
            cache_key = self._get_key('cache', key)
            result = await self.client.delete(cache_key)
            return result > 0
            
        except Exception as e:
            self.logger.error(f"删除缓存失败: {e}")
            return False
    
    # ==================== 发布订阅 ====================
    
    async def publish(self, channel: str, message: Any) -> bool:
        """
        发布消息
        
        Args:
            channel: 频道名
            message: 消息内容
            
        Returns:
            bool: 发布是否成功
        """
        try:
            channel_key = self._get_key('channel', channel)
            data = json.dumps(message) if not isinstance(message, str) else message
            
            result = await self.client.publish(channel_key, data)
            return result > 0
            
        except Exception as e:
            self.logger.error(f"发布消息失败: {e}")
            return False
    
    async def subscribe(self, channel: str, callback):
        """
        订阅频道
        
        Args:
            channel: 频道名
            callback: 回调函数
        """
        try:
            channel_key = self._get_key('channel', channel)
            pubsub = self.client.pubsub()
            await pubsub.subscribe(channel_key)
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                    except json.JSONDecodeError:
                        data = message['data']
                    
                    await callback(data)
                    
        except Exception as e:
            self.logger.error(f"订阅频道失败: {e}")
    
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
                'db': self.db
            },
            'performance': self.stats.copy(),
            'config': {
                'default_ttl': self.default_ttl,
                'ticker_ttl': self.ticker_ttl,
                'kline_ttl': self.kline_ttl,
                'max_connections': self.max_connections
            }
        }
    
    async def clear_cache(self, pattern: Optional[str] = None) -> bool:
        """
        清空缓存
        
        Args:
            pattern: 键模式，如果为None则清空所有缓存
            
        Returns:
            bool: 清空是否成功
        """
        try:
            if pattern:
                # 删除匹配模式的键
                search_pattern = self._get_key('*', pattern)
                keys = await self.client.keys(search_pattern)
                if keys:
                    await self.client.delete(*keys)
            else:
                # 清空当前数据库
                await self.client.flushdb()
            
            self.logger.info(f"缓存清空完成，模式: {pattern or '全部'}")
            return True
            
        except Exception as e:
            self.logger.error(f"清空缓存失败: {e}")
            return False