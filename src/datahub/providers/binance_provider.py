"""Binance Data Provider

币安数据提供商实现，提供币安交易所的实时和历史数据。

功能特性：
- WebSocket实时数据流
- REST API历史数据
- 自动重连机制
- 数据质量监控
"""

import asyncio
import json
import websockets
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal

from .base import BaseDataProvider, DataType, SubscriptionStatus
from src.common.models.market import Ticker, Kline, OrderBook, OrderBookEntry
from src.common.models.trading import Symbol
from src.common.exceptions.data import (
    DataConnectionError, DataNotFoundError, DataValidationError
)
from src.common.utils.datetime_utils import timestamp_to_datetime, datetime_to_timestamp


class BinanceProvider(BaseDataProvider):
    """币安数据提供商
    
    提供币安交易所的实时和历史数据服务。
    支持现货、期货等多种市场数据。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化币安数据提供商
        
        Args:
            config: 配置参数，包含API密钥、WebSocket URL等
        """
        super().__init__("binance", config)
        
        # API配置
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', False)
        
        # WebSocket配置
        self.ws_base_url = config.get('ws_url', 'wss://stream.binance.com:9443/ws')
        if self.testnet:
            self.ws_base_url = 'wss://testnet.binance.vision/ws'
        
        # REST API配置
        self.rest_base_url = config.get('rest_url', 'https://api.binance.com')
        if self.testnet:
            self.rest_base_url = 'https://testnet.binance.vision'
        
        # 连接管理
        self.websocket = None
        self.reconnect_interval = config.get('reconnect_interval', 5)
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        self.reconnect_attempts = 0
        
        # 订阅管理
        self.stream_names: List[str] = []
        self.pending_subscriptions: List[Dict] = []
        
        # 数据缓存
        self.symbol_info_cache: Dict[str, Symbol] = {}
        self.last_prices: Dict[str, Decimal] = {}
        
        self.logger.info(f"币安数据提供商初始化完成，测试网模式: {self.testnet}")
    
    async def connect(self) -> bool:
        """
        建立WebSocket连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.logger.info("正在连接币安WebSocket...")
            
            # 构建WebSocket URL
            if self.stream_names:
                stream_url = f"{self.ws_base_url}/stream?streams={''.join(self.stream_names)}"
            else:
                stream_url = self.ws_base_url
            
            # 建立WebSocket连接
            self.websocket = await websockets.connect(
                stream_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.is_connected = True
            self.last_heartbeat = datetime.now()
            self.stats['connection_time'] = datetime.now()
            self.reconnect_attempts = 0
            
            self.logger.info("币安WebSocket连接成功")
            
            # 启动消息处理任务
            asyncio.create_task(self._message_handler())
            
            # 处理待订阅的流
            if self.pending_subscriptions:
                await self._process_pending_subscriptions()
            
            return True
            
        except Exception as e:
            self.logger.error(f"币安WebSocket连接失败: {e}")
            self.is_connected = False
            self._update_stats(error=True)
            return False
    
    async def disconnect(self) -> bool:
        """
        断开WebSocket连接
        
        Returns:
            bool: 断开是否成功
        """
        try:
            self.logger.info("正在断开币安WebSocket连接...")
            
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
            
            self.is_connected = False
            self.websocket = None
            
            # 重置订阅状态
            for key in self.subscriptions:
                self.subscriptions[key] = SubscriptionStatus.STOPPED
            
            self.logger.info("币安WebSocket连接已断开")
            return True
            
        except Exception as e:
            self.logger.error(f"断开币安WebSocket连接失败: {e}")
            return False
    
    async def subscribe_ticker(self, symbol: str, callback: callable) -> bool:
        """
        订阅行情数据
        
        Args:
            symbol: 交易对符号
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        try:
            symbol = symbol.upper()
            subscription_key = self._generate_subscription_key(DataType.TICKER, symbol)
            stream_name = f"{symbol.lower()}@ticker"
            
            # 添加回调函数
            self.add_subscription_callback(subscription_key, callback)
            
            # 添加到流列表
            if stream_name not in self.stream_names:
                self.stream_names.append(stream_name)
            
            # 更新订阅状态
            self.subscriptions[subscription_key] = SubscriptionStatus.ACTIVE
            
            self.logger.info(f"订阅币安行情数据: {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"订阅币安行情数据失败 {symbol}: {e}")
            return False
    
    async def subscribe_kline(self, symbol: str, interval: str, callback: callable) -> bool:
        """
        订阅K线数据
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        try:
            symbol = symbol.upper()
            subscription_key = self._generate_subscription_key(DataType.KLINE, symbol, interval=interval)
            stream_name = f"{symbol.lower()}@kline_{interval}"
            
            # 添加回调函数
            self.add_subscription_callback(subscription_key, callback)
            
            # 添加到流列表
            if stream_name not in self.stream_names:
                self.stream_names.append(stream_name)
            
            # 更新订阅状态
            self.subscriptions[subscription_key] = SubscriptionStatus.ACTIVE
            
            self.logger.info(f"订阅币安K线数据: {symbol} {interval}")
            return True
            
        except Exception as e:
            self.logger.error(f"订阅币安K线数据失败 {symbol} {interval}: {e}")
            return False
    
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
        try:
            symbol = symbol.upper()
            subscription_key = self._generate_subscription_key(DataType.ORDER_BOOK, symbol, depth=depth)
            
            # 币安订单簿流名称格式
            if depth in [5, 10, 20]:
                stream_name = f"{symbol.lower()}@depth{depth}@100ms"
            else:
                stream_name = f"{symbol.lower()}@depth@100ms"
            
            # 添加回调函数
            self.add_subscription_callback(subscription_key, callback)
            
            # 添加到流列表
            if stream_name not in self.stream_names:
                self.stream_names.append(stream_name)
            
            # 更新订阅状态
            self.subscriptions[subscription_key] = SubscriptionStatus.ACTIVE
            
            self.logger.info(f"订阅币安订单簿数据: {symbol} depth={depth}")
            return True
            
        except Exception as e:
            self.logger.error(f"订阅币安订单簿数据失败 {symbol}: {e}")
            return False
    
    async def unsubscribe(self, subscription_key: str) -> bool:
        """
        取消订阅
        
        Args:
            subscription_key: 订阅键
            
        Returns:
            bool: 取消订阅是否成功
        """
        try:
            if subscription_key in self.subscriptions:
                self.subscriptions[subscription_key] = SubscriptionStatus.STOPPED
                
                # 清理回调函数
                if subscription_key in self.subscription_callbacks:
                    del self.subscription_callbacks[subscription_key]
                
                self.logger.info(f"取消订阅: {subscription_key}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"取消订阅失败 {subscription_key}: {e}")
            return False
    
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
        try:
            # TODO: 实现REST API调用获取历史K线数据
            # 这里需要使用aiohttp或类似库调用币安REST API
            self.logger.warning("历史K线数据获取功能待实现")
            return []
            
        except Exception as e:
            self.logger.error(f"获取历史K线数据失败 {symbol}: {e}")
            raise DataNotFoundError(f"获取历史K线数据失败: {e}")
    
    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """
        获取当前行情数据
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[Ticker]: 行情数据
        """
        try:
            # TODO: 实现REST API调用获取当前行情
            self.logger.warning("当前行情数据获取功能待实现")
            return None
            
        except Exception as e:
            self.logger.error(f"获取当前行情数据失败 {symbol}: {e}")
            return None
    
    async def get_order_book(self, symbol: str, depth: int = 20) -> Optional[OrderBook]:
        """
        获取当前订单簿数据
        
        Args:
            symbol: 交易对符号
            depth: 订单簿深度
            
        Returns:
            Optional[OrderBook]: 订单簿数据
        """
        try:
            # TODO: 实现REST API调用获取订单簿
            self.logger.warning("订单簿数据获取功能待实现")
            return None
            
        except Exception as e:
            self.logger.error(f"获取订单簿数据失败 {symbol}: {e}")
            return None
    
    async def get_symbols(self) -> List[Symbol]:
        """
        获取支持的交易对列表
        
        Returns:
            List[Symbol]: 交易对列表
        """
        try:
            # TODO: 实现REST API调用获取交易对信息
            self.logger.warning("交易对列表获取功能待实现")
            return []
            
        except Exception as e:
            self.logger.error(f"获取交易对列表失败: {e}")
            return []
    
    async def _message_handler(self):
        """
        WebSocket消息处理器
        """
        try:
            while self.is_connected and self.websocket:
                try:
                    # 接收消息
                    message = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=30
                    )
                    
                    self._update_stats(message_received=True)
                    self.last_heartbeat = datetime.now()
                    
                    # 解析消息
                    await self._process_message(message)
                    
                except asyncio.TimeoutError:
                    self.logger.warning("WebSocket消息接收超时")
                    continue
                    
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("WebSocket连接已关闭")
                    break
                    
                except Exception as e:
                    self.logger.error(f"处理WebSocket消息失败: {e}")
                    self._update_stats(error=True)
                    continue
            
            # 连接断开，尝试重连
            if self.is_connected:
                await self._handle_reconnect()
                
        except Exception as e:
            self.logger.error(f"WebSocket消息处理器异常: {e}")
            self._update_stats(error=True)
    
    async def _process_message(self, message: str):
        """
        处理WebSocket消息
        
        Args:
            message: 原始消息字符串
        """
        try:
            data = json.loads(message)
            
            # 处理不同类型的消息
            if 'stream' in data and 'data' in data:
                stream_name = data['stream']
                stream_data = data['data']
                
                await self._process_stream_data(stream_name, stream_data)
            
            self._update_stats(message_processed=True)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            self._update_stats(error=True)
        except Exception as e:
            self.logger.error(f"处理消息失败: {e}")
            self._update_stats(error=True)
    
    async def _process_stream_data(self, stream_name: str, data: Dict[str, Any]):
        """
        处理流数据
        
        Args:
            stream_name: 流名称
            data: 流数据
        """
        try:
            # 解析流名称
            parts = stream_name.split('@')
            if len(parts) < 2:
                return
            
            symbol = parts[0].upper()
            stream_type = parts[1]
            
            # 根据流类型处理数据
            if stream_type == 'ticker':
                await self._process_ticker_data(symbol, data)
            elif stream_type.startswith('kline'):
                interval = stream_type.split('_')[1]
                await self._process_kline_data(symbol, interval, data)
            elif stream_type.startswith('depth'):
                await self._process_depth_data(symbol, data)
            
        except Exception as e:
            self.logger.error(f"处理流数据失败 {stream_name}: {e}")
            self._update_stats(error=True)
    
    async def _process_ticker_data(self, symbol: str, data: Dict[str, Any]):
        """
        处理行情数据
        
        Args:
            symbol: 交易对符号
            data: 行情数据
        """
        try:
            ticker = Ticker(
                symbol=symbol,
                price=Decimal(data['c']),  # 最新价格
                bid_price=Decimal(data['b']),  # 买一价
                ask_price=Decimal(data['a']),  # 卖一价
                volume=Decimal(data['v']),  # 24小时成交量
                quote_volume=Decimal(data['q']),  # 24小时成交额
                open_price=Decimal(data['o']),  # 开盘价
                high_price=Decimal(data['h']),  # 最高价
                low_price=Decimal(data['l']),  # 最低价
                price_change=Decimal(data['p']),  # 价格变化
                price_change_percent=Decimal(data['P']),  # 价格变化百分比
                timestamp=timestamp_to_datetime(int(data['E']))  # 事件时间
            )
            
            # 更新价格缓存
            self.last_prices[symbol] = ticker.price
            
            # 通知回调函数
            subscription_key = self._generate_subscription_key(DataType.TICKER, symbol)
            await self._notify_callbacks(subscription_key, ticker)
            
        except Exception as e:
            self.logger.error(f"处理行情数据失败 {symbol}: {e}")
            self._update_stats(error=True)
    
    async def _process_kline_data(self, symbol: str, interval: str, data: Dict[str, Any]):
        """
        处理K线数据
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            data: K线数据
        """
        try:
            kline_data = data['k']
            
            kline = Kline(
                symbol=symbol,
                interval=interval,
                open_time=timestamp_to_datetime(int(kline_data['t'])),
                close_time=timestamp_to_datetime(int(kline_data['T'])),
                open_price=Decimal(kline_data['o']),
                high_price=Decimal(kline_data['h']),
                low_price=Decimal(kline_data['l']),
                close_price=Decimal(kline_data['c']),
                volume=Decimal(kline_data['v']),
                quote_volume=Decimal(kline_data['q']),
                trade_count=int(kline_data['n']),
                is_closed=bool(kline_data['x'])  # K线是否完结
            )
            
            # 通知回调函数
            subscription_key = self._generate_subscription_key(DataType.KLINE, symbol, interval=interval)
            await self._notify_callbacks(subscription_key, kline)
            
        except Exception as e:
            self.logger.error(f"处理K线数据失败 {symbol} {interval}: {e}")
            self._update_stats(error=True)
    
    async def _process_depth_data(self, symbol: str, data: Dict[str, Any]):
        """
        处理订单簿数据
        
        Args:
            symbol: 交易对符号
            data: 订单簿数据
        """
        try:
            # 解析买单和卖单
            bids = [OrderBookEntry(price=Decimal(bid[0]), quantity=Decimal(bid[1])) 
                   for bid in data['b']]
            asks = [OrderBookEntry(price=Decimal(ask[0]), quantity=Decimal(ask[1])) 
                   for ask in data['a']]
            
            order_book = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=timestamp_to_datetime(int(data['E']))
            )
            
            # 通知回调函数
            subscription_key = self._generate_subscription_key(DataType.ORDER_BOOK, symbol)
            await self._notify_callbacks(subscription_key, order_book)
            
        except Exception as e:
            self.logger.error(f"处理订单簿数据失败 {symbol}: {e}")
            self._update_stats(error=True)
    
    async def _handle_reconnect(self):
        """
        处理重连逻辑
        """
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(f"重连次数超过限制 ({self.max_reconnect_attempts})，停止重连")
            return
        
        self.reconnect_attempts += 1
        self.logger.info(f"尝试重连 ({self.reconnect_attempts}/{self.max_reconnect_attempts})...")
        
        await asyncio.sleep(self.reconnect_interval)
        
        success = await self.connect()
        if not success:
            await self._handle_reconnect()
    
    async def _process_pending_subscriptions(self):
        """
        处理待订阅的流
        """
        for subscription in self.pending_subscriptions:
            # TODO: 重新订阅待处理的订阅
            pass
        
        self.pending_subscriptions.clear()