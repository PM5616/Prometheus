"""Yahoo Finance Data Provider

Yahoo Finance数据提供商实现，提供股票、指数、外汇等金融数据。

功能特性：
- 股票历史数据
- 实时行情数据
- 财务数据
- 经济指标
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import yfinance as yf

from .base import BaseDataProvider, DataType, SubscriptionStatus
from src.common.models.market import Ticker, Kline, OrderBook
from src.common.models.trading import Symbol
from src.common.exceptions.data import (
    DataConnectionError, DataNotFoundError, DataValidationError
)
from src.common.utils.datetime_utils import datetime_to_timestamp


class YahooProvider(BaseDataProvider):
    """Yahoo Finance数据提供商
    
    提供Yahoo Finance的股票、指数、外汇等金融数据。
    主要用于获取历史数据和基本面数据。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Yahoo Finance数据提供商
        
        Args:
            config: 配置参数
        """
        super().__init__("yahoo", config)
        
        # 配置参数
        self.session_timeout = config.get('session_timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1)
        
        # HTTP会话
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 支持的时间间隔映射
        self.interval_mapping = {
            '1m': '1m',
            '2m': '2m', 
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '60m': '1h',
            '90m': '90m',
            '1h': '1h',
            '1d': '1d',
            '5d': '5d',
            '1wk': '1wk',
            '1mo': '1mo',
            '3mo': '3mo'
        }
        
        self.logger.info("Yahoo Finance数据提供商初始化完成")
    
    async def connect(self) -> bool:
        """
        建立连接（创建HTTP会话）
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.logger.info("正在创建Yahoo Finance HTTP会话...")
            
            # 创建HTTP会话
            timeout = aiohttp.ClientTimeout(total=self.session_timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            self.is_connected = True
            self.last_heartbeat = datetime.now()
            self.stats['connection_time'] = datetime.now()
            
            self.logger.info("Yahoo Finance HTTP会话创建成功")
            return True
            
        except Exception as e:
            self.logger.error(f"创建Yahoo Finance HTTP会话失败: {e}")
            self.is_connected = False
            self._update_stats(error=True)
            return False
    
    async def disconnect(self) -> bool:
        """
        断开连接（关闭HTTP会话）
        
        Returns:
            bool: 断开是否成功
        """
        try:
            self.logger.info("正在关闭Yahoo Finance HTTP会话...")
            
            if self.session and not self.session.closed:
                await self.session.close()
            
            self.is_connected = False
            self.session = None
            
            # 重置订阅状态
            for key in self.subscriptions:
                self.subscriptions[key] = SubscriptionStatus.STOPPED
            
            self.logger.info("Yahoo Finance HTTP会话已关闭")
            return True
            
        except Exception as e:
            self.logger.error(f"关闭Yahoo Finance HTTP会话失败: {e}")
            return False
    
    async def subscribe_ticker(self, symbol: str, callback: callable) -> bool:
        """
        订阅行情数据（Yahoo Finance不支持实时订阅，使用轮询模式）
        
        Args:
            symbol: 交易对符号
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        try:
            subscription_key = self._generate_subscription_key(DataType.TICKER, symbol)
            
            # 添加回调函数
            self.add_subscription_callback(subscription_key, callback)
            
            # 更新订阅状态
            self.subscriptions[subscription_key] = SubscriptionStatus.ACTIVE
            
            # 启动轮询任务
            asyncio.create_task(self._ticker_polling_task(symbol, callback))
            
            self.logger.info(f"订阅Yahoo Finance行情数据: {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"订阅Yahoo Finance行情数据失败 {symbol}: {e}")
            return False
    
    async def subscribe_kline(self, symbol: str, interval: str, callback: callable) -> bool:
        """
        订阅K线数据（Yahoo Finance不支持实时订阅）
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        self.logger.warning("Yahoo Finance不支持实时K线数据订阅")
        return False
    
    async def subscribe_order_book(self, symbol: str, depth: int, callback: callable) -> bool:
        """
        订阅订单簿数据（Yahoo Finance不支持）
        
        Args:
            symbol: 交易对符号
            depth: 订单簿深度
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        self.logger.warning("Yahoo Finance不支持订单簿数据")
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
            # 映射时间间隔
            yf_interval = self.interval_mapping.get(interval)
            if not yf_interval:
                raise DataValidationError(f"不支持的时间间隔: {interval}")
            
            # 获取数据
            ticker = yf.Ticker(symbol)
            
            # 设置结束时间
            if end_time is None:
                end_time = datetime.now()
            
            # 获取历史数据
            hist = ticker.history(
                start=start_time,
                end=end_time,
                interval=yf_interval,
                auto_adjust=True,
                prepost=True
            )
            
            if hist.empty:
                self.logger.warning(f"未找到历史数据: {symbol}")
                return []
            
            # 转换为Kline对象
            klines = []
            for index, row in hist.iterrows():
                kline = Kline(
                    symbol=symbol,
                    interval=interval,
                    open_time=index,
                    close_time=index + self._get_interval_timedelta(interval),
                    open_price=Decimal(str(row['Open'])),
                    high_price=Decimal(str(row['High'])),
                    low_price=Decimal(str(row['Low'])),
                    close_price=Decimal(str(row['Close'])),
                    volume=Decimal(str(row['Volume'])),
                    quote_volume=Decimal('0'),  # Yahoo Finance不提供成交额
                    trade_count=0,  # Yahoo Finance不提供成交笔数
                    is_closed=True
                )
                klines.append(kline)
            
            # 应用限制
            if limit and len(klines) > limit:
                klines = klines[-limit:]
            
            self.logger.info(f"获取到 {len(klines)} 条历史K线数据: {symbol} {interval}")
            return klines
            
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
            # 获取实时数据
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            # 构建Ticker对象
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                return None
            
            ticker_data = Ticker(
                symbol=symbol,
                price=Decimal(str(current_price)),
                bid_price=Decimal(str(info.get('bid', current_price))),
                ask_price=Decimal(str(info.get('ask', current_price))),
                volume=Decimal(str(info.get('volume', 0))),
                quote_volume=Decimal('0'),  # Yahoo Finance不提供
                open_price=Decimal(str(info.get('open', current_price))),
                high_price=Decimal(str(info.get('dayHigh', current_price))),
                low_price=Decimal(str(info.get('dayLow', current_price))),
                price_change=Decimal('0'),  # 需要计算
                price_change_percent=Decimal(str(info.get('regularMarketChangePercent', 0))),
                timestamp=datetime.now()
            )
            
            return ticker_data
            
        except Exception as e:
            self.logger.error(f"获取当前行情数据失败 {symbol}: {e}")
            return None
    
    async def get_order_book(self, symbol: str, depth: int = 20) -> Optional[OrderBook]:
        """
        获取当前订单簿数据（Yahoo Finance不支持）
        
        Args:
            symbol: 交易对符号
            depth: 订单簿深度
            
        Returns:
            Optional[OrderBook]: 订单簿数据
        """
        self.logger.warning("Yahoo Finance不支持订单簿数据")
        return None
    
    async def get_symbols(self) -> List[Symbol]:
        """
        获取支持的交易对列表
        
        Returns:
            List[Symbol]: 交易对列表
        """
        try:
            # Yahoo Finance支持的常见股票代码
            common_symbols = [
                'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
                'NVDA', 'META', 'NFLX', 'AMD', 'INTC',
                '^GSPC', '^DJI', '^IXIC',  # 指数
                'EURUSD=X', 'GBPUSD=X', 'USDJPY=X',  # 外汇
                'GC=F', 'SI=F', 'CL=F'  # 商品期货
            ]
            
            symbols = []
            for symbol_str in common_symbols:
                symbol = Symbol(
                    symbol=symbol_str,
                    base_asset=symbol_str.split('=')[0] if '=' in symbol_str else symbol_str,
                    quote_asset='USD',
                    status='TRADING',
                    min_qty=Decimal('0.01'),
                    max_qty=Decimal('1000000'),
                    step_size=Decimal('0.01'),
                    min_price=Decimal('0.01'),
                    max_price=Decimal('1000000'),
                    tick_size=Decimal('0.01')
                )
                symbols.append(symbol)
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"获取交易对列表失败: {e}")
            return []
    
    async def _ticker_polling_task(self, symbol: str, callback: callable):
        """
        行情数据轮询任务
        
        Args:
            symbol: 交易对符号
            callback: 回调函数
        """
        subscription_key = self._generate_subscription_key(DataType.TICKER, symbol)
        
        while (subscription_key in self.subscriptions and 
               self.subscriptions[subscription_key] == SubscriptionStatus.ACTIVE):
            try:
                # 获取行情数据
                ticker_data = await self.get_ticker(symbol)
                
                if ticker_data:
                    # 通知回调函数
                    await self._notify_callbacks(subscription_key, ticker_data)
                    self._update_stats(message_received=True, message_processed=True)
                
                # 等待下次轮询
                await asyncio.sleep(5)  # 5秒轮询一次
                
            except Exception as e:
                self.logger.error(f"行情数据轮询失败 {symbol}: {e}")
                self._update_stats(error=True)
                await asyncio.sleep(10)  # 错误时等待更长时间
    
    def _get_interval_timedelta(self, interval: str) -> timedelta:
        """
        获取时间间隔对应的timedelta
        
        Args:
            interval: 时间间隔字符串
            
        Returns:
            timedelta: 时间间隔
        """
        interval_map = {
            '1m': timedelta(minutes=1),
            '2m': timedelta(minutes=2),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '90m': timedelta(minutes=90),
            '1d': timedelta(days=1),
            '5d': timedelta(days=5),
            '1wk': timedelta(weeks=1),
            '1mo': timedelta(days=30),
            '3mo': timedelta(days=90)
        }
        
        return interval_map.get(interval, timedelta(minutes=1))
    
    async def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取公司基本信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            Optional[Dict[str, Any]]: 公司信息
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            # 提取关键信息
            company_info = {
                'symbol': symbol,
                'company_name': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'description': info.get('longBusinessSummary')
            }
            
            return company_info
            
        except Exception as e:
            self.logger.error(f"获取公司信息失败 {symbol}: {e}")
            return None
    
    async def get_financial_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取财务数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            Optional[Dict[str, Any]]: 财务数据
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # 获取财务报表
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            financial_data = {
                'symbol': symbol,
                'financials': financials.to_dict() if not financials.empty else {},
                'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                'cashflow': cashflow.to_dict() if not cashflow.empty else {}
            }
            
            return financial_data
            
        except Exception as e:
            self.logger.error(f"获取财务数据失败 {symbol}: {e}")
            return None