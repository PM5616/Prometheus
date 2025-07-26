"""File Data Provider

文件数据提供商实现，支持从本地文件读取历史数据。

支持格式：
- CSV文件
- Excel文件
- JSON文件
- Parquet文件
"""

import asyncio
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import json

from .base import BaseDataProvider, DataType, SubscriptionStatus
from src.common.models.market import Ticker, Kline, OrderBook
from src.common.models.trading import Symbol
from src.common.exceptions.data import (
    DataConnectionError, DataNotFoundError, DataValidationError
)
from src.common.utils.datetime_utils import parse_datetime


class FileProvider(BaseDataProvider):
    """文件数据提供商
    
    支持从本地文件读取历史数据，包括CSV、Excel、JSON、Parquet等格式。
    主要用于回测和历史数据分析。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文件数据提供商
        
        Args:
            config: 配置参数
                - data_directory: 数据文件目录
                - file_format: 文件格式 (csv, excel, json, parquet)
                - encoding: 文件编码 (默认utf-8)
                - date_column: 日期列名
                - symbol_column: 交易对列名
        """
        super().__init__("file", config)
        
        # 配置参数
        self.data_directory = Path(config.get('data_directory', './data'))
        self.file_format = config.get('file_format', 'csv').lower()
        self.encoding = config.get('encoding', 'utf-8')
        self.date_column = config.get('date_column', 'timestamp')
        self.symbol_column = config.get('symbol_column', 'symbol')
        
        # 列名映射
        self.column_mapping = config.get('column_mapping', {
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'price': 'price',
            'bid': 'bid',
            'ask': 'ask'
        })
        
        # 数据缓存
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.cache_enabled = config.get('cache_enabled', True)
        
        # 支持的文件格式
        self.supported_formats = ['csv', 'excel', 'json', 'parquet']
        
        if self.file_format not in self.supported_formats:
            raise DataValidationError(f"不支持的文件格式: {self.file_format}")
        
        self.logger.info(f"文件数据提供商初始化完成，数据目录: {self.data_directory}")
    
    async def connect(self) -> bool:
        """
        建立连接（检查数据目录）
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.logger.info(f"正在检查数据目录: {self.data_directory}")
            
            # 检查数据目录是否存在
            if not self.data_directory.exists():
                self.logger.warning(f"数据目录不存在，正在创建: {self.data_directory}")
                self.data_directory.mkdir(parents=True, exist_ok=True)
            
            # 检查目录权限
            if not self.data_directory.is_dir():
                raise DataConnectionError(f"数据路径不是目录: {self.data_directory}")
            
            self.is_connected = True
            self.last_heartbeat = datetime.now()
            self.stats['connection_time'] = datetime.now()
            
            self.logger.info("文件数据提供商连接成功")
            return True
            
        except Exception as e:
            self.logger.error(f"文件数据提供商连接失败: {e}")
            self.is_connected = False
            self._update_stats(error=True)
            return False
    
    async def disconnect(self) -> bool:
        """
        断开连接（清理缓存）
        
        Returns:
            bool: 断开是否成功
        """
        try:
            self.logger.info("正在断开文件数据提供商连接...")
            
            # 清理缓存
            self.data_cache.clear()
            
            self.is_connected = False
            
            # 重置订阅状态
            for key in self.subscriptions:
                self.subscriptions[key] = SubscriptionStatus.STOPPED
            
            self.logger.info("文件数据提供商连接已断开")
            return True
            
        except Exception as e:
            self.logger.error(f"断开文件数据提供商连接失败: {e}")
            return False
    
    async def subscribe_ticker(self, symbol: str, callback: callable) -> bool:
        """
        订阅行情数据（文件模式不支持实时订阅）
        
        Args:
            symbol: 交易对符号
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        self.logger.warning("文件数据提供商不支持实时行情订阅")
        return False
    
    async def subscribe_kline(self, symbol: str, interval: str, callback: callable) -> bool:
        """
        订阅K线数据（文件模式不支持实时订阅）
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        self.logger.warning("文件数据提供商不支持实时K线订阅")
        return False
    
    async def subscribe_order_book(self, symbol: str, depth: int, callback: callable) -> bool:
        """
        订阅订单簿数据（文件模式不支持）
        
        Args:
            symbol: 交易对符号
            depth: 订单簿深度
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        self.logger.warning("文件数据提供商不支持订单簿数据")
        return False
    
    async def unsubscribe(self, subscription_key: str) -> bool:
        """
        取消订阅
        
        Args:
            subscription_key: 订阅键
            
        Returns:
            bool: 取消订阅是否成功
        """
        # 文件模式不支持订阅，直接返回True
        return True
    
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
            # 加载数据
            df = await self._load_data(symbol, interval)
            
            if df is None or df.empty:
                self.logger.warning(f"未找到数据文件: {symbol} {interval}")
                return []
            
            # 过滤时间范围
            df = self._filter_by_time_range(df, start_time, end_time)
            
            # 应用限制
            if limit and len(df) > limit:
                df = df.tail(limit)
            
            # 转换为Kline对象
            klines = self._dataframe_to_klines(df, symbol, interval)
            
            self.logger.info(f"从文件加载 {len(klines)} 条K线数据: {symbol} {interval}")
            return klines
            
        except Exception as e:
            self.logger.error(f"获取历史K线数据失败 {symbol}: {e}")
            raise DataNotFoundError(f"获取历史K线数据失败: {e}")
    
    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """
        获取当前行情数据（从文件读取最新数据）
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[Ticker]: 行情数据
        """
        try:
            # 尝试从ticker文件加载
            df = await self._load_ticker_data(symbol)
            
            if df is None or df.empty:
                return None
            
            # 获取最新数据
            latest_row = df.iloc[-1]
            
            # 构建Ticker对象
            ticker_data = Ticker(
                symbol=symbol,
                price=Decimal(str(latest_row.get(self.column_mapping['price'], 0))),
                bid_price=Decimal(str(latest_row.get(self.column_mapping['bid'], 0))),
                ask_price=Decimal(str(latest_row.get(self.column_mapping['ask'], 0))),
                volume=Decimal(str(latest_row.get(self.column_mapping['volume'], 0))),
                quote_volume=Decimal('0'),
                open_price=Decimal(str(latest_row.get(self.column_mapping['open'], 0))),
                high_price=Decimal(str(latest_row.get(self.column_mapping['high'], 0))),
                low_price=Decimal(str(latest_row.get(self.column_mapping['low'], 0))),
                price_change=Decimal('0'),
                price_change_percent=Decimal('0'),
                timestamp=parse_datetime(latest_row[self.date_column])
            )
            
            return ticker_data
            
        except Exception as e:
            self.logger.error(f"获取当前行情数据失败 {symbol}: {e}")
            return None
    
    async def get_order_book(self, symbol: str, depth: int = 20) -> Optional[OrderBook]:
        """
        获取当前订单簿数据（文件模式不支持）
        
        Args:
            symbol: 交易对符号
            depth: 订单簿深度
            
        Returns:
            Optional[OrderBook]: 订单簿数据
        """
        self.logger.warning("文件数据提供商不支持订单簿数据")
        return None
    
    async def get_symbols(self) -> List[Symbol]:
        """
        获取支持的交易对列表（从文件目录扫描）
        
        Returns:
            List[Symbol]: 交易对列表
        """
        try:
            symbols = []
            
            # 扫描数据目录
            for file_path in self.data_directory.glob(f"*.{self.file_format}"):
                # 从文件名提取交易对
                symbol_name = file_path.stem
                
                # 创建Symbol对象
                symbol = Symbol(
                    symbol=symbol_name,
                    base_asset=symbol_name.split('_')[0] if '_' in symbol_name else symbol_name,
                    quote_asset=symbol_name.split('_')[1] if '_' in symbol_name else 'USDT',
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
    
    async def _load_data(self, symbol: str, interval: str = None) -> Optional[pd.DataFrame]:
        """
        加载数据文件
        
        Args:
            symbol: 交易对符号
            interval: K线间隔（可选）
            
        Returns:
            Optional[pd.DataFrame]: 数据DataFrame
        """
        try:
            # 构建文件名
            if interval:
                filename = f"{symbol}_{interval}.{self.file_format}"
            else:
                filename = f"{symbol}.{self.file_format}"
            
            file_path = self.data_directory / filename
            
            # 检查缓存
            cache_key = str(file_path)
            if self.cache_enabled and cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            # 检查文件是否存在
            if not file_path.exists():
                return None
            
            # 根据文件格式加载数据
            if self.file_format == 'csv':
                df = pd.read_csv(file_path, encoding=self.encoding)
            elif self.file_format == 'excel':
                df = pd.read_excel(file_path)
            elif self.file_format == 'json':
                df = pd.read_json(file_path)
            elif self.file_format == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                raise DataValidationError(f"不支持的文件格式: {self.file_format}")
            
            # 处理日期列
            if self.date_column in df.columns:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
                df = df.sort_values(self.date_column)
            
            # 缓存数据
            if self.cache_enabled:
                self.data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            self.logger.error(f"加载数据文件失败 {symbol}: {e}")
            return None
    
    async def _load_ticker_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        加载行情数据文件
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[pd.DataFrame]: 行情数据DataFrame
        """
        # 尝试加载ticker专用文件
        ticker_df = await self._load_data(f"{symbol}_ticker")
        
        if ticker_df is not None:
            return ticker_df
        
        # 如果没有专用文件，尝试从K线数据获取
        kline_df = await self._load_data(symbol, '1d')
        
        return kline_df
    
    def _filter_by_time_range(
        self, 
        df: pd.DataFrame, 
        start_time: datetime, 
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        按时间范围过滤数据
        
        Args:
            df: 数据DataFrame
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        if self.date_column not in df.columns:
            return df
        
        # 过滤开始时间
        df = df[df[self.date_column] >= start_time]
        
        # 过滤结束时间
        if end_time:
            df = df[df[self.date_column] <= end_time]
        
        return df
    
    def _dataframe_to_klines(self, df: pd.DataFrame, symbol: str, interval: str) -> List[Kline]:
        """
        将DataFrame转换为Kline对象列表
        
        Args:
            df: 数据DataFrame
            symbol: 交易对符号
            interval: K线间隔
            
        Returns:
            List[Kline]: K线数据列表
        """
        klines = []
        
        for _, row in df.iterrows():
            try:
                kline = Kline(
                    symbol=symbol,
                    interval=interval,
                    open_time=parse_datetime(row[self.date_column]),
                    close_time=parse_datetime(row[self.date_column]),  # 文件数据通常只有一个时间
                    open_price=Decimal(str(row[self.column_mapping['open']])),
                    high_price=Decimal(str(row[self.column_mapping['high']])),
                    low_price=Decimal(str(row[self.column_mapping['low']])),
                    close_price=Decimal(str(row[self.column_mapping['close']])),
                    volume=Decimal(str(row[self.column_mapping['volume']])),
                    quote_volume=Decimal('0'),  # 文件数据通常不包含成交额
                    trade_count=0,  # 文件数据通常不包含成交笔数
                    is_closed=True
                )
                klines.append(kline)
                
            except Exception as e:
                self.logger.warning(f"跳过无效数据行: {e}")
                continue
        
        return klines
    
    async def save_data(self, symbol: str, data: Union[List[Kline], List[Ticker]], interval: str = None):
        """
        保存数据到文件
        
        Args:
            symbol: 交易对符号
            data: 数据列表
            interval: K线间隔（可选）
        """
        try:
            if not data:
                return
            
            # 构建文件名
            if interval:
                filename = f"{symbol}_{interval}.{self.file_format}"
            else:
                filename = f"{symbol}.{self.file_format}"
            
            file_path = self.data_directory / filename
            
            # 转换数据为DataFrame
            if isinstance(data[0], Kline):
                df_data = []
                for kline in data:
                    df_data.append({
                        self.date_column: kline.open_time,
                        self.column_mapping['open']: float(kline.open_price),
                        self.column_mapping['high']: float(kline.high_price),
                        self.column_mapping['low']: float(kline.low_price),
                        self.column_mapping['close']: float(kline.close_price),
                        self.column_mapping['volume']: float(kline.volume)
                    })
            elif isinstance(data[0], Ticker):
                df_data = []
                for ticker in data:
                    df_data.append({
                        self.date_column: ticker.timestamp,
                        self.column_mapping['price']: float(ticker.price),
                        self.column_mapping['bid']: float(ticker.bid_price),
                        self.column_mapping['ask']: float(ticker.ask_price),
                        self.column_mapping['volume']: float(ticker.volume)
                    })
            else:
                raise DataValidationError("不支持的数据类型")
            
            df = pd.DataFrame(df_data)
            
            # 保存文件
            if self.file_format == 'csv':
                df.to_csv(file_path, index=False, encoding=self.encoding)
            elif self.file_format == 'excel':
                df.to_excel(file_path, index=False)
            elif self.file_format == 'json':
                df.to_json(file_path, orient='records', date_format='iso')
            elif self.file_format == 'parquet':
                df.to_parquet(file_path, index=False)
            
            # 更新缓存
            if self.cache_enabled:
                cache_key = str(file_path)
                self.data_cache[cache_key] = df
            
            self.logger.info(f"数据已保存到文件: {file_path}")
            
        except Exception as e:
            self.logger.error(f"保存数据到文件失败 {symbol}: {e}")
            raise
    
    def clear_cache(self):
        """
        清理数据缓存
        """
        self.data_cache.clear()
        self.logger.info("数据缓存已清理")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            Dict[str, Any]: 缓存信息
        """
        return {
            'cache_enabled': self.cache_enabled,
            'cached_files': len(self.data_cache),
            'cache_keys': list(self.data_cache.keys())
        }