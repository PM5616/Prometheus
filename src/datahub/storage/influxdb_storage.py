"""InfluxDB Storage Implementation

InfluxDB存储实现，专门用于存储时序数据。

功能特性：
- 高性能时序数据存储
- 自动数据压缩和保留策略
- 灵活的查询语言
- 批量写入优化
- 数据聚合和降采样
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
from influxdb_client.client.exceptions import InfluxDBError

from .base import BaseStorage
from src.common.models.market import Ticker, Kline, OrderBook
from src.common.models.trading import Symbol, Order, Trade
from src.common.exceptions.data import (
    DataConnectionError, DataNotFoundError, DataValidationError
)
from src.common.utils.datetime_utils import datetime_to_timestamp, timestamp_to_datetime


class InfluxDBStorage(BaseStorage):
    """InfluxDB存储实现
    
    专门用于存储时序数据，如行情数据、K线数据、订单簿数据等。
    提供高性能的时序数据读写和查询功能。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化InfluxDB存储
        
        Args:
            config: 配置参数
                - host: InfluxDB主机地址
                - port: InfluxDB端口
                - token: 访问令牌
                - org: 组织名称
                - bucket: 存储桶名称
                - timeout: 连接超时时间
                - batch_size: 批量写入大小
        """
        super().__init__("influxdb", config)
        
        # InfluxDB特定配置
        self.token = config.get('token')
        self.org = config.get('org', 'prometheus')
        self.bucket = config.get('bucket', 'market_data')
        self.timeout = config.get('timeout', 30000)  # 30秒
        self.batch_size = config.get('batch_size', 1000)
        
        # 客户端和API对象
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.query_api = None
        self.delete_api = None
        
        # 数据保留策略
        self.retention_policies = config.get('retention_policies', {
            'ticker': '7d',      # 行情数据保留7天
            'kline_1m': '30d',   # 1分钟K线保留30天
            'kline_5m': '90d',   # 5分钟K线保留90天
            'kline_1h': '1y',    # 1小时K线保留1年
            'kline_1d': '5y',    # 日K线保留5年
            'order_book': '1d',  # 订单簿保留1天
            'orders': '1y',      # 订单保留1年
            'trades': '1y'       # 成交保留1年
        })
        
        # 构建连接URL
        self.url = f"http://{self.host}:{self.port or 8086}"
        
        self.logger.info(f"InfluxDB存储初始化完成: {self.url}")
    
    async def connect(self) -> bool:
        """
        建立InfluxDB连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.logger.info(f"正在连接InfluxDB: {self.url}")
            
            # 创建InfluxDB客户端
            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                timeout=self.timeout
            )
            
            # 创建API对象
            self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)
            self.query_api = self.client.query_api()
            self.delete_api = self.client.delete_api()
            
            # 测试连接
            health = self.client.health()
            if health.status != "pass":
                raise DataConnectionError(f"InfluxDB健康检查失败: {health.message}")
            
            self.is_connected = True
            self.stats['connection_time'] = datetime.now()
            
            self.logger.info("InfluxDB连接成功")
            return True
            
        except Exception as e:
            self.logger.error(f"InfluxDB连接失败: {e}")
            self.is_connected = False
            self._update_stats(error=True)
            return False
    
    async def disconnect(self) -> bool:
        """
        断开InfluxDB连接
        
        Returns:
            bool: 断开是否成功
        """
        try:
            self.logger.info("正在断开InfluxDB连接...")
            
            if self.write_api:
                self.write_api.close()
            
            if self.client:
                self.client.close()
            
            self.is_connected = False
            self.client = None
            self.write_api = None
            self.query_api = None
            self.delete_api = None
            
            self.logger.info("InfluxDB连接已断开")
            return True
            
        except Exception as e:
            self.logger.error(f"断开InfluxDB连接失败: {e}")
            return False
    
    async def health_check(self) -> bool:
        """
        InfluxDB健康检查
        
        Returns:
            bool: 存储是否健康
        """
        try:
            if not self.client:
                return False
            
            health = self.client.health()
            return health.status == "pass"
            
        except Exception as e:
            self.logger.error(f"InfluxDB健康检查失败: {e}")
            return False
    
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
            point = Point("ticker") \
                .tag("symbol", ticker.symbol) \
                .field("price", float(ticker.price)) \
                .field("bid_price", float(ticker.bid_price)) \
                .field("ask_price", float(ticker.ask_price)) \
                .field("volume", float(ticker.volume)) \
                .field("quote_volume", float(ticker.quote_volume)) \
                .field("open_price", float(ticker.open_price)) \
                .field("high_price", float(ticker.high_price)) \
                .field("low_price", float(ticker.low_price)) \
                .field("price_change", float(ticker.price_change)) \
                .field("price_change_percent", float(ticker.price_change_percent)) \
                .time(ticker.timestamp)
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            self._update_stats(write=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存行情数据失败: {e}")
            self._update_stats(error=True)
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
            points = []
            for ticker in tickers:
                point = Point("ticker") \
                    .tag("symbol", ticker.symbol) \
                    .field("price", float(ticker.price)) \
                    .field("bid_price", float(ticker.bid_price)) \
                    .field("ask_price", float(ticker.ask_price)) \
                    .field("volume", float(ticker.volume)) \
                    .field("quote_volume", float(ticker.quote_volume)) \
                    .field("open_price", float(ticker.open_price)) \
                    .field("high_price", float(ticker.high_price)) \
                    .field("low_price", float(ticker.low_price)) \
                    .field("price_change", float(ticker.price_change)) \
                    .field("price_change_percent", float(ticker.price_change_percent)) \
                    .time(ticker.timestamp)
                points.append(point)
            
            # 批量写入
            for i in range(0, len(points), self.batch_size):
                batch = points[i:i + self.batch_size]
                self.write_api.write(bucket=self.bucket, org=self.org, record=batch)
            
            self._update_stats(write=True)
            self.logger.info(f"批量保存 {len(tickers)} 条行情数据")
            
            return True
            
        except Exception as e:
            self.logger.error(f"批量保存行情数据失败: {e}")
            self._update_stats(error=True)
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
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: -1h)
                |> filter(fn: (r) => r._measurement == "ticker")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> last()
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = self.query_api.query(query, org=self.org)
            
            if not result or not result[0].records:
                return None
            
            record = result[0].records[0]
            values = record.values
            
            ticker = Ticker(
                symbol=symbol,
                price=Decimal(str(values.get('price', 0))),
                bid_price=Decimal(str(values.get('bid_price', 0))),
                ask_price=Decimal(str(values.get('ask_price', 0))),
                volume=Decimal(str(values.get('volume', 0))),
                quote_volume=Decimal(str(values.get('quote_volume', 0))),
                open_price=Decimal(str(values.get('open_price', 0))),
                high_price=Decimal(str(values.get('high_price', 0))),
                low_price=Decimal(str(values.get('low_price', 0))),
                price_change=Decimal(str(values.get('price_change', 0))),
                price_change_percent=Decimal(str(values.get('price_change_percent', 0))),
                timestamp=values.get('_time')
            )
            
            self._update_stats(read=True)
            return ticker
            
        except Exception as e:
            self.logger.error(f"获取最新行情数据失败 {symbol}: {e}")
            self._update_stats(error=True)
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
            # 构建时间范围
            if end_time is None:
                end_time = datetime.now()
            
            start_str = start_time.isoformat() + "Z"
            end_str = end_time.isoformat() + "Z"
            
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: {start_str}, stop: {end_str})
                |> filter(fn: (r) => r._measurement == "ticker")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
            '''
            
            if limit:
                query += f" |> limit(n: {limit})"
            
            result = self.query_api.query(query, org=self.org)
            
            tickers = []
            for table in result:
                for record in table.records:
                    values = record.values
                    ticker = Ticker(
                        symbol=symbol,
                        price=Decimal(str(values.get('price', 0))),
                        bid_price=Decimal(str(values.get('bid_price', 0))),
                        ask_price=Decimal(str(values.get('ask_price', 0))),
                        volume=Decimal(str(values.get('volume', 0))),
                        quote_volume=Decimal(str(values.get('quote_volume', 0))),
                        open_price=Decimal(str(values.get('open_price', 0))),
                        high_price=Decimal(str(values.get('high_price', 0))),
                        low_price=Decimal(str(values.get('low_price', 0))),
                        price_change=Decimal(str(values.get('price_change', 0))),
                        price_change_percent=Decimal(str(values.get('price_change_percent', 0))),
                        timestamp=values.get('_time')
                    )
                    tickers.append(ticker)
            
            self._update_stats(read=True)
            self.logger.info(f"获取到 {len(tickers)} 条历史行情数据: {symbol}")
            
            return tickers
            
        except Exception as e:
            self.logger.error(f"获取历史行情数据失败 {symbol}: {e}")
            self._update_stats(error=True)
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
            point = Point("kline") \
                .tag("symbol", kline.symbol) \
                .tag("interval", kline.interval) \
                .field("open_price", float(kline.open_price)) \
                .field("high_price", float(kline.high_price)) \
                .field("low_price", float(kline.low_price)) \
                .field("close_price", float(kline.close_price)) \
                .field("volume", float(kline.volume)) \
                .field("quote_volume", float(kline.quote_volume)) \
                .field("trade_count", kline.trade_count) \
                .field("is_closed", kline.is_closed) \
                .time(kline.open_time)
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            self._update_stats(write=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存K线数据失败: {e}")
            self._update_stats(error=True)
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
            points = []
            for kline in klines:
                point = Point("kline") \
                    .tag("symbol", kline.symbol) \
                    .tag("interval", kline.interval) \
                    .field("open_price", float(kline.open_price)) \
                    .field("high_price", float(kline.high_price)) \
                    .field("low_price", float(kline.low_price)) \
                    .field("close_price", float(kline.close_price)) \
                    .field("volume", float(kline.volume)) \
                    .field("quote_volume", float(kline.quote_volume)) \
                    .field("trade_count", kline.trade_count) \
                    .field("is_closed", kline.is_closed) \
                    .time(kline.open_time)
                points.append(point)
            
            # 批量写入
            for i in range(0, len(points), self.batch_size):
                batch = points[i:i + self.batch_size]
                self.write_api.write(bucket=self.bucket, org=self.org, record=batch)
            
            self._update_stats(write=True)
            self.logger.info(f"批量保存 {len(klines)} 条K线数据")
            
            return True
            
        except Exception as e:
            self.logger.error(f"批量保存K线数据失败: {e}")
            self._update_stats(error=True)
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
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: -1d)
                |> filter(fn: (r) => r._measurement == "kline")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.interval == "{interval}")
                |> last()
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = self.query_api.query(query, org=self.org)
            
            if not result or not result[0].records:
                return None
            
            record = result[0].records[0]
            values = record.values
            
            kline = Kline(
                symbol=symbol,
                interval=interval,
                open_time=values.get('_time'),
                close_time=values.get('_time'),  # InfluxDB中使用相同时间
                open_price=Decimal(str(values.get('open_price', 0))),
                high_price=Decimal(str(values.get('high_price', 0))),
                low_price=Decimal(str(values.get('low_price', 0))),
                close_price=Decimal(str(values.get('close_price', 0))),
                volume=Decimal(str(values.get('volume', 0))),
                quote_volume=Decimal(str(values.get('quote_volume', 0))),
                trade_count=int(values.get('trade_count', 0)),
                is_closed=bool(values.get('is_closed', True))
            )
            
            self._update_stats(read=True)
            return kline
            
        except Exception as e:
            self.logger.error(f"获取最新K线数据失败 {symbol} {interval}: {e}")
            self._update_stats(error=True)
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
            # 构建时间范围
            if end_time is None:
                end_time = datetime.now()
            
            start_str = start_time.isoformat() + "Z"
            end_str = end_time.isoformat() + "Z"
            
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: {start_str}, stop: {end_str})
                |> filter(fn: (r) => r._measurement == "kline")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.interval == "{interval}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
            '''
            
            if limit:
                query += f" |> limit(n: {limit})"
            
            result = self.query_api.query(query, org=self.org)
            
            klines = []
            for table in result:
                for record in table.records:
                    values = record.values
                    kline = Kline(
                        symbol=symbol,
                        interval=interval,
                        open_time=values.get('_time'),
                        close_time=values.get('_time'),
                        open_price=Decimal(str(values.get('open_price', 0))),
                        high_price=Decimal(str(values.get('high_price', 0))),
                        low_price=Decimal(str(values.get('low_price', 0))),
                        close_price=Decimal(str(values.get('close_price', 0))),
                        volume=Decimal(str(values.get('volume', 0))),
                        quote_volume=Decimal(str(values.get('quote_volume', 0))),
                        trade_count=int(values.get('trade_count', 0)),
                        is_closed=bool(values.get('is_closed', True))
                    )
                    klines.append(kline)
            
            self._update_stats(read=True)
            self.logger.info(f"获取到 {len(klines)} 条K线数据: {symbol} {interval}")
            
            return klines
            
        except Exception as e:
            self.logger.error(f"获取K线数据失败 {symbol} {interval}: {e}")
            self._update_stats(error=True)
            return []
    
    # ==================== 其他数据操作（简化实现） ====================
    
    async def save_order_book(self, order_book: OrderBook) -> bool:
        """保存订单簿数据"""
        # 简化实现，实际应用中需要更复杂的结构
        return True
    
    async def get_latest_order_book(self, symbol: str) -> Optional[OrderBook]:
        """获取最新订单簿数据"""
        return None
    
    async def save_order(self, order: Order) -> bool:
        """保存订单数据"""
        return True
    
    async def save_trade(self, trade: Trade) -> bool:
        """保存成交数据"""
        return True
    
    async def get_orders(self, symbol: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: Optional[int] = None) -> List[Order]:
        """获取订单数据"""
        return []
    
    async def get_trades(self, symbol: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: Optional[int] = None) -> List[Trade]:
        """获取成交数据"""
        return []
    
    async def save_data(self, table: str, data: Dict[str, Any]) -> bool:
        """保存通用数据"""
        return True
    
    async def get_data(self, table: str, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取通用数据"""
        return []
    
    async def update_data(self, table: str, filters: Dict[str, Any], updates: Dict[str, Any]) -> bool:
        """更新数据"""
        return True
    
    async def delete_data(self, table: str, filters: Dict[str, Any]) -> bool:
        """删除数据"""
        return True
    
    async def create_table(self, table: str, schema: Dict[str, Any]) -> bool:
        """创建表"""
        return True
    
    async def drop_table(self, table: str) -> bool:
        """删除表"""
        return True
    
    async def list_tables(self) -> List[str]:
        """列出所有表"""
        return []
    
    # ==================== InfluxDB特有功能 ====================
    
    async def create_bucket(self, bucket_name: str, retention_period: str = "30d") -> bool:
        """
        创建存储桶
        
        Args:
            bucket_name: 存储桶名称
            retention_period: 数据保留期
            
        Returns:
            bool: 创建是否成功
        """
        try:
            buckets_api = self.client.buckets_api()
            
            # 检查存储桶是否已存在
            existing_buckets = buckets_api.find_buckets()
            for bucket in existing_buckets.buckets:
                if bucket.name == bucket_name:
                    self.logger.info(f"存储桶已存在: {bucket_name}")
                    return True
            
            # 创建新存储桶
            bucket = buckets_api.create_bucket(
                bucket_name=bucket_name,
                retention_rules=[{"type": "expire", "everySeconds": self._parse_retention_period(retention_period)}],
                org=self.org
            )
            
            self.logger.info(f"存储桶创建成功: {bucket_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建存储桶失败 {bucket_name}: {e}")
            return False
    
    def _parse_retention_period(self, period: str) -> int:
        """
        解析保留期字符串为秒数
        
        Args:
            period: 保留期字符串 (如 "7d", "30d", "1y")
            
        Returns:
            int: 秒数
        """
        unit = period[-1].lower()
        value = int(period[:-1])
        
        if unit == 'd':
            return value * 24 * 60 * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'm':
            return value * 60
        elif unit == 'y':
            return value * 365 * 24 * 60 * 60
        else:
            return value  # 默认为秒
    
    async def aggregate_klines(
        self, 
        symbol: str, 
        from_interval: str, 
        to_interval: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> List[Kline]:
        """
        聚合K线数据（如从1分钟聚合到5分钟）
        
        Args:
            symbol: 交易对符号
            from_interval: 源时间间隔
            to_interval: 目标时间间隔
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[Kline]: 聚合后的K线数据
        """
        try:
            if end_time is None:
                end_time = datetime.now()
            
            # 计算聚合窗口
            window = self._get_aggregation_window(to_interval)
            
            start_str = start_time.isoformat() + "Z"
            end_str = end_time.isoformat() + "Z"
            
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: {start_str}, stop: {end_str})
                |> filter(fn: (r) => r._measurement == "kline")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.interval == "{from_interval}")
                |> aggregateWindow(every: {window}, fn: mean, createEmpty: false)
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
            '''
            
            result = self.query_api.query(query, org=self.org)
            
            klines = []
            for table in result:
                for record in table.records:
                    values = record.values
                    kline = Kline(
                        symbol=symbol,
                        interval=to_interval,
                        open_time=values.get('_time'),
                        close_time=values.get('_time'),
                        open_price=Decimal(str(values.get('open_price', 0))),
                        high_price=Decimal(str(values.get('high_price', 0))),
                        low_price=Decimal(str(values.get('low_price', 0))),
                        close_price=Decimal(str(values.get('close_price', 0))),
                        volume=Decimal(str(values.get('volume', 0))),
                        quote_volume=Decimal(str(values.get('quote_volume', 0))),
                        trade_count=int(values.get('trade_count', 0)),
                        is_closed=True
                    )
                    klines.append(kline)
            
            self.logger.info(f"聚合K线数据完成: {symbol} {from_interval} -> {to_interval}, {len(klines)}条")
            return klines
            
        except Exception as e:
            self.logger.error(f"聚合K线数据失败: {e}")
            return []
    
    def _get_aggregation_window(self, interval: str) -> str:
        """
        获取聚合窗口字符串
        
        Args:
            interval: 时间间隔
            
        Returns:
            str: InfluxDB聚合窗口字符串
        """
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w'
        }
        
        return interval_map.get(interval, '1m')