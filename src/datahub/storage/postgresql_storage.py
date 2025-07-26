"""PostgreSQL Storage Implementation

PostgreSQL存储实现，用于结构化数据存储。

功能特性：
- 关系型数据存储
- ACID事务支持
- 复杂查询支持
- 数据完整性约束
- 高并发处理
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging

try:
    import asyncpg
except ImportError:
    asyncpg = None

from .base import BaseStorage, StorageType
from src.common.models.market import Ticker, Kline, OrderBook
from src.common.models.trading import Symbol, Order, Trade
from src.common.exceptions.data import (
    DataConnectionError, DataNotFoundError, DataValidationError
)


class PostgreSQLStorage(BaseStorage):
    """PostgreSQL存储实现
    
    用于结构化数据的持久化存储。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化PostgreSQL存储
        
        Args:
            config: PostgreSQL配置
                - host: 数据库主机地址
                - port: 数据库端口
                - username: 用户名
                - password: 密码
                - database: 数据库名
                - min_size: 连接池最小连接数
                - max_size: 连接池最大连接数
                - command_timeout: 命令超时时间
        """
        super().__init__(StorageType.POSTGRESQL.value, config)
        
        if not asyncpg:
            raise ImportError("asyncpg package is required for PostgreSQLStorage")
        
        # PostgreSQL配置
        self.min_size = config.get('min_size', 1)
        self.max_size = config.get('max_size', 10)
        self.command_timeout = config.get('command_timeout', 60)
        
        # 连接池
        self.pool = None
        
        # 表名配置
        self.table_prefix = config.get('table_prefix', 'prometheus_')
        
        self.logger.info("PostgreSQL存储初始化完成")
    
    def _get_table_name(self, table_type: str) -> str:
        """
        获取表名
        
        Args:
            table_type: 表类型
            
        Returns:
            str: 完整表名
        """
        return f"{self.table_prefix}{table_type}"
    
    async def connect(self) -> bool:
        """
        建立PostgreSQL连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            start_time = datetime.now()
            
            # 创建连接池
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port or 5432,
                user=self.username,
                password=self.password,
                database=self.database,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=self.command_timeout
            )
            
            # 测试连接
            async with self.pool.acquire() as conn:
                await conn.execute('SELECT 1')
            
            # 初始化表结构
            await self._init_tables()
            
            self.is_connected = True
            self.stats['connection_time'] = datetime.now() - start_time
            
            self.logger.info(f"PostgreSQL连接成功: {self.host}:{self.port}/{self.database}")
            return True
            
        except Exception as e:
            self.logger.error(f"PostgreSQL连接失败: {e}")
            self.stats['total_errors'] += 1
            raise DataConnectionError(f"PostgreSQL连接失败: {e}")
    
    async def disconnect(self) -> bool:
        """
        断开PostgreSQL连接
        
        Returns:
            bool: 断开是否成功
        """
        try:
            if self.pool:
                await self.pool.close()
            
            self.is_connected = False
            self.logger.info("PostgreSQL连接已断开")
            return True
            
        except Exception as e:
            self.logger.error(f"PostgreSQL断开连接失败: {e}")
            return False
    
    async def health_check(self) -> bool:
        """
        PostgreSQL健康检查
        
        Returns:
            bool: 是否健康
        """
        try:
            if not self.is_connected or not self.pool:
                return False
            
            async with self.pool.acquire() as conn:
                result = await conn.fetchval('SELECT 1')
                return result == 1
            
        except Exception as e:
            self.logger.error(f"PostgreSQL健康检查失败: {e}")
            return False
    
    async def _init_tables(self):
        """
        初始化数据库表结构
        """
        try:
            async with self.pool.acquire() as conn:
                # 创建行情数据表
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._get_table_name('tickers')} (
                        id BIGSERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        price DECIMAL(20, 8) NOT NULL,
                        volume DECIMAL(20, 8) NOT NULL,
                        change_amount DECIMAL(20, 8),
                        change_percent DECIMAL(10, 4),
                        high DECIMAL(20, 8),
                        low DECIMAL(20, 8),
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # 创建K线数据表
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._get_table_name('klines')} (
                        id BIGSERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        interval VARCHAR(10) NOT NULL,
                        open_time TIMESTAMP WITH TIME ZONE NOT NULL,
                        close_time TIMESTAMP WITH TIME ZONE NOT NULL,
                        open_price DECIMAL(20, 8) NOT NULL,
                        high_price DECIMAL(20, 8) NOT NULL,
                        low_price DECIMAL(20, 8) NOT NULL,
                        close_price DECIMAL(20, 8) NOT NULL,
                        volume DECIMAL(20, 8) NOT NULL,
                        quote_volume DECIMAL(20, 8),
                        trades_count INTEGER,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        UNIQUE(symbol, interval, open_time)
                    )
                """)
                
                # 创建订单表
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._get_table_name('orders')} (
                        id BIGSERIAL PRIMARY KEY,
                        order_id VARCHAR(50) UNIQUE NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        type VARCHAR(20) NOT NULL,
                        quantity DECIMAL(20, 8) NOT NULL,
                        price DECIMAL(20, 8),
                        status VARCHAR(20) NOT NULL,
                        filled_quantity DECIMAL(20, 8) DEFAULT 0,
                        avg_price DECIMAL(20, 8),
                        commission DECIMAL(20, 8),
                        commission_asset VARCHAR(10),
                        created_time TIMESTAMP WITH TIME ZONE NOT NULL,
                        updated_time TIMESTAMP WITH TIME ZONE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # 创建交易记录表
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._get_table_name('trades')} (
                        id BIGSERIAL PRIMARY KEY,
                        trade_id VARCHAR(50) UNIQUE NOT NULL,
                        order_id VARCHAR(50) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        quantity DECIMAL(20, 8) NOT NULL,
                        price DECIMAL(20, 8) NOT NULL,
                        commission DECIMAL(20, 8),
                        commission_asset VARCHAR(10),
                        trade_time TIMESTAMP WITH TIME ZONE NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # 创建策略配置表
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._get_table_name('strategies')} (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) UNIQUE NOT NULL,
                        description TEXT,
                        strategy_type VARCHAR(50) NOT NULL,
                        parameters JSONB,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # 创建风控限制表
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._get_table_name('risk_limits')} (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        limit_type VARCHAR(50) NOT NULL,
                        symbol VARCHAR(20),
                        max_position DECIMAL(20, 8),
                        max_daily_loss DECIMAL(20, 8),
                        max_drawdown DECIMAL(10, 4),
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # 创建系统日志表
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._get_table_name('system_logs')} (
                        id BIGSERIAL PRIMARY KEY,
                        level VARCHAR(20) NOT NULL,
                        logger_name VARCHAR(100) NOT NULL,
                        message TEXT NOT NULL,
                        module VARCHAR(100),
                        function VARCHAR(100),
                        line_number INTEGER,
                        exception TEXT,
                        extra_data JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # 创建索引
                await self._create_indexes(conn)
                
                self.logger.info("数据库表结构初始化完成")
                
        except Exception as e:
            self.logger.error(f"初始化数据库表结构失败: {e}")
            raise
    
    async def _create_indexes(self, conn):
        """
        创建数据库索引
        
        Args:
            conn: 数据库连接
        """
        indexes = [
            # 行情数据索引
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}tickers_symbol_timestamp ON {self._get_table_name('tickers')} (symbol, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}tickers_timestamp ON {self._get_table_name('tickers')} (timestamp DESC)",
            
            # K线数据索引
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}klines_symbol_interval_time ON {self._get_table_name('klines')} (symbol, interval, open_time DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}klines_open_time ON {self._get_table_name('klines')} (open_time DESC)",
            
            # 订单索引
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}orders_symbol_status ON {self._get_table_name('orders')} (symbol, status)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}orders_created_time ON {self._get_table_name('orders')} (created_time DESC)",
            
            # 交易记录索引
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}trades_symbol_time ON {self._get_table_name('trades')} (symbol, trade_time DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}trades_order_id ON {self._get_table_name('trades')} (order_id)",
            
            # 系统日志索引
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}system_logs_level_time ON {self._get_table_name('system_logs')} (level, created_at DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}system_logs_logger_time ON {self._get_table_name('system_logs')} (logger_name, created_at DESC)"
        ]
        
        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
            except Exception as e:
                self.logger.warning(f"创建索引失败: {e}")
    
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
            async with self.pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self._get_table_name('tickers')} 
                    (symbol, price, volume, change_amount, change_percent, high, low, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    ticker.symbol,
                    float(ticker.price),
                    float(ticker.volume),
                    float(ticker.change) if ticker.change else None,
                    float(ticker.change_percent) if ticker.change_percent else None,
                    float(ticker.high) if ticker.high else None,
                    float(ticker.low) if ticker.low else None,
                    ticker.timestamp
                )
            
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
            async with self.pool.acquire() as conn:
                # 准备批量插入数据
                records = [
                    (
                        ticker.symbol,
                        float(ticker.price),
                        float(ticker.volume),
                        float(ticker.change) if ticker.change else None,
                        float(ticker.change_percent) if ticker.change_percent else None,
                        float(ticker.high) if ticker.high else None,
                        float(ticker.low) if ticker.low else None,
                        ticker.timestamp
                    )
                    for ticker in tickers
                ]
                
                # 批量插入
                await conn.executemany(
                    f"""
                    INSERT INTO {self._get_table_name('tickers')} 
                    (symbol, price, volume, change_amount, change_percent, high, low, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    records
                )
            
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
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT symbol, price, volume, change_amount, change_percent, 
                           high, low, timestamp
                    FROM {self._get_table_name('tickers')}
                    WHERE symbol = $1
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    symbol
                )
                
                if row:
                    ticker = Ticker(
                        symbol=row['symbol'],
                        price=row['price'],
                        volume=row['volume'],
                        change=row['change_amount'],
                        change_percent=row['change_percent'],
                        high=row['high'],
                        low=row['low'],
                        timestamp=row['timestamp']
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
            async with self.pool.acquire() as conn:
                # 构建查询条件
                conditions = ["symbol = $1", "timestamp >= $2"]
                params = [symbol, start_time]
                
                if end_time:
                    conditions.append("timestamp <= $3")
                    params.append(end_time)
                
                # 构建SQL
                sql = f"""
                    SELECT symbol, price, volume, change_amount, change_percent, 
                           high, low, timestamp
                    FROM {self._get_table_name('tickers')}
                    WHERE {' AND '.join(conditions)}
                    ORDER BY timestamp DESC
                """
                
                if limit:
                    sql += f" LIMIT {limit}"
                
                rows = await conn.fetch(sql, *params)
                
                tickers = [
                    Ticker(
                        symbol=row['symbol'],
                        price=row['price'],
                        volume=row['volume'],
                        change=row['change_amount'],
                        change_percent=row['change_percent'],
                        high=row['high'],
                        low=row['low'],
                        timestamp=row['timestamp']
                    )
                    for row in rows
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
            async with self.pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self._get_table_name('klines')} 
                    (symbol, interval, open_time, close_time, open_price, high_price, 
                     low_price, close_price, volume, quote_volume, trades_count)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (symbol, interval, open_time) 
                    DO UPDATE SET
                        close_time = EXCLUDED.close_time,
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        quote_volume = EXCLUDED.quote_volume,
                        trades_count = EXCLUDED.trades_count
                    """,
                    kline.symbol,
                    kline.interval,
                    kline.open_time,
                    kline.close_time,
                    float(kline.open),
                    float(kline.high),
                    float(kline.low),
                    float(kline.close),
                    float(kline.volume),
                    float(kline.quote_volume) if kline.quote_volume else None,
                    kline.trades_count
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
            async with self.pool.acquire() as conn:
                # 准备批量插入数据
                records = [
                    (
                        kline.symbol,
                        kline.interval,
                        kline.open_time,
                        kline.close_time,
                        float(kline.open),
                        float(kline.high),
                        float(kline.low),
                        float(kline.close),
                        float(kline.volume),
                        float(kline.quote_volume) if kline.quote_volume else None,
                        kline.trades_count
                    )
                    for kline in klines
                ]
                
                # 批量插入（使用ON CONFLICT处理重复数据）
                await conn.executemany(
                    f"""
                    INSERT INTO {self._get_table_name('klines')} 
                    (symbol, interval, open_time, close_time, open_price, high_price, 
                     low_price, close_price, volume, quote_volume, trades_count)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (symbol, interval, open_time) 
                    DO UPDATE SET
                        close_time = EXCLUDED.close_time,
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        quote_volume = EXCLUDED.quote_volume,
                        trades_count = EXCLUDED.trades_count
                    """,
                    records
                )
            
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
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT symbol, interval, open_time, close_time, open_price, 
                           high_price, low_price, close_price, volume, quote_volume, trades_count
                    FROM {self._get_table_name('klines')}
                    WHERE symbol = $1 AND interval = $2
                    ORDER BY open_time DESC
                    LIMIT 1
                    """,
                    symbol, interval
                )
                
                if row:
                    kline = Kline(
                        symbol=row['symbol'],
                        interval=row['interval'],
                        open_time=row['open_time'],
                        close_time=row['close_time'],
                        open=row['open_price'],
                        high=row['high_price'],
                        low=row['low_price'],
                        close=row['close_price'],
                        volume=row['volume'],
                        quote_volume=row['quote_volume'],
                        trades_count=row['trades_count']
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
            async with self.pool.acquire() as conn:
                # 构建查询条件
                conditions = ["symbol = $1", "interval = $2", "open_time >= $3"]
                params = [symbol, interval, start_time]
                
                if end_time:
                    conditions.append("open_time <= $4")
                    params.append(end_time)
                
                # 构建SQL
                sql = f"""
                    SELECT symbol, interval, open_time, close_time, open_price, 
                           high_price, low_price, close_price, volume, quote_volume, trades_count
                    FROM {self._get_table_name('klines')}
                    WHERE {' AND '.join(conditions)}
                    ORDER BY open_time DESC
                """
                
                if limit:
                    sql += f" LIMIT {limit}"
                
                rows = await conn.fetch(sql, *params)
                
                klines = [
                    Kline(
                        symbol=row['symbol'],
                        interval=row['interval'],
                        open_time=row['open_time'],
                        close_time=row['close_time'],
                        open=row['open_price'],
                        high=row['high_price'],
                        low=row['low_price'],
                        close=row['close_price'],
                        volume=row['volume'],
                        quote_volume=row['quote_volume'],
                        trades_count=row['trades_count']
                    )
                    for row in rows
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
            async with self.pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self._get_table_name('orders')} 
                    (order_id, symbol, side, type, quantity, price, status, 
                     filled_quantity, avg_price, commission, commission_asset, 
                     created_time, updated_time)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (order_id) 
                    DO UPDATE SET
                        status = EXCLUDED.status,
                        filled_quantity = EXCLUDED.filled_quantity,
                        avg_price = EXCLUDED.avg_price,
                        commission = EXCLUDED.commission,
                        commission_asset = EXCLUDED.commission_asset,
                        updated_time = EXCLUDED.updated_time
                    """,
                    order.order_id,
                    order.symbol,
                    order.side,
                    order.type,
                    float(order.quantity),
                    float(order.price) if order.price else None,
                    order.status,
                    float(order.filled_quantity) if order.filled_quantity else 0,
                    float(order.avg_price) if order.avg_price else None,
                    float(order.commission) if order.commission else None,
                    order.commission_asset,
                    order.created_time,
                    order.updated_time
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
            async with self.pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self._get_table_name('trades')} 
                    (trade_id, order_id, symbol, side, quantity, price, 
                     commission, commission_asset, trade_time)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (trade_id) DO NOTHING
                    """,
                    trade.trade_id,
                    trade.order_id,
                    trade.symbol,
                    trade.side,
                    float(trade.quantity),
                    float(trade.price),
                    float(trade.commission) if trade.commission else None,
                    trade.commission_asset,
                    trade.trade_time
                )
            
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
            async with self.pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self._get_table_name('strategies')} 
                    (name, description, strategy_type, parameters, is_active, updated_at)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    ON CONFLICT (name) 
                    DO UPDATE SET
                        description = EXCLUDED.description,
                        strategy_type = EXCLUDED.strategy_type,
                        parameters = EXCLUDED.parameters,
                        is_active = EXCLUDED.is_active,
                        updated_at = NOW()
                    """,
                    name,
                    config.get('description', ''),
                    config.get('strategy_type', ''),
                    json.dumps(config.get('parameters', {})),
                    config.get('is_active', True)
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
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT name, description, strategy_type, parameters, is_active
                    FROM {self._get_table_name('strategies')}
                    WHERE name = $1
                    """,
                    name
                )
                
                if row:
                    return {
                        'name': row['name'],
                        'description': row['description'],
                        'strategy_type': row['strategy_type'],
                        'parameters': json.loads(row['parameters']) if row['parameters'] else {},
                        'is_active': row['is_active']
                    }
                
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
            async with self.pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self._get_table_name('system_logs')} 
                    (level, logger_name, message, module, function, line_number, 
                     exception, extra_data)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    log_record.get('level'),
                    log_record.get('logger_name'),
                    log_record.get('message'),
                    log_record.get('module'),
                    log_record.get('function'),
                    log_record.get('line_number'),
                    log_record.get('exception'),
                    json.dumps(log_record.get('extra_data', {}))
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存系统日志失败: {e}")
            return False
    
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
                'min_size': self.min_size,
                'max_size': self.max_size,
                'command_timeout': self.command_timeout,
                'table_prefix': self.table_prefix
            }
        }
    
    async def get_table_stats(self) -> Dict[str, Any]:
        """
        获取表统计信息
        
        Returns:
            Dict[str, Any]: 表统计信息
        """
        try:
            async with self.pool.acquire() as conn:
                tables = ['tickers', 'klines', 'orders', 'trades', 'strategies', 'system_logs']
                stats = {}
                
                for table in tables:
                    table_name = self._get_table_name(table)
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                    stats[table] = {'count': count}
                
                return stats
            
        except Exception as e:
            self.logger.error(f"获取表统计信息失败: {e}")
            return {}