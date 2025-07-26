"""File Storage Implementation

文件存储实现，用于数据备份和离线分析。

功能特性：
- 多种文件格式支持
- 数据压缩和归档
- 批量导入导出
- 数据备份恢复
- 离线数据分析
"""

import os
import json
import csv
import gzip
import pickle
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None

from .base import BaseStorage, StorageType
from src.common.models.market import Ticker, Kline, OrderBook
from src.common.models.trading import Symbol, Order, Trade
from src.common.exceptions.data import (
    DataConnectionError, DataNotFoundError, DataValidationError
)


class FileStorage(BaseStorage):
    """文件存储实现
    
    用于数据的文件化存储和备份。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文件存储
        
        Args:
            config: 文件存储配置
                - base_path: 基础存储路径
                - format: 文件格式 (json, csv, parquet, pickle)
                - compression: 压缩格式 (none, gzip, bz2)
                - partition_by: 分区字段 (date, symbol, type)
                - max_file_size: 最大文件大小(MB)
                - backup_enabled: 是否启用备份
                - backup_path: 备份路径
                - retention_days: 数据保留天数
        """
        super().__init__(StorageType.FILE.value, config)
        
        # 文件存储配置
        self.base_path = Path(config.get('base_path', './data'))
        self.format = config.get('format', 'json')
        self.compression = config.get('compression', 'none')
        self.partition_by = config.get('partition_by', 'date')
        self.max_file_size = config.get('max_file_size', 100) * 1024 * 1024  # MB to bytes
        self.backup_enabled = config.get('backup_enabled', False)
        self.backup_path = Path(config.get('backup_path', './backup')) if self.backup_enabled else None
        self.retention_days = config.get('retention_days', 30)
        
        # 支持的文件格式
        self.supported_formats = ['json', 'csv', 'parquet', 'pickle']
        self.supported_compressions = ['none', 'gzip', 'bz2']
        
        # 验证配置
        self._validate_config()
        
        # 创建目录结构
        self._create_directories()
        
        self.logger.info("文件存储初始化完成")
    
    def _validate_config(self):
        """
        验证配置参数
        """
        if self.format not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {self.format}")
        
        if self.compression not in self.supported_compressions:
            raise ValueError(f"不支持的压缩格式: {self.compression}")
        
        if self.format == 'parquet' and not pq:
            raise ImportError("parquet格式需要安装pyarrow包")
        
        if self.format == 'csv' and not pd:
            raise ImportError("csv格式需要安装pandas包")
    
    def _create_directories(self):
        """
        创建目录结构
        """
        # 创建基础目录
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 创建数据类型目录
        data_types = ['tickers', 'klines', 'orders', 'trades', 'strategies', 'logs']
        for data_type in data_types:
            (self.base_path / data_type).mkdir(exist_ok=True)
        
        # 创建备份目录
        if self.backup_enabled and self.backup_path:
            self.backup_path.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(
        self, 
        data_type: str, 
        symbol: Optional[str] = None, 
        date: Optional[datetime] = None,
        suffix: str = ''
    ) -> Path:
        """
        获取文件路径
        
        Args:
            data_type: 数据类型
            symbol: 交易对符号
            date: 日期
            suffix: 文件后缀
            
        Returns:
            Path: 文件路径
        """
        base_dir = self.base_path / data_type
        
        # 根据分区策略构建路径
        if self.partition_by == 'date' and date:
            date_str = date.strftime('%Y/%m/%d')
            file_dir = base_dir / date_str
        elif self.partition_by == 'symbol' and symbol:
            file_dir = base_dir / symbol
        elif self.partition_by == 'type':
            file_dir = base_dir
        else:
            file_dir = base_dir
        
        file_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建文件名
        filename_parts = [data_type]
        
        if symbol:
            filename_parts.append(symbol)
        
        if date:
            filename_parts.append(date.strftime('%Y%m%d'))
        
        if suffix:
            filename_parts.append(suffix)
        
        filename = '_'.join(filename_parts)
        
        # 添加文件扩展名
        if self.format == 'json':
            ext = '.json'
        elif self.format == 'csv':
            ext = '.csv'
        elif self.format == 'parquet':
            ext = '.parquet'
        elif self.format == 'pickle':
            ext = '.pkl'
        else:
            ext = '.dat'
        
        # 添加压缩扩展名
        if self.compression == 'gzip':
            ext += '.gz'
        elif self.compression == 'bz2':
            ext += '.bz2'
        
        return file_dir / (filename + ext)
    
    def _open_file(self, file_path: Path, mode: str = 'r'):
        """
        打开文件（支持压缩）
        
        Args:
            file_path: 文件路径
            mode: 打开模式
            
        Returns:
            文件对象
        """
        if self.compression == 'gzip':
            return gzip.open(file_path, mode + 't', encoding='utf-8')
        elif self.compression == 'bz2':
            import bz2
            return bz2.open(file_path, mode + 't', encoding='utf-8')
        else:
            return open(file_path, mode, encoding='utf-8')
    
    async def connect(self) -> bool:
        """
        建立文件存储连接（检查目录权限）
        
        Returns:
            bool: 连接是否成功
        """
        try:
            start_time = datetime.now()
            
            # 检查基础目录权限
            if not os.access(self.base_path, os.R_OK | os.W_OK):
                raise PermissionError(f"没有目录访问权限: {self.base_path}")
            
            # 检查备份目录权限
            if self.backup_enabled and self.backup_path:
                if not os.access(self.backup_path, os.R_OK | os.W_OK):
                    raise PermissionError(f"没有备份目录访问权限: {self.backup_path}")
            
            # 清理过期文件
            await self._cleanup_expired_files()
            
            self.is_connected = True
            self.stats['connection_time'] = datetime.now() - start_time
            
            self.logger.info(f"文件存储连接成功: {self.base_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"文件存储连接失败: {e}")
            self.stats['total_errors'] += 1
            raise DataConnectionError(f"文件存储连接失败: {e}")
    
    async def disconnect(self) -> bool:
        """
        断开文件存储连接
        
        Returns:
            bool: 断开是否成功
        """
        try:
            # 执行备份（如果启用）
            if self.backup_enabled:
                await self._backup_data()
            
            self.is_connected = False
            self.logger.info("文件存储连接已断开")
            return True
            
        except Exception as e:
            self.logger.error(f"文件存储断开连接失败: {e}")
            return False
    
    async def health_check(self) -> bool:
        """
        文件存储健康检查
        
        Returns:
            bool: 是否健康
        """
        try:
            if not self.is_connected:
                return False
            
            # 检查目录是否存在且可访问
            if not self.base_path.exists() or not os.access(self.base_path, os.R_OK | os.W_OK):
                return False
            
            # 测试写入
            test_file = self.base_path / '.health_check'
            test_file.write_text('health_check')
            test_file.unlink()
            
            return True
            
        except Exception as e:
            self.logger.error(f"文件存储健康检查失败: {e}")
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
            file_path = self._get_file_path(
                'tickers', 
                ticker.symbol, 
                ticker.timestamp
            )
            
            # 准备数据
            data = {
                'symbol': ticker.symbol,
                'price': float(ticker.price),
                'volume': float(ticker.volume),
                'change': float(ticker.change) if ticker.change else None,
                'change_percent': float(ticker.change_percent) if ticker.change_percent else None,
                'high': float(ticker.high) if ticker.high else None,
                'low': float(ticker.low) if ticker.low else None,
                'timestamp': ticker.timestamp.isoformat(),
                'created_at': datetime.now().isoformat()
            }
            
            # 保存数据
            await self._save_data(file_path, [data], append=True)
            
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
            # 按日期和符号分组
            grouped_data = {}
            
            for ticker in tickers:
                key = (ticker.symbol, ticker.timestamp.date())
                if key not in grouped_data:
                    grouped_data[key] = []
                
                data = {
                    'symbol': ticker.symbol,
                    'price': float(ticker.price),
                    'volume': float(ticker.volume),
                    'change': float(ticker.change) if ticker.change else None,
                    'change_percent': float(ticker.change_percent) if ticker.change_percent else None,
                    'high': float(ticker.high) if ticker.high else None,
                    'low': float(ticker.low) if ticker.low else None,
                    'timestamp': ticker.timestamp.isoformat(),
                    'created_at': datetime.now().isoformat()
                }
                
                grouped_data[key].append(data)
            
            # 分组保存
            for (symbol, date), data_list in grouped_data.items():
                file_path = self._get_file_path(
                    'tickers', 
                    symbol, 
                    datetime.combine(date, datetime.min.time())
                )
                
                await self._save_data(file_path, data_list, append=True)
            
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
            # 查找最近几天的文件
            for days_back in range(7):
                date = datetime.now() - timedelta(days=days_back)
                file_path = self._get_file_path('tickers', symbol, date)
                
                if file_path.exists():
                    data_list = await self._load_data(file_path)
                    
                    if data_list:
                        # 找到最新的记录
                        latest_data = max(data_list, key=lambda x: x['timestamp'])
                        
                        ticker = Ticker(
                            symbol=latest_data['symbol'],
                            price=latest_data['price'],
                            volume=latest_data['volume'],
                            change=latest_data.get('change'),
                            change_percent=latest_data.get('change_percent'),
                            high=latest_data.get('high'),
                            low=latest_data.get('low'),
                            timestamp=datetime.fromisoformat(latest_data['timestamp'])
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
            end_time = end_time or datetime.now()
            tickers = []
            
            # 遍历日期范围
            current_date = start_time.date()
            end_date = end_time.date()
            
            while current_date <= end_date:
                file_path = self._get_file_path(
                    'tickers', 
                    symbol, 
                    datetime.combine(current_date, datetime.min.time())
                )
                
                if file_path.exists():
                    data_list = await self._load_data(file_path)
                    
                    for data in data_list:
                        timestamp = datetime.fromisoformat(data['timestamp'])
                        
                        if start_time <= timestamp <= end_time:
                            ticker = Ticker(
                                symbol=data['symbol'],
                                price=data['price'],
                                volume=data['volume'],
                                change=data.get('change'),
                                change_percent=data.get('change_percent'),
                                high=data.get('high'),
                                low=data.get('low'),
                                timestamp=timestamp
                            )
                            
                            tickers.append(ticker)
                
                current_date += timedelta(days=1)
            
            # 按时间排序
            tickers.sort(key=lambda x: x.timestamp, reverse=True)
            
            # 应用限制
            if limit:
                tickers = tickers[:limit]
            
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
            file_path = self._get_file_path(
                'klines', 
                kline.symbol, 
                kline.open_time,
                kline.interval
            )
            
            # 准备数据
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
                'trades_count': kline.trades_count,
                'created_at': datetime.now().isoformat()
            }
            
            # 保存数据
            await self._save_data(file_path, [data], append=True)
            
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
            # 按符号、间隔和日期分组
            grouped_data = {}
            
            for kline in klines:
                key = (kline.symbol, kline.interval, kline.open_time.date())
                if key not in grouped_data:
                    grouped_data[key] = []
                
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
                    'trades_count': kline.trades_count,
                    'created_at': datetime.now().isoformat()
                }
                
                grouped_data[key].append(data)
            
            # 分组保存
            for (symbol, interval, date), data_list in grouped_data.items():
                file_path = self._get_file_path(
                    'klines', 
                    symbol, 
                    datetime.combine(date, datetime.min.time()),
                    interval
                )
                
                await self._save_data(file_path, data_list, append=True)
            
            self.stats['total_writes'] += len(klines)
            self.stats['last_operation_time'] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"批量保存K线数据失败: {e}")
            self.stats['total_errors'] += 1
            return False
    
    # ==================== 数据保存和加载 ====================
    
    async def _save_data(self, file_path: Path, data: List[Dict], append: bool = False):
        """
        保存数据到文件
        
        Args:
            file_path: 文件路径
            data: 数据列表
            append: 是否追加模式
        """
        try:
            # 检查文件大小
            if file_path.exists() and file_path.stat().st_size > self.max_file_size:
                # 文件过大，创建新文件
                timestamp = datetime.now().strftime('%H%M%S')
                file_path = file_path.with_name(f"{file_path.stem}_{timestamp}{file_path.suffix}")
                append = False
            
            if self.format == 'json':
                await self._save_json(file_path, data, append)
            elif self.format == 'csv':
                await self._save_csv(file_path, data, append)
            elif self.format == 'parquet':
                await self._save_parquet(file_path, data, append)
            elif self.format == 'pickle':
                await self._save_pickle(file_path, data, append)
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
            raise
    
    async def _load_data(self, file_path: Path) -> List[Dict]:
        """
        从文件加载数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Dict]: 数据列表
        """
        try:
            if not file_path.exists():
                return []
            
            if self.format == 'json':
                return await self._load_json(file_path)
            elif self.format == 'csv':
                return await self._load_csv(file_path)
            elif self.format == 'parquet':
                return await self._load_parquet(file_path)
            elif self.format == 'pickle':
                return await self._load_pickle(file_path)
            
            return []
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return []
    
    async def _save_json(self, file_path: Path, data: List[Dict], append: bool):
        """
        保存JSON格式数据
        """
        def _save():
            if append and file_path.exists():
                # 读取现有数据
                with self._open_file(file_path, 'r') as f:
                    existing_data = json.load(f)
                
                # 合并数据
                if isinstance(existing_data, list):
                    existing_data.extend(data)
                else:
                    existing_data = [existing_data] + data
                
                # 写入合并后的数据
                with self._open_file(file_path, 'w') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
            else:
                # 直接写入新数据
                with self._open_file(file_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        
        await asyncio.get_event_loop().run_in_executor(None, _save)
    
    async def _load_json(self, file_path: Path) -> List[Dict]:
        """
        加载JSON格式数据
        """
        def _load():
            with self._open_file(file_path, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        
        return await asyncio.get_event_loop().run_in_executor(None, _load)
    
    async def _save_csv(self, file_path: Path, data: List[Dict], append: bool):
        """
        保存CSV格式数据
        """
        if not pd:
            raise ImportError("pandas is required for CSV format")
        
        def _save():
            df = pd.DataFrame(data)
            
            if append and file_path.exists():
                # 追加模式
                if self.compression == 'gzip':
                    df.to_csv(file_path, mode='a', header=False, index=False, compression='gzip')
                else:
                    df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                # 覆盖模式
                if self.compression == 'gzip':
                    df.to_csv(file_path, index=False, compression='gzip')
                else:
                    df.to_csv(file_path, index=False)
        
        await asyncio.get_event_loop().run_in_executor(None, _save)
    
    async def _load_csv(self, file_path: Path) -> List[Dict]:
        """
        加载CSV格式数据
        """
        if not pd:
            raise ImportError("pandas is required for CSV format")
        
        def _load():
            if self.compression == 'gzip':
                df = pd.read_csv(file_path, compression='gzip')
            else:
                df = pd.read_csv(file_path)
            
            return df.to_dict('records')
        
        return await asyncio.get_event_loop().run_in_executor(None, _load)
    
    async def _save_parquet(self, file_path: Path, data: List[Dict], append: bool):
        """
        保存Parquet格式数据
        """
        if not pq:
            raise ImportError("pyarrow is required for Parquet format")
        
        def _save():
            df = pd.DataFrame(data)
            table = pa.Table.from_pandas(df)
            
            if append and file_path.exists():
                # 读取现有数据并合并
                existing_table = pq.read_table(file_path)
                combined_table = pa.concat_tables([existing_table, table])
                pq.write_table(combined_table, file_path)
            else:
                pq.write_table(table, file_path)
        
        await asyncio.get_event_loop().run_in_executor(None, _save)
    
    async def _load_parquet(self, file_path: Path) -> List[Dict]:
        """
        加载Parquet格式数据
        """
        if not pq:
            raise ImportError("pyarrow is required for Parquet format")
        
        def _load():
            table = pq.read_table(file_path)
            df = table.to_pandas()
            return df.to_dict('records')
        
        return await asyncio.get_event_loop().run_in_executor(None, _load)
    
    async def _save_pickle(self, file_path: Path, data: List[Dict], append: bool):
        """
        保存Pickle格式数据
        """
        def _save():
            if append and file_path.exists():
                # 读取现有数据
                with open(file_path, 'rb') as f:
                    existing_data = pickle.load(f)
                
                # 合并数据
                if isinstance(existing_data, list):
                    existing_data.extend(data)
                else:
                    existing_data = [existing_data] + data
                
                # 写入合并后的数据
                with open(file_path, 'wb') as f:
                    pickle.dump(existing_data, f)
            else:
                # 直接写入新数据
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
        
        await asyncio.get_event_loop().run_in_executor(None, _save)
    
    async def _load_pickle(self, file_path: Path) -> List[Dict]:
        """
        加载Pickle格式数据
        """
        def _load():
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data if isinstance(data, list) else [data]
        
        return await asyncio.get_event_loop().run_in_executor(None, _load)
    
    # ==================== 维护操作 ====================
    
    async def _cleanup_expired_files(self):
        """
        清理过期文件
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            for data_type_dir in self.base_path.iterdir():
                if data_type_dir.is_dir():
                    await self._cleanup_directory(data_type_dir, cutoff_date)
            
            self.logger.info(f"清理过期文件完成，保留{self.retention_days}天内的数据")
            
        except Exception as e:
            self.logger.error(f"清理过期文件失败: {e}")
    
    async def _cleanup_directory(self, directory: Path, cutoff_date: datetime):
        """
        清理目录中的过期文件
        
        Args:
            directory: 目录路径
            cutoff_date: 截止日期
        """
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                # 获取文件修改时间
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if mtime < cutoff_date:
                    try:
                        file_path.unlink()
                        self.logger.debug(f"删除过期文件: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"删除文件失败: {file_path}, {e}")
    
    async def _backup_data(self):
        """
        备份数据
        """
        if not self.backup_enabled or not self.backup_path:
            return
        
        try:
            import shutil
            
            # 创建备份目录
            backup_dir = self.backup_path / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制数据文件
            for item in self.base_path.iterdir():
                if item.is_dir():
                    shutil.copytree(item, backup_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, backup_dir)
            
            self.logger.info(f"数据备份完成: {backup_dir}")
            
        except Exception as e:
            self.logger.error(f"数据备份失败: {e}")
    
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
                'base_path': str(self.base_path),
                'backup_path': str(self.backup_path) if self.backup_path else None
            },
            'performance': self.stats.copy(),
            'config': {
                'format': self.format,
                'compression': self.compression,
                'partition_by': self.partition_by,
                'max_file_size': self.max_file_size,
                'backup_enabled': self.backup_enabled,
                'retention_days': self.retention_days
            }
        }
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        获取存储空间统计信息
        
        Returns:
            Dict[str, Any]: 存储统计信息
        """
        try:
            stats = {}
            total_size = 0
            total_files = 0
            
            for data_type_dir in self.base_path.iterdir():
                if data_type_dir.is_dir():
                    dir_size = 0
                    dir_files = 0
                    
                    for file_path in data_type_dir.rglob('*'):
                        if file_path.is_file():
                            file_size = file_path.stat().st_size
                            dir_size += file_size
                            dir_files += 1
                    
                    stats[data_type_dir.name] = {
                        'size_bytes': dir_size,
                        'size_mb': round(dir_size / 1024 / 1024, 2),
                        'file_count': dir_files
                    }
                    
                    total_size += dir_size
                    total_files += dir_files
            
            stats['total'] = {
                'size_bytes': total_size,
                'size_mb': round(total_size / 1024 / 1024, 2),
                'size_gb': round(total_size / 1024 / 1024 / 1024, 2),
                'file_count': total_files
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取存储统计信息失败: {e}")
            return {}