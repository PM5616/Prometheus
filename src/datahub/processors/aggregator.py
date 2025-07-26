"""Data Aggregator Module

数据聚合处理器，用于对历史数据进行聚合分析和统计计算。

功能特性：
- 多维度数据聚合
- 时间序列聚合
- 统计指标计算
- 自定义聚合函数
- 增量聚合支持
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import time
import logging

from .base import BaseProcessor, ProcessorType, ProcessingResult
from src.common.models.market import Ticker, Kline, OrderBook
from src.common.exceptions.data import DataProcessingError


class AggregationConfig:
    """聚合配置类"""
    
    def __init__(self,
                 time_column: str = 'timestamp',
                 group_by: Optional[List[str]] = None,
                 aggregation_functions: Optional[Dict[str, Union[str, Callable]]] = None,
                 time_intervals: Optional[List[str]] = None,
                 rolling_windows: Optional[List[int]] = None,
                 percentiles: Optional[List[float]] = None):
        """
        初始化聚合配置
        
        Args:
            time_column: 时间列名
            group_by: 分组字段
            aggregation_functions: 聚合函数映射
            time_intervals: 时间间隔列表 ['1min', '5min', '1h', '1d']
            rolling_windows: 滚动窗口大小列表
            percentiles: 百分位数列表
        """
        self.time_column = time_column
        self.group_by = group_by or []
        self.aggregation_functions = aggregation_functions or {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'price': 'mean'
        }
        self.time_intervals = time_intervals or ['1min', '5min', '15min', '1h', '4h', '1d']
        self.rolling_windows = rolling_windows or [5, 10, 20, 50, 100, 200]
        self.percentiles = percentiles or [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


class AggregationResult:
    """聚合结果类"""
    
    def __init__(self,
                 aggregated_data: pd.DataFrame,
                 metadata: Dict[str, Any],
                 statistics: Dict[str, Any]):
        """
        初始化聚合结果
        
        Args:
            aggregated_data: 聚合后的数据
            metadata: 元数据信息
            statistics: 统计信息
        """
        self.aggregated_data = aggregated_data
        self.metadata = metadata
        self.statistics = statistics
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'data': self.aggregated_data.to_dict('records'),
            'metadata': self.metadata,
            'statistics': self.statistics,
            'created_at': self.created_at.isoformat()
        }
    
    def to_json(self) -> str:
        """转换为JSON格式"""
        import json
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False, indent=2)


class DataAggregator(BaseProcessor):
    """数据聚合处理器
    
    提供多维度数据聚合和统计分析功能。
    """
    
    def __init__(self, name: str = "DataAggregator", config: Optional[Dict] = None):
        """
        初始化数据聚合器
        
        Args:
            name: 处理器名称
            config: 聚合配置
        """
        super().__init__(name, ProcessorType.AGGREGATOR, config)
        
        # 聚合配置
        self.agg_config = AggregationConfig(**self.config.get('aggregation', {}))
        
        # 缓存聚合结果
        self.aggregation_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5分钟缓存
        
        # 增量聚合状态
        self.incremental_state = defaultdict(dict)
        
        self.logger.info(f"数据聚合器 {name} 初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化聚合器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 清理缓存
            self.aggregation_cache.clear()
            self.incremental_state.clear()
            
            # 验证聚合函数
            self._validate_aggregation_functions()
            
            self._initialized = True
            self.logger.info(f"数据聚合器 {self.name} 初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"数据聚合器初始化失败: {e}")
            return False
    
    def _validate_aggregation_functions(self):
        """
        验证聚合函数配置
        """
        valid_functions = {
            'sum', 'mean', 'median', 'min', 'max', 'std', 'var',
            'count', 'nunique', 'first', 'last', 'skew', 'kurt'
        }
        
        for column, func in self.agg_config.aggregation_functions.items():
            if isinstance(func, str) and func not in valid_functions:
                if not callable(func):
                    raise ValueError(f"无效的聚合函数: {func}")
    
    def validate_input(self, data: Any) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 数据是否有效
        """
        if data is None:
            return False
        
        # 检查数据类型
        if isinstance(data, (list, pd.DataFrame)):
            return len(data) > 0
        
        if isinstance(data, dict):
            return bool(data)
        
        if isinstance(data, (Ticker, Kline, OrderBook)):
            return True
        
        return False
    
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """
        处理数据聚合
        
        Args:
            data: 待聚合的数据
            **kwargs: 额外参数
                - aggregation_type: 聚合类型 ('time_series', 'group_by', 'rolling', 'custom')
                - time_interval: 时间间隔
                - custom_functions: 自定义聚合函数
                - incremental: 是否增量聚合
                
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            # 验证输入
            if not self.validate_input(data):
                return ProcessingResult(
                    data=None,
                    success=False,
                    message="输入数据验证失败",
                    processing_time=time.time() - start_time
                )
            
            # 转换数据格式
            df = self._convert_to_dataframe(data)
            
            # 获取聚合参数
            aggregation_type = kwargs.get('aggregation_type', 'time_series')
            time_interval = kwargs.get('time_interval', '1min')
            custom_functions = kwargs.get('custom_functions', {})
            incremental = kwargs.get('incremental', False)
            
            # 执行聚合
            if aggregation_type == 'time_series':
                result = self._aggregate_time_series(df, time_interval, incremental)
            elif aggregation_type == 'group_by':
                result = self._aggregate_group_by(df, custom_functions)
            elif aggregation_type == 'rolling':
                window_size = kwargs.get('window_size', 20)
                result = self._aggregate_rolling(df, window_size)
            elif aggregation_type == 'statistical':
                result = self._aggregate_statistical(df)
            elif aggregation_type == 'custom':
                result = self._aggregate_custom(df, custom_functions)
            else:
                raise ValueError(f"不支持的聚合类型: {aggregation_type}")
            
            # 缓存结果
            cache_key = self._generate_cache_key(aggregation_type, time_interval, df)
            self.aggregation_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            processing_result = ProcessingResult(
                data=result,
                success=True,
                message=f"数据聚合完成，类型: {aggregation_type}",
                metadata={
                    'aggregation_type': aggregation_type,
                    'time_interval': time_interval,
                    'input_rows': len(df),
                    'output_rows': len(result.aggregated_data)
                },
                processing_time=time.time() - start_time
            )
            
            self._update_stats(processing_result)
            return processing_result
            
        except Exception as e:
            self.logger.error(f"数据聚合失败: {e}")
            result = ProcessingResult(
                data=None,
                success=False,
                message=str(e),
                processing_time=time.time() - start_time
            )
            self._update_stats(result)
            return result
    
    def _convert_to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        转换数据为DataFrame格式
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 转换后的DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        if isinstance(data, list):
            if not data:
                return pd.DataFrame()
            
            # 处理不同类型的列表数据
            if isinstance(data[0], (Ticker, Kline, OrderBook)):
                return self._convert_market_data_to_df(data)
            elif isinstance(data[0], dict):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame({'value': data})
        
        if isinstance(data, dict):
            return pd.DataFrame([data])
        
        if isinstance(data, (Ticker, Kline, OrderBook)):
            return self._convert_market_data_to_df([data])
        
        raise ValueError(f"不支持的数据类型: {type(data)}")
    
    def _convert_market_data_to_df(self, data: List[Union[Ticker, Kline, OrderBook]]) -> pd.DataFrame:
        """
        转换市场数据为DataFrame
        
        Args:
            data: 市场数据列表
            
        Returns:
            pd.DataFrame: 转换后的DataFrame
        """
        records = []
        
        for item in data:
            if isinstance(item, Ticker):
                record = {
                    'symbol': item.symbol,
                    'price': float(item.price),
                    'volume': float(item.volume),
                    'change': float(item.change) if item.change else None,
                    'change_percent': float(item.change_percent) if item.change_percent else None,
                    'high': float(item.high) if item.high else None,
                    'low': float(item.low) if item.low else None,
                    'timestamp': item.timestamp
                }
            elif isinstance(item, Kline):
                record = {
                    'symbol': item.symbol,
                    'interval': item.interval,
                    'open_time': item.open_time,
                    'close_time': item.close_time,
                    'open': float(item.open),
                    'high': float(item.high),
                    'low': float(item.low),
                    'close': float(item.close),
                    'volume': float(item.volume),
                    'quote_volume': float(item.quote_volume) if item.quote_volume else None,
                    'trades_count': item.trades_count,
                    'timestamp': item.open_time
                }
            elif isinstance(item, OrderBook):
                record = {
                    'symbol': item.symbol,
                    'bids_count': len(item.bids),
                    'asks_count': len(item.asks),
                    'best_bid': float(item.bids[0][0]) if item.bids else None,
                    'best_ask': float(item.asks[0][0]) if item.asks else None,
                    'spread': float(item.asks[0][0]) - float(item.bids[0][0]) if item.bids and item.asks else None,
                    'timestamp': item.timestamp
                }
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    # ==================== 聚合方法 ====================
    
    def _aggregate_time_series(self, df: pd.DataFrame, time_interval: str, incremental: bool = False) -> AggregationResult:
        """
        时间序列聚合
        
        Args:
            df: 数据DataFrame
            time_interval: 时间间隔
            incremental: 是否增量聚合
            
        Returns:
            AggregationResult: 聚合结果
        """
        # 确保时间列存在
        if self.agg_config.time_column not in df.columns:
            raise ValueError(f"时间列 {self.agg_config.time_column} 不存在")
        
        # 设置时间索引
        df_copy = df.copy()
        df_copy[self.agg_config.time_column] = pd.to_datetime(df_copy[self.agg_config.time_column])
        df_copy.set_index(self.agg_config.time_column, inplace=True)
        
        # 按时间间隔重采样
        resampled = df_copy.resample(time_interval)
        
        # 应用聚合函数
        aggregated_data = pd.DataFrame()
        
        for column, func in self.agg_config.aggregation_functions.items():
            if column in df_copy.columns:
                if isinstance(func, str):
                    aggregated_data[column] = getattr(resampled[column], func)()
                elif callable(func):
                    aggregated_data[column] = resampled[column].apply(func)
        
        # 计算统计信息
        statistics = self._calculate_statistics(aggregated_data)
        
        # 元数据
        metadata = {
            'time_interval': time_interval,
            'start_time': df_copy.index.min(),
            'end_time': df_copy.index.max(),
            'total_periods': len(aggregated_data),
            'aggregation_functions': self.agg_config.aggregation_functions
        }
        
        return AggregationResult(aggregated_data, metadata, statistics)
    
    def _aggregate_group_by(self, df: pd.DataFrame, custom_functions: Dict[str, Callable] = None) -> AggregationResult:
        """
        分组聚合
        
        Args:
            df: 数据DataFrame
            custom_functions: 自定义聚合函数
            
        Returns:
            AggregationResult: 聚合结果
        """
        if not self.agg_config.group_by:
            raise ValueError("分组聚合需要指定group_by字段")
        
        # 检查分组字段是否存在
        missing_columns = [col for col in self.agg_config.group_by if col not in df.columns]
        if missing_columns:
            raise ValueError(f"分组字段不存在: {missing_columns}")
        
        # 分组聚合
        grouped = df.groupby(self.agg_config.group_by)
        
        # 应用聚合函数
        agg_functions = self.agg_config.aggregation_functions.copy()
        if custom_functions:
            agg_functions.update(custom_functions)
        
        aggregated_data = grouped.agg(agg_functions).reset_index()
        
        # 计算统计信息
        statistics = self._calculate_statistics(aggregated_data)
        statistics['group_count'] = len(aggregated_data)
        
        # 元数据
        metadata = {
            'group_by': self.agg_config.group_by,
            'aggregation_functions': agg_functions,
            'group_count': len(aggregated_data)
        }
        
        return AggregationResult(aggregated_data, metadata, statistics)
    
    def _aggregate_rolling(self, df: pd.DataFrame, window_size: int) -> AggregationResult:
        """
        滚动窗口聚合
        
        Args:
            df: 数据DataFrame
            window_size: 窗口大小
            
        Returns:
            AggregationResult: 聚合结果
        """
        # 确保数据按时间排序
        if self.agg_config.time_column in df.columns:
            df_copy = df.sort_values(self.agg_config.time_column).copy()
        else:
            df_copy = df.copy()
        
        # 计算滚动统计
        rolling_data = pd.DataFrame()
        
        for column in df_copy.select_dtypes(include=[np.number]).columns:
            rolling = df_copy[column].rolling(window=window_size)
            
            rolling_data[f'{column}_mean'] = rolling.mean()
            rolling_data[f'{column}_std'] = rolling.std()
            rolling_data[f'{column}_min'] = rolling.min()
            rolling_data[f'{column}_max'] = rolling.max()
            rolling_data[f'{column}_sum'] = rolling.sum()
        
        # 添加原始时间列
        if self.agg_config.time_column in df_copy.columns:
            rolling_data[self.agg_config.time_column] = df_copy[self.agg_config.time_column]
        
        # 移除NaN值
        rolling_data = rolling_data.dropna()
        
        # 计算统计信息
        statistics = self._calculate_statistics(rolling_data)
        
        # 元数据
        metadata = {
            'window_size': window_size,
            'total_rows': len(rolling_data),
            'valid_rows': len(rolling_data.dropna())
        }
        
        return AggregationResult(rolling_data, metadata, statistics)
    
    def _aggregate_statistical(self, df: pd.DataFrame) -> AggregationResult:
        """
        统计聚合
        
        Args:
            df: 数据DataFrame
            
        Returns:
            AggregationResult: 聚合结果
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            raise ValueError("没有数值列可进行统计聚合")
        
        # 基础统计
        stats_data = {
            'count': df[numeric_columns].count(),
            'mean': df[numeric_columns].mean(),
            'std': df[numeric_columns].std(),
            'min': df[numeric_columns].min(),
            'max': df[numeric_columns].max(),
            'median': df[numeric_columns].median(),
            'skew': df[numeric_columns].skew(),
            'kurt': df[numeric_columns].kurtosis()
        }
        
        # 百分位数
        for p in self.agg_config.percentiles:
            stats_data[f'p{int(p*100)}'] = df[numeric_columns].quantile(p)
        
        # 转换为DataFrame
        aggregated_data = pd.DataFrame(stats_data).T
        
        # 计算相关性矩阵
        correlation_matrix = df[numeric_columns].corr()
        
        # 统计信息
        statistics = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'numeric_columns': list(numeric_columns),
            'total_observations': len(df)
        }
        
        # 元数据
        metadata = {
            'statistical_measures': list(stats_data.keys()),
            'percentiles': self.agg_config.percentiles,
            'numeric_columns_count': len(numeric_columns)
        }
        
        return AggregationResult(aggregated_data, metadata, statistics)
    
    def _aggregate_custom(self, df: pd.DataFrame, custom_functions: Dict[str, Callable]) -> AggregationResult:
        """
        自定义聚合
        
        Args:
            df: 数据DataFrame
            custom_functions: 自定义聚合函数
            
        Returns:
            AggregationResult: 聚合结果
        """
        if not custom_functions:
            raise ValueError("自定义聚合需要提供聚合函数")
        
        # 应用自定义函数
        results = {}
        
        for name, func in custom_functions.items():
            try:
                if callable(func):
                    results[name] = func(df)
                else:
                    raise ValueError(f"函数 {name} 不可调用")
            except Exception as e:
                self.logger.error(f"自定义聚合函数 {name} 执行失败: {e}")
                results[name] = None
        
        # 转换为DataFrame
        aggregated_data = pd.DataFrame([results])
        
        # 计算统计信息
        statistics = self._calculate_statistics(aggregated_data)
        
        # 元数据
        metadata = {
            'custom_functions': list(custom_functions.keys()),
            'successful_functions': [k for k, v in results.items() if v is not None]
        }
        
        return AggregationResult(aggregated_data, metadata, statistics)
    
    # ==================== 辅助方法 ====================
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算数据统计信息
        
        Args:
            df: 数据DataFrame
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        statistics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_columns),
            'null_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        if len(numeric_columns) > 0:
            statistics.update({
                'numeric_summary': df[numeric_columns].describe().to_dict()
            })
        
        return statistics
    
    def _generate_cache_key(self, aggregation_type: str, time_interval: str, df: pd.DataFrame) -> str:
        """
        生成缓存键
        
        Args:
            aggregation_type: 聚合类型
            time_interval: 时间间隔
            df: 数据DataFrame
            
        Returns:
            str: 缓存键
        """
        import hashlib
        
        # 创建数据指纹
        data_hash = hashlib.md5(
            f"{len(df)}_{df.columns.tolist()}_{aggregation_type}_{time_interval}".encode()
        ).hexdigest()[:8]
        
        return f"{self.name}_{aggregation_type}_{time_interval}_{data_hash}"
    
    def get_cached_result(self, cache_key: str) -> Optional[AggregationResult]:
        """
        获取缓存结果
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[AggregationResult]: 缓存的聚合结果
        """
        if cache_key in self.aggregation_cache:
            cached_item = self.aggregation_cache[cache_key]
            
            # 检查缓存是否过期
            if (datetime.now() - cached_item['timestamp']).total_seconds() < self.cache_ttl:
                return cached_item['result']
            else:
                # 删除过期缓存
                del self.aggregation_cache[cache_key]
        
        return None
    
    def clear_cache(self):
        """
        清理聚合缓存
        """
        self.aggregation_cache.clear()
        self.logger.info("聚合缓存已清理")
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """
        获取聚合器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        base_stats = self.get_stats()
        
        return {
            'processor_stats': base_stats,
            'cache_stats': {
                'cache_size': len(self.aggregation_cache),
                'cache_ttl': self.cache_ttl
            },
            'config': {
                'time_column': self.agg_config.time_column,
                'group_by': self.agg_config.group_by,
                'aggregation_functions': self.agg_config.aggregation_functions,
                'time_intervals': self.agg_config.time_intervals,
                'rolling_windows': self.agg_config.rolling_windows,
                'percentiles': self.agg_config.percentiles
            }
        }
    
    def shutdown(self):
        """
        关闭聚合器
        """
        # 清理缓存
        self.clear_cache()
        
        # 清理增量状态
        self.incremental_state.clear()
        
        super().shutdown()
        self.logger.info(f"数据聚合器 {self.name} 已关闭")