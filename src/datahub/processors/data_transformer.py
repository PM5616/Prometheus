"""Data Transformer Module

数据转换器，用于数据格式转换、时间序列重采样、数据聚合等操作。

主要功能：
- 数据格式转换（DataFrame、JSON、CSV等）
- 时间序列重采样和聚合
- 数据透视和重塑
- 列操作（重命名、删除、添加）
- 数据合并和连接
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import time
import json

from .base import BaseProcessor, ProcessorType, ProcessingResult
from ...common.models.market import Kline, Ticker, OrderBook
from ...common.exceptions.data import DataProcessingError


class DataTransformer(BaseProcessor):
    """数据转换器
    
    提供各种数据转换功能，包括格式转换、时间序列处理、
    数据聚合和重塑等。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化数据转换器
        
        Args:
            config: 转换配置
                - resample_freq: 重采样频率 ('1min', '5min', '1H', '1D')
                - resample_method: 重采样方法 ('ohlc', 'mean', 'sum', 'first', 'last')
                - datetime_column: 时间列名
                - group_columns: 分组列
                - pivot_columns: 透视表配置
        """
        super().__init__("DataTransformer", ProcessorType.TRANSFORMER, config)
        
        # 默认配置
        self.resample_freq = self.config.get('resample_freq', '1min')
        self.resample_method = self.config.get('resample_method', 'ohlc')
        self.datetime_column = self.config.get('datetime_column', 'timestamp')
        self.group_columns = self.config.get('group_columns', [])
        
        # 列映射配置
        self.column_mapping = self.config.get('column_mapping', {})
        self.drop_columns = self.config.get('drop_columns', [])
        
        # 转换统计
        self.transform_stats = {
            'records_transformed': 0,
            'resampled_count': 0,
            'format_conversions': 0,
            'aggregations_performed': 0
        }
    
    def initialize(self) -> bool:
        """初始化转换器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info(f"初始化数据转换器: {self.name}")
            self.logger.info(f"重采样频率: {self.resample_freq}")
            self.logger.info(f"重采样方法: {self.resample_method}")
            self.logger.info(f"时间列: {self.datetime_column}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"初始化数据转换器失败: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 数据是否有效
        """
        if data is None:
            return False
        
        if isinstance(data, pd.DataFrame):
            return not data.empty
        elif isinstance(data, (list, tuple)):
            return len(data) > 0
        elif isinstance(data, dict):
            return bool(data)
        elif isinstance(data, str):
            return len(data) > 0
        
        return True
    
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """转换数据
        
        Args:
            data: 待转换的数据
            **kwargs: 额外参数
                - operation: 转换操作类型
                - target_format: 目标格式
                - resample_freq: 重采样频率
                - agg_method: 聚合方法
                
        Returns:
            ProcessingResult: 转换结果
        """
        start_time = time.time()
        
        try:
            if not self.validate_input(data):
                return ProcessingResult(
                    data=None,
                    success=False,
                    message="输入数据无效"
                )
            
            operation = kwargs.get('operation', 'format_conversion')
            
            # 根据操作类型进行转换
            if operation == 'format_conversion':
                result_data = self._convert_format(data, **kwargs)
            elif operation == 'resample':
                result_data = self._resample_data(data, **kwargs)
            elif operation == 'aggregate':
                result_data = self._aggregate_data(data, **kwargs)
            elif operation == 'pivot':
                result_data = self._pivot_data(data, **kwargs)
            elif operation == 'merge':
                result_data = self._merge_data(data, **kwargs)
            elif operation == 'reshape':
                result_data = self._reshape_data(data, **kwargs)
            else:
                # 默认格式转换
                result_data = self._convert_format(data, **kwargs)
            
            # 应用列操作
            if isinstance(result_data, pd.DataFrame):
                result_data = self._apply_column_operations(result_data)
            
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self.transform_stats['records_transformed'] += 1
            if operation == 'resample':
                self.transform_stats['resampled_count'] += 1
            elif operation == 'format_conversion':
                self.transform_stats['format_conversions'] += 1
            elif operation == 'aggregate':
                self.transform_stats['aggregations_performed'] += 1
            
            result = ProcessingResult(
                data=result_data,
                success=True,
                message=f"数据转换完成，操作类型: {operation}",
                metadata={'operation': operation, 'processing_time': processing_time},
                processing_time=processing_time
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"数据转换失败: {str(e)}"
            self.logger.error(error_msg)
            
            result = ProcessingResult(
                data=None,
                success=False,
                message=error_msg,
                processing_time=processing_time
            )
            
            self._update_stats(result)
            return result
    
    def _convert_format(self, data: Any, **kwargs) -> Any:
        """格式转换
        
        Args:
            data: 输入数据
            **kwargs: 转换参数
                - target_format: 目标格式 ('dataframe', 'json', 'dict', 'list')
                
        Returns:
            Any: 转换后的数据
        """
        target_format = kwargs.get('target_format', 'dataframe')
        
        if target_format == 'dataframe':
            return self._to_dataframe(data)
        elif target_format == 'json':
            return self._to_json(data)
        elif target_format == 'dict':
            return self._to_dict(data)
        elif target_format == 'list':
            return self._to_list(data)
        elif target_format == 'kline':
            return self._to_kline_objects(data)
        else:
            raise DataProcessingError(f"不支持的目标格式: {target_format}")
    
    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """转换为DataFrame
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: DataFrame格式数据
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                return pd.DataFrame(data)
            elif len(data) > 0 and hasattr(data[0], '__dict__'):
                # 对象列表转DataFrame
                return pd.DataFrame([obj.__dict__ for obj in data])
            else:
                return pd.DataFrame({'value': data})
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, str):
            # 尝试解析JSON字符串
            try:
                json_data = json.loads(data)
                return self._to_dataframe(json_data)
            except json.JSONDecodeError:
                return pd.DataFrame({'text': [data]})
        else:
            raise DataProcessingError(f"无法将 {type(data)} 转换为DataFrame")
    
    def _to_json(self, data: Any) -> str:
        """转换为JSON字符串
        
        Args:
            data: 输入数据
            
        Returns:
            str: JSON字符串
        """
        if isinstance(data, pd.DataFrame):
            return data.to_json(orient='records', date_format='iso')
        elif isinstance(data, (dict, list)):
            return json.dumps(data, default=str, ensure_ascii=False)
        else:
            return json.dumps({'value': str(data)}, ensure_ascii=False)
    
    def _to_dict(self, data: Any) -> Dict:
        """转换为字典
        
        Args:
            data: 输入数据
            
        Returns:
            Dict: 字典格式数据
        """
        if isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, dict):
            return data.copy()
        elif isinstance(data, list):
            return {'data': data}
        else:
            return {'value': data}
    
    def _to_list(self, data: Any) -> List:
        """转换为列表
        
        Args:
            data: 输入数据
            
        Returns:
            List: 列表格式数据
        """
        if isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, list):
            return data.copy()
        elif isinstance(data, dict):
            return [data]
        else:
            return [data]
    
    def _to_kline_objects(self, data: Any) -> List[Kline]:
        """转换为Kline对象列表
        
        Args:
            data: 输入数据（DataFrame或字典列表）
            
        Returns:
            List[Kline]: Kline对象列表
        """
        if isinstance(data, pd.DataFrame):
            klines = []
            for _, row in data.iterrows():
                kline = Kline(
                    symbol=row.get('symbol', ''),
                    timestamp=row.get('timestamp', datetime.now()),
                    open_price=float(row.get('open', 0)),
                    high_price=float(row.get('high', 0)),
                    low_price=float(row.get('low', 0)),
                    close_price=float(row.get('close', 0)),
                    volume=float(row.get('volume', 0))
                )
                klines.append(kline)
            return klines
        elif isinstance(data, list):
            return [Kline(**item) if isinstance(item, dict) else item for item in data]
        else:
            raise DataProcessingError("无法转换为Kline对象")
    
    def _resample_data(self, data: Any, **kwargs) -> pd.DataFrame:
        """重采样数据
        
        Args:
            data: 输入数据
            **kwargs: 重采样参数
                - freq: 重采样频率
                - method: 重采样方法
                
        Returns:
            pd.DataFrame: 重采样后的数据
        """
        df = self._to_dataframe(data)
        
        freq = kwargs.get('freq', self.resample_freq)
        method = kwargs.get('method', self.resample_method)
        
        # 确保有时间索引
        if self.datetime_column in df.columns:
            df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
            df = df.set_index(self.datetime_column)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise DataProcessingError("数据缺少时间索引，无法进行重采样")
        
        # 执行重采样
        if method == 'ohlc':
            # OHLC重采样（适用于价格数据）
            price_cols = ['open', 'high', 'low', 'close']
            available_price_cols = [col for col in price_cols if col in df.columns]
            
            if available_price_cols:
                resampled = df[available_price_cols].resample(freq).ohlc()
                # 展平多级列索引
                resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
            else:
                resampled = df.resample(freq).mean()
            
            # 处理成交量
            if 'volume' in df.columns:
                volume_resampled = df['volume'].resample(freq).sum()
                resampled['volume'] = volume_resampled
                
        elif method == 'mean':
            resampled = df.resample(freq).mean()
        elif method == 'sum':
            resampled = df.resample(freq).sum()
        elif method == 'first':
            resampled = df.resample(freq).first()
        elif method == 'last':
            resampled = df.resample(freq).last()
        else:
            resampled = df.resample(freq).mean()
        
        return resampled.reset_index()
    
    def _aggregate_data(self, data: Any, **kwargs) -> pd.DataFrame:
        """聚合数据
        
        Args:
            data: 输入数据
            **kwargs: 聚合参数
                - group_by: 分组列
                - agg_method: 聚合方法
                
        Returns:
            pd.DataFrame: 聚合后的数据
        """
        df = self._to_dataframe(data)
        
        group_by = kwargs.get('group_by', self.group_columns)
        agg_method = kwargs.get('agg_method', 'mean')
        
        if not group_by:
            raise DataProcessingError("未指定分组列")
        
        # 执行分组聚合
        if isinstance(agg_method, str):
            aggregated = df.groupby(group_by).agg(agg_method)
        elif isinstance(agg_method, dict):
            aggregated = df.groupby(group_by).agg(agg_method)
        else:
            aggregated = df.groupby(group_by).mean()
        
        return aggregated.reset_index()
    
    def _pivot_data(self, data: Any, **kwargs) -> pd.DataFrame:
        """透视数据
        
        Args:
            data: 输入数据
            **kwargs: 透视参数
                - index: 行索引
                - columns: 列索引
                - values: 值列
                
        Returns:
            pd.DataFrame: 透视后的数据
        """
        df = self._to_dataframe(data)
        
        index = kwargs.get('index')
        columns = kwargs.get('columns')
        values = kwargs.get('values')
        
        if not all([index, columns, values]):
            raise DataProcessingError("透视操作需要指定index、columns和values参数")
        
        pivoted = df.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=kwargs.get('aggfunc', 'mean')
        )
        
        return pivoted.reset_index()
    
    def _merge_data(self, data: Any, **kwargs) -> pd.DataFrame:
        """合并数据
        
        Args:
            data: 主数据
            **kwargs: 合并参数
                - right_data: 要合并的数据
                - on: 合并键
                - how: 合并方式
                
        Returns:
            pd.DataFrame: 合并后的数据
        """
        left_df = self._to_dataframe(data)
        right_data = kwargs.get('right_data')
        
        if right_data is None:
            raise DataProcessingError("未提供要合并的数据")
        
        right_df = self._to_dataframe(right_data)
        
        on = kwargs.get('on')
        how = kwargs.get('how', 'inner')
        
        merged = pd.merge(left_df, right_df, on=on, how=how)
        return merged
    
    def _reshape_data(self, data: Any, **kwargs) -> pd.DataFrame:
        """重塑数据
        
        Args:
            data: 输入数据
            **kwargs: 重塑参数
                - operation: 重塑操作 ('melt', 'stack', 'unstack')
                
        Returns:
            pd.DataFrame: 重塑后的数据
        """
        df = self._to_dataframe(data)
        operation = kwargs.get('operation', 'melt')
        
        if operation == 'melt':
            id_vars = kwargs.get('id_vars')
            value_vars = kwargs.get('value_vars')
            reshaped = pd.melt(df, id_vars=id_vars, value_vars=value_vars)
        elif operation == 'stack':
            reshaped = df.stack().reset_index()
        elif operation == 'unstack':
            reshaped = df.unstack().reset_index()
        else:
            raise DataProcessingError(f"不支持的重塑操作: {operation}")
        
        return reshaped
    
    def _apply_column_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用列操作
        
        Args:
            df: 输入DataFrame
            
        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        result_df = df.copy()
        
        # 重命名列
        if self.column_mapping:
            result_df = result_df.rename(columns=self.column_mapping)
        
        # 删除列
        if self.drop_columns:
            columns_to_drop = [col for col in self.drop_columns if col in result_df.columns]
            if columns_to_drop:
                result_df = result_df.drop(columns=columns_to_drop)
        
        return result_df
    
    def get_transform_stats(self) -> Dict:
        """获取转换统计信息
        
        Returns:
            Dict: 转换统计信息
        """
        return self.transform_stats.copy()
    
    def reset_transform_stats(self) -> None:
        """重置转换统计信息"""
        self.transform_stats = {
            'records_transformed': 0,
            'resampled_count': 0,
            'format_conversions': 0,
            'aggregations_performed': 0
        }
        self.logger.info("转换统计信息已重置")