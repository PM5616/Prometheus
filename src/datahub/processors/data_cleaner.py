"""Data Cleaner Module

数据清洗器，用于处理异常值、缺失值、重复数据和数据标准化。

主要功能：
- 异常值检测和处理
- 缺失值填充
- 重复数据去除
- 数据标准化和归一化
- 数据类型转换
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import time

from .base import BaseProcessor, ProcessorType, ProcessingResult
from ...common.exceptions.data import DataProcessingError


class DataCleaner(BaseProcessor):
    """数据清洗器
    
    提供各种数据清洗功能，包括异常值处理、缺失值填充、
    重复数据去除和数据标准化等。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化数据清洗器
        
        Args:
            config: 清洗配置
                - outlier_method: 异常值检测方法 ('iqr', 'zscore', 'isolation')
                - outlier_threshold: 异常值阈值
                - missing_method: 缺失值处理方法 ('drop', 'forward', 'backward', 'mean', 'median')
                - normalize_method: 标准化方法 ('zscore', 'minmax', 'robust')
                - remove_duplicates: 是否去除重复数据
        """
        super().__init__("DataCleaner", ProcessorType.CLEANER, config)
        
        # 默认配置
        self.outlier_method = self.config.get('outlier_method', 'iqr')
        self.outlier_threshold = self.config.get('outlier_threshold', 1.5)
        self.missing_method = self.config.get('missing_method', 'forward')
        self.normalize_method = self.config.get('normalize_method', 'zscore')
        self.remove_duplicates = self.config.get('remove_duplicates', True)
        
        # 数值列配置
        self.numeric_columns = self.config.get('numeric_columns', [])
        self.datetime_columns = self.config.get('datetime_columns', [])
        
        # 清洗统计
        self.cleaning_stats = {
            'outliers_removed': 0,
            'missing_filled': 0,
            'duplicates_removed': 0,
            'rows_processed': 0
        }
    
    def initialize(self) -> bool:
        """初始化清洗器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info(f"初始化数据清洗器: {self.name}")
            self.logger.info(f"异常值检测方法: {self.outlier_method}")
            self.logger.info(f"缺失值处理方法: {self.missing_method}")
            self.logger.info(f"标准化方法: {self.normalize_method}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"初始化数据清洗器失败: {e}")
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
        
        return True
    
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """清洗数据
        
        Args:
            data: 待清洗的数据（DataFrame或其他格式）
            **kwargs: 额外参数
                - skip_outliers: 跳过异常值处理
                - skip_missing: 跳过缺失值处理
                - skip_duplicates: 跳过重复值处理
                - skip_normalize: 跳过标准化
                
        Returns:
            ProcessingResult: 清洗结果
        """
        start_time = time.time()
        
        try:
            if not self.validate_input(data):
                return ProcessingResult(
                    data=None,
                    success=False,
                    message="输入数据无效"
                )
            
            # 转换为DataFrame
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                df = self._convert_to_dataframe(data)
            
            original_rows = len(df)
            cleaning_info = {
                'original_rows': original_rows,
                'outliers_removed': 0,
                'missing_filled': 0,
                'duplicates_removed': 0
            }
            
            # 1. 处理重复数据
            if not kwargs.get('skip_duplicates', False) and self.remove_duplicates:
                df, duplicates_count = self._remove_duplicates(df)
                cleaning_info['duplicates_removed'] = duplicates_count
            
            # 2. 处理异常值
            if not kwargs.get('skip_outliers', False):
                df, outliers_count = self._handle_outliers(df)
                cleaning_info['outliers_removed'] = outliers_count
            
            # 3. 处理缺失值
            if not kwargs.get('skip_missing', False):
                df, missing_count = self._handle_missing_values(df)
                cleaning_info['missing_filled'] = missing_count
            
            # 4. 数据标准化
            if not kwargs.get('skip_normalize', False):
                df = self._normalize_data(df)
            
            # 5. 数据类型转换
            df = self._convert_data_types(df)
            
            cleaning_info['final_rows'] = len(df)
            
            # 更新统计信息
            self.cleaning_stats['outliers_removed'] += cleaning_info['outliers_removed']
            self.cleaning_stats['missing_filled'] += cleaning_info['missing_filled']
            self.cleaning_stats['duplicates_removed'] += cleaning_info['duplicates_removed']
            self.cleaning_stats['rows_processed'] += original_rows
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                data=df,
                success=True,
                message=f"数据清洗完成，原始行数: {original_rows}, 最终行数: {len(df)}",
                metadata=cleaning_info,
                processing_time=processing_time
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"数据清洗失败: {str(e)}"
            self.logger.error(error_msg)
            
            result = ProcessingResult(
                data=None,
                success=False,
                message=error_msg,
                processing_time=processing_time
            )
            
            self._update_stats(result)
            return result
    
    def _convert_to_dataframe(self, data: Any) -> pd.DataFrame:
        """将数据转换为DataFrame
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 转换后的DataFrame
        """
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame({'value': data})
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise DataProcessingError(f"不支持的数据类型: {type(data)}")
    
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """去除重复数据
        
        Args:
            df: 输入DataFrame
            
        Returns:
            Tuple[pd.DataFrame, int]: 清洗后的DataFrame和去除的重复行数
        """
        original_count = len(df)
        df_cleaned = df.drop_duplicates()
        duplicates_removed = original_count - len(df_cleaned)
        
        if duplicates_removed > 0:
            self.logger.info(f"去除了 {duplicates_removed} 行重复数据")
        
        return df_cleaned, duplicates_removed
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """处理异常值
        
        Args:
            df: 输入DataFrame
            
        Returns:
            Tuple[pd.DataFrame, int]: 处理后的DataFrame和处理的异常值数量
        """
        outliers_removed = 0
        df_cleaned = df.copy()
        
        # 获取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if self.numeric_columns:
            numeric_cols = [col for col in numeric_cols if col in self.numeric_columns]
        
        for col in numeric_cols:
            if self.outlier_method == 'iqr':
                outlier_mask = self._detect_outliers_iqr(df[col])
            elif self.outlier_method == 'zscore':
                outlier_mask = self._detect_outliers_zscore(df[col])
            else:
                continue
            
            outliers_count = outlier_mask.sum()
            if outliers_count > 0:
                # 移除异常值行
                df_cleaned = df_cleaned[~outlier_mask]
                outliers_removed += outliers_count
                self.logger.info(f"列 {col} 检测到 {outliers_count} 个异常值")
        
        return df_cleaned, outliers_removed
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """使用IQR方法检测异常值
        
        Args:
            series: 数据序列
            
        Returns:
            pd.Series: 异常值掩码
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """使用Z-Score方法检测异常值
        
        Args:
            series: 数据序列
            
        Returns:
            pd.Series: 异常值掩码
        """
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > self.outlier_threshold
    
    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """处理缺失值
        
        Args:
            df: 输入DataFrame
            
        Returns:
            Tuple[pd.DataFrame, int]: 处理后的DataFrame和填充的缺失值数量
        """
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            return df, 0
        
        df_filled = df.copy()
        
        if self.missing_method == 'drop':
            df_filled = df_filled.dropna()
            filled_count = missing_count
        elif self.missing_method == 'forward':
            df_filled = df_filled.fillna(method='ffill')
            filled_count = missing_count
        elif self.missing_method == 'backward':
            df_filled = df_filled.fillna(method='bfill')
            filled_count = missing_count
        elif self.missing_method == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
            filled_count = missing_count
        elif self.missing_method == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            filled_count = missing_count
        else:
            filled_count = 0
        
        if filled_count > 0:
            self.logger.info(f"处理了 {filled_count} 个缺失值")
        
        return df_filled, filled_count
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据标准化
        
        Args:
            df: 输入DataFrame
            
        Returns:
            pd.DataFrame: 标准化后的DataFrame
        """
        df_normalized = df.copy()
        
        # 获取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if self.numeric_columns:
            numeric_cols = [col for col in numeric_cols if col in self.numeric_columns]
        
        for col in numeric_cols:
            if self.normalize_method == 'zscore':
                df_normalized[col] = (df[col] - df[col].mean()) / df[col].std()
            elif self.normalize_method == 'minmax':
                df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            elif self.normalize_method == 'robust':
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                df_normalized[col] = (df[col] - median) / mad
        
        return df_normalized
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型
        
        Args:
            df: 输入DataFrame
            
        Returns:
            pd.DataFrame: 类型转换后的DataFrame
        """
        df_converted = df.copy()
        
        # 转换日期时间列
        for col in self.datetime_columns:
            if col in df_converted.columns:
                try:
                    df_converted[col] = pd.to_datetime(df_converted[col])
                except Exception as e:
                    self.logger.warning(f"无法转换列 {col} 为日期时间类型: {e}")
        
        return df_converted
    
    def get_cleaning_stats(self) -> Dict:
        """获取清洗统计信息
        
        Returns:
            Dict: 清洗统计信息
        """
        return self.cleaning_stats.copy()
    
    def reset_cleaning_stats(self) -> None:
        """重置清洗统计信息"""
        self.cleaning_stats = {
            'outliers_removed': 0,
            'missing_filled': 0,
            'duplicates_removed': 0,
            'rows_processed': 0
        }
        self.logger.info("清洗统计信息已重置")