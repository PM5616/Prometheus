"""Data Filter Module

数据过滤处理器，用于对数据进行筛选、过滤和条件判断。

功能特性：
- 多条件数据过滤
- 动态过滤规则
- 数据质量检查
- 异常值检测
- 自定义过滤函数
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import re
import time
import logging

from .base import BaseProcessor, ProcessorType, ProcessingResult
from src.common.models.market import Ticker, Kline, OrderBook
from src.common.exceptions.data import DataProcessingError


class FilterRule:
    """过滤规则类"""
    
    def __init__(self,
                 column: str,
                 operator: str,
                 value: Any,
                 rule_type: str = 'include',
                 description: str = ""):
        """
        初始化过滤规则
        
        Args:
            column: 列名
            operator: 操作符 ('>', '<', '>=', '<=', '==', '!=', 'in', 'not_in', 'contains', 'regex')
            value: 比较值
            rule_type: 规则类型 ('include', 'exclude')
            description: 规则描述
        """
        self.column = column
        self.operator = operator
        self.value = value
        self.rule_type = rule_type
        self.description = description
        self.created_at = datetime.now()
    
    def apply(self, df: pd.DataFrame) -> pd.Series:
        """
        应用过滤规则
        
        Args:
            df: 数据DataFrame
            
        Returns:
            pd.Series: 布尔掩码
        """
        if self.column not in df.columns:
            raise ValueError(f"列 {self.column} 不存在")
        
        column_data = df[self.column]
        
        # 应用操作符
        if self.operator == '>':
            mask = column_data > self.value
        elif self.operator == '<':
            mask = column_data < self.value
        elif self.operator == '>=':
            mask = column_data >= self.value
        elif self.operator == '<=':
            mask = column_data <= self.value
        elif self.operator == '==':
            mask = column_data == self.value
        elif self.operator == '!=':
            mask = column_data != self.value
        elif self.operator == 'in':
            mask = column_data.isin(self.value)
        elif self.operator == 'not_in':
            mask = ~column_data.isin(self.value)
        elif self.operator == 'contains':
            mask = column_data.astype(str).str.contains(str(self.value), na=False)
        elif self.operator == 'regex':
            mask = column_data.astype(str).str.match(str(self.value), na=False)
        elif self.operator == 'is_null':
            mask = column_data.isnull()
        elif self.operator == 'not_null':
            mask = column_data.notnull()
        else:
            raise ValueError(f"不支持的操作符: {self.operator}")
        
        # 应用规则类型
        if self.rule_type == 'exclude':
            mask = ~mask
        
        return mask
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'column': self.column,
            'operator': self.operator,
            'value': self.value,
            'rule_type': self.rule_type,
            'description': self.description,
            'created_at': self.created_at.isoformat()
        }


class FilterConfig:
    """过滤配置类"""
    
    def __init__(self,
                 rules: List[FilterRule] = None,
                 logic_operator: str = 'AND',
                 quality_checks: Dict[str, Any] = None,
                 outlier_detection: Dict[str, Any] = None):
        """
        初始化过滤配置
        
        Args:
            rules: 过滤规则列表
            logic_operator: 逻辑操作符 ('AND', 'OR')
            quality_checks: 数据质量检查配置
            outlier_detection: 异常值检测配置
        """
        self.rules = rules or []
        self.logic_operator = logic_operator
        self.quality_checks = quality_checks or {
            'check_nulls': True,
            'check_duplicates': True,
            'check_data_types': True,
            'null_threshold': 0.1,  # 空值比例阈值
            'duplicate_threshold': 0.05  # 重复值比例阈值
        }
        self.outlier_detection = outlier_detection or {
            'method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
            'threshold': 3.0,
            'columns': []  # 空列表表示检测所有数值列
        }


class FilterResult:
    """过滤结果类"""
    
    def __init__(self,
                 filtered_data: pd.DataFrame,
                 filter_stats: Dict[str, Any],
                 quality_report: Dict[str, Any],
                 outlier_report: Dict[str, Any]):
        """
        初始化过滤结果
        
        Args:
            filtered_data: 过滤后的数据
            filter_stats: 过滤统计信息
            quality_report: 数据质量报告
            outlier_report: 异常值报告
        """
        self.filtered_data = filtered_data
        self.filter_stats = filter_stats
        self.quality_report = quality_report
        self.outlier_report = outlier_report
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'data': self.filtered_data.to_dict('records'),
            'filter_stats': self.filter_stats,
            'quality_report': self.quality_report,
            'outlier_report': self.outlier_report,
            'created_at': self.created_at.isoformat()
        }


class DataFilter(BaseProcessor):
    """数据过滤处理器
    
    提供多条件数据过滤和数据质量检查功能。
    """
    
    def __init__(self, name: str = "DataFilter", config: Optional[Dict] = None):
        """
        初始化数据过滤器
        
        Args:
            name: 处理器名称
            config: 过滤配置
        """
        super().__init__(name, ProcessorType.FILTER, config)
        
        # 过滤配置
        self.filter_config = FilterConfig(**self.config.get('filter', {}))
        
        # 过滤历史
        self.filter_history = []
        self.max_history = self.config.get('max_history', 100)
        
        # 性能统计
        self.performance_stats = {
            'total_filtered': 0,
            'total_removed': 0,
            'avg_filter_time': 0.0,
            'quality_issues_found': 0,
            'outliers_detected': 0
        }
        
        self.logger.info(f"数据过滤器 {name} 初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化过滤器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 验证过滤规则
            self._validate_filter_rules()
            
            # 清理历史记录
            self.filter_history.clear()
            
            # 重置统计信息
            self.performance_stats = {
                'total_filtered': 0,
                'total_removed': 0,
                'avg_filter_time': 0.0,
                'quality_issues_found': 0,
                'outliers_detected': 0
            }
            
            self._initialized = True
            self.logger.info(f"数据过滤器 {self.name} 初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"数据过滤器初始化失败: {e}")
            return False
    
    def _validate_filter_rules(self):
        """
        验证过滤规则配置
        """
        valid_operators = {
            '>', '<', '>=', '<=', '==', '!=', 'in', 'not_in',
            'contains', 'regex', 'is_null', 'not_null'
        }
        
        for rule in self.filter_config.rules:
            if rule.operator not in valid_operators:
                raise ValueError(f"无效的操作符: {rule.operator}")
            
            if rule.rule_type not in ['include', 'exclude']:
                raise ValueError(f"无效的规则类型: {rule.rule_type}")
    
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
        处理数据过滤
        
        Args:
            data: 待过滤的数据
            **kwargs: 额外参数
                - custom_rules: 自定义过滤规则
                - skip_quality_check: 跳过数据质量检查
                - skip_outlier_detection: 跳过异常值检测
                
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
            original_count = len(df)
            
            # 获取过滤参数
            custom_rules = kwargs.get('custom_rules', [])
            skip_quality_check = kwargs.get('skip_quality_check', False)
            skip_outlier_detection = kwargs.get('skip_outlier_detection', False)
            
            # 应用过滤规则
            filtered_df, filter_stats = self._apply_filter_rules(df, custom_rules)
            
            # 数据质量检查
            quality_report = {}
            if not skip_quality_check:
                quality_report = self._check_data_quality(filtered_df)
            
            # 异常值检测
            outlier_report = {}
            if not skip_outlier_detection:
                filtered_df, outlier_report = self._detect_outliers(filtered_df)
            
            # 创建过滤结果
            filter_result = FilterResult(
                filtered_data=filtered_df,
                filter_stats=filter_stats,
                quality_report=quality_report,
                outlier_report=outlier_report
            )
            
            # 更新统计信息
            self._update_performance_stats(original_count, len(filtered_df), time.time() - start_time)
            
            # 记录过滤历史
            self._record_filter_history(filter_result)
            
            processing_result = ProcessingResult(
                data=filter_result,
                success=True,
                message=f"数据过滤完成，从 {original_count} 行过滤到 {len(filtered_df)} 行",
                metadata={
                    'original_count': original_count,
                    'filtered_count': len(filtered_df),
                    'removed_count': original_count - len(filtered_df),
                    'removal_rate': (original_count - len(filtered_df)) / original_count if original_count > 0 else 0
                },
                processing_time=time.time() - start_time
            )
            
            self._update_stats(processing_result)
            return processing_result
            
        except Exception as e:
            self.logger.error(f"数据过滤失败: {e}")
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
    
    # ==================== 过滤方法 ====================
    
    def _apply_filter_rules(self, df: pd.DataFrame, custom_rules: List[FilterRule] = None) -> tuple:
        """
        应用过滤规则
        
        Args:
            df: 数据DataFrame
            custom_rules: 自定义过滤规则
            
        Returns:
            tuple: (过滤后的DataFrame, 过滤统计信息)
        """
        if df.empty:
            return df, {'rules_applied': 0, 'rows_removed': 0}
        
        # 合并规则
        all_rules = self.filter_config.rules.copy()
        if custom_rules:
            all_rules.extend(custom_rules)
        
        if not all_rules:
            return df, {'rules_applied': 0, 'rows_removed': 0}
        
        original_count = len(df)
        
        # 应用过滤规则
        masks = []
        rule_stats = []
        
        for rule in all_rules:
            try:
                mask = rule.apply(df)
                masks.append(mask)
                
                # 统计规则效果
                rows_affected = mask.sum() if rule.rule_type == 'include' else (~mask).sum()
                rule_stats.append({
                    'rule': rule.to_dict(),
                    'rows_affected': int(rows_affected),
                    'effect_rate': float(rows_affected / len(df)) if len(df) > 0 else 0.0
                })
                
            except Exception as e:
                self.logger.error(f"过滤规则应用失败: {rule.to_dict()}, 错误: {e}")
                continue
        
        # 组合掩码
        if masks:
            if self.filter_config.logic_operator == 'AND':
                combined_mask = pd.Series([True] * len(df), index=df.index)
                for mask in masks:
                    combined_mask &= mask
            else:  # OR
                combined_mask = pd.Series([False] * len(df), index=df.index)
                for mask in masks:
                    combined_mask |= mask
            
            filtered_df = df[combined_mask]
        else:
            filtered_df = df
        
        # 统计信息
        filter_stats = {
            'rules_applied': len(all_rules),
            'successful_rules': len(rule_stats),
            'original_count': original_count,
            'filtered_count': len(filtered_df),
            'rows_removed': original_count - len(filtered_df),
            'removal_rate': (original_count - len(filtered_df)) / original_count if original_count > 0 else 0.0,
            'rule_details': rule_stats,
            'logic_operator': self.filter_config.logic_operator
        }
        
        return filtered_df, filter_stats
    
    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        检查数据质量
        
        Args:
            df: 数据DataFrame
            
        Returns:
            Dict[str, Any]: 数据质量报告
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'issues_found': [],
            'quality_score': 1.0
        }
        
        if df.empty:
            return quality_report
        
        issues_count = 0
        
        # 检查空值
        if self.filter_config.quality_checks.get('check_nulls', True):
            null_counts = df.isnull().sum()
            null_rates = null_counts / len(df)
            threshold = self.filter_config.quality_checks.get('null_threshold', 0.1)
            
            high_null_columns = null_rates[null_rates > threshold]
            if not high_null_columns.empty:
                quality_report['issues_found'].append({
                    'type': 'high_null_rate',
                    'columns': high_null_columns.to_dict(),
                    'threshold': threshold
                })
                issues_count += len(high_null_columns)
            
            quality_report['null_statistics'] = {
                'total_nulls': int(null_counts.sum()),
                'null_rate_by_column': null_rates.to_dict()
            }
        
        # 检查重复值
        if self.filter_config.quality_checks.get('check_duplicates', True):
            duplicate_count = df.duplicated().sum()
            duplicate_rate = duplicate_count / len(df)
            threshold = self.filter_config.quality_checks.get('duplicate_threshold', 0.05)
            
            if duplicate_rate > threshold:
                quality_report['issues_found'].append({
                    'type': 'high_duplicate_rate',
                    'duplicate_count': int(duplicate_count),
                    'duplicate_rate': float(duplicate_rate),
                    'threshold': threshold
                })
                issues_count += 1
            
            quality_report['duplicate_statistics'] = {
                'duplicate_count': int(duplicate_count),
                'duplicate_rate': float(duplicate_rate)
            }
        
        # 检查数据类型
        if self.filter_config.quality_checks.get('check_data_types', True):
            type_issues = []
            
            for column in df.columns:
                # 检查数值列中的非数值数据
                if df[column].dtype in ['object', 'string']:
                    # 尝试转换为数值
                    numeric_convertible = pd.to_numeric(df[column], errors='coerce')
                    if not numeric_convertible.isnull().all():
                        non_numeric_count = numeric_convertible.isnull().sum() - df[column].isnull().sum()
                        if non_numeric_count > 0:
                            type_issues.append({
                                'column': column,
                                'issue': 'mixed_types',
                                'non_numeric_count': int(non_numeric_count)
                            })
            
            if type_issues:
                quality_report['issues_found'].append({
                    'type': 'data_type_issues',
                    'details': type_issues
                })
                issues_count += len(type_issues)
        
        # 计算质量分数
        total_possible_issues = len(df.columns) * 3  # 每列最多3种问题
        quality_report['quality_score'] = max(0.0, 1.0 - (issues_count / total_possible_issues))
        
        # 更新统计
        self.performance_stats['quality_issues_found'] += len(quality_report['issues_found'])
        
        return quality_report
    
    def _detect_outliers(self, df: pd.DataFrame) -> tuple:
        """
        检测异常值
        
        Args:
            df: 数据DataFrame
            
        Returns:
            tuple: (处理后的DataFrame, 异常值报告)
        """
        outlier_report = {
            'method': self.filter_config.outlier_detection.get('method', 'iqr'),
            'outliers_detected': 0,
            'outliers_removed': 0,
            'outlier_details': {}
        }
        
        if df.empty:
            return df, outlier_report
        
        # 确定要检测的列
        detection_columns = self.filter_config.outlier_detection.get('columns', [])
        if not detection_columns:
            detection_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        method = self.filter_config.outlier_detection.get('method', 'iqr')
        threshold = self.filter_config.outlier_detection.get('threshold', 3.0)
        
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        for column in detection_columns:
            if column not in df.columns:
                continue
            
            column_data = df[column].dropna()
            if len(column_data) == 0:
                continue
            
            # 检测异常值
            if method == 'iqr':
                Q1 = column_data.quantile(0.25)
                Q3 = column_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                column_outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                column_outliers = z_scores > threshold
                
            elif method == 'isolation_forest':
                try:
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(df[[column]].fillna(df[column].mean()))
                    column_outliers = pd.Series(outlier_labels == -1, index=df.index)
                except ImportError:
                    self.logger.warning("sklearn未安装，使用IQR方法替代")
                    # 回退到IQR方法
                    Q1 = column_data.quantile(0.25)
                    Q3 = column_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    column_outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            
            else:
                raise ValueError(f"不支持的异常值检测方法: {method}")
            
            # 记录异常值详情
            outlier_count = column_outliers.sum()
            if outlier_count > 0:
                outlier_report['outlier_details'][column] = {
                    'count': int(outlier_count),
                    'rate': float(outlier_count / len(df)),
                    'outlier_values': df.loc[column_outliers, column].tolist()[:10]  # 最多记录10个异常值
                }
            
            # 合并异常值掩码
            outlier_mask |= column_outliers.fillna(False)
        
        # 统计异常值
        total_outliers = outlier_mask.sum()
        outlier_report['outliers_detected'] = int(total_outliers)
        
        # 移除异常值（可选）
        remove_outliers = self.filter_config.outlier_detection.get('remove_outliers', False)
        if remove_outliers:
            filtered_df = df[~outlier_mask]
            outlier_report['outliers_removed'] = int(total_outliers)
        else:
            filtered_df = df
            outlier_report['outliers_removed'] = 0
        
        # 更新统计
        self.performance_stats['outliers_detected'] += total_outliers
        
        return filtered_df, outlier_report
    
    # ==================== 辅助方法 ====================
    
    def _update_performance_stats(self, original_count: int, filtered_count: int, processing_time: float):
        """
        更新性能统计信息
        
        Args:
            original_count: 原始数据量
            filtered_count: 过滤后数据量
            processing_time: 处理时间
        """
        self.performance_stats['total_filtered'] += original_count
        self.performance_stats['total_removed'] += (original_count - filtered_count)
        
        # 更新平均处理时间
        current_avg = self.performance_stats['avg_filter_time']
        total_processed = self.stats['total_processed']
        
        if total_processed > 0:
            self.performance_stats['avg_filter_time'] = (
                (current_avg * (total_processed - 1) + processing_time) / total_processed
            )
        else:
            self.performance_stats['avg_filter_time'] = processing_time
    
    def _record_filter_history(self, filter_result: FilterResult):
        """
        记录过滤历史
        
        Args:
            filter_result: 过滤结果
        """
        history_record = {
            'timestamp': filter_result.created_at,
            'filter_stats': filter_result.filter_stats,
            'quality_score': filter_result.quality_report.get('quality_score', 0.0),
            'outliers_detected': filter_result.outlier_report.get('outliers_detected', 0)
        }
        
        self.filter_history.append(history_record)
        
        # 限制历史记录数量
        if len(self.filter_history) > self.max_history:
            self.filter_history = self.filter_history[-self.max_history:]
    
    def add_filter_rule(self, rule: FilterRule):
        """
        添加过滤规则
        
        Args:
            rule: 过滤规则
        """
        self.filter_config.rules.append(rule)
        self.logger.info(f"添加过滤规则: {rule.to_dict()}")
    
    def remove_filter_rule(self, index: int) -> bool:
        """
        移除过滤规则
        
        Args:
            index: 规则索引
            
        Returns:
            bool: 是否成功移除
        """
        if 0 <= index < len(self.filter_config.rules):
            removed_rule = self.filter_config.rules.pop(index)
            self.logger.info(f"移除过滤规则: {removed_rule.to_dict()}")
            return True
        return False
    
    def clear_filter_rules(self):
        """
        清空所有过滤规则
        """
        self.filter_config.rules.clear()
        self.logger.info("已清空所有过滤规则")
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """
        获取过滤器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        base_stats = self.get_stats()
        
        return {
            'processor_stats': base_stats,
            'performance_stats': self.performance_stats,
            'filter_config': {
                'rules_count': len(self.filter_config.rules),
                'logic_operator': self.filter_config.logic_operator,
                'quality_checks': self.filter_config.quality_checks,
                'outlier_detection': self.filter_config.outlier_detection
            },
            'history_stats': {
                'history_count': len(self.filter_history),
                'max_history': self.max_history
            }
        }
    
    def get_filter_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取过滤历史记录
        
        Args:
            limit: 返回记录数量限制
            
        Returns:
            List[Dict[str, Any]]: 历史记录列表
        """
        return self.filter_history[-limit:] if limit > 0 else self.filter_history
    
    def shutdown(self):
        """
        关闭过滤器
        """
        # 清理历史记录
        self.filter_history.clear()
        
        super().shutdown()
        self.logger.info(f"数据过滤器 {self.name} 已关闭")