"""Data Validator Module

数据验证器，用于检查数据完整性、格式验证和质量控制。

主要功能：
- 数据完整性检查
- 格式验证
- 数据质量评估
- 异常数据检测
- 业务规则验证
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import time
import re

from .base import BaseProcessor, ProcessorType, ProcessingResult
from ...common.exceptions.data import DataValidationError


class ValidationRule:
    """验证规则类"""
    
    def __init__(self, 
                 name: str,
                 validator: Callable,
                 error_message: str,
                 severity: str = 'error'):
        """初始化验证规则
        
        Args:
            name: 规则名称
            validator: 验证函数
            error_message: 错误消息
            severity: 严重程度 ('error', 'warning', 'info')
        """
        self.name = name
        self.validator = validator
        self.error_message = error_message
        self.severity = severity


class ValidationResult:
    """验证结果类"""
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.info = []
        self.stats = {}
    
    def add_issue(self, rule_name: str, message: str, severity: str, details: Dict = None):
        """添加验证问题
        
        Args:
            rule_name: 规则名称
            message: 问题描述
            severity: 严重程度
            details: 详细信息
        """
        issue = {
            'rule': rule_name,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now()
        }
        
        if severity == 'error':
            self.errors.append(issue)
            self.is_valid = False
        elif severity == 'warning':
            self.warnings.append(issue)
        elif severity == 'info':
            self.info.append(issue)
    
    def get_summary(self) -> Dict:
        """获取验证摘要
        
        Returns:
            Dict: 验证摘要
        """
        return {
            'is_valid': self.is_valid,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'info_count': len(self.info),
            'total_issues': len(self.errors) + len(self.warnings) + len(self.info)
        }


class DataValidator(BaseProcessor):
    """数据验证器
    
    提供全面的数据验证功能，包括格式检查、完整性验证、
    质量评估和业务规则验证。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化数据验证器
        
        Args:
            config: 验证配置
                - required_columns: 必需列
                - column_types: 列类型要求
                - value_ranges: 数值范围
                - date_format: 日期格式
                - business_rules: 业务规则
        """
        super().__init__("DataValidator", ProcessorType.VALIDATOR, config)
        
        # 基础验证配置
        self.required_columns = self.config.get('required_columns', [])
        self.column_types = self.config.get('column_types', {})
        self.value_ranges = self.config.get('value_ranges', {})
        self.date_format = self.config.get('date_format', '%Y-%m-%d %H:%M:%S')
        
        # 数据质量阈值
        self.quality_thresholds = self.config.get('quality_thresholds', {
            'missing_rate': 0.1,  # 缺失率阈值
            'duplicate_rate': 0.05,  # 重复率阈值
            'outlier_rate': 0.02  # 异常值率阈值
        })
        
        # 验证规则
        self.validation_rules = []
        self._setup_default_rules()
        
        # 验证统计
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'total_records_validated': 0
        }
    
    def initialize(self) -> bool:
        """初始化验证器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info(f"初始化数据验证器: {self.name}")
            self.logger.info(f"必需列: {self.required_columns}")
            self.logger.info(f"验证规则数量: {len(self.validation_rules)}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"初始化数据验证器失败: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 数据是否有效
        """
        return data is not None
    
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """验证数据
        
        Args:
            data: 待验证的数据
            **kwargs: 额外参数
                - validation_level: 验证级别 ('basic', 'standard', 'strict')
                - custom_rules: 自定义验证规则
                - skip_rules: 跳过的规则
                
        Returns:
            ProcessingResult: 验证结果
        """
        start_time = time.time()
        
        try:
            if not self.validate_input(data):
                return ProcessingResult(
                    data=None,
                    success=False,
                    message="输入数据为空"
                )
            
            validation_level = kwargs.get('validation_level', 'standard')
            custom_rules = kwargs.get('custom_rules', [])
            skip_rules = kwargs.get('skip_rules', [])
            
            # 转换为DataFrame进行验证
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                df = self._convert_to_dataframe(data)
            
            # 执行验证
            validation_result = self._perform_validation(
                df, validation_level, custom_rules, skip_rules
            )
            
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self.validation_stats['total_validations'] += 1
            self.validation_stats['total_records_validated'] += len(df)
            
            if validation_result.is_valid:
                self.validation_stats['passed_validations'] += 1
                success = True
                message = "数据验证通过"
            else:
                self.validation_stats['failed_validations'] += 1
                success = False
                message = f"数据验证失败: {len(validation_result.errors)} 个错误"
            
            result = ProcessingResult(
                data=validation_result,
                success=success,
                message=message,
                metadata={
                    'validation_summary': validation_result.get_summary(),
                    'validation_level': validation_level,
                    'records_validated': len(df)
                },
                processing_time=processing_time
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"数据验证失败: {str(e)}"
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
            return pd.DataFrame({'data': [data]})
    
    def _setup_default_rules(self):
        """设置默认验证规则"""
        # 必需列检查
        self.validation_rules.append(
            ValidationRule(
                name="required_columns",
                validator=self._check_required_columns,
                error_message="缺少必需的列",
                severity="error"
            )
        )
        
        # 数据类型检查
        self.validation_rules.append(
            ValidationRule(
                name="column_types",
                validator=self._check_column_types,
                error_message="列数据类型不匹配",
                severity="error"
            )
        )
        
        # 数值范围检查
        self.validation_rules.append(
            ValidationRule(
                name="value_ranges",
                validator=self._check_value_ranges,
                error_message="数值超出允许范围",
                severity="warning"
            )
        )
        
        # 数据质量检查
        self.validation_rules.append(
            ValidationRule(
                name="data_quality",
                validator=self._check_data_quality,
                error_message="数据质量不达标",
                severity="warning"
            )
        )
        
        # 重复数据检查
        self.validation_rules.append(
            ValidationRule(
                name="duplicates",
                validator=self._check_duplicates,
                error_message="存在重复数据",
                severity="info"
            )
        )
    
    def _perform_validation(self, 
                          df: pd.DataFrame, 
                          validation_level: str,
                          custom_rules: List[ValidationRule],
                          skip_rules: List[str]) -> ValidationResult:
        """执行验证
        
        Args:
            df: 数据DataFrame
            validation_level: 验证级别
            custom_rules: 自定义规则
            skip_rules: 跳过的规则
            
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult()
        
        # 合并验证规则
        all_rules = self.validation_rules + custom_rules
        
        # 根据验证级别过滤规则
        if validation_level == 'basic':
            rules_to_run = [r for r in all_rules if r.severity == 'error']
        elif validation_level == 'standard':
            rules_to_run = [r for r in all_rules if r.severity in ['error', 'warning']]
        else:  # strict
            rules_to_run = all_rules
        
        # 过滤跳过的规则
        rules_to_run = [r for r in rules_to_run if r.name not in skip_rules]
        
        # 执行验证规则
        for rule in rules_to_run:
            try:
                rule_result = rule.validator(df)
                if not rule_result['passed']:
                    result.add_issue(
                        rule.name,
                        rule.error_message,
                        rule.severity,
                        rule_result.get('details', {})
                    )
            except Exception as e:
                self.logger.error(f"执行验证规则 {rule.name} 时发生错误: {e}")
                result.add_issue(
                    rule.name,
                    f"规则执行失败: {str(e)}",
                    'error'
                )
        
        # 添加统计信息
        result.stats = self._calculate_data_stats(df)
        
        return result
    
    def _check_required_columns(self, df: pd.DataFrame) -> Dict:
        """检查必需列
        
        Args:
            df: 数据DataFrame
            
        Returns:
            Dict: 检查结果
        """
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        
        return {
            'passed': len(missing_columns) == 0,
            'details': {
                'missing_columns': missing_columns,
                'required_columns': self.required_columns
            }
        }
    
    def _check_column_types(self, df: pd.DataFrame) -> Dict:
        """检查列数据类型
        
        Args:
            df: 数据DataFrame
            
        Returns:
            Dict: 检查结果
        """
        type_mismatches = []
        
        for column, expected_type in self.column_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if not self._is_compatible_type(actual_type, expected_type):
                    type_mismatches.append({
                        'column': column,
                        'expected': expected_type,
                        'actual': actual_type
                    })
        
        return {
            'passed': len(type_mismatches) == 0,
            'details': {
                'type_mismatches': type_mismatches
            }
        }
    
    def _check_value_ranges(self, df: pd.DataFrame) -> Dict:
        """检查数值范围
        
        Args:
            df: 数据DataFrame
            
        Returns:
            Dict: 检查结果
        """
        range_violations = []
        
        for column, range_config in self.value_ranges.items():
            if column in df.columns:
                min_val = range_config.get('min')
                max_val = range_config.get('max')
                
                violations = []
                
                if min_val is not None:
                    below_min = df[df[column] < min_val]
                    if not below_min.empty:
                        violations.extend(below_min.index.tolist())
                
                if max_val is not None:
                    above_max = df[df[column] > max_val]
                    if not above_max.empty:
                        violations.extend(above_max.index.tolist())
                
                if violations:
                    range_violations.append({
                        'column': column,
                        'range': range_config,
                        'violation_indices': violations,
                        'violation_count': len(violations)
                    })
        
        return {
            'passed': len(range_violations) == 0,
            'details': {
                'range_violations': range_violations
            }
        }
    
    def _check_data_quality(self, df: pd.DataFrame) -> Dict:
        """检查数据质量
        
        Args:
            df: 数据DataFrame
            
        Returns:
            Dict: 检查结果
        """
        quality_issues = []
        
        # 检查缺失率
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_rate > self.quality_thresholds['missing_rate']:
            quality_issues.append({
                'issue': 'high_missing_rate',
                'value': missing_rate,
                'threshold': self.quality_thresholds['missing_rate']
            })
        
        # 检查重复率
        duplicate_rate = df.duplicated().sum() / len(df)
        if duplicate_rate > self.quality_thresholds['duplicate_rate']:
            quality_issues.append({
                'issue': 'high_duplicate_rate',
                'value': duplicate_rate,
                'threshold': self.quality_thresholds['duplicate_rate']
            })
        
        return {
            'passed': len(quality_issues) == 0,
            'details': {
                'quality_issues': quality_issues,
                'missing_rate': missing_rate,
                'duplicate_rate': duplicate_rate
            }
        }
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """检查重复数据
        
        Args:
            df: 数据DataFrame
            
        Returns:
            Dict: 检查结果
        """
        duplicate_count = df.duplicated().sum()
        duplicate_indices = df[df.duplicated()].index.tolist()
        
        return {
            'passed': duplicate_count == 0,
            'details': {
                'duplicate_count': duplicate_count,
                'duplicate_indices': duplicate_indices,
                'total_records': len(df)
            }
        }
    
    def _is_compatible_type(self, actual_type: str, expected_type: str) -> bool:
        """检查数据类型是否兼容
        
        Args:
            actual_type: 实际类型
            expected_type: 期望类型
            
        Returns:
            bool: 是否兼容
        """
        type_mapping = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32', 'float16'],
            'string': ['object', 'string'],
            'datetime': ['datetime64[ns]', 'datetime64'],
            'bool': ['bool']
        }
        
        compatible_types = type_mapping.get(expected_type, [expected_type])
        return actual_type in compatible_types
    
    def _calculate_data_stats(self, df: pd.DataFrame) -> Dict:
        """计算数据统计信息
        
        Args:
            df: 数据DataFrame
            
        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'column_stats': {}
        }
        
        # 计算每列的统计信息
        for column in df.columns:
            col_stats = {
                'dtype': str(df[column].dtype),
                'missing_count': df[column].isnull().sum(),
                'unique_count': df[column].nunique()
            }
            
            # 数值列的额外统计
            if pd.api.types.is_numeric_dtype(df[column]):
                col_stats.update({
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max()
                })
            
            stats['column_stats'][column] = col_stats
        
        return stats
    
    def add_validation_rule(self, rule: ValidationRule) -> None:
        """添加验证规则
        
        Args:
            rule: 验证规则
        """
        self.validation_rules.append(rule)
        self.logger.info(f"添加验证规则: {rule.name}")
    
    def remove_validation_rule(self, rule_name: str) -> bool:
        """移除验证规则
        
        Args:
            rule_name: 规则名称
            
        Returns:
            bool: 是否成功移除
        """
        for i, rule in enumerate(self.validation_rules):
            if rule.name == rule_name:
                del self.validation_rules[i]
                self.logger.info(f"移除验证规则: {rule_name}")
                return True
        return False
    
    def get_validation_stats(self) -> Dict:
        """获取验证统计信息
        
        Returns:
            Dict: 验证统计信息
        """
        stats = self.validation_stats.copy()
        
        if stats['total_validations'] > 0:
            stats['success_rate'] = stats['passed_validations'] / stats['total_validations']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_validation_stats(self) -> None:
        """重置验证统计信息"""
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'total_records_validated': 0
        }
        self.logger.info("验证统计信息已重置")