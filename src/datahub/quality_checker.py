"""Data Quality Checker Module

数据质量检查器，用于监控和评估数据质量。

功能特性：
- 数据完整性检查
- 数据准确性验证
- 数据一致性检查
- 异常值检测
- 质量报告生成
"""

import time
import logging
import statistics
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from src.common.models.market import Ticker, Kline, OrderBook
from src.common.models import QualityLevel, CheckType
from src.common.exceptions.data import DataQualityError, DataValidationError


@dataclass
class QualityRule:
    """质量规则"""
    name: str
    check_type: CheckType
    description: str
    check_function: Callable
    threshold: float = 0.95  # 质量阈值
    severity: str = "warning"  # error, warning, info
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.check_type, str):
            self.check_type = CheckType(self.check_type)


@dataclass
class QualityResult:
    """质量检查结果"""
    rule_name: str
    check_type: CheckType
    passed: bool
    score: float  # 0.0 - 1.0
    threshold: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: float = 0.0
    
    @property
    def quality_level(self) -> QualityLevel:
        """根据分数确定质量等级"""
        if self.score >= 0.95:
            return QualityLevel.EXCELLENT
        elif self.score >= 0.85:
            return QualityLevel.GOOD
        elif self.score >= 0.70:
            return QualityLevel.FAIR
        elif self.score >= 0.50:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL


@dataclass
class QualityReport:
    """质量报告"""
    data_source: str
    check_time: datetime
    total_rules: int
    passed_rules: int
    failed_rules: int
    overall_score: float
    quality_level: QualityLevel
    results: List[QualityResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        """通过率"""
        if self.total_rules == 0:
            return 1.0
        return self.passed_rules / self.total_rules


class DataQualityChecker:
    """数据质量检查器
    
    用于监控和评估数据质量，支持多种质量检查规则。
    """
    
    def __init__(self, name: str = "DataQualityChecker", config: Optional[Dict] = None):
        """
        初始化数据质量检查器
        
        Args:
            name: 检查器名称
            config: 配置参数
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # 质量规则
        self.rules: Dict[str, QualityRule] = {}
        self.rule_groups: Dict[str, List[str]] = {}  # 规则分组
        
        # 检查历史
        self.check_history: List[QualityReport] = []
        self.max_history = self.config.get('max_history', 1000)
        
        # 统计信息
        self.stats = {
            'total_checks': 0,
            'total_rules_executed': 0,
            'total_rules_passed': 0,
            'total_rules_failed': 0,
            'avg_execution_time': 0.0,
            'last_check_time': None
        }
        
        # 配置参数
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5分钟
        self.parallel_execution = self.config.get('parallel_execution', False)
        self.max_workers = self.config.get('max_workers', 4)
        
        # 缓存
        self.result_cache: Dict[str, Tuple[QualityResult, datetime]] = {}
        
        # 初始化默认规则
        self._init_default_rules()
        
        self.logger.info(f"数据质量检查器 {name} 初始化完成")
    
    def _init_default_rules(self):
        """初始化默认质量规则"""
        
        # 完整性检查规则
        self.add_rule(QualityRule(
            name="null_check",
            check_type=CheckType.COMPLETENESS,
            description="检查空值比例",
            check_function=self._check_null_values,
            threshold=0.95,
            severity="warning"
        ))
        
        self.add_rule(QualityRule(
            name="missing_values_check",
            check_type=CheckType.COMPLETENESS,
            description="检查缺失值",
            check_function=self._check_missing_values,
            threshold=0.90,
            severity="warning"
        ))
        
        # 准确性检查规则
        self.add_rule(QualityRule(
            name="data_type_check",
            check_type=CheckType.ACCURACY,
            description="检查数据类型",
            check_function=self._check_data_types,
            threshold=1.0,
            severity="error"
        ))
        
        self.add_rule(QualityRule(
            name="range_check",
            check_type=CheckType.ACCURACY,
            description="检查数值范围",
            check_function=self._check_value_ranges,
            threshold=0.95,
            severity="warning"
        ))
        
        # 一致性检查规则
        self.add_rule(QualityRule(
            name="format_consistency_check",
            check_type=CheckType.CONSISTENCY,
            description="检查格式一致性",
            check_function=self._check_format_consistency,
            threshold=0.95,
            severity="warning"
        ))
        
        # 唯一性检查规则
        self.add_rule(QualityRule(
            name="duplicate_check",
            check_type=CheckType.UNIQUENESS,
            description="检查重复值",
            check_function=self._check_duplicates,
            threshold=0.95,
            severity="warning"
        ))
        
        # 及时性检查规则
        self.add_rule(QualityRule(
            name="timeliness_check",
            check_type=CheckType.TIMELINESS,
            description="检查数据时效性",
            check_function=self._check_timeliness,
            threshold=0.90,
            severity="warning"
        ))
        
        # 异常值检查规则
        self.add_rule(QualityRule(
            name="outlier_check",
            check_type=CheckType.OUTLIER,
            description="检查异常值",
            check_function=self._check_outliers,
            threshold=0.95,
            severity="info"
        ))
        
        # 业务规则检查
        self.add_rule(QualityRule(
            name="market_data_business_rules",
            check_type=CheckType.BUSINESS_RULE,
            description="检查市场数据业务规则",
            check_function=self._check_market_data_business_rules,
            threshold=0.95,
            severity="error"
        ))
    
    def add_rule(self, rule: QualityRule):
        """
        添加质量规则
        
        Args:
            rule: 质量规则
        """
        self.rules[rule.name] = rule
        self.logger.info(f"添加质量规则: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        移除质量规则
        
        Args:
            rule_name: 规则名称
            
        Returns:
            bool: 是否成功移除
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
            
            # 从分组中移除
            for group_rules in self.rule_groups.values():
                if rule_name in group_rules:
                    group_rules.remove(rule_name)
            
            self.logger.info(f"移除质量规则: {rule_name}")
            return True
        return False
    
    def add_rule_group(self, group_name: str, rule_names: List[str]):
        """
        添加规则分组
        
        Args:
            group_name: 分组名称
            rule_names: 规则名称列表
        """
        # 验证规则是否存在
        valid_rules = [name for name in rule_names if name in self.rules]
        if len(valid_rules) != len(rule_names):
            invalid_rules = set(rule_names) - set(valid_rules)
            self.logger.warning(f"分组 {group_name} 中包含无效规则: {invalid_rules}")
        
        self.rule_groups[group_name] = valid_rules
        self.logger.info(f"添加规则分组: {group_name}, 包含 {len(valid_rules)} 个规则")
    
    def check_quality(self, data: Any, rules: Optional[List[str]] = None, 
                     group: Optional[str] = None) -> QualityReport:
        """
        检查数据质量
        
        Args:
            data: 待检查的数据
            rules: 指定的规则列表
            group: 规则分组名称
            
        Returns:
            QualityReport: 质量报告
        """
        start_time = time.time()
        check_time = datetime.now()
        
        # 确定要执行的规则
        if group and group in self.rule_groups:
            rule_names = self.rule_groups[group]
        elif rules:
            rule_names = [name for name in rules if name in self.rules]
        else:
            rule_names = list(self.rules.keys())
        
        # 过滤启用的规则
        enabled_rules = [name for name in rule_names if self.rules[name].enabled]
        
        if not enabled_rules:
            self.logger.warning("没有启用的质量规则")
            return QualityReport(
                data_source=str(type(data).__name__),
                check_time=check_time,
                total_rules=0,
                passed_rules=0,
                failed_rules=0,
                overall_score=1.0,
                quality_level=QualityLevel.EXCELLENT,
                results=[]
            )
        
        # 执行质量检查
        results = []
        for rule_name in enabled_rules:
            try:
                result = self._execute_rule(rule_name, data)
                results.append(result)
            except Exception as e:
                self.logger.error(f"执行规则 {rule_name} 失败: {e}")
                # 创建失败结果
                results.append(QualityResult(
                    rule_name=rule_name,
                    check_type=self.rules[rule_name].check_type,
                    passed=False,
                    score=0.0,
                    threshold=self.rules[rule_name].threshold,
                    message=f"规则执行失败: {str(e)}",
                    severity="error"
                ))
        
        # 计算总体质量分数
        if results:
            overall_score = sum(result.score for result in results) / len(results)
        else:
            overall_score = 1.0
        
        # 统计通过和失败的规则
        passed_rules = sum(1 for result in results if result.passed)
        failed_rules = len(results) - passed_rules
        
        # 确定质量等级
        if overall_score >= 0.95:
            quality_level = QualityLevel.EXCELLENT
        elif overall_score >= 0.85:
            quality_level = QualityLevel.GOOD
        elif overall_score >= 0.70:
            quality_level = QualityLevel.FAIR
        elif overall_score >= 0.50:
            quality_level = QualityLevel.POOR
        else:
            quality_level = QualityLevel.CRITICAL
        
        # 生成摘要
        summary = self._generate_summary(results)
        
        # 生成建议
        recommendations = self._generate_recommendations(results)
        
        # 创建质量报告
        report = QualityReport(
            data_source=str(type(data).__name__),
            check_time=check_time,
            total_rules=len(results),
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            overall_score=overall_score,
            quality_level=quality_level,
            results=results,
            summary=summary,
            recommendations=recommendations
        )
        
        # 更新统计信息
        execution_time = time.time() - start_time
        self._update_stats(len(results), passed_rules, failed_rules, execution_time)
        
        # 保存到历史记录
        self._save_to_history(report)
        
        self.logger.info(f"质量检查完成，总分: {overall_score:.2f}, 等级: {quality_level.value}")
        
        return report
    
    def _execute_rule(self, rule_name: str, data: Any) -> QualityResult:
        """
        执行单个质量规则
        
        Args:
            rule_name: 规则名称
            data: 数据
            
        Returns:
            QualityResult: 检查结果
        """
        rule = self.rules[rule_name]
        
        # 检查缓存
        if self.enable_caching:
            cache_key = f"{rule_name}_{hash(str(data))}"
            if cache_key in self.result_cache:
                cached_result, cache_time = self.result_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                    return cached_result
        
        start_time = time.time()
        
        try:
            # 执行检查函数
            score, message, details = rule.check_function(data, rule)
            
            # 确保分数在有效范围内
            score = max(0.0, min(1.0, score))
            
            # 判断是否通过
            passed = score >= rule.threshold
            
            execution_time = time.time() - start_time
            
            result = QualityResult(
                rule_name=rule_name,
                check_type=rule.check_type,
                passed=passed,
                score=score,
                threshold=rule.threshold,
                message=message,
                details=details,
                severity=rule.severity,
                execution_time=execution_time
            )
            
            # 更新缓存
            if self.enable_caching:
                self.result_cache[cache_key] = (result, datetime.now())
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return QualityResult(
                rule_name=rule_name,
                check_type=rule.check_type,
                passed=False,
                score=0.0,
                threshold=rule.threshold,
                message=f"执行失败: {str(e)}",
                severity="error",
                execution_time=execution_time
            )
    
    # 默认检查函数实现
    
    def _check_null_values(self, data: Any, rule: QualityRule) -> Tuple[float, str, Dict]:
        """
        检查空值比例
        
        Args:
            data: 数据
            rule: 规则
            
        Returns:
            Tuple[float, str, Dict]: (分数, 消息, 详情)
        """
        if isinstance(data, pd.DataFrame):
            total_cells = data.size
            null_cells = data.isnull().sum().sum()
            null_ratio = null_cells / total_cells if total_cells > 0 else 0
            
            score = 1.0 - null_ratio
            message = f"空值比例: {null_ratio:.2%}"
            details = {
                'total_cells': total_cells,
                'null_cells': null_cells,
                'null_ratio': null_ratio,
                'null_columns': data.isnull().sum().to_dict()
            }
            
        elif isinstance(data, (list, tuple)):
            total_items = len(data)
            null_items = sum(1 for item in data if item is None)
            null_ratio = null_items / total_items if total_items > 0 else 0
            
            score = 1.0 - null_ratio
            message = f"空值比例: {null_ratio:.2%}"
            details = {
                'total_items': total_items,
                'null_items': null_items,
                'null_ratio': null_ratio
            }
            
        else:
            # 单个值检查
            is_null = data is None
            score = 0.0 if is_null else 1.0
            message = "值为空" if is_null else "值不为空"
            details = {'is_null': is_null}
        
        return score, message, details
    
    def _check_missing_values(self, data: Any, rule: QualityRule) -> Tuple[float, str, Dict]:
        """
        检查缺失值
        
        Args:
            data: 数据
            rule: 规则
            
        Returns:
            Tuple[float, str, Dict]: (分数, 消息, 详情)
        """
        if isinstance(data, pd.DataFrame):
            total_rows = len(data)
            complete_rows = len(data.dropna())
            completeness_ratio = complete_rows / total_rows if total_rows > 0 else 1.0
            
            score = completeness_ratio
            message = f"数据完整性: {completeness_ratio:.2%}"
            details = {
                'total_rows': total_rows,
                'complete_rows': complete_rows,
                'incomplete_rows': total_rows - complete_rows,
                'completeness_ratio': completeness_ratio
            }
            
        elif isinstance(data, (list, tuple)):
            if data and hasattr(data[0], '__dict__'):
                # 对象列表
                total_objects = len(data)
                complete_objects = 0
                
                for obj in data:
                    if all(getattr(obj, attr, None) is not None for attr in vars(obj)):
                        complete_objects += 1
                
                completeness_ratio = complete_objects / total_objects if total_objects > 0 else 1.0
                
                score = completeness_ratio
                message = f"对象完整性: {completeness_ratio:.2%}"
                details = {
                    'total_objects': total_objects,
                    'complete_objects': complete_objects,
                    'completeness_ratio': completeness_ratio
                }
            else:
                # 简单列表
                return self._check_null_values(data, rule)
        else:
            # 单个对象
            if hasattr(data, '__dict__'):
                attrs = vars(data)
                total_attrs = len(attrs)
                complete_attrs = sum(1 for value in attrs.values() if value is not None)
                completeness_ratio = complete_attrs / total_attrs if total_attrs > 0 else 1.0
                
                score = completeness_ratio
                message = f"属性完整性: {completeness_ratio:.2%}"
                details = {
                    'total_attributes': total_attrs,
                    'complete_attributes': complete_attrs,
                    'completeness_ratio': completeness_ratio
                }
            else:
                return self._check_null_values(data, rule)
        
        return score, message, details
    
    def _check_data_types(self, data: Any, rule: QualityRule) -> Tuple[float, str, Dict]:
        """
        检查数据类型
        
        Args:
            data: 数据
            rule: 规则
            
        Returns:
            Tuple[float, str, Dict]: (分数, 消息, 详情)
        """
        if isinstance(data, pd.DataFrame):
            # 检查DataFrame列的数据类型一致性
            type_consistency = {}
            total_columns = len(data.columns)
            consistent_columns = 0
            
            for column in data.columns:
                series = data[column].dropna()
                if len(series) > 0:
                    # 检查类型一致性
                    first_type = type(series.iloc[0])
                    consistent = all(isinstance(val, first_type) for val in series)
                    type_consistency[column] = {
                        'consistent': consistent,
                        'expected_type': first_type.__name__,
                        'sample_types': list(set(type(val).__name__ for val in series.head(10)))
                    }
                    if consistent:
                        consistent_columns += 1
                else:
                    type_consistency[column] = {
                        'consistent': True,
                        'expected_type': 'unknown',
                        'sample_types': []
                    }
                    consistent_columns += 1
            
            score = consistent_columns / total_columns if total_columns > 0 else 1.0
            message = f"类型一致性: {score:.2%}"
            details = {
                'total_columns': total_columns,
                'consistent_columns': consistent_columns,
                'type_consistency': type_consistency
            }
            
        elif isinstance(data, (list, tuple)):
            if not data:
                score = 1.0
                message = "空列表，类型检查通过"
                details = {'empty_list': True}
            else:
                # 检查列表元素类型一致性
                first_type = type(data[0])
                consistent_items = sum(1 for item in data if isinstance(item, first_type))
                total_items = len(data)
                
                score = consistent_items / total_items
                message = f"类型一致性: {score:.2%}"
                details = {
                    'total_items': total_items,
                    'consistent_items': consistent_items,
                    'expected_type': first_type.__name__,
                    'type_distribution': {}
                }
                
                # 统计类型分布
                type_counts = {}
                for item in data:
                    type_name = type(item).__name__
                    type_counts[type_name] = type_counts.get(type_name, 0) + 1
                details['type_distribution'] = type_counts
        
        else:
            # 单个值，总是通过
            score = 1.0
            message = f"单个值类型: {type(data).__name__}"
            details = {'data_type': type(data).__name__}
        
        return score, message, details
    
    def _check_value_ranges(self, data: Any, rule: QualityRule) -> Tuple[float, str, Dict]:
        """
        检查数值范围
        
        Args:
            data: 数据
            rule: 规则
            
        Returns:
            Tuple[float, str, Dict]: (分数, 消息, 详情)
        """
        # 从规则元数据中获取范围配置
        range_config = rule.metadata.get('ranges', {})
        
        if isinstance(data, pd.DataFrame):
            total_values = 0
            valid_values = 0
            range_violations = {}
            
            for column in data.columns:
                if column in range_config:
                    min_val, max_val = range_config[column]
                    series = data[column].dropna()
                    
                    if pd.api.types.is_numeric_dtype(series):
                        column_total = len(series)
                        column_valid = len(series[(series >= min_val) & (series <= max_val)])
                        
                        total_values += column_total
                        valid_values += column_valid
                        
                        if column_valid < column_total:
                            range_violations[column] = {
                                'expected_range': [min_val, max_val],
                                'violations': column_total - column_valid,
                                'min_found': float(series.min()),
                                'max_found': float(series.max())
                            }
            
            score = valid_values / total_values if total_values > 0 else 1.0
            message = f"范围有效性: {score:.2%}"
            details = {
                'total_values': total_values,
                'valid_values': valid_values,
                'range_violations': range_violations
            }
            
        elif isinstance(data, (list, tuple)):
            if 'default_range' in range_config:
                min_val, max_val = range_config['default_range']
                numeric_data = [x for x in data if isinstance(x, (int, float))]
                
                if numeric_data:
                    total_values = len(numeric_data)
                    valid_values = sum(1 for x in numeric_data if min_val <= x <= max_val)
                    
                    score = valid_values / total_values
                    message = f"范围有效性: {score:.2%}"
                    details = {
                        'total_values': total_values,
                        'valid_values': valid_values,
                        'expected_range': [min_val, max_val],
                        'actual_range': [min(numeric_data), max(numeric_data)]
                    }
                else:
                    score = 1.0
                    message = "无数值数据"
                    details = {'no_numeric_data': True}
            else:
                score = 1.0
                message = "未配置范围检查"
                details = {'no_range_config': True}
        
        else:
            # 单个值
            if 'default_range' in range_config and isinstance(data, (int, float)):
                min_val, max_val = range_config['default_range']
                in_range = min_val <= data <= max_val
                
                score = 1.0 if in_range else 0.0
                message = f"值 {data} {'在' if in_range else '不在'}范围 [{min_val}, {max_val}] 内"
                details = {
                    'value': data,
                    'expected_range': [min_val, max_val],
                    'in_range': in_range
                }
            else:
                score = 1.0
                message = "无范围限制或非数值类型"
                details = {'no_range_check': True}
        
        return score, message, details
    
    def _check_format_consistency(self, data: Any, rule: QualityRule) -> Tuple[float, str, Dict]:
        """
        检查格式一致性
        
        Args:
            data: 数据
            rule: 规则
            
        Returns:
            Tuple[float, str, Dict]: (分数, 消息, 详情)
        """
        import re
        
        # 从规则元数据中获取格式模式
        format_patterns = rule.metadata.get('patterns', {})
        
        if isinstance(data, pd.DataFrame):
            total_values = 0
            valid_values = 0
            format_violations = {}
            
            for column in data.columns:
                if column in format_patterns:
                    pattern = format_patterns[column]
                    series = data[column].dropna().astype(str)
                    
                    column_total = len(series)
                    column_valid = sum(1 for val in series if re.match(pattern, val))
                    
                    total_values += column_total
                    valid_values += column_valid
                    
                    if column_valid < column_total:
                        invalid_samples = [val for val in series if not re.match(pattern, val)][:5]
                        format_violations[column] = {
                            'expected_pattern': pattern,
                            'violations': column_total - column_valid,
                            'invalid_samples': invalid_samples
                        }
            
            score = valid_values / total_values if total_values > 0 else 1.0
            message = f"格式一致性: {score:.2%}"
            details = {
                'total_values': total_values,
                'valid_values': valid_values,
                'format_violations': format_violations
            }
            
        elif isinstance(data, (list, tuple)):
            if 'default_pattern' in format_patterns:
                pattern = format_patterns['default_pattern']
                string_data = [str(x) for x in data]
                
                total_values = len(string_data)
                valid_values = sum(1 for val in string_data if re.match(pattern, val))
                
                score = valid_values / total_values if total_values > 0 else 1.0
                message = f"格式一致性: {score:.2%}"
                details = {
                    'total_values': total_values,
                    'valid_values': valid_values,
                    'expected_pattern': pattern
                }
            else:
                score = 1.0
                message = "未配置格式检查"
                details = {'no_format_config': True}
        
        else:
            # 单个值
            if 'default_pattern' in format_patterns:
                pattern = format_patterns['default_pattern']
                matches = bool(re.match(pattern, str(data)))
                
                score = 1.0 if matches else 0.0
                message = f"格式 {'匹配' if matches else '不匹配'} 模式 {pattern}"
                details = {
                    'value': str(data),
                    'pattern': pattern,
                    'matches': matches
                }
            else:
                score = 1.0
                message = "无格式要求"
                details = {'no_format_check': True}
        
        return score, message, details
    
    def _check_duplicates(self, data: Any, rule: QualityRule) -> Tuple[float, str, Dict]:
        """
        检查重复值
        
        Args:
            data: 数据
            rule: 规则
            
        Returns:
            Tuple[float, str, Dict]: (分数, 消息, 详情)
        """
        if isinstance(data, pd.DataFrame):
            total_rows = len(data)
            unique_rows = len(data.drop_duplicates())
            duplicate_rows = total_rows - unique_rows
            
            uniqueness_ratio = unique_rows / total_rows if total_rows > 0 else 1.0
            
            score = uniqueness_ratio
            message = f"唯一性: {uniqueness_ratio:.2%}"
            details = {
                'total_rows': total_rows,
                'unique_rows': unique_rows,
                'duplicate_rows': duplicate_rows,
                'uniqueness_ratio': uniqueness_ratio
            }
            
        elif isinstance(data, (list, tuple)):
            total_items = len(data)
            unique_items = len(set(str(item) for item in data))  # 转换为字符串以处理不可哈希类型
            duplicate_items = total_items - unique_items
            
            uniqueness_ratio = unique_items / total_items if total_items > 0 else 1.0
            
            score = uniqueness_ratio
            message = f"唯一性: {uniqueness_ratio:.2%}"
            details = {
                'total_items': total_items,
                'unique_items': unique_items,
                'duplicate_items': duplicate_items,
                'uniqueness_ratio': uniqueness_ratio
            }
        
        else:
            # 单个值，总是唯一
            score = 1.0
            message = "单个值，唯一性检查通过"
            details = {'single_value': True}
        
        return score, message, details
    
    def _check_timeliness(self, data: Any, rule: QualityRule) -> Tuple[float, str, Dict]:
        """
        检查数据时效性
        
        Args:
            data: 数据
            rule: 规则
            
        Returns:
            Tuple[float, str, Dict]: (分数, 消息, 详情)
        """
        # 从规则元数据中获取时效性配置
        max_age_seconds = rule.metadata.get('max_age_seconds', 3600)  # 默认1小时
        timestamp_field = rule.metadata.get('timestamp_field', 'timestamp')
        
        current_time = datetime.now()
        
        if isinstance(data, pd.DataFrame):
            if timestamp_field in data.columns:
                timestamps = pd.to_datetime(data[timestamp_field], errors='coerce')
                valid_timestamps = timestamps.dropna()
                
                if len(valid_timestamps) > 0:
                    # 计算数据年龄
                    ages = [(current_time - ts).total_seconds() for ts in valid_timestamps]
                    timely_count = sum(1 for age in ages if age <= max_age_seconds)
                    
                    timeliness_ratio = timely_count / len(valid_timestamps)
                    
                    score = timeliness_ratio
                    message = f"时效性: {timeliness_ratio:.2%}"
                    details = {
                        'total_records': len(valid_timestamps),
                        'timely_records': timely_count,
                        'max_age_seconds': max_age_seconds,
                        'oldest_age_seconds': max(ages),
                        'newest_age_seconds': min(ages),
                        'timeliness_ratio': timeliness_ratio
                    }
                else:
                    score = 0.0
                    message = "无有效时间戳"
                    details = {'no_valid_timestamps': True}
            else:
                score = 1.0
                message = f"未找到时间戳字段 {timestamp_field}"
                details = {'timestamp_field_missing': True}
        
        elif hasattr(data, timestamp_field):
            # 单个对象
            timestamp = getattr(data, timestamp_field)
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    
                    age_seconds = (current_time - timestamp).total_seconds()
                    is_timely = age_seconds <= max_age_seconds
                    
                    score = 1.0 if is_timely else 0.0
                    message = f"数据年龄: {age_seconds:.0f}秒, {'及时' if is_timely else '过期'}"
                    details = {
                        'age_seconds': age_seconds,
                        'max_age_seconds': max_age_seconds,
                        'is_timely': is_timely,
                        'timestamp': timestamp.isoformat()
                    }
                except Exception as e:
                    score = 0.0
                    message = f"时间戳解析失败: {e}"
                    details = {'timestamp_parse_error': str(e)}
            else:
                score = 0.0
                message = "时间戳为空"
                details = {'empty_timestamp': True}
        
        else:
            score = 1.0
            message = "无时效性要求"
            details = {'no_timeliness_check': True}
        
        return score, message, details
    
    def _check_outliers(self, data: Any, rule: QualityRule) -> Tuple[float, str, Dict]:
        """
        检查异常值
        
        Args:
            data: 数据
            rule: 规则
            
        Returns:
            Tuple[float, str, Dict]: (分数, 消息, 详情)
        """
        # 从规则元数据中获取异常值检测配置
        method = rule.metadata.get('method', 'iqr')  # iqr, zscore, isolation_forest
        threshold = rule.metadata.get('threshold', 3.0)
        
        if isinstance(data, pd.DataFrame):
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                score = 1.0
                message = "无数值列"
                details = {'no_numeric_columns': True}
                return score, message, details
            
            total_values = 0
            normal_values = 0
            outlier_details = {}
            
            for column in numeric_columns:
                series = data[column].dropna()
                if len(series) < 3:  # 需要至少3个值来检测异常值
                    continue
                
                column_total = len(series)
                total_values += column_total
                
                if method == 'iqr':
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    normal_count = column_total - len(outliers)
                    
                elif method == 'zscore':
                    z_scores = np.abs((series - series.mean()) / series.std())
                    outliers = series[z_scores > threshold]
                    normal_count = column_total - len(outliers)
                
                else:  # 默认使用IQR
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    normal_count = column_total - len(outliers)
                
                normal_values += normal_count
                
                if len(outliers) > 0:
                    outlier_details[column] = {
                        'outlier_count': len(outliers),
                        'outlier_ratio': len(outliers) / column_total,
                        'outlier_values': outliers.tolist()[:10]  # 最多显示10个异常值
                    }
            
            score = normal_values / total_values if total_values > 0 else 1.0
            message = f"正常值比例: {score:.2%}"
            details = {
                'total_values': total_values,
                'normal_values': normal_values,
                'outlier_values': total_values - normal_values,
                'method': method,
                'outlier_details': outlier_details
            }
            
        elif isinstance(data, (list, tuple)):
            numeric_data = [x for x in data if isinstance(x, (int, float))]
            
            if len(numeric_data) < 3:
                score = 1.0
                message = "数据点太少，无法检测异常值"
                details = {'insufficient_data': True}
                return score, message, details
            
            if method == 'iqr':
                Q1 = np.percentile(numeric_data, 25)
                Q3 = np.percentile(numeric_data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = [x for x in numeric_data if x < lower_bound or x > upper_bound]
                
            elif method == 'zscore':
                mean_val = np.mean(numeric_data)
                std_val = np.std(numeric_data)
                z_scores = [abs((x - mean_val) / std_val) for x in numeric_data]
                outliers = [numeric_data[i] for i, z in enumerate(z_scores) if z > threshold]
            
            else:  # 默认使用IQR
                Q1 = np.percentile(numeric_data, 25)
                Q3 = np.percentile(numeric_data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = [x for x in numeric_data if x < lower_bound or x > upper_bound]
            
            normal_count = len(numeric_data) - len(outliers)
            score = normal_count / len(numeric_data)
            message = f"正常值比例: {score:.2%}"
            details = {
                'total_values': len(numeric_data),
                'normal_values': normal_count,
                'outlier_values': len(outliers),
                'method': method,
                'outliers': outliers[:10]  # 最多显示10个异常值
            }
        
        else:
            # 单个值无法检测异常值
            score = 1.0
            message = "单个值，无法检测异常值"
            details = {'single_value': True}
        
        return score, message, details
    
    def _check_market_data_business_rules(self, data: Any, rule: QualityRule) -> Tuple[float, str, Dict]:
        """
        检查市场数据业务规则
        
        Args:
            data: 数据
            rule: 规则
            
        Returns:
            Tuple[float, str, Dict]: (分数, 消息, 详情)
        """
        violations = []
        total_checks = 0
        passed_checks = 0
        
        if isinstance(data, Ticker):
            total_checks += 5
            
            # 检查价格为正数
            if data.price and data.price > 0:
                passed_checks += 1
            else:
                violations.append(f"价格无效: {data.price}")
            
            # 检查成交量为非负数
            if data.volume is not None and data.volume >= 0:
                passed_checks += 1
            else:
                violations.append(f"成交量无效: {data.volume}")
            
            # 检查买卖价差合理性
            if data.bid_price and data.ask_price and data.bid_price <= data.ask_price:
                passed_checks += 1
            else:
                violations.append(f"买卖价差异常: bid={data.bid_price}, ask={data.ask_price}")
            
            # 检查时间戳
            if data.timestamp:
                age = (datetime.now() - data.timestamp).total_seconds()
                if age <= 3600:  # 1小时内
                    passed_checks += 1
                else:
                    violations.append(f"数据过期: {age}秒")
            else:
                violations.append("缺少时间戳")
            
            # 检查交易对格式
            if data.symbol and '/' in data.symbol:
                passed_checks += 1
            else:
                violations.append(f"交易对格式错误: {data.symbol}")
        
        elif isinstance(data, Kline):
            total_checks += 6
            
            # 检查OHLC价格关系
            if (data.open_price and data.high_price and data.low_price and data.close_price and
                data.low_price <= data.open_price <= data.high_price and
                data.low_price <= data.close_price <= data.high_price):
                passed_checks += 1
            else:
                violations.append(f"OHLC价格关系异常: O={data.open_price}, H={data.high_price}, L={data.low_price}, C={data.close_price}")
            
            # 检查成交量为非负数
            if data.volume is not None and data.volume >= 0:
                passed_checks += 1
            else:
                violations.append(f"成交量无效: {data.volume}")
            
            # 检查时间戳
            if data.timestamp:
                passed_checks += 1
            else:
                violations.append("缺少时间戳")
            
            # 检查交易对格式
            if data.symbol and '/' in data.symbol:
                passed_checks += 1
            else:
                violations.append(f"交易对格式错误: {data.symbol}")
            
            # 检查时间间隔
            if data.interval:
                passed_checks += 1
            else:
                violations.append("缺少时间间隔")
            
            # 检查价格变化合理性（不超过20%）
            if (data.open_price and data.close_price and
                abs(data.close_price - data.open_price) / data.open_price <= 0.20):
                passed_checks += 1
            else:
                violations.append(f"价格变化过大: {abs(data.close_price - data.open_price) / data.open_price:.2%}")
        
        elif isinstance(data, OrderBook):
            total_checks += 4
            
            # 检查买卖盘数据
            if data.bids and data.asks:
                passed_checks += 1
            else:
                violations.append("缺少买卖盘数据")
            
            # 检查价格排序
            if data.bids and all(data.bids[i][0] >= data.bids[i+1][0] for i in range(len(data.bids)-1)):
                passed_checks += 1
            else:
                violations.append("买盘价格排序错误")
            
            if data.asks and all(data.asks[i][0] <= data.asks[i+1][0] for i in range(len(data.asks)-1)):
                passed_checks += 1
            else:
                violations.append("卖盘价格排序错误")
            
            # 检查时间戳
            if data.timestamp:
                passed_checks += 1
            else:
                violations.append("缺少时间戳")
        
        elif isinstance(data, (list, tuple)):
            # 批量数据检查
            for i, item in enumerate(data):
                if isinstance(item, (Ticker, Kline, OrderBook)):
                    item_score, item_message, item_details = self._check_market_data_business_rules(item, rule)
                    item_total = item_details.get('total_checks', 1)
                    item_passed = item_details.get('passed_checks', 1 if item_score > 0.5 else 0)
                    
                    total_checks += item_total
                    passed_checks += item_passed
                    
                    if item_score < 1.0:
                        violations.extend([f"项目{i}: {v}" for v in item_details.get('violations', [])])
        
        else:
            # 非市场数据，跳过业务规则检查
            total_checks = 1
            passed_checks = 1
        
        score = passed_checks / total_checks if total_checks > 0 else 1.0
        message = f"业务规则通过率: {score:.2%}"
        details = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'violations': violations
        }
        
        return score, message, details
    
    def _generate_summary(self, results: List[QualityResult]) -> Dict[str, Any]:
        """
        生成质量检查摘要
        
        Args:
            results: 检查结果列表
            
        Returns:
            Dict[str, Any]: 摘要信息
        """
        if not results:
            return {}
        
        # 按检查类型分组统计
        type_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'avg_score': 0.0})
        
        for result in results:
            check_type = result.check_type.value
            type_stats[check_type]['total'] += 1
            if result.passed:
                type_stats[check_type]['passed'] += 1
        
        # 计算平均分数
        for check_type in type_stats:
            type_results = [r for r in results if r.check_type.value == check_type]
            if type_results:
                type_stats[check_type]['avg_score'] = sum(r.score for r in type_results) / len(type_results)
        
        # 按严重程度统计
        severity_stats = defaultdict(int)
        for result in results:
            if not result.passed:
                severity_stats[result.severity] += 1
        
        # 执行时间统计
        execution_times = [r.execution_time for r in results if r.execution_time > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        
        return {
            'check_type_stats': dict(type_stats),
            'severity_stats': dict(severity_stats),
            'avg_execution_time': avg_execution_time,
            'total_execution_time': sum(execution_times),
            'fastest_check': min(execution_times) if execution_times else 0.0,
            'slowest_check': max(execution_times) if execution_times else 0.0
        }
    
    def _generate_recommendations(self, results: List[QualityResult]) -> List[str]:
        """
        生成改进建议
        
        Args:
            results: 检查结果列表
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 分析失败的检查
        failed_results = [r for r in results if not r.passed]
        
        if not failed_results:
            recommendations.append("数据质量优秀，继续保持！")
            return recommendations
        
        # 按检查类型分组失败结果
        failed_by_type = defaultdict(list)
        for result in failed_results:
            failed_by_type[result.check_type].append(result)
        
        # 生成针对性建议
        for check_type, failed_list in failed_by_type.items():
            if check_type == CheckType.COMPLETENESS:
                recommendations.append("建议：完善数据收集流程，减少缺失值")
            elif check_type == CheckType.ACCURACY:
                recommendations.append("建议：加强数据验证，确保数据类型和格式正确")
            elif check_type == CheckType.CONSISTENCY:
                recommendations.append("建议：统一数据格式标准，保持一致性")
            elif check_type == CheckType.UNIQUENESS:
                recommendations.append("建议：实施去重机制，避免重复数据")
            elif check_type == CheckType.TIMELINESS:
                recommendations.append("建议：优化数据更新频率，确保数据时效性")
            elif check_type == CheckType.OUTLIER:
                recommendations.append("建议：建立异常值监控机制，及时发现数据异常")
            elif check_type == CheckType.BUSINESS_RULE:
                recommendations.append("建议：加强业务规则验证，确保数据符合业务逻辑")
        
        # 根据严重程度添加建议
        error_count = sum(1 for r in failed_results if r.severity == "error")
        warning_count = sum(1 for r in failed_results if r.severity == "warning")
        
        if error_count > 0:
            recommendations.append(f"紧急：发现 {error_count} 个严重错误，需要立即处理")
        
        if warning_count > 0:
            recommendations.append(f"注意：发现 {warning_count} 个警告，建议尽快处理")
        
        # 根据整体质量分数添加建议
        avg_score = sum(r.score for r in results) / len(results)
        if avg_score < 0.7:
            recommendations.append("数据质量较差，建议全面检查数据处理流程")
        elif avg_score < 0.85:
            recommendations.append("数据质量一般，建议重点关注失败的检查项")
        
        return recommendations
    
    def _update_stats(self, total_rules: int, passed_rules: int, failed_rules: int, execution_time: float):
        """
        更新统计信息
        
        Args:
            total_rules: 总规则数
            passed_rules: 通过规则数
            failed_rules: 失败规则数
            execution_time: 执行时间
        """
        self.stats['total_checks'] += 1
        self.stats['total_rules_executed'] += total_rules
        self.stats['total_rules_passed'] += passed_rules
        self.stats['total_rules_failed'] += failed_rules
        self.stats['last_check_time'] = datetime.now().isoformat()
        
        # 更新平均执行时间
        current_avg = self.stats['avg_execution_time']
        total_checks = self.stats['total_checks']
        self.stats['avg_execution_time'] = (
            (current_avg * (total_checks - 1) + execution_time) / total_checks
        )
    
    def _save_to_history(self, report: QualityReport):
        """
        保存报告到历史记录
        
        Args:
            report: 质量报告
        """
        self.check_history.append(report)
        
        # 限制历史记录数量
        if len(self.check_history) > self.max_history:
            self.check_history = self.check_history[-self.max_history:]
    
    def get_history(self, limit: Optional[int] = None) -> List[QualityReport]:
        """
        获取检查历史
        
        Args:
            limit: 限制返回数量
            
        Returns:
            List[QualityReport]: 历史报告列表
        """
        if limit:
            return self.check_history[-limit:]
        return self.check_history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.stats.copy()
        
        # 添加计算字段
        if stats['total_rules_executed'] > 0:
            stats['overall_pass_rate'] = stats['total_rules_passed'] / stats['total_rules_executed']
        else:
            stats['overall_pass_rate'] = 1.0
        
        # 添加规则统计
        stats['total_rules_configured'] = len(self.rules)
        stats['enabled_rules'] = sum(1 for rule in self.rules.values() if rule.enabled)
        stats['disabled_rules'] = len(self.rules) - stats['enabled_rules']
        
        return stats
    
    def reset_stats(self):
        """
        重置统计信息
        """
        self.stats = {
            'total_checks': 0,
            'total_rules_executed': 0,
            'total_rules_passed': 0,
            'total_rules_failed': 0,
            'avg_execution_time': 0.0,
            'last_check_time': None
        }
        self.logger.info("统计信息已重置")
    
    def clear_cache(self):
        """
        清空缓存
        """
        self.result_cache.clear()
        self.logger.info("缓存已清空")
    
    def export_rules(self) -> Dict[str, Any]:
        """
        导出规则配置
        
        Returns:
            Dict[str, Any]: 规则配置
        """
        rules_config = {}
        
        for name, rule in self.rules.items():
            rules_config[name] = {
                'check_type': rule.check_type.value,
                'description': rule.description,
                'threshold': rule.threshold,
                'severity': rule.severity,
                'enabled': rule.enabled,
                'tags': rule.tags,
                'metadata': rule.metadata
            }
        
        return {
            'rules': rules_config,
            'rule_groups': self.rule_groups,
            'config': self.config
        }
    
    def import_rules(self, rules_config: Dict[str, Any]):
        """
        导入规则配置
        
        Args:
            rules_config: 规则配置
        """
        if 'rules' in rules_config:
            for name, config in rules_config['rules'].items():
                # 创建检查函数（这里需要根据实际情况映射）
                check_function = self._get_check_function(config['check_type'])
                
                rule = QualityRule(
                    name=name,
                    check_type=CheckType(config['check_type']),
                    description=config['description'],
                    check_function=check_function,
                    threshold=config.get('threshold', 0.95),
                    severity=config.get('severity', 'warning'),
                    enabled=config.get('enabled', True),
                    tags=config.get('tags', []),
                    metadata=config.get('metadata', {})
                )
                
                self.add_rule(rule)
        
        if 'rule_groups' in rules_config:
            self.rule_groups.update(rules_config['rule_groups'])
        
        if 'config' in rules_config:
            self.config.update(rules_config['config'])
        
        self.logger.info(f"导入了 {len(rules_config.get('rules', {}))} 个规则")
    
    def _get_check_function(self, check_type: str) -> Callable:
        """
        根据检查类型获取对应的检查函数
        
        Args:
            check_type: 检查类型
            
        Returns:
            Callable: 检查函数
        """
        function_map = {
            'completeness': self._check_null_values,
            'accuracy': self._check_data_types,
            'consistency': self._check_format_consistency,
            'validity': self._check_data_types,
            'uniqueness': self._check_duplicates,
            'timeliness': self._check_timeliness,
            'outlier': self._check_outliers,
            'schema': self._check_data_types,
            'business_rule': self._check_market_data_business_rules
        }
        
        return function_map.get(check_type, self._check_null_values)
    
    def generate_report_html(self, report: QualityReport) -> str:
        """
        生成HTML格式的质量报告
        
        Args:
            report: 质量报告
            
        Returns:
            str: HTML报告
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>数据质量报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .results {{ margin: 20px 0; }}
                .result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .passed {{ border-left-color: #4CAF50; }}
                .failed {{ border-left-color: #f44336; }}
                .score {{ font-weight: bold; }}
                .recommendations {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>数据质量报告</h1>
                <p><strong>数据源:</strong> {report.data_source}</p>
                <p><strong>检查时间:</strong> {report.check_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>总体评分:</strong> <span class="score">{report.overall_score:.2%}</span></p>
                <p><strong>质量等级:</strong> {report.quality_level.value}</p>
            </div>
            
            <div class="summary">
                <h2>检查摘要</h2>
                <p>总规则数: {report.total_rules}</p>
                <p>通过规则数: {report.passed_rules}</p>
                <p>失败规则数: {report.failed_rules}</p>
                <p>通过率: {report.pass_rate:.2%}</p>
            </div>
            
            <div class="results">
                <h2>详细结果</h2>
        """
        
        for result in report.results:
            status_class = "passed" if result.passed else "failed"
            status_text = "通过" if result.passed else "失败"
            
            html += f"""
                <div class="result {status_class}">
                    <h3>{result.rule_name} - {status_text}</h3>
                    <p><strong>类型:</strong> {result.check_type.value}</p>
                    <p><strong>分数:</strong> {result.score:.2%}</p>
                    <p><strong>阈值:</strong> {result.threshold:.2%}</p>
                    <p><strong>消息:</strong> {result.message}</p>
                    <p><strong>严重程度:</strong> {result.severity}</p>
                    <p><strong>执行时间:</strong> {result.execution_time:.3f}秒</p>
                </div>
            """
        
        if report.recommendations:
            html += """
                </div>
                
                <div class="recommendations">
                    <h2>改进建议</h2>
                    <ul>
            """
            
            for recommendation in report.recommendations:
                html += f"<li>{recommendation}</li>"
            
            html += "</ul></div>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def __str__(self) -> str:
        """
        字符串表示
        
        Returns:
            str: 字符串描述
        """
        return f"DataQualityChecker(name={self.name}, rules={len(self.rules)}, checks={self.stats['total_checks']})"