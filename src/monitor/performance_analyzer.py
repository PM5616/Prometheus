"""Performance Analyzer Module

性能分析器，负责分析系统和应用的性能指标。

主要功能：
- 性能指标分析
- 趋势分析
- 异常检测
- 性能报告生成
- 基准测试
"""

import time
import asyncio
import statistics
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque, defaultdict
from datetime import datetime, timedelta

from ..common.logging.logger import get_logger
from ..common.exceptions.monitor_exceptions import MonitorError
from .system_monitor import SystemMetric, MetricType


class AnalysisType(Enum):
    """分析类型"""
    TREND = "trend"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    FORECAST = "forecast"
    BENCHMARK = "benchmark"
    THRESHOLD = "threshold"


class TrendDirection(Enum):
    """趋势方向"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class AnomalyType(Enum):
    """异常类型"""
    SPIKE = "spike"  # 尖峰
    DIP = "dip"  # 低谷
    DRIFT = "drift"  # 漂移
    OUTLIER = "outlier"  # 离群值
    PATTERN_BREAK = "pattern_break"  # 模式中断


@dataclass
class PerformanceThreshold:
    """性能阈值"""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str = ">"
    enabled: bool = True
    
    def check_threshold(self, value: float) -> Tuple[bool, str]:
        """检查阈值
        
        Args:
            value: 指标值
            
        Returns:
            (是否超过阈值, 阈值级别)
        """
        if not self.enabled:
            return False, "normal"
        
        if self.comparison_operator == ">":
            if value > self.critical_threshold:
                return True, "critical"
            elif value > self.warning_threshold:
                return True, "warning"
        elif self.comparison_operator == "<":
            if value < self.critical_threshold:
                return True, "critical"
            elif value < self.warning_threshold:
                return True, "warning"
        
        return False, "normal"


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    metric_type: MetricType
    direction: TrendDirection
    slope: float
    confidence: float
    duration: float
    start_time: float
    end_time: float
    r_squared: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'metric_type': self.metric_type.value,
            'direction': self.direction.value,
            'slope': self.slope,
            'confidence': self.confidence,
            'duration': self.duration,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'r_squared': self.r_squared
        }


@dataclass
class AnomalyDetection:
    """异常检测结果"""
    metric_type: MetricType
    anomaly_type: AnomalyType
    timestamp: float
    value: float
    expected_value: float
    deviation: float
    severity: float
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'metric_type': self.metric_type.value,
            'anomaly_type': self.anomaly_type.value,
            'timestamp': self.timestamp,
            'value': self.value,
            'expected_value': self.expected_value,
            'deviation': self.deviation,
            'severity': self.severity,
            'confidence': self.confidence,
            'context': self.context
        }


@dataclass
class PerformanceReport:
    """性能报告"""
    start_time: float
    end_time: float
    metrics_summary: Dict[MetricType, Dict[str, float]]
    trend_analysis: List[TrendAnalysis]
    anomaly_detections: List[AnomalyDetection]
    threshold_violations: List[Dict[str, Any]]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.end_time - self.start_time,
            'metrics_summary': {
                metric_type.value: summary 
                for metric_type, summary in self.metrics_summary.items()
            },
            'trend_analysis': [trend.to_dict() for trend in self.trend_analysis],
            'anomaly_detections': [anomaly.to_dict() for anomaly in self.anomaly_detections],
            'threshold_violations': self.threshold_violations,
            'recommendations': self.recommendations
        }


class PerformanceAnalyzer:
    """性能分析器
    
    分析系统和应用的性能指标。
    """
    
    def __init__(self, analysis_window: int = 100):
        """初始化性能分析器
        
        Args:
            analysis_window: 分析窗口大小
        """
        self.analysis_window = analysis_window
        self.logger = get_logger("performance_analyzer")
        
        # 指标数据存储
        self.metrics_data: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=analysis_window * 2) 
            for metric_type in MetricType
        }
        
        # 分析结果存储
        self.trend_results: Dict[MetricType, TrendAnalysis] = {}
        self.anomaly_history: deque = deque(maxlen=1000)
        self.threshold_violations: deque = deque(maxlen=1000)
        
        # 性能阈值
        self.thresholds: Dict[MetricType, PerformanceThreshold] = self._init_default_thresholds()
        
        # 基准数据
        self.baselines: Dict[MetricType, Dict[str, float]] = {}
        
        # 分析配置
        self.anomaly_sensitivity = 2.0  # 异常检测敏感度（标准差倍数）
        self.trend_min_points = 10  # 趋势分析最少数据点
        
        # 回调函数
        self.anomaly_callbacks: List[Callable[[AnomalyDetection], None]] = []
        self.threshold_callbacks: List[Callable[[MetricType, float, str], None]] = []
        
        self.logger.info(f"性能分析器初始化完成，分析窗口: {analysis_window}")
    
    def _init_default_thresholds(self) -> Dict[MetricType, PerformanceThreshold]:
        """初始化默认阈值
        
        Returns:
            默认阈值配置
        """
        return {
            MetricType.CPU_USAGE: PerformanceThreshold(
                metric_type=MetricType.CPU_USAGE,
                warning_threshold=70.0,
                critical_threshold=90.0
            ),
            MetricType.MEMORY_USAGE: PerformanceThreshold(
                metric_type=MetricType.MEMORY_USAGE,
                warning_threshold=80.0,
                critical_threshold=95.0
            ),
            MetricType.DISK_USAGE: PerformanceThreshold(
                metric_type=MetricType.DISK_USAGE,
                warning_threshold=85.0,
                critical_threshold=95.0
            ),
            MetricType.LOAD_AVERAGE: PerformanceThreshold(
                metric_type=MetricType.LOAD_AVERAGE,
                warning_threshold=2.0,
                critical_threshold=5.0
            )
        }
    
    def add_metric(self, metric: SystemMetric) -> None:
        """添加指标数据
        
        Args:
            metric: 系统指标
        """
        if metric.type in self.metrics_data:
            self.metrics_data[metric.type].append(metric)
            
            # 检查阈值
            self._check_threshold(metric)
            
            # 异常检测
            anomaly = self._detect_anomaly(metric)
            if anomaly:
                self.anomaly_history.append(anomaly)
                
                # 调用异常回调
                for callback in self.anomaly_callbacks:
                    try:
                        callback(anomaly)
                    except Exception as e:
                        self.logger.error(f"异常回调执行失败: {e}")
    
    def _check_threshold(self, metric: SystemMetric) -> None:
        """检查阈值
        
        Args:
            metric: 系统指标
        """
        if metric.type in self.thresholds:
            threshold = self.thresholds[metric.type]
            exceeded, level = threshold.check_threshold(metric.value)
            
            if exceeded:
                violation = {
                    'metric_type': metric.type.value,
                    'timestamp': metric.timestamp,
                    'value': metric.value,
                    'threshold_level': level,
                    'threshold_value': (
                        threshold.critical_threshold if level == 'critical' 
                        else threshold.warning_threshold
                    ),
                    'labels': metric.labels
                }
                
                self.threshold_violations.append(violation)
                
                # 调用阈值回调
                for callback in self.threshold_callbacks:
                    try:
                        callback(metric.type, metric.value, level)
                    except Exception as e:
                        self.logger.error(f"阈值回调执行失败: {e}")
    
    def _detect_anomaly(self, metric: SystemMetric) -> Optional[AnomalyDetection]:
        """检测异常
        
        Args:
            metric: 系统指标
            
        Returns:
            异常检测结果
        """
        data = list(self.metrics_data[metric.type])
        
        if len(data) < 10:  # 数据点太少，无法检测异常
            return None
        
        # 获取历史数据值
        values = [m.value for m in data[:-1]]  # 排除当前值
        current_value = metric.value
        
        if not values:
            return None
        
        # 计算统计指标
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0
        
        if std_value == 0:
            return None
        
        # 计算Z分数
        z_score = abs(current_value - mean_value) / std_value
        
        # 异常检测
        if z_score > self.anomaly_sensitivity:
            # 确定异常类型
            anomaly_type = AnomalyType.OUTLIER
            
            if current_value > mean_value + self.anomaly_sensitivity * std_value:
                anomaly_type = AnomalyType.SPIKE
            elif current_value < mean_value - self.anomaly_sensitivity * std_value:
                anomaly_type = AnomalyType.DIP
            
            # 计算严重程度和置信度
            severity = min(z_score / self.anomaly_sensitivity, 1.0)
            confidence = min(z_score / 3.0, 1.0)  # 3-sigma规则
            
            return AnomalyDetection(
                metric_type=metric.type,
                anomaly_type=anomaly_type,
                timestamp=metric.timestamp,
                value=current_value,
                expected_value=mean_value,
                deviation=current_value - mean_value,
                severity=severity,
                confidence=confidence,
                context={
                    'z_score': z_score,
                    'mean': mean_value,
                    'std': std_value,
                    'sample_size': len(values)
                }
            )
        
        return None
    
    def analyze_trend(self, metric_type: MetricType, window_size: Optional[int] = None) -> Optional[TrendAnalysis]:
        """分析趋势
        
        Args:
            metric_type: 指标类型
            window_size: 分析窗口大小
            
        Returns:
            趋势分析结果
        """
        if metric_type not in self.metrics_data:
            return None
        
        data = list(self.metrics_data[metric_type])
        
        if window_size:
            data = data[-window_size:]
        
        if len(data) < self.trend_min_points:
            return None
        
        # 提取时间和值
        timestamps = [m.timestamp for m in data]
        values = [m.value for m in data]
        
        # 线性回归分析
        try:
            # 使用numpy进行线性回归
            x = np.array(timestamps)
            y = np.array(values)
            
            # 标准化时间戳
            x_norm = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.zeros_like(x)
            
            # 计算线性回归
            coeffs = np.polyfit(x_norm, y, 1)
            slope = coeffs[0]
            
            # 计算R²
            y_pred = np.polyval(coeffs, x_norm)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # 确定趋势方向
            direction = TrendDirection.STABLE
            if abs(slope) > 0.1:  # 阈值可配置
                if slope > 0:
                    direction = TrendDirection.INCREASING
                else:
                    direction = TrendDirection.DECREASING
            
            # 检查波动性
            if np.std(y) / np.mean(y) > 0.2:  # 变异系数 > 20%
                direction = TrendDirection.VOLATILE
            
            # 计算置信度
            confidence = min(r_squared, 1.0)
            
            trend_analysis = TrendAnalysis(
                metric_type=metric_type,
                direction=direction,
                slope=slope,
                confidence=confidence,
                duration=timestamps[-1] - timestamps[0],
                start_time=timestamps[0],
                end_time=timestamps[-1],
                r_squared=r_squared
            )
            
            # 缓存结果
            self.trend_results[metric_type] = trend_analysis
            
            return trend_analysis
            
        except Exception as e:
            self.logger.error(f"趋势分析失败: {e}")
            return None
    
    def set_baseline(self, metric_type: MetricType, window_size: int = 50) -> None:
        """设置基准
        
        Args:
            metric_type: 指标类型
            window_size: 基准窗口大小
        """
        if metric_type not in self.metrics_data:
            return
        
        data = list(self.metrics_data[metric_type])[-window_size:]
        
        if len(data) < 10:
            self.logger.warning(f"数据点不足，无法设置基准: {metric_type.value}")
            return
        
        values = [m.value for m in data]
        
        baseline = {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'sample_size': len(values),
            'timestamp': time.time()
        }
        
        self.baselines[metric_type] = baseline
        self.logger.info(f"已设置基准: {metric_type.value}")
    
    def compare_with_baseline(self, metric_type: MetricType) -> Optional[Dict[str, Any]]:
        """与基准比较
        
        Args:
            metric_type: 指标类型
            
        Returns:
            比较结果
        """
        if metric_type not in self.baselines or metric_type not in self.metrics_data:
            return None
        
        baseline = self.baselines[metric_type]
        recent_data = list(self.metrics_data[metric_type])[-50:]  # 最近50个数据点
        
        if not recent_data:
            return None
        
        recent_values = [m.value for m in recent_data]
        current_stats = {
            'mean': statistics.mean(recent_values),
            'median': statistics.median(recent_values),
            'std': statistics.stdev(recent_values) if len(recent_values) > 1 else 0,
            'min': min(recent_values),
            'max': max(recent_values),
            'p95': np.percentile(recent_values, 95),
            'p99': np.percentile(recent_values, 99)
        }
        
        comparison = {
            'baseline': baseline,
            'current': current_stats,
            'changes': {
                'mean_change': ((current_stats['mean'] - baseline['mean']) / baseline['mean'] * 100) if baseline['mean'] != 0 else 0,
                'median_change': ((current_stats['median'] - baseline['median']) / baseline['median'] * 100) if baseline['median'] != 0 else 0,
                'std_change': ((current_stats['std'] - baseline['std']) / baseline['std'] * 100) if baseline['std'] != 0 else 0,
                'p95_change': ((current_stats['p95'] - baseline['p95']) / baseline['p95'] * 100) if baseline['p95'] != 0 else 0
            },
            'analysis_time': time.time()
        }
        
        return comparison
    
    def generate_report(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> PerformanceReport:
        """生成性能报告
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            性能报告
        """
        if end_time is None:
            end_time = time.time()
        if start_time is None:
            start_time = end_time - 3600  # 默认1小时
        
        # 指标摘要
        metrics_summary = {}
        for metric_type, data in self.metrics_data.items():
            filtered_data = [
                m for m in data 
                if start_time <= m.timestamp <= end_time
            ]
            
            if filtered_data:
                values = [m.value for m in filtered_data]
                metrics_summary[metric_type] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'p95': float(np.percentile(values, 95)),
                    'p99': float(np.percentile(values, 99))
                }
        
        # 趋势分析
        trend_analysis = []
        for metric_type in MetricType:
            trend = self.analyze_trend(metric_type)
            if trend and start_time <= trend.start_time <= end_time:
                trend_analysis.append(trend)
        
        # 异常检测
        anomaly_detections = [
            anomaly for anomaly in self.anomaly_history
            if start_time <= anomaly.timestamp <= end_time
        ]
        
        # 阈值违规
        threshold_violations = [
            violation for violation in self.threshold_violations
            if start_time <= violation['timestamp'] <= end_time
        ]
        
        # 生成建议
        recommendations = self._generate_recommendations(
            metrics_summary, trend_analysis, anomaly_detections, threshold_violations
        )
        
        return PerformanceReport(
            start_time=start_time,
            end_time=end_time,
            metrics_summary=metrics_summary,
            trend_analysis=trend_analysis,
            anomaly_detections=anomaly_detections,
            threshold_violations=threshold_violations,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, 
                                metrics_summary: Dict[MetricType, Dict[str, float]],
                                trend_analysis: List[TrendAnalysis],
                                anomaly_detections: List[AnomalyDetection],
                                threshold_violations: List[Dict[str, Any]]) -> List[str]:
        """生成建议
        
        Args:
            metrics_summary: 指标摘要
            trend_analysis: 趋势分析
            anomaly_detections: 异常检测
            threshold_violations: 阈值违规
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 基于阈值违规的建议
        cpu_violations = [v for v in threshold_violations if v['metric_type'] == 'cpu_usage']
        if cpu_violations:
            recommendations.append("检测到CPU使用率过高，建议检查高CPU消耗的进程并优化")
        
        memory_violations = [v for v in threshold_violations if v['metric_type'] == 'memory_usage']
        if memory_violations:
            recommendations.append("检测到内存使用率过高，建议检查内存泄漏或增加内存容量")
        
        disk_violations = [v for v in threshold_violations if v['metric_type'] == 'disk_usage']
        if disk_violations:
            recommendations.append("检测到磁盘使用率过高，建议清理不必要的文件或扩展存储空间")
        
        # 基于趋势分析的建议
        for trend in trend_analysis:
            if trend.direction == TrendDirection.INCREASING and trend.confidence > 0.7:
                if trend.metric_type == MetricType.CPU_USAGE:
                    recommendations.append(f"CPU使用率呈上升趋势，建议监控并优化性能")
                elif trend.metric_type == MetricType.MEMORY_USAGE:
                    recommendations.append(f"内存使用率呈上升趋势，可能存在内存泄漏")
        
        # 基于异常检测的建议
        spike_anomalies = [a for a in anomaly_detections if a.anomaly_type == AnomalyType.SPIKE]
        if len(spike_anomalies) > 5:  # 频繁出现尖峰
            recommendations.append("检测到频繁的性能尖峰，建议检查系统负载和资源分配")
        
        # 基于指标摘要的建议
        for metric_type, summary in metrics_summary.items():
            if summary['std'] / summary['mean'] > 0.5:  # 高变异系数
                recommendations.append(f"{metric_type.value}指标波动较大，建议检查系统稳定性")
        
        return recommendations
    
    def set_threshold(self, metric_type: MetricType, warning: float, critical: float, 
                     operator: str = ">") -> None:
        """设置阈值
        
        Args:
            metric_type: 指标类型
            warning: 警告阈值
            critical: 严重阈值
            operator: 比较操作符
        """
        self.thresholds[metric_type] = PerformanceThreshold(
            metric_type=metric_type,
            warning_threshold=warning,
            critical_threshold=critical,
            comparison_operator=operator
        )
        
        self.logger.info(f"已设置阈值: {metric_type.value} 警告={warning} 严重={critical}")
    
    def add_anomaly_callback(self, callback: Callable[[AnomalyDetection], None]) -> None:
        """添加异常回调
        
        Args:
            callback: 回调函数
        """
        self.anomaly_callbacks.append(callback)
    
    def add_threshold_callback(self, callback: Callable[[MetricType, float, str], None]) -> None:
        """添加阈值回调
        
        Args:
            callback: 回调函数
        """
        self.threshold_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息
        """
        return {
            'metrics_count': {metric_type.value: len(data) for metric_type, data in self.metrics_data.items()},
            'anomalies_detected': len(self.anomaly_history),
            'threshold_violations': len(self.threshold_violations),
            'baselines_set': list(self.baselines.keys()),
            'analysis_config': {
                'analysis_window': self.analysis_window,
                'anomaly_sensitivity': self.anomaly_sensitivity,
                'trend_min_points': self.trend_min_points
            }
        }
    
    def clear_data(self) -> None:
        """清空数据"""
        for data in self.metrics_data.values():
            data.clear()
        self.anomaly_history.clear()
        self.threshold_violations.clear()
        self.trend_results.clear()
        self.logger.info("分析数据已清空")