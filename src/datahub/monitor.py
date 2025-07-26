"""Data Monitor Module

数据监控器，用于实时监控数据流和系统状态。

功能特性：
- 实时数据流监控
- 系统性能监控
- 异常检测和告警
- 监控指标收集
- 健康检查
"""

import time
import logging
import threading
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import psutil
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from src.common.models.market import Ticker, Kline, OrderBook
from src.common.exceptions.data import DataMonitorError


class MonitorType(Enum):
    """监控类型枚举"""
    DATA_FLOW = "data_flow"  # 数据流监控
    SYSTEM_PERFORMANCE = "system_performance"  # 系统性能监控
    ERROR_RATE = "error_rate"  # 错误率监控
    LATENCY = "latency"  # 延迟监控
    THROUGHPUT = "throughput"  # 吞吐量监控
    HEALTH_CHECK = "health_check"  # 健康检查
    CUSTOM = "custom"  # 自定义监控


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"  # 计数器
    GAUGE = "gauge"  # 仪表盘
    HISTOGRAM = "histogram"  # 直方图
    SUMMARY = "summary"  # 摘要


@dataclass
class MonitorConfig:
    """监控配置"""
    name: str
    monitor_type: MonitorType
    enabled: bool = True
    interval: float = 60.0  # 监控间隔（秒）
    threshold: Optional[float] = None
    alert_level: AlertLevel = AlertLevel.WARNING
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.monitor_type, str):
            self.monitor_type = MonitorType(self.monitor_type)
        if isinstance(self.alert_level, str):
            self.alert_level = AlertLevel(self.alert_level)


@dataclass
class Metric:
    """监控指标"""
    name: str
    metric_type: MetricType
    value: Union[float, int]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.metric_type, str):
            self.metric_type = MetricType(self.metric_type)


@dataclass
class Alert:
    """告警信息"""
    id: str
    monitor_name: str
    level: AlertLevel
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.level, str):
            self.level = AlertLevel(self.level)


@dataclass
class HealthStatus:
    """健康状态"""
    component: str
    status: str  # healthy, degraded, unhealthy
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class DataMonitor:
    """数据监控器
    
    用于实时监控数据流和系统状态。
    """
    
    def __init__(self, name: str = "DataMonitor", config: Optional[Dict] = None):
        """
        初始化数据监控器
        
        Args:
            name: 监控器名称
            config: 配置参数
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # 监控配置
        self.monitors: Dict[str, MonitorConfig] = {}
        
        # 指标存储
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_history_size = self.config.get('metric_history_size', 1000)
        
        # 告警管理
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.max_alerts = self.config.get('max_alerts', 1000)
        
        # 健康检查
        self.health_checks: Dict[str, Callable[[], HealthStatus]] = {}
        self.health_status: Dict[str, HealthStatus] = {}
        
        # 监控状态
        self.running = False
        self.monitor_threads: Dict[str, threading.Thread] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        
        # 统计信息
        self.stats = {
            'start_time': None,
            'total_metrics_collected': 0,
            'total_alerts_generated': 0,
            'active_monitors': 0,
            'last_health_check': None
        }
        
        # 数据流监控
        self.data_flow_stats = {
            'total_records_processed': 0,
            'records_per_second': 0.0,
            'last_record_time': None,
            'error_count': 0,
            'error_rate': 0.0
        }
        
        # 系统性能监控
        self.system_stats = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_io': {'bytes_sent': 0, 'bytes_recv': 0}
        }
        
        # 初始化默认监控
        self._init_default_monitors()
        
        self.logger.info(f"数据监控器 {name} 初始化完成")
    
    def _init_default_monitors(self):
        """初始化默认监控"""
        
        # 数据流监控
        self.add_monitor(MonitorConfig(
            name="data_flow_rate",
            monitor_type=MonitorType.DATA_FLOW,
            interval=30.0,
            threshold=100.0,  # 每秒处理记录数阈值
            alert_level=AlertLevel.WARNING
        ))
        
        # 错误率监控
        self.add_monitor(MonitorConfig(
            name="error_rate",
            monitor_type=MonitorType.ERROR_RATE,
            interval=60.0,
            threshold=0.05,  # 5%错误率阈值
            alert_level=AlertLevel.ERROR
        ))
        
        # 系统性能监控
        self.add_monitor(MonitorConfig(
            name="cpu_usage",
            monitor_type=MonitorType.SYSTEM_PERFORMANCE,
            interval=30.0,
            threshold=80.0,  # 80% CPU使用率阈值
            alert_level=AlertLevel.WARNING
        ))
        
        self.add_monitor(MonitorConfig(
            name="memory_usage",
            monitor_type=MonitorType.SYSTEM_PERFORMANCE,
            interval=30.0,
            threshold=85.0,  # 85%内存使用率阈值
            alert_level=AlertLevel.WARNING
        ))
        
        # 健康检查监控
        self.add_monitor(MonitorConfig(
            name="health_check",
            monitor_type=MonitorType.HEALTH_CHECK,
            interval=120.0,  # 2分钟检查一次
            alert_level=AlertLevel.ERROR
        ))
    
    def add_monitor(self, monitor_config: MonitorConfig):
        """
        添加监控配置
        
        Args:
            monitor_config: 监控配置
        """
        self.monitors[monitor_config.name] = monitor_config
        self.logger.info(f"添加监控: {monitor_config.name}")
    
    def remove_monitor(self, monitor_name: str) -> bool:
        """
        移除监控配置
        
        Args:
            monitor_name: 监控名称
            
        Returns:
            bool: 是否成功移除
        """
        if monitor_name in self.monitors:
            del self.monitors[monitor_name]
            
            # 停止监控线程
            if monitor_name in self.monitor_threads:
                thread = self.monitor_threads[monitor_name]
                if thread.is_alive():
                    # 这里需要实现线程停止机制
                    pass
                del self.monitor_threads[monitor_name]
            
            self.logger.info(f"移除监控: {monitor_name}")
            return True
        return False
    
    def add_health_check(self, component: str, check_function: Callable[[], HealthStatus]):
        """
        添加健康检查
        
        Args:
            component: 组件名称
            check_function: 检查函数
        """
        self.health_checks[component] = check_function
        self.logger.info(f"添加健康检查: {component}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """
        添加告警处理器
        
        Args:
            handler: 告警处理函数
        """
        self.alert_handlers.append(handler)
        self.logger.info("添加告警处理器")
    
    def start(self):
        """
        启动监控
        """
        if self.running:
            self.logger.warning("监控器已在运行")
            return
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        # 启动各个监控线程
        for monitor_name, monitor_config in self.monitors.items():
            if monitor_config.enabled:
                thread = threading.Thread(
                    target=self._monitor_loop,
                    args=(monitor_name, monitor_config),
                    daemon=True
                )
                thread.start()
                self.monitor_threads[monitor_name] = thread
                self.stats['active_monitors'] += 1
        
        self.logger.info(f"监控器启动，活跃监控数: {self.stats['active_monitors']}")
    
    def stop(self):
        """
        停止监控
        """
        if not self.running:
            self.logger.warning("监控器未在运行")
            return
        
        self.running = False
        
        # 等待所有监控线程结束
        for thread in self.monitor_threads.values():
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        self.monitor_threads.clear()
        self.stats['active_monitors'] = 0
        
        self.logger.info("监控器已停止")
    
    def _monitor_loop(self, monitor_name: str, monitor_config: MonitorConfig):
        """
        监控循环
        
        Args:
            monitor_name: 监控名称
            monitor_config: 监控配置
        """
        self.logger.info(f"启动监控线程: {monitor_name}")
        
        while self.running:
            try:
                start_time = time.time()
                
                # 执行监控检查
                if monitor_config.monitor_type == MonitorType.DATA_FLOW:
                    self._check_data_flow(monitor_name, monitor_config)
                elif monitor_config.monitor_type == MonitorType.ERROR_RATE:
                    self._check_error_rate(monitor_name, monitor_config)
                elif monitor_config.monitor_type == MonitorType.SYSTEM_PERFORMANCE:
                    self._check_system_performance(monitor_name, monitor_config)
                elif monitor_config.monitor_type == MonitorType.HEALTH_CHECK:
                    self._check_health(monitor_name, monitor_config)
                elif monitor_config.monitor_type == MonitorType.CUSTOM:
                    self._check_custom(monitor_name, monitor_config)
                
                # 计算执行时间
                execution_time = time.time() - start_time
                
                # 记录监控执行时间指标
                self.record_metric(Metric(
                    name=f"{monitor_name}_execution_time",
                    metric_type=MetricType.GAUGE,
                    value=execution_time,
                    timestamp=datetime.now(),
                    unit="seconds"
                ))
                
                # 等待下次检查
                time.sleep(monitor_config.interval)
                
            except Exception as e:
                self.logger.error(f"监控 {monitor_name} 执行失败: {e}")
                time.sleep(monitor_config.interval)
        
        self.logger.info(f"监控线程结束: {monitor_name}")
    
    def _check_data_flow(self, monitor_name: str, monitor_config: MonitorConfig):
        """
        检查数据流
        
        Args:
            monitor_name: 监控名称
            monitor_config: 监控配置
        """
        current_time = datetime.now()
        
        # 计算数据处理速率
        if self.data_flow_stats['last_record_time']:
            time_diff = (current_time - self.data_flow_stats['last_record_time']).total_seconds()
            if time_diff > 0:
                self.data_flow_stats['records_per_second'] = (
                    self.data_flow_stats['total_records_processed'] / time_diff
                )
        
        # 记录指标
        self.record_metric(Metric(
            name="data_flow_rate",
            metric_type=MetricType.GAUGE,
            value=self.data_flow_stats['records_per_second'],
            timestamp=current_time,
            unit="records/second"
        ))
        
        self.record_metric(Metric(
            name="total_records_processed",
            metric_type=MetricType.COUNTER,
            value=self.data_flow_stats['total_records_processed'],
            timestamp=current_time,
            unit="records"
        ))
        
        # 检查阈值
        if (monitor_config.threshold and 
            self.data_flow_stats['records_per_second'] < monitor_config.threshold):
            self._generate_alert(
                monitor_name=monitor_name,
                level=monitor_config.alert_level,
                message=f"数据流速率过低: {self.data_flow_stats['records_per_second']:.2f} < {monitor_config.threshold}"
            )
    
    def _check_error_rate(self, monitor_name: str, monitor_config: MonitorConfig):
        """
        检查错误率
        
        Args:
            monitor_name: 监控名称
            monitor_config: 监控配置
        """
        current_time = datetime.now()
        
        # 计算错误率
        total_processed = self.data_flow_stats['total_records_processed']
        error_count = self.data_flow_stats['error_count']
        
        if total_processed > 0:
            error_rate = error_count / total_processed
        else:
            error_rate = 0.0
        
        self.data_flow_stats['error_rate'] = error_rate
        
        # 记录指标
        self.record_metric(Metric(
            name="error_rate",
            metric_type=MetricType.GAUGE,
            value=error_rate,
            timestamp=current_time,
            unit="ratio"
        ))
        
        self.record_metric(Metric(
            name="error_count",
            metric_type=MetricType.COUNTER,
            value=error_count,
            timestamp=current_time,
            unit="errors"
        ))
        
        # 检查阈值
        if monitor_config.threshold and error_rate > monitor_config.threshold:
            self._generate_alert(
                monitor_name=monitor_name,
                level=monitor_config.alert_level,
                message=f"错误率过高: {error_rate:.2%} > {monitor_config.threshold:.2%}"
            )
    
    def _check_system_performance(self, monitor_name: str, monitor_config: MonitorConfig):
        """
        检查系统性能
        
        Args:
            monitor_name: 监控名称
            monitor_config: 监控配置
        """
        current_time = datetime.now()
        
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_stats['cpu_usage'] = cpu_percent
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.system_stats['memory_usage'] = memory_percent
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_stats['disk_usage'] = disk_percent
            
            # 网络IO
            net_io = psutil.net_io_counters()
            self.system_stats['network_io'] = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
            
            # 记录指标
            if monitor_name == "cpu_usage":
                self.record_metric(Metric(
                    name="cpu_usage",
                    metric_type=MetricType.GAUGE,
                    value=cpu_percent,
                    timestamp=current_time,
                    unit="percent"
                ))
                
                # 检查CPU阈值
                if monitor_config.threshold and cpu_percent > monitor_config.threshold:
                    self._generate_alert(
                        monitor_name=monitor_name,
                        level=monitor_config.alert_level,
                        message=f"CPU使用率过高: {cpu_percent:.1f}% > {monitor_config.threshold}%"
                    )
            
            elif monitor_name == "memory_usage":
                self.record_metric(Metric(
                    name="memory_usage",
                    metric_type=MetricType.GAUGE,
                    value=memory_percent,
                    timestamp=current_time,
                    unit="percent"
                ))
                
                # 检查内存阈值
                if monitor_config.threshold and memory_percent > monitor_config.threshold:
                    self._generate_alert(
                        monitor_name=monitor_name,
                        level=monitor_config.alert_level,
                        message=f"内存使用率过高: {memory_percent:.1f}% > {monitor_config.threshold}%"
                    )
            
            # 记录其他系统指标
            self.record_metric(Metric(
                name="disk_usage",
                metric_type=MetricType.GAUGE,
                value=disk_percent,
                timestamp=current_time,
                unit="percent"
            ))
            
            self.record_metric(Metric(
                name="network_bytes_sent",
                metric_type=MetricType.COUNTER,
                value=net_io.bytes_sent,
                timestamp=current_time,
                unit="bytes"
            ))
            
            self.record_metric(Metric(
                name="network_bytes_recv",
                metric_type=MetricType.COUNTER,
                value=net_io.bytes_recv,
                timestamp=current_time,
                unit="bytes"
            ))
            
        except Exception as e:
            self.logger.error(f"系统性能监控失败: {e}")
    
    def _check_health(self, monitor_name: str, monitor_config: MonitorConfig):
        """
        执行健康检查
        
        Args:
            monitor_name: 监控名称
            monitor_config: 监控配置
        """
        current_time = datetime.now()
        self.stats['last_health_check'] = current_time
        
        unhealthy_components = []
        
        for component, check_function in self.health_checks.items():
            try:
                health_status = check_function()
                self.health_status[component] = health_status
                
                # 记录健康状态指标
                status_value = 1.0 if health_status.status == "healthy" else 0.0
                self.record_metric(Metric(
                    name=f"health_{component}",
                    metric_type=MetricType.GAUGE,
                    value=status_value,
                    timestamp=current_time,
                    tags={"component": component, "status": health_status.status}
                ))
                
                if health_status.status != "healthy":
                    unhealthy_components.append(f"{component}: {health_status.message}")
                    
            except Exception as e:
                self.logger.error(f"健康检查失败 {component}: {e}")
                unhealthy_components.append(f"{component}: 检查失败 - {str(e)}")
        
        # 生成告警
        if unhealthy_components:
            self._generate_alert(
                monitor_name=monitor_name,
                level=monitor_config.alert_level,
                message=f"发现不健康组件: {', '.join(unhealthy_components)}"
            )
    
    def _check_custom(self, monitor_name: str, monitor_config: MonitorConfig):
        """
        执行自定义监控
        
        Args:
            monitor_name: 监控名称
            monitor_config: 监控配置
        """
        # 自定义监控逻辑由用户在metadata中定义
        custom_function = monitor_config.metadata.get('check_function')
        if custom_function and callable(custom_function):
            try:
                result = custom_function()
                
                # 记录自定义指标
                if isinstance(result, dict):
                    for metric_name, value in result.items():
                        self.record_metric(Metric(
                            name=f"{monitor_name}_{metric_name}",
                            metric_type=MetricType.GAUGE,
                            value=value,
                            timestamp=datetime.now(),
                            tags=monitor_config.tags
                        ))
                
                # 检查阈值
                if monitor_config.threshold and isinstance(result, (int, float)):
                    if result > monitor_config.threshold:
                        self._generate_alert(
                            monitor_name=monitor_name,
                            level=monitor_config.alert_level,
                            message=f"自定义监控阈值超限: {result} > {monitor_config.threshold}"
                        )
                        
            except Exception as e:
                self.logger.error(f"自定义监控执行失败 {monitor_name}: {e}")
    
    def record_metric(self, metric: Metric):
        """
        记录监控指标
        
        Args:
            metric: 监控指标
        """
        self.metrics[metric.name].append(metric)
        self.stats['total_metrics_collected'] += 1
    
    def record_data_event(self, event_type: str, data: Any = None, error: Optional[Exception] = None):
        """
        记录数据事件
        
        Args:
            event_type: 事件类型 (processed, error, etc.)
            data: 数据
            error: 错误信息
        """
        current_time = datetime.now()
        
        if event_type == "processed":
            self.data_flow_stats['total_records_processed'] += 1
            self.data_flow_stats['last_record_time'] = current_time
            
        elif event_type == "error":
            self.data_flow_stats['error_count'] += 1
            
            # 记录错误指标
            self.record_metric(Metric(
                name="data_processing_error",
                metric_type=MetricType.COUNTER,
                value=1,
                timestamp=current_time,
                tags={"error_type": type(error).__name__ if error else "unknown"}
            ))
    
    def _generate_alert(self, monitor_name: str, level: AlertLevel, message: str, 
                       tags: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        生成告警
        
        Args:
            monitor_name: 监控名称
            level: 告警级别
            message: 告警消息
            tags: 标签
            metadata: 元数据
        """
        alert_id = f"{monitor_name}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            monitor_name=monitor_name,
            level=level,
            message=message,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        self.stats['total_alerts_generated'] += 1
        
        # 限制告警数量
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # 调用告警处理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"告警处理器执行失败: {e}")
        
        self.logger.warning(f"生成告警 [{level.value}] {monitor_name}: {message}")
    
    def get_metrics(self, metric_name: Optional[str] = None, 
                   start_time: Optional[datetime] = None, 
                   end_time: Optional[datetime] = None) -> Dict[str, List[Metric]]:
        """
        获取监控指标
        
        Args:
            metric_name: 指标名称
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            Dict[str, List[Metric]]: 指标数据
        """
        result = {}
        
        metrics_to_query = [metric_name] if metric_name else self.metrics.keys()
        
        for name in metrics_to_query:
            if name in self.metrics:
                metrics = list(self.metrics[name])
                
                # 时间过滤
                if start_time or end_time:
                    filtered_metrics = []
                    for metric in metrics:
                        if start_time and metric.timestamp < start_time:
                            continue
                        if end_time and metric.timestamp > end_time:
                            continue
                        filtered_metrics.append(metric)
                    metrics = filtered_metrics
                
                result[name] = metrics
        
        return result
    
    def get_alerts(self, level: Optional[AlertLevel] = None, 
                  resolved: Optional[bool] = None,
                  limit: Optional[int] = None) -> List[Alert]:
        """
        获取告警信息
        
        Args:
            level: 告警级别过滤
            resolved: 是否已解决过滤
            limit: 限制数量
            
        Returns:
            List[Alert]: 告警列表
        """
        alerts = self.alerts.copy()
        
        # 过滤
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        if resolved is not None:
            alerts = [alert for alert in alerts if alert.resolved == resolved]
        
        # 排序（最新的在前）
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        # 限制数量
        if limit:
            alerts = alerts[:limit]
        
        return alerts
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        解决告警
        
        Args:
            alert_id: 告警ID
            
        Returns:
            bool: 是否成功解决
        """
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self.logger.info(f"告警已解决: {alert_id}")
                return True
        return False
    
    def get_health_status(self, component: Optional[str] = None) -> Union[HealthStatus, Dict[str, HealthStatus]]:
        """
        获取健康状态
        
        Args:
            component: 组件名称
            
        Returns:
            Union[HealthStatus, Dict[str, HealthStatus]]: 健康状态
        """
        if component:
            return self.health_status.get(component)
        return self.health_status.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取监控统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.stats.copy()
        
        # 添加运行时间
        if stats['start_time']:
            stats['uptime_seconds'] = (datetime.now() - stats['start_time']).total_seconds()
        
        # 添加数据流统计
        stats['data_flow'] = self.data_flow_stats.copy()
        
        # 添加系统统计
        stats['system'] = self.system_stats.copy()
        
        # 添加告警统计
        stats['alerts'] = {
            'total': len(self.alerts),
            'unresolved': len([a for a in self.alerts if not a.resolved]),
            'by_level': {}
        }
        
        for level in AlertLevel:
            count = len([a for a in self.alerts if a.level == level])
            stats['alerts']['by_level'][level.value] = count
        
        return stats
    
    def export_metrics(self, format: str = "json") -> str:
        """
        导出监控指标
        
        Args:
            format: 导出格式 (json, csv)
            
        Returns:
            str: 导出数据
        """
        if format == "json":
            data = {}
            for name, metrics in self.metrics.items():
                data[name] = [
                    {
                        'name': m.name,
                        'type': m.metric_type.value,
                        'value': m.value,
                        'timestamp': m.timestamp.isoformat(),
                        'tags': m.tags,
                        'unit': m.unit,
                        'description': m.description
                    }
                    for m in metrics
                ]
            return json.dumps(data, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # 写入标题
            writer.writerow(['metric_name', 'type', 'value', 'timestamp', 'tags', 'unit', 'description'])
            
            # 写入数据
            for name, metrics in self.metrics.items():
                for metric in metrics:
                    writer.writerow([
                        metric.name,
                        metric.metric_type.value,
                        metric.value,
                        metric.timestamp.isoformat(),
                        json.dumps(metric.tags),
                        metric.unit or '',
                        metric.description or ''
                    ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def __str__(self) -> str:
        """
        字符串表示
        
        Returns:
            str: 字符串描述
        """
        return f"DataMonitor(name={self.name}, monitors={len(self.monitors)}, running={self.running})"