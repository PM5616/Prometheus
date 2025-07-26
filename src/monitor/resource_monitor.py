"""Resource Monitor Module

资源监控器，负责监控系统资源使用情况。

主要功能：
- CPU使用率监控
- 内存使用监控
- 磁盘使用监控
- 网络流量监控
- 进程资源监控
- 资源告警
"""

import time
import asyncio
import psutil
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timedelta

from src.common.logging.logger import get_logger
from src.common.exceptions.monitor_exceptions import MonitorError


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"
    GPU = "gpu"


from src.common.models import AlertLevel


@dataclass
class ResourceThreshold:
    """资源阈值"""
    resource_type: ResourceType
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    enabled: bool = True
    description: str = ""
    
    def check_threshold(self, value: float) -> Optional[AlertLevel]:
        """检查阈值
        
        Args:
            value: 当前值
            
        Returns:
            告警级别，如果没有超过阈值则返回None
        """
        if not self.enabled:
            return None
        
        if value >= self.critical_threshold:
            return AlertLevel.CRITICAL
        elif value >= self.warning_threshold:
            return AlertLevel.WARNING
        
        return None


@dataclass
class ResourceMetric:
    """资源指标"""
    timestamp: float
    resource_type: ResourceType
    metric_name: str
    value: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'resource_type': self.resource_type.value,
            'metric_name': self.metric_name,
            'value': self.value,
            'unit': self.unit,
            'tags': self.tags
        }


@dataclass
class ResourceAlert:
    """资源告警"""
    id: str
    resource_type: ResourceType
    metric_name: str
    level: AlertLevel
    threshold: float
    current_value: float
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'resource_type': self.resource_type.value,
            'metric_name': self.metric_name,
            'level': self.level.value,
            'threshold': self.threshold,
            'current_value': self.current_value,
            'message': self.message,
            'timestamp': self.timestamp,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at
        }


@dataclass
class ResourceStats:
    """资源统计"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    active_processes: int = 0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'active_processes': self.active_processes,
            'load_average': self.load_average
        }


class ResourceMonitor:
    """资源监控器
    
    负责监控系统资源使用情况。
    """
    
    def __init__(self, 
                 collection_interval: float = 5.0,
                 max_metrics: int = 1000):
        """初始化资源监控器
        
        Args:
            collection_interval: 数据收集间隔（秒）
            max_metrics: 最大指标数量
        """
        self.collection_interval = collection_interval
        self.max_metrics = max_metrics
        self.logger = get_logger("resource_monitor")
        
        # 指标存储
        self.metrics: deque = deque(maxlen=max_metrics)
        self.thresholds: Dict[str, ResourceThreshold] = {}
        self.alerts: Dict[str, ResourceAlert] = {}
        
        # 统计信息
        self.stats = ResourceStats()
        
        # 运行状态
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # 回调函数
        self.metric_callbacks: List[Callable[[ResourceMetric], None]] = []
        self.alert_callbacks: List[Callable[[ResourceAlert], None]] = []
        
        # 网络统计基线
        self._network_baseline: Optional[psutil._common.snetio] = None
        
        # 初始化默认阈值
        self._init_default_thresholds()
        
        self.logger.info(f"资源监控器初始化完成，收集间隔: {collection_interval}秒")
    
    def _init_default_thresholds(self) -> None:
        """初始化默认阈值"""
        default_thresholds = [
            ResourceThreshold(
                resource_type=ResourceType.CPU,
                metric_name="usage_percent",
                warning_threshold=70.0,
                critical_threshold=90.0,
                description="CPU使用率阈值"
            ),
            ResourceThreshold(
                resource_type=ResourceType.MEMORY,
                metric_name="usage_percent",
                warning_threshold=80.0,
                critical_threshold=95.0,
                description="内存使用率阈值"
            ),
            ResourceThreshold(
                resource_type=ResourceType.DISK,
                metric_name="usage_percent",
                warning_threshold=80.0,
                critical_threshold=95.0,
                description="磁盘使用率阈值"
            ),
            ResourceThreshold(
                resource_type=ResourceType.PROCESS,
                metric_name="count",
                warning_threshold=500.0,
                critical_threshold=1000.0,
                description="进程数量阈值"
            )
        ]
        
        for threshold in default_thresholds:
            key = f"{threshold.resource_type.value}_{threshold.metric_name}"
            self.thresholds[key] = threshold
    
    async def start(self) -> None:
        """启动资源监控器"""
        if self._running:
            self.logger.warning("资源监控器已在运行")
            return
        
        self._running = True
        
        # 获取网络基线
        try:
            self._network_baseline = psutil.net_io_counters()
        except Exception as e:
            self.logger.warning(f"获取网络基线失败: {e}")
        
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("资源监控器已启动")
    
    async def stop(self) -> None:
        """停止资源监控器"""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("资源监控器已停止")
    
    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                # 收集指标
                await self._collect_metrics()
                
                # 等待下次收集
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"资源监控出错: {e}")
                await asyncio.sleep(5.0)
    
    async def _collect_metrics(self) -> None:
        """收集指标"""
        timestamp = time.time()
        
        # 收集CPU指标
        await self._collect_cpu_metrics(timestamp)
        
        # 收集内存指标
        await self._collect_memory_metrics(timestamp)
        
        # 收集磁盘指标
        await self._collect_disk_metrics(timestamp)
        
        # 收集网络指标
        await self._collect_network_metrics(timestamp)
        
        # 收集进程指标
        await self._collect_process_metrics(timestamp)
        
        # 更新统计信息
        self._update_stats()
    
    async def _collect_cpu_metrics(self, timestamp: float) -> None:
        """收集CPU指标
        
        Args:
            timestamp: 时间戳
        """
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            metric = ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.CPU,
                metric_name="usage_percent",
                value=cpu_percent,
                unit="%"
            )
            self._add_metric(metric)
            
            # CPU核心数
            cpu_count = psutil.cpu_count(logical=True)
            metric = ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.CPU,
                metric_name="core_count",
                value=cpu_count,
                unit="count"
            )
            self._add_metric(metric)
            
            # 负载平均值（仅Unix系统）
            try:
                load_avg = psutil.getloadavg()
                for i, period in enumerate(['1min', '5min', '15min']):
                    metric = ResourceMetric(
                        timestamp=timestamp,
                        resource_type=ResourceType.CPU,
                        metric_name=f"load_avg_{period}",
                        value=load_avg[i],
                        unit="load"
                    )
                    self._add_metric(metric)
                
                self.stats.load_average = load_avg
            except (AttributeError, OSError):
                # Windows系统不支持getloadavg
                pass
            
            self.stats.cpu_usage = cpu_percent
            
        except Exception as e:
            self.logger.error(f"收集CPU指标失败: {e}")
    
    async def _collect_memory_metrics(self, timestamp: float) -> None:
        """收集内存指标
        
        Args:
            timestamp: 时间戳
        """
        try:
            # 虚拟内存
            vmem = psutil.virtual_memory()
            
            metrics = [
                ("total", vmem.total, "bytes"),
                ("available", vmem.available, "bytes"),
                ("used", vmem.used, "bytes"),
                ("free", vmem.free, "bytes"),
                ("usage_percent", vmem.percent, "%")
            ]
            
            for name, value, unit in metrics:
                metric = ResourceMetric(
                    timestamp=timestamp,
                    resource_type=ResourceType.MEMORY,
                    metric_name=name,
                    value=value,
                    unit=unit
                )
                self._add_metric(metric)
            
            # 交换内存
            swap = psutil.swap_memory()
            
            swap_metrics = [
                ("swap_total", swap.total, "bytes"),
                ("swap_used", swap.used, "bytes"),
                ("swap_free", swap.free, "bytes"),
                ("swap_percent", swap.percent, "%")
            ]
            
            for name, value, unit in swap_metrics:
                metric = ResourceMetric(
                    timestamp=timestamp,
                    resource_type=ResourceType.MEMORY,
                    metric_name=name,
                    value=value,
                    unit=unit
                )
                self._add_metric(metric)
            
            self.stats.memory_usage = vmem.percent
            
        except Exception as e:
            self.logger.error(f"收集内存指标失败: {e}")
    
    async def _collect_disk_metrics(self, timestamp: float) -> None:
        """收集磁盘指标
        
        Args:
            timestamp: 时间戳
        """
        try:
            # 磁盘分区
            partitions = psutil.disk_partitions()
            
            total_usage = 0.0
            partition_count = 0
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    
                    metrics = [
                        ("total", usage.total, "bytes"),
                        ("used", usage.used, "bytes"),
                        ("free", usage.free, "bytes"),
                        ("usage_percent", usage.percent, "%")
                    ]
                    
                    for name, value, unit in metrics:
                        metric = ResourceMetric(
                            timestamp=timestamp,
                            resource_type=ResourceType.DISK,
                            metric_name=name,
                            value=value,
                            unit=unit,
                            tags={"device": partition.device, "mountpoint": partition.mountpoint}
                        )
                        self._add_metric(metric)
                    
                    total_usage += usage.percent
                    partition_count += 1
                    
                except (PermissionError, OSError):
                    # 某些分区可能无法访问
                    continue
            
            # 平均磁盘使用率
            if partition_count > 0:
                avg_usage = total_usage / partition_count
                self.stats.disk_usage = avg_usage
            
            # 磁盘IO
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    io_metrics = [
                        ("read_count", disk_io.read_count, "count"),
                        ("write_count", disk_io.write_count, "count"),
                        ("read_bytes", disk_io.read_bytes, "bytes"),
                        ("write_bytes", disk_io.write_bytes, "bytes"),
                        ("read_time", disk_io.read_time, "ms"),
                        ("write_time", disk_io.write_time, "ms")
                    ]
                    
                    for name, value, unit in io_metrics:
                        metric = ResourceMetric(
                            timestamp=timestamp,
                            resource_type=ResourceType.DISK,
                            metric_name=name,
                            value=value,
                            unit=unit
                        )
                        self._add_metric(metric)
            except Exception:
                pass
            
        except Exception as e:
            self.logger.error(f"收集磁盘指标失败: {e}")
    
    async def _collect_network_metrics(self, timestamp: float) -> None:
        """收集网络指标
        
        Args:
            timestamp: 时间戳
        """
        try:
            # 网络IO
            net_io = psutil.net_io_counters()
            if net_io:
                metrics = [
                    ("bytes_sent", net_io.bytes_sent, "bytes"),
                    ("bytes_recv", net_io.bytes_recv, "bytes"),
                    ("packets_sent", net_io.packets_sent, "count"),
                    ("packets_recv", net_io.packets_recv, "count"),
                    ("errin", net_io.errin, "count"),
                    ("errout", net_io.errout, "count"),
                    ("dropin", net_io.dropin, "count"),
                    ("dropout", net_io.dropout, "count")
                ]
                
                for name, value, unit in metrics:
                    metric = ResourceMetric(
                        timestamp=timestamp,
                        resource_type=ResourceType.NETWORK,
                        metric_name=name,
                        value=value,
                        unit=unit
                    )
                    self._add_metric(metric)
                
                self.stats.network_bytes_sent = net_io.bytes_sent
                self.stats.network_bytes_recv = net_io.bytes_recv
            
            # 网络连接
            try:
                connections = psutil.net_connections()
                connection_count = len(connections)
                
                metric = ResourceMetric(
                    timestamp=timestamp,
                    resource_type=ResourceType.NETWORK,
                    metric_name="connection_count",
                    value=connection_count,
                    unit="count"
                )
                self._add_metric(metric)
            except (PermissionError, psutil.AccessDenied):
                # 某些系统需要管理员权限
                pass
            
        except Exception as e:
            self.logger.error(f"收集网络指标失败: {e}")
    
    async def _collect_process_metrics(self, timestamp: float) -> None:
        """收集进程指标
        
        Args:
            timestamp: 时间戳
        """
        try:
            # 进程数量
            pids = psutil.pids()
            process_count = len(pids)
            
            metric = ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.PROCESS,
                metric_name="count",
                value=process_count,
                unit="count"
            )
            self._add_metric(metric)
            
            self.stats.active_processes = process_count
            
            # 进程状态统计
            status_counts = defaultdict(int)
            
            for pid in pids[:100]:  # 限制检查的进程数量以提高性能
                try:
                    proc = psutil.Process(pid)
                    status = proc.status()
                    status_counts[status] += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            for status, count in status_counts.items():
                metric = ResourceMetric(
                    timestamp=timestamp,
                    resource_type=ResourceType.PROCESS,
                    metric_name="status_count",
                    value=count,
                    unit="count",
                    tags={"status": status}
                )
                self._add_metric(metric)
            
        except Exception as e:
            self.logger.error(f"收集进程指标失败: {e}")
    
    def _add_metric(self, metric: ResourceMetric) -> None:
        """添加指标
        
        Args:
            metric: 资源指标
        """
        # 存储指标
        self.metrics.append(metric)
        
        # 检查阈值
        self._check_thresholds(metric)
        
        # 调用回调
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                self.logger.error(f"指标回调执行失败: {e}")
    
    def _check_thresholds(self, metric: ResourceMetric) -> None:
        """检查阈值
        
        Args:
            metric: 资源指标
        """
        threshold_key = f"{metric.resource_type.value}_{metric.metric_name}"
        
        if threshold_key not in self.thresholds:
            return
        
        threshold = self.thresholds[threshold_key]
        alert_level = threshold.check_threshold(metric.value)
        
        if alert_level:
            self._create_alert(metric, threshold, alert_level)
        else:
            # 检查是否需要解决现有告警
            self._resolve_alert(metric, threshold)
    
    def _create_alert(self, metric: ResourceMetric, threshold: ResourceThreshold, level: AlertLevel) -> None:
        """创建告警
        
        Args:
            metric: 资源指标
            threshold: 资源阈值
            level: 告警级别
        """
        alert_id = f"{metric.resource_type.value}_{metric.metric_name}_{level.value}"
        
        # 如果告警已存在且未解决，更新告警
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            alert = self.alerts[alert_id]
            alert.current_value = metric.value
            alert.timestamp = metric.timestamp
            return
        
        # 创建新告警
        threshold_value = (threshold.critical_threshold if level == AlertLevel.CRITICAL 
                          else threshold.warning_threshold)
        
        message = (f"{metric.resource_type.value.upper()} {metric.metric_name} "
                  f"超过{level.value}阈值: {metric.value:.2f}{metric.unit} > {threshold_value:.2f}{metric.unit}")
        
        alert = ResourceAlert(
            id=alert_id,
            resource_type=metric.resource_type,
            metric_name=metric.metric_name,
            level=level,
            threshold=threshold_value,
            current_value=metric.value,
            message=message,
            timestamp=metric.timestamp
        )
        
        self.alerts[alert_id] = alert
        
        # 调用告警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"告警回调执行失败: {e}")
        
        self.logger.warning(f"资源告警: {message}")
    
    def _resolve_alert(self, metric: ResourceMetric, threshold: ResourceThreshold) -> None:
        """解决告警
        
        Args:
            metric: 资源指标
            threshold: 资源阈值
        """
        # 检查是否有需要解决的告警
        for level in [AlertLevel.CRITICAL, AlertLevel.WARNING]:
            alert_id = f"{metric.resource_type.value}_{metric.metric_name}_{level.value}"
            
            if alert_id in self.alerts and not self.alerts[alert_id].resolved:
                alert = self.alerts[alert_id]
                
                # 如果当前值低于警告阈值，解决告警
                if metric.value < threshold.warning_threshold:
                    alert.resolved = True
                    alert.resolved_at = metric.timestamp
                    
                    self.logger.info(f"资源告警已解决: {alert.message}")
    
    def _update_stats(self) -> None:
        """更新统计信息"""
        # 统计信息在收集指标时已更新
        pass
    
    def add_threshold(self, threshold: ResourceThreshold) -> None:
        """添加阈值
        
        Args:
            threshold: 资源阈值
        """
        key = f"{threshold.resource_type.value}_{threshold.metric_name}"
        self.thresholds[key] = threshold
        self.logger.info(f"已添加资源阈值: {key}")
    
    def remove_threshold(self, resource_type: ResourceType, metric_name: str) -> bool:
        """移除阈值
        
        Args:
            resource_type: 资源类型
            metric_name: 指标名称
            
        Returns:
            是否成功移除
        """
        key = f"{resource_type.value}_{metric_name}"
        
        if key in self.thresholds:
            del self.thresholds[key]
            self.logger.info(f"已移除资源阈值: {key}")
            return True
        
        return False
    
    def get_metrics(self, 
                   resource_type: Optional[ResourceType] = None,
                   metric_name: Optional[str] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   limit: int = 100) -> List[ResourceMetric]:
        """获取指标
        
        Args:
            resource_type: 资源类型过滤
            metric_name: 指标名称过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制数量
            
        Returns:
            指标列表
        """
        metrics = list(self.metrics)
        
        # 过滤
        if resource_type:
            metrics = [m for m in metrics if m.resource_type == resource_type]
        
        if metric_name:
            metrics = [m for m in metrics if m.metric_name == metric_name]
        
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        
        # 按时间倒序排列
        metrics.sort(key=lambda x: x.timestamp, reverse=True)
        
        return metrics[:limit] if limit > 0 else metrics
    
    def get_alerts(self, 
                  resolved: Optional[bool] = None,
                  level: Optional[AlertLevel] = None) -> List[ResourceAlert]:
        """获取告警
        
        Args:
            resolved: 是否已解决过滤
            level: 告警级别过滤
            
        Returns:
            告警列表
        """
        alerts = list(self.alerts.values())
        
        # 过滤
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        # 按时间倒序排列
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts
    
    def get_stats(self) -> ResourceStats:
        """获取统计信息
        
        Returns:
            资源统计信息
        """
        return self.stats
    
    def add_metric_callback(self, callback: Callable[[ResourceMetric], None]) -> None:
        """添加指标回调
        
        Args:
            callback: 回调函数
        """
        self.metric_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]) -> None:
        """添加告警回调
        
        Args:
            callback: 回调函数
        """
        self.alert_callbacks.append(callback)
    
    def clear_metrics(self) -> None:
        """清空指标"""
        self.metrics.clear()
        self.logger.info("指标数据已清空")
    
    def clear_alerts(self) -> None:
        """清空告警"""
        self.alerts.clear()
        self.logger.info("告警数据已清空")