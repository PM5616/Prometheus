"""System Monitor Module

系统监控器，负责监控系统级别的性能指标。

主要功能：
- CPU监控
- 内存监控
- 磁盘监控
- 网络监控
- 进程监控
"""

import time
import asyncio
import psutil
import platform
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque, defaultdict

from src.common.logging.logger import get_logger
from src.common.exceptions.monitor_exceptions import MonitorError


class MetricType(Enum):
    """指标类型"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    PROCESS_COUNT = "process_count"
    LOAD_AVERAGE = "load_average"
    TEMPERATURE = "temperature"


@dataclass
class SystemMetric:
    """系统指标"""
    type: MetricType
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'type': self.type.value,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp,
            'labels': self.labels
        }


@dataclass
class CPUInfo:
    """CPU信息"""
    usage_percent: float
    core_count: int
    frequency: float
    per_core_usage: List[float]
    load_average: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'usage_percent': self.usage_percent,
            'core_count': self.core_count,
            'frequency': self.frequency,
            'per_core_usage': self.per_core_usage,
            'load_average': self.load_average
        }


@dataclass
class MemoryInfo:
    """内存信息"""
    total: int
    available: int
    used: int
    free: int
    usage_percent: float
    swap_total: int
    swap_used: int
    swap_free: int
    swap_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total': self.total,
            'available': self.available,
            'used': self.used,
            'free': self.free,
            'usage_percent': self.usage_percent,
            'swap_total': self.swap_total,
            'swap_used': self.swap_used,
            'swap_free': self.swap_free,
            'swap_percent': self.swap_percent
        }


@dataclass
class DiskInfo:
    """磁盘信息"""
    device: str
    mountpoint: str
    fstype: str
    total: int
    used: int
    free: int
    usage_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'device': self.device,
            'mountpoint': self.mountpoint,
            'fstype': self.fstype,
            'total': self.total,
            'used': self.used,
            'free': self.free,
            'usage_percent': self.usage_percent
        }


@dataclass
class NetworkInfo:
    """网络信息"""
    interface: str
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    errors_in: int
    errors_out: int
    drops_in: int
    drops_out: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'interface': self.interface,
            'bytes_sent': self.bytes_sent,
            'bytes_recv': self.bytes_recv,
            'packets_sent': self.packets_sent,
            'packets_recv': self.packets_recv,
            'errors_in': self.errors_in,
            'errors_out': self.errors_out,
            'drops_in': self.drops_in,
            'drops_out': self.drops_out
        }


@dataclass
class ProcessInfo:
    """进程信息"""
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    memory_vms: int
    status: str
    create_time: float
    num_threads: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'pid': self.pid,
            'name': self.name,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_rss': self.memory_rss,
            'memory_vms': self.memory_vms,
            'status': self.status,
            'create_time': self.create_time,
            'num_threads': self.num_threads
        }


@dataclass
class SystemSnapshot:
    """系统快照"""
    timestamp: float
    cpu_info: CPUInfo
    memory_info: MemoryInfo
    disk_info: List[DiskInfo]
    network_info: List[NetworkInfo]
    process_count: int
    uptime: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'cpu_info': self.cpu_info.to_dict(),
            'memory_info': self.memory_info.to_dict(),
            'disk_info': [disk.to_dict() for disk in self.disk_info],
            'network_info': [net.to_dict() for net in self.network_info],
            'process_count': self.process_count,
            'uptime': self.uptime
        }


class SystemMonitor:
    """系统监控器
    
    监控系统级别的性能指标。
    """
    
    def __init__(self, collection_interval: float = 10.0):
        """初始化系统监控器
        
        Args:
            collection_interval: 数据收集间隔（秒）
        """
        self.collection_interval = collection_interval
        self.logger = get_logger("system_monitor")
        
        # 数据存储
        self.metrics_history: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=1000) for metric_type in MetricType
        }
        
        # 快照历史
        self.snapshots: deque = deque(maxlen=100)
        
        # 网络IO基线（用于计算速率）
        self._network_baseline: Optional[Dict[str, Any]] = None
        self._disk_baseline: Optional[Dict[str, Any]] = None
        
        # 监控任务
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # 回调函数
        self.metric_callbacks: List[Callable[[SystemMetric], None]] = []
        self.snapshot_callbacks: List[Callable[[SystemSnapshot], None]] = []
        
        # 系统信息
        self.system_info = self._get_system_info()
        
        self.logger.info(f"系统监控器初始化完成，收集间隔: {collection_interval}秒")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息
        
        Returns:
            系统信息
        """
        try:
            return {
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'boot_time': psutil.boot_time()
            }
        except Exception as e:
            self.logger.error(f"获取系统信息失败: {e}")
            return {}
    
    async def start(self) -> None:
        """启动监控"""
        if self._running:
            self.logger.warning("系统监控器已在运行")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("系统监控器已启动")
    
    async def stop(self) -> None:
        """停止监控"""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("系统监控器已停止")
    
    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                # 收集系统快照
                snapshot = await self._collect_system_snapshot()
                self.snapshots.append(snapshot)
                
                # 提取指标
                metrics = self._extract_metrics(snapshot)
                
                # 存储指标
                for metric in metrics:
                    self.metrics_history[metric.type].append(metric)
                    
                    # 调用回调
                    for callback in self.metric_callbacks:
                        try:
                            callback(metric)
                        except Exception as e:
                            self.logger.error(f"指标回调执行失败: {e}")
                
                # 调用快照回调
                for callback in self.snapshot_callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        self.logger.error(f"快照回调执行失败: {e}")
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环出错: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_snapshot(self) -> SystemSnapshot:
        """收集系统快照
        
        Returns:
            系统快照
        """
        timestamp = time.time()
        
        # 收集CPU信息
        cpu_info = await self._collect_cpu_info()
        
        # 收集内存信息
        memory_info = await self._collect_memory_info()
        
        # 收集磁盘信息
        disk_info = await self._collect_disk_info()
        
        # 收集网络信息
        network_info = await self._collect_network_info()
        
        # 进程数量
        process_count = len(psutil.pids())
        
        # 系统运行时间
        uptime = timestamp - psutil.boot_time()
        
        return SystemSnapshot(
            timestamp=timestamp,
            cpu_info=cpu_info,
            memory_info=memory_info,
            disk_info=disk_info,
            network_info=network_info,
            process_count=process_count,
            uptime=uptime
        )
    
    async def _collect_cpu_info(self) -> CPUInfo:
        """收集CPU信息
        
        Returns:
            CPU信息
        """
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 每核心使用率
        per_core_usage = psutil.cpu_percent(interval=1, percpu=True)
        
        # CPU频率
        cpu_freq = psutil.cpu_freq()
        frequency = cpu_freq.current if cpu_freq else 0.0
        
        # 负载平均值
        try:
            load_avg = psutil.getloadavg()
        except AttributeError:
            # Windows不支持getloadavg
            load_avg = [0.0, 0.0, 0.0]
        
        return CPUInfo(
            usage_percent=cpu_percent,
            core_count=psutil.cpu_count(),
            frequency=frequency,
            per_core_usage=per_core_usage,
            load_average=list(load_avg)
        )
    
    async def _collect_memory_info(self) -> MemoryInfo:
        """收集内存信息
        
        Returns:
            内存信息
        """
        # 虚拟内存
        virtual_memory = psutil.virtual_memory()
        
        # 交换内存
        swap_memory = psutil.swap_memory()
        
        return MemoryInfo(
            total=virtual_memory.total,
            available=virtual_memory.available,
            used=virtual_memory.used,
            free=virtual_memory.free,
            usage_percent=virtual_memory.percent,
            swap_total=swap_memory.total,
            swap_used=swap_memory.used,
            swap_free=swap_memory.free,
            swap_percent=swap_memory.percent
        )
    
    async def _collect_disk_info(self) -> List[DiskInfo]:
        """收集磁盘信息
        
        Returns:
            磁盘信息列表
        """
        disk_info = []
        
        # 获取所有磁盘分区
        partitions = psutil.disk_partitions()
        
        for partition in partitions:
            try:
                # 获取磁盘使用情况
                usage = psutil.disk_usage(partition.mountpoint)
                
                disk_info.append(DiskInfo(
                    device=partition.device,
                    mountpoint=partition.mountpoint,
                    fstype=partition.fstype,
                    total=usage.total,
                    used=usage.used,
                    free=usage.free,
                    usage_percent=(usage.used / usage.total) * 100 if usage.total > 0 else 0
                ))
                
            except (PermissionError, OSError) as e:
                self.logger.debug(f"无法访问磁盘分区 {partition.device}: {e}")
                continue
        
        return disk_info
    
    async def _collect_network_info(self) -> List[NetworkInfo]:
        """收集网络信息
        
        Returns:
            网络信息列表
        """
        network_info = []
        
        # 获取网络IO统计
        net_io = psutil.net_io_counters(pernic=True)
        
        for interface, stats in net_io.items():
            network_info.append(NetworkInfo(
                interface=interface,
                bytes_sent=stats.bytes_sent,
                bytes_recv=stats.bytes_recv,
                packets_sent=stats.packets_sent,
                packets_recv=stats.packets_recv,
                errors_in=stats.errin,
                errors_out=stats.errout,
                drops_in=stats.dropin,
                drops_out=stats.dropout
            ))
        
        return network_info
    
    def _extract_metrics(self, snapshot: SystemSnapshot) -> List[SystemMetric]:
        """从快照中提取指标
        
        Args:
            snapshot: 系统快照
            
        Returns:
            指标列表
        """
        metrics = []
        
        # CPU指标
        metrics.append(SystemMetric(
            type=MetricType.CPU_USAGE,
            value=snapshot.cpu_info.usage_percent,
            unit="percent",
            timestamp=snapshot.timestamp
        ))
        
        # 内存指标
        metrics.append(SystemMetric(
            type=MetricType.MEMORY_USAGE,
            value=snapshot.memory_info.usage_percent,
            unit="percent",
            timestamp=snapshot.timestamp
        ))
        
        # 磁盘指标
        for disk in snapshot.disk_info:
            metrics.append(SystemMetric(
                type=MetricType.DISK_USAGE,
                value=disk.usage_percent,
                unit="percent",
                timestamp=snapshot.timestamp,
                labels={'device': disk.device, 'mountpoint': disk.mountpoint}
            ))
        
        # 进程数量指标
        metrics.append(SystemMetric(
            type=MetricType.PROCESS_COUNT,
            value=snapshot.process_count,
            unit="count",
            timestamp=snapshot.timestamp
        ))
        
        # 负载平均值指标
        if snapshot.cpu_info.load_average:
            metrics.append(SystemMetric(
                type=MetricType.LOAD_AVERAGE,
                value=snapshot.cpu_info.load_average[0],  # 1分钟负载
                unit="load",
                timestamp=snapshot.timestamp,
                labels={'period': '1min'}
            ))
        
        return metrics
    
    def get_current_snapshot(self) -> Optional[SystemSnapshot]:
        """获取当前系统快照
        
        Returns:
            当前系统快照
        """
        if self.snapshots:
            return self.snapshots[-1]
        return None
    
    def get_metric_history(self, metric_type: MetricType, limit: int = 100) -> List[SystemMetric]:
        """获取指标历史
        
        Args:
            metric_type: 指标类型
            limit: 限制数量
            
        Returns:
            指标历史列表
        """
        if metric_type in self.metrics_history:
            history = list(self.metrics_history[metric_type])
            return history[-limit:] if limit > 0 else history
        return []
    
    def get_snapshot_history(self, limit: int = 10) -> List[SystemSnapshot]:
        """获取快照历史
        
        Args:
            limit: 限制数量
            
        Returns:
            快照历史列表
        """
        history = list(self.snapshots)
        return history[-limit:] if limit > 0 else history
    
    def get_top_processes(self, limit: int = 10, sort_by: str = 'cpu') -> List[ProcessInfo]:
        """获取资源使用最高的进程
        
        Args:
            limit: 限制数量
            sort_by: 排序字段（cpu或memory）
            
        Returns:
            进程信息列表
        """
        processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 
                                           'memory_info', 'status', 'create_time', 'num_threads']):
                try:
                    proc_info = proc.info
                    if proc_info['memory_info']:
                        processes.append(ProcessInfo(
                            pid=proc_info['pid'],
                            name=proc_info['name'] or 'Unknown',
                            cpu_percent=proc_info['cpu_percent'] or 0.0,
                            memory_percent=proc_info['memory_percent'] or 0.0,
                            memory_rss=proc_info['memory_info'].rss,
                            memory_vms=proc_info['memory_info'].vms,
                            status=proc_info['status'],
                            create_time=proc_info['create_time'],
                            num_threads=proc_info['num_threads'] or 0
                        ))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # 排序
            if sort_by == 'cpu':
                processes.sort(key=lambda x: x.cpu_percent, reverse=True)
            elif sort_by == 'memory':
                processes.sort(key=lambda x: x.memory_percent, reverse=True)
            
            return processes[:limit]
            
        except Exception as e:
            self.logger.error(f"获取进程信息失败: {e}")
            return []
    
    def add_metric_callback(self, callback: Callable[[SystemMetric], None]) -> None:
        """添加指标回调
        
        Args:
            callback: 回调函数
        """
        self.metric_callbacks.append(callback)
    
    def add_snapshot_callback(self, callback: Callable[[SystemSnapshot], None]) -> None:
        """添加快照回调
        
        Args:
            callback: 回调函数
        """
        self.snapshot_callbacks.append(callback)
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息
        
        Returns:
            系统信息
        """
        return self.system_info.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息
        """
        current_snapshot = self.get_current_snapshot()
        
        stats = {
            'system_info': self.get_system_info(),
            'monitoring_status': {
                'running': self._running,
                'collection_interval': self.collection_interval,
                'snapshots_collected': len(self.snapshots),
                'metrics_collected': sum(len(history) for history in self.metrics_history.values())
            }
        }
        
        if current_snapshot:
            stats['current_status'] = current_snapshot.to_dict()
        
        return stats
    
    async def force_collection(self) -> SystemSnapshot:
        """强制收集一次数据
        
        Returns:
            系统快照
        """
        snapshot = await self._collect_system_snapshot()
        self.snapshots.append(snapshot)
        
        # 提取并存储指标
        metrics = self._extract_metrics(snapshot)
        for metric in metrics:
            self.metrics_history[metric.type].append(metric)
        
        return snapshot
    
    def clear_history(self) -> None:
        """清空历史数据"""
        for history in self.metrics_history.values():
            history.clear()
        self.snapshots.clear()
        self.logger.info("历史数据已清空")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, '_monitor_task') and self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()