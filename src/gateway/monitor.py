"""Gateway Monitor Module

API网关监控模块，负责指标收集、监控和告警。

主要功能：
- 请求指标收集
- 性能监控
- 健康检查
- 告警管理
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import psutil
import aiohttp

from ..common.logging.logger import get_logger
from ..common.exceptions.gateway_exceptions import MonitorError


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"                  # 计数器
    GAUGE = "gauge"                      # 仪表盘
    HISTOGRAM = "histogram"              # 直方图
    SUMMARY = "summary"                  # 摘要
    TIMER = "timer"                      # 计时器


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"                        # 信息
    WARNING = "warning"                  # 警告
    ERROR = "error"                      # 错误
    CRITICAL = "critical"                # 严重


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"                  # 健康
    DEGRADED = "degraded"                # 降级
    UNHEALTHY = "unhealthy"              # 不健康
    UNKNOWN = "unknown"                  # 未知


@dataclass
class MetricConfig:
    """指标配置"""
    name: str
    type: MetricType
    description: str = ""
    labels: List[str] = field(default_factory=list)
    buckets: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.5, 5.0, 10.0])
    quantiles: List[float] = field(default_factory=lambda: [0.5, 0.9, 0.95, 0.99])
    max_age: int = 600                   # 最大保存时间（秒）
    enabled: bool = True


@dataclass
class Metric:
    """指标数据"""
    name: str
    type: MetricType
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'type': self.type.value,
            'value': self.value,
            'labels': self.labels,
            'timestamp': self.timestamp
        }


@dataclass
class Alert:
    """告警信息"""
    id: str
    level: AlertLevel
    title: str
    message: str
    metric_name: str = ""
    metric_value: Union[int, float] = 0
    threshold: Union[int, float] = 0
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'labels': self.labels,
            'timestamp': self.timestamp,
            'resolved': self.resolved,
            'resolved_time': self.resolved_time
        }


@dataclass
class HealthCheck:
    """健康检查配置"""
    name: str
    url: str
    method: str = "GET"
    timeout: float = 5.0
    interval: float = 30.0
    headers: Dict[str, str] = field(default_factory=dict)
    expected_status: int = 200
    expected_content: Optional[str] = None
    enabled: bool = True


@dataclass
class HealthResult:
    """健康检查结果"""
    name: str
    status: HealthStatus
    response_time: float
    status_code: Optional[int] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'status': self.status.value,
            'response_time': self.response_time,
            'status_code': self.status_code,
            'error': self.error,
            'timestamp': self.timestamp
        }


class MetricCollector:
    """指标收集器
    
    负责收集和存储各种指标数据。
    """
    
    def __init__(self):
        """初始化指标收集器"""
        self.logger = get_logger("metric_collector")
        
        # 指标配置
        self.configs: Dict[str, MetricConfig] = {}
        
        # 指标数据存储
        self.counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.histograms: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.timers: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        
        # 时间序列数据
        self.time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 注册默认指标
        self._register_default_metrics()
    
    def _register_default_metrics(self) -> None:
        """注册默认指标"""
        default_metrics = [
            MetricConfig("gateway_requests_total", MetricType.COUNTER, "总请求数", ["method", "path", "status"]),
            MetricConfig("gateway_request_duration", MetricType.HISTOGRAM, "请求耗时", ["method", "path"]),
            MetricConfig("gateway_active_connections", MetricType.GAUGE, "活跃连接数"),
            MetricConfig("gateway_error_rate", MetricType.GAUGE, "错误率"),
            MetricConfig("gateway_throughput", MetricType.GAUGE, "吞吐量"),
            MetricConfig("gateway_memory_usage", MetricType.GAUGE, "内存使用率"),
            MetricConfig("gateway_cpu_usage", MetricType.GAUGE, "CPU使用率"),
        ]
        
        for metric in default_metrics:
            self.register_metric(metric)
    
    def register_metric(self, config: MetricConfig) -> None:
        """注册指标
        
        Args:
            config: 指标配置
        """
        self.configs[config.name] = config
        self.logger.debug(f"注册指标: {config.name} ({config.type.value})")
    
    def increment(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """增加计数器
        
        Args:
            name: 指标名称
            value: 增加值
            labels: 标签
        """
        if name not in self.configs or self.configs[name].type != MetricType.COUNTER:
            return
        
        if not self.configs[name].enabled:
            return
        
        labels = labels or {}
        label_key = self._get_label_key(labels)
        self.counters[name][label_key] += value
        
        # 记录时间序列
        metric = Metric(name, MetricType.COUNTER, self.counters[name][label_key], labels)
        self.time_series[name].append(metric)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """设置仪表盘值
        
        Args:
            name: 指标名称
            value: 值
            labels: 标签
        """
        if name not in self.configs or self.configs[name].type != MetricType.GAUGE:
            return
        
        if not self.configs[name].enabled:
            return
        
        labels = labels or {}
        label_key = self._get_label_key(labels)
        self.gauges[name][label_key] = value
        
        # 记录时间序列
        metric = Metric(name, MetricType.GAUGE, value, labels)
        self.time_series[name].append(metric)
    
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """观察直方图值
        
        Args:
            name: 指标名称
            value: 观察值
            labels: 标签
        """
        if name not in self.configs or self.configs[name].type != MetricType.HISTOGRAM:
            return
        
        if not self.configs[name].enabled:
            return
        
        labels = labels or {}
        label_key = self._get_label_key(labels)
        self.histograms[name][label_key].append(value)
        
        # 限制历史数据大小
        if len(self.histograms[name][label_key]) > 1000:
            self.histograms[name][label_key] = self.histograms[name][label_key][-1000:]
        
        # 记录时间序列
        metric = Metric(name, MetricType.HISTOGRAM, value, labels)
        self.time_series[name].append(metric)
    
    def time_operation(self, name: str, labels: Optional[Dict[str, str]] = None):
        """计时装饰器
        
        Args:
            name: 指标名称
            labels: 标签
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.observe(name, duration, labels)
            return wrapper
        return decorator
    
    def _get_label_key(self, labels: Dict[str, str]) -> str:
        """获取标签键
        
        Args:
            labels: 标签字典
            
        Returns:
            标签键字符串
        """
        if not labels:
            return "__default__"
        
        # 按键排序确保一致性
        sorted_items = sorted(labels.items())
        return "|".join(f"{k}={v}" for k, v in sorted_items)
    
    def get_metric_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Union[int, float]]:
        """获取指标值
        
        Args:
            name: 指标名称
            labels: 标签
            
        Returns:
            指标值
        """
        if name not in self.configs:
            return None
        
        label_key = self._get_label_key(labels or {})
        config = self.configs[name]
        
        if config.type == MetricType.COUNTER:
            return self.counters[name].get(label_key, 0)
        elif config.type == MetricType.GAUGE:
            return self.gauges[name].get(label_key, 0.0)
        elif config.type == MetricType.HISTOGRAM:
            values = self.histograms[name].get(label_key, [])
            return sum(values) / len(values) if values else 0.0
        
        return None
    
    def get_all_metrics(self) -> List[Metric]:
        """获取所有指标
        
        Returns:
            指标列表
        """
        metrics = []
        current_time = time.time()
        
        # 计数器指标
        for name, label_data in self.counters.items():
            if name in self.configs and self.configs[name].enabled:
                for label_key, value in label_data.items():
                    labels = self._parse_label_key(label_key)
                    metrics.append(Metric(name, MetricType.COUNTER, value, labels, current_time))
        
        # 仪表盘指标
        for name, label_data in self.gauges.items():
            if name in self.configs and self.configs[name].enabled:
                for label_key, value in label_data.items():
                    labels = self._parse_label_key(label_key)
                    metrics.append(Metric(name, MetricType.GAUGE, value, labels, current_time))
        
        # 直方图指标
        for name, label_data in self.histograms.items():
            if name in self.configs and self.configs[name].enabled:
                for label_key, values in label_data.items():
                    if values:
                        labels = self._parse_label_key(label_key)
                        avg_value = sum(values) / len(values)
                        metrics.append(Metric(name, MetricType.HISTOGRAM, avg_value, labels, current_time))
        
        return metrics
    
    def _parse_label_key(self, label_key: str) -> Dict[str, str]:
        """解析标签键
        
        Args:
            label_key: 标签键字符串
            
        Returns:
            标签字典
        """
        if label_key == "__default__":
            return {}
        
        labels = {}
        for item in label_key.split("|"):
            if "=" in item:
                k, v = item.split("=", 1)
                labels[k] = v
        
        return labels
    
    def reset_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """重置指标
        
        Args:
            name: 指标名称
            labels: 标签
        """
        if name not in self.configs:
            return
        
        label_key = self._get_label_key(labels or {})
        config = self.configs[name]
        
        if config.type == MetricType.COUNTER:
            if name in self.counters and label_key in self.counters[name]:
                self.counters[name][label_key] = 0
        elif config.type == MetricType.GAUGE:
            if name in self.gauges and label_key in self.gauges[name]:
                self.gauges[name][label_key] = 0.0
        elif config.type == MetricType.HISTOGRAM:
            if name in self.histograms and label_key in self.histograms[name]:
                self.histograms[name][label_key].clear()
        
        self.logger.debug(f"重置指标: {name} (labels: {labels})")


class AlertManager:
    """告警管理器
    
    负责告警规则管理和告警通知。
    """
    
    def __init__(self, collector: MetricCollector):
        """初始化告警管理器
        
        Args:
            collector: 指标收集器
        """
        self.collector = collector
        self.logger = get_logger("alert_manager")
        
        # 告警规则
        self.rules: Dict[str, Dict[str, Any]] = {}
        
        # 活跃告警
        self.active_alerts: Dict[str, Alert] = {}
        
        # 告警历史
        self.alert_history: deque = deque(maxlen=1000)
        
        # 告警回调
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # 监控任务
        self._monitor_task: Optional[asyncio.Task] = None
        
        # 注册默认告警规则
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """注册默认告警规则"""
        default_rules = {
            "high_error_rate": {
                "metric": "gateway_error_rate",
                "condition": ">",
                "threshold": 0.1,
                "level": AlertLevel.WARNING,
                "title": "高错误率告警",
                "message": "网关错误率超过10%"
            },
            "high_memory_usage": {
                "metric": "gateway_memory_usage",
                "condition": ">",
                "threshold": 0.8,
                "level": AlertLevel.WARNING,
                "title": "高内存使用率告警",
                "message": "网关内存使用率超过80%"
            },
            "high_cpu_usage": {
                "metric": "gateway_cpu_usage",
                "condition": ">",
                "threshold": 0.8,
                "level": AlertLevel.WARNING,
                "title": "高CPU使用率告警",
                "message": "网关CPU使用率超过80%"
            }
        }
        
        for rule_id, rule in default_rules.items():
            self.add_rule(rule_id, rule)
    
    def add_rule(self, rule_id: str, rule: Dict[str, Any]) -> None:
        """添加告警规则
        
        Args:
            rule_id: 规则ID
            rule: 规则配置
        """
        self.rules[rule_id] = rule
        self.logger.info(f"添加告警规则: {rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除告警规则
        
        Args:
            rule_id: 规则ID
            
        Returns:
            是否成功移除
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"移除告警规则: {rule_id}")
            return True
        return False
    
    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """添加告警回调
        
        Args:
            callback: 告警回调函数
        """
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self) -> None:
        """开始监控"""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            self.logger.info("开始告警监控")
    
    async def stop_monitoring(self) -> None:
        """停止监控"""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self.logger.info("停止告警监控")
    
    async def _monitor_loop(self) -> None:
        """监控循环"""
        while True:
            try:
                await self._check_rules()
                await asyncio.sleep(10)  # 每10秒检查一次
            except Exception as e:
                self.logger.error(f"告警监控出错: {e}")
                await asyncio.sleep(10)
    
    async def _check_rules(self) -> None:
        """检查告警规则"""
        for rule_id, rule in self.rules.items():
            try:
                metric_name = rule["metric"]
                condition = rule["condition"]
                threshold = rule["threshold"]
                level = AlertLevel(rule["level"]) if isinstance(rule["level"], str) else rule["level"]
                
                # 获取指标值
                metric_value = self.collector.get_metric_value(metric_name)
                if metric_value is None:
                    continue
                
                # 检查条件
                triggered = False
                if condition == ">" and metric_value > threshold:
                    triggered = True
                elif condition == "<" and metric_value < threshold:
                    triggered = True
                elif condition == ">=" and metric_value >= threshold:
                    triggered = True
                elif condition == "<=" and metric_value <= threshold:
                    triggered = True
                elif condition == "==" and metric_value == threshold:
                    triggered = True
                elif condition == "!=" and metric_value != threshold:
                    triggered = True
                
                if triggered:
                    # 触发告警
                    if rule_id not in self.active_alerts:
                        alert = Alert(
                            id=rule_id,
                            level=level,
                            title=rule["title"],
                            message=rule["message"],
                            metric_name=metric_name,
                            metric_value=metric_value,
                            threshold=threshold
                        )
                        
                        self.active_alerts[rule_id] = alert
                        self.alert_history.append(alert)
                        
                        # 调用回调
                        for callback in self.alert_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                self.logger.error(f"告警回调执行失败: {e}")
                        
                        self.logger.warning(f"触发告警: {alert.title} (值: {metric_value}, 阈值: {threshold})")
                else:
                    # 解除告警
                    if rule_id in self.active_alerts:
                        alert = self.active_alerts[rule_id]
                        alert.resolved = True
                        alert.resolved_time = time.time()
                        
                        del self.active_alerts[rule_id]
                        
                        self.logger.info(f"解除告警: {alert.title}")
                        
            except Exception as e:
                self.logger.error(f"检查告警规则 {rule_id} 时出错: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警
        
        Returns:
            活跃告警列表
        """
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史
        
        Args:
            limit: 限制数量
            
        Returns:
            告警历史列表
        """
        return list(self.alert_history)[-limit:]


class HealthMonitor:
    """健康监控器
    
    负责服务健康检查和状态监控。
    """
    
    def __init__(self):
        """初始化健康监控器"""
        self.logger = get_logger("health_monitor")
        
        # 健康检查配置
        self.checks: Dict[str, HealthCheck] = {}
        
        # 健康检查结果
        self.results: Dict[str, HealthResult] = {}
        
        # 监控任务
        self._monitor_tasks: Dict[str, asyncio.Task] = {}
        
        # HTTP会话
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> None:
        """初始化异步组件"""
        self._session = aiohttp.ClientSession()
        self.logger.info("健康监控器初始化完成")
    
    def add_check(self, check: HealthCheck) -> None:
        """添加健康检查
        
        Args:
            check: 健康检查配置
        """
        self.checks[check.name] = check
        
        # 启动监控任务
        if check.enabled:
            self._start_check_task(check.name)
        
        self.logger.info(f"添加健康检查: {check.name}")
    
    def remove_check(self, name: str) -> bool:
        """移除健康检查
        
        Args:
            name: 检查名称
            
        Returns:
            是否成功移除
        """
        if name in self.checks:
            # 停止监控任务
            self._stop_check_task(name)
            
            del self.checks[name]
            if name in self.results:
                del self.results[name]
            
            self.logger.info(f"移除健康检查: {name}")
            return True
        return False
    
    def _start_check_task(self, name: str) -> None:
        """启动检查任务
        
        Args:
            name: 检查名称
        """
        if name in self._monitor_tasks:
            self._stop_check_task(name)
        
        task = asyncio.create_task(self._check_loop(name))
        self._monitor_tasks[name] = task
    
    def _stop_check_task(self, name: str) -> None:
        """停止检查任务
        
        Args:
            name: 检查名称
        """
        if name in self._monitor_tasks:
            task = self._monitor_tasks[name]
            if not task.done():
                task.cancel()
            del self._monitor_tasks[name]
    
    async def _check_loop(self, name: str) -> None:
        """检查循环
        
        Args:
            name: 检查名称
        """
        while True:
            try:
                if name in self.checks:
                    check = self.checks[name]
                    if check.enabled:
                        result = await self._perform_check(check)
                        self.results[name] = result
                    
                    await asyncio.sleep(check.interval)
                else:
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"健康检查 {name} 出错: {e}")
                await asyncio.sleep(30)
    
    async def _perform_check(self, check: HealthCheck) -> HealthResult:
        """执行健康检查
        
        Args:
            check: 健康检查配置
            
        Returns:
            健康检查结果
        """
        start_time = time.time()
        
        try:
            if not self._session:
                await self.initialize()
            
            async with self._session.request(
                method=check.method,
                url=check.url,
                headers=check.headers,
                timeout=aiohttp.ClientTimeout(total=check.timeout)
            ) as response:
                response_time = time.time() - start_time
                
                # 检查状态码
                if response.status == check.expected_status:
                    # 检查内容（如果指定）
                    if check.expected_content:
                        content = await response.text()
                        if check.expected_content in content:
                            status = HealthStatus.HEALTHY
                        else:
                            status = HealthStatus.UNHEALTHY
                    else:
                        status = HealthStatus.HEALTHY
                else:
                    status = HealthStatus.UNHEALTHY
                
                return HealthResult(
                    name=check.name,
                    status=status,
                    response_time=response_time,
                    status_code=response.status
                )
                
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return HealthResult(
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                error="Timeout"
            )
        except Exception as e:
            response_time = time.time() - start_time
            return HealthResult(
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                error=str(e)
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取整体健康状态
        
        Returns:
            健康状态信息
        """
        if not self.results:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'checks': {},
                'summary': {
                    'total': 0,
                    'healthy': 0,
                    'unhealthy': 0,
                    'degraded': 0
                }
            }
        
        summary = {
            'total': len(self.results),
            'healthy': 0,
            'unhealthy': 0,
            'degraded': 0
        }
        
        checks = {}
        for name, result in self.results.items():
            checks[name] = result.to_dict()
            
            if result.status == HealthStatus.HEALTHY:
                summary['healthy'] += 1
            elif result.status == HealthStatus.UNHEALTHY:
                summary['unhealthy'] += 1
            elif result.status == HealthStatus.DEGRADED:
                summary['degraded'] += 1
        
        # 确定整体状态
        if summary['unhealthy'] > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif summary['degraded'] > 0:
            overall_status = HealthStatus.DEGRADED
        elif summary['healthy'] > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return {
            'status': overall_status.value,
            'checks': checks,
            'summary': summary
        }
    
    async def close(self) -> None:
        """关闭健康监控器"""
        # 停止所有监控任务
        for name in list(self._monitor_tasks.keys()):
            self._stop_check_task(name)
        
        # 等待任务完成
        if self._monitor_tasks:
            await asyncio.gather(*self._monitor_tasks.values(), return_exceptions=True)
        
        # 关闭HTTP会话
        if self._session:
            await self._session.close()
        
        self.logger.info("健康监控器已关闭")


class GatewayMonitor:
    """网关监控器
    
    整合指标收集、告警管理和健康监控。
    """
    
    def __init__(self):
        """初始化网关监控器"""
        self.logger = get_logger("gateway_monitor")
        
        # 组件
        self.collector = MetricCollector()
        self.alert_manager = AlertManager(self.collector)
        self.health_monitor = HealthMonitor()
        
        # 系统监控任务
        self._system_monitor_task: Optional[asyncio.Task] = None
        
        self.logger.info("网关监控器初始化完成")
    
    async def initialize(self) -> None:
        """初始化异步组件"""
        await self.health_monitor.initialize()
        await self.alert_manager.start_monitoring()
        
        # 启动系统监控
        self._system_monitor_task = asyncio.create_task(self._system_monitor_loop())
        
        self.logger.info("网关监控器启动完成")
    
    async def _system_monitor_loop(self) -> None:
        """系统监控循环"""
        while True:
            try:
                # 收集系统指标
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                self.collector.set_gauge("gateway_cpu_usage", cpu_percent / 100.0)
                self.collector.set_gauge("gateway_memory_usage", memory.percent / 100.0)
                
                await asyncio.sleep(30)  # 每30秒收集一次
            except Exception as e:
                self.logger.error(f"系统监控出错: {e}")
                await asyncio.sleep(30)
    
    def record_request(self, method: str, path: str, status_code: int, duration: float) -> None:
        """记录请求指标
        
        Args:
            method: HTTP方法
            path: 请求路径
            status_code: 状态码
            duration: 请求耗时
        """
        labels = {
            'method': method,
            'path': path,
            'status': str(status_code)
        }
        
        # 记录请求总数
        self.collector.increment("gateway_requests_total", 1, labels)
        
        # 记录请求耗时
        duration_labels = {'method': method, 'path': path}
        self.collector.observe("gateway_request_duration", duration, duration_labels)
        
        # 计算错误率
        self._update_error_rate()
    
    def _update_error_rate(self) -> None:
        """更新错误率"""
        try:
            total_requests = 0
            error_requests = 0
            
            # 统计所有请求
            for label_key, count in self.collector.counters["gateway_requests_total"].items():
                total_requests += count
                
                # 解析标签获取状态码
                labels = self.collector._parse_label_key(label_key)
                status = labels.get('status', '200')
                
                if status.startswith('4') or status.startswith('5'):
                    error_requests += count
            
            if total_requests > 0:
                error_rate = error_requests / total_requests
                self.collector.set_gauge("gateway_error_rate", error_rate)
            
        except Exception as e:
            self.logger.error(f"更新错误率失败: {e}")
    
    def set_active_connections(self, count: int) -> None:
        """设置活跃连接数
        
        Args:
            count: 连接数
        """
        self.collector.set_gauge("gateway_active_connections", count)
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """获取所有指标
        
        Returns:
            指标列表
        """
        metrics = self.collector.get_all_metrics()
        return [metric.to_dict() for metric in metrics]
    
    def get_alerts(self) -> Dict[str, Any]:
        """获取告警信息
        
        Returns:
            告警信息
        """
        return {
            'active_alerts': [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
            'alert_history': [alert.to_dict() for alert in self.alert_manager.get_alert_history()]
        }
    
    def get_health(self) -> Dict[str, Any]:
        """获取健康状态
        
        Returns:
            健康状态
        """
        return self.health_monitor.get_health_status()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取监控面板数据
        
        Returns:
            监控面板数据
        """
        return {
            'metrics': self.get_metrics(),
            'alerts': self.get_alerts(),
            'health': self.get_health(),
            'timestamp': time.time()
        }
    
    async def close(self) -> None:
        """关闭监控器"""
        # 停止系统监控
        if self._system_monitor_task and not self._system_monitor_task.done():
            self._system_monitor_task.cancel()
            try:
                await self._system_monitor_task
            except asyncio.CancelledError:
                pass
        
        # 关闭组件
        await self.alert_manager.stop_monitoring()
        await self.health_monitor.close()
        
        self.logger.info("网关监控器已关闭")