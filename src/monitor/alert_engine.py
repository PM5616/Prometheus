"""Alert Engine Module

告警引擎，负责实时监控和告警通知。

主要功能：
- 告警规则管理
- 实时告警检测
- 告警通知发送
- 告警历史记录
- 告警抑制和聚合
"""

import time
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timedelta

from src.common.logging.logger import get_logger
from src.common.exceptions.monitor_exceptions import MonitorError
from src.common.models import AlertLevel, AlertStatus, NotificationChannel
from .system_monitor import SystemMetric, MetricType
from .performance_analyzer import AnomalyDetection, PerformanceThreshold


@dataclass
class AlertRule:
    """告警规则"""
    id: str
    name: str
    description: str
    metric_type: MetricType
    condition: str  # 条件表达式，如 "value > 80"
    level: AlertLevel
    duration: float = 0  # 持续时间（秒）
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    def evaluate(self, value: float) -> bool:
        """评估规则
        
        Args:
            value: 指标值
            
        Returns:
            是否触发告警
        """
        try:
            # 简单的条件评估
            condition = self.condition.replace('value', str(value))
            return eval(condition)
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'metric_type': self.metric_type.value,
            'condition': self.condition,
            'level': self.level.value,
            'duration': self.duration,
            'enabled': self.enabled,
            'labels': self.labels,
            'annotations': self.annotations
        }


@dataclass
class Alert:
    """告警"""
    id: str
    rule_id: str
    rule_name: str
    level: AlertLevel
    status: AlertStatus
    message: str
    metric_type: MetricType
    metric_value: float
    start_time: float
    end_time: Optional[float] = None
    acknowledged_time: Optional[float] = None
    acknowledged_by: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """告警持续时间"""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'level': self.level.value,
            'status': self.status.value,
            'message': self.message,
            'metric_type': self.metric_type.value,
            'metric_value': self.metric_value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'acknowledged_time': self.acknowledged_time,
            'acknowledged_by': self.acknowledged_by,
            'labels': self.labels,
            'annotations': self.annotations
        }


@dataclass
class NotificationConfig:
    """通知配置"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'channel': self.channel.value,
            'enabled': self.enabled,
            'config': self.config
        }


@dataclass
class AlertGroup:
    """告警组"""
    id: str
    name: str
    rules: List[str]  # 规则ID列表
    notification_configs: List[NotificationConfig]
    suppression_rules: List[str] = field(default_factory=list)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'rules': self.rules,
            'notification_configs': [config.to_dict() for config in self.notification_configs],
            'suppression_rules': self.suppression_rules,
            'enabled': self.enabled
        }


class AlertEngine:
    """告警引擎
    
    负责实时监控和告警通知。
    """
    
    def __init__(self, check_interval: float = 10.0):
        """初始化告警引擎
        
        Args:
            check_interval: 检查间隔（秒）
        """
        self.check_interval = check_interval
        self.logger = get_logger("alert_engine")
        
        # 规则和告警存储
        self.rules: Dict[str, AlertRule] = {}
        self.groups: Dict[str, AlertGroup] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # 通知配置
        self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
        
        # 抑制和聚合
        self.suppressed_alerts: Set[str] = set()
        self.alert_counters: Dict[str, int] = defaultdict(int)
        
        # 运行状态
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        
        # 回调函数
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.notification_callbacks: List[Callable[[Alert, NotificationChannel], None]] = []
        
        # 指标缓存（用于持续时间检查）
        self.metric_cache: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=100) for metric_type in MetricType
        }
        
        # 初始化默认规则
        self._init_default_rules()
        
        self.logger.info(f"告警引擎初始化完成，检查间隔: {check_interval}秒")
    
    def _init_default_rules(self) -> None:
        """初始化默认告警规则"""
        default_rules = [
            AlertRule(
                id="cpu_high",
                name="CPU使用率过高",
                description="CPU使用率超过阈值",
                metric_type=MetricType.CPU_USAGE,
                condition="value > 80",
                level=AlertLevel.WARNING,
                duration=60  # 持续1分钟
            ),
            AlertRule(
                id="cpu_critical",
                name="CPU使用率严重过高",
                description="CPU使用率超过严重阈值",
                metric_type=MetricType.CPU_USAGE,
                condition="value > 95",
                level=AlertLevel.CRITICAL,
                duration=30  # 持续30秒
            ),
            AlertRule(
                id="memory_high",
                name="内存使用率过高",
                description="内存使用率超过阈值",
                metric_type=MetricType.MEMORY_USAGE,
                condition="value > 85",
                level=AlertLevel.WARNING,
                duration=60
            ),
            AlertRule(
                id="memory_critical",
                name="内存使用率严重过高",
                description="内存使用率超过严重阈值",
                metric_type=MetricType.MEMORY_USAGE,
                condition="value > 95",
                level=AlertLevel.CRITICAL,
                duration=30
            ),
            AlertRule(
                id="disk_high",
                name="磁盘使用率过高",
                description="磁盘使用率超过阈值",
                metric_type=MetricType.DISK_USAGE,
                condition="value > 90",
                level=AlertLevel.WARNING,
                duration=300  # 持续5分钟
            ),
            AlertRule(
                id="load_high",
                name="系统负载过高",
                description="系统负载超过阈值",
                metric_type=MetricType.LOAD_AVERAGE,
                condition="value > 5.0",
                level=AlertLevel.WARNING,
                duration=120  # 持续2分钟
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
        
        # 创建默认告警组
        default_group = AlertGroup(
            id="system_alerts",
            name="系统告警",
            rules=list(self.rules.keys()),
            notification_configs=[]
        )
        self.groups[default_group.id] = default_group
    
    async def start(self) -> None:
        """启动告警引擎"""
        if self._running:
            self.logger.warning("告警引擎已在运行")
            return
        
        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        self.logger.info("告警引擎已启动")
    
    async def stop(self) -> None:
        """停止告警引擎"""
        if not self._running:
            return
        
        self._running = False
        
        if self._check_task and not self._check_task.done():
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("告警引擎已停止")
    
    async def _check_loop(self) -> None:
        """检查循环"""
        while self._running:
            try:
                await self._check_alerts()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"告警检查循环出错: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_alerts(self) -> None:
        """检查告警"""
        current_time = time.time()
        
        # 检查活跃告警是否需要解决
        alerts_to_resolve = []
        for alert_id, alert in self.active_alerts.items():
            if self._should_resolve_alert(alert):
                alerts_to_resolve.append(alert_id)
        
        # 解决告警
        for alert_id in alerts_to_resolve:
            await self._resolve_alert(alert_id)
    
    def _should_resolve_alert(self, alert: Alert) -> bool:
        """判断告警是否应该解决
        
        Args:
            alert: 告警
            
        Returns:
            是否应该解决
        """
        # 获取对应的规则
        rule = self.rules.get(alert.rule_id)
        if not rule:
            return True  # 规则不存在，解决告警
        
        # 检查最近的指标数据
        recent_metrics = list(self.metric_cache[alert.metric_type])[-10:]  # 最近10个数据点
        
        if not recent_metrics:
            return False
        
        # 检查是否所有最近的数据点都不满足告警条件
        for metric in recent_metrics:
            if rule.evaluate(metric.value):
                return False  # 仍然满足告警条件
        
        return True  # 不再满足告警条件
    
    async def _resolve_alert(self, alert_id: str) -> None:
        """解决告警
        
        Args:
            alert_id: 告警ID
        """
        if alert_id not in self.active_alerts:
            return
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.end_time = time.time()
        
        # 移动到历史记录
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]
        
        # 发送解决通知
        await self._send_notifications(alert)
        
        self.logger.info(f"告警已解决: {alert.rule_name} (持续时间: {alert.duration:.1f}秒)")
    
    def add_metric(self, metric: SystemMetric) -> None:
        """添加指标数据
        
        Args:
            metric: 系统指标
        """
        # 缓存指标数据
        if metric.type in self.metric_cache:
            self.metric_cache[metric.type].append(metric)
        
        # 检查所有相关规则
        for rule in self.rules.values():
            if rule.metric_type == metric.type and rule.enabled:
                self._check_rule(rule, metric)
    
    def _check_rule(self, rule: AlertRule, metric: SystemMetric) -> None:
        """检查规则
        
        Args:
            rule: 告警规则
            metric: 系统指标
        """
        # 评估规则条件
        if not rule.evaluate(metric.value):
            return
        
        # 检查是否已有活跃告警
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.rule_id == rule.id and alert.metric_type == metric.type:
                existing_alert = alert
                break
        
        if existing_alert:
            # 更新现有告警
            existing_alert.metric_value = metric.value
            return
        
        # 检查持续时间要求
        if rule.duration > 0:
            if not self._check_duration(rule, metric):
                return
        
        # 创建新告警
        alert = self._create_alert(rule, metric)
        
        # 检查抑制规则
        if self._is_suppressed(alert):
            self.suppressed_alerts.add(alert.id)
            return
        
        # 添加到活跃告警
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # 发送通知
        asyncio.create_task(self._send_notifications(alert))
        
        # 调用回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"告警回调执行失败: {e}")
        
        self.logger.warning(f"新告警: {alert.rule_name} - {alert.message}")
    
    def _check_duration(self, rule: AlertRule, metric: SystemMetric) -> bool:
        """检查持续时间要求
        
        Args:
            rule: 告警规则
            metric: 系统指标
            
        Returns:
            是否满足持续时间要求
        """
        # 获取最近的指标数据
        recent_metrics = [
            m for m in self.metric_cache[metric.type]
            if metric.timestamp - m.timestamp <= rule.duration
        ]
        
        if not recent_metrics:
            return False
        
        # 检查是否在整个持续时间内都满足条件
        for m in recent_metrics:
            if not rule.evaluate(m.value):
                return False
        
        # 检查时间跨度是否足够
        if recent_metrics:
            time_span = metric.timestamp - recent_metrics[0].timestamp
            return time_span >= rule.duration
        
        return False
    
    def _create_alert(self, rule: AlertRule, metric: SystemMetric) -> Alert:
        """创建告警
        
        Args:
            rule: 告警规则
            metric: 系统指标
            
        Returns:
            告警对象
        """
        alert_id = f"{rule.id}_{metric.type.value}_{int(metric.timestamp)}"
        
        message = f"{rule.name}: {metric.type.value}={metric.value:.2f}"
        if rule.annotations.get('message_template'):
            message = rule.annotations['message_template'].format(
                metric_type=metric.type.value,
                value=metric.value,
                rule_name=rule.name
            )
        
        return Alert(
            id=alert_id,
            rule_id=rule.id,
            rule_name=rule.name,
            level=rule.level,
            status=AlertStatus.ACTIVE,
            message=message,
            metric_type=metric.type,
            metric_value=metric.value,
            start_time=metric.timestamp,
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy()
        )
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """检查告警是否被抑制
        
        Args:
            alert: 告警
            
        Returns:
            是否被抑制
        """
        # 简单的抑制逻辑：相同规则的告警在短时间内只触发一次
        recent_alerts = [
            a for a in self.alert_history
            if a.rule_id == alert.rule_id and 
               alert.start_time - a.start_time < 300  # 5分钟内
        ]
        
        return len(recent_alerts) > 0
    
    async def _send_notifications(self, alert: Alert) -> None:
        """发送通知
        
        Args:
            alert: 告警
        """
        # 查找告警所属的组
        alert_groups = [
            group for group in self.groups.values()
            if alert.rule_id in group.rules and group.enabled
        ]
        
        for group in alert_groups:
            for notification_config in group.notification_configs:
                if notification_config.enabled:
                    await self._send_notification(alert, notification_config)
    
    async def _send_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """发送单个通知
        
        Args:
            alert: 告警
            config: 通知配置
        """
        try:
            # 调用通知回调
            for callback in self.notification_callbacks:
                try:
                    callback(alert, config.channel)
                except Exception as e:
                    self.logger.error(f"通知回调执行失败: {e}")
            
            # 这里可以实现具体的通知发送逻辑
            # 例如：发送邮件、短信、Webhook等
            self.logger.info(f"发送通知: {config.channel.value} - {alert.message}")
            
        except Exception as e:
            self.logger.error(f"发送通知失败: {e}")
    
    def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则
        
        Args:
            rule: 告警规则
        """
        self.rules[rule.id] = rule
        self.logger.info(f"已添加告警规则: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除告警规则
        
        Args:
            rule_id: 规则ID
            
        Returns:
            是否成功移除
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            
            # 解决相关的活跃告警
            alerts_to_resolve = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.rule_id == rule_id
            ]
            
            for alert_id in alerts_to_resolve:
                asyncio.create_task(self._resolve_alert(alert_id))
            
            self.logger.info(f"已移除告警规则: {rule_id}")
            return True
        
        return False
    
    def add_group(self, group: AlertGroup) -> None:
        """添加告警组
        
        Args:
            group: 告警组
        """
        self.groups[group.id] = group
        self.logger.info(f"已添加告警组: {group.name}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认告警
        
        Args:
            alert_id: 告警ID
            acknowledged_by: 确认人
            
        Returns:
            是否成功确认
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_time = time.time()
            alert.acknowledged_by = acknowledged_by
            
            self.logger.info(f"告警已确认: {alert.rule_name} by {acknowledged_by}")
            return True
        
        return False
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """获取活跃告警
        
        Args:
            level: 告警级别过滤
            
        Returns:
            活跃告警列表
        """
        alerts = list(self.active_alerts.values())
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return sorted(alerts, key=lambda x: x.start_time, reverse=True)
    
    def get_alert_history(self, limit: int = 100, level: Optional[AlertLevel] = None) -> List[Alert]:
        """获取告警历史
        
        Args:
            limit: 限制数量
            level: 告警级别过滤
            
        Returns:
            告警历史列表
        """
        history = list(self.alert_history)
        
        if level:
            history = [alert for alert in history if alert.level == level]
        
        # 按时间倒序排列
        history.sort(key=lambda x: x.start_time, reverse=True)
        
        return history[:limit] if limit > 0 else history
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """添加告警回调
        
        Args:
            callback: 回调函数
        """
        self.alert_callbacks.append(callback)
    
    def add_notification_callback(self, callback: Callable[[Alert, NotificationChannel], None]) -> None:
        """添加通知回调
        
        Args:
            callback: 回调函数
        """
        self.notification_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息
        """
        return {
            'engine_status': {
                'running': self._running,
                'check_interval': self.check_interval
            },
            'rules': {
                'total': len(self.rules),
                'enabled': len([r for r in self.rules.values() if r.enabled])
            },
            'groups': {
                'total': len(self.groups),
                'enabled': len([g for g in self.groups.values() if g.enabled])
            },
            'alerts': {
                'active': len(self.active_alerts),
                'suppressed': len(self.suppressed_alerts),
                'total_history': len(self.alert_history)
            },
            'alert_levels': {
                level.value: len([
                    alert for alert in self.active_alerts.values() 
                    if alert.level == level
                ]) for level in AlertLevel
            }
        }
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置
        
        Returns:
            配置数据
        """
        return {
            'rules': {rule_id: rule.to_dict() for rule_id, rule in self.rules.items()},
            'groups': {group_id: group.to_dict() for group_id, group in self.groups.items()},
            'notification_configs': {
                channel.value: config.to_dict() 
                for channel, config in self.notification_configs.items()
            }
        }
    
    def import_config(self, config: Dict[str, Any]) -> None:
        """导入配置
        
        Args:
            config: 配置数据
        """
        # 导入规则
        if 'rules' in config:
            for rule_data in config['rules'].values():
                rule = AlertRule(
                    id=rule_data['id'],
                    name=rule_data['name'],
                    description=rule_data['description'],
                    metric_type=MetricType(rule_data['metric_type']),
                    condition=rule_data['condition'],
                    level=AlertLevel(rule_data['level']),
                    duration=rule_data.get('duration', 0),
                    enabled=rule_data.get('enabled', True),
                    labels=rule_data.get('labels', {}),
                    annotations=rule_data.get('annotations', {})
                )
                self.rules[rule.id] = rule
        
        # 导入组
        if 'groups' in config:
            for group_data in config['groups'].values():
                notification_configs = [
                    NotificationConfig(
                        channel=NotificationChannel(nc['channel']),
                        enabled=nc.get('enabled', True),
                        config=nc.get('config', {})
                    ) for nc in group_data.get('notification_configs', [])
                ]
                
                group = AlertGroup(
                    id=group_data['id'],
                    name=group_data['name'],
                    rules=group_data['rules'],
                    notification_configs=notification_configs,
                    suppression_rules=group_data.get('suppression_rules', []),
                    enabled=group_data.get('enabled', True)
                )
                self.groups[group.id] = group
        
        self.logger.info("配置导入完成")
    
    def clear_history(self) -> None:
        """清空历史数据"""
        self.alert_history.clear()
        self.suppressed_alerts.clear()
        self.alert_counters.clear()
        self.logger.info("告警历史已清空")