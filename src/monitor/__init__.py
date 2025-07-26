"""Monitor Module

监控模块，负责系统监控、性能分析、实时告警等功能。

主要功能：
- 系统监控：CPU、内存、磁盘、网络等系统资源监控
- 性能分析：性能指标收集、分析和报告
- 实时告警：基于规则的告警系统
- 日志分析：日志收集、解析和异常检测
- 资源使用监控：应用程序资源使用情况监控
- 服务健康检查：服务可用性和健康状态监控

核心组件：
- SystemMonitor: 系统级别监控
- PerformanceAnalyzer: 性能分析器
- AlertEngine: 告警引擎
- LogAnalyzer: 日志分析器
- ResourceMonitor: 资源监控器
- HealthChecker: 健康检查器
- MonitorManager: 监控管理器

监控指标：
- 系统指标：CPU使用率、内存使用率、磁盘使用率、网络流量等
- 应用指标：请求响应时间、吞吐量、错误率等
- 业务指标：交易量、用户活跃度等

告警类型：
- 阈值告警：基于指标阈值的告警
- 趋势告警：基于趋势分析的告警
- 异常告警：基于异常检测的告警
- 复合告警：基于多个条件的复合告警
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .system_monitor import SystemMonitor
from .performance_analyzer import PerformanceAnalyzer
from .alert_engine import AlertEngine
from .log_analyzer import LogAnalyzer
from .resource_monitor import ResourceMonitor
from .health_checker import HealthChecker

from ..common.logging.logger import get_logger


@dataclass
class MonitorConfig:
    """监控配置"""
    # 系统监控配置
    system_monitor_enabled: bool = True
    system_collection_interval: float = 5.0
    
    # 性能分析配置
    performance_analyzer_enabled: bool = True
    performance_analysis_interval: float = 60.0
    
    # 告警引擎配置
    alert_engine_enabled: bool = True
    
    # 日志分析配置
    log_analyzer_enabled: bool = True
    
    # 资源监控配置
    resource_monitor_enabled: bool = True
    resource_collection_interval: float = 5.0
    
    # 健康检查配置
    health_checker_enabled: bool = True
    
    # 通用配置
    max_metrics: int = 10000
    max_alerts: int = 1000
    max_logs: int = 10000
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'system_monitor_enabled': self.system_monitor_enabled,
            'system_collection_interval': self.system_collection_interval,
            'performance_analyzer_enabled': self.performance_analyzer_enabled,
            'performance_analysis_interval': self.performance_analysis_interval,
            'alert_engine_enabled': self.alert_engine_enabled,
            'log_analyzer_enabled': self.log_analyzer_enabled,
            'resource_monitor_enabled': self.resource_monitor_enabled,
            'resource_collection_interval': self.resource_collection_interval,
            'health_checker_enabled': self.health_checker_enabled,
            'max_metrics': self.max_metrics,
            'max_alerts': self.max_alerts,
            'max_logs': self.max_logs
        }


class MonitorManager:
    """监控管理器
    
    统一管理所有监控组件。
    """
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        """初始化监控管理器
        
        Args:
            config: 监控配置
        """
        self.config = config or MonitorConfig()
        self.logger = get_logger("monitor_manager")
        
        # 初始化监控组件
        self.system_monitor: Optional[SystemMonitor] = None
        self.performance_analyzer: Optional[PerformanceAnalyzer] = None
        self.alert_engine: Optional[AlertEngine] = None
        self.log_analyzer: Optional[LogAnalyzer] = None
        self.resource_monitor: Optional[ResourceMonitor] = None
        self.health_checker: Optional[HealthChecker] = None
        
        # 运行状态
        self._running = False
        
        # 初始化组件
        self._init_components()
        
        self.logger.info("监控管理器初始化完成")
    
    def _init_components(self) -> None:
        """初始化监控组件"""
        try:
            # 初始化系统监控器
            if self.config.system_monitor_enabled:
                self.system_monitor = SystemMonitor(
                    collection_interval=self.config.system_collection_interval,
                    max_metrics=self.config.max_metrics
                )
                self.logger.info("系统监控器初始化完成")
            
            # 初始化性能分析器
            if self.config.performance_analyzer_enabled:
                self.performance_analyzer = PerformanceAnalyzer(
                    max_metrics=self.config.max_metrics
                )
                self.logger.info("性能分析器初始化完成")
            
            # 初始化告警引擎
            if self.config.alert_engine_enabled:
                self.alert_engine = AlertEngine(
                    max_alerts=self.config.max_alerts
                )
                self.logger.info("告警引擎初始化完成")
            
            # 初始化日志分析器
            if self.config.log_analyzer_enabled:
                self.log_analyzer = LogAnalyzer(
                    max_logs=self.config.max_logs
                )
                self.logger.info("日志分析器初始化完成")
            
            # 初始化资源监控器
            if self.config.resource_monitor_enabled:
                self.resource_monitor = ResourceMonitor(
                    collection_interval=self.config.resource_collection_interval,
                    max_metrics=self.config.max_metrics
                )
                self.logger.info("资源监控器初始化完成")
            
            # 初始化健康检查器
            if self.config.health_checker_enabled:
                self.health_checker = HealthChecker(
                    max_history=self.config.max_metrics
                )
                self.logger.info("健康检查器初始化完成")
            
            # 设置组件间的集成
            self._setup_integrations()
            
        except Exception as e:
            self.logger.error(f"监控组件初始化失败: {e}")
            raise
    
    def _setup_integrations(self) -> None:
        """设置组件间的集成"""
        try:
            # 将系统监控指标发送到性能分析器
            if self.system_monitor and self.performance_analyzer:
                self.system_monitor.add_callback(
                    lambda snapshot: self._forward_system_metrics(snapshot)
                )
            
            # 将资源监控指标发送到告警引擎
            if self.resource_monitor and self.alert_engine:
                self.resource_monitor.add_alert_callback(
                    lambda alert: self._forward_resource_alert(alert)
                )
            
            # 将健康检查结果发送到告警引擎
            if self.health_checker and self.alert_engine:
                self.health_checker.add_check_callback(
                    lambda result: self._forward_health_check(result)
                )
            
            # 将日志告警发送到告警引擎
            if self.log_analyzer and self.alert_engine:
                self.log_analyzer.add_alert_callback(
                    lambda alert: self._forward_log_alert(alert)
                )
            
            self.logger.info("监控组件集成设置完成")
            
        except Exception as e:
            self.logger.error(f"监控组件集成设置失败: {e}")
    
    def _forward_system_metrics(self, snapshot) -> None:
        """转发系统指标到性能分析器
        
        Args:
            snapshot: 系统快照
        """
        if not self.performance_analyzer:
            return
        
        try:
            # 转换系统指标为性能指标
            timestamp = snapshot.timestamp
            
            # CPU指标
            if hasattr(snapshot, 'cpu_info'):
                self.performance_analyzer.add_metric(
                    'cpu_usage', snapshot.cpu_info.usage_percent, timestamp
                )
            
            # 内存指标
            if hasattr(snapshot, 'memory_info'):
                self.performance_analyzer.add_metric(
                    'memory_usage', snapshot.memory_info.usage_percent, timestamp
                )
            
            # 磁盘指标
            if hasattr(snapshot, 'disk_info'):
                for disk in snapshot.disk_info:
                    self.performance_analyzer.add_metric(
                        f'disk_usage_{disk.device}', disk.usage_percent, timestamp
                    )
            
        except Exception as e:
            self.logger.error(f"转发系统指标失败: {e}")
    
    def _forward_resource_alert(self, alert) -> None:
        """转发资源告警到告警引擎
        
        Args:
            alert: 资源告警
        """
        if not self.alert_engine:
            return
        
        try:
            # 转换资源告警为通用告警
            from .alert_engine import Alert, AlertLevel as EngineAlertLevel
            
            # 映射告警级别
            level_mapping = {
                'warning': EngineAlertLevel.WARNING,
                'critical': EngineAlertLevel.CRITICAL
            }
            
            engine_alert = Alert(
                id=alert.id,
                name=f"resource_{alert.resource_type.value}_{alert.metric_name}",
                level=level_mapping.get(alert.level.value, EngineAlertLevel.WARNING),
                message=alert.message,
                timestamp=alert.timestamp,
                source="resource_monitor",
                tags={
                    'resource_type': alert.resource_type.value,
                    'metric_name': alert.metric_name,
                    'threshold': str(alert.threshold),
                    'current_value': str(alert.current_value)
                }
            )
            
            self.alert_engine._add_alert(engine_alert)
            
        except Exception as e:
            self.logger.error(f"转发资源告警失败: {e}")
    
    def _forward_health_check(self, result) -> None:
        """转发健康检查结果到告警引擎
        
        Args:
            result: 健康检查结果
        """
        if not self.alert_engine or result.status.value == 'healthy':
            return
        
        try:
            from .alert_engine import Alert, AlertLevel as EngineAlertLevel
            
            # 根据健康状态映射告警级别
            level_mapping = {
                'unhealthy': EngineAlertLevel.CRITICAL,
                'degraded': EngineAlertLevel.WARNING,
                'unknown': EngineAlertLevel.INFO
            }
            
            engine_alert = Alert(
                id=f"health_{result.check_name}_{result.status.value}",
                name=f"health_check_{result.check_name}",
                level=level_mapping.get(result.status.value, EngineAlertLevel.WARNING),
                message=result.error or result.message or f"健康检查失败: {result.check_name}",
                timestamp=result.timestamp,
                source="health_checker",
                tags={
                    'check_name': result.check_name,
                    'status': result.status.value,
                    'response_time': str(result.response_time)
                }
            )
            
            self.alert_engine._add_alert(engine_alert)
            
        except Exception as e:
            self.logger.error(f"转发健康检查结果失败: {e}")
    
    def _forward_log_alert(self, alert) -> None:
        """转发日志告警到告警引擎
        
        Args:
            alert: 日志告警
        """
        if not self.alert_engine:
            return
        
        try:
            from .alert_engine import Alert, AlertLevel as EngineAlertLevel
            
            # 映射告警级别
            level_mapping = {
                'info': EngineAlertLevel.INFO,
                'warning': EngineAlertLevel.WARNING,
                'error': EngineAlertLevel.ERROR,
                'critical': EngineAlertLevel.CRITICAL
            }
            
            engine_alert = Alert(
                id=f"log_{alert.id}",
                name=f"log_alert_{alert.pattern_name}",
                level=level_mapping.get(alert.level.value, EngineAlertLevel.WARNING),
                message=alert.message,
                timestamp=alert.timestamp,
                source="log_analyzer",
                tags={
                    'pattern_name': alert.pattern_name,
                    'log_level': alert.level.value,
                    'count': str(alert.count)
                }
            )
            
            self.alert_engine._add_alert(engine_alert)
            
        except Exception as e:
            self.logger.error(f"转发日志告警失败: {e}")
    
    async def start(self) -> None:
        """启动监控管理器"""
        if self._running:
            self.logger.warning("监控管理器已在运行")
            return
        
        self._running = True
        
        try:
            # 启动各个组件
            if self.system_monitor:
                await self.system_monitor.start()
            
            if self.performance_analyzer:
                await self.performance_analyzer.start()
            
            if self.alert_engine:
                await self.alert_engine.start()
            
            if self.log_analyzer:
                await self.log_analyzer.start()
            
            if self.resource_monitor:
                await self.resource_monitor.start()
            
            if self.health_checker:
                await self.health_checker.start()
            
            self.logger.info("监控管理器已启动")
            
        except Exception as e:
            self.logger.error(f"监控管理器启动失败: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """停止监控管理器"""
        if not self._running:
            return
        
        self._running = False
        
        try:
            # 停止各个组件
            if self.health_checker:
                await self.health_checker.stop()
            
            if self.resource_monitor:
                await self.resource_monitor.stop()
            
            if self.log_analyzer:
                await self.log_analyzer.stop()
            
            if self.alert_engine:
                await self.alert_engine.stop()
            
            if self.performance_analyzer:
                await self.performance_analyzer.stop()
            
            if self.system_monitor:
                await self.system_monitor.stop()
            
            self.logger.info("监控管理器已停止")
            
        except Exception as e:
            self.logger.error(f"监控管理器停止失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取监控状态
        
        Returns:
            监控状态信息
        """
        status = {
            'running': self._running,
            'components': {},
            'config': self.config.to_dict()
        }
        
        # 获取各组件状态
        if self.system_monitor:
            status['components']['system_monitor'] = {
                'running': self.system_monitor._running,
                'metrics_count': len(self.system_monitor.metrics)
            }
        
        if self.performance_analyzer:
            status['components']['performance_analyzer'] = {
                'running': self.performance_analyzer._running,
                'metrics_count': len(self.performance_analyzer.metrics)
            }
        
        if self.alert_engine:
            status['components']['alert_engine'] = {
                'running': self.alert_engine._running,
                'active_alerts': len([a for a in self.alert_engine.alerts.values() if not a.resolved]),
                'total_alerts': len(self.alert_engine.alerts)
            }
        
        if self.log_analyzer:
            status['components']['log_analyzer'] = {
                'running': self.log_analyzer._running,
                'logs_count': len(self.log_analyzer.logs),
                'monitored_files': len(self.log_analyzer.monitored_files)
            }
        
        if self.resource_monitor:
            status['components']['resource_monitor'] = {
                'running': self.resource_monitor._running,
                'metrics_count': len(self.resource_monitor.metrics),
                'active_alerts': len([a for a in self.resource_monitor.alerts.values() if not a.resolved])
            }
        
        if self.health_checker:
            status['components']['health_checker'] = {
                'running': self.health_checker._running,
                'checks_count': len(self.health_checker.checks),
                'services_count': len(self.health_checker.service_health)
            }
        
        return status
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要
        
        Returns:
            监控摘要信息
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'unknown',
            'active_alerts': 0,
            'system_stats': {},
            'resource_stats': {},
            'service_health': {}
        }
        
        try:
            # 获取整体健康状态
            if self.health_checker:
                summary['overall_health'] = self.health_checker.get_overall_health().value
                summary['service_health'] = {
                    name: health.to_dict() 
                    for name, health in self.health_checker.get_service_health().items()
                }
            
            # 获取活跃告警数量
            if self.alert_engine:
                active_alerts = [a for a in self.alert_engine.alerts.values() if not a.resolved]
                summary['active_alerts'] = len(active_alerts)
            
            # 获取系统统计
            if self.system_monitor and hasattr(self.system_monitor, 'get_latest_snapshot'):
                latest = self.system_monitor.get_latest_snapshot()
                if latest:
                    summary['system_stats'] = {
                        'cpu_usage': getattr(latest.cpu_info, 'usage_percent', 0) if hasattr(latest, 'cpu_info') else 0,
                        'memory_usage': getattr(latest.memory_info, 'usage_percent', 0) if hasattr(latest, 'memory_info') else 0,
                        'timestamp': latest.timestamp
                    }
            
            # 获取资源统计
            if self.resource_monitor:
                stats = self.resource_monitor.get_stats()
                summary['resource_stats'] = stats.to_dict()
            
        except Exception as e:
            self.logger.error(f"获取监控摘要失败: {e}")
        
        return summary


# 导出主要类和函数
__all__ = [
    'MonitorConfig',
    'MonitorManager',
    'SystemMonitor',
    'PerformanceAnalyzer',
    'AlertEngine',
    'LogAnalyzer',
    'ResourceMonitor',
    'HealthChecker'
]

__version__ = '1.0.0'
__author__ = 'Prometheus Team'
__description__ = 'Comprehensive monitoring and alerting system'