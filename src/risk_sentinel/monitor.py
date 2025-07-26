"""Risk Monitor

风险监控器 - 负责实时监控各种风险指标
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from collections import defaultdict, deque
from dataclasses import dataclass
import threading
import time

from .models import (
    RiskType, RiskLevel, AlertType, AlertStatus, RiskMetric, 
    RiskAlert, RiskLimit, RiskEvent, RiskConfig
)
from src.common.logging import get_logger
from src.common.events import EventBus
from src.portfolio_manager.models import Portfolio, Position


class RiskMonitor:
    """风险监控器
    
    负责:
    1. 实时监控各种风险指标
    2. 检测风险阈值突破
    3. 生成风险预警
    4. 异常检测
    5. 趋势分析
    """
    
    def __init__(self, config: RiskConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.logger = get_logger(self.__class__.__name__)
        
        # 监控状态
        self.is_running = False
        self.is_paused = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 风险指标存储
        self.risk_metrics: Dict[str, RiskMetric] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 风险限制
        self.risk_limits: Dict[str, RiskLimit] = {}
        
        # 预警管理
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []
        self.last_alert_time: Dict[str, datetime] = {}
        
        # 监控回调
        self.metric_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.alert_callbacks: List[Callable] = []
        
        # 异常检测
        self.anomaly_detectors: Dict[str, 'AnomalyDetector'] = {}
        
        # 统计信息
        self.monitoring_stats = {
            'total_metrics_processed': 0,
            'total_alerts_generated': 0,
            'total_anomalies_detected': 0,
            'last_update_time': None
        }
        
        self._setup_default_limits()
        self._setup_anomaly_detectors()
    
    def start(self) -> None:
        """启动风险监控"""
        if self.is_running:
            self.logger.warning("Risk monitor is already running")
            return
        
        self.logger.info("Starting risk monitor")
        self.is_running = True
        self.is_paused = False
        self._stop_event.clear()
        
        # 启动监控线程
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        # 发布启动事件
        self.event_bus.publish('risk_monitor.started', {'timestamp': datetime.now()})
    
    def stop(self) -> None:
        """停止风险监控"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping risk monitor")
        self.is_running = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        # 发布停止事件
        self.event_bus.publish('risk_monitor.stopped', {'timestamp': datetime.now()})
    
    def pause(self) -> None:
        """暂停监控"""
        self.is_paused = True
        self.logger.info("Risk monitor paused")
    
    def resume(self) -> None:
        """恢复监控"""
        self.is_paused = False
        self.logger.info("Risk monitor resumed")
    
    def add_risk_limit(self, limit: RiskLimit) -> None:
        """添加风险限制"""
        self.risk_limits[limit.name] = limit
        self.logger.info(f"Added risk limit: {limit.name} = {limit.limit_value}")
    
    def update_risk_limit(self, name: str, new_limit: float) -> None:
        """更新风险限制"""
        if name in self.risk_limits:
            old_limit = self.risk_limits[name].limit_value
            self.risk_limits[name].limit_value = new_limit
            self.risk_limits[name].updated_at = datetime.now()
            self.logger.info(f"Updated risk limit {name}: {old_limit} -> {new_limit}")
    
    def remove_risk_limit(self, name: str) -> None:
        """移除风险限制"""
        if name in self.risk_limits:
            del self.risk_limits[name]
            self.logger.info(f"Removed risk limit: {name}")
    
    def update_metric(self, metric: RiskMetric) -> None:
        """更新风险指标"""
        # 存储指标
        self.risk_metrics[metric.name] = metric
        self.metric_history[metric.name].append(metric)
        
        # 更新统计
        self.monitoring_stats['total_metrics_processed'] += 1
        self.monitoring_stats['last_update_time'] = datetime.now()
        
        # 检查阈值突破
        self._check_threshold_breach(metric)
        
        # 异常检测
        self._detect_anomaly(metric)
        
        # 调用回调函数
        for callback in self.metric_callbacks[metric.name]:
            try:
                callback(metric)
            except Exception as e:
                self.logger.error(f"Error in metric callback: {e}")
    
    def update_portfolio_metrics(self, portfolio: Portfolio) -> None:
        """更新投资组合风险指标"""
        try:
            # 计算各种风险指标
            metrics = self._calculate_portfolio_metrics(portfolio)
            
            # 更新每个指标
            for metric in metrics:
                self.update_metric(metric)
                
        except Exception as e:
            self.logger.error(f"Error updating portfolio metrics: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        active_alerts_by_level = defaultdict(int)
        for alert in self.active_alerts.values():
            active_alerts_by_level[alert.risk_level.value] += 1
        
        breached_limits = [
            limit for limit in self.risk_limits.values() 
            if limit.is_breached
        ]
        
        warning_limits = [
            limit for limit in self.risk_limits.values() 
            if limit.is_warning and not limit.is_breached
        ]
        
        return {
            'monitoring_status': 'running' if self.is_running else 'stopped',
            'is_paused': self.is_paused,
            'total_metrics': len(self.risk_metrics),
            'total_limits': len(self.risk_limits),
            'active_alerts': len(self.active_alerts),
            'alerts_by_level': dict(active_alerts_by_level),
            'breached_limits': len(breached_limits),
            'warning_limits': len(warning_limits),
            'monitoring_stats': self.monitoring_stats.copy(),
            'last_update': self.monitoring_stats['last_update_time']
        }
    
    def get_active_alerts(self) -> List[RiskAlert]:
        """获取活跃预警"""
        return list(self.active_alerts.values())
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """确认预警"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledge(user)
            self.logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决预警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolve()
            
            # 移动到历史记录
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert {alert_id} resolved")
            return True
        return False
    
    def add_metric_callback(self, metric_name: str, callback: Callable) -> None:
        """添加指标回调"""
        self.metric_callbacks[metric_name].append(callback)
    
    def add_alert_callback(self, callback: Callable) -> None:
        """添加预警回调"""
        self.alert_callbacks.append(callback)
    
    def _monitor_loop(self) -> None:
        """监控主循环"""
        while self.is_running and not self._stop_event.is_set():
            try:
                if not self.is_paused:
                    # 检查风险限制
                    self._check_risk_limits()
                    
                    # 清理过期预警
                    self._cleanup_expired_alerts()
                
                # 等待下一个监控周期
                self._stop_event.wait(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                time.sleep(1)
    
    def _check_threshold_breach(self, metric: RiskMetric) -> None:
        """检查阈值突破"""
        if not metric.is_breach:
            return
        
        # 检查预警冷却时间
        alert_key = f"threshold_{metric.name}"
        if self._is_in_cooldown(alert_key):
            return
        
        # 生成预警
        alert = RiskAlert(
            alert_id=f"alert_{int(datetime.now().timestamp())}_{metric.name}",
            alert_type=AlertType.THRESHOLD_BREACH,
            risk_type=metric.risk_type,
            risk_level=metric.risk_level,
            title=f"Risk Threshold Breach: {metric.name}",
            message=f"Metric {metric.name} breached threshold. Value: {metric.value}, Threshold: {metric.threshold}",
            source="risk_monitor",
            metrics=[metric]
        )
        
        self._generate_alert(alert)
        self.last_alert_time[alert_key] = datetime.now()
    
    def _detect_anomaly(self, metric: RiskMetric) -> None:
        """异常检测"""
        detector = self.anomaly_detectors.get(metric.name)
        if not detector:
            return
        
        is_anomaly = detector.detect(metric.value)
        if is_anomaly:
            # 生成异常预警
            alert = RiskAlert(
                alert_id=f"anomaly_{int(datetime.now().timestamp())}_{metric.name}",
                alert_type=AlertType.ANOMALY_DETECTION,
                risk_type=metric.risk_type,
                risk_level=RiskLevel.MEDIUM,
                title=f"Anomaly Detected: {metric.name}",
                message=f"Anomalous value detected for {metric.name}: {metric.value}",
                source="anomaly_detector",
                metrics=[metric]
            )
            
            self._generate_alert(alert)
            self.monitoring_stats['total_anomalies_detected'] += 1
    
    def _check_risk_limits(self) -> None:
        """检查风险限制"""
        for limit in self.risk_limits.values():
            if not limit.enabled:
                continue
            
            # 检查是否突破限制
            if limit.is_breached:
                self._handle_limit_breach(limit)
            elif limit.is_critical:
                self._handle_critical_limit(limit)
            elif limit.is_warning:
                self._handle_warning_limit(limit)
    
    def _handle_limit_breach(self, limit: RiskLimit) -> None:
        """处理限制突破"""
        alert_key = f"breach_{limit.name}"
        if self._is_in_cooldown(alert_key):
            return
        
        alert = RiskAlert(
            alert_id=f"breach_{int(datetime.now().timestamp())}_{limit.name}",
            alert_type=AlertType.THRESHOLD_BREACH,
            risk_type=limit.risk_type,
            risk_level=RiskLevel.CRITICAL,
            title=f"Risk Limit Breached: {limit.name}",
            message=f"Risk limit {limit.name} breached. Current: {limit.current_value}, Limit: {limit.limit_value}",
            source="risk_monitor"
        )
        
        self._generate_alert(alert)
        self.last_alert_time[alert_key] = datetime.now()
    
    def _handle_critical_limit(self, limit: RiskLimit) -> None:
        """处理严重限制"""
        alert_key = f"critical_{limit.name}"
        if self._is_in_cooldown(alert_key):
            return
        
        alert = RiskAlert(
            alert_id=f"critical_{int(datetime.now().timestamp())}_{limit.name}",
            alert_type=AlertType.THRESHOLD_BREACH,
            risk_type=limit.risk_type,
            risk_level=RiskLevel.HIGH,
            title=f"Critical Risk Level: {limit.name}",
            message=f"Risk limit {limit.name} at critical level. Current: {limit.current_value}, Limit: {limit.limit_value}",
            source="risk_monitor"
        )
        
        self._generate_alert(alert)
        self.last_alert_time[alert_key] = datetime.now()
    
    def _handle_warning_limit(self, limit: RiskLimit) -> None:
        """处理警告限制"""
        alert_key = f"warning_{limit.name}"
        if self._is_in_cooldown(alert_key):
            return
        
        alert = RiskAlert(
            alert_id=f"warning_{int(datetime.now().timestamp())}_{limit.name}",
            alert_type=AlertType.THRESHOLD_BREACH,
            risk_type=limit.risk_type,
            risk_level=RiskLevel.MEDIUM,
            title=f"Risk Warning: {limit.name}",
            message=f"Risk limit {limit.name} at warning level. Current: {limit.current_value}, Limit: {limit.limit_value}",
            source="risk_monitor"
        )
        
        self._generate_alert(alert)
        self.last_alert_time[alert_key] = datetime.now()
    
    def _generate_alert(self, alert: RiskAlert) -> None:
        """生成预警"""
        # 存储预警
        self.active_alerts[alert.alert_id] = alert
        
        # 更新统计
        self.monitoring_stats['total_alerts_generated'] += 1
        
        # 调用预警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        # 发布预警事件
        self.event_bus.publish('risk_alert.generated', {
            'alert': alert.to_dict(),
            'timestamp': datetime.now()
        })
        
        self.logger.warning(f"Generated alert: {alert.title}")
    
    def _is_in_cooldown(self, alert_key: str) -> bool:
        """检查是否在冷却时间内"""
        if alert_key not in self.last_alert_time:
            return False
        
        time_since_last = datetime.now() - self.last_alert_time[alert_key]
        return time_since_last.total_seconds() < self.config.alert_cooldown
    
    def _cleanup_expired_alerts(self) -> None:
        """清理过期预警"""
        # 自动解决已确认超过24小时的预警
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        expired_alerts = []
        for alert_id, alert in self.active_alerts.items():
            if (alert.status == AlertStatus.ACKNOWLEDGED and 
                alert.acknowledged_at and 
                alert.acknowledged_at < cutoff_time):
                expired_alerts.append(alert_id)
        
        for alert_id in expired_alerts:
            self.resolve_alert(alert_id)
    
    def _calculate_portfolio_metrics(self, portfolio: Portfolio) -> List[RiskMetric]:
        """计算投资组合风险指标"""
        metrics = []
        
        try:
            # 总价值
            total_value = portfolio.total_value
            if total_value <= 0:
                return metrics
            
            # 计算集中度风险
            max_position_weight = 0
            sector_weights = defaultdict(float)
            
            for position in portfolio.positions.values():
                if position.quantity != 0:
                    weight = abs(position.market_value) / total_value
                    max_position_weight = max(max_position_weight, weight)
                    
                    # 假设有行业信息
                    sector = getattr(position, 'sector', 'Unknown')
                    sector_weights[sector] += weight
            
            # 最大单一持仓集中度
            metrics.append(RiskMetric(
                name="max_position_concentration",
                value=max_position_weight,
                threshold=self.config.max_position_concentration,
                risk_type=RiskType.CONCENTRATION_RISK,
                risk_level=self._get_risk_level(max_position_weight, self.config.max_position_concentration),
                description="Maximum single position concentration",
                unit="ratio"
            ))
            
            # 最大行业集中度
            max_sector_weight = max(sector_weights.values()) if sector_weights else 0
            metrics.append(RiskMetric(
                name="max_sector_concentration",
                value=max_sector_weight,
                threshold=self.config.max_sector_concentration,
                risk_type=RiskType.CONCENTRATION_RISK,
                risk_level=self._get_risk_level(max_sector_weight, self.config.max_sector_concentration),
                description="Maximum sector concentration",
                unit="ratio"
            ))
            
            # 杠杆率
            leverage = portfolio.leverage if hasattr(portfolio, 'leverage') else 1.0
            metrics.append(RiskMetric(
                name="portfolio_leverage",
                value=leverage,
                threshold=self.config.max_leverage,
                risk_type=RiskType.LEVERAGE_RISK,
                risk_level=self._get_risk_level(leverage, self.config.max_leverage),
                description="Portfolio leverage ratio",
                unit="ratio"
            ))
            
            # 最大回撤
            max_drawdown = portfolio.max_drawdown if hasattr(portfolio, 'max_drawdown') else 0.0
            metrics.append(RiskMetric(
                name="max_drawdown",
                value=abs(max_drawdown),
                threshold=self.config.max_drawdown,
                risk_type=RiskType.DRAWDOWN_RISK,
                risk_level=self._get_risk_level(abs(max_drawdown), self.config.max_drawdown),
                description="Maximum drawdown",
                unit="ratio"
            ))
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
        
        return metrics
    
    def _get_risk_level(self, value: float, threshold: float) -> RiskLevel:
        """根据值和阈值确定风险等级"""
        if value <= threshold * 0.5:
            return RiskLevel.LOW
        elif value <= threshold * 0.8:
            return RiskLevel.MEDIUM
        elif value <= threshold:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _setup_default_limits(self) -> None:
        """设置默认风险限制"""
        default_limits = [
            RiskLimit(
                name="portfolio_var",
                risk_type=RiskType.VAR_RISK,
                limit_value=self.config.max_portfolio_var,
                description="Portfolio Value at Risk",
                unit="ratio"
            ),
            RiskLimit(
                name="position_concentration",
                risk_type=RiskType.CONCENTRATION_RISK,
                limit_value=self.config.max_position_concentration,
                description="Maximum single position concentration",
                unit="ratio"
            ),
            RiskLimit(
                name="sector_concentration",
                risk_type=RiskType.CONCENTRATION_RISK,
                limit_value=self.config.max_sector_concentration,
                description="Maximum sector concentration",
                unit="ratio"
            ),
            RiskLimit(
                name="leverage",
                risk_type=RiskType.LEVERAGE_RISK,
                limit_value=self.config.max_leverage,
                description="Portfolio leverage ratio",
                unit="ratio"
            ),
            RiskLimit(
                name="drawdown",
                risk_type=RiskType.DRAWDOWN_RISK,
                limit_value=self.config.max_drawdown,
                description="Maximum drawdown",
                unit="ratio"
            )
        ]
        
        for limit in default_limits:
            self.add_risk_limit(limit)
    
    def _setup_anomaly_detectors(self) -> None:
        """设置异常检测器"""
        # 为关键指标设置异常检测器
        key_metrics = [
            "portfolio_var", "max_position_concentration", 
            "max_sector_concentration", "portfolio_leverage", "max_drawdown"
        ]
        
        for metric_name in key_metrics:
            self.anomaly_detectors[metric_name] = AnomalyDetector(
                window_size=50,
                threshold=2.0  # 2个标准差
            )


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, window_size: int = 50, threshold: float = 2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.values = deque(maxlen=window_size)
    
    def detect(self, value: float) -> bool:
        """检测异常值"""
        self.values.append(value)
        
        if len(self.values) < 10:  # 需要足够的历史数据
            return False
        
        # 计算统计量
        values_array = np.array(self.values)
        mean = np.mean(values_array[:-1])  # 排除当前值
        std = np.std(values_array[:-1])
        
        if std == 0:
            return False
        
        # 计算Z分数
        z_score = abs(value - mean) / std
        
        return z_score > self.threshold