"""Risk Manager

风险管理器 - 整合风险监控、控制和报告功能
整合了原risk模块的RiskManager功能
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import threading
import time
import json

from .models import (
    RiskType, RiskLevel, AlertType, AlertStatus, RiskMetric, 
    RiskAlert, RiskLimit, RiskEvent, RiskConfig, ControlAction
)
from .monitor import RiskMonitor
from .reporter import RiskReporter
# 延迟导入controller以避免循环导入
from src.common.logging import get_logger
from src.common.events import EventBus
from src.portfolio_manager.models import Portfolio, Position


class RiskManager:
    """风险管理器
    
    整合风险监控、控制和报告功能，提供统一的风险管理接口
    整合了原risk模块的RiskManager功能
    """
    
    def __init__(self, config: RiskConfig, event_bus: EventBus, portfolio_manager=None):
        self.config = config
        self.event_bus = event_bus
        self.portfolio_manager = portfolio_manager
        self.logger = get_logger(self.__class__.__name__)
        
        # 初始化子模块
        self.monitor = RiskMonitor(config, event_bus)
        self.controller = None  # 延迟初始化
        
        # 创建报告配置并初始化reporter
        from .reporter import ReportConfig
        report_config = ReportConfig()
        self.reporter = RiskReporter(report_config)
        
        # 管理状态
        self.is_running = False
        self.is_emergency_mode = False
        self.enabled = True
        
        # 风险限制管理（整合原risk模块功能）
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.default_limits = self._create_default_limits()
        
        # 风险事件管理
        self.risk_events: Dict[str, RiskEvent] = {}
        self.event_callbacks: List[Callable] = []
        self.emergency_callbacks: List[Callable] = []
        
        # 价格和基准数据
        self.current_prices: Dict[str, float] = {}
        self.benchmark_prices: Dict[str, float] = {}
        
        # 风险指标历史
        self.risk_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 风险统计
        self.risk_stats = {
            'total_events': 0,
            'resolved_events': 0,
            'active_events': 0,
            'emergency_stops': 0,
            'last_check_time': None
        }
        
        # 设置事件监听
        self._setup_event_listeners()
        
        # 初始化默认风险限制
        self._setup_default_limits()
    
    def start(self) -> None:
        """启动风险管理"""
        if self.is_running:
            self.logger.warning("Risk manager is already running")
            return
        
        # 延迟导入并初始化controller
        if self.controller is None:
            from .controller import RiskController
            self.controller = RiskController(self.config, self.event_bus)
        
        self.logger.info("Starting risk manager")
        self.is_running = True
        
        # 启动子模块
        self.monitor.start()
        self.controller.start()
        
        # 发布启动事件
        self.event_bus.publish('risk_manager.started', {'timestamp': datetime.now()})
    
    def stop(self) -> None:
        """停止风险管理"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping risk manager")
        self.is_running = False
        
        # 停止子模块
        self.monitor.stop()
        if self.controller:
            self.controller.stop()
        
        # 发布停止事件
        self.event_bus.publish('risk_manager.stopped', {'timestamp': datetime.now()})
    
    def enable(self) -> None:
        """启用风险管理"""
        self.enabled = True
        self.logger.info("Risk management enabled")
    
    def disable(self) -> None:
        """禁用风险管理"""
        self.enabled = False
        self.logger.warning("Risk management disabled")
    
    # ========== 风险限制管理 ==========
    
    def add_risk_limit(self, limit: RiskLimit) -> None:
        """添加风险限制"""
        self.risk_limits[limit.name] = limit
        self.monitor.add_risk_limit(limit)
        self.logger.info(f"Added risk limit: {limit.name} = {limit.limit_value}")
    
    def remove_risk_limit(self, name: str) -> None:
        """移除风险限制"""
        if name in self.risk_limits:
            del self.risk_limits[name]
            self.monitor.remove_risk_limit(name)
            self.logger.info(f"Removed risk limit: {name}")
    
    def enable_risk_limit(self, name: str) -> None:
        """启用风险限制"""
        if name in self.risk_limits:
            self.risk_limits[name].enabled = True
            self.logger.info(f"Enabled risk limit: {name}")
    
    def disable_risk_limit(self, name: str) -> None:
        """禁用风险限制"""
        if name in self.risk_limits:
            self.risk_limits[name].enabled = False
            self.logger.info(f"Disabled risk limit: {name}")
    
    def update_risk_limit(self, name: str, new_limit: float) -> None:
        """更新风险限制"""
        if name in self.risk_limits:
            old_limit = self.risk_limits[name].limit_value
            self.risk_limits[name].limit_value = new_limit
            self.risk_limits[name].updated_at = datetime.now()
            self.monitor.update_risk_limit(name, new_limit)
            self.logger.info(f"Updated risk limit {name}: {old_limit} -> {new_limit}")
    
    # ========== 价格和基准更新 ==========
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """更新当前价格"""
        self.current_prices.update(prices)
        
        # 触发风险检查
        if self.enabled and self.portfolio_manager:
            self._check_portfolio_risk()
    
    def update_benchmark(self, benchmark_prices: Dict[str, float]) -> None:
        """更新基准价格"""
        self.benchmark_prices.update(benchmark_prices)
    
    # ========== 风险检查 ==========
    
    def check_signal_risk(self, signal: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """检查信号风险"""
        if not self.enabled:
            return True, []
        
        violations = []
        
        try:
            symbol = signal.get('symbol')
            action = signal.get('action')
            quantity = signal.get('quantity', 0)
            
            if not symbol or not action:
                return True, []
            
            # 检查持仓集中度
            if self.portfolio_manager:
                portfolio = self.portfolio_manager.get_portfolio()
                if self._check_concentration_risk(portfolio, symbol, quantity):
                    violations.append(f"Concentration risk: {symbol}")
            
            # 检查杠杆风险
            if self._check_leverage_risk(signal):
                violations.append(f"Leverage risk: {symbol}")
            
            # 检查流动性风险
            if self._check_liquidity_risk(symbol, quantity):
                violations.append(f"Liquidity risk: {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error checking signal risk: {e}")
            return False, ["Risk check error"]
        
        return len(violations) == 0, violations
    
    def _check_portfolio_risk(self) -> None:
        """检查投资组合风险"""
        if not self.portfolio_manager:
            return
        
        try:
            portfolio = self.portfolio_manager.get_portfolio()
            
            # 更新投资组合风险指标
            self.monitor.update_portfolio_metrics(portfolio)
            
            # 计算风险指标
            metrics = self._calculate_risk_metrics(portfolio)
            
            # 检查每个指标
            for metric in metrics:
                self._process_risk_metric(metric)
                
        except Exception as e:
            self.logger.error(f"Error checking portfolio risk: {e}")
    
    def _calculate_risk_metrics(self, portfolio: Portfolio) -> List[RiskMetric]:
        """计算风险指标"""
        metrics = []
        
        try:
            # 计算杠杆率
            leverage = self._calculate_leverage(portfolio)
            metrics.append(RiskMetric(
                name="portfolio_leverage",
                value=leverage,
                threshold=self.config.max_leverage,
                risk_type=RiskType.LEVERAGE_RISK,
                risk_level=self._get_risk_level(leverage, self.config.max_leverage),
                description="Portfolio leverage ratio"
            ))
            
            # 计算集中度
            max_concentration = self._calculate_concentration(portfolio)
            metrics.append(RiskMetric(
                name="max_position_concentration",
                value=max_concentration,
                threshold=self.config.max_position_concentration,
                risk_type=RiskType.CONCENTRATION_RISK,
                risk_level=self._get_risk_level(max_concentration, self.config.max_position_concentration),
                description="Maximum position concentration"
            ))
            
            # 计算VaR
            var = self._calculate_var(portfolio)
            metrics.append(RiskMetric(
                name="portfolio_var",
                value=var,
                threshold=self.config.max_portfolio_var,
                risk_type=RiskType.VAR_RISK,
                risk_level=self._get_risk_level(var, self.config.max_portfolio_var),
                description="Portfolio Value at Risk"
            ))
            
            # 计算最大回撤
            max_drawdown = self._calculate_max_drawdown(portfolio)
            metrics.append(RiskMetric(
                name="max_drawdown",
                value=abs(max_drawdown),
                threshold=self.config.max_drawdown,
                risk_type=RiskType.DRAWDOWN_RISK,
                risk_level=self._get_risk_level(abs(max_drawdown), self.config.max_drawdown),
                description="Maximum drawdown"
            ))
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
        
        return metrics
    
    def _process_risk_metric(self, metric: RiskMetric) -> None:
        """处理风险指标"""
        # 存储历史
        self.risk_metrics_history[metric.name].append(metric)
        
        # 更新监控器
        self.monitor.update_metric(metric)
        
        # 检查是否需要生成风险事件
        if metric.is_breach:
            self._generate_risk_event(metric)
    
    def _generate_risk_event(self, metric: RiskMetric) -> None:
        """生成风险事件"""
        event_id = f"risk_{int(datetime.now().timestamp())}_{metric.name}"
        
        event = RiskEvent(
            event_id=event_id,
            event_type="metric_breach",
            risk_type=metric.risk_type,
            risk_level=metric.risk_level,
            title=f"Risk Metric Breach: {metric.name}",
            description=f"Metric {metric.name} breached threshold. Value: {metric.value}, Threshold: {metric.threshold}",
            source="risk_manager"
        )
        
        self.risk_events[event_id] = event
        self.risk_stats['total_events'] += 1
        self.risk_stats['active_events'] += 1
        
        # 确定处理动作
        action = self._determine_risk_action(metric)
        
        # 执行风险控制动作
        self._execute_risk_action(action, event)
        
        # 调用事件回调
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in event callback: {e}")
        
        self.logger.warning(f"Generated risk event: {event.title}")
    
    def _determine_risk_action(self, metric: RiskMetric) -> ControlAction:
        """确定风险处理动作"""
        breach_ratio = abs(metric.breach_ratio)
        
        if breach_ratio >= 2.0:  # 超限100%以上
            return ControlAction.EMERGENCY_STOP
        elif breach_ratio >= 1.5:  # 超限50%以上
            return ControlAction.CLOSE_POSITION
        elif breach_ratio >= 1.2:  # 超限20%以上
            return ControlAction.REDUCE_POSITION
        elif breach_ratio >= 1.1:  # 超限10%以上
            return ControlAction.WARNING
        else:
            return ControlAction.NONE
    
    def _execute_risk_action(self, action: ControlAction, event: RiskEvent) -> None:
        """执行风险控制动作"""
        try:
            if action == ControlAction.WARNING:
                self.logger.warning(f"Risk warning: {event.title}")
            
            elif action == ControlAction.REDUCE_POSITION:
                self._reduce_positions(event)
            
            elif action == ControlAction.CLOSE_POSITION:
                self._close_positions(event)
            
            elif action == ControlAction.EMERGENCY_STOP:
                self._emergency_stop(event)
            
        except Exception as e:
            self.logger.error(f"Error executing risk action {action}: {e}")
    
    def _reduce_positions(self, event: RiskEvent) -> None:
        """减仓"""
        if not self.portfolio_manager:
            return
        
        try:
            # 减少50%的风险敞口
            reduction_ratio = 0.5
            self.portfolio_manager.reduce_risk_exposure(reduction_ratio)
            self.logger.warning(f"Reduced positions by {reduction_ratio*100}% due to: {event.title}")
        except Exception as e:
            self.logger.error(f"Error reducing positions: {e}")
    
    def _close_positions(self, event: RiskEvent) -> None:
        """平仓"""
        if not self.portfolio_manager:
            return
        
        try:
            # 平掉所有风险头寸
            self.portfolio_manager.close_risk_positions()
            self.logger.warning(f"Closed risk positions due to: {event.title}")
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
    
    def _emergency_stop(self, event: RiskEvent) -> None:
        """紧急停止"""
        try:
            self.is_emergency_mode = True
            self.risk_stats['emergency_stops'] += 1
            
            # 停止所有交易
            if self.portfolio_manager:
                self.portfolio_manager.emergency_stop()
            
            # 调用紧急回调
            for callback in self.emergency_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in emergency callback: {e}")
            
            self.logger.critical(f"EMERGENCY STOP triggered by: {event.title}")
            
        except Exception as e:
            self.logger.error(f"Error in emergency stop: {e}")
    
    # ========== 风险计算方法 ==========
    
    def _calculate_leverage(self, portfolio: Portfolio) -> float:
        """计算杠杆率"""
        try:
            if hasattr(portfolio, 'leverage'):
                return portfolio.leverage
            
            total_exposure = sum(abs(pos.market_value) for pos in portfolio.positions.values())
            if portfolio.total_value <= 0:
                return 0.0
            
            return total_exposure / portfolio.total_value
        except Exception:
            return 0.0
    
    def _calculate_concentration(self, portfolio: Portfolio) -> float:
        """计算最大持仓集中度"""
        try:
            if portfolio.total_value <= 0:
                return 0.0
            
            max_weight = 0.0
            for position in portfolio.positions.values():
                if position.quantity != 0:
                    weight = abs(position.market_value) / portfolio.total_value
                    max_weight = max(max_weight, weight)
            
            return max_weight
        except Exception:
            return 0.0
    
    def _calculate_var(self, portfolio: Portfolio) -> float:
        """计算VaR"""
        try:
            # 简化的VaR计算
            if hasattr(portfolio, 'var'):
                return portfolio.var
            
            # 使用历史波动率估算
            returns = self._get_portfolio_returns(portfolio)
            if len(returns) < 30:
                return 0.0
            
            confidence_level = self.config.var_confidence_level
            var_percentile = (1 - confidence_level) * 100
            
            return abs(np.percentile(returns, var_percentile))
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown(self, portfolio: Portfolio) -> float:
        """计算最大回撤"""
        try:
            if hasattr(portfolio, 'max_drawdown'):
                return portfolio.max_drawdown
            
            # 使用历史净值计算
            nav_history = self._get_nav_history(portfolio)
            if len(nav_history) < 2:
                return 0.0
            
            peak = nav_history[0]
            max_dd = 0.0
            
            for nav in nav_history:
                if nav > peak:
                    peak = nav
                else:
                    drawdown = (peak - nav) / peak
                    max_dd = max(max_dd, drawdown)
            
            return max_dd
        except Exception:
            return 0.0
    
    def _get_portfolio_returns(self, portfolio: Portfolio) -> List[float]:
        """获取投资组合收益率序列"""
        # 这里应该从历史数据中获取，简化实现
        return []
    
    def _get_nav_history(self, portfolio: Portfolio) -> List[float]:
        """获取净值历史"""
        # 这里应该从历史数据中获取，简化实现
        return [portfolio.total_value] if portfolio.total_value > 0 else []
    
    def _get_risk_level(self, value: float, threshold: float) -> RiskLevel:
        """根据值和阈值确定风险等级"""
        if threshold <= 0:
            return RiskLevel.LOW
        
        ratio = abs(value) / abs(threshold)
        
        if ratio <= 0.5:
            return RiskLevel.LOW
        elif ratio <= 0.8:
            return RiskLevel.MEDIUM
        elif ratio <= 1.0:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    # ========== 风险检查辅助方法 ==========
    
    def _check_concentration_risk(self, portfolio: Portfolio, symbol: str, quantity: float) -> bool:
        """检查集中度风险"""
        try:
            if portfolio.total_value <= 0:
                return False
            
            # 计算新的持仓权重
            current_position = portfolio.positions.get(symbol)
            current_quantity = current_position.quantity if current_position else 0
            new_quantity = current_quantity + quantity
            
            # 估算新的市值权重
            price = self.current_prices.get(symbol, 0)
            if price <= 0:
                return False
            
            new_market_value = abs(new_quantity * price)
            new_weight = new_market_value / portfolio.total_value
            
            return new_weight > self.config.max_position_concentration
        except Exception:
            return False
    
    def _check_leverage_risk(self, signal: Dict[str, Any]) -> bool:
        """检查杠杆风险"""
        # 简化实现
        return False
    
    def _check_liquidity_risk(self, symbol: str, quantity: float) -> bool:
        """检查流动性风险"""
        # 简化实现
        return False
    
    # ========== 事件管理 ==========
    
    def resolve_risk_event(self, event_id: str) -> bool:
        """解决风险事件"""
        if event_id in self.risk_events:
            self.risk_events[event_id].status = "resolved"
            self.risk_stats['resolved_events'] += 1
            self.risk_stats['active_events'] -= 1
            self.logger.info(f"Resolved risk event: {event_id}")
            return True
        return False
    
    def get_active_events(self) -> List[RiskEvent]:
        """获取活跃风险事件"""
        return [event for event in self.risk_events.values() if event.status == "open"]
    
    def add_event_callback(self, callback: Callable) -> None:
        """添加事件回调"""
        self.event_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable) -> None:
        """添加紧急回调"""
        self.emergency_callbacks.append(callback)
    
    def reset_emergency_mode(self) -> None:
        """重置紧急模式"""
        self.is_emergency_mode = False
        self.logger.info("Emergency mode reset")
    
    # ========== 风险摘要和报告 ==========
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        monitor_summary = self.monitor.get_risk_summary()
        
        return {
            **monitor_summary,
            'risk_manager_status': {
                'is_running': self.is_running,
                'enabled': self.enabled,
                'is_emergency_mode': self.is_emergency_mode,
                'total_limits': len(self.risk_limits),
                'enabled_limits': sum(1 for limit in self.risk_limits.values() if limit.enabled)
            },
            'risk_events': {
                'total': self.risk_stats['total_events'],
                'active': self.risk_stats['active_events'],
                'resolved': self.risk_stats['resolved_events']
            },
            'emergency_stats': {
                'total_stops': self.risk_stats['emergency_stops'],
                'is_emergency_mode': self.is_emergency_mode
            }
        }
    
    def export_risk_events(self, start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """导出风险事件"""
        events = []
        
        for event in self.risk_events.values():
            if start_date and event.timestamp < start_date:
                continue
            if end_date and event.timestamp > end_date:
                continue
            
            events.append(event.to_dict())
        
        return events
    
    # ========== 初始化方法 ==========
    
    def _create_default_limits(self) -> Dict[str, RiskLimit]:
        """创建默认风险限制"""
        return {
            'portfolio_var': RiskLimit(
                name='portfolio_var',
                risk_type=RiskType.VAR_RISK,
                limit_value=self.config.max_portfolio_var,
                description='Portfolio Value at Risk limit'
            ),
            'max_leverage': RiskLimit(
                name='max_leverage',
                risk_type=RiskType.LEVERAGE_RISK,
                limit_value=self.config.max_leverage,
                description='Maximum leverage ratio'
            ),
            'max_concentration': RiskLimit(
                name='max_concentration',
                risk_type=RiskType.CONCENTRATION_RISK,
                limit_value=self.config.max_position_concentration,
                description='Maximum position concentration'
            ),
            'max_drawdown': RiskLimit(
                name='max_drawdown',
                risk_type=RiskType.DRAWDOWN_RISK,
                limit_value=self.config.max_drawdown,
                description='Maximum drawdown limit'
            )
        }
    
    def _setup_default_limits(self) -> None:
        """设置默认风险限制"""
        for limit in self.default_limits.values():
            self.add_risk_limit(limit)
    
    def _setup_event_listeners(self) -> None:
        """设置事件监听器"""
        # 监听价格更新事件
        self.event_bus.subscribe('market.price_update', self._on_price_update)
        
        # 监听投资组合更新事件
        self.event_bus.subscribe('portfolio.updated', self._on_portfolio_update)
        
        # 监听风险预警事件
        self.event_bus.subscribe('risk_alert.generated', self._on_risk_alert)
    
    def _on_price_update(self, event_data: Dict[str, Any]) -> None:
        """处理价格更新事件"""
        try:
            prices = event_data.get('prices', {})
            self.update_prices(prices)
        except Exception as e:
            self.logger.error(f"Error handling price update: {e}")
    
    def _on_portfolio_update(self, event_data: Dict[str, Any]) -> None:
        """处理投资组合更新事件"""
        try:
            if self.enabled:
                self._check_portfolio_risk()
        except Exception as e:
            self.logger.error(f"Error handling portfolio update: {e}")
    
    def _on_risk_alert(self, event_data: Dict[str, Any]) -> None:
        """处理风险预警事件"""
        try:
            alert = event_data.get('alert', {})
            self.logger.warning(f"Risk alert received: {alert.get('title', 'Unknown')}")
        except Exception as e:
            self.logger.error(f"Error handling risk alert: {e}")
    
    def __str__(self) -> str:
        return f"RiskManager(running={self.is_running}, enabled={self.enabled}, emergency={self.is_emergency_mode})"
    
    def __repr__(self) -> str:
        return self.__str__()