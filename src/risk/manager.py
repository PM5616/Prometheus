"""Risk Manager Module

风险管理器模块，负责整体风险管理和控制。

主要功能：
- 风险限制设置和检查
- 实时风险监控
- 风险预警和处理
- 风险报告生成
- 紧急风险控制
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import json
import warnings

from ..common.logging import get_logger
from ..common.exceptions.risk import RiskManagerError, RiskLimitExceededError
from ..strategy.base import StrategySignal
from ..strategy.portfolio_manager import Position, PortfolioManager


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"              # 低风险
    MEDIUM = "medium"        # 中等风险
    HIGH = "high"            # 高风险
    CRITICAL = "critical"    # 严重风险
    EMERGENCY = "emergency"  # 紧急风险


class RiskType(Enum):
    """风险类型枚举"""
    MARKET = "market"              # 市场风险
    CREDIT = "credit"              # 信用风险
    LIQUIDITY = "liquidity"        # 流动性风险
    OPERATIONAL = "operational"    # 操作风险
    MODEL = "model"                # 模型风险
    CONCENTRATION = "concentration" # 集中度风险
    LEVERAGE = "leverage"          # 杠杆风险
    DRAWDOWN = "drawdown"          # 回撤风险


class RiskAction(Enum):
    """风险处理动作枚举"""
    MONITOR = "monitor"        # 监控
    WARN = "warn"              # 警告
    LIMIT = "limit"            # 限制
    REDUCE = "reduce"          # 减仓
    CLOSE = "close"            # 平仓
    STOP = "stop"              # 停止交易
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止


@dataclass
class RiskLimit:
    """风险限制"""
    name: str
    risk_type: RiskType
    threshold: float
    action: RiskAction
    enabled: bool = True
    description: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 风险限制字典
        """
        return {
            'name': self.name,
            'risk_type': self.risk_type.value,
            'threshold': self.threshold,
            'action': self.action.value,
            'enabled': self.enabled,
            'description': self.description,
            'metadata': self.metadata
        }


@dataclass
class RiskEvent:
    """风险事件"""
    event_id: str
    risk_type: RiskType
    risk_level: RiskLevel
    current_value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: Optional[str] = None
    strategy_id: Optional[str] = None
    action_taken: Optional[RiskAction] = None
    resolved: bool = False
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 风险事件字典
        """
        return {
            'event_id': self.event_id,
            'risk_type': self.risk_type.value,
            'risk_level': self.risk_level.value,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'strategy_id': self.strategy_id,
            'action_taken': self.action_taken.value if self.action_taken else None,
            'resolved': self.resolved,
            'metadata': self.metadata
        }


@dataclass
class RiskMetrics:
    """风险指标"""
    var_95: float = 0.0          # 95% VaR
    var_99: float = 0.0          # 99% VaR
    cvar_95: float = 0.0         # 95% CVaR
    cvar_99: float = 0.0         # 99% CVaR
    max_drawdown: float = 0.0    # 最大回撤
    volatility: float = 0.0      # 波动率
    beta: float = 0.0            # Beta系数
    correlation: float = 0.0     # 相关性
    leverage: float = 0.0        # 杠杆率
    concentration: float = 0.0   # 集中度
    liquidity_risk: float = 0.0  # 流动性风险
    stress_loss: float = 0.0     # 压力损失
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 风险指标字典
        """
        return {
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'beta': self.beta,
            'correlation': self.correlation,
            'leverage': self.leverage,
            'concentration': self.concentration,
            'liquidity_risk': self.liquidity_risk,
            'stress_loss': self.stress_loss,
            'last_updated': self.last_updated.isoformat()
        }


class RiskManager:
    """风险管理器
    
    负责整体风险管理和控制。
    """
    
    def __init__(self, portfolio_manager: PortfolioManager, config: Dict = None):
        """初始化风险管理器
        
        Args:
            portfolio_manager: 投资组合管理器
            config: 配置参数
        """
        self.portfolio_manager = portfolio_manager
        self.config = config or {}
        
        # 风险限制
        self.risk_limits: Dict[str, RiskLimit] = {}
        
        # 风险事件
        self.risk_events: List[RiskEvent] = []
        self.active_events: Dict[str, RiskEvent] = {}
        
        # 风险指标历史
        self.metrics_history: deque = deque(maxlen=1000)
        
        # 价格历史（用于风险计算）
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))  # 一年数据
        self.return_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        
        # 基准数据
        self.benchmark_returns: deque = deque(maxlen=252)
        
        # 状态管理
        self.enabled = True
        self.emergency_mode = False
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 回调函数
        self.on_risk_event_callbacks: List[Callable] = []
        self.on_limit_exceeded_callbacks: List[Callable] = []
        self.on_emergency_callbacks: List[Callable] = []
        
        # 日志记录
        self.logger = get_logger("RiskManager")
        
        # 统计信息
        self.stats = {
            'total_events': 0,
            'critical_events': 0,
            'emergency_events': 0,
            'limits_exceeded': 0,
            'actions_taken': 0,
            'last_check_time': None
        }
        
        # 初始化默认风险限制
        self._setup_default_limits()
        
        self.logger.info("风险管理器初始化完成")
    
    def _setup_default_limits(self):
        """设置默认风险限制"""
        default_limits = [
            RiskLimit(
                name="max_drawdown",
                risk_type=RiskType.DRAWDOWN,
                threshold=0.20,  # 20%最大回撤
                action=RiskAction.WARN,
                description="最大回撤限制"
            ),
            RiskLimit(
                name="max_leverage",
                risk_type=RiskType.LEVERAGE,
                threshold=3.0,  # 3倍杠杆
                action=RiskAction.LIMIT,
                description="最大杠杆限制"
            ),
            RiskLimit(
                name="max_concentration",
                risk_type=RiskType.CONCENTRATION,
                threshold=0.30,  # 30%集中度
                action=RiskAction.WARN,
                description="最大集中度限制"
            ),
            RiskLimit(
                name="var_95_limit",
                risk_type=RiskType.MARKET,
                threshold=0.05,  # 5% VaR
                action=RiskAction.WARN,
                description="95% VaR限制"
            ),
            RiskLimit(
                name="emergency_drawdown",
                risk_type=RiskType.DRAWDOWN,
                threshold=0.30,  # 30%紧急回撤
                action=RiskAction.EMERGENCY_STOP,
                description="紧急回撤限制"
            )
        ]
        
        for limit in default_limits:
            self.add_risk_limit(limit)
    
    def add_risk_limit(self, risk_limit: RiskLimit):
        """添加风险限制
        
        Args:
            risk_limit: 风险限制
        """
        with self._lock:
            self.risk_limits[risk_limit.name] = risk_limit
            self.logger.info(f"添加风险限制: {risk_limit.name} ({risk_limit.threshold})")
    
    def remove_risk_limit(self, name: str) -> bool:
        """移除风险限制
        
        Args:
            name: 风险限制名称
            
        Returns:
            bool: 是否成功移除
        """
        with self._lock:
            if name in self.risk_limits:
                del self.risk_limits[name]
                self.logger.info(f"移除风险限制: {name}")
                return True
            return False
    
    def enable_risk_limit(self, name: str, enabled: bool = True):
        """启用/禁用风险限制
        
        Args:
            name: 风险限制名称
            enabled: 是否启用
        """
        with self._lock:
            if name in self.risk_limits:
                self.risk_limits[name].enabled = enabled
                status = "启用" if enabled else "禁用"
                self.logger.info(f"{status}风险限制: {name}")
    
    def update_prices(self, prices: Dict[str, float]):
        """更新价格数据
        
        Args:
            prices: 价格字典
        """
        with self._lock:
            for symbol, price in prices.items():
                # 更新价格历史
                if self.price_history[symbol]:
                    prev_price = self.price_history[symbol][-1]
                    if prev_price > 0:
                        return_rate = (price - prev_price) / prev_price
                        self.return_history[symbol].append(return_rate)
                
                self.price_history[symbol].append(price)
    
    def update_benchmark(self, benchmark_price: float):
        """更新基准价格
        
        Args:
            benchmark_price: 基准价格
        """
        with self._lock:
            if self.benchmark_returns:
                prev_price = list(self.benchmark_returns)[-1] if self.benchmark_returns else benchmark_price
                if prev_price > 0:
                    return_rate = (benchmark_price - prev_price) / prev_price
                    self.benchmark_returns.append(return_rate)
            else:
                self.benchmark_returns.append(0.0)  # 第一个数据点
    
    def check_risks(self) -> List[RiskEvent]:
        """检查风险
        
        Returns:
            List[RiskEvent]: 风险事件列表
        """
        if not self.enabled:
            return []
        
        with self._lock:
            events = []
            
            try:
                # 计算当前风险指标
                metrics = self.calculate_risk_metrics()
                
                # 检查各项风险限制
                for limit_name, limit in self.risk_limits.items():
                    if not limit.enabled:
                        continue
                    
                    event = self._check_single_limit(limit, metrics)
                    if event:
                        events.append(event)
                        self._handle_risk_event(event)
                
                # 更新统计
                self.stats['last_check_time'] = datetime.now()
                
                return events
                
            except Exception as e:
                self.logger.error(f"风险检查失败: {e}")
                return []
    
    def _check_single_limit(self, limit: RiskLimit, metrics: RiskMetrics) -> Optional[RiskEvent]:
        """检查单个风险限制
        
        Args:
            limit: 风险限制
            metrics: 风险指标
            
        Returns:
            Optional[RiskEvent]: 风险事件（如果超限）
        """
        current_value = None
        
        # 根据风险类型获取当前值
        if limit.risk_type == RiskType.DRAWDOWN:
            current_value = metrics.max_drawdown
        elif limit.risk_type == RiskType.LEVERAGE:
            current_value = metrics.leverage
        elif limit.risk_type == RiskType.CONCENTRATION:
            current_value = metrics.concentration
        elif limit.risk_type == RiskType.MARKET:
            if "var_95" in limit.name:
                current_value = metrics.var_95
            elif "var_99" in limit.name:
                current_value = metrics.var_99
        
        if current_value is None:
            return None
        
        # 检查是否超限
        if current_value > limit.threshold:
            # 确定风险等级
            risk_level = self._determine_risk_level(current_value, limit.threshold, limit.action)
            
            # 生成事件ID
            event_id = f"{limit.name}_{int(datetime.now().timestamp())}"
            
            # 创建风险事件
            event = RiskEvent(
                event_id=event_id,
                risk_type=limit.risk_type,
                risk_level=risk_level,
                current_value=current_value,
                threshold=limit.threshold,
                message=f"{limit.description}: {current_value:.4f} > {limit.threshold:.4f}",
                action_taken=limit.action
            )
            
            return event
        
        return None
    
    def _determine_risk_level(self, current_value: float, threshold: float, action: RiskAction) -> RiskLevel:
        """确定风险等级
        
        Args:
            current_value: 当前值
            threshold: 阈值
            action: 处理动作
            
        Returns:
            RiskLevel: 风险等级
        """
        excess_ratio = (current_value - threshold) / threshold
        
        if action == RiskAction.EMERGENCY_STOP:
            return RiskLevel.EMERGENCY
        elif excess_ratio > 0.5:  # 超出50%
            return RiskLevel.CRITICAL
        elif excess_ratio > 0.2:  # 超出20%
            return RiskLevel.HIGH
        elif excess_ratio > 0.1:  # 超出10%
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _handle_risk_event(self, event: RiskEvent):
        """处理风险事件
        
        Args:
            event: 风险事件
        """
        # 记录事件
        self.risk_events.append(event)
        self.active_events[event.event_id] = event
        
        # 更新统计
        self.stats['total_events'] += 1
        if event.risk_level == RiskLevel.CRITICAL:
            self.stats['critical_events'] += 1
        elif event.risk_level == RiskLevel.EMERGENCY:
            self.stats['emergency_events'] += 1
        
        # 执行相应动作
        if event.action_taken:
            self._execute_risk_action(event)
        
        # 调用回调函数
        for callback in self.on_risk_event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"风险事件回调函数执行失败: {e}")
        
        self.logger.warning(f"风险事件: {event.message}")
    
    def _execute_risk_action(self, event: RiskEvent):
        """执行风险处理动作
        
        Args:
            event: 风险事件
        """
        try:
            if event.action_taken == RiskAction.WARN:
                self.logger.warning(f"风险警告: {event.message}")
            
            elif event.action_taken == RiskAction.LIMIT:
                self.logger.warning(f"风险限制: {event.message}")
                # 可以在这里实现具体的限制逻辑
            
            elif event.action_taken == RiskAction.REDUCE:
                self._reduce_positions(event)
            
            elif event.action_taken == RiskAction.CLOSE:
                self._close_positions(event)
            
            elif event.action_taken == RiskAction.STOP:
                self._stop_trading(event)
            
            elif event.action_taken == RiskAction.EMERGENCY_STOP:
                self._emergency_stop(event)
            
            self.stats['actions_taken'] += 1
            
        except Exception as e:
            self.logger.error(f"执行风险动作失败: {e}")
    
    def _reduce_positions(self, event: RiskEvent):
        """减仓
        
        Args:
            event: 风险事件
        """
        # 减少所有持仓的50%
        positions = self.portfolio_manager.get_all_positions()
        
        for symbol, position in positions.items():
            if position.quantity != 0:
                reduce_quantity = position.quantity * 0.5
                self.portfolio_manager.place_order(
                    symbol=symbol,
                    quantity=-reduce_quantity,
                    strategy_id="risk_manager"
                )
        
        self.logger.info(f"风险减仓执行完成: {event.event_id}")
    
    def _close_positions(self, event: RiskEvent):
        """平仓
        
        Args:
            event: 风险事件
        """
        positions = self.portfolio_manager.get_all_positions()
        
        for symbol, position in positions.items():
            if position.quantity != 0:
                self.portfolio_manager.place_order(
                    symbol=symbol,
                    quantity=-position.quantity,
                    strategy_id="risk_manager"
                )
        
        self.logger.warning(f"风险平仓执行完成: {event.event_id}")
    
    def _stop_trading(self, event: RiskEvent):
        """停止交易
        
        Args:
            event: 风险事件
        """
        self.enabled = False
        self.logger.warning(f"交易已停止: {event.event_id}")
    
    def _emergency_stop(self, event: RiskEvent):
        """紧急停止
        
        Args:
            event: 风险事件
        """
        self.emergency_mode = True
        self.enabled = False
        
        # 取消所有挂单
        orders = self.portfolio_manager.orders
        for order_id in list(orders.keys()):
            self.portfolio_manager.cancel_order(order_id)
        
        # 平掉所有持仓
        self._close_positions(event)
        
        # 调用紧急回调
        for callback in self.on_emergency_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"紧急回调函数执行失败: {e}")
        
        self.logger.critical(f"紧急停止执行完成: {event.event_id}")
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """计算风险指标
        
        Returns:
            RiskMetrics: 风险指标
        """
        metrics = RiskMetrics()
        
        try:
            # 获取投资组合数据
            portfolio_value = self.portfolio_manager.get_portfolio_value()
            positions = self.portfolio_manager.get_all_positions()
            
            # 计算杠杆率
            total_exposure = sum(abs(pos.market_value) for pos in positions.values())
            metrics.leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # 计算集中度（最大单一持仓占比）
            if portfolio_value > 0:
                max_position_value = max(
                    (abs(pos.market_value) for pos in positions.values()),
                    default=0
                )
                metrics.concentration = max_position_value / portfolio_value
            
            # 计算投资组合收益率序列
            portfolio_returns = self._calculate_portfolio_returns()
            
            if len(portfolio_returns) > 10:  # 需要足够的数据
                # 计算VaR
                metrics.var_95 = np.percentile(portfolio_returns, 5) * -1
                metrics.var_99 = np.percentile(portfolio_returns, 1) * -1
                
                # 计算CVaR
                var_95_threshold = np.percentile(portfolio_returns, 5)
                var_99_threshold = np.percentile(portfolio_returns, 1)
                
                tail_95 = [r for r in portfolio_returns if r <= var_95_threshold]
                tail_99 = [r for r in portfolio_returns if r <= var_99_threshold]
                
                if tail_95:
                    metrics.cvar_95 = np.mean(tail_95) * -1
                if tail_99:
                    metrics.cvar_99 = np.mean(tail_99) * -1
                
                # 计算波动率
                metrics.volatility = np.std(portfolio_returns) * np.sqrt(252)  # 年化
                
                # 计算Beta（如果有基准数据）
                if len(self.benchmark_returns) > 10:
                    benchmark_array = np.array(list(self.benchmark_returns))
                    portfolio_array = np.array(portfolio_returns[-len(benchmark_array):])
                    
                    if len(benchmark_array) == len(portfolio_array):
                        covariance = np.cov(portfolio_array, benchmark_array)[0, 1]
                        benchmark_variance = np.var(benchmark_array)
                        
                        if benchmark_variance > 0:
                            metrics.beta = covariance / benchmark_variance
                
                # 计算最大回撤
                metrics.max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            # 记录历史
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算风险指标失败: {e}")
            return metrics
    
    def _calculate_portfolio_returns(self) -> List[float]:
        """计算投资组合收益率序列
        
        Returns:
            List[float]: 收益率序列
        """
        # 这里简化处理，实际应该根据历史净值计算
        # 可以从portfolio_manager的历史数据中获取
        returns = []
        
        # 获取投资组合历史价值
        portfolio_history = self.portfolio_manager.portfolio_history
        
        if len(portfolio_history) > 1:
            for i in range(1, len(portfolio_history)):
                prev_value = portfolio_history[i-1].get('metrics', {}).get('total_value', 0)
                curr_value = portfolio_history[i].get('metrics', {}).get('total_value', 0)
                
                if prev_value > 0:
                    return_rate = (curr_value - prev_value) / prev_value
                    returns.append(return_rate)
        
        return returns
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """计算最大回撤
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 最大回撤
        """
        if not returns:
            return 0.0
        
        # 计算累计收益
        cumulative = np.cumprod(1 + np.array(returns))
        
        # 计算回撤
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        
        return abs(np.min(drawdown))
    
    def check_signal_risk(self, signal: StrategySignal) -> bool:
        """检查信号风险
        
        Args:
            signal: 策略信号
            
        Returns:
            bool: 信号是否通过风险检查
        """
        if not self.enabled or self.emergency_mode:
            return False
        
        try:
            # 模拟执行信号后的风险状况
            # 这里可以实现更复杂的风险预测逻辑
            
            # 检查持仓集中度
            if signal.symbol:
                current_position = self.portfolio_manager.get_position(signal.symbol)
                portfolio_value = self.portfolio_manager.get_portfolio_value()
                
                if current_position and portfolio_value > 0:
                    position_ratio = abs(current_position.market_value) / portfolio_value
                    
                    # 检查是否会导致过度集中
                    concentration_limit = self.risk_limits.get('max_concentration')
                    if concentration_limit and concentration_limit.enabled:
                        if position_ratio > concentration_limit.threshold:
                            self.logger.warning(f"信号被风险控制拒绝: 集中度过高 {signal.symbol}")
                            return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"信号风险检查失败: {e}")
            return False
    
    def resolve_event(self, event_id: str):
        """解决风险事件
        
        Args:
            event_id: 事件ID
        """
        with self._lock:
            if event_id in self.active_events:
                event = self.active_events[event_id]
                event.resolved = True
                del self.active_events[event_id]
                
                self.logger.info(f"风险事件已解决: {event_id}")
    
    def get_active_events(self) -> List[RiskEvent]:
        """获取活跃的风险事件
        
        Returns:
            List[RiskEvent]: 活跃事件列表
        """
        return list(self.active_events.values())
    
    def get_risk_summary(self) -> Dict:
        """获取风险摘要
        
        Returns:
            Dict: 风险摘要
        """
        metrics = self.calculate_risk_metrics()
        active_events = self.get_active_events()
        
        return {
            'enabled': self.enabled,
            'emergency_mode': self.emergency_mode,
            'metrics': metrics.to_dict(),
            'active_events': len(active_events),
            'critical_events': len([e for e in active_events if e.risk_level == RiskLevel.CRITICAL]),
            'stats': self.stats.copy(),
            'limits': {name: limit.to_dict() for name, limit in self.risk_limits.items()}
        }
    
    def add_risk_event_callback(self, callback: Callable):
        """添加风险事件回调
        
        Args:
            callback: 回调函数
        """
        self.on_risk_event_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable):
        """添加紧急回调
        
        Args:
            callback: 回调函数
        """
        self.on_emergency_callbacks.append(callback)
    
    def enable(self, enabled: bool = True):
        """启用/禁用风险管理
        
        Args:
            enabled: 是否启用
        """
        self.enabled = enabled
        status = "启用" if enabled else "禁用"
        self.logger.info(f"风险管理已{status}")
    
    def reset_emergency_mode(self):
        """重置紧急模式"""
        with self._lock:
            self.emergency_mode = False
            self.enabled = True
            self.active_events.clear()
            
            self.logger.info("紧急模式已重置")
    
    def export_events(self, start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> List[Dict]:
        """导出风险事件
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[Dict]: 事件列表
        """
        filtered_events = self.risk_events
        
        if start_date:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_date]
        
        if end_date:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_date]
        
        return [event.to_dict() for event in filtered_events]
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 风险管理器字典
        """
        return {
            'enabled': self.enabled,
            'emergency_mode': self.emergency_mode,
            'risk_limits': {name: limit.to_dict() for name, limit in self.risk_limits.items()},
            'active_events': {eid: event.to_dict() for eid, event in self.active_events.items()},
            'metrics': self.calculate_risk_metrics().to_dict(),
            'stats': self.stats.copy()
        }
    
    def __str__(self) -> str:
        active_events = len(self.active_events)
        status = "启用" if self.enabled else "禁用"
        return f"RiskManager(status={status}, active_events={active_events})"
    
    def __repr__(self) -> str:
        return self.__str__()