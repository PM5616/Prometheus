"""Risk Controller

风险控制器 - 负责执行风险控制动作
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass
from enum import Enum

from .models import (
    RiskType, RiskLevel, ControlAction, RiskAlert, RiskLimit, 
    RiskEvent, RiskConfig
)
from ..common.logger import get_logger
from ..common.events import EventBus
from ..portfolio_manager.models import Portfolio, Position
from ..execution.models import ExecutionOrder, OrderSide, OrderType


class ControlStatus(Enum):
    """控制状态"""
    PENDING = "pending"  # 待执行
    EXECUTING = "executing"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


@dataclass
class ControlCommand:
    """控制命令"""
    command_id: str
    action: ControlAction
    target_symbol: Optional[str] = None
    target_strategy: Optional[str] = None
    parameters: Dict[str, Any] = None
    priority: int = 0  # 优先级，数字越大优先级越高
    created_at: datetime = None
    executed_at: Optional[datetime] = None
    status: ControlStatus = ControlStatus.PENDING
    result: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.parameters is None:
            self.parameters = {}


class RiskController:
    """风险控制器
    
    负责:
    1. 执行风险控制动作
    2. 管理控制命令队列
    3. 与执行引擎交互
    4. 记录控制历史
    5. 紧急停止机制
    """
    
    def __init__(self, config: RiskConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.logger = get_logger(self.__class__.__name__)
        
        # 控制状态
        self.is_enabled = True
        self.emergency_mode = False
        
        # 命令队列
        self.command_queue: List[ControlCommand] = []
        self.executing_commands: Dict[str, ControlCommand] = {}
        self.command_history: List[ControlCommand] = []
        
        # 控制限制
        self.max_daily_actions: Dict[ControlAction, int] = {
            ControlAction.REDUCE_POSITION: 50,
            ControlAction.CLOSE_POSITION: 20,
            ControlAction.STOP_TRADING: 5,
            ControlAction.EMERGENCY_STOP: 3
        }
        self.daily_action_count: Dict[ControlAction, int] = {}
        self.last_reset_date = datetime.now().date()
        
        # 回调函数
        self.action_callbacks: Dict[ControlAction, List[Callable]] = {}
        
        # 外部服务接口
        self.execution_engine = None
        self.portfolio_manager = None
        
        # 统计信息
        self.control_stats = {
            'total_commands_executed': 0,
            'successful_commands': 0,
            'failed_commands': 0,
            'emergency_stops': 0,
            'last_action_time': None
        }
        
        self._setup_event_handlers()
    
    def set_execution_engine(self, execution_engine) -> None:
        """设置执行引擎"""
        self.execution_engine = execution_engine
    
    def set_portfolio_manager(self, portfolio_manager) -> None:
        """设置投资组合管理器"""
        self.portfolio_manager = portfolio_manager
    
    def enable(self) -> None:
        """启用风险控制"""
        self.is_enabled = True
        self.logger.info("Risk controller enabled")
    
    def disable(self) -> None:
        """禁用风险控制"""
        self.is_enabled = False
        self.logger.warning("Risk controller disabled")
    
    def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """紧急停止"""
        self.logger.critical(f"Emergency stop triggered: {reason}")
        
        self.emergency_mode = True
        
        # 创建紧急停止命令
        command = ControlCommand(
            command_id=f"emergency_{int(datetime.now().timestamp())}",
            action=ControlAction.EMERGENCY_STOP,
            priority=1000,  # 最高优先级
            parameters={'reason': reason}
        )
        
        # 立即执行
        self._execute_command_immediately(command)
        
        # 更新统计
        self.control_stats['emergency_stops'] += 1
        
        # 发布事件
        self.event_bus.publish('risk_control.emergency_stop', {
            'reason': reason,
            'timestamp': datetime.now()
        })
    
    def exit_emergency_mode(self) -> None:
        """退出紧急模式"""
        self.emergency_mode = False
        self.logger.info("Exited emergency mode")
    
    def execute_control_action(self, action: ControlAction, **kwargs) -> str:
        """执行控制动作"""
        if not self.is_enabled:
            self.logger.warning("Risk controller is disabled, ignoring control action")
            return None
        
        # 检查日常限制
        if not self._check_daily_limits(action):
            self.logger.warning(f"Daily limit exceeded for action: {action}")
            return None
        
        # 创建控制命令
        command = ControlCommand(
            command_id=f"cmd_{int(datetime.now().timestamp())}_{action.value}",
            action=action,
            target_symbol=kwargs.get('symbol'),
            target_strategy=kwargs.get('strategy'),
            parameters=kwargs,
            priority=self._get_action_priority(action)
        )
        
        # 添加到队列
        self._add_to_queue(command)
        
        self.logger.info(f"Queued control action: {action} for {command.target_symbol or 'all'}")
        return command.command_id
    
    def handle_risk_alert(self, alert: RiskAlert) -> None:
        """处理风险预警"""
        if not self.config.enable_auto_risk_control:
            return
        
        # 根据预警级别和类型确定控制动作
        actions = self._determine_control_actions(alert)
        
        for action in actions:
            # 执行控制动作
            self.execute_control_action(
                action,
                alert_id=alert.alert_id,
                risk_type=alert.risk_type.value,
                risk_level=alert.risk_level.value,
                symbols=alert.affected_symbols
            )
    
    def process_command_queue(self) -> None:
        """处理命令队列"""
        if not self.command_queue:
            return
        
        # 按优先级排序
        self.command_queue.sort(key=lambda x: x.priority, reverse=True)
        
        # 处理命令
        while self.command_queue:
            command = self.command_queue.pop(0)
            
            try:
                self._execute_command(command)
            except Exception as e:
                self.logger.error(f"Error executing command {command.command_id}: {e}")
                command.status = ControlStatus.FAILED
                command.error_message = str(e)
                self._complete_command(command)
    
    def cancel_command(self, command_id: str) -> bool:
        """取消命令"""
        # 从队列中移除
        for i, command in enumerate(self.command_queue):
            if command.command_id == command_id:
                command.status = ControlStatus.CANCELLED
                self.command_queue.pop(i)
                self._complete_command(command)
                return True
        
        # 取消执行中的命令
        if command_id in self.executing_commands:
            command = self.executing_commands[command_id]
            command.status = ControlStatus.CANCELLED
            # 这里可以添加取消执行中命令的逻辑
            self._complete_command(command)
            return True
        
        return False
    
    def get_command_status(self, command_id: str) -> Optional[ControlCommand]:
        """获取命令状态"""
        # 检查执行中的命令
        if command_id in self.executing_commands:
            return self.executing_commands[command_id]
        
        # 检查历史命令
        for command in self.command_history:
            if command.command_id == command_id:
                return command
        
        # 检查队列中的命令
        for command in self.command_queue:
            if command.command_id == command_id:
                return command
        
        return None
    
    def get_control_summary(self) -> Dict[str, Any]:
        """获取控制摘要"""
        return {
            'is_enabled': self.is_enabled,
            'emergency_mode': self.emergency_mode,
            'queue_size': len(self.command_queue),
            'executing_commands': len(self.executing_commands),
            'daily_action_count': self.daily_action_count.copy(),
            'control_stats': self.control_stats.copy()
        }
    
    def add_action_callback(self, action: ControlAction, callback: Callable) -> None:
        """添加动作回调"""
        if action not in self.action_callbacks:
            self.action_callbacks[action] = []
        self.action_callbacks[action].append(callback)
    
    def _setup_event_handlers(self) -> None:
        """设置事件处理器"""
        self.event_bus.subscribe('risk_alert.generated', self._on_risk_alert)
    
    def _on_risk_alert(self, event_data: Dict[str, Any]) -> None:
        """处理风险预警事件"""
        alert_data = event_data.get('alert')
        if alert_data:
            # 重构预警对象
            alert = RiskAlert(
                alert_id=alert_data['alert_id'],
                alert_type=AlertType(alert_data['alert_type']),
                risk_type=RiskType(alert_data['risk_type']),
                risk_level=RiskLevel(alert_data['risk_level']),
                title=alert_data['title'],
                message=alert_data['message']
            )
            self.handle_risk_alert(alert)
    
    def _add_to_queue(self, command: ControlCommand) -> None:
        """添加命令到队列"""
        self.command_queue.append(command)
        
        # 如果是紧急动作，立即处理
        if command.action in [ControlAction.EMERGENCY_STOP, ControlAction.STOP_TRADING]:
            self.process_command_queue()
    
    def _execute_command(self, command: ControlCommand) -> None:
        """执行命令"""
        command.status = ControlStatus.EXECUTING
        command.executed_at = datetime.now()
        self.executing_commands[command.command_id] = command
        
        try:
            # 根据动作类型执行相应操作
            if command.action == ControlAction.WARNING:
                self._execute_warning(command)
            elif command.action == ControlAction.REDUCE_POSITION:
                self._execute_reduce_position(command)
            elif command.action == ControlAction.CLOSE_POSITION:
                self._execute_close_position(command)
            elif command.action == ControlAction.STOP_TRADING:
                self._execute_stop_trading(command)
            elif command.action == ControlAction.EMERGENCY_STOP:
                self._execute_emergency_stop(command)
            elif command.action == ControlAction.INCREASE_MARGIN:
                self._execute_increase_margin(command)
            elif command.action == ControlAction.LIMIT_EXPOSURE:
                self._execute_limit_exposure(command)
            else:
                raise ValueError(f"Unknown control action: {command.action}")
            
            command.status = ControlStatus.COMPLETED
            command.result = "Success"
            
        except Exception as e:
            command.status = ControlStatus.FAILED
            command.error_message = str(e)
            self.logger.error(f"Failed to execute command {command.command_id}: {e}")
        
        finally:
            self._complete_command(command)
    
    def _execute_command_immediately(self, command: ControlCommand) -> None:
        """立即执行命令（用于紧急情况）"""
        self.command_queue.insert(0, command)  # 插入到队列前端
        self.process_command_queue()
    
    def _execute_warning(self, command: ControlCommand) -> None:
        """执行警告"""
        message = command.parameters.get('message', 'Risk warning triggered')
        self.logger.warning(f"Risk Control Warning: {message}")
        
        # 调用回调
        self._call_action_callbacks(command.action, command)
    
    def _execute_reduce_position(self, command: ControlCommand) -> None:
        """执行减仓"""
        symbol = command.target_symbol
        reduction_ratio = command.parameters.get('reduction_ratio', 0.5)  # 默认减仓50%
        
        if not symbol or not self.portfolio_manager:
            raise ValueError("Symbol or portfolio manager not available")
        
        # 获取当前持仓
        portfolio = self.portfolio_manager.get_portfolio()
        if symbol not in portfolio.positions:
            raise ValueError(f"No position found for symbol: {symbol}")
        
        position = portfolio.positions[symbol]
        if position.quantity == 0:
            return  # 没有持仓
        
        # 计算减仓数量
        reduce_quantity = abs(position.quantity) * reduction_ratio
        
        # 创建减仓订单
        order = ExecutionOrder(
            order_id=f"reduce_{command.command_id}",
            symbol=symbol,
            side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=reduce_quantity,
            metadata={'risk_control': True, 'command_id': command.command_id}
        )
        
        # 提交订单
        if self.execution_engine:
            self.execution_engine.submit_order(order)
        
        self.logger.info(f"Submitted reduce position order for {symbol}: {reduce_quantity}")
        
        # 调用回调
        self._call_action_callbacks(command.action, command)
    
    def _execute_close_position(self, command: ControlCommand) -> None:
        """执行平仓"""
        symbol = command.target_symbol
        
        if not symbol or not self.portfolio_manager:
            raise ValueError("Symbol or portfolio manager not available")
        
        # 获取当前持仓
        portfolio = self.portfolio_manager.get_portfolio()
        if symbol not in portfolio.positions:
            raise ValueError(f"No position found for symbol: {symbol}")
        
        position = portfolio.positions[symbol]
        if position.quantity == 0:
            return  # 没有持仓
        
        # 创建平仓订单
        order = ExecutionOrder(
            order_id=f"close_{command.command_id}",
            symbol=symbol,
            side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity),
            metadata={'risk_control': True, 'command_id': command.command_id}
        )
        
        # 提交订单
        if self.execution_engine:
            self.execution_engine.submit_order(order)
        
        self.logger.info(f"Submitted close position order for {symbol}")
        
        # 调用回调
        self._call_action_callbacks(command.action, command)
    
    def _execute_stop_trading(self, command: ControlCommand) -> None:
        """执行停止交易"""
        strategy = command.target_strategy
        
        if strategy and self.portfolio_manager:
            # 停止特定策略
            self.portfolio_manager.pause_strategy(strategy)
            self.logger.info(f"Stopped trading for strategy: {strategy}")
        elif self.execution_engine:
            # 停止所有交易
            self.execution_engine.pause()
            self.logger.info("Stopped all trading")
        
        # 调用回调
        self._call_action_callbacks(command.action, command)
    
    def _execute_emergency_stop(self, command: ControlCommand) -> None:
        """执行紧急停止"""
        reason = command.parameters.get('reason', 'Emergency stop')
        
        # 停止所有交易
        if self.execution_engine:
            self.execution_engine.emergency_stop()
        
        # 停止投资组合管理
        if self.portfolio_manager:
            self.portfolio_manager.emergency_stop()
        
        self.logger.critical(f"Emergency stop executed: {reason}")
        
        # 调用回调
        self._call_action_callbacks(command.action, command)
    
    def _execute_increase_margin(self, command: ControlCommand) -> None:
        """执行增加保证金"""
        # 这里需要与经纪商API集成
        self.logger.info("Increase margin action executed")
        
        # 调用回调
        self._call_action_callbacks(command.action, command)
    
    def _execute_limit_exposure(self, command: ControlCommand) -> None:
        """执行限制敞口"""
        # 设置敞口限制
        self.logger.info("Limit exposure action executed")
        
        # 调用回调
        self._call_action_callbacks(command.action, command)
    
    def _complete_command(self, command: ControlCommand) -> None:
        """完成命令"""
        # 从执行中移除
        if command.command_id in self.executing_commands:
            del self.executing_commands[command.command_id]
        
        # 添加到历史
        self.command_history.append(command)
        
        # 更新统计
        self.control_stats['total_commands_executed'] += 1
        if command.status == ControlStatus.COMPLETED:
            self.control_stats['successful_commands'] += 1
        elif command.status == ControlStatus.FAILED:
            self.control_stats['failed_commands'] += 1
        
        self.control_stats['last_action_time'] = datetime.now()
        
        # 更新日常计数
        self._update_daily_count(command.action)
        
        # 发布事件
        self.event_bus.publish('risk_control.command_completed', {
            'command': {
                'command_id': command.command_id,
                'action': command.action.value,
                'status': command.status.value,
                'result': command.result,
                'error_message': command.error_message
            },
            'timestamp': datetime.now()
        })
    
    def _call_action_callbacks(self, action: ControlAction, command: ControlCommand) -> None:
        """调用动作回调"""
        if action in self.action_callbacks:
            for callback in self.action_callbacks[action]:
                try:
                    callback(command)
                except Exception as e:
                    self.logger.error(f"Error in action callback: {e}")
    
    def _determine_control_actions(self, alert: RiskAlert) -> List[ControlAction]:
        """根据预警确定控制动作"""
        actions = []
        
        # 根据风险级别确定动作
        if alert.risk_level == RiskLevel.EMERGENCY:
            actions.append(ControlAction.EMERGENCY_STOP)
        elif alert.risk_level == RiskLevel.CRITICAL:
            if alert.risk_type in [RiskType.CONCENTRATION_RISK, RiskType.LEVERAGE_RISK]:
                actions.append(ControlAction.REDUCE_POSITION)
            else:
                actions.append(ControlAction.STOP_TRADING)
        elif alert.risk_level == RiskLevel.HIGH:
            actions.append(ControlAction.REDUCE_POSITION)
        elif alert.risk_level == RiskLevel.MEDIUM:
            actions.append(ControlAction.WARNING)
        
        return actions
    
    def _get_action_priority(self, action: ControlAction) -> int:
        """获取动作优先级"""
        priority_map = {
            ControlAction.EMERGENCY_STOP: 1000,
            ControlAction.STOP_TRADING: 900,
            ControlAction.CLOSE_POSITION: 800,
            ControlAction.REDUCE_POSITION: 700,
            ControlAction.LIMIT_EXPOSURE: 600,
            ControlAction.INCREASE_MARGIN: 500,
            ControlAction.WARNING: 100,
            ControlAction.NONE: 0
        }
        return priority_map.get(action, 0)
    
    def _check_daily_limits(self, action: ControlAction) -> bool:
        """检查日常限制"""
        # 重置日常计数（如果是新的一天）
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_action_count.clear()
            self.last_reset_date = today
        
        # 检查限制
        if action in self.max_daily_actions:
            current_count = self.daily_action_count.get(action, 0)
            max_count = self.max_daily_actions[action]
            return current_count < max_count
        
        return True
    
    def _update_daily_count(self, action: ControlAction) -> None:
        """更新日常计数"""
        if action in self.max_daily_actions:
            self.daily_action_count[action] = self.daily_action_count.get(action, 0) + 1