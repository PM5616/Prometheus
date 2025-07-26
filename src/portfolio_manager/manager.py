"""Portfolio Manager

投资组合管理器核心实现
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

from ..common.exceptions.base import PrometheusException
from ..common.models.base import BaseModel
from .models import (
    Portfolio, Position, Signal, RiskMetrics, PerformanceMetrics,
    OptimizationConfig, RiskConfig, SignalType, PositionType, PositionStatus
)
from .optimizer import PortfolioOptimizer
from .risk_manager import RiskManager
from .signal_processor import SignalProcessor
from .position_manager import PositionManager
from .performance_tracker import PerformanceTracker


class PortfolioManagerException(PrometheusException):
    """投资组合管理器异常"""
    pass


class PortfolioManager(BaseModel):
    """投资组合管理器
    
    负责：
    1. 接收和处理来自策略引擎的信号
    2. 进行投资组合优化和风险管理
    3. 生成执行指令
    4. 监控投资组合表现
    """
    
    def __init__(
        self,
        portfolio_id: UUID,
        initial_capital: Decimal,
        optimization_config: OptimizationConfig,
        risk_config: RiskConfig,
        base_currency: str = "USDT"
    ):
        super().__init__()
        self.portfolio_id = portfolio_id
        self.base_currency = base_currency
        self.optimization_config = optimization_config
        self.risk_config = risk_config
        
        # 初始化投资组合
        self.portfolio = Portfolio(
            id=portfolio_id,
            name=f"Portfolio_{portfolio_id}",
            base_currency=base_currency,
            total_value=initial_capital,
            available_cash=initial_capital
        )
        
        # 初始化子模块
        self.optimizer = PortfolioOptimizer(optimization_config)
        self.risk_manager = RiskManager(risk_config)
        self.signal_processor = SignalProcessor()
        self.position_manager = PositionManager()
        self.performance_tracker = PerformanceTracker()
        
        # 状态管理
        self.signals_buffer: List[Signal] = []
        self.pending_orders: Dict[str, Any] = {}
        self.last_rebalance: Optional[datetime] = None
        self.is_running = False
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Portfolio Manager initialized for portfolio {portfolio_id}")
    
    async def start(self) -> None:
        """启动投资组合管理器"""
        if self.is_running:
            self.logger.warning("Portfolio Manager is already running")
            return
        
        self.is_running = True
        self.logger.info("Starting Portfolio Manager")
        
        # 启动子模块
        await self.risk_manager.start()
        await self.performance_tracker.start()
        
        # 启动主循环
        asyncio.create_task(self._main_loop())
    
    async def stop(self) -> None:
        """停止投资组合管理器"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("Stopping Portfolio Manager")
        
        # 停止子模块
        await self.risk_manager.stop()
        await self.performance_tracker.stop()
    
    async def _main_loop(self) -> None:
        """主循环"""
        while self.is_running:
            try:
                # 处理信号
                await self._process_signals()
                
                # 检查是否需要重新平衡
                if self._should_rebalance():
                    await self._rebalance_portfolio()
                
                # 更新风险指标
                await self._update_risk_metrics()
                
                # 更新绩效指标
                await self._update_performance_metrics()
                
                # 检查风险限制
                await self._check_risk_limits()
                
                await asyncio.sleep(1)  # 1秒循环
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)  # 错误时等待5秒
    
    async def add_signal(self, signal: Signal) -> None:
        """添加交易信号"""
        if signal.is_expired():
            self.logger.warning(f"Ignoring expired signal: {signal.id}")
            return
        
        # 验证信号
        if not self._validate_signal(signal):
            self.logger.warning(f"Invalid signal: {signal.id}")
            return
        
        self.signals_buffer.append(signal)
        self.logger.info(f"Added signal: {signal.strategy_id} - {signal.symbol} - {signal.signal_type}")
    
    async def _process_signals(self) -> None:
        """处理信号缓冲区中的信号"""
        if not self.signals_buffer:
            return
        
        # 处理信号
        processed_signals = await self.signal_processor.process_signals(
            self.signals_buffer, self.portfolio
        )
        
        # 生成交易指令
        for signal in processed_signals:
            await self._generate_trade_instruction(signal)
        
        # 清空缓冲区
        self.signals_buffer.clear()
    
    async def _generate_trade_instruction(self, signal: Signal) -> None:
        """根据信号生成交易指令"""
        try:
            current_position = self.portfolio.get_position(signal.symbol)
            
            if signal.signal_type == SignalType.BUY:
                await self._handle_buy_signal(signal, current_position)
            elif signal.signal_type == SignalType.SELL:
                await self._handle_sell_signal(signal, current_position)
            elif signal.signal_type == SignalType.CLOSE:
                await self._handle_close_signal(signal, current_position)
            
        except Exception as e:
            self.logger.error(f"Error generating trade instruction for signal {signal.id}: {e}")
    
    async def _handle_buy_signal(self, signal: Signal, current_position: Optional[Position]) -> None:
        """处理买入信号"""
        # 计算仓位大小
        position_size = await self._calculate_position_size(signal)
        
        if position_size <= 0:
            self.logger.warning(f"Invalid position size for buy signal: {signal.id}")
            return
        
        # 检查资金是否充足
        required_capital = position_size * (signal.target_price or Decimal('0'))
        if required_capital > self.portfolio.available_cash:
            self.logger.warning(f"Insufficient capital for buy signal: {signal.id}")
            return
        
        # 创建或更新持仓
        if current_position is None:
            position = Position(
                symbol=signal.symbol,
                position_type=PositionType.LONG,
                status=PositionStatus.PENDING,
                quantity=position_size,
                entry_price=signal.target_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy_id=signal.strategy_id,
                entry_time=datetime.now()
            )
            self.portfolio.add_position(position)
        else:
            # 增加现有持仓
            current_position.quantity += position_size
            current_position.status = PositionStatus.PENDING
        
        # 更新可用资金
        self.portfolio.available_cash -= required_capital
        
        self.logger.info(f"Generated buy instruction: {signal.symbol} - {position_size}")
    
    async def _handle_sell_signal(self, signal: Signal, current_position: Optional[Position]) -> None:
        """处理卖出信号"""
        if current_position is None or current_position.quantity <= 0:
            self.logger.warning(f"No position to sell for signal: {signal.id}")
            return
        
        # 计算卖出数量
        sell_quantity = min(signal.position_size or current_position.quantity, 
                           current_position.quantity)
        
        # 更新持仓
        current_position.quantity -= sell_quantity
        current_position.status = PositionStatus.PENDING
        
        if current_position.quantity <= 0:
            current_position.status = PositionStatus.CLOSING
        
        # 更新可用资金
        proceeds = sell_quantity * (signal.target_price or Decimal('0'))
        self.portfolio.available_cash += proceeds
        
        self.logger.info(f"Generated sell instruction: {signal.symbol} - {sell_quantity}")
    
    async def _handle_close_signal(self, signal: Signal, current_position: Optional[Position]) -> None:
        """处理平仓信号"""
        if current_position is None:
            self.logger.warning(f"No position to close for signal: {signal.id}")
            return
        
        # 平仓
        current_position.status = PositionStatus.CLOSING
        current_position.exit_time = datetime.now()
        
        # 计算收益
        if current_position.current_price:
            proceeds = current_position.quantity * current_position.current_price
            self.portfolio.available_cash += proceeds
        
        self.logger.info(f"Generated close instruction: {signal.symbol}")
    
    async def _calculate_position_size(self, signal: Signal) -> Decimal:
        """计算仓位大小"""
        if signal.position_size is not None:
            return signal.position_size
        
        # 使用Kelly公式计算仓位大小
        kelly_fraction = await self._calculate_kelly_fraction(signal)
        
        # 应用风险限制
        max_position_size = self.portfolio.total_value * self.risk_config.position_size_limit
        
        # 计算最终仓位大小
        target_price = signal.target_price or Decimal('1')
        position_value = self.portfolio.total_value * Decimal(str(kelly_fraction))
        position_size = position_value / target_price
        
        return min(position_size, max_position_size / target_price)
    
    async def _calculate_kelly_fraction(self, signal: Signal) -> float:
        """计算Kelly公式的仓位比例"""
        # 简化的Kelly公式实现
        # 实际应用中需要根据历史数据计算胜率和赔率
        win_rate = signal.confidence  # 使用信号置信度作为胜率
        avg_win = 0.05  # 平均盈利5%
        avg_loss = 0.02  # 平均亏损2%
        
        if avg_loss == 0:
            return 0.0
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # 限制Kelly比例在合理范围内
        return max(0.0, min(kelly_fraction, 0.25))  # 最大25%
    
    def _should_rebalance(self) -> bool:
        """检查是否需要重新平衡"""
        if self.last_rebalance is None:
            return True
        
        # 根据配置的重新平衡频率决定
        if self.optimization_config.rebalance_frequency == "daily":
            return datetime.now() - self.last_rebalance >= timedelta(days=1)
        elif self.optimization_config.rebalance_frequency == "weekly":
            return datetime.now() - self.last_rebalance >= timedelta(weeks=1)
        elif self.optimization_config.rebalance_frequency == "monthly":
            return datetime.now() - self.last_rebalance >= timedelta(days=30)
        
        return False
    
    async def _rebalance_portfolio(self) -> None:
        """重新平衡投资组合"""
        try:
            self.logger.info("Starting portfolio rebalancing")
            
            # 获取当前持仓权重
            current_weights = await self._get_current_weights()
            
            # 计算目标权重
            target_weights = await self.optimizer.optimize_portfolio(
                self.portfolio, current_weights
            )
            
            # 生成重新平衡指令
            rebalance_orders = await self._generate_rebalance_orders(
                current_weights, target_weights
            )
            
            # 执行重新平衡
            for order in rebalance_orders:
                await self._execute_rebalance_order(order)
            
            self.last_rebalance = datetime.now()
            self.logger.info("Portfolio rebalancing completed")
            
        except Exception as e:
            self.logger.error(f"Error during portfolio rebalancing: {e}")
    
    async def _get_current_weights(self) -> Dict[str, float]:
        """获取当前持仓权重"""
        total_value = self.portfolio.get_total_market_value()
        if total_value == 0:
            return {}
        
        weights = {}
        for symbol, position in self.portfolio.positions.items():
            market_value = position.get_market_value()
            weights[symbol] = float(market_value / total_value)
        
        return weights
    
    async def _generate_rebalance_orders(self, current_weights: Dict[str, float], 
                                       target_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """生成重新平衡订单"""
        orders = []
        total_value = self.portfolio.get_total_market_value()
        
        for symbol in set(current_weights.keys()) | set(target_weights.keys()):
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.01:  # 1%的阈值
                order_value = total_value * Decimal(str(weight_diff))
                orders.append({
                    'symbol': symbol,
                    'order_type': 'buy' if weight_diff > 0 else 'sell',
                    'value': abs(order_value),
                    'weight_diff': weight_diff
                })
        
        return orders
    
    async def _execute_rebalance_order(self, order: Dict[str, Any]) -> None:
        """执行重新平衡订单"""
        # 这里应该调用执行引擎
        # 暂时只记录日志
        self.logger.info(f"Rebalance order: {order['order_type']} {order['symbol']} - {order['value']}")
    
    async def _update_risk_metrics(self) -> None:
        """更新风险指标"""
        try:
            risk_metrics = await self.risk_manager.calculate_risk_metrics(self.portfolio)
            # 存储风险指标
            self.logger.debug(f"Risk metrics updated: VaR95={risk_metrics.var_95}")
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")
    
    async def _update_performance_metrics(self) -> None:
        """更新绩效指标"""
        try:
            performance_metrics = await self.performance_tracker.calculate_performance(
                self.portfolio
            )
            # 存储绩效指标
            self.logger.debug(f"Performance metrics updated: Return={performance_metrics.total_return}")
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _check_risk_limits(self) -> None:
        """检查风险限制"""
        try:
            violations = await self.risk_manager.check_risk_limits(self.portfolio)
            
            if violations:
                self.logger.warning(f"Risk limit violations detected: {violations}")
                await self._handle_risk_violations(violations)
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    async def _handle_risk_violations(self, violations: List[str]) -> None:
        """处理风险违规"""
        for violation in violations:
            self.logger.critical(f"Risk violation: {violation}")
            
            # 根据违规类型采取相应措施
            if "max_drawdown" in violation:
                await self._emergency_liquidation()
            elif "var" in violation:
                await self._reduce_exposure()
    
    async def _emergency_liquidation(self) -> None:
        """紧急平仓"""
        self.logger.critical("Initiating emergency liquidation")
        
        for position in self.portfolio.positions.values():
            if position.status == PositionStatus.OPEN:
                position.status = PositionStatus.CLOSING
                # 发送平仓指令到执行引擎
    
    async def _reduce_exposure(self) -> None:
        """减少敞口"""
        self.logger.warning("Reducing portfolio exposure")
        
        # 减少最大的持仓
        largest_position = max(
            self.portfolio.positions.values(),
            key=lambda p: p.get_market_value(),
            default=None
        )
        
        if largest_position:
            # 减少50%的持仓
            largest_position.quantity *= Decimal('0.5')
    
    def _validate_signal(self, signal: Signal) -> bool:
        """验证信号有效性"""
        if not signal.symbol:
            return False
        if signal.confidence < 0 or signal.confidence > 1:
            return False
        if signal.target_price is not None and signal.target_price <= 0:
            return False
        return True
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取投资组合摘要"""
        return {
            'portfolio_id': str(self.portfolio.id),
            'total_value': float(self.portfolio.total_value),
            'available_cash': float(self.portfolio.available_cash),
            'total_market_value': float(self.portfolio.get_total_market_value()),
            'total_pnl': float(self.portfolio.get_total_pnl()),
            'exposure': float(self.portfolio.get_exposure()),
            'positions_count': len(self.portfolio.positions),
            'last_updated': self.portfolio.updated_at.isoformat()
        }
    
    async def update_position_prices(self, prices: Dict[str, Decimal]) -> None:
        """更新持仓价格"""
        for symbol, price in prices.items():
            position = self.portfolio.get_position(symbol)
            if position:
                position.update_price(price)
        
        # 更新投资组合总值
        self.portfolio.total_value = (
            self.portfolio.available_cash + 
            self.portfolio.get_total_market_value()
        )
        self.portfolio.updated_at = datetime.now()