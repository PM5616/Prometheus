"""Portfolio Rebalancer

投资组合再平衡器，负责根据优化结果和风险控制要求进行投资组合再平衡
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..common.exceptions.base import PrometheusException
from ..common.models.base import BaseModel
from .models import Portfolio, Position, Signal, SignalType, PositionType


class RebalanceFrequency(Enum):
    """再平衡频率"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SIGNAL_DRIVEN = "signal_driven"  # 信号驱动
    THRESHOLD_DRIVEN = "threshold_driven"  # 阈值驱动


class RebalanceMethod(Enum):
    """再平衡方法"""
    FULL_REBALANCE = "full_rebalance"  # 完全再平衡
    PARTIAL_REBALANCE = "partial_rebalance"  # 部分再平衡
    THRESHOLD_REBALANCE = "threshold_rebalance"  # 阈值再平衡
    COST_AWARE_REBALANCE = "cost_aware_rebalance"  # 成本感知再平衡


@dataclass
class RebalanceConfig:
    """再平衡配置"""
    frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY
    method: RebalanceMethod = RebalanceMethod.THRESHOLD_REBALANCE
    
    # 阈值设置
    weight_threshold: float = 0.05  # 权重偏差阈值（5%）
    value_threshold: float = 10000.0  # 价值偏差阈值
    
    # 交易成本设置
    transaction_cost_rate: float = 0.001  # 交易成本率（0.1%）
    min_trade_amount: float = 1000.0  # 最小交易金额
    
    # 流动性约束
    max_position_size: float = 0.1  # 单个资产最大仓位（10%）
    max_turnover_rate: float = 0.5  # 最大换手率（50%）
    
    # 时间约束
    min_holding_period: int = 1  # 最小持有期（天）
    max_rebalance_frequency: int = 1  # 最大再平衡频率（天）
    
    # 风险控制
    max_tracking_error: float = 0.05  # 最大跟踪误差（5%）
    max_concentration_risk: float = 0.3  # 最大集中度风险（30%）


@dataclass
class RebalanceOrder:
    """再平衡订单"""
    symbol: str
    current_weight: float
    target_weight: float
    current_quantity: float
    target_quantity: float
    trade_quantity: float
    trade_value: float
    order_type: str  # 'BUY' or 'SELL'
    priority: int = 1  # 优先级（1-10）
    estimated_cost: float = 0.0
    
    @property
    def weight_deviation(self) -> float:
        """权重偏差"""
        return abs(self.target_weight - self.current_weight)
    
    @property
    def is_significant(self) -> bool:
        """是否为显著交易"""
        return abs(self.trade_value) > 1000.0  # 交易金额大于1000


class RebalancerException(PrometheusException):
    """再平衡器异常"""
    pass


class PortfolioRebalancer(BaseModel):
    """投资组合再平衡器
    
    负责根据优化结果和风险控制要求进行投资组合再平衡，包括：
    1. 再平衡需求检测
    2. 再平衡方案生成
    3. 交易成本优化
    4. 流动性约束处理
    5. 风险控制验证
    """
    
    def __init__(self, config: Optional[RebalanceConfig] = None):
        super().__init__()
        self.config = config or RebalanceConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 再平衡历史
        self.rebalance_history: List[Dict[str, Any]] = []
        
        # 最后再平衡时间
        self.last_rebalance_time: Optional[datetime] = None
        
        # 持仓历史（用于最小持有期检查）
        self.position_history: Dict[str, datetime] = {}
        
        self.logger.info("Portfolio Rebalancer initialized")
    
    async def check_rebalance_need(
        self, 
        portfolio: Portfolio,
        target_weights: Dict[str, float],
        current_time: Optional[datetime] = None
    ) -> bool:
        """检查是否需要再平衡
        
        Args:
            portfolio: 当前投资组合
            target_weights: 目标权重
            current_time: 当前时间
            
        Returns:
            是否需要再平衡
        """
        try:
            current_time = current_time or datetime.now()
            
            # 检查时间频率
            if not self._check_time_frequency(current_time):
                return False
            
            # 检查权重偏差
            if self._check_weight_deviation(portfolio, target_weights):
                self.logger.info("Rebalance needed: weight deviation threshold exceeded")
                return True
            
            # 检查价值偏差
            if self._check_value_deviation(portfolio, target_weights):
                self.logger.info("Rebalance needed: value deviation threshold exceeded")
                return True
            
            # 检查风险指标
            if await self._check_risk_deviation(portfolio, target_weights):
                self.logger.info("Rebalance needed: risk threshold exceeded")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking rebalance need: {e}")
            return False
    
    async def generate_rebalance_orders(
        self,
        portfolio: Portfolio,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        current_time: Optional[datetime] = None
    ) -> List[RebalanceOrder]:
        """生成再平衡订单
        
        Args:
            portfolio: 当前投资组合
            target_weights: 目标权重
            current_prices: 当前价格
            current_time: 当前时间
            
        Returns:
            再平衡订单列表
        """
        try:
            current_time = current_time or datetime.now()
            orders = []
            
            # 计算当前权重
            current_weights = self._calculate_current_weights(portfolio)
            total_value = float(portfolio.total_value)
            
            # 为每个资产生成订单
            all_symbols = set(current_weights.keys()) | set(target_weights.keys())
            
            for symbol in all_symbols:
                current_weight = current_weights.get(symbol, 0.0)
                target_weight = target_weights.get(symbol, 0.0)
                
                # 跳过权重偏差小的资产
                if abs(target_weight - current_weight) < self.config.weight_threshold:
                    continue
                
                # 检查最小持有期
                if not self._check_min_holding_period(symbol, current_time):
                    continue
                
                # 计算交易数量
                current_quantity = 0.0
                if symbol in portfolio.positions:
                    current_quantity = float(portfolio.positions[symbol].quantity)
                
                current_price = current_prices.get(symbol, 0.0)
                if current_price <= 0:
                    continue
                
                target_value = total_value * target_weight
                target_quantity = target_value / current_price
                trade_quantity = target_quantity - current_quantity
                trade_value = trade_quantity * current_price
                
                # 检查最小交易金额
                if abs(trade_value) < self.config.min_trade_amount:
                    continue
                
                # 计算交易成本
                estimated_cost = abs(trade_value) * self.config.transaction_cost_rate
                
                # 创建订单
                order = RebalanceOrder(
                    symbol=symbol,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    current_quantity=current_quantity,
                    target_quantity=target_quantity,
                    trade_quantity=trade_quantity,
                    trade_value=trade_value,
                    order_type='BUY' if trade_quantity > 0 else 'SELL',
                    priority=self._calculate_order_priority(current_weight, target_weight),
                    estimated_cost=estimated_cost
                )
                
                orders.append(order)
            
            # 根据方法优化订单
            optimized_orders = await self._optimize_orders(orders, portfolio, total_value)
            
            # 按优先级排序
            optimized_orders.sort(key=lambda x: (-x.priority, -abs(x.trade_value)))
            
            self.logger.info(f"Generated {len(optimized_orders)} rebalance orders")
            return optimized_orders
            
        except Exception as e:
            self.logger.error(f"Error generating rebalance orders: {e}")
            raise RebalancerException(f"Failed to generate rebalance orders: {e}")
    
    async def execute_rebalance(
        self,
        portfolio: Portfolio,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        current_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """执行再平衡
        
        Args:
            portfolio: 当前投资组合
            target_weights: 目标权重
            current_prices: 当前价格
            current_time: 当前时间
            
        Returns:
            再平衡结果
        """
        try:
            current_time = current_time or datetime.now()
            
            # 检查是否需要再平衡
            if not await self.check_rebalance_need(portfolio, target_weights, current_time):
                return {
                    'status': 'skipped',
                    'reason': 'No rebalance needed',
                    'timestamp': current_time.isoformat()
                }
            
            # 生成再平衡订单
            orders = await self.generate_rebalance_orders(
                portfolio, target_weights, current_prices, current_time
            )
            
            if not orders:
                return {
                    'status': 'skipped',
                    'reason': 'No valid orders generated',
                    'timestamp': current_time.isoformat()
                }
            
            # 验证再平衡方案
            validation_result = await self._validate_rebalance_plan(orders, portfolio)
            if not validation_result['valid']:
                return {
                    'status': 'failed',
                    'reason': f"Validation failed: {validation_result['reason']}",
                    'timestamp': current_time.isoformat()
                }
            
            # 计算再平衡统计
            stats = self._calculate_rebalance_stats(orders, portfolio)
            
            # 记录再平衡历史
            rebalance_record = {
                'timestamp': current_time.isoformat(),
                'orders_count': len(orders),
                'total_trade_value': sum(abs(order.trade_value) for order in orders),
                'total_cost': sum(order.estimated_cost for order in orders),
                'turnover_rate': stats['turnover_rate'],
                'orders': [{
                    'symbol': order.symbol,
                    'current_weight': order.current_weight,
                    'target_weight': order.target_weight,
                    'trade_quantity': order.trade_quantity,
                    'trade_value': order.trade_value,
                    'order_type': order.order_type,
                    'estimated_cost': order.estimated_cost
                } for order in orders]
            }
            
            self.rebalance_history.append(rebalance_record)
            self.last_rebalance_time = current_time
            
            # 更新持仓历史
            for order in orders:
                if order.order_type == 'BUY':
                    self.position_history[order.symbol] = current_time
            
            return {
                'status': 'success',
                'orders': orders,
                'stats': stats,
                'timestamp': current_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing rebalance: {e}")
            return {
                'status': 'error',
                'reason': str(e),
                'timestamp': current_time.isoformat() if current_time else datetime.now().isoformat()
            }
    
    def _check_time_frequency(self, current_time: datetime) -> bool:
        """检查时间频率"""
        if self.last_rebalance_time is None:
            return True
        
        time_diff = current_time - self.last_rebalance_time
        
        if self.config.frequency == RebalanceFrequency.DAILY:
            return time_diff >= timedelta(days=1)
        elif self.config.frequency == RebalanceFrequency.WEEKLY:
            return time_diff >= timedelta(weeks=1)
        elif self.config.frequency == RebalanceFrequency.MONTHLY:
            return time_diff >= timedelta(days=30)
        elif self.config.frequency == RebalanceFrequency.QUARTERLY:
            return time_diff >= timedelta(days=90)
        else:
            return time_diff >= timedelta(days=self.config.max_rebalance_frequency)
    
    def _check_weight_deviation(self, portfolio: Portfolio, target_weights: Dict[str, float]) -> bool:
        """检查权重偏差"""
        current_weights = self._calculate_current_weights(portfolio)
        
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            if abs(target_weight - current_weight) > self.config.weight_threshold:
                return True
        
        return False
    
    def _check_value_deviation(self, portfolio: Portfolio, target_weights: Dict[str, float]) -> bool:
        """检查价值偏差"""
        current_weights = self._calculate_current_weights(portfolio)
        total_value = float(portfolio.total_value)
        
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            value_deviation = abs(target_weight - current_weight) * total_value
            if value_deviation > self.config.value_threshold:
                return True
        
        return False
    
    async def _check_risk_deviation(self, portfolio: Portfolio, target_weights: Dict[str, float]) -> bool:
        """检查风险偏差"""
        # 简化实现：检查集中度风险
        max_weight = max(target_weights.values()) if target_weights else 0.0
        return max_weight > self.config.max_concentration_risk
    
    def _calculate_current_weights(self, portfolio: Portfolio) -> Dict[str, float]:
        """计算当前权重"""
        total_value = float(portfolio.total_value)
        if total_value <= 0:
            return {}
        
        weights = {}
        for symbol, position in portfolio.positions.items():
            weight = float(position.market_value) / total_value
            weights[symbol] = weight
        
        return weights
    
    def _check_min_holding_period(self, symbol: str, current_time: datetime) -> bool:
        """检查最小持有期"""
        if symbol not in self.position_history:
            return True
        
        holding_time = current_time - self.position_history[symbol]
        return holding_time >= timedelta(days=self.config.min_holding_period)
    
    def _calculate_order_priority(self, current_weight: float, target_weight: float) -> int:
        """计算订单优先级"""
        deviation = abs(target_weight - current_weight)
        
        if deviation > 0.1:  # 10%以上偏差
            return 10
        elif deviation > 0.05:  # 5-10%偏差
            return 8
        elif deviation > 0.02:  # 2-5%偏差
            return 6
        else:
            return 4
    
    async def _optimize_orders(self, orders: List[RebalanceOrder], portfolio: Portfolio, total_value: float) -> List[RebalanceOrder]:
        """优化订单"""
        if self.config.method == RebalanceMethod.FULL_REBALANCE:
            return orders
        
        elif self.config.method == RebalanceMethod.THRESHOLD_REBALANCE:
            # 只保留超过阈值的订单
            return [order for order in orders if order.weight_deviation > self.config.weight_threshold]
        
        elif self.config.method == RebalanceMethod.COST_AWARE_REBALANCE:
            # 成本感知优化
            return self._cost_aware_optimization(orders, total_value)
        
        else:
            return orders
    
    def _cost_aware_optimization(self, orders: List[RebalanceOrder], total_value: float) -> List[RebalanceOrder]:
        """成本感知优化"""
        optimized_orders = []
        
        for order in orders:
            # 计算收益成本比
            weight_improvement = order.weight_deviation
            cost_ratio = order.estimated_cost / total_value
            
            # 只有收益大于成本的订单才执行
            if weight_improvement > cost_ratio * 2:  # 收益至少是成本的2倍
                optimized_orders.append(order)
        
        return optimized_orders
    
    async def _validate_rebalance_plan(self, orders: List[RebalanceOrder], portfolio: Portfolio) -> Dict[str, Any]:
        """验证再平衡方案"""
        try:
            # 检查换手率
            total_trade_value = sum(abs(order.trade_value) for order in orders)
            turnover_rate = total_trade_value / float(portfolio.total_value)
            
            if turnover_rate > self.config.max_turnover_rate:
                return {
                    'valid': False,
                    'reason': f'Turnover rate {turnover_rate:.2%} exceeds limit {self.config.max_turnover_rate:.2%}'
                }
            
            # 检查单个资产仓位限制
            for order in orders:
                if order.target_weight > self.config.max_position_size:
                    return {
                        'valid': False,
                        'reason': f'Position size {order.target_weight:.2%} for {order.symbol} exceeds limit {self.config.max_position_size:.2%}'
                    }
            
            return {'valid': True, 'reason': 'Validation passed'}
            
        except Exception as e:
            return {'valid': False, 'reason': f'Validation error: {e}'}
    
    def _calculate_rebalance_stats(self, orders: List[RebalanceOrder], portfolio: Portfolio) -> Dict[str, Any]:
        """计算再平衡统计"""
        total_trade_value = sum(abs(order.trade_value) for order in orders)
        total_cost = sum(order.estimated_cost for order in orders)
        turnover_rate = total_trade_value / float(portfolio.total_value)
        
        buy_orders = [order for order in orders if order.order_type == 'BUY']
        sell_orders = [order for order in orders if order.order_type == 'SELL']
        
        return {
            'total_orders': len(orders),
            'buy_orders': len(buy_orders),
            'sell_orders': len(sell_orders),
            'total_trade_value': total_trade_value,
            'total_cost': total_cost,
            'cost_ratio': total_cost / total_trade_value if total_trade_value > 0 else 0.0,
            'turnover_rate': turnover_rate,
            'avg_weight_deviation': np.mean([order.weight_deviation for order in orders]),
            'max_weight_deviation': max([order.weight_deviation for order in orders]) if orders else 0.0
        }
    
    def get_rebalance_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取再平衡历史"""
        return self.rebalance_history[-limit:]
    
    def get_rebalance_stats(self) -> Dict[str, Any]:
        """获取再平衡统计"""
        if not self.rebalance_history:
            return {}
        
        total_rebalances = len(self.rebalance_history)
        total_trade_value = sum(record['total_trade_value'] for record in self.rebalance_history)
        total_cost = sum(record['total_cost'] for record in self.rebalance_history)
        avg_turnover = np.mean([record['turnover_rate'] for record in self.rebalance_history])
        
        return {
            'total_rebalances': total_rebalances,
            'total_trade_value': total_trade_value,
            'total_cost': total_cost,
            'avg_cost_ratio': total_cost / total_trade_value if total_trade_value > 0 else 0.0,
            'avg_turnover_rate': avg_turnover,
            'last_rebalance': self.last_rebalance_time.isoformat() if self.last_rebalance_time else None
        }
    
    def update_config(self, config: RebalanceConfig) -> None:
        """更新配置"""
        self.config = config
        self.logger.info("Rebalancer configuration updated")
    
    def clear_history(self) -> None:
        """清空历史记录"""
        self.rebalance_history.clear()
        self.position_history.clear()
        self.last_rebalance_time = None
        self.logger.info("Rebalancer history cleared")