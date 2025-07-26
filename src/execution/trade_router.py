"""Trade Router Module

交易路由器模块，负责将订单路由到合适的市场和交易所。

主要功能：
- 市场选择和路由
- 流动性聚合
- 最优执行路径
- 负载均衡
- 故障转移
"""

import threading
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import json
import heapq

from ..common.logging import get_logger
from ..common.exceptions.execution import TradeRouterError, RoutingError
from .engine import ExecutionOrder, OrderSide, OrderType


class RoutingStrategy(Enum):
    """路由策略枚举"""
    BEST_PRICE = "best_price"          # 最优价格
    LOWEST_COST = "lowest_cost"        # 最低成本
    FASTEST_EXECUTION = "fastest_execution"  # 最快执行
    HIGHEST_LIQUIDITY = "highest_liquidity"  # 最高流动性
    LOAD_BALANCE = "load_balance"      # 负载均衡
    SMART_ORDER = "smart_order"        # 智能路由
    CUSTOM = "custom"                  # 自定义


class MarketStatus(Enum):
    """市场状态枚举"""
    ACTIVE = "active"        # 活跃
    INACTIVE = "inactive"    # 非活跃
    SUSPENDED = "suspended"  # 暂停
    MAINTENANCE = "maintenance"  # 维护
    ERROR = "error"          # 错误


@dataclass
class MarketInfo:
    """市场信息"""
    market_id: str
    name: str
    status: MarketStatus = MarketStatus.ACTIVE
    supported_symbols: List[str] = field(default_factory=list)
    supported_order_types: List[OrderType] = field(default_factory=list)
    
    # 费用信息
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    min_order_size: float = 0.0
    max_order_size: float = float('inf')
    
    # 性能指标
    avg_latency: float = 0.0  # 平均延迟（毫秒）
    success_rate: float = 1.0  # 成功率
    uptime: float = 1.0       # 可用性
    
    # 流动性信息
    liquidity_score: float = 1.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    spread: float = 0.0
    
    # 连接信息
    connector_class: str = ""
    config: Dict = field(default_factory=dict)
    
    # 统计信息
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    total_volume: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_cost(self, order: ExecutionOrder) -> float:
        """计算订单成本
        
        Args:
            order: 执行订单
            
        Returns:
            float: 预估成本
        """
        if not order.price:
            return 0.0
        
        trade_value = order.quantity * order.price
        
        # 根据订单类型选择费率
        if order.order_type == OrderType.MARKET:
            fee_rate = self.taker_fee
        else:
            fee_rate = self.maker_fee
        
        # 基础费用
        base_cost = trade_value * fee_rate
        
        # 滑点成本（简化模型）
        slippage_cost = trade_value * (self.spread / 2)
        
        return base_cost + slippage_cost
    
    def can_handle_order(self, order: ExecutionOrder) -> bool:
        """检查是否可以处理订单
        
        Args:
            order: 执行订单
            
        Returns:
            bool: 是否可以处理
        """
        # 检查市场状态
        if self.status != MarketStatus.ACTIVE:
            return False
        
        # 检查品种支持
        if self.supported_symbols and order.symbol not in self.supported_symbols:
            return False
        
        # 检查订单类型支持
        if self.supported_order_types and order.order_type not in self.supported_order_types:
            return False
        
        # 检查订单大小
        if order.quantity < self.min_order_size or order.quantity > self.max_order_size:
            return False
        
        return True
    
    def update_performance(self, latency: float, success: bool):
        """更新性能指标
        
        Args:
            latency: 延迟时间
            success: 是否成功
        """
        # 更新延迟（移动平均）
        alpha = 0.1
        self.avg_latency = alpha * latency + (1 - alpha) * self.avg_latency
        
        # 更新成功率
        self.total_orders += 1
        if success:
            self.successful_orders += 1
        else:
            self.failed_orders += 1
        
        self.success_rate = self.successful_orders / self.total_orders
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 市场信息字典
        """
        return {
            'market_id': self.market_id,
            'name': self.name,
            'status': self.status.value,
            'supported_symbols': self.supported_symbols,
            'supported_order_types': [ot.value for ot in self.supported_order_types],
            'maker_fee': self.maker_fee,
            'taker_fee': self.taker_fee,
            'min_order_size': self.min_order_size,
            'max_order_size': self.max_order_size,
            'avg_latency': self.avg_latency,
            'success_rate': self.success_rate,
            'uptime': self.uptime,
            'liquidity_score': self.liquidity_score,
            'bid_depth': self.bid_depth,
            'ask_depth': self.ask_depth,
            'spread': self.spread,
            'connector_class': self.connector_class,
            'config': self.config,
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'failed_orders': self.failed_orders,
            'total_volume': self.total_volume,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class RoutingRule:
    """路由规则"""
    rule_id: str
    name: str
    priority: int = 0
    enabled: bool = True
    
    # 匹配条件
    symbols: List[str] = field(default_factory=list)
    order_types: List[OrderType] = field(default_factory=list)
    sides: List[OrderSide] = field(default_factory=list)
    min_quantity: Optional[float] = None
    max_quantity: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    strategy_ids: List[str] = field(default_factory=list)
    
    # 路由目标
    target_markets: List[str] = field(default_factory=list)
    routing_strategy: RoutingStrategy = RoutingStrategy.BEST_PRICE
    
    # 分配权重（用于负载均衡）
    market_weights: Dict[str, float] = field(default_factory=dict)
    
    # 时间条件
    start_time: Optional[str] = None  # HH:MM:SS
    end_time: Optional[str] = None    # HH:MM:SS
    weekdays: List[int] = field(default_factory=list)  # 0-6, Monday=0
    
    def matches(self, order: ExecutionOrder) -> bool:
        """检查订单是否匹配规则
        
        Args:
            order: 执行订单
            
        Returns:
            bool: 是否匹配
        """
        if not self.enabled:
            return False
        
        # 检查品种
        if self.symbols and order.symbol not in self.symbols:
            return False
        
        # 检查订单类型
        if self.order_types and order.order_type not in self.order_types:
            return False
        
        # 检查订单方向
        if self.sides and order.side not in self.sides:
            return False
        
        # 检查数量范围
        if self.min_quantity is not None and order.quantity < self.min_quantity:
            return False
        
        if self.max_quantity is not None and order.quantity > self.max_quantity:
            return False
        
        # 检查价值范围
        if order.price:
            order_value = order.quantity * order.price
            
            if self.min_value is not None and order_value < self.min_value:
                return False
            
            if self.max_value is not None and order_value > self.max_value:
                return False
        
        # 检查策略ID
        if self.strategy_ids and order.strategy_id not in self.strategy_ids:
            return False
        
        # 检查时间条件
        if not self._check_time_conditions():
            return False
        
        return True
    
    def _check_time_conditions(self) -> bool:
        """检查时间条件
        
        Returns:
            bool: 是否满足时间条件
        """
        now = datetime.now()
        
        # 检查星期
        if self.weekdays and now.weekday() not in self.weekdays:
            return False
        
        # 检查时间范围
        if self.start_time and self.end_time:
            current_time = now.time()
            start_time = datetime.strptime(self.start_time, "%H:%M:%S").time()
            end_time = datetime.strptime(self.end_time, "%H:%M:%S").time()
            
            if not (start_time <= current_time <= end_time):
                return False
        
        return True
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 路由规则字典
        """
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'priority': self.priority,
            'enabled': self.enabled,
            'symbols': self.symbols,
            'order_types': [ot.value for ot in self.order_types],
            'sides': [s.value for s in self.sides],
            'min_quantity': self.min_quantity,
            'max_quantity': self.max_quantity,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'strategy_ids': self.strategy_ids,
            'target_markets': self.target_markets,
            'routing_strategy': self.routing_strategy.value,
            'market_weights': self.market_weights,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'weekdays': self.weekdays
        }


@dataclass
class RoutingResult:
    """路由结果"""
    order_id: str
    target_market: str
    routing_strategy: RoutingStrategy
    estimated_cost: float
    estimated_latency: float
    confidence: float
    reason: str
    alternatives: List[Tuple[str, float]] = field(default_factory=list)  # (market, score)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 路由结果字典
        """
        return {
            'order_id': self.order_id,
            'target_market': self.target_market,
            'routing_strategy': self.routing_strategy.value,
            'estimated_cost': self.estimated_cost,
            'estimated_latency': self.estimated_latency,
            'confidence': self.confidence,
            'reason': self.reason,
            'alternatives': self.alternatives,
            'timestamp': self.timestamp.isoformat()
        }


class TradeRouter:
    """交易路由器
    
    负责将订单路由到合适的市场和交易所。
    """
    
    def __init__(self, config: Dict = None):
        """初始化交易路由器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 市场信息
        self.markets: Dict[str, MarketInfo] = {}
        
        # 路由规则
        self.routing_rules: List[RoutingRule] = []
        
        # 默认路由策略
        self.default_strategy = RoutingStrategy.BEST_PRICE
        self.default_market = self.config.get('default_market', 'default')
        
        # 负载均衡
        self.load_balancer_weights: Dict[str, float] = {}
        self.market_loads: Dict[str, int] = defaultdict(int)
        
        # 故障转移
        self.failover_enabled = self.config.get('failover_enabled', True)
        self.max_retries = self.config.get('max_retries', 3)
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 路由历史
        self.routing_history: deque = deque(maxlen=10000)
        
        # 回调函数
        self.on_routing_callbacks: List[Callable] = []
        self.on_failover_callbacks: List[Callable] = []
        
        # 日志记录
        self.logger = get_logger("TradeRouter")
        
        # 统计信息
        self.stats = {
            'total_routes': 0,
            'successful_routes': 0,
            'failed_routes': 0,
            'failover_count': 0,
            'avg_routing_time': 0.0,
            'last_routing_time': None
        }
        
        self.logger.info("交易路由器初始化完成")
    
    def add_market(self, market_info: MarketInfo):
        """添加市场
        
        Args:
            market_info: 市场信息
        """
        with self._lock:
            self.markets[market_info.market_id] = market_info
            self.load_balancer_weights[market_info.market_id] = 1.0
            self.market_loads[market_info.market_id] = 0
            
            self.logger.info(f"添加市场: {market_info.market_id} ({market_info.name})")
    
    def remove_market(self, market_id: str):
        """移除市场
        
        Args:
            market_id: 市场ID
        """
        with self._lock:
            if market_id in self.markets:
                del self.markets[market_id]
                self.load_balancer_weights.pop(market_id, None)
                self.market_loads.pop(market_id, None)
                
                self.logger.info(f"移除市场: {market_id}")
    
    def update_market_status(self, market_id: str, status: MarketStatus):
        """更新市场状态
        
        Args:
            market_id: 市场ID
            status: 市场状态
        """
        with self._lock:
            if market_id in self.markets:
                old_status = self.markets[market_id].status
                self.markets[market_id].status = status
                self.markets[market_id].last_updated = datetime.now()
                
                self.logger.info(f"市场状态更新: {market_id} {old_status.value} -> {status.value}")
    
    def add_routing_rule(self, rule: RoutingRule):
        """添加路由规则
        
        Args:
            rule: 路由规则
        """
        with self._lock:
            self.routing_rules.append(rule)
            # 按优先级排序
            self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
            
            self.logger.info(f"添加路由规则: {rule.rule_id} ({rule.name})")
    
    def remove_routing_rule(self, rule_id: str):
        """移除路由规则
        
        Args:
            rule_id: 规则ID
        """
        with self._lock:
            self.routing_rules = [r for r in self.routing_rules if r.rule_id != rule_id]
            
            self.logger.info(f"移除路由规则: {rule_id}")
    
    def route_order(self, order: ExecutionOrder) -> RoutingResult:
        """路由订单
        
        Args:
            order: 执行订单
            
        Returns:
            RoutingResult: 路由结果
            
        Raises:
            RoutingError: 路由失败
        """
        start_time = datetime.now()
        
        try:
            with self._lock:
                # 查找匹配的路由规则
                matching_rule = self._find_matching_rule(order)
                
                if matching_rule:
                    result = self._route_by_rule(order, matching_rule)
                else:
                    result = self._route_by_default_strategy(order)
                
                # 记录路由历史
                self.routing_history.append(result)
                
                # 更新统计
                self.stats['total_routes'] += 1
                self.stats['successful_routes'] += 1
                
                routing_time = (datetime.now() - start_time).total_seconds()
                self.stats['avg_routing_time'] = (
                    self.stats['avg_routing_time'] * (self.stats['total_routes'] - 1) + routing_time
                ) / self.stats['total_routes']
                self.stats['last_routing_time'] = datetime.now()
                
                # 更新市场负载
                self.market_loads[result.target_market] += 1
                
                # 调用回调
                self._notify_routing(result)
                
                self.logger.info(f"订单路由完成: {order.order_id} -> {result.target_market}")
                return result
                
        except Exception as e:
            self.stats['failed_routes'] += 1
            self.logger.error(f"订单路由失败: {order.order_id} - {e}")
            raise RoutingError(f"路由失败: {e}")
    
    def _find_matching_rule(self, order: ExecutionOrder) -> Optional[RoutingRule]:
        """查找匹配的路由规则
        
        Args:
            order: 执行订单
            
        Returns:
            Optional[RoutingRule]: 匹配的规则
        """
        for rule in self.routing_rules:
            if rule.matches(order):
                return rule
        
        return None
    
    def _route_by_rule(self, order: ExecutionOrder, rule: RoutingRule) -> RoutingResult:
        """按规则路由
        
        Args:
            order: 执行订单
            rule: 路由规则
            
        Returns:
            RoutingResult: 路由结果
        """
        # 获取候选市场
        candidate_markets = self._get_candidate_markets(order, rule.target_markets)
        
        if not candidate_markets:
            raise RoutingError("没有可用的候选市场")
        
        # 根据策略选择市场
        if rule.routing_strategy == RoutingStrategy.LOAD_BALANCE:
            target_market = self._select_by_load_balance(candidate_markets, rule.market_weights)
        else:
            target_market = self._select_by_strategy(order, candidate_markets, rule.routing_strategy)
        
        # 计算预估指标
        market_info = self.markets[target_market]
        estimated_cost = market_info.calculate_cost(order)
        estimated_latency = market_info.avg_latency
        
        return RoutingResult(
            order_id=order.order_id,
            target_market=target_market,
            routing_strategy=rule.routing_strategy,
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency,
            confidence=0.9,
            reason=f"匹配规则: {rule.name}",
            alternatives=[(m, self._calculate_market_score(order, m, rule.routing_strategy)) 
                         for m in candidate_markets if m != target_market]
        )
    
    def _route_by_default_strategy(self, order: ExecutionOrder) -> RoutingResult:
        """按默认策略路由
        
        Args:
            order: 执行订单
            
        Returns:
            RoutingResult: 路由结果
        """
        # 获取所有可用市场
        candidate_markets = self._get_candidate_markets(order)
        
        if not candidate_markets:
            # 使用默认市场
            if self.default_market in self.markets:
                target_market = self.default_market
            else:
                raise RoutingError("没有可用的市场")
        else:
            # 按默认策略选择
            target_market = self._select_by_strategy(order, candidate_markets, self.default_strategy)
        
        # 计算预估指标
        market_info = self.markets[target_market]
        estimated_cost = market_info.calculate_cost(order)
        estimated_latency = market_info.avg_latency
        
        return RoutingResult(
            order_id=order.order_id,
            target_market=target_market,
            routing_strategy=self.default_strategy,
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency,
            confidence=0.7,
            reason="默认策略路由",
            alternatives=[(m, self._calculate_market_score(order, m, self.default_strategy)) 
                         for m in candidate_markets if m != target_market]
        )
    
    def _get_candidate_markets(self, order: ExecutionOrder, 
                              target_markets: List[str] = None) -> List[str]:
        """获取候选市场
        
        Args:
            order: 执行订单
            target_markets: 目标市场列表
            
        Returns:
            List[str]: 候选市场列表
        """
        candidates = []
        
        markets_to_check = target_markets if target_markets else list(self.markets.keys())
        
        for market_id in markets_to_check:
            if market_id in self.markets:
                market_info = self.markets[market_id]
                if market_info.can_handle_order(order):
                    candidates.append(market_id)
        
        return candidates
    
    def _select_by_strategy(self, order: ExecutionOrder, candidates: List[str],
                           strategy: RoutingStrategy) -> str:
        """按策略选择市场
        
        Args:
            order: 执行订单
            candidates: 候选市场
            strategy: 路由策略
            
        Returns:
            str: 选中的市场ID
        """
        if not candidates:
            raise RoutingError("没有候选市场")
        
        if len(candidates) == 1:
            return candidates[0]
        
        # 计算每个市场的得分
        market_scores = []
        for market_id in candidates:
            score = self._calculate_market_score(order, market_id, strategy)
            market_scores.append((market_id, score))
        
        # 按得分排序
        market_scores.sort(key=lambda x: x[1], reverse=True)
        
        return market_scores[0][0]
    
    def _calculate_market_score(self, order: ExecutionOrder, market_id: str,
                               strategy: RoutingStrategy) -> float:
        """计算市场得分
        
        Args:
            order: 执行订单
            market_id: 市场ID
            strategy: 路由策略
            
        Returns:
            float: 市场得分
        """
        market_info = self.markets[market_id]
        
        if strategy == RoutingStrategy.BEST_PRICE:
            # 价格优先（考虑费用和滑点）
            cost = market_info.calculate_cost(order)
            return 1.0 / (1.0 + cost) if cost > 0 else 1.0
        
        elif strategy == RoutingStrategy.LOWEST_COST:
            # 成本最低
            cost = market_info.calculate_cost(order)
            return 1.0 / (1.0 + cost) if cost > 0 else 1.0
        
        elif strategy == RoutingStrategy.FASTEST_EXECUTION:
            # 执行最快
            latency = market_info.avg_latency
            return 1.0 / (1.0 + latency) if latency > 0 else 1.0
        
        elif strategy == RoutingStrategy.HIGHEST_LIQUIDITY:
            # 流动性最高
            return market_info.liquidity_score
        
        elif strategy == RoutingStrategy.SMART_ORDER:
            # 智能路由（综合考虑）
            cost_score = 1.0 / (1.0 + market_info.calculate_cost(order))
            latency_score = 1.0 / (1.0 + market_info.avg_latency)
            liquidity_score = market_info.liquidity_score
            success_score = market_info.success_rate
            
            # 加权平均
            return (cost_score * 0.3 + latency_score * 0.2 + 
                   liquidity_score * 0.3 + success_score * 0.2)
        
        else:
            # 默认得分
            return market_info.success_rate
    
    def _select_by_load_balance(self, candidates: List[str], 
                               weights: Dict[str, float] = None) -> str:
        """按负载均衡选择市场
        
        Args:
            candidates: 候选市场
            weights: 权重配置
            
        Returns:
            str: 选中的市场ID
        """
        if not candidates:
            raise RoutingError("没有候选市场")
        
        if len(candidates) == 1:
            return candidates[0]
        
        # 计算加权负载
        weighted_loads = []
        for market_id in candidates:
            current_load = self.market_loads.get(market_id, 0)
            weight = weights.get(market_id, 1.0) if weights else self.load_balancer_weights.get(market_id, 1.0)
            
            # 负载越低，得分越高
            weighted_load = current_load / weight if weight > 0 else float('inf')
            weighted_loads.append((market_id, weighted_load))
        
        # 选择负载最低的市场
        weighted_loads.sort(key=lambda x: x[1])
        return weighted_loads[0][0]
    
    def handle_routing_failure(self, order: ExecutionOrder, 
                              failed_market: str, error: Exception) -> Optional[RoutingResult]:
        """处理路由失败
        
        Args:
            order: 执行订单
            failed_market: 失败的市场
            error: 错误信息
            
        Returns:
            Optional[RoutingResult]: 故障转移结果
        """
        if not self.failover_enabled:
            return None
        
        try:
            with self._lock:
                # 更新市场状态
                if failed_market in self.markets:
                    self.markets[failed_market].update_performance(0, False)
                
                # 获取备选市场
                candidate_markets = self._get_candidate_markets(order)
                candidate_markets = [m for m in candidate_markets if m != failed_market]
                
                if not candidate_markets:
                    self.logger.error(f"故障转移失败，没有备选市场: {order.order_id}")
                    return None
                
                # 选择备选市场
                target_market = self._select_by_strategy(order, candidate_markets, 
                                                        RoutingStrategy.SMART_ORDER)
                
                # 创建故障转移结果
                market_info = self.markets[target_market]
                result = RoutingResult(
                    order_id=order.order_id,
                    target_market=target_market,
                    routing_strategy=RoutingStrategy.SMART_ORDER,
                    estimated_cost=market_info.calculate_cost(order),
                    estimated_latency=market_info.avg_latency,
                    confidence=0.6,
                    reason=f"故障转移，原市场: {failed_market}"
                )
                
                # 更新统计
                self.stats['failover_count'] += 1
                
                # 调用回调
                self._notify_failover(order, failed_market, target_market, error)
                
                self.logger.info(f"故障转移成功: {order.order_id} {failed_market} -> {target_market}")
                return result
                
        except Exception as e:
            self.logger.error(f"故障转移异常: {order.order_id} - {e}")
            return None
    
    def update_market_performance(self, market_id: str, latency: float, success: bool):
        """更新市场性能
        
        Args:
            market_id: 市场ID
            latency: 延迟时间
            success: 是否成功
        """
        with self._lock:
            if market_id in self.markets:
                self.markets[market_id].update_performance(latency, success)
    
    def update_market_liquidity(self, market_id: str, bid_depth: float, 
                               ask_depth: float, spread: float):
        """更新市场流动性
        
        Args:
            market_id: 市场ID
            bid_depth: 买盘深度
            ask_depth: 卖盘深度
            spread: 价差
        """
        with self._lock:
            if market_id in self.markets:
                market_info = self.markets[market_id]
                market_info.bid_depth = bid_depth
                market_info.ask_depth = ask_depth
                market_info.spread = spread
                
                # 更新流动性得分
                total_depth = bid_depth + ask_depth
                if total_depth > 0 and spread > 0:
                    market_info.liquidity_score = total_depth / spread
                
                market_info.last_updated = datetime.now()
    
    def get_market_info(self, market_id: str) -> Optional[MarketInfo]:
        """获取市场信息
        
        Args:
            market_id: 市场ID
            
        Returns:
            Optional[MarketInfo]: 市场信息
        """
        return self.markets.get(market_id)
    
    def get_all_markets(self) -> Dict[str, MarketInfo]:
        """获取所有市场信息
        
        Returns:
            Dict[str, MarketInfo]: 市场信息字典
        """
        return self.markets.copy()
    
    def get_routing_rules(self) -> List[RoutingRule]:
        """获取路由规则
        
        Returns:
            List[RoutingRule]: 路由规则列表
        """
        return self.routing_rules.copy()
    
    def get_routing_history(self, limit: int = 100) -> List[RoutingResult]:
        """获取路由历史
        
        Args:
            limit: 返回数量限制
            
        Returns:
            List[RoutingResult]: 路由历史
        """
        return list(self.routing_history)[-limit:]
    
    def _notify_routing(self, result: RoutingResult):
        """通知路由完成
        
        Args:
            result: 路由结果
        """
        for callback in self.on_routing_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"路由回调执行失败: {e}")
    
    def _notify_failover(self, order: ExecutionOrder, failed_market: str,
                        target_market: str, error: Exception):
        """通知故障转移
        
        Args:
            order: 执行订单
            failed_market: 失败的市场
            target_market: 目标市场
            error: 错误信息
        """
        for callback in self.on_failover_callbacks:
            try:
                callback(order, failed_market, target_market, error)
            except Exception as e:
                self.logger.error(f"故障转移回调执行失败: {e}")
    
    def add_routing_callback(self, callback: Callable):
        """添加路由回调
        
        Args:
            callback: 回调函数
        """
        self.on_routing_callbacks.append(callback)
    
    def add_failover_callback(self, callback: Callable):
        """添加故障转移回调
        
        Args:
            callback: 回调函数
        """
        self.on_failover_callbacks.append(callback)
    
    def set_load_balancer_weights(self, weights: Dict[str, float]):
        """设置负载均衡权重
        
        Args:
            weights: 权重配置
        """
        with self._lock:
            self.load_balancer_weights.update(weights)
            self.logger.info(f"负载均衡权重已更新: {weights}")
    
    def reset_market_loads(self):
        """重置市场负载"""
        with self._lock:
            self.market_loads.clear()
            for market_id in self.markets:
                self.market_loads[market_id] = 0
            
            self.logger.info("市场负载已重置")
    
    def get_statistics(self) -> Dict:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self._lock:
            market_stats = {}
            for market_id, market_info in self.markets.items():
                market_stats[market_id] = {
                    'status': market_info.status.value,
                    'total_orders': market_info.total_orders,
                    'success_rate': market_info.success_rate,
                    'avg_latency': market_info.avg_latency,
                    'current_load': self.market_loads.get(market_id, 0)
                }
            
            return {
                'router_stats': self.stats.copy(),
                'market_stats': market_stats,
                'total_markets': len(self.markets),
                'active_markets': len([m for m in self.markets.values() 
                                     if m.status == MarketStatus.ACTIVE]),
                'total_rules': len(self.routing_rules),
                'enabled_rules': len([r for r in self.routing_rules if r.enabled])
            }
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 路由器字典
        """
        return {
            'markets': {mid: market.to_dict() for mid, market in self.markets.items()},
            'routing_rules': [rule.to_dict() for rule in self.routing_rules],
            'default_strategy': self.default_strategy.value,
            'default_market': self.default_market,
            'load_balancer_weights': self.load_balancer_weights.copy(),
            'market_loads': dict(self.market_loads),
            'statistics': self.get_statistics(),
            'config': self.config.copy()
        }
    
    def __str__(self) -> str:
        return f"TradeRouter(markets={len(self.markets)}, rules={len(self.routing_rules)})"
    
    def __repr__(self) -> str:
        return self.__str__()