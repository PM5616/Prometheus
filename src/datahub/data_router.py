"""Data Router Module

数据路由器，负责将数据分发到不同的处理器、存储系统或下游服务。

功能特性：
- 智能数据路由
- 负载均衡
- 故障转移
- 路由规则配置
- 性能监控
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.common.models.market import Ticker, Kline, OrderBook
from src.common.exceptions.data import DataProcessingError, DataRoutingError
from .processors.base import BaseProcessor
from .storage.base import BaseStorage


class RoutingStrategy(Enum):
    """路由策略枚举"""
    ROUND_ROBIN = "round_robin"  # 轮询
    WEIGHTED = "weighted"  # 加权
    HASH = "hash"  # 哈希
    RANDOM = "random"  # 随机
    PRIORITY = "priority"  # 优先级
    LOAD_BASED = "load_based"  # 基于负载
    CONTENT_BASED = "content_based"  # 基于内容


class RouteTarget:
    """路由目标类"""
    
    def __init__(self,
                 name: str,
                 target: Union[BaseProcessor, BaseStorage, Callable],
                 weight: float = 1.0,
                 priority: int = 0,
                 max_load: int = 100,
                 enabled: bool = True,
                 health_check_interval: int = 30):
        """
        初始化路由目标
        
        Args:
            name: 目标名称
            target: 目标对象（处理器、存储或函数）
            weight: 权重（用于加权路由）
            priority: 优先级（数值越小优先级越高）
            max_load: 最大负载
            enabled: 是否启用
            health_check_interval: 健康检查间隔（秒）
        """
        self.name = name
        self.target = target
        self.weight = weight
        self.priority = priority
        self.max_load = max_load
        self.enabled = enabled
        self.health_check_interval = health_check_interval
        
        # 运行时状态
        self.current_load = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.last_health_check = datetime.now()
        self.is_healthy = True
        self.avg_response_time = 0.0
        self.response_times = deque(maxlen=100)  # 保留最近100次响应时间
        
        # 创建时间
        self.created_at = datetime.now()
    
    def is_available(self) -> bool:
        """
        检查目标是否可用
        
        Returns:
            bool: 是否可用
        """
        return (
            self.enabled and 
            self.is_healthy and 
            self.current_load < self.max_load
        )
    
    def update_stats(self, success: bool, response_time: float):
        """
        更新统计信息
        
        Args:
            success: 是否成功
            response_time: 响应时间
        """
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # 更新响应时间
        self.response_times.append(response_time)
        if self.response_times:
            self.avg_response_time = sum(self.response_times) / len(self.response_times)
    
    def get_success_rate(self) -> float:
        """
        获取成功率
        
        Returns:
            float: 成功率
        """
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'name': self.name,
            'weight': self.weight,
            'priority': self.priority,
            'max_load': self.max_load,
            'enabled': self.enabled,
            'current_load': self.current_load,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.get_success_rate(),
            'is_healthy': self.is_healthy,
            'avg_response_time': self.avg_response_time,
            'created_at': self.created_at.isoformat()
        }


class RoutingRule:
    """路由规则类"""
    
    def __init__(self,
                 name: str,
                 condition: Callable[[Any], bool],
                 targets: List[str],
                 strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN,
                 priority: int = 0,
                 enabled: bool = True):
        """
        初始化路由规则
        
        Args:
            name: 规则名称
            condition: 条件函数
            targets: 目标名称列表
            strategy: 路由策略
            priority: 优先级
            enabled: 是否启用
        """
        self.name = name
        self.condition = condition
        self.targets = targets
        self.strategy = strategy
        self.priority = priority
        self.enabled = enabled
        self.created_at = datetime.now()
        
        # 统计信息
        self.match_count = 0
        self.route_count = 0
    
    def matches(self, data: Any) -> bool:
        """
        检查数据是否匹配规则
        
        Args:
            data: 数据
            
        Returns:
            bool: 是否匹配
        """
        if not self.enabled:
            return False
        
        try:
            result = self.condition(data)
            if result:
                self.match_count += 1
            return result
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'name': self.name,
            'targets': self.targets,
            'strategy': self.strategy.value,
            'priority': self.priority,
            'enabled': self.enabled,
            'match_count': self.match_count,
            'route_count': self.route_count,
            'created_at': self.created_at.isoformat()
        }


class DataRouter:
    """数据路由器
    
    负责将数据智能分发到不同的处理器、存储系统或下游服务。
    """
    
    def __init__(self, name: str = "DataRouter", config: Optional[Dict] = None):
        """
        初始化数据路由器
        
        Args:
            name: 路由器名称
            config: 配置参数
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # 路由目标和规则
        self.targets: Dict[str, RouteTarget] = {}
        self.rules: List[RoutingRule] = []
        
        # 路由状态
        self.round_robin_index = 0
        self.routing_stats = {
            'total_routed': 0,
            'successful_routes': 0,
            'failed_routes': 0,
            'avg_routing_time': 0.0
        }
        
        # 配置参数
        self.max_concurrent_routes = self.config.get('max_concurrent_routes', 10)
        self.default_timeout = self.config.get('default_timeout', 30.0)
        self.health_check_enabled = self.config.get('health_check_enabled', True)
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        
        # 线程池
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent_routes,
            thread_name_prefix=f"{name}_router"
        )
        
        # 健康检查任务
        self._health_check_task = None
        self._running = False
        
        self.logger.info(f"数据路由器 {name} 初始化完成")
    
    def start(self):
        """
        启动路由器
        """
        if self._running:
            return
        
        self._running = True
        
        # 启动健康检查
        if self.health_check_enabled:
            self._start_health_check()
        
        self.logger.info(f"数据路由器 {self.name} 已启动")
    
    def stop(self):
        """
        停止路由器
        """
        if not self._running:
            return
        
        self._running = False
        
        # 停止健康检查
        if self._health_check_task:
            self._health_check_task.cancel()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        self.logger.info(f"数据路由器 {self.name} 已停止")
    
    def add_target(self, target: RouteTarget):
        """
        添加路由目标
        
        Args:
            target: 路由目标
        """
        self.targets[target.name] = target
        self.logger.info(f"添加路由目标: {target.name}")
    
    def remove_target(self, name: str) -> bool:
        """
        移除路由目标
        
        Args:
            name: 目标名称
            
        Returns:
            bool: 是否成功移除
        """
        if name in self.targets:
            del self.targets[name]
            self.logger.info(f"移除路由目标: {name}")
            return True
        return False
    
    def add_rule(self, rule: RoutingRule):
        """
        添加路由规则
        
        Args:
            rule: 路由规则
        """
        self.rules.append(rule)
        # 按优先级排序
        self.rules.sort(key=lambda r: r.priority)
        self.logger.info(f"添加路由规则: {rule.name}")
    
    def remove_rule(self, name: str) -> bool:
        """
        移除路由规则
        
        Args:
            name: 规则名称
            
        Returns:
            bool: 是否成功移除
        """
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                del self.rules[i]
                self.logger.info(f"移除路由规则: {name}")
                return True
        return False
    
    def route(self, data: Any, **kwargs) -> List[Any]:
        """
        路由数据
        
        Args:
            data: 待路由的数据
            **kwargs: 额外参数
                - force_targets: 强制指定目标列表
                - strategy: 强制指定路由策略
                - timeout: 超时时间
                - async_mode: 是否异步模式
                
        Returns:
            List[Any]: 路由结果列表
        """
        start_time = time.time()
        
        try:
            # 获取路由参数
            force_targets = kwargs.get('force_targets')
            force_strategy = kwargs.get('strategy')
            timeout = kwargs.get('timeout', self.default_timeout)
            async_mode = kwargs.get('async_mode', False)
            
            # 确定路由目标
            if force_targets:
                target_names = force_targets
                strategy = force_strategy or RoutingStrategy.ROUND_ROBIN
            else:
                target_names, strategy = self._find_matching_targets(data)
            
            if not target_names:
                raise DataRoutingError("没有找到匹配的路由目标")
            
            # 选择具体目标
            selected_targets = self._select_targets(target_names, strategy, data)
            
            if not selected_targets:
                raise DataRoutingError("没有可用的路由目标")
            
            # 执行路由
            if async_mode:
                results = self._route_async(data, selected_targets, timeout)
            else:
                results = self._route_sync(data, selected_targets, timeout)
            
            # 更新统计
            self._update_routing_stats(True, time.time() - start_time)
            
            return results
            
        except Exception as e:
            self.logger.error(f"数据路由失败: {e}")
            self._update_routing_stats(False, time.time() - start_time)
            raise DataRoutingError(f"数据路由失败: {e}")
    
    def _find_matching_targets(self, data: Any) -> tuple:
        """
        查找匹配的路由目标
        
        Args:
            data: 数据
            
        Returns:
            tuple: (目标名称列表, 路由策略)
        """
        # 按优先级检查规则
        for rule in self.rules:
            if rule.matches(data):
                rule.route_count += 1
                return rule.targets, rule.strategy
        
        # 如果没有匹配的规则，使用默认目标
        default_targets = [name for name, target in self.targets.items() if target.is_available()]
        return default_targets, RoutingStrategy.ROUND_ROBIN
    
    def _select_targets(self, target_names: List[str], strategy: RoutingStrategy, data: Any) -> List[RouteTarget]:
        """
        根据策略选择目标
        
        Args:
            target_names: 候选目标名称列表
            strategy: 路由策略
            data: 数据
            
        Returns:
            List[RouteTarget]: 选中的目标列表
        """
        # 过滤可用目标
        available_targets = [
            self.targets[name] for name in target_names 
            if name in self.targets and self.targets[name].is_available()
        ]
        
        if not available_targets:
            return []
        
        # 根据策略选择
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available_targets)
        elif strategy == RoutingStrategy.WEIGHTED:
            return self._select_weighted(available_targets)
        elif strategy == RoutingStrategy.HASH:
            return self._select_hash(available_targets, data)
        elif strategy == RoutingStrategy.RANDOM:
            return self._select_random(available_targets)
        elif strategy == RoutingStrategy.PRIORITY:
            return self._select_priority(available_targets)
        elif strategy == RoutingStrategy.LOAD_BASED:
            return self._select_load_based(available_targets)
        elif strategy == RoutingStrategy.CONTENT_BASED:
            return self._select_content_based(available_targets, data)
        else:
            return [available_targets[0]]  # 默认选择第一个
    
    def _select_round_robin(self, targets: List[RouteTarget]) -> List[RouteTarget]:
        """
        轮询选择
        
        Args:
            targets: 可用目标列表
            
        Returns:
            List[RouteTarget]: 选中的目标
        """
        if not targets:
            return []
        
        selected = targets[self.round_robin_index % len(targets)]
        self.round_robin_index += 1
        return [selected]
    
    def _select_weighted(self, targets: List[RouteTarget]) -> List[RouteTarget]:
        """
        加权选择
        
        Args:
            targets: 可用目标列表
            
        Returns:
            List[RouteTarget]: 选中的目标
        """
        import random
        
        if not targets:
            return []
        
        # 计算权重
        weights = [target.weight for target in targets]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return [targets[0]]
        
        # 加权随机选择
        rand_val = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for target in targets:
            cumulative_weight += target.weight
            if rand_val <= cumulative_weight:
                return [target]
        
        return [targets[-1]]
    
    def _select_hash(self, targets: List[RouteTarget], data: Any) -> List[RouteTarget]:
        """
        哈希选择
        
        Args:
            targets: 可用目标列表
            data: 数据
            
        Returns:
            List[RouteTarget]: 选中的目标
        """
        if not targets:
            return []
        
        # 生成数据哈希
        data_str = str(data)
        if hasattr(data, 'symbol'):
            data_str = data.symbol
        elif isinstance(data, dict) and 'symbol' in data:
            data_str = data['symbol']
        
        hash_val = int(hashlib.md5(data_str.encode()).hexdigest(), 16)
        index = hash_val % len(targets)
        
        return [targets[index]]
    
    def _select_random(self, targets: List[RouteTarget]) -> List[RouteTarget]:
        """
        随机选择
        
        Args:
            targets: 可用目标列表
            
        Returns:
            List[RouteTarget]: 选中的目标
        """
        import random
        
        if not targets:
            return []
        
        return [random.choice(targets)]
    
    def _select_priority(self, targets: List[RouteTarget]) -> List[RouteTarget]:
        """
        优先级选择
        
        Args:
            targets: 可用目标列表
            
        Returns:
            List[RouteTarget]: 选中的目标
        """
        if not targets:
            return []
        
        # 按优先级排序（数值越小优先级越高）
        sorted_targets = sorted(targets, key=lambda t: t.priority)
        return [sorted_targets[0]]
    
    def _select_load_based(self, targets: List[RouteTarget]) -> List[RouteTarget]:
        """
        基于负载选择
        
        Args:
            targets: 可用目标列表
            
        Returns:
            List[RouteTarget]: 选中的目标
        """
        if not targets:
            return []
        
        # 选择负载最低的目标
        min_load_target = min(targets, key=lambda t: t.current_load)
        return [min_load_target]
    
    def _select_content_based(self, targets: List[RouteTarget], data: Any) -> List[RouteTarget]:
        """
        基于内容选择
        
        Args:
            targets: 可用目标列表
            data: 数据
            
        Returns:
            List[RouteTarget]: 选中的目标
        """
        # 这里可以根据数据内容特征选择目标
        # 例如：根据数据类型、大小、来源等
        
        if isinstance(data, Ticker):
            # Ticker数据优先选择实时处理器
            realtime_targets = [t for t in targets if 'realtime' in t.name.lower()]
            if realtime_targets:
                return [realtime_targets[0]]
        
        elif isinstance(data, Kline):
            # K线数据优先选择历史数据处理器
            historical_targets = [t for t in targets if 'historical' in t.name.lower()]
            if historical_targets:
                return [historical_targets[0]]
        
        # 默认选择第一个
        return [targets[0]] if targets else []
    
    def _route_sync(self, data: Any, targets: List[RouteTarget], timeout: float) -> List[Any]:
        """
        同步路由
        
        Args:
            data: 数据
            targets: 目标列表
            timeout: 超时时间
            
        Returns:
            List[Any]: 结果列表
        """
        results = []
        
        for target in targets:
            try:
                # 增加负载
                target.current_load += 1
                
                start_time = time.time()
                
                # 执行路由
                result = self._execute_route(target, data, timeout)
                
                response_time = time.time() - start_time
                target.update_stats(True, response_time)
                
                results.append({
                    'target': target.name,
                    'success': True,
                    'result': result,
                    'response_time': response_time
                })
                
            except Exception as e:
                response_time = time.time() - start_time
                target.update_stats(False, response_time)
                
                results.append({
                    'target': target.name,
                    'success': False,
                    'error': str(e),
                    'response_time': response_time
                })
                
                self.logger.error(f"路由到目标 {target.name} 失败: {e}")
            
            finally:
                # 减少负载
                target.current_load = max(0, target.current_load - 1)
        
        return results
    
    def _route_async(self, data: Any, targets: List[RouteTarget], timeout: float) -> List[Any]:
        """
        异步路由
        
        Args:
            data: 数据
            targets: 目标列表
            timeout: 超时时间
            
        Returns:
            List[Any]: 结果列表
        """
        results = []
        
        # 提交异步任务
        futures = []
        for target in targets:
            future = self.executor.submit(self._execute_route_with_stats, target, data, timeout)
            futures.append((target, future))
        
        # 收集结果
        for target, future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                results.append({
                    'target': target.name,
                    'success': False,
                    'error': str(e),
                    'response_time': 0.0
                })
                self.logger.error(f"异步路由到目标 {target.name} 失败: {e}")
        
        return results
    
    def _execute_route_with_stats(self, target: RouteTarget, data: Any, timeout: float) -> Dict[str, Any]:
        """
        执行路由并更新统计信息
        
        Args:
            target: 路由目标
            data: 数据
            timeout: 超时时间
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        try:
            # 增加负载
            target.current_load += 1
            
            start_time = time.time()
            
            # 执行路由
            result = self._execute_route(target, data, timeout)
            
            response_time = time.time() - start_time
            target.update_stats(True, response_time)
            
            return {
                'target': target.name,
                'success': True,
                'result': result,
                'response_time': response_time
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            target.update_stats(False, response_time)
            
            return {
                'target': target.name,
                'success': False,
                'error': str(e),
                'response_time': response_time
            }
        
        finally:
            # 减少负载
            target.current_load = max(0, target.current_load - 1)
    
    def _execute_route(self, target: RouteTarget, data: Any, timeout: float) -> Any:
        """
        执行具体的路由操作
        
        Args:
            target: 路由目标
            data: 数据
            timeout: 超时时间
            
        Returns:
            Any: 执行结果
        """
        # 根据目标类型执行不同操作
        if isinstance(target.target, BaseProcessor):
            # 处理器
            return target.target.process(data)
        
        elif isinstance(target.target, BaseStorage):
            # 存储
            if isinstance(data, Ticker):
                return target.target.save_ticker(data)
            elif isinstance(data, Kline):
                return target.target.save_kline(data)
            elif isinstance(data, list):
                if data and isinstance(data[0], Ticker):
                    return target.target.save_tickers(data)
                elif data and isinstance(data[0], Kline):
                    return target.target.save_klines(data)
            
            # 默认保存为通用数据
            return target.target.save_data(data)
        
        elif callable(target.target):
            # 可调用对象
            return target.target(data)
        
        else:
            raise ValueError(f"不支持的目标类型: {type(target.target)}")
    
    def _update_routing_stats(self, success: bool, routing_time: float):
        """
        更新路由统计信息
        
        Args:
            success: 是否成功
            routing_time: 路由时间
        """
        self.routing_stats['total_routed'] += 1
        
        if success:
            self.routing_stats['successful_routes'] += 1
        else:
            self.routing_stats['failed_routes'] += 1
        
        # 更新平均路由时间
        total_routes = self.routing_stats['total_routed']
        current_avg = self.routing_stats['avg_routing_time']
        
        self.routing_stats['avg_routing_time'] = (
            (current_avg * (total_routes - 1) + routing_time) / total_routes
        )
    
    def _start_health_check(self):
        """
        启动健康检查
        """
        async def health_check_loop():
            while self._running:
                try:
                    await self._perform_health_checks()
                    await asyncio.sleep(self.health_check_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"健康检查失败: {e}")
                    await asyncio.sleep(self.health_check_interval)
        
        # 在新的事件循环中运行
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        self._health_check_task = loop.create_task(health_check_loop())
    
    async def _perform_health_checks(self):
        """
        执行健康检查
        """
        for target in self.targets.values():
            try:
                # 检查是否需要健康检查
                time_since_last_check = (datetime.now() - target.last_health_check).total_seconds()
                if time_since_last_check < target.health_check_interval:
                    continue
                
                # 执行健康检查
                if hasattr(target.target, 'health_check'):
                    is_healthy = await asyncio.to_thread(target.target.health_check)
                else:
                    # 简单的可用性检查
                    is_healthy = target.target is not None
                
                target.is_healthy = is_healthy
                target.last_health_check = datetime.now()
                
                if not is_healthy:
                    self.logger.warning(f"目标 {target.name} 健康检查失败")
                
            except Exception as e:
                target.is_healthy = False
                target.last_health_check = datetime.now()
                self.logger.error(f"目标 {target.name} 健康检查异常: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取路由器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        target_stats = {name: target.to_dict() for name, target in self.targets.items()}
        rule_stats = [rule.to_dict() for rule in self.rules]
        
        return {
            'router_name': self.name,
            'routing_stats': self.routing_stats,
            'targets': target_stats,
            'rules': rule_stats,
            'config': {
                'max_concurrent_routes': self.max_concurrent_routes,
                'default_timeout': self.default_timeout,
                'health_check_enabled': self.health_check_enabled,
                'health_check_interval': self.health_check_interval,
                'retry_attempts': self.retry_attempts,
                'retry_delay': self.retry_delay
            },
            'status': {
                'running': self._running,
                'targets_count': len(self.targets),
                'rules_count': len(self.rules),
                'available_targets': len([t for t in self.targets.values() if t.is_available()])
            }
        }
    
    def get_target_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定目标的统计信息
        
        Args:
            name: 目标名称
            
        Returns:
            Optional[Dict[str, Any]]: 目标统计信息
        """
        if name in self.targets:
            return self.targets[name].to_dict()
        return None
    
    def reset_stats(self):
        """
        重置统计信息
        """
        self.routing_stats = {
            'total_routed': 0,
            'successful_routes': 0,
            'failed_routes': 0,
            'avg_routing_time': 0.0
        }
        
        for target in self.targets.values():
            target.total_requests = 0
            target.successful_requests = 0
            target.failed_requests = 0
            target.response_times.clear()
            target.avg_response_time = 0.0
        
        for rule in self.rules:
            rule.match_count = 0
            rule.route_count = 0
        
        self.logger.info("路由器统计信息已重置")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()