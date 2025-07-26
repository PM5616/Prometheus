"""Signal Manager Module

信号管理器模块，负责信号的生成、过滤、路由和执行。

主要功能：
- 信号收集和聚合
- 信号过滤和验证
- 信号路由和分发
- 信号执行管理
- 信号历史记录
"""

import time
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import threading
from queue import Queue, Empty
import json

from .base import StrategySignal, SignalType, SignalStrength
from ..common.logging import get_logger
from ..common.exceptions.strategy import SignalManagerError, SignalValidationError


class SignalStatus(Enum):
    """信号状态枚举"""
    PENDING = "pending"        # 待处理
    VALIDATED = "validated"    # 已验证
    FILTERED = "filtered"      # 已过滤
    ROUTED = "routed"          # 已路由
    EXECUTED = "executed"      # 已执行
    REJECTED = "rejected"      # 已拒绝
    EXPIRED = "expired"        # 已过期
    ERROR = "error"            # 错误状态


class SignalFilter:
    """信号过滤器基类"""
    
    def __init__(self, name: str, config: Dict = None):
        """初始化过滤器
        
        Args:
            name: 过滤器名称
            config: 过滤器配置
        """
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'total_passed': 0,
            'total_rejected': 0,
            'last_process_time': None
        }
    
    def filter(self, signal: StrategySignal) -> Tuple[bool, str]:
        """过滤信号
        
        Args:
            signal: 策略信号
            
        Returns:
            Tuple[bool, str]: (是否通过, 拒绝原因)
        """
        if not self.enabled:
            return True, ""
        
        self.stats['total_processed'] += 1
        self.stats['last_process_time'] = datetime.now()
        
        try:
            passed, reason = self._do_filter(signal)
            
            if passed:
                self.stats['total_passed'] += 1
            else:
                self.stats['total_rejected'] += 1
            
            return passed, reason
            
        except Exception as e:
            self.stats['total_rejected'] += 1
            return False, f"过滤器错误: {e}"
    
    def _do_filter(self, signal: StrategySignal) -> Tuple[bool, str]:
        """执行过滤逻辑（子类实现）
        
        Args:
            signal: 策略信号
            
        Returns:
            Tuple[bool, str]: (是否通过, 拒绝原因)
        """
        return True, ""
    
    def get_stats(self) -> Dict:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = self.stats.copy()
        stats['name'] = self.name
        stats['enabled'] = self.enabled
        return stats


class DuplicateSignalFilter(SignalFilter):
    """重复信号过滤器"""
    
    def __init__(self, config: Dict = None):
        super().__init__("DuplicateSignalFilter", config)
        self.time_window = self.config.get('time_window_seconds', 60)
        self.recent_signals = defaultdict(deque)
    
    def _do_filter(self, signal: StrategySignal) -> Tuple[bool, str]:
        """过滤重复信号"""
        key = f"{signal.symbol}_{signal.signal_type.value}"
        current_time = signal.timestamp
        
        # 清理过期信号
        cutoff_time = current_time - timedelta(seconds=self.time_window)
        recent = self.recent_signals[key]
        
        while recent and recent[0] < cutoff_time:
            recent.popleft()
        
        # 检查是否重复
        if recent:
            return False, f"在 {self.time_window} 秒内存在重复信号"
        
        # 记录新信号
        recent.append(current_time)
        return True, ""


class FrequencyLimitFilter(SignalFilter):
    """频率限制过滤器"""
    
    def __init__(self, config: Dict = None):
        super().__init__("FrequencyLimitFilter", config)
        self.max_signals_per_minute = self.config.get('max_signals_per_minute', 10)
        self.signal_times = defaultdict(deque)
    
    def _do_filter(self, signal: StrategySignal) -> Tuple[bool, str]:
        """限制信号频率"""
        key = signal.symbol
        current_time = signal.timestamp
        
        # 清理一分钟前的信号
        cutoff_time = current_time - timedelta(minutes=1)
        times = self.signal_times[key]
        
        while times and times[0] < cutoff_time:
            times.popleft()
        
        # 检查频率限制
        if len(times) >= self.max_signals_per_minute:
            return False, f"超过频率限制 {self.max_signals_per_minute}/分钟"
        
        # 记录新信号时间
        times.append(current_time)
        return True, ""


class StrengthFilter(SignalFilter):
    """信号强度过滤器"""
    
    def __init__(self, config: Dict = None):
        super().__init__("StrengthFilter", config)
        self.min_strength = SignalStrength(self.config.get('min_strength', 1))
    
    def _do_filter(self, signal: StrategySignal) -> Tuple[bool, str]:
        """过滤信号强度"""
        if signal.strength.value < self.min_strength.value:
            return False, f"信号强度 {signal.strength.value} 低于最小要求 {self.min_strength.value}"
        
        return True, ""


class PriceRangeFilter(SignalFilter):
    """价格范围过滤器"""
    
    def __init__(self, config: Dict = None):
        super().__init__("PriceRangeFilter", config)
        self.price_ranges = self.config.get('price_ranges', {})
    
    def _do_filter(self, signal: StrategySignal) -> Tuple[bool, str]:
        """过滤价格范围"""
        if signal.price is None:
            return True, ""
        
        if signal.symbol in self.price_ranges:
            min_price, max_price = self.price_ranges[signal.symbol]
            if not (min_price <= signal.price <= max_price):
                return False, f"价格 {signal.price} 超出范围 [{min_price}, {max_price}]"
        
        return True, ""


class SignalRouter:
    """信号路由器"""
    
    def __init__(self, name: str, config: Dict = None):
        """初始化路由器
        
        Args:
            name: 路由器名称
            config: 路由器配置
        """
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # 路由规则
        self.routes = {}
        
        # 统计信息
        self.stats = {
            'total_routed': 0,
            'routes_used': defaultdict(int),
            'last_route_time': None
        }
    
    def add_route(self, condition: Callable[[StrategySignal], bool], 
                  handler: Callable[[StrategySignal], None],
                  name: str = None):
        """添加路由规则
        
        Args:
            condition: 路由条件函数
            handler: 处理函数
            name: 路由名称
        """
        route_name = name or f"route_{len(self.routes)}"
        self.routes[route_name] = {
            'condition': condition,
            'handler': handler,
            'enabled': True
        }
    
    def route_signal(self, signal: StrategySignal) -> List[str]:
        """路由信号
        
        Args:
            signal: 策略信号
            
        Returns:
            List[str]: 匹配的路由名称列表
        """
        if not self.enabled:
            return []
        
        matched_routes = []
        
        for route_name, route_info in self.routes.items():
            if not route_info['enabled']:
                continue
            
            try:
                if route_info['condition'](signal):
                    route_info['handler'](signal)
                    matched_routes.append(route_name)
                    self.stats['routes_used'][route_name] += 1
            except Exception as e:
                # 记录路由错误但继续处理其他路由
                pass
        
        self.stats['total_routed'] += 1
        self.stats['last_route_time'] = datetime.now()
        
        return matched_routes
    
    def get_stats(self) -> Dict:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = self.stats.copy()
        stats['name'] = self.name
        stats['enabled'] = self.enabled
        stats['total_routes'] = len(self.routes)
        return stats


class SignalRecord:
    """信号记录"""
    
    def __init__(self, signal: StrategySignal):
        """初始化信号记录
        
        Args:
            signal: 策略信号
        """
        self.signal = signal
        self.status = SignalStatus.PENDING
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.processing_history = []
        self.rejection_reasons = []
        self.execution_result = None
        self.metadata = {}
    
    def update_status(self, status: SignalStatus, reason: str = None, metadata: Dict = None):
        """更新信号状态
        
        Args:
            status: 新状态
            reason: 状态变化原因
            metadata: 附加元数据
        """
        old_status = self.status
        self.status = status
        self.updated_at = datetime.now()
        
        # 记录处理历史
        history_entry = {
            'timestamp': self.updated_at,
            'old_status': old_status.value,
            'new_status': status.value,
            'reason': reason,
            'metadata': metadata or {}
        }
        self.processing_history.append(history_entry)
        
        # 记录拒绝原因
        if status == SignalStatus.REJECTED and reason:
            self.rejection_reasons.append(reason)
        
        # 更新元数据
        if metadata:
            self.metadata.update(metadata)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 信号记录字典
        """
        return {
            'signal': self.signal.to_dict(),
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'processing_history': self.processing_history,
            'rejection_reasons': self.rejection_reasons,
            'execution_result': self.execution_result,
            'metadata': self.metadata
        }


class SignalManager:
    """信号管理器
    
    负责信号的收集、过滤、路由和执行管理。
    """
    
    def __init__(self, config: Dict = None):
        """初始化信号管理器
        
        Args:
            config: 管理器配置
        """
        self.config = config or {}
        
        # 信号队列
        self.signal_queue = Queue(maxsize=self.config.get('max_queue_size', 1000))
        
        # 信号记录
        self.signal_records: Dict[str, SignalRecord] = {}
        self.max_records = self.config.get('max_records', 10000)
        
        # 过滤器
        self.filters: List[SignalFilter] = []
        self._setup_default_filters()
        
        # 路由器
        self.routers: List[SignalRouter] = []
        
        # 执行器
        self.executors = {}
        
        # 管理器状态
        self.is_running = False
        self.processing_thread = None
        
        # 日志记录
        self.logger = get_logger("SignalManager")
        
        # 统计信息
        self.stats = {
            'total_received': 0,
            'total_processed': 0,
            'total_executed': 0,
            'total_rejected': 0,
            'processing_time': 0.0,
            'last_signal_time': None,
            'error_count': 0
        }
        
        # 回调函数
        self.on_signal_processed_callbacks = []
        self.on_signal_executed_callbacks = []
        self.on_signal_rejected_callbacks = []
        
        self.logger.info("信号管理器初始化完成")
    
    def _setup_default_filters(self):
        """设置默认过滤器"""
        filter_configs = self.config.get('filters', {})
        
        # 重复信号过滤器
        if filter_configs.get('duplicate_filter', {}).get('enabled', True):
            self.add_filter(DuplicateSignalFilter(filter_configs.get('duplicate_filter', {})))
        
        # 频率限制过滤器
        if filter_configs.get('frequency_filter', {}).get('enabled', True):
            self.add_filter(FrequencyLimitFilter(filter_configs.get('frequency_filter', {})))
        
        # 信号强度过滤器
        if filter_configs.get('strength_filter', {}).get('enabled', True):
            self.add_filter(StrengthFilter(filter_configs.get('strength_filter', {})))
        
        # 价格范围过滤器
        if filter_configs.get('price_range_filter', {}).get('enabled', True):
            self.add_filter(PriceRangeFilter(filter_configs.get('price_range_filter', {})))
    
    def add_filter(self, signal_filter: SignalFilter):
        """添加信号过滤器
        
        Args:
            signal_filter: 信号过滤器
        """
        self.filters.append(signal_filter)
        self.logger.info(f"添加过滤器: {signal_filter.name}")
    
    def remove_filter(self, filter_name: str) -> bool:
        """移除信号过滤器
        
        Args:
            filter_name: 过滤器名称
            
        Returns:
            bool: 移除是否成功
        """
        for i, signal_filter in enumerate(self.filters):
            if signal_filter.name == filter_name:
                del self.filters[i]
                self.logger.info(f"移除过滤器: {filter_name}")
                return True
        return False
    
    def add_router(self, router: SignalRouter):
        """添加信号路由器
        
        Args:
            router: 信号路由器
        """
        self.routers.append(router)
        self.logger.info(f"添加路由器: {router.name}")
    
    def add_executor(self, signal_type: SignalType, executor: Callable[[StrategySignal], Any]):
        """添加信号执行器
        
        Args:
            signal_type: 信号类型
            executor: 执行器函数
        """
        self.executors[signal_type] = executor
        self.logger.info(f"添加执行器: {signal_type.value}")
    
    def submit_signal(self, signal: StrategySignal) -> bool:
        """提交信号
        
        Args:
            signal: 策略信号
            
        Returns:
            bool: 提交是否成功
        """
        try:
            if not self.is_running:
                self.logger.warning("信号管理器未运行")
                return False
            
            # 创建信号记录
            record = SignalRecord(signal)
            self.signal_records[signal.signal_id] = record
            
            # 限制记录数量
            if len(self.signal_records) > self.max_records:
                # 删除最旧的记录
                oldest_id = min(self.signal_records.keys(), 
                               key=lambda x: self.signal_records[x].created_at)
                del self.signal_records[oldest_id]
            
            # 将信号放入队列
            try:
                self.signal_queue.put_nowait(signal)
                self.stats['total_received'] += 1
                self.stats['last_signal_time'] = datetime.now()
                return True
            except:
                # 队列满时拒绝信号
                record.update_status(SignalStatus.REJECTED, "信号队列已满")
                self.stats['total_rejected'] += 1
                return False
            
        except Exception as e:
            self.logger.error(f"提交信号失败: {e}")
            self.stats['error_count'] += 1
            return False
    
    def start(self) -> bool:
        """启动信号管理器
        
        Returns:
            bool: 启动是否成功
        """
        try:
            if self.is_running:
                self.logger.warning("信号管理器已在运行中")
                return True
            
            self.is_running = True
            
            # 启动处理线程
            self.processing_thread = threading.Thread(target=self._process_signals, daemon=True)
            self.processing_thread.start()
            
            self.logger.info("信号管理器启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"启动信号管理器失败: {e}")
            self.is_running = False
            return False
    
    def stop(self) -> bool:
        """停止信号管理器
        
        Returns:
            bool: 停止是否成功
        """
        try:
            if not self.is_running:
                self.logger.warning("信号管理器未在运行中")
                return True
            
            self.is_running = False
            
            # 等待处理线程结束
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            self.logger.info("信号管理器停止成功")
            return True
            
        except Exception as e:
            self.logger.error(f"停止信号管理器失败: {e}")
            return False
    
    def _process_signals(self):
        """处理信号的主循环"""
        while self.is_running:
            try:
                # 从队列获取信号
                signal = self.signal_queue.get(timeout=1.0)
                
                start_time = time.time()
                
                # 处理信号
                self._process_single_signal(signal)
                
                # 更新统计
                processing_time = time.time() - start_time
                self.stats['processing_time'] += processing_time
                self.stats['total_processed'] += 1
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"处理信号时发生错误: {e}")
                self.stats['error_count'] += 1
    
    def _process_single_signal(self, signal: StrategySignal):
        """处理单个信号
        
        Args:
            signal: 策略信号
        """
        record = self.signal_records.get(signal.signal_id)
        if not record:
            return
        
        try:
            # 验证信号
            if not self._validate_signal(signal, record):
                return
            
            # 过滤信号
            if not self._filter_signal(signal, record):
                return
            
            # 路由信号
            self._route_signal(signal, record)
            
            # 执行信号
            self._execute_signal(signal, record)
            
        except Exception as e:
            self.logger.error(f"处理信号 {signal.signal_id} 时发生错误: {e}")
            record.update_status(SignalStatus.ERROR, str(e))
            self.stats['error_count'] += 1
    
    def _validate_signal(self, signal: StrategySignal, record: SignalRecord) -> bool:
        """验证信号
        
        Args:
            signal: 策略信号
            record: 信号记录
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 基本验证
            if not signal.symbol:
                record.update_status(SignalStatus.REJECTED, "交易对符号为空")
                return False
            
            if signal.price is not None and signal.price <= 0:
                record.update_status(SignalStatus.REJECTED, "价格必须大于0")
                return False
            
            if signal.quantity is not None and signal.quantity <= 0:
                record.update_status(SignalStatus.REJECTED, "数量必须大于0")
                return False
            
            # 检查信号是否过期
            signal_age = (datetime.now() - signal.timestamp).total_seconds()
            max_age = self.config.get('max_signal_age_seconds', 300)  # 5分钟
            if signal_age > max_age:
                record.update_status(SignalStatus.EXPIRED, f"信号已过期 {signal_age} 秒")
                return False
            
            record.update_status(SignalStatus.VALIDATED)
            return True
            
        except Exception as e:
            record.update_status(SignalStatus.ERROR, f"验证失败: {e}")
            return False
    
    def _filter_signal(self, signal: StrategySignal, record: SignalRecord) -> bool:
        """过滤信号
        
        Args:
            signal: 策略信号
            record: 信号记录
            
        Returns:
            bool: 过滤是否通过
        """
        try:
            for signal_filter in self.filters:
                passed, reason = signal_filter.filter(signal)
                if not passed:
                    record.update_status(SignalStatus.REJECTED, f"{signal_filter.name}: {reason}")
                    self.stats['total_rejected'] += 1
                    
                    # 调用拒绝回调
                    for callback in self.on_signal_rejected_callbacks:
                        try:
                            callback(signal, reason)
                        except Exception as e:
                            self.logger.error(f"拒绝回调函数执行失败: {e}")
                    
                    return False
            
            record.update_status(SignalStatus.FILTERED)
            return True
            
        except Exception as e:
            record.update_status(SignalStatus.ERROR, f"过滤失败: {e}")
            return False
    
    def _route_signal(self, signal: StrategySignal, record: SignalRecord):
        """路由信号
        
        Args:
            signal: 策略信号
            record: 信号记录
        """
        try:
            matched_routes = []
            
            for router in self.routers:
                routes = router.route_signal(signal)
                matched_routes.extend(routes)
            
            record.update_status(SignalStatus.ROUTED, metadata={'matched_routes': matched_routes})
            
        except Exception as e:
            record.update_status(SignalStatus.ERROR, f"路由失败: {e}")
    
    def _execute_signal(self, signal: StrategySignal, record: SignalRecord):
        """执行信号
        
        Args:
            signal: 策略信号
            record: 信号记录
        """
        try:
            # 查找执行器
            executor = self.executors.get(signal.signal_type)
            if not executor:
                record.update_status(SignalStatus.REJECTED, f"未找到 {signal.signal_type.value} 类型的执行器")
                return
            
            # 执行信号
            result = executor(signal)
            
            # 记录执行结果
            record.execution_result = result
            record.update_status(SignalStatus.EXECUTED, metadata={'execution_result': result})
            
            self.stats['total_executed'] += 1
            
            # 调用执行回调
            for callback in self.on_signal_executed_callbacks:
                try:
                    callback(signal, result)
                except Exception as e:
                    self.logger.error(f"执行回调函数执行失败: {e}")
            
            self.logger.info(f"信号执行成功: {signal}")
            
        except Exception as e:
            record.update_status(SignalStatus.ERROR, f"执行失败: {e}")
            self.logger.error(f"信号执行失败: {e}")
        
        finally:
            # 调用处理完成回调
            for callback in self.on_signal_processed_callbacks:
                try:
                    callback(signal, record)
                except Exception as e:
                    self.logger.error(f"处理回调函数执行失败: {e}")
    
    def get_signal_record(self, signal_id: str) -> Optional[SignalRecord]:
        """获取信号记录
        
        Args:
            signal_id: 信号ID
            
        Returns:
            Optional[SignalRecord]: 信号记录
        """
        return self.signal_records.get(signal_id)
    
    def get_signals_by_status(self, status: SignalStatus) -> List[SignalRecord]:
        """根据状态获取信号
        
        Args:
            status: 信号状态
            
        Returns:
            List[SignalRecord]: 信号记录列表
        """
        return [record for record in self.signal_records.values() if record.status == status]
    
    def get_signals_by_symbol(self, symbol: str) -> List[SignalRecord]:
        """根据交易对获取信号
        
        Args:
            symbol: 交易对符号
            
        Returns:
            List[SignalRecord]: 信号记录列表
        """
        return [record for record in self.signal_records.values() if record.signal.symbol == symbol]
    
    def get_recent_signals(self, minutes: int = 60) -> List[SignalRecord]:
        """获取最近的信号
        
        Args:
            minutes: 时间范围（分钟）
            
        Returns:
            List[SignalRecord]: 信号记录列表
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            record for record in self.signal_records.values()
            if record.created_at > cutoff_time
        ]
    
    def add_signal_processed_callback(self, callback: Callable[[StrategySignal, SignalRecord], None]):
        """添加信号处理完成回调
        
        Args:
            callback: 回调函数
        """
        self.on_signal_processed_callbacks.append(callback)
    
    def add_signal_executed_callback(self, callback: Callable[[StrategySignal, Any], None]):
        """添加信号执行回调
        
        Args:
            callback: 回调函数
        """
        self.on_signal_executed_callbacks.append(callback)
    
    def add_signal_rejected_callback(self, callback: Callable[[StrategySignal, str], None]):
        """添加信号拒绝回调
        
        Args:
            callback: 回调函数
        """
        self.on_signal_rejected_callbacks.append(callback)
    
    def get_stats(self) -> Dict:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = self.stats.copy()
        
        # 添加状态统计
        status_counts = defaultdict(int)
        for record in self.signal_records.values():
            status_counts[record.status.value] += 1
        
        stats.update({
            'is_running': self.is_running,
            'queue_size': self.signal_queue.qsize(),
            'total_records': len(self.signal_records),
            'status_counts': dict(status_counts),
            'filter_stats': [f.get_stats() for f in self.filters],
            'router_stats': [r.get_stats() for r in self.routers]
        })
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_received': 0,
            'total_processed': 0,
            'total_executed': 0,
            'total_rejected': 0,
            'processing_time': 0.0,
            'last_signal_time': None,
            'error_count': 0
        }
        
        # 重置过滤器统计
        for signal_filter in self.filters:
            signal_filter.stats = {
                'total_processed': 0,
                'total_passed': 0,
                'total_rejected': 0,
                'last_process_time': None
            }
        
        # 重置路由器统计
        for router in self.routers:
            router.stats = {
                'total_routed': 0,
                'routes_used': defaultdict(int),
                'last_route_time': None
            }
        
        self.logger.info("信号管理器统计信息已重置")
    
    def clear_old_records(self, hours: int = 24):
        """清理旧的信号记录
        
        Args:
            hours: 保留时间（小时）
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        old_records = [
            signal_id for signal_id, record in self.signal_records.items()
            if record.created_at < cutoff_time
        ]
        
        for signal_id in old_records:
            del self.signal_records[signal_id]
        
        self.logger.info(f"清理了 {len(old_records)} 条旧记录")
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 管理器字典
        """
        return {
            'config': self.config,
            'stats': self.get_stats(),
            'filters': [f.name for f in self.filters],
            'routers': [r.name for r in self.routers],
            'executors': list(self.executors.keys())
        }
    
    def __str__(self) -> str:
        return f"SignalManager(running={self.is_running}, records={len(self.signal_records)})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()