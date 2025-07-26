"""Stream Processor Module

流处理器，用于实时数据流的处理、分析和转发。

主要功能：
- 实时数据流处理
- 数据流转换
- 事件驱动处理
- 流式聚合
- 数据路由
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator
from datetime import datetime, timedelta
import time
import threading
from queue import Queue, Empty
from collections import deque, defaultdict
import json

from .base import BaseProcessor, ProcessorType, ProcessingResult
from ...common.exceptions.data import DataProcessingError


class StreamWindow:
    """流窗口类"""
    
    def __init__(self, 
                 window_type: str = 'time',
                 size: Union[int, timedelta] = 100,
                 slide: Optional[Union[int, timedelta]] = None):
        """初始化流窗口
        
        Args:
            window_type: 窗口类型 ('time', 'count', 'session')
            size: 窗口大小
            slide: 滑动间隔
        """
        self.window_type = window_type
        self.size = size
        self.slide = slide or size
        self.data = deque()
        self.timestamps = deque()
        self.last_emit = None
    
    def add_data(self, data: Any, timestamp: datetime = None) -> bool:
        """添加数据到窗口
        
        Args:
            data: 数据
            timestamp: 时间戳
            
        Returns:
            bool: 是否触发窗口输出
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.data.append(data)
        self.timestamps.append(timestamp)
        
        # 清理过期数据
        self._cleanup_expired_data(timestamp)
        
        # 检查是否需要触发输出
        return self._should_emit(timestamp)
    
    def _cleanup_expired_data(self, current_time: datetime):
        """清理过期数据
        
        Args:
            current_time: 当前时间
        """
        if self.window_type == 'time':
            cutoff_time = current_time - self.size
            while self.timestamps and self.timestamps[0] < cutoff_time:
                self.data.popleft()
                self.timestamps.popleft()
        elif self.window_type == 'count':
            while len(self.data) > self.size:
                self.data.popleft()
                self.timestamps.popleft()
    
    def _should_emit(self, current_time: datetime) -> bool:
        """检查是否应该触发输出
        
        Args:
            current_time: 当前时间
            
        Returns:
            bool: 是否应该触发
        """
        if not self.data:
            return False
        
        if self.last_emit is None:
            self.last_emit = current_time
            return True
        
        if self.window_type == 'time':
            return current_time - self.last_emit >= self.slide
        elif self.window_type == 'count':
            return len(self.data) >= self.slide
        
        return False
    
    def get_window_data(self) -> List[Any]:
        """获取窗口数据
        
        Returns:
            List[Any]: 窗口内的数据
        """
        return list(self.data)
    
    def reset(self):
        """重置窗口"""
        self.data.clear()
        self.timestamps.clear()
        self.last_emit = None


class StreamAggregator:
    """流聚合器"""
    
    def __init__(self, aggregation_functions: Dict[str, Callable]):
        """初始化流聚合器
        
        Args:
            aggregation_functions: 聚合函数字典
        """
        self.aggregation_functions = aggregation_functions
        self.results = {}
    
    def aggregate(self, data: List[Any]) -> Dict[str, Any]:
        """执行聚合
        
        Args:
            data: 数据列表
            
        Returns:
            Dict[str, Any]: 聚合结果
        """
        if not data:
            return {}
        
        # 转换为DataFrame进行聚合
        if isinstance(data[0], dict):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({'value': data})
        
        results = {}
        for name, func in self.aggregation_functions.items():
            try:
                if callable(func):
                    results[name] = func(df)
                else:
                    # 预定义的聚合函数
                    if func == 'mean':
                        results[name] = df.select_dtypes(include=[np.number]).mean().to_dict()
                    elif func == 'sum':
                        results[name] = df.select_dtypes(include=[np.number]).sum().to_dict()
                    elif func == 'count':
                        results[name] = len(df)
                    elif func == 'min':
                        results[name] = df.select_dtypes(include=[np.number]).min().to_dict()
                    elif func == 'max':
                        results[name] = df.select_dtypes(include=[np.number]).max().to_dict()
                    elif func == 'std':
                        results[name] = df.select_dtypes(include=[np.number]).std().to_dict()
            except Exception as e:
                results[name] = f"聚合错误: {str(e)}"
        
        return results


class StreamProcessor(BaseProcessor):
    """流处理器
    
    提供实时数据流处理功能，支持窗口操作、流式聚合、
    事件驱动处理和数据路由。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化流处理器
        
        Args:
            config: 流处理配置
                - buffer_size: 缓冲区大小
                - window_config: 窗口配置
                - aggregation_config: 聚合配置
                - routing_rules: 路由规则
                - event_handlers: 事件处理器
        """
        super().__init__("StreamProcessor", ProcessorType.STREAM, config)
        
        # 流处理配置
        self.buffer_size = self.config.get('buffer_size', 1000)
        self.window_config = self.config.get('window_config', {})
        self.aggregation_config = self.config.get('aggregation_config', {})
        self.routing_rules = self.config.get('routing_rules', [])
        
        # 数据缓冲区
        self.data_buffer = Queue(maxsize=self.buffer_size)
        self.output_buffer = Queue(maxsize=self.buffer_size)
        
        # 流窗口
        self.windows = {}
        self._setup_windows()
        
        # 流聚合器
        self.aggregators = {}
        self._setup_aggregators()
        
        # 事件处理器
        self.event_handlers = self.config.get('event_handlers', {})
        
        # 处理线程
        self.processing_thread = None
        self.is_running = False
        
        # 流统计
        self.stream_stats = {
            'total_processed': 0,
            'total_output': 0,
            'processing_rate': 0.0,
            'buffer_usage': 0.0,
            'window_triggers': 0,
            'aggregation_count': 0
        }
        
        # 性能监控
        self.performance_window = deque(maxlen=100)
        self.last_stats_update = time.time()
    
    def initialize(self) -> bool:
        """初始化流处理器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info(f"初始化流处理器: {self.name}")
            self.logger.info(f"缓冲区大小: {self.buffer_size}")
            self.logger.info(f"窗口数量: {len(self.windows)}")
            self.logger.info(f"聚合器数量: {len(self.aggregators)}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"初始化流处理器失败: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 数据是否有效
        """
        return data is not None
    
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """处理流数据
        
        Args:
            data: 流数据
            **kwargs: 额外参数
                - async_mode: 是否异步处理
                - window_name: 指定窗口
                - aggregator_name: 指定聚合器
                - route_to: 路由目标
                
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            if not self.validate_input(data):
                return ProcessingResult(
                    data=None,
                    success=False,
                    message="输入数据为空"
                )
            
            async_mode = kwargs.get('async_mode', False)
            
            if async_mode:
                # 异步处理模式
                return self._process_async(data, **kwargs)
            else:
                # 同步处理模式
                return self._process_sync(data, **kwargs)
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"流处理失败: {str(e)}"
            self.logger.error(error_msg)
            
            result = ProcessingResult(
                data=None,
                success=False,
                message=error_msg,
                processing_time=processing_time
            )
            
            self._update_stats(result)
            return result
    
    def _process_sync(self, data: Any, **kwargs) -> ProcessingResult:
        """同步处理数据
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        window_name = kwargs.get('window_name')
        aggregator_name = kwargs.get('aggregator_name')
        route_to = kwargs.get('route_to')
        
        results = []
        
        # 窗口处理
        if window_name and window_name in self.windows:
            window = self.windows[window_name]
            should_emit = window.add_data(data)
            
            if should_emit:
                window_data = window.get_window_data()
                
                # 执行聚合
                if aggregator_name and aggregator_name in self.aggregators:
                    aggregator = self.aggregators[aggregator_name]
                    aggregated_result = aggregator.aggregate(window_data)
                    results.append({
                        'type': 'aggregation',
                        'window': window_name,
                        'aggregator': aggregator_name,
                        'result': aggregated_result,
                        'data_count': len(window_data)
                    })
                    self.stream_stats['aggregation_count'] += 1
                else:
                    results.append({
                        'type': 'window',
                        'window': window_name,
                        'data': window_data,
                        'data_count': len(window_data)
                    })
                
                window.last_emit = datetime.now()
                self.stream_stats['window_triggers'] += 1
        
        # 路由处理
        if route_to:
            routed_data = self._apply_routing(data, route_to)
            if routed_data:
                results.append({
                    'type': 'routing',
                    'route': route_to,
                    'data': routed_data
                })
        
        # 事件处理
        event_results = self._handle_events(data)
        if event_results:
            results.extend(event_results)
        
        # 如果没有特殊处理，直接返回原数据
        if not results:
            results = [{'type': 'passthrough', 'data': data}]
        
        processing_time = time.time() - start_time
        self.stream_stats['total_processed'] += 1
        self.stream_stats['total_output'] += len(results)
        
        # 更新性能统计
        self._update_performance_stats(processing_time)
        
        result = ProcessingResult(
            data=results,
            success=True,
            message=f"流处理完成，生成 {len(results)} 个结果",
            metadata={
                'results_count': len(results),
                'window_triggered': any(r['type'] in ['window', 'aggregation'] for r in results),
                'routed': any(r['type'] == 'routing' for r in results)
            },
            processing_time=processing_time
        )
        
        self._update_stats(result)
        return result
    
    def _process_async(self, data: Any, **kwargs) -> ProcessingResult:
        """异步处理数据
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            # 将数据放入缓冲区
            self.data_buffer.put_nowait({
                'data': data,
                'timestamp': datetime.now(),
                'kwargs': kwargs
            })
            
            # 启动处理线程（如果未运行）
            if not self.is_running:
                self.start_processing()
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                data={'status': 'queued', 'buffer_size': self.data_buffer.qsize()},
                success=True,
                message="数据已加入处理队列",
                processing_time=processing_time
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"异步处理失败: {str(e)}"
            
            result = ProcessingResult(
                data=None,
                success=False,
                message=error_msg,
                processing_time=processing_time
            )
            
            self._update_stats(result)
            return result
    
    def _setup_windows(self):
        """设置流窗口"""
        for window_name, window_config in self.window_config.items():
            window_type = window_config.get('type', 'time')
            size = window_config.get('size', 100)
            slide = window_config.get('slide')
            
            # 处理时间窗口的大小格式
            if window_type == 'time' and isinstance(size, str):
                size = self._parse_time_duration(size)
            if window_type == 'time' and isinstance(slide, str):
                slide = self._parse_time_duration(slide)
            
            self.windows[window_name] = StreamWindow(
                window_type=window_type,
                size=size,
                slide=slide
            )
            
            self.logger.info(f"创建流窗口: {window_name} ({window_type}, {size})")
    
    def _setup_aggregators(self):
        """设置流聚合器"""
        for agg_name, agg_config in self.aggregation_config.items():
            functions = agg_config.get('functions', {})
            self.aggregators[agg_name] = StreamAggregator(functions)
            self.logger.info(f"创建流聚合器: {agg_name}")
    
    def _parse_time_duration(self, duration_str: str) -> timedelta:
        """解析时间持续时间字符串
        
        Args:
            duration_str: 时间字符串，如 '1m', '30s', '1h'
            
        Returns:
            timedelta: 时间间隔
        """
        import re
        
        pattern = r'(\d+)([smhd])'
        match = re.match(pattern, duration_str.lower())
        
        if not match:
            raise ValueError(f"无效的时间格式: {duration_str}")
        
        value, unit = match.groups()
        value = int(value)
        
        if unit == 's':
            return timedelta(seconds=value)
        elif unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        else:
            raise ValueError(f"不支持的时间单位: {unit}")
    
    def _apply_routing(self, data: Any, route_to: str) -> Optional[Any]:
        """应用路由规则
        
        Args:
            data: 数据
            route_to: 路由目标
            
        Returns:
            Optional[Any]: 路由后的数据
        """
        for rule in self.routing_rules:
            if rule.get('name') == route_to:
                condition = rule.get('condition')
                transform = rule.get('transform')
                
                # 检查条件
                if condition and not self._evaluate_condition(data, condition):
                    continue
                
                # 应用转换
                if transform:
                    return self._apply_transform(data, transform)
                else:
                    return data
        
        return None
    
    def _evaluate_condition(self, data: Any, condition: Dict) -> bool:
        """评估路由条件
        
        Args:
            data: 数据
            condition: 条件配置
            
        Returns:
            bool: 条件是否满足
        """
        # 简单的条件评估实现
        # 可以根据需要扩展更复杂的条件逻辑
        try:
            if isinstance(data, dict):
                field = condition.get('field')
                operator = condition.get('operator')
                value = condition.get('value')
                
                if field in data:
                    data_value = data[field]
                    
                    if operator == 'eq':
                        return data_value == value
                    elif operator == 'gt':
                        return data_value > value
                    elif operator == 'lt':
                        return data_value < value
                    elif operator == 'gte':
                        return data_value >= value
                    elif operator == 'lte':
                        return data_value <= value
                    elif operator == 'in':
                        return data_value in value
            
            return True
            
        except Exception:
            return False
    
    def _apply_transform(self, data: Any, transform: Dict) -> Any:
        """应用数据转换
        
        Args:
            data: 原始数据
            transform: 转换配置
            
        Returns:
            Any: 转换后的数据
        """
        transform_type = transform.get('type')
        
        if transform_type == 'filter':
            # 字段过滤
            fields = transform.get('fields', [])
            if isinstance(data, dict) and fields:
                return {k: v for k, v in data.items() if k in fields}
        
        elif transform_type == 'map':
            # 字段映射
            mapping = transform.get('mapping', {})
            if isinstance(data, dict) and mapping:
                return {mapping.get(k, k): v for k, v in data.items()}
        
        elif transform_type == 'aggregate':
            # 简单聚合
            if isinstance(data, list):
                agg_func = transform.get('function', 'sum')
                if agg_func == 'sum':
                    return sum(data)
                elif agg_func == 'mean':
                    return sum(data) / len(data)
                elif agg_func == 'count':
                    return len(data)
        
        return data
    
    def _handle_events(self, data: Any) -> List[Dict]:
        """处理事件
        
        Args:
            data: 数据
            
        Returns:
            List[Dict]: 事件处理结果
        """
        results = []
        
        for event_name, handler_config in self.event_handlers.items():
            try:
                trigger = handler_config.get('trigger')
                action = handler_config.get('action')
                
                # 检查触发条件
                if self._check_event_trigger(data, trigger):
                    # 执行动作
                    action_result = self._execute_event_action(data, action)
                    if action_result:
                        results.append({
                            'type': 'event',
                            'event': event_name,
                            'result': action_result
                        })
            
            except Exception as e:
                self.logger.error(f"处理事件 {event_name} 时发生错误: {e}")
        
        return results
    
    def _check_event_trigger(self, data: Any, trigger: Dict) -> bool:
        """检查事件触发条件
        
        Args:
            data: 数据
            trigger: 触发条件
            
        Returns:
            bool: 是否触发
        """
        if not trigger:
            return True
        
        # 实现触发条件检查逻辑
        return self._evaluate_condition(data, trigger)
    
    def _execute_event_action(self, data: Any, action: Dict) -> Any:
        """执行事件动作
        
        Args:
            data: 数据
            action: 动作配置
            
        Returns:
            Any: 动作结果
        """
        action_type = action.get('type')
        
        if action_type == 'log':
            message = action.get('message', 'Event triggered')
            self.logger.info(f"事件动作: {message}")
            return {'logged': message}
        
        elif action_type == 'alert':
            alert_message = action.get('message', 'Alert triggered')
            # 这里可以集成告警系统
            self.logger.warning(f"告警: {alert_message}")
            return {'alert': alert_message}
        
        elif action_type == 'callback':
            callback_func = action.get('function')
            if callable(callback_func):
                return callback_func(data)
        
        return None
    
    def _update_performance_stats(self, processing_time: float):
        """更新性能统计
        
        Args:
            processing_time: 处理时间
        """
        self.performance_window.append(processing_time)
        
        current_time = time.time()
        if current_time - self.last_stats_update >= 1.0:  # 每秒更新一次
            # 计算处理速率
            if self.performance_window:
                avg_processing_time = sum(self.performance_window) / len(self.performance_window)
                self.stream_stats['processing_rate'] = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            
            # 计算缓冲区使用率
            self.stream_stats['buffer_usage'] = self.data_buffer.qsize() / self.buffer_size
            
            self.last_stats_update = current_time
    
    def start_processing(self) -> bool:
        """启动流处理线程
        
        Returns:
            bool: 是否成功启动
        """
        if self.is_running:
            return True
        
        try:
            self.is_running = True
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            self.logger.info("流处理线程已启动")
            return True
            
        except Exception as e:
            self.logger.error(f"启动流处理线程失败: {e}")
            self.is_running = False
            return False
    
    def stop_processing(self) -> bool:
        """停止流处理线程
        
        Returns:
            bool: 是否成功停止
        """
        if not self.is_running:
            return True
        
        try:
            self.is_running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            self.logger.info("流处理线程已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止流处理线程失败: {e}")
            return False
    
    def _processing_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                # 从缓冲区获取数据
                item = self.data_buffer.get(timeout=1.0)
                
                # 处理数据
                result = self._process_sync(
                    item['data'],
                    **item['kwargs']
                )
                
                # 将结果放入输出缓冲区
                if result.success:
                    try:
                        self.output_buffer.put_nowait({
                            'result': result,
                            'timestamp': datetime.now()
                        })
                    except:
                        # 输出缓冲区满，丢弃最旧的结果
                        try:
                            self.output_buffer.get_nowait()
                            self.output_buffer.put_nowait({
                                'result': result,
                                'timestamp': datetime.now()
                            })
                        except:
                            pass
                
                self.data_buffer.task_done()
                
            except Empty:
                # 超时，继续循环
                continue
            except Exception as e:
                self.logger.error(f"处理循环中发生错误: {e}")
    
    def get_output(self, timeout: float = 1.0) -> Optional[ProcessingResult]:
        """获取处理输出
        
        Args:
            timeout: 超时时间
            
        Returns:
            Optional[ProcessingResult]: 处理结果
        """
        try:
            item = self.output_buffer.get(timeout=timeout)
            return item['result']
        except Empty:
            return None
    
    def get_stream_stats(self) -> Dict:
        """获取流处理统计信息
        
        Returns:
            Dict: 流处理统计信息
        """
        stats = self.stream_stats.copy()
        stats.update({
            'is_running': self.is_running,
            'buffer_size': self.data_buffer.qsize(),
            'output_buffer_size': self.output_buffer.qsize(),
            'windows_count': len(self.windows),
            'aggregators_count': len(self.aggregators)
        })
        return stats
    
    def reset_stream_stats(self) -> None:
        """重置流处理统计信息"""
        self.stream_stats = {
            'total_processed': 0,
            'total_output': 0,
            'processing_rate': 0.0,
            'buffer_usage': 0.0,
            'window_triggers': 0,
            'aggregation_count': 0
        }
        self.performance_window.clear()
        self.logger.info("流处理统计信息已重置")
    
    def add_window(self, name: str, window_config: Dict) -> bool:
        """添加流窗口
        
        Args:
            name: 窗口名称
            window_config: 窗口配置
            
        Returns:
            bool: 是否成功添加
        """
        try:
            window_type = window_config.get('type', 'time')
            size = window_config.get('size', 100)
            slide = window_config.get('slide')
            
            if window_type == 'time' and isinstance(size, str):
                size = self._parse_time_duration(size)
            if window_type == 'time' and isinstance(slide, str):
                slide = self._parse_time_duration(slide)
            
            self.windows[name] = StreamWindow(
                window_type=window_type,
                size=size,
                slide=slide
            )
            
            self.logger.info(f"添加流窗口: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加流窗口失败: {e}")
            return False
    
    def remove_window(self, name: str) -> bool:
        """移除流窗口
        
        Args:
            name: 窗口名称
            
        Returns:
            bool: 是否成功移除
        """
        if name in self.windows:
            del self.windows[name]
            self.logger.info(f"移除流窗口: {name}")
            return True
        return False
    
    def add_aggregator(self, name: str, functions: Dict[str, Callable]) -> bool:
        """添加流聚合器
        
        Args:
            name: 聚合器名称
            functions: 聚合函数
            
        Returns:
            bool: 是否成功添加
        """
        try:
            self.aggregators[name] = StreamAggregator(functions)
            self.logger.info(f"添加流聚合器: {name}")
            return True
        except Exception as e:
            self.logger.error(f"添加流聚合器失败: {e}")
            return False
    
    def remove_aggregator(self, name: str) -> bool:
        """移除流聚合器
        
        Args:
            name: 聚合器名称
            
        Returns:
            bool: 是否成功移除
        """
        if name in self.aggregators:
            del self.aggregators[name]
            self.logger.info(f"移除流聚合器: {name}")
            return True
        return False