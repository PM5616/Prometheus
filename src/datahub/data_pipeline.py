"""Data Pipeline Module

数据管道，用于构建复杂的数据处理流水线。

功能特性：
- 流水线构建
- 阶段管理
- 并行处理
- 错误处理
- 监控统计
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from src.common.models.market import Ticker, Kline, OrderBook
from src.common.exceptions.data import DataProcessingError, DataPipelineError
from .processors.base import BaseProcessor
from .storage.base import BaseStorage
from .data_router import DataRouter


class StageType(Enum):
    """阶段类型枚举"""
    INPUT = "input"  # 输入阶段
    PROCESS = "process"  # 处理阶段
    TRANSFORM = "transform"  # 转换阶段
    VALIDATE = "validate"  # 验证阶段
    FILTER = "filter"  # 过滤阶段
    AGGREGATE = "aggregate"  # 聚合阶段
    OUTPUT = "output"  # 输出阶段
    BRANCH = "branch"  # 分支阶段
    MERGE = "merge"  # 合并阶段
    CUSTOM = "custom"  # 自定义阶段


class ExecutionMode(Enum):
    """执行模式枚举"""
    SEQUENTIAL = "sequential"  # 顺序执行
    PARALLEL = "parallel"  # 并行执行
    ASYNC = "async"  # 异步执行
    BATCH = "batch"  # 批处理
    STREAM = "stream"  # 流处理


@dataclass
class StageConfig:
    """阶段配置"""
    name: str
    stage_type: StageType
    processor: Union[BaseProcessor, Callable, str]
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enabled: bool = True
    parallel_workers: int = 1
    batch_size: int = 100
    buffer_size: int = 1000
    dependencies: List[str] = field(default_factory=list)
    conditions: List[Callable] = field(default_factory=list)
    error_handling: str = "raise"  # raise, skip, retry, fallback
    fallback_processor: Optional[Callable] = None
    metrics_enabled: bool = True
    
    def __post_init__(self):
        if isinstance(self.stage_type, str):
            self.stage_type = StageType(self.stage_type)
        if isinstance(self.execution_mode, str):
            self.execution_mode = ExecutionMode(self.execution_mode)


@dataclass
class StageResult:
    """阶段执行结果"""
    stage_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    processed_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineStage:
    """流水线阶段"""
    
    def __init__(self, config: StageConfig):
        """
        初始化阶段
        
        Args:
            config: 阶段配置
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        
        # 统计信息
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0,
            'total_processed': 0,
            'last_execution': None,
            'error_count': 0,
            'retry_count': 0
        }
        
        # 运行时状态
        self.is_running = False
        self.current_load = 0
        self.max_load = config.parallel_workers
        
        # 线程池（如果需要并行执行）
        self.executor = None
        if config.execution_mode == ExecutionMode.PARALLEL:
            self.executor = ThreadPoolExecutor(
                max_workers=config.parallel_workers,
                thread_name_prefix=f"stage_{config.name}"
            )
        
        # 数据缓冲区（用于批处理和流处理）
        self.buffer = deque(maxlen=config.buffer_size)
        self.buffer_lock = threading.Lock()
        
        self.logger.info(f"阶段 {config.name} 初始化完成")
    
    def execute(self, data: Any, context: Optional[Dict] = None) -> StageResult:
        """
        执行阶段
        
        Args:
            data: 输入数据
            context: 执行上下文
            
        Returns:
            StageResult: 执行结果
        """
        if not self.config.enabled:
            return StageResult(
                stage_name=self.config.name,
                success=True,
                data=data,
                metadata={'skipped': True}
            )
        
        # 检查条件
        if not self._check_conditions(data, context):
            return StageResult(
                stage_name=self.config.name,
                success=True,
                data=data,
                metadata={'condition_failed': True}
            )
        
        start_time = datetime.now()
        execution_start = time.time()
        
        try:
            self.is_running = True
            self.current_load += 1
            
            # 根据执行模式执行
            if self.config.execution_mode == ExecutionMode.SEQUENTIAL:
                result_data = self._execute_sequential(data, context)
            elif self.config.execution_mode == ExecutionMode.PARALLEL:
                result_data = self._execute_parallel(data, context)
            elif self.config.execution_mode == ExecutionMode.ASYNC:
                result_data = self._execute_async(data, context)
            elif self.config.execution_mode == ExecutionMode.BATCH:
                result_data = self._execute_batch(data, context)
            elif self.config.execution_mode == ExecutionMode.STREAM:
                result_data = self._execute_stream(data, context)
            else:
                result_data = self._execute_sequential(data, context)
            
            execution_time = time.time() - execution_start
            end_time = datetime.now()
            
            # 更新统计信息
            self._update_stats(True, execution_time, self._get_data_count(result_data))
            
            return StageResult(
                stage_name=self.config.name,
                success=True,
                data=result_data,
                execution_time=execution_time,
                processed_count=self._get_data_count(result_data),
                start_time=start_time,
                end_time=end_time
            )
            
        except Exception as e:
            execution_time = time.time() - execution_start
            end_time = datetime.now()
            
            # 错误处理
            error_result = self._handle_error(e, data, context, execution_time)
            
            # 更新统计信息
            self._update_stats(False, execution_time, 0)
            
            error_result.start_time = start_time
            error_result.end_time = end_time
            
            return error_result
        
        finally:
            self.is_running = False
            self.current_load = max(0, self.current_load - 1)
    
    def _check_conditions(self, data: Any, context: Optional[Dict]) -> bool:
        """
        检查执行条件
        
        Args:
            data: 数据
            context: 上下文
            
        Returns:
            bool: 是否满足条件
        """
        if not self.config.conditions:
            return True
        
        try:
            for condition in self.config.conditions:
                if not condition(data, context):
                    return False
            return True
        except Exception as e:
            self.logger.warning(f"条件检查失败: {e}")
            return False
    
    def _execute_sequential(self, data: Any, context: Optional[Dict]) -> Any:
        """
        顺序执行
        
        Args:
            data: 数据
            context: 上下文
            
        Returns:
            Any: 处理结果
        """
        return self._process_data(data, context)
    
    def _execute_parallel(self, data: Any, context: Optional[Dict]) -> Any:
        """
        并行执行
        
        Args:
            data: 数据
            context: 上下文
            
        Returns:
            Any: 处理结果
        """
        if not isinstance(data, (list, tuple)):
            return self._process_data(data, context)
        
        if not self.executor:
            return self._execute_sequential(data, context)
        
        # 并行处理数据列表
        futures = []
        for item in data:
            future = self.executor.submit(self._process_data, item, context)
            futures.append(future)
        
        results = []
        for future in as_completed(futures, timeout=self.config.timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"并行处理失败: {e}")
                if self.config.error_handling == "raise":
                    raise
                # 其他错误处理策略
        
        return results
    
    def _execute_async(self, data: Any, context: Optional[Dict]) -> Any:
        """
        异步执行
        
        Args:
            data: 数据
            context: 上下文
            
        Returns:
            Any: 处理结果
        """
        # 简化的异步执行，实际应该使用asyncio
        return self._process_data(data, context)
    
    def _execute_batch(self, data: Any, context: Optional[Dict]) -> Any:
        """
        批处理执行
        
        Args:
            data: 数据
            context: 上下文
            
        Returns:
            Any: 处理结果
        """
        # 将数据添加到缓冲区
        with self.buffer_lock:
            if isinstance(data, (list, tuple)):
                self.buffer.extend(data)
            else:
                self.buffer.append(data)
            
            # 检查是否达到批处理大小
            if len(self.buffer) >= self.config.batch_size:
                batch_data = list(self.buffer)
                self.buffer.clear()
                return self._process_data(batch_data, context)
        
        return None  # 未达到批处理大小
    
    def _execute_stream(self, data: Any, context: Optional[Dict]) -> Any:
        """
        流处理执行
        
        Args:
            data: 数据
            context: 上下文
            
        Returns:
            Any: 处理结果
        """
        # 流处理：逐个处理数据项
        return self._process_data(data, context)
    
    def _process_data(self, data: Any, context: Optional[Dict]) -> Any:
        """
        处理数据
        
        Args:
            data: 数据
            context: 上下文
            
        Returns:
            Any: 处理结果
        """
        processor = self.config.processor
        
        if isinstance(processor, BaseProcessor):
            return processor.process(data)
        elif callable(processor):
            if context:
                return processor(data, context)
            else:
                return processor(data)
        elif isinstance(processor, str):
            # 处理器名称，需要从注册表中获取
            raise NotImplementedError("处理器名称解析未实现")
        else:
            raise ValueError(f"不支持的处理器类型: {type(processor)}")
    
    def _handle_error(self, error: Exception, data: Any, context: Optional[Dict], execution_time: float) -> StageResult:
        """
        处理错误
        
        Args:
            error: 异常
            data: 数据
            context: 上下文
            execution_time: 执行时间
            
        Returns:
            StageResult: 错误结果
        """
        self.stats['error_count'] += 1
        error_msg = str(error)
        
        if self.config.error_handling == "raise":
            raise DataPipelineError(f"阶段 {self.config.name} 执行失败: {error_msg}")
        
        elif self.config.error_handling == "skip":
            self.logger.warning(f"阶段 {self.config.name} 跳过错误: {error_msg}")
            return StageResult(
                stage_name=self.config.name,
                success=False,
                data=data,  # 返回原始数据
                error=error_msg,
                execution_time=execution_time,
                metadata={'skipped_error': True}
            )
        
        elif self.config.error_handling == "retry":
            return self._retry_execution(data, context, error)
        
        elif self.config.error_handling == "fallback":
            return self._fallback_execution(data, context, error, execution_time)
        
        else:
            raise DataPipelineError(f"未知的错误处理策略: {self.config.error_handling}")
    
    def _retry_execution(self, data: Any, context: Optional[Dict], original_error: Exception) -> StageResult:
        """
        重试执行
        
        Args:
            data: 数据
            context: 上下文
            original_error: 原始错误
            
        Returns:
            StageResult: 重试结果
        """
        for attempt in range(self.config.retry_attempts):
            try:
                self.stats['retry_count'] += 1
                time.sleep(self.config.retry_delay * (attempt + 1))  # 指数退避
                
                start_time = time.time()
                result_data = self._process_data(data, context)
                execution_time = time.time() - start_time
                
                self.logger.info(f"阶段 {self.config.name} 重试成功，尝试次数: {attempt + 1}")
                
                return StageResult(
                    stage_name=self.config.name,
                    success=True,
                    data=result_data,
                    execution_time=execution_time,
                    processed_count=self._get_data_count(result_data),
                    metadata={'retry_attempt': attempt + 1}
                )
                
            except Exception as e:
                self.logger.warning(f"阶段 {self.config.name} 重试失败，尝试次数: {attempt + 1}, 错误: {e}")
                if attempt == self.config.retry_attempts - 1:
                    # 最后一次重试失败
                    return StageResult(
                        stage_name=self.config.name,
                        success=False,
                        error=f"重试失败: {str(original_error)}",
                        execution_time=0.0,
                        metadata={'retry_exhausted': True, 'retry_attempts': self.config.retry_attempts}
                    )
        
        # 不应该到达这里
        return StageResult(
            stage_name=self.config.name,
            success=False,
            error=str(original_error),
            execution_time=0.0
        )
    
    def _fallback_execution(self, data: Any, context: Optional[Dict], error: Exception, execution_time: float) -> StageResult:
        """
        回退执行
        
        Args:
            data: 数据
            context: 上下文
            error: 错误
            execution_time: 执行时间
            
        Returns:
            StageResult: 回退结果
        """
        if not self.config.fallback_processor:
            return StageResult(
                stage_name=self.config.name,
                success=False,
                error=str(error),
                execution_time=execution_time,
                metadata={'no_fallback': True}
            )
        
        try:
            self.logger.info(f"阶段 {self.config.name} 使用回退处理器")
            
            start_time = time.time()
            if context:
                result_data = self.config.fallback_processor(data, context)
            else:
                result_data = self.config.fallback_processor(data)
            fallback_time = time.time() - start_time
            
            return StageResult(
                stage_name=self.config.name,
                success=True,
                data=result_data,
                execution_time=execution_time + fallback_time,
                processed_count=self._get_data_count(result_data),
                metadata={'used_fallback': True, 'original_error': str(error)}
            )
            
        except Exception as fallback_error:
            self.logger.error(f"回退处理器也失败: {fallback_error}")
            return StageResult(
                stage_name=self.config.name,
                success=False,
                error=f"原始错误: {str(error)}, 回退错误: {str(fallback_error)}",
                execution_time=execution_time,
                metadata={'fallback_failed': True}
            )
    
    def _get_data_count(self, data: Any) -> int:
        """
        获取数据数量
        
        Args:
            data: 数据
            
        Returns:
            int: 数据数量
        """
        if data is None:
            return 0
        elif isinstance(data, (list, tuple)):
            return len(data)
        else:
            return 1
    
    def _update_stats(self, success: bool, execution_time: float, processed_count: int):
        """
        更新统计信息
        
        Args:
            success: 是否成功
            execution_time: 执行时间
            processed_count: 处理数量
        """
        self.stats['total_executions'] += 1
        self.stats['total_execution_time'] += execution_time
        self.stats['total_processed'] += processed_count
        self.stats['last_execution'] = datetime.now().isoformat()
        
        if success:
            self.stats['successful_executions'] += 1
        else:
            self.stats['failed_executions'] += 1
        
        # 更新平均执行时间
        if self.stats['total_executions'] > 0:
            self.stats['avg_execution_time'] = (
                self.stats['total_execution_time'] / self.stats['total_executions']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        success_rate = 0.0
        if self.stats['total_executions'] > 0:
            success_rate = self.stats['successful_executions'] / self.stats['total_executions']
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'current_load': self.current_load,
            'max_load': self.max_load,
            'is_running': self.is_running,
            'config': {
                'name': self.config.name,
                'stage_type': self.config.stage_type.value,
                'execution_mode': self.config.execution_mode.value,
                'enabled': self.config.enabled,
                'timeout': self.config.timeout,
                'retry_attempts': self.config.retry_attempts,
                'error_handling': self.config.error_handling
            }
        }
    
    def reset_stats(self):
        """
        重置统计信息
        """
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0,
            'total_processed': 0,
            'last_execution': None,
            'error_count': 0,
            'retry_count': 0
        }
    
    def cleanup(self):
        """
        清理资源
        """
        if self.executor:
            self.executor.shutdown(wait=True)
        
        with self.buffer_lock:
            self.buffer.clear()


class DataPipeline:
    """数据管道
    
    用于构建和执行复杂的数据处理流水线。
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        初始化数据管道
        
        Args:
            name: 管道名称
            config: 配置参数
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # 阶段管理
        self.stages: Dict[str, PipelineStage] = {}
        self.stage_order: List[str] = []
        self.dependencies: Dict[str, List[str]] = {}
        
        # 执行状态
        self.is_running = False
        self.current_execution_id = None
        self.execution_history: List[Dict] = []
        
        # 配置参数
        self.max_execution_time = self.config.get('max_execution_time', 300.0)  # 5分钟
        self.enable_monitoring = self.config.get('enable_monitoring', True)
        self.save_intermediate_results = self.config.get('save_intermediate_results', False)
        self.parallel_stages = self.config.get('parallel_stages', False)
        
        # 统计信息
        self.pipeline_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0,
            'last_execution': None
        }
        
        self.logger.info(f"数据管道 {name} 初始化完成")
    
    def add_stage(self, stage_config: StageConfig) -> 'DataPipeline':
        """
        添加阶段
        
        Args:
            stage_config: 阶段配置
            
        Returns:
            DataPipeline: 管道实例（支持链式调用）
        """
        stage = PipelineStage(stage_config)
        self.stages[stage_config.name] = stage
        
        # 添加到执行顺序
        if stage_config.name not in self.stage_order:
            self.stage_order.append(stage_config.name)
        
        # 处理依赖关系
        if stage_config.dependencies:
            self.dependencies[stage_config.name] = stage_config.dependencies
        
        self.logger.info(f"添加阶段: {stage_config.name}")
        return self
    
    def remove_stage(self, stage_name: str) -> bool:
        """
        移除阶段
        
        Args:
            stage_name: 阶段名称
            
        Returns:
            bool: 是否成功移除
        """
        if stage_name in self.stages:
            # 清理资源
            self.stages[stage_name].cleanup()
            
            # 移除阶段
            del self.stages[stage_name]
            
            # 从执行顺序中移除
            if stage_name in self.stage_order:
                self.stage_order.remove(stage_name)
            
            # 移除依赖关系
            if stage_name in self.dependencies:
                del self.dependencies[stage_name]
            
            # 移除其他阶段对此阶段的依赖
            for deps in self.dependencies.values():
                if stage_name in deps:
                    deps.remove(stage_name)
            
            self.logger.info(f"移除阶段: {stage_name}")
            return True
        
        return False
    
    def execute(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行管道
        
        Args:
            data: 输入数据
            context: 执行上下文
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        if self.is_running:
            raise DataPipelineError("管道正在执行中")
        
        execution_id = f"{self.name}_{int(time.time() * 1000)}"
        self.current_execution_id = execution_id
        
        start_time = datetime.now()
        execution_start = time.time()
        
        try:
            self.is_running = True
            
            # 初始化执行上下文
            if context is None:
                context = {}
            
            context.update({
                'pipeline_name': self.name,
                'execution_id': execution_id,
                'start_time': start_time,
                'intermediate_results': {} if self.save_intermediate_results else None
            })
            
            # 验证依赖关系
            self._validate_dependencies()
            
            # 确定执行顺序
            execution_order = self._resolve_execution_order()
            
            # 执行阶段
            if self.parallel_stages:
                results = self._execute_parallel_stages(data, context, execution_order)
            else:
                results = self._execute_sequential_stages(data, context, execution_order)
            
            execution_time = time.time() - execution_start
            end_time = datetime.now()
            
            # 构建执行结果
            execution_result = {
                'execution_id': execution_id,
                'pipeline_name': self.name,
                'success': True,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'execution_time': execution_time,
                'stages_executed': len(results),
                'stage_results': results,
                'final_data': results[-1]['data'] if results else data,
                'context': context
            }
            
            # 更新统计信息
            self._update_pipeline_stats(True, execution_time)
            
            # 保存执行历史
            self._save_execution_history(execution_result)
            
            self.logger.info(f"管道 {self.name} 执行成功，耗时: {execution_time:.2f}秒")
            
            return execution_result
            
        except Exception as e:
            execution_time = time.time() - execution_start
            end_time = datetime.now()
            
            error_result = {
                'execution_id': execution_id,
                'pipeline_name': self.name,
                'success': False,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'execution_time': execution_time,
                'error': str(e),
                'context': context
            }
            
            # 更新统计信息
            self._update_pipeline_stats(False, execution_time)
            
            # 保存执行历史
            self._save_execution_history(error_result)
            
            self.logger.error(f"管道 {self.name} 执行失败: {e}")
            
            raise DataPipelineError(f"管道执行失败: {e}")
        
        finally:
            self.is_running = False
            self.current_execution_id = None
    
    def _validate_dependencies(self):
        """
        验证依赖关系
        """
        # 检查循环依赖
        visited = set()
        rec_stack = set()
        
        def has_cycle(stage_name: str) -> bool:
            visited.add(stage_name)
            rec_stack.add(stage_name)
            
            for dep in self.dependencies.get(stage_name, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(stage_name)
            return False
        
        for stage_name in self.stages:
            if stage_name not in visited:
                if has_cycle(stage_name):
                    raise DataPipelineError(f"检测到循环依赖，涉及阶段: {stage_name}")
        
        # 检查依赖的阶段是否存在
        for stage_name, deps in self.dependencies.items():
            for dep in deps:
                if dep not in self.stages:
                    raise DataPipelineError(f"阶段 {stage_name} 依赖的阶段 {dep} 不存在")
    
    def _resolve_execution_order(self) -> List[str]:
        """
        解析执行顺序（拓扑排序）
        
        Returns:
            List[str]: 执行顺序
        """
        # 计算入度
        in_degree = {stage: 0 for stage in self.stages}
        for deps in self.dependencies.values():
            for dep in deps:
                in_degree[dep] += 1
        
        # 拓扑排序
        queue = deque([stage for stage, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # 更新依赖此阶段的其他阶段的入度
            for stage, deps in self.dependencies.items():
                if current in deps:
                    in_degree[stage] -= 1
                    if in_degree[stage] == 0:
                        queue.append(stage)
        
        if len(result) != len(self.stages):
            raise DataPipelineError("无法解析执行顺序，可能存在循环依赖")
        
        return result
    
    def _execute_sequential_stages(self, data: Any, context: Dict, execution_order: List[str]) -> List[Dict[str, Any]]:
        """
        顺序执行阶段
        
        Args:
            data: 输入数据
            context: 执行上下文
            execution_order: 执行顺序
            
        Returns:
            List[Dict[str, Any]]: 阶段结果列表
        """
        results = []
        current_data = data
        
        for stage_name in execution_order:
            if stage_name not in self.stages:
                continue
            
            stage = self.stages[stage_name]
            
            # 检查依赖是否已完成
            if not self._check_dependencies_completed(stage_name, results):
                raise DataPipelineError(f"阶段 {stage_name} 的依赖未完成")
            
            # 执行阶段
            stage_result = stage.execute(current_data, context)
            
            # 保存中间结果
            if self.save_intermediate_results and context.get('intermediate_results') is not None:
                context['intermediate_results'][stage_name] = stage_result.data
            
            # 构建结果
            result_dict = {
                'stage_name': stage_result.stage_name,
                'success': stage_result.success,
                'data': stage_result.data,
                'error': stage_result.error,
                'execution_time': stage_result.execution_time,
                'processed_count': stage_result.processed_count,
                'start_time': stage_result.start_time.isoformat() if stage_result.start_time else None,
                'end_time': stage_result.end_time.isoformat() if stage_result.end_time else None,
                'metadata': stage_result.metadata
            }
            
            results.append(result_dict)
            
            # 如果阶段失败，根据配置决定是否继续
            if not stage_result.success:
                if stage.config.error_handling == "raise":
                    raise DataPipelineError(f"阶段 {stage_name} 执行失败: {stage_result.error}")
                # 其他错误处理策略继续执行
            
            # 更新当前数据
            if stage_result.success and stage_result.data is not None:
                current_data = stage_result.data
        
        return results
    
    def _execute_parallel_stages(self, data: Any, context: Dict, execution_order: List[str]) -> List[Dict[str, Any]]:
        """
        并行执行阶段（简化实现）
        
        Args:
            data: 输入数据
            context: 执行上下文
            execution_order: 执行顺序
            
        Returns:
            List[Dict[str, Any]]: 阶段结果列表
        """
        # 简化实现：仍然按依赖顺序执行，但可以并行执行无依赖的阶段
        # 实际实现需要更复杂的调度逻辑
        return self._execute_sequential_stages(data, context, execution_order)
    
    def _check_dependencies_completed(self, stage_name: str, completed_results: List[Dict]) -> bool:
        """
        检查依赖是否已完成
        
        Args:
            stage_name: 阶段名称
            completed_results: 已完成的结果列表
            
        Returns:
            bool: 依赖是否已完成
        """
        dependencies = self.dependencies.get(stage_name, [])
        if not dependencies:
            return True
        
        completed_stages = {result['stage_name'] for result in completed_results if result['success']}
        
        return all(dep in completed_stages for dep in dependencies)
    
    def _update_pipeline_stats(self, success: bool, execution_time: float):
        """
        更新管道统计信息
        
        Args:
            success: 是否成功
            execution_time: 执行时间
        """
        self.pipeline_stats['total_executions'] += 1
        self.pipeline_stats['total_execution_time'] += execution_time
        self.pipeline_stats['last_execution'] = datetime.now().isoformat()
        
        if success:
            self.pipeline_stats['successful_executions'] += 1
        else:
            self.pipeline_stats['failed_executions'] += 1
        
        # 更新平均执行时间
        if self.pipeline_stats['total_executions'] > 0:
            self.pipeline_stats['avg_execution_time'] = (
                self.pipeline_stats['total_execution_time'] / self.pipeline_stats['total_executions']
            )
    
    def _save_execution_history(self, execution_result: Dict):
        """
        保存执行历史
        
        Args:
            execution_result: 执行结果
        """
        # 限制历史记录数量
        max_history = self.config.get('max_execution_history', 100)
        
        self.execution_history.append(execution_result)
        
        if len(self.execution_history) > max_history:
            self.execution_history = self.execution_history[-max_history:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取管道统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stage_stats = {name: stage.get_stats() for name, stage in self.stages.items()}
        
        success_rate = 0.0
        if self.pipeline_stats['total_executions'] > 0:
            success_rate = self.pipeline_stats['successful_executions'] / self.pipeline_stats['total_executions']
        
        return {
            'pipeline_name': self.name,
            'pipeline_stats': {
                **self.pipeline_stats,
                'success_rate': success_rate
            },
            'stage_stats': stage_stats,
            'config': {
                'max_execution_time': self.max_execution_time,
                'enable_monitoring': self.enable_monitoring,
                'save_intermediate_results': self.save_intermediate_results,
                'parallel_stages': self.parallel_stages
            },
            'status': {
                'is_running': self.is_running,
                'current_execution_id': self.current_execution_id,
                'stages_count': len(self.stages),
                'execution_history_count': len(self.execution_history)
            }
        }
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取执行历史
        
        Args:
            limit: 限制数量
            
        Returns:
            List[Dict]: 执行历史
        """
        if limit:
            return self.execution_history[-limit:]
        return self.execution_history.copy()
    
    def reset_stats(self):
        """
        重置统计信息
        """
        self.pipeline_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0,
            'last_execution': None
        }
        
        for stage in self.stages.values():
            stage.reset_stats()
        
        self.execution_history.clear()
        
        self.logger.info(f"管道 {self.name} 统计信息已重置")
    
    def cleanup(self):
        """
        清理资源
        """
        for stage in self.stages.values():
            stage.cleanup()
        
        self.stages.clear()
        self.stage_order.clear()
        self.dependencies.clear()
        self.execution_history.clear()
        
        self.logger.info(f"管道 {self.name} 资源已清理")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()