"""Base Processor Module

数据处理器基类，定义所有数据处理器的通用接口和规范。

主要功能：
- 定义处理器接口规范
- 提供通用的处理流程
- 支持批量和流式处理
- 处理结果验证和监控
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import pandas as pd
import logging

from ...common.models.market_data import Kline, Ticker, OrderBook
from ...common.exceptions.data import DataProcessingError


class ProcessorType(Enum):
    """处理器类型枚举"""
    CLEANER = "cleaner"  # 数据清洗
    TRANSFORMER = "transformer"  # 数据转换
    INDICATOR = "indicator"  # 技术指标
    VALIDATOR = "validator"  # 数据验证
    STREAM = "stream"  # 流式处理
    AGGREGATOR = "aggregator"  # 数据聚合
    FILTER = "filter"  # 数据过滤


class ProcessingResult:
    """处理结果类"""
    
    def __init__(self, 
                 data: Any,
                 success: bool = True,
                 message: str = "",
                 metadata: Optional[Dict] = None,
                 processing_time: Optional[float] = None):
        """初始化处理结果
        
        Args:
            data: 处理后的数据
            success: 处理是否成功
            message: 处理消息或错误信息
            metadata: 处理元数据
            processing_time: 处理耗时（秒）
        """
        self.data = data
        self.success = success
        self.message = message
        self.metadata = metadata or {}
        self.processing_time = processing_time
        self.timestamp = datetime.now()
    
    def __repr__(self) -> str:
        return f"ProcessingResult(success={self.success}, message='{self.message}')"


class BaseProcessor(ABC):
    """数据处理器基类
    
    所有数据处理器都应该继承此基类，并实现相应的抽象方法。
    """
    
    def __init__(self, 
                 name: str,
                 processor_type: ProcessorType,
                 config: Optional[Dict] = None):
        """初始化处理器
        
        Args:
            name: 处理器名称
            processor_type: 处理器类型
            config: 处理器配置
        """
        self.name = name
        self.processor_type = processor_type
        self.config = config or {}
        self.logger = logging.getLogger(f"datahub.processors.{name}")
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'success_count': 0,
            'error_count': 0,
            'total_processing_time': 0.0,
            'last_processed': None
        }
        
        # 回调函数
        self.callbacks = {
            'on_success': [],
            'on_error': [],
            'on_complete': []
        }
        
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化处理器
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """处理数据
        
        Args:
            data: 待处理的数据
            **kwargs: 额外参数
            
        Returns:
            ProcessingResult: 处理结果
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 数据是否有效
        """
        pass
    
    def process_batch(self, data_list: List[Any], **kwargs) -> List[ProcessingResult]:
        """批量处理数据
        
        Args:
            data_list: 数据列表
            **kwargs: 额外参数
            
        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        results = []
        
        for data in data_list:
            try:
                result = self.process(data, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"批量处理数据时发生错误: {e}")
                results.append(ProcessingResult(
                    data=None,
                    success=False,
                    message=str(e)
                ))
        
        return results
    
    def add_callback(self, event: str, callback: Callable) -> None:
        """添加回调函数
        
        Args:
            event: 事件类型 ('on_success', 'on_error', 'on_complete')
            callback: 回调函数
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            self.logger.warning(f"未知的事件类型: {event}")
    
    def remove_callback(self, event: str, callback: Callable) -> None:
        """移除回调函数
        
        Args:
            event: 事件类型
            callback: 回调函数
        """
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
    
    def _trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        """触发回调函数
        
        Args:
            event: 事件类型
            *args: 位置参数
            **kwargs: 关键字参数
        """
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"回调函数执行失败: {e}")
    
    def _update_stats(self, result: ProcessingResult) -> None:
        """更新统计信息
        
        Args:
            result: 处理结果
        """
        self.stats['total_processed'] += 1
        self.stats['last_processed'] = datetime.now()
        
        if result.success:
            self.stats['success_count'] += 1
            self._trigger_callbacks('on_success', result)
        else:
            self.stats['error_count'] += 1
            self._trigger_callbacks('on_error', result)
        
        if result.processing_time:
            self.stats['total_processing_time'] += result.processing_time
        
        self._trigger_callbacks('on_complete', result)
    
    def get_stats(self) -> Dict:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = self.stats.copy()
        
        # 计算平均处理时间
        if stats['total_processed'] > 0:
            stats['avg_processing_time'] = (
                stats['total_processing_time'] / stats['total_processed']
            )
            stats['success_rate'] = (
                stats['success_count'] / stats['total_processed']
            )
        else:
            stats['avg_processing_time'] = 0.0
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_processed': 0,
            'success_count': 0,
            'error_count': 0,
            'total_processing_time': 0.0,
            'last_processed': None
        }
        self.logger.info(f"处理器 {self.name} 统计信息已重置")
    
    def is_initialized(self) -> bool:
        """检查处理器是否已初始化
        
        Returns:
            bool: 是否已初始化
        """
        return self._initialized
    
    def shutdown(self) -> None:
        """关闭处理器"""
        self.logger.info(f"处理器 {self.name} 正在关闭")
        self._initialized = False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.processor_type.value})"