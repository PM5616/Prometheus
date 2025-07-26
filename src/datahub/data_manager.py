"""Data Manager Module

数据管理器，作为DataHub模块的核心协调器。

主要功能：
- 统一管理数据提供商
- 协调数据存储
- 管理数据处理流水线
- 提供数据访问接口
- 监控数据流状态
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .providers.base import BaseDataProvider, DataType
from .providers.binance_provider import BinanceProvider
from .providers.yahoo_provider import YahooProvider
from .providers.file_provider import FileProvider

from .storage.base import BaseStorage
from .storage.influxdb_storage import InfluxDBStorage

from .processors.base import BaseProcessor
from .processors.data_cleaner import DataCleaner
from .processors.data_transformer import DataTransformer
from .processors.technical_indicators import TechnicalIndicators
from .processors.data_validator import DataValidator
from .processors.stream_processor import StreamProcessor

from ..common.exceptions.data import DataManagerError, DataProviderError, DataStorageError
from ..common.logging import get_logger
from ..common.models.market_data import Kline, Ticker, OrderBook


class DataPipeline:
    """数据处理流水线"""
    
    def __init__(self, name: str, processors: List[BaseProcessor]):
        """初始化数据流水线
        
        Args:
            name: 流水线名称
            processors: 处理器列表
        """
        self.name = name
        self.processors = processors
        self.is_enabled = True
        self.stats = {
            'total_processed': 0,
            'success_count': 0,
            'error_count': 0,
            'avg_processing_time': 0.0
        }
    
    def process(self, data: Any) -> Any:
        """执行流水线处理
        
        Args:
            data: 输入数据
            
        Returns:
            Any: 处理后的数据
        """
        if not self.is_enabled:
            return data
        
        start_time = time.time()
        current_data = data
        
        try:
            for processor in self.processors:
                if processor.is_initialized():
                    result = processor.process(current_data)
                    if result.success:
                        current_data = result.data
                    else:
                        raise DataManagerError(f"处理器 {processor.name} 处理失败: {result.message}")
            
            # 更新统计信息
            processing_time = time.time() - start_time
            self.stats['total_processed'] += 1
            self.stats['success_count'] += 1
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['success_count'] - 1) + processing_time) /
                self.stats['success_count']
            )
            
            return current_data
            
        except Exception as e:
            self.stats['total_processed'] += 1
            self.stats['error_count'] += 1
            raise DataManagerError(f"流水线 {self.name} 处理失败: {str(e)}")
    
    def get_stats(self) -> Dict:
        """获取流水线统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = self.stats.copy()
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['success_count'] / stats['total_processed']
            stats['error_rate'] = stats['error_count'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
            stats['error_rate'] = 0.0
        return stats


class DataManager:
    """数据管理器
    
    作为DataHub模块的核心协调器，统一管理数据的获取、
    存储、处理和访问。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化数据管理器
        
        Args:
            config: 配置信息
                - providers: 数据提供商配置
                - storage: 存储配置
                - processors: 处理器配置
                - pipelines: 流水线配置
                - cache_config: 缓存配置
        """
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
        
        # 数据提供商
        self.providers: Dict[str, BaseDataProvider] = {}
        self.provider_configs = self.config.get('providers', {})
        
        # 数据存储
        self.storage: Optional[BaseStorage] = None
        self.storage_config = self.config.get('storage', {})
        
        # 数据处理器
        self.processors: Dict[str, BaseProcessor] = {}
        self.processor_configs = self.config.get('processors', {})
        
        # 数据流水线
        self.pipelines: Dict[str, DataPipeline] = {}
        self.pipeline_configs = self.config.get('pipelines', {})
        
        # 缓存配置
        self.cache_config = self.config.get('cache_config', {
            'enabled': True,
            'ttl': 300,  # 5分钟
            'max_size': 1000
        })
        
        # 数据缓存
        self.data_cache = {} if self.cache_config.get('enabled') else None
        self.cache_timestamps = {}
        
        # 订阅管理
        self.subscriptions = {}
        self.subscription_callbacks = {}
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # 管理器状态
        self.is_initialized = False
        self.is_running = False
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'provider_calls': 0,
            'storage_operations': 0,
            'pipeline_executions': 0
        }
    
    def initialize(self) -> bool:
        """初始化数据管理器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info("开始初始化数据管理器")
            
            # 初始化数据提供商
            if not self._initialize_providers():
                return False
            
            # 初始化存储
            if not self._initialize_storage():
                return False
            
            # 初始化处理器
            if not self._initialize_processors():
                return False
            
            # 初始化流水线
            if not self._initialize_pipelines():
                return False
            
            self.is_initialized = True
            self.logger.info("数据管理器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"数据管理器初始化失败: {e}")
            return False
    
    def _initialize_providers(self) -> bool:
        """初始化数据提供商
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            for provider_name, provider_config in self.provider_configs.items():
                provider_type = provider_config.get('type')
                
                if provider_type == 'binance':
                    provider = BinanceProvider(provider_config)
                elif provider_type == 'yahoo':
                    provider = YahooProvider(provider_config)
                elif provider_type == 'file':
                    provider = FileProvider(provider_config)
                else:
                    self.logger.warning(f"未知的数据提供商类型: {provider_type}")
                    continue
                
                if provider.initialize():
                    self.providers[provider_name] = provider
                    self.logger.info(f"数据提供商 {provider_name} 初始化成功")
                else:
                    self.logger.error(f"数据提供商 {provider_name} 初始化失败")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"初始化数据提供商失败: {e}")
            return False
    
    def _initialize_storage(self) -> bool:
        """初始化数据存储
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            if not self.storage_config:
                self.logger.info("未配置数据存储")
                return True
            
            storage_type = self.storage_config.get('type')
            
            if storage_type == 'influxdb':
                self.storage = InfluxDBStorage(self.storage_config)
            else:
                self.logger.warning(f"未知的存储类型: {storage_type}")
                return True
            
            if self.storage.initialize():
                self.logger.info(f"数据存储 {storage_type} 初始化成功")
                return True
            else:
                self.logger.error(f"数据存储 {storage_type} 初始化失败")
                return False
                
        except Exception as e:
            self.logger.error(f"初始化数据存储失败: {e}")
            return False
    
    def _initialize_processors(self) -> bool:
        """初始化数据处理器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            for processor_name, processor_config in self.processor_configs.items():
                processor_type = processor_config.get('type')
                
                if processor_type == 'cleaner':
                    processor = DataCleaner(processor_config)
                elif processor_type == 'transformer':
                    processor = DataTransformer(processor_config)
                elif processor_type == 'indicators':
                    processor = TechnicalIndicators(processor_config)
                elif processor_type == 'validator':
                    processor = DataValidator(processor_config)
                elif processor_type == 'stream':
                    processor = StreamProcessor(processor_config)
                else:
                    self.logger.warning(f"未知的处理器类型: {processor_type}")
                    continue
                
                if processor.initialize():
                    self.processors[processor_name] = processor
                    self.logger.info(f"数据处理器 {processor_name} 初始化成功")
                else:
                    self.logger.error(f"数据处理器 {processor_name} 初始化失败")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"初始化数据处理器失败: {e}")
            return False
    
    def _initialize_pipelines(self) -> bool:
        """初始化数据流水线
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            for pipeline_name, pipeline_config in self.pipeline_configs.items():
                processor_names = pipeline_config.get('processors', [])
                processors = []
                
                for processor_name in processor_names:
                    if processor_name in self.processors:
                        processors.append(self.processors[processor_name])
                    else:
                        self.logger.warning(f"流水线 {pipeline_name} 中的处理器 {processor_name} 不存在")
                
                if processors:
                    pipeline = DataPipeline(pipeline_name, processors)
                    self.pipelines[pipeline_name] = pipeline
                    self.logger.info(f"数据流水线 {pipeline_name} 初始化成功")
            
            return True
            
        except Exception as e:
            self.logger.error(f"初始化数据流水线失败: {e}")
            return False
    
    def get_historical_data(self, 
                          symbol: str,
                          start_time: datetime,
                          end_time: datetime,
                          interval: str = '1m',
                          provider: str = None,
                          use_cache: bool = True,
                          pipeline: str = None) -> Optional[List[Kline]]:
        """获取历史数据
        
        Args:
            symbol: 交易对符号
            start_time: 开始时间
            end_time: 结束时间
            interval: 时间间隔
            provider: 指定数据提供商
            use_cache: 是否使用缓存
            pipeline: 数据处理流水线
            
        Returns:
            Optional[List[Kline]]: 历史K线数据
        """
        try:
            self.stats['total_requests'] += 1
            
            # 生成缓存键
            cache_key = f"historical_{symbol}_{start_time}_{end_time}_{interval}_{provider}"
            
            # 检查缓存
            if use_cache and self._check_cache(cache_key):
                self.stats['cache_hits'] += 1
                return self.data_cache[cache_key]
            
            self.stats['cache_misses'] += 1
            
            # 选择数据提供商
            selected_provider = self._select_provider(provider)
            if not selected_provider:
                raise DataProviderError("没有可用的数据提供商")
            
            # 获取数据
            self.stats['provider_calls'] += 1
            data = selected_provider.get_historical_klines(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                interval=interval
            )
            
            if data is None:
                return None
            
            # 应用数据处理流水线
            if pipeline and pipeline in self.pipelines:
                self.stats['pipeline_executions'] += 1
                data = self.pipelines[pipeline].process(data)
            
            # 存储到缓存
            if use_cache and self.data_cache is not None:
                self._update_cache(cache_key, data)
            
            # 存储到持久化存储
            if self.storage and isinstance(data, list):
                self.stats['storage_operations'] += 1
                for kline in data:
                    self.storage.save_kline(kline)
            
            return data
            
        except Exception as e:
            self.logger.error(f"获取历史数据失败: {e}")
            return None
    
    def get_latest_ticker(self, 
                         symbol: str,
                         provider: str = None,
                         use_cache: bool = True,
                         pipeline: str = None) -> Optional[Ticker]:
        """获取最新行情
        
        Args:
            symbol: 交易对符号
            provider: 指定数据提供商
            use_cache: 是否使用缓存
            pipeline: 数据处理流水线
            
        Returns:
            Optional[Ticker]: 最新行情数据
        """
        try:
            self.stats['total_requests'] += 1
            
            # 生成缓存键
            cache_key = f"ticker_{symbol}_{provider}"
            
            # 检查缓存（行情数据缓存时间较短）
            if use_cache and self._check_cache(cache_key, ttl=30):  # 30秒缓存
                self.stats['cache_hits'] += 1
                return self.data_cache[cache_key]
            
            self.stats['cache_misses'] += 1
            
            # 选择数据提供商
            selected_provider = self._select_provider(provider)
            if not selected_provider:
                raise DataProviderError("没有可用的数据提供商")
            
            # 获取数据
            self.stats['provider_calls'] += 1
            data = selected_provider.get_latest_ticker(symbol)
            
            if data is None:
                return None
            
            # 应用数据处理流水线
            if pipeline and pipeline in self.pipelines:
                self.stats['pipeline_executions'] += 1
                data = self.pipelines[pipeline].process(data)
            
            # 存储到缓存
            if use_cache and self.data_cache is not None:
                self._update_cache(cache_key, data)
            
            # 存储到持久化存储
            if self.storage:
                self.stats['storage_operations'] += 1
                self.storage.save_ticker(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"获取最新行情失败: {e}")
            return None
    
    def subscribe_data(self, 
                      symbol: str,
                      data_type: DataType,
                      callback: Callable,
                      provider: str = None,
                      pipeline: str = None) -> bool:
        """订阅实时数据
        
        Args:
            symbol: 交易对符号
            data_type: 数据类型
            callback: 回调函数
            provider: 指定数据提供商
            pipeline: 数据处理流水线
            
        Returns:
            bool: 订阅是否成功
        """
        try:
            # 选择数据提供商
            selected_provider = self._select_provider(provider)
            if not selected_provider:
                raise DataProviderError("没有可用的数据提供商")
            
            # 创建包装回调函数
            def wrapped_callback(data):
                try:
                    # 应用数据处理流水线
                    processed_data = data
                    if pipeline and pipeline in self.pipelines:
                        self.stats['pipeline_executions'] += 1
                        processed_data = self.pipelines[pipeline].process(data)
                    
                    # 存储到持久化存储
                    if self.storage:
                        self.stats['storage_operations'] += 1
                        if data_type == DataType.TICKER:
                            self.storage.save_ticker(processed_data)
                        elif data_type == DataType.KLINE:
                            self.storage.save_kline(processed_data)
                        elif data_type == DataType.ORDERBOOK:
                            self.storage.save_orderbook(processed_data)
                    
                    # 调用用户回调
                    callback(processed_data)
                    
                except Exception as e:
                    self.logger.error(f"处理订阅数据时发生错误: {e}")
            
            # 执行订阅
            success = selected_provider.subscribe(symbol, data_type, wrapped_callback)
            
            if success:
                # 记录订阅信息
                subscription_key = f"{symbol}_{data_type.value}_{provider or 'default'}"
                self.subscriptions[subscription_key] = {
                    'symbol': symbol,
                    'data_type': data_type,
                    'provider': selected_provider,
                    'callback': callback,
                    'pipeline': pipeline,
                    'created_at': datetime.now()
                }
                
                self.logger.info(f"订阅成功: {subscription_key}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"订阅数据失败: {e}")
            return False
    
    def unsubscribe_data(self, 
                        symbol: str,
                        data_type: DataType,
                        provider: str = None) -> bool:
        """取消订阅
        
        Args:
            symbol: 交易对符号
            data_type: 数据类型
            provider: 指定数据提供商
            
        Returns:
            bool: 取消订阅是否成功
        """
        try:
            subscription_key = f"{symbol}_{data_type.value}_{provider or 'default'}"
            
            if subscription_key in self.subscriptions:
                subscription = self.subscriptions[subscription_key]
                provider_instance = subscription['provider']
                
                # 取消订阅
                success = provider_instance.unsubscribe(symbol, data_type)
                
                if success:
                    del self.subscriptions[subscription_key]
                    self.logger.info(f"取消订阅成功: {subscription_key}")
                
                return success
            else:
                self.logger.warning(f"订阅不存在: {subscription_key}")
                return False
                
        except Exception as e:
            self.logger.error(f"取消订阅失败: {e}")
            return False
    
    def _select_provider(self, provider_name: str = None) -> Optional[BaseDataProvider]:
        """选择数据提供商
        
        Args:
            provider_name: 指定的提供商名称
            
        Returns:
            Optional[BaseDataProvider]: 选中的数据提供商
        """
        if provider_name:
            return self.providers.get(provider_name)
        
        # 选择第一个可用的提供商
        for provider in self.providers.values():
            if provider.is_connected():
                return provider
        
        return None
    
    def _check_cache(self, cache_key: str, ttl: int = None) -> bool:
        """检查缓存是否有效
        
        Args:
            cache_key: 缓存键
            ttl: 生存时间（秒）
            
        Returns:
            bool: 缓存是否有效
        """
        if not self.data_cache or cache_key not in self.data_cache:
            return False
        
        if ttl is None:
            ttl = self.cache_config.get('ttl', 300)
        
        cache_time = self.cache_timestamps.get(cache_key)
        if cache_time is None:
            return False
        
        return (datetime.now() - cache_time).total_seconds() < ttl
    
    def _update_cache(self, cache_key: str, data: Any):
        """更新缓存
        
        Args:
            cache_key: 缓存键
            data: 缓存数据
        """
        if not self.data_cache:
            return
        
        # 检查缓存大小限制
        max_size = self.cache_config.get('max_size', 1000)
        if len(self.data_cache) >= max_size:
            # 删除最旧的缓存项
            oldest_key = min(self.cache_timestamps.keys(), 
                           key=lambda k: self.cache_timestamps[k])
            del self.data_cache[oldest_key]
            del self.cache_timestamps[oldest_key]
        
        self.data_cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.now()
    
    def clear_cache(self):
        """清空缓存"""
        if self.data_cache:
            self.data_cache.clear()
            self.cache_timestamps.clear()
            self.logger.info("缓存已清空")
    
    def get_provider_status(self) -> Dict[str, Dict]:
        """获取数据提供商状态
        
        Returns:
            Dict[str, Dict]: 提供商状态信息
        """
        status = {}
        for name, provider in self.providers.items():
            status[name] = {
                'connected': provider.is_connected(),
                'stats': provider.get_stats(),
                'health': provider.health_check()
            }
        return status
    
    def get_storage_status(self) -> Optional[Dict]:
        """获取存储状态
        
        Returns:
            Optional[Dict]: 存储状态信息
        """
        if self.storage:
            return {
                'connected': self.storage.is_connected(),
                'stats': self.storage.get_stats()
            }
        return None
    
    def get_processor_status(self) -> Dict[str, Dict]:
        """获取处理器状态
        
        Returns:
            Dict[str, Dict]: 处理器状态信息
        """
        status = {}
        for name, processor in self.processors.items():
            status[name] = {
                'initialized': processor.is_initialized(),
                'stats': processor.get_stats()
            }
        return status
    
    def get_pipeline_status(self) -> Dict[str, Dict]:
        """获取流水线状态
        
        Returns:
            Dict[str, Dict]: 流水线状态信息
        """
        status = {}
        for name, pipeline in self.pipelines.items():
            status[name] = {
                'enabled': pipeline.is_enabled,
                'processors': len(pipeline.processors),
                'stats': pipeline.get_stats()
            }
        return status
    
    def get_manager_stats(self) -> Dict:
        """获取管理器统计信息
        
        Returns:
            Dict: 管理器统计信息
        """
        stats = self.stats.copy()
        
        # 计算缓存命中率
        total_cache_requests = stats['cache_hits'] + stats['cache_misses']
        if total_cache_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_requests
        else:
            stats['cache_hit_rate'] = 0.0
        
        # 添加系统状态
        stats.update({
            'providers_count': len(self.providers),
            'processors_count': len(self.processors),
            'pipelines_count': len(self.pipelines),
            'subscriptions_count': len(self.subscriptions),
            'cache_size': len(self.data_cache) if self.data_cache else 0,
            'is_initialized': self.is_initialized,
            'is_running': self.is_running
        })
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'provider_calls': 0,
            'storage_operations': 0,
            'pipeline_executions': 0
        }
        self.logger.info("管理器统计信息已重置")
    
    def start(self) -> bool:
        """启动数据管理器
        
        Returns:
            bool: 启动是否成功
        """
        if not self.is_initialized:
            self.logger.error("数据管理器未初始化")
            return False
        
        try:
            # 连接所有数据提供商
            for name, provider in self.providers.items():
                if not provider.connect():
                    self.logger.warning(f"数据提供商 {name} 连接失败")
            
            # 连接存储
            if self.storage and not self.storage.connect():
                self.logger.warning("数据存储连接失败")
            
            self.is_running = True
            self.logger.info("数据管理器已启动")
            return True
            
        except Exception as e:
            self.logger.error(f"启动数据管理器失败: {e}")
            return False
    
    def stop(self) -> bool:
        """停止数据管理器
        
        Returns:
            bool: 停止是否成功
        """
        try:
            # 取消所有订阅
            for subscription_key in list(self.subscriptions.keys()):
                subscription = self.subscriptions[subscription_key]
                provider = subscription['provider']
                provider.unsubscribe(subscription['symbol'], subscription['data_type'])
            
            self.subscriptions.clear()
            
            # 断开所有数据提供商
            for provider in self.providers.values():
                provider.disconnect()
            
            # 断开存储
            if self.storage:
                self.storage.disconnect()
            
            # 关闭线程池
            self.thread_pool.shutdown(wait=True)
            
            self.is_running = False
            self.logger.info("数据管理器已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止数据管理器失败: {e}")
            return False
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.is_initialized:
            self.initialize()
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()