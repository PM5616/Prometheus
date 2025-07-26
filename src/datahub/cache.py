"""Data Cache Module

数据缓存模块，提供高性能的数据缓存功能。

功能特性：
- 多级缓存支持
- LRU/LFU/TTL缓存策略
- 缓存预热和失效
- 分布式缓存支持
- 缓存统计和监控
"""

import time
import logging
import threading
import pickle
import hashlib
import json
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import weakref

from src.common.models.market import Ticker, Kline, OrderBook
from src.common.exceptions.data import DataCacheError


class CachePolicy(Enum):
    """缓存策略枚举"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    TTL = "ttl"  # 生存时间
    FIFO = "fifo"  # 先进先出
    LIFO = "lifo"  # 后进先出
    RANDOM = "random"  # 随机淘汰


class CacheLevel(Enum):
    """缓存级别枚举"""
    L1 = "l1"  # 一级缓存（内存）
    L2 = "l2"  # 二级缓存（本地存储）
    L3 = "l3"  # 三级缓存（分布式）


@dataclass
class CacheConfig:
    """缓存配置"""
    name: str
    policy: CachePolicy = CachePolicy.LRU
    max_size: int = 1000
    ttl: Optional[float] = None  # 秒
    enable_stats: bool = True
    enable_compression: bool = False
    compression_threshold: int = 1024  # 字节
    serializer: str = "pickle"  # pickle, json
    
    def __post_init__(self):
        if isinstance(self.policy, str):
            self.policy = CachePolicy(self.policy)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    compressed: bool = False
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """更新访问时间和次数"""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    memory_usage: int = 0
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """未命中率"""
        return 1.0 - self.hit_rate


class BaseCache(ABC):
    """缓存基类"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.stats = CacheStats()
        self._lock = threading.RLock()
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass
    
    @abstractmethod
    def clear(self):
        """清空缓存"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """获取缓存大小"""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """获取所有键"""
        pass
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return self.get(key) is not None
    
    def get_stats(self) -> CacheStats:
        """获取统计信息"""
        return self.stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = CacheStats()


class MemoryCache(BaseCache):
    """内存缓存实现"""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._data: Dict[str, CacheEntry] = {}
        self._access_order: OrderedDict = OrderedDict()  # LRU
        self._frequency: Dict[str, int] = defaultdict(int)  # LFU
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._data:
                self.stats.misses += 1
                return None
            
            entry = self._data[key]
            
            # 检查过期
            if entry.is_expired():
                self._remove_entry(key)
                self.stats.misses += 1
                self.stats.expirations += 1
                return None
            
            # 更新访问信息
            entry.touch()
            self._update_access_order(key)
            
            self.stats.hits += 1
            
            # 解压缩
            value = entry.value
            if entry.compressed:
                value = self._decompress(value)
            
            return value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        with self._lock:
            try:
                # 序列化和压缩
                serialized_value = self._serialize(value)
                compressed_value, is_compressed = self._compress(serialized_value)
                
                # 计算大小
                size = len(compressed_value) if isinstance(compressed_value, bytes) else len(str(compressed_value))
                
                # 创建缓存条目
                entry = CacheEntry(
                    key=key,
                    value=compressed_value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    ttl=ttl or self.config.ttl,
                    size=size,
                    compressed=is_compressed
                )
                
                # 如果键已存在，先删除
                if key in self._data:
                    self._remove_entry(key)
                
                # 检查容量限制
                while len(self._data) >= self.config.max_size:
                    self._evict_one()
                
                # 添加新条目
                self._data[key] = entry
                self._update_access_order(key)
                
                # 更新统计
                self.stats.size = len(self._data)
                self.stats.memory_usage += size
                
                return True
                
            except Exception as e:
                self.logger.error(f"缓存设置失败 {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            if key in self._data:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._data.clear()
            self._access_order.clear()
            self._frequency.clear()
            self.stats.size = 0
            self.stats.memory_usage = 0
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self._data)
    
    def keys(self) -> List[str]:
        """获取所有键"""
        return list(self._data.keys())
    
    def _remove_entry(self, key: str):
        """移除缓存条目"""
        if key in self._data:
            entry = self._data[key]
            del self._data[key]
            
            if key in self._access_order:
                del self._access_order[key]
            
            if key in self._frequency:
                del self._frequency[key]
            
            self.stats.size = len(self._data)
            self.stats.memory_usage -= entry.size
    
    def _update_access_order(self, key: str):
        """更新访问顺序"""
        if self.config.policy == CachePolicy.LRU:
            if key in self._access_order:
                del self._access_order[key]
            self._access_order[key] = True
        elif self.config.policy == CachePolicy.LFU:
            self._frequency[key] += 1
    
    def _evict_one(self):
        """淘汰一个条目"""
        if not self._data:
            return
        
        key_to_evict = None
        
        if self.config.policy == CachePolicy.LRU:
            # 淘汰最近最少使用的
            key_to_evict = next(iter(self._access_order))
        
        elif self.config.policy == CachePolicy.LFU:
            # 淘汰使用频率最低的
            min_freq = min(self._frequency.values())
            for key, freq in self._frequency.items():
                if freq == min_freq:
                    key_to_evict = key
                    break
        
        elif self.config.policy == CachePolicy.TTL:
            # 淘汰最早过期的
            earliest_key = None
            earliest_time = float('inf')
            
            for key, entry in self._data.items():
                if entry.ttl and entry.created_at < earliest_time:
                    earliest_time = entry.created_at
                    earliest_key = key
            
            key_to_evict = earliest_key or next(iter(self._data))
        
        elif self.config.policy == CachePolicy.FIFO:
            # 淘汰最早添加的
            key_to_evict = next(iter(self._data))
        
        elif self.config.policy == CachePolicy.LIFO:
            # 淘汰最晚添加的
            key_to_evict = next(reversed(self._data))
        
        elif self.config.policy == CachePolicy.RANDOM:
            # 随机淘汰
            import random
            key_to_evict = random.choice(list(self._data.keys()))
        
        if key_to_evict:
            self._remove_entry(key_to_evict)
            self.stats.evictions += 1
    
    def _serialize(self, value: Any) -> bytes:
        """序列化值"""
        if self.config.serializer == "json":
            return json.dumps(value, ensure_ascii=False).encode('utf-8')
        else:  # pickle
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """反序列化值"""
        if self.config.serializer == "json":
            return json.loads(data.decode('utf-8'))
        else:  # pickle
            return pickle.loads(data)
    
    def _compress(self, data: bytes) -> Tuple[bytes, bool]:
        """压缩数据"""
        if not self.config.enable_compression or len(data) < self.config.compression_threshold:
            return data, False
        
        try:
            import gzip
            compressed = gzip.compress(data)
            return compressed, True
        except Exception:
            return data, False
    
    def _decompress(self, data: bytes) -> Any:
        """解压缩数据"""
        try:
            import gzip
            decompressed = gzip.decompress(data)
            return self._deserialize(decompressed)
        except Exception:
            return self._deserialize(data)


class MultiLevelCache:
    """多级缓存"""
    
    def __init__(self, configs: List[CacheConfig]):
        self.caches: List[BaseCache] = []
        self.logger = logging.getLogger(f"{__name__}.MultiLevelCache")
        
        # 创建各级缓存
        for config in configs:
            if config.name.startswith("l1") or "memory" in config.name.lower():
                cache = MemoryCache(config)
            else:
                # 其他类型的缓存可以在这里扩展
                cache = MemoryCache(config)
            
            self.caches.append(cache)
        
        self.logger.info(f"多级缓存初始化完成，共 {len(self.caches)} 级")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值（从高级到低级查找）"""
        for i, cache in enumerate(self.caches):
            value = cache.get(key)
            if value is not None:
                # 将值提升到更高级的缓存中
                for j in range(i):
                    self.caches[j].put(key, value)
                return value
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值（写入所有级别）"""
        success = True
        for cache in self.caches:
            if not cache.put(key, value, ttl):
                success = False
        return success
    
    def delete(self, key: str) -> bool:
        """删除缓存值（从所有级别删除）"""
        success = True
        for cache in self.caches:
            if not cache.delete(key):
                success = False
        return success
    
    def clear(self):
        """清空所有级别的缓存"""
        for cache in self.caches:
            cache.clear()
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """获取所有级别的统计信息"""
        stats = {}
        for i, cache in enumerate(self.caches):
            stats[f"L{i+1}"] = cache.get_stats()
        return stats


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.CacheManager")
        
        # 缓存实例
        self.caches: Dict[str, Union[BaseCache, MultiLevelCache]] = {}
        
        # 缓存预热任务
        self.warmup_tasks: List[Callable] = []
        
        # 缓存失效监听器
        self.invalidation_listeners: List[Callable[[str, str], None]] = []
        
        # 统计信息
        self.global_stats = {
            'total_operations': 0,
            'cache_count': 0,
            'start_time': datetime.now()
        }
        
        self.logger.info("缓存管理器初始化完成")
    
    def create_cache(self, config: CacheConfig) -> BaseCache:
        """创建缓存实例"""
        cache = MemoryCache(config)
        self.caches[config.name] = cache
        self.global_stats['cache_count'] += 1
        
        self.logger.info(f"创建缓存: {config.name}")
        return cache
    
    def create_multi_level_cache(self, name: str, configs: List[CacheConfig]) -> MultiLevelCache:
        """创建多级缓存"""
        cache = MultiLevelCache(configs)
        self.caches[name] = cache
        self.global_stats['cache_count'] += 1
        
        self.logger.info(f"创建多级缓存: {name}")
        return cache
    
    def get_cache(self, name: str) -> Optional[Union[BaseCache, MultiLevelCache]]:
        """获取缓存实例"""
        return self.caches.get(name)
    
    def remove_cache(self, name: str) -> bool:
        """移除缓存实例"""
        if name in self.caches:
            self.caches[name].clear()
            del self.caches[name]
            self.global_stats['cache_count'] -= 1
            self.logger.info(f"移除缓存: {name}")
            return True
        return False
    
    def get(self, cache_name: str, key: str) -> Optional[Any]:
        """从指定缓存获取值"""
        cache = self.get_cache(cache_name)
        if cache:
            self.global_stats['total_operations'] += 1
            return cache.get(key)
        return None
    
    def put(self, cache_name: str, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """向指定缓存设置值"""
        cache = self.get_cache(cache_name)
        if cache:
            self.global_stats['total_operations'] += 1
            return cache.put(key, value, ttl)
        return False
    
    def delete(self, cache_name: str, key: str) -> bool:
        """从指定缓存删除值"""
        cache = self.get_cache(cache_name)
        if cache:
            self.global_stats['total_operations'] += 1
            success = cache.delete(key)
            
            # 通知失效监听器
            for listener in self.invalidation_listeners:
                try:
                    listener(cache_name, key)
                except Exception as e:
                    self.logger.error(f"缓存失效监听器执行失败: {e}")
            
            return success
        return False
    
    def invalidate_pattern(self, cache_name: str, pattern: str) -> int:
        """根据模式失效缓存"""
        cache = self.get_cache(cache_name)
        if not cache:
            return 0
        
        import fnmatch
        
        keys_to_delete = []
        for key in cache.keys():
            if fnmatch.fnmatch(key, pattern):
                keys_to_delete.append(key)
        
        count = 0
        for key in keys_to_delete:
            if cache.delete(key):
                count += 1
                
                # 通知失效监听器
                for listener in self.invalidation_listeners:
                    try:
                        listener(cache_name, key)
                    except Exception as e:
                        self.logger.error(f"缓存失效监听器执行失败: {e}")
        
        self.logger.info(f"模式失效 {cache_name}:{pattern}，删除 {count} 个键")
        return count
    
    def add_warmup_task(self, task: Callable):
        """添加缓存预热任务"""
        self.warmup_tasks.append(task)
        self.logger.info("添加缓存预热任务")
    
    def warmup(self):
        """执行缓存预热"""
        self.logger.info(f"开始缓存预热，共 {len(self.warmup_tasks)} 个任务")
        
        for i, task in enumerate(self.warmup_tasks):
            try:
                self.logger.info(f"执行预热任务 {i+1}/{len(self.warmup_tasks)}")
                task()
            except Exception as e:
                self.logger.error(f"预热任务执行失败: {e}")
        
        self.logger.info("缓存预热完成")
    
    def add_invalidation_listener(self, listener: Callable[[str, str], None]):
        """添加缓存失效监听器"""
        self.invalidation_listeners.append(listener)
        self.logger.info("添加缓存失效监听器")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        stats = self.global_stats.copy()
        
        # 添加运行时间
        stats['uptime'] = datetime.now() - stats['start_time']
        
        # 添加各缓存统计
        cache_stats = {}
        total_hits = 0
        total_misses = 0
        total_size = 0
        total_memory = 0
        
        for name, cache in self.caches.items():
            if isinstance(cache, MultiLevelCache):
                cache_stats[name] = cache.get_stats()
            else:
                cache_stat = cache.get_stats()
                cache_stats[name] = cache_stat
                total_hits += cache_stat.hits
                total_misses += cache_stat.misses
                total_size += cache_stat.size
                total_memory += cache_stat.memory_usage
        
        stats['caches'] = cache_stats
        stats['total_hits'] = total_hits
        stats['total_misses'] = total_misses
        stats['total_hit_rate'] = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
        stats['total_size'] = total_size
        stats['total_memory_usage'] = total_memory
        
        return stats
    
    def clear_all(self):
        """清空所有缓存"""
        for cache in self.caches.values():
            cache.clear()
        self.logger.info("清空所有缓存")
    
    def export_cache_data(self, cache_name: str, format: str = "json") -> Optional[str]:
        """导出缓存数据"""
        cache = self.get_cache(cache_name)
        if not cache or isinstance(cache, MultiLevelCache):
            return None
        
        data = {}
        for key in cache.keys():
            value = cache.get(key)
            if value is not None:
                data[key] = value
        
        if format == "json":
            return json.dumps(data, indent=2, ensure_ascii=False, default=str)
        elif format == "pickle":
            return pickle.dumps(data).hex()
        
        return None
    
    def import_cache_data(self, cache_name: str, data: str, format: str = "json") -> bool:
        """导入缓存数据"""
        cache = self.get_cache(cache_name)
        if not cache or isinstance(cache, MultiLevelCache):
            return False
        
        try:
            if format == "json":
                cache_data = json.loads(data)
            elif format == "pickle":
                cache_data = pickle.loads(bytes.fromhex(data))
            else:
                return False
            
            for key, value in cache_data.items():
                cache.put(key, value)
            
            self.logger.info(f"导入缓存数据到 {cache_name}，共 {len(cache_data)} 个键")
            return True
            
        except Exception as e:
            self.logger.error(f"导入缓存数据失败: {e}")
            return False
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"CacheManager(caches={len(self.caches)}, operations={self.global_stats['total_operations']})"