"""Rate Limiter Module

限流器模块，负责API网关的限流控制。

主要功能：
- 多种限流算法
- 分布式限流
- 动态限流配置
- 限流统计
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import redis.asyncio as redis

from src.common.logging import get_logger
from src.common.exceptions.gateway_exceptions import RateLimitError


class LimitAlgorithm(Enum):
    """限流算法"""
    TOKEN_BUCKET = "token_bucket"        # 令牌桶
    LEAKY_BUCKET = "leaky_bucket"        # 漏桶
    FIXED_WINDOW = "fixed_window"        # 固定窗口
    SLIDING_WINDOW = "sliding_window"    # 滑动窗口
    SLIDING_LOG = "sliding_log"          # 滑动日志


class LimitScope(Enum):
    """限流范围"""
    GLOBAL = "global"                    # 全局限流
    IP = "ip"                            # IP限流
    USER = "user"                        # 用户限流
    API_KEY = "api_key"                  # API Key限流
    PATH = "path"                        # 路径限流
    CUSTOM = "custom"                    # 自定义限流


@dataclass
class LimitConfig:
    """限流配置"""
    # 基本配置
    algorithm: LimitAlgorithm = LimitAlgorithm.TOKEN_BUCKET
    scope: LimitScope = LimitScope.IP
    
    # 限流参数
    rate: int = 100                      # 速率（请求数/时间窗口）
    burst: int = 200                     # 突发容量
    window_size: int = 60                # 时间窗口大小（秒）
    
    # 高级配置
    enable_burst: bool = True            # 是否允许突发
    reject_on_limit: bool = True         # 达到限制时是否拒绝请求
    delay_on_limit: bool = False         # 达到限制时是否延迟请求
    max_delay: float = 5.0               # 最大延迟时间（秒）
    
    # 分布式配置
    distributed: bool = False            # 是否使用分布式限流
    redis_key_prefix: str = "rate_limit:"
    
    # 动态配置
    auto_scale: bool = False             # 是否自动调整限流参数
    scale_factor: float = 1.5            # 调整因子
    scale_threshold: float = 0.8         # 调整阈值
    
    # 白名单
    whitelist: List[str] = field(default_factory=list)
    
    # 自定义标识符函数
    custom_identifier: Optional[str] = None


@dataclass
class LimitState:
    """限流状态"""
    identifier: str
    algorithm: LimitAlgorithm
    
    # 令牌桶状态
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)
    
    # 滑动窗口状态
    requests: deque = field(default_factory=deque)
    
    # 固定窗口状态
    window_start: float = field(default_factory=time.time)
    window_count: int = 0
    
    # 统计信息
    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    last_request_time: float = field(default_factory=time.time)


class RateLimiter:
    """限流器
    
    实现多种限流算法和策略。
    """
    
    def __init__(self, config: LimitConfig):
        """初始化限流器
        
        Args:
            config: 限流配置
        """
        self.config = config
        self.logger = get_logger("rate_limiter")
        
        # 状态存储
        self.states: Dict[str, LimitState] = {}
        
        # Redis连接
        self.redis: Optional[redis.Redis] = None
        
        # 统计信息
        self.stats = {
            'total_checks': 0,
            'allowed_requests': 0,
            'rejected_requests': 0,
            'delayed_requests': 0,
            'active_limiters': 0
        }
        
        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        
        self.logger.info(f"限流器初始化完成，算法: {config.algorithm.value}，范围: {config.scope.value}")
    
    async def initialize(self, redis_url: Optional[str] = None) -> None:
        """初始化异步组件
        
        Args:
            redis_url: Redis连接URL
        """
        # 初始化Redis连接（用于分布式限流）
        if self.config.distributed and redis_url:
            try:
                self.redis = redis.from_url(redis_url)
                await self.redis.ping()
                self.logger.info("Redis连接初始化成功，启用分布式限流")
            except Exception as e:
                self.logger.error(f"Redis连接初始化失败: {e}")
                self.redis = None
                self.config.distributed = False
        
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_states())
    
    async def check_limit(self, identifier: str, request_count: int = 1) -> bool:
        """检查限流
        
        Args:
            identifier: 限流标识符
            request_count: 请求数量
            
        Returns:
            是否允许请求
            
        Raises:
            RateLimitError: 当达到限流限制时
        """
        self.stats['total_checks'] += 1
        
        # 检查白名单
        if identifier in self.config.whitelist:
            self.stats['allowed_requests'] += 1
            return True
        
        try:
            # 分布式限流
            if self.config.distributed and self.redis:
                allowed = await self._check_distributed_limit(identifier, request_count)
            else:
                # 本地限流
                allowed = await self._check_local_limit(identifier, request_count)
            
            if allowed:
                self.stats['allowed_requests'] += 1
                return True
            else:
                self.stats['rejected_requests'] += 1
                
                if self.config.reject_on_limit:
                    raise RateLimitError(f"Rate limit exceeded for {identifier}")
                elif self.config.delay_on_limit:
                    # 延迟请求
                    delay = min(self.config.max_delay, 1.0)
                    await asyncio.sleep(delay)
                    self.stats['delayed_requests'] += 1
                    return True
                else:
                    return False
                    
        except RateLimitError:
            raise
        except Exception as e:
            self.logger.error(f"限流检查失败: {e}")
            # 出错时默认允许请求
            return True
    
    async def _check_local_limit(self, identifier: str, request_count: int) -> bool:
        """检查本地限流
        
        Args:
            identifier: 限流标识符
            request_count: 请求数量
            
        Returns:
            是否允许请求
        """
        # 获取或创建状态
        if identifier not in self.states:
            self.states[identifier] = LimitState(
                identifier=identifier,
                algorithm=self.config.algorithm
            )
            self.stats['active_limiters'] = len(self.states)
        
        state = self.states[identifier]
        current_time = time.time()
        
        # 更新统计
        state.total_requests += request_count
        state.last_request_time = current_time
        
        # 根据算法检查限流
        if self.config.algorithm == LimitAlgorithm.TOKEN_BUCKET:
            allowed = self._check_token_bucket(state, request_count, current_time)
        elif self.config.algorithm == LimitAlgorithm.LEAKY_BUCKET:
            allowed = self._check_leaky_bucket(state, request_count, current_time)
        elif self.config.algorithm == LimitAlgorithm.FIXED_WINDOW:
            allowed = self._check_fixed_window(state, request_count, current_time)
        elif self.config.algorithm == LimitAlgorithm.SLIDING_WINDOW:
            allowed = self._check_sliding_window(state, request_count, current_time)
        elif self.config.algorithm == LimitAlgorithm.SLIDING_LOG:
            allowed = self._check_sliding_log(state, request_count, current_time)
        else:
            allowed = True
        
        if allowed:
            state.allowed_requests += request_count
        else:
            state.rejected_requests += request_count
        
        return allowed
    
    def _check_token_bucket(self, state: LimitState, request_count: int, current_time: float) -> bool:
        """令牌桶算法
        
        Args:
            state: 限流状态
            request_count: 请求数量
            current_time: 当前时间
            
        Returns:
            是否允许请求
        """
        # 计算需要添加的令牌数
        time_passed = current_time - state.last_refill
        tokens_to_add = time_passed * (self.config.rate / self.config.window_size)
        
        # 更新令牌数（不超过桶容量）
        bucket_capacity = self.config.burst if self.config.enable_burst else self.config.rate
        state.tokens = min(bucket_capacity, state.tokens + tokens_to_add)
        state.last_refill = current_time
        
        # 检查是否有足够的令牌
        if state.tokens >= request_count:
            state.tokens -= request_count
            return True
        else:
            return False
    
    def _check_leaky_bucket(self, state: LimitState, request_count: int, current_time: float) -> bool:
        """漏桶算法
        
        Args:
            state: 限流状态
            request_count: 请求数量
            current_time: 当前时间
            
        Returns:
            是否允许请求
        """
        # 计算漏出的请求数
        time_passed = current_time - state.last_refill
        leaked_requests = time_passed * (self.config.rate / self.config.window_size)
        
        # 更新桶中的请求数
        bucket_capacity = self.config.burst if self.config.enable_burst else self.config.rate
        current_requests = max(0, state.tokens - leaked_requests)
        
        # 检查是否可以添加新请求
        if current_requests + request_count <= bucket_capacity:
            state.tokens = current_requests + request_count
            state.last_refill = current_time
            return True
        else:
            return False
    
    def _check_fixed_window(self, state: LimitState, request_count: int, current_time: float) -> bool:
        """固定窗口算法
        
        Args:
            state: 限流状态
            request_count: 请求数量
            current_time: 当前时间
            
        Returns:
            是否允许请求
        """
        # 检查是否需要重置窗口
        if current_time - state.window_start >= self.config.window_size:
            state.window_start = current_time
            state.window_count = 0
        
        # 检查窗口内请求数是否超限
        if state.window_count + request_count <= self.config.rate:
            state.window_count += request_count
            return True
        else:
            return False
    
    def _check_sliding_window(self, state: LimitState, request_count: int, current_time: float) -> bool:
        """滑动窗口算法
        
        Args:
            state: 限流状态
            request_count: 请求数量
            current_time: 当前时间
            
        Returns:
            是否允许请求
        """
        # 清理过期的请求记录
        cutoff_time = current_time - self.config.window_size
        while state.requests and state.requests[0] < cutoff_time:
            state.requests.popleft()
        
        # 检查窗口内请求数是否超限
        if len(state.requests) + request_count <= self.config.rate:
            # 添加当前请求时间戳
            for _ in range(request_count):
                state.requests.append(current_time)
            return True
        else:
            return False
    
    def _check_sliding_log(self, state: LimitState, request_count: int, current_time: float) -> bool:
        """滑动日志算法
        
        Args:
            state: 限流状态
            request_count: 请求数量
            current_time: 当前时间
            
        Returns:
            是否允许请求
        """
        # 与滑动窗口算法类似，但记录精确的时间戳
        return self._check_sliding_window(state, request_count, current_time)
    
    async def _check_distributed_limit(self, identifier: str, request_count: int) -> bool:
        """检查分布式限流
        
        Args:
            identifier: 限流标识符
            request_count: 请求数量
            
        Returns:
            是否允许请求
        """
        if not self.redis:
            return await self._check_local_limit(identifier, request_count)
        
        try:
            # 使用Redis实现分布式限流
            key = f"{self.config.redis_key_prefix}{identifier}"
            current_time = time.time()
            
            if self.config.algorithm == LimitAlgorithm.TOKEN_BUCKET:
                return await self._redis_token_bucket(key, request_count, current_time)
            elif self.config.algorithm == LimitAlgorithm.SLIDING_WINDOW:
                return await self._redis_sliding_window(key, request_count, current_time)
            else:
                # 其他算法回退到本地限流
                return await self._check_local_limit(identifier, request_count)
                
        except Exception as e:
            self.logger.error(f"分布式限流检查失败: {e}")
            # 回退到本地限流
            return await self._check_local_limit(identifier, request_count)
    
    async def _redis_token_bucket(self, key: str, request_count: int, current_time: float) -> bool:
        """Redis令牌桶实现
        
        Args:
            key: Redis键
            request_count: 请求数量
            current_time: 当前时间
            
        Returns:
            是否允许请求
        """
        # Lua脚本实现原子性操作
        lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local burst = tonumber(ARGV[2])
        local window_size = tonumber(ARGV[3])
        local request_count = tonumber(ARGV[4])
        local current_time = tonumber(ARGV[5])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or burst
        local last_refill = tonumber(bucket[2]) or current_time
        
        -- 计算需要添加的令牌
        local time_passed = current_time - last_refill
        local tokens_to_add = time_passed * (rate / window_size)
        tokens = math.min(burst, tokens + tokens_to_add)
        
        -- 检查是否有足够的令牌
        if tokens >= request_count then
            tokens = tokens - request_count
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
            redis.call('EXPIRE', key, window_size * 2)
            return 1
        else
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
            redis.call('EXPIRE', key, window_size * 2)
            return 0
        end
        """
        
        result = await self.redis.eval(
            lua_script,
            1,
            key,
            self.config.rate,
            self.config.burst if self.config.enable_burst else self.config.rate,
            self.config.window_size,
            request_count,
            current_time
        )
        
        return bool(result)
    
    async def _redis_sliding_window(self, key: str, request_count: int, current_time: float) -> bool:
        """Redis滑动窗口实现
        
        Args:
            key: Redis键
            request_count: 请求数量
            current_time: 当前时间
            
        Returns:
            是否允许请求
        """
        # 使用有序集合实现滑动窗口
        cutoff_time = current_time - self.config.window_size
        
        # 清理过期记录
        await self.redis.zremrangebyscore(key, 0, cutoff_time)
        
        # 获取当前窗口内的请求数
        current_count = await self.redis.zcard(key)
        
        if current_count + request_count <= self.config.rate:
            # 添加新的请求记录
            pipeline = self.redis.pipeline()
            for i in range(request_count):
                pipeline.zadd(key, {f"{current_time}_{i}": current_time})
            pipeline.expire(key, self.config.window_size * 2)
            await pipeline.execute()
            return True
        else:
            return False
    
    async def _cleanup_expired_states(self) -> None:
        """清理过期的限流状态"""
        while True:
            try:
                current_time = time.time()
                expired_keys = []
                
                for identifier, state in self.states.items():
                    # 如果状态超过一定时间未使用，则清理
                    if current_time - state.last_request_time > self.config.window_size * 2:
                        expired_keys.append(identifier)
                
                for key in expired_keys:
                    del self.states[key]
                
                if expired_keys:
                    self.stats['active_limiters'] = len(self.states)
                    self.logger.debug(f"清理了 {len(expired_keys)} 个过期的限流状态")
                
                # 每分钟清理一次
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"清理限流状态时出错: {e}")
                await asyncio.sleep(60)
    
    def get_limit_info(self, identifier: str) -> Optional[Dict[str, Any]]:
        """获取限流信息
        
        Args:
            identifier: 限流标识符
            
        Returns:
            限流信息
        """
        if identifier not in self.states:
            return None
        
        state = self.states[identifier]
        current_time = time.time()
        
        info = {
            'identifier': identifier,
            'algorithm': self.config.algorithm.value,
            'rate': self.config.rate,
            'window_size': self.config.window_size,
            'total_requests': state.total_requests,
            'allowed_requests': state.allowed_requests,
            'rejected_requests': state.rejected_requests,
            'last_request_time': state.last_request_time
        }
        
        # 添加算法特定信息
        if self.config.algorithm == LimitAlgorithm.TOKEN_BUCKET:
            # 更新令牌数
            time_passed = current_time - state.last_refill
            tokens_to_add = time_passed * (self.config.rate / self.config.window_size)
            bucket_capacity = self.config.burst if self.config.enable_burst else self.config.rate
            current_tokens = min(bucket_capacity, state.tokens + tokens_to_add)
            
            info.update({
                'current_tokens': current_tokens,
                'bucket_capacity': bucket_capacity
            })
        
        elif self.config.algorithm == LimitAlgorithm.SLIDING_WINDOW:
            # 清理过期请求
            cutoff_time = current_time - self.config.window_size
            while state.requests and state.requests[0] < cutoff_time:
                state.requests.popleft()
            
            info.update({
                'current_window_requests': len(state.requests),
                'remaining_requests': max(0, self.config.rate - len(state.requests))
            })
        
        return info
    
    def reset_limit(self, identifier: str) -> bool:
        """重置限流状态
        
        Args:
            identifier: 限流标识符
            
        Returns:
            是否成功重置
        """
        if identifier in self.states:
            del self.states[identifier]
            self.stats['active_limiters'] = len(self.states)
            self.logger.info(f"重置限流状态: {identifier}")
            return True
        return False
    
    def update_config(self, config: LimitConfig) -> None:
        """更新限流配置
        
        Args:
            config: 新的限流配置
        """
        old_algorithm = self.config.algorithm
        self.config = config
        
        # 如果算法改变，清理所有状态
        if old_algorithm != config.algorithm:
            self.states.clear()
            self.stats['active_limiters'] = 0
            self.logger.info(f"限流算法从 {old_algorithm.value} 更改为 {config.algorithm.value}，清理所有状态")
        
        self.logger.info("限流配置已更新")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息
        """
        return {
            'limiter_stats': self.stats,
            'config': {
                'algorithm': self.config.algorithm.value,
                'scope': self.config.scope.value,
                'rate': self.config.rate,
                'burst': self.config.burst,
                'window_size': self.config.window_size,
                'distributed': self.config.distributed
            },
            'active_limiters': len(self.states)
        }
    
    async def close(self) -> None:
        """关闭限流器"""
        # 停止清理任务
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 关闭Redis连接
        if self.redis:
            await self.redis.close()
        
        self.logger.info("限流器已关闭")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, '_cleanup_task') and self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()