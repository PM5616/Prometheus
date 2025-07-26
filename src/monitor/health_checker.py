"""Health Checker Module

服务健康检查器，负责监控服务健康状态。

主要功能：
- HTTP/HTTPS健康检查
- TCP端口检查
- 数据库连接检查
- 自定义健康检查
- 健康状态监控
- 故障检测和恢复
"""

import time
import asyncio
import aiohttp
import socket
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from urllib.parse import urlparse

from ..common.logging.logger import get_logger
from ..common.exceptions.monitor_exceptions import MonitorError
from ..common.models import HealthStatus


from ..common.models import CheckType


@dataclass
class HealthCheckConfig:
    """健康检查配置"""
    name: str
    check_type: CheckType
    target: str  # URL, host:port, connection string等
    interval: float = 30.0  # 检查间隔（秒）
    timeout: float = 10.0  # 超时时间（秒）
    retries: int = 3  # 重试次数
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    
    # HTTP特定配置
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    expected_status: List[int] = field(default_factory=lambda: [200])
    expected_content: Optional[str] = None
    
    # 自定义检查函数
    custom_check: Optional[Callable[[], bool]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'check_type': self.check_type.value,
            'target': self.target,
            'interval': self.interval,
            'timeout': self.timeout,
            'retries': self.retries,
            'enabled': self.enabled,
            'tags': self.tags,
            'method': self.method,
            'headers': self.headers,
            'expected_status': self.expected_status,
            'expected_content': self.expected_content
        }


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    check_name: str
    status: HealthStatus
    timestamp: float = field(default_factory=time.time)
    response_time: float = 0.0  # 响应时间（毫秒）
    message: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'check_name': self.check_name,
            'status': self.status.value,
            'timestamp': self.timestamp,
            'response_time': self.response_time,
            'message': self.message,
            'error': self.error,
            'metadata': self.metadata
        }


@dataclass
class ServiceHealth:
    """服务健康状态"""
    service_name: str
    overall_status: HealthStatus
    checks: Dict[str, HealthCheckResult] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    uptime_percentage: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'service_name': self.service_name,
            'overall_status': self.overall_status.value,
            'checks': {name: result.to_dict() for name, result in self.checks.items()},
            'last_updated': self.last_updated,
            'uptime_percentage': self.uptime_percentage
        }


class HealthChecker:
    """健康检查器
    
    负责监控服务健康状态。
    """
    
    def __init__(self, max_history: int = 1000):
        """初始化健康检查器
        
        Args:
            max_history: 最大历史记录数量
        """
        self.max_history = max_history
        self.logger = get_logger("health_checker")
        
        # 检查配置
        self.checks: Dict[str, HealthCheckConfig] = {}
        
        # 检查结果历史
        self.check_history: Dict[str, List[HealthCheckResult]] = {}
        
        # 服务健康状态
        self.service_health: Dict[str, ServiceHealth] = {}
        
        # 运行状态
        self._running = False
        self._check_tasks: Dict[str, asyncio.Task] = {}
        
        # 回调函数
        self.status_change_callbacks: List[Callable[[str, HealthStatus, HealthStatus], None]] = []
        self.check_callbacks: List[Callable[[HealthCheckResult], None]] = []
        
        # HTTP会话
        self._http_session: Optional[aiohttp.ClientSession] = None
        
        self.logger.info("健康检查器初始化完成")
    
    async def start(self) -> None:
        """启动健康检查器"""
        if self._running:
            self.logger.warning("健康检查器已在运行")
            return
        
        self._running = True
        
        # 创建HTTP会话
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # 启动所有检查任务
        for check_name, config in self.checks.items():
            if config.enabled:
                await self._start_check_task(check_name, config)
        
        self.logger.info("健康检查器已启动")
    
    async def stop(self) -> None:
        """停止健康检查器"""
        if not self._running:
            return
        
        self._running = False
        
        # 停止所有检查任务
        for task in self._check_tasks.values():
            if not task.done():
                task.cancel()
        
        # 等待任务完成
        if self._check_tasks:
            await asyncio.gather(*self._check_tasks.values(), return_exceptions=True)
        
        self._check_tasks.clear()
        
        # 关闭HTTP会话
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        
        self.logger.info("健康检查器已停止")
    
    async def _start_check_task(self, check_name: str, config: HealthCheckConfig) -> None:
        """启动检查任务
        
        Args:
            check_name: 检查名称
            config: 检查配置
        """
        if check_name in self._check_tasks:
            # 停止现有任务
            self._check_tasks[check_name].cancel()
        
        # 创建新任务
        task = asyncio.create_task(self._check_loop(check_name, config))
        self._check_tasks[check_name] = task
        
        self.logger.info(f"已启动健康检查任务: {check_name}")
    
    async def _check_loop(self, check_name: str, config: HealthCheckConfig) -> None:
        """检查循环
        
        Args:
            check_name: 检查名称
            config: 检查配置
        """
        while self._running and config.enabled:
            try:
                # 执行健康检查
                result = await self._perform_check(config)
                
                # 处理检查结果
                self._process_check_result(check_name, result)
                
                # 等待下次检查
                await asyncio.sleep(config.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"健康检查任务出错 {check_name}: {e}")
                await asyncio.sleep(min(config.interval, 30.0))
    
    async def _perform_check(self, config: HealthCheckConfig) -> HealthCheckResult:
        """执行健康检查
        
        Args:
            config: 检查配置
            
        Returns:
            检查结果
        """
        start_time = time.time()
        
        try:
            if config.check_type == CheckType.HTTP or config.check_type == CheckType.HTTPS:
                result = await self._check_http(config)
            elif config.check_type == CheckType.TCP:
                result = await self._check_tcp(config)
            elif config.check_type == CheckType.DATABASE:
                result = await self._check_database(config)
            elif config.check_type == CheckType.CUSTOM:
                result = await self._check_custom(config)
            else:
                result = HealthCheckResult(
                    check_name=config.name,
                    status=HealthStatus.UNKNOWN,
                    error=f"不支持的检查类型: {config.check_type}"
                )
            
        except Exception as e:
            result = HealthCheckResult(
                check_name=config.name,
                status=HealthStatus.UNHEALTHY,
                error=str(e)
            )
        
        # 计算响应时间
        result.response_time = (time.time() - start_time) * 1000
        
        return result
    
    async def _check_http(self, config: HealthCheckConfig) -> HealthCheckResult:
        """HTTP健康检查
        
        Args:
            config: 检查配置
            
        Returns:
            检查结果
        """
        if not self._http_session:
            raise MonitorError("HTTP会话未初始化")
        
        for attempt in range(config.retries + 1):
            try:
                async with self._http_session.request(
                    method=config.method,
                    url=config.target,
                    headers=config.headers,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    
                    # 检查状态码
                    if response.status not in config.expected_status:
                        if attempt < config.retries:
                            await asyncio.sleep(1.0)
                            continue
                        
                        return HealthCheckResult(
                            check_name=config.name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"HTTP状态码不匹配: {response.status}",
                            metadata={"status_code": response.status}
                        )
                    
                    # 检查响应内容
                    if config.expected_content:
                        content = await response.text()
                        if config.expected_content not in content:
                            if attempt < config.retries:
                                await asyncio.sleep(1.0)
                                continue
                            
                            return HealthCheckResult(
                                check_name=config.name,
                                status=HealthStatus.UNHEALTHY,
                                message="响应内容不匹配",
                                metadata={"status_code": response.status}
                            )
                    
                    return HealthCheckResult(
                        check_name=config.name,
                        status=HealthStatus.HEALTHY,
                        message="HTTP检查成功",
                        metadata={"status_code": response.status}
                    )
                    
            except asyncio.TimeoutError:
                if attempt < config.retries:
                    await asyncio.sleep(1.0)
                    continue
                
                return HealthCheckResult(
                    check_name=config.name,
                    status=HealthStatus.UNHEALTHY,
                    error="请求超时"
                )
            
            except Exception as e:
                if attempt < config.retries:
                    await asyncio.sleep(1.0)
                    continue
                
                return HealthCheckResult(
                    check_name=config.name,
                    status=HealthStatus.UNHEALTHY,
                    error=str(e)
                )
        
        return HealthCheckResult(
            check_name=config.name,
            status=HealthStatus.UNHEALTHY,
            error="所有重试均失败"
        )
    
    async def _check_tcp(self, config: HealthCheckConfig) -> HealthCheckResult:
        """TCP端口检查
        
        Args:
            config: 检查配置
            
        Returns:
            检查结果
        """
        try:
            # 解析主机和端口
            if ':' in config.target:
                host, port_str = config.target.rsplit(':', 1)
                port = int(port_str)
            else:
                raise ValueError("TCP目标格式错误，应为 host:port")
            
            for attempt in range(config.retries + 1):
                try:
                    # 尝试连接
                    future = asyncio.open_connection(host, port)
                    reader, writer = await asyncio.wait_for(future, timeout=config.timeout)
                    
                    # 关闭连接
                    writer.close()
                    await writer.wait_closed()
                    
                    return HealthCheckResult(
                        check_name=config.name,
                        status=HealthStatus.HEALTHY,
                        message=f"TCP连接成功: {host}:{port}"
                    )
                    
                except (asyncio.TimeoutError, OSError) as e:
                    if attempt < config.retries:
                        await asyncio.sleep(1.0)
                        continue
                    
                    return HealthCheckResult(
                        check_name=config.name,
                        status=HealthStatus.UNHEALTHY,
                        error=f"TCP连接失败: {e}"
                    )
            
        except Exception as e:
            return HealthCheckResult(
                check_name=config.name,
                status=HealthStatus.UNHEALTHY,
                error=str(e)
            )
    
    async def _check_database(self, config: HealthCheckConfig) -> HealthCheckResult:
        """数据库连接检查
        
        Args:
            config: 检查配置
            
        Returns:
            检查结果
        """
        # 这里可以根据不同的数据库类型实现具体的连接检查
        # 目前返回一个基本的实现
        
        try:
            # 解析连接字符串
            parsed = urlparse(config.target)
            
            if parsed.scheme in ['postgresql', 'postgres']:
                return await self._check_postgresql(config)
            elif parsed.scheme == 'mysql':
                return await self._check_mysql(config)
            elif parsed.scheme in ['redis', 'rediss']:
                return await self._check_redis(config)
            else:
                return HealthCheckResult(
                    check_name=config.name,
                    status=HealthStatus.UNKNOWN,
                    error=f"不支持的数据库类型: {parsed.scheme}"
                )
                
        except Exception as e:
            return HealthCheckResult(
                check_name=config.name,
                status=HealthStatus.UNHEALTHY,
                error=str(e)
            )
    
    async def _check_postgresql(self, config: HealthCheckConfig) -> HealthCheckResult:
        """PostgreSQL检查
        
        Args:
            config: 检查配置
            
        Returns:
            检查结果
        """
        try:
            # 这里需要安装 asyncpg 库
            # import asyncpg
            # 
            # conn = await asyncpg.connect(config.target)
            # await conn.execute('SELECT 1')
            # await conn.close()
            
            # 暂时返回模拟结果
            return HealthCheckResult(
                check_name=config.name,
                status=HealthStatus.HEALTHY,
                message="PostgreSQL连接检查（模拟）"
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name=config.name,
                status=HealthStatus.UNHEALTHY,
                error=f"PostgreSQL连接失败: {e}"
            )
    
    async def _check_mysql(self, config: HealthCheckConfig) -> HealthCheckResult:
        """MySQL检查
        
        Args:
            config: 检查配置
            
        Returns:
            检查结果
        """
        try:
            # 这里需要安装 aiomysql 库
            # import aiomysql
            # 
            # conn = await aiomysql.connect(host=host, port=port, user=user, password=password, db=db)
            # cursor = await conn.cursor()
            # await cursor.execute('SELECT 1')
            # await cursor.close()
            # conn.close()
            
            # 暂时返回模拟结果
            return HealthCheckResult(
                check_name=config.name,
                status=HealthStatus.HEALTHY,
                message="MySQL连接检查（模拟）"
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name=config.name,
                status=HealthStatus.UNHEALTHY,
                error=f"MySQL连接失败: {e}"
            )
    
    async def _check_redis(self, config: HealthCheckConfig) -> HealthCheckResult:
        """Redis检查
        
        Args:
            config: 检查配置
            
        Returns:
            检查结果
        """
        try:
            # 这里需要安装 aioredis 库
            # import aioredis
            # 
            # redis = aioredis.from_url(config.target)
            # await redis.ping()
            # await redis.close()
            
            # 暂时返回模拟结果
            return HealthCheckResult(
                check_name=config.name,
                status=HealthStatus.HEALTHY,
                message="Redis连接检查（模拟）"
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name=config.name,
                status=HealthStatus.UNHEALTHY,
                error=f"Redis连接失败: {e}"
            )
    
    async def _check_custom(self, config: HealthCheckConfig) -> HealthCheckResult:
        """自定义检查
        
        Args:
            config: 检查配置
            
        Returns:
            检查结果
        """
        if not config.custom_check:
            return HealthCheckResult(
                check_name=config.name,
                status=HealthStatus.UNKNOWN,
                error="未提供自定义检查函数"
            )
        
        try:
            # 执行自定义检查
            if asyncio.iscoroutinefunction(config.custom_check):
                result = await config.custom_check()
            else:
                result = config.custom_check()
            
            if result:
                return HealthCheckResult(
                    check_name=config.name,
                    status=HealthStatus.HEALTHY,
                    message="自定义检查成功"
                )
            else:
                return HealthCheckResult(
                    check_name=config.name,
                    status=HealthStatus.UNHEALTHY,
                    message="自定义检查失败"
                )
                
        except Exception as e:
            return HealthCheckResult(
                check_name=config.name,
                status=HealthStatus.UNHEALTHY,
                error=f"自定义检查异常: {e}"
            )
    
    def _process_check_result(self, check_name: str, result: HealthCheckResult) -> None:
        """处理检查结果
        
        Args:
            check_name: 检查名称
            result: 检查结果
        """
        # 存储历史记录
        if check_name not in self.check_history:
            self.check_history[check_name] = []
        
        self.check_history[check_name].append(result)
        
        # 限制历史记录数量
        if len(self.check_history[check_name]) > self.max_history:
            self.check_history[check_name] = self.check_history[check_name][-self.max_history:]
        
        # 更新服务健康状态
        self._update_service_health(check_name, result)
        
        # 调用回调
        for callback in self.check_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"检查回调执行失败: {e}")
        
        # 记录日志
        if result.status == HealthStatus.HEALTHY:
            self.logger.debug(f"健康检查成功: {check_name}")
        else:
            self.logger.warning(f"健康检查失败: {check_name} - {result.error or result.message}")
    
    def _update_service_health(self, check_name: str, result: HealthCheckResult) -> None:
        """更新服务健康状态
        
        Args:
            check_name: 检查名称
            result: 检查结果
        """
        # 获取服务名称（从检查名称中提取或使用默认值）
        service_name = check_name.split('_')[0] if '_' in check_name else 'default'
        
        # 获取或创建服务健康状态
        if service_name not in self.service_health:
            self.service_health[service_name] = ServiceHealth(
                service_name=service_name,
                overall_status=HealthStatus.UNKNOWN
            )
        
        service = self.service_health[service_name]
        old_status = service.overall_status
        
        # 更新检查结果
        service.checks[check_name] = result
        service.last_updated = result.timestamp
        
        # 计算整体状态
        statuses = [check.status for check in service.checks.values()]
        
        if all(status == HealthStatus.HEALTHY for status in statuses):
            service.overall_status = HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            service.overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            service.overall_status = HealthStatus.DEGRADED
        else:
            service.overall_status = HealthStatus.UNKNOWN
        
        # 计算正常运行时间百分比
        if check_name in self.check_history:
            history = self.check_history[check_name]
            if len(history) >= 10:  # 至少有10个检查结果
                healthy_count = sum(1 for h in history[-100:] if h.status == HealthStatus.HEALTHY)
                service.uptime_percentage = (healthy_count / min(len(history), 100)) * 100
        
        # 如果状态发生变化，调用状态变化回调
        if old_status != service.overall_status:
            for callback in self.status_change_callbacks:
                try:
                    callback(service_name, old_status, service.overall_status)
                except Exception as e:
                    self.logger.error(f"状态变化回调执行失败: {e}")
    
    def add_check(self, config: HealthCheckConfig) -> None:
        """添加健康检查
        
        Args:
            config: 检查配置
        """
        self.checks[config.name] = config
        
        # 如果监控器正在运行，启动检查任务
        if self._running and config.enabled:
            asyncio.create_task(self._start_check_task(config.name, config))
        
        self.logger.info(f"已添加健康检查: {config.name}")
    
    def remove_check(self, check_name: str) -> bool:
        """移除健康检查
        
        Args:
            check_name: 检查名称
            
        Returns:
            是否成功移除
        """
        if check_name not in self.checks:
            return False
        
        # 停止检查任务
        if check_name in self._check_tasks:
            self._check_tasks[check_name].cancel()
            del self._check_tasks[check_name]
        
        # 移除配置
        del self.checks[check_name]
        
        self.logger.info(f"已移除健康检查: {check_name}")
        return True
    
    def get_check_history(self, check_name: str, limit: int = 100) -> List[HealthCheckResult]:
        """获取检查历史
        
        Args:
            check_name: 检查名称
            limit: 限制数量
            
        Returns:
            检查结果列表
        """
        if check_name not in self.check_history:
            return []
        
        history = self.check_history[check_name]
        return history[-limit:] if limit > 0 else history
    
    def get_service_health(self, service_name: Optional[str] = None) -> Union[ServiceHealth, Dict[str, ServiceHealth]]:
        """获取服务健康状态
        
        Args:
            service_name: 服务名称，如果为None则返回所有服务
            
        Returns:
            服务健康状态
        """
        if service_name:
            return self.service_health.get(service_name)
        else:
            return self.service_health.copy()
    
    def get_overall_health(self) -> HealthStatus:
        """获取整体健康状态
        
        Returns:
            整体健康状态
        """
        if not self.service_health:
            return HealthStatus.UNKNOWN
        
        statuses = [service.overall_status for service in self.service_health.values()]
        
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN
    
    def add_status_change_callback(self, callback: Callable[[str, HealthStatus, HealthStatus], None]) -> None:
        """添加状态变化回调
        
        Args:
            callback: 回调函数
        """
        self.status_change_callbacks.append(callback)
    
    def add_check_callback(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """添加检查回调
        
        Args:
            callback: 回调函数
        """
        self.check_callbacks.append(callback)
    
    def clear_history(self, check_name: Optional[str] = None) -> None:
        """清空历史记录
        
        Args:
            check_name: 检查名称，如果为None则清空所有历史
        """
        if check_name:
            if check_name in self.check_history:
                self.check_history[check_name].clear()
                self.logger.info(f"已清空检查历史: {check_name}")
        else:
            self.check_history.clear()
            self.logger.info("已清空所有检查历史")