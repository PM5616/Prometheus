"""Event Manager

事件管理器，提供事件系统的统一管理接口。
"""

import asyncio
from typing import Dict, List, Optional, Callable
from .base import Event, EventHandler, EventBus
from .types import EventType


class EventManager:
    """事件管理器"""
    
    _instance: Optional['EventManager'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls) -> 'EventManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._event_bus = EventBus()
            self._task: Optional[asyncio.Task] = None
            self._initialized = True
    
    async def start(self) -> None:
        """启动事件管理器"""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._event_bus.start())
    
    async def stop(self) -> None:
        """停止事件管理器"""
        self._event_bus.stop()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理器
        """
        self._event_bus.subscribe(event_type, handler)
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """订阅所有事件
        
        Args:
            handler: 事件处理器
        """
        self._event_bus.subscribe_all(handler)
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """取消订阅
        
        Args:
            event_type: 事件类型
            handler: 事件处理器
        """
        self._event_bus.unsubscribe(event_type, handler)
    
    async def publish(self, event: Event) -> None:
        """发布事件
        
        Args:
            event: 事件对象
        """
        await self._event_bus.publish(event)
    
    async def emit(self, event_type: str, data: Dict = None, source: str = None) -> None:
        """发送事件（便捷方法）
        
        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件源
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source
        )
        await self.publish(event)


# 全局事件管理器实例
event_manager = EventManager()