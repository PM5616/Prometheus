"""Event Base Classes

事件系统的基础类定义。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import uuid
import asyncio
from dataclasses import dataclass, field


@dataclass
class Event:
    """事件基类"""
    
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'correlation_id': self.correlation_id
        }


class EventHandler(ABC):
    """事件处理器基类"""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """处理事件
        
        Args:
            event: 要处理的事件
        """
        pass
    
    def can_handle(self, event: Event) -> bool:
        """检查是否可以处理该事件
        
        Args:
            event: 事件对象
            
        Returns:
            是否可以处理
        """
        return True


class EventBus:
    """事件总线"""
    
    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理器
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """订阅所有事件
        
        Args:
            handler: 事件处理器
        """
        self._global_handlers.append(handler)
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """取消订阅
        
        Args:
            event_type: 事件类型
            handler: 事件处理器
        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    async def publish(self, event: Event) -> None:
        """发布事件
        
        Args:
            event: 事件对象
        """
        await self._event_queue.put(event)
    
    async def start(self) -> None:
        """启动事件总线"""
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing event: {e}")
    
    def stop(self) -> None:
        """停止事件总线"""
        self._running = False
    
    async def _process_event(self, event: Event) -> None:
        """处理事件
        
        Args:
            event: 事件对象
        """
        # 处理特定类型的处理器
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                if handler.can_handle(event):
                    try:
                        await handler.handle(event)
                    except Exception as e:
                        print(f"Error in handler {handler}: {e}")
        
        # 处理全局处理器
        for handler in self._global_handlers:
            if handler.can_handle(event):
                try:
                    await handler.handle(event)
                except Exception as e:
                    print(f"Error in global handler {handler}: {e}")