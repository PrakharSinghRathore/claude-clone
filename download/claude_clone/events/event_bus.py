"""
Central event bus — publish/subscribe event routing.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Type, Union

from events.base_events import BaseEvent, EventPriority

logger = logging.getLogger(__name__)

EventHandler = Callable[[BaseEvent], Any]


class EventBus:
    """
    Central publish/subscribe event bus.
    
    Components register handlers for specific event types. When an event
    is emitted, all registered handlers for that type (and parent types)
    are called in priority order.
    
    Attributes:
        handlers: Registered handlers keyed by event type string.
        global_handlers: Handlers that receive ALL events.
        event_history: Recent event history for debugging.
    """
    
    def __init__(self, max_history: int = 1000):
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []
        self._event_history: List[BaseEvent] = []
        self._max_history = max_history
        self._lock = threading.RLock()
        self._enabled = True
        self._filter_event_types: Optional[Set[str]] = None
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    def enable(self) -> None:
        """Enable event processing."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable event processing (events are silently dropped)."""
        self._enabled = False
    
    def set_filter(self, event_types: Optional[Set[str]]) -> None:
        """Only process events matching these types. None = all."""
        self._filter_event_types = event_types
    
    def on(
        self,
        event_type: str,
        handler: Optional[EventHandler] = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> Any:
        """
        Register a handler for a specific event type.
        
        Can be used as a decorator or as a method call.
        
        Usage as decorator::
        
            @event_bus.on("task_completed")
            def handle_task(event):
                print(f"Task {event.data['name']} completed!")
        
        Usage as method::
        
            event_bus.on("task_completed", my_handler)
        """
        def decorator(fn: EventHandler) -> EventHandler:
            with self._lock:
                self._handlers[event_type].append(fn)
            logger.debug("Registered handler %s for event type '%s'", fn.__name__, event_type)
            return fn
        
        if handler is not None:
            return decorator(handler)
        return decorator
    
    def on_any(self, handler: EventHandler) -> None:
        """
        Register a handler that receives ALL events.
        
        Args:
            handler: Callable that accepts a BaseEvent.
        """
        with self._lock:
            self._global_handlers.append(handler)
        logger.debug("Registered global handler %s", handler.__name__)
    
    def off(self, event_type: str, handler: Optional[EventHandler] = None) -> None:
        """
        Remove a handler for a specific event type.
        
        If handler is None, removes ALL handlers for that event type.
        """
        with self._lock:
            if handler is None:
                self._handlers.pop(event_type, None)
            elif event_type in self._handlers:
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type] if h != handler
                ]
    
    def off_any(self, handler: EventHandler) -> None:
        """Remove a global handler."""
        with self._lock:
            self._global_handlers = [h for h in self._global_handlers if h != handler]
    
    def emit(
        self,
        source: Any,
        event: BaseEvent,
    ) -> None:
        """
        Emit an event to all registered handlers.
        
        Args:
            source: The object emitting the event (used for logging).
            event: The event to emit.
        """
        if not self._enabled:
            return
        
        if self._filter_event_types and event.event_type not in self._filter_event_types:
            return
        
        # Track history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
        
        # Collect handlers
        handlers_to_call: List[EventHandler] = []
        with self._lock:
            handlers_to_call.extend(self._global_handlers)
            handlers_to_call.extend(self._handlers.get(event.event_type, []))
            # Also check for wildcard handlers
            if "*" in self._handlers:
                handlers_to_call.extend(self._handlers["*"])
        
        if not handlers_to_call:
            return
        
        # Sort by priority (from event)
        source_name = type(source).__name__ if source else "unknown"
        logger.debug(
            "Emitting %s from %s to %d handlers",
            event.event_type, source_name, len(handlers_to_call),
        )
        
        # Execute handlers
        errors = []
        for handler in handlers_to_call:
            try:
                if asyncio.iscoroutinefunction(handler):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(handler(event))
                    except RuntimeError:
                        asyncio.run(handler(event))
                else:
                    handler(event)
            except Exception as e:
                errors.append((handler.__name__, str(e)))
                logger.error(
                    "Handler %s failed for event %s: %s",
                    handler.__name__, event.event_type, e,
                )
        
        if errors:
            logger.warning(
                "%d handler(s) failed for event %s",
                len(errors), event.event_type,
            )
    
    async def emit_async(
        self,
        source: Any,
        event: BaseEvent,
    ) -> None:
        """
        Emit an event asynchronously.
        
        Args:
            source: The object emitting the event.
            event: The event to emit.
        """
        if not self._enabled:
            return
        
        if self._filter_event_types and event.event_type not in self._filter_event_types:
            return
        
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
        
        handlers_to_call: List[EventHandler] = []
        with self._lock:
            handlers_to_call.extend(self._global_handlers)
            handlers_to_call.extend(self._handlers.get(event.event_type, []))
            if "*" in self._handlers:
                handlers_to_call.extend(self._handlers["*"])
        
        if not handlers_to_call:
            return
        
        for handler in handlers_to_call:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(
                    "Async handler %s failed for event %s: %s",
                    handler.__name__, event.event_type, e,
                )
    
    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[BaseEvent]:
        """
        Get recent event history.
        
        Args:
            event_type: Filter by event type. None returns all types.
            limit: Maximum number of events to return.
        
        Returns:
            List of recent events, newest first.
        """
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return list(reversed(events[-limit:]))
    
    def clear_history(self) -> None:
        """Clear all event history."""
        self._event_history.clear()
    
    def get_handler_count(self, event_type: Optional[str] = None) -> int:
        """Get the number of registered handlers."""
        if event_type:
            return len(self._handlers.get(event_type, []))
        total = len(self._global_handlers)
        for handlers in self._handlers.values():
            total += len(handlers)
        return total
    
    def __repr__(self) -> str:
        return (
            f"EventBus(handlers={self.get_handler_count()}, "
            f"history={len(self._event_history)}, enabled={self._enabled})"
        )


# Global singleton instance
event_bus = EventBus()
