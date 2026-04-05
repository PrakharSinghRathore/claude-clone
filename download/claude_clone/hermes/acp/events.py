"""
ACP Event System — Event types, streaming via WebSocket,
event filtering, and subscription management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events that can be emitted through the ACP system."""

    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    DONE = "done"
    THINKING = "thinking"
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    SESSION_CREATED = "session_created"
    SESSION_ENDED = "session_ended"
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_RESPONSE = "permission_response"
    SYSTEM = "system"


@dataclass
class ACPEvent:
    """A single event in the ACP system."""

    type: EventType
    data: dict = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the event to a JSON string."""
        return json.dumps({
            "id": self.event_id,
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "metadata": self.metadata,
        })

    @classmethod
    def from_json(cls, raw: str) -> Optional["ACPEvent"]:
        """Deserialize an event from a JSON string."""
        try:
            data = json.loads(raw)
            return cls(
                type=EventType(data["type"]),
                data=data.get("data", {}),
                event_id=data.get("id", str(uuid.uuid4())),
                timestamp=data.get("timestamp", time.time()),
                session_id=data.get("session_id"),
                metadata=data.get("metadata", {}),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Failed to parse event from JSON")
            return None


@dataclass
class EventSubscription:
    """A subscription to filtered events."""

    subscription_id: str
    event_types: set[EventType] = field(default_factory=lambda: {e for e in EventType})
    session_filter: Optional[str] = None
    queue: asyncio.Queue[ACPEvent] = field(default_factory=lambda: asyncio.Queue(maxsize=1000))
    callback: Optional[Callable[[ACPEvent], Any]] = None
    created_at: float = field(default_factory=time.time)

    def matches(self, event: ACPEvent) -> bool:
        """Check if an event matches this subscription's filters."""
        if event.type not in self.event_types:
            return False
        if self.session_filter and event.session_id != self.session_filter:
            return False
        return True


class EventManager:
    """
    Central event bus for the ACP system.

    Supports event emission, subscription with filtering,
    WebSocket streaming, and event history.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._subscriptions: dict[str, EventSubscription] = {}
        self._history: list[ACPEvent] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        event_types: Optional[list[EventType]] = None,
        session_filter: Optional[str] = None,
        callback: Optional[Callable[[ACPEvent], Any]] = None,
    ) -> str:
        """
        Create a new event subscription.

        Returns a subscription ID that can be used to read events
        or cancel the subscription.
        """
        sub_id = str(uuid.uuid4())
        types = set(event_types) if event_types else {e for e in EventType}
        sub = EventSubscription(
            subscription_id=sub_id,
            event_types=types,
            session_filter=session_filter,
            callback=callback,
        )
        async with self._lock:
            self._subscriptions[sub_id] = sub
        logger.debug("Event subscription created: %s (types=%s)", sub_id, [t.value for t in types])
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Cancel a subscription."""
        async with self._lock:
            sub = self._subscriptions.pop(subscription_id, None)
            return sub is not None

    async def get_event(self, subscription_id: str, timeout: float = 30.0) -> Optional[ACPEvent]:
        """
        Get the next event for a subscription.

        Blocks until an event is available or timeout expires.
        """
        sub = self._subscriptions.get(subscription_id)
        if sub is None:
            return None
        try:
            return await asyncio.wait_for(sub.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    async def emit(self, event: ACPEvent) -> int:
        """
        Emit an event to all matching subscriptions.

        Returns the number of subscriptions that received the event.
        """
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        delivered = 0
        async with self._lock:
            for sub in self._subscriptions.values():
                if sub.matches(event):
                    try:
                        sub.queue.put_nowait(event)
                        delivered += 1
                    except asyncio.QueueFull:
                        logger.warning("Event queue full for subscription %s", sub.subscription_id)
                    if sub.callback:
                        try:
                            result = sub.callback(event)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception:
                            logger.exception("Event callback failed for %s", sub.subscription_id)

        return delivered

    async def emit_message(self, content: str, session_id: Optional[str] = None, **kwargs: Any) -> ACPEvent:
        """Convenience: emit a MESSAGE event."""
        event = ACPEvent(type=EventType.MESSAGE, data={"content": content}, session_id=session_id, **kwargs)
        await self.emit(event)
        return event

    async def emit_tool_call(self, tool_name: str, params: dict, session_id: Optional[str] = None) -> ACPEvent:
        """Convenience: emit a TOOL_CALL event."""
        event = ACPEvent(
            type=EventType.TOOL_CALL,
            data={"tool": tool_name, "params": params},
            session_id=session_id,
        )
        await self.emit(event)
        return event

    async def emit_tool_result(self, tool_name: str, result: Any, session_id: Optional[str] = None) -> ACPEvent:
        """Convenience: emit a TOOL_RESULT event."""
        event = ACPEvent(
            type=EventType.TOOL_RESULT,
            data={"tool": tool_name, "result": result},
            session_id=session_id,
        )
        await self.emit(event)
        return event

    async def emit_error(self, message: str, session_id: Optional[str] = None, **kwargs: Any) -> ACPEvent:
        """Convenience: emit an ERROR event."""
        event = ACPEvent(type=EventType.ERROR, data={"message": message}, session_id=session_id, **kwargs)
        await self.emit(event)
        return event

    async def emit_thinking(self, content: str, session_id: Optional[str] = None) -> ACPEvent:
        """Convenience: emit a THINKING event."""
        event = ACPEvent(type=EventType.THINKING, data={"content": content}, session_id=session_id)
        await self.emit(event)
        return event

    async def emit_done(self, session_id: Optional[str] = None) -> ACPEvent:
        """Convenience: emit a DONE event."""
        event = ACPEvent(type=EventType.DONE, data={}, session_id=session_id)
        await self.emit(event)
        return event

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> list[ACPEvent]:
        """Retrieve historical events with optional filtering."""
        events = self._history
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def clear_history(self, session_id: Optional[str] = None) -> int:
        """Clear event history, optionally for a specific session only."""
        if session_id:
            before = len(self._history)
            self._history = [e for e in self._history if e.session_id != session_id]
            return before - len(self._history)
        removed = len(self._history)
        self._history.clear()
        return removed

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return event system statistics."""
        return {
            "active_subscriptions": len(self._subscriptions),
            "history_size": len(self._history),
            "max_history": self._max_history,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup_stale_subscriptions(self, max_age_seconds: float = 3600.0) -> int:
        """Remove subscriptions that have been idle for too long."""
        now = time.time()
        stale: list[str] = []
        async with self._lock:
            for sub_id, sub in self._subscriptions.items():
                if now - sub.created_at > max_age_seconds:
                    stale.append(sub_id)
            for sub_id in stale:
                del self._subscriptions[sub_id]
        if stale:
            logger.info("Cleaned up %d stale subscriptions", len(stale))
        return len(stale)
