"""
Atlas Canvas Push — Agent-to-UI (A2UI) push system.

Manages real-time pushing of UI elements, updates, and events to canvas clients.
Supports throttling, batching, subscriptions, and multiple element types.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class ElementType(Enum):
    """Supported canvas element types."""
    TEXT = "text"
    IMAGE = "image"
    CHART = "chart"
    TABLE = "table"
    CODE = "code"
    PROGRESS = "progress"
    BUTTON = "button"
    FORM = "form"
    HEADING = "heading"
    DIVIDER = "divider"
    LIST = "list"
    CONTAINER = "container"
    EMBED = "embed"


class CanvasEventType(Enum):
    """Types of canvas events."""
    ELEMENT_ADDED = "element_added"
    ELEMENT_UPDATED = "element_updated"
    ELEMENT_REMOVED = "element_removed"
    STATE_CHANGED = "state_changed"
    CANVAS_CLEARED = "canvas_cleared"
    INTERACTION = "interaction"
    ERROR = "error"
    SYNC = "sync"


class PushAction(Enum):
    """Actions for canvas updates."""
    ADD = "add"
    UPDATE = "update"
    REMOVE = "remove"
    CLEAR = "clear"
    REPLACE = "replace"
    REORDER = "reorder"


@dataclass
class CanvasElement:
    """Represents a UI element on a canvas."""
    id: str = ""
    type: ElementType = ElementType.TEXT
    content: Any = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    children: List["CanvasElement"] = field(default_factory=list)
    style: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"el_{uuid.uuid4().hex[:10]}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize element to dictionary."""
        result: Dict[str, Any] = {
            "_id": self.id,
            "type": self.type.value,
            "content": self.content,
            "properties": self.properties,
            "style": self.style,
            "metadata": self.metadata,
        }
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanvasElement":
        """Create element from dictionary."""
        el_type = data.get("type", "text")
        if isinstance(el_type, str):
            try:
                el_type = ElementType(el_type)
            except ValueError:
                el_type = ElementType.TEXT

        children = [
            cls.from_dict(c) for c in data.get("children", [])
            if isinstance(c, dict)
        ]

        return cls(
            id=data.get("_id", data.get("id", "")),
            type=el_type,
            content=data.get("content", ""),
            properties=data.get("properties", {}),
            children=children,
            style=data.get("style", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CanvasUpdate:
    """Represents an update operation on a canvas."""
    canvas_name: str
    element_id: Optional[str] = None
    action: PushAction = PushAction.ADD
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    source: str = "agent"
    version: int = 0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize update to dictionary."""
        return {
            "canvas_name": self.canvas_name,
            "element_id": self.element_id,
            "action": self.action.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
            "version": self.version,
        }

    def to_json(self) -> str:
        """Serialize update to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class PushBatch:
    """A batch of updates to be pushed together."""
    updates: List[CanvasUpdate] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    flush_at: float = 0.0

    def add(self, update: CanvasUpdate) -> None:
        """Add an update to the batch."""
        self.updates.append(update)

    @property
    def is_empty(self) -> bool:
        """Whether the batch has no updates."""
        return len(self.updates) == 0

    @property
    def size(self) -> int:
        """Number of updates in the batch."""
        return len(self.updates)

    def to_json(self) -> str:
        """Serialize batch to JSON."""
        return json.dumps({
            "type": "batch",
            "updates": [u.to_dict() for u in self.updates],
            "count": len(self.updates),
            "created_at": self.created_at,
        })


class PushThrottle:
    """
    Throttles push updates to prevent flooding clients.

    Configurable with minimum interval between flushes,
    maximum batch size, and maximum queue depth.
    """

    def __init__(
        self,
        min_interval_ms: float = 50.0,
        max_batch_size: int = 20,
        max_queue_depth: int = 500,
    ) -> None:
        """
        Initialize the throttle.

        Args:
            min_interval_ms: Minimum milliseconds between flushes.
            max_batch_size: Maximum updates per batch.
            max_queue_depth: Maximum pending updates before dropping.
        """
        self._min_interval = min_interval_ms / 1000.0
        self._max_batch_size = max_batch_size
        self._max_queue_depth = max_queue_depth
        self._last_flush = 0.0
        self._pending: Deque[CanvasUpdate] = deque()
        self._dropped_count = 0

    @property
    def pending_count(self) -> int:
        """Number of pending updates."""
        return len(self._pending)

    @property
    def dropped_count(self) -> int:
        """Number of dropped updates due to overflow."""
        return self._dropped_count

    def should_flush(self) -> bool:
        """Check if pending updates should be flushed."""
        if not self._pending:
            return False

        now = time.time()
        elapsed_ms = (now - self._last_flush) * 1000

        # Flush if interval elapsed or batch is full
        return elapsed_ms >= self._min_interval or len(self._pending) >= self._max_batch_size

    def enqueue(self, update: CanvasUpdate) -> bool:
        """
        Add an update to the pending queue.

        Returns:
            True if enqueued, False if dropped (queue full).
        """
        if len(self._pending) >= self._max_queue_depth:
            self._dropped_count += 1
            logger.warning("Push throttle queue overflow, dropping update (total dropped: %d)", self._dropped_count)
            return False

        self._pending.append(update)
        return True

    def dequeue_batch(self) -> List[CanvasUpdate]:
        """
        Dequeue a batch of pending updates.

        Returns:
            List of updates to flush.
        """
        batch: List[CanvasUpdate] = []
        while self._pending and len(batch) < self._max_batch_size:
            batch.append(self._pending.popleft())

        if batch:
            self._last_flush = time.time()

        return batch

    def clear(self) -> int:
        """Clear all pending updates and return count cleared."""
        count = len(self._pending)
        self._pending.clear()
        return count


# Type alias for subscription callbacks
SubscriptionCallback = Callable[[CanvasUpdate], Coroutine[Any, Any, None]]


class A2UIPushManager:
    """
    Agent-to-UI push system for real-time canvas updates.

    Manages pushing UI elements, updating existing elements, removing
    elements, and pushing events to subscribed clients. Includes
    throttling and batching for efficient delivery.

    Example:
        >>> manager = A2UIPushManager()
        >>> await manager.push_element("workspace", CanvasElement(
        ...     type=ElementType.TEXT,
        ...     content="Hello, world!",
        ... ))
        >>> await manager.push_update("workspace", "el_123", {"content": "Updated!"})
    """

    def __init__(
        self,
        throttle_interval_ms: float = 50.0,
        max_batch_size: int = 20,
        max_subscriptions: int = 100,
    ) -> None:
        """
        Initialize the A2UIPushManager.

        Args:
            throttle_interval_ms: Minimum interval between batch flushes.
            max_batch_size: Maximum updates per batch.
            max_subscriptions: Maximum subscriptions per canvas.
        """
        self._subscriptions: Dict[str, List[SubscriptionCallback]] = {}
        self._throttles: Dict[str, PushThrottle] = {}
        self._element_registry: Dict[str, CanvasElement] = {}  # element_id -> element
        self._canvas_elements: Dict[str, Set[str]] = {}  # canvas_name -> set of element_ids

        self._throttle_interval = throttle_interval_ms
        self._max_batch_size = max_batch_size
        self._max_subscriptions = max_subscriptions

        self._flush_tasks: Dict[str, asyncio.Task] = {}
        self._stats = {
            "total_pushed": 0,
            "total_updated": 0,
            "total_removed": 0,
            "total_events": 0,
            "total_dropped": 0,
        }

        logger.info(
            "A2UIPushManager initialized: throttle=%.1fms, batch=%d",
            throttle_interval_ms, max_batch_size,
        )

    @property
    def stats(self) -> Dict[str, int]:
        """Get push statistics."""
        return dict(self._stats)

    def _get_throttle(self, canvas_name: str) -> PushThrottle:
        """Get or create throttle for a canvas."""
        if canvas_name not in self._throttles:
            self._throttles[canvas_name] = PushThrottle(
                min_interval_ms=self._throttle_interval,
                max_batch_size=self._max_batch_size,
            )
        return self._throttles[canvas_name]

    async def push_element(
        self,
        canvas_name: str,
        element: Union[CanvasElement, Dict[str, Any]],
        source: str = "agent",
    ) -> str:
        """
        Push a new UI element to a canvas.

        Args:
            canvas_name: Target canvas name.
            element: CanvasElement or dict to add.
            source: Source identifier.

        Returns:
            Element ID.
        """
        if isinstance(element, dict):
            element = CanvasElement.from_dict(element)

        # Register element
        self._element_registry[element.id] = element
        if canvas_name not in self._canvas_elements:
            self._canvas_elements[canvas_name] = set()
        self._canvas_elements[canvas_name].add(element.id)

        # Create update
        update = CanvasUpdate(
            canvas_name=canvas_name,
            element_id=element.id,
            action=PushAction.ADD,
            data={"element": element.to_dict()},
            source=source,
        )

        # Enqueue and potentially flush
        throttle = self._get_throttle(canvas_name)
        throttle.enqueue(update)
        self._stats["total_pushed"] += 1

        if throttle.should_flush():
            await self._flush(canvas_name)

        logger.debug("Pushed element %s to canvas '%s'", element.id, canvas_name)
        return element.id

    async def push_update(
        self,
        canvas_name: str,
        element_id: str,
        properties: Dict[str, Any],
        source: str = "agent",
    ) -> bool:
        """
        Update properties of an existing element.

        Args:
            canvas_name: Target canvas name.
            element_id: ID of the element to update.
            properties: New properties to apply.
            source: Source identifier.

        Returns:
            True if the element was found and updated.
        """
        element = self._element_registry.get(element_id)
        if not element:
            logger.warning("Element %s not found for update", element_id)
            return False

        # Merge properties
        element.properties.update(properties)
        if "style" in properties and isinstance(properties["style"], dict):
            element.style.update(properties["style"])

        update = CanvasUpdate(
            canvas_name=canvas_name,
            element_id=element_id,
            action=PushAction.UPDATE,
            data={"properties": properties},
            source=source,
        )

        throttle = self._get_throttle(canvas_name)
        throttle.enqueue(update)
        self._stats["total_updated"] += 1

        if throttle.should_flush():
            await self._flush(canvas_name)

        logger.debug("Updated element %s on canvas '%s'", element_id, canvas_name)
        return True

    async def remove_element(
        self,
        canvas_name: str,
        element_id: str,
        source: str = "agent",
    ) -> bool:
        """
        Remove an element from a canvas.

        Args:
            canvas_name: Target canvas name.
            element_id: ID of the element to remove.
            source: Source identifier.

        Returns:
            True if the element was found and removed.
        """
        if element_id not in self._element_registry:
            logger.warning("Element %s not found for removal", element_id)
            return False

        # Unregister element
        del self._element_registry[element_id]
        if canvas_name in self._canvas_elements:
            self._canvas_elements[canvas_name].discard(element_id)

        update = CanvasUpdate(
            canvas_name=canvas_name,
            element_id=element_id,
            action=PushAction.REMOVE,
            data={},
            source=source,
        )

        throttle = self._get_throttle(canvas_name)
        throttle.enqueue(update)
        self._stats["total_removed"] += 1

        if throttle.should_flush():
            await self._flush(canvas_name)

        logger.debug("Removed element %s from canvas '%s'", element_id, canvas_name)
        return True

    async def push_event(
        self,
        canvas_name: str,
        event: Union[Dict[str, Any], CanvasUpdate],
        immediate: bool = True,
    ) -> None:
        """
        Push an event notification to canvas subscribers.

        Events are informational and don't modify canvas elements.

        Args:
            canvas_name: Target canvas name.
            event: Event data dict or CanvasUpdate.
            immediate: Whether to deliver immediately (bypass throttle).
        """
        if isinstance(event, dict):
            event_type = event.get("type", "custom")
            update = CanvasUpdate(
                canvas_name=canvas_name,
                action=PushAction.ADD,
                data=event,
                source="event",
            )
        else:
            update = event

        self._stats["total_events"] += 1

        if immediate:
            # Deliver immediately to all subscribers
            await self._deliver_to_subscribers(canvas_name, [update])
        else:
            throttle = self._get_throttle(canvas_name)
            throttle.enqueue(update)
            if throttle.should_flush():
                await self._flush(canvas_name)

    async def subscribe(
        self,
        canvas_name: str,
        callback: SubscriptionCallback,
    ) -> str:
        """
        Subscribe to updates for a canvas.

        Args:
            canvas_name: Canvas to subscribe to.
            callback: Async callback for updates.

        Returns:
            Subscription ID.
        """
        if canvas_name not in self._subscriptions:
            self._subscriptions[canvas_name] = []

        if len(self._subscriptions[canvas_name]) >= self._max_subscriptions:
            raise RuntimeError(
                f"Max subscriptions ({self._max_subscriptions}) reached for canvas '{canvas_name}'"
            )

        sub_id = f"sub_{uuid.uuid4().hex[:8]}"
        self._subscriptions[canvas_name].append(callback)
        logger.info("Subscription %s added for canvas '%s'", sub_id, canvas_name)
        return sub_id

    async def unsubscribe(
        self,
        canvas_name: str,
        callback: Optional[SubscriptionCallback] = None,
    ) -> int:
        """
        Unsubscribe from canvas updates.

        Args:
            canvas_name: Canvas to unsubscribe from.
            callback: Specific callback to remove. None removes all.

        Returns:
            Number of subscriptions removed.
        """
        subs = self._subscriptions.get(canvas_name, [])
        if callback is None:
            count = len(subs)
            subs.clear()
            return count

        original_count = len(subs)
        self._subscriptions[canvas_name] = [s for s in subs if s != callback]
        return original_count - len(self._subscriptions[canvas_name])

    async def clear_canvas(
        self,
        canvas_name: str,
        source: str = "agent",
    ) -> int:
        """
        Clear all elements from a canvas.

        Args:
            canvas_name: Canvas to clear.

        Returns:
            Number of elements removed.
        """
        element_ids = self._canvas_elements.get(canvas_name, set()).copy()
        count = len(element_ids)

        if not count:
            return 0

        for eid in element_ids:
            self._element_registry.pop(eid, None)
        self._canvas_elements[canvas_name].clear()

        update = CanvasUpdate(
            canvas_name=canvas_name,
            action=PushAction.CLEAR,
            data={"cleared_count": count},
            source=source,
        )

        throttle = self._get_throttle(canvas_name)
        throttle.enqueue(update)
        await self._flush(canvas_name)

        logger.info("Cleared %d elements from canvas '%s'", count, canvas_name)
        return count

    async def flush_all(self) -> Dict[str, int]:
        """
        Flush all pending updates across all canvases.

        Returns:
            Dict mapping canvas names to flushed update counts.
        """
        results = {}
        for canvas_name in list(self._throttles.keys()):
            count = await self._flush(canvas_name)
            if count > 0:
                results[canvas_name] = count
        return results

    def get_canvas_element_count(self, canvas_name: str) -> int:
        """Get the number of elements on a canvas."""
        return len(self._canvas_elements.get(canvas_name, set()))

    def get_element(self, element_id: str) -> Optional[CanvasElement]:
        """Get an element by ID."""
        return self._element_registry.get(element_id)

    def get_canvas_elements(self, canvas_name: str) -> List[CanvasElement]:
        """Get all elements on a canvas."""
        ids = self._canvas_elements.get(canvas_name, set())
        return [self._element_registry[eid] for eid in ids if eid in self._element_registry]

    def get_pending_count(self, canvas_name: str) -> int:
        """Get number of pending updates for a canvas."""
        throttle = self._throttles.get(canvas_name)
        return throttle.pending_count if throttle else 0

    async def _flush(self, canvas_name: str) -> int:
        """Flush pending updates for a canvas."""
        throttle = self._throttles.get(canvas_name)
        if not throttle or throttle.is_empty:
            return 0

        batch = throttle.dequeue_batch()
        if not batch:
            return 0

        await self._deliver_to_subscribers(canvas_name, batch)

        # Update stats with dropped count
        self._stats["total_dropped"] += throttle.dropped_count

        return len(batch)

    async def _deliver_to_subscribers(
        self,
        canvas_name: str,
        updates: List[CanvasUpdate],
    ) -> int:
        """Deliver updates to all subscribers of a canvas."""
        callbacks = self._subscriptions.get(canvas_name, [])
        if not callbacks:
            return 0

        delivered = 0
        for callback in callbacks:
            for update in updates:
                try:
                    result = callback(update)
                    if asyncio.iscoroutine(result):
                        await result
                    delivered += 1
                except Exception as e:
                    logger.error(
                        "Subscription callback error for canvas '%s': %s",
                        canvas_name, e,
                    )

        return delivered

    async def reset(self) -> None:
        """Reset all state (for testing/cleanup)."""
        for canvas_name in list(self._subscriptions.keys()):
            self._subscriptions[canvas_name].clear()
        self._subscriptions.clear()
        self._throttles.clear()
        self._element_registry.clear()
        self._canvas_elements.clear()

        # Cancel flush tasks
        for task in self._flush_tasks.values():
            task.cancel()
        self._flush_tasks.clear()

        logger.info("A2UIPushManager reset")
