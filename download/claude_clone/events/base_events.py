"""
Base event definitions for the event system.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class EventPriority(int, Enum):
    """Priority levels for event processing."""
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class BaseEvent:
    """
    Base class for all events in the system.
    
    Attributes:
        event_type: String identifier for the event type.
        source: Component that emitted the event.
        priority: Processing priority.
        data: Payload data attached to the event.
        id: Unique event identifier.
        timestamp: When the event was created.
        metadata: Additional metadata.
    """
    
    event_type: str = "base"
    
    def __init__(
        self,
        source: str = "",
        data: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id: str = uuid.uuid4().hex
        self.source = source
        self.data = data or {}
        self.priority = priority
        self.metadata = metadata or {}
        self.timestamp: str = datetime.now().isoformat()
        self._created_at: float = time.time()
    
    @property
    def age_seconds(self) -> float:
        """Seconds elapsed since event creation."""
        return time.time() - self._created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event to a dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "source": self.source,
            "priority": self.priority.value,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    def __repr__(self) -> str:
        data_preview = str(self.data)[:80] if self.data else "{}"
        return (
            f"<{self.__class__.__name__} type={self.event_type!r} "
            f"source={self.source!r} data={data_preview!r}>"
        )
