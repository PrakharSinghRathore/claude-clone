"""Memory-related events."""

from __future__ import annotations
from typing import Any, Dict, Optional
from events.base_events import BaseEvent, EventPriority


class MemoryEvent(BaseEvent):
    """Event emitted during memory operations."""
    event_type = "memory"
    
    def __init__(
        self,
        action: str = "",
        memory_type: str = "",
        query: str = "",
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            source=memory_type,
            data={"action": action, "memory_type": memory_type, "query": query[:200], **(data or {})},
            **kwargs,
        )
        self.action = action
        self.memory_type = memory_type
