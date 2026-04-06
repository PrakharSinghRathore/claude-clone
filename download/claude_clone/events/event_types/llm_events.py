"""LLM-related events."""

from __future__ import annotations
from typing import Any, Dict, Optional
from events.base_events import BaseEvent, EventPriority


class LLMEvent(BaseEvent):
    """Event emitted during LLM interactions."""
    event_type = "llm"
    
    def __init__(
        self,
        action: str = "",
        model: str = "",
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            source=model,
            data={"action": action, "model": model, **(data or {})},
            **kwargs,
        )
        self.action = action
        self.model = model
