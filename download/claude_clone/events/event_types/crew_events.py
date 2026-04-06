"""Crew-related events."""

from __future__ import annotations
from typing import Any, Dict, Optional
from events.base_events import BaseEvent, EventPriority


class CrewEvent(BaseEvent):
    """Event emitted by crew orchestration."""
    event_type = "crew"
    
    def __init__(
        self,
        action: str = "",
        crew_name: str = "",
        process: str = "",
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            source=crew_name,
            data={"action": action, "crew_name": crew_name, "process": process, **(data or {})},
            **kwargs,
        )
        self.action = action
        self.crew_name = crew_name
        self.process = process
