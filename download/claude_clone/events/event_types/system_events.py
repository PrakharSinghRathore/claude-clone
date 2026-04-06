"""System-level events."""

from __future__ import annotations
from typing import Any, Dict, Optional
from events.base_events import BaseEvent, EventPriority


class SystemEvent(BaseEvent):
    """System-level event (startup, shutdown, config change, etc.)."""
    event_type = "system"
    
    def __init__(
        self,
        action: str = "",
        component: str = "",
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            source=component,
            data={"action": action, "component": component, **(data or {})},
            **kwargs,
        )
        self.action = action
        self.component = component


class ErrorEvent(BaseEvent):
    """Error event emitted when something goes wrong."""
    event_type = "error"
    
    def __init__(
        self,
        error: str = "",
        component: str = "",
        error_type: str = "",
        stack_trace: str = "",
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            source=component,
            data={"error": error, "error_type": error_type, "stack_trace": stack_trace[:500], **(data or {})},
            priority=EventPriority.HIGH,
            **kwargs,
        )
        self.error = error
        self.component = component
        self.error_type = error_type
