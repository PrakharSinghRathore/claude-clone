"""Flow-related events."""

from __future__ import annotations
from typing import Any, Dict, Optional
from events.base_events import BaseEvent, EventPriority


class FlowEvent(BaseEvent):
    """Event emitted during flow execution."""
    event_type = "flow"
    
    def __init__(
        self,
        action: str = "",
        flow_name: str = "",
        step_name: str = "",
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            source=flow_name,
            data={"action": action, "flow_name": flow_name, "step_name": step_name, **(data or {})},
            **kwargs,
        )
        self.action = action
        self.flow_name = flow_name
        self.step_name = step_name
