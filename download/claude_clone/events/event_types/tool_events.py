"""Tool-related events."""

from __future__ import annotations
from typing import Any, Dict, Optional
from events.base_events import BaseEvent, EventPriority


class ToolEvent(BaseEvent):
    """Event emitted during tool usage."""
    event_type = "tool"
    
    def __init__(
        self,
        action: str = "",
        tool_name: str = "",
        tool_id: str = "",
        agent: str = "",
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            source=tool_name,
            data={"action": action, "tool_name": tool_name, "tool_id": tool_id, "agent": agent, **(data or {})},
            **kwargs,
        )
        self.action = action
        self.tool_name = tool_name
        self.tool_id = tool_id
