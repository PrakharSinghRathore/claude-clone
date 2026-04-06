"""Agent-related events."""

from __future__ import annotations
from typing import Any, Dict, Optional
from events.base_events import BaseEvent, EventPriority


class AgentEvent(BaseEvent):
    """Event emitted by agents during execution."""
    event_type = "agent"
    
    def __init__(
        self,
        action: str = "",
        agent_role: str = "",
        agent_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            source=agent_role,
            data={"action": action, "agent_role": agent_role, "agent_id": agent_id, **(data or {})},
            **kwargs,
        )
        self.action = action
        self.agent_role = agent_role
        self.agent_id = agent_id
