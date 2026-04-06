"""Task-related events."""

from __future__ import annotations
from typing import Any, Dict, Optional
from events.base_events import BaseEvent, EventPriority


class TaskEvent(BaseEvent):
    """Event emitted during task execution."""
    event_type = "task"
    
    def __init__(
        self,
        action: str = "",
        task_name: str = "",
        task_id: str = "",
        agent: str = "",
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            source=agent or task_name,
            data={"action": action, "task_name": task_name, "task_id": task_id, "agent": agent, **(data or {})},
            **kwargs,
        )
        self.action = action
        self.task_name = task_name
        self.task_id = task_id
        self.agent = agent
