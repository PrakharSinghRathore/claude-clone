"""
Hook type definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class HookContext:
    """Context passed to hook functions."""
    event_type: str = ""  # "llm_call", "tool_call"
    agent_role: str = ""
    method_name: str = ""
    messages: Optional[list] = None
    tools: Optional[list] = None
    response: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookResult:
    """Result returned by a hook function."""
    modified: bool = False
    modified_data: Optional[Any] = None
    should_abort: bool = False
    reason: str = ""
