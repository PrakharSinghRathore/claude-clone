"""
Hooks — before/after callbacks for LLM calls and tool usage.
"""

from hooks.decorators import before_llm_call, after_llm_call, before_tool_call, after_tool_call
from hooks.types import HookContext, HookResult

__all__ = [
    "HookContext",
    "HookResult",
    "after_llm_call",
    "after_tool_call",
    "before_llm_call",
    "before_tool_call",
]
