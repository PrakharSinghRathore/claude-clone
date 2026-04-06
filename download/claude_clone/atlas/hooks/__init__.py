"""
atlas.hooks - Comprehensive hook and event system.

Provides a priority-based hook system with async support,
error isolation, hook chaining, and result aggregation for
extending the Claude Clone agent at well-defined extension points.
"""

from atlas.hooks.system import (
    HookPoint,
    HookPriority,
    HookContext,
    HookResult,
    Hook,
    HookSystem,
)

__all__ = [
    "HookPoint",
    "HookPriority",
    "HookContext",
    "HookResult",
    "Hook",
    "HookSystem",
]
