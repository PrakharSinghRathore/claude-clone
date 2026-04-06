"""
Hook decorators for LLM and tool calls.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional

from hooks.types import HookContext, HookResult

logger = logging.getLogger(__name__)

# Global hook registries
_before_llm_hooks: list[Callable] = []
_after_llm_hooks: list[Callable] = []
_before_tool_hooks: list[Callable] = []
_after_tool_hooks: list[Callable] = []


def before_llm_call(fn: Callable) -> Callable:
    """Register a hook to run before every LLM call."""
    _before_llm_hooks.append(fn)
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def after_llm_call(fn: Callable) -> Callable:
    """Register a hook to run after every LLM call."""
    _after_llm_hooks.append(fn)
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def before_tool_call(fn: Callable) -> Callable:
    """Register a hook to run before every tool call."""
    _before_tool_hooks.append(fn)
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def after_tool_call(fn: Callable) -> Callable:
    """Register a hook to run after every tool call."""
    _after_tool_hooks.append(fn)
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def run_before_llm_hooks(context: HookContext) -> HookResult:
    """Execute all registered before-LLM hooks."""
    for hook in _before_llm_hooks:
        try:
            result = hook(context)
            if isinstance(result, HookResult) and result.should_abort:
                return result
        except Exception as e:
            logger.error("Before-LLM hook error: %s", e)
    return HookResult()


def run_after_llm_hooks(context: HookContext) -> HookResult:
    """Execute all registered after-LLM hooks."""
    for hook in _after_llm_hooks:
        try:
            result = hook(context)
            if isinstance(result, HookResult) and result.should_abort:
                return result
        except Exception as e:
            logger.error("After-LLM hook error: %s", e)
    return HookResult()


def run_before_tool_hooks(context: HookContext) -> HookResult:
    """Execute all registered before-tool hooks."""
    for hook in _before_tool_hooks:
        try:
            result = hook(context)
            if isinstance(result, HookResult) and result.should_abort:
                return result
        except Exception as e:
            logger.error("Before-tool hook error: %s", e)
    return HookResult()


def run_after_tool_hooks(context: HookContext) -> HookResult:
    """Execute all registered after-tool hooks."""
    for hook in _after_tool_hooks:
        try:
            result = hook(context)
            if isinstance(result, HookResult) and result.should_abort:
                return result
        except Exception as e:
            logger.error("After-tool hook error: %s", e)
    return HookResult()
