"""
Extensible hook system for the Atlas Gateway.

Provides pre/post message hooks, authentication hooks, custom command
hooks, and plugin hooks for platform extensions.

Usage::

    hooks = HookSystem()
    hooks.register(HookType.PRE_MESSAGE, my_pre_processor)
    hooks.register(HookType.POST_MESSAGE, my_post_processor)
    await hooks.execute(HookType.PRE_MESSAGE, message_context)
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("atlas.gateway.hooks")


# ──────────────────────────────────────────────────────────────────────────────
# Hook Types
# ──────────────────────────────────────────────────────────────────────────────

class HookType(str, Enum):
    """Available hook points in the message processing pipeline."""

    # Message lifecycle hooks
    PRE_MESSAGE = "pre_message"           # Before message is processed
    POST_MESSAGE = "post_message"         # After message has been processed
    PRE_RESPONSE = "pre_response"         # Before agent response is sent
    POST_RESPONSE = "post_response"       # After agent response is delivered

    # Authentication hooks
    AUTH_CHECK = "auth_check"             # Verify user authentication
    AUTH_SUCCESS = "auth_success"         # User authenticated successfully
    AUTH_FAILURE = "auth_failure"         # Authentication failed

    # Command hooks
    COMMAND_REGISTER = "command_register"  # Register custom commands
    COMMAND_EXECUTE = "command_execute"    # Execute custom command

    # Platform hooks
    PLATFORM_CONNECT = "platform_connect"  # Platform adapter connected
    PLATFORM_DISCONNECT = "platform_disconnect"  # Platform adapter disconnected
    PLATFORM_ERROR = "platform_error"      # Platform adapter error

    # Session hooks
    SESSION_CREATE = "session_create"      # New session created
    SESSION_RESET = "session_reset"        # Session was reset
    SESSION_DESTROY = "session_destroy"    # Session was destroyed

    # Delivery hooks
    PRE_DELIVER = "pre_deliver"            # Before message delivery
    POST_DELIVER = "post_deliver"          # After message delivery
    DELIVER_FAILURE = "deliver_failure"    # Delivery failed

    # Streaming hooks
    STREAM_START = "stream_start"          # Response stream started
    STREAM_CHUNK = "stream_chunk"          # Stream chunk received
    STREAM_END = "stream_end"              # Response stream ended

    # System hooks
    GATEWAY_START = "gateway_start"        # Gateway is starting
    GATEWAY_STOP = "gateway_stop"          # Gateway is stopping
    GATEWAY_ERROR = "gateway_error"        # Gateway-level error

    # Plugin hooks
    PLUGIN_LOAD = "plugin_load"            # Plugin loaded
    PLUGIN_UNLOAD = "plugin_unload"        # Plugin unloaded

    # Custom hooks (for extensions)
    CUSTOM = "custom"


# ──────────────────────────────────────────────────────────────────────────────
# Hook Result
# ──────────────────────────────────────────────────────────────────────────────

class HookResult:
    """Result from executing a hook."""

    def __init__(
        self,
        success: bool = True,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        modified: bool = False,
        abort: bool = False,
        message: Optional[str] = None,
    ):
        self.success = success
        self.data = data or {}
        self.error = error
        self.modified = modified
        self.abort = abort  # If True, abort the current operation
        self.message = message
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "modified": self.modified,
            "abort": self.abort,
            "message": self.message,
            "timestamp": self.timestamp,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Hook Registration
# ──────────────────────────────────────────────────────────────────────────────

class HookRegistration:
    """A registered hook with metadata."""

    def __init__(
        self,
        hook_type: HookType,
        handler: Callable,
        name: Optional[str] = None,
        priority: int = 100,
        platform_filter: Optional[Set[str]] = None,
        enabled: bool = True,
    ):
        self.hook_type = hook_type
        self.handler = handler
        self.name = name or handler.__name__
        self.priority = priority  # Lower = executed first
        self.platform_filter = platform_filter  # None = all platforms
        self.enabled = enabled
        self.call_count = 0
        self.last_called: Optional[str] = None
        self.error_count = 0
        self.total_time_ms: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Hook System
# ──────────────────────────────────────────────────────────────────────────────

class HookSystem:
    """
    Extensible hook system for the Atlas Gateway.

    Manages registration, execution, and lifecycle of hooks across
    all stages of message processing.

    Features:
    - Priority-based execution order
    - Platform filtering
    - Async and sync handler support
    - Hook chaining with abort support
    - Plugin auto-loading from hooks directory
    - Execution statistics tracking
    """

    def __init__(
        self,
        hooks_dir: Optional[str] = None,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._hooks_dir = Path(hooks_dir).expanduser().resolve() if hooks_dir else None
        self._hooks: Dict[HookType, List[HookRegistration]] = defaultdict(list)
        self._command_handlers: Dict[str, Callable] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    # ── Registration ──────────────────────────────────────────────────────

    def register(
        self,
        hook_type: HookType,
        handler: Callable,
        name: Optional[str] = None,
        priority: int = 100,
        platform_filter: Optional[Set[str]] = None,
    ) -> None:
        """
        Register a hook handler.

        Parameters
        ----------
        hook_type:
            The hook point to attach to.
        handler:
            Callable that receives a context dict and returns HookResult.
            Can be sync or async.
        name:
            Optional human-readable name for the hook.
        priority:
            Execution priority (lower = earlier).
        platform_filter:
            If set, only execute for messages from these platforms.
        """
        reg = HookRegistration(
            hook_type=hook_type,
            handler=handler,
            name=name,
            priority=priority,
            platform_filter=platform_filter,
        )
        self._hooks[hook_type].append(reg)
        # Sort by priority (lower first)
        self._hooks[hook_type].sort(key=lambda r: r.priority)
        logger.debug(
            "Registered hook '%s' for %s (priority=%d)",
            reg.name, hook_type.value, priority,
        )

    def unregister(self, hook_type: HookType, name: str) -> bool:
        """
        Unregister a hook by name and type.

        Returns True if the hook was found and removed.
        """
        hooks = self._hooks.get(hook_type, [])
        before = len(hooks)
        self._hooks[hook_type] = [h for h in hooks if h.name != name]
        return len(self._hooks[hook_type]) < before

    def register_command(self, command: str, handler: Callable) -> None:
        """
        Register a custom command handler.

        Parameters
        ----------
        command:
            The command trigger (e.g., "/status", "/help").
        handler:
            Callable that receives a context dict and returns a response string.
        """
        self._command_handlers[command] = handler
        logger.debug("Registered command handler for '%s'", command)

    # ── Execution ─────────────────────────────────────────────────────────

    async def execute(
        self,
        hook_type: HookType,
        context: Dict[str, Any],
        platform: Optional[str] = None,
    ) -> HookResult:
        """
        Execute all registered hooks for a given type.

        Parameters
        ----------
        hook_type:
            The hook point to execute.
        context:
            Shared context dictionary passed to all hooks.
            Hooks can read and modify this dict.
        platform:
            If provided, used for platform filtering.

        Returns
        -------
        HookResult
            Combined result from all hook executions.
        """
        if not self._enabled:
            return HookResult()

        registrations = self._hooks.get(hook_type, [])
        combined = HookResult()

        for reg in registrations:
            if not reg.enabled:
                continue

            # Platform filter
            if reg.platform_filter and platform and platform not in reg.platform_filter:
                continue

            start_time = time.time()
            try:
                result = await self._call_handler(reg.handler, context)

                if isinstance(result, HookResult):
                    if not result.success:
                        combined.error = result.error
                        reg.error_count += 1
                    if result.modified:
                        combined.modified = True
                    if result.abort:
                        combined.abort = True
                    if result.data:
                        combined.data.update(result.data)
                    if result.message:
                        combined.message = result.message
                elif isinstance(result, dict):
                    combined.data.update(result)
                    combined.modified = True
                elif isinstance(result, str):
                    combined.message = result

                reg.call_count += 1
                reg.last_called = datetime.now(timezone.utc).isoformat()

            except Exception as e:
                logger.error(
                    "Hook '%s' (%s) failed: %s",
                    reg.name, hook_type.value, e,
                )
                reg.error_count += 1
                combined.success = False
                combined.error = f"Hook '{reg.name}' error: {e}"

            elapsed = (time.time() - start_time) * 1000
            reg.total_time_ms += elapsed

            # If any hook aborts, stop processing
            if combined.abort:
                logger.debug(
                    "Hook '%s' requested abort for %s",
                    reg.name, hook_type.value,
                )
                break

        return combined

    async def _call_handler(
        self, handler: Callable, context: Dict[str, Any],
    ) -> Any:
        """Call a handler, handling both sync and async callables."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(context)
        elif asyncio.iscoroutine(handler):
            return await handler
        else:
            return handler(context)

    # ── Command Processing ────────────────────────────────────────────────

    async def process_command(
        self, command: str, context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Process a custom command.

        Returns the command response, or None if not handled.
        """
        handler = self._command_handlers.get(command)
        if handler is None:
            return None

        try:
            result = await self._call_handler(handler, context)
            return str(result) if result else None
        except Exception as e:
            logger.error("Command handler for '%s' failed: %s", command, e)
            return f"Error executing command '{command}': {e}"

    def is_command(self, text: str) -> bool:
        """Check if text starts with a registered command."""
        return text.strip().split()[0].strip() in self._command_handlers

    def get_registered_commands(self) -> List[str]:
        """Return all registered command triggers."""
        return list(self._command_handlers.keys())

    # ── Plugin Auto-Loading ───────────────────────────────────────────────

    async def load_plugins(self) -> int:
        """
        Load hook plugins from the hooks directory.

        Each plugin file should define a ``register(hook_system)`` function.

        Returns the number of plugins loaded.
        """
        if self._hooks_dir is None or not self._hooks_dir.exists():
            return 0

        count = 0
        for py_file in sorted(self._hooks_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            try:
                spec = importlib.util.spec_from_file_location(
                    f"atlas_hooks.{py_file.stem}", str(py_file)
                )
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "register"):
                    result = module.register(self)
                    if asyncio.iscoroutine(result):
                        await result
                    count += 1
                    logger.info("Loaded hook plugin: %s", py_file.name)

                # Execute plugin_load hook
                await self.execute(HookType.PLUGIN_LOAD, {
                    "plugin": py_file.stem,
                    "path": str(py_file),
                })

            except Exception as e:
                logger.error("Failed to load plugin %s: %s", py_file.name, e)

        return count

    # ── Statistics ────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return hook system statistics."""
        stats: Dict[str, Any] = {
            "enabled": self._enabled,
            "hooks_dir": str(self._hooks_dir) if self._hooks_dir else None,
            "total_hooks": 0,
            "total_commands": len(self._command_handlers),
            "hooks_by_type": {},
        }

        total = 0
        for hook_type, regs in self._hooks.items():
            total += len(regs)
            stats["hooks_by_type"][hook_type.value] = {
                "count": len(regs),
                "handlers": [
                    {
                        "name": r.name,
                        "priority": r.priority,
                        "enabled": r.enabled,
                        "call_count": r.call_count,
                        "error_count": r.error_count,
                        "total_time_ms": round(r.total_time_ms, 2),
                        "platform_filter": list(r.platform_filter) if r.platform_filter else None,
                    }
                    for r in regs
                ],
            }

        stats["total_hooks"] = total
        return stats
