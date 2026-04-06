"""
atlas.hooks.system - Comprehensive hook and event system.

Implements a priority-based hook system with support for async hooks,
error isolation, hook chaining, result aggregation, and dynamic
registration/unregistration at well-defined extension points throughout
the Claude Clone agent lifecycle.

Hook Points:
    The system defines hook points for key moments in the agent lifecycle:

    - PRE_EXECUTION / POST_EXECUTION: Before and after agent execution
    - PRE_TOOL_CALL / POST_TOOL_CALL: Before and after tool invocation
    - ON_ERROR: When an error occurs during execution
    - ON_MESSAGE: When a user message is received
    - ON_RESPONSE: When the agent generates a response
    - PRE_SEND / POST_SEND: Before and after sending messages
    - ON_CONNECT / ON_DISCONNECT: Connection lifecycle
    - SESSION_START / SESSION_END: Session lifecycle
    - CONFIG_CHANGE: When configuration changes
    - PLUGIN_LOAD / PLUGIN_UNLOAD: Plugin lifecycle

Example::

    hook_system = HookSystem()

    # Register a hook
    async def my_pre_execution_hook(context: HookContext) -> HookResult:
        print(f"About to execute: {context.data.get('query')}")
        return HookResult(handled=False)

    hook_system.register(
        HookPoint.PRE_EXECUTION,
        my_pre_execution_hook,
        priority=HookPriority.HIGH,
        name="my_pre_exec",
    )

    # Execute hooks for a point
    context = HookContext(hook_point=HookPoint.PRE_EXECUTION, data={"query": "hello"})
    results = await hook_system.execute(HookPoint.PRE_EXECUTION, context)

    # Execute until a hook handles the event
    result = await hook_system.execute_until(
        HookPoint.ON_MESSAGE,
        context,
        predicate=lambda r: r.handled,
    )
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

# Type aliases for handler functions
SyncHookHandler = Callable[["HookContext"], "HookResult"]
AsyncHookHandler = Callable[["HookContext"], Awaitable["HookResult"]]
HookHandler = Union[SyncHookHandler, AsyncHookHandler]

# Generic type for execute_until predicate
T = TypeVar("T")


class HookPoint(Enum):
    """Well-defined extension points in the agent lifecycle.

    Each hook point represents a specific moment where custom
    behavior can be injected.
    """
    # Execution lifecycle
    PRE_EXECUTION = "pre_execution"
    POST_EXECUTION = "post_execution"

    # Tool call lifecycle
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"

    # Error handling
    ON_ERROR = "on_error"

    # Message lifecycle
    ON_MESSAGE = "on_message"
    ON_RESPONSE = "on_response"

    # Send lifecycle
    PRE_SEND = "pre_send"
    POST_SEND = "post_send"

    # Connection lifecycle
    ON_CONNECT = "on_connect"
    ON_DISCONNECT = "on_disconnect"

    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Configuration
    CONFIG_CHANGE = "config_change"

    # Plugin lifecycle
    PLUGIN_LOAD = "plugin_load"
    PLUGIN_UNLOAD = "plugin_unload"

    # Custom hooks (plugins can define their own)
    CUSTOM = "custom"


class HookPriority(Enum):
    """Hook execution priority.

    Hooks with higher priority are executed first within each
    hook point. The order within the same priority level is
    the order of registration.
    """
    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


@dataclass
class HookContext:
    """Context passed to hook handlers.

    Contains information about the hook point, the data being
    processed, and mutable state that hooks can modify.

    Attributes:
        hook_point: The hook point being executed.
        timestamp: When the hook was triggered.
        data: The primary data payload for this hook.
        metadata: Additional metadata about the context.
        cancelled: Whether the operation has been cancelled.
        error: Any error that triggered this hook (for ON_ERROR).
        session_id: The session identifier.
        source: The source that triggered the hook.
        extra: Additional key-value data for extensibility.
    """
    hook_point: HookPoint = HookPoint.CUSTOM
    timestamp: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    error: Optional[Exception] = None
    session_id: str = ""
    source: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = time.time()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the data dictionary.

        Args:
            key: The key to look up.
            default: Default value if key not found.

        Returns:
            The value, or the default.
        """
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the data dictionary.

        Args:
            key: The key to set.
            value: The value to set.
        """
        self.data[key] = value

    def cancel(self, reason: str = "") -> None:
        """Cancel the current operation.

        Args:
            reason: Optional reason for cancellation.
        """
        self.cancelled = True
        if reason:
            self.metadata["cancel_reason"] = reason

    @property
    def datetime(self) -> datetime:
        """Get the timestamp as a datetime object."""
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "hook_point": self.hook_point.value,
            "timestamp": self.timestamp,
            "datetime": self.datetime.isoformat(),
            "data": self.data,
            "metadata": self.metadata,
            "cancelled": self.cancelled,
            "session_id": self.session_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HookContext:
        """Create a HookContext from a dictionary."""
        hook_point = data.get("hook_point", "custom")
        if isinstance(hook_point, str):
            hook_point = HookPoint(hook_point)

        return cls(
            hook_point=hook_point,
            timestamp=data.get("timestamp", time.time()),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
            cancelled=data.get("cancelled", False),
            session_id=data.get("session_id", ""),
            source=data.get("source", ""),
        )


@dataclass
class HookResult:
    """Result returned by a hook handler.

    Attributes:
        handled: Whether the hook handled the event (stops propagation).
        data: Data to pass back to the caller.
        error: Error if the hook encountered an issue.
        modified: Whether the hook modified the context.
        message: Optional message from the hook.
        priority: Execution priority (set by the system).
        handler_name: Name of the handler that produced this result.
        execution_time_ms: Time taken to execute the handler.
    """
    handled: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    modified: bool = False
    message: str = ""
    priority: int = HookPriority.NORMAL.value
    handler_name: str = ""
    execution_time_ms: float = 0.0

    @classmethod
    def handled_result(
        cls,
        data: Optional[Dict[str, Any]] = None,
        message: str = "",
    ) -> HookResult:
        """Create a result indicating the event was handled.

        Args:
            data: Optional data to include.
            message: Optional message.

        Returns:
            A HookResult with handled=True.
        """
        return cls(
            handled=True,
            data=data or {},
            message=message,
        )

    @classmethod
    def error_result(
        cls,
        error: str,
        message: str = "",
    ) -> HookResult:
        """Create a result indicating an error.

        Args:
            error: Error description.
            message: Optional message.

        Returns:
            A HookResult with the error set.
        """
        return cls(
            error=error,
            message=message or error,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "handled": self.handled,
            "data": self.data,
            "error": self.error,
            "modified": self.modified,
            "message": self.message,
            "handler_name": self.handler_name,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


@dataclass
class Hook:
    """A registered hook with its configuration.

    Attributes:
        name: Unique name for this hook.
        hook_point: The hook point this hook is registered for.
        handler: The handler function (sync or async).
        priority: Execution priority.
        enabled: Whether the hook is currently enabled.
        plugin_name: Name of the plugin that registered this hook.
        tags: Tags for categorizing hooks.
        max_retries: Maximum retry attempts on failure.
        timeout: Maximum execution time in seconds.
        description: Human-readable description.
        call_count: Number of times this hook has been called.
        error_count: Number of times this hook has errored.
        last_called: Timestamp of last call.
        total_execution_time_ms: Cumulative execution time.
    """
    name: str
    hook_point: HookPoint
    handler: HookHandler
    priority: int = HookPriority.NORMAL.value
    enabled: bool = True
    plugin_name: str = ""
    tags: Set[str] = field(default_factory=set)
    max_retries: int = 0
    timeout: Optional[float] = None
    description: str = ""
    call_count: int = 0
    error_count: int = 0
    last_called: float = 0.0
    total_execution_time_ms: float = 0.0

    @property
    def is_async(self) -> bool:
        """Whether this hook handler is asynchronous."""
        return asyncio.iscoroutinefunction(self.handler)

    @property
    def average_execution_time_ms(self) -> float:
        """Average execution time in milliseconds."""
        if self.call_count == 0:
            return 0.0
        return self.total_execution_time_ms / self.call_count

    @property
    def success_rate(self) -> float:
        """Hook success rate (0.0 - 1.0)."""
        if self.call_count == 0:
            return 1.0
        return (self.call_count - self.error_count) / self.call_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "hook_point": self.hook_point.value,
            "priority": self.priority,
            "enabled": self.enabled,
            "plugin_name": self.plugin_name,
            "tags": sorted(self.tags),
            "is_async": self.is_async,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "description": self.description,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "success_rate": round(self.success_rate, 4),
            "average_execution_time_ms": round(self.average_execution_time_ms, 2),
            "last_called": self.last_called,
        }


class HookSystem:
    """Comprehensive hook system with priority-based execution.

    Manages hook registration, execution, and lifecycle. Supports
    both synchronous and asynchronous hooks with error isolation,
    timeout handling, and result aggregation.

    Features:
        - Priority-based ordered execution
        - Error isolation (one hook failure doesn't affect others)
        - Async hook support
        - Hook chaining with early termination
        - Result aggregation
        - Dynamic registration/unregistration
        - Enable/disable hooks
        - Statistics tracking
        - Hook groups by plugin

    Example::

        hooks = HookSystem()

        # Register hooks
        hooks.register(HookPoint.PRE_EXECUTION, my_handler, name="pre_exec")
        hooks.register(HookPoint.ON_MESSAGE, another_handler, priority=HookPriority.HIGH)

        # Execute all hooks for a point
        context = HookContext(hook_point=HookPoint.PRE_EXECUTION)
        results = await hooks.execute(HookPoint.PRE_EXECUTION, context)

        # Execute until a handler returns handled=True
        result = await hooks.execute_until(
            HookPoint.ON_MESSAGE, context,
            predicate=lambda r: r.handled,
        )

        # Unregister
        hooks.unregister("pre_exec")
    """

    def __init__(self, error_isolation: bool = True) -> None:
        """Initialize the HookSystem.

        Args:
            error_isolation: If True, one hook failure won't prevent
                other hooks from executing.
        """
        self._hooks: Dict[HookPoint, List[Hook]] = defaultdict(list)
        self._hooks_by_name: Dict[str, Hook] = {}
        self._error_isolation = error_isolation
        self._lock = asyncio.Lock()
        self._execution_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000
        self._global_stats = {
            "total_executions": 0,
            "total_errors": 0,
            "total_execution_time_ms": 0.0,
        }

    def register(
        self,
        hook_point: HookPoint,
        handler: HookHandler,
        priority: Union[int, HookPriority] = HookPriority.NORMAL,
        name: Optional[str] = None,
        plugin_name: str = "",
        tags: Optional[Set[str]] = None,
        description: str = "",
        max_retries: int = 0,
        timeout: Optional[float] = None,
    ) -> Hook:
        """Register a hook handler.

        Args:
            hook_point: The hook point to register for.
            handler: The handler function (sync or async).
            priority: Execution priority (HookPriority enum or int).
            name: Unique name for the hook. Auto-generated if None.
            plugin_name: Name of the registering plugin.
            tags: Tags for categorization.
            description: Human-readable description.
            max_retries: Max retry attempts on failure.
            timeout: Max execution time in seconds.

        Returns:
            The registered Hook object.

        Raises:
            ValueError: If a hook with the same name already exists.
        """
        # Resolve priority
        if isinstance(priority, HookPriority):
            priority_value = priority.value
        else:
            priority_value = int(priority)

        # Generate name if not provided
        if not name:
            name = f"{hook_point.value}_{uuid.uuid4().hex[:8]}"
        elif name in self._hooks_by_name:
            raise ValueError(f"Hook with name '{name}' already exists")

        # Validate handler
        if not callable(handler):
            raise TypeError(f"Handler must be callable, got {type(handler)}")

        # Create hook object
        hook = Hook(
            name=name,
            hook_point=hook_point,
            handler=handler,
            priority=priority_value,
            plugin_name=plugin_name,
            tags=tags or set(),
            description=description,
            max_retries=max_retries,
            timeout=timeout,
        )

        # Register
        self._hooks[hook_point].append(hook)
        self._hooks_by_name[name] = hook

        # Sort by priority (lower = higher priority = executed first)
        self._hooks[hook_point].sort(key=lambda h: h.priority)

        logger.info(
            "Registered hook '%s' for %s (priority=%d, async=%s)",
            name,
            hook_point.value,
            priority_value,
            hook.is_async,
        )
        return hook

    def unregister(self, name: str) -> bool:
        """Unregister a hook by name.

        Args:
            name: The unique name of the hook.

        Returns:
            True if the hook was found and removed.
        """
        hook = self._hooks_by_name.pop(name, None)
        if not hook:
            logger.warning("Hook '%s' not found for unregistration", name)
            return False

        # Remove from the hook point list
        hook_list = self._hooks.get(hook.hook_point, [])
        self._hooks[hook.hook_point] = [
            h for h in hook_list if h.name != name
        ]

        logger.info("Unregistered hook '%s' from %s", name, hook.hook_point.value)
        return True

    def unregister_by_plugin(self, plugin_name: str) -> int:
        """Unregister all hooks registered by a plugin.

        Args:
            plugin_name: The name of the plugin.

        Returns:
            The number of hooks removed.
        """
        to_remove = [
            name for name, hook in self._hooks_by_name.items()
            if hook.plugin_name == plugin_name
        ]

        for name in to_remove:
            self.unregister(name)

        logger.info(
            "Unregistered %d hooks for plugin '%s'",
            len(to_remove),
            plugin_name,
        )
        return len(to_remove)

    def unregister_by_tag(self, tag: str) -> int:
        """Unregister all hooks with a specific tag.

        Args:
            tag: The tag to filter by.

        Returns:
            The number of hooks removed.
        """
        to_remove = [
            name for name, hook in self._hooks_by_name.items()
            if tag in hook.tags
        ]

        for name in to_remove:
            self.unregister(name)

        logger.info(
            "Unregistered %d hooks with tag '%s'",
            len(to_remove),
            tag,
        )
        return len(to_remove)

    async def execute(
        self,
        hook_point: HookPoint,
        context: HookContext,
    ) -> List[HookResult]:
        """Execute all hooks registered for a hook point.

        Hooks are executed in priority order. If error isolation
        is enabled, a single hook failure won't prevent other
        hooks from executing.

        Args:
            hook_point: The hook point to execute.
            context: The hook context.

        Returns:
            A list of HookResult objects from all executed hooks.
        """
        hooks = self._get_enabled_hooks(hook_point)

        if not hooks:
            return []

        # Update global stats
        self._global_stats["total_executions"] += 1

        results: List[HookResult] = []
        start_time = time.monotonic()

        for hook in hooks:
            if context.cancelled:
                logger.debug(
                    "Execution cancelled, skipping hook '%s'", hook.name
                )
                break

            result = await self._execute_single_hook(hook, context)
            results.append(result)

            # Check if context was modified
            if result.modified:
                # Update context from result data if provided
                for key, value in result.data.items():
                    if key not in context.extra:
                        context.extra[key] = value

        # Record execution history
        total_time_ms = (time.monotonic() - start_time) * 1000
        self._global_stats["total_execution_time_ms"] += total_time_ms

        self._record_execution(
            hook_point=hook_point,
            hook_count=len(hooks),
            result_count=len(results),
            total_time_ms=total_time_ms,
            cancelled=context.cancelled,
        )

        logger.debug(
            "Executed %d hooks for %s in %.1fms",
            len(results),
            hook_point.value,
            total_time_ms,
        )
        return results

    async def execute_until(
        self,
        hook_point: HookPoint,
        context: HookContext,
        predicate: Callable[[HookResult], bool],
    ) -> Optional[HookResult]:
        """Execute hooks until a predicate is satisfied.

        Hooks are executed in priority order until one returns
        a result that satisfies the predicate.

        Args:
            hook_point: The hook point to execute.
            context: The hook context.
            predicate: A function that takes a HookResult and
                returns True to stop execution.

        Returns:
            The first HookResult that satisfies the predicate,
            or None if no hook matched.
        """
        hooks = self._get_enabled_hooks(hook_point)

        for hook in hooks:
            result = await self._execute_single_hook(hook, context)

            if predicate(result):
                logger.debug(
                    "Hook '%s' satisfied predicate for %s",
                    hook.name,
                    hook_point.value,
                )
                return result

        return None

    def list_hooks(
        self,
        hook_point: Optional[HookPoint] = None,
        plugin_name: Optional[str] = None,
        enabled_only: bool = False,
    ) -> List[Hook]:
        """List registered hooks.

        Args:
            hook_point: Filter by hook point. If None, show all.
            plugin_name: Filter by plugin name. If None, show all.
            enabled_only: Only show enabled hooks.

        Returns:
            A list of Hook objects matching the filters.
        """
        hooks: List[Hook] = []

        if hook_point:
            source_list = self._hooks.get(hook_point, [])
        else:
            source_list = list(self._hooks_by_name.values())

        for hook in source_list:
            if enabled_only and not hook.enabled:
                continue
            if plugin_name and hook.plugin_name != plugin_name:
                continue
            hooks.append(hook)

        # Sort by priority
        hooks.sort(key=lambda h: (h.hook_point.value, h.priority))
        return hooks

    def get_hook(self, name: str) -> Optional[Hook]:
        """Get a hook by name.

        Args:
            name: The unique hook name.

        Returns:
            The Hook object, or None if not found.
        """
        return self._hooks_by_name.get(name)

    def enable(self, name: str) -> bool:
        """Enable a hook.

        Args:
            name: The hook name.

        Returns:
            True if the hook was found and enabled.
        """
        hook = self._hooks_by_name.get(name)
        if hook:
            hook.enabled = True
            logger.info("Hook '%s' enabled", name)
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a hook.

        Args:
            name: The hook name.

        Returns:
            True if the hook was found and disabled.
        """
        hook = self._hooks_by_name.get(name)
        if hook:
            hook.enabled = False
            logger.info("Hook '%s' disabled", name)
            return True
        return False

    def enable_by_plugin(self, plugin_name: str) -> int:
        """Enable all hooks for a plugin.

        Args:
            plugin_name: The plugin name.

        Returns:
            Number of hooks enabled.
        """
        count = 0
        for hook in self._hooks_by_name.values():
            if hook.plugin_name == plugin_name:
                hook.enabled = True
                count += 1
        return count

    def disable_by_plugin(self, plugin_name: str) -> int:
        """Disable all hooks for a plugin.

        Args:
            plugin_name: The plugin name.

        Returns:
            Number of hooks disabled.
        """
        count = 0
        for hook in self._hooks_by_name.values():
            if hook.plugin_name == plugin_name:
                hook.enabled = False
                count += 1
        return count

    def clear(self, hook_point: Optional[HookPoint] = None) -> int:
        """Remove all registered hooks.

        Args:
            hook_point: If specified, only clear hooks for this point.

        Returns:
            Number of hooks removed.
        """
        if hook_point:
            count = len(self._hooks.get(hook_point, []))
            for hook in self._hooks.get(hook_point, []):
                self._hooks_by_name.pop(hook.name, None)
            self._hooks[hook_point] = []
        else:
            count = len(self._hooks_by_name)
            self._hooks.clear()
            self._hooks_by_name.clear()

        logger.info("Cleared %d hooks", count)
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get hook system statistics.

        Returns:
            A dictionary with comprehensive statistics.
        """
        # Per-hook-point stats
        hook_point_stats: Dict[str, Dict[str, int]] = {}
        for point, hooks in self._hooks.items():
            enabled = sum(1 for h in hooks if h.enabled)
            hook_point_stats[point.value] = {
                "total": len(hooks),
                "enabled": enabled,
                "disabled": len(hooks) - enabled,
            }

        # Per-plugin stats
        plugin_stats: Dict[str, int] = {}
        for hook in self._hooks_by_name.values():
            if hook.plugin_name:
                plugin_stats[hook.plugin_name] = (
                    plugin_stats.get(hook.plugin_name, 0) + 1
                )

        return {
            "total_hooks": len(self._hooks_by_name),
            "hook_points": hook_point_stats,
            "plugins": plugin_stats,
            "global": {
                "total_executions": self._global_stats["total_executions"],
                "total_errors": self._global_stats["total_errors"],
                "avg_execution_time_ms": (
                    self._global_stats["total_execution_time_ms"]
                    / max(self._global_stats["total_executions"], 1)
                ),
            },
            "history_size": len(self._execution_history),
        }

    def get_hook_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific hook.

        Args:
            name: The hook name.

        Returns:
            A dictionary with hook statistics, or None.
        """
        hook = self._hooks_by_name.get(name)
        if hook:
            return hook.to_dict()
        return None

    def get_execution_history(
        self, limit: int = 50, hook_point: Optional[HookPoint] = None
    ) -> List[Dict[str, Any]]:
        """Get recent execution history.

        Args:
            limit: Maximum number of entries to return.
            hook_point: Filter by hook point.

        Returns:
            A list of execution history entries.
        """
        history = self._execution_history
        if hook_point:
            history = [
                h for h in history
                if h.get("hook_point") == hook_point.value
            ]

        return list(reversed(history[-limit:]))

    def _get_enabled_hooks(self, hook_point: HookPoint) -> List[Hook]:
        """Get enabled hooks for a hook point, sorted by priority.

        Args:
            hook_point: The hook point.

        Returns:
            A list of enabled Hook objects.
        """
        hooks = self._hooks.get(hook_point, [])
        return [h for h in hooks if h.enabled]

    async def _execute_single_hook(
        self, hook: Hook, context: HookContext
    ) -> HookResult:
        """Execute a single hook with error handling.

        Implements retry logic, timeout handling, and error isolation.

        Args:
            hook: The hook to execute.
            context: The hook context.

        Returns:
            A HookResult from the handler.
        """
        start_time = time.monotonic()

        # Update hook stats
        hook.call_count += 1
        hook.last_called = time.time()

        result = HookResult(
            handler_name=hook.name,
            priority=hook.priority,
        )

        # Execute with retries
        last_error: Optional[str] = None
        for attempt in range(hook.max_retries + 1):
            try:
                if hook.is_async:
                    coro_or_result = hook.handler(context)
                    if hook.timeout:
                        handler_result = await asyncio.wait_for(
                            coro_or_result, timeout=hook.timeout
                        )
                    else:
                        handler_result = await coro_or_result
                else:
                    # Run sync handler in executor
                    loop = asyncio.get_event_loop()
                    if hook.timeout:
                        handler_result = await asyncio.wait_for(
                            loop.run_in_executor(
                                None, functools.partial(hook.handler, context)
                            ),
                            timeout=hook.timeout,
                        )
                    else:
                        handler_result = await loop.run_in_executor(
                            None, functools.partial(hook.handler, context)
                        )

                # Process result
                if isinstance(handler_result, HookResult):
                    result = handler_result
                    result.handler_name = hook.name
                    result.priority = hook.priority
                elif isinstance(handler_result, dict):
                    result = HookResult(
                        **{k: v for k, v in handler_result.items()
                           if k in HookResult.__dataclass_fields__},
                        handler_name=hook.name,
                        priority=hook.priority,
                    )
                elif isinstance(handler_result, bool):
                    result.handled = handler_result
                elif handler_result is not None:
                    result.message = str(handler_result)

                # Success - break out of retry loop
                last_error = None
                break

            except asyncio.TimeoutError:
                last_error = f"Hook '{hook.name}' timed out after {hook.timeout}s"
                logger.error(last_error)
                if attempt == hook.max_retries:
                    result.error = last_error

            except asyncio.CancelledError:
                last_error = f"Hook '{hook.name}' was cancelled"
                logger.debug(last_error)
                result.error = last_error
                break

            except Exception as e:
                last_error = f"Hook '{hook.name}' error: {str(e)}"
                logger.error(last_error, exc_info=True)
                if attempt == hook.max_retries:
                    result.error = last_error

                if not self._error_isolation:
                    break

                # Wait before retry (exponential backoff)
                if attempt < hook.max_retries:
                    await asyncio.sleep(0.1 * (2 ** attempt))

        # Update stats
        execution_time_ms = (time.monotonic() - start_time) * 1000
        result.execution_time_ms = execution_time_ms
        hook.total_execution_time_ms += execution_time_ms

        if result.error:
            hook.error_count += 1
            self._global_stats["total_errors"] += 1

        return result

    def _record_execution(
        self,
        hook_point: HookPoint,
        hook_count: int,
        result_count: int,
        total_time_ms: float,
        cancelled: bool,
    ) -> None:
        """Record an execution event in the history.

        Args:
            hook_point: The hook point that was executed.
            hook_count: Number of hooks executed.
            result_count: Number of results returned.
            total_time_ms: Total execution time.
            cancelled: Whether the execution was cancelled.
        """
        entry = {
            "hook_point": hook_point.value,
            "hook_count": hook_count,
            "result_count": result_count,
            "total_time_ms": round(total_time_ms, 2),
            "cancelled": cancelled,
            "timestamp": time.time(),
        }

        self._execution_history.append(entry)

        # Trim history if too large
        if len(self._execution_history) > self._max_history_size:
            self._execution_history = (
                self._execution_history[-self._max_history_size:]
            )


# Decorator for registering hooks
def on_hook(
    hook_point: HookPoint,
    priority: Union[int, HookPriority] = HookPriority.NORMAL,
    name: Optional[str] = None,
    plugin_name: str = "",
    description: str = "",
) -> Callable[[HookHandler], HookHandler]:
    """Decorator to register a function as a hook handler.

    Usage::

        hook_system = HookSystem()

        @on_hook(HookPoint.ON_MESSAGE, priority=HookPriority.HIGH)
        async def my_message_handler(context: HookContext) -> HookResult:
            print(f"Got message: {context.get('message')}")
            return HookResult()

        # The decorator stores metadata; you still need to register
        # with the hook system:
        hook_system.register(
            HookPoint.ON_MESSAGE,
            my_message_handler,
            name="my_message_handler",
            priority=HookPriority.HIGH,
        )

    Args:
        hook_point: The hook point to register for.
        priority: Execution priority.
        name: Optional hook name.
        plugin_name: Plugin name.
        description: Hook description.

    Returns:
        The decorated function with added metadata.
    """
    def decorator(func: HookHandler) -> HookHandler:
        func._atlas_hook_point = hook_point  # type: ignore[attr-defined]
        func._atlas_hook_priority = (  # type: ignore[attr-defined]
            priority.value if isinstance(priority, HookPriority) else priority
        )
        func._atlas_hook_name = name  # type: ignore[attr-defined]
        func._atlas_hook_plugin = plugin_name  # type: ignore[attr-defined]
        func._atlas_hook_desc = description  # type: ignore[attr-defined]
        return func

    return decorator


def register_hooks(
    hook_system: HookSystem,
    module: Any,
    plugin_name: str = "",
) -> int:
    """Auto-discover and register all hook-decorated functions in a module.

    Scans a module for functions decorated with @on_hook and registers
    them with the given HookSystem.

    Args:
        hook_system: The hook system to register with.
        module: The module to scan.
        plugin_name: Plugin name prefix.

    Returns:
        Number of hooks registered.
    """
    count = 0

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if not callable(attr):
            continue

        hook_point = getattr(attr, "_atlas_hook_point", None)
        if hook_point is None:
            continue

        hook_system.register(
            hook_point=hook_point,
            handler=attr,
            priority=getattr(attr, "_atlas_hook_priority", HookPriority.NORMAL.value),
            name=getattr(attr, "_atlas_hook_name", None) or attr_name,
            plugin_name=plugin_name or getattr(attr, "_atlas_hook_plugin", ""),
            description=getattr(attr, "_atlas_hook_desc", ""),
        )
        count += 1

    if count > 0:
        logger.info("Auto-registered %d hooks from module", count)

    return count
