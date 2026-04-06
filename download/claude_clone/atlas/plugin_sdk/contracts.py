"""
Plugin Contracts — Abstract base classes that define the plugin API surface.

Every Atlas plugin must implement (or subclass) one of the contract classes
defined in this module.  The contract system ensures a uniform lifecycle,
a consistent interface for the loader, and clear expectations for what each
plugin type contributes.

Architecture
------------
```
                  BasePlugin  (lifecycle + identity)
                  /    |    \
                 /     |     \
        ToolPlugin  ChannelPlugin  ProviderPlugin
              |           |              |
        CommandPlugin    HookPlugin
```

Only the leaf classes (``ToolPlugin``, ``ChannelPlugin``, etc.) are
*directly* instantiated by the loader.  Intermediate ABCs provide shared
behaviour through mixins.

Usage
-----
In a plugin's ``entry_point`` module::

    from atlas.plugin_sdk.contracts import ToolPlugin, tool

    class MyPlugin(ToolPlugin):
        @tool(name="hello", description="Say hello")
        async def hello(self, name: str = "world") -> str:
            return f"Hello, {name}!"

    setup = MyPlugin  # <- loader calls setup() to instantiate
"""

from __future__ import annotations

import abc
import asyncio
import inspect
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncCallable,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from atlas.plugin_sdk.core import (
    PluginCapability,
    PluginManifest,
    PluginPermission,
    PluginState,
    PluginInfo,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decorators for declaring tools and hooks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDefinition:
    """Metadata for a single tool exposed by a plugin.

    Attributes:
        name: Unique tool name within the plugin (prefixed at runtime).
        description: Human-readable summary shown to the model.
        parameters: JSON-Schema-style dict describing expected inputs.
        handler: The actual async/sync callable that implements the tool.
        is_async: Whether the handler is a coroutine function.
        timeout: Maximum execution time in seconds (0 = no limit).
        required_permissions: Permissions the tool needs beyond the plugin's base set.
    """

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    handler: Optional[Callable[..., Any]] = None
    is_async: bool = False
    timeout: float = 0.0
    required_permissions: Tuple[PluginPermission, ...] = ()

    def __post_init__(self) -> None:
        if self.handler is not None:
            object.__setattr__(self, "is_async", asyncio.iscoroutinefunction(self.handler))


def tool(
    name: str,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
    timeout: float = 0.0,
) -> Callable[[Callable[..., Any]], ToolDefinition]:
    """Decorator that marks a method as a Claude-callable tool.

    Example::

        @tool(name="echo", description="Echo the input")
        async def echo(self, text: str) -> str:
            return text

    The decorated method is **replaced** by the :class:`ToolDefinition`
    descriptor so that the loader can introspect it.
    """
    def decorator(func: Callable[..., Any]) -> ToolDefinition:
        sig = inspect.signature(func)
        props: Dict[str, Any] = {}
        required: List[str] = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else "string"
            props[pname] = {"type": _python_type_to_json(annotation), "description": pname}
            if param.default == inspect.Parameter.empty:
                required.append(pname)
        schema = parameters or {
            "type": "object",
            "properties": props,
            "required": required,
        }
        return ToolDefinition(
            name=name,
            description=description or func.__doc__ or "",
            parameters=schema,
            handler=func,
            is_async=asyncio.iscoroutinefunction(func),
            timeout=timeout,
        )
    return decorator


@dataclass(frozen=True)
class HookDefinition:
    """Metadata for a lifecycle/event hook provided by a plugin.

    Attributes:
        event: The event name to subscribe to.
        handler: Async callable invoked when the event fires.
        priority: Lower values run earlier (default 100).
        once: If ``True``, the hook auto-removes after the first invocation.
    """

    event: str
    handler: Optional[Callable[..., Awaitable[Any]]] = None
    priority: int = 100
    once: bool = False


def hook(
    event: str,
    priority: int = 100,
    once: bool = False,
) -> Callable[[Callable[..., Awaitable[Any]]], HookDefinition]:
    """Decorator that marks a method as an event hook.

    Example::

        @hook(event="session.start", priority=50)
        async def on_session_start(self, session_id: str) -> None:
            logger.info("Session %s started", session_id)
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> HookDefinition:
        return HookDefinition(
            event=event,
            handler=func,
            priority=priority,
            once=once,
        )
    return decorator


@dataclass(frozen=True)
class CommandDefinition:
    """Metadata for a CLI command exposed by a plugin.

    Attributes:
        name: Command name (e.g. ``"deploy"``).
        description: Help text shown in ``atlas --help``.
        handler: Callable that implements the command.
        aliases: Alternative names for the command.
    """

    name: str
    description: str = ""
    handler: Optional[Callable[..., Any]] = None
    aliases: Tuple[str, ...] = ()


def command(
    name: str,
    description: str = "",
    aliases: Optional[Sequence[str]] = None,
) -> Callable[[Callable[..., Any]], CommandDefinition]:
    """Decorator that marks a method as a CLI sub-command.

    Example::

        @command(name="status", description="Show plugin status")
        async def status(self) -> str:
            return "OK"
    """
    def decorator(func: Callable[..., Any]) -> CommandDefinition:
        return CommandDefinition(
            name=name,
            description=description or func.__doc__ or "",
            handler=func,
            aliases=tuple(aliases or ()),
        )
    return decorator


def _python_type_to_json(annotation: Any) -> str:
    """Map a Python type annotation to a JSON Schema type string."""
    origin = getattr(annotation, "__origin__", None)
    if origin is list or origin is List:
        return "array"
    if origin is dict or origin is Dict:
        return "object"
    if origin is set or origin is Set:
        return "array"
    mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
    }
    return mapping.get(annotation, "string")


# ---------------------------------------------------------------------------
# Base Plugin
# ---------------------------------------------------------------------------

class BasePlugin(abc.ABC):
    """Abstract base class for all Atlas plugins.

    Every plugin **must** subclass :class:`BasePlugin` and implement the
    required lifecycle methods.  The loader calls these methods in order:

    1. ``on_register(manifest)`` — receives the validated manifest.
    2. ``on_load()`` — initialize resources, open connections, etc.
    3. (plugin is now ACTIVE)
    4. ``on_unload()`` — release resources when disabled/unloaded.

    Attributes:
        manifest: The validated :class:`PluginManifest` (set by the loader).
        plugin_info: Live :class:`PluginInfo` tracking runtime state.
        logger: A ``logging.Logger`` named after the plugin.
        _tools_cache: Internal cache of discovered tool definitions.
        _hooks_cache: Internal cache of discovered hook definitions.
        _commands_cache: Internal cache of discovered command definitions.
    """

    def __init__(self) -> None:
        self.manifest: Optional[PluginManifest] = None
        self.plugin_info: Optional[PluginInfo] = None
        self.logger: logging.Logger = logging.getLogger(
            f"atlas.plugin.{type(self).__module__}"
        )
        self._tools_cache: Optional[List[ToolDefinition]] = None
        self._hooks_cache: Optional[List[HookDefinition]] = None
        self._commands_cache: Optional[List[CommandDefinition]] = None
        self._initialized = False

    # -- Lifecycle methods (override in subclasses) ------------------------

    @abc.abstractmethod
    def on_register(self, manifest: PluginManifest) -> None:
        """Called when the plugin is registered with the plugin registry.

        Use this to store the manifest and perform early validation.
        """
        ...

    @abc.abstractmethod
    def on_load(self) -> None:
        """Called when the plugin is activated.

        Initialize resources, spawn background tasks, register event
        listeners, etc.
        """
        ...

    def on_unload(self) -> None:
        """Called when the plugin is deactivated or unloaded.

        Release resources, cancel background tasks, etc.
        The default implementation is a no-op.
        """
        self.logger.info("Plugin %s unloaded (no custom cleanup)", self.__class__.__name__)

    def on_error(self, error: Exception) -> None:
        """Called when an unhandled error occurs in the plugin.

        Override to implement custom error reporting / recovery.
        """
        self.logger.error(
            "Unhandled error in plugin %s: %s\n%s",
            self.__class__.__name__,
            error,
            traceback.format_exc(),
        )

    # -- Introspection helpers ---------------------------------------------

    @property
    def name(self) -> str:
        """Return the plugin's manifest name, or the class name as fallback."""
        return self.manifest.name if self.manifest else type(self).__name__

    @property
    def version(self) -> str:
        """Return the plugin's manifest version, or ``"0.0.0"`` as fallback."""
        return self.manifest.version if self.manifest else "0.0.0"

    @property
    def is_initialized(self) -> bool:
        """Return ``True`` after ``on_load`` has completed successfully."""
        return self._initialized

    def get_tools(self) -> List[ToolDefinition]:
        """Return all :class:`ToolDefinition` instances declared on this plugin.

        Results are cached after the first call.
        """
        if self._tools_cache is not None:
            return self._tools_cache
        self._tools_cache = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, ToolDefinition):
                self._tools_cache.append(attr)
        return self._tools_cache

    def get_hooks(self) -> List[HookDefinition]:
        """Return all :class:`HookDefinition` instances declared on this plugin."""
        if self._hooks_cache is not None:
            return self._hooks_cache
        self._hooks_cache = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, HookDefinition):
                self._hooks_cache.append(attr)
        return self._hooks_cache

    def get_commands(self) -> List[CommandDefinition]:
        """Return all :class:`CommandDefinition` instances declared on this plugin."""
        if self._commands_cache is not None:
            return self._commands_cache
        self._commands_cache = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, CommandDefinition):
                self._commands_cache.append(attr)
        return self._commands_cache

    def invalidate_cache(self) -> None:
        """Clear introspection caches (useful after hot-reload)."""
        self._tools_cache = None
        self._hooks_cache = None
        self._commands_cache = None

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary dict for logging / debugging."""
        return {
            "class": type(self).__name__,
            "name": self.name,
            "version": self.version,
            "module": type(self).__module__,
            "tools": len(self.get_tools()),
            "hooks": len(self.get_hooks()),
            "commands": len(self.get_commands()),
            "initialized": self._initialized,
        }

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} name={self.name!r} "
            f"version={self.version!r} initialized={self._initialized}>"
        )


# ---------------------------------------------------------------------------
# Tool Plugin
# ---------------------------------------------------------------------------

class ToolPlugin(BasePlugin):
    """A plugin that provides Claude-callable tools.

    Tool plugins use the ``@tool`` decorator to declare their tools::

        class MyTools(ToolPlugin):
            @tool(name="search", description="Search the web")
            async def search(self, query: str) -> str:
                return await web_search(query)

    The loader introspects the class and extracts every ``ToolDefinition``.
    """

    @property
    def capability(self) -> PluginCapability:
        """Return :data:`PluginCapability.TOOLS`."""
        return PluginCapability.TOOLS

    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Alias for :meth:`BasePlugin.get_tools` with a more specific name."""
        return self.get_tools()

    async def invoke_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke a named tool by looking it up in the plugin's tool registry.

        Raises:
            KeyError: If *tool_name* is not found.
            Exception: If the tool handler raises.
        """
        for tdef in self.get_tools():
            if tdef.name == tool_name and tdef.handler is not None:
                try:
                    if tdef.is_async:
                        return await tdef.handler(self, **kwargs)
                    return tdef.handler(self, **kwargs)
                except Exception as exc:
                    self.on_error(exc)
                    raise
        raise KeyError(f"Tool {tool_name!r} not found in plugin {self.name!r}")


# ---------------------------------------------------------------------------
# Channel Plugin
# ---------------------------------------------------------------------------

class ChannelPlugin(BasePlugin):
    """A plugin that provides a messaging channel (Slack, Discord, etc.).

    Channel plugins must implement :meth:`start` and :meth:`stop` to
    manage their connection lifecycle independently of the core on_load /
    on_unload hooks.
    """

    @property
    def capability(self) -> PluginCapability:
        """Return :data:`PluginCapability.CHANNELS`."""
        return PluginCapability.CHANNELS

    @abc.abstractmethod
    async def start(self) -> None:
        """Open the channel connection (e.g. connect to Slack WebSocket)."""
        ...

    @abc.abstractmethod
    async def stop(self) -> None:
        """Gracefully close the channel connection."""
        ...

    async def send_message(self, channel_id: str, message: str, **kwargs: Any) -> Any:
        """Send a message to a channel.  Override for custom behaviour."""
        raise NotImplementedError(
            f"Channel plugin {self.name!r} must implement send_message()"
        )

    async def on_message(self, raw_message: Dict[str, Any]) -> Optional[str]:
        """Handle an incoming message.  Return a reply or ``None``."""
        return None


# ---------------------------------------------------------------------------
# Provider Plugin
# ---------------------------------------------------------------------------

class ProviderPlugin(BasePlugin):
    """A plugin that provides an AI model provider backend.

    Provider plugins expose new model backends that Claude can route to
    via the smart-routing engine.
    """

    @property
    def capability(self) -> PluginCapability:
        """Return :data:`PluginCapability.PROVIDERS`."""
        return PluginCapability.PROVIDERS

    @abc.abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """Return a list of available models from this provider.

        Each model dict should contain at least ``name``, ``id``,
        and optionally ``context_window``, ``pricing``, etc.
        """
        ...

    @abc.abstractmethod
    async def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a chat-completion request to the provider.

        Args:
            model: Model identifier.
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.).

        Returns:
            A dict with at least ``content`` and ``usage`` keys.
        """
        ...

    def health_check(self) -> Dict[str, Any]:
        """Return health status for monitoring dashboards.

        Returns:
            A dict with ``status`` ("ok" | "degraded" | "down") and
            optional ``latency_ms``, ``error`` keys.
        """
        return {"status": "ok", "provider": self.name}


# ---------------------------------------------------------------------------
# Command Plugin
# ---------------------------------------------------------------------------

class CommandPlugin(BasePlugin):
    """A plugin that provides CLI sub-commands for Atlas.

    Use the ``@command`` decorator to declare commands.
    """

    @property
    def capability(self) -> PluginCapability:
        """Return :data:`PluginCapability.COMMANDS`."""
        return PluginCapability.COMMANDS

    def get_command_definitions(self) -> List[CommandDefinition]:
        """Alias for :meth:`BasePlugin.get_commands`."""
        return self.get_commands()

    async def execute_command(self, command_name: str, **kwargs: Any) -> Any:
        """Execute a named command.

        Raises:
            KeyError: If *command_name* is not found.
        """
        for cdef in self.get_commands():
            if cdef.name == command_name and cdef.handler is not None:
                try:
                    if asyncio.iscoroutinefunction(cdef.handler):
                        return await cdef.handler(self, **kwargs)
                    return cdef.handler(self, **kwargs)
                except Exception as exc:
                    self.on_error(exc)
                    raise
        raise KeyError(f"Command {command_name!r} not found in plugin {self.name!r}")


# ---------------------------------------------------------------------------
# Hook Plugin
# ---------------------------------------------------------------------------

class HookPlugin(BasePlugin):
    """A plugin that provides lifecycle and event hooks.

    Use the ``@hook`` decorator to subscribe to events.
    """

    @property
    def capability(self) -> PluginCapability:
        """Return :data:`PluginCapability.HOOKS`."""
        return PluginCapability.HOOKS

    def get_hook_definitions(self) -> List[HookDefinition]:
        """Alias for :meth:`BasePlugin.get_hooks`."""
        return self.get_hooks()

    async def emit(self, event: str, payload: Any = None) -> List[Any]:
        """Run all handlers subscribed to *event* and collect results."""
        results: List[Any] = []
        hooks = sorted(
            [h for h in self.get_hooks() if h.event == event],
            key=lambda h: h.priority,
        )
        for hdef in hooks:
            if hdef.handler is None:
                continue
            try:
                if asyncio.iscoroutinefunction(hdef.handler):
                    result = await hdef.handler(self, payload)
                else:
                    result = hdef.handler(self, payload)
                results.append(result)
                if hdef.once:
                    # Mark for removal (lazy cleanup)
                    object.__setattr__(hdef, "handler", None)
            except Exception as exc:
                self.on_error(exc)
        return results


# ---------------------------------------------------------------------------
# Plugin factory helper
# ---------------------------------------------------------------------------

# Registry of capability → contract class mapping
CAPABILITY_CONTRACT_MAP: Dict[PluginCapability, Type[BasePlugin]] = {
    PluginCapability.TOOLS: ToolPlugin,
    PluginCapability.CHANNELS: ChannelPlugin,
    PluginCapability.PROVIDERS: ProviderPlugin,
    PluginCapability.COMMANDS: CommandPlugin,
    PluginCapability.HOOKS: HookPlugin,
}


def get_contract_class(capability: PluginCapability) -> Type[BasePlugin]:
    """Return the base contract class for a given capability.

    Raises:
        ValueError: If *capability* is not mapped.
    """
    cls = CAPABILITY_CONTRACT_MAP.get(capability)
    if cls is None:
        raise ValueError(f"No contract class registered for capability {capability!r}")
    return cls


def resolve_plugin_class(obj: Any) -> Optional[Type[BasePlugin]]:
    """Determine the most specific contract class for *obj*.

    Walks the MRO and returns the first match in
    :data:`CAPABILITY_CONTRACT_MAP`.

    Returns ``None`` if *obj* is not a :class:`BasePlugin` subclass.
    """
    if not isinstance(obj, type):
        obj = type(obj)
    for klass in obj.__mro__:
        if klass in CAPABILITY_CONTRACT_MAP.values():
            return klass
    if issubclass(obj, BasePlugin):
        return BasePlugin
    return None
