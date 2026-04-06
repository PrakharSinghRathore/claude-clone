"""
Hermes Bridge — Central integration point between Claude Clone and the Hermes Agent system.

This module provides a single, configuration-driven entry point for initializing
all Hermes subsystems and exposing them to the Claude Clone agent.  Every
subsystem import is wrapped in ``try / except`` so that a missing dependency
or a misconfigured component never crashes the bridge.

Usage
-----
    from hermes_bridge import HermesBridge

    bridge = HermesBridge()
    bridge.initialize(config)

    # Get Hermes tools to merge with Claude Clone tools
    hermes_tools = bridge.get_tools()
    agent.tools.update(hermes_tools)

    # Compress long conversations
    messages = await bridge.compress_context(messages, model="claude-sonnet-4")

    # Smart routing
    route = await bridge.get_smart_route("code_generation", "Write a REST API")
    print(f"Best model: {route.model_name}")

    # Cleanup
    bridge.shutdown()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration schema
# ---------------------------------------------------------------------------

@dataclass
class SubsystemConfig:
    """Generic toggling config for a single subsystem."""

    enabled: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HermesConfig:
    """
    Top-level configuration for the Hermes Bridge.

    Each key maps to a subsystem section.  The bridge will only attempt to
    initialise a subsystem when ``enabled: True`` is set in its section.
    """

    # Core subsystems
    context_compression: SubsystemConfig = field(default_factory=lambda: SubsystemConfig(enabled=True))
    prompt_builder: SubsystemConfig = field(default_factory=lambda: SubsystemConfig(enabled=True))
    smart_routing: SubsystemConfig = field(default_factory=lambda: SubsystemConfig(enabled=True))
    insights: SubsystemConfig = field(default_factory=lambda: SubsystemConfig(enabled=True))
    trajectory: SubsystemConfig = field(default_factory=lambda: SubsystemConfig(enabled=True))

    # Tools
    tools: SubsystemConfig = field(default_factory=lambda: SubsystemConfig(enabled=True))

    # Memory
    memory: SubsystemConfig = field(default_factory=lambda: SubsystemConfig(enabled=True))

    # Skills
    skills: SubsystemConfig = field(default_factory=lambda: SubsystemConfig(enabled=True))

    # Cron scheduler
    cron: SubsystemConfig = field(default_factory=lambda: SubsystemConfig(enabled=False))

    # Gateway (initialise but do NOT start)
    gateway: SubsystemConfig = field(default_factory=lambda: SubsystemConfig(enabled=False))

    # ACP server (initialise but do NOT start)
    acp: SubsystemConfig = field(default_factory=lambda: SubsystemConfig(enabled=False))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HermesConfig":
        """
        Build a ``HermesConfig`` from a plain dict.

        Top-level keys must be subsystem names.  Each value may be ``True``,
        ``False``, or a dict with at least an ``enabled`` key.

        Example::

            cfg = HermesConfig.from_dict({
                "tools": {"enabled": True},
                "memory": True,
                "gateway": {"enabled": True, "extra": {"config_path": "gw.yaml"}},
            })
        """
        known_subsystems = set(cls.__dataclass_fields__.keys())
        config = cls()
        for key, value in data.items():
            if key not in known_subsystems:
                logger.warning("Unknown Hermes subsystem in config: %r — ignoring.", key)
                continue
            if isinstance(value, bool):
                setattr(config, key, SubsystemConfig(enabled=value))
            elif isinstance(value, dict):
                enabled = value.pop("enabled", False)
                setattr(config, key, SubsystemConfig(enabled=enabled, extra=value))
            else:
                logger.warning("Unsupported config value for %r: %s", key, type(value))
        return config


# ---------------------------------------------------------------------------
# Subsystem status helper
# ---------------------------------------------------------------------------

@dataclass
class _SubsystemStatus:
    """Tracks the state of a single subsystem inside the bridge."""

    name: str
    initialized: bool = False
    available: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "initialized": self.initialized,
            "available": self.available,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# HermesBridge
# ---------------------------------------------------------------------------

class HermesBridge:
    """
    Singleton facade that manages the entire Hermes lifecycle.

    The bridge lazily discovers and initialises each Hermes subsystem based
    on the provided :class:`HermesConfig`.  It exposes a high-level API
    suitable for direct consumption by the Claude Clone agent loop.

    Design goals:

    * **Configuration-driven** — each subsystem is only created when its
      config section has ``enabled: True``.
    * **Graceful degradation** — every subsystem import is wrapped in
      ``try / except``.  Failures are logged but never propagated.
    * **Thread-safe singleton** — a module-level ``_instance`` is guarded by
      a ``threading.Lock``.
    """

    _instance: Optional["HermesBridge"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "HermesBridge":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._reset_state()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton so a fresh instance can be created (useful for tests)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Initialise / reset all internal state to empty defaults."""
        self._config: Optional[HermesConfig] = None
        self._initialized = False

        # Subsystem references
        self._tool_registry: Any = None
        self._compressor: Any = None
        self._prompt_builder: Any = None
        self._smart_router: Any = None
        self._insights: Any = None
        self._trajectory_recorder: Any = None
        self._memory_manager: Any = None
        self._skill_manager: Any = None
        self._cron_scheduler: Any = None
        self._gateway_runner: Any = None
        self._acp_server: Any = None

        # Subsystem status map
        self._status: Dict[str, _SubsystemStatus] = {
            name: _SubsystemStatus(name=name)
            for name in (
                "context_compression", "prompt_builder", "smart_routing",
                "insights", "trajectory", "tools", "memory",
                "skills", "cron", "gateway", "acp",
            )
        }

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self, config: Any) -> None:
        """
        Initialise all Hermes subsystems based on *config*.

        Parameters
        ----------
        config:
            Either a :class:`HermesConfig` instance, or a plain ``dict`` that
            will be passed through :meth:`HermesConfig.from_dict`.
        """
        if self._initialized:
            logger.info("HermesBridge already initialised — skipping.")
            return

        if isinstance(config, dict):
            self._config = HermesConfig.from_dict(config)
        elif isinstance(config, HermesConfig):
            self._config = config
        else:
            raise TypeError(f"config must be a dict or HermesConfig, got {type(config)}")

        logger.info("Initialising HermesBridge …")

        # Core subsystems
        self._init_tools()
        self._init_context_compression()
        self._init_prompt_builder()
        self._init_smart_routing()
        self._init_insights()
        self._init_trajectory()
        self._init_memory()
        self._init_skills()
        self._init_cron()
        self._init_gateway()
        self._init_acp()

        self._initialized = True
        logger.info("HermesBridge initialised (subsystems: %s)", self._status_summary())

    # ------------------------------------------------------------------
    # Public API — Tools
    # ------------------------------------------------------------------

    def get_tools(self) -> Dict[str, Callable]:
        """
        Return all Hermes tools merged into Claude Clone format.

        Returns
        -------
        dict[str, Callable]
            Mapping of tool name → async function, compatible with the
            existing ``Agent(tools=...)`` constructor.
        """
        if self._tool_registry is not None:
            try:
                return self._tool_registry.get_tools_dict(enabled_only=True)
            except Exception as exc:
                logger.warning("Failed to get tools dict: %s", exc)
        return {}

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Return Hermes tool schemas in Anthropic format.

        Returns
        -------
        list[dict]
            Each dict has ``name``, ``description``, and ``input_schema`` keys.
        """
        if self._tool_registry is not None:
            try:
                return self._tool_registry.get_schemas(enabled_only=True)
            except Exception as exc:
                logger.warning("Failed to get tool schemas: %s", exc)
        return []

    # ------------------------------------------------------------------
    # Public API — Memory
    # ------------------------------------------------------------------

    async def get_memory_context(
        self,
        query: str,
        max_tokens: int = 4000,
    ) -> str:
        """
        Retrieve relevant memory context for the current query.

        Parameters
        ----------
        query:
            The user's prompt text.
        max_tokens:
            Token budget for the retrieved context.

        Returns
        -------
        str
            Formatted context block for prompt injection, or ``""`` if memory
            is not available.
        """
        if self._memory_manager is None:
            return ""
        try:
            return await self._memory_manager.prefetch(query, max_tokens=max_tokens)
        except Exception as exc:
            logger.warning("Memory prefetch failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Public API — Context Compression
    # ------------------------------------------------------------------

    async def compress_context(
        self,
        messages: List[Dict[str, Any]],
        model: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Compress a conversation to fit within the model's context window.

        Parameters
        ----------
        messages:
            The full conversation message list.
        model:
            Model name (used to select an appropriate context budget).  If
            empty, uses the default budget of 200 000 tokens.

        Returns
        -------
        list[dict]
            The (possibly compressed) message list.
        """
        if self._compressor is None:
            return messages
        try:
            # Map well-known model names to context budgets
            budget = _context_budget_for_model(model)
            result = await self._compressor.compress(
                messages,
                system_prompt=None,
                strategy=self._compressor.CompressionStrategy.AUTO,
            )
            return result.messages
        except Exception as exc:
            logger.warning("Context compression failed: %s", exc)
            return messages

    # ------------------------------------------------------------------
    # Public API — Smart Routing
    # ------------------------------------------------------------------

    async def get_smart_route(
        self,
        task_type: str,
        prompt: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Get a smart routing decision for the given task.

        Parameters
        ----------
        task_type:
            Explicit task type (e.g. ``"reasoning"``, ``"code"``).  May be
            empty for auto-classification from *prompt*.
        prompt:
            The user prompt for classification.
        constraints:
            Optional dict of routing constraints forwarded to
            :class:`~hermes.core.smart_routing.RoutingConstraint`.

        Returns
        -------
        RoutingDecision
            The routing decision with ``model_name``, ``reason``, etc.
        """
        if self._smart_router is None:
            from hermes.core.smart_routing import RoutingDecision
            return RoutingDecision(
                model_name="claude-sonnet-4-20250514",
                reason="Smart routing not initialised; using default model.",
                confidence=0.3,
            )
        try:
            constraint_obj = None
            if constraints:
                from hermes.core.smart_routing import RoutingConstraint
                constraint_obj = RoutingConstraint(**{
                    k: v for k, v in constraints.items()
                    if k in RoutingConstraint.__dataclass_fields__
                })
            return await self._smart_router.route(
                task_type=task_type,
                prompt=prompt,
                constraints=constraint_obj,
            )
        except Exception as exc:
            logger.warning("Smart routing failed: %s", exc)
            from hermes.core.smart_routing import RoutingDecision
            return RoutingDecision(
                model_name="claude-sonnet-4-20250514",
                reason=f"Smart routing error: {exc}; using default model.",
                confidence=0.2,
            )

    # ------------------------------------------------------------------
    # Public API — Insights
    # ------------------------------------------------------------------

    def record_insights(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Record usage insights.

        Parameters
        ----------
        event_type:
            One of ``"usage"``, ``"tool_usage"``, ``"error"``, or any
            custom event type.
        data:
            Event payload.  The dict keys depend on *event_type*:

            * ``"usage"``: ``model``, ``input_tokens``, ``output_tokens``,
              ``cost_usd``, ``duration_ms``, ``session_id``, ``tool_calls_count``
            * ``"tool_usage"``: ``tool_name``, ``duration_ms``, ``success``
            * ``"error"``: ``model``, ``error_type``, ``error_message``
        """
        if self._insights is None:
            return
        try:
            if event_type == "usage":
                self._insights.record_usage(**data)
            elif event_type == "tool_usage":
                self._insights.record_tool_usage(**data)
            elif event_type == "error":
                self._insights.record_error(**data)
            else:
                logger.debug("Unknown insights event_type: %r", event_type)
        except Exception as exc:
            logger.warning("Failed to record insights (%s): %s", event_type, exc)

    # ------------------------------------------------------------------
    # Public API — Trajectory
    # ------------------------------------------------------------------

    def record_trajectory(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Record trajectory data for RL training data collection.

        Parameters
        ----------
        event_type:
            One of ``"start_turn"``, ``"end_turn"``, ``"tool_call"``,
            ``"tool_result"``, ``"model_response"``.
        data:
            Event payload whose contents depend on *event_type*.
        """
        if self._trajectory_recorder is None:
            return
        try:
            if event_type == "start_turn":
                self._trajectory_recorder.start_turn(data.get("user_message", ""))
            elif event_type == "end_turn":
                self._trajectory_recorder.end_turn(
                    model_response=data.get("model_response"),
                    input_tokens=data.get("input_tokens", 0),
                    output_tokens=data.get("output_tokens", 0),
                    model_name=data.get("model_name"),
                    cost_usd=data.get("cost_usd", 0.0),
                )
            elif event_type == "tool_call":
                self._trajectory_recorder.add_tool_call(
                    tool_name=data.get("tool_name", ""),
                    tool_input=data.get("tool_input", {}),
                    tool_id=data.get("tool_id", ""),
                )
            elif event_type == "tool_result":
                self._trajectory_recorder.add_tool_result(
                    tool_id=data.get("tool_id", ""),
                    tool_name=data.get("tool_name", ""),
                    result=data.get("result", ""),
                    is_error=data.get("is_error", False),
                )
            elif event_type == "model_response":
                self._trajectory_recorder.set_model_response(data.get("response", ""))
            else:
                logger.debug("Unknown trajectory event_type: %r", event_type)
        except Exception as exc:
            logger.warning("Failed to record trajectory (%s): %s", event_type, exc)

    # ------------------------------------------------------------------
    # Public API — Cron
    # ------------------------------------------------------------------

    def get_cron_jobs(self) -> List[Dict[str, Any]]:
        """
        List registered cron jobs.

        Returns
        -------
        list[dict]
            Each dict describes a single job with at least ``id``, ``name``,
            ``status``, and ``cron_expr`` keys.
        """
        if self._cron_scheduler is None:
            return []
        try:
            jobs = self._cron_scheduler._job_manager._jobs.values()
            return [
                {
                    "id": j.id,
                    "name": j.name,
                    "status": j.status.value if hasattr(j.status, "value") else str(j.status),
                    "cron_expr": j.cron_expr,
                    "schedule_type": j.schedule_type,
                    "priority": j.priority,
                    "next_run": str(j.next_run) if j.next_run else None,
                }
                for j in jobs
            ]
        except Exception as exc:
            logger.warning("Failed to list cron jobs: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Public API — Skills
    # ------------------------------------------------------------------

    def get_skills(self) -> List[Dict[str, Any]]:
        """
        List installed skills.

        Returns
        -------
        list[dict]
            Each dict describes a skill with ``name``, ``description``,
            ``category``, ``enabled``, and ``tags``.
        """
        if self._skill_manager is None:
            return []
        try:
            return [
                {
                    "name": s.name,
                    "description": s.description or "",
                    "category": s.category or "",
                    "enabled": s.enabled,
                    "tags": list(s.tags or []),
                }
                for s in self._skill_manager.list_all()
            ]
        except Exception as exc:
            logger.warning("Failed to list skills: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Public API — Gateway
    # ------------------------------------------------------------------

    def get_gateway_status(self) -> Dict[str, Any]:
        """
        Return the current gateway status.

        Returns
        -------
        dict
            Gateway status report, or a stub dict if the gateway was not
            initialised.
        """
        if self._gateway_runner is None:
            return {"status": "not_initialised"}
        try:
            return self._gateway_runner.get_status()
        except Exception as exc:
            logger.warning("Failed to get gateway status: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Public API — ACP
    # ------------------------------------------------------------------

    def get_acp_status(self) -> Dict[str, Any]:
        """
        Return the current ACP server status.

        Returns
        -------
        dict
            ACP server status information.
        """
        if self._acp_server is None:
            return {"status": "not_initialised"}
        try:
            event_stats = self._acp_server.events.stats() if self._acp_server.events else {}
            return {
                "status": "initialised",
                "host": self._acp_server.host,
                "port": self._acp_server.port,
                "sessions": len(getattr(self._acp_server.sessions, "_sessions", {})),
                "events": event_stats,
            }
        except Exception as exc:
            logger.warning("Failed to get ACP status: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """
        Return a dict with the status of every subsystem.

        Each entry contains ``initialized``, ``available``, and ``error``
        fields.
        """
        return {
            "initialized": self._initialized,
            "subsystems": {name: st.to_dict() for name, st in self._status.items()},
        }

    def _status_summary(self) -> str:
        """Compact summary of subsystem statuses for logging."""
        parts: list[str] = []
        for name, st in self._status.items():
            if st.initialized and st.available:
                parts.append(f"{name}=ok")
            elif st.initialized and not st.available:
                parts.append(f"{name}=unavailable")
            elif st.error:
                parts.append(f"{name}=err")
            else:
                parts.append(f"{name}=off")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Direct subsystem access
    # ------------------------------------------------------------------

    @property
    def tool_registry(self) -> Any:
        """Direct access to the underlying :class:`ToolRegistry` (or ``None``)."""
        return self._tool_registry

    @property
    def memory_manager(self) -> Any:
        """Direct access to the underlying :class:`MemoryManager` (or ``None``)."""
        return self._memory_manager

    @property
    def skill_manager(self) -> Any:
        """Direct access to the underlying :class:`SkillManager` (or ``None``)."""
        return self._skill_manager

    @property
    def insights_manager(self) -> Any:
        """Direct access to the underlying :class:`InsightsManager` (or ``None``)."""
        return self._insights

    @property
    def trajectory_recorder(self) -> Any:
        """Direct access to the underlying :class:`TrajectoryRecorder` (or ``None``)."""
        return self._trajectory_recorder

    @property
    def cron_scheduler(self) -> Any:
        """Direct access to the underlying :class:`CronScheduler` (or ``None``)."""
        return self._cron_scheduler

    @property
    def gateway_runner(self) -> Any:
        """Direct access to the underlying :class:`GatewayRunner` (or ``None``)."""
        return self._gateway_runner

    @property
    def acp_server(self) -> Any:
        """Direct access to the underlying :class:`ACPServer` (or ``None``)."""
        return self._acp_server

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Gracefully shut down all initialised subsystems."""
        logger.info("HermesBridge shutting down …")

        # Cron
        if self._cron_scheduler is not None:
            self._safe_call("cron", "stop", self._cron_scheduler.stop, async_=True)

        # Gateway
        if self._gateway_runner is not None:
            self._safe_call("gateway", "stop", self._gateway_runner.stop, async_=True)

        # Memory
        if self._memory_manager is not None:
            self._safe_call("memory", "close", self._memory_manager.close, async_=True)

        # Trajectory
        if self._trajectory_recorder is not None:
            try:
                import asyncio as _aio
                _aio.get_event_loop().run_until_complete(self._trajectory_recorder.save())
            except Exception:
                pass

        # ACP
        if self._acp_server is not None:
            # ACP doesn't have a shutdown, but log it
            logger.info("ACP server reference released (was at %s:%s)", self._acp_server.host, self._acp_server.port)

        self._initialized = False
        logger.info("HermesBridge shut down complete.")

    # ------------------------------------------------------------------
    # Private — subsystem initialisers
    # ------------------------------------------------------------------

    def _init_tools(self) -> None:
        """Discover and register all Hermes tools."""
        if self._config is None or not self._config.tools.enabled:
            return
        try:
            from hermes.tools import discover_tools
            self._tool_registry = discover_tools()
            self._mark_ok("tools")
            logger.info("Hermes tools discovered: %d tool(s)", len(self._tool_registry))
        except Exception as exc:
            self._mark_err("tools", exc)

    def _init_context_compression(self) -> None:
        """Initialise the context compressor."""
        if self._config is None or not self._config.context_compression.enabled:
            return
        try:
            from hermes.core.context_compressor import ContextCompressor
            extra = self._config.context_compression.extra
            self._compressor = ContextCompressor(
                max_context_tokens=extra.get("max_context_tokens", 200_000),
                reserve_tokens=extra.get("reserve_tokens", 8_000),
                preserve_recent_turns=extra.get("preserve_recent_turns", 6),
            )
            self._mark_ok("context_compression")
        except Exception as exc:
            self._mark_err("context_compression", exc)

    def _init_prompt_builder(self) -> None:
        """Initialise the prompt builder."""
        if self._config is None or not self._config.prompt_builder.enabled:
            return
        try:
            from hermes.core.prompt_builder import PromptBuilder
            self._prompt_builder = PromptBuilder()
            self._mark_ok("prompt_builder")
        except Exception as exc:
            self._mark_err("prompt_builder", exc)

    def _init_smart_routing(self) -> None:
        """Initialise the smart router."""
        if self._config is None or not self._config.smart_routing.enabled:
            return
        try:
            from hermes.core.smart_routing import SmartRouter
            extra = self._config.smart_routing.extra
            self._smart_router = SmartRouter(
                default_model=extra.get("default_model", "claude-sonnet-4-20250514"),
            )
            self._mark_ok("smart_routing")
        except Exception as exc:
            self._mark_err("smart_routing", exc)

    def _init_insights(self) -> None:
        """Initialise the insights manager."""
        if self._config is None or not self._config.insights.enabled:
            return
        try:
            from hermes.core.insights import InsightsManager
            extra = self._config.insights.extra
            self._insights = InsightsManager(
                session_id=extra.get("session_id", "default"),
                retention_days=extra.get("retention_days", 90),
            )
            self._mark_ok("insights")
        except Exception as exc:
            self._mark_err("insights", exc)

    def _init_trajectory(self) -> None:
        """Initialise the trajectory recorder."""
        if self._config is None or not self._config.trajectory.enabled:
            return
        try:
            from hermes.core.trajectory import TrajectoryRecorder
            extra = self._config.trajectory.extra
            self._trajectory_recorder = TrajectoryRecorder(
                session_id=extra.get("session_id", "bridge_default"),
                model_name=extra.get("model_name", ""),
            )
            self._mark_ok("trajectory")
        except Exception as exc:
            self._mark_err("trajectory", exc)

    def _init_memory(self) -> None:
        """Initialise the memory manager with built-in and optional plugin provider."""
        if self._config is None or not self._config.memory.enabled:
            return
        try:
            from hermes.core.memory_manager import MemoryConfig, MemoryManager

            extra = self._config.memory.extra
            mem_cfg = MemoryConfig(
                enabled=True,
                max_context_tokens=extra.get("max_context_tokens", 4000),
                auto_save=extra.get("auto_save", True),
                prefetch_enabled=extra.get("prefetch_enabled", True),
                builtin_memory_dir=extra.get("builtin_memory_dir"),
            )

            # Optional external plugin
            external_provider = self._load_memory_plugin(extra)
            if external_provider is not None:
                mem_cfg.external_provider = external_provider

            self._memory_manager = MemoryManager(config=mem_cfg)
            self._mark_ok("memory")
            logger.info("Memory manager created (external=%s)", external_provider is not None)
        except Exception as exc:
            self._mark_err("memory", exc)

    def _load_memory_plugin(self, extra: Dict[str, Any]) -> Any:
        """
        Attempt to load an external memory plugin from config.

        Returns a ``MemoryProvider`` or ``None``.
        """
        plugin_name = extra.get("external_plugin")
        if not plugin_name:
            return None

        try:
            from hermes.plugins.memory.registry import MemoryPluginRegistry
            from hermes.plugins.memory.base import MemoryConfig as PluginMemoryConfig

            registry = MemoryPluginRegistry()
            plugin_cfg = PluginMemoryConfig(**{
                k: v for k, v in extra.get("plugin_config", {}).items()
                if k in PluginMemoryConfig.__dataclass_fields__
            })
            registry.set_config(plugin_name, plugin_cfg)

            # Note: this is synchronous in setup; actual initialisation
            # happens lazily when the MemoryManager calls initialize().
            plugin = registry.get_plugin(plugin_name)
            if plugin is None:
                # Try to import directly by common module paths
                plugin = self._try_direct_plugin_import(plugin_name, extra)
            return plugin
        except Exception as exc:
            logger.warning("Failed to load memory plugin %r: %s", plugin_name, exc)
            return None

    @staticmethod
    def _try_direct_plugin_import(plugin_name: str, extra: Dict[str, Any]) -> Any:
        """Try common import paths for built-in memory plugins."""
        import importlib
        module_map = {
            "mem0": "hermes.plugins.memory.mem0_plugin",
            "holographic": "hermes.plugins.memory.holographic",
            "honcho": "hermes.plugins.memory.honcho",
            "retaindb": "hermes.plugins.memory.retaindb",
            "hindsight": "hermes.plugins.memory.hindsight",
            "openviking": "hermes.plugins.memory.openviking",
            "byterover": "hermes.plugins.memory.byterover",
        }
        module_path = module_map.get(plugin_name, f"hermes.plugins.memory.{plugin_name}")
        try:
            mod = importlib.import_module(module_path)
            # Look for a plugin class in the module
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if (
                    isinstance(obj, type)
                    and obj.__name__ not in ("type", "ABCMeta")
                    and hasattr(obj, "initialize")
                ):
                    from hermes.plugins.memory.base import BaseMemoryPlugin
                    if issubclass(obj, BaseMemoryPlugin) and obj is not BaseMemoryPlugin:
                        from hermes.plugins.memory.base import MemoryConfig as PMC
                        cfg = PMC(**{
                            k: v for k, v in extra.get("plugin_config", {}).items()
                            if k in PMC.__dataclass_fields__
                        })
                        return obj(config=cfg)
        except Exception:
            pass
        return None

    def _init_skills(self) -> None:
        """Initialise the skill manager and load built-in skills."""
        if self._config is None or not self._config.skills.enabled:
            return
        try:
            from hermes.skills.manager import SkillManager

            extra = self._config.skills.extra
            skills_dirs = extra.get("skills_dirs")
            self._skill_manager = SkillManager(skills_dirs=skills_dirs)
            self._mark_ok("skills")
            logger.info("Skill manager created")
        except Exception as exc:
            self._mark_err("skills", exc)

    def _init_cron(self) -> None:
        """Initialise the cron scheduler (but do NOT start it)."""
        if self._config is None or not self._config.cron.enabled:
            return
        try:
            from hermes.cron.scheduler import CronScheduler

            extra = self._config.cron.extra
            self._cron_scheduler = CronScheduler(
                data_dir=extra.get("data_dir"),
                tick_interval=extra.get("tick_interval", 60.0),
                default_timezone=extra.get("default_timezone", "UTC"),
            )
            self._mark_ok("cron")
            logger.info("Cron scheduler created (not started)")
        except Exception as exc:
            self._mark_err("cron", exc)

    def _init_gateway(self) -> None:
        """Initialise the gateway runner (but do NOT start it)."""
        if self._config is None or not self._config.gateway.enabled:
            return
        try:
            from hermes.gateway.config import GatewayConfig
            from hermes.gateway.runner import GatewayRunner

            extra = self._config.gateway.extra
            config_path = extra.get("config_path")
            if config_path:
                gw_config = GatewayConfig.load(config_path)
            else:
                # Build a minimal default config
                gw_config = GatewayConfig(
                    session_timeout=extra.get("session_timeout", 3600),
                    session_max_tokens=extra.get("session_max_tokens", 200_000),
                )
            self._gateway_runner = GatewayRunner(config=gw_config)
            self._mark_ok("gateway")
            logger.info("Gateway runner created (not started)")
        except Exception as exc:
            self._mark_err("gateway", exc)

    def _init_acp(self) -> None:
        """Initialise the ACP server (but do NOT start it)."""
        if self._config is None or not self._config.acp.enabled:
            return
        try:
            from hermes.acp.server import ACPServer

            extra = self._config.acp.extra
            self._acp_server = ACPServer(
                host=extra.get("host", "localhost"),
                port=extra.get("port", 8765),
                data_dir=extra.get("data_dir"),
                secret_key=extra.get("secret_key"),
            )
            self._mark_ok("acp")
            logger.info("ACP server created at %s:%s (not started)",
                         self._acp_server.host, self._acp_server.port)
        except Exception as exc:
            self._mark_err("acp", exc)

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------

    def _mark_ok(self, name: str) -> None:
        """Mark a subsystem as initialised and available."""
        st = self._status.get(name)
        if st is not None:
            st.initialized = True
            st.available = True
            st.error = ""

    def _mark_err(self, name: str, exc: Exception) -> None:
        """Mark a subsystem as failed."""
        st = self._status.get(name)
        if st is not None:
            st.initialized = True
            st.available = False
            st.error = str(exc)
        logger.warning("Hermes subsystem %r failed to initialise: %s", name, exc)

    @staticmethod
    def _safe_call(
        subsystem: str,
        label: str,
        fn: Callable,
        async_: bool = False,
    ) -> None:
        """Call *fn* with full exception handling; log but never crash."""
        try:
            if async_:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(fn())
                else:
                    loop.run_until_complete(fn())
            else:
                fn()
        except Exception as exc:
            logger.warning("Error during %s %s: %s", subsystem, label, exc)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# Model name → approximate context window (tokens)
_MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o1": 200_000,
    "o3-mini": 200_000,
    "deepseek-chat": 64_000,
    "gemini-2.0-pro": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
}


def _context_budget_for_model(model: str) -> int:
    """Return a sensible default context budget for a given model name."""
    if model:
        for key, budget in _MODEL_CONTEXT_WINDOWS.items():
            if key in model:
                return budget
    return 200_000
