"""
Atlas Plugin SDK — A comprehensive plugin system for the Claude Clone platform.

This package provides all the building blocks needed to create, discover,
load, sandbox, and manage plugins that extend Atlas with new tools,
messaging channels, AI providers, CLI commands, and lifecycle hooks.

Quick Start
-----------
Register and load a plugin::

    from atlas.plugin_sdk import PluginRegistry, PluginLoader, ManifestParser
    from pathlib import Path

    # Parse a manifest
    parser = ManifestParser()
    manifest = parser.parse(Path("my_plugin/plugin.json"))

    # Register
    registry = PluginRegistry.instance()
    registry.register(manifest, path=Path("my_plugin"))

    # Load
    loader = PluginLoader(registry=registry)
    result = loader.load("my-plugin")

    # Access tools
    tools = loader.get_plugin_tools("my-plugin")

Module Structure
----------------
* **core**        — Data classes, enums, exceptions, and utility functions.
* **contracts**   — Abstract base classes (``BasePlugin``, ``ToolPlugin``, etc.)
                    and decorators (``@tool``, ``@hook``, ``@command``).
* **manifest**    — Manifest parsing, validation, serialization, and generation.
* **registry**    — Thread-safe singleton registry for plugin metadata.
* **loader**      — Plugin import, instantiation, lifecycle management, hot-reload.
* **sandbox**     — Sandboxed execution with permissions, path guards, and
                    resource limits.
"""

from atlas.plugin_sdk.core import (
    DEFAULT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    PluginCapability,
    PluginError,
    PluginInfo,
    PluginLoadError,
    PluginManifest,
    PluginNotFoundError,
    PluginPermission,
    PluginPermissionError,
    PluginSandboxError,
    PluginState,
    PluginValidationError,
    compute_dependency_graph,
    normalize_plugin_name,
    parse_semver,
    topological_sort,
    validate_plugin_name,
    version_sort_key,
)

from atlas.plugin_sdk.contracts import (
    BasePlugin,
    CAPABILITY_CONTRACT_MAP,
    ChannelPlugin,
    CommandDefinition,
    CommandPlugin,
    HookDefinition,
    HookPlugin,
    ProviderPlugin,
    ToolDefinition,
    ToolPlugin,
    command,
    get_contract_class,
    hook,
    resolve_plugin_class,
    tool,
)

from atlas.plugin_sdk.manifest import (
    MANIFEST_FILENAMES,
    ManifestParser,
    ValidationResult,
)

from atlas.plugin_sdk.registry import (
    PluginRegistry,
    RegisterCallback,
    StateChangeCallback,
    UnregisterCallback,
)

from atlas.plugin_sdk.loader import (
    LoadedPlugin,
    LoadResult,
    PluginLoader,
)

from atlas.plugin_sdk.sandbox import (
    ExecutionResult,
    ImportGuard,
    PathGuard,
    PluginSandbox,
    ResourceLimits,
    ResourceMonitor,
    SandboxConfig,
)

__all__ = [
    # core
    "DEFAULT_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "PluginCapability",
    "PluginError",
    "PluginInfo",
    "PluginLoadError",
    "PluginManifest",
    "PluginNotFoundError",
    "PluginPermission",
    "PluginPermissionError",
    "PluginSandboxError",
    "PluginState",
    "PluginValidationError",
    "compute_dependency_graph",
    "normalize_plugin_name",
    "parse_semver",
    "topological_sort",
    "validate_plugin_name",
    "version_sort_key",
    # contracts
    "BasePlugin",
    "CAPABILITY_CONTRACT_MAP",
    "ChannelPlugin",
    "CommandDefinition",
    "CommandPlugin",
    "HookDefinition",
    "HookPlugin",
    "ProviderPlugin",
    "ToolDefinition",
    "ToolPlugin",
    "command",
    "get_contract_class",
    "hook",
    "resolve_plugin_class",
    "tool",
    # manifest
    "MANIFEST_FILENAMES",
    "ManifestParser",
    "ValidationResult",
    # registry
    "PluginRegistry",
    "RegisterCallback",
    "StateChangeCallback",
    "UnregisterCallback",
    # loader
    "LoadedPlugin",
    "LoadResult",
    "PluginLoader",
    # sandbox
    "ExecutionResult",
    "ImportGuard",
    "PathGuard",
    "PluginSandbox",
    "ResourceLimits",
    "ResourceMonitor",
    "SandboxConfig",
]

__version__ = "1.0.0"
