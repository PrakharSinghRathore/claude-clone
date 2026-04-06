"""
Plugin SDK Core — Fundamental types and data structures for the Atlas plugin system.

This module defines the core data classes, enumerations, and shared utilities
that every other component in the plugin SDK depends on.  All plugin manifests,
state tracking, permission models, and capability flags live here.

Design Principles
-----------------
* **Immutability-first** — dataclasses use ``frozen=True`` where possible so that
  plugin metadata cannot be accidentally mutated after registration.
* **Version-aware** — every manifest carries a ``schema_version`` field so that
  the parser and loader can evolve the format gracefully.
* **Strict typing** — every public symbol has complete type annotations to
  support static analysis and IDE auto-completion.
"""

from __future__ import annotations

import enum
import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Semantic-versioning helpers
# ---------------------------------------------------------------------------

SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


def parse_semver(version: str) -> Tuple[int, int, int]:
    """Parse a semantic version string into a ``(major, minor, patch)`` tuple.

    Raises:
        ValueError: If *version* does not match the semver pattern.
    """
    match = SEMVER_RE.match(version.strip())
    if not match:
        raise ValueError(f"Invalid semantic version: {version!r}")
    return int(match.group("major")), int(match.group("minor")), int(match.group("patch"))


def version_sort_key(version: str) -> Tuple[int, int, int]:
    """Return a sort key for a version string."""
    try:
        return parse_semver(version)
    except ValueError:
        return (0, 0, 0)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PluginState(enum.Enum):
    """Lifecycle state of a plugin inside the registry.

    Attributes:
        DISCOVERED: The plugin manifest has been found but not yet loaded.
        LOADING: The plugin's entry-point module is being imported.
        ACTIVE: The plugin is loaded and ready to receive calls.
        DISABLED: The plugin has been explicitly disabled by the user.
        ERROR: The plugin failed to load or threw an unhandled exception.
    """

    DISCOVERED = "discovered"
    LOADING = "loading"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value

    @property
    def is_running(self) -> bool:
        """Return ``True`` if the plugin is in a functional state."""
        return self in (PluginState.ACTIVE, PluginState.LOADING)

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` if the plugin is in a terminal (non-transitioning) state."""
        return self in (PluginState.ACTIVE, PluginState.DISABLED, PluginState.ERROR)


class PluginPermission(enum.Enum):
    """Permission flags that control what a plugin is allowed to do.

    Permissions are enforced by :class:`atlas.plugin_sdk.sandbox.PluginSandbox`.

    Attributes:
        FILE_READ: Read files on the host filesystem.
        FILE_WRITE: Write / create / delete files on the host filesystem.
        NETWORK: Make outbound HTTP / TCP / UDP requests.
        SHELL: Execute shell commands or subprocesses.
        CLAUDE_ACCESS: Interact with the Claude conversation API directly.
        SYSTEM: Modify system-level configuration (env vars, services, etc.).
    """

    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    NETWORK = "network"
    SHELL = "shell"
    CLAUDE_ACCESS = "claude_access"
    SYSTEM = "system"

    def __str__(self) -> str:
        return self.value

    # Convenience groupings ------------------------------------------------

    @classmethod
    def minimal_set(cls) -> FrozenSet[PluginPermission]:
        """Minimum safe permissions with no filesystem or network access."""
        return frozenset()

    @classmethod
    def standard_set(cls) -> FrozenSet[PluginPermission]:
        """Reasonable defaults for most tool plugins (read + network)."""
        return frozenset({cls.FILE_READ, cls.NETWORK})

    @classmethod
    def full_set(cls) -> FrozenSet[PluginPermission]:
        """All permissions — use only for fully-trusted plugins."""
        return frozenset(cls)

    def __lt__(self, other: object) -> bool:
        """Order by enum value string for deterministic serialization."""
        if isinstance(other, PluginPermission):
            return self.value < other.value
        return NotImplemented


class PluginCapability(enum.Enum):
    """Capability flags advertised by a plugin manifest.

    These flags determine *what kind* of extension points the plugin
    intends to hook into.

    Attributes:
        TOOLS: Provides Claude-callable tools.
        CHANNELS: Provides messaging-channel integrations (Slack, Discord, etc.).
        PROVIDERS: Provides AI model provider backends.
        COMMANDS: Provides CLI sub-commands for Atlas.
        HOOKS: Provides lifecycle / event hooks.
    """

    TOOLS = "tools"
    CHANNELS = "channels"
    PROVIDERS = "providers"
    COMMANDS = "commands"
    HOOKS = "hooks"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Plugin Manifest
# ---------------------------------------------------------------------------

# Supported schema versions for plugin manifests
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"v1", "v2"})
DEFAULT_SCHEMA_VERSION: str = "v2"


@dataclass(frozen=True)
class PluginManifest:
    """Immutable, validated manifest for an Atlas plugin.

    A manifest is the *declaration of intent* for a plugin.  It describes
    the plugin's metadata, its entry-point, required permissions, advertised
    capabilities, and any peer dependencies.

    Attributes:
        name: Unique slug identifier (``kebab-case``, no spaces).
        version: Semantic version string (e.g. ``"1.2.3"``).
        description: Human-readable one-liner.
        author: Author name or handle.
        license: SPDX identifier (e.g. ``"MIT"``, ``"Apache-2.0"``).
        entry_point: Dotted Python import path (e.g. ``"my_plugin.plugin:setup"``).
        permissions: Set of :class:`PluginPermission` values the plugin requests.
        capabilities: Set of :class:`PluginCapability` values the plugin advertises.
        dependencies: Mapping of peer plugin names to minimum version constraints.
        schema_version: Manifest schema version (``"v1"`` or ``"v2"``).
        homepage: Optional URL to the plugin's project page.
        tags: Optional list of searchable tags.
        icon: Optional path (relative to plugin root) to an icon file.
        min_atlas_version: Minimum Atlas version required (semver string).
        max_atlas_version: Maximum Atlas version supported (semver string).
    """

    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = "MIT"
    entry_point: str = ""
    permissions: FrozenSet[PluginPermission] = frozenset()
    capabilities: FrozenSet[PluginCapability] = frozenset()
    dependencies: Dict[str, str] = field(default_factory=dict)
    schema_version: str = DEFAULT_SCHEMA_VERSION
    homepage: str = ""
    tags: Tuple[str, ...] = ()
    icon: str = ""
    min_atlas_version: str = ""
    max_atlas_version: str = ""

    def __post_init__(self) -> None:
        """Validate invariant constraints after construction."""
        if not self.name or not self.name.strip():
            raise ValueError("Plugin name must be a non-empty string")
        if " " in self.name:
            raise ValueError(f"Plugin name {self.name!r} must not contain spaces")
        # Coerce mutable defaults to immutable containers
        if isinstance(self.dependencies, dict):
            object.__setattr__(self, "dependencies", dict(self.dependencies))
        if isinstance(self.permissions, (list, set, frozenset)):
            perms = set(self.permissions)
            coerced: Set[PluginPermission] = set()
            for p in perms:
                if isinstance(p, PluginPermission):
                    coerced.add(p)
                elif isinstance(p, str):
                    coerced.add(PluginPermission(p))
                else:
                    raise TypeError(f"Unknown permission type: {type(p)}")
            object.__setattr__(self, "permissions", frozenset(coerced))
        if isinstance(self.capabilities, (list, set, frozenset)):
            caps = set(self.capabilities)
            coerced_caps: Set[PluginCapability] = set()
            for c in caps:
                if isinstance(c, PluginCapability):
                    coerced_caps.add(c)
                elif isinstance(c, str):
                    coerced_caps.add(PluginCapability(c))
                else:
                    raise TypeError(f"Unknown capability type: {type(c)}")
            object.__setattr__(self, "capabilities", frozenset(coerced_caps))
        if isinstance(self.tags, list):
            object.__setattr__(self, "tags", tuple(self.tags))

    # Identity helpers -----------------------------------------------------

    @property
    def qualified_name(self) -> str:
        """Return ``"{name}@{version}"`` — a globally unique identifier."""
        return f"{self.name}@{self.version}"

    @property
    def fingerprint(self) -> str:
        """SHA-256 fingerprint of the manifest's canonical JSON representation.

        Useful for cache-busting and integrity checks.
        """
        blob = self.to_canonical_json().encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @property
    def sorted_permissions(self) -> List[PluginPermission]:
        """Permissions in deterministic order for display."""
        return sorted(self.permissions)

    @property
    def sorted_capabilities(self) -> List[PluginCapability]:
        """Capabilities in deterministic order for display."""
        return sorted(self.capabilities)

    # Serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary suitable for JSON/YAML dumping."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "entry_point": self.entry_point,
            "permissions": [str(p) for p in self.sorted_permissions],
            "capabilities": [str(c) for c in self.sorted_capabilities],
            "dependencies": dict(self.dependencies),
            "homepage": self.homepage,
            "tags": list(self.tags),
            "icon": self.icon,
            "min_atlas_version": self.min_atlas_version,
            "max_atlas_version": self.max_atlas_version,
        }

    def to_canonical_json(self) -> str:
        """Return a deterministic JSON string (sorted keys, no whitespace)."""
        import json
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PluginManifest:
        """Deserialize from a plain dictionary, with sensible defaults."""
        raw_perms = data.get("permissions", [])
        raw_caps = data.get("capabilities", [])
        return cls(
            schema_version=data.get("schema_version", DEFAULT_SCHEMA_VERSION),
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", "MIT"),
            entry_point=data.get("entry_point", ""),
            permissions=raw_perms,
            capabilities=raw_caps,
            dependencies=data.get("dependencies", {}),
            homepage=data.get("homepage", ""),
            tags=data.get("tags", []),
            icon=data.get("icon", ""),
            min_atlas_version=data.get("min_atlas_version", ""),
            max_atlas_version=data.get("max_atlas_version", ""),
        )


# ---------------------------------------------------------------------------
# Plugin Info (runtime record)
# ---------------------------------------------------------------------------

@dataclass
class PluginInfo:
    """Runtime information about a registered plugin.

    Unlike :class:`PluginManifest` (which is immutable), this object tracks
    the *live* state of a plugin inside the registry.

    Attributes:
        manifest: The plugin's validated manifest.
        state: Current lifecycle state.
        path: Filesystem path to the plugin root directory.
        error: Last error message, set when ``state == ERROR``.
        loaded_at: Timestamp when the plugin transitioned to ``ACTIVE``.
        instance: Reference to the live plugin object (after loading).
    """

    manifest: PluginManifest
    state: PluginState = PluginState.DISCOVERED
    path: Optional[Path] = None
    error: Optional[str] = None
    loaded_at: Optional[float] = None
    instance: Optional[Any] = None
    _metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        """Shortcut for the plugin's manifest name."""
        return self.manifest.name

    @property
    def version(self) -> str:
        """Shortcut for the plugin's manifest version."""
        return self.manifest.version

    @property
    def qualified_name(self) -> str:
        """Shortcut for ``manifest.qualified_name``."""
        return self.manifest.qualified_name

    @property
    def duration_active(self) -> float:
        """Seconds since the plugin became active (or 0 if not active)."""
        if self.loaded_at is None:
            return 0.0
        return time.monotonic() - self.loaded_at

    # State transitions ----------------------------------------------------

    def transition_to(self, new_state: PluginState, error: Optional[str] = None) -> None:
        """Atomically move to *new_state*, optionally recording an error.

        Raises:
            ValueError: If the transition is illegal (e.g. ACTIVE → DISCOVERED).
        """
        old = self.state
        # Validate transition
        legal: Dict[PluginState, Set[PluginState]] = {
            PluginState.DISCOVERED: {PluginState.LOADING, PluginState.DISABLED},
            PluginState.LOADING: {PluginState.ACTIVE, PluginState.ERROR, PluginState.DISABLED},
            PluginState.ACTIVE: {PluginState.DISABLED, PluginState.ERROR, PluginState.LOADING},
            PluginState.DISABLED: {PluginState.LOADING, PluginState.ACTIVE},
            PluginState.ERROR: {PluginState.LOADING, PluginState.DISABLED, PluginState.DISCOVERED},
        }
        if new_state not in legal.get(old, set()):
            raise ValueError(
                f"Illegal state transition for plugin {self.name!r}: "
                f"{old.value} → {new_state.value}"
            )
        self.state = new_state
        self.error = error
        if new_state == PluginState.ACTIVE:
            self.loaded_at = time.monotonic()
        if new_state in (PluginState.DISABLED, PluginState.ERROR):
            self.loaded_at = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize runtime info to a dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "state": str(self.state),
            "path": str(self.path) if self.path else None,
            "error": self.error,
            "loaded_at": (
                datetime.fromtimestamp(self.loaded_at, tz=timezone.utc).isoformat()
                if self.loaded_at
                else None
            ),
            "duration_active": round(self.duration_active, 3),
            "manifest": self.manifest.to_dict(),
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PluginError(Exception):
    """Base exception for all plugin SDK errors."""

    def __init__(self, message: str, plugin_name: str = "") -> None:
        self.plugin_name = plugin_name
        super().__init__(message)


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin does not exist in the registry."""

    pass


class PluginValidationError(PluginError):
    """Raised when a manifest fails validation."""

    def __init__(self, message: str, plugin_name: str = "", errors: Optional[List[str]] = None) -> None:
        super().__init__(message, plugin_name)
        self.errors = errors or []


class PluginLoadError(PluginError):
    """Raised when a plugin's entry-point cannot be imported."""

    def __init__(self, message: str, plugin_name: str = "", original: Optional[Exception] = None) -> None:
        super().__init__(message, plugin_name)
        self.original = original


class PluginPermissionError(PluginError):
    """Raised when a plugin attempts an action beyond its permissions."""

    def __init__(self, message: str, plugin_name: str = "", permission: str = "") -> None:
        super().__init__(message, plugin_name)
        self.permission = permission


class PluginDependencyError(PluginError):
    """Raised when a plugin's peer dependency cannot be satisfied."""

    def __init__(
        self,
        message: str,
        plugin_name: str = "",
        missing: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message, plugin_name)
        self.missing = missing or []


class PluginSandboxError(PluginError):
    """Raised when sandboxed execution violates a resource limit."""

    def __init__(
        self,
        message: str,
        plugin_name: str = "",
        resource: str = "",
        limit: Optional[float] = None,
    ) -> None:
        super().__init__(message, plugin_name)
        self.resource = resource
        self.limit = limit


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def validate_plugin_name(name: str) -> bool:
    """Check that *name* is a valid plugin slug.

    Valid slugs start with a letter and contain only lowercase letters,
    digits, hyphens, and underscores.

    Returns:
        ``True`` if the name is valid.
    """
    return bool(re.match(r"^[a-z][a-z0-9_-]*$", name))


def normalize_plugin_name(name: str) -> str:
    """Normalize a plugin name to a canonical slug.

    - Lowercase.
    - Replace spaces with hyphens.
    - Strip leading/trailing non-alphanumeric chars.
    - Collapse consecutive hyphens/underscores.
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_-]+", "-", name)
    name = re.sub(r"[-_]{2,}", "-", name)
    name = name.strip("-_")
    return name


def compute_dependency_graph(
    plugins: Sequence[PluginManifest],
) -> Dict[str, Set[str]]:
    """Compute a dependency graph from a collection of manifests.

    Returns a mapping of plugin name → set of plugin names it depends on.

    Raises:
        :class:`PluginDependencyError`: If a circular dependency is detected.
    """
    graph: Dict[str, Set[str]] = {}
    for plugin in plugins:
        graph[plugin.name] = set(plugin.dependencies.keys())

    # Detect cycles via DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in graph}
    path: List[str] = []

    def dfs(node: str) -> None:
        color[node] = GRAY
        path.append(node)
        for dep in graph.get(node, set()):
            if dep not in color:
                continue
            if color[dep] == GRAY:
                cycle_start = path.index(dep)
                cycle = path[cycle_start:] + [dep]
                raise PluginDependencyError(
                    f"Circular dependency detected: {' → '.join(cycle)}",
                    missing=list(cycle),
                )
            if color[dep] == WHITE:
                dfs(dep)
        path.pop()
        color[node] = BLACK

    for node in list(graph):
        if color[node] == WHITE:
            dfs(node)

    return graph


def topological_sort(plugins: Sequence[PluginManifest]) -> List[str]:
    """Return plugin names sorted so that dependencies come first.

    Raises:
        :class:`PluginDependencyError`: On circular dependencies.
    """
    graph = compute_dependency_graph(plugins)
    visited: Set[str] = set()
    order: List[str] = []

    def visit(node: str) -> None:
        if node in visited:
            return
        visited.add(node)
        for dep in graph.get(node, set()):
            visit(dep)
        order.append(node)

    for node in graph:
        visit(node)

    return order
