"""
Plugin Registry — Centralised, thread-safe store for plugin metadata.

The registry is the **single source of truth** for which plugins are known
to the Atlas runtime.  It holds :class:`PluginInfo` records (which wrap a
frozen :class:`PluginManifest` plus live runtime state) and provides APIs
for discovery, registration, querying, and validation.

Design
------
* **Singleton** — ``PluginRegistry.instance()`` returns the process-wide
  registry.  This avoids global-module state while still guaranteeing a
  single authoritative collection.
* **Thread-safe** — all mutating operations acquire a :class:`threading.Lock`.
  Reads that do not hold the lock are safe because the internal dict is
  only replaced atomically on write.
* **Event callbacks** — callers can subscribe to ``on_register``, ``on_unregister``,
  and ``on_state_change`` hooks to react to registry mutations.

Usage::

    registry = PluginRegistry.instance()

    # Discover plugins from a directory
    manifests = registry.discover(Path("~/.atlas/plugins"))

    # Register a manifest
    info = registry.register(manifest)

    # Query
    tools_plugins = registry.list_by_capability(PluginCapability.TOOLS)
    my_plugin = registry.get("my-plugin")
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from atlas.plugin_sdk.core import (
    PluginCapability,
    PluginInfo,
    PluginManifest,
    PluginState,
    PluginValidationError,
    compute_dependency_graph,
    topological_sort,
)
from atlas.plugin_sdk.manifest import ManifestParser, ValidationResult

logger = logging.getLogger(__name__)


# Type aliases for callback signatures
RegisterCallback = Callable[[PluginInfo], None]
UnregisterCallback = Callable[[str], None]
StateChangeCallback = Callable[[str, PluginState, PluginState], None]


class PluginRegistry:
    """Thread-safe, singleton registry for Atlas plugins.

    Attributes:
        _plugins: Mapping of plugin name → :class:`PluginInfo`.
        _lock: Reentrant lock guarding all mutations.
        _manifest_parser: Parser used during directory discovery.
    """

    _instance: Optional[PluginRegistry] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        manifest_parser: Optional[ManifestParser] = None,
    ) -> None:
        self._plugins: Dict[str, PluginInfo] = {}
        self._lock = threading.RLock()
        self._manifest_parser = manifest_parser or ManifestParser()

        # Event callbacks
        self._on_register_callbacks: List[RegisterCallback] = []
        self._on_unregister_callbacks: List[UnregisterCallback] = []
        self._on_state_change_callbacks: List[StateChangeCallback] = []

        # Statistics
        self._total_registered: int = 0
        self._total_unregistered: int = 0

    # -- Singleton ---------------------------------------------------------

    @classmethod
    def instance(cls) -> PluginRegistry:
        """Return the global singleton registry.

        Creates the instance on first call (lazy init).
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.debug("PluginRegistry singleton initialised")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing).

        Drops all registered plugins and callbacks.
        """
        with cls._instance_lock:
            cls._instance = None

    # -- Registration ------------------------------------------------------

    def register(
        self,
        manifest: Union[PluginManifest, Dict[str, Any]],
        path: Optional[Union[str, Path]] = None,
    ) -> PluginInfo:
        """Register a plugin with the registry.

        Args:
            manifest: Either a :class:`PluginManifest` instance or a raw dict
                      that will be validated and converted.
            path: Optional filesystem path to the plugin root.

        Returns:
            The created :class:`PluginInfo`.

        Raises:
            PluginValidationError: If the manifest is invalid.
            ValueError: If a plugin with the same name is already registered
                        in a non-ERROR state.
        """
        if isinstance(manifest, dict):
            result = self._manifest_parser.validate(manifest)
            if not result.valid:
                raise PluginValidationError(
                    f"Manifest validation failed: {'; '.join(result.errors)}",
                    plugin_name=manifest.get("name", ""),
                    errors=result.errors,
                )
            manifest = self._manifest_parser.from_dict(manifest, result)

        name = manifest.name

        with self._lock:
            existing = self._plugins.get(name)
            if existing and existing.state not in (PluginState.ERROR, PluginState.DISABLED):
                raise ValueError(
                    f"Plugin {name!r} is already registered (state={existing.state.value}). "
                    f"Unregister or disable it first."
                )

            info = PluginInfo(
                manifest=manifest,
                state=PluginState.DISCOVERED,
                path=Path(path).resolve() if path else None,
            )

            self._plugins[name] = info
            self._total_registered += 1

            logger.info(
                "Registered plugin %s@%s (capabilities=%s, permissions=%s)",
                name,
                manifest.version,
                [str(c) for c in manifest.sorted_capabilities],
                [str(p) for p in manifest.sorted_permissions],
            )

            # Fire callbacks outside the lock
            callbacks = list(self._on_register_callbacks)

        for cb in callbacks:
            try:
                cb(info)
            except Exception:
                logger.exception("on_register callback failed for plugin %s", name)

        return info

    def unregister(self, name: str) -> bool:
        """Remove a plugin from the registry.

        Args:
            name: Plugin name to unregister.

        Returns:
            ``True`` if the plugin was found and removed.
        """
        with self._lock:
            info = self._plugins.pop(name, None)
            if info is None:
                logger.warning("Cannot unregister unknown plugin %r", name)
                return False
            self._total_unregistered += 1
            callbacks = list(self._on_unregister_callbacks)

        logger.info("Unregistered plugin %s@%s", name, info.version)
        for cb in callbacks:
            try:
                cb(name)
            except Exception:
                logger.exception("on_unregister callback failed for plugin %s", name)

        return True

    # -- Querying ----------------------------------------------------------

    def get(self, name: str) -> Optional[PluginInfo]:
        """Look up a plugin by name.

        Returns ``None`` if not found.
        """
        return self._plugins.get(name)

    def get_or_raise(self, name: str) -> PluginInfo:
        """Look up a plugin by name, raising :class:`KeyError` if absent."""
        info = self._plugins.get(name)
        if info is None:
            raise KeyError(f"Plugin {name!r} not found in registry")
        return info

    def list_all(self) -> List[PluginInfo]:
        """Return a snapshot of all registered plugins."""
        with self._lock:
            return list(self._plugins.values())

    def list_by_state(self, state: PluginState) -> List[PluginInfo]:
        """Return all plugins in a given lifecycle state."""
        return [info for info in self._plugins.values() if info.state == state]

    def list_by_capability(self, capability: PluginCapability) -> List[PluginInfo]:
        """Return all plugins that advertise a given capability."""
        return [
            info
            for info in self._plugins.values()
            if capability in info.manifest.capabilities
        ]

    def list_by_permission(self, permission) -> List[PluginInfo]:
        """Return all plugins that have a given permission.

        Args:
            permission: A :class:`PluginPermission` or permission string.
        """
        from atlas.plugin_sdk.core import PluginPermission as PP
        perm = permission if isinstance(permission, PP) else PP(permission)
        return [
            info
            for info in self._plugins.values()
            if perm in info.manifest.permissions
        ]

    def list_active(self) -> List[PluginInfo]:
        """Return all currently active plugins."""
        return self.list_by_state(PluginState.ACTIVE)

    def list_errors(self) -> List[PluginInfo]:
        """Return all plugins in the ERROR state."""
        return self.list_by_state(PluginState.ERROR)

    def count(self) -> int:
        """Return the total number of registered plugins."""
        return len(self._plugins)

    def has(self, name: str) -> bool:
        """Return ``True`` if a plugin with *name* is registered."""
        return name in self._plugins

    def names(self) -> List[str]:
        """Return a sorted list of all registered plugin names."""
        return sorted(self._plugins.keys())

    # -- State management --------------------------------------------------

    def set_state(
        self,
        name: str,
        new_state: PluginState,
        error: Optional[str] = None,
    ) -> bool:
        """Transition a plugin to a new state.

        Args:
            name: Plugin name.
            new_state: Target state.
            error: Optional error message (used when transitioning to ERROR).

        Returns:
            ``True`` if the transition succeeded.
        """
        with self._lock:
            info = self._plugins.get(name)
            if info is None:
                logger.warning("Cannot set state for unknown plugin %r", name)
                return False
            old_state = info.state
            try:
                info.transition_to(new_state, error=error)
            except ValueError as exc:
                logger.warning(
                    "Illegal state transition for %s: %s → %s: %s",
                    name,
                    old_state.value,
                    new_state.value,
                    exc,
                )
                return False
            callbacks = list(self._on_state_change_callbacks)

        logger.info("Plugin %s: %s → %s", name, old_state.value, new_state.value)
        for cb in callbacks:
            try:
                cb(name, old_state, new_state)
            except Exception:
                logger.exception(
                    "on_state_change callback failed for plugin %s", name
                )
        return True

    # -- Discovery ---------------------------------------------------------

    def discover(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
    ) -> List[PluginManifest]:
        """Scan a directory for plugin manifests and register them.

        Args:
            directory: Path to scan.
            recursive: If ``True``, descend into subdirectories.

        Returns:
            List of successfully parsed :class:`PluginManifest` instances.
        """
        root = Path(directory).resolve()
        if not root.is_dir():
            logger.warning("Discovery path is not a directory: %s", root)
            return []

        manifest_paths = self._manifest_parser.scan_directory(root, recursive=recursive)
        registered: List[PluginManifest] = []
        errors: List[Tuple[str, str]] = []

        for mp in manifest_paths:
            try:
                manifest = self._manifest_parser.parse(mp)
                self.register(manifest, path=mp.parent)
                registered.append(manifest)
            except Exception as exc:
                plugin_name = mp.parent.name
                errors.append((plugin_name, str(exc)))
                logger.error(
                    "Failed to discover plugin from %s: %s", mp, exc
                )

        logger.info(
            "Discovery complete: %d registered, %d errors from %d candidates",
            len(registered),
            len(errors),
            len(manifest_paths),
        )

        for name, err in errors:
            logger.warning("Plugin discovery error [%s]: %s", name, err)

        return registered

    # -- Validation --------------------------------------------------------

    def validate(
        self,
        manifest: Union[PluginManifest, Dict[str, Any]],
    ) -> ValidationResult:
        """Validate a manifest against the schema.

        This is a convenience wrapper around :class:`ManifestParser.validate`.

        Returns:
            A :class:`ValidationResult` with errors/warnings.
        """
        if isinstance(manifest, PluginManifest):
            data = manifest.to_dict()
        else:
            data = manifest
        return self._manifest_parser.validate(data)

    def validate_dependencies(
        self,
        manifest: PluginManifest,
    ) -> List[str]:
        """Check if all of *manifest*'s peer dependencies are satisfied.

        Returns a list of missing dependency names (empty if all satisfied).
        """
        missing: List[str] = []
        for dep_name, min_version in manifest.dependencies.items():
            dep_info = self._plugins.get(dep_name)
            if dep_info is None:
                missing.append(dep_name)
                continue
            if min_version:
                try:
                    from atlas.plugin_sdk.core import parse_semver
                    dep_ver = parse_semver(dep_info.version)
                    min_ver = parse_semver(min_version)
                    if dep_ver < min_ver:
                        missing.append(
                            f"{dep_name}>={min_version} "
                            f"(found {dep_info.version})"
                        )
                except ValueError:
                    missing.append(dep_name)
        return missing

    def validate_all_dependencies(self) -> Dict[str, List[str]]:
        """Validate dependencies for all registered plugins.

        Returns a mapping of plugin name → list of missing/unsatisfied deps.
        """
        result: Dict[str, List[str]] = {}
        for name, info in self._plugins.items():
            missing = self.validate_dependencies(info.manifest)
            if missing:
                result[name] = missing
        return result

    def get_load_order(self) -> List[str]:
        """Return plugin names sorted for loading (dependencies first).

        Only includes plugins in DISCOVERED state.
        """
        discovered = [
            info.manifest
            for info in self._plugins.values()
            if info.state == PluginState.DISCOVERED
        ]
        return topological_sort(discovered)

    # -- Event callbacks ---------------------------------------------------

    def on_register(self, callback: RegisterCallback) -> RegisterCallback:
        """Register a callback fired after a plugin is registered.

        The callback receives the :class:`PluginInfo`.

        Usage::

            def handler(info: PluginInfo) -> None:
                print(f"New plugin: {info.name}")

            registry.on_register(handler)
        """
        self._on_register_callbacks.append(callback)
        return callback

    def on_unregister(self, callback: UnregisterCallback) -> UnregisterCallback:
        """Register a callback fired after a plugin is unregistered."""
        self._on_unregister_callbacks.append(callback)
        return callback

    def on_state_change(
        self, callback: StateChangeCallback
    ) -> StateChangeCallback:
        """Register a callback fired when a plugin changes state.

        The callback receives ``(name, old_state, new_state)``.
        """
        self._on_state_change_callbacks.append(callback)
        return callback

    def remove_callback(self, callback: Any) -> bool:
        """Remove a previously registered callback (any type)."""
        removed = False
        for lst in (
            self._on_register_callbacks,
            self._on_unregister_callbacks,
            self._on_state_change_callbacks,
        ):
            try:
                lst.remove(callback)
                removed = True
            except ValueError:
                pass
        return removed

    # -- Bulk operations ---------------------------------------------------

    def clear(self) -> int:
        """Remove all plugins from the registry.

        Returns the number of plugins that were removed.
        """
        with self._lock:
            count = len(self._plugins)
            self._plugins.clear()
            self._total_unregistered += count
        logger.info("Cleared registry (%d plugins removed)", count)
        return count

    def disable_all(self) -> int:
        """Transition all ACTIVE plugins to DISABLED.

        Returns the number of plugins disabled.
        """
        count = 0
        for info in self.list_all():
            if info.state == PluginState.ACTIVE:
                self.set_state(info.name, PluginState.DISABLED)
                count += 1
        logger.info("Disabled %d plugin(s)", count)
        return count

    def clear_errors(self) -> int:
        """Reset all ERROR-state plugins back to DISCOVERED.

        Returns the number of plugins reset.
        """
        count = 0
        for info in self.list_errors():
            self.set_state(info.name, PluginState.DISCOVERED)
            count += 1
        logger.info("Reset %d plugin(s) from ERROR to DISCOVERED", count)
        return count

    # -- Statistics --------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Return a snapshot of registry statistics."""
        states: Dict[str, int] = {}
        capabilities: Dict[str, int] = {}
        for info in self._plugins.values():
            state_key = info.state.value
            states[state_key] = states.get(state_key, 0) + 1
            for cap in info.manifest.capabilities:
                cap_key = str(cap)
                capabilities[cap_key] = capabilities.get(cap_key, 0) + 1

        return {
            "total": len(self._plugins),
            "total_registered_ever": self._total_registered,
            "total_unregistered_ever": self._total_unregistered,
            "by_state": states,
            "by_capability": capabilities,
        }

    # -- Representation ----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<PluginRegistry plugins={len(self._plugins)} "
            f"active={len(self.list_active())} "
            f"errors={len(self.list_errors())}>"
        )

    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, name: object) -> bool:
        if isinstance(name, str):
            return name in self._plugins
        return False
