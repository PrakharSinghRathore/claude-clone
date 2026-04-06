"""
Plugin Loader — Imports plugin entry-points and manages their lifecycle.

The loader is responsible for:

1. **Importing** plugin modules via ``importlib`` with optional sys.path
   manipulation so that each plugin is isolated to its own directory.
2. **Instantiating** the plugin object returned by the entry-point callable.
3. **Driving** the lifecycle: ``on_register`` → ``on_load`` → ACTIVE, and
   ``on_unload`` → DISABLED on teardown.
4. **Extracting** tools, hooks, and commands from loaded plugins so that
   the rest of the Atlas runtime can consume them.
5. **Hot-reloading** plugins at runtime without restarting the host process.

Thread Safety
-------------
All public methods acquire the internal ``_lock`` (a :class:`threading.RLock`)
before mutating state.  Reads without the lock are safe because the internal
mappings are only replaced atomically.

Usage::

    loader = PluginLoader(registry=PluginRegistry.instance())

    # Load all discovered plugins
    loader.load_all()

    # Hot-reload a single plugin
    loader.reload("my-plugin")
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import importlib.util
import logging
import sys
import threading
import time
import traceback
import types
import uuid
from dataclasses import dataclass, field
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
    Type,
    Union,
)

from atlas.plugin_sdk.contracts import (
    BasePlugin,
    CommandDefinition,
    HookDefinition,
    ToolDefinition,
    resolve_plugin_class,
)
from atlas.plugin_sdk.core import (
    PluginCapability,
    PluginInfo,
    PluginLoadError,
    PluginManifest,
    PluginState,
)
from atlas.plugin_sdk.registry import PluginRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LoadResult:
    """Outcome of a plugin load attempt.

    Attributes:
        name: Plugin name.
        success: Whether the load succeeded.
        duration_ms: Wall-clock time taken to load.
        error: Error message if the load failed.
        tools_count: Number of tools extracted.
        hooks_count: Number of hooks extracted.
        commands_count: Number of commands extracted.
    """

    name: str
    success: bool = True
    duration_ms: float = 0.0
    error: Optional[str] = None
    tools_count: int = 0
    hooks_count: int = 0
    commands_count: int = 0


@dataclass
class LoadedPlugin:
    """Bookkeeping for a plugin that has been imported and instantiated.

    Attributes:
        info: The registry :class:`PluginInfo`.
        instance: The live :class:`BasePlugin` instance.
        module: The imported Python module (if any).
        import_time: Monotonic timestamp when the module was first imported.
        reload_count: How many times the plugin has been hot-reloaded.
        sys_path_patches: sys.path entries added during loading.
    """

    info: PluginInfo
    instance: BasePlugin
    module: Optional[types.ModuleType] = None
    import_time: float = field(default_factory=time.monotonic)
    reload_count: int = 0
    sys_path_patches: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PluginLoader
# ---------------------------------------------------------------------------

class PluginLoader:
    """Manages plugin discovery, import, instantiation, and lifecycle.

    Args:
        registry: The :class:`PluginRegistry` to read manifests from and
                  update states in.
        isolation: If ``True``, each plugin is loaded in an isolated
                   ``importlib`` namespace to prevent name collisions.
        auto_dependencies: If ``True``, automatically load dependency plugins
                           before loading dependents.
        timeout: Maximum seconds to wait for ``on_load`` to complete (0=unlimited).
    """

    def __init__(
        self,
        registry: Optional[PluginRegistry] = None,
        isolation: bool = True,
        auto_dependencies: bool = True,
        timeout: float = 0.0,
    ) -> None:
        self._registry = registry or PluginRegistry.instance()
        self._loaded: Dict[str, LoadedPlugin] = {}
        self._lock = threading.RLock()
        self._isolation = isolation
        self._auto_dependencies = auto_dependencies
        self._timeout = timeout

        # Event hooks
        self._pre_load_hooks: List[Callable[[str], None]] = []
        self._post_load_hooks: List[Callable[[LoadedPlugin], None]] = []
        self._pre_unload_hooks: List[Callable[[str], None]] = []
        self._post_unload_hooks: List[Callable[[str], None]] = []

    # -- Properties --------------------------------------------------------

    @property
    def registry(self) -> PluginRegistry:
        """Return the associated :class:`PluginRegistry`."""
        return self._registry

    @property
    def loaded_plugins(self) -> Dict[str, LoadedPlugin]:
        """Return a shallow copy of the loaded-plugins mapping."""
        return dict(self._loaded)

    def is_loaded(self, name: str) -> bool:
        """Return ``True`` if the plugin is loaded and active."""
        lp = self._loaded.get(name)
        return lp is not None and lp.info.state == PluginState.ACTIVE

    # -- Single plugin loading ---------------------------------------------

    def load(self, name: str) -> LoadResult:
        """Load a single plugin by name.

        Steps:
        1. Transition state to LOADING.
        2. Resolve and import the entry-point module.
        3. Call the entry-point callable to get a plugin instance.
        4. Drive lifecycle: ``on_register`` → ``on_load``.
        5. Transition state to ACTIVE (or ERROR on failure).

        Args:
            name: The plugin name as registered in the registry.

        Returns:
            A :class:`LoadResult` describing the outcome.
        """
        start = time.monotonic()

        # Pre-flight
        info = self._registry.get(name)
        if info is None:
            return LoadResult(name=name, success=False, error=f"Plugin {name!r} not found in registry")

        if info.state == PluginState.ACTIVE:
            logger.warning("Plugin %s is already active; skipping load", name)
            return LoadResult(name=name, success=True, duration_ms=0.0)

        # Auto-load dependencies
        if self._auto_dependencies:
            dep_errors = self._load_dependencies(name)
            if dep_errors:
                return LoadResult(
                    name=name,
                    success=False,
                    error=f"Dependency load failures: {'; '.join(dep_errors)}",
                )

        # Fire pre-load hooks
        for hook in self._pre_load_hooks:
            try:
                hook(name)
            except Exception:
                logger.exception("Pre-load hook error for %s", name)

        # Transition to LOADING
        self._registry.set_state(name, PluginState.LOADING)

        try:
            lp = self._do_load(info)
            duration = (time.monotonic() - start) * 1000

            self._registry.set_state(name, PluginState.ACTIVE)
            info.instance = lp.instance

            result = LoadResult(
                name=name,
                success=True,
                duration_ms=round(duration, 2),
                tools_count=len(lp.instance.get_tools()),
                hooks_count=len(lp.instance.get_hooks()),
                commands_count=len(lp.instance.get_commands()),
            )

            # Fire post-load hooks
            for hook in self._post_load_hooks:
                try:
                    hook(lp)
                except Exception:
                    logger.exception("Post-load hook error for %s", name)

            logger.info(
                "Loaded plugin %s@%s in %.1fms "
                "(%d tools, %d hooks, %d commands)",
                name,
                info.version,
                duration,
                result.tools_count,
                result.hooks_count,
                result.commands_count,
            )
            return result

        except Exception as exc:
            tb = traceback.format_exc()
            self._registry.set_state(name, PluginState.ERROR, error=str(exc))
            logger.error("Failed to load plugin %s:\n%s", name, tb)
            return LoadResult(
                name=name,
                success=False,
                duration_ms=round((time.monotonic() - start) * 1000, 2),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _do_load(self, info: PluginInfo) -> LoadedPlugin:
        """Core load logic — import + instantiate + lifecycle calls."""
        manifest = info.manifest
        entry_point = manifest.entry_point
        if not entry_point:
            raise PluginLoadError(
                "Plugin has no entry_point",
                plugin_name=info.name,
            )

        # Parse entry_point: "module.path:callable_name"
        if ":" in entry_point:
            module_path, callable_name = entry_point.rsplit(":", 1)
        else:
            module_path = entry_point
            callable_name = "setup"

        # Import the module
        module = self._import_module(module_path, info)
        if module is None:
            raise PluginLoadError(
                f"Could not import module {module_path!r}",
                plugin_name=info.name,
            )

        # Resolve the callable
        factory = getattr(module, callable_name, None)
        if factory is None:
            raise PluginLoadError(
                f"Entry-point callable {callable_name!r} not found in module {module_path!r}",
                plugin_name=info.name,
            )

        # If factory is a class, instantiate it; if callable, call it
        if isinstance(factory, type) and issubclass(factory, BasePlugin):
            instance = factory()
        elif callable(factory):
            instance = factory()
        else:
            raise PluginLoadError(
                f"Entry-point {callable_name!r} is not callable or a BasePlugin subclass",
                plugin_name=info.name,
            )

        if not isinstance(instance, BasePlugin):
            raise PluginLoadError(
                f"Entry-point returned {type(instance).__name__}, expected BasePlugin subclass",
                plugin_name=info.name,
            )

        # Lifecycle: on_register
        instance.plugin_info = info
        try:
            instance.on_register(manifest)
        except Exception as exc:
            raise PluginLoadError(
                f"on_register() failed: {exc}",
                plugin_name=info.name,
                original=exc,
            ) from exc

        # Lifecycle: on_load
        try:
            if asyncio.iscoroutinefunction(instance.on_load):
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Inside an event loop — schedule on_load via create_task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(asyncio.run, instance.on_load())
                        future.result(timeout=self._timeout or None)
                else:
                    loop.run_until_complete(instance.on_load())
            else:
                instance.on_load()
        except Exception as exc:
            raise PluginLoadError(
                f"on_load() failed: {exc}",
                plugin_name=info.name,
                original=exc,
            ) from exc

        instance._initialized = True

        # Track sys.path patches for cleanup
        patches: List[str] = []
        if info.path and str(info.path) not in sys.path:
            sys.path.insert(0, str(info.path))
            patches.append(str(info.path))

        return LoadedPlugin(
            info=info,
            instance=instance,
            module=module,
            import_time=time.monotonic(),
            sys_path_patches=patches,
        )

    def _import_module(
        self,
        module_path: str,
        info: PluginInfo,
    ) -> Optional[types.ModuleType]:
        """Import a module, optionally with isolation."""
        plugin_dir = info.path
        patches: List[str] = []

        try:
            # If the plugin has a directory, add it to sys.path temporarily
            if plugin_dir and plugin_dir.is_dir():
                dir_str = str(plugin_dir)
                if dir_str not in sys.path:
                    sys.path.insert(0, dir_str)
                    patches.append(dir_str)
                    logger.debug("Added %s to sys.path for plugin %s", dir_str, info.name)

            if self._isolation:
                return self._import_isolated(module_path, info)
            else:
                try:
                    return importlib.import_module(module_path)
                except ImportError:
                    # Retry with the parent package if module_path is dotted
                    parts = module_path.split(".")
                    if len(parts) > 1:
                        parent = ".".join(parts[:-1])
                        try:
                            return importlib.import_module(parent)
                        except ImportError:
                            pass
                    raise
        except ImportError as exc:
            logger.error(
                "Failed to import %s for plugin %s: %s",
                module_path,
                info.name,
                exc,
            )
            return None
        finally:
            # Restore sys.path
            for p in reversed(patches):
                try:
                    sys.path.remove(p)
                except ValueError:
                    pass

    def _import_isolated(
        self,
        module_path: str,
        info: PluginInfo,
    ) -> types.ModuleType:
        """Import a module in an isolated namespace."""
        plugin_dir = info.path
        base_name = module_path.split(".")[0]

        # Look for the module file
        module_file = None
        search_dirs = []
        if plugin_dir:
            search_dirs.append(plugin_dir)
        search_dirs.extend(Path(p) for p in sys.path if Path(p).is_dir())

        for d in search_dirs:
            candidate_py = d / f"{base_name.replace('.', '/')}.py"
            candidate_pkg = d / base_name.replace(".", "/") / "__init__.py"
            if candidate_py.exists():
                module_file = str(candidate_py)
                break
            if candidate_pkg.exists():
                module_file = str(candidate_pkg)
                break

        if module_file is None:
            # Fall back to standard import
            return importlib.import_module(module_path)

        # Use importlib.util for isolated loading
        unique_name = f"_atlas_plugin_{info.name}_{base_name}_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(unique_name, module_file)
        if spec is None or spec.loader is None:
            return importlib.import_module(module_path)

        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(unique_name, None)
            raise ImportError(f"Isolated import failed: {exc}") from exc

        # For dotted sub-module imports, also register under the original path
        sys.modules[module_path] = module
        logger.debug(
            "Isolated import of %s as %s (from %s)",
            module_path,
            unique_name,
            module_file,
        )
        return module

    def _load_dependencies(self, name: str) -> List[str]:
        """Load all dependencies of *name* that are not yet loaded.

        Returns a list of error messages (empty on success).
        """
        info = self._registry.get(name)
        if info is None:
            return [f"Plugin {name!r} not found"]

        errors: List[str] = []
        for dep_name in info.manifest.dependencies:
            if not self.is_loaded(dep_name):
                if not self._registry.has(dep_name):
                    errors.append(f"Dependency {dep_name!r} is not registered")
                    continue
                result = self.load(dep_name)
                if not result.success:
                    errors.append(
                        f"Failed to load dependency {dep_name!r}: {result.error}"
                    )
        return errors

    # -- Unloading ---------------------------------------------------------

    def unload(self, name: str) -> bool:
        """Unload a plugin and call ``on_unload``.

        Returns:
            ``True`` if the plugin was found and unloaded.
        """
        with self._lock:
            lp = self._loaded.pop(name, None)

        if lp is None:
            logger.warning("Cannot unload unknown plugin %r", name)
            return False

        # Fire pre-unload hooks
        for hook in self._pre_unload_hooks:
            try:
                hook(name)
            except Exception:
                logger.exception("Pre-unload hook error for %s", name)

        # Call on_unload
        try:
            instance = lp.instance
            if asyncio.iscoroutinefunction(instance.on_unload):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(instance.on_unload())
                except RuntimeError:
                    asyncio.run(instance.on_unload())
            else:
                instance.on_unload()
        except Exception as exc:
            logger.error("Error during on_unload for plugin %s: %s", name, exc)
        finally:
            instance._initialized = False

        # Clean up sys.path patches
        for p in lp.sys_path_patches:
            try:
                sys.path.remove(p)
            except ValueError:
                pass

        # Clean up sys.modules
        if lp.module is not None:
            module_name = lp.module.__name__
            sys.modules.pop(module_name, None)
            # Also clean up the original module path if registered during isolation
            ep = lp.info.manifest.entry_point
            if ":" in ep:
                mod_path = ep.rsplit(":", 1)[0]
                sys.modules.pop(mod_path, None)

        self._registry.set_state(name, PluginState.DISABLED)
        lp.info.instance = None

        # Fire post-unload hooks
        for hook in self._post_unload_hooks:
            try:
                hook(name)
            except Exception:
                logger.exception("Post-unload hook error for %s", name)

        logger.info("Unloaded plugin %s", name)
        return True

    # -- Hot reload --------------------------------------------------------

    def reload(self, name: str) -> LoadResult:
        """Hot-reload a plugin: unload, re-import, re-instantiate.

        Returns:
            A :class:`LoadResult` for the re-load attempt.
        """
        info = self._registry.get(name)
        if info is None:
            return LoadResult(name=name, success=False, error=f"Plugin {name!r} not found")

        # Unload first
        self.unload(name)

        # Re-register (transition back to DISCOVERED)
        self._registry.set_state(name, PluginState.DISCOVERED)

        # Re-import
        if info.module:
            try:
                importlib.reload(info.module)
            except Exception:
                pass

        # Load again
        result = self.load(name)

        # Track reload count
        with self._lock:
            if name in self._loaded:
                lp = self._loaded[name]
                object.__setattr__(lp, "reload_count", lp.reload_count + 1)

        if result.success:
            logger.info("Hot-reloaded plugin %s (attempt #%d)", name, self._loaded.get(name, LoadedPlugin(info=info, instance=None)).reload_count if name in self._loaded else 1)
        else:
            logger.warning("Hot-reload failed for plugin %s: %s", name, result.error)

        return result

    # -- Bulk operations ---------------------------------------------------

    def load_all(self) -> List[LoadResult]:
        """Load all discovered plugins in dependency order.

        Returns a list of :class:`LoadResult` for each plugin attempted.
        """
        load_order = self._registry.get_load_order()
        results: List[LoadResult] = []

        logger.info("Loading %d plugin(s) in order: %s", len(load_order), load_order)

        for name in load_order:
            result = self.load(name)
            results.append(result)

        # Summary
        succeeded = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        logger.info(
            "load_all complete: %d succeeded, %d failed out of %d",
            succeeded,
            failed,
            len(results),
        )

        return results

    def unload_all(self) -> int:
        """Unload all loaded plugins in reverse dependency order.

        Returns the number of plugins unloaded.
        """
        # Reverse dependency order
        load_order = list(reversed(self._registry.get_load_order()))
        # Also add any active plugins not in the load order
        for name in list(self._loaded.keys()):
            if name not in load_order:
                load_order.append(name)

        count = 0
        for name in load_order:
            if self.is_loaded(name):
                self.unload(name)
                count += 1

        logger.info("unload_all complete: %d plugin(s) unloaded", count)
        return count

    # -- Tool / hook / command extraction ----------------------------------

    def get_plugin_tools(self, name: str) -> List[ToolDefinition]:
        """Return the tools exposed by a loaded plugin.

        Raises:
            PluginLoadError: If the plugin is not loaded.
        """
        lp = self._loaded.get(name)
        if lp is None:
            raise PluginLoadError(
                f"Plugin {name!r} is not loaded",
                plugin_name=name,
            )
        return lp.instance.get_tools()

    def get_plugin_hooks(self, name: str) -> List[HookDefinition]:
        """Return the hooks exposed by a loaded plugin.

        Raises:
            PluginLoadError: If the plugin is not loaded.
        """
        lp = self._loaded.get(name)
        if lp is None:
            raise PluginLoadError(
                f"Plugin {name!r} is not loaded",
                plugin_name=name,
            )
        return lp.instance.get_hooks()

    def get_plugin_commands(self, name: str) -> List[CommandDefinition]:
        """Return the commands exposed by a loaded plugin.

        Raises:
            PluginLoadError: If the plugin is not loaded.
        """
        lp = self._loaded.get(name)
        if lp is None:
            raise PluginLoadError(
                f"Plugin {name!r} is not loaded",
                plugin_name=name,
            )
        return lp.instance.get_commands()

    def get_all_tools(self) -> Dict[str, List[ToolDefinition]]:
        """Return tools from all loaded plugins, grouped by plugin name."""
        result: Dict[str, List[ToolDefinition]] = {}
        for name, lp in self._loaded.items():
            if lp.info.state == PluginState.ACTIVE:
                tools = lp.instance.get_tools()
                if tools:
                    result[name] = tools
        return result

    def get_all_hooks(self) -> Dict[str, List[HookDefinition]]:
        """Return hooks from all loaded plugins, grouped by plugin name."""
        result: Dict[str, List[HookDefinition]] = {}
        for name, lp in self._loaded.items():
            if lp.info.state == PluginState.ACTIVE:
                hooks = lp.instance.get_hooks()
                if hooks:
                    result[name] = hooks
        return result

    def get_all_commands(self) -> Dict[str, List[CommandDefinition]]:
        """Return commands from all loaded plugins, grouped by plugin name."""
        result: Dict[str, List[CommandDefinition]] = {}
        for name, lp in self._loaded.items():
            if lp.info.state == PluginState.ACTIVE:
                commands = lp.instance.get_commands()
                if commands:
                    result[name] = commands
        return result

    # -- Lifecycle hooks ---------------------------------------------------

    def on_pre_load(self, callback: Callable[[str], None]) -> Callable[[str], None]:
        """Register a callback fired before a plugin is loaded."""
        self._pre_load_hooks.append(callback)
        return callback

    def on_post_load(self, callback: Callable[[LoadedPlugin], None]) -> Callable[[LoadedPlugin], None]:
        """Register a callback fired after a plugin is loaded successfully."""
        self._post_load_hooks.append(callback)
        return callback

    def on_pre_unload(self, callback: Callable[[str], None]) -> Callable[[str], None]:
        """Register a callback fired before a plugin is unloaded."""
        self._pre_unload_hooks.append(callback)
        return callback

    def on_post_unload(self, callback: Callable[[str], None]) -> Callable[[str], None]:
        """Register a callback fired after a plugin is unloaded."""
        self._post_unload_hooks.append(callback)
        return callback

    # -- Introspection -----------------------------------------------------

    def get_plugin_instance(self, name: str) -> Optional[BasePlugin]:
        """Return the live :class:`BasePlugin` instance for *name*, or ``None``."""
        lp = self._loaded.get(name)
        return lp.instance if lp else None

    def get_plugin_class(self, name: str) -> Optional[Type[BasePlugin]]:
        """Return the concrete class of a loaded plugin."""
        lp = self._loaded.get(name)
        if lp is None:
            return None
        return type(lp.instance)

    def get_plugin_module(self, name: str) -> Optional[types.ModuleType]:
        """Return the imported module for a loaded plugin."""
        lp = self._loaded.get(name)
        return lp.module if lp else None

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the loader's current state."""
        plugins: List[Dict[str, Any]] = []
        for name, lp in self._loaded.items():
            plugins.append({
                "name": name,
                "state": str(lp.info.state),
                "version": lp.info.version,
                "class": type(lp.instance).__name__,
                "tools": len(lp.instance.get_tools()),
                "hooks": len(lp.instance.get_hooks()),
                "commands": len(lp.instance.get_commands()),
                "reload_count": lp.reload_count,
                "duration_s": round(lp.info.duration_active, 1),
            })

        return {
            "loaded_count": len(self._loaded),
            "plugins": plugins,
        }

    def __repr__(self) -> str:
        return f"<PluginLoader loaded={len(self._loaded)}>"
