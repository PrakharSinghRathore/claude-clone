"""
Memory Plugin Registry — Discovery, loading, health checks, and configuration.

Plugins are discovered from YAML manifests or registered programmatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Type

import yaml

from .base import (
    BaseMemoryPlugin,
    MemoryConfig,
    MemoryPluginMetadata,
    MemoryPluginType,
)

logger = logging.getLogger(__name__)

# Built-in plugin classes mapped by name
_BUILTIN_PLUGINS: dict[str, Type[BaseMemoryPlugin]] = {}


def register_builtin(name: str):
    """Decorator to register a plugin class as a built-in."""
    def decorator(cls: Type[BaseMemoryPlugin]) -> Type[BaseMemoryPlugin]:
        _BUILTIN_PLUGINS[name] = cls
        return cls
    return decorator


class MemoryPluginRegistry:
    """
    Central registry for memory plugins.

    Discovers plugins from YAML manifests in a plugins directory,
    loads them dynamically, and provides unified access to all
    registered backends.
    """

    def __init__(self, plugins_dir: Optional[str | Path] = None) -> None:
        self.plugins_dir = Path(plugins_dir).expanduser().resolve() if plugins_dir else None
        self._plugins: dict[str, BaseMemoryPlugin] = {}
        self._metadata: dict[str, MemoryPluginMetadata] = {}
        self._configs: dict[str, MemoryConfig] = {}
        self._lock = __import__("asyncio").Lock()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def discover(self) -> list[MemoryPluginMetadata]:
        """
        Scan the plugins directory for YAML manifests and return
        metadata for every discovered plugin.
        """
        discovered: list[MemoryPluginMetadata] = []

        if self.plugins_dir and self.plugins_dir.is_dir():
            for manifest_path in sorted(self.plugins_dir.glob("plugin.yaml")):
                try:
                    meta = self._load_manifest(manifest_path)
                    if meta:
                        self._metadata[meta.name] = meta
                        discovered.append(meta)
                except Exception:
                    logger.exception("Failed to load manifest %s", manifest_path)

        # Also register built-in plugins
        for name, cls in _BUILTIN_PLUGINS.items():
            if hasattr(cls, "metadata") and not isinstance(cls.metadata, property):
                meta = cls.metadata  # type: ignore[attr-defined]
                if isinstance(meta, MemoryPluginMetadata):
                    self._metadata[name] = meta
                    if meta not in discovered:
                        discovered.append(meta)

        logger.info("Discovered %d memory plugin(s)", len(discovered))
        return discovered

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    async def load_plugin(
        self,
        name: str,
        config: Optional[MemoryConfig] = None,
    ) -> BaseMemoryPlugin:
        """
        Load and initialize a plugin by name.

        Checks built-in plugins first, then tries to dynamically import
        from the plugins directory.
        """
        async with self._lock:
            if name in self._plugins:
                logger.debug("Plugin %r already loaded", name)
                return self._plugins[name]

            plugin_class = self._resolve_plugin_class(name)
            plugin_cfg = config or self._configs.get(name, MemoryConfig())
            plugin = plugin_class(config=plugin_cfg)
            await plugin.initialize()
            self._plugins[name] = plugin
            logger.info("Loaded memory plugin %r", name)
            return plugin

    async def load_all(self) -> dict[str, BaseMemoryPlugin]:
        """Load all discovered plugins with their default configs."""
        await self.discover()
        for name, meta in self._metadata.items():
            if meta.plugin_type.value not in ("post_hoc",):
                try:
                    await self.load_plugin(name)
                except Exception:
                    logger.exception("Failed to load plugin %r", name)
        return dict(self._plugins)

    async def unload_plugin(self, name: str) -> None:
        """Unload and shut down a plugin."""
        async with self._lock:
            plugin = self._plugins.pop(name, None)
            if plugin:
                await plugin.shutdown()
                logger.info("Unloaded memory plugin %r", name)

    async def unload_all(self) -> None:
        """Shut down all loaded plugins."""
        for name in list(self._plugins):
            await self.unload_plugin(name)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_plugin(self, name: str) -> Optional[BaseMemoryPlugin]:
        """Return a loaded plugin by name."""
        return self._plugins.get(name)

    def list_loaded(self) -> dict[str, BaseMemoryPlugin]:
        """Return all loaded plugins."""
        return dict(self._plugins)

    def list_metadata(self) -> dict[str, MemoryPluginMetadata]:
        """Return metadata for all discovered plugins."""
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def health_check_all(self) -> dict[str, dict]:
        """Run health checks on all loaded plugins."""
        results: dict[str, dict] = {}
        for name, plugin in self._plugins.items():
            try:
                result = await plugin.health_check()
                results[name] = result
            except Exception:
                logger.exception("Health check failed for %r", name)
                results[name] = {"status": "unhealthy", "error": "check failed"}
        return results

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_config(self, name: str, config: MemoryConfig) -> None:
        """Set configuration for a plugin (before loading)."""
        self._configs[name] = config

    def get_config(self, name: str) -> Optional[MemoryConfig]:
        """Get the stored configuration for a plugin."""
        return self._configs.get(name)

    # ------------------------------------------------------------------
    # Unified memory operations
    # ------------------------------------------------------------------

    async def unified_store(self, content: str, metadata: Optional[dict] = None, tags: Optional[list[str]] = None) -> dict[str, str]:
        """Store content across all loaded plugins. Returns {plugin_name: entry_id}."""
        from .base import MemoryEntry
        results: dict[str, str] = {}
        for name, plugin in self._plugins.items():
            try:
                entry = MemoryEntry(
                    id="",
                    content=content,
                    metadata=metadata or {},
                    tags=tags or [],
                    source=name,
                )
                eid = await plugin.store(entry)
                results[name] = eid
            except Exception:
                logger.exception("Store failed for plugin %r", name)
        return results

    async def unified_search(self, query: str, limit: int = 10) -> list[dict]:
        """Search across all loaded plugins and merge results."""
        all_results: list[dict] = []
        for name, plugin in self._plugins.items():
            try:
                entries = await plugin.search(query, limit=limit)
                for entry in entries:
                    all_results.append({
                        "plugin": name,
                        "id": entry.id,
                        "content": entry.content,
                        "relevance": entry.relevance_score,
                        "tags": entry.tags,
                    })
            except Exception:
                logger.exception("Search failed for plugin %r", name)
        all_results.sort(key=lambda r: r["relevance"], reverse=True)
        return all_results[:limit]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_plugin_class(self, name: str) -> Type[BaseMemoryPlugin]:
        """Resolve a plugin name to its class."""
        # Check built-ins
        if name in _BUILTIN_PLUGINS:
            return _BUILTIN_PLUGINS[name]

        # Check module attribute for metadata-based class name
        meta = self._metadata.get(name)
        if meta and hasattr(meta, "plugin_type"):
            class_name = meta.plugin_type.value
            if class_name in _BUILTIN_PLUGINS:
                return _BUILTIN_PLUGINS[class_name]

        # Dynamic import from plugins directory
        if self.plugins_dir:
            module_path = self.plugins_dir / f"{name}.py"
            if module_path.exists():
                return self._import_from_path(name, module_path)

        raise ImportError(f"Memory plugin {name!r} not found")

    @staticmethod
    def _import_from_path(name: str, path: Path) -> Type[BaseMemoryPlugin]:
        """Dynamically import a plugin class from a .py file."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"hermes_memory_{name}", str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseMemoryPlugin)
                and attr is not BaseMemoryPlugin
            ):
                return attr
        raise ImportError(f"No BaseMemoryPlugin subclass found in {path}")

    @staticmethod
    def _load_manifest(path: Path) -> Optional[MemoryPluginMetadata]:
        """Parse a plugin.yaml manifest."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or not isinstance(data, dict):
            return None
        return MemoryPluginMetadata(
            name=data.get("name", path.stem),
            display_name=data.get("display_name", data.get("name", path.stem)),
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            plugin_type=MemoryPluginType(data.get("type", "semantic")),
            author=data.get("author", ""),
            homepage=data.get("homepage", ""),
            config_schema=data.get("config_schema", {}),
            required_packages=data.get("required_packages", []),
        )
