"""
Dynamic Plugin System for Claude Clone.

Provides hot-reloadable plugins with dependency management, hook-based
extensibility, tool registration, and a built-in template generator
for scaffolding new plugins.

Each plugin is a standard Python module living in the plugin directory
(`~/.claude_clone/plugins` by default) that exposes a well-known interface:

    plugin_name           (required)  - Unique identifier string
    plugin_version        (required)  - PEP-440 compatible version
    plugin_description    (required)  - One-line human-readable summary

    plugin_author         (optional)  - Author name / contact
    plugin_tools          (optional)  - List of tool definitions
    plugin_hooks          (optional)  - Mapping of PluginHook -> callable
    plugin_config_schema  (optional)  - JSON Schema for plugin settings
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PLUGIN_DIR = "~/.claude_clone/plugins"
_CONFIG_FILENAME = "plugin_config.json"
_MANIFEST_FILENAME = "plugin_manifest.json"

_PLUGIN_TEMPLATE = '''\
"""
{plugin_name} – {description}

A plugin for Claude Clone.
"""

from claude_clone.plugins.loader import PluginHook

# ── Required metadata ──────────────────────────────────────────────────────
plugin_name = "{plugin_name}"
plugin_version = "0.1.0"
plugin_description = "{description}"

# ── Optional metadata ──────────────────────────────────────────────────────
plugin_author = "{author}"

# ── Tools ──────────────────────────────────────────────────────────────────
# Each entry must have: name, description, parameters (JSON Schema), handler
#
# plugin_tools = [
#     {{
#         "name": "example_tool",
#         "description": "A brief description of what this tool does.",
#         "parameters": {{
#             "type": "object",
#             "properties": {{
#                 "query": {{
#                     "type": "string",
#                     "description": "The input query to process."
#                 }},
#             }},
#             "required": ["query"],
#         }},
#         "handler": example_handler,
#         "category": "example",
#         "dangerous": False,
#     }},
# ]

# ── Hooks ──────────────────────────────────────────────────────────────────
# plugin_hooks = {{
#     PluginHook.PRE_TOOL_CALL: on_pre_tool_call,
#     PluginHook.POST_TOOL_CALL: on_post_tool_call,
# }}

# ── Configuration schema (JSON Schema) ────────────────────────────────────
# plugin_config_schema = {{
#     "type": "object",
#     "properties": {{
#         "api_key": {{
#             "type": "string",
#             "description": "Optional API key for the service.",
#         }},
#     }},
# }}


# ── Implementations ────────────────────────────────────────────────────────

async def example_handler(context: dict) -> dict:
    """Example tool handler – replace with real logic."""
    query = context.get("query", "")
    return {{
        "output": f"Processed: {{query}}",
        "success": True,
    }}


async def on_pre_tool_call(context: dict) -> dict | None:
    """Hook: runs before a tool is called."""
    # Return a dict to modify the context, or None to pass-through.
    return None


async def on_post_tool_call(context: dict) -> dict | None:
    """Hook: runs after a tool call completes."""
    return None
'''

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class PluginHook(str, Enum):
    """Lifecycle hooks that plugins can subscribe to."""

    PRE_EXECUTION = "pre_execution"
    POST_EXECUTION = "post_execution"
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"
    ON_ERROR = "on_error"
    ON_STARTUP = "on_startup"
    ON_SHUTDOWN = "on_shutdown"
    ON_MESSAGE = "on_message"
    ON_RESPONSE = "on_response"


@dataclass
class PluginTool:
    """A single tool exposed by a plugin."""

    name: str
    description: str
    parameters: dict
    handler: Callable
    category: str = "general"
    dangerous: bool = False


@dataclass
class Plugin:
    """Represents a loaded plugin and all of its registered extensions."""

    name: str
    version: str
    description: str
    author: str = ""
    tools: list[PluginTool] = field(default_factory=list)
    hooks: dict[str, Callable] = field(default_factory=dict)
    enabled: bool = True
    config_schema: dict = field(default_factory=dict)

    # --- internal bookkeeping (not part of the public API) ---
    _module: Any = field(default=None, repr=False)
    _source_path: str | None = field(default=None, repr=False)
    _load_time: float = field(default_factory=time.time, repr=False)
    _config: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary of this plugin."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "dangerous": t.dangerous,
                }
                for t in self.tools
            ],
            "hooks": list(self.hooks.keys()),
            "enabled": self.enabled,
            "config_schema": self.config_schema,
        }


# ---------------------------------------------------------------------------
# Plugin Manager
# ---------------------------------------------------------------------------


class PluginManager:
    """
    Central manager for discovering, loading, hot-reloading, installing,
    and executing Claude Clone plugins.
    """

    def __init__(self, plugin_dir: str = _DEFAULT_PLUGIN_DIR) -> None:
        self.plugin_dir = Path(plugin_dir).expanduser().resolve()
        self._plugins: dict[str, Plugin] = {}
        self._module_cache: dict[str, Any] = {}
        self._watcher_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._initialised = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Discover and load every plugin found in *plugin_dir*."""
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        names = self.discover_plugins()
        for name in names:
            try:
                await self.load_plugin(name)
            except Exception:
                logger.exception("Failed to load plugin %r during initialisation", name)
        self._setup_watcher()
        self._initialised = True
        logger.info(
            "Plugin system initialised – %d plugin(s) discovered, %d loaded",
            len(names),
            len(self._plugins),
        )

    # ------------------------------------------------------------------
    # Load / unload / reload
    # ------------------------------------------------------------------

    async def load_plugin(self, name: str) -> Plugin:
        """
        Load a single plugin by *name* from the plugin directory.

        The plugin file must be ``<plugin_dir>/<name>.py`` or a package
        directory ``<plugin_dir>/<name>/__init__.py``.
        """
        async with self._lock:
            if name in self._plugins:
                logger.debug("Plugin %r already loaded; reloading.", name)
                return await self._reload_plugin_internal(name)

            module = self._import_plugin_module(name)
            plugin = self._validate_plugin(module)
            plugin._source_path = str(self._find_plugin_path(name))

            # Apply any persisted configuration
            stored_config = await self._read_stored_config(name)
            if stored_config:
                plugin._config = stored_config

            # Run ON_STARTUP hooks
            if PluginHook.ON_STARTUP in plugin.hooks:
                try:
                    result = plugin.hooks[PluginHook.ON_STARTUP]({"plugin_name": name})
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception("ON_STARTUP hook failed for plugin %r", name)

            self._plugins[name] = plugin
            self._module_cache[name] = module
            logger.info("Loaded plugin %r v%s", name, plugin.version)
            return plugin

    async def unload_plugin(self, name: str) -> None:
        """Unload a plugin, running its ON_SHUTDOWN hooks first."""
        async with self._lock:
            plugin = self._plugins.pop(name, None)
            if plugin is None:
                logger.warning("Cannot unload plugin %r – not loaded", name)
                return

            # Run ON_SHUTDOWN hooks
            if PluginHook.ON_SHUTDOWN in plugin.hooks:
                try:
                    result = plugin.hooks[PluginHook.ON_SHUTDOWN]({"plugin_name": name})
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception("ON_SHUTDOWN hook failed for plugin %r", name)

            # Remove from sys.modules to release references
            mod_key = f"{name}.plugin" if name in sys.modules else name
            for key in list(sys.modules):
                if key == mod_key or key.startswith(f"{mod_key}."):
                    del sys.modules[key]
            self._module_cache.pop(name, None)
            logger.info("Unloaded plugin %r", name)

    async def reload_plugin(self, name: str) -> Plugin:
        """Public hot-reload entry-point (acquires the lock)."""
        async with self._lock:
            return await self._reload_plugin_internal(name)

    async def _reload_plugin_internal(self, name: str) -> Plugin:
        """Internal reload without additional locking."""
        old_plugin = self._plugins.get(name)
        if old_plugin is None:
            raise KeyError(f"Plugin {name!r} is not loaded")

        # Run ON_SHUTDOWN on the old version
        if PluginHook.ON_SHUTDOWN in old_plugin.hooks:
            try:
                result = old_plugin.hooks[PluginHook.ON_SHUTDOWN]({"plugin_name": name})
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("ON_SHUTDOWN hook failed for plugin %r during reload", name)

        # Re-import the module
        module = self._import_plugin_module(name, force_reload=True)
        plugin = self._validate_plugin(module)
        plugin._source_path = str(self._find_plugin_path(name))

        # Preserve persisted config
        stored_config = await self._read_stored_config(name)
        if stored_config:
            plugin._config = stored_config

        # Run ON_STARTUP on the new version
        if PluginHook.ON_STARTUP in plugin.hooks:
            try:
                result = plugin.hooks[PluginHook.ON_STARTUP]({"plugin_name": name})
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("ON_STARTUP hook failed for plugin %r during reload", name)

        self._plugins[name] = plugin
        self._module_cache[name] = module
        logger.info("Reloaded plugin %r -> v%s", name, plugin.version)
        return plugin

    async def reload_all(self) -> list[Plugin]:
        """Hot-reload every currently-loaded plugin."""
        reloaded: list[Plugin] = []
        for name in list(self._plugins):
            try:
                p = await self.reload_plugin(name)
                reloaded.append(p)
            except Exception:
                logger.exception("Failed to reload plugin %r", name)
        return reloaded

    # ------------------------------------------------------------------
    # Install / uninstall
    # ------------------------------------------------------------------

    async def install_plugin(self, source: str) -> Plugin:
        """
        Install a plugin from *source* which can be:

        * A **pip package name** (e.g. ``claude-clone-plugin-foo``)
        * A **git URL** (https://... or git://... or git+https://...)
        * A **local file-system path** to a ``.py`` file or directory
        """
        plugin_name: str | None = None

        # --- Git URL ---
        if source.startswith(("git://", "https://", "git+https://", "git+ssh://")):
            plugin_name = await self._install_from_git(source)

        # --- Local path ---
        elif os.path.exists(source):
            plugin_name = await self._install_from_path(source)

        # --- pip package ---
        else:
            plugin_name = await self._install_from_pip(source)

        if plugin_name is None:
            raise RuntimeError(f"Could not determine plugin name from source {source!r}")

        return await self.load_plugin(plugin_name)

    async def uninstall_plugin(self, name: str) -> None:
        """Unload and delete a plugin from the plugin directory."""
        await self.unload_plugin(name)
        path = self._find_plugin_path(name)
        if path is None:
            logger.warning("No files found for plugin %r to remove", name)
            return
        target = Path(path)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.is_file():
            target.unlink()
        # Clean up persisted config
        config_path = self.plugin_dir / name / _CONFIG_FILENAME
        if config_path.exists():
            config_path.unlink()
        logger.info("Uninstalled and removed plugin %r", name)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def list_plugins(self) -> list[Plugin]:
        """Return all loaded plugins."""
        return list(self._plugins.values())

    async def get_plugin(self, name: str) -> Plugin:
        """Retrieve a single plugin by name."""
        if name not in self._plugins:
            raise KeyError(f"Plugin {name!r} not loaded")
        return self._plugins[name]

    async def get_all_tools(self) -> list[PluginTool]:
        """Aggregate tools from every enabled plugin."""
        tools: list[PluginTool] = []
        for plugin in self._plugins.values():
            if plugin.enabled:
                tools.extend(plugin.tools)
        return tools

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    async def execute_hook(self, hook: PluginHook, context: dict) -> list[Any]:
        """
        Execute all handlers registered for *hook* across every enabled
        plugin.  Results (or None for pass-through) are collected in a list.
        """
        results: list[Any] = []
        for plugin in self._plugins.values():
            if not plugin.enabled:
                continue
            handler = plugin.hooks.get(hook)
            if handler is None:
                continue
            try:
                enriched_ctx = {**context, "__plugin_name__": plugin.name}
                result = handler(enriched_ctx)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, dict):
                    context.update(result)
                results.append(result)
            except Exception:
                logger.exception(
                    "Hook %r handler failed in plugin %r", hook, plugin.name
                )
        return results

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    async def check_updates(self) -> list[dict]:
        """
        Check for available updates on plugins that were installed from
        pip or git.  Returns a list of ``{"name": ..., "current": ..., "latest": ...}``.
        """
        updates: list[dict] = []
        for plugin in self._plugins.values():
            manifest = self._load_manifest(plugin.name)
            if not manifest or "install_source" not in manifest:
                continue
            source = manifest["install_source"]
            try:
                if source.startswith(("git://", "https://", "git+https://", "git+ssh://")):
                    latest = await self._git_latest_version(source)
                else:
                    latest = await self._pip_latest_version(source)
                if latest and latest != plugin.version:
                    updates.append(
                        {
                            "name": plugin.name,
                            "current": plugin.version,
                            "latest": latest,
                        }
                    )
            except Exception:
                logger.exception("Update check failed for plugin %r", plugin.name)
        return updates

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    async def get_plugin_config(self, name: str) -> dict:
        """Return the current runtime config merged with persisted values."""
        plugin = self._plugins.get(name)
        if plugin is None:
            raise KeyError(f"Plugin {name!r} not loaded")
        stored = await self._read_stored_config(name)
        merged = {**plugin._config, **stored}
        return merged

    async def set_plugin_config(self, name: str, config: dict) -> None:
        """Persist configuration for a plugin."""
        plugin = self._plugins.get(name)
        if plugin is None:
            raise KeyError(f"Plugin {name!r} not loaded")
        plugin._config = {**plugin._config, **config}
        await self._write_stored_config(name, plugin._config)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_plugins(self) -> list[str]:
        """
        Scan *plugin_dir* for plugin modules and return their names.

        A plugin is either:
        * ``<name>.py``  – single-file module
        * ``<name>/__init__.py`` – package module
        """
        names: list[str] = []
        if not self.plugin_dir.is_dir():
            return names
        for entry in sorted(self.plugin_dir.iterdir()):
            if entry.is_file() and entry.suffix == ".py" and not entry.name.startswith("_"):
                names.append(entry.stem)
            elif entry.is_dir() and (entry / "__init__.py").is_file() and not entry.name.startswith("_"):
                names.append(entry.name)
        return names

    # ------------------------------------------------------------------
    # Template generator
    # ------------------------------------------------------------------

    async def generate_plugin(
        self,
        name: str,
        description: str = "A new Claude Clone plugin.",
        author: str = "Unknown",
        path: str | None = None,
    ) -> str:
        """
        Create a skeleton plugin file from the built-in template.

        Returns the absolute path of the generated file.
        """
        sanitised = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        content = _PLUGIN_TEMPLATE.format(
            plugin_name=sanitised,
            description=description,
            author=author,
        )
        if path is None:
            target_dir = self.plugin_dir
        else:
            target_dir = Path(path).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{sanitised}.py"
        target_file.write_text(content, encoding="utf-8")
        logger.info("Generated plugin template at %s", target_file)
        return str(target_file)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_plugin(self, plugin_module: Any) -> Plugin:
        """
        Validate that *plugin_module* exposes the required interface and
        return a :class:`Plugin` instance.
        """
        for attr in ("plugin_name", "plugin_version", "plugin_description"):
            if not hasattr(plugin_module, attr):
                raise ValueError(
                    f"Plugin module {getattr(plugin_module, '__file__', '<unknown>')} "
                    f"is missing required attribute {attr!r}"
                )

        name = str(getattr(plugin_module, "plugin_name"))
        version = str(getattr(plugin_module, "plugin_version"))
        description = str(getattr(plugin_module, "plugin_description"))
        author = str(getattr(plugin_module, "plugin_author", ""))

        # --- Tools ---
        tools: list[PluginTool] = []
        raw_tools = getattr(plugin_module, "plugin_tools", None) or []
        for tool_def in raw_tools:
            if not isinstance(tool_def, dict):
                raise TypeError(f"Tool definition must be a dict, got {type(tool_def)}")
            required_keys = {"name", "description", "parameters", "handler"}
            missing = required_keys - tool_def.keys()
            if missing:
                raise ValueError(f"Tool {tool_def.get('name')!r} is missing keys: {missing}")
            if not callable(tool_def["handler"]):
                raise TypeError(f"Tool {tool_def['name']!r} handler must be callable")
            tools.append(
                PluginTool(
                    name=str(tool_def["name"]),
                    description=str(tool_def["description"]),
                    parameters=tool_def["parameters"],
                    handler=tool_def["handler"],
                    category=str(tool_def.get("category", "general")),
                    dangerous=bool(tool_def.get("dangerous", False)),
                )
            )

        # --- Hooks ---
        hooks: dict[str, Callable] = {}
        raw_hooks = getattr(plugin_module, "plugin_hooks", None) or {}
        for hook_key, handler in raw_hooks.items():
            if isinstance(hook_key, PluginHook):
                key = hook_key.value
            elif isinstance(hook_key, str):
                key = hook_key
            else:
                key = str(hook_key)
            if not callable(handler):
                raise TypeError(f"Hook handler for {key!r} must be callable")
            hooks[key] = handler

        # --- Config schema ---
        config_schema = getattr(plugin_module, "plugin_config_schema", None) or {}

        return Plugin(
            name=name,
            version=version,
            description=description,
            author=author,
            tools=tools,
            hooks=hooks,
            enabled=True,
            config_schema=config_schema,
            _module=plugin_module,
        )

    # ------------------------------------------------------------------
    # Internal: module loading
    # ------------------------------------------------------------------

    def _find_plugin_path(self, name: str) -> Path | None:
        """Return the filesystem path for plugin *name*, or None."""
        py_file = self.plugin_dir / f"{name}.py"
        if py_file.is_file():
            return py_file
        pkg_dir = self.plugin_dir / name
        if pkg_dir.is_dir() and (pkg_dir / "__init__.py").is_file():
            return pkg_dir
        return None

    def _import_plugin_module(self, name: str, force_reload: bool = False) -> Any:
        """Import a plugin module by name from the plugin directory."""
        plugin_path = self._find_plugin_path(name)
        if plugin_path is None:
            raise FileNotFoundError(f"Plugin {name!r} not found in {self.plugin_dir}")

        if force_reload and name in sys.modules:
            del sys.modules[name]

        if plugin_path.is_dir():
            # Package import
            spec = importlib.util.spec_from_file_location(
                name, str(plugin_path / "__init__.py"), submodule_search_locations=[str(plugin_path)]
            )
        else:
            spec = importlib.util.spec_from_file_location(name, str(plugin_path))

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for plugin {name!r}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    # ------------------------------------------------------------------
    # Internal: install helpers
    # ------------------------------------------------------------------

    async def _install_from_pip(self, package_name: str) -> str:
        """Install a plugin via pip and copy it into the plugin directory."""
        logger.info("Installing plugin from pip: %s", package_name)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", "--target", str(self.plugin_dir), package_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"pip install failed: {stderr.decode(errors='replace')}")

        # Discover the installed plugin name from the directory
        discovered = self.discover_plugins()
        if not discovered:
            raise RuntimeError("pip install succeeded but no plugin found in plugin dir")
        # Heuristic: pick the newest .py file we didn't have before
        return discovered[-1] if discovered else package_name

    async def _install_from_git(self, url: str) -> str:
        """Clone a git repo, install it, and copy into plugin dir."""
        logger.info("Installing plugin from git: %s", url)
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth", "1", url, tmpdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"git clone failed: {stderr.decode(errors='replace')}")

            # Look for plugin files in the repo
            repo = Path(tmpdir)
            plugin_files = list(repo.glob("*.py")) + [
                d for d in repo.iterdir() if d.is_dir() and (d / "__init__.py").is_file()
            ]
            if not plugin_files:
                # Try common subdirectories
                for subdir in ("src", "plugin", "plugins", repo.name):
                    candidate = repo / subdir
                    if candidate.is_dir():
                        plugin_files = list(candidate.glob("*.py")) + [
                            d for d in candidate.iterdir()
                            if d.is_dir() and (d / "__init__.py").is_file()
                        ]
                        if plugin_files:
                            break
            if not plugin_files:
                raise RuntimeError("No plugin module found in the git repository")

            plugin_name = plugin_files[0].stem if plugin_files[0].is_file() else plugin_files[0].name
            dest = self.plugin_dir / plugin_files[0].name
            if plugin_files[0].is_dir():
                shutil.copytree(str(plugin_files[0]), str(dest), dirs_exist_ok=True)
            else:
                shutil.copy2(str(plugin_files[0]), dest)

            # Save manifest for update tracking
            self._save_manifest(plugin_name, {"install_source": url, "install_type": "git"})
            return plugin_name

    async def _install_from_path(self, source: str) -> str:
        """Copy a local plugin file or directory into the plugin directory."""
        src = Path(source).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Path not found: {source}")

        if src.is_file():
            if src.suffix != ".py":
                raise ValueError("Source file must be a .py module")
            name = src.stem
            dest = self.plugin_dir / src.name
            shutil.copy2(str(src), dest)
        elif src.is_dir():
            if not (src / "__init__.py").is_file():
                raise ValueError("Source directory must contain __init__.py")
            name = src.name
            dest = self.plugin_dir / name
            shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
        else:
            raise ValueError(f"Unsupported source type: {source}")

        self._save_manifest(name, {"install_source": str(source), "install_type": "local"})
        return name

    # ------------------------------------------------------------------
    # Internal: update helpers
    # ------------------------------------------------------------------

    async def _pip_latest_version(self, package_name: str) -> str | None:
        """Query PyPI for the latest version of a pip package."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "index", "versions", package_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        output = stdout.decode(errors="replace")
        # Parse "Available versions: 1.2.3, 1.2.4, ..."
        match = re.search(r"Available versions:\s*([\d.,\s]+)", output)
        if match:
            versions = [v.strip() for v in match.group(1).split(",") if v.strip()]
            if versions:
                # Try to sort by semantic versioning
                def _sort_key(v: str) -> tuple:
                    parts = re.split(r"[.-]", v)
                    result: list[int] = []
                    for p in parts:
                        try:
                            result.append(int(p))
                        except ValueError:
                            result.append(0)
                    return tuple(result)

                return max(versions, key=_sort_key)
        return None

    async def _git_latest_version(self, url: str) -> str | None:
        """Try to determine the latest git tag / commit for update checks."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "ls-remote", "--tags", "--sort=-v:refname", url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            lines = stdout.decode(errors="replace").strip().splitlines()
            for line in lines:
                tag = line.split("\t")[-1].strip().replace("refs/tags/", "")
                # Skip ^{} dereferenced tags
                if tag.endswith("^{}"):
                    continue
                # Return the most recent tag
                return tag.lstrip("v")
        except Exception:
            logger.debug("Git version check failed for %s", url, exc_info=True)
        return None

    # ------------------------------------------------------------------
    # Internal: config persistence
    # ------------------------------------------------------------------

    async def _read_stored_config(self, name: str) -> dict:
        """Read persisted config from disk (non-blocking)."""
        config_path = self.plugin_dir / name / _CONFIG_FILENAME
        if not config_path.exists():
            # Fallback: top-level json file
            config_path = self.plugin_dir / f"{name}_{_CONFIG_FILENAME}"
        if not config_path.exists():
            return {}
        try:
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, config_path.read_text, "utf-8")
            return json.loads(content)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read config for plugin %r: %s", name, exc)
            return {}

    async def _write_stored_config(self, name: str, config: dict) -> None:
        """Persist plugin config to disk (non-blocking)."""
        config_dir = self.plugin_dir / name
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / _CONFIG_FILENAME
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, config_path.write_text, json.dumps(config, indent=2), "utf-8"
            )
        except OSError as exc:
            logger.error("Could not write config for plugin %r: %s", name, exc)

    # ------------------------------------------------------------------
    # Internal: manifest helpers
    # ------------------------------------------------------------------

    def _manifest_path(self, name: str) -> Path:
        return self.plugin_dir / name / _MANIFEST_FILENAME

    def _save_manifest(self, name: str, data: dict) -> None:
        path = self._manifest_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_manifest(self, name: str) -> dict | None:
        path = self._manifest_path(name)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # ------------------------------------------------------------------
    # Internal: file watcher for hot-reload
    # ------------------------------------------------------------------

    def _setup_watcher(self) -> None:
        """
        Start a background asyncio task that watches the plugin directory
        for file changes and auto-reloads modified plugins.

        Uses a simple polling approach; falls back gracefully if ``watchdog``
        is not installed.
        """
        try:
            self._watcher = self._PollingWatcher(self)
            self._watcher_task = asyncio.create_task(self._watcher.run())
            logger.info("File watcher started (polling mode)")
        except Exception:
            logger.warning("Could not start file watcher", exc_info=True)

    async def _on_file_changed(self, plugin_name: str) -> None:
        """Callback invoked by the watcher when a plugin file changes."""
        if plugin_name in self._plugins:
            try:
                await self.reload_plugin(plugin_name)
                logger.info("Auto-reloaded plugin %r due to file change", plugin_name)
            except Exception:
                logger.exception("Auto-reload failed for plugin %r", plugin_name)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully shut down the plugin system and all plugins."""
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass
        for name in list(self._plugins):
            try:
                await self.unload_plugin(name)
            except Exception:
                logger.exception("Error shutting down plugin %r", name)
        self._initialised = False
        logger.info("Plugin system shut down")


# =========================================================================
# Polling file watcher (no third-party dependencies required)
# =========================================================================


class _PollingWatcher:
    """
    Lightweight polling file watcher that triggers plugin reloads when
    source files change.  Checks modification times every ``interval``
    seconds.
    """

    def __init__(
        self,
        manager: PluginManager,
        interval: float = 2.0,
    ) -> None:
        self.manager = manager
        self.interval = interval
        self._mtimes: dict[str, float] = {}
        self._running = True

    async def run(self) -> None:
        """Main watch loop."""
        # Prime the mtime cache
        for name in self.manager.discover_plugins():
            self._snapshot(name)

        while self._running:
            await asyncio.sleep(self.interval)
            for name in self.manager.discover_plugins():
                if self._has_changed(name):
                    asyncio.ensure_future(self.manager._on_file_changed(name))
                    self._snapshot(name)

    def _snapshot(self, name: str) -> None:
        """Record the current mtime for a plugin."""
        path = self.manager._find_plugin_path(name)
        if path is None:
            return
        if path.is_dir():
            mtimes = []
            for f in path.rglob("*.py"):
                mtimes.append(f.stat().st_mtime)
            self._mtimes[name] = max(mtimes) if mtimes else 0.0
        else:
            self._mtimes[name] = path.stat().st_mtime

    def _has_changed(self, name: str) -> bool:
        """Return True if the plugin's source files have changed."""
        path = self.manager._find_plugin_path(name)
        if path is None:
            return False
        old_mtime = self._mtimes.get(name, 0.0)
        if path.is_dir():
            for f in path.rglob("*.py"):
                if f.stat().st_mtime > old_mtime:
                    return True
            return False
        return path.stat().st_mtime > old_mtime
