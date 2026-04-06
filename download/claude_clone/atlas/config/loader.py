"""
Atlas Configuration Loader — Multi-source configuration loading.

Loads configuration from multiple sources with priority ordering:
CLI arguments > environment variables > config file > defaults.

Supports YAML, JSON, and TOML formats, config file watching,
secret resolution, and version migration.

Usage::

    from atlas.config.loader import ConfigLoader

    loader = ConfigLoader()
    config = loader.load("config.yaml")
    print(config.agent.model)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .schema import AppConfig, get_defaults, merge, validate
from .types import LogLevel, LogFormat, resolve_env_var

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_NAME = "atlas_config"
_DEFAULT_SEARCH_PATHS = [
    "atlas_config.yaml",
    "atlas_config.yml",
    "atlas_config.json",
    "atlas_config.toml",
    ".atlas.yaml",
    ".atlas.yml",
    ".atlas.json",
]

# Current config schema version for migration
_CONFIG_VERSION = 1

# Environment variable prefix
_ENV_PREFIX = "ATLAS_"


# ──────────────────────────────────────────────────────────────────────────────
# Secret Resolution
# ──────────────────────────────────────────────────────────────────────────────

class SecretResolver:
    """
    Resolves secret values from various sources.

    Supports environment variables, file references (``file:///path``),
    and direct values. Secrets in configuration can reference environment
    variables using ``${VAR}`` or ``${VAR:default}`` syntax.
    """

    def __init__(self) -> None:
        self._resolved_cache: Dict[str, str] = {}
        self._secret_keys: set[str] = set()
        self._file_pattern = re.compile(r"^file://(.+)$", re.IGNORECASE)

    def resolve(self, value: Any, key_path: str = "") -> Any:
        """
        Resolve a configuration value, handling secret references.

        Parameters
        ----------
        value:
            The configuration value to resolve.
        key_path:
            Dot-separated path for logging (e.g., ``"providers.anthropic.api_key"``).

        Returns
        -------
        Any
            The resolved value.
        """
        if not isinstance(value, str):
            return value

        # Check for file reference
        file_match = self._file_pattern.match(value.strip())
        if file_match:
            file_path = file_match.group(1)
            return self._resolve_file(file_path, key_path)

        # Check for environment variable reference
        if "${" in value:
            return resolve_env_var(value)

        return value

    def resolve_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolve all secret values in a configuration dictionary.

        Parameters
        ----------
        data:
            The configuration dictionary.

        Returns
        -------
        dict
            A new dictionary with resolved values.
        """
        result: Dict[str, Any] = {}
        for key, value in data.items():
            key_path = f"{key_path}.{key}" if key_path else key
            if isinstance(value, dict):
                result[key] = self.resolve_dict(value)
            elif isinstance(value, list):
                result[key] = [self.resolve(item, key_path) for item in value]
            else:
                resolved = self.resolve(value, key_path)
                result[key] = resolved
                if resolved != value:
                    self._secret_keys.add(key_path)
        return result

    def _resolve_file(self, file_path: str, key_path: str) -> str:
        """Read a secret from a file."""
        cache_key = f"file://{file_path}"
        if cache_key in self._resolved_cache:
            return self._resolved_cache[cache_key]

        try:
            path = Path(file_path).expanduser().resolve()
            if not path.exists():
                logger.warning("Secret file not found for %s: %s", key_path, path)
                return ""
            content = path.read_text(encoding="utf-8").strip()
            self._resolved_cache[cache_key] = content
            logger.debug("Resolved secret from file for %s: %s", key_path, path)
            return content
        except Exception:
            logger.exception("Failed to read secret file for %s: %s", key_path, file_path)
            return ""

    def get_resolved_keys(self) -> set[str]:
        """Return the set of key paths that were resolved as secrets."""
        return set(self._secret_keys)


# ──────────────────────────────────────────────────────────────────────────────
# Config File Watcher
# ──────────────────────────────────────────────────────────────────────────────

class ConfigWatcher:
    """
    Watches a configuration file for changes and invokes a callback.

    Uses file modification time polling with configurable interval.
    Supports debouncing to avoid rapid successive reloads.

    Parameters
    ----------
    path:
        Path to the configuration file to watch.
    callback:
        Async callback invoked when the file changes.
    interval:
        Polling interval in seconds (default 5).
    debounce:
        Debounce time in seconds (default 2).
    """

    def __init__(
        self,
        path: str | Path,
        callback: Callable[[Dict[str, Any]], Any],
        interval: float = 5.0,
        debounce: float = 2.0,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self._callback = callback
        self._interval = interval
        self._debounce = debounce
        self._last_mtime: float = 0.0
        self._last_load: float = 0.0
        self._task = None
        self._running = False

    @property
    def last_mtime(self) -> float:
        """Last observed modification time of the config file."""
        return self._last_mtime

    async def start(self) -> None:
        """Start watching the configuration file."""
        if self._running:
            return
        if not self.path.exists():
            logger.warning("Config file to watch does not exist: %s", self.path)
            return
        self._last_mtime = self.path.stat().st_mtime
        self._running = True
        import asyncio
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("Watching config file: %s (interval=%.1fs)", self.path, self._interval)

    async def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, ImportError):
                pass
        logger.info("Stopped watching config file: %s", self.path)

    async def _watch_loop(self) -> None:
        """Polling loop that checks for file changes."""
        import asyncio
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                if not self._running:
                    break
                if not self.path.exists():
                    continue

                current_mtime = self.path.stat().st_mtime
                if current_mtime > self._last_mtime:
                    now = time.time()
                    if now - self._last_load < self._debounce:
                        continue

                    logger.info("Config file changed: %s", self.path)
                    self._last_mtime = current_mtime
                    self._last_load = now

                    try:
                        new_config = _parse_file(self.path)
                        result = self._callback(new_config)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        logger.exception("Error processing config change")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in config watch loop")


# ──────────────────────────────────────────────────────────────────────────────
# Config Migration
# ──────────────────────────────────────────────────────────────────────────────

class ConfigMigrator:
    """
    Migrates configuration between schema versions.

    Handles breaking changes between versions by applying transformation
    functions in order. Each migration function receives the config dict
    and returns the updated dict.
    """

    # Registry of version -> migration function
    _migrations: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    @classmethod
    def register(cls, from_version: int, to_version: int):
        """Decorator to register a migration function."""
        def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]):
            cls._migrations[from_version] = (func, to_version)
            return func
        return decorator

    @classmethod
    def migrate(cls, config: Dict[str, Any], target_version: int = _CONFIG_VERSION) -> Dict[str, Any]:
        """
        Migrate a configuration dict to the target version.

        Parameters
        ----------
        config:
            The configuration dictionary.
        target_version:
            The target schema version.

        Returns
        -------
        dict
            The migrated configuration.
        """
        current_version = config.get("_version", 0)

        if current_version == target_version:
            return config

        logger.info(
            "Migrating config from version %d to %d",
            current_version,
            target_version,
        )

        while current_version < target_version:
            if current_version in cls._migrations:
                func, next_version = cls._migrations[current_version]
                config = func(config)
                current_version = next_version
                logger.info("Migrated to version %d", current_version)
            else:
                # No migration available; bump version
                current_version += 1

        config["_version"] = target_version
        return config


# Register built-in migrations
@ConfigMigrator.register(0, 1)
def _migrate_v0_to_v1(config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate from version 0 (unversioned) to version 1."""
    # Ensure all top-level sections exist
    defaults = get_defaults()
    for key in defaults:
        if key not in config:
            config[key] = copy.deepcopy(defaults[key])

    # Move old-style flat keys to nested structure
    flat_mappings = {
        "model": ("agent", "model"),
        "temperature": ("agent", "temperature"),
        "max_tokens": ("agent", "max_tokens"),
        "system_prompt": ("agent", "system_prompt"),
        "log_level": None,  # Keep at root
        "log_format": None,
    }

    for old_key, new_location in flat_mappings.items():
        if old_key in config and new_location is not None:
            section, field_name = new_location
            if section not in config:
                config[section] = {}
            if field_name not in config[section]:
                config[section][field_name] = config.pop(old_key)
            else:
                config.pop(old_key, None)

    return config


# ──────────────────────────────────────────────────────────────────────────────
# File Parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_file(path: Path) -> Dict[str, Any]:
    """Parse a configuration file (YAML, JSON, or TOML)."""
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(raw) or {}
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install pyyaml"
            )
    elif suffix == ".json":
        return json.loads(raw)
    elif suffix == ".toml":
        try:
            import tomllib
            return tomllib.loads(raw)
        except ImportError:
            try:
                import tomli as tomllib
                return tomllib.loads(raw)
            except ImportError:
                raise ImportError(
                    "Python 3.11+ or tomli is required to load TOML config files. "
                    "Install it with: pip install tomli"
                )
    else:
        # Try JSON first, then YAML
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                import yaml
                return yaml.safe_load(raw) or {}
            except ImportError:
                raise ValueError(
                    f"Cannot determine config format for {path}. "
                    f"Use .yaml, .yml, .json, or .toml extension."
                )


def _serialize(
    data: Dict[str, Any],
    format: str = "yaml",
    path: Optional[Path] = None,
) -> str:
    """Serialize a configuration dictionary to a string."""
    if format in ("yaml", "yml"):
        try:
            import yaml
            return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except ImportError:
            raise ImportError("PyYAML is required for YAML output.")
    elif format == "json":
        return json.dumps(data, indent=2, default=str, ensure_ascii=False)
    elif format == "toml":
        # Basic TOML serialization for simple types
        lines = ['# Atlas Configuration', f'_version = {_CONFIG_VERSION}', '']
        for key, value in data.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict):
                lines.append(f"[{key}]")
                for k, v in value.items():
                    lines.append(f"{k} = {_toml_value(v)}")
                lines.append("")
            else:
                lines.append(f"{key} = {_toml_value(value)}")
        return "\n".join(lines)
    else:
        raise ValueError(f"Unsupported format: {format!r}")


def _toml_value(value: Any) -> str:
    """Convert a Python value to a TOML-compatible string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        items = ", ".join(_toml_value(v) for v in value)
        return f"[{items}]"
    if isinstance(value, dict):
        # Inline table
        items = ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items())
        return f"{{{items}}}"
    return f'"{value}"'


# ──────────────────────────────────────────────────────────────────────────────
# Config Loader
# ──────────────────────────────────────────────────────────────────────────────

class ConfigLoader:
    """
    Multi-source configuration loader with priority resolution.

    Loading priority (highest to lowest):
    1. CLI arguments (explicitly applied)
    2. Environment variables (``ATLAS_*``)
    3. Configuration file (YAML/JSON/TOML)
    4. Built-in defaults

    Parameters
    ----------
    search_paths:
        Additional paths to search for config files.
    env_prefix:
        Environment variable prefix (default ``ATLAS_``).

    Example
    -------
    >>> loader = ConfigLoader()
    >>> config = loader.load("atlas_config.yaml")
    >>> config.agent.model  # doctest: +SKIP
    'claude-sonnet-4-20250514'
    """

    def __init__(
        self,
        search_paths: Optional[List[str]] = None,
        env_prefix: str = _ENV_PREFIX,
    ) -> None:
        self._search_paths = search_paths or []
        self._env_prefix = env_prefix
        self._secret_resolver = SecretResolver()
        self._watchers: List[ConfigWatcher] = []
        self._last_loaded_path: Optional[Path] = None

    def load(
        self,
        config_path: Optional[str] = None,
    ) -> AppConfig:
        """
        Load configuration from a file, falling back to defaults.

        The loading process:
        1. Start with built-in defaults.
        2. Search for and load a config file.
        3. Apply environment variable overrides.
        4. Resolve secrets and ``${VAR}`` references.
        5. Run validation.

        Parameters
        ----------
        config_path:
            Explicit path to a config file. If ``None``, searches
            default locations.

        Returns
        -------
        AppConfig
            The fully loaded and validated configuration.
        """
        # Step 1: Defaults
        config_dict = get_defaults()
        config_dict["_version"] = _CONFIG_VERSION

        # Step 2: Load from file
        file_config = self._find_and_load(config_path)
        if file_config:
            config_dict = merge(config_dict, file_config)
            logger.info("Configuration loaded from file")

        # Step 3: Environment variable overrides
        env_config = self.load_from_env()
        if env_config:
            config_dict = merge(config_dict, env_config)
            logger.debug("Applied environment variable overrides")

        # Step 4: Migration
        config_dict = ConfigMigrator.migrate(config_dict)

        # Step 5: Secret resolution
        config_dict = self._secret_resolver.resolve_dict(config_dict)

        # Step 6: Create AppConfig
        app_config = AppConfig.from_dict(config_dict)

        # Step 7: Apply env overrides on the typed config
        app_config.apply_env_overrides()

        # Step 8: Validate
        errors = app_config.validate()
        if errors:
            for error in errors:
                logger.warning("Configuration validation: %s", error)

        return app_config

    def load_from_env(self) -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.

        Environment variable naming convention:
        ``ATLAS_<SECTION>_<KEY>`` (e.g., ``ATLAS_AGENT_MODEL=claude-3-5-sonnet``)

        Nested sections use double underscore:
        ``ATLAS_PROVIDERS_ANTHROPIC_API_KEY=sk-...``

        Returns
        -------
        dict
            Configuration overrides from environment variables.
        """
        overrides: Dict[str, Any] = {}

        for key, value in os.environ.items():
            if not key.startswith(self._env_prefix):
                continue

            # Strip prefix
            config_key = key[len(self._env_prefix):].lower()

            # Handle nested keys with double underscore
            parts = config_key.split("__")
            current = overrides
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            field_name = parts[-1]
            # Attempt type conversion
            current[field_name] = self._parse_env_value(value)

        return overrides

    def load_from_cli(self, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load configuration overrides from command-line arguments.

        Supports ``--set KEY=VALUE`` syntax for arbitrary config overrides.
        Common flags are mapped to their config paths.

        Parameters
        ----------
        args:
            Command-line arguments. If ``None``, reads ``sys.argv``.

        Returns
        -------
        dict
            Configuration overrides from CLI arguments.
        """
        if args is None:
            import sys
            args = sys.argv[1:]

        overrides: Dict[str, Any] = {}

        # Map common flags to config paths
        flag_map: Dict[str, Tuple[str, Any]] = {
            "--model": ("agent.model", None),
            "--provider": ("agent.provider", None),
            "--temperature": ("agent.temperature", float),
            "--max-tokens": ("agent.max_tokens", int),
            "--verbose": ("log_level", "DEBUG"),
            "--quiet": ("log_level", "WARNING"),
        }

        i = 0
        while i < len(args):
            arg = args[i]

            # Check for --set KEY=VALUE
            if arg == "--set" and i + 1 < len(args):
                i += 1
                kv = args[i]
                if "=" in kv:
                    key, value = kv.split("=", 1)
                    self._set_nested(overrides, key, self._parse_env_value(value))

            # Check for mapped flags
            elif arg in flag_map:
                config_path, value_type = flag_map[arg]
                if value_type is None and i + 1 < len(args):
                    i += 1
                    value = args[i]
                elif value_type is not None:
                    if value_type == bool:
                        value = True
                    elif value_type == int:
                        value = int(args[i + 1]) if i + 1 < len(args) else "0"
                        i += 1
                    elif value_type == float:
                        value = float(args[i + 1]) if i + 1 < len(args) else "0.0"
                        i += 1
                    else:
                        value = value_type
                else:
                    value = True
                self._set_nested(overrides, config_path, value)

            i += 1

        return overrides

    def save(
        self,
        config: Union[AppConfig, Dict[str, Any]],
        path: str | Path,
        format: str = "yaml",
        mask_secrets: bool = False,
    ) -> Path:
        """
        Save configuration to a file.

        Parameters
        ----------
        config:
            The configuration to save.
        path:
            Destination file path.
        format:
            Output format: ``"yaml"``, ``"json"``, or ``"toml"``.
        mask_secrets:
            If ``True``, mask API keys and secrets before saving.

        Returns
        -------
        Path
            The path to the saved file.
        """
        save_path = Path(path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(config, AppConfig):
            data = config.to_dict(mask_secrets=mask_secrets)
        else:
            data = copy.deepcopy(config)

        data["_version"] = _CONFIG_VERSION

        # Infer format from extension if not specified
        if format == "auto":
            ext = save_path.suffix.lstrip(".")
            if ext in ("yml", "yaml"):
                format = "yaml"
            elif ext == "json":
                format = "json"
            elif ext == "toml":
                format = "toml"
            else:
                format = "yaml"

        content = _serialize(data, format)
        save_path.write_text(content, encoding="utf-8")

        # Restrict permissions for config files with secrets
        if not mask_secrets:
            os.chmod(str(save_path), 0o600)

        logger.info("Configuration saved to %s (format=%s)", save_path, format)
        return save_path

    def watch(
        self,
        path: str | Path,
        callback: Callable[[Dict[str, Any]], Any],
        interval: float = 5.0,
    ) -> ConfigWatcher:
        """
        Watch a configuration file for changes.

        Parameters
        ----------
        path:
            Path to the configuration file.
        callback:
            Function invoked with the new config when changes are detected.
        interval:
            Polling interval in seconds.

        Returns
        -------
        ConfigWatcher
            The watcher instance (call ``stop()`` to stop watching).
        """
        watcher = ConfigWatcher(path, callback, interval=interval)
        self._watchers.append(watcher)
        return watcher

    async def stop_all_watchers(self) -> None:
        """Stop all active configuration watchers."""
        for watcher in self._watchers:
            await watcher.stop()
        self._watchers.clear()

    # ── Internal ───────────────────────────────────────────────────────

    def _find_and_load(self, config_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Find and load a config file."""
        # Explicit path
        if config_path:
            path = Path(config_path).expanduser().resolve()
            if path.exists():
                self._last_loaded_path = path
                return _parse_file(path)
            logger.warning("Config file not found: %s", path)
            return None

        # Search default paths
        search_order = self._search_paths + _DEFAULT_SEARCH_PATHS
        # Also search in XDG config directory
        xdg_config = os.environ.get("XDG_CONFIG_HOME", "")
        if xdg_config:
            search_order.append(str(Path(xdg_config) / "atlas" / "atlas_config.yaml"))
        else:
            search_order.append(str(Path.home() / ".config" / "atlas" / "atlas_config.yaml"))

        for candidate in search_order:
            path = Path(candidate).expanduser().resolve()
            if path.exists():
                self._last_loaded_path = path
                logger.info("Found config file: %s", path)
                return _parse_file(path)

        logger.debug("No config file found; using defaults")
        return None

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse an environment variable value with type inference."""
        # Boolean
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        if value.lower() in ("false", "0", "no", "off"):
            return False

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # JSON (list or dict)
        if value.startswith(("{", "[")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        return value

    @staticmethod
    def _set_nested(data: Dict[str, Any], path: str, value: Any) -> None:
        """Set a value at a nested path in a dictionary."""
        parts = path.split(".")
        current = data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    @property
    def last_loaded_path(self) -> Optional[Path]:
        """Return the path of the last loaded config file."""
        return self._last_loaded_path
