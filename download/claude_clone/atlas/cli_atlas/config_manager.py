"""
Configuration management for Atlas CLI.

Extends the base JSON config with YAML support, profile management,
validation, migration, environment-specific overrides, and config diff/merge.
"""

import json
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


DEFAULT_CONFIG_DIR = Path.home() / ".claude_clone"
DEFAULT_ATLAS_DIR = DEFAULT_CONFIG_DIR / "atlas"
PROFILES_DIR = DEFAULT_ATLAS_DIR / "profiles"
THEMES_DIR = DEFAULT_ATLAS_DIR / "themes"
SESSIONS_DIR = DEFAULT_ATLAS_DIR / "sessions"
CRON_DIR = DEFAULT_ATLAS_DIR / "cron"
SKILLS_DIR = DEFAULT_ATLAS_DIR / "skills"


class ConfigManager:
    """Manages all Atlas CLI configuration with YAML/JSON support."""

    ATLAS_DEFAULTS = {
        "version": "1.0.0",
        "active_profile": "default",
        "theme": "dark",
        "skin": "nord",
        "prompt_style": "atlas",
        "editor": os.environ.get("EDITOR", "vim"),
        "pager": os.environ.get("PAGER", "less"),
        "clipboard": "auto",
        "sound_enabled": False,
        "notification_enabled": False,
        "auto_save": True,
        "save_interval": 60,
        "max_history": 1000,
        "scrollback_lines": 5000,
        "markdown_render": True,
        "syntax_highlight": True,
        "emoji_enabled": True,
        "timestamps": True,
        "token_display": True,
        "cost_warnings": True,
        "cost_warning_threshold": 1.0,
        "streaming": True,
        "typing_indicator": True,
        "tool_animation": True,
        "confirm_before_tool": False,
        "gateway": {
            "enabled": False,
            "platforms": ["cli", "web", "desktop"],
            "host": "localhost",
            "port": 8765,
            "auto_start": False,
            "session_timeout": 3600,
        },
        "cron": {
            "enabled": False,
            "max_jobs": 50,
            "log_retention_days": 30,
        },
        "mcp": {
            "servers": [],
            "auto_connect": True,
            "timeout": 30,
        },
        "providers": {},
        "pinned_models": [],
        "tool_permissions": {},
        "skill_settings": {},
        "keybindings": {
            "cancel": "c-c",
            "exit": "c-d",
            "clear": "c-l",
            "save": "c-s",
            "help": "f1",
        },
    }

    CONFIG_SCHEMA_VERSION = 1

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or DEFAULT_ATLAS_DIR
        self.config_file = self.config_dir / "config.yaml"
        self.config_json_file = self.config_dir / "config.json"
        self.base_config_file = DEFAULT_CONFIG_DIR / "config.json"
        self._config: Dict[str, Any] = {}
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Ensure all required directories exist."""
        for d in [self.config_dir, PROFILES_DIR, THEMES_DIR, SESSIONS_DIR, CRON_DIR, SKILLS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def load(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration, merging defaults with saved config."""
        self._config = deepcopy(self.ATLAS_DEFAULTS)

        # Load from YAML first (preferred)
        if HAS_YAML and self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f) or {}
                self._merge_config(self._config, yaml_data)
            except Exception as e:
                print(f"Warning: Failed to load YAML config: {e}")

        # Fall back to JSON
        elif self.config_json_file.exists():
            try:
                with open(self.config_json_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                self._merge_config(self._config, json_data)
            except Exception as e:
                print(f"Warning: Failed to load JSON config: {e}")

        # Apply profile overrides
        if profile and profile != "default":
            profile_config = self._load_profile_config(profile)
            if profile_config:
                self._merge_config(self._config, profile_config)

        # Environment variable overrides
        self._apply_env_overrides()

        return self._config

    def save(self, config: Optional[Dict[str, Any]] = None) -> Path:
        """Save current configuration to YAML (preferred) or JSON."""
        data = config or self._config
        data["_saved_at"] = datetime.now().isoformat()
        data["_schema_version"] = self.CONFIG_SCHEMA_VERSION

        if HAS_YAML:
            path = self.config_file
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        else:
            path = self.config_json_file
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        os.chmod(path, 0o600)
        return path

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-separated key."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a config value by dot-separated key."""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def _merge_config(self, base: Dict, override: Dict) -> Dict:
        """Deep merge override into base config."""
        for key, value in override.items():
            if key.startswith("_"):
                continue
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
        return base

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_map = {
            "ATLAS_THEME": "theme",
            "ATLAS_SKIN": "skin",
            "ATLAS_EDITOR": "editor",
            "ATLAS_PAGER": "pager",
            "CLAUDE_MODEL": None,  # Handled by base config
            "CLAUDE_THEME": "theme",
        }
        for env_var, config_key in env_map.items():
            value = os.environ.get(env_var)
            if value and config_key:
                self.set(config_key, value)

    def _load_profile_config(self, profile_name: str) -> Optional[Dict]:
        """Load a profile-specific config overlay."""
        profile_path = PROFILES_DIR / f"{profile_name}.yaml"
        if HAS_YAML and profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        profile_json = PROFILES_DIR / f"{profile_name}.json"
        if profile_json.exists():
            try:
                with open(profile_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def validate(self) -> Tuple[List[str], List[str]]:
        """Validate configuration. Returns (errors, warnings)."""
        errors = []
        warnings = []

        # Check required fields
        if not self.get("theme"):
            errors.append("Theme is not set")
        if not self.get("active_profile"):
            warnings.append("No active profile set, using 'default'")

        # Check theme exists
        theme = self.get("theme")
        if theme and theme not in ("dark", "light"):
            warnings.append(f"Unknown theme: {theme}")

        # Check skin exists
        skin = self.get("skin")
        if skin:
            skin_file = THEMES_DIR / f"{skin}.yaml"
            if not skin_file.exists() and skin not in SkinEngine.BUILTIN_THEMES:
                warnings.append(f"Custom skin '{skin}' not found at {skin_file}")

        # Check gateway settings
        port = self.get("gateway.port")
        if port and not (1 <= int(port) <= 65535):
            errors.append(f"Gateway port {port} is out of range [1, 65535]")

        # Check cost warning threshold
        threshold = self.get("cost_warning_threshold")
        if threshold is not None and threshold <= 0:
            warnings.append(f"Cost warning threshold {threshold} should be positive")

        # Check session timeout
        timeout = self.get("gateway.session_timeout")
        if timeout is not None and timeout <= 0:
            errors.append(f"Session timeout {timeout} must be positive")

        return errors, warnings

    def migrate(self) -> bool:
        """Migrate configuration from older formats."""
        migrated = False

        # If only JSON exists and YAML lib available, migrate
        if HAS_YAML and self.config_json_file.exists() and not self.config_file.exists():
            try:
                with open(self.config_json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.save(data)
                migrated = True
            except Exception:
                pass

        # Migrate base config fields
        if self.base_config_file.exists():
            try:
                with open(self.base_config_file, "r", encoding="utf-8") as f:
                    base_data = json.load(f)
                # Sync relevant fields
                for field in ("model", "max_tokens", "temperature", "provider", "base_url"):
                    if field in base_data and not self.get(field):
                        self.set(field, base_data[field])
                        migrated = True
            except Exception:
                pass

        return migrated

    def diff(self, other: Dict) -> Dict[str, Tuple[Any, Any]]:
        """Compare current config with another config dict. Returns changes."""
        changes = {}

        def _compare(path: str, a: Any, b: Any):
            if type(a) != type(b):
                changes[path] = (a, b)
            elif isinstance(a, dict):
                for k in set(list(a.keys()) + list(b.keys())):
                    _compare(f"{path}.{k}", a.get(k), b.get(k))
            elif a != b:
                changes[path] = (a, b)

        _compare("", self._config, other)
        return changes

    def to_dict(self) -> Dict[str, Any]:
        """Return full config as dictionary."""
        return deepcopy(self._config)

    def export(self, path: Optional[str] = None, include_secrets: bool = False) -> Path:
        """Export configuration to a file."""
        data = deepcopy(self._config)

        if not include_secrets:
            # Remove sensitive fields
            for provider in data.get("providers", {}).values():
                if isinstance(provider, dict):
                    provider.pop("api_key", None)
                    provider.pop("key", None)

        export_path = Path(path) if path else self.config_dir / "export.yaml"
        if HAS_YAML:
            with open(export_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        return export_path

    def import_config(self, path: str) -> bool:
        """Import configuration from a YAML or JSON file."""
        import_path = Path(path)
        if not import_path.exists():
            return False

        try:
            if import_path.suffix in (".yaml", ".yml") and HAS_YAML:
                with open(import_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            else:
                with open(import_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            self._merge_config(self._config, data)
            self.save()
            return True
        except Exception:
            return False

    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._config = deepcopy(self.ATLAS_DEFAULTS)
        self.save()

    def list_sections(self) -> List[str]:
        """List all top-level config sections."""
        return [k for k in self._config.keys() if not k.startswith("_")]


# Late import to avoid circular dependency
from atlas.cli_atlas.skin_engine import SkinEngine
