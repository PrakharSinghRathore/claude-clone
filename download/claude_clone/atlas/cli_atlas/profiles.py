"""
Multi-profile support for Atlas CLI.

Create, switch, delete, and manage configuration profiles.
Each profile can have its own settings that override the base config.
"""

import json
import os
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from atlas.cli_atlas.config_manager import PROFILES_DIR


class ProfileManager:
    """Manages configuration profiles for different environments."""

    RESERVED_PROFILES = {"default", "dev", "prod", "test"}

    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or PROFILES_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all available profiles."""
        profiles = []
        for f in sorted(self.profiles_dir.glob("*.yaml")) + sorted(self.profiles_dir.glob("*.json")):
            name = f.stem
            config = self._load_profile(name)
            profiles.append({
                "name": name,
                "path": str(f),
                "description": config.get("description", ""),
                "inherits": config.get("inherits"),
                "created": config.get("_created", "unknown"),
                "modified": config.get("_modified", "unknown"),
                "is_builtin": name in self.RESERVED_PROFILES,
            })
        return profiles

    def create(
        self,
        name: str,
        description: str = "",
        inherits: Optional[str] = None,
        settings: Optional[Dict] = None,
    ) -> Path:
        """Create a new profile."""
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid profile name: {name}")

        profile_path = self.profiles_dir / f"{name}.yaml"
        if profile_path.exists():
            raise FileExistsError(f"Profile '{name}' already exists")

        config = {
            "description": description,
            "inherits": inherits or "default",
            "_created": datetime.now().isoformat(),
            "_modified": datetime.now().isoformat(),
        }

        if settings:
            config.update(settings)

        if HAS_YAML:
            with open(profile_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            json_path = self.profiles_dir / f"{name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

        return profile_path

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a profile's configuration."""
        return self._load_profile(name)

    def update(self, name: str, settings: Dict[str, Any]) -> bool:
        """Update a profile's settings."""
        config = self._load_profile(name)
        if config is None:
            return False

        for key, value in settings.items():
            if not key.startswith("_"):
                config[key] = value
        config["_modified"] = datetime.now().isoformat()

        self._save_profile(name, config)
        return True

    def delete(self, name: str, force: bool = False) -> bool:
        """Delete a profile."""
        if name in self.RESERVED_PROFILES and not force:
            return False

        for ext in ("yaml", "json"):
            path = self.profiles_dir / f"{name}.{ext}"
            if path.exists():
                path.unlink()
                return True
        return False

    def rename(self, old_name: str, new_name: str) -> bool:
        """Rename a profile."""
        for ext in ("yaml", "json"):
            old_path = self.profiles_dir / f"{old_name}.{ext}"
            new_path = self.profiles_dir / f"{new_name}.{ext}"
            if old_path.exists():
                shutil.move(str(old_path), str(new_path))
                return True
        return False

    def copy(self, source: str, dest: str) -> bool:
        """Copy a profile."""
        source_path = self._get_profile_path(source)
        if not source_path:
            return False

        if HAS_YAML:
            dest_path = self.profiles_dir / f"{dest}.yaml"
        else:
            dest_path = self.profiles_dir / f"{dest}.json"

        shutil.copy2(str(source_path), str(dest_path))
        return True

    def export_profile(self, name: str, path: str) -> bool:
        """Export a profile to a file."""
        config = self._load_profile(name)
        if config is None:
            return False

        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(export_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, default=str)

        return True

    def import_profile(self, path: str, name: Optional[str] = None) -> bool:
        """Import a profile from a file."""
        import_path = Path(path)
        if not import_path.exists():
            return False

        profile_name = name or import_path.stem
        dest = self.profiles_dir / f"{profile_name}.yaml"

        if HAS_YAML:
            try:
                with open(import_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                with open(dest, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                return True
            except Exception:
                pass

        json_dest = self.profiles_dir / f"{profile_name}.json"
        shutil.copy2(str(import_path), str(json_dest))
        return True

    def compare(self, profile1: str, profile2: str) -> Dict[str, Tuple[Any, Any]]:
        """Compare two profiles and return differences."""
        p1 = self._load_profile(profile1) or {}
        p2 = self._load_profile(profile2) or {}

        all_keys = set(list(p1.keys()) + list(p2.keys()))
        differences = {}

        for key in sorted(all_keys):
            if key.startswith("_"):
                continue
            v1 = p1.get(key)
            v2 = p2.get(key)
            if v1 != v2:
                differences[key] = (v1, v2)

        return differences

    def get_inheritance_chain(self, name: str) -> List[str]:
        """Get the full inheritance chain for a profile."""
        chain = []
        visited = set()
        current = name

        while current and current not in visited:
            if current in visited:
                break
            visited.add(current)
            chain.append(current)
            config = self._load_profile(current)
            current = config.get("inherits") if config else None

        return chain

    def resolve_effective_config(self, name: str) -> Dict[str, Any]:
        """Resolve the effective config by walking the inheritance chain."""
        chain = self.get_inheritance_chain(name)
        merged = {}

        for profile_name in reversed(chain):
            config = self._load_profile(profile_name)
            if config:
                self._deep_merge(merged, {k: v for k, v in config.items() if not k.startswith("_")})

        return merged

    def _load_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """Load profile config from YAML or JSON file."""
        for ext in ("yaml", "json"):
            path = self.profiles_dir / f"{name}.{ext}"
            if path.exists():
                try:
                    if ext == "yaml" and HAS_YAML:
                        with open(path, "r", encoding="utf-8") as f:
                            return yaml.safe_load(f) or {}
                    else:
                        with open(path, "r", encoding="utf-8") as f:
                            return json.load(f)
                except Exception:
                    continue
        return None

    def _save_profile(self, name: str, config: Dict[str, Any]) -> Path:
        """Save profile config to file."""
        if HAS_YAML:
            path = self.profiles_dir / f"{name}.yaml"
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            path = self.profiles_dir / f"{name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, default=str)
        return path

    def _get_profile_path(self, name: str) -> Optional[Path]:
        """Get the file path for a profile."""
        for ext in ("yaml", "json"):
            path = self.profiles_dir / f"{name}.{ext}"
            if path.exists():
                return path
        return None

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = deepcopy(value)
        return base
