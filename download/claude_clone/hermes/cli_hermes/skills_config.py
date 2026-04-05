"""
Skills configuration for Hermes CLI.

List installed skills, manage skill settings,
import/export skills, and check for updates.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes.cli_hermes.config_manager import SKILLS_DIR, ConfigManager


# ──────────────────────────────────────────────
# Skill definitions
# ──────────────────────────────────────────────

SKILL_CATEGORIES = {
    "development": {
        "name": "Development",
        "emoji": "\U0001f4bb",
        "description": "Code development and programming skills",
    },
    "analysis": {
        "name": "Analysis",
        "emoji": "\U0001f50d",
        "description": "Code analysis, debugging, and optimization",
    },
    "security": {
        "name": "Security",
        "emoji": "\U0001f6e1\ufe0f",
        "description": "Security scanning and vulnerability detection",
    },
    "devops": {
        "name": "DevOps",
        "emoji": "\u2699\ufe0f",
        "description": "Deployment, CI/CD, and infrastructure",
    },
    "data": {
        "name": "Data",
        "emoji": "\U0001f4ca",
        "description": "Data processing, transformation, and analysis",
    },
    "communication": {
        "name": "Communication",
        "emoji": "\U0001f4e3",
        "description": "Communication and collaboration",
    },
    "productivity": {
        "name": "Productivity",
        "emoji": "\u26a1",
        "description": "Workflow automation and productivity tools",
    },
    "creative": {
        "name": "Creative",
        "emoji": "\U0001f3a8",
        "description": "Creative and content generation skills",
    },
}


class SkillConfigManager:
    """Manages skill configuration, installation, and lifecycle."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.skills_dir = SKILLS_DIR
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def list_installed_skills(self) -> List[Dict[str, Any]]:
        """List all installed skills."""
        skills = []
        settings = self.config.get("skill_settings", {})

        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue

            manifest = skill_dir / "manifest.json"
            if manifest.exists():
                try:
                    with open(manifest, "r", encoding="utf-8") as f:
                        skill_data = json.load(f)
                except Exception:
                    continue
            else:
                skill_data = {
                    "name": skill_dir.name,
                    "id": skill_dir.name,
                    "version": "0.0.0",
                    "description": f"Skill: {skill_dir.name}",
                }

            skill_id = skill_data.get("id", skill_dir.name)
            skill_settings = settings.get(skill_id, {})

            skills.append({
                **skill_data,
                "id": skill_id,
                "path": str(skill_dir),
                "enabled": skill_settings.get("enabled", True),
                "settings": skill_settings,
                "installed": True,
            })

        return skills

    def get_skill_info(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a skill."""
        skill_dir = self.skills_dir / skill_id
        if not skill_dir.exists():
            return None

        manifest = skill_dir / "manifest.json"
        if manifest.exists():
            try:
                with open(manifest, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            "id": skill_id,
            "name": skill_id,
            "path": str(skill_dir),
        }

    def enable_skill(self, skill_id: str) -> bool:
        """Enable a skill."""
        settings = dict(self.config.get("skill_settings", {}))
        if skill_id not in settings:
            settings[skill_id] = {}
        settings[skill_id]["enabled"] = True
        self.config.set("skill_settings", settings)
        self.config.save()
        return True

    def disable_skill(self, skill_id: str) -> bool:
        """Disable a skill."""
        settings = dict(self.config.get("skill_settings", {}))
        if skill_id not in settings:
            settings[skill_id] = {}
        settings[skill_id]["enabled"] = False
        self.config.set("skill_settings", settings)
        self.config.save()
        return True

    def update_skill_setting(self, skill_id: str, key: str, value: Any) -> bool:
        """Update a specific skill setting."""
        settings = dict(self.config.get("skill_settings", {}))
        if skill_id not in settings:
            settings[skill_id] = {}
        settings[skill_id][key] = value
        self.config.set("skill_settings", settings)
        self.config.save()
        return True

    def export_skill(self, skill_id: str, output_path: str) -> bool:
        """Export a skill to an archive file."""
        skill_dir = self.skills_dir / skill_id
        if not skill_dir.exists():
            return False

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if skill_dir.is_dir():
            export_name = output_path if output.suffix else f"{output_path}.zip"
            try:
                shutil.make_archive(
                    output_path.replace(".zip", ""),
                    "zip",
                    skill_dir.parent,
                    skill_dir.name,
                )
                return True
            except Exception:
                return False
        return False

    def import_skill(self, archive_path: str) -> bool:
        """Import a skill from an archive file."""
        archive = Path(archive_path)
        if not archive.exists():
            return False

        try:
            shutil.unpack_archive(str(archive), str(self.skills_dir))
            return True
        except Exception:
            return False

    def check_updates(self) -> List[Dict[str, Any]]:
        """Check for skill updates (stub - would connect to hub)."""
        skills = self.list_installed_skills()
        updates = []

        for skill in skills:
            # In a real implementation, this would check a remote registry
            # For now, just return skills that have update metadata
            if skill.get("update_url"):
                updates.append({
                    "skill_id": skill["id"],
                    "current_version": skill.get("version", "0.0.0"),
                    "latest_version": skill.get("latest_version", skill.get("version", "0.0.0")),
                    "has_update": skill.get("version", "0.0.0") != skill.get("latest_version", "0.0.0"),
                })

        return updates

    def get_skill_files(self, skill_id: str) -> List[str]:
        """List files in a skill directory."""
        skill_dir = self.skills_dir / skill_id
        if not skill_dir.exists():
            return []

        files = []
        for f in sorted(skill_dir.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                rel = f.relative_to(skill_dir)
                files.append(str(rel))

        return files

    def format_skills_table(self, skills: Optional[List[Dict]] = None) -> str:
        """Format skills as a readable table."""
        skills = skills or self.list_installed_skills()

        if not skills:
            return "  No skills installed. Use /hub to browse and install skills."

        lines = []
        lines.append(f"  {'Skill':<25} {'Version':<10} {'Category':<15} {'Status'}")
        lines.append("  " + "-" * 65)

        for skill in skills:
            name = skill.get("name", skill["id"])
            version = skill.get("version", "?")
            category = skill.get("category", "other")
            status = "\033[32menabled\033[0m" if skill.get("enabled", True) else "\033[31mdisabled\033[0m"
            lines.append(f"  {name:<25} {version:<10} {category:<15} {status}")

        return "\n".join(lines)
