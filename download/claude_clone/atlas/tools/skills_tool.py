"""
Atlas Skills Tool — skill management interface.

Features:
- List, create, install, uninstall skills
- Skill execution
- Skill validation
- Skill sharing/import
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from atlas.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Skill storage
# ---------------------------------------------------------------------------

_SKILLS_DIR = Path.home() / ".claude_clone" / "skills"
_INSTALLED_DIR = _SKILLS_DIR / "installed"


def _ensure_dirs() -> None:
    _SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    _INSTALLED_DIR.mkdir(parents=True, exist_ok=True)


def _skill_manifest_path(skill_name: str) -> Path:
    return _INSTALLED_DIR / skill_name / "manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def atlas_skill_list(category: str = "") -> str:
    """List all installed skills.

    param category (str): — Filter by category. Empty = all.
    """
    def _do():
        _ensure_dirs()
        if not _INSTALLED_DIR.exists():
            return "No skills installed yet."

        skills = []
        for skill_dir in sorted(_INSTALLED_DIR.iterdir()):
            if not skill_dir.is_dir():
                continue
            manifest_path = skill_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if category and manifest.get("category", "") != category:
                    continue
                skills.append(manifest)
            except (json.JSONDecodeError, IOError):
                continue

        if not skills:
            return f"No skills found" + (f" in category '{category}'" if category else "") + "."

        lines = [f"Installed skills ({len(skills)}):\n"]
        for s in skills:
            lines.append(
                f"  {s.get('name', '?')} v{s.get('version', '?')} "
                f"[{s.get('category', 'general')}] — {s.get('description', '')[:80]}"
            )
        return "\n".join(lines)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error listing skills: {e}"


async def atlas_skill_create(
    name: str,
    description: str = "",
    instructions: str = "",
    category: str = "general",
    version: str = "1.0.0",
) -> str:
    """Create a new skill with instructions.

    param name (str): — Unique skill name (snake_case).
    param description (str): — Brief description of what the skill does.
    param instructions (str): — Detailed instructions for the skill.
    param category (str): — Skill category. Default: general.
    param version (str): — Skill version. Default: 1.0.0.
    """
    def _do():
        _ensure_dirs()
        skill_dir = _INSTALLED_DIR / name
        if skill_dir.exists():
            return f"Error: Skill '{name}' already exists. Use atlas_skill_update to modify it."

        skill_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "name": name,
            "description": description,
            "instructions": instructions,
            "category": category,
            "version": version,
            "created_at": _now(),
            "updated_at": _now(),
        }

        (skill_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if instructions:
            (skill_dir / "instructions.md").write_text(instructions, encoding="utf-8")

        return f"Created skill '{name}' (v{version}) in category '{category}'"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error creating skill: {e}"


async def atlas_skill_execute(name: str, input_data: str = "") -> str:
    """Execute a skill by name with optional input data.

    param name (str): — Name of the skill to execute.
    param input_data (str): — Input data or context for the skill.
    """
    def _do():
        _ensure_dirs()
        manifest_path = _skill_manifest_path(name)
        if not manifest_path.exists():
            return f"Error: Skill '{name}' not found. Use atlas_skill_list to see available skills."

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return f"Error: Corrupted manifest for skill '{name}'"

        instructions = manifest.get("instructions", "")
        if not instructions:
            instr_path = _INSTALLED_DIR / name / "instructions.md"
            if instr_path.exists():
                instructions = instr_path.read_text(encoding="utf-8")

        if not instructions:
            return f"Error: Skill '{name}' has no instructions."

        # Return the skill instructions + input for the agent to follow
        parts = [
            f"Executing skill: {name} (v{manifest.get('version', '?')})",
            f"Description: {manifest.get('description', '')}",
            "",
            "=== SKILL INSTRUCTIONS ===",
            instructions,
        ]
        if input_data:
            parts.append("")
            parts.append("=== INPUT DATA ===")
            parts.append(input_data)

        return "\n".join(parts)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error executing skill: {e}"


async def atlas_skill_uninstall(name: str) -> str:
    """Uninstall a skill.

    param name (str): — Name of the skill to uninstall.
    """
    def _do():
        skill_dir = _INSTALLED_DIR / name
        if not skill_dir.exists():
            return f"Error: Skill '{name}' is not installed."

        shutil.rmtree(skill_dir)
        return f"Uninstalled skill '{name}'"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error uninstalling skill: {e}"


async def atlas_skill_update(name: str, instructions: str = "", description: str = "") -> str:
    """Update an existing skill's instructions or description.

    param name (str): — Name of the skill to update.
    param instructions (str): — New instructions (empty = keep current).
    param description (str): — New description (empty = keep current).
    """
    def _do():
        _ensure_dirs()
        manifest_path = _skill_manifest_path(name)
        if not manifest_path.exists():
            return f"Error: Skill '{name}' not found."

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if description:
            manifest["description"] = description
        if instructions:
            manifest["instructions"] = instructions
            (manifest_path.parent / "instructions.md").write_text(
                instructions, encoding="utf-8"
            )

        manifest["updated_at"] = _now()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        return f"Updated skill '{name}'"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error updating skill: {e}"


async def atlas_skill_export(name: str, output_path: str = "") -> str:
    """Export a skill as a JSON file for sharing.

    param name (str): — Skill to export.
    param output_path (str): — Output path. Default: auto-generated.
    """
    def _do():
        manifest_path = _skill_manifest_path(name)
        if not manifest_path.exists():
            return f"Error: Skill '{name}' not found."

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Also include instructions file if present
        instr_path = manifest_path.parent / "instructions.md"
        if instr_path.exists():
            manifest["instructions_file"] = instr_path.read_text(encoding="utf-8")

        if not output_path:
            output_path = str(Path.home() / f"skill_{name}_export.json")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        return f"Exported skill '{name}' to {out}"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error exporting skill: {e}"


async def atlas_skill_import(path: str) -> str:
    """Import a skill from a JSON file.

    param path (str): — Path to the skill JSON file.
    """
    def _do():
        _ensure_dirs()
        src = Path(path).expanduser().resolve()
        if not src.exists():
            return f"Error: File not found: {src}"

        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "Error: Invalid JSON file."

        name = data.get("name", "")
        if not name:
            return "Error: Skill JSON must have a 'name' field."

        skill_dir = _INSTALLED_DIR / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Write manifest
        data["imported_at"] = _now()
        data["updated_at"] = _now()
        (skill_dir / "manifest.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Write instructions file
        instructions = data.get("instructions", "") or data.get("instructions_file", "")
        if instructions:
            (skill_dir / "instructions.md").write_text(instructions, encoding="utf-8")

        return f"Imported skill '{name}' from {src}"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error importing skill: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="atlas_skill_list",
    func=atlas_skill_list,
    description="List all installed skills with filtering.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="atlas_skill_create",
    func=atlas_skill_create,
    description="Create a new skill with instructions and metadata.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="atlas_skill_execute",
    func=atlas_skill_execute,
    description="Execute a skill by name, returning its instructions for the agent to follow.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="atlas_skill_uninstall",
    func=atlas_skill_uninstall,
    description="Uninstall a skill by name.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="atlas_skill_update",
    func=atlas_skill_update,
    description="Update an existing skill's instructions or description.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="atlas_skill_export",
    func=atlas_skill_export,
    description="Export a skill as a JSON file for sharing.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="atlas_skill_import",
    func=atlas_skill_import,
    description="Import a skill from a JSON file.",
    toolset="skills",
)
