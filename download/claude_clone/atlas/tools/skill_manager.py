"""
Atlas Skill Manager — advanced skill lifecycle management.

Features:
- Create skills from conversation patterns
- Skill versioning
- Skill dependencies
- Skill testing framework
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from atlas.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SKILLS_DIR = Path.home() / ".claude_clone" / "skills"
_INSTALLED_DIR = _SKILLS_DIR / "installed"
_VERSIONS_DIR = _SKILLS_DIR / "versions"
_TESTS_DIR = _SKILLS_DIR / "tests"


def _ensure_dirs() -> None:
    for d in (_SKILLS_DIR, _INSTALLED_DIR, _VERSIONS_DIR, _TESTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run_sync(func, *args, **kwargs):
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def atlas_skill_from_conversation(
    name: str,
    pattern_description: str,
    instructions: str = "",
) -> str:
    """Create a skill from a recurring conversation pattern.

    param name (str): — Skill name.
    param pattern_description (str): — Description of the pattern this skill captures.
    param instructions (str): — Detailed instructions derived from the pattern.
    """
    def _do():
        _ensure_dirs()
        skill_dir = _INSTALLED_DIR / name

        if not instructions:
            instructions = (
                f"## Pattern: {pattern_description}\n\n"
                f"When the user's request matches this pattern, follow these steps:\n"
                f"1. Analyze the request for {pattern_description}\n"
                f"2. Apply the standard approach for this pattern\n"
                f"3. Validate the result\n"
                f"4. Report the outcome\n"
            )

        manifest = {
            "name": name,
            "description": f"Auto-created from pattern: {pattern_description}",
            "instructions": instructions,
            "category": "auto-detected",
            "version": "1.0.0",
            "source": "conversation_pattern",
            "pattern": pattern_description,
            "created_at": _now(),
            "updated_at": _now(),
        }

        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (skill_dir / "instructions.md").write_text(instructions, encoding="utf-8")

        return f"Created skill '{name}' from conversation pattern"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error creating skill from pattern: {e}"


async def atlas_skill_version(name: str, version: str = "", message: str = "") -> str:
    """Create a version snapshot of a skill.

    param name (str): — Skill name.
    param version (str): — Version tag. Default: auto-increment.
    param message (str): — Version message/notes.
    """
    def _do():
        _ensure_dirs()
        manifest_path = _INSTALLED_DIR / name / "manifest.json"
        if not manifest_path.exists():
            return f"Error: Skill '{name}' not found."

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_ver = manifest.get("version", "1.0.0")

        if not version:
            # Auto-increment patch version
            parts = current_ver.split(".")
            if len(parts) >= 3:
                parts[2] = str(int(parts[2]) + 1)
                version = ".".join(parts)
            else:
                version = f"{current_ver}.1"

        # Save version snapshot
        ver_dir = _VERSIONS_DIR / name / version
        ver_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files from skill dir
        skill_dir = _INSTALLED_DIR / name
        for f in skill_dir.iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(ver_dir / f.name))

        # Write version metadata
        ver_meta = {
            "skill_name": name,
            "version": version,
            "previous_version": current_ver,
            "message": message,
            "created_at": _now(),
        }
        (ver_dir / "version_meta.json").write_text(
            json.dumps(ver_meta, indent=2), encoding="utf-8"
        )

        # Update manifest
        manifest["version"] = version
        manifest["updated_at"] = _now()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return f"Versioned skill '{name}' as v{version} (was v{current_ver})"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error versioning skill: {e}"


async def atlas_skill_versions(name: str) -> str:
    """List all versions of a skill.

    param name (str): — Skill name.
    """
    def _do():
        ver_dir = _VERSIONS_DIR / name
        if not ver_dir.exists():
            return f"No versions found for skill '{name}'"

        versions = []
        for v_dir in sorted(ver_dir.iterdir()):
            meta_path = v_dir / "version_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    versions.append(meta)
                except json.JSONDecodeError:
                    versions.append({"version": v_dir.name})

        if not versions:
            return f"No versions found for skill '{name}'"

        lines = [f"Versions of '{name}' ({len(versions)}):\n"]
        for v in versions:
            msg = v.get("message", "")
            lines.append(
                f"  v{v.get('version', '?')}: "
                f"{v.get('created_at', '')[:16]} "
                f"{'— ' + msg if msg else ''}"
            )

        return "\n".join(lines)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error listing skill versions: {e}"


async def atlas_skill_add_dependency(name: str, dependency: str) -> str:
    """Add a dependency to a skill.

    param name (str): — Skill name.
    param dependency (str): — Name of the required skill dependency.
    """
    def _do():
        _ensure_dirs()
        manifest_path = _INSTALLED_DIR / name / "manifest.json"
        if not manifest_path.exists():
            return f"Error: Skill '{name}' not found."

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        deps = manifest.get("dependencies", [])
        if dependency in deps:
            return f"Skill '{name}' already depends on '{dependency}'"

        deps.append(dependency)
        manifest["dependencies"] = deps
        manifest["updated_at"] = _now()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return f"Added dependency '{dependency}' to skill '{name}' (total: {len(deps)})"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error adding dependency: {e}"


async def atlas_skill_test(name: str, test_input: str = "") -> str:
    """Run a basic validation test on a skill.

    param name (str): — Skill name to test.
    param test_input (str): — Optional test input to validate against.
    """
    def _do():
        _ensure_dirs()
        manifest_path = _INSTALLED_DIR / name / "manifest.json"
        if not manifest_path.exists():
            return f"Error: Skill '{name}' not found."

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results = []

        # Test 1: Manifest validity
        required_fields = ["name", "description", "instructions", "version"]
        for field in required_fields:
            if manifest.get(field):
                results.append(f"  PASS: Manifest has '{field}'")
            else:
                results.append(f"  FAIL: Manifest missing '{field}'")

        # Test 2: Instructions non-empty
        instr = manifest.get("instructions", "")
        if len(instr) > 20:
            results.append(f"  PASS: Instructions present ({len(instr)} chars)")
        else:
            results.append(f"  FAIL: Instructions too short ({len(instr)} chars)")

        # Test 3: Instructions file exists
        instr_file = manifest_path.parent / "instructions.md"
        if instr_file.exists():
            results.append(f"  PASS: instructions.md exists")
        else:
            results.append(f"  WARN: instructions.md missing")

        # Test 4: Dependencies satisfied
        deps = manifest.get("dependencies", [])
        for dep in deps:
            dep_path = _INSTALLED_DIR / dep / "manifest.json"
            if dep_path.exists():
                results.append(f"  PASS: Dependency '{dep}' satisfied")
            else:
                results.append(f"  FAIL: Dependency '{dep}' not installed")

        # Test 5: Test input handling
        if test_input:
            if "{input}" in instr or "$INPUT" in instr:
                results.append(f"  PASS: Instructions reference input variable")
            else:
                results.append(f"  INFO: Instructions do not reference input variable (may be fine)")

        passed = sum(1 for r in results if "PASS" in r)
        failed = sum(1 for r in results if "FAIL" in r)
        total = len(results)

        lines = [
            f"Skill Test Results: {name}",
            f"  {passed}/{total} passed, {failed} failed",
            "",
        ]
        lines.extend(results)

        return "\n".join(lines)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error testing skill: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="atlas_skill_from_conversation",
    func=atlas_skill_from_conversation,
    description="Create a new skill from a recurring conversation pattern.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="atlas_skill_version",
    func=atlas_skill_version,
    description="Create a version snapshot of a skill.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="atlas_skill_versions",
    func=atlas_skill_versions,
    description="List all saved versions of a skill.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="atlas_skill_add_dependency",
    func=atlas_skill_add_dependency,
    description="Add a dependency requirement to a skill.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="atlas_skill_test",
    func=atlas_skill_test,
    description="Run validation tests on a skill.",
    toolset="skills",
)
