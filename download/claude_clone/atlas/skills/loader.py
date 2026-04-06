"""
Skill Loader — Loads skills from SKILL.md files on the filesystem.

Handles metadata extraction, template processing, and script
registration for each discovered skill.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """Represents a loaded skill with metadata and instructions."""

    name: str
    description: str
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    instructions: str = ""
    source_path: Optional[Path] = None
    category: str = "general"
    author: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    parameters: dict = field(default_factory=dict)
    scripts: list[str] = field(default_factory=list)
    enabled: bool = True
    execution_count: int = 0
    success_count: int = 0
    last_executed: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "instructions": self.instructions[:500] + ("..." if len(self.instructions) > 500 else ""),
            "source_path": str(self.source_path) if self.source_path else None,
            "category": self.category,
            "author": self.author,
            "enabled": self.enabled,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
        }


class SkillLoader:
    """
    Discovers and loads skills from SKILL.md files.

    Each skill is a directory containing a ``SKILL.md`` file with
    YAML front-matter for metadata and Markdown body for instructions.

    Example SKILL.md::

        ---
        name: research
        description: Web research skill
        version: 1.0.0
        tags: [web, research, information]
        dependencies: []
        ---

        # Research Skill

        When this skill is active, follow these steps...
    """

    SKILL_FILENAME = "SKILL.md"

    def __init__(self, skills_dirs: Optional[list[str | Path]] = None) -> None:
        self._skills_dirs: list[Path] = []
        if skills_dirs:
            for d in skills_dirs:
                self._skills_dirs.append(Path(d).expanduser().resolve())

    def add_skills_dir(self, directory: str | Path) -> None:
        """Add a directory to the skills search path."""
        path = Path(directory).expanduser().resolve()
        if path not in self._skills_dirs:
            self._skills_dirs.append(path)

    def discover_skills(self) -> list[Path]:
        """
        Scan all skills directories and return paths to SKILL.md files.

        Returns a sorted list of discovered skill directories.
        """
        discovered: list[Path] = []
        for skills_dir in self._skills_dirs:
            if not skills_dir.is_dir():
                continue
            for entry in sorted(skills_dir.iterdir()):
                if entry.is_dir():
                    skill_file = entry / self.SKILL_FILENAME
                    if skill_file.exists():
                        discovered.append(entry)
        logger.info("Discovered %d skill(s) from %d directories", len(discovered), len(self._skills_dirs))
        return discovered

    def load_skill(self, skill_dir: Path) -> Optional[Skill]:
        """
        Load a skill from its directory.

        Parses the SKILL.md file, extracts metadata from YAML front-matter,
        and processes the instruction body.
        """
        skill_file = skill_dir / self.SKILL_FILENAME
        if not skill_file.exists():
            logger.warning("SKILL.md not found in %s", skill_dir)
            return None

        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to read %s: %s", skill_file, exc)
            return None

        # Parse front-matter and body
        metadata, body = self._parse_frontmatter(content)
        if not metadata.get("name"):
            logger.warning("Skill in %s has no name; skipping", skill_dir)
            return None

        # Extract scripts from the skill directory
        scripts = self._discover_scripts(skill_dir)

        # Determine category from directory name
        category = skill_dir.parent.name if skill_dir.parent.name != "builtins" else skill_dir.name

        skill = Skill(
            name=metadata["name"],
            description=metadata.get("description", ""),
            version=metadata.get("version", "1.0.0"),
            tags=metadata.get("tags", []),
            dependencies=metadata.get("dependencies", []),
            instructions=body.strip(),
            source_path=skill_dir,
            category=category,
            author=metadata.get("author", ""),
            parameters=metadata.get("parameters", {}),
            scripts=scripts,
        )

        # Process templates in instructions
        skill.instructions = self._process_templates(skill.instructions, metadata)

        logger.info("Loaded skill %r v%s from %s", skill.name, skill.version, skill_dir)
        return skill

    def load_all_skills(self) -> list[Skill]:
        """Discover and load all skills from all directories."""
        skills: list[Skill] = []
        for skill_dir in self.discover_skills():
            skill = self.load_skill(skill_dir)
            if skill:
                skills.append(skill)
        return skills

    # ------------------------------------------------------------------
    # Template processing
    # ------------------------------------------------------------------

    @staticmethod
    def _process_templates(instructions: str, metadata: dict) -> str:
        """
        Process {{variable}} placeholders in instructions using
        metadata values and known context variables.
        """
        context: dict[str, str] = {
            "skill_name": metadata.get("name", ""),
            "skill_version": metadata.get("version", "1.0.0"),
            "skill_author": metadata.get("author", ""),
        }
        # Add all metadata keys
        for key, value in metadata.items():
            if isinstance(value, str):
                context[key] = value

        for var_name, var_value in context.items():
            instructions = instructions.replace(f"{{{{{var_name}}}}}", var_value)

        return instructions

    # ------------------------------------------------------------------
    # Front-matter parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_frontmatter(content: str) -> tuple[dict, str]:
        """
        Parse YAML front-matter from a Markdown file.

        Returns (metadata_dict, body_string).
        """
        pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)
        match = pattern.match(content)
        if not match:
            return {}, content

        frontmatter_str = match.group(1)
        body = match.group(2)

        try:
            import yaml
            metadata = yaml.safe_load(frontmatter_str)
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            logger.debug("YAML parse failed for front-matter; using empty metadata")
            metadata = {}

        return metadata, body

    # ------------------------------------------------------------------
    # Script discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_scripts(skill_dir: Path) -> list[str]:
        """Find executable scripts in the skill directory."""
        scripts: list[str] = []
        for script_file in skill_dir.iterdir():
            if script_file.is_file() and script_file.name != "SKILL.md":
                if script_file.suffix in (".py", ".sh", ".bash"):
                    scripts.append(str(script_file))
        return scripts

    # ------------------------------------------------------------------
    # Skill creation (self-improving loop)
    # ------------------------------------------------------------------

    @staticmethod
    def create_skill_file(
        directory: str | Path,
        name: str,
        description: str,
        instructions: str,
        version: str = "1.0.0",
        tags: Optional[list[str]] = None,
        dependencies: Optional[list[str]] = None,
        author: str = "Agent (auto-generated)",
    ) -> Path:
        """
        Create a new SKILL.md file. Used by the self-improving loop
        when the agent generates a skill from a complex task.

        Returns the path to the created skill directory.
        """
        skill_dir = Path(directory) / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        frontmatter = {
            "name": name,
            "description": description,
            "version": version,
            "tags": tags or [],
            "dependencies": dependencies or [],
            "author": author,
        }

        try:
            import yaml
            yaml_str = yaml.dump(frontmatter, default_flow_style=False)
        except Exception:
            yaml_str = "\n".join(f"{k}: {v}" for k, v in frontmatter.items())

        content = f"---\n{yaml_str}---\n\n# {name}\n\n{instructions}\n"
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")

        logger.info("Created skill file at %s", skill_file)
        return skill_dir
