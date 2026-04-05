"""
Skill Manager — Orchestration layer for the skills system.

Coordinates discovery, loading, registration, and execution
of skills with dependency resolution and the self-improving loop.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Optional

from .executor import ExecutionResult, SkillExecutor
from .loader import Skill, SkillLoader
from .registry import SkillRegistry

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_DIR = Path(__file__).parent / "builtins"


class SkillManager:
    """
    Top-level manager that coordinates all skill subsystems.

    Provides a unified interface for:
    - Discovering and loading skills from the filesystem
    - Registering and managing skills in the registry
    - Executing skills with parameter substitution
    - Creating new skills from complex tasks (self-improving loop)
    """

    def __init__(
        self,
        skills_dirs: Optional[list[str | Path]] = None,
        step_handler: Optional[Callable[[str, dict], Any]] = None,
    ) -> None:
        dirs = skills_dirs or [str(_DEFAULT_SKILLS_DIR)]
        self._loader = SkillLoader(skills_dirs=dirs)
        self._registry = SkillRegistry()
        self._executor = SkillExecutor(step_handler=step_handler)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Discover and load all skills from configured directories.

        Should be called once at application startup.
        """
        skills = self._loader.load_all_skills()
        await self._registry.register_batch(skills)

        # Log conflicts
        conflicts = self._registry.detect_conflicts()
        if conflicts:
            for conflict in conflicts:
                logger.warning(
                    "Skill conflict: %s (%s)",
                    conflict["type"],
                    conflict.get("skill", conflict.get("missing", "")),
                )

        logger.info(
            "Skills system initialized: %d skill(s) loaded, %d conflict(s)",
            len(skills),
            len(conflicts),
        )

    # ------------------------------------------------------------------
    # Skill discovery and loading
    # ------------------------------------------------------------------

    def add_skills_dir(self, directory: str | Path) -> None:
        """Add a directory to the skills search path."""
        self._loader.add_skills_dir(directory)

    async def reload(self) -> list[Skill]:
        """Re-scan all directories and reload all skills."""
        skills = self._loader.load_all_skills()
        await self._registry.register_batch(skills)
        logger.info("Reloaded %d skill(s)", len(skills))
        return skills

    async def load_skill_from_dir(self, directory: str | Path) -> Optional[Skill]:
        """Load a single skill from a directory."""
        skill = self._loader.load_skill(Path(directory))
        if skill:
            await self._registry.register(skill)
        return skill

    # ------------------------------------------------------------------
    # Skill execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        skill_name: str,
        params: Optional[dict] = None,
        context: Optional[dict] = None,
    ) -> ExecutionResult:
        """
        Execute a registered skill by name.

        Automatically resolves dependencies before execution.
        """
        skill = self._registry.get(skill_name)
        if skill is None:
            return ExecutionResult(
                skill_name=skill_name,
                success=False,
                error=f"Skill {skill_name!r} not found",
            )
        if not skill.enabled:
            return ExecutionResult(
                skill_name=skill_name,
                success=False,
                error=f"Skill {skill_name!r} is disabled",
            )

        # Resolve dependencies
        try:
            deps = self._registry.resolve_dependencies(skill_name)
            logger.debug("Resolved %d dependencies for %r", len(deps) - 1, skill_name)
        except ValueError as exc:
            return ExecutionResult(
                skill_name=skill_name,
                success=False,
                error=str(exc),
            )

        return await self._executor.execute(skill, params, context)

    async def execute_stream(
        self,
        skill_name: str,
        params: Optional[dict] = None,
        context: Optional[dict] = None,
    ) -> AsyncGenerator:
        """Execute a skill and yield step results in real-time."""
        skill = self._registry.get(skill_name)
        if skill is None:
            raise KeyError(f"Skill {skill_name!r} not found")
        async for step in self._executor.execute_stream(skill, params, context):
            yield step

    # ------------------------------------------------------------------
    # Skill queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Skill]:
        """Get a registered skill by name."""
        return self._registry.get(name)

    def list_all(self) -> list[Skill]:
        """List all registered skills."""
        return self._registry.list_all()

    def search(
        self,
        query: str = "",
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[Skill]:
        """Search for skills."""
        return self._registry.search(query, category, tags)

    def stats(self) -> dict:
        """Get skills system statistics."""
        return self._registry.stats()

    # ------------------------------------------------------------------
    # Self-improving loop
    # ------------------------------------------------------------------

    async def create_skill_from_task(
        self,
        task_description: str,
        steps_taken: list[str],
        outcome: str,
        name: Optional[str] = None,
        target_dir: Optional[str | Path] = None,
    ) -> Optional[Skill]:
        """
        Create a new skill from a completed complex task.

        This is the core of the self-improving loop: when the agent
        successfully completes a complex multi-step task, it can
        codify the approach as a reusable skill.

        Returns the newly created Skill, or None on failure.
        """
        skill_dir = target_dir or str(_DEFAULT_SKILLS_DIR)
        skill_name = name

        content = self._executor.generate_skill_from_task(
            task_description=task_description,
            steps_taken=steps_taken,
            outcome=outcome,
            name=skill_name,
        )

        created_dir = self._loader.create_skill_file(
            directory=skill_dir,
            name=skill_name or "unnamed",
            description=f"Auto-generated from task: {task_description[:100]}",
            instructions=content,
        )

        # Reload and register
        skill = self._loader.load_skill(created_dir)
        if skill:
            await self._registry.register(skill)
            logger.info(
                "Self-improving loop created skill %r from task",
                skill.name,
            )
        return skill

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    async def enable_skill(self, name: str) -> bool:
        """Enable a registered skill."""
        skill = self._registry.get(name)
        if skill:
            skill.enabled = True
            return True
        return False

    async def disable_skill(self, name: str) -> bool:
        """Disable a registered skill."""
        skill = self._registry.get(name)
        if skill:
            skill.enabled = False
            return True
        return False

    async def unregister_skill(self, name: str) -> bool:
        """Unregister a skill."""
        return await self._registry.unregister(name)
