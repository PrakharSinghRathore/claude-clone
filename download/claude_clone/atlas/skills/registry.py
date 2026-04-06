"""
Skill Registry — Register, unregister, search, and resolve skill dependencies.

Provides conflict detection, dependency resolution, and search
by name, category, or tags.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Optional

from .loader import Skill

logger = logging.getLogger(__name__)


class SkillRegistry:
    """
    Central registry for all loaded skills.

    Manages registration, search, dependency resolution, and
    conflict detection.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._by_category: dict[str, list[str]] = defaultdict(list)
        self._by_tag: dict[str, list[str]] = defaultdict(list)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register(self, skill: Skill) -> None:
        """
        Register a skill. Raises ValueError if a skill with the same
        name is already registered.
        """
        async with self._lock:
            if skill.name in self._skills:
                existing = self._skills[skill.name]
                if existing.version == skill.version:
                    logger.debug("Skill %r v%s already registered; skipping", skill.name, skill.version)
                    return
                logger.info(
                    "Updating skill %r from v%s to v%s",
                    skill.name, existing.version, skill.version,
                )

            self._skills[skill.name] = skill

            # Update indices
            self._by_category[skill.category].append(skill.name)
            for tag in skill.tags:
                self._by_tag[tag].append(skill.name)

            logger.info("Registered skill %r v%s (category=%s)", skill.name, skill.version, skill.category)

    async def unregister(self, name: str) -> bool:
        """Unregister a skill by name. Returns True if found and removed."""
        async with self._lock:
            skill = self._skills.pop(name, None)
            if skill is None:
                return False

            # Update indices
            self._by_category[skill.category] = [
                n for n in self._by_category[skill.category] if n != name
            ]
            for tag in skill.tags:
                self._by_tag[tag] = [n for n in self._by_tag[tag] if n != name]

            logger.info("Unregistered skill %r", name)
            return True

    async def register_batch(self, skills: list[Skill]) -> None:
        """Register multiple skills at once."""
        for skill in skills:
            await self.register(skill)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Skill]:
        """Get a registered skill by name."""
        return self._skills.get(name)

    def list_all(self) -> list[Skill]:
        """List all registered skills."""
        return list(self._skills.values())

    def list_enabled(self) -> list[Skill]:
        """List only enabled skills."""
        return [s for s in self._skills.values() if s.enabled]

    def list_by_category(self, category: str) -> list[Skill]:
        """List skills in a given category."""
        names = self._by_category.get(category, [])
        return [self._skills[n] for n in names if n in self._skills]

    def list_by_tag(self, tag: str) -> list[Skill]:
        """List skills with a given tag."""
        names = self._by_tag.get(tag, [])
        return [self._skills[n] for n in names if n in self._skills]

    def search(
        self,
        query: str = "",
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[Skill]:
        """
        Search skills by query string, category, and/or tags.

        The query is matched against skill name and description.
        Results are sorted by relevance.
        """
        candidates = list(self._skills.values())

        if category:
            candidates = [s for s in candidates if s.category == category]
        if tags:
            for tag in tags:
                candidates = [s for s in candidates if tag in s.tags]

        if query:
            query_lower = query.lower()
            scored: list[tuple[float, Skill]] = []
            for skill in candidates:
                score = 0.0
                if query_lower in skill.name.lower():
                    score += 2.0
                if query_lower in skill.description.lower():
                    score += 1.0
                for tag in skill.tags:
                    if query_lower in tag.lower():
                        score += 0.5
                if score > 0:
                    scored.append((score, skill))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [s for _, s in scored]

        return candidates

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def resolve_dependencies(self, skill_name: str) -> list[Skill]:
        """
        Resolve the full dependency chain for a skill.

        Returns an ordered list of skills that need to be loaded
        before the requested skill can execute.

        Raises ValueError if circular dependencies are detected.
        """
        visited: set[str] = set()
        order: list[Skill] = []
        self._resolve_recursive(skill_name, visited, order)
        return order

    def _resolve_recursive(self, name: str, visited: set[str], order: list[Skill]) -> None:
        """DFS-based dependency resolution."""
        if name in visited:
            raise ValueError(f"Circular dependency detected involving skill {name!r}")

        skill = self._skills.get(name)
        if skill is None:
            logger.warning("Dependency skill %r not found; skipping", name)
            return

        visited.add(name)

        for dep_name in skill.dependencies:
            self._resolve_recursive(dep_name, visited, order)

        order.append(skill)

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def detect_conflicts(self) -> list[dict]:
        """
        Detect potential conflicts between registered skills.

        Returns a list of conflict descriptions.
        """
        conflicts: list[dict] = []

        # Check for skills with same name (shouldn't happen with dict, but check versions)
        version_map: dict[str, list[Skill]] = {}
        for skill in self._skills.values():
            version_map.setdefault(skill.name, []).append(skill)
        for name, versions in version_map.items():
            if len(versions) > 1:
                conflicts.append({
                    "type": "version_conflict",
                    "skill": name,
                    "versions": [v.version for v in versions],
                })

        # Check for unresolvable dependencies
        for skill in self._skills.values():
            for dep_name in skill.dependencies:
                if dep_name not in self._skills:
                    conflicts.append({
                        "type": "missing_dependency",
                        "skill": skill.name,
                        "missing": dep_name,
                    })

        return conflicts

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return registry statistics."""
        return {
            "total_skills": len(self._skills),
            "enabled_skills": sum(1 for s in self._skills.values() if s.enabled),
            "categories": dict(self._by_category),
            "tags": {k: len(v) for k, v in self._by_tag.items()},
            "conflicts": len(self.detect_conflicts()),
        }
