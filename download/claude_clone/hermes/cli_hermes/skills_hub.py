"""
Skills Hub — Marketplace browser for Hermes CLI skills.

Browse, search, install, and manage skills from a marketplace.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes.cli_hermes.config_manager import SKILLS_DIR, ConfigManager
from hermes.cli_hermes.skills_config import SKILL_CATEGORIES


# ──────────────────────────────────────────────
# Local skills registry (marketplace simulation)
# ──────────────────────────────────────────────

MARKETPLACE_SKILLS = [
    {
        "id": "code-review",
        "name": "Code Review",
        "version": "1.2.0",
        "category": "analysis",
        "description": "Automated code review with best practices checking",
        "author": "Hermes Team",
        "rating": 4.8,
        "downloads": 1520,
        "tags": ["review", "quality", "best-practices"],
        "dependencies": [],
        "size": "12KB",
    },
    {
        "id": "test-generator",
        "name": "Test Generator",
        "version": "1.1.0",
        "category": "development",
        "description": "Generate unit tests from existing code",
        "author": "Hermes Team",
        "rating": 4.6,
        "downloads": 980,
        "tags": ["testing", "unittest", "pytest"],
        "dependencies": [],
        "size": "8KB",
    },
    {
        "id": "doc-generator",
        "name": "Documentation Generator",
        "version": "1.0.0",
        "category": "development",
        "description": "Auto-generate documentation from code",
        "author": "Hermes Team",
        "rating": 4.5,
        "downloads": 750,
        "tags": ["docs", "documentation", "readme"],
        "dependencies": [],
        "size": "15KB",
    },
    {
        "id": "api-explorer",
        "name": "API Explorer",
        "version": "1.3.0",
        "category": "development",
        "description": "Explore and test REST APIs interactively",
        "author": "Hermes Team",
        "rating": 4.7,
        "downloads": 1200,
        "tags": ["api", "rest", "http"],
        "dependencies": ["httpx"],
        "size": "20KB",
    },
    {
        "id": "sql-helper",
        "name": "SQL Helper",
        "version": "1.0.0",
        "category": "data",
        "description": "SQL query builder and database management",
        "author": "Hermes Team",
        "rating": 4.4,
        "downloads": 620,
        "tags": ["sql", "database", "query"],
        "dependencies": [],
        "size": "10KB",
    },
    {
        "id": "git-workflow",
        "name": "Git Workflow",
        "version": "1.1.0",
        "category": "devops",
        "description": "Advanced git workflow management and automation",
        "author": "Hermes Team",
        "rating": 4.3,
        "downloads": 540,
        "tags": ["git", "workflow", "branching"],
        "dependencies": [],
        "size": "6KB",
    },
    {
        "id": "perf-analyzer",
        "name": "Performance Analyzer",
        "version": "1.0.0",
        "category": "analysis",
        "description": "Profile and analyze code performance",
        "author": "Hermes Team",
        "rating": 4.2,
        "downloads": 380,
        "tags": ["performance", "profiling", "optimization"],
        "dependencies": [],
        "size": "14KB",
    },
    {
        "id": "secrets-scanner",
        "name": "Secrets Scanner",
        "version": "1.2.0",
        "category": "security",
        "description": "Scan code for hardcoded secrets and credentials",
        "author": "Hermes Team",
        "rating": 4.9,
        "downloads": 2100,
        "tags": ["security", "secrets", "credentials"],
        "dependencies": [],
        "size": "9KB",
    },
    {
        "id": "yaml-builder",
        "name": "YAML Builder",
        "version": "1.0.0",
        "category": "devops",
        "description": "Build and validate YAML configurations",
        "author": "Hermes Team",
        "rating": 4.1,
        "downloads": 290,
        "tags": ["yaml", "config", "validation"],
        "dependencies": ["pyyaml"],
        "size": "5KB",
    },
    {
        "id": "regex-helper",
        "name": "Regex Helper",
        "version": "1.1.0",
        "category": "development",
        "description": "Build, test, and explain regular expressions",
        "author": "Hermes Team",
        "rating": 4.5,
        "downloads": 870,
        "tags": ["regex", "pattern", "matching"],
        "dependencies": [],
        "size": "7KB",
    },
    {
        "id": "commit-writer",
        "name": "Commit Writer",
        "version": "1.0.0",
        "category": "development",
        "description": "Generate conventional commit messages from diffs",
        "author": "Hermes Team",
        "rating": 4.6,
        "downloads": 1100,
        "tags": ["git", "commit", "conventional"],
        "dependencies": [],
        "size": "4KB",
    },
    {
        "id": "markdown-formatter",
        "name": "Markdown Formatter",
        "version": "1.0.0",
        "category": "creative",
        "description": "Format and lint markdown files",
        "author": "Hermes Team",
        "rating": 4.3,
        "downloads": 450,
        "tags": ["markdown", "formatting", "linting"],
        "dependencies": [],
        "size": "6KB",
    },
]


class SkillsHub:
    """Browse, search, install, and manage skills from the marketplace."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.skills_dir = SKILLS_DIR
        self._marketplace = {s["id"]: s for s in MARKETPLACE_SKILLS}

    def browse(
        self,
        category: Optional[str] = None,
        sort_by: str = "rating",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Browse available skills in the marketplace."""
        skills = list(self._marketplace.values())

        if category:
            skills = [s for s in skills if s.get("category") == category]

        # Sort
        reverse = True
        if sort_by == "name":
            skills.sort(key=lambda s: s.get("name", "").lower(), reverse=False)
        elif sort_by == "rating":
            skills.sort(key=lambda s: s.get("rating", 0), reverse=reverse)
        elif sort_by == "downloads":
            skills.sort(key=lambda s: s.get("downloads", 0), reverse=reverse)
        elif sort_by == "newest":
            skills.sort(key=lambda s: s.get("version", "0.0.0"), reverse=reverse)

        return skills[:limit]

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search skills by name, description, or tags."""
        query = query.lower()
        results = []

        for skill in self._marketplace.values():
            search_text = f"{skill['name']} {skill['description']} {' '.join(skill.get('tags', []))}".lower()
            if query in search_text:
                results.append(skill)

        # Score by relevance
        def _score(s):
            score = 0
            if query in s["name"].lower():
                score += 10
            if query in s["description"].lower():
                score += 5
            for tag in s.get("tags", []):
                if query in tag.lower():
                    score += 3
            score += s.get("rating", 0)
            return score

        results.sort(key=_score, reverse=True)
        return results[:limit]

    def get_skill_details(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a marketplace skill."""
        return self._marketplace.get(skill_id)

    def install(self, skill_id: str) -> Dict[str, Any]:
        """Install a skill from the marketplace."""
        skill = self._marketplace.get(skill_id)
        if not skill:
            return {"success": False, "error": f"Skill '{skill_id}' not found in marketplace"}

        skill_dir = self.skills_dir / skill_id
        if skill_dir.exists():
            return {"success": False, "error": f"Skill '{skill_id}' is already installed"}

        # Create skill directory and manifest
        skill_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            **skill,
            "installed_at": datetime.now().isoformat(),
            "install_source": "hub",
        }

        manifest_path = skill_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Create a placeholder skill file
        skill_file = skill_dir / f"{skill_id}.py"
        skill_file.write_text(f'"""\n{skill["name"]} - {skill["description"]}\n\nInstalled from Hermes Skills Hub.\n"""\n\n# Skill implementation goes here\n')

        return {
            "success": True,
            "skill_id": skill_id,
            "path": str(skill_dir),
        }

    def uninstall(self, skill_id: str) -> bool:
        """Uninstall a skill."""
        skill_dir = self.skills_dir / skill_id
        if not skill_dir.exists():
            return False

        import shutil
        shutil.rmtree(skill_dir)
        return True

    def is_installed(self, skill_id: str) -> bool:
        """Check if a skill is installed."""
        return (self.skills_dir / skill_id).exists()

    def get_categories(self) -> List[Dict[str, str]]:
        """Get available skill categories."""
        categories = []
        for cat_id, cat_info in SKILL_CATEGORIES.items():
            count = sum(1 for s in self._marketplace.values() if s.get("category") == cat_id)
            if count > 0:
                categories.append({
                    "id": cat_id,
                    "name": cat_info["name"],
                    "emoji": cat_info.get("emoji", ""),
                    "description": cat_info.get("description", ""),
                    "skill_count": count,
                })
        return categories

    def get_installed_count(self) -> int:
        """Get count of installed skills."""
        return sum(1 for d in self.skills_dir.iterdir() if d.is_dir())

    def get_marketplace_count(self) -> int:
        """Get total marketplace skill count."""
        return len(self._marketplace)

    def format_skill_card(self, skill: Dict[str, Any]) -> str:
        """Format a skill as a display card."""
        rating_stars = "\u2605" * int(skill.get("rating", 0)) + "\u2606" * (5 - int(skill.get("rating", 0)))
        installed_marker = "  \033[32m[installed]\033[0m" if self.is_installed(skill["id"]) else ""

        lines = [
            f"  \033[1m{skill.get('name', skill['id'])}\033[0m v{skill.get('version', '?')}{installed_marker}",
            f"    {skill.get('description', '')}",
            f"    Rating: {rating_stars} ({skill.get('rating', 0)})  |  Downloads: {skill.get('downloads', 0):,}",
            f"    Category: {skill.get('category', '')}  |  Size: {skill.get('size', '?')}",
        ]

        if skill.get("tags"):
            lines.append(f"    Tags: {', '.join(skill['tags'])}")

        return "\n".join(lines)

    def format_marketplace_table(
        self,
        skills: Optional[List[Dict]] = None,
    ) -> str:
        """Format marketplace skills as a table."""
        skills = skills or self.browse()

        lines = []
        lines.append(f"  {'Skill':<22} {'Version':<8} {'Category':<12} {'Rating':>7} {'Downloads':>10}")
        lines.append("  " + "-" * 65)

        for skill in skills:
            name = skill.get("name", skill["id"])
            installed = " \u2713" if self.is_installed(skill["id"]) else ""
            lines.append(
                f"  {name:<22} {skill.get('version', '?'):<8} "
                f"{skill.get('category', ''):<12} "
                f"{skill.get('rating', 0):>6.1f}  "
                f"{skill.get('downloads', 0):>9,}{installed}"
            )

        return "\n".join(lines)
