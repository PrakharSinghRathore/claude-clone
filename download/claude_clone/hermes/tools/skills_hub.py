"""
Hermes Skills Hub — browse and install skills from a community marketplace.

Features:
- Browse available skills
- Search by category/tag
- Install from URLs (JSON manifest)
- Rate and review skills
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# In-memory skills catalog (simulated marketplace)
# ---------------------------------------------------------------------------

_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "code_review",
        "description": "Automated code review with best practices checking",
        "category": "development",
        "tags": ["code", "review", "quality"],
        "version": "1.2.0",
        "author": "hermes-community",
        "downloads": 1520,
        "rating": 4.5,
        "source_url": "",
    },
    {
        "name": "api_testing",
        "description": "REST API testing and validation framework",
        "category": "development",
        "tags": ["api", "testing", "rest"],
        "version": "2.0.1",
        "author": "hermes-community",
        "downloads": 890,
        "rating": 4.2,
        "source_url": "",
    },
    {
        "name": "doc_generator",
        "description": "Generate documentation from code comments and structure",
        "category": "development",
        "tags": ["docs", "documentation", "generator"],
        "version": "1.0.3",
        "author": "hermes-community",
        "downloads": 645,
        "rating": 3.9,
        "source_url": "",
    },
    {
        "name": "data_analyzer",
        "description": "Analyze CSV/JSON data and generate summary reports",
        "category": "data",
        "tags": ["data", "analysis", "csv", "json"],
        "version": "1.1.0",
        "author": "hermes-community",
        "downloads": 1120,
        "rating": 4.3,
        "source_url": "",
    },
    {
        "name": "git_workflow",
        "description": "Standardized Git workflow automation (branch, PR, merge)",
        "category": "development",
        "tags": ["git", "workflow", "automation"],
        "version": "1.5.0",
        "author": "hermes-community",
        "downloads": 2030,
        "rating": 4.7,
        "source_url": "",
    },
    {
        "name": "security_scanner",
        "description": "Scan code for common security vulnerabilities",
        "category": "security",
        "tags": ["security", "scanning", "vulnerabilities"],
        "version": "1.3.0",
        "author": "hermes-community",
        "downloads": 780,
        "rating": 4.4,
        "source_url": "",
    },
    {
        "name": "email_composer",
        "description": "Compose professional emails with templates",
        "category": "communication",
        "tags": ["email", "writing", "templates"],
        "version": "1.0.0",
        "author": "hermes-community",
        "downloads": 450,
        "rating": 3.8,
        "source_url": "",
    },
    {
        "name": "project_setup",
        "description": "Initialize new projects with best-practice structure",
        "category": "development",
        "tags": ["project", "setup", "scaffolding"],
        "version": "2.1.0",
        "author": "hermes-community",
        "downloads": 3200,
        "rating": 4.6,
        "source_url": "",
    },
]

_RATINGS_DB_PATH = Path.home() / ".claude_clone" / "skills_ratings.json"


def _load_ratings() -> Dict[str, Any]:
    if _RATINGS_DB_PATH.exists():
        try:
            return json.loads(_RATINGS_DB_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_ratings(data: Dict[str, Any]) -> None:
    _RATINGS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RATINGS_DB_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def hermes_hub_browse(category: str = "", sort_by: str = "downloads") -> str:
    """Browse the skills marketplace.

    param category (str): — Filter by category.
    param sort_by (str): — Sort field: downloads, rating, name. Default: downloads.
    """
    results = list(_CATALOG)

    if category:
        results = [s for s in results if s.get("category", "") == category]

    reverse = sort_by in ("downloads", "rating")
    results.sort(key=lambda s: s.get(sort_by, 0), reverse=reverse)

    if not results:
        return f"No skills found" + (f" in category '{category}'" if category else "") + "."

    lines = [f"Skills Hub — {len(results)} skills" + (f" in '{category}'" if category else "") + ":\n"]
    for s in results:
        rating_str = f"{s['rating']:.1f}" if isinstance(s["rating"], (int, float)) else "?"
        lines.append(
            f"  {s['name']} v{s['version']} [{s['category']}] "
            f"rating={rating_str} downloads={s['downloads']}\n"
            f"    {s['description']}"
        )

    return "\n".join(lines)


async def hermes_hub_search(query: str) -> str:
    """Search the skills marketplace by keyword.

    param query (str): — Search query (matches name, description, tags).
    """
    query_lower = query.lower()
    results = []

    for skill in _CATALOG:
        searchable = (
            skill["name"] + " " + skill["description"] + " " +
            " ".join(skill.get("tags", [])) + " " + skill.get("category", "")
        ).lower()

        if query_lower in searchable:
            results.append(skill)

    if not results:
        return f"No skills found matching '{query}'."

    lines = [f"Search results for '{query}' ({len(results)} found):\n"]
    for s in sorted(results, key=lambda x: x.get("downloads", 0), reverse=True):
        lines.append(
            f"  {s['name']} v{s['version']} — {s['description'][:80]}"
        )

    return "\n".join(lines)


async def hermes_hub_install(name: str) -> str:
    """Install a skill from the marketplace.

    param name (str): — Skill name to install.
    """
    skill = None
    for s in _CATALOG:
        if s["name"] == name:
            skill = s
            break

    if skill is None:
        return f"Error: Skill '{name}' not found in the marketplace."

    from hermes.tools.skills_tool import hermes_skill_create

    result = await hermes_skill_create(
        name=skill["name"],
        description=skill["description"],
        instructions=f"[Marketplace Skill: {skill['name']}]\n\n{skill['description']}\n\nTags: {', '.join(skill.get('tags', []))}",
        category=skill.get("category", "general"),
        version=skill["version"],
    )

    return f"{result}\nInstalled from Hermes Skills Hub (downloads: {skill['downloads']})"


async def hermes_hub_info(name: str) -> str:
    """Get detailed info about a marketplace skill.

    param name (str): — Skill name.
    """
    skill = None
    for s in _CATALOG:
        if s["name"] == name:
            skill = s
            break

    if skill is None:
        return f"Error: Skill '{name}' not found in the marketplace."

    ratings = _load_ratings()
    user_rating = ratings.get(name, {}).get("rating")

    lines = [
        f"Skill: {skill['name']}",
        f"Version: {skill['version']}",
        f"Author: {skill.get('author', 'unknown')}",
        f"Category: {skill.get('category', 'general')}",
        f"Description: {skill['description']}",
        f"Tags: {', '.join(skill.get('tags', []))}",
        f"Rating: {skill['rating']:.1f}/5.0 ({skill['downloads']} downloads)",
    ]
    if user_rating:
        lines.append(f"Your rating: {user_rating}/5")

    return "\n".join(lines)


async def hermes_hub_rate(name: str, rating: int = 5, review: str = "") -> str:
    """Rate and review a marketplace skill.

    param name (str): — Skill name.
    param rating (int): — Rating 1-5. Default: 5.
    param review (str): — Optional review text.
    """
    rating = max(1, min(5, rating))

    def _do():
        ratings = _load_ratings()
        if name not in ratings:
            ratings[name] = {}
        ratings[name]["rating"] = rating
        if review:
            ratings[name]["review"] = review
        ratings[name]["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_ratings(ratings)
        return f"Rated '{name}' {rating}/5" + (f" — {review}" if review else "")

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error rating skill: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="hermes_hub_browse",
    func=hermes_hub_browse,
    description="Browse available skills in the marketplace with filtering and sorting.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="hermes_hub_search",
    func=hermes_hub_search,
    description="Search the skills marketplace by keyword.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="hermes_hub_install",
    func=hermes_hub_install,
    description="Install a skill from the marketplace.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="hermes_hub_info",
    func=hermes_hub_info,
    description="Get detailed information about a marketplace skill.",
    toolset="skills",
)

ToolRegistry.instance().register(
    name="hermes_hub_rate",
    func=hermes_hub_rate,
    description="Rate and review a marketplace skill.",
    toolset="skills",
)
