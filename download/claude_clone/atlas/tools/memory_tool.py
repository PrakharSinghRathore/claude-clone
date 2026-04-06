"""
Atlas Memory Tool — persistent memory operations with tags and importance.

Features:
- Store, retrieve, search, delete memories
- Tag-based organization
- Importance scoring (1-10)
- Auto-summarization triggers
- SQLite-backed persistence
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from atlas.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

_DB_PATH = Path.home() / ".claude_clone" / "atlas_memory.db"


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id          TEXT PRIMARY KEY,
            content     TEXT NOT NULL,
            tags        TEXT DEFAULT '[]',
            importance  INTEGER DEFAULT 5,
            category    TEXT DEFAULT 'general',
            summary     TEXT DEFAULT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            metadata    TEXT DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)
    """)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Auto-summarization
# ---------------------------------------------------------------------------

_SUMMARIZE_THRESHOLD = 20  # Trigger summarization if content > this many words


def _should_summarize(content: str) -> bool:
    return len(content.split()) > _SUMMARIZE_THRESHOLD


def _auto_summarize(content: str) -> str:
    """Simple extractive summarization: first sentence + key phrases."""
    sentences = re.split(r"(?<=[.!?])\s+", content)
    first = sentences[0].strip() if sentences else ""

    # Extract key phrases (capitalized multi-word phrases)
    phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", content)
    unique_phrases = list(dict.fromkeys(phrases))[:5]

    summary = first
    if unique_phrases:
        summary += " Keywords: " + ", ".join(unique_phrases)

    if len(summary) > 300:
        summary = summary[:297] + "..."

    return summary


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def atlas_memory_store(
    content: str,
    tags: str = "[]",
    importance: int = 5,
    category: str = "general",
) -> str:
    """Store a new memory with tags and importance score.

    param content (str): — The memory content to store.
    param tags (str): — JSON array of tag strings. Default: [].
    param importance (int): — Importance score 1-10. Default: 5.
    param category (str): — Category label. Default: general.
    """
    try:
        tag_list = json.loads(tags) if isinstance(tags, str) else tags
        if not isinstance(tag_list, list):
            tag_list = [str(tag_list)]
    except (json.JSONDecodeError, TypeError):
        tag_list = [tags]

    importance = max(1, min(10, importance))
    now = _now()
    mem_id = _gen_id()

    summary = None
    if _should_summarize(content):
        summary = _auto_summarize(content)

    def _do():
        conn = _get_conn()
        conn.execute(
            "INSERT INTO memories (id, content, tags, importance, category, summary, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (mem_id, content, json.dumps(tag_list), importance, category, summary, now, now),
        )
        conn.commit()
        conn.close()

    try:
        await _run_sync(_do)
        return f"Stored memory {mem_id} (importance={importance}, category={category}, tags={tag_list})"
    except Exception as e:
        return f"Error storing memory: {e}"


async def atlas_memory_retrieve(memory_id: str = "", query: str = "", limit: int = 5) -> str:
    """Retrieve memories by ID or search query.

    param memory_id (str): — Specific memory ID to retrieve.
    param query (str): — Search query for keyword matching.
    param limit (int): — Max results. Default: 5.
    """
    def _do():
        conn = _get_conn()

        if memory_id:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            conn.close()
            if row is None:
                return f"Memory not found: {memory_id}"
            return _format_memory(dict(row))

        if query:
            # Keyword search with LIKE
            like_pattern = f"%{query}%"
            rows = conn.execute(
                "SELECT * FROM memories WHERE content LIKE ? OR tags LIKE ? OR category LIKE ? "
                "ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (like_pattern, like_pattern, like_pattern, limit),
            ).fetchall()
            conn.close()

            if not rows:
                return f"No memories found matching '{query}'"
            return f"Found {len(rows)} memories matching '{query}':\n\n" + "\n---\n".join(
                _format_memory(dict(r)) for r in rows
            )

        # List recent
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()

        if not rows:
            return "No memories stored yet."
        return f"Recent memories ({len(rows)}):\n\n" + "\n---\n".join(
            _format_memory(dict(r)) for r in rows
        )

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error retrieving memory: {e}"


def _format_memory(row: Dict[str, Any]) -> str:
    tags = json.loads(row.get("tags", "[]"))
    summary = row.get("summary", "")
    content = summary or row.get("content", "")
    if len(content) > 500:
        content = content[:500] + "..."
    lines = [
        f"ID: {row['id']}",
        f"Category: {row.get('category', 'general')}",
        f"Importance: {row.get('importance', 5)}/10",
        f"Tags: {', '.join(tags) if tags else '(none)'}",
        f"Updated: {row.get('updated_at', '')}",
        f"",
        content,
    ]
    return "\n".join(lines)


async def atlas_memory_delete(memory_id: str = "", category: str = "") -> str:
    """Delete a memory by ID or entire category.

    param memory_id (str): — Memory ID to delete.
    param category (str): — Delete all memories in this category.
    """
    def _do():
        conn = _get_conn()
        if memory_id:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            conn.close()
            return f"Deleted {cur.rowcount} memory(ies) with ID {memory_id}" if cur.rowcount else f"Memory not found: {memory_id}"
        elif category:
            cur = conn.execute("DELETE FROM memories WHERE category = ?", (category,))
            conn.commit()
            conn.close()
            return f"Deleted {cur.rowcount} memories in category '{category}'"
        else:
            conn.close()
            return "Error: Provide memory_id or category"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error deleting memory: {e}"


async def atlas_memory_search(query: str, limit: int = 10) -> str:
    """Full-text search across all memories.

    param query (str): — Search query.
    param limit (int): — Max results. Default: 10.
    """
    return await atlas_memory_retrieve(query=query, limit=limit)


async def atlas_memory_list(
    category: str = "",
    tag: str = "",
    min_importance: int = 0,
    limit: int = 20,
) -> str:
    """List memories with optional filtering.

    param category (str): — Filter by category.
    param tag (str): — Filter by tag.
    param min_importance (int): — Minimum importance score. Default: 0.
    param limit (int): — Max results. Default: 20.
    """
    def _do():
        conn = _get_conn()
        clauses = []
        params: List[Any] = []

        if category:
            clauses.append("category = ?")
            params.append(category)
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        if min_importance > 0:
            clauses.append("importance >= ?")
            params.append(min_importance)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        rows = conn.execute(
            f"SELECT * FROM memories{where} ORDER BY importance DESC, updated_at DESC LIMIT ?",
            params,
        ).fetchall()
        conn.close()

        if not rows:
            return "No memories found matching filters."

        return f"Found {len(rows)} memories:\n\n" + "\n---\n".join(
            _format_memory(dict(r)) for r in rows
        )

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error listing memories: {e}"


async def atlas_memory_stats() -> str:
    """Show memory statistics."""
    def _do():
        conn = _get_conn()
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        categories = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM memories GROUP BY category ORDER BY cnt DESC"
        ).fetchall()
        tag_rows = conn.execute("SELECT tags FROM memories WHERE tags != '[]'").fetchall()

        tag_counts: Dict[str, int] = {}
        for r in tag_rows:
            for t in json.loads(r["tags"]):
                tag_counts[t] = tag_counts.get(t, 0) + 1

        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:10]

        avg_importance = conn.execute("SELECT AVG(importance) FROM memories").fetchone()[0]

        conn.close()

        lines = [
            f"Total memories: {total}",
            f"Average importance: {avg_importance:.1f}/10" if total > 0 else "Average importance: N/A",
            "",
            "Categories:",
        ]
        for cat in categories:
            lines.append(f"  {cat['category']}: {cat['cnt']}")

        if top_tags:
            lines.append("")
            lines.append("Top tags:")
            for t, c in top_tags:
                lines.append(f"  {t}: {c}")

        return "\n".join(lines)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error getting memory stats: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="atlas_memory_store",
    func=atlas_memory_store,
    description="Store a new memory with tags, importance score, and category.",
    toolset="memory",
)

ToolRegistry.instance().register(
    name="atlas_memory_retrieve",
    func=atlas_memory_retrieve,
    description="Retrieve memories by ID or search by keyword query.",
    toolset="memory",
)

ToolRegistry.instance().register(
    name="atlas_memory_delete",
    func=atlas_memory_delete,
    description="Delete a memory by ID or entire category.",
    toolset="memory",
)

ToolRegistry.instance().register(
    name="atlas_memory_search",
    func=atlas_memory_search,
    description="Full-text search across all stored memories.",
    toolset="memory",
)

ToolRegistry.instance().register(
    name="atlas_memory_list",
    func=atlas_memory_list,
    description="List memories with optional category, tag, and importance filters.",
    toolset="memory",
)

ToolRegistry.instance().register(
    name="atlas_memory_stats",
    func=atlas_memory_stats,
    description="Show memory statistics (count, categories, top tags).",
    toolset="memory",
)
