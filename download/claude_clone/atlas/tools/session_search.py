"""
Atlas Session Search — full-text search across conversation history.

Features:
- FTS5-based search (SQLite FTS5 virtual table)
- Keyword search with relevance ranking
- Result highlighting
- Export search results
- Session-scoped search
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from atlas.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_DB_PATH = Path.home() / ".claude_clone" / "atlas_sessions.db"
_FTSPATH = Path.home() / ".claude_clone" / "atlas_sessions_fts.db"


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_FTSPATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            metadata    TEXT DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            name        TEXT DEFAULT '',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)

    # FTS5 virtual table
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content, session_id, role, timestamp,
                content='messages', content_rowid='rowid'
            )
        """)
    except sqlite3.OperationalError:
        pass  # FTS5 table already exists

    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp)")
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _highlight(text: str, query: str, max_len: int = 200) -> str:
    """Highlight query terms in text and return a snippet."""
    terms = re.findall(r"\w+", query.lower())
    if not terms:
        return text[:max_len]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        sent_lower = sent.lower()
        for term in terms:
            if term in sent_lower:
                snippet = sent.strip()
                if len(snippet) > max_len:
                    snippet = snippet[:max_len] + "..."
                # Bold the matching terms
                for t in terms:
                    pattern = re.compile(re.escape(t), re.IGNORECASE)
                    snippet = pattern.sub(f"**{t}**", snippet)
                return snippet

    return text[:max_len] + "..."


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def atlas_session_store(
    session_id: str = "default",
    role: str = "user",
    content: str = "",
) -> str:
    """Store a message in the session search index.

    param session_id (str): — Session identifier. Default: default.
    param role (str): — Message role (user, assistant, system).
    param content (str): — Message content to index.
    """
    import uuid

    def _do():
        conn = _get_conn()
        now = _now()
        msg_id = uuid.uuid4().hex[:12]

        # Ensure session exists
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, name, created_at, updated_at) VALUES (?, '', ?, ?)",
            (session_id, now, now),
        )

        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (msg_id, session_id, role, content, now),
        )

        # Sync FTS
        try:
            conn.execute(
                "INSERT INTO messages_fts(content, session_id, role, timestamp) VALUES (?, ?, ?, ?)",
                (content, session_id, role, now),
            )
        except Exception:
            pass

        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        conn.commit()
        conn.close()
        return f"Indexed message {msg_id} in session {session_id}"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error storing session message: {e}"


async def atlas_session_search(
    query: str,
    session_id: str = "",
    limit: int = 10,
) -> str:
    """Full-text search across conversation history.

    param query (str): — Search query (supports FTS5 syntax).
    param session_id (str): — Limit search to a specific session.
    param limit (int): — Max results. Default: 10.
    """
    def _do():
        conn = _get_conn()

        # Build FTS query
        fts_query = query
        if not re.search(r"[AND OR NOT *:]", query):
            fts_query = " OR ".join(query.split())

        try:
            if session_id:
                rows = conn.execute("""
                    SELECT m.content, m.role, m.timestamp, m.session_id,
                           rank
                    FROM messages_fts f
                    JOIN messages m ON m.rowid = f.rowid
                    WHERE messages_fts MATCH ? AND m.session_id = ?
                    ORDER BY rank
                    LIMIT ?
                """, (fts_query, session_id, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT m.content, m.role, m.timestamp, m.session_id,
                           rank
                    FROM messages_fts f
                    JOIN messages m ON m.rowid = f.rowid
                    WHERE messages_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (fts_query, limit)).fetchall()
        except sqlite3.OperationalError:
            # FTS query syntax error — fallback to LIKE
            like = f"%{query}%"
            params: List[Any] = [like]
            sql = "SELECT content, role, timestamp, session_id, 0 as rank FROM messages WHERE content LIKE ?"
            if session_id:
                sql += " AND session_id = ?"
                params.append(session_id)
            sql += " LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()

        conn.close()

        if not rows:
            scope = f" in session {session_id}" if session_id else ""
            return f"No results found for '{query}'{scope}."

        lines = [f"Found {len(rows)} result(s) for '{query}':\n"]
        for i, r in enumerate(rows, 1):
            snippet = _highlight(r["content"], query)
            lines.append(
                f"{i}. [{r['role']}] {r['timestamp'][:16]} (session: {r['session_id']})\n"
                f"   {snippet}"
            )

        return "\n".join(lines)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error searching sessions: {e}"


async def atlas_session_list(limit: int = 20) -> str:
    """List all indexed sessions.

    param limit (int): — Max sessions to list. Default: 20.
    """
    def _do():
        conn = _get_conn()
        rows = conn.execute("""
            SELECT s.session_id, s.name, s.created_at, s.updated_at,
                   COUNT(m.id) as msg_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()

        if not rows:
            return "No sessions indexed yet."

        lines = [f"Indexed sessions ({len(rows)}):\n"]
        for r in rows:
            lines.append(
                f"  {r['session_id']}: {r['msg_count']} messages, "
                f"updated {r['updated_at'][:16]}"
            )

        return "\n".join(lines)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error listing sessions: {e}"


async def atlas_session_export(session_id: str, output_path: str = "") -> str:
    """Export a session's messages to a JSON file.

    param session_id (str): — Session to export.
    param output_path (str): — Output file path. Default: auto-generated.
    """
    if not output_path:
        output_path = str(
            Path.home() / ".claude_clone" / f"session_{session_id}_export.json"
        )

    def _do():
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
        conn.close()

        if not rows:
            return f"No messages found for session {session_id}"

        messages = [
            {"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]}
            for r in rows
        ]

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8")
        return f"Exported {len(messages)} messages to {out}"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error exporting session: {e}"


async def atlas_session_delete(session_id: str = "") -> str:
    """Delete a session and all its indexed messages.

    param session_id (str): — Session ID to delete. Empty = delete all.
    """
    def _do():
        conn = _get_conn()
        if session_id:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            # FTS rebuild needed
            try:
                conn.execute("DELETE FROM messages_fts WHERE session_id = ?", (session_id,))
            except Exception:
                pass
            conn.commit()
            conn.close()
            return f"Deleted session {session_id} and all its messages"
        else:
            count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM sessions")
            try:
                conn.execute("DELETE FROM messages_fts")
            except Exception:
                pass
            conn.commit()
            conn.close()
            return f"Deleted all sessions ({count} messages)"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error deleting session: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="atlas_session_store",
    func=atlas_session_store,
    description="Index a message in the session search database.",
    toolset="session",
)

ToolRegistry.instance().register(
    name="atlas_session_search",
    func=atlas_session_search,
    description="Full-text search across all indexed conversation sessions.",
    toolset="session",
)

ToolRegistry.instance().register(
    name="atlas_session_list",
    func=atlas_session_list,
    description="List all indexed conversation sessions.",
    toolset="session",
)

ToolRegistry.instance().register(
    name="atlas_session_export",
    func=atlas_session_export,
    description="Export a session's messages to a JSON file.",
    toolset="session",
)

ToolRegistry.instance().register(
    name="atlas_session_delete",
    func=atlas_session_delete,
    description="Delete a session and all its indexed messages.",
    toolset="session",
)
