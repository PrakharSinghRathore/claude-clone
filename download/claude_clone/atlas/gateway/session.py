"""
Session management for the Atlas Gateway.

Manages per-user conversation state across all platforms with support
for session persistence (JSON/SQLite), reset policies, and
multi-platform session linking.

Usage::

    store = SessionStore(reset_policy="token_limit", max_tokens=100000)
    ctx = await store.get_or_create(user_id="alice", platform="telegram")
    ctx.add_message("user", "Hello!")
    await store.save(ctx)
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SESSION_DB_PATH = "~/.claude_clone/gateway_sessions.db"
DEFAULT_SESSION_DIR = "~/.claude_clone/gateway_sessions"


class SessionResetPolicy(str, Enum):
    """Strategies for resetting conversation sessions."""

    MANUAL = "manual"           # Only reset when explicitly requested
    TIMED = "timed"             # Reset after N seconds of inactivity
    TOKEN_LIMIT = "token_limit"  # Reset after cumulative token count
    MESSAGE_COUNT = "message_count"  # Reset after N messages


# ──────────────────────────────────────────────────────────────────────────────
# Session Context
# ──────────────────────────────────────────────────────────────────────────────

class SessionContext:
    """
    Represents a single user's conversation context on a platform.

    Attributes
    ----------
    session_id:
        Unique session identifier.
    user_id:
        Platform-specific user identifier.
    platform:
        Platform name (e.g., "telegram", "discord").
    chat_id:
        Platform-specific chat/group identifier.
    messages:
        Ordered list of conversation messages.
    metadata:
        Arbitrary key-value metadata attached to the session.
    preferences:
        User preferences (format, language, etc.).
    created_at:
        ISO-8601 creation timestamp.
    updated_at:
        ISO-8601 last-activity timestamp.
    linked_platforms:
        Other platforms linked to this session for cross-platform continuity.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: str = "",
        platform: str = "",
        chat_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        linked_platforms: Optional[List[Dict[str, str]]] = None,
    ):
        self.session_id = session_id or uuid.uuid4().hex[:16]
        self.user_id = user_id
        self.platform = platform
        self.chat_id = chat_id or user_id
        self.messages = messages or []
        self.metadata = metadata or {}
        self.preferences = preferences or {
            "format": "markdown",
            "language": "en",
            "timezone": "UTC",
        }
        self.created_at = created_at or _now_iso()
        self.updated_at = updated_at or _now_iso()
        self.linked_platforms = linked_platforms or []

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Append a message to the conversation history.

        Parameters
        ----------
        role:
            One of ``"user"``, ``"assistant"``, ``"system"``.
        content:
            Message text content.
        metadata:
            Optional message-level metadata (token counts, etc.).

        Returns
        -------
        dict
            The newly-added message entry.
        """
        msg: Dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "role": role,
            "content": content,
            "timestamp": _now_iso(),
            "metadata": metadata or {},
        }
        self.messages.append(msg)
        self.updated_at = _now_iso()
        return msg

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return conversation messages, optionally limited to the most recent N."""
        if limit:
            return self.messages[-limit:]
        return list(self.messages)

    def get_message_count(self) -> int:
        """Return the total number of messages in the session."""
        return len(self.messages)

    def estimate_tokens(self) -> int:
        """
        Estimate the total token count for the session.

        Uses a heuristic of ``len(text) // 4`` per message.
        """
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            total += max(1, len(content) // 4)
        return total

    def clear_messages(self) -> int:
        """
        Clear all messages and return the count of cleared messages.
        """
        count = len(self.messages)
        self.messages = []
        self.updated_at = _now_iso()
        return count

    def link_platform(self, platform: str, user_id: str) -> None:
        """Link another platform to this session for cross-platform continuity."""
        existing = {(lp["platform"], lp["user_id"]) for lp in self.linked_platforms}
        if (platform, user_id) not in existing:
            self.linked_platforms.append({
                "platform": platform,
                "user_id": user_id,
                "linked_at": _now_iso(),
            })

    def unlink_platform(self, platform: str) -> None:
        """Remove a linked platform."""
        self.linked_platforms = [
            lp for lp in self.linked_platforms if lp["platform"] != platform
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to a dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "platform": self.platform,
            "chat_id": self.chat_id,
            "messages": self.messages,
            "metadata": self.metadata,
            "preferences": self.preferences,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "linked_platforms": self.linked_platforms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionContext":
        """Reconstruct a SessionContext from a serialized dictionary."""
        return cls(
            session_id=data.get("session_id"),
            user_id=data.get("user_id", ""),
            platform=data.get("platform", ""),
            chat_id=data.get("chat_id", ""),
            messages=data.get("messages", []),
            metadata=data.get("metadata", {}),
            preferences=data.get("preferences", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            linked_platforms=data.get("linked_platforms", []),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Session Store
# ──────────────────────────────────────────────────────────────────────────────

class SessionStore:
    """
    Manages per-user conversation sessions with persistence and reset policies.

    Supports both SQLite and JSON-based persistence backends.

    Parameters
    ----------
    persist_path:
        Path to the SQLite database or JSON directory.
    reset_policy:
        Session reset strategy (see :class:`SessionResetPolicy`).
    timeout:
        Session timeout in seconds (for ``TIMED`` policy).
    max_tokens:
        Maximum tokens before auto-reset (for ``TOKEN_LIMIT`` policy).
    max_messages:
        Maximum messages before auto-reset (for ``MESSAGE_COUNT`` policy).
    backend:
        Persistence backend: ``"sqlite"`` or ``"json"``.
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        reset_policy: str = "token_limit",
        timeout: int = 3600,
        max_tokens: int = 100000,
        max_messages: int = 200,
        backend: str = "sqlite",
    ):
        self._policy = SessionResetPolicy(reset_policy)
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._max_messages = max_messages
        self._backend = backend

        if backend == "sqlite":
            self._db_path = Path(persist_path or DEFAULT_SESSION_DB_PATH).expanduser().resolve()
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn: Optional[sqlite3.Connection] = None
        else:
            self._json_dir = Path(persist_path or DEFAULT_SESSION_DIR).expanduser().resolve()
            self._json_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for active sessions
        self._cache: Dict[str, SessionContext] = {}

    # ── Initialization / Teardown ─────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize the persistence backend."""
        if self._backend == "sqlite":
            await self._init_sqlite()
        # JSON backend requires no initialization

    async def close(self) -> None:
        """Persist all cached sessions and close connections."""
        if self._backend == "sqlite" and self._conn:
            await self._run_sync(self._conn.close)
            self._conn = None

    def _init_sqlite(self) -> None:
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                platform     TEXT NOT NULL,
                chat_id      TEXT NOT NULL DEFAULT '',
                messages     TEXT DEFAULT '[]',
                metadata     TEXT DEFAULT '{}',
                preferences  TEXT DEFAULT '{}',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                linked_platforms TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user_platform
                ON sessions(user_id, platform);
            CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at);
        """)
        self._conn.commit()

    async def _run_sync(self, func, *args, **kwargs):
        """Execute a synchronous function in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: func(*args, **kwargs)
        )

    # ── Session CRUD ──────────────────────────────────────────────────────

    async def get_or_create(
        self,
        user_id: str,
        platform: str,
        chat_id: str = "",
    ) -> SessionContext:
        """
        Get an existing session or create a new one.

        Parameters
        ----------
        user_id:
            Platform-specific user identifier.
        platform:
            Platform name.
        chat_id:
            Platform-specific chat/group identifier.
        """
        cache_key = self._cache_key(user_id, platform, chat_id)

        # Check cache first
        if cache_key in self._cache:
            ctx = self._cache[cache_key]
            # Check reset policy
            if self._should_reset(ctx):
                ctx.clear_messages()
                await self.save(ctx)
            return ctx

        # Load from persistence
        ctx = await self._load(user_id, platform, chat_id)
        if ctx:
            if self._should_reset(ctx):
                ctx.clear_messages()
                await self.save(ctx)
            self._cache[cache_key] = ctx
            return ctx

        # Create new session
        ctx = SessionContext(
            user_id=user_id,
            platform=platform,
            chat_id=chat_id or user_id,
        )
        await self.save(ctx)
        self._cache[cache_key] = ctx
        return ctx

    async def get(
        self,
        user_id: str,
        platform: str,
        chat_id: str = "",
    ) -> Optional[SessionContext]:
        """Get an existing session, or None if not found."""
        cache_key = self._cache_key(user_id, platform, chat_id)
        if cache_key in self._cache:
            return self._cache[cache_key]
        ctx = await self._load(user_id, platform, chat_id)
        if ctx:
            self._cache[cache_key] = ctx
        return ctx

    async def save(self, ctx: SessionContext) -> None:
        """Persist a session context."""
        ctx.updated_at = _now_iso()
        cache_key = self._cache_key(ctx.user_id, ctx.platform, ctx.chat_id)
        self._cache[cache_key] = ctx

        if self._backend == "sqlite":
            await self._save_sqlite(ctx)
        else:
            await self._save_json(ctx)

    async def delete(self, user_id: str, platform: str, chat_id: str = "") -> bool:
        """Delete a session. Returns True if it existed."""
        cache_key = self._cache_key(user_id, platform, chat_id)
        self._cache.pop(cache_key, None)

        if self._backend == "sqlite":
            return await self._delete_sqlite(user_id, platform, chat_id)
        else:
            return await self._delete_json(user_id, platform, chat_id)

    async def reset(self, ctx: SessionContext) -> int:
        """Reset a session's messages. Returns count of cleared messages."""
        count = ctx.clear_messages()
        await self.save(ctx)
        return count

    async def list_sessions(
        self, platform: Optional[str] = None, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List all sessions, optionally filtered by platform."""
        if self._backend == "sqlite":
            return await self._list_sqlite(platform, limit)
        else:
            return await self._list_json(platform, limit)

    # ── Reset Policy ──────────────────────────────────────────────────────

    def _should_reset(self, ctx: SessionContext) -> bool:
        """Check if a session should be reset based on the active policy."""
        if self._policy == SessionResetPolicy.MANUAL:
            return False
        elif self._policy == SessionResetPolicy.TIMED:
            try:
                updated = datetime.fromisoformat(ctx.updated_at)
                elapsed = (datetime.now(timezone.utc) - updated).total_seconds()
                return elapsed > self._timeout
            except (ValueError, TypeError):
                return False
        elif self._policy == SessionResetPolicy.TOKEN_LIMIT:
            return ctx.estimate_tokens() > self._max_tokens
        elif self._policy == SessionResetPolicy.MESSAGE_COUNT:
            return len(ctx.messages) > self._max_messages
        return False

    # ── SQLite Backend ────────────────────────────────────────────────────

    async def _save_sqlite(self, ctx: SessionContext) -> None:
        def _do() -> None:
            assert self._conn is not None
            self._conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, user_id, platform, chat_id, messages, metadata,
                    preferences, created_at, updated_at, linked_platforms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ctx.session_id, ctx.user_id, ctx.platform, ctx.chat_id,
                    json.dumps(ctx.messages, default=str),
                    json.dumps(ctx.metadata, default=str),
                    json.dumps(ctx.preferences, default=str),
                    ctx.created_at, ctx.updated_at,
                    json.dumps(ctx.linked_platforms, default=str),
                ),
            )
            self._conn.commit()

        await self._run_sync(_do)

    async def _load(
        self, user_id: str, platform: str, chat_id: str,
    ) -> Optional[SessionContext]:
        if self._backend == "sqlite":
            return await self._load_sqlite(user_id, platform, chat_id)
        else:
            return await self._load_json(user_id, platform, chat_id)

    async def _load_sqlite(
        self, user_id: str, platform: str, chat_id: str,
    ) -> Optional[SessionContext]:
        def _do() -> Optional[Dict[str, Any]]:
            assert self._conn is not None
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? AND platform = ? AND chat_id = ?",
                (user_id, platform, chat_id),
            ).fetchone()
            if row is None:
                return None
            return dict(row)

        data = await self._run_sync(_do)
        if data is None:
            return None
        return self._row_to_context(data)

    async def _delete_sqlite(self, user_id: str, platform: str, chat_id: str) -> bool:
        def _do() -> bool:
            assert self._conn is not None
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE user_id = ? AND platform = ? AND chat_id = ?",
                (user_id, platform, chat_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

        return await self._run_sync(_do)

    async def _list_sqlite(
        self, platform: Optional[str], limit: int,
    ) -> List[Dict[str, Any]]:
        def _do() -> List[Dict[str, Any]]:
            assert self._conn is not None
            if platform:
                rows = self._conn.execute(
                    "SELECT session_id, user_id, platform, chat_id, created_at, updated_at, "
                    "LENGTH(messages) as msg_size FROM sessions "
                    "WHERE platform = ? ORDER BY updated_at DESC LIMIT ?",
                    (platform, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT session_id, user_id, platform, chat_id, created_at, updated_at, "
                    "LENGTH(messages) as msg_size FROM sessions "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

        return await self._run_sync(_do)

    def _row_to_context(self, row: Dict[str, Any]) -> SessionContext:
        """Convert a database row to a SessionContext."""
        messages = json.loads(row.get("messages", "[]"))
        metadata = json.loads(row.get("metadata", "{}"))
        preferences = json.loads(row.get("preferences", "{}"))
        linked = json.loads(row.get("linked_platforms", "[]"))
        return SessionContext(
            session_id=row["session_id"],
            user_id=row["user_id"],
            platform=row["platform"],
            chat_id=row.get("chat_id", ""),
            messages=messages,
            metadata=metadata,
            preferences=preferences,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            linked_platforms=linked,
        )

    # ── JSON Backend ──────────────────────────────────────────────────────

    async def _save_json(self, ctx: SessionContext) -> None:
        filename = f"{ctx.platform}_{ctx.user_id}_{ctx.chat_id}.json"
        filepath = self._json_dir / filename

        def _do() -> None:
            filepath.write_text(
                json.dumps(ctx.to_dict(), indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )

        await self._run_sync(_do)

    async def _load_json(
        self, user_id: str, platform: str, chat_id: str,
    ) -> Optional[SessionContext]:
        filename = f"{platform}_{user_id}_{chat_id}.json"
        filepath = self._json_dir / filename

        if not filepath.exists():
            return None

        def _do() -> Dict[str, Any]:
            return json.loads(filepath.read_text(encoding="utf-8"))

        data = await self._run_sync(_do)
        return SessionContext.from_dict(data)

    async def _delete_json(self, user_id: str, platform: str, chat_id: str) -> bool:
        filename = f"{platform}_{user_id}_{chat_id}.json"
        filepath = self._json_dir / filename

        def _do() -> bool:
            if filepath.exists():
                filepath.unlink()
                return True
            return False

        return await self._run_sync(_do)

    async def _list_json(
        self, platform: Optional[str], limit: int,
    ) -> List[Dict[str, Any]]:
        def _do() -> List[Dict[str, Any]]:
            results = []
            for fp in sorted(self._json_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
                if len(results) >= limit:
                    break
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    if platform and data.get("platform") != platform:
                        continue
                    results.append({
                        "session_id": data.get("session_id"),
                        "user_id": data.get("user_id"),
                        "platform": data.get("platform"),
                        "chat_id": data.get("chat_id"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "message_count": len(data.get("messages", [])),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
            return results

        return await self._run_sync(_do)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(user_id: str, platform: str, chat_id: str) -> str:
        return f"{platform}:{user_id}:{chat_id}"

    async def get_stats(self) -> Dict[str, Any]:
        """Return session store statistics."""
        sessions = await self.list_sessions(limit=100000)
        total_messages = sum(s.get("msg_size", s.get("message_count", 0)) for s in sessions)
        platforms = {}
        for s in sessions:
            p = s.get("platform", "unknown")
            platforms[p] = platforms.get(p, 0) + 1

        return {
            "total_sessions": len(sessions),
            "cached_sessions": len(self._cache),
            "backend": self._backend,
            "reset_policy": self._policy.value,
            "platforms": platforms,
            "total_messages_approx": total_messages,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()
