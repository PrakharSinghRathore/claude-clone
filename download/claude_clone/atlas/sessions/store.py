"""
Session Store — Persistent session storage with multiple backends.

Provides the ``Session`` dataclass and ``SessionStore`` class for
persisting session state across restarts. Supports JSON file and SQLite
backends with automatic save-on-update and session continuity guarantees.

Usage::

    from atlas.sessions.store import Session, SessionStore, SessionStatus

    store = SessionStore(backend="json", data_dir="./sessions")
    session = Session(
        id="abc123",
        agent_id="agent-1",
        channel="whatsapp",
        peer_id="+1234567890",
    )
    store.save(session)
    loaded = store.load("abc123")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Session Status
# ──────────────────────────────────────────────────────────────────────────────

class SessionStatus(Enum):
    """Lifecycle status of a session."""

    ACTIVE = "active"
    """Session is currently active and receiving messages."""

    INACTIVE = "inactive"
    """Session exists but is not currently active."""

    CLOSED = "closed"
    """Session has been explicitly closed by user or system."""

    ARCHIVED = "archived"
    """Session has been archived for long-term storage."""


# ──────────────────────────────────────────────────────────────────────────────
# Session Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    """
    Represents a single conversation session.

    Attributes
    ----------
    id:
        Unique session identifier (auto-generated UUID by default).
    agent_id:
        Identifier of the agent owning this session.
    channel:
        Messaging channel/platform (e.g., ``"whatsapp"``, ``"telegram"``).
    peer_id:
        Identifier of the remote peer.
    created_at:
        ISO 8601 timestamp of session creation.
    updated_at:
        ISO 8601 timestamp of last update.
    last_activity:
        ISO 8601 timestamp of last message activity.
    message_count:
        Total number of messages exchanged in this session.
    token_count:
        Approximate token usage for this session.
    status:
        Current lifecycle status.
    metadata:
        Arbitrary key-value metadata attached to the session.
    title:
        Optional human-readable session title.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    agent_id: str = ""
    channel: str = ""
    peer_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_activity: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    message_count: int = 0
    token_count: int = 0
    status: str = SessionStatus.ACTIVE.value
    metadata: Dict[str, Any] = field(default_factory=dict)
    title: Optional[str] = None

    @property
    def status_enum(self) -> SessionStatus:
        """Return the status as a ``SessionStatus`` enum member."""
        try:
            return SessionStatus(self.status)
        except ValueError:
            return SessionStatus.ACTIVE

    @property
    def created_at_dt(self) -> datetime:
        """Parse ``created_at`` to a timezone-aware datetime."""
        return self._parse_iso(self.created_at)

    @property
    def updated_at_dt(self) -> datetime:
        """Parse ``updated_at`` to a timezone-aware datetime."""
        return self._parse_iso(self.updated_at)

    @property
    def last_activity_dt(self) -> datetime:
        """Parse ``last_activity`` to a timezone-aware datetime."""
        return self._parse_iso(self.last_activity)

    @property
    def age_seconds(self) -> float:
        """Age of the session in seconds since creation."""
        return (datetime.now(timezone.utc) - self.created_at_dt).total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Seconds since last activity."""
        return (datetime.now(timezone.utc) - self.last_activity_dt).total_seconds()

    def touch(self) -> None:
        """Update ``updated_at`` and ``last_activity`` to now."""
        now = datetime.now(timezone.utc).isoformat()
        self.updated_at = now
        self.last_activity = now

    def increment_messages(self, count: int = 1) -> None:
        """Increment the message counter."""
        self.message_count += count
        self.touch()

    def add_tokens(self, count: int) -> None:
        """Add to the token counter."""
        self.token_count += count
        self.touch()

    def set_status(self, status: SessionStatus) -> None:
        """Set the session status."""
        self.status = status.value
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the session to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Session:
        """Deserialize a session from a dictionary."""
        # Filter to known fields only
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @staticmethod
    def _parse_iso(iso_str: str) -> datetime:
        """Parse an ISO 8601 string to a timezone-aware datetime."""
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)


# ──────────────────────────────────────────────────────────────────────────────
# Session Store Backends
# ──────────────────────────────────────────────────────────────────────────────

class _JSONBackend:
    """JSON file-based session storage backend."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.data_dir / "sessions_index.json"
        self._cache: Dict[str, Session] = {}
        self._dirty: set[str] = set()
        self._load_index()

    def save(self, session: Session) -> None:
        """Save a session to the JSON index."""
        self._cache[session.id] = session
        self._dirty.add(session.id)
        self._flush()

    def load(self, session_id: str) -> Optional[Session]:
        """Load a session by ID."""
        if session_id in self._cache:
            return self._cache[session_id]
        return None

    def delete(self, session_id: str) -> None:
        """Delete a session."""
        self._cache.pop(session_id, None)
        self._dirty.discard(session_id)
        self._flush()

    def list_all(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> List[Session]:
        """List sessions with optional filters."""
        sessions = list(self._cache.values())
        if agent_id:
            sessions = [s for s in sessions if s.agent_id == agent_id]
        if status:
            sessions = [s for s in sessions if s.status == status]
        if channel:
            sessions = [s for s in sessions if s.channel == channel]
        return sessions

    def count(self) -> int:
        """Return the total number of sessions."""
        return len(self._cache)

    def flush(self) -> None:
        """Force write all cached sessions to disk."""
        self._flush()

    def _flush(self) -> None:
        """Write the in-memory cache to the JSON index file."""
        data = {
            sid: session.to_dict()
            for sid, session in self._cache.items()
        }
        tmp_path = self._index_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(self._index_path)
            self._dirty.clear()
        except Exception:
            logger.exception("Failed to flush session index to %s", self._index_path)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _load_index(self) -> None:
        """Load sessions from the JSON index file."""
        if not self._index_path.exists():
            return
        try:
            raw = self._index_path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
            for sid, sdata in data.items():
                if isinstance(sdata, dict):
                    self._cache[sid] = Session.from_dict(sdata)
        except Exception:
            logger.exception("Failed to load session index from %s", self._index_path)


class _SQLiteBackend:
    """SQLite-backed session storage backend."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.data_dir / "sessions.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=10.0,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Create the sessions table if it doesn't exist."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                channel TEXT NOT NULL DEFAULT '',
                peer_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                token_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                title TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_agent
            ON sessions(agent_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_status
            ON sessions(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_channel
            ON sessions(channel)
        """)
        conn.commit()

    def save(self, session: Session) -> None:
        """Upsert a session into the database."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO sessions (id, agent_id, channel, peer_id, created_at,
                updated_at, last_activity, message_count, token_count, status,
                title, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                agent_id = excluded.agent_id,
                channel = excluded.channel,
                peer_id = excluded.peer_id,
                updated_at = excluded.updated_at,
                last_activity = excluded.last_activity,
                message_count = excluded.message_count,
                token_count = excluded.token_count,
                status = excluded.status,
                title = excluded.title,
                metadata = excluded.metadata
        """, (
            session.id,
            session.agent_id,
            session.channel,
            session.peer_id,
            session.created_at,
            session.updated_at,
            session.last_activity,
            session.message_count,
            session.token_count,
            session.status,
            session.title,
            json.dumps(session.metadata, default=str, ensure_ascii=False),
        ))
        conn.commit()

    def load(self, session_id: str) -> Optional[Session]:
        """Load a session by ID from the database."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def delete(self, session_id: str) -> None:
        """Delete a session from the database."""
        conn = self._get_conn()
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()

    def list_all(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> List[Session]:
        """List sessions with optional filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []

        if agent_id is not None:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if channel is not None:
            conditions.append("channel = ?")
            params.append(channel)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        rows = conn.execute(
            f"SELECT * FROM sessions {where} ORDER BY updated_at DESC",
            params,
        ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def count(self) -> int:
        """Return the total number of sessions."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return row[0] if row else 0

    def flush(self) -> None:
        """SQLite auto-commits; this is a no-op."""
        pass

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        """Convert a database row to a Session dataclass."""
        metadata = {}
        try:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            pass
        return Session(
            id=row["id"],
            agent_id=row["agent_id"],
            channel=row["channel"],
            peer_id=row["peer_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_activity=row["last_activity"],
            message_count=row["message_count"],
            token_count=row["token_count"],
            status=row["status"],
            title=row["title"],
            metadata=metadata,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Session Store
# ──────────────────────────────────────────────────────────────────────────────

class SessionStore:
    """
    Persistent session storage with pluggable backends.

    Provides a unified API for saving, loading, querying, and exporting
    sessions. Supports ``"json"`` and ``"sqlite"`` backends, selected
    at initialization time.

    Parameters
    ----------
    backend:
        Storage backend type: ``"json"`` or ``"sqlite"``.
    data_dir:
        Directory for storing session data.
    auto_save:
        If ``True`` (default), save sessions immediately on update.

    Example
    -------
    >>> store = SessionStore(backend="sqlite", data_dir="./data/sessions")
    >>> session = Session(agent_id="a1", channel="telegram", peer_id="u1")
    >>> store.save(session)
    >>> store.load(session.id)  # doctest: +SKIP
    Session(id='...', agent_id='a1', ...)
    """

    def __init__(
        self,
        backend: str = "json",
        data_dir: str | Path = Path.home() / ".claude_clone" / "atlas_sessions",
        auto_save: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.auto_save = auto_save
        self._backend = self._create_backend(backend)

    def _create_backend(
        self, backend: str
    ) -> Union[_JSONBackend, _SQLiteBackend]:
        """Instantiate the storage backend."""
        if backend == "sqlite":
            return _SQLiteBackend(self.data_dir)
        elif backend == "json":
            return _JSONBackend(self.data_dir)
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. "
                f"Supported backends: 'json', 'sqlite'"
            )

    # ── CRUD Operations ────────────────────────────────────────────────

    def save(self, session: Session) -> None:
        """
        Persist a session to storage.

        If ``auto_save`` is enabled, writes immediately. Otherwise,
        the session is queued until :meth:`flush` is called.
        """
        if not session.id:
            raise ValueError("Session must have a non-empty id")
        self._backend.save(session)
        logger.debug("Session %s saved (%s)", session.id, session.status)

    def load(self, session_id: str) -> Optional[Session]:
        """
        Load a session by ID.

        Returns ``None`` if the session does not exist.
        """
        return self._backend.load(session_id)

    def delete(self, session_id: str) -> bool:
        """
        Delete a session by ID.

        Returns
        -------
        bool
            ``True`` if the session was found and deleted, ``False`` otherwise.
        """
        existing = self._backend.load(session_id)
        if existing is None:
            return False
        self._backend.delete(session_id)
        logger.info("Session %s deleted", session_id)
        return True

    # ── Query Operations ───────────────────────────────────────────────

    def list_all(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> List[Session]:
        """
        List sessions with optional filters.

        Parameters
        ----------
        agent_id:
            Filter by agent identifier.
        status:
            Filter by status string (e.g., ``"active"``, ``"closed"``).
        channel:
            Filter by channel type.

        Returns
        -------
        list[Session]
            Matching sessions ordered by most recently updated.
        """
        return self._backend.list_all(
            agent_id=agent_id,
            status=status,
            channel=channel,
        )

    def get_by_peer(self, channel: str, peer_id: str) -> Optional[Session]:
        """
        Find the most recent active session for a given channel and peer.

        Parameters
        ----------
        channel:
            The messaging channel type.
        peer_id:
            The peer's identifier.

        Returns
        -------
        Session or None
        """
        sessions = self.list_all(channel=channel)
        for session in sessions:
            if session.peer_id == peer_id and session.status == SessionStatus.ACTIVE.value:
                return session
        return None

    def count(self, status: Optional[str] = None) -> int:
        """
        Count sessions, optionally filtered by status.

        Parameters
        ----------
        status:
            If provided, count only sessions with this status.

        Returns
        -------
        int
        """
        sessions = self.list_all(status=status)
        return len(sessions)

    # ── Export Operations ──────────────────────────────────────────────

    def export(
        self,
        format: str = "json",
        path: Optional[str | Path] = None,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Path:
        """
        Export sessions to a file.

        Parameters
        ----------
        format:
            Export format: ``"json"`` or ``"jsonl"``.
        path:
            Destination file path. Defaults to ``<data_dir>/export.<format>``.
        agent_id:
            Optional filter by agent ID.
        status:
            Optional filter by status.

        Returns
        -------
        Path
            The path to the exported file.
        """
        sessions = self.list_all(agent_id=agent_id, status=status)

        if path is None:
            suffix = f".{format}"
            export_path = self.data_dir / f"export_{int(time.time())}{suffix}"
        else:
            export_path = Path(path).expanduser().resolve()

        export_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = [s.to_dict() for s in sessions]
            export_path.write_text(
                json.dumps(data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
        elif format == "jsonl":
            lines = [json.dumps(s.to_dict(), default=str, ensure_ascii=False) for s in sessions]
            export_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            raise ValueError(f"Unsupported export format: {format!r}")

        logger.info(
            "Exported %d sessions to %s",
            len(sessions),
            export_path,
        )
        return export_path

    def import_sessions(self, path: str | Path) -> int:
        """
        Import sessions from a JSON or JSONL file.

        Parameters
        ----------
        path:
            Path to the import file.

        Returns
        -------
        int
            Number of sessions imported.
        """
        import_path = Path(path).expanduser().resolve()
        if not import_path.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")

        content = import_path.read_text(encoding="utf-8")
        count = 0

        if import_path.suffix == ".jsonl":
            for line in content.strip().split("\n"):
                if line.strip():
                    data = json.loads(line)
                    session = Session.from_dict(data)
                    self.save(session)
                    count += 1
        else:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        session = Session.from_dict(item)
                        self.save(session)
                        count += 1
            elif isinstance(data, dict):
                # Could be {id: session_data} format
                for sid, sdata in data.items():
                    if isinstance(sdata, dict):
                        sdata["id"] = sdata.get("id", sid)
                        session = Session.from_dict(sdata)
                        self.save(session)
                        count += 1

        logger.info("Imported %d sessions from %s", count, import_path)
        return count

    # ── Maintenance ────────────────────────────────────────────────────

    def flush(self) -> None:
        """Force-write all pending changes to disk."""
        self._backend.flush()

    def close(self) -> None:
        """Close the backend and release resources."""
        self.flush()
        if hasattr(self._backend, "close"):
            self._backend.close()

    def vacuum(self) -> None:
        """Perform maintenance on the storage backend."""
        if isinstance(self._backend, _SQLiteBackend):
            conn = self._backend._get_conn()
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            conn.commit()
            logger.info("SQLite vacuum completed")

    def stats(self) -> Dict[str, Any]:
        """
        Return storage statistics.

        Returns
        -------
        dict
            Keys: ``total``, ``active``, ``inactive``, ``closed``, ``archived``,
            ``by_channel``, ``by_agent``, ``storage_size_bytes``.
        """
        all_sessions = self.list_all()
        status_counts: Dict[str, int] = {}
        channel_counts: Dict[str, int] = {}
        agent_counts: Dict[str, int] = {}

        for s in all_sessions:
            status_counts[s.status] = status_counts.get(s.status, 0) + 1
            channel_counts[s.channel] = channel_counts.get(s.channel, 0) + 1
            agent_counts[s.agent_id] = agent_counts.get(s.agent_id, 0) + 1

        storage_size = 0
        if self.data_dir.exists():
            for f in self.data_dir.iterdir():
                if f.is_file():
                    storage_size += f.stat().st_size

        return {
            "total": len(all_sessions),
            "active": status_counts.get("active", 0),
            "inactive": status_counts.get("inactive", 0),
            "closed": status_counts.get("closed", 0),
            "archived": status_counts.get("archived", 0),
            "by_channel": channel_counts,
            "by_agent": agent_counts,
            "storage_size_bytes": storage_size,
        }
