"""
ACP Session Management — Session creation, lifecycle, conversation
state, tool call tracking, and session export/import.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path.home() / ".claude_clone" / "acp"
_SESSIONS_DIR = "sessions"


class SessionState(str, Enum):
    """Lifecycle states of an ACP session."""

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


@dataclass
class MessageRecord:
    """A single message in the session history."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class ToolCallRecord:
    """A record of a tool invocation within a session."""

    call_id: str
    tool_name: str
    params: dict = field(default_factory=dict)
    result: Any = None
    status: str = "pending"  # pending | completed | failed
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ACPSession:
    """An active ACP session with full conversation state."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = SessionState.CREATED
    user_id: Optional[str] = None
    role: str = "user"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    messages: list[MessageRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    # Aggregate stats
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tool_calls: int = 0
    successful_tool_calls: int = 0

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None) -> MessageRecord:
        """Add a message to the session history."""
        record = MessageRecord(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(record)
        self.updated_at = time.time()
        return record

    def add_tool_call(self, tool_name: str, params: dict) -> ToolCallRecord:
        """Record a tool invocation."""
        call = ToolCallRecord(
            call_id=str(uuid.uuid4()),
            tool_name=tool_name,
            params=params,
        )
        self.tool_calls.append(call)
        self.total_tool_calls += 1
        self.updated_at = time.time()
        return call

    def complete_tool_call(self, call_id: str, result: Any, error: Optional[str] = None) -> bool:
        """Mark a tool call as completed or failed."""
        for call in self.tool_calls:
            if call.call_id == call_id:
                call.finished_at = time.time()
                call.duration_ms = (call.finished_at - call.started_at) * 1000
                call.result = result
                call.error = error
                call.status = "failed" if error else "completed"
                if not error:
                    self.successful_tool_calls += 1
                self.updated_at = time.time()
                return True
        return False

    def to_dict(self) -> dict:
        """Serialize the session to a dict."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "user_id": self.user_id,
            "role": self.role,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content[:500] + ("..." if len(m.content) > 500 else ""),
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in self.messages
            ],
            "stats": {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tool_calls": self.total_tool_calls,
                "successful_tool_calls": self.successful_tool_calls,
            },
        }

    def export(self) -> dict:
        """Full export for backup/transfer."""
        return {
            "version": 1,
            "session_id": self.session_id,
            "state": self.state.value,
            "user_id": self.user_id,
            "role": self.role,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                    "tool_calls": m.tool_calls,
                }
                for m in self.messages
            ],
            "tool_calls": [
                {
                    "call_id": tc.call_id,
                    "tool_name": tc.tool_name,
                    "params": tc.params,
                    "result": tc.result,
                    "status": tc.status,
                    "started_at": tc.started_at,
                    "finished_at": tc.finished_at,
                    "duration_ms": tc.duration_ms,
                    "error": tc.error,
                }
                for tc in self.tool_calls
            ],
            "stats": {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tool_calls": self.total_tool_calls,
                "successful_tool_calls": self.successful_tool_calls,
            },
        }

    @classmethod
    def import_session(cls, data: dict) -> "ACPSession":
        """Reconstruct a session from exported data."""
        session = cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            state=SessionState(data.get("state", "active")),
            user_id=data.get("user_id"),
            role=data.get("role", "user"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
        )
        for msg_data in data.get("messages", []):
            record = MessageRecord(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=msg_data.get("timestamp", time.time()),
                metadata=msg_data.get("metadata", {}),
                tool_calls=msg_data.get("tool_calls", []),
            )
            session.messages.append(record)
        for tc_data in data.get("tool_calls", []):
            record = ToolCallRecord(
                call_id=tc_data["call_id"],
                tool_name=tc_data["tool_name"],
                params=tc_data.get("params", {}),
                result=tc_data.get("result"),
                status=tc_data.get("status", "completed"),
                started_at=tc_data.get("started_at", time.time()),
                finished_at=tc_data.get("finished_at"),
                duration_ms=tc_data.get("duration_ms", 0.0),
                error=tc_data.get("error"),
            )
            session.tool_calls.append(record)
        stats = data.get("stats", {})
        session.total_input_tokens = stats.get("total_input_tokens", 0)
        session.total_output_tokens = stats.get("total_output_tokens", 0)
        session.total_tool_calls = stats.get("total_tool_calls", 0)
        session.successful_tool_calls = stats.get("successful_tool_calls", 0)
        return session


class SessionManager:
    """
    Manages ACP session lifecycles with persistence and cleanup.
    """

    def __init__(self, data_dir: str | Path = _DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self._sessions_path = self.data_dir / _SESSIONS_DIR
        self._sessions_path.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ACPSession] = {}

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    async def create_session(
        self,
        user_id: Optional[str] = None,
        role: str = "user",
        metadata: Optional[dict] = None,
    ) -> ACPSession:
        """Create a new session and persist it."""
        session = ACPSession(
            user_id=user_id,
            role=role,
            metadata=metadata or {},
            state=SessionState.ACTIVE,
        )
        self._sessions[session.session_id] = session
        self._persist_session(session)
        logger.info("Session created: %s (user=%s)", session.session_id[:8], user_id)
        return session

    async def get_session(self, session_id: str) -> Optional[ACPSession]:
        """Get a session by ID, loading from disk if not in memory."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        session = self._load_session(session_id)
        if session:
            self._sessions[session_id] = session
        return session

    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        state: Optional[SessionState] = None,
    ) -> list[ACPSession]:
        """List sessions with optional filtering."""
        sessions = list(self._sessions.values())
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if state:
            sessions = [s for s in sessions if s.state == state]
        return sessions

    async def end_session(self, session_id: str) -> bool:
        """End a session."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.state = SessionState.ENDED
        session.updated_at = time.time()
        self._persist_session(session)
        logger.info("Session ended: %s", session_id[:8])
        return True

    async def pause_session(self, session_id: str) -> bool:
        """Pause an active session."""
        session = self._sessions.get(session_id)
        if session is None or session.state != SessionState.ACTIVE:
            return False
        session.state = SessionState.PAUSED
        session.updated_at = time.time()
        self._persist_session(session)
        return True

    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused session."""
        session = self._sessions.get(session_id)
        if session is None or session.state != SessionState.PAUSED:
            return False
        session.state = SessionState.ACTIVE
        session.updated_at = time.time()
        self._persist_session(session)
        return True

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its persisted data."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session_file = self._sessions_path / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
        return True

    # ------------------------------------------------------------------
    # Export/Import
    # ------------------------------------------------------------------

    async def export_session(self, session_id: str) -> Optional[dict]:
        """Export a session's full data."""
        session = await self.get_session(session_id)
        if session is None:
            return None
        return session.export()

    async def import_session(self, data: dict) -> ACPSession:
        """Import a session from exported data."""
        session = ACPSession.import_session(data)
        self._sessions[session.session_id] = session
        self._persist_session(session)
        logger.info("Session imported: %s", session.session_id[:8])
        return session

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup_ended(self, max_age_hours: int = 24) -> int:
        """Remove ended sessions older than max_age_hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        to_remove: list[str] = []
        for sid, session in self._sessions.items():
            if session.state == SessionState.ENDED and session.updated_at < cutoff:
                to_remove.append(sid)
        for sid in to_remove:
            await self.delete_session(sid)
        if to_remove:
            logger.info("Cleaned up %d ended sessions", len(to_remove))
        return len(to_remove)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_session(self, session: ACPSession) -> None:
        """Save a session to disk."""
        session_file = self._sessions_path / f"{session.session_id}.json"
        try:
            session_file.write_text(
                json.dumps(session.export(), indent=2, default=str),
                encoding="utf-8",
            )
        except OSError:
            logger.exception("Failed to persist session %s", session.session_id[:8])

    def _load_session(self, session_id: str) -> Optional[ACPSession]:
        """Load a session from disk."""
        session_file = self._sessions_path / f"{session_id}.json"
        if not session_file.exists():
            return None
        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
            return ACPSession.import_session(data)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to load session %s", session_id[:8])
            return None
