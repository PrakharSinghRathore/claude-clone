"""
Transcript Store — Conversation transcript management.

Manages conversation transcripts for sessions using JSONL (JSON Lines)
format for append-friendly storage. Supports search, summarization,
compaction, and pruning of transcript entries.

Usage::

    from atlas.sessions.transcript import TranscriptEntry, TranscriptStore

    store = TranscriptStore(data_dir="./transcripts")
    entry = TranscriptEntry(role="user", content="Hello, world!")
    store.append("session-123", entry)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Transcript Entry
# ──────────────────────────────────────────────────────────────────────────────

class TranscriptRole(Enum):
    """Role of the transcript entry author."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass
class TranscriptEntry:
    """
    A single entry in a conversation transcript.

    Attributes
    ----------
    role:
        The role of the author (user, assistant, system, tool, etc.).
    content:
        The text content of the entry.
    timestamp:
        ISO 8601 timestamp of when the entry was created.
    tokens:
        Estimated token count for this entry.
    model:
        The model used to generate this entry (for assistant entries).
    tool_calls:
        List of tool call records (for assistant entries).
    metadata:
        Arbitrary key-value metadata.
    entry_id:
        Unique identifier for this entry.
    """

    role: str = TranscriptRole.USER.value
    content: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tokens: int = 0
    model: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def role_enum(self) -> TranscriptRole:
        """Return the role as a ``TranscriptRole`` enum member."""
        try:
            return TranscriptRole(self.role)
        except ValueError:
            return TranscriptRole.USER

    @property
    def timestamp_dt(self) -> datetime:
        """Parse timestamp to a timezone-aware datetime."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)

    @property
    def word_count(self) -> int:
        """Approximate word count of the content."""
        return len(self.content.split()) if self.content else 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entry to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TranscriptEntry:
        """Deserialize an entry from a dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def user(cls, content: str, **kwargs: Any) -> TranscriptEntry:
        """Create a user message entry."""
        return cls(role=TranscriptRole.USER.value, content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str, model: Optional[str] = None, **kwargs: Any) -> TranscriptEntry:
        """Create an assistant message entry."""
        return cls(
            role=TranscriptRole.ASSISTANT.value,
            content=content,
            model=model,
            **kwargs,
        )

    @classmethod
    def system(cls, content: str, **kwargs: Any) -> TranscriptEntry:
        """Create a system message entry."""
        return cls(role=TranscriptRole.SYSTEM.value, content=content, **kwargs)

    @classmethod
    def tool_call(cls, name: str, arguments: Dict[str, Any], **kwargs: Any) -> TranscriptEntry:
        """Create a tool call entry."""
        tool_calls = kwargs.pop("tool_calls", [])
        tool_calls.append({
            "name": name,
            "arguments": arguments,
            "id": kwargs.pop("call_id", uuid.uuid4().hex[:8]),
        })
        return cls(
            role=TranscriptRole.TOOL_CALL.value,
            content=f"Call tool: {name}",
            tool_calls=tool_calls,
            **kwargs,
        )

    @classmethod
    def tool_result(cls, name: str, result: str, success: bool = True, **kwargs: Any) -> TranscriptEntry:
        """Create a tool result entry."""
        return cls(
            role=TranscriptRole.TOOL_RESULT.value,
            content=result,
            metadata={"tool_name": name, "success": success},
            **kwargs,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Compaction & Summarization
# ──────────────────────────────────────────────────────────────────────────────

class TranscriptCompactor:
    """
    Compacts transcripts by summarizing old entries.

    Replaces old entries with a single summary entry, preserving the most
    recent entries intact. This reduces token usage while maintaining
    conversation context.
    """

    def __init__(
        self,
        preserve_recent: int = 10,
        max_tokens_per_summary: int = 500,
    ) -> None:
        self.preserve_recent = preserve_recent
        self.max_tokens_per_summary = max_tokens_per_summary

    def compact(self, entries: List[TranscriptEntry]) -> List[TranscriptEntry]:
        """
        Compact a transcript by summarizing old entries.

        Parameters
        ----------
        entries:
            The full transcript entries list.

        Returns
        -------
        list[TranscriptEntry]
            Compacted transcript with old entries replaced by a summary.
        """
        if len(entries) <= self.preserve_recent + 1:
            return entries

        old_entries = entries[: -self.preserve_recent]
        recent_entries = entries[-self.preserve_recent:]

        # Generate summary from old entries
        summary = self._generate_summary(old_entries)
        summary_tokens = len(summary.split()) // 4  # rough token estimate

        summary_entry = TranscriptEntry(
            role=TranscriptRole.SYSTEM.value,
            content=f"[Previous conversation summary ({len(old_entries)} messages): {summary}]",
            metadata={
                "compacted": True,
                "original_count": len(old_entries),
                "summary_tokens": summary_tokens,
                "compacted_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        return [summary_entry] + recent_entries

    def _generate_summary(self, entries: List[TranscriptEntry]) -> str:
        """
        Generate a text summary of transcript entries.

        Uses extractive summarization: collects the first sentence of
        each user message and the first sentence of each assistant response,
        then truncates to max_tokens_per_summary.

        Parameters
        ----------
        entries:
            The entries to summarize.

        Returns
        -------
        str
            A text summary.
        """
        key_points: list[str] = []

        for entry in entries:
            if entry.role == TranscriptRole.USER.value and entry.content:
                first_line = entry.content.split(".")[0].strip()
                if first_line:
                    key_points.append(f"User asked about: {first_line}")
            elif entry.role == TranscriptRole.ASSISTANT.value and entry.content:
                first_line = entry.content.split(".")[0].strip()
                if first_line:
                    key_points.append(f"Assistant: {first_line}")
            elif entry.role == TranscriptRole.TOOL_CALL.value:
                for tc in entry.tool_calls:
                    key_points.append(f"Tool called: {tc.get('name', 'unknown')}")
            elif entry.role == TranscriptRole.TOOL_RESULT.value:
                success = entry.metadata.get("success", True)
                tool_name = entry.metadata.get("tool_name", "unknown")
                key_points.append(f"Tool {tool_name} result: {'success' if success else 'failed'}")

        # Truncate to fit token budget
        summary = "; ".join(key_points)
        word_budget = self.max_tokens_per_summary * 4  # rough words per token
        words = summary.split()
        if len(words) > word_budget:
            summary = " ".join(words[:word_budget]) + "..."

        return summary


# ──────────────────────────────────────────────────────────────────────────────
# Transcript Store
# ──────────────────────────────────────────────────────────────────────────────

class TranscriptStore:
    """
    Manages conversation transcripts with JSONL persistence.

    Each session's transcript is stored as a JSONL file (one JSON object
    per line), which is efficient for append operations and resistant to
    corruption on crash.

    Parameters
    ----------
    data_dir:
        Directory for storing transcript files.
    compactor:
        Optional ``TranscriptCompactor`` for automatic compaction.
        If ``None``, a default compactor is created.
    max_file_size_mb:
        Maximum transcript file size before auto-rotation (in MB).

    Example
    -------
    >>> store = TranscriptStore(data_dir="./transcripts")
    >>> entry = TranscriptEntry.user("Hello!")
    >>> store.append("sess-1", entry)
    >>> transcript = store.get_transcript("sess-1")
    """

    def __init__(
        self,
        data_dir: str | Path = Path.home() / ".claude_clone" / "atlas_transcripts",
        compactor: Optional[TranscriptCompactor] = None,
        max_file_size_mb: int = 50,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._compactor = compactor or TranscriptCompactor()
        self._max_file_size_bytes = max_file_size_mb * 1024 * 1024
        # In-memory cache: session_id -> list of entries
        self._cache: Dict[str, List[TranscriptEntry]] = {}

    def _transcript_path(self, session_id: str) -> Path:
        """Get the JSONL file path for a session's transcript."""
        # Sanitize session_id to prevent path traversal
        safe_id = re.sub(r"[^\w\-]", "_", session_id)
        return self.data_dir / f"{safe_id}.jsonl"

    # ── Read Operations ────────────────────────────────────────────────

    def append(self, session_id: str, entry: TranscriptEntry) -> None:
        """
        Append an entry to a session's transcript.

        Writes to the JSONL file immediately and updates the in-memory cache.

        Parameters
        ----------
        session_id:
            The session identifier.
        entry:
            The transcript entry to append.
        """
        if session_id not in self._cache:
            self._load_transcript(session_id)

        self._cache[session_id].append(entry)

        # Write to file
        path = self._transcript_path(session_id)
        line = json.dumps(entry.to_dict(), default=str, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        # Check for auto-rotation
        if path.exists() and path.stat().st_size > self._max_file_size_bytes:
            logger.info(
                "Transcript for session %s exceeded %d MB, rotating",
                session_id,
                self._max_file_size_bytes // (1024 * 1024),
            )
            self._rotate_transcript(session_id)

    def get_transcript(self, session_id: str) -> List[TranscriptEntry]:
        """
        Get the full transcript for a session.

        Parameters
        ----------
        session_id:
            The session identifier.

        Returns
        -------
        list[TranscriptEntry]
            All entries in chronological order.
        """
        if session_id not in self._cache:
            self._load_transcript(session_id)
        return list(self._cache.get(session_id, []))

    def get_recent(
        self,
        session_id: str,
        limit: int = 20,
    ) -> List[TranscriptEntry]:
        """
        Get the most recent entries from a session's transcript.

        Parameters
        ----------
        session_id:
            The session identifier.
        limit:
            Maximum number of entries to return.

        Returns
        -------
        list[TranscriptEntry]
            The most recent entries, newest last.
        """
        transcript = self.get_transcript(session_id)
        return transcript[-limit:] if len(transcript) > limit else transcript

    def get_entry(
        self,
        session_id: str,
        entry_id: str,
    ) -> Optional[TranscriptEntry]:
        """
        Get a specific entry by its ID.

        Parameters
        ----------
        session_id:
            The session identifier.
        entry_id:
            The entry identifier.

        Returns
        -------
        TranscriptEntry or None
        """
        transcript = self.get_transcript(session_id)
        for entry in transcript:
            if entry.entry_id == entry_id:
                return entry
        return None

    def search(
        self,
        session_id: str,
        query: str,
        *,
        role: Optional[str] = None,
        limit: int = 50,
        case_sensitive: bool = False,
    ) -> List[TranscriptEntry]:
        """
        Search a session's transcript for entries matching a query.

        Parameters
        ----------
        session_id:
            The session identifier.
        query:
            The search string (supports simple substring matching).
        role:
            If provided, filter to entries with this role.
        limit:
            Maximum number of results to return.
        case_sensitive:
            If ``True``, perform case-sensitive matching.

        Returns
        -------
        list[TranscriptEntry]
            Matching entries in chronological order.
        """
        transcript = self.get_transcript(session_id)
        results: List[TranscriptEntry] = []

        search_query = query if case_sensitive else query.lower()

        for entry in transcript:
            if role and entry.role != role:
                continue

            content = entry.content if case_sensitive else entry.content.lower()
            if search_query in content:
                results.append(entry)
                if len(results) >= limit:
                    break

        return results

    def search_all(
        self,
        query: str,
        *,
        role: Optional[str] = None,
        limit: int = 50,
        case_sensitive: bool = False,
    ) -> Dict[str, List[TranscriptEntry]]:
        """
        Search across all sessions' transcripts.

        Parameters
        ----------
        query:
            The search string.
        role:
            If provided, filter to entries with this role.
        limit:
            Maximum results per session.
        case_sensitive:
            If ``True``, perform case-sensitive matching.

        Returns
        -------
        dict[str, list[TranscriptEntry]]
            Mapping of session ID to matching entries.
        """
        results: Dict[str, List[TranscriptEntry]] = {}
        session_ids = self._discover_sessions()

        for sid in session_ids:
            matches = self.search(
                sid, query,
                role=role,
                limit=limit,
                case_sensitive=case_sensitive,
            )
            if matches:
                results[sid] = matches

        return results

    # ── Write Operations ───────────────────────────────────────────────

    def prune(
        self,
        session_id: str,
        max_entries: int = 100,
    ) -> int:
        """
        Prune old entries from a session's transcript.

        Keeps the most recent ``max_entries`` entries and removes older ones.

        Parameters
        ----------
        session_id:
            The session identifier.
        max_entries:
            Maximum number of entries to keep.

        Returns
        -------
        int
            Number of entries removed.
        """
        transcript = self.get_transcript(session_id)
        if len(transcript) <= max_entries:
            return 0

        removed_count = len(transcript) - max_entries
        kept = transcript[-max_entries:]

        # Rewrite the file with kept entries
        self._cache[session_id] = kept
        self._rewrite_transcript(session_id, kept)

        logger.info(
            "Pruned %d entries from session %s (kept %d)",
            removed_count,
            session_id,
            max_entries,
        )
        return removed_count

    def summarize(self, session_id: str) -> str:
        """
        Generate a summary of a session's transcript.

        Uses the compactor to create an extractive summary of all entries.

        Parameters
        ----------
        session_id:
            The session identifier.

        Returns
        -------
        str
            A text summary of the conversation.
        """
        transcript = self.get_transcript(session_id)
        if not transcript:
            return "(empty transcript)"

        return self._compactor._generate_summary(transcript)

    def compact(self, session_id: str) -> List[TranscriptEntry]:
        """
        Compact a session's transcript using the compactor.

        Parameters
        ----------
        session_id:
            The session identifier.

        Returns
        -------
        list[TranscriptEntry]
            The compacted transcript.
        """
        transcript = self.get_transcript(session_id)
        compacted = self._compactor.compact(transcript)

        # Rewrite the transcript with compacted entries
        self._cache[session_id] = compacted
        self._rewrite_transcript(session_id, compacted)

        original_count = len(transcript)
        new_count = len(compacted)
        logger.info(
            "Compacted session %s: %d entries -> %d entries",
            session_id,
            original_count,
            new_count,
        )
        return compacted

    def delete(self, session_id: str) -> bool:
        """
        Delete a session's transcript.

        Parameters
        ----------
        session_id:
            The session identifier.

        Returns
        -------
        bool
            ``True`` if the transcript was deleted, ``False`` if not found.
        """
        self._cache.pop(session_id, None)
        path = self._transcript_path(session_id)
        if path.exists():
            path.unlink()
            logger.info("Deleted transcript for session %s", session_id)
            return True
        return False

    # ── Statistics ─────────────────────────────────────────────────────

    def stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session's transcript.

        Parameters
        ----------
        session_id:
            The session identifier.

        Returns
        -------
        dict
            Keys: ``total_entries``, ``user_entries``, ``assistant_entries``,
            ``system_entries``, ``tool_entries``, ``total_tokens``,
            ``total_words``, ``file_size_bytes``, ``first_entry``, ``last_entry``.
        """
        transcript = self.get_transcript(session_id)
        if not transcript:
            return {"total_entries": 0}

        role_counts: Dict[str, int] = {}
        total_tokens = 0
        total_words = 0

        for entry in transcript:
            role_counts[entry.role] = role_counts.get(entry.role, 0) + 1
            total_tokens += entry.tokens
            total_words += entry.word_count

        path = self._transcript_path(session_id)
        file_size = path.stat().st_size if path.exists() else 0

        return {
            "total_entries": len(transcript),
            "user_entries": role_counts.get("user", 0),
            "assistant_entries": role_counts.get("assistant", 0),
            "system_entries": role_counts.get("system", 0),
            "tool_entries": role_counts.get("tool_call", 0) + role_counts.get("tool_result", 0),
            "total_tokens": total_tokens,
            "total_words": total_words,
            "file_size_bytes": file_size,
            "first_entry": transcript[0].timestamp if transcript else None,
            "last_entry": transcript[-1].timestamp if transcript else None,
            "role_distribution": role_counts,
        }

    def global_stats(self) -> Dict[str, Any]:
        """
        Get statistics across all session transcripts.

        Returns
        -------
        dict
            Keys: ``total_sessions``, ``total_entries``, ``total_tokens``,
            ``total_size_bytes``, ``largest_session``.
        """
        session_ids = self._discover_sessions()
        total_entries = 0
        total_tokens = 0
        total_size = 0
        largest_session = ""
        largest_count = 0

        for sid in session_ids:
            s = self.stats(sid)
            count = s.get("total_entries", 0)
            total_entries += count
            total_tokens += s.get("total_tokens", 0)
            total_size += s.get("file_size_bytes", 0)
            if count > largest_count:
                largest_count = count
                largest_session = sid

        return {
            "total_sessions": len(session_ids),
            "total_entries": total_entries,
            "total_tokens": total_tokens,
            "total_size_bytes": total_size,
            "largest_session": largest_session,
            "largest_session_entries": largest_count,
        }

    # ── Internal ───────────────────────────────────────────────────────

    def _load_transcript(self, session_id: str) -> None:
        """Load a transcript from the JSONL file into the cache."""
        path = self._transcript_path(session_id)
        entries: List[TranscriptEntry] = []

        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                entries.append(TranscriptEntry.from_dict(data))
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Skipping malformed JSONL line in %s",
                                    path,
                                )
            except Exception:
                logger.exception("Failed to read transcript from %s", path)

        self._cache[session_id] = entries

    def _rewrite_transcript(
        self,
        session_id: str,
        entries: List[TranscriptEntry],
    ) -> None:
        """Rewrite a transcript file with the given entries."""
        path = self._transcript_path(session_id)
        tmp_path = path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    line = json.dumps(entry.to_dict(), default=str, ensure_ascii=False)
                    f.write(line + "\n")
            tmp_path.replace(path)
        except Exception:
            logger.exception("Failed to rewrite transcript for session %s", session_id)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _rotate_transcript(self, session_id: str) -> None:
        """Rotate a transcript file: compact and rename old file."""
        # Compact
        self.compact(session_id)

        # Archive old entries if there's a pre-existing backup
        path = self._transcript_path(session_id)
        if path.exists():
            archive_path = path.with_suffix(
                f".{int(time.time())}.jsonl.bak"
            )
            shutil_copy = False
            try:
                import shutil
                shutil.copy2(str(path), str(archive_path))
                shutil_copy = True
            except Exception:
                logger.warning("Failed to archive transcript for session %s", session_id)

            if shutil_copy:
                # Keep only the last 3 archives
                archives = sorted(
                    path.parent.glob(f"{path.stem}.*.jsonl.bak"),
                    key=lambda p: p.stat().st_mtime,
                )
                for old_archive in archives[:-3]:
                    try:
                        old_archive.unlink()
                    except Exception:
                        pass

    def _discover_sessions(self) -> List[str]:
        """Discover all session IDs that have transcript files."""
        sessions: List[str] = []
        for f in self.data_dir.glob("*.jsonl"):
            # Extract session_id from filename
            name = f.stem
            if not name.endswith(".tmp"):
                sessions.append(name)
        return sessions
