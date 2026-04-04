"""
Conversation Memory with RAG (Retrieval-Augmented Generation) module.

Provides persistent conversation storage with semantic search, auto-summarization,
session management, context window optimization, tagging, and JSON export/import.

All database operations are async via ``sqlite3`` + ``asyncio.run_in_executor`` so
that no external async database drivers (e.g. aiosqlite) are required.

Usage::

    mem = ConversationMemory()
    await mem.initialize()
    session_id = await mem.create_session("project-alpha")
    await mem.add("user", "How do I deploy to AWS?", tags=["aws", "devops"])
    results = await mem.search("cloud deployment")
    context = await mem.get_context_for_prompt("deploy to ECS")
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sqlite3
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "~/.claude_clone/memory.db"

# Comprehensive English stop-words list for TF-IDF token filtering.
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren", "arent", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "couldn", "couldnt", "d", "did", "didn", "didnt", "do", "does",
    "doesn", "doesnt", "doing", "don", "dont", "down", "during", "each",
    "few", "for", "from", "further", "get", "got", "had", "hadn", "hadnt",
    "has", "hasn", "hasnt", "have", "haven", "havent", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "id", "if", "in", "into", "is", "isn", "isnt", "it", "its",
    "itself", "just", "ll", "let", "m", "ma", "me", "might", "mightn",
    "more", "most", "must", "mustn", "my", "myself", "need", "no", "nor",
    "not", "now", "o", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same",
    "shan", "she", "should", "shouldn", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "ve", "was", "wasn", "we", "were", "weren",
    "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won", "would", "wouldn", "you", "youd", "your",
    "yours", "yourself", "yourselves",
}

# Regex for tokenization: split on non-alphanumeric characters.
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Vocabulary size for TF-IDF vectors.  In practice the vocabulary is built
# dynamically from stored content; this cap prevents unbounded growth.
_MAX_VOCAB_SIZE = 10_000

# Auto-summarization threshold: entries older than this many days may be
# candidates for summarization if the session exceeds the summary threshold.
_AUTO_SUMMARIZE_AGE_DAYS = 7

# Number of entries in a session that triggers auto-summarization of old
# entries within that session.
_AUTO_SUMMARIZE_ENTRY_THRESHOLD = 50

# Maximum length of an auto-generated summary (characters).
_MAX_SUMMARY_LENGTH = 600


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory record stored in the database."""

    id: str
    session_id: str
    role: str  # user | assistant | system | tool
    content: str
    timestamp: str  # ISO-8601
    tags: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    embedding_vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _generate_id() -> str:
    """Return a short random UUID suitable for primary keys."""
    return uuid.uuid4().hex[:16]


def _now_iso() -> str:
    """Current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    """
    Lowercase, split on non-alphanumeric boundaries, drop stop-words and
    single-character tokens.
    """
    raw = _TOKEN_RE.findall(text.lower())
    return [tok for tok in raw if tok not in STOP_WORDS and len(tok) > 1]


def _token_count(text: str) -> int:
    """
    Approximate the number of tokens that *text* would consume.

    A simple heuristic: ``len(text) // 4``, which works well for
    English text and common sub-word tokenizers (BPE / SentencePiece).
    """
    return max(1, len(text) // 4)


# ──────────────────────────────────────────────────────────────────────────────
# SQLite row helpers
# ──────────────────────────────────────────────────────────────────────────────

def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    """Convert a database row to a :class:`MemoryEntry`."""
    tags = json.loads(row["tags"]) if row["tags"] else []
    vector = json.loads(row["embedding_vector"]) if row["embedding_vector"] else None
    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    return MemoryEntry(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        timestamp=row["timestamp"],
        tags=tags,
        summary=row["summary"],
        embedding_vector=vector,
        metadata=metadata,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ConversationMemory
# ──────────────────────────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Persistent conversation memory with semantic search (TF-IDF + cosine
    similarity), session management, auto-summarization, tagging, and
    import/export.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  ``~`` is expanded.  The parent
        directory is created automatically if it does not exist.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self._conn: Optional[sqlite3.Connection] = None
        # In-memory IDF cache: token -> idf value
        self._idf_cache: Dict[str, float] = {}
        # Vocabulary (sorted list of unique tokens across all entries)
        self._vocabulary: List[str] = []
        self._vocab_index: Dict[str, int] = {}

    # ── Initialisation / teardown ─────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Create the database schema (if needed) and pre-load the IDF cache
        and vocabulary.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(self._connect)
        await self._run_sync(self._create_tables)
        await self._rebuild_idf_and_vocab()

    async def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn is not None:
            await self._run_sync(self._conn.close)
            self._conn = None

    def _connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _create_tables(self) -> None:
        assert self._conn is not None
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                metadata        TEXT DEFAULT '{}',
                summary         TEXT DEFAULT NULL
            );

            CREATE TABLE IF NOT EXISTS entries (
                id              TEXT PRIMARY KEY,
                session_id      TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                tags            TEXT DEFAULT '[]',
                summary         TEXT DEFAULT NULL,
                embedding_vector TEXT DEFAULT NULL,
                metadata        TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_entries_session
                ON entries(session_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_entries_role
                ON entries(role);

            CREATE INDEX IF NOT EXISTS idx_entries_timestamp
                ON entries(timestamp);

            CREATE TABLE IF NOT EXISTS idf_cache (
                token           TEXT PRIMARY KEY,
                idf             REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS vocabulary (
                token           TEXT PRIMARY KEY,
                idx             INTEGER NOT NULL
            );
        """)
        self._conn.commit()

    # ── Async wrapper ─────────────────────────────────────────────────────

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous function in a thread-pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(
                "ConversationMemory is not initialized. "
                "Call `await mem.initialize()` first."
            )
        return self._conn

    # ── Session management ────────────────────────────────────────────────

    async def create_session(self, name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Create a new named session and return its id.

        Parameters
        ----------
        name:
            Human-readable label for the session.
        metadata:
            Optional arbitrary key-value metadata attached to the session.

        Returns
        -------
        str
            The newly-created session id.
        """
        session_id = _generate_id()
        now = _now_iso()

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO sessions (id, name, created_at, updated_at, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, name, now, now, json.dumps(metadata or {})),
            )
            conn.commit()

        await self._run_sync(_do)
        return session_id

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        Return a list of all sessions with summary metadata.

        Each entry is a dict with keys: ``id``, ``name``, ``created_at``,
        ``updated_at``, ``entry_count``, ``metadata``, ``summary``.
        """
        def _do() -> List[Dict[str, Any]]:
            conn = self._ensure_conn()
            rows = conn.execute("""
                SELECT s.id, s.name, s.created_at, s.updated_at, s.metadata, s.summary,
                       COUNT(e.id) AS entry_count
                FROM sessions s
                LEFT JOIN entries e ON e.session_id = s.id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
            """).fetchall()
            result = []
            for r in rows:
                result.append({
                    "id": r["id"],
                    "name": r["name"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "entry_count": r["entry_count"],
                    "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                    "summary": r["summary"],
                })
            return result

        return await self._run_sync(_do)

    async def delete_session(self, session_id: str) -> None:
        """
        Delete a session and all of its entries (cascade).

        Raises ``KeyError`` if the session does not exist.
        """
        def _do() -> None:
            conn = self._ensure_conn()
            cur = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
            if cur.fetchone() is None:
                raise KeyError(f"Session '{session_id}' not found")
            conn.execute("DELETE FROM entries WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()

        await self._run_sync(_do)

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a single session, or ``None`` if not found."""
        def _do() -> Optional[Dict[str, Any]]:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT id, name, created_at, updated_at, metadata, summary "
                "FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row["id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "summary": row["summary"],
            }

        return await self._run_sync(_do)

    # ── Adding entries ────────────────────────────────────────────────────

    async def add(
        self,
        role: str,
        content: str,
        session_id: str = "default",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a new memory entry and return its id.

        Parameters
        ----------
        role:
            One of ``"user"``, ``"assistant"``, ``"system"``, ``"tool"``.
        content:
            The text body of the entry.
        session_id:
            The session to attach the entry to.  If ``"default"`` and no
            default session exists, one will be created automatically.
        tags:
            Optional list of tags for categorisation and filtering.
        metadata:
            Optional arbitrary key-value metadata.

        Returns
        -------
        str
            The new entry id.
        """
        if role not in ("user", "assistant", "system", "tool"):
            raise ValueError(f"Invalid role: {role!r}. Must be one of user/assistant/system/tool.")

        entry_id = _generate_id()
        now = _now_iso()
        tags_json = json.dumps(tags or [])
        meta_json = json.dumps(metadata or {})

        # Compute TF-IDF embedding.
        tokens = _tokenize(content)
        embedding = self._compute_tfidf_tokens(tokens)

        def _do() -> None:
            conn = self._ensure_conn()

            # Ensure default session exists.
            if session_id == "default":
                cur = conn.execute("SELECT id FROM sessions WHERE id = 'default'")
                if cur.fetchone() is None:
                    conn.execute(
                        "INSERT INTO sessions (id, name, created_at, updated_at, metadata) "
                        "VALUES ('default', 'default', ?, ?, '{}')",
                        (now, now),
                    )
                    conn.commit()

            conn.execute(
                "INSERT INTO entries "
                "(id, session_id, role, content, timestamp, tags, embedding_vector, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (entry_id, session_id, role, content, now, tags_json,
                 json.dumps(embedding) if embedding else None, meta_json),
            )

            # Bump session updated_at.
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
            conn.commit()

            # Persist vocabulary entries.
            for tok in set(tokens):
                conn.execute(
                    "INSERT OR IGNORE INTO vocabulary (token, idx) VALUES (?, "
                    "(SELECT COALESCE(MAX(idx), -1) + 1 FROM vocabulary))",
                    (tok,),
                )
            conn.commit()

        await self._run_sync(_do)

        # Refresh IDF cache and vocabulary in the background.
        await self._rebuild_idf_and_vocab()

        return entry_id

    # ── Retrieval ─────────────────────────────────────────────────────────

    async def get_session(
        self, session_id: str, limit: int = 50, offset: int = 0,
    ) -> List[MemoryEntry]:
        """
        Retrieve entries for a session ordered by timestamp.

        Parameters
        ----------
        session_id:
            The session to query.
        limit:
            Maximum entries to return (default 50).
        offset:
            Skip the first *offset* entries.

        Returns
        -------
        list[MemoryEntry]
        """
        def _do() -> List[MemoryEntry]:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT * FROM entries WHERE session_id = ? "
                "ORDER BY timestamp ASC LIMIT ? OFFSET ?",
                (session_id, limit, offset),
            ).fetchall()
            return [_row_to_entry(r) for r in rows]

        return await self._run_sync(_do)

    async def search(
        self,
        query: str,
        limit: int = 10,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[MemoryEntry]:
        """
        Semantic search using TF-IDF cosine similarity.

        Results are ranked by relevance and optionally filtered by session
        and/or tags.

        Parameters
        ----------
        query:
            The search query.
        limit:
            Maximum results.
        session_id:
            Restrict to a single session.
        tags:
            Only return entries that have **all** of the given tags.

        Returns
        -------
        list[MemoryEntry]
            Entries sorted by descending relevance.
        """
        query_tokens = _tokenize(query)
        query_vec = self._compute_tfidf_tokens(query_tokens)

        if query_vec is None or all(v == 0.0 for v in query_vec):
            # Degenerate query — fall back to keyword substring match.
            return await self._keyword_search(query, limit, session_id, tags)

        def _do() -> List[MemoryEntry]:
            conn = self._ensure_conn()
            clauses: List[str] = []
            params: List[Any] = []

            if session_id is not None:
                clauses.append("session_id = ?")
                params.append(session_id)

            if tags:
                for tag in tags:
                    clauses.append("tags LIKE ?")
                    params.append(f'%"{tag}"%')

            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

            rows = conn.execute(
                f"SELECT * FROM entries{where} ORDER BY timestamp DESC",
                params,
            ).fetchall()

            scored: List[Tuple[float, MemoryEntry]] = []
            for row in rows:
                entry = _row_to_entry(row)
                # Recompute embedding on-the-fly from content using the
                # current vocabulary/IDF so that vectors are always
                # consistent regardless of when the entry was written.
                entry_tokens = _tokenize(entry.content)
                entry_vec = self._compute_tfidf_tokens(entry_tokens)
                if entry_vec is None:
                    continue
                sim = self._cosine_similarity(query_vec, entry_vec)
                if sim > 0.0:
                    scored.append((sim, entry))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [entry for _, entry in scored[:limit]]

        return await self._run_sync(_do)

    async def _keyword_search(
        self,
        query: str,
        limit: int,
        session_id: Optional[str],
        tags: Optional[List[str]],
    ) -> List[MemoryEntry]:
        """Fallback substring-based search when TF-IDF cannot produce a vector."""
        def _do() -> List[MemoryEntry]:
            conn = self._ensure_conn()
            clauses: List[str] = ["content LIKE ?"]
            params: List[Any] = [f"%{query}%"]

            if session_id is not None:
                clauses.append("session_id = ?")
                params.append(session_id)

            if tags:
                for tag in tags:
                    clauses.append("tags LIKE ?")
                    params.append(f'%"{tag}"%')

            where = " WHERE " + " AND ".join(clauses)
            rows = conn.execute(
                f"SELECT * FROM entries{where} ORDER BY timestamp DESC LIMIT ?",
                params + [limit],
            ).fetchall()
            return [_row_to_entry(r) for r in rows]

        return await self._run_sync(_do)

    # ── Context window optimisation ───────────────────────────────────────

    async def get_context_for_prompt(
        self,
        query: str,
        max_tokens: int = 4000,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Build a context string from the most relevant memories, fitting
        within *max_tokens* tokens.

        Strategy:
        1. Search for semantically relevant entries (top 20).
        2. If a *session_id* is given, also pull the most recent N entries.
        3. Merge, deduplicate, and greedily pack until the token budget is
           exhausted (most relevant first).

        Parameters
        ----------
        query:
            The prompt for which context is being gathered.
        max_tokens:
            Approximate token budget for the context string.

        Returns
        -------
        str
            A formatted context block (may be empty).
        """
        relevant: List[MemoryEntry] = await self.search(
            query, limit=20, session_id=session_id
        )

        # Also fetch the most recent session entries.
        if session_id is not None:
            recent = await self.get_session(session_id, limit=20)
            seen_ids = {e.id for e in relevant}
            for entry in recent:
                if entry.id not in seen_ids:
                    relevant.append(entry)

        # Sort by relevance score (recency-weighted for same-score entries).
        # Recompute each entry's embedding with the current vocabulary so
        # vectors are always consistent.
        query_vec = self._compute_tfidf_tokens(_tokenize(query))
        if query_vec is not None:
            def _relevance(entry: MemoryEntry) -> float:
                vec = self._compute_tfidf_tokens(_tokenize(entry.content))
                if vec is None:
                    return 0.0
                return self._cosine_similarity(query_vec, vec)

            relevant.sort(key=lambda e: -_relevance(e))

        # Greedily pack entries.
        parts: List[str] = []
        used_tokens = 0

        for entry in relevant:
            display = entry.summary or entry.content
            if entry.role == "user":
                label = "User"
            elif entry.role == "assistant":
                label = "Assistant"
            elif entry.role == "system":
                label = "System"
            else:
                label = "Tool"
            block = f"[{label}] {display}"
            block_tokens = _token_count(block)

            if used_tokens + block_tokens > max_tokens:
                continue

            parts.append(block)
            used_tokens += block_tokens

        if not parts:
            return ""

        header = (
            "<memory_context>\n"
            "The following relevant past conversation fragments are provided "
            "for context:\n"
        )
        footer = "\n</memory_context>"
        return header + "\n---\n".join(parts) + footer

    # ── Summarization ─────────────────────────────────────────────────────

    async def summarize_session(self, session_id: str) -> str:
        """
        Produce a concise summary of all entries in a session.

        The summarizer groups entries into time windows, extracts the user's
        intent and the assistant's response for each window, and produces a
        readable paragraph.  The result is also stored on the session.

        Returns
        -------
        str
            The generated summary.
        """
        entries = await self.get_session(session_id, limit=1000)
        if not entries:
            return ""

        # Group into chunks of ~10 exchanges (user+assistant pairs).
        chunks: List[List[MemoryEntry]] = []
        chunk: List[MemoryEntry] = []
        for entry in entries:
            chunk.append(entry)
            if len(chunk) >= 20:
                chunks.append(chunk)
                chunk = []
        if chunk:
            chunks.append(chunk)

        summary_parts: List[str] = []
        for i, chunk_entries in enumerate(chunks):
            user_parts: List[str] = []
            assistant_parts: List[str] = []
            for entry in chunk_entries:
                text = entry.summary or entry.content
                if entry.role == "user":
                    user_parts.append(text[:200])
                elif entry.role == "assistant":
                    assistant_parts.append(text[:200])

            user_summary = "; ".join(user_parts[:5]) if user_parts else "(no user input)"
            asst_summary = "; ".join(assistant_parts[:5]) if assistant_parts else "(no response)"

            ts = chunk_entries[0].timestamp[:16]
            summary_parts.append(
                f"[{ts}] User asked about {user_summary}. "
                f"Assistant provided {asst_summary}."
            )

        full_summary = " ".join(summary_parts)
        if len(full_summary) > _MAX_SUMMARY_LENGTH:
            full_summary = full_summary[:_MAX_SUMMARY_LENGTH] + "..."

        # Persist the session summary.
        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = ? WHERE id = ?",
                (full_summary, _now_iso(), session_id),
            )
            conn.commit()

        await self._run_sync(_do)
        return full_summary

    async def _summarize_entries_batch(self, entries: List[MemoryEntry]) -> str:
        """
        Summarize a batch of entries into a single concise string.

        Uses extractive summarization: pick the most representative sentences
        from user and assistant content.
        """
        user_sentences: List[str] = []
        assistant_sentences: List[str] = []

        for entry in entries:
            text = (entry.summary or entry.content).strip()
            if not text:
                continue
            # Split into sentences (rough).
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 10:
                    continue
                if entry.role == "user":
                    user_sentences.append(sent)
                elif entry.role == "assistant":
                    assistant_sentences.append(sent)

        parts: List[str] = []
        # Take up to 3 user sentences and 3 assistant sentences.
        for sent in user_sentences[:3]:
            parts.append(f"User: {sent}")
        for sent in assistant_sentences[:3]:
            parts.append(f"Assistant: {sent}")

        summary = " ".join(parts)
        if len(summary) > _MAX_SUMMARY_LENGTH:
            summary = summary[:_MAX_SUMMARY_LENGTH] + "..."
        return summary if summary else "Summary not available."

    async def auto_summarize(self) -> int:
        """
        Scan all sessions for old entries that should be summarized and
        replace their content with a summary to reclaim context space.

        Entries are candidates for summarization when:
        * They belong to a session with more than
          ``_AUTO_SUMMARIZE_ENTRY_THRESHOLD`` entries.
        * The entry is older than ``_AUTO_SUMMARIZE_AGE_DAYS`` days.
        * The entry does not already have a summary.

        Returns
        -------
        int
            The number of entries that were summarized.
        """
        threshold = _AUTO_SUMMARIZE_AGE_DAYS
        now = datetime.now(timezone.utc)

        def _find_candidates() -> List[Dict[str, Any]]:
            """Find sessions exceeding the entry threshold."""
            conn = self._ensure_conn()
            rows = conn.execute("""
                SELECT session_id, COUNT(*) AS cnt
                FROM entries
                WHERE summary IS NULL
                GROUP BY session_id
                HAVING cnt > ?
            """, (_AUTO_SUMMARIZE_ENTRY_THRESHOLD,)).fetchall()
            return [dict(r) for r in rows]

        big_sessions = await self._run_sync(_find_candidates)
        count = 0

        for session_info in big_sessions:
            sid = session_info["session_id"]
            entries = await self.get_session(sid, limit=1000)

            # Identify old entries without summaries.
            old_entries: List[MemoryEntry] = []
            for entry in entries:
                if entry.summary is not None:
                    continue
                try:
                    entry_time = datetime.fromisoformat(entry.timestamp)
                except (ValueError, TypeError):
                    continue
                age = (now - entry_time).days
                if age >= threshold:
                    old_entries.append(entry)

            if not old_entries:
                continue

            # Group into batches of 10 for summarization.
            batch_size = 10
            for i in range(0, len(old_entries), batch_size):
                batch = old_entries[i : i + batch_size]
                batch_summary = await self._summarize_entries_batch(batch)

                def _update(entry_ids: List[str], summary: str) -> None:
                    conn = self._ensure_conn()
                    for eid in entry_ids:
                        conn.execute(
                            "UPDATE entries SET summary = ? WHERE id = ?",
                            (summary, eid),
                        )
                    conn.commit()

                await self._run_sync(_update, [e.id for e in batch], batch_summary)
                count += len(batch)

        # Also re-summarize large sessions.
        for session_info in big_sessions:
            await self.summarize_session(session_info["session_id"])

        return count

    # ── Export / Import ───────────────────────────────────────────────────

    async def export_session(self, session_id: str, filepath: str) -> None:
        """
        Export a session (metadata + all entries) to a JSON file.

        Parameters
        ----------
        session_id:
            The session to export.
        filepath:
            Destination path for the JSON file.  Parent directories are
            created automatically.
        """
        session_info = await self.get_session_info(session_id)
        if session_info is None:
            raise KeyError(f"Session '{session_id}' not found")

        entries = await self.get_session(session_id, limit=100_000)

        payload = {
            "exported_at": _now_iso(),
            "session": session_info,
            "entries": [asdict(e) for e in entries],
        }

        path = Path(filepath).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        def _write() -> None:
            path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

        await self._run_sync(_write)

    async def import_session(self, filepath: str, name_override: Optional[str] = None) -> str:
        """
        Import a previously exported session from a JSON file.

        A new session id is assigned so that the import is idempotent
        (re-importing the same file creates a new copy).

        Parameters
        ----------
        filepath:
            Path to the JSON file produced by :meth:`export_session`.
        name_override:
            If given, override the session name from the file.

        Returns
        -------
        str
            The id of the newly created session.
        """
        path = Path(filepath).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Import file not found: {path}")

        def _read() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            session_data = data.get("session", {})
            entries_data = data.get("entries", [])
            return session_data, entries_data

        session_data, entries_data = await self._run_sync(_read)

        session_name = name_override or session_data.get("name", "imported")
        session_meta = session_data.get("metadata", {})
        new_session_id = await self.create_session(session_name, metadata=session_meta)

        for entry_dict in entries_data:
            role = entry_dict.get("role", "user")
            content = entry_dict.get("content", "")
            tags = entry_dict.get("tags", [])
            meta = entry_dict.get("metadata", {})
            timestamp = entry_dict.get("timestamp", _now_iso())

            entry_id = _generate_id()
            tokens = _tokenize(content)
            embedding = self._compute_tfidf_tokens(tokens)
            tags_json = json.dumps(tags)
            meta_json = json.dumps(meta)

            def _insert(
                eid: str, sid: str, r: str, c: str, ts: str,
                tj: str, emb: Optional[List[float]], mj: str,
            ) -> None:
                conn = self._ensure_conn()
                conn.execute(
                    "INSERT INTO entries "
                    "(id, session_id, role, content, timestamp, tags, embedding_vector, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (eid, sid, r, c, ts, tj,
                     json.dumps(emb) if emb else None, mj),
                )
                conn.commit()

            await self._run_sync(
                _insert, entry_id, new_session_id, role, content,
                timestamp, tags_json, embedding, meta_json,
            )

        await self._rebuild_idf_and_vocab()
        return new_session_id

    # ── Statistics ────────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """
        Return usage statistics.

        Keys: ``total_entries``, ``total_sessions``, ``db_size_bytes``,
        ``sessions``, ``tag_counts``, ``oldest_entry``, ``newest_entry``.
        """
        def _do() -> Dict[str, Any]:
            conn = self._ensure_conn()
            total_entries = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            oldest = conn.execute("SELECT MIN(timestamp) FROM entries").fetchone()[0]
            newest = conn.execute("SELECT MAX(timestamp) FROM entries").fetchone()[0]

            # Tag frequency.
            tag_rows = conn.execute("SELECT tags FROM entries WHERE tags != '[]'").fetchall()
            tag_counter: Counter = Counter()
            for r in tag_rows:
                for tag in json.loads(r["tags"]):
                    tag_counter[tag] += 1

            return {
                "total_entries": total_entries,
                "total_sessions": total_sessions,
                "db_size_bytes": db_size,
                "db_size_mb": round(db_size / (1024 * 1024), 2),
                "oldest_entry": oldest,
                "newest_entry": newest,
                "tag_counts": dict(tag_counter.most_common(20)),
                "vocabulary_size": len(self._vocabulary),
            }

        return await self._run_sync(_do)

    # ── Tag management ────────────────────────────────────────────────────

    async def add_tags(self, entry_id: str, tags: List[str]) -> None:
        """Append tags to an existing entry."""
        def _do() -> None:
            conn = self._ensure_conn()
            row = conn.execute("SELECT tags FROM entries WHERE id = ?", (entry_id,)).fetchone()
            if row is None:
                raise KeyError(f"Entry '{entry_id}' not found")
            existing = set(json.loads(row["tags"]))
            existing.update(tags)
            conn.execute(
                "UPDATE entries SET tags = ? WHERE id = ?",
                (json.dumps(sorted(existing)), entry_id),
            )
            conn.commit()

        await self._run_sync(_do)

    async def remove_tags(self, entry_id: str, tags: List[str]) -> None:
        """Remove tags from an existing entry."""
        def _do() -> None:
            conn = self._ensure_conn()
            row = conn.execute("SELECT tags FROM entries WHERE id = ?", (entry_id,)).fetchone()
            if row is None:
                raise KeyError(f"Entry '{entry_id}' not found")
            existing = set(json.loads(row["tags"]))
            for t in tags:
                existing.discard(t)
            conn.execute(
                "UPDATE entries SET tags = ? WHERE id = ?",
                (json.dumps(sorted(existing)), entry_id),
            )
            conn.commit()

        await self._run_sync(_do)

    async def get_entries_by_tag(self, tag: str, limit: int = 50) -> List[MemoryEntry]:
        """Return entries that have a specific tag."""
        def _do() -> List[MemoryEntry]:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT * FROM entries WHERE tags LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f'%"{tag}"%', limit),
            ).fetchall()
            return [_row_to_entry(r) for r in rows]

        return await self._run_sync(_do)

    async def get_all_tags(self) -> Dict[str, int]:
        """Return a mapping of tag -> count across all entries."""
        def _do() -> Dict[str, int]:
            conn = self._ensure_conn()
            rows = conn.execute("SELECT tags FROM entries WHERE tags != '[]'").fetchall()
            counter: Counter = Counter()
            for r in rows:
                for tag in json.loads(r["tags"]):
                    counter[tag] += 1
            return dict(counter)

        return await self._run_sync(_do)

    # ── Entry management ──────────────────────────────────────────────────

    async def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a single entry by id."""
        def _do() -> Optional[MemoryEntry]:
            conn = self._ensure_conn()
            row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
            return _row_to_entry(row) if row else None

        return await self._run_sync(_do)

    async def delete_entry(self, entry_id: str) -> None:
        """Delete a single entry by id."""
        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            conn.commit()

        await self._run_sync(_do)

    async def update_entry_content(self, entry_id: str, content: str) -> None:
        """Update the content and re-compute the embedding for an entry."""
        tokens = _tokenize(content)
        embedding = self._compute_tfidf_tokens(tokens)

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "UPDATE entries SET content = ?, embedding_vector = ? WHERE id = ?",
                (content, json.dumps(embedding) if embedding else None, entry_id),
            )
            conn.commit()

        await self._run_sync(_do)

    # ── TF-IDF computation ───────────────────────────────────────────────

    def _compute_tfidf(self, text: str) -> List[float]:
        """
        Public convenience wrapper: tokenize *text* and compute its TF-IDF
        vector using the current vocabulary and IDF cache.
        """
        tokens = _tokenize(text)
        return self._compute_tfidf_tokens(tokens) or []

    def _compute_tfidf_tokens(self, tokens: List[str]) -> Optional[List[float]]:
        """
        Compute the TF-IDF vector for *tokens*.

        The vector has the same dimensionality as ``self._vocabulary``.  If
        the vocabulary is empty (no entries stored yet) ``None`` is returned.
        """
        if not self._vocabulary:
            return None

        dim = len(self._vocabulary)
        vec = [0.0] * dim

        if not tokens:
            return vec

        # Term frequency (normalized).
        tf_counts: Counter = Counter(tokens)
        total = len(tokens)
        for token, count in tf_counts.items():
            idx = self._vocab_index.get(token)
            if idx is not None:
                vec[idx] = count / total

        # Multiply by IDF.
        for i in range(dim):
            token = self._vocabulary[i]
            idf = self._idf_cache.get(token, 0.0)
            vec[i] *= idf

        return vec

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.  Returns ``0.0`` for
        zero-norm vectors (no division by zero).
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return dot / (norm1 * norm2)

    async def _rebuild_idf_and_vocab(self) -> None:
        """
        Rebuild the IDF cache and vocabulary from all stored entries.

        IDF(token) = log(N / df(token))  where N = total docs, df = docs
        containing the token.  The vocabulary is capped at
        ``_MAX_VOCAB_SIZE`` by selecting the most frequent tokens.
        """
        def _do() -> Tuple[Dict[str, int], int]:
            conn = self._ensure_conn()
            rows = conn.execute("SELECT content FROM entries").fetchall()
            return {r["content"]: idx for idx, r in enumerate(rows)}, len(rows)

        content_map, total_docs = await self._run_sync(_do)

        if total_docs == 0:
            self._idf_cache = {}
            self._vocabulary = []
            self._vocab_index = {}
            return

        # Document frequency.
        df: Counter = Counter()
        global_freq: Counter = Counter()
        for content in content_map:
            tokens = set(_tokenize(content))
            for tok in tokens:
                df[tok] += 1

        # Also collect raw frequencies for vocabulary selection.
        for content in content_map:
            for tok in _tokenize(content):
                global_freq[tok] += 1

        # Select top tokens for vocabulary.
        top_tokens = [t for t, _ in global_freq.most_common(_MAX_VOCAB_SIZE)]

        # Compute IDF.
        new_idf: Dict[str, float] = {}
        for token in top_tokens:
            doc_freq = df.get(token, 0)
            if doc_freq > 0:
                new_idf[token] = math.log((total_docs + 1) / (doc_freq + 1)) + 1.0
            else:
                new_idf[token] = 1.0

        self._idf_cache = new_idf
        self._vocabulary = sorted(top_tokens)
        self._vocab_index = {tok: i for i, tok in enumerate(self._vocabulary)}

        # Persist IDF cache to DB for fast restart.
        def _persist() -> None:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM idf_cache")
            conn.executemany(
                "INSERT OR REPLACE INTO idf_cache (token, idf) VALUES (?, ?)",
                [(tok, score) for tok, score in new_idf.items()],
            )
            conn.commit()

        await self._run_sync(_persist)

    # ── Cleanup ───────────────────────────────────────────────────────────

    async def vacuum(self) -> None:
        """Run VACUUM to reclaim disk space after large deletions."""
        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute("VACUUM")

        await self._run_sync(_do)

    async def clear_all(self) -> None:
        """Delete all sessions, entries, and cached indexes."""
        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM entries")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM idf_cache")
            conn.execute("DELETE FROM vocabulary")
            conn.commit()

        await self._run_sync(_do)
        self._idf_cache = {}
        self._vocabulary = []
        self._vocab_index = {}

    # ── Dunder helpers ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"<ConversationMemory db={self.db_path!s} vocab={len(self._vocabulary)}>"

    async def __aenter__(self) -> "ConversationMemory":
        await self.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: module-level singleton (lazy)
# ──────────────────────────────────────────────────────────────────────────────

_default_instance: Optional[ConversationMemory] = None


async def get_memory(db_path: str = DEFAULT_DB_PATH) -> ConversationMemory:
    """
    Return a module-level :class:`ConversationMemory` singleton, initializing
    it on first use.  Subsequent calls return the same instance regardless of
    *db_path*.
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = ConversationMemory(db_path=db_path)
        await _default_instance.initialize()
    return _default_instance


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper for manual inspection / testing
# ──────────────────────────────────────────────────────────────────────────────

async def _cli_main() -> None:
    """Simple CLI for inspecting memory from the terminal."""
    import argparse

    parser = argparse.ArgumentParser(description="Claude Clone Memory Inspector")
    sub = parser.add_subparsers(dest="cmd")

    p_stats = sub.add_parser("stats", help="Show memory statistics")
    p_stats.add_argument("--db", default=DEFAULT_DB_PATH)

    p_sessions = sub.add_parser("sessions", help="List all sessions")
    p_sessions.add_argument("--db", default=DEFAULT_DB_PATH)

    p_search = sub.add_parser("search", help="Semantic search")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--session", default=None)
    p_search.add_argument("--tags", nargs="*", default=None)
    p_search.add_argument("--db", default=DEFAULT_DB_PATH)

    p_export = sub.add_parser("export", help="Export a session to JSON")
    p_export.add_argument("session_id", help="Session ID")
    p_export.add_argument("filepath", help="Output file path")
    p_export.add_argument("--db", default=DEFAULT_DB_PATH)

    p_import = sub.add_parser("import", help="Import a session from JSON")
    p_import.add_argument("filepath", help="Input JSON file path")
    p_import.add_argument("--name", default=None, help="Override session name")
    p_import.add_argument("--db", default=DEFAULT_DB_PATH)

    p_auto_summarize = sub.add_parser("summarize", help="Auto-summarize old entries")
    p_auto_summarize.add_argument("--db", default=DEFAULT_DB_PATH)

    args = parser.parse_args()

    if args.cmd == "stats":
        async with ConversationMemory(db_path=args.db) as mem:
            stats = await mem.get_stats()
            print(json.dumps(stats, indent=2, default=str))

    elif args.cmd == "sessions":
        async with ConversationMemory(db_path=args.db) as mem:
            sessions = await mem.list_sessions()
            if not sessions:
                print("No sessions found.")
            for s in sessions:
                print(
                    f"  {s['id'][:12]}  {s['name']:<30}  "
                    f"{s['entry_count']} entries  "
                    f"updated {s['updated_at'][:16]}"
                )

    elif args.cmd == "search":
        async with ConversationMemory(db_path=args.db) as mem:
            results = await mem.search(
                args.query, limit=args.limit,
                session_id=args.session, tags=args.tags,
            )
            if not results:
                print("No results found.")
            for r in results:
                preview = (r.summary or r.content)[:120].replace("\n", " ")
                print(f"  [{r.role}] {r.id[:12]}  {preview}")

    elif args.cmd == "export":
        async with ConversationMemory(db_path=args.db) as mem:
            await mem.export_session(args.session_id, args.filepath)
            print(f"Exported session {args.session_id} to {args.filepath}")

    elif args.cmd == "import":
        async with ConversationMemory(db_path=args.db) as mem:
            sid = await mem.import_session(args.filepath, name_override=args.name)
            print(f"Imported as session {sid}")

    elif args.cmd == "summarize":
        async with ConversationMemory(db_path=args.db) as mem:
            count = await mem.auto_summarize()
            print(f"Summarized {count} entries.")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(_cli_main())
