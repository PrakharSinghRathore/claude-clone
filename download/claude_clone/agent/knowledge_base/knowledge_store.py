"""
Knowledge Base Storage Engine.

A structured, persistent store of knowledge entries — facts, code patterns,
solutions, best practices, domain concepts, troubleshooting steps, and more.

All database operations are async via ``sqlite3`` + ``asyncio.run_in_executor``
so that no external async database drivers (e.g. aiosqlite) are required.

The store maintains three tables:

- ``knowledge_entries``  — Core entry fields (title, content, category, scores …)
- ``knowledge_tags``     — Junction table for efficient tag queries
- ``knowledge_relations`` — Directed edges of the knowledge graph

Usage::

    store = KnowledgeStore()
    await store.initialize()

    entry_id = await store.add(KnowledgeEntry(
        title="FastAPI dependency injection pattern",
        content="Use ``Depends()`` with callable classes …",
        category="pattern",
        tags=["python", "fastapi", "di"],
        source="conversation",
        confidence=0.9,
        importance=0.8,
    ))

    results = await store.search("dependency injection fastapi")
    stats = await store.get_stats()
    await store.close()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "~/.claude_clone/knowledge.db"

#: Allowed entry categories.
VALID_CATEGORIES: Tuple[str, ...] = (
    "pattern",
    "solution",
    "concept",
    "troubleshooting",
    "reference",
    "snippet",
    "decision",
    "lesson",
)

#: Allowed entry sources.
VALID_SOURCES: Tuple[str, ...] = (
    "conversation",
    "code",
    "web",
    "manual",
    "import",
)

#: Comprehensive English stop-words list for tokenization / search.
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

#: Regex for tokenization: split on non-alphanumeric boundaries.
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

#: Default age threshold for pruning old low-confidence entries (days).
_DEFAULT_PRUNE_MAX_AGE_DAYS = 90

#: Default confidence floor for pruning.
_DEFAULT_PRUNE_MIN_CONFIDENCE = 0.3

#: Max entries returned by ``search()`` when no limit is given.
_DEFAULT_SEARCH_LIMIT = 20

#: Max entries returned by category / tag listing when no limit is given.
_DEFAULT_LIST_LIMIT = 50

#: Max entries returned by ``get_top_entries()`` when no limit is given.
_DEFAULT_TOP_LIMIT = 20


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _generate_id() -> str:
    """Return a short random UUID suitable for primary keys (16 hex chars)."""
    return uuid.uuid4().hex[:16]


def _now_iso() -> str:
    """Current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    """
    Lowercase, split on non-alphanumeric boundaries, and drop stop-words
    plus single-character tokens.

    Parameters
    ----------
    text:
        The raw text to tokenize.

    Returns
    -------
    list[str]
        Cleaned, lowercased tokens.
    """
    raw = _TOKEN_RE.findall(text.lower())
    return [tok for tok in raw if tok not in STOP_WORDS and len(tok) > 1]


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the range [*lo*, *hi*]."""
    return max(lo, min(hi, value))


# ──────────────────────────────────────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeEntry:
    """
    A single knowledge entry in the store.

    Attributes
    ----------
    id:
        Unique identifier (16-char hex string).
    title:
        Short human-readable title.
    content:
        Full body text — may be multi-paragraph Markdown.
    category:
        One of :data:`VALID_CATEGORIES`.
    tags:
        Free-form list of tags for filtering and search.
    source:
        Where this knowledge originated (one of :data:`VALID_SOURCES`).
    confidence:
        Reliability score in [0.0, 1.0].
    importance:
        Relevance / priority score in [0.0, 1.0].
    references:
        IDs of related :class:`KnowledgeEntry` records.
    metadata:
        Arbitrary extra data (language, framework, project, etc.).
    created_at:
        ISO-8601 creation timestamp.
    updated_at:
        ISO-8601 last-modified timestamp.
    access_count:
        How many times the entry has been retrieved.
    last_accessed:
        ISO-8601 timestamp of most recent retrieval.
    version:
        Monotonically increasing version number for edit tracking.
    """

    id: str = ""
    title: str = ""
    content: str = ""
    category: str = "reference"
    tags: List[str] = field(default_factory=list)
    source: str = "manual"
    confidence: float = 0.5
    importance: float = 0.5
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    access_count: int = 0
    last_accessed: str = ""
    version: int = 1


# ──────────────────────────────────────────────────────────────────────────────
# SQLite row helpers
# ──────────────────────────────────────────────────────────────────────────────

def _row_to_entry(row: sqlite3.Row, tags: Optional[List[str]] = None) -> KnowledgeEntry:
    """
    Convert a database row (``knowledge_entries`` table) to a
    :class:`KnowledgeEntry`.

    Parameters
    ----------
    row:
        A ``sqlite3.Row`` from the ``knowledge_entries`` table.
    tags:
        Pre-fetched tag list.  If *None*, the tags JSON column is used.

    Returns
    -------
    KnowledgeEntry
    """
    if tags is None:
        raw_tags = row["tags_json"]
        tags = json.loads(raw_tags) if raw_tags else []
    references = json.loads(row["references_json"]) if row["references_json"] else []
    metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    return KnowledgeEntry(
        id=row["id"],
        title=row["title"],
        content=row["content"],
        category=row["category"],
        tags=tags,
        source=row["source"],
        confidence=row["confidence"],
        importance=row["importance"],
        references=references,
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        access_count=row["access_count"],
        last_accessed=row["last_accessed"] or "",
        version=row["version"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# KnowledgeStore
# ──────────────────────────────────────────────────────────────────────────────

class KnowledgeStore:
    """
    SQLite-backed persistent knowledge base with keyword search, tagging,
    category filtering, merging, import/export, and pruning.

    All public methods are ``async`` and run blocking SQLite calls in a
    thread-pool executor (``asyncio.run_in_executor``).

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  ``~`` is expanded automatically.
        The parent directory is created on :meth:`initialize` if it does not
        already exist.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path: Path = Path(db_path).expanduser().resolve()
        self._conn: Optional[sqlite3.Connection] = None

    # ── Initialisation / teardown ─────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Create the database schema (tables + indexes) if they do not yet
        exist and open a WAL-mode connection.

        Must be called exactly once before any other method.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(self._connect)
        await self._run_sync(self._create_tables)
        logger.info("KnowledgeStore initialized at %s", self.db_path)

    async def close(self) -> None:
        """
        Close the underlying database connection.

        Safe to call multiple times.
        """
        if self._conn is not None:
            await self._run_sync(self._conn.close)
            self._conn = None
            logger.info("KnowledgeStore connection closed")

    def _connect(self) -> None:
        """Open a SQLite connection with WAL journaling and foreign keys."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _create_tables(self) -> None:
        """Create the three core tables and supporting indexes."""
        assert self._conn is not None
        self._conn.executescript("""
            -- ── Core entries table ──────────────────────────────────────
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id              TEXT PRIMARY KEY,
                title           TEXT NOT NULL,
                content         TEXT NOT NULL DEFAULT '',
                category        TEXT NOT NULL DEFAULT 'reference',
                tags_json       TEXT DEFAULT '[]',
                source          TEXT NOT NULL DEFAULT 'manual',
                confidence      REAL NOT NULL DEFAULT 0.5,
                importance      REAL NOT NULL DEFAULT 0.5,
                references_json TEXT DEFAULT '[]',
                metadata_json   TEXT DEFAULT '{}',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                access_count    INTEGER NOT NULL DEFAULT 0,
                last_accessed   TEXT DEFAULT '',
                version         INTEGER NOT NULL DEFAULT 1,
                archived        INTEGER NOT NULL DEFAULT 0
            );

            -- ── Tag junction table ──────────────────────────────────────
            CREATE TABLE IF NOT EXISTS knowledge_tags (
                entry_id    TEXT NOT NULL,
                tag         TEXT NOT NULL,
                PRIMARY KEY (entry_id, tag),
                FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id)
                    ON DELETE CASCADE
            );

            -- ── Knowledge-graph relations ───────────────────────────────
            CREATE TABLE IF NOT EXISTS knowledge_relations (
                from_id        TEXT NOT NULL,
                to_id          TEXT NOT NULL,
                relation_type  TEXT NOT NULL DEFAULT 'related',
                created_at     TEXT NOT NULL,
                PRIMARY KEY (from_id, to_id, relation_type),
                FOREIGN KEY (from_id) REFERENCES knowledge_entries(id)
                    ON DELETE CASCADE,
                FOREIGN KEY (to_id) REFERENCES knowledge_entries(id)
                    ON DELETE CASCADE
            );

            -- ── Indexes for fast lookups ────────────────────────────────
            CREATE INDEX IF NOT EXISTS idx_ke_category
                ON knowledge_entries(category);

            CREATE INDEX IF NOT EXISTS idx_ke_source
                ON knowledge_entries(source);

            CREATE INDEX IF NOT EXISTS idx_ke_confidence
                ON knowledge_entries(confidence);

            CREATE INDEX IF NOT EXISTS idx_ke_importance
                ON knowledge_entries(importance);

            CREATE INDEX IF NOT EXISTS idx_ke_created_at
                ON knowledge_entries(created_at);

            CREATE INDEX IF NOT EXISTS idx_ke_updated_at
                ON knowledge_entries(updated_at);

            CREATE INDEX IF NOT EXISTS idx_ke_access_count
                ON knowledge_entries(access_count);

            CREATE INDEX IF NOT EXISTS idx_ke_archived
                ON knowledge_entries(archived);

            CREATE INDEX IF NOT EXISTS idx_kt_tag
                ON knowledge_tags(tag);

            CREATE INDEX IF NOT EXISTS idx_kt_entry_id
                ON knowledge_tags(entry_id);

            CREATE INDEX IF NOT EXISTS idx_kr_from
                ON knowledge_relations(from_id);

            CREATE INDEX IF NOT EXISTS idx_kr_to
                ON knowledge_relations(to_id);

            CREATE INDEX IF NOT EXISTS idx_kr_relation_type
                ON knowledge_relations(relation_type);
        """)
        self._conn.commit()

    # ── Async wrapper ─────────────────────────────────────────────────────

    async def _run_sync(self, func, *args, **kwargs):
        """
        Run a synchronous callable in the default thread-pool executor.

        This is the core async bridge — every SQLite interaction goes through
        here so that the event loop is never blocked.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def _ensure_conn(self) -> sqlite3.Connection:
        """
        Return the active SQLite connection or raise ``RuntimeError`` if
        :meth:`initialize` has not been called.
        """
        if self._conn is None:
            raise RuntimeError(
                "KnowledgeStore is not initialized. "
                "Call `await store.initialize()` first."
            )
        return self._conn

    # ── Validation helpers ────────────────────────────────────────────────

    @staticmethod
    def _validate_category(category: str) -> None:
        """Raise :class:`ValueError` if *category* is not in :data:`VALID_CATEGORIES`."""
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category {category!r}. Must be one of: {VALID_CATEGORIES}"
            )

    @staticmethod
    def _validate_source(source: str) -> None:
        """Raise :class:`ValueError` if *source* is not in :data:`VALID_SOURCES`."""
        if source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source {source!r}. Must be one of: {VALID_SOURCES}"
            )

    # ── Add ───────────────────────────────────────────────────────────────

    async def add(self, entry: KnowledgeEntry) -> str:
        """
        Persist a new knowledge entry and return its id.

        If *entry.id* is empty a new UUID will be generated.  Timestamps are
        set to "now" when not provided.

        Parameters
        ----------
        entry:
            The :class:`KnowledgeEntry` to insert.

        Returns
        -------
        str
            The id of the newly-created entry.
        """
        self._validate_category(entry.category)
        self._validate_source(entry.source)

        entry_id = entry.id or _generate_id()
        now = _now_iso()
        created = entry.created_at or now
        updated = entry.updated_at or now

        confidence = _clamp(float(entry.confidence), 0.0, 1.0)
        importance = _clamp(float(entry.importance), 0.0, 1.0)

        tags_list: List[str] = entry.tags or []
        refs_list: List[str] = entry.references or []
        meta_dict: Dict[str, Any] = entry.metadata or {}

        tags_json = json.dumps(sorted(set(tags_list)))
        refs_json = json.dumps(sorted(set(refs_list)))
        meta_json = json.dumps(meta_dict)

        version = max(1, int(entry.version))

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO knowledge_entries "
                "(id, title, content, category, tags_json, source, "
                " confidence, importance, references_json, metadata_json, "
                " created_at, updated_at, access_count, last_accessed, version, archived) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry_id, entry.title, entry.content, entry.category,
                    tags_json, entry.source, confidence, importance,
                    refs_json, meta_json, created, updated,
                    int(entry.access_count), entry.last_accessed or "",
                    version, 0,
                ),
            )

            # Populate junction table for tag queries.
            for tag in sorted(set(tags_list)):
                conn.execute(
                    "INSERT OR IGNORE INTO knowledge_tags (entry_id, tag) VALUES (?, ?)",
                    (entry_id, tag),
                )

            # Populate relation edges for references.
            for ref_id in sorted(set(refs_list)):
                conn.execute(
                    "INSERT OR IGNORE INTO knowledge_relations "
                    "(from_id, to_id, relation_type, created_at) VALUES (?, ?, 'reference', ?)",
                    (entry_id, ref_id, now),
                )

            conn.commit()

        await self._run_sync(_do)
        logger.debug("Added knowledge entry %s: %s", entry_id, entry.title[:60])
        return entry_id

    # ── Get ───────────────────────────────────────────────────────────────

    async def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """
        Retrieve a single knowledge entry by its id.

        Parameters
        ----------
        entry_id:
            The primary key to look up.

        Returns
        -------
        KnowledgeEntry | None
            The matching entry, or ``None`` if not found (or archived).
        """
        def _do() -> Optional[KnowledgeEntry]:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT * FROM knowledge_entries WHERE id = ? AND archived = 0",
                (entry_id,),
            ).fetchone()
            if row is None:
                return None
            # Fetch tags from the junction table.
            tag_rows = conn.execute(
                "SELECT tag FROM knowledge_tags WHERE entry_id = ? ORDER BY tag",
                (entry_id,),
            ).fetchall()
            tags = [r["tag"] for r in tag_rows]
            return _row_to_entry(row, tags=tags)

        return await self._run_sync(_do)

    # ── Update ────────────────────────────────────────────────────────────

    async def update(self, entry_id: str, **fields: Any) -> KnowledgeEntry:
        """
        Update arbitrary fields on an existing entry.

        Recognised field names (snake_case) are:

        - ``title``, ``content``, ``category``, ``tags``, ``source``,
          ``confidence``, ``importance``, ``references``, ``metadata``

        The ``version`` counter is automatically bumped by 1 and
        ``updated_at`` is set to the current time.

        Parameters
        ----------
        entry_id:
            The id of the entry to update.
        **fields:
            Key-value pairs of fields to change.

        Returns
        -------
        KnowledgeEntry
            The updated entry (fetched fresh from the database).

        Raises
        ------
        KeyError
            If the entry does not exist.
        ValueError
            If an invalid category or source value is supplied.
        """
        # Build SET clause dynamically.
        SET_MAP: Dict[str, str] = {
            "title": "title = ?",
            "content": "content = ?",
            "category": "category = ?",
            "source": "source = ?",
            "confidence": "confidence = ?",
            "importance": "importance = ?",
        }

        # Validate category / source early.
        if "category" in fields:
            self._validate_category(fields["category"])
        if "source" in fields:
            self._validate_source(fields["source"])

        # Handle special JSON fields.
        tags_list: Optional[List[str]] = None
        if "tags" in fields:
            tags_list = fields.pop("tags")
            SET_MAP["tags_json"] = "tags_json = ?"

        refs_list: Optional[List[str]] = None
        if "references" in fields:
            refs_list = fields.pop("references")
            SET_MAP["references_json"] = "references_json = ?"

        meta_dict: Optional[Dict[str, Any]] = None
        if "metadata" in fields:
            meta_dict = fields.pop("metadata")
            SET_MAP["metadata_json"] = "metadata_json = ?"

        # Keep only recognised fields.
        recognised = [k for k in fields if k in SET_MAP]
        set_clauses = [SET_MAP[k] for k in recognised]
        values: List[Any] = [fields[k] for k in recognised]

        if tags_list is not None:
            values.append(json.dumps(sorted(set(tags_list))))
        if refs_list is not None:
            values.append(json.dumps(sorted(set(refs_list))))
        if meta_dict is not None:
            values.append(json.dumps(meta_dict))

        if not set_clauses:
            raise ValueError("No recognised fields to update")

        now = _now_iso()

        def _do() -> None:
            conn = self._ensure_conn()

            # Verify existence.
            cur = conn.execute(
                "SELECT id FROM knowledge_entries WHERE id = ? AND archived = 0",
                (entry_id,),
            )
            if cur.fetchone() is None:
                raise KeyError(f"Knowledge entry '{entry_id}' not found")

            # Bump version and updated_at.
            set_clauses.append("version = version + 1")
            set_clauses.append("updated_at = ?")
            values.append(now)

            sql = (
                "UPDATE knowledge_entries SET "
                + ", ".join(set_clauses)
                + " WHERE id = ?"
            )
            values.append(entry_id)
            conn.execute(sql, values)

            # Refresh tags junction table.
            if tags_list is not None:
                conn.execute(
                    "DELETE FROM knowledge_tags WHERE entry_id = ?",
                    (entry_id,),
                )
                for tag in sorted(set(tags_list)):
                    conn.execute(
                        "INSERT OR IGNORE INTO knowledge_tags (entry_id, tag) VALUES (?, ?)",
                        (entry_id, tag),
                    )

            # Refresh relations for references.
            if refs_list is not None:
                conn.execute(
                    "DELETE FROM knowledge_relations WHERE from_id = ? AND relation_type = 'reference'",
                    (entry_id,),
                )
                for ref_id in sorted(set(refs_list)):
                    conn.execute(
                        "INSERT OR IGNORE INTO knowledge_relations "
                        "(from_id, to_id, relation_type, created_at) VALUES (?, ?, 'reference', ?)",
                        (entry_id, ref_id, now),
                    )

            conn.commit()

        await self._run_sync(_do)

        result = await self.get(entry_id)
        if result is None:
            raise KeyError(f"Knowledge entry '{entry_id}' not found after update")
        logger.debug("Updated knowledge entry %s (v%d)", entry_id, result.version)
        return result

    # ── Delete (soft) ─────────────────────────────────────────────────────

    async def delete(self, entry_id: str) -> None:
        """
        Soft-delete an entry by setting ``archived = 1``.

        The entry remains in the database but is excluded from all normal
        queries.  Use :meth:`prune` to permanently remove archived entries.

        Parameters
        ----------
        entry_id:
            The id of the entry to archive.

        Raises
        ------
        KeyError
            If the entry does not exist or is already archived.
        """
        def _do() -> None:
            conn = self._ensure_conn()
            cur = conn.execute(
                "SELECT id, archived FROM knowledge_entries WHERE id = ?",
                (entry_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"Knowledge entry '{entry_id}' not found")
            if row["archived"]:
                raise KeyError(f"Knowledge entry '{entry_id}' is already archived")
            conn.execute(
                "UPDATE knowledge_entries SET archived = 1, updated_at = ? WHERE id = ?",
                (_now_iso(), entry_id),
            )
            conn.commit()

        await self._run_sync(_do)
        logger.info("Archived knowledge entry %s", entry_id)

    # ── Search ────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = _DEFAULT_SEARCH_LIMIT,
    ) -> List[KnowledgeEntry]:
        """
        Keyword search across title and content with optional filters.

        The query is tokenized and matched via SQL ``LIKE``.  Results are
        ranked by a simple relevance score that favours title matches, tag
        overlap, and higher importance.

        Parameters
        ----------
        query:
            Free-text search string.
        category:
            Restrict results to a specific category.
        tags:
            Only return entries that have **all** of the given tags.
        limit:
            Maximum number of results (default 20).

        Returns
        -------
        list[KnowledgeEntry]
            Entries sorted by descending relevance.
        """
        if category is not None:
            self._validate_category(category)

        tokens = _tokenize(query)
        if not tokens:
            # Fallback: raw substring match.
            tokens = [query.strip().lower()] if query.strip() else []
        if not tokens:
            return []

        # Build SQL clauses.
        like_clauses: List[str] = []
        like_params: List[Any] = []
        for token in tokens:
            like_clauses.append("(title LIKE ? OR content LIKE ?)")
            like_params.extend([f"%{token}%", f"%{token}%"])

        where_parts: List[str] = ["archived = 0", "(" + " OR ".join(like_clauses) + ")"]
        all_params: List[Any] = list(like_params)

        if category is not None:
            where_parts.append("category = ?")
            all_params.append(category)

        if tags:
            # Use the junction table: each tag must be present.
            for tag in tags:
                where_parts.append(
                    "id IN (SELECT entry_id FROM knowledge_tags WHERE tag = ?)"
                )
                all_params.append(tag)

        where_sql = " WHERE " + " AND ".join(where_parts)

        def _do() -> List[KnowledgeEntry]:
            conn = self._ensure_conn()
            rows = conn.execute(
                f"SELECT * FROM knowledge_entries{where_sql} "
                f"ORDER BY importance DESC, updated_at DESC LIMIT ?",
                all_params + [limit],
            ).fetchall()

            # Score and re-rank.
            query_set = set(tokens)
            tag_set = set(tags or [])
            scored: List[Tuple[float, sqlite3.Row]] = []
            for row in rows:
                score = 0.0
                title_tokens = set(_tokenize(row["title"]))
                content_tokens = set(_tokenize(row["content"]))
                # Title matches weighted higher.
                score += len(title_tokens & query_set) * 3.0
                score += len(content_tokens & query_set) * 1.0
                score += row["importance"] * 2.0
                score += row["confidence"]
                if tag_set:
                    row_tags = set(_tokenize(row["tags_json"] or ""))
                    score += len(row_tags & tag_set) * 2.0
                scored.append((score, row))

            scored.sort(key=lambda x: x[0], reverse=True)

            results: List[KnowledgeEntry] = []
            for _, row in scored:
                tag_rows = conn.execute(
                    "SELECT tag FROM knowledge_tags WHERE entry_id = ? ORDER BY tag",
                    (row["id"],),
                ).fetchall()
                entry_tags = [r["tag"] for r in tag_rows]
                results.append(_row_to_entry(row, tags=entry_tags))
            return results

        return await self._run_sync(_do)

    # ── Get by category ───────────────────────────────────────────────────

    async def get_by_category(
        self,
        category: str,
        limit: int = _DEFAULT_LIST_LIMIT,
    ) -> List[KnowledgeEntry]:
        """
        Return all non-archived entries of a given category, ordered by
        descending importance.

        Parameters
        ----------
        category:
            One of :data:`VALID_CATEGORIES`.
        limit:
            Maximum entries to return (default 50).

        Returns
        -------
        list[KnowledgeEntry]
        """
        self._validate_category(category)

        def _do() -> List[KnowledgeEntry]:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT * FROM knowledge_entries "
                "WHERE category = ? AND archived = 0 "
                "ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (category, limit),
            ).fetchall()
            results: List[KnowledgeEntry] = []
            for row in rows:
                tag_rows = conn.execute(
                    "SELECT tag FROM knowledge_tags WHERE entry_id = ? ORDER BY tag",
                    (row["id"],),
                ).fetchall()
                tags = [r["tag"] for r in tag_rows]
                results.append(_row_to_entry(row, tags=tags))
            return results

        return await self._run_sync(_do)

    # ── Get by tags ───────────────────────────────────────────────────────

    async def get_by_tags(
        self,
        tags: List[str],
        match_all: bool = True,
        limit: int = _DEFAULT_LIST_LIMIT,
    ) -> List[KnowledgeEntry]:
        """
        Return entries that match the given tags.

        Parameters
        ----------
        tags:
            Tags to filter by.
        match_all:
            If ``True`` (default), entries must have **all** tags.
            If ``False``, entries must have **at least one** tag.
        limit:
            Maximum entries to return (default 50).

        Returns
        -------
        list[KnowledgeEntry]
            Entries ordered by descending importance.
        """
        if not tags:
            return []

        if match_all:
            # Require every tag via INTERSECT / COUNT.
            tag_filter = (
                "id IN ("
                "  SELECT entry_id FROM knowledge_tags WHERE tag IN ({subs})"
                "  GROUP BY entry_id HAVING COUNT(DISTINCT tag) = {count}"
                ")"
            ).format(subs=",".join("?" for _ in tags), count=len(tags))
        else:
            tag_filter = "id IN (SELECT entry_id FROM knowledge_tags WHERE tag IN ({subs}))".format(
                subs=",".join("?" for _ in tags)
            )

        def _do() -> List[KnowledgeEntry]:
            conn = self._ensure_conn()
            rows = conn.execute(
                f"SELECT * FROM knowledge_entries "
                f"WHERE archived = 0 AND {tag_filter} "
                f"ORDER BY importance DESC, updated_at DESC LIMIT ?",
                tags + [limit],
            ).fetchall()
            results: List[KnowledgeEntry] = []
            for row in rows:
                tag_rows = conn.execute(
                    "SELECT tag FROM knowledge_tags WHERE entry_id = ? ORDER BY tag",
                    (row["id"],),
                ).fetchall()
                entry_tags = [r["tag"] for r in tag_rows]
                results.append(_row_to_entry(row, tags=entry_tags))
            return results

        return await self._run_sync(_do)

    # ── Top entries ───────────────────────────────────────────────────────

    async def get_top_entries(
        self,
        limit: int = _DEFAULT_TOP_LIMIT,
        sort_by: str = "importance",
    ) -> List[KnowledgeEntry]:
        """
        Return the top-ranked non-archived entries.

        Parameters
        ----------
        limit:
            Maximum entries to return (default 20).
        sort_by:
            One of ``"importance"``, ``"confidence"``, ``"access_count"``,
            ``"created_at"``, ``"updated_at"``.

        Returns
        -------
        list[KnowledgeEntry]
        """
        valid_sorts = {
            "importance": "importance DESC",
            "confidence": "confidence DESC",
            "access_count": "access_count DESC",
            "created_at": "created_at DESC",
            "updated_at": "updated_at DESC",
        }
        order_clause = valid_sorts.get(sort_by)
        if order_clause is None:
            raise ValueError(
                f"Invalid sort_by={sort_by!r}. "
                f"Must be one of: {list(valid_sorts.keys())}"
            )

        def _do() -> List[KnowledgeEntry]:
            conn = self._ensure_conn()
            rows = conn.execute(
                f"SELECT * FROM knowledge_entries "
                f"WHERE archived = 0 ORDER BY {order_clause} LIMIT ?",
                (limit,),
            ).fetchall()
            results: List[KnowledgeEntry] = []
            for row in rows:
                tag_rows = conn.execute(
                    "SELECT tag FROM knowledge_tags WHERE entry_id = ? ORDER BY tag",
                    (row["id"],),
                ).fetchall()
                tags = [r["tag"] for r in tag_rows]
                results.append(_row_to_entry(row, tags=tags))
            return results

        return await self._run_sync(_do)

    # ── Aggregate queries ─────────────────────────────────────────────────

    async def get_all_categories(self) -> Dict[str, int]:
        """
        Return a mapping of category name → entry count (non-archived).

        Returns
        -------
        dict[str, int]
        """
        def _do() -> Dict[str, int]:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT category, COUNT(*) AS cnt "
                "FROM knowledge_entries WHERE archived = 0 "
                "GROUP BY category ORDER BY cnt DESC"
            ).fetchall()
            return {r["category"]: r["cnt"] for r in rows}

        return await self._run_sync(_do)

    async def get_all_tags(self) -> Dict[str, int]:
        """
        Return a mapping of tag name → entry count (non-archived).

        Returns
        -------
        dict[str, int]
        """
        def _do() -> Dict[str, int]:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT t.tag, COUNT(*) AS cnt "
                "FROM knowledge_tags t "
                "INNER JOIN knowledge_entries e ON e.id = t.entry_id "
                "WHERE e.archived = 0 "
                "GROUP BY t.tag ORDER BY cnt DESC"
            ).fetchall()
            return {r["tag"]: r["cnt"] for r in rows}

        return await self._run_sync(_do)

    async def get_stats(self) -> Dict[str, Any]:
        """
        Return aggregate statistics about the knowledge base.

        Keys:

        - ``total_entries`` — non-archived entry count
        - ``archived_entries`` — archived entry count
        - ``total_tags`` — number of unique tags
        - ``total_relations`` — number of relation edges
        - ``by_category`` — ``{category: count}``
        - ``by_source`` — ``{source: count}``
        - ``avg_confidence`` — mean confidence across active entries
        - ``avg_importance`` — mean importance across active entries
        - ``db_size_bytes``, ``db_size_mb``
        - ``oldest_entry``, ``newest_entry``
        - ``most_accessed`` — top-5 entries by access count
        """
        def _do() -> Dict[str, Any]:
            conn = self._ensure_conn()
            total = conn.execute(
                "SELECT COUNT(*) FROM knowledge_entries WHERE archived = 0"
            ).fetchone()[0]
            archived = conn.execute(
                "SELECT COUNT(*) FROM knowledge_entries WHERE archived = 1"
            ).fetchone()[0]
            total_tags = conn.execute(
                "SELECT COUNT(DISTINCT tag) FROM knowledge_tags"
            ).fetchone()[0]
            total_relations = conn.execute(
                "SELECT COUNT(*) FROM knowledge_relations"
            ).fetchone()[0]

            # By category.
            cat_rows = conn.execute(
                "SELECT category, COUNT(*) AS cnt "
                "FROM knowledge_entries WHERE archived = 0 "
                "GROUP BY category"
            ).fetchall()
            by_category = {r["category"]: r["cnt"] for r in cat_rows}

            # By source.
            src_rows = conn.execute(
                "SELECT source, COUNT(*) AS cnt "
                "FROM knowledge_entries WHERE archived = 0 "
                "GROUP BY source"
            ).fetchall()
            by_source = {r["source"]: r["cnt"] for r in src_rows}

            # Averages.
            avg_row = conn.execute(
                "SELECT AVG(confidence), AVG(importance) "
                "FROM knowledge_entries WHERE archived = 0"
            ).fetchone()
            avg_confidence = round(avg_row[0] or 0.0, 4)
            avg_importance = round(avg_row[1] or 0.0, 4)

            # Date range.
            date_row = conn.execute(
                "SELECT MIN(created_at), MAX(created_at) "
                "FROM knowledge_entries WHERE archived = 0"
            ).fetchone()
            oldest = date_row[0] or ""
            newest = date_row[1] or ""

            # Most accessed.
            access_rows = conn.execute(
                "SELECT id, title, access_count "
                "FROM knowledge_entries WHERE archived = 0 "
                "ORDER BY access_count DESC LIMIT 5"
            ).fetchall()
            most_accessed = [
                {"id": r["id"], "title": r["title"], "access_count": r["access_count"]}
                for r in access_rows
            ]

            # DB file size.
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "total_entries": total,
                "archived_entries": archived,
                "total_tags": total_tags,
                "total_relations": total_relations,
                "by_category": by_category,
                "by_source": by_source,
                "avg_confidence": avg_confidence,
                "avg_importance": avg_importance,
                "oldest_entry": oldest,
                "newest_entry": newest,
                "most_accessed": most_accessed,
                "db_size_bytes": db_size,
                "db_size_mb": round(db_size / (1024 * 1024), 2),
            }

        return await self._run_sync(_do)

    # ── Access tracking ───────────────────────────────────────────────────

    async def increment_access(self, entry_id: str) -> None:
        """
        Atomically increment the ``access_count`` of an entry and set
        ``last_accessed`` to the current time.

        Parameters
        ----------
        entry_id:
            The id of the entry to bump.

        Raises
        ------
        KeyError
            If the entry does not exist or is archived.
        """
        now = _now_iso()

        def _do() -> None:
            conn = self._ensure_conn()
            cur = conn.execute(
                "UPDATE knowledge_entries "
                "SET access_count = access_count + 1, last_accessed = ? "
                "WHERE id = ? AND archived = 0",
                (now, entry_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Knowledge entry '{entry_id}' not found or archived")
            conn.commit()

        await self._run_sync(_do)

    # ── Merge ─────────────────────────────────────────────────────────────

    async def merge_entries(
        self,
        source_ids: List[str],
        title: str,
        content: str,
    ) -> str:
        """
        Merge multiple existing entries into a single new entry, archive
        the source entries, and link the new entry to the originals via
        ``reference`` relations.

        Parameters
        ----------
        source_ids:
            IDs of the entries to merge.
        title:
            Title for the new merged entry.
        content:
            Body content for the new merged entry.

        Returns
        -------
        str
            The id of the newly-created merged entry.

        Raises
        ------
        ValueError
            If fewer than two source ids are provided.
        KeyError
            If any source id does not exist.
        """
        if len(source_ids) < 2:
            raise ValueError("merge_entries requires at least 2 source_ids")

        # Fetch all source entries to gather metadata for the merge.
        sources: List[KnowledgeEntry] = []
        for sid in source_ids:
            entry = await self.get(sid)
            if entry is None:
                raise KeyError(f"Source entry '{sid}' not found")
            sources.append(entry)

        # Aggregate tags, references, metadata.
        all_tags: Set[str] = set()
        all_refs: Set[str] = set(source_ids)
        merged_meta: Dict[str, Any] = {"merged_from": source_ids}
        combined_categories: Counter = Counter()
        combined_sources: Counter = Counter()

        for src in sources:
            all_tags.update(src.tags)
            all_refs.update(src.references)
            combined_categories[src.category] += 1
            combined_sources[src.source] += 1
            merged_meta.setdefault("source_metadata", []).append({
                "id": src.id,
                "title": src.title,
                "category": src.category,
            })

        # Determine best category and source (most common).
        merged_category = combined_categories.most_common(1)[0][0]
        merged_source = combined_sources.most_common(1)[0][0]

        # Max confidence / importance among sources.
        merged_confidence = max(src.confidence for src in sources)
        merged_importance = max(src.importance for src in sources)

        # Remove self-references.
        all_refs -= set(source_ids)

        merged_entry = KnowledgeEntry(
            title=title,
            content=content,
            category=merged_category,
            tags=sorted(all_tags),
            source=merged_source,
            confidence=merged_confidence,
            importance=merged_importance,
            references=sorted(all_refs),
            metadata=merged_meta,
        )

        new_id = await self.add(merged_entry)

        # Archive source entries.
        for sid in source_ids:
            try:
                await self.delete(sid)
            except KeyError:
                pass  # Already archived or gone.

        # Add "merged_into" relation from each source to the new entry.
        now = _now_iso()

        def _add_relations() -> None:
            conn = self._ensure_conn()
            for sid in source_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO knowledge_relations "
                    "(from_id, to_id, relation_type, created_at) "
                    "VALUES (?, ?, 'merged_into', ?)",
                    (sid, new_id, now),
                )
            conn.commit()

        await self._run_sync(_add_relations)

        logger.info(
            "Merged %d entries into %s: %s",
            len(source_ids), new_id, title[:60],
        )
        return new_id

    # ── Export / Import ───────────────────────────────────────────────────

    async def export_category(self, category: str, filepath: str) -> None:
        """
        Export all non-archived entries of a given category to a JSON file.

        Parameters
        ----------
        category:
            One of :data:`VALID_CATEGORIES`.
        filepath:
            Destination path.  Parent directories are created automatically.

        Raises
        ------
        KeyError
            If the category is empty (no entries to export).
        """
        self._validate_category(category)
        entries = await self.get_by_category(category, limit=100_000)

        if not entries:
            raise KeyError(f"No entries found for category '{category}'")

        payload = {
            "exported_at": _now_iso(),
            "category": category,
            "count": len(entries),
            "entries": [asdict(e) for e in entries],
        }

        path = Path(filepath).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        def _write() -> None:
            path.write_text(
                json.dumps(payload, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )

        await self._run_sync(_write)
        logger.info(
            "Exported %d entries of category '%s' to %s",
            len(entries), category, path,
        )

    async def import_from_file(self, filepath: str) -> int:
        """
        Import entries from a JSON file previously created by
        :meth:`export_category`.

        Imported entries receive new ids to ensure idempotency (re-importing
        the same file creates duplicates, which can be de-duplicated via
        :meth:`merge_entries` if desired).

        Parameters
        ----------
        filepath:
            Path to the JSON export file.

        Returns
        -------
        int
            The number of entries imported.
        """
        path = Path(filepath).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Import file not found: {path}")

        def _read() -> Dict[str, Any]:
            raw = path.read_text(encoding="utf-8")
            return json.loads(raw)

        data = await self._run_sync(_read)
        entries_data = data.get("entries", [])

        if not entries_data:
            logger.warning("Import file contains no entries: %s", path)
            return 0

        count = 0
        for entry_dict in entries_data:
            entry = KnowledgeEntry(
                id=_generate_id(),  # Fresh id.
                title=entry_dict.get("title", ""),
                content=entry_dict.get("content", ""),
                category=entry_dict.get("category", "reference"),
                tags=entry_dict.get("tags", []),
                source="import",
                confidence=entry_dict.get("confidence", 0.5),
                importance=entry_dict.get("importance", 0.5),
                references=entry_dict.get("references", []),
                metadata=entry_dict.get("metadata", {}),
            )
            try:
                await self.add(entry)
                count += 1
            except Exception as exc:
                logger.warning("Failed to import entry %s: %s", entry.title[:60], exc)

        logger.info("Imported %d entries from %s", count, path)
        return count

    # ── Prune ─────────────────────────────────────────────────────────────

    async def prune(
        self,
        max_age_days: int = _DEFAULT_PRUNE_MAX_AGE_DAYS,
        min_confidence: float = _DEFAULT_PRUNE_MIN_CONFIDENCE,
    ) -> int:
        """
        Remove (hard-delete) old, low-confidence entries.

        Candidates are non-archived entries where:

        * ``created_at`` is older than *max_age_days*, **and**
        * ``confidence`` is below *min_confidence*.

        Archived entries older than twice *max_age_days* are also
        permanently removed.

        Parameters
        ----------
        max_age_days:
            Age threshold in days (default 90).
        min_confidence:
            Confidence floor (default 0.3).

        Returns
        -------
        int
            The number of entries permanently removed.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        archive_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max_age_days * 2)
        ).isoformat()

        def _do() -> int:
            conn = self._ensure_conn()

            # Remove old, low-confidence active entries.
            cur = conn.execute(
                "SELECT id FROM knowledge_entries "
                "WHERE archived = 0 AND created_at < ? AND confidence < ?",
                (cutoff, min_confidence),
            )
            stale_ids = [r["id"] for r in cur.fetchall()]

            # Remove old archived entries.
            cur2 = conn.execute(
                "SELECT id FROM knowledge_entries "
                "WHERE archived = 1 AND updated_at < ?",
                (archive_cutoff,),
            )
            stale_archived_ids = [r["id"] for r in cur2.fetchall()]

            all_ids = stale_ids + stale_archived_ids
            if not all_ids:
                return 0

            # Delete relations and tags (cascaded by FK, but explicit is safe).
            placeholders = ",".join("?" for _ in all_ids)
            conn.execute(
                f"DELETE FROM knowledge_relations WHERE from_id IN ({placeholders})",
                all_ids,
            )
            conn.execute(
                f"DELETE FROM knowledge_relations WHERE to_id IN ({placeholders})",
                all_ids,
            )
            conn.execute(
                f"DELETE FROM knowledge_tags WHERE entry_id IN ({placeholders})",
                all_ids,
            )
            conn.execute(
                f"DELETE FROM knowledge_entries WHERE id IN ({placeholders})",
                all_ids,
            )
            conn.commit()
            return len(all_ids)

        removed = await self._run_sync(_do)
        if removed:
            logger.info(
                "Pruned %d entries (max_age=%dd, min_confidence=%.2f)",
                removed, max_age_days, min_confidence,
            )
        return removed

    # ── Knowledge graph helpers ───────────────────────────────────────────

    async def add_relation(
        self,
        from_id: str,
        to_id: str,
        relation_type: str = "related",
    ) -> None:
        """
        Add a directed edge between two knowledge entries.

        Parameters
        ----------
        from_id:
            Source entry id.
        to_id:
            Target entry id.
        relation_type:
            Type of relation (e.g. ``"related"``, ``"supersedes"``,
            ``"depends_on"``).  Default is ``"related"``.
        """
        now = _now_iso()

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT OR IGNORE INTO knowledge_relations "
                "(from_id, to_id, relation_type, created_at) VALUES (?, ?, ?, ?)",
                (from_id, to_id, relation_type, now),
            )
            conn.commit()

        await self._run_sync(_do)

    async def get_related(
        self,
        entry_id: str,
        relation_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[KnowledgeEntry]:
        """
        Get entries related to a given entry via the knowledge graph.

        Searches both directions (outgoing and incoming relations).

        Parameters
        ----------
        entry_id:
            The entry whose relations to fetch.
        relation_type:
            If given, filter to a specific relation type.
        limit:
            Maximum results.

        Returns
        -------
        list[KnowledgeEntry]
        """
        def _do() -> List[KnowledgeEntry]:
            conn = self._ensure_conn()

            if relation_type:
                rows = conn.execute(
                    "SELECT DISTINCT e.* FROM knowledge_entries e "
                    "INNER JOIN knowledge_relations r ON (r.from_id = ? AND r.to_id = e.id) "
                    "  OR (r.to_id = ? AND r.from_id = e.id) "
                    "WHERE e.archived = 0 AND r.relation_type = ? "
                    "ORDER BY e.importance DESC LIMIT ?",
                    (entry_id, entry_id, relation_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT DISTINCT e.* FROM knowledge_entries e "
                    "INNER JOIN knowledge_relations r ON (r.from_id = ? AND r.to_id = e.id) "
                    "  OR (r.to_id = ? AND r.from_id = e.id) "
                    "WHERE e.archived = 0 "
                    "ORDER BY e.importance DESC LIMIT ?",
                    (entry_id, entry_id, limit),
                ).fetchall()

            results: List[KnowledgeEntry] = []
            for row in rows:
                tag_rows = conn.execute(
                    "SELECT tag FROM knowledge_tags WHERE entry_id = ? ORDER BY tag",
                    (row["id"],),
                ).fetchall()
                tags = [r["tag"] for r in tag_rows]
                results.append(_row_to_entry(row, tags=tags))
            return results

        return await self._run_sync(_do)

    async def remove_relation(
        self,
        from_id: str,
        to_id: str,
        relation_type: str = "related",
    ) -> None:
        """
        Remove a specific relation edge from the knowledge graph.

        Parameters
        ----------
        from_id:
            Source entry id.
        to_id:
            Target entry id.
        relation_type:
            The type of relation to remove.
        """
        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "DELETE FROM knowledge_relations "
                "WHERE from_id = ? AND to_id = ? AND relation_type = ?",
                (from_id, to_id, relation_type),
            )
            conn.commit()

        await self._run_sync(_do)

    # ── Restore (un-archive) ──────────────────────────────────────────────

    async def restore(self, entry_id: str) -> KnowledgeEntry:
        """
        Un-archive a previously soft-deleted entry.

        Parameters
        ----------
        entry_id:
            The id of the archived entry to restore.

        Returns
        -------
        KnowledgeEntry
            The restored entry.

        Raises
        ------
        KeyError
            If the entry does not exist or is not archived.
        """
        now = _now_iso()

        def _do() -> None:
            conn = self._ensure_conn()
            cur = conn.execute(
                "SELECT id, archived FROM knowledge_entries WHERE id = ?",
                (entry_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"Knowledge entry '{entry_id}' not found")
            if not row["archived"]:
                raise KeyError(f"Knowledge entry '{entry_id}' is not archived")
            conn.execute(
                "UPDATE knowledge_entries SET archived = 0, updated_at = ? WHERE id = ?",
                (now, entry_id),
            )
            conn.commit()

        await self._run_sync(_do)
        result = await self.get(entry_id)
        if result is None:
            raise KeyError(f"Knowledge entry '{entry_id}' not found after restore")
        logger.info("Restored knowledge entry %s", entry_id)
        return result

    # ── Context retrieval for prompts ─────────────────────────────────────

    async def get_context_for_prompt(
        self,
        query: str,
        max_entries: int = 10,
        category: Optional[str] = None,
        max_chars: int = 4000,
    ) -> str:
        """
        Build a context string from the most relevant knowledge entries,
        suitable for injection into an LLM prompt.

        Strategy:
        1. Search for relevant entries via :meth:`search`.
        2. Greedily pack entries (most relevant first) until *max_chars*
           is reached.
        3. Format as a structured block.

        Parameters
        ----------
        query:
            The prompt or topic to find knowledge for.
        max_entries:
            Maximum number of entries to consider.
        category:
            Optional category filter.
        max_chars:
            Approximate character budget for the context block.

        Returns
        -------
        str
            A formatted context block (may be empty).
        """
        results = await self.search(
            query, category=category, limit=max_entries,
        )

        if not results:
            return ""

        parts: List[str] = []
        used_chars = 0

        for entry in results:
            block = f"[{entry.category.upper()}] {entry.title}\n{entry.content}"
            if used_chars + len(block) > max_chars:
                # Try to include a truncated version.
                remaining = max_chars - used_chars
                if remaining > 100:
                    block = f"[{entry.category.upper()}] {entry.title}\n{entry.content[:remaining - 80]}…"
                else:
                    continue
            parts.append(block)
            used_chars += len(block)

        if not parts:
            return ""

        header = (
            "<knowledge_base>\n"
            "The following relevant knowledge entries are provided for context:\n"
        )
        footer = "\n</knowledge_base>"
        return header + "\n---\n".join(parts) + footer
