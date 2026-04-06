"""
SQLite-Backed Vector Embedding Cache.

Persistent storage for document embeddings with cosine-similarity-based
top-k retrieval.  Embeddings are stored as JSON blobs alongside document
text and arbitrary metadata.

All database operations are async via ``sqlite3`` + ``asyncio.run_in_executor``
so that no external async database drivers are required.

Schema
------
``embeddings`` table:

- ``doc_id``       — TEXT PRIMARY KEY
- ``text``         — TEXT  (original document text)
- ``embedding``    — TEXT  (JSON-encoded list of floats)
- ``metadata``     — TEXT  (JSON-encoded dict)
- ``created_at``   — TEXT  (ISO-8601 UTC timestamp)
- ``updated_at``   — TEXT  (ISO-8601 UTC timestamp)

Usage::

    cache = EmbeddingCache()
    await cache.initialize()
    await cache.store("doc-1", "Hello world", [0.1, 0.2, ...], {"source": "cli"})
    results = await cache.search([0.1, 0.3, ...], limit=5)
    await cache.close()
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: Default database path for the embeddings cache.
DEFAULT_DB_PATH: str = "~/.claude_clone/embeddings.db"

#: Default number of results returned by :meth:`EmbeddingCache.search`.
DEFAULT_SEARCH_LIMIT: int = 10

#: Minimum cosine similarity score for a result to be included in
#: search output.  Results below this threshold are filtered out.
_MIN_SIMILARITY_THRESHOLD: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    a:
        First vector.
    b:
        Second vector (must be same length as *a*).

    Returns
    -------
    float
        Similarity in [0.0, 1.0].  Returns 0.0 if vectors have
        zero magnitude or different lengths.
    """
    if len(a) != len(b):
        return 0.0

    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0
    for va, vb in zip(a, b):
        dot += va * vb
        mag_a += va * va
        mag_b += vb * vb

    denom = math.sqrt(mag_a) * math.sqrt(mag_b)
    if denom == 0.0:
        return 0.0

    return min(max(dot / denom, 0.0), 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingCache
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingCache:
    """SQLite-backed persistent cache for document embeddings.

    Stores embeddings as JSON blobs and supports top-k retrieval by
    cosine similarity.  All public methods are async.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  ``~`` is expanded
        automatically.  The parent directory is created on
        :meth:`initialize` if it does not exist.

    Examples
    --------
    >>> cache = EmbeddingCache()
    >>> await cache.initialize()
    >>> await cache.store("d1", "hello world", [0.1, 0.2], {"k": "v"})
    >>> results = await cache.search([0.1, 0.2], limit=5)
    >>> await cache.close()
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path: Path = Path(db_path).expanduser().resolve()
        self._conn: Optional[sqlite3.Connection] = None

    # ── Initialisation / teardown ─────────────────────────────────────────

    async def initialize(self) -> None:
        """Create the database schema and open a WAL-mode connection.

        Must be called exactly once before any other method.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(self._connect)
        await self._run_sync(self._create_tables)
        logger.info("EmbeddingCache initialized at %s", self.db_path)

    async def close(self) -> None:
        """Close the underlying database connection.

        Safe to call multiple times.
        """
        if self._conn is not None:
            await self._run_sync(self._conn.close)
            self._conn = None
            logger.info("EmbeddingCache connection closed")

    def _connect(self) -> None:
        """Open a SQLite connection with WAL journaling."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

    def _create_tables(self) -> None:
        """Create the ``embeddings`` table if it does not exist."""
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id      TEXT PRIMARY KEY,
                text        TEXT NOT NULL,
                embedding   TEXT NOT NULL,
                metadata    TEXT DEFAULT '{}',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_embeddings_created_at
                ON embeddings(created_at);
        """)
        self._conn.commit()

    # ── Async wrapper ─────────────────────────────────────────────────────

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous callable in the default thread-pool executor.

        This is the core async bridge — every SQLite interaction goes
        through here so that the event loop is never blocked.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: func(*args, **kwargs)
        )

    def _ensure_conn(self) -> sqlite3.Connection:
        """Return the active connection or raise ``RuntimeError``."""
        if self._conn is None:
            raise RuntimeError(
                "EmbeddingCache is not initialized. "
                "Call `await cache.initialize()` first."
            )
        return self._conn

    # ── CRUD operations ───────────────────────────────────────────────────

    async def store(
        self,
        doc_id: str,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a document embedding.

        If a document with the same *doc_id* already exists it is
        replaced (upsert semantics).

        Parameters
        ----------
        doc_id:
            Unique identifier for the document.
        text:
            The original document text.
        embedding:
            Dense float vector (list of floats).
        metadata:
            Optional arbitrary key-value metadata attached to the
            document.
        """
        now = _now_iso()
        embedding_json = json.dumps(embedding)
        meta_json = json.dumps(metadata or {})

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT OR REPLACE INTO embeddings "
                "(doc_id, text, embedding, metadata, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, text, embedding_json, meta_json, now, now),
            )
            conn.commit()

        await self._run_sync(_do)
        logger.debug("Stored embedding for doc %s (%d dims)", doc_id, len(embedding))

    async def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored document by its id.

        Parameters
        ----------
        doc_id:
            The document id to look up.

        Returns
        -------
        dict | None
            A dict with keys ``doc_id``, ``text``, ``embedding``,
            ``metadata``, ``created_at``, ``updated_at`` — or ``None``
            if the document does not exist.
        """
        def _do() -> Optional[Dict[str, Any]]:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT * FROM embeddings WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "doc_id": row["doc_id"],
                "text": row["text"],
                "embedding": json.loads(row["embedding"]),
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

        return await self._run_sync(_do)

    async def delete(self, doc_id: str) -> None:
        """Delete a document from the cache.

        Parameters
        ----------
        doc_id:
            The document id to delete.

        Raises
        ------
        KeyError
            If the document does not exist.
        """
        def _do() -> None:
            conn = self._ensure_conn()
            cur = conn.execute(
                "SELECT doc_id FROM embeddings WHERE doc_id = ?",
                (doc_id,),
            )
            if cur.fetchone() is None:
                raise KeyError(f"Document '{doc_id}' not found in embedding cache")
            conn.execute(
                "DELETE FROM embeddings WHERE doc_id = ?",
                (doc_id,),
            )
            conn.commit()

        await self._run_sync(_do)
        logger.debug("Deleted embedding for doc %s", doc_id)

    # ── Search ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query_embedding: List[float],
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Search for the most similar documents by cosine similarity.

        All stored embeddings are loaded and scored against the query
        embedding.  Results are sorted by descending similarity and
        truncated to *limit*.

        Parameters
        ----------
        query_embedding:
            Dense float vector to compare against stored embeddings.
        limit:
            Maximum number of results to return (default 10).

        Returns
        -------
        list[dict]
            Each dict contains ``doc_id``, ``text``, ``score``,
            ``metadata``, ``created_at``, ``updated_at``.  Results
            are sorted by descending ``score``.
        """
        def _do() -> List[Dict[str, Any]]:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT * FROM embeddings ORDER BY updated_at DESC"
            ).fetchall()

            scored: List[tuple[float, Dict[str, Any]]] = []
            for row in rows:
                doc_embedding = json.loads(row["embedding"])
                sim = _cosine_similarity(query_embedding, doc_embedding)
                if sim >= _MIN_SIMILARITY_THRESHOLD:
                    scored.append((sim, {
                        "doc_id": row["doc_id"],
                        "text": row["text"],
                        "score": round(sim, 6),
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in scored[:limit]]

        return await self._run_sync(_do)

    # ── Statistics ─────────────────────────────────────────────────────────

    async def stats(self) -> Dict[str, Any]:
        """Return usage statistics for the embedding cache.

        Returns
        -------
        dict
            Keys: ``count`` (total stored documents),
            ``db_size_bytes``, ``db_size_mb``.
        """
        def _do() -> Dict[str, Any]:
            conn = self._ensure_conn()
            count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            return {
                "count": count,
                "db_size_bytes": db_size,
                "db_size_mb": round(db_size / (1024 * 1024), 2),
            }

        return await self._run_sync(_do)
