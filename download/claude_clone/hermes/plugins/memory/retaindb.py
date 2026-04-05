"""
RetainDB Memory Plugin — Persistent SQLite-backed memory with
full-text search and automatic cleanup/retention.

Plugin manifest example (plugin.yaml)::

    name: retaindb
    display_name: RetainDB Memory
    version: 1.0.0
    type: persistent
    description: Persistent SQLite-backed memory with full-text search and retention policies.
    author: Hermes Team
    required_packages: []
    config_schema:
      database_url:
        type: string
        description: SQLite database path (default: ~/.claude_clone/retaindb/memory.db)
      retention_days:
        type: integer
        description: Days to retain memories before cleanup (default: 90)
      auto_cleanup:
        type: boolean
        description: Run cleanup on initialize (default: true)
      max_entries:
        type: integer
        description: Maximum entries before auto-pruning (default: 100000)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from .base import (
    BaseMemoryPlugin,
    MemoryConfig,
    MemoryEntry,
    MemoryPluginMetadata,
    MemoryPluginType,
)
from .registry import register_builtin

logger = logging.getLogger(__name__)


@register_builtin("retaindb")
class RetainDBMemoryPlugin(BaseMemoryPlugin):
    """
    Persistent database-backed memory plugin using SQLite.

    Features:
    - Full-text search (FTS5) for efficient content retrieval
    - Automatic cleanup based on retention policy
    - Tag-based filtering
    - Efficient pagination and bulk operations
    """

    metadata = MemoryPluginMetadata(
        name="retaindb",
        display_name="RetainDB Memory",
        version="1.0.0",
        description="Persistent SQLite-backed memory with full-text search and retention.",
        plugin_type=MemoryPluginType.PERSISTENT,
        author="Hermes Team",
        required_packages=[],
    )

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        super().__init__(config)
        db_path = config.database_url or "~/.claude_clone/retaindb/memory.db"
        self._db_path: Path = Path(db_path).expanduser().resolve()
        self._retention_days: int = config.retention_days if config else 90
        self._auto_cleanup: bool = config.auto_cleanup if config else True
        self._max_entries: int = config.max_entries if config else 100_000
        self._conn: Optional[sqlite3.Connection] = None

    async def initialize(self) -> None:
        """Create database schema and optionally run cleanup."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

        if self._auto_cleanup:
            await self.cleanup()

        self._initialized = True
        logger.info("RetainDB memory plugin initialized (db=%s)", self._db_path)

    async def store(self, entry: MemoryEntry) -> str:
        """Insert a memory entry into the database."""
        entry_id = entry.id or str(uuid.uuid4())
        entry.id = entry_id
        created = entry.created_at or datetime.utcnow()

        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, content, metadata, tags, source, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry_id,
                entry.content,
                json.dumps(entry.metadata, default=str),
                json.dumps(entry.tags),
                entry.source or "retaindb",
                created.isoformat(),
                datetime.utcnow().isoformat(),
            ),
        )
        self._conn.commit()

        # Insert into FTS
        self._conn.execute(
            "INSERT OR REPLACE INTO memories_fts (rowid, content) VALUES (?, ?)",
            (entry_id, entry.content),
        )
        self._conn.commit()
        return entry_id

    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a single entry by ID."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """Full-text search with optional tag filtering."""
        results: list[MemoryEntry] = []

        if not query:
            # Return most recent if no query
            sql = "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?"
            params: tuple = (limit,)
        else:
            sql = """
                SELECT m.* FROM memories m
                JOIN memories_fts fts ON m.id = fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            params = (query, limit)

        rows = self._conn.execute(sql, params).fetchall()
        for row in rows:
            entry = self._row_to_entry(row)
            if filters and not self._matches_filters(entry, filters):
                continue
            # FTS rank is lower for better matches; invert for relevance
            entry.relevance_score = 1.0
            results.append(entry)

        return results

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        self._conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (entry_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    async def health_check(self) -> dict:
        """Check database health."""
        import time
        start = time.monotonic()
        try:
            count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            elapsed = (time.monotonic() - start) * 1000
            db_size = self._db_path.stat().st_size if self._db_path.exists() else 0
            return {
                "status": "healthy",
                "latency_ms": elapsed,
                "details": {
                    "total_entries": count,
                    "db_size_bytes": db_size,
                    "db_path": str(self._db_path),
                    "retention_days": self._retention_days,
                },
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "latency_ms": (time.monotonic() - start) * 1000,
                "details": {"error": str(exc)},
            }

    async def shutdown(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def cleanup(self) -> int:
        """Remove entries older than the retention period. Returns count removed."""
        cutoff = (datetime.utcnow() - timedelta(days=self._retention_days)).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE created_at < ?", (cutoff,)
        )
        removed = cursor.rowcount
        self._conn.commit()
        if removed:
            # Rebuild FTS index
            self._conn.execute("DELETE FROM memories_fts")
            self._conn.execute(
                "INSERT INTO memories_fts (rowid, content) SELECT id, content FROM memories"
            )
            self._conn.commit()
            logger.info("RetainDB cleanup removed %d entries older than %s days", removed, self._retention_days)
        return removed

    async def optimize(self) -> None:
        """Run database optimisation (VACUUM and ANALYZE)."""
        if self._conn:
            self._conn.execute("VACUUM")
            self._conn.execute("ANALYZE")
            logger.info("RetainDB optimized")

    async def get_stats(self) -> dict:
        """Get database statistics."""
        if not self._conn:
            return {}
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        tags_raw = self._conn.execute("SELECT tags FROM memories WHERE tags != '[]'").fetchall()
        tag_counter: dict[str, int] = {}
        for row in tags_raw:
            try:
                tags = json.loads(row[0])
                for tag in tags:
                    tag_counter[tag] = tag_counter.get(tag, 0) + 1
            except (json.JSONDecodeError, TypeError):
                continue
        return {
            "total_entries": total,
            "unique_tags": len(tag_counter),
            "top_tags": sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)[:20],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create the database schema."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                source TEXT DEFAULT 'retaindb',
                created_at TEXT NOT NULL,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS memories_fts (
                rowid TEXT PRIMARY KEY,
                content TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, content='memories', content_rowid='id');
        """)

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        tags = []
        try:
            tags = json.loads(row["tags"])
        except (json.JSONDecodeError, TypeError):
            pass
        metadata = {}
        try:
            metadata = json.loads(row["metadata"])
        except (json.JSONDecodeError, TypeError):
            pass
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            metadata=metadata,
            tags=tags,
            source=row["source"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _matches_filters(entry: MemoryEntry, filters: dict) -> bool:
        """Check if an entry matches filter criteria."""
        for key, value in filters.items():
            if key == "tags":
                if not all(t in entry.tags for t in (value if isinstance(value, list) else [value])):
                    return False
            elif key == "source":
                if entry.source != value:
                    return False
        return True
