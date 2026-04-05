"""
ByteRover Memory Plugin — Lightweight, file-based memory with
tag-based organisation and minimal overhead.

Plugin manifest example (plugin.yaml)::

    name: byterover
    display_name: ByteRover Memory
    version: 1.0.0
    type: lightweight
    description: Lightweight file-based memory with tag organisation.
    author: Hermes Team
    required_packages: []
    config_schema:
      storage_path:
        type: string
        description: Directory for memory files (default: ~/.claude_clone/byterover)
      index_file:
        type: string
        description: Name of the tag index file (default: tag_index.json)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
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


@register_builtin("byterover")
class ByteRoverMemoryPlugin(BaseMemoryPlugin):
    """
    Lightweight file-based memory plugin.

    Stores memories as individual JSON files organised by tags.
    Minimal dependencies and low overhead make it ideal for
    quick prototyping and environments with limited resources.
    """

    metadata = MemoryPluginMetadata(
        name="byterover",
        display_name="ByteRover Memory",
        version="1.0.0",
        description="Lightweight file-based memory with tag organisation.",
        plugin_type=MemoryPluginType.LIGHTWEIGHT,
        author="Hermes Team",
        required_packages=[],
    )

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        super().__init__(config)
        self._storage_path: Path = Path(
            config.storage_path or "~/.claude_clone/byterover"
        ).expanduser().resolve()
        self._index_path: Path = self._storage_path / "tag_index.json"
        self._tag_index: dict[str, list[str]] = {}  # tag -> [entry_ids]

    async def initialize(self) -> None:
        """Create storage directory and load tag index."""
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._load_index()
        self._initialized = True
        logger.info(
            "ByteRover memory plugin initialized (path=%s)",
            self._storage_path,
        )

    async def store(self, entry: MemoryEntry) -> str:
        """Store an entry as a JSON file."""
        entry_id = entry.id or str(uuid.uuid4())
        entry.id = entry_id
        entry.created_at = entry.created_at or datetime.utcnow()

        # Write entry file
        entry_file = self._storage_path / f"{entry_id}.json"
        data = {
            "id": entry_id,
            "content": entry.content,
            "metadata": entry.metadata,
            "tags": entry.tags,
            "source": entry.source or "byterover",
            "created_at": entry.created_at.isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        entry_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        # Update tag index
        for tag in entry.tags:
            self._tag_index.setdefault(tag, []).append(entry_id)
        self._save_index()

        return entry_id

    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve an entry from its JSON file."""
        return self._read_entry_file(entry_id)

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """Search entries by keyword match and optional tag filter."""
        results: list[MemoryEntry] = []
        query_lower = query.lower()
        candidate_ids: Optional[list[str]] = None

        # If filtering by tag, narrow candidates
        if filters and "tags" in filters:
            filter_tags = filters["tags"] if isinstance(filters["tags"], list) else [filters["tags"]]
            candidate_ids = self._get_entries_by_tags(filter_tags)

        # Scan entries
        entries_to_scan = candidate_ids or self._get_all_entry_ids()
        for eid in entries_to_scan:
            entry = self._read_entry_file(eid)
            if entry is None:
                continue
            # Check keyword match
            if query_lower and query_lower not in entry.content.lower():
                continue
            # Score: simple keyword overlap
            words = set(query_lower.split())
            content_words = set(entry.content.lower().split())
            overlap = len(words & content_words) / max(len(words), 1)
            entry.relevance_score = min(overlap * 2, 1.0) if words else 0.5
            results.append(entry)

        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results[:limit]

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry and update the tag index."""
        entry = self._read_entry_file(entry_id)
        if entry is None:
            return False

        entry_file = self._storage_path / f"{entry_id}.json"
        if entry_file.exists():
            entry_file.unlink()

        # Remove from tag index
        for tag in entry.tags:
            if tag in self._tag_index:
                self._tag_index[tag] = [eid for eid in self._tag_index[tag] if eid != entry_id]
                if not self._tag_index[tag]:
                    del self._tag_index[tag]
        self._save_index()
        return True

    async def health_check(self) -> dict:
        """Check plugin health and storage stats."""
        import time
        start = time.monotonic()
        entry_count = len(self._get_all_entry_ids())
        tag_count = len(self._tag_index)
        elapsed = (time.monotonic() - start) * 1000
        return {
            "status": "healthy",
            "latency_ms": elapsed,
            "details": {
                "entries": entry_count,
                "tags": tag_count,
                "storage_path": str(self._storage_path),
            },
        }

    async def shutdown(self) -> None:
        """Persist index and clean up."""
        self._save_index()
        self._initialized = False

    # ------------------------------------------------------------------
    # Tag operations
    # ------------------------------------------------------------------

    async def get_by_tag(self, tag: str) -> list[MemoryEntry]:
        """Get all entries associated with a tag."""
        entry_ids = self._tag_index.get(tag, [])
        results: list[MemoryEntry] = []
        for eid in entry_ids:
            entry = self._read_entry_file(eid)
            if entry:
                results.append(entry)
        return results

    async def add_tag(self, entry_id: str, tag: str) -> bool:
        """Add a tag to an existing entry."""
        entry = self._read_entry_file(entry_id)
        if entry is None:
            return False
        if tag not in entry.tags:
            entry.tags.append(tag)
            self._tag_index.setdefault(tag, []).append(entry_id)
            # Re-write entry file
            entry_file = self._storage_path / f"{entry_id}.json"
            data = json.loads(entry_file.read_text(encoding="utf-8"))
            data["tags"] = entry.tags
            entry_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            self._save_index()
        return True

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self, max_age_days: int = 90) -> int:
        """Remove entries older than max_age_days. Returns count removed."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        removed = 0
        for eid in self._get_all_entry_ids():
            entry = self._read_entry_file(eid)
            if entry and entry.created_at and entry.created_at.replace(tzinfo=None) < cutoff:
                await self.delete(eid)
                removed += 1
        logger.info("ByteRover cleanup removed %d entries", removed)
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_entry_file(self, entry_id: str) -> Optional[MemoryEntry]:
        """Read a single entry from disk."""
        entry_file = self._storage_path / f"{entry_id}.json"
        if not entry_file.exists():
            return None
        try:
            data = json.loads(entry_file.read_text(encoding="utf-8"))
            return MemoryEntry(
                id=data["id"],
                content=data["content"],
                metadata=data.get("metadata", {}),
                tags=data.get("tags", []),
                source=data.get("source", "byterover"),
                created_at=data.get("created_at", ""),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _get_all_entry_ids(self) -> list[str]:
        """Get all entry IDs from the storage directory."""
        ids: list[str] = []
        for f in self._storage_path.glob("*.json"):
            if f.name != "tag_index.json":
                ids.append(f.stem)
        return ids

    def _get_entries_by_tags(self, tags: list[str]) -> list[str]:
        """Get entry IDs that have ALL specified tags."""
        if not tags:
            return self._get_all_entry_ids()
        sets = []
        for tag in tags:
            ids = set(self._tag_index.get(tag, []))
            if not ids:
                return []  # No entries have this tag
            sets.append(ids)
        return list(set.intersection(*sets))

    def _load_index(self) -> None:
        """Load the tag index from disk."""
        if self._index_path.exists():
            try:
                self._tag_index = json.loads(self._index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._tag_index = {}

    def _save_index(self) -> None:
        """Persist the tag index to disk."""
        try:
            self._index_path.write_text(
                json.dumps(self._tag_index, indent=2), encoding="utf-8"
            )
        except OSError:
            logger.exception("Failed to save tag index")
