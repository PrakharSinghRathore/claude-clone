"""
Abstract Base Class for Memory Providers.

Defines the interface that all memory implementations must conform to.
The Atlas memory system supports a built-in file-based provider and at
most one external plugin provider, both orchestrated by ``MemoryManager``.

Usage
-----
    class MyMemoryProvider(MemoryProvider):
        async def initialize(self) -> None: ...
        async def search(self, query, limit=10) -> list[MemoryEntry]: ...
        async def store(self, entry: MemoryEntry) -> str: ...
        async def close(self) -> None: ...

    provider = MyMemoryProvider()
    await provider.initialize()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """
    A single memory record.

    Attributes
    ----------
    id:
        Unique identifier for this entry.
    content:
        The main text content of the memory.
    role:
        The role associated with this memory (user, assistant, system, tool).
    tags:
        List of tags for categorization and filtering.
    timestamp:
        ISO-8601 timestamp of when this entry was created.
    source:
        Origin of the memory (builtin, plugin, session, etc.).
    metadata:
        Arbitrary key-value metadata attached to the entry.
    session_id:
        Optional session identifier.
    importance:
        Subjective importance score (0.0–1.0) for prioritization.
    """

    id: str
    content: str
    role: str = "user"
    tags: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "builtin"
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    importance: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this entry to a dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryEntry:
        """Deserialize an entry from a dictionary."""
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            role=data.get("role", "user"),
            tags=data.get("tags", []),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            source=data.get("source", "builtin"),
            metadata=data.get("metadata", {}),
            session_id=data.get("session_id"),
            importance=data.get("importance", 0.5),
        )


class MemoryProvider(ABC):
    """
    Abstract base class for memory providers.

    All memory implementations (built-in file-based, external plugins, etc.)
    must implement this interface to be compatible with ``MemoryManager``.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the memory provider.

        Called once before any other methods. Implementations should use
        this to set up connections, load data, etc.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """
        Clean up resources used by the memory provider.

        Implementations should close connections, flush buffers, etc.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[MemoryEntry]:
        """
        Search for relevant memories.

        Parameters
        ----------
        query:
            The search query text.
        limit:
            Maximum number of results to return.
        session_id:
            Optional session filter.
        tags:
            Optional tag filter — only return entries with all specified tags.

        Returns
        -------
        list[MemoryEntry]
            Results sorted by relevance (most relevant first).
        """
        ...

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """
        Store a new memory entry.

        Parameters
        ----------
        entry:
            The memory entry to store.

        Returns
        -------
        str
            The ID of the stored entry (may differ from ``entry.id``).
        """
        ...

    @abstractmethod
    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a specific memory entry by ID.

        Parameters
        ----------
        entry_id:
            The unique identifier of the entry.

        Returns
        -------
        MemoryEntry or None
            The entry, or ``None`` if not found.
        """
        ...

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry.

        Parameters
        ----------
        entry_id:
            The unique identifier of the entry to delete.

        Returns
        -------
        bool
            ``True`` if the entry was deleted, ``False`` if not found.
        """
        ...

    @abstractmethod
    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        session_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """
        List memory entries.

        Parameters
        ----------
        limit:
            Maximum entries to return.
        offset:
            Number of entries to skip.
        session_id:
            Optional session filter.

        Returns
        -------
        list[MemoryEntry]
            Entries ordered by timestamp (newest first).
        """
        ...

    # ── Optional methods with defaults ────────────────────────────────────

    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update fields of an existing memory entry.

        Default implementation retrieves, applies updates, and re-stores.
        Subclasses may override for efficiency.

        Parameters
        ----------
        entry_id:
            The ID of the entry to update.
        updates:
            Dict of fields to update (only recognized fields are applied).

        Returns
        -------
        bool
            ``True`` if the entry was updated successfully.
        """
        entry = await self.get(entry_id)
        if entry is None:
            return False

        for key, value in updates.items():
            if hasattr(entry, key):
                setattr(entry, key, value)

        await self.store(entry)
        return True

    async def get_context_for_prompt(
        self,
        query: str,
        max_tokens: int = 4000,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Build a context string from relevant memories for prompt injection.

        Default implementation searches for relevant memories and formats
        them into a readable block. Subclasses may override for custom
        formatting or ranking.

        Parameters
        ----------
        query:
            The user prompt for which context is being gathered.
        max_tokens:
            Approximate token budget for the context string.
        session_id:
            Optional session filter.

        Returns
        -------
        str
            A formatted context block suitable for injection into a system
            prompt. Returns an empty string if no relevant memories are found.
        """
        entries = await self.search(query, limit=10, session_id=session_id)
        if not entries:
            return ""

        parts: List[str] = []
        used_chars = 0
        max_chars = max_tokens * 4  # Rough token-to-char estimate

        for entry in entries:
            content = entry.content.strip()
            if not content:
                continue
            display = content[:500]
            role_label = entry.role.capitalize() if entry.role else "Note"
            block = f"[{role_label}] {display}"
            if used_chars + len(block) > max_chars:
                break
            parts.append(block)
            used_chars += len(block)

        if not parts:
            return ""

        return (
            "<memory_context>\n"
            "The following relevant information was retrieved from memory:\n"
            + "\n---\n".join(parts)
            + "\n</memory_context>"
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the memory provider.

        Returns
        -------
        dict
            A dict with ``status`` (``"healthy"`` or ``"unhealthy"``) and
            optional ``error`` and ``details`` keys.
        """
        return {"status": "healthy"}

    async def get_stats(self) -> Dict[str, Any]:
        """
        Return usage statistics for the memory provider.

        Returns
        -------
        dict
            Statistics including entry counts, storage size, etc.
        """
        return {"total_entries": 0, "status": "available"}
