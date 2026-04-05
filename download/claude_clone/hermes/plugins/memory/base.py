"""
Abstract Memory Plugin Interface.

All memory backend plugins must subclass ``BaseMemoryPlugin`` and
implement the required methods: store, retrieve, search, delete,
and health_check.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MemoryPluginType(str, Enum):
    """Supported memory plugin categories."""

    DIALECTIC = "dialectic"          # Honcho-style user modelling
    SEMANTIC = "semantic"            # Vector/embedding-based retrieval
    KEYWORD = "keyword"              # Full-text / tag-based
    GRAPH = "graph"                  # Graph-based relationship mapping
    POST_HOC = "post_hoc"           # Retrospective analysis
    PERSISTENT = "persistent"        # Database-backed storage
    LIGHTWEIGHT = "lightweight"      # File-based, minimal overhead


@dataclass
class MemoryConfig:
    """Configuration schema for a memory plugin."""

    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    storage_path: Optional[str] = None
    database_url: Optional[str] = None
    max_entries: int = 10_000
    retention_days: int = 90
    auto_cleanup: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class MemoryPluginMetadata:
    """Metadata describing a memory plugin."""

    name: str
    display_name: str
    version: str
    description: str
    plugin_type: MemoryPluginType
    author: str = ""
    homepage: str = ""
    config_schema: dict = field(default_factory=dict)
    required_packages: list[str] = field(default_factory=list)


@dataclass
class MemoryEntry:
    """A single memory record stored/retrieved by plugins."""

    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    updated_at: Optional[datetime] = None
    source: str = ""  # Which plugin created it
    relevance_score: float = 0.0


class BaseMemoryPlugin(abc.ABC):
    """
    Abstract base class for all memory plugins.

    Subclasses must implement: store, retrieve, search, delete, health_check.
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config = config or MemoryConfig()
        self._initialized = False

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Set up connections, load data, etc. Called once at startup."""

    @abc.abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry. Returns the entry ID."""

    @abc.abstractmethod
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a single memory entry by ID."""

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """Search memory for entries matching *query*."""

    @abc.abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by ID. Returns True if found and deleted."""

    @abc.abstractmethod
    async def health_check(self) -> dict:
        """
        Check plugin health. Returns a dict with keys:
            status: "healthy" | "degraded" | "unhealthy"
            latency_ms: float
            details: dict
        """

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""

    async def store_batch(self, entries: list[MemoryEntry]) -> list[str]:
        """Store multiple entries. Default: sequential store calls."""
        ids: list[str] = []
        for entry in entries:
            eid = await self.store(entry)
            ids.append(eid)
        return ids

    async def count(self) -> int:
        """Return total number of stored entries. Default: search with wildcard."""
        results = await self.search("", limit=self.config.max_entries)
        return len(results)

    @property
    def is_initialized(self) -> bool:
        return self._initialized
