"""
Mem0 Memory Plugin — Mem0 memory service integration for automatic
memory extraction, semantic search, and persistent user memory.

Plugin manifest example (plugin.yaml)::

    name: mem0
    display_name: Mem0 Memory
    version: 1.0.0
    type: semantic
    description: Automatic memory extraction and semantic search via Mem0.
    author: Atlas Team
    required_packages: [mem0ai]
    config_schema:
      api_key:
        type: string
        description: Mem0 API key (or leave blank for local mode)
      user_id:
        type: string
        description: Default user ID for memory operations
"""

from __future__ import annotations

import logging
import uuid
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


@register_builtin("mem0")
class Mem0MemoryPlugin(BaseMemoryPlugin):
    """
    Memory plugin backed by Mem0 (mem0ai).

    Provides automatic memory extraction from conversations,
    semantic search, and persistent per-user memory.
    """

    metadata = MemoryPluginMetadata(
        name="mem0",
        display_name="Mem0 Memory",
        version="1.0.0",
        description="Automatic memory extraction and semantic search via Mem0.",
        plugin_type=MemoryPluginType.SEMANTIC,
        author="Atlas Team",
        required_packages=["mem0ai"],
    )

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        super().__init__(config)
        self._client: Any = None
        self._default_user_id: Optional[str] = None

    async def initialize(self) -> None:
        """Initialize Mem0 client."""
        try:
            from mem0 import Memory  # type: ignore[import-untyped]
            config_dict: dict[str, Any] = {}
            if self.config.api_key:
                config_dict["api_key"] = self.config.api_key
            if self.config.base_url:
                config_dict["base_url"] = self.config.base_url
            self._client = Memory.from_config(config_dict) if config_dict else Memory()
            self._default_user_id = self.config.extra.get("user_id")
            self._initialized = True
            logger.info("Mem0 memory plugin initialized")
        except ImportError:
            logger.warning("mem0ai package not installed; plugin in degraded mode")
            self._initialized = True

    async def store(self, entry: MemoryEntry) -> str:
        """Add a memory using Mem0's automatic extraction."""
        entry_id = entry.id or str(uuid.uuid4())
        if self._client is None:
            logger.warning("Mem0 client not available")
            return entry_id

        user_id = entry.metadata.get("user_id") or self._default_user_id
        metadata = dict(entry.metadata)
        metadata.pop("user_id", None)

        try:
            self._client.add(
                entry.content,
                user_id=user_id or "default",
                metadata=metadata,
            )
            logger.debug("Stored memory via Mem0 for user %s", user_id)
        except Exception:
            logger.exception("Failed to store memory via Mem0")
        return entry_id

    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry by ID."""
        if self._client is None:
            return None
        try:
            result = self._client.get(entry_id)
            if result:
                return MemoryEntry(
                    id=result.get("id", entry_id),
                    content=result.get("memory", ""),
                    metadata=result.get("metadata", {}),
                    source="mem0",
                )
        except Exception:
            logger.exception("Failed to retrieve memory from Mem0")
        return None

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """Search memories using Mem0's semantic search."""
        results: list[MemoryEntry] = []
        if self._client is None:
            return results

        user_id = (filters or {}).get("user_id") or self._default_user_id
        try:
            memos = self._client.search(
                query,
                user_id=user_id or "default",
                limit=limit,
            )
            for memo in memos:
                entry = MemoryEntry(
                    id=memo.get("id", str(uuid.uuid4())),
                    content=memo.get("memory", ""),
                    metadata=memo.get("metadata", {}),
                    source="mem0",
                    relevance_score=memo.get("score", 0.0),
                )
                results.append(entry)
        except Exception:
            logger.exception("Failed to search Mem0 memories")
        return results

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory by ID."""
        if self._client is None:
            return False
        try:
            self._client.delete(entry_id)
            return True
        except Exception:
            logger.exception("Failed to delete memory from Mem0")
            return False

    async def health_check(self) -> dict:
        """Check Mem0 service health."""
        import time
        start = time.monotonic()
        if self._client is None:
            return {
                "status": "degraded",
                "latency_ms": 0.0,
                "details": {"reason": "Mem0 client not initialized"},
            }
        try:
            # Basic search to test connectivity
            self._client.search("health check ping", limit=1)
            elapsed = (time.monotonic() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": elapsed,
                "details": {"user_id": self._default_user_id},
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "latency_ms": (time.monotonic() - start) * 1000,
                "details": {"error": str(exc)},
            }

    async def shutdown(self) -> None:
        """Clean up Mem0 resources."""
        self._client = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Conversation memory helpers
    # ------------------------------------------------------------------

    async def add_conversation(
        self,
        messages: list[dict],
        user_id: Optional[str] = None,
    ) -> None:
        """
        Add a full conversation for automatic memory extraction.

        param messages: — List of {"role": "user"|"assistant", "content": "..."} dicts.
        """
        if self._client is None:
            return
        uid = user_id or self._default_user_id or "default"
        try:
            self._client.add(
                messages,
                user_id=uid,
                agent_id="claude_clone",
            )
        except Exception:
            logger.exception("Failed to add conversation to Mem0")

    async def get_all_memories(self, user_id: Optional[str] = None) -> list[MemoryEntry]:
        """Get all stored memories for a user."""
        if self._client is None:
            return []
        uid = user_id or self._default_user_id or "default"
        try:
            all_memos = self._client.get_all(user_id=uid)
            entries = []
            for memo in all_memos:
                entries.append(MemoryEntry(
                    id=memo.get("id", str(uuid.uuid4())),
                    content=memo.get("memory", ""),
                    metadata=memo.get("metadata", {}),
                    source="mem0",
                ))
            return entries
        except Exception:
            logger.exception("Failed to get all memories from Mem0")
            return []
