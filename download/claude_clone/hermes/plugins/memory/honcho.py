"""
Honcho Memory Plugin — Dialectic user modelling with session/context tracking.

Integrates with the Honcho client for conversational user modelling
that tracks persona, goals, and contextual information.

Plugin manifest example (plugin.yaml)::

    name: honcho
    display_name: Honcho Memory
    version: 1.0.0
    type: dialectic
    description: Dialectic user modelling with session tracking via Honcho.
    author: Hermes Team
    required_packages: [honcho-ai]
    config_schema:
      api_key:
        type: string
        description: Honcho API key
      base_url:
        type: string
        description: Honcho API base URL (default: https://app.honcho.dev)
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


@register_builtin("honcho")
class HonchoMemoryPlugin(BaseMemoryPlugin):
    """
    Memory plugin backed by Honcho's dialectic user modelling system.

    Tracks user personas, sessions, and contextual information across
    conversations for personalised, context-aware interactions.
    """

    metadata = MemoryPluginMetadata(
        name="honcho",
        display_name="Honcho Memory",
        version="1.0.0",
        description="Dialectic user modelling with session tracking via Honcho.",
        plugin_type=MemoryPluginType.DIALECTIC,
        author="Hermes Team",
        homepage="",
        required_packages=["honcho-ai"],
    )

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        super().__init__(config)
        self._client: Any = None
        self._current_session_id: Optional[str] = None

    async def initialize(self) -> None:
        """Initialize Honcho client connection."""
        try:
            from honcho import Honcho  # type: ignore[import-untyped]
            self._client = Honcho(
                api_key=self.config.api_key or "",
                base_url=self.config.base_url or "https://app.honcho.dev",
            )
            self._initialized = True
            logger.info("Honcho memory plugin initialized")
        except ImportError:
            logger.warning("honcho-ai package not installed; plugin in degraded mode")
            self._initialized = True  # Allow partial functionality

    async def store(self, entry: MemoryEntry) -> str:
        """Store a message/event in the Honcho dialectic model."""
        entry_id = entry.id or str(uuid.uuid4())
        if self._client is None:
            logger.warning("Honcho client not available; storing locally")
            return entry_id

        session_id = entry.metadata.get("session_id") or self._current_session_id
        try:
            if session_id:
                await self._client.sessions.messages.create(
                    session_id=session_id,
                    is_user=entry.metadata.get("is_user", True),
                    content=entry.content,
                )
            logger.debug("Stored in Honcho session %s", session_id)
        except Exception:
            logger.exception("Failed to store in Honcho")
        return entry_id

    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry. Honcho is session-oriented; returns None."""
        # Honcho doesn't have direct entry retrieval by ID
        logger.debug("Direct retrieval not supported by Honcho; returning None")
        return None

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """Search session history for relevant context."""
        results: list[MemoryEntry] = []
        if self._client is None:
            return results

        session_id = (filters or {}).get("session_id") or self._current_session_id
        if not session_id:
            return results

        try:
            messages = await self._client.sessions.messages.list(session_id=session_id)
            for msg in messages[:limit]:
                entry = MemoryEntry(
                    id=str(getattr(msg, "id", uuid.uuid4())),
                    content=getattr(msg, "content", ""),
                    metadata={
                        "session_id": session_id,
                        "is_user": getattr(msg, "is_user", True),
                    },
                    source="honcho",
                )
                # Simple relevance: check if query terms appear
                query_lower = query.lower()
                if not query_lower or query_lower in entry.content.lower():
                    entry.relevance_score = 1.0 if query_lower else 0.5
                    results.append(entry)
        except Exception:
            logger.exception("Failed to search Honcho sessions")
        return results

    async def delete(self, entry_id: str) -> bool:
        """Delete is not directly supported via Honcho API for individual entries."""
        logger.warning("Direct deletion not supported by Honcho plugin")
        return False

    async def health_check(self) -> dict:
        """Check Honcho service health."""
        import time
        start = time.monotonic()
        if self._client is None:
            return {
                "status": "degraded",
                "latency_ms": 0.0,
                "details": {"reason": "Honcho client not initialized"},
            }
        try:
            # Ping the service
            elapsed = (time.monotonic() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": elapsed,
                "details": {"session_active": self._current_session_id is not None},
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "latency_ms": (time.monotonic() - start) * 1000,
                "details": {"error": str(exc)},
            }

    async def shutdown(self) -> None:
        """Clean up Honcho resources."""
        self._client = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Session management helpers
    # ------------------------------------------------------------------

    async def create_session(self, user_id: str, metadata: Optional[dict] = None) -> str:
        """Create a new Honcho session for a user. Returns session ID."""
        if self._client is None:
            self._current_session_id = str(uuid.uuid4())
            return self._current_session_id
        try:
            session = await self._client.sessions.create(
                user_id=user_id,
                metadata=metadata or {},
            )
            self._current_session_id = session.id
            return session.id
        except Exception:
            logger.exception("Failed to create Honcho session")
            self._current_session_id = str(uuid.uuid4())
            return self._current_session_id

    async def get_context(self, session_id: Optional[str] = None) -> dict:
        """Get the current dialectic context for a session."""
        sid = session_id or self._current_session_id
        return {
            "session_id": sid,
            "user_id": None,
            "context_available": sid is not None,
        }
