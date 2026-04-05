"""
Memory Manager — Orchestrates built-in memory and at most one external plugin.

The MemoryManager is the single integration point for all memory operations.
It manages a built-in memory provider (file-based) and optionally one
external plugin provider, coordinating search, storage, retrieval, and
context augmentation across both.

Pre-turn: prefetches relevant memory context for prompt injection.
Post-turn: syncs new information from the conversation into memory.

Usage
-----
    manager = MemoryManager()
    await manager.initialize()

    # Pre-turn: get context for the current prompt
    context = await manager.prefetch("How do I deploy this project?")

    # Post-turn: save the exchange
    await manager.sync_turn(
        user_message="How do I deploy?",
        assistant_response="You can deploy with...",
        metadata={"model": "claude-sonnet-4"},
    )
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hermes.core.memory_provider import MemoryEntry, MemoryProvider

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for the MemoryManager."""

    enabled: bool = True
    max_context_tokens: int = 4000
    auto_save: bool = True
    prefetch_enabled: bool = True
    external_provider: Optional[MemoryProvider] = None
    builtin_memory_dir: Optional[str] = None


class MemoryManager:
    """
    Orchestrates memory operations across built-in and optional plugin providers.

    The manager wraps a built-in ``MemoryProvider`` and at most one external
    plugin ``MemoryProvider``. All memory operations (search, store, retrieve,
    context augmentation) go through this single interface.

    Parameters
    ----------
    config:
        Configuration options. If ``None``, uses defaults.
    builtin_provider:
        Explicit built-in provider instance. If ``None``, uses
        ``BuiltinMemoryProvider`` (lazily initialized).
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        builtin_provider: Optional[MemoryProvider] = None,
    ) -> None:
        self._config = config or MemoryConfig()
        self._builtin: Optional[MemoryProvider] = builtin_provider
        self._external: Optional[MemoryProvider] = self._config.external_provider
        self._initialized = False

    # ── Initialization ────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Initialize both memory providers.

        If the built-in provider was not explicitly passed, lazily creates
        a ``BuiltinMemoryProvider``.
        """
        if self._initialized:
            return

        if self._builtin is None:
            try:
                from hermes.core.builtin_memory import BuiltinMemoryProvider

                memory_dir = self._config.builtin_memory_dir
                self._builtin = BuiltinMemoryProvider(memory_dir=memory_dir)
            except Exception as e:
                logger.error("Failed to initialize built-in memory provider: %s", e)
                self._builtin = None

        if self._builtin is not None:
            try:
                await self._builtin.initialize()
                logger.info("Built-in memory provider initialized")
            except Exception as e:
                logger.error("Built-in memory init error: %s", e)
                self._builtin = None

        if self._external is not None:
            try:
                await self._external.initialize()
                logger.info("External memory provider initialized")
            except Exception as e:
                logger.error("External memory init error: %s", e)
                self._external = None

        self._initialized = True

    async def close(self) -> None:
        """Close both memory providers and release resources."""
        for provider in (self._builtin, self._external):
            if provider is not None:
                try:
                    await provider.close()
                except Exception as e:
                    logger.warning("Error closing memory provider: %s", e)
        self._initialized = False

    @property
    def is_available(self) -> bool:
        """Whether at least one memory provider is active."""
        return self._builtin is not None or self._external is not None

    @property
    def has_external(self) -> bool:
        """Whether an external plugin provider is active."""
        return self._external is not None

    # ── Pre-turn: Context prefetch ────────────────────────────────────────

    async def prefetch(
        self,
        query: str,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Retrieve relevant memory context for the current user prompt.

        Called before sending the prompt to the model. Searches both
        providers and merges results into a single context block.

        Parameters
        ----------
        query:
            The user's message or prompt.
        max_tokens:
            Token budget for the context. Defaults to config value.
        session_id:
            Optional session filter.

        Returns
        -------
        str
            Formatted context block for prompt injection, or empty string.
        """
        if not self._config.enabled or not self._config.prefetch_enabled:
            return ""
        if not self.is_available:
            return ""

        max_tok = max_tokens or self._config.max_context_tokens
        # Split budget between providers if both are active
        if self._builtin is not None and self._external is not None:
            builtin_budget = max_tok // 2
            external_budget = max_tok - builtin_budget
        else:
            builtin_budget = max_tok
            external_budget = max_tok

        parts: List[str] = []

        # Search built-in memory
        if self._builtin is not None:
            try:
                ctx = await self._builtin.get_context_for_prompt(
                    query, max_tokens=builtin_budget, session_id=session_id,
                )
                if ctx:
                    parts.append(ctx)
            except Exception as e:
                logger.warning("Built-in memory prefetch error: %s", e)

        # Search external memory
        if self._external is not None:
            try:
                ctx = await self._external.get_context_for_prompt(
                    query, max_tokens=external_budget, session_id=session_id,
                )
                if ctx:
                    parts.append(ctx)
            except Exception as e:
                logger.warning("External memory prefetch error: %s", e)

        return "\n\n".join(parts)

    # ── Post-turn: Sync ───────────────────────────────────────────────────

    async def sync_turn(
        self,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Save a conversation turn to memory.

        Called after the agent finishes processing a turn. Stores the user
        message and assistant response as memory entries.

        Parameters
        ----------
        user_message:
            The user's input message.
        assistant_response:
            The agent's response text.
        metadata:
            Optional metadata (model name, token counts, etc.).
        session_id:
            Optional session identifier.
        """
        if not self._config.enabled or not self._config.auto_save:
            return
        if not self.is_available:
            return

        now = datetime.now(timezone.utc).isoformat()

        # Save user message
        user_entry = MemoryEntry(
            id=uuid.uuid4().hex[:16],
            content=user_message,
            role="user",
            timestamp=now,
            source="builtin",
            metadata=metadata or {},
            session_id=session_id,
        )

        # Save assistant response
        assistant_entry = MemoryEntry(
            id=uuid.uuid4().hex[:16],
            content=assistant_response[:5000],  # Truncate very long responses
            role="assistant",
            timestamp=now,
            source="builtin",
            metadata=metadata or {},
            session_id=session_id,
        )

        # Store in built-in
        if self._builtin is not None:
            try:
                await self._builtin.store(user_entry)
                await self._builtin.store(assistant_entry)
            except Exception as e:
                logger.warning("Built-in memory sync error: %s", e)

        # Store in external
        if self._external is not None:
            try:
                await self._external.store(user_entry)
                await self._external.store(assistant_entry)
            except Exception as e:
                logger.warning("External memory sync error: %s", e)

    # ── Direct memory operations ──────────────────────────────────────────

    async def search(
        self,
        query: str,
        limit: int = 10,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """
        Search across memory providers.

        Parameters
        ----------
        query:
            Search query text.
        limit:
            Maximum results per provider.
        session_id:
            Optional session filter.
        tags:
            Optional tag filter.
        source:
            If ``"builtin"`` or ``"external"``, search only that provider.
            If ``None``, search both and merge results.

        Returns
        -------
        list[MemoryEntry]
            Deduplicated results sorted by relevance.
        """
        results: List[MemoryEntry] = []
        seen_ids: set = set()

        providers: List[MemoryProvider] = []
        if source is None or source == "builtin":
            if self._builtin is not None:
                providers.append(self._builtin)
        if source is None or source == "external":
            if self._external is not None:
                providers.append(self._external)

        for provider in providers:
            try:
                entries = await provider.search(
                    query, limit=limit, session_id=session_id, tags=tags,
                )
                for entry in entries:
                    if entry.id not in seen_ids:
                        seen_ids.add(entry.id)
                        results.append(entry)
            except Exception as e:
                logger.warning("Memory search error (%s): %s", provider.__class__.__name__, e)

        # Sort by timestamp (newest first) and respect limit
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    async def store(self, entry: MemoryEntry, target: Optional[str] = None) -> str:
        """
        Store a memory entry.

        Parameters
        ----------
        entry:
            The memory entry to store.
        target:
            ``"builtin"``, ``"external"``, or ``None`` (both).

        Returns
        -------
        str
            The stored entry's ID.
        """
        stored_id = entry.id

        if target is None or target == "builtin":
            if self._builtin is not None:
                try:
                    stored_id = await self._builtin.store(entry)
                except Exception as e:
                    logger.warning("Built-in store error: %s", e)

        if target is None or target == "external":
            if self._external is not None:
                try:
                    ext_id = await self._external.store(entry)
                    if target == "external":
                        stored_id = ext_id
                except Exception as e:
                    logger.warning("External store error: %s", e)

        return stored_id

    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID, checking both providers."""
        if self._builtin is not None:
            try:
                entry = await self._builtin.get(entry_id)
                if entry is not None:
                    return entry
            except Exception:
                pass

        if self._external is not None:
            try:
                return await self._external.get(entry_id)
            except Exception:
                pass

        return None

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry from all providers."""
        deleted = False
        for provider in (self._builtin, self._external):
            if provider is not None:
                try:
                    if await provider.delete(entry_id):
                        deleted = True
                except Exception:
                    pass
        return deleted

    # ── System prompt augmentation ────────────────────────────────────────

    async def augment_system_prompt(
        self,
        system_prompt: str,
        query: str,
        max_tokens: int = 4000,
    ) -> str:
        """
        Augment a system prompt with relevant memory context.

        Parameters
        ----------
        system_prompt:
            The base system prompt.
        query:
            The user query for context retrieval.
        max_tokens:
            Maximum tokens for the memory context injection.

        Returns
        -------
        str
            The augmented system prompt.
        """
        memory_ctx = await self.prefetch(query, max_tokens=max_tokens)
        if not memory_ctx:
            return system_prompt
        return f"{system_prompt}\n\n{memory_ctx}"

    # ── Diagnostics ───────────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        """Return health status for all memory providers."""
        result: Dict[str, Any] = {
            "enabled": self._config.enabled,
            "builtin": None,
            "external": None,
        }
        if self._builtin is not None:
            try:
                result["builtin"] = await self._builtin.health_check()
            except Exception as e:
                result["builtin"] = {"status": "unhealthy", "error": str(e)}
        if self._external is not None:
            try:
                result["external"] = await self._external.health_check()
            except Exception as e:
                result["external"] = {"status": "unhealthy", "error": str(e)}
        return result

    async def get_stats(self) -> Dict[str, Any]:
        """Return statistics from all active providers."""
        stats: Dict[str, Any] = {}
        if self._builtin is not None:
            try:
                stats["builtin"] = await self._builtin.get_stats()
            except Exception:
                stats["builtin"] = {"error": "failed to retrieve"}
        if self._external is not None:
            try:
                stats["external"] = await self._external.get_stats()
            except Exception:
                stats["external"] = {"error": "failed to retrieve"}
        return stats
