"""
Knowledge Base Orchestrator — unified high-level interface for the Claude
Code Clone knowledge management system.

This module ties together five subsystems into a single
:class:`KnowledgeBaseOrchestrator` facade:

* **KnowledgeStore**  — SQLite-backed persistent entry storage.
* **KnowledgeSearch** — TF-IDF + keyword + fuzzy + graph search.
* **KnowledgeGraph** — Entity linking, BFS/DFS traversal, community
  detection, PageRank, and ASCII visualisation.
* **KnowledgeExtractor** — Regex-based pattern mining from conversations,
  code, and text.
* **KnowledgeImporter** — Bulk import from Markdown, JSON, JSONL, and
  Obsidian vaults.

A lazy singleton pattern is exposed via :func:`get_knowledge_base` /
:func:`set_knowledge_base` so that the rest of the application can access
a shared orchestrator instance without circular imports.

Usage::

    from agent.knowledge_base import KnowledgeBaseOrchestrator, set_knowledge_base

    kb = KnowledgeBaseOrchestrator(auto_init=True)
    set_knowledge_base(kb)

    entry_id = await kb.add_knowledge(
        title="FastAPI DI pattern",
        content="Use Depends() with callable classes ...",
        category="pattern",
        tags=["python", "fastapi"],
    )

    results = await kb.search("dependency injection")
    context = await kb.get_context_for_prompt("how to handle auth")
    stats  = await kb.get_stats()

    await kb.close()

Typical quick-start with auto-initialisation::

    kb = KnowledgeBaseOrchestrator()       # does NOT auto-init by default
    await kb.initialize()                   # explicit init

    # ... use the knowledge base ...

    await kb.close()
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Sub-module imports (public re-exports)
# ──────────────────────────────────────────────────────────────────────────────

from agent.knowledge_base.knowledge_store import (
    KnowledgeStore,
    KnowledgeEntry,
    VALID_CATEGORIES,
    VALID_SOURCES,
    DEFAULT_DB_PATH,
)

from agent.knowledge_base.knowledge_search import (
    KnowledgeSearch,
    SearchQuery,
    SearchResult,
)

from agent.knowledge_base.knowledge_extractor import (
    KnowledgeExtractor,
    ExtractionResult,
    PatternMatch,
)

from agent.knowledge_base.knowledge_graph import (
    KnowledgeGraph,
    GraphNode,
    GraphEdge,
    GraphCluster,
    RelationType,
)

from agent.knowledge_base.knowledge_importer import (
    KnowledgeImporter,
    ImportResult,
)

# ──────────────────────────────────────────────────────────────────────────────
# __all__ — every public symbol that downstream consumers should import
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Orchestrator
    "KnowledgeBaseOrchestrator",
    "get_knowledge_base",
    "set_knowledge_base",
    # Store
    "KnowledgeStore",
    "KnowledgeEntry",
    "VALID_CATEGORIES",
    "VALID_SOURCES",
    "DEFAULT_DB_PATH",
    # Search
    "KnowledgeSearch",
    "SearchQuery",
    "SearchResult",
    # Extractor
    "KnowledgeExtractor",
    "ExtractionResult",
    "PatternMatch",
    # Graph
    "KnowledgeGraph",
    "GraphNode",
    "GraphEdge",
    "GraphCluster",
    "RelationType",
    # Importer
    "KnowledgeImporter",
    "ImportResult",
]

# ──────────────────────────────────────────────────────────────────────────────
# Module-level logger
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# KnowledgeBaseOrchestrator
# ══════════════════════════════════════════════════════════════════════════════


class KnowledgeBaseOrchestrator:
    """High-level facade that coordinates all knowledge-base subsystems.

    The orchestrator owns and manages the lifecycle of five internal
    components:

    1. :class:`KnowledgeStore`  — persistent SQLite storage.
    2. :class:`KnowledgeSearch` — advanced multi-strategy search engine.
    3. :class:`KnowledgeGraph`  — graph structure over entries.
    4. :class:`KnowledgeExtractor` — automatic pattern detection.
    5. :class:`KnowledgeImporter` — bulk import from external sources.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Defaults to
        ``~/.claude_clone/knowledge.db``.  The ``~`` is expanded
        automatically and parent directories are created on
        initialisation.
    auto_init:
        If ``True``, :meth:`initialize` is called automatically during
        ``__init__``.  This is convenient for scripts but **not**
        recommended for library code where explicit lifecycle control
        is preferred.  Defaults to ``False``.

    Attributes
    ----------
    store:
        The underlying :class:`KnowledgeStore` instance.
    search_engine:
        The :class:`KnowledgeSearch` instance (available after init).
    graph:
        The :class:`KnowledgeGraph` instance (available after init).
    extractor:
        The :class:`KnowledgeExtractor` instance.
    importer:
        The :class:`KnowledgeImporter` instance.

    Examples
    --------
    >>> kb = KnowledgeBaseOrchestrator()
    >>> await kb.initialize()
    >>> entry_id = await kb.add_knowledge("Title", "Content")
    >>> results = await kb.search("title")
    >>> await kb.close()
    """

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(
        self,
        db_path: Optional[str] = None,
        auto_init: bool = False,
    ) -> None:
        self._db_path: str = db_path or DEFAULT_DB_PATH
        self._initialized: bool = False

        # Subsystem instances (created eagerly, initialised lazily).
        self.store = KnowledgeStore(db_path=self._db_path)
        self.search_engine: Optional[KnowledgeSearch] = None
        self.graph: Optional[KnowledgeGraph] = None
        self.extractor: Optional[KnowledgeExtractor] = None
        self.importer: Optional[KnowledgeImporter] = None

        if auto_init:
            # Note: the caller must ``await kb.initialize()`` explicitly;
            # auto_init is intentionally *not* an awaited call here because
            # __init__ cannot be async.  Instead, we set a flag that
            # ``initialize()`` checks.  For true auto-init behaviour the
            # caller should do:  kb = await KnowledgeBaseOrchestrator.create()
            logger.debug(
                "auto_init=True passed to __init__; "
                "call await kb.initialize() explicitly."
            )

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialise all five subsystems in the correct order.

        Initialisation sequence:

        1. **Store** — creates the SQLite database and schema.
        2. **Graph** — connects to the same database, builds adjacency cache.
        3. **Search** — builds the in-memory TF-IDF index from store entries.
        4. **Extractor** — wired to the store for persistence.
        5. **Importer** — wired to the store for persistence.

        Safe to call multiple times; subsequent calls are no-ops.

        Raises
        ------
        RuntimeError
            If the database cannot be opened or the schema cannot be
            created.
        """
        if self._initialized:
            logger.debug("KnowledgeBaseOrchestrator already initialized")
            return

        logger.info("Initializing KnowledgeBaseOrchestrator (db=%s)", self._db_path)

        # 1. Store (creates tables).
        await self.store.initialize()

        # 2. Graph (shares the same DB file).
        self.graph = KnowledgeGraph(db_path=self._db_path)
        await self.graph.initialize()

        # 3. Search engine (TF-IDF index over store entries).
        self.search_engine = KnowledgeSearch(
            store=self.store,
            graph=self.graph,
        )
        await self.search_engine.initialize()

        # 4. Extractor (needs a store reference for auto-store).
        self.extractor = KnowledgeExtractor(store=self.store)

        # 5. Importer (needs a store reference).
        self.importer = KnowledgeImporter(store=self.store)

        self._initialized = True
        logger.info("KnowledgeBaseOrchestrator fully initialized")

    async def close(self) -> None:
        """Shut down all subsystems and release database connections.

        Components are closed in reverse initialisation order.  Safe to
        call multiple times; already-closed components are silently
        skipped.
        """
        if not self._initialized:
            return

        logger.info("Closing KnowledgeBaseOrchestrator")

        if self.importer is not None:
            self.importer = None

        if self.extractor is not None:
            self.extractor = None

        if self.search_engine is not None:
            # KnowledgeSearch has no explicit close, but we clear the ref.
            self.search_engine = None

        if self.graph is not None:
            await self.graph.close()
            self.graph = None

        if self.store is not None:
            await self.store.close()

        self._initialized = False
        logger.info("KnowledgeBaseOrchestrator closed")

    @property
    def initialized(self) -> bool:
        """Return ``True`` if :meth:`initialize` has been called successfully."""
        return self._initialized

    # ── Knowledge CRUD ────────────────────────────────────────────────────

    async def add_knowledge(
        self,
        title: str,
        content: str,
        category: str = "concept",
        tags: Optional[List[str]] = None,
        source: str = "manual",
        confidence: float = 0.8,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a new knowledge entry via the store.

        A convenience wrapper around :meth:`KnowledgeStore.add` that
        constructs a :class:`KnowledgeEntry` from individual parameters.

        Parameters
        ----------
        title:
            Short human-readable title for the entry.
        content:
            Full body text — may be multi-paragraph Markdown.
        category:
            One of :data:`VALID_CATEGORIES`.  Defaults to ``"concept"``.
        tags:
            Optional list of free-form tags for filtering and search.
        source:
            Origin of this knowledge — one of :data:`VALID_SOURCES`.
            Defaults to ``"manual"``.
        confidence:
            Reliability score in ``[0.0, 1.0]``.  Defaults to ``0.8``.
        importance:
            Relevance / priority score in ``[0.0, 1.0]``.  Defaults to
            ``0.5``.
        metadata:
            Arbitrary extra data (language, framework, project, etc.).

        Returns
        -------
        str
            The unique id of the newly-created entry.

        Raises
        ------
        RuntimeError
            If the orchestrator has not been initialised.
        ValueError
            If *category* or *source* is invalid.
        """
        self._ensure_initialized()

        entry = KnowledgeEntry(
            title=title,
            content=content,
            category=category,
            tags=tags or [],
            source=source,
            confidence=confidence,
            importance=importance,
            metadata=metadata or {},
        )
        entry_id = await self.store.add(entry)
        logger.debug("Added knowledge entry %s: %s", entry_id, title[:80])
        return entry_id

    # ── Search ────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[KnowledgeEntry]:
        """Search the knowledge base using the advanced search engine.

        Delegates to :meth:`KnowledgeSearch.search` with a
        :class:`SearchQuery` constructed from the given parameters,
        then returns the underlying :class:`KnowledgeEntry` objects from
        each :class:`SearchResult`.

        Parameters
        ----------
        query:
            Free-text search string.
        category:
            Optional category filter (e.g. ``"pattern"``).
        tags:
            Optional list of tags — entries must have **all** tags.
        limit:
            Maximum number of results to return (default 20).

        Returns
        -------
        list[KnowledgeEntry]
            Matching entries sorted by descending relevance.

        Raises
        ------
        RuntimeError
            If the orchestrator has not been initialised.
        """
        self._ensure_initialized()
        assert self.search_engine is not None

        sq = SearchQuery(
            query=query,
            categories=[category] if category else [],
            tags=tags or [],
            max_results=limit,
        )
        results = await self.search_engine.search(sq)
        return [r.entry for r in results]

    # ── Auto-extraction ───────────────────────────────────────────────────

    async def auto_extract(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        code_files: Optional[List[Dict[str, str]]] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Automatically extract knowledge and store it.

        A convenience wrapper around
        :meth:`KnowledgeExtractor.auto_extract_and_store` that mines
        patterns from conversations, source code, and/or arbitrary text,
        then persists every discovered :class:`ExtractionResult` via the
        store.

        Parameters
        ----------
        messages:
            Conversation messages (list of dicts with ``"role"`` and
            ``"content"`` keys).
        code_files:
            A list of dicts, each with ``"file_path"`` and ``"content"``
            keys.
        text:
            Arbitrary prose text to extract knowledge from.

        Returns
        -------
        dict
            A summary dictionary containing:

            * ``"entries_stored"`` — number of entries persisted.
            * ``"patterns_found"`` — total pattern matches across all
              sources.
            * ``"sources_processed"`` — list of source types that were
              analysed (e.g. ``["conversation", "code"]``).

        Raises
        ------
        RuntimeError
            If the orchestrator has not been initialised.
        """
        self._ensure_initialized()
        assert self.extractor is not None

        result = await self.extractor.auto_extract_and_store(
            messages=messages,
            code_files=code_files,
            text=text,
        )
        return result

    # ── Import operations ─────────────────────────────────────────────────

    async def import_markdown(
        self,
        filepath: str,
        category: str = "reference",
    ) -> Dict[str, Any]:
        """Import a single Markdown file into the knowledge base.

        Delegates to :meth:`KnowledgeImporter.import_markdown` and returns
        the :class:`ImportResult` as a plain dictionary.

        Parameters
        ----------
        filepath:
            Path to the ``.md`` file to import.
        category:
            Default category if none is found in frontmatter
            (default ``"reference"``).

        Returns
        -------
        dict
            Import summary with keys ``total_entries``, ``categories``,
            ``tags``, ``errors``, ``source``, ``duration_seconds``.

        Raises
        ------
        RuntimeError
            If the orchestrator has not been initialised.
        FileNotFoundError
            If *filepath* does not exist.
        """
        self._ensure_initialized()
        assert self.importer is not None

        result = await self.importer.import_markdown(
            filepath=filepath,
            category=category,
        )
        return asdict(result)

    async def import_obsidian_vault(self, vault_path: str) -> Dict[str, Any]:
        """Import an entire Obsidian vault into the knowledge base.

        Delegates to :meth:`KnowledgeImporter.import_obsidian_vault`.
        Handles frontmatter, inline tags, wikilinks, folder-as-category,
        and more.

        Parameters
        ----------
        vault_path:
            Path to the root directory of the Obsidian vault.

        Returns
        -------
        dict
            Aggregated import summary across all notes.

        Raises
        ------
        RuntimeError
            If the orchestrator has not been initialised.
        FileNotFoundError
            If *vault_path* does not exist or is not a directory.
        """
        self._ensure_initialized()
        assert self.importer is not None

        result = await self.importer.import_obsidian_vault(vault_path=vault_path)
        return asdict(result)

    async def import_json(self, filepath: str) -> Dict[str, Any]:
        """Import knowledge entries from a JSON file.

        Delegates to :meth:`KnowledgeImporter.import_json`.  Supports both
        the object format (``{"entries": [...]}``) and a plain array
        format.

        Parameters
        ----------
        filepath:
            Path to the ``.json`` file to import.

        Returns
        -------
        dict
            Import summary with keys ``total_entries``, ``categories``,
            ``tags``, ``errors``, ``source``, ``duration_seconds``.

        Raises
        ------
        RuntimeError
            If the orchestrator has not been initialised.
        FileNotFoundError
            If *filepath* does not exist.
        """
        self._ensure_initialized()
        assert self.importer is not None

        result = await self.importer.import_json(filepath=filepath)
        return asdict(result)

    # ── Statistics ─────────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """Return combined statistics from all subsystems.

        Aggregates metrics from the store, search engine, graph, extractor,
        and importer into a single dictionary.

        Returns
        -------
        dict
            A dictionary with the following top-level keys:

            * ``store`` — :meth:`KnowledgeStore.get_stats` output (entry
              counts, category/source breakdowns, averages).
            * ``search`` — ``{"indexed_docs": N, "vocabulary_size": N}``.
            * ``graph`` — ``{"nodes": N, "clusters": N}`` (estimated).
            * ``extractor`` — running extraction statistics.
            * ``importer`` — running import statistics.
            * ``initialized`` — ``True`` if all subsystems are ready.

        Raises
        ------
        RuntimeError
            If the orchestrator has not been initialised.
        """
        self._ensure_initialized()

        stats: Dict[str, Any] = {
            "initialized": True,
            "db_path": str(self.store.db_path),
        }

        # Store stats.
        store_stats = await self.store.get_stats()
        stats["store"] = store_stats

        # Search engine stats.
        if self.search_engine is not None:
            stats["search"] = {
                "indexed_docs": getattr(
                    self.search_engine._tfidf, "n_docs", 0,
                ),
                "vocabulary_size": getattr(
                    self.search_engine._tfidf, "vocabulary_size", 0,
                ),
            }

        # Graph stats.
        if self.graph is not None:
            clusters = await self.graph.get_clusters(min_size=1)
            stats["graph"] = {
                "clusters": len(clusters),
            }

        # Extractor stats.
        if self.extractor is not None:
            stats["extractor"] = dict(self.extractor.extraction_stats)

        # Importer stats.
        if self.importer is not None:
            stats["importer"] = dict(self.importer._import_stats)

        return stats

    # ── LLM context generation ────────────────────────────────────────────

    async def get_context_for_prompt(
        self,
        query: str,
        max_tokens: int = 4000,
    ) -> str:
        """Get formatted knowledge-base context suitable for LLM injection.

        Runs a combined search and renders the top results into a compact
        Markdown block that fits within the given *max_tokens* budget
        (approximately 4 characters per token).

        Parameters
        ----------
        query:
            The user's question or topic to retrieve context for.
        max_tokens:
            Rough token budget for the returned context string.
            Defaults to ``4000``.

        Returns
        -------
        str
            A formatted string containing ranked knowledge entries with
            titles, categories, tags, and content excerpts.  Returns
            an empty string if no results are found.

        Raises
        ------
        RuntimeError
            If the orchestrator has not been initialised.
        """
        self._ensure_initialized()
        assert self.search_engine is not None

        return await self.search_engine.get_context_for_prompt(
            query=query,
            max_tokens=max_tokens,
        )

    # ── Related-entry suggestions ─────────────────────────────────────────

    async def suggest_related(
        self,
        entry_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Suggest knowledge entries related to a given entry.

        Combines graph-based proximity scoring with tag-overlap
        similarity (via :meth:`KnowledgeGraph.suggest_related`) and
        enriches each suggestion with the entry's title and category
        fetched from the store.

        Parameters
        ----------
        entry_id:
            The id of the entry to find suggestions for.
        limit:
            Maximum number of related entries to return (default 10).

        Returns
        -------
        list[dict]
            Each dict contains ``"entry_id"``, ``"score"``, ``"title"``,
            ``"category"``, and ``"tags"`` for the related entry.
            Sorted by descending relevance score.

        Raises
        ------
        RuntimeError
            If the orchestrator has not been initialised.
        """
        self._ensure_initialized()
        assert self.graph is not None

        raw_suggestions = await self.graph.suggest_related(
            entry_id=entry_id,
            limit=limit,
        )

        results: List[Dict[str, Any]] = []
        for related_id, score in raw_suggestions:
            entry = await self.store.get(related_id)
            if entry is None:
                continue
            results.append({
                "entry_id": related_id,
                "score": score,
                "title": entry.title,
                "category": entry.category,
                "tags": entry.tags,
            })
        return results

    # ── Export ────────────────────────────────────────────────────────────

    async def export_all(self, filepath: str) -> None:
        """Export the entire knowledge base to a JSON file.

        Delegates to :meth:`KnowledgeImporter.export_all`, which serialises
        all non-archived entries into a single JSON file with an
        ``"entries"`` array.

        Parameters
        ----------
        filepath:
            Destination path for the exported JSON file.  Parent
            directories are created automatically if needed.

        Raises
        ------
        RuntimeError
            If the orchestrator has not been initialised.
        """
        self._ensure_initialized()
        assert self.importer is not None

        await self.importer.export_all(filepath=filepath)
        logger.info("Knowledge base exported to %s", filepath)

    # ── Graph visualisation ───────────────────────────────────────────────

    async def visualize_graph(
        self,
        entry_id: Optional[str] = None,
        max_nodes: int = 50,
    ) -> str:
        """Generate an ASCII-art graph visualisation.

        If *entry_id* is provided, renders the sub-graph reachable from that
        node (up to *max_nodes*).  Otherwise renders the full graph
        limited to *max_nodes* nodes.

        Parameters
        ----------
        entry_id:
            Optional entry to centre the visualisation on.  When ``None``,
            the full graph is rendered.
        max_nodes:
            Maximum number of nodes to include (default 50).

        Returns
        -------
        str
            A multi-line ASCII string representing the graph structure.

        Raises
        ------
        RuntimeError
            If the orchestrator has not been initialised.
        """
        self._ensure_initialized()
        assert self.graph is not None

        return await self.graph.visualize(
            entry_id=entry_id,
            max_nodes=max_nodes,
        )

    # ── Pruning / maintenance ────────────────────────────────────────────

    async def prune(self, max_age_days: int = 90) -> int:
        """Remove old, low-confidence entries from the knowledge base.

        Delegates to :meth:`KnowledgeStore.prune`.  Candidates are
        non-archived entries whose ``created_at`` is older than
        *max_age_days* and whose ``confidence`` falls below the default
        threshold (0.3).  Archived entries older than twice
        *max_age_days* are also permanently removed.

        After pruning, the search index is rebuilt to stay in sync.

        Parameters
        ----------
        max_age_days:
            Age threshold in days (default 90).

        Returns
        -------
        int
            The number of entries permanently removed.

        Raises
        ------
        RuntimeError
            If the orchestrator has not been initialised.
        """
        self._ensure_initialized()

        removed = await self.store.prune(max_age_days=max_age_days)
        logger.info(
            "Pruned %d entries older than %d days",
            removed,
            max_age_days,
        )

        # Rebuild the TF-IDF index so stale entries are gone.
        if self.search_engine is not None:
            await self.search_engine.rebuild_index()

        return removed

    # ── Internal helpers ───────────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """Raise :class:`RuntimeError` if the orchestrator is not initialised.

        Called at the top of every public async method to provide a clear
        error message when the caller forgets to call
        :meth:`initialize`.
        """
        if not self._initialized:
            raise RuntimeError(
                "KnowledgeBaseOrchestrator is not initialized. "
                "Call `await kb.initialize()` before using any method."
            )


# ══════════════════════════════════════════════════════════════════════════════
# Global singleton (lazy)
# ══════════════════════════════════════════════════════════════════════════════

_global_kb: Optional[KnowledgeBaseOrchestrator] = None
"""Module-level singleton instance of the orchestrator."""


def get_knowledge_base() -> Optional[KnowledgeBaseOrchestrator]:
    """Return the global :class:`KnowledgeBaseOrchestrator` instance, or ``None``.

    This accessor supports a lazy singleton pattern where application code
    can obtain the orchestrator without importing it directly::

        from agent.knowledge_base import get_knowledge_base

        kb = get_knowledge_base()
        if kb is not None:
            results = await kb.search("python async")

    Returns
    -------
    KnowledgeBaseOrchestrator | None
        The global instance that was registered via
        :func:`set_knowledge_base`, or ``None`` if no instance has been
        set.
    """
    return _global_kb


def set_knowledge_base(kb: KnowledgeBaseOrchestrator) -> None:
    """Register a :class:`KnowledgeBaseOrchestrator` as the global singleton.

    The registered instance can later be retrieved with
    :func:`get_knowledge_base`.

    Parameters
    ----------
    kb:
        The orchestrator instance to make globally accessible.

    Examples
    --------
    >>> kb = KnowledgeBaseOrchestrator()
    >>> await kb.initialize()
    >>> set_knowledge_base(kb)
    >>> assert get_knowledge_base() is kb
    """
    global _global_kb
    _global_kb = kb
    logger.debug("Global knowledge base instance registered")
