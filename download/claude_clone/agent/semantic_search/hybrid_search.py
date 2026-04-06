"""
Semantic Search Engine and Hybrid Search Engine.

Combines the lightweight :class:`EmbeddingEngine` with the
:class:`EmbeddingCache` vector store to provide a full semantic search
pipeline.  The :class:`HybridSearchEngine` further merges semantic results
with an existing keyword/TF-IDF search backend using **Reciprocal Rank
Fusion** (RRF).

Data classes
------------
- :class:`SearchResult` — single semantic search result
- :class:`HybridResult`  — merged hybrid search result

Engines
-------
- :class:`SemanticSearchEngine` — embeddings-only search
- :class:`HybridSearchEngine`   — semantic + keyword fusion

Usage::

    from agent.semantic_search import SemanticSearchEngine, HybridSearchEngine

    # Pure semantic search
    engine = SemanticSearchEngine()
    await engine.initialize()
    await engine.index_document("doc-1", "How to deploy to AWS", {"source": "cli"})
    results = await engine.search("cloud deployment")
    await engine.close()

    # Hybrid search (semantic + keyword)
    hybrid = HybridSearchEngine(
        knowledge_base_search_fn=my_kb_search,
    )
    await hybrid.initialize()
    results = await hybrid.search("fastapi auth middleware")
    await hybrid.close()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from agent.semantic_search.embedder import EmbeddingEngine
from agent.semantic_search.vector_store import EmbeddingCache

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: Default embedding dimensionality used by the engines.
DEFAULT_EMBEDDING_DIM: int = 256

#: Default database path for the embedding vector store.
DEFAULT_DB_PATH: str = "~/.claude_clone/embeddings.db"

#: Default number of results returned by :meth:`SemanticSearchEngine.search`.
DEFAULT_SEARCH_LIMIT: int = 10

#: RRF constant *k* — controls the strength of rank-based fusion.
#: A typical value is 60 (from the original RRF paper).
_RRF_K: int = 60


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single result from semantic search.

    Attributes
    ----------
    doc_id:
        Unique document identifier.
    text:
        The original document text.
    score:
        Cosine similarity score in [0.0, 1.0].
    metadata:
        Arbitrary key-value metadata attached to the document.
    match_type:
        Always ``"semantic"`` for results from
        :class:`SemanticSearchEngine`.
    """

    doc_id: str = ""
    text: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    match_type: str = "semantic"


@dataclass
class HybridResult:
    """A single result from hybrid (semantic + keyword) search.

    Attributes
    ----------
    doc_id:
        Unique document identifier.
    text:
        The original document text.
    semantic_score:
        Cosine similarity score from the semantic engine.
    keyword_score:
        Score from the keyword/TF-IDF search backend.
    combined_score:
        Final fused score produced by Reciprocal Rank Fusion.
    metadata:
        Arbitrary key-value metadata attached to the document.
    match_type:
        Always ``"hybrid"`` for results from
        :class:`HybridSearchEngine`.
    """

    doc_id: str = ""
    text: str = ""
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    match_type: str = "hybrid"


# ──────────────────────────────────────────────────────────────────────────────
# Type alias for the optional keyword search callback
# ──────────────────────────────────────────────────────────────────────────────

# The keyword search callback should be an async callable that accepts
# a query string and returns a list of dicts with at least ``doc_id``,
# ``text``, and ``score`` keys.
KeywordSearchFn = Callable[
    [str, int],
    Coroutine[Any, Any, List[Dict[str, Any]]],
]


# ──────────────────────────────────────────────────────────────────────────────
# SemanticSearchEngine
# ──────────────────────────────────────────────────────────────────────────────

class SemanticSearchEngine:
    """Semantic search engine powered by embedding vectors.

    Combines :class:`EmbeddingEngine` (for encoding text into vectors)
    with :class:`EmbeddingCache` (for persistent vector storage and
    retrieval) to provide a complete semantic search pipeline.

    Parameters
    ----------
    dim:
        Embedding dimensionality (default 256).
    db_path:
        Path to the SQLite database for the embedding cache.

    Examples
    --------
    >>> engine = SemanticSearchEngine()
    >>> await engine.initialize()
    >>> await engine.index_document("d1", "Python async patterns", {})
    >>> results = await engine.search("asyncio await")
    >>> await engine.close()
    """

    def __init__(
        self,
        dim: int = DEFAULT_EMBEDDING_DIM,
        db_path: str = DEFAULT_DB_PATH,
    ) -> None:
        self._embedder = EmbeddingEngine(dim=dim)
        self._cache = EmbeddingCache(db_path=db_path)
        self._initialised: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialise the embedding cache and mark the engine as ready.

        Must be called once before any search or indexing operations.
        """
        await self._cache.initialize()
        self._initialised = True
        logger.info("SemanticSearchEngine initialized (dim=%d)", self._embedder.dim)

    async def close(self) -> None:
        """Close the underlying embedding cache connection.

        Safe to call multiple times.
        """
        await self._cache.close()
        self._initialised = False
        logger.info("SemanticSearchEngine closed")

    def _ensure_initialised(self) -> None:
        """Raise ``RuntimeError`` if the engine has not been initialised."""
        if not self._initialised:
            raise RuntimeError(
                "SemanticSearchEngine is not initialized. "
                "Call `await engine.initialize()` first."
            )

    # ── Indexing ───────────────────────────────────────────────────────────

    async def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Encode and store a document embedding.

        The document text is encoded into a dense vector using the
        internal :class:`EmbeddingEngine` and then persisted via the
        :class:`EmbeddingCache`.

        Parameters
        ----------
        doc_id:
            Unique identifier for the document.
        text:
            The document text to embed and store.
        metadata:
            Optional arbitrary key-value metadata.
        """
        self._ensure_initialised()

        embedding = self._embedder.encode(text)
        await self._cache.store(doc_id, text, embedding, metadata or {})
        logger.debug("Indexed document %s (%d chars)", doc_id, len(text))

    async def reindex_all(self, documents: List[Dict[str, Any]]) -> int:
        """Bulk-index a list of documents.

        Each document dict must contain at least ``doc_id`` and ``text``
        keys.  An optional ``metadata`` key provides additional metadata.

        Parameters
        ----------
        documents:
            List of dicts with ``doc_id``, ``text``, and optionally
            ``metadata`` keys.

        Returns
        -------
        int
            The number of documents indexed.
        """
        self._ensure_initialised()

        count = 0
        for doc in documents:
            doc_id = doc.get("doc_id", "")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            if not doc_id or not text:
                logger.warning("Skipping document with missing doc_id or text")
                continue
            await self.index_document(doc_id, text, metadata)
            count += 1

        logger.info("Reindexed %d documents", count)
        return count

    # ── Search ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> List[SearchResult]:
        """Search for documents semantically similar to the query.

        The query text is encoded into a vector and compared against all
        stored document embeddings using cosine similarity.  Results are
        returned sorted by descending similarity score.

        Parameters
        ----------
        query:
            Free-text search query.
        limit:
            Maximum number of results to return (default 10).

        Returns
        -------
        list[SearchResult]
            Results sorted by descending similarity score.
        """
        self._ensure_initialised()

        query_embedding = self._embedder.encode(query)
        raw_results = await self._cache.search(query_embedding, limit=limit)

        results: List[SearchResult] = []
        for item in raw_results:
            results.append(SearchResult(
                doc_id=item["doc_id"],
                text=item["text"],
                score=item["score"],
                metadata=item.get("metadata", {}),
                match_type="semantic",
            ))

        logger.debug(
            "Semantic search for %r returned %d results",
            query[:50],
            len(results),
        )
        return results

    # ── Stats ──────────────────────────────────────────────────────────────

    async def stats(self) -> Dict[str, Any]:
        """Return statistics about the semantic search engine.

        Returns
        -------
        dict
            Keys: ``embedding_dim``, ``initialised``, plus all keys from
            :meth:`EmbeddingCache.stats` (``count``, ``db_size_bytes``,
            ``db_size_mb``).
        """
        cache_stats = await self._cache.stats()
        return {
            "embedding_dim": self._embedder.dim,
            "initialised": self._initialised,
            **cache_stats,
        }


# ──────────────────────────────────────────────────────────────────────────────
# HybridSearchEngine
# ──────────────────────────────────────────────────────────────────────────────

class HybridSearchEngine:
    """Hybrid search engine combining semantic and keyword search.

    Uses **Reciprocal Rank Fusion** (RRF) to merge ranked result lists
    from the semantic embedding engine and an external keyword/TF-IDF
    search backend.  The RRF score for a document at rank *r* in a list
    is ``1 / (k + r)`` where *k* is a constant (default 60).

    Parameters
    ----------
    semantic_weight:
        Weight assigned to semantic search scores in the combined
        result (default 0.6).
    keyword_weight:
        Weight assigned to keyword search scores in the combined
        result (default 0.4).
    dim:
        Embedding dimensionality for the internal semantic engine
        (default 256).
    db_path:
        Path to the SQLite database for the embedding cache.
    knowledge_base_search_fn:
        Optional async callable that performs keyword search.
        Signature::

            async def search(query: str, limit: int) -> list[dict]

        Each dict must contain ``doc_id``, ``text``, and ``score`` keys.

    Examples
    --------
    >>> hybrid = HybridSearchEngine(
    ...     knowledge_base_search_fn=my_kb.search,
    ... )
    >>> await hybrid.initialize()
    >>> results = await hybrid.search("fastapi dependency injection")
    >>> for r in results:
    ...     print(f"{r.doc_id}: semantic={r.semantic_score:.3f} "
    ...           f"keyword={r.keyword_score:.3f} "
    ...           f"combined={r.combined_score:.3f}")
    >>> await hybrid.close()
    """

    def __init__(
        self,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        dim: int = DEFAULT_EMBEDDING_DIM,
        db_path: str = DEFAULT_DB_PATH,
        knowledge_base_search_fn: Optional[KeywordSearchFn] = None,
    ) -> None:
        self._semantic_engine = SemanticSearchEngine(dim=dim, db_path=db_path)
        self._semantic_weight: float = semantic_weight
        self._keyword_weight: float = keyword_weight
        self._kb_search_fn: Optional[KeywordSearchFn] = knowledge_base_search_fn
        self._initialised: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialise the underlying semantic search engine.

        Must be called once before any search or indexing operations.
        """
        await self._semantic_engine.initialize()
        self._initialised = True
        logger.info(
            "HybridSearchEngine initialized "
            "(semantic_weight=%.2f, keyword_weight=%.2f, kb_search=%s)",
            self._semantic_weight,
            self._keyword_weight,
            "enabled" if self._kb_search_fn else "disabled",
        )

    async def close(self) -> None:
        """Close the underlying semantic search engine.

        Safe to call multiple times.
        """
        await self._semantic_engine.close()
        self._initialised = False
        logger.info("HybridSearchEngine closed")

    def _ensure_initialised(self) -> None:
        """Raise ``RuntimeError`` if the engine has not been initialised."""
        if not self._initialised:
            raise RuntimeError(
                "HybridSearchEngine is not initialized. "
                "Call `await engine.initialize()` first."
            )

    # ── Indexing (delegated to semantic engine) ────────────────────────────

    async def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index a document into the semantic vector store.

        Parameters
        ----------
        doc_id:
            Unique document identifier.
        text:
            Document text to embed and store.
        metadata:
            Optional arbitrary key-value metadata.
        """
        self._ensure_initialised()
        await self._semantic_engine.index_document(doc_id, text, metadata)

    async def reindex_all(self, documents: List[Dict[str, Any]]) -> int:
        """Bulk-index a list of documents.

        Parameters
        ----------
        documents:
            List of dicts with ``doc_id``, ``text``, and optionally
            ``metadata`` keys.

        Returns
        -------
        int
            Number of documents indexed.
        """
        self._ensure_initialised()
        return await self._semantic_engine.reindex_all(documents)

    # ── Hybrid search ──────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        semantic_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
    ) -> List[HybridResult]:
        """Search using both semantic and keyword strategies, fused via RRF.

        The search proceeds in three steps:

        1. **Semantic search** — encode the query, retrieve top candidates
           from the embedding cache.
        2. **Keyword search** — if a ``knowledge_base_search_fn`` was
           provided at construction, call it with the query.
        3. **Fusion** — merge the two ranked lists using Reciprocal Rank
           Fusion and compute weighted combined scores.

        If only one search backend is available (e.g. no keyword search
        function), results from the available backend are returned as
        hybrid results with the missing score set to 0.0.

        Parameters
        ----------
        query:
            Free-text search query.
        limit:
            Maximum number of results to return (default 10).
        semantic_weight:
            Override the default semantic weight for this query.
        keyword_weight:
            Override the default keyword weight for this query.

        Returns
        -------
        list[HybridResult]
            Results sorted by descending ``combined_score``.
        """
        self._ensure_initialised()

        sw = semantic_weight if semantic_weight is not None else self._semantic_weight
        kw = keyword_weight if keyword_weight is not None else self._keyword_weight

        # ── Step 1: Semantic search ────────────────────────────────────
        semantic_results = await self._semantic_engine.search(query, limit=limit * 3)

        # ── Step 2: Keyword search (optional) ──────────────────────────
        keyword_results: List[Dict[str, Any]] = []
        if self._kb_search_fn is not None:
            try:
                keyword_results = await self._kb_search_fn(query, limit=limit * 3)
            except Exception as exc:
                logger.warning(
                    "Keyword search failed for query %r: %s",
                    query[:50],
                    exc,
                )

        # ── Step 3: Reciprocal Rank Fusion ──────────────────────────────
        # Score each document by its RRF rank across both lists.
        doc_scores: Dict[str, Dict[str, float]] = {}

        for rank, result in enumerate(semantic_results):
            doc_id = result.doc_id
            rrf_score = 1.0 / (_RRF_K + rank + 1)
            doc_scores.setdefault(doc_id, {
                "semantic_score": result.score,
                "keyword_score": 0.0,
                "rrf": 0.0,
                "text": result.text,
                "metadata": result.metadata,
            })
            doc_scores[doc_id]["rrf"] += rrf_score
            doc_scores[doc_id]["semantic_score"] = result.score

        for rank, item in enumerate(keyword_results):
            doc_id = item.get("doc_id", "")
            if not doc_id:
                continue
            rrf_score = 1.0 / (_RRF_K + rank + 1)
            doc_scores.setdefault(doc_id, {
                "semantic_score": 0.0,
                "keyword_score": 0.0,
                "rrf": 0.0,
                "text": item.get("text", ""),
                "metadata": item.get("metadata", {}),
            })
            doc_scores[doc_id]["rrf"] += rrf_score
            doc_scores[doc_id]["keyword_score"] = item.get("score", 0.0)
            # Merge text/metadata from keyword results if not already set.
            if not doc_scores[doc_id]["text"] and item.get("text"):
                doc_scores[doc_id]["text"] = item["text"]
            if not doc_scores[doc_id]["metadata"] and item.get("metadata"):
                doc_scores[doc_id]["metadata"] = item["metadata"]

        # ── Step 4: Compute combined scores ─────────────────────────────
        results: List[HybridResult] = []
        for doc_id, scores in doc_scores.items():
            sem_score = scores["semantic_score"]
            kw_score = scores["keyword_score"]

            # Normalise individual scores for the weighted combination.
            # If one score is zero (backend unavailable or no match),
            # the other score carries full weight.
            if sem_score > 0.0 and kw_score > 0.0:
                combined = sw * sem_score + kw * kw_score
            elif sem_score > 0.0:
                combined = sem_score
            elif kw_score > 0.0:
                combined = kw_score
            else:
                combined = scores["rrf"]

            results.append(HybridResult(
                doc_id=doc_id,
                text=scores["text"],
                semantic_score=round(sem_score, 6),
                keyword_score=round(kw_score, 6),
                combined_score=round(combined, 6),
                metadata=scores["metadata"],
                match_type="hybrid",
            ))

        # Sort by combined score descending.
        results.sort(key=lambda r: r.combined_score, reverse=True)

        logger.debug(
            "Hybrid search for %r: %d semantic, %d keyword, %d fused",
            query[:50],
            len(semantic_results),
            len(keyword_results),
            len(results),
        )
        return results[:limit]

    # ── Semantic-only passthrough ──────────────────────────────────────────

    async def semantic_search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> List[SearchResult]:
        """Perform semantic-only search (bypasses keyword fusion).

        Delegates directly to the internal :class:`SemanticSearchEngine`.

        Parameters
        ----------
        query:
            Free-text search query.
        limit:
            Maximum number of results (default 10).

        Returns
        -------
        list[SearchResult]
            Results sorted by descending semantic similarity.
        """
        self._ensure_initialised()
        return await self._semantic_engine.search(query, limit=limit)

    # ── Stats ──────────────────────────────────────────────────────────────

    async def stats(self) -> Dict[str, Any]:
        """Return statistics about the hybrid search engine.

        Returns
        -------
        dict
            Keys: ``semantic_weight``, ``keyword_weight``,
            ``keyword_search_enabled``, ``embedding_dim``,
            ``initialised``, plus keys from
            :meth:`EmbeddingCache.stats`.
        """
        engine_stats = await self._semantic_engine.stats()
        return {
            "semantic_weight": self._semantic_weight,
            "keyword_weight": self._keyword_weight,
            "keyword_search_enabled": self._kb_search_fn is not None,
            **engine_stats,
        }
