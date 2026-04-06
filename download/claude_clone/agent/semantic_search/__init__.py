"""
Semantic Memory Search Module.

Provides true semantic embedding-based search using lightweight hash-based
vector embeddings (character n-gram hashing) combined with SQLite-backed
vector storage.  No external dependencies — only Python stdlib + sqlite3.

Submodules
----------
embedder
    Lightweight embedding engine using character n-gram hashing into
    256-dimensional vectors with TF weighting and L2 normalization.
vector_store
    SQLite-backed vector cache with async CRUD operations and cosine-
    similarity-based top-k retrieval.
hybrid_search
    ``SemanticSearchEngine`` (embeddings + vector store) and
    ``HybridSearchEngine`` (reciprocal-rank fusion of semantic and
    keyword/TF-IDF results).

Usage::

    from agent.semantic_search import SemanticSearchEngine

    engine = SemanticSearchEngine()
    await engine.initialize()
    await engine.index_document("doc-1", "How to deploy to AWS ECS", {"source": "cli"})
    results = await engine.search("cloud deployment")
    await engine.close()
"""

from __future__ import annotations

from agent.semantic_search.embedder import EmbeddingEngine
from agent.semantic_search.vector_store import EmbeddingCache
from agent.semantic_search.hybrid_search import (
    SemanticSearchEngine,
    HybridSearchEngine,
    SearchResult,
    HybridResult,
)

__all__ = [
    "EmbeddingEngine",
    "EmbeddingCache",
    "SemanticSearchEngine",
    "HybridSearchEngine",
    "SearchResult",
    "HybridResult",
]
