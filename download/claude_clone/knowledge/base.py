"""
Base classes for knowledge sources and knowledge base.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KnowledgeSource(ABC):
    """
    Abstract base class for knowledge sources.
    
    A knowledge source provides content that can be ingested into a
    knowledge base for retrieval-augmented generation.
    """
    
    @abstractmethod
    def load(self) -> List[str]:
        """
        Load content from the source.
        
        Returns:
            A list of content chunks (strings) ready for embedding.
        """
        ...
    
    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the type identifier for this source."""
        ...
    
    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about this source."""
        ...


class KnowledgeBase:
    """
    In-memory knowledge base that stores and retrieves content chunks.
    
    Args:
        name: Optional name for this knowledge base.
        embedding_fn: Optional callable for embedding text chunks.
    """
    
    def __init__(self, name: str = "default", embedding_fn=None):
        self.name = name
        self._chunks: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._embedding_fn = embedding_fn
        self._embeddings: Optional[List[List[float]]] = None
    
    def add_source(self, source: KnowledgeSource) -> int:
        """
        Ingest a knowledge source into the knowledge base.
        
        Args:
            source: A KnowledgeSource instance.
        
        Returns:
            The number of chunks added.
        """
        chunks = source.load()
        meta = source.metadata
        for chunk in chunks:
            self._chunks.append(chunk)
            self._metadata.append({
                "source_type": source.source_type,
                **meta,
            })
        logger.info(
            "Added %d chunks from %s source to knowledge base '%s'",
            len(chunks), source.source_type, self.name,
        )
        return len(chunks)
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> int:
        """
        Directly add document chunks to the knowledge base.
        
        Args:
            documents: List of text chunks.
            metadata: Optional per-document metadata.
        
        Returns:
            The number of chunks added.
        """
        for i, doc in enumerate(documents):
            self._chunks.append(doc)
            self._metadata.append(metadata[i] if metadata and i < len(metadata) else {})
        return len(documents)
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant chunks.
        
        Uses simple keyword matching if no embedding function is provided.
        
        Args:
            query: The search query.
            limit: Maximum number of results to return.
        
        Returns:
            A list of result dictionaries with 'content' and 'metadata' keys.
        """
        if not self._chunks:
            return []
        
        if self._embedding_fn and self._embeddings:
            return self._vector_search(query, limit)
        
        return self._keyword_search(query, limit)
    
    def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Simple keyword-based search."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        scored = []
        for i, chunk in enumerate(self._chunks):
            chunk_lower = chunk.lower()
            score = sum(1 for term in query_terms if term in chunk_lower)
            if score > 0:
                scored.append((score, i, chunk))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx, chunk in scored[:limit]:
            results.append({
                "content": chunk,
                "metadata": self._metadata[idx],
                "score": score,
            })
        return results
    
    def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Vector similarity search using embeddings."""
        if not self._embedding_fn or not self._embeddings:
            return self._keyword_search(query, limit)
        
        try:
            query_embedding = self._embedding_fn(query)
            scored = []
            for i, emb in enumerate(self._embeddings):
                score = self._cosine_similarity(query_embedding, emb)
                scored.append((score, i))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            for score, idx in scored[:limit]:
                results.append({
                    "content": self._chunks[idx],
                    "metadata": self._metadata[idx],
                    "score": score,
                })
            return results
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return self._keyword_search(query, limit)
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    @property
    def chunk_count(self) -> int:
        """Return the number of stored chunks."""
        return len(self._chunks)
    
    def clear(self) -> None:
        """Clear all stored knowledge."""
        self._chunks.clear()
        self._metadata.clear()
        self._embeddings = None
    
    def __repr__(self) -> str:
        return f"KnowledgeBase(name={self.name!r}, chunks={self.chunk_count})"
