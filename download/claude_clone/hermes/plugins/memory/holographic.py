"""
Holographic Memory Plugin — Vector-based retrieval with embedding
generation, similarity search, and memory consolidation.

Plugin manifest example (plugin.yaml)::

    name: holographic
    display_name: Holographic Memory
    version: 1.0.0
    type: semantic
    description: Vector-based retrieval with embedding generation and similarity search.
    author: Hermes Team
    required_packages: [numpy, sentence-transformers]
    config_schema:
      embedding_model:
        type: string
        description: HuggingFace model name for embeddings (default: all-MiniLM-L6-v2)
      storage_path:
        type: string
        description: Path to vector storage directory
      similarity_threshold:
        type: number
        description: Minimum cosine similarity for search results (0.0-1.0)
"""

from __future__ import annotations

import json
import logging
import math
import uuid
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


@register_builtin("holographic")
class HolographicMemoryPlugin(BaseMemoryPlugin):
    """
    Memory plugin with vector-based retrieval.

    Generates embeddings for stored memories and uses cosine similarity
    for semantic search. Supports memory consolidation (merging similar
    memories to reduce redundancy).
    """

    metadata = MemoryPluginMetadata(
        name="holographic",
        display_name="Holographic Memory",
        version="1.0.0",
        description="Vector-based retrieval with embedding generation and similarity search.",
        plugin_type=MemoryPluginType.SEMANTIC,
        author="Hermes Team",
        required_packages=["numpy", "sentence-transformers"],
    )

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        super().__init__(config)
        self._model: Any = None
        self._vectors: dict[str, list[float]] = {}
        self._entries: dict[str, MemoryEntry] = {}
        self._storage_path: Path = Path(
            config.storage_path or "~/.claude_clone/holographic_memory"
        ).expanduser().resolve()
        self._similarity_threshold: float = config.extra.get("similarity_threshold", 0.5)
        self._embedding_dim: int = 384  # Default for MiniLM

    async def initialize(self) -> None:
        """Load embedding model and persisted vectors."""
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._load_vectors()

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            model_name = self.config.embedding_model or "all-MiniLM-L6-v2"
            self._model = SentenceTransformer(model_name)
            # Infer dimension
            test_emb = self._model.encode(["test"])
            self._embedding_dim = len(test_emb[0])
            self._initialized = True
            logger.info("Holographic memory plugin initialized (model=%s, dim=%d)", model_name, self._embedding_dim)
        except ImportError:
            logger.warning("sentence-transformers not installed; using fallback hashing")
            self._initialized = True

    async def store(self, entry: MemoryEntry) -> str:
        """Store an entry with its embedding vector."""
        entry_id = entry.id or str(uuid.uuid4())
        entry.id = entry_id

        # Generate embedding
        if self._model is not None:
            try:
                embedding = self._model.encode([entry.content])[0].tolist()
                entry.embedding = embedding
                self._vectors[entry_id] = embedding
            except Exception:
                logger.exception("Failed to generate embedding")

        self._entries[entry_id] = entry
        await self._persist_entry(entry)
        return entry_id

    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve an entry by ID."""
        return self._entries.get(entry_id)

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """Search entries by cosine similarity."""
        if self._model is None or not self._vectors:
            # Fallback: simple keyword match
            return self._keyword_search(query, limit, filters)

        query_embedding = self._model.encode([query])[0].tolist()
        scored: list[tuple[float, MemoryEntry]] = []

        for entry_id, vec in self._vectors.items():
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            if filters and not self._matches_filters(entry, filters):
                continue
            sim = self._cosine_similarity(query_embedding, vec)
            if sim >= self._similarity_threshold:
                entry.relevance_score = sim
                scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        self._entries.pop(entry_id, None)
        self._vectors.pop(entry_id, None)
        # Remove persisted file
        entry_file = self._storage_path / f"{entry_id}.json"
        if entry_file.exists():
            entry_file.unlink()
        return True

    async def health_check(self) -> dict:
        """Check plugin health."""
        import time
        start = time.monotonic()
        status = "healthy" if self._model else "degraded"
        elapsed = (time.monotonic() - start) * 1000
        return {
            "status": status,
            "latency_ms": elapsed,
            "details": {
                "entries": len(self._entries),
                "vectors": len(self._vectors),
                "embedding_dim": self._embedding_dim,
                "model_loaded": self._model is not None,
            },
        }

    async def shutdown(self) -> None:
        """Persist all data and clean up."""
        self._save_vectors()
        self._model = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Memory consolidation
    # ------------------------------------------------------------------

    async def consolidate(self, similarity_threshold: float = 0.95) -> int:
        """
        Merge similar memories to reduce redundancy.

        Returns the number of entries merged.
        """
        merged_count = 0
        ids = list(self._vectors.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                if id_a not in self._vectors or id_b not in self._vectors:
                    continue
                sim = self._cosine_similarity(self._vectors[id_a], self._vectors[id_b])
                if sim >= similarity_threshold:
                    entry_a = self._entries.get(id_a)
                    entry_b = self._entries.get(id_b)
                    if entry_a and entry_b:
                        # Merge B into A
                        entry_a.content += f"\n[Also: {entry_b.content}]"
                        entry_a.tags = list(set(entry_a.tags + entry_b.tags))
                        entry_a.metadata["merged_from"] = entry_b.id
                        await self.delete(id_b)
                        merged_count += 1
        if merged_count:
            self._save_vectors()
        logger.info("Consolidated %d memory entries", merged_count)
        return merged_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _keyword_search(
        self, query: str, limit: int, filters: Optional[dict]
    ) -> list[MemoryEntry]:
        """Fallback keyword-based search when no embedding model is available."""
        query_lower = query.lower()
        results: list[MemoryEntry] = []
        for entry in self._entries.values():
            if filters and not self._matches_filters(entry, filters):
                continue
            if query_lower in entry.content.lower():
                entry.relevance_score = 0.7
                results.append(entry)
        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results[:limit]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _matches_filters(entry: MemoryEntry, filters: dict) -> bool:
        """Check if an entry matches all filter criteria."""
        for key, value in filters.items():
            if key == "tags":
                if not all(t in entry.tags for t in value if isinstance(value, list)):
                    return False
            elif key == "source":
                if entry.source != value:
                    return False
        return True

    def _persist_entry(self, entry: MemoryEntry) -> None:
        """Persist a single entry to disk."""
        entry_file = self._storage_path / f"{entry.id}.json"
        try:
            data = {
                "id": entry.id,
                "content": entry.content,
                "metadata": entry.metadata,
                "tags": entry.tags,
                "source": entry.source,
                "created_at": entry.created_at.isoformat(),
            }
            entry_file.write_text(json.dumps(data, default=str), encoding="utf-8")
        except OSError:
            logger.exception("Failed to persist entry %s", entry.id)

    def _load_vectors(self) -> None:
        """Load persisted entries and regenerate vectors on init."""
        import numpy as np  # type: ignore[import-untyped]
        vec_path = self._storage_path / "vectors.npy"
        idx_path = self._storage_path / "index.json"
        if vec_path.exists() and idx_path.exists():
            try:
                vecs = np.load(str(vec_path), allow_pickle=True)
                index = json.loads(idx_path.read_text(encoding="utf-8"))
                for i, entry_id in enumerate(index):
                    if i < len(vecs):
                        self._vectors[entry_id] = vecs[i].tolist()
                        entry = self._load_entry(entry_id)
                        if entry:
                            self._entries[entry_id] = entry
            except Exception:
                logger.exception("Failed to load persisted vectors")

    def _load_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Load a single entry from disk."""
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
                source=data.get("source", "holographic"),
                created_at=data.get("created_at", ""),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _save_vectors(self) -> None:
        """Persist all vectors and index to disk."""
        try:
            import numpy as np
            if not self._vectors:
                return
            ids = list(self._vectors.keys())
            vecs = [self._vectors[eid] for eid in ids]
            np.save(str(self._storage_path / "vectors.npy"), np.array(vecs))
            (self._storage_path / "index.json").write_text(
                json.dumps(ids), encoding="utf-8"
            )
        except Exception:
            logger.exception("Failed to save vectors")
