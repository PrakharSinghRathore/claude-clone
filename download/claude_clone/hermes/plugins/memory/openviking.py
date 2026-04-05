"""
OpenViking Memory Plugin — Open-source graph-based memory backend
with relationship mapping and knowledge graph storage.

Plugin manifest example (plugin.yaml)::

    name: openviking
    display_name: OpenViking Memory
    version: 1.0.0
    type: graph
    description: Graph-based memory with relationship mapping and knowledge graph storage.
    author: Hermes Team
    required_packages: [networkx]
    config_schema:
      storage_path:
        type: string
        description: Directory for graph data files
      max_nodes:
        type: integer
        description: Maximum nodes in the graph (default: 50000)
      auto_link_threshold:
        type: number
        description: Similarity threshold for auto-linking nodes (0.0-1.0, default: 0.7)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
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


@register_builtin("openviking")
class OpenVikingMemoryPlugin(BaseMemoryPlugin):
    """
    Graph-based memory plugin using NetworkX for relationship mapping.

    Stores memories as nodes in a directed graph with typed edges
    representing relationships (e.g., related_to, derived_from,
    contradicts, supports).
    """

    metadata = MemoryPluginMetadata(
        name="openviking",
        display_name="OpenViking Memory",
        version="1.0.0",
        description="Graph-based memory with relationship mapping and knowledge graph.",
        plugin_type=MemoryPluginType.GRAPH,
        author="Hermes Team",
        required_packages=["networkx"],
    )

    # Relationship types
    REL_RELATED = "related_to"
    REL_DERIVED = "derived_from"
    REL_CONTRADICTS = "contradicts"
    REL_SUPPORTS = "supports"
    REL_CONTAINS = "contains"
    REL_SEQUENCE = "sequence"

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        super().__init__(config)
        self._graph: Any = None
        self._storage_path: Path = Path(
            config.storage_path or "~/.claude_clone/openviking"
        ).expanduser().resolve()
        self._max_nodes = config.extra.get("max_nodes", 50000)
        self._auto_link_threshold = config.extra.get("auto_link_threshold", 0.7)
        self._entries: dict[str, MemoryEntry] = {}

    async def initialize(self) -> None:
        """Initialize graph and load persisted data."""
        self._storage_path.mkdir(parents=True, exist_ok=True)
        try:
            import networkx as nx
            self._graph = nx.DiGraph()
        except ImportError:
            logger.warning("networkx not installed; using simple dict-based graph")
            self._graph = _SimpleGraph()
        self._load_graph()
        self._initialized = True
        logger.info("OpenViking memory plugin initialized (path=%s)", self._storage_path)

    async def store(self, entry: MemoryEntry) -> str:
        """Store an entry as a graph node."""
        entry_id = entry.id or str(uuid.uuid4())
        entry.id = entry_id

        # Check node limit
        if self._graph.number_of_nodes() >= self._max_nodes:
            logger.warning("Graph node limit reached (%d)", self._max_nodes)
            # Prune oldest nodes
            await self._prune_oldest(100)

        # Add node
        self._graph.add_node(
            entry_id,
            content=entry.content,
            tags=entry.tags,
            metadata=entry.metadata,
            source=entry.source or "openviking",
            created_at=entry.created_at.isoformat() if entry.created_at else datetime.utcnow().isoformat(),
        )
        self._entries[entry_id] = entry

        # Auto-link to similar existing nodes
        await self._auto_link(entry_id, entry)

        self._save_graph()
        return entry_id

    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a node entry by ID."""
        if entry_id not in self._entries:
            entry = self._entries.get(entry_id)
            return entry
        return self._entries.get(entry_id)

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """
        Search graph nodes by keyword with optional graph traversal
        to include related nodes.
        """
        results: list[MemoryEntry] = []
        query_lower = query.lower()
        visited: set[str] = set()

        # Direct matches
        for node_id in self._graph.nodes():
            node_data = self._graph.nodes[node_id]
            content = node_data.get("content", "")
            if query_lower in content.lower():
                entry = self._node_to_entry(node_id, node_data)
                entry.relevance_score = 0.9
                results.append(entry)
                visited.add(node_id)

        # Expand to neighbours (depth 1)
        expanded: list[MemoryEntry] = []
        for entry in results[:5]:
            neighbours = list(self._graph.neighbors(entry.id))
            for neighbour_id in neighbours:
                if neighbour_id in visited:
                    continue
                visited.add(neighbour_id)
                node_data = self._graph.nodes[neighbour_id]
                rel_entry = self._node_to_entry(neighbour_id, node_data)
                rel_entry.relevance_score = 0.5
                expanded.append(rel_entry)

        results.extend(expanded)
        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results[:limit]

    async def delete(self, entry_id: str) -> bool:
        """Delete a node and all its edges."""
        if entry_id not in self._graph:
            return False
        self._graph.remove_node(entry_id)
        self._entries.pop(entry_id, None)
        self._save_graph()
        return True

    async def health_check(self) -> dict:
        """Check graph health and statistics."""
        import time
        start = time.monotonic()
        node_count = self._graph.number_of_nodes()
        edge_count = self._graph.number_of_edges()
        elapsed = (time.monotonic() - start) * 1000
        return {
            "status": "healthy",
            "latency_ms": elapsed,
            "details": {
                "nodes": node_count,
                "edges": edge_count,
                "density": round(edge_count / max(node_count * (node_count - 1), 1), 4),
                "max_nodes": self._max_nodes,
            },
        }

    async def shutdown(self) -> None:
        """Persist graph and clean up."""
        self._save_graph()
        self._initialized = False

    # ------------------------------------------------------------------
    # Graph operations
    # ------------------------------------------------------------------

    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Add a directed relationship between two nodes."""
        if source_id not in self._graph or target_id not in self._graph:
            return False
        self._graph.add_edge(source_id, target_id, type=rel_type, metadata=metadata or {})
        self._save_graph()
        return True

    async def get_relationships(self, node_id: str) -> list[dict]:
        """Get all relationships for a node."""
        if node_id not in self._graph:
            return []
        relationships = []
        for neighbor in self._graph.neighbors(node_id):
            edge_data = self._graph.get_edge_data(node_id, neighbor)
            if edge_data:
                relationships.append({
                    "target": neighbor,
                    "type": edge_data.get("type", "unknown"),
                    "metadata": edge_data.get("metadata", {}),
                })
        return relationships

    async def get_neighbours(self, node_id: str, depth: int = 1) -> list[MemoryEntry]:
        """Get neighbouring entries up to a given depth."""
        visited: set[str] = {node_id}
        current_level = {node_id}

        for _ in range(depth):
            next_level: set[str] = set()
            for nid in current_level:
                for neighbour in self._graph.neighbors(nid):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        next_level.add(neighbour)
            current_level = next_level

        results: list[MemoryEntry] = []
        for nid in visited:
            if nid == node_id:
                continue
            node_data = self._graph.nodes[nid]
            entry = self._node_to_entry(nid, node_data)
            results.append(entry)
        return results

    async def find_paths(self, source_id: str, target_id: str, max_length: int = 5) -> list[list[str]]:
        """Find paths between two nodes in the graph."""
        try:
            if hasattr(self._graph, "shortest_path"):
                path = self._graph.shortest_path(source_id, target_id)
                if len(path) <= max_length:
                    return [path]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _auto_link(self, entry_id: str, entry: MemoryEntry) -> None:
        """Automatically create relationships to similar existing nodes."""
        content_lower = entry.content.lower()
        for node_id in list(self._graph.nodes()):
            if node_id == entry_id:
                continue
            node_data = self._graph.nodes[node_id]
            existing_content = node_data.get("content", "").lower()
            # Simple word overlap for similarity
            words_a = set(content_lower.split())
            words_b = set(existing_content.split())
            if not words_a or not words_b:
                continue
            overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
            if overlap >= self._auto_link_threshold:
                self._graph.add_edge(entry_id, node_id, type=self.REL_RELATED, auto=True)

    async def _prune_oldest(self, count: int) -> None:
        """Remove the oldest nodes from the graph."""
        nodes_by_date = sorted(
            self._graph.nodes(),
            key=lambda n: self._graph.nodes[n].get("created_at", ""),
        )
        for node_id in nodes_by_date[:count]:
            self._graph.remove_node(node_id)
            self._entries.pop(node_id, None)
        logger.info("Pruned %d oldest nodes from graph", count)

    def _node_to_entry(self, node_id: str, node_data: dict) -> MemoryEntry:
        """Convert a graph node to a MemoryEntry."""
        return MemoryEntry(
            id=node_id,
            content=node_data.get("content", ""),
            metadata=node_data.get("metadata", {}),
            tags=node_data.get("tags", []),
            source=node_data.get("source", "openviking"),
            created_at=node_data.get("created_at", ""),
        )

    def _save_graph(self) -> None:
        """Persist the graph to disk."""
        try:
            import networkx as nx
            if isinstance(self._graph, nx.DiGraph):
                path = self._storage_path / "graph.gpickle"
                nx.write_gpickle(self._graph, str(path))
            else:
                # Simple graph fallback
                data = {"nodes": dict(self._graph._nodes), "edges": self._graph._edges}
                (self._storage_path / "graph.json").write_text(
                    json.dumps(data, default=str), encoding="utf-8"
                )
        except Exception:
            logger.exception("Failed to save graph")

    def _load_graph(self) -> None:
        """Load the graph from disk."""
        try:
            import networkx as nx
            gpickle_path = self._storage_path / "graph.gpickle"
            json_path = self._storage_path / "graph.json"
            if gpickle_path.exists() and isinstance(self._graph, nx.DiGraph):
                self._graph = nx.read_gpickle(str(gpickle_path))
                # Rebuild entries cache
                for node_id in self._graph.nodes():
                    node_data = self._graph.nodes[node_id]
                    self._entries[node_id] = self._node_to_entry(node_id, node_data)
            elif json_path.exists() and isinstance(self._graph, _SimpleGraph):
                data = json.loads(json_path.read_text(encoding="utf-8"))
                self._graph._nodes = data.get("nodes", {})
                self._graph._edges = data.get("edges", [])
        except Exception:
            logger.exception("Failed to load graph")


class _SimpleGraph:
    """Minimal fallback graph when networkx is not available."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._edges: list[dict] = []

    def add_node(self, node_id: str, **kwargs: Any) -> None:
        self._nodes[node_id] = kwargs

    def remove_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        self._edges = [e for e in self._edges if e["source"] != node_id and e["target"] != node_id]

    def nodes(self):
        return self._nodes.keys()

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        return len(self._edges)

    def add_edge(self, source: str, target: str, **kwargs: Any) -> None:
        self._edges.append({"source": source, "target": target, **kwargs})

    def get_edge_data(self, source: str, target: str) -> Optional[dict]:
        for edge in self._edges:
            if edge["source"] == source and edge["target"] == target:
                return edge
        return None

    def neighbors(self, node_id: str) -> list[str]:
        result: list[str] = []
        for edge in self._edges:
            if edge["source"] == node_id:
                result.append(edge["target"])
        return result
