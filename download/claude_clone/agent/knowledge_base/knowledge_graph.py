"""
Knowledge Graph module for mapping relationships between knowledge entries.

Provides entity linking, graph traversal (BFS/DFS), community detection via
label propagation, PageRank-based importance scoring, and ASCII-art
visualisation — all backed by an in-memory adjacency-list cache and a
persistent SQLite ``knowledge_relations`` table.

Every public method is ``async`` and delegates blocking SQLite / CPU-bound
work to ``asyncio.run_in_executor`` so that no external async drivers are
required.

Usage::

    from agent.knowledge_base.knowledge_graph import KnowledgeGraph, RelationType

    kg = KnowledgeGraph()
    await kg.initialize("~/.claude_clone/knowledge.db")
    await kg.add_relation("entry_a", "entry_b", RelationType.RELATED_TO)
    neighbors = await kg.get_neighbors("entry_a", depth=2)
    path = await kg.find_path("entry_a", "entry_c")
    clusters = await kg.get_clusters()
    related = await kg.suggest_related("entry_a", limit=5)
    await kg.close()
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Default damping factor for PageRank (Google's original value).
_PAGERANK_DAMPING: float = 0.85

# Convergence tolerance for PageRank iterations.
_PAGERANK_TOLERANCE: float = 1e-6

# Maximum number of PageRank iterations.
_PAGERANK_MAX_ITER: int = 200

# Default weight assigned to a new relation when not explicitly provided.
_DEFAULT_WEIGHT: float = 0.5

# Maximum depth for BFS traversals that lack an explicit cap.
_MAX_BFS_DEPTH: int = 10

# Default node budget for ASCII visualisation.
_DEFAULT_VIS_MAX_NODES: int = 50

# Box-drawing characters for ASCII graph rendering.
_BOX_TL = "\u250c"   # ┌
_BOX_TR = "\u2510"   # ┐
_BOX_BL = "\u2514"   # └
_BOX_BR = "\u2518"   # ┘
_BOX_H  = "\u2500"   # ─
_BOX_V  = "\u2502"   # │
_BOX_LJ = "\u251c"   # ├
_BOX_RJ = "\u2524"   # ┤
_BOX_TJ = "\u252c"   # ┬
_BOX_BJ = "\u2534"   # ┴
_BOX_XJ = "\u253c"   # ┼


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class RelationType(str, Enum):
    """Supported edge (relation) types in the knowledge graph.

    Each value is a human-readable label suitable for storage and display.
    """

    RELATED_TO = "RELATED_TO"
    """General-purpose relationship between two entries."""

    DEPENDS_ON = "DEPENDS_ON"
    """The source entry depends on the target entry."""

    EXTENDS = "EXTENDS"
    """The source entry extends or builds upon the target entry."""

    CONTRADICTS = "CONTRADICTS"
    """The source entry contradicts the target entry."""

    SUPERSEDES = "SUPERSEDES"
    """The source entry supersedes / replaces the target entry."""

    PART_OF = "PART_OF"
    """The source entry is a part of the target entry."""

    APPLIES_TO = "APPLIES_TO"
    """The source entry applies to the target entry."""

    EXAMPLE_OF = "EXAMPLE_OF"
    """The source entry is an example of the target entry."""

    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    """The source entry is an alternative to the target entry."""

    PREREQUISITE = "PREREQUISITE"
    """The target entry is a prerequisite for the source entry."""


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """A lightweight representation of a node in the knowledge graph.

    Attributes
    ----------
    entry_id:
        Unique identifier of the knowledge entry (foreign key into the
        knowledge store).
    title:
        Human-readable title of the entry.
    category:
        Category label (e.g. ``"concept"``, ``"procedure"``).
    importance:
        Derived importance score (0.0–1.0), often from PageRank.
    tags:
        List of tags associated with the entry.
    """

    entry_id: str
    title: str = ""
    category: str = ""
    importance: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    """A directed, weighted edge between two knowledge entries.

    Attributes
    ----------
    from_id:
        Source node id.
    to_id:
        Target node id.
    relation_type:
        Semantic relationship (see :class:`RelationType`).
    weight:
        Edge weight in the range [0.0, 1.0].  Higher values indicate
        stronger relationships.
    metadata:
        Arbitrary key-value metadata attached to the edge.
    """

    from_id: str
    to_id: str
    relation_type: str
    weight: float = _DEFAULT_WEIGHT
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphCluster:
    """A cluster (community) of related entries detected by label propagation.

    Attributes
    ----------
    cluster_id:
        Unique identifier for the cluster (auto-generated).
    entry_ids:
        Set of entry ids belonging to the cluster.
    centroid_id:
        The entry id with the highest within-cluster degree (most connected).
    label:
        A human-readable label derived from the most common tags or titles.
    coherence_score:
        Internal coherence in the range [0.0, 1.0], based on the average
        pairwise edge weight among cluster members.
    """

    cluster_id: str
    entry_ids: List[str] = field(default_factory=list)
    centroid_id: str = ""
    label: str = ""
    coherence_score: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# KnowledgeGraph
# ──────────────────────────────────────────────────────────────────────────────

class KnowledgeGraph:
    """Persistent knowledge graph backed by SQLite.

    Edges are stored in a ``knowledge_relations`` table.  An in-memory
    adjacency-list cache is maintained for fast traversal.  The cache is
    rebuilt lazily when the underlying data is believed to be stale.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  ``~`` is expanded.  The parent
        directory is created automatically if it does not exist.
    """

    # ── Construction / lifecycle ─────────────────────────────────────────

    def __init__(self, db_path: str = "~/.claude_clone/knowledge.db") -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self._conn: Optional[sqlite3.Connection] = None

        # In-memory adjacency list: node_id -> list of GraphEdge
        self._outgoing: Dict[str, List[GraphEdge]] = defaultdict(list)
        # Reverse adjacency list: node_id -> list of GraphEdge (incoming)
        self._incoming: Dict[str, List[GraphEdge]] = defaultdict(list)
        # Set of all known node ids (populated on demand from the store).
        self._node_ids: Set[str] = set()

        # Cache invalidation flag.  Set to ``True`` whenever edges are
        # added / removed so that the next traversal can rebuild the cache.
        self._cache_dirty: bool = True

    # ── Initialisation / teardown ─────────────────────────────────────────

    async def initialize(self, db_path: Optional[str] = None) -> None:
        """Connect to the database, create tables, and warm the cache.

        Parameters
        ----------
        db_path:
            Override the database path supplied at construction time.
        """
        if db_path is not None:
            self.db_path = Path(db_path).expanduser().resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(self._connect)
        await self._run_sync(self._create_tables)
        await self._rebuild_cache()

    async def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn is not None:
            await self._run_sync(self._conn.close)
            self._conn = None
        self._cache_dirty = True

    def _connect(self) -> None:
        """Open the SQLite connection with WAL mode."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _create_tables(self) -> None:
        """Create or migrate the ``knowledge_relations`` table and indexes.

        The KnowledgeStore may have already created this table with a
        narrower schema (no ``weight`` / ``metadata`` columns).  This
        method uses ``ALTER TABLE … ADD COLUMN`` to add any missing
        columns so that both modules can coexist in the same database.
        """
        assert self._conn is not None
        conn = self._conn

        # Determine if the table already exists and its column set.
        existing_cols: Set[str] = set()
        try:
            tbl_info = conn.execute("PRAGMA table_info(knowledge_relations)").fetchall()
            existing_cols = {row["name"] for row in tbl_info}
        except Exception:
            pass

        if not existing_cols:
            # Fresh table — create with the full schema.
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_relations (
                    from_id        TEXT    NOT NULL,
                    to_id          TEXT    NOT NULL,
                    relation_type  TEXT    NOT NULL,
                    weight         REAL    NOT NULL DEFAULT 0.5,
                    metadata       TEXT    DEFAULT '{}',
                    created_at     TEXT    DEFAULT (datetime('now')),
                    PRIMARY KEY (from_id, to_id, relation_type)
                );
            """)
        else:
            # Migrate: add columns that the KnowledgeStore may not have.
            if "weight" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge_relations ADD COLUMN weight REAL NOT NULL DEFAULT 0.5"
                )
            if "metadata" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge_relations ADD COLUMN metadata TEXT DEFAULT '{}'"
                )

        # Ensure indexes exist (idempotent).
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kr_weight ON knowledge_relations(weight)")
        conn.commit()

    # ── Async wrapper ─────────────────────────────────────────────────────

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous function in the default thread-pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: func(*args, **kwargs)
        )

    def _ensure_conn(self) -> sqlite3.Connection:
        """Return the connection or raise if not initialised."""
        if self._conn is None:
            raise RuntimeError(
                "KnowledgeGraph is not initialized. "
                "Call `await kg.initialize()` first."
            )
        return self._conn

    # ── Cache management ──────────────────────────────────────────────────

    def _rebuild_cache_sync(self) -> None:
        """Rebuild the in-memory adjacency lists from the database.

        This is a **synchronous** helper — callers must wrap it in
        ``_run_sync`` or call it from another sync method already running
        inside the executor.

        Handles both the full schema (with ``weight`` / ``metadata``)
        and the narrower KnowledgeStore schema gracefully.
        """
        conn = self._ensure_conn()
        self._outgoing.clear()
        self._incoming.clear()
        self._node_ids.clear()

        # Detect which columns are available.
        existing_cols = {row["name"] for row in
                        conn.execute("PRAGMA table_info(knowledge_relations)").fetchall()}
        has_weight = "weight" in existing_cols
        has_metadata = "metadata" in existing_cols

        if has_weight and has_metadata:
            select_sql = (
                "SELECT from_id, to_id, relation_type, weight, metadata "
                "FROM knowledge_relations"
            )
        else:
            select_sql = (
                "SELECT from_id, to_id, relation_type, "
                "COALESCE(weight, 0.5) AS weight, "
                "COALESCE(metadata, '{}') AS metadata "
                "FROM knowledge_relations"
            )

        rows = conn.execute(select_sql).fetchall()

        for row in rows:
            from_id = row["from_id"]
            to_id = row["to_id"]
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            edge = GraphEdge(
                from_id=from_id,
                to_id=to_id,
                relation_type=row["relation_type"],
                weight=row["weight"],
                metadata=meta,
            )
            self._outgoing[from_id].append(edge)
            self._incoming[to_id].append(edge)
            self._node_ids.add(from_id)
            self._node_ids.add(to_id)

        self._cache_dirty = False

    async def _rebuild_cache(self) -> None:
        """Async wrapper around ``_rebuild_cache_sync``."""
        await self._run_sync(self._rebuild_cache_sync)

    async def _ensure_cache(self) -> None:
        """Rebuild cache on demand if dirty."""
        if self._cache_dirty:
            await self._rebuild_cache()

    # ── Edge CRUD ─────────────────────────────────────────────────────────

    async def add_relation(
        self,
        from_id: str,
        to_id: str,
        relation_type: RelationType,
        weight: float = _DEFAULT_WEIGHT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a directed relation (edge) between two entries.

        Parameters
        ----------
        from_id:
            Source entry id.
        to_id:
            Target entry id.
        relation_type:
            Semantic relationship.
        weight:
            Edge strength in [0.0, 1.0].
        metadata:
            Optional arbitrary metadata attached to the edge.

        Raises
        ------
        ValueError
            If ``weight`` is outside the [0.0, 1.0] range.
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(
                f"weight must be in [0.0, 1.0], got {weight}"
            )

        rt = relation_type.value if isinstance(relation_type, RelationType) else str(relation_type)
        meta_json = json.dumps(metadata or {})

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT OR REPLACE INTO knowledge_relations "
                "(from_id, to_id, relation_type, weight, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (from_id, to_id, rt, weight, meta_json),
            )
            conn.commit()

        await self._run_sync(_do)
        self._cache_dirty = True

    async def remove_relation(self, from_id: str, to_id: str) -> int:
        """Remove **all** relations from *from_id* to *to_id*.

        Returns
        -------
        int
            The number of edges removed.
        """
        def _do() -> int:
            conn = self._ensure_conn()
            cur = conn.execute(
                "SELECT COUNT(*) FROM knowledge_relations "
                "WHERE from_id = ? AND to_id = ?",
                (from_id, to_id),
            )
            count = cur.fetchone()[0]
            conn.execute(
                "DELETE FROM knowledge_relations "
                "WHERE from_id = ? AND to_id = ?",
                (from_id, to_id),
            )
            conn.commit()
            return count

        result = await self._run_sync(_do)
        if result > 0:
            self._cache_dirty = True
        return result

    async def get_relations(
        self,
        entry_id: str,
        direction: str = "both",
    ) -> List[GraphEdge]:
        """Return all edges incident to *entry_id*.

        Parameters
        ----------
        entry_id:
            The node to query.
        direction:
            ``"outgoing"`` — edges where *entry_id* is the source.
            ``"incoming"`` — edges where *entry_id* is the target.
            ``"both"``    — union of outgoing and incoming.

        Returns
        -------
        list[GraphEdge]
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(
                f"direction must be 'outgoing', 'incoming', or 'both', "
                f"got {direction!r}"
            )

        await self._ensure_cache()

        def _do() -> List[GraphEdge]:
            edges: List[GraphEdge] = []
            if direction in ("outgoing", "both"):
                edges.extend(self._outgoing.get(entry_id, []))
            if direction in ("incoming", "both"):
                edges.extend(self._incoming.get(entry_id, []))
            return edges

        return await self._run_sync(_do)

    # ── Traversal ─────────────────────────────────────────────────────────

    async def get_neighbors(
        self,
        entry_id: str,
        depth: int = 1,
    ) -> Dict[str, List[GraphNode]]:
        """BFS traversal returning neighbours at each depth level.

        Parameters
        ----------
        entry_id:
            Starting node.
        depth:
            Maximum traversal depth (default 1).

        Returns
        -------
        dict[str, list[GraphNode]]
            Keys are depth-level strings (``"1"``, ``"2"``, …) and values
            are lists of :class:`GraphNode` reachable at that depth.
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        depth = min(depth, _MAX_BFS_DEPTH)

        await self._ensure_cache()

        def _do() -> Dict[str, List[GraphNode]]:
            visited: Set[str] = {entry_id}
            result: Dict[str, List[GraphNode]] = defaultdict(list)

            for level in range(1, depth + 1):
                frontier: Set[str] = set()

                if level == 1:
                    # Seed from direct neighbours of entry_id
                    for edge in self._outgoing.get(entry_id, []):
                        if edge.to_id not in visited:
                            frontier.add(edge.to_id)
                    for edge in self._incoming.get(entry_id, []):
                        if edge.from_id not in visited:
                            frontier.add(edge.from_id)
                else:
                    # Expand from the previous level's nodes
                    prev_nodes = result[str(level - 1)]
                    for node in prev_nodes:
                        nid = node.entry_id
                        for edge in self._outgoing.get(nid, []):
                            if edge.to_id not in visited:
                                frontier.add(edge.to_id)
                        for edge in self._incoming.get(nid, []):
                            if edge.from_id not in visited:
                                frontier.add(edge.from_id)

                visited.update(frontier)
                for nid in frontier:
                    result[str(level)].append(
                        GraphNode(entry_id=nid)
                    )

            return dict(result)

        return await self._run_sync(_do)

    # ── Path finding ──────────────────────────────────────────────────────

    async def find_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 5,
    ) -> List[str]:
        """Find the shortest path between two entries using unweighted BFS.

        Parameters
        ----------
        from_id:
            Starting node.
        to_id:
            Destination node.
        max_depth:
            Maximum search depth (default 5).

        Returns
        -------
        list[str]
            Ordered list of entry ids from *from_id* to *to_id*.
            Returns an empty list if no path exists within *max_depth*.
        """
        if from_id == to_id:
            return [from_id]

        max_depth = min(max_depth, _MAX_BFS_DEPTH)
        await self._ensure_cache()

        def _do() -> List[str]:
            visited: Set[str] = {from_id}
            # queue stores (current_node, path_so_far)
            queue: deque[Tuple[str, List[str]]] = deque()
            queue.append((from_id, [from_id]))

            while queue:
                current, path = queue.popleft()

                if len(path) > max_depth:
                    continue

                # Collect neighbours (both directions for undirected feel)
                neighbours: Set[str] = set()
                for edge in self._outgoing.get(current, []):
                    neighbours.add(edge.to_id)
                for edge in self._incoming.get(current, []):
                    neighbours.add(edge.from_id)

                for neighbour in sorted(neighbours):
                    if neighbour == to_id:
                        return path + [to_id]
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append((neighbour, path + [neighbour]))

            return []

        return await self._run_sync(_do)

    async def find_shortest_paths(
        self,
        entry_id: str,
        max_depth: int = 3,
    ) -> Dict[str, List[str]]:
        """Compute shortest paths from *entry_id* to all reachable nodes.

        Uses unweighted BFS.  Results are limited to *max_depth* hops.

        Parameters
        ----------
        entry_id:
            Starting node.
        max_depth:
            Maximum path length (default 3).

        Returns
        -------
        dict[str, list[str]]
            Mapping of destination node id -> ordered path list.
        """
        max_depth = min(max_depth, _MAX_BFS_DEPTH)
        await self._ensure_cache()

        def _do() -> Dict[str, List[str]]:
            visited: Set[str] = {entry_id}
            # parent map: node -> (predecessor, path)
            parent_map: Dict[str, List[str]] = {entry_id: [entry_id]}
            queue: deque[Tuple[str, int]] = deque()
            queue.append((entry_id, 0))

            while queue:
                current, depth = queue.popleft()

                if depth >= max_depth:
                    continue

                neighbours: Set[str] = set()
                for edge in self._outgoing.get(current, []):
                    neighbours.add(edge.to_id)
                for edge in self._incoming.get(current, []):
                    neighbours.add(edge.from_id)

                for neighbour in sorted(neighbours):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        parent_map[neighbour] = parent_map[current] + [neighbour]
                        queue.append((neighbour, depth + 1))

            # Exclude the starting node itself from results
            return {
                nid: path
                for nid, path in parent_map.items()
                if nid != entry_id
            }

        return await self._run_sync(_do)

    # ── Community detection ───────────────────────────────────────────────

    async def get_clusters(
        self,
        min_size: int = 3,
        max_distance: float = 0.3,
    ) -> List[GraphCluster]:
        """Detect communities using label propagation.

        The algorithm proceeds as follows:

        1. Build an undirected adjacency structure from all edges.
        2. Initialise each node with a unique label.
        3. Iterate: each node adopts the most frequent label among its
           neighbours (weighted by edge weight), with a preference for the
           current label as a tiebreaker.
        4. Repeat until convergence (no label changes) or a maximum number
           of iterations is reached.
        5. Filter clusters by *min_size* and compute coherence scores.

        Parameters
        ----------
        min_size:
            Discard clusters with fewer than this many members.
        max_distance:
            Maximum weight threshold — edges with weight below this
            value are ignored during propagation (acts as a distance
            cutoff).

        Returns
        -------
        list[GraphCluster]
            Clusters sorted by descending coherence score.
        """
        await self._ensure_cache()

        def _do() -> List[GraphCluster]:
            all_nodes = self._node_ids.copy()
            if not all_nodes:
                return []

            # Build undirected adjacency: node -> {neighbour: max_weight}
            adj: Dict[str, Dict[str, float]] = defaultdict(dict)
            for src, edges in self._outgoing.items():
                for edge in edges:
                    if edge.weight >= max_distance:
                        adj[src][edge.to_id] = max(
                            adj[src].get(edge.to_id, 0.0), edge.weight
                        )
                        adj[edge.to_id][src] = max(
                            adj[edge.to_id].get(src, 0.0), edge.weight
                        )

            # Initialise labels: each node starts with its own id.
            labels: Dict[str, str] = {nid: nid for nid in all_nodes}

            # Label propagation iterations
            max_iter = min(len(all_nodes), 100)
            node_list = sorted(all_nodes)

            for _ in range(max_iter):
                changed = False
                # Shuffle iteration order for non-determinism break
                import random
                shuffled = list(node_list)
                random.shuffle(shuffled)

                for node in shuffled:
                    neighbours = adj.get(node, {})
                    if not neighbours:
                        continue

                    # Weighted vote among neighbours
                    label_weights: Dict[str, float] = defaultdict(float)
                    for nbr, w in neighbours.items():
                        label_weights[labels[nbr]] += w

                    # Tiebreak: prefer current label
                    label_weights[labels[node]] += 0.01

                    best_label = max(label_weights, key=label_weights.get)
                    if best_label != labels[node]:
                        labels[node] = best_label
                        changed = True

                if not changed:
                    break

            # Group by label
            groups: Dict[str, Set[str]] = defaultdict(set)
            for node, lbl in labels.items():
                groups[lbl].add(node)

            # Build GraphCluster objects
            clusters: List[GraphCluster] = []
            for lbl, members in groups.items():
                if len(members) < min_size:
                    continue

                member_list = sorted(members)
                centroid_id = self._find_centroid_sync(member_list, adj)
                coherence = self._compute_coherence_sync(member_list, adj)
                label_text = self._cluster_label_sync(member_list)

                clusters.append(GraphCluster(
                    cluster_id=uuid.uuid4().hex[:12],
                    entry_ids=member_list,
                    centroid_id=centroid_id,
                    label=label_text,
                    coherence_score=coherence,
                ))

            clusters.sort(key=lambda c: c.coherence_score, reverse=True)
            return clusters

        return await self._run_sync(_do)

    def _find_centroid_sync(
        self,
        members: List[str],
        adj: Dict[str, Dict[str, float]],
    ) -> str:
        """Return the member with the highest within-group weighted degree."""
        best_id = members[0]
        best_score = -1.0
        member_set = set(members)
        for nid in members:
            score = sum(
                w for nbr, w in adj.get(nid, {}).items() if nbr in member_set
            )
            if score > best_score:
                best_score = score
                best_id = nid
        return best_id

    def _compute_coherence_sync(
        self,
        members: List[str],
        adj: Dict[str, Dict[str, float]],
    ) -> float:
        """Compute average pairwise edge weight among cluster members."""
        member_set = set(members)
        total_weight = 0.0
        pair_count = 0
        for nid in members:
            for nbr, w in adj.get(nid, {}).items():
                if nbr in member_set and nid < nbr:
                    total_weight += w
                    pair_count += 1
        if pair_count == 0:
            return 0.0
        return round(total_weight / pair_count, 4)

    def _cluster_label_sync(self, members: List[str]) -> str:
        """Derive a label from the most common tags across cluster members.

        Queries the ``knowledge_tags`` junction table first (preferred),
        then falls back to the ``tags_json`` column on ``knowledge_entries``.
        Returns the first member id truncated if no tags are found.
        """
        conn = self._ensure_conn()
        tag_counter: Counter = Counter()
        placeholders = ",".join("?" * len(members))

        # Strategy 1: junction table (preferred, matches KnowledgeStore schema)
        try:
            rows = conn.execute(
                f"SELECT tag, COUNT(*) AS cnt FROM knowledge_tags "
                f"WHERE entry_id IN ({placeholders}) "
                f"GROUP BY tag ORDER BY cnt DESC LIMIT 10",
                members,
            ).fetchall()
            for row in rows:
                tag_counter[row["tag"]] = row["cnt"]
        except Exception:
            pass

        # Strategy 2: tags_json column (fallback)
        if not tag_counter:
            try:
                rows = conn.execute(
                    f"SELECT tags_json FROM knowledge_entries "
                    f"WHERE id IN ({placeholders})",
                    members,
                ).fetchall()
                for row in rows:
                    try:
                        tags = json.loads(row["tags_json"]) if row["tags_json"] else []
                        for t in tags:
                            tag_counter[t] += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
            except Exception:
                pass

        if tag_counter:
            return ", ".join(t for t, _ in tag_counter.most_common(3))

        return members[0][:30] if members else ""

    # ── Related-entry suggestions ─────────────────────────────────────────

    async def suggest_related(
        self,
        entry_id: str,
        limit: int = 10,
    ) -> List[Tuple[str, float]]:
        """Suggest entries related to *entry_id* based on graph structure.

        Scoring combines:
        * **Graph proximity**: BFS distance weighted by edge weights.
        * **Tag overlap**: Jaccard similarity of tag sets.

        Parameters
        ----------
        entry_id:
            The entry to find suggestions for.
        limit:
            Maximum number of suggestions.

        Returns
        -------
        list[tuple[str, float]]
            ``(entry_id, score)`` pairs sorted by descending score.
        """
        await self._ensure_cache()

        def _do() -> List[Tuple[str, float]]:
            conn = self._ensure_conn()

            # 1. Load tags for the query entry
            query_tags: Set[str] = set()
            # Prefer junction table (KnowledgeStore schema)
            try:
                tag_rows = conn.execute(
                    "SELECT tag FROM knowledge_tags WHERE entry_id = ?",
                    (entry_id,),
                ).fetchall()
                query_tags = {r["tag"] for r in tag_rows}
            except Exception:
                pass
            if not query_tags:
                try:
                    row = conn.execute(
                        "SELECT tags_json FROM knowledge_entries WHERE id = ?",
                        (entry_id,),
                    ).fetchone()
                    if row and row["tags_json"]:
                        query_tags = set(json.loads(row["tags_json"]))
                except Exception:
                    pass

            # 2. BFS to collect reachable nodes with weighted distances
            visited: Dict[str, float] = {entry_id: 0.0}
            queue: deque[Tuple[str, float]] = deque()
            queue.append((entry_id, 0.0))

            while queue:
                current, dist = queue.popleft()
                if dist > 3:
                    continue

                for edge in self._outgoing.get(current, []):
                    new_dist = dist + (1.0 - edge.weight)
                    if edge.to_id not in visited or new_dist < visited[edge.to_id]:
                        visited[edge.to_id] = new_dist
                        queue.append((edge.to_id, new_dist))

                for edge in self._incoming.get(current, []):
                    new_dist = dist + (1.0 - edge.weight)
                    if edge.from_id not in visited or new_dist < visited[edge.from_id]:
                        visited[edge.from_id] = new_dist
                        queue.append((edge.from_id, new_dist))

            # 3. Score each candidate
            scores: List[Tuple[str, float]] = []
            for nid, dist in visited.items():
                if nid == entry_id:
                    continue

                # Graph proximity score (inverse distance, normalised)
                proximity = 1.0 / (1.0 + dist)

                # Tag overlap score (Jaccard)
                tag_score = 0.0
                if query_tags:
                    try:
                        other_tags: Set[str] = set()
                        tag_rows = conn.execute(
                            "SELECT tag FROM knowledge_tags WHERE entry_id = ?",
                            (nid,),
                        ).fetchall()
                        other_tags = {r["tag"] for r in tag_rows}
                        if not other_tags:
                            row = conn.execute(
                                "SELECT tags_json FROM knowledge_entries WHERE id = ?",
                                (nid,),
                            ).fetchone()
                            if row and row["tags_json"]:
                                other_tags = set(json.loads(row["tags_json"]))
                        if other_tags:
                            intersection = len(query_tags & other_tags)
                            union = len(query_tags | other_tags)
                            tag_score = intersection / union if union > 0 else 0.0
                    except Exception:
                        pass

                # Combined score: 60% graph + 40% tags
                combined = 0.6 * proximity + 0.4 * tag_score
                scores.append((nid, round(combined, 4)))

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:limit]

        return await self._run_sync(_do)

    # ── Orphan detection ──────────────────────────────────────────────────

    async def get_orphan_entries(self, store: Any = None) -> List[str]:
        """Find knowledge entries that have no connections in the graph.

        If *store* is provided it should expose a ``list_entries()`` async
        method (or similar) that returns all entry ids.  If *store* is
        ``None`` the method falls back to querying a ``knowledge_entries``
        table directly.

        Parameters
        ----------
        store:
            Optional knowledge store object.

        Returns
        -------
        list[str]
            Entry ids with zero incoming or outgoing edges.
        """
        await self._ensure_cache()

        def _get_all_entry_ids() -> Set[str]:
            conn = self._ensure_conn()
            ids: Set[str] = set()

            if store is not None:
                # Try calling list_entries() on the store
                try:
                    all_entries = store.list_entries()  # type: ignore[union-attr]
                    if asyncio.iscoroutine(all_entries):
                        raise RuntimeError(
                            "list_entries() is async — pass store to "
                            "get_orphan_entries before calling _run_sync"
                        )
                    for entry in all_entries:
                        if isinstance(entry, dict):
                            ids.add(entry.get("id", ""))
                        elif hasattr(entry, "id"):
                            ids.add(str(entry.id))
                except Exception:
                    pass

            if not ids:
                # Prefer non-archived entries (KnowledgeStore convention)
                try:
                    rows = conn.execute(
                        "SELECT id FROM knowledge_entries WHERE archived = 0"
                    ).fetchall()
                    for r in rows:
                        ids.add(r["id"])
                except Exception:
                    try:
                        rows = conn.execute(
                            "SELECT id FROM knowledge_entries"
                        ).fetchall()
                        for r in rows:
                            ids.add(r["id"])
                    except Exception:
                        pass

            return ids

        all_ids = await self._run_sync(_get_all_entry_ids)
        orphans = sorted(all_ids - self._node_ids)
        return orphans

    # ── Auto-linking ──────────────────────────────────────────────────────

    async def auto_link_entries(
        self,
        store: Any = None,
        threshold: float = 0.7,
    ) -> int:
        """Automatically link entries that share overlapping tags.

        For every pair of unconnected entries, compute the Jaccard
        similarity of their tag sets.  If the similarity exceeds
        *threshold*, create a ``RELATED_TO`` edge with weight equal to
        the similarity score.

        Parameters
        ----------
        store:
            Optional knowledge store (not used if entries are in the DB).
        threshold:
            Minimum Jaccard similarity to create a link (default 0.7).

        Returns
        -------
        int
            Number of new edges created.
        """
        await self._ensure_cache()

        def _do() -> int:
            conn = self._ensure_conn()
            entries: Dict[str, Set[str]] = {}

            try:
                # Load entry ids (non-archived) and their tags.
                # Prefer junction table; fall back to tags_json column.
                entry_rows = conn.execute(
                    "SELECT id FROM knowledge_entries WHERE archived = 0"
                ).fetchall()
                entry_ids_in_db = {r["id"] for r in entry_rows}

                for eid in entry_ids_in_db:
                    tags: Set[str] = set()
                    try:
                        tag_rows = conn.execute(
                            "SELECT tag FROM knowledge_tags WHERE entry_id = ?",
                            (eid,),
                        ).fetchall()
                        tags = {tr["tag"] for tr in tag_rows}
                    except Exception:
                        pass
                    if not tags:
                        try:
                            row = conn.execute(
                                "SELECT tags_json FROM knowledge_entries WHERE id = ?",
                                (eid,),
                            ).fetchone()
                            if row and row["tags_json"]:
                                tags = set(json.loads(row["tags_json"]))
                        except Exception:
                            pass
                    entries[eid] = tags
            except Exception:
                return 0

            entry_ids = sorted(entries.keys())
            new_edges = 0

            # Collect the set of existing edges for fast lookup
            existing: Set[Tuple[str, str]] = set()
            for src, edges in self._outgoing.items():
                for edge in edges:
                    existing.add((src, edge.to_id))

            for i, id_a in enumerate(entry_ids):
                tags_a = entries[id_a]
                if not tags_a:
                    continue

                for id_b in entry_ids[i + 1:]:
                    tags_b = entries[id_b]
                    if not tags_b:
                        continue

                    # Skip if already connected in either direction
                    if (id_a, id_b) in existing or (id_b, id_a) in existing:
                        continue

                    intersection = len(tags_a & tags_b)
                    union = len(tags_a | tags_b)
                    if union == 0:
                        continue

                    similarity = intersection / union
                    if similarity >= threshold:
                        weight = round(similarity, 4)
                        conn.execute(
                            "INSERT OR REPLACE INTO knowledge_relations "
                            "(from_id, to_id, relation_type, weight, metadata) "
                            "VALUES (?, ?, ?, ?, '{}')",
                            (id_a, id_b, RelationType.RELATED_TO.value, weight),
                        )
                        existing.add((id_a, id_b))
                        new_edges += 1

            if new_edges > 0:
                conn.commit()

            return new_edges

        count = await self._run_sync(_do)
        if count > 0:
            self._cache_dirty = True
            await self._rebuild_cache()
        return count

    # ── Graph statistics ──────────────────────────────────────────────────

    async def get_graph_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the knowledge graph.

        Keys
        ----
        node_count:
            Number of distinct nodes.
        edge_count:
            Number of directed edges.
        relation_type_counts:
            Breakdown by :class:`RelationType`.
        avg_degree:
            Average (undirected) degree across all nodes.
        max_degree:
            Maximum degree.
        connected_components:
            Number of connected components.
        density:
            Ratio of actual edges to possible edges.
        """
        await self._ensure_cache()

        def _do() -> Dict[str, Any]:
            conn = self._ensure_conn()

            # Edge count and type breakdown
            rows = conn.execute(
                "SELECT relation_type, COUNT(*) AS cnt, AVG(weight) AS avg_w "
                "FROM knowledge_relations GROUP BY relation_type"
            ).fetchall()

            edge_count = 0
            type_counts: Dict[str, int] = {}
            type_avg_weights: Dict[str, float] = {}
            for r in rows:
                edge_count += r["cnt"]
                type_counts[r["relation_type"]] = r["cnt"]
                type_avg_weights[r["relation_type"]] = round(r["avg_w"], 4) if r["avg_w"] else 0.0

            node_count = len(self._node_ids)

            # Degree computation
            degrees: Dict[str, int] = defaultdict(int)
            for nid in self._node_ids:
                deg = (
                    len(self._outgoing.get(nid, []))
                    + len(self._incoming.get(nid, []))
                )
                degrees[nid] = deg

            avg_degree = round(
                sum(degrees.values()) / node_count, 2
            ) if node_count > 0 else 0.0

            max_degree = max(degrees.values()) if degrees else 0

            # Connected components via BFS
            visited: Set[str] = set()
            components = 0
            all_nodes_sorted = sorted(self._node_ids)

            for node in all_nodes_sorted:
                if node in visited:
                    continue
                components += 1
                queue = deque([node])
                visited.add(node)
                while queue:
                    current = queue.popleft()
                    for edge in self._outgoing.get(current, []):
                        if edge.to_id not in visited:
                            visited.add(edge.to_id)
                            queue.append(edge.to_id)
                    for edge in self._incoming.get(current, []):
                        if edge.from_id not in visited:
                            visited.add(edge.from_id)
                            queue.append(edge.from_id)

            # Density: for an undirected view, max edges = n*(n-1)/2
            max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 1
            density = round(edge_count / max_edges, 4) if max_edges > 0 else 0.0

            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "relation_type_counts": type_counts,
                "relation_type_avg_weights": type_avg_weights,
                "avg_degree": avg_degree,
                "max_degree": max_degree,
                "connected_components": components,
                "density": density,
            }

        return await self._run_sync(_do)

    # ── ASCII visualisation ───────────────────────────────────────────────

    async def visualize(
        self,
        entry_id: Optional[str] = None,
        max_nodes: int = _DEFAULT_VIS_MAX_NODES,
    ) -> str:
        """Generate an ASCII-art graph visualisation.

        If *entry_id* is given, visualises the sub-graph reachable from that
        node (up to *max_nodes*).  Otherwise shows the full graph limited to
        *max_nodes* nodes.

        Parameters
        ----------
        entry_id:
            Optional focal node id.
        max_nodes:
            Maximum number of nodes to include.

        Returns
        -------
        str
            Multi-line ASCII string using box-drawing characters.
        """
        await self._ensure_cache()

        def _do() -> str:
            conn = self._ensure_conn()

            # Collect node metadata (title, category) from the store
            node_meta: Dict[str, Dict[str, str]] = {}

            try:
                rows = conn.execute(
                    "SELECT id, title, category FROM knowledge_entries"
                ).fetchall()
                for r in rows:
                    node_meta[r["id"]] = {
                        "title": r["title"] or r["id"][:16],
                        "category": r["category"] or "",
                    }
            except Exception:
                pass

            # Determine which nodes to include
            focus_nodes: Set[str] = set()
            edges_to_show: List[GraphEdge] = []

            if entry_id:
                # BFS from entry_id
                visited: Set[str] = set()
                queue = deque([entry_id])
                visited.add(entry_id)

                while queue and len(visited) <= max_nodes:
                    current = queue.popleft()
                    focus_nodes.add(current)

                    for edge in self._outgoing.get(current, []):
                        if edge.to_id not in visited and len(visited) < max_nodes:
                            visited.add(edge.to_id)
                            queue.append(edge.to_id)
                        edges_to_show.append(edge)

                    for edge in self._incoming.get(current, []):
                        if edge.from_id not in visited and len(visited) < max_nodes:
                            visited.add(edge.from_id)
                            queue.append(edge.from_id)
                        if (edge.from_id in focus_nodes or edge.from_id == entry_id):
                            edges_to_show.append(edge)
            else:
                # Show all nodes up to max_nodes
                for nid in sorted(self._node_ids)[:max_nodes]:
                    focus_nodes.add(nid)
                for nid in focus_nodes:
                    for edge in self._outgoing.get(nid, []):
                        if edge.to_id in focus_nodes:
                            edges_to_show.append(edge)

            if not focus_nodes:
                return "(empty graph)"

            # Build adjacency for rendering
            adj_render: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
            for edge in edges_to_show:
                if edge.from_id in focus_nodes and edge.to_id in focus_nodes:
                    # Normalise to avoid duplicates in undirected display
                    if edge.to_id < edge.from_id:
                        adj_render[edge.to_id].append(
                            (edge.from_id, edge.relation_type, edge.weight)
                        )
                    else:
                        adj_render[edge.from_id].append(
                            (edge.to_id, edge.relation_type, edge.weight)
                        )

            # Build lines
            lines: List[str] = []
            lines.append(f"{'─' * 60}")
            lines.append("  Knowledge Graph")
            if entry_id:
                label = node_meta.get(entry_id, {}).get("title", entry_id[:16])
                lines.append(f"  Focus: {label} ({entry_id[:12]})")
            lines.append(f"  Nodes: {len(focus_nodes)}  Edges: {len(edges_to_show)}")
            lines.append(f"{'─' * 60}")
            lines.append("")

            rendered: Set[str] = set()
            root_nodes = sorted(
                focus_nodes,
                key=lambda n: len(adj_render.get(n, [])),
                reverse=True,
            )

            for i, root in enumerate(root_nodes):
                if root in rendered:
                    continue
                rendered.add(root)

                meta = node_meta.get(root, {})
                title = meta.get("title", root[:20])
                category = meta.get("category", "")
                cat_label = f" [{category}]" if category else ""

                is_last_root = (i == len(root_nodes) - 1) or all(
                    n in rendered for n in root_nodes[i + 1:]
                )
                prefix = f"{_BOX_BL}{_BOX_H}{_BOX_H} " if is_last_root else f"{_BOX_LJ}{_BOX_H}{_BOX_H} "

                lines.append(f"{_BOX_TL}{_BOX_H}{_BOX_H} {title}{cat_label}")
                children = adj_render.get(root, [])

                for j, (child_id, rel_type, weight) in enumerate(sorted(children)):
                    if child_id in rendered and child_id != root:
                        continue
                    rendered.add(child_id)

                    child_meta = node_meta.get(child_id, {})
                    child_title = child_meta.get("title", child_id[:20])
                    child_cat = child_meta.get("category", "")
                    child_cat_label = f" [{child_cat}]" if child_cat else ""
                    weight_bar = _weight_bar(weight)

                    is_last_child = (j == len(children) - 1)
                    conn_char = _BOX_BL if is_last_child else _BOX_LJ

                    rel_short = rel_type[:10] if len(rel_type) > 10 else rel_type
                    lines.append(
                        f"{prefix}{_BOX_V}  {conn_char}{_BOX_H}{_BOX_H} "
                        f"{child_title}{child_cat_label} "
                        f"  {weight_bar} {rel_short}"
                    )

                # Separator between top-level trees
                if i < len(root_nodes) - 1:
                    lines.append(f"{prefix}  {_BOX_V}")
                    lines.append("")

            # Legend
            lines.append("")
            lines.append(f"{'─' * 60}")
            lines.append("  Legend:")
            lines.append(f"  {_BOX_TL}{_BOX_H}{_BOX_H} node      tree root")
            lines.append(f"  {_BOX_V}          vertical connector")
            lines.append(f"  {_BOX_LJ}{_BOX_H}{_BOX_H} child     branch connector")
            lines.append("  ░▒▓█        edge weight (low → high)")
            lines.append(f"{'─' * 60}")

            return "\n".join(lines)

        return await self._run_sync(_do)

    # ── Export ────────────────────────────────────────────────────────────

    async def export_graph(self, filepath: str) -> None:
        """Export the complete graph as a JSON file.

        The output contains two top-level keys:

        * ``"nodes"`` — list of node objects with ``id`` and ``title``
          (when available).
        * ``"edges"`` — list of edge objects.

        Parameters
        ----------
        filepath:
            Destination path.  Parent directories are created if needed.
        """
        await self._ensure_cache()

        def _do() -> None:
            conn = self._ensure_conn()

            # Load node metadata — try tags_json column (KnowledgeStore)
            node_data: Dict[str, Dict[str, Any]] = {}
            try:
                rows = conn.execute(
                    "SELECT id, title, category, tags_json FROM knowledge_entries"
                ).fetchall()
                for r in rows:
                    try:
                        tags = json.loads(r["tags_json"]) if r["tags_json"] else []
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                    node_data[r["id"]] = {
                        "id": r["id"],
                        "title": r.get("title", ""),
                        "category": r.get("category", ""),
                        "tags": tags,
                    }
                # Also enrich with junction-table tags
                tag_rows = conn.execute(
                    "SELECT entry_id, tag FROM knowledge_tags"
                ).fetchall()
                for tr in tag_rows:
                    nid = tr["entry_id"]
                    if nid in node_data:
                        node_data[nid]["tags"].append(tr["tag"])
                    else:
                        node_data[nid] = {
                            "id": nid, "title": "", "category": "", "tags": [tr["tag"]],
                        }
                # Deduplicate tags
                for nd in node_data.values():
                    nd["tags"] = sorted(set(nd["tags"]))
            except Exception:
                pass

            # Build nodes list (include nodes that only appear in edges)
            nodes: List[Dict[str, Any]] = []
            for nid in sorted(self._node_ids):
                nd = node_data.get(nid, {"id": nid, "title": "", "category": "", "tags": []})
                nodes.append(nd)

            # Build edges list
            edges: List[Dict[str, Any]] = []
            existing_cols = {row["name"] for row in
                            conn.execute("PRAGMA table_info(knowledge_relations)").fetchall()}
            has_weight = "weight" in existing_cols
            has_metadata = "metadata" in existing_cols

            if has_weight and has_metadata:
                edge_rows = conn.execute(
                    "SELECT from_id, to_id, relation_type, weight, metadata "
                    "FROM knowledge_relations"
                ).fetchall()
            else:
                edge_rows = conn.execute(
                    "SELECT from_id, to_id, relation_type, "
                    "COALESCE(weight, 0.5) AS weight, "
                    "COALESCE(metadata, '{}') AS metadata "
                    "FROM knowledge_relations"
                ).fetchall()

            for r in edge_rows:
                meta = json.loads(r["metadata"]) if r["metadata"] else {}
                edges.append({
                    "from": r["from_id"],
                    "to": r["to_id"],
                    "relation_type": r["relation_type"],
                    "weight": r["weight"],
                    "metadata": meta,
                })

            payload = {
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
            }

            path = Path(filepath).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )

        await self._run_sync(_do)

    # ── PageRank ──────────────────────────────────────────────────────────

    async def compute_pagerank(
        self,
        damping: float = _PAGERANK_DAMPING,
        tolerance: float = _PAGERANK_TOLERANCE,
        max_iter: int = _PAGERANK_MAX_ITER,
    ) -> Dict[str, float]:
        """Compute PageRank scores for all nodes in the graph.

        Implements the iterative PageRank algorithm treating the directed
        graph as a web graph.  Dangling nodes (no outgoing edges)
        redistribute their rank equally to all nodes.

        Parameters
        ----------
        damping:
            Probability of following a link (default 0.85).
        tolerance:
            Convergence threshold (default 1e-6).
        max_iter:
            Maximum iterations (default 200).

        Returns
        -------
        dict[str, float]
            Mapping of node id to PageRank score (normalised to sum to 1.0).
        """
        await self._ensure_cache()

        def _do() -> Dict[str, float]:
            all_nodes = sorted(self._node_ids)
            n = len(all_nodes)
            if n == 0:
                return {}
            if n == 1:
                return {all_nodes[0]: 1.0}

            node_index = {nid: i for i, nid in enumerate(all_nodes)}

            # Build out-degree counts and adjacency
            out_degree: Dict[str, int] = defaultdict(int)
            for src, edges in self._outgoing.items():
                out_degree[src] = len(edges)

            # Identify dangling nodes (no outgoing edges)
            dangling: Set[str] = set(all_nodes) - set(out_degree.keys())

            # Initialise ranks uniformly
            rank: Dict[str, float] = {nid: 1.0 / n for nid in all_nodes}

            for iteration in range(max_iter):
                new_rank: Dict[str, float] = {nid: 0.0 for nid in all_nodes}

                # Dangling node contribution
                dangling_sum = sum(rank[nid] for nid in dangling)
                dangling_contrib = damping * dangling_sum / n

                # Contributions from regular edges
                for src, edges in self._outgoing.items():
                    src_rank = rank[src]
                    out_deg = out_degree[src]
                    if out_deg == 0:
                        continue
                    contrib = damping * src_rank / out_deg
                    for edge in edges:
                        if edge.to_id in new_rank:
                            new_rank[edge.to_id] += contrib

                # Apply teleportation and dangling contribution
                for nid in all_nodes:
                    new_rank[nid] += (1.0 - damping) / n + dangling_contrib

                # Check convergence (L1 norm of difference)
                diff = sum(
                    abs(new_rank[nid] - rank[nid]) for nid in all_nodes
                )
                rank = new_rank

                if diff < tolerance:
                    break

            # Normalise so scores sum to 1.0
            total = sum(rank.values())
            if total > 0:
                rank = {nid: score / total for nid, score in rank.items()}

            # Round for readability
            return {nid: round(score, 6) for nid, score in rank.items()}

        return await self._run_sync(_do)


# ──────────────────────────────────────────────────────────────────────────────
# ASCII helpers
# ──────────────────────────────────────────────────────────────────────────────

def _weight_bar(weight: float, width: int = 4) -> str:
    """Return a small Unicode bar representing an edge weight.

    The bar uses characters ``░ ▒ ▓ █`` to encode weight from 0.0 to 1.0.
    """
    if weight <= 0.0:
        return "░" * width
    if weight >= 1.0:
        return "█" * width
    filled = int(round(weight * width))
    chars = ["░", "▒", "▓", "█"]
    parts: List[str] = []
    for i in range(width):
        if i < filled:
            parts.append(chars[min(filled - 1, len(chars) - 1)])
        else:
            parts.append(chars[0])
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience re-exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "RelationType",
    "GraphNode",
    "GraphEdge",
    "GraphCluster",
    "KnowledgeGraph",
    "_weight_bar",
]
