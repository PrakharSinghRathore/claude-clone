"""
Evolution Tracker — The DNA of Self-Improvement.

Tracks the complete evolutionary history of the self-improving system:
- Maintains a chronological lineage of all modifications
- Calculates improvement metrics over time (code quality, performance, capability)
- Generates "evolution reports" showing growth trajectory
- Tracks which changes were successful and which were rolled back
- Builds a "knowledge graph" of what was tried and what worked
- Provides "evolution score" — a single number representing overall improvement
- Enables "time travel" — comparing any two points in evolution

This module is purely observational — it doesn't modify anything.

Usage:
    tracker = EvolutionTracker(project_root="/path/to/claude_clone")
    await tracker.initialize()
    score = await tracker.get_evolution_score()
    report = await tracker.generate_evolution_report()
    lineage = await tracker.get_change_lineage()
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "~/.claude_clone/self_improve_evolution.db"

# Evolution dimensions and their weights
EVOLUTION_DIMENSIONS = {
    "code_quality": {"weight": 0.25, "description": "Code quality score (0-100)"},
    "performance": {"weight": 0.20, "description": "Performance improvements"},
    "capabilities": {"weight": 0.20, "description": "New tools/capabilities added"},
    "bug_fixes": {"weight": 0.15, "description": "Bugs found and fixed"},
    "user_satisfaction": {"weight": 0.10, "description": "User feedback scores"},
    "stability": {"weight": 0.10, "description": "Rollback rate and stability"},
}


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvolutionEvent:
    """A single event in the evolution timeline."""
    event_id: str
    event_type: str  # "patch", "optimize", "extend", "rollback", "evaluation", "learning"
    file_path: str
    description: str
    before_score: float
    after_score: float
    improvement: float  # Positive = improvement, negative = regression
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionScore:
    """Composite evolution score across all dimensions."""
    overall_score: float  # 0.0 to 1.0
    dimensions: Dict[str, float]
    timestamp: str
    generation: int  # How many improvement cycles have completed
    total_changes: int
    total_improvements: int
    total_regressions: int
    total_rollbacks: int
    stability_rate: float
    trajectory: str  # "improving", "stable", "declining"


@dataclass
class ChangeLineage:
    """Lineage of changes to a specific file."""
    file_path: str
    total_changes: int
    first_modified: str
    last_modified: str
    events: List[EvolutionEvent]
    net_improvement: float
    current_score: float
    evolution_stages: List[Dict[str, Any]]


@dataclass
class Generation:
    """A single improvement cycle/generation."""
    generation_id: int
    start_time: str
    end_time: str
    events: List[EvolutionEvent]
    patches_attempted: int
    patches_applied: int
    optimizations_attempted: int
    extensions_added: int
    rollbacks: int
    score_before: float
    score_after: float
    improvement: float


# ──────────────────────────────────────────────────────────────────────────────
# EvolutionTracker
# ──────────────────────────────────────────────────────────────────────────────

class EvolutionTracker:
    """
    Tracks and analyzes the evolutionary history of the self-improving system.

    Each time the system makes a modification (patch, optimization, extension),
    an evolution event is recorded. The tracker aggregates these events to
    compute:
    - Overall evolution score (weighted composite of all dimensions)
    - Per-dimension trends over time
    - File-level evolution lineage
    - Generation-level improvement metrics
    - Stability and regression detection
    """

    def __init__(self, project_root: str, db_path: str = DEFAULT_DB_PATH):
        self.project_root = Path(project_root).resolve()
        self.db_path = Path(db_path).expanduser().resolve()
        self._conn = None
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(self._connect)
        await self._run_sync(self._create_tables)
        self._initialized = True

    async def close(self) -> None:
        if self._conn:
            await self._run_sync(self._conn.close)
            self._conn = None

    def _connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS evolution_events (
                event_id      TEXT PRIMARY KEY,
                event_type    TEXT NOT NULL,
                file_path     TEXT NOT NULL,
                description   TEXT NOT NULL,
                before_score  REAL DEFAULT 0.0,
                after_score   REAL DEFAULT 0.0,
                improvement   REAL DEFAULT 0.0,
                timestamp     TEXT NOT NULL,
                generation    INTEGER DEFAULT 0,
                metadata      TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS evolution_snapshots (
                snapshot_id   TEXT PRIMARY KEY,
                timestamp     TEXT NOT NULL,
                overall_score REAL DEFAULT 0.0,
                dimensions    TEXT DEFAULT '{}',
                generation    INTEGER DEFAULT 0,
                total_changes INTEGER DEFAULT 0,
                total_improvements INTEGER DEFAULT 0,
                total_regressions INTEGER DEFAULT 0,
                total_rollbacks INTEGER DEFAULT 0,
                stability_rate REAL DEFAULT 1.0,
                trajectory    TEXT DEFAULT 'stable'
            );

            CREATE TABLE IF NOT EXISTS generations (
                generation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time     TEXT NOT NULL,
                end_time       TEXT NOT NULL,
                patches_attempted INTEGER DEFAULT 0,
                patches_applied   INTEGER DEFAULT 0,
                optimizations      INTEGER DEFAULT 0,
                extensions         INTEGER DEFAULT 0,
                rollbacks          INTEGER DEFAULT 0,
                score_before    REAL DEFAULT 0.0,
                score_after     REAL DEFAULT 0.0,
                improvement     REAL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_events_type ON evolution_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_file ON evolution_events(file_path);
            CREATE INDEX IF NOT EXISTS idx_events_ts ON evolution_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_gen ON evolution_events(generation);
        """)
        self._conn.commit()

    async def _run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def _ensure_conn(self):
        if self._conn is None:
            raise RuntimeError("EvolutionTracker not initialized.")

    # ── Event Recording ───────────────────────────────────────────────────

    async def record_event(
        self,
        event_type: str,
        file_path: str,
        description: str,
        before_score: float = 0.0,
        after_score: float = 0.0,
        generation: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record an evolution event."""
        import hashlib
        event_id = hashlib.sha256(
            f"{event_type}:{file_path}:{description}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]
        improvement = after_score - before_score
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO evolution_events "
                "(event_id, event_type, file_path, description, before_score, after_score, "
                "improvement, timestamp, generation, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, event_type, file_path, description, before_score, after_score,
                 improvement, now, generation, json.dumps(metadata or {}, default=str)),
            )
            conn.commit()

        await self._run_sync(_do)
        return event_id

    async def start_generation(self) -> int:
        """Start a new improvement generation cycle."""
        def _do() -> int:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO generations (start_time, end_time) VALUES (?, ?)",
                (datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            row = conn.execute("SELECT last_insert_rowid()").fetchone()
            return row[0] if row else 0

        return await self._run_sync(_do)

    async def end_generation(self, generation_id: int, score_before: float, score_after: float) -> None:
        """End a generation cycle with final scores."""
        improvement = score_after - score_before

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "UPDATE generations SET end_time=?, score_before=?, score_after=?, improvement=? WHERE generation_id=?",
                (datetime.now(timezone.utc).isoformat(), score_before, score_after, improvement, generation_id),
            )
            conn.commit()

        await self._run_sync(_do)

    # ── Evolution Score ───────────────────────────────────────────────────

    async def get_evolution_score(self) -> EvolutionScore:
        """Calculate the current evolution score."""
        def _do() -> Tuple[List[Dict], int, List[Dict]]:
            conn = self._ensure_conn()
            events = [dict(r) for r in conn.execute(
                "SELECT * FROM evolution_events ORDER BY timestamp ASC"
            ).fetchall()]
            generations = [dict(r) for r in conn.execute(
                "SELECT * FROM generations ORDER BY generation_id DESC LIMIT 10"
            ).fetchall()]
            return events, len(events), generations

        events, total_changes, recent_gens = await self._run_sync(_do)

        if not events:
            return EvolutionScore(
                overall_score=0.0,
                dimensions={d: 0.0 for d in EVOLUTION_DIMENSIONS},
                timestamp=datetime.now(timezone.utc).isoformat(),
                generation=0,
                total_changes=0,
                total_improvements=0,
                total_regressions=0,
                total_rollbacks=0,
                stability_rate=1.0,
                trajectory="stable",
            )

        # Calculate per-dimension scores
        dimensions: Dict[str, float] = {}
        type_counts: Dict[str, int] = Counter(e["event_type"] for e in events)

        for dim_name, dim_config in EVOLUTION_DIMENSIONS.items():
            if dim_name == "code_quality":
                # Average improvement from patches
                patches = [e for e in events if e["event_type"] == "patch" and e["improvement"] != 0]
                if patches:
                    avg_imp = sum(e["improvement"] for e in patches) / len(patches)
                    dimensions[dim_name] = max(0.0, min(1.0, 0.5 + avg_imp))
                else:
                    dimensions[dim_name] = 0.5

            elif dim_name == "performance":
                opts = [e for e in events if e["event_type"] == "optimize"]
                if opts:
                    avg_imp = sum(e["improvement"] for e in opts) / len(opts)
                    dimensions[dim_name] = max(0.0, min(1.0, 0.5 + avg_imp))
                else:
                    dimensions[dim_name] = 0.5

            elif dim_name == "capabilities":
                exts = type_counts.get("extend", 0)
                dimensions[dim_name] = min(1.0, 0.3 + exts * 0.1)

            elif dim_name == "bug_fixes":
                fixes = type_counts.get("patch", 0)
                dimensions[dim_name] = min(1.0, 0.3 + fixes * 0.05)

            elif dim_name == "user_satisfaction":
                evals = [e for e in events if e["event_type"] == "evaluation"]
                if evals:
                    dimensions[dim_name] = max(0.0, min(1.0, evals[-1].get("after_score", 0.5)))
                else:
                    dimensions[dim_name] = 0.5

            elif dim_name == "stability":
                rollbacks = type_counts.get("rollback", 0)
                total = total_changes
                dimensions[dim_name] = max(0.0, 1.0 - (rollbacks / max(1, total)) * 2)

        # Weighted composite
        overall = sum(
            dimensions.get(d, 0.5) * EVOLUTION_DIMENSIONS[d]["weight"]
            for d in EVOLUTION_DIMENSIONS
        )

        # Calculate summary stats
        improvements = sum(1 for e in events if e["improvement"] > 0.01)
        regressions = sum(1 for e in events if e["improvement"] < -0.01)
        rollbacks = type_counts.get("rollback", 0)
        stability = 1.0 - (rollbacks / max(1, total_changes))

        # Trajectory
        recent = events[-10:] if len(events) >= 10 else events
        recent_avg = sum(e["improvement"] for e in recent) / len(recent) if recent else 0
        if recent_avg > 0.02:
            trajectory = "improving"
        elif recent_avg < -0.02:
            trajectory = "declining"
        else:
            trajectory = "stable"

        # Current generation
        current_gen = events[-1].get("generation", 0) if events else 0

        return EvolutionScore(
            overall_score=round(overall, 3),
            dimensions={d: round(v, 3) for d, v in dimensions.items()},
            timestamp=datetime.now(timezone.utc).isoformat(),
            generation=current_gen,
            total_changes=total_changes,
            total_improvements=improvements,
            total_regressions=regressions,
            total_rollbacks=rollbacks,
            stability_rate=round(stability, 3),
            trajectory=trajectory,
        )

    async def save_evolution_snapshot(self, score: EvolutionScore) -> str:
        """Save a point-in-time evolution snapshot."""
        import hashlib
        snapshot_id = hashlib.sha256(score.timestamp.encode()).hexdigest()[:12]

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO evolution_snapshots "
                "(snapshot_id, timestamp, overall_score, dimensions, generation, "
                "total_changes, total_improvements, total_regressions, total_rollbacks, "
                "stability_rate, trajectory) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id, score.timestamp, score.overall_score,
                    json.dumps(score.dimensions),
                    score.generation, score.total_changes, score.total_improvements,
                    score.total_regressions, score.total_rollbacks,
                    score.stability_rate, score.trajectory,
                ),
            )
            conn.commit()

        await self._run_sync(_do)
        return snapshot_id

    # ── Lineage & History ────────────────────────────────────────────────

    async def get_change_lineage(self, file_path: Optional[str] = None) -> List[ChangeLineage]:
        """Get the evolution lineage for files."""
        def _do() -> List[Dict]:
            conn = self._ensure_conn()
            if file_path:
                rows = conn.execute(
                    "SELECT * FROM evolution_events WHERE file_path = ? ORDER BY timestamp ASC",
                    (file_path,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM evolution_events ORDER BY timestamp ASC"
                ).fetchall()
            return [dict(r) for r in rows]

        events = await self._run_sync(_do)

        # Group by file
        file_events: Dict[str, List[EvolutionEvent]] = defaultdict(list)
        for e in events:
            evt = EvolutionEvent(
                event_id=e["event_id"],
                event_type=e["event_type"],
                file_path=e["file_path"],
                description=e["description"],
                before_score=e["before_score"],
                after_score=e["after_score"],
                improvement=e["improvement"],
                timestamp=e["timestamp"],
                metadata=json.loads(e.get("metadata", "{}")),
            )
            file_events[e["file_path"]].append(evt)

        lineages = []
        for fp, evts in file_events.items():
            if not evts:
                continue

            net_imp = sum(e.improvement for e in evts)
            current = evts[-1].after_score if evts else 0.0

            # Group into stages
            stages = []
            current_type = None
            stage_events = []
            for e in evts:
                if e.event_type != current_type:
                    if stage_events:
                        stages.append({
                            "type": current_type,
                            "count": len(stage_events),
                            "avg_improvement": round(sum(e2.improvement for e2 in stage_events) / len(stage_events), 3),
                        })
                    current_type = e.event_type
                    stage_events = [e]
                else:
                    stage_events.append(e)
            if stage_events:
                stages.append({
                    "type": current_type,
                    "count": len(stage_events),
                    "avg_improvement": round(sum(e2.improvement for e2 in stage_events) / len(stage_events), 3),
                })

            lineages.append(ChangeLineage(
                file_path=fp,
                total_changes=len(evts),
                first_modified=evts[0].timestamp,
                last_modified=evts[-1].timestamp,
                events=evts,
                net_improvement=round(net_imp, 3),
                current_score=current,
                evolution_stages=stages,
            ))

        lineages.sort(key=lambda l: -l.total_changes)
        return lineages

    async def get_generations(self, limit: int = 20) -> List[Generation]:
        """Get the history of improvement generations."""
        def _do() -> List[Dict]:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT * FROM generations ORDER BY generation_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

        gens = await self._run_sync(_do)
        return [
            Generation(
                generation_id=g["generation_id"],
                start_time=g["start_time"],
                end_time=g["end_time"],
                events=[],
                patches_attempted=g.get("patches_attempted", 0),
                patches_applied=g.get("patches_applied", 0),
                optimizations_attempted=g.get("optimizations", 0),
                extensions_added=g.get("extensions", 0),
                rollbacks=g.get("rollbacks", 0),
                score_before=g.get("score_before", 0.0),
                score_after=g.get("score_after", 0.0),
                improvement=g.get("improvement", 0.0),
            )
            for g in gens
        ]

    async def get_event_timeline(self, limit: int = 50) -> List[EvolutionEvent]:
        """Get the chronological timeline of all events."""
        def _do() -> List[Dict]:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT * FROM evolution_events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

        events = await self._run_sync(_do)
        return [
            EvolutionEvent(
                event_id=e["event_id"],
                event_type=e["event_type"],
                file_path=e["file_path"],
                description=e["description"],
                before_score=e["before_score"],
                after_score=e["after_score"],
                improvement=e["improvement"],
                timestamp=e["timestamp"],
                metadata=json.loads(e.get("metadata", "{}")),
            )
            for e in events
        ]

    async def compare_snapshots(self, snapshot_a: str, snapshot_b: str) -> Dict[str, Any]:
        """Compare two evolution snapshots."""
        def _do() -> Tuple[Optional[Dict], Optional[Dict]]:
            conn = self._ensure_conn()
            a = conn.execute("SELECT * FROM evolution_snapshots WHERE snapshot_id = ?", (snapshot_a,)).fetchone()
            b = conn.execute("SELECT * FROM evolution_snapshots WHERE snapshot_id = ?", (snapshot_b,)).fetchone()
            return (dict(a) if a else None, dict(b) if b else None)

        snap_a, snap_b = await self._run_sync(_do)
        if not snap_a or not snap_b:
            return {"error": "Snapshot not found"}

        dims_a = json.loads(snap_a.get("dimensions", "{}"))
        dims_b = json.loads(snap_b.get("dimensions", "{}"))

        return {
            "snapshot_a": snapshot_a,
            "snapshot_b": snapshot_b,
            "time_a": snap_a["timestamp"],
            "time_b": snap_b["timestamp"],
            "score_delta": round(snap_b["overall_score"] - snap_a["overall_score"], 3),
            "dimension_deltas": {
                d: round(dims_b.get(d, 0) - dims_a.get(d, 0), 3)
                for d in EVOLUTION_DIMENSIONS
            },
            "generation_delta": snap_b.get("generation", 0) - snap_a.get("generation", 0),
        }

    # ── Reports ──────────────────────────────────────────────────────────

    async def generate_evolution_report(self) -> str:
        """Generate a comprehensive evolution report."""
        score = await self.get_evolution_score()
        lineages = await self.get_change_lineage()
        generations = await self.get_generations()
        timeline = await self.get_event_timeline(limit=20)

        # Trajectory indicator
        trajectory_icons = {
            "improving": "📈",
            "stable": "➡️",
            "declining": "📉",
        }

        lines = [
            "# Evolution Report — Self-Improving System",
            "",
            f"**Overall Evolution Score:** {score.overall_score:.2f} / 1.00",
            f"**Trajectory:** {trajectory_icons.get(score.trajectory, '➡️')} {score.trajectory}",
            f"**Generation:** {score.generation}",
            f"**Total Changes:** {score.total_changes}",
            f"**Improvements:** {score.total_improvements} | **Regressions:** {score.total_regressions} | **Rollbacks:** {score.total_rollbacks}",
            f"**Stability Rate:** {score.stability_rate:.1%}",
            "",
            "## Dimension Scores",
        ]

        for dim_name, dim_config in EVOLUTION_DIMENSIONS.items():
            val = score.dimensions.get(dim_name, 0.0)
            bar = "\u2588" * int(val * 20)
            lines.append(f"- **{dim_name}** ({dim_config['description']}): {val:.2f} {bar}")

        lines.append("")

        if generations:
            lines.append("## Recent Generations")
            for gen in generations[:10]:
                status = "📈" if gen.improvement > 0.01 else ("📉" if gen.improvement < -0.01 else "➡️")
                lines.append(
                    f"- Gen {gen.generation_id}: {status} "
                    f"score {gen.score_before:.2f} → {gen.score_after:.2f} "
                    f"({gen.improvement:+.3f})"
                )
            lines.append("")

        if lineages:
            lines.append("## Most Evolved Files")
            for lin in lineages[:10]:
                status = "📈" if lin.net_improvement > 0 else ("📉" if lin.net_improvement < 0 else "➡️")
                lines.append(
                    f"- {status} `{lin.file_path}`: {lin.total_changes} changes, "
                    f"net improvement: {lin.net_improvement:+.3f}"
                )
            lines.append("")

        if timeline:
            lines.append("## Recent Events")
            for evt in reversed(timeline[:15]):
                icon = {
                    "patch": "🔧", "optimize": "⚡", "extend": "🧬",
                    "rollback": "⏪", "evaluation": "📊", "learning": "🧠",
                }.get(evt.event_type, "📝")
                imp_str = f" ({evt.improvement:+.3f})" if evt.improvement != 0 else ""
                lines.append(
                    f"- {icon} [{evt.event_type}] {evt.file_path}: {evt.description[:80]}{imp_str}"
                )

        return "\n".join(lines)
