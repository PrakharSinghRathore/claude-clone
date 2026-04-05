"""
Human Feedback Loop system.

Collects user feedback on agent responses (thumbs up/down, star ratings,
free-text comments), stores it in SQLite with full context, calculates
quality metrics, identifies weak areas, generates improvement suggestions,
and exports data for analysis or A/B comparison.

All database operations are async via ``sqlite3`` + ``asyncio.run_in_executor``.

Usage::

    fb = FeedbackCollector()
    await fb.initialize()
    await fb.submit_feedback(
        prompt="Explain async/await in Python",
        response="async/await is syntactic sugar for coroutines...",
        rating=FeedbackRating.THUMBS_UP,
        model="claude-sonnet-4-20250514",
        tokens=340,
        duration=1.2,
        task_type="explanation",
    )
    stats = await fb.get_stats()
    weak = await fb.identify_weak_areas()
"""

from __future__ import annotations

import asyncio
import enum
import json
import sqlite3
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "~/.claude_clone/feedback.db"

# Known task categories for classification.
_TASK_TYPES = [
    "explanation", "code_generation", "code_review", "debugging",
    "refactoring", "file_edit", "testing", "deployment", "security",
    "documentation", "planning", "general",
]


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class FeedbackRating(enum.Enum):
    """Supported feedback rating types."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    STAR_1 = "star_1"
    STAR_2 = "star_2"
    STAR_3 = "star_3"
    STAR_4 = "star_4"
    STAR_5 = "star_5"


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FeedbackEntry:
    """A single piece of user feedback with full context."""

    id: str
    session_id: str
    prompt: str
    response: str
    rating: str  # FeedbackRating.value
    comment: str = ""
    model: str = ""
    tokens: int = 0
    duration: float = 0.0
    timestamp: str = ""
    task_type: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _generate_id() -> str:
    return uuid.uuid4().hex[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rating_to_score(rating: str) -> float:
    """Convert a rating to a numeric score for averaging.

    Returns a value between 0 and 1 where 1 = best.
    """
    mapping = {
        FeedbackRating.THUMBS_UP.value: 1.0,
        FeedbackRating.THUMBS_DOWN.value: 0.0,
        FeedbackRating.STAR_1.value: 0.2,
        FeedbackRating.STAR_2.value: 0.4,
        FeedbackRating.STAR_3.value: 0.6,
        FeedbackRating.STAR_4.value: 0.8,
        FeedbackRating.STAR_5.value: 1.0,
    }
    return mapping.get(rating, 0.5)


def _is_positive(rating: str) -> bool:
    """Return True if the rating is generally positive."""
    score = _rating_to_score(rating)
    return score >= 0.6


def _row_to_entry(row: sqlite3.Row) -> FeedbackEntry:
    meta = json.loads(row["metadata"]) if row["metadata"] else {}
    return FeedbackEntry(
        id=row["id"],
        session_id=row["session_id"],
        prompt=row["prompt"],
        response=row["response"],
        rating=row["rating"],
        comment=row["comment"],
        model=row["model"],
        tokens=row["tokens"] or 0,
        duration=row["duration"] or 0.0,
        timestamp=row["timestamp"],
        task_type=row["task_type"],
        metadata=meta,
    )


# ──────────────────────────────────────────────────────────────────────────────
# FeedbackCollector
# ──────────────────────────────────────────────────────────────────────────────

class FeedbackCollector:
    """
    Collects, stores, and analyzes user feedback on agent responses.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  ``~`` is expanded automatically.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self._conn: Optional[sqlite3.Connection] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create the database schema and open the connection."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(self._connect)
        await self._run_sync(self._create_tables)

    async def close(self) -> None:
        if self._conn is not None:
            await self._run_sync(self._conn.close)
            self._conn = None

    def _connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS feedback (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                prompt      TEXT NOT NULL,
                response    TEXT NOT NULL,
                rating      TEXT NOT NULL,
                comment     TEXT DEFAULT '',
                model       TEXT DEFAULT '',
                tokens      INTEGER DEFAULT 0,
                duration    REAL DEFAULT 0.0,
                timestamp   TEXT NOT NULL,
                task_type   TEXT DEFAULT 'general',
                metadata    TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_fb_session
                ON feedback(session_id);

            CREATE INDEX IF NOT EXISTS idx_fb_rating
                ON feedback(rating);

            CREATE INDEX IF NOT EXISTS idx_fb_task_type
                ON feedback(task_type);

            CREATE INDEX IF NOT EXISTS idx_fb_timestamp
                ON feedback(timestamp);
        """)
        self._conn.commit()

    # ── Async wrapper ─────────────────────────────────────────────────────

    async def _run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("FeedbackCollector not initialized. Call `await fc.initialize()` first.")
        return self._conn

    # ── Feedback submission ───────────────────────────────────────────────

    async def submit_feedback(
        self,
        prompt: str,
        response: str,
        rating: FeedbackRating,
        session_id: str = "default",
        comment: str = "",
        model: str = "",
        tokens: int = 0,
        duration: float = 0.0,
        task_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a piece of user feedback.

        Parameters
        ----------
        prompt:
            The user's original prompt.
        response:
            The agent's response that was rated.
        rating:
            A :class:`FeedbackRating` enum value.
        session_id:
            Optional session identifier for grouping.
        comment:
            Optional free-text comment from the user.
        model:
            The model that generated the response.
        tokens:
            Number of tokens used.
        duration:
            Response generation time in seconds.
        task_type:
            Category of the task (e.g. ``"code_generation"``).
        metadata:
            Optional additional context.

        Returns
        -------
        str
            The feedback entry id.
        """
        entry_id = _generate_id()
        now = _now_iso()

        if isinstance(rating, FeedbackRating):
            rating_val = rating.value
        else:
            rating_val = str(rating)

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO feedback "
                "(id, session_id, prompt, response, rating, comment, model, "
                "tokens, duration, timestamp, task_type, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry_id, session_id, prompt, response, rating_val, comment,
                 model, tokens, duration, now, task_type,
                 json.dumps(metadata or {}, default=str)),
            )
            conn.commit()

        await self._run_sync(_do)
        return entry_id

    # ── Statistics ────────────────────────────────────────────────────────

    async def get_stats(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute overall quality metrics.

        Returns a dict with ``total_feedback``, ``approval_rate``,
        ``average_score``, ``rating_distribution``, ``model_breakdown``,
        ``total_tokens``, ``average_duration``, ``most_common_task_type``,
        and ``date_range``.
        """
        def _do() -> Dict[str, Any]:
            conn = self._ensure_conn()
            clauses: List[str] = []
            params: List[Any] = []
            if model is not None:
                clauses.append("model = ?")
                params.append(model)

            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

            rows = conn.execute(
                f"SELECT * FROM feedback{where} ORDER BY timestamp DESC", params
            ).fetchall()

            if not rows:
                return {
                    "total_feedback": 0,
                    "approval_rate": 0.0,
                    "average_score": 0.0,
                    "rating_distribution": {},
                    "model_breakdown": {},
                    "total_tokens": 0,
                    "average_duration": 0.0,
                }

            entries = [_row_to_entry(r) for r in rows]
            total = len(entries)
            positive = sum(1 for e in entries if _is_positive(e.rating))
            scores = [_rating_to_score(e.rating) for e in entries]
            avg_score = sum(scores) / len(scores)

            # Rating distribution.
            rating_counts: Counter = Counter()
            for e in entries:
                rating_counts[e.rating] += 1

            # Model breakdown.
            model_counts: Counter = Counter()
            for e in entries:
                key = e.model or "unknown"
                model_counts[key] += 1

            # Tokens & duration.
            total_tokens = sum(e.tokens for e in entries)
            durations = [e.duration for e in entries if e.duration > 0]
            avg_duration = sum(durations) / len(durations) if durations else 0.0

            # Task type frequency.
            task_counts: Counter = Counter()
            for e in entries:
                task_counts[e.task_type] += 1

            return {
                "total_feedback": total,
                "approval_rate": round(positive / total, 3),
                "average_score": round(avg_score, 3),
                "rating_distribution": dict(rating_counts.most_common()),
                "model_breakdown": dict(model_counts.most_common()),
                "total_tokens": total_tokens,
                "average_duration": round(avg_duration, 3),
                "task_type_distribution": dict(task_counts.most_common()),
                "date_range": {
                    "earliest": entries[-1].timestamp,
                    "latest": entries[0].timestamp,
                },
            }

        return await self._run_sync(_do)

    # ── Ratings by category ───────────────────────────────────────────────

    async def get_ratings_by_category(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute approval rate and average score broken down by ``task_type``.

        Returns a dict mapping each task type to ``{count, approval_rate,
        average_score, negative_count}``.
        """
        def _do() -> Dict[str, Dict[str, Any]]:
            conn = self._ensure_conn()
            rows = conn.execute("SELECT * FROM feedback ORDER BY timestamp DESC").fetchall()
            entries = [_row_to_entry(r) for r in rows]

            by_type: Dict[str, List[FeedbackEntry]] = defaultdict(list)
            for e in entries:
                by_type[e.task_type].append(e)

            result: Dict[str, Dict[str, Any]] = {}
            for task_type, items in by_type.items():
                count = len(items)
                positive = sum(1 for e in items if _is_positive(e.rating))
                negative = sum(1 for e in items if not _is_positive(e.rating))
                scores = [_rating_to_score(e.rating) for e in items]
                avg = sum(scores) / len(scores) if scores else 0.0

                result[task_type] = {
                    "count": count,
                    "approval_rate": round(positive / count, 3) if count else 0.0,
                    "average_score": round(avg, 3),
                    "negative_count": negative,
                }

            return result

        return await self._run_sync(_do)

    # ── Weak areas ────────────────────────────────────────────────────────

    async def identify_weak_areas(self, threshold: float = 0.6, min_entries: int = 3) -> List[Dict[str, Any]]:
        """
        Identify task categories with below-average satisfaction.

        Parameters
        ----------
        threshold:
            Categories with an average score below this value are flagged.
        min_entries:
            Minimum number of entries before a category is considered
            statistically meaningful.

        Returns
        -------
        list[dict]
            Sorted by ascending average score (worst first).  Each dict has
            ``task_type``, ``count``, ``average_score``, ``approval_rate``,
            and ``suggestion``.
        """
        by_cat = await self.get_ratings_by_category()

        weak: List[Dict[str, Any]] = []
        for task_type, info in by_cat.items():
            if info["count"] < min_entries:
                continue
            if info["average_score"] < threshold:
                suggestion = self._generate_suggestion(task_type, info)
                weak.append({
                    "task_type": task_type,
                    "count": info["count"],
                    "average_score": info["average_score"],
                    "approval_rate": info["approval_rate"],
                    "suggestion": suggestion,
                })

        weak.sort(key=lambda x: x["average_score"])
        return weak

    @staticmethod
    def _generate_suggestion(task_type: str, info: Dict[str, Any]) -> str:
        """Generate a human-readable improvement suggestion."""
        suggestions: Dict[str, str] = {
            "code_generation": (
                "Review code generation quality. Consider adding more examples in "
                "the system prompt and enabling post-generation validation / lint checks."
            ),
            "code_review": (
                "Code review feedback is below target. Ensure the agent examines "
                "full file context before making suggestions and references specific lines."
            ),
            "debugging": (
                "Debugging accuracy needs improvement. Verify the agent reads relevant "
                "logs and stack traces before proposing fixes."
            ),
            "explanation": (
                "Explanations are rated low. Encourage the agent to use concrete "
                "examples and avoid jargon when explaining concepts."
            ),
            "file_edit": (
                "File edits are getting negative feedback. Double-check that the agent "
                "reads files before editing and uses exact text matching."
            ),
            "testing": (
                "Test-related tasks are underperforming. Ensure the agent runs tests "
                "after writing them and validates output."
            ),
            "deployment": (
                "Deployment tasks need attention. Add deployment-specific context "
                "and validate commands before execution."
            ),
            "security": (
                "Security analysis is rated low. Consider running additional security "
                "scanners and providing more detailed reports."
            ),
            "documentation": (
                "Documentation quality is sub-par. Prompt the agent to include code "
                "examples, usage instructions, and parameter descriptions."
            ),
            "refactoring": (
                "Refactoring suggestions are not well-received. Ensure the agent "
                "maintains backward compatibility and runs tests after changes."
            ),
            "planning": (
                "Planning tasks need improvement. Encourage more detailed step-by-step "
                "plans with clear dependencies and risk assessments."
            ),
        }
        return suggestions.get(
            task_type,
            f"Consider reviewing and improving the agent's approach to {task_type} tasks. "
            f"Current approval rate is {info['approval_rate']:.0%} with an average score "
            f"of {info['average_score']:.2f}.",
        )

    # ── Report generation ─────────────────────────────────────────────────

    async def generate_report(self, model: Optional[str] = None) -> str:
        """
        Produce a comprehensive text report summarising all feedback.

        Includes overall stats, per-category breakdowns, weak areas, and
        recent negative feedback for review.
        """
        stats = await self.get_stats(model=model)
        by_cat = await self.get_ratings_by_category()
        weak = await self.identify_weak_areas()

        lines: List[str] = [
            "# Agent Feedback Report",
            "",
            f"**Total Feedback:** {stats['total_feedback']}",
            f"**Approval Rate:** {stats['approval_rate']:.1%}",
            f"**Average Score:** {stats['average_score']:.2f} / 1.00",
            f"**Total Tokens Processed:** {stats['total_tokens']:,}",
            f"**Average Response Duration:** {stats['average_duration']:.2f}s",
        ]

        if stats.get("date_range"):
            lines.append(
                f"**Date Range:** {stats['date_range']['earliest'][:10]} → "
                f"{stats['date_range']['latest'][:10]}"
            )

        lines.append("")
        lines.append("## Rating Distribution")
        for rating, count in stats.get("rating_distribution", {}).items():
            lines.append(f"- {rating}: {count}")

        if stats.get("model_breakdown"):
            lines.append("")
            lines.append("## Model Breakdown")
            for mdl, count in stats["model_breakdown"].items():
                lines.append(f"- {mdl}: {count} feedback entries")

        lines.append("")
        lines.append("## Ratings by Task Type")
        for task_type, info in sorted(by_cat.items(), key=lambda x: x[1]["average_score"]):
            bar = "\u2588" * int(info["average_score"] * 20)
            lines.append(
                f"- **{task_type}**: score={info['average_score']:.2f}  "
                f"approval={info['approval_rate']:.0%}  "
                f"(n={info['count']})  {bar}"
            )

        if weak:
            lines.append("")
            lines.append("## ⚠️ Weak Areas Requiring Improvement")
            for area in weak:
                lines.append(
                    f"- **{area['task_type']}** (score={area['average_score']:.2f}, "
                    f"n={area['count']}): {area['suggestion']}"
                )
        else:
            lines.append("")
            lines.append("## ✅ No weak areas identified (all categories above threshold)")

        # Recent negative feedback.
        recent_negative = await self.get_recent_feedback(limit=5, positive_only=False, model=model)
        negative_entries = [e for e in recent_negative if not _is_positive(e.rating)]
        if negative_entries:
            lines.append("")
            lines.append("## Recent Negative Feedback")
            for entry in negative_entries:
                prompt_preview = entry.prompt[:100] + ("..." if len(entry.prompt) > 100 else "")
                lines.append(f"- [{entry.rating}] \"{prompt_preview}\"")
                if entry.comment:
                    lines.append(f"  Comment: {entry.comment}")

        return "\n".join(lines)

    # ── Recent feedback ───────────────────────────────────────────────────

    async def get_recent_feedback(
        self,
        limit: int = 20,
        positive_only: bool = False,
        negative_only: bool = False,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[FeedbackEntry]:
        """
        Retrieve recent feedback entries with optional filters.

        Parameters
        ----------
        limit:
            Maximum entries to return.
        positive_only:
            Only return entries with positive ratings.
        negative_only:
            Only return entries with negative ratings.
        model:
            Filter by model name.
        task_type:
            Filter by task type.

        Returns
        -------
        list[FeedbackEntry]
            Most-recent first.
        """
        def _do() -> List[FeedbackEntry]:
            conn = self._ensure_conn()
            clauses: List[str] = []
            params: List[Any] = []

            if positive_only:
                # Scores >= 0.6
                thumbs_up = FeedbackRating.THUMBS_UP.value
                star_4 = FeedbackRating.STAR_4.value
                star_5 = FeedbackRating.STAR_5.value
                clauses.append("rating IN (?, ?, ?)")
                params.extend([thumbs_up, star_4, star_5])
            elif negative_only:
                thumbs_down = FeedbackRating.THUMBS_DOWN.value
                star_1 = FeedbackRating.STAR_1.value
                star_2 = FeedbackRating.STAR_2.value
                clauses.append("rating IN (?, ?, ?)")
                params.extend([thumbs_down, star_1, star_2])

            if model is not None:
                clauses.append("model = ?")
                params.append(model)
            if task_type is not None:
                clauses.append("task_type = ?")
                params.append(task_type)

            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = conn.execute(
                f"SELECT * FROM feedback{where} ORDER BY timestamp DESC LIMIT ?",
                params + [limit],
            ).fetchall()
            return [_row_to_entry(r) for r in rows]

        return await self._run_sync(_do)

    # ── Export ────────────────────────────────────────────────────────────

    async def export_data(
        self,
        filepath: Optional[str] = None,
        format: str = "json",  # noqa: A002
    ) -> str:
        """
        Export all feedback data.

        Parameters
        ----------
        filepath:
            If given, write the output to this file.
        format:
            ``"json"`` or ``"csv"``.

        Returns
        -------
        str
            The exported content.
        """
        def _fetch_all() -> List[FeedbackEntry]:
            conn = self._ensure_conn()
            rows = conn.execute("SELECT * FROM feedback ORDER BY timestamp DESC").fetchall()
            return [_row_to_entry(r) for r in rows]

        entries = await self._run_sync(_fetch_all)

        if format == "json":
            content = json.dumps([asdict(e) for e in entries], indent=2, default=str, ensure_ascii=False)
        elif format == "csv":
            import csv
            import io
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["id", "session_id", "prompt", "response", "rating",
                             "comment", "model", "tokens", "duration", "timestamp",
                             "task_type"])
            for e in entries:
                writer.writerow([
                    e.id, e.session_id, e.prompt[:200], e.response[:200],
                    e.rating, e.comment, e.model, e.tokens, e.duration,
                    e.timestamp, e.task_type,
                ])
            content = buf.getvalue()
        else:
            raise ValueError(f"Unknown format: {format!r}. Use 'json' or 'csv'.")

        if filepath is not None:
            path = Path(filepath).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)

            def _write() -> None:
                path.write_text(content, encoding="utf-8")

            await self._run_sync(_write)

        return content

    # ── A/B comparison ────────────────────────────────────────────────────

    async def compare_models(self, model_a: str, model_b: str) -> Dict[str, Any]:
        """
        Compare feedback between two models (A/B test).

        Returns a dict with per-model stats and a ``winner`` field indicating
        which model scored higher.
        """
        stats_a = await self.get_stats(model=model_a)
        stats_b = await self.get_stats(model=model_b)

        a_score = stats_a.get("average_score", 0.0)
        b_score = stats_b.get("average_score", 0.0)
        a_rate = stats_a.get("approval_rate", 0.0)
        b_rate = stats_b.get("approval_rate", 0.0)

        # Determine winner by combined score.
        a_combined = a_score * 0.6 + a_rate * 0.4
        b_combined = b_score * 0.6 + b_rate * 0.4

        if a_combined > b_combined:
            winner = model_a
        elif b_combined > a_combined:
            winner = model_b
        else:
            winner = "tie"

        return {
            "model_a": {
                "name": model_a,
                "total_feedback": stats_a.get("total_feedback", 0),
                "average_score": a_score,
                "approval_rate": a_rate,
            },
            "model_b": {
                "name": model_b,
                "total_feedback": stats_b.get("total_feedback", 0),
                "average_score": b_score,
                "approval_rate": b_rate,
            },
            "winner": winner,
            "score_difference": round(a_score - b_score, 3),
            "approval_difference": round(a_rate - b_rate, 3),
        }
