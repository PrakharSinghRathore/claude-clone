"""
Self-Learner — The AI that adapts from feedback.

Analyzes user behavior patterns to improve agent responses:
- Tracks which suggestions are accepted vs rejected
- Identifies user preferences (code style, verbosity, tool preferences)
- Learns from feedback ratings to improve weak areas
- Adapts system prompt based on observed patterns
- Builds user preference profiles over time
- Adjusts tool selection and ordering based on success rates
- Generates personalized recommendations

This module makes the agent MORE like the user over time,
not just a generic assistant.

Usage:
    learner = SelfLearner(agent, project_root="/path/to/claude_clone")
    await learner.initialize()
    await learner.record_interaction(prompt, response, accepted=True)
    profile = await learner.get_user_profile()
    adaptation = await learner.generate_adaptation()
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "~/.claude_clone/self_improve_learner.db"

# Interaction signals
SIGNAL_ACCEPT = "accept"
SIGNAL_REJECT = "reject"
SIGNAL_EDIT = "edit"
SIGNAL_RERUN = "rerun"
SIGNAL_IGNORE = "ignore"


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InteractionRecord:
    """A single user-agent interaction with feedback signal."""
    id: str
    session_id: str
    prompt: str
    response: str
    signal: str  # accept, reject, edit, rerun, ignore
    tool_calls: List[str]
    model: str
    tokens_used: int
    duration: float
    task_type: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreference:
    """A learned user preference."""
    category: str
    key: str
    value: Any
    confidence: float  # 0.0 to 1.0
    sample_size: int
    last_observed: str
    description: str


@dataclass
class ToolPreference:
    """Learned preference for a specific tool."""
    tool_name: str
    success_rate: float
    avg_satisfaction: float
    usage_count: int
    preferred_position: int  # 0 = most preferred
    contexts: List[str]  # contexts where this tool works best


@dataclass
class Adaptation:
    """A suggested adaptation to improve agent behavior."""
    category: str
    description: str
    priority: float  # 0.0 to 1.0
    evidence: str
    confidence: float
    actionable: bool
    code_change: str = ""  # Optional code snippet to apply


@dataclass
class UserProfile:
    """Comprehensive user preference profile."""
    session_id: str
    total_interactions: int
    acceptance_rate: float
    avg_tokens_per_interaction: float
    avg_duration: float
    preferences: List[UserPreference]
    tool_preferences: List[ToolPreference]
    task_type_distribution: Dict[str, float]
    strengths: List[str]  # Task types where acceptance is high
    weaknesses: List[str]  # Task types where acceptance is low
    style_indicators: Dict[str, Any]
    last_updated: str


# ──────────────────────────────────────────────────────────────────────────────
# SelfLearner
# ──────────────────────────────────────────────────────────────────────────────

class SelfLearner:
    """
    Learns from user interactions to improve agent behavior over time.

    The learner observes:
    - Which responses are accepted vs rejected (thumbs up/down)
    - Which tools are used successfully vs which fail
    - Code style preferences (naming, structure, comments)
    - Verbosity preferences (short vs detailed responses)
    - Task type frequency and success rates
    - Time-of-day patterns (when the user is most productive)

    From these observations, it generates:
    - User preference profile
    - Personalized system prompt adaptations
    - Tool selection recommendations
    - Behavior modification suggestions
    """

    def __init__(self, project_root: str, db_path: str = DEFAULT_DB_PATH):
        self.project_root = Path(project_root).resolve()
        self.db_path = Path(db_path).expanduser().resolve()
        self._conn = None
        self._interaction_buffer: List[InteractionRecord] = []
        self._preference_cache: Dict[str, UserPreference] = {}
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize database and load preference cache."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(self._connect)
        await self._run_sync(self._create_tables)
        await self._load_preferences()
        self._initialized = True

    async def close(self) -> None:
        # Flush buffer before closing
        if self._interaction_buffer:
            await self._flush_buffer()
        if self._conn:
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
            CREATE TABLE IF NOT EXISTS interactions (
                id           TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL DEFAULT 'default',
                prompt       TEXT NOT NULL,
                response     TEXT NOT NULL,
                signal       TEXT NOT NULL,
                tool_calls   TEXT DEFAULT '[]',
                model        TEXT DEFAULT '',
                tokens_used  INTEGER DEFAULT 0,
                duration     REAL DEFAULT 0.0,
                task_type    TEXT DEFAULT 'general',
                timestamp    TEXT NOT NULL,
                metadata     TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS preferences (
                category     TEXT NOT NULL,
                key          TEXT NOT NULL,
                value        TEXT NOT NULL,
                confidence   REAL DEFAULT 0.0,
                sample_size  INTEGER DEFAULT 0,
                last_observed TEXT NOT NULL,
                description  TEXT DEFAULT '',
                PRIMARY KEY (category, key)
            );

            CREATE TABLE IF NOT EXISTS tool_stats (
                tool_name    TEXT PRIMARY KEY,
                success_count INTEGER DEFAULT 0,
                fail_count   INTEGER DEFAULT 0,
                total_satisfaction REAL DEFAULT 0.0,
                usage_count  INTEGER DEFAULT 0,
                contexts     TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS adaptations (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                category     TEXT NOT NULL,
                description  TEXT NOT NULL,
                priority     REAL DEFAULT 0.0,
                evidence     TEXT DEFAULT '',
                confidence   REAL DEFAULT 0.0,
                applied      INTEGER DEFAULT 0,
                timestamp    TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id);
            CREATE INDEX IF NOT EXISTS idx_interactions_signal ON interactions(signal);
            CREATE INDEX IF NOT EXISTS idx_interactions_task ON interactions(task_type);
            CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(timestamp);
        """)
        self._conn.commit()

    async def _run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def _ensure_conn(self):
        if self._conn is None:
            raise RuntimeError("SelfLearner not initialized.")

    async def _load_preferences(self) -> None:
        """Load learned preferences from database."""
        def _do() -> None:
            conn = self._ensure_conn()
            rows = conn.execute("SELECT * FROM preferences").fetchall()
            for row in rows:
                pref = UserPreference(
                    category=row["category"],
                    key=row["key"],
                    value=json.loads(row["value"]) if row["value"].startswith("[") or row["value"].startswith("{") else row["value"],
                    confidence=row["confidence"],
                    sample_size=row["sample_size"],
                    last_observed=row["last_observed"],
                    description=row["description"],
                )
                cache_key = f"{row['category']}:{row['key']}"
                self._preference_cache[cache_key] = pref
        await self._run_sync(_do)

    # ── Recording Interactions ────────────────────────────────────────────

    async def record_interaction(
        self,
        prompt: str,
        response: str,
        signal: str = SIGNAL_ACCEPT,
        tool_calls: Optional[List[str]] = None,
        model: str = "",
        tokens_used: int = 0,
        duration: float = 0.0,
        task_type: str = "general",
        session_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a user-agent interaction with feedback.

        Parameters
        ----------
        prompt: The user's prompt.
        response: The agent's response.
        signal: One of 'accept', 'reject', 'edit', 'rerun', 'ignore'.
        tool_calls: List of tools called during this interaction.
        model: The model used.
        tokens_used: Number of tokens consumed.
        duration: Response duration in seconds.
        task_type: Category of the task.
        session_id: Session identifier.
        metadata: Additional context.

        Returns
        -------
        The interaction record ID.
        """
        import uuid
        record_id = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()

        record = InteractionRecord(
            id=record_id,
            session_id=session_id,
            prompt=prompt,
            response=response,
            signal=signal,
            tool_calls=tool_calls or [],
            model=model,
            tokens_used=tokens_used,
            duration=duration,
            task_type=task_type,
            timestamp=now,
            metadata=metadata or {},
        )

        self._interaction_buffer.append(record)

        # Flush buffer if it gets too large
        if len(self._interaction_buffer) >= 50:
            await self._flush_buffer()

        # Update preferences in real-time
        await self._update_preferences_from_interaction(record)

        return record_id

    async def _flush_buffer(self) -> None:
        """Write buffered interactions to the database."""
        if not self._interaction_buffer:
            return

        records = self._interaction_buffer[:]
        self._interaction_buffer.clear()

        def _do() -> None:
            conn = self._ensure_conn()
            for r in records:
                conn.execute(
                    "INSERT INTO interactions "
                    "(id, session_id, prompt, response, signal, tool_calls, model, "
                    "tokens_used, duration, task_type, timestamp, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        r.id, r.session_id, r.prompt, r.response, r.signal,
                        json.dumps(r.tool_calls), r.model, r.tokens_used,
                        r.duration, r.task_type, r.timestamp,
                        json.dumps(r.metadata, default=str),
                    ),
                )
            conn.commit()

        await self._run_sync(_do)

    # ── Preference Learning ───────────────────────────────────────────────

    async def _update_preferences_from_interaction(self, record: InteractionRecord) -> None:
        """Update preferences based on a single interaction."""
        is_positive = record.signal in (SIGNAL_ACCEPT, SIGNAL_RERUN)
        is_negative = record.signal in (SIGNAL_REJECT,)

        # Learn response length preference
        response_length = len(record.response)
        await self._update_preference(
            category="style",
            key="response_length",
            value=response_length,
            is_positive=is_positive,
            description="Preferred response length",
        )

        # Learn tool preference
        for tool_name in record.tool_calls:
            await self._update_tool_stat(
                tool_name=tool_name,
                success=is_positive,
                context=record.task_type,
            )

        # Learn task type preference
        await self._update_preference(
            category="task",
            key=record.task_type,
            value=1.0 if is_positive else 0.0,
            is_positive=is_positive,
            description=f"Preference for {record.task_type} tasks",
        )

        # Learn verbosity preference from response
        if record.response:
            # Count sentences as a proxy for verbosity
            sentences = len(re.split(r'[.!?]+', record.response))
            await self._update_preference(
                category="style",
                key="verbosity",
                value=sentences,
                is_positive=is_positive,
                description="Preferred verbosity level (sentence count)",
            )

        # Learn code preference (if response contains code)
        if "```" in record.response:
            code_blocks = record.response.count("```") // 2
            await self._update_preference(
                category="style",
                key="code_block_count",
                value=code_blocks,
                is_positive=is_positive,
                description="Preferred number of code blocks per response",
            )

    async def _update_preference(
        self, category: str, key: str, value: Any, is_positive: bool, description: str
    ) -> None:
        """Update a preference using exponential moving average."""
        cache_key = f"{category}:{key}"
        existing = self._preference_cache.get(cache_key)

        alpha = 0.15  # Learning rate
        if isinstance(value, (int, float)):
            if existing:
                old_val = float(existing.value) if not isinstance(existing.value, (int, float)) else existing.value
                new_val = old_val * (1 - alpha) + float(value) * alpha
                new_confidence = existing.confidence * (1 - alpha * 0.1) + (0.1 if is_positive else 0)
            else:
                new_val = float(value)
                new_confidence = 0.3 if is_positive else 0.1
            new_sample = (existing.sample_size + 1) if existing else 1
        else:
            new_val = value
            new_confidence = min(1.0, (existing.confidence + 0.1) if is_positive else max(0.0, existing.confidence - 0.05)) if existing else (0.3 if is_positive else 0.1)
            new_sample = (existing.sample_size + 1) if existing else 1

        pref = UserPreference(
            category=category,
            key=key,
            value=new_val,
            confidence=min(1.0, new_confidence),
            sample_size=new_sample,
            last_observed=datetime.now(timezone.utc).isoformat(),
            description=description,
        )
        self._preference_cache[cache_key] = pref

        # Persist to database
        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT OR REPLACE INTO preferences (category, key, value, confidence, sample_size, last_observed, description) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    category, key,
                    json.dumps(new_val) if isinstance(new_val, (list, dict)) else str(new_val),
                    pref.confidence, pref.sample_size,
                    pref.last_observed, description,
                ),
            )
            conn.commit()

        await self._run_sync(_do)

    async def _update_tool_stat(self, tool_name: str, success: bool, context: str) -> None:
        """Update tool success statistics."""
        def _do() -> None:
            conn = self._ensure_conn()
            row = conn.execute("SELECT * FROM tool_stats WHERE tool_name = ?", (tool_name,)).fetchone()

            if row:
                success_count = row["success_count"] + (1 if success else 0)
                fail_count = row["fail_count"] + (0 if success else 1)
                usage_count = row["usage_count"] + 1
                # Update satisfaction using EMA
                current_satisfaction = row["total_satisfaction"]
                new_satisfaction = current_satisfaction * 0.9 + (1.0 if success else 0.0) * 0.1
                contexts = json.loads(row.get("contexts", "[]"))
                if context and context not in contexts:
                    contexts.append(context)
                    contexts = contexts[-20:]  # Keep last 20 contexts

                conn.execute(
                    "UPDATE tool_stats SET success_count=?, fail_count=?, total_satisfaction=?, usage_count=?, contexts=? WHERE tool_name=?",
                    (success_count, fail_count, new_satisfaction, usage_count, json.dumps(contexts), tool_name),
                )
            else:
                conn.execute(
                    "INSERT INTO tool_stats (tool_name, success_count, fail_count, total_satisfaction, usage_count, contexts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (tool_name, 1 if success else 0, 0 if success else 1, 1.0 if success else 0.0, 1, json.dumps([context] if context else [])),
                )
            conn.commit()

        await self._run_sync(_do)

    # ── Profile & Analysis ────────────────────────────────────────────────

    async def get_user_profile(self, session_id: str = "default") -> UserProfile:
        """Generate a comprehensive user preference profile."""
        def _do() -> Tuple[List[Dict], Dict[str, Dict]]:
            conn = self._ensure_conn()
            interactions = conn.execute(
                "SELECT * FROM interactions WHERE session_id = ? ORDER BY timestamp DESC LIMIT 500",
                (session_id,),
            ).fetchall()

            tool_stats = {}
            for row in conn.execute("SELECT * FROM tool_stats").fetchall():
                tool_stats[row["tool_name"]] = dict(row)

            return [dict(r) for r in interactions], tool_stats

        interactions, tool_stats = await self._run_sync(_do)

        if not interactions:
            return UserProfile(
                session_id=session_id,
                total_interactions=0,
                acceptance_rate=0.0,
                avg_tokens_per_interaction=0.0,
                avg_duration=0.0,
                preferences=list(self._preference_cache.values()),
                tool_preferences=[],
                task_type_distribution={},
                strengths=[],
                weaknesses=[],
                style_indicators={},
                last_updated=datetime.now(timezone.utc).isoformat(),
            )

        # Calculate acceptance rate
        signals = [i["signal"] for i in interactions]
        accepted = sum(1 for s in signals if s in (SIGNAL_ACCEPT, SIGNAL_RERUN))
        acceptance_rate = accepted / len(interactions)

        # Average tokens and duration
        avg_tokens = sum(i.get("tokens_used", 0) for i in interactions) / len(interactions)
        avg_duration = sum(i.get("duration", 0) for i in interactions) / len(interactions)

        # Task type distribution
        task_types = Counter(i.get("task_type", "general") for i in interactions)
        total_tasks = sum(task_types.values())
        task_dist = {t: c / total_tasks for t, c in task_types.most_common()}

        # Task success rates (strengths and weaknesses)
        task_success: Dict[str, List[bool]] = defaultdict(list)
        for i in interactions:
            task = i.get("task_type", "general")
            is_positive = i["signal"] in (SIGNAL_ACCEPT, SIGNAL_RERUN)
            task_success[task].append(is_positive)

        strengths = []
        weaknesses = []
        for task, results in task_success.items():
            if len(results) < 3:
                continue
            rate = sum(results) / len(results)
            if rate >= 0.7:
                strengths.append(f"{task} ({rate:.0%} acceptance)")
            elif rate <= 0.3:
                weaknesses.append(f"{task} ({rate:.0%} acceptance)")

        # Tool preferences
        tool_prefs = []
        for tool_name, stats in sorted(tool_stats.items(), key=lambda x: -(x[1].get("success_count", 0) / max(1, x[1].get("usage_count", 1)))):
            usage = stats.get("usage_count", 0)
            if usage == 0:
                continue
            success_rate = stats.get("success_count", 0) / usage
            satisfaction = stats.get("total_satisfaction", 0.0)
            contexts = json.loads(stats.get("contexts", "[]"))

            tool_prefs.append(ToolPreference(
                tool_name=tool_name,
                success_rate=round(success_rate, 2),
                avg_satisfaction=round(satisfaction, 2),
                usage_count=usage,
                preferred_position=0,  # Will be set after sorting
                contexts=contexts,
            ))

        for i, tp in enumerate(tool_prefs):
            tp.preferred_position = i

        # Style indicators
        style = {}
        for key, pref in self._preference_cache.items():
            if pref.category == "style" and pref.confidence > 0.3:
                style[pref.key] = {"value": pref.value, "confidence": round(pref.confidence, 2)}

        return UserProfile(
            session_id=session_id,
            total_interactions=len(interactions),
            acceptance_rate=round(acceptance_rate, 3),
            avg_tokens_per_interaction=round(avg_tokens, 1),
            avg_duration=round(avg_duration, 2),
            preferences=list(self._preference_cache.values()),
            tool_preferences=tool_prefs,
            task_type_distribution=dict(task_dist),
            strengths=strengths,
            weaknesses=weaknesses,
            style_indicators=style,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    # ── Adaptation Generation ────────────────────────────────────────────

    async def generate_adaptations(self) -> List[Adaptation]:
        """Generate suggested adaptations based on learned preferences."""
        profile = await self.get_user_profile()
        adaptations = []

        # Adaptation: Response length
        length_pref = self._preference_cache.get("style:response_length")
        if length_pref and length_pref.sample_size >= 10:
            avg_length = length_pref.value
            if avg_length < 200:
                adaptations.append(Adaptation(
                    category="verbosity",
                    description="User prefers concise, short responses. Reduce verbosity.",
                    priority=0.8,
                    evidence=f"Average accepted response length: {avg_length:.0f} chars (from {length_pref.sample_size} samples)",
                    confidence=length_pref.confidence,
                    actionable=True,
                ))
            elif avg_length > 2000:
                adaptations.append(Adaptation(
                    category="verbosity",
                    description="User prefers detailed, comprehensive responses. Be thorough.",
                    priority=0.8,
                    evidence=f"Average accepted response length: {avg_length:.0f} chars (from {length_pref.sample_size} samples)",
                    confidence=length_pref.confidence,
                    actionable=True,
                ))

        # Adaptation: Code preference
        code_pref = self._preference_cache.get("style:code_block_count")
        if code_pref and code_pref.sample_size >= 10:
            avg_blocks = code_pref.value
            if avg_blocks < 1:
                adaptations.append(Adaptation(
                    category="code_style",
                    description="User prefers explanations over code. Minimize code blocks.",
                    priority=0.6,
                    evidence=f"Average code blocks in accepted responses: {avg_blocks:.1f}",
                    confidence=code_pref.confidence,
                    actionable=True,
                ))
            elif avg_blocks > 3:
                adaptations.append(Adaptation(
                    category="code_style",
                    description="User prefers code-heavy responses. Include more code examples.",
                    priority=0.6,
                    evidence=f"Average code blocks in accepted responses: {avg_blocks:.1f}",
                    confidence=code_pref.confidence,
                    actionable=True,
                ))

        # Adaptation: Weak areas
        for weakness in profile.weaknesses:
            adaptations.append(Adaptation(
                category="weak_area",
                description=f"Improve quality for task type: {weakness}",
                priority=0.7,
                evidence=f"Low acceptance rate detected for: {weakness}",
                confidence=0.5,
                actionable=True,
            ))

        # Adaptation: Tool optimization
        if profile.tool_preferences:
            worst_tool = min(profile.tool_preferences, key=lambda t: t.success_rate)
            if worst_tool.success_rate < 0.5 and worst_tool.usage_count >= 5:
                adaptations.append(Adaptation(
                    category="tool_optimization",
                    description=f"Tool '{worst_tool.tool_name}' has low success rate ({worst_tool.success_rate:.0%}). Consider improving it.",
                    priority=0.6,
                    evidence=f"Success rate: {worst_tool.success_rate:.0%} from {worst_tool.usage_count} uses",
                    confidence=0.7,
                    actionable=True,
                ))

        adaptations.sort(key=lambda a: -a.priority)
        return adaptations

    async def generate_system_prompt_additions(self) -> str:
        """Generate additions to the system prompt based on learned preferences."""
        profile = await self.get_user_profile()
        adaptations = await self.generate_adaptations()

        additions = []

        if profile.acceptance_rate < 0.5 and profile.total_interactions >= 20:
            additions.append("## USER FEEDBACK WARNING\nThe user has been rejecting many of your responses. Be more careful and ask for clarification before proceeding.")

        # Style adaptations
        style_adaptations = [a for a in adaptations if a.category in ("verbosity", "code_style")]
        if style_adaptations:
            additions.append("## LEARNED STYLE PREFERENCES")
            for a in style_adaptations:
                if a.confidence > 0.4:
                    additions.append(f"- {a.description} (confidence: {a.confidence:.0%})")

        # Weak area adaptations
        weak_adaptations = [a for a in adaptations if a.category == "weak_area"]
        if weak_adaptations:
            additions.append("## AREAS NEEDING IMPROVEMENT")
            for a in weak_adaptations[:3]:
                additions.append(f"- {a.description}")

        return "\n".join(additions) if additions else ""

    # ── Statistics ────────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        def _do() -> Dict[str, Any]:
            conn = self._ensure_conn()
            total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            accepted = conn.execute("SELECT COUNT(*) FROM interactions WHERE signal IN ('accept', 'rerun')").fetchone()[0]
            rejected = conn.execute("SELECT COUNT(*) FROM interactions WHERE signal = 'reject'").fetchone()[0]
            edited = conn.execute("SELECT COUNT(*) FROM interactions WHERE signal = 'edit'").fetchone()[0]
            total_prefs = conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
            high_confidence = conn.execute("SELECT COUNT(*) FROM preferences WHERE confidence > 0.5").fetchone()[0]
            total_tools = conn.execute("SELECT COUNT(*) FROM tool_stats").fetchone()[0]
            avg_satisfaction = conn.execute("SELECT AVG(total_satisfaction) FROM tool_stats WHERE usage_count > 2").fetchone()[0] or 0.0

            return {
                "total_interactions": total,
                "accepted": accepted,
                "rejected": rejected,
                "edited": edited,
                "acceptance_rate": round(accepted / max(1, total), 3),
                "total_preferences": total_prefs,
                "high_confidence_preferences": high_confidence,
                "tools_tracked": total_tools,
                "avg_tool_satisfaction": round(avg_satisfaction, 2),
            }

        return await self._run_sync(_do)

    async def generate_report(self) -> str:
        """Generate a human-readable learning report."""
        profile = await self.get_user_profile()
        stats = await self.get_stats()
        adaptations = await self.generate_adaptations()

        lines = [
            "# Self-Learning Report",
            "",
            f"**Total Interactions:** {stats['total_interactions']}",
            f"**Acceptance Rate:** {stats['acceptance_rate']:.1%}",
            f"**Accepted:** {stats['accepted']} | **Rejected:** {stats['rejected']} | **Edited:** {stats['edited']}",
            f"**Preferences Learned:** {stats['total_preferences']} ({stats['high_confidence_preferences']} high confidence)",
            f"**Tools Tracked:** {stats['tools_tracked']}",
            f"**Avg Tool Satisfaction:** {stats['avg_tool_satisfaction']:.2f}",
            "",
        ]

        if profile.strengths:
            lines.append("## Strengths (high acceptance)")
            for s in profile.strengths:
                lines.append(f"- {s}")
            lines.append("")

        if profile.weaknesses:
            lines.append("## Weaknesses (low acceptance)")
            for w in profile.weaknesses:
                lines.append(f"- {w}")
            lines.append("")

        if profile.style_indicators:
            lines.append("## Style Preferences")
            for key, info in profile.style_indicators.items():
                lines.append(f"- **{key}**: {info['value']} (confidence: {info['confidence']:.0%})")
            lines.append("")

        if profile.tool_preferences:
            lines.append("## Tool Preferences (sorted by success rate)")
            for tp in profile.tool_preferences[:10]:
                indicator = "🟢" if tp.success_rate >= 0.7 else ("🟡" if tp.success_rate >= 0.4 else "🔴")
                lines.append(
                    f"- {indicator} `{tp.tool_name}`: success={tp.success_rate:.0%}, "
                    f"used={tp.usage_count}x, satisfaction={tp.avg_satisfaction:.2f}"
                )
            lines.append("")

        if adaptations:
            lines.append("## Suggested Adaptations")
            for a in adaptations[:10]:
                lines.append(f"- [{a.category}] {a.description} (priority: {a.priority:.1f}, confidence: {a.confidence:.0%})")

        return "\n".join(lines)
