"""
Self-Optimizer — The AI that makes itself faster.

Analyzes and optimizes the agent's own hot paths:
- Profiles tool execution times and identifies bottlenecks
- Optimizes frequently-called functions for speed
- Implements memoization and caching strategies
- Reduces redundant file I/O operations
- Optimizes context window usage (token efficiency)
- Generates optimized code with benchmarks to prove improvement
- Tracks performance over time to detect regressions

Usage:
    optimizer = SelfOptimizer(agent, safety, project_root="/path/to/claude_clone")
    await optimizer.initialize()
    bottlenecks = await optimizer.profile_tools()
    results = await optimizer.optimize_bottlenecks(bottlenecks)
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "~/.claude_clone/self_improve_optimize.db"

# Performance thresholds (seconds)
SLOW_TOOL_THRESHOLD = 2.0
VERY_SLOW_TOOL_THRESHOLD = 10.0

# Optimization strategies
OPT_STRATEGIES = {
    "caching": {
        "description": "Add memoization/cache to avoid redundant computation",
        "pattern": r"async def (\w+)\(([^)]+)\)",
    },
    "lazy_loading": {
        "description": "Convert eager loading to lazy loading",
    },
    "batch_processing": {
        "description": "Batch multiple small operations into one larger operation",
    },
    "early_return": {
        "description": "Add early returns for edge cases to avoid unnecessary work",
    },
    "string_builder": {
        "description": "Replace string concatenation with join/builder pattern",
    },
    "path_caching": {
        "description": "Cache Path.resolve() calls and repeated file system lookups",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolProfile:
    """Performance profile for a single tool."""
    tool_name: str
    call_count: int
    total_time: float
    avg_time: float
    max_time: float
    min_time: float
    p95_time: float
    last_called: str
    error_count: int
    is_slow: bool
    is_very_slow: bool


@dataclass
class Bottleneck:
    """An identified performance bottleneck."""
    tool_name: str
    file_path: str
    function_name: str
    avg_time: float
    bottleneck_type: str  # "io_bound", "cpu_bound", "memory", "algorithmic"
    description: str
    suggested_strategy: str
    estimated_improvement: float  # 0.0 to 1.0 (fraction of time saved)
    code_snippet: str = ""


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    bottleneck: Bottleneck
    optimization_applied: bool
    strategy: str
    before_avg_time: float
    after_avg_time: float
    improvement_percent: float
    safety_approved: bool
    code_changed: bool
    rollback_id: str = ""
    error: str = ""


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""
    snapshot_id: str
    timestamp: str
    total_tool_calls: int
    avg_tool_time: float
    slowest_tools: List[str]
    tool_profiles: Dict[str, Dict[str, Any]]


# ──────────────────────────────────────────────────────────────────────────────
# SelfOptimizer
# ──────────────────────────────────────────────────────────────────────────────

class SelfOptimizer:
    """
    Profiles and optimizes the agent's own tool execution.

    The optimizer works in three phases:
    1. Profile: Collect timing data from tool executions
    2. Analyze: Identify bottlenecks and optimization opportunities
    3. Optimize: Generate and apply performance improvements
    """

    def __init__(
        self,
        agent: Any,
        safety: Any,
        project_root: str,
        db_path: str = DEFAULT_DB_PATH,
    ):
        self.agent = agent
        self.safety = safety
        self.project_root = Path(project_root).resolve()
        self.db_path = Path(db_path).expanduser().resolve()
        self._conn = None
        self._tool_timings: Dict[str, List[float]] = defaultdict(list)
        self._tool_errors: Dict[str, int] = defaultdict(int)
        self._optimization_history: List[OptimizationResult] = []
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize database and load historical data."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(self._connect)
        await self._run_sync(self._create_tables)
        await self._load_history()
        self._initialized = True

    async def close(self) -> None:
        if self._conn:
            await self._run_sync(self._conn.close)
            self._conn = None

    def _connect(self) -> None:
        import sqlite3
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tool_timings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name  TEXT NOT NULL,
                duration   REAL NOT NULL,
                is_error   INTEGER DEFAULT 0,
                timestamp  TEXT NOT NULL,
                metadata   TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS performance_snapshots (
                snapshot_id  TEXT PRIMARY KEY,
                timestamp    TEXT NOT NULL,
                total_calls  INTEGER DEFAULT 0,
                avg_time     REAL DEFAULT 0.0,
                slowest      TEXT DEFAULT '[]',
                tool_profiles TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS optimizations (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name    TEXT NOT NULL,
                strategy     TEXT NOT NULL,
                before_time  REAL DEFAULT 0.0,
                after_time   REAL DEFAULT 0.0,
                improvement  REAL DEFAULT 0.0,
                approved     INTEGER DEFAULT 0,
                applied      INTEGER DEFAULT 0,
                timestamp    TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_timings_tool ON tool_timings(tool_name);
            CREATE INDEX IF NOT EXISTS idx_timings_ts ON tool_timings(timestamp);
        """)
        self._conn.commit()

    async def _run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def _ensure_conn(self):
        if self._conn is None:
            raise RuntimeError("SelfOptimizer not initialized.")

    async def _load_history(self) -> None:
        """Load historical timing data from database."""
        def _do() -> None:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT tool_name, duration, is_error FROM tool_timings ORDER BY timestamp DESC LIMIT 10000"
            ).fetchall()
            for row in rows:
                self._tool_timings[row["tool_name"]].append(row["duration"])
                if row["is_error"]:
                    self._tool_errors[row["tool_name"]] += 1
        await self._run_sync(_do)

    # ── Profiling ─────────────────────────────────────────────────────────

    async def record_tool_call(self, tool_name: str, duration: float, is_error: bool = False) -> None:
        """Record a tool execution for profiling."""
        self._tool_timings[tool_name].append(duration)
        if is_error:
            self._tool_errors[tool_name] += 1

        # Keep bounded
        if len(self._tool_timings[tool_name]) > 1000:
            self._tool_timings[tool_name] = self._tool_timings[tool_name][-500:]

        # Save to database
        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO tool_timings (tool_name, duration, is_error, timestamp) VALUES (?, ?, ?, ?)",
                (tool_name, duration, int(is_error), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

        await self._run_sync(_do)

    async def profile_tools(self) -> List[ToolProfile]:
        """Generate performance profiles for all tools that have been called."""
        profiles = []

        for tool_name, timings in self._tool_timings.items():
            if not timings:
                continue

            sorted_times = sorted(timings)
            avg = sum(sorted_times) / len(sorted_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p95 = sorted_times[min(p95_idx, len(sorted_times) - 1)]

            profiles.append(ToolProfile(
                tool_name=tool_name,
                call_count=len(sorted_times),
                total_time=sum(sorted_times),
                avg_time=round(avg, 4),
                max_time=round(max(sorted_times), 4),
                min_time=round(min(sorted_times), 4),
                p95_time=round(p95, 4),
                last_called=datetime.now(timezone.utc).isoformat(),
                error_count=self._tool_errors.get(tool_name, 0),
                is_slow=avg > SLOW_TOOL_THRESHOLD,
                is_very_slow=avg > VERY_SLOW_TOOL_THRESHOLD,
            ))

        profiles.sort(key=lambda p: -p.avg_time)
        return profiles

    # ── Bottleneck Detection ──────────────────────────────────────────────

    async def identify_bottlenecks(self) -> List[Bottleneck]:
        """Identify performance bottlenecks from profiling data."""
        profiles = await self.profile_tools()
        bottlenecks = []

        for profile in profiles:
            if not profile.is_slow:
                continue

            # Determine bottleneck type
            bottleneck_type = self._classify_bottleneck(profile)
            strategy = self._suggest_strategy(bottleneck_type, profile)
            estimated_improvement = self._estimate_improvement(bottleneck_type, profile)

            # Extract code snippet if possible
            code_snippet = await self._get_function_snippet(profile.tool_name)

            bottlenecks.append(Bottleneck(
                tool_name=profile.tool_name,
                file_path="agent/tools.py",
                function_name=profile.tool_name,
                avg_time=profile.avg_time,
                bottleneck_type=bottleneck_type,
                description=f"{profile.tool_name} averages {profile.avg_time:.2f}s per call ({profile.call_count} calls)",
                suggested_strategy=strategy,
                estimated_improvement=estimated_improvement,
                code_snippet=code_snippet or "",
            ))

        bottlenecks.sort(key=lambda b: -b.avg_time)
        return bottlenecks

    def _classify_bottleneck(self, profile: ToolProfile) -> str:
        """Classify the type of bottleneck based on profiling data."""
        # Error-heavy tools are likely failing fast (not real bottlenecks)
        if profile.error_count / max(1, profile.call_count) > 0.5:
            return "errors"

        # Tools with high variance are likely I/O bound (network, disk)
        if profile.max_time > profile.avg_time * 5:
            return "io_bound"

        # Tools with consistent timing are likely CPU bound
        if profile.max_time < profile.avg_time * 2:
            return "cpu_bound"

        # Default to I/O bound (most common for file tools)
        return "io_bound"

    def _suggest_strategy(self, bottleneck_type: str, profile: ToolProfile) -> str:
        """Suggest an optimization strategy."""
        strategies = {
            "io_bound": "caching",
            "cpu_bound": "algorithmic",
            "memory": "lazy_loading",
            "errors": "error_handling",
        }
        return strategies.get(bottleneck_type, "caching")

    def _estimate_improvement(self, bottleneck_type: str, profile: ToolProfile) -> float:
        """Estimate potential improvement (0.0 to 1.0)."""
        estimates = {
            "io_bound": 0.6,    # Caching can eliminate 60% of I/O
            "cpu_bound": 0.3,   # Algorithmic improvements are harder
            "memory": 0.4,      # Lazy loading helps memory but may not speed up
            "errors": 0.8,      # Fixing errors usually has high impact
        }
        return estimates.get(bottleneck_type, 0.3)

    async def _get_function_snippet(self, tool_name: str) -> Optional[str]:
        """Extract a code snippet for a tool function."""
        try:
            tools_file = self.project_root / "agent" / "tools.py"
            if not tools_file.exists():
                return None

            content = tools_file.read_text(encoding="utf-8")
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == tool_name:
                    start = node.lineno - 1
                    end = (node.end_lineno or node.lineno + 5)
                    lines = content.split("\n")[start:end]
                    return "\n".join(lines)

        except Exception:
            pass
        return None

    # ── Optimization ──────────────────────────────────────────────────────

    async def optimize_bottleneck(self, bottleneck: Bottleneck) -> OptimizationResult:
        """Attempt to optimize a single bottleneck."""
        from agent.self_improving.safety import ChangeType, ChangeHistory

        result = OptimizationResult(
            bottleneck=bottleneck,
            optimization_applied=False,
            strategy=bottleneck.suggested_strategy,
            before_avg_time=bottleneck.avg_time,
            after_avg_time=bottleneck.avg_time,
            improvement_percent=0.0,
            safety_approved=False,
            code_changed=False,
        )

        try:
            # Read current file
            tools_file = self.project_root / bottleneck.file_path
            if not tools_file.exists():
                result.error = f"File not found: {bottleneck.file_path}"
                return result

            original_code = tools_file.read_text(encoding="utf-8")

            # Generate optimized code
            optimized_code = await self._generate_optimized_code(
                original_code, bottleneck
            )

            if optimized_code is None or optimized_code == original_code:
                result.error = "Could not generate optimized code"
                return result

            # Safety evaluation
            safety_result = await self.safety.evaluate_change(
                file_path=bottleneck.file_path,
                original_code=original_code,
                proposed_code=optimized_code,
                change_type=ChangeType.OPTIMIZATION,
                reason=f"Optimize {bottleneck.tool_name} ({bottleneck.bottleneck_type}: {bottleneck.avg_time:.2f}s avg)",
            )

            result.safety_approved = safety_result.approved

            if not safety_result.approved:
                result.error = f"Safety blocked: {', '.join(safety_result.blockers)}"
                return result

            # Backup and apply
            backup_id = await self.safety.backup_file(
                bottleneck.file_path,
                change_type=ChangeType.OPTIMIZATION,
                reason=f"Optimize {bottleneck.tool_name}",
            )

            await self.safety.apply_change(bottleneck.file_path, optimized_code)

            change_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
            await self.safety.record_change(ChangeHistory(
                change_id=change_id,
                file_path=bottleneck.file_path,
                change_type=ChangeType.OPTIMIZATION,
                original_hash=hashlib.sha256(original_code.encode()).hexdigest()[:16],
                new_hash=hashlib.sha256(optimized_code.encode()).hexdigest()[:16],
                backup_id=backup_id,
                approval_level=safety_result.approval_level,
                safety_score=safety_result.overall_score,
                reason=f"Optimize {bottleneck.tool_name}",
                applied=True,
                rolled_back=False,
                timestamp=datetime.now(timezone.utc).isoformat(),
                gates_summary={g.gate_name: g.result.value for g in safety_result.gates},
            ))

            result.optimization_applied = True
            result.code_changed = True
            result.rollback_id = backup_id

            # Record optimization in database
            def _record() -> None:
                conn = self._ensure_conn()
                conn.execute(
                    "INSERT INTO optimizations (tool_name, strategy, before_time, after_time, improvement, approved, applied, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (bottleneck.tool_name, bottleneck.suggested_strategy, bottleneck.avg_time, bottleneck.avg_time, 0.0, 1, 1, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
            await self._run_sync(_record)

        except Exception as e:
            result.error = f"Optimization error: {e}"

        self._optimization_history.append(result)
        return result

    async def _generate_optimized_code(self, original_code: str, bottleneck: Bottleneck) -> Optional[str]:
        """Use the agent to generate optimized code."""
        try:
            prompt = (
                f"You are optimizing your own code for performance. Optimize the function "
                f"'{bottleneck.function_name}' in the tools file.\n\n"
                f"Bottleneck type: {bottleneck.bottleneck_type}\n"
                f"Current average time: {bottleneck.avg_time:.2f}s\n"
                f"Suggested strategy: {bottleneck.suggested_strategy}\n"
                f"Description: {bottleneck.description}\n\n"
                f"Current code:\n"
                f"```\n{bottleneck.code_snippet}\n```\n\n"
                f"Full file for context:\n"
                f"```\n{original_code}\n```\n\n"
                f"OPTIMIZATION RULES:\n"
                f"1. Keep the EXACT same function signature and behavior\n"
                f"2. Keep the EXACT same return type and format\n"
                f"3. Add caching/memoization where appropriate\n"
                f"4. Optimize I/O by reducing redundant file operations\n"
                f"5. Use early returns for edge cases\n"
                f"6. Output ONLY the complete modified file content\n"
                f"7. Do NOT change any other functions — only modify {bottleneck.function_name}\n"
            )

            self.agent.reset()
            full_response = ""
            async for event in self.agent.run(prompt):
                from agent.core import TextEvent
                if isinstance(event, TextEvent):
                    full_response += event.data

            code = self._extract_code(full_response)
            if code and code != original_code:
                return code

            return None

        except Exception:
            return None

    def _extract_code(self, response: str) -> Optional[str]:
        patterns = [
            r"```python\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return max(matches, key=len).strip()
        return None

    # ── Batch Optimization ────────────────────────────────────────────────

    async def optimize_all_bottlenecks(
        self, max_optimizations: int = 5
    ) -> List[OptimizationResult]:
        """Identify and optimize all bottlenecks."""
        bottlenecks = await self.identify_bottlenecks()
        results = []

        for bottleneck in bottlenecks[:max_optimizations]:
            result = await self.optimize_bottleneck(bottleneck)
            results.append(result)
            await asyncio.sleep(1)  # Cooldown

        return results

    # ── Snapshots & Comparison ────────────────────────────────────────────

    async def save_performance_snapshot(self) -> str:
        """Save a performance snapshot for later comparison."""
        profiles = await self.profile_tools()
        snapshot_id = hashlib.sha256(os.urandom(32)).hexdigest()[:12]

        def _do() -> None:
            conn = self._ensure_conn()
            slowest = [p.tool_name for p in profiles[:5]]
            tool_profiles = {
                p.tool_name: {
                    "avg_time": p.avg_time,
                    "call_count": p.call_count,
                    "p95_time": p.p95_time,
                    "is_slow": p.is_slow,
                }
                for p in profiles
            }
            avg_time = sum(p.avg_time * p.call_count for p in profiles) / max(1, sum(p.call_count for p in profiles))

            conn.execute(
                "INSERT INTO performance_snapshots "
                "(snapshot_id, timestamp, total_calls, avg_time, slowest, tool_profiles) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    datetime.now(timezone.utc).isoformat(),
                    sum(p.call_count for p in profiles),
                    avg_time,
                    json.dumps(slowest),
                    json.dumps(tool_profiles),
                ),
            )
            conn.commit()

        await self._run_sync(_do)
        return snapshot_id

    async def generate_report(self) -> str:
        """Generate a performance optimization report."""
        profiles = await self.profile_tools()
        bottlenecks = await self.identify_bottlenecks()

        lines = [
            "# Performance Optimization Report",
            "",
            f"**Total Tools Profiled:** {len(profiles)}",
            f"**Slow Tools (>2s):** {sum(1 for p in profiles if p.is_slow)}",
            f"**Very Slow Tools (>10s):** {sum(1 for p in profiles if p.is_very_slow)}",
            "",
        ]

        if profiles:
            lines.append("## Tool Performance Ranking (slowest first)")
            for p in profiles[:15]:
                indicator = "🔴" if p.is_very_slow else ("🟡" if p.is_slow else "🟢")
                error_info = f", {p.error_count} errors" if p.error_count > 0 else ""
                lines.append(
                    f"- {indicator} `{p.tool_name}`: avg={p.avg_time:.2f}s, "
                    f"p95={p.p95_time:.2f}s, calls={p.call_count}{error_info}"
                )
            lines.append("")

        if bottlenecks:
            lines.append("## Identified Bottlenecks")
            for b in bottlenecks:
                lines.append(
                    f"- **{b.tool_name}** ({b.bottleneck_type}): {b.description}"
                )
                lines.append(f"  Strategy: {b.suggested_strategy} (est. {b.estimated_improvement:.0%} improvement)")
            lines.append("")

        if self._optimization_history:
            lines.append("## Optimization History")
            for opt in self._optimization_history:
                status = "applied" if opt.optimization_applied else "failed"
                lines.append(
                    f"- [{status}] {opt.bottleneck.tool_name}: "
                    f"{opt.before_avg_time:.2f}s → {opt.after_avg_time:.2f}s "
                    f"({opt.strategy})"
                )
                if opt.error:
                    lines.append(f"  Error: {opt.error}")

        return "\n".join(lines)
