"""
Usage Analytics and Insights — Token usage patterns, cost trends, tool frequency.

Aggregates usage data across sessions and time periods to provide actionable
insights about model usage, cost optimization opportunities, and tool
utilization patterns.

Usage
-----
    insights = InsightsManager()
    insights.record_usage(model="claude-sonnet-4", input_tokens=1500, output_tokens=500)
    insights.record_tool_usage("read_file", duration_ms=150)
    report = insights.get_report()
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class UsageSnapshot:
    """
    A point-in-time snapshot of usage metrics.

    Attributes
    ----------
    timestamp:
        ISO-8601 timestamp of the snapshot.
    model:
        Model name used.
    input_tokens:
        Input token count.
    output_tokens:
        Output token count.
    cost_usd:
        Estimated cost.
    duration_ms:
        Request duration in milliseconds.
    session_id:
        Session identifier.
    tool_calls_count:
        Number of tool calls made.
    task_type:
        Optional task classification.
    """

    timestamp: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    session_id: str = ""
    tool_calls_count: int = 0
    task_type: str = ""


@dataclass
class ToolUsageRecord:
    """Record of a single tool usage event."""

    tool_name: str
    timestamp: str
    duration_ms: float = 0.0
    success: bool = True
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a single model."""

    model_name: str
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_duration_ms: float = 0.0
    avg_tokens_per_request: int = 0
    error_count: int = 0
    success_rate: float = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# InsightsManager
# ──────────────────────────────────────────────────────────────────────────────

class InsightsManager:
    """
    Usage analytics and insights manager.

    Aggregates token usage, cost data, tool usage, and model performance
    to provide comprehensive analytics for optimization.

    Parameters
    ----------
    session_id:
        Current session identifier.
    retention_days:
        Number of days to retain data (default: 90).
    persistence_dir:
        Directory for persisting analytics data.
    """

    def __init__(
        self,
        session_id: str = "default",
        retention_days: int = 90,
        persistence_dir: Optional[str] = None,
    ) -> None:
        self._session_id = session_id
        self._retention_days = retention_days
        self._persistence_dir = Path(persistence_dir) if persistence_dir else Path.home() / ".atlas" / "insights"

        # Storage
        self._usage_snapshots: List[UsageSnapshot] = []
        self._tool_usage: List[ToolUsageRecord] = []
        self._errors: List[Dict[str, Any]] = []

    # ── Recording ─────────────────────────────────────────────────────────

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
        duration_ms: float = 0.0,
        session_id: Optional[str] = None,
        tool_calls_count: int = 0,
        task_type: str = "",
    ) -> UsageSnapshot:
        """
        Record a usage snapshot.

        Parameters
        ----------
        model:
            Model name used.
        input_tokens:
            Input token count.
        output_tokens:
            Output token count.
        cost_usd:
            Estimated cost.
        duration_ms:
            Request duration.
        session_id:
            Override session ID.
        tool_calls_count:
            Number of tool calls made.
        task_type:
            Task classification.

        Returns
        -------
        UsageSnapshot
            The recorded snapshot.
        """
        snapshot = UsageSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            session_id=session_id or self._session_id,
            tool_calls_count=tool_calls_count,
            task_type=task_type,
        )
        self._usage_snapshots.append(snapshot)
        self._prune_if_needed()
        return snapshot

    def record_tool_usage(
        self,
        tool_name: str,
        duration_ms: float = 0.0,
        success: bool = True,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolUsageRecord:
        """
        Record a tool usage event.

        Parameters
        ----------
        tool_name:
            Name of the tool used.
        duration_ms:
            Execution duration.
        success:
            Whether the tool call succeeded.
        session_id:
            Override session ID.
        metadata:
            Additional metadata.

        Returns
        -------
        ToolUsageRecord
            The recorded tool usage.
        """
        record = ToolUsageRecord(
            tool_name=tool_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
            success=success,
            session_id=session_id or self._session_id,
            metadata=metadata or {},
        )
        self._tool_usage.append(record)
        return record

    def record_error(
        self,
        model: str,
        error_type: str,
        error_message: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Record an error event for analytics."""
        self._errors.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "error_type": error_type,
            "error_message": error_message[:500],
            "session_id": session_id or self._session_id,
        })

    # ── Analytics queries ─────────────────────────────────────────────────

    def get_token_usage_patterns(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze token usage patterns over time.

        Parameters
        ----------
        days:
            Number of days to analyze.

        Returns
        -------
        dict
            Usage patterns including daily breakdowns and averages.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        relevant = [s for s in self._usage_snapshots if s.timestamp >= cutoff]

        # Daily aggregation
        daily: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "requests": 0}
        )

        for snapshot in relevant:
            date_key = snapshot.timestamp[:10]  # YYYY-MM-DD
            daily[date_key]["input_tokens"] += snapshot.input_tokens
            daily[date_key]["output_tokens"] += snapshot.output_tokens
            daily[date_key]["cost_usd"] += snapshot.cost_usd
            daily[date_key]["requests"] += 1

        # Averages
        total_input = sum(s.input_tokens for s in relevant)
        total_output = sum(s.output_tokens for s in relevant)
        total_requests = len(relevant)
        avg_input = total_input // max(1, total_requests)
        avg_output = total_output // max(1, total_requests)

        return {
            "period_days": days,
            "total_requests": total_requests,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "avg_input_tokens_per_request": avg_input,
            "avg_output_tokens_per_request": avg_output,
            "daily_breakdown": dict(daily),
        }

    def get_cost_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze cost trends over time.

        Returns
        -------
        dict
            Cost trends including daily costs, moving averages, and projections.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        relevant = [s for s in self._usage_snapshots if s.timestamp >= cutoff]

        # Daily costs
        daily_costs: Dict[str, float] = defaultdict(float)
        for snapshot in relevant:
            date_key = snapshot.timestamp[:10]
            daily_costs[date_key] += snapshot.cost_usd

        # Calculate moving average (7-day)
        sorted_dates = sorted(daily_costs.keys())
        moving_avg: Dict[str, float] = {}
        for i, date in enumerate(sorted_dates):
            window_start = max(0, i - 6)
            window = [daily_costs[d] for d in sorted_dates[window_start:i + 1]]
            moving_avg[date] = sum(window) / len(window)

        total_cost = sum(s.cost_usd for s in relevant)
        avg_daily = total_cost / max(1, len(daily_costs))

        # Projected monthly cost
        projected_monthly = avg_daily * 30

        return {
            "period_days": days,
            "total_cost_usd": round(total_cost, 6),
            "average_daily_cost_usd": round(avg_daily, 6),
            "projected_monthly_cost_usd": round(projected_monthly, 6),
            "daily_costs": {k: round(v, 6) for k, v in daily_costs.items()},
            "moving_average_7d": {k: round(v, 6) for k, v in moving_avg.items()},
            "costliest_day": max(daily_costs.items(), key=lambda x: x[1]) if daily_costs else ("N/A", 0),
            "cheapest_day": min(daily_costs.items(), key=lambda x: x[1]) if daily_costs else ("N/A", 0),
        }

    def get_tool_usage_frequency(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze tool usage patterns.

        Returns
        -------
        dict
            Tool usage statistics including frequency, success rates, and
            average durations.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        relevant = [t for t in self._tool_usage if t.timestamp >= cutoff]

        tool_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "success_count": 0, "total_duration_ms": 0.0, "errors": 0}
        )

        for record in relevant:
            stats = tool_stats[record.tool_name]
            stats["count"] += 1
            stats["total_duration_ms"] += record.duration_ms
            if record.success:
                stats["success_count"] += 1
            else:
                stats["errors"] += 1

        result: Dict[str, Any] = {}
        for tool_name, stats in sorted(tool_stats.items(), key=lambda x: x[1]["count"], reverse=True):
            count = stats["count"]
            result[tool_name] = {
                "count": count,
                "success_count": stats["success_count"],
                "success_rate": round(stats["success_count"] / max(1, count), 4),
                "avg_duration_ms": round(stats["total_duration_ms"] / max(1, count), 2),
                "errors": stats["errors"],
            }

        return {
            "period_days": days,
            "total_tool_calls": len(relevant),
            "unique_tools": len(tool_stats),
            "tool_breakdown": result,
            "most_used_tool": max(tool_stats.items(), key=lambda x: x[1]["count"])[0] if tool_stats else "N/A",
        }

    def get_model_performance(self) -> Dict[str, ModelPerformanceMetrics]:
        """
        Get performance metrics for each model used.

        Returns
        -------
        dict[str, ModelPerformanceMetrics]
            Metrics keyed by model name.
        """
        model_metrics: Dict[str, ModelPerformanceMetrics] = {}

        for snapshot in self._usage_snapshots:
            model = snapshot.model
            if model not in model_metrics:
                model_metrics[model] = ModelPerformanceMetrics(model_name=model)

            metrics = model_metrics[model]
            metrics.total_requests += 1
            metrics.total_input_tokens += snapshot.input_tokens
            metrics.total_output_tokens += snapshot.output_tokens
            metrics.total_cost_usd += snapshot.cost_usd

        # Calculate averages
        for metrics in model_metrics.values():
            if metrics.total_requests > 0:
                durations = [
                    s.duration_ms for s in self._usage_snapshots
                    if s.model == metrics.model_name and s.duration_ms > 0
                ]
                metrics.avg_duration_ms = (
                    sum(durations) / len(durations) if durations else 0.0
                )
                metrics.avg_tokens_per_request = (
                    (metrics.total_input_tokens + metrics.total_output_tokens) // metrics.total_requests
                )

            # Calculate error rate
            error_count = sum(
                1 for e in self._errors if e.get("model") == metrics.model_name
            )
            metrics.error_count = error_count
            metrics.success_rate = (
                1.0 - (error_count / max(1, metrics.total_requests))
            )

        return model_metrics

    # ── Comprehensive report ───────────────────────────────────────────────

    def get_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate a comprehensive analytics report.

        Parameters
        ----------
        days:
            Number of days to include in the report.

        Returns
        -------
        dict
            Complete analytics report with all metrics.
        """
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period_days": days,
            "session_id": self._session_id,
            "token_usage": self.get_token_usage_patterns(days),
            "cost_trends": self.get_cost_trends(days),
            "tool_usage": self.get_tool_usage_frequency(days),
            "model_performance": {
                name: {
                    "total_requests": m.total_requests,
                    "total_cost_usd": round(m.total_cost_usd, 6),
                    "avg_duration_ms": round(m.avg_duration_ms, 2),
                    "success_rate": round(m.success_rate, 4),
                }
                for name, m in self.get_model_performance().items()
            },
            "error_summary": {
                "total_errors": len(self._errors),
                "error_types": dict(Counter(e.get("error_type", "unknown") for e in self._errors)),
            },
        }

    # ── Persistence ───────────────────────────────────────────────────────

    async def save(self) -> None:
        """Persist analytics data to disk."""
        import asyncio

        self._persistence_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._persistence_dir / f"insights_{self._session_id}.json"

        data = {
            "usage_snapshots": [
                {
                    "timestamp": s.timestamp,
                    "model": s.model,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cost_usd": s.cost_usd,
                    "duration_ms": s.duration_ms,
                    "session_id": s.session_id,
                    "tool_calls_count": s.tool_calls_count,
                    "task_type": s.task_type,
                }
                for s in self._usage_snapshots
            ],
            "tool_usage": [
                {
                    "tool_name": t.tool_name,
                    "timestamp": t.timestamp,
                    "duration_ms": t.duration_ms,
                    "success": t.success,
                    "session_id": t.session_id,
                }
                for t in self._tool_usage
            ],
            "errors": self._errors[-1000:],  # Keep last 1000 errors
        }

        def _write():
            filepath.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write)

    async def load(self) -> None:
        """Load analytics data from disk."""
        import asyncio

        filepath = self._persistence_dir / f"insights_{self._session_id}.json"
        if not filepath.exists():
            return

        def _read():
            return json.loads(filepath.read_text(encoding="utf-8"))

        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, _read)

            self._usage_snapshots = [
                UsageSnapshot(**s) for s in data.get("usage_snapshots", [])
            ]
            self._tool_usage = [
                ToolUsageRecord(**t) for t in data.get("tool_usage", [])
            ]
            self._errors = data.get("errors", [])
        except Exception as e:
            logger.warning("Failed to load insights data: %s", e)

    def clear(self) -> None:
        """Clear all recorded data."""
        self._usage_snapshots.clear()
        self._tool_usage.clear()
        self._errors.clear()

    # ── Internal ──────────────────────────────────────────────────────────

    def _prune_if_needed(self) -> None:
        """Remove entries older than the retention period."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._retention_days)).isoformat()
        self._usage_snapshots = [
            s for s in self._usage_snapshots if s.timestamp >= cutoff
        ]
        self._tool_usage = [
            t for t in self._tool_usage if t.timestamp >= cutoff
        ]
        self._errors = [
            e for e in self._errors if e.get("timestamp", "") >= cutoff
        ]
