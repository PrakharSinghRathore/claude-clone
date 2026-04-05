"""
Hindsight Memory Plugin — Post-hoc memory analysis with pattern
recognition in past conversations and insight extraction.

Plugin manifest example (plugin.yaml)::

    name: hindsight
    display_name: Hindsight Memory
    version: 1.0.0
    type: post_hoc
    description: Post-hoc analysis of past conversations for pattern recognition and insight extraction.
    author: Hermes Team
    required_packages: []
    config_schema:
      storage_path:
        type: string
        description: Directory for conversation archives and insights
      analysis_interval_hours:
        type: integer
        description: How often to run post-hoc analysis (default: 24)
      min_conversations_for_pattern:
        type: integer
        description: Minimum conversations before pattern extraction (default: 10)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from datetime import datetime, timedelta
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


@register_builtin("hindsight")
class HindsightMemoryPlugin(BaseMemoryPlugin):
    """
    Post-hoc memory analysis plugin.

    Periodically analyses past conversations to identify patterns,
    extract insights, and surface recommendations. Operates on
    stored conversation logs rather than real-time interaction.
    """

    metadata = MemoryPluginMetadata(
        name="hindsight",
        display_name="Hindsight Memory",
        version="1.0.0",
        description="Post-hoc analysis of conversations for pattern recognition and insights.",
        plugin_type=MemoryPluginType.POST_HOC,
        author="Hermes Team",
        required_packages=[],
    )

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        super().__init__(config)
        self._storage_path: Path = Path(
            config.storage_path or "~/.claude_clone/hindsight"
        ).expanduser().resolve()
        self._conversations_path: Path = self._storage_path / "conversations"
        self._insights_path: Path = self._storage_path / "insights"
        self._patterns_path: Path = self._storage_path / "patterns.json"
        self._patterns: dict[str, Any] = {}
        self._analysis_interval = config.extra.get("analysis_interval_hours", 24)
        self._min_conversations = config.extra.get("min_conversations_for_pattern", 10)

    async def initialize(self) -> None:
        """Set up storage directories and load existing patterns."""
        self._conversations_path.mkdir(parents=True, exist_ok=True)
        self._insights_path.mkdir(parents=True, exist_ok=True)
        self._load_patterns()
        self._initialized = True
        logger.info("Hindsight memory plugin initialized (path=%s)", self._storage_path)

    async def store(self, entry: MemoryEntry) -> str:
        """Store a conversation log entry for later analysis."""
        entry_id = entry.id or str(uuid.uuid4())
        entry.id = entry_id

        conv_file = self._conversations_path / f"{entry_id}.json"
        data = {
            "id": entry_id,
            "content": entry.content,
            "metadata": entry.metadata,
            "tags": entry.tags,
            "created_at": datetime.utcnow().isoformat(),
        }
        conv_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return entry_id

    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a stored conversation entry."""
        conv_file = self._conversations_path / f"{entry_id}.json"
        if not conv_file.exists():
            return None
        try:
            data = json.loads(conv_file.read_text(encoding="utf-8"))
            return MemoryEntry(
                id=data["id"],
                content=data["content"],
                metadata=data.get("metadata", {}),
                tags=data.get("tags", []),
                source="hindsight",
            )
        except (json.JSONDecodeError, KeyError):
            return None

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """Search stored conversations by keyword."""
        results: list[MemoryEntry] = []
        query_lower = query.lower()

        for conv_file in sorted(self._conversations_path.glob("*.json"), reverse=True):
            if len(results) >= limit:
                break
            try:
                data = json.loads(conv_file.read_text(encoding="utf-8"))
                content = data.get("content", "")
                if query_lower in content.lower():
                    entry = MemoryEntry(
                        id=data.get("id", conv_file.stem),
                        content=content,
                        metadata=data.get("metadata", {}),
                        tags=data.get("tags", []),
                        source="hindsight",
                        relevance_score=0.8,
                    )
                    results.append(entry)
            except (json.JSONDecodeError, OSError):
                continue
        return results[:limit]

    async def delete(self, entry_id: str) -> bool:
        """Delete a stored conversation entry."""
        conv_file = self._conversations_path / f"{entry_id}.json"
        if conv_file.exists():
            conv_file.unlink()
            return True
        return False

    async def health_check(self) -> dict:
        """Check plugin health and analysis status."""
        import time
        start = time.monotonic()
        conv_count = len(list(self._conversations_path.glob("*.json")))
        insight_count = len(list(self._insights_path.glob("*.json")))
        pattern_count = len(self._patterns)
        elapsed = (time.monotonic() - start) * 1000
        return {
            "status": "healthy",
            "latency_ms": elapsed,
            "details": {
                "conversations": conv_count,
                "insights": insight_count,
                "patterns": pattern_count,
                "analysis_interval_hours": self._analysis_interval,
            },
        }

    async def shutdown(self) -> None:
        """Save patterns and clean up."""
        self._save_patterns()
        self._initialized = False

    # ------------------------------------------------------------------
    # Post-hoc analysis
    # ------------------------------------------------------------------

    async def analyze_conversations(self) -> list[dict]:
        """
        Run post-hoc analysis on all stored conversations.

        Identifies:
        - Recurring topics and themes
        - Common error patterns
        - Task frequency
        - Time-based patterns
        - Improvement suggestions

        Returns a list of insight dicts.
        """
        conversations = self._load_all_conversations()
        if len(conversations) < 3:
            logger.debug("Not enough conversations for analysis (%d)", len(conversations))
            return []

        insights: list[dict] = []

        # Extract topics
        topic_patterns = self._extract_topics(conversations)
        if topic_patterns:
            insights.append({
                "type": "topic_patterns",
                "data": topic_patterns,
                "generated_at": datetime.utcnow().isoformat(),
            })

        # Extract task patterns
        task_patterns = self._extract_task_patterns(conversations)
        if task_patterns:
            insights.append({
                "type": "task_patterns",
                "data": task_patterns,
                "generated_at": datetime.utcnow().isoformat(),
            })

        # Extract error patterns
        error_patterns = self._extract_error_patterns(conversations)
        if error_patterns:
            insights.append({
                "type": "error_patterns",
                "data": error_patterns,
                "generated_at": datetime.utcnow().isoformat(),
            })

        # Generate recommendations
        recommendations = self._generate_recommendations(topic_patterns, task_patterns, error_patterns)
        if recommendations:
            insights.append({
                "type": "recommendations",
                "data": recommendations,
                "generated_at": datetime.utcnow().isoformat(),
            })

        # Persist insights
        for insight in insights:
            insight_id = str(uuid.uuid4())
            insight_file = self._insights_path / f"{insight_id}.json"
            insight_file.write_text(
                json.dumps(insight, indent=2, default=str), encoding="utf-8"
            )

        # Update patterns cache
        self._patterns.update({
            "last_analysis": datetime.utcnow().isoformat(),
            "total_conversations": len(conversations),
            "total_insights": len(insights),
        })
        self._save_patterns()

        logger.info("Post-hoc analysis produced %d insights from %d conversations", len(insights), len(conversations))
        return insights

    async def get_insights(self, insight_type: Optional[str] = None, limit: int = 20) -> list[dict]:
        """Retrieve previously generated insights."""
        results: list[dict] = []
        for insight_file in sorted(self._insights_path.glob("*.json"), reverse=True):
            if len(results) >= limit:
                break
            try:
                data = json.loads(insight_file.read_text(encoding="utf-8"))
                if insight_type and data.get("type") != insight_type:
                    continue
                results.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return results

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _load_all_conversations(self) -> list[dict]:
        """Load all conversation entries from storage."""
        conversations: list[dict] = []
        for conv_file in self._conversations_path.glob("*.json"):
            try:
                data = json.loads(conv_file.read_text(encoding="utf-8"))
                conversations.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return conversations

    @staticmethod
    def _extract_topics(conversations: list[dict]) -> dict[str, int]:
        """Extract recurring topics from conversation content."""
        # Common topic keywords
        topic_keywords = {
            "code_review": ["review", "refactor", "clean", "improve", "quality"],
            "debugging": ["debug", "error", "fix", "bug", "issue", "broken"],
            "deployment": ["deploy", "release", "production", "server", "ship"],
            "testing": ["test", "spec", "coverage", "unit test", "integration"],
            "documentation": ["docs", "readme", "comment", "document"],
            "git": ["commit", "branch", "merge", "pull request", "push"],
            "security": ["security", "vulnerability", "auth", "permission"],
            "performance": ["performance", "optimize", "speed", "slow", "latency"],
        }

        topic_counts: dict[str, int] = {}
        for conv in conversations:
            content = conv.get("content", "").lower()
            for topic, keywords in topic_keywords.items():
                if any(kw in content for kw in keywords):
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1

        return dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def _extract_task_patterns(conversations: list[dict]) -> dict[str, Any]:
        """Identify common task types and frequencies."""
        task_types: Counter = Counter()
        for conv in conversations:
            tags = conv.get("tags", [])
            for tag in tags:
                task_types[tag] += 1

        return {
            "task_type_counts": dict(task_types.most_common(20)),
            "total_tasks": sum(task_types.values()),
            "unique_types": len(task_types),
        }

    @staticmethod
    def _extract_error_patterns(conversations: list[dict]) -> list[dict]:
        """Identify recurring error patterns."""
        error_indicators = ["error", "exception", "traceback", "failed", "failure"]
        error_counts: Counter = Counter()

        for conv in conversations:
            content = conv.get("content", "").lower()
            for indicator in error_indicators:
                if indicator in content:
                    # Extract context around the error
                    words = content.split()
                    for i, word in enumerate(words):
                        if indicator in word:
                            context = " ".join(words[max(0, i-3):i+3])
                            error_counts[context] += 1

        return [
            {"pattern": pattern, "count": count}
            for pattern, count in error_counts.most_common(10)
        ]

    @staticmethod
    def _generate_recommendations(
        topics: dict,
        tasks: dict,
        errors: list[dict],
    ) -> list[str]:
        """Generate actionable recommendations from analysis."""
        recs: list[str] = []

        if topics:
            top_topic = next(iter(topics))
            recs.append(f"Most common topic: '{top_topic}' ({topics[top_topic]} occurrences). Consider creating a dedicated skill.")

        if tasks and tasks.get("unique_types", 0) > 5:
            recs.append(f"High task diversity ({tasks['unique_types']} types). Consider grouping similar tasks for efficiency.")

        if errors and len(errors) >= 3:
            top_error = errors[0]["pattern"][:60]
            recs.append(f"Recurring error pattern detected: '{top_error}...' — consider adding error prevention checks.")

        return recs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_patterns(self) -> None:
        """Load cached patterns from disk."""
        if self._patterns_path.exists():
            try:
                self._patterns = json.loads(self._patterns_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._patterns = {}

    def _save_patterns(self) -> None:
        """Persist patterns to disk."""
        try:
            self._patterns_path.write_text(
                json.dumps(self._patterns, indent=2, default=str), encoding="utf-8"
            )
        except OSError:
            logger.exception("Failed to save patterns")
