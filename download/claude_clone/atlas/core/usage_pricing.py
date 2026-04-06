"""
Usage Pricing — Per-model pricing database and cost tracking.

Maintains a comprehensive pricing database across multiple providers
(Anthropic, OpenAI, OpenRouter, Nous, Google, DeepSeek, etc.) and tracks
session and daily cost accumulation.

Usage
-----
    pricing = UsagePricing()
    cost = pricing.calculate("claude-sonnet-4-20250514", 1500, 500)
    print(f"${cost:.6f}")

    # Track usage
    pricing.record("claude-sonnet-4-20250514", 1500, 500)
    print(pricing.session_summary())
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from atlas.constants import (
    COST_TRACKING_RETENTION_DAYS,
    COST_WARNING_THRESHOLD,
    ATLAS_DATA_HOME,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CostEntry:
    """
    A single cost record.

    Attributes
    ----------
    model:
        Model name used.
    provider:
        API provider (anthropic, openai, etc.).
    input_tokens:
        Number of input tokens.
    output_tokens:
        Number of output tokens.
    cost_usd:
        Calculated cost in USD.
    timestamp:
        ISO-8601 timestamp.
    session_id:
        Session identifier.
    metadata:
        Additional metadata (task type, etc.).
    """

    model: str
    provider: str = "anthropic"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Pricing Database
# ──────────────────────────────────────────────────────────────────────────────

# Pricing: model_key → (input_cost_per_1M_tokens, output_cost_per_1M_tokens)
PRICING_DB: Dict[str, Tuple[float, float]] = {
    # Anthropic Claude
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.8, 4.0),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # OpenAI
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    "o3-mini": (1.10, 4.40),
    # Google
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-pro": (1.25, 10.0),
    "gemini-1.5-pro": (1.25, 5.0),
    # DeepSeek
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
    # Meta (via OpenRouter / providers)
    "llama-3.3-70b": (0.39, 0.39),
    "llama-3.1-405b": (2.0, 2.0),
    # Mistral
    "mistral-large": (2.0, 6.0),
    "mistral-small": (0.20, 0.60),
    "codestral": (0.30, 0.90),
    # NousResearch (via OpenRouter)
    "atlas-3-llama-3.1-405b": (2.7, 2.7),
    "atlas-3-llama-3.1-70b": (0.39, 0.39),
    # Google Gemini (additional)
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.15, 0.60),
    # xAI Grok
    "grok-3": (3.0, 15.0),
    "grok-3-mini": (0.30, 0.50),
    # Mistral (additional)
    "mistral-medium": (2.70, 8.10),
    # DeepSeek (additional)
    "deepseek-coder": (0.14, 0.28),
    # Groq
    "llama-3.3-70b-groq": (0.59, 0.79),
    "mixtral-8x7b-groq": (0.24, 0.24),
    # Together AI
    "llama-3.3-70b-together": (0.88, 0.88),
    "qwen-2.5-72b-together": (0.80, 0.80),
    # Fireworks AI
    "llama-3.3-70b-fireworks": (0.90, 0.90),
    "mixtral-8x7b-fireworks": (0.50, 0.50),
    # OpenRouter (aggregator)
    "openrouter-auto": (3.0, 15.0),
    # Amazon Bedrock
    "claude-via-bedrock": (3.0, 15.0),
    # NVIDIA NIM
    "llama-3.1-nemotron": (0.90, 0.90),
    # Perplexity
    "sonar": (1.0, 1.0),
    "sonar-pro": (3.0, 3.0),
    # Venice
    "venice-llama": (2.0, 2.0),
    # MiniMax
    "minimax-text": (0.20, 0.20),
    # vLLM (local — free)
    "vllm-custom": (0.0, 0.0),
    # SGLang (local — free)
    "sglang-custom": (0.0, 0.0),
    # LiteLLM (universal proxy — defaults to mid-tier pricing)
    "litellm-proxy": (3.0, 15.0),
}

# Default pricing for unknown models (per 1M tokens)
DEFAULT_INPUT_PRICE = 3.0
DEFAULT_OUTPUT_PRICE = 15.0


def lookup_pricing(model_name: str) -> Tuple[float, float]:
    """
    Look up pricing for a model.

    Strips provider prefixes and checks aliases.

    Parameters
    ----------
    model_name:
        The model name (optionally with provider prefix).

    Returns
    -------
    tuple[float, float]
        (input_cost_per_1M_tokens, output_cost_per_1M_tokens)
    """
    # Strip provider prefix
    lookup = model_name.split("/")[-1] if "/" in model_name else model_name

    # Direct lookup
    if lookup in PRICING_DB:
        return PRICING_DB[lookup]

    # Normalize: lowercase
    lookup_lower = lookup.lower()
    for key, pricing in PRICING_DB.items():
        if key.lower() == lookup_lower:
            return pricing

    return DEFAULT_INPUT_PRICE, DEFAULT_OUTPUT_PRICE


# ──────────────────────────────────────────────────────────────────────────────
# UsagePricing
# ──────────────────────────────────────────────────────────────────────────────

class UsagePricing:
    """
    Per-model pricing calculator and cost tracker.

    Tracks usage costs across sessions and time periods, with daily budget
    alerts and multi-provider support.

    Parameters
    ----------
    session_id:
        Current session identifier.
    budget_threshold:
        Cost warning threshold in USD (default: $1.00).
    persistence_dir:
        Directory for persisting cost history. If ``None``, uses default.
    """

    def __init__(
        self,
        session_id: str = "default",
        budget_threshold: float = COST_WARNING_THRESHOLD,
        persistence_dir: Optional[str] = None,
    ) -> None:
        self._session_id = session_id
        self._budget_threshold = budget_threshold
        self._persistence_dir = Path(persistence_dir) if persistence_dir else ATLAS_DATA_HOME / "usage"
        self._entries: List[CostEntry] = []
        self._warnings_shown: set = set()

    @property
    def session_id(self) -> str:
        """Current session identifier."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value

    # ── Cost calculation ──────────────────────────────────────────────────

    @staticmethod
    def calculate(
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost in USD for given model and token counts.

        Parameters
        ----------
        model_name:
            The model name.
        input_tokens:
            Number of input tokens.
        output_tokens:
            Number of output tokens.

        Returns
        -------
        float
            Cost in USD.
        """
        input_price, output_price = lookup_pricing(model_name)
        return (
            (input_tokens / 1_000_000) * input_price
            + (output_tokens / 1_000_000) * output_price
        )

    @staticmethod
    def estimate_for_model(model_name: str, prompt_tokens: int = 1000, response_tokens: int = 500) -> float:
        """
        Quick estimate for a typical request to a given model.

        Parameters
        ----------
        model_name:
            The model name.
        prompt_tokens:
            Estimated prompt tokens.
        response_tokens:
            Estimated response tokens.

        Returns
        -------
        float
            Estimated cost in USD.
        """
        return UsagePricing.calculate(model_name, prompt_tokens, response_tokens)

    def get_pricing(self, model_name: str) -> Tuple[float, float]:
        """
        Get the per-1M-token pricing for a model.

        Returns
        -------
        tuple[float, float]
            (input_price_per_1M, output_price_per_1M)
        """
        return lookup_pricing(model_name)

    def get_all_pricing(self) -> Dict[str, Tuple[float, float]]:
        """Return the full pricing database."""
        return dict(PRICING_DB)

    # ── Usage tracking ────────────────────────────────────────────────────

    def record(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostEntry:
        """
        Record a usage event and return the cost entry.

        Parameters
        ----------
        model_name:
            The model used.
        input_tokens:
            Input token count.
        output_tokens:
            Output token count.
        session_id:
            Override session ID for this entry.
        metadata:
            Additional metadata.

        Returns
        -------
        CostEntry
            The recorded cost entry.
        """
        cost = self.calculate(model_name, input_tokens, output_tokens)
        provider = "unknown"
        if "/" in model_name:
            provider = model_name.split("/")[0]
        elif model_name.startswith("claude"):
            provider = "anthropic"
        elif model_name.startswith("gpt") or model_name.startswith("o1") or model_name.startswith("o3"):
            provider = "openai"
        elif model_name.startswith("gemini"):
            provider = "google"
        elif model_name.startswith("deepseek"):
            provider = "deepseek"
        elif model_name.startswith("grok"):
            provider = "xai"
        elif model_name.startswith("mistral") or model_name.startswith("codestral"):
            provider = "mistral"
        elif model_name.startswith("llama-3.3-70b-groq") or model_name.startswith("mixtral-8x7b-groq"):
            provider = "groq"
        elif model_name.startswith("llama-3.3-70b-together") or model_name.startswith("qwen-2.5-72b"):
            provider = "together"
        elif model_name.startswith("llama-3.3-70b-fireworks") or model_name.startswith("mixtral-8x7b-fireworks"):
            provider = "fireworks"
        elif model_name.startswith("openrouter"):
            provider = "openrouter"
        elif model_name.startswith("claude-via-bedrock"):
            provider = "amazon_bedrock"
        elif model_name.startswith("llama-3.1-nemotron"):
            provider = "nvidia_nim"
        elif model_name.startswith("sonar"):
            provider = "perplexity"
        elif model_name.startswith("venice"):
            provider = "venice"
        elif model_name.startswith("minimax"):
            provider = "minimax"
        elif model_name.startswith("vllm"):
            provider = "vllm"
        elif model_name.startswith("sglang"):
            provider = "sglang"
        elif model_name.startswith("litellm"):
            provider = "litellm"

        entry = CostEntry(
            model=model_name,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id or self._session_id,
            metadata=metadata or {},
        )

        self._entries.append(entry)
        self._check_budget(entry)

        return entry

    def _check_budget(self, entry: CostEntry) -> None:
        """Check if the budget threshold has been exceeded."""
        session_cost = self.session_total_cost(entry.session_id)
        key = f"session:{entry.session_id}"
        if session_cost >= self._budget_threshold and key not in self._warnings_shown:
            self._warnings_shown.add(key)
            logger.warning(
                "Session %s cost $%.4f exceeds threshold of $%.2f",
                entry.session_id, session_cost, self._budget_threshold,
            )

    # ── Aggregation ───────────────────────────────────────────────────────

    def session_total_cost(self, session_id: Optional[str] = None) -> float:
        """Calculate total cost for the current (or specified) session."""
        sid = session_id or self._session_id
        return sum(e.cost_usd for e in self._entries if e.session_id == sid)

    def session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a cost summary for the current (or specified) session.

        Returns
        -------
        dict
            Summary with ``total_cost``, ``total_tokens``, ``model_breakdown``,
            ``entry_count``, etc.
        """
        sid = session_id or self._session_id
        entries = [e for e in self._entries if e.session_id == sid]

        model_costs: Dict[str, float] = defaultdict(float)
        model_tokens: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
        total_input = 0
        total_output = 0
        total_cost = 0.0

        for entry in entries:
            model_costs[entry.model] += entry.cost_usd
            inp, out = model_tokens[entry.model]
            model_tokens[entry.model] = (inp + entry.input_tokens, out + entry.output_tokens)
            total_input += entry.input_tokens
            total_output += entry.output_tokens
            total_cost += entry.cost_usd

        return {
            "session_id": sid,
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "entry_count": len(entries),
            "model_breakdown": {
                model: {
                    "cost_usd": round(cost, 6),
                    "input_tokens": tokens[0],
                    "output_tokens": tokens[1],
                }
                for model, cost, tokens in [
                    (m, c, model_tokens[m]) for m, c in model_costs.items()
                ]
            },
        }

    def daily_cost(self, date: Optional[str] = None) -> float:
        """
        Calculate total cost for a specific date.

        Parameters
        ----------
        date:
            Date string in YYYY-MM-DD format. If ``None``, uses today.

        Returns
        -------
        float
            Total cost for that day.
        """
        target = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return sum(
            e.cost_usd for e in self._entries
            if e.timestamp.startswith(target)
        )

    def daily_summary(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Generate daily cost summaries for the past N days.

        Parameters
        ----------
        days:
            Number of days to include.

        Returns
        -------
        list[dict]
            List of daily summaries with ``date``, ``total_cost``,
            ``total_tokens``, and ``entry_count``.
        """
        summaries: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)

        for i in range(days):
            date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            day_entries = [e for e in self._entries if e.timestamp.startswith(date)]

            summaries.append({
                "date": date,
                "total_cost_usd": round(sum(e.cost_usd for e in day_entries), 6),
                "total_tokens": sum(e.input_tokens + e.output_tokens for e in day_entries),
                "entry_count": len(day_entries),
            })

        return summaries

    def top_models(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top models by total cost.

        Returns
        -------
        list[dict]
            Sorted list of model cost records.
        """
        model_costs: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "entries": 0}
        )
        for entry in self._entries:
            m = model_costs[entry.model]
            m["cost_usd"] += entry.cost_usd
            m["input_tokens"] += entry.input_tokens
            m["output_tokens"] += entry.output_tokens
            m["entries"] += 1

        sorted_models = sorted(model_costs.items(), key=lambda x: x[1]["cost_usd"], reverse=True)
        return [
            {"model": model, **stats} for model, stats in sorted_models[:limit]
        ]

    # ── Persistence ───────────────────────────────────────────────────────

    async def save(self) -> None:
        """Persist cost entries to disk."""
        import asyncio

        self._persistence_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._persistence_dir / f"costs_{self._session_id}.json"

        # Prune old entries
        cutoff = (datetime.now(timezone.utc) - timedelta(days=COST_TRACKING_RETENTION_DAYS)).isoformat()
        recent_entries = [e for e in self._entries if e.timestamp >= cutoff]

        data = [e.to_dict() for e in recent_entries]

        def _write():
            filepath.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write)

    async def load(self, session_id: Optional[str] = None) -> None:
        """Load cost entries from disk."""
        import asyncio

        sid = session_id or self._session_id
        filepath = self._persistence_dir / f"costs_{sid}.json"
        if not filepath.exists():
            return

        def _read():
            content = filepath.read_text(encoding="utf-8")
            return [CostEntry(**d) for d in json.loads(content)]

        loop = asyncio.get_running_loop()
        try:
            self._entries = await loop.run_in_executor(None, _read)
        except Exception as e:
            logger.warning("Failed to load cost data: %s", e)

    def clear(self) -> None:
        """Clear all recorded entries."""
        self._entries.clear()
        self._warnings_shown.clear()
