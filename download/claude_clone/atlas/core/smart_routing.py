"""
Smart Model Routing — Task-aware, cost-aware, latency-aware model selection.

Routes tasks to the most appropriate model based on task type classification,
cost constraints, latency requirements, and model capabilities. Integrates
with the existing ``agent/model_router.py`` while providing Atlas-specific
enhancements.

Usage
-----
    router = SmartRouter()
    decision = await router.route(
        task_type="reasoning",
        prompt="Analyze the trade-offs between microservices and monoliths...",
        max_cost=0.05,
    )
    print(decision.model_name, decision.reason)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from atlas.constants import ROUTING_COST_WEIGHT, ROUTING_LATENCY_WEIGHT, ROUTING_QUALITY_WEIGHT
from atlas.core.model_metadata import ModelCapability, ModelInfo, ModelMetadata, get_model_info

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Task types
# ──────────────────────────────────────────────────────────────────────────────

class TaskCategory(Enum):
    """High-level task categories for routing decisions."""

    REASONING = "reasoning"
    """Complex analysis, multi-step problems, strategic thinking."""

    CODE = "code"
    """Code generation, debugging, refactoring, code review."""

    CHAT = "chat"
    """General conversation, Q&A, quick questions."""

    SUMMARIZATION = "summarization"
    """Summarizing, condensing, extracting key points."""

    CREATIVE = "creative"
    """Creative writing, brainstorming, ideation."""

    CLASSIFICATION = "classification"
    """Categorization, labeling, sentiment analysis."""

    TRANSLATION = "translation"
    """Language translation and localization."""

    EMBEDDING = "embedding"
    """Text embedding and similarity computation."""

    VISION = "vision"
    """Image analysis, visual understanding, screenshot interpretation."""

    AUDIO = "audio"
    """Audio transcription, speech processing, voice analysis."""

    CODE_REVIEW = "code_review"
    """Code review, PR analysis, quality assessment, security audit."""

    DEPLOYMENT = "deployment"
    """CI/CD, infrastructure, deployment scripts, configuration."""

    DATA_ANALYSIS = "data_analysis"
    """Data processing, analytics, statistical analysis, visualization."""

    UNKNOWN = "unknown"
    """Could not classify the task."""


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """
    Result of a routing decision.

    Attributes
    ----------
    model_name:
        Selected model name.
    provider:
        Model provider.
    task_category:
        Classified task category.
    reason:
        Human-readable explanation of the decision.
    estimated_cost:
        Estimated cost in USD.
    estimated_latency_ms:
        Estimated response latency in milliseconds.
    confidence:
        Confidence score (0.0–1.0) in the routing decision.
    alternatives:
        Alternative models that were considered.
    """

    model_name: str
    provider: str = ""
    task_category: TaskCategory = TaskCategory.UNKNOWN
    reason: str = ""
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0
    confidence: float = 0.0
    alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "task_category": self.task_category.value,
            "reason": self.reason,
            "estimated_cost": round(self.estimated_cost, 8),
            "estimated_latency_ms": round(self.estimated_latency_ms, 1),
            "confidence": round(self.confidence, 4),
            "alternatives": self.alternatives,
        }


@dataclass
class RoutingConstraint:
    """Constraints for the routing decision."""

    max_cost: Optional[float] = None
    max_latency_ms: Optional[float] = None
    prefer_local: bool = False
    prefer_cheap: bool = False
    required_capabilities: Optional[Set[ModelCapability]] = None
    excluded_models: List[str] = field(default_factory=list)
    preferred_models: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Task classification patterns
# ──────────────────────────────────────────────────────────────────────────────

_REASONING_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(analyze|analyse|reason|think|consider|evaluate|compare)\b", re.I),
    re.compile(r"\b(trade[- ]off|architecture|design\s+decision|strategy)\b", re.I),
    re.compile(r"what\s+(would|should|could|will)\s+.*\s+(happen|if|be)", re.I),
    re.compile(r"\b(explain|why|how\s+does|what\s+causes?)\b", re.I),
]

_CODE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(write|create|generate|implement|build|add|fix|debug|refactor)\b.*\b(code|function|class|module|script|handler|endpoint|api|bug|error)\b", re.I),
    re.compile(r"\b(how\s+to|show\s+me|give\s+me)\s+.*\bcode\b", re.I),
    re.compile(r"```[\w]*\n", re.MULTILINE),
    re.compile(r"\b(coding|programming|developer|software)\b", re.I),
]

_CHAT_PATTERNS: List[re.Pattern] = [
    re.compile(r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|thanks|thank\s+you)\b", re.I),
    re.compile(r"^(yes|no|ok|sure|maybe|fine)\b", re.I),
]

_SUMMARIZATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(summarize|summarise|summary|tldr|tl;dr|recap|overview|condense)\b", re.I),
    re.compile(r"\b(key\s+points|main\s+ideas|in\s+short|bottom\s+line)\b", re.I),
]

_CREATIVE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(write|create|compose|draft|craft)\b.*\b(story|poem|article|essay|blog|email|letter|song|script)\b", re.I),
    re.compile(r"\b(brainstorm|idea|creative|imagine|fictional|narrative)\b", re.I),
]

_CLASSIFICATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(classify|categorize|label|sentiment|detect|identify)\b", re.I),
    re.compile(r"\b(what\s+(type|kind|category|class)\s+(is|of|are))\b", re.I),
]

_TRANSLATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(translate|translation|localize|convert\s+(to|from))\b", re.I),
    re.compile(r"\b(in\s+(french|spanish|german|japanese|chinese|korean|italian|portuguese))\b", re.I),
]

_VISION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(describe|analyze|look at|what\s+(is|do\s+you\s+see|are))\s+.*(image|picture|photo|screenshot|diagram|chart|figure|graph|plot)\b", re.I),
    re.compile(r"\b(image|screenshot|photo|picture|visual|diagram)\b.*\b(analyze|describe|explain|read|interpret|what)\b", re.I),
    re.compile(r"\b(ocr|extract\s+text|read\s+(this|the))\b.*\b(image|photo|screenshot|document)\b", re.I),
    re.compile(r"\b(vision|visual\s+understanding|object\s+detection|image\s+classification)\b", re.I),
]

_AUDIO_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(transcribe|transcription|speech[- ]to[- ]text|stt|voice[- ]to[- ]text)\b", re.I),
    re.compile(r"\b(audio|voice|recording|podcast|speech)\b.*\b(transcribe|analyze|process|convert)\b", re.I),
    re.compile(r"\b(process|analyze|summarize)\s+.*(audio|voice|recording|podcast|speech)\b", re.I),
    re.compile(r"\b(speaker|diarization|speech\s+recognition|audio\s+processing)\b", re.I),
]

_CODE_REVIEW_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(review|code\s*review|pr\s*review|pull\s*request)\b.*\b(code|changes|diff|patch|commit)\b", re.I),
    re.compile(r"\b(critique|assess|evaluate|audit)\s+.*(code|pull\s+request|pr|diff|merge\s+request)\b", re.I),
    re.compile(r"\b(code\s+quality|best\s+practice|anti[- ]pattern|smell|lint|static\s+analysis)\b", re.I),
    re.compile(r"\b(security\s+(audit|review|scan|check|vulnerability))\b.*\b(code|source|repository)\b", re.I),
]

_DEPLOYMENT_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(deploy|deployment|deploying|rollout|ship|release)\b", re.I),
    re.compile(r"\b(docker|kubernetes|k8s|terraform|ansible|cloudformation|pulumi)\b", re.I),
    re.compile(r"\b(ci/cd|cicd|pipeline|github\s+actions|gitlab\s+ci|jenkins|build\s+pipeline)\b", re.I),
    re.compile(r"\b(infra|infrastructure|provision|configure\s+(server|environment|cluster))\b", re.I),
    re.compile(r"\b(dockerfile|docker[- ]compose|helm|kustomize|yaml|configmap)\b.*\b(deploy|setup|create|write|build|generate)\b", re.I),
    re.compile(r"\b(create|write|build|generate|make)\b.*\b(dockerfile|docker[- ]compose|helm|kustomize)\b", re.I),
]

_DATA_ANALYSIS_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(analyze|analysis|analytics|explore|investigate)\s+.*(data|dataset|csv|json|database|table|spreadsheet)\b", re.I),
    re.compile(r"\b(statistic|correlation|regression|distribution|outlier|trend|forecast|predict)\b", re.I),
    re.compile(r"\b(data\s+(cleaning|wrangling|transformation|pipeline|processing|visualization))\b", re.I),
    re.compile(r"\b(pandas|sql|query|aggregate|pivot|group\s+by|join|merge)\b.*\b(data|table|dataset)\b", re.I),
    re.compile(r"\b(chart|graph|plot|visualization|dashboard|report)\b.*\b(data|create|generate|build)\b", re.I),
]


# ──────────────────────────────────────────────────────────────────────────────
# Model recommendations per category
# ──────────────────────────────────────────────────────────────────────────────

# Preferred models per task category (ordered by preference)
_CATEGORY_MODEL_MAP: Dict[TaskCategory, List[str]] = {
    TaskCategory.REASONING: [
        "claude-opus-4-20250514",
        "o1",
        "claude-sonnet-4-20250514",
        "gemini-2.0-pro",
    ],
    TaskCategory.CODE: [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "deepseek-chat",
        "codestral",
    ],
    TaskCategory.CHAT: [
        "claude-3-5-haiku-20241022",
        "gpt-4o-mini",
        "claude-sonnet-4-20250514",
        "deepseek-chat",
    ],
    TaskCategory.SUMMARIZATION: [
        "claude-3-5-haiku-20241022",
        "gpt-4o-mini",
        "gemini-2.0-flash",
    ],
    TaskCategory.CREATIVE: [
        "claude-sonnet-4-20250514",
        "gpt-4o",
        "claude-opus-4-20250514",
    ],
    TaskCategory.CLASSIFICATION: [
        "claude-3-5-haiku-20241022",
        "gpt-4o-mini",
    ],
    TaskCategory.TRANSLATION: [
        "gpt-4o",
        "claude-sonnet-4-20250514",
        "gemini-2.0-flash",
    ],
    TaskCategory.VISION: [
        "claude-sonnet-4-20250514",
        "gpt-4o",
        "gemini-2.5-pro",
        "claude-opus-4-20250514",
    ],
    TaskCategory.AUDIO: [
        "gpt-4o",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
    ],
    TaskCategory.CODE_REVIEW: [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "deepseek-chat",
        "codestral",
    ],
    TaskCategory.DEPLOYMENT: [
        "claude-sonnet-4-20250514",
        "gpt-4o",
        "deepseek-chat",
        "claude-3-5-haiku-20241022",
    ],
    TaskCategory.DATA_ANALYSIS: [
        "claude-sonnet-4-20250514",
        "gpt-4o",
        "gemini-2.5-pro",
        "deepseek-chat",
    ],
}

# Expected average latency per model category (ms)
_MODEL_LATENCY: Dict[str, float] = {
    # Anthropic
    "claude-opus-4-20250514": 3000.0,
    "claude-sonnet-4-20250514": 2000.0,
    "claude-3-5-haiku-20241022": 500.0,
    # OpenAI
    "gpt-4o": 1500.0,
    "gpt-4o-mini": 400.0,
    "o1": 5000.0,
    "o3-mini": 1500.0,
    # Google
    "deepseek-chat": 800.0,
    "gemini-2.0-flash": 300.0,
    "gemini-2.0-pro": 2500.0,
    "gemini-2.5-pro": 2800.0,
    "gemini-2.5-flash": 350.0,
    # xAI
    "grok-3": 2200.0,
    "grok-3-mini": 400.0,
    # Mistral
    "mistral-large": 1500.0,
    "mistral-medium": 1200.0,
    "mistral-small": 350.0,
    "codestral": 500.0,
    # DeepSeek
    "deepseek-coder": 600.0,
    "deepseek-reasoner": 3000.0,
    # Groq (ultra-fast LPU inference)
    "llama-3.3-70b-groq": 150.0,
    "mixtral-8x7b-groq": 100.0,
    # Together AI
    "llama-3.3-70b-together": 700.0,
    "qwen-2.5-72b-together": 750.0,
    # Fireworks AI
    "llama-3.3-70b-fireworks": 600.0,
    "mixtral-8x7b-fireworks": 250.0,
    # OpenRouter
    "openrouter-auto": 2000.0,
    # Amazon Bedrock
    "claude-via-bedrock": 2200.0,
    # NVIDIA NIM
    "llama-3.1-nemotron": 650.0,
    # Perplexity
    "sonar": 800.0,
    "sonar-pro": 1200.0,
    # Venice
    "venice-llama": 1000.0,
    # MiniMax
    "minimax-text": 600.0,
    # Local models
    "vllm-custom": 400.0,
    "sglang-custom": 350.0,
    # LiteLLM proxy (depends on backend)
    "litellm-proxy": 2000.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# SmartRouter
# ──────────────────────────────────────────────────────────────────────────────

class SmartRouter:
    """
    Task-aware, cost-aware model router.

    Classifies tasks, evaluates model candidates based on capabilities,
    cost, and latency constraints, and returns the best routing decision.

    Parameters
    ----------
    model_metadata:
        Optional ``ModelMetadata`` instance. If ``None``, creates a default.
    default_model:
        Fallback model when routing fails.
    cost_weight:
        Weight for cost in scoring (0.0–1.0).
    latency_weight:
        Weight for latency in scoring (0.0–1.0).
    quality_weight:
        Weight for quality in scoring (0.0–1.0).
    """

    def __init__(
        self,
        model_metadata: Optional[ModelMetadata] = None,
        default_model: str = "claude-sonnet-4-20250514",
        cost_weight: float = ROUTING_COST_WEIGHT,
        latency_weight: float = ROUTING_LATENCY_WEIGHT,
        quality_weight: float = ROUTING_QUALITY_WEIGHT,
    ) -> None:
        self._metadata = model_metadata or ModelMetadata()
        self._default_model = default_model
        self._cost_weight = cost_weight
        self._latency_weight = latency_weight
        self._quality_weight = quality_weight

        # Latency history for adaptive routing
        self._latency_history: Dict[str, List[float]] = {}

    # ── Public API ────────────────────────────────────────────────────────

    async def route(
        self,
        task_type: str = "",
        prompt: str = "",
        constraints: Optional[RoutingConstraint] = None,
        context_tokens: int = 0,
    ) -> RoutingDecision:
        """
        Route a request to the best model.

        Parameters
        ----------
        task_type:
            Explicit task type string (e.g., ``"reasoning"``, ``"code"``).
            If empty, auto-classifies from the prompt.
        prompt:
            The user prompt for classification (used if task_type is empty).
        constraints:
            Optional routing constraints.
        context_tokens:
            Estimated context token count for the request.

        Returns
        -------
        RoutingDecision
            The routing decision with model name, reason, and estimates.
        """
        constraint = constraints or RoutingConstraint()

        # Classify task
        category = self._classify_task(task_type, prompt)

        # Get candidate models for this category
        candidates = self._get_candidates(category, constraint)

        if not candidates:
            # Fallback
            return RoutingDecision(
                model_name=self._default_model,
                task_category=category,
                reason="No candidates matched constraints; using default model.",
                confidence=0.3,
            )

        # Score candidates
        scored = self._score_candidates(
            candidates, category, constraint, context_tokens,
        )

        best_name, best_score, best_reason = scored[0]
        alternatives = [name for name, _, _ in scored[1:4]]

        # Look up model info for estimates
        info = get_model_info(best_name)
        estimated_cost = self._estimate_cost(info, context_tokens)
        estimated_latency = self._estimate_latency(best_name)

        return RoutingDecision(
            model_name=best_name,
            provider=info.provider.value if info else "unknown",
            task_category=category,
            reason=best_reason,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            confidence=min(1.0, best_score / 100.0),
            alternatives=alternatives,
        )

    def classify(self, prompt: str) -> TaskCategory:
        """
        Classify a prompt into a task category.

        Parameters
        ----------
        prompt:
            The user prompt text.

        Returns
        -------
        TaskCategory
            The classified task category.
        """
        return self._classify_task("", prompt)

    def record_latency(self, model_name: str, latency_ms: float) -> None:
        """
        Record actual latency for a model (for adaptive routing).

        Parameters
        ----------
        model_name:
            The model that was used.
        latency_ms:
            Actual response latency in milliseconds.
        """
        if model_name not in self._latency_history:
            self._latency_history[model_name] = []
        history = self._latency_history[model_name]
        history.append(latency_ms)
        # Keep last 100 measurements
        if len(history) > 100:
            self._latency_history[model_name] = history[-100:]

    # ── Task classification ───────────────────────────────────────────────

    def _classify_task(self, task_type: str, prompt: str) -> TaskCategory:
        """Classify a task into a category."""
        # Explicit task type
        if task_type:
            try:
                return TaskCategory(task_type.lower())
            except ValueError:
                pass

        if not prompt or not prompt.strip():
            return TaskCategory.CHAT

        text = prompt.strip()

        # Check each category in priority order
        category_checks = [
            (TaskCategory.VISION, _VISION_PATTERNS),
            (TaskCategory.AUDIO, _AUDIO_PATTERNS),
            (TaskCategory.CODE_REVIEW, _CODE_REVIEW_PATTERNS),
            (TaskCategory.DEPLOYMENT, _DEPLOYMENT_PATTERNS),
            (TaskCategory.DATA_ANALYSIS, _DATA_ANALYSIS_PATTERNS),
            (TaskCategory.REASONING, _REASONING_PATTERNS),
            (TaskCategory.CODE, _CODE_PATTERNS),
            (TaskCategory.SUMMARIZATION, _SUMMARIZATION_PATTERNS),
            (TaskCategory.TRANSLATION, _TRANSLATION_PATTERNS),
            (TaskCategory.CLASSIFICATION, _CLASSIFICATION_PATTERNS),
            (TaskCategory.CREATIVE, _CREATIVE_PATTERNS),
        ]

        for category, patterns in category_checks:
            for pattern in patterns:
                if pattern.search(text):
                    return category

        # Check for chat (very short messages)
        for pattern in _CHAT_PATTERNS:
            if pattern.search(text):
                return TaskCategory.CHAT

        if len(text) < 30:
            return TaskCategory.CHAT

        return TaskCategory.UNKNOWN

    # ── Candidate selection ───────────────────────────────────────────────

    def _get_candidates(
        self,
        category: TaskCategory,
        constraints: RoutingConstraint,
    ) -> List[ModelInfo]:
        """Get candidate models that match the constraints."""
        # Start with category-preferred models
        preferred_names = _CATEGORY_MODEL_MAP.get(category, [])
        all_models = self._metadata.list()

        # Build candidate list (preferred first, then all others)
        candidates: List[ModelInfo] = []
        seen: set = set()

        for name in preferred_names:
            info = self._metadata.get(name)
            if info and name not in constraints.excluded_models:
                candidates.append(info)
                seen.add(name)

        for info in all_models:
            if info.name not in seen and info.name not in constraints.excluded_models:
                # Check required capabilities
                if constraints.required_capabilities:
                    if not constraints.required_capabilities.issubset(info.capabilities):
                        continue
                # Check context window (simplified)
                candidates.append(info)
                seen.add(info.name)

        # Apply constraints
        filtered: List[ModelInfo] = []
        for candidate in candidates:
            if constraints.prefer_local and not candidate.is_local:
                continue
            filtered.append(candidate)

        return filtered if filtered else candidates

    # ── Scoring ───────────────────────────────────────────────────────────

    def _score_candidates(
        self,
        candidates: List[ModelInfo],
        category: TaskCategory,
        constraints: RoutingConstraint,
        context_tokens: int,
    ) -> List[Tuple[str, float, str]]:
        """
        Score and rank candidates.

        Returns a list of (model_name, score, reason) tuples, sorted by score.
        """
        preferred_names = set(_CATEGORY_MODEL_MAP.get(category, []))

        scored: List[Tuple[str, float, str]] = []
        for candidate in candidates:
            score = 0.0
            reasons: List[str] = []

            # Quality score: bonus for preferred models
            if candidate.name in preferred_names:
                quality_score = 80.0
                idx = list(preferred_names).index(candidate.name)
                quality_score -= idx * 10  # Lower rank = higher score
                reasons.append(f"Preferred for {category.value}")
            else:
                quality_score = 40.0

            # Capability match
            if ModelCapability.RELIABLE in candidate.capabilities:
                quality_score += 10
            if ModelCapability.FAST in candidate.capabilities and category in (
                TaskCategory.CHAT, TaskCategory.SUMMARIZATION, TaskCategory.CLASSIFICATION,
            ):
                quality_score += 15

            # Cost score: lower cost = higher score
            cost_estimate = candidate.cost_per_million_input + candidate.cost_per_million_output
            if cost_estimate == 0:
                cost_score = 100.0
                reasons.append("Free (local)")
            else:
                cost_score = max(0, 100 - cost_estimate * 2)
            if constraints.prefer_cheap:
                cost_score *= 1.5

            # Check max cost constraint
            if constraints.max_cost is not None and cost_estimate > 0:
                expected_output = 500  # Rough estimate
                est_cost = (context_tokens * candidate.cost_per_million_input / 1_000_000) + \
                           (expected_output * candidate.cost_per_million_output / 1_000_000)
                if est_cost > constraints.max_cost:
                    cost_score *= 0.1
                    reasons.append(f"May exceed budget (${est_cost:.4f})")

            # Latency score: lower latency = higher score
            latency = self._estimate_latency(candidate.name)
            if latency <= 500:
                latency_score = 100.0
            elif latency <= 1500:
                latency_score = 75.0
            elif latency <= 3000:
                latency_score = 50.0
            elif latency <= 5000:
                latency_score = 25.0
            else:
                latency_score = 10.0

            # Check max latency constraint
            if constraints.max_latency_ms is not None:
                if latency > constraints.max_latency_ms:
                    latency_score *= 0.1
                    reasons.append(f"Exceeds latency constraint ({latency:.0f}ms)")

            # Weighted combination
            total_score = (
                quality_score * self._quality_weight
                + cost_score * self._cost_weight
                + latency_score * self._latency_weight
            )

            reason_str = "; ".join(reasons) if reasons else "General candidate"
            scored.append((candidate.name, total_score, reason_str))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── Estimation helpers ────────────────────────────────────────────────

    @staticmethod
    def _estimate_cost(model_info: Optional[ModelInfo], context_tokens: int) -> float:
        """Estimate cost for a request."""
        if model_info is None or model_info.is_free:
            return 0.0
        expected_output = 500
        return (
            (context_tokens * model_info.cost_per_million_input / 1_000_000)
            + (expected_output * model_info.cost_per_million_output / 1_000_000)
        )

    def _estimate_latency(self, model_name: str) -> float:
        """Estimate response latency for a model."""
        # Use adaptive history if available
        if model_name in self._latency_history:
            history = self._latency_history[model_name]
            if history:
                return sum(history) / len(history)

        return _MODEL_LATENCY.get(model_name, 1500.0)
