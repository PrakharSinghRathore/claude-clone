"""
Multi-Model Router with local model support.

Provides intelligent routing of tasks to the best AI model based on capabilities,
cost, speed, health, and task requirements. Supports Anthropic, OpenAI, Ollama,
llama.cpp, and custom providers with auto-failover, health monitoring, cost tracking,
rate limiting, and model comparison.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncGenerator, Callable, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelProvider(Enum):
    """Supported model providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    CUSTOM = "custom"


class ModelCapability(Enum):
    """Capabilities a model may advertise."""
    CHAT = "chat"
    CODE_COMPLETION = "code_completion"
    REASONING = "reasoning"
    VISION = "vision"
    AUDIO = "audio"
    FAST_RESPONSE = "fast_response"
    CHEAP = "cheap"
    RELIABLE = "reliable"
    LOCAL = "local"


class TaskType(Enum):
    """Recognised task types for routing."""
    AUTOCOMPLETE = "autocomplete"
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    DEBUG = "debug"
    REFACTOR = "refactor"
    SUMMARIZE = "summarize"
    EXPLAIN = "explain"
    QUICK_ANSWER = "quick_answer"
    COMPLEX_ANALYSIS = "complex_analysis"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Full configuration for a registered model."""
    name: str
    provider: ModelProvider
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 4096
    context_window: int = 8192
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    capabilities: set[ModelCapability] = field(default_factory=set)
    priority: int = 0
    max_rpm: int = 60
    enabled: bool = True

    # Optional human-friendly metadata
    display_name: str = ""
    description: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name
        if not self.base_url:
            self.base_url = _DEFAULT_BASE_URLS.get(self.provider, "")


@dataclass
class ModelHealth:
    """Real-time health statistics for a model."""
    model_name: str
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    tokens_per_second: float = 0.0
    last_check: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    is_healthy: bool = True
    consecutive_errors: int = 0
    _response_times: list[float] = field(default_factory=list, repr=False)
    _max_samples: int = field(default=100, repr=False)

    # --- helpers ----------------------------------------------------------

    def record(self, response_time_ms: float, tokens: int, success: bool) -> None:
        """Record a single request outcome."""
        self.total_requests += 1
        self.last_check = time.time()

        if success:
            self.consecutive_errors = 0
            self._response_times.append(response_time_ms)
            if len(self._response_times) > self._max_samples:
                self._response_times = self._response_times[-self._max_samples:]
            self.avg_response_time_ms = (
                sum(self._response_times) / len(self._response_times)
            )
            if response_time_ms > 0:
                self.tokens_per_second = (tokens / response_time_ms) * 1000.0
        else:
            self.total_errors += 1
            self.consecutive_errors += 1

        self.error_rate = (
            self.total_errors / self.total_requests if self.total_requests else 0.0
        )
        self.is_healthy = self.error_rate < 0.5 and self.consecutive_errors < 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "last_check": self.last_check,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "is_healthy": self.is_healthy,
            "consecutive_errors": self.consecutive_errors,
        }


@dataclass
class RouteDecision:
    """Result of the routing decision process."""
    model_name: str
    provider: ModelProvider
    reason: str
    estimated_cost: float
    estimated_time_ms: float
    score: float = 0.0


@dataclass
class ModelResponse:
    """Standardised response from any model."""
    model_name: str
    content: str
    tokens_input: int = 0
    tokens_output: int = 0
    response_time_ms: float = 0.0
    cost: float = 0.0
    cached: bool = False
    error: Optional[str] = None
    finish_reason: str = ""
    raw_metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "content": self.content,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "response_time_ms": round(self.response_time_ms, 2),
            "cost": round(self.cost, 8),
            "cached": self.cached,
            "error": self.error,
            "finish_reason": self.finish_reason,
        }


@dataclass
class ComparisonResult:
    """Result of running the same prompt on multiple models."""
    prompt: str
    responses: list[ModelResponse]
    winner: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cost & rate-limit tracking helpers
# ---------------------------------------------------------------------------

@dataclass
class _CostEntry:
    model_name: str
    session_id: str
    task_type: str
    tokens_input: int
    tokens_output: int
    cost: float
    timestamp: float


@dataclass
class _RateLimitState:
    timestamps: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default base URLs
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URLS: dict[ModelProvider, str] = {
    ModelProvider.ANTHROPIC: "https://api.anthropic.com",
    ModelProvider.OPENAI: "https://api.openai.com/v1",
    ModelProvider.OPENROUTER: "https://openrouter.ai/api/v1",
    ModelProvider.OLLAMA: "http://localhost:11434",
    ModelProvider.LLAMACPP: "http://localhost:8080",
    ModelProvider.CUSTOM: "",
}


# ---------------------------------------------------------------------------
# Built-in model catalogue
# ---------------------------------------------------------------------------

_BUILTIN_MODELS: list[dict[str, Any]] = [
    {
        "name": "claude-opus-4",
        "display_name": "Claude Opus 4",
        "provider": ModelProvider.ANTHROPIC,
        "max_tokens": 16384,
        "context_window": 200000,
        "cost_per_input_token": 0.000015,
        "cost_per_output_token": 0.000075,
        "capabilities": {
            ModelCapability.CHAT,
            ModelCapability.CODE_COMPLETION,
            ModelCapability.REASONING,
            ModelCapability.VISION,
            ModelCapability.RELIABLE,
        },
        "priority": 100,
        "max_rpm": 50,
        "description": "Most capable Claude model — best for complex reasoning, analysis, and code generation.",
    },
    {
        "name": "claude-sonnet-4",
        "display_name": "Claude Sonnet 4",
        "provider": ModelProvider.ANTHROPIC,
        "max_tokens": 16384,
        "context_window": 200000,
        "cost_per_input_token": 0.000003,
        "cost_per_output_token": 0.000015,
        "capabilities": {
            ModelCapability.CHAT,
            ModelCapability.CODE_COMPLETION,
            ModelCapability.REASONING,
            ModelCapability.VISION,
            ModelCapability.RELIABLE,
            ModelCapability.FAST_RESPONSE,
        },
        "priority": 90,
        "max_rpm": 60,
        "description": "Balanced Claude model — great code generation and reasoning at moderate cost.",
    },
    {
        "name": "claude-haiku",
        "display_name": "Claude Haiku",
        "provider": ModelProvider.ANTHROPIC,
        "max_tokens": 8192,
        "context_window": 200000,
        "cost_per_input_token": 0.0000008,
        "cost_per_output_token": 0.000004,
        "capabilities": {
            ModelCapability.CHAT,
            ModelCapability.CODE_COMPLETION,
            ModelCapability.FAST_RESPONSE,
            ModelCapability.CHEAP,
        },
        "priority": 80,
        "max_rpm": 100,
        "description": "Fast and affordable Claude — ideal for autocomplete, quick answers, and chat.",
    },
    {
        "name": "gpt-4o",
        "display_name": "GPT-4o",
        "provider": ModelProvider.OPENAI,
        "max_tokens": 16384,
        "context_window": 128000,
        "cost_per_input_token": 0.0000025,
        "cost_per_output_token": 0.00001,
        "capabilities": {
            ModelCapability.CHAT,
            ModelCapability.CODE_COMPLETION,
            ModelCapability.REASONING,
            ModelCapability.VISION,
            ModelCapability.AUDIO,
            ModelCapability.RELIABLE,
        },
        "priority": 85,
        "max_rpm": 60,
        "description": "OpenAI flagship — strong all-rounder with vision and audio support.",
    },
    {
        "name": "gpt-4o-mini",
        "display_name": "GPT-4o Mini",
        "provider": ModelProvider.OPENAI,
        "max_tokens": 16384,
        "context_window": 128000,
        "cost_per_input_token": 0.00000015,
        "cost_per_output_token": 0.0000006,
        "capabilities": {
            ModelCapability.CHAT,
            ModelCapability.CODE_COMPLETION,
            ModelCapability.FAST_RESPONSE,
            ModelCapability.CHEAP,
        },
        "priority": 75,
        "max_rpm": 100,
        "description": "Tiny, cheap, fast — good for quick answers and simple tasks.",
    },
    {
        "name": "deepseek-coder",
        "display_name": "DeepSeek Coder",
        "provider": ModelProvider.OLLAMA,
        "base_url": "http://localhost:11434",
        "max_tokens": 8192,
        "context_window": 32768,
        "cost_per_input_token": 0.0,
        "cost_per_output_token": 0.0,
        "capabilities": {
            ModelCapability.CHAT,
            ModelCapability.CODE_COMPLETION,
            ModelCapability.LOCAL,
            ModelCapability.CHEAP,
        },
        "priority": 50,
        "max_rpm": 30,
        "description": "Free local model via Ollama — code-specialised, no API key needed.",
    },
    {
        "name": "llama3",
        "display_name": "Llama 3",
        "provider": ModelProvider.OLLAMA,
        "base_url": "http://localhost:11434",
        "max_tokens": 8192,
        "context_window": 8192,
        "cost_per_input_token": 0.0,
        "cost_per_output_token": 0.0,
        "capabilities": {
            ModelCapability.CHAT,
            ModelCapability.REASONING,
            ModelCapability.LOCAL,
            ModelCapability.CHEAP,
        },
        "priority": 45,
        "max_rpm": 30,
        "description": "Free local model via Ollama — general-purpose, no API key needed.",
    },
    {
        "name": "mistral",
        "display_name": "Mistral",
        "provider": ModelProvider.OLLAMA,
        "base_url": "http://localhost:11434",
        "max_tokens": 8192,
        "context_window": 32768,
        "cost_per_input_token": 0.0,
        "cost_per_output_token": 0.0,
        "capabilities": {
            ModelCapability.CHAT,
            ModelCapability.CODE_COMPLETION,
            ModelCapability.LOCAL,
            ModelCapability.CHEAP,
            ModelCapability.FAST_RESPONSE,
        },
        "priority": 48,
        "max_rpm": 30,
        "description": "Free local model via Ollama — fast and efficient, no API key needed.",
    },
]


# ---------------------------------------------------------------------------
# Task → required capabilities mapping
# ---------------------------------------------------------------------------

_TASK_CAPABILITY_WEIGHTS: dict[str, dict[ModelCapability, float]] = {
    TaskType.AUTOCOMPLETE.value: {
        ModelCapability.FAST_RESPONSE: 3.0,
        ModelCapability.CODE_COMPLETION: 2.5,
        ModelCapability.CHEAP: 1.5,
    },
    TaskType.CHAT.value: {
        ModelCapability.CHAT: 2.5,
        ModelCapability.FAST_RESPONSE: 1.5,
        ModelCapability.CHEAP: 1.0,
    },
    TaskType.CODE_GENERATION.value: {
        ModelCapability.CODE_COMPLETION: 3.0,
        ModelCapability.REASONING: 2.0,
        ModelCapability.RELIABLE: 1.5,
    },
    TaskType.REASONING.value: {
        ModelCapability.REASONING: 3.5,
        ModelCapability.RELIABLE: 2.0,
        ModelCapability.CHAT: 1.0,
    },
    TaskType.DEBUG.value: {
        ModelCapability.REASONING: 2.5,
        ModelCapability.CODE_COMPLETION: 2.0,
        ModelCapability.CHAT: 1.5,
    },
    TaskType.REFACTOR.value: {
        ModelCapability.CODE_COMPLETION: 3.0,
        ModelCapability.REASONING: 2.0,
        ModelCapability.RELIABLE: 1.0,
    },
    TaskType.SUMMARIZE.value: {
        ModelCapability.CHAT: 2.0,
        ModelCapability.CHEAP: 1.5,
        ModelCapability.FAST_RESPONSE: 1.0,
    },
    TaskType.EXPLAIN.value: {
        ModelCapability.CHAT: 2.5,
        ModelCapability.REASONING: 1.5,
        ModelCapability.CHEAP: 1.0,
    },
    TaskType.QUICK_ANSWER.value: {
        ModelCapability.FAST_RESPONSE: 3.0,
        ModelCapability.CHEAP: 2.5,
        ModelCapability.CHAT: 1.0,
    },
    TaskType.COMPLEX_ANALYSIS.value: {
        ModelCapability.REASONING: 3.5,
        ModelCapability.RELIABLE: 2.5,
        ModelCapability.CHAT: 1.5,
    },
}

# Expected output length per task (rough token estimate for cost calculations).
_TASK_EXPECTED_OUTPUT_TOKENS: dict[str, int] = {
    TaskType.AUTOCOMPLETE.value: 50,
    TaskType.CHAT.value: 300,
    TaskType.CODE_GENERATION.value: 800,
    TaskType.REASONING.value: 1000,
    TaskType.DEBUG.value: 500,
    TaskType.REFACTOR.value: 600,
    TaskType.SUMMARIZE.value: 400,
    TaskType.EXPLAIN.value: 350,
    TaskType.QUICK_ANSWER.value: 100,
    TaskType.COMPLEX_ANALYSIS.value: 1200,
}

# ---------------------------------------------------------------------------
# Task classification heuristics
# ---------------------------------------------------------------------------

_AUTOCOMPLETE_PATTERNS: list[re.Pattern] = [
    re.compile(r"complete\s+(the\s+)?(code|function|line|snippet)", re.I),
    re.compile(r"fill\s+(in|out)\s+(the\s+)?(rest|missing)", re.I),
    re.compile(r"^(def |class |async def |import |from |func |pub fn |const |let |var )\S.*\s*$", re.MULTILINE),
]

_DEBUG_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(debug|fix|error|bug|traceback|exception|stack\s*trace)\b", re.I),
    re.compile(r"(IndexError|KeyError|ValueError|TypeError|NameError|AttributeError|RuntimeError|SyntaxError|ImportError|OSError|ZeroDivisionError|FileNotFoundError|StopIteration)", re.I),
    re.compile(r"\b(throws|raised|raising|causes?|results?\s+in)\s+(an?\s+)?(error|exception)", re.I),
    re.compile(r"why\s+(does|is|do|are)\s+", re.I),
    re.compile(r"(not\s+working|doesn'?t\s+work|broken|failing|fail)", re.I),
    re.compile(r"undefined|null\s*reference|type\s*error|syntax\s*error", re.I),
]

_REFACTOR_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(refactor|restructure|clean\s*up|simplify|optimise|optimize|improve)\b", re.I),
    re.compile(r"make\s+(this|it|the\s+code)\s+(cleaner|better|faster|more\s+readable)", re.I),
    re.compile(r"reduce\s+(complexity|duplication|cognitive\s+load)", re.I),
]

_CODEGEN_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(write|create|generate|implement|build|add)\b.*\b(code|function|class|module|script|handler|endpoint|api)\b", re.I),
    re.compile(r"\b(how\s+to|show\s+me|give\s+me)\s+.*\bcode\b", re.I),
    re.compile(r"^(def |class |async def |import |from |func |pub fn )", re.MULTILINE),
]

_REASONING_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(analyze|analyse|reason|think|consider|evaluate|compare)\b", re.I),
    re.compile(r"\b(trade[- ]off|architecture|design\s+decision|strategy|approach)\b", re.I),
    re.compile(r"what\s+(would|should|could|will)\s+.*\s+(happen|if|be)", re.I),
]

_SUMMARIZE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(summarize|summarise|summary|tldr|tl;dr|recap|overview)\b", re.I),
    re.compile(r"(key\s+points|main\s+ideas|in\s+short|bottom\s+line)", re.I),
]

_EXPLAIN_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(explain|what\s+does|how\s+does|what\s+is|meaning\s+of|describe)\b", re.I),
]

_QUICK_ANSWER_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(yes|no|true|false|\d+)$", re.I),
    re.compile(r"^(what|who|when|where)\s+(is|are|was|were)\s+\S", re.I),
    re.compile(r"^(\.\w+|ls |cd |pwd|echo )", re.I),
]


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter:
    """Intelligent multi-model router for AI completions.

    Features:
    * Automatic task classification and smart model selection.
    * Health monitoring with auto-failover.
    * Per-model and per-session cost tracking.
    * Rate limiting per model.
    * Ollama and llama.cpp local model support.
    * Side-by-side model comparison.
    * Streaming completions.
    """

    def __init__(
        self,
        default_provider: str = "anthropic",
        fallback_providers: list[str] | None = None,
        http_timeout: float = 120.0,
    ) -> None:
        self._default_provider = ModelProvider(default_provider.lower())
        self._fallback_providers: list[ModelProvider] = (
            [ModelProvider(p.lower()) for p in (fallback_providers or ["openai", "ollama"])]
        )
        self._http_timeout = http_timeout

        # Model registry  {name: ModelConfig}
        self._models: dict[str, ModelConfig] = {}
        # Health stats    {name: ModelHealth}
        self._health: dict[str, ModelHealth] = {}
        # Rate-limit state {name: _RateLimitState}
        self._rate_limits: dict[str, _RateLimitState] = {}
        # Cost ledger
        self._cost_entries: list[_CostEntry] = []
        # Session → model history (for mixing)
        self._session_models: dict[str, list[str]] = defaultdict(list)
        # Current session id
        self._current_session: str = ""

        self._initialized: bool = False
        self._ollama_available: bool = False
        self._llamacpp_available: bool = False

        # Lock for thread-safety inside the event loop
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load built-in models, probe local providers, and prepare state."""
        if self._initialized:
            logger.debug("ModelRouter already initialized; skipping.")
            return

        # Register built-in models
        for spec in _BUILTIN_MODELS:
            cfg = ModelConfig(
                name=spec["name"],
                display_name=spec.get("display_name", spec["name"]),
                provider=spec["provider"],
                base_url=spec.get("base_url", _DEFAULT_BASE_URLS.get(spec["provider"], "")),
                max_tokens=spec.get("max_tokens", 4096),
                context_window=spec.get("context_window", 8192),
                cost_per_input_token=spec.get("cost_per_input_token", 0.0),
                cost_per_output_token=spec.get("cost_per_output_token", 0.0),
                capabilities=spec.get("capabilities", set()),
                priority=spec.get("priority", 0),
                max_rpm=spec.get("max_rpm", 60),
                description=spec.get("description", ""),
            )
            self._models[cfg.name] = cfg
            self._health[cfg.name] = ModelHealth(model_name=cfg.name)
            self._rate_limits[cfg.name] = _RateLimitState()

        # Probe local providers
        await self._probe_ollama()
        await self._probe_llamacpp()

        # Disable local models if their provider is not available
        for name, cfg in self._models.items():
            if cfg.provider == ModelProvider.OLLAMA and not self._ollama_available:
                cfg.enabled = False
                logger.info("Ollama unavailable — disabled model %s", name)
            elif cfg.provider == ModelProvider.LLAMACPP and not self._llamacpp_available:
                cfg.enabled = False
                logger.info("llama.cpp unavailable — disabled model %s", name)
            else:
                logger.debug("Registered model %s (%s)", name, cfg.provider.value)

        self._initialized = True
        logger.info(
            "ModelRouter initialised — %d models registered, Ollama=%s, llama.cpp=%s",
            len(self._models),
            self._ollama_available,
            self._llamacpp_available,
        )

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    async def register_model(self, config: ModelConfig) -> None:
        """Register (or update) a model configuration."""
        async with self._lock:
            self._models[config.name] = config
            if config.name not in self._health:
                self._health[config.name] = ModelHealth(model_name=config.name)
            if config.name not in self._rate_limits:
                self._rate_limits[config.name] = _RateLimitState()
            logger.info("Registered model %s (%s)", config.name, config.provider.value)

    async def unregister_model(self, name: str) -> None:
        """Remove a model from the registry."""
        async with self._lock:
            self._models.pop(name, None)
            self._health.pop(name, None)
            self._rate_limits.pop(name, None)
            logger.info("Unregistered model %s", name)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def route(
        self,
        task_type: str,
        prompt: str,
        context_tokens: int | None = None,
        prefer_local: bool = False,
        max_cost: float | None = None,
    ) -> RouteDecision:
        """Pick the best model for the given *task_type* and *prompt*.

        Returns a :class:`RouteDecision` with model name, estimated cost/time,
        and a human-readable reason.
        """
        if not self._initialized:
            await self.initialize()

        if context_tokens is None:
            context_tokens = _estimate_tokens(prompt)

        # Clamp task_type to known values
        task_key = task_type.lower()
        if task_key not in _TASK_CAPABILITY_WEIGHTS:
            # Try to classify automatically
            task_key = self._classify_task(prompt)

        candidates: list[tuple[float, ModelConfig]] = []
        for name, cfg in self._models.items():
            if not cfg.enabled:
                continue
            health = self._health.get(name)
            if health and not health.is_healthy:
                continue
            if cfg.context_window < context_tokens:
                continue
            if prefer_local and ModelCapability.LOCAL not in cfg.capabilities:
                continue
            if max_cost is not None:
                expected_output = _TASK_EXPECTED_OUTPUT_TOKENS.get(task_key, 500)
                est_cost = (
                    context_tokens * cfg.cost_per_input_token
                    + expected_output * cfg.cost_per_output_token
                )
                if est_cost > max_cost:
                    continue

            score = self._score_model(cfg, task_key, context_tokens)
            candidates.append((score, cfg))

        if not candidates:
            # Fall back to any enabled model ignoring health/cost
            for name, cfg in self._models.items():
                if cfg.enabled:
                    candidates.append((0.0, cfg))
            if not candidates:
                raise RuntimeError("No models available for routing")

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_cfg = candidates[0]

        expected_output = _TASK_EXPECTED_OUTPUT_TOKENS.get(task_key, 500)
        est_cost = (
            context_tokens * best_cfg.cost_per_input_token
            + expected_output * best_cfg.cost_per_output_token
        )
        health = self._health.get(best_cfg.name)
        est_time = health.avg_response_time_ms if health and health.total_requests > 0 else 500.0

        reason = self._build_routing_reason(best_cfg, task_key, best_score, prefer_local)

        return RouteDecision(
            model_name=best_cfg.name,
            provider=best_cfg.provider,
            reason=reason,
            estimated_cost=est_cost,
            estimated_time_ms=est_time,
            score=best_score,
        )

    # ------------------------------------------------------------------
    # Completions
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict],
        model_name: str | None = None,
        task_type: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Send a completion request to the specified (or auto-selected) model.

        Implements auto-failover: if the primary model fails, the router tries
        the next best candidate(s) automatically.
        """
        if not self._initialized:
            await self.initialize()

        sid = session_id or self._current_session
        if not sid:
            sid = str(uuid.uuid4())[:8]
            self._current_session = sid

        # Determine model
        if model_name is None:
            prompt = messages[-1].get("content", "") if messages else ""
            task = task_type or self._classify_task(prompt)
            decision = await self.route(task, prompt)
            model_name = decision.model_name
            task_type = task
        elif task_type is None:
            prompt = messages[-1].get("content", "") if messages else ""
            task_type = self._classify_task(prompt)

        cfg = self._models.get(model_name)
        if cfg is None or not cfg.enabled:
            raise ValueError(f"Model {model_name!r} not found or disabled")

        # Build ordered candidate list for failover
        prompt = messages[-1].get("content", "") if messages else ""
        task_key = task_type or self._classify_task(prompt)
        all_candidates = [
            (self._score_model(m, task_key, 0), m)
            for m in self._models.values()
            if m.enabled and m.name != model_name
        ]
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        fallback_names = [m.name for _, m in all_candidates if self._health.get(m.name, ModelHealth(m.name)).is_healthy]

        last_error: Optional[str] = None
        tried = [model_name] + fallback_names[:3]  # try up to 4 models

        for try_name in tried:
            try_cfg = self._models.get(try_name)
            if try_cfg is None:
                continue
            if not self._check_rate_limit(try_name):
                logger.warning("Rate-limit hit for %s, skipping", try_name)
                continue

            resp = await self._call_model(
                try_cfg, messages, max_tokens=max_tokens, temperature=temperature, **kwargs
            )

            if resp.error is None:
                # Record health, cost, session
                self._record_health(try_name, resp.response_time_ms, True, resp.tokens_input + resp.tokens_output)
                self._record_cost(try_name, sid, task_type or "unknown", resp.tokens_input, resp.tokens_output, resp.cost)
                self._session_models[sid].append(try_name)
                return resp
            else:
                last_error = resp.error
                self._record_health(try_name, resp.response_time_ms, False, 0)
                logger.warning("Model %s failed: %s — trying next fallback", try_name, resp.error)

        # All models failed
        return ModelResponse(
            model_name=model_name,
            content="",
            error=f"All models failed. Last error: {last_error}",
        )

    async def complete_with_routing(
        self,
        messages: list[dict],
        task_type: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Convenience: classify task, route, then complete in one call."""
        prompt = messages[-1].get("content", "") if messages else ""
        task = task_type or self._classify_task(prompt)
        return await self.complete(messages, task_type=task, **kwargs)

    async def stream(
        self,
        messages: list[dict],
        model_name: str | None = None,
        task_type: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Streaming completion — yields text chunks as they arrive.

        Uses the same routing logic as :meth:`complete` but streams the
        response body instead of buffering.
        """
        if not self._initialized:
            await self.initialize()

        if model_name is None:
            prompt = messages[-1].get("content", "") if messages else ""
            task = task_type or self._classify_task(prompt)
            decision = await self.route(task, prompt)
            model_name = decision.model_name

        cfg = self._models.get(model_name)
        if cfg is None or not cfg.enabled:
            raise ValueError(f"Model {model_name!r} not found or disabled")

        if not self._check_rate_limit(model_name):
            raise RuntimeError(f"Rate limit exceeded for model {model_name!r}")

        async for chunk in self._stream_model(cfg, messages, **kwargs):
            yield chunk

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    async def compare(
        self,
        prompt: str,
        models: list[str] | None = None,
        task_type: str | None = None,
        max_tokens: int | None = None,
    ) -> ComparisonResult:
        """Run the same prompt on multiple models and pick a winner.

        If *models* is ``None``, the top 3 models for *task_type* are chosen.
        The winner is selected based on a combination of output length,
        inferred quality signals, and response speed.
        """
        if not self._initialized:
            await self.initialize()

        messages = [{"role": "user", "content": prompt}]
        task = task_type or self._classify_task(prompt)

        if models is None:
            decision = await self.route(task, prompt)
            models = [decision.model_name]
            # Add alternatives
            for name, cfg in self._models.items():
                if cfg.enabled and cfg.name != decision.model_name:
                    models.append(name)
                if len(models) >= 3:
                    break

        responses: list[ModelResponse] = []
        for name in models:
            cfg = self._models.get(name)
            if cfg is None or not cfg.enabled:
                continue
            try:
                resp = await self._call_model(cfg, messages, max_tokens=max_tokens)
                responses.append(resp)
            except Exception as exc:
                logger.warning("compare: model %s failed — %s", name, exc)
                responses.append(
                    ModelResponse(model_name=name, content="", error=str(exc))
                )

        # Pick winner
        winner = self._pick_comparison_winner(responses, task)

        return ComparisonResult(
            prompt=prompt,
            responses=responses,
            winner=winner,
            metadata={
                "task_type": task,
                "num_models": len(responses),
                "successful": sum(1 for r in responses if r.error is None),
            },
        )

    # ------------------------------------------------------------------
    # Health & diagnostics
    # ------------------------------------------------------------------

    async def get_health(self, model_name: str | None = None) -> ModelHealth | list[ModelHealth]:
        """Return health info for one model or all registered models."""
        if model_name is not None:
            return self._health.get(model_name, ModelHealth(model_name=model_name))
        return list(self._health.values())

    async def check_ollama(self) -> list[dict]:
        """Query Ollama for a list of locally available models.

        Returns a list of dicts with keys: ``name``, ``size``, ``modified_at``,
        ``details``.
        """
        if not self._ollama_available:
            return []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("http://localhost:11434/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return data.get("models", [])
        except Exception as exc:
            logger.warning("Failed to query Ollama: %s", exc)
            return []

    async def get_model_info(self, name: str) -> ModelConfig:
        """Return the configuration of a registered model."""
        cfg = self._models.get(name)
        if cfg is None:
            raise KeyError(f"Model {name!r} not found")
        return cfg

    async def list_models(self) -> list[ModelConfig]:
        """Return all registered model configurations."""
        return list(self._models.values())

    async def get_cost_summary(self, session_id: str | None = None) -> dict:
        """Return cost breakdown aggregated by model, task type, and session."""
        entries = self._cost_entries
        if session_id is not None:
            entries = [e for e in entries if e.session_id == session_id]

        by_model: dict[str, float] = defaultdict(float)
        by_task: dict[str, float] = defaultdict(float)
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost = 0.0

        for e in entries:
            by_model[e.model_name] += e.cost
            by_task[e.task_type] += e.cost
            total_tokens_in += e.tokens_input
            total_tokens_out += e.tokens_output
            total_cost += e.cost

        return {
            "total_cost": round(total_cost, 8),
            "total_tokens_input": total_tokens_in,
            "total_tokens_output": total_tokens_out,
            "total_requests": len(entries),
            "by_model": {k: round(v, 8) for k, v in by_model.items()},
            "by_task_type": {k: round(v, 8) for k, v in by_task.items()},
            "session_id": session_id,
        }

    async def get_recommendation(
        self,
        task_type: str,
        constraints: dict[str, Any] | None = None,
    ) -> str:
        """Recommend the best model name for *task_type* given optional *constraints*.

        Supported constraint keys:
        * ``max_cost`` (float): maximum acceptable cost.
        * ``prefer_local`` (bool): prefer locally-hosted models.
        * ``context_tokens`` (int): required context window size.
        * ``max_latency_ms`` (float): maximum acceptable latency.
        """
        if not self._initialized:
            await self.initialize()

        constraints = constraints or {}
        prompt = constraints.get("prompt", "")

        decision = await self.route(
            task_type=task_type,
            prompt=prompt,
            context_tokens=constraints.get("context_tokens"),
            prefer_local=constraints.get("prefer_local", False),
            max_cost=constraints.get("max_cost"),
        )

        # Apply latency constraint if specified
        max_latency = constraints.get("max_latency_ms")
        if max_latency is not None:
            health = self._health.get(decision.model_name)
            if health and health.avg_response_time_ms > max_latency:
                # Search for a faster alternative
                for name, cfg in self._models.items():
                    if not cfg.enabled or name == decision.model_name:
                        continue
                    h = self._health.get(name)
                    if h and h.is_healthy and h.avg_response_time_ms <= max_latency:
                        if ModelCapability.LOCAL in cfg.capabilities or cfg.provider != ModelProvider.OLLAMA:
                            return name
                # No faster alternative — return original
                pass

        return decision.model_name

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def set_session(self, session_id: str) -> None:
        """Set the current session id for cost tracking and model mixing."""
        self._current_session = session_id

    def get_session_history(self, session_id: str | None = None) -> list[str]:
        """Return the list of model names used in a session (model mixing)."""
        sid = session_id or self._current_session
        return list(self._session_models.get(sid, []))

    # ------------------------------------------------------------------
    # Task classification
    # ------------------------------------------------------------------

    def _classify_task(self, prompt: str, context: dict | None = None) -> str:
        """Classify a prompt into a task type string.

        Uses pattern matching heuristics. Falls back to ``"chat"``.
        """
        if not prompt or not prompt.strip():
            return TaskType.QUICK_ANSWER.value

        text = prompt.strip()

        # Quick answer: very short, no-task-keyword prompts
        if len(text) < 30:
            has_code_symbols = bool(re.search(r"[{}();]", text))
            has_task_keyword = any(
                kw in text.lower()
                for kw in ("debug", "fix", "error", "bug", "refactor", "summarize",
                           "explain", "write", "create", "analyze", "code",
                           "optimize", "optimise", "improve", "simplify")
            )
            if not has_code_symbols and not has_task_keyword:
                return TaskType.QUICK_ANSWER.value

        # Check each category in priority order
        checks = [
            (TaskType.DEBUG.value, _DEBUG_PATTERNS),
            (TaskType.REFACTOR.value, _REFACTOR_PATTERNS),
            (TaskType.SUMMARIZE.value, _SUMMARIZE_PATTERNS),
            (TaskType.EXPLAIN.value, _EXPLAIN_PATTERNS),
            (TaskType.AUTOCOMPLETE.value, _AUTOCOMPLETE_PATTERNS),
            (TaskType.CODE_GENERATION.value, _CODEGEN_PATTERNS),
            (TaskType.REASONING.value, _REASONING_PATTERNS),
            (TaskType.QUICK_ANSWER.value, _QUICK_ANSWER_PATTERNS),
        ]

        for task_name, patterns in checks:
            for pat in patterns:
                if pat.search(text):
                    # For autocomplete, verify the prompt is short
                    if task_name == TaskType.AUTOCOMPLETE.value and len(text) > 200:
                        continue
                    return task_name

        # Use context hints if provided
        if context:
            ctx_task = context.get("task_type")
            if ctx_task and ctx_task in _TASK_CAPABILITY_WEIGHTS:
                return ctx_task

        return TaskType.CHAT.value

    # ------------------------------------------------------------------
    # Model scoring
    # ------------------------------------------------------------------

    def _score_model(
        self,
        model: ModelConfig,
        task: str,
        context_tokens: int,
    ) -> float:
        """Score a model's suitability for *task* on a 0-100 scale.

        Considers:
        * Capability match (primary signal).
        * Model priority (manual preference).
        * Health (error rate, response time).
        * Cost efficiency.
        * Context window fitness.
        """
        score = 0.0

        # 1. Capability match  (0 – 50 pts)
        cap_weights = _TASK_CAPABILITY_WEIGHTS.get(task, {})
        if cap_weights:
            max_possible = sum(cap_weights.values())
            matched = sum(
                w for cap, w in cap_weights.items()
                if cap in model.capabilities
            )
            score += (matched / max_possible * 50.0) if max_possible else 25.0
        else:
            # Unknown task — give base score for general capabilities
            general = sum(
                1 for c in (ModelCapability.CHAT, ModelCapability.REASONING)
                if c in model.capabilities
            )
            score += general * 15.0

        # 2. Priority  (0 – 15 pts)
        score += min(model.priority / 100.0 * 15.0, 15.0)

        # 3. Health  (0 – 20 pts)
        health = self._health.get(model.name)
        if health and health.total_requests > 0:
            health_score = max(0.0, 1.0 - health.error_rate) * 10.0
            # Penalise slow models for tasks that require speed
            if ModelCapability.FAST_RESPONSE in cap_weights or task in (
                TaskType.AUTOCOMPLETE.value,
                TaskType.QUICK_ANSWER.value,
            ):
                if health.avg_response_time_ms > 3000:
                    health_score *= 0.5
                elif health.avg_response_time_ms > 1000:
                    health_score *= 0.8
            score += min(health_score, 20.0)
        else:
            score += 10.0  # neutral for untested models

        # 4. Cost efficiency  (0 – 10 pts)
        expected_output = _TASK_EXPECTED_OUTPUT_TOKENS.get(task, 500)
        est_cost = (
            context_tokens * model.cost_per_input_token
            + expected_output * model.cost_per_output_token
        )
        if est_cost == 0:
            score += 10.0  # free models get full points
        elif est_cost < 0.001:
            score += 8.0
        elif est_cost < 0.01:
            score += 5.0
        elif est_cost < 0.1:
            score += 2.0
        # else 0 pts (expensive)

        # 5. Context window fitness  (0 – 5 pts)
        if model.context_window >= context_tokens * 2:
            score += 5.0
        elif model.context_window >= context_tokens:
            score += 3.0

        return round(score, 2)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self, model_name: str) -> bool:
        """Return ``True`` if the model is under its RPM limit."""
        state = self._rate_limits.get(model_name)
        cfg = self._models.get(model_name)
        if not state or not cfg:
            return True

        now = time.time()
        window = 60.0  # 1-minute rolling window
        # Prune old timestamps
        state.timestamps = [ts for ts in state.timestamps if now - ts < window]
        return len(state.timestamps) < cfg.max_rpm

    def _record_request(self, model_name: str) -> None:
        """Record a request timestamp for rate limiting."""
        state = self._rate_limits.get(model_name)
        if state:
            state.timestamps.append(time.time())

    # ------------------------------------------------------------------
    # Health recording
    # ------------------------------------------------------------------

    def _record_health(
        self,
        model_name: str,
        response_time: float,
        success: bool,
        tokens: int = 0,
    ) -> None:
        """Update health stats for a model after a request."""
        health = self._health.get(model_name)
        if health:
            health.record(response_time_ms=response_time, tokens=tokens, success=success)

    # ------------------------------------------------------------------
    # Cost recording
    # ------------------------------------------------------------------

    def _record_cost(
        self,
        model_name: str,
        session_id: str,
        task_type: str,
        tokens_input: int,
        tokens_output: int,
        cost: float,
    ) -> None:
        """Append a cost entry to the ledger."""
        self._cost_entries.append(
            _CostEntry(
                model_name=model_name,
                session_id=session_id,
                task_type=task_type,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost=cost,
                timestamp=time.time(),
            )
        )

    # ------------------------------------------------------------------
    # Provider-specific API calls
    # ------------------------------------------------------------------

    async def _call_model(
        self,
        cfg: ModelConfig,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Dispatch a completion request to the correct provider backend."""
        self._record_request(cfg.name)
        start = time.perf_counter()

        try:
            if cfg.provider == ModelProvider.ANTHROPIC:
                resp = await self._call_anthropic(cfg, messages, max_tokens, temperature, **kwargs)
            elif cfg.provider == ModelProvider.OPENAI:
                resp = await self._call_openai(cfg, messages, max_tokens, temperature, **kwargs)
            elif cfg.provider == ModelProvider.OPENROUTER:
                resp = await self._call_openrouter(cfg, messages, max_tokens, temperature, **kwargs)
            elif cfg.provider == ModelProvider.OLLAMA:
                resp = await self._call_ollama(cfg, messages, max_tokens, temperature, **kwargs)
            elif cfg.provider == ModelProvider.LLAMACPP:
                resp = await self._call_llamacpp(cfg, messages, max_tokens, temperature, **kwargs)
            elif cfg.provider == ModelProvider.CUSTOM:
                resp = await self._call_custom(cfg, messages, max_tokens, temperature, **kwargs)
            else:
                resp = ModelResponse(model_name=cfg.name, content="", error=f"Unsupported provider: {cfg.provider}")

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            resp.response_time_ms = elapsed_ms
            return resp

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.error("Model %s error: %s", cfg.name, exc)
            return ModelResponse(
                model_name=cfg.name,
                content="",
                error=str(exc),
                response_time_ms=elapsed_ms,
            )

    # -- Anthropic -----------------------------------------------------

    async def _call_anthropic(
        self,
        cfg: ModelConfig,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        **kwargs: Any,
    ) -> ModelResponse:
        url = f"{cfg.base_url}/v1/messages"
        headers = {
            "x-api-key": cfg.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": cfg.name,
            "messages": self._convert_messages_anthropic(messages),
            "max_tokens": max_tokens or cfg.max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        content_blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        usage = data.get("usage", {})

        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        cost = inp * cfg.cost_per_input_token + out * cfg.cost_per_output_token

        return ModelResponse(
            model_name=cfg.name,
            content=text,
            tokens_input=inp,
            tokens_output=out,
            cost=cost,
            cached=usage.get("cache_creation_input_tokens", 0) > 0 or usage.get("cache_read_input_tokens", 0) > 0,
            finish_reason=data.get("stop_reason", ""),
            raw_metadata=data,
        )

    @staticmethod
    def _convert_messages_anthropic(messages: list[dict]) -> list[dict]:
        """Convert generic messages to Anthropic format, handling system messages."""
        result: list[dict] = []
        system_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                result.append({"role": "user", "content": content})
            elif role == "assistant":
                result.append({"role": "assistant", "content": content})
            else:
                # Map other roles to user
                result.append({"role": "user", "content": f"[{role}] {content}"})
        # Prepend system if any
        if system_parts:
            sys_text = "\n".join(system_parts)
            result.insert(0, {"role": "user", "content": f"<system>\n{sys_text}\n</system>"})
        return result

    # -- OpenAI --------------------------------------------------------

    async def _call_openai(
        self,
        cfg: ModelConfig,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        **kwargs: Any,
    ) -> ModelResponse:
        url = f"{cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": cfg.name,
            "messages": messages,
            "max_tokens": max_tokens or cfg.max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content", "")
        usage = data.get("usage", {})

        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        cost = inp * cfg.cost_per_input_token + out * cfg.cost_per_output_token

        return ModelResponse(
            model_name=cfg.name,
            content=text,
            tokens_input=inp,
            tokens_output=out,
            cost=cost,
            finish_reason=choice.get("finish_reason", ""),
            raw_metadata=data,
        )

    # -- OpenRouter ----------------------------------------------------

    async def _call_openrouter(
        self,
        cfg: ModelConfig,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        **kwargs: Any,
    ) -> ModelResponse:
        url = f"{cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://claude-clone.local",
            "X-Title": "Claude Clone",
        }
        body: dict[str, Any] = {
            "model": cfg.name,
            "messages": messages,
            "max_tokens": max_tokens or cfg.max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content", "")
        usage = data.get("usage", {})

        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        cost = inp * cfg.cost_per_input_token + out * cfg.cost_per_output_token

        return ModelResponse(
            model_name=cfg.name,
            content=text,
            tokens_input=inp,
            tokens_output=out,
            cost=cost,
            finish_reason=choice.get("finish_reason", ""),
            raw_metadata=data,
        )

    # -- Ollama --------------------------------------------------------

    async def _call_ollama(
        self,
        cfg: ModelConfig,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        **kwargs: Any,
    ) -> ModelResponse:
        url = f"{cfg.base_url}/api/chat"
        body: dict[str, Any] = {
            "model": cfg.name,
            "messages": messages,
            "stream": False,
        }
        if max_tokens is not None:
            body["options"] = body.get("options", {})
            body["options"]["num_predict"] = max_tokens
        if temperature is not None:
            body["options"] = body.get("options", {})
            body["options"]["temperature"] = temperature

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()

        text = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_duration = data.get("eval_duration", 0)
        prompt_eval_duration = data.get("prompt_eval_duration", 0)

        return ModelResponse(
            model_name=cfg.name,
            content=text,
            tokens_input=prompt_eval_count,
            tokens_output=eval_count,
            cost=0.0,  # local model — free
            raw_metadata=data,
        )

    # -- llama.cpp -----------------------------------------------------

    async def _call_llamacpp(
        self,
        cfg: ModelConfig,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        **kwargs: Any,
    ) -> ModelResponse:
        url = f"{cfg.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        body: dict[str, Any] = {
            "messages": messages,
        }
        if max_tokens is not None:
            body["n_predict"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        text = data.get("content", "")
        tokens_prompt = data.get("tokens_prompt", 0)
        tokens_eval = data.get("tokens_eval", 0)

        return ModelResponse(
            model_name=cfg.name,
            content=text,
            tokens_input=tokens_prompt,
            tokens_output=tokens_eval,
            cost=0.0,  # local model — free
            raw_metadata=data,
        )

    # -- Custom --------------------------------------------------------

    async def _call_custom(
        self,
        cfg: ModelConfig,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Call a custom endpoint using OpenAI-compatible API format."""
        if not cfg.base_url:
            return ModelResponse(model_name=cfg.name, content="", error="Custom provider has no base_url configured")

        url = f"{cfg.base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        body: dict[str, Any] = {
            "model": cfg.name,
            "messages": messages,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content", "")
        usage = data.get("usage", {})

        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        cost = inp * cfg.cost_per_input_token + out * cfg.cost_per_output_token

        return ModelResponse(
            model_name=cfg.name,
            content=text,
            tokens_input=inp,
            tokens_output=out,
            cost=cost,
            finish_reason=choice.get("finish_reason", ""),
            raw_metadata=data,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def _stream_model(
        self,
        cfg: ModelConfig,
        messages: list[dict],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream text chunks from a model."""
        self._record_request(cfg.name)

        if cfg.provider in (ModelProvider.OLLAMA,):
            async for chunk in self._stream_ollama(cfg, messages, **kwargs):
                yield chunk
        elif cfg.provider in (ModelProvider.OPENAI, ModelProvider.OPENROUTER, ModelProvider.CUSTOM):
            async for chunk in self._stream_openai_compatible(cfg, messages, **kwargs):
                yield chunk
        elif cfg.provider == ModelProvider.ANTHROPIC:
            async for chunk in self._stream_anthropic(cfg, messages, **kwargs):
                yield chunk
        elif cfg.provider == ModelProvider.LLAMACPP:
            async for chunk in self._stream_llamacpp(cfg, messages, **kwargs):
                yield chunk
        else:
            yield f"[Error: streaming not supported for provider {cfg.provider.value}]"

    async def _stream_ollama(
        self,
        cfg: ModelConfig,
        messages: list[dict],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        url = f"{cfg.base_url}/api/chat"
        body: dict[str, Any] = {
            "model": cfg.name,
            "messages": messages,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            async with client.stream("POST", url, json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    async def _stream_openai_compatible(
        self,
        cfg: ModelConfig,
        messages: list[dict],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        url = f"{cfg.base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        body: dict[str, Any] = {
            "model": cfg.name,
            "messages": messages,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        chunk = delta.get("content", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue

    async def _stream_anthropic(
        self,
        cfg: ModelConfig,
        messages: list[dict],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        url = f"{cfg.base_url}/v1/messages"
        headers = {
            "x-api-key": cfg.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": cfg.name,
            "messages": self._convert_messages_anthropic(messages),
            "max_tokens": cfg.max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                        event_type = data.get("type", "")
                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                        elif event_type == "message_stop":
                            break
                    except json.JSONDecodeError:
                        continue

    async def _stream_llamacpp(
        self,
        cfg: ModelConfig,
        messages: list[dict],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        url = f"{cfg.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        body: dict[str, Any] = {
            "messages": messages,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                        chunk = data.get("content", "")
                        if chunk:
                            yield chunk
                        if data.get("stop", False):
                            break
                    except json.JSONDecodeError:
                        continue

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    async def _probe_ollama(self) -> None:
        """Check if Ollama is running locally."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:11434/api/tags")
                if resp.status_code == 200:
                    self._ollama_available = True
                    models = resp.json().get("models", [])
                    logger.info("Ollama detected — %d local models available", len(models))
                else:
                    self._ollama_available = False
        except Exception:
            self._ollama_available = False

    async def _probe_llamacpp(self) -> None:
        """Check if llama.cpp server is running locally."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:8080/health")
                if resp.status_code == 200:
                    self._llamacpp_available = True
                    logger.info("llama.cpp server detected at localhost:8080")
                else:
                    self._llamacpp_available = False
        except Exception:
            try:
                # Try alternate health endpoint
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get("http://localhost:8080/v1/models")
                    if resp.status_code == 200:
                        self._llamacpp_available = True
                        logger.info("llama.cpp server detected (v1/models endpoint)")
                    else:
                        self._llamacpp_available = False
            except Exception:
                self._llamacpp_available = False

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_comparison_winner(responses: list[ModelResponse], task: str) -> str:
        """Pick a winner from comparison responses.

        Uses a simple heuristic:
        * Failed responses are eliminated.
        * Score = (quality_signal * 0.5) + (speed_signal * 0.3) + (cost_signal * 0.2)
        * Quality signal: output length appropriateness + code block presence.
        * Speed signal: inverse of response time (faster = better).
        * Cost signal: inverse of cost (cheaper = better).
        """
        candidates: list[tuple[float, str]] = []

        for resp in responses:
            if resp.error is not None:
                continue
            if not resp.content.strip():
                continue

            # Quality: longer outputs for complex tasks, code blocks
            quality = 0.5
            if resp.content.strip():
                quality = min(len(resp.content.strip()) / 500.0, 1.0)
            if "```" in resp.content:
                quality = min(quality + 0.2, 1.0)

            # Speed: faster is better (normalised to 0-1, 10s = 0, 0s = 1)
            speed = max(0.0, 1.0 - (resp.response_time_ms / 10000.0))

            # Cost: cheaper is better
            cost = 1.0 if resp.cost == 0 else max(0.0, 1.0 - (resp.cost / 0.1))

            score = quality * 0.5 + speed * 0.3 + cost * 0.2
            candidates.append((score, resp.model_name))

        if not candidates:
            return "none"

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # ------------------------------------------------------------------
    # Routing reason builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_routing_reason(
        cfg: ModelConfig,
        task: str,
        score: float,
        prefer_local: bool,
    ) -> str:
        """Build a human-readable explanation for why a model was chosen."""
        parts: list[str] = []

        if prefer_local:
            parts.append("local model preferred")

        cap_names = {
            ModelCapability.FAST_RESPONSE: "fast",
            ModelCapability.CHEAP: "cheap",
            ModelCapability.REASONING: "strong reasoning",
            ModelCapability.CODE_COMPLETION: "code-specialised",
            ModelCapability.RELIABLE: "reliable",
            ModelCapability.VISION: "vision-capable",
        }
        caps = [cap_names.get(c, c.value) for c in cfg.capabilities if c in cap_names]
        if caps:
            parts.append(f"offers {', '.join(caps[:3])}")

        parts.append(f"scored {score:.1f}/100")

        if cfg.cost_per_input_token == 0 and cfg.cost_per_output_token == 0:
            parts.append("free to use")
        elif cfg.cost_per_input_token < 0.000001:
            parts.append("very low cost")

        return f"Selected {cfg.display_name}: " + "; ".join(parts) + f". Best match for '{task}' task."


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 chars per token for English)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _human_readable_cost(cost: float) -> str:
    """Format a cost value as a human-readable string."""
    if cost == 0:
        return "$0.00 (free)"
    if cost < 0.0001:
        return f"${cost:.8f}"
    if cost < 0.01:
        return f"${cost:.6f}"
    if cost < 1:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


# ---------------------------------------------------------------------------
# Convenience: module-level singleton (lazy)
# ---------------------------------------------------------------------------

_router_instance: ModelRouter | None = None


async def get_router() -> ModelRouter:
    """Return the module-level :class:`ModelRouter` singleton (lazy init)."""
    global _router_instance
    if _router_instance is None:
        _router_instance = ModelRouter()
        await _router_instance.initialize()
    return _router_instance


async def reset_router() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _router_instance
    _router_instance = None
