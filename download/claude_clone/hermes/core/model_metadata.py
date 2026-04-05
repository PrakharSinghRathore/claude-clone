"""
Model Metadata — Model catalog, token estimation, and context limit detection.

Provides a comprehensive model catalog covering Anthropic, OpenAI, Google,
DeepSeek, Meta, Mistral, NousResearch, and local endpoints. Includes token
estimation, context limit detection, and model capability metadata.

Usage
-----
    from hermes.core.model_metadata import ModelMetadata, get_model_info

    info = get_model_info("claude-sonnet-4-20250514")
    print(info.context_window)  # 200000
    print(info.cost_per_million_input)  # 3.0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

class ModelProvider(Enum):
    """Supported model providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    META = "meta"
    MISTRAL = "mistral"
    NOUS = "nous"
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"


class ModelCapability(Enum):
    """Capabilities a model may have."""
    CHAT = "chat"
    CODE = "code"
    REASONING = "reasoning"
    VISION = "vision"
    AUDIO = "audio"
    THINKING = "thinking"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    FAST = "fast"
    CHEAP = "cheap"
    RELIABLE = "reliable"
    LOCAL = "local"


@dataclass
class ModelInfo:
    """
    Comprehensive metadata for a single model.

    Attributes
    ----------
    name:
        Canonical model name (e.g., ``"claude-sonnet-4-20250514"``).
    display_name:
        Human-readable name for UI display.
    provider:
        The provider that hosts this model.
    context_window:
        Maximum context window in tokens.
    max_output_tokens:
        Maximum output tokens per request.
    cost_per_million_input:
        Cost per 1M input tokens in USD.
    cost_per_million_output:
        Cost per 1M output tokens in USD.
    capabilities:
        Set of model capabilities.
    description:
        Human-readable description of the model.
    aliases:
        Alternative names that map to this model.
    release_date:
        Approximate release date (YYYY-MM-DD).
    architecture:
        Model architecture family.
    supports_system_prompt:
        Whether the model supports system prompts.
    supports_tools:
        Whether the model supports function/tool calling.
    """

    name: str
    display_name: str = ""
    provider: ModelProvider = ModelProvider.ANTHROPIC
    context_window: int = 128_000
    max_output_tokens: int = 4096
    cost_per_million_input: float = 0.0
    cost_per_million_output: float = 0.0
    capabilities: set = field(default_factory=set)
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    release_date: str = ""
    architecture: str = ""
    supports_system_prompt: bool = True
    supports_tools: bool = True

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "provider": self.provider.value,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "cost_per_million_input": self.cost_per_million_input,
            "cost_per_million_output": self.cost_per_million_output,
            "capabilities": [c.value for c in self.capabilities],
            "description": self.description,
            "aliases": self.aliases,
            "release_date": self.release_date,
            "architecture": self.architecture,
            "supports_system_prompt": self.supports_system_prompt,
            "supports_tools": self.supports_tools,
        }

    @property
    def is_free(self) -> bool:
        """Whether this model has zero cost."""
        return self.cost_per_million_input == 0 and self.cost_per_million_output == 0

    @property
    def is_local(self) -> bool:
        """Whether this is a locally-hosted model."""
        return ModelCapability.LOCAL in self.capabilities


# ──────────────────────────────────────────────────────────────────────────────
# Model Catalog
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CATALOG: Dict[str, ModelInfo] = {}


def _register(name: str, provider: str, **kwargs) -> None:
    """Register a model in the catalog."""
    MODEL_CATALOG[name.lower()] = ModelInfo(
        name=name,
        provider=ModelProvider(provider),
        **kwargs,
    )


# ── Anthropic Claude ─────────────────────────────────────────────────────

_register("claude-opus-4-20250514", "anthropic",
          display_name="Claude Opus 4",
          context_window=200_000, max_output_tokens=16_384,
          cost_per_million_input=15.0, cost_per_million_output=75.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.VISION, ModelCapability.THINKING, ModelCapability.FUNCTION_CALLING,
                       ModelCapability.RELIABLE},
          description="Most capable Claude model — best for complex reasoning, analysis, and code generation.",
          aliases=["claude-opus-4", "opus-4"],
          release_date="2025-05-14", architecture="claude")

_register("claude-sonnet-4-20250514", "anthropic",
          display_name="Claude Sonnet 4",
          context_window=200_000, max_output_tokens=16_384,
          cost_per_million_input=3.0, cost_per_million_output=15.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.VISION, ModelCapability.THINKING, ModelCapability.FUNCTION_CALLING,
                       ModelCapability.FAST, ModelCapability.RELIABLE},
          description="Balanced Claude model — great code generation and reasoning at moderate cost.",
          aliases=["claude-sonnet-4", "sonnet-4"],
          release_date="2025-05-14", architecture="claude")

_register("claude-3-5-haiku-20241022", "anthropic",
          display_name="Claude 3.5 Haiku",
          context_window=200_000, max_output_tokens=8192,
          cost_per_million_input=0.8, cost_per_million_output=4.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                       ModelCapability.FUNCTION_CALLING, ModelCapability.FAST, ModelCapability.CHEAP},
          description="Fast and affordable Claude — ideal for quick answers, chat, and simple tasks.",
          aliases=["claude-3.5-haiku", "claude-haiku"],
          release_date="2024-10-22", architecture="claude")

_register("claude-3-5-sonnet-20241022", "anthropic",
          display_name="Claude 3.5 Sonnet",
          context_window=200_000, max_output_tokens=8192,
          cost_per_million_input=3.0, cost_per_million_output=15.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.VISION, ModelCapability.FUNCTION_CALLING, ModelCapability.FAST},
          description="Previous-generation Claude — strong coding and reasoning.",
          aliases=["claude-3.5-sonnet", "claude-sonnet-3.5"],
          release_date="2024-10-22", architecture="claude")

_register("claude-3-opus-20240229", "anthropic",
          display_name="Claude 3 Opus",
          context_window=200_000, max_output_tokens=4096,
          cost_per_million_input=15.0, cost_per_million_output=75.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.VISION, ModelCapability.FUNCTION_CALLING, ModelCapability.RELIABLE},
          description="Previous-generation flagship — powerful but slower.",
          aliases=["claude-3-opus", "claude-opus"],
          release_date="2024-02-29", architecture="claude")

_register("claude-3-haiku-20240307", "anthropic",
          display_name="Claude 3 Haiku",
          context_window=200_000, max_output_tokens=4096,
          cost_per_million_input=0.25, cost_per_million_output=1.25,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                       ModelCapability.FUNCTION_CALLING, ModelCapability.FAST, ModelCapability.CHEAP},
          description="Budget Claude 3 — fast and cheap.",
          aliases=["claude-3-haiku"],
          release_date="2024-03-07", architecture="claude")

# ── OpenAI GPT ────────────────────────────────────────────────────────────

_register("gpt-4o", "openai",
          display_name="GPT-4o",
          context_window=128_000, max_output_tokens=16_384,
          cost_per_million_input=2.5, cost_per_million_output=10.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.VISION, ModelCapability.AUDIO, ModelCapability.FUNCTION_CALLING,
                       ModelCapability.JSON_MODE, ModelCapability.RELIABLE},
          description="OpenAI flagship — strong all-rounder with vision and audio.",
          aliases=["gpt4o"],
          release_date="2024-05-13", architecture="gpt")

_register("gpt-4o-mini", "openai",
          display_name="GPT-4o Mini",
          context_window=128_000, max_output_tokens=16_384,
          cost_per_million_input=0.15, cost_per_million_output=0.6,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                       ModelCapability.FUNCTION_CALLING, ModelCapability.JSON_MODE,
                       ModelCapability.FAST, ModelCapability.CHEAP},
          description="Tiny, cheap, fast — good for quick answers.",
          aliases=["gpt4o-mini"],
          release_date="2024-07-18", architecture="gpt")

_register("gpt-4-turbo", "openai",
          display_name="GPT-4 Turbo",
          context_window=128_000, max_output_tokens=4096,
          cost_per_million_input=10.0, cost_per_million_output=30.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.VISION, ModelCapability.FUNCTION_CALLING, ModelCapability.JSON_MODE},
          description="Fast GPT-4 variant with 128K context.",
          aliases=["gpt-4-1106-preview", "gpt4-turbo"],
          release_date="2024-04-09", architecture="gpt")

_register("o1", "openai",
          display_name="OpenAI o1",
          context_window=200_000, max_output_tokens=32_768,
          cost_per_million_input=15.0, cost_per_million_output=60.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.FUNCTION_CALLING, ModelCapability.RELIABLE},
          description="OpenAI reasoning model — excels at complex multi-step problems.",
          aliases=["o1-preview", "o1-2024-12-17"],
          release_date="2024-12-17", architecture="o1")

_register("o1-mini", "openai",
          display_name="OpenAI o1 Mini",
          context_window=128_000, max_output_tokens=65_536,
          cost_per_million_input=3.0, cost_per_million_output=12.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.FUNCTION_CALLING, ModelCapability.FAST},
          description="Faster, cheaper o1 variant.",
          aliases=["o1-mini-2024-09-12"],
          release_date="2024-09-12", architecture="o1")

_register("o3-mini", "openai",
          display_name="OpenAI o3-mini",
          context_window=200_000, max_output_tokens=100_000,
          cost_per_million_input=1.10, cost_per_million_output=4.40,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.FUNCTION_CALLING, ModelCapability.FAST, ModelCapability.CHEAP},
          description="Latest small reasoning model — excellent value.",
          aliases=["o3-mini-2025-01-31"],
          release_date="2025-01-31", architecture="o3")

# ── Google Gemini ─────────────────────────────────────────────────────────

_register("gemini-2.0-flash", "google",
          display_name="Gemini 2.0 Flash",
          context_window=1_048_576, max_output_tokens=8192,
          cost_per_million_input=0.10, cost_per_million_output=0.40,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.VISION, ModelCapability.AUDIO, ModelCapability.FUNCTION_CALLING,
                       ModelCapability.FAST, ModelCapability.CHEAP},
          description="Google's fast multimodal model with 1M token context.",
          aliases=["gemini-2.0-flash-001"],
          release_date="2024-12-11", architecture="gemini")

_register("gemini-2.0-pro", "google",
          display_name="Gemini 2.0 Pro",
          context_window=2_097_152, max_output_tokens=8192,
          cost_per_million_input=1.25, cost_per_million_output=10.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.VISION, ModelCapability.FUNCTION_CALLING, ModelCapability.RELIABLE},
          description="Google's most capable model with 2M token context.",
          aliases=["gemini-2.0-pro-001"],
          release_date="2025-02-05", architecture="gemini")

_register("gemini-1.5-pro", "google",
          display_name="Gemini 1.5 Pro",
          context_window=2_097_152, max_output_tokens=8192,
          cost_per_million_input=1.25, cost_per_million_output=5.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.VISION, ModelCapability.AUDIO, ModelCapability.FUNCTION_CALLING},
          description="Previous-gen Gemini with 2M context.",
          aliases=["gemini-1.5-pro-002"],
          release_date="2024-02-15", architecture="gemini")

# ── DeepSeek ──────────────────────────────────────────────────────────────

_register("deepseek-chat", "deepseek",
          display_name="DeepSeek V3",
          context_window=131_072, max_output_tokens=8192,
          cost_per_million_input=0.14, cost_per_million_output=0.28,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.FUNCTION_CALLING, ModelCapability.CHEAP, ModelCapability.FAST},
          description="DeepSeek's chat model — excellent value for money.",
          aliases=["deepseek-v3"],
          release_date="2024-12-26", architecture="deepseek")

_register("deepseek-reasoner", "deepseek",
          display_name="DeepSeek R1",
          context_window=131_072, max_output_tokens=16_384,
          cost_per_million_input=0.55, cost_per_million_output=2.19,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.THINKING, ModelCapability.FUNCTION_CALLING},
          description="DeepSeek's reasoning model — strong performance at low cost.",
          aliases=["deepseek-r1"],
          release_date="2025-01-20", architecture="deepseek")

# ── Meta Llama ────────────────────────────────────────────────────────────

_register("llama-3.3-70b", "meta",
          display_name="Llama 3.3 70B",
          context_window=131_072, max_output_tokens=16_384,
          cost_per_million_input=0.39, cost_per_million_output=0.39,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.FUNCTION_CALLING, ModelCapability.FAST, ModelCapability.CHEAP},
          description="Meta's open-weight model — excellent cost/performance.",
          aliases=["llama-3.3-70b-instruct"],
          release_date="2024-12-06", architecture="llama")

_register("llama-3.1-405b", "meta",
          display_name="Llama 3.1 405B",
          context_window=131_072, max_output_tokens=16_384,
          cost_per_million_input=2.0, cost_per_million_output=2.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.FUNCTION_CALLING, ModelCapability.RELIABLE},
          description="Meta's largest open-weight model — near frontier performance.",
          aliases=["llama-3.1-405b-instruct"],
          release_date="2024-07-23", architecture="llama")

# ── Mistral ───────────────────────────────────────────────────────────────

_register("mistral-large", "mistral",
          display_name="Mistral Large",
          context_window=128_000, max_output_tokens=4096,
          cost_per_million_input=2.0, cost_per_million_output=6.0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.FUNCTION_CALLING, ModelCapability.JSON_MODE},
          description="Mistral's flagship — strong multilingual capabilities.",
          aliases=["mistral-large-2411"],
          release_date="2024-11-01", architecture="mistral")

_register("mistral-small", "mistral",
          display_name="Mistral Small",
          context_window=128_000, max_output_tokens=4096,
          cost_per_million_input=0.20, cost_per_million_output=0.60,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.FUNCTION_CALLING,
                       ModelCapability.FAST, ModelCapability.CHEAP},
          description="Budget Mistral — fast and affordable.",
          aliases=["mistral-small-2411"],
          release_date="2024-11-01", architecture="mistral")

_register("codestral", "mistral",
          display_name="Codestral",
          context_window=32_768, max_output_tokens=8192,
          cost_per_million_input=0.30, cost_per_million_output=0.90,
          capabilities={ModelCapability.CODE, ModelCapability.CHAT, ModelCapability.FAST, ModelCapability.CHEAP},
          description="Mistral's code-specialized model.",
          aliases=["codestral-2501"],
          release_date="2025-01-01", architecture="mistral")

# ── NousResearch ──────────────────────────────────────────────────────────

_register("hermes-3-llama-3.1-405b", "nous",
          display_name="Hermes 3 405B",
          context_window=131_072, max_output_tokens=16_384,
          cost_per_million_input=2.7, cost_per_million_output=2.7,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.FUNCTION_CALLING, ModelCapability.JSON_MODE, ModelCapability.RELIABLE},
          description="NousResearch fine-tune of Llama 3.1 405B — optimized for agentic use.",
          aliases=["hermes-3-405b"],
          release_date="2024-09-15", architecture="llama")

_register("hermes-3-llama-3.1-70b", "nous",
          display_name="Hermes 3 70B",
          context_window=131_072, max_output_tokens=16_384,
          cost_per_million_input=0.39, cost_per_million_output=0.39,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                       ModelCapability.FUNCTION_CALLING, ModelCapability.JSON_MODE,
                       ModelCapability.FAST, ModelCapability.CHEAP},
          description="NousResearch fine-tune of Llama 3.1 70B.",
          aliases=["hermes-3-70b"],
          release_date="2024-09-15", architecture="llama")

# ── Local models (free) ──────────────────────────────────────────────────

_register("deepseek-coder-v2", "ollama",
          display_name="DeepSeek Coder V2 (Local)",
          context_window=128_000, max_output_tokens=8192,
          cost_per_million_input=0, cost_per_million_output=0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.LOCAL,
                       ModelCapability.CHEAP, ModelCapability.FUNCTION_CALLING},
          description="Free local model via Ollama — code-specialized.",
          aliases=["deepseek-coder"],
          release_date="2024-06-01", architecture="deepseek")

_register("llama3", "ollama",
          display_name="Llama 3 (Local)",
          context_window=8192, max_output_tokens=4096,
          cost_per_million_input=0, cost_per_million_output=0,
          capabilities={ModelCapability.CHAT, ModelCapability.LOCAL, ModelCapability.CHEAP},
          description="Free local model via Ollama — general purpose.",
          aliases=["llama-3", "llama3:8b"],
          release_date="2024-04-18", architecture="llama")

_register("qwen2.5-coder", "ollama",
          display_name="Qwen 2.5 Coder (Local)",
          context_window=131_072, max_output_tokens=8192,
          cost_per_million_input=0, cost_per_million_output=0,
          capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.LOCAL,
                       ModelCapability.CHEAP, ModelCapability.FUNCTION_CALLING},
          description="Free local coding model via Ollama — strong code capabilities.",
          aliases=["qwen2.5-coder:32b"],
          release_date="2024-11-01", architecture="qwen")


# ──────────────────────────────────────────────────────────────────────────────
# Model Metadata API
# ──────────────────────────────────────────────────────────────────────────────

def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """
    Look up model metadata by name.

    Handles provider prefixes (e.g., ``"anthropic/claude-sonnet-4-20250514"``),
    aliases, and case-insensitive lookup.

    Parameters
    ----------
    model_name:
        The model name or alias to look up.

    Returns
    -------
    ModelInfo or None
        The model metadata, or ``None`` if not found.
    """
    # Strip provider prefix
    lookup = model_name
    if "/" in lookup:
        lookup = lookup.split("/", 1)[-1]

    # Direct lookup (case-insensitive)
    key = lookup.lower()
    if key in MODEL_CATALOG:
        return MODEL_CATALOG[key]

    # Alias lookup
    for info in MODEL_CATALOG.values():
        if lookup.lower() in [a.lower() for a in info.aliases]:
            return info

    # Fuzzy match: check if the name starts with a known model name
    for catalog_key, info in MODEL_CATALOG.items():
        if key.startswith(catalog_key) or catalog_key.startswith(key):
            return info

    return None


def list_models(
    provider: Optional[str] = None,
    capability: Optional[ModelCapability] = None,
) -> List[ModelInfo]:
    """
    List models from the catalog, optionally filtered.

    Parameters
    ----------
    provider:
        Filter by provider name.
    capability:
        Filter by required capability.

    Returns
    -------
    list[ModelInfo]
        Matching models.
    """
    results = list(MODEL_CATALOG.values())
    if provider is not None:
        results = [m for m in results if m.provider.value == provider.lower()]
    if capability is not None:
        results = [m for m in results if capability in m.capabilities]
    return results


def estimate_tokens(text: str, model_name: str = "") -> int:
    """
    Estimate token count for text.

    Uses tiktoken when available, otherwise falls back to a character heuristic.
    The ``model_name`` parameter is reserved for model-specific tokenizers.

    Parameters
    ----------
    text:
        The text to estimate tokens for.
    model_name:
        Optional model name (reserved for model-specific tokenizers).

    Returns
    -------
    int
        Estimated token count.
    """
    if not text:
        return 0

    # Try tiktoken
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback heuristic
    return max(1, len(text) // 4)


def detect_context_limit(model_name: str) -> int:
    """
    Detect the context window limit for a model.

    Falls back to 128,000 if the model is not in the catalog.

    Parameters
    ----------
    model_name:
        The model name to look up.

    Returns
    -------
    int
        Context window in tokens.
    """
    info = get_model_info(model_name)
    if info:
        return info.context_window
    return 128_000


def is_local_model(model_name: str) -> bool:
    """Check if a model is a locally-hosted model."""
    info = get_model_info(model_name)
    return info.is_local if info else False


class ModelMetadata:
    """
    High-level model metadata manager.

    Provides methods for looking up models, estimating costs, and managing
    model configurations.

    Parameters
    ----------
    custom_models:
        Optional list of custom ``ModelInfo`` objects to add to the catalog.
    """

    def __init__(self, custom_models: Optional[List[ModelInfo]] = None) -> None:
        if custom_models:
            for model in custom_models:
                MODEL_CATALOG[model.name.lower()] = model

    def get(self, model_name: str) -> Optional[ModelInfo]:
        """Look up a model by name."""
        return get_model_info(model_name)

    def list(
        self,
        provider: Optional[str] = None,
        capability: Optional[ModelCapability] = None,
    ) -> List[ModelInfo]:
        """List models, optionally filtered."""
        return list_models(provider, capability)

    def estimate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost in USD for a given model and token counts.

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
            Estimated cost in USD.
        """
        info = get_model_info(model_name)
        if info:
            return (
                (input_tokens / 1_000_000) * info.cost_per_million_input
                + (output_tokens / 1_000_000) * info.cost_per_million_output
            )
        return 0.0

    def detect_context_limit(self, model_name: str) -> int:
        """Detect context window limit for a model."""
        return detect_context_limit(model_name)

    def register(self, model: ModelInfo) -> None:
        """Register a custom model in the catalog."""
        MODEL_CATALOG[model.name.lower()] = model
