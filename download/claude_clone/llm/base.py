"""
Base LLM abstraction layer.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for an LLM connection."""
    model: str = "anthropic/claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = 1.0
    timeout: int = 120
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cost_estimate(self) -> float:
        """Rough cost estimate in USD (very approximate)."""
        # Claude Sonnet: ~$3/MTok input, ~$15/MTok output
        input_cost = (self.prompt_tokens / 1_000_000) * 3.0
        output_cost = (self.completion_tokens / 1_000_000) * 15.0
        return input_cost + output_cost


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    
    Subclasses must implement :meth:`complete` for synchronous calls
    and optionally :meth:`complete_async` for async calls.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
    
    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        tools: Optional[List[Dict]] = None,
        response_format: Optional[Any] = None,
    ) -> LLMResponse:
        """Make a synchronous LLM completion call."""
        ...
    
    async def complete_async(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        tools: Optional[List[Dict]] = None,
        response_format: Optional[Any] = None,
    ) -> LLMResponse:
        """Make an asynchronous LLM completion call."""
        import asyncio
        return await asyncio.to_thread(
            self.complete, messages, system, tools, response_format
        )
    
    def get_token_usage_summary(self) -> Dict[str, int]:
        """Return cumulative token usage."""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
        }
    
    def reset_usage(self) -> None:
        """Reset token usage counters."""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
