"""
LLM — multi-provider language model abstraction.

Supports Anthropic, OpenAI, Azure, Google Gemini, Ollama,
and any OpenAI-compatible API.
"""

from llm.base import BaseLLM, LLMConfig, LLMResponse
from llm.provider import LLMProvider

__all__ = [
    "BaseLLM",
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
]
