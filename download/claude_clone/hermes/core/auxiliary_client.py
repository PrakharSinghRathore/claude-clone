"""
Auxiliary Client — Secondary model client for subtasks.

Provides a lightweight HTTP client for calling cheaper/faster models for
subtasks like title generation, summarization, classification, and
similar operations that don't require the primary model's capabilities.

Uses connection pooling and shared sessions for efficiency. Supports
Anthropic, OpenAI, and OpenRouter-compatible APIs.

Usage
-----
    client = AuxiliaryClient(
        api_key="sk-...",
        model="claude-3-5-haiku-20241022",
        provider="openrouter",
    )
    result = await client.complete(
        system_prompt="Summarize concisely.",
        user_message="Long text to summarize...",
        max_tokens=100,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AuxiliaryResponse:
    """
    Response from an auxiliary model call.

    Attributes
    ----------
    content:
        The model's text response.
    model:
        The model that was used.
    input_tokens:
        Input token count (if available).
    output_tokens:
        Output token count (if available).
    cost_usd:
        Estimated cost.
    duration_ms:
        Response duration in milliseconds.
    error:
        Error message if the call failed.
    """

    content: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Whether the call succeeded."""
        return self.error is None


# ──────────────────────────────────────────────────────────────────────────────
# Provider configurations
# ──────────────────────────────────────────────────────────────────────────────

_PROVIDER_CONFIGS: Dict[str, Dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "chat_endpoint": "/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "chat_endpoint": "/v1/messages",
        "auth_header": "x-api-key",
        "auth_prefix": "",
        "anthropic_version": "2023-06-01",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "chat_endpoint": "/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "chat_endpoint": "/models/{model}:generateContent",
        "auth_header": "x-goog-api-key",
        "auth_prefix": "",
    },
}

# Default models per provider (fast/cheap options)
_DEFAULT_MODELS: Dict[str, str] = {
    "openrouter": "anthropic/claude-3-5-haiku-20241022",
    "anthropic": "claude-3-5-haiku-20241022",
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
}


# ──────────────────────────────────────────────────────────────────────────────
# AuxiliaryClient
# ──────────────────────────────────────────────────────────────────────────────

class AuxiliaryClient:
    """
    Lightweight HTTP client for secondary model calls.

    Designed for subtasks that don't require the primary model's full
    capabilities. Uses httpx with connection pooling for efficiency.

    Parameters
    ----------
    api_key:
        API key for the provider.
    model:
        Model name to use. If ``None``, uses provider default.
    base_url:
        Custom API base URL. If ``None``, uses provider default.
    provider:
        API provider (``"openrouter"``, ``"anthropic"``, ``"openai"``, ``"google"``).
    timeout:
        Request timeout in seconds (default: 30).
    max_connections:
        Maximum concurrent connections (default: 10).
    """

    def __init__(
        self,
        api_key: str = "",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openrouter",
        timeout: float = 30.0,
        max_connections: int = 10,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._provider = provider.lower()
        self._timeout = timeout
        self._max_connections = max_connections
        self._client: Any = None
        self._request_count = 0

    async def _get_client(self):
        """Get or create the HTTP client with connection pooling."""
        if self._client is not None:
            return self._client

        try:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=5,
                ),
            )
            return self._client
        except ImportError:
            raise ImportError("httpx is required for AuxiliaryClient. Install with: pip install httpx")

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def model(self) -> str:
        """Current model name."""
        return self._model or _DEFAULT_MODELS.get(self._provider, "gpt-4o-mini")

    @property
    def provider(self) -> str:
        """Current provider name."""
        return self._provider

    @property
    def request_count(self) -> int:
        """Total number of requests made."""
        return self._request_count

    # ── Public API ────────────────────────────────────────────────────────

    async def complete(
        self,
        user_message: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        """
        Send a completion request and return the response text.

        Parameters
        ----------
        user_message:
            The user prompt.
        system_prompt:
            Optional system prompt.
        max_tokens:
            Maximum tokens for the response.
        temperature:
            Sampling temperature.
        json_mode:
            Request JSON mode output.

        Returns
        -------
        str
            The model's text response. Returns empty string on error.
        """
        response = await self.complete_raw(
            user_message=user_message,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
        return response.content

    async def complete_raw(
        self,
        user_message: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> AuxiliaryResponse:
        """
        Send a completion request and return the full response object.

        Returns
        -------
        AuxiliaryResponse
            Complete response with metadata.
        """
        import time

        start = time.monotonic()
        self._request_count += 1

        try:
            if self._provider == "anthropic":
                result = await self._call_anthropic(
                    user_message, system_prompt, max_tokens, temperature,
                )
            else:
                # OpenRouter, OpenAI, and Google use OpenAI-compatible format
                result = await self._call_openai_compatible(
                    user_message, system_prompt, max_tokens, temperature, json_mode,
                )

            result.duration_ms = (time.monotonic() - start) * 1000
            return result

        except Exception as e:
            logger.debug("AuxiliaryClient error: %s", e)
            return AuxiliaryResponse(
                content="",
                model=self.model,
                error=str(e),
                duration_ms=(time.monotonic() - start) * 1000,
            )

    async def batch_complete(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        concurrency: int = 5,
    ) -> List[str]:
        """
        Process multiple messages concurrently.

        Parameters
        ----------
        messages:
            List of ``{"role": ..., "content": ...}`` dicts.
        system_prompt:
            Optional system prompt.
        max_tokens:
            Max tokens per response.
        temperature:
            Sampling temperature.
        concurrency:
            Maximum concurrent requests.

        Returns
        -------
        list[str]
            List of response strings (in the same order as input).
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _process_one(msg: Dict[str, str]) -> str:
            async with semaphore:
                content = msg.get("content", "")
                return await self.complete(
                    user_message=content,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

        results = await asyncio.gather(
            *[_process_one(m) for m in messages],
            return_exceptions=True,
        )

        return [
            str(r) if isinstance(r, Exception) else r
            for r in results
        ]

    # ── Provider-specific implementations ─────────────────────────────────

    async def _call_openai_compatible(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> AuxiliaryResponse:
        """Call an OpenAI-compatible API endpoint."""
        client = await self._get_client()
        config = _PROVIDER_CONFIGS.get(self._provider, _PROVIDER_CONFIGS["openrouter"])
        base = self._base_url or config["base_url"]

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            config["auth_header"]: f"{config['auth_prefix']}{self._api_key}",
            "Content-Type": "application/json",
        }

        # OpenRouter requires additional headers
        if self._provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/claude-clone"
            headers["X-Title"] = "Claude Clone (Auxiliary)"

        url = f"{base}{config['chat_endpoint']}"
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        content = ""
        input_tokens = 0
        output_tokens = 0

        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")

        if "usage" in data:
            usage = data["usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        return AuxiliaryResponse(
            content=content or "",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def _call_anthropic(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AuxiliaryResponse:
        """Call the Anthropic Messages API directly."""
        client = await self._get_client()
        config = _PROVIDER_CONFIGS["anthropic"]
        base = self._base_url or config["base_url"]

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user_message}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": config.get("anthropic_version", "2023-06-01"),
            "Content-Type": "application/json",
        }

        url = f"{base}{config['chat_endpoint']}"
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        content = ""
        input_tokens = 0
        output_tokens = 0

        if "content" in data:
            for block in data["content"]:
                if block.get("type") == "text":
                    content += block.get("text", "")

        if "usage" in data:
            input_tokens = data["usage"].get("input_tokens", 0)
            output_tokens = data["usage"].get("output_tokens", 0)

        return AuxiliaryResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
