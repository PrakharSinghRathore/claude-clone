"""
LLM provider factory and implementations.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from llm.base import BaseLLM, LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class LLMProvider(BaseLLM):
    """
    Universal LLM provider supporting multiple backends.
    
    Automatically detects the correct provider from the model string
    or configuration and routes accordingly.
    
    Supported providers:
        - Anthropic (direct)
        - OpenAI (direct)
        - Azure OpenAI
        - Google Gemini
        - Ollama (local)
        - Any OpenAI-compatible API (via base_url)
    """
    
    PROVIDERS = {
        "anthropic": "anthropic",
        "openai": "openai",
        "azure": "azure",
        "gemini": "gemini",
        "ollama": "ollama",
    }
    
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._provider = self._detect_provider()
        self._client = None
    
    def _detect_provider(self) -> str:
        """Detect the LLM provider from config."""
        if self.config.base_url:
            return "openai_compatible"
        
        model = self.config.model.lower()
        
        if "claude" in model or "anthropic" in model:
            return "anthropic"
        if "gpt" in model or "o1" in model or "o3" in model:
            return "openai"
        if "gemini" in model or "google" in model:
            return "gemini"
        if "/" in model:
            # Check provider prefix
            prefix = model.split("/")[0]
            for key, val in self.PROVIDERS.items():
                if key in prefix:
                    return val
        
        # Default: try OpenRouter-compatible
        return "openai_compatible"
    
    def _get_client(self):
        """Lazily initialize the provider client."""
        if self._client is not None:
            return self._client
        
        if self._provider == "anthropic":
            import anthropic
            kwargs = {"api_key": self.config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
                kwargs["default_headers"] = {
                    "HTTP-Referer": "https://github.com/claude-clone",
                    "X-Title": "Claude Clone",
                }
            self._client = anthropic.Anthropic(**kwargs)
        
        elif self._provider == "openai":
            from openai import OpenAI
            kwargs = {"api_key": self.config.api_key or os.environ.get("OPENAI_API_KEY", "")}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**kwargs)
        
        elif self._provider == "openai_compatible":
            import anthropic
            api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
            kwargs = {"api_key": api_key}
            base_url = self.config.base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
            kwargs["base_url"] = base_url
            kwargs["default_headers"] = {
                "HTTP-Referer": "https://github.com/claude-clone",
                "X-Title": "Claude Clone",
            }
            self._client = anthropic.Anthropic(**kwargs)
        
        elif self._provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = self.config.api_key or os.environ.get("GOOGLE_API_KEY", "")
                genai.configure(api_key=api_key)
                self._client = genai.GenerativeModel(self.config.model.split("/")[-1])
            except ImportError:
                logger.warning("google-generativeai not installed")
        
        elif self._provider == "ollama":
            pass  # Ollama uses REST API directly
        
        return self._client
    
    def complete(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        tools: Optional[List[Dict]] = None,
        response_format: Optional[Any] = None,
    ) -> LLMResponse:
        """Route to the appropriate provider's completion method."""
        if self._provider == "anthropic":
            return self._complete_anthropic(messages, system, tools)
        elif self._provider == "openai":
            return self._complete_openai(messages, system, tools)
        elif self._provider == "openai_compatible":
            return self._complete_anthropic(messages, system, tools)
        elif self._provider == "gemini":
            return self._complete_gemini(messages, system)
        elif self._provider == "ollama":
            return self._complete_ollama(messages, system)
        
        return LLMResponse(content="Error: Unknown provider", model=self.config.model)
    
    def _complete_anthropic(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        client = self._get_client()
        model = self.config.model.split("/")[-1] if "/" in self.config.model else self.config.model
        
        kwargs = {
            "model": model,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if self.config.temperature != 1.0:
            kwargs["temperature"] = self.config.temperature
        
        response = client.messages.create(**kwargs)
        
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        
        prompt_tokens = response.usage.input_tokens if response.usage else 0
        completion_tokens = response.usage.output_tokens if response.usage else 0
        
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        
        return LLMResponse(
            content=text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    
    def _complete_openai(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        client = self._get_client()
        model = self.config.model.split("/")[-1] if "/" in self.config.model else self.config.model
        
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)
        
        kwargs = {
            "model": model,
            "max_tokens": self.config.max_tokens,
            "messages": all_messages,
        }
        if tools:
            kwargs["tools"] = tools
        
        response = client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        
        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=response.choices[0].finish_reason or "",
        )
    
    def _complete_gemini(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
    ) -> LLMResponse:
        client = self._get_client()
        if client is None:
            return LLMResponse(content="Error: Gemini client not available", model=self.config.model)
        
        prompt = system + "\n" + "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        response = client.generate_content(prompt)
        
        content = response.text if response and response.text else ""
        
        return LLMResponse(content=content, model=self.config.model)
    
    def _complete_ollama(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
    ) -> LLMResponse:
        import urllib.request
        import urllib.error
        
        model = self.config.model.split("/")[-1] if "/" in self.config.model else self.config.model
        base_url = self.config.base_url or "http://localhost:11434"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if system:
            payload["system"] = system
        
        try:
            req = urllib.request.Request(
                f"{base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            
            content = result.get("message", {}).get("content", "")
            eval_info = result.get("eval_count", 0)
            
            return LLMResponse(
                content=content,
                model=model,
                completion_tokens=eval_info,
                total_tokens=eval_info,
            )
        except Exception as e:
            return LLMResponse(content=f"Error: Ollama request failed: {e}", model=model)
