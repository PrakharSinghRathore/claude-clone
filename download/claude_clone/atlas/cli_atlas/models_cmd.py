"""
Model management commands for Atlas CLI.

List, switch, compare, and test AI models.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from atlas.cli_atlas.config_manager import ConfigManager

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# ──────────────────────────────────────────────
# Model catalog
# ──────────────────────────────────────────────

MODEL_CATALOG = {
    "anthropic": {
        "claude-opus-4-20250514": {
            "name": "Claude Opus 4",
            "description": "Most capable model for complex tasks",
            "context_window": 200000,
            "input_price": 15.0,
            "output_price": 75.0,
            "tier": "premium",
        },
        "claude-sonnet-4-20250514": {
            "name": "Claude Sonnet 4",
            "description": "Balanced performance and speed",
            "context_window": 200000,
            "input_price": 3.0,
            "output_price": 15.0,
            "tier": "standard",
        },
        "claude-3-5-haiku-20241022": {
            "name": "Claude 3.5 Haiku",
            "description": "Fast and efficient for simple tasks",
            "context_window": 200000,
            "input_price": 0.8,
            "output_price": 4.0,
            "tier": "budget",
        },
    },
    "google": {
        "gemini-2.5-pro-preview": {
            "name": "Gemini 2.5 Pro",
            "description": "Google's latest multimodal model",
            "context_window": 1000000,
            "input_price": 1.25,
            "output_price": 10.0,
            "tier": "standard",
        },
    },
    "openai": {
        "gpt-4o": {
            "name": "GPT-4o",
            "description": "OpenAI's flagship model",
            "context_window": 128000,
            "input_price": 2.5,
            "output_price": 10.0,
            "tier": "standard",
        },
        "gpt-4o-mini": {
            "name": "GPT-4o Mini",
            "description": "Fast and affordable",
            "context_window": 128000,
            "input_price": 0.15,
            "output_price": 0.6,
            "tier": "budget",
        },
    },
    "meta-llama": {
        "llama-4-maverick": {
            "name": "Llama 4 Maverick",
            "description": "Meta's open-weight model",
            "context_window": 1000000,
            "input_price": 0.2,
            "output_price": 0.6,
            "tier": "budget",
        },
    },
}


class ModelManager:
    """Manages AI model configuration and selection."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()

    def list_models(
        self,
        provider: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List available models."""
        models = []

        for provider_name, provider_models in MODEL_CATALOG.items():
            if provider and provider != provider_name:
                continue

            for model_id, model_info in provider_models.items():
                if tier and model_info.get("tier") != tier:
                    continue

                models.append({
                    "id": model_id,
                    "full_id": f"{provider_name}/{model_id}",
                    "provider": provider_name,
                    **model_info,
                })

        return models

    def get_current_model(self) -> str:
        """Get the currently selected model."""
        return self.config.get("model", "anthropic/claude-sonnet-4-20250514")

    def set_model(self, model_id: str) -> bool:
        """Set the active model."""
        # Validate model exists in catalog or allow custom
        if "/" in model_id:
            provider, model = model_id.split("/", 1)
            if provider in MODEL_CATALOG and model in MODEL_CATALOG[provider]:
                self.config.set("model", model_id)
                self.config.save()
                return True
            # Allow custom models too
            self.config.set("model", model_id)
            self.config.save()
            return True

        # Try to find model without provider prefix
        for provider_name, provider_models in MODEL_CATALOG.items():
            if model_id in provider_models:
                full_id = f"{provider_name}/{model_id}"
                self.config.set("model", full_id)
                self.config.save()
                return True

        # Allow custom model names
        self.config.set("model", model_id)
        self.config.save()
        return True

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        if "/" in model_id:
            provider, model = model_id.split("/", 1)
            if provider in MODEL_CATALOG and model in MODEL_CATALOG[provider]:
                info = MODEL_CATALOG[provider][model].copy()
                info["id"] = model
                info["full_id"] = model_id
                info["provider"] = provider
                return info

        # Search without provider
        for provider_name, provider_models in MODEL_CATALOG.items():
            if model_id in provider_models:
                info = provider_models[model_id].copy()
                info["id"] = model_id
                info["full_id"] = f"{provider_name}/{model_id}"
                info["provider"] = provider_name
                return info

        return None

    def compare_models(self, model_ids: List[str]) -> List[Dict[str, Any]]:
        """Compare multiple models side by side."""
        comparisons = []
        for model_id in model_ids:
            info = self.get_model_info(model_id)
            if info:
                comparisons.append(info)
        return comparisons

    def estimate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate API cost for a given usage."""
        info = self.get_model_info(model_id)
        if not info:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * info.get("input_price", 3.0)
        output_cost = (output_tokens / 1_000_000) * info.get("output_price", 15.0)
        return input_cost + output_cost

    def test_connectivity(self, model_id: str) -> Dict[str, Any]:
        """Test connectivity to a model."""
        if not HAS_HTTPX:
            return {"success": False, "error": "httpx not installed"}

        api_key = self.config.get("api_key") or os.environ.get(
            "OPENROUTER_API_KEY", ""
        ) or os.environ.get("ANTHROPIC_API_KEY", "")

        if not api_key:
            return {"success": False, "error": "No API key configured"}

        base_url = self.config.get("base_url", "https://openrouter.ai/api/v1")

        try:
            start_time = time.time()
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5,
                    },
                )
            elapsed = time.time() - start_time

            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "response_time": round(elapsed, 2),
                    "tokens_used": data.get("usage", {}),
                    "model": data.get("model", model_id),
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}",
                    "response_time": round(elapsed, 2),
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def pin_model(self, model_id: str) -> None:
        """Add a model to pinned favorites."""
        pinned = self.config.get("pinned_models", [])
        full_id = model_id
        if model_id not in pinned:
            pinned.append(full_id)
            self.config.set("pinned_models", pinned)
            self.config.save()

    def unpin_model(self, model_id: str) -> None:
        """Remove a model from pinned favorites."""
        pinned = self.config.get("pinned_models", [])
        if model_id in pinned:
            pinned.remove(model_id)
            self.config.set("pinned_models", pinned)
            self.config.save()

    def get_pinned_models(self) -> List[str]:
        """Get list of pinned models."""
        return self.config.get("pinned_models", [])

    def format_model_table(
        self,
        models: Optional[List[Dict]] = None,
    ) -> str:
        """Format models as a readable table."""
        models = models or self.list_models()
        pinned = self.get_pinned_models()
        current = self.get_current_model()

        lines = []
        lines.append(f"  {'Model':<40} {'Provider':<12} {'Context':>8} {'In $':>8} {'Out $':>8} {'Tier':<8}")
        lines.append("  " + "-" * 92)

        for model in models:
            full_id = model.get("full_id", model.get("id", ""))
            name = model.get("name", full_id)
            provider = model.get("provider", "")
            ctx = model.get("context_window", 0)
            in_price = model.get("input_price", 0)
            out_price = model.get("output_price", 0)
            tier = model.get("tier", "")

            # Markers
            prefix = ""
            if full_id == current:
                prefix = "\u2713 "
            elif full_id in pinned:
                prefix = "\u2605 "

            lines.append(
                f"  {prefix}{name:<38} {provider:<12} {ctx:>7,} "
                f"${in_price:>6.2f} ${out_price:>6.2f} {tier:<8}"
            )

        return "\n".join(lines)
