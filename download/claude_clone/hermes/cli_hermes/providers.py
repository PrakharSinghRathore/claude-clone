"""
Provider management for Hermes CLI.

Add/remove API providers, provider-specific configuration,
key rotation, health checks, and multi-provider failover.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes.cli_hermes.config_manager import ConfigManager

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# ──────────────────────────────────────────────
# Default providers
# ──────────────────────────────────────────────

DEFAULT_PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "env_var": "OPENROUTER_API_KEY",
        "key_url": "https://openrouter.ai/keys",
        "type": "gateway",
        "supports_streaming": True,
        "supports_vision": True,
        "rate_limit": 60,
        "priority": 1,
    },
    "anthropic": {
        "name": "Anthropic Direct",
        "base_url": "https://api.anthropic.com",
        "env_var": "ANTHROPIC_API_KEY",
        "key_url": "https://console.anthropic.com/account/keys",
        "type": "direct",
        "supports_streaming": True,
        "supports_vision": True,
        "rate_limit": 1000,
        "priority": 2,
    },
}


class ProviderManager:
    """Manages API providers for the Hermes CLI."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all configured providers."""
        providers = self.config.get("providers", {})
        result = []

        # Include defaults that aren't overridden
        all_providers = {**DEFAULT_PROVIDERS, **providers}
        for pid, pinfo in all_providers.items():
            result.append({
                "id": pid,
                "name": pinfo.get("name", pid),
                "base_url": pinfo.get("base_url", ""),
                "type": pinfo.get("type", "direct"),
                "has_key": bool(self._get_api_key(pid, pinfo)),
                "supports_streaming": pinfo.get("supports_streaming", False),
                "supports_vision": pinfo.get("supports_vision", False),
                "priority": pinfo.get("priority", 999),
            })

        return sorted(result, key=lambda p: p.get("priority", 999))

    def add_provider(
        self,
        provider_id: str,
        name: str,
        base_url: str,
        api_key: Optional[str] = None,
        env_var: Optional[str] = None,
        provider_type: str = "direct",
        supports_streaming: bool = True,
        supports_vision: bool = False,
        priority: int = 10,
    ) -> bool:
        """Add a new API provider."""
        providers = dict(self.config.get("providers", {}))

        providers[provider_id] = {
            "name": name,
            "base_url": base_url,
            "api_key": api_key,
            "env_var": env_var,
            "type": provider_type,
            "supports_streaming": supports_streaming,
            "supports_vision": supports_vision,
            "priority": priority,
            "added_at": datetime.now().isoformat(),
        }

        self.config.set("providers", providers)
        self.config.save()
        return True

    def remove_provider(self, provider_id: str) -> bool:
        """Remove a provider."""
        if provider_id in DEFAULT_PROVIDERS:
            return False

        providers = dict(self.config.get("providers", {}))
        if provider_id in providers:
            del providers[provider_id]
            self.config.set("providers", providers)
            self.config.save()
            return True
        return False

    def get_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Get a provider's configuration."""
        providers = self.config.get("providers", {})
        all_providers = {**DEFAULT_PROVIDERS, **providers}
        pinfo = all_providers.get(provider_id)
        if pinfo:
            result = dict(pinfo)
            result["id"] = provider_id
            result["has_key"] = bool(self._get_api_key(provider_id, pinfo))
            return result
        return None

    def get_active_provider(self) -> str:
        """Get the currently active provider."""
        return self.config.get("provider", "openrouter")

    def set_active_provider(self, provider_id: str) -> bool:
        """Set the active provider."""
        all_providers = list(self.list_providers())
        provider_ids = [p["id"] for p in all_providers]

        if provider_id not in provider_ids:
            return False

        pinfo = self.get_provider(provider_id)
        if pinfo:
            self.config.set("provider", provider_id)
            self.config.set("base_url", pinfo.get("base_url", ""))
            self.config.save()
            return True
        return False

    def health_check(self, provider_id: str) -> Dict[str, Any]:
        """Check provider health and connectivity."""
        pinfo = self.get_provider(provider_id)
        if not pinfo:
            return {"status": "error", "message": f"Provider '{provider_id}' not found"}

        api_key = self._get_api_key(provider_id, pinfo)
        if not api_key:
            return {"status": "error", "message": "No API key configured", "provider": provider_id}

        if not HAS_HTTPX:
            return {"status": "unknown", "message": "httpx not installed", "provider": provider_id}

        base_url = pinfo.get("base_url", "")
        try:
            start_time = time.time()
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    base_url.rstrip("/v1").rstrip("/") + "/models" if "openrouter" in base_url else base_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    follow_redirects=True,
                )
            elapsed = time.time() - start_time

            if resp.status_code in (200, 401, 403):
                return {
                    "status": "reachable" if resp.status_code == 200 else "auth_error",
                    "response_time_ms": round(elapsed * 1000),
                    "http_status": resp.status_code,
                    "provider": provider_id,
                }
            else:
                return {
                    "status": "error",
                    "response_time_ms": round(elapsed * 1000),
                    "http_status": resp.status_code,
                    "provider": provider_id,
                }
        except Exception as e:
            return {
                "status": "unreachable",
                "error": str(e),
                "provider": provider_id,
            }

    def setup_key_rotation(
        self,
        provider_id: str,
        keys: List[str],
    ) -> bool:
        """Set up API key rotation for a provider."""
        providers = dict(self.config.get("providers", {}))
        all_providers = {**DEFAULT_PROVIDERS, **providers}

        if provider_id not in all_providers:
            return False

        if provider_id not in providers:
            providers[provider_id] = {}

        providers[provider_id]["key_rotation"] = {
            "keys": [k for k in keys if k],  # Filter empty keys
            "current_index": 0,
            "rotation_strategy": "round_robin",
            "last_rotated": datetime.now().isoformat(),
        }

        self.config.set("providers", providers)
        self.config.save()
        return True

    def rotate_key(self, provider_id: str) -> Optional[str]:
        """Rotate to the next API key."""
        providers = dict(self.config.get("providers", {}))
        all_providers = {**DEFAULT_PROVIDERS, **providers}
        pinfo = all_providers.get(provider_id, {})

        rotation = pinfo.get("key_rotation")
        if not rotation or not rotation.get("keys"):
            return None

        keys = rotation["keys"]
        current = rotation.get("current_index", 0)
        next_index = (current + 1) % len(keys)

        if provider_id not in providers:
            providers[provider_id] = {}

        providers[provider_id]["key_rotation"] = {
            **rotation,
            "current_index": next_index,
            "last_rotated": datetime.now().isoformat(),
        }

        self.config.set("providers", providers)
        self.config.save()
        return keys[next_index]

    def configure_failover(
        self,
        provider_order: List[str],
        strategy: str = "priority",
    ) -> bool:
        """Configure multi-provider failover."""
        self.config.set("failover", {
            "enabled": True,
            "provider_order": provider_order,
            "strategy": strategy,
            "configured_at": datetime.now().isoformat(),
        })
        self.config.save()
        return True

    def get_failover_config(self) -> Dict[str, Any]:
        """Get current failover configuration."""
        return self.config.get("failover", {
            "enabled": False,
            "provider_order": ["openrouter"],
            "strategy": "priority",
        })

    def format_provider_table(self) -> str:
        """Format providers as a readable table."""
        providers = self.list_providers()
        active = self.get_active_provider()

        lines = []
        lines.append(f"  {'Provider':<16} {'Type':<10} {'Base URL':<35} {'Key':<8} {'Active'}")
        lines.append("  " + "-" * 80)

        for p in providers:
            marker = "\u2713" if p["id"] == active else ""
            key_status = "\u2713" if p["has_key"] else "\u2717"
            lines.append(
                f"  {p['name']:<16} {p['type']:<10} {p['base_url']:<35} {key_status:<8} {marker}"
            )

        return "\n".join(lines)

    def _get_api_key(self, provider_id: str, pinfo: Dict) -> Optional[str]:
        """Get API key from config or environment."""
        # Check provider-specific config
        if pinfo.get("api_key"):
            return pinfo["api_key"]

        # Check environment variable
        env_var = pinfo.get("env_var")
        if env_var:
            return os.environ.get(env_var)

        # Check common env vars
        if provider_id == "openrouter":
            return os.environ.get("OPENROUTER_API_KEY")
        elif provider_id == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")

        return None
