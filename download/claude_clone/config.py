"""
Configuration management for Claude Clone.
Handles API keys, model settings, MCP servers, and user preferences.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


DEFAULT_CONFIG_DIR = Path.home() / ".claude_clone"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"


class Config:
    """Manages all configuration for the Claude Clone application."""

    DEFAULTS = {
        "model": "anthropic/claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "max_iterations": 10,
        "theme": "dark",
        "temperature": 1.0,
        "mcp_servers": [],
        "allowed_tools": [],
        "disabled_tools": [],
        "auto_approve_tools": [],
        "system_prompt_overrides": {},
        "context_files": [],
        "cost_warning_threshold": 1.0,
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "active_agent": None,
        "sandbox": {
            "enabled": True,
            "max_memory_mb": 512,
            "default_timeout": 30,
            "auto_cleanup": True,
            "allowed_languages": ["python", "javascript", "bash"]
        },
        "memory": {
            "enabled": True,
            "db_path": "~/.claude_clone/memory.db",
            "auto_summarize": True,
            "max_context_tokens": 4000,
            "retention_days": 90
        },
        "analyzer": {
            "enabled": True,
            "auto_analyze": False,
            "snapshot_on_analyze": True,
            "max_complexity_threshold": 15,
            "min_quality_score": 60
        },
        "security": {
            "enabled": True,
            "auto_scan": False,
            "severity_threshold": "MEDIUM",
            "ignore_file": ".claudescanignore",
            "scan_on_save": False
        },
        "deployment": {
            "default_platform": "docker",
            "history_limit": 50,
            "auto_health_check": True,
            "health_check_timeout": 30
        },
        "plugins": {
            "enabled": True,
            "plugin_dir": "~/.claude_clone/plugins",
            "auto_reload": True,
            "hot_reload_interval": 2
        },
        "collaboration": {
            "enabled": False,
            "server_host": "localhost",
            "server_port": 8765,
            "default_room": "default"
        },
        "desktop": {
            "enabled": True,
            "mode": "ACTIVE",
            "awareness": {
                "monitor_clipboard": True,
                "monitor_windows": True,
                "screenshot_on_request": True,
                "ocr_enabled": True,
                "snapshot_interval": 5,
                "window_history_minutes": 30
            },
            "voice": {
                "enabled": False,
                "stt_engine": "google",
                "tts_engine": "pyttsx3",
                "wake_word": "hey claude",
                "language": "en-US",
                "continuous": False,
                "volume": 0.8,
                "rate": 1.0
            },
            "controller": {
                "enabled": True,
                "smooth_mouse": True,
                "human_typing": True,
                "recording_enabled": True
            },
            "permissions": {
                "level": "STANDARD",
                "auto_approve_read": True,
                "ask_before_delete": True,
                "ask_before_install": True,
                "ask_before_shell": True,
                "audit_log": True,
                "max_audit_entries": 50000
            }
        },
        "self_improving": {
            "enabled": False,
            "auto_improve": False,
            "improvement_interval": 3600,
            "max_patches_per_cycle": 10,
            "max_extensions_per_cycle": 3,
            "max_optimizations_per_cycle": 5,
        },
        "knowledge_base": {
            "enabled": False,
            "db_path": "~/.claude_clone/knowledge.db",
            "auto_extract": True,
            "max_context_tokens": 2000,
            "auto_prune_days": 90,
            "import_obsidian": False,
            "obsidian_vault_path": None,
        },
    }

    # OpenRouter / Anthropic base URL
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    ANTHROPIC_BASE_URL = "https://api.anthropic.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_iterations: Optional[int] = None,
        theme: Optional[str] = None,
        temperature: Optional[float] = None,
        mcp_servers: Optional[List[Dict]] = None,
        allowed_tools: Optional[List[str]] = None,
        disabled_tools: Optional[List[str]] = None,
        auto_approve_tools: Optional[List[str]] = None,
        system_prompt_overrides: Optional[Dict[str, str]] = None,
        context_files: Optional[List[str]] = None,
        cost_warning_threshold: Optional[float] = None,
        cwd: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        active_agent: Optional[str] = None,
    ):
        # API key: check OpenRouter first, then Anthropic
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        # Provider: openrouter or anthropic
        self.provider = provider or self._detect_provider()
        # Base URL: derived from provider
        self.base_url = base_url or self._get_base_url()
        self.model = model or self.DEFAULTS["model"]
        self.max_tokens = max_tokens or self.DEFAULTS["max_tokens"]
        self.max_iterations = max_iterations or self.DEFAULTS["max_iterations"]
        self.theme = theme or self.DEFAULTS["theme"]
        self.temperature = temperature if temperature is not None else self.DEFAULTS["temperature"]
        self.mcp_servers = mcp_servers or list(self.DEFAULTS["mcp_servers"])
        self.allowed_tools = allowed_tools or list(self.DEFAULTS["allowed_tools"])
        self.disabled_tools = disabled_tools or list(self.DEFAULTS["disabled_tools"])
        self.auto_approve_tools = auto_approve_tools or list(self.DEFAULTS["auto_approve_tools"])
        self.system_prompt_overrides = system_prompt_overrides or dict(self.DEFAULTS["system_prompt_overrides"])
        self.context_files = context_files or list(self.DEFAULTS["context_files"])
        self.cost_warning_threshold = cost_warning_threshold or self.DEFAULTS["cost_warning_threshold"]
        self.active_agent = active_agent
        self.cwd = cwd or os.getcwd()
        self._config_path = DEFAULT_CONFIG_FILE
        self.sandbox = dict(self.DEFAULTS["sandbox"])
        self.memory = dict(self.DEFAULTS["memory"])
        self.analyzer = dict(self.DEFAULTS["analyzer"])
        self.security = dict(self.DEFAULTS["security"])
        self.deployment = dict(self.DEFAULTS["deployment"])
        self.plugins = dict(self.DEFAULTS["plugins"])
        self.collaboration = dict(self.DEFAULTS["collaboration"])
        self.desktop = dict(self.DEFAULTS["desktop"])
        self.self_improving = dict(self.DEFAULTS["self_improving"])
        self.knowledge_base = dict(self.DEFAULTS["knowledge_base"])

    def _detect_provider(self) -> str:
        """Detect API provider from available keys."""
        if os.environ.get("OPENROUTER_API_KEY"):
            return "openrouter"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        return self.DEFAULTS["provider"]

    def _get_base_url(self) -> str:
        """Get the API base URL based on provider."""
        if self.provider == "openrouter":
            return self.OPENROUTER_BASE_URL
        return self.ANTHROPIC_BASE_URL

    @classmethod
    def from_env(cls) -> "Config":
        """Create a Config instance from environment variables and config file."""
        api_key = os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        model = os.environ.get("CLAUDE_MODEL")
        max_tokens = os.environ.get("CLAUDE_MAX_TOKENS")
        max_iterations = os.environ.get("CLAUDE_MAX_ITERATIONS")
        theme = os.environ.get("CLAUDE_THEME")
        provider = os.environ.get("API_PROVIDER")

        config = cls(
            api_key=api_key or None,
            model=model or None,
            max_tokens=int(max_tokens) if max_tokens else None,
            max_iterations=int(max_iterations) if max_iterations else None,
            theme=theme or None,
            provider=provider or None,
        )

        if not config.api_key:
            config.load()

        return config

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """Load configuration from a JSON file."""
        config_path = Path(path) if path else DEFAULT_CONFIG_FILE

        if not config_path.exists():
            config = cls()
            config._config_path = config_path
            return config

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return cls()

        kwargs = {}
        for key in cls.DEFAULTS:
            if key in data:
                kwargs[key] = data[key]

        config = cls(**kwargs)
        config._config_path = config_path

        if "api_key" in data and data["api_key"]:
            config.api_key = data["api_key"]

        # Load provider settings
        if "provider" in data:
            config.provider = data["provider"]
        if "base_url" in data:
            config.base_url = data["base_url"]
        elif config.provider:
            config.base_url = config._get_base_url()
        if "active_agent" in data:
            config.active_agent = data["active_agent"]

        # Load new feature section configs
        for section in ("sandbox", "memory", "analyzer", "security", "deployment", "plugins", "collaboration", "desktop", "self_improving", "knowledge_base"):
            if section in data and isinstance(data[section], dict):
                merged = dict(getattr(config, section))
                merged.update(data[section])
                setattr(config, section, merged)

        return config

    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to a JSON file."""
        config_path = Path(path) if path else self._config_path
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "max_iterations": self.max_iterations,
            "theme": self.theme,
            "temperature": self.temperature,
            "mcp_servers": self.mcp_servers,
            "allowed_tools": self.allowed_tools,
            "disabled_tools": self.disabled_tools,
            "auto_approve_tools": self.auto_approve_tools,
            "system_prompt_overrides": self.system_prompt_overrides,
            "context_files": self.context_files,
            "cost_warning_threshold": self.cost_warning_threshold,
            "provider": self.provider,
            "base_url": self.base_url,
            "active_agent": self.active_agent,
            "sandbox": self.sandbox,
            "memory": self.memory,
            "analyzer": self.analyzer,
            "security": self.security,
            "deployment": self.deployment,
            "plugins": self.plugins,
            "collaboration": self.collaboration,
            "desktop": self.desktop,
            "self_improving": self.self_improving,
            "knowledge_base": self.knowledge_base,
        }

        # Only save API key if explicitly set (not from env)
        api_key_env = os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        if self.api_key and self.api_key != api_key_env:
            data["api_key"] = self.api_key

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        os.chmod(config_path, 0o600)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "max_iterations": self.max_iterations,
            "theme": self.theme,
            "temperature": self.temperature,
            "cwd": self.cwd,
            "api_key_set": bool(self.api_key),
            "provider": self.provider,
            "base_url": self.base_url,
            "active_agent": self.active_agent,
            "mcp_servers_count": len(self.mcp_servers),
            "allowed_tools": self.allowed_tools,
            "disabled_tools": self.disabled_tools,
            "sandbox": self.sandbox,
            "memory": self.memory,
            "analyzer": self.analyzer,
            "security": self.security,
            "deployment": self.deployment,
            "plugins": self.plugins,
            "collaboration": self.collaboration,
            "desktop": self.desktop,
            "self_improving": self.self_improving,
            "knowledge_base": self.knowledge_base,
        }

    def validate(self) -> List[str]:
        """Validate the current configuration and return a list of warnings."""
        warnings = []

        if not self.api_key:
            warnings.append(
                "No API key set. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY. "
                "Get an OpenRouter key at: https://openrouter.ai/keys"
            )

        if self.max_tokens < 1 or self.max_tokens > 200000:
            warnings.append(f"max_tokens={self.max_tokens} is outside recommended range [1, 200000]")

        if self.max_iterations < 1 or self.max_iterations > 100:
            warnings.append(f"max_iterations={self.max_iterations} is outside recommended range [1, 100]")

        if self.theme not in ("dark", "light"):
            warnings.append(f"Unknown theme: {self.theme}. Use 'dark' or 'light'")

        if self.temperature < 0 or self.temperature > 2:
            warnings.append(f"temperature={self.temperature} is outside range [0, 2]")

        return warnings

    def get_effective_tools(self, all_tools: Dict) -> Dict:
        """Filter the tool registry based on allowed/disabled lists."""
        if self.allowed_tools:
            return {k: v for k, v in all_tools.items() if k in self.allowed_tools}
        result = dict(all_tools)
        for t in self.disabled_tools:
            result.pop(t, None)
        return result

    def ensure_config_dir(self) -> Path:
        """Ensure the configuration directory exists."""
        DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return DEFAULT_CONFIG_DIR

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD based on model pricing."""
        # Strip provider prefix for model name (e.g. "anthropic/claude-sonnet-4" → "claude-sonnet-4")
        model_key = self.model
        if "/" in model_key:
            model_key = model_key.split("/")[-1]

        pricing = {
            "claude-opus-4-20250514": (15.0, 75.0),
            "claude-sonnet-4-20250514": (3.0, 15.0),
            "claude-3-5-sonnet-20241022": (3.0, 15.0),
            "claude-3-5-haiku-20241022": (0.8, 4.0),
            "claude-3-opus-20240229": (15.0, 75.0),
            "claude-3-sonnet-20240229": (3.0, 15.0),
            "claude-3-haiku-20240307": (0.25, 1.25),
        }
        input_cost, output_cost = pricing.get(model_key, (3.0, 15.0))
        return (input_tokens / 1_000_000) * input_cost + (output_tokens / 1_000_000) * output_cost

    # ── Feature-section helpers ──────────────────────────────────────

    def get_sandbox_config(self) -> dict:
        """Return a copy of the sandbox configuration."""
        return dict(self.sandbox)

    def get_memory_config(self) -> dict:
        """Return a copy of the memory configuration."""
        return dict(self.memory)

    def get_analyzer_config(self) -> dict:
        """Return a copy of the analyzer configuration."""
        return dict(self.analyzer)

    def get_security_config(self) -> dict:
        """Return a copy of the security configuration."""
        return dict(self.security)

    def get_deployment_config(self) -> dict:
        """Return a copy of the deployment configuration."""
        return dict(self.deployment)

    def get_plugin_config(self) -> dict:
        """Return a copy of the plugin configuration."""
        return dict(self.plugins)

    def get_collaboration_config(self) -> dict:
        """Return a copy of the collaboration configuration."""
        return dict(self.collaboration)

    def get_desktop_config(self) -> dict:
        """Return a copy of the desktop configuration."""
        return dict(self.desktop)
