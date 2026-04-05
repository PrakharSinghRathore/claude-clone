"""
Gateway and platform configuration management.

Provides Pydantic-based configuration classes for the Hermes gateway,
with support for YAML/JSON config files, environment variables, and
per-platform rate limits and preferences.

Usage::

    from hermes.gateway.config import GatewayConfig

    config = GatewayConfig.load("gateway.yaml")
    # or
    config = GatewayConfig.from_env()
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ──────────────────────────────────────────────────────────────────────────────
# Platform Config
# ──────────────────────────────────────────────────────────────────────────────

class PlatformConfig:
    """Configuration for a single messaging platform."""

    def __init__(
        self,
        name: str,
        enabled: bool = False,
        token: Optional[str] = None,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 8080,
        rate_limit: int = 30,
        rate_limit_window: int = 60,
        timeout: int = 30,
        max_message_length: int = 4096,
        max_file_size_mb: int = 25,
        allowed_chat_types: Optional[List[str]] = None,
        admin_ids: Optional[List[str]] = None,
        allowed_user_ids: Optional[List[str]] = None,
        blocked_user_ids: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.enabled = enabled
        self.token = token
        self.webhook_url = webhook_url
        self.webhook_secret = webhook_secret
        self.api_url = api_url
        self.api_key = api_key
        self.host = host
        self.port = port
        self.rate_limit = rate_limit
        self.rate_limit_window = rate_limit_window
        self.timeout = timeout
        self.max_message_length = max_message_length
        self.max_file_size_mb = max_file_size_mb
        self.allowed_chat_types = allowed_chat_types or ["private", "group"]
        self.admin_ids = admin_ids or []
        self.allowed_user_ids = allowed_user_ids or []
        self.blocked_user_ids = blocked_user_ids or []
        self.extra = extra or {}

    def resolve_token(self, env_prefix: str) -> str:
        """Resolve the authentication token from config or environment variables."""
        if self.token:
            return self.token
        env_key = f"{env_prefix.upper()}_TOKEN"
        val = os.environ.get(env_key, "")
        if val:
            self.token = val
        return val

    def resolve_api_key(self, env_prefix: str) -> str:
        """Resolve the API key from config or environment variables."""
        if self.api_key:
            return self.api_key
        env_key = f"{env_prefix.upper()}_API_KEY"
        val = os.environ.get(env_key, "")
        if val:
            self.api_key = val
        return val

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary (secrets masked)."""
        d: Dict[str, Any] = {
            "name": self.name,
            "enabled": self.enabled,
            "host": self.host,
            "port": self.port,
            "rate_limit": self.rate_limit,
            "rate_limit_window": self.rate_limit_window,
            "timeout": self.timeout,
            "max_message_length": self.max_message_length,
            "max_file_size_mb": self.max_file_size_mb,
            "allowed_chat_types": self.allowed_chat_types,
            "admin_ids": self.admin_ids,
            "allowed_user_ids": self.allowed_user_ids,
            "blocked_user_ids": self.blocked_user_ids,
            "extra": self.extra,
        }
        if self.token:
            d["token"] = "***" + self.token[-4:] if len(self.token) > 4 else "***"
        if self.api_key:
            d["api_key"] = "***" + self.api_key[-4:] if len(self.api_key) > 4 else "***"
        if self.webhook_url:
            d["webhook_url"] = self.webhook_url
        if self.api_url:
            d["api_url"] = self.api_url
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any], name: str) -> "PlatformConfig":
        """Create a PlatformConfig from a dictionary."""
        return cls(name=name, **{k: v for k, v in data.items() if k in cls.__init__.__code__.co_varnames and k != "name"})


# ──────────────────────────────────────────────────────────────────────────────
# Gateway Config
# ──────────────────────────────────────────────────────────────────────────────

class GatewayConfig:
    """
    Top-level configuration for the Hermes Gateway.

    Manages platform-specific settings, gateway-wide preferences,
    persistence options, and session defaults.
    """

    DEFAULT_CONFIG_PATHS = [
        "gateway.yaml",
        "gateway.yml",
        "gateway.json",
        "~/.claude_clone/gateway.yaml",
        "~/.claude_clone/gateway.json",
    ]

    def __init__(
        self,
        platforms: Optional[Dict[str, PlatformConfig]] = None,
        session_timeout: int = 3600,
        session_max_tokens: int = 100000,
        session_persist_path: Optional[str] = None,
        session_reset_policy: str = "token_limit",
        delivery_retry_count: int = 3,
        delivery_retry_delay: float = 2.0,
        streaming_enabled: bool = True,
        streaming_chunk_size: int = 100,
        streaming_edit_supported: Optional[List[str]] = None,
        mirroring_enabled: bool = False,
        mirror_pairs: Optional[List[Dict[str, Any]]] = None,
        hooks_enabled: bool = True,
        hooks_dir: Optional[str] = None,
        pairing_enabled: bool = True,
        pairing_secret: Optional[str] = None,
        admin_secret: Optional[str] = None,
        log_level: str = "INFO",
        health_check_interval: int = 60,
        status_endpoint: bool = True,
        status_port: int = 9090,
        worker_threads: int = 4,
        message_format_default: str = "markdown",
        gateway_host: str = "0.0.0.0",
        gateway_port: int = 8000,
    ):
        self.platforms = platforms or {}
        self.session_timeout = session_timeout
        self.session_max_tokens = session_max_tokens
        self.session_persist_path = session_persist_path or "~/.claude_clone/gateway_sessions"
        self.session_reset_policy = session_reset_policy
        self.delivery_retry_count = delivery_retry_count
        self.delivery_retry_delay = delivery_retry_delay
        self.streaming_enabled = streaming_enabled
        self.streaming_chunk_size = streaming_chunk_size
        self.streaming_edit_supported = streaming_edit_supported or ["telegram", "discord", "slack"]
        self.mirroring_enabled = mirroring_enabled
        self.mirror_pairs = mirror_pairs or []
        self.hooks_enabled = hooks_enabled
        self.hooks_dir = hooks_dir or "~/.claude_clone/gateway_hooks"
        self.pairing_enabled = pairing_enabled
        self.pairing_secret = pairing_secret
        self.admin_secret = admin_secret
        self.log_level = log_level
        self.health_check_interval = health_check_interval
        self.status_endpoint = status_endpoint
        self.status_port = status_port
        self.worker_threads = worker_threads
        self.message_format_default = message_format_default
        self.gateway_host = gateway_host
        self.gateway_port = gateway_port

    def get_platform(self, name: str) -> Optional[PlatformConfig]:
        """Get a platform configuration by name, or None if not configured."""
        return self.platforms.get(name)

    def get_enabled_platforms(self) -> Dict[str, PlatformConfig]:
        """Return all enabled platform configurations."""
        return {k: v for k, v in self.platforms.items() if v.enabled}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full configuration to a dictionary (secrets masked)."""
        return {
            "session_timeout": self.session_timeout,
            "session_max_tokens": self.session_max_tokens,
            "session_persist_path": self.session_persist_path,
            "session_reset_policy": self.session_reset_policy,
            "delivery_retry_count": self.delivery_retry_count,
            "delivery_retry_delay": self.delivery_retry_delay,
            "streaming_enabled": self.streaming_enabled,
            "streaming_chunk_size": self.streaming_chunk_size,
            "streaming_edit_supported": self.streaming_edit_supported,
            "mirroring_enabled": self.mirroring_enabled,
            "mirror_pairs": self.mirror_pairs,
            "hooks_enabled": self.hooks_enabled,
            "hooks_dir": self.hooks_dir,
            "pairing_enabled": self.pairing_enabled,
            "log_level": self.log_level,
            "health_check_interval": self.health_check_interval,
            "status_endpoint": self.status_endpoint,
            "status_port": self.status_port,
            "worker_threads": self.worker_threads,
            "message_format_default": self.message_format_default,
            "gateway_host": self.gateway_host,
            "gateway_port": self.gateway_port,
            "platforms": {
                name: pc.to_dict() for name, pc in self.platforms.items()
            },
        }

    # ── Config Loading ───────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Optional[str] = None) -> "GatewayConfig":
        """
        Load gateway configuration from a YAML or JSON file.

        Searches ``path`` first, then falls back to default paths.

        Parameters
        ----------
        path:
            Explicit config file path. If ``None``, searches default paths.
        """
        config_path = cls._find_config(path)
        if config_path is None:
            return cls.from_env()

        return cls._load_file(config_path)

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        """
        Create a GatewayConfig from environment variables.

        Environment variable naming convention:
            ``HERMES_<SECTION>_<KEY>`` (e.g., ``HERMES_SESSION_TIMEOUT``)

        Platform-specific variables:
            ``HERMES_TELEGRAM_TOKEN``, ``HERMES_DISCORD_TOKEN``, etc.
        """
        platforms: Dict[str, PlatformConfig] = {}
        platform_names = [
            "telegram", "discord", "slack", "whatsapp", "signal",
            "matrix", "email", "sms", "webhook", "api",
            "dingtalk", "feishu", "wecom", "mattermost",
        ]

        for name in platform_names:
            env_prefix = f"HERMES_{name.upper()}"
            enabled_key = f"{env_prefix}_ENABLED"
            token_key = f"{env_prefix}_TOKEN"
            api_key_key = f"{env_prefix}_API_KEY"
            webhook_key = f"{env_prefix}_WEBHOOK_URL"
            host_key = f"{env_prefix}_HOST"
            port_key = f"{env_prefix}_PORT"

            is_enabled = os.environ.get(enabled_key, "false").lower() in ("true", "1", "yes")

            if is_enabled:
                pc = PlatformConfig(
                    name=name,
                    enabled=True,
                    token=os.environ.get(token_key),
                    api_key=os.environ.get(api_key_key),
                    webhook_url=os.environ.get(webhook_key),
                    host=os.environ.get(host_key, "localhost"),
                    port=int(os.environ.get(port_key, str(8080))),
                )
                platforms[name] = pc

        return cls(
            platforms=platforms,
            session_timeout=int(os.environ.get("HERMES_SESSION_TIMEOUT", "3600")),
            session_max_tokens=int(os.environ.get("HERMES_SESSION_MAX_TOKENS", "100000")),
            session_persist_path=os.environ.get("HERMES_SESSION_PERSIST_PATH"),
            session_reset_policy=os.environ.get("HERMES_SESSION_RESET_POLICY", "token_limit"),
            delivery_retry_count=int(os.environ.get("HERMES_DELIVERY_RETRY_COUNT", "3")),
            delivery_retry_delay=float(os.environ.get("HERMES_DELIVERY_RETRY_DELAY", "2.0")),
            streaming_enabled=os.environ.get("HERMES_STREAMING_ENABLED", "true").lower() in ("true", "1", "yes"),
            streaming_chunk_size=int(os.environ.get("HERMES_STREAMING_CHUNK_SIZE", "100")),
            mirroring_enabled=os.environ.get("HERMES_MIRRORING_ENABLED", "false").lower() in ("true", "1", "yes"),
            hooks_enabled=os.environ.get("HERMES_HOOKS_ENABLED", "true").lower() in ("true", "1", "yes"),
            pairing_enabled=os.environ.get("HERMES_PAIRING_ENABLED", "true").lower() in ("true", "1", "yes"),
            pairing_secret=os.environ.get("HERMES_PAIRING_SECRET"),
            admin_secret=os.environ.get("HERMES_ADMIN_SECRET"),
            log_level=os.environ.get("HERMES_LOG_LEVEL", "INFO"),
            health_check_interval=int(os.environ.get("HERMES_HEALTH_CHECK_INTERVAL", "60")),
            status_port=int(os.environ.get("HERMES_STATUS_PORT", "9090")),
            worker_threads=int(os.environ.get("HERMES_WORKER_THREADS", "4")),
            gateway_host=os.environ.get("HERMES_GATEWAY_HOST", "0.0.0.0"),
            gateway_port=int(os.environ.get("HERMES_GATEWAY_PORT", "8000")),
        )

    @classmethod
    def _find_config(cls, path: Optional[str]) -> Optional[Path]:
        """Find a config file from explicit path or default search paths."""
        if path:
            p = Path(path).expanduser().resolve()
            if p.exists():
                return p
            return None

        for candidate in cls.DEFAULT_CONFIG_PATHS:
            p = Path(candidate).expanduser().resolve()
            if p.exists():
                return p
        return None

    @classmethod
    def _load_file(cls, path: Path) -> "GatewayConfig":
        """Parse a YAML or JSON config file and return a GatewayConfig."""
        raw = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError(
                    "PyYAML is required to load YAML config files. "
                    "Install it with: pip install pyyaml"
                )
            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw)

        return cls._from_parsed(data)

    @classmethod
    def _from_parsed(cls, data: Dict[str, Any]) -> "GatewayConfig":
        """Build a GatewayConfig from a parsed dictionary."""
        # Parse platforms
        platforms: Dict[str, PlatformConfig] = {}
        raw_platforms = data.get("platforms", {})
        if isinstance(raw_platforms, dict):
            for name, pc_data in raw_platforms.items():
                if isinstance(pc_data, dict):
                    platforms[name] = PlatformConfig.from_dict(pc_data, name)

        # Parse mirror pairs
        mirror_pairs = data.get("mirror_pairs", [])
        if not isinstance(mirror_pairs, list):
            mirror_pairs = []

        # Parse streaming edit supported platforms
        streaming_edit = data.get("streaming_edit_supported", None)
        if isinstance(streaming_edit, list):
            pass  # already a list
        elif isinstance(streaming_edit, str):
            streaming_edit = [s.strip() for s in streaming_edit.split(",")]
        else:
            streaming_edit = None

        # Pick up secrets from env
        for name, pc in platforms.items():
            env_prefix = f"HERMES_{name.upper()}"
            if not pc.token:
                pc.resolve_token(env_prefix)
            if not pc.api_key:
                pc.resolve_api_key(env_prefix)
            if not pc.webhook_secret:
                secret_key = f"{env_prefix}_WEBHOOK_SECRET"
                pc.webhook_secret = os.environ.get(secret_key)

        return cls(
            platforms=platforms,
            session_timeout=data.get("session_timeout", 3600),
            session_max_tokens=data.get("session_max_tokens", 100000),
            session_persist_path=data.get("session_persist_path"),
            session_reset_policy=data.get("session_reset_policy", "token_limit"),
            delivery_retry_count=data.get("delivery_retry_count", 3),
            delivery_retry_delay=data.get("delivery_retry_delay", 2.0),
            streaming_enabled=data.get("streaming_enabled", True),
            streaming_chunk_size=data.get("streaming_chunk_size", 100),
            streaming_edit_supported=streaming_edit,
            mirroring_enabled=data.get("mirroring_enabled", False),
            mirror_pairs=mirror_pairs,
            hooks_enabled=data.get("hooks_enabled", True),
            hooks_dir=data.get("hooks_dir"),
            pairing_enabled=data.get("pairing_enabled", True),
            pairing_secret=os.environ.get("HERMES_PAIRING_SECRET") or data.get("pairing_secret"),
            admin_secret=os.environ.get("HERMES_ADMIN_SECRET") or data.get("admin_secret"),
            log_level=data.get("log_level", "INFO"),
            health_check_interval=data.get("health_check_interval", 60),
            status_endpoint=data.get("status_endpoint", True),
            status_port=data.get("status_port", 9090),
            worker_threads=data.get("worker_threads", 4),
            message_format_default=data.get("message_format_default", "markdown"),
            gateway_host=data.get("gateway_host", "0.0.0.0"),
            gateway_port=data.get("gateway_port", 8000),
        )

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the current configuration to a YAML or JSON file.

        Parameters
        ----------
        path:
            Destination path. Extension determines format (.yaml/.yml/.json).
        """
        config_path = Path(path or "gateway.yaml").expanduser().resolve()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        # Reconstruct platform configs with secrets for saving
        for name, pc in self.platforms.items():
            data["platforms"][name] = {
                "name": pc.name,
                "enabled": pc.enabled,
                "token": pc.token,
                "api_key": pc.api_key,
                "webhook_url": pc.webhook_url,
                "webhook_secret": pc.webhook_secret,
                "api_url": pc.api_url,
                "host": pc.host,
                "port": pc.port,
                "rate_limit": pc.rate_limit,
                "rate_limit_window": pc.rate_limit_window,
                "timeout": pc.timeout,
                "max_message_length": pc.max_message_length,
                "max_file_size_mb": pc.max_file_size_mb,
                "allowed_chat_types": pc.allowed_chat_types,
                "admin_ids": pc.admin_ids,
                "allowed_user_ids": pc.allowed_user_ids,
                "blocked_user_ids": pc.blocked_user_ids,
                "extra": pc.extra,
            }

        if config_path.suffix in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError("PyYAML is required to save YAML config files.")
            import yaml
            raw = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        else:
            raw = json.dumps(data, indent=2, default=str, ensure_ascii=False)

        config_path.write_text(raw, encoding="utf-8")
        os.chmod(str(config_path), 0o600)
