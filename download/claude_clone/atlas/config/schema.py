"""
Atlas Configuration Schema — Comprehensive configuration dataclasses.

Provides zero-dependency configuration classes using Python dataclasses
with validation, defaults, merging, and environment variable substitution.
All configuration sub-systems are combined into a single ``AppConfig``.

Usage::

    from atlas.config.schema import AppConfig, get_defaults, merge

    defaults = get_defaults()
    overrides = {"agent": {"model": "claude-3-5-sonnet-20241022"}}
    config = merge(defaults, overrides)
    app_config = AppConfig.from_dict(config)
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .types import (
    DMPolicy,
    LogFormat,
    LogLevel,
    MemoryBackend,
    ProviderType,
    SandboxType,
    parse_bool,
    parse_float,
    parse_int,
    parse_string_list,
    resolve_env_var,
    validate_port,
    validate_path,
    is_valid_url,
    mask_secret,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Agent Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """
    Configuration for the AI agent's core behavior.

    Attributes
    ----------
    model:
        Default model identifier (e.g., ``"claude-sonnet-4-20250514"``).
    provider:
        Model provider (e.g., ``"anthropic"``).
    temperature:
        Sampling temperature (0.0 - 2.0).
    max_tokens:
        Maximum tokens per response.
    system_prompt:
        Custom system prompt template.
    tools:
        List of enabled tool names (empty = all).
    disabled_tools:
        List of disabled tool names.
    features:
        Set of enabled feature flags.
    max_tool_calls:
        Maximum tool calls per turn.
    streaming:
        Whether to stream responses.
    """

    model: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    temperature: float = 1.0
    max_tokens: int = 8192
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    disabled_tools: List[str] = field(default_factory=list)
    features: Set[str] = field(default_factory=lambda: {
        "memory", "web_search", "file_ops", "code_execution",
    })
    max_tool_calls: int = 10
    streaming: bool = True

    def validate(self) -> List[str]:
        """Validate agent configuration. Returns a list of error messages."""
        errors: List[str] = []
        if not self.model:
            errors.append("agent.model must not be empty")
        if not (0.0 <= self.temperature <= 2.0):
            errors.append(f"agent.temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_tokens < 1:
            errors.append(f"agent.max_tokens must be >= 1, got {self.max_tokens}")
        if self.max_tool_calls < 1:
            errors.append(f"agent.max_tool_calls must be >= 1, got {self.max_tool_calls}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["features"] = list(self.features)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        if "features" in filtered and isinstance(filtered["features"], (list, tuple, set)):
            filtered["features"] = set(filtered["features"])
        return cls(**filtered)


# ──────────────────────────────────────────────────────────────────────────────
# Gateway Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GatewayConfig:
    """
    Configuration for the messaging gateway.

    Attributes
    ----------
    host:
        Gateway bind host.
    port:
        Gateway bind port.
    tls:
        Whether to enable TLS.
    tls_cert:
        Path to TLS certificate file.
    tls_key:
        Path to TLS private key file.
    platforms:
        Dict of platform-specific configurations.
    session_timeout:
        Session idle timeout in seconds.
    max_concurrent:
        Maximum concurrent sessions.
    enabled:
        Whether the gateway is enabled.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    tls: bool = False
    tls_cert: str = ""
    tls_key: str = ""
    platforms: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_timeout: int = 3600
    max_concurrent: int = 50
    enabled: bool = False

    def validate(self) -> List[str]:
        errors: List[str] = []
        try:
            validate_port(self.port)
        except ValueError as e:
            errors.append(str(e))
        if self.session_timeout < 0:
            errors.append(f"gateway.session_timeout must be >= 0, got {self.session_timeout}")
        if self.max_concurrent < 1:
            errors.append(f"gateway.max_concurrent must be >= 1, got {self.max_concurrent}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GatewayConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ──────────────────────────────────────────────────────────────────────────────
# Channel Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChannelConfig:
    """
    Configuration for a specific messaging channel.

    Attributes
    ----------
    type:
        Channel type (e.g., ``"whatsapp"``, ``"telegram"``).
    enabled:
        Whether this channel is enabled.
    credentials:
        Dict of credential fields (tokens, API keys, etc.).
    settings:
        Channel-specific settings (rate limits, preferences, etc.).
    """

    type: str = ""
    enabled: bool = False
    credentials: Dict[str, str] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.type:
            errors.append("channel.type must not be empty")
        return errors

    def to_dict(self, mask_secrets: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if mask_secrets:
            d["credentials"] = {
                k: mask_secret(v) for k, v in self.credentials.items()
            }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChannelConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ──────────────────────────────────────────────────────────────────────────────
# Security Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SecurityConfig:
    """
    Security and access control configuration.

    Attributes
    ----------
    audit_enabled:
        Whether to log security-relevant events.
    audit_log_path:
        Path to the audit log file.
    sandbox_type:
        Code execution sandbox type.
    dm_policy:
        Direct message access policy.
    admin_ids:
        List of admin user identifiers.
    allowed_user_ids:
        Allowlisted user identifiers (for ALLOWLIST DM policy).
    blocked_user_ids:
        Blocklisted user identifiers.
    tool_policies:
        Dict of tool_name -> access policy.
    file_policies:
        Dict of path_pattern -> access policy.
    max_file_size_mb:
        Maximum file upload size in megabytes.
    rate_limit_per_minute:
        Maximum requests per minute per user.
    """

    audit_enabled: bool = True
    audit_log_path: str = ""
    sandbox_type: str = "none"
    dm_policy: str = "open"
    admin_ids: List[str] = field(default_factory=list)
    allowed_user_ids: List[str] = field(default_factory=list)
    blocked_user_ids: List[str] = field(default_factory=list)
    tool_policies: Dict[str, str] = field(default_factory=dict)
    file_policies: Dict[str, str] = field(default_factory=dict)
    max_file_size_mb: int = 25
    rate_limit_per_minute: int = 30

    def validate(self) -> List[str]:
        errors: List[str] = []
        valid_policies = {"open", "allowlist", "blocklist", "admin_only", "disabled"}
        if self.dm_policy not in valid_policies:
            errors.append(f"security.dm_policy must be one of {valid_policies}, got {self.dm_policy!r}")
        if self.max_file_size_mb < 1:
            errors.append(f"security.max_file_size_mb must be >= 1, got {self.max_file_size_mb}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SecurityConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ──────────────────────────────────────────────────────────────────────────────
# Memory Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryConfig:
    """
    Memory and context management configuration.

    Attributes
    ----------
    enabled:
        Whether the memory system is enabled.
    backend:
        Memory storage backend type.
    max_context_tokens:
        Maximum tokens allocated for memory context injection.
    persistence_dir:
        Directory for persistent memory storage.
    auto_save:
        Whether to auto-save memory after each turn.
    search_enabled:
        Whether to enable memory search.
    max_entries:
        Maximum number of memory entries.
    """

    enabled: bool = True
    backend: str = "builtin"
    max_context_tokens: int = 4000
    persistence_dir: str = ""
    auto_save: bool = True
    search_enabled: bool = True
    max_entries: int = 1000

    def validate(self) -> List[str]:
        errors: List[str] = []
        valid_backends = {"builtin", "sqlite", "redis", "postgres", "chromadb", "mem0", "custom"}
        if self.backend not in valid_backends:
            errors.append(f"memory.backend must be one of {valid_backends}, got {self.backend!r}")
        if self.max_context_tokens < 0:
            errors.append(f"memory.max_context_tokens must be >= 0, got {self.max_context_tokens}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ──────────────────────────────────────────────────────────────────────────────
# Model Provider Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelProviderConfig:
    """
    Configuration for a specific model provider.

    Attributes
    ----------
    name:
        Provider name (e.g., ``"anthropic"``).
    api_key:
        API key (or environment variable reference).
    base_url:
        Custom API base URL.
    models:
        Dict of model_id -> model configuration.
    rate_limit_rpm:
        Requests per minute limit.
    rate_limit_tpm:
        Tokens per minute limit.
    timeout:
        Request timeout in seconds.
    max_retries:
        Maximum retry attempts on failure.
    """

    name: str = ""
    api_key: str = ""
    base_url: str = ""
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000
    timeout: int = 120
    max_retries: int = 3

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.name:
            errors.append("provider.name must not be empty")
        if self.timeout < 1:
            errors.append(f"provider.timeout must be >= 1, got {self.timeout}")
        return errors

    def to_dict(self, mask_secrets: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if mask_secrets:
            d["api_key"] = mask_secret(self.api_key)
        return d

    def resolve_api_key(self) -> str:
        """Resolve the API key, handling environment variable references."""
        if not self.api_key:
            return ""
        resolved = resolve_env_var(self.api_key, strict=False)
        if resolved.startswith("${"):
            # Try environment variable directly
            env_key = self.name.upper() + "_API_KEY"
            return os.environ.get(env_key, "")
        return resolved

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelProviderConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ──────────────────────────────────────────────────────────────────────────────
# Cron Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CronConfig:
    """
    Cron job scheduler configuration.

    Attributes
    ----------
    enabled:
        Whether the cron scheduler is enabled.
    data_dir:
        Directory for cron job data.
    timezone:
        Default timezone for scheduling.
    max_jobs:
        Maximum number of concurrent jobs.
    tick_interval:
        Scheduler tick interval in seconds.
    """

    enabled: bool = False
    data_dir: str = ""
    timezone: str = "UTC"
    max_jobs: int = 20
    tick_interval: float = 60.0

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.max_jobs < 1:
            errors.append(f"cron.max_jobs must be >= 1, got {self.max_jobs}")
        if self.tick_interval < 1.0:
            errors.append(f"cron.tick_interval must be >= 1.0, got {self.tick_interval}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CronConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ──────────────────────────────────────────────────────────────────────────────
# Skills Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillsConfig:
    """
    Skills system configuration.

    Attributes
    ----------
    enabled:
        Whether the skills system is enabled.
    dirs:
        List of directories to search for skills.
    marketplace_url:
        URL for the skills marketplace.
    auto_load:
        Whether to auto-load discovered skills.
    max_concurrent:
        Maximum concurrent skill executions.
    """

    enabled: bool = True
    dirs: List[str] = field(default_factory=list)
    marketplace_url: str = ""
    auto_load: bool = True
    max_concurrent: int = 5

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.max_concurrent < 1:
            errors.append(f"skills.max_concurrent must be >= 1, got {self.max_concurrent}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillsConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ──────────────────────────────────────────────────────────────────────────────
# Media Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MediaSubConfig:
    """Configuration for a single media generation service."""

    enabled: bool = False
    provider: str = ""
    api_key: str = ""
    base_url: str = ""
    default_model: str = ""

    def to_dict(self, mask_secrets: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if mask_secrets:
            d["api_key"] = mask_secret(self.api_key)
        return d


@dataclass
class MediaConfig:
    """
    Media generation and processing configuration.

    Attributes
    ----------
    image_gen:
        Image generation service configuration.
    video_gen:
        Video generation service configuration.
    music_gen:
        Music generation service configuration.
    tts:
        Text-to-speech configuration.
    stt:
        Speech-to-text configuration.
    vision:
        Vision/image analysis configuration.
    max_file_size_mb:
        Maximum media file size in megabytes.
    """

    image_gen: MediaSubConfig = field(default_factory=MediaSubConfig)
    video_gen: MediaSubConfig = field(default_factory=MediaSubConfig)
    music_gen: MediaSubConfig = field(default_factory=MediaSubConfig)
    tts: MediaSubConfig = field(default_factory=lambda: MediaSubConfig(
        enabled=False, provider="edge", default_model="en-US-AriaNeural",
    ))
    stt: MediaSubConfig = field(default_factory=MediaSubConfig)
    vision: MediaSubConfig = field(default_factory=lambda: MediaSubConfig(
        enabled=True, provider="builtin", default_model="claude-sonnet-4-20250514",
    ))
    max_file_size_mb: int = 50

    def validate(self) -> List[str]:
        return []

    def to_dict(self, mask_secrets: bool = False) -> Dict[str, Any]:
        return {
            "image_gen": self.image_gen.to_dict(mask_secrets),
            "video_gen": self.video_gen.to_dict(mask_secrets),
            "music_gen": self.music_gen.to_dict(mask_secrets),
            "tts": self.tts.to_dict(mask_secrets),
            "stt": self.stt.to_dict(mask_secrets),
            "vision": self.vision.to_dict(mask_secrets),
            "max_file_size_mb": self.max_file_size_mb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MediaConfig:
        sub_fields = {f.name for f in cls.__dataclass_fields__.values()}

        def _parse_sub(name: str, default: Any = None) -> Any:
            if name in data and isinstance(data[name], dict):
                return MediaSubConfig(**{
                    k: v for k, v in data[name].items()
                    if k in {f.name for f in MediaSubConfig.__dataclass_fields__.values()}
                })
            return default or MediaSubConfig()

        return cls(
            image_gen=_parse_sub("image_gen"),
            video_gen=_parse_sub("video_gen"),
            music_gen=_parse_sub("music_gen"),
            tts=_parse_sub("tts", MediaSubConfig(enabled=False, provider="edge", default_model="en-US-AriaNeural")),
            stt=_parse_sub("stt"),
            vision=_parse_sub("vision", MediaSubConfig(enabled=True, provider="builtin", default_model="claude-sonnet-4-20250514")),
            max_file_size_mb=data.get("max_file_size_mb", 50),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Canvas Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CanvasConfig:
    """
    Canvas/whiteboard feature configuration.

    Attributes
    ----------
    enabled:
        Whether the canvas feature is enabled.
    host:
        Canvas server bind host.
    port:
        Canvas server bind port.
    max_canvases:
        Maximum concurrent canvases per user.
    default_size:
        Default canvas size (width x height).
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8081
    max_canvases: int = 5
    default_size: str = "1024x768"

    def validate(self) -> List[str]:
        errors: List[str] = []
        try:
            validate_port(self.port)
        except ValueError as e:
            errors.append(str(e))
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanvasConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ──────────────────────────────────────────────────────────────────────────────
# App Configuration (Root)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AppConfig:
    """
    Root application configuration combining all sub-configs.

    This is the top-level configuration object that is created from
    merged defaults, file config, environment variables, and CLI args.

    Attributes
    ----------
    agent:
        AI agent configuration.
    gateway:
        Messaging gateway configuration.
    channels:
        Dict of channel name -> ChannelConfig.
    security:
        Security and access control configuration.
    memory:
        Memory system configuration.
    providers:
        Dict of provider name -> ModelProviderConfig.
    cron:
        Cron scheduler configuration.
    skills:
        Skills system configuration.
    media:
        Media generation configuration.
    canvas:
        Canvas feature configuration.
    log_level:
        Application-wide log level.
    log_format:
        Log output format.
    data_dir:
        Root data directory.
    config_dir:
        Configuration directory.
    cache_dir:
        Cache directory.
    """

    agent: AgentConfig = field(default_factory=AgentConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    channels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cron: CronConfig = field(default_factory=CronConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    log_level: str = "INFO"
    log_format: str = "text"
    data_dir: str = ""
    config_dir: str = ""
    cache_dir: str = ""

    def validate(self) -> List[str]:
        """
        Validate the entire configuration.

        Returns
        -------
        list[str]
            A list of validation error messages. Empty list means valid.
        """
        errors: List[str] = []

        # Validate each sub-config
        sub_configs = [
            ("agent", self.agent),
            ("gateway", self.gateway),
            ("security", self.security),
            ("memory", self.memory),
            ("cron", self.cron),
            ("skills", self.skills),
            ("media", self.media),
            ("canvas", self.canvas),
        ]

        for name, sub in sub_configs:
            sub_errors = sub.validate()
            for err in sub_errors:
                errors.append(f"{name}.{err}")

        # Validate log level
        try:
            LogLevel.from_string(self.log_level)
        except ValueError:
            errors.append(f"log_level must be one of: {', '.join(LogLevel.all_values())}")

        # Validate log format
        try:
            LogFormat.from_string(self.log_format)
        except ValueError:
            errors.append(f"log_format must be one of: {', '.join(LogFormat.all_values())}")

        return errors

    def to_dict(self, mask_secrets: bool = False) -> Dict[str, Any]:
        """
        Serialize the configuration to a dictionary.

        Parameters
        ----------
        mask_secrets:
            If ``True``, mask API keys and other secrets.

        Returns
        -------
        dict
        """
        return {
            "agent": self.agent.to_dict(),
            "gateway": self.gateway.to_dict(),
            "channels": self.channels,
            "security": self.security.to_dict(),
            "memory": self.memory.to_dict(),
            "providers": {
                k: ModelProviderConfig.from_dict(v).to_dict(mask_secrets)
                for k, v in self.providers.items()
            } if mask_secrets else self.providers,
            "cron": self.cron.to_dict(),
            "skills": self.skills.to_dict(),
            "media": self.media.to_dict(mask_secrets),
            "canvas": self.canvas.to_dict(),
            "log_level": self.log_level,
            "log_format": self.log_format,
            "data_dir": self.data_dir,
            "config_dir": self.config_dir,
            "cache_dir": self.cache_dir,
        }

    def apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides to the configuration.

        Reads ``ATLAS_<SECTION>_<KEY>`` environment variables and
        overrides the corresponding configuration values.

        Supports ``${VAR}`` substitution in string values.
        """
        # Agent overrides
        self.agent.model = os.environ.get("ATLAS_AGENT_MODEL", self.agent.model)
        self.agent.provider = os.environ.get("ATLAS_AGENT_PROVIDER", self.agent.provider)
        self.agent.temperature = parse_float(
            os.environ.get("ATLAS_AGENT_TEMPERATURE"), self.agent.temperature,
            minimum=0.0, maximum=2.0,
        )
        self.agent.max_tokens = parse_int(
            os.environ.get("ATLAS_AGENT_MAX_TOKENS"), self.agent.max_tokens, minimum=1,
        )

        # Gateway overrides
        self.gateway.host = os.environ.get("ATLAS_GATEWAY_HOST", self.gateway.host)
        self.gateway.port = parse_int(
            os.environ.get("ATLAS_GATEWAY_PORT"), self.gateway.port,
        )
        self.gateway.enabled = parse_bool(
            os.environ.get("ATLAS_GATEWAY_ENABLED", str(self.gateway.enabled)),
        )

        # Logging overrides
        self.log_level = os.environ.get("ATLAS_LOG_LEVEL", self.log_level)
        self.log_format = os.environ.get("ATLAS_LOG_FORMAT", self.log_format)

        # Directory overrides
        self.data_dir = os.environ.get("ATLAS_DATA_DIR", self.data_dir)
        self.config_dir = os.environ.get("ATLAS_CONFIG_DIR", self.config_dir)
        self.cache_dir = os.environ.get("ATLAS_CACHE_DIR", self.cache_dir)

        # Resolve env var references in all string fields
        self._resolve_env_refs()

    def _resolve_env_refs(self) -> None:
        """Resolve ``${VAR}`` references in all string configuration values."""
        self.agent.system_prompt = resolve_env_var(self.agent.system_prompt)
        self.gateway.tls_cert = resolve_env_var(self.gateway.tls_cert)
        self.gateway.tls_key = resolve_env_var(self.gateway.tls_key)
        self.memory.persistence_dir = resolve_env_var(self.memory.persistence_dir)
        self.cron.data_dir = resolve_env_var(self.cron.data_dir)
        self.data_dir = resolve_env_var(self.data_dir)
        self.config_dir = resolve_env_var(self.config_dir)
        self.cache_dir = resolve_env_var(self.cache_dir)

        for provider_data in self.providers.values():
            if isinstance(provider_data, dict) and "api_key" in provider_data:
                provider_data["api_key"] = resolve_env_var(provider_data["api_key"])
            if isinstance(provider_data, dict) and "base_url" in provider_data:
                provider_data["base_url"] = resolve_env_var(provider_data["base_url"])

        for channel_data in self.channels.values():
            if isinstance(channel_data, dict) and "credentials" in channel_data:
                creds = channel_data["credentials"]
                if isinstance(creds, dict):
                    for key in creds:
                        if isinstance(creds[key], str):
                            creds[key] = resolve_env_var(creds[key])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppConfig:
        """
        Create an AppConfig from a dictionary.

        Missing keys use default values.

        Parameters
        ----------
        data:
            Configuration dictionary.

        Returns
        -------
        AppConfig
        """
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}

        # Parse sub-configs
        if "agent" in filtered and isinstance(filtered["agent"], dict):
            filtered["agent"] = AgentConfig.from_dict(filtered["agent"])
        if "gateway" in filtered and isinstance(filtered["gateway"], dict):
            filtered["gateway"] = GatewayConfig.from_dict(filtered["gateway"])
        if "security" in filtered and isinstance(filtered["security"], dict):
            filtered["security"] = SecurityConfig.from_dict(filtered["security"])
        if "memory" in filtered and isinstance(filtered["memory"], dict):
            filtered["memory"] = MemoryConfig.from_dict(filtered["memory"])
        if "cron" in filtered and isinstance(filtered["cron"], dict):
            filtered["cron"] = CronConfig.from_dict(filtered["cron"])
        if "skills" in filtered and isinstance(filtered["skills"], dict):
            filtered["skills"] = SkillsConfig.from_dict(filtered["skills"])
        if "media" in filtered and isinstance(filtered["media"], dict):
            filtered["media"] = MediaConfig.from_dict(filtered["media"])
        if "canvas" in filtered and isinstance(filtered["canvas"], dict):
            filtered["canvas"] = CanvasConfig.from_dict(filtered["canvas"])

        return cls(**filtered)


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def get_defaults() -> Dict[str, Any]:
    """
    Get the complete default configuration dictionary.

    Returns
    -------
    dict
        A dictionary with all default values for every configuration section.
    """
    config = AppConfig()
    return config.to_dict()


def merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge two configuration dictionaries.

    Values from ``override`` take precedence. Nested dictionaries are
    merged recursively. Lists and sets from ``override`` replace those
    in ``base`` entirely.

    Parameters
    ----------
    base:
        The base configuration dictionary.
    override:
        The override dictionary with values to apply on top.

    Returns
    -------
    dict
        A new dictionary with merged values.
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge(result[key], value)
        elif value is not None:
            result[key] = copy.deepcopy(value)
        # If value is None, keep the base value

    return result


def validate(config: Dict[str, Any]) -> List[str]:
    """
    Validate a configuration dictionary.

    Creates an ``AppConfig`` from the dictionary and runs all
    sub-config validators.

    Parameters
    ----------
    config:
        The configuration dictionary to validate.

    Returns
    -------
    list[str]
        A list of validation error messages.
    """
    app_config = AppConfig.from_dict(config)
    return app_config.validate()
