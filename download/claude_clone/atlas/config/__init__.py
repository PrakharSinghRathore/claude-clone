"""
Atlas Configuration — Multi-source configuration management.

Provides comprehensive configuration loading from files, environment variables,
and CLI arguments with validation, merging, secret resolution, and file watching.

Exports:
    AppConfig             – Root application configuration dataclass.
    AgentConfig           – AI agent behavior configuration.
    GatewayConfig         – Messaging gateway configuration.
    ChannelConfig         – Per-channel configuration.
    SecurityConfig        – Security and access control configuration.
    MemoryConfig          – Memory system configuration.
    ModelProviderConfig   – Model provider configuration.
    CronConfig            – Cron scheduler configuration.
    SkillsConfig          – Skills system configuration.
    MediaConfig           – Media generation configuration.
    CanvasConfig          – Canvas feature configuration.
    ConfigLoader          – Multi-source configuration loader.
    SecretResolver        – Secret and environment variable resolver.
    ConfigWatcher         – Configuration file change watcher.
    ConfigMigrator        – Configuration version migration.
    ProviderType          – AI model provider enum.
    ChannelType           – Messaging channel enum.
    MemoryBackend         – Memory storage backend enum.
    LogFormat             – Log output format enum.
    LogLevel              – Logging level enum.
    SandboxType           – Sandbox type enum.
    DMPolicy              – Direct message policy enum.
    get_defaults          – Get default configuration dictionary.
    merge                 – Deep-merge configuration dictionaries.
    validate              – Validate a configuration dictionary.
    resolve_env_var       – Resolve ``${VAR}`` references.
    parse_bool            – Parse boolean from various types.
    parse_int             – Parse integer with constraints.
    parse_float           – Parse float with constraints.
    parse_string_list     – Parse list from string or sequence.
    validate_port         – Validate network port number.
    validate_path         – Validate file system path.
    is_valid_url          – Validate URL format.
    mask_secret           – Mask secret for display.
"""

from .schema import (
    AppConfig,
    AgentConfig,
    GatewayConfig,
    ChannelConfig,
    SecurityConfig,
    MemoryConfig,
    ModelProviderConfig,
    CronConfig,
    SkillsConfig,
    MediaConfig,
    CanvasConfig,
    get_defaults,
    merge,
    validate,
)
from .loader import (
    ConfigLoader,
    SecretResolver,
    ConfigWatcher,
    ConfigMigrator,
)
from .types import (
    ProviderType,
    ChannelType,
    MemoryBackend,
    LogFormat,
    LogLevel,
    SandboxType,
    DMPolicy,
    resolve_env_var,
    parse_bool,
    parse_int,
    parse_float,
    parse_string_list,
    validate_port,
    validate_path,
    is_valid_url,
    mask_secret,
)

__all__ = [
    # Schema
    "AppConfig",
    "AgentConfig",
    "GatewayConfig",
    "ChannelConfig",
    "SecurityConfig",
    "MemoryConfig",
    "ModelProviderConfig",
    "CronConfig",
    "SkillsConfig",
    "MediaConfig",
    "CanvasConfig",
    "get_defaults",
    "merge",
    "validate",
    # Loader
    "ConfigLoader",
    "SecretResolver",
    "ConfigWatcher",
    "ConfigMigrator",
    # Types
    "ProviderType",
    "ChannelType",
    "MemoryBackend",
    "LogFormat",
    "LogLevel",
    "SandboxType",
    "DMPolicy",
    "resolve_env_var",
    "parse_bool",
    "parse_int",
    "parse_float",
    "parse_string_list",
    "validate_port",
    "validate_path",
    "is_valid_url",
    "mask_secret",
]
