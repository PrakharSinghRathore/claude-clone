"""
Atlas Configuration Types — Shared type definitions for the configuration system.

Defines enumerations for provider types, channel types, memory backends,
logging formats, and log levels. Includes helper functions for type
conversion, validation, and lookup.

Usage::

    from atlas.config.types import ProviderType, ChannelType, MemoryBackend

    provider = ProviderType.from_string("anthropic")
    assert provider == ProviderType.ANTHROPIC
"""

from __future__ import annotations

import enum
import logging
import os
import re
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Provider Types
# ──────────────────────────────────────────────────────────────────────────────

class ProviderType(enum.Enum):
    """
    Supported AI model providers.

    Each value maps to a canonical string identifier used in configuration
    files, environment variables, and API routing.
    """

    ANTHROPIC = "anthropic"
    """Anthropic Claude models."""

    OPENAI = "openai"
    """OpenAI GPT models."""

    GOOGLE = "google"
    """Google Gemini models."""

    XAI = "xai"
    """xAI Grok models."""

    OLLAMA = "ollama"
    """Ollama local models."""

    MISTRAL = "mistral"
    """Mistral AI models."""

    GROQ = "groq"
    """Groq fast inference."""

    DEEPSEEK = "deepseek"
    """DeepSeek models."""

    TOGETHER = "together"
    """Together AI models."""

    FIREWORKS = "fireworks"
    """Fireworks AI models."""

    OPENROUTER = "openrouter"
    """OpenRouter aggregation."""

    BEDROCK = "bedrock"
    """AWS Bedrock hosted models."""

    CUSTOM = "custom"
    """Custom/self-hosted provider endpoint."""

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> ProviderType:
        """
        Parse a provider type from a string.

        Parameters
        ----------
        value:
            Case-insensitive provider name (e.g., ``"Anthropic"``, ``"ANTHROPIC"``).

        Returns
        -------
        ProviderType

        Raises
        ------
        ValueError
            If the provider name is not recognized.
        """
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        # Alias support
        aliases: Dict[str, str] = {
            "claude": "anthropic",
            "gpt": "openai",
            "gemini": "google",
            "grok": "xai",
            "local": "ollama",
            "mistralai": "mistral",
            "bedrock_anthropic": "bedrock",
            "aws": "bedrock",
        }
        resolved = aliases.get(normalized)
        if resolved:
            for member in cls:
                if member.value == resolved:
                    return member
        raise ValueError(
            f"Unknown provider type: {value!r}. "
            f"Valid types: {', '.join(m.value for m in cls)}"
        )

    @classmethod
    def all_values(cls) -> List[str]:
        """Return all valid provider type strings."""
        return [m.value for m in cls]

    @property
    def env_prefix(self) -> str:
        """Environment variable prefix for this provider (e.g., ``ANTHROPIC_``)."""
        return self.value.upper() + "_"

    @property
    def requires_api_key(self) -> bool:
        """Whether this provider requires an API key."""
        return self not in (ProviderType.OLLAMA, ProviderType.CUSTOM)

    @property
    def is_cloud(self) -> bool:
        """Whether this is a cloud-hosted provider."""
        return self not in (ProviderType.OLLAMA, ProviderType.CUSTOM)


# ──────────────────────────────────────────────────────────────────────────────
# Channel Types
# ──────────────────────────────────────────────────────────────────────────────

class ChannelType(enum.Enum):
    """
    Supported messaging channels/platforms.

    Each value maps to a canonical identifier used in gateway configuration,
    session keys, and platform adapter selection.
    """

    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    SIGNAL = "signal"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    MATRIX = "matrix"
    DINGTALK = "dingtalk"
    FEISHU = "feishu"
    WECOM = "wecom"
    MATTERMOST = "mattermost"
    API = "api"
    WEB = "web"
    CLI = "cli"
    DESKTOP = "desktop"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> ChannelType:
        """
        Parse a channel type from a string.

        Parameters
        ----------
        value:
            Case-insensitive channel name.

        Returns
        -------
        ChannelType

        Raises
        ------
        ValueError
            If the channel name is not recognized.
        """
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        # Aliases
        aliases: Dict[str, str] = {
            "wechat": "wecom",
            "wechat_work": "wecom",
            "lark": "feishu",
            "twilio": "sms",
            "http": "webhook",
            "rest": "api",
            "terminal": "cli",
        }
        resolved = aliases.get(normalized)
        if resolved:
            for member in cls:
                if member.value == resolved:
                    return member
        raise ValueError(
            f"Unknown channel type: {value!r}. "
            f"Valid types: {', '.join(m.value for m in cls)}"
        )

    @classmethod
    def all_values(cls) -> List[str]:
        """Return all valid channel type strings."""
        return [m.value for m in cls]

    @classmethod
    def messaging_channels(cls) -> FrozenSet[ChannelType]:
        """Return the set of two-way messaging channel types."""
        return frozenset({
            cls.WHATSAPP, cls.TELEGRAM, cls.SLACK, cls.DISCORD,
            cls.SIGNAL, cls.EMAIL, cls.SMS, cls.MATRIX,
            cls.DINGTALK, cls.FEISHU, cls.WECOM, cls.MATTERMOST,
        })


# ──────────────────────────────────────────────────────────────────────────────
# Memory Backends
# ──────────────────────────────────────────────────────────────────────────────

class MemoryBackend(enum.Enum):
    """
    Supported memory storage backends.

    Determines where conversation memory and context are persisted
    for long-term recall.
    """

    BUILTIN = "builtin"
    """Built-in file-based memory (MEMORY.md, JSON)."""

    SQLITE = "sqlite"
    """SQLite database backend."""

    REDIS = "redis"
    """Redis in-memory store."""

    POSTGRES = "postgres"
    """PostgreSQL database."""

    CHROMADB = "chromadb"
    """ChromaDB vector database."""

    MEM0 = "mem0"
    """Mem0 memory service."""

    CUSTOM = "custom"
    """Custom memory backend implementation."""

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> MemoryBackend:
        """
        Parse a memory backend from a string.

        Raises
        ------
        ValueError
            If the backend name is not recognized.
        """
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        aliases: Dict[str, str] = {
            "file": "builtin",
            "json": "builtin",
            "postgresql": "postgres",
            "chroma": "chromadb",
        }
        resolved = aliases.get(normalized)
        if resolved:
            for member in cls:
                if member.value == resolved:
                    return member
        raise ValueError(
            f"Unknown memory backend: {value!r}. "
            f"Valid types: {', '.join(m.value for m in cls)}"
        )

    @property
    def requires_external(self) -> bool:
        """Whether this backend requires an external service."""
        return self not in (MemoryBackend.BUILTIN, MemoryBackend.SQLITE)


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

class LogFormat(enum.Enum):
    """Supported log output formats."""

    TEXT = "text"
    """Plain text log format."""

    JSON = "json"
    """Structured JSON log format (one JSON object per line)."""

    PRETTY = "pretty"
    """Colored, human-readable format."""

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> LogFormat:
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown log format: {value!r}")


class LogLevel(enum.Enum):
    """Standard logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> LogLevel:
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        # Python logging names
        py_names = {"warn": "warning", "fatal": "critical"}
        normalized = py_names.get(normalized, normalized)
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown log level: {value!r}")

    @property
    def python_level(self) -> int:
        """Map to Python's ``logging`` module level constants."""
        mapping = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        return mapping.get(self.value, logging.INFO)


# ──────────────────────────────────────────────────────────────────────────────
# Sandbox Types
# ──────────────────────────────────────────────────────────────────────────────

class SandboxType(enum.Enum):
    """Code execution sandbox types."""

    NONE = "none"
    """No sandboxing (direct execution)."""

    DOCKER = "docker"
    """Docker container sandbox."""

    PROCESS = "process"
    """Subprocess isolation."""

    NSJAIL = "nsjail"
    """nsjail-based sandbox."""

    FIREJAIL = "firejail"
    """Firejail sandbox."""

    def __str__(self) -> str:
        return self.value


# ──────────────────────────────────────────────────────────────────────────────
# DM Policy Types
# ──────────────────────────────────────────────────────────────────────────────

class DMPolicy(enum.Enum):
    """Direct message access policy."""

    OPEN = "open"
    """Anyone can send direct messages."""

    ALLOWLIST = "allowlist"
    """Only allowlisted users can send DMs."""

    BLOCKLIST = "blocklist"
    """All users except blocklisted ones can send DMs."""

    ADMIN_ONLY = "admin_only"
    """Only admin users can send DMs."""

    DISABLED = "disabled"
    """Direct messages are disabled entirely."""

    def __str__(self) -> str:
        return self.value


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def parse_bool(value: Any) -> bool:
    """
    Parse a boolean from various input types.

    Accepts: bool, int (0/1), str (``"true"/"1"/"yes"/"on"/"false"/"0"/"no"/"off"``).

    Parameters
    ----------
    value:
        The value to parse.

    Returns
    -------
    bool
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on", "enabled")
    return False


def parse_int(value: Any, default: int = 0, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    """
    Parse an integer from various input types.

    Parameters
    ----------
    value:
        The value to parse.
    default:
        Default value if parsing fails.
    minimum:
        Optional minimum value constraint.
    maximum:
        Optional maximum value constraint.

    Returns
    -------
    int
    """
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        result = max(result, minimum)
    if maximum is not None:
        result = min(result, maximum)
    return result


def parse_float(value: Any, default: float = 0.0, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    """
    Parse a float from various input types.

    Parameters
    ----------
    value:
        The value to parse.
    default:
        Default value if parsing fails.
    minimum:
        Optional minimum value constraint.
    maximum:
        Optional maximum value constraint.

    Returns
    -------
    float
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        result = max(result, minimum)
    if maximum is not None:
        result = min(result, maximum)
    return result


def parse_string_list(value: Any, separator: str = ",") -> List[str]:
    """
    Parse a list of strings from various input types.

    Accepts: list, tuple, comma-separated string.

    Parameters
    ----------
    value:
        The value to parse.
    separator:
        String separator for split operations.

    Returns
    -------
    list[str]
    """
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(separator) if item.strip()]
    return []


def validate_port(port: Any) -> int:
    """
    Validate and parse a network port number.

    Parameters
    ----------
    port:
        The port value to validate.

    Returns
    -------
    int
        Validated port number (1-65535).

    Raises
    ------
    ValueError
        If the port is out of range.
    """
    p = parse_int(port, default=0, minimum=1, maximum=65535)
    if p == 0:
        raise ValueError(f"Invalid port number: {port!r}. Must be between 1 and 65535.")
    return p


def validate_path(path: Any, must_exist: bool = False, must_be_dir: bool = False) -> str:
    """
    Validate and normalize a file system path.

    Parameters
    ----------
    path:
        The path to validate.
    must_exist:
        If ``True``, raise ``ValueError`` if the path doesn't exist.
    must_be_dir:
        If ``True``, raise ``ValueError`` if the path is not a directory.

    Returns
    -------
    str
        The normalized path string.

    Raises
    ------
    ValueError
        If validation fails.
    """
    if not path or not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"Invalid path: {path!r}")

    from pathlib import Path as P
    normalized = str(P(str(path)).expanduser().resolve())

    if must_exist and not os.path.exists(normalized):
        raise ValueError(f"Path does not exist: {normalized!r}")
    if must_be_dir and not os.path.isdir(normalized):
        raise ValueError(f"Path is not a directory: {normalized!r}")

    return normalized


def resolve_env_var(template: str, *, strict: bool = False) -> str:
    """
    Resolve environment variable references in a string template.

    Supports ``${VAR}`` and ``${VAR:default}`` syntax.

    Parameters
    ----------
    template:
        A string possibly containing ``${VAR}`` or ``${VAR:default}`` references.
    strict:
        If ``True``, raise ``ValueError`` for undefined variables without defaults.

    Returns
    -------
    str
        The resolved string with all references replaced.

    Example
    -------
    >>> os.environ["TEST_VAR"] = "hello"
    >>> resolve_env_var("${TEST_VAR}")
    'hello'
    >>> resolve_env_var("${MISSING:fallback}")
    'fallback'
    """
    if not isinstance(template, str):
        return str(template)

    pattern = re.compile(r"\$\{([^}:]+):([^}]*)\}|\$\{([^}]+)\}")

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1) or match.group(3)
        default_value = match.group(2) if match.group(2) is not None else ""

        env_value = os.environ.get(var_name)
        if env_value is not None:
            return env_value

        if match.group(2) is not None:
            return default_value

        if strict:
            raise ValueError(f"Environment variable not found: {var_name!r}")

        return match.group(0)  # Leave unresolved

    return pattern.sub(replacer, template)


def is_valid_url(url: str) -> bool:
    """
    Validate that a string is a well-formed URL.

    Parameters
    ----------
    url:
        The URL string to validate.

    Returns
    -------
    bool
    """
    if not url or not isinstance(url, str):
        return False
    url_pattern = re.compile(
        r"^(https?|wss?|tcp|udp|file)://"
        r"[^\s/$.?#].[^\s]*$",
        re.IGNORECASE,
    )
    return bool(url_pattern.match(url.strip()))


def mask_secret(value: Optional[str], visible_chars: int = 4) -> str:
    """
    Mask a secret string for safe display.

    Parameters
    ----------
    value:
        The secret to mask.
    visible_chars:
        Number of trailing characters to keep visible.

    Returns
    -------
    str
        The masked string (e.g., ``"****abcd"``).
    """
    if not value:
        return "(not set)"
    if len(value) <= visible_chars:
        return "****"
    return "*" * 4 + value[-visible_chars:]
