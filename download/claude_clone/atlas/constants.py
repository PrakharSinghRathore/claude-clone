"""
Shared constants for the Atlas Agent system.

Defines paths, version information, default configurations, and module-wide
constants used across all Atlas core components.
"""

from __future__ import annotations

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Version & Identity
# ──────────────────────────────────────────────────────────────────────────────

ATLAS_VERSION: str = "0.1.0"
ATLAS_NAME: str = "Atlas Agent Core"
ATLAS_MODULE_NAME: str = "atlas"

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

# Base directory for all Atlas data (respects XDG_DATA_HOME)
ATLAS_DATA_HOME: Path = Path(
    os.environ.get("ATLAS_DATA_HOME", "")
    or os.environ.get("XDG_DATA_HOME", "")
    or (Path.home() / ".local" / "share")
) / "atlas"

# Configuration directory (respects XDG_CONFIG_HOME)
ATLAS_CONFIG_HOME: Path = Path(
    os.environ.get("ATLAS_CONFIG_HOME", "")
    or os.environ.get("XDG_CONFIG_HOME", "")
    or (Path.home() / ".config")
) / "atlas"

# Cache directory (respects XDG_CACHE_HOME)
ATLAS_CACHE_HOME: Path = Path(
    os.environ.get("ATLAS_CACHE_HOME", "")
    or os.environ.get("XDG_CACHE_HOME", "")
    or (Path.home() / ".cache")
) / "atlas"

# ──────────────────────────────────────────────────────────────────────────────
# Subdirectory constants
# ──────────────────────────────────────────────────────────────────────────────

# Built-in memory files directory
MEMORY_DIR: str = "memory"
MEMORY_FILE: str = "MEMORY.md"
USER_PROFILE_FILE: str = "USER.md"
SESSIONS_DIR: str = "sessions"

# Trajectory recording directory
TRAJECTORY_DIR: str = "trajectories"

# Insights / analytics directory
INSIGHTS_DIR: str = "insights"

# Credential storage
CREDENTIALS_FILE: str = "credentials.json"

# ──────────────────────────────────────────────────────────────────────────────
# Token & Context Defaults
# ──────────────────────────────────────────────────────────────────────────────

# Default maximum context window (tokens) for compression target
DEFAULT_CONTEXT_WINDOW: int = 200_000

# Reserve this many tokens for the system prompt + response
CONTEXT_RESERVE_TOKENS: int = 8_000

# Maximum tokens allocated for memory context injection
MAX_MEMORY_CONTEXT_TOKENS: int = 4_000

# Maximum tokens for session search results in context
MAX_SESSION_SEARCH_TOKENS: int = 2_000

# Default summarization target (tokens) for compressed conversation turns
SUMMARIZATION_TARGET_TOKENS: int = 500

# ──────────────────────────────────────────────────────────────────────────────
# Compression
# ──────────────────────────────────────────────────────────────────────────────

# Number of recent turns to always preserve (never compressed)
PRESERVE_RECENT_TURNS: int = 6

# Number of system messages to always preserve
PRESERVE_SYSTEM_MESSAGES: int = 5

# Minimum turns before compression is triggered
MIN_TURNS_BEFORE_COMPRESSION: int = 10

# ──────────────────────────────────────────────────────────────────────────────
# Title Generation
# ──────────────────────────────────────────────────────────────────────────────

MAX_TITLE_LENGTH: int = 60
MIN_TITLE_LENGTH: int = 10
TITLE_GENERATION_MAX_TOKENS: int = 30

# ──────────────────────────────────────────────────────────────────────────────
# Usage & Pricing
# ──────────────────────────────────────────────────────────────────────────────

COST_WARNING_THRESHOLD: float = 1.0  # USD
COST_TRACKING_RETENTION_DAYS: int = 90
MAX_DAILY_BUDGET: float = 50.0  # USD

# ──────────────────────────────────────────────────────────────────────────────
# Credential Pool
# ──────────────────────────────────────────────────────────────────────────────

# Disable a key after this many consecutive failures
MAX_CONSECUTIVE_FAILURES: int = 5

# Cooldown period for failed keys (seconds)
KEY_COOLDOWN_SECONDS: int = 300

# ──────────────────────────────────────────────────────────────────────────────
# Smart Routing
# ──────────────────────────────────────────────────────────────────────────────

ROUTING_LATENCY_WEIGHT: float = 0.3
ROUTING_COST_WEIGHT: float = 0.4
ROUTING_QUALITY_WEIGHT: float = 0.3

# ──────────────────────────────────────────────────────────────────────────────
# PII Redaction
# ──────────────────────────────────────────────────────────────────────────────

REDACTION_PLACEHOLDER_PREFIX: str = "[REDACTED"
REDACTION_MAP_FILE: str = "redaction_map.json"
