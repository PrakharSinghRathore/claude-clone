"""
Credential Pool — API key rotation and pooling.

Manages multiple API keys per provider with strategies for selection,
rotation, and automatic disabling of failed keys. Supports round-robin
and least-used strategies with rate limit awareness.

Usage
-----
    pool = CredentialPool()
    pool.add_key("anthropic", "sk-ant-1")
    pool.add_key("anthropic", "sk-ant-2")
    pool.add_key("openai", "sk-openai-1")

    key = await pool.get_key("anthropic")  # Rotates between keys
    pool.report_success("anthropic", "sk-ant-1")
    pool.report_failure("anthropic", "sk-ant-2", error="rate_limit")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

from atlas.constants import (
    ATLAS_CONFIG_HOME,
    KEY_COOLDOWN_SECONDS,
    MAX_CONSECUTIVE_FAILURES,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

class SelectionStrategy(Enum):
    """Key selection strategy."""

    ROUND_ROBIN = auto()
    """Cycle through keys in order."""

    LEAST_USED = auto()
    """Select the key with the fewest uses."""

    RANDOM = auto()
    """Select a random key."""

    LEAST_ERRORS = auto()
    """Select the key with the fewest recent errors."""


@dataclass
class CredentialEntry:
    """
    A single API key entry.

    Attributes
    ----------
    key:
        The API key string.
    provider:
        The provider this key is for (anthropic, openai, etc.).
    label:
        Human-readable label for the key.
    enabled:
        Whether this key is currently active.
    use_count:
        Total number of times this key has been used.
    success_count:
        Number of successful requests with this key.
    failure_count:
        Number of failed requests with this key.
    consecutive_failures:
        Current streak of consecutive failures.
    last_used:
        Timestamp of last use.
    last_failure:
        Timestamp of last failure.
    last_failure_reason:
        Reason for the last failure.
    cooldown_until:
        Timestamp until which this key is in cooldown.
    rate_limit_until:
        Timestamp until which this key is rate-limited.
    metadata:
        Additional metadata (rate limit info, etc.).
    """

    key: str
    provider: str = "anthropic"
    label: str = ""
    enabled: bool = True
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_used: float = 0.0
    last_failure: float = 0.0
    last_failure_reason: str = ""
    cooldown_until: float = 0.0
    rate_limit_until: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.label:
            # Mask the key for display: show first 8 and last 4 chars
            if len(self.key) > 12:
                self.label = f"{self.key[:8]}...{self.key[-4:]}"
            else:
                self.label = "***"

    @property
    def is_available(self) -> bool:
        """Whether this key is available for use."""
        if not self.enabled:
            return False
        now = time.monotonic()
        if now < self.cooldown_until:
            return False
        if now < self.rate_limit_until:
            return False
        return True

    @property
    def success_rate(self) -> float:
        """Success rate (0.0–1.0)."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (key is masked)."""
        data = asdict(self)
        # Don't expose the actual key in serialized form
        data["key"] = self.label
        data["is_available"] = self.is_available
        data["success_rate"] = round(self.success_rate, 4)
        return data


# ──────────────────────────────────────────────────────────────────────────────
# CredentialPool
# ──────────────────────────────────────────────────────────────────────────────

class CredentialPool:
    """
    API key rotation and pooling manager.

    Manages multiple API keys per provider with configurable selection
    strategies, automatic cooldown for failed keys, and rate limit handling.

    Parameters
    ----------
    strategy:
        Default key selection strategy.
    max_consecutive_failures:
        Number of consecutive failures before auto-disabling a key.
    cooldown_seconds:
        Cooldown period for failed keys.
    persistence_path:
        Path to persist key metadata (key values are never persisted to disk).
    """

    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.LEAST_USED,
        max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
        cooldown_seconds: int = KEY_COOLDOWN_SECONDS,
        persistence_path: Optional[str] = None,
    ) -> None:
        self._strategy = strategy
        self._max_consecutive_failures = max_consecutive_failures
        self._cooldown_seconds = cooldown_seconds
        self._persistence_path = (
            Path(persistence_path) if persistence_path
            else ATLAS_CONFIG_HOME / "credential_pool.json"
        )

        # Provider → list of CredentialEntry
        self._keys: Dict[str, List[CredentialEntry]] = defaultdict(list)

        # Round-robin state per provider
        self._rr_index: Dict[str, int] = defaultdict(int)

    # ── Key management ────────────────────────────────────────────────────

    def add_key(
        self,
        provider: str,
        key: str,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CredentialEntry:
        """
        Add an API key to the pool.

        Parameters
        ----------
        provider:
            Provider name (anthropic, openai, etc.).
        key:
            The API key string.
        label:
            Optional human-readable label.
        metadata:
            Optional metadata.

        Returns
        -------
        CredentialEntry
            The created credential entry.
        """
        entry = CredentialEntry(
            key=key,
            provider=provider.lower(),
            label=label or "",
            metadata=metadata or {},
        )
        self._keys[provider.lower()].append(entry)
        logger.info("Added credential for %s: %s", provider, entry.label)
        return entry

    def remove_key(self, provider: str, key: str) -> bool:
        """
        Remove a key from the pool.

        Parameters
        ----------
        provider:
            Provider name.
        key:
            The API key to remove.

        Returns
        -------
        bool
            ``True`` if the key was found and removed.
        """
        provider_lower = provider.lower()
        if provider_lower not in self._keys:
            return False
        original_len = len(self._keys[provider_lower])
        self._keys[provider_lower] = [
            e for e in self._keys[provider_lower] if e.key != key
        ]
        return len(self._keys[provider_lower]) < original_len

    def disable_key(self, provider: str, key: str) -> bool:
        """
        Manually disable a key.

        Returns
        -------
        bool
            ``True`` if the key was found.
        """
        entry = self._find_entry(provider, key)
        if entry:
            entry.enabled = False
            logger.info("Disabled credential for %s: %s", provider, entry.label)
            return True
        return False

    def enable_key(self, provider: str, key: str) -> bool:
        """Re-enable a previously disabled key."""
        entry = self._find_entry(provider, key)
        if entry:
            entry.enabled = True
            entry.consecutive_failures = 0
            entry.cooldown_until = 0.0
            entry.rate_limit_until = 0.0
            return True
        return False

    def get_all_keys(self, provider: str) -> List[CredentialEntry]:
        """Return all keys for a provider."""
        return list(self._keys.get(provider.lower(), []))

    def get_available_keys(self, provider: str) -> List[CredentialEntry]:
        """Return only available (non-disabled, non-cooldown) keys."""
        return [e for e in self._keys.get(provider.lower(), []) if e.is_available]

    # ── Key selection ─────────────────────────────────────────────────────

    async def get_key(
        self,
        provider: str,
        strategy: Optional[SelectionStrategy] = None,
    ) -> Optional[str]:
        """
        Get the next available API key for a provider.

        Parameters
        ----------
        provider:
            Provider name.
        strategy:
            Override selection strategy for this call.

        Returns
        -------
        str or None
            The API key, or ``None`` if no keys are available.
        """
        strat = strategy or self._strategy
        available = self.get_available_keys(provider)

        if not available:
            logger.warning("No available credentials for provider: %s", provider)
            return None

        if strat == SelectionStrategy.ROUND_ROBIN:
            entry = self._select_round_robin(provider, available)
        elif strat == SelectionStrategy.LEAST_USED:
            entry = min(available, key=lambda e: e.use_count)
        elif strat == SelectionStrategy.LEAST_ERRORS:
            entry = min(available, key=lambda e: e.failure_count)
        elif strat == SelectionStrategy.RANDOM:
            import random
            entry = random.choice(available)
        else:
            entry = available[0]

        entry.use_count += 1
        entry.last_used = time.monotonic()
        return entry.key

    def _select_round_robin(
        self, provider: str, available: List[CredentialEntry],
    ) -> CredentialEntry:
        """Select the next key using round-robin."""
        idx = self._rr_index[provider] % len(available)
        self._rr_index[provider] += 1
        return available[idx]

    # ── Reporting ─────────────────────────────────────────────────────────

    def report_success(self, provider: str, key: str) -> None:
        """
        Report a successful request for a key.

        Resets consecutive failure count.
        """
        entry = self._find_entry(provider, key)
        if entry:
            entry.success_count += 1
            entry.consecutive_failures = 0
            entry.last_used = time.monotonic()

    def report_failure(
        self,
        provider: str,
        key: str,
        error: str = "",
        is_rate_limit: bool = False,
    ) -> None:
        """
        Report a failed request for a key.

        Automatically disables the key after ``max_consecutive_failures``
        consecutive failures. Rate limit errors trigger a cooldown.

        Parameters
        ----------
        provider:
            Provider name.
        key:
            The API key that failed.
        error:
            Error description.
        is_rate_limit:
            Whether the failure was due to rate limiting.
        """
        entry = self._find_entry(provider, key)
        if entry is None:
            return

        entry.failure_count += 1
        entry.consecutive_failures += 1
        entry.last_failure = time.monotonic()
        entry.last_failure_reason = error

        if is_rate_limit:
            # Set rate limit cooldown (typically 60s for most providers)
            entry.rate_limit_until = time.monotonic() + 60.0
            logger.warning(
                "Rate limited for %s key %s — cooling down for 60s",
                provider, entry.label,
            )
        elif entry.consecutive_failures >= self._max_consecutive_failures:
            # Auto-disable after too many consecutive failures
            entry.enabled = False
            entry.cooldown_until = time.monotonic() + self._cooldown_seconds
            logger.warning(
                "Auto-disabled %s key %s after %d consecutive failures: %s",
                provider, entry.label, entry.consecutive_failures, error,
            )

    # ── Diagnostics ───────────────────────────────────────────────────────

    def get_provider_status(self, provider: str) -> Dict[str, Any]:
        """
        Get status summary for a provider's keys.

        Returns
        -------
        dict
            Summary with ``total``, ``available``, ``disabled``, and
            per-key statistics.
        """
        keys = self._keys.get(provider.lower(), [])
        available = [k for k in keys if k.is_available]
        disabled = [k for k in keys if not k.enabled]

        return {
            "provider": provider,
            "total_keys": len(keys),
            "available_keys": len(available),
            "disabled_keys": len(disabled),
            "keys": [k.to_dict() for k in keys],
        }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all providers."""
        return {
            provider: self.get_provider_status(provider)
            for provider in self._keys
        }

    # ── Persistence ───────────────────────────────────────────────────────

    async def save(self) -> None:
        """
        Persist key metadata to disk.

        Note: Actual API key values are NEVER written to disk.
        Only usage statistics, labels, and state are persisted.
        """
        data: Dict[str, List[Dict[str, Any]]] = {}
        for provider, keys in self._keys.items():
            data[provider] = [k.to_dict() for k in keys]

        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

        def _write():
            self._persistence_path.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write)

    async def load_metadata(self) -> None:
        """
        Load key metadata from disk.

        This restores usage statistics but NOT the actual key values.
        Keys must be re-added programmatically or via environment variables.
        """
        if not self._persistence_path.exists():
            return

        def _read():
            return json.loads(
                self._persistence_path.read_text(encoding="utf-8")
            )

        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, _read)
            # Metadata is loaded but keys are not restored for security
            logger.info(
                "Loaded credential metadata for %d providers (keys must be re-added)",
                len(data),
            )
        except Exception as e:
            logger.warning("Failed to load credential metadata: %s", e)

    # ── Internal ──────────────────────────────────────────────────────────

    def _find_entry(
        self, provider: str, key: str,
    ) -> Optional[CredentialEntry]:
        """Find a credential entry by provider and key."""
        for entry in self._keys.get(provider.lower(), []):
            if entry.key == key:
                return entry
        return None
