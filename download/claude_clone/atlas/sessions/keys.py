"""
Session Key Derivation — Deterministic session key generation.

Provides SHA-256-based key derivation for sessions, ensuring consistent
identifiers across restarts and processes. Handles case-insensitive and
whitespace-normalized identifiers for channels, accounts, and peers.

Usage::

    from atlas.sessions.keys import SessionKeyDerivation

    deriv = SessionKeyDerivation()
    key = deriv.derive("whatsapp", "+1234567890", "user@example.com")
"""

from __future__ import annotations

import hashlib
import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_KEY_SEPARATOR = "::"
_PREFIX = "atlas:session:v1"


class KeyScope(Enum):
    """Scope for session key derivation."""

    DIRECT = "direct"
    """One-to-one session between an agent and a peer."""

    GROUP = "group"
    """Group session involving multiple participants."""


class ChannelNormalizer:
    """
    Normalizes channel identifiers for consistent key derivation.

    Handles case normalization, whitespace trimming, and platform-specific
    cleanup rules for various channel types.
    """

    # Known channel types that need special normalization
    _CHANNEL_PATTERNS: dict[str, re.Pattern[str]] = {
        "whatsapp": re.compile(r"^\+?(\d+)$"),
        "telegram": re.compile(r"^@?(\w{5,})$"),
        "discord": re.compile(r"^<?@?&?!?(\d{15,21})>?$"),
        "slack": re.compile(r"^[A-Z][A-Z0-9]+$"),
        "signal": re.compile(r"^\+?(\d{10,15})$"),
        "sms": re.compile(r"^\+?(\d{10,15})$"),
        "email": re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"),
        "matrix": re.compile(r"^@?[\w./:-]+:[\w.-]+$"),
        "dingtalk": re.compile(r"^[\w@.]+$"),
        "feishu": re.compile(r"^ou_[a-f0-9]+$"),
        "wecom": re.compile(r"^[a-zA-Z0-9-]+$"),
        "mattermost": re.compile(r"^[a-z0-9]+[a-z0-9_-]*$"),
    }

    @classmethod
    def normalize_channel_type(cls, channel_type: str) -> str:
        """
        Normalize a channel type string to lowercase, trimmed form.

        Parameters
        ----------
        channel_type:
            Raw channel type (e.g., ``"WhatsApp"``, ``"  telegram  "``).

        Returns
        -------
        str
            Normalized channel type (e.g., ``"whatsapp"``, ``"telegram"``).
        """
        if not channel_type:
            return "unknown"
        normalized = channel_type.strip().lower().replace("-", "_").replace(" ", "_")
        # Remove common prefixes/suffixes
        normalized = re.sub(r"^(platform_|channel_|msg_)", "", normalized)
        return normalized

    @classmethod
    def normalize_identifier(cls, channel_type: str, identifier: str) -> str:
        """
        Normalize an identifier based on channel-specific rules.

        Applies pattern matching for known channel types, stripping
        prefixes, suffixes, and normalizing format. Falls back to
        lowercase whitespace-trimmed string for unknown channels.

        Parameters
        ----------
        channel_type:
            The channel type (used to select normalization rules).
        identifier:
            The raw identifier to normalize.

        Returns
        -------
        str
            The normalized identifier.
        """
        if not identifier:
            return ""

        # Step 1: Trim whitespace
        normalized = identifier.strip()

        # Step 2: Internal whitespace collapse
        normalized = re.sub(r"\s+", " ", normalized)

        # Step 3: Channel-specific normalization
        channel = cls.normalize_channel_type(channel_type)
        pattern = cls._CHANNEL_PATTERNS.get(channel)

        if pattern:
            match = pattern.match(normalized)
            if match:
                # Extract the core identifier from the capture group
                extracted = match.group(1) if match.lastindex else match.group(0)
                normalized = extracted

        # Step 4: Lowercase for text identifiers (not pure numbers)
        if not normalized.isdigit():
            normalized = normalized.lower()

        return normalized


class SessionKeyDerivation:
    """
    Derives consistent, deterministic session keys using SHA-256.

    Session keys are generated from channel type, account ID, and peer ID
    using a normalized composite string hashed with SHA-256. The resulting
    hex digest is truncated to a configurable length for use as session
    identifiers.

    The derivation is deterministic: the same inputs always produce the
    same key, ensuring session continuity across restarts.

    Parameters
    ----------
    prefix:
        A prefix prepended to the hash input for namespace isolation.
    key_length:
        Number of hex characters in the derived key (default 32 = 16 bytes).
    separator:
        Separator used between components in the hash input.
    """

    def __init__(
        self,
        prefix: str = _PREFIX,
        key_length: int = 32,
        separator: str = _KEY_SEPARATOR,
    ) -> None:
        if key_length < 8:
            raise ValueError(f"key_length must be at least 8, got {key_length}")
        if key_length > 64:
            raise ValueError(f"key_length must be at most 64, got {key_length}")
        self._prefix = prefix
        self._key_length = key_length
        self._separator = separator
        self._normalizer = ChannelNormalizer()

    def derive(
        self,
        channel_type: str,
        account_id: str,
        peer_id: str,
    ) -> str:
        """
        Derive a session key for a direct (1:1) conversation.

        Parameters
        ----------
        channel_type:
            The messaging platform (e.g., ``"whatsapp"``, ``"telegram"``).
        account_id:
            The agent's identifier on this channel.
        peer_id:
            The remote peer's identifier on this channel.

        Returns
        -------
        str
            A deterministic hex session key.

        Example
        -------
        >>> deriv = SessionKeyDerivation()
        >>> key = deriv.derive("whatsapp", "+1234567890", "+0987654321")
        >>> len(key)
        32
        """
        return self._derive_key(
            scope=KeyScope.DIRECT,
            channel_type=channel_type,
            primary=account_id,
            secondary=peer_id,
        )

    def derive_group(
        self,
        channel_type: str,
        group_id: str,
    ) -> str:
        """
        Derive a session key for a group conversation.

        Parameters
        ----------
        channel_type:
            The messaging platform.
        group_id:
            The group identifier on this channel.

        Returns
        -------
        str
            A deterministic hex session key.
        """
        return self._derive_key(
            scope=KeyScope.GROUP,
            channel_type=channel_type,
            primary=group_id,
            secondary="",
        )

    def derive_with_metadata(
        self,
        channel_type: str,
        account_id: str,
        peer_id: str,
        *,
        metadata_key: Optional[str] = None,
    ) -> str:
        """
        Derive a session key with optional metadata component.

        Useful for creating sub-sessions or keyed variants within the
        same conversation (e.g., different tool contexts or workflow states).

        Parameters
        ----------
        channel_type:
            The messaging platform.
        account_id:
            The agent's identifier.
        peer_id:
            The remote peer's identifier.
        metadata_key:
            Optional additional component for key variation.

        Returns
        -------
        str
            A deterministic hex session key.
        """
        composite = self._build_composite(
            scope=KeyScope.DIRECT,
            channel_type=channel_type,
            primary=account_id,
            secondary=peer_id,
            extra=metadata_key or "",
        )
        return self._hash(composite)

    def validate_key(self, key: str) -> bool:
        """
        Validate that a string looks like a derived session key.

        Checks length and hexadecimal character set.

        Parameters
        ----------
        key:
            The key string to validate.

        Returns
        -------
        bool
            True if the key matches the expected format.
        """
        if not key or len(key) != self._key_length:
            return False
        try:
            int(key, 16)
            return True
        except ValueError:
            return False

    def fingerprint(self, key: str) -> str:
        """
        Generate a short human-readable fingerprint of a session key.

        Takes the first 8 characters for display purposes.

        Parameters
        ----------
        key:
            A full session key.

        Returns
        -------
        str
            A short fingerprint string (e.g., ``"a1b2c3d4"``).
        """
        return key[:8] if len(key) >= 8 else key

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _derive_key(
        self,
        scope: KeyScope,
        channel_type: str,
        primary: str,
        secondary: str,
        extra: str = "",
    ) -> str:
        """Build the composite string and hash it."""
        composite = self._build_composite(scope, channel_type, primary, secondary, extra)
        return self._hash(composite)

    def _build_composite(
        self,
        scope: KeyScope,
        channel_type: str,
        primary: str,
        secondary: str,
        extra: str = "",
    ) -> str:
        """Build the deterministic composite string for hashing."""
        norm_channel = self._normalizer.normalize_channel_type(channel_type)
        norm_primary = self._normalizer.normalize_identifier(channel_type, primary)

        # For direct sessions, sort the two IDs to ensure the same key
        # regardless of which side initiates
        if scope == KeyScope.DIRECT and secondary:
            norm_secondary = self._normalizer.normalize_identifier(channel_type, secondary)
            sorted_ids = sorted([norm_primary, norm_secondary])
            norm_primary = sorted_ids[0]
            norm_secondary = sorted_ids[1]
            parts = [
                self._prefix,
                scope.value,
                norm_channel,
                norm_primary,
                norm_secondary,
            ]
        else:
            parts = [
                self._prefix,
                scope.value,
                norm_channel,
                norm_primary,
            ]

        if extra:
            norm_extra = self._normalizer.normalize_identifier(channel_type, extra)
            parts.append(norm_extra)

        return self._separator.join(parts)

    def _hash(self, composite: str) -> str:
        """Compute SHA-256 hex digest truncated to key_length."""
        digest = hashlib.sha256(composite.encode("utf-8")).hexdigest()
        return digest[: self._key_length]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────────────

# Module-level singleton for common usage
_default_derivation = SessionKeyDerivation()


def derive_session_key(
    channel_type: str,
    account_id: str,
    peer_id: str,
) -> str:
    """
    Derive a session key using the module-level default deriver.

    Convenience wrapper around :meth:`SessionKeyDerivation.derive`.
    """
    return _default_derivation.derive(channel_type, account_id, peer_id)


def derive_group_key(
    channel_type: str,
    group_id: str,
) -> str:
    """
    Derive a group session key using the module-level default deriver.

    Convenience wrapper around :meth:`SessionKeyDerivation.derive_group`.
    """
    return _default_derivation.derive_group(channel_type, group_id)
