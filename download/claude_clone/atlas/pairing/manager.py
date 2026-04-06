"""
Device and application pairing manager.

Implements a secure pairing protocol with 6-digit codes, expiration
handling, rate limiting, and device trust management.

Usage::

    manager = PairingManager()
    code = manager.generate_pairing_code()
    pairing = manager.pair(code, {"name": "Alice's iPhone", "type": "mobile"})
    devices = manager.list_paired()
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("atlas.pairing.manager")


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PairingCode:
    """A pending pairing code awaiting confirmation."""

    code: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    attempts: int = 0
    max_attempts: int = 10
    is_used: bool = False

    @property
    def is_expired(self) -> bool:
        """Check if the code has expired."""
        return time.time() > self.expires_at

    @property
    def remaining_seconds(self) -> float:
        """Seconds remaining before expiration."""
        return max(0.0, self.expires_at - time.time())

    @property
    def remaining_attempts(self) -> int:
        """Remaining validation attempts."""
        return max(0, self.max_attempts - self.attempts)


@dataclass
class PairedDevice:
    """A successfully paired device."""

    pairing_id: str
    device_id: str
    device_info: Dict[str, Any] = field(default_factory=dict)
    code_used: str = ""
    paired_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_active: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_active: bool = True
    trust_level: int = 0  # 0 = new, higher = more trusted
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pairing_id": self.pairing_id,
            "device_id": self.device_id,
            "device_info": self.device_info,
            "code_used": self.code_used,
            "paired_at": self.paired_at,
            "last_active": self.last_active,
            "is_active": self.is_active,
            "trust_level": self.trust_level,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }

    def touch(self) -> None:
        """Update the last_active timestamp."""
        self.last_active = datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class PairingError(Exception):
    """Base exception for pairing operations."""


class InvalidCodeError(PairingError):
    """Raised when an invalid pairing code is provided."""


class ExpiredCodeError(PairingError):
    """Raised when an expired pairing code is used."""


class RateLimitError(PairingError):
    """Raised when rate limit is exceeded."""


class AlreadyPairedError(PairingError):
    """Raised when a device is already paired."""


# ──────────────────────────────────────────────────────────────────────────────
# Pairing Manager
# ──────────────────────────────────────────────────────────────────────────────

class PairingManager:
    """
    Manage device and application pairing with 6-digit codes.

    Features:
    - 6-digit alphanumeric pairing code generation
    - Configurable code expiration
    - Rate limiting on validation attempts
    - Device trust levels and tagging
    - Paired device lifecycle management
    - Code cleanup for expired entries

    Parameters
    ----------
    code_length:
        Length of the pairing code. Default 6.
    expiry_seconds:
        How long a code remains valid. Default 300 (5 minutes).
    max_attempts:
        Max validation attempts per code. Default 10.
    rate_limit_window:
        Rate limit window in seconds. Default 60.
    rate_limit_max:
        Max attempts per rate limit window per IP/client. Default 5.
    """

    def __init__(
        self,
        code_length: int = 6,
        expiry_seconds: float = 300.0,
        max_attempts: int = 10,
        rate_limit_window: float = 60.0,
        rate_limit_max: int = 5,
    ) -> None:
        self._code_length = code_length
        self._expiry_seconds = expiry_seconds
        self._max_attempts = max_attempts
        self._rate_limit_window = rate_limit_window
        self._rate_limit_max = rate_limit_max

        # Active pairing codes: code -> PairingCode
        self._codes: Dict[str, PairingCode] = {}

        # Paired devices: pairing_id -> PairedDevice
        self._paired: Dict[str, PairedDevice] = {}

        # Device ID index for fast lookup: device_id -> pairing_id
        self._device_index: Dict[str, str] = {}

        # Rate limiting: client_id -> [timestamps]
        self._rate_limits: Dict[str, List[float]] = defaultdict(list)

    # ── Code Generation ──────────────────────────────────────────────────

    def generate_pairing_code(
        self,
        expiry_seconds: Optional[float] = None,
        client_id: Optional[str] = None,
    ) -> str:
        """
        Generate a new 6-digit pairing code.

        Parameters
        ----------
        expiry_seconds:
            Override for code expiration duration.
        client_id:
            Client identifier for rate limiting.

        Returns
        -------
        str
            The generated pairing code.

        Raises
        ------
        RateLimitError
            If the client has exceeded rate limits.
        """
        # Check rate limit
        if client_id:
            self._check_rate_limit(client_id)

        # Generate unique code
        code = self._generate_unique_code()
        expiry = expiry_seconds or self._expiry_seconds
        now = time.time()

        self._codes[code] = PairingCode(
            code=code,
            created_at=now,
            expires_at=now + expiry,
            max_attempts=self._max_attempts,
        )

        logger.info(
            "Generated pairing code (expires in %.0fs, client=%s)",
            expiry, client_id or "anonymous",
        )

        return code

    def _generate_unique_code(self) -> str:
        """Generate a unique pairing code not already in use."""
        charset = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # No ambiguous chars
        for _ in range(1000):
            code = "".join(secrets.choice(charset) for _ in range(self._code_length))
            if code not in self._codes:
                return code
        # Fallback: use full random
        code = secrets.token_hex(self._code_length).upper()[: self._code_length]
        return code

    # ── Code Validation ──────────────────────────────────────────────────

    def validate_code(
        self,
        code: str,
        client_id: Optional[str] = None,
    ) -> bool:
        """
        Validate a pairing code without consuming it.

        Parameters
        ----------
        code:
            The pairing code to validate.
        client_id:
            Client identifier for rate limiting.

        Returns
        -------
        bool
            True if the code is valid and not expired.

        Raises
        ------
        RateLimitError
            If the client has exceeded rate limits.
        """
        if client_id:
            self._check_rate_limit(client_id)

        pairing_code = self._codes.get(code.upper())
        if pairing_code is None:
            return False
        if pairing_code.is_expired or pairing_code.is_used:
            return False

        return True

    # ── Pairing ──────────────────────────────────────────────────────────

    def pair(
        self,
        code: str,
        device_info: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None,
    ) -> PairedDevice:
        """
        Pair a device using a valid pairing code.

        Parameters
        ----------
        code:
            The pairing code.
        device_info:
            Information about the device being paired (name, type, os, etc.).
        client_id:
            Client identifier for rate limiting.

        Returns
        -------
        PairedDevice
            The newly paired device.

        Raises
        ------
        InvalidCodeError
            If the code does not exist.
        ExpiredCodeError
            If the code has expired.
        RateLimitError
            If rate limited.
        AlreadyPairedError
            If the device is already paired.
        """
        if client_id:
            self._check_rate_limit(client_id)

        code_upper = code.upper()
        pairing_code = self._codes.get(code_upper)

        if pairing_code is None:
            raise InvalidCodeError(f"Invalid pairing code: {code}")

        if pairing_code.is_expired:
            raise ExpiredCodeError("Pairing code has expired")

        if pairing_code.is_used:
            raise InvalidCodeError("Pairing code has already been used")

        # Increment attempts
        pairing_code.attempts += 1
        if pairing_code.attempts > pairing_code.max_attempts:
            del self._codes[code_upper]
            raise RateLimitError(
                f"Max validation attempts ({pairing_code.max_attempts}) exceeded"
            )

        # Check if device already paired
        info = device_info or {}
        device_id = info.get("device_id", info.get("id", ""))
        if device_id and device_id in self._device_index:
            raise AlreadyPairedError(
                f"Device {device_id} is already paired"
            )

        # Generate a unique device ID if not provided
        if not device_id:
            device_id = uuid.uuid4().hex[:12]

        # Create paired device
        pairing_id = uuid.uuid4().hex[:12]
        paired_device = PairedDevice(
            pairing_id=pairing_id,
            device_id=device_id,
            device_info=info,
            code_used=code_upper,
            tags=info.get("tags", set()) if isinstance(info.get("tags"), set) else set(),
            metadata=info.get("metadata", {}),
        )

        # Mark code as used
        pairing_code.is_used = True

        # Store
        self._paired[pairing_id] = paired_device
        self._device_index[device_id] = pairing_id

        logger.info(
            "Device paired: %s (pairing_id=%s, device=%s)",
            code_upper, pairing_id, device_id,
        )

        return paired_device

    # ── Unpairing ────────────────────────────────────────────────────────

    def unpair(self, pairing_id: str) -> bool:
        """
        Remove a paired device.

        Parameters
        ----------
        pairing_id:
            The pairing ID to remove.

        Returns
        -------
        bool
            True if found and removed, False otherwise.
        """
        device = self._paired.get(pairing_id)
        if device is None:
            logger.warning("Cannot unpair: pairing_id %s not found", pairing_id)
            return False

        # Remove from device index
        self._device_index.pop(device.device_id, None)

        # Remove from paired
        del self._paired[pairing_id]

        logger.info("Device unpaired: %s (device=%s)", pairing_id, device.device_id)
        return True

    # ── Queries ──────────────────────────────────────────────────────────

    def list_paired(
        self,
        active_only: bool = False,
        tag: Optional[str] = None,
    ) -> List[PairedDevice]:
        """
        List paired devices with optional filtering.

        Parameters
        ----------
        active_only:
            Only return active devices. Default False.
        tag:
            Filter by tag. Default None.

        Returns
        -------
        List[PairedDevice]
            Matching paired devices.
        """
        devices = list(self._paired.values())
        if active_only:
            devices = [d for d in devices if d.is_active]
        if tag:
            devices = [d for d in devices if tag in d.tags]
        return devices

    def is_paired(self, device_id: str) -> bool:
        """
        Check if a device is paired.

        Parameters
        ----------
        device_id:
            The device identifier to check.

        Returns
        -------
        bool
            True if the device is currently paired and active.
        """
        pairing_id = self._device_index.get(device_id)
        if pairing_id is None:
            return False
        device = self._paired.get(pairing_id)
        return device is not None and device.is_active

    def get_pairing_info(self, pairing_id: str) -> Optional[PairedDevice]:
        """
        Get detailed pairing information.

        Parameters
        ----------
        pairing_id:
            The pairing ID.

        Returns
        -------
        Optional[PairedDevice]
            Pairing details or None if not found.
        """
        return self._paired.get(pairing_id)

    def get_device_by_id(self, device_id: str) -> Optional[PairedDevice]:
        """
        Look up a paired device by its device_id.

        Parameters
        ----------
        device_id:
            The device identifier.

        Returns
        -------
        Optional[PairedDevice]
            The paired device or None if not found.
        """
        pairing_id = self._device_index.get(device_id)
        if pairing_id is None:
            return None
        return self._paired.get(pairing_id)

    # ── Cleanup ──────────────────────────────────────────────────────────

    def cleanup_expired_codes(self) -> int:
        """
        Remove expired and used pairing codes.

        Returns
        -------
        int
            Number of codes removed.
        """
        now = time.time()
        expired = [
            code for code, pc in self._codes.items()
            if pc.is_expired or pc.is_used
        ]
        for code in expired:
            del self._codes[code]

        if expired:
            logger.info("Cleaned up %d expired/used pairing codes", len(expired))
        return len(expired)

    def cleanup_rate_limits(self) -> None:
        """Remove stale rate limit entries."""
        now = time.time()
        cutoff = now - self._rate_limit_window
        stale_clients = [
            client_id for client_id, timestamps in self._rate_limits.items()
            if not timestamps or timestamps[-1] < cutoff
        ]
        for client_id in stale_clients:
            del self._rate_limits[client_id]

    # ── Trust Management ─────────────────────────────────────────────────

    def set_trust_level(self, pairing_id: str, level: int) -> bool:
        """
        Set the trust level for a paired device.

        Parameters
        ----------
        pairing_id:
            The pairing ID.
        level:
            Trust level (0 = new, higher = more trusted).

        Returns
        -------
        bool
            True if the device was found.
        """
        device = self._paired.get(pairing_id)
        if device is None:
            return False
        device.trust_level = max(0, min(100, level))
        return True

    def add_tag(self, pairing_id: str, tag: str) -> bool:
        """Add a tag to a paired device."""
        device = self._paired.get(pairing_id)
        if device is None:
            return False
        device.tags.add(tag)
        return True

    def remove_tag(self, pairing_id: str, tag: str) -> bool:
        """Remove a tag from a paired device."""
        device = self._paired.get(pairing_id)
        if device is None:
            return False
        device.tags.discard(tag)
        return True

    # ── Rate Limiting ────────────────────────────────────────────────────

    def _check_rate_limit(self, client_id: str) -> None:
        """Check rate limit and raise RateLimitError if exceeded."""
        now = time.time()
        cutoff = now - self._rate_limit_window
        timestamps = [
            ts for ts in self._rate_limits[client_id] if ts > cutoff
        ]
        self._rate_limits[client_id] = timestamps

        if len(timestamps) >= self._rate_limit_max:
            # Calculate when the oldest entry in window will expire
            oldest = min(timestamps)
            wait = self._rate_limit_window - (now - oldest)
            raise RateLimitError(
                f"Rate limit exceeded. Try again in {wait:.0f} seconds"
            )

        timestamps.append(now)

    # ── Statistics ───────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return pairing manager statistics."""
        paired = list(self._paired.values())
        active_codes = [
            c for c in self._codes.values() if not c.is_expired and not c.is_used
        ]
        return {
            "total_paired_devices": len(paired),
            "active_paired": sum(1 for d in paired if d.is_active),
            "inactive_paired": sum(1 for d in paired if not d.is_active),
            "active_codes": len(active_codes),
            "total_codes_issued": len(self._codes),
            "rate_limited_clients": len(self._rate_limits),
        }
