"""
Atlas Security — Pairing Manager.

Manages secure DM (direct message) pairing for new contacts. Implements
a 6-digit pairing code protocol with expiration, rate limiting, and
persistent pairing state.

Inspired by OpenClaw's DM pairing security architecture.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from atlas.channels.base import ChannelType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PendingPairingCode:
    """A pending pairing code waiting to be validated.

    Attributes:
        code: 6-digit pairing code.
        channel_type: Channel type the pairing is for.
        peer_id: Peer/user identifier requesting pairing.
        created_at: When the code was generated.
        expires_at: When the code expires.
        attempts: Number of validation attempts so far.
        max_attempts: Maximum allowed attempts before invalidation.
    """

    code: str = ""
    channel_type: ChannelType = ChannelType.WEBCHAT
    peer_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(
        default_factory=lambda: (
            datetime.now(timezone.utc).replace(
                minute=datetime.now(timezone.utc).minute + 5,
            )
        )
    )
    attempts: int = 0
    max_attempts: int = 5

    @property
    def is_expired(self) -> bool:
        """Whether this code has expired."""
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_exhausted(self) -> bool:
        """Whether all attempts have been used."""
        return self.attempts >= self.max_attempts

    @property
    def is_valid(self) -> bool:
        """Whether this code can still be used."""
        return not self.is_expired and not self.is_exhausted

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "code": self.code,
            "channel_type": self.channel_type.value,
            "peer_id": self.peer_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "is_valid": self.is_valid,
        }


@dataclass
class PairedContact:
    """A successfully paired contact.

    Attributes:
        channel_type: The channel type.
        peer_id: The peer/user identifier.
        display_name: Optional display name.
        paired_at: When the pairing was established.
        pairing_code: The code used for pairing (for audit).
        metadata: Additional metadata (preferences, tags).
        trusted: Whether this contact is marked as trusted.
    """

    channel_type: ChannelType = ChannelType.WEBCHAT
    peer_id: str = ""
    display_name: str = ""
    paired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pairing_code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    trusted: bool = False

    def composite_key(self) -> str:
        """Unique key combining channel type and peer ID."""
        return f"{self.channel_type.value}:{self.peer_id}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "channel_type": self.channel_type.value,
            "peer_id": self.peer_id,
            "display_name": self.display_name,
            "paired_at": self.paired_at.isoformat(),
            "pairing_code": self.pairing_code,
            "metadata": self.metadata,
            "trusted": self.trusted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PairedContact":
        """Deserialize from dictionary."""
        data = data.copy()
        if "channel_type" in data and isinstance(data["channel_type"], str):
            data["channel_type"] = ChannelType(data["channel_type"])
        if "paired_at" in data and isinstance(data["paired_at"], str):
            data["paired_at"] = datetime.fromisoformat(data["paired_at"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Pairing Manager
# ---------------------------------------------------------------------------

class PairingManager:
    """Manages DM pairing with 6-digit codes, expiration, and rate limiting.

    Provides a secure mechanism for pairing new messaging contacts using
    time-limited 6-digit codes. Codes expire after 5 minutes and have a
    maximum of 5 validation attempts to prevent brute-force attacks.

    Usage::

        pm = PairingManager("/path/to/pairings.json")

        # Generate code for a new contact
        code = pm.generate_pairing_code(ChannelType.TELEGRAM, "user_123")
        # Send code to user via out-of-band channel
        print(f"Your pairing code: {code}")

        # Validate code from user's message
        contact = pm.validate_pairing_code(code)
        if contact:
            pm.pair(ChannelType.TELEGRAM, "user_123", code)
            print("Paired successfully!")
        else:
            print("Invalid or expired code")

        # Check pairing status
        if pm.is_paired(ChannelType.TELEGRAM, "user_123"):
            print("Contact is paired")
    """

    CODE_EXPIRY_SECONDS = 300  # 5 minutes
    MAX_ATTEMPTS = 5
    MAX_PENDING_PER_PEER = 3
    MAX_PAIRING_CODES = 1000

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        auto_save: bool = True,
        code_expiry_seconds: int = CODE_EXPIRY_SECONDS,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        """Initialize the pairing manager.

        Args:
            persistence_path: Path to the JSON pairings file.
            auto_save: Automatically save after mutations.
            code_expiry_seconds: How long codes remain valid.
            max_attempts: Maximum validation attempts per code.
        """
        self._paired: Dict[str, PairedContact] = {}
        self._pending_codes: Dict[str, PendingPairingCode] = {}
        self._rate_limit_tracker: Dict[str, List[float]] = {}
        self._auto_save = auto_save
        self._code_expiry_seconds = code_expiry_seconds
        self._max_attempts = max_attempts
        self._lock = threading.Lock()

        if persistence_path:
            self._persistence_path = Path(persistence_path)
        else:
            self._persistence_path = Path("pairings.json")

        self._load()
        self._cleanup_expired_codes()

        logger.info(
            "PairingManager initialized (%d paired, %d pending, expiry=%ds)",
            len(self._paired), len(self._pending_codes), code_expiry_seconds,
        )

    # ------------------------------------------------------------------
    # Code Generation
    # ------------------------------------------------------------------

    def generate_pairing_code(
        self,
        channel_type: ChannelType,
        peer_id: str,
    ) -> str:
        """Generate a new 6-digit pairing code.

        Args:
            channel_type: The channel type for the pairing.
            peer_id: The peer/user requesting pairing.

        Returns:
            6-digit numeric code string.

        Raises:
            RuntimeError: If rate limited or too many pending codes.
        """
        with self._lock:
            # Check rate limit
            self._check_rate_limit(peer_id)

            # Check pending code limit per peer
            peer_pending = [
                code for code in self._pending_codes.values()
                if code.peer_id == peer_id and code.is_valid
            ]
            if len(peer_pending) >= self.MAX_PENDING_PER_PEER:
                raise RuntimeError(
                    f"Too many pending pairing codes for {peer_id} "
                    f"(max {self.MAX_PENDING_PER_PEER})"
                )

            # Check total pending codes
            if len(self._pending_codes) >= self.MAX_PAIRING_CODES:
                self._cleanup_expired_codes()
                if len(self._pending_codes) >= self.MAX_PAIRING_CODES:
                    raise RuntimeError("Too many pending pairing codes globally")

            # Generate unique 6-digit code
            now = datetime.now(timezone.utc)
            code = self._generate_unique_code()
            expires_at = datetime.fromtimestamp(
                now.timestamp() + self._code_expiry_seconds,
                tz=timezone.utc,
            )

            pending = PendingPairingCode(
                code=code,
                channel_type=channel_type,
                peer_id=peer_id,
                created_at=now,
                expires_at=expires_at,
                max_attempts=self._max_attempts,
            )

            self._pending_codes[code] = pending

            if self._auto_save:
                self.save()

            logger.info(
                "Generated pairing code for %s:%s (expires in %ds)",
                channel_type.value, peer_id, self._code_expiry_seconds,
            )
            return code

    def validate_pairing_code(self, code: str) -> Optional[PendingPairingCode]:
        """Validate and consume a pairing code.

        The code is marked as used after successful validation.

        Args:
            code: The 6-digit code to validate.

        Returns:
            PendingPairingCode if valid, None otherwise.
        """
        with self._lock:
            pending = self._pending_codes.get(code)

            if pending is None:
                logger.debug("Pairing code not found: %s", code)
                return None

            pending.attempts += 1

            if pending.is_expired:
                logger.info("Pairing code expired: %s", code)
                self._pending_codes.pop(code, None)
                return None

            if pending.is_exhausted:
                logger.warning(
                    "Pairing code exhausted (attempts=%d): %s",
                    pending.attempts, code,
                )
                self._pending_codes.pop(code, None)
                return None

            if self._auto_save:
                self.save()

            return pending

    # ------------------------------------------------------------------
    # Pairing Operations
    # ------------------------------------------------------------------

    def pair(
        self,
        channel_type: ChannelType,
        peer_id: str,
        code: str,
        display_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[PairedContact]:
        """Pair a new contact using a validated code.

        Args:
            channel_type: The channel type.
            peer_id: The peer/user identifier.
            code: The validated pairing code.
            display_name: Optional display name.
            metadata: Additional metadata.

        Returns:
            PairedContact if successful, None if code is invalid.
        """
        pending = self.validate_pairing_code(code)
        if pending is None:
            return None

        # Verify the code matches the channel and peer
        if pending.channel_type != channel_type or pending.peer_id != peer_id:
            logger.warning(
                "Pairing code mismatch: expected %s:%s, got %s:%s",
                pending.channel_type.value, pending.peer_id,
                channel_type.value, peer_id,
            )
            return None

        with self._lock:
            now = datetime.now(timezone.utc)
            contact = PairedContact(
                channel_type=channel_type,
                peer_id=peer_id,
                display_name=display_name,
                paired_at=now,
                pairing_code=code,
                metadata=metadata or {},
                trusted=False,
            )

            key = contact.composite_key()
            self._paired[key] = contact

            # Remove used code
            self._pending_codes.pop(code, None)

            if self._auto_save:
                self.save()

        logger.info(
            "Paired contact %s:%s (code=%s)",
            channel_type.value, peer_id, code,
        )
        return contact

    def unpair(
        self,
        channel_type: ChannelType,
        peer_id: str,
    ) -> bool:
        """Remove a pairing.

        Args:
            channel_type: The channel type.
            peer_id: The peer/user identifier.

        Returns:
            True if the pairing was found and removed.
        """
        key = f"{channel_type.value}:{peer_id}"

        with self._lock:
            contact = self._paired.pop(key, None)

        if contact is not None:
            if self._auto_save:
                self.save()
            logger.info("Unpaired %s:%s", channel_type.value, peer_id)
            return True

        logger.debug("No pairing found for %s:%s", channel_type.value, peer_id)
        return False

    def is_paired(
        self,
        channel_type: ChannelType,
        peer_id: str,
    ) -> bool:
        """Check if a contact is paired.

        Args:
            channel_type: The channel type.
            peer_id: The peer/user identifier.

        Returns:
            True if the contact is paired.
        """
        key = f"{channel_type.value}:{peer_id}"
        return key in self._paired

    def get_paired(
        self,
        channel_type: ChannelType,
        peer_id: str,
    ) -> Optional[PairedContact]:
        """Get a paired contact.

        Args:
            channel_type: The channel type.
            peer_id: The peer/user identifier.

        Returns:
            PairedContact if found, None otherwise.
        """
        key = f"{channel_type.value}:{peer_id}"
        return self._paired.get(key)

    # ------------------------------------------------------------------
    # Query Operations
    # ------------------------------------------------------------------

    def list_paired(
        self,
        channel_type: Optional[ChannelType] = None,
        trusted_only: bool = False,
    ) -> List[PairedContact]:
        """List paired contacts.

        Args:
            channel_type: Filter by channel type.
            trusted_only: Only return trusted contacts.

        Returns:
            List of paired contacts.
        """
        contacts = list(self._paired.values())

        if channel_type is not None:
            contacts = [c for c in contacts if c.channel_type == channel_type]

        if trusted_only:
            contacts = [c for c in contacts if c.trusted]

        return sorted(contacts, key=lambda c: c.paired_at)

    def list_paired_dict(
        self,
        channel_type: Optional[ChannelType] = None,
    ) -> List[Dict[str, Any]]:
        """List paired contacts as dictionaries.

        Args:
            channel_type: Filter by channel type.

        Returns:
            List of contact dictionaries.
        """
        return [c.to_dict() for c in self.list_paired(channel_type)]

    def list_pending_codes(self) -> List[Dict[str, Any]]:
        """List all valid pending pairing codes.

        Returns:
            List of pending code dictionaries.
        """
        self._cleanup_expired_codes()
        return [
            code.to_dict()
            for code in self._pending_codes.values()
            if code.is_valid
        ]

    def count_paired(self, channel_type: Optional[ChannelType] = None) -> int:
        """Count paired contacts.

        Args:
            channel_type: Filter by channel type.

        Returns:
            Number of paired contacts.
        """
        if channel_type is None:
            return len(self._paired)
        return sum(
            1 for c in self._paired.values()
            if c.channel_type == channel_type
        )

    def trust_contact(
        self,
        channel_type: ChannelType,
        peer_id: str,
    ) -> bool:
        """Mark a contact as trusted.

        Args:
            channel_type: The channel type.
            peer_id: The peer/user identifier.

        Returns:
            True if found and updated.
        """
        contact = self.get_paired(channel_type, peer_id)
        if contact:
            contact.trusted = True
            if self._auto_save:
                self.save()
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save pairings to the persistence file."""
        if self._persistence_path is None:
            return

        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "paired": {
                key: contact.to_dict()
                for key, contact in self._paired.items()
            },
            "pending_codes": {
                code: pending.to_dict()
                for code, pending in self._pending_codes.items()
                if pending.is_valid
            },
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        temp_path = str(self._persistence_path) + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, str(self._persistence_path))

        logger.debug("Saved pairings to %s", self._persistence_path)

    def _load(self) -> None:
        """Load pairings from the persistence file."""
        if not self._persistence_path.exists():
            return

        try:
            with open(self._persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load paired contacts
            for key, contact_data in data.get("paired", {}).items():
                try:
                    contact = PairedContact.from_dict(contact_data)
                    self._paired[contact.composite_key()] = contact
                except Exception as exc:
                    logger.warning("Failed to load paired contact: %s", exc)

            # Load pending codes
            for code, code_data in data.get("pending_codes", {}).items():
                try:
                    pending = PendingPairingCode(
                        code=code_data["code"],
                        channel_type=ChannelType(code_data["channel_type"]),
                        peer_id=code_data["peer_id"],
                        created_at=datetime.fromisoformat(code_data["created_at"]),
                        expires_at=datetime.fromisoformat(code_data["expires_at"]),
                        attempts=code_data.get("attempts", 0),
                        max_attempts=code_data.get("max_attempts", self._max_attempts),
                    )
                    self._pending_codes[code] = pending
                except Exception as exc:
                    logger.warning("Failed to load pending code: %s", exc)

            logger.info(
                "Loaded %d paired contacts and %d pending codes from %s",
                len(self._paired), len(self._pending_codes),
                self._persistence_path,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load pairings: %s", exc)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_unique_code(self) -> str:
        """Generate a unique 6-digit code.

        Returns:
            6-digit numeric string that doesn't collide with existing codes.
        """
        for _ in range(100):
            code = str(secrets.randbelow(1000000)).zfill(6)
            if code not in self._pending_codes:
                return code
        # Fallback: use timestamp-based code
        return str(int(time.time() * 1000))[-6:]

    def _check_rate_limit(self, peer_id: str) -> None:
        """Check and enforce rate limiting for pairing code generation.

        Args:
            peer_id: The peer to check.

        Raises:
            RuntimeError: If rate limited.
        """
        now = time.monotonic()
        window = 60.0  # 1 minute window
        max_codes_per_window = 3

        if peer_id not in self._rate_limit_tracker:
            self._rate_limit_tracker[peer_id] = []

        # Prune old timestamps
        self._rate_limit_tracker[peer_id] = [
            t for t in self._rate_limit_tracker[peer_id]
            if now - t < window
        ]

        if len(self._rate_limit_tracker[peer_id]) >= max_codes_per_window:
            raise RuntimeError(
                f"Rate limited: {peer_id} has generated too many "
                f"pairing codes in the last minute. Please wait."
            )

        self._rate_limit_tracker[peer_id].append(now)

    def _cleanup_expired_codes(self) -> int:
        """Remove expired and exhausted pending codes.

        Returns:
            Number of codes cleaned up.
        """
        expired_keys = [
            code for code, pending in self._pending_codes.items()
            if not pending.is_valid
        ]
        for key in expired_keys:
            del self._pending_codes[key]

        if expired_keys:
            logger.debug("Cleaned up %d expired pairing codes", len(expired_keys))

        return len(expired_keys)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of pairing state.

        Returns:
            Dictionary with counts and status.
        """
        self._cleanup_expired_codes()
        return {
            "total_paired": len(self._paired),
            "total_pending": len(self._pending_codes),
            "code_expiry_seconds": self._code_expiry_seconds,
            "max_attempts": self._max_attempts,
            "by_channel": {
                ct.value: sum(1 for c in self._paired.values() if c.channel_type == ct)
                for ct in ChannelType
                if any(c.channel_type == ct for c in self._paired.values())
            },
            "trusted_count": sum(1 for c in self._paired.values() if c.trusted),
            "persistence_path": str(self._persistence_path),
        }

    def __len__(self) -> int:
        return len(self._paired)

    def __repr__(self) -> str:
        return f"<PairingManager paired={len(self._paired)} pending={len(self._pending_codes)}>"
