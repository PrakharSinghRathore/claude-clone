"""
DM pairing and authentication for the Hermes Gateway.

Implements a secure pairing protocol for connecting DM conversations
to the gateway, with token-based authentication, admin/user roles,
rate limiting per user, and whitelist/blacklist management.

Usage::

    pairing = PairingManager(secret="my-secret-key")
    token = pairing.generate_pairing_token(user_id="alice", role="admin")
    result = pairing.authenticate(token)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("hermes.gateway.pairing")


# ──────────────────────────────────────────────────────────────────────────────
# Roles
# ──────────────────────────────────────────────────────────────────────────────

class PairingRole(str, Enum):
    """User roles for paired DMs."""

    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    BLOCKED = "blocked"


# ──────────────────────────────────────────────────────────────────────────────
# Pairing Entry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PairingEntry:
    """A paired user/device entry."""

    user_id: str
    platform: str
    role: PairingRole = PairingRole.USER
    paired_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_active: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    pairing_token_hash: Optional[str] = None
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "platform": self.platform,
            "role": self.role.value,
            "paired_at": self.paired_at,
            "last_active": self.last_active,
            "metadata": self.metadata,
            "is_active": self.is_active,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Rate Limit Entry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RateLimitEntry:
    """Tracks rate limit state for a user."""

    user_id: str
    platform: str
    request_count: int = 0
    window_start: float = field(default_factory=time.time)
    blocked_until: float = 0.0

    def is_rate_limited(self, max_requests: int, window_seconds: int) -> bool:
        """Check if the user is rate limited."""
        now = time.time()
        if now < self.blocked_until:
            return True

        if now - self.window_start > window_seconds:
            self.window_start = now
            self.request_count = 0

        return self.request_count >= max_requests

    def record_request(self, max_requests: int, window_seconds: int) -> Optional[float]:
        """
        Record a request and return the block duration if rate limited.

        Returns None if not limited, or the number of seconds until
        the rate limit expires.
        """
        now = time.time()
        if now < self.blocked_until:
            return self.blocked_until - now

        if now - self.window_start > window_seconds:
            self.window_start = now
            self.request_count = 0

        self.request_count += 1

        if self.request_count > max_requests:
            self.blocked_until = self.window_start + window_seconds
            return window_seconds

        return None


# ──────────────────────────────────────────────────────────────────────────────
# Pairing Manager
# ──────────────────────────────────────────────────────────────────────────────

class PairingManager:
    """
    Manages DM pairing and authentication for the Hermes Gateway.

    Features:
    - Secure token-based pairing protocol
    - Admin/user/guest/blocked roles
    - Per-user rate limiting with configurable windows
    - Whitelist and blacklist management
    - Pairing token expiry
    - Persistent pairing storage (JSON)
    """

    # Default rate limits per role
    DEFAULT_RATE_LIMITS: Dict[PairingRole, Tuple[int, int]] = {
        PairingRole.ADMIN: (120, 60),    # 120 requests per 60 seconds
        PairingRole.USER: (30, 60),      # 30 requests per 60 seconds
        PairingRole.GUEST: (10, 60),     # 10 requests per 60 seconds
        PairingRole.BLOCKED: (0, 60),    # No requests allowed
    }

    def __init__(
        self,
        secret: Optional[str] = None,
        enabled: bool = True,
        require_pairing: bool = False,
        token_expiry: int = 86400,       # 24 hours
        persist_path: Optional[str] = None,
        admin_ids: Optional[List[str]] = None,
        whitelist: Optional[List[str]] = None,
        blacklist: Optional[List[str]] = None,
        rate_limits: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        self._enabled = enabled
        self._require_pairing = require_pairing
        self._secret = secret or os.environ.get("HERMES_PAIRING_SECRET", secrets.token_hex(32))
        self._token_expiry = token_expiry
        self._persist_path = Path(persist_path).expanduser().resolve() if persist_path else None

        # Paired users: key = f"{platform}:{user_id}"
        self._paired: Dict[str, PairingEntry] = {}

        # Pending tokens: token -> (platform, user_id, role, created_at)
        self._pending_tokens: Dict[str, Tuple[str, str, PairingRole, float]] = {}

        # Rate limits
        self._rate_limits: Dict[str, RateLimitEntry] = {}
        self._role_rate_limits = dict(self.DEFAULT_RATE_LIMITS)
        if rate_limits:
            for role_str, limit in rate_limits.items():
                try:
                    role = PairingRole(role_str)
                    self._role_rate_limits[role] = limit
                except ValueError:
                    pass

        # Whitelist/blacklist
        self._whitelist: Set[str] = set(whitelist or admin_ids or [])
        self._blacklist: Set[str] = set(blacklist or [])

        # Admin IDs (auto-paired as admin)
        self._admin_ids: Set[str] = set(admin_ids or [])

        # Load persisted pairings
        if self._persist_path:
            self._load_persist()

    # ── Token Generation ──────────────────────────────────────────────────

    def generate_pairing_token(
        self,
        user_id: str,
        platform: str,
        role: PairingRole = PairingRole.USER,
        expiry_seconds: Optional[int] = None,
    ) -> str:
        """
        Generate a one-time pairing token.

        Parameters
        ----------
        user_id:
            User identifier to pair.
        platform:
            Platform name.
        role:
            Role to assign after pairing.
        expiry_seconds:
            Token validity duration. Defaults to configured expiry.

        Returns
        -------
        str
            The pairing token.
        """
        expiry = expiry_seconds or self._token_expiry
        token = secrets.token_urlsafe(32)
        created_at = time.time()

        self._pending_tokens[token] = (platform, user_id, role, created_at + expiry)
        logger.info(
            "Generated pairing token for %s:%s (role=%s, expires=%ds)",
            platform, user_id, role.value, expiry,
        )
        return token

    # ── Authentication ────────────────────────────────────────────────────

    async def authenticate(
        self,
        token: str,
        platform: Optional[str] = None,
    ) -> Tuple[bool, Optional[PairingEntry], Optional[str]]:
        """
        Authenticate using a pairing token.

        Parameters
        ----------
        token:
            The pairing token to authenticate.
        platform:
            Optional platform to verify against.

        Returns
        -------
        Tuple[bool, Optional[PairingEntry], Optional[str]]
            (success, pairing_entry, error_message)
        """
        if not self._enabled:
            return True, None, None

        if token not in self._pending_tokens:
            return False, None, "Invalid or expired pairing token"

        pending_platform, user_id, role, expires_at = self._pending_tokens.pop(token)

        if time.time() > expires_at:
            return False, None, "Pairing token has expired"

        if platform and pending_platform != platform:
            return False, None, f"Token is for platform '{pending_platform}', not '{platform}'"

        # Create pairing entry
        entry = PairingEntry(
            user_id=user_id,
            platform=pending_platform or platform or "unknown",
            role=role,
            pairing_token_hash=self._hash_token(token),
        )
        self._paired[f"{entry.platform}:{entry.user_id}"] = entry

        # Persist
        if self._persist_path:
            self._save_persist()

        logger.info(
            "Paired user %s:%s with role %s",
            entry.platform, entry.user_id, role.value,
        )
        return True, entry, None

    async def check_access(
        self,
        user_id: str,
        platform: str,
    ) -> Tuple[bool, Optional[PairingRole], Optional[str]]:
        """
        Check if a user has access to the gateway.

        Parameters
        ----------
        user_id:
            Platform-specific user identifier.
        platform:
            Platform name.

        Returns
        -------
        Tuple[bool, Optional[PairingRole], Optional[str]]
            (has_access, role, error_message)
        """
        if not self._enabled:
            return True, PairingRole.USER, None

        key = f"{platform}:{user_id}"

        # Check blacklist
        if user_id in self._blacklist or key in self._blacklist:
            return False, PairingRole.BLOCKED, "User is blacklisted"

        # Check whitelist (if set, only whitelisted users can access)
        if self._whitelist and user_id not in self._whitelist and key not in self._whitelist:
            return False, None, "User is not in the whitelist"

        # Check admin auto-pair
        if user_id in self._admin_ids:
            entry = self._paired.get(key)
            if entry is None:
                entry = PairingEntry(
                    user_id=user_id, platform=platform, role=PairingRole.ADMIN,
                )
                self._paired[key] = entry
            return True, PairingRole.ADMIN, None

        # Check paired users
        entry = self._paired.get(key)
        if entry is None:
            if self._require_pairing:
                return False, None, "User not paired. Use a pairing token to authenticate."
            # Auto-pair as guest if pairing not required
            entry = PairingEntry(
                user_id=user_id, platform=platform, role=PairingRole.GUEST,
            )
            self._paired[key] = entry
            return True, PairingRole.GUEST, None

        if not entry.is_active:
            return False, PairingRole.BLOCKED, "User pairing is inactive"

        # Update last active
        entry.last_active = datetime.now(timezone.utc).isoformat()
        return True, entry.role, None

    # ── Rate Limiting ─────────────────────────────────────────────────────

    async def check_rate_limit(
        self,
        user_id: str,
        platform: str,
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ) -> Optional[float]:
        """
        Check rate limit for a user.

        Returns None if not rate limited, or the number of seconds
        until the rate limit expires.
        """
        if not self._enabled:
            return None

        key = f"{platform}:{user_id}"

        # Get role-based limits
        entry = self._paired.get(key)
        role = entry.role if entry else PairingRole.GUEST
        role_max, role_window = self._role_rate_limits.get(role, (30, 60))

        if max_requests is None:
            max_requests = role_max
        if window_seconds is None:
            window_seconds = role_window

        rate_entry = self._rate_limits.get(key)
        if rate_entry is None:
            rate_entry = RateLimitEntry(user_id=user_id, platform=platform)
            self._rate_limits[key] = rate_entry

        return rate_entry.record_request(max_requests, window_seconds)

    # ── User Management ───────────────────────────────────────────────────

    def get_user(self, user_id: str, platform: str) -> Optional[PairingEntry]:
        """Get a paired user entry."""
        return self._paired.get(f"{platform}:{user_id}")

    def list_users(
        self,
        platform: Optional[str] = None,
        role: Optional[PairingRole] = None,
    ) -> List[PairingEntry]:
        """List paired users, optionally filtered."""
        users = list(self._paired.values())
        if platform:
            users = [u for u in users if u.platform == platform]
        if role:
            users = [u for u in users if u.role == role]
        return users

    def set_role(self, user_id: str, platform: str, role: PairingRole) -> bool:
        """Set the role for a paired user. Returns True if found."""
        key = f"{platform}:{user_id}"
        entry = self._paired.get(key)
        if entry is None:
            return False
        entry.role = role
        if self._persist_path:
            self._save_persist()
        return True

    def unpair(self, user_id: str, platform: str) -> bool:
        """Remove a paired user. Returns True if found."""
        key = f"{platform}:{user_id}"
        if key in self._paired:
            del self._paired[key]
            if self._persist_path:
                self._save_persist()
            return True
        return False

    # ── Whitelist / Blacklist ─────────────────────────────────────────────

    def add_to_whitelist(self, identifier: str) -> None:
        """Add a user ID or platform:user_id to the whitelist."""
        self._whitelist.add(identifier)

    def remove_from_whitelist(self, identifier: str) -> None:
        """Remove from whitelist."""
        self._whitelist.discard(identifier)

    def add_to_blacklist(self, identifier: str) -> None:
        """Add a user ID or platform:user_id to the blacklist."""
        self._blacklist.add(identifier)

    def remove_from_blacklist(self, identifier: str) -> None:
        """Remove from blacklist."""
        self._blacklist.discard(identifier)

    def get_whitelist(self) -> Set[str]:
        return set(self._whitelist)

    def get_blacklist(self) -> Set[str]:
        return set(self._blacklist)

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_persist(self) -> None:
        """Save paired users to a JSON file."""
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "paired": {
                key: entry.to_dict() for key, entry in self._paired.items()
            },
            "whitelist": list(self._whitelist),
            "blacklist": list(self._blacklist),
        }
        self._persist_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        os.chmod(str(self._persist_path), 0o600)

    def _load_persist(self) -> None:
        """Load paired users from a JSON file."""
        if self._persist_path is None or not self._persist_path.exists():
            return

        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for key, entry_data in data.get("paired", {}).items():
                entry_data["role"] = PairingRole(entry_data.get("role", "user"))
                self._paired[key] = PairingEntry(**{
                    k: v for k, v in entry_data.items()
                    if k in PairingEntry.__dataclass_fields__
                })
            self._whitelist = set(data.get("whitelist", []))
            self._blacklist = set(data.get("blacklist", []))
        except Exception as e:
            logger.error("Failed to load pairing data: %s", e)

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _hash_token(token: str) -> str:
        """Hash a token for storage (not reversible)."""
        return hashlib.sha256(token.encode()).hexdigest()

    def cleanup_expired_tokens(self) -> int:
        """Remove expired pending tokens. Returns count removed."""
        now = time.time()
        expired = [
            token for token, (_, _, _, expires_at) in self._pending_tokens.items()
            if now > expires_at
        ]
        for token in expired:
            del self._pending_tokens[token]
        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Return pairing manager statistics."""
        return {
            "enabled": self._enabled,
            "require_pairing": self._require_pairing,
            "total_paired": len(self._paired),
            "pending_tokens": len(self._pending_tokens),
            "whitelist_size": len(self._whitelist),
            "blacklist_size": len(self._blacklist),
            "admin_count": sum(
                1 for e in self._paired.values() if e.role == PairingRole.ADMIN
            ),
            "user_count": sum(
                1 for e in self._paired.values() if e.role == PairingRole.USER
            ),
            "guest_count": sum(
                1 for e in self._paired.values() if e.role == PairingRole.GUEST
            ),
            "rate_limited": sum(
                1 for e in self._rate_limits.values()
                if e.blocked_until > time.time()
            ),
        }
