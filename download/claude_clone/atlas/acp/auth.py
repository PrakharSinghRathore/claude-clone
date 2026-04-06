"""
ACP Authentication — API key authentication, JWT token support,
role-based access control, and token refresh.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path.home() / ".claude_clone" / "acp"
_SECRETS_FILE = "secrets.json"
_API_KEYS_FILE = "api_keys.json"


class Role:
    """Predefined roles for access control."""

    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    TOOL = "tool"
    IDE = "ide"


@dataclass
class APIKey:
    """An API key with associated metadata."""

    key_hash: str
    name: str
    role: str = Role.USER
    created_at: str = ""
    last_used: str = ""
    rate_limit: int = 100  # requests per minute
    is_active: bool = True
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def hash_key(key: str) -> str:
        """Hash an API key for safe storage."""
        return hashlib.sha256(key.encode()).hexdigest()


@dataclass
class JWTToken:
    """A JWT-like token (simplified, HMAC-SHA256 signed)."""

    token_id: str
    user_id: str
    role: str
    issued_at: float
    expires_at: float
    session_id: Optional[str] = None


class AuthManager:
    """
    Manages API key authentication, JWT tokens, and role-based access.

    API keys are stored hashed on disk. Tokens are signed with a
    server secret and support refresh.
    """

    def __init__(
        self,
        data_dir: str | Path = _DEFAULT_DATA_DIR,
        secret_key: Optional[str] = None,
        token_expiry_seconds: float = 3600.0,
        refresh_expiry_seconds: float = 86400.0,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._secret_key = (secret_key or os.environ.get("ACP_SECRET_KEY") or secrets.token_hex(32)).encode()
        self._token_expiry = token_expiry_seconds
        self._refresh_expiry = refresh_expiry_seconds
        self._api_keys: dict[str, APIKey] = {}  # key_hash -> APIKey
        self._revoked_tokens: set[str] = set()
        self._load_data()

    # ------------------------------------------------------------------
    # API Key management
    # ------------------------------------------------------------------

    def create_api_key(self, name: str, role: str = Role.USER) -> str:
        """
        Create a new API key. Returns the plaintext key (shown only once).

        The key is stored as a SHA-256 hash; the plaintext is returned
        to the caller for immediate use.
        """
        raw_key = f"hcp_{secrets.token_urlsafe(32)}"
        key_hash = APIKey.hash_key(raw_key)
        now = datetime.now(timezone.utc).isoformat()
        api_key = APIKey(
            key_hash=key_hash,
            name=name,
            role=role,
            created_at=now,
        )
        self._api_keys[key_hash] = api_key
        self._save_api_keys()
        logger.info("Created API key %r (role=%s)", name, role)
        return raw_key

    def validate_api_key(self, key: str) -> Optional[APIKey]:
        """
        Validate an API key and return its metadata if valid.

        Returns None if the key is invalid, inactive, or rate-limited.
        """
        key_hash = APIKey.hash_key(key)
        api_key = self._api_keys.get(key_hash)
        if api_key is None:
            return None
        if not api_key.is_active:
            logger.warning("API key %r is inactive", api_key.name)
            return None
        # Update last used
        api_key.last_used = datetime.now(timezone.utc).isoformat()
        return api_key

    def revoke_api_key(self, key: str) -> bool:
        """Revoke an API key by its plaintext value."""
        key_hash = APIKey.hash_key(key)
        api_key = self._api_keys.get(key_hash)
        if api_key is None:
            return False
        api_key.is_active = False
        self._save_api_keys()
        logger.info("Revoked API key %r", api_key.name)
        return True

    def list_api_keys(self) -> list[dict]:
        """List all API keys (without exposing hashed values)."""
        return [
            {
                "name": k.name,
                "role": k.role,
                "created_at": k.created_at,
                "last_used": k.last_used,
                "is_active": k.is_active,
                "rate_limit": k.rate_limit,
            }
            for k in self._api_keys.values()
        ]

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def create_token(self, user_id: str, role: str = Role.USER, session_id: Optional[str] = None) -> str:
        """
        Create a signed authentication token.

        Returns a JSON string containing the token payload and signature.
        """
        now = time.time()
        token = JWTToken(
            token_id=secrets.token_hex(16),
            user_id=user_id,
            role=role,
            issued_at=now,
            expires_at=now + self._token_expiry,
            session_id=session_id,
        )
        payload = json.dumps({
            "tid": token.token_id,
            "uid": token.user_id,
            "role": token.role,
            "iat": token.issued_at,
            "exp": token.expires_at,
            "sid": token.session_id,
        }, separators=(",", ":"))
        signature = hmac.new(self._secret_key, payload.encode(), hashlib.sha256).hexdigest()
        token_str = f"{payload}.{signature}"
        return token_str

    def validate_token(self, token_str: str) -> Optional[JWTToken]:
        """
        Validate a token string and return its parsed data.

        Returns None if the token is expired, revoked, or has an invalid signature.
        """
        try:
            parts = token_str.rsplit(".", 1)
            if len(parts) != 2:
                return None
            payload_str, signature = parts
            expected_sig = hmac.new(self._secret_key, payload_str.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected_sig):
                logger.warning("Invalid token signature")
                return None
            data = json.loads(payload_str)
            if data["tid"] in self._revoked_tokens:
                return None
            if time.time() > data["exp"]:
                return None
            return JWTToken(
                token_id=data["tid"],
                user_id=data["uid"],
                role=data["role"],
                issued_at=data["iat"],
                expires_at=data["exp"],
                session_id=data.get("sid"),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def refresh_token(self, token_str: str) -> Optional[str]:
        """
        Refresh an expiring token.

        Returns a new token string if the existing token is within
        the refresh window, or None if it cannot be refreshed.
        """
        token = self.validate_token(token_str)
        if token is None:
            return None
        # Check if within refresh window
        if time.time() > token.issued_at + self._refresh_expiry:
            return None
        # Revoke old token
        self._revoked_tokens.add(token.token_id)
        # Issue new token
        return self.create_token(token.user_id, token.role, token.session_id)

    def revoke_token(self, token_str: str) -> bool:
        """Revoke a token so it cannot be used again."""
        token = self.validate_token(token_str)
        if token is None:
            return False
        self._revoked_tokens.add(token.token_id)
        return True

    # ------------------------------------------------------------------
    # Role-based access control
    # ------------------------------------------------------------------

    @staticmethod
    def has_permission(role: str, required_role: str) -> bool:
        """
        Check if a role has at least the permissions of the required role.

        Hierarchy: admin > user > tool > ide > readonly
        """
        hierarchy = {
            Role.ADMIN: 5,
            Role.USER: 4,
            Role.TOOL: 3,
            Role.IDE: 2,
            Role.READONLY: 1,
        }
        return hierarchy.get(role, 0) >= hierarchy.get(required_role, 0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load API keys from disk."""
        keys_file = self.data_dir / _API_KEYS_FILE
        if keys_file.exists():
            try:
                data = json.loads(keys_file.read_text(encoding="utf-8"))
                for key_data in data.get("keys", []):
                    api_key = APIKey(**key_data)
                    self._api_keys[api_key.key_hash] = api_key
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load API keys")

    def _save_api_keys(self) -> None:
        """Persist API keys to disk."""
        keys_file = self.data_dir / _API_KEYS_FILE
        try:
            data = {
                "keys": [
                    {
                        "key_hash": k.key_hash,
                        "name": k.name,
                        "role": k.role,
                        "created_at": k.created_at,
                        "last_used": k.last_used,
                        "rate_limit": k.rate_limit,
                        "is_active": k.is_active,
                        "metadata": k.metadata,
                    }
                    for k in self._api_keys.values()
                ],
            }
            keys_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            logger.exception("Failed to save API keys")
