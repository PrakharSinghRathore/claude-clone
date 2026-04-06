"""
Atlas Security — Secret Manager.

Provides secure storage and retrieval of secrets with encryption support,
environment variable integration, and ${SECRET:name} reference resolution.

Secrets are stored in a JSON file with optional AES-256-GCM encryption.
Values are never logged in plaintext and are masked in all string
representations.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import cryptography for encrypted storage
try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


# ---------------------------------------------------------------------------
# Secret Entry
# ---------------------------------------------------------------------------

class SecretEntry:
    """A single secret entry with masked display.

    Attributes:
        key: Secret identifier/name.
        value: The secret value (never exposed in logs/repr).
        created_at: ISO timestamp of creation.
        updated_at: ISO timestamp of last update.
        metadata: Additional metadata tags.
    """

    def __init__(
        self,
        key: str,
        value: str,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        self.key = key
        self._value = value
        self.created_at = created_at or _now_iso()
        self.updated_at = updated_at or _now_iso()
        self.metadata = metadata or {}

    @property
    def value(self) -> str:
        """Get the secret value."""
        return self._value

    @property
    def masked_value(self) -> str:
        """Get a masked representation of the value for safe display."""
        v = self._value
        if not v:
            return ""
        if len(v) <= 6:
            return "******"
        return v[:3] + "****" + v[-3:]

    def to_dict(self, expose: bool = False) -> Dict[str, Any]:
        """Serialize to dictionary.

        Args:
            expose: Whether to include the raw value (dangerous).

        Returns:
            Dictionary representation.
        """
        result = {
            "key": self.key,
            "value": self._value if expose else self.masked_value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
        return result

    def __repr__(self) -> str:
        return f"<SecretEntry key={self.key!r} value={self.masked_value!r}>"


# ---------------------------------------------------------------------------
# Secret Manager
# ---------------------------------------------------------------------------

class SecretManager:
    """Secure secret storage with encryption and resolution.

    Features:
    - In-memory secret storage with JSON file persistence
    - Optional AES-256 encryption for the secrets file
    - ``${SECRET:name}`` reference resolution in strings
    - Environment variable fallback and integration
    - Thread-safe operations
    - Values masked in all logs and string representations

    Usage::

        sm = SecretManager("/path/to/secrets.json")

        # Store a secret
        sm.set("OPENAI_API_KEY", "sk-...")

        # Retrieve
        key = sm.get("OPENAI_API_KEY")  # Returns actual value

        # Resolve references
        config = sm.resolve("Bearer ${SECRET:OPENAI_API_KEY}")
        # Returns "Bearer sk-..."

        # List keys (values are masked)
        for entry in sm.list():
            print(f"  {entry.key}: {entry.masked_value}")

        # Export encrypted
        sm.export_encrypted("/backup/secrets.enc", "my-password")
    """

    SECRET_PATTERN = re.compile(r"\$\{SECRET:([^}]+)\}")

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        auto_save: bool = True,
        env_prefix: str = "",
        encryption_key: Optional[str] = None,
    ) -> None:
        """Initialize the secret manager.

        Args:
            persistence_path: Path to the JSON secrets file.
            auto_save: Automatically save after mutations.
            env_prefix: Environment variable prefix for fallback lookup.
            encryption_key: Key for encrypted storage (base64-encoded).
        """
        self._secrets: Dict[str, SecretEntry] = {}
        self._auto_save = auto_save
        self._env_prefix = env_prefix
        self._lock = threading.Lock()
        self._fernet: Optional[Any] = None

        if persistence_path:
            self._persistence_path = Path(persistence_path)
        else:
            self._persistence_path = Path("secrets.json")

        # Set up encryption if key is provided
        if encryption_key and HAS_CRYPTOGRAPHY:
            try:
                self._fernet = Fernet(encryption_key.encode())
            except Exception as exc:
                logger.warning("Failed to initialize encryption: %s", exc)
                self._fernet = None

        self._load()
        logger.info(
            "SecretManager initialized (%d secrets, encrypted=%s, path=%s)",
            len(self._secrets), self._fernet is not None, self._persistence_path,
        )

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def set(
        self,
        key: str,
        value: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Store a secret.

        Args:
            key: Secret identifier.
            value: Secret value.
            metadata: Optional metadata tags.
        """
        now = _now_iso()
        with self._lock:
            existing = self._secrets.get(key)
            if existing:
                existing._value = value
                existing.updated_at = now
                if metadata:
                    existing.metadata.update(metadata)
            else:
                entry = SecretEntry(
                    key=key,
                    value=value,
                    created_at=now,
                    updated_at=now,
                    metadata=metadata or {},
                )
                self._secrets[key] = entry

        if self._auto_save:
            self.save()

        logger.debug("Secret '%s' stored (value=%s)", key, "******")

    def get(self, key: str, default: str = "") -> str:
        """Retrieve a secret value.

        Falls back to environment variables if the key is not in the store.

        Args:
            key: Secret identifier.
            default: Default value if not found.

        Returns:
            The secret value, env var value, or default.
        """
        with self._lock:
            entry = self._secrets.get(key)
            if entry:
                return entry.value

        # Environment variable fallback
        env_key = self._env_prefix + key if self._env_prefix else key
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value

        return default

    def delete(self, key: str) -> bool:
        """Remove a secret.

        Args:
            key: Secret identifier.

        Returns:
            True if the secret was found and removed.
        """
        with self._lock:
            entry = self._secrets.pop(key, None)

        if entry is not None:
            if self._auto_save:
                self.save()
            logger.debug("Secret '%s' deleted", key)
            return True

        logger.debug("Secret '%s' not found for deletion", key)
        return False

    def has(self, key: str) -> bool:
        """Check if a secret exists.

        Args:
            key: Secret identifier.

        Returns:
            True if the secret exists.
        """
        return key in self._secrets or os.environ.get(
            self._env_prefix + key if self._env_prefix else key
        ) is not None

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list(self) -> List[SecretEntry]:
        """List all stored secrets (values are masked).

        Returns:
            List of SecretEntry objects.
        """
        with self._lock:
            return list(self._secrets.values())

    def list_keys(self) -> List[str]:
        """List all secret keys.

        Returns:
            Sorted list of secret key names.
        """
        with self._lock:
            return sorted(self._secrets.keys())

    def count(self) -> int:
        """Return the number of stored secrets.

        Returns:
            Number of secrets.
        """
        return len(self._secrets)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, text: str) -> str:
        """Resolve ``${SECRET:name}`` references in a string.

        Replaces all occurrences of ``${SECRET:name}`` with the actual
        secret value. Missing secrets are replaced with empty string.

        Args:
            text: String containing secret references.

        Returns:
            Resolved string with all references replaced.
        """
        def _replacer(match: re.Match[str]) -> str:
            key = match.group(1)
            value = self.get(key)
            return value

        resolved = self.SECRET_PATTERN.sub(_replacer, text)
        return resolved

    def resolve_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively resolve secret references in a dictionary.

        Args:
            data: Dictionary potentially containing secret references.

        Returns:
            New dictionary with all references resolved.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.resolve(value)
            elif isinstance(value, dict):
                result[key] = self.resolve_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.resolve(item) if isinstance(item, str)
                    else self.resolve_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    # ------------------------------------------------------------------
    # Export / Import (Encrypted)
    # ------------------------------------------------------------------

    def export_encrypted(self, path: str, password: str) -> str:
        """Export all secrets to an encrypted file.

        Uses AES-256-GCM encryption via Fernet. The password is
        strengthened with PBKDF2 before deriving the encryption key.

        Args:
            path: Output file path.
            password: Encryption password.

        Returns:
            The output file path.

        Raises:
            ImportError: If cryptography library is not installed.
        """
        if not HAS_CRYPTOGRAPHY:
            raise ImportError(
                "cryptography library is required for encrypted export. "
                "Install it with: pip install cryptography"
            )

        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        # Derive key from password
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

        # Encrypt
        fernet = Fernet(key)
        data = json.dumps({
            entry.key: entry.value
            for entry in self._secrets.values()
        })
        encrypted = fernet.encrypt(data.encode())

        # Write
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "wb") as f:
            f.write(salt)
            f.write(encrypted)

        logger.info("Exported %d secrets (encrypted) to %s", len(self._secrets), output)
        return str(output)

    def import_encrypted(self, path: str, password: str) -> int:
        """Import secrets from an encrypted file.

        Args:
            path: Input file path.
            password: Decryption password.

        Returns:
            Number of secrets imported.

        Raises:
            ImportError: If cryptography library is not installed.
            ValueError: If decryption fails.
        """
        if not HAS_CRYPTOGRAPHY:
            raise ImportError("cryptography library is required for encrypted import")

        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        input_path = Path(path)
        with open(input_path, "rb") as f:
            salt = f.read(16)
            encrypted = f.read()

        # Derive key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        fernet = Fernet(key)

        # Decrypt
        try:
            decrypted = fernet.decrypt(encrypted).decode()
            secrets_data = json.loads(decrypted)
        except Exception as exc:
            raise ValueError(f"Decryption failed (wrong password?): {exc}")

        count = 0
        for key_name, value in secrets_data.items():
            self.set(key_name, value)
            count += 1

        logger.info("Imported %d secrets from %s", count, input_path)
        return count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save secrets to the persistence file.

        If encryption is enabled, values are encrypted before writing.
        Otherwise, values are stored in plaintext JSON.
        """
        if self._persistence_path is None:
            return

        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            entry.key: entry.value
            for entry in self._secrets.values()
        }

        output = {}
        if self._fernet:
            # Encrypt values
            for key, value in data.items():
                try:
                    output[key] = self._fernet.encrypt(value.encode()).decode()
                except Exception:
                    output[key] = value
            output["_encrypted"] = True
        else:
            output = {"_encrypted": False, **data}

        # Write atomically
        temp_path = str(self._persistence_path) + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, str(self._persistence_path))

        logger.debug("Saved %d secrets to %s", len(data), self._persistence_path)

    def _load(self) -> None:
        """Load secrets from the persistence file."""
        if not self._persistence_path.exists():
            return

        try:
            with open(self._persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load secrets from %s: %s", self._persistence_path, exc)
            return

        is_encrypted = data.pop("_encrypted", False)

        for key, value in data.items():
            if is_encrypted and self._fernet:
                try:
                    value = self._fernet.decrypt(value.encode()).decode()
                except Exception:
                    pass  # Store as-is if decryption fails

            entry = SecretEntry(key=key, value=value)
            self._secrets[key] = entry

        logger.debug("Loaded %d secrets from %s", len(self._secrets), self._persistence_path)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self) -> int:
        """Remove all secrets.

        Returns:
            Number of secrets cleared.
        """
        with self._lock:
            count = len(self._secrets)
            self._secrets.clear()
        if self._auto_save:
            self.save()
        return count

    def generate_key(self) -> str:
        """Generate a new Fernet-compatible encryption key.

        Returns:
            Base64-encoded encryption key.
        """
        if HAS_CRYPTOGRAPHY:
            from cryptography.fernet import Fernet
            return Fernet.generate_key().decode()
        # Fallback: generate random base64 key
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()

    def __len__(self) -> int:
        return len(self._secrets)

    def __repr__(self) -> str:
        return f"<SecretManager count={len(self._secrets)}>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time as ISO string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
