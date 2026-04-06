"""
Atlas Security — Allowlist Manager.

Manages allowlists for users, domains, IPs, tools, paths, and channels.
Provides fast lookup for access control decisions with JSON persistence.

Inspired by OpenClaw's allowlist security architecture.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AllowlistEntryType(Enum):
    """Types of allowlist entries."""

    USER = "user"
    DOMAIN = "domain"
    IP = "ip"
    TOOL = "tool"
    PATH = "path"
    CHANNEL = "channel"
    AGENT = "agent"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Allowlist Manager
# ---------------------------------------------------------------------------

class AllowlistManager:
    """Manages allowlists for access control.

    Maintains per-type sets of allowed identifiers with O(1) lookup
    performance. Supports wildcard patterns for flexible matching and
    persists to JSON for durability.

    Usage::

        alm = AllowlistManager("/path/to/allowlist.json")

        # Add entries
        alm.add(AllowlistEntryType.USER, "admin@example.com")
        alm.add(AllowlistEntryType.DOMAIN, "github.com")
        alm.add(AllowlistEntryType.TOOL, "bash")
        alm.add(AllowlistEntryType.IP, "192.168.1.*")

        # Check access
        if alm.check(AllowlistEntryType.USER, "admin@example.com"):
            print("Access granted")

        # List all allowed tools
        tools = alm.list_allowed(AllowlistEntryType.TOOL)
    """

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        auto_save: bool = True,
        default_allow: bool = False,
    ) -> None:
        """Initialize the allowlist manager.

        Args:
            persistence_path: Path to the JSON allowlist file.
            auto_save: Automatically save after mutations.
            default_allow: Whether to allow by default when no allowlist
                           is configured for a type.
        """
        self._lists: Dict[AllowlistEntryType, Set[str]] = {
            entry_type: set() for entry_type in AllowlistEntryType
        }
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._auto_save = auto_save
        self._default_allow = default_allow

        if persistence_path:
            self._persistence_path = Path(persistence_path)
        else:
            self._persistence_path = Path("allowlist.json")

        self._load()
        logger.info(
            "AllowlistManager initialized (%d entries, default_allow=%s)",
            self._total_count(), default_allow,
        )

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def add(
        self,
        entry_type: AllowlistEntryType,
        identifier: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an entry to an allowlist.

        Args:
            entry_type: The type of allowlist to add to.
            identifier: The identifier to allow.
            metadata: Optional metadata for this entry.
        """
        normalized = identifier.strip().lower()
        self._lists[entry_type].add(normalized)

        if metadata:
            meta_key = f"{entry_type.value}:{normalized}"
            self._metadata[meta_key] = metadata

        if self._auto_save:
            self.save()

        logger.debug("Allowlisted %s:%s", entry_type.value, identifier)

    def add_batch(
        self,
        entry_type: AllowlistEntryType,
        identifiers: List[str],
    ) -> int:
        """Add multiple entries to an allowlist.

        Args:
            entry_type: The type of allowlist.
            identifiers: List of identifiers to add.

        Returns:
            Number of entries added.
        """
        count = 0
        for identifier in identifiers:
            normalized = identifier.strip().lower()
            if normalized not in self._lists[entry_type]:
                self._lists[entry_type].add(normalized)
                count += 1

        if self._auto_save and count > 0:
            self.save()

        logger.info("Added %d entries to %s allowlist", count, entry_type.value)
        return count

    def remove(
        self,
        entry_type: AllowlistEntryType,
        identifier: str,
    ) -> bool:
        """Remove an entry from an allowlist.

        Args:
            entry_type: The type of allowlist.
            identifier: The identifier to remove.

        Returns:
            True if found and removed.
        """
        normalized = identifier.strip().lower()
        removed = self._lists[entry_type].discard(normalized)

        # Clean up metadata
        meta_key = f"{entry_type.value}:{normalized}"
        self._metadata.pop(meta_key, None)

        if removed and self._auto_save:
            self.save()

        if normalized in self._lists[entry_type]:
            self._lists[entry_type].discard(normalized)

        return True

    def check(
        self,
        entry_type: AllowlistEntryType,
        identifier: str,
    ) -> bool:
        """Check if an identifier is allowed.

        Supports exact matching and wildcard patterns (``*``).

        If no entries exist for the given type and ``default_allow`` is
        True, returns True.

        Args:
            entry_type: The type of allowlist to check.
            identifier: The identifier to look up.

        Returns:
            True if the identifier is allowed.
        """
        normalized = identifier.strip().lower()
        entry_set = self._lists[entry_type]

        # Exact match
        if normalized in entry_set:
            return True

        # Wildcard pattern matching
        for pattern in entry_set:
            if "*" in pattern:
                if self._match_wildcard(normalized, pattern):
                    return True

        # Default allow for empty lists
        if not entry_set and self._default_allow:
            return True

        return False

    def is_empty(self, entry_type: AllowlistEntryType) -> bool:
        """Check if an allowlist has no entries.

        Args:
            entry_type: The type of allowlist.

        Returns:
            True if the allowlist is empty.
        """
        return len(self._lists[entry_type]) == 0

    # ------------------------------------------------------------------
    # Query Operations
    # ------------------------------------------------------------------

    def list_allowed(self, entry_type: AllowlistEntryType) -> List[str]:
        """List all entries in an allowlist.

        Args:
            entry_type: The type of allowlist.

        Returns:
            Sorted list of allowed identifiers.
        """
        return sorted(self._lists[entry_type])

    def list_all(self) -> Dict[str, List[str]]:
        """List all entries across all allowlist types.

        Returns:
            Dictionary mapping entry type names to sorted entry lists.
        """
        return {
            entry_type.value: sorted(entries)
            for entry_type, entries in self._lists.items()
            if entries
        }

    def count(self, entry_type: Optional[AllowlistEntryType] = None) -> int:
        """Count allowlist entries.

        Args:
            entry_type: Specific type to count (None for total).

        Returns:
            Number of entries.
        """
        if entry_type is not None:
            return len(self._lists[entry_type])
        return self._total_count()

    def get_metadata(
        self,
        entry_type: AllowlistEntryType,
        identifier: str,
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific allowlist entry.

        Args:
            entry_type: The entry type.
            identifier: The identifier.

        Returns:
            Metadata dictionary, or None if not found.
        """
        normalized = identifier.strip().lower()
        meta_key = f"{entry_type.value}:{normalized}"
        return self._metadata.get(meta_key)

    # ------------------------------------------------------------------
    # Bulk Operations
    # ------------------------------------------------------------------

    def clear_type(self, entry_type: AllowlistEntryType) -> int:
        """Clear all entries for a specific allowlist type.

        Args:
            entry_type: The type to clear.

        Returns:
            Number of entries removed.
        """
        count = len(self._lists[entry_type])
        self._lists[entry_type].clear()

        # Clean up related metadata
        keys_to_remove = [
            k for k in self._metadata
            if k.startswith(f"{entry_type.value}:")
        ]
        for key in keys_to_remove:
            del self._metadata[key]

        if self._auto_save and count > 0:
            self.save()

        logger.info("Cleared %d entries from %s allowlist", count, entry_type.value)
        return count

    def clear_all(self) -> int:
        """Clear all allowlists.

        Returns:
            Total number of entries removed.
        """
        total = self._total_count()
        for entry_type in self._lists:
            self._lists[entry_type].clear()
        self._metadata.clear()
        if self._auto_save:
            self.save()
        logger.info("Cleared all %d allowlist entries", total)
        return total

    def import_entries(
        self,
        data: Dict[str, List[str]],
    ) -> int:
        """Import entries from a dictionary.

        Args:
            data: Dictionary mapping entry type names to identifier lists.

        Returns:
            Number of entries imported.
        """
        count = 0
        for type_name, identifiers in data.items():
            try:
                entry_type = AllowlistEntryType(type_name)
                for identifier in identifiers:
                    self._lists[entry_type].add(identifier.strip().lower())
                    count += 1
            except ValueError:
                logger.warning("Unknown allowlist type: %s", type_name)

        if self._auto_save and count > 0:
            self.save()

        logger.info("Imported %d allowlist entries", count)
        return count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save allowlists to the persistence file."""
        if self._persistence_path is None:
            return

        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "default_allow": self._default_allow,
            "allowlists": {
                entry_type.value: sorted(entries)
                for entry_type, entries in self._lists.items()
                if entries
            },
            "metadata": self._metadata,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        temp_path = str(self._persistence_path) + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, str(self._persistence_path))

        logger.debug("Saved allowlists to %s", self._persistence_path)

    def _load(self) -> None:
        """Load allowlists from the persistence file."""
        if not self._persistence_path.exists():
            return

        try:
            with open(self._persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            allowlists = data.get("allowlists", {})
            for type_name, identifiers in allowlists.items():
                try:
                    entry_type = AllowlistEntryType(type_name)
                    self._lists[entry_type] = set(i.lower() for i in identifiers)
                except ValueError:
                    logger.warning("Unknown allowlist type in file: %s", type_name)

            self._metadata = data.get("metadata", {})
            self._default_allow = data.get("default_allow", self._default_allow)

            logger.info(
                "Loaded allowlists from %s (%d entries)",
                self._persistence_path, self._total_count(),
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load allowlists: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _total_count(self) -> int:
        """Count total entries across all types."""
        return sum(len(entries) for entries in self._lists.values())

    @staticmethod
    def _match_wildcard(text: str, pattern: str) -> bool:
        """Match text against a wildcard pattern.

        Args:
            text: Text to match.
            pattern: Pattern with * wildcards.

        Returns:
            True if the text matches the pattern.
        """
        import fnmatch
        return fnmatch.fnmatch(text, pattern)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all allowlists.

        Returns:
            Dictionary with counts per type and total.
        """
        return {
            "total_entries": self._total_count(),
            "default_allow": self._default_allow,
            "by_type": {
                entry_type.value: len(entries)
                for entry_type, entries in self._lists.items()
            },
            "persistence_path": str(self._persistence_path),
        }

    def __len__(self) -> int:
        return self._total_count()

    def __repr__(self) -> str:
        return f"<AllowlistManager entries={self._total_count()}>"
