"""
Atlas Channels — Channel-to-Agent Bindings.

Manages persistent bindings between messaging channels/accounts and agent
identifiers. Enables consistent routing of messages from specific users
across channels to the appropriate agent instance.

Persists bindings to a JSON file for durability across restarts.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from atlas.channels.base import ChannelType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class BindingEntry:
    """A single channel-to-agent binding record.

    Attributes:
        channel_type: The messaging channel type.
        account_id: Platform-specific user/account identifier.
        agent_id: The agent ID to handle messages from this account.
        session_key: Consistent session key for conversation continuity.
        display_name: Optional human-readable name for the bound user.
        created_at: When this binding was created.
        updated_at: When this binding was last updated.
        metadata: Additional metadata (preferences, tags, notes).
    """

    def __init__(
        self,
        channel_type: ChannelType = ChannelType.WEBCHAT,
        account_id: str = "",
        agent_id: str = "",
        session_key: str = "",
        display_name: str = "",
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.channel_type = channel_type
        self.account_id = account_id
        self.agent_id = agent_id
        self.session_key = session_key
        self.display_name = display_name
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.updated_at = updated_at or datetime.now(timezone.utc).isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize binding to a dictionary."""
        return {
            "channel_type": self.channel_type.value,
            "account_id": self.account_id,
            "agent_id": self.agent_id,
            "session_key": self.session_key,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BindingEntry":
        """Deserialize binding from a dictionary."""
        data = data.copy()
        if "channel_type" in data and isinstance(data["channel_type"], str):
            data["channel_type"] = ChannelType(data["channel_type"])
        return cls(**data)

    @property
    def composite_key(self) -> str:
        """Composite key combining channel type and account ID."""
        return f"{self.channel_type.value}:{self.account_id}"

    def __repr__(self) -> str:
        return (
            f"<BindingEntry "
            f"channel={self.channel_type.value} "
            f"account={self.account_id} "
            f"agent={self.agent_id}>"
        )


# ---------------------------------------------------------------------------
# Channel Bindings Manager
# ---------------------------------------------------------------------------

class ChannelBindings:
    """Manages channel-to-agent bindings with JSON persistence.

    Provides CRUD operations for bindings, supports filtering by channel type,
    and persists the binding database to a JSON file for durability.

    The binding database is organized as a dictionary keyed by composite keys
    (``channel_type:account_id``) for O(1) lookup performance.

    Usage::

        bindings = ChannelBindings("/path/to/bindings.json")

        # Create a binding
        bindings.bind(
            channel_type=ChannelType.TELEGRAM,
            account_id="user_123",
            agent_id="agent_main",
        )

        # Look up a binding
        entry = bindings.get_binding(ChannelType.TELEGRAM, "user_123")
        print(entry.agent_id)  # "agent_main"

        # List all Telegram bindings
        telegram_bindings = bindings.list_bindings(ChannelType.TELEGRAM)
    """

    DEFAULT_FILENAME = "channel_bindings.json"

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        auto_save: bool = True,
    ) -> None:
        """Initialize the channel bindings manager.

        Args:
            persistence_path: Path to the JSON persistence file.
                              If None, uses default location in current directory.
            auto_save: Whether to automatically save after each mutation.
        """
        self._bindings: Dict[str, BindingEntry] = {}
        self._auto_save = auto_save

        if persistence_path:
            self._persistence_path = Path(persistence_path)
        else:
            self._persistence_path = Path(self.DEFAULT_FILENAME)

        self._load()

        logger.info(
            "ChannelBindings initialized (%d bindings, auto_save=%s, path=%s)",
            len(self._bindings), auto_save, self._persistence_path,
        )

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def bind(
        self,
        channel_type: ChannelType,
        account_id: str,
        agent_id: str,
        session_key: str = "",
        display_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BindingEntry:
        """Create or update a channel-to-agent binding.

        If a binding for the same channel_type + account_id already exists,
        it will be updated with the new agent_id and metadata.

        Args:
            channel_type: The messaging channel type.
            account_id: Platform-specific user/account identifier.
            agent_id: The agent ID to route messages to.
            session_key: Session key for conversation continuity.
            display_name: Optional display name.
            metadata: Additional metadata.

        Returns:
            The created or updated BindingEntry.
        """
        composite = f"{channel_type.value}:{account_id}"
        now = datetime.now(timezone.utc).isoformat()

        existing = self._bindings.get(composite)
        if existing:
            # Update existing binding
            existing.agent_id = agent_id
            existing.session_key = session_key or existing.session_key
            existing.display_name = display_name or existing.display_name
            existing.updated_at = now
            if metadata:
                existing.metadata.update(metadata)
            entry = existing
            logger.info(
                "Updated binding: %s -> agent %s",
                composite, agent_id,
            )
        else:
            # Create new binding
            entry = BindingEntry(
                channel_type=channel_type,
                account_id=account_id,
                agent_id=agent_id,
                session_key=session_key,
                display_name=display_name,
                metadata=metadata or {},
            )
            logger.info(
                "Created binding: %s -> agent %s",
                composite, agent_id,
            )

        self._bindings[composite] = entry

        if self._auto_save:
            self.save()

        return entry

    def unbind(
        self,
        channel_type: ChannelType,
        account_id: str,
    ) -> bool:
        """Remove a channel-to-agent binding.

        Args:
            channel_type: The messaging channel type.
            account_id: The account identifier to unbind.

        Returns:
            True if the binding was found and removed.
        """
        composite = f"{channel_type.value}:{account_id}"
        entry = self._bindings.pop(composite, None)

        if entry is not None:
            logger.info("Removed binding: %s", composite)
            if self._auto_save:
                self.save()
            return True

        logger.debug("Binding not found: %s", composite)
        return False

    def get_binding(
        self,
        channel_type: ChannelType,
        account_id: str,
    ) -> Optional[BindingEntry]:
        """Look up a specific binding by channel type and account ID.

        Args:
            channel_type: The messaging channel type.
            account_id: The account identifier.

        Returns:
            The BindingEntry if found, None otherwise.
        """
        composite = f"{channel_type.value}:{account_id}"
        return self._bindings.get(composite)

    def get_agent_for_account(
        self,
        channel_type: ChannelType,
        account_id: str,
    ) -> Optional[str]:
        """Get the agent ID for a specific account.

        Convenience shortcut that returns just the agent_id string.

        Args:
            channel_type: The messaging channel type.
            account_id: The account identifier.

        Returns:
            Agent ID if binding exists, None otherwise.
        """
        entry = self.get_binding(channel_type, account_id)
        return entry.agent_id if entry else None

    def get_session_key(
        self,
        channel_type: ChannelType,
        account_id: str,
    ) -> str:
        """Get the session key for a specific account.

        Args:
            channel_type: The messaging channel type.
            account_id: The account identifier.

        Returns:
            Session key string, or empty string if no binding exists.
        """
        entry = self.get_binding(channel_type, account_id)
        return entry.session_key if entry else ""

    # ------------------------------------------------------------------
    # Query Operations
    # ------------------------------------------------------------------

    def list_bindings(
        self,
        channel_type: Optional[ChannelType] = None,
        agent_id: Optional[str] = None,
    ) -> List[BindingEntry]:
        """List bindings with optional filtering.

        Args:
            channel_type: Filter by channel type (None for all).
            agent_id: Filter by agent ID (None for all).

        Returns:
            List of matching BindingEntry objects.
        """
        results = list(self._bindings.values())

        if channel_type is not None:
            results = [
                e for e in results
                if e.channel_type == channel_type
            ]

        if agent_id is not None:
            results = [
                e for e in results
                if e.agent_id == agent_id
            ]

        return results

    def list_bindings_dict(
        self,
        channel_type: Optional[ChannelType] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List bindings as dictionaries.

        Args:
            channel_type: Filter by channel type.
            agent_id: Filter by agent ID.

        Returns:
            List of binding dictionaries.
        """
        entries = self.list_bindings(channel_type, agent_id)
        return [e.to_dict() for e in entries]

    def list_channels(self) -> List[ChannelType]:
        """List all channel types that have bindings.

        Returns:
            Sorted list of unique ChannelType values.
        """
        types = {e.channel_type for e in self._bindings.values()}
        return sorted(types, key=lambda ct: ct.value)

    def list_agents(self) -> List[str]:
        """List all agent IDs that have bindings.

        Returns:
            Sorted list of unique agent IDs.
        """
        agents = {e.agent_id for e in self._bindings.values()}
        return sorted(agents)

    def count_bindings(
        self,
        channel_type: Optional[ChannelType] = None,
        agent_id: Optional[str] = None,
    ) -> int:
        """Count bindings with optional filtering.

        Args:
            channel_type: Filter by channel type.
            agent_id: Filter by agent ID.

        Returns:
            Number of matching bindings.
        """
        return len(self.list_bindings(channel_type, agent_id))

    def has_binding(
        self,
        channel_type: ChannelType,
        account_id: str,
    ) -> bool:
        """Check if a binding exists.

        Args:
            channel_type: The messaging channel type.
            account_id: The account identifier.

        Returns:
            True if a binding exists.
        """
        composite = f"{channel_type.value}:{account_id}"
        return composite in self._bindings

    # ------------------------------------------------------------------
    # Bulk Operations
    # ------------------------------------------------------------------

    def unbind_channel(self, channel_type: ChannelType) -> int:
        """Remove all bindings for a channel type.

        Args:
            channel_type: The channel type to clear.

        Returns:
            Number of bindings removed.
        """
        to_remove = [
            key for key, entry in self._bindings.items()
            if entry.channel_type == channel_type
        ]
        for key in to_remove:
            del self._bindings[key]

        logger.info(
            "Removed %d bindings for channel %s",
            len(to_remove), channel_type.value,
        )

        if self._auto_save and to_remove:
            self.save()

        return len(to_remove)

    def unbind_agent(self, agent_id: str) -> int:
        """Remove all bindings for an agent.

        Args:
            agent_id: The agent ID to clear.

        Returns:
            Number of bindings removed.
        """
        to_remove = [
            key for key, entry in self._bindings.items()
            if entry.agent_id == agent_id
        ]
        for key in to_remove:
            del self._bindings[key]

        logger.info(
            "Removed %d bindings for agent %s",
            len(to_remove), agent_id,
        )

        if self._auto_save and to_remove:
            self.save()

        return len(to_remove)

    def import_bindings(self, bindings: List[Dict[str, Any]]) -> int:
        """Import bindings from a list of dictionaries.

        Args:
            bindings: List of binding dictionaries.

        Returns:
            Number of bindings imported.
        """
        count = 0
        for data in bindings:
            try:
                entry = BindingEntry.from_dict(data)
                composite = entry.composite_key
                self._bindings[composite] = entry
                count += 1
            except Exception as exc:
                logger.error("Failed to import binding: %s", exc)

        if self._auto_save and count > 0:
            self.save()

        logger.info("Imported %d bindings", count)
        return count

    def clear_all(self) -> int:
        """Remove all bindings.

        Returns:
            Number of bindings removed.
        """
        count = len(self._bindings)
        self._bindings.clear()
        if self._auto_save:
            self.save()
        logger.info("Cleared all %d bindings", count)
        return count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save current bindings to the persistence file.

        Creates parent directories if needed. Writes atomically using a
        temporary file to prevent corruption.
        """
        if self._persistence_path is None:
            return

        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "1.0",
                "bindings": [
                    entry.to_dict()
                    for entry in self._bindings.values()
                ],
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "count": len(self._bindings),
            }

            # Atomic write via temp file
            temp_path = str(self._persistence_path) + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            os.replace(temp_path, str(self._persistence_path))

            logger.debug(
                "Saved %d bindings to %s",
                len(self._bindings), self._persistence_path,
            )
        except Exception as exc:
            logger.error(
                "Failed to save bindings to %s: %s",
                self._persistence_path, exc,
            )

    def _load(self) -> None:
        """Load bindings from the persistence file.

        Silently handles missing or corrupted files.
        """
        if self._persistence_path is None:
            return

        if not self._persistence_path.exists():
            logger.debug(
                "No persistence file at %s, starting empty",
                self._persistence_path,
            )
            return

        try:
            with open(self._persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            bindings_data = data.get("bindings", [])
            for entry_data in bindings_data:
                try:
                    entry = BindingEntry.from_dict(entry_data)
                    self._bindings[entry.composite_key] = entry
                except Exception as exc:
                    logger.warning(
                        "Failed to load binding entry: %s", exc,
                    )

            logger.info(
                "Loaded %d bindings from %s",
                len(self._bindings), self._persistence_path,
            )
        except json.JSONDecodeError as exc:
            logger.error(
                "Corrupted bindings file %s: %s",
                self._persistence_path, exc,
            )
        except Exception as exc:
            logger.error(
                "Failed to load bindings from %s: %s",
                self._persistence_path, exc,
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current bindings.

        Returns:
            Dictionary with counts, channel breakdown, and agent breakdown.
        """
        channel_counts: Dict[str, int] = {}
        agent_counts: Dict[str, int] = {}

        for entry in self._bindings.values():
            ct = entry.channel_type.value
            channel_counts[ct] = channel_counts.get(ct, 0) + 1
            agent_counts[entry.agent_id] = agent_counts.get(entry.agent_id, 0) + 1

        return {
            "total_bindings": len(self._bindings),
            "unique_channels": len(channel_counts),
            "unique_agents": len(agent_counts),
            "by_channel": channel_counts,
            "by_agent": agent_counts,
            "persistence_path": str(self._persistence_path),
        }

    def __len__(self) -> int:
        return len(self._bindings)

    def __repr__(self) -> str:
        return f"<ChannelBindings count={len(self._bindings)} path={self._persistence_path}>"
