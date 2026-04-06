"""
Cross-platform message mirroring for the Atlas Gateway.

Mirrors messages between platforms with format conversion, thread
linking, and direction control (one-way or two-way).

Usage::

    mirror = MessageMirror(adapters, config)
    mirror.add_mirror_rule(
        source="telegram", target="discord",
        direction=MirrorDirection.BOTH,
    )
    await mirror.mirror_message(
        source_platform="telegram",
        source_chat_id="12345",
        text="Hello from Telegram!",
        source_user_id="alice",
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from atlas.gateway.config import GatewayConfig
from atlas.gateway.delivery import (
    DeliveryRouter, FormatConverter, PLATFORM_FORMAT_PREFERENCES,
)

logger = logging.getLogger("atlas.gateway.mirror")


# ──────────────────────────────────────────────────────────────────────────────
# Mirror Direction
# ──────────────────────────────────────────────────────────────────────────────

class MirrorDirection(str, Enum):
    """Direction of message mirroring."""

    ONE_WAY = "one_way"       # Source -> Target only
    BOTH = "both"             # Source <-> Target
    ONE_WAY_REVERSE = "one_way_reverse"  # Target -> Source only


# ──────────────────────────────────────────────────────────────────────────────
# Mirror Rule
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MirrorRule:
    """A rule defining message mirroring between two platforms."""

    source: str
    target: str
    direction: MirrorDirection = MirrorDirection.BOTH
    chat_mapping: Dict[str, str] = field(default_factory=dict)  # source_chat -> target_chat
    format_source: str = "markdown"
    format_target: str = "markdown"
    include_attachments: bool = True
    include_edits: bool = False
    include_reactions: bool = False
    include_replies: bool = True
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    enabled: bool = True
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "direction": self.direction.value,
            "chat_mapping": self.chat_mapping,
            "include_attachments": self.include_attachments,
            "include_edits": self.include_edits,
            "include_reactions": self.include_reactions,
            "include_replies": self.include_replies,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "enabled": self.enabled,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Mirror Event
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MirrorEvent:
    """A message event to be mirrored."""

    source_platform: str
    source_chat_id: str
    source_message_id: Optional[str]
    source_user_id: str
    source_username: Optional[str]
    text: str
    attachments: Optional[List[str]] = None
    reply_to: Optional[str] = None
    is_edit: bool = False
    is_system: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ──────────────────────────────────────────────────────────────────────────────
# Mirror Link
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MirrorLink:
    """Tracks the relationship between mirrored messages."""

    source_platform: str
    source_chat_id: str
    source_message_id: str
    target_platform: str
    target_chat_id: str
    target_message_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ──────────────────────────────────────────────────────────────────────────────
# Message Mirror
# ──────────────────────────────────────────────────────────────────────────────

class MessageMirror:
    """
    Cross-platform message mirroring engine.

    Features:
    - One-way and two-way mirroring between any platform pair
    - Per-chat mapping (source chat -> target chat)
    - Format conversion between platforms
    - Thread linking for cross-platform conversations
    - Edit mirroring
    - Configurable prefixes/suffixes
    - Mirror link tracking
    """

    def __init__(
        self,
        adapters: Optional[Dict[str, Any]] = None,
        config: Optional[GatewayConfig] = None,
        delivery_router: Optional[DeliveryRouter] = None,
    ):
        self._adapters = adapters or {}
        self._config = config
        self._delivery_router = delivery_router
        self._converter = FormatConverter()

        # Mirror rules
        self._rules: List[MirrorRule] = []

        # Thread links: (source_platform, source_chat_id) -> (target_platform, target_chat_id)
        self._thread_links: Dict[Tuple[str, str], Tuple[str, str]] = {}

        # Message links for edit mirroring
        self._message_links: Dict[Tuple[str, str, str], MirrorLink] = {}

        # Statistics
        self._stats = defaultdict(lambda: {"mirrored": 0, "failed": 0, "bytes": 0})

        # Load rules from config if available
        if config and config.mirror_pairs:
            for pair_data in config.mirror_pairs:
                rule = MirrorRule(
                    source=pair_data.get("source", ""),
                    target=pair_data.get("target", ""),
                    direction=MirrorDirection(pair_data.get("direction", "both")),
                    chat_mapping=pair_data.get("chat_mapping", {}),
                    prefix=pair_data.get("prefix"),
                    suffix=pair_data.get("suffix"),
                    include_attachments=pair_data.get("include_attachments", True),
                    include_edits=pair_data.get("include_edits", False),
                    name=pair_data.get("name"),
                )
                self._rules.append(rule)

    # ── Rule Management ───────────────────────────────────────────────────

    def add_mirror_rule(self, **kwargs: Any) -> MirrorRule:
        """
        Add a mirror rule.

        Parameters
        ----------
        source:
            Source platform name.
        target:
            Target platform name.
        direction:
            Mirror direction (one_way, both, one_way_reverse).
        chat_mapping:
            Dict mapping source chat IDs to target chat IDs.
        """
        rule = MirrorRule(**kwargs)
        self._rules.append(rule)
        logger.info(
            "Added mirror rule: %s -> %s (%s)",
            rule.source, rule.target, rule.direction.value,
        )
        return rule

    def remove_mirror_rule(self, name: Optional[str] = None, index: int = -1) -> bool:
        """Remove a mirror rule by name or index."""
        if name:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r.name != name]
            return len(self._rules) < before
        elif 0 <= index < len(self._rules):
            self._rules.pop(index)
            return True
        return False

    def get_rules(self) -> List[MirrorRule]:
        """Return all mirror rules."""
        return list(self._rules)

    def get_rules_for_platform(self, platform: str) -> List[MirrorRule]:
        """Return mirror rules involving a specific platform."""
        return [
            r for r in self._rules
            if r.enabled and (r.source == platform or r.target == platform)
        ]

    # ── Thread Linking ────────────────────────────────────────────────────

    def link_threads(
        self,
        source_platform: str,
        source_chat_id: str,
        target_platform: str,
        target_chat_id: str,
    ) -> None:
        """Link two chats for bidirectional mirroring."""
        self._thread_links[(source_platform, source_chat_id)] = (
            target_platform, target_chat_id
        )
        self._thread_links[(target_platform, target_chat_id)] = (
            source_platform, source_chat_id
        )

    def unlink_threads(
        self,
        platform: str,
        chat_id: str,
    ) -> None:
        """Remove thread links for a chat."""
        key = (platform, chat_id)
        if key in self._thread_links:
            other_key = self._thread_links.pop(key)
            self._thread_links.pop(other_key, None)

    def get_linked_chat(
        self, platform: str, chat_id: str,
    ) -> Optional[Tuple[str, str]]:
        """Get the linked chat for a platform/chat pair."""
        return self._thread_links.get((platform, chat_id))

    # ── Message Mirroring ─────────────────────────────────────────────────

    async def mirror_message(
        self,
        source_platform: str,
        source_chat_id: str,
        text: str,
        source_user_id: str = "",
        source_username: Optional[str] = None,
        source_message_id: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        reply_to: Optional[str] = None,
        is_edit: bool = False,
        is_system: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Mirror a message to all configured target platforms.

        Parameters
        ----------
        source_platform:
            Source platform name.
        source_chat_id:
            Source chat identifier.
        text:
            Message text.
        source_user_id:
            User ID on the source platform.
        source_username:
            Username display name.
        source_message_id:
            Message ID on the source platform (for edit linking).
        attachments:
            File paths to mirror.
        reply_to:
            Source message ID being replied to.
        is_edit:
            Whether this is an edit to an existing message.
        is_system:
            Whether this is a system message.

        Returns
        -------
        list[dict]
            List of mirror results with platform, chat_id, success, etc.
        """
        event = MirrorEvent(
            source_platform=source_platform,
            source_chat_id=source_chat_id,
            source_message_id=source_message_id,
            source_user_id=source_user_id,
            source_username=source_username,
            text=text,
            attachments=attachments,
            reply_to=reply_to,
            is_edit=is_edit,
            is_system=is_system,
        )

        results = []
        rules = self.get_rules_for_platform(source_platform)

        for rule in rules:
            if not rule.enabled:
                continue

            # Determine if this direction is allowed
            if rule.source == source_platform:
                if rule.direction == MirrorDirection.ONE_WAY_REVERSE:
                    continue
                target_platform = rule.target
            elif rule.target == source_platform:
                if rule.direction == MirrorDirection.ONE_WAY:
                    continue
                target_platform = rule.source
            else:
                continue

            # Find target chat ID
            target_chat_id = rule.chat_mapping.get(source_chat_id)

            # Check thread links
            if target_chat_id is None:
                linked = self.get_linked_chat(source_platform, source_chat_id)
                if linked and linked[0] == target_platform:
                    target_chat_id = linked[1]

            if target_chat_id is None:
                logger.debug(
                    "No target chat mapping for %s:%s -> %s",
                    source_platform, source_chat_id, target_platform,
                )
                continue

            # Skip edits if not configured
            if is_edit and not rule.include_edits:
                continue

            # Skip system messages
            if is_system:
                continue

            result = await self._mirror_to_platform(event, rule, target_platform, target_chat_id)
            results.append(result)

        return results

    async def _mirror_to_platform(
        self,
        event: MirrorEvent,
        rule: MirrorRule,
        target_platform: str,
        target_chat_id: str,
    ) -> Dict[str, Any]:
        """
        Mirror a single message event to a target platform.
        """
        adapter = self._adapters.get(target_platform)

        # Format the message
        target_format = PLATFORM_FORMAT_PREFERENCES.get(
            target_platform, "plain_text"
        )

        # Add prefix/suffix
        text = event.text
        username = event.source_username or event.source_user_id
        if rule.prefix:
            text = rule.prefix.replace("{user}", username).replace("{platform}", event.source_platform) + text
        if rule.suffix:
            text = text + rule.suffix.replace("{user}", username).replace("{platform}", event.source_platform)

        # Convert format
        formatted = self._converter.convert(text, rule.format_source, target_format)

        # Add mirror attribution if not already present
        if "from" not in text.lower()[:50]:
            attr = f"\n_via {event.source_platform}_"
            formatted = formatted + attr

        try:
            # Handle edits
            if event.is_edit and event.source_message_id:
                link_key = (event.source_platform, event.source_chat_id, event.source_message_id)
                link = self._message_links.get(link_key)
                if link and hasattr(adapter, "edit_message"):
                    await adapter.edit_message(
                        chat_id=link.target_chat_id,
                        message_id=link.target_message_id,
                        text=formatted,
                    )
                    return {
                        "platform": target_platform,
                        "chat_id": target_chat_id,
                        "success": True,
                        "action": "edit",
                    }

            # Handle new messages
            if adapter and hasattr(adapter, "send_message"):
                message_id = await adapter.send_message(target_chat_id, formatted)

                # Track message link
                if event.source_message_id:
                    link = MirrorLink(
                        source_platform=event.source_platform,
                        source_chat_id=event.source_chat_id,
                        source_message_id=event.source_message_id,
                        target_platform=target_platform,
                        target_chat_id=target_chat_id,
                        target_message_id=message_id or "",
                    )
                    self._message_links[
                        (event.source_platform, event.source_chat_id, event.source_message_id)
                    ] = link

                # Mirror attachments
                if rule.include_attachments and event.attachments and hasattr(adapter, "send_file"):
                    for file_path in event.attachments:
                        try:
                            await adapter.send_file(target_chat_id, file_path)
                        except Exception as att_err:
                            logger.warning("Mirror attachment failed: %s", att_err)

                # Update stats
                self._stats[(event.source_platform, target_platform)]["mirrored"] += 1
                self._stats[(event.source_platform, target_platform)]["bytes"] += len(formatted)

                return {
                    "platform": target_platform,
                    "chat_id": target_chat_id,
                    "success": True,
                    "action": "send",
                    "message_id": message_id,
                }

            return {
                "platform": target_platform,
                "chat_id": target_chat_id,
                "success": False,
                "error": "No adapter or send_message method",
            }

        except Exception as e:
            logger.error(
                "Mirror failed: %s:%s -> %s:%s: %s",
                event.source_platform, event.source_chat_id,
                target_platform, target_chat_id, e,
            )
            self._stats[(event.source_platform, target_platform)]["failed"] += 1
            return {
                "platform": target_platform,
                "chat_id": target_chat_id,
                "success": False,
                "error": str(e),
            }

    # ── Statistics ────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return mirroring statistics."""
        stats: Dict[str, Any] = {
            "total_rules": len(self._rules),
            "active_rules": sum(1 for r in self._rules if r.enabled),
            "thread_links": len(self._thread_links) // 2,
            "message_links": len(self._message_links),
            "platform_stats": {},
        }

        for (source, target), data in self._stats.items():
            key = f"{source} -> {target}"
            stats["platform_stats"][key] = data

        return stats
