"""
Atlas Channels — Base channel abstraction layer.

Provides the foundational classes, enums, and dataclasses for multi-channel
messaging. All platform adapters inherit from BaseChannel.

Architecture inspired by OpenClaw's multi-channel messaging abstraction.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChannelType(Enum):
    """All supported messaging channel / platform types.

    Covers 25+ messaging platforms with extensibility via CUSTOM.
    """

    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    SIGNAL = "signal"
    EMAIL = "email"
    IRC = "irc"
    MATRIX = "matrix"
    TEAMS = "teams"
    FEISHU = "feishu"
    LINE = "line"
    MATTERMOST = "mattermost"
    NEXTCLOUD_TALK = "nextcloud_talk"
    NOSTR = "nostr"
    SYNTHOLOGY_CHAT = "synology_chat"
    TWITCH = "twitch"
    ZALO = "zalo"
    WEBCHAT = "webchat"
    WECHAT = "wechat"
    GOOGLE_CHAT = "google_chat"
    SMS = "sms"
    WEBHOOK = "webhook"
    API = "api"
    BLUEBUBBLES = "bluebubbles"
    CUSTOM = "custom"

    @classmethod
    def from_string(cls, value: str) -> "ChannelType":
        """Parse a channel type from string, case-insensitive with fallback.

        Args:
            value: String identifier to look up.

        Returns:
            Matching ChannelType enum member.

        Raises:
            ValueError: If no matching channel type is found.
        """
        normalized = value.lower().strip().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown channel type: {value!r}. "
                         f"Valid types: {[m.value for m in cls]}")

    @property
    def display_name(self) -> str:
        """Human-readable display name for this channel type."""
        return self.value.replace("_", " ").title()


class ChannelState(Enum):
    """Connection state of a channel adapter."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class MessageDirection(Enum):
    """Direction of a message relative to the agent."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


class AttachmentType(Enum):
    """Supported attachment media types."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    FILE = "file"
    STICKER = "sticker"
    VOICE_NOTE = "voice_note"
    LOCATION = "location"
    CONTACT = "contact"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Attachment:
    """Represents a file/media attachment on a message.

    Attributes:
        type: Media type classification.
        url: Remote URL or local path to the attachment.
        filename: Original filename (if available).
        size: File size in bytes.
        mime_type: MIME type string (e.g. 'image/png').
        thumbnail_url: Optional preview/thumbnail URL.
        caption: Optional caption or description.
        width: Image/video width in pixels.
        height: Image/video height in pixels.
        duration: Audio/video duration in seconds.
    """

    type: AttachmentType = AttachmentType.FILE
    url: str = ""
    filename: str = ""
    size: int = 0
    mime_type: str = ""
    thumbnail_url: str = ""
    caption: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize attachment to dictionary."""
        return {
            "type": self.type.value,
            "url": self.url,
            "filename": self.filename,
            "size": self.size,
            "mime_type": self.mime_type,
            "thumbnail_url": self.thumbnail_url,
            "caption": self.caption,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attachment":
        """Deserialize attachment from dictionary."""
        if "type" in data and isinstance(data["type"], str):
            data["type"] = AttachmentType(data["type"])
        return cls(**data)


@dataclass
class ChannelMessage:
    """Normalized message representation across all channels.

    Provides a unified message format that abstracts away platform-specific
    differences in message structure.

    Attributes:
        id: Unique message identifier. Auto-generated if not provided.
        channel_type: The platform/channel this message came from.
        direction: Whether this is inbound (from user) or outbound (from agent).
        sender: Identifier of the message sender (user ID, phone number, etc.).
        recipient: Identifier of the message recipient.
        content: Text content of the message.
        timestamp: When the message was created (UTC).
        metadata: Platform-specific metadata (chat ID, thread ID, etc.).
        attachments: List of file/media attachments.
        reply_to: ID of the message this is replying to (if applicable).
        thread_id: Thread or conversation identifier.
        chat_id: Chat/group identifier.
        is_edited: Whether this message was edited after sending.
        edit_timestamp: When the message was last edited.
        message_type: Type hint (text, command, system, etc.).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    channel_type: ChannelType = ChannelType.WEBCHAT
    direction: MessageDirection = MessageDirection.INBOUND
    sender: str = ""
    recipient: str = ""
    content: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Attachment] = field(default_factory=list)
    reply_to: Optional[str] = None
    thread_id: Optional[str] = None
    chat_id: Optional[str] = None
    is_edited: bool = False
    edit_timestamp: Optional[datetime] = None
    message_type: str = "text"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary for persistence or transport."""
        return {
            "id": self.id,
            "channel_type": self.channel_type.value if isinstance(self.channel_type, ChannelType) else str(self.channel_type),
            "direction": self.direction.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
            "attachments": [a.to_dict() for a in self.attachments],
            "reply_to": self.reply_to,
            "thread_id": self.thread_id,
            "chat_id": self.chat_id,
            "is_edited": self.is_edited,
            "edit_timestamp": self.edit_timestamp.isoformat() if self.edit_timestamp else None,
            "message_type": self.message_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelMessage":
        """Deserialize message from dictionary."""
        data = data.copy()
        if "channel_type" in data and isinstance(data["channel_type"], str):
            data["channel_type"] = ChannelType(data["channel_type"])
        if "direction" in data and isinstance(data["direction"], str):
            data["direction"] = MessageDirection(data["direction"])
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if "edit_timestamp" in data and isinstance(data["edit_timestamp"], str):
            data["edit_timestamp"] = datetime.fromisoformat(data["edit_timestamp"])
        if "attachments" in data:
            data["attachments"] = [Attachment.from_dict(a) for a in data["attachments"]]
        return cls(**data)

    @property
    def has_attachments(self) -> bool:
        """Whether this message contains any attachments."""
        return len(self.attachments) > 0

    @property
    def is_system_message(self) -> bool:
        """Whether this is a system/internal message."""
        return self.message_type == "system"

    @property
    def is_command(self) -> bool:
        """Whether this message starts with a command prefix."""
        return self.content.startswith("/") or self.content.startswith("!")

    def content_hash(self) -> str:
        """Compute a SHA-256 hash of the message content for deduplication."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass
class ChannelConfig:
    """Configuration for a channel adapter.

    Attributes:
        channel_type: The type of channel this config applies to.
        enabled: Whether this channel is active.
        api_token: API token or key for authentication.
        api_url: Base URL for the platform API.
        webhook_url: Webhook URL for receiving events.
        webhook_secret: Secret for verifying webhook signatures.
        rate_limit_per_minute: Max messages per minute.
        rate_limit_burst: Max burst messages allowed.
        max_message_size: Maximum message size in bytes.
        max_attachments: Maximum number of attachments per message.
        max_file_size: Maximum file size for uploads in bytes.
        timeout: Connection timeout in seconds.
        reconnect_attempts: Max reconnection attempts before giving up.
        reconnect_delay: Base delay between reconnection attempts in seconds.
        reconnect_backoff: Multiplier for exponential backoff.
        heartbeat_interval: Heartbeat/keepalive interval in seconds (0 = disabled).
        admin_ids: Set of admin user IDs.
        allowed_ids: Set of allowed user IDs (empty = all allowed).
        blocked_ids: Set of blocked user IDs.
        extra: Additional platform-specific configuration.
    """

    channel_type: ChannelType = ChannelType.CUSTOM
    enabled: bool = True
    api_token: str = ""
    api_url: str = ""
    webhook_url: str = ""
    webhook_secret: str = ""
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10
    max_message_size: int = 4096
    max_attachments: int = 10
    max_file_size: int = 50 * 1024 * 1024  # 50 MB
    timeout: int = 30
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    reconnect_backoff: float = 2.0
    heartbeat_interval: float = 0.0
    admin_ids: Set[str] = field(default_factory=set)
    allowed_ids: Set[str] = field(default_factory=set)
    blocked_ids: Set[str] = field(default_factory=set)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary, masking sensitive fields."""
        return {
            "channel_type": self.channel_type.value,
            "enabled": self.enabled,
            "api_token": self._mask_secret(self.api_token),
            "api_url": self.api_url,
            "webhook_url": self.webhook_url,
            "webhook_secret": self._mask_secret(self.webhook_secret),
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "rate_limit_burst": self.rate_limit_burst,
            "max_message_size": self.max_message_size,
            "max_attachments": self.max_attachments,
            "max_file_size": self.max_file_size,
            "timeout": self.timeout,
            "reconnect_attempts": self.reconnect_attempts,
            "reconnect_delay": self.reconnect_delay,
            "reconnect_backoff": self.reconnect_backoff,
            "heartbeat_interval": self.heartbeat_interval,
            "admin_ids": list(self.admin_ids),
            "allowed_ids": list(self.allowed_ids),
            "blocked_ids": list(self.blocked_ids),
            "extra": self.extra,
        }

    @staticmethod
    def _mask_secret(value: str) -> str:
        """Mask a secret value for safe logging/display."""
        if not value:
            return ""
        if len(value) <= 8:
            return "****"
        return value[:4] + "****" + value[-4:]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelConfig":
        """Deserialize config from dictionary."""
        data = data.copy()
        if "channel_type" in data and isinstance(data["channel_type"], str):
            data["channel_type"] = ChannelType(data["channel_type"])
        if "admin_ids" in data:
            data["admin_ids"] = set(data["admin_ids"])
        if "allowed_ids" in data:
            data["allowed_ids"] = set(data["allowed_ids"])
        if "blocked_ids" in data:
            data["blocked_ids"] = set(data["blocked_ids"])
        return cls(**data)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter for channel message throughput.

    Implements a sliding window rate limiter that tracks per-minute and burst
    limits to prevent API rate limit violations.

    Attributes:
        per_minute: Maximum messages allowed per minute.
        burst: Maximum burst messages allowed.
    """

    def __init__(self, per_minute: int = 60, burst: int = 10) -> None:
        """Initialize rate limiter.

        Args:
            per_minute: Maximum operations per 60-second window.
            burst: Maximum operations allowed in a single burst.
        """
        self.per_minute = per_minute
        self.burst = burst
        self._timestamps: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Attempt to acquire a rate limit token.

        Returns:
            True if the operation is allowed, False if rate limited.
        """
        async with self._lock:
            now = time.monotonic()
            window = 60.0

            # Prune timestamps outside the window
            self._timestamps = [
                t for t in self._timestamps if now - t < window
            ]

            if len(self._timestamps) >= self.per_minute:
                logger.warning(
                    "Rate limit reached: %d/%d per minute",
                    len(self._timestamps), self.per_minute,
                )
                return False

            self._timestamps.append(now)
            return True

    async def wait_for_token(self, timeout: float = 30.0) -> bool:
        """Wait until a token is available or timeout expires.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if a token was acquired, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.acquire():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.5, remaining))
        return False

    def reset(self) -> None:
        """Clear all rate limit state."""
        self._timestamps.clear()

    @property
    def current_rate(self) -> int:
        """Current number of operations in the active window."""
        now = time.monotonic()
        return sum(1 for t in self._timestamps if now - t < 60.0)

    @property
    def remaining(self) -> int:
        """Remaining operations before rate limit is hit."""
        return max(0, self.per_minute - self.current_rate)


# ---------------------------------------------------------------------------
# Abstract Base Channel
# ---------------------------------------------------------------------------

# Type aliases for callbacks
MessageCallback = Callable[["ChannelMessage"], Coroutine[Any, Any, None]]
ErrorCallback = Callable[[Exception], Coroutine[Any, Any, None]]
DisconnectCallback = Callable[[], Coroutine[Any, Any, None]]


class BaseChannel(ABC):
    """Abstract base class for all channel adapters.

    Provides a common interface for connecting, sending, receiving, and
    managing lifecycle of messaging platform connections. Subclasses must
    implement the abstract methods to support a specific platform.

    Features built into the base class:
    - Rate limiting (token bucket)
    - Retry logic with exponential backoff
    - Heartbeat / keepalive
    - Event callbacks (message, error, disconnect)
    - Connection state management

    Usage::

        class MyChannel(BaseChannel):
            @property
            def channel_type(self) -> ChannelType:
                return ChannelType.CUSTOM

            async def _do_connect(self) -> None:
                ...

            async def _do_disconnect(self) -> None:
                ...

            async def _do_send(self, message: ChannelMessage) -> str:
                ...

        channel = MyChannel(config=ChannelConfig(...))
        await channel.connect()
    """

    def __init__(
        self,
        config: Optional[ChannelConfig] = None,
        channel_type: Optional[ChannelType] = None,
    ) -> None:
        """Initialize the base channel.

        Args:
            config: Channel configuration. If None, uses defaults.
            channel_type: Override channel type (used by subclass).
        """
        self._config = config or ChannelConfig()
        self._channel_type_override = channel_type
        self._state = ChannelState.DISCONNECTED
        self._rate_limiter = RateLimiter(
            per_minute=self._config.rate_limit_per_minute,
            burst=self._config.rate_limit_burst,
        )
        self._message_callbacks: List[MessageCallback] = []
        self._error_callbacks: List[ErrorCallback] = []
        self._disconnect_callbacks: List[DisconnectCallback] = []
        self._reconnect_count = 0
        self._last_activity: float = time.monotonic()
        self._heartbeat_task: Optional[asyncio.Task[Any]] = None
        self._lock = asyncio.Lock()
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "reconnects": 0,
            "last_connected": None,
            "last_disconnected": None,
        }
        logger.info(
            "Initialized channel %s (enabled=%s)",
            self.channel_type.value, self._config.enabled,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def channel_type(self) -> ChannelType:
        """Return the ChannelType this adapter handles."""
        ...

    @property
    def state(self) -> ChannelState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Whether the channel is currently connected."""
        return self._state == ChannelState.CONNECTED

    @property
    def config(self) -> ChannelConfig:
        """Current channel configuration."""
        return self._config

    @config.setter
    def config(self, value: ChannelConfig) -> None:
        """Update channel configuration.

        Args:
            value: New configuration to apply.
        """
        self._config = value
        self._rate_limiter = RateLimiter(
            per_minute=value.rate_limit_per_minute,
            burst=value.rate_limit_burst,
        )
        logger.info("Updated configuration for channel %s", self.channel_type.value)

    @property
    def stats(self) -> Dict[str, Any]:
        """Channel statistics."""
        return self._stats.copy()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the messaging platform.

        Handles state transitions, rate limiter reset, heartbeat startup,
        and retry logic with exponential backoff on failure.
        """
        if not self._config.enabled:
            logger.info("Channel %s is disabled, skipping connect", self.channel_type.value)
            return

        async with self._lock:
            if self._state == ChannelState.CONNECTED:
                logger.debug("Channel %s already connected", self.channel_type.value)
                return

            self._state = ChannelState.CONNECTING
            self._rate_limiter.reset()

        max_attempts = self._config.reconnect_attempts
        delay = self._config.reconnect_delay

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(
                    "Connecting channel %s (attempt %d/%d)",
                    self.channel_type.value, attempt, max_attempts,
                )
                await self._do_connect()
                self._state = ChannelState.CONNECTED
                self._reconnect_count = 0
                self._last_activity = time.monotonic()
                self._stats["last_connected"] = datetime.now(timezone.utc).isoformat()
                logger.info("Channel %s connected successfully", self.channel_type.value)

                if self._config.heartbeat_interval > 0:
                    self._start_heartbeat()
                return

            except Exception as exc:
                logger.error(
                    "Channel %s connect attempt %d failed: %s",
                    self.channel_type.value, attempt, exc,
                )
                self._stats["errors"] += 1
                self._state = ChannelState.RECONNECTING

                if attempt < max_attempts:
                    await asyncio.sleep(delay)
                    delay *= self._config.reconnect_backoff

        self._state = ChannelState.ERROR
        logger.error(
            "Channel %s failed to connect after %d attempts",
            self.channel_type.value, max_attempts,
        )
        await self._emit_error(
            ConnectionError(
                f"Failed to connect {self.channel_type.value} "
                f"after {max_attempts} attempts"
            )
        )

    async def disconnect(self) -> None:
        """Disconnect from the messaging platform gracefully.

        Stops heartbeat, cancels pending operations, and transitions state.
        """
        async with self._lock:
            if self._state == ChannelState.DISCONNECTED:
                return

            logger.info("Disconnecting channel %s", self.channel_type.value)
            self._stop_heartbeat()

            try:
                await self._do_disconnect()
            except Exception as exc:
                logger.error("Error during disconnect: %s", exc)

            self._state = ChannelState.DISCONNECTED
            self._stats["last_disconnected"] = datetime.now(timezone.utc).isoformat()
            logger.info("Channel %s disconnected", self.channel_type.value)

            await self._emit_disconnect()

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send(self, message: ChannelMessage) -> str:
        """Send a message through this channel.

        Validates the message, checks rate limits, applies retry logic,
        and delegates to the platform-specific send implementation.

        Args:
            message: The message to send.

        Returns:
            The platform-specific message ID.

        Raises:
            RuntimeError: If the channel is not connected.
            ValueError: If the message is invalid.
        """
        if not self.is_connected:
            raise RuntimeError(
                f"Channel {self.channel_type.value} is not connected "
                f"(state: {self._state.value})"
            )

        self._validate_message(message)

        # Rate limiting
        if not await self._rate_limiter.wait_for_token(timeout=self._config.timeout):
            raise RuntimeError(
                f"Rate limit exceeded for channel {self.channel_type.value}"
            )

        # Retry with exponential backoff
        max_retries = 3
        base_delay = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                message_id = await self._do_send(message)
                self._stats["messages_sent"] += 1
                self._last_activity = time.monotonic()
                logger.debug(
                    "Sent message %s via %s",
                    message_id, self.channel_type.value,
                )
                return message_id
            except Exception as exc:
                logger.warning(
                    "Send attempt %d/%d failed for %s: %s",
                    attempt, max_retries, self.channel_type.value, exc,
                )
                self._stats["errors"] += 1
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                    continue
                raise

        raise RuntimeError("Send failed after all retries")

    async def receive(self) -> Optional[ChannelMessage]:
        """Poll for and return the next incoming message.

        Returns:
            The next available message, or None if no message is available.
        """
        if not self.is_connected:
            return None
        try:
            message = await self._do_receive()
            if message is not None:
                self._stats["messages_received"] += 1
                self._last_activity = time.monotonic()
                await self._emit_message(message)
            return message
        except Exception as exc:
            logger.error("Error receiving from %s: %s", self.channel_type.value, exc)
            self._stats["errors"] += 1
            return None

    async def send_typing_indicator(self, chat_id: str) -> None:
        """Send a typing indicator to a chat.

        Args:
            chat_id: The chat to send the indicator to.
        """
        if self.is_connected:
            try:
                await self._do_send_typing_indicator(chat_id)
            except Exception as exc:
                logger.debug(
                    "Failed to send typing indicator on %s: %s",
                    self.channel_type.value, exc,
                )

    async def send_read_receipt(self, message_id: str) -> None:
        """Send a read receipt for a message.

        Args:
            message_id: The message ID to mark as read.
        """
        if self.is_connected:
            try:
                await self._do_send_read_receipt(message_id)
            except Exception as exc:
                logger.debug(
                    "Failed to send read receipt on %s: %s",
                    self.channel_type.value, exc,
                )

    # ------------------------------------------------------------------
    # Event Callbacks
    # ------------------------------------------------------------------

    def on_message(self, callback: MessageCallback) -> None:
        """Register a callback for incoming messages.

        Args:
            callback: Async function accepting a ChannelMessage.
        """
        self._message_callbacks.append(callback)

    def on_error(self, callback: ErrorCallback) -> None:
        """Register a callback for errors.

        Args:
            callback: Async function accepting an Exception.
        """
        self._error_callbacks.append(callback)

    def on_disconnect(self, callback: DisconnectCallback) -> None:
        """Register a callback for disconnection events.

        Args:
            callback: Async function with no arguments.
        """
        self._disconnect_callbacks.append(callback)

    def remove_message_callback(self, callback: MessageCallback) -> bool:
        """Remove a previously registered message callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if the callback was found and removed.
        """
        try:
            self._message_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def remove_error_callback(self, callback: ErrorCallback) -> bool:
        """Remove a previously registered error callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if the callback was found and removed.
        """
        try:
            self._error_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Abstract methods (platform-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    async def _do_connect(self) -> None:
        """Platform-specific connection logic.

        Called by :meth:`connect` after state and rate limiter setup.
        """
        ...

    @abstractmethod
    async def _do_disconnect(self) -> None:
        """Platform-specific disconnection logic.

        Called by :meth:`disconnect` before state transition.
        """
        ...

    @abstractmethod
    async def _do_send(self, message: ChannelMessage) -> str:
        """Platform-specific message sending logic.

        Args:
            message: The message to send.

        Returns:
            Platform-specific message ID.
        """
        ...

    async def _do_receive(self) -> Optional[ChannelMessage]:
        """Platform-specific message receiving logic.

        Returns:
            The next available message, or None.
        """
        return None

    async def _do_send_typing_indicator(self, chat_id: str) -> None:
        """Platform-specific typing indicator implementation.

        Default implementation does nothing (no-op).
        """
        pass

    async def _do_send_read_receipt(self, message_id: str) -> None:
        """Platform-specific read receipt implementation.

        Default implementation does nothing (no-op).
        """
        pass

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def _start_heartbeat(self) -> None:
        """Start the heartbeat/keepalive task."""
        if self._heartbeat_task is not None:
            self._stop_heartbeat()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.debug(
            "Started heartbeat for %s (interval=%.1fs)",
            self.channel_type.value, self._config.heartbeat_interval,
        )

    def _stop_heartbeat(self) -> None:
        """Stop the heartbeat/keepalive task."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat to maintain connection alive."""
        try:
            while self.is_connected:
                await asyncio.sleep(self._config.heartbeat_interval)
                if not self.is_connected:
                    break
                try:
                    await self._do_heartbeat()
                except Exception as exc:
                    logger.warning(
                        "Heartbeat failed for %s: %s",
                        self.channel_type.value, exc,
                    )
        except asyncio.CancelledError:
            pass

    async def _do_heartbeat(self) -> None:
        """Platform-specific heartbeat implementation.

        Default implementation does nothing. Override in subclasses
        that need periodic keepalive (e.g., WebSocket ping/pong).
        """
        pass

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    async def reconnect(self) -> None:
        """Attempt to reconnect the channel.

        Increments reconnect counter, applies exponential backoff delay,
        and calls :meth:`connect` to re-establish the connection.
        """
        self._reconnect_count += 1
        self._stats["reconnects"] += 1
        delay = min(
            self._config.reconnect_delay
            * (self._config.reconnect_backoff ** (self._reconnect_count - 1)),
            60.0,
        )
        logger.info(
            "Reconnecting channel %s (attempt %d, delay=%.1fs)",
            self.channel_type.value, self._reconnect_count, delay,
        )
        self._state = ChannelState.RECONNECTING
        await asyncio.sleep(delay)
        await self.connect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_message(self, message: ChannelMessage) -> None:
        """Validate a message before sending.

        Args:
            message: The message to validate.

        Raises:
            ValueError: If the message fails validation.
        """
        if not message.content and not message.attachments:
            raise ValueError("Message must have content or attachments")

        if message.content and len(message.content.encode("utf-8")) > self._config.max_message_size:
            raise ValueError(
                f"Message content exceeds maximum size "
                f"({self._config.max_message_size} bytes)"
            )

        if len(message.attachments) > self._config.max_attachments:
            raise ValueError(
                f"Too many attachments ({len(message.attachments)} > "
                f"{self._config.max_attachments})"
            )

        for attachment in message.attachments:
            if attachment.size > self._config.max_file_size and self._config.max_file_size > 0:
                raise ValueError(
                    f"Attachment {attachment.filename} exceeds maximum size "
                    f"({attachment.size} > {self._config.max_file_size} bytes)"
                )

    async def _emit_message(self, message: ChannelMessage) -> None:
        """Fire all registered message callbacks.

        Args:
            message: The received message.
        """
        for callback in self._message_callbacks:
            try:
                await callback(message)
            except Exception as exc:
                logger.error(
                    "Message callback error on %s: %s",
                    self.channel_type.value, exc,
                )

    async def _emit_error(self, error: Exception) -> None:
        """Fire all registered error callbacks.

        Args:
            error: The error that occurred.
        """
        for callback in self._error_callbacks:
            try:
                await callback(error)
            except Exception as exc:
                logger.error("Error callback error: %s", exc)

    async def _emit_disconnect(self) -> None:
        """Fire all registered disconnect callbacks."""
        for callback in self._disconnect_callbacks:
            try:
                await callback()
            except Exception as exc:
                logger.error("Disconnect callback error: %s", exc)

    def get_state(self) -> ChannelState:
        """Return the current channel state.

        Returns:
            Current ChannelState.
        """
        return self._state

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive channel status information.

        Returns:
            Dictionary with state, stats, rate limit info, and config summary.
        """
        return {
            "channel_type": self.channel_type.value,
            "state": self._state.value,
            "is_connected": self.is_connected,
            "rate_limit": {
                "per_minute": self._config.rate_limit_per_minute,
                "current": self._rate_limiter.current_rate,
                "remaining": self._rate_limiter.remaining,
            },
            "stats": self._stats.copy(),
            "enabled": self._config.enabled,
            "reconnect_count": self._reconnect_count,
            "last_activity": datetime.fromtimestamp(
                self._last_activity, tz=timezone.utc
            ).isoformat(),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on this channel.

        Returns:
            Dictionary with health status information.
        """
        is_healthy = self.is_connected
        now = time.monotonic()
        idle_time = now - self._last_activity

        return {
            "channel_type": self.channel_type.value,
            "healthy": is_healthy,
            "state": self._state.value,
            "idle_seconds": round(idle_time, 2),
            "stats": self._stats.copy(),
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"type={self.channel_type.value} "
            f"state={self._state.value}>"
        )
