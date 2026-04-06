"""
Atlas Channels — Channel Adapter Manager.

Provides a central registry for managing multiple channel adapters,
handling connect/disconnect lifecycle, message routing, and broadcasting
across all registered channels.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from atlas.channels.base import (
    BaseChannel,
    ChannelConfig,
    ChannelMessage,
    ChannelState,
    ChannelType,
    ErrorCallback,
    MessageCallback,
    DisconnectCallback,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AdapterStats:
    """Statistics for the channel adapter manager.

    Attributes:
        total_channels: Total number of registered channels.
        connected_channels: Number of currently connected channels.
        total_messages_sent: Cumulative messages sent across all channels.
        total_messages_received: Cumulative messages received across all channels.
        total_errors: Cumulative error count across all channels.
        started_at: When the adapter manager was initialized.
    """

    total_channels: int = 0
    connected_channels: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_errors: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ChannelRegistration:
    """Record tracking a registered channel instance.

    Attributes:
        channel: The channel instance.
        registered_at: When the channel was registered.
        priority: Channel priority for broadcast ordering (lower = first).
    """

    channel: BaseChannel
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 0


# ---------------------------------------------------------------------------
# Channel Adapter
# ---------------------------------------------------------------------------

class ChannelAdapter:
    """Central manager for all channel adapters.

    Provides a unified interface for registering, connecting, and communicating
    through multiple messaging channels simultaneously. Handles message routing,
    event delegation, and lifecycle management.

    Usage::

        adapter = ChannelAdapter()

        # Register channels
        adapter.register_channel(telegram_channel)
        adapter.register_channel(discord_channel)

        # Set up message handler
        async def handle_message(msg: ChannelMessage):
            print(f"Received from {msg.channel_type}: {msg.content}")

        adapter.on_message(handle_message)

        # Connect all
        await adapter.connect_all()

        # Send via specific channel
        await adapter.send(ChannelType.TELEGRAM, message)

        # Broadcast to all
        await adapter.broadcast(message)
    """

    def __init__(self) -> None:
        """Initialize the channel adapter manager."""
        self._channels: Dict[ChannelType, ChannelRegistration] = {}
        self._message_callbacks: List[MessageCallback] = []
        self._error_callbacks: List[ErrorCallback] = []
        self._disconnect_callbacks: List[DisconnectCallback] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._receive_tasks: Dict[ChannelType, asyncio.Task[Any]] = {}
        self._stats = AdapterStats()
        self._message_history: List[Dict[str, Any]] = []
        self._max_history: int = 1000

        logger.info("ChannelAdapter initialized")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register_channel(
        self,
        channel: BaseChannel,
        priority: int = 0,
        auto_connect: bool = False,
    ) -> None:
        """Register a channel adapter instance.

        Args:
            channel: The channel instance to register.
            priority: Priority for broadcast ordering (lower = first).
            auto_connect: Whether to connect immediately after registration.

        Raises:
            ValueError: If a channel of the same type is already registered.
        """
        async with self._lock:
            if channel.channel_type in self._channels:
                existing = self._channels[channel.channel_type].channel
                raise ValueError(
                    f"Channel {channel.channel_type.value} is already registered "
                    f"as {existing.__class__.__name__}. Unregister it first."
                )

            registration = ChannelRegistration(
                channel=channel,
                priority=priority,
            )
            self._channels[channel.channel_type] = registration
            self._stats.total_channels = len(self._channels)

            # Wire up internal event handlers
            channel.on_message(self._internal_message_handler)
            channel.on_error(self._internal_error_handler)
            channel.on_disconnect(self._internal_disconnect_handler)

            logger.info(
                "Registered channel %s (priority=%d, auto_connect=%s)",
                channel.channel_type.value, priority, auto_connect,
            )

        if auto_connect:
            try:
                await channel.connect()
            except Exception as exc:
                logger.error(
                    "Auto-connect failed for %s: %s",
                    channel.channel_type.value, exc,
                )

    async def unregister_channel(self, channel_type: ChannelType) -> bool:
        """Unregister and disconnect a channel.

        Args:
            channel_type: The type of channel to remove.

        Returns:
            True if the channel was found and removed.
        """
        async with self._lock:
            registration = self._channels.pop(channel_type, None)
            if registration is None:
                logger.warning("Channel %s not registered", channel_type.value)
                return False

            channel = registration.channel

            # Stop receive loop
            task = self._receive_tasks.pop(channel_type, None)
            if task is not None:
                task.cancel()

            # Disconnect
            try:
                await channel.disconnect()
            except Exception as exc:
                logger.error(
                    "Error disconnecting %s during unregister: %s",
                    channel_type.value, exc,
                )

            self._stats.total_channels = len(self._channels)
            logger.info("Unregistered channel %s", channel_type.value)
            return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect_all(self) -> Dict[ChannelType, bool]:
        """Connect all registered channels.

        Channels are connected concurrently with error isolation — a failure
        on one channel does not prevent others from connecting.

        Returns:
            Dictionary mapping channel types to their connection success status.
        """
        if not self._channels:
            logger.warning("No channels registered, nothing to connect")
            return {}

        logger.info("Connecting %d channels...", len(self._channels))
        self._running = True
        results: Dict[ChannelType, bool] = {}

        # Connect all concurrently
        tasks = {}
        for channel_type, registration in self._channels.items():
            if registration.channel.config.enabled:
                task = asyncio.create_task(
                    self._safe_connect(registration.channel),
                )
                tasks[channel_type] = task

        if tasks:
            channel_types = list(tasks.keys())
            done_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for channel_type, result in zip(channel_types, done_results):
                if isinstance(result, Exception):
                    logger.error(
                        "Failed to connect %s: %s",
                        channel_type.value, result,
                    )
                    results[channel_type] = False
                else:
                    results[channel_type] = result

        self._update_connected_count()
        connected = sum(1 for v in results.values() if v)
        logger.info(
            "Connected %d/%d channels: %s",
            connected, len(results),
            {k.value: v for k, v in results.items()},
        )

        # Start receive loops for connected channels
        for channel_type, success in results.items():
            if success:
                self._start_receive_loop(channel_type)

        return results

    async def disconnect_all(self) -> None:
        """Disconnect all registered channels gracefully.

        Stops all receive loops and disconnects each channel.
        """
        logger.info("Disconnecting all %d channels...", len(self._channels))
        self._running = False

        # Stop all receive tasks
        for channel_type, task in list(self._receive_tasks.items()):
            task.cancel()
            logger.debug("Stopped receive loop for %s", channel_type.value)
        self._receive_tasks.clear()

        # Disconnect all channels concurrently
        tasks = []
        for channel_type, registration in self._channels.items():
            if registration.channel.is_connected:
                tasks.append(
                    self._safe_disconnect(registration.channel)
                )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._update_connected_count()
        logger.info("All channels disconnected")

    async def reconnect_channel(self, channel_type: ChannelType) -> bool:
        """Reconnect a specific channel.

        Args:
            channel_type: The channel type to reconnect.

        Returns:
            True if reconnection was successful.
        """
        registration = self._channels.get(channel_type)
        if registration is None:
            logger.warning("Cannot reconnect: %s not registered", channel_type.value)
            return False

        try:
            await registration.channel.reconnect()
            return registration.channel.is_connected
        except Exception as exc:
            logger.error("Reconnect failed for %s: %s", channel_type.value, exc)
            return False

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send(
        self,
        channel_type: ChannelType,
        message: ChannelMessage,
    ) -> str:
        """Send a message through a specific channel.

        Args:
            channel_type: The channel to send through.
            message: The message to send.

        Returns:
            Platform-specific message ID.

        Raises:
            RuntimeError: If the channel is not registered.
        """
        registration = self._channels.get(channel_type)
        if registration is None:
            raise RuntimeError(
                f"Channel {channel_type.value} is not registered"
            )

        # Ensure message channel type matches
        message.channel_type = channel_type
        message.direction = ChannelMessage.__dataclass_fields__.get(
            "direction", None
        )
        if hasattr(message, "direction"):
            from atlas.channels.base import MessageDirection
            message.direction = MessageDirection.OUTBOUND

        message_id = await registration.channel.send(message)
        self._stats.total_messages_sent += 1

        # Record in history
        self._record_history(message, "sent")

        return message_id

    async def broadcast(
        self,
        message: ChannelMessage,
        exclude: Optional[List[ChannelType]] = None,
    ) -> Dict[ChannelType, str]:
        """Send a message to all connected channels.

        Args:
            message: The message to broadcast.
            exclude: List of channel types to exclude from broadcast.

        Returns:
            Dictionary mapping channel types to their message IDs.
            Failed channels are not included.
        """
        exclude_set = set(exclude or [])
        results: Dict[ChannelType, str] = {}

        # Sort by priority
        sorted_channels = sorted(
            self._channels.items(),
            key=lambda x: x[1].priority,
        )

        for channel_type, registration in sorted_channels:
            if channel_type in exclude_set:
                continue
            if not registration.channel.is_connected:
                logger.debug(
                    "Skipping broadcast to %s (not connected)",
                    channel_type.value,
                )
                continue

            try:
                # Create a copy with the correct channel type
                channel_msg = ChannelMessage(
                    content=message.content,
                    attachments=message.attachments,
                    metadata=message.metadata.copy(),
                    channel_type=channel_type,
                    direction=getattr(
                        message, "direction", None
                    ),
                    sender=message.sender,
                    recipient=message.recipient,
                    reply_to=message.reply_to,
                    thread_id=message.thread_id,
                    chat_id=message.chat_id,
                    message_type=message.message_type,
                )
                from atlas.channels.base import MessageDirection
                channel_msg.direction = MessageDirection.OUTBOUND

                msg_id = await registration.channel.send(channel_msg)
                results[channel_type] = msg_id
                self._stats.total_messages_sent += 1
                self._record_history(channel_msg, "sent")
            except Exception as exc:
                logger.error(
                    "Broadcast to %s failed: %s",
                    channel_type.value, exc,
                )
                self._stats.total_errors += 1

        logger.info(
            "Broadcast to %d/%d channels",
            len(results),
            len(self._channels) - len(exclude_set),
        )
        return results

    async def send_to_multiple(
        self,
        channel_types: List[ChannelType],
        message: ChannelMessage,
    ) -> Dict[ChannelType, str]:
        """Send a message to multiple specific channels.

        Args:
            channel_types: List of channel types to send to.
            message: The message to send.

        Returns:
            Dictionary mapping channel types to message IDs.
        """
        results: Dict[ChannelType, str] = {}
        tasks = {}

        for ct in channel_types:
            tasks[ct] = asyncio.create_task(self._safe_send(ct, message))

        if tasks:
            channel_types_list = list(tasks.keys())
            done = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for ct, result in zip(channel_types_list, done):
                if isinstance(result, Exception):
                    logger.error("Send to %s failed: %s", ct.value, result)
                    self._stats.total_errors += 1
                else:
                    results[ct] = result

        return results

    # ------------------------------------------------------------------
    # Channel Access
    # ------------------------------------------------------------------

    def get_channel(self, channel_type: ChannelType) -> Optional[BaseChannel]:
        """Get a registered channel instance by type.

        Args:
            channel_type: The channel type to look up.

        Returns:
            The channel instance, or None if not registered.
        """
        registration = self._channels.get(channel_type)
        return registration.channel if registration else None

    def list_channels(self) -> List[Dict[str, Any]]:
        """List all registered channels with their status.

        Returns:
            List of dictionaries with channel type, state, and status info.
        """
        result = []
        for channel_type in sorted(self._channels.keys(), key=lambda x: x.value):
            registration = self._channels[channel_type]
            channel = registration.channel
            result.append({
                "channel_type": channel_type.value,
                "display_name": channel_type.display_name,
                "class": channel.__class__.__name__,
                "state": channel.state.value,
                "is_connected": channel.is_connected,
                "enabled": channel.config.enabled,
                "priority": registration.priority,
                "stats": channel.stats,
                "registered_at": registration.registered_at.isoformat(),
            })
        return result

    def list_connected(self) -> List[ChannelType]:
        """List currently connected channel types.

        Returns:
            List of connected ChannelType values.
        """
        return [
            ct for ct, reg in self._channels.items()
            if reg.channel.is_connected
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all channels.

        Returns:
            Dictionary with total stats and per-channel breakdown.
        """
        self._update_connected_count()
        return {
            "total_channels": self._stats.total_channels,
            "connected_channels": self._stats.connected_channels,
            "total_messages_sent": self._stats.total_messages_sent,
            "total_messages_received": self._stats.total_messages_received,
            "total_errors": self._stats.total_errors,
            "running": self._running,
            "started_at": self._stats.started_at.isoformat(),
            "channels": {
                ct.value: channel.stats
                for ct, reg in self._channels.items()
            },
        }

    # ------------------------------------------------------------------
    # Event Callbacks
    # ------------------------------------------------------------------

    def on_message(self, callback: MessageCallback) -> None:
        """Register a global message callback for all channels.

        Args:
            callback: Async function accepting a ChannelMessage.
        """
        self._message_callbacks.append(callback)

    def on_error(self, callback: ErrorCallback) -> None:
        """Register a global error callback for all channels.

        Args:
            callback: Async function accepting an Exception.
        """
        self._error_callbacks.append(callback)

    def on_disconnect(self, callback: DisconnectCallback) -> None:
        """Register a global disconnect callback for all channels.

        Args:
            callback: Async function with no arguments.
        """
        self._disconnect_callbacks.append(callback)

    def remove_message_callback(self, callback: MessageCallback) -> bool:
        """Remove a global message callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if found and removed.
        """
        try:
            self._message_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def remove_error_callback(self, callback: ErrorCallback) -> bool:
        """Remove a global error callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if found and removed.
        """
        try:
            self._error_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Internal Event Handlers
    # ------------------------------------------------------------------

    async def _internal_message_handler(self, message: ChannelMessage) -> None:
        """Handle incoming messages from any channel.

        Fires all registered global callbacks and updates statistics.

        Args:
            message: The incoming message.
        """
        self._stats.total_messages_received += 1
        self._record_history(message, "received")

        for callback in self._message_callbacks:
            try:
                await callback(message)
            except Exception as exc:
                logger.error(
                    "Global message callback error: %s", exc,
                )
                self._stats.total_errors += 1

    async def _internal_error_handler(self, error: Exception) -> None:
        """Handle errors from any channel.

        Args:
            error: The error that occurred.
        """
        self._stats.total_errors += 1
        for callback in self._error_callbacks:
            try:
                await callback(error)
            except Exception as exc:
                logger.error("Global error callback error: %s", exc)

    async def _internal_disconnect_handler(self) -> None:
        """Handle disconnection events from any channel."""
        self._update_connected_count()
        for callback in self._disconnect_callbacks:
            try:
                await callback()
            except Exception as exc:
                logger.error("Global disconnect callback error: %s", exc)

    # ------------------------------------------------------------------
    # Receive Loops
    # ------------------------------------------------------------------

    def _start_receive_loop(self, channel_type: ChannelType) -> None:
        """Start an async receive loop for a channel.

        Args:
            channel_type: The channel type to start receiving from.
        """
        if channel_type in self._receive_tasks:
            return

        async def _loop() -> None:
            channel = self._channels[channel_type].channel
            logger.info("Receive loop started for %s", channel_type.value)
            while self._running and channel.is_connected:
                try:
                    message = await channel.receive()
                    if message is None:
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error(
                        "Receive error on %s: %s",
                        channel_type.value, exc,
                    )
                    self._stats.total_errors += 1
                    await asyncio.sleep(1.0)

        self._receive_tasks[channel_type] = asyncio.create_task(_loop())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _safe_connect(self, channel: BaseChannel) -> bool:
        """Safely connect a channel with error handling.

        Args:
            channel: The channel to connect.

        Returns:
            True if connected successfully.
        """
        try:
            await channel.connect()
            return channel.is_connected
        except Exception as exc:
            logger.error(
                "Connect error for %s: %s",
                channel.channel_type.value, exc,
            )
            return False

    async def _safe_disconnect(self, channel: BaseChannel) -> None:
        """Safely disconnect a channel with error handling.

        Args:
            channel: The channel to disconnect.
        """
        try:
            await channel.disconnect()
        except Exception as exc:
            logger.error(
                "Disconnect error for %s: %s",
                channel.channel_type.value, exc,
            )

    async def _safe_send(
        self,
        channel_type: ChannelType,
        message: ChannelMessage,
    ) -> str:
        """Safely send a message to a channel.

        Args:
            channel_type: Target channel type.
            message: Message to send.

        Returns:
            Platform-specific message ID.

        Raises:
            RuntimeError: If channel is not registered.
        """
        return await self.send(channel_type, message)

    def _update_connected_count(self) -> None:
        """Update the connected channel count in stats."""
        self._stats.connected_channels = sum(
            1 for reg in self._channels.values()
            if reg.channel.is_connected
        )

    def _record_history(
        self,
        message: ChannelMessage,
        action: str,
    ) -> None:
        """Record a message in the history buffer.

        Args:
            message: The message involved.
            action: 'sent' or 'received'.
        """
        entry = {
            "action": action,
            "channel_type": message.channel_type.value,
            "message_id": message.id,
            "sender": message.sender,
            "content_preview": message.content[:200] if message.content else "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._message_history.append(entry)

        # Trim history if needed
        if len(self._message_history) > self._max_history:
            self._message_history = self._message_history[-self._max_history:]

    def get_history(
        self,
        channel_type: Optional[ChannelType] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent message history.

        Args:
            channel_type: Filter by channel type (None for all).
            limit: Maximum number of entries to return.

        Returns:
            List of history entries.
        """
        history = self._message_history
        if channel_type is not None:
            history = [
                entry for entry in history
                if entry["channel_type"] == channel_type.value
            ]
        return history[-limit:]

    async def shutdown(self) -> None:
        """Full shutdown — disconnect all channels and clean up."""
        logger.info("ChannelAdapter shutting down...")
        self._running = False
        await self.disconnect_all()
        logger.info("ChannelAdapter shutdown complete")

    def __repr__(self) -> str:
        connected = self._stats.connected_channels
        total = self._stats.total_channels
        return f"<ChannelAdapter channels={total} connected={connected}>"
