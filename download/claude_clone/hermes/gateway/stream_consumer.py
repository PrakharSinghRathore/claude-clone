"""
StreamConsumer — Streams agent responses to connected platforms.

Handles chunked streaming for long responses, edit/update support
(Telegram message edit, Discord message edit), typing indicators,
and abort/cancel functionality.

Usage::

    consumer = StreamConsumer(config, adapters, delivery_router)
    stream_id = await consumer.start_stream(
        platform="telegram", chat_id="12345", user_id="alice",
    )
    await consumer.send_chunk(stream_id, "Hello, ")
    await consumer.send_chunk(stream_id, "world!")
    await consumer.finish_stream(stream_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from hermes.gateway.config import GatewayConfig
from hermes.gateway.delivery import DeliveryRouter, FormatConverter, PLATFORM_FORMAT_PREFERENCES

logger = logging.getLogger("hermes.gateway.stream_consumer")


# ──────────────────────────────────────────────────────────────────────────────
# Stream State
# ──────────────────────────────────────────────────────────────────────────────

class StreamState(str, Enum):
    """States of an active response stream."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    EDITING = "editing"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    ERROR = "error"


# ──────────────────────────────────────────────────────────────────────────────
# Active Stream
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ActiveStream:
    """Represents an active response stream to a platform."""

    stream_id: str
    platform: str
    chat_id: str
    user_id: str
    state: StreamState = StreamState.INITIALIZING
    buffer: str = ""
    chunks_sent: int = 0
    last_message_id: Optional[str] = None
    last_edit_time: float = 0.0
    edit_interval: float = 1.0  # Minimum seconds between edits
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Typing indicator state
    typing_active: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Stream Consumer
# ──────────────────────────────────────────────────────────────────────────────

class StreamConsumer:
    """
    Streams agent responses to connected platforms.

    Features:
    - Chunked streaming for long responses
    - Edit/update support (Telegram, Discord, Slack)
    - Typing indicators
    - Abort/cancel streaming
    - Configurable buffer and flush intervals
    """

    # Platforms that support message editing
    EDIT_SUPPORTED_PLATFORMS: Set[str] = {"telegram", "discord", "slack", "matrix", "mattermost"}

    # Platforms that support typing indicators
    TYPING_SUPPORTED_PLATFORMS: Set[str] = {
        "telegram", "discord", "slack", "matrix", "mattermost", "signal",
    }

    # Default edit interval per platform (seconds)
    DEFAULT_EDIT_INTERVALS: Dict[str, float] = {
        "telegram": 1.0,
        "discord": 1.5,
        "slack": 2.0,
        "matrix": 2.0,
        "mattermost": 1.5,
    }

    def __init__(
        self,
        config: GatewayConfig,
        adapters: Optional[Dict[str, Any]] = None,
        delivery_router: Optional[DeliveryRouter] = None,
    ):
        self._config = config
        self._adapters = adapters or {}
        self._delivery_router = delivery_router

        # Active streams
        self._streams: Dict[str, ActiveStream] = {}

        # Converter for format transformations
        self._converter = FormatConverter()

        # Background tasks for auto-flush
        self._flush_tasks: Dict[str, asyncio.Task] = {}

        # Callbacks
        self._on_stream_start: Optional[Callable] = None
        self._on_stream_end: Optional[Callable] = None
        self._on_stream_error: Optional[Callable] = None

    def register_adapter(self, name: str, adapter: Any) -> None:
        """Register a platform adapter."""
        self._adapters[name] = adapter

    def set_callbacks(
        self,
        on_start: Optional[Callable] = None,
        on_end: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        """Set stream lifecycle callbacks."""
        self._on_stream_start = on_start
        self._on_stream_end = on_end
        self._on_stream_error = on_error

    # ── Stream Lifecycle ──────────────────────────────────────────────────

    async def start_stream(
        self,
        platform: str,
        chat_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a new response stream.

        Parameters
        ----------
        platform:
            Target platform name.
        chat_id:
            Target chat identifier.
        user_id:
            Target user identifier.
        metadata:
            Optional stream metadata.

        Returns
        -------
        str
            The stream ID for subsequent chunk operations.
        """
        stream_id = uuid.uuid4().hex[:16]
        stream = ActiveStream(
            stream_id=stream_id,
            platform=platform,
            chat_id=chat_id,
            user_id=user_id,
            metadata=metadata or {},
            edit_interval=self.DEFAULT_EDIT_INTERVALS.get(platform, 1.0),
        )
        self._streams[stream_id] = stream

        # Start typing indicator
        if platform in self.TYPING_SUPPORTED_PLATFORMS:
            await self._start_typing(stream)

        # Send initial placeholder message (for edit-supporting platforms)
        if self._config.streaming_enabled and platform in self.EDIT_SUPPORTED_PLATFORMS:
            try:
                adapter = self._adapters.get(platform)
                if adapter and hasattr(adapter, "send_message"):
                    placeholder = "▍"
                    msg_id = await adapter.send_message(chat_id, placeholder)
                    stream.last_message_id = msg_id
                    stream.state = StreamState.EDITING
            except Exception as e:
                logger.warning(
                    "Failed to send placeholder for stream %s: %s",
                    stream_id, e,
                )
                stream.state = StreamState.ACTIVE
        else:
            stream.state = StreamState.ACTIVE

        # Start auto-flush task
        self._flush_tasks[stream_id] = asyncio.create_task(
            self._auto_flush_loop(stream_id)
        )

        if self._on_stream_start:
            try:
                if asyncio.iscoroutinefunction(self._on_stream_start):
                    await self._on_stream_start(stream)
                else:
                    self._on_stream_start(stream)
            except Exception:
                pass

        return stream_id

    async def send_chunk(self, stream_id: str, chunk: str) -> None:
        """
        Append a text chunk to an active stream.

        The chunk is buffered and periodically flushed to the platform.
        """
        stream = self._streams.get(stream_id)
        if stream is None:
            logger.warning("send_chunk: stream %s not found", stream_id)
            return

        if stream.state in (StreamState.FINISHED, StreamState.CANCELLED, StreamState.ERROR):
            return

        stream.buffer += chunk
        stream.chunks_sent += 1

        # If streaming with edit is not supported, just buffer
        if stream.platform not in self.EDIT_SUPPORTED_PLATFORMS:
            return

        # Check if we should flush (based on chunk size)
        chunk_size = self._config.streaming_chunk_size
        if len(stream.buffer) >= chunk_size:
            await self._flush_stream(stream_id)

    async def finish_stream(self, stream_id: str) -> None:
        """
        Finish an active stream and send any remaining buffered content.
        """
        stream = self._streams.get(stream_id)
        if stream is None:
            return

        # Flush remaining buffer
        await self._flush_stream(stream_id)

        # Final edit to clean up
        if stream.last_message_id and stream.platform in self.EDIT_SUPPORTED_PLATFORMS:
            adapter = self._adapters.get(stream.platform)
            if adapter and hasattr(adapter, "edit_message"):
                try:
                    await adapter.edit_message(
                        chat_id=stream.chat_id,
                        message_id=stream.last_message_id,
                        text=stream.buffer,
                    )
                except Exception as e:
                    logger.warning("Final edit failed for stream %s: %s", stream_id, e)

        # Stop typing indicator
        await self._stop_typing(stream)

        stream.state = StreamState.FINISHED
        stream.finished_at = datetime.now(timezone.utc).isoformat()

        # Cancel flush task
        task = self._flush_tasks.pop(stream_id, None)
        if task and not task.done():
            task.cancel()

        if self._on_stream_end:
            try:
                if asyncio.iscoroutinefunction(self._on_stream_end):
                    await self._on_stream_end(stream)
                else:
                    self._on_stream_end(stream)
            except Exception:
                pass

    async def cancel_stream(self, stream_id: str) -> None:
        """Cancel an active stream."""
        stream = self._streams.get(stream_id)
        if stream is None:
            return

        await self._stop_typing(stream)
        stream.state = StreamState.CANCELLED
        stream.finished_at = datetime.now(timezone.utc).isoformat()

        # Edit message to show cancellation
        if stream.last_message_id and stream.platform in self.EDIT_SUPPORTED_PLATFORMS:
            adapter = self._adapters.get(stream.platform)
            if adapter and hasattr(adapter, "edit_message"):
                try:
                    cancelled_text = stream.buffer + "\n\n_✗ Response cancelled_"
                    await adapter.edit_message(
                        chat_id=stream.chat_id,
                        message_id=stream.last_message_id,
                        text=cancelled_text,
                    )
                except Exception:
                    pass

        # Cancel flush task
        task = self._flush_tasks.pop(stream_id, None)
        if task and not task.done():
            task.cancel()

    def get_stream(self, stream_id: str) -> Optional[ActiveStream]:
        """Get an active stream by ID."""
        return self._streams.get(stream_id)

    def get_active_streams(self) -> List[ActiveStream]:
        """Return all non-finished streams."""
        return [
            s for s in self._streams.values()
            if s.state in (StreamState.INITIALIZING, StreamState.ACTIVE, StreamState.EDITING)
        ]

    # ── Internal Flush ────────────────────────────────────────────────────

    async def _flush_stream(self, stream_id: str) -> None:
        """Flush buffered content to the platform."""
        stream = self._streams.get(stream_id)
        if stream is None or not stream.buffer:
            return

        adapter = self._adapters.get(stream.platform)
        if adapter is None:
            return

        # Format the buffered content
        target_format = PLATFORM_FORMAT_PREFERENCES.get(
            stream.platform, "plain_text"
        )
        formatted = self._converter.convert(
            stream.buffer, self._config.message_format_default, target_format
        )

        if stream.last_message_id and stream.platform in self.EDIT_SUPPORTED_PLATFORMS:
            # Edit existing message
            if hasattr(adapter, "edit_message"):
                now = time.time()
                if now - stream.last_edit_time >= stream.edit_interval:
                    try:
                        await adapter.edit_message(
                            chat_id=stream.chat_id,
                            message_id=stream.last_message_id,
                            text=formatted,
                        )
                        stream.last_edit_time = now
                    except Exception as e:
                        logger.debug("Edit failed for stream %s: %s", stream_id, e)
        else:
            # Send new message (for non-edit platforms, send when done)
            pass

    async def _auto_flush_loop(self, stream_id: str) -> None:
        """Background loop that periodically flushes active streams."""
        try:
            while True:
                await asyncio.sleep(0.5)
                stream = self._streams.get(stream_id)
                if stream is None:
                    return
                if stream.state in (StreamState.FINISHED, StreamState.CANCELLED, StreamState.ERROR):
                    return
                await self._flush_stream(stream_id)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("Auto-flush error for stream %s: %s", stream_id, e)

    # ── Typing Indicators ─────────────────────────────────────────────────

    async def _start_typing(self, stream: ActiveStream) -> None:
        """Start sending typing indicators for a stream."""
        adapter = self._adapters.get(stream.platform)
        if adapter is None:
            return

        stream.typing_active = True

        async def _typing_loop():
            try:
                while stream.typing_active:
                    if hasattr(adapter, "send_typing"):
                        try:
                            await adapter.send_typing(stream.chat_id)
                        except Exception:
                            pass
                    await asyncio.sleep(3.0)  # Typing indicator typically lasts ~5s
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        stream._typing_task = asyncio.create_task(_typing_loop())

    async def _stop_typing(self, stream: ActiveStream) -> None:
        """Stop typing indicators for a stream."""
        stream.typing_active = False
        typing_task = getattr(stream, "_typing_task", None)
        if typing_task and not typing_task.done():
            typing_task.cancel()

    # ── Non-Streaming Delivery ────────────────────────────────────────────

    async def deliver_complete(
        self,
        platform: str,
        chat_id: str,
        user_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Deliver a complete message without streaming.

        Used as a fallback for platforms that don't support editing.
        """
        if self._delivery_router:
            await self._delivery_router.deliver(
                user_id=user_id,
                platform=platform,
                chat_id=chat_id,
                text=text,
                metadata=metadata,
            )
            return

        adapter = self._adapters.get(platform)
        if adapter and hasattr(adapter, "send_message"):
            target_format = PLATFORM_FORMAT_PREFERENCES.get(
                platform, "plain_text"
            )
            formatted = self._converter.convert(
                text, self._config.message_format_default, target_format
            )
            await adapter.send_message(chat_id, formatted)

    # ── Cleanup ───────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Cancel all active streams and clean up."""
        for stream_id in list(self._streams.keys()):
            await self.cancel_stream(stream_id)

        for task in self._flush_tasks.values():
            if not task.done():
                task.cancel()

        self._streams.clear()
        self._flush_tasks.clear()
