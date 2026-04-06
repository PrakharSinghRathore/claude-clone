"""
Discord Bot adapter for the Atlas Gateway.

Supports slash commands, embed messages, thread support, voice channel
integration, and reaction handling.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.discord import DiscordAdapter

    config = PlatformConfig(name="discord", token="DISCORD_TOKEN", enabled=True)
    adapter = DiscordAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.discord")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

DISCORD_API_BASE = "https://discord.com/api/v10"


class DiscordAdapter:
    """
    Discord Bot adapter using the Discord REST API and WebSocket gateway.

    Parameters
    ----------
    config:
        Platform configuration with ``token``.
    """

    MAX_MESSAGE_LENGTH = 2000

    def __init__(self, config: Any):
        self._config = config
        self._token = config.token or os.environ.get("DISCORD_BOT_TOKEN", "")
        self._timeout = config.timeout or 30
        self._connected = False
        self._session: Optional[Any] = None
        self._ws: Optional[Any] = None
        self._guild_id: Optional[str] = None
        self._heartbeat_interval: float = 41.25
        self._last_sequence: Optional[int] = None
        self._session_id: Optional[str] = None
        self._resume_url: Optional[str] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._ws_task: Optional[asyncio.Task] = None

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to Discord via WebSocket gateway."""
        if not self._token:
            raise ValueError("Discord bot token is required")
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
            headers={"Authorization": f"Bot {self._token}"},
        )

        # Get gateway URL
        gateway_url = await self._get_gateway_url()
        self._ws_url = f"{gateway_url}?v=10&encoding=json"

        # Start WebSocket connection
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._connected = True

        logger.info("Discord adapter connecting...")

    async def disconnect(self) -> None:
        """Disconnect from Discord."""
        self._connected = False
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()
            self._session = None

    async def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected and self._ws is not None and not self._ws.closed

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a message to a Discord channel."""
        text = self._truncate(text)
        url = f"{DISCORD_API_BASE}/channels/{chat_id}/messages"

        payload: Dict[str, Any] = {"content": text}

        if kwargs.get("reply_to"):
            payload["message_reference"] = {
                "message_id": kwargs["reply_to"],
            }

        # Embed support
        embed = kwargs.get("embed")
        if embed:
            payload["embeds"] = [embed] if isinstance(embed, dict) else embed

        result = await self._discord_post(url, payload)
        return str(result.get("id")) if result else None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a file to a Discord channel."""
        url = f"{DISCORD_API_BASE}/channels/{chat_id}/messages"

        try:
            if not self._session or not os.path.exists(file_path):
                return None

            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=os.path.basename(file_path))
                data.add_field("payload_json", json.dumps({
                    "content": kwargs.get("caption", ""),
                }))

                async with self._session.post(url, data=data) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        return str(result.get("id", ""))
                    logger.error("Discord upload error: %s", result)
                    return None
        except Exception as e:
            logger.error("Failed to send file to Discord: %s", e)
            return None

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll the internal message queue for new messages."""
        messages: List[IncomingMessage] = []
        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    # ── Extended Interface ────────────────────────────────────────────────

    async def edit_message(
        self, chat_id: str, message_id: str, text: str, **kwargs: Any,
    ) -> bool:
        """Edit an existing message."""
        url = f"{DISCORD_API_BASE}/channels/{chat_id}/messages/{message_id}"
        payload: Dict[str, Any] = {"content": self._truncate(text)}

        embed = kwargs.get("embed")
        if embed:
            payload["embeds"] = [embed] if isinstance(embed, dict) else embed

        result = await self._discord_patch(url, payload)
        return bool(result)

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message."""
        url = f"{DISCORD_API_BASE}/channels/{chat_id}/messages/{message_id}"
        async with self._session.delete(url) as resp:
            return resp.status == 204

    async def send_typing(self, chat_id: str) -> None:
        """Send a typing indicator."""
        url = f"{DISCORD_API_BASE}/channels/{chat_id}/typing"
        await self._discord_post(url, {})

    async def create_thread(
        self, channel_id: str, name: str, message_id: Optional[str] = None,
    ) -> Optional[str]:
        """Create a thread in a channel."""
        url = f"{DISCORD_API_BASE}/channels/{channel_id}/threads"
        payload: Dict[str, Any] = {
            "name": name,
            "auto_archive_duration": 1440,
            "type": 11,  # PUBLIC_THREAD
        }
        if message_id:
            payload["message_id"] = message_id

        result = await self._discord_post(url, payload)
        return str(result.get("id")) if result else None

    async def add_reaction(self, channel_id: str, message_id: str, emoji: str) -> bool:
        """Add a reaction to a message."""
        url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages/{message_id}/reactions/{emoji}/@me"
        async with self._session.put(url) as resp:
            return resp.status == 204

    def create_embed(
        self,
        title: str,
        description: str = "",
        color: int = 0x5865F2,
        fields: Optional[List[Dict[str, Any]]] = None,
        footer: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a Discord embed object."""
        embed: Dict[str, Any] = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if fields:
            embed["fields"] = fields
        if footer:
            embed["footer"] = {"text": footer}
        if thumbnail_url:
            embed["thumbnail"] = {"url": thumbnail_url}
        return embed

    # ── WebSocket ─────────────────────────────────────────────────────────

    async def _get_gateway_url(self) -> str:
        """Get the WebSocket gateway URL."""
        async with self._session.get(f"{DISCORD_API_BASE}/gateway") as resp:
            data = await resp.json()
            return data.get("url", "wss://gateway.discord.gg")

    async def _ws_loop(self) -> None:
        """Main WebSocket event loop."""
        reconnect_delay = 1.0

        while self._connected:
            try:
                async with self._session.ws_connect(self._ws_url) as ws:
                    self._ws = ws
                    logger.info("Discord WebSocket connected")

                    # Handle HELLO
                    hello = await ws.receive_json()
                    if hello.get("op") == 10:
                        self._heartbeat_interval = hello["d"]["heartbeat_interval"] / 1000.0
                        asyncio.create_task(self._heartbeat_loop(ws))

                    # Send IDENTIFY or RESUME
                    if self._session_id and self._resume_url:
                        await ws.send_json({
                            "op": 6,
                            "d": {
                                "token": self._token,
                                "session_id": self._session_id,
                                "seq": self._last_sequence,
                            }
                        })
                    else:
                        await ws.send_json({
                            "op": 2,
                            "d": {
                                "token": self._token,
                                "intents": 32509,
                                "properties": {
                                    "os": "linux",
                                    "browser": "atlas-gateway",
                                    "device": "atlas-gateway",
                                }
                            }
                        })

                    reconnect_delay = 1.0

                    # Process events
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_ws_event(json.loads(msg.data), ws)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Discord WebSocket error: %s", e)

            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30.0)

    async def _heartbeat_loop(self, ws: Any) -> None:
        """Send periodic heartbeats."""
        try:
            while self._connected:
                await asyncio.sleep(self._heartbeat_interval)
                payload = {"op": 1, "d": self._last_sequence}
                await ws.send_json(payload)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Discord heartbeat error: %s", e)

    async def _handle_ws_event(self, event: Dict[str, Any], ws: Any) -> None:
        """Handle a Discord WebSocket event."""
        op = event.get("op")
        data = event.get("d")
        event_type = event.get("t")

        if op == 1:  # Heartbeat request
            await ws.send_json({"op": 1, "d": self._last_sequence})
        elif op == 9:  # Invalid session
            self._session_id = None
            self._resume_url = None
        elif op == 10:  # HELLO
            self._heartbeat_interval = data["heartbeat_interval"] / 1000.0
        elif op == 11:  # Heartbeat ACK
            pass
        elif op == 0:  # Dispatch
            self._last_sequence = event.get("s")
            if event_type == "READY":
                self._session_id = data.get("session_id")
                self._resume_url = data.get("resume_gateway_url")
                logger.info("Discord bot ready (session_id=%s)", self._session_id[:8])
            elif event_type == "MESSAGE_CREATE":
                msg = self._parse_message(data)
                if msg:
                    await self._message_queue.put(msg)
            elif event_type == "MESSAGE_UPDATE":
                msg = self._parse_message(data, is_edit=True)
                if msg and msg.text:
                    await self._message_queue.put(msg)

    def _parse_message(
        self, data: Dict[str, Any], is_edit: bool = False,
    ) -> Optional[IncomingMessage]:
        """Parse a Discord message event into IncomingMessage."""
        author = data.get("author", {})
        bot = author.get("bot", False)

        # Ignore bot messages
        if bot:
            return None

        text = data.get("content", "")
        if not text and not data.get("attachments"):
            return None

        attachments = [
            att.get("url", "") for att in data.get("attachments", [])
        ] if data.get("attachments") else None

        message_reference = data.get("message_reference")
        reply_to = message_reference.get("message_id") if message_reference else None

        return IncomingMessage(
            platform="discord",
            chat_id=str(data.get("channel_id", "")),
            user_id=str(author.get("id", "")),
            text=text,
            message_id=str(data.get("id", "")),
            username=author.get("username"),
            reply_to=reply_to,
            attachments=attachments,
            is_command=text.startswith("/") if text else False,
            is_edit=is_edit,
            metadata={
                "guild_id": data.get("guild_id"),
                "channel_type": data.get("channel_type"),
                "mentions": [m.get("id") for m in data.get("mentions", [])],
            },
        )

    # ── HTTP Helpers ──────────────────────────────────────────────────────

    async def _discord_post(self, url: str, payload: Dict[str, Any]) -> Optional[Dict]:
        if not self._session:
            return None
        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status in (200, 201):
                    return await resp.json()
                logger.error("Discord POST %s returned %d", url, resp.status)
                return None
        except Exception as e:
            logger.error("Discord POST error %s: %s", url, e)
            return None

    async def _discord_patch(self, url: str, payload: Dict[str, Any]) -> Optional[Dict]:
        if not self._session:
            return None
        try:
            async with self._session.patch(url, json=payload) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error("Discord PATCH error: %s", e)
            return None

    def _truncate(self, text: str) -> str:
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return text
        return text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n[...truncated]"
