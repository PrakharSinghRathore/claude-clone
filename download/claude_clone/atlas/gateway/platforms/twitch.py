"""
Twitch chat adapter for the Atlas Gateway.

Supports Twitch IRC-based chat, whisper support, channel management,
subscriber/broadcaster badges, and mod/vip command handling.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.twitch import TwitchAdapter

    config = PlatformConfig(
        name="twitch",
        token="OAUTH_TOKEN",
        api_key="CLIENT_ID",
        enabled=True,
        extra={
            "nickname": "atlasbot",
            "channels": ["#partner_channel", "#streamer_channel"],
            "initial_channels": ["#first_channel"],
        },
    )
    adapter = TwitchAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.twitch")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

TWITCH_IRC_HOST = "irc.chat.twitch.tv"
TWITCH_IRC_PORT = 6697
TWITCH_API_BASE = "https://api.twitch.tv/helix"
TWITCH_WHISPER_LIMIT = 100  # whispers per day
TWITCH_MSG_LIMIT = 100  # messages per 30 seconds for mods


@dataclass
class TwitchConfig:
    """Configuration for the Twitch adapter."""

    oauth_token: str = ""
    client_id: str = ""
    nickname: str = "atlasbot"
    channels: List[str] = field(default_factory=list)
    initial_channels: List[str] = field(default_factory=list)
    timeout: int = 30
    command_prefix: str = "!"
    rate_limit_delay: float = 1.5  # seconds between messages
    reconnect_delay: float = 5.0


class TwitchAdapter:
    """
    Twitch chat adapter using the IRC interface (tmi.twitch.tv).

    Connects to Twitch's IRC server over TLS, joins configured
    channels, and processes chat messages into IncomingMessage objects.
    Also supports whispers and Twitch API v2 calls for channel info.

    Parameters
    ----------
    config:
        Platform configuration with ``token`` (OAuth token, without
        the ``oauth:`` prefix) and ``api_key`` (client ID).
    """

    MAX_MESSAGE_LENGTH = 500

    def __init__(self, config: Any):
        self._config = config
        extra = getattr(config, "extra", {}) or {}

        self._twitch_config = TwitchConfig(
            oauth_token=config.token or os.environ.get("TWITCH_OAUTH_TOKEN", ""),
            client_id=config.api_key or os.environ.get("TWITCH_CLIENT_ID", ""),
            nickname=extra.get("nickname") or os.environ.get("TWITCH_NICKNAME", "atlasbot"),
            channels=extra.get("channels", []),
            initial_channels=extra.get("initial_channels", []),
            timeout=config.timeout or 30,
            command_prefix=extra.get("command_prefix", "!"),
            rate_limit_delay=extra.get("rate_limit_delay", 1.5),
        )

        self._connected = False
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._session: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._read_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._joined_channels: set = set()
        self._last_message_time: float = 0
        self._message_count: int = 0
        self._message_window_start: float = 0

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to Twitch IRC and join configured channels."""
        oauth = self._twitch_config.oauth_token
        if not oauth:
            raise ValueError(
                "Twitch OAuth token is required. "
                "Set TWITCH_OAUTH_TOKEN env-var or config.token."
            )

        # Ensure oauth: prefix
        if not oauth.startswith("oauth:"):
            oauth = f"oauth:{oauth}"

        try:
            ctx = ssl.create_default_context()
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    TWITCH_IRC_HOST, TWITCH_IRC_PORT, ssl=ctx,
                ),
                timeout=self._twitch_config.timeout,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Twitch IRC connection timed out")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Twitch IRC: {e}")

        # Authenticate
        self._send_raw(f"PASS {oauth}")
        self._send_raw(f"NICK {self._twitch_config.nickname}")
        self._send_raw("CAP REQ :twitch.tv/membership twitch.tv/tags twitch.tv/commands")

        # Create HTTP session for API calls
        if HAS_AIOHTTP:
            headers = {
                "Authorization": f"Bearer {self._twitch_config.oauth_token.replace('oauth:', '')}",
                "Client-Id": self._twitch_config.client_id,
            }
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._twitch_config.timeout),
                headers=headers,
            )

        self._read_task = asyncio.create_task(self._read_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())
        self._connected = True

        # Join initial channels
        channels = (
            self._twitch_config.initial_channels
            or self._twitch_config.channels
        )
        for channel in channels:
            await self.join_channel(channel)

        logger.info(
            "Twitch adapter connecting as %s to %d channels",
            self._twitch_config.nickname, len(channels),
        )

    async def disconnect(self) -> None:
        """Disconnect from Twitch IRC."""
        self._connected = False

        if self._writer:
            self._send_raw("QUIT :Atlas Gateway shutting down")
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None

        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()
            self._session = None

        self._reader = None
        self._joined_channels.clear()

    async def is_connected(self) -> bool:
        """Check if the Twitch IRC connection is active."""
        return self._connected and self._writer is not None and not self._writer.is_closing()

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a message to a Twitch channel.

        Parameters
        ----------
        chat_id:
            Channel name (e.g. ``#partner_channel``).
        text:
            Message text. Will be split if exceeds Twitch limits.
        """
        if not self._writer:
            return None

        await self._enforce_rate_limit()

        if not chat_id.startswith("#"):
            chat_id = f"#{chat_id}"

        text = self._truncate(text)

        for line in text.split("\n"):
            line = line.strip()
            if line:
                self._send_raw(f"PRIVMSG {chat_id} :{line}")
                self._last_message_time = time.time()
                self._message_count += 1
                await asyncio.sleep(self._twitch_config.rate_limit_delay)

        return f"twitch_{int(time.time())}"

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a file reference to Twitch chat.

        Twitch does not support file uploads in chat. This sends
        the filename as a text message.
        """
        filename = os.path.basename(file_path)
        caption = kwargs.get("caption", f"📎 File: {filename}")
        return await self.send_message(chat_id, caption)

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for new Twitch chat messages."""
        messages: List[IncomingMessage] = []
        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    # ── Extended Interface ────────────────────────────────────────────────

    async def join_channel(self, channel: str) -> bool:
        """Join a Twitch channel."""
        if not self._writer:
            return False
        if not channel.startswith("#"):
            channel = f"#{channel}"
        self._send_raw(f"JOIN {channel}")
        self._joined_channels.add(channel.lower())
        logger.info("Twitch joining channel: %s", channel)
        return True

    async def leave_channel(self, channel: str) -> bool:
        """Leave a Twitch channel."""
        if not self._writer:
            return False
        if not channel.startswith("#"):
            channel = f"#{channel}"
        self._send_raw(f"PART {channel}")
        self._joined_channels.discard(channel.lower())
        return True

    async def send_whisper(self, username: str, text: str) -> Optional[str]:
        """Send a whisper (private message) to a user."""
        if not self._writer:
            return None
        text = self._truncate(text)
        self._send_raw(f"PRIVMSG #jtv :/w {username} {text}")
        logger.info("Twitch whisper sent to %s", username)
        return f"whisper_{int(time.time())}"

    async def send_me_action(self, chat_id: str, text: str) -> Optional[str]:
        """Send a /me action message."""
        if not self._writer:
            return None
        if not chat_id.startswith("#"):
            chat_id = f"#{chat_id}"
        await self._enforce_rate_limit()
        self._send_raw(f"PRIVMSG {chat_id} :/me {text}")
        return f"action_{int(time.time())}"

    async def timeout_user(
        self, channel: str, username: str, duration: int = 600,
        reason: str = "",
    ) -> bool:
        """Timeout a user in a channel."""
        if not self._writer:
            return False
        if not channel.startswith("#"):
            channel = f"#{channel}"
        command = f"PRIVMSG {channel} :/timeout {username} {duration}"
        if reason:
            command += f" {reason}"
        self._send_raw(command)
        return True

    async def ban_user(
        self, channel: str, username: str, reason: str = "",
    ) -> bool:
        """Ban a user from a channel."""
        if not self._writer:
            return False
        if not channel.startswith("#"):
            channel = f"#{channel}"
        command = f"PRIVMSG {channel} :/ban {username}"
        if reason:
            command += f" {reason}"
        self._send_raw(command)
        return True

    async def get_stream_info(
        self, broadcaster_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get live stream information via the Twitch Helix API."""
        if not self._session:
            return None

        url = f"{TWITCH_API_BASE}/streams"
        params = {"user_login": broadcaster_name.lower()}

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    streams = data.get("data", [])
                    return streams[0] if streams else None
                return None
        except Exception as e:
            logger.error("Twitch API stream info error: %s", e)
            return None

    # ── Internal ──────────────────────────────────────────────────────────

    def _send_raw(self, line: str) -> None:
        """Send a raw IRC line to Twitch."""
        if self._writer and not self._writer.is_closing():
            self._writer.write(f"{line}\r\n".encode("utf-8"))

    async def _enforce_rate_limit(self) -> None:
        """Enforce Twitch chat rate limits."""
        now = time.time()

        # Reset counter every 30 seconds
        if now - self._message_window_start > 30:
            self._message_count = 0
            self._message_window_start = now

        if self._message_count >= TWITCH_MSG_LIMIT:
            wait = 30 - (now - self._message_window_start)
            if wait > 0:
                logger.debug("Twitch rate limit: waiting %.1fs", wait)
                await asyncio.sleep(wait)
            self._message_count = 0
            self._message_window_start = time.time()

        # Minimum delay between messages
        elapsed = now - self._last_message_time
        if elapsed < self._twitch_config.rate_limit_delay:
            await asyncio.sleep(self._twitch_config.rate_limit_delay - elapsed)

    async def _read_loop(self) -> None:
        """Read and dispatch IRC lines from Twitch."""
        while self._connected and self._reader:
            try:
                raw = await self._reader.readline()
                if not raw:
                    logger.info("Twitch IRC connection closed")
                    self._connected = False
                    break

                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    await self._handle_line(line)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Twitch IRC read error: %s", e)
                await asyncio.sleep(2)

    async def _ping_loop(self) -> None:
        """Send PING to keep the connection alive."""
        while self._connected:
            await asyncio.sleep(120)
            if self._connected and self._writer:
                self._send_raw("PING :tmi.twitch.tv")

    async def _handle_line(self, line: str) -> None:
        """Parse and handle a Twitch IRC line."""
        logger.debug("Twitch < %s", line)

        # PING/PONG
        if line.startswith("PING "):
            self._send_raw(f"PONG :{line[5:]}")
            return

        # Parse IRC message
        prefix, command, params = self._parse_irc_line(line)

        if command == "001":
            logger.info("Twitch IRC authenticated as %s", self._twitch_config.nickname)
        elif command == "376" or command == "422":
            logger.info("Twitch IRC: MOTD complete")
        elif command == "JOIN":
            nick = prefix.split("!")[0] if prefix else ""
            channel = params.get("trailing", "")
            if nick == self._twitch_config.nickname:
                logger.info("Twitch joined: %s", channel)
        elif command == "PART":
            nick = prefix.split("!")[0] if prefix else ""
            channel = params.get("middle", [""])[0]
            if nick == self._twitch_config.nickname:
                logger.info("Twitch left: %s", channel)
        elif command == "PRIVMSG":
            await self._handle_privmsg(prefix, params)
        elif command == "WHISPER":
            await self._handle_whisper(prefix, params)
        elif command == "USERNOTICE":
            await self._handle_usernotice(prefix, params)
        elif command == "CLEARCHAT":
            nick = params.get("trailing", "")
            logger.info("Twitch: chat cleared for %s", nick)
        elif command == "ROOMSTATE":
            logger.debug("Twitch roomstate update")

    async def _handle_privmsg(
        self, prefix: str, params: Dict[str, Any],
    ) -> None:
        """Handle a PRIVMSG from Twitch chat."""
        sender_nick = prefix.split("!")[0] if prefix else ""
        middle = params.get("middle", [])
        channel = middle[0] if middle else ""
        text = params.get("trailing", "")

        if not text.strip():
            return

        # Parse badges and metadata from IRCv3 tags
        tags = self._parse_tags(prefix)
        is_action = False
        if text.startswith("\x01ACTION ") and text.endswith("\x01"):
            text = text[8:-1]
            is_action = True

        is_command = text.startswith(self._twitch_config.command_prefix)

        msg = IncomingMessage(
            platform="twitch",
            chat_id=channel,
            user_id=sender_nick,
            text=text,
            username=sender_nick,
            is_command=is_command,
            metadata={
                "channel": channel,
                "is_action": is_action,
                "badges": tags.get("badges", ""),
                "user_type": tags.get("user-type", ""),
                "display_name": tags.get("display-name", sender_nick),
                "subscriber": tags.get("subscriber") == "1",
                "turbo": tags.get("turbo") == "1",
                "color": tags.get("color", ""),
                "emotes": tags.get("emotes", ""),
                "message_id": tags.get("id", ""),
            },
        )
        await self._message_queue.put(msg)

    async def _handle_whisper(
        self, prefix: str, params: Dict[str, Any],
    ) -> None:
        """Handle a whisper message."""
        sender_nick = prefix.split("!")[0] if prefix else ""
        text = params.get("trailing", "").strip()

        if not text:
            return

        tags = self._parse_tags(prefix)

        msg = IncomingMessage(
            platform="twitch",
            chat_id=f"whisper:{sender_nick}",
            user_id=sender_nick,
            text=text,
            username=sender_nick,
            metadata={
                "is_whisper": True,
                "display_name": tags.get("display-name", sender_nick),
                "badges": tags.get("badges", ""),
            },
        )
        await self._message_queue.put(msg)

    async def _handle_usernotice(
        self, prefix: str, params: Dict[str, Any],
    ) -> None:
        """Handle a USERNOTICE (sub, resub, gifted sub, raid)."""
        tags = self._parse_tags(prefix)
        middle = params.get("middle", [])
        channel = middle[0] if middle else ""
        system_msg = params.get("trailing", "")
        msg_id = tags.get("msg-id", "")

        event_text = f"[{msg_id}] {system_msg}"

        metadata = {
            "event_type": msg_id,
            "channel": channel,
            "system_message": system_msg,
            "display_name": tags.get("display-name", ""),
            "msg_param_months": tags.get("msg-param-months", ""),
            "msg_param_sub_plan": tags.get("msg-param-sub-plan", ""),
        }

        if msg_id == "raid":
            metadata["raiding_from"] = tags.get("msg-param-displayName", "")
            metadata["raiding_viewers"] = tags.get("msg-param-viewerCount", "")

        msg = IncomingMessage(
            platform="twitch",
            chat_id=channel,
            user_id=tags.get("login", ""),
            text=event_text,
            username=tags.get("display-name", ""),
            metadata=metadata,
        )
        await self._message_queue.put(msg)

    @staticmethod
    def _parse_tags(prefix: str) -> Dict[str, str]:
        """Parse IRCv3 tags from the prefix string."""
        tags: Dict[str, str] = {}
        if prefix.startswith("@"):
            tag_str = prefix.split(" ")[0][1:]
            for tag in tag_str.split(";"):
                if "=" in tag:
                    key, value = tag.split("=", 1)
                    tags[key] = value.replace("\\s", " ").replace("\\:", ";").replace("\\\\", "\\")
                else:
                    tags[tag] = ""
        return tags

    @staticmethod
    def _parse_irc_line(line: str) -> Tuple[str, str, Dict[str, Any]]:
        """Parse a raw IRC line into (prefix, command, params)."""
        prefix = ""
        trailing = ""

        # Strip tags
        if line.startswith("@"):
            # Tags are part of the prefix
            pass

        if line.startswith(":"):
            prefix, line = line[1:].split(" ", 1)

        if " :" in line:
            line, trailing = line.split(" :", 1)

        parts = line.split()
        command = parts[0].upper() if parts else ""
        middle = parts[1:]

        return prefix, command, {"middle": middle, "trailing": trailing}

    def _truncate(self, text: str) -> str:
        """Truncate text to Twitch chat message length limit."""
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return text
        return text[:self.MAX_MESSAGE_LENGTH - 3] + "..."
