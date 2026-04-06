"""
IRC channel adapter for the Atlas Gateway.

Supports connecting to IRC servers, joining channels, sending and receiving
messages, handling IRC commands, CTCP responses, and NickServ authentication.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.irc import IRCAdapter

    config = PlatformConfig(
        name="irc",
        token="",  # Not used; password via extra
        enabled=True,
        extra={
            "server": "irc.libera.chat",
            "port": 6697,
            "nickname": "AtlasBot",
            "password": "optional_nickserv_pass",
            "channels": ["#atlas-test"],
            "use_ssl": True,
        },
    )
    adapter = IRCAdapter(config)
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

logger = logging.getLogger("atlas.gateway.platforms.irc")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class IRCConfig:
    """Configuration for the IRC adapter."""

    server: str = "irc.libera.chat"
    port: int = 6697
    nickname: str = "AtlasBot"
    realname: str = "Atlas Gateway Bot"
    password: str = ""
    channels: List[str] = field(default_factory=list)
    use_ssl: bool = True
    timeout: int = 30
    command_prefix: str = "!"
    ping_interval: int = 120


class IRCAdapter:
    """
    IRC protocol adapter using raw TCP sockets with optional TLS.

    Supports channel messaging, private messages, NOTICE responses,
    CTCP handling, NickServ identification, and basic IRC protocol parsing.

    Parameters
    ----------
    config:
        Platform configuration. Server details are read from ``config.extra``
        or environment variables.
    """

    MAX_MESSAGE_LENGTH = 510  # IRC protocol limit
    READ_BUFFER_SIZE = 4096

    def __init__(self, config: Any):
        self._config = config
        extra = getattr(config, "extra", {}) or {}

        self._irc_config = IRCConfig(
            server=extra.get("server") or os.environ.get("IRC_SERVER", "irc.libera.chat"),
            port=int(extra.get("port") or os.environ.get("IRC_PORT", 6697)),
            nickname=extra.get("nickname") or os.environ.get("IRC_NICKNAME", "AtlasBot"),
            realname=extra.get("realname") or os.environ.get("IRC_REALNAME", "Atlas Gateway Bot"),
            password=extra.get("password") or os.environ.get("IRC_PASSWORD", ""),
            channels=extra.get("channels") or os.environ.get("IRC_CHANNELS", "").split(","),
            use_ssl=bool(extra.get("use_ssl", True)),
            timeout=config.timeout or 30,
            command_prefix=extra.get("command_prefix", "!"),
        )

        self._connected = False
        self._registered = False
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._read_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._current_nick: str = self._irc_config.nickname
        self._joined_channels: set = set()

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to the IRC server and join configured channels."""
        irc = self._irc_config

        try:
            if irc.use_ssl:
                ctx = ssl.create_default_context()
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(irc.server, irc.port, ssl=ctx),
                    timeout=irc.timeout,
                )
            else:
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(irc.server, irc.port),
                    timeout=irc.timeout,
                )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Connection to {irc.server}:{irc.port} timed out")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to IRC server: {e}")

        # Send registration
        self._send_raw(f"NICK {irc.nickname}")
        self._send_raw(f"USER {irc.nickname} 0 * :{irc.realname}")

        self._read_task = asyncio.create_task(self._read_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())

        self._connected = True
        logger.info(
            "Connecting to IRC %s:%s as %s",
            irc.server, irc.port, irc.nickname,
        )

    async def disconnect(self) -> None:
        """Gracefully disconnect from the IRC server."""
        self._connected = False
        self._registered = False

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

        self._reader = None
        self._joined_channels.clear()

    async def is_connected(self) -> bool:
        """Check if the IRC connection is active and registered."""
        return self._connected and self._registered and self._writer is not None

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a message to an IRC channel or user.

        Parameters
        ----------
        chat_id:
            Channel name (e.g. ``#atlas``) or user nick.
        text:
            Message text to send.
        """
        if not self._writer:
            return None

        text = self._truncate(text)
        # IRC requires lines split at 510 chars including CR-LF
        for line in text.split("\n"):
            if line.strip():
                self._send_raw(f"PRIVMSG {chat_id} :{line}")

        return f"irc_{int(time.time())}"

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a file via DCC SEND or as a URL reference.

        IRC has no native file transfer in channel mode, so this
        sends a notice with the filename as a fallback.
        """
        if not self._writer:
            return None

        filename = os.path.basename(file_path)
        notice = kwargs.get("caption", f"File shared: {filename}")
        self._send_raw(f"NOTICE {chat_id} :{notice}")
        logger.info("IRC file send: sent notice for %s to %s", filename, chat_id)
        return f"notice_{int(time.time())}"

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for new IRC messages from the internal queue."""
        messages: List[IncomingMessage] = []
        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    # ── Extended Interface ────────────────────────────────────────────────

    async def join_channel(self, channel: str, key: str = "") -> bool:
        """Join an IRC channel."""
        if not self._writer:
            return False

        if not channel.startswith("#"):
            channel = f"#{channel}"

        command = f"JOIN {channel}"
        if key:
            command += f" {key}"
        self._send_raw(command)
        self._joined_channels.add(channel.lower())
        logger.info("IRC joining channel: %s", channel)
        return True

    async def leave_channel(self, channel: str, reason: str = "") -> bool:
        """Leave an IRC channel."""
        if not self._writer:
            return False

        command = f"PART {channel}"
        if reason:
            command += f" :{reason}"
        self._send_raw(command)
        self._joined_channels.discard(channel.lower())
        return True

    async def send_action(self, chat_id: str, text: str) -> Optional[str]:
        """Send a CTCP ACTION (/me) message."""
        if not self._writer:
            return None
        self._send_raw(f"PRIVMSG {chat_id} :\x01ACTION {text}\x01")
        return f"action_{int(time.time())}"

    async def send_notice(self, chat_id: str, text: str) -> Optional[str]:
        """Send a NOTICE message (typically used by bots)."""
        if not self._writer:
            return None
        self._send_raw(f"NOTICE {chat_id} :{text}")
        return f"notice_{int(time.time())}"

    async def set_topic(self, channel: str, topic: str) -> bool:
        """Set a channel topic (requires operator privileges)."""
        if not self._writer:
            return False
        self._send_raw(f"TOPIC {channel} :{topic}")
        return True

    async def kick_user(
        self, channel: str, nickname: str, reason: str = "",
    ) -> bool:
        """Kick a user from a channel (requires operator privileges)."""
        if not self._writer:
            return False
        command = f"KICK {channel} {nickname}"
        if reason:
            command += f" :{reason}"
        self._send_raw(command)
        return True

    async def change_nick(self, new_nick: str) -> bool:
        """Change the bot's nickname."""
        if not self._writer:
            return False
        self._send_raw(f"NICK {new_nick}")
        self._current_nick = new_nick
        return True

    async def whois(self, nickname: str) -> Optional[Dict[str, str]]:
        """
        Request WHOIS information for a user.

        Returns a dict with available fields (may be partial).
        """
        if not self._writer:
            return None

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._whois_futures[nickname.lower()] = future
        self._send_raw(f"WHOIS {nickname}")

        try:
            return await asyncio.wait_for(future, timeout=10)
        except asyncio.TimeoutError:
            logger.warning("WHOIS for %s timed out", nickname)
            return None

    # ── Internal ──────────────────────────────────────────────────────────

    def _send_raw(self, line: str) -> None:
        """Send a raw line to the IRC server."""
        if self._writer and not self._writer.is_closing():
            self._writer.write(f"{line}\r\n".encode("utf-8"))

    def _whois_futures(self) -> Dict[str, asyncio.Future]:
        """Lazily initialise the whois futures dict."""
        if not hasattr(self, "_whois_fut_map"):
            self._whois_fut_map: Dict[str, asyncio.Future] = {}
        return self._whois_fut_map

    async def _read_loop(self) -> None:
        """Read and dispatch lines from the IRC server."""
        while self._connected and self._reader:
            try:
                raw = await self._reader.readline()
                if not raw:
                    logger.info("IRC connection closed by server")
                    self._connected = False
                    break

                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    await self._handle_line(line)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("IRC read error: %s", e)
                await asyncio.sleep(2)

    async def _ping_loop(self) -> None:
        """Send periodic PING to keep the connection alive."""
        while self._connected:
            await asyncio.sleep(self._irc_config.ping_interval)
            if self._connected and self._writer:
                self._send_raw(f"PING :{self._irc_config.server}")

    async def _handle_line(self, line: str) -> None:
        """Parse and handle a single IRC protocol line."""
        logger.debug("IRC < %s", line)

        # Server PING — must PONG back
        if line.startswith("PING "):
            token = line[5:]
            self._send_raw(f"PONG :{token}")
            return

        # Parse IRC message: :<prefix> <command> <params> :<trailing>
        prefix, command, params = self._parse_irc_line(line)

        if command == "001" or command == "RPL_WELCOME":
            self._registered = True
            logger.info("IRC registered successfully")

            # Identify with NickServ if password is set
            if self._irc_config.password:
                self._send_raw(
                    f"PRIVMSG NickServ :IDENTIFY {self._irc_config.password}"
                )
                await asyncio.sleep(2)

            # Join configured channels
            for channel in self._irc_config.channels:
                await self.join_channel(channel)

        elif command == "433":  # Nickname already in use
            logger.warning("IRC nickname %s is already in use", self._current_nick)
            self._current_nick = f"{self._current_nick}_"
            self._send_raw(f"NICK {self._current_nick}")

        elif command == "PRIVMSG":
            await self._handle_privmsg(prefix, params)

        elif command == "NOTICE":
            # Silently handle NOTICE from NickServ etc.
            sender = prefix.split("!")[0] if prefix else ""
            if sender.lower() == "nickserv":
                logger.debug("NickServ notice: %s", params.get("trailing", ""))

        elif command == "JOIN":
            nick = prefix.split("!")[0] if prefix else ""
            channel = params.get("trailing", params.get("middle", [""])[0])
            if nick != self._current_nick:
                logger.info("IRC: %s joined %s", nick, channel)

        elif command == "PART":
            nick = prefix.split("!")[0] if prefix else ""
            channel = params.get("middle", [""])[0]
            if nick != self._current_nick:
                logger.info("IRC: %s left %s", nick, channel)

        elif command == "KICK":
            middle = params.get("middle", [])
            if len(middle) >= 2:
                logger.info(
                    "IRC: %s was kicked from %s", middle[1], middle[0],
                )

    async def _handle_privmsg(
        self, prefix: str, params: Dict[str, Any],
    ) -> None:
        """Handle a PRIVMSG and convert it to an IncomingMessage."""
        sender_full = prefix or ""
        sender_nick = sender_full.split("!")[0]
        sender_user = sender_full.split("!")[1].split("@")[0] if "!" in sender_full else ""
        sender_host = sender_full.split("@")[1] if "@" in sender_full else ""

        middle = params.get("middle", [])
        channel = middle[0] if middle else ""
        text = params.get("trailing", "")

        # Strip CTCP markers for display
        is_action = False
        if text.startswith("\x01ACTION ") and text.endswith("\x01"):
            text = text[8:-1]
            is_action = True

        if not text.strip():
            return

        # Determine chat_id (channel or private query)
        chat_id = channel if channel != self._current_nick else sender_nick

        is_command = text.startswith(self._irc_config.command_prefix)

        msg = IncomingMessage(
            platform="irc",
            chat_id=chat_id,
            user_id=sender_nick,
            text=text,
            username=sender_nick,
            is_command=is_command,
            metadata={
                "full_host": sender_full,
                "user": sender_user,
                "host": sender_host,
                "is_action": is_action,
                "channel": channel,
            },
        )
        await self._message_queue.put(msg)

    @staticmethod
    def _parse_irc_line(line: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parse a raw IRC protocol line into (prefix, command, params).

        IRC message format::

            [:<prefix>] <command> <params> [:<trailing>]
        """
        prefix = ""
        trailing = ""

        if line.startswith(":"):
            prefix, line = line[1:].split(" ", 1)

        if " :" in line:
            line, trailing = line.split(" :", 1)

        parts = line.split()
        command = parts[0].upper() if parts else ""
        middle = parts[1:]

        return prefix, command, {"middle": middle, "trailing": trailing}

    def _truncate(self, text: str) -> str:
        """Truncate text to IRC's maximum line length."""
        limit = self.MAX_MESSAGE_LENGTH - 60  # Room for PRIVMSG header
        lines = text.split("\n")
        truncated = []
        for line in lines:
            if len(line.encode("utf-8")) <= limit:
                truncated.append(line)
            else:
                truncated.append(line[:limit - 3] + "...")
        return "\n".join(truncated)
