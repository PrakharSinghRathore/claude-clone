"""
Signal adapter for the Atlas Gateway (via signal-cli).

Supports message send/receive, group management, and attachment support
through the signal-cli JSON RPC interface.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.signal import SignalAdapter

    config = PlatformConfig(
        name="signal",
        token="+1234567890",  # Phone number
        api_url="http://localhost:7583",
        enabled=True,
    )
    adapter = SignalAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.signal")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

SIGNAL_CLI_DEFAULT_URL = "http://localhost:7583"


class SignalAdapter:
    """
    Signal adapter using signal-cli JSON RPC API.

    Requires signal-cli running with JSON RPC enabled.

    Parameters
    ----------
    config:
        Platform configuration. Requires:
        - ``token``: Signal phone number (with country code)
        - ``api_url``: signal-cli JSON RPC URL (default: localhost:7583)
    """

    def __init__(self, config: Any):
        self._config = config
        self._phone_number = config.token or os.environ.get("SIGNAL_PHONE_NUMBER", "")
        self._api_url = config.api_url or SIGNAL_CLI_DEFAULT_URL
        self._timeout = config.timeout or 30
        self._connected = False
        self._session: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._ws: Optional[Any] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._request_id = 0

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to the signal-cli JSON RPC interface."""
        if not self._phone_number:
            raise ValueError("Signal phone number is required")
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        )

        # Verify signal-cli is running
        try:
            result = await self._rpc_call("getVersion")
            if result:
                logger.info("Signal adapter connected (signal-cli version: %s)", result)
            else:
                logger.warning("Could not verify signal-cli version")
        except Exception as e:
            logger.warning("Signal-cli connection issue: %s", e)

        # Start WebSocket listener
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from signal-cli."""
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
        return self._connected and self._session is not None

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a text message via signal-cli."""
        params = {
            "account": self._phone_number,
            "recipient": [chat_id],
            "message": text,
        }
        result = await self._rpc_call("send", params)
        return str(result.get("timestamp", "")) if result else None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send an attachment via signal-cli."""
        if not os.path.exists(file_path):
            return None

        params = {
            "account": self._phone_number,
            "recipient": [chat_id],
            "attachments": [file_path],
        }
        caption = kwargs.get("caption")
        if caption:
            params["message"] = caption

        result = await self._rpc_call("send", params)
        return str(result.get("timestamp", "")) if result else None

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll the message queue for new Signal messages."""
        messages: List[IncomingMessage] = []
        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    # ── Extended Interface ────────────────────────────────────────────────

    async def send_group_message(self, group_id: str, text: str) -> Optional[str]:
        """Send a message to a Signal group."""
        params = {
            "account": self._phone_number,
            "groupId": group_id,
            "message": text,
        }
        result = await self._rpc_call("send", params)
        return str(result.get("timestamp", "")) if result else None

    async def list_groups(self) -> List[Dict[str, Any]]:
        """List all Signal groups."""
        result = await self._rpc_call("listGroups", {"account": self._phone_number})
        return result if isinstance(result, list) else []

    async def send_typing(self, chat_id: str) -> None:
        """Send a typing indicator (not supported by signal-cli)."""
        pass

    async def react(self, chat_id: str, message_id: str, emoji: str) -> bool:
        """React to a message with an emoji."""
        params = {
            "account": self._phone_number,
            "recipient": [chat_id],
            "reaction": {"emoji": emoji, "targetAuthor": chat_id, "targetTimestamp": message_id},
        }
        result = await self._rpc_call("send", params)
        return bool(result)

    # ── WebSocket ─────────────────────────────────────────────────────────

    async def _ws_loop(self) -> None:
        """Listen for incoming messages via WebSocket."""
        ws_url = self._api_url.replace("http", "ws") + "/v1/receive/{account}".format(
            account=self._phone_number,
        )

        reconnect_delay = 1.0
        while self._connected:
            try:
                async with self._session.ws_connect(ws_url) as ws:
                    self._ws = ws
                    logger.info("Signal WebSocket connected")
                    reconnect_delay = 1.0

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                incoming = self._parse_ws_message(data)
                                if incoming:
                                    await self._message_queue.put(incoming)
                            except json.JSONDecodeError:
                                pass
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Signal WebSocket error: %s", e)

            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30.0)

    def _parse_ws_message(self, data: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Parse a WebSocket message from signal-cli."""
        if data.get("type") != "incoming":
            return None

        envelope = data.get("envelope", {})
        source = envelope.get("source", "")
        source_number = envelope.get("sourceNumber", source)
        source_name = envelope.get("sourceName", "")
        timestamp = envelope.get("timestamp", "")
        group_info = envelope.get("groupInfo")
        chat_id = group_info.get("groupId", source_number) if group_info else source_number

        data_msg = envelope.get("dataMessage", {})
        text = data_msg.get("message", "")

        attachments: List[str] = []
        for att in data_msg.get("attachments", []):
            att_path = att.get("storedFilename") or att.get("filename")
            if att_path:
                attachments.append(att_path)

        # Handle reactions
        reaction = envelope.get("reaction")
        if reaction:
            text = reaction.get("emoji", "[reaction]")

        return IncomingMessage(
            platform="signal",
            chat_id=chat_id,
            user_id=source_number,
            text=text,
            message_id=str(timestamp),
            username=source_name or source_number,
            attachments=attachments if attachments else None,
            metadata={
                "group_info": group_info,
                "is_group": bool(group_info),
            },
        )

    # ── RPC Helper ────────────────────────────────────────────────────────

    async def _rpc_call(
        self, method: str, params: Optional[Dict] = None,
    ) -> Any:
        """Make a JSON RPC call to signal-cli."""
        if not self._session:
            return None

        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }

        try:
            async with self._session.post(
                f"{self._api_url}/v1/rpc",
                json=payload,
            ) as resp:
                data = await resp.json()
                if "result" in data:
                    return data["result"]
                elif "error" in data:
                    logger.error("signal-cli RPC error: %s", data["error"])
                    return None
                return None
        except Exception as e:
            logger.error("signal-cli RPC call failed (%s): %s", method, e)
            return None
