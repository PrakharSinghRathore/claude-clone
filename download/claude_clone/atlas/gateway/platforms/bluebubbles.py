"""
BlueBubbles / iMessage adapter for the Atlas Gateway.

Supports the BlueBubbles server API for sending and receiving iMessages,
SMS texts, and RCS messages via a self-hosted BlueBubbles instance.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.bluebubbles import BlueBubblesAdapter

    config = PlatformConfig(
        name="bluebubbles",
        token="BLUEBUBBLES_API_KEY",
        api_key="BLUEBUBBLES_API_KEY",  # same as token
        enabled=True,
        extra={
            "server_url": "http://localhost:1234",
        },
    )
    adapter = BlueBubblesAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.bluebubbles")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

BLUEBUBBLES_API_VERSION = "v1"
BLUEBUBBLES_WS_PATH = "/api/v1/ws"


@dataclass
class BlueBubblesConfig:
    """Configuration for the BlueBubbles adapter."""

    server_url: str = ""
    api_key: str = ""
    password: str = ""
    timeout: int = 30
    auto_read: bool = True
    max_reconnect_delay: float = 30.0
    private_api: bool = True


class BlueBubblesAdapter:
    """
    BlueBubbles / iMessage adapter using the BlueBubbles REST API.

    BlueBubbles is a self-hosted server that provides an API for
    interacting with Apple's iMessage, SMS, and RCS services.

    Messages are received via WebSocket connection and outgoing
    messages use the REST API endpoints for text, attachments,
    reactions, and read receipts.

    Parameters
    ----------
    config:
        Platform configuration with ``token`` or ``api_key``
        (BlueBubbles API password/key) and ``server_url`` in extra.
    """

    MAX_MESSAGE_LENGTH = 65536  # iMessages support long texts

    def __init__(self, config: Any):
        self._config = config
        extra = getattr(config, "extra", {}) or {}

        self._bb_config = BlueBubblesConfig(
            server_url=extra.get("server_url") or os.environ.get("BLUEBUBBLES_URL", "http://localhost:1234"),
            api_key=config.token or config.api_key or os.environ.get("BLUEBUBBLES_API_KEY", ""),
            password=extra.get("password") or os.environ.get("BLUEBUBBLES_PASSWORD", ""),
            timeout=config.timeout or 30,
            auto_read=extra.get("auto_read", True),
            private_api=extra.get("private_api", True),
        )

        self._connected = False
        self._session: Optional[Any] = None
        self._ws: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._ws_task: Optional[asyncio.Task] = None
        self._chat_cache: Dict[str, Dict[str, Any]] = {}
        self._api_base = ""

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to the BlueBubbles server."""
        server = self._bb_config.server_url.rstrip("/")
        self._api_base = f"{server}/api/{BLUEBUBBLES_API_VERSION}"

        if not self._bb_config.api_key:
            raise ValueError(
                "BlueBubbles API key is required. "
                "Set BLUEBUBBLES_API_KEY env-var or config.token."
            )
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._bb_config.timeout),
            headers={"Authorization": f"Bearer {self._bb_config.api_key}"},
        )

        # Verify connection
        try:
            async with self._session.get(
                f"{self._api_base}/about",
            ) as resp:
                if resp.status == 200:
                    info = await resp.json()
                    logger.info(
                        "BlueBubbles connected: v%s (server=%s)",
                        info.get("version", "unknown"),
                        self._bb_config.server_url,
                    )
                else:
                    error = await resp.text()
                    logger.warning(
                        "BlueBubbles about endpoint returned %d: %s",
                        resp.status, error[:200],
                    )
        except Exception as e:
            logger.error("Could not reach BlueBubbles server: %s", e)
            await self.disconnect()
            raise RuntimeError(f"BlueBubbles server unreachable: {e}")

        # Start WebSocket for real-time updates
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the BlueBubbles server."""
        self._connected = False

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None

        if self._session:
            await self._session.close()
            self._session = None

        self._chat_cache.clear()

    async def is_connected(self) -> bool:
        """Check if the adapter is connected."""
        return (
            self._connected
            and self._session is not None
            and (self._ws is not None and not self._ws.closed)
        )

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a text message via iMessage/SMS.

        Parameters
        ----------
        chat_id:
            BlueBubbles chat GUID or a phone number / Apple ID.
        text:
            Message text to send.
        """
        if not self._session:
            return None

        text = self._truncate(text)

        payload: Dict[str, Any] = {
            "chatGuid": chat_id,
            "tempGuid": str(uuid.uuid4()),
            "message": text,
            "method": "apple-script" if self._bb_config.private_api else "xmpp",
        }

        # Style options
        if kwargs.get("bold"):
            payload["style"] = payload.get("style", {})
            payload["style"]["bold"] = True
        if kwargs.get("italic"):
            payload["style"] = payload.get("style", {})
            payload["style"]["italic"] = True
        if kwargs.get("subject"):
            payload["subject"] = kwargs["subject"]

        # Reply support
        if kwargs.get("reply_to"):
            payload["selectedMessageGuid"] = kwargs["reply_to"]

        # Effect support (echo, slam, etc.)
        if kwargs.get("effect"):
            payload["effectId"] = kwargs["effect"]

        url = f"{self._api_base}/message/text"
        result = await self._bb_post(url, payload)

        if result and result.get("status") == 200:
            return result.get("data", {}).get("guid")
        return None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a file attachment via iMessage.

        Supports images, videos, audio files, contacts, and
        location pins.
        """
        if not self._session or not os.path.exists(file_path):
            return None

        filename = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        try:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("attachment", f, filename=filename)
                data.add_field("chatGuid", chat_id)
                data.add_field("tempGuid", str(uuid.uuid4()))

                # Platform-specific path for attachment injection
                if kwargs.get("apple_script_path"):
                    data.add_field(
                        "attachmentPath", kwargs["apple_script_path"],
                    )

                if kwargs.get("subject"):
                    data.add_field("subject", kwargs["subject"])

                url = f"{self._api_base}/message/attachment"
                result = await self._bb_post(url, data, is_form=True)

                if result and result.get("status") == 200:
                    return result.get("data", {}).get("guid")
                logger.error(
                    "BlueBubbles send file error: %s",
                    result.get("error", "unknown") if result else "no response",
                )
                return None
        except Exception as e:
            logger.error("Failed to send file via BlueBubbles: %s", e)
            return None

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for new messages from the WebSocket queue."""
        messages: List[IncomingMessage] = []
        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    # ── Extended Interface ────────────────────────────────────────────────

    async def send_reaction(
        self, chat_id: str, message_guid: str, reaction: str,
    ) -> bool:
        """
        React to a message with an emoji reaction.

        Parameters
        ----------
        chat_id:
            BlueBubbles chat GUID.
        message_guid:
            GUID of the message to react to.
        reaction:
            Emoji character or 'remove' to remove a reaction.
        """
        if not self._session:
            return False

        payload = {
            "chatGuid": chat_id,
            "tempGuid": str(uuid.uuid4()),
            "messageGuid": message_guid,
            "reaction": reaction,
        }

        url = f"{self._api_base}/message/react"
        result = await self._bb_post(url, payload)
        return bool(result and result.get("status") == 200)

    async def mark_as_read(
        self, chat_id: str, message_guid: Optional[str] = None,
    ) -> bool:
        """Mark a conversation or message as read."""
        if not self._session:
            return False

        payload: Dict[str, Any] = {"chatGuid": chat_id}
        if message_guid:
            payload["lastMessageGuid"] = message_guid

        url = f"{self._api_base}/message/read"
        result = await self._bb_post(url, payload)
        return bool(result and result.get("status") == 200)

    async def get_chat_list(self) -> List[Dict[str, Any]]:
        """Get a list of all recent chats."""
        if not self._session:
            return []

        url = f"{self._api_base}/chat"
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("data", [])
                return []
        except Exception as e:
            logger.error("BlueBubbles chat list error: %s", e)
            return []

    async def get_chat_info(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific chat."""
        if not self._session:
            return None

        url = f"{self._api_base}/chat/{chat_id}"
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("data")
                return None
        except Exception as e:
            logger.error("BlueBubbles chat info error: %s", e)
            return None

    async def send_location(
        self, chat_id: str, lat: float, lng: float,
        label: str = "", address: str = "",
    ) -> Optional[str]:
        """Send a location pin via iMessage."""
        if not self._session:
            return None

        payload: Dict[str, Any] = {
            "chatGuid": chat_id,
            "tempGuid": str(uuid.uuid4()),
            "latitude": lat,
            "longitude": lng,
            "method": "apple-script" if self._bb_config.private_api else "xmpp",
        }

        if label:
            payload["label"] = label
        if address:
            payload["address"] = address

        url = f"{self._api_base}/message/location"
        result = await self._bb_post(url, payload)

        if result and result.get("status") == 200:
            return result.get("data", {}).get("guid")
        return None

    # ── WebSocket ─────────────────────────────────────────────────────────

    async def _ws_loop(self) -> None:
        """WebSocket loop for receiving real-time message events."""
        reconnect_delay = 1.0
        server = self._bb_config.server_url.rstrip("/")

        while self._connected:
            try:
                ws_url = f"{server.replace('http', 'ws')}{BLUEBUBBLES_WS_PATH}"
                headers = {"Authorization": f"Bearer {self._bb_config.api_key}"}

                async with self._session.ws_connect(ws_url, headers=headers) as ws:
                    self._ws = ws
                    logger.info("BlueBubbles WebSocket connected")
                    reconnect_delay = 1.0

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_ws_message(msg.data)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            logger.warning("BlueBubbles WebSocket disconnected")
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("BlueBubbles WebSocket error: %s", e)

            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(
                reconnect_delay * 2, self._bb_config.max_reconnect_delay,
            )

    async def _handle_ws_message(self, raw: str) -> None:
        """Handle a WebSocket message from BlueBubbles."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("BlueBubbles WS: non-JSON message: %s", raw[:100])
            return

        event_type = data.get("type", "")
        payload = data.get("data", {})

        if event_type == "message":
            await self._process_message(payload)
        elif event_type == "typing":
            chat_id = payload.get("chatGuid", "")
            logger.debug("BlueBubbles: typing in %s", chat_id)
        elif event_type == "read":
            chat_id = payload.get("chatGuid", "")
            logger.debug("BlueBubbles: read in %s", chat_id)

    async def _process_message(self, message: Dict[str, Any]) -> None:
        """Process a received iMessage/SMS message."""
        chat_guid = message.get("chatGuid", "")
        text = message.get("text", "")
        message_guid = message.get("guid", "")
        sender_guid = message.get("handleId", "")
        sender = message.get("sender", "")
        is_from_me = message.get("isFromMe", False)

        # Ignore messages sent by this account
        if is_from_me:
            return

        if not text.strip():
            # Handle attachments
            attachments = message.get("attachments", [])
            if attachments:
                att_urls = [a.get("url", "") for a in attachments if a.get("url")]
                text = "[Attachment]" if att_urls else ""
                if not text:
                    return
            else:
                return

        # Extract sender info
        sender_name = message.get("senderDisplayName", "") or message.get("sender", "")
        phone_number = message.get("sourceGuid", "") or message.get("sender", "")

        metadata: Dict[str, Any] = {
            "chat_guid": chat_guid,
            "message_guid": message_guid,
            "sender_guid": sender_guid,
            "is_from_me": is_from_me,
            "date_created": message.get("dateCreated", ""),
            "date_read": message.get("dateRead", ""),
            "error_code": message.get("errorCode"),
            "subject": message.get("subject", ""),
        }

        # Attachment metadata
        attachments = message.get("attachments", [])
        if attachments:
            metadata["attachments"] = [
                {
                    "url": a.get("url", ""),
                    "filename": a.get("filename", ""),
                    "mime_type": a.get("mimeType", ""),
                }
                for a in attachments
            ]

        # Tapback/reaction detection
        tapback = message.get("tapback", {})
        if tapback:
            metadata["tapback"] = tapback
            metadata["is_reaction"] = True

        # Auto-mark as read
        if self._bb_config.auto_read:
            asyncio.create_task(self.mark_as_read(chat_guid))

        msg = IncomingMessage(
            platform="bluebubbles",
            chat_id=chat_guid,
            user_id=sender_guid or phone_number,
            text=text,
            message_id=message_guid,
            username=sender_name,
            metadata=metadata,
        )
        await self._message_queue.put(msg)

    # ── Internal ──────────────────────────────────────────────────────────

    async def _bb_post(
        self, url: str, payload: Any, is_form: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Make a POST request to the BlueBubbles API."""
        if not self._session:
            return None

        try:
            if is_form:
                async with self._session.post(url, data=payload) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    error = await resp.text()
                    logger.error("BlueBubbles POST error (%d): %s", resp.status, error[:200])
                    return None
            else:
                async with self._session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    error = await resp.text()
                    logger.error("BlueBubbles POST error (%d): %s", resp.status, error[:200])
                    return None
        except Exception as e:
            logger.error("BlueBubbles API request failed: %s", e)
            return None

    def _truncate(self, text: str) -> str:
        """Truncate text to iMessage length limit."""
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return text
        return text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n[...truncated]"
