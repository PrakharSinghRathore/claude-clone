"""
LINE Messaging API adapter for the Atlas Gateway.

Supports the LINE Messaging API v2.1, reply/webhook handling,
rich menus, flex messages, quick replies, and multimedia messages.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.line import LINEAdapter

    config = PlatformConfig(
        name="line",
        token="LINE_CHANNEL_ACCESS_TOKEN",
        api_key="LINE_CHANNEL_SECRET",
        enabled=True,
    )
    adapter = LINEAdapter(config)
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
import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.line")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

LINE_API_BASE = "https://api.line.me/v2/bot"
LINE_DATA_API_BASE = "https://api-data.line.me/v2/bot"


@dataclass
class LINEConfig:
    """Configuration for the LINE Messaging adapter."""

    channel_access_token: str = ""
    channel_secret: str = ""
    timeout: int = 30
    allowed_user_ids: List[str] = field(default_factory=list)
    allowed_group_ids: List[str] = field(default_factory=list)
    auto_reply: bool = True


class LINEAdapter:
    """
    LINE Messaging API v2.1 adapter.

    Uses webhook events for receiving messages and the push/ reply APIs
    for sending. Supports text, image, video, audio, sticker, location,
    and Flex Message types.

    Parameters
    ----------
    config:
        Platform configuration with ``token`` (channel access token) and
        ``api_key`` (channel secret).
    """

    MAX_MESSAGE_LENGTH = 5000

    def __init__(self, config: Any):
        self._config = config
        extra = getattr(config, "extra", {}) or {}

        self._line_config = LINEConfig(
            channel_access_token=config.token or os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", ""),
            channel_secret=config.api_key or os.environ.get("LINE_CHANNEL_SECRET", ""),
            timeout=config.timeout or 30,
            allowed_user_ids=extra.get("allowed_user_ids", []),
            allowed_group_ids=extra.get("allowed_group_ids", []),
            auto_reply=extra.get("auto_reply", True),
        )

        self._connected = False
        self._session: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._reply_tokens: Dict[str, float] = {}

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize the LINE Messaging API adapter."""
        if not self._line_config.channel_access_token:
            raise ValueError(
                "LINE channel access token is required. "
                "Set LINE_CHANNEL_ACCESS_TOKEN env-var or config.token."
            )
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._line_config.timeout),
            headers={
                "Authorization": f"Bearer {self._line_config.channel_access_token}",
            },
        )

        # Verify bot info
        try:
            async with self._session.get(f"{LINE_API_BASE}/info") as resp:
                if resp.status == 200:
                    info = await resp.json()
                    logger.info(
                        "LINE adapter connected: %s (@%s)",
                        info.get("displayName", "unknown"),
                        info.get("userId", "unknown"),
                    )
        except Exception as e:
            logger.warning("Could not verify LINE bot info: %s", e)

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the LINE Messaging API."""
        self._connected = False
        if self._session:
            await self._session.close()
            self._session = None
        self._reply_tokens.clear()

    async def is_connected(self) -> bool:
        """Check if the adapter is connected."""
        return self._connected and self._session is not None

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a push message to a LINE user, group, or room.

        Parameters
        ----------
        chat_id:
            User ID, group ID, or room ID.
        text:
            Message text to send.
        """
        if not self._session:
            return None

        text = self._truncate(text)

        payload: Dict[str, Any] = {
            "to": chat_id,
            "messages": [{"type": "text", "text": text}],
        }

        # Quick reply support
        quick_reply = kwargs.get("quick_reply")
        if quick_reply:
            payload["messages"][0]["quickReply"] = quick_reply

        # Flex message support
        flex = kwargs.get("flex")
        if flex:
            payload["messages"] = [flex]

        url = f"{LINE_API_BASE}/message/push"
        return await self._line_post(url, payload)

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send an image, video, or audio file to a LINE chat.

        LINE has separate upload and send steps for media.
        """
        if not self._session or not os.path.exists(file_path):
            return None

        filename = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        # Determine message type from extension
        media_type = None
        content_type = None
        if ext in (".jpg", ".jpeg", ".png"):
            media_type = "image"
            content_type = "image/jpeg" if ext != ".png" else "image/png"
        elif ext in (".mp4",):
            media_type = "video"
            content_type = "video/mp4"
        elif ext in (".m4a", ".mp3"):
            media_type = "audio"
            content_type = "audio/mpeg"
        else:
            # Unsupported type — send as text reference
            return await self.send_message(
                chat_id, f"📎 File: {filename}",
            )

        try:
            # Upload content
            upload_url = f"{LINE_DATA_API_BASE}/message/{media_type}/{chat_id}"

            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=filename, content_type=content_type)

                async with self._session.post(upload_url, data=data) as resp:
                    if resp.status == 200:
                        logger.info("LINE %s sent to %s", media_type, chat_id)
                        return f"{media_type}_{int(time.time())}"
                    error = await resp.text()
                    logger.error("LINE upload error: %s", error[:200])
                    return None
        except Exception as e:
            logger.error("Failed to send file via LINE: %s", e)
            return None

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for new messages from the webhook queue."""
        messages: List[IncomingMessage] = []
        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    # ── Webhook / Event Handling ──────────────────────────────────────────

    def parse_webhook_event(
        self, event: Dict[str, Any],
    ) -> Optional[IncomingMessage]:
        """
        Parse a LINE webhook event into an IncomingMessage.

        Handles message, follow, unfollow, join, leave, and
        postback events.
        """
        event_type = event.get("type", "")
        source = event.get("source", {})
        source_type = source.get("type", "")

        if source_type == "user":
            chat_id = source.get("userId", "")
        elif source_type == "group":
            chat_id = source.get("groupId", "")
        elif source_type == "room":
            chat_id = source.get("roomId", "")
        else:
            return None

        if event_type == "message":
            return self._parse_message_event(event, chat_id, source)
        elif event_type == "follow":
            logger.info("LINE: user %s followed the bot", chat_id)
            return None
        elif event_type == "unfollow":
            logger.info("LINE: user %s unfollowed the bot", chat_id)
            return None
        elif event_type == "join":
            logger.info("LINE: bot joined %s", chat_id)
            return None
        elif event_type == "postback":
            data = event.get("postback", {}).get("data", "")
            return IncomingMessage(
                platform="line",
                chat_id=chat_id,
                user_id=source.get("userId", ""),
                text=data,
                message_id=event.get("webhookEventId"),
                is_command=True,
                metadata={"event_type": "postback"},
            )

        return None

    def enqueue_event(self, event: Dict[str, Any]) -> None:
        """Parse and enqueue a webhook event."""
        msg = self.parse_webhook_event(event)
        if msg:
            self._message_queue.put_nowait(msg)

    async def reply_to_event(
        self, reply_token: str, text: str, **kwargs: Any,
    ) -> bool:
        """
        Send a reply message using a reply token.

        Reply tokens expire after a short period and can only be used once.
        """
        if not self._session:
            return False

        text = self._truncate(text)

        payload: Dict[str, Any] = {
            "replyToken": reply_token,
            "messages": [{"type": "text", "text": text}],
        }

        # Flex message
        flex = kwargs.get("flex")
        if flex:
            payload["messages"] = [flex]

        url = f"{LINE_API_BASE}/message/reply"
        result = await self._line_post(url, payload, expect_body=False)
        return bool(result)

    def verify_webhook_signature(self, body: bytes, signature: str) -> bool:
        """Verify the LINE webhook signature using HMAC-SHA256."""
        secret = self._line_config.channel_secret
        if not secret or not signature:
            return False

        expected = hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256,
        ).digest()
        expected_b64 = base64.b64encode(expected).decode("utf-8")

        return hmac.compare_digest(expected_b64, signature)

    # ── Flex Message Builder ──────────────────────────────────────────────

    @staticmethod
    def create_flex_bubble(
        body_contents: List[Dict[str, Any]],
        header_contents: Optional[List[Dict[str, Any]]] = None,
        footer_contents: Optional[List[Dict[str, Any]]] = None,
        styles: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a Flex Message bubble container."""
        bubble: Dict[str, Any] = {
            "type": "bubble",
            "body": {"type": "box", "layout": "vertical", "contents": body_contents},
        }
        if header_contents:
            bubble["header"] = {"type": "box", "layout": "vertical", "contents": header_contents}
        if footer_contents:
            bubble["footer"] = {"type": "box", "layout": "vertical", "contents": footer_contents}
        if styles:
            bubble["styles"] = styles
        return bubble

    @staticmethod
    def create_flex_message(
        alt_text: str, bubble: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Wrap a Flex bubble in a full Flex Message envelope."""
        return {
            "type": "flex",
            "altText": alt_text,
            "contents": bubble,
        }

    @staticmethod
    def create_quick_reply(
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create a quick reply object with action buttons."""
        return {"items": items}

    @staticmethod
    def create_quick_reply_button(
        label: str, action_type: str = "message", data: str = "",
        image_url: str = "",
    ) -> Dict[str, Any]:
        """Create a single quick reply item."""
        action: Dict[str, Any] = {
            "type": action_type,
            "label": label,
        }
        if action_type == "message":
            action["text"] = data or label
        elif action_type == "postback":
            action["data"] = data
        elif action_type == "uri":
            action["uri"] = data
        if image_url:
            action["imageUrl"] = image_url
        return action

    # ── Profile API ───────────────────────────────────────────────────────

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's display name and profile image."""
        if not self._session:
            return None
        url = f"{LINE_API_BASE}/profile/{user_id}"
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error("LINE get profile error: %s", e)
            return None

    async def get_group_member_profile(
        self, group_id: str, user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a group member's profile."""
        if not self._session:
            return None
        url = f"{LINE_API_BASE}/group/{group_id}/member/{user_id}"
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error("LINE get group member profile error: %s", e)
            return None

    # ── Internal ──────────────────────────────────────────────────────────

    def _parse_message_event(
        self, event: Dict[str, Any], chat_id: str, source: Dict[str, Any],
    ) -> Optional[IncomingMessage]:
        """Parse a LINE message event into IncomingMessage."""
        message = event.get("message", {})
        msg_type = message.get("type", "")

        user_id = source.get("userId", "")
        reply_token = event.get("replyToken", "")

        # Store reply token with expiry
        if reply_token:
            self._reply_tokens[reply_token] = time.time() + 1800  # 30 min

        # Check allow-lists
        if self._line_config.allowed_user_ids and user_id not in self._line_config.allowed_user_ids:
            return None

        text = message.get("text", "")
        metadata: Dict[str, Any] = {
            "msg_type": msg_type,
            "reply_token": reply_token,
            "source_type": source.get("type"),
        }

        if msg_type == "text":
            text = text.strip()
            if not text:
                return None
        elif msg_type == "image":
            metadata["content_provider"] = message.get("contentProvider", {})
            text = "[Image]"
        elif msg_type == "video":
            metadata["content_provider"] = message.get("contentProvider", {})
            text = "[Video]"
        elif msg_type == "audio":
            metadata["duration"] = message.get("duration", 0)
            text = "[Audio]"
        elif msg_type == "sticker":
            metadata["sticker_id"] = message.get("stickerId")
            metadata["package_id"] = message.get("packageId")
            text = f"[Sticker: {message.get('stickerId', '')}]"
        elif msg_type == "location":
            loc = message.get("location", {})
            metadata["location"] = {
                "title": loc.get("title", ""),
                "address": loc.get("address", ""),
                "lat": loc.get("latitude", 0),
                "lng": loc.get("longitude", 0),
            }
            text = f"[Location: {loc.get('title', loc.get('address', ''))}]"
        else:
            metadata["raw_message"] = message
            text = f"[{msg_type}]"

        return IncomingMessage(
            platform="line",
            chat_id=chat_id,
            user_id=user_id,
            text=text,
            message_id=event.get("webhookEventId"),
            metadata=metadata,
        )

    async def _line_post(
        self, url: str, payload: Dict[str, Any], expect_body: bool = True,
    ) -> Optional[str]:
        """Make a POST request to the LINE API."""
        if not self._session:
            return None

        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    if expect_body:
                        result = await resp.json()
                        return result.get("status", "sent")
                    return "sent"
                error = await resp.text()
                logger.error("LINE API error (%s): %s", url, error[:200])
                return None
        except Exception as e:
            logger.error("LINE API request failed: %s", e)
            return None

    def _truncate(self, text: str) -> str:
        """Truncate text to LINE's message length limit."""
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return text
        return text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n[...truncated]"
