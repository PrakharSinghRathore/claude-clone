"""
WhatsApp adapter for the Atlas Gateway (via webhook).

Supports message sending/receiving, media messages, group support,
and read receipts through the WhatsApp Business API webhook interface.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.whatsapp import WhatsAppAdapter

    config = PlatformConfig(
        name="whatsapp",
        token="WHATSAPP_TOKEN",
        api_url="https://graph.facebook.com/v18.0",
        api_key="PHONE_NUMBER_ID",
        enabled=True,
    )
    adapter = WhatsAppAdapter(config)
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
from typing import Any, Dict, List, Optional

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.whatsapp")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class WhatsAppAdapter:
    """
    WhatsApp Business API adapter.

    Uses the Cloud API (formerly Business API) with webhook for receiving
    and REST API for sending messages.

    Parameters
    ----------
    config:
        Platform configuration. Requires:
        - ``token``: WhatsApp Business API access token
        - ``api_url``: API base URL (default: Meta Graph API)
        - ``api_key``: Phone number ID for sending
        - ``webhook_secret``: Verify token for webhook verification
    """

    DEFAULT_API_URL = "https://graph.facebook.com/v18.0"

    def __init__(self, config: Any):
        self._config = config
        self._token = config.token or os.environ.get("WHATSAPP_TOKEN", "")
        self._api_url = config.api_url or self.DEFAULT_API_URL
        self._phone_number_id = config.api_key or os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
        self._verify_token = config.webhook_secret or os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
        self._timeout = config.timeout or 30
        self._connected = False
        self._session: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize the WhatsApp adapter (validates token)."""
        if not self._token:
            raise ValueError("WhatsApp access token is required")
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
            headers={"Authorization": f"Bearer {self._token}"},
        )
        self._connected = True
        logger.info("WhatsApp adapter initialized (phone_number_id=%s)", self._phone_number_id)

    async def disconnect(self) -> None:
        """Disconnect the WhatsApp adapter."""
        self._connected = False
        if self._session:
            await self._session.close()
            self._session = None

    async def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a text message to a WhatsApp number."""
        url = f"{self._api_url}/{self._phone_number_id}/messages"
        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": chat_id,
            "type": "text",
            "text": {"body": text[:65536]},
        }

        # Reply context
        if kwargs.get("reply_to"):
            payload["context"] = {"message_id": kwargs["reply_to"]}

        result = await self._api_post(url, payload)
        return result.get("messages", [{}])[0].get("id") if result else None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a file/media message to a WhatsApp number."""
        if not self._session or not os.path.exists(file_path):
            return None

        # Upload media first
        media_id = await self._upload_media(file_path)
        if not media_id:
            return None

        caption = kwargs.get("caption", "")
        url = f"{self._api_url}/{self._phone_number_id}/messages"

        # Determine media type
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            media_type = "image"
        elif ext in (".mp4", ".3gp"):
            media_type = "video"
        elif ext in (".mp3", ".ogg", ".aac", ".m4a"):
            media_type = "audio"
        else:
            media_type = "document"

        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": chat_id,
            "type": media_type,
            media_type: {"id": media_id},
        }

        if caption and media_type in ("image", "video", "document"):
            payload[media_type]["caption"] = caption[:1024]

        result = await self._api_post(url, payload)
        return result.get("messages", [{}])[0].get("id") if result else None

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

    # ── Extended Interface ────────────────────────────────────────────────

    async def send_template(
        self, chat_id: str, template_name: str,
        language_code: str = "en_US",
        components: Optional[List[Dict]] = None,
    ) -> Optional[str]:
        """Send a template message."""
        url = f"{self._api_url}/{self._phone_number_id}/messages"
        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": chat_id,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language_code},
            },
        }
        if components:
            payload["template"]["components"] = components

        result = await self._api_post(url, payload)
        return result.get("messages", [{}])[0].get("id") if result else None

    async def mark_as_read(self, message_id: str) -> bool:
        """Mark a message as read."""
        url = f"{self._api_url}/{self._phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        result = await self._api_post(url, payload)
        return bool(result and result.get("success"))

    # ── Webhook Handling ──────────────────────────────────────────────────

    def verify_webhook(
        self, mode: str, challenge: str, token: str,
    ) -> Optional[str]:
        """
        Verify a webhook subscription.

        Returns the challenge string if valid, None otherwise.
        """
        if mode == "subscribe" and token == self._verify_token:
            return challenge
        return None

    def verify_webhook_signature(self, body: bytes, signature: str) -> bool:
        """Verify the X-Hub-Signature-256 header."""
        app_secret = getattr(self._config, "webhook_secret", "")
        if not app_secret:
            return True  # Skip verification if no secret
        expected = "sha256=" + hmac.new(
            app_secret.encode(), body, hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def handle_webhook_event(self, data: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Parse a webhook event into an IncomingMessage."""
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                contacts = value.get("contacts", [])

                contact_map: Dict[str, str] = {}
                for c in contacts:
                    wa_id = c.get("wa_id", "")
                    name = c.get("profile", {}).get("name", "")
                    contact_map[wa_id] = name

                for msg_data in messages:
                    msg = self._parse_message(msg_data, contact_map)
                    if msg:
                        return msg
        return None

    # ── Internal ──────────────────────────────────────────────────────────

    async def _upload_media(self, file_path: str) -> Optional[str]:
        """Upload a media file and return its media ID."""
        url = f"{self._api_url}/{self._phone_number_id}/media"
        mime_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".mp4": "video/mp4",
            ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
            ".pdf": "application/pdf",
        }

        ext = os.path.splitext(file_path)[1].lower()
        mime = mime_types.get(ext, "application/octet-stream")

        try:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=os.path.basename(file_path), content_type=mime)
                data.add_field("messaging_product", "whatsapp")
                data.add_field("type", mime)

                async with self._session.post(url, data=data) as resp:
                    result = await resp.json()
                    if result.get("id"):
                        return result["id"]
                    logger.error("WhatsApp media upload error: %s", result)
                    return None
        except Exception as e:
            logger.error("WhatsApp media upload failed: %s", e)
            return None

    def _parse_message(
        self, msg_data: Dict[str, Any], contact_map: Dict[str, str],
    ) -> Optional[IncomingMessage]:
        """Parse a WhatsApp message."""
        msg_type = msg_data.get("type", "")
        sender = msg_data.get("from", "")
        msg_id = msg_data.get("id", "")

        text = ""
        metadata: Dict[str, Any] = {"message_type": msg_type}

        if msg_type == "text":
            text = msg_data.get("text", {}).get("body", "")
        elif msg_type == "image":
            text = msg_data.get("image", {}).get("caption", "")
            metadata["media_id"] = msg_data.get("image", {}).get("id")
        elif msg_type == "document":
            text = msg_data.get("document", {}).get("caption", "")
            metadata["media_id"] = msg_data.get("document", {}).get("id")
            metadata["filename"] = msg_data.get("document", {}).get("filename")
        elif msg_type == "audio":
            metadata["media_id"] = msg_data.get("audio", {}).get("id")
        elif msg_type == "video":
            text = msg_data.get("video", {}).get("caption", "")
            metadata["media_id"] = msg_data.get("video", {}).get("id")
        elif msg_type == "interactive":
            text = msg_data.get("interactive", {}).get("button_reply", {}).get("title", "")
        elif msg_type == "sticker":
            metadata["media_id"] = msg_data.get("sticker", {}).get("id")

        context = msg_data.get("context")
        reply_to = context.get("id") if context else None

        return IncomingMessage(
            platform="whatsapp",
            chat_id=sender,
            user_id=sender,
            text=text,
            message_id=msg_id,
            username=contact_map.get(sender),
            reply_to=reply_to,
            metadata=metadata,
        )

    async def _api_post(
        self, url: str, payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Make a POST API call."""
        if not self._session:
            return None
        try:
            async with self._session.post(url, json=payload) as resp:
                data = await resp.json()
                if resp.status == 200:
                    return data
                logger.error("WhatsApp API error: %s", data.get("error", {}).get("message", data))
                return None
        except Exception as e:
            logger.error("WhatsApp API call failed: %s", e)
            return None
