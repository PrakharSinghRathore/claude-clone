"""
Feishu/Lark adapter for the Hermes Gateway.

Supports bot message sending, card messages, and group/DM messaging
through the Feishu Open API.

Usage::

    from hermes.gateway.config import PlatformConfig
    from hermes.gateway.platforms.feishu import FeishuAdapter

    config = PlatformConfig(
        name="feishu",
        token="FEISHU_APP_ID",
        api_key="FEISHU_APP_SECRET",
        enabled=True,
    )
    adapter = FeishuAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from hermes.gateway.runner import IncomingMessage

logger = logging.getLogger("hermes.gateway.platforms.feishu")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

FEISHU_API_BASE = "https://open.feishu.cn/open-apis"


class FeishuAdapter:
    """
    Feishu/Lark bot adapter.

    Parameters
    ----------
    config:
        Platform configuration. Requires:
        - ``token``: App ID
        - ``api_key``: App Secret
        - ``webhook_secret``: Verification token for event callbacks
    """

    def __init__(self, config: Any):
        self._config = config
        self._app_id = config.token or os.environ.get("FEISHU_APP_ID", "")
        self._app_secret = config.api_key or os.environ.get("FEISHU_APP_SECRET", "")
        self._verification_token = config.webhook_secret or os.environ.get("FEISHU_VERIFICATION_TOKEN", "")
        self._encrypt_key = (config.extra or {}).get("encrypt_key", "")
        self._timeout = config.timeout or 30
        self._connected = False
        self._session: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._access_token: Optional[str] = None
        self._token_expires: float = 0

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to Feishu API."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        )

        if self._app_id and self._app_secret:
            await self._refresh_access_token()
            logger.info("Feishu adapter connected (app_id=%s...)", self._app_id[:8])
        else:
            logger.info("Feishu adapter in webhook-only mode")

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from Feishu."""
        self._connected = False
        if self._session:
            await self._session.close()
            self._session = None

    async def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a text message to a Feishu user or group."""
        if not self._access_token:
            await self._refresh_access_token()
        if not self._access_token:
            return None

        url = f"{FEISHU_API_BASE}/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        receive_id_type = kwargs.get("receive_id_type", "chat_id")
        msg_type = kwargs.get("msg_type", "text")

        payload: Dict[str, Any] = {
            "receive_id": chat_id,
            "msg_type": msg_type,
            "content": json.dumps({"text": text}) if msg_type == "text" else text,
        }

        params = {"receive_id_type": receive_id_type}

        try:
            async with self._session.post(url, json=payload, headers=headers, params=params) as resp:
                data = await resp.json()
                if data.get("code") == 0:
                    return data.get("data", {}).get("message_id")
                logger.error("Feishu send error: %s", data.get("msg", data))
                return None
        except Exception as e:
            logger.error("Feishu send failed: %s", e)
            return None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a file message to Feishu."""
        if not self._access_token:
            await self._refresh_access_token()
        if not self._access_token or not os.path.exists(file_path):
            return None

        # Upload file first
        upload_url = f"{FEISHU_API_BASE}/im/v1/files"
        headers = {"Authorization": f"Bearer {self._access_token}"}

        file_type = kwargs.get("file_type", "stream")
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        with open(file_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file_type", file_type)
            data.add_field("file_name", file_name)
            data.add_field("file", f)

            async with self._session.post(upload_url, data=data, headers=headers) as resp:
                result = await resp.json()
                if result.get("code") != 0:
                    logger.error("Feishu file upload error: %s", result)
                    return None
                file_key = result.get("data", {}).get("file_key", "")

        # Send file message
        content = json.dumps({"file_key": file_key})
        return await self.send_message(
            chat_id, content,
            msg_type="file",
            receive_id_type=kwargs.get("receive_id_type", "chat_id"),
        )

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for messages from the Feishu event queue."""
        messages: List[IncomingMessage] = []
        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    # ── Extended Interface ────────────────────────────────────────────────

    async def send_typing(self, chat_id: str) -> None:
        """Not supported by Feishu API."""
        pass

    def create_card_message(
        self,
        header_title: str,
        header_color: str = "blue",
        elements: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Create a Feishu card message JSON string."""
        card: Dict[str, Any] = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": header_title},
                "template": header_color,
            },
            "elements": elements or [],
        }
        return json.dumps(card)

    def create_card_div(self, text: str) -> Dict[str, Any]:
        """Create a card div element."""
        return {"tag": "div", "text": {"tag": "lark_md", "content": text}}

    def create_card_button(
        self, text: str, url: str, button_type: str = "primary",
    ) -> Dict[str, Any]:
        """Create a card button element."""
        return {
            "tag": "action",
            "actions": [{
                "tag": "button",
                "text": {"tag": "plain_text", "content": text},
                "url": url,
                "type": button_type,
            }],
        }

    # ── Event Handling ────────────────────────────────────────────────────

    def verify_event(self, timestamp: str, nonce: str, body: str, signature: str) -> bool:
        """Verify a Feishu event callback signature."""
        if not self._encrypt_key:
            return True
        content = f"{timestamp}{nonce}{self._encrypt_key}{body}"
        calculated = hashlib.sha256(content.encode()).hexdigest()
        return hmac.compare_digest(calculated, signature) if hasattr(hmac, 'compare_digest') else calculated == signature

    def handle_event(self, event_data: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Handle an incoming Feishu event callback."""
        # URL verification challenge
        if event_data.get("type") == "url_verification":
            return None

        event = event_data.get("event", {})
        if not event:
            return None

        event_type = event.get("type", "")
        message = event.get("message", {})

        if event_type == "im.message.receive_v1":
            msg_type = message.get("message_type", "text")
            content = json.loads(message.get("content", "{}"))
            text = content.get("text", content.get("content", ""))

            sender = event.get("sender", {})
            sender_id = sender.get("sender_id", {})
            user_id = sender_id.get("user_id", sender_id.get("open_id", ""))

            chat_id = message.get("chat_id", "")
            message_id = message.get("message_id", "")

            return IncomingMessage(
                platform="feishu",
                chat_id=chat_id,
                user_id=user_id,
                text=text,
                message_id=message_id,
                username=sender.get("sender_id", {}).get("name"),
                metadata={
                    "msg_type": msg_type,
                    "chat_type": message.get("chat_type"),
                    "tenant_key": event.get("tenant_key"),
                },
            )

        return None

    # ── Internal ──────────────────────────────────────────────────────────

    async def _refresh_access_token(self) -> None:
        """Refresh the Feishu access token."""
        url = f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self._app_id,
            "app_secret": self._app_secret,
        }

        try:
            async with self._session.post(url, json=payload) as resp:
                data = await resp.json()
                if data.get("code") == 0:
                    token_data = data.get("tenant_access_token", "")
                    self._access_token = token_data
                    self._token_expires = time.time() + data.get("expire", 7200) - 300
                else:
                    logger.error("Feishu token error: %s", data.get("msg"))
        except Exception as e:
            logger.error("Feishu token refresh failed: %s", e)
