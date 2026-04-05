"""
DingTalk adapter for the Hermes Gateway.

Supports robot message sending, interactive cards, and group messaging
through the DingTalk Open Platform API.

Usage::

    from hermes.gateway.config import PlatformConfig
    from hermes.gateway.platforms.dingtalk import DingTalkAdapter

    config = PlatformConfig(
        name="dingtalk",
        token="DINGTALK_APP_KEY",
        api_key="DINGTALK_APP_SECRET",
        enabled=True,
    )
    adapter = DingTalkAdapter(config)
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

from hermes.gateway.runner import IncomingMessage

logger = logging.getLogger("hermes.gateway.platforms.dingtalk")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

DINGTALK_API_BASE = "https://oapi.dingtalk.com"
DINGTALK_NEW_API_BASE = "https://api.dingtalk.com"


class DingTalkAdapter:
    """
    DingTalk robot adapter using the Open Platform API.

    Parameters
    ----------
    config:
        Platform configuration. Requires:
        - ``token``: App Key or Robot Webhook access token
        - ``api_key``: App Secret
        - ``webhook_secret``: Robot webhook secret (for signature verification)
    """

    def __init__(self, config: Any):
        self._config = config
        self._app_key = config.token or os.environ.get("DINGTALK_APP_KEY", "")
        self._app_secret = config.api_key or os.environ.get("DINGTALK_APP_SECRET", "")
        self._webhook_secret = config.webhook_secret or os.environ.get("DINGTALK_WEBHOOK_SECRET", "")
        self._robot_code = (config.extra or {}).get("robot_code", "")
        self._timeout = config.timeout or 30
        self._connected = False
        self._session: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._access_token: Optional[str] = None
        self._token_expires: float = 0

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to DingTalk API."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        )

        # Get access token if app credentials are provided
        if self._app_key and self._app_secret:
            await self._refresh_access_token()
            logger.info("DingTalk adapter connected (app_key=%s...)", self._app_key[:8])
        else:
            logger.info("DingTalk adapter in webhook-only mode")

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from DingTalk."""
        self._connected = False
        if self._session:
            await self._session.close()
            self._session = None

    async def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a text message via DingTalk robot."""
        if self._robot_code and self._access_token:
            return await self._send_via_robot(chat_id, text, **kwargs)
        else:
            # Fallback to webhook mode
            return await self._send_via_webhook(text, **kwargs)

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a file via DingTalk robot."""
        if not self._access_token or not self._robot_code:
            logger.warning("File sending requires robot code and access token")
            return None

        # Upload media first
        media_id = await self._upload_media(chat_id, file_path)
        if not media_id:
            return None

        return await self._send_via_robot(chat_id, "", media_id=media_id, **kwargs)

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for messages from the DingTalk callback queue."""
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
        """Not supported by DingTalk API."""
        pass

    async def send_action_card(
        self,
        chat_id: str,
        title: str,
        text: str,
        single_title: str = "",
        single_url: str = "",
        **kwargs: Any,
    ) -> Optional[str]:
        """Send an action card message."""
        payload: Dict[str, Any] = {
            "msgtype": "actionCard",
            "actionCard": {
                "title": title,
                "text": text,
                "singleTitle": single_title,
                "singleURL": single_url,
            },
        }
        if kwargs.get("btn_orientation"):
            payload["actionCard"]["btnOrientation"] = kwargs["btn_orientation"]

        return await self._send_via_robot(chat_id, "", extra=payload)

    def create_card_buttons(
        self,
        title: str,
        action_url: str,
    ) -> Dict[str, str]:
        """Create a button object for card messages."""
        return {"title": title, "actionURL": action_url}

    # ── Webhook Handling ──────────────────────────────────────────────────

    def verify_callback(
        self,
        timestamp: str,
        sign: str,
        body: str,
    ) -> bool:
        """Verify a DingTalk callback signature."""
        if not self._webhook_secret:
            return True
        string_to_sign = f"{timestamp}\n{self._webhook_secret}"
        hmac_code = hmac.new(
            self._webhook_secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        expected = hmac_code.hex()
        return hmac.compare_digest(expected.upper(), sign.upper())

    def handle_callback(self, data: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Handle an incoming DingTalk callback event."""
        msg_type = data.get("msgtype", "")
        text_content = ""
        sender_id = ""
        conversation_id = ""

        if "text" in data:
            text_content = data["text"].get("content", "").strip()
        elif "content" in data:
            text_content = data["content"]

        sender_staff_id = data.get("senderStaffId", "")
        sender_nick = data.get("senderNick", "")
        conversation_id = data.get("conversationId", "")
        sender_id = data.get("senderId", "") or sender_staff_id
        msg_id = data.get("msgId", "") or data.get("messageId", "")

        if not text_content:
            return None

        return IncomingMessage(
            platform="dingtalk",
            chat_id=conversation_id or "default",
            user_id=sender_id,
            text=text_content,
            message_id=str(msg_id),
            username=sender_nick,
            metadata={
                "msgtype": msg_type,
                "conversation_type": data.get("conversationType"),
                "is_admin": data.get("isAdmin", False),
                "chatbot_user_id": data.get("chatbotUserId"),
            },
        )

    # ── Internal ──────────────────────────────────────────────────────────

    async def _refresh_access_token(self) -> None:
        """Refresh the DingTalk access token."""
        url = f"{DINGTALK_API_BASE}/gettoken"
        params = {
            "appkey": self._app_key,
            "appsecret": self._app_secret,
        }

        try:
            async with self._session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("errcode") == 0:
                    self._access_token = data.get("access_token", "")
                    self._token_expires = time.time() + data.get("expires_in", 7200) - 300
                else:
                    logger.error("DingTalk token error: %s", data)
        except Exception as e:
            logger.error("DingTalk token refresh failed: %s", e)

    def _ensure_token(self) -> bool:
        """Ensure we have a valid access token."""
        if not self._access_token or time.time() > self._token_expires:
            return False
        return True

    async def _send_via_robot(
        self,
        chat_id: str,
        text: str,
        media_id: Optional[str] = None,
        extra: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Send a message via the DingTalk robot API."""
        if not self._ensure_token():
            await self._refresh_access_token()

        if not self._access_token:
            return None

        headers = {"x-acs-dingtalk-access-token": self._access_token}
        url = f"{DINGTALK_NEW_API_BASE}/v1.0/robot/oToMessages/batchSend"

        if extra:
            payload = extra
        else:
            payload: Dict[str, Any] = {
                "robotCode": self._robot_code,
                "userIds": [chat_id],
                "msgKey": "sampleText" if text else "sampleFile",
                "msgParam": json.dumps({"content": text}) if text else json.dumps({"mediaId": media_id}),
            }

        try:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json()
                if resp.status == 200:
                    return str(data.get("processQueryKey", ""))
                logger.error("DingTalk send error: %s", data)
                return None
        except Exception as e:
            logger.error("DingTalk send failed: %s", e)
            return None

    async def _send_via_webhook(self, text: str, **kwargs: Any) -> Optional[str]:
        """Send a message via DingTalk webhook (simpler mode)."""
        webhook_url = self._config.webhook_url or os.environ.get("DINGTALK_WEBHOOK_URL", "")
        if not webhook_url:
            logger.warning("No DingTalk webhook URL configured")
            return None

        payload: Dict[str, Any] = {
            "msgtype": "text",
            "text": {"content": text},
        }

        # Sign if secret is set
        if self._webhook_secret:
            timestamp = str(int(time.time() * 1000))
            string_to_sign = f"{timestamp}\n{self._webhook_secret}"
            hmac_code = hmac.new(
                self._webhook_secret.encode("utf-8"),
                string_to_sign.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            webhook_url = f"{webhook_url}&timestamp={timestamp}&sign={hmac_code}"

        try:
            async with self._session.post(webhook_url, json=payload) as resp:
                data = await resp.json()
                if data.get("errcode") == 0:
                    return str(data.get("messageId", ""))
                logger.error("DingTalk webhook error: %s", data)
                return None
        except Exception as e:
            logger.error("DingTalk webhook failed: %s", e)
            return None

    async def _upload_media(
        self, chat_id: str, file_path: str,
    ) -> Optional[str]:
        """Upload media to DingTalk."""
        if not self._access_token or not self._robot_code:
            return None

        headers = {"x-acs-dingtalk-access-token": self._access_token}
        url = f"{DINGTALK_NEW_API_BASE}/v1.0/robot/asset/upload"

        try:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("asset", f, filename=os.path.basename(file_path))
                data.add_field("robotCode", self._robot_code)

                async with self._session.post(url, data=data, headers=headers) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        return result.get("assetId") or result.get("mediaId")
                    logger.error("DingTalk upload error: %s", result)
                    return None
        except Exception as e:
            logger.error("DingTalk media upload failed: %s", e)
            return None
