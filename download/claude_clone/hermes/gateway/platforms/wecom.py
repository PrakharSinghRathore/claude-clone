"""
WeCom (Enterprise WeChat) adapter for the Hermes Gateway.

Supports enterprise WeChat messaging, app message push, and
department/group messaging through the WeCom API.

Usage::

    from hermes.gateway.config import PlatformConfig
    from hermes.gateway.platforms.wecom import WeComAdapter

    config = PlatformConfig(
        name="wecom",
        token="WECOM_CORP_ID",
        api_key="WECOM_CORP_SECRET",
        enabled=True,
        extra={"agent_id": "1000002"},
    )
    adapter = WeComAdapter(config)
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

logger = logging.getLogger("hermes.gateway.platforms.wecom")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

WECOM_API_BASE = "https://qyapi.weixin.qq.com/cgi-bin"


class WeComAdapter:
    """
    WeCom (Enterprise WeChat) adapter.

    Parameters
    ----------
    config:
        Platform configuration. Requires:
        - ``token``: Corp ID
        - ``api_key``: Corp Secret
        - ``webhook_secret``: Callback token for event verification
        - ``extra.agent_id``: Application agent ID
    """

    def __init__(self, config: Any):
        self._config = config
        self._corp_id = config.token or os.environ.get("WECOM_CORP_ID", "")
        self._corp_secret = config.api_key or os.environ.get("WECOM_CORP_SECRET", "")
        self._agent_id = (config.extra or {}).get("agent_id", os.environ.get("WECOM_AGENT_ID", ""))
        self._callback_token = config.webhook_secret or os.environ.get("WECOM_CALLBACK_TOKEN", "")
        self._callback_aes_key = (config.extra or {}).get("callback_aes_key", "")
        self._timeout = config.timeout or 30
        self._connected = False
        self._session: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._access_token: Optional[str] = None
        self._token_expires: float = 0

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to WeCom API."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        )

        if self._corp_id and self._corp_secret:
            await self._refresh_access_token()
            logger.info("WeCom adapter connected (corp_id=%s...)", self._corp_id[:8])
        else:
            logger.info("WeCom adapter in webhook-only mode")

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from WeCom."""
        self._connected = False
        if self._session:
            await self._session.close()
            self._session = None

    async def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a text message via WeCom."""
        if not self._access_token:
            await self._refresh_access_token()
        if not self._access_token:
            return None

        url = f"{WECOM_API_BASE}/message/send"
        params = {"access_token": self._access_token}

        payload: Dict[str, Any] = {
            "touser": chat_id if not kwargs.get("is_group") else "",
            "toparty": kwargs.get("department_id", ""),
            "totag": kwargs.get("tag_id", ""),
            "msgtype": "text",
            "agentid": int(self._agent_id) if self._agent_id else kwargs.get("agent_id", 0),
            "text": {"content": text[:2048]},
            "safe": kwargs.get("safe", 0),
        }

        if kwargs.get("is_group"):
            payload["chatid"] = chat_id
            payload.pop("touser")

        try:
            async with self._session.post(url, json=payload, params=params) as resp:
                data = await resp.json()
                if data.get("errcode") == 0:
                    return data.get("msgid", "")
                logger.error("WeCom send error: %s", data)
                return None
        except Exception as e:
            logger.error("WeCom send failed: %s", e)
            return None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a file message via WeCom."""
        if not self._access_token:
            await self._refresh_access_token()
        if not self._access_token or not os.path.exists(file_path):
            return None

        # Upload media first
        media_id = await self._upload_media(file_path, "file")
        if not media_id:
            return None

        url = f"{WECOM_API_BASE}/message/send"
        params = {"access_token": self._access_token}

        payload: Dict[str, Any] = {
            "touser": chat_id,
            "msgtype": "file",
            "agentid": int(self._agent_id) if self._agent_id else 0,
            "file": {"media_id": media_id},
        }

        try:
            async with self._session.post(url, json=payload, params=params) as resp:
                data = await resp.json()
                if data.get("errcode") == 0:
                    return data.get("msgid", "")
                return None
        except Exception as e:
            logger.error("WeCom file send failed: %s", e)
            return None

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for messages from the WeCom event queue."""
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
        """Not supported by WeCom API."""
        pass

    async def send_markdown(
        self, chat_id: str, content: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a markdown message."""
        if not self._access_token:
            await self._refresh_access_token()
        if not self._access_token:
            return None

        url = f"{WECOM_API_BASE}/message/send"
        params = {"access_token": self._access_token}

        payload: Dict[str, Any] = {
            "touser": chat_id,
            "msgtype": "markdown",
            "agentid": int(self._agent_id) if self._agent_id else 0,
            "markdown": {"content": content},
        }

        try:
            async with self._session.post(url, json=payload, params=params) as resp:
                data = await resp.json()
                if data.get("errcode") == 0:
                    return data.get("msgid", "")
                return None
        except Exception as e:
            logger.error("WeCom markdown send failed: %s", e)
            return None

    async def send_to_department(
        self, department_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a message to an entire department."""
        return await self.send_message(
            chat_id="",
            text=text,
            department_id=department_id,
            **kwargs,
        )

    # ── Event Handling ────────────────────────────────────────────────────

    def verify_event(
        self, msg_signature: str, timestamp: str, nonce: str, echo_str: str,
    ) -> Optional[str]:
        """
        Verify a WeCom callback URL.

        Returns the decrypted echo_str for URL verification, or None.
        """
        if not self._callback_token:
            return echo_str

        token = self._callback_token
        sort_list = sorted([token, timestamp, nonce, echo_str])
        sha = hashlib.sha1("".join(sort_list).encode()).hexdigest()

        if hmac.compare_digest(sha, msg_signature) if hasattr(hmac, 'compare_digest') else sha == msg_signature:
            return echo_str
        return None

    def decrypt_event(self, encrypted_xml: str) -> Optional[Dict[str, Any]]:
        """
        Decrypt an encrypted WeCom event.

        Requires the callback_aes_key to be configured.
        """
        if not self._callback_aes_key:
            logger.warning("No AES key configured — cannot decrypt WeCom events")
            return None

        try:
            from Crypto.Cipher import AES
            import base64

            key = base64.b64decode(self._callback_aes_key + "=")
            iv = key[:16]
            cipher = AES.new(key, AES.MODE_CBC, iv)
            decrypted = cipher.decrypt(base64.b64decode(encrypted_xml))

            # Remove PKCS#7 padding
            pad = decrypted[-1]
            decrypted = decrypted[:-pad]

            # Parse the decrypted content (XML or JSON)
            content = decrypted[16:].decode("utf-8")  # Skip random 16 bytes

            # Try to extract from XML-like format
            if "<![CDATA[" in content:
                import re
                match = re.search(r'<!\[CDATA\[(.+?)\]\]>', content, re.DOTALL)
                if match:
                    return json.loads(match.group(1))

            return json.loads(content) if content.startswith("{") else None

        except ImportError:
            logger.warning("pycryptodome required for WeCom event decryption")
            return None
        except Exception as e:
            logger.error("WeCom event decryption failed: %s", e)
            return None

    def handle_event(self, event_data: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Handle a parsed WeCom event callback."""
        msg_type = event_data.get("MsgType", "")
        content = event_data.get("Content", "")
        from_user = event_data.get("FromUserName", "")
        to_user = event_data.get("ToUserName", "")
        msg_id = event_data.get("MsgId", "")
        create_time = event_data.get("CreateTime", "")

        if not content:
            return None

        return IncomingMessage(
            platform="wecom",
            chat_id=from_user,
            user_id=from_user,
            text=content,
            message_id=str(msg_id),
            metadata={
                "to_user": to_user,
                "msg_type": msg_type,
                "create_time": create_time,
                "agent_id": self._agent_id,
            },
        )

    # ── Internal ──────────────────────────────────────────────────────────

    async def _refresh_access_token(self) -> None:
        """Refresh the WeCom access token."""
        url = f"{WECOM_API_BASE}/gettoken"
        params = {
            "corpid": self._corp_id,
            "corpsecret": self._corp_secret,
        }

        try:
            async with self._session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("errcode") == 0:
                    self._access_token = data.get("access_token", "")
                    self._token_expires = time.time() + data.get("expires_in", 7200) - 300
                else:
                    logger.error("WeCom token error: %s", data.get("errmsg"))
        except Exception as e:
            logger.error("WeCom token refresh failed: %s", e)

    async def _upload_media(self, file_path: str, media_type: str = "file") -> Optional[str]:
        """Upload media to WeCom."""
        url = f"{WECOM_API_BASE}/media/upload"
        params = {"access_token": self._access_token, "type": media_type}

        try:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("media", f, filename=os.path.basename(file_path))

                async with self._session.post(url, data=data, params=params) as resp:
                    result = await resp.json()
                    if result.get("errcode") == 0:
                        return result.get("media_id")
                    logger.error("WeCom upload error: %s", result.get("errmsg"))
                    return None
        except Exception as e:
            logger.error("WeCom media upload failed: %s", e)
            return None
