"""
Telegram Bot API adapter for the Atlas Gateway.

Supports webhook and polling modes, message/photo/document/voice handling,
inline keyboard support, typing indicators, and channel/group support.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.telegram import TelegramAdapter

    config = PlatformConfig(name="telegram", token="BOT_TOKEN", enabled=True)
    adapter = TelegramAdapter(config)
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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.telegram")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class TelegramAdapter:
    """
    Telegram Bot API adapter.

    Implements the common adapter interface for the Telegram messaging
    platform using the Bot API.

    Parameters
    ----------
    config:
        Platform configuration with ``token``, ``api_url``, etc.
    """

    BASE_URL = "https://api.telegram.org/bot"
    FILE_URL = "https://api.telegram.org/file/bot"
    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, config: Any):
        self._config = config
        self._token = config.token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._api_url = config.api_url or f"{self.BASE_URL}{self._token}"
        self._file_url = config.api_url or f"{self.FILE_URL}{self._token}"
        self._timeout = config.timeout or 30
        self._connected = False
        self._offset = 0
        self._webhook_set = False
        self._polling = False
        self._session: Optional[Any] = None
        self._allowed_updates = [
            "message", "edited_message", "channel_post",
            "callback_query", "inline_query",
        ]

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to the Telegram Bot API."""
        if not self._token:
            raise ValueError("Telegram bot token is required")

        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for Telegram adapter. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))

        # Verify bot identity
        try:
            me = await self._api_call("getMe")
            logger.info(
                "Connected as Telegram bot: @%s (id=%s)",
                me.get("username", "unknown"), me.get("id", "unknown"),
            )
            self._connected = True
        except Exception as e:
            await self.disconnect()
            raise RuntimeError(f"Failed to connect to Telegram: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the Telegram Bot API."""
        self._polling = False
        self._connected = False
        if self._session:
            await self._session.close()
            self._session = None

    async def is_connected(self) -> bool:
        """Check if the adapter is connected."""
        if not self._connected or not self._session:
            return False
        try:
            me = await self._api_call("getMe")
            return bool(me.get("ok"))
        except Exception:
            return False

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a text message to a Telegram chat.

        Returns the message ID on success.
        """
        # Truncate if needed
        text = self._truncate(text)

        params: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": kwargs.get("parse_mode", "HTML"),
        }

        if kwargs.get("reply_to"):
            params["reply_to_message_id"] = kwargs["reply_to"]
        if kwargs.get("disable_notification"):
            params["disable_notification"] = True

        # Inline keyboard
        reply_markup = kwargs.get("reply_markup")
        if reply_markup:
            params["reply_markup"] = json.dumps(reply_markup)

        result = await self._api_call("sendMessage", params)
        return str(result.get("message_id")) if result else None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a file/document to a Telegram chat."""
        if not self._session or not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("chat_id", str(chat_id))
                data.add_field(
                    "document", f,
                    filename=os.path.basename(file_path),
                )

                if kwargs.get("reply_to"):
                    data.add_field("reply_to_message_id", str(kwargs["reply_to"]))

                async with self._session.post(
                    f"{self._api_url}/sendDocument", data=data,
                ) as resp:
                    result = await resp.json()
                    if result.get("ok"):
                        return str(result["result"]["message_id"])
                    logger.error("Telegram sendDocument error: %s", result)
                    return None
        except Exception as e:
            logger.error("Failed to send file to Telegram: %s", e)
            return None

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for new messages from Telegram."""
        if not self._connected or not self._session:
            return []

        try:
            params = {
                "offset": self._offset + 1,
                "timeout": 20,
                "allowed_updates": json.dumps(self._allowed_updates),
            }
            result = await self._api_call("getUpdates", params, long_poll=True)

            if not result:
                return []

            messages: List[IncomingMessage] = []

            if isinstance(result, list):
                updates = result
            elif isinstance(result, dict):
                updates = result.get("result", [])
            else:
                return []

            for update in updates:
                update_id = update.get("update_id", 0)
                if update_id > self._offset:
                    self._offset = update_id

                msg = self._parse_update(update)
                if msg:
                    messages.append(msg)

            return messages

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Telegram get_updates error: %s", e)
            await asyncio.sleep(2)
            return []

    # ── Extended Interface ────────────────────────────────────────────────

    async def edit_message(
        self, chat_id: str, message_id: str, text: str, **kwargs: Any,
    ) -> bool:
        """Edit an existing message."""
        text = self._truncate(text)
        params: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": kwargs.get("parse_mode", "HTML"),
        }
        result = await self._api_call("editMessageText", params)
        return bool(result and result.get("ok"))

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message."""
        params = {"chat_id": chat_id, "message_id": message_id}
        result = await self._api_call("deleteMessage", params)
        return bool(result and result.get("ok"))

    async def send_typing(self, chat_id: str) -> None:
        """Send a typing action indicator."""
        params = {"chat_id": chat_id, "action": "typing"}
        await self._api_call("sendChatAction", params)

    async def send_photo(
        self, chat_id: str, file_path: str, caption: str = "", **kwargs: Any,
    ) -> Optional[str]:
        """Send a photo to a chat."""
        if not self._session or not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("chat_id", str(chat_id))
                data.add_field("photo", f, filename=os.path.basename(file_path))
                if caption:
                    data.add_field("caption", caption[:1024])
                if kwargs.get("reply_to"):
                    data.add_field("reply_to_message_id", str(kwargs["reply_to"]))

                async with self._session.post(
                    f"{self._api_url}/sendPhoto", data=data,
                ) as resp:
                    result = await resp.json()
                    if result.get("ok"):
                        return str(result["result"]["message_id"])
                    return None
        except Exception as e:
            logger.error("Failed to send photo: %s", e)
            return None

    async def answer_callback_query(
        self, callback_query_id: str, text: str = "",
    ) -> bool:
        """Answer a callback query from inline keyboards."""
        params = {"callback_query_id": callback_query_id, "text": text}
        result = await self._api_call("answerCallbackQuery", params)
        return bool(result and result.get("ok"))

    async def get_chat_info(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a chat."""
        return await self._api_call("getChat", {"chat_id": chat_id})

    # ── Webhook Support ───────────────────────────────────────────────────

    async def set_webhook(
        self, url: str, secret: Optional[str] = None,
    ) -> bool:
        """Set a webhook URL for receiving updates."""
        params: Dict[str, Any] = {"url": url}
        if secret:
            params["secret_token"] = secret
        result = await self._api_call("setWebhook", params)
        self._webhook_set = bool(result and result.get("ok"))
        return self._webhook_set

    async def delete_webhook(self) -> bool:
        """Remove the webhook and switch back to polling."""
        result = await self._api_call("deleteWebhook", {"drop_pending_updates": True})
        self._webhook_set = False
        return bool(result and result.get("ok"))

    def verify_webhook_signature(self, data: bytes, signature: str) -> bool:
        """Verify the webhook signature."""
        secret = getattr(self._config, "webhook_secret", "")
        if not secret:
            return False
        expected = hmac.new(
            secret.encode(), data, hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def parse_webhook_update(self, data: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Parse a webhook update into an IncomingMessage."""
        return self._parse_update(data)

    # ── Internal ──────────────────────────────────────────────────────────

    async def _api_call(
        self, method: str, params: Optional[Dict[str, Any]] = None,
        long_poll: bool = False,
    ) -> Any:
        """Make an API call to the Telegram Bot API."""
        if not self._session:
            return None

        timeout = aiohttp.ClientTimeout(total=30 if long_poll else self._timeout)
        url = f"{self._api_url}/{method}"

        try:
            async with self._session.post(url, json=params, timeout=timeout) as resp:
                result = await resp.json()
                if result.get("ok"):
                    return result.get("result")
                else:
                    description = result.get("description", "Unknown error")
                    logger.error("Telegram API error (%s): %s", method, description)
                    return None
        except asyncio.TimeoutError:
            logger.warning("Telegram API timeout: %s", method)
            return None
        except Exception as e:
            logger.error("Telegram API call failed (%s): %s", method, e)
            return None

    def _parse_update(self, update: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Parse a Telegram update into an IncomingMessage."""
        message = (
            update.get("message")
            or update.get("edited_message")
            or update.get("channel_post")
            or update.get("edited_channel_post")
        )

        if not message:
            # Handle callback queries
            callback = update.get("callback_query")
            if callback:
                msg = callback.get("message", {})
                return IncomingMessage(
                    platform="telegram",
                    chat_id=str(msg.get("chat_id", "")),
                    user_id=str(callback.get("from", {}).get("id", "")),
                    text=callback.get("data", ""),
                    message_id=str(msg.get("message_id", "")),
                    username=callback.get("from", {}).get("username"),
                    metadata={"callback_query_id": callback.get("id")},
                    is_command=True,
                )
            return None

        chat = message.get("chat", {})
        from_user = message.get("from", {})

        chat_id = str(chat.get("id", ""))
        user_id = str(from_user.get("id", ""))
        text = message.get("text", "") or message.get("caption", "")

        # Handle attachments
        attachments: List[str] = []
        metadata: Dict[str, Any] = {}

        if message.get("photo"):
            photo = message["photo"][-1]  # Largest size
            file_id = photo.get("file_id", "")
            metadata["photo_file_id"] = file_id

        if message.get("document"):
            doc = message["document"]
            metadata["document_file_id"] = doc.get("file_id", "")
            metadata["document_filename"] = doc.get("file_name", "")

        if message.get("voice"):
            metadata["voice_file_id"] = message["voice"].get("file_id", "")

        if message.get("video"):
            metadata["video_file_id"] = message["video"].get("file_id", "")

        reply_to_msg = message.get("reply_to_message")
        reply_to = str(reply_to_msg.get("message_id", "")) if reply_to_msg else None

        is_edit = "edited_message" in update or "edited_channel_post" in update
        is_command = text.startswith("/") if text else False

        return IncomingMessage(
            platform="telegram",
            chat_id=chat_id,
            user_id=user_id,
            text=text,
            message_id=str(message.get("message_id", "")),
            username=from_user.get("username"),
            reply_to=reply_to,
            attachments=attachments if attachments else None,
            is_command=is_command,
            is_edit=is_edit,
            metadata=metadata,
        )

    def _truncate(self, text: str) -> str:
        """Truncate text to Telegram's message length limit."""
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return text
        return text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n[...truncated]"
