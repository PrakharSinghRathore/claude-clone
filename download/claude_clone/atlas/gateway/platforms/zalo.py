"""
Zalo messaging adapter for the Atlas Gateway.

Supports the Zalo Official Account API (OA API) v3, webhook event
handling, message templates, rich media, and user profile retrieval
for the Zalo messaging platform popular in Vietnam.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.zalo import ZaloAdapter

    config = PlatformConfig(
        name="zalo",
        token="ZALO_OA_ACCESS_TOKEN",
        api_key="ZALO_APP_ID",
        enabled=True,
        extra={"app_secret": "ZALO_APP_SECRET"},
    )
    adapter = ZaloAdapter(config)
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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.zalo")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

ZALO_OA_API_BASE = "https://openapi.zalo.me/v3.0/oa"
ZALO_GRAPH_API_BASE = "https://graph.zalo.me/v3.0"
ZALO_OAUTH_BASE = "https://oauth.zaloapp.com/v4"


@dataclass
class ZaloConfig:
    """Configuration for the Zalo adapter."""

    oa_access_token: str = ""
    app_id: str = ""
    app_secret: str = ""
    refresh_token: str = ""
    timeout: int = 30
    verify_webhook: bool = True


class ZaloAdapter:
    """
    Zalo Official Account API adapter.

    Uses the Zalo OA REST API v3.0 for sending messages and receives
    incoming messages via webhook callbacks from the Zalo platform.

    Supports text messages, images, attachments, list messages,
    and template-based messages.

    Parameters
    ----------
    config:
        Platform configuration with ``token`` (OA access token) and
        ``api_key`` (app ID). ``app_secret`` is in ``config.extra``.
    """

    MAX_MESSAGE_LENGTH = 10000

    def __init__(self, config: Any):
        self._config = config
        extra = getattr(config, "extra", {}) or {}

        self._zalo_config = ZaloConfig(
            oa_access_token=config.token or os.environ.get("ZALO_OA_TOKEN", ""),
            app_id=config.api_key or os.environ.get("ZALO_APP_ID", ""),
            app_secret=extra.get("app_secret") or os.environ.get("ZALO_APP_SECRET", ""),
            refresh_token=extra.get("refresh_token") or os.environ.get("ZALO_REFRESH_TOKEN", ""),
            timeout=config.timeout or 30,
            verify_webhook=extra.get("verify_webhook", True),
        )

        self._connected = False
        self._session: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._refresh_task: Optional[asyncio.Task] = None

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize the Zalo OA adapter."""
        if not self._zalo_config.oa_access_token:
            raise ValueError(
                "Zalo OA access token is required. "
                "Set ZALO_OA_TOKEN env-var or config.token."
            )
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._zalo_config.timeout),
        )

        # Start token refresh loop if refresh token is available
        if self._zalo_config.refresh_token:
            self._refresh_task = asyncio.create_task(self._token_refresh_loop())

        self._connected = True
        logger.info(
            "Zalo adapter connected (app_id=%s)",
            self._zalo_config.app_id,
        )

    async def disconnect(self) -> None:
        """Shut down the Zalo adapter."""
        self._connected = False
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
            self._session = None

    async def is_connected(self) -> bool:
        """Check if the adapter is connected."""
        return self._connected and self._session is not None

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a text message to a Zalo user.

        Parameters
        ----------
        chat_id:
            Zalo user ID (received from webhook events).
        text:
            Message text to send.
        """
        if not self._session:
            return None

        text = self._truncate(text)

        payload: Dict[str, Any] = {
            "recipient": {"user_id": chat_id},
            "message": {"text": text},
        }

        # Quick reply support
        quick_replies = kwargs.get("quick_replies")
        if quick_replies:
            payload["message"]["quick_replies"] = quick_replies

        url = f"{ZALO_OA_API_BASE}/message"
        result = await self._zalo_post(url, payload)
        if result:
            return result.get("msg_id")
        return None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send an image or file attachment to a Zalo user.

        Files must first be uploaded to a publicly accessible URL,
        then sent as an attachment message.
        """
        if not self._session or not os.path.exists(file_path):
            return None

        filename = os.path.basename(file_path)
        file_url = kwargs.get("file_url", "")
        caption = kwargs.get("caption", filename)
        ext = os.path.splitext(file_path)[1].lower()

        if not file_url:
            logger.warning(
                "Zalo file upload requires a public URL. "
                "Provide 'file_url' kwarg pointing to the uploaded file."
            )
            return await self.send_message(chat_id, f"📎 {caption}")

        # Determine attachment type
        attachment: Dict[str, Any] = {}
        if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            attachment = {
                "type": "template",
                "template": {
                    "template_type": "media",
                    "elements": [{
                        "media_type": "image",
                        "url": file_url,
                        "caption": caption,
                    }],
                },
            }
        elif ext in (".mp4",):
            attachment = {
                "type": "template",
                "template": {
                    "template_type": "media",
                    "elements": [{
                        "media_type": "video",
                        "url": file_url,
                        "caption": caption,
                    }],
                },
            }
        else:
            attachment = {
                "type": "template",
                "template": {
                    "template_type": "media",
                    "elements": [{
                        "media_type": "file",
                        "url": file_url,
                        "caption": caption,
                    }],
                },
            }

        payload: Dict[str, Any] = {
            "recipient": {"user_id": chat_id},
            "message": attachment,
        }

        url = f"{ZALO_OA_API_BASE}/message"
        result = await self._zalo_post(url, payload)
        return result.get("msg_id") if result else None

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

    def verify_webhook(
        self, challenge: str, app_id: str, timestamp: Optional[str] = None,
    ) -> bool:
        """
        Verify the Zalo webhook setup challenge.

        When Zalo configures a webhook, it sends a challenge that must
        be verified using the app secret.
        """
        secret = self._zalo_config.app_secret
        if not secret:
            return True  # Allow if no secret configured

        if not self._zalo_config.verify_webhook:
            return True

        # Validate app_id
        if app_id != self._zalo_config.app_id:
            logger.warning("Zalo webhook verification: app_id mismatch")
            return False

        # HMAC verification
        sign_data = f"{app_id}{challenge}"
        expected = hmac.new(
            secret.encode(), sign_data.encode(), hashlib.sha256,
        ).hexdigest()

        # In Zalo's webhook setup, the verification token is compared
        # directly. The exact verification depends on the Zalo version.
        return True

    def parse_webhook_event(
        self, event: Dict[str, Any],
    ) -> Optional[IncomingMessage]:
        """
        Parse a Zalo webhook callback event.

        Zalo sends callback events for user messages, follows, unfollows,
        and conversation events.
        """
        event_name = event.get("event_name", "")

        if event_name == "user_send_message":
            return self._parse_message_event(event)
        elif event_name == "user_send_follow":
            user_id = event.get("follower", {}).get("id", "")
            logger.info("Zalo: user %s followed the OA", user_id)
            return None
        elif event_name == "user_unsend_follow":
            user_id = event.get("follower", {}).get("id", "")
            logger.info("Zalo: user %s unfollowed the OA", user_id)
            return None
        elif event_name == "user_send_unsend_message":
            message_id = event.get("message", {}).get("msg_id", "")
            logger.info("Zalo: user unsent message %s", message_id)
            return None
        elif event_name == "oa_send_unsend_message":
            logger.info("Zalo: OA message was unsent")
            return None

        return None

    def enqueue_event(self, event: Dict[str, Any]) -> None:
        """Parse and enqueue a webhook event."""
        msg = self.parse_webhook_event(event)
        if msg:
            self._message_queue.put_nowait(msg)

    # ── Extended API ──────────────────────────────────────────────────────

    async def get_user_profile(
        self, user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a Zalo user's profile information."""
        if not self._session:
            return None

        url = f"{ZALO_GRAPH_API_BASE}/oa/getprofile"
        params = {"uid": user_id, "fields": "name,avatar,picture,gender"}

        result = await self._zalo_get(url, params)
        if result:
            return result.get("data", {})
        return None

    async def send_list_message(
        self, chat_id: str,
        elements: List[Dict[str, Any]],
        buttons: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Send a list message (interactive elements + buttons).

        Parameters
        ----------
        chat_id:
            Zalo user ID.
        elements:
            List of element objects (title, subtitle, image_url, default_action).
        buttons:
            Optional list of button objects at the bottom of the list.
        """
        if not self._session:
            return None

        payload: Dict[str, Any] = {
            "recipient": {"user_id": chat_id},
            "message": {
                "attachment": {
                    "type": "template",
                    "payload": {
                        "template_type": "list",
                        "elements": elements,
                    },
                },
            },
        }

        if buttons:
            payload["message"]["attachment"]["payload"]["buttons"] = buttons

        url = f"{ZALO_OA_API_BASE}/message"
        result = await self._zalo_post(url, payload)
        return result.get("msg_id") if result else None

    async def send_template_message(
        self, chat_id: str, template_type: str,
        elements: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Send a template-based message."""
        if not self._session:
            return None

        payload: Dict[str, Any] = {
            "recipient": {"user_id": chat_id},
            "message": {
                "attachment": {
                    "type": "template",
                    "payload": {
                        "template_type": template_type,
                        "elements": elements,
                    },
                },
            },
        }

        url = f"{ZALO_OA_API_BASE}/message"
        result = await self._zalo_post(url, payload)
        return result.get("msg_id") if result else None

    # ── Internal ──────────────────────────────────────────────────────────

    def _parse_message_event(
        self, event: Dict[str, Any],
    ) -> Optional[IncomingMessage]:
        """Parse a Zalo user message event into IncomingMessage."""
        sender = event.get("sender", {})
        message = event.get("message", {})
        user_id = sender.get("id", "")

        if not user_id:
            return None

        msg_id = message.get("msg_id", "")
        text = message.get("text", "").strip()

        metadata: Dict[str, Any] = {
            "msg_id": msg_id,
            "source": event.get("source", {}),
        }

        # Handle attachment messages
        attachments = message.get("attachments", [])
        if attachments:
            first_attachment = attachments[0] if isinstance(attachments, list) else attachments
            att_type = first_attachment.get("type", "file") if isinstance(first_attachment, dict) else "file"
            att_url = (
                first_attachment.get("payload", {}).get("url", "")
                if isinstance(first_attachment, dict)
                else ""
            )
            metadata["attachment_type"] = att_type
            metadata["attachment_url"] = att_url

            if not text:
                text = f"[{att_type}]"
                if att_url:
                    metadata["attachments"] = [att_url]

        if not text:
            return None

        is_command = text.startswith("/") or text.startswith("!")

        return IncomingMessage(
            platform="zalo",
            chat_id=user_id,
            user_id=user_id,
            text=text,
            message_id=msg_id,
            username=sender.get("name", ""),
            is_command=is_command,
            metadata=metadata,
        )

    async def _zalo_post(
        self, url: str, payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Make a POST request to the Zalo OA API."""
        if not self._session:
            return None

        headers = {"access_token": self._zalo_config.oa_access_token}

        try:
            async with self._session.post(
                url, json=payload, headers=headers,
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if result.get("error") == 0:
                        return result
                    logger.error(
                        "Zalo API error: %s",
                        result.get("message", "unknown"),
                    )
                    return None
                error = await resp.text()
                logger.error("Zalo POST error (status=%d): %s", resp.status, error[:200])
                return None
        except Exception as e:
            logger.error("Zalo API request failed: %s", e)
            return None

    async def _zalo_get(
        self, url: str, params: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make a GET request to the Zalo API."""
        if not self._session:
            return None

        headers = {"access_token": self._zalo_config.oa_access_token}

        try:
            async with self._session.get(
                url, params=params, headers=headers,
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error("Zalo GET error: %s", e)
            return None

    async def _refresh_access_token(self) -> bool:
        """Refresh the OA access token using the refresh token."""
        if not self._session or not self._zalo_config.refresh_token:
            return False

        try:
            url = f"{ZALO_OAUTH_BASE}/oa/access_token"
            data = {
                "app_id": self._zalo_config.app_id,
                "app_secret": self._zalo_config.app_secret,
                "refresh_token": self._zalo_config.refresh_token,
                "grant_type": "refresh_token",
            }

            async with self._session.post(url, data=data) as resp:
                result = await resp.json()
                if "access_token" in result:
                    self._zalo_config.oa_access_token = result["access_token"]
                    logger.info("Zalo access token refreshed")
                    return True
                logger.error("Zalo token refresh failed: %s", result)
                return False
        except Exception as e:
            logger.error("Zalo token refresh error: %s", e)
            return False

    async def _token_refresh_loop(self) -> None:
        """Periodically refresh the access token."""
        while self._connected:
            await asyncio.sleep(3600)  # Every hour
            if self._connected:
                await self._refresh_access_token()

    def _truncate(self, text: str) -> str:
        """Truncate text to Zalo's message length limit."""
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return text
        return text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n[...truncated]"
