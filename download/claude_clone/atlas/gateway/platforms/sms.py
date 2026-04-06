"""
SMS adapter for the Atlas Gateway (via Twilio).

Supports send/receive SMS, MMS support, number management,
and delivery status tracking.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.sms import SMSAdapter

    config = PlatformConfig(
        name="sms",
        token="TWILIO_ACCOUNT_SID",
        api_key="TWILIO_AUTH_TOKEN",
        api_url="+1234567890",  # Twilio phone number
        enabled=True,
    )
    adapter = SMSAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import logging
import os
import urllib.parse
from typing import Any, Dict, List, Optional

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.sms")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"


class SMSAdapter:
    """
    Twilio SMS adapter.

    Uses the Twilio REST API for sending and webhook for receiving SMS/MMS.

    Parameters
    ----------
    config:
        Platform configuration. Requires:
        - ``token``: Twilio Account SID
        - ``api_key``: Twilio Auth Token
        - ``api_url``: Twilio phone number (sender)
    """

    MAX_SMS_LENGTH = 1600

    def __init__(self, config: Any):
        self._config = config
        self._account_sid = config.token or os.environ.get("TWILIO_ACCOUNT_SID", "")
        self._auth_token = config.api_key or os.environ.get("TWILIO_AUTH_TOKEN", "")
        self._from_number = config.api_url or os.environ.get("TWILIO_PHONE_NUMBER", "")
        self._timeout = config.timeout or 30
        self._connected = False
        self._session: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._delivery_status: Dict[str, str] = {}

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize the Twilio SMS adapter."""
        if not self._account_sid or not self._auth_token:
            raise ValueError("Twilio Account SID and Auth Token are required")
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
            auth=aiohttp.BasicAuth(self._account_sid, self._auth_token),
        )

        # Verify credentials
        try:
            async with self._session.get(
                f"https://api.twilio.com/2010-04-01/Accounts/{self._account_sid}.json"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info("SMS adapter connected (account: %s)", self._account_sid)
                else:
                    logger.warning("Twilio credential verification returned %d", resp.status)
        except Exception as e:
            logger.warning("Twilio verification error: %s", e)

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect the Twilio adapter."""
        self._connected = False
        if self._session:
            await self._session.close()
            self._session = None

    async def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send an SMS message.

        ``chat_id`` is the recipient phone number in E.164 format.
        """
        url = (
            f"{TWILIO_API_BASE}/Accounts/{self._account_sid}"
            f"/Messages.json"
        )
        payload = {
            "From": self._from_number,
            "To": chat_id,
            "Body": text[:self.MAX_SMS_LENGTH],
        }

        if kwargs.get("media_url"):
            payload["MediaUrl"] = kwargs["media_url"]

        try:
            async with self._session.post(url, data=payload) as resp:
                data = await resp.json()
                if resp.status in (200, 201):
                    sid = data.get("sid", "")
                    self._delivery_status[sid] = "queued"
                    return sid
                logger.error("Twilio SMS error: %s", data)
                return None
        except Exception as e:
            logger.error("Twilio send failed: %s", e)
            return None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send an MMS message with media."""
        # For MMS, we need a publicly accessible URL
        # In production, upload to a storage service first
        media_url = kwargs.get("media_url")
        if not media_url:
            logger.warning("SMS file sending requires a publicly accessible media_url")
            return None

        return await self.send_message(chat_id, "", media_url=media_url, **kwargs)

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for new SMS messages from the webhook queue."""
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
        """Not applicable for SMS."""
        pass

    async def get_delivery_status(self, message_sid: str) -> Optional[str]:
        """Get the delivery status of a sent message."""
        return self._delivery_status.get(message_sid)

    def update_delivery_status(self, message_sid: str, status: str) -> None:
        """Update delivery status from a webhook callback."""
        self._delivery_status[message_sid] = status

    async def lookup_number(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Look up information about a phone number."""
        url = (
            f"https://lookups.twilio.com/v1/PhoneNumbers/"
            f"{urllib.parse.quote(phone_number)}"
        )
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error("Twilio lookup failed: %s", e)
            return None

    # ── Webhook Handling ──────────────────────────────────────────────────

    def handle_webhook_sms(self, form_data: Dict[str, str]) -> Optional[IncomingMessage]:
        """Handle an incoming SMS webhook from Twilio."""
        body = form_data.get("Body", "")
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        message_sid = form_data.get("MessageSid", "")
        num_media = int(form_data.get("NumMedia", "0"))

        metadata: Dict[str, Any] = {
            "from": from_number,
            "to": to_number,
            "num_media": num_media,
        }

        media_urls = []
        for i in range(num_media):
            media_url = form_data.get(f"MediaUrl{i}", "")
            if media_url:
                media_urls.append(media_url)
        if media_urls:
            metadata["media_urls"] = media_urls

        return IncomingMessage(
            platform="sms",
            chat_id=from_number,
            user_id=from_number,
            text=body,
            message_id=message_sid,
            metadata=metadata,
        )

    def handle_webhook_status(self, form_data: Dict[str, str]) -> None:
        """Handle a delivery status webhook from Twilio."""
        message_sid = form_data.get("MessageSid", "")
        status = form_data.get("MessageStatus", "")
        if message_sid and status:
            self._delivery_status[message_sid] = status

    def validate_webhook_signature(
        self,
        url: str,
        params: Dict[str, str],
        signature: str,
    ) -> bool:
        """Validate the Twilio webhook signature."""
        try:
            from twilio.request_validator import RequestValidator
            validator = RequestValidator(self._auth_token)
            return validator.validate(url, params, signature)
        except ImportError:
            logger.warning("twilio package not installed — webhook validation skipped")
            return True
