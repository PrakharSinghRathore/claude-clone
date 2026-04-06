"""
Generic Webhook adapter for the Atlas Gateway.

Supports custom webhook endpoints, signature verification,
payload parsing, and response formatting.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.webhook import WebhookAdapter

    config = PlatformConfig(
        name="webhook",
        webhook_url="https://example.com/webhook",
        webhook_secret="shared-secret",
        enabled=True,
    )
    adapter = WebhookAdapter(config)
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
from typing import Any, Callable, Dict, List, Optional

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.webhook")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class WebhookAdapter:
    """
    Generic webhook adapter for custom integrations.

    Can both receive webhooks (as a server) and send webhooks (as a client).

    Parameters
    ----------
    config:
        Platform configuration. Useful fields:
        - ``webhook_url``: Outgoing webhook URL
        - ``webhook_secret``: Shared secret for signature verification
        - ``extra``: May contain ``path`` for incoming webhook path
    """

    def __init__(self, config: Any):
        self._config = config
        self._webhook_url = config.webhook_url or ""
        self._webhook_secret = config.webhook_secret or ""
        self._timeout = config.timeout or 30
        self._connected = False
        self._session: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._path = (config.extra or {}).get("path", "/webhook")
        self._payload_parser: Optional[Callable] = None
        self._response_formatter: Optional[Callable] = None
        self._custom_headers: Dict[str, str] = (config.extra or {}).get("headers", {})

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize the webhook adapter."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
            headers=self._custom_headers,
        )
        self._connected = True
        logger.info("Webhook adapter initialized (outgoing: %s)", self._webhook_url or "none")

    async def disconnect(self) -> None:
        """Disconnect the webhook adapter."""
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
        Send a message via outgoing webhook.

        The webhook URL is taken from config. ``chat_id`` can be used
        as a routing key appended to the URL.
        """
        url = self._webhook_url
        if chat_id:
            url = f"{url}/{chat_id}" if not url.endswith("/") else f"{url}{chat_id}"

        payload = {
            "text": text,
            "chat_id": chat_id,
            "timestamp": time.time(),
            **kwargs.get("metadata", {}),
        }

        headers = {"Content-Type": "application/json"}

        # Sign the payload if secret is configured
        if self._webhook_secret:
            body_str = json.dumps(payload, sort_keys=True)
            signature = hmac.new(
                self._webhook_secret.encode(),
                body_str.encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        try:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
                if resp.status < 400:
                    return data.get("id") or data.get("message_id")
                logger.error("Webhook send error: HTTP %d — %s", resp.status, data)
                return None
        except Exception as e:
            logger.error("Webhook send failed: %s", e)
            return None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a file via webhook (as multipart upload or URL reference)."""
        url = self._webhook_url
        if chat_id:
            url = f"{url}/{chat_id}"

        import os
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=os.path.basename(file_path))
                data.add_field("chat_id", chat_id)

                async with self._session.post(url, data=data) as resp:
                    if resp.status < 400:
                        return str(getattr(resp, "id", ""))
                    return None
        except Exception as e:
            logger.error("Webhook file send failed: %s", e)
            return None

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for messages received via incoming webhooks."""
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
        """Not applicable for webhooks."""
        pass

    def set_payload_parser(self, parser: Callable[[Dict[str, Any]], Optional[IncomingMessage]]) -> None:
        """Set a custom payload parser for incoming webhooks."""
        self._payload_parser = parser

    def set_response_formatter(self, formatter: Callable[[str], Dict[str, Any]]) -> None:
        """Set a custom response formatter for outgoing webhooks."""
        self._response_formatter = formatter

    # ── Incoming Webhook ──────────────────────────────────────────────────

    def get_webhook_path(self) -> str:
        """Get the incoming webhook path."""
        return self._path

    def verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify an incoming webhook signature."""
        if not self._webhook_secret:
            return True  # Skip verification if no secret configured

        expected = "sha256=" + hmac.new(
            self._webhook_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def handle_incoming_webhook(
        self,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[IncomingMessage]:
        """
        Handle an incoming webhook payload.

        Uses the custom parser if set, otherwise tries default parsing.
        """
        if self._payload_parser:
            try:
                return self._payload_parser(payload)
            except Exception as e:
                logger.error("Custom payload parser error: %s", e)
                return None

        return self._default_parse(payload)

    def _default_parse(self, payload: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Default payload parser that extracts common fields."""
        text = (
            payload.get("text")
            or payload.get("message")
            or payload.get("body")
            or payload.get("content")
            or ""
        )

        if not text:
            return None

        chat_id = (
            payload.get("chat_id")
            or payload.get("channel")
            or payload.get("conversation")
            or "default"
        )

        user_id = (
            payload.get("user_id")
            or payload.get("from")
            or payload.get("sender")
            or "unknown"
        )

        message_id = payload.get("message_id") or payload.get("id")

        return IncomingMessage(
            platform="webhook",
            chat_id=str(chat_id),
            user_id=str(user_id),
            text=str(text),
            message_id=str(message_id) if message_id else None,
            metadata={"raw_payload": {k: v for k, v in payload.items() if k not in ("text", "message", "body")}},
        )

    def format_response(self, text: str) -> Dict[str, Any]:
        """Format a response for the webhook."""
        if self._response_formatter:
            try:
                return self._response_formatter(text)
            except Exception as e:
                logger.error("Response formatter error: %s", e)

        return {
            "status": "ok",
            "response": text,
            "timestamp": time.time(),
        }
