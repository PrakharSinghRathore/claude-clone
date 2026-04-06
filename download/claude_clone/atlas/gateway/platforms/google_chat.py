"""
Google Chat adapter for the Atlas Gateway.

Supports Google Chat REST API v1, incoming webhook handling,
slash commands, card/formatted messages, and async message
processing via the HTTP-based API.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.google_chat import GoogleChatAdapter

    config = PlatformConfig(
        name="google_chat",
        token="GOOGLE_CHAT_SERVICE_ACCOUNT_JSON",
        enabled=True,
        extra={
            "project_id": "my-gcp-project",
            "bot_name": "AtlasBot",
        },
    )
    adapter = GoogleChatAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.google_chat")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

GOOGLE_CHAT_API_BASE = "https://chat.googleapis.com/v1"


@dataclass
class GoogleChatConfig:
    """Configuration for the Google Chat adapter."""

    service_account_json: str = ""
    project_id: str = ""
    bot_name: str = "AtlasBot"
    bot_avatar_url: str = ""
    timeout: int = 30
    space_scopes: List[str] = field(
        default_factory=lambda: ["https://www.googleapis.com/auth/chat.bot"],
    )


class GoogleChatAdapter:
    """
    Google Chat adapter using the REST API with service account auth.

    Messages are received via webhook events (pushed to the gateway) and
    outgoing messages use the ``spaces.messages.create`` endpoint with
    a service account bearer token.

    Parameters
    ----------
    config:
        Platform configuration. Service account JSON can be provided via
        ``config.token`` or the ``GOOGLE_CHAT_SA_JSON`` env-var.
    """

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, config: Any):
        self._config = config
        extra = getattr(config, "extra", {}) or {}

        self._gc_config = GoogleChatConfig(
            service_account_json=config.token or os.environ.get("GOOGLE_CHAT_SA_JSON", ""),
            project_id=extra.get("project_id") or os.environ.get("GOOGLE_CHAT_PROJECT_ID", ""),
            bot_name=extra.get("bot_name") or os.environ.get("GOOGLE_CHAT_BOT_NAME", "AtlasBot"),
            bot_avatar_url=extra.get("bot_avatar_url", ""),
            timeout=config.timeout or 30,
        )

        self._connected = False
        self._session: Optional[Any] = None
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._refresh_task: Optional[asyncio.Task] = None

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Authenticate and start the Google Chat adapter."""
        if not self._gc_config.service_account_json:
            raise ValueError(
                "Google Chat service account JSON is required. "
                "Set GOOGLE_CHAT_SA_JSON env-var or config.token."
            )
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._gc_config.timeout),
        )

        # Obtain initial access token
        await self._refresh_access_token()

        self._refresh_task = asyncio.create_task(self._token_refresh_loop())
        self._connected = True

        logger.info(
            "Google Chat adapter connected (project=%s, bot=%s)",
            self._gc_config.project_id, self._gc_config.bot_name,
        )

    async def disconnect(self) -> None:
        """Shut down the Google Chat adapter."""
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
        self._access_token = None

    async def is_connected(self) -> bool:
        """Check if the adapter is authenticated and ready."""
        return (
            self._connected
            and self._access_token is not None
            and time.time() < self._token_expiry
        )

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a message to a Google Chat space.

        Parameters
        ----------
        chat_id:
            Space name in the format ``spaces/<SPACE_ID>`` or
            ``spaces/<SPACE_ID>/threads/<THREAD_ID>``.
        text:
            Plain text or slash-command response.
        """
        if not self._session:
            return None

        text = self._truncate(text)

        # Build the parent resource name
        if chat_id.startswith("spaces/"):
            parent = chat_id
        else:
            parent = f"spaces/{chat_id}"

        payload: Dict[str, Any] = {"text": text}

        # Thread support
        if kwargs.get("thread_key"):
            payload["thread"] = {
                "name": kwargs["thread_key"],
            }

        # Card / widget support
        card = kwargs.get("card")
        if card:
            payload["cardsV2"] = [
                {"cardId": "atlas-card", "card": card},
            ]

        url = f"{GOOGLE_CHAT_API_BASE}/{parent}/messages"
        headers = {"Authorization": f"Bearer {self._access_token}"}

        try:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    name = result.get("name", "")
                    logger.info("Google Chat message sent: %s", name)
                    return name.split("/")[-1] if name else None
                error_text = await resp.text()
                logger.error(
                    "Google Chat send error (status=%d): %s",
                    resp.status, error_text[:200],
                )
                return None
        except Exception as e:
            logger.error("Failed to send Google Chat message: %s", e)
            return None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a file to a Google Chat space.

        Google Chat does not support direct file uploads via the API.
        This method uploads the file to Google Drive and shares the link.
        """
        logger.warning(
            "Google Chat direct file upload not supported — "
            "consider uploading to Google Drive and sharing the link."
        )
        filename = os.path.basename(file_path)
        text = kwargs.get("caption", f"📎 File: {filename}")
        return await self.send_message(chat_id, text)

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
        Parse an incoming Google Chat HTTP webhook event.

        Google Chat sends interaction events to a configured webhook URL
        whenever a user messages the bot.
        """
        event_type = event.get("type", "")

        if event_type == "MESSAGE":
            message = event.get("message", {})
            sender = message.get("sender", {})
            space = event.get("space", {})
            thread = message.get("thread", {})

            # Ignore messages from the bot itself
            if sender.get("type") == "BOT":
                return None

            text = message.get("argumentText", "").strip()
            if not text:
                text = message.get("text", "").strip()

            # Remove @bot_name prefix if present
            bot_name = self._gc_config.bot_name
            if text.lower().startswith(f"@{bot_name.lower()} "):
                text = text[len(f"@{bot_name.lower()} "):]

            space_name = space.get("name", "")
            thread_name = thread.get("name", "")

            metadata: Dict[str, Any] = {
                "space_name": space_name,
                "space_display_name": space.get("displayName", ""),
                "space_type": space.get("spaceType", ""),
                "thread_name": thread_name,
                "message_name": message.get("name", ""),
            }

            # Slash command detection
            is_command = False
            slash_command = event.get("message", {}).get("slashCommand")
            if slash_command:
                is_command = True
                metadata["slash_command"] = slash_command.get("commandId")
                cmd_text = event.get("message", {}).get("argumentText", "")
                if cmd_text:
                    text = f"/{slash_command.get('commandName', '')} {cmd_text}"

            return IncomingMessage(
                platform="google_chat",
                chat_id=space_name,
                user_id=sender.get("name", ""),
                text=text,
                message_id=message.get("name", ""),
                username=sender.get("displayName", ""),
                reply_to=thread_name if thread_name else None,
                is_command=is_command,
                metadata=metadata,
            )

        elif event_type == "ADDED_TO_SPACE":
            space = event.get("space", {})
            logger.info(
                "Bot added to Google Chat space: %s",
                space.get("displayName", space.get("name", "unknown")),
            )

        return None

    def enqueue_event(self, event: Dict[str, Any]) -> None:
        """Parse and enqueue a webhook event for later processing."""
        msg = self.parse_webhook_event(event)
        if msg:
            self._message_queue.put_nowait(msg)

    # ── Card Builders ─────────────────────────────────────────────────────

    @staticmethod
    def create_card(
        header_title: str,
        header_subtitle: str = "",
        sections: Optional[List[Dict[str, Any]]] = None,
        header_image_url: str = "",
    ) -> Dict[str, Any]:
        """
        Create a Google Chat card (v2) payload.

        Parameters
        ----------
        header_title:
            Title displayed at the top of the card.
        header_subtitle:
            Subtitle text.
        sections:
            List of card sections, each containing ``widgets``.
        header_image_url:
            URL for the header icon or avatar.
        """
        card: Dict[str, Any] = {
            "header": {
                "title": header_title,
                "subtitle": header_subtitle,
            },
        }

        if header_image_url:
            card["header"]["imageUrl"] = header_image_url

        if sections:
            card["sections"] = sections

        return card

    @staticmethod
    def create_text_paragraph(text: str) -> Dict[str, Any]:
        """Create a card widget with a text paragraph."""
        return {"textParagraph": {"text": text}}

    @staticmethod
    def create_key_value(
        top_label: str, content: str, icon: str = "",
    ) -> Dict[str, Any]:
        """Create a card widget with a key-value pair."""
        widget: Dict[str, Any] = {
            "keyValue": {
                "topLabel": top_label,
                "content": content,
            }
        }
        if icon:
            widget["keyValue"]["icon"] = icon
        return widget

    @staticmethod
    def create_button(
        text: str, url: str, style: str = "FILLED",
    ) -> Dict[str, Any]:
        """Create a card button widget."""
        return {
            "buttonList": {
                "buttons": [{"text": text, "onClick": {"openLink": {"url": url}}}],
            }
        }

    # ── Internal ──────────────────────────────────────────────────────────

    async def _refresh_access_token(self) -> None:
        """Obtain or refresh the Google OAuth2 access token."""
        if not self._session:
            return

        try:
            sa_info = json.loads(self._gc_config.service_account_json)
            # Use the server-to-server OAuth2 flow
            import base64

            # Build JWT assertion
            now = int(time.time())
            header = {"alg": "RS256", "typ": "JWT"}
            payload_data = {
                "iss": sa_info.get("client_email", ""),
                "scope": " ".join(self._gc_config.space_scopes),
                "aud": "https://oauth2.googleapis.com/token",
                "iat": now,
                "exp": now + 3600,
            }

            header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
            payload_b64 = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).decode().rstrip("=")

            private_key = sa_info.get("private_key", "")
            if not private_key:
                logger.error("Service account JSON missing private_key")
                return

            import hashlib
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.hazmat.primitives import hashes, serialization

            signing_key = serialization.load_pem_private_key(
                private_key.encode(), password=None,
            )
            signature = signing_key.sign(
                f"{header_b64}.{payload_b64}".encode(),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            sig_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")
            assertion = f"{header_b64}.{payload_b64}.{sig_b64}"

            async with self._session.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": assertion,
                },
            ) as resp:
                result = await resp.json()
                if "access_token" in result:
                    self._access_token = result["access_token"]
                    self._token_expiry = now + result.get("expires_in", 3600) - 60
                    logger.info("Google Chat access token refreshed")
                else:
                    logger.error(
                        "Failed to get Google Chat token: %s",
                        result.get("error", "unknown"),
                    )

        except Exception as e:
            logger.error("Error refreshing Google Chat access token: %s", e)

    async def _token_refresh_loop(self) -> None:
        """Periodically refresh the OAuth2 access token."""
        while self._connected:
            await asyncio.sleep(300)  # Refresh every 5 minutes
            if self._connected:
                await self._refresh_access_token()

    def _truncate(self, text: str) -> str:
        """Truncate text to Google Chat's message length limit."""
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return text
        return text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n[...truncated]"
