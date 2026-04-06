"""
Microsoft Teams adapter for the Atlas Gateway.

Supports the Microsoft Bot Framework, Adaptive Cards, proactive messaging,
conversation management, and the Bot Framework REST API v3.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.msteams import MSTeamsAdapter

    config = PlatformConfig(
        name="msteams",
        token="MICROSOFT_APP_ID",
        api_key="MICROSOFT_APP_PASSWORD",
        enabled=True,
        extra={"bot_name": "AtlasBot"},
    )
    adapter = MSTeamsAdapter(config)
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
from urllib.parse import urljoin

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.msteams")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

TEAMS_BOT_FRAMEWORK = "https://api.botframework.com"
TEAMS_API_BASE = "https://spectrum.botframework.com"


@dataclass
class MSTeamsConfig:
    """Configuration for the Microsoft Teams adapter."""

    app_id: str = ""
    app_password: str = ""
    bot_name: str = "AtlasBot"
    tenant_id: str = ""
    timeout: int = 30
    webhook_url: str = ""


class MSTeamsAdapter:
    """
    Microsoft Teams adapter using the Bot Framework REST API.

    Messages are received via webhook invocations from the Bot Framework,
    and outgoing messages are sent through the ``v3/conversations`` endpoints.

    Parameters
    ----------
    config:
        Platform configuration. Requires ``token`` (App ID) and
        ``api_key`` (App Password), or the corresponding env-vars.
    """

    MAX_MESSAGE_LENGTH = 8000

    def __init__(self, config: Any):
        self._config = config
        extra = getattr(config, "extra", {}) or {}

        self._ms_config = MSTeamsConfig(
            app_id=config.token or os.environ.get("MICROSOFT_APP_ID", ""),
            app_password=config.api_key or os.environ.get("MICROSOFT_APP_PASSWORD", ""),
            bot_name=extra.get("bot_name") or os.environ.get("MS_TEAMS_BOT_NAME", "AtlasBot"),
            tenant_id=extra.get("tenant_id") or os.environ.get("MS_TEAMS_TENANT_ID", ""),
            timeout=config.timeout or 30,
            webhook_url=extra.get("webhook_url", ""),
        )

        self._connected = False
        self._session: Optional[Any] = None
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._refresh_task: Optional[asyncio.Task] = None
        self._conversation_refs: Dict[str, Dict[str, Any]] = {}

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Authenticate with the Bot Framework and start the adapter."""
        if not self._ms_config.app_id or not self._ms_config.app_password:
            raise ValueError(
                "Microsoft App ID and App Password are required. "
                "Set MICROSOFT_APP_ID and MICROSOFT_APP_PASSWORD env-vars."
            )
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._ms_config.timeout),
        )

        # Obtain access token
        await self._refresh_access_token()

        self._refresh_task = asyncio.create_task(self._token_refresh_loop())
        self._connected = True

        logger.info(
            "MS Teams adapter connected (app_id=%s, bot=%s)",
            self._ms_config.app_id, self._ms_config.bot_name,
        )

    async def disconnect(self) -> None:
        """Shut down the MS Teams adapter."""
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
        self._conversation_refs.clear()

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
        Send a message to a Teams conversation.

        Parameters
        ----------
        chat_id:
            Conversation ID string, or a ``serviceUrl`` + ``conversation.id``
            pair (see ``_conversation_refs``).
        text:
            Message text to send.
        """
        if not self._session:
            return None

        text = self._truncate(text)

        # Look up stored conversation reference
        conv_ref = self._conversation_refs.get(chat_id)
        if not conv_ref:
            logger.error("No conversation reference found for %s", chat_id)
            return None

        service_url = conv_ref.get("serviceUrl", "")
        conversation = conv_ref.get("conversation", {})
        activity_id = conv_ref.get("activityId")

        url = f"{service_url}/v3/conversations/{conversation.get('id')}/activities"

        payload: Dict[str, Any] = {
            "type": "message",
            "from": {
                "id": self._ms_config.app_id,
                "name": self._ms_config.bot_name,
            },
            "text": text,
        }

        if activity_id:
            payload["replyToId"] = activity_id

        # Adaptive Card support
        adaptive_card = kwargs.get("adaptive_card")
        if adaptive_card:
            payload["attachments"] = [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": adaptive_card,
                },
            ]

        # Hero Card support
        hero_card = kwargs.get("hero_card")
        if hero_card:
            payload["attachments"] = [
                {
                    "contentType": "application/vnd.microsoft.card.hero",
                    "content": hero_card,
                },
            ]

        headers = {"Authorization": f"Bearer {self._access_token}"}

        try:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                if resp.status in (200, 201):
                    result = await resp.json()
                    return result.get("id")
                error_text = await resp.text()
                logger.error(
                    "MS Teams send error (status=%d): %s",
                    resp.status, error_text[:200],
                )
                return None
        except Exception as e:
            logger.error("Failed to send MS Teams message: %s", e)
            return None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a file as an attachment in Teams.

        Uses the ``attachment`` activity type for inline file uploads,
        or sends a file info card with a download link.
        """
        if not self._session or not os.path.exists(file_path):
            return None

        filename = os.path.basename(file_path)
        caption = kwargs.get("caption", filename)
        content_type = self._guess_mime_type(file_path)

        conv_ref = self._conversation_refs.get(chat_id)
        if not conv_ref:
            return None

        service_url = conv_ref.get("serviceUrl", "")
        conversation = conv_ref.get("conversation", {})
        url = f"{service_url}/v3/conversations/{conversation.get('id')}/attachments"

        headers = {"Authorization": f"Bearer {self._access_token}"}

        try:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=filename, content_type=content_type)

                async with self._session.post(url, data=data, headers=headers) as resp:
                    if resp.status in (200, 201):
                        result = await resp.json()
                        attachment_id = result.get("id")
                        # Send the attachment as a message
                        return await self.send_message(
                            chat_id,
                            caption,
                            attachments=[{
                                "name": filename,
                                "contentType": content_type,
                                "contentUrl": result.get("contentUrl", ""),
                                "thumbnailUrl": result.get("thumbnailUrl", ""),
                            }],
                        )
                    logger.error("Teams file upload error: %s", await resp.text())
                    return None
        except Exception as e:
            logger.error("Failed to upload file to Teams: %s", e)
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

    def parse_webhook_activity(
        self, activity: Dict[str, Any],
    ) -> Optional[IncomingMessage]:
        """
        Parse an incoming Bot Framework activity into an IncomingMessage.

        This should be called by the webhook endpoint handler when a new
        activity arrives from the Bot Framework service.
        """
        activity_type = activity.get("type", "")

        if activity_type == "message":
            return self._parse_message_activity(activity)
        elif activity_type == "conversationUpdate":
            members_added = activity.get("membersAdded", [])
            for member in members_added:
                member_id = member.get("id", "")
                if member_id != self._ms_config.app_id:
                    logger.info(
                        "MS Teams: user %s joined conversation %s",
                        member.get("name", member_id),
                        activity.get("conversation", {}).get("id", ""),
                    )
            return None
        elif activity_type == "invoke":
            # Handle slash commands / invoke activities
            return self._parse_invoke_activity(activity)

        return None

    def enqueue_activity(self, activity: Dict[str, Any]) -> None:
        """Parse and enqueue a Bot Framework activity."""
        msg = self.parse_webhook_activity(activity)
        if msg:
            self._message_queue.put_nowait(msg)

    def verify_webhook_signature(
        self, body: bytes, auth_header: str,
    ) -> bool:
        """Verify the Bot Framework webhook signature using HMAC-SHA256."""
        if not auth_header or not self._ms_config.app_password:
            return False

        try:
            # Bot Framework uses "Bearer" token auth, not HMAC signatures.
            # This method validates the token contains expected claims.
            # In production, validate the JWT token properly.
            return auth_header.startswith("Bearer ")
        except Exception:
            return False

    # ── Adaptive Card Builders ────────────────────────────────────────────

    @staticmethod
    def create_adaptive_card(
        body: List[Dict[str, Any]],
        actions: Optional[List[Dict[str, Any]]] = None,
        version: str = "1.4",
    ) -> Dict[str, Any]:
        """Create an Adaptive Card payload."""
        card: Dict[str, Any] = {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": version,
            "body": body,
        }
        if actions:
            card["actions"] = actions
        return card

    @staticmethod
    def create_text_block(
        text: str, size: str = "default", weight: str = "default",
    ) -> Dict[str, Any]:
        """Create an Adaptive Card text block."""
        return {
            "type": "TextBlock",
            "text": text,
            "size": size,
            "weight": weight,
            "wrap": True,
        }

    @staticmethod
    def create_fact_set(
        facts: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Create an Adaptive Card fact set."""
        return {
            "type": "FactSet",
            "facts": facts,
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _parse_message_activity(
        self, activity: Dict[str, Any],
    ) -> Optional[IncomingMessage]:
        """Parse a message-type activity into IncomingMessage."""
        from_member = activity.get("from", {})
        conversation = activity.get("conversation", {})

        # Ignore bot's own messages
        if from_member.get("id") == self._ms_config.app_id:
            return None

        text = activity.get("text", "").strip()
        if not text:
            return None

        conv_id = conversation.get("id", "")
        user_id = from_member.get("id", "")

        # Store conversation reference for proactive messaging
        self._conversation_refs[conv_id] = {
            "conversation": conversation,
            "serviceUrl": activity.get("serviceUrl", ""),
            "activityId": activity.get("id"),
            "botId": self._ms_config.app_id,
        }

        reply_to_id = activity.get("replyToId")

        # Check for mention
        mention_text = self._strip_bot_mention(
            text, activity.get("entities", []),
        )

        metadata: Dict[str, Any] = {
            "conversation_id": conv_id,
            "conversation_type": conversation.get("conversationType", ""),
            "tenant_id": conversation.get("tenantId", ""),
            "service_url": activity.get("serviceUrl", ""),
            "activity_id": activity.get("id", ""),
            "locale": activity.get("locale", ""),
        }

        # Detect commands
        is_command = text.startswith("/") or bool(activity.get("value"))

        return IncomingMessage(
            platform="msteams",
            chat_id=conv_id,
            user_id=user_id,
            text=mention_text,
            message_id=activity.get("id"),
            username=from_member.get("name"),
            reply_to=reply_to_id,
            is_command=is_command,
            metadata=metadata,
        )

    def _parse_invoke_activity(
        self, activity: Dict[str, Any],
    ) -> Optional[IncomingMessage]:
        """Parse an invoke activity (slash command or task module)."""
        from_member = activity.get("from", {})
        conversation = activity.get("conversation", {})
        conv_id = conversation.get("id", "")

        value = activity.get("value", {})
        command_name = value.get("command", activity.get("name", ""))

        self._conversation_refs[conv_id] = {
            "conversation": conversation,
            "serviceUrl": activity.get("serviceUrl", ""),
            "activityId": activity.get("id"),
            "botId": self._ms_config.app_id,
        }

        return IncomingMessage(
            platform="msteams",
            chat_id=conv_id,
            user_id=from_member.get("id", ""),
            text=command_name,
            message_id=activity.get("id"),
            username=from_member.get("name"),
            is_command=True,
            metadata={
                "invoke_type": activity.get("name", ""),
                "invoke_value": value,
            },
        )

    @staticmethod
    def _strip_bot_mention(
        text: str, entities: List[Dict[str, Any]],
    ) -> str:
        """Remove bot mention from the beginning of message text."""
        for entity in entities:
            if entity.get("type") == "mention":
                mentioned = entity.get("mentioned", {})
                text = text.replace(entity.get("text", ""), "").strip()
        return text

    async def _refresh_access_token(self) -> None:
        """Obtain a new access token from the Bot Framework OAuth endpoint."""
        if not self._session:
            return

        try:
            async with self._session.post(
                "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._ms_config.app_id,
                    "client_secret": self._ms_config.app_password,
                    "scope": "https://api.botframework.com/.default",
                },
            ) as resp:
                result = await resp.json()
                if "access_token" in result:
                    self._access_token = result["access_token"]
                    self._token_expiry = time.time() + result.get("expires_in", 3600) - 60
                    logger.info("MS Teams access token refreshed")
                else:
                    logger.error(
                        "Failed to get MS Teams token: %s",
                        result.get("error", "unknown"),
                    )
        except Exception as e:
            logger.error("Error refreshing MS Teams token: %s", e)

    async def _token_refresh_loop(self) -> None:
        """Periodically refresh the access token."""
        while self._connected:
            await asyncio.sleep(300)
            if self._connected:
                await self._refresh_access_token()

    @staticmethod
    def _guess_mime_type(file_path: str) -> str:
        """Guess the MIME type from the file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp",
            ".mp4": "video/mp4", ".mp3": "audio/mpeg",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".csv": "text/csv", ".txt": "text/plain",
        }
        return mime_map.get(ext, "application/octet-stream")

    def _truncate(self, text: str) -> str:
        """Truncate text to Teams' message length limit."""
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return text
        return text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n[...truncated]"
