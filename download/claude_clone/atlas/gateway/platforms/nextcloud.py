"""
Nextcloud Talk adapter for the Atlas Gateway.

Supports the Nextcloud Talk API for sending/receiving messages in
conversations (rooms), file sharing, reaction handling, and Bot API
integration with Nextcloud servers.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.nextcloud import NextcloudAdapter

    config = PlatformConfig(
        name="nextcloud",
        token="NEXTCLOUD_TALK_BOT_TOKEN",
        enabled=True,
        extra={
            "server_url": "https://cloud.example.com",
            "bot_secret": "shared_secret",
        },
    )
    adapter = NextcloudAdapter(config)
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

logger = logging.getLogger("atlas.gateway.platforms.nextcloud")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class NextcloudConfig:
    """Configuration for the Nextcloud Talk adapter."""

    server_url: str = ""
    bot_token: str = ""
    bot_secret: str = ""
    admin_token: str = ""
    timeout: int = 30
    verify_ssl: bool = True


class NextcloudAdapter:
    """
    Nextcloud Talk adapter using the Talk Bot API.

    Messages are received via webhook events from the Nextcloud server
    and outgoing messages use the Talk Bot API endpoints. Supports
    rich messages, file sharing, and reaction handling.

    Parameters
    ----------
    config:
        Platform configuration. Requires ``token`` (bot token) and
        ``server_url`` (Nextcloud instance URL).
    """

    MAX_MESSAGE_LENGTH = 32000

    def __init__(self, config: Any):
        self._config = config
        extra = getattr(config, "extra", {}) or {}

        self._nc_config = NextcloudConfig(
            server_url=extra.get("server_url") or os.environ.get("NEXTCLOUD_URL", ""),
            bot_token=config.token or os.environ.get("NEXTCLOUD_TALK_BOT_TOKEN", ""),
            bot_secret=extra.get("bot_secret") or os.environ.get("NEXTCLOUD_BOT_SECRET", ""),
            admin_token=extra.get("admin_token") or os.environ.get("NEXTCLOUD_ADMIN_TOKEN", ""),
            timeout=config.timeout or 30,
            verify_ssl=extra.get("verify_ssl", True),
        )

        self._connected = False
        self._session: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._api_base = ""

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to the Nextcloud Talk API."""
        if not self._nc_config.server_url:
            raise ValueError(
                "Nextcloud server URL is required. "
                "Set NEXTCLOUD_URL env-var or config.extra.server_url."
            )
        if not self._nc_config.bot_token:
            raise ValueError(
                "Nextcloud Talk bot token is required. "
                "Set NEXTCLOUD_TALK_BOT_TOKEN env-var or config.token."
            )
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        server = self._nc_config.server_url.rstrip("/")
        self._api_base = f"{server}/ocs/v2.php/apps/spreed/api/v1"

        connector = aiohttp.TCPConnector(ssl=self._nc_config.verify_ssl)
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._nc_config.timeout),
            headers={
                "Authorization": f"Bearer {self._nc_config.admin_token}",
                "OCS-APIRequest": "true",
            },
            connector=connector,
        )

        # Verify server connection
        try:
            async with self._session.get(
                f"{server}/status.php",
            ) as resp:
                if resp.status == 200:
                    status = await resp.json()
                    logger.info(
                        "Nextcloud adapter connected: %s (v%s)",
                        self._nc_config.server_url,
                        status.get("version", "unknown"),
                    )
                else:
                    logger.warning("Nextcloud status check returned %d", resp.status)
        except Exception as e:
            logger.error("Could not reach Nextcloud server: %s", e)

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from Nextcloud Talk."""
        self._connected = False
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
        Send a message to a Nextcloud Talk conversation.

        Parameters
        ----------
        chat_id:
            Nextcloud Talk room token.
        text:
            Message text to send.
        """
        if not self._session:
            return None

        text = self._truncate(text)

        payload: Dict[str, Any] = {
            "token": chat_id,
            "message": text,
            "actorDisplayName": "Atlas Bot",
            "actorType": "bots",
        }

        # Reply support
        if kwargs.get("reply_to"):
            payload["replyTo"] = kwargs["reply_to"]

        url = f"{self._api_base}/bot/{chat_id}/message"
        headers = {"Authorization": f"Bearer {self._nc_config.bot_token}"}

        try:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                if resp.status in (200, 201, 202):
                    data = await resp.json()
                    msg_id = data.get("id")
                    logger.info("Nextcloud message sent (id=%s)", msg_id)
                    return str(msg_id) if msg_id else f"nc_{int(time.time())}"
                error = await resp.text()
                logger.error("Nextcloud send error (status=%d): %s", resp.status, error[:200])
                return None
        except Exception as e:
            logger.error("Failed to send Nextcloud message: %s", e)
            return None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a file to a Nextcloud Talk conversation.

        Uploads the file to Nextcloud and shares it in the room.
        """
        if not self._session or not os.path.exists(file_path):
            return None

        filename = os.path.basename(file_path)
        server = self._nc_config.server_url.rstrip("/")

        try:
            # Upload file to Nextcloud
            upload_url = f"{server}/remote.php/dav/files/admin/AtlasBot/{filename}"

            with open(file_path, "rb") as f:
                put_headers = {"Authorization": f"Bearer {self._nc_config.admin_token}"}
                async with self._session.put(
                    upload_url, data=f, headers=put_headers,
                ) as resp:
                    if resp.status not in (201, 204):
                        error = await resp.text()
                        logger.error("Nextcloud file upload error: %s", error[:200])
                        return None

            # Share the file in the room via a rich object message
            file_share_path = f"admin/files/AtlasBot/{filename}"
            caption = kwargs.get("caption", filename)

            payload = {
                "token": chat_id,
                "message": caption,
                "actorDisplayName": "Atlas Bot",
                "actorType": "bots",
                "messageType": "comment",
                "richObject": json.dumps({
                    "type": "file",
                    "id": file_share_path,
                    "name": filename,
                    "path": file_share_path,
                    "link": f"{server}/f/{filename}",
                    "mimetype": self._guess_mime_type(file_path),
                    "size": os.path.getsize(file_path),
                }),
            }

            url = f"{self._api_base}/bot/{chat_id}/message"
            headers = {"Authorization": f"Bearer {self._nc_config.bot_token}"}

            async with self._session.post(url, json=payload, headers=headers) as resp:
                if resp.status in (200, 201, 202):
                    return f"file_{int(time.time())}"
                logger.error("Nextcloud file share error: %s", await resp.text())
                return None
        except Exception as e:
            logger.error("Failed to send file via Nextcloud: %s", e)
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

    def verify_webhook_signature(
        self, body: bytes, signature: str,
    ) -> bool:
        """
        Verify the Nextcloud Talk Bot webhook HMAC signature.

        Nextcloud sends a base64-encoded HMAC-SHA256 of the request body
        signed with the shared bot secret.
        """
        secret = self._nc_config.bot_secret
        if not secret or not signature:
            return False

        expected = hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256,
        ).digest()
        expected_b64 = hmac.new(
            b"", expected, hashlib.sha256,
        ).hexdigest()

        try:
            return hmac.compare_digest(expected_b64, signature)
        except Exception:
            return False

    def parse_webhook_event(
        self, event: Dict[str, Any],
    ) -> Optional[IncomingMessage]:
        """
        Parse a Nextcloud Talk Bot webhook event.

        Nextcloud sends events for new messages, reactions, edits,
        and conversation updates.
        """
        event_type = event.get("event", {})
        if not event_type:
            return None

        event_kind = event_type.get("type", "")

        if event_kind == "message":
            return self._parse_message_event(event)
        elif event_kind == "reaction":
            return self._parse_reaction_event(event)
        elif event_kind == "conversationCreated":
            logger.info(
                "Nextcloud: conversation %s created",
                event.get("conversation", {}).get("token", "unknown"),
            )
            return None

        return None

    def enqueue_event(self, event: Dict[str, Any]) -> None:
        """Parse and enqueue a webhook event."""
        msg = self.parse_webhook_event(event)
        if msg:
            self._message_queue.put_nowait(msg)

    # ── Room Management ───────────────────────────────────────────────────

    async def get_room_list(self) -> List[Dict[str, Any]]:
        """List all conversations the bot is a participant in."""
        if not self._session:
            return []

        url = f"{self._api_base}/room"
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    ocs = data.get("ocs", {})
                    return ocs.get("data", [])
                return []
        except Exception as e:
            logger.error("Nextcloud room list error: %s", e)
            return []

    async def join_room(self, room_token: str) -> bool:
        """Join a Nextcloud Talk conversation."""
        if not self._session:
            return False

        url = f"{self._api_base}/room/{room_token}/participants/active"
        try:
            async with self._session.post(url) as resp:
                return resp.status in (200, 201)
        except Exception as e:
            logger.error("Nextcloud join room error: %s", e)
            return False

    async def send_reaction(
        self, room_token: str, message_id: str, emoji: str,
    ) -> bool:
        """React to a message with an emoji."""
        if not self._session:
            return False

        url = f"{self._api_base}/reaction/{room_token}/{message_id}"
        headers = {"Authorization": f"Bearer {self._nc_config.bot_token}"}
        payload = {"reaction": emoji}

        try:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                return resp.status in (200, 201)
        except Exception as e:
            logger.error("Nextcloud reaction error: %s", e)
            return False

    # ── Internal ──────────────────────────────────────────────────────────

    def _parse_message_event(
        self, event: Dict[str, Any],
    ) -> Optional[IncomingMessage]:
        """Parse a Nextcloud Talk message event."""
        message = event.get("message", {})
        sender = event.get("actor", {})

        # Ignore bot's own messages
        actor_type = sender.get("type", "")
        if actor_type == "bots":
            return None

        text = message.get("message", "").strip()
        if not text:
            return None

        room_token = event.get("conversation", {}).get("token", "")
        message_id = str(message.get("id", ""))
        user_id = sender.get("id", "")
        username = sender.get("displayName", "")

        metadata: Dict[str, Any] = {
            "actor_type": actor_type,
            "message_type": message.get("messageType", "comment"),
            "room_name": event.get("conversation", {}).get("name", ""),
            "room_type": event.get("conversation", {}).get("type", ""),
            "timestamp": message.get("timestamp", ""),
        }

        # Detect reply
        reply_to = message.get("replyTo")

        # Detect rich objects (files, etc.)
        rich_objects = message.get("richObject")
        if rich_objects:
            try:
                parsed = json.loads(rich_objects) if isinstance(rich_objects, str) else rich_objects
                metadata["rich_object"] = parsed
            except (json.JSONDecodeError, TypeError):
                pass

        return IncomingMessage(
            platform="nextcloud",
            chat_id=room_token,
            user_id=user_id,
            text=text,
            message_id=message_id,
            username=username,
            reply_to=str(reply_to) if reply_to else None,
            metadata=metadata,
        )

    def _parse_reaction_event(
        self, event: Dict[str, Any],
    ) -> Optional[IncomingMessage]:
        """Parse a Nextcloud Talk reaction event."""
        reaction = event.get("reaction", {})
        message_id = reaction.get("messageId", "")
        emoji = reaction.get("reaction", "")

        room_token = event.get("conversation", {}).get("token", "")
        sender = event.get("actor", {})
        user_id = sender.get("id", "")

        if not emoji or not message_id:
            return None

        return IncomingMessage(
            platform="nextcloud",
            chat_id=room_token,
            user_id=user_id,
            text=f"[Reaction: {emoji} on message {message_id}]",
            message_id=f"reaction_{message_id}",
            username=sender.get("displayName", ""),
            metadata={
                "event_type": "reaction",
                "reaction_emoji": emoji,
                "reaction_message_id": message_id,
            },
        )

    @staticmethod
    def _guess_mime_type(file_path: str) -> str:
        """Guess MIME type from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".mp4": "video/mp4", ".webm": "video/webm",
            ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
            ".txt": "text/plain", ".csv": "text/csv",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        return mime_map.get(ext, "application/octet-stream")

    def _truncate(self, text: str) -> str:
        """Truncate text to Nextcloud Talk's message length limit."""
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return text
        return text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n[...truncated]"
