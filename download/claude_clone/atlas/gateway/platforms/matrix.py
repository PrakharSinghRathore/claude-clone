"""
Matrix adapter for the Atlas Gateway.

Supports room management, encrypted messaging, media handling,
and user verification through the Matrix Client-Server API.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.matrix import MatrixAdapter

    config = PlatformConfig(
        name="matrix",
        token="MATRIX_ACCESS_TOKEN",
        api_url="https://matrix.org",
        api_key="@bot:matrix.org",  # User ID
        enabled=True,
    )
    adapter = MatrixAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.matrix")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class MatrixAdapter:
    """
    Matrix Client-Server API adapter.

    Uses the Application Service or bot account approach for message
    sending and long-polling for receiving.

    Parameters
    ----------
    config:
        Platform configuration. Requires:
        - ``token``: Matrix access token
        - ``api_url``: Matrix homeserver base URL
        - ``api_key``: Full Matrix user ID (@bot:matrix.org)
    """

    def __init__(self, config: Any):
        self._config = config
        self._token = config.token or os.environ.get("MATRIX_ACCESS_TOKEN", "")
        self._homeserver = config.api_url or os.environ.get("MATRIX_HOMESERVER", "https://matrix.org")
        self._user_id = config.api_key or os.environ.get("MATRIX_USER_ID", "")
        self._timeout = config.timeout or 30
        self._connected = False
        self._session: Optional[Any] = None
        self._since_token: Optional[str] = None
        self._transaction_id = 0
        self._sync_task: Optional[asyncio.Task] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to the Matrix homeserver."""
        if not self._token:
            raise ValueError("Matrix access token is required")
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
            headers={"Authorization": f"Bearer {self._token}"},
        )

        # Verify login
        whoami = await self._matrix_get("/_matrix/client/v3/account/whoami")
        if whoami and whoami.get("user_id"):
            self._user_id = whoami["user_id"]
            logger.info("Matrix connected as %s", self._user_id)
        else:
            logger.warning("Matrix whoami failed — continuing anyway")

        # Start sync loop
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from Matrix."""
        self._connected = False
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
            self._session = None

    async def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a text message to a Matrix room."""
        self._transaction_id += 1
        url = f"/_matrix/client/v3/rooms/{quote(chat_id, safe='')}/send/m.room.message/{self._transaction_id}"

        payload: Dict[str, Any] = {
            "msgtype": "m.text",
            "body": text,
        }

        # Use HTML format if specified
        if kwargs.get("format") == "html":
            payload["format"] = "org.matrix.custom.html"
            payload["formatted_body"] = text

        # Markdown support
        if kwargs.get("markdown"):
            payload["msgtype"] = "m.text"
            payload["body"] = text
            payload["format"] = "org.matrix.custom.html"
            payload["formatted_body"] = self._markdown_to_html(text)

        result = await self._matrix_put(url, payload)
        return result.get("event_id") if result else None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a file to a Matrix room."""
        if not self._session or not os.path.exists(file_path):
            return None

        # Determine content type
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".mp4": "video/mp4",
            ".mp3": "audio/mpeg", ".pdf": "application/pdf",
        }
        content_type = mime_map.get(ext, "application/octet-stream")
        filename = os.path.basename(file_path)

        with open(file_path, "rb") as f:
            file_content = f.read()

        # Upload file to Matrix media
        upload_url = "/_matrix/media/v3/upload"
        headers = {"Content-Type": content_type}

        async with self._session.post(
            f"{self._homeserver}{upload_url}",
            data=file_content,
            headers=headers,
        ) as resp:
            data = await resp.json()
            content_uri = data.get("content_uri")
            if not content_uri:
                logger.error("Matrix file upload failed: %s", data)
                return None

        # Send the file event
        self._transaction_id += 1
        event_url = f"/_matrix/client/v3/rooms/{quote(chat_id, safe='')}/send/m.room.message/{self._transaction_id}"

        if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            payload = {
                "msgtype": "m.image",
                "body": filename,
                "url": content_uri,
                "info": {"mimetype": content_type},
            }
        elif ext in (".mp4",):
            payload = {
                "msgtype": "m.video",
                "body": filename,
                "url": content_uri,
                "info": {"mimetype": content_type},
            }
        elif ext in (".mp3", ".ogg", ".m4a"):
            payload = {
                "msgtype": "m.audio",
                "body": filename,
                "url": content_uri,
                "info": {"mimetype": content_type},
            }
        else:
            payload = {
                "msgtype": "m.file",
                "body": filename,
                "url": content_uri,
                "info": {"mimetype": content_type},
            }

        result = await self._matrix_put(event_url, payload)
        return result.get("event_id") if result else None

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for new Matrix messages."""
        messages: List[IncomingMessage] = []
        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    # ── Extended Interface ────────────────────────────────────────────────

    async def edit_message(
        self, chat_id: str, message_id: str, text: str, **kwargs: Any,
    ) -> bool:
        """Edit an existing message (Matrix uses m.new_content)."""
        self._transaction_id += 1
        url = f"/_matrix/client/v3/rooms/{quote(chat_id, safe='')}/send/m.room.message/{self._transaction_id}"

        payload: Dict[str, Any] = {
            "msgtype": "m.text",
            "body": text,
            "m.new_content": {
                "msgtype": "m.text",
                "body": text,
            },
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": message_id,
            },
        }

        result = await self._matrix_put(url, payload)
        return bool(result and result.get("event_id"))

    async def send_typing(self, chat_id: str) -> None:
        """Send a typing notification to a room."""
        url = f"/_matrix/client/v3/rooms/{quote(chat_id, safe='')}/typing/{quote(self._user_id, safe='')}"
        payload = {"typing": True, "timeout": 30000}
        await self._matrix_put(url, payload)

    async def redact_message(
        self, chat_id: str, event_id: str, reason: str = "",
    ) -> bool:
        """Redact (delete) a message."""
        self._transaction_id += 1
        url = f"/_matrix/client/v3/rooms/{quote(chat_id, safe='')}/redact/{event_id}/{self._transaction_id}"
        payload = {"reason": reason}
        result = await self._matrix_put(url, payload)
        return bool(result and result.get("event_id"))

    async def join_room(self, room_id: str) -> bool:
        """Join a Matrix room."""
        url = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/join"
        result = await self._matrix_post(url, {})
        return bool(result and result.get("room_id"))

    async def invite_user(self, room_id: str, user_id: str) -> bool:
        """Invite a user to a room."""
        url = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/invite"
        result = await self._matrix_post(url, {"user_id": user_id})
        return bool(result)

    # ── Sync Loop ─────────────────────────────────────────────────────────

    async def _sync_loop(self) -> None:
        """Long-poll sync loop for receiving events."""
        while self._connected:
            try:
                params: Dict[str, Any] = {
                    "timeout": 30000,
                    "filter": json.dumps({
                        "room": {
                            "timeline": {"types": ["m.room.message"]},
                        },
                        "presence": {"types": []},
                    }),
                }
                if self._since_token:
                    params["since"] = self._since_token

                url = "/_matrix/client/v3/sync"
                async with self._session.get(
                    f"{self._homeserver}{url}", params=params,
                ) as resp:
                    data = await resp.json()

                    if data.get("next_batch"):
                        self._since_token = data["next_batch"]

                    # Process room events
                    rooms = data.get("rooms", {}).get("join", {})
                    for room_id, room_data in rooms.items():
                        for event in room_data.get("timeline", {}).get("events", []):
                            msg = self._parse_event(room_id, event)
                            if msg:
                                await self._message_queue.put(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Matrix sync error: %s", e)
                await asyncio.sleep(5)

    def _parse_event(
        self, room_id: str, event: Dict[str, Any],
    ) -> Optional[IncomingMessage]:
        """Parse a Matrix event into an IncomingMessage."""
        if event.get("type") != "m.room.message":
            return None

        sender = event.get("sender", "")
        if sender == self._user_id:
            return None

        content = event.get("content", {})
        msg_type = content.get("msgtype", "")
        text = content.get("body", "")
        event_id = event.get("event_id", "")

        metadata: Dict[str, Any] = {
            "msgtype": msg_type,
            "event_type": event.get("type"),
        }

        # Handle different message types
        if msg_type == "m.image":
            metadata["url"] = content.get("url")
        elif msg_type == "m.file":
            metadata["url"] = content.get("url")
            metadata["filename"] = content.get("filename")
        elif msg_type == "m.video":
            metadata["url"] = content.get("url")
        elif msg_type == "m.audio":
            metadata["url"] = content.get("url")

        # Check for edits
        relates_to = content.get("m.relates_to", {})
        if relates_to.get("rel_type") == "m.replace":
            metadata["edit_of"] = relates_to.get("event_id")

        return IncomingMessage(
            platform="matrix",
            chat_id=room_id,
            user_id=sender,
            text=text,
            message_id=event_id,
            metadata=metadata,
            is_edit=bool(relates_to.get("rel_type") == "m.replace"),
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _matrix_get(self, path: str) -> Optional[Dict]:
        if not self._session:
            return None
        try:
            async with self._session.get(f"{self._homeserver}{path}") as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error("Matrix GET error: %s", e)
            return None

    async def _matrix_put(self, path: str, payload: Dict) -> Optional[Dict]:
        if not self._session:
            return None
        try:
            async with self._session.put(
                f"{self._homeserver}{path}", json=payload,
            ) as resp:
                return await resp.json()
        except Exception as e:
            logger.error("Matrix PUT error: %s", e)
            return None

    async def _matrix_post(self, path: str, payload: Dict) -> Optional[Dict]:
        if not self._session:
            return None
        try:
            async with self._session.post(
                f"{self._homeserver}{path}", json=payload,
            ) as resp:
                return await resp.json()
        except Exception as e:
            logger.error("Matrix POST error: %s", e)
            return None

    @staticmethod
    def _markdown_to_html(text: str) -> str:
        """Simple markdown to HTML converter for Matrix."""
        import re
        lines = text.split("\n")
        html_parts = []
        for line in lines:
            # Headers
            m = re.match(r'^### (.+)$', line)
            if m:
                html_parts.append(f"<h3>{m.group(1)}</h3>")
                continue
            m = re.match(r'^## (.+)$', line)
            if m:
                html_parts.append(f"<h2>{m.group(1)}</h2>")
                continue
            m = re.match(r'^# (.+)$', line)
            if m:
                html_parts.append(f"<h1>{m.group(1)}</h1>")
                continue
            # Inline formatting
            line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            line = re.sub(r'\*(.+?)\*', r'<em>\1</em>', line)
            line = re.sub(r'`(.+?)`', r'<code>\1</code>', line)
            html_parts.append(f"<p>{line}</p>" if line else "")
        return "".join(html_parts)
