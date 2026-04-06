"""
Mattermost adapter for the Atlas Gateway.

Supports WebSocket integration, message formatting, channel management,
and attachment support through the Mattermost API.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.mattermost import MattermostAdapter

    config = PlatformConfig(
        name="mattermost",
        token="MATTERMOST_ACCESS_TOKEN",
        api_url="https://mattermost.example.com",
        api_key="BOT_USER_ID",
        enabled=True,
    )
    adapter = MattermostAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.mattermost")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class MattermostAdapter:
    """
    Mattermost adapter using the REST API v4 and WebSocket.

    Parameters
    ----------
    config:
        Platform configuration. Requires:
        - ``token``: Personal Access Token or bot token
        - ``api_url``: Mattermost server base URL
        - ``api_key``: Bot user ID
    """

    MAX_MESSAGE_LENGTH = 16384

    def __init__(self, config: Any):
        self._config = config
        self._token = config.token or os.environ.get("MATTERMOST_TOKEN", "")
        self._base_url = (config.api_url or os.environ.get("MATTERMOST_URL", "")).rstrip("/")
        self._bot_user_id = config.api_key or os.environ.get("MATTERMOST_BOT_USER_ID", "")
        self._timeout = config.timeout or 30
        self._connected = False
        self._session: Optional[Any] = None
        self._ws: Optional[Any] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._team_id: Optional[str] = None
        self._team_name: Optional[str] = None

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to Mattermost via REST API and WebSocket."""
        if not self._token:
            raise ValueError("Mattermost access token is required")
        if not self._base_url:
            raise ValueError("Mattermost server URL is required")
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
            headers={"Authorization": f"Bearer {self._token}"},
        )

        # Verify authentication and get user info
        me = await self._api_get("/api/v4/users/me")
        if me:
            self._bot_user_id = me.get("id", self._bot_user_id)
            logger.info("Mattermost connected as %s", me.get("username", "unknown"))

        # Get default team
        teams = await self._api_get("/api/v4/users/me/teams")
        if teams and len(teams) > 0:
            self._team_id = teams[0].get("id")
            self._team_name = teams[0].get("name")
            logger.info("Mattermost team: %s", self._team_name)

        # Start WebSocket
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from Mattermost."""
        self._connected = False
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()
            self._session = None

    async def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a message to a Mattermost channel or direct message."""
        text = self._truncate(text)
        url = "/api/v4/posts"

        payload: Dict[str, Any] = {
            "channel_id": chat_id,
            "message": text,
        }

        if kwargs.get("root_id"):
            payload["root_id"] = kwargs["root_id"]

        # File IDs to attach
        file_ids = kwargs.get("file_ids")
        if file_ids:
            payload["file_ids"] = file_ids if isinstance(file_ids, list) else [file_ids]

        # Props for special formatting
        props = kwargs.get("props")
        if props:
            payload["props"] = props

        result = await self._api_post(url, payload)
        return result.get("id") if result else None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Upload a file and send it to a channel."""
        if not self._session or not os.path.exists(file_path):
            return None

        # Upload file
        upload_url = f"/api/v4/files"
        try:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("files", f, filename=os.path.basename(file_path))
                data.add_field("channel_id", chat_id)

                async with self._session.post(
                    f"{self._base_url}{upload_url}",
                    data=data,
                    headers={"Authorization": f"Bearer {self._token}"},
                ) as resp:
                    result = await resp.json()
                    if resp.status == 201:
                        file_infos = result.get("file_infos", [])
                        if file_infos:
                            file_id = file_infos[0].get("id")
                            # Now send a post with the file
                            return await self.send_message(
                                chat_id, kwargs.get("caption", ""),
                                file_ids=[file_id],
                            )
                    logger.error("Mattermost file upload error: %s", result)
                    return None
        except Exception as e:
            logger.error("Mattermost file send failed: %s", e)
            return None

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for new Mattermost messages."""
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
        """Edit an existing post."""
        url = f"/api/v4/posts/{message_id}"
        payload = {"id": message_id, "message": self._truncate(text)}
        result = await self._api_put(url, payload)
        return bool(result and result.get("id"))

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a post."""
        url = f"/api/v4/posts/{message_id}"
        status = await self._api_delete(url)
        return status == 200

    async def send_typing(self, chat_id: str) -> None:
        """Not directly supported by Mattermost REST API."""
        pass

    async def send_direct_message(
        self, user_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a direct message to a user."""
        # Get or create DM channel
        url = "/api/v4/channels/direct"
        payload = [self._bot_user_id, user_id]
        channel = await self._api_post(url, payload)
        if not channel:
            return None

        channel_id = channel.get("id")
        return await self.send_message(channel_id, text, **kwargs)

    async def get_channel_list(self) -> List[Dict[str, Any]]:
        """List channels in the default team."""
        if not self._team_id:
            return []
        url = f"/api/v4/teams/{self._team_id}/channels"
        return await self._api_get(url) or []

    async def add_reaction(
        self, post_id: str, emoji_name: str,
    ) -> bool:
        """Add a reaction to a post."""
        url = "/api/v4/reactions"
        payload = {
            "user_id": self._bot_user_id,
            "post_id": post_id,
            "emoji_name": emoji_name,
        }
        result = await self._api_post(url, payload)
        return bool(result)

    # ── WebSocket ─────────────────────────────────────────────────────────

    async def _ws_loop(self) -> None:
        """WebSocket event loop."""
        ws_url = self._base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/api/v4/websocket"

        reconnect_delay = 1.0
        while self._connected:
            try:
                async with self._session.ws_connect(ws_url) as ws:
                    self._ws = ws
                    logger.info("Mattermost WebSocket connected")
                    reconnect_delay = 1.0

                    # Handle sequence
                    team_events = {"posted", "post_edited"}

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            event_type = data.get("event", "")

                            if event_type == "posted":
                                post_data = json.loads(data.get("data", {}).get("post", "{}"))
                                incoming = self._parse_post(post_data, data.get("data", {}))
                                if incoming:
                                    await self._message_queue.put(incoming)
                            elif event_type == "post_edited":
                                post_data = json.loads(data.get("data", {}).get("post", "{}"))
                                incoming = self._parse_post(post_data, data.get("data", {}), is_edit=True)
                                if incoming and incoming.text:
                                    await self._message_queue.put(incoming)
                            elif event_type == "hello":
                                seq = data.get("seq", 0)
                                if seq:
                                    # Acknowledge
                                    await ws.send_json({"seq_reply": seq})

                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Mattermost WebSocket error: %s", e)

            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30.0)

    def _parse_post(
        self, post: Dict[str, Any], event_data: Dict[str, Any],
        is_edit: bool = False,
    ) -> Optional[IncomingMessage]:
        """Parse a Mattermost post into an IncomingMessage."""
        user_id = post.get("user_id", "")
        if user_id == self._bot_user_id:
            return None

        text = post.get("message", "")
        if not text:
            return None

        channel_id = post.get("channel_id", "")
        post_id = post.get("id", "")
        root_id = post.get("root_id")

        # Get sender info
        sender_info = {}
        if "sender_name" in event_data:
            sender_info["username"] = event_data["sender_name"]

        # Get channel type
        channel_type = event_data.get("channel_type", "")
        channel_name = event_data.get("channel_name", "")

        return IncomingMessage(
            platform="mattermost",
            chat_id=channel_id,
            user_id=user_id,
            text=text,
            message_id=post_id,
            reply_to=root_id,
            metadata={
                "channel_type": channel_type,
                "channel_name": channel_name,
                "team_id": event_data.get("team_id"),
                "props": post.get("props", {}),
            },
            is_edit=is_edit,
        )

    # ── API Helpers ───────────────────────────────────────────────────────

    async def _api_get(self, path: str) -> Optional[Any]:
        if not self._session:
            return None
        try:
            async with self._session.get(f"{self._base_url}{path}") as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error("Mattermost GET error (%s): %s", path, e)
            return None

    async def _api_post(self, path: str, payload: Dict) -> Optional[Dict]:
        if not self._session:
            return None
        try:
            async with self._session.post(f"{self._base_url}{path}", json=payload) as resp:
                if resp.status in (200, 201):
                    return await resp.json()
                logger.error("Mattermost POST error (%s): %d", path, resp.status)
                return None
        except Exception as e:
            logger.error("Mattermost POST error (%s): %s", path, e)
            return None

    async def _api_put(self, path: str, payload: Dict) -> Optional[Dict]:
        if not self._session:
            return None
        try:
            async with self._session.put(f"{self._base_url}{path}", json=payload) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error("Mattermost PUT error (%s): %s", path, e)
            return None

    async def _api_delete(self, path: str) -> Optional[int]:
        if not self._session:
            return None
        try:
            async with self._session.delete(f"{self._base_url}{path}") as resp:
                return resp.status
        except Exception as e:
            logger.error("Mattermost DELETE error (%s): %s", path, e)
            return None

    def _truncate(self, text: str) -> str:
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return text
        return text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n[...truncated]"
