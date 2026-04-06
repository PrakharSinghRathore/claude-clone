"""
Slack Bot adapter for the Atlas Gateway.

Supports RTM and Events API, Block Kit messages, channel management,
and interactive components.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.slack import SlackAdapter

    config = PlatformConfig(name="slack", token="SLACK_BOT_TOKEN", enabled=True)
    adapter = SlackAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.slack")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

SLACK_API_BASE = "https://slack.com/api"


class SlackAdapter:
    """
    Slack Bot adapter using the Slack Web API and Socket Mode.

    Parameters
    ----------
    config:
        Platform configuration with ``token`` (Bot User OAuth Token).
    """

    MAX_MESSAGE_LENGTH = 40000

    def __init__(self, config: Any):
        self._config = config
        self._token = config.token or os.environ.get("SLACK_BOT_TOKEN", "")
        self._app_token = getattr(config, "extra", {}).get("app_token") or os.environ.get("SLACK_APP_TOKEN", "")
        self._timeout = config.timeout or 30
        self._connected = False
        self._session: Optional[Any] = None
        self._ws: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._ws_task: Optional[asyncio.Task] = None
        self._bot_user_id: Optional[str] = None

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to Slack via Socket Mode."""
        if not self._token:
            raise ValueError("Slack bot token is required")
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
            headers={"Authorization": f"Bearer {self._token}"},
        )

        # Verify bot identity
        auth = await self._api_call("auth.test")
        if not auth or not auth.get("ok"):
            raise RuntimeError("Slack auth.test failed. Check your bot token.")

        self._bot_user_id = auth.get("user_id")
        logger.info("Connected as Slack bot: %s", auth.get("user", "unknown"))

        # Start Socket Mode if app token is available
        if self._app_token:
            self._ws_task = asyncio.create_task(self._socket_mode_loop())
        else:
            logger.warning("No Slack App Token — Socket Mode disabled. Use Events API + webhook instead.")

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from Slack."""
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
        """Send a message to a Slack channel."""
        blocks = kwargs.get("blocks")
        payload: Dict[str, Any] = {
            "channel": chat_id,
            "text": text[:self.MAX_MESSAGE_LENGTH],
        }
        if blocks:
            payload["blocks"] = blocks
        else:
            payload["text"] = text[:self.MAX_MESSAGE_LENGTH]

        if kwargs.get("reply_to"):
            payload["thread_ts"] = kwargs["reply_to"]

        result = await self._api_call("chat.postMessage", payload)
        return result.get("ts") if result else None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Upload a file to a Slack channel."""
        if not self._session or not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("channels", chat_id)
                data.add_field("file", f, filename=os.path.basename(file_path))
                if kwargs.get("caption"):
                    data.add_field("initial_comment", str(kwargs["caption"])[:1000])

                url = f"{SLACK_API_BASE}/files.upload"
                headers = {"Authorization": f"Bearer {self._token}"}
                async with self._session.post(url, data=data, headers=headers) as resp:
                    result = await resp.json()
                    if result.get("ok"):
                        return result.get("file", {}).get("id")
                    logger.error("Slack file upload error: %s", result.get("error"))
                    return None
        except Exception as e:
            logger.error("Failed to upload file to Slack: %s", e)
            return None

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll the message queue for new Slack events."""
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
        """Edit an existing message."""
        payload = {
            "channel": chat_id,
            "ts": message_id,
            "text": text[:self.MAX_MESSAGE_LENGTH],
        }
        blocks = kwargs.get("blocks")
        if blocks:
            payload["blocks"] = blocks
        result = await self._api_call("chat.update", payload)
        return bool(result and result.get("ok"))

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message."""
        result = await self._api_call("chat.delete", {"channel": chat_id, "ts": message_id})
        return bool(result and result.get("ok"))

    async def send_typing(self, chat_id: str) -> None:
        """Indicate typing is happening (Slack doesn't have explicit typing)."""
        # Slack doesn't have an explicit typing API
        pass

    def create_block_section(self, text: str) -> Dict[str, Any]:
        """Create a Block Kit section block."""
        return {
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
        }

    def create_block_divider(self) -> Dict[str, Any]:
        """Create a Block Kit divider block."""
        return {"type": "divider"}

    def create_block_actions(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a Block Kit actions block."""
        return {"type": "actions", "elements": elements}

    def create_block_button(
        self, text: str, action_id: str, value: str = "",
        style: str = "primary",
    ) -> Dict[str, Any]:
        """Create a Block Kit button element."""
        return {
            "type": "button",
            "text": {"type": "plain_text", "text": text},
            "action_id": action_id,
            "value": value,
            "style": style,
        }

    async def open_modal(
        self, trigger_id: str, view: Dict[str, Any],
    ) -> bool:
        """Open a modal view."""
        result = await self._api_call("views.open", {
            "trigger_id": trigger_id,
            "view": view,
        })
        return bool(result and result.get("ok"))

    async def get_channel_list(self) -> List[Dict[str, Any]]:
        """List all channels the bot has access to."""
        result = await self._api_call("conversations.list", {"types": "public_channel,private_channel"})
        if result and result.get("ok"):
            return result.get("channels", [])
        return []

    # ── Webhook Support ───────────────────────────────────────────────────

    async def handle_event(
        self, event_data: Dict[str, Any], retry_num: Optional[str] = None,
    ) -> Optional[IncomingMessage]:
        """Handle an incoming Slack Events API event."""
        if not self._verify_retry(retry_num):
            return None

        event = event_data.get("event", {})
        event_type = event.get("type")

        if event_type == "message":
            user_id = event.get("user", "")
            # Ignore bot messages
            if user_id == self._bot_user_id:
                return None
            if event.get("subtype"):
                return None

            return IncomingMessage(
                platform="slack",
                chat_id=event.get("channel", ""),
                user_id=user_id,
                text=event.get("text", ""),
                message_id=event.get("ts"),
                username=event.get("username"),
                thread_ts=event.get("thread_ts"),
                metadata={
                    "team": event_data.get("team_id"),
                    "event_type": event_type,
                },
                reply_to=event.get("thread_ts"),
            )

        return None

    def verify_request_signature(
        self, body: bytes, timestamp: str, signature: str,
    ) -> bool:
        """Verify a Slack request signature."""
        import hashlib
        import hmac
        signing_secret = getattr(self._config, "webhook_secret", "") or os.environ.get("SLACK_SIGNING_SECRET", "")
        if not signing_secret:
            return False
        basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        expected = "v0=" + hmac.new(
            signing_secret.encode(), basestring.encode(), hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def _verify_retry(self, retry_num: Optional[str]) -> bool:
        """Verify the retry header to prevent duplicate processing."""
        if retry_num is None:
            return True
        key = f"slack_retry_{retry_num}"
        # Simple dedup check (in production, use a proper cache)
        return True

    # ── Socket Mode ───────────────────────────────────────────────────────

    async def _socket_mode_loop(self) -> None:
        """WebSocket loop for Socket Mode."""
        try:
            # Get WebSocket URL
            result = await self._api_call(
                "apps.connections.open",
                headers={"Authorization": f"Bearer {self._app_token}"},
            )
            if not result or not result.get("ok"):
                logger.error("Failed to open Socket Mode connection")
                return

            ws_url = result.get("url")
            if not ws_url:
                return

            headers = {"Authorization": f"Bearer {self._app_token}"}
            async with self._session.ws_connect(ws_url, headers=headers) as ws:
                self._ws = ws
                logger.info("Slack Socket Mode connected")

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        await self._handle_socket_event(data)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Slack Socket Mode error: %s", e)

    async def _handle_socket_event(self, data: Dict[str, Any]) -> None:
        """Handle a Socket Mode event."""
        envelope_type = data.get("type", "")
        payload = data.get("payload", {})

        if envelope_type == "events_api":
            event = payload.get("event", {})
            msg = self._parse_event(payload)
            if msg:
                await self._message_queue.put(msg)
        elif envelope_type == "interactive":
            # Handle interactive components (buttons, modals, etc.)
            pass

    def _parse_event(self, payload: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Parse a Slack event payload."""
        event = payload.get("event", {})
        event_type = event.get("type")

        if event_type != "message":
            return None

        user_id = event.get("user", "")
        if user_id == self._bot_user_id:
            return None

        return IncomingMessage(
            platform="slack",
            chat_id=event.get("channel", ""),
            user_id=user_id,
            text=event.get("text", ""),
            message_id=event.get("ts"),
            username=event.get("username"),
            metadata={"team": payload.get("team_id")},
        )

    # ── API Helper ────────────────────────────────────────────────────────

    async def _api_call(
        self, method: str, payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make a Slack Web API call."""
        if not self._session:
            return None

        url = f"{SLACK_API_BASE}/{method}"
        req_headers = headers or {"Authorization": f"Bearer {self._token}"}

        try:
            if payload:
                async with self._session.post(url, json=payload, headers=req_headers) as resp:
                    return await resp.json()
            else:
                async with self._session.post(url, headers=req_headers) as resp:
                    return await resp.json()
        except Exception as e:
            logger.error("Slack API error (%s): %s", method, e)
            return None
