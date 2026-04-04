"""
WebSocket Collaboration Server & Client.

Provides real-time collaborative editing features including cursor sharing,
file edits, chat messages, room management, and file sync — all over
plain WebSocket connections.

Requirements: ``websockets``  (pip install websockets)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event / message helpers
# ---------------------------------------------------------------------------

EventType = str  # e.g. "user_joined", "cursor_move", etc.

VALID_EVENTS: frozenset[str] = frozenset({
    "user_joined",
    "user_left",
    "cursor_move",
    "file_edit",
    "chat_message",
    "file_sync",
    "room_created",
    "typing_indicator",
})


def _make_message(event_type: str, data: dict, sender: str = "") -> str:
    """Build a JSON-encoded collaboration message."""
    payload: dict[str, Any] = {"event": event_type, "data": data}
    if sender:
        payload["sender"] = sender
    return json.dumps(payload)


def _parse_message(raw: str) -> dict[str, Any]:
    """Parse an incoming JSON collaboration message."""
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Room & user bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class CollabUser:
    """Represents a connected collaborator."""
    user_id: str
    display_name: str
    room_id: Optional[str] = None
    websocket: Any = None  # websockets.WebSocketServerProtocol

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "room_id": self.room_id,
        }


@dataclass
class Room:
    """A collaboration room."""
    room_id: str
    users: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "room_id": self.room_id,
            "users": self.users,
            "files": list(self.files.keys()),
        }


# ---------------------------------------------------------------------------
# CollabServer
# ---------------------------------------------------------------------------

class CollabServer:
    """WebSocket collaboration server.

    Manages rooms, broadcasts events, and routes private messages.
    """

    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._users: dict[str, CollabUser] = {}
        self._rooms: dict[str, Room] = {}
        self._server: Any = None
        self._running = False
        self._on_connect: Optional[Callable] = None
        self._on_disconnect: Optional[Callable] = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Start the WebSocket server (non-blocking — schedules on loop)."""
        import websockets

        self._running = True
        self._server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
        )
        logger.info("CollabServer listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Shut down the WebSocket server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("CollabServer stopped")

    # -- broadcasting ------------------------------------------------------

    async def broadcast(
        self,
        event_type: str,
        data: dict,
        exclude: Optional[str] = None,
    ) -> None:
        """Send an event to every connected user, optionally excluding one."""
        msg = _make_message(event_type, data, sender="server")
        targets = [
            u for u in self._users.values()
            if u.websocket is not None and u.user_id != exclude
        ]
        await asyncio.gather(
            *(u.websocket.send(msg) for u in targets),
            return_exceptions=True,
        )

    # -- user management ---------------------------------------------------

    async def get_connected_users(self) -> list[dict]:
        """Return information about all connected users."""
        return [u.to_dict() for u in self._users.values()]

    # -- room management ---------------------------------------------------

    async def create_room(self, room_id: str) -> None:
        """Create a new collaboration room."""
        if room_id not in self._rooms:
            self._rooms[room_id] = Room(room_id=room_id)
            await self.broadcast("room_created", {"room_id": room_id})
            logger.info("Room created: %s", room_id)

    async def join_room(self, user_id: str, room_id: str) -> None:
        """Add *user_id* to *room_id*, creating the room if necessary."""
        if room_id not in self._rooms:
            await self.create_room(room_id)

        room = self._rooms[room_id]
        user = self._users.get(user_id)
        if user is None:
            logger.warning("join_room: unknown user %s", user_id)
            return

        # Leave previous room if any
        if user.room_id and user.room_id != room_id:
            await self.leave_room(user_id)

        if user_id not in room.users:
            room.users.append(user_id)
        user.room_id = room_id

        # Notify room mates
        msg = _make_message(
            "user_joined",
            {"room_id": room_id, "user": user.to_dict()},
            sender=user_id,
        )
        await self._send_to_room(room_id, msg, exclude=user_id)
        logger.info("%s joined room %s", user_id, room_id)

    async def leave_room(self, user_id: str) -> None:
        """Remove *user_id* from their current room."""
        user = self._users.get(user_id)
        if user is None or user.room_id is None:
            return

        room = self._rooms.get(user.room_id)
        if room and user_id in room.users:
            room.users.remove(user_id)

        msg = _make_message(
            "user_left",
            {"room_id": user.room_id, "user_id": user_id},
            sender=user_id,
        )
        await self._send_to_room(user.room_id, msg, exclude=user_id)

        # Clean up empty rooms
        if room and not room.users:
            del self._rooms[room.room_id]
            logger.info("Empty room removed: %s", room.room_id)

        logger.info("%s left room %s", user_id, user.room_id)
        user.room_id = None

    # -- direct messaging --------------------------------------------------

    async def send_to_user(
        self,
        user_id: str,
        event_type: str,
        data: dict,
    ) -> None:
        """Send a targeted event to a specific user."""
        user = self._users.get(user_id)
        if user is None or user.websocket is None:
            logger.warning("send_to_user: user %s not connected", user_id)
            return
        msg = _make_message(event_type, data, sender="server")
        await user.websocket.send(msg)

    # -- internal ----------------------------------------------------------

    async def _send_to_room(
        self,
        room_id: str,
        message: str,
        exclude: Optional[str] = None,
    ) -> None:
        """Deliver *message* to all users in *room_id*."""
        room = self._rooms.get(room_id)
        if room is None:
            return
        targets = [
            self._users[uid]
            for uid in room.users
            if uid != exclude and uid in self._users
            and self._users[uid].websocket is not None
        ]
        await asyncio.gather(
            *(u.websocket.send(message) for u in targets),
            return_exceptions=True,
        )

    async def _handler(self, websocket: Any) -> None:
        """Connection handler — registers user and routes messages."""
        user_id = str(uuid.uuid4())[:8]
        user = CollabUser(user_id=user_id, display_name=f"User-{user_id}")
        user.websocket = websocket
        self._users[user_id] = user

        try:
            # Wait for the first message to set display name
            first = await websocket.recv()
            init = _parse_message(first)
            if init.get("event") == "auth":
                user.display_name = init.get("data", {}).get(
                    "display_name", user.display_name
                )
                user_id = init.get("data", {}).get("user_id", user_id)
                user.user_id = user_id
                # Re-register under the new id if different
                if user_id != list(self._users.keys())[-1]:
                    del self._users[list(self._users.keys())[-1]]
                    if user_id in self._users:
                        # Close existing
                        old = self._users.pop(user_id)
                        if old.websocket:
                            await old.websocket.close()
                    self._users[user_id] = user

            await self.broadcast("user_joined", {"user": user.to_dict()})
            logger.info("User connected: %s (%s)", user.display_name, user_id)

            async for raw in websocket:
                try:
                    message = _parse_message(raw)
                except json.JSONDecodeError:
                    continue

                event = message.get("event", "")
                data = message.get("data", {})

                if event == "join_room":
                    await self.join_room(user.user_id, data.get("room_id", ""))
                elif event == "leave_room":
                    await self.leave_room(user.user_id)
                elif event == "cursor_move":
                    data["user_id"] = user.user_id
                    if user.room_id:
                        await self._send_to_room(
                            user.room_id,
                            _make_message("cursor_move", data, sender=user.user_id),
                            exclude=user.user_id,
                        )
                elif event == "file_edit":
                    data["user_id"] = user.user_id
                    if user.room_id:
                        await self._send_to_room(
                            user.room_id,
                            _make_message("file_edit", data, sender=user.user_id),
                            exclude=user.user_id,
                        )
                elif event == "chat_message":
                    data["user_id"] = user.user_id
                    data["display_name"] = user.display_name
                    target_room = data.get("room_id") or user.room_id
                    if target_room:
                        await self._send_to_room(
                            target_room,
                            _make_message("chat_message", data, sender=user.user_id),
                        )
                elif event == "typing_indicator":
                    data["user_id"] = user.user_id
                    if user.room_id:
                        await self._send_to_room(
                            user.room_id,
                            _make_message("typing_indicator", data, sender=user.user_id),
                            exclude=user.user_id,
                        )
                elif event == "request_sync":
                    await self._handle_sync_request(user, data)
                elif event == "file_sync":
                    # Store file content in the room
                    if user.room_id:
                        room = self._rooms.get(user.room_id)
                        if room:
                            room.files[data.get("file", "")] = data.get("content", "")

        except Exception as exc:
            logger.debug("Connection error for %s: %s", user_id, exc)
        finally:
            await self.leave_room(user.user_id)
            self._users.pop(user.user_id, None)
            await self.broadcast("user_left", {"user_id": user.user_id})
            logger.info("User disconnected: %s", user_id)

    async def _handle_sync_request(self, user: CollabUser, data: dict) -> None:
        """Send the current file state back to the requesting user."""
        if user.room_id is None:
            return
        room = self._rooms.get(user.room_id)
        if room is None:
            return
        filename = data.get("file", "")
        content = room.files.get(filename, "")
        await self.send_to_user(
            user.user_id,
            "file_sync",
            {"file": filename, "content": content},
        )


# ---------------------------------------------------------------------------
# CollabClient
# ---------------------------------------------------------------------------

class CollabClient:
    """WebSocket client for the collaboration server."""

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url
        self._ws: Optional[Any] = None
        self._user_id: str = ""
        self._display_name: str = ""
        self._handlers: dict[str, Callable] = {}
        self._receiver_task: Optional[asyncio.Task] = None
        self._connected = False

    # -- lifecycle ---------------------------------------------------------

    async def connect(self, user_id: str, display_name: str) -> None:
        """Open a WebSocket connection and authenticate."""
        import websockets

        self._user_id = user_id
        self._display_name = display_name

        self._ws = await websockets.connect(self.server_url)
        self._connected = True

        # Send auth message
        await self._ws.send(
            _make_message(
                "auth",
                {"user_id": user_id, "display_name": display_name},
                sender=user_id,
            )
        )

        # Start background receiver
        self._receiver_task = asyncio.create_task(self._receive_loop())
        logger.info("Connected to %s as %s", self.server_url, display_name)

    async def disconnect(self) -> None:
        """Close the connection cleanly."""
        self._connected = False
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("Disconnected from %s", self.server_url)

    # -- actions -----------------------------------------------------------

    async def join_room(self, room_id: str) -> None:
        """Join a collaboration room."""
        await self._send("join_room", {"room_id": room_id})

    async def send_cursor(self, file: str, line: int, col: int) -> None:
        """Broadcast cursor position to room mates."""
        await self._send("cursor_move", {
            "file": file,
            "line": line,
            "col": col,
        })

    async def send_edit(self, file: str, changes: list[dict]) -> None:
        """Broadcast a set of text edits to room mates."""
        await self._send("file_edit", {
            "file": file,
            "changes": changes,
        })

    async def send_message(self, room_id: str, content: str) -> None:
        """Send a chat message to a room."""
        await self._send("chat_message", {
            "room_id": room_id,
            "content": content,
        })

    async def request_sync(self, file: str) -> None:
        """Request the latest content of *file* from the server."""
        await self._send("request_sync", {"file": file})

    # -- event registration ------------------------------------------------

    def on_event(
        self,
        event_type: str,
        handler: Callable[[dict], Coroutine[Any, Any, None]],
    ) -> None:
        """Register an async handler for *event_type*.

        Example::

            client.on_event("chat_message", my_handler)
        """
        self._handlers[event_type] = handler

    # -- internal ----------------------------------------------------------

    async def _send(self, event_type: str, data: dict) -> None:
        if self._ws is None:
            raise RuntimeError("Client is not connected")
        await self._ws.send(
            _make_message(event_type, data, sender=self._user_id)
        )

    async def _receive_loop(self) -> None:
        """Background loop that dispatches incoming events to handlers."""
        try:
            async for raw in self._ws:
                try:
                    message = _parse_message(raw)
                except json.JSONDecodeError:
                    continue
                event = message.get("event", "")
                data = message.get("data", {})
                sender = message.get("sender", "")

                handler = self._handlers.get(event)
                if handler:
                    payload = {**data, "_sender": sender}
                    asyncio.create_task(self._safe_call(handler, payload))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if self._connected:
                logger.error("Receive loop error: %s", exc)

    @staticmethod
    async def _safe_call(
        handler: Callable,
        payload: dict,
    ) -> None:
        """Invoke a handler, swallowing exceptions to keep the loop alive."""
        try:
            await handler(payload)
        except Exception as exc:
            logger.error("Event handler error: %s", exc, exc_info=True)
