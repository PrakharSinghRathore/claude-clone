"""
Atlas Canvas Host — Manages the visual workspace for Agent-to-UI rendering.

The CanvasHost creates, destroys, and manages canvas instances, handles
WebSocket transport for real-time updates, maintains canvas state, and
provides broadcast capabilities to connected clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CanvasStatus(Enum):
    """Status of a canvas."""
    CREATING = "creating"
    ACTIVE = "active"
    PAUSED = "paused"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"


@dataclass
class CanvasState:
    """Complete state of a canvas."""
    name: str
    width: int = 1024
    height: int = 768
    elements: List[Dict[str, Any]] = field(default_factory=list)
    styles: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    status: CanvasStatus = CanvasStatus.ACTIVE
    version: int = 0
    background_color: str = "#ffffff"
    title: str = ""
    layout: str = "flexbox"  # flexbox, grid, absolute

    def to_dict(self) -> Dict[str, Any]:
        """Serialize canvas state to dictionary."""
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "elements": self.elements,
            "styles": self.styles,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "version": self.version,
            "background_color": self.background_color,
            "title": self.title,
            "layout": self.layout,
            "element_count": len(self.elements),
        }


@dataclass
class CanvasClient:
    """A connected client to a canvas."""
    client_id: str
    canvas_name: str
    connected_at: float = 0.0
    last_heartbeat: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    pending_updates: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "canvas_name": self.canvas_name,
            "connected_at": self.connected_at,
            "last_heartbeat": self.last_heartbeat,
            "metadata": self.metadata,
            "pending_updates": self.pending_updates,
        }


@dataclass
class CanvasEvent:
    """An event to broadcast to clients."""
    event_type: str
    canvas_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "canvas_name": self.canvas_name,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class CanvasError(Exception):
    """Raised for canvas-related errors."""
    pass


class CanvasNotFoundError(CanvasError):
    """Raised when a canvas is not found."""
    pass


class CanvasAlreadyExistsError(CanvasError):
    """Raised when a canvas with the same name already exists."""
    pass


class ClientNotFoundError(CanvasError):
    """Raised when a client is not found."""
    pass


class CanvasHost:
    """
    Manages visual workspaces (canvases) for Agent-to-UI rendering.

    Provides canvas lifecycle management, WebSocket transport for real-time
    updates, state management, and broadcast capabilities. Each canvas is
    isolated with its own state and connected clients.

    Example:
        >>> host = CanvasHost()
        >>> canvas = await host.create_canvas("workspace", 1200, 800)
        >>> await host.push_update("workspace", {"type": "text", "content": "Hello"})
        >>> state = await host.get_state("workspace")
    """

    def __init__(
        self,
        max_canvases: int = 50,
        max_clients_per_canvas: int = 100,
        heartbeat_interval: float = 30.0,
        cleanup_interval: float = 60.0,
        max_state_history: int = 100,
    ) -> None:
        """
        Initialize the CanvasHost.

        Args:
            max_canvases: Maximum number of active canvases.
            max_clients_per_canvas: Maximum clients per canvas.
            heartbeat_interval: Client heartbeat check interval in seconds.
            cleanup_interval: Cleanup dead connections interval in seconds.
            max_state_history: Maximum state versions to keep per canvas.
        """
        self._canvases: Dict[str, CanvasState] = {}
        self._clients: Dict[str, CanvasClient] = {}  # client_id -> CanvasClient
        self._canvas_clients: Dict[str, Set[str]] = {}  # canvas_name -> set of client_ids
        self._update_queues: Dict[str, asyncio.Queue] = {}  # canvas_name -> queue
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._state_history: Dict[str, List[Dict[str, Any]]] = {}

        self._max_canvases = max_canvases
        self._max_clients_per_canvas = max_clients_per_canvas
        self._heartbeat_interval = heartbeat_interval
        self._cleanup_interval = cleanup_interval
        self._max_state_history = max_state_history

        self._stats = {
            "total_canvases_created": 0,
            "total_canvases_destroyed": 0,
            "total_clients_connected": 0,
            "total_updates_pushed": 0,
            "total_events_broadcast": 0,
        }

        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info("CanvasHost initialized: max_canvases=%d", max_canvases)

    @property
    def canvas_count(self) -> int:
        """Number of active canvases."""
        return len(self._canvases)

    @property
    def client_count(self) -> int:
        """Total number of connected clients."""
        return len(self._clients)

    async def start(self) -> None:
        """Start background tasks (heartbeat, cleanup)."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("CanvasHost heartbeat task started")

        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("CanvasHost cleanup task started")

    async def stop(self) -> None:
        """Stop background tasks and clean up."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

        # Disconnect all clients
        for client_id in list(self._clients.keys()):
            await self._disconnect_client(client_id)

        logger.info("CanvasHost stopped")

    async def create_canvas(
        self,
        name: str,
        width: int = 1024,
        height: int = 768,
        title: str = "",
        layout: str = "flexbox",
        background_color: str = "#ffffff",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CanvasState:
        """
        Create a new canvas.

        Args:
            name: Unique name for the canvas.
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            title: Canvas title.
            layout: Layout engine (flexbox, grid, absolute).
            background_color: Background color CSS value.
            metadata: Optional metadata dictionary.

        Returns:
            The created CanvasState.

        Raises:
            CanvasAlreadyExistsError: If a canvas with the same name exists.
            CanvasError: If max canvases reached.
        """
        if name in self._canvases:
            raise CanvasAlreadyExistsError(f"Canvas '{name}' already exists")

        if len(self._canvases) >= self._max_canvases:
            raise CanvasError(
                f"Maximum canvases reached ({self._max_canvases}). "
                f"Destroy a canvas before creating a new one."
            )

        now = time.time()
        canvas = CanvasState(
            name=name,
            width=max(100, min(4096, width)),
            height=max(100, min(4096, height)),
            title=title or name,
            layout=layout,
            background_color=background_color,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        self._canvases[name] = canvas
        self._canvas_clients[name] = set()
        self._update_queues[name] = asyncio.Queue(maxsize=1000)
        self._state_history[name] = []
        self._stats["total_canvases_created"] += 1

        # Save initial state to history
        await self._save_state_history(name)

        logger.info(
            "Created canvas '%s': %dx%d, layout=%s",
            name, width, height, layout,
        )

        # Notify event handlers
        await self._emit_event("canvas_created", name, {"width": width, "height": height})

        return canvas

    async def destroy_canvas(self, name: str, force: bool = False) -> bool:
        """
        Destroy a canvas and disconnect all its clients.

        Args:
            name: Canvas name.
            force: Force destroy even with connected clients.

        Returns:
            True if the canvas was destroyed, False otherwise.
        """
        if name not in self._canvases:
            raise CanvasNotFoundError(f"Canvas '{name}' not found")

        canvas = self._canvases[name]
        clients = self._canvas_clients.get(name, set())

        if clients and not force:
            logger.warning(
                "Canvas '%s' has %d connected clients. Use force=True to destroy.",
                name, len(clients),
            )
            return False

        # Disconnect all clients
        for client_id in list(clients):
            await self._disconnect_client(client_id)

        # Update status
        canvas.status = CanvasStatus.DESTROYING
        await self._emit_event("canvas_destroying", name, {})

        # Remove canvas
        del self._canvases[name]
        if name in self._canvas_clients:
            del self._canvas_clients[name]
        if name in self._update_queues:
            del self._update_queues[name]
        if name in self._state_history:
            del self._state_history[name]

        self._stats["total_canvases_destroyed"] += 1
        logger.info("Destroyed canvas '%s'", name)

        await self._emit_event("canvas_destroyed", name, {})
        return True

    async def list_canvases(self) -> List[Dict[str, Any]]:
        """
        List all active canvases with summary info.

        Returns:
            List of canvas summary dictionaries.
        """
        result = []
        for name, canvas in self._canvases.items():
            client_count = len(self._canvas_clients.get(name, set()))
            result.append({
                "name": canvas.name,
                "width": canvas.width,
                "height": canvas.height,
                "status": canvas.status.value,
                "element_count": len(canvas.elements),
                "client_count": client_count,
                "version": canvas.version,
                "title": canvas.title,
                "created_at": canvas.created_at,
                "updated_at": canvas.updated_at,
            })
        return result

    async def get_state(self, name: str, version: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the current state of a canvas.

        Args:
            name: Canvas name.
            version: Specific version to retrieve. None for current.

        Returns:
            Canvas state dictionary.

        Raises:
            CanvasNotFoundError: If the canvas doesn't exist.
        """
        if name not in self._canvases:
            raise CanvasNotFoundError(f"Canvas '{name}' not found")

        if version is not None:
            history = self._state_history.get(name, [])
            for state in reversed(history):
                if state.get("version") == version:
                    return state
            raise CanvasError(f"Version {version} not found for canvas '{name}'")

        return self._canvases[name].to_dict()

    async def push_update(
        self,
        canvas_name: str,
        update: Dict[str, Any],
        source: str = "agent",
    ) -> None:
        """
        Push an A2UI update to a canvas.

        The update can add, modify, or remove elements on the canvas.

        Args:
            canvas_name: Target canvas name.
            update: Update data with action, element info, etc.
            source: Source identifier (agent, system, user).

        Raises:
            CanvasNotFoundError: If the canvas doesn't exist.
        """
        if canvas_name not in self._canvases:
            raise CanvasNotFoundError(f"Canvas '{canvas_name}' not found")

        canvas = self._canvases[canvas_name]
        action = update.get("action", "add")

        # Apply the update to canvas state
        if action == "add":
            element = update.get("element", update)
            element["_id"] = element.get("_id", f"el_{uuid.uuid4().hex[:8]}")
            element["_added_at"] = time.time()
            canvas.elements.append(element)
        elif action == "update":
            element_id = update.get("element_id", update.get("_id"))
            properties = update.get("properties", update.get("changes", {}))
            for el in canvas.elements:
                if el.get("_id") == element_id:
                    el.update(properties)
                    el["_updated_at"] = time.time()
                    break
        elif action == "remove":
            element_id = update.get("element_id", update.get("_id"))
            canvas.elements = [
                el for el in canvas.elements
                if el.get("_id") != element_id
            ]
        elif action == "clear":
            canvas.elements.clear()
        elif action == "set_state":
            # Replace entire state
            new_elements = update.get("elements", [])
            canvas.elements = new_elements
            if "styles" in update:
                canvas.styles.update(update["styles"])
            if "title" in update:
                canvas.title = update["title"]
            if "background_color" in update:
                canvas.background_color = update["background_color"]

        canvas.version += 1
        canvas.updated_at = time.time()

        # Save to history
        await self._save_state_history(canvas_name)

        # Push to update queue for WebSocket broadcasting
        queue = self._update_queues.get(canvas_name)
        if queue and not queue.full():
            await queue.put({
                "type": "canvas_update",
                "canvas_name": canvas_name,
                "update": update,
                "version": canvas.version,
                "timestamp": canvas.updated_at,
                "source": source,
            })

        self._stats["total_updates_pushed"] += 1

        # Emit event
        await self._emit_event("canvas_updated", canvas_name, {
            "action": action,
            "version": canvas.version,
        })

    async def broadcast(self, event: CanvasEvent) -> int:
        """
        Broadcast an event to all connected clients.

        Args:
            event: The CanvasEvent to broadcast.

        Returns:
            Number of clients that received the event.
        """
        if event.canvas_name:
            # Broadcast to specific canvas clients
            client_ids = self._canvas_clients.get(event.canvas_name, set())
        else:
            # Broadcast to all clients
            client_ids = set(self._clients.keys())

        count = 0
        for client_id in client_ids:
            client = self._clients.get(client_id)
            if client:
                queue = self._update_queues.get(
                    client.canvas_name if event.canvas_name is None else event.canvas_name
                )
                if queue:
                    try:
                        queue.put_nowait(event.to_dict())
                        count += 1
                    except asyncio.QueueFull:
                        logger.warning("Update queue full for client %s", client_id)

        self._stats["total_events_broadcast"] += 1
        return count

    async def connect_client(
        self,
        canvas_name: str,
        client_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CanvasClient:
        """
        Register a new client connection to a canvas.

        Args:
            canvas_name: Canvas to connect to.
            client_id: Optional client ID. Auto-generated if None.
            metadata: Optional client metadata.

        Returns:
            The connected CanvasClient.

        Raises:
            CanvasNotFoundError: If the canvas doesn't exist.
            CanvasError: If max clients reached.
        """
        if canvas_name not in self._canvases:
            raise CanvasNotFoundError(f"Canvas '{canvas_name}' not found")

        clients = self._canvas_clients.get(canvas_name, set())
        if len(clients) >= self._max_clients_per_canvas:
            raise CanvasError(
                f"Max clients ({self._max_clients_per_canvas}) reached for canvas '{canvas_name}'"
            )

        client_id = client_id or f"client_{uuid.uuid4().hex[:12]}"
        now = time.time()

        client = CanvasClient(
            client_id=client_id,
            canvas_name=canvas_name,
            connected_at=now,
            last_heartbeat=now,
            metadata=metadata or {},
        )

        self._clients[client_id] = client
        self._canvas_clients[canvas_name].add(client_id)
        self._stats["total_clients_connected"] += 1

        logger.info("Client %s connected to canvas '%s'", client_id, canvas_name)

        # Send current state to the new client
        queue = self._update_queues.get(canvas_name)
        if queue:
            state = self._canvases[canvas_name].to_dict()
            try:
                queue.put_nowait({
                    "type": "canvas_state",
                    "canvas_name": canvas_name,
                    "state": state,
                    "timestamp": now,
                })
            except asyncio.QueueFull:
                logger.warning("Queue full, state not sent to client %s", client_id)

        await self._emit_event("client_connected", canvas_name, {"client_id": client_id})
        return client

    async def disconnect_client(self, client_id: str) -> bool:
        """
        Disconnect a client from its canvas.

        Args:
            client_id: Client ID to disconnect.

        Returns:
            True if the client was disconnected.
        """
        return await self._disconnect_client(client_id)

    async def _disconnect_client(self, client_id: str) -> bool:
        """Internal client disconnection."""
        client = self._clients.pop(client_id, None)
        if not client:
            return False

        canvas_name = client.canvas_name
        if canvas_name in self._canvas_clients:
            self._canvas_clients[canvas_name].discard(client_id)

        logger.info("Client %s disconnected from canvas '%s'", client_id, canvas_name)
        await self._emit_event("client_disconnected", canvas_name, {"client_id": client_id})
        return True

    async def get_update_stream(self, canvas_name: str) -> Optional[asyncio.Queue]:
        """
        Get the update queue for a canvas (for WebSocket consumption).

        Args:
            canvas_name: Canvas name.

        Returns:
            AsyncQueue of updates or None if canvas doesn't exist.
        """
        return self._update_queues.get(canvas_name)

    async def on(self, event_type: str, handler: Callable) -> None:
        """
        Register an event handler.

        Args:
            event_type: Event type to listen for.
            handler: Async callback function.
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        logger.debug("Registered handler for event: %s", event_type)

    async def off(self, event_type: str, handler: Optional[Callable] = None) -> None:
        """
        Remove an event handler.

        Args:
            event_type: Event type.
            handler: Specific handler to remove. None removes all for the type.
        """
        if event_type in self._event_handlers:
            if handler is None:
                del self._event_handlers[event_type]
            else:
                self._event_handlers[event_type] = [
                    h for h in self._event_handlers[event_type] if h != handler
                ]

    async def get_canvas_clients(self, canvas_name: str) -> List[Dict[str, Any]]:
        """
        Get all clients connected to a canvas.

        Args:
            canvas_name: Canvas name.

        Returns:
            List of client info dictionaries.
        """
        client_ids = self._canvas_clients.get(canvas_name, set())
        return [
            self._clients[cid].to_dict()
            for cid in client_ids
            if cid in self._clients
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get host statistics."""
        return {
            **self._stats,
            "active_canvases": len(self._canvases),
            "total_connected_clients": len(self._clients),
            "canvas_details": {
                name: {
                    "elements": len(canvas.elements),
                    "clients": len(self._canvas_clients.get(name, set())),
                    "version": canvas.version,
                }
                for name, canvas in self._canvases.items()
            },
        }

    # ── Private methods ──────────────────────────────────────────

    async def _emit_event(self, event_type: str, canvas_name: str, data: Dict[str, Any]) -> None:
        """Emit an event to registered handlers."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                result = handler(event_type, canvas_name, data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Event handler error for '%s': %s", event_type, e)

    async def _save_state_history(self, canvas_name: str) -> None:
        """Save current state to history."""
        if canvas_name not in self._state_history:
            self._state_history[canvas_name] = []

        history = self._state_history[canvas_name]
        history.append(self._canvases[canvas_name].to_dict())

        # Trim to max history size
        while len(history) > self._max_state_history:
            history.pop(0)

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat check for clients."""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                now = time.time()
                dead_clients = []

                for client_id, client in self._clients.items():
                    if now - client.last_heartbeat > self._heartbeat_interval * 2:
                        dead_clients.append(client_id)

                for client_id in dead_clients:
                    logger.warning("Client %s heartbeat timeout, disconnecting", client_id)
                    await self._disconnect_client(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat loop error: %s", e)

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale resources."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)

                # Clean empty queues
                for name, queue in list(self._update_queues.items()):
                    if name not in self._canvases:
                        del self._update_queues[name]

                logger.debug("Canvas cleanup completed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error: %s", e)

    async def __aenter__(self) -> "CanvasHost":
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()
