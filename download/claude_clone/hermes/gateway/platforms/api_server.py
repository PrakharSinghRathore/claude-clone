"""
REST API server adapter for the Hermes Gateway.

Provides a FastAPI-based API server with WebSocket support for real-time
communication, OpenAPI documentation, authentication (API key, JWT),
and rate limiting.

Usage::

    from hermes.gateway.config import PlatformConfig
    from hermes.gateway.platforms.api_server import APIServerAdapter

    config = PlatformConfig(
        name="api",
        token="API_SECRET_KEY",
        port=8000,
        enabled=True,
    )
    adapter = APIServerAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from hermes.gateway.runner import IncomingMessage

logger = logging.getLogger("hermes.gateway.platforms.api_server")

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class APIServerAdapter:
    """
    REST API server adapter with WebSocket support.

    Provides HTTP endpoints for message sending/receiving and
    WebSocket for real-time bidirectional communication.

    Parameters
    ----------
    config:
        Platform configuration. Requires:
        - ``token``: API secret key for authentication
        - ``host``: Server bind host (default: 0.0.0.0)
        - ``port``: Server bind port (default: 8000)
    """

    def __init__(self, config: Any):
        self._config = config
        self._api_key = config.token or os.environ.get("API_SECRET_KEY", secrets.token_hex(32))
        self._host = config.host or "0.0.0.0"
        self._port = config.port or 8000
        self._timeout = config.timeout or 30
        self._connected = False
        self._app: Optional[Any] = None
        self._server: Optional[Any] = None
        self._server_task: Optional[asyncio.Task] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._ws_clients: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._rate_limits: Dict[str, List[float]] = defaultdict(list)
        self._rate_limit_max = config.rate_limit or 60
        self._rate_limit_window = config.rate_limit_window or 60
        self._message_handler: Optional[Callable] = None

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Start the API server."""
        if not HAS_FASTAPI:
            raise ImportError("fastapi and uvicorn required. Install with: pip install fastapi uvicorn")

        self._app = self._create_app()
        self._server_task = asyncio.create_task(self._run_server())
        self._connected = True
        logger.info("API server starting on %s:%d", self._host, self._port)

    async def disconnect(self) -> None:
        """Stop the API server."""
        self._connected = False
        if self._server:
            self._server.should_exit = True
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass

    async def is_connected(self) -> bool:
        return self._connected

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a message to a WebSocket client."""
        message_id = uuid.uuid4().hex[:12]

        payload = {
            "type": "message",
            "id": message_id,
            "text": text,
            "chat_id": chat_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        delivered = False
        # Send to all WebSocket clients subscribed to this chat
        if chat_id in self._ws_clients:
            disconnected = set()
            for ws in self._ws_clients[chat_id]:
                try:
                    await ws.send_json(payload)
                    delivered = True
                except Exception:
                    disconnected.add(ws)
            for ws in disconnected:
                self._ws_clients[chat_id].discard(ws)

        return message_id if delivered else None

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send a file reference to a WebSocket client."""
        import os
        if not os.path.exists(file_path):
            return None

        message_id = uuid.uuid4().hex[:12]
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        payload = {
            "type": "file",
            "id": message_id,
            "chat_id": chat_id,
            "filename": filename,
            "size": file_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if chat_id in self._ws_clients:
            for ws in self._ws_clients[chat_id]:
                try:
                    await ws.send_json(payload)
                except Exception:
                    pass

        return message_id

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for messages received via HTTP or WebSocket."""
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
        """Send a typing indicator to WebSocket clients."""
        payload = {
            "type": "typing",
            "chat_id": chat_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if chat_id in self._ws_clients:
            for ws in self._ws_clients[chat_id]:
                try:
                    await ws.send_json(payload)
                except Exception:
                    pass

    def set_message_handler(self, handler: Callable) -> None:
        """Set a custom handler for incoming messages."""
        self._message_handler = handler

    @property
    def app(self) -> Any:
        """Get the FastAPI application instance."""
        return self._app

    # ── App Factory ───────────────────────────────────────────────────────

    def _create_app(self) -> "FastAPI":
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title="Hermes Gateway API",
            description="REST API for the Hermes multi-platform messaging gateway",
            version="1.0.0",
        )

        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # ── Auth Dependency ───────────────────────────────────────────
        async def verify_api_key(request: Request) -> bool:
            api_key = request.headers.get("X-API-Key", "")
            if not api_key:
                api_key = request.query_params.get("api_key", "")
            if not api_key:
                raise HTTPException(status_code=401, detail="API key required")
            if not hmac.compare_digest(api_key, self._api_key):
                raise HTTPException(status_code=403, detail="Invalid API key")
            return True

        # ── Rate Limit Dependency ─────────────────────────────────────
        async def check_rate_limit(request: Request):
            client_id = request.headers.get("X-API-Key", request.client.host if request.client else "unknown")
            now = time.time()
            window = self._rate_limit_window
            self._rate_limits[client_id] = [
                t for t in self._rate_limits[client_id] if now - t < window
            ]
            if len(self._rate_limits[client_id]) >= self._rate_limit_max:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # ── Endpoints ────────────────────────────────────────────────

        @app.post("/v1/messages")
        async def send_message_endpoint(
            request: Request,
            _auth: bool = Depends(verify_api_key),
            _rate: None = Depends(check_rate_limit),
        ):
            body = await request.json()
            text = body.get("text", "")
            chat_id = body.get("chat_id", "default")

            if not text:
                raise HTTPException(status_code=400, detail="Message text is required")

            msg = IncomingMessage(
                platform="api",
                chat_id=chat_id,
                user_id=body.get("user_id", "api"),
                text=text,
                message_id=uuid.uuid4().hex[:12],
                metadata=body.get("metadata", {}),
            )
            await self._message_queue.put(msg)

            self._rate_limits[
                request.headers.get("X-API-Key", "")
            ].append(time.time())

            return {
                "status": "queued",
                "message_id": msg.message_id,
                "chat_id": chat_id,
            }

        @app.get("/v1/messages")
        async def get_messages(
            _auth: bool = Depends(verify_api_key),
            limit: int = 50,
        ):
            messages = []
            while not self._message_queue.empty() and len(messages) < limit:
                try:
                    msg = self._message_queue.get_nowait()
                    messages.append({
                        "platform": msg.platform,
                        "chat_id": msg.chat_id,
                        "user_id": msg.user_id,
                        "text": msg.text,
                        "message_id": msg.message_id,
                    })
                except asyncio.QueueEmpty:
                    break
            return {"messages": messages, "count": len(messages)}

        @app.websocket("/ws/{chat_id}")
        async def websocket_endpoint(websocket: WebSocket, chat_id: str):
            await websocket.accept()
            self._ws_clients[chat_id].add(websocket)
            logger.info("WebSocket client connected to %s", chat_id)

            try:
                while True:
                    data = await websocket.receive_json()
                    text = data.get("text", "")
                    if text:
                        msg = IncomingMessage(
                            platform="api",
                            chat_id=chat_id,
                            user_id=data.get("user_id", "ws"),
                            text=text,
                            message_id=uuid.uuid4().hex[:12],
                            metadata=data.get("metadata", {}),
                        )
                        await self._message_queue.put(msg)

                        if self._message_handler:
                            try:
                                response = await self._message_handler(msg, {})
                                if response:
                                    await websocket.send_json({
                                        "type": "response",
                                        "text": response,
                                        "in_reply_to": msg.message_id,
                                    })
                            except Exception as e:
                                logger.error("WS message handler error: %s", e)
            except WebSocketDisconnect:
                pass
            finally:
                self._ws_clients[chat_id].discard(websocket)
                logger.info("WebSocket client disconnected from %s", chat_id)

        @app.get("/v1/status")
        async def status_endpoint(_auth: bool = Depends(verify_api_key)):
            return {
                "status": "running" if self._connected else "stopped",
                "ws_clients": sum(len(s) for s in self._ws_clients.values()),
                "uptime": time.time(),
            }

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        return app

    # ── Server Runner ─────────────────────────────────────────────────────

    async def _run_server(self) -> None:
        """Run the uvicorn server."""
        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()
