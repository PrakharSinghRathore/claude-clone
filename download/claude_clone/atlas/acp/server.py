"""
ACP Server — FastAPI-based Agent Communication Protocol server.

Provides REST and WebSocket endpoints for session management,
message handling, tool invocation, streaming events, and
editor/IDE integration.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .auth import AuthManager
from .events import EventManager, EventType
from .permissions import PermissionManager, PermissionLevel
from .session import SessionManager, SessionState

logger = logging.getLogger(__name__)


class ACPServer:
    """
    Agent Communication Protocol server.

    Coordinates authentication, sessions, events, and permissions
    into a unified API. Can be mounted on a FastAPI application
    or run standalone.

    Usage::

        server = ACPServer()
        app = server.create_app()  # Returns a FastAPI app

        # Or mount on existing app:
        app.mount("/acp", server.create_app())
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        data_dir: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.auth = AuthManager(secret_key=secret_key)
        self.sessions = SessionManager()
        self.events = EventManager()
        self.permissions = PermissionManager()
        self._app: Any = None
        self._message_handler: Any = None
        self._tool_handler: Any = None

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def set_message_handler(self, handler: Any) -> None:
        """
        Set the message handler callable.

        The handler receives (session_id, message_content, metadata)
        and should return a response string or dict.
        """
        self._message_handler = handler

    def set_tool_handler(self, handler: Any) -> None:
        """
        Set the tool invocation handler.

        The handler receives (session_id, tool_name, params)
        and should return a result.
        """
        self._tool_handler = handler

    # ------------------------------------------------------------------
    # Application creation
    # ------------------------------------------------------------------

    def create_app(self) -> Any:
        """
        Create and return a FastAPI application with all ACP routes.

        Raises ImportError if FastAPI is not installed.
        """
        try:
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel
        except ImportError:
            raise ImportError(
                "FastAPI is required for the ACP server. "
                "Install it with: pip install fastapi uvicorn"
            )

        app = FastAPI(
            title="Atlas ACP Server",
            description="Agent Communication Protocol server for Claude Clone",
            version="1.0.0",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Store reference for later use
        self._app = app

        # ---- Request/Response models ----

        class MessageRequest(BaseModel):
            content: str
            metadata: dict = {}

        class ToolCallRequest(BaseModel):
            tool: str
            params: dict = {}
            session_id: Optional[str] = None

        class PermissionSetRequest(BaseModel):
            tool: str
            level: str
            session_id: Optional[str] = None

        class TokenRefreshRequest(BaseModel):
            token: str

        class SessionCreateRequest(BaseModel):
            user_id: Optional[str] = None
            role: str = "user"
            metadata: dict = {}

        class TemplateApplyRequest(BaseModel):
            template_name: str
            session_id: Optional[str] = None

        class ExportRequest(BaseModel):
            session_id: str

        class ImportRequest(BaseModel):
            data: dict

        # ---- Authentication dependency ----

        async def get_current_user(
            authorization: Optional[str] = Header(None),
            x_api_key: Optional[str] = Header(None),
        ) -> dict:
            """Validate authentication from API key or Bearer token."""
            # Try Bearer token first
            if authorization and authorization.startswith("Bearer "):
                token_str = authorization[7:]
                token = self.auth.validate_token(token_str)
                if token:
                    return {"user_id": token.user_id, "role": token.role, "session_id": token.session_id}

            # Try API key
            if x_api_key:
                api_key = self.auth.validate_api_key(x_api_key)
                if api_key:
                    return {"user_id": api_key.name, "role": api_key.role, "session_id": None}

            raise HTTPException(status_code=401, detail="Invalid or missing authentication")

        # ---- REST endpoints ----

        @app.post("/auth/token")
        async def create_token(api_key: str):
            """Create an authentication token from an API key."""
            key_data = self.auth.validate_api_key(api_key)
            if key_data is None:
                raise HTTPException(status_code=401, detail="Invalid API key")
            token = self.auth.create_token(user_id=key_data.name, role=key_data.role)
            return {"token": token, "expires_in": 3600}

        @app.post("/auth/token/refresh")
        async def refresh_token(request: TokenRefreshRequest):
            """Refresh an existing token."""
            new_token = self.auth.refresh_token(request.token)
            if new_token is None:
                raise HTTPException(status_code=401, detail="Token cannot be refreshed")
            return {"token": new_token, "expires_in": 3600}

        @app.post("/auth/keys")
        async def create_api_key(name: str = "default", role: str = "user"):
            """Create a new API key."""
            from .auth import Role
            if role not in (Role.ADMIN, Role.USER, Role.READONLY, Role.TOOL, Role.IDE):
                raise HTTPException(status_code=400, detail=f"Invalid role: {role}")
            key = self.auth.create_api_key(name=name, role=role)
            return {"api_key": key, "name": name, "role": role}

        @app.get("/auth/keys")
        async def list_keys(user: dict = Depends(get_current_user)):
            """List all API keys."""
            return {"keys": self.auth.list_api_keys()}

        @app.delete("/auth/keys/{key_hash}")
        async def revoke_key(key_hash: str, user: dict = Depends(get_current_user)):
            """Revoke an API key."""
            # For simplicity, accept the full key and revoke it
            return {"revoked": self.auth.revoke_api_key(key_hash)}

        # ---- Session endpoints ----

        @app.post("/sessions")
        async def create_session(request: SessionCreateRequest, user: dict = Depends(get_current_user)):
            """Create a new ACP session."""
            session = await self.sessions.create_session(
                user_id=user["user_id"],
                role=request.role,
                metadata=request.metadata,
            )
            await self.events.emit(
                self.events.__class__.__bases__[0](
                    type=EventType.SESSION_CREATED,
                    data={"session_id": session.session_id},
                    session_id=session.session_id,
                )
            )
            return {"session_id": session.session_id, "state": session.state.value}

        @app.get("/sessions")
        async def list_sessions(user: dict = Depends(get_current_user)):
            """List active sessions."""
            sessions = await self.sessions.list_sessions(user_id=user["user_id"])
            return {
                "sessions": [
                    {"session_id": s.session_id, "state": s.state.value, "created_at": s.created_at}
                    for s in sessions
                ],
            }

        @app.get("/sessions/{session_id}")
        async def get_session(session_id: str, user: dict = Depends(get_current_user)):
            """Get session details."""
            session = await self.sessions.get_session(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            return session.to_dict()

        @app.post("/sessions/{session_id}/end")
        async def end_session(session_id: str, user: dict = Depends(get_current_user)):
            """End a session."""
            success = await self.sessions.end_session(session_id)
            if not success:
                raise HTTPException(status_code=404, detail="Session not found")
            return {"ended": True}

        @app.post("/sessions/export")
        async def export_session(request: ExportRequest, user: dict = Depends(get_current_user)):
            """Export a session for backup."""
            data = await self.sessions.export_session(request.session_id)
            if data is None:
                raise HTTPException(status_code=404, detail="Session not found")
            return data

        @app.post("/sessions/import")
        async def import_session(request: ImportRequest, user: dict = Depends(get_current_user)):
            """Import a session from backup data."""
            session = await self.sessions.import_session(request.data)
            return {"session_id": session.session_id, "imported": True}

        # ---- Message endpoints ----

        @app.post("/messages")
        async def send_message(request: MessageRequest, user: dict = Depends(get_current_user)):
            """Send a message and get a response."""
            session_id = request.metadata.get("session_id")

            if session_id:
                session = await self.sessions.get_session(session_id)
                if session:
                    session.add_message("user", request.content)

            # Emit thinking event
            await self.events.emit_thinking("Processing message...", session_id)

            # Call handler
            if self._message_handler:
                try:
                    response = self._message_handler(
                        session_id, request.content, request.metadata
                    )
                    if hasattr(response, "__await__"):
                        response = await response
                    await self.events.emit_message(
                        str(response), session_id
                    )
                    return {"response": response}
                except Exception as exc:
                    await self.events.emit_error(str(exc), session_id)
                    raise HTTPException(status_code=500, detail=str(exc))
            else:
                msg = "No message handler configured"
                await self.events.emit_error(msg, session_id)
                return {"response": msg, "note": "Configure a message handler"}

        # ---- Tool endpoints ----

        @app.post("/tools/call")
        async def call_tool(request: ToolCallRequest, user: dict = Depends(get_current_user)):
            """Invoke a tool with permission checking."""
            session_id = request.session_id

            # Check permissions
            perm = self.permissions.check_permission(request.tool, session_id)
            if perm == PermissionLevel.DENY:
                raise HTTPException(status_code=403, detail=f"Tool '{request.tool}' is denied")
            if perm == PermissionLevel.ASK:
                # Emit permission request event
                await self.events.emit(
                    self.events.__class__.__bases__[0](
                        type=EventType.PERMISSION_REQUEST,
                        data={"tool": request.tool, "params": request.params},
                        session_id=session_id,
                    )
                )
                return {
                    "status": "pending_permission",
                    "tool": request.tool,
                    "message": "Tool requires permission approval",
                }

            # Emit tool call event
            await self.events.emit_tool_call(request.tool, request.params, session_id)

            # Record tool call in session
            if session_id:
                session = await self.sessions.get_session(session_id)
                if session:
                    session.add_tool_call(request.tool, request.params)

            # Execute
            if self._tool_handler:
                try:
                    result = self._tool_handler(session_id, request.tool, request.params)
                    if hasattr(result, "__await__"):
                        result = await result
                    await self.events.emit_tool_result(request.tool, result, session_id)
                    return {"tool": request.tool, "result": result, "status": "completed"}
                except Exception as exc:
                    await self.events.emit_error(f"Tool error: {exc}", session_id)
                    return {"tool": request.tool, "error": str(exc), "status": "failed"}
            else:
                return {"tool": request.tool, "result": None, "status": "no_handler"}

        # ---- Permission endpoints ----

        @app.get("/permissions")
        async def get_permissions(
            session_id: Optional[str] = None,
            user: dict = Depends(get_current_user),
        ):
            """Get current permission configuration."""
            return {
                "permissions": self.permissions.get_permission_summary(session_id),
                "default_template": self.permissions._default_template_name,
            }

        @app.post("/permissions")
        async def set_permission(request: PermissionSetRequest, user: dict = Depends(get_current_user)):
            """Set a tool permission."""
            try:
                level = PermissionLevel(request.level)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid level: {request.level}")
            self.permissions.set_permission(request.tool, level, request.session_id)
            return {"tool": request.tool, "level": level.value}

        @app.get("/permissions/templates")
        async def list_templates(user: dict = Depends(get_current_user)):
            """List available permission templates."""
            return {
                "templates": {
                    name: {"name": t.name, "description": t.description, "is_default": t.is_default}
                    for name, t in self.permissions.list_templates().items()
                }
            }

        @app.post("/permissions/templates/apply")
        async def apply_template(request: TemplateApplyRequest, user: dict = Depends(get_current_user)):
            """Apply a permission template."""
            success = self.permissions.apply_template(request.template_name, request.session_id)
            if not success:
                raise HTTPException(status_code=404, detail=f"Template '{request.template_name}' not found")
            return {"applied": request.template_name}

        # ---- Event endpoints ----

        @app.get("/events/history")
        async def get_event_history(
            session_id: Optional[str] = None,
            event_type: Optional[str] = None,
            limit: int = 100,
            user: dict = Depends(get_current_user),
        ):
            """Get event history."""
            ev_type = EventType(event_type) if event_type else None
            events = self.events.get_history(session_id=session_id, event_type=ev_type, limit=limit)
            return {
                "events": [
                    {
                        "id": e.event_id,
                        "type": e.type.value,
                        "data": e.data,
                        "timestamp": e.timestamp,
                    }
                    for e in events
                ],
            }

        @app.get("/events/stats")
        async def get_event_stats(user: dict = Depends(get_current_user)):
            """Get event system statistics."""
            return self.events.stats()

        # ---- IDE integration endpoint ----

        @app.get("/ide/status")
        async def ide_status(user: dict = Depends(get_current_user)):
            """IDE integration status endpoint."""
            return {
                "status": "connected",
                "version": "1.0.0",
                "sessions": len(self.sessions._sessions),
                "events": self.events.stats(),
            }

        # ---- WebSocket endpoint ----

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time event streaming."""
            await websocket.accept()

            # Authenticate via query param or first message
            params = websocket.query_params
            api_key = params.get("api_key") or params.get("token")

            if api_key:
                if api_key.startswith("hcp_"):
                    validated = self.auth.validate_api_key(api_key)
                else:
                    validated = self.auth.validate_token(api_key)
                if not validated:
                    await websocket.close(code=4001, reason="Authentication failed")
                    return
            else:
                await websocket.close(code=4001, reason="Missing authentication")
                return

            # Subscribe to all events
            sub_id = await self.events.subscribe()

            try:
                while True:
                    # Send queued events
                    event = await self.events.get_event(sub_id, timeout=5.0)
                    if event:
                        await websocket.send_text(event.to_json())

                    # Check for incoming messages
                    try:
                        data = await asyncio_wait_with_timeout(websocket.receive_text(), timeout=0.01)
                        if data:
                            # Client sent a message; echo via events
                            await self.events.emit(
                                self.events.__class__.__bases__[0](
                                    type=EventType.MESSAGE,
                                    data={"content": data, "source": "websocket"},
                                )
                            )
                    except Exception:
                        pass

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception:
                logger.exception("WebSocket error")
            finally:
                await self.events.unsubscribe(sub_id)

        # ---- Health check ----

        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "service": "atlas-acp",
                "version": "1.0.0",
            }

        return app

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the server using uvicorn."""
        try:
            import uvicorn
        except ImportError:
            raise ImportError("uvicorn is required. Install with: pip install uvicorn")

        app = self.create_app()
        await uvicorn.serve(app, host=self.host, port=self.port, log_level="info")


# ------------------------------------------------------------------
# WebSocket helper
# ------------------------------------------------------------------

import asyncio


async def asyncio_wait_with_timeout(coro: Any, timeout: float) -> Any:
    """Run a coroutine with a timeout, returning None on timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return None
