"""
Atlas ACP (Agent Communication Protocol) Adapter.

Provides a FastAPI-based server for external tool and agent
communication, with session management, authentication,
permission control, WebSocket streaming, and IDE integration.
"""

from .server import ACPServer
from .auth import AuthManager
from .session import SessionManager
from .events import EventManager, EventType
from .permissions import PermissionManager

__all__ = [
    "ACPServer",
    "AuthManager",
    "SessionManager",
    "EventManager",
    "EventType",
    "PermissionManager",
]
