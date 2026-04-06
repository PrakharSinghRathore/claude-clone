"""
Atlas Canvas / A2UI (Agent-to-UI) System.

Provides a visual workspace for agents to push UI elements,
render them in multiple formats, and manage canvas state
with real-time WebSocket updates.
"""

from .host import CanvasHost, CanvasState, CanvasClient
from .renderer import CanvasRenderer, RenderFormat, LayoutEngine
from .push import (
    A2UIPushManager,
    CanvasElement,
    CanvasUpdate,
    CanvasEventType,
    ElementType,
)

__all__ = [
    # Host
    "CanvasHost",
    "CanvasState",
    "CanvasClient",
    # Renderer
    "CanvasRenderer",
    "RenderFormat",
    "LayoutEngine",
    # Push
    "A2UIPushManager",
    "CanvasElement",
    "CanvasUpdate",
    "CanvasEventType",
    "ElementType",
]
