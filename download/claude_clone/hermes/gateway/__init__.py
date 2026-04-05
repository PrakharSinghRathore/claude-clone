"""
Hermes Gateway — Multi-platform messaging gateway.

Orchestrates message routing, session management, delivery, streaming,
authentication, and cross-platform mirroring for Claude Clone.

Core classes:
    - GatewayRunner: Main orchestrator for platform adapters
    - SessionStore: Per-user conversation state management
    - DeliveryRouter: Routes messages to appropriate platforms
    - StreamConsumer: Streams agent responses to connected platforms
    - HookSystem: Extensible pre/post message hooks
    - PairingManager: DM pairing and authentication
    - MessageMirror: Cross-platform message mirroring
    - GatewayStatus: Health checks and status reporting

Usage::

    from hermes.gateway import GatewayRunner, GatewayConfig

    config = GatewayConfig.load("gateway.yaml")
    runner = GatewayRunner(config)
    await runner.start()
"""

from hermes.gateway.config import GatewayConfig, PlatformConfig
from hermes.gateway.runner import GatewayRunner
from hermes.gateway.session import SessionStore, SessionContext, SessionResetPolicy
from hermes.gateway.delivery import DeliveryRouter
from hermes.gateway.stream_consumer import StreamConsumer
from hermes.gateway.hooks import HookSystem, HookType
from hermes.gateway.pairing import PairingManager, PairingRole
from hermes.gateway.mirror import MessageMirror, MirrorDirection
from hermes.gateway.status import GatewayStatus

__all__ = [
    # Configuration
    "GatewayConfig",
    "PlatformConfig",
    # Core
    "GatewayRunner",
    # Sessions
    "SessionStore",
    "SessionContext",
    "SessionResetPolicy",
    # Delivery
    "DeliveryRouter",
    # Streaming
    "StreamConsumer",
    # Hooks
    "HookSystem",
    "HookType",
    # Pairing
    "PairingManager",
    "PairingRole",
    # Mirroring
    "MessageMirror",
    "MirrorDirection",
    # Status
    "GatewayStatus",
]

__version__ = "1.0.0"
