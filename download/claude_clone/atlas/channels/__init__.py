"""
Atlas Channels — Channel Abstraction Layer.

Provides a comprehensive multi-channel messaging abstraction supporting 25+
platforms with unified message handling, routing, and adapter management.

Modules:
    base — Enums, dataclasses, and BaseChannel abstract class
    adapter — ChannelAdapter for multi-channel lifecycle management
    routing — MessageRouter for intelligent message routing
    bindings — ChannelBindings for persistent channel-to-agent mappings

Usage::

    from atlas.channels import (
        ChannelType, ChannelState, MessageDirection,
        ChannelMessage, Attachment, ChannelConfig,
        BaseChannel, RateLimiter,
        ChannelAdapter, AdapterStats,
        MessageRouter, RouteRule, RouteMatchType, RoutingResult,
        ChannelBindings, BindingEntry,
    )
"""

from atlas.channels.base import (
    Attachment,
    AttachmentType,
    BaseChannel,
    ChannelConfig,
    ChannelMessage,
    ChannelState,
    ChannelType,
    MessageCallback,
    MessageDirection,
    RateLimiter,
)
from atlas.channels.adapter import AdapterStats, ChannelAdapter
from atlas.channels.routing import (
    AccountBinding,
    MessageHandler,
    MessageRouter,
    RouteMatchType,
    RouteRule,
    RoutingDecision,
    RoutingResult,
)
from atlas.channels.bindings import BindingEntry, ChannelBindings

__all__ = [
    # Enums
    "ChannelType",
    "ChannelState",
    "MessageDirection",
    "AttachmentType",
    "RouteMatchType",
    "RoutingDecision",
    # Dataclasses
    "Attachment",
    "ChannelMessage",
    "ChannelConfig",
    "RouteRule",
    "AccountBinding",
    "RoutingResult",
    "AdapterStats",
    "BindingEntry",
    # Classes
    "BaseChannel",
    "RateLimiter",
    "ChannelAdapter",
    "MessageRouter",
    "ChannelBindings",
    # Type aliases
    "MessageCallback",
    "MessageHandler",
]
