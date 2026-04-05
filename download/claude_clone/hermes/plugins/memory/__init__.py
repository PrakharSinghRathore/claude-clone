"""
Memory Plugin System — Plugin loader, registry, and base interface.

Discovers memory plugins from YAML manifests, dynamically loads them,
and provides a unified registry for health checks and configuration.
"""

from .base import BaseMemoryPlugin, MemoryPluginMetadata, MemoryConfig
from .registry import MemoryPluginRegistry

__all__ = [
    "BaseMemoryPlugin",
    "MemoryPluginMetadata",
    "MemoryConfig",
    "MemoryPluginRegistry",
]
