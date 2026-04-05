"""
Claude Clone Plugin System

A dynamic plugin system with hot-reload, dependency management,
and a built-in template generator for creating new plugins.
"""

from .loader import Plugin, PluginManager, PluginTool, PluginHook

__all__ = [
    "Plugin",
    "PluginManager",
    "PluginTool",
    "PluginHook",
]
