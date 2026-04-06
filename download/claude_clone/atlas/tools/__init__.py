"""
Atlas Tools System — self-registering tool registry with 18+ tool implementations.

Every tool module in this package calls ``ToolRegistry.instance().register()``
at import time, so simply importing this package activates all tools.

Usage::

    from atlas.tools import ToolRegistry

    # All tools are already registered
    reg = ToolRegistry.instance()
    print(reg)                    # <ToolRegistry 60+ tools (60+ enabled, 10 toolsets)>

    # Get Anthropic-format schemas
    schemas = reg.get_schemas()

    # Get {name: func} dict (compatible with Agent constructor)
    tools = reg.get_tools_dict()

    # Dispatch a tool call
    result = await reg.dispatch("atlas_web_search", query="python async")

    # Enable/disable at runtime
    reg.disable("atlas_exec_bash")
    reg.enable_toolset("web")
"""

from atlas.tools.registry import ToolRegistry, ToolDefinition

# ──────────────────────────────────────────────
# Auto-discover and import all tool modules.
# Each module self-registers its tools on import.
# ──────────────────────────────────────────────

_TOOL_MODULES = [
    "atlas.tools.terminal_tool",
    "atlas.tools.web_tools",
    "atlas.tools.browser_tool",
    "atlas.tools.file_tools",
    "atlas.tools.memory_tool",
    "atlas.tools.session_search",
    "atlas.tools.skills_tool",
    "atlas.tools.skill_manager",
    "atlas.tools.skills_hub",
    "atlas.tools.code_execution",
    "atlas.tools.delegate_tool",
    "atlas.tools.mcp_tool",
    "atlas.tools.tts_tool",
    "atlas.tools.transcription_tool",
    "atlas.tools.vision_tool",
    "atlas.tools.cronjob_tool",
    "atlas.tools.send_message_tool",
    "atlas.tools.image_gen_tool",
    "atlas.tools.todo_tool",
    "atlas.tools.mixture_of_agents_tool",
]


def discover_tools() -> ToolRegistry:
    """Import all tool modules to trigger self-registration.

    Returns the global ToolRegistry singleton with all tools loaded.
    """
    import importlib
    import sys

    for module_name in _TOOL_MODULES:
        try:
            if module_name not in sys.modules:
                importlib.import_module(module_name)
        except Exception as e:
            # Non-critical: individual tool module failures should not
            # prevent the rest of the system from working.
            pass

    return ToolRegistry.instance()


def get_tools_dict() -> dict:
    """Convenience: return {name: async_func} for all enabled tools.

    This dict is directly compatible with the existing ``Agent(tools=...)``
    constructor in ``agent/core.py``.
    """
    return ToolRegistry.instance().get_tools_dict()


def get_schemas(toolset: str = None) -> list:
    """Convenience: return Anthropic-format tool schemas.

    Optionally filter by toolset name.
    """
    return ToolRegistry.instance().get_schemas(toolset=toolset)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

__all__ = [
    "ToolRegistry",
    "ToolDefinition",
    "discover_tools",
    "get_tools_dict",
    "get_schemas",
]
