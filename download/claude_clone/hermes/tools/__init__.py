"""
Hermes Tools System — self-registering tool registry with 18+ tool implementations.

Every tool module in this package calls ``ToolRegistry.instance().register()``
at import time, so simply importing this package activates all tools.

Usage::

    from hermes.tools import ToolRegistry

    # All tools are already registered
    reg = ToolRegistry.instance()
    print(reg)                    # <ToolRegistry 60+ tools (60+ enabled, 10 toolsets)>

    # Get Anthropic-format schemas
    schemas = reg.get_schemas()

    # Get {name: func} dict (compatible with Agent constructor)
    tools = reg.get_tools_dict()

    # Dispatch a tool call
    result = await reg.dispatch("hermes_web_search", query="python async")

    # Enable/disable at runtime
    reg.disable("hermes_exec_bash")
    reg.enable_toolset("web")
"""

from hermes.tools.registry import ToolRegistry, ToolDefinition

# ──────────────────────────────────────────────
# Auto-discover and import all tool modules.
# Each module self-registers its tools on import.
# ──────────────────────────────────────────────

_TOOL_MODULES = [
    "hermes.tools.terminal_tool",
    "hermes.tools.web_tools",
    "hermes.tools.browser_tool",
    "hermes.tools.file_tools",
    "hermes.tools.memory_tool",
    "hermes.tools.session_search",
    "hermes.tools.skills_tool",
    "hermes.tools.skill_manager",
    "hermes.tools.skills_hub",
    "hermes.tools.code_execution",
    "hermes.tools.delegate_tool",
    "hermes.tools.mcp_tool",
    "hermes.tools.tts_tool",
    "hermes.tools.transcription_tool",
    "hermes.tools.vision_tool",
    "hermes.tools.cronjob_tool",
    "hermes.tools.send_message_tool",
    "hermes.tools.image_gen_tool",
    "hermes.tools.todo_tool",
    "hermes.tools.mixture_of_agents_tool",
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
