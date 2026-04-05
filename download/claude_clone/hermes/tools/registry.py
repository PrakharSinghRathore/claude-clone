"""
Hermes Tool Registry — singleton self-registering tool system.

Every tool module calls ``ToolRegistry.instance().register(...)`` at import time.
The registry supports:
- Named toolsets (terminal, web, file, memory, …)
- Enable / disable tools at runtime
- Anthropic-format schema export for API calls
- Function-call dispatch by name
- Thread-safe registration via asyncio.Lock
"""

from __future__ import annotations

import asyncio
import inspect
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """Metadata for a single registered tool."""
    name: str
    func: Callable
    description: str
    input_schema: Dict[str, Any]
    toolset: str = "default"
    enabled: bool = True
    tags: List[str] = field(default_factory=list)

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Return an Anthropic-compatible tool schema dict."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# ---------------------------------------------------------------------------
# Parameter-type mapping (mirrors the existing agent/tools.py approach)
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "path": "string",
    "Path": "string",
}


def _extract_schema(func: Callable, description: str = "") -> Dict[str, Any]:
    """Auto-generate an Anthropic-style input_schema from a function's signature and docstring."""
    if description is None:
        description = ""

    doc = (func.__doc__ or "").strip()
    if not description:
        description = doc.split("\n")[0] if doc else f"Tool: {func.__name__}"

    params: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    # Collect type hints
    hints = getattr(func, "__annotations__", {})
    hints.pop("return", None)

    # Parse docstring ``param name (type): description`` lines
    param_lines = [l.strip() for l in doc.split("\n") if l.strip().startswith("param")]

    for pline in param_lines:
        m = re.match(r"(\w+)\s*[:\(]\s*([^\):]+)[\):]\s*[-—]\s*(.+)", pline)
        if m:
            pname, ptype_str, pdesc = m.group(1), m.group(2).strip(), m.group(3).strip()
            json_type = _TYPE_MAP.get(ptype_str, "string")
            params["properties"][pname] = {"type": json_type, "description": pdesc}
            if pname in hints and "Optional" not in str(hints[pname]):
                if pname not in params["required"]:
                    params["required"].append(pname)

    # Fallback: use type hints only
    if not param_lines and hints:
        for pname, ptype in hints.items():
            type_str = ptype.__name__ if hasattr(ptype, "__name__") else str(ptype)
            json_type = _TYPE_MAP.get(type_str, "string")
            params["properties"][pname] = {"type": json_type, "description": f"Parameter {pname}"}
            if "Optional" not in str(ptype) and "None" not in str(ptype):
                params["required"].append(pname)

    return params


# ---------------------------------------------------------------------------
# ToolRegistry singleton
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Thread-safe, singleton tool registry.

    Usage (in a tool module)::

        from hermes.tools.registry import ToolRegistry

        async def my_tool(arg: str) -> str:
            '''Do something useful.

            param arg (str): — The argument.
            '''
            return "ok"

        ToolRegistry.instance().register(
            name="my_tool",
            func=my_tool,
            description="Does something useful",
            toolset="example",
        )
    """

    _instance: Optional["ToolRegistry"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._toolsets: Dict[str, Set[str]] = {}
        self._async_lock = asyncio.Lock()

    # -- singleton ----------------------------------------------------------

    @classmethod
    def instance(cls) -> "ToolRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for tests)."""
        with cls._lock:
            cls._instance = None

    # -- registration -------------------------------------------------------

    def register(
        self,
        name: str,
        func: Callable,
        description: str = "",
        input_schema: Optional[Dict] = None,
        toolset: str = "default",
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register a tool.  This is safe to call at import-time.

        Parameters
        ----------
        name:
            Unique tool name (used as the Anthropic tool name).
        func:
            The async callable that implements the tool.
        description:
            Human-readable description.  If empty, derived from docstring.
        input_schema:
            Anthropic-format input_schema dict.  If ``None``, auto-generated
            from the function signature and docstring.
        toolset:
            Logical group name (e.g. ``"terminal"``, ``"web"``).
        tags:
            Optional list of free-form tags for categorisation.
        """
        if input_schema is None:
            input_schema = _extract_schema(func, description)

        tool_def = ToolDefinition(
            name=name,
            func=func,
            description=description or input_schema.get("description", f"Tool: {name}"),
            input_schema=input_schema,
            toolset=toolset,
            tags=tags or [],
        )

        self._tools[name] = tool_def

        if toolset not in self._toolsets:
            self._toolsets[toolset] = set()
        self._toolsets[toolset].add(name)

    # -- queries ------------------------------------------------------------

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def get_func(self, name: str) -> Optional[Callable]:
        tool = self._tools.get(name)
        return tool.func if tool and tool.enabled else None

    def list_tools(self, enabled_only: bool = True) -> List[ToolDefinition]:
        tools = list(self._tools.values())
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        return tools

    def list_names(self, enabled_only: bool = True) -> List[str]:
        return [t.name for t in self.list_tools(enabled_only=enabled_only)]

    def list_toolsets(self) -> Dict[str, List[str]]:
        """Return {toolset_name: [tool_name, ...]} for all toolsets."""
        return {
            ts: sorted(names) for ts, names in self._toolsets.items()
        }

    def get_tools_by_toolset(self, toolset: str) -> List[ToolDefinition]:
        names = self._toolsets.get(toolset, set())
        return [self._tools[n] for n in names if n in self._tools and self._tools[n].enabled]

    def get_tools_dict(self, enabled_only: bool = True) -> Dict[str, Callable]:
        """Return {name: async_func} — compatible with the existing Agent constructor."""
        result = {}
        for tdef in self.list_tools(enabled_only=enabled_only):
            result[tdef.name] = tdef.func
        return result

    # -- Anthropic schema export ---------------------------------------------

    def get_schemas(self, toolset: Optional[str] = None, enabled_only: bool = True) -> List[Dict]:
        """
        Export tool schemas in Anthropic format.

        Parameters
        ----------
        toolset:
            If given, only export tools in this toolset.
        enabled_only:
            Skip disabled tools (default True).
        """
        tools = self.get_tools_by_toolset(toolset) if toolset else self.list_tools(enabled_only=enabled_only)
        return [t.to_anthropic_schema() for t in tools]

    # -- enable / disable ---------------------------------------------------

    def enable(self, name: str) -> bool:
        tool = self._tools.get(name)
        if tool:
            tool.enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        tool = self._tools.get(name)
        if tool:
            tool.enabled = False
            return True
        return False

    def enable_toolset(self, toolset: str) -> int:
        count = 0
        for name in self._toolsets.get(toolset, set()):
            tool = self._tools.get(name)
            if tool and not tool.enabled:
                tool.enabled = True
                count += 1
        return count

    def disable_toolset(self, toolset: str) -> int:
        count = 0
        for name in self._toolsets.get(toolset, set()):
            tool = self._tools.get(name)
            if tool and tool.enabled:
                tool.enabled = False
                count += 1
        return count

    # -- dispatch -----------------------------------------------------------

    async def dispatch(self, name: str, **kwargs) -> Any:
        """
        Look up a tool by name and call it with the given kwargs.

        Returns the tool's return value (usually a string or dict).
        Raises ``KeyError`` if the tool is not found or is disabled.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name}")
        if not tool.enabled:
            raise KeyError(f"Tool is disabled: {name}")

        func = tool.func
        if asyncio.iscoroutinefunction(func):
            return await func(**kwargs)
        # Synchronous function — run in executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(**kwargs))

    # -- introspection ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        enabled = sum(1 for t in self._tools.values() if t.enabled)
        return f"<ToolRegistry {len(self._tools)} tools ({enabled} enabled, {len(self._toolsets)} toolsets)>"
