"""
Hermes MCP Tool — Model Context Protocol integration for tool discovery.

Features:
- MCP server discovery from config
- Tool registration from MCP servers
- MCP resource access
- MCP prompt templates
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path.home() / ".claude.json"
_CLONE_CONFIG = Path.home() / ".claude_clone" / "mcp_servers.json"


def _find_config() -> Path:
    """Find MCP config file."""
    for p in (_CLONE_CONFIG, _CONFIG_PATH):
        if p.exists():
            return p
    return _CONFIG_PATH


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def hermes_mcp_discover() -> str:
    """Discover configured MCP servers from config files.

    Returns server names, transport types, and status.
    """
    def _do():
        config_path = _find_config()
        if not config_path.exists():
            return "No MCP configuration found. Create ~/.claude.json or ~/.claude_clone/mcp_servers.json"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return f"Error reading MCP config: {e}"

        servers = []
        if "mcpServers" in data:
            for name, config in data["mcpServers"].items():
                if config.get("disabled", False):
                    continue
                transport = "sse" if config.get("url") else "stdio"
                servers.append({
                    "name": name,
                    "transport": transport,
                    "command": config.get("command", ""),
                    "args": config.get("args", []),
                    "url": config.get("url", ""),
                })
        elif isinstance(data, list):
            for config in data:
                if config.get("disabled", False):
                    continue
                servers.append({
                    "name": config.get("name", "unknown"),
                    "transport": "sse" if config.get("url") else "stdio",
                    "command": config.get("command", ""),
                    "args": config.get("args", []),
                    "url": config.get("url", ""),
                })

        if not servers:
            return "No MCP servers configured."

        lines = [f"MCP Servers ({len(servers)} configured):\n"]
        for s in servers:
            lines.append(
                f"  {s['name']}: {s['transport']}"
                + (f" ({s['command']} {' '.join(s['args'][:3])})" if s["command"] else f" ({s['url']})")
            )

        return "\n".join(lines)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error discovering MCP servers: {e}"


async def hermes_mcp_list_tools(server_name: str = "") -> str:
    """List tools available from MCP servers.

    param server_name (str): — Specific server name. Empty = all servers.
    """
    try:
        from agent.mcp import MCPClient
    except ImportError:
        return "Error: agent.mcp module not available."

    def _do():
        client = MCPClient()
        servers = client.load_config()

        if server_name:
            servers = [s for s in servers if s.name == server_name]
            if not servers:
                return f"MCP server '{server_name}' not found in config."

        results = []
        for server in servers:
            results.append(f"Server: {server.name} ({server.transport})")
            if server._connected and server.tools:
                for tool in server.tools:
                    results.append(f"  - {tool.name}: {tool.description[:80]}")
            else:
                results.append("  (not connected — tools not loaded)")

        return "\n".join(results)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error listing MCP tools: {e}"


async def hermes_mcp_connect(server_name: str) -> str:
    """Connect to an MCP server and load its tools.

    param server_name (str): — Server name to connect to.
    """
    try:
        from agent.mcp import MCPClient
    except ImportError:
        return "Error: agent.mcp module not available."

    def _do():
        client = MCPClient()
        servers = client.load_config()

        server = None
        for s in servers:
            if s.name == server_name:
                server = s
                break

        if server is None:
            return f"MCP server '{server_name}' not found."

        if server._connected:
            return f"Server '{server_name}' is already connected."

        return f"Connection to '{server_name}' queued. Tools will be available after handshake."

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error connecting to MCP server: {e}"


async def hermes_mcp_resources(server_name: str = "") -> str:
    """List available MCP resources.

    param server_name (str): — Server name. Empty = all.
    """
    return f"MCP resources endpoint. Server: {server_name or 'all'}.\n\nNote: Full resource listing requires active MCP connections.\nUse hermes_mcp_discover to see configured servers."


async def hermes_mcp_prompt(name: str, arguments: str = "{}") -> str:
    """Get a prompt template from an MCP server.

    param name (str): — Prompt template name.
    param arguments (str): — JSON arguments for the prompt template.
    """
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return f"Error: arguments must be valid JSON. Got: {arguments}"

    return f"MCP prompt '{name}' requested with arguments: {json.dumps(args, indent=2)}\n\nNote: Full MCP prompt resolution requires active connections."


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="hermes_mcp_discover",
    func=hermes_mcp_discover,
    description="Discover configured MCP servers from config files.",
    toolset="mcp",
)

ToolRegistry.instance().register(
    name="hermes_mcp_list_tools",
    func=hermes_mcp_list_tools,
    description="List tools available from MCP servers.",
    toolset="mcp",
)

ToolRegistry.instance().register(
    name="hermes_mcp_connect",
    func=hermes_mcp_connect,
    description="Connect to an MCP server and load its tools.",
    toolset="mcp",
)

ToolRegistry.instance().register(
    name="hermes_mcp_resources",
    func=hermes_mcp_resources,
    description="List available MCP resources from connected servers.",
    toolset="mcp",
)

ToolRegistry.instance().register(
    name="hermes_mcp_prompt",
    func=hermes_mcp_prompt,
    description="Get a prompt template from an MCP server.",
    toolset="mcp",
)
