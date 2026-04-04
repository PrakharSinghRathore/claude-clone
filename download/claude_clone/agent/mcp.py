"""
MCP (Model Context Protocol) Client implementation.

Supports:
- stdio transport (launch MCP servers as subprocesses)
- SSE transport (connect to MCP servers over HTTP)
- Config loading from ~/.claude.json
- Tool discovery and invocation
- Merging MCP tools into the agent's tool registry
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

class MCPTool:
    """Represents a tool provided by an MCP server."""

    def __init__(self, name: str, description: str, input_schema: dict, server_name: str = ""):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.server_name = server_name

    def to_anthropic_schema(self) -> dict:
        """Convert to Anthropic tool schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self):
        return f"<MCPTool {self.name}: {self.description[:50]}>"


class MCPServer:
    """Represents a connected MCP server."""

    def __init__(self, name: str, transport: str = "stdio", command: str = "", args: List[str] = None,
                 url: str = "", env: dict = None):
        self.name = name
        self.transport = transport  # "stdio" or "sse"
        self.command = command
        self.args = args or []
        self.url = url
        self.env = env or {}
        self.tools: List[MCPTool] = []
        self._process = None
        self._connected = False
        self._request_id = 0

    def __repr__(self):
        status = "connected" if self._connected else "disconnected"
        return f"<MCPServer {self.name} ({self.transport}, {status}, {len(self.tools)} tools)>"


class MCPServerConfig:
    """Configuration for an MCP server loaded from JSON."""

    def __init__(self, data: dict):
        self.name = data.get("name", "unknown")
        self.command = data.get("command", "")
        self.args = data.get("args", [])
        self.env = data.get("env", {})
        self.url = data.get("url", "")
        self.disabled = data.get("disabled", False)


# ──────────────────────────────────────────────
# JSON-RPC helpers
# ──────────────────────────────────────────────

def _jsonrpc_request(method: str, params: dict = None, request_id: int = 0) -> dict:
    """Create a JSON-RPC request."""
    req = {
        "jsonrpc": "2.0",
        "method": method,
        "id": request_id,
    }
    if params is not None:
        req["params"] = params
    return req


def _jsonrpc_response(data: dict) -> Optional[dict]:
    """Parse a JSON-RPC response and return result or raise error."""
    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"JSON-RPC error {err.get('code', -1)}: {err.get('message', 'unknown error')}")
    return data.get("result")


# ──────────────────────────────────────────────
# MCP Client
# ──────────────────────────────────────────────

class MCPClient:
    """
    MCP client that manages connections to MCP servers.

    Usage:
        client = MCPClient()
        servers = client.load_config()
        for server in servers:
            if server.transport == "stdio":
                await client.connect_stdio(server)
            else:
                await client.connect_sse(server)
            tools = await client.list_tools(server)
    """

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self._config_path = Path.home() / ".claude.json"

    def load_config(self, path: str = None) -> List[MCPServer]:
        """
        Load MCP server configuration from a JSON file.
        Supports the same format as Claude Code's ~/.claude.json.
        """
        config_path = Path(path) if path else self._config_path

        if not config_path.exists():
            # Try claude_clone config
            clone_path = Path.home() / ".claude_clone" / "mcp_servers.json"
            if clone_path.exists():
                config_path = clone_path
            else:
                return []

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return []

        servers = []

        # Handle different config formats
        # Format 1: { "mcpServers": { "name": { "command": ..., "args": ... } } }
        if "mcpServers" in data:
            for name, config in data["mcpServers"].items():
                server_config = MCPServerConfig({"name": name, **config})
                if server_config.disabled:
                    continue

                if server_config.url:
                    server = MCPServer(
                        name=name, transport="sse", url=server_config.url, env=server_config.env
                    )
                elif server_config.command:
                    server = MCPServer(
                        name=name, transport="stdio", command=server_config.command,
                        args=server_config.args, env=server_config.env,
                    )
                else:
                    continue

                servers.append(server)
                self.servers[name] = server

        # Format 2: [{ "name": ..., "command": ..., "args": ... }]
        elif isinstance(data, list):
            for config in data:
                server_config = MCPServerConfig(config)
                if server_config.disabled:
                    continue

                if server_config.url:
                    server = MCPServer(
                        name=server_config.name, transport="sse",
                        url=server_config.url, env=server_config.env,
                    )
                elif server_config.command:
                    server = MCPServer(
                        name=server_config.name, transport="stdio",
                        command=server_config.command, args=server_config.args,
                        env=server_config.env,
                    )
                else:
                    continue

                servers.append(server)
                self.servers[server.name] = server

        return servers

    async def connect_stdio(self, server: MCPServer) -> MCPServer:
        """
        Connect to an MCP server via stdio transport.
        Launches the server as a subprocess and communicates via stdin/stdout.
        """
        env = os.environ.copy()
        env.update(server.env)

        try:
            process = await asyncio.create_subprocess_exec(
                server.command, *server.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            server._process = process

            # Perform handshake: send initialize
            server._request_id += 1
            init_request = _jsonrpc_request(
                "initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "claude-clone", "version": "1.0.0"},
                },
                request_id=server._request_id,
            )

            request_bytes = json.dumps(init_request).encode() + b"\n"
            process.stdin.write(request_bytes)
            await process.stdin.drain()

            # Read response
            response_line = await asyncio.wait_for(process.stdout.readline(), timeout=10)
            if response_line:
                response = json.loads(response_line.decode().strip())
                _jsonrpc_response(response)  # Raises on error

            # Send initialized notification
            initialized_notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            process.stdin.write(json.dumps(initialized_notification).encode() + b"\n")
            await process.stdin.drain()

            server._connected = True
            return server

        except asyncio.TimeoutError:
            raise RuntimeError(f"Timeout connecting to MCP server '{server.name}' via stdio")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to MCP server '{server.name}': {e}")

    async def connect_sse(self, server: MCPServer) -> MCPServer:
        """
        Connect to an MCP server via SSE transport.
        Connects to an HTTP endpoint that streams events.
        """
        if not HAS_HTTPX:
            raise ImportError("httpx is required for SSE transport. Install it with: pip install httpx")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Try to connect and get session info
                response = await client.get(server.url)
                response.raise_for_status()

                # For SSE, we'd normally parse the event stream
                # For simplicity, try to parse as JSON first
                try:
                    data = response.json()
                    if "tools" in data:
                        for tool_data in data["tools"]:
                            tool = MCPTool(
                                name=tool_data.get("name", ""),
                                description=tool_data.get("description", ""),
                                input_schema=tool_data.get("inputSchema", {}),
                                server_name=server.name,
                            )
                            server.tools.append(tool)
                except json.JSONDecodeError:
                    pass

            server._connected = True
            return server

        except Exception as e:
            raise RuntimeError(f"Failed to connect to MCP server '{server.name}' via SSE: {e}")

    async def list_tools(self, server: MCPServer) -> List[MCPTool]:
        """List all tools provided by an MCP server."""
        if not server._connected:
            raise RuntimeError(f"Server '{server.name}' is not connected")

        # If tools were already loaded during SSE connect
        if server.tools:
            return server.tools

        # For stdio, send tools/list request
        if server.transport == "stdio" and server._process:
            server._request_id += 1
            request = _jsonrpc_request("tools/list", request_id=server._request_id)

            try:
                request_bytes = json.dumps(request).encode() + b"\n"
                server._process.stdin.write(request_bytes)
                await server._process.stdin.drain()

                # Read response
                response_line = await asyncio.wait_for(server._process.stdout.readline(), timeout=10)
                if response_line:
                    response = json.loads(response_line.decode().strip())
                    result = _jsonrpc_response(response)

                    if result and "tools" in result:
                        for tool_data in result["tools"]:
                            tool = MCPTool(
                                name=tool_data.get("name", ""),
                                description=tool_data.get("description", ""),
                                input_schema=tool_data.get("inputSchema", {}),
                                server_name=server.name,
                            )
                            server.tools.append(tool)

            except asyncio.TimeoutError:
                raise RuntimeError(f"Timeout listing tools from '{server.name}'")
            except Exception as e:
                raise RuntimeError(f"Failed to list tools from '{server.name}': {e}")

        return server.tools

    async def call_tool(self, server: MCPServer, tool_name: str, args: dict) -> str:
        """Call a tool on an MCP server."""
        if not server._connected:
            raise RuntimeError(f"Server '{server.name}' is not connected")

        if server.transport == "stdio" and server._process:
            server._request_id += 1
            request = _jsonrpc_request(
                "tools/call",
                params={"name": tool_name, "arguments": args},
                request_id=server._request_id,
            )

            try:
                request_bytes = json.dumps(request).encode() + b"\n"
                server._process.stdin.write(request_bytes)
                await server._process.stdin.drain()

                response_line = await asyncio.wait_for(server._process.stdout.readline(), timeout=60)
                if response_line:
                    response = json.loads(response_line.decode().strip())
                    result = _jsonrpc_response(response)

                    if result:
                        # Extract text content from MCP response
                        if "content" in result:
                            texts = []
                            for block in result["content"]:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    texts.append(block["text"])
                            return "\n".join(texts)
                        return json.dumps(result, indent=2, default=str)

                    return "Tool returned no result"

            except asyncio.TimeoutError:
                return f"Error: Timeout calling tool '{tool_name}' on '{server.name}'"
            except Exception as e:
                return f"Error calling tool '{tool_name}' on '{server.name}': {e}"

        return f"Error: Cannot call tool on '{server.name}' — unsupported transport"

    def get_merged_tools(self, existing_tools: Dict[str, Callable]) -> Dict[str, Callable]:
        """
        Merge MCP tools into the existing tool registry.
        MCP tools are wrapped as async callables.
        """
        merged = dict(existing_tools)

        for server_name, server in self.servers.items():
            if not server._connected:
                continue

            for tool in server.tools:
                # Avoid name collisions by prefixing with server name
                tool_key = f"mcp_{server_name}_{tool.name}" if tool.name in merged else tool.name

                async def _make_wrapper(t_name=tool.name, s=server, key=tool_key, sn=server_name, desc=tool.description):
                    async def wrapper(**kwargs):
                        return await self.call_tool(s, t_name, kwargs)
                    wrapper.__name__ = key
                    wrapper.__doc__ = f"MCP tool from {sn}: {desc}"
                    return wrapper

                # Create the wrapper synchronously (it's just a function factory)
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await — schedule as task and return a placeholder
                    merged[tool_key] = _make_mcp_placeholder(tool_key, server_name, tool.description)
                else:
                    merged[tool_key] = loop.run_until_complete(_make_wrapper())

        return merged

    def get_merged_schemas(self, existing_schemas: List[dict]) -> List[dict]:
        """Merge MCP tool schemas into existing tool schemas."""
        merged = list(existing_schemas)

        for server_name, server in self.servers.items():
            if not server._connected:
                continue

            for tool in server.tools:
                schema = tool.to_anthropic_schema()
                # Prefix if collision
                existing_names = {s["name"] for s in merged}
                if schema["name"] in existing_names:
                    schema["name"] = f"mcp_{server_name}_{tool.name}"
                    schema["description"] = f"[MCP/{server_name}] {tool.description}"
                merged.append(schema)

        return merged

    async def disconnect_all(self):
        """Disconnect all MCP servers."""
        for name, server in self.servers.items():
            await self.disconnect(server)

    async def disconnect(self, server: MCPServer):
        """Disconnect a single MCP server."""
        if server._process:
            try:
                server._process.terminate()
                await asyncio.wait_for(server._process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    server._process.kill()
                except ProcessLookupError:
                    pass
            server._process = None

        server._connected = False


def _make_mcp_placeholder(tool_key: str, server_name: str, description: str) -> Callable:
    """Create a placeholder async callable for MCP tools when called from a running loop."""
    async def placeholder(**kwargs):
        return f"MCP tool '{tool_key}' from '{server_name}' is not available in this context."
    placeholder.__name__ = tool_key
    placeholder.__doc__ = f"MCP tool from {server_name}: {description}"
    return placeholder
