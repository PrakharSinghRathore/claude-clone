"""
MCP (Model Context Protocol) server configuration for Hermes CLI.

Add/remove MCP servers, browse MCP tools and resources,
manage server health, and import/export configurations.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes.cli_hermes.config_manager import ConfigManager


# ──────────────────────────────────────────────
# MCP server defaults
# ──────────────────────────────────────────────

KNOWN_MCP_SERVERS = {
    "filesystem": {
        "name": "Filesystem",
        "description": "File system access and management",
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-filesystem", "/tmp"],
        "env": {},
        "category": "core",
    },
    "git": {
        "name": "Git",
        "description": "Git repository operations",
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-git"],
        "env": {},
        "category": "dev",
    },
    "github": {
        "name": "GitHub",
        "description": "GitHub API integration",
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-github"],
        "env": {"GITHUB_TOKEN": ""},
        "category": "dev",
    },
    "postgres": {
        "name": "PostgreSQL",
        "description": "PostgreSQL database access",
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-postgres", "postgresql://localhost/mydb"],
        "env": {},
        "category": "data",
    },
    "web-search": {
        "name": "Web Search",
        "description": "Search the web via Brave Search",
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-brave-search"],
        "env": {"BRAVE_API_KEY": ""},
        "category": "search",
    },
    "puppeteer": {
        "name": "Puppeteer",
        "description": "Browser automation",
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-puppeteer"],
        "env": {},
        "category": "browser",
    },
}


class MCPConfigManager:
    """Manages MCP server configurations."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()

    def list_servers(self) -> List[Dict[str, Any]]:
        """List all configured MCP servers."""
        configured = self.config.get("mcp.servers", [])
        result = []

        for server in configured:
            sid = server.get("id", server.get("name", ""))
            result.append({
                "id": sid,
                "name": server.get("name", sid),
                "description": server.get("description", ""),
                "command": server.get("command", ""),
                "args": server.get("args", []),
                "enabled": server.get("enabled", True),
                "auto_connect": server.get("auto_connect", True),
                "category": server.get("category", "other"),
                "has_required_env": self._check_env(server),
            })

        return result

    def add_server(
        self,
        server_id: str,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        description: str = "",
        category: str = "other",
        auto_connect: bool = True,
    ) -> bool:
        """Add a new MCP server configuration."""
        servers = list(self.config.get("mcp.servers", []))

        # Check for duplicates
        for s in servers:
            if s.get("id") == server_id:
                return False

        servers.append({
            "id": server_id,
            "name": name,
            "description": description,
            "command": command,
            "args": args or [],
            "env": env or {},
            "category": category,
            "enabled": True,
            "auto_connect": auto_connect,
            "added_at": datetime.now().isoformat(),
        })

        self.config.set("mcp.servers", servers)
        self.config.save()
        return True

    def remove_server(self, server_id: str) -> bool:
        """Remove an MCP server."""
        servers = list(self.config.get("mcp.servers", []))
        original_len = len(servers)

        servers = [s for s in servers if s.get("id") != server_id]

        if len(servers) < original_len:
            self.config.set("mcp.servers", servers)
            self.config.save()
            return True
        return False

    def get_server(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific MCP server configuration."""
        for server in self.config.get("mcp.servers", []):
            if server.get("id") == server_id:
                return server
        return None

    def update_server(self, server_id: str, **kwargs) -> bool:
        """Update an MCP server configuration."""
        servers = list(self.config.get("mcp.servers", []))

        for i, server in enumerate(servers):
            if server.get("id") == server_id:
                for key, value in kwargs.items():
                    servers[i][key] = value
                self.config.set("mcp.servers", servers)
                self.config.save()
                return True

        return False

    def enable_server(self, server_id: str) -> bool:
        """Enable an MCP server."""
        return self.update_server(server_id, enabled=True)

    def disable_server(self, server_id: str) -> bool:
        """Disable an MCP server."""
        return self.update_server(server_id, enabled=False)

    def health_check(self, server_id: str) -> Dict[str, Any]:
        """Check MCP server health."""
        server = self.get_server(server_id)
        if not server:
            return {"status": "not_found", "server_id": server_id}

        command = server.get("command", "")
        if not command:
            return {"status": "error", "message": "No command configured"}

        # Check if command exists
        import shutil
        cmd_path = shutil.which(command)
        if cmd_path:
            return {
                "status": "healthy",
                "server_id": server_id,
                "command_path": cmd_path,
            }
        else:
            return {
                "status": "unhealthy",
                "server_id": server_id,
                "message": f"Command not found: {command}",
            }

    def health_check_all(self) -> List[Dict[str, Any]]:
        """Run health checks on all servers."""
        results = []
        for server in self.list_servers():
            result = self.health_check(server["id"])
            results.append(result)
        return results

    def browse_tools(self, server_id: str) -> List[Dict[str, Any]]:
        """Browse tools available from an MCP server (stub)."""
        server = self.get_server(server_id)
        if not server:
            return []

        # In a real implementation, this would connect to the MCP server
        # and retrieve its tool list
        return [
            {
                "name": f"{server_id}_tool_1",
                "description": f"Tool from {server.get('name', server_id)}",
                "server": server_id,
            }
        ]

    def browse_resources(self, server_id: str) -> List[Dict[str, Any]]:
        """Browse resources available from an MCP server (stub)."""
        server = self.get_server(server_id)
        if not server:
            return []
        return []

    def export_config(self, path: str, include_secrets: bool = False) -> bool:
        """Export MCP configuration to a file."""
        servers = list(self.config.get("mcp.servers", []))

        if not include_secrets:
            for server in servers:
                if "env" in server:
                    server["env"] = {k: ("***" if v else "") for k, v in server["env"].items()}

        try:
            export_path = Path(path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump({"mcp_servers": servers}, f, indent=2)
            return True
        except Exception:
            return False

    def import_config(self, path: str, overwrite: bool = False) -> bool:
        """Import MCP configuration from a file."""
        import_path = Path(path)
        if not import_path.exists():
            return False

        try:
            with open(import_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            new_servers = data.get("mcp_servers", [])
            if not new_servers:
                return False

            existing = list(self.config.get("mcp.servers", []))

            for server in new_servers:
                sid = server.get("id", server.get("name", ""))
                if not sid:
                    continue

                # Check for duplicates
                found = False
                for i, s in enumerate(existing):
                    if s.get("id") == sid:
                        if overwrite:
                            existing[i] = server
                        found = True
                        break

                if not found:
                    existing.append(server)

            self.config.set("mcp.servers", existing)
            self.config.save()
            return True
        except Exception:
            return False

    def install_from_known(self, server_id: str) -> bool:
        """Install a known MCP server template."""
        template = KNOWN_MCP_SERVERS.get(server_id)
        if not template:
            return False

        return self.add_server(
            server_id=server_id,
            name=template["name"],
            command=template["command"],
            args=template.get("args", []),
            env=template.get("env", {}),
            description=template.get("description", ""),
            category=template.get("category", "other"),
        )

    def get_known_servers(self) -> List[Dict[str, Any]]:
        """Get list of known MCP server templates."""
        configured = {s.get("id") for s in self.config.get("mcp.servers", [])}

        result = []
        for sid, info in KNOWN_MCP_SERVERS.items():
            result.append({
                "id": sid,
                "name": info["name"],
                "description": info["description"],
                "category": info.get("category", "other"),
                "installed": sid in configured,
            })

        return result

    def format_servers_table(self) -> str:
        """Format MCP servers as a table."""
        servers = self.list_servers()

        if not servers:
            return "  No MCP servers configured."

        lines = []
        lines.append(f"  {'Server':<18} {'Command':<20} {'Category':<10} {'Env':<8} {'Status'}")
        lines.append("  " + "-" * 70)

        for server in servers:
            env_status = "\033[32mOK\033[0m" if server.get("has_required_env") else "\033[33mMissing\033[0m"
            status = "\033[32menabled\033[0m" if server.get("enabled") else "\033[31mdisabled\033[0m"
            lines.append(
                f"  {server['name']:<18} {server['command']:<20} "
                f"{server['category']:<10} {env_status:<8} {status}"
            )

        return "\n".join(lines)

    def _check_env(self, server: Dict) -> bool:
        """Check if required environment variables are set."""
        env = server.get("env", {})
        for key, value in env.items():
            if value and not value.startswith("***"):
                # Value is pre-configured
                continue
            if not os.environ.get(key):
                return False
        return True
