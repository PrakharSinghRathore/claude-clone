"""
Gateway management commands for Hermes CLI.

Start/stop/restart the gateway, manage platform sessions,
view message history, and control platform enable/disable.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes.cli_hermes.config_manager import ConfigManager, SESSIONS_DIR


# ──────────────────────────────────────────────
# Gateway state management
# ──────────────────────────────────────────────

GATEWAY_STATE_FILE = Path.home() / ".claude_clone" / "hermes" / "gateway_state.json"

KNOWN_PLATFORMS = {
    "cli": {"name": "CLI", "icon": "\U0001f4bb", "description": "Terminal interface"},
    "web": {"name": "Web UI", "icon": "\U0001f310", "description": "Browser-based interface"},
    "desktop": {"name": "Desktop", "icon": "\U0001f5a5\ufe0f", "description": "Desktop application"},
    "api": {"name": "API", "icon": "\u2699\ufe0f", "description": "REST API endpoint"},
    "discord": {"name": "Discord", "icon": "\U0001f535", "description": "Discord bot"},
    "slack": {"name": "Slack", "icon": "\U0001f4e8", "description": "Slack integration"},
    "telegram": {"name": "Telegram", "icon": "\u2708\ufe0f", "description": "Telegram bot"},
}


class GatewayManager:
    """Manages the Hermes multi-platform gateway."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.state_file = GATEWAY_STATE_FILE
        self._state: Optional[Dict] = None

    def get_state(self) -> Dict[str, Any]:
        """Get current gateway state."""
        if self._state is None:
            self._load_state()
        return self._state or {"running": False, "platforms": {}, "sessions": []}

    def _load_state(self):
        """Load gateway state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self._state = json.load(f)
            except Exception:
                self._state = {"running": False, "platforms": {}, "sessions": []}
        else:
            self._state = {"running": False, "platforms": {}, "sessions": []}

    def _save_state(self):
        """Save gateway state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, default=str)

    def start(self, host: Optional[str] = None, port: Optional[int] = None) -> Dict[str, Any]:
        """Start the gateway."""
        state = self.get_state()
        if state.get("running"):
            return {
                "success": False,
                "error": "Gateway is already running",
                "pid": state.get("pid"),
            }

        gateway_host = host or self.config.get("gateway.host", "localhost")
        gateway_port = port or self.config.get("gateway.port", 8765)

        # In a real implementation, this would start the gateway server
        # For now, we just update the state
        self._state = {
            "running": True,
            "pid": os.getpid(),
            "host": gateway_host,
            "port": gateway_port,
            "started_at": datetime.now().isoformat(),
            "platforms": {},
            "sessions": [],
            "messages_total": 0,
        }

        # Enable configured platforms
        platforms = self.config.get("gateway.platforms", ["cli"])
        for platform_id in platforms:
            if platform_id in KNOWN_PLATFORMS:
                self._state["platforms"][platform_id] = {
                    "enabled": True,
                    "status": "active",
                    "connected_at": datetime.now().isoformat(),
                    "messages": 0,
                }

        self._save_state()
        return {
            "success": True,
            "host": gateway_host,
            "port": gateway_port,
            "platforms": list(self._state["platforms"].keys()),
        }

    def stop(self) -> Dict[str, Any]:
        """Stop the gateway."""
        state = self.get_state()
        if not state.get("running"):
            return {"success": False, "error": "Gateway is not running"}

        self._state["running"] = False
        self._state["stopped_at"] = datetime.now().isoformat()

        for pid in self._state.get("platforms", {}):
            self._state["platforms"][pid]["status"] = "disconnected"

        self._save_state()
        return {"success": True}

    def restart(self, host: Optional[str] = None, port: Optional[int] = None) -> Dict[str, Any]:
        """Restart the gateway."""
        self.stop()
        time.sleep(0.5)
        return self.start(host, port)

    def status(self) -> Dict[str, Any]:
        """Get gateway status."""
        state = self.get_state()
        config_enabled = self.config.get("gateway.enabled", False)

        return {
            "configured": config_enabled,
            "running": state.get("running", False),
            "pid": state.get("pid"),
            "host": state.get("host"),
            "port": state.get("port"),
            "started_at": state.get("started_at"),
            "uptime": self._calc_uptime(state.get("started_at")),
            "platforms": state.get("platforms", {}),
            "total_sessions": len(state.get("sessions", [])),
            "total_messages": state.get("messages_total", 0),
        }

    def enable_platform(self, platform_id: str) -> bool:
        """Enable a platform."""
        if platform_id not in KNOWN_PLATFORMS:
            return False

        platforms = list(self.config.get("gateway.platforms", []))
        if platform_id not in platforms:
            platforms.append(platform_id)
            self.config.set("gateway.platforms", platforms)
            self.config.save()

        state = self.get_state()
        if platform_id not in state.get("platforms", {}):
            state.setdefault("platforms", {})[platform_id] = {
                "enabled": True,
                "status": "active",
                "connected_at": datetime.now().isoformat(),
                "messages": 0,
            }
            self._save_state()

        return True

    def disable_platform(self, platform_id: str) -> bool:
        """Disable a platform."""
        if platform_id not in KNOWN_PLATFORMS:
            return False

        platforms = list(self.config.get("gateway.platforms", []))
        if platform_id in platforms:
            platforms.remove(platform_id)
            self.config.set("gateway.platforms", platforms)
            self.config.save()

        state = self.get_state()
        if platform_id in state.get("platforms", {}):
            state["platforms"][platform_id]["status"] = "disabled"
            self._save_state()

        return True

    def list_sessions(self, platform: Optional[str] = None) -> List[Dict[str, Any]]:
        """List gateway sessions."""
        state = self.get_state()
        sessions = state.get("sessions", [])

        if platform:
            sessions = [s for s in sessions if s.get("platform") == platform]

        return sessions

    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages for a session."""
        # Check saved session files
        session_dir = SESSIONS_DIR
        session_file = session_dir / f"{session_id}.json"

        if session_file.exists():
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("messages", [])[-limit:]
            except Exception:
                pass

        return []

    def get_logs(self, limit: int = 100) -> List[str]:
        """Get gateway log entries."""
        log_file = Path.home() / ".claude_clone" / "hermes" / "gateway.log"
        if not log_file.exists():
            return []

        try:
            lines = log_file.read_text(encoding="utf-8").strip().split("\n")
            return lines[-limit:]
        except Exception:
            return []

    def format_status_dashboard(self) -> str:
        """Format gateway status as a dashboard display."""
        status = self.status()
        state = status

        lines = []
        lines.append("\n  \033[1mGateway Status\033[0m\n")

        # Main status
        if state["running"]:
            status_text = "\033[32mRunning\033[0m"
        else:
            status_text = "\033[31mStopped\033[0m"

        lines.append(f"  Status:  {status_text}")
        lines.append(f"  Host:    {state.get('host', 'N/A')}")
        lines.append(f"  Port:    {state.get('port', 'N/A')}")
        lines.append(f"  PID:     {state.get('pid', 'N/A')}")
        lines.append(f"  Uptime:  {state.get('uptime', 'N/A')}")
        lines.append(f"  Sessions: {state.get('total_sessions', 0)}")
        lines.append(f"  Messages: {state.get('total_messages', 0)}")

        # Platform status
        platforms = state.get("platforms", {})
        if platforms:
            lines.append(f"\n  \033[1mPlatforms:\033[0m")
            for pid, pinfo in platforms.items():
                pinfo_data = KNOWN_PLATFORMS.get(pid, {"name": pid, "icon": ""})
                p_status = pinfo.get("status", "unknown")
                if p_status == "active":
                    p_status_display = "\033[32mactive\033[0m"
                elif p_status == "disabled":
                    p_status_display = "\033[33mdisabled\033[0m"
                else:
                    p_status_display = f"\033[31m{p_status}\033[0m"
                lines.append(f"    {pinfo_data['icon']} {pinfo_data['name']:<12} {p_status_display}  ({pinfo.get('messages', 0)} msgs)")

        lines.append("")
        return "\n".join(lines)

    def format_platform_table(self) -> str:
        """Format platforms as a table."""
        lines = []
        lines.append(f"  {'Platform':<14} {'Icon':<4} {'Status':<12} {'Messages':>8}")
        lines.append("  " + "-" * 42)

        state = self.get_state()
        active_platforms = state.get("platforms", {})

        for pid, pinfo in KNOWN_PLATFORMS.items():
            platform_state = active_platforms.get(pid, {})
            status = platform_state.get("status", "available")
            messages = platform_state.get("messages", 0)

            if status == "active":
                status_display = "\033[32mactive\033[0m"
            elif status == "disabled":
                status_display = "\033[33mdisabled\033[0m"
            else:
                status_display = "\033[90mavailable\033[0m"

            lines.append(
                f"  {pinfo['name']:<14} {pinfo['icon']:<4} "
                f"{status_display:<12} {messages:>8}"
            )

        return "\n".join(lines)

    @staticmethod
    def _calc_uptime(started_at: Optional[str]) -> str:
        """Calculate uptime from start time."""
        if not started_at:
            return "N/A"

        try:
            start = datetime.fromisoformat(started_at)
            delta = datetime.now() - start

            days = delta.days
            hours, remainder = divmod(delta.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            parts = []
            if days > 0:
                parts.append(f"{days}d")
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")
            parts.append(f"{seconds}s")

            return " ".join(parts)
        except Exception:
            return "N/A"
