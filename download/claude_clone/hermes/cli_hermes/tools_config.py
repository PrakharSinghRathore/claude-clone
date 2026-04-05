"""
Tool enable/disable UI and configuration for Hermes CLI.

List tools by category, enable/disable individual tools,
show usage statistics, and manage permission levels.
"""

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes.cli_hermes.config_manager import ConfigManager

try:
    # Try to import the tool registry from the parent project
    sys_path = str(Path(__file__).parent.parent.parent)
    if sys_path not in __import__("sys").path:
        __import__("sys").path.insert(0, sys_path)
    from agent.tools import TOOLS_REGISTRY
    HAS_TOOLS = True
except ImportError:
    HAS_TOOLS = False
    TOOLS_REGISTRY = {}


# ──────────────────────────────────────────────
# Tool categories
# ──────────────────────────────────────────────

TOOL_CATEGORIES = {
    "file": {
        "name": "File Operations",
        "emoji": "\U0001f4c4",
        "tools": ["read_file", "write_file", "edit_file", "append_file", "delete_file", "move_file", "copy_file"],
    },
    "directory": {
        "name": "Directory Operations",
        "emoji": "\U0001f4c2",
        "tools": ["list_directory", "create_directory", "get_project_structure"],
    },
    "search": {
        "name": "Search & Find",
        "emoji": "\U0001f50d",
        "tools": ["search_files", "grep", "find_definition"],
    },
    "execution": {
        "name": "Code Execution",
        "emoji": "\u26a1",
        "tools": ["run_command", "run_python", "run_script"],
    },
    "web": {
        "name": "Web & Network",
        "emoji": "\U0001f310",
        "tools": ["web_search", "fetch_url"],
    },
    "code": {
        "name": "Code Quality",
        "emoji": "\U0001f527",
        "tools": ["lint_python", "format_python"],
    },
    "git": {
        "name": "Git",
        "emoji": "\U0001f500",
        "tools": ["get_git_status", "git_diff", "git_log"],
    },
    "system": {
        "name": "System",
        "emoji": "\U0001f4bb",
        "tools": ["get_environment", "install_package"],
    },
    "memory": {
        "name": "Memory",
        "emoji": "\U0001f9e0",
        "tools": ["memory_search", "memory_store", "memory_list"],
    },
    "security": {
        "name": "Security",
        "emoji": "\U0001f6e1\ufe0f",
        "tools": ["security_scan", "scan_secrets"],
    },
}

# Permission levels
PERMISSION_LEVELS = {
    "auto": "Auto-approve (runs without confirmation)",
    "confirm": "Confirm (asks before each use)",
    "deny": "Denied (cannot be used)",
}


class ToolConfigManager:
    """Manages tool enable/disable state and permissions."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()

    def list_all_tools(self) -> List[Dict[str, Any]]:
        """List all available tools with their status."""
        tools = []
        disabled = set(self.config.get("disabled_tools", []))
        auto_approve = set(self.config.get("auto_approve_tools", []))
        allowed = set(self.config.get("allowed_tools", []))

        # Merge registry with known categories
        all_tool_names = set(TOOLS_REGISTRY.keys()) if HAS_TOOLS else set()
        for cat_info in TOOL_CATEGORIES.values():
            all_tool_names.update(cat_info["tools"])

        for tool_name in sorted(all_tool_names):
            is_enabled = tool_name not in disabled
            if allowed and tool_name not in allowed:
                is_enabled = False

            permission = "auto"
            if tool_name in disabled:
                permission = "deny"
            elif tool_name in auto_approve:
                permission = "auto"
            else:
                permission = "confirm"

            # Get category
            category = "other"
            for cat_id, cat_info in TOOL_CATEGORIES.items():
                if tool_name in cat_info["tools"]:
                    category = cat_id
                    break

            # Get description from registry
            description = ""
            if HAS_TOOLS and tool_name in TOOLS_REGISTRY:
                func = TOOLS_REGISTRY[tool_name]
                doc = (func.__doc__ or "").strip()
                description = doc.split("\n")[0] if doc else ""

            tools.append({
                "name": tool_name,
                "category": category,
                "enabled": is_enabled,
                "permission": permission,
                "description": description,
            })

        return tools

    def list_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """List tools grouped by category."""
        all_tools = self.list_all_tools()
        categories: Dict[str, List[Dict[str, Any]]] = {}

        for tool in all_tools:
            cat = tool["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tool)

        return categories

    def enable_tool(self, tool_name: str) -> bool:
        """Enable a specific tool."""
        disabled = list(self.config.get("disabled_tools", []))
        if tool_name in disabled:
            disabled.remove(tool_name)
            self.config.set("disabled_tools", disabled)
            self.config.save()
            return True
        return True  # Already enabled

    def disable_tool(self, tool_name: str) -> bool:
        """Disable a specific tool."""
        disabled = list(self.config.get("disabled_tools", []))
        if tool_name not in disabled:
            disabled.append(tool_name)
            self.config.set("disabled_tools", disabled)
            self.config.save()
            return True
        return True  # Already disabled

    def enable_category(self, category: str) -> int:
        """Enable all tools in a category."""
        cat_info = TOOL_CATEGORIES.get(category)
        if not cat_info:
            return 0

        count = 0
        for tool_name in cat_info["tools"]:
            if self.enable_tool(tool_name):
                count += 1
        return count

    def disable_category(self, category: str) -> int:
        """Disable all tools in a category."""
        cat_info = TOOL_CATEGORIES.get(category)
        if not cat_info:
            return 0

        count = 0
        for tool_name in cat_info["tools"]:
            if self.disable_tool(tool_name):
                count += 1
        return count

    def set_permission(self, tool_name: str, level: str) -> bool:
        """Set permission level for a tool."""
        if level not in PERMISSION_LEVELS:
            return False

        if level == "auto":
            auto_approve = list(self.config.get("auto_approve_tools", []))
            disabled = list(self.config.get("disabled_tools", []))

            if tool_name not in auto_approve:
                auto_approve.append(tool_name)
            if tool_name in disabled:
                disabled.remove(tool_name)

            self.config.set("auto_approve_tools", auto_approve)
            self.config.set("disabled_tools", disabled)

        elif level == "confirm":
            auto_approve = list(self.config.get("auto_approve_tools", []))
            disabled = list(self.config.get("disabled_tools", []))

            if tool_name in auto_approve:
                auto_approve.remove(tool_name)
            if tool_name in disabled:
                disabled.remove(tool_name)

            self.config.set("auto_approve_tools", auto_approve)
            self.config.set("disabled_tools", disabled)

        elif level == "deny":
            disabled = list(self.config.get("disabled_tools", []))
            auto_approve = list(self.config.get("auto_approve_tools", []))

            if tool_name not in disabled:
                disabled.append(tool_name)
            if tool_name in auto_approve:
                auto_approve.remove(tool_name)

            self.config.set("disabled_tools", disabled)
            self.config.set("auto_approve_tools", auto_approve)

        self.config.save()
        return True

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        # Read from local stats file if it exists
        stats_file = Path.home() / ".claude_clone" / "hermes" / "tool_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            "total_calls": 0,
            "by_tool": {},
            "by_category": {},
            "last_used": {},
        }

    def record_usage(self, tool_name: str, duration: float = 0, success: bool = True) -> None:
        """Record tool usage for statistics."""
        stats = self.get_usage_stats()
        stats["total_calls"] = stats.get("total_calls", 0) + 1

        tool_stats = stats.get("by_tool", {}).get(tool_name, {})
        tool_stats["count"] = tool_stats.get("count", 0) + 1
        tool_stats["total_duration"] = tool_stats.get("total_duration", 0) + duration
        tool_stats["successes"] = tool_stats.get("successes", 0) + (1 if success else 0)
        stats["by_tool"][tool_name] = tool_stats

        stats["last_used"][tool_name] = datetime.now().isoformat()

        # Save stats
        stats_file = Path.home() / ".claude_clone" / "hermes" / "tool_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)

    def format_tools_table(self, tools: Optional[List[Dict]] = None) -> str:
        """Format tools as a readable table."""
        tools = tools or self.list_all_tools()
        categories = TOOL_CATEGORIES

        lines = []
        lines.append(f"  {'Tool':<25} {'Category':<12} {'Status':<10} {'Permission'}")
        lines.append("  " + "-" * 70)

        current_category = ""
        for tool in tools:
            cat_info = categories.get(tool["category"], {})
            cat_name = cat_info.get("name", tool["category"])
            status = "\033[32menabled\033[0m" if tool["enabled"] else "\033[31mdisabled\033[0m"
            permission = tool["permission"]
            lines.append(f"  {tool['name']:<25} {cat_name:<12} {status:<18} {permission}")

        total = len(tools)
        enabled = sum(1 for t in tools if t["enabled"])
        lines.append(f"\n  Total: {total} tools, {enabled} enabled, {total - enabled} disabled")
        return "\n".join(lines)

    def format_category_summary(self) -> str:
        """Format tool categories as a summary table."""
        by_category = self.list_by_category()

        lines = []
        lines.append(f"  {'Category':<20} {'Tools':>6} {'Enabled':>8} {'Disabled':>8}")
        lines.append("  " + "-" * 46)

        for cat_id, tools in sorted(by_category.items()):
            cat_info = TOOL_CATEGORIES.get(cat_id, {"name": cat_id, "emoji": ""})
            name = f"{cat_info.get('emoji', '')} {cat_info.get('name', cat_id)}"
            total = len(tools)
            enabled = sum(1 for t in tools if t["enabled"])
            disabled = total - enabled
            lines.append(f"  {name:<20} {total:>6} {enabled:>8} {disabled:>8}")

        return "\n".join(lines)
