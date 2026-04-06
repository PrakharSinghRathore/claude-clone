"""
CLI slash command implementations for Atlas TUI.

All slash commands with handlers, command parsing/validation,
help system with categorized command list, and command history.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from atlas.cli_atlas.skin_engine import SkinEngine, RESET


# ──────────────────────────────────────────────
# Command Registry
# ──────────────────────────────────────────────

class CommandInfo:
    """Metadata for a slash command."""

    def __init__(
        self,
        name: str,
        aliases: List[str],
        description: str,
        usage: str,
        category: str,
        handler: Optional[Callable] = None,
        subcommands: Optional[Dict[str, "CommandInfo"]] = None,
        examples: Optional[List[str]] = None,
        args: Optional[List[Dict]] = None,
    ):
        self.name = name
        self.aliases = aliases
        self.description = description
        self.usage = usage
        self.category = category
        self.handler = handler
        self.subcommands = subcommands or {}
        self.examples = examples or []
        self.args = args or []


class CommandRegistry:
    """Registry of all available slash commands."""

    CATEGORIES = {
        "general": "General",
        "conversation": "Conversation",
        "model": "Model & Provider",
        "tools": "Tools & Skills",
        "session": "Session Management",
        "appearance": "Appearance & Theme",
        "gateway": "Gateway",
        "advanced": "Advanced",
        "config": "Configuration",
    }

    def __init__(self):
        self._commands: Dict[str, CommandInfo] = {}
        self._register_builtin_commands()

    def register(self, cmd: CommandInfo) -> None:
        """Register a command."""
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._commands[alias] = cmd

    def get(self, name: str) -> Optional[CommandInfo]:
        """Get a command by name or alias."""
        return self._commands.get(name.lower())

    def list_all(self) -> List[CommandInfo]:
        """List all registered commands."""
        seen = set()
        result = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                result.append(cmd)
        return sorted(result, key=lambda c: c.category + c.name)

    def list_by_category(self) -> Dict[str, List[CommandInfo]]:
        """List commands grouped by category."""
        categories: Dict[str, List[CommandInfo]] = {}
        seen = set()
        for cmd in self._commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                cat = cmd.category
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(cmd)
        return categories

    def autocomplete(self, prefix: str) -> List[str]:
        """Get autocomplete suggestions for a prefix."""
        prefix = prefix.lower()
        suggestions = set()
        for name, cmd in self._commands.items():
            if name.startswith(prefix):
                if name == cmd.name:
                    suggestions.add(cmd.name)
                else:
                    suggestions.add(name)
        return sorted(suggestions)

    def _register_builtin_commands(self):
        """Register all built-in slash commands."""

        # ─── General ───
        self.register(CommandInfo(
            name="/help", aliases=["/h", "/?"],
            description="Show help and list all commands",
            usage="/help [command]",
            category="general",
            examples=["/help", "/help /model"],
        ))

        self.register(CommandInfo(
            name="/quit", aliases=["/exit", "/q"],
            description="Exit the CLI",
            usage="/quit",
            category="general",
        ))

        self.register(CommandInfo(
            name="/version", aliases=["/v"],
            description="Show version information",
            usage="/version",
            category="general",
        ))

        self.register(CommandInfo(
            name="/doctor",
            description="Run system diagnostics",
            usage="/doctor [check|fix|report]",
            category="general",
            examples=["/doctor", "/doctor fix"],
        ))

        # ─── Conversation ───
        self.register(CommandInfo(
            name="/clear", aliases=["/cl"],
            description="Clear conversation history",
            usage="/clear",
            category="conversation",
        ))

        self.register(CommandInfo(
            name="/context", aliases=["/ctx"],
            description="Add files to conversation context",
            usage="/context <file-path>",
            category="conversation",
            examples=["/context main.py", "/context src/**/*.py"],
        ))

        self.register(CommandInfo(
            name="/compact", aliases=["/summarize"],
            description="Compress conversation history",
            usage="/compact",
            category="conversation",
        ))

        self.register(CommandInfo(
            name="/export",
            description="Export conversation to Markdown",
            usage="/export [path]",
            category="conversation",
            examples=["/export", "/export ~/output.md"],
        ))

        self.register(CommandInfo(
            name="/undo",
            description="Undo the last message",
            usage="/undo",
            category="conversation",
        ))

        # ─── Model & Provider ───
        self.register(CommandInfo(
            name="/model", aliases=["/m"],
            description="Switch or list AI models",
            usage="/model [model-name]",
            category="model",
            examples=["/model", "/model anthropic/claude-sonnet-4-20250514"],
        ))

        self.register(CommandInfo(
            name="/provider",
            description="Show or switch API provider",
            usage="/provider [name]",
            category="model",
            examples=["/provider", "/provider anthropic"],
        ))

        self.register(CommandInfo(
            name="/cost",
            description="Show token usage and cost estimates",
            usage="/cost",
            category="model",
        ))

        # ─── Agent ───
        self.register(CommandInfo(
            name="/agent", aliases=["/a"],
            description="Switch to a specialized agent",
            usage="/agent [id|name]",
            category="model",
            examples=["/agent", "/agent codegen", "/agent debug"],
        ))

        self.register(CommandInfo(
            name="/agents",
            description="List all available agents",
            usage="/agents",
            category="model",
        ))

        self.register(CommandInfo(
            name="/auto",
            description="Auto-select best agent for a task",
            usage="/auto <task description>",
            category="model",
            examples=["/auto fix the bug in login.py"],
        ))

        # ─── Tools ───
        self.register(CommandInfo(
            name="/tools",
            description="List available tools",
            usage="/tools [list|enable|disable|info <name>]",
            category="tools",
            examples=["/tools", "/tools enable bash", "/tools info read_file"],
        ))

        self.register(CommandInfo(
            name="/skills",
            description="Manage skills",
            usage="/skills [list|info|enable|disable <name>]",
            category="tools",
            examples=["/skills", "/skills list", "/skills info web-dev"],
        ))

        self.register(CommandInfo(
            name="/hub",
            description="Browse skills marketplace",
            usage="/hub [search|install|uninstall|browse]",
            category="tools",
            examples=["/hub search python", "/hub install code-review"],
        ))

        # ─── Session Management ───
        self.register(CommandInfo(
            name="/save",
            description="Save current session",
            usage="/save [name]",
            category="session",
            examples=["/save", "/save my-project"],
        ))

        self.register(CommandInfo(
            name="/load",
            description="Load a saved session",
            usage="/load <name>",
            category="session",
            examples=["/load my-project", "/load latest"],
        ))

        self.register(CommandInfo(
            name="/sessions", aliases=["/session"],
            description="List saved sessions",
            usage="/sessions",
            category="session",
        ))

        self.register(CommandInfo(
            name="/switch", aliases=["/sw"],
            description="Switch to a different session",
            usage="/switch <name>",
            category="session",
            examples=["/switch my-project"],
        ))

        self.register(CommandInfo(
            name="/delete",
            description="Delete a saved session",
            usage="/delete <name>",
            category="session",
            examples=["/delete old-session"],
        ))

        # ─── Appearance & Theme ───
        self.register(CommandInfo(
            name="/theme",
            description="Change visual theme",
            usage="/theme [name|list]",
            category="appearance",
            examples=["/theme list", "/theme nord", "/theme dracula"],
        ))

        self.register(CommandInfo(
            name="/prompt",
            description="Change prompt style",
            usage="/prompt [style]",
            category="appearance",
            examples=["/prompt atlas", "/prompt minimal", "/prompt starship"],
        ))

        self.register(CommandInfo(
            name="/vim",
            description="Toggle vim keybindings",
            usage="/vim",
            category="appearance",
        ))

        # ─── Gateway ───
        self.register(CommandInfo(
            name="/gateway", aliases=["/gw"],
            description="Gateway management",
            usage="/gateway [start|stop|status|sessions]",
            category="gateway",
            examples=["/gateway start", "/gateway status"],
        ))

        # ─── Advanced ───
        self.register(CommandInfo(
            name="/mcp",
            description="MCP server management",
            usage="/mcp [list|add|remove|status]",
            category="advanced",
            examples=["/mcp list", "/mcp add filesystem"],
        ))

        self.register(CommandInfo(
            name="/cron",
            description="Cron job management",
            usage="/cron [list|add|remove|pause|resume]",
            category="advanced",
            examples=["/cron list", "/cron add 'daily backup' '0 9 * * *'"],
        ))

        self.register(CommandInfo(
            name="/sandbox", aliases=["/sb"],
            description="Execute code in sandbox",
            usage="/sandbox [python|js|bash] <code>",
            category="advanced",
            examples=["/sandbox python print('hello')"],
        ))

        self.register(CommandInfo(
            name="/scan",
            description="Security scan",
            usage="/scan [secrets|deps|full]",
            category="advanced",
            examples=["/scan", "/scan secrets"],
        ))

        self.register(CommandInfo(
            name="/analyze",
            description="Project analysis",
            usage="/analyze [complexity|deps|dead-code|full]",
            category="advanced",
            examples=["/analyze", "/analyze complexity main.py"],
        ))

        # ─── Configuration ───
        self.register(CommandInfo(
            name="/config", aliases=["/cfg"],
            description="Configuration management",
            usage="/config [get|set|list|edit|reset]",
            category="config",
            examples=["/config list", "/config set theme nord"],
        ))

        self.register(CommandInfo(
            name="/profile",
            description="Profile management",
            usage="/profile [list|switch|create|delete]",
            category="config",
            examples=["/profile list", "/profile switch dev"],
        ))

        self.register(CommandInfo(
            name="/setup",
            description="Run first-time setup wizard",
            usage="/setup",
            category="config",
        ))

        self.register(CommandInfo(
            name="/env",
            description="Show environment information",
            usage="/env",
            category="config",
        ))

        # ─── Misc ───
        self.register(CommandInfo(
            name="/memory", aliases=["/mem"],
            description="Memory management",
            usage="/memory [search|save|sessions|export]",
            category="advanced",
            examples=["/memory search python", "/memory sessions"],
        ))

        self.register(CommandInfo(
            name="/git",
            description="Git operations",
            usage="/git [status|log|diff|smart-commit|stats|blame]",
            category="advanced",
            examples=["/git status", "/git smart-commit"],
        ))

        self.register(CommandInfo(
            name="/diff",
            description="Show file diff",
            usage="/diff [file1] [file2]",
            category="advanced",
            examples=["/diff a.py b.py"],
        ))

        self.register(CommandInfo(
            name="/db",
            description="Database operations",
            usage="/db [query|tables|schema]",
            category="advanced",
            examples=["/db tables"],
        ))

        self.register(CommandInfo(
            name="/deploy",
            description="Deployment commands",
            usage="/deploy [platform|detect|status]",
            category="advanced",
            examples=["/deploy detect"],
        ))

        self.register(CommandInfo(
            name="/collab",
            description="Collaboration commands",
            usage="/collab [start|join|status]",
            category="advanced",
            examples=["/collab start"],
        ))

        self.register(CommandInfo(
            name="/plugins",
            description="Plugin management",
            usage="/plugins [list|install|reload]",
            category="advanced",
            examples=["/plugins list"],
        ))


def generate_help_text(skin: Optional[SkinEngine] = None) -> str:
    """Generate the full help text."""
    registry = CommandRegistry()
    categories = registry.list_by_category()
    cat_labels = CommandRegistry.CATEGORIES

    theme = skin.get_theme() if skin else SkinEngine().get_theme()
    colors = theme.get("colors", {})

    lines = []
    for cat_key in ["general", "conversation", "model", "tools", "session", "appearance", "gateway", "advanced", "config"]:
        cat_label = cat_labels.get(cat_key, cat_key)
        cmds = categories.get(cat_key, [])
        if not cmds:
            continue

        lines.append(f"  \033[1m{cat_label}:\033[0m")
        for cmd in cmds:
            alias_str = ""
            if cmd.aliases:
                alias_str = f", {', '.join(cmd.aliases)}"
            lines.append(f"    \033[36m{cmd.name}{alias_str}\033[0m  {cmd.description}")
        lines.append("")

    # Keyboard shortcuts
    lines.append("  \033[1mKeyboard Shortcuts:\033[0m")
    shortcuts = [
        ("Shift+Enter", "New line in input"),
        ("Up/Down", "Navigate command history"),
        ("Tab", "Autocomplete commands/files"),
        ("Ctrl+C", "Cancel current generation"),
        ("Ctrl+D", "Exit CLI"),
        ("Ctrl+L", "Clear screen"),
        ("Ctrl+S", "Save current session"),
        ("Ctrl+R", "Search history"),
        ("@<path>", "File path autocomplete"),
    ]
    for key, desc in shortcuts:
        lines.append(f"    \033[36m{key:<18}\033[0m  {desc}")

    return "\n".join(lines)
