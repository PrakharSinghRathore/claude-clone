"""
Main CLI entry point for Atlas CLI.

Uses argparse for subcommand routing with subcommands:
chat, gateway, model, tools, skills, cron, doctor, config, mcp, profile, setup.

Also serves as the primary entry point: `python -m atlas.cli_atlas`
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path if needed
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from atlas.cli_atlas.__init__ import __version__
from atlas.cli_atlas.config_manager import ConfigManager
from atlas.cli_atlas.banner import Banner
from atlas.cli_atlas.skin_engine import SkinEngine


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Atlas CLI — Interactive terminal interface for Claude Clone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  atlas chat                  Start interactive chat
  atlas chat --theme nord     Start chat with Nord theme
  atlas model list            List available models
  atlas model test claude-sonnet-4  Test model connectivity
  atlas doctor                Run diagnostics
  atlas setup                 Run first-time setup
  atlas config list           List configuration
  atlas gateway start         Start the gateway

Run 'atlas <command> --help' for more information on a command.
        """,
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--profile", "-p",
        type=str,
        default=None,
        help="Configuration profile to use",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
    )

    # ─── chat ───
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start interactive chat session (default command)",
        description="Start the interactive TUI chat session. This is the primary interface.",
    )
    chat_parser.add_argument(
        "--theme", "-t",
        type=str,
        default=None,
        help="Visual theme to use (dark, light, nord, dracula, etc.)",
    )
    chat_parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="AI model to use",
    )
    chat_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="API provider to use",
    )
    chat_parser.add_argument(
        "--agent", "-a",
        type=str,
        default=None,
        help="Specialized agent to activate",
    )
    chat_parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Initial prompt to send",
    )
    chat_parser.add_argument(
        "--no-banner",
        action="store_true",
        default=False,
        help="Skip startup banner",
    )
    chat_parser.add_argument(
        "--compact",
        action="store_true",
        default=False,
        help="Use compact banner",
    )

    # ─── gateway ───
    gw_parser = subparsers.add_parser(
        "gateway",
        help="Gateway management",
        description="Manage the multi-platform gateway server.",
    )
    gw_sub = gw_parser.add_subparsers(dest="gateway_command")
    gw_sub.add_parser("start", help="Start the gateway")
    gw_sub.add_parser("stop", help="Stop the gateway")
    gw_sub.add_parser("restart", help="Restart the gateway")
    gw_sub.add_parser("status", help="Show gateway status")
    gw_sub.add_parser("sessions", help="List gateway sessions")
    gw_sub.add_parser("platforms", help="List platforms")

    gw_parser.add_argument("--host", type=str, default=None, help="Gateway host")
    gw_parser.add_argument("--port", type=int, default=None, help="Gateway port")

    # ─── model ───
    model_parser = subparsers.add_parser(
        "model",
        help="Model management",
        description="List, switch, test, and compare AI models.",
    )
    model_sub = model_parser.add_subparsers(dest="model_command")
    model_sub.add_parser("list", help="List available models")
    model_sub.add_parser("current", help="Show current model")
    model_sub.add_parser("pinned", help="Show pinned models")
    model_sub.add_parser("catalog", help="Show full model catalog")

    model_parser.add_argument("model_args", nargs="*", help="Model arguments")

    # ─── tools ───
    tools_parser = subparsers.add_parser(
        "tools",
        help="Tool management",
        description="List, enable, and disable tools.",
    )
    tools_sub = tools_parser.add_subparsers(dest="tools_command")
    tools_sub.add_parser("list", help="List all tools")
    tools_sub.add_parser("categories", help="List tools by category")
    tools_sub.add_parser("stats", help="Show tool usage statistics")

    tools_parser.add_argument("tools_args", nargs="*", help="Tool arguments")

    # ─── skills ───
    skills_parser = subparsers.add_parser(
        "skills",
        help="Skills management",
        description="Manage installed skills.",
    )
    skills_sub = skills_parser.add_subparsers(dest="skills_command")
    skills_sub.add_parser("list", help="List installed skills")
    skills_sub.add_parser("files", help="Show skill files")

    skills_parser.add_argument("skills_args", nargs="*", help="Skill arguments")

    # ─── cron ───
    cron_parser = subparsers.add_parser(
        "cron",
        help="Cron job management",
        description="Manage scheduled jobs.",
    )
    cron_sub = cron_parser.add_subparsers(dest="cron_command")
    cron_sub.add_parser("list", help="List cron jobs")
    cron_sub.add_parser("history", help="Show execution history")

    cron_parser.add_argument("cron_args", nargs="*", help="Cron arguments")

    # ─── doctor ───
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run system diagnostics",
        description="Check system health, dependencies, and configuration.",
    )
    doctor_parser.add_argument(
        "--fix",
        action="store_true",
        default=False,
        help="Attempt to fix common issues",
    )
    doctor_parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Generate full diagnostic report",
    )
    doctor_parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Run quick checks only",
    )

    # ─── config ───
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management",
        description="View and modify configuration settings.",
    )
    config_sub = config_parser.add_subparsers(dest="config_command")
    config_sub.add_parser("list", help="List all config sections")
    config_sub.add_parser("show", help="Show full configuration")
    config_sub.add_parser("edit", help="Open config in editor")
    config_sub.add_parser("reset", help="Reset to defaults")
    config_sub.add_parser("export", help="Export configuration")
    config_sub.add_argument("config_args", nargs="*", help="Config arguments")

    config_parser.add_argument("--key", type=str, default=None, help="Config key to get/set")
    config_parser.add_argument("--value", type=str, default=None, help="Config value to set")
    config_parser.add_argument("--output", type=str, default=None, help="Export output path")
    config_parser.add_argument("--no-secrets", action="store_true", help="Exclude secrets from export")

    # ─── mcp ───
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="MCP server management",
        description="Manage Model Context Protocol servers.",
    )
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_command")
    mcp_sub.add_parser("list", help="List configured MCP servers")
    mcp_sub.add_parser("known", help="List available MCP server templates")
    mcp_sub.add_parser("health", help="Check MCP server health")

    mcp_parser.add_argument("mcp_args", nargs="*", help="MCP arguments")

    # ─── profile ───
    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile management",
        description="Create, switch, and manage configuration profiles.",
    )
    profile_sub = profile_parser.add_subparsers(dest="profile_command")
    profile_sub.add_parser("list", help="List profiles")
    profile_sub.add_parser("current", help="Show current profile")

    profile_parser.add_argument("profile_args", nargs="*", help="Profile arguments")

    # ─── setup ───
    setup_parser = subparsers.add_parser(
        "setup",
        help="Run first-time setup wizard",
        description="Interactive setup wizard for initial configuration.",
    )
    setup_parser.add_argument(
        "--skip-api",
        action="store_true",
        default=False,
        help="Skip API key setup (use existing)",
    )

    return parser


class AtlasCLI:
    """
    Main Atlas CLI application.

    Handles subcommand routing and delegates to appropriate modules.

    Usage:
        cli = AtlasCLI()
        cli.main()
    """

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()

    def main(self, argv: Optional[List[str]] = None):
        """Main entry point."""
        parser = create_parser()
        args = parser.parse_args(argv)

        # Load configuration
        config_path = getattr(args, "config", None)
        profile = getattr(args, "profile", None)
        if config_path:
            self.config = ConfigManager(Path(config_path).parent)
        self.config.load(profile)

        # Route to subcommand
        command = args.command

        if command is None or command == "chat":
            self._cmd_chat(args)
        elif command == "gateway":
            self._cmd_gateway(args)
        elif command == "model":
            self._cmd_model(args)
        elif command == "tools":
            self._cmd_tools(args)
        elif command == "skills":
            self._cmd_skills(args)
        elif command == "cron":
            self._cmd_cron(args)
        elif command == "doctor":
            self._cmd_doctor(args)
        elif command == "config":
            self._cmd_config(args)
        elif command == "mcp":
            self._cmd_mcp(args)
        elif command == "profile":
            self._cmd_profile(args)
        elif command == "setup":
            self._cmd_setup(args)
        else:
            parser.print_help()

    def _cmd_chat(self, args):
        """Handle chat subcommand."""
        from atlas.cli_atlas.tui import AtlasTUI

        # Apply overrides
        if getattr(args, "theme", None):
            self.config.set("skin", args.theme)
            self.config.set("theme", args.theme)
        if getattr(args, "model", None):
            self.config.set("model", args.model)
        if getattr(args, "provider", None):
            self.config.set("provider", args.provider)
        if getattr(args, "agent", None):
            self.config.set("active_agent", args.agent)
        self.config.save()

        tui = AtlasTUI(
            config_manager=self.config,
            verbose=getattr(args, "verbose", False),
        )

        # Handle initial prompt
        if getattr(args, "prompt", None):
            initial_prompt = args.prompt
        else:
            initial_prompt = None

        # Run TUI
        tui.run()

        # Send initial prompt if provided
        if initial_prompt:
            tui._process_input(initial_prompt)
            tui.run()

    def _cmd_gateway(self, args):
        """Handle gateway subcommand."""
        from atlas.cli_atlas.gateway_cmd import GatewayManager
        mgr = GatewayManager(self.config)

        cmd = getattr(args, "gateway_command", None)

        if cmd == "start":
            result = mgr.start(getattr(args, "host", None), getattr(args, "port", None))
            if result["success"]:
                print(f"  Gateway started on {result['host']}:{result['port']}")
                print(f"  Platforms: {', '.join(result['platforms'])}")
            else:
                print(f"  Error: {result.get('error', 'Failed')}")
        elif cmd == "stop":
            result = mgr.stop()
            print("  Gateway stopped." if result["success"] else f"  Error: {result.get('error')}")
        elif cmd == "restart":
            result = mgr.restart(getattr(args, "host", None), getattr(args, "port", None))
            if result["success"]:
                print(f"  Gateway restarted on {result['host']}:{result['port']}")
            else:
                print(f"  Error: {result.get('error')}")
        elif cmd == "status":
            print(mgr.format_status_dashboard())
        elif cmd == "sessions":
            sessions = mgr.list_sessions()
            if sessions:
                for s in sessions:
                    print(f"  {s.get('id', 'unknown'):>20}  {s.get('platform', '')}")
            else:
                print("  No active sessions.")
        elif cmd == "platforms":
            print(mgr.format_platform_table())
        else:
            print(mgr.format_status_dashboard())

    def _cmd_model(self, args):
        """Handle model subcommand."""
        from atlas.cli_atlas.models_cmd import ModelManager
        mgr = ModelManager(self.config)

        cmd = getattr(args, "model_command", None)
        model_args = getattr(args, "model_args", [])

        if cmd == "list" or not cmd:
            print(mgr.format_model_table())
        elif cmd == "current":
            current = mgr.get_current_model()
            info = mgr.get_model_info(current)
            if info:
                print(f"  Current model: {info.get('full_id', current)}")
                print(f"  Name: {info.get('name', 'Unknown')}")
                print(f"  Description: {info.get('description', '')}")
                print(f"  Context: {info.get('context_window', 'N/A'):,} tokens")
                print(f"  Input: ${info.get('input_price', 0):.2f}/M tokens")
                print(f"  Output: ${info.get('output_price', 0):.2f}/M tokens")
            else:
                print(f"  Current model: {current}")
        elif cmd == "pinned":
            pinned = mgr.get_pinned_models()
            if pinned:
                print("  Pinned models:")
                for m in pinned:
                    print(f"    \u2605 {m}")
            else:
                print("  No pinned models. Use /model pin <model>")
        elif cmd == "catalog":
            models = mgr.list_models()
            print(mgr.format_model_table(models))
        elif model_args:
            action = model_args[0] if model_args else ""
            if action == "set" and len(model_args) > 1:
                model_id = model_args[1]
                mgr.set_model(model_id)
                print(f"  Model set to: {model_id}")
            elif action == "test" and len(model_args) > 1:
                model_id = model_args[1]
                result = mgr.test_connectivity(model_id)
                if result["success"]:
                    print(f"  \u2713 Connected to {model_id}")
                    print(f"    Response time: {result.get('response_time', 'N/A')}s")
                else:
                    print(f"  \u2717 Failed: {result.get('error', 'Unknown error')}")
            elif action == "pin" and len(model_args) > 1:
                model_id = model_args[1]
                mgr.pin_model(model_id)
                print(f"  Pinned: {model_id}")
            elif action == "unpin" and len(model_args) > 1:
                model_id = model_args[1]
                mgr.unpin_model(model_id)
                print(f"  Unpinned: {model_id}")
            elif action == "compare" and len(model_args) > 1:
                model_ids = model_args[1:]
                print(mgr.format_model_table(mgr.compare_models(model_ids)))
            else:
                print("  Usage: atlas model [list|set|test|pin|unpin|compare] [args]")
        else:
            print(mgr.format_model_table())

    def _cmd_tools(self, args):
        """Handle tools subcommand."""
        from atlas.cli_atlas.tools_config import ToolConfigManager
        mgr = ToolConfigManager(self.config)

        cmd = getattr(args, "tools_command", None)
        tools_args = getattr(args, "tools_args", [])

        if cmd == "list" or not cmd:
            print(mgr.format_tools_table())
        elif cmd == "categories":
            print(mgr.format_category_summary())
        elif cmd == "stats":
            stats = mgr.get_usage_stats()
            print(f"  Total tool calls: {stats.get('total_calls', 0)}")
            by_tool = stats.get("by_tool", {})
            if by_tool:
                print("\n  By tool:")
                for name, data in sorted(by_tool.items(), key=lambda x: x[1].get("count", 0), reverse=True)[:20]:
                    print(f"    {name:<25} {data.get('count', 0):>6} calls")
        elif tools_args:
            action = tools_args[0]
            if action == "enable" and len(tools_args) > 1:
                for tool in tools_args[1:]:
                    mgr.enable_tool(tool)
                print(f"  Enabled: {', '.join(tools_args[1:])}")
            elif action == "disable" and len(tools_args) > 1:
                for tool in tools_args[1:]:
                    mgr.disable_tool(tool)
                print(f"  Disabled: {', '.join(tools_args[1:])}")
            elif action == "info" and len(tools_args) > 1:
                all_tools = mgr.list_all_tools()
                for tool in all_tools:
                    if tool["name"] == tools_args[1]:
                        print(f"  Tool: {tool['name']}")
                        print(f"  Category: {tool['category']}")
                        print(f"  Enabled: {tool['enabled']}")
                        print(f"  Permission: {tool['permission']}")
                        print(f"  Description: {tool['description']}")
                        break
        else:
            print(mgr.format_tools_table())

    def _cmd_skills(self, args):
        """Handle skills subcommand."""
        from atlas.cli_atlas.skills_config import SkillConfigManager
        mgr = SkillConfigManager(self.config)

        cmd = getattr(args, "skills_command", None)

        if cmd == "list" or not cmd:
            print(mgr.format_skills_table())
        elif cmd == "files":
            skills = mgr.list_installed_skills()
            for skill in skills:
                files = mgr.get_skill_files(skill["id"])
                print(f"\n  {skill['name']} ({skill['id']}):")
                for f in files:
                    print(f"    {f}")
        else:
            print(mgr.format_skills_table())

    def _cmd_cron(self, args):
        """Handle cron subcommand."""
        from atlas.cli_atlas.cron_cmd import CronManager
        mgr = CronManager(self.config)

        cmd = getattr(args, "cron_command", None)
        cron_args = getattr(args, "cron_args", [])

        if cmd == "list" or not cmd:
            print(mgr.format_jobs_table())
        elif cmd == "history":
            history = mgr.get_execution_history()
            if history:
                print(f"\n  {'Job':<25} {'Last Run':<20} {'Runs':>6} {'Errors':>6}")
                print("  " + "-" * 62)
                for h in history:
                    last = h.get("last_run", "never")[:16]
                    print(f"  {h['job_name']:<25} {last:<20} {h['run_count']:>6} {h['error_count']:>6}")
            else:
                print("  No execution history.")
        elif cron_args:
            action = cron_args[0]
            if action == "add" and len(cron_args) > 1:
                desc = " ".join(cron_args[1:])
                job = mgr.create_from_natural_language(desc)
                if job:
                    print(f"  Created: {job.name} ({job.schedule})")
                else:
                    print("  Could not parse schedule.")
            elif action == "pause" and len(cron_args) > 1:
                mgr.pause_job(cron_args[1])
                print(f"  Paused: {cron_args[1]}")
            elif action == "resume" and len(cron_args) > 1:
                mgr.resume_job(cron_args[1])
                print(f"  Resumed: {cron_args[1]}")
            elif action == "delete" and len(cron_args) > 1:
                mgr.delete_job(cron_args[1])
                print(f"  Deleted: {cron_args[1]}")

    def _cmd_doctor(self, args):
        """Handle doctor subcommand."""
        from atlas.cli_atlas.doctor import Doctor
        doc = Doctor(self.config)

        if getattr(args, "fix", False):
            fixes = doc.fix_issues()
            print(f"  Applied {len(fixes)} fixes:")
            for f in fixes:
                print(f"    \u2713 {f}")
        elif getattr(args, "report", False):
            print(doc.generate_report())
        elif getattr(args, "quick", False):
            results = doc.run_quick()
            print(doc.format_results(results))
        else:
            results = doc.run_all()
            print(doc.format_results(results))

    def _cmd_config(self, args):
        """Handle config subcommand."""
        cmd = getattr(args, "config_command", None)
        config_args = getattr(args, "config_args", [])

        if cmd == "list" or (not cmd and not config_args):
            sections = self.config.list_sections()
            print(f"\n  \033[1mConfiguration Sections:\033[0m\n")
            for s in sections:
                val = self.config.get(s, "")
                display = str(val)[:60] if val else "(empty)"
                print(f"    {s:<25} {display}")
            print()
        elif cmd == "show":
            import json
            data = self.config.to_dict()
            print(json.dumps(data, indent=2, default=str))
        elif cmd == "edit":
            editor = self.config.get("editor", "vim")
            cfg = self.config.config_file if self.config.config_file.exists() else self.config.config_json_file
            import subprocess
            subprocess.run([editor, str(cfg)])
        elif cmd == "reset":
            self.config.reset()
            print("  Configuration reset to defaults.")
        elif cmd == "export":
            output = getattr(args, "output", None) or "atlas_config_export.yaml"
            no_secrets = getattr(args, "no_secrets", False)
            path = self.config.export(output, include_secrets=not no_secrets)
            print(f"  Exported to: {path}")
        elif getattr(args, "key", None) and getattr(args, "value", None) is not None:
            self.config.set(args.key, args.value)
            self.config.save()
            print(f"  Set {args.key} = {args.value}")
        elif getattr(args, "key", None):
            value = self.config.get(args.key, "(not set)")
            print(f"  {args.key} = {value}")
        elif config_args:
            key = config_args[0]
            if len(config_args) > 1:
                value = " ".join(config_args[1:])
                self.config.set(key, value)
                self.config.save()
                print(f"  Set {key} = {value}")
            else:
                value = self.config.get(key, "(not set)")
                print(f"  {key} = {value}")
        else:
            print("  Usage: atlas config [list|show|edit|reset|export|get|set]")

    def _cmd_mcp(self, args):
        """Handle mcp subcommand."""
        from atlas.cli_atlas.mcp_config import MCPConfigManager
        mgr = MCPConfigManager(self.config)

        cmd = getattr(args, "mcp_command", None)
        mcp_args = getattr(args, "mcp_args", [])

        if cmd == "list" or not cmd:
            print(mgr.format_servers_table())
        elif cmd == "known":
            servers = mgr.get_known_servers()
            print(f"\n  \033[1mAvailable MCP Servers:\033[0m\n")
            for s in servers:
                installed = " \u2713" if s["installed"] else ""
                print(f"    {s['id']:<15} {s['description']:<40}{installed}")
            print()
        elif cmd == "health":
            results = mgr.health_check_all()
            for r in results:
                status = "\u2713" if r.get("status") == "healthy" else "\u2717"
                print(f"  {status} {r.get('server_id', 'unknown')}: {r.get('status', 'unknown')}")
        elif mcp_args:
            action = mcp_args[0]
            if action == "add" and len(mcp_args) > 1:
                server_id = mcp_args[1]
                if mgr.install_from_known(server_id):
                    print(f"  Added: {server_id}")
                else:
                    print(f"  Unknown server: {server_id}")
            elif action == "remove" and len(mcp_args) > 1:
                server_id = mcp_args[1]
                if mgr.remove_server(server_id):
                    print(f"  Removed: {server_id}")
                else:
                    print(f"  Not found: {server_id}")

    def _cmd_profile(self, args):
        """Handle profile subcommand."""
        from atlas.cli_atlas.profiles import ProfileManager
        mgr = ProfileManager()

        cmd = getattr(args, "profile_command", None)
        profile_args = getattr(args, "profile_args", [])

        if cmd == "list" or (not cmd and not profile_args):
            profiles = mgr.list_profiles()
            current = self.config.get("active_profile", "default")
            print(f"\n  \033[1mProfiles:\033[0m\n")
            for p in profiles:
                marker = " \u2713" if p["name"] == current else ""
                desc = p.get("description", "")
                print(f"    {p['name']:<20} {desc}{marker}")
            print()
        elif cmd == "current":
            current = self.config.get("active_profile", "default")
            print(f"  Current profile: {current}")
        elif profile_args:
            action = profile_args[0]
            if action == "switch" and len(profile_args) > 1:
                name = profile_args[1]
                self.config.set("active_profile", name)
                self.config.load(name)
                self.config.save()
                print(f"  Switched to profile: {name}")
            elif action == "create" and len(profile_args) > 1:
                name = profile_args[1]
                desc = " ".join(profile_args[2:]) if len(profile_args) > 2 else ""
                mgr.create(name, description=desc)
                print(f"  Created profile: {name}")
            elif action == "delete" and len(profile_args) > 1:
                name = profile_args[1]
                if mgr.delete(name, force=True):
                    print(f"  Deleted profile: {name}")
                else:
                    print(f"  Cannot delete built-in profile: {name}")
            elif action == "export" and len(profile_args) > 1:
                name = profile_args[1]
                output = profile_args[2] if len(profile_args) > 2 else f"{name}_export.yaml"
                if mgr.export_profile(name, output):
                    print(f"  Exported to: {output}")
                else:
                    print(f"  Profile not found: {name}")
        else:
            print("  Usage: atlas profile [list|current|switch|create|delete|export]")

    def _cmd_setup(self, args):
        """Handle setup subcommand."""
        from atlas.cli_atlas.setup import SetupWizard
        wizard = SetupWizard(self.config)
        wizard.run(skip_api=getattr(args, "skip_api", False))


def main(argv: Optional[List[str]] = None):
    """Entry point for the Atlas CLI."""
    try:
        cli = AtlasCLI()
        cli.main(argv)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  Error: {e}", file=sys.stderr)
        if os.environ.get("ATLAS_DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
