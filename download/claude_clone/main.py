"""
Claude Clone v3.0.0 — Atlas Integration + OpenClaw Features + 233 Modules

Entry point for the application. Supports multiple modes:
  - GUI mode (default): Launch the Cowork desktop application
  - CLI mode (--cli): Launch the Claude Code terminal interface
  - Atlas Agent mode (--atlas): Enable the Atlas agentic sub-system
  - Atlas TUI (--atlas-cli): Full-featured Atlas terminal interface
  - Doctor mode (--doctor): Run diagnostics on all subsystems
  - Stats mode (--stats): Show project statistics

Usage:
    python main.py              # Launch GUI
    python main.py --cli        # Launch CLI
    python main.py --cli --vim  # Launch CLI with vim keybindings
    python main.py --atlas      # Enable Atlas Agent mode
    python main.py --atlas-cli  # Launch Atlas TUI
    python main.py --doctor     # Run diagnostics
    python main.py --stats      # Show project statistics
"""

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Claude Clone v3.0.0 — Agentic coding assistant (Atlas Integration + OpenClaw Features + 233 Modules)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              Launch the desktop GUI (Cowork)
  python main.py --cli        Launch the terminal CLI (Claude Code)
  python main.py --cli --vim  Launch CLI with vim keybindings
  python main.py --atlas      Enable Atlas Agent mode
  python main.py --atlas-cli  Launch the Atlas TUI
  python main.py --doctor     Run full diagnostics
  python main.py --stats      Show project statistics
  python main.py --version    Show version info

Backward Compatibility:
  --hermes    Alias for --atlas
  --hermes-cli Alias for --atlas-cli

Configuration:
  API Key: Set ANTHROPIC_API_KEY env var or configure in ~/.claude_clone/config.json
  Settings: Run the app and go to Tools → Settings (GUI) or /help (CLI)
        """,
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        default=False,
        help="Launch in CLI mode (Claude Code terminal interface)",
    )

    parser.add_argument(
        "--vim",
        action="store_true",
        default=False,
        help="Enable vim keybindings in CLI mode",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="AI model to use (e.g., claude-sonnet-4-20250514, claude-opus-4-20250514)",
    )

    parser.add_argument(
        "--theme",
        type=str,
        choices=["dark", "light"],
        default=None,
        help="Color theme (dark or light)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per response (default: 8192)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum agentic loop iterations (default: 10)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Claude Clone v3.0.0 — Atlas Integration + OpenClaw Features + 233 Modules",
    )

    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Set working directory",
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["openrouter", "anthropic"],
        default=None,
        help="API provider (default: openrouter)",
    )

    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Start with a specific agent (e.g., debug, codegen, search)",
    )

    parser.add_argument(
        "--self-improve",
        action="store_true",
        default=False,
        help="Enable the self-improving system (auto-evaluate, patch, extend, optimize)",
    )

    parser.add_argument(
        "--knowledge-base",
        action="store_true",
        default=False,
        help="Enable the knowledge base system (persistent knowledge storage, search, extraction)",
    )

    parser.add_argument(
        "--atlas",
        "--hermes",  # backward-compat alias
        action="store_true",
        default=False,
        help="Enable Atlas Agent mode (context compression, smart routing, skills, etc.)",
    )

    parser.add_argument(
        "--atlas-cli",
        "--hermes-cli",  # backward-compat alias
        action="store_true",
        default=False,
        help="Launch the Atlas Agent TUI (full-featured terminal interface)",
    )

    parser.add_argument(
        "--gateway",
        action="store_true",
        default=False,
        help="Start the Atlas Gateway server (multi-platform messaging)",
    )

    parser.add_argument(
        "--acp",
        action="store_true",
        default=False,
        help="Start the Atlas ACP server (editor/IDE integration)",
    )

    # Atlas sub-system controls
    parser.add_argument(
        "--security",
        action="store_true",
        default=False,
        help="Enable security policy enforcement",
    )

    parser.add_argument(
        "--plugin-dir",
        type=str,
        default=None,
        help="Load plugins from directory",
    )

    parser.add_argument(
        "--canvas",
        action="store_true",
        default=False,
        help="Enable Canvas/A2UI visual workspace",
    )

    parser.add_argument(
        "--voice",
        action="store_true",
        default=False,
        help="Enable real-time voice mode",
    )

    parser.add_argument(
        "--locale",
        type=str,
        default=None,
        help="Set UI locale (e.g., en, es, zh)",
    )

    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Load config from file",
    )

    parser.add_argument(
        "--sandbox-type",
        type=str,
        choices=["none", "docker", "process", "restricted"],
        default=None,
        help="Sandbox type for code execution",
    )

    parser.add_argument(
        "--doctor",
        action="store_true",
        default=False,
        help="Run diagnostics (Python version, dependencies, Atlas subsystems, module status)",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        default=False,
        help="Show project statistics (files, LOC, tools, models, channels, plugins, skills)",
    )

    args = parser.parse_args()

    # Set working directory if specified
    if args.cwd:
        os.chdir(args.cwd)

    # Import configuration
    try:
        from config import Config
    except ImportError:
        print("Error: Could not import config module. Make sure you're running from the claude_clone directory.")
        print("  cd claude_clone && python main.py")
        sys.exit(1)

    # Load configuration (from file if --config-file specified)
    if args.config_file:
        config = Config.load(path=args.config_file)
    else:
        config = Config.from_env()

    # Apply command-line overrides
    if args.provider:
        config.provider = args.provider
        config.base_url = config._get_base_url()
    if args.model:
        config.model = args.model
    if args.theme:
        config.theme = args.theme
    if args.max_tokens:
        config.max_tokens = args.max_tokens
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    if args.agent:
        config.active_agent = args.agent
    if args.self_improve:
        config.self_improving = {"enabled": True}
    if args.knowledge_base:
        config.knowledge_base = {"enabled": True}
    if args.security:
        config.security = {**config.security, "enabled": True, "auto_scan": True}
    if args.plugin_dir:
        config.plugins = {**config.plugins, "plugin_dir": args.plugin_dir}
    if args.canvas:
        config.atlas_canvas = {"enabled": True}
    if args.voice:
        config.atlas_voice = {"enabled": True}
    if args.locale:
        config.atlas_i18n = {**getattr(config, 'atlas_i18n', {}), "locale": args.locale}
    if args.sandbox_type:
        config.atlas_sandbox = {"type": args.sandbox_type, "enabled": args.sandbox_type != "none"}

    # Validate configuration
    warnings = config.validate()
    for warning in warnings:
        print(f"Warning: {warning}")

    print(f"Provider: {config.provider} ({config.base_url})")
    if config.active_agent:
        print(f"Agent:    {config.active_agent}")
    print(f"Model:    {config.model}")
    if getattr(config, 'self_improving', None) and config.self_improving.get("enabled"):
        print(f"Self-Improve: enabled")
    if args.atlas:
        config.Atlas = {"enabled": True}
        config.atlas_mode = True
    if getattr(config, 'knowledge_base', None) and config.knowledge_base.get("enabled"):
        print(f"Knowledge Base: enabled")
    if config.Atlas.get("enabled"):
        print(f"Atlas Agent: enabled")
    if getattr(config, 'atlas_canvas', None) and config.atlas_canvas.get("enabled"):
        print(f"Canvas/A2UI:  enabled")
    if getattr(config, 'atlas_voice', None) and config.atlas_voice.get("enabled"):
        print(f"Voice Mode:    enabled")
    if getattr(config, 'atlas_i18n', None) and config.atlas_i18n.get("locale"):
        print(f"Locale:        {config.atlas_i18n['locale']}")
    if getattr(config, 'atlas_sandbox', None) and config.atlas_sandbox.get("enabled"):
        print(f"Sandbox:       {config.atlas_sandbox.get('type', 'default')}")
    if config.security.get("enabled") and config.security.get("auto_scan"):
        print(f"Security:      enforced")

    # ── Doctor Mode ──
    if args.doctor:
        run_doctor_mode(config)
        return

    # ── Stats Mode ──
    if args.stats:
        run_stats_mode(config)
        return

    # ── Atlas ACP Server Mode ──
    if args.acp:
        try:
            import uvicorn
            from atlas.acp.server import create_acp_app
            app = create_acp_app(
                host=config.atlas_acp.get("host", "0.0.0.0"),
                port=config.atlas_acp.get("port", 8765),
                api_key=config.atlas_acp.get("api_key", ""),
                cors_origins=config.atlas_acp.get("cors_origins", ["*"]),
            )
            print(f"Starting Atlas ACP Server on {config.atlas_acp.get('host', '0.0.0.0')}:{config.atlas_acp.get('port', 8765)}")
            uvicorn.run(app, host=config.atlas_acp.get("host", "0.0.0.0"), port=config.atlas_acp.get("port", 8765))
        except ImportError as e:
            print(f"Missing dependency for ACP server: {e}")
            print("Install with: pip install fastapi uvicorn")
            sys.exit(1)
        except Exception as e:
            print(f"Error starting ACP server: {e}")
            sys.exit(1)
        return

    # ── Atlas Gateway Mode ──
    if args.gateway:
        try:
            from atlas.gateway.runner import GatewayRunner
            from atlas.gateway.config import GatewayConfig
            gateway_config = GatewayConfig(
                platforms=config.atlas_gateway.get("platforms", {}),
                session_timeout=config.atlas_gateway.get("session_timeout", 3600),
                max_concurrent_sessions=config.atlas_gateway.get("max_concurrent_sessions", 100),
            )
            runner = GatewayRunner(gateway_config)
            print(f"Starting Atlas Gateway with {len(gateway_config.platforms)} platform(s)")
            import asyncio
            asyncio.run(runner.start())
        except ImportError as e:
            print(f"Missing dependency for gateway: {e}")
            print("Install with: pip install -r requirements.txt")
            sys.exit(1)
        except Exception as e:
            print(f"Error starting gateway: {e}")
            sys.exit(1)
        return

    # ── Atlas CLI Mode ──
    if args.atlas_cli:
        try:
            from atlas.cli_atlas.main import AtlasCLI
            cli = AtlasCLI(config=config)
            cli.run()
        except ImportError as e:
            print(f"Missing dependency for Atlas CLI: {e}")
            sys.exit(1)
        return

    if args.cli:
        # ── Launch CLI Mode ──
        try:
            from cli.app import ClaudeCodeCLI
            if args.vim:
                config.theme = config.theme or "dark"  # Vim mode works best with dark theme
            cli = ClaudeCodeCLI(config=config)
            if args.vim:
                cli.vim_mode = True
            cli.run()
        except ImportError as e:
            missing = str(e).split("'")[-2] if "'" in str(e) else str(e)
            print(f"Missing dependency: {missing}")
            print("Install with: pip install -r requirements.txt")
            sys.exit(1)
    else:
        # ── Launch GUI Mode ──
        try:
            from gui.app import CoworkApp
            app = CoworkApp(config=config)
            app.run()
        except ImportError as e:
            missing = str(e).split("'")[-2] if "'" in str(e) else str(e)
            print(f"Missing dependency: {missing}")
            print("Install with: pip install -r requirements.txt")
            sys.exit(1)
        except tk.TclError:
            print("Error: Cannot display GUI. Are you running in a headless environment?")
            print("Try using --cli mode instead: python main.py --cli")
            sys.exit(1)


def run_doctor_mode(config):
    """Run full diagnostics on the Claude Clone installation."""
    sep = "═" * 60
    print()
    print(f"  {sep}")
    print(f"  Claude Clone v3.0.0 — Doctor Diagnostics")
    print(f"  {sep}")
    print()

    # 1. Python version
    py_ver = sys.version_info
    py_ok = py_ver >= (3, 9)
    py_status = "✅" if py_ok else "❌"
    print(f"  {py_status} Python Version:   {py_ver.major}.{py_ver.minor}.{py_ver.micro} {'(ok)' if py_ok else '(need >= 3.9)'}")

    # 2. Platform
    print(f"     Platform:         {platform.system()} {platform.release()}")
    print(f"     Architecture:     {platform.machine()}")
    print()

    # 3. Required dependencies
    required_deps = [
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("rich", "rich"),
        ("prompt_toolkit", "prompt_toolkit"),
        ("tiktoken", "tiktoken"),
        ("httpx", "httpx"),
        ("pydantic", "pydantic"),
        ("aiohttp", "aiohttp"),
    ]
    optional_deps = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("websockets", "websockets"),
        ("PIL / Pillow", "PIL"),
        ("python-dotenv", "dotenv"),
        ("pyttsx3", "pyttsx3"),
        ("SpeechRecognition", "speech_recognition"),
        ("pyaudio", "pyaudio"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
    ]

    print(f"  ── Required Dependencies ──")
    req_ok = 0
    for name, module in required_deps:
        try:
            importlib.import_module(module)
            print(f"  ✅ {name}")
            req_ok += 1
        except ImportError:
            print(f"  ❌ {name}")
    print(f"  Result: {req_ok}/{len(required_deps)} installed")
    print()

    print(f"  ── Optional Dependencies ──")
    opt_ok = 0
    for name, module in optional_deps:
        try:
            importlib.import_module(module)
            print(f"  ✅ {name}")
            opt_ok += 1
        except ImportError:
            print(f"  ⬜ {name}")
    print(f"  Result: {opt_ok}/{len(optional_deps)} installed")
    print()

    # 4. Atlas subsystems
    atlas_subsystems = [
        ("Atlas Core", "atlas.core"),
        ("Atlas Tools", "atlas.tools"),
        ("Atlas Skills", "atlas.skills"),
        ("Atlas Channels", "atlas.channels"),
        ("Atlas Gateway", "atlas.gateway"),
        ("Atlas ACP", "atlas.acp"),
        ("Atlas CLI", "atlas.cli_atlas"),
        ("Atlas Security", "atlas.security"),
        ("Atlas Plugin SDK", "atlas.plugin_sdk"),
        ("Atlas Canvas", "atlas.canvas"),
        ("Atlas Media", "atlas.media"),
        ("Atlas Config", "atlas.config"),
        ("Atlas Cron", "atlas.cron"),
        ("Atlas Sessions", "atlas.sessions"),
        ("Atlas Realtime/Voice", "atlas.realtime"),
        ("Atlas i18n", "atlas.i18n"),
        ("Atlas Hooks", "atlas.hooks"),
        ("Atlas Memory Plugins", "atlas.plugins.memory"),
        ("Atlas Tasks", "atlas.tasks"),
        ("Atlas Polls", "atlas.polls"),
        ("Atlas Pairing", "atlas.pairing"),
        ("Atlas Node Host", "atlas.node_host"),
        ("Atlas Link Understanding", "atlas.link_understanding"),
        ("Atlas Web", "atlas.web"),
    ]

    print(f"  ── Atlas Subsystems ──")
    atlas_ok = 0
    for name, module in atlas_subsystems:
        try:
            importlib.import_module(module)
            print(f"  ✅ {name} ({module})")
            atlas_ok += 1
        except Exception:
            print(f"  ❌ {name} ({module})")
    print(f"  Result: {atlas_ok}/{len(atlas_subsystems)} loaded")
    print()

    # 5. Module status — count all Python files and lines
    project_root = Path(__file__).parent
    py_files = list(project_root.rglob("*.py"))
    total_files = len(py_files)
    total_lines = 0
    for pf in py_files:
        try:
            total_lines += len(pf.read_text(encoding="utf-8", errors="ignore").splitlines())
        except Exception:
            pass

    # Count directories as modules
    modules = set()
    for pf in py_files:
        parts = pf.relative_to(project_root).parts[:-1]
        if parts:
            modules.add(parts[0])
    total_modules = len(modules)

    print(f"  ── Module Status ──")
    print(f"  Total Python files:  {total_files}")
    print(f"  Total lines of code: {total_lines:,}")
    print(f"  Top-level modules:   {total_modules}")
    print(f"  Module list:         {', '.join(sorted(modules))}")
    print()

    # 6. Config check
    print(f"  ── Configuration ──")
    print(f"  Config dir:   {config._config_path}")
    print(f"  Config exists: {config._config_path.exists() if hasattr(config, '_config_path') else 'N/A'}")
    print(f"  API key set:  {bool(config.api_key)}")
    print(f"  Provider:     {config.provider}")
    print(f"  Model:        {config.model}")
    print()

    # Summary
    all_ok = py_ok and req_ok >= 4 and atlas_ok >= 16
    status_icon = "✅" if all_ok else "⚠️"
    print(f"  {sep}")
    print(f"  {status_icon} Doctor Summary")
    print(f"  {sep}")
    print(f"  Required deps: {req_ok}/{len(required_deps)}")
    print(f"  Optional deps: {opt_ok}/{len(optional_deps)}")
    print(f"  Atlas modules: {atlas_ok}/{len(atlas_subsystems)}")
    print(f"  Python files:  {total_files} ({total_lines:,} LOC)")
    print(f"  Overall:       {'HEALTHY' if all_ok else 'ISSUES FOUND'}")
    print()


def run_stats_mode(config):
    """Show project statistics."""
    sep = "═" * 60
    print()
    print(f"  {sep}")
    print(f"  Claude Clone v3.0.0 — Project Statistics")
    print(f"  {sep}")
    print()

    project_root = Path(__file__).parent

    # Count Python files and LOC
    py_files = list(project_root.rglob("*.py"))
    total_py_files = len(py_files)
    total_lines = 0
    for pf in py_files:
        try:
            total_lines += len(pf.read_text(encoding="utf-8", errors="ignore").splitlines())
        except Exception:
            pass

    # Count tools (scan atlas/tools/ and agent/tools.py)
    tool_count = 0
    atlas_tools_dir = project_root / "atlas" / "tools"
    if atlas_tools_dir.exists():
        tool_count = len(list(atlas_tools_dir.glob("*_tool.py"))) + len(list(atlas_tools_dir.glob("*_tools.py")))
    agent_tools = project_root / "agent" / "tools.py"
    if agent_tools.exists():
        # Estimate tools from agent/tools.py
        try:
            content = agent_tools.read_text(encoding="utf-8", errors="ignore")
            tool_count += content.count('def tool_') + content.count('"name":') + content.count("'name':") // 2
        except Exception:
            pass

    # Count supported models (scan model_router.py or similar)
    model_count = 0
    model_router = project_root / "agent" / "model_router.py"
    if model_router.exists():
        try:
            content = model_router.read_text(encoding="utf-8", errors="ignore")
            model_count = max(1, content.count("claude-") + content.count("gpt-") + content.count("gemini-") + content.count("llama-"))
        except Exception:
            pass
    if model_count == 0:
        # Fallback: known models
        model_count = 8  # claude family
    model_metadata = project_root / "atlas" / "core" / "model_metadata.py"
    if model_metadata.exists():
        try:
            content = model_metadata.read_text(encoding="utf-8", errors="ignore")
            mm_count = content.count("claude-") + content.count("gpt-") + content.count("gemini-")
            if mm_count > model_count:
                model_count = mm_count
        except Exception:
            pass

    # Count channels (scan atlas/channels/)
    channel_count = 0
    channels_dir = project_root / "atlas" / "channels"
    if channels_dir.exists():
        # Count platform adapters
        platforms_dir = project_root / "atlas" / "gateway" / "platforms"
        if platforms_dir.exists():
            channel_count = len(list(platforms_dir.glob("*.py"))) - 1  # minus __init__.py

    # Count plugins (scan atlas/plugins/)
    plugin_count = 0
    plugins_dir = project_root / "atlas" / "plugins"
    if plugins_dir.exists():
        plugin_count = len(list(plugins_dir.rglob("*.py"))) - len(list(plugins_dir.rglob("__init__.py")))

    # Count skills (scan atlas/skills/)
    skill_count = 0
    skills_dir = project_root / "atlas" / "skills"
    if skills_dir.exists():
        skill_count = len(list(skills_dir.rglob("SKILL.md")))
    if skill_count == 0:
        # Fallback: count skill directories
        builtins_dir = project_root / "atlas" / "skills" / "builtins"
        if builtins_dir.exists():
            skill_count = len([d for d in builtins_dir.iterdir() if d.is_dir() and not d.name.startswith("_")])

    print(f"  ── Code Statistics ──")
    print(f"  Total Python files:  {total_py_files}")
    print(f"  Total lines of code: {total_lines:,}")
    print()

    print(f"  ── Feature Statistics ──")
    print(f"  Tools:               {tool_count}")
    print(f"  Supported models:    {model_count}")
    print(f"  Channels/platforms:  {channel_count}")
    print(f"  Plugins:             {plugin_count}")
    print(f"  Skills:              {skill_count}")
    print()

    # Atlas subsystem breakdown
    atlas_dir = project_root / "atlas"
    if atlas_dir.exists():
        atlas_dirs = [d for d in atlas_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
        atlas_py = list(atlas_dir.rglob("*.py"))
        atlas_lines = 0
        for pf in atlas_py:
            try:
                atlas_lines += len(pf.read_text(encoding="utf-8", errors="ignore").splitlines())
            except Exception:
                pass
        print(f"  ── Atlas Breakdown ──")
        print(f"  Atlas subsystems:   {len(atlas_dirs)}")
        print(f"  Atlas Python files: {len(atlas_py)}")
        print(f"  Atlas lines of code:{atlas_lines:,}")
        print()

    # Per-module breakdown
    print(f"  ── Top-Level Module Breakdown ──")
    module_stats = {}
    for top_dir in project_root.iterdir():
        if not top_dir.is_dir() or top_dir.name.startswith("_") or top_dir.name.startswith("."):
            continue
        py_files_in_dir = list(top_dir.rglob("*.py"))
        if not py_files_in_dir:
            continue
        lines = 0
        for pf in py_files_in_dir:
            try:
                lines += len(pf.read_text(encoding="utf-8", errors="ignore").splitlines())
            except Exception:
                pass
        module_stats[top_dir.name] = (len(py_files_in_dir), lines)

    for name, (files, lines) in sorted(module_stats.items(), key=lambda x: -x[1][1]):
        print(f"  {name:<22s} {files:>4d} files  {lines:>6,} lines")

    print()
    print(f"  {sep}")
    print()


# Required for GUI import (tkinter might not be named tk)
try:
    import tkinter as tk
except ImportError:
    tk = None


if __name__ == "__main__":
    main()
