"""
Claude Clone — A Python clone of Claude Code (CLI) and Cowork (GUI).

Entry point for the application. Supports two modes:
  - GUI mode (default): Launch the Cowork desktop application
  - CLI mode (--cli): Launch the Claude Code terminal interface

Usage:
    python main.py              # Launch GUI
    python main.py --cli        # Launch CLI
    python main.py --cli --vim  # Launch CLI with vim keybindings
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Claude Clone — Agentic coding assistant (Python clone of Claude Code + Cowork)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              Launch the desktop GUI (Cowork)
  python main.py --cli        Launch the terminal CLI (Claude Code)
  python main.py --cli --vim  Launch CLI with vim keybindings
  python main.py --version    Show version info

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
        version="Claude Clone v2.0.0 — Hermes Integration + Self-Improving + Agent Teams + OpenRouter",
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
        "--hermes",
        action="store_true",
        default=False,
        help="Enable Hermes Agent mode (context compression, smart routing, skills, etc.)",
    )

    parser.add_argument(
        "--gateway",
        action="store_true",
        default=False,
        help="Start the Hermes Gateway server (multi-platform messaging)",
    )

    parser.add_argument(
        "--acp",
        action="store_true",
        default=False,
        help="Start the Hermes ACP server (editor/IDE integration)",
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

    # Load configuration
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
    if args.hermes:
        config.hermes = {"enabled": True}
    if getattr(config, 'knowledge_base', None) and config.knowledge_base.get("enabled"):
        print(f"Knowledge Base: enabled")
    if config.hermes.get("enabled"):
        print(f"Hermes Agent: enabled")

    # ── Hermes ACP Server Mode ──
    if args.acp:
        try:
            import uvicorn
            from hermes.acp.server import create_acp_app
            app = create_acp_app(
                host=config.hermes_acp.get("host", "0.0.0.0"),
                port=config.hermes_acp.get("port", 8765),
                api_key=config.hermes_acp.get("api_key", ""),
                cors_origins=config.hermes_acp.get("cors_origins", ["*"]),
            )
            print(f"Starting Hermes ACP Server on {config.hermes_acp.get('host', '0.0.0.0')}:{config.hermes_acp.get('port', 8765)}")
            uvicorn.run(app, host=config.hermes_acp.get("host", "0.0.0.0"), port=config.hermes_acp.get("port", 8765))
        except ImportError as e:
            print(f"Missing dependency for ACP server: {e}")
            print("Install with: pip install fastapi uvicorn")
            sys.exit(1)
        except Exception as e:
            print(f"Error starting ACP server: {e}")
            sys.exit(1)
        return

    # ── Hermes Gateway Mode ──
    if args.gateway:
        try:
            from hermes.gateway.runner import GatewayRunner
            from hermes.gateway.config import GatewayConfig
            gateway_config = GatewayConfig(
                platforms=config.hermes_gateway.get("platforms", {}),
                session_timeout=config.hermes_gateway.get("session_timeout", 3600),
                max_concurrent_sessions=config.hermes_gateway.get("max_concurrent_sessions", 100),
            )
            runner = GatewayRunner(gateway_config)
            print(f"Starting Hermes Gateway with {len(gateway_config.platforms)} platform(s)")
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


# Required for GUI import (tkinter might not be named tk)
try:
    import tkinter as tk
except ImportError:
    tk = None


if __name__ == "__main__":
    main()
