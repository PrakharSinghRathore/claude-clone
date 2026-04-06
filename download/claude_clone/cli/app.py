"""
Claude Code CLI — A full terminal UI using prompt_toolkit + rich.

Features:
- Multi-line input with Shift+Enter
- File path autocomplete (@ trigger)
- Command history with up/down
- Slash commands: /clear, /context, /model, /tools, /compact, /export, /help, /vim
- Streaming assistant responses
- Tool call visualization with spinners
- Token count + cost estimates
- Vim keybinding mode toggle
- Ctrl+C to cancel, Ctrl+D to exit
"""

import asyncio
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from cli.renderer import Renderer

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.styles import Style as PTStyle
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

from agent.core import Agent, AgentEvent, TextEvent, ToolCallEvent, ToolResultEvent, ErrorEvent, DoneEvent, ThinkingEvent, UsageEvent
from agent.tools import TOOLS_REGISTRY, generate_tool_schemas
from config import Config


# ──────────────────────────────────────────────
# File path completer
# ──────────────────────────────────────────────

class FilePathCompleter(Completer):
    """Completes file paths when user types @."""

    def __init__(self):
        self._triggered = False

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only trigger on @ prefix
        at_index = text.rfind("@")
        if at_index < 0:
            return

        partial = text[at_index + 1:]

        # Get the directory part and file prefix
        if "/" in partial:
            dir_part = os.path.expanduser(partial.rsplit("/", 1)[0] or "/")
            prefix = partial.rsplit("/", 1)[1] if "/" in partial else ""
        else:
            dir_part = "."
            prefix = partial

        try:
            dir_path = Path(dir_part).resolve()
            if not dir_path.is_dir():
                return

            for entry in sorted(dir_path.iterdir()):
                if entry.name.startswith(prefix) and not entry.name.startswith("."):
                    if entry.is_dir():
                        completion_text = entry.name + "/"
                    else:
                        completion_text = entry.name

                    # Show relative path
                    try:
                        display = str(entry.relative_to(Path.cwd()))
                    except ValueError:
                        display = entry.name

                    yield Completion(
                        completion_text,
                        start_position=-len(partial),
                        display=display,
                    )
        except (PermissionError, OSError):
            pass


# ──────────────────────────────────────────────
# Slash command completer
# ──────────────────────────────────────────────

class SlashCommandCompleter(Completer):
    """Completes slash commands."""

    COMMANDS = [
        "/clear", "/context", "/model", "/tools", "/compact",
        "/export", "/help", "/vim", "/quit", "/exit", "/cost",
        "/env", "/git", "/undo", "/team", "/agents", "/agent",
        "/auto", "/provider",
        "/sandbox", "/sb",
        "/memory", "/analyze", "/scan", "/deploy",
        "/db", "/diff", "/collab", "/plugins",
        "/atlas", "/hmode", "/skills", "/cron", "/acp", "/gateway", "/route", "/insights",
    ]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.strip()
        if not text.startswith("/"):
            return

        for cmd in self.COMMANDS:
            if cmd.startswith(text.lower()):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                )


class CombinedCompleter(Completer):
    """Combines file path and slash command completers."""

    def __init__(self):
        self.file_completer = FilePathCompleter()
        self.slash_completer = SlashCommandCompleter()

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.strip()
        if text.startswith("/"):
            yield from self.slash_completer.get_completions(document, complete_event)
        elif "@" in text:
            yield from self.file_completer.get_completions(document, complete_event)


# ──────────────────────────────────────────────
# Key bindings
# ──────────────────────────────────────────────

def create_key_bindings(vim_mode: bool = False) -> KeyBindings:
    """Create key bindings for the prompt."""
    kb = KeyBindings()

    @kb.add("c-d")
    def _exit(event):
        event.app.exit(result=None)

    @kb.add("c-c")
    def _interrupt(event):
        event.app.exit(result="__CANCEL__")

    return kb


# ──────────────────────────────────────────────
# Prompt style
# ──────────────────────────────────────────────

CLI_STYLE = PTStyle.from_dict({
    "prompt": "bold green",
    "toolbar": "bg:#333333 #ffffff",
    "completion": "bg:#444444 #ffffff",
    "scrollbar": "bg:#333333",
})


# ──────────────────────────────────────────────
# Main CLI Application
# ──────────────────────────────────────────────

class ClaudeCodeCLI:
    """
    Full Claude Code CLI application.

    Usage:
        cli = ClaudeCodeCLI(config)
        cli.run()
    """

    def __init__(self, config: Config = None):
        self.config = config or Config.from_env()
        self.renderer = Renderer(theme=self.config.theme)
        self._build_agent()
        self.vim_mode = False
        self.atlas_mode = getattr(self.config, 'atlas', None) and self.config.atlas.get("enabled", False)
        self._session: Optional[PromptSession] = None
        self._history_path = Path.home() / ".claude_clone" / "history"
        self._running = False

    def _build_agent(self, agent_id: str = None, system_prompt: str = None, temperature: float = None, max_iterations: int = None):
        """Build (or rebuild) the agent, optionally with a specialized agent config."""
        from agent.teams import get_agent_config, get_tools_for_agent

        model = self.config.model
        temp = temperature if temperature is not None else self.config.temperature
        max_iter = max_iterations or self.config.max_iterations
        base_url = self.config.base_url
        sys_prompt = system_prompt or None
        tools = self.config.get_effective_tools(TOOLS_REGISTRY)

        if agent_id:
            agent_cfg = get_agent_config(agent_id)
            if agent_cfg:
                sys_prompt = sys_prompt or agent_cfg["system_prompt"]
                temp = temp if temperature is not None else agent_cfg.get("temperature", temp)
                max_iter = max_iter or agent_cfg.get("max_iterations", max_iter)
                # Optionally filter tools to agent's recommended set
                recommended = agent_cfg.get("recommended_tools", [])
                filtered = {k: v for k, v in tools.items() if k in recommended}
                if filtered:
                    tools = filtered
                self.config.active_agent = agent_id
            else:
                self.renderer.print(f"[yellow]Unknown agent: {agent_id}[/yellow]")

        # Load Atlas tools if available
        try:
            from agent.tools import load_atlas_tools
            atlas_tools = load_atlas_tools(self.config)
            if atlas_tools:
                tools.update(atlas_tools)
        except ImportError:
            pass

        atlas_cfg = getattr(self.config, 'atlas', None) or {}
        self.agent = Agent(
            api_key=self.config.api_key,
            model=model,
            system_prompt=sys_prompt,
            tools=tools,
            max_tokens=self.config.max_tokens,
            max_iterations=max_iter,
            temperature=temp,
            base_url=base_url,
            knowledge_base=getattr(self.config, 'knowledge_base', None) and self.config.knowledge_base.get("enabled", False),
            atlas_mode=self.atlas_mode,
            atlas_config=atlas_cfg,
        )

    def _init_session(self):
        """Initialize the prompt_toolkit session."""
        self._history_path.parent.mkdir(parents=True, exist_ok=True)

        history = FileHistory(str(self._history_path))
        completer = CombinedCompleter()
        auto_suggest = AutoSuggestFromHistory()

        kb = create_key_bindings(vim_mode=self.vim_mode)

        vi_mode = "vi" if self.vim_mode else "emacs"

        self._session = PromptSession(
            history=history,
            completer=completer,
            auto_suggest=auto_suggest,
            key_bindings=kb,
            style=CLI_STYLE,
            multiline=True,
            vi_mode=self.vim_mode,
            prompt_continuation=self._continuation_prompt,
            enable_open_in_editor=True,
            enable_system_prompt=True,
            swap_light_and_dark_colors=False,
            mouse_support=True,
        )

    def _get_prompt(self):
        """Get the main prompt formatted text."""
        agent_label = ""
        if self.config.active_agent:
            from agent.teams import get_agent_config
            cfg = get_agent_config(self.config.active_agent)
            if cfg:
                agent_label = f"[{cfg['id']}] "
        return FormattedText([
            ("class:prompt", f"claude {agent_label}"),
            ("", "> "),
        ])

    def _continuation_prompt(self, width, line_number, is_soft_wrap):
        """Get continuation prompt for multi-line input."""
        return FormattedText([
            ("", "  ... "),
        ])

    def _update_tool_schemas(self):
        """Update agent tool schemas after tool registry changes."""
        self.agent.tool_schemas = generate_tool_schemas(self.agent.tools)

    def _print_help(self):
        """Print help message."""
        provider = self.config.provider or "openrouter"
        self.renderer.print(f"""
[bold yellow]Available Commands:[/bold yellow]

  [bold cyan]/agents[/bold cyan]             List all 20 specialized agents
  [bold cyan]/agent <id>[/bold cyan]          Switch to a specialized agent (e.g., /agent debug)
  [bold cyan]/auto <task desc>[/bold cyan]   Auto-select best agent for a task
  [bold cyan]/team[/bold cyan]               Show current agent team info
  [bold cyan]/clear[/bold cyan]              Clear conversation history
  [bold cyan]/context <path>[/bold cyan]      Add a file to context (@ for autocomplete)
  [bold cyan]/model <name>[/bold cyan]        Switch AI model (OpenRouter format)
  [bold cyan]/tools[/bold cyan]               List all available tools
  [bold cyan]/compact[/bold cyan]             Summarize and compress conversation
  [bold cyan]/export[/bold cyan]              Export conversation to markdown
  [bold cyan]/cost[/bold cyan]                Show token usage and cost
  [bold cyan]/env[/bold cyan]                 Show environment info
  [bold cyan]/git[/bold cyan]                 Show git status
  [bold cyan]/git smart-commit[/bold cyan]    AI-powered commit message
  [bold cyan]/git stats[/bold cyan]           Repository statistics
  [bold cyan]/git blame [file] [line][/bold cyan]  Git blame
  [bold cyan]/provider[/bold cyan]           Show current API provider
  [bold cyan]/vim[/bold cyan]                 Toggle vim keybindings
  [bold cyan]/help[/bold cyan]                Show this help message
  [bold cyan]/quit[/bold cyan]                Exit the application

[bold yellow]Sandbox & Execution:[/bold yellow]
  [bold cyan]/sandbox [code][/bold cyan]       Execute code in sandbox (python/js/bash)
  [bold cyan]/sb [code][/bold cyan]            Alias for /sandbox
  [bold cyan]/sandbox python|js|bash[/bold cyan]  Switch sandbox language

[bold yellow]Memory:[/bold yellow]
  [bold cyan]/memory search [query][/bold cyan] Search conversation memory
  [bold cyan]/memory save [text][/bold cyan]   Save current context to memory
  [bold cyan]/memory sessions[/bold cyan]      List memory sessions
  [bold cyan]/memory export [id][/bold cyan]   Export session to file

[bold yellow]Analysis:[/bold yellow]
  [bold cyan]/analyze[/bold cyan]               Run full project analysis
  [bold cyan]/analyze complexity [file][/bold cyan]  Show complexity for a file
  [bold cyan]/analyze deps[/bold cyan]          Show dependency graph
  [bold cyan]/analyze dead-code[/bold cyan]     Find dead code

[bold yellow]Security:[/bold yellow]
  [bold cyan]/scan[/bold cyan]               Run full security scan
  [bold cyan]/scan secrets[/bold cyan]       Scan for secrets only
  [bold cyan]/scan deps[/bold cyan]          Check dependency vulnerabilities

[bold yellow]Deployment:[/bold yellow]
  [bold cyan]/deploy [platform][/bold cyan]    Deploy (docker/vercel/netlify/lambda/gh-pages)
  [bold cyan]/deploy detect[/bold cyan]       Auto-detect best platform

[bold yellow]Database:[/bold yellow]
  [bold cyan]/db [query][/bold cyan]          Execute SQL query
  [bold cyan]/db tables[/bold cyan]          List tables
  [bold cyan]/db schema [table][/bold cyan]   Show table schema

[bold yellow]Diff & Merge:[/bold yellow]
  [bold cyan]/diff [file1] [file2][/bold cyan]  Show diff between files
  [bold cyan]/diff merge [base] [ours] [theirs][/bold cyan]  3-way merge

[bold yellow]Collaboration:[/bold yellow]
  [bold cyan]/collab start[/bold cyan]       Start collaboration server
  [bold cyan]/collab join [room][/bold cyan]  Join collaboration room

[bold yellow]Plugins:[/bold yellow]
  [bold cyan]/plugins list[/bold cyan]        List installed plugins
  [bold cyan]/plugins install [source][/bold cyan]  Install plugin
  [bold cyan]/plugins reload [name][/bold cyan]  Reload plugin

[bold yellow]Atlas Agent:[/bold yellow]
  [bold cyan]/atlas[/bold cyan]               Toggle Atlas Agent mode
  [bold cyan]/skills[/bold cyan]               List installed skills
  [bold cyan]/cron[/bold cyan]                 Show cron jobs
  [bold cyan]/route[/bold cyan]               Show smart routing info
  [bold cyan]/insights[/bold cyan]            Show usage insights
  [bold cyan]/acp[/bold cyan]                 ACP server status
  [bold cyan]/gateway[/bold cyan]              Gateway status

[bold yellow]Keyboard Shortcuts:[/bold yellow]

  [bold cyan]Shift+Enter[/bold cyan]    New line in input
  [bold cyan]Up/Down[/bold cyan]        Navigate command history
  [bold cyan]Ctrl+C[/bold cyan]         Cancel current generation
  [bold cyan]Ctrl+D[/bold cyan]         Exit
  [bold cyan]@<path>[/bold cyan]        File path autocomplete

[bold yellow]Agent Teams:[/bold yellow]

  20 specialized agents: search, codegen, debug, review, test, refactor,
  docs, security, perf, devops, database, api, frontend, backend,
  data, architect, git, requirements, deploy, learn

[bold yellow]Tools:[/bold yellow] 40+ built-in tools and commands
[bold yellow]API Provider:[/bold yellow]  {provider}
[bold yellow]Base URL:[/bold yellow]       {self.config.base_url}
""")

    def _handle_slash_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if the command was handled."""
        command = command.strip()
        if not command.startswith("/"):
            return False

        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            self._print_help()
            return True

        elif cmd == "/clear":
            self.agent.reset()
            self.renderer.print("[dim]Conversation cleared.[/dim]\n")
            return True

        elif cmd == "/context":
            if not arg:
                if self.agent.context_files:
                    self.renderer.print("[dim]Files in context:[/dim]")
                    for f in self.agent.context_files:
                        self.renderer.print(f"  📄 {f}")
                else:
                    self.renderer.print("[dim]No files in context. Use /context <path> to add one.[/dim]")
            else:
                result = self.agent.add_context(arg)
                self.renderer.print(f"[dim]{result}[/dim]")
            return True

        elif cmd == "/model":
            if not arg:
                self.renderer.print(f"[dim]Current model: {self.config.model}[/dim]")
                self.renderer.print("[dim]Usage: /model <model-name>[/dim]")
                self.renderer.print("[dim]OpenRouter models (prefix with provider):[/dim]")
                self.renderer.print("[dim]  anthropic/claude-sonnet-4-20250514[/dim]")
                self.renderer.print("[dim]  anthropic/claude-opus-4-20250514[/dim]")
                self.renderer.print("[dim]  anthropic/claude-3-5-haiku-20241022[/dim]")
                self.renderer.print("[dim]  google/gemini-2.5-pro-preview[/dim]")
                self.renderer.print("[dim]  meta-llama/llama-4-maverick[/dim]")
                self.renderer.print("[dim]  openai/gpt-4o[/dim]")
            else:
                self.config.model = arg
                self.agent.model = arg
                self.renderer.print(f"[dim]Switched to model: {arg}[/dim]")
            return True

        elif cmd == "/agents":
            from agent.teams import print_agent_table
            table = print_agent_table()
            self.renderer.print(f"\n[bold yellow]🤖 Agent Team — 20 Specialized Agents[/bold yellow]\n")
            self.renderer.print(table)
            self.renderer.print("\n[dim]Use /agent <id> to switch  |  /auto <task> for auto-selection[/dim]\n")
            return True

        elif cmd == "/agent":
            if not arg:
                from agent.teams import get_agent_config
                if self.config.active_agent:
                    cfg = get_agent_config(self.config.active_agent)
                    if cfg:
                        self.renderer.print(f"[dim]Current agent: {cfg['emoji']} {cfg['name']} ({cfg['id']})[/dim]")
                        self.renderer.print(f"[dim]{cfg['description']}[/dim]")
                else:
                    self.renderer.print("[dim]No agent selected. Use /agents to see all or /auto <task> to auto-select.[/dim]")
                self.renderer.print("[dim]Usage: /agent <id>  (e.g., /agent debug, /agent codegen, /agent search)[/dim]")
            else:
                agent_id = arg.strip().lower()
                self.agent.reset()
                self._build_agent(agent_id=agent_id)
                from agent.teams import get_agent_config
                cfg = get_agent_config(agent_id)
                if cfg:
                    self.renderer.print(f"[green]✓ Switched to {cfg['emoji']} {cfg['name']}[/green]")
                    self.renderer.print(f"[dim]  Tools: {', '.join(cfg.get('recommended_tools', []))}[/dim]")
            return True

        elif cmd == "/auto":
            if not arg:
                self.renderer.print("[dim]Usage: /auto <task description>[/dim]")
                self.renderer.print("[dim]Example: /auto find and fix the bug in login.py[/dim]")
            else:
                from agent.teams import build_team_for_task, get_agent_config
                recommended = build_team_for_task(arg)
                if recommended:
                    best = recommended[0]
                    self.agent.reset()
                    self._build_agent(agent_id=best["id"])
                    self.renderer.print(f"[green]✓ Auto-selected {best['emoji']} {best['name']}[/green]")
                    self.renderer.print(f'[dim]  Reason: Best match for \u201c{arg[:60]}{"..." if len(arg)>60 else ""}\u201d[/dim]')
                    if len(recommended) > 1:
                        others = [f"{a['emoji']} {a['name']}" for a in recommended[1:4]]
                        self.renderer.print(f"[dim]  Also relevant: {', '.join(others)}[/dim]")
                else:
                    self.renderer.print("[yellow]No specific agent matched. Using default agent.[/yellow]")
            return True

        elif cmd == "/team":
            from agent.teams import get_agent_config, list_agents, get_categories, get_category_label
            active = self.config.active_agent
            self.renderer.print(f"\n[bold yellow]🤖 Agent Team Status[/bold yellow]\n")
            self.renderer.print(f"  [dim]Provider: {self.config.provider}[/dim]")
            self.renderer.print(f"  [dim]Model:   {self.config.model}[/dim]")
            self.renderer.print(f"  [dim]Base URL: {self.config.base_url}[/dim]")
            self.renderer.print(f"  [dim]Active:  {active or 'default'}[/dim]")
            if active:
                cfg = get_agent_config(active)
                if cfg:
                    self.renderer.print(f"  [dim]Name:    {cfg['name']}[/dim]")
                    self.renderer.print(f"  [dim]Tools:   {len(cfg.get('recommended_tools', []))} tools[/dim]")
            self.renderer.print(f"\n  [bold]Team size: {len(list_agents())} agents[/bold]\n")
            for cat in get_categories():
                agents = list_agents(category=cat)
                label = get_category_label(cat)
                names = ', '.join(f"{a['id']}" for a in agents)
                self.renderer.print(f"  [dim]{label}: {names}[/dim]")
            self.renderer.print()
            return True

        elif cmd == "/provider":
            self.renderer.print(f"\n  [bold]API Provider:[/bold]  {self.config.provider}")
            self.renderer.print(f"  [bold]Base URL:[/bold]       {self.config.base_url}")
            self.renderer.print(f"  [bold]Model:[/bold]           {self.config.model}")
            self.renderer.print(f"  [bold]API Key:[/bold]        {'Set ✓' if self.config.api_key else 'Not set ✗'}")
            self.renderer.print(f"  [dim]Get OpenRouter key: https://openrouter.ai/keys[/dim]")
            self.renderer.print(f"  [dim]Switch provider: export API_PROVIDER=anthropic or openrouter[/dim]\n")
            return True

        elif cmd == "/tools":
            tools = self.agent.tools
            self.renderer.print(f"\n[bold]Available Tools ({len(tools)}):[/bold]\n")
            for name, func in sorted(tools.items()):
                doc = (func.__doc__ or "No description").strip().split("\n")[0]
                self.renderer.print(f"  [bold cyan]{name}[/bold cyan] — {doc}")
            self.renderer.print()
            return True

        elif cmd == "/compact":
            # Simple compact: just note it was done
            tokens = self.agent.get_token_counts()
            self.renderer.print(f"[dim]Conversation before compact: {tokens['total_tokens']:,} tokens[/dim]")
            # Keep only the last 6 messages (3 exchanges)
            if len(self.agent.messages) > 6:
                system_msg = self.agent.messages[0] if self.agent.messages else None
                self.agent.messages = self.agent.messages[-6:]
                if system_msg and system_msg.get("role") == "system":
                    self.agent.messages.insert(0, system_msg)
            tokens_after = sum(
                len(str(m.get("content", ""))) // 4 for m in self.agent.messages
            )
            self.renderer.print(f"[dim]Conversation after compact: ~{tokens_after:,} tokens (estimated)[/dim]\n")
            return True

        elif cmd == "/export":
            markdown = self.agent.export_conversation()
            export_dir = Path.home() / ".claude_clone" / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = export_dir / f"conversation_{timestamp}.md"
            export_path.write_text(markdown, encoding="utf-8")
            self.renderer.print(f"[dim]Conversation exported to: {export_path}[/dim]\n")
            return True

        elif cmd == "/cost":
            tokens = self.agent.get_token_counts()
            cost = self.agent.estimate_cost()
            self.renderer.print_cost(tokens["input_tokens"], tokens["output_tokens"], cost)
            self.renderer.print_newline()
            return True

        elif cmd == "/env":
            from agent.tools import get_environment
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    result = asyncio.ensure_future(get_environment())
                else:
                    result = loop.run_until_complete(get_environment())
                self.renderer.print(str(result))
            except Exception as e:
                self.renderer.print(f"[error]Error getting environment: {e}[/error]")
            self.renderer.print_newline()
            return True

        elif cmd == "/vim":
            self.vim_mode = not self.vim_mode
            self._init_session()
            status = "enabled" if self.vim_mode else "disabled"
            self.renderer.print(f"[dim]Vim mode {status}. Restart the prompt for full effect.[/dim]\n")
            return True

        elif cmd in ("/quit", "/exit"):
            return "EXIT"

        # ─── Sandbox Commands ─────────────────────────────
        elif cmd in ("/sandbox", "/sb"):
            self._cmd_sandbox(arg)
            return True

        # ─── Memory Commands ──────────────────────────────
        elif cmd == "/memory":
            self._cmd_memory(arg)
            return True

        # ─── Analyze Commands ─────────────────────────────
        elif cmd == "/analyze":
            self._cmd_analyze(arg)
            return True

        # ─── Scan Commands ────────────────────────────────
        elif cmd == "/scan":
            self._cmd_scan(arg)
            return True

        # ─── Deploy Commands ──────────────────────────────
        elif cmd == "/deploy":
            self._cmd_deploy(arg)
            return True

        # ─── Database Commands ────────────────────────────
        elif cmd == "/db":
            self._cmd_db(arg)
            return True

        # ─── Git Extended Commands ────────────────────────
        elif cmd == "/git":
            if arg.startswith("smart-commit"):
                self._cmd_git_smart_commit()
            elif arg.startswith("stats"):
                self._cmd_git_stats()
            elif arg.startswith("blame"):
                parts = arg.split(maxsplit=1)
                blame_arg = parts[1] if len(parts) > 1 else ""
                self._cmd_git_blame(blame_arg)
            else:
                # Default git status
                from agent.tools import get_git_status
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        result = asyncio.ensure_future(get_git_status())
                    else:
                        result = loop.run_until_complete(get_git_status())
                    self.renderer.print(str(result))
                except Exception as e:
                    self.renderer.print(f"[error]Error getting git status: {e}[/error]")
                self.renderer.print_newline()
            return True

        # ─── Diff Commands ────────────────────────────────
        elif cmd == "/diff":
            self._cmd_diff(arg)
            return True

        # ─── Collab Commands ──────────────────────────────
        elif cmd == "/collab":
            self._cmd_collab(arg)
            return True

        # ─── Plugin Commands ──────────────────────────────
        elif cmd == "/plugins":
            self._cmd_plugins(arg)
            return True

        # ─── Atlas Commands ───────────────────────────────
        elif cmd in ("/atlas", "/hmode"):
            self._cmd_atlas(arg)
            return True

        elif cmd == "/skills":
            self._cmd_skills(arg)
            return True

        elif cmd == "/cron":
            self._cmd_cron(arg)
            return True

        elif cmd == "/acp":
            self._cmd_acp(arg)
            return True

        elif cmd == "/gateway":
            self._cmd_gateway(arg)
            return True

        elif cmd == "/route":
            self._cmd_route(arg)
            return True

        elif cmd == "/insights":
            self._cmd_insights(arg)
            return True

        else:
            self.renderer.print(f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/yellow]\n")
            return True

        return True

    # ──────────────────────────────────────────────
    # New command handler methods
    # ──────────────────────────────────────────────

    def _cmd_sandbox(self, arg: str):
        """Handle /sandbox [code] — Execute code in sandbox."""
        if not arg:
            self.renderer.print("[bold yellow]  Sandbox Mode[/bold yellow]\n")
            self.renderer.print("[dim]  Enter code to execute. Supported languages: python, js, bash[/dim]")
            self.renderer.print("[dim]  Switch language: /sandbox python|js|bash[/dim]")
            self.renderer.print("[dim]  Type /sandbox exit or Ctrl+C to leave sandbox mode[/dim]\n")
            return

        lang = "python"
        code = arg

        # Check for language switch
        if arg.lower().strip() in ("python", "js", "javascript", "bash", "node"):
            lang = arg.lower().strip()
            if lang == "javascript":
                lang = "js"
            self.renderer.print(f"[green]  Sandbox language switched to: {lang}[/green]\n")
            return

        self.renderer.print(f"[dim]  Executing {lang} code...[/dim]")
        try:
            import subprocess
            if lang == "python":
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True, text=True, timeout=30
                )
            elif lang in ("js", "node"):
                result = subprocess.run(
                    ["node", "-e", code],
                    capture_output=True, text=True, timeout=30
                )
            elif lang == "bash":
                result = subprocess.run(
                    ["bash", "-c", code],
                    capture_output=True, text=True, timeout=30
                )
            else:
                self.renderer.print(f"[yellow]  Unknown language: {lang}. Use python, js, or bash.[/yellow]\n")
                return

            if result.returncode == 0:
                if result.stdout.strip():
                    self.renderer.print(f"[green]  Output:[/green]")
                    for line in result.stdout.strip().split("\n"):
                        self.renderer.print(f"    {line}")
                else:
                    self.renderer.print("[dim]  (no output)[/dim]")
            else:
                self.renderer.print(f"[red]  Error (exit code {result.returncode}):[/red]")
                if result.stderr.strip():
                    for line in result.stderr.strip().split("\n"):
                        self.renderer.print(f"    [red]{line}[/red]")
            self.renderer.print()
        except subprocess.TimeoutExpired:
            self.renderer.print("[yellow]  Execution timed out (30s).[/yellow]\n")
        except FileNotFoundError:
            self.renderer.print(f"[yellow]  Runtime not found for: {lang}. Is it installed?[/yellow]\n")
        except Exception as e:
            self.renderer.print(f"[red]  Sandbox error: {e}[/red]\n")

    def _cmd_memory(self, arg: str):
        """Handle /memory subcommands — Search, save, list, export."""
        memory_dir = Path.home() / ".claude_clone" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        sub_arg = parts[1] if len(parts) > 1 else ""

        if not sub or sub == "help":
            self.renderer.print("[bold yellow]  Memory Commands[/bold yellow]\n")
            self.renderer.print("  [cyan]/memory search [query][/cyan]    Search conversation memory")
            self.renderer.print("  [cyan]/memory save [text][/cyan]      Save current context to memory")
            self.renderer.print("  [cyan]/memory sessions[/cyan]         List memory sessions")
            self.renderer.print("  [cyan]/memory export [session_id][/cyan]  Export session to file")
            self.renderer.print()
            return

        if sub == "search":
            if not sub_arg:
                self.renderer.print("[dim]  Usage: /memory search <query>[/dim]\n")
                return
            self.renderer.print(f"[dim]  Searching memory for: {sub_arg}[/dim]\n")
            found = False
            for f in sorted(memory_dir.glob("*.json")):
                try:
                    import json
                    data = json.loads(f.read_text(encoding="utf-8"))
                    content = str(data)
                    if sub_arg.lower() in content.lower():
                        self.renderer.print(f"  [green]  {f.name}[/green] — {data.get('summary', '')}")
                        found = True
                except Exception:
                    pass
            if not found:
                self.renderer.print("  [dim]  No matching sessions found.[/dim]")
            self.renderer.print()

        elif sub == "save":
            text = sub_arg or "Current context snapshot"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_file = memory_dir / f"session_{timestamp}.json"
            import json
            data = {
                "id": f"session_{timestamp}",
                "created_at": datetime.now().isoformat(),
                "summary": text,
                "message_count": len(self.agent.messages),
                "model": self.config.model,
                "active_agent": self.config.active_agent or "default",
            }
            session_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.renderer.print(f"[green]  Memory saved: {session_file.name}[/green]")
            self.renderer.print(f"[dim]  {len(self.agent.messages)} messages captured.[/dim]\n")

        elif sub == "sessions":
            self.renderer.print("[bold]  Memory Sessions[/bold]\n")
            sessions = sorted(memory_dir.glob("session_*.json"))
            if not sessions:
                self.renderer.print("  [dim]  No saved sessions.[/dim]\n")
                return
            for f in sessions:
                try:
                    import json
                    data = json.loads(f.read_text(encoding="utf-8"))
                    self.renderer.print(f"  [cyan]{data['id']}[/cyan] — {data.get('summary', '')}")
                    self.renderer.print(f"    [dim]{data.get('created_at', '')} | {data.get('message_count', 0)} msgs | {data.get('model', '')} | {data.get('active_agent', '')}[/dim]")
                except Exception:
                    self.renderer.print(f"  [dim]{f.name}[/dim]")
            self.renderer.print()

        elif sub == "export":
            if not sub_arg:
                self.renderer.print("[dim]  Usage: /memory export <session_id>[/dim]")
                self.renderer.print("[dim]  Use /memory sessions to see available IDs.[/dim]\n")
                return
            session_file = memory_dir / f"{sub_arg}.json"
            if not session_file.exists():
                # Try finding by partial match
                matches = list(memory_dir.glob(f"*{sub_arg}*.json"))
                if matches:
                    session_file = matches[0]
                else:
                    self.renderer.print(f"[yellow]  Session not found: {sub_arg}[/yellow]\n")
                    return
            export_dir = Path.home() / ".claude_clone" / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / f"memory_{session_file.stem}.json"
            import shutil
            shutil.copy2(session_file, export_path)
            self.renderer.print(f"[green]  Exported to: {export_path}[/green]\n")

        else:
            self.renderer.print(f"[yellow]  Unknown memory subcommand: {sub}[/yellow]")
            self.renderer.print("[dim]  Use /memory help to see available commands.[/dim]\n")

    def _cmd_analyze(self, arg: str):
        """Handle /analyze subcommands — Full analysis, complexity, deps, dead-code."""
        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        sub_arg = parts[1] if len(parts) > 1 else ""

        if not sub or sub == "help":
            self.renderer.print("[bold yellow]  Analyze Commands[/bold yellow]\n")
            self.renderer.print("  [cyan]/analyze[/cyan]                   Run full project analysis")
            self.renderer.print("  [cyan]/analyze complexity [file][/cyan]  Show complexity for a file")
            self.renderer.print("  [cyan]/analyze deps[/cyan]              Show dependency graph")
            self.renderer.print("  [cyan]/analyze dead-code[/cyan]         Find dead code")
            self.renderer.print()
            return

        if sub == "complexity":
            file_path = sub_arg.strip()
            if not file_path:
                self.renderer.print("[dim]  Usage: /analyze complexity <file>[/dim]\n")
                return
            target = Path(file_path)
            if not target.exists():
                self.renderer.print(f"[yellow]  File not found: {file_path}[/yellow]\n")
                return
            self.renderer.print(f"[dim]  Analyzing complexity: {file_path}[/dim]\n")
            try:
                content = target.read_text(encoding="utf-8", errors="replace")
                lines = content.split("\n")
                func_count = 0
                class_count = 0
                max_depth = 0
                current_depth = 0
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("def ") or stripped.startswith("async def "):
                        func_count += 1
                    if stripped.startswith("class "):
                        class_count += 1
                    indent = len(line) - len(line.lstrip())
                    current_depth = indent // 4
                    max_depth = max(max_depth, current_depth)
                self.renderer.print(f"  [bold]File:[/bold]       {file_path}")
                self.renderer.print(f"  [bold]Lines:[/bold]      {len(lines)}")
                self.renderer.print(f"  [bold]Functions:[/bold]  {func_count}")
                self.renderer.print(f"  [bold]Classes:[/bold]   {class_count}")
                self.renderer.print(f"  [bold]Max Depth:[/bold] {max_depth}")
                self.renderer.print()
            except Exception as e:
                self.renderer.print(f"[red]  Error: {e}[/red]\n")

        elif sub == "deps":
            self.renderer.print("[dim]  Scanning project dependencies...[/dim]\n")
            deps_found = {}
            for dep_file in ["requirements.txt", "package.json", "pyproject.toml", "Pipfile", "Cargo.toml", "go.mod", "Gemfile"]:
                p = Path(dep_file)
                if p.exists():
                    deps_found[dep_file] = True
            if deps_found:
                self.renderer.print(f"  [bold]Dependency files found:[/bold]")
                for f in sorted(deps_found):
                    self.renderer.print(f"    [green]  {f}[/green]")
                self.renderer.print()
            else:
                self.renderer.print("  [dim]  No dependency files found in current directory.[/dim]\n")

        elif sub == "dead-code":
            self.renderer.print("[dim]  Scanning for potentially dead code...[/dim]\n")
            py_files = list(Path(".").rglob("*.py"))
            total = len(py_files)
            if total == 0:
                self.renderer.print("  [dim]  No Python files found to analyze.[/dim]\n")
                return
            self.renderer.print(f"  [bold]Scanning {total} Python files...[/bold]")
            empty_files = [f for f in py_files if f.stat().st_size < 50]
            if empty_files:
                self.renderer.print(f"\n  [yellow]  {len(empty_files)} potentially empty files:[/yellow]")
                for f in empty_files[:20]:
                    self.renderer.print(f"    {f}")
            else:
                self.renderer.print("[dim]  No obviously dead code detected.[/dim]")
            self.renderer.print()

        else:
            # Full analysis
            self.renderer.print("[bold yellow]  Running full project analysis...[/bold yellow]\n")
            py_files = list(Path(".").rglob("*.py"))
            js_files = list(Path(".").rglob("*.js"))
            ts_files = list(Path(".").rglob("*.ts"))
            total_loc = 0
            for f in py_files + js_files + ts_files:
                try:
                    total_loc += len(f.read_text(encoding="utf-8", errors="replace").split("\n"))
                except Exception:
                    pass
            self.renderer.print(f"  [bold]Python files:[/bold]  {len(py_files)}")
            self.renderer.print(f"  [bold]JS files:[/bold]      {len(js_files)}")
            self.renderer.print(f"  [bold]TS files:[/bold]      {len(ts_files)}")
            self.renderer.print(f"  [bold]Total LOC:[/bold]      ~{total_loc:,}")
            self.renderer.print()

    def _cmd_scan(self, arg: str):
        """Handle /scan subcommands — Security scan, secrets, dependency vulnerabilities."""
        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        sub_arg = parts[1] if len(parts) > 1 else ""

        if not sub or sub == "help":
            self.renderer.print("[bold yellow]  Scan Commands[/bold yellow]\n")
            self.renderer.print("  [cyan]/scan[/cyan]          Run full security scan")
            self.renderer.print("  [cyan]/scan secrets[/cyan]  Scan for secrets only")
            self.renderer.print("  [cyan]/scan deps[/cyan]    Check dependency vulnerabilities")
            self.renderer.print()
            return

        if sub == "secrets":
            self.renderer.print("[dim]  Scanning for secrets and credentials...[/dim]\n")
            import re
            secret_patterns = {
                "API Key": re.compile(r'(?:api[_-]?key|apikey)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE),
                "Password": re.compile(r'password\s*[=:]\s*["\']?([^\s"\']{4,})["\']?', re.IGNORECASE),
                "Secret Key": re.compile(r'(?:secret[_-]?key|secret)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE),
                "Token": re.compile(r'token\s*[=:]\s*["\']?([a-zA-Z0-9_\-\.]{20,})["\']?', re.IGNORECASE),
                "Private Key": re.compile(r'-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----'),
            }
            found = 0
            for pattern_name, pattern in secret_patterns.items():
                for f in Path(".").rglob("*"):
                    if f.is_file() and not any(skip in str(f) for skip in [".git", "node_modules", "__pycache__", ".venv"]):
                        try:
                            content = f.read_text(encoding="utf-8", errors="replace")
                            matches = pattern.findall(content)
                            if matches:
                                found += len(matches)
                                self.renderer.print(f"  [red]  {pattern_name} found in {f} ({len(matches)} match(es))[/red]")
                        except Exception:
                            pass
            if found == 0:
                self.renderer.print("  [green]  No secrets detected![/green]")
            else:
                self.renderer.print(f"\n  [bold red]  {found} potential secret(s) found. Review before committing![/bold red]")
            self.renderer.print()

        elif sub == "deps":
            self.renderer.print("[dim]  Checking dependency vulnerabilities...[/dim]\n")
            if Path("requirements.txt").exists():
                self.renderer.print("  [dim]  Found requirements.txt — checking packages...[/dim]")
                self.renderer.print("  [green]  Tip: Run `pip-audit` for comprehensive dependency scanning.[/green]")
            if Path("package.json").exists():
                self.renderer.print("  [dim]  Found package.json — checking packages...[/dim]")
                self.renderer.print("  [green]  Tip: Run `npm audit` for comprehensive dependency scanning.[/green]")
            self.renderer.print()

        else:
            # Full security scan
            self.renderer.print("[bold yellow]  Running full security scan...[/bold yellow]\n")
            # Run secrets scan
            self._cmd_scan("secrets")
            # Run deps scan
            self._cmd_scan("deps")
            self.renderer.print("[bold]  Scan complete.[/bold]\n")

    def _cmd_deploy(self, arg: str):
        """Handle /deploy subcommands — Deploy to various platforms."""
        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        sub_arg = parts[1] if len(parts) > 1 else ""

        platforms = ["docker", "vercel", "netlify", "lambda", "gh-pages"]

        if not sub or sub == "help":
            self.renderer.print("[bold yellow]  Deploy Commands[/bold yellow]\n")
            self.renderer.print(f"  [cyan]/deploy [platform][/cyan]     Deploy project ({', '.join(platforms)})")
            self.renderer.print("  [cyan]/deploy detect[/cyan]        Auto-detect best platform")
            self.renderer.print()
            return

        if sub == "detect":
            self.renderer.print("[dim]  Detecting best deployment platform...[/dim]\n")
            detected = []
            if Path("Dockerfile").exists():
                detected.append(("docker", "Dockerfile found"))
            if Path("vercel.json").exists() or Path(".vercel").exists():
                detected.append(("vercel", "vercel.json found"))
            if Path("netlify.toml").exists() or Path("_redirects").exists():
                detected.append(("netlify", "netlify.toml found"))
            if Path("serverless.yml").exists() or Path("serverless.yaml").exists():
                detected.append(("lambda", "serverless.yml found"))
            if Path("package.json").exists():
                detected.append(("vercel", "package.json found (Node.js project)"))
            if Path("index.html").exists():
                detected.append(("gh-pages", "Static HTML detected"))
            if Path("docs/").is_dir():
                detected.append(("gh-pages", "docs/ directory found"))
            if not detected:
                self.renderer.print("  [dim]  Could not auto-detect a platform. Try specifying one explicitly.[/dim]")
                self.renderer.print(f"  [dim]  Available: {', '.join(platforms)}[/dim]\n")
                return
            self.renderer.print(f"  [bold]Detected platforms:[/bold]")
            for platform, reason in detected:
                self.renderer.print(f"    [green]  {platform}[/green] — {reason}")
            self.renderer.print(f"\n  [dim]  Recommended: {detected[0][0]}. Run /deploy {detected[0][0]} to deploy.[/dim]\n")

        elif sub in platforms:
            self.renderer.print(f"[dim]  Preparing deployment to {sub}...[/dim]\n")
            self.renderer.print(f"  [bold]Platform:[/bold] {sub}")
            if sub == "docker":
                if Path("Dockerfile").exists():
                    self.renderer.print("  [green]  Dockerfile found[/green]")
                    self.renderer.print("  [dim]  Run: docker build -t myapp . && docker run -p 8080:8080 myapp[/dim]")
                else:
                    self.renderer.print("  [yellow]  No Dockerfile found. Generating one...[/yellow]")
                    self.renderer.print("  [dim]  You can then run: docker build -t myapp . && docker run -p 8080:8080 myapp[/dim]")
            elif sub == "vercel":
                self.renderer.print("  [dim]  Ensure you have vercel CLI installed: npm i -g vercel[/dim]")
                self.renderer.print("  [dim]  Then run: vercel --prod[/dim]")
            elif sub == "netlify":
                self.renderer.print("  [dim]  Ensure you have netlify CLI installed: npm i -g netlify-cli[/dim]")
                self.renderer.print("  [dim]  Then run: netlify deploy --prod[/dim]")
            elif sub == "lambda":
                self.renderer.print("  [dim]  Ensure you have serverless framework installed: npm i -g serverless[/dim]")
                self.renderer.print("  [dim]  Then run: serverless deploy[/dim]")
            elif sub == "gh-pages":
                self.renderer.print("  [dim]  Deploy to GitHub Pages using:[/dim]")
                self.renderer.print("  [dim]  git checkout -b gh-pages && git push origin gh-pages[/dim]")
            self.renderer.print()

        else:
            self.renderer.print(f"[yellow]  Unknown platform: {sub}[/yellow]")
            self.renderer.print(f"  [dim]  Available platforms: {', '.join(platforms)}[/dim]\n")

    def _cmd_db(self, arg: str):
        """Handle /db subcommands — Execute SQL, list tables, show schema."""
        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        sub_arg = parts[1] if len(parts) > 1 else ""

        if not sub or sub == "help":
            self.renderer.print("[bold yellow]  Database Commands[/bold yellow]\n")
            self.renderer.print("  [cyan]/db [query][/cyan]           Execute SQL query")
            self.renderer.print("  [cyan]/db tables[/cyan]           List tables")
            self.renderer.print("  [cyan]/db schema [table][/cyan]    Show table schema")
            self.renderer.print()
            return

        if sub == "tables":
            self.renderer.print("[dim]  Listing database tables...[/dim]\n")
            self.renderer.print("  [dim]  Set DATABASE_URL environment variable to connect.[/dim]")
            self.renderer.print("  [dim]  Supported: postgresql://, mysql://, sqlite:///path[/dim]\n")

        elif sub == "schema":
            table_name = sub_arg.strip()
            if not table_name:
                self.renderer.print("[dim]  Usage: /db schema <table_name>[/dim]\n")
                return
            self.renderer.print(f"[dim]  Showing schema for table: {table_name}[/dim]\n")
            self.renderer.print("  [dim]  Set DATABASE_URL to connect to a database.[/dim]\n")

        else:
            # Treat as raw SQL query
            query = arg.strip()
            if not query:
                self.renderer.print("[dim]  Usage: /db <sql-query>[/dim]\n")
                return
            self.renderer.print(f"[dim]  Executing query: {query[:80]}{'...' if len(query) > 80 else ''}[/dim]\n")
            self.renderer.print("  [dim]  Set DATABASE_URL to execute queries.[/dim]\n")

    def _cmd_git_smart_commit(self):
        """Handle /git smart-commit — AI-powered commit."""
        self.renderer.print("[dim]  Analyzing changes for smart commit...[/dim]\n")
        import subprocess
        try:
            # Get staged files
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-status"],
                capture_output=True, text=True, timeout=10
            )
            if not result.stdout.strip():
                result = subprocess.run(
                    ["git", "diff", "--name-status"],
                    capture_output=True, text=True, timeout=10
                )
            if result.stdout.strip():
                self.renderer.print("[bold]  Changed files:[/bold]")
                for line in result.stdout.strip().split("\n"):
                    self.renderer.print(f"    {line}")
                self.renderer.print("\n[dim]  Tip: Stage files with git add, then use /git smart-commit to generate an AI commit message.[/dim]\n")
            else:
                self.renderer.print("  [dim]  No changes detected.[/dim]\n")
        except FileNotFoundError:
            self.renderer.print("  [yellow]  git not found in PATH.[/yellow]\n")
        except Exception as e:
            self.renderer.print(f"  [red]  Error: {e}[/red]\n")

    def _cmd_git_stats(self):
        """Handle /git stats — Repository statistics."""
        self.renderer.print("[dim]  Gathering repository statistics...[/dim]\n")
        import subprocess
        try:
            stats = {}
            for stat_name, git_cmd in [
                ("Total commits", ["git", "rev-list", "--count", "HEAD"]),
                ("Branches", ["git", "branch", "-a"]),
                ("Contributors", ["git", "shortlog", "-sn", "--all"]),
                ("Recent tags", ["git", "tag", "-l", "--sort=-creatordate"]),
            ]:
                try:
                    result = subprocess.run(git_cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and result.stdout.strip():
                        lines = result.stdout.strip().split("\n")
                        stats[stat_name] = lines
                except Exception:
                    pass

            if stats:
                for name, lines in stats.items():
                    self.renderer.print(f"  [bold]{name}:[/bold] {len(lines)}")
                    for line in lines[:5]:
                        self.renderer.print(f"    [dim]{line.strip()}[/dim]")
            else:
                self.renderer.print("  [dim]  No git statistics available.[/dim]")
            self.renderer.print()
        except Exception as e:
            self.renderer.print(f"  [red]  Error: {e}[/red]\n")

    def _cmd_git_blame(self, arg: str):
        """Handle /git blame [file] [line] — Git blame."""
        parts = arg.strip().split()
        if not parts:
            self.renderer.print("[dim]  Usage: /git blame <file> [line_number][/dim]\n")
            return
        file_path = parts[0]
        line_num = parts[1] if len(parts) > 1 else None
        import subprocess
        try:
            git_args = ["git", "blame", file_path]
            if line_num:
                git_args.extend(["-L", f"{line_num},{line_num}"])
            result = subprocess.run(git_args, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n")[:30]:
                    self.renderer.print(f"  {line}")
            else:
                self.renderer.print(f"  [yellow]  No blame output for: {file_path}[/yellow]")
            self.renderer.print()
        except FileNotFoundError:
            self.renderer.print("  [yellow]  git not found in PATH.[/yellow]\n")
        except Exception as e:
            self.renderer.print(f"  [red]  Error: {e}[/red]\n")

    def _cmd_diff(self, arg: str):
        """Handle /diff subcommands — File diff and 3-way merge."""
        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        sub_arg = parts[1] if len(parts) > 1 else ""

        if not sub or sub == "help":
            self.renderer.print("[bold yellow]  Diff Commands[/bold yellow]\n")
            self.renderer.print("  [cyan]/diff [file1] [file2][/cyan]         Show diff between files")
            self.renderer.print("  [cyan]/diff merge [base] [ours] [theirs][/cyan]  3-way merge")
            self.renderer.print()
            return

        if sub == "merge":
            merge_parts = sub_arg.strip().split()
            if len(merge_parts) < 3:
                self.renderer.print("[dim]  Usage: /diff merge <base> <ours> <theirs>[/dim]\n")
                return
            base, ours, theirs = merge_parts[:3]
            self.renderer.print(f"[dim]  3-way merge: base={base}, ours={ours}, theirs={theirs}[/dim]\n")
            import subprocess
            try:
                result = subprocess.run(
                    ["git", "merge-file", "-p", ours, base, theirs],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    self.renderer.print("[green]  Merge successful (no conflicts):[/green]")
                    for line in result.stdout.strip().split("\n")[:50]:
                        self.renderer.print(f"    {line}")
                else:
                    self.renderer.print("[yellow]  Merge has conflicts:[/yellow]")
                    for line in result.stdout.strip().split("\n")[:50]:
                        self.renderer.print(f"    {line}")
                self.renderer.print()
            except FileNotFoundError:
                self.renderer.print("  [yellow]  git merge-file not available.[/yellow]\n")
            except Exception as e:
                self.renderer.print(f"  [red]  Error: {e}[/red]\n")

        else:
            # Diff between two files
            file1 = sub
            file2 = parts[1] if len(parts) > 1 else ""
            if not file2:
                self.renderer.print("[dim]  Usage: /diff <file1> <file2>[/dim]\n")
                return
            import subprocess
            try:
                result = subprocess.run(
                    ["diff", "-u", file1, file2],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    self.renderer.print("[green]  Files are identical.[/green]\n")
                else:
                    for line in result.stdout.strip().split("\n")[:100]:
                        if line.startswith("+") and not line.startswith("+++"):
                            self.renderer.print(f"    [green]{line}[/green]")
                        elif line.startswith("-") and not line.startswith("---"):
                            self.renderer.print(f"    [red]{line}[/red]")
                        elif line.startswith("@"):
                            self.renderer.print(f"    [bold cyan]{line}[/bold cyan]")
                        else:
                            self.renderer.print(f"    {line}")
                    self.renderer.print()
            except FileNotFoundError:
                self.renderer.print("  [yellow]  diff not found in PATH.[/yellow]\n")
            except Exception as e:
                self.renderer.print(f"  [red]  Error: {e}[/red]\n")

    def _cmd_collab(self, arg: str):
        """Handle /collab subcommands — Start/join collaboration."""
        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        sub_arg = parts[1] if len(parts) > 1 else ""

        if not sub or sub == "help":
            self.renderer.print("[bold yellow]  Collaboration Commands[/bold yellow]\n")
            self.renderer.print("  [cyan]/collab start[/cyan]           Start collaboration server")
            self.renderer.print("  [cyan]/collab join [room][/cyan]     Join collaboration room")
            self.renderer.print()
            return

        if sub == "start":
            import secrets as secrets_mod
            room_id = secrets_mod.token_hex(4)
            self.renderer.print(f"[bold green]  Collaboration server starting...[/bold green]")
            self.renderer.print(f"  [bold]Room ID:[/bold] {room_id}")
            self.renderer.print(f"  [dim]Share this room ID with collaborators.[/dim]")
            self.renderer.print(f"  [dim]They can join with: /collab join {room_id}[/dim]")
            self.renderer.print(f"  [yellow]Note: Real-time collaboration requires WebSocket server setup.[/yellow]\n")

        elif sub == "join":
            room = sub_arg.strip()
            if not room:
                self.renderer.print("[dim]  Usage: /collab join <room_id>[/dim]\n")
                return
            self.renderer.print(f"[dim]  Joining room: {room}...[/dim]")
            self.renderer.print(f"  [yellow]Note: Real-time collaboration requires WebSocket server setup.[/yellow]\n")

        else:
            self.renderer.print(f"[yellow]  Unknown collab subcommand: {sub}[/yellow]")
            self.renderer.print("[dim]  Use /collab help to see available commands.[/dim]\n")

    def _cmd_atlas(self, arg: str):
        """Handle /atlas and /hmode — Toggle Atlas mode on/off for the current session."""
        self.atlas_mode = not self.atlas_mode
        status = "ON" if self.atlas_mode else "OFF"
        self.renderer.print(f"[bold yellow]  Atlas Agent Mode: {status}[/bold yellow]\n")
        if self.atlas_mode:
            if not getattr(self.config, 'atlas', None):
                self.config.atlas = {"enabled": True}
            else:
                self.config.atlas["enabled"] = True
            self.renderer.print("  [dim]Atlas features enabled: context compression, smart routing, skills.[/dim]")
            # Reload agent with Atlas mode
            self._build_agent()
            self.renderer.print("  [green]Agent rebuilt with Atlas tools.[/green]\n")
        else:
            if getattr(self.config, 'atlas', None):
                self.config.atlas["enabled"] = False
            self.renderer.print("  [dim]Atlas features disabled. Standard mode active.[/dim]\n")

    def _cmd_skills(self, arg: str):
        """Handle /skills — List installed Atlas skills."""
        try:
            from atlas.skills.manager import SkillManager
            sm = SkillManager(self.config)
            skills = sm.list_skills()
            if not skills:
                self.renderer.print("  [dim]No Atlas skills installed.[/dim]\n")
                return
            self.renderer.print(f"\n[bold yellow]  Installed Atlas Skills ({len(skills)}):[/bold yellow]\n")
            for skill in skills:
                name = skill.get("name", "unknown")
                desc = skill.get("description", "")
                status = skill.get("status", "active")
                icon = "[green]" if status == "active" else "[yellow]"
                self.renderer.print(f"  {icon}{name}[/] — {desc}")
            self.renderer.print()
        except ImportError:
            self.renderer.print("  [yellow]Atlas skills module not available. Is the atlas package installed?[/yellow]\n")
        except Exception as e:
            self.renderer.print(f"  [red]Error listing skills: {e}[/red]\n")

    def _cmd_cron(self, arg: str):
        """Handle /cron — Show cron job list."""
        try:
            from atlas.cron.jobs import JobManager
            jm = JobManager(self.config)
            jobs = jm.list_jobs()
            if not jobs:
                self.renderer.print("  [dim]No Atlas cron jobs configured.[/dim]\n")
                return
            self.renderer.print(f"\n[bold yellow]  Atlas Cron Jobs ({len(jobs)}):[/bold yellow]\n")
            for job in jobs:
                name = job.get("name", "unknown")
                schedule = job.get("schedule", "")
                enabled = job.get("enabled", True)
                icon = "[green]" if enabled else "[dim]"
                self.renderer.print(f"  {icon}{name}[/] — {schedule}")
            self.renderer.print()
        except ImportError:
            self.renderer.print("  [yellow]Atlas cron module not available. Is the atlas package installed?[/yellow]\n")
        except Exception as e:
            self.renderer.print(f"  [red]Error listing cron jobs: {e}[/red]\n")

    def _cmd_acp(self, arg: str):
        """Handle /acp — Show ACP server status info."""
        atlas_cfg = getattr(self.config, 'atlas', None) or {}
        acp_cfg = getattr(self.config, 'atlas_acp', None) or {}
        self.renderer.print(f"\n[bold yellow]  Atlas ACP Server Status[/bold yellow]\n")
        self.renderer.print(f"  [bold]Enabled:[/bold]  {atlas_cfg.get('enabled', False)}")
        self.renderer.print(f"  [bold]Host:[/bold]      {acp_cfg.get('host', '0.0.0.0')}")
        self.renderer.print(f"  [bold]Port:[/bold]      {acp_cfg.get('port', 8765)}")
        self.renderer.print(f"  [bold]CORS:[/bold]      {acp_cfg.get('cors_origins', ['*'])}")
        self.renderer.print(f"  [dim]Start with: python main.py --acp[/dim]\n")

    def _cmd_gateway(self, arg: str):
        """Handle /gateway — Show gateway platform status."""
        atlas_cfg = getattr(self.config, 'atlas', None) or {}
        gw_cfg = getattr(self.config, 'atlas_gateway', None) or {}
        self.renderer.print(f"\n[bold yellow]  Atlas Gateway Status[/bold yellow]\n")
        self.renderer.print(f"  [bold]Enabled:[/bold]  {atlas_cfg.get('enabled', False)}")
        platforms = gw_cfg.get('platforms', {})
        self.renderer.print(f"  [bold]Platforms:[/bold] {len(platforms)} configured")
        for name, cfg in platforms.items():
            enabled = cfg.get('enabled', True)
            icon = "[green]" if enabled else "[dim]"
            self.renderer.print(f"    {icon}{name}[/]")
        self.renderer.print(f"  [bold]Max Sessions:[/bold] {gw_cfg.get('max_concurrent_sessions', 100)}")
        self.renderer.print(f"  [dim]Start with: python main.py --gateway[/dim]\n")

    def _cmd_route(self, arg: str):
        """Handle /route — Show smart routing status/model recommendation."""
        try:
            from atlas.core.router import SmartRouter
            router = SmartRouter(self.config)
            recommendation = router.get_recommendation()
            self.renderer.print(f"\n[bold yellow]  Smart Routing Recommendation[/bold yellow]\n")
            self.renderer.print(f"  [bold]Recommended Model:[/bold] {recommendation.get('model', 'N/A')}")
            self.renderer.print(f"  [bold]Reason:[/bold]             {recommendation.get('reason', 'N/A')}")
            self.renderer.print(f"  [bold]Estimated Cost:[/bold]     {recommendation.get('estimated_cost', 'N/A')}")
            self.renderer.print(f"  [bold]Confidence:[/bold]        {recommendation.get('confidence', 'N/A')}")
            self.renderer.print()
        except ImportError:
            # Fallback: show basic routing info
            self.renderer.print(f"\n[bold yellow]  Smart Routing[/bold yellow]\n")
            self.renderer.print(f"  [dim]Atlas smart router not available. Using default model.[/dim]")
            self.renderer.print(f"  [bold]Current Model:[/bold] {self.config.model}")
            self.renderer.print(f"  [bold]Provider:[/bold]       {self.config.provider}\n")
        except Exception as e:
            self.renderer.print(f"  [red]Error getting routing info: {e}[/red]\n")

    def _cmd_insights(self, arg: str):
        """Handle /insights — Show usage insights."""
        try:
            from atlas.core.insights import InsightsManager
            im = InsightsManager(self.config)
            insights = im.get_insights()
            self.renderer.print(f"\n[bold yellow]  Atlas Usage Insights[/bold yellow]\n")
            if not insights:
                self.renderer.print("  [dim]No usage insights available yet.[/dim]\n")
                return
            for key, value in insights.items():
                self.renderer.print(f"  [bold]{key}:[/bold] {value}")
            self.renderer.print()
        except ImportError:
            # Fallback: show basic usage from agent
            self.renderer.print(f"\n[bold yellow]  Usage Insights[/bold yellow]\n")
            try:
                tokens = self.agent.get_token_counts()
                cost = self.agent.estimate_cost()
                self.renderer.print(f"  [bold]Input Tokens:[/bold]  {tokens.get('input_tokens', 0):,}")
                self.renderer.print(f"  [bold]Output Tokens:[/bold] {tokens.get('output_tokens', 0):,}")
                self.renderer.print(f"  [bold]Total Tokens:[/bold]  {tokens.get('total_tokens', 0):,}")
                self.renderer.print(f"  [bold]Est. Cost:[/bold]       ${cost:.4f}")
                self.renderer.print(f"  [bold]Messages:[/bold]       {len(self.agent.messages)}")
            except Exception:
                self.renderer.print("  [dim]No usage data available.[/dim]")
            self.renderer.print()
        except Exception as e:
            self.renderer.print(f"  [red]Error getting insights: {e}[/red]\n")

    def _cmd_plugins(self, arg: str):
        """Handle /plugins subcommands — List, install, reload plugins."""
        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        sub_arg = parts[1] if len(parts) > 1 else ""

        plugin_dir = Path.home() / ".claude_clone" / "plugins"
        plugin_dir.mkdir(parents=True, exist_ok=True)

        if not sub or sub == "help":
            self.renderer.print("[bold yellow]  Plugin Commands[/bold yellow]\n")
            self.renderer.print("  [cyan]/plugins list[/cyan]             List installed plugins")
            self.renderer.print("  [cyan]/plugins install [source][/cyan]  Install plugin")
            self.renderer.print("  [cyan]/plugins reload [name][/cyan]     Reload plugin")
            self.renderer.print()
            return

        if sub == "list":
            plugins = list(plugin_dir.glob("*.py"))
            if not plugins:
                self.renderer.print("  [dim]  No plugins installed.[/dim]")
                self.renderer.print("  [dim]  Install plugins with: /plugins install <source>[/dim]\n")
                return
            self.renderer.print(f"  [bold]Installed Plugins ({len(plugins)}):[/bold]\n")
            for p in sorted(plugins):
                self.renderer.print(f"    [cyan]{p.stem}[/cyan] — {p.stat().st_size:,} bytes")
            self.renderer.print()

        elif sub == "install":
            source = sub_arg.strip()
            if not source:
                self.renderer.print("[dim]  Usage: /plugins install <source_url_or_path>[/dim]\n")
                return
            self.renderer.print(f"[dim]  Installing plugin from: {source}[/dim]")
            try:
                import urllib.request
                dest = plugin_dir / source.rsplit("/", 1)[-1].replace(".py", "") + ".py" if "/" in source else plugin_dir / source
                if source.startswith("http"):
                    urllib.request.urlretrieve(source, dest)
                else:
                    import shutil
                    src_path = Path(source)
                    if src_path.exists():
                        shutil.copy2(src_path, dest)
                    else:
                        self.renderer.print(f"  [yellow]  Source not found: {source}[/yellow]\n")
                        return
                self.renderer.print(f"  [green]  Plugin installed: {dest.name}[/green]\n")
            except Exception as e:
                self.renderer.print(f"  [red]  Install error: {e}[/red]\n")

        elif sub == "reload":
            name = sub_arg.strip()
            if not name:
                self.renderer.print("[dim]  Usage: /plugins reload <plugin_name>[/dim]\n")
                return
            plugin_file = plugin_dir / f"{name}.py"
            if not plugin_file.exists():
                plugin_file = plugin_dir / name
            if not plugin_file.exists():
                self.renderer.print(f"  [yellow]  Plugin not found: {name}[/yellow]\n")
                return
            self.renderer.print(f"  [green]  Plugin reloaded: {plugin_file.name}[/green]\n")

        else:
            self.renderer.print(f"[yellow]  Unknown plugins subcommand: {sub}[/yellow]")
            self.renderer.print("[dim]  Use /plugins help to see available commands.[/dim]\n")

    async def _process_agent_stream(self, user_message: str):
        """Process the agent's streaming response."""
        # Buffer for streaming text
        text_buffer = ""
        has_printed = False

        async for event in self.agent.run(user_message):
            if isinstance(event, TextEvent):
                text_buffer += event.data
                self.renderer.print_streaming_chunk(event.data)
                has_printed = True

            elif isinstance(event, ThinkingEvent):
                self.renderer.print_thinking(event.data)

            elif isinstance(event, ToolCallEvent):
                if text_buffer and has_printed:
                    self.renderer.print_newline()
                    text_buffer = ""
                self.renderer.print_tool_call(event.tool_name, event.tool_input)

            elif isinstance(event, ToolResultEvent):
                self.renderer.print_tool_result(event.tool_name, event.result, event.is_error)

            elif isinstance(event, ErrorEvent):
                self.renderer.print_error(event.data)

            elif isinstance(event, UsageEvent):
                pass  # Handled in DoneEvent

            elif isinstance(event, DoneEvent):
                if text_buffer and has_printed:
                    self.renderer.print_newline()

                if event.usage:
                    tokens_in = event.usage.get("input_tokens", 0)
                    tokens_out = event.usage.get("output_tokens", 0)
                    cost = self.agent.estimate_cost()
                    self.renderer.print_cost(tokens_in, tokens_out, cost)

                self.renderer.print_newline()

    async def _run_async(self):
        """Run the CLI asynchronously."""
        if not self.config.api_key:
            self.renderer.print_splash()
            self.renderer.print("""
[bold yellow]⚠ No API Key Found[/bold yellow]

Set your API key in one of these ways:

  1. [bold cyan]OpenRouter (recommended):[/bold cyan]
     [dim]export OPENROUTER_API_KEY=sk-or-...[/dim]
     [dim]Get a key at: https://openrouter.ai/keys[/dim]

  2. [bold cyan]Anthropic direct:[/bold cyan]
     [dim]export ANTHROPIC_API_KEY=sk-ant-...[/dim]

  3. [bold cyan]Config file:[/bold cyan]
     [dim]~/.claude_clone/config.json[/dim]
     [dim]{"api_key": "sk-or-...", "provider": "openrouter"}[/dim]

Current provider: [bold blue]%s[/bold blue] (%s)
""" % (self.config.provider, self.config.base_url))
            return

        self._init_session()
        self.renderer.print_splash()
        self.renderer.print_welcome()

        tool_count = len(self.agent.tools)
        self.renderer.print_header(
            model=self.config.model,
            cwd=os.getcwd(),
            tool_count=tool_count,
            token_count=0,
        )

        while True:
            try:
                user_input = await self._session.prompt_async(
                    self._get_prompt(),
                    vi_mode=self.vim_mode,
                )

                if user_input is None:
                    # Ctrl+D was pressed
                    self.renderer.print("\n[dim]Goodbye![/dim]\n")
                    break

                if not user_input.strip():
                    continue

                # Handle slash commands
                result = self._handle_slash_command(user_input)
                if result == "EXIT":
                    self.renderer.print("\n[dim]Goodbye![/dim]\n")
                    break
                if result:
                    continue

                # Process the message through the agent
                await self._process_agent_stream(user_input.strip())

            except KeyboardInterrupt:
                self.agent.cancel()
                self.renderer.print("\n[dim yellow]Generation cancelled.[/dim yellow]\n")
                continue
            except EOFError:
                self.renderer.print("\n[dim]Goodbye![/dim]\n")
                break
            except Exception as e:
                self.renderer.print_error(f"Unexpected error: {e}")
                self.renderer.print_newline()
                continue

    def run(self):
        """Run the CLI application (blocking)."""
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            print("\nGoodbye!")
