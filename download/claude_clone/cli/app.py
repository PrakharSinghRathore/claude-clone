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
        "/env", "/git", "/undo",
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
        self.agent = Agent(
            api_key=self.config.api_key,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            max_iterations=self.config.max_iterations,
            temperature=self.config.temperature,
            tools=self.config.get_effective_tools(TOOLS_REGISTRY),
        )
        self.vim_mode = False
        self._session: Optional[PromptSession] = None
        self._history_path = Path.home() / ".claude_clone" / "history"
        self._running = False

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
        return FormattedText([
            ("class:prompt", "claude"),
            ("", " > "),
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
        self.renderer.print("""
[bold yellow]Available Commands:[/bold yellow]

  [bold cyan]/clear[/bold cyan]          Clear conversation history
  [bold cyan]/context <path>[/bold cyan]  Add a file to context (@ for autocomplete)
  [bold cyan]/model <name>[/bold cyan]    Switch AI model
  [bold cyan]/tools[/bold cyan]           List all available tools
  [bold cyan]/compact[/bold cyan]         Summarize and compress conversation
  [bold cyan]/export[/bold cyan]          Export conversation to markdown
  [bold cyan]/cost[/bold cyan]            Show token usage and cost
  [bold cyan]/env[/bold cyan]             Show environment info
  [bold cyan]/git[/bold cyan]             Show git status
  [bold cyan]/vim[/bold cyan]             Toggle vim keybindings
  [bold cyan]/help[/bold cyan]            Show this help message
  [bold cyan]/quit[/bold cyan]            Exit the application

[bold yellow]Keyboard Shortcuts:[/bold yellow]

  [bold cyan]Shift+Enter[/bold cyan]    New line in input
  [bold cyan]Up/Down[/bold cyan]        Navigate command history
  [bold cyan]Ctrl+C[/bold cyan]         Cancel current generation
  [bold cyan]Ctrl+D[/bold cyan]         Exit
  [bold cyan]@<path>[/bold cyan]        File path autocomplete

[bold yellow]Tips:[/bold yellow]

  Type @ followed by a path for file autocomplete
  Use /context to add files for the agent to read
  Use /model to switch between claude-sonnet-4, claude-opus-4, etc.
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
                self.renderer.print("[dim]Available: claude-sonnet-4-20250514, claude-opus-4-20250514, claude-3-5-haiku-20241022[/dim]")
            else:
                self.config.model = arg
                self.agent.model = arg
                self.renderer.print(f"[dim]Switched to model: {arg}[/dim]")
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

        elif cmd == "/git":
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

        elif cmd == "/vim":
            self.vim_mode = not self.vim_mode
            self._init_session()
            status = "enabled" if self.vim_mode else "disabled"
            self.renderer.print(f"[dim]Vim mode {status}. Restart the prompt for full effect.[/dim]\n")
            return True

        elif cmd in ("/quit", "/exit"):
            return "EXIT"

        else:
            self.renderer.print(f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/yellow]\n")
            return True

        return True

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

Set your Anthropic API key in one of these ways:

  1. [bold cyan]Environment variable:[/bold cyan]
     [dim]export ANTHROPIC_API_KEY=sk-ant-...[/dim]

  2. [bold cyan]Config file:[/bold cyan]
     [dim]~/.claude_clone/config.json[/dim]
     [dim]{"api_key": "sk-ant-..."}[/dim]

  3. [bold cyan].env file:[/bold cyan]
     [dim]ANTHROPIC_API_KEY=sk-ant-...[/dim]

Get your API key at: [bold blue]https://console.anthropic.com/[/bold blue]
""")
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
