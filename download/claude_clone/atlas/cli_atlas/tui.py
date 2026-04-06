"""
Interactive Terminal UI (TUI) for Atlas CLI — the main user interface.

Built with prompt_toolkit and rich:
- Multi-line editing with syntax highlighting
- Slash-command autocomplete
- Conversation history with scrollback
- Streaming output with typing indicator
- Interrupt handling (Ctrl+C)
- Tool call visualization in real-time
- Markdown rendering (tables, code blocks, headers)
- Color themes (dark, light, custom)
- Session management (/save, /load, /switch, /delete)
- Copy-to-clipboard support
- Keyboard shortcuts
"""

import asyncio
import json
import os
import re
import sys
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from atlas.cli_atlas.config_manager import ConfigManager, SESSIONS_DIR
from atlas.cli_atlas.skin_engine import SkinEngine, RESET, BOLD, DIM, ANSI_ESCAPE
from atlas.cli_atlas.banner import Banner
from atlas.cli_atlas.callbacks import CallbackManager, StandardCallbacks
from atlas.cli_atlas.commands import CommandRegistry, generate_help_text

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion, WordCompleter
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.styles import Style as PTStyle
    from prompt_toolkit.formatted_text import FormattedText, HTML
    from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.layout import Layout
    from prompt_toolkit.layout.dimension import D
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.progress import Progress
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ──────────────────────────────────────────────
# Custom completers
# ──────────────────────────────────────────────

class SlashCommandCompleter(Completer):
    """Autocompletes slash commands with descriptions."""

    def __init__(self, command_registry: Optional[CommandRegistry] = None):
        self.registry = command_registry or CommandRegistry()

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.strip()
        if not text.startswith("/"):
            return

        for suggestion in self.registry.autocomplete(text):
            cmd = self.registry.get(suggestion)
            if cmd:
                display = f"{cmd.name}  {cmd.description}"
                yield Completion(
                    cmd.name,
                    start_position=-len(text),
                    display=display,
                )


class FilePathCompleter(Completer):
    """Autocompletes file paths when user types @."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        at_index = text.rfind("@")
        if at_index < 0:
            return

        partial = text[at_index + 1:]
        try:
            if "/" in partial:
                dir_part = os.path.expanduser(partial.rsplit("/", 1)[0] or "/")
                prefix = partial.rsplit("/", 1)[1]
            else:
                dir_part = "."
                prefix = partial

            dir_path = Path(dir_part).resolve()
            if not dir_path.is_dir():
                return

            for entry in sorted(dir_path.iterdir()):
                if entry.name.startswith(prefix) and not entry.name.startswith("."):
                    suffix = "/" if entry.is_dir() else ""
                    yield Completion(
                        entry.name + suffix,
                        start_position=-len(partial),
                        display=entry.name + suffix,
                    )
        except (PermissionError, OSError):
            pass


class CombinedCompleter(Completer):
    """Combines slash command and file path completers."""

    def __init__(self):
        self.slash_completer = SlashCommandCompleter()
        self.file_completer = FilePathCompleter()

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.strip()
        if text.startswith("/"):
            yield from self.slash_completer.get_completions(document, complete_event)
        elif "@" in text:
            yield from self.file_completer.get_completions(document, complete_event)


# ──────────────────────────────────────────────
# Markdown renderer
# ──────────────────────────────────────────────

class MarkdownRenderer:
    """Renders markdown text to terminal output."""

    def __init__(self, skin: Optional[SkinEngine] = None):
        self.skin = skin or SkinEngine()
        self.console = Console() if RICH_AVAILABLE else None
        self.enabled = True

    def render(self, text: str) -> str:
        """Render markdown to formatted terminal output."""
        if not self.enabled:
            return text

        if self.console and RICH_AVAILABLE:
            try:
                md = Markdown(text)
                with self.console.capture() as capture:
                    self.console.print(md)
                return capture.get()
            except Exception:
                pass

        # Fallback: simple formatting
        return self._simple_render(text)

    def _simple_render(self, text: str) -> str:
        """Simple markdown rendering without rich."""
        lines = text.split("\n")
        result = []

        for line in lines:
            # Headers
            if line.startswith("### "):
                result.append(f"\033[1m{line[4:]}\033[0m")
            elif line.startswith("## "):
                result.append(f"\033[1m\033[4m{line[3:]}\033[0m")
            elif line.startswith("# "):
                result.append(f"\033[1m\033[4m\033[36m{line[2:]}\033[0m")

            # Code blocks
            elif line.startswith("```"):
                result.append(f"\033[90m{line}\033[0m")

            # Bold
            elif "**" in line:
                line = re.sub(r'\*\*(.+?)\*\*', r'\033[1m\1\033[0m', line)
                result.append(line)

            # Italic
            elif "*" in line:
                line = re.sub(r'\*(.+?)\*', r'\033[3m\1\033[0m', line)
                result.append(line)

            # Inline code
            elif "`" in line:
                line = re.sub(r'`(.+?)`', r'\033[32m\1\033[0m', line)
                result.append(line)

            # Lists
            elif re.match(r'^\s*[-*+]\s', line):
                result.append(f"  {line}")

            # Tables
            elif "|" in line:
                result.append(line)

            else:
                result.append(line)

        return "\n".join(result)

    def render_code_block(self, code: str, language: str = "") -> str:
        """Render a code block with syntax highlighting."""
        if self.console and RICH_AVAILABLE:
            try:
                syntax = Syntax(code, language or "python", theme="monokai",
                               line_numbers=False, word_wrap=True)
                with self.console.capture() as capture:
                    self.console.print(syntax)
                return capture.get()
            except Exception:
                pass

        # Fallback: simple code block
        lines = code.split("\n")
        return "\n".join(f"  \033[90m{l}\033[0m" for l in lines)

    def render_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Render a table."""
        if self.console and RICH_AVAILABLE:
            try:
                table = Table(show_header=True, header_style="bold magenta",
                             border_style="dim", pad_edge=False)
                for h in headers:
                    table.add_column(h)
                for row in rows:
                    table.add_row(*row)
                with self.console.capture() as capture:
                    self.console.print(table)
                return capture.get()
            except Exception:
                pass

        # Fallback: simple table
        lines = []
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        # Header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))

        # Rows
        for row in rows:
            lines.append(" | ".join(c.ljust(w) for c, w in zip(row, col_widths)))

        return "\n".join(lines)


# ──────────────────────────────────────────────
# Output formatter
# ──────────────────────────────────────────────

class OutputFormatter:
    """Formats and colorizes output for the terminal."""

    def __init__(self, skin: Optional[SkinEngine] = None, markdown_enabled: bool = True):
        self.skin = skin or SkinEngine()
        self.markdown = MarkdownRenderer(skin)
        self.markdown_enabled = markdown_enabled
        self.emoji_enabled = True

    def user_message(self, text: str) -> str:
        """Format a user message."""
        theme = self.skin.get_theme()
        colors = theme.get("colors", {})
        user_color = colors.get("bright_blue", "#7aa2f7")
        r, g, b = self.skin._hex_to_rgb(user_color)

        return f"\n{rgb_color(r, g, b)}\033[1m  You:\033[0m {text}\n"

    def assistant_message(self, text: str) -> str:
        """Format an assistant message."""
        if self.markdown_enabled and self._is_markdown(text):
            rendered = self.markdown.render(text)
            theme = self.skin.get_theme()
            colors = theme.get("colors", {})
            asst_color = colors.get("bright_green", "#9ece6a")
            r, g, b = self.skin._hex_to_rgb(asst_color)

            return f"\n{rgb_color(r, g, b)}\033[1m  Atlas:\033[0m\n{rendered}\n"
        else:
            return f"\n  \033[1m\033[32mAtlas:\033[0m {text}\n"

    def tool_call(self, tool_name: str, args: str = "") -> str:
        """Format a tool call display."""
        theme = self.skin.get_theme()
        colors = theme.get("colors", {})
        tool_color = colors.get("yellow", "#e0af68")
        r, g, b = self.skin._hex_to_rgb(tool_color)

        emoji = "\U0001f527" if self.emoji_enabled else ""
        args_display = f" {args}" if args else ""
        return f"  {emoji} {rgb_color(r, g, b)}{tool_name}{args_display}\033[0m"

    def tool_result(self, tool_name: str, result: str, success: bool = True) -> str:
        """Format a tool result display."""
        if success:
            emoji = "\u2705" if self.emoji_enabled else "[OK]"
            color_code = "\033[32m"
        else:
            emoji = "\u274c" if self.emoji_enabled else "[FAIL]"
            color_code = "\033[31m"

        # Truncate long results
        if len(result) > 500:
            result = result[:500] + "\n  ... (truncated)"

        return f"  {emoji} {color_code}{tool_name}\033[0m: {result[:200]}"

    def error(self, message: str) -> str:
        """Format an error message."""
        return f"  \033[31m\u274c Error: {message}\033[0m"

    def warning(self, message: str) -> str:
        """Format a warning message."""
        return f"  \033[33m\u26a0 {message}\033[0m"

    def info(self, message: str) -> str:
        """Format an info message."""
        return f"  \033[36m\u2139 {message}\033[0m"

    def success(self, message: str) -> str:
        """Format a success message."""
        return f"  \033[32m\u2705 {message}\033[0m"

    def typing_indicator(self) -> str:
        """Show a typing indicator."""
        if self.emoji_enabled:
            return f"\r  \033[90m...\033[0m"

    def separator(self) -> str:
        """Print a visual separator."""
        theme = self.skin.get_theme()
        colors = theme.get("colors", {})
        border = colors.get("bright_black", "#414868")
        r, g, b = self.skin._hex_to_rgb(border)

        return f"\n  {rgb_color(r, g, b)}{'─' * 60}\033[0m\n"

    def timestamp(self) -> str:
        """Get a timestamp string."""
        return datetime.now().strftime("%H:%M:%S")

    def _is_markdown(self, text: str) -> bool:
        """Check if text contains markdown formatting."""
        indicators = ["```", "**", "## ", "- ", "| ", "1. ", "> "]
        return any(ind in text for ind in indicators)


def rgb_color(r: int, g: int, b: int) -> str:
    """Generate true-color ANSI code."""
    return f"\033[38;2;{r};{g};{b}m"


# ──────────────────────────────────────────────
# Session Manager
# ──────────────────────────────────────────────

class SessionManager:
    """Manages conversation sessions."""

    def __init__(self):
        self.sessions_dir = SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_id: Optional[str] = None
        self._messages: List[Dict[str, Any]] = []

    def save(self, name: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """Save current session."""
        session_id = name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_file = self.sessions_dir / f"{session_id}.json"

        data = {
            "id": session_id,
            "messages": self._messages,
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat(),
            "message_count": len(self._messages),
        }

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        self.current_session_id = session_id
        return session_id

    def load(self, name: str) -> bool:
        """Load a saved session."""
        session_file = self.sessions_dir / f"{name}.json"
        if not session_file.exists():
            # Try partial match
            matches = list(self.sessions_dir.glob(f"*{name}*.json"))
            if matches:
                session_file = matches[-1]  # Most recent
            else:
                return False

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._messages = data.get("messages", [])
            self.current_session_id = data.get("id", session_file.stem)
            return True
        except Exception:
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions."""
        sessions = []
        for f in sorted(self.sessions_dir.glob("session_*.json"), reverse=True):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                sessions.append({
                    "id": data.get("id", f.stem),
                    "name": f.stem,
                    "message_count": data.get("message_count", 0),
                    "saved_at": data.get("saved_at", ""),
                })
            except Exception:
                sessions.append({
                    "id": f.stem,
                    "name": f.stem,
                    "message_count": 0,
                    "saved_at": "",
                })

        return sessions

    def delete(self, name: str) -> bool:
        """Delete a saved session."""
        session_file = self.sessions_dir / f"{name}.json"
        if not session_file.exists():
            matches = list(self.sessions_dir.glob(f"*{name}*.json"))
            if matches:
                session_file = matches[-1]
            else:
                return False

        session_file.unlink()
        return True

    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to the current session."""
        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        })

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages."""
        return self._messages

    def clear(self):
        """Clear current session messages."""
        self._messages.clear()
        self.current_session_id = None


# ──────────────────────────────────────────────
# Main TUI Class
# ──────────────────────────────────────────────

class AtlasTUI:
    """
    Interactive Terminal UI for Atlas CLI.

    This is the main user interface that provides:
    - Rich, interactive prompt with multi-line editing
    - Streaming output display
    - Slash command system
    - Tool call visualization
    - Session management
    - Theme support

    Usage:
        tui = AtlasTUI(config)
        tui.run()
    """

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        skin: Optional[SkinEngine] = None,
        verbose: bool = False,
    ):
        self.config = config_manager or ConfigManager()
        self.config.load()

        theme_name = self.config.get("theme", "dark")
        skin_name = self.config.get("skin", "dark")
        self.skin = skin or SkinEngine(theme_name)
        self.skin.load_theme(skin_name or theme_name)

        self.banner = Banner(self.skin)
        self.formatter = OutputFormatter(
            self.skin,
            markdown_enabled=self.config.get("markdown_render", True),
        )
        self.formatter.emoji_enabled = self.config.get("emoji_enabled", True)

        self.sessions = SessionManager()
        self.command_registry = CommandRegistry()
        self.callbacks = CallbackManager(
            sound_enabled=self.config.get("sound_enabled", False),
            notification_enabled=self.config.get("notification_enabled", False),
        )
        self._register_callbacks()

        self.verbose = verbose
        self.vim_mode = False
        self._running = False
        self._generating = False
        self._cancelled = False
        self._session: Optional[PromptSession] = None
        self._history_path = Path.home() / ".claude_clone" / "atlas" / "history"

        # Agent (lazy loaded)
        self._agent = None

    def _register_callbacks(self):
        """Register standard callbacks."""
        StandardCallbacks(self.callbacks)

    def _lazy_agent(self):
        """Lazily load the agent."""
        if self._agent is None:
            try:
                from agent.core import Agent
                from agent.tools import TOOLS_REGISTRY, generate_tool_schemas
                from config import Config as BaseConfig

                base_config = BaseConfig.from_env()
                tools = base_config.get_effective_tools(TOOLS_REGISTRY)

                self._agent = Agent(
                    api_key=base_config.api_key,
                    model=base_config.model,
                    system_prompt=None,
                    tools=tools,
                    max_tokens=base_config.max_tokens,
                    max_iterations=base_config.max_iterations,
                    temperature=base_config.temperature,
                    base_url=base_config.base_url,
                )
            except Exception as e:
                self.print_error(f"Failed to initialize agent: {e}")
        return self._agent

    def run(self):
        """Run the main TUI loop."""
        if not PROMPT_TOOLKIT_AVAILABLE:
            self._run_fallback()
            return

        self._running = True
        self._print_banner()

        try:
            while self._running:
                try:
                    user_input = self._prompt_user()
                    if user_input is None:
                        # Ctrl+D was pressed
                        break

                    if user_input == "__CANCEL__":
                        self.print_info("Cancelled.")
                        continue

                    self._process_input(user_input)

                except KeyboardInterrupt:
                    if self._generating:
                        self._cancelled = True
                        self.print_warning("Generation cancelled.")
                    else:
                        print()
                        continue

        except EOFError:
            pass
        finally:
            # Auto-save session
            if self.config.get("auto_save", True):
                try:
                    self.sessions.save("autosave")
                except Exception:
                    pass
            print(f"\n  {self.formatter.success('Goodbye!')}")

    def _run_fallback(self):
        """Fallback input loop without prompt_toolkit."""
        self._print_banner()
        print("  \033[33mNote: prompt_toolkit not installed. Using basic input mode.\033[0m\n")

        while self._running:
            try:
                prompt_str = self.skin.build_prompt()
                user_input = input(f"{prompt_str}").strip()

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "/q", "/quit"):
                    break

                self._process_input(user_input)
            except (EOFError, KeyboardInterrupt):
                break

    def _print_banner(self):
        """Print the startup banner."""
        banner = self.banner.show(
            show_logo=True,
            show_system=True,
            show_tip=True,
            show_quote=False,
        )
        print(banner)

        # Show current profile info
        model = self.config.get("model", "unknown")
        provider = self.config.get("provider", "unknown")
        profile = self.config.get("active_profile", "default")
        print(self.banner.show_profile_info(profile, model, provider))

        # Load auto-save session if available
        if self.config.get("auto_save", True):
            try:
                if self.sessions.load("autosave"):
                    msg_count = len(self.sessions.get_messages())
                    if msg_count > 0:
                        print(f"  \033[90mRestored {msg_count} messages from autosave\033[0m")
            except Exception:
                pass

        print()

    def _init_session(self):
        """Initialize the prompt_toolkit session."""
        self._history_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            history = FileHistory(str(self._history_path))
        except Exception:
            history = InMemoryHistory()

        completer = CombinedCompleter()
        auto_suggest = AutoSuggestFromHistory()
        kb = self._create_keybindings()

        # Build prompt style from theme
        theme = self.skin.get_theme()
        colors = theme.get("colors", {})

        pt_style = PTStyle.from_dict({
            "prompt": f"bold {colors.get('bright_magenta', '#bb9af7')}",
            "continuation": colors.get("bright_black", "#414868"),
            "toolbar": f"bg:{colors.get('black', '#15161e')} {colors.get('bright_white', '#c0caf5')}",
            "completion": f"bg:{colors.get('bright_black', '#414868')} {colors.get('white', '#a9b1d6')}",
            "scrollbar": colors.get("bright_black", "#414868"),
        })

        self._session = PromptSession(
            history=history,
            completer=completer,
            auto_suggest=auto_suggest,
            key_bindings=kb,
            style=pt_style,
            multiline=True,
            vi_mode=self.vim_mode,
            prompt_continuation=lambda w, ln, sw: FormattedText([("class:continuation", "  ... ")]),
            enable_open_in_editor=True,
            mouse_support=True,
            bottom_toolbar=self._bottom_toolbar,
        )

    def _create_keybindings(self) -> KeyBindings:
        """Create custom keybindings."""
        kb = KeyBindings()

        @kb.add("c-d")
        def _exit(event):
            event.app.exit(result=None)

        @kb.add("c-c")
        def _interrupt(event):
            if self._generating:
                self._cancelled = True
            event.app.exit(result="__CANCEL__")

        @kb.add("c-l")
        def _clear(event):
            # Clear screen
            os.system("clear" if os.name != "nt" else "cls")

        @kb.add("c-s")
        def _save(event):
            try:
                name = self.sessions.save()
                event.app.exit(result=f"__SAVED__{name}")
            except Exception:
                event.app.exit(result="__CANCEL__")

        return kb

    def _bottom_toolbar(self):
        """Generate bottom toolbar text."""
        model = self.config.get("model", "unknown")
        if "/" in model:
            model = model.split("/")[-1]

        profile = self.config.get("active_profile", "default")
        msg_count = len(self.sessions.get_messages())

        parts = [
            f" {model}",
            f" [{profile}]",
            f" {msg_count} msgs",
        ]

        if self._generating:
            parts.append(" \u23f3 generating...")

        return FormattedText([("class:toolbar", "".join(parts))])

    def _prompt_user(self) -> Optional[str]:
        """Prompt the user for input."""
        if self._session is None:
            self._init_session()

        # Build prompt
        agent_name = ""
        active_agent = self.config.get("active_agent")
        if active_agent:
            agent_name = f"[{active_agent}] "

        prompt_text = self.skin.build_prompt(agent_name)

        try:
            result = self._session.prompt(
                FormattedText([
                    ("class:prompt", f" {prompt_text} "),
                ]),
                refresh_interval=0.1,
            )
            return result.strip()
        except Exception:
            return None

    def _process_input(self, user_input: str):
        """Process user input - handle commands or send to agent."""
        if not user_input.strip():
            return

        # Handle slash commands
        if user_input.strip().startswith("/"):
            self._handle_slash_command(user_input.strip())
            return

        # Handle @file references
        processed = self._expand_file_references(user_input)

        # Send to agent
        self._send_to_agent(processed)

    def _expand_file_references(self, text: str) -> str:
        """Expand @file references in input."""
        def _replace(match):
            file_path = match.group(1)
            try:
                p = Path(file_path).expanduser().resolve()
                if p.exists() and p.is_file():
                    content = p.read_text(encoding="utf-8", errors="replace")
                    return f"\n--- File: {p} ---\n{content}\n--- End File ---\n"
                return match.group(0)
            except Exception:
                return match.group(0)

        return re.sub(r"@(\S+)", _replace, text)

    def _handle_slash_command(self, command: str):
        """Handle a slash command."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        # Internal commands
        if cmd in ("/quit", "/exit", "/q"):
            self._running = False
            return

        if cmd in ("/help", "/h", "/?"):
            print(generate_help_text(self.skin))
            return

        if cmd in ("/clear", "/cl"):
            self.sessions.clear()
            if self._agent:
                self._agent.reset()
            print(self.formatter.info("Conversation cleared."))
            return

        if cmd in ("/version", "/v"):
            print(f"  Atlas CLI v1.0.0")
            return

        if cmd == "/theme":
            self._cmd_theme(arg)
            return

        if cmd == "/prompt":
            self._cmd_prompt(arg)
            return

        if cmd in ("/model", "/m"):
            self._cmd_model(arg)
            return

        if cmd == "/provider":
            self._cmd_provider(arg)
            return

        if cmd in ("/agent", "/a"):
            self._cmd_agent(arg)
            return

        if cmd == "/agents":
            self._cmd_agents()
            return

        if cmd == "/auto":
            self._cmd_auto(arg)
            return

        if cmd in ("/save",):
            self._cmd_save(arg)
            return

        if cmd == "/load":
            self._cmd_load(arg)
            return

        if cmd in ("/sessions", "/session"):
            self._cmd_sessions()
            return

        if cmd in ("/switch", "/sw"):
            self._cmd_switch(arg)
            return

        if cmd == "/delete":
            self._cmd_delete(arg)
            return

        if cmd in ("/tools",):
            self._cmd_tools(arg)
            return

        if cmd == "/cost":
            self._cmd_cost()
            return

        if cmd in ("/export",):
            self._cmd_export()
            return

        if cmd in ("/compact", "/summarize"):
            self._cmd_compact()
            return

        if cmd in ("/context", "/ctx"):
            self._cmd_context(arg)
            return

        if cmd in ("/config", "/cfg"):
            self._cmd_config(arg)
            return

        if cmd == "/profile":
            self._cmd_profile(arg)
            return

        if cmd == "/setup":
            self._cmd_setup()
            return

        if cmd == "/doctor":
            self._cmd_doctor(arg)
            return

        if cmd == "/env":
            self._cmd_env()
            return

        if cmd == "/vim":
            self.vim_mode = not self.vim_mode
            self._session = None  # Reset session
            status = "enabled" if self.vim_mode else "disabled"
            print(self.formatter.info(f"Vim mode {status}."))
            return

        if cmd in ("/gateway", "/gw"):
            self._cmd_gateway(arg)
            return

        if cmd == "/mcp":
            self._cmd_mcp(arg)
            return

        if cmd == "/cron":
            self._cmd_cron(arg)
            return

        if cmd in ("/skills",):
            self._cmd_skills(arg)
            return

        if cmd == "/hub":
            self._cmd_hub(arg)
            return

        # Delegate to existing CLI commands via agent if possible
        self._delegate_command(cmd, arg)

    # ─── Slash command implementations ────────

    def _cmd_theme(self, arg: str):
        """Handle /theme command."""
        from atlas.cli_atlas.skin_engine import SkinEngine

        if not arg or arg == "list":
            themes = SkinEngine().list_themes()
            current = self.config.get("skin", "dark")
            print(f"\n  \033[1mAvailable Themes:\033[0m\n")
            for t in themes:
                marker = " \u2713" if t["name"] == current else ""
                source = f"({t['source']})"
                print(f"    {t['name']:<15} {t['description']:<35} {source}{marker}")
            print()
        else:
            self.skin.load_theme(arg)
            self.config.set("skin", arg)
            self.config.set("theme", arg)
            self.config.save()
            self.formatter.skin = self.skin
            self.banner.skin = self.skin
            print(self.formatter.success(f"Theme changed to {arg}."))

    def _cmd_prompt(self, arg: str):
        """Handle /prompt command."""
        from atlas.cli_atlas.skin_engine import PROMPT_STYLES

        if not arg:
            print("\n  Available prompt styles:")
            for name, desc in PROMPT_STYLES.items():
                current = " (active)" if name == self.config.get("prompt_style") else ""
                print(f"    {name:<12} {desc}{current}")
            print()
        else:
            self.config.set("prompt_style", arg)
            self.config.save()
            print(self.formatter.success(f"Prompt style set to {arg}."))

    def _cmd_model(self, arg: str):
        """Handle /model command."""
        if not arg:
            current = self.config.get("model", "unknown")
            print(self.formatter.info(f"Current model: {current}"))
            print("  Use /model <name> to switch. Examples:")
            print("    /model anthropic/claude-sonnet-4-20250514")
            print("    /model anthropic/claude-opus-4-20250514")
            print("    /model google/gemini-2.5-pro-preview")
        else:
            self.config.set("model", arg)
            self.config.save()
            if self._agent:
                self._agent.model = arg
            print(self.formatter.success(f"Model switched to {arg}"))

    def _cmd_provider(self, arg: str):
        """Handle /provider command."""
        if not arg:
            provider = self.config.get("provider", "unknown")
            base_url = self.config.get("base_url", "unknown")
            print(f"  Provider: {provider}")
            print(f"  Base URL: {base_url}")
        else:
            self.config.set("provider", arg)
            self.config.save()
            print(self.formatter.success(f"Provider set to {arg}"))

    def _cmd_agent(self, arg: str):
        """Handle /agent command."""
        if not arg:
            active = self.config.get("active_agent")
            if active:
                print(self.formatter.info(f"Current agent: {active}"))
            else:
                print(self.formatter.info("No active agent. Use /agents to list or /auto <task> to auto-select."))
            return

        agent_id = arg.strip().lower()
        self.config.set("active_agent", agent_id)
        self.config.save()
        if self._agent:
            self._agent.reset()
        print(self.formatter.success(f"Switched to agent: {agent_id}"))

    def _cmd_agents(self):
        """Handle /agents command."""
        try:
            from agent.teams import print_agent_table
            table = print_agent_table()
            print(f"\n\033[1m\033[33m  Agent Team\033[0m\n")
            print(table)
            print("\n  Use /agent <id> to switch  |  /auto <task> for auto-selection\n")
        except ImportError:
            print(self.formatter.warning("Agent teams module not available."))

    def _cmd_auto(self, arg: str):
        """Handle /auto command."""
        if not arg:
            print(self.formatter.info("Usage: /auto <task description>"))
            return

        try:
            from agent.teams import build_team_for_task
            recommended = build_team_for_task(arg)
            if recommended:
                best = recommended[0]
                self.config.set("active_agent", best["id"])
                self.config.save()
                print(self.formatter.success(f"Auto-selected: {best['emoji']} {best['name']} ({best['id']})"))
            else:
                print(self.formatter.warning("No specific agent matched. Using default."))
        except ImportError:
            print(self.formatter.warning("Agent teams module not available."))

    def _cmd_save(self, arg: str):
        """Handle /save command."""
        try:
            name = self.sessions.save(arg or None)
            print(self.formatter.success(f"Session saved: {name}"))
        except Exception as e:
            print(self.formatter.error(f"Failed to save session: {e}"))

    def _cmd_load(self, arg: str):
        """Handle /load command."""
        if not arg:
            print(self.formatter.info("Usage: /load <session-name>"))
            return

        if self.sessions.load(arg):
            print(self.formatter.success(f"Session loaded: {self.sessions.current_session_id}"))
        else:
            print(self.formatter.error(f"Session not found: {arg}"))

    def _cmd_sessions(self):
        """Handle /sessions command."""
        sessions = self.sessions.list_sessions()
        if not sessions:
            print(self.formatter.info("No saved sessions."))
            return

        print(f"\n  \033[1mSaved Sessions ({len(sessions)}):\033[0m\n")
        for s in sessions[:20]:
            current = " \u2713" if s["id"] == self.sessions.current_session_id else ""
            saved = s.get("saved_at", "")[:16]
            print(f"    {s['id']:<30} {s['message_count']:>4} msgs  {saved}{current}")
        print()

    def _cmd_switch(self, arg: str):
        """Handle /switch command."""
        self._cmd_load(arg)

    def _cmd_delete(self, arg: str):
        """Handle /delete command."""
        if not arg:
            print(self.formatter.info("Usage: /delete <session-name>"))
            return

        if self.sessions.delete(arg):
            print(self.formatter.success(f"Session deleted: {arg}"))
        else:
            print(self.formatter.error(f"Session not found: {arg}"))

    def _cmd_tools(self, arg: str):
        """Handle /tools command."""
        try:
            from atlas.cli_atlas.tools_config import ToolConfigManager
            mgr = ToolConfigManager(self.config)
            print(f"\n{mgr.format_category_summary()}\n")
            print(f"{mgr.format_tools_table()}\n")
        except Exception as e:
            print(self.formatter.error(str(e)))

    def _cmd_cost(self):
        """Handle /cost command."""
        if self._agent:
            try:
                tokens = self._agent.get_token_counts()
                cost = self._agent.estimate_cost()
                print(f"\n  Token Usage:")
                print(f"    Input:  {tokens.get('input_tokens', 0):,}")
                print(f"    Output: {tokens.get('output_tokens', 0):,}")
                print(f"    Total:  {tokens.get('total_tokens', 0):,}")
                print(f"    Cost:   ${cost:.4f}\n")
            except Exception as e:
                print(self.formatter.error(str(e)))
        else:
            print(self.formatter.info("No active agent to calculate cost."))

    def _cmd_export(self):
        """Handle /export command."""
        try:
            export_dir = Path.home() / ".claude_clone" / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = export_dir / f"conversation_{timestamp}.md"

            lines = []
            for msg in self.sessions.get_messages():
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                ts = msg.get("timestamp", "")
                lines.append(f"### {role.title()} ({ts})\n\n{content}\n")

            export_path.write_text("\n".join(lines), encoding="utf-8")
            print(self.formatter.success(f"Conversation exported to: {export_path}"))
        except Exception as e:
            print(self.formatter.error(f"Export failed: {e}"))

    def _cmd_compact(self):
        """Handle /compact command."""
        msgs = self.sessions.get_messages()
        before = len(msgs)
        if before > 6:
            self.sessions._messages = msgs[-6:]
        after = len(self.sessions.get_messages())
        print(self.formatter.info(f"Compact: {before} -> {after} messages"))

    def _cmd_context(self, arg: str):
        """Handle /context command."""
        if not arg:
            print(self.formatter.info("Usage: /context <file-path>"))
            return
        try:
            p = Path(arg).expanduser().resolve()
            content = p.read_text(encoding="utf-8", errors="replace")
            print(self.formatter.info(f"Added to context: {p} ({len(content)} bytes)"))
            self.sessions.add_message("system", f"Context file: {p}\n{content}")
        except Exception as e:
            print(self.formatter.error(str(e)))

    def _cmd_config(self, arg: str):
        """Handle /config command."""
        if not arg or arg == "list":
            sections = self.config.list_sections()
            print(f"\n  \033[1mConfiguration Sections:\033[0m\n")
            for s in sections:
                val = self.config.get(s, "")
                display = str(val)[:60] if val else "(empty)"
                print(f"    {s:<25} {display}")
            print()
        elif arg.startswith("get "):
            key = arg[4:].strip()
            value = self.config.get(key, "(not set)")
            print(f"  {key} = {value}")
        elif arg.startswith("set "):
            parts = arg[4:].strip().split(maxsplit=1)
            if len(parts) == 2:
                self.config.set(parts[0], parts[1])
                self.config.save()
                print(self.formatter.success(f"Set {parts[0]} = {parts[1]}"))
            else:
                print(self.formatter.info("Usage: /config set <key> <value>"))
        elif arg == "edit":
            editor = self.config.get("editor", "vim")
            import subprocess
            cfg_file = str(self.config.config_file if self.config.config_file.exists() else self.config.config_json_file)
            subprocess.run([editor, cfg_file])
        elif arg == "reset":
            self.config.reset()
            print(self.formatter.success("Configuration reset to defaults."))
        else:
            print(self.formatter.info("Usage: /config [list|get|set|edit|reset]"))

    def _cmd_profile(self, arg: str):
        """Handle /profile command."""
        from atlas.cli_atlas.profiles import ProfileManager
        mgr = ProfileManager()

        if not arg or arg == "list":
            profiles = mgr.list_profiles()
            current = self.config.get("active_profile", "default")
            print(f"\n  \033[1mProfiles:\033[0m\n")
            for p in profiles:
                marker = " \u2713" if p["name"] == current else ""
                desc = p.get("description", "")
                print(f"    {p['name']:<20} {desc}{marker}")
            print()
        elif arg.startswith("switch "):
            name = arg[7:].strip()
            self.config.set("active_profile", name)
            self.config.load(name)
            self.config.save()
            print(self.formatter.success(f"Switched to profile: {name}"))
        elif arg.startswith("create "):
            parts = arg[7:].strip().split(maxsplit=1)
            name = parts[0]
            desc = parts[1] if len(parts) > 1 else ""
            mgr.create(name, description=desc)
            print(self.formatter.success(f"Created profile: {name}"))
        elif arg.startswith("delete "):
            name = arg[7:].strip()
            if mgr.delete(name, force=True):
                print(self.formatter.success(f"Deleted profile: {name}"))
            else:
                print(self.formatter.error(f"Cannot delete profile: {name}"))
        else:
            print(self.formatter.info("Usage: /profile [list|switch|create|delete]"))

    def _cmd_setup(self):
        """Handle /setup command."""
        from atlas.cli_atlas.setup import SetupWizard
        wizard = SetupWizard(self.config)
        wizard.run(skip_api=bool(self.config.get("api_key")))
        self.config.load()

    def _cmd_doctor(self, arg: str):
        """Handle /doctor command."""
        from atlas.cli_atlas.doctor import Doctor
        doc = Doctor(self.config)

        if arg == "fix":
            fixes = doc.fix_issues()
            print(self.formatter.success(f"Applied {len(fixes)} fixes:"))
            for f in fixes:
                print(f"    \u2713 {f}")
        elif arg == "report":
            print(doc.generate_report())
        else:
            results = doc.run_quick()
            print(doc.format_results(results))

    def _cmd_env(self):
        """Handle /env command."""
        import platform
        print(f"  Python:    {sys.version}")
        print(f"  OS:        {platform.system()} {platform.release()}")
        print(f"  Machine:   {platform.machine()}")
        print(f"  CWD:       {os.getcwd()}")
        print(f"  Terminal:  {os.environ.get('TERM', 'unknown')}")
        print(f"  Shell:     {os.environ.get('SHELL', 'unknown')}")
        print(f"  Model:     {self.config.get('model', 'unknown')}")
        print(f"  Provider:  {self.config.get('provider', 'unknown')}")
        print(f"  Theme:     {self.config.get('skin', 'dark')}")

    def _cmd_gateway(self, arg: str):
        """Handle /gateway command."""
        from atlas.cli_atlas.gateway_cmd import GatewayManager
        mgr = GatewayManager(self.config)

        if not arg or arg == "status":
            print(mgr.format_status_dashboard())
        elif arg == "start":
            result = mgr.start()
            if result["success"]:
                print(self.formatter.success(f"Gateway started on {result['host']}:{result['port']}"))
            else:
                print(self.formatter.error(result.get("error", "Failed to start")))
        elif arg == "stop":
            result = mgr.stop()
            if result["success"]:
                print(self.formatter.success("Gateway stopped."))
            else:
                print(self.formatter.error("Gateway is not running."))
        elif arg == "restart":
            result = mgr.restart()
            if result["success"]:
                print(self.formatter.success(f"Gateway restarted on {result['host']}:{result['port']}"))
            else:
                print(self.formatter.error(result.get("error", "Failed to restart")))
        elif arg == "platforms":
            print(mgr.format_platform_table())
        elif arg == "sessions":
            sessions = mgr.list_sessions()
            if sessions:
                for s in sessions:
                    print(f"    {s.get('id', 'unknown'):>20} {s.get('platform', '')}")
            else:
                print(self.formatter.info("No active sessions."))
        else:
            print(self.formatter.info("Usage: /gateway [start|stop|restart|status|platforms|sessions]"))

    def _cmd_mcp(self, arg: str):
        """Handle /mcp command."""
        from atlas.cli_atlas.mcp_config import MCPConfigManager
        mgr = MCPConfigManager(self.config)

        if not arg or arg == "list":
            print(f"\n{mgr.format_servers_table()}\n")
        elif arg.startswith("add "):
            server_id = arg[4:].strip()
            if mgr.install_from_known(server_id):
                print(self.formatter.success(f"Added MCP server: {server_id}"))
            else:
                print(self.formatter.error(f"Unknown server: {server_id}"))
        elif arg.startswith("remove "):
            server_id = arg[7:].strip()
            if mgr.remove_server(server_id):
                print(self.formatter.success(f"Removed MCP server: {server_id}"))
            else:
                print(self.formatter.error(f"Server not found: {server_id}"))
        elif arg == "known":
            servers = mgr.get_known_servers()
            print(f"\n  \033[1mAvailable MCP Servers:\033[0m\n")
            for s in servers:
                installed = " \u2713" if s["installed"] else ""
                print(f"    {s['id']:<15} {s['description']:<40}{installed}")
            print()
        elif arg.startswith("health"):
            results = mgr.health_check_all()
            for r in results:
                status = "\u2713" if r.get("status") == "healthy" else "\u2717"
                print(f"  {status} {r.get('server_id', 'unknown')}: {r.get('status', 'unknown')}")
        else:
            print(self.formatter.info("Usage: /mcp [list|add|remove|known|health]"))

    def _cmd_cron(self, arg: str):
        """Handle /cron command."""
        from atlas.cli_atlas.cron_cmd import CronManager
        mgr = CronManager(self.config)

        if not arg or arg == "list":
            print(f"\n{mgr.format_jobs_table()}\n")
        elif arg.startswith("add "):
            desc = arg[4:].strip()
            job = mgr.create_from_natural_language(desc)
            if job:
                print(self.formatter.success(f"Created job: {job.name} (schedule: {job.schedule})"))
            else:
                print(self.formatter.error("Could not parse schedule. Try: /cron add \"every day at 9am\""))
        elif arg.startswith("pause "):
            job_id = arg[6:].strip()
            if mgr.pause_job(job_id):
                print(self.formatter.success(f"Paused: {job_id}"))
            else:
                print(self.formatter.error(f"Job not found: {job_id}"))
        elif arg.startswith("resume "):
            job_id = arg[7:].strip()
            if mgr.resume_job(job_id):
                print(self.formatter.success(f"Resumed: {job_id}"))
            else:
                print(self.formatter.error(f"Job not found: {job_id}"))
        elif arg.startswith("delete "):
            job_id = arg[7:].strip()
            if mgr.delete_job(job_id):
                print(self.formatter.success(f"Deleted: {job_id}"))
            else:
                print(self.formatter.error(f"Job not found: {job_id}"))
        else:
            print(self.formatter.info("Usage: /cron [list|add|pause|resume|delete]"))

    def _cmd_skills(self, arg: str):
        """Handle /skills command."""
        from atlas.cli_atlas.skills_config import SkillConfigManager
        mgr = SkillConfigManager(self.config)

        if not arg or arg == "list":
            print(f"\n{mgr.format_skills_table()}\n")
        elif arg.startswith("enable "):
            skill_id = arg[7:].strip()
            mgr.enable_skill(skill_id)
            print(self.formatter.success(f"Enabled: {skill_id}"))
        elif arg.startswith("disable "):
            skill_id = arg[8:].strip()
            mgr.disable_skill(skill_id)
            print(self.formatter.success(f"Disabled: {skill_id}"))
        else:
            print(self.formatter.info("Usage: /skills [list|enable|disable]"))

    def _cmd_hub(self, arg: str):
        """Handle /hub command."""
        from atlas.cli_atlas.skills_hub import SkillsHub
        hub = SkillsHub(self.config)

        if not arg or arg == "browse":
            print(f"\n{hub.format_marketplace_table()}\n")
        elif arg.startswith("search "):
            query = arg[7:].strip()
            results = hub.search(query)
            print(f"\n  \033[1mSearch results for '{query}':\033[0m\n")
            for skill in results:
                print(hub.format_skill_card(skill))
                print()
        elif arg.startswith("install "):
            skill_id = arg[8:].strip()
            result = hub.install(skill_id)
            if result["success"]:
                print(self.formatter.success(f"Installed: {skill_id}"))
            else:
                print(self.formatter.error(result.get("error", "Failed to install")))
        elif arg.startswith("uninstall "):
            skill_id = arg[10:].strip()
            if hub.uninstall(skill_id):
                print(self.formatter.success(f"Uninstalled: {skill_id}"))
            else:
                print(self.formatter.error(f"Not installed: {skill_id}"))
        elif arg == "categories":
            categories = hub.get_categories()
            for cat in categories:
                print(f"  {cat['emoji']} {cat['name']:<20} {cat['description']} ({cat['skill_count']} skills)")
        else:
            print(self.formatter.info("Usage: /hub [browse|search|install|uninstall|categories]"))

    def _delegate_command(self, cmd: str, arg: str):
        """Delegate unknown commands to the existing CLI handler."""
        # Some commands are handled by the existing ClaudeCodeCLI
        delegate_commands = {
            "/sandbox", "/sb", "/memory", "/mem", "/analyze",
            "/scan", "/deploy", "/db", "/git", "/diff",
            "/collab", "/plugins", "/undo", "/team",
        }

        if cmd in delegate_commands:
            print(self.formatter.info(f"Command '{cmd}' is handled by the base CLI."))
            return

        print(self.formatter.warning(f"Unknown command: {cmd}. Type /help for available commands."))

    # ─── Agent communication ──────────────────

    def _send_to_agent(self, message: str):
        """Send a message to the agent and display the response."""
        agent = self._lazy_agent()
        if agent is None:
            print(self.formatter.error("Agent not available. Check your configuration with /doctor."))
            return

        # Show user message
        print(self.formatter.user_message(message))
        self.sessions.add_message("user", message)

        # Emit pre-message callback
        self.callbacks.emit("pre_message", {"content": message})

        # Send to agent and stream response
        self._generating = True
        self._cancelled = False

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, use ensure_future
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._run_agent(message))
                    future.result(timeout=120)
            else:
                loop.run_until_complete(self._run_agent(message))

        except Exception as e:
            if not self._cancelled:
                print(self.formatter.error(str(e)))
        finally:
            self._generating = False
            self._cancelled = False

    async def _run_agent(self, message: str):
        """Run the agent with streaming output."""
        agent = self._lazy_agent()
        if agent is None:
            return

        full_response = ""
        tool_calls = []

        try:
            async for event in agent.run_stream(message):
                if self._cancelled:
                    break

                if isinstance(event, str):
                    # Plain text event
                    sys.stdout.write(event)
                    sys.stdout.flush()
                    full_response += event

                elif hasattr(event, "type"):
                    event_type = getattr(event, "type", "")

                    if event_type == "text":
                        text = getattr(event, "text", "")
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        full_response += text

                    elif event_type == "thinking":
                        if self.verbose:
                            text = getattr(event, "thinking", "")
                            sys.stdout.write(f"\033[90m[{text}]\033[0m")
                            sys.stdout.flush()

                    elif event_type == "tool_call":
                        tool_name = getattr(event, "name", "unknown")
                        tool_args = getattr(event, "input", {})
                        tool_calls.append(tool_name)
                        print(f"\n{self.formatter.tool_call(tool_name, json.dumps(tool_args)[:100])}")

                    elif event_type == "tool_result":
                        tool_name = getattr(event, "name", "unknown")
                        result = getattr(event, "result", "")
                        success = getattr(event, "is_error", True) is not True
                        truncated = str(result)[:200]
                        print(self.formatter.tool_result(tool_name, truncated, success))

                    elif event_type == "error":
                        error_msg = getattr(event, "error", "Unknown error")
                        print(self.formatter.error(error_msg))

                    elif event_type == "usage":
                        if self.verbose:
                            input_t = getattr(event, "input_tokens", 0)
                            output_t = getattr(event, "output_tokens", 0)
                            print(f"\n  \033[90mTokens: {input_t:,} in / {output_t:,} out\033[0m")

                    elif event_type == "done":
                        pass

        except asyncio.CancelledError:
            print(self.formatter.warning("Generation cancelled."))
        except Exception as e:
            if not self._cancelled:
                print(self.formatter.error(str(e)))

        # Store response
        if full_response:
            self.sessions.add_message("assistant", full_response)

        # Emit post-message callback
        self.callbacks.emit("post_message", {
            "content": full_response,
            "tool_calls": tool_calls,
            "token_count": agent.get_token_counts() if agent else {},
        })

        print()  # Newline after response

    # ─── Print helpers ────────────────────────

    def print(self, text: str):
        """Print text to terminal."""
        print(text)

    def print_error(self, message: str):
        """Print an error message."""
        print(self.formatter.error(message))

    def print_warning(self, message: str):
        """Print a warning message."""
        print(self.formatter.warning(message))

    def print_info(self, message: str):
        """Print an info message."""
        print(self.formatter.info(message))

    def print_success(self, message: str):
        """Print a success message."""
        print(self.formatter.success(message))
