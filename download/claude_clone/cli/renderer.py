"""
Renderer for the CLI — Markdown rendering with syntax highlighting.
Uses rich for terminal output formatting.
"""

import re
from typing import Optional

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.text import Text
    from rich.theme import Theme
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# Custom dark theme optimized for code assistant output
DARK_THEME = Theme({
    "markdown.heading": "bold cyan",
    "markdown.paragraph": "white",
    "markdown.code": "on black",
    "markdown.code_block": "on black",
    "markdown.link": "blue underline",
    "markdown.list": "yellow",
    "markdown.item": "yellow",
    "info": "dim cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "tool.name": "bold yellow",
    "tool.result": "dim white",
    "tool.input": "dim cyan",
    "spinner": "bold blue",
    "prompt": "bold green",
    "cost": "dim magenta",
})


LIGHT_THEME = Theme({
    "markdown.heading": "bold blue",
    "markdown.paragraph": "black",
    "markdown.code": "on gray93",
    "markdown.code_block": "on gray93",
    "markdown.link": "blue underline",
    "markdown.list": "dark_red",
    "markdown.item": "dark_red",
    "info": "dim blue",
    "warning": "dark_red",
    "error": "bold red",
    "success": "bold green",
    "tool.name": "bold yellow",
    "tool.result": "dim black",
    "tool.input": "dim blue",
    "spinner": "bold blue",
    "prompt": "bold green",
    "cost": "dim magenta",
})


class Renderer:
    """Handles all rendering for the CLI using rich."""

    def __init__(self, theme: str = "dark", width: int = None):
        self.theme_name = theme
        if HAS_RICH:
            rich_theme = DARK_THEME if theme == "dark" else LIGHT_THEME
            self.console = Console(theme=rich_theme, width=width, highlight=False)
        else:
            self.console = None
        self._current_tool_output: list = []
        self._in_tool_block = False

    def print(self, *args, **kwargs):
        """Print to console."""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            text = " ".join(str(a) for a in args)
            print(text)

    def print_markdown(self, text: str):
        """Render markdown text with syntax highlighting."""
        if not text.strip():
            return

        if self.console and HAS_RICH:
            try:
                md = Markdown(text, code_theme="monokai" if self.theme_name == "dark" else "default")
                self.console.print(md)
            except Exception:
                # Fallback: print raw text
                self.console.print(text)
        else:
            print(text)

    def print_code(self, code: str, language: str = "python", title: str = None):
        """Render a code block with syntax highlighting."""
        if self.console and HAS_RICH:
            try:
                syntax = Syntax(
                    code,
                    language,
                    theme="monokai" if self.theme_name == "dark" else "default",
                    line_numbers=True,
                    word_wrap=True,
                )
                if title:
                    panel = Panel(syntax, title=title, border_style="dim")
                    self.console.print(panel)
                else:
                    self.console.print(syntax)
            except Exception:
                # Syntax highlighter may not support this language
                self.console.print(f"```{language}\n{code}\n```")
        else:
            print(f"```{language}\n{code}\n```")

    def print_tool_call(self, tool_name: str, tool_input: dict):
        """Render a tool call with formatted input."""
        if self.console and HAS_RICH:
            text = Text()
            text.append("⟳ ", style="bold yellow")
            text.append(f"Running: ", style="bold")
            text.append(f"{tool_name}(", style="bold yellow")
            args_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                                 for k, v in tool_input.items())
            text.append(args_str, style="cyan")
            text.append(")", style="bold yellow")
            self.console.print(text)
        else:
            args_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                                 for k, v in tool_input.items())
            print(f"⟳ Running: {tool_name}({args_str})")

    def print_tool_result(self, tool_name: str, result: str, is_error: bool = False):
        """Render a tool result in a collapsible panel."""
        if self.console and HAS_RICH:
            style = "red" if is_error else "dim"
            border_style = "red" if is_error else "dim"

            # Truncate very long results
            display_result = result
            if len(display_result) > 3000:
                display_result = display_result[:3000] + f"\n... ({len(result) - 3000} more chars)"

            icon = "✗" if is_error else "✓"
            title = f"{icon} {tool_name}"

            panel = Panel(
                display_result,
                title=title,
                border_style=border_style,
                style=style,
                padding=(0, 1),
            )
            self.console.print(panel)
        else:
            icon = "✗" if is_error else "✓"
            print(f"{icon} {tool_name}: {result[:500]}")

    def print_error(self, message: str):
        """Render an error message."""
        if self.console and HAS_RICH:
            self.console.print(f"❌ Error: {message}", style="bold red")
        else:
            print(f"❌ Error: {message}")

    def print_thinking(self, text: str):
        """Render a thinking/thought block."""
        if not text or not text.strip():
            return
        if self.console and HAS_RICH:
            self.console.print(Text(f"💭 {text[:200]}", style="dim italic"))
        else:
            print(f"💭 {text[:200]}")

    def print_cost(self, input_tokens: int, output_tokens: int, cost: float):
        """Render token usage and cost."""
        if self.console and HAS_RICH:
            text = Text()
            text.append(f"  Tokens: ", style="dim")
            text.append(f"{input_tokens:,}", style="cyan")
            text.append(" in / ", style="dim")
            text.append(f"{output_tokens:,}", style="cyan")
            text.append(" out", style="dim")
            text.append(f"  |  Cost: ${cost:.4f}", style="dim magenta")
            self.console.print(text)
        else:
            print(f"  Tokens: {input_tokens:,} in / {output_tokens:,} out  |  Cost: ${cost:.4f}")

    def print_header(self, model: str, cwd: str, tool_count: int, token_count: int):
        """Render the CLI header bar."""
        if self.console and HAS_RICH:
            table = Table(show_header=False, box=box.ROUNDED, padding=(0, 1),
                         border_style="bright_blue")
            table.add_column(style="bold white")
            table.add_column(style="dim")
            table.add_column(style="dim")
            table.add_column(style="dim")

            table.add_row(
                f"  Claude Code  ",
                f"Model: {model}",
                f"Tools: {tool_count} active",
                f"CWD: {cwd}",
            )

            self.console.print(table)
            self.console.print()
        else:
            print(f"Claude Code — Model: {model} | Tools: {tool_count} | CWD: {cwd}")

    def print_welcome(self):
        """Print welcome message."""
        if self.console and HAS_RICH:
            self.console.print()
            welcome = Text()
            welcome.append("  Claude Code Clone", style="bold bright_cyan")
            welcome.append(" — Python agentic coding assistant\n", style="dim")
            welcome.append("  Type ", style="dim")
            welcome.append("/help", style="bold yellow")
            welcome.append(" for commands, ", style="dim")
            welcome.append("Ctrl+D", style="bold yellow")
            welcome.append(" to exit\n", style="dim")
            self.console.print(welcome)
        else:
            print("\n  Claude Code Clone — Python agentic coding assistant")
            print("  Type /help for commands, Ctrl+D to exit\n")

    def print_splash(self):
        """Print an ASCII art splash screen."""
        if self.console and HAS_RICH:
            splash = Text()
            splash.append("\n", style="")
            splash.append("  ╭─────────────────────────────────────╮\n", style="bright_blue")
            splash.append("  │", style="bright_blue")
            splash.append("         CLAUDE CODE CLONE            ", style="bold bright_cyan")
            splash.append("│\n", style="bright_blue")
            splash.append("  │", style="bright_blue")
            splash.append("    Agentic Coding Assistant v1.0     ", style="dim")
            splash.append("│\n", style="bright_blue")
            splash.append("  ╰─────────────────────────────────────╯\n", style="bright_blue")
            self.console.print(splash)
        else:
            print("""
  ╭─────────────────────────────────────╮
  │         CLAUDE CODE CLONE            │
  │    Agentic Coding Assistant v1.0     │
  ╰─────────────────────────────────────╯
""")

    def print_streaming_chunk(self, chunk: str):
        """Print a streaming text chunk without newline."""
        if self.console and HAS_RICH:
            self.console.print(chunk, end="", highlight=False)
        else:
            print(chunk, end="", flush=True)

    def print_newline(self):
        """Print a blank line."""
        if self.console and HAS_RICH:
            self.console.print()
        else:
            print()

    def clear(self):
        """Clear the console."""
        if self.console and HAS_RICH:
            self.console.clear()

    def rule(self, title: str = ""):
        """Print a horizontal rule."""
        if self.console and HAS_RICH:
            self.console.rule(title)
        else:
            print(f"{'─' * 50} {title}")
