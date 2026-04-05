"""
UI Display Helpers — Spinners, progress bars, emoji rendering, color output.

Provides terminal UI utilities for agent interaction including animated
spinners, progress bars, tool call preview formatting, and color-coded
output. Designed for use in CLI environments.

Usage
-----
    with Spinner("Processing..."):
        await do_work()

    bar = ProgressBar(total=100)
    bar.update(50)
    bar.finish()

    print(colorize("Success!", ColorCode.GREEN))
    print(render_emoji("rocket"))  # 🚀
"""

from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from enum import Enum, auto
from typing import Any, Dict, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Color codes
# ──────────────────────────────────────────────────────────────────────────────

class ColorCode(Enum):
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


def colorize(text: str, color: ColorCode, bold: bool = False) -> str:
    """
    Apply ANSI color codes to text.

    Parameters
    ----------
    text:
        The text to colorize.
    color:
        The color to apply.
    bold:
        Whether to also apply bold formatting.

    Returns
    -------
    str
        Colorized text with reset code appended.
    """
    prefix = color.value
    if bold:
        prefix = ColorCode.BOLD.value + prefix
    return f"{prefix}{text}{ColorCode.RESET.value}"


def strip_ansi(text: str) -> str:
    """
    Remove ANSI escape codes from text.

    Parameters
    ----------
    text:
        Text potentially containing ANSI codes.

    Returns
    -------
    str
        Plain text without ANSI codes.
    """
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)


# ──────────────────────────────────────────────────────────────────────────────
# Emoji helpers
# ──────────────────────────────────────────────────────────────────────────────

_EMOJI_MAP: Dict[str, str] = {
    "rocket": "🚀",
    "check": "✅",
    "cross": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "thinking": "🤔",
    "code": "💻",
    "search": "🔍",
    "tool": "🔧",
    "file": "📄",
    "brain": "🧠",
    "sparkles": "✨",
    "clock": "⏱️",
    "money": "💰",
    "shield": "🛡️",
    "bug": "🐛",
    "fire": "🔥",
    "star": "⭐",
    "success": "🎉",
    "error": "💥",
    "loading": "⏳",
    "memory": "🧠",
    "route": "🔀",
    "key": "🔑",
    "lock": "🔒",
    "globe": "🌐",
    "chart": "📊",
    "zap": "⚡",
    "speech": "💬",
    "robot": "🤖",
}


def render_emoji(name: str, fallback: str = "") -> str:
    """
    Render an emoji by name.

    Parameters
    ----------
    name:
        The emoji name (e.g., ``"rocket"``, ``"check"``).
    fallback:
        Fallback text if emoji is not found.

    Returns
    -------
    str
        The emoji character, or the fallback.
    """
    return _EMOJI_MAP.get(name.lower(), fallback or f"[{name}]")


# ──────────────────────────────────────────────────────────────────────────────
# Spinner
# ──────────────────────────────────────────────────────────────────────────────

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_SPINNER_FRAMES_SIMPLE = ["|", "/", "-", "\\"]
_SPINNER_FRAMES_DOTS = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]


class Spinner:
    """
    Terminal spinner animation.

    Displays an animated spinner with a status message in the terminal.
    Supports both sync (``with``) and async (``async with``) contexts.

    Parameters
    ----------
    message:
        Status message to display alongside the spinner.
    frames:
        Custom spinner frames. If ``None``, uses default frames.
    color:
        Color for the spinner and message.
    """

    def __init__(
        self,
        message: str = "",
        frames: Optional[list] = None,
        color: ColorCode = ColorCode.CYAN,
    ) -> None:
        self._message = message
        self._frames = frames or _SPINNER_FRAMES
        self._color = color
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        """Whether the spinner is currently animating."""
        return self._running

    def _render_frame(self, frame_idx: int) -> str:
        """Render a single spinner frame."""
        frame = self._frames[frame_idx % len(self._frames)]
        return f"\r{self._color.value}{frame} {self._message}{ColorCode.RESET.value}"

    async def _animate(self) -> None:
        """Animation loop."""
        idx = 0
        while self._running:
            sys.stdout.write(self._render_frame(idx))
            sys.stdout.flush()
            idx += 1
            await asyncio.sleep(0.08)

    def start(self) -> None:
        """Start the spinner animation."""
        self._running = True
        self._task = asyncio.create_task(self._animate())

    def stop(self, final_message: Optional[str] = None) -> None:
        """Stop the spinner and optionally display a final message."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

        # Clear the line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

        if final_message:
            sys.stdout.write(final_message + "\n")
            sys.stdout.flush()

    def __enter__(self) -> Spinner:
        """Sync context manager — starts the spinner."""
        # For sync contexts, we need to be in an event loop
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._animate())
            self._running = True
        except RuntimeError:
            self._running = True
        return self

    def __exit__(self, *args: Any) -> None:
        """Sync context manager — stops the spinner."""
        self.stop()

    async def __aenter__(self) -> Spinner:
        """Async context manager — starts the spinner."""
        self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager — stops the spinner."""
        self.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Progress Bar
# ──────────────────────────────────────────────────────────────────────────────

class ProgressBar:
    """
    Terminal progress bar.

    Parameters
    ----------
    total:
        Total items/percentage (default 100).
    width:
        Width of the progress bar in characters (default 40).
    prefix:
        Text displayed before the bar.
    suffix:
        Text displayed after the bar.
    fill_char:
        Character used for the filled portion.
    empty_char:
        Character used for the empty portion.
    color:
        Color for the filled portion.
    """

    def __init__(
        self,
        total: int = 100,
        width: int = 40,
        prefix: str = "",
        suffix: str = "",
        fill_char: str = "█",
        empty_char: str = "░",
        color: ColorCode = ColorCode.GREEN,
    ) -> None:
        self._total = max(1, total)
        self._width = width
        self._prefix = prefix
        self._suffix = suffix
        self._fill_char = fill_char
        self._empty_char = empty_char
        self._color = color
        self._current = 0
        self._start_time: Optional[float] = None

    @property
    def progress(self) -> float:
        """Current progress as a fraction (0.0–1.0)."""
        return self._current / self._total

    @property
    def percentage(self) -> float:
        """Current progress as a percentage."""
        return self.progress * 100

    def update(self, current: int) -> None:
        """
        Update the progress bar to a new value.

        Parameters
        ----------
        current:
            Current progress value.
        """
        if self._start_time is None:
            self._start_time = time.monotonic()
        self._current = min(max(0, current), self._total)
        self._render()

    def increment(self, amount: int = 1) -> None:
        """Increment progress by the given amount."""
        self.update(self._current + amount)

    def finish(self, message: Optional[str] = None) -> None:
        """
        Mark progress as complete.

        Parameters
        ----------
        message:
            Optional completion message.
        """
        self._current = self._total
        self._render()

        elapsed = 0.0
        if self._start_time:
            elapsed = time.monotonic() - self._start_time

        # Print final line
        sys.stdout.write("\n")
        if message:
            sys.stdout.write(
                f"  {colorize(message, ColorCode.GREEN)} "
                f"({elapsed:.1f}s)\n"
            )
        sys.stdout.flush()

    def _render(self) -> None:
        """Render the progress bar to stdout."""
        pct = self.percentage
        filled = int(self._width * self.progress)
        bar = (
            self._fill_char * filled
            + self._empty_char * (self._width - filled)
        )
        line = (
            f"\r  {self._prefix}"
            f"{self._color.value}{bar}{ColorCode.RESET.value}"
            f" {pct:5.1f}%"
            f" {self._suffix}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()


# ──────────────────────────────────────────────────────────────────────────────
# Tool Preview
# ──────────────────────────────────────────────────────────────────────────────

class ToolPreview:
    """
    Formats tool call previews for terminal display.

    Provides color-coded, compact formatting for tool names, inputs,
    and results suitable for CLI output.
    """

    @staticmethod
    def format_tool_call(tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Format a tool call for display.

        Parameters
        ----------
        tool_name:
            Name of the tool being called.
        tool_input:
            Tool input parameters.

        Returns
        -------
        str
            Formatted tool call string.
        """
        import json

        emoji = render_emoji("tool")
        name = colorize(tool_name, ColorCode.BRIGHT_CYAN, bold=True)

        # Truncate input display
        input_str = json.dumps(tool_input, default=str, ensure_ascii=False)
        if len(input_str) > 200:
            input_str = input_str[:200] + "..."

        return f"  {emoji} {name} {colorize(input_str, ColorCode.DIM)}"

    @staticmethod
    def format_tool_result(
        tool_name: str,
        result: str,
        is_error: bool = False,
    ) -> str:
        """
        Format a tool result for display.

        Parameters
        ----------
        tool_name:
            Name of the tool.
        result:
            The tool result text.
        is_error:
            Whether the result is an error.

        Returns
        -------
        str
            Formatted result string.
        """
        emoji = render_emoji("cross" if is_error else "check")
        color = ColorCode.RED if is_error else ColorCode.DIM

        # Truncate result
        display = result[:300].replace("\n", "\\n")
        if len(result) > 300:
            display += "..."

        return f"  {emoji} {colorize(tool_name, color)} → {colorize(display, color)}"

    @staticmethod
    def format_cost(cost_usd: float) -> str:
        """Format a cost value with emoji."""
        emoji = render_emoji("money")
        if cost_usd < 0.01:
            color = ColorCode.GREEN
        elif cost_usd < 0.10:
            color = ColorCode.YELLOW
        else:
            color = ColorCode.RED
        return f"  {emoji} {colorize(f'${cost_usd:.4f}', color)}"

    @staticmethod
    def format_tokens(input_tokens: int, output_tokens: int) -> str:
        """Format token counts."""
        return (
            f"  {render_emoji('chart')} "
            f"{colorize(f'{input_tokens:,}', ColorCode.BRIGHT_BLUE)} in / "
            f"{colorize(f'{output_tokens:,}', ColorCode.BRIGHT_MAGENTA)} out"
        )

    @staticmethod
    def format_model(model_name: str) -> str:
        """Format a model name display."""
        emoji = render_emoji("robot")
        return f"  {emoji} {colorize(model_name, ColorCode.BRIGHT_WHITE, bold=True)}"


# ──────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def status_message(message: str, color: ColorCode = ColorCode.CYAN):
    """
    Context manager that shows a spinner with a message.

    Replaces the spinner with a check mark on success.

    Usage
    -----
        async with status_message("Loading model..."):
            await model.load()
    """
    spinner = Spinner(message, color=color)
    spinner.start()
    try:
        yield spinner
        spinner.stop(f"  {render_emoji('check')} {colorize(message, ColorCode.GREEN)}")
    except Exception:
        spinner.stop(f"  {render_emoji('cross')} {colorize(message, ColorCode.RED)}")
        raise


def print_info(message: str) -> None:
    """Print an info message with emoji."""
    print(f"  {render_emoji('info')} {colorize(message, ColorCode.BRIGHT_BLUE)}")


def print_success(message: str) -> None:
    """Print a success message with emoji."""
    print(f"  {render_emoji('check')} {colorize(message, ColorCode.GREEN)}")


def print_warning(message: str) -> None:
    """Print a warning message with emoji."""
    print(f"  {render_emoji('warning')} {colorize(message, ColorCode.YELLOW)}")


def print_error(message: str) -> None:
    """Print an error message with emoji."""
    print(f"  {render_emoji('cross')} {colorize(message, ColorCode.RED)}")
