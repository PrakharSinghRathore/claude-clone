"""
Startup banner, branding, and system info display for Hermes CLI.
"""

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes.cli_hermes.skin_engine import SkinEngine, rgb_color, RESET, BOLD, DIM

__version__ = "1.0.0"


# ──────────────────────────────────────────────
# ASCII art banners
# ──────────────────────────────────────────────

HERMES_LOGO = """
    ╔══════════════════════════════════════════╗
    ║                                          ║
    ║   ██╗  ██╗ ██████╗ ███╗   ██╗████████╗  ║
    ║   ██║ ██╔╝██╔═══██╗████╗  ██║╚══██╔══╝  ║
    ║   █████╔╝ ██║   ██║██╔██╗ ██║   ██║     ║
    ║   ██╔═██╗ ██║   ██║██║╚██╗██║   ██║     ║
    ║   ██║  ██╗╚██████╔╝██║ ╚████║   ██║     ║
    ║   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝     ║
    ║                                          ║
    ║        Claude Clone · Interactive CLI    ║
    ║                                          ║
    ╚══════════════════════════════════════════╝
"""

HERMES_MINI = """
    ⚡ Hermes CLI v{version}
"""

# Motivational quotes
QUOTES = [
    ("The best way to predict the future is to invent it.", "Alan Kay"),
    ("Any sufficiently advanced technology is indistinguishable from magic.", "Arthur C. Clarke"),
    ("Simplicity is the ultimate sophistication.", "Leonardo da Vinci"),
    ("Talk is cheap. Show me the code.", "Linus Torvalds"),
    ("First, solve the problem. Then, write the code.", "John Johnson"),
    ("The only way to learn a new programming language is by writing programs in it.", "Dennis Ritchie"),
    ("Programs must be written for people to read.", "Harold Abelson"),
    ("The most important property of a program is whether it accomplishes the intention of its user.", "C.A.R. Hoare"),
    ("Code is like humor. When you have to explain it, it's bad.", "Cory House"),
    ("Experience is the name everyone gives to their mistakes.", "Oscar Wilde"),
    ("Make it work, make it right, make it fast.", "Kent Beck"),
    ("The function of good software is to make the complex appear to be simple.", "Grady Booch"),
    ("Before software can be reusable it first has to be usable.", "Ralph Johnson"),
    ("It's not a bug — it's an undocumented feature.", "Anonymous"),
    ("The computer was born to solve problems that did not exist before.", "Bill Gates"),
]

# Quick tips
TIPS = [
    "Use /help to see all available commands",
    "Press Shift+Enter for multi-line input",
    "Type /model to switch AI models on the fly",
    "Use /save and /load to manage sessions",
    "Press Tab to autocomplete commands and file paths",
    "Use /export to save conversations as Markdown",
    "Type /theme to change visual themes",
    "Use /profile to switch configuration profiles",
    "Press Ctrl+C to cancel current generation",
    "Type /doctor to run system diagnostics",
    "Use /agent to switch between specialized AI agents",
    "Press Ctrl+L to clear the screen",
]


class Banner:
    """Manages startup banner, branding, and system info display."""

    def __init__(self, skin: Optional[SkinEngine] = None):
        self.skin = skin or SkinEngine()
        self._tip_index = 0

    def show(
        self,
        show_logo: bool = True,
        show_system: bool = True,
        show_tip: bool = True,
        show_quote: bool = False,
        compact: bool = False,
    ) -> str:
        """Generate the full startup banner."""
        lines = []

        if compact:
            lines.append(self.show_mini())
            return "\n".join(lines)

        if show_logo:
            lines.append(self._colorize_logo())

        lines.append(self._version_line())

        if show_system:
            lines.append(self._system_info())

        if show_tip:
            lines.append(self._random_tip())

        if show_quote:
            lines.append(self._random_quote())

        lines.append("")  # blank line
        return "\n".join(lines)

    def show_mini(self) -> str:
        """Generate a minimal one-line banner."""
        return HERMES_MINI.format(version=__version__).strip()

    def _colorize_logo(self) -> str:
        """Colorize the ASCII logo with theme colors."""
        theme = self.skin.get_theme()
        colors = theme.get("colors", {})

        # Get theme colors for gradient
        accent = colors.get("bright_magenta", colors.get("magenta", "#bb9af7"))
        secondary = colors.get("bright_blue", colors.get("blue", "#7aa2f7"))
        tertiary = colors.get("bright_cyan", colors.get("cyan", "#7dcfff"))

        lines = HERMES_LOGO.strip().split("\n")
        colored_lines = []

        for i, line in enumerate(lines):
            # Simple gradient based on position
            ratio = i / max(len(lines) - 1, 1)
            if ratio < 0.3:
                color = accent
            elif ratio < 0.7:
                color = secondary
            else:
                color = tertiary

            r, g, b = self.skin._hex_to_rgb(color)
            colored_lines.append(f"{rgb_color(r, g, b)}{line}{RESET}")

        return "\n".join(colored_lines)

    def _version_line(self) -> str:
        """Generate version and build info line."""
        theme = self.skin.get_theme()
        colors = theme.get("colors", {})
        accent = colors.get("bright_cyan", colors.get("cyan", "#7dcfff"))
        r, g, b = self.skin._hex_to_rgb(accent)

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"{rgb_color(r, g, b)}  Version {__version__}  ·  {now}  ·  {self._python_version()}{RESET}"

    def _system_info(self) -> str:
        """Generate system info summary."""
        theme = self.skin.get_theme()
        colors = theme.get("colors", {})
        muted = colors.get("bright_black", colors.get("black", "#414868"))
        r, g, b = self.skin._hex_to_rgb(muted)

        info_parts = [
            f"{platform.system()} {platform.release()}",
            f"Python {self._python_version()}",
            platform.machine() or "",
            os.environ.get("TERM", "unknown terminal"),
        ]

        info_str = "  ·  ".join(p for p in info_parts if p)
        return f"{DIM}{info_str}{RESET}"

    def _python_version(self) -> str:
        """Get Python version string."""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _random_tip(self) -> str:
        """Get a random tip."""
        import random
        tip = TIPS[self._tip_index % len(TIPS)]
        self._tip_index += 1

        theme = self.skin.get_theme()
        colors = theme.get("colors", {})
        yellow = colors.get("yellow", colors.get("bright_yellow", "#e0af68"))
        r, g, b = self.skin._hex_to_rgb(yellow)

        return f"\n  {rgb_color(r, g, b)}\u2b50 {tip}{RESET}"

    def _random_quote(self) -> str:
        """Get a random motivational quote."""
        import random
        quote, author = random.choice(QUOTES)

        theme = self.skin.get_theme()
        colors = theme.get("colors", {})
        muted = colors.get("bright_black", colors.get("black", "#414868"))
        r, g, b = self.skin._hex_to_rgb(muted)

        return f"\n  {DIM}\u201c{quote}\u201d \u2014 {author}{RESET}"

    def show_welcome(self, username: Optional[str] = None) -> str:
        """Show a personalized welcome message."""
        name = username or os.environ.get("USER", "User")
        hour = datetime.now().hour

        if hour < 6:
            greeting = "Good night"
        elif hour < 12:
            greeting = "Good morning"
        elif hour < 18:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"

        theme = self.skin.get_theme()
        colors = theme.get("colors", {})
        accent = colors.get("bright_green", colors.get("green", "#9ece6a"))
        r, g, b = self.skin._hex_to_rgb(accent)

        return f"\n  {rgb_color(r, g, b)}{greeting}, {name}! Welcome to Hermes CLI.{RESET}"

    def show_profile_info(self, profile: str, model: str, provider: str) -> str:
        """Show current profile, model, and provider info."""
        theme = self.skin.get_theme()
        colors = theme.get("colors", {})
        muted = colors.get("bright_black", colors.get("black", "#414868"))
        r, g, b = self.skin._hex_to_rgb(muted)

        lines = [
            f"  Profile: {profile}  |  Model: {model}  |  Provider: {provider}",
        ]
        return f"{DIM}{'  ·  '.join(lines)}{RESET}"
