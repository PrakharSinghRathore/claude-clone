"""
Skin/Theme engine for Hermes CLI.

Provides customizable visual themes with:
- Built-in themes (dark, light, solarized, nord, dracula, catppuccin, monokai, gruvbox)
- Custom theme creation from YAML files
- Color palette management
- Syntax highlighting themes
- PS1-style prompt customization
- Dynamic theming based on time/context
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

THEMES_DIR = Path.home() / ".claude_clone" / "hermes" / "themes"


# ──────────────────────────────────────────────
# Color definitions (ANSI-compatible)
# ──────────────────────────────────────────────

ANSI_ESCAPE = re.compile(r"\033\[[0-9;]*m")
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
REVERSE = "\033[7m"

# 256-color helper
def color256(fg: int, bg: Optional[int] = None) -> str:
    """Generate 256-color ANSI code."""
    if bg is not None:
        return f"\033[38;5;{fg}m\033[48;5;{bg}m"
    return f"\033[38;5;{fg}m"


def rgb_color(r: int, g: int, b: int) -> str:
    """Generate true-color ANSI code."""
    return f"\033[38;2;{r};{g};{b}m"


def bg_rgb(r: int, g: int, b: int) -> str:
    """Generate true-color background ANSI code."""
    return f"\033[48;2;{r};{g};{b}m"


# ──────────────────────────────────────────────
# Built-in themes
# ──────────────────────────────────────────────

BUILTIN_THEMES = {
    "dark": {
        "name": "Dark",
        "description": "Default dark theme",
        "background": "#1a1b26",
        "foreground": "#c0caf5",
        "cursor": "#c0caf5",
        "selection_bg": "#33467c",
        "border": "#3b4261",
        "colors": {
            "black": "#15161e",
            "red": "#f7768e",
            "green": "#9ece6a",
            "yellow": "#e0af68",
            "blue": "#7aa2f7",
            "magenta": "#bb9af7",
            "cyan": "#7dcfff",
            "white": "#a9b1d6",
            "bright_black": "#414868",
            "bright_red": "#f7768e",
            "bright_green": "#9ece6a",
            "bright_yellow": "#e0af68",
            "bright_blue": "#7aa2f7",
            "bright_magenta": "#bb9af7",
            "bright_cyan": "#7dcfff",
            "bright_white": "#c0caf5",
        },
        "ui": {
            "user_input": "bright_blue",
            "assistant_output": "white",
            "tool_call": "yellow",
            "tool_result": "green",
            "error": "red",
            "warning": "yellow",
            "info": "cyan",
            "success": "green",
            "heading": "bright_magenta",
            "link": "bright_blue",
            "code": "green",
            "code_bg": "black",
            "table_border": "bright_black",
            "muted": "bright_black",
            "dim": "black",
        },
        "prompt": {
            "style": "hermes",
            "prefix": ">>>",
            "prefix_color": "bright_magenta",
            "separator": ">",
            "separator_color": "bright_blue",
            "suffix_color": "white",
        },
    },
    "light": {
        "name": "Light",
        "description": "Clean light theme",
        "background": "#f5f5f5",
        "foreground": "#333333",
        "cursor": "#333333",
        "selection_bg": "#d0d0d0",
        "border": "#cccccc",
        "colors": {
            "black": "#000000",
            "red": "#cc0000",
            "green": "#009900",
            "yellow": "#cc9900",
            "blue": "#0055cc",
            "magenta": "#9900cc",
            "cyan": "#009999",
            "white": "#cccccc",
            "bright_black": "#666666",
            "bright_red": "#ff3333",
            "bright_green": "#33cc33",
            "bright_yellow": "#ffcc00",
            "bright_blue": "#3366ff",
            "bright_magenta": "#cc33ff",
            "bright_cyan": "#33cccc",
            "bright_white": "#ffffff",
        },
        "ui": {
            "user_input": "blue",
            "assistant_output": "black",
            "tool_call": "yellow",
            "tool_result": "green",
            "error": "red",
            "warning": "#cc6600",
            "info": "blue",
            "success": "green",
            "heading": "magenta",
            "link": "blue",
            "code": "green",
            "code_bg": "#e8e8e8",
            "table_border": "#cccccc",
            "muted": "#999999",
            "dim": "#cccccc",
        },
        "prompt": {
            "style": "hermes",
            "prefix": ">>>",
            "prefix_color": "magenta",
            "separator": ">",
            "separator_color": "blue",
            "suffix_color": "black",
        },
    },
    "solarized": {
        "name": "Solarized",
        "description": "Solarized color scheme",
        "background": "#002b36",
        "foreground": "#839496",
        "cursor": "#93a1a1",
        "selection_bg": "#073642",
        "border": "#586e75",
        "colors": {
            "black": "#073642",
            "red": "#dc322f",
            "green": "#859900",
            "yellow": "#b58900",
            "blue": "#268bd2",
            "magenta": "#d33682",
            "cyan": "#2aa198",
            "white": "#eee8d5",
            "bright_black": "#002b36",
            "bright_red": "#cb4b16",
            "bright_green": "#586e75",
            "bright_yellow": "#657b83",
            "bright_blue": "#839496",
            "bright_magenta": "#6c71c4",
            "bright_cyan": "#93a1a1",
            "bright_white": "#fdf6e3",
        },
        "ui": {
            "user_input": "blue",
            "assistant_output": "white",
            "tool_call": "yellow",
            "tool_result": "green",
            "error": "red",
            "warning": "yellow",
            "info": "cyan",
            "success": "green",
            "heading": "magenta",
            "link": "blue",
            "code": "green",
            "code_bg": "black",
            "table_border": "bright_black",
            "muted": "bright_black",
            "dim": "black",
        },
        "prompt": {
            "style": "hermes",
            "prefix": ">>>",
            "prefix_color": "magenta",
            "separator": ">",
            "separator_color": "blue",
            "suffix_color": "white",
        },
    },
    "nord": {
        "name": "Nord",
        "description": "Nord color palette - arctic blue",
        "background": "#2e3440",
        "foreground": "#d8dee9",
        "cursor": "#d8dee9",
        "selection_bg": "#434c5e",
        "border": "#4c566a",
        "colors": {
            "black": "#3b4252",
            "red": "#bf616a",
            "green": "#a3be8c",
            "yellow": "#ebcb8b",
            "blue": "#81a1c1",
            "magenta": "#b48ead",
            "cyan": "#88c0d0",
            "white": "#e5e9f0",
            "bright_black": "#4c566a",
            "bright_red": "#bf616a",
            "bright_green": "#a3be8c",
            "bright_yellow": "#ebcb8b",
            "bright_blue": "#81a1c1",
            "bright_magenta": "#b48ead",
            "bright_cyan": "#8fbcbb",
            "bright_white": "#eceff4",
        },
        "ui": {
            "user_input": "blue",
            "assistant_output": "white",
            "tool_call": "yellow",
            "tool_result": "green",
            "error": "red",
            "warning": "yellow",
            "info": "cyan",
            "success": "green",
            "heading": "magenta",
            "link": "blue",
            "code": "green",
            "code_bg": "black",
            "table_border": "bright_black",
            "muted": "bright_black",
            "dim": "black",
        },
        "prompt": {
            "style": "hermes",
            "prefix": ">>>",
            "prefix_color": "magenta",
            "separator": ">",
            "separator_color": "blue",
            "suffix_color": "white",
        },
    },
    "dracula": {
        "name": "Dracula",
        "description": "Dark purple theme",
        "background": "#282a36",
        "foreground": "#f8f8f2",
        "cursor": "#f8f8f2",
        "selection_bg": "#44475a",
        "border": "#6272a4",
        "colors": {
            "black": "#21222c",
            "red": "#ff5555",
            "green": "#50fa7b",
            "yellow": "#f1fa8c",
            "blue": "#bd93f9",
            "magenta": "#ff79c6",
            "cyan": "#8be9fd",
            "white": "#f8f8f2",
            "bright_black": "#6272a4",
            "bright_red": "#ff6e6e",
            "bright_green": "#69ff94",
            "bright_yellow": "#ffffa5",
            "bright_blue": "#d6acff",
            "bright_magenta": "#ff92df",
            "bright_cyan": "#a4ffff",
            "bright_white": "#ffffff",
        },
        "ui": {
            "user_input": "blue",
            "assistant_output": "white",
            "tool_call": "yellow",
            "tool_result": "green",
            "error": "red",
            "warning": "yellow",
            "info": "cyan",
            "success": "green",
            "heading": "magenta",
            "link": "blue",
            "code": "green",
            "code_bg": "black",
            "table_border": "bright_black",
            "muted": "bright_black",
            "dim": "black",
        },
        "prompt": {
            "style": "hermes",
            "prefix": ">>>",
            "prefix_color": "magenta",
            "separator": ">",
            "separator_color": "blue",
            "suffix_color": "white",
        },
    },
    "catppuccin": {
        "name": "Catppuccin Mocha",
        "description": "Soothing dark theme",
        "background": "#1e1e2e",
        "foreground": "#cdd6f4",
        "cursor": "#f5e0dc",
        "selection_bg": "#45475a",
        "border": "#585b70",
        "colors": {
            "black": "#45475a",
            "red": "#f38ba8",
            "green": "#a6e3a1",
            "yellow": "#f9e2af",
            "blue": "#89b4fa",
            "magenta": "#f5c2e7",
            "cyan": "#94e2d5",
            "white": "#bac2de",
            "bright_black": "#585b70",
            "bright_red": "#f38ba8",
            "bright_green": "#a6e3a1",
            "bright_yellow": "#f9e2af",
            "bright_blue": "#89b4fa",
            "bright_magenta": "#f5c2e7",
            "bright_cyan": "#94e2d5",
            "bright_white": "#a6adc8",
        },
        "ui": {
            "user_input": "blue",
            "assistant_output": "white",
            "tool_call": "yellow",
            "tool_result": "green",
            "error": "red",
            "warning": "yellow",
            "info": "cyan",
            "success": "green",
            "heading": "magenta",
            "link": "blue",
            "code": "green",
            "code_bg": "black",
            "table_border": "bright_black",
            "muted": "bright_black",
            "dim": "black",
        },
        "prompt": {
            "style": "hermes",
            "prefix": ">>>",
            "prefix_color": "magenta",
            "separator": ">",
            "separator_color": "blue",
            "suffix_color": "white",
        },
    },
    "monokai": {
        "name": "Monokai",
        "description": "Classic Monokai theme",
        "background": "#272822",
        "foreground": "#f8f8f2",
        "cursor": "#f8f8f2",
        "selection_bg": "#49483e",
        "border": "#75715e",
        "colors": {
            "black": "#272822",
            "red": "#f92672",
            "green": "#a6e22e",
            "yellow": "#f4bf75",
            "blue": "#66d9ef",
            "magenta": "#ae81ff",
            "cyan": "#a1efe4",
            "white": "#f8f8f2",
            "bright_black": "#75715e",
            "bright_red": "#f92672",
            "bright_green": "#a6e22e",
            "bright_yellow": "#e6db74",
            "bright_blue": "#66d9ef",
            "bright_magenta": "#ae81ff",
            "bright_cyan": "#a1efe4",
            "bright_white": "#f9f8f5",
        },
        "ui": {
            "user_input": "blue",
            "assistant_output": "white",
            "tool_call": "yellow",
            "tool_result": "green",
            "error": "red",
            "warning": "yellow",
            "info": "cyan",
            "success": "green",
            "heading": "magenta",
            "link": "blue",
            "code": "green",
            "code_bg": "black",
            "table_border": "bright_black",
            "muted": "bright_black",
            "dim": "black",
        },
        "prompt": {
            "style": "hermes",
            "prefix": ">>>",
            "prefix_color": "magenta",
            "separator": ">",
            "separator_color": "blue",
            "suffix_color": "white",
        },
    },
    "gruvbox": {
        "name": "Gruvbox Dark",
        "description": "Warm retro theme",
        "background": "#282828",
        "foreground": "#ebdbb2",
        "cursor": "#ebdbb2",
        "selection_bg": "#504945",
        "border": "#665c54",
        "colors": {
            "black": "#282828",
            "red": "#cc241d",
            "green": "#98971a",
            "yellow": "#d79921",
            "blue": "#458588",
            "magenta": "#b16286",
            "cyan": "#689d6a",
            "white": "#a89984",
            "bright_black": "#928374",
            "bright_red": "#fb4934",
            "bright_green": "#b8bb26",
            "bright_yellow": "#fabd2f",
            "bright_blue": "#83a598",
            "bright_magenta": "#d3869b",
            "bright_cyan": "#8ec07c",
            "bright_white": "#ebdbb2",
        },
        "ui": {
            "user_input": "blue",
            "assistant_output": "white",
            "tool_call": "yellow",
            "tool_result": "green",
            "error": "red",
            "warning": "yellow",
            "info": "cyan",
            "success": "green",
            "heading": "magenta",
            "link": "blue",
            "code": "green",
            "code_bg": "black",
            "table_border": "bright_black",
            "muted": "bright_black",
            "dim": "black",
        },
        "prompt": {
            "style": "hermes",
            "prefix": ">>>",
            "prefix_color": "magenta",
            "separator": ">",
            "separator_color": "blue",
            "suffix_color": "white",
        },
    },
}

# Prompt style templates
PROMPT_STYLES = {
    "hermes": {
        "prefix": ">>>",
        "format": "{prefix}{agent}{separator} ",
        "colors": ("bright_magenta", "bright_blue", "white"),
    },
    "claude": {
        "prefix": "claude",
        "format": "{prefix} {agent}{separator} ",
        "colors": ("green", "white", "white"),
    },
    "minimal": {
        "prefix": ">",
        "format": "{prefix} ",
        "colors": ("white", "white", "white"),
    },
    "powerline": {
        "prefix": "\u276f",
        "format": " {prefix} {agent} {separator} ",
        "colors": ("cyan", "blue", "white"),
    },
    "starship": {
        "prefix": "\u276f",
        "format": "{prefix}[{agent}] {separator} ",
        "colors": ("bright_green", "bright_blue", "bright_cyan"),
    },
    "fancy": {
        "prefix": "\u2666",
        "format": " {prefix} {agent}{separator} ",
        "colors": ("magenta", "yellow", "white"),
    },
}


class SkinEngine:
    """Manages visual themes and skins for the Hermes CLI."""

    BUILTIN_THEMES = BUILTIN_THEMES

    def __init__(self, theme_name: str = "dark", themes_dir: Optional[Path] = None):
        self.themes_dir = themes_dir or THEMES_DIR
        self.themes_dir.mkdir(parents=True, exist_ok=True)
        self.current_theme_name = theme_name
        self._theme: Optional[Dict] = None
        self.load_theme(theme_name)

    def load_theme(self, name: str) -> Dict[str, Any]:
        """Load a theme by name."""
        # Check built-ins first
        if name in BUILTIN_THEMES:
            self._theme = deepcopy(BUILTIN_THEMES[name])
            self.current_theme_name = name
            return self._theme

        # Check custom themes
        custom = self._load_custom_theme(name)
        if custom:
            self._theme = custom
            self.current_theme_name = name
            return self._theme

        # Fall back to dark
        self._theme = deepcopy(BUILTIN_THEMES["dark"])
        self.current_theme_name = "dark"
        return self._theme

    def get_theme(self) -> Dict[str, Any]:
        """Get current active theme."""
        return self._theme or BUILTIN_THEMES["dark"]

    def get_color(self, role: str) -> str:
        """Get a hex color for a UI role."""
        theme = self.get_theme()
        ui = theme.get("ui", {})
        color_name = ui.get(role, "white")
        colors = theme.get("colors", {})

        # Direct color lookup
        if color_name in colors:
            return colors[color_name]

        # Check if it's already a hex color
        if color_name.startswith("#"):
            return color_name

        return "#ffffff"

    def get_color_ansi(self, role: str) -> str:
        """Get ANSI color code for a UI role."""
        hex_color = self.get_color(role)
        r, g, b = self._hex_to_rgb(hex_color)
        return rgb_color(r, g, b)

    def get_bg_ansi(self) -> str:
        """Get background ANSI code."""
        theme = self.get_theme()
        bg = theme.get("background", "#1a1b26")
        r, g, b = self._hex_to_rgb(bg)
        return bg_rgb(r, g, b)

    def get_prompt_style(self, style_name: Optional[str] = None) -> Dict:
        """Get prompt style configuration."""
        style_key = style_name or self.get_theme().get("prompt", {}).get("style", "hermes")
        return PROMPT_STYLES.get(style_key, PROMPT_STYLES["hermes"])

    def build_prompt(self, agent_name: str = "", style_name: Optional[str] = None) -> str:
        """Build a formatted prompt string with colors."""
        theme = self.get_theme()
        prompt_cfg = theme.get("prompt", {})
        style = self.get_prompt_style(style_name or prompt_cfg.get("style"))

        prefix = style["prefix"]
        colors = style["colors"]
        theme_colors = theme.get("colors", {})

        parts = []
        for i, text in enumerate([prefix, agent_name, style.get("separator", ">")]):
            color_name = colors[i] if i < len(colors) else "white"
            hex_color = theme_colors.get(color_name, "#ffffff")
            r, g, b = self._hex_to_rgb(hex_color)
            parts.append(f"{rgb_color(r, g, b)}{text}{RESET}")

        return "".join(parts)

    def list_themes(self) -> List[Dict[str, str]]:
        """List all available themes."""
        themes = []

        for name, theme in BUILTIN_THEMES.items():
            themes.append({
                "name": name,
                "description": theme.get("description", ""),
                "source": "builtin",
                "background": theme.get("background", ""),
            })

        # Custom themes
        for f in self.themes_dir.glob("*.yaml"):
            try:
                custom = self._load_custom_theme(f.stem)
                if custom:
                    themes.append({
                        "name": f.stem,
                        "description": custom.get("description", "Custom theme"),
                        "source": "custom",
                        "background": custom.get("background", ""),
                    })
            except Exception:
                pass

        return themes

    def create_custom_theme(self, name: str, base: str = "dark", overrides: Optional[Dict] = None) -> Path:
        """Create a custom theme based on an existing one."""
        base_theme = BUILTIN_THEMES.get(base, BUILTIN_THEMES["dark"])
        custom = deepcopy(base_theme)
        custom["name"] = name

        if overrides:
            self._deep_merge(custom, overrides)

        path = self.themes_dir / f"{name}.yaml"
        if HAS_YAML:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(custom, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(custom, f, indent=2)

        return path

    def delete_custom_theme(self, name: str) -> bool:
        """Delete a custom theme."""
        for ext in ("yaml", "json"):
            path = self.themes_dir / f"{name}.{ext}"
            if path.exists():
                path.unlink()
                return True
        return False

    def export_theme(self, name: str, path: str) -> bool:
        """Export a theme to a file."""
        theme = self.load_theme(name)
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(export_path, "w", encoding="utf-8") as f:
                yaml.dump(theme, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(theme, f, indent=2)
        return True

    def import_theme(self, path: str) -> bool:
        """Import a theme from a file."""
        import_path = Path(path)
        if not import_path.exists():
            return False

        try:
            if HAS_YAML:
                with open(import_path, "r", encoding="utf-8") as f:
                    theme = yaml.safe_load(f) or {}
            else:
                with open(import_path, "r", encoding="utf-8") as f:
                    theme = json.load(f)

            name = theme.get("name", import_path.stem)
            dest = self.themes_dir / f"{name}.yaml"
            if HAS_YAML:
                with open(dest, "w", encoding="utf-8") as f:
                    yaml.dump(theme, f, default_flow_style=False, allow_unicode=True)
            else:
                with open(dest.with_suffix(".json"), "w", encoding="utf-8") as f:
                    json.dump(theme, f, indent=2)
            return True
        except Exception:
            return False

    def get_dynamic_theme(self) -> Optional[str]:
        """Suggest a theme based on current time/context."""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "light"
        elif 12 <= hour < 18:
            return "nord"
        elif 18 <= hour < 21:
            return "dracula"
        else:
            return "dark"

    def _load_custom_theme(self, name: str) -> Optional[Dict]:
        """Load a custom theme from file."""
        for ext in ("yaml", "json"):
            path = self.themes_dir / f"{name}.{ext}"
            if path.exists():
                try:
                    if ext == "yaml" and HAS_YAML:
                        with open(path, "r", encoding="utf-8") as f:
                            return yaml.safe_load(f) or {}
                    else:
                        with open(path, "r", encoding="utf-8") as f:
                            return json.load(f)
                except Exception:
                    continue
        return None

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (255, 255, 255)

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                SkinEngine._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    @staticmethod
    def strip_ansi(text: str) -> str:
        """Remove ANSI escape sequences from text."""
        return ANSI_ESCAPE.sub("", text)
