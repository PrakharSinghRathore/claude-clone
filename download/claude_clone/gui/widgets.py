"""
Custom tkinter widgets for the Cowork desktop GUI.

Includes:
- Themed widgets with dark/light mode support
- Code viewer with syntax highlighting
- Tool call card (collapsible)
- Markdown text widget
- Streaming text display
- Prompt input with placeholder
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import re
from typing import Optional


# ──────────────────────────────────────────────
# Color themes
# ──────────────────────────────────────────────

THEMES = {
    "dark": {
        "bg": "#1e1e1e",
        "bg_secondary": "#252526",
        "bg_tertiary": "#2d2d30",
        "fg": "#d4d4d4",
        "fg_dim": "#808080",
        "fg_bright": "#ffffff",
        "accent": "#007acc",
        "accent_hover": "#1a8ad4",
        "success": "#4ec9b0",
        "warning": "#dcdcaa",
        "error": "#f44747",
        "tool_bg": "#1e1e2e",
        "tool_border": "#44475a",
        "input_bg": "#1e1e1e",
        "input_fg": "#d4d4d4",
        "border": "#3e3e42",
        "selection": "#264f78",
        "scrollbar": "#424242",
        "user_msg": "#1a472a",
        "assistant_msg": "#1e1e2e",
        "code_bg": "#1e1e1e",
        "header_bg": "#007acc",
        "header_fg": "#ffffff",
        "button_bg": "#3a3d41",
        "button_fg": "#cccccc",
        "button_hover": "#45494e",
        "sidebar_bg": "#252526",
    },
    "light": {
        "bg": "#ffffff",
        "bg_secondary": "#f3f3f3",
        "bg_tertiary": "#e8e8e8",
        "fg": "#1e1e1e",
        "fg_dim": "#6e6e6e",
        "fg_bright": "#000000",
        "accent": "#0066b8",
        "accent_hover": "#005a9e",
        "success": "#16825d",
        "warning": "#795e26",
        "error": "#d1242f",
        "tool_bg": "#f0f0f0",
        "tool_border": "#d4d4d4",
        "input_bg": "#ffffff",
        "input_fg": "#1e1e1e",
        "border": "#d4d4d4",
        "selection": "#add6ff",
        "scrollbar": "#c1c1c1",
        "user_msg": "#dff6dd",
        "assistant_msg": "#f0f0ff",
        "code_bg": "#f5f5f5",
        "header_bg": "#007acc",
        "header_fg": "#ffffff",
        "button_bg": "#e4e6e7",
        "button_fg": "#1e1e1e",
        "button_hover": "#d0d3d5",
        "sidebar_bg": "#f3f3f3",
    },
}

# Simple syntax highlighting colors (approximate)
PYTHON_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
    "try", "while", "with", "yield",
}

SYNTAX_COLORS_DARK = {
    "keyword": "#569cd6",
    "string": "#ce9178",
    "comment": "#6a9955",
    "function": "#dcdcaa",
    "number": "#b5cea8",
    "decorator": "#dcdcaa",
    "builtin": "#4ec9b0",
    "class": "#4ec9b0",
}

SYNTAX_COLORS_LIGHT = {
    "keyword": "#0000ff",
    "string": "#a31515",
    "comment": "#008000",
    "function": "#795e26",
    "number": "#098658",
    "decorator": "#795e26",
    "builtin": "#267f99",
    "class": "#267f99",
}


def get_theme_colors(theme: str = "dark") -> dict:
    """Get theme color dictionary."""
    return THEMES.get(theme, THEMES["dark"])


# ──────────────────────────────────────────────
# Custom Widgets
# ──────────────────────────────────────────────

class ThemedFrame(tk.Frame):
    """A frame with theme support."""

    def __init__(self, parent, theme: str = "dark", bg_key: str = "bg", **kwargs):
        self.theme = theme
        self.colors = get_theme_colors(theme)
        kwargs.setdefault("bg", self.colors[bg_key])
        super().__init__(parent, **kwargs)

    def update_theme(self, theme: str):
        self.theme = theme
        self.colors = get_theme_colors(theme)
        self.configure(bg=self.colors["bg"])


class ThemedButton(tk.Button):
    """A themed button."""

    def __init__(self, parent, text: str = "", command=None, theme: str = "dark",
                 style: str = "default", **kwargs):
        self.theme = theme
        self.colors = get_theme_colors(theme)
        self._style = style

        bg = self.colors.get("button_bg", "#3a3d41")
        fg = self.colors.get("button_fg", "#cccccc")
        active_bg = self.colors.get("button_hover", "#45494e")
        active_fg = fg
        relief = "flat"
        borderwidth = 1
        padx = 12
        pady = 6

        if style == "primary":
            bg = self.colors.get("accent", "#007acc")
            fg = "#ffffff"
            active_bg = self.colors.get("accent_hover", "#1a8ad4")
            active_fg = "#ffffff"

        if style == "danger":
            bg = self.colors.get("error", "#f44747")
            fg = "#ffffff"
            active_bg = "#d32f2f"
            active_fg = "#ffffff"

        kwargs.setdefault("bg", bg)
        kwargs.setdefault("fg", fg)
        kwargs.setdefault("activebackground", active_bg)
        kwargs.setdefault("activeforeground", active_fg)
        kwargs.setdefault("relief", relief)
        kwargs.setdefault("borderwidth", borderwidth)
        kwargs.setdefault("padx", padx)
        kwargs.setdefault("pady", pady)
        kwargs.setdefault("cursor", "hand2")
        kwargs.setdefault("font", ("Segoe UI", 10))
        kwargs.setdefault("command", command)

        super().__init__(parent, **kwargs)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, event):
        self.configure(bg=self.colors.get("accent_hover", "#1a8ad4") if self._style == "primary"
                      else self.colors.get("button_hover", "#45494e"))

    def _on_leave(self, event):
        bg = self.colors.get("accent", "#007acc") if self._style == "primary" else self.colors.get("button_bg", "#3a3d41")
        self.configure(bg=bg)


class ThemedEntry(tk.Entry):
    """A themed entry field."""

    def __init__(self, parent, theme: str = "dark", placeholder: str = "", **kwargs):
        self.theme = theme
        self.colors = get_theme_colors(theme)
        self._placeholder = placeholder
        self._has_placeholder = True

        kwargs.setdefault("bg", self.colors.get("input_bg", "#1e1e1e"))
        kwargs.setdefault("fg", self.colors.get("input_fg", "#d4d4d4"))
        kwargs.setdefault("insertbackground", self.colors.get("fg", "#d4d4d4"))
        kwargs.setdefault("relief", "flat")
        kwargs.setdefault("borderwidth", 2)
        kwargs.setdefault("font", ("Consolas", 11))

        super().__init__(parent, **kwargs)

        self.insert(0, placeholder)
        self.configure(fg=self.colors.get("fg_dim", "#808080"))

        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Key>", self._on_key)

    def _on_focus_in(self, event):
        if self._has_placeholder:
            self.delete(0, "end")
            self.configure(fg=self.colors.get("input_fg", "#d4d4d4"))
            self._has_placeholder = False

    def _on_focus_out(self, event):
        if not self.get().strip():
            self.insert(0, self._placeholder)
            self.configure(fg=self.colors.get("fg_dim", "#808080"))
            self._has_placeholder = True

    def _on_key(self, event):
        if self._has_placeholder:
            self.delete(0, "end")
            self.configure(fg=self.colors.get("input_fg", "#d4d4d4"))
            self._has_placeholder = False

    def get_text(self) -> str:
        if self._has_placeholder:
            return ""
        return self.get()


class ChatMessage(tk.Frame):
    """A chat message widget with role, content, and optional tool info."""

    def __init__(self, parent, role: str = "assistant", theme: str = "dark", **kwargs):
        self.theme = theme
        self.colors = get_theme_colors(theme)
        self._role = role

        bg = self.colors.get("assistant_msg", "#1e1e2e") if role == "assistant" else self.colors.get("user_msg", "#1a472a")

        super().__init__(parent, bg=bg, **kwargs)

        # Header
        header_frame = tk.Frame(self, bg=bg)
        header_frame.pack(fill="x", padx=10, pady=(8, 2))

        role_label = role.capitalize()
        if role == "assistant":
            role_label = "🤖 Assistant"
        elif role == "user":
            role_label = "👤 You"
        elif role == "system":
            role_label = "⚙️ System"

        tk.Label(
            header_frame, text=role_label,
            bg=bg, fg=self.colors.get("fg_bright", "#ffffff"),
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")

        # Content
        self._content_text = tk.Text(
            self,
            wrap="word",
            bg=bg,
            fg=self.colors.get("fg", "#d4d4d4"),
            font=("Segoe UI", 10),
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=(2, 8),
            height=1,
            cursor="arrow",
        )
        self._content_text.pack(fill="x")
        self._content_text.configure(state="disabled")

    def add_content(self, text: str):
        """Add text content to the message."""
        self._content_text.configure(state="normal")
        self._content_text.insert("end", text)
        self._content_text.configure(state="disabled")
        self._content_text.configure(height=max(1, self._content_text.get("1.0", "end").count("\n")))

    def set_content(self, text: str):
        """Set the full content of the message."""
        self._content_text.configure(state="normal")
        self._content_text.delete("1.0", "end")
        self._content_text.insert("1.0", text)
        self._content_text.configure(state="disabled")
        self._content_text.configure(height=max(1, text.count("\n")))


class ToolCallCard(tk.Frame):
    """A collapsible card showing a tool call and its result."""

    def __init__(self, parent, tool_name: str, tool_input: dict, theme: str = "dark", **kwargs):
        self.theme = theme
        self.colors = get_theme_colors(theme)
        self._expanded = False

        super().__init__(parent, bg=self.colors.get("tool_bg", "#1e1e2e"),
                        highlightbackground=self.colors.get("tool_border", "#44475a"),
                        highlightthickness=1, **kwargs)

        # Header (clickable to expand/collapse)
        self._header = tk.Frame(self, bg=self.colors.get("tool_bg", "#1e1e2e"), cursor="hand2")
        self._header.pack(fill="x", padx=4, pady=4)

        self._toggle_icon = tk.Label(
            self._header, text="▶", bg=self.colors.get("tool_bg", "#1e1e2e"),
            fg=self.colors.get("warning", "#dcdcaa"), font=("Segoe UI", 8),
        )
        self._toggle_icon.pack(side="left", padx=(4, 2))

        # Spinner
        self._spinner_label = tk.Label(
            self._header, text="⟳", bg=self.colors.get("tool_bg", "#1e1e2e"),
            fg=self.colors.get("accent", "#007acc"), font=("Segoe UI", 10),
        )
        self._spinner_label.pack(side="left", padx=(0, 4))

        tk.Label(
            self._header, text=f"tool: {tool_name}",
            bg=self.colors.get("tool_bg", "#1e1e2e"),
            fg=self.colors.get("warning", "#dcdcaa"), font=("Consolas", 9, "bold"),
        ).pack(side="left")

        self._header.bind("<Button-1>", self._toggle)
        self._toggle_icon.bind("<Button-1>", self._toggle)

        # Input preview (always visible, truncated)
        input_str = ", ".join(f"{k}={v!r}" for k, v in tool_input.items())
        if len(input_str) > 80:
            input_str = input_str[:80] + "..."

        self._input_preview = tk.Label(
            self, text=f"  {input_str}",
            bg=self.colors.get("tool_bg", "#1e1e2e"),
            fg=self.colors.get("fg_dim", "#808080"),
            font=("Consolas", 8), anchor="w",
        )
        self._input_preview.pack(fill="x", padx=8)

        # Result (collapsed by default)
        self._result_frame = tk.Frame(self, bg=self.colors.get("tool_bg", "#1e1e2e"))

        self._result_text = tk.Text(
            self._result_frame,
            wrap="word",
            bg=self.colors.get("bg", "#1e1e1e"),
            fg=self.colors.get("fg", "#d4d4d4"),
            font=("Consolas", 9),
            relief="flat",
            borderwidth=0,
            padx=8, pady=4,
            height=5,
            cursor="arrow",
        )
        self._result_text.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self._result_text.configure(state="disabled")

    def _toggle(self, event=None):
        """Toggle the expanded/collapsed state."""
        self._expanded = not self._expanded
        if self._expanded:
            self._toggle_icon.configure(text="▼")
            self._result_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        else:
            self._toggle_icon.configure(text="▶")
            self._result_frame.pack_forget()

    def set_result(self, result: str, is_error: bool = False):
        """Set the tool result."""
        self._spinner_label.configure(text="✓" if not is_error else "✗")
        self._spinner_label.configure(
            fg=self.colors.get("success", "#4ec9b0") if not is_error
            else self.colors.get("error", "#f44747")
        )

        # Truncate very long results
        display_result = result
        if len(display_result) > 5000:
            display_result = display_result[:5000] + "\n... (truncated)"

        self._result_text.configure(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert("1.0", display_result)
        self._result_text.configure(state="disabled")

        if is_error:
            self._result_text.configure(fg=self.colors.get("error", "#f44747"))


class ThemedText(tk.Text):
    """A themed text widget with scrollbar and syntax highlighting."""

    def __init__(self, parent, theme: str = "dark", readonly: bool = True, **kwargs):
        self.theme = theme
        self.colors = get_theme_colors(theme)

        kwargs.setdefault("bg", self.colors.get("bg", "#1e1e1e"))
        kwargs.setdefault("fg", self.colors.get("fg", "#d4d4d4"))
        kwargs.setdefault("insertbackground", self.colors.get("fg", "#d4d4d4"))
        kwargs.setdefault("font", ("Consolas", 10))
        kwargs.setdefault("wrap", "word")
        kwargs.setdefault("relief", "flat")
        kwargs.setdefault("borderwidth", 0)
        kwargs.setdefault("padx", 8)
        kwargs.setdefault("pady", 4)

        super().__init__(parent, **kwargs)

        # Scrollbar
        self._scrollbar = tk.Scrollbar(
            self.master, command=self.yview,
            bg=self.colors.get("scrollbar", "#424242"),
            troughcolor=self.colors.get("bg", "#1e1e1e"),
        )
        self.configure(yscrollcommand=self._scrollbar.set)

        if readonly:
            self.configure(state="disabled", cursor="arrow")

        # Tags for syntax highlighting
        syntax = SYNTAX_COLORS_DARK if theme == "dark" else SYNTAX_COLORS_LIGHT
        for tag_name, color in syntax.items():
            self.tag_configure(tag_name, foreground=color)

        self.tag_configure("bold", font=("Consolas", 10, "bold"))
        self.tag_configure("italic", font=("Consolas", 10, "italic"))
        self.tag_configure("header1", font=("Segoe UI", 16, "bold"),
                          foreground=self.colors.get("accent", "#007acc"))
        self.tag_configure("header2", font=("Segoe UI", 13, "bold"),
                          foreground=self.colors.get("accent", "#007acc"))
        self.tag_configure("header3", font=("Segoe UI", 11, "bold"),
                          foreground=self.colors.get("accent", "#007acc"))
        self.tag_configure("code_block", font=("Consolas", 9),
                          background=self.colors.get("code_bg", "#1e1e1e"))

    def pack_with_scrollbar(self, **pack_kwargs):
        """Pack the text widget with its scrollbar."""
        self._scrollbar.pack(side="right", fill="y")
        self.pack(**pack_kwargs)

    def set_text(self, text: str):
        """Set text content (enables editing, sets text, disables editing)."""
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.insert("1.0", text)
        self.configure(state="disabled")

    def append_text(self, text: str):
        """Append text to the end."""
        self.configure(state="normal")
        self.insert("end", text)
        self.see("end")
        self.configure(state="disabled")

    def insert_message(self, role: str, text: str):
        """Insert a formatted message."""
        self.configure(state="normal")

        if role == "user":
            self.insert("end", f"\n👤 You:\n", "bold")
        elif role == "assistant":
            self.insert("end", f"\n🤖 Assistant:\n", "bold")
        elif role == "error":
            self.insert("end", f"\n❌ Error: ", "bold")

        self.insert("end", f"{text}\n")
        self.see("end")
        self.configure(state="disabled")

    def highlight_python(self, code: str):
        """Insert Python code with syntax highlighting."""
        self.configure(state="normal")
        start_index = self.index("end")

        lines = code.split("\n")
        for line in lines:
            # Comment
            if line.strip().startswith("#"):
                self.insert("end", line + "\n", "comment")
                continue

            # Simple tokenization
            tokens = re.findall(r'(\w+|[^\w\s]|\s+)', line)
            for token in tokens:
                if token.strip() == "":
                    self.insert("end", token)
                elif token in PYTHON_KEYWORDS:
                    self.insert("end", token, "keyword")
                elif re.match(r'^["\']', token):
                    self.insert("end", token, "string")
                elif re.match(r'^\d', token):
                    self.insert("end", token, "number")
                elif token.startswith("@"):
                    self.insert("end", token, "decorator")
                else:
                    self.insert("end", token)

        self.configure(state="disabled")
