"""
Cowork Desktop GUI — A full desktop application using tkinter.

Features:
- Chat area with streamed markdown responses
- Left sidebar: live file tree (watchdog), task checklist, quick actions
- Tool call cards: collapsible, show input/output
- Settings dialog: API key, model, theme (dark/light)
- Save/load conversations to JSON
- Export conversation to Markdown
- Drag and drop files onto chat
- Resizable panels
- System tray icon (best effort)
"""

import asyncio
import json
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from gui.widgets import (
    ThemedFrame, ThemedButton, ThemedEntry, ChatMessage, ToolCallCard,
    ThemedText, get_theme_colors, THEMES,
)
from gui.sidebar import Sidebar

from agent.core import Agent, AgentEvent, TextEvent, ToolCallEvent, ToolResultEvent, ErrorEvent, DoneEvent, ThinkingEvent, UsageEvent
from agent.tools import TOOLS_REGISTRY
from config import Config


# ──────────────────────────────────────────────
# Settings Dialog
# ──────────────────────────────────────────────

class SettingsDialog(tk.Toplevel):
    """Settings dialog for API key, model, and theme configuration."""

    def __init__(self, parent, config: Config, theme: str = "dark", on_save: Callable = None):
        super().__init__(parent)
        self.config = config
        self.theme = theme
        self.colors = get_theme_colors(theme)
        self.on_save = on_save
        self.result = None

        self.title("Settings")
        self.configure(bg=self.colors["bg"])
        self.geometry("450x400")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self._center_on_parent(parent)

    def _center_on_parent(self, parent):
        """Center the dialog on the parent window."""
        self.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_x()
        py = parent.winfo_y()
        w = self.winfo_width()
        h = self.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"+{x}+{y}")

    def _build_ui(self):
        """Build the settings UI."""
        padding = {"padx": 16, "pady": 6}

        # Title
        tk.Label(
            self, text="⚙️ Settings", bg=self.colors["bg"],
            fg=self.colors["fg_bright"], font=("Segoe UI", 14, "bold"),
        ).pack(**padding, anchor="w")

        sep = tk.Frame(self, bg=self.colors["border"], height=1)
        sep.pack(fill="x", padx=16)

        # API Key
        tk.Label(
            self, text="API Key", bg=self.colors["bg"],
            fg=self.colors["fg"], font=("Segoe UI", 10),
        ).pack(**padding, anchor="w")

        self._api_key_var = tk.StringVar(value=self.config.api_key[:8] + "..." if self.config.api_key else "")
        api_entry = tk.Entry(
            self, textvariable=self._api_key_var,
            bg=self.colors["input_bg"], fg=self.colors["input_fg"],
            insertbackground=self.colors["fg"],
            relief="flat", font=("Consolas", 10),
            show="•",
        )
        api_entry.pack(padx=16, fill="x", ipady=4)

        # Model
        tk.Label(
            self, text="Model", bg=self.colors["bg"],
            fg=self.colors["fg"], font=("Segoe UI", 10),
        ).pack(**padding, anchor="w")

        models = [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
        ]

        self._model_var = tk.StringVar(value=self.config.model)
        model_combo = ttk.Combobox(
            self, textvariable=self._model_var, values=models,
            state="readonly",
        )
        model_combo.pack(padx=16, fill="x", ipady=3)

        # Theme
        tk.Label(
            self, text="Theme", bg=self.colors["bg"],
            fg=self.colors["fg"], font=("Segoe UI", 10),
        ).pack(**padding, anchor="w")

        theme_frame = tk.Frame(self, bg=self.colors["bg"])
        theme_frame.pack(padx=16, anchor="w")

        self._theme_var = tk.StringVar(value=self.theme)
        for theme_name in ["dark", "light"]:
            rb = tk.Radiobutton(
                theme_frame, text=theme_name.capitalize(),
                variable=self._theme_var, value=theme_name,
                bg=self.colors["bg"], fg=self.colors["fg"],
                selectcolor=self.colors["bg_tertiary"],
                activebackground=self.colors["bg"],
                activeforeground=self.colors["fg"],
                font=("Segoe UI", 10),
            )
            rb.pack(side="left", padx=(0, 16))

        # Max iterations
        tk.Label(
            self, text="Max Iterations", bg=self.colors["bg"],
            fg=self.colors["fg"], font=("Segoe UI", 10),
        ).pack(**padding, anchor="w")

        self._iterations_var = tk.StringVar(value=str(self.config.max_iterations))
        iter_entry = tk.Entry(
            self, textvariable=self._iterations_var,
            bg=self.colors["input_bg"], fg=self.colors["input_fg"],
            insertbackground=self.colors["fg"],
            relief="flat", font=("Consolas", 10), width=10,
        )
        iter_entry.pack(padx=16, anchor="w")

        # Buttons
        btn_frame = tk.Frame(self, bg=self.colors["bg"])
        btn_frame.pack(fill="x", padx=16, pady=16, side="bottom")

        ThemedButton(
            btn_frame, text="Cancel", command=self.destroy,
            theme=self.theme, style="default",
        ).pack(side="right", padx=(8, 0))

        ThemedButton(
            btn_frame, text="Save", command=self._save,
            theme=self.theme, style="primary",
        ).pack(side="right")

    def _save(self):
        """Save settings."""
        self.result = {
            "model": self._model_var.get(),
            "theme": self._theme_var.get(),
            "max_iterations": int(self._iterations_var.get()),
            "api_key": self._api_key_var.get(),
        }
        if self.on_save:
            self.on_save(self.result)
        self.destroy()


# ──────────────────────────────────────────────
# Main GUI Application
# ──────────────────────────────────────────────

class CoworkApp:
    """
    Main Cowork desktop GUI application.

    Usage:
        app = CoworkApp(config)
        app.run()
    """

    def __init__(self, config: Config = None):
        self.config = config or Config.from_env()
        self.theme = self.config.theme or "dark"
        self.colors = get_theme_colors(self.theme)

        self._root = None
        self._agent: Optional[Agent] = None
        self._sidebar: Optional[Sidebar] = None
        self._chat_container = None
        self._chat_scrollbar = None
        self._chat_canvas = None
        self._input_frame = None
        self._input_entry = None
        self._send_button = None
        self._model_selector = None
        self._messages: List[Dict] = []
        self._message_widgets: List[tk.Widget] = []
        self._current_assistant_widget = None
        self._is_generating = False
        self._status_var = None
        self._cost_var = None
        self._cancel_event = threading.Event()

        # Conversations
        self._conversations_dir = Path.home() / ".claude_clone" / "conversations"
        self._conversations_dir.mkdir(parents=True, exist_ok=True)

    def _init_agent(self):
        """Initialize the agent."""
        self._agent = Agent(
            api_key=self.config.api_key,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            max_iterations=self.config.max_iterations,
            temperature=self.config.temperature,
            tools=self.config.get_effective_tools(TOOLS_REGISTRY),
        )

    def _build_root(self):
        """Build the root window."""
        self._root = tk.Tk()
        self._root.title("Claude Clone — Cowork")
        self._root.geometry("1100x700")
        self._root.minsize(800, 500)

        # Set window background
        self._root.configure(bg=self.colors["bg"])

        # Set icon (best effort)
        try:
            icon_path = Path(__file__).parent / "icon.png"
            if icon_path.exists():
                self._root.iconphoto(False, tk.PhotoImage(file=str(icon_path)))
        except Exception:
            pass

        # Protocol handlers
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Drag and drop setup
        try:
            self._root.drop_target_register("DND_Files")
            self._root.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass  # TkinterDnD not available

    def _build_ui(self):
        """Build the complete UI."""
        self._build_menu_bar()
        self._build_header()
        self._build_body()
        self._build_status_bar()

    def _build_menu_bar(self):
        """Build the menu bar."""
        menubar = tk.Menu(self._root, bg=self.colors["bg_secondary"],
                         fg=self.colors["fg"], activebackground=self.colors["accent"],
                         activeforeground="#ffffff", relief="flat")

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0,
                           bg=self.colors["bg_secondary"], fg=self.colors["fg"],
                           activebackground=self.colors["accent"], activeforeground="#ffffff")
        file_menu.add_command(label="New Conversation", command=self._new_conversation)
        file_menu.add_command(label="Save Conversation", command=self._save_conversation)
        file_menu.add_command(label="Load Conversation", command=self._load_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="Export to Markdown", command=self._export_markdown)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0,
                           bg=self.colors["bg_secondary"], fg=self.colors["fg"],
                           activebackground=self.colors["accent"], activeforeground="#ffffff")
        edit_menu.add_command(label="Copy Last Response", command=self._copy_last_response)
        edit_menu.add_command(label="Clear Chat", command=self._clear_chat)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0,
                           bg=self.colors["bg_secondary"], fg=self.colors["fg"],
                           activebackground=self.colors["accent"], activeforeground="#ffffff")
        view_menu.add_command(label="Toggle Sidebar", command=self._toggle_sidebar)
        view_menu.add_command(label="Toggle Theme", command=self._toggle_theme)
        menubar.add_cascade(label="View", menu=view_menu)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0,
                            bg=self.colors["bg_secondary"], fg=self.colors["fg"],
                            activebackground=self.colors["accent"], activeforeground="#ffffff")
        tools_menu.add_command(label="Settings", command=self._show_settings)
        tools_menu.add_command(label="Environment Info", command=self._show_env_info)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0,
                           bg=self.colors["bg_secondary"], fg=self.colors["fg"],
                           activebackground=self.colors["accent"], activeforeground="#ffffff")
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Documentation", command=lambda: webbrowser.open("https://docs.anthropic.com"))
        menubar.add_cascade(label="Help", menu=help_menu)

        self._root.config(menu=menubar)

    def _build_header(self):
        """Build the header bar."""
        header = tk.Frame(self._root, bg=self.colors["header_bg"], height=40)
        header.pack(fill="x")
        header.pack_propagate(False)

        # App title
        tk.Label(
            header, text="  Claude Clone",
            bg=self.colors["header_bg"],
            fg=self.colors["header_fg"],
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left", padx=8)

        # Model selector (right side)
        self._model_var = tk.StringVar(value=self.config.model)

        model_label = tk.Label(
            header, text="Model:",
            bg=self.colors["header_bg"], fg=self.colors["header_fg"],
            font=("Segoe UI", 9),
        )
        model_label.pack(side="right", padx=(8, 2))

        model_combo = ttk.Combobox(
            header, textvariable=self._model_var,
            values=["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-5-haiku-20241022"],
            width=28, state="readonly",
        )
        model_combo.pack(side="right", padx=(0, 8))
        model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

    def _build_body(self):
        """Build the main body with sidebar and chat area."""
        body = tk.PanedWindow(
            self._root, orient="horizontal",
            bg=self.colors["bg"],
            sashwidth=4,
            sashrelief="flat",
            showhandle=False,
        )
        body.pack(fill="both", expand=True)

        # ── Sidebar ──
        self._sidebar = Sidebar(
            body, theme=self.theme,
            on_file_select=self._on_file_select,
            on_quick_action=self._on_quick_action,
            bg=self.colors["sidebar_bg"],
        )
        body.add(self._sidebar, width=260, minsize=200)

        # ── Main Chat Area ──
        chat_frame = tk.Frame(body, bg=self.colors["bg"])
        body.add(chat_frame)

        # Chat area (scrollable)
        chat_outer = tk.Frame(chat_frame, bg=self.colors["bg"])
        chat_outer.pack(fill="both", expand=True)

        self._chat_canvas = tk.Canvas(
            chat_outer,
            bg=self.colors["bg"],
            highlightthickness=0,
        )

        self._chat_scrollbar = tk.Scrollbar(
            chat_outer, orient="vertical",
            command=self._chat_canvas.yview,
            bg=self.colors["scrollbar"],
            troughcolor=self.colors["bg"],
        )
        self._chat_canvas.configure(yscrollcommand=self._chat_scrollbar.set)

        self._chat_scrollbar.pack(side="right", fill="y")
        self._chat_canvas.pack(side="left", fill="both", expand=True)

        self._chat_container = tk.Frame(self._chat_canvas, bg=self.colors["bg"])
        self._chat_canvas.create_window((0, 0), window=self._chat_container, anchor="nw")

        self._chat_container.bind("<Configure>", self._on_chat_configure)

        # Mouse wheel scrolling
        self._chat_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._chat_container.bind("<MouseWheel>", self._on_mousewheel)

        # ── Input Area ──
        self._build_input_area(chat_frame)

        # Show welcome message
        self._show_welcome()

        # Start file watching
        self._sidebar.refresh_tree()
        self._sidebar.start_watching()

    def _build_input_area(self, parent):
        """Build the input area at the bottom."""
        self._input_frame = tk.Frame(parent, bg=self.colors["bg_secondary"], height=60)
        self._input_frame.pack(fill="x", side="bottom")
        self._input_frame.pack_propagate(False)

        # Inner frame
        inner = tk.Frame(self._input_frame, bg=self.colors["bg_secondary"])
        inner.pack(fill="x", padx=12, pady=10)

        # Input field
        self._input_entry = tk.Text(
            inner,
            height=2,
            bg=self.colors["input_bg"],
            fg=self.colors["input_fg"],
            insertbackground=self.colors["fg"],
            font=("Segoe UI", 11),
            relief="flat",
            borderwidth=2,
            wrap="word",
            padx=10, pady=6,
        )
        self._input_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        # Highlight border
        self._input_entry.configure(
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )

        # Bind Enter to send, Shift+Enter for newline
        self._input_entry.bind("<Return>", self._on_return_key)
        self._input_entry.bind("<Control-Return>", self._on_shift_return)
        self._input_entry.bind("<Shift-Return>", self._on_shift_return)

        # Send button
        self._send_button = ThemedButton(
            inner, text="Send", command=self._send_message,
            theme=self.theme, style="primary",
        )
        self._send_button.pack(side="right")

        # Cancel button (hidden by default)
        self._cancel_button = ThemedButton(
            inner, text="Cancel", command=self._cancel_generation,
            theme=self.theme, style="danger",
        )
        # Don't pack yet

    def _build_status_bar(self):
        """Build the status bar at the bottom."""
        status_bar = tk.Frame(self._root, bg=self.colors["bg_tertiary"], height=24)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)

        self._status_var = tk.StringVar(value="Ready")
        self._cost_var = tk.StringVar(value="")

        tk.Label(
            status_bar, textvariable=self._status_var,
            bg=self.colors["bg_tertiary"], fg=self.colors["fg_dim"],
            font=("Segoe UI", 8), anchor="w",
        ).pack(side="left", padx=8)

        tk.Label(
            status_bar, textvariable=self._cost_var,
            bg=self.colors["bg_tertiary"], fg=self.colors["fg_dim"],
            font=("Segoe UI", 8), anchor="e",
        ).pack(side="right", padx=8)

    # ── Event Handlers ──

    def _on_return_key(self, event):
        """Handle Enter key — send message."""
        self._send_message()
        return "break"

    def _on_shift_return(self, event):
        """Handle Shift+Enter — insert newline."""
        self._input_entry.insert("insert", "\n")
        return "break"

    def _on_model_change(self, event=None):
        """Handle model selection change."""
        self.config.model = self._model_var.get()
        if self._agent:
            self._agent.model = self.config.model

    def _on_file_select(self, path: str):
        """Handle file selection from sidebar."""
        if self._agent:
            self._agent.add_context(path)
        self._set_status(f"Added {path} to context")

    def _on_quick_action(self, action_id: str):
        """Handle quick action button click."""
        prompts = {
            "explain_code": "Please explain the code in the current project. Focus on the main architecture and key components.",
            "fix_bugs": "Please analyze the code for potential bugs, issues, or problems. List each issue with a suggested fix.",
            "write_tests": "Please write comprehensive unit tests for the code in this project. Cover the main functions and edge cases.",
            "refactor": "Please analyze the code and suggest improvements for refactoring. Focus on readability, maintainability, and performance.",
        }
        prompt = prompts.get(action_id, "")
        if prompt:
            self._input_entry.delete("1.0", "end")
            self._input_entry.insert("1.0", prompt)
            self._input_entry.focus_set()

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self._chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_chat_configure(self, event):
        """Update canvas scroll region when chat area changes."""
        self._chat_canvas.configure(scrollregion=self._chat_canvas.bbox("all"))
        # Auto-scroll to bottom
        self._chat_canvas.yview_moveto(1.0)

    def _on_drop(self, event):
        """Handle file drag and drop."""
        files = self._root.tk.splitlist(event.data)
        for file_path in files:
            file_path = file_path.strip("{}")
            if os.path.isfile(file_path):
                if self._agent:
                    self._agent.add_context(file_path)
                self._set_status(f"Added {file_path} to context")

    def _on_close(self):
        """Handle window close."""
        self._sidebar.stop_watching()
        if self._root:
            self._root.destroy()

    # ── Message Handling ──

    def _show_welcome(self):
        """Show welcome message in chat."""
        welcome = tk.Frame(self._chat_container, bg=self.colors["bg"])
        welcome.pack(fill="x", padx=16, pady=20)

        tk.Label(
            welcome, text="Claude Clone",
            bg=self.colors["bg"], fg=self.colors["accent"],
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w")

        tk.Label(
            welcome,
            text="I'm your agentic coding assistant. I can read, write, and edit files,\n"
                 "run commands, search code, and much more.\n\n"
                 "Try typing a message or use the quick actions on the left.",
            bg=self.colors["bg"], fg=self.colors["fg_dim"],
            font=("Segoe UI", 11),
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        self._message_widgets.append(welcome)

    def _add_user_message(self, text: str):
        """Add a user message to the chat."""
        msg = ChatMessage(self._chat_container, role="user", theme=self.theme)
        msg.add_content(text)
        msg.pack(fill="x", padx=8, pady=4)
        self._message_widgets.append(msg)

        self._messages.append({"role": "user", "content": text})
        self._scroll_to_bottom()

    def _create_assistant_message(self):
        """Create an empty assistant message widget for streaming."""
        msg = ChatMessage(self._chat_container, role="assistant", theme=self.theme)
        msg.pack(fill="x", padx=8, pady=4)
        self._message_widgets.append(msg)
        self._current_assistant_widget = msg
        return msg

    def _add_tool_card(self, tool_name: str, tool_input: dict):
        """Add a tool call card to the chat."""
        card = ToolCallCard(self._chat_container, tool_name, tool_input, theme=self.theme)
        card.pack(fill="x", padx=16, pady=2)
        self._message_widgets.append(card)
        self._scroll_to_bottom()
        return card

    def _add_error_message(self, text: str):
        """Add an error message to the chat."""
        msg = ChatMessage(self._chat_container, role="system", theme=self.theme)
        msg.add_content(f"❌ {text}")
        msg.pack(fill="x", padx=8, pady=4)
        self._message_widgets.append(msg)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        """Scroll chat to bottom."""
        self._root.after(50, lambda: self._chat_canvas.yview_moveto(1.0))

    # ── Agent Communication ──

    def _send_message(self):
        """Send the user's message to the agent."""
        if self._is_generating:
            return

        text = self._input_entry.get("1.0", "end").strip()
        if not text:
            return

        if not self.config.api_key:
            self._add_error_message("No API key configured. Go to Tools → Settings to set your API key.")
            return

        self._input_entry.delete("1.0", "end")
        self._add_user_message(text)

        # Start generation in a thread
        self._is_generating = True
        self._cancel_event.clear()
        self._set_status("Generating...")

        # Swap buttons
        self._send_button.pack_forget()
        self._cancel_button.pack(side="right")

        thread = threading.Thread(target=self._run_agent, args=(text,), daemon=True)
        thread.start()

    def _run_agent(self, user_message: str):
        """Run the agent in a background thread."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            self._create_assistant_message()

            async def process():
                async for event in self._agent.run(user_message):
                    if self._cancel_event.is_set():
                        self._agent.cancel()
                        break

                    # Schedule UI updates on the main thread
                    self._root.after(0, self._handle_event, event)

            loop.run_until_complete(process())
            loop.close()

        except Exception as e:
            self._root.after(0, self._add_error_message, f"Agent error: {e}")

        finally:
            self._is_generating = False
            self._root.after(0, self._on_generation_done)

    def _handle_event(self, event: AgentEvent):
        """Handle an agent event (called on the main thread)."""
        if isinstance(event, TextEvent):
            if self._current_assistant_widget:
                self._current_assistant_widget.add_content(event.data)
                self._scroll_to_bottom()

        elif isinstance(event, ThinkingEvent):
            pass  # Thinking events are not shown in GUI by default

        elif isinstance(event, ToolCallEvent):
            self._add_tool_card(event.tool_name, event.tool_input)

        elif isinstance(event, ToolResultEvent):
            # Find the last tool card and set its result
            for widget in reversed(self._message_widgets):
                if isinstance(widget, ToolCallCard):
                    widget.set_result(event.result, event.is_error)
                    self._scroll_to_bottom()
                    break

        elif isinstance(event, ErrorEvent):
            self._add_error_message(event.data)

        elif isinstance(event, DoneEvent):
            if event.usage:
                total = event.usage.get("input_tokens", 0) + event.usage.get("output_tokens", 0)
                cost = self._agent.estimate_cost()
                self._cost_var.set(f"Tokens: {total:,} | Cost: ${cost:.4f}")
            self._set_status("Ready")

    def _on_generation_done(self):
        """Called when generation is complete."""
        self._cancel_button.pack_forget()
        self._send_button.pack(side="right")
        self._current_assistant_widget = None
        self._set_status("Ready")

    def _cancel_generation(self):
        """Cancel the current generation."""
        self._cancel_event.set()
        self._set_status("Cancelling...")

    # ── Utility Methods ──

    def _set_status(self, text: str):
        """Set the status bar text."""
        if self._status_var:
            self._status_var.set(text)

    def _clear_chat(self):
        """Clear the chat area."""
        for widget in self._chat_container.winfo_children():
            widget.destroy()
        self._message_widgets.clear()
        self._messages.clear()
        if self._agent:
            self._agent.reset()
        self._show_welcome()

    def _new_conversation(self):
        """Start a new conversation."""
        if self._messages:
            if not messagebox.askyesno("New Conversation", "Start a new conversation? Current conversation will be lost unless saved."):
                return
        self._clear_chat()
        self._cost_var.set("")

    def _save_conversation(self):
        """Save the current conversation to JSON."""
        if not self._messages:
            messagebox.showinfo("Save", "No conversation to save.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(self._conversations_dir),
        )
        if not path:
            return

        data = {
            "messages": self._messages,
            "model": self.config.model,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            self._set_status(f"Conversation saved to {path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save: {e}")

    def _load_conversation(self):
        """Load a conversation from JSON."""
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(self._conversations_dir),
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._clear_chat()
            messages = data.get("messages", [])

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    self._add_user_message(content)
                elif role == "assistant":
                    widget = self._create_assistant_message()
                    widget.add_content(content if isinstance(content, str) else str(content))

            self._set_status(f"Loaded conversation from {path}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load: {e}")

    def _export_markdown(self):
        """Export conversation to Markdown."""
        if not self._agent:
            messagebox.showinfo("Export", "No conversation to export.")
            return

        markdown = self._agent.export_conversation()
        path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
            initialdir=str(self._conversations_dir),
        )
        if not path:
            return

        try:
            Path(path).write_text(markdown, encoding="utf-8")
            self._set_status(f"Exported to {path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")

    def _copy_last_response(self):
        """Copy the last assistant response to clipboard."""
        if self._agent and self._agent.messages:
            for msg in reversed(self._agent.messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Extract text from content blocks
                        texts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                texts.append(block["text"])
                        content = "\n".join(texts)
                    self._root.clipboard_clear()
                    self._root.clipboard_append(str(content))
                    self._set_status("Copied last response to clipboard")
                    return
        self._set_status("No response to copy")

    def _toggle_sidebar(self):
        """Toggle the sidebar visibility."""
        if self._sidebar.winfo_ismapped():
            self._sidebar.pack_forget()
        else:
            self._sidebar.pack(side="left", fill="y", before=self._chat_canvas.master)

    def _toggle_theme(self):
        """Toggle between dark and light themes."""
        self.theme = "light" if self.theme == "dark" else "dark"
        self.colors = get_theme_colors(self.theme)
        self.config.theme = self.theme
        self.config.save()
        messagebox.showinfo("Theme", f"Theme changed to {self.theme}. Restart the app to apply fully.")

    def _show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self._root, self.config, self.theme, on_save=self._apply_settings)
        self._root.wait_window(dialog)

    def _apply_settings(self, settings: dict):
        """Apply settings from the settings dialog."""
        self.config.model = settings.get("model", self.config.model)
        self.config.theme = settings.get("theme", self.theme)
        self.config.max_iterations = settings.get("max_iterations", self.config.max_iterations)

        if settings.get("api_key") and settings["api_key"] != "..." + str(self.config.api_key[:5] if self.config.api_key else ""):
            self.config.api_key = settings["api_key"]

        self.config.save()

        if self._agent:
            self._agent.model = self.config.model
            self._agent.max_iterations = self.config.max_iterations

        self._model_var.set(self.config.model)

        self._set_status("Settings saved")

    def _show_env_info(self):
        """Show environment information in a dialog."""
        import platform
        info = (
            f"Python: {sys.version}\n"
            f"OS: {platform.system()} {platform.release()}\n"
            f"CWD: {os.getcwd()}\n"
            f"Model: {self.config.model}\n"
            f"Theme: {self.theme}\n"
            f"API Key: {'Set' if self.config.api_key else 'Not set'}\n"
        )
        messagebox.showinfo("Environment", info)

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About Claude Clone",
            "Claude Clone v1.0.0\n\n"
            "A Python clone of Claude Code and Cowork.\n"
            "Agentic coding assistant with full tool support.\n\n"
            "Built with Python, tkinter, and Anthropic SDK."
        )

    # ── Run ──

    def run(self):
        """Run the GUI application."""
        self._init_agent()
        self._build_root()
        self._build_ui()

        # Check for API key
        if not self.config.api_key:
            self._root.after(100, lambda: messagebox.showwarning(
                "No API Key",
                "No Anthropic API key found.\n\n"
                "Set it in:\n"
                "1. Tools → Settings\n"
                "2. Environment variable: ANTHROPIC_API_KEY\n"
                "3. Config file: ~/.claude_clone/config.json\n\n"
                "Get your key at: https://console.anthropic.com/"
            ))

        self._root.mainloop()
