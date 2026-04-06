"""
Sidebar component for the Cowork desktop GUI.

Includes:
- Live file tree with collapsible directories
- Task checklist with status tracking
- Quick action buttons
- File watching integration with watchdog
"""

import os
import tkinter as tk
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from gui.widgets import ThemedFrame, ThemedButton, get_theme_colors

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False


# ──────────────────────────────────────────────
# File tree item
# ──────────────────────────────────────────────

class FileTreeItem:
    """Represents a node in the file tree."""

    def __init__(self, path: Path, is_dir: bool = False, parent=None):
        self.path = path
        self.name = path.name
        self.is_dir = is_dir
        self.parent = parent
        self.children: List["FileTreeItem"] = []
        self.expanded = False
        self.widget: Optional[tk.Frame] = None
        self.children_frame: Optional[tk.Frame] = None
        self.icon_label: Optional[tk.Label] = None


# ──────────────────────────────────────────────
# File system event handler
# ──────────────────────────────────────────────

if HAS_WATCHDOG:
    class FileChangeHandler(FileSystemEventHandler):
        """Handles file system changes and triggers tree refresh."""

        def __init__(self, callback: Callable):
            self.callback = callback

        def on_any_event(self, event):
            if event.src_path.endswith((".pyc", ".pyo", "__pycache__")):
                return
            try:
                self.callback()
            except Exception:
                pass
else:
    class FileChangeHandler:
        """Fallback no-op handler when watchdog is not installed."""

        def __init__(self, callback: Callable):
            self.callback = callback

        def on_any_event(self, event):
            pass


# ──────────────────────────────────────────────
# Task item
# ──────────────────────────────────────────────

class TaskItem:
    """Represents a task in the task checklist."""

    def __init__(self, text: str, completed: bool = False):
        self.text = text
        self.completed = completed
        self.widget: Optional[tk.Frame] = None
        self.check_var: Optional[tk.BooleanVar] = None


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

class Sidebar(ThemedFrame):
    """
    Left sidebar with file tree, task checklist, and quick actions.

    Features:
    - Live file tree with collapsible directories
    - File watching for auto-refresh
    - Task checklist with status tracking
    - Quick action buttons
    """

    def __init__(self, parent, theme: str = "dark",
                 on_file_select: Callable = None,
                 on_quick_action: Callable = None,
                 on_atlas_action: Callable = None,
                 **kwargs):
        super().__init__(parent, theme=theme, **kwargs)

        self.theme = theme
        self.colors = get_theme_colors(theme)
        self.on_file_select = on_file_select
        self.on_quick_action = on_quick_action
        self.on_atlas_action = on_atlas_action
        self._atlas_enabled = False

        self._tree_root: Optional[FileTreeItem] = None
        self._tree_items: Dict[str, FileTreeItem] = {}
        self._tasks: List[TaskItem] = []
        self._observer = None
        self._watching = False

        self._width = 260
        self.configure(width=self._width)

        self._build_ui()

    def _build_ui(self):
        """Build the sidebar UI."""
        # Paned window for resizable sections
        self._paned = tk.PanedWindow(
            self, orient="vertical",
            bg=self.colors["bg"],
            sashwidth=3,
            sashrelief="flat",
            showhandle=False,
        )
        self._paned.pack(fill="both", expand=True)

        # ── File Tree Section ──
        self._build_file_tree_section()

        # ── Tasks Section ──
        self._build_tasks_section()

        # ── Quick Actions Section ──
        self._build_quick_actions_section()

        # ── Atlas Agent Section ──
        self._build_atlas_section()

    def _build_file_tree_section(self):
        """Build the file tree section."""
        file_frame = tk.Frame(self._paned, bg=self.colors["bg"])
        self._paned.add(file_frame, height=200, minsize=100)

        # Header
        header = tk.Frame(file_frame, bg=self.colors["bg_secondary"])
        header.pack(fill="x")

        tk.Label(
            header, text="  📁 Files",
            bg=self.colors["bg_secondary"],
            fg=self.colors["fg_bright"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(side="left", fill="x", pady=4)

        # Refresh button
        refresh_btn = tk.Label(
            header, text="↻", bg=self.colors["bg_secondary"],
            fg=self.colors["fg_dim"], font=("Segoe UI", 10),
            cursor="hand2",
        )
        refresh_btn.pack(side="right", padx=4)
        refresh_btn.bind("<Button-1>", lambda e: self.refresh_tree())

        # File tree scrollable area
        tree_container = tk.Frame(file_frame, bg=self.colors["bg"])
        tree_container.pack(fill="both", expand=True)

        self._tree_canvas = tk.Canvas(
            tree_container,
            bg=self.colors["bg"],
            highlightthickness=0,
            width=self._width,
        )
        tree_scrollbar = tk.Scrollbar(
            tree_container, orient="vertical",
            command=self._tree_canvas.yview,
            bg=self.colors["scrollbar"],
            troughcolor=self.colors["bg"],
        )
        self._tree_canvas.configure(yscrollcommand=tree_scrollbar.set)

        tree_scrollbar.pack(side="right", fill="y")
        self._tree_canvas.pack(side="left", fill="both", expand=True)

        self._tree_inner = tk.Frame(self._tree_canvas, bg=self.colors["bg"])
        self._tree_canvas.create_window((0, 0), window=self._tree_inner, anchor="nw")

        self._tree_inner.bind("<Configure>", self._on_tree_configure)

    def _on_tree_configure(self, event):
        """Update canvas scroll region when tree changes."""
        self._tree_canvas.configure(scrollregion=self._tree_canvas.bbox("all"))

    def _build_tasks_section(self):
        """Build the task checklist section."""
        task_frame = tk.Frame(self._paned, bg=self.colors["bg"])
        self._paned.add(task_frame, height=150, minsize=80)

        # Header
        header = tk.Frame(task_frame, bg=self.colors["bg_secondary"])
        header.pack(fill="x")

        tk.Label(
            header, text="  ✅ Tasks",
            bg=self.colors["bg_secondary"],
            fg=self.colors["fg_bright"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(side="left", fill="x", pady=4)

        # Add task button
        add_btn = tk.Label(
            header, text="+", bg=self.colors["bg_secondary"],
            fg=self.colors["fg_dim"], font=("Segoe UI", 12, "bold"),
            cursor="hand2",
        )
        add_btn.pack(side="right", padx=4)
        add_btn.bind("<Button-1>", lambda e: self.add_task("New task"))

        # Tasks scrollable area
        tasks_container = tk.Frame(task_frame, bg=self.colors["bg"])
        tasks_container.pack(fill="both", expand=True)

        self._tasks_canvas = tk.Canvas(
            tasks_container,
            bg=self.colors["bg"],
            highlightthickness=0,
            width=self._width,
        )
        tasks_scrollbar = tk.Scrollbar(
            tasks_container, orient="vertical",
            command=self._tasks_canvas.yview,
            bg=self.colors["scrollbar"],
            troughcolor=self.colors["bg"],
        )
        self._tasks_canvas.configure(yscrollcommand=tasks_scrollbar.set)

        tasks_scrollbar.pack(side="right", fill="y")
        self._tasks_canvas.pack(side="left", fill="both", expand=True)

        self._tasks_inner = tk.Frame(self._tasks_canvas, bg=self.colors["bg"])
        self._tasks_canvas.create_window((0, 0), window=self._tasks_inner, anchor="nw")

        self._tasks_inner.bind("<Configure>", self._on_tasks_configure)

    def _on_tasks_configure(self, event):
        """Update canvas scroll region when tasks change."""
        self._tasks_canvas.configure(scrollregion=self._tasks_canvas.bbox("all"))

    def _build_quick_actions_section(self):
        """Build the quick actions section with prompt actions and agent team buttons."""
        actions_frame = tk.Frame(self._paned, bg=self.colors["bg"])
        self._paned.add(actions_frame, height=200, minsize=100)

        # ── Quick Prompt Actions ──
        header = tk.Frame(actions_frame, bg=self.colors["bg_secondary"])
        header.pack(fill="x")

        tk.Label(
            header, text="  ⚡ Quick Actions",
            bg=self.colors["bg_secondary"],
            fg=self.colors["fg_bright"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(side="left", fill="x", pady=4)

        btn_frame = tk.Frame(actions_frame, bg=self.colors["bg"])
        btn_frame.pack(fill="x", padx=8, pady=4)

        actions = [
            ("🔍 Explain Code", "explain_code"),
            ("🐛 Fix Bugs", "fix_bugs"),
            ("✅ Write Tests", "write_tests"),
            ("♻️ Refactor", "refactor"),
        ]

        for i, (label, action_id) in enumerate(actions):
            row = i // 2
            col = i % 2
            btn = tk.Label(
                btn_frame, text=label,
                bg=self.colors["button_bg"],
                fg=self.colors["button_fg"],
                font=("Segoe UI", 8),
                relief="flat",
                padx=4, pady=3,
                cursor="hand2",
                anchor="w",
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            btn.bind("<Button-1>", lambda e, a=action_id: self._on_quick_action(a))
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=self.colors["button_hover"]))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=self.colors["button_bg"]))

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        # ── Agent Team Selector ──
        team_header = tk.Frame(actions_frame, bg=self.colors["bg_secondary"])
        team_header.pack(fill="x")

        tk.Label(
            team_header, text="  🤖 Agent Team",
            bg=self.colors["bg_secondary"],
            fg=self.colors["fg_bright"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(side="left", fill="x", pady=4)

        agent_frame = tk.Frame(actions_frame, bg=self.colors["bg"])
        agent_frame.pack(fill="x", padx=8, pady=4)

        agent_buttons = [
            ("🔍 Search", "search"),
            ("💻 Code Gen", "codegen"),
            ("🐛 Debug", "debug"),
            ("👀 Review", "review"),
            ("🧪 Test", "test"),
            ("♻️ Refactor", "refactor"),
            ("📝 Docs", "docs"),
            ("🔐 Security", "security"),
        ]

        for i, (label, agent_id) in enumerate(agent_buttons):
            row = i // 4
            col = i % 4
            btn = tk.Label(
                agent_frame, text=label,
                bg=self.colors["accent"],
                fg="#ffffff",
                font=("Segoe UI", 7),
                relief="flat",
                padx=3, pady=2,
                cursor="hand2",
                anchor="w",
            )
            btn.grid(row=row, column=col, padx=1, pady=1, sticky="ew")
            btn.bind("<Button-1>", lambda e, a=agent_id: self._on_quick_action(a))
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=self.colors["accent_hover"]))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=self.colors["accent"]))

        for c in range(4):
            agent_frame.columnconfigure(c, weight=1)

    def _on_quick_action(self, action_id: str):
        """Handle quick action button click."""
        if self.on_quick_action:
            self.on_quick_action(action_id)

    def _build_atlas_section(self):
        """Build the Atlas Agent section with mode toggle and sub-controls."""
        atlas_frame = tk.Frame(self._paned, bg=self.colors["bg"])
        self._paned.add(atlas_frame, height=160, minsize=100)

        # ── Header ──
        header = tk.Frame(atlas_frame, bg=self.colors["bg_secondary"])
        header.pack(fill="x")

        tk.Label(
            header, text="  🏛️ Atlas Agent",
            bg=self.colors["bg_secondary"],
            fg=self.colors["fg_bright"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(side="left", fill="x", pady=4)

        # ── Atlas Mode Toggle ──
        self._atlas_mode_var = tk.BooleanVar(value=False)
        mode_row = tk.Frame(atlas_frame, bg=self.colors["bg"])
        mode_row.pack(fill="x", padx=8, pady=(6, 2))

        self._atlas_toggle_cb = tk.Checkbutton(
            mode_row,
            text="Atlas Mode",
            variable=self._atlas_mode_var,
            bg=self.colors["bg"],
            fg=self.colors["fg"],
            selectcolor=self.colors["bg_tertiary"],
            activebackground=self.colors["bg"],
            activeforeground=self.colors["fg"],
            font=("Segoe UI", 9, "bold"),
            command=self._on_atlas_mode_toggle,
        )
        self._atlas_toggle_cb.pack(side="left", padx=2, pady=2)

        # ── Control Buttons ──
        btn_frame = tk.Frame(atlas_frame, bg=self.colors["bg"])
        btn_frame.pack(fill="x", padx=8, pady=2)

        atlas_buttons = [
            ("🧩 Skills", "atlas_skills"),
            ("⏰ Cron Jobs", "atlas_cron"),
            ("🌐 Gateway", "atlas_gateway"),
        ]

        for i, (label, action_id) in enumerate(atlas_buttons):
            row = i // 2
            col = i % 2
            btn = tk.Label(
                btn_frame, text=label,
                bg=self.colors["button_bg"],
                fg=self.colors["button_fg"],
                font=("Segoe UI", 8),
                relief="flat",
                padx=4, pady=3,
                cursor="hand2",
                anchor="w",
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            btn.bind("<Button-1>", lambda e, a=action_id: self._on_atlas_button(a))
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=self.colors["button_hover"]))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=self.colors["button_bg"]))

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        # ── Smart Routing Status Label ──
        routing_row = tk.Frame(atlas_frame, bg=self.colors["bg"])
        routing_row.pack(fill="x", padx=8, pady=(4, 2))

        tk.Label(
            routing_row, text="🔀 Smart Routing:",
            bg=self.colors["bg"],
            fg=self.colors["fg_dim"],
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(side="left")

        self._routing_status_var = tk.StringVar(value="Inactive")
        self._routing_status_label = tk.Label(
            routing_row,
            textvariable=self._routing_status_var,
            bg=self.colors["bg"],
            fg=self.colors["fg_dim"],
            font=("Segoe UI", 8, "italic"),
            anchor="w",
        )
        self._routing_status_label.pack(side="left", padx=(4, 0))

    def _on_atlas_mode_toggle(self):
        """Handle Atlas mode toggle."""
        self._atlas_enabled = self._atlas_mode_var.get()
        if self.on_atlas_action:
            self.on_atlas_action("atlas_toggle", self._atlas_enabled)

    def _on_atlas_button(self, action_id: str):
        """Handle Atlas sub-button click."""
        if self.on_atlas_action:
            self.on_atlas_action(action_id, None)

    def set_routing_status(self, status: str):
        """Update the smart routing status label."""
        self._routing_status_var.set(status)
        # Color based on status
        if status == "Active":
            self._routing_status_label.configure(fg=self.colors.get("success", "#4ec9b0"))
        elif status == "Inactive":
            self._routing_status_label.configure(fg=self.colors["fg_dim"])
        else:
            self._routing_status_label.configure(fg=self.colors.get("warning", "#dcdcaa"))

    # ── File Tree Methods ──

    def refresh_tree(self, root_path: str = None):
        """Refresh the file tree from disk."""
        path = Path(root_path) if root_path else Path.cwd()

        # Clear existing tree
        for widget in self._tree_inner.winfo_children():
            widget.destroy()
        self._tree_items.clear()

        # Build tree
        try:
            self._tree_root = self._build_tree_item(self._tree_inner, path, depth=0)
        except PermissionError:
            pass

        self._tree_inner.update_idletasks()
        self._on_tree_configure(None)

    def _build_tree_item(self, parent_frame: tk.Frame, path: Path, depth: int = 0) -> FileTreeItem:
        """Recursively build a file tree item."""
        is_dir = path.is_dir()

        item = FileTreeItem(path, is_dir=is_dir)
        self._tree_items[str(path)] = item

        # Create widget
        item.widget = tk.Frame(parent_frame, bg=self.colors["bg"], cursor="hand2")
        item.widget.pack(fill="x")

        # Icon
        icon = "📁" if is_dir else "📄"
        if path.suffix in (".py",):
            icon = "🐍"
        elif path.suffix in (".js", ".ts", ".jsx", ".tsx"):
            icon = "📜"
        elif path.suffix in (".md", ".txt", ".rst"):
            icon = "📝"
        elif path.suffix in (".json", ".yaml", ".yml", ".toml"):
            icon = "⚙️"
        elif path.suffix in (".html", ".css"):
            icon = "🌐"
        elif path.name == "requirements.txt":
            icon = "📦"

        # Expand icon for directories
        expand_text = "▼" if item.expanded else "▶" if is_dir else "  "

        icon_label = tk.Label(
            item.widget,
            text=f"{'  ' * depth}{expand_text} {icon} {path.name}",
            bg=self.colors["bg"],
            fg=self.colors["fg"] if is_dir else self.colors["fg_dim"],
            font=("Consolas", 9),
            anchor="w",
        )
        icon_label.pack(side="left", fill="x", expand=True, padx=2, pady=1)
        item.icon_label = icon_label

        # Bind click events
        icon_label.bind("<Button-1>", lambda e, i=item: self._on_tree_item_click(i))
        item.widget.bind("<Button-1>", lambda e, i=item: self._on_tree_item_click(i))

        # Double click to select file
        icon_label.bind("<Double-Button-1>", lambda e, p=path: self._on_file_double_click(p))
        item.widget.bind("<Double-Button-1>", lambda e, p=path: self._on_file_double_click(p))

        # Children frame (for directories)
        if is_dir:
            item.children_frame = tk.Frame(parent_frame, bg=self.colors["bg"])

            # Load children (limited depth)
            if depth < 4:
                try:
                    entries = sorted(
                        [e for e in path.iterdir() if not e.name.startswith(".")],
                        key=lambda x: (not x.is_dir(), x.name.lower()),
                    )
                    # Skip common non-project dirs
                    skip = {"__pycache__", "node_modules", ".git", ".venv", "venv",
                            "dist", "build", ".tox", ".mypy_cache", ".pytest_cache"}
                    entries = [e for e in entries if e.name not in skip]

                    for entry in entries[:100]:  # Limit children
                        child = self._build_tree_item(item.children_frame, entry, depth + 1)
                        item.children.append(child)
                except PermissionError:
                    pass

        return item

    def _on_tree_item_click(self, item: FileTreeItem):
        """Handle click on a tree item (expand/collapse)."""
        if item.is_dir and item.children_frame:
            item.expanded = not item.expanded
            if item.expanded:
                item.children_frame.pack(fill="x")
                if item.icon_label:
                    icon_text = item.icon_label.cget("text")
                    item.icon_label.configure(text=icon_text.replace("▶", "▼"))
            else:
                item.children_frame.pack_forget()
                if item.icon_label:
                    icon_text = item.icon_label.cget("text")
                    item.icon_label.configure(text=icon_text.replace("▼", "▶"))

    def _on_file_double_click(self, path: Path):
        """Handle double-click on a file."""
        if self.on_file_select and path.is_file():
            self.on_file_select(str(path))

    # ── Task Methods ──

    def add_task(self, text: str, completed: bool = False):
        """Add a task to the checklist."""
        task = TaskItem(text, completed)
        self._tasks.append(task)

        task.widget = tk.Frame(self._tasks_inner, bg=self.colors["bg"])
        task.widget.pack(fill="x", pady=1)

        task.check_var = tk.BooleanVar(value=completed)

        cb = tk.Checkbutton(
            task.widget,
            variable=task.check_var,
            text=text,
            bg=self.colors["bg"],
            fg=self.colors["fg"],
            selectcolor=self.colors["bg_tertiary"],
            activebackground=self.colors["bg"],
            activeforeground=self.colors["fg"],
            font=("Segoe UI", 9),
            command=lambda: self._on_task_toggle(task),
        )
        cb.pack(side="left", fill="x", padx=4, pady=2)

        return task

    def _on_task_toggle(self, task: TaskItem):
        """Handle task checkbox toggle."""
        task.completed = task.check_var.get()
        fg = self.colors["fg_dim"] if task.completed else self.colors["fg"]
        for child in task.widget.winfo_children():
            child.configure(fg=fg)

    def clear_tasks(self):
        """Clear all tasks."""
        for widget in self._tasks_inner.winfo_children():
            widget.destroy()
        self._tasks.clear()

    def complete_all_tasks(self):
        """Mark all tasks as completed."""
        for task in self._tasks:
            task.completed = True
            if task.check_var:
                task.check_var.set(True)
            for child in task.widget.winfo_children():
                child.configure(fg=self.colors["fg_dim"])

    # ── File Watching ──

    def start_watching(self, path: str = None):
        """Start watching the file system for changes."""
        if not HAS_WATCHDOG:
            return

        watch_path = path or os.getcwd()
        if self._observer:
            self.stop_watching()

        try:
            handler = FileChangeHandler(self.refresh_tree)
            self._observer = Observer()
            self._observer.schedule(handler, watch_path, recursive=False)
            self._observer.start()
            self._watching = True
        except Exception:
            self._watching = False

    def stop_watching(self):
        """Stop file watching."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
            self._watching = False

    def update_theme(self, theme: str):
        """Update the sidebar theme."""
        self.theme = theme
        self.colors = get_theme_colors(theme)
        self.configure(bg=self.colors["bg"])
        self.refresh_tree()
