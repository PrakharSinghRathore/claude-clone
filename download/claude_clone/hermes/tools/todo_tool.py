"""
Hermes Todo Tool — task management with priorities, due dates, and subtasks.

Features:
- Create, update, delete, list todos
- Priority and due date support
- Progress tracking
- Subtask support
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_TODO_DB_PATH = Path.home() / ".claude_clone" / "hermes_todos.json"


def _load_db() -> Dict[str, Any]:
    if _TODO_DB_PATH.exists():
        try:
            return json.loads(_TODO_DB_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            pass
    return {"todos": {}, "lists": {"default": []}}


def _save_db(data: Dict[str, Any]) -> None:
    _TODO_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TODO_DB_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gen_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def hermes_todo_create(
    title: str,
    description: str = "",
    priority: str = "medium",
    due_date: str = "",
    tags: str = "",
    list_name: str = "default",
) -> str:
    """Create a new todo item.

    param title (str): — Todo title.
    param description (str): — Detailed description.
    param priority (str): — Priority: low, medium, high, critical. Default: medium.
    param due_date (str): — Due date (ISO format or relative like 'tomorrow', '3d').
    param tags (str): — Comma-separated tags.
    param list_name (str): — Todo list name. Default: default.
    """
    # Parse due date
    if due_date:
        due = _parse_relative_date(due_date)
        if due:
            due_date = due
    else:
        due_date = ""

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    priority = priority.lower()
    if priority not in ("low", "medium", "high", "critical"):
        priority = "medium"

    def _do():
        db = _load_db()
        todo_id = _gen_id()

        todo = {
            "id": todo_id,
            "title": title,
            "description": description,
            "priority": priority,
            "due_date": due_date,
            "tags": tag_list,
            "list": list_name,
            "status": "pending",
            "progress": 0,
            "subtasks": [],
            "created_at": _now(),
            "updated_at": _now(),
            "completed_at": None,
        }

        db["todos"][todo_id] = todo

        if list_name not in db["lists"]:
            db["lists"][list_name] = []
        db["lists"][list_name].append(todo_id)

        _save_db(db)

        due_str = f" (due: {due_date})" if due_date else ""
        return f"Created todo {todo_id}: {title}{due_str} [{priority}]"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error creating todo: {e}"


def _parse_relative_date(text: str) -> Optional[str]:
    """Parse relative date strings."""
    text_lower = text.strip().lower()

    if text_lower == "today":
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    elif text_lower == "tomorrow":
        return (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    elif text_lower == "next week":
        return (datetime.now(timezone.utc) + timedelta(weeks=1)).strftime("%Y-%m-%d")

    m = __import__("re").match(r"(\d+)(d|days?)", text_lower)
    if m:
        n = int(m.group(1))
        return (datetime.now(timezone.utc) + timedelta(days=n)).strftime("%Y-%m-%d")

    # Try ISO format
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
        return text
    except ValueError:
        return None


async def hermes_todo_update(
    todo_id: str,
    title: str = "",
    status: str = "",
    progress: int = -1,
    priority: str = "",
) -> str:
    """Update an existing todo item.

    param todo_id (str): — Todo ID to update.
    param title (str): — New title (empty = keep current).
    param status (str): — New status: pending, in_progress, done, cancelled.
    param progress (int): — Progress percentage 0-100. -1 = keep current.
    param priority (str): — New priority.
    """
    def _do():
        db = _load_db()
        todo = db["todos"].get(todo_id)
        if not todo:
            return f"Error: Todo {todo_id} not found."

        changed = []

        if title:
            todo["title"] = title
            changed.append("title")

        if status and status in ("pending", "in_progress", "done", "cancelled"):
            todo["status"] = status
            todo["updated_at"] = _now()
            if status == "done":
                todo["completed_at"] = _now()
                todo["progress"] = 100
            changed.append("status")

        if progress >= 0:
            todo["progress"] = max(0, min(100, progress))
            changed.append("progress")

        if priority and priority in ("low", "medium", "high", "critical"):
            todo["priority"] = priority
            changed.append("priority")

        if not changed:
            return f"No changes made to todo {todo_id}."

        _save_db(db)
        return f"Updated todo {todo_id}: {', '.join(changed)}"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error updating todo: {e}"


async def hermes_todo_delete(todo_id: str) -> str:
    """Delete a todo item.

    param todo_id (str): — Todo ID to delete.
    """
    def _do():
        db = _load_db()
        if todo_id not in db["todos"]:
            return f"Error: Todo {todo_id} not found."

        todo = db["todos"][todo_id]
        list_name = todo.get("list", "default")

        # Remove from list
        if list_name in db["lists"] and todo_id in db["lists"][list_name]:
            db["lists"][list_name].remove(todo_id)

        del db["todos"][todo_id]
        _save_db(db)
        return f"Deleted todo {todo_id}: {todo.get('title', '')}"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error deleting todo: {e}"


async def hermes_todo_list(list_name: str = "", status: str = "", show_all: bool = False) -> str:
    """List todo items with optional filtering.

    param list_name (str): — Filter by list name. Empty = all.
    param status (str): — Filter by status.
    param show_all (bool): — Show completed items. Default: False.
    """
    def _do():
        db = _load_db()
        todos = list(db["todos"].values())

        if list_name:
            list_ids = set(db["lists"].get(list_name, []))
            todos = [t for t in todos if t["id"] in list_ids]

        if status:
            todos = [t for t in todos if t.get("status") == status]
        elif not show_all:
            todos = [t for t in todos if t.get("status") != "done"]

        if not todos:
            return "No todos found."

        # Sort by priority (critical first), then due date
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        todos.sort(key=lambda t: (
            priority_order.get(t.get("priority", "medium"), 2),
            t.get("due_date") or "9999-99-99",
        ))

        lines = [f"Todos ({len(todos)} items):\n"]
        for t in todos:
            status_icon = {
                "pending": "o",
                "in_progress": "~",
                "done": "+",
                "cancelled": "x",
            }.get(t.get("status", "pending"), "?")

            priority = t.get("priority", "medium")
            progress = t.get("progress", 0)
            due = f" (due: {t['due_date']})" if t.get("due_date") else ""

            line = f"  [{status_icon}] {t['id']}: {t['title']} [{priority}]{due}"
            if progress > 0 and progress < 100:
                line += f" ({progress}%)"

            subtasks = t.get("subtasks", [])
            if subtasks:
                done_sub = sum(1 for s in subtasks if s.get("done"))
                line += f" [{done_sub}/{len(subtasks)} subtasks]"

            lines.append(line)

        return "\n".join(lines)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error listing todos: {e}"


async def hermes_todo_add_subtask(todo_id: str, title: str) -> str:
    """Add a subtask to a todo item.

    param todo_id (str): — Parent todo ID.
    param title (str): — Subtask title.
    """
    def _do():
        db = _load_db()
        todo = db["todos"].get(todo_id)
        if not todo:
            return f"Error: Todo {todo_id} not found."

        subtask = {
            "id": _gen_id(),
            "title": title,
            "done": False,
            "created_at": _now(),
        }

        todo.setdefault("subtasks", []).append(subtask)
        todo["updated_at"] = _now()
        _save_db(db)

        total = len(todo["subtasks"])
        return f"Added subtask '{title}' to todo {todo_id} (total: {total} subtasks)"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error adding subtask: {e}"


async def hermes_todo_complete_subtask(todo_id: str, subtask_id: str) -> str:
    """Mark a subtask as completed.

    param todo_id (str): — Parent todo ID.
    param subtask_id (str): — Subtask ID to complete.
    """
    def _do():
        db = _load_db()
        todo = db["todos"].get(todo_id)
        if not todo:
            return f"Error: Todo {todo_id} not found."

        for subtask in todo.get("subtasks", []):
            if subtask["id"] == subtask_id:
                subtask["done"] = True
                todo["updated_at"] = _now()

                # Auto-calculate progress
                all_subs = todo.get("subtasks", [])
                if all_subs:
                    done_count = sum(1 for s in all_subs if s.get("done"))
                    todo["progress"] = int((done_count / len(all_subs)) * 100)

                _save_db(db)
                return f"Completed subtask in todo {todo_id} (progress: {todo['progress']}%)"

        return f"Error: Subtask {subtask_id} not found in todo {todo_id}"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error completing subtask: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="hermes_todo_create",
    func=hermes_todo_create,
    description="Create a new todo item with priority, due date, and tags.",
    toolset="productivity",
)

ToolRegistry.instance().register(
    name="hermes_todo_update",
    func=hermes_todo_update,
    description="Update an existing todo (status, progress, priority, title).",
    toolset="productivity",
)

ToolRegistry.instance().register(
    name="hermes_todo_delete",
    func=hermes_todo_delete,
    description="Delete a todo item by ID.",
    toolset="productivity",
)

ToolRegistry.instance().register(
    name="hermes_todo_list",
    func=hermes_todo_list,
    description="List todo items with optional filtering by list, status, and priority.",
    toolset="productivity",
)

ToolRegistry.instance().register(
    name="hermes_todo_add_subtask",
    func=hermes_todo_add_subtask,
    description="Add a subtask to a todo item.",
    toolset="productivity",
)

ToolRegistry.instance().register(
    name="hermes_todo_complete_subtask",
    func=hermes_todo_complete_subtask,
    description="Mark a subtask as completed and auto-update parent progress.",
    toolset="productivity",
)
