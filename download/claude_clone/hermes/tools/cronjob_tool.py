"""
Hermes Cron Job Tool — scheduled task management.

Features:
- Create, list, remove, pause, resume cron jobs
- Natural language scheduling
- Job execution logging
- Delivery to connected platforms
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_CRON_DB_PATH = Path.home() / ".claude_clone" / "hermes_cron.json"


def _load_cron_db() -> Dict[str, Any]:
    if _CRON_DB_PATH.exists():
        try:
            return json.loads(_CRON_DB_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            pass
    return {"jobs": {}, "logs": []}


def _save_cron_db(data: Dict[str, Any]) -> None:
    _CRON_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Keep logs manageable
    if len(data.get("logs", [])) > 1000:
        data["logs"] = data["logs"][-500:]
    _CRON_DB_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gen_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Natural language schedule parsing
# ---------------------------------------------------------------------------

def _parse_natural_schedule(text: str) -> Dict[str, Any]:
    """Parse natural language into cron-like schedule."""
    text_lower = text.lower().strip()

    schedule = {
        "type": "interval",
        "interval_seconds": 0,
        "cron_expression": "",
        "natural": text,
    }

    # "every N minutes/hours/days"
    m = re.match(r"every\s+(\d+)\s*(min(?:ute)?s?|hours?|days?)", text_lower)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("min"):
            schedule["interval_seconds"] = n * 60
        elif unit.startswith("hour"):
            schedule["interval_seconds"] = n * 3600
        elif unit.startswith("day"):
            schedule["interval_seconds"] = n * 86400
        return schedule

    # "every day at HH:MM"
    m = re.match(r"every\s+day\s+at\s+(\d{1,2}):(\d{2})", text_lower)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        schedule["type"] = "daily"
        schedule["cron_expression"] = f"{minute} {hour} * * *"
        return schedule

    # "every weekday at HH:MM"
    m = re.match(r"every\s+weekday\s+at\s+(\d{1,2}):(\d{2})", text_lower)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        schedule["type"] = "weekday"
        schedule["cron_expression"] = f"{minute} {hour} * * 1-5"
        return schedule

    # "every Monday/Wednesday at HH:MM"
    day_map = {
        "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
        "friday": 5, "saturday": 6, "sunday": 0,
    }
    for day_name, day_num in day_map.items():
        m = re.match(rf"every\s+{day_name}\s+at\s+(\d{{1,2}}):(\d{{2}})", text_lower)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))
            schedule["type"] = "weekly"
            schedule["cron_expression"] = f"{minute} {hour} * * {day_num}"
            return schedule

    # Direct cron expression (5 fields)
    parts = text_lower.split()
    if len(parts) == 5 and all(re.match(r"[\d\*,\-/]+", p) for p in parts):
        schedule["type"] = "cron"
        schedule["cron_expression"] = " ".join(parts)
        return schedule

    # Fallback: treat as one-time
    schedule["type"] = "once"
    schedule["run_at"] = text

    return schedule


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def hermes_cron_create(
    name: str,
    command: str,
    schedule: str,
) -> str:
    """Create a new cron job.

    param name (str): — Unique job name.
    param command (str): — Command or task to execute.
    param schedule (str): — Schedule (natural language or cron expression).
    """
    def _do():
        db = _load_cron_db()

        if name in db["jobs"]:
            return f"Error: Job '{name}' already exists. Remove it first or use a different name."

        parsed = _parse_natural_schedule(schedule)

        job = {
            "name": name,
            "command": command,
            "schedule_raw": schedule,
            "schedule": parsed,
            "enabled": True,
            "created_at": _now(),
            "last_run": None,
            "run_count": 0,
            "next_run": None,
        }

        db["jobs"][name] = job
        _save_cron_db(db)

        sched_desc = parsed.get("cron_expression") or f"{parsed.get('interval_seconds', 0)}s interval"
        return f"Created cron job '{name}': {command}\nSchedule: {schedule} ({sched_desc})"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error creating cron job: {e}"


async def hermes_cron_list() -> str:
    """List all cron jobs with their status."""
    def _do():
        db = _load_cron_db()
        jobs = db.get("jobs", {})

        if not jobs:
            return "No cron jobs configured."

        lines = [f"Cron Jobs ({len(jobs)}):\n"]
        for name, job in sorted(jobs.items()):
            status = "enabled" if job.get("enabled") else "PAUSED"
            sched = job.get("schedule_raw", "?")
            last = job.get("last_run", "never")
            count = job.get("run_count", 0)
            lines.append(
                f"  [{status}] {name}: {job.get('command', '')[:60]}\n"
                f"    Schedule: {sched} | Runs: {count} | Last: {last}"
            )

        return "\n".join(lines)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error listing cron jobs: {e}"


async def hermes_cron_remove(name: str) -> str:
    """Remove a cron job.

    param name (str): — Job name to remove.
    """
    def _do():
        db = _load_cron_db()
        if name not in db["jobs"]:
            return f"Error: Job '{name}' not found."

        del db["jobs"][name]
        _save_cron_db(db)
        return f"Removed cron job '{name}'"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error removing cron job: {e}"


async def hermes_cron_pause(name: str) -> str:
    """Pause a cron job.

    param name (str): — Job name to pause.
    """
    def _do():
        db = _load_cron_db()
        if name not in db["jobs"]:
            return f"Error: Job '{name}' not found."

        db["jobs"][name]["enabled"] = False
        _save_cron_db(db)
        return f"Paused cron job '{name}'"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error pausing cron job: {e}"


async def hermes_cron_resume(name: str) -> str:
    """Resume a paused cron job.

    param name (str): — Job name to resume.
    """
    def _do():
        db = _load_cron_db()
        if name not in db["jobs"]:
            return f"Error: Job '{name}' not found."

        db["jobs"][name]["enabled"] = True
        _save_cron_db(db)
        return f"Resumed cron job '{name}'"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error resuming cron job: {e}"


async def hermes_cron_logs(name: str = "", limit: int = 20) -> str:
    """View cron job execution logs.

    param name (str): — Filter by job name. Empty = all.
    param limit (int): — Max log entries. Default: 20.
    """
    def _do():
        db = _load_cron_db()
        logs = db.get("logs", [])

        if name:
            logs = [l for l in logs if l.get("job") == name]

        if not logs:
            return "No execution logs found."

        logs = logs[-limit:]

        lines = [f"Cron execution logs ({len(logs)} most recent):\n"]
        for log in reversed(logs):
            status = log.get("status", "?")
            icon = "+" if status == "success" else "!"
            lines.append(
                f"  [{icon}] {log.get('timestamp', '')[:16]} "
                f"{log.get('job', '?')}: {status} — {str(log.get('output', ''))[:80]}"
            )

        return "\n".join(lines)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error reading cron logs: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="hermes_cron_create",
    func=hermes_cron_create,
    description="Create a new cron job with natural language or cron expression scheduling.",
    toolset="automation",
)

ToolRegistry.instance().register(
    name="hermes_cron_list",
    func=hermes_cron_list,
    description="List all cron jobs with their status and run history.",
    toolset="automation",
)

ToolRegistry.instance().register(
    name="hermes_cron_remove",
    func=hermes_cron_remove,
    description="Remove a cron job by name.",
    toolset="automation",
)

ToolRegistry.instance().register(
    name="hermes_cron_pause",
    func=hermes_cron_pause,
    description="Pause a running cron job.",
    toolset="automation",
)

ToolRegistry.instance().register(
    name="hermes_cron_resume",
    func=hermes_cron_resume,
    description="Resume a paused cron job.",
    toolset="automation",
)

ToolRegistry.instance().register(
    name="hermes_cron_logs",
    func=hermes_cron_logs,
    description="View cron job execution logs.",
    toolset="automation",
)
