"""
Cron job management CLI for Hermes CLI.

List, create, edit, pause, resume, and delete cron jobs.
View job execution history and logs.
"""

import json
import os
import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes.cli_hermes.config_manager import CRON_DIR, ConfigManager


CRON_STATE_FILE = CRON_DIR / "cron_state.json"


class CronJob:
    """Represents a scheduled cron job."""

    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id", "")
        self.name = data.get("name", "")
        self.description = data.get("description", "")
        self.schedule = data.get("schedule", "")
        self.schedule_human = data.get("schedule_human", "")
        self.command = data.get("command", "")
        self.prompt = data.get("prompt", "")
        self.enabled = data.get("enabled", True)
        self.status = data.get("status", "idle")
        self.created_at = data.get("created_at", "")
        self.last_run = data.get("last_run")
        self.next_run = data.get("next_run")
        self.run_count = data.get("run_count", 0)
        self.error_count = data.get("error_count", 0)
        self.last_error = data.get("last_error")
        self.tags = data.get("tags", [])
        self.max_retries = data.get("max_retries", 3)
        self.timeout = data.get("timeout", 300)
        self._data = data

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


class CronManager:
    """Manages cron jobs for the Hermes CLI."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.cron_dir = CRON_DIR
        self.cron_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Optional[List[CronJob]] = None

    def _load_jobs(self) -> List[CronJob]:
        """Load all cron jobs from state file."""
        if self._jobs is not None:
            return self._jobs

        if CRON_STATE_FILE.exists():
            try:
                with open(CRON_STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._jobs = [CronJob(j) for j in data.get("jobs", [])]
            except Exception:
                self._jobs = []
        else:
            self._jobs = []

        return self._jobs

    def _save_jobs(self):
        """Save all cron jobs to state file."""
        CRON_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "jobs": [j.to_dict() for j in self._jobs],
            "updated_at": datetime.now().isoformat(),
        }
        with open(CRON_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def list_jobs(self, status_filter: Optional[str] = None) -> List[CronJob]:
        """List all cron jobs."""
        jobs = self._load_jobs()

        if status_filter:
            if status_filter == "active":
                jobs = [j for j in jobs if j.enabled and j.status != "error"]
            elif status_filter == "paused":
                jobs = [j for j in jobs if not j.enabled]
            elif status_filter == "error":
                jobs = [j for j in jobs if j.status == "error"]
            elif status_filter == "running":
                jobs = [j for j in jobs if j.status == "running"]

        return jobs

    def create_job(
        self,
        name: str,
        schedule: str,
        command: Optional[str] = None,
        prompt: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        timeout: int = 300,
        max_retries: int = 3,
    ) -> CronJob:
        """Create a new cron job."""
        self._load_jobs()

        # Validate schedule format (cron expression or natural language)
        if not self._validate_schedule(schedule):
            # Try to convert from natural language
            schedule_human = schedule
            schedule = self._natural_to_cron(schedule)
            if not schedule:
                raise ValueError(f"Invalid schedule: {schedule_human}")

        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        job_data = {
            "id": job_id,
            "name": name,
            "description": description or f"Cron job: {name}",
            "schedule": schedule,
            "schedule_human": schedule_human if 'schedule_human' in dir() else schedule,
            "command": command or "",
            "prompt": prompt or "",
            "enabled": True,
            "status": "idle",
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "next_run": None,
            "run_count": 0,
            "error_count": 0,
            "tags": tags or [],
            "timeout": timeout,
            "max_retries": max_retries,
        }

        job = CronJob(job_data)
        self._jobs.append(job)
        self._save_jobs()
        return job

    def edit_job(self, job_id: str, **kwargs) -> Optional[CronJob]:
        """Edit an existing cron job."""
        self._load_jobs()

        for job in self._jobs:
            if job.id == job_id:
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                        job._data[key] = value
                self._save_jobs()
                return job

        return None

    def pause_job(self, job_id: str) -> bool:
        """Pause a cron job."""
        return self.edit_job(job_id, enabled=False) is not None

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused cron job."""
        return self.edit_job(job_id, enabled=True) is not None

    def delete_job(self, job_id: str) -> bool:
        """Delete a cron job."""
        self._load_jobs()

        for i, job in enumerate(self._jobs):
            if job.id == job_id:
                self._jobs.pop(i)
                self._save_jobs()
                return True

        return False

    def get_job(self, job_id: str) -> Optional[CronJob]:
        """Get a specific cron job."""
        for job in self._load_jobs():
            if job.id == job_id:
                return job
        return None

    def get_job_logs(self, job_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get execution logs for a job."""
        log_file = self.cron_dir / f"{job_id}_logs.json"
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    return json.load(f)[-limit:]
            except Exception:
                pass
        return []

    def get_execution_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent execution history across all jobs."""
        history = []
        for job in self._load_jobs():
            if job.last_run:
                history.append({
                    "job_id": job.id,
                    "job_name": job.name,
                    "last_run": job.last_run,
                    "run_count": job.run_count,
                    "error_count": job.error_count,
                    "status": job.status,
                })

        history.sort(key=lambda x: x.get("last_run", ""), reverse=True)
        return history[:limit]

    def create_from_natural_language(self, description: str) -> Optional[CronJob]:
        """Create a job from natural language description."""
        # Parse common patterns
        patterns = [
            (r'every\s+(\d+)\s+(minute|minutes|min)', lambda m: f"*/{m.group(1)} * * * *"),
            (r'every\s+(\d+)\s+(hour|hours|hr)', lambda m: f"0 */{m.group(1)} * * *"),
            (r'every\s+day\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', self._parse_daily),
            (r'weekly\s+on\s+(\w+)', self._parse_weekly),
            (r'monthly', lambda m: "0 0 1 * *"),
            (r'daily', lambda m: "0 0 * * *"),
            (r'hourly', lambda m: "0 * * * *"),
        ]

        schedule = None
        name = description

        for pattern, parser in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                schedule = parser(match)
                break

        if not schedule:
            return None

        return self.create_job(
            name=f"Cron: {description[:50]}",
            schedule=schedule,
            prompt=description,
            description=description,
        )

    def _validate_schedule(self, schedule: str) -> bool:
        """Validate a cron expression."""
        parts = schedule.split()
        if len(parts) != 5:
            return False
        for part in parts:
            if part in ("*",):
                continue
            if re.match(r'^[\d,\-*/]+$', part):
                continue
            return False
        return True

    def _natural_to_cron(self, text: str) -> Optional[str]:
        """Convert natural language to cron expression."""
        text = text.lower().strip()

        if text in ("hourly", "every hour"):
            return "0 * * * *"
        elif text in ("daily", "every day", "midnight"):
            return "0 0 * * *"
        elif text in ("weekly", "every week"):
            return "0 0 * * 0"
        elif text in ("monthly", "every month"):
            return "0 0 1 * *"

        # "every N minutes/hours"
        m = re.match(r'every\s+(\d+)\s+(min|minute|hour|hr)', text)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            if unit in ("min", "minute"):
                return f"*/{n} * * * *"
            else:
                return f"0 */{n} * * *"

        return None

    def _parse_daily(self, match):
        """Parse daily schedule."""
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        ampm = match.group(3)

        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0

        return f"{minute} {hour} * * *"

    def _parse_weekly(self, match):
        """Parse weekly schedule."""
        day_map = {
            "monday": 1, "tuesday": 2, "wednesday": 3,
            "thursday": 4, "friday": 5, "saturday": 6, "sunday": 0,
        }
        day = day_map.get(match.group(1).lower(), 0)
        return f"0 0 * * {day}"

    def format_jobs_table(self, jobs: Optional[List[CronJob]] = None) -> str:
        """Format cron jobs as a table."""
        jobs = jobs or self.list_jobs()

        if not jobs:
            return "  No cron jobs configured. Use /cron add to create one."

        lines = []
        lines.append(f"  {'Name':<25} {'Schedule':<20} {'Status':<10} {'Runs':>6} {'Errors':>6} {'Last Run'}")
        lines.append("  " + "-" * 90)

        for job in jobs:
            status = "\033[32mactive\033[0m" if job.enabled else "\033[33mpaused\033[0m"
            if job.status == "error":
                status = "\033[31merror\033[0m"
            elif job.status == "running":
                status = "\033[36mrunning\033[0m"

            last = "never"
            if job.last_run:
                try:
                    dt = datetime.fromisoformat(job.last_run)
                    last = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    last = job.last_run[:16]

            lines.append(
                f"  {job.name:<25} {job.schedule:<20} "
                f"{status:<18} {job.run_count:>6} {job.error_count:>6} {last}"
            )

        return "\n".join(lines)
