"""
Cron Job Management — CRUD operations, cron expression parsing,
fixed-rate/one-time support, execution history, and self-scheduling.

Jobs are persisted as JSON in ``<data_dir>/jobs.json``.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Lifecycle status of a scheduled job."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    REMOVED = "removed"


class Job:
    """
    Represents a single scheduled job with full metadata,
    retry policy, dependency chain, and execution history.
    """

    def __init__(
        self,
        name: str,
        schedule_type: str = "recurring",
        cron_expr: Optional[str] = None,
        command: Optional[str] = None,
        priority: int = 5,
        tags: Optional[list[str]] = None,
        retry_policy: Optional[dict] = None,
        dependencies: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        timezone_str: str = "UTC",
        job_id: Optional[str] = None,
    ) -> None:
        self.id: str = job_id or str(uuid.uuid4())
        self.name: str = name
        self.schedule_type: str = schedule_type  # recurring | fixed_rate | one_time
        self.cron_expr: Optional[str] = cron_expr
        self.command: Optional[str] = command
        self.priority: int = priority  # 1=highest, 10=lowest
        self.tags: list[str] = tags or []
        self.retry_policy: dict = retry_policy or {
            "max_retries": 0,
            "retry_delay": 60,
        }
        self.dependencies: list[str] = dependencies or []
        self.metadata: dict = metadata or {}
        self.timezone_str: str = timezone_str

        self.status: JobStatus = JobStatus.ACTIVE
        self.created_at: datetime = datetime.now(timezone.utc)
        self.updated_at: datetime = datetime.now(timezone.utc)
        self.next_run: Optional[datetime] = None
        self.last_run: Optional[datetime] = None
        self.history: list[dict] = []

        # Compute initial next_run
        if self.cron_expr and self.schedule_type == "recurring":
            self.next_run = self._parse_cron_next(self.cron_expr)
        elif self.schedule_type == "fixed_rate":
            rate_seconds = self.metadata.get("rate_seconds", 60)
            self.next_run = datetime.now(timezone.utc) + timedelta(seconds=rate_seconds)
        elif self.schedule_type == "one_time":
            run_at = self.metadata.get("run_at")
            if run_at:
                self.next_run = self._parse_datetime(run_at)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of this job."""
        return {
            "id": self.id,
            "name": self.name,
            "schedule_type": self.schedule_type,
            "cron_expr": self.cron_expr,
            "command": self.command,
            "priority": self.priority,
            "tags": self.tags,
            "retry_policy": self.retry_policy,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "timezone_str": self.timezone_str,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "history": self.history[-100:],  # Keep last 100 entries
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        """Reconstruct a Job from a serialised dict."""
        job = cls(
            name=data["name"],
            schedule_type=data.get("schedule_type", "recurring"),
            cron_expr=data.get("cron_expr"),
            command=data.get("command"),
            priority=data.get("priority", 5),
            tags=data.get("tags", []),
            retry_policy=data.get("retry_policy", {}),
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {}),
            timezone_str=data.get("timezone_str", "UTC"),
            job_id=data.get("id"),
        )
        job.status = JobStatus(data.get("status", "active"))
        if data.get("created_at"):
            job.created_at = cls._parse_datetime(data["created_at"])
        if data.get("updated_at"):
            job.updated_at = cls._parse_datetime(data["updated_at"])
        if data.get("next_run"):
            job.next_run = cls._parse_datetime(data["next_run"])
        if data.get("last_run"):
            job.last_run = cls._parse_datetime(data["last_run"])
        job.history = data.get("history", [])
        return job

    # ------------------------------------------------------------------
    # Cron expression helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cron_next(expr: str, after: Optional[datetime] = None) -> datetime:
        """
        Parse a 6-field cron expression and return the next run time.

        Fields: second  minute  hour  day  month  weekday

        If ``croniter`` is available it is used for full parsing;
        otherwise a simple interval-based fallback is used.
        """
        after = after or datetime.now(timezone.utc)
        try:
            from croniter import croniter
            cron = croniter(expr, after)
            return cron.get_next(datetime)
        except Exception:
            logger.debug("croniter unavailable; falling back to interval for %r", expr)
            return after + timedelta(minutes=5)

    @staticmethod
    def validate_cron_expr(expr: str) -> bool:
        """
        Validate a 6-field cron expression format.

        Returns True if the expression matches the expected pattern
        (each field may contain digits, commas, dashes, slashes, asterisks).
        """
        fields = expr.strip().split()
        if len(fields) != 6:
            return False
        pattern = re.compile(r"^[0-9,\-*/]+$")
        return all(pattern.match(f) for f in fields)

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        """Parse an ISO-format datetime string into a timezone-aware datetime."""
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)


# =========================================================================
# Job Manager — CRUD and persistence
# =========================================================================


class JobManager:
    """
    Manages the lifecycle of scheduled jobs with JSON-file persistence.

    Provides create, list, get, remove, update, pause, resume, and
    trigger operations, as well as self-scheduling for agent reminders.
    """

    def __init__(self, data_dir: str | Path = Path.home() / ".claude_clone" / "cron") -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._jobs_path = self.data_dir / "jobs.json"
        self._jobs: dict[str, Job] = {}
        self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create_job(self, job: Job) -> Job:
        """Register a new job and persist it."""
        if job.id in self._jobs:
            raise ValueError(f"Job with id {job.id!r} already exists")
        self._jobs[job.id] = job
        self._save()
        logger.info("Created job %r (id=%s)", job.name, job.id)
        return job

    async def create_job_from_params(
        self,
        name: str,
        schedule_type: str = "recurring",
        cron_expr: Optional[str] = None,
        command: Optional[str] = None,
        priority: int = 5,
        tags: Optional[list[str]] = None,
        retry_policy: Optional[dict] = None,
        dependencies: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Job:
        """Convenience: create and register a job from keyword arguments."""
        job = Job(
            name=name,
            schedule_type=schedule_type,
            cron_expr=cron_expr,
            command=command,
            priority=priority,
            tags=tags,
            retry_policy=retry_policy,
            dependencies=dependencies,
            metadata=metadata,
        )
        return await self.create_job(job)

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        tag: Optional[str] = None,
    ) -> list[Job]:
        """Return jobs optionally filtered by status and/or tag."""
        jobs = list(self._jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        if tag is not None:
            jobs = [j for j in jobs if tag in j.tags]
        return jobs

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a single job by ID."""
        return self._jobs.get(job_id)

    async def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID. Returns True if found and removed."""
        job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        job.status = JobStatus.REMOVED
        self._save()
        logger.info("Removed job %r (id=%s)", job.name, job_id)
        return True

    async def update_job(self, job_id: str, job: Job) -> Job:
        """Update an existing job and persist changes."""
        if job_id not in self._jobs:
            raise KeyError(f"Job {job_id!r} not found")
        job.updated_at = datetime.now(timezone.utc)
        self._jobs[job_id] = job
        self._save()
        logger.info("Updated job %r (id=%s)", job.name, job_id)
        return job

    # ------------------------------------------------------------------
    # Pause / Resume / Trigger
    # ------------------------------------------------------------------

    async def pause_job(self, job_id: str) -> bool:
        """Pause an active job. Returns True on success."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.ACTIVE:
            return False
        job.status = JobStatus.PAUSED
        self._save()
        logger.info("Paused job %r", job.name)
        return True

    async def resume_job(self, job_id: str) -> bool:
        """Resume a paused job. Returns True on success."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.PAUSED:
            return False
        job.status = JobStatus.ACTIVE
        if job.cron_expr and job.schedule_type == "recurring":
            job.next_run = job._parse_cron_next(job.cron_expr)
        self._save()
        logger.info("Resumed job %r", job.name)
        return True

    async def trigger_job(self, job_id: str) -> Optional[Job]:
        """
        Immediately trigger a job by setting ``next_run`` to now.

        The scheduler will pick it up on the next tick.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        job.status = JobStatus.ACTIVE
        job.next_run = datetime.now(timezone.utc)
        self._save()
        logger.info("Triggered job %r for immediate execution", job.name)
        return job

    # ------------------------------------------------------------------
    # Self-scheduling (agent creates its own reminders)
    # ------------------------------------------------------------------

    async def schedule_reminder(
        self,
        message: str,
        run_at: str,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Job:
        """
        Schedule a one-time reminder job. This is the primary entry
        point for the agent's self-scheduling capability.

        param message: — The reminder text to surface to the agent.
        param run_at:  — ISO 8601 datetime when the reminder fires.
        param tags:    — Optional tags for categorisation.
        """
        meta = metadata or {}
        meta["reminder_message"] = message
        meta["run_at"] = run_at
        job = Job(
            name=f"Reminder: {message[:80]}",
            schedule_type="one_time",
            priority=1,
            tags=["reminder"] + (tags or []),
            metadata=meta,
        )
        return await self.create_job(job)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup_completed(self, max_age_days: int = 30) -> int:
        """
        Remove completed/failed jobs older than *max_age_days*.

        Returns the number of jobs removed.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        to_remove: list[str] = []
        for job_id, job in self._jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                if job.updated_at and job.updated_at < cutoff:
                    to_remove.append(job_id)
        for job_id in to_remove:
            del self._jobs[job_id]
        if to_remove:
            self._save()
            logger.info("Cleaned up %d old completed/failed jobs", len(to_remove))
        return len(to_remove)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load jobs from the JSON file."""
        if not self._jobs_path.exists():
            return
        try:
            data = json.loads(self._jobs_path.read_text(encoding="utf-8"))
            for job_data in data.get("jobs", []):
                job = Job.from_dict(job_data)
                self._jobs[job.id] = job
            logger.info("Loaded %d job(s) from %s", len(self._jobs), self._jobs_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load jobs: %s", exc)

    def _save(self) -> None:
        """Persist all jobs to the JSON file."""
        data = {
            "version": 1,
            "jobs": [j.to_dict() for j in self._jobs.values()],
        }
        try:
            self._jobs_path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to save jobs: %s", exc)
