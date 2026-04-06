"""
Cron Scheduler Engine.

Timezone-aware scheduling with file-locked execution to prevent
duplicate runs, priority-based ordering, and missed-job catch-up.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .jobs import Job, JobManager, JobStatus

logger = logging.getLogger(__name__)

# Default data directory for cron state
_DEFAULT_DATA_DIR = Path.home() / ".claude_clone" / "cron"
_LOCK_FILE = ".cron_scheduler.lock"


class CronScheduler:
    """
    Main cron scheduler engine.

    The ``tick()`` method should be called at a regular interval
    (default 60 seconds) by an external loop or ``asyncio.create_task``.
    A file lock prevents multiple scheduler instances from running
    simultaneously.
    """

    def __init__(
        self,
        job_manager: Optional[JobManager] = None,
        data_dir: str | Path = _DEFAULT_DATA_DIR,
        tick_interval: float = 60.0,
        default_timezone: str = "UTC",
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tick_interval = tick_interval
        self.default_tz = self._resolve_timezone(default_timezone)
        self._job_manager = job_manager or JobManager(data_dir=self.data_dir)
        self._lock_path = self.data_dir / _LOCK_FILE
        self._lock_fd: int | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._handlers: dict[str, Callable[..., Awaitable[Any]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the scheduler loop. Idempotent."""
        if self._running:
            return
        if not self._acquire_lock():
            logger.warning("Another scheduler instance is already running; aborting start.")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Cron scheduler started (interval=%.1fs, tz=%s)",
            self.tick_interval,
            self.default_tz,
        )

    async def stop(self) -> None:
        """Gracefully stop the scheduler loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._release_lock()
        logger.info("Cron scheduler stopped")

    async def tick(self) -> list[dict]:
        """
        Run one scheduler tick — evaluate every active job and execute
        those whose next_run has arrived.

        Returns a list of execution result dicts.
        """
        now = datetime.now(self.default_tz)
        jobs = await self._job_manager.list_jobs(status=JobStatus.ACTIVE)
        results: list[dict] = []

        # Sort by priority (lower number = higher priority) then by next_run
        jobs.sort(key=lambda j: (j.priority, j.next_run or datetime.max.replace(tzinfo=self.default_tz)))

        executed_ids: set[str] = set()

        for job in jobs:
            if job.id in executed_ids:
                continue

            # Check dependency chain — only proceed if all deps are satisfied
            if not self._deps_satisfied(job, executed_ids):
                continue

            # Check if job is due
            if job.next_run and job.next_run <= now:
                result = await self._execute_job(job, now)
                results.append(result)
                executed_ids.add(job.id)

        return results

    def register_handler(self, job_id: str, handler: Callable[..., Awaitable[Any]]) -> None:
        """Register an async callback for a specific job ``job_id``."""
        self._handlers[job_id] = handler

    # ------------------------------------------------------------------
    # Internal: main loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.tick()
            except Exception:
                logger.exception("Error in cron scheduler tick")
            await asyncio.sleep(self.tick_interval)

    # ------------------------------------------------------------------
    # Internal: job execution
    # ------------------------------------------------------------------

    async def _execute_job(self, job: Job, now: datetime) -> dict:
        """
        Execute a single job, record the result, and reschedule.

        Handles missed-job catch-up and retry policies.
        """
        start = datetime.now(self.default_tz)
        result: dict = {
            "job_id": job.id,
            "name": job.name,
            "started_at": start.isoformat(),
            "status": "success",
        }

        try:
            handler = self._handlers.get(job.id)
            if handler is not None:
                output = await handler(job.metadata.get("params", {}))
                result["output"] = output
            elif job.metadata.get("command"):
                output = await self._run_command(job.metadata["command"])
                result["output"] = output
            else:
                result["output"] = f"Job {job.name!r} executed (no handler or command)."

        except Exception as exc:
            logger.exception("Job %r execution failed", job.name)
            result["status"] = "error"
            result["error"] = str(exc)

            # Retry logic
            retry_count = job.metadata.get("retry_count", 0)
            max_retries = job.retry_policy.get("max_retries", 0)
            if retry_count < max_retries:
                job.metadata["retry_count"] = retry_count + 1
                delay = job.retry_policy.get("retry_delay", 60) * (retry_count + 1)
                job.next_run = now + timedelta(seconds=delay)
                await self._job_manager.update_job(job.id, job)
                result["retry_scheduled"] = job.next_run.isoformat()
                logger.info("Job %r retry #%d scheduled at %s", job.name, retry_count + 1, job.next_run)
                return result

        # Record execution history
        end = datetime.now(self.default_tz)
        result["duration_ms"] = (end - start).total_seconds() * 1000
        result["finished_at"] = end.isoformat()
        job.history.append(result)

        # Reschedule or complete
        if job.schedule_type == "recurring":
            job.next_run = self._compute_next_run(job, now)
            job.metadata["retry_count"] = 0
            await self._job_manager.update_job(job.id, job)
        elif job.schedule_type == "fixed_rate":
            rate_seconds = job.metadata.get("rate_seconds", 60)
            job.next_run = now + timedelta(seconds=rate_seconds)
            job.metadata["retry_count"] = 0
            await self._job_manager.update_job(job.id, job)
        else:
            # One-time job — mark completed
            job.status = JobStatus.COMPLETED
            job.next_run = None
            await self._job_manager.update_job(job.id, job)

        logger.info("Job %r executed in %.1fms [%s]", job.name, result["duration_ms"], result["status"])
        return result

    def _compute_next_run(self, job: Job, after: datetime) -> datetime:
        """
        Compute next run time based on cron expression.

        Supports 6-field cron expressions: second minute hour day month weekday.
        Falls back to simple interval increment if parsing fails.
        """
        if not job.cron_expr:
            return after + timedelta(seconds=60)

        try:
            from croniter import croniter
            cron = croniter(job.cron_expr, after)
            return cron.get_next(datetime)
        except Exception:
            logger.debug("croniter not available or invalid expression, using interval fallback")
            return after + timedelta(minutes=5)

    def _deps_satisfied(self, job: Job, executed: set[str]) -> bool:
        """Return True if all dependency jobs have been executed this tick."""
        for dep_id in job.dependencies:
            if dep_id not in executed:
                # Check if the dependency job has been completed historically
                dep_jobs = self._job_manager._jobs.get(dep_id)
                if dep_jobs and dep_jobs.status != JobStatus.COMPLETED:
                    return False
        return True

    # ------------------------------------------------------------------
    # Internal: command execution
    # ------------------------------------------------------------------

    @staticmethod
    async def _run_command(command: str) -> str:
        """Run a shell command and return its output."""
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        output = stdout.decode(errors="replace")
        if stderr:
            output += "\n" + stderr.decode(errors="replace")
        return output.strip()

    # ------------------------------------------------------------------
    # Internal: file locking
    # ------------------------------------------------------------------

    def _acquire_lock(self) -> bool:
        """Acquire an exclusive file lock. Returns False if already held."""
        try:
            self._lock_fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, BlockingIOError):
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            return False

    def _release_lock(self) -> None:
        """Release the file lock."""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None

    # ------------------------------------------------------------------
    # Internal: timezone
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_timezone(tz_name: str) -> timezone:
        """Resolve a timezone name string to a ``datetime.tzinfo``."""
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(tz_name)
        except Exception:
            return timezone.utc
