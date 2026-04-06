"""
Task Manager — Background task lifecycle management.

Manages the full lifecycle of background tasks including submission,
cancellation, status tracking, waiting, and automatic cleanup. Uses a
priority queue for scheduling and the task executor for running tasks.

Usage::

    from atlas.tasks.manager import TaskManager, TaskPriority

    mgr = TaskManager(max_concurrent=5)
    task_id = await mgr.submit(my_func, args=(1,), priority=TaskPriority.HIGH)
    status = mgr.get_status(task_id)
    result = await mgr.wait(task_id, timeout=30.0)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Sequence

from .executor import (
    ExecutionResult,
    ExecutionStatus,
    RetryPolicy,
    TaskDefinition,
    TaskExecutor,
)
from .queue import PriorityTaskQueue, TaskPriority

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Internal Task Record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskRecord:
    """
    Internal record tracking a submitted task's lifecycle.

    Attributes
    ----------
    task_id:
        Unique task identifier.
    name:
        Human-readable task name.
    status:
        Current execution status.
    priority:
        Task priority level.
    submitted_at:
        ISO 8601 timestamp of submission.
    started_at:
        ISO 8601 timestamp of execution start.
    finished_at:
        ISO 8601 timestamp of execution end.
    result:
        The execution result (when completed).
    future:
        The asyncio future for the task.
    retry_policy:
        Retry configuration.
    timeout:
        Maximum execution time in seconds.
    """

    task_id: str = ""
    name: str = ""
    status: str = ExecutionStatus.PENDING.value
    priority: int = TaskPriority.MEDIUM
    submitted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: str = ""
    finished_at: str = ""
    result: Optional[ExecutionResult] = None
    future: Optional[asyncio.Future] = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    timeout: float = 0.0

    @property
    def status_enum(self) -> ExecutionStatus:
        """Return the status as an ExecutionStatus enum."""
        try:
            return ExecutionStatus(self.status)
        except ValueError:
            return ExecutionStatus.PENDING

    @property
    def is_terminal(self) -> bool:
        """Whether the task is in a terminal state."""
        return self.status_enum in (
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.TIMEOUT,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status,
            "priority": TaskPriority(self.priority).name,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self._duration,
            "error": self.result.error if self.result else None,
            "attempts": self.result.attempts if self.result else 0,
        }

    @property
    def _duration(self) -> float:
        """Calculate duration in seconds."""
        if not self.started_at:
            return 0.0
        try:
            start = datetime.fromisoformat(self.started_at)
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            end_str = self.finished_at or datetime.now(timezone.utc).isoformat()
            end = datetime.fromisoformat(end_str)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            return (end - start).total_seconds()
        except (ValueError, TypeError):
            return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Task Manager
# ──────────────────────────────────────────────────────────────────────────────

class TaskManager:
    """
    Background task lifecycle manager.

    Manages task submission, scheduling via priority queue, execution
    via the task executor, status tracking, and result retrieval.
    Supports waiting for individual tasks or all tasks, cancellation,
    and automatic cleanup of old results.

    Parameters
    ----------
    max_concurrent:
        Maximum number of tasks executing simultaneously.
    default_timeout:
        Default timeout in seconds for tasks without explicit timeout.
    max_retries:
        Default maximum retry attempts.
    result_ttl:
        Time-to-live for completed task results in seconds.
    task_name_prefix:
        Prefix for auto-generated task names.

    Example
    -------
    >>> import asyncio
    >>> async def demo():
    ...     mgr = TaskManager(max_concurrent=3)
    ...     await mgr.start()
    ...     tid = await mgr.submit(print, args=("Hello",), name="greet")
    ...     result = await mgr.wait(tid)
    ...     await mgr.stop()
    >>> asyncio.run(demo())  # doctest: +SKIP
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        default_timeout: float = 300.0,
        max_retries: int = 2,
        result_ttl: float = 3600.0,
        task_name_prefix: str = "task",
    ) -> None:
        self.max_concurrent = max_concurrent
        self._default_timeout = default_timeout
        self._max_retries = max_retries
        self._result_ttl = result_ttl
        self._task_name_prefix = task_name_prefix

        # Components
        self._queue = PriorityTaskQueue(max_size=max_concurrent * 10)
        self._executor = TaskExecutor(
            default_timeout=default_timeout,
            default_retry_policy=RetryPolicy(max_retries=max_retries),
        )

        # Task registry
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()

        # Background scheduler
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        self._active_count = 0
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Cleanup
        self._cleanup_interval = 300.0
        self._cleanup_task: Optional[asyncio.Task] = None

        # Counters
        self._task_counter = 0

        logger.info(
            "TaskManager initialized (max_concurrent=%d, timeout=%.0fs, retries=%d)",
            max_concurrent,
            default_timeout,
            max_retries,
        )

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background task scheduler."""
        if self._running:
            return
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("TaskManager started")

    async def stop(self, *, cancel_pending: bool = True) -> None:
        """
        Stop the task manager.

        Parameters
        ----------
        cancel_pending:
            If ``True``, cancel all pending tasks.
        """
        self._running = False

        if cancel_pending:
            await self.cancel_all()

        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("TaskManager stopped")

    # ── Task Submission ────────────────────────────────────────────────

    async def submit(
        self,
        func: Callable[..., Awaitable[Any]],
        *,
        args: Optional[tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
        callback: Optional[Callable[[ExecutionResult], Awaitable[None]]] = None,
    ) -> str:
        """
        Submit a task for execution.

        The task is placed in the priority queue and will be executed
        when a worker slot becomes available.

        Parameters
        ----------
        func:
            The async function to execute.
        args:
            Positional arguments.
        kwargs:
            Keyword arguments.
        priority:
            Task priority level.
        name:
            Human-readable task name.
        timeout:
            Maximum execution time in seconds.
        retry_policy:
            Retry configuration.
        callback:
            Optional callback invoked on completion.

        Returns
        -------
        str
            The task identifier.
        """
        async with self._lock:
            self._task_counter += 1
            task_id = uuid.uuid4().hex[:12]
            task_name = name or f"{self._task_name_prefix}-{self._task_counter}"

            # Create task record
            record = TaskRecord(
                task_id=task_id,
                name=task_name,
                priority=priority,
                timeout=timeout or self._default_timeout,
                retry_policy=retry_policy or RetryPolicy(max_retries=self._max_retries),
                future=asyncio.get_event_loop().create_future(),
            )
            self._tasks[task_id] = record

        # Create task definition
        task_def = TaskDefinition(
            task_id=task_id,
            func=func,
            args=args or (),
            kwargs=kwargs or {},
            priority=priority,
            name=task_name,
            timeout=timeout or self._default_timeout,
            retry_policy=retry_policy or RetryPolicy(max_retries=self._max_retries),
            callback=callback,
        )

        # Enqueue
        await self._queue.put(task_def, priority=priority, task_id=task_id)

        logger.info(
            "Task submitted: %s (%s, priority=%s)",
            task_id, task_name, priority.name,
        )
        return task_id

    # ── Task Query ─────────────────────────────────────────────────────

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.

        Parameters
        ----------
        task_id:
            The task identifier.

        Returns
        -------
        dict or None
            Task status information, or ``None`` if not found.
        """
        record = self._tasks.get(task_id)
        if record is None:
            return None
        return record.to_dict()

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[TaskPriority] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List all tracked tasks with optional filters.

        Parameters
        ----------
        status:
            Filter by execution status.
        priority:
            Filter by priority level.
        limit:
            Maximum number of tasks to return.

        Returns
        -------
        list[dict]
            Task status dictionaries.
        """
        tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]
        if priority is not None:
            tasks = [t for t in tasks if t.priority == priority]

        # Sort by submission time (newest first)
        tasks.sort(key=lambda t: t.submitted_at, reverse=True)
        return [t.to_dict() for t in tasks[:limit]]

    def get_result(self, task_id: str) -> Optional[ExecutionResult]:
        """
        Get the execution result of a completed task.

        Parameters
        ----------
        task_id:
            The task identifier.

        Returns
        -------
        ExecutionResult or None
        """
        record = self._tasks.get(task_id)
        if record and record.result:
            return record.result
        return None

    # ── Waiting ────────────────────────────────────────────────────────

    async def wait(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[ExecutionResult]:
        """
        Wait for a task to complete.

        Parameters
        ----------
        task_id:
            The task identifier.
        timeout:
            Maximum wait time in seconds. ``None`` means wait indefinitely.

        Returns
        -------
        ExecutionResult or None
            The result, or ``None`` if the task was not found or timed out.
        """
        record = self._tasks.get(task_id)
        if record is None:
            logger.warning("Cannot wait for unknown task %s", task_id)
            return None

        if record.is_terminal:
            return record.result

        if record.future is None:
            return None

        try:
            await asyncio.wait_for(
                asyncio.shield(record.future),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Wait timed out for task %s", task_id)
            return None
        except asyncio.CancelledError:
            logger.info("Wait cancelled for task %s", task_id)
            return None

        return record.result

    async def wait_all(
        self,
        task_ids: Optional[Sequence[str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Optional[ExecutionResult]]:
        """
        Wait for multiple tasks to complete.

        Parameters
        ----------
        task_ids:
            Specific task IDs to wait for. If ``None``, waits for all
            non-terminal tasks.
        timeout:
            Maximum total wait time in seconds.

        Returns
        -------
        dict
            Mapping of task ID to execution result.
        """
        if task_ids is None:
            task_ids = [
                tid for tid, record in self._tasks.items()
                if not record.is_terminal
            ]

        if not task_ids:
            return {}

        tasks_to_wait = [
            self.wait(tid, timeout=timeout)
            for tid in task_ids
        ]

        results_raw = await asyncio.gather(*tasks_to_wait, return_exceptions=True)

        results: Dict[str, Optional[ExecutionResult]] = {}
        for tid, result in zip(task_ids, results_raw):
            if isinstance(result, ExecutionResult):
                results[tid] = result
            elif isinstance(result, Exception):
                logger.exception("Error waiting for task %s: %s", tid, result)
                results[tid] = None
            else:
                results[tid] = result

        return results

    # ── Cancellation ───────────────────────────────────────────────────

    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.

        Parameters
        ----------
        task_id:
            The task identifier.

        Returns
        -------
        bool
            ``True`` if the task was successfully cancelled.
        """
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return False

            if record.is_terminal:
                logger.debug("Task %s is already terminal (%s)", task_id, record.status)
                return False

            # Remove from queue if pending
            self._queue.remove(task_id)

            # Cancel the future
            if record.future and not record.future.done():
                record.future.cancel()
                try:
                    await record.future
                except asyncio.CancelledError:
                    pass

            record.status = ExecutionStatus.CANCELLED.value
            record.finished_at = datetime.now(timezone.utc).isoformat()

            logger.info("Task %s cancelled", task_id)
            return True

    async def cancel_all(self, *, running_only: bool = False) -> int:
        """
        Cancel all pending or running tasks.

        Parameters
        ----------
        running_only:
            If ``True``, only cancel running tasks (not queued ones).

        Returns
        -------
        int
            Number of tasks cancelled.
        """
        async with self._lock:
            count = 0
            to_cancel: List[str] = []

            for tid, record in self._tasks.items():
                if record.is_terminal:
                    continue
                if running_only and record.status == ExecutionStatus.PENDING.value:
                    continue
                to_cancel.append(tid)

            for tid in to_cancel:
                success = await self.cancel(tid)
                if success:
                    count += 1

            return count

    # ── Statistics ─────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """
        Return task manager statistics.

        Returns
        -------
        dict
            Keys: ``total``, ``pending``, ``running``, ``completed``,
            ``failed``, ``cancelled``, ``timeout``, ``queue_size``,
            ``active_count``, ``max_concurrent``.
        """
        status_counts: Dict[str, int] = {}
        for record in self._tasks.values():
            status_counts[record.status] = status_counts.get(record.status, 0) + 1

        return {
            "total": len(self._tasks),
            "pending": status_counts.get("pending", 0),
            "running": status_counts.get("running", 0),
            "completed": status_counts.get("completed", 0),
            "failed": status_counts.get("failed", 0),
            "cancelled": status_counts.get("cancelled", 0),
            "timeout": status_counts.get("timeout", 0),
            "queue_size": self._queue.size(),
            "active_count": self._active_count,
            "max_concurrent": self.max_concurrent,
        }

    # ── Internal: Scheduler Loop ───────────────────────────────────────

    async def _scheduler_loop(self) -> None:
        """Background loop that dispatches tasks from the queue."""
        while self._running:
            try:
                # Check if we can accept more tasks
                if self._queue.is_empty():
                    await asyncio.sleep(0.1)
                    continue

                if self._active_count >= self.max_concurrent:
                    await asyncio.sleep(0.1)
                    continue

                # Get next task from queue
                try:
                    task_data = await self._queue.get(timeout=1.0)
                except asyncio.QueueEmpty:
                    continue

                if not isinstance(task_data, TaskDefinition):
                    logger.warning("Invalid task data type in queue: %s", type(task_data))
                    continue

                # Dispatch task for execution
                asyncio.create_task(self._run_task(task_data))

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in task scheduler loop")
                await asyncio.sleep(1.0)

    async def _run_task(self, task_def: TaskDefinition) -> None:
        """Execute a task and update its record."""
        task_id = task_def.task_id
        record = self._tasks.get(task_id)
        if record is None:
            return

        async with self._semaphore:
            self._active_count += 1
            record.status = ExecutionStatus.RUNNING.value
            record.started_at = datetime.now(timezone.utc).isoformat()

            logger.info("Task %s started (%s)", task_id, record.name)

            try:
                result = await self._executor.execute_with_retry(
                    task_def.func or (lambda: None),
                    args=task_def.args,
                    kwargs=task_def.kwargs,
                    task_id=task_id,
                    name=task_def.name,
                    timeout=task_def.timeout if task_def.timeout > 0 else None,
                    retry_policy=task_def.retry_policy,
                )

                record.result = result
                record.status = result.status
                record.finished_at = datetime.now(timezone.utc).isoformat()

                # Resolve the future
                if record.future and not record.future.done():
                    record.future.set_result(result)

                # Invoke callback
                if task_def.callback:
                    try:
                        await task_def.callback(result)
                    except Exception:
                        logger.exception("Task callback failed for %s", task_id)

                logger.info(
                    "Task %s %s (%.2fs, attempts=%d)",
                    task_id,
                    result.status,
                    result.duration_seconds,
                    result.attempts,
                )

            except Exception as exc:
                record.status = ExecutionStatus.FAILED.value
                record.finished_at = datetime.now(timezone.utc).isoformat()
                record.result = ExecutionResult(
                    task_id=task_id,
                    status=ExecutionStatus.FAILED.value,
                    error=str(exc),
                )

                if record.future and not record.future.done():
                    record.future.set_exception(exc)

                logger.error("Task %s failed: %s", task_id, exc)

            finally:
                self._active_count -= 1

    # ── Internal: Cleanup Loop ─────────────────────────────────────────

    async def _cleanup_loop(self) -> None:
        """Periodically clean up old task records."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                if not self._running:
                    break

                now = time.monotonic()
                to_remove: List[str] = []

                for tid, record in self._tasks.items():
                    if not record.is_terminal:
                        continue
                    if record.finished_at:
                        try:
                            finished = datetime.fromisoformat(record.finished_at)
                            if finished.tzinfo is None:
                                finished = finished.replace(tzinfo=timezone.utc)
                            age = (datetime.now(timezone.utc) - finished).total_seconds()
                            if age > self._result_ttl:
                                to_remove.append(tid)
                        except (ValueError, TypeError):
                            to_remove.append(tid)

                async with self._lock:
                    for tid in to_remove:
                        self._tasks.pop(tid, None)

                if to_remove:
                    logger.debug("Cleaned up %d old task records", len(to_remove))

                # Also clean executor cache
                self._executor.clear_cache(max_age_seconds=self._result_ttl)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in cleanup loop")
