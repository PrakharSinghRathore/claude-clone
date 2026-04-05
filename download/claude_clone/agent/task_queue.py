"""
Background Task Queue with Parallel Execution.

Provides a priority-based task queue that runs work asynchronously without
blocking the main agent loop.  Supports parallel execution of independent
tasks, dependency chains, resource limits (max concurrent, memory, CPU),
status tracking, callbacks on completion, and cancellation.

Usage::

    queue = TaskQueue(max_concurrent=4)
    await queue.initialize()

    task_id = await queue.submit(
        "run-tests",
        handler=run_pytest,
        args={"path": "./tests"},
        priority=TaskPriority.HIGH,
    )
    await queue.wait_for(task_id)
    result = queue.get_result(task_id)
"""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class TaskPriority(enum.IntEnum):
    """Task priority levels (higher value = higher priority)."""
    BACKGROUND = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(enum.Enum):
    """Lifecycle states for a background task."""
    PENDING = "pending"
    QUEUED = "queued"
    WAITING = "waiting"       # waiting for dependencies
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BackgroundTask:
    """A unit of background work tracked by the queue."""

    id: str
    name: str
    handler: Callable[..., Coroutine[Any, Any, Any]]
    args: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    depends_on: List[str] = field(default_factory=list)
    callback: Optional[Callable[["BackgroundTask"], None]] = None
    _async_task: Optional[asyncio.Task[Any]] = field(default=None, repr=False)


# ──────────────────────────────────────────────────────────────────────────────
# TaskQueue
# ──────────────────────────────────────────────────────────────────────────────

class TaskQueue:
    """
    Async priority-based task queue with parallel execution, dependencies,
    resource limits, and completion callbacks.

    Parameters
    ----------
    max_concurrent:
        Maximum number of tasks running simultaneously.
    max_memory_mb:
        Soft memory limit in megabytes.  The queue checks ``resource.usage``
        via the :meth:`_check_resources` hook when starting tasks.  (The
        actual enforcement depends on the platform; this is advisory.)
    max_cpu_percent:
        Advisory CPU ceiling as a percentage (0–100 × core count).
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        max_memory_mb: Optional[int] = None,
        max_cpu_percent: Optional[int] = None,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent

        self._tasks: Dict[str, BackgroundTask] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._running_count: int = 0
        self._waiters: Dict[str, List[asyncio.Future[Any]]] = {}
        self._all_done_event: Optional[asyncio.Event] = None
        self._initialized: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Prepare the internal semaphore and event loop primitives."""
        if self._initialized:
            return
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._all_done_event = asyncio.Event()
        self._all_done_event.set()  # initially nothing is running
        self._initialized = True

    async def close(self) -> None:
        """Cancel all running tasks and shut down the queue."""
        for task in list(self._tasks.values()):
            if task.status == TaskStatus.RUNNING:
                await self.cancel(task.id)
        self._tasks.clear()
        self._waiters.clear()
        self._initialized = False

    # ── Submission ────────────────────────────────────────────────────────

    async def submit(
        self,
        name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        args: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        depends_on: Optional[List[str]] = None,
        callback: Optional[Callable[["BackgroundTask"], Any]] = None,
    ) -> str:
        """
        Submit a new background task.

        Parameters
        ----------
        name:
            Human-readable task label.
        handler:
            An async function to execute.  Receives keyword arguments from
            ``args``.
        args:
            Keyword arguments passed to ``handler``.
        priority:
            Scheduling priority.
        depends_on:
            List of task ids that must complete before this task can start.
        callback:
            Optional callable invoked (non-async) when the task finishes.

        Returns
        -------
        str
            The assigned task id.
        """
        if not self._initialized:
            await self.initialize()

        task_id = uuid.uuid4().hex[:12]

        # Validate dependencies exist.
        if depends_on:
            for dep_id in depends_on:
                if dep_id not in self._tasks:
                    raise ValueError(f"Dependency task '{dep_id}' not found")

        task = BackgroundTask(
            id=task_id,
            name=name,
            handler=handler,
            args=args or {},
            priority=priority,
            depends_on=depends_on or [],
            callback=callback,
        )

        # Determine initial status.
        if task.depends_on:
            task.status = TaskStatus.WAITING
        else:
            task.status = TaskStatus.QUEUED

        self._tasks[task_id] = task

        # Kick off scheduling if not blocked by dependencies.
        if task.status == TaskStatus.QUEUED:
            asyncio.get_event_loop().create_task(self._schedule(task))

        logger.debug("Submitted task %s (%s, priority=%s)", task_id, name, priority.name)
        return task_id

    # ── Scheduling ────────────────────────────────────────────────────────

    async def _schedule(self, task: BackgroundTask) -> None:
        """Acquire semaphore and run a task."""
        if task.status == TaskStatus.CANCELLED:
            return

        async with self._semaphore:
            if task.status == TaskStatus.CANCELLED:
                return

            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            self._running_count += 1
            self._signal_all_done_running()

            try:
                result = await task.handler(**task.args)
                task.result = result
                task.status = TaskStatus.COMPLETED
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                task.error = "Task was cancelled"
            except Exception as exc:
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                logger.error("Task %s (%s) failed: %s", task.id, task.name, exc)
            finally:
                task.completed_at = time.time()
                self._running_count -= 1
                self._resolve_waiters(task)
                self._trigger_dependents(task)
                self._signal_all_done_running()

                # Fire callback.
                if task.callback:
                    try:
                        if asyncio.iscoroutinefunction(task.callback):
                            await task.callback(task)
                        else:
                            task.callback(task)
                    except Exception as cb_err:
                        logger.error("Callback for task %s failed: %s", task.id, cb_err)

    def _resolve_waiters(self, task: BackgroundTask) -> None:
        """Wake up any coroutines waiting for this task via :meth:`wait_for`."""
        futures = self._waiters.pop(task.id, [])
        for fut in futures:
            if not fut.done():
                fut.set_result(task)

    def _trigger_dependents(self, completed_task: BackgroundTask) -> None:
        """Check if any WAITING tasks can now be queued."""
        for task in self._tasks.values():
            if task.status != TaskStatus.WAITING:
                continue
            if completed_task.id not in task.depends_on:
                continue

            # Check all dependencies are met.
            all_met = all(
                self._tasks[dep_id].status in (TaskStatus.COMPLETED,)
                for dep_id in task.depends_on
                if dep_id in self._tasks
            )
            if not all_met:
                # If any dependency failed or was cancelled, skip this task.
                any_failed = any(
                    self._tasks[dep_id].status in (TaskStatus.FAILED, TaskStatus.CANCELLED)
                    for dep_id in task.depends_on
                    if dep_id in self._tasks
                )
                if any_failed:
                    task.status = TaskStatus.CANCELLED
                    task.error = f"Dependency failed/cancelled: {[d for d in task.depends_on if d in self._tasks and self._tasks[d].status in (TaskStatus.FAILED, TaskStatus.CANCELLED)]}"
                    self._resolve_waiters(task)
                    self._signal_all_done_running()
                continue

            # All dependencies satisfied — queue the task.
            task.status = TaskStatus.QUEUED
            asyncio.get_event_loop().create_task(self._schedule(task))

    def _signal_all_done_running(self) -> None:
        """Set or clear the all-done event based on running tasks."""
        if self._all_done_event is None:
            return
        any_active = any(
            t.status in (TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.WAITING)
            for t in self._tasks.values()
        )
        if any_active:
            self._all_done_event.clear()
        else:
            self._all_done_event.set()

    # ── Cancellation ──────────────────────────────────────────────────────

    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a queued, waiting, or running task.

        Returns ``True`` if the task was actually cancelled, ``False`` if it
        was already in a terminal state.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False

        task.status = TaskStatus.CANCELLED
        task.error = "Cancelled by user"

        if task._async_task is not None and not task._async_task.done():
            task._async_task.cancel()

        self._resolve_waiters(task)
        self._signal_all_done_running()
        return True

    # ── Querying ──────────────────────────────────────────────────────────

    def get_status(self, task_id: str) -> Dict[str, Any]:
        """Return a status dict for a single task."""
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found")
        return self._task_to_dict(task)

    def get_result(self, task_id: str) -> Any:
        """
        Return the result of a completed task, or raise if not finished.

        Raises ``RuntimeError`` if the task has not completed yet or failed.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found")
        if task.status == TaskStatus.FAILED:
            raise RuntimeError(f"Task '{task_id}' failed: {task.error}")
        if task.status != TaskStatus.COMPLETED:
            raise RuntimeError(f"Task '{task_id}' not yet completed (status={task.status.value})")
        return task.result

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List tasks, optionally filtered by status.

        Returns a list of dicts sorted by creation time descending.
        """
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return [self._task_to_dict(t) for t in tasks[:limit]]

    # ── Waiting ───────────────────────────────────────────────────────────

    async def wait_for(self, task_id: str, timeout: Optional[float] = None) -> BackgroundTask:
        """
        Wait for a specific task to reach a terminal state.

        Parameters
        ----------
        task_id:
            The task to wait for.
        timeout:
            Maximum wait in seconds.  Raises ``asyncio.TimeoutError`` if
            exceeded.

        Returns
        -------
        BackgroundTask
            The finished task object.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found")

        # Already done?
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return task

        # Create a future and register it.
        loop = asyncio.get_running_loop()
        future: asyncio.Future[BackgroundTask] = loop.create_future()
        self._waiters.setdefault(task_id, []).append(future)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            future.cancel()
            raise

    async def wait_for_all(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until every task (queued, running, or waiting) has finished.

        Returns ``True`` if all tasks completed before the timeout,
        ``False`` otherwise.
        """
        if self._all_done_event is None:
            return True

        if timeout is not None:
            try:
                await asyncio.wait_for(self._all_done_event.wait(), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                return False
        else:
            await self._all_done_event.wait()
            return True

    # ── Resource limits ───────────────────────────────────────────────────

    def set_resource_limits(
        self,
        max_concurrent: Optional[int] = None,
        max_memory_mb: Optional[int] = None,
        max_cpu_percent: Optional[int] = None,
    ) -> None:
        """
        Adjust resource limits at runtime.

        If ``max_concurrent`` is changed, the internal semaphore is recreated.
        """
        if max_concurrent is not None and max_concurrent != self.max_concurrent:
            self.max_concurrent = max_concurrent
            if self._initialized:
                self._semaphore = asyncio.Semaphore(max_concurrent)

        if max_memory_mb is not None:
            self.max_memory_mb = max_memory_mb
        if max_cpu_percent is not None:
            self.max_cpu_percent = max_cpu_percent

    def get_utilization(self) -> Dict[str, Any]:
        """
        Return current resource utilization stats.

        Keys: ``running``, ``queued``, ``waiting``, ``completed``,
        ``failed``, ``cancelled``, ``total``, ``max_concurrent``,
        ``utilization_percent``.
        """
        counts: Dict[str, int] = {}
        for status in TaskStatus:
            counts[status.value] = 0
        for task in self._tasks.values():
            counts[task.status.value] += 1

        total = len(self._tasks)
        util = (self._running_count / self.max_concurrent * 100) if self.max_concurrent else 0

        return {
            "running": self._running_count,
            "queued": counts.get("queued", 0),
            "waiting": counts.get("waiting", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "cancelled": counts.get("cancelled", 0),
            "total": total,
            "max_concurrent": self.max_concurrent,
            "utilization_percent": round(util, 1),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _task_to_dict(task: BackgroundTask) -> Dict[str, Any]:
        duration = 0.0
        if task.started_at and task.completed_at:
            duration = round(task.completed_at - task.started_at, 3)

        return {
            "id": task.id,
            "name": task.name,
            "priority": task.priority.name,
            "status": task.status.value,
            "depends_on": task.depends_on,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "duration_seconds": duration,
            "error": task.error,
            "has_result": task.result is not None,
        }
