"""
Task Executor — Asynchronous task execution with retry, timeout, and batching.

Provides the core execution engine for running background tasks with
configurable retry policies, timeouts, parallel batch execution, progress
tracking, and resource cleanup.

Usage::

    from atlas.tasks.executor import TaskExecutor

    executor = TaskExecutor()
    result = await executor.execute(func, args=(1, 2), kwargs={})
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Sequence, Tuple

from .queue import TaskPriority

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Task Status & Result
# ──────────────────────────────────────────────────────────────────────────────

class ExecutionStatus(Enum):
    """Status of a task execution."""

    PENDING = "pending"
    """Task is queued but not yet started."""

    RUNNING = "running"
    """Task is currently executing."""

    COMPLETED = "completed"
    """Task completed successfully."""

    FAILED = "failed"
    """Task failed with an error."""

    CANCELLED = "cancelled"
    """Task was cancelled before or during execution."""

    TIMEOUT = "timeout"
    """Task exceeded its time limit."""

    RETRYING = "retrying"
    """Task failed and is being retried."""


class RetryPolicy:
    """
    Configuration for task retry behavior.

    Parameters
    ----------
    max_retries:
        Maximum number of retry attempts (0 = no retries).
    backoff_base:
        Base delay in seconds for exponential backoff.
    backoff_max:
        Maximum backoff delay in seconds.
    backoff_jitter:
        Whether to add random jitter to backoff delays.
    retryable_exceptions:
        Tuple of exception types that should trigger a retry.
        If empty, all exceptions trigger retries.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        backoff_jitter: bool = True,
        retryable_exceptions: Tuple[type, ...] = (),
    ) -> None:
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.backoff_jitter = backoff_jitter
        self.retryable_exceptions = retryable_exceptions

    def get_delay(self, attempt: int) -> float:
        """
        Calculate the backoff delay for a given retry attempt.

        Uses exponential backoff: ``base * 2^attempt`` capped at ``backoff_max``.

        Parameters
        ----------
        attempt:
            The retry attempt number (0-indexed).

        Returns
        -------
        float
            Delay in seconds.
        """
        import random
        delay = min(self.backoff_base * (2 ** attempt), self.backoff_max)
        if self.backoff_jitter:
            delay *= (0.5 + random.random() * 0.5)
        return delay

    def should_retry(self, error: Exception) -> bool:
        """
        Determine whether an error should trigger a retry.

        Parameters
        ----------
        error:
            The exception that was raised.

        Returns
        -------
        bool
        """
        if self.max_retries <= 0:
            return False
        if not self.retryable_exceptions:
            return True
        return isinstance(error, self.retryable_exceptions)


# ──────────────────────────────────────────────────────────────────────────────
# Execution Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """
    Result of a task execution attempt.

    Attributes
    ----------
    task_id:
        Unique task identifier.
    status:
        Final execution status.
    result:
        The return value (if successful).
    error:
        The error message (if failed).
    error_traceback:
        The full traceback (if failed).
    started_at:
        ISO 8601 timestamp of execution start.
    finished_at:
        ISO 8601 timestamp of execution end.
    duration_seconds:
        Execution duration in seconds.
    attempts:
        Number of execution attempts (including retries).
    """

    task_id: str = ""
    status: str = ExecutionStatus.PENDING.value
    result: Any = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    attempts: int = 1

    @property
    def success(self) -> bool:
        """Whether the execution was successful."""
        return self.status == ExecutionStatus.COMPLETED.value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        from dataclasses import asdict
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Task Definition
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskDefinition:
    """
    Definition of a task to be executed.

    Attributes
    ----------
    task_id:
        Unique task identifier.
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
        Maximum execution time in seconds (0 = no timeout).
    retry_policy:
        Retry configuration.
    callback:
        Optional callback invoked on completion.
    on_progress:
        Optional progress callback.
    """

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    func: Optional[Callable[..., Awaitable[Any]]] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = TaskPriority.MEDIUM
    name: str = ""
    timeout: float = 0.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    callback: Optional[Callable[[ExecutionResult], Awaitable[None]]] = None
    on_progress: Optional[Callable[[int, int], None]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Progress Tracker
# ──────────────────────────────────────────────────────────────────────────────

class ProgressTracker:
    """
    Tracks progress across batch task executions.

    Provides callbacks and statistics for monitoring the execution
    of multiple tasks.
    """

    def __init__(self, total: int = 0) -> None:
        self.total = total
        self.completed = 0
        self.failed = 0
        self.cancelled = 0
        self._callbacks: List[Callable[[int, int, int, int], None]] = []

    def update(self, success: bool = True, cancelled: bool = False) -> None:
        """Update progress after a task completes."""
        if cancelled:
            self.cancelled += 1
        elif success:
            self.completed += 1
        else:
            self.failed += 1

        for cb in self._callbacks:
            try:
                cb(self.completed, self.failed, self.cancelled, self.total)
            except Exception:
                logger.exception("Error in progress callback")

    def on_progress(self, callback: Callable[[int, int, int, int], None]) -> None:
        """Register a progress callback: (completed, failed, cancelled, total)."""
        self._callbacks.append(callback)

    @property
    def finished(self) -> int:
        """Total number of finished tasks."""
        return self.completed + self.failed + self.cancelled

    @property
    def remaining(self) -> int:
        """Number of tasks remaining."""
        return max(0, self.total - self.finished)

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 - 1.0)."""
        if self.finished == 0:
            return 0.0
        return self.completed / self.finished

    @property
    def is_done(self) -> bool:
        """Whether all tasks have been processed."""
        return self.finished >= self.total


# ──────────────────────────────────────────────────────────────────────────────
# Task Executor
# ──────────────────────────────────────────────────────────────────────────────

class TaskExecutor:
    """
    Asynchronous task execution engine.

    Executes tasks with support for retry, timeout, progress tracking,
    and resource cleanup. Provides both single-task and batch execution.

    Parameters
    ----------
    default_timeout:
        Default timeout in seconds for tasks without explicit timeout.
    default_retry_policy:
        Default retry policy for tasks without explicit policy.
    """

    def __init__(
        self,
        default_timeout: float = 300.0,
        default_retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        self._default_timeout = default_timeout
        self._default_retry_policy = default_retry_policy or RetryPolicy(
            max_retries=2,
            backoff_base=1.0,
        )
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._results_cache: Dict[str, ExecutionResult] = {}
        self._lock = asyncio.Lock()

    # ── Single Task Execution ──────────────────────────────────────────

    async def execute(
        self,
        func: Callable[..., Awaitable[Any]],
        *,
        args: Optional[tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        name: str = "",
        timeout: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> ExecutionResult:
        """
        Execute a single async task.

        Parameters
        ----------
        func:
            The async function to execute.
        args:
            Positional arguments for the function.
        kwargs:
            Keyword arguments for the function.
        task_id:
            Optional task identifier.
        name:
            Human-readable task name.
        timeout:
            Execution timeout in seconds.
        retry_policy:
            Retry configuration.

        Returns
        -------
        ExecutionResult
            The execution result.
        """
        tid = task_id or uuid.uuid4().hex[:12]
        task_name = name or func.__name__ if hasattr(func, "__name__") else "unnamed"

        args = args or ()
        kwargs = kwargs or {}
        effective_timeout = timeout if timeout is not None else self._default_timeout
        effective_retry = retry_policy or self._default_retry_policy

        logger.info("Executing task %s (%s)", tid, task_name)
        result = ExecutionResult(
            task_id=tid,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        start_time = time.monotonic()

        try:
            coro = func(*args, **kwargs)

            if effective_timeout > 0:
                raw_result = await asyncio.wait_for(coro, timeout=effective_timeout)
            else:
                raw_result = await coro

            result.status = ExecutionStatus.COMPLETED.value
            result.result = raw_result

        except asyncio.TimeoutError:
            result.status = ExecutionStatus.TIMEOUT.value
            result.error = f"Task timed out after {effective_timeout}s"
            logger.warning("Task %s timed out after %.1fs", tid, effective_timeout)

        except asyncio.CancelledError:
            result.status = ExecutionStatus.CANCELLED.value
            result.error = "Task was cancelled"
            logger.info("Task %s was cancelled", tid)

        except Exception as exc:
            result.status = ExecutionStatus.FAILED.value
            result.error = str(exc)
            result.error_traceback = traceback.format_exc()
            logger.error(
                "Task %s failed: %s\n%s",
                tid, exc, traceback.format_exc(),
            )

        finally:
            result.finished_at = datetime.now(timezone.utc).isoformat()
            result.duration_seconds = time.monotonic() - start_time

        # Cache the result
        async with self._lock:
            self._results_cache[tid] = result

        return result

    # ── Retry Execution ────────────────────────────────────────────────

    async def execute_with_retry(
        self,
        func: Callable[..., Awaitable[Any]],
        *,
        args: Optional[tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        name: str = "",
        timeout: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> ExecutionResult:
        """
        Execute a task with retry on failure.

        Parameters
        ----------
        func:
            The async function to execute.
        args:
            Positional arguments.
        kwargs:
            Keyword arguments.
        task_id:
            Optional task identifier.
        name:
            Human-readable task name.
        timeout:
            Timeout per attempt in seconds.
        retry_policy:
            Retry configuration.

        Returns
        -------
        ExecutionResult
            The result of the last execution attempt.
        """
        tid = task_id or uuid.uuid4().hex[:12]
        effective_retry = retry_policy or self._default_retry_policy

        last_result: Optional[ExecutionResult] = None
        attempt = 0

        while attempt <= effective_retry.max_retries:
            if attempt > 0:
                delay = effective_retry.get_delay(attempt - 1)
                logger.info(
                    "Retrying task %s (attempt %d/%d, delay=%.1fs)",
                    tid, attempt, effective_retry.max_retries, delay,
                )
                await asyncio.sleep(delay)

            result = await self.execute(
                func,
                args=args,
                kwargs=kwargs,
                task_id=tid,
                name=name,
                timeout=timeout,
            )
            last_result = result
            attempt += 1

            # Check if we should retry
            if result.success:
                return result

            if result.status in (
                ExecutionStatus.CANCELLED.value,
                ExecutionStatus.TIMEOUT.value,
            ):
                # Don't retry cancelled or timed-out tasks
                break

            error = Exception(result.error or "Unknown error")
            if not effective_retry.should_retry(error):
                break

        # Update result with total attempts
        if last_result:
            last_result.attempts = attempt

        return last_result or ExecutionResult(
            task_id=tid,
            status=ExecutionStatus.FAILED.value,
            error="No execution attempts made",
        )

    # ── Timeout Execution ──────────────────────────────────────────────

    async def execute_with_timeout(
        self,
        func: Callable[..., Awaitable[Any]],
        timeout: float,
        *,
        args: Optional[tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        cleanup: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> ExecutionResult:
        """
        Execute a task with a strict timeout and optional cleanup.

        Parameters
        ----------
        func:
            The async function to execute.
        timeout:
            Timeout in seconds.
        args:
            Positional arguments.
        kwargs:
            Keyword arguments.
        task_id:
            Optional task identifier.
        cleanup:
            Optional async cleanup function called on timeout or failure.

        Returns
        -------
        ExecutionResult
        """
        tid = task_id or uuid.uuid4().hex[:12]
        result = await self.execute(
            func, args=args, kwargs=kwargs, task_id=tid, timeout=timeout,
        )

        if result.status == ExecutionStatus.TIMEOUT.value and cleanup:
            try:
                await cleanup()
                logger.info("Cleanup executed for timed-out task %s", tid)
            except Exception:
                logger.exception("Cleanup failed for task %s", tid)

        return result

    # ── Batch Execution ────────────────────────────────────────────────

    async def execute_batch(
        self,
        tasks: List[TaskDefinition],
        *,
        max_concurrent: int = 5,
        on_progress: Optional[Callable[[int, int, int, int], None]] = None,
        fail_fast: bool = False,
    ) -> List[ExecutionResult]:
        """
        Execute multiple tasks in parallel with concurrency control.

        Parameters
        ----------
        tasks:
            List of task definitions to execute.
        max_concurrent:
            Maximum number of concurrent task executions.
        on_progress:
            Progress callback: (completed, failed, cancelled, total).
        fail_fast:
            If ``True``, stop on first failure.

        Returns
        -------
        list[ExecutionResult]
            Results for all tasks, in the same order as input.
        """
        if not tasks:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)
        results: List[Optional[ExecutionResult]] = [None] * len(tasks)
        progress = ProgressTracker(total=len(tasks))

        if on_progress:
            progress.on_progress(on_progress)

        async def _run_task(index: int, task: TaskDefinition) -> None:
            """Run a single task with semaphore control."""
            async with semaphore:
                # Check for cancellation
                if fail_fast and progress.failed > 0:
                    results[index] = ExecutionResult(
                        task_id=task.task_id,
                        status=ExecutionStatus.CANCELLED.value,
                        error="Cancelled due to fail_fast",
                    )
                    progress.update(success=False, cancelled=True)
                    return

                try:
                    result = await self.execute_with_retry(
                        task.func or (lambda: None),
                        args=task.args,
                        kwargs=task.kwargs,
                        task_id=task.task_id,
                        name=task.name,
                        timeout=task.timeout if task.timeout > 0 else None,
                        retry_policy=task.retry_policy,
                    )
                    results[index] = result
                    progress.update(success=result.success)

                    if task.callback:
                        try:
                            await task.callback(result)
                        except Exception:
                            logger.exception(
                                "Task callback failed for %s",
                                task.task_id,
                            )

                except Exception as exc:
                    results[index] = ExecutionResult(
                        task_id=task.task_id,
                        status=ExecutionStatus.FAILED.value,
                        error=str(exc),
                    )
                    progress.update(success=False)

        # Launch all tasks concurrently (semaphore controls actual parallelism)
        async_tasks = [
            asyncio.create_task(_run_task(i, task))
            for i, task in enumerate(tasks)
        ]

        # Wait for all tasks to complete
        await asyncio.gather(*async_tasks, return_exceptions=True)

        return [r for r in results if r is not None]

    # ── Result Management ──────────────────────────────────────────────

    def get_result(self, task_id: str) -> Optional[ExecutionResult]:
        """
        Get a cached execution result by task ID.

        Parameters
        ----------
        task_id:
            The task identifier.

        Returns
        -------
        ExecutionResult or None
        """
        return self._results_cache.get(task_id)

    def clear_cache(self, max_age_seconds: float = 3600.0) -> int:
        """
        Clear old results from the cache.

        Parameters
        ----------
        max_age_seconds:
            Maximum age of results to keep.

        Returns
        -------
        int
            Number of results cleared.
        """
        now = time.monotonic()
        to_remove: List[str] = []

        for tid, result in self._results_cache.items():
            if result.finished_at:
                try:
                    from datetime import datetime, timezone
                    finished = datetime.fromisoformat(result.finished_at)
                    if finished.tzinfo is None:
                        finished = finished.replace(tzinfo=timezone.utc)
                    age = (datetime.now(timezone.utc) - finished).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(tid)
                except (ValueError, TypeError):
                    to_remove.append(tid)

        for tid in to_remove:
            del self._results_cache[tid]

        if to_remove:
            logger.debug("Cleared %d cached results", len(to_remove))
        return len(to_remove)

    # ── Active Task Management ─────────────────────────────────────────

    async def cancel(self, task_id: str) -> bool:
        """
        Cancel an active task.

        Parameters
        ----------
        task_id:
            The task identifier.

        Returns
        -------
        bool
            ``True`` if the task was found and cancelled.
        """
        async with self._lock:
            task = self._active_tasks.get(task_id)
            if task and not task.done():
                task.cancel()
                del self._active_tasks[task_id]
                logger.info("Task %s cancelled", task_id)
                return True
        return False

    async def cancel_all(self) -> int:
        """
        Cancel all active tasks.

        Returns
        -------
        int
            Number of tasks cancelled.
        """
        async with self._lock:
            count = 0
            for tid, task in list(self._active_tasks.items()):
                if not task.done():
                    task.cancel()
                    count += 1
            self._active_tasks.clear()
            if count:
                logger.info("Cancelled %d active tasks", count)
            return count

    # ── Statistics ─────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """
        Return executor statistics.

        Returns
        -------
        dict
            Keys: ``active_tasks``, ``cached_results``, ``default_timeout``.
        """
        return {
            "active_tasks": len(self._active_tasks),
            "cached_results": len(self._results_cache),
            "default_timeout": self._default_timeout,
        }
