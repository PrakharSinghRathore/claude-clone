"""
Priority Task Queue — Asyncio-based priority queue for background tasks.

Implements a priority-based task queue with FIFO ordering within the same
priority level. Supports peek, remove, size tracking, and async get/put
operations.

Usage::

    from atlas.tasks.queue import PriorityTaskQueue, TaskPriority

    queue = PriorityTaskQueue()
    queue.put({"id": "task-1"}, TaskPriority.HIGH)
    task = await queue.get()
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ──────────────────────────────────────────────────────────────────────────────
# Task Priority
# ──────────────────────────────────────────────────────────────────────────────

class TaskPriority(IntEnum):
    """
    Task priority levels (lower value = higher priority).

    Tasks with the same priority are ordered FIFO by insertion time.
    """

    CRITICAL = 0
    """Critical tasks: system-critical, must execute immediately."""

    HIGH = 1
    """High priority: urgent tasks that should execute soon."""

    MEDIUM = 2
    """Medium priority: normal tasks (default)."""

    LOW = 3
    """Low priority: background tasks, can wait."""

    @classmethod
    def from_string(cls, value: str) -> TaskPriority:
        """
        Parse a priority from a string.

        Parameters
        ----------
        value:
            Case-insensitive priority name (e.g., ``"high"``, ``"HIGH"``).

        Returns
        -------
        TaskPriority

        Raises
        ------
        ValueError
            If the priority name is not recognized.
        """
        normalized = value.strip().upper()
        for member in cls:
            if member.name == normalized:
                return member
        raise ValueError(
            f"Unknown priority: {value!r}. "
            f"Valid: {', '.join(m.name for m in cls)}"
        )

    @classmethod
    def default(cls) -> TaskPriority:
        """Return the default task priority (MEDIUM)."""
        return cls.MEDIUM


# ──────────────────────────────────────────────────────────────────────────────
# Queue Entry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class QueueEntry:
    """
    Internal representation of an item in the priority queue.

    The ``sort_key`` tuple ensures that items with the same priority
    are ordered by insertion sequence (FIFO within priority).

    Attributes
    ----------
    sort_key:
        Tuple of (priority, sequence) for heapq ordering.
    priority:
        Task priority level.
    sequence:
        Monotonically increasing insertion counter.
    task_id:
        Unique identifier for this queued task.
    data:
        The task data payload.
    enqueued_at:
        Timestamp when the entry was enqueued.
    """

    sort_key: tuple = field(compare=True)
    priority: int = field(compare=False)
    sequence: int = field(compare=False)
    task_id: str = field(compare=False)
    data: Any = field(compare=False, default=None)
    enqueued_at: float = field(compare=False, default_factory=time.time)

    def __post_init__(self) -> None:
        if self.sort_key == () :
            self.sort_key = (self.priority, self.sequence)


# ──────────────────────────────────────────────────────────────────────────────
# Priority Task Queue
# ──────────────────────────────────────────────────────────────────────────────

class PriorityTaskQueue:
    """
    Asyncio-based priority queue with FIFO within priority levels.

    Uses a min-heap for efficient priority-based retrieval. Within the
    same priority level, tasks are ordered by insertion time (FIFO).

    Parameters
    ----------
    max_size:
        Maximum number of items in the queue. ``0`` means unlimited.

    Example
    -------
    >>> import asyncio
    >>> async def demo():
    ...     q = PriorityTaskQueue(max_size=100)
    ...     q.put("low-task", TaskPriority.LOW)
    ...     q.put("high-task", TaskPriority.HIGH)
    ...     task = await q.get()
    ...     print(task)  # "high-task" (higher priority first)
    >>> asyncio.run(demo())  # doctest: +SKIP
    """

    def __init__(self, max_size: int = 0) -> None:
        self._max_size = max_size
        self._heap: list[QueueEntry] = []
        self._sequence: int = 0
        self._task_ids: Set[str] = set()
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0

    # ── Core Operations ────────────────────────────────────────────────

    async def put(
        self,
        data: Any,
        priority: TaskPriority = TaskPriority.MEDIUM,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Add a task to the queue.

        Parameters
        ----------
        data:
            The task data payload.
        priority:
            Task priority level.
        task_id:
            Optional explicit task identifier. If not provided, a UUID is generated.

        Returns
        -------
        str
            The task identifier.

        Raises
        ------
        asyncio.QueueFull
            If the queue is full and ``max_size`` is set.
        """
        async with self._lock:
            if self._max_size > 0 and len(self._heap) >= self._max_size:
                raise asyncio.QueueFull(
                    f"Priority queue is full (max_size={self._max_size})"
                )

            tid = task_id or uuid.uuid4().hex[:12]
            if tid in self._task_ids:
                raise ValueError(f"Task ID {tid!r} already exists in the queue")

            entry = QueueEntry(
                sort_key=(priority, self._sequence),
                priority=priority,
                sequence=self._sequence,
                task_id=tid,
                data=data,
            )

            heapq.heappush(self._heap, entry)
            self._task_ids.add(tid)
            self._sequence += 1
            self._total_enqueued += 1

            # Signal waiters
            self._event.set()

            logger.debug(
                "Task %s enqueued (priority=%s, queue_size=%d)",
                tid, priority.name, len(self._heap),
            )
            return tid

    def put_nowait(
        self,
        data: Any,
        priority: TaskPriority = TaskPriority.MEDIUM,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Add a task without waiting for lock acquisition.

        Use only when it's guaranteed no concurrent access is happening
        (e.g., during initialization).

        Returns
        -------
        str
            The task identifier.
        """
        if self._max_size > 0 and len(self._heap) >= self._max_size:
            raise asyncio.QueueFull(
                f"Priority queue is full (max_size={self._max_size})"
            )

        tid = task_id or uuid.uuid4().hex[:12]
        if tid in self._task_ids:
            raise ValueError(f"Task ID {tid!r} already exists in the queue")

        entry = QueueEntry(
            sort_key=(priority, self._sequence),
            priority=priority,
            sequence=self._sequence,
            task_id=tid,
            data=data,
        )

        heapq.heappush(self._heap, entry)
        self._task_ids.add(tid)
        self._sequence += 1
        self._total_enqueued += 1
        self._event.set()

        return tid

    async def get(self, timeout: Optional[float] = None) -> Any:
        """
        Get the highest-priority task from the queue.

        Blocks until a task is available or timeout is reached.

        Parameters
        ----------
        timeout:
            Maximum wait time in seconds. ``None`` means wait indefinitely.

        Returns
        -------
        Any
            The task data payload.

        Raises
        ------
        asyncio.QueueEmpty
            If the queue is empty and timeout was reached.
        """
        if self._heap:
            return self._get_nowait()

        # Wait for a task to be enqueued
        self._event.clear()
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.QueueEmpty("No task available within timeout")

        return self._get_nowait()

    def _get_nowait(self) -> Any:
        """Get the highest-priority task without blocking."""
        if not self._heap:
            raise asyncio.QueueEmpty("Queue is empty")

        entry = heapq.heappop(self._heap)
        self._task_ids.discard(entry.task_id)
        self._total_dequeued += 1

        logger.debug(
            "Task %s dequeued (priority=%s, queue_size=%d)",
            entry.task_id, TaskPriority(entry.priority).name, len(self._heap),
        )
        return entry.data

    # ── Query Operations ───────────────────────────────────────────────

    def peek(self) -> Optional[Any]:
        """
        Peek at the next task without removing it.

        Returns
        -------
        Any or None
            The task data payload of the next task, or ``None`` if empty.
        """
        if not self._heap:
            return None
        return self._heap[0].data

    def peek_entry(self) -> Optional[QueueEntry]:
        """
        Peek at the next queue entry without removing it.

        Returns
        -------
        QueueEntry or None
            The next entry, or ``None`` if empty.
        """
        return self._heap[0] if self._heap else None

    def size(self) -> int:
        """Return the current number of items in the queue."""
        return len(self._heap)

    def is_empty(self) -> bool:
        """Return whether the queue is empty."""
        return len(self._heap) == 0

    def is_full(self) -> bool:
        """Return whether the queue has reached max_size."""
        if self._max_size <= 0:
            return False
        return len(self._heap) >= self._max_size

    def contains(self, task_id: str) -> bool:
        """
        Check if a task ID is in the queue.

        Parameters
        ----------
        task_id:
            The task identifier to check.

        Returns
        -------
        bool
        """
        return task_id in self._task_ids

    # ── Mutation Operations ────────────────────────────────────────────

    def remove(self, task_id: str) -> bool:
        """
        Remove a specific task from the queue.

        Parameters
        ----------
        task_id:
            The task identifier to remove.

        Returns
        -------
        bool
            ``True`` if the task was found and removed.
        """
        for i, entry in enumerate(self._heap):
            if entry.task_id == task_id:
                # Replace with last entry and re-heapify
                self._heap[i] = self._heap[-1]
                self._heap.pop()
                heapq.heapify(self._heap)
                self._task_ids.discard(task_id)
                logger.debug("Task %s removed from queue", task_id)
                return True
        return False

    def clear(self) -> int:
        """
        Remove all tasks from the queue.

        Returns
        -------
        int
            Number of tasks cleared.
        """
        count = len(self._heap)
        self._heap.clear()
        self._task_ids.clear()
        self._event.clear()
        logger.debug("Queue cleared (%d tasks removed)", count)
        return count

    def reprioritize(self, task_id: str, new_priority: TaskPriority) -> bool:
        """
        Change the priority of a queued task.

        Parameters
        ----------
        task_id:
            The task identifier.
        new_priority:
            The new priority level.

        Returns
        -------
        bool
            ``True`` if the task was found and reprioritized.
        """
        for i, entry in enumerate(self._heap):
            if entry.task_id == task_id:
                self._sequence += 1
                new_entry = QueueEntry(
                    sort_key=(new_priority, self._sequence),
                    priority=new_priority,
                    sequence=self._sequence,
                    task_id=entry.task_id,
                    data=entry.data,
                    enqueued_at=entry.enqueued_at,
                )
                self._heap[i] = new_entry
                heapq.heapify(self._heap)
                logger.debug(
                    "Task %s reprioritized: %s -> %s",
                    task_id,
                    TaskPriority(entry.priority).name,
                    new_priority.name,
                )
                return True
        return False

    # ── Bulk Operations ────────────────────────────────────────────────

    def get_all(self) -> List[Any]:
        """
        Get all tasks from the queue without removing them.

        Returns
        -------
        list[Any]
            All task data payloads, ordered by priority.
        """
        return [entry.data for entry in sorted(self._heap)]

    def get_by_priority(self, priority: TaskPriority) -> List[Any]:
        """
        Get all tasks with a specific priority.

        Parameters
        ----------
        priority:
            The priority level to filter by.

        Returns
        -------
        list[Any]
            Matching task data payloads.
        """
        return [
            entry.data for entry in self._heap
            if entry.priority == priority
        ]

    # ── Statistics ─────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """
        Return queue statistics.

        Returns
        -------
        dict
            Keys: ``size``, ``max_size``, ``is_empty``, ``is_full``,
            ``by_priority``, ``total_enqueued``, ``total_dequeued``.
        """
        by_priority: Dict[str, int] = {}
        for entry in self._heap:
            name = TaskPriority(entry.priority).name
            by_priority[name] = by_priority.get(name, 0) + 1

        return {
            "size": len(self._heap),
            "max_size": self._max_size if self._max_size > 0 else "unlimited",
            "is_empty": self.is_empty(),
            "is_full": self.is_full(),
            "by_priority": by_priority,
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
        }

    # ── Magic Methods ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def __repr__(self) -> str:
        return (
            f"PriorityTaskQueue(size={len(self._heap)}, "
            f"max_size={self._max_size or 'unlimited'})"
        )
