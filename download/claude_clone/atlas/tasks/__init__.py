"""
Atlas Tasks — Background task execution and management.

Provides a priority-based task queue, async executor with retry/timeout,
and a full lifecycle manager for background tasks.

Exports:
    TaskManager       – Background task lifecycle management.
    TaskExecutor      – Async task execution with retry and timeout.
    PriorityTaskQueue – Priority-based asyncio task queue.
    TaskPriority      – Task priority levels enum.
    TaskDefinition    – Task definition dataclass.
    ExecutionResult   – Task execution result dataclass.
    ExecutionStatus   – Task execution status enum.
    RetryPolicy       – Task retry configuration.
    ProgressTracker   – Batch execution progress tracker.
"""

from .manager import TaskManager
from .executor import (
    TaskExecutor,
    ExecutionResult,
    ExecutionStatus,
    RetryPolicy,
    TaskDefinition,
    ProgressTracker,
)
from .queue import PriorityTaskQueue, TaskPriority

__all__ = [
    # Manager
    "TaskManager",
    # Executor
    "TaskExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    "RetryPolicy",
    "TaskDefinition",
    "ProgressTracker",
    # Queue
    "PriorityTaskQueue",
    "TaskPriority",
]
