"""
Parallel Process — Concurrent task execution for CrewAI-style orchestration.

Executes independent tasks concurrently using asyncio.gather. Tasks that
depend on other tasks (via context) wait for those dependencies to complete
before starting. This enables maximum throughput for workflows where
multiple agents can work simultaneously.

Usage:
    from crew.process import Process
    crew = Crew(agents=[a1, a2, a3], tasks=[t1, t2, t3], process=Process.parallel)
    result = await crew.kickoff_async()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from crew.task import Task, TaskOutput

logger = logging.getLogger(__name__)


class ParallelExecutionEngine:
    """
    Executes crew tasks in parallel where possible, respecting dependencies.

    Dependency resolution:
    - Tasks with no context dependencies start immediately
    - Tasks that list other tasks in their context wait for those to finish
    - Dependency chains are resolved using topological sort
    - Independent tasks at the same dependency level run concurrently

    The engine tracks:
    - Task execution order and timing
    - Parallelism level (max concurrent tasks)
    - Total wall-clock time saved vs sequential execution
    - Per-task timing and status

    Args:
        max_concurrent: Maximum number of tasks that can run simultaneously.
                        Defaults to 4.
    """

    def __init__(self, max_concurrent: int = 4) -> None:
        self.max_concurrent = max_concurrent

    async def execute(
        self,
        tasks: List[Task],
        agents: List,  # List of CrewAgent objects
        verbose: bool = False,
        task_callback: Optional[callable] = None,
    ) -> Tuple[List[TaskOutput], Dict[str, Any]]:
        """
        Execute tasks in parallel respecting dependencies.

        Builds a dependency graph from task context references, computes
        topological levels, and executes each level concurrently (up to
        ``max_concurrent`` tasks at a time).

        Args:
            tasks:         List of :class:`Task` objects to execute.
            agents:        List of :class:`CrewAgent` objects available.
            verbose:       If ``True``, emit detailed log messages.
            task_callback: Optional callable invoked after each task completes.

        Returns:
            Tuple of ``(task_outputs, execution_stats)``.
            - task_outputs: List of :class:`TaskOutput` in completion order.
            - execution_stats: Dict with keys:
                - total_tasks:               Total number of tasks executed.
                - parallel_levels:           Number of topological levels.
                - max_concurrent_reached:    Highest concurrent task count.
                - wall_time_seconds:         Actual wall-clock time.
                - sequential_estimate_seconds: Estimated sequential time.
                - time_saved_seconds:        Wall time saved vs sequential.
                - time_saved_percent:        Percentage of time saved.
                - level_details:             List of dicts per parallel level.
        """
        if not tasks:
            return [], self._empty_stats()

        wall_start = time.monotonic()

        # Step 1: Build dependency graph
        graph = self._build_dependency_graph(tasks)

        # Step 2: Compute topological levels
        levels = self._topological_levels(tasks, graph)

        if verbose:
            logger.info(
                "Parallel engine: %d tasks grouped into %d levels",
                len(tasks), len(levels),
            )

        # Step 3: Execute each level
        completed_outputs: Dict[str, TaskOutput] = {}  # task_id -> output
        all_outputs: List[TaskOutput] = []
        level_details: List[Dict[str, Any]] = []
        max_concurrent_reached = 0
        sequential_estimate = 0.0

        for level_idx, level_tasks in enumerate(levels):
            if verbose:
                task_keys = [t.key for t in level_tasks]
                logger.info(
                    "Parallel level %d: executing %d tasks: %s",
                    level_idx + 1, len(level_tasks), task_keys,
                )

            level_outputs = await self._execute_level(
                level_tasks=level_tasks,
                completed_outputs=completed_outputs,
                verbose=verbose,
                task_callback=task_callback,
            )

            # Track max concurrency for this level
            max_concurrent_reached = max(
                max_concurrent_reached, len(level_tasks),
            )

            # Record per-task timing
            for task, output in zip(level_tasks, level_outputs):
                completed_outputs[task.id] = output
                all_outputs.append(output)
                duration = task.execution_duration or 0.0
                sequential_estimate += duration

                if verbose:
                    logger.info(
                        "Task '%s' completed in %.2fs (agent: %s)",
                        task.key, duration, output.agent,
                    )

            level_wall = sum(
                (t.execution_duration or 0.0) for t in level_tasks
            )
            level_details.append({
                "level": level_idx + 1,
                "task_count": len(level_tasks),
                "task_keys": [t.key for t in level_tasks],
                "wall_time_seconds": level_wall,
            })

        wall_end = time.monotonic()
        wall_time = wall_end - wall_start

        time_saved = max(0.0, sequential_estimate - wall_time)
        time_saved_pct = (
            (time_saved / sequential_estimate * 100)
            if sequential_estimate > 0
            else 0.0
        )

        stats: Dict[str, Any] = {
            "total_tasks": len(tasks),
            "parallel_levels": len(levels),
            "max_concurrent_reached": max_concurrent_reached,
            "wall_time_seconds": round(wall_time, 4),
            "sequential_estimate_seconds": round(sequential_estimate, 4),
            "time_saved_seconds": round(time_saved, 4),
            "time_saved_percent": round(time_saved_pct, 2),
            "level_details": level_details,
        }

        return all_outputs, stats

    def _build_dependency_graph(self, tasks: List[Task]) -> Dict[str, Set[str]]:
        """
        Build task dependency graph from context references.

        Maps each task's ``id`` to the set of task ``id`` values it depends on
        (i.e. tasks listed in its ``context`` field).

        Args:
            tasks: List of :class:`Task` objects.

        Returns:
            Dict mapping ``task_id`` to a set of dependency ``task_id`` values.
        """
        # Build a lookup: task_id -> Task
        task_map: Dict[str, Task] = {t.id: t for t in tasks}

        graph: Dict[str, Set[str]] = {}
        for task in tasks:
            deps: Set[str] = set()
            if task.context:
                for ctx_task in task.context:
                    if isinstance(ctx_task, Task) and ctx_task.id in task_map:
                        # Only add as a dependency if it's part of our task list
                        deps.add(ctx_task.id)
            graph[task.id] = deps

        return graph

    def _topological_levels(
        self,
        tasks: List[Task],
        graph: Dict[str, Set[str]],
    ) -> List[List[Task]]:
        """
        Group tasks into parallel execution levels using BFS-based topological sort.

        Tasks with no unresolved dependencies form level 0. Once level 0
        completes, tasks whose dependencies are all in level 0 form level 1,
        and so on.

        Args:
            tasks: List of :class:`Task` objects.
            graph: Dependency graph from :meth:`_build_dependency_graph`.

        Returns:
            A list of lists. Each inner list contains tasks that can execute
            concurrently within that level.
        """
        task_map: Dict[str, Task] = {t.id: t for t in tasks}
        task_ids = set(task_map.keys())

        # Compute in-degree for each task
        in_degree: Dict[str, int] = {tid: 0 for tid in task_ids}
        for tid, deps in graph.items():
            in_degree[tid] = len(deps)

        levels: List[List[Task]] = []
        remaining = set(task_ids)

        while remaining:
            # Find all tasks with in-degree 0 (no unresolved dependencies)
            ready = {tid for tid in remaining if in_degree[tid] == 0}

            if not ready:
                # Circular dependency detected — log warning and execute
                # remaining tasks sequentially as a fallback.
                logger.warning(
                    "Circular dependency detected among tasks: %s. "
                    "Executing remaining tasks as a single level.",
                    remaining,
                )
                levels.append([task_map[tid] for tid in remaining])
                break

            level = [task_map[tid] for tid in ready]
            levels.append(level)

            # Remove ready tasks and decrement in-degrees
            for tid in ready:
                remaining.remove(tid)
                for other_tid in remaining:
                    if tid in graph.get(other_tid, set()):
                        in_degree[other_tid] -= 1

        return levels

    async def _execute_level(
        self,
        level_tasks: List[Task],
        completed_outputs: Dict[str, TaskOutput],
        verbose: bool = False,
        task_callback: Optional[callable] = None,
    ) -> List[TaskOutput]:
        """
        Run all tasks in a single level concurrently.

        Uses an :class:`asyncio.Semaphore` to limit the maximum number of
        concurrent tasks to ``self.max_concurrent``.

        Args:
            level_tasks:        Tasks in this topological level.
            completed_outputs:  Outputs from previously completed levels
                                (keyed by task id), used to resolve context.
            verbose:            Emit detailed logs.
            task_callback:      Optional callable after each task.

        Returns:
            List of :class:`TaskOutput` in the same order as ``level_tasks``.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _run_single(task: Task) -> TaskOutput:
            async with semaphore:
                # Resolve context outputs for this task
                context_outputs: Optional[List[TaskOutput]] = None
                if task.context:
                    context_outputs = []
                    for ctx_task in task.context:
                        if isinstance(ctx_task, Task) and ctx_task.id in completed_outputs:
                            context_outputs.append(completed_outputs[ctx_task.id])

                try:
                    output = await task.execute_async(
                        agent=task.agent,
                        context=context_outputs if context_outputs else None,
                        tools=task.tools,
                    )
                except Exception as exc:
                    logger.error(
                        "Parallel task '%s' failed: %s", task.key, exc,
                    )
                    output = TaskOutput(
                        raw=f"Error: {exc}",
                        agent=task.agent.role if hasattr(task.agent, "role") else "unknown",
                        output_format="raw",
                    )
                    output.name = task.name or ""

                # Invoke task callback
                if task_callback:
                    try:
                        task_callback(output)
                    except Exception as cb_err:
                        logger.error("Task callback error: %s", cb_err)

                if verbose:
                    logger.info(
                        "Parallel task '%s' finished (%d chars)",
                        task.key, len(output.raw),
                    )

                return output

        # Run all tasks in this level concurrently
        results = await asyncio.gather(*[_run_single(t) for t in level_tasks])
        return list(results)

    @staticmethod
    def _empty_stats() -> Dict[str, Any]:
        """Return a zeroed stats dict for the no-tasks case."""
        return {
            "total_tasks": 0,
            "parallel_levels": 0,
            "max_concurrent_reached": 0,
            "wall_time_seconds": 0.0,
            "sequential_estimate_seconds": 0.0,
            "time_saved_seconds": 0.0,
            "time_saved_percent": 0.0,
            "level_details": [],
        }
