"""
Flow — decorator-based workflow orchestration.

A Flow is a stateful class whose methods are wired together using decorators:

- ``@start`` — marks the entry point of the flow
- ``@listen("method_name")`` — runs after the named method completes
- ``@router`` — returns the name of the next method to run
- ``@and_("a", "b")`` — runs only after ALL listed methods complete
- ``@or_("a", "b")`` — runs as soon as ANY listed method completes

Example::

    class ResearchFlow(Flow):
        @start()
        def begin(self):
            return {"topic": "AI"}

        @listen("begin")
        def search(self):
            # Perform search
            return {"results": [...]}

        @router("search")
        def route_results(self):
            if len(self.context.get("results", [])) > 5:
                return "summarize"
            return "deep_research"

        @listen("summarize")
        @listen("deep_research")
        def finalize(self):
            return {"done": True}
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

from flow.context import FlowContext

logger = logging.getLogger(__name__)
T = TypeVar("T")


# ──────────────────────────────────────────────
# Decorators
# ──────────────────────────────────────────────

def start(method: Optional[Callable] = None, *, retries: int = 0) -> Any:
    """
    Mark a method as the entry point of a flow.

    Usage::

        @start()
        def begin(self):
            ...

    Or with retries::

        @start(retries=3)
        def begin(self):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(self, *args, **kwargs):
            self._current_step = fn.__name__
            self.context.current_step = fn.__name__
            self.context.step_count += 1
            logger.info("[%s] Step '%s' started", self.__class__.__name__, fn.__name__)
            try:
                result = await fn(self, *args, **kwargs)
                if result is not None:
                    if isinstance(result, dict):
                        self.context.update(result)
                    else:
                        self.context.set(f"_{fn.__name__}_result", result)
                return result
            except Exception as e:
                logger.error("[%s] Step '%s' failed: %s", self.__class__.__name__, fn.__name__, e)
                raise

        @functools.wraps(fn)
        def sync_wrapper(self, *args, **kwargs):
            self._current_step = fn.__name__
            self.context.current_step = fn.__name__
            self.context.step_count += 1
            logger.info("[%s] Step '%s' started", self.__class__.__name__, fn.__name__)
            try:
                result = fn(self, *args, **kwargs)
                if result is not None:
                    if isinstance(result, dict):
                        self.context.update(result)
                    else:
                        self.context.set(f"_{fn.__name__}_result", result)
                return result
            except Exception as e:
                logger.error("[%s] Step '%s' failed: %s", self.__class__.__name__, fn.__name__, e)
                raise

        wrapper = async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper
        wrapper._flow_start = True  # type: ignore[attr-defined]
        wrapper._flow_step_name = fn.__name__  # type: ignore[attr-defined]
        wrapper._flow_retries = retries  # type: ignore[attr-defined]
        return wrapper

    if method is not None:
        return decorator(method)
    return decorator


def listen(*method_names: str, retries: int = 0) -> Callable:
    """
    Mark a method to execute after the named method(s) complete.

    Can listen to multiple methods — will execute after ALL have completed.

    Usage::

        @listen("research")
        def write(self):
            ...

        @listen("step_a", "step_b")
        def merge(self):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(self, *args, **kwargs):
            self._current_step = fn.__name__
            self.context.current_step = fn.__name__
            self.context.step_count += 1
            logger.info("[%s] Step '%s' started (listening to %s)",
                       self.__class__.__name__, fn.__name__, method_names)
            try:
                result = await fn(self, *args, **kwargs)
                if result is not None:
                    if isinstance(result, dict):
                        self.context.update(result)
                    else:
                        self.context.set(f"_{fn.__name__}_result", result)
                return result
            except Exception as e:
                logger.error("[%s] Step '%s' failed: %s", self.__class__.__name__, fn.__name__, e)
                raise

        @functools.wraps(fn)
        def sync_wrapper(self, *args, **kwargs):
            self._current_step = fn.__name__
            self.context.current_step = fn.__name__
            self.context.step_count += 1
            logger.info("[%s] Step '%s' started (listening to %s)",
                       self.__class__.__name__, fn.__name__, method_names)
            try:
                result = fn(self, *args, **kwargs)
                if result is not None:
                    if isinstance(result, dict):
                        self.context.update(result)
                    else:
                        self.context.set(f"_{fn.__name__}_result", result)
                return result
            except Exception as e:
                logger.error("[%s] Step '%s' failed: %s", self.__class__.__name__, fn.__name__, e)
                raise

        wrapper = async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper
        wrapper._flow_listen = list(method_names)  # type: ignore[attr-defined]
        wrapper._flow_step_name = fn.__name__  # type: ignore[attr-defined]
        wrapper._flow_retries = retries  # type: ignore[attr-defined]
        return wrapper

    return decorator


def router(*method_names: str, retries: int = 0) -> Callable:
    """
    Mark a method as a router that decides the next step.

    The method must return a string matching the name of the next method.

    Usage::

        @router("analyze")
        def decide(self):
            if self.context.get("confidence", 0) > 0.8:
                return "finalize"
            return "research_more"
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(self, *args, **kwargs):
            self._current_step = fn.__name__
            self.context.current_step = fn.__name__
            self.context.step_count += 1
            logger.info("[%s] Router '%s' started", self.__class__.__name__, fn.__name__)
            result = await fn(self, *args, **kwargs)
            if isinstance(result, str):
                self._next_step = result
                logger.info("[%s] Router '%s' -> '%s'", self.__class__.__name__, fn.__name__, result)
            return result

        @functools.wraps(fn)
        def sync_wrapper(self, *args, **kwargs):
            self._current_step = fn.__name__
            self.context.current_step = fn.__name__
            self.context.step_count += 1
            logger.info("[%s] Router '%s' started", self.__class__.__name__, fn.__name__)
            result = fn(self, *args, **kwargs)
            if isinstance(result, str):
                self._next_step = result
                logger.info("[%s] Router '%s' -> '%s'", self.__class__.__name__, fn.__name__, result)
            return result

        wrapper = async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper
        wrapper._flow_router = list(method_names)  # type: ignore[attr-defined]
        wrapper._flow_step_name = fn.__name__  # type: ignore[attr-defined]
        wrapper._flow_retries = retries  # type: ignore[attr-defined]
        return wrapper

    return decorator


def and_(*method_names: str, retries: int = 0) -> Callable:
    """
    Mark a method to execute only after ALL named methods complete.

    Similar to ``@listen`` but explicitly indicates a convergence point.

    Usage::

        @and_("parallel_search", "parallel_fetch")
        def merge_results(self):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(self, *args, **kwargs):
            self._current_step = fn.__name__
            self.context.current_step = fn.__name__
            self.context.step_count += 1
            logger.info("[%s] AND step '%s' started (waiting for %s)",
                       self.__class__.__name__, fn.__name__, method_names)
            result = await fn(self, *args, **kwargs)
            if result is not None:
                if isinstance(result, dict):
                    self.context.update(result)
                else:
                    self.context.set(f"_{fn.__name__}_result", result)
            return result

        @functools.wraps(fn)
        def sync_wrapper(self, *args, **kwargs):
            self._current_step = fn.__name__
            self.context.current_step = fn.__name__
            self.context.step_count += 1
            logger.info("[%s] AND step '%s' started (waiting for %s)",
                       self.__class__.__name__, fn.__name__, method_names)
            result = fn(self, *args, **kwargs)
            if result is not None:
                if isinstance(result, dict):
                    self.context.update(result)
                else:
                    self.context.set(f"_{fn.__name__}_result", result)
            return result

        wrapper = async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper
        wrapper._flow_and = list(method_names)  # type: ignore[attr-defined]
        wrapper._flow_listen = list(method_names)  # type: ignore[attr-defined]
        wrapper._flow_step_name = fn.__name__  # type: ignore[attr-defined]
        wrapper._flow_retries = retries  # type: ignore[attr-defined]
        return wrapper

    return decorator


def or_(*method_names: str, retries: int = 0) -> Callable:
    """
    Mark a method to execute as soon as ANY of the named methods completes.

    Useful for race conditions or fallback patterns.

    Usage::

        @or_("cache_lookup", "db_lookup")
        def use_result(self):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(self, *args, **kwargs):
            self._current_step = fn.__name__
            self.context.current_step = fn.__name__
            self.context.step_count += 1
            logger.info("[%s] OR step '%s' started (any of %s)",
                       self.__class__.__name__, fn.__name__, method_names)
            result = await fn(self, *args, **kwargs)
            if result is not None:
                if isinstance(result, dict):
                    self.context.update(result)
                else:
                    self.context.set(f"_{fn.__name__}_result", result)
            return result

        @functools.wraps(fn)
        def sync_wrapper(self, *args, **kwargs):
            self._current_step = fn.__name__
            self.context.current_step = fn.__name__
            self.context.step_count += 1
            logger.info("[%s] OR step '%s' started (any of %s)",
                       self.__class__.__name__, fn.__name__, method_names)
            result = fn(self, *args, **kwargs)
            if result is not None:
                if isinstance(result, dict):
                    self.context.update(result)
                else:
                    self.context.set(f"_{fn.__name__}_result", result)
            return result

        wrapper = async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper
        wrapper._flow_or = list(method_names)  # type: ignore[attr-defined]
        wrapper._flow_listen = list(method_names)  # type: ignore[attr-defined]
        wrapper._flow_step_name = fn.__name__  # type: ignore[attr-defined]
        wrapper._flow_retries = retries  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ──────────────────────────────────────────────
# Flow base class
# ──────────────────────────────────────────────

class Flow:
    """
    Base class for defining flows using decorators.

    Subclass this and use ``@start``, ``@listen``, ``@router``, ``@and_``,
    and ``@or_`` decorators to wire methods together.

    Attributes:
        context: The flow's shared state context.
        id: Unique flow execution identifier.
    """

    def __init__(self):
        self.context = FlowContext()
        self.id: str = uuid.uuid4().hex
        self._current_step: Optional[str] = None
        self._next_step: Optional[str] = None
        self._completed_steps: Set[str] = set()
        self._step_results: Dict[str, Any] = {}
        self._execution_log: List[Dict[str, Any]] = []

    def _discover_methods(self) -> Dict[str, Callable]:
        """Discover all decorated methods on this flow instance."""
        methods = {}
        for name in dir(self):
            if name.startswith("_"):
                continue
            attr = getattr(self, name, None)
            if attr is None or not callable(attr):
                continue
            if hasattr(attr, "_flow_step_name"):
                methods[name] = attr
        return methods

    def _get_start_method(self, methods: Dict[str, Callable]) -> Optional[Callable]:
        """Find the @start decorated method."""
        for name, method in methods.items():
            if hasattr(method, "_flow_start"):
                return method
        return None

    def _get_listeners(self, methods: Dict[str, Callable], step_name: str) -> List[Callable]:
        """Get all methods that listen to a given step."""
        listeners = []
        for name, method in methods.items():
            if hasattr(method, "_flow_listen"):
                if step_name in method._flow_listen:
                    listeners.append(method)
        return listeners

    def _get_router_targets(self, methods: Dict[str, Callable], step_name: str) -> List[str]:
        """Get router targets for a given step."""
        for name, method in methods.items():
            if hasattr(method, "_flow_router"):
                if step_name in method._flow_router:
                    return method._flow_router
        return []

    def _is_router(self, method: Callable) -> bool:
        """Check if a method is a router."""
        return hasattr(method, "_flow_router")

    def _execute_step(self, method: Callable) -> Any:
        """Execute a single step and track completion."""
        step_name = method._flow_step_name
        retries = getattr(method, "_flow_retries", 0)
        last_error = None

        for attempt in range(retries + 1):
            try:
                result = method(self)
                self._completed_steps.add(step_name)
                self._step_results[step_name] = result
                self._execution_log.append({
                    "step": step_name,
                    "status": "completed",
                    "attempt": attempt + 1,
                })
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    "[%s] Step '%s' failed (attempt %d/%d): %s",
                    self.__class__.__name__, step_name, attempt + 1, retries + 1, e,
                )

        self._execution_log.append({
            "step": step_name,
            "status": "failed",
            "error": str(last_error),
        })
        raise last_error  # type: ignore[misc]

    def kickoff(
        self,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the flow synchronously.

        Args:
            inputs: Optional initial data to populate the context.

        Returns:
            The final context state dictionary.
        """
        if inputs:
            self.context.update(inputs)

        self._discover_and_run()
        return self.context.snapshot()

    async def kickoff_async(
        self,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the flow asynchronously.

        Args:
            inputs: Optional initial data to populate the context.

        Returns:
            The final context state dictionary.
        """
        if inputs:
            self.context.update(inputs)

        await self._discover_and_run_async()
        return self.context.snapshot()

    def _discover_and_run(self) -> None:
        """Discover decorated methods and execute the flow."""
        methods = self._discover_methods()

        # Find start method
        start_method = self._get_start_method(methods)
        if start_method is None:
            raise RuntimeError(
                f"Flow '{self.__class__.__name__}' has no @start method"
            )

        # Execute start
        self._execute_step(start_method)

        # Execute the dependency chain
        self._run_chain(methods, start_method._flow_step_name)

    async def _discover_and_run_async(self) -> None:
        """Discover decorated methods and execute the flow asynchronously."""
        methods = self._discover_methods()

        start_method = self._get_start_method(methods)
        if start_method is None:
            raise RuntimeError(
                f"Flow '{self.__class__.__name__}' has no @start method"
            )

        # Execute start
        step_name = start_method._flow_step_name
        if asyncio.iscoroutinefunction(start_method):
            await start_method(self)
            self._completed_steps.add(step_name)
        else:
            self._execute_step(start_method)

        await self._run_chain_async(methods, step_name)

    def _run_chain(self, methods: Dict[str, Callable], completed_step: str) -> None:
        """Run the execution chain from a completed step."""
        max_steps = getattr(self, "_flow_config", None)
        if max_steps and hasattr(max_steps, "max_iterations"):
            max_steps = max_steps.max_iterations
        else:
            max_steps = 100

        queue = [completed_step]
        visited = set()

        while queue and self.context.step_count < max_steps:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            listeners = self._get_listeners(methods, current)
            for listener in listeners:
                listener_name = listener._flow_step_name

                # Check if all prerequisites are met
                prereqs = listener._flow_listen
                if all(p in self._completed_steps for p in prereqs):
                    if listener_name not in self._completed_steps:
                        result = self._execute_step(listener)
                        queue.append(listener_name)

            # Handle router
            for name, method in methods.items():
                if self._is_router(method):
                    if hasattr(method, "_flow_router") and current in method._flow_router:
                        if name not in self._completed_steps:
                            result = self._execute_step(method)
                            if self._next_step:
                                target = self._next_step
                                self._next_step = None
                                # Execute the routed step directly
                                target_method = methods.get(target)
                                if target_method and target not in self._completed_steps:
                                    self._execute_step(target_method)
                                    queue.append(target)

    async def _run_chain_async(self, methods: Dict[str, Callable], completed_step: str) -> None:
        """Run the execution chain asynchronously."""
        max_steps = getattr(self, "_flow_config", None)
        if max_steps and hasattr(max_steps, "max_iterations"):
            max_steps = max_steps.max_iterations
        else:
            max_steps = 100

        queue = [completed_step]
        visited = set()

        while queue and self.context.step_count < max_steps:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            listeners = self._get_listeners(methods, current)
            for listener in listeners:
                listener_name = listener._flow_step_name
                prereqs = listener._flow_listen
                if all(p in self._completed_steps for p in prereqs):
                    if listener_name not in self._completed_steps:
                        if asyncio.iscoroutinefunction(listener):
                            await listener(self)
                            self._completed_steps.add(listener_name)
                        else:
                            self._execute_step(listener)
                        queue.append(listener_name)

            for name, method in methods.items():
                if self._is_router(method):
                    if hasattr(method, "_flow_router") and current in method._flow_router:
                        if name not in self._completed_steps:
                            if asyncio.iscoroutinefunction(method):
                                await method(self)
                                self._completed_steps.add(name)
                            else:
                                self._execute_step(method)
                            if self._next_step:
                                target = self._next_step
                                self._next_step = None
                                target_method = methods.get(target)
                                if target_method and target not in self._completed_steps:
                                    if asyncio.iscoroutinefunction(target_method):
                                        await target_method(self)
                                        self._completed_steps.add(target)
                                    else:
                                        self._execute_step(target_method)
                                    queue.append(target)

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Return the execution log for this flow run."""
        return list(self._execution_log)

    def get_step_results(self) -> Dict[str, Any]:
        """Return all step results."""
        return dict(self._step_results)

    def __repr__(self) -> str:
        return (
            f"Flow(name={self.__class__.__name__}, id={self.id[:8]}, "
            f"steps_completed={len(self._completed_steps)})"
        )
