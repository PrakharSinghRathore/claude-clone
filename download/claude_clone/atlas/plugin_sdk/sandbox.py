"""
Plugin Sandbox — Sandboxed execution environment for Atlas plugins.

The sandbox provides a controlled runtime that enforces:

* **Permission checks** — plugins can only perform actions for which they
  have been granted permission.
* **Path restrictions** — file I/O is confined to an allow-list of
  directories (or a per-plugin whitelist).
* **Resource limits** — configurable caps on memory usage and CPU time
  to prevent runaway plugins from consuming host resources.
* **Audit logging** — every sandbox action is logged for forensics.

Architecture
------------
```
  Plugin code
       │
       ▼
  ┌──────────┐   permission check   ┌──────────────┐
  │ Sandbox  │ ──────────────────▶  │ Permission   │
  │ Executor │                      │ Store        │
  └────┬─────┘                      └──────────────┘
       │
       ▼
  ┌──────────┐   resource limits   ┌──────────────┐
  │ Path     │ ◀────────────────  │ Resource     │
  │ Guard    │                     │ Monitor      │
  └──────────┘                     └──────────────┘
```

Usage::

    sandbox = PluginSandbox()

    # Grant permissions
    sandbox.grant_permission("my-plugin", PluginPermission.FILE_READ)
    sandbox.add_allowed_path("my-plugin", Path("/tmp/plugin-data"))

    # Execute code
    result = sandbox.execute("my-plugin", "open('/tmp/data.txt').read()")
"""

from __future__ import annotations

import abc
import builtins
import ctypes
import functools
import importlib
import io
import logging
import os
import resource
import signal
import sys
import threading
import time
import traceback
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from atlas.plugin_sdk.core import (
    PluginCapability,
    PluginPermission,
    PluginSandboxError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResourceLimits:
    """Configurable resource caps for sandboxed execution.

    Attributes:
        max_memory_mb: Maximum RSS in megabytes (0 = unlimited).
        max_cpu_seconds: Maximum CPU time in seconds (0 = unlimited).
        max_wall_seconds: Maximum wall-clock time in seconds (0 = unlimited).
        max_open_files: Maximum open file descriptors (0 = unlimited).
        max_process_threads: Maximum threads the plugin can spawn (0 = unlimited).
    """

    max_memory_mb: int = 0
    max_cpu_seconds: float = 0.0
    max_wall_seconds: float = 30.0
    max_open_files: int = 0
    max_process_threads: int = 0

    @property
    def memory_bytes(self) -> int:
        """Return the memory limit in bytes."""
        return self.max_memory_mb * 1024 * 1024

    @property
    def has_limits(self) -> bool:
        """Return ``True`` if any limit is configured."""
        return (
            self.max_memory_mb > 0
            or self.max_cpu_seconds > 0
            or self.max_wall_seconds > 0
            or self.max_open_files > 0
            or self.max_process_threads > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_seconds": self.max_cpu_seconds,
            "max_wall_seconds": self.max_wall_seconds,
            "max_open_files": self.max_open_files,
            "max_process_threads": self.max_process_threads,
        }

    @classmethod
    def default(cls) -> ResourceLimits:
        """Return default limits suitable for most plugins."""
        return cls(
            max_memory_mb=256,
            max_cpu_seconds=30.0,
            max_wall_seconds=60.0,
            max_open_files=64,
            max_process_threads=4,
        )

    @classmethod
    def unrestricted(cls) -> ResourceLimits:
        """Return an unrestricted limit set (for trusted plugins)."""
        return cls()

    @classmethod
    def strict(cls) -> ResourceLimits:
        """Return a strict limit set for untrusted plugins."""
        return cls(
            max_memory_mb=64,
            max_cpu_seconds=5.0,
            max_wall_seconds=10.0,
            max_open_files=16,
            max_process_threads=1,
        )


@dataclass
class SandboxConfig:
    """Configuration for a single plugin's sandbox.

    Attributes:
        allowed_paths: Paths the plugin is allowed to read/write.
        permissions: Granted permissions.
        resource_limits: Resource caps.
        denied_modules: Python modules the plugin cannot import.
        network_allowed: Whether network access is permitted.
        stdin_enabled: Whether the plugin can read from stdin.
        stdout_capture: Whether to capture stdout.
        stderr_capture: Whether to capture stderr.
    """

    allowed_paths: Set[Path] = field(default_factory=set)
    permissions: Set[PluginPermission] = field(default_factory=set)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits.default)
    denied_modules: FrozenSet[str] = frozenset({
        "os",
        "subprocess",
        "signal",
        "ctypes",
        "multiprocessing",
        "shutil",
        "sys",
    })
    network_allowed: bool = False
    stdin_enabled: bool = False
    stdout_capture: bool = True
    stderr_capture: bool = True


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution.

    Attributes:
        success: Whether the code executed without errors.
        return_value: The return value of the executed code.
        stdout: Captured stdout content.
        stderr: Captured stderr content.
        error: Error message if execution failed.
        execution_time_ms: Wall-clock time in milliseconds.
        memory_used_mb: Peak memory usage in MB (if measurable).
        timed_out: Whether execution was terminated due to a timeout.
    """

    success: bool = True
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    timed_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "success": self.success,
            "return_value": str(self.return_value),
            "stdout": self.stdout[:1000],
            "stderr": self.stderr[:1000],
            "error": self.error,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "memory_used_mb": round(self.memory_used_mb, 2),
            "timed_out": self.timed_out,
        }


# ---------------------------------------------------------------------------
# Path Guard
# ---------------------------------------------------------------------------

class PathGuard:
    """Enforces filesystem path restrictions for a plugin.

    Only paths under the allowed directories are accessible.  All other
    paths are blocked with a :class:`PluginSandboxError`.

    Attributes:
        plugin_name: Name of the plugin (for error messages).
        allowed_paths: Set of allowed root directories.
    """

    def __init__(
        self,
        plugin_name: str,
        allowed_paths: Optional[Set[Path]] = None,
    ) -> None:
        self.plugin_name = plugin_name
        self.allowed_paths: Set[Path] = set()
        if allowed_paths:
            for p in allowed_paths:
                self.allowed_paths.add(Path(p).resolve())

    def add_path(self, path: Union[str, Path]) -> None:
        """Add a directory to the allow-list."""
        self.allowed_paths.add(Path(path).resolve())

    def remove_path(self, path: Union[str, Path]) -> None:
        """Remove a directory from the allow-list."""
        self.allowed_paths.discard(Path(path).resolve())

    def check(self, path: Union[str, Path], *, write: bool = False) -> bool:
        """Check if *path* is accessible under the current restrictions.

        Args:
            path: The path to check.
            write: If ``True``, also verify write permission.

        Returns:
            ``True`` if the path is allowed.

        Raises:
            PluginSandboxError: If the path is outside the allowed set.
        """
        resolved = Path(path).resolve()

        if not self.allowed_paths:
            # No restrictions configured
            return True

        for allowed in self.allowed_paths:
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                continue

        logger.warning(
            "Path guard blocked access for plugin %s to %s (write=%s)",
            self.plugin_name,
            resolved,
            write,
        )
        raise PluginSandboxError(
            f"Path {resolved} is outside the allowed directories for plugin {self.plugin_name!r}",
            plugin_name=self.plugin_name,
        )

    def wrap_open(
        self,
        original_open: Callable[..., Any],
    ) -> Callable[..., Any]:
        """Return a wrapped ``open()`` that enforces path restrictions.

        The wrapper checks the file path before delegating to the original.
        """
        @functools.wraps(original_open)
        def sandboxed_open(
            file: Union[str, bytes, int, os.PathLike],
            mode: str = "r",
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if isinstance(file, (str, os.PathLike)):
                is_write = any(m in mode for m in ("w", "a", "x", "+"))
                self.check(file, write=is_write)
            return original_open(file, mode, *args, **kwargs)

        return sandboxed_open


# ---------------------------------------------------------------------------
# Resource Monitor
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Tracks and enforces resource usage for a plugin.

    On Linux/macOS, uses ``resource.getrusage`` for memory measurement.
    Falls back to ``psutil`` if available, otherwise tracks wall-clock time
    only.
    """

    def __init__(
        self,
        plugin_name: str,
        limits: Optional[ResourceLimits] = None,
    ) -> None:
        self.plugin_name = plugin_name
        self.limits = limits or ResourceLimits.default()
        self._start_time: Optional[float] = None
        self._start_rss: int = 0
        self._peak_rss: int = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Mark the start of a timed section."""
        with self._lock:
            self._start_time = time.monotonic()
            self._start_rss = self._get_memory_kb()
            self._peak_rss = self._start_rss

            if self.limits.max_memory_mb > 0:
                self._set_memory_limit(self.limits.memory_bytes)

            if self.limits.max_cpu_seconds > 0:
                self._set_cpu_limit(self.limits.max_cpu_seconds)

    def stop(self) -> Tuple[float, float]:
        """Mark the end of a timed section.

        Returns:
            ``(wall_seconds, memory_used_mb)``
        """
        with self._lock:
            if self._start_time is None:
                return 0.0, 0.0

            wall = time.monotonic() - self._start_time
            current_rss = self._get_memory_kb()
            peak = max(self._peak_rss, current_rss)
            self._start_time = None

            # Clear limits
            self._clear_limits()

            mem_mb = (peak - self._start_rss) / 1024.0
            return wall, mem_mb

    def check_wall_timeout(self) -> None:
        """Raise :class:`PluginSandboxError` if wall-clock time exceeded."""
        if self._start_time is None or self.limits.max_wall_seconds <= 0:
            return
        elapsed = time.monotonic() - self._start_time
        if elapsed > self.limits.max_wall_seconds:
            raise PluginSandboxError(
                f"Plugin {self.plugin_name!r} exceeded wall-clock limit "
                f"({elapsed:.1f}s > {self.limits.max_wall_seconds}s)",
                plugin_name=self.plugin_name,
                resource="wall_time",
                limit=self.limits.max_wall_seconds,
            )

    @staticmethod
    def _get_memory_kb() -> int:
        """Return current process RSS in kilobytes."""
        try:
            import resource as res
            usage = res.getrusage(res.RUSAGE_SELF)
            return usage.ru_maxrss
        except (ImportError, AttributeError):
            pass

        try:
            import psutil
            return psutil.Process().memory_info().rss // 1024
        except ImportError:
            pass

        return 0

    @staticmethod
    def _set_memory_limit(bytes_limit: int) -> None:
        """Set the process memory limit (soft + hard)."""
        try:
            import resource as res
            res.setrlimit(res.RLIMIT_AS, (bytes_limit, bytes_limit))
        except (ImportError, ValueError, OSError):
            pass

    @staticmethod
    def _set_cpu_limit(seconds: float) -> None:
        """Set the process CPU time limit."""
        try:
            import resource as res
            res.setrlimit(res.RLIMIT_CPU, (int(seconds), int(seconds + 1)))
        except (ImportError, ValueError, OSError):
            pass

    @staticmethod
    def _clear_limits() -> None:
        """Reset resource limits to unrestricted."""
        try:
            import resource as res
            # RLIM_INFINITY varies by platform
            try:
                inf = res.RLIM_INFINITY
            except AttributeError:
                inf = (1 << 63) - 1
            res.setrlimit(res.RLIMIT_AS, (inf, inf))
            res.setrlimit(res.RLIMIT_CPU, (inf, inf))
        except (ImportError, ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# Import Guard
# ---------------------------------------------------------------------------

class ImportGuard:
    """Restricts which Python modules a plugin can import.

    Usage::

        guard = ImportGuard(
            plugin_name="my-plugin",
            denied={"os", "subprocess", "ctypes"},
        )
        guard.install()
        # ... plugin code runs ...
        guard.uninstall()
    """

    def __init__(
        self,
        plugin_name: str,
        denied: Optional[FrozenSet[str]] = None,
        allowed: Optional[FrozenSet[str]] = None,
    ) -> None:
        self.plugin_name = plugin_name
        self.denied = denied or frozenset()
        self.allowed = allowed  # If set, ONLY these modules can be imported
        self._original_import: Optional[Callable[..., Any]] = None

    def install(self) -> None:
        """Install the import hook."""
        self._original_import = builtins.__import__
        guard = self

        def restricted_import(
            name: str,
            globals: Optional[Dict[str, Any]] = None,
            locals: Optional[Dict[str, Any]] = None,
            fromlist: Tuple[str, ...] = (),
            level: int = 0,
        ) -> types.ModuleType:
            # Check denied list
            base_module = name.split(".")[0]
            if base_module in guard.denied:
                raise ImportError(
                    f"Plugin {guard.plugin_name!r} is not allowed to import {name!r}"
                )

            # Check allowed list (if configured)
            if guard.allowed is not None and base_module not in guard.allowed:
                raise ImportError(
                    f"Plugin {guard.plugin_name!r} can only import allowed modules; "
                    f"{name!r} is not in the allow-list"
                )

            return guard._original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = restricted_import
        logger.debug("Import guard installed for plugin %s", self.plugin_name)

    def uninstall(self) -> None:
        """Restore the original import function."""
        if self._original_import is not None:
            builtins.__import__ = self._original_import
            self._original_import = None
            logger.debug("Import guard uninstalled for plugin %s", self.plugin_name)


# ---------------------------------------------------------------------------
# Plugin Sandbox
# ---------------------------------------------------------------------------

class PluginSandbox:
    """Sandboxed execution environment for Atlas plugins.

    The sandbox enforces permissions, path restrictions, import guards,
    and resource limits.  Each plugin gets its own isolated sandbox
    configuration.

    Usage::

        sandbox = PluginSandbox()
        sandbox.grant_permission("my-plugin", PluginPermission.FILE_READ)
        sandbox.add_allowed_path("my-plugin", Path("/data"))
        result = sandbox.execute("my-plugin", "open('/data/file.txt').read()")
    """

    def __init__(
        self,
        default_limits: Optional[ResourceLimits] = None,
        default_allowed_paths: Optional[Set[Path]] = None,
    ) -> None:
        self._configs: Dict[str, SandboxConfig] = {}
        self._path_guards: Dict[str, PathGuard] = {}
        self._resource_monitors: Dict[str, ResourceMonitor] = {}
        self._import_guards: Dict[str, ImportGuard] = {}
        self._lock = threading.Lock()
        self._default_limits = default_limits or ResourceLimits.default()
        self._default_paths = default_allowed_paths or set()

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

    # -- Permission management ---------------------------------------------

    def _get_or_create_config(self, plugin_name: str) -> SandboxConfig:
        """Get or create a sandbox config for a plugin."""
        if plugin_name not in self._configs:
            self._configs[plugin_name] = SandboxConfig(
                allowed_paths=set(self._default_paths),
                resource_limits=self._default_limits,
            )
            self._path_guards[plugin_name] = PathGuard(plugin_name)
            for p in self._default_paths:
                self._path_guards[plugin_name].add_path(p)
            self._resource_monitors[plugin_name] = ResourceMonitor(
                plugin_name, self._default_limits
            )
        return self._configs[plugin_name]

    def check_permission(
        self,
        plugin_name: str,
        permission: Union[PluginPermission, str],
    ) -> bool:
        """Check if a plugin has a specific permission.

        Args:
            plugin_name: The plugin to check.
            permission: A :class:`PluginPermission` enum or string value.

        Returns:
            ``True`` if the permission is granted.
        """
        with self._lock:
            config = self._configs.get(plugin_name)
            if config is None:
                return False
            perm = (
                permission
                if isinstance(permission, PluginPermission)
                else PluginPermission(permission)
            )
            result = perm in config.permissions
            self._audit("check_permission", plugin_name, permission=str(perm), result=result)
            return result

    def grant_permission(
        self,
        plugin_name: str,
        permission: Union[PluginPermission, str],
    ) -> None:
        """Grant a permission to a plugin.

        Args:
            plugin_name: The plugin to grant the permission to.
            permission: A :class:`PluginPermission` enum or string value.
        """
        with self._lock:
            config = self._get_or_create_config(plugin_name)
            perm = (
                permission
                if isinstance(permission, PluginPermission)
                else PluginPermission(permission)
            )
            config.permissions.add(perm)

            # Auto-configure related features
            if perm == PluginPermission.NETWORK:
                config.network_allowed = True
            if perm in (PluginPermission.FILE_READ, PluginPermission.FILE_WRITE):
                # Ensure at least one allowed path
                if not config.allowed_paths:
                    config.allowed_paths.add(Path.cwd())
                    self._path_guards[plugin_name].add_path(Path.cwd())

            self._audit("grant_permission", plugin_name, permission=str(perm))

        logger.info(
            "Granted permission %s to plugin %s",
            perm.value,
            plugin_name,
        )

    def revoke_permission(
        self,
        plugin_name: str,
        permission: Union[PluginPermission, str],
    ) -> None:
        """Revoke a permission from a plugin.

        Args:
            plugin_name: The plugin to revoke the permission from.
            permission: A :class:`PluginPermission` enum or string value.
        """
        with self._lock:
            config = self._configs.get(plugin_name)
            if config is None:
                return
            perm = (
                permission
                if isinstance(permission, PluginPermission)
                else PluginPermission(permission)
            )
            config.permissions.discard(perm)

            if perm == PluginPermission.NETWORK:
                config.network_allowed = False

            self._audit("revoke_permission", plugin_name, permission=str(perm))

        logger.info(
            "Revoked permission %s from plugin %s",
            perm.value,
            plugin_name,
        )

    def get_permissions(self, plugin_name: str) -> Set[PluginPermission]:
        """Return the set of permissions granted to a plugin."""
        with self._lock:
            config = self._configs.get(plugin_name)
            return set(config.permissions) if config else set()

    def grant_all_permissions(self, plugin_name: str) -> None:
        """Grant all permissions to a plugin (trusted mode)."""
        for perm in PluginPermission:
            self.grant_permission(plugin_name, perm)

    def revoke_all_permissions(self, plugin_name: str) -> None:
        """Revoke all permissions from a plugin."""
        for perm in list(self.get_permissions(plugin_name)):
            self.revoke_permission(plugin_name, perm)

    # -- Path management ---------------------------------------------------

    def add_allowed_path(
        self,
        plugin_name: str,
        path: Union[str, Path],
    ) -> None:
        """Add a directory to the plugin's filesystem allow-list."""
        with self._lock:
            config = self._get_or_create_config(plugin_name)
            resolved = Path(path).resolve()
            config.allowed_paths.add(resolved)
            self._path_guards[plugin_name].add_path(resolved)
            self._audit("add_allowed_path", plugin_name, path=str(resolved))

    def remove_allowed_path(
        self,
        plugin_name: str,
        path: Union[str, Path],
    ) -> None:
        """Remove a directory from the plugin's filesystem allow-list."""
        with self._lock:
            config = self._configs.get(plugin_name)
            if config is None:
                return
            resolved = Path(path).resolve()
            config.allowed_paths.discard(resolved)
            self._path_guards[plugin_name].remove_path(resolved)
            self._audit("remove_allowed_path", plugin_name, path=str(resolved))

    def get_allowed_paths(self, plugin_name: str) -> Set[Path]:
        """Return the set of allowed paths for a plugin."""
        with self._lock:
            config = self._configs.get(plugin_name)
            return set(config.allowed_paths) if config else set()

    # -- Resource limits ---------------------------------------------------

    def set_resource_limits(
        self,
        plugin_name: str,
        limits: ResourceLimits,
    ) -> None:
        """Set resource limits for a plugin."""
        with self._lock:
            self._get_or_create_config(plugin_name)
            self._resource_monitors[plugin_name] = ResourceMonitor(plugin_name, limits)
            self._configs[plugin_name].resource_limits = limits
            self._audit("set_resource_limits", plugin_name, limits=limits.to_dict())

    def get_resource_limits(self, plugin_name: str) -> ResourceLimits:
        """Return the resource limits for a plugin."""
        with self._lock:
            config = self._configs.get(plugin_name)
            return config.resource_limits if config else ResourceLimits.default()

    # -- Code execution ----------------------------------------------------

    def execute(
        self,
        plugin_name: str,
        code: str,
        *,
        permissions: Optional[Set[PluginPermission]] = None,
        timeout: Optional[float] = None,
        allowed_paths: Optional[Set[Path]] = None,
    ) -> ExecutionResult:
        """Execute code in the plugin's sandbox.

        Args:
            plugin_name: The plugin on whose behalf the code runs.
            code: Python source code to execute.
            permissions: Override permissions (if ``None``, use the plugin's
                         stored permissions).
            timeout: Override wall-clock timeout in seconds.
            allowed_paths: Override allowed paths.

        Returns:
            An :class:`ExecutionResult` with the outcome.
        """
        start = time.monotonic()

        with self._lock:
            config = self._get_or_create_config(plugin_name)

        # Apply overrides
        effective_perms = permissions if permissions is not None else config.permissions
        if allowed_paths is not None:
            for p in allowed_paths:
                self._path_guards[plugin_name].add_path(p)

        # Check SHELL permission for exec
        if PluginPermission.SHELL not in effective_perms:
            # Still allow execution of the code itself (this IS the sandbox),
            # but warn if code looks like it tries to exec
            if any(dangerous in code for dangerous in ("subprocess", "os.system", "os.popen", "exec(")):
                self._audit(
                    "execute_warning",
                    plugin_name,
                    message="Code may attempt subprocess execution without SHELL permission",
                )

        # Set up execution environment
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_open = builtins.open

        monitor = self._resource_monitors.get(
            plugin_name,
            ResourceMonitor(plugin_name, self._default_limits),
        )
        if timeout is not None:
            effective_limits = ResourceLimits(
                max_memory_mb=config.resource_limits.max_memory_mb,
                max_cpu_seconds=config.resource_limits.max_cpu_seconds,
                max_wall_seconds=timeout,
                max_open_files=config.resource_limits.max_open_files,
                max_process_threads=config.resource_limits.max_process_threads,
            )
            monitor = ResourceMonitor(plugin_name, effective_limits)

        import_guard = ImportGuard(
            plugin_name=plugin_name,
            denied=config.denied_modules,
        )

        try:
            # Capture stdout/stderr
            if config.stdout_capture:
                sys.stdout = stdout_capture
            if config.stderr_capture:
                sys.stderr = stderr_capture

            # Install path guard
            path_guard = self._path_guards.get(plugin_name, PathGuard(plugin_name))
            builtins.open = path_guard.wrap_open(old_open)

            # Install import guard
            import_guard.install()

            # Start resource monitoring
            monitor.start()

            # Execute
            exec_globals: Dict[str, Any] = {
                "__builtins__": builtins,
                "__name__": f"sandbox:{plugin_name}",
                "__plugin_name__": plugin_name,
            }
            exec_result: Any = None

            # Check wall-clock timeout periodically via threading
            def timeout_watcher() -> None:
                while monitor._start_time is not None:
                    try:
                        monitor.check_wall_timeout()
                    except PluginSandboxError:
                        raise
                    time.sleep(0.1)

            watcher: Optional[threading.Thread] = None
            timed_out = False

            if config.resource_limits.max_wall_seconds > 0 or timeout is not None:
                watcher = threading.Thread(target=timeout_watcher, daemon=True)
                watcher.start()

            try:
                exec(compile(code, f"<sandbox:{plugin_name}>", "exec"), exec_globals)
                exec_result = exec_globals.get("__return__")
            except PluginSandboxError as exc:
                if "wall-clock limit" in str(exc):
                    timed_out = True
                raise

            wall, mem = monitor.stop()

            self._audit(
                "execute",
                plugin_name,
                success=True,
                duration_ms=(time.monotonic() - start) * 1000,
                memory_mb=mem,
            )

            return ExecutionResult(
                success=True,
                return_value=exec_result,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time_ms=wall * 1000,
                memory_used_mb=mem,
                timed_out=False,
            )

        except PluginSandboxError as exc:
            monitor.stop()
            elapsed = (time.monotonic() - start) * 1000
            self._audit(
                "execute",
                plugin_name,
                success=False,
                error=str(exc),
                duration_ms=elapsed,
            )
            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error=str(exc),
                execution_time_ms=elapsed,
                timed_out="wall-clock" in str(exc),
            )

        except Exception as exc:
            monitor.stop()
            elapsed = (time.monotonic() - start) * 1000
            tb = traceback.format_exc()
            self._audit(
                "execute",
                plugin_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed,
            )
            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue() + "\n" + tb,
                error=f"{type(exc).__name__}: {exc}",
                execution_time_ms=elapsed,
            )

        finally:
            # Restore everything
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            builtins.open = old_open
            import_guard.uninstall()
            if watcher:
                watcher.join(timeout=1.0)

    def execute_function(
        self,
        plugin_name: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a Python function in the plugin's sandbox.

        This is safer than :meth:`execute` because the function is a
        proper callable rather than arbitrary source code.
        """
        # Serialize the call into an exec string
        import inspect
        source = inspect.getsource(func)
        module = inspect.getmodule(func)
        func_name = func.__name__

        # Build code that defines and calls the function
        code = f"{source}\n__return__ = {func_name}(*__args__, **__kwargs__)"

        return self.execute(
            plugin_name,
            code,
            timeout=kwargs.pop("_timeout", None),
        )

    # -- Configuration management ------------------------------------------

    def configure(
        self,
        plugin_name: str,
        config: SandboxConfig,
    ) -> None:
        """Apply a full sandbox configuration to a plugin."""
        with self._lock:
            self._configs[plugin_name] = config
            self._path_guards[plugin_name] = PathGuard(
                plugin_name, config.allowed_paths
            )
            self._resource_monitors[plugin_name] = ResourceMonitor(
                plugin_name, config.resource_limits
            )
            self._audit("configure", plugin_name, config=str(config))

    def reset(self, plugin_name: str) -> None:
        """Remove all sandbox configuration for a plugin."""
        with self._lock:
            self._configs.pop(plugin_name, None)
            self._path_guards.pop(plugin_name, None)
            self._resource_monitors.pop(plugin_name, None)
            self._import_guards.pop(plugin_name, None)
            self._audit("reset", plugin_name)

    def reset_all(self) -> None:
        """Remove all sandbox configurations."""
        with self._lock:
            self._configs.clear()
            self._path_guards.clear()
            self._resource_monitors.clear()
            self._import_guards.clear()
        logger.info("All sandbox configurations reset")

    # -- Audit logging -----------------------------------------------------

    def _audit(self, action: str, plugin_name: str, **kwargs: Any) -> None:
        """Record an audit log entry."""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "plugin_name": plugin_name,
            **kwargs,
        }
        self._audit_log.append(entry)
        if len(self._audit_log) > 10000:
            # Keep only the last 10000 entries
            self._audit_log = self._audit_log[-5000:]

    def get_audit_log(
        self,
        plugin_name: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return audit log entries, optionally filtered.

        Args:
            plugin_name: Filter by plugin name.
            action: Filter by action type.
            limit: Maximum entries to return.
        """
        entries = self._audit_log
        if plugin_name:
            entries = [e for e in entries if e.get("plugin_name") == plugin_name]
        if action:
            entries = [e for e in entries if e.get("action") == action]
        return entries[-limit:]

    # -- Introspection -----------------------------------------------------

    def list_configured_plugins(self) -> List[str]:
        """Return names of all plugins with sandbox configurations."""
        return sorted(self._configs.keys())

    def get_config(self, plugin_name: str) -> Optional[SandboxConfig]:
        """Return the sandbox config for a plugin."""
        return self._configs.get(plugin_name)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of sandbox state."""
        return {
            "configured_plugins": len(self._configs),
            "audit_entries": len(self._audit_log),
            "plugins": {
                name: {
                    "permissions": [str(p) for p in cfg.permissions],
                    "allowed_paths": len(cfg.allowed_paths),
                    "limits": cfg.resource_limits.to_dict(),
                    "network_allowed": cfg.network_allowed,
                }
                for name, cfg in self._configs.items()
            },
        }

    def __repr__(self) -> str:
        return f"<PluginSandbox plugins={len(self._configs)} audit={len(self._audit_log)}>"
