"""
Atlas Security — Sandbox Execution Environment.

Provides isolated code execution with resource limits, supporting multiple
sandbox types including Docker-based and process-based isolation. Ensures
safe execution of untrusted code with configurable memory, CPU, and time
limits.

Inspired by OpenClaw's security sandbox architecture.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# resource module is Unix-only; handle gracefully
try:
    import resource as _resource_module  # type: ignore
    HAS_RESOURCE_MODULE = True
except ImportError:
    _resource_module = None  # type: ignore
    HAS_RESOURCE_MODULE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SandboxType(Enum):
    """Types of sandbox isolation available."""

    NONE = "none"
    DOCKER = "docker"
    PROCESS = "process"
    RESTRICTED_PATH = "restricted_path"


class ExecutionStatus(Enum):
    """Status of a sandbox execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    ERROR = "error"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution.

    Attributes:
        max_memory_mb: Maximum memory usage in megabytes.
        max_cpu_percent: Maximum CPU usage percentage (0-100).
        max_time_seconds: Maximum wall-clock execution time in seconds.
        max_output_bytes: Maximum stdout/stderr output in bytes.
        max_processes: Maximum number of child processes.
        tmpfs_size_mb: Size of tmpfs mount for /tmp in megabytes.
    """

    max_memory_mb: int = 512
    max_cpu_percent: int = 50
    max_time_seconds: int = 300
    max_output_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_processes: int = 100
    tmpfs_size_mb: int = 64

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "max_time_seconds": self.max_time_seconds,
            "max_output_bytes": self.max_output_bytes,
            "max_processes": self.max_processes,
            "tmpfs_size_mb": self.tmpfs_size_mb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceLimits":
        """Deserialize from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass
class ExecutionResult:
    """Result of a sandbox execution.

    Attributes:
        status: Final execution status.
        exit_code: Process exit code (if available).
        stdout: Standard output captured.
        stderr: Standard error captured.
        return_value: Return value (if applicable).
        duration_seconds: Wall-clock execution time in seconds.
        peak_memory_mb: Peak memory usage in MB (if measurable).
        timed_out: Whether the execution timed out.
        error_message: Error message if execution failed.
        metadata: Additional execution metadata.
    """

    status: ExecutionStatus = ExecutionStatus.PENDING
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    duration_seconds: float = 0.0
    peak_memory_mb: Optional[float] = None
    timed_out: bool = False
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "status": self.status.value,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:self.metadata.get("max_output_preview", 10000)],
            "stderr": self.stderr[:self.metadata.get("max_output_preview", 10000)],
            "duration_seconds": round(self.duration_seconds, 3),
            "peak_memory_mb": self.peak_memory_mb,
            "timed_out": self.timed_out,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @property
    def success(self) -> bool:
        """Whether the execution completed successfully."""
        return self.status == ExecutionStatus.COMPLETED and self.exit_code == 0


@dataclass
class DockerConfig:
    """Configuration for Docker-based sandbox.

    Attributes:
        image: Docker image to use for execution.
        auto_remove: Whether to remove the container after execution.
        network_mode: Docker network mode (none, bridge, host).
        read_only: Whether to mount the filesystem as read-only.
        user: User to run as inside the container.
        extra_mounts: Additional volume mounts (host:container).
        security_opts: Docker security options.
    """

    image: str = "python:3.11-slim"
    auto_remove: bool = True
    network_mode: str = "none"
    read_only: bool = True
    user: str = "nobody"
    extra_mounts: List[str] = field(default_factory=list)
    security_opts: List[str] = field(default_factory=lambda: [
        "--security-opt=no-new-privileges",
    ])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "image": self.image,
            "auto_remove": self.auto_remove,
            "network_mode": self.network_mode,
            "read_only": self.read_only,
            "user": self.user,
            "extra_mounts": self.extra_mounts,
            "security_opts": self.security_opts,
        }


# ---------------------------------------------------------------------------
# Sandbox Executor
# ---------------------------------------------------------------------------

class SandboxExecutor:
    """Sandboxed code execution engine.

    Supports multiple isolation strategies:
    - **NONE**: No isolation (for testing only)
    - **DOCKER**: Full container-based isolation
    - **PROCESS**: Subprocess-based isolation with resource limits
    - **RESTRICTED_PATH**: Chroot-like path restriction

    Usage::

        executor = SandboxExecutor(
            sandbox_type=SandboxType.PROCESS,
            default_limits=ResourceLimits(max_time_seconds=30),
        )

        result = await executor.execute(
            command=["python3", "-c", "print('hello')"],
            cwd="/tmp/workspace",
        )

        if result.success:
            print(result.stdout)
        else:
            print(f"Error: {result.error_message}")
    """

    def __init__(
        self,
        sandbox_type: SandboxType = SandboxType.PROCESS,
        default_limits: Optional[ResourceLimits] = None,
        docker_config: Optional[DockerConfig] = None,
        working_directory: Optional[str] = None,
    ) -> None:
        """Initialize the sandbox executor.

        Args:
            sandbox_type: Type of sandbox isolation to use.
            default_limits: Default resource limits for executions.
            docker_config: Docker configuration (for DOCKER type).
            working_directory: Default working directory.
        """
        self._sandbox_type = sandbox_type
        self._default_limits = default_limits or ResourceLimits()
        self._docker_config = docker_config or DockerConfig()
        self._working_directory = working_directory or tempfile.gettempdir()
        self._active_processes: Dict[str, subprocess.Popen] = {}
        self._stats = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "timeout": 0,
            "memory_exceeded": 0,
        }
        self._docker_available: Optional[bool] = None

        logger.info(
            "SandboxExecutor initialized (type=%s, workdir=%s)",
            sandbox_type.value, self._working_directory,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sandbox_type(self) -> SandboxType:
        """Current sandbox type."""
        return self._sandbox_type

    @property
    def stats(self) -> Dict[str, Any]:
        """Execution statistics."""
        return self._stats.copy()

    # ------------------------------------------------------------------
    # Availability Check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if the configured sandbox type is available.

        Returns:
            True if the sandbox type can be used.
        """
        if self._sandbox_type == SandboxType.NONE:
            return True
        elif self._sandbox_type == SandboxType.DOCKER:
            return self._check_docker_available()
        elif self._sandbox_type == SandboxType.PROCESS:
            return True
        elif self._sandbox_type == SandboxType.RESTRICTED_PATH:
            return True
        return False

    def _check_docker_available(self) -> bool:
        """Check if Docker is available and running.

        Returns:
            True if Docker CLI is available and daemon is responsive.
        """
        if self._docker_available is not None:
            return self._docker_available

        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            self._docker_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            self._docker_available = False

        if not self._docker_available:
            logger.warning("Docker is not available, falling back to process isolation")
        return self._docker_available

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        limits: Optional[ResourceLimits] = None,
        stdin_data: Optional[str] = None,
        timeout_override: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute a command in the sandbox.

        Args:
            command: Command and arguments to execute.
            cwd: Working directory (overrides default).
            env: Environment variables (overrides default).
            limits: Resource limits (overrides default).
            stdin_data: Data to pipe to stdin.
            timeout_override: Override max_time_seconds.

        Returns:
            ExecutionResult with output and status.
        """
        effective_limits = limits or self._default_limits
        if timeout_override:
            effective_limits = ResourceLimits(
                max_memory_mb=effective_limits.max_memory_mb,
                max_cpu_percent=effective_limits.max_cpu_percent,
                max_time_seconds=timeout_override,
                max_output_bytes=effective_limits.max_output_bytes,
                max_processes=effective_limits.max_processes,
                tmpfs_size_mb=effective_limits.tmpfs_size_mb,
            )

        working_dir = cwd or self._working_directory
        self._stats["total_executions"] += 1

        logger.info(
            "Sandbox execute: %s (type=%s, timeout=%ds)",
            " ".join(command[:3]),
            self._sandbox_type.value,
            effective_limits.max_time_seconds,
        )

        # Dispatch to appropriate backend
        if self._sandbox_type == SandboxType.DOCKER:
            result = await self._execute_docker(
                command, working_dir, env, effective_limits, stdin_data,
            )
        elif self._sandbox_type == SandboxType.PROCESS:
            result = await self._execute_process(
                command, working_dir, env, effective_limits, stdin_data,
            )
        elif self._sandbox_type == SandboxType.RESTRICTED_PATH:
            result = await self._execute_restricted(
                command, working_dir, env, effective_limits, stdin_data,
            )
        else:
            result = await self._execute_none(
                command, working_dir, env, effective_limits, stdin_data,
            )

        # Update stats
        if result.success:
            self._stats["successful"] += 1
        elif result.timed_out:
            self._stats["timeout"] += 1
        elif result.status == ExecutionStatus.MEMORY_EXCEEDED:
            self._stats["memory_exceeded"] += 1
        else:
            self._stats["failed"] += 1

        return result

    # ------------------------------------------------------------------
    # Docker Backend
    # ------------------------------------------------------------------

    async def _execute_docker(
        self,
        command: List[str],
        cwd: str,
        env: Optional[Dict[str, str]],
        limits: ResourceLimits,
        stdin_data: Optional[str],
    ) -> ExecutionResult:
        """Execute command in a Docker container.

        Provides strong isolation with resource limits via Docker's
        built-in cgroup enforcement.
        """
        if not self._check_docker_available():
            logger.warning("Docker not available, falling back to process isolation")
            return await self._execute_process(command, cwd, env, limits, stdin_data)

        start_time = time.monotonic()
        cfg = self._docker_config

        docker_cmd = [
            "docker", "run",
            "--rm" if cfg.auto_remove else "",
            f"--network={cfg.network_mode}",
            "--read-only" if cfg.read_only else "",
            f"--user={cfg.user}",
            f"--memory={limits.max_memory_mb}m",
            f"--cpus={limits.max_cpu_percent / 100.0}",
            f"--timeout={limits.max_time_seconds}",
            f"--pids-limit={limits.max_processes}",
            f"--workdir=/workspace",
            f"-v{cwd}:/workspace:ro",
        ]

        # Add tmpfs for writable /tmp
        docker_cmd.append(f"--tmpfs=/tmp:rw,size={limits.tmpfs_size_mb}m")

        # Add extra mounts
        for mount in cfg.extra_mounts:
            docker_cmd.append(f"-v{mount}")

        # Add security options
        for opt in cfg.security_opts:
            docker_cmd.append(opt)

        # Add environment variables
        if env:
            for key, value in env.items():
                docker_cmd.extend(["-e", f"{key}={value}"])

        # Add image and command
        docker_cmd.append(cfg.image)
        docker_cmd.extend(command)

        # Filter empty strings
        docker_cmd = [arg for arg in docker_cmd if arg]

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin_data else asyncio.subprocess.DEVNULL,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(
                        input=stdin_data.encode("utf-8") if stdin_data else None
                    ),
                    timeout=limits.max_time_seconds,
                )

                duration = time.monotonic() - start_time
                stdout = stdout_bytes.decode("utf-8", errors="replace")[:limits.max_output_bytes]
                stderr = stderr_bytes.decode("utf-8", errors="replace")[:limits.max_output_bytes]

                return ExecutionResult(
                    status=ExecutionStatus.COMPLETED,
                    exit_code=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                    metadata={"sandbox_type": "docker", "image": cfg.image},
                )

            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                duration = time.monotonic() - start_time
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    exit_code=-1,
                    duration_seconds=duration,
                    timed_out=True,
                    error_message=f"Execution timed out after {limits.max_time_seconds}s",
                    metadata={"sandbox_type": "docker"},
                )

        except FileNotFoundError:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message="Docker command not found",
                metadata={"sandbox_type": "docker"},
            )
        except Exception as exc:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=str(exc),
                metadata={"sandbox_type": "docker"},
            )

    # ------------------------------------------------------------------
    # Process Backend
    # ------------------------------------------------------------------

    async def _execute_process(
        self,
        command: List[str],
        cwd: str,
        env: Optional[Dict[str, str]],
        limits: ResourceLimits,
        stdin_data: Optional[str],
    ) -> ExecutionResult:
        """Execute command in a subprocess with resource limits.

        Uses setrlimit for memory limits and asyncio.wait_for for timeouts.
        Falls back gracefully if setrlimit is not available.
        """
        start_time = time.monotonic()

        # Build environment
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)
        # Strip sensitive env vars
        for key in list(exec_env.keys()):
            key_lower = key.lower()
            if any(s in key_lower for s in ("token", "key", "secret", "password")):
                exec_env[key] = "[REDACTED]"

        # Prepare resource limit pre-exec hook (Unix-only)
        preexec_fn = None
        if HAS_RESOURCE_MODULE and _resource_module is not None:
            memory_bytes = limits.max_memory_mb * 1024 * 1024
            def _set_limits() -> None:
                try:
                    _resource_module.setrlimit(
                        _resource_module.RLIMIT_AS,
                        (memory_bytes, memory_bytes),
                    )
                    _resource_module.setrlimit(
                        _resource_module.RLIMIT_NPROC,
                        (limits.max_processes, limits.max_processes),
                    )
                except (ValueError, OSError):
                    pass
            preexec_fn = _set_limits

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin_data else asyncio.subprocess.DEVNULL,
                cwd=cwd,
                env=exec_env,
                preexec_fn=preexec_fn,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(
                        input=stdin_data.encode("utf-8") if stdin_data else None
                    ),
                    timeout=limits.max_time_seconds,
                )

                duration = time.monotonic() - start_time
                stdout = stdout_bytes.decode("utf-8", errors="replace")[:limits.max_output_bytes]
                stderr = stderr_bytes.decode("utf-8", errors="replace")[:limits.max_output_bytes]

                # Check for memory errors
                status = ExecutionStatus.COMPLETED
                if proc.returncode == -9 or "Killed" in stderr or "MemoryError" in stderr:
                    status = ExecutionStatus.MEMORY_EXCEEDED

                return ExecutionResult(
                    status=status,
                    exit_code=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                    metadata={"sandbox_type": "process"},
                )

            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                duration = time.monotonic() - start_time
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    exit_code=-1,
                    duration_seconds=duration,
                    timed_out=True,
                    error_message=f"Execution timed out after {limits.max_time_seconds}s",
                    metadata={"sandbox_type": "process"},
                )

        except MemoryError:
            return ExecutionResult(
                status=ExecutionStatus.MEMORY_EXCEEDED,
                error_message=f"Memory limit exceeded ({limits.max_memory_mb} MB)",
                metadata={"sandbox_type": "process"},
            )
        except Exception as exc:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=str(exc),
                duration_seconds=duration,
                metadata={"sandbox_type": "process"},
            )

    # ------------------------------------------------------------------
    # Restricted Path Backend
    # ------------------------------------------------------------------

    async def _execute_restricted(
        self,
        command: List[str],
        cwd: str,
        env: Optional[Dict[str, str]],
        limits: ResourceLimits,
        stdin_data: Optional[str],
    ) -> ExecutionResult:
        """Execute with path-based restrictions.

        Validates that the working directory is within allowed paths
        before executing as a subprocess.
        """
        cwd_path = Path(cwd).resolve()
        workdir_path = Path(self._working_directory).resolve()

        if not str(cwd_path).startswith(str(workdir_path)):
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=(
                    f"Working directory {cwd} is outside restricted path "
                    f"{self._working_directory}"
                ),
                metadata={"sandbox_type": "restricted_path"},
            )

        # Execute with process backend but with restricted working dir
        return await self._execute_process(command, cwd, env, limits, stdin_data)

    # ------------------------------------------------------------------
    # No Sandbox Backend
    # ------------------------------------------------------------------

    async def _execute_none(
        self,
        command: List[str],
        cwd: str,
        env: Optional[Dict[str, str]],
        limits: ResourceLimits,
        stdin_data: Optional[str],
    ) -> ExecutionResult:
        """Execute without any sandboxing (testing/diagnostics only).

        Still enforces time limits for safety.
        """
        logger.warning("Executing without sandbox isolation!")

        start_time = time.monotonic()
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin_data else asyncio.subprocess.DEVNULL,
                cwd=cwd,
                env=exec_env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(
                        input=stdin_data.encode("utf-8") if stdin_data else None
                    ),
                    timeout=limits.max_time_seconds,
                )

                duration = time.monotonic() - start_time
                stdout = stdout_bytes.decode("utf-8", errors="replace")[:limits.max_output_bytes]
                stderr = stderr_bytes.decode("utf-8", errors="replace")[:limits.max_output_bytes]

                return ExecutionResult(
                    status=ExecutionStatus.COMPLETED,
                    exit_code=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                    metadata={"sandbox_type": "none"},
                )

            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                duration = time.monotonic() - start_time
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    exit_code=-1,
                    duration_seconds=duration,
                    timed_out=True,
                    error_message=f"Execution timed out after {limits.max_time_seconds}s",
                    metadata={"sandbox_type": "none"},
                )

        except Exception as exc:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=str(exc),
                duration_seconds=duration,
                metadata={"sandbox_type": "none"},
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Clean up all active sandbox resources.

        Kills any running processes and removes temporary files.
        """
        for exec_id, proc in list(self._active_processes.items()):
            try:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                logger.debug("Cleaned up process %s", exec_id)
            except Exception as exc:
                logger.warning("Failed to clean up process %s: %s", exec_id, exc)

        self._active_processes.clear()
        logger.info("SandboxExecutor cleanup complete")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive sandbox executor status.

        Returns:
            Dictionary with availability, type, limits, and stats.
        """
        return {
            "sandbox_type": self._sandbox_type.value,
            "available": self.is_available(),
            "working_directory": self._working_directory,
            "default_limits": self._default_limits.to_dict(),
            "stats": self._stats.copy(),
            "active_processes": len(self._active_processes),
            "docker_available": self._docker_available,
        }

    def __repr__(self) -> str:
        return (
            f"<SandboxExecutor type={self._sandbox_type.value} "
            f"available={self.is_available()}>"
        )
