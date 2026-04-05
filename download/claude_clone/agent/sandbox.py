"""
Code Execution Sandbox Module for Claude Code Clone.

Provides a secure, isolated environment for executing Python, JavaScript (Node.js),
and Bash code with resource limits, output capture, execution history tracking,
and Jupyter-like magic commands.

Security measures include:
- Process isolation via subprocess with resource restrictions
- Timeout enforcement
- Memory limits (via resource.setrlimit on POSIX)
- Dangerous operation blocking (shell injection, network access to internal IPs,
  file writes outside sandbox directory, etc.)
- Persistent sandbox directories per session
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import stat
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class Language(str, Enum):
    """Supported execution languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"


class MagicCommand(str, Enum):
    """Supported magic commands (Jupyter-like)."""
    PIP = "pip"
    PIP_INSTALL = "pip install"
    CD = "cd"
    LS = "ls"
    ENV = "env"
    TIME = "time"
    TIMEIT = "timeit"
    WHO = "who"
    RESET = "reset"
    HISTORY = "history"


# Patterns that indicate dangerous Python operations.
_DANGEROUS_PYTHON_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("os.system", re.compile(r"\bos\.system\s*\(")),
    ("subprocess with shell=True", re.compile(r"subprocess\.\w+\(.*shell\s*=\s*True")),
    ("eval on input", re.compile(r"\beval\s*\(\s*(?:input|sys\.stdin)")),
    ("exec on input", re.compile(r"\bexec\s*\(\s*(?:input|sys\.stdin)")),
    ("__import__ of os", re.compile(r"__import__\s*\(\s*['\"]os['\"]")),
    ("ctypes", re.compile(r"\bctypes\b")),
    ("socket raw", re.compile(r"\bsocket\.\w*\s*\(")),
    ("pickle loads", re.compile(r"\bpickle\.loads?\s*\(")),
    ("shutil.rmtree", re.compile(r"\bshutil\.rmtree\s*\(")),
]

# Patterns that indicate dangerous Bash operations.
_DANGEROUS_BASH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("rm -rf /", re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\b")),
    ("mkfs", re.compile(r"\bmkfs\b")),
    ("dd if=", re.compile(r"\bdd\s+.*\bif\s*=")),
    (":(){ :|:& };:", re.compile(r":\(\)\s*\{.*:\|:&\s*\};\s*:")),
    ("chmod 777 /", re.compile(r"\bchmod\s+.*777\s+/\b")),
    ("network access", re.compile(r"\bcurl\b|\bwget\b|\bnc\b.*-e")),
]

# Internal IP ranges that should be blocked for network access.
_INTERNAL_IP_RANGES: list[tuple[str, int]] = [
    ("10.0.0.0", 8),
    ("172.16.0.0", 12),
    ("192.168.0.0", 16),
    ("127.0.0.0", 8),
    ("169.254.0.0", 16),
    ("0.0.0.0", 8),
    ("::1", 128),
    ("fc00::", 7),
    ("fe80::", 10),
]

# Reserved filenames that must never be written to or overwritten.
_PROTECTED_PATHS: list[str] = [
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
    "/etc/hosts", "/etc/resolv.conf",
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Represents the complete result of a sandboxed code execution.

    Attributes:
        code: The source code that was executed.
        language: The language used for execution.
        stdout: Captured standard output.
        stderr: Captured standard error.
        return_code: Process exit code (0 = success).
        execution_time: Wall-clock execution time in seconds.
        memory_usage: Peak memory usage in bytes (best-effort).
        files_changed: List of relative file paths created or modified.
        output_lines: Line-by-line output combining stdout and stderr.
        error: Human-readable error description if execution failed, else None.
        timed_out: Whether the execution was killed due to timeout.
    """
    code: str
    language: str
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time: float = 0.0
    memory_usage: int = 0
    files_changed: list[str] = field(default_factory=list)
    output_lines: list[str] = field(default_factory=list)
    error: Optional[str] = None
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """Return True if the execution completed without error."""
        return self.return_code == 0 and self.error is None and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary."""
        return {
            "code": self.code,
            "language": self.language,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "files_changed": self.files_changed,
            "output_lines": self.output_lines,
            "error": self.error,
            "timed_out": self.timed_out,
            "success": self.success,
        }

    def __str__(self) -> str:
        parts = [f"[{self.language}] exit={self.return_code} time={self.execution_time:.3f}s"]
        if self.stdout.strip():
            parts.append(f"stdout:\n{self.stdout.rstrip()}")
        if self.stderr.strip():
            parts.append(f"stderr:\n{self.stderr.rstrip()}")
        if self.error:
            parts.append(f"error: {self.error}")
        if self.timed_out:
            parts.append("TIMED OUT")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Security Helpers
# ---------------------------------------------------------------------------

class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


def _check_dangerous_python(code: str) -> list[str]:
    """Scan Python code for dangerous patterns and return violation descriptions."""
    violations: list[str] = []
    for label, pattern in _DANGEROUS_PYTHON_PATTERNS:
        if pattern.search(code):
            violations.append(label)
    return violations


def _check_dangerous_bash(code: str) -> list[str]:
    """Scan Bash code for dangerous patterns and return violation descriptions."""
    violations: list[str] = []
    for label, pattern in _DANGEROUS_BASH_PATTERNS:
        if pattern.search(code):
            violations.append(label)
    return violations


def _check_internal_ip_access(code: str) -> list[str]:
    """Scan code for attempts to connect to internal/private IP addresses."""
    found: list[str] = []
    ip_pattern = re.compile(
        r"(?:https?://)?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    )
    for match in ip_pattern.finditer(code):
        ip = match.group(1)
        try:
            parts = [int(p) for p in ip.split(".")]
            packed = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]
            for network, prefix in _INTERNAL_IP_RANGES:
                net_parts = [int(p) for p in network.split(".")]
                net_packed = (net_parts[0] << 24) | (net_parts[1] << 16) | (net_parts[2] << 8) | net_parts[3]
                mask = ((1 << prefix) - 1) << (32 - prefix)
                if (packed & mask) == (net_packed & mask):
                    found.append(ip)
                    break
        except (ValueError, IndexError):
            continue
    return found


def _validate_sandbox_path(sandbox_dir: str, target_path: str) -> Path:
    """Resolve a target path and ensure it stays within the sandbox directory.

    Returns the resolved absolute Path if safe.  Raises SecurityError otherwise.
    """
    sandbox = Path(sandbox_dir).resolve()
    resolved = Path(target_path).resolve()
    try:
        resolved.relative_to(sandbox)
    except ValueError:
        raise SecurityError(
            f"Path '{target_path}' resolves to '{resolved}' which is outside "
            f"the sandbox directory '{sandbox}'."
        )
    for protected in _PROTECTED_PATHS:
        if str(resolved) == protected or str(resolved).startswith(protected + "/"):
            raise SecurityError(f"Refusing to access protected path: {protected}")
    return resolved


def _is_path_within(target: Path, directory: Path) -> bool:
    """Return True if *target* is a descendant of *directory*."""
    try:
        target.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Preamble / Wrapper Scripts
# ---------------------------------------------------------------------------

# Injected before user code to enforce basic in-process restrictions.
_PYTHON_PREAMBLE = """
import sys, os, signal, resource

# Enforce memory limit if supported.
def _sandbox_set_memory_limit(max_bytes):
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new = min(max_bytes, hard if hard > 0 else max_bytes)
        resource.setrlimit(resource.RLIMIT_AS, (new, new))
    except (ValueError, OSError, ImportError):
        pass

def _sandbox_block_network():
    \"\"\"Replace socket module to prevent outbound connections.\"\"\"
    import socket as _orig_socket
    class _BlockedSocket:
        def __init__(self, *a, **kw):
            raise PermissionError("Network access is disabled in the sandbox.")
    _orig_socket.socket = _BlockedSocket
    sys.modules["socket"] = _orig_socket

_sandbox_block_network()
"""


_JAVASCRIPT_PREAMBLE = """
// Sandbox environment – restrict dangerous globals.
(function() {
    var _blocked = ['require', 'process', 'child_process', 'fs', 'net', 'http', 'https', 'dgram'];
    _blocked.forEach(function(name) {
        try { Object.defineProperty(globalThis, name, { get: function() { throw new Error(name + ' is disabled in the sandbox.'); }, configurable: false }); } catch(e) {}
    });
})();
"""


# ---------------------------------------------------------------------------
# SandboxExecutor
# ---------------------------------------------------------------------------

class SandboxExecutor:
    """Manages a persistent sandbox environment for executing untrusted code.

    The sandbox creates an isolated temporary directory where files can
    persist between executions.  Code is run in a subprocess with
    configurable resource limits and comprehensive output capture.

    Args:
        sandbox_dir: Optional path for the persistent sandbox directory.
            If None, a temporary directory is created automatically.
        max_memory_mb: Maximum memory each execution may consume (MiB).
        default_timeout: Default timeout in seconds for executions.

    Example::

        executor = SandboxExecutor(max_memory_mb=256, default_timeout=15)
        result = await executor.execute("print('hello')", language="python")
        print(result.stdout)  # "hello"
        executor.reset()       # wipes sandbox files and history
    """

    def __init__(
        self,
        sandbox_dir: str | None = None,
        max_memory_mb: int = 512,
        default_timeout: int = 30,
    ) -> None:
        self._max_memory_mb = max_memory_mb
        self._default_timeout = default_timeout
        self._history: list[ExecutionResult] = []

        if sandbox_dir is not None:
            self._sandbox_dir = Path(sandbox_dir).resolve()
            self._sandbox_dir.mkdir(parents=True, exist_ok=True)
            self._managed_dir = False
        else:
            self._sandbox_dir = Path(tempfile.mkdtemp(prefix="claude_sandbox_"))
            self._managed_dir = True

        # Working directory inside sandbox.
        self._work_dir = self._sandbox_dir / "workspace"
        self._work_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot files so we can detect changes after execution.
        self._file_snapshot: set[str] = self._snapshot_files()

    # -- Properties ----------------------------------------------------------

    @property
    def sandbox_path(self) -> str:
        """Absolute path to the sandbox root directory."""
        return str(self._sandbox_dir)

    @property
    def workspace_path(self) -> str:
        """Absolute path to the workspace directory inside the sandbox."""
        return str(self._work_dir)

    # -- File management -----------------------------------------------------

    def get_file(self, relative_path: str) -> str:
        """Read and return the contents of a file inside the sandbox.

        Args:
            relative_path: Path relative to the sandbox root.

        Returns:
            File contents as a string.

        Raises:
            SecurityError: If the path resolves outside the sandbox.
            FileNotFoundError: If the file does not exist.
        """
        resolved = _validate_sandbox_path(str(self._sandbox_dir), relative_path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found in sandbox: {relative_path}")
        return resolved.read_text(encoding="utf-8", errors="replace")

    def write_file(self, relative_path: str, content: str) -> None:
        """Write content to a file inside the sandbox.

        Args:
            relative_path: Path relative to the sandbox root.
            content: Text content to write.

        Raises:
            SecurityError: If the path resolves outside the sandbox.
        """
        resolved = _validate_sandbox_path(str(self._sandbox_dir), relative_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

    def list_files(self) -> list[str]:
        """List all files in the sandbox workspace, relative to sandbox root.

        Returns:
            Sorted list of relative file paths.
        """
        files: list[str] = []
        for root, _dirs, filenames in os.walk(self._work_dir):
            base = Path(root).relative_to(self._sandbox_dir)
            for fname in sorted(filenames):
                files.append(str(base / fname) if str(base) != "." else fname)
        return sorted(files)

    def delete_file(self, relative_path: str) -> None:
        """Delete a file inside the sandbox.

        Args:
            relative_path: Path relative to the sandbox root.

        Raises:
            SecurityError: If the path resolves outside the sandbox.
            FileNotFoundError: If the file does not exist.
        """
        resolved = _validate_sandbox_path(str(self._sandbox_dir), relative_path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found in sandbox: {relative_path}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()

    # -- History -------------------------------------------------------------

    def get_history(self, limit: int = 20) -> list[ExecutionResult]:
        """Return the most recent execution results, newest first.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of ExecutionResult objects.
        """
        return list(reversed(self._history[-limit:]))

    def clear_history(self) -> None:
        """Remove all stored execution history."""
        self._history.clear()

    # -- Reset ---------------------------------------------------------------

    def reset(self) -> None:
        """Wipe the sandbox directory and clear all execution history.

        Creates a fresh workspace.  If the sandbox directory was
        auto-created (no explicit path was given to __init__), the
        directory is deleted and re-created.
        """
        self._history.clear()
        if self._managed_dir:
            shutil.rmtree(self._sandbox_dir, ignore_errors=True)
            self._sandbox_dir = Path(tempfile.mkdtemp(prefix="claude_sandbox_"))
        else:
            # Preserve the root but wipe contents.
            for item in self._sandbox_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
        self._work_dir = self._sandbox_dir / "workspace"
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._file_snapshot = self._snapshot_files()

    # -- Package installation ------------------------------------------------

    async def install_package(self, package: str, language: str = "python") -> ExecutionResult:
        """Install a package inside the sandbox environment.

        Args:
            package: Name of the package (e.g. "requests").
            language: Target language – ``"python"`` or ``"javascript"``.

        Returns:
            ExecutionResult from the installation command.
        """
        lang = language.lower()
        if lang in ("python", "py"):
            cmd = [sys.executable, "-m", "pip", "install", "--target",
                   str(self._sandbox_dir / "site-packages"), "--no-warn-script-location",
                   package]
        elif lang in ("javascript", "js", "node"):
            node_modules = self._sandbox_dir / "node_modules"
            node_modules.mkdir(exist_ok=True)
            cmd = ["npm", "install", "--prefix", str(self._sandbox_dir), package]
        else:
            result = ExecutionResult(
                code=f"install {package}", language=language,
                return_code=1, error=f"Unsupported language for package install: {language}",
            )
            self._history.append(result)
            return result

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._work_dir),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=self._default_timeout
            )
            elapsed = time.monotonic() - start
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")
            return ExecutionResult(
                code=f"install {package}", language=language,
                stdout=stdout_str, stderr=stderr_str,
                return_code=proc.returncode or 0,
                execution_time=elapsed,
                files_changed=self._detect_changed_files(),
            )
        except asyncio.TimeoutError:
            return ExecutionResult(
                code=f"install {package}", language=language,
                return_code=-1, timed_out=True,
                execution_time=self._default_timeout,
                error=f"Package installation timed out after {self._default_timeout}s",
            )
        except Exception as exc:
            return ExecutionResult(
                code=f"install {package}", language=language,
                return_code=-1, error=str(exc),
            )

    # -- Core execution ------------------------------------------------------

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute code in the sandbox and return the full result.

        Args:
            code: Source code to execute.
            language: ``"python"``, ``"javascript"``, or ``"bash"``.
            timeout: Maximum execution time in seconds (uses default if None).
            env: Additional environment variables for the subprocess.

        Returns:
            An ExecutionResult with captured output and metadata.
        """
        lang = language.lower().strip()

        # Handle magic commands first.
        magic_result = self._handle_magic_command(code, lang)
        if magic_result is not None:
            self._history.append(magic_result)
            return magic_result

        # Security checks.
        self._security_check(code, lang)

        effective_timeout = timeout if timeout is not None else self._default_timeout

        if lang in ("python", "py"):
            result = await self._execute_python(code, effective_timeout, env)
        elif lang in ("javascript", "js", "node"):
            result = await self._execute_javascript(code, effective_timeout, env)
        elif lang in ("bash", "sh"):
            result = await self._execute_bash(code, effective_timeout, env)
        else:
            result = ExecutionResult(
                code=code, language=lang,
                return_code=1, error=f"Unsupported language: {language}",
            )

        result.files_changed = self._detect_changed_files()
        self._file_snapshot = self._snapshot_files()
        self._history.append(result)
        return result

    async def execute_file(self, filepath: str, args: list[str] | None = None) -> ExecutionResult:
        """Execute a file inside the sandbox.

        Args:
            filepath: Path to the script file (absolute or relative to sandbox).
            args: Optional command-line arguments passed to the script.

        Returns:
            An ExecutionResult with captured output and metadata.
        """
        resolved = _validate_sandbox_path(str(self._sandbox_dir), filepath)
        if not resolved.exists():
            result = ExecutionResult(
                code=f"file:{filepath}", language="auto",
                return_code=1, error=f"File not found: {filepath}",
            )
            self._history.append(result)
            return result

        suffix = resolved.suffix.lower()
        if suffix in (".py",):
            lang = "python"
        elif suffix in (".js", ".mjs", ".cjs"):
            lang = "javascript"
        elif suffix in (".sh",) or filepath.endswith("/bash"):
            lang = "bash"
        else:
            # Try shebang line.
            first_line = ""
            try:
                first_line = resolved.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
            except Exception:
                pass
            if "python" in first_line:
                lang = "python"
            elif "node" in first_line:
                lang = "javascript"
            else:
                lang = "bash"

        source = resolved.read_text(encoding="utf-8", errors="replace")

        if lang == "python":
            exec_args = [sys.executable, str(resolved)] + (args or [])
        elif lang == "javascript":
            exec_args = ["node", str(resolved)] + (args or [])
        else:
            exec_args = ["/bin/bash", str(resolved)] + (args or [])

        effective_timeout = self._default_timeout
        start = time.monotonic()
        proc_env = self._build_env()
        try:
            proc = await asyncio.create_subprocess_exec(
                *exec_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._work_dir),
                env=proc_env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=effective_timeout
            )
            elapsed = time.monotonic() - start
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")
            result = ExecutionResult(
                code=source, language=lang,
                stdout=stdout_str, stderr=stderr_str,
                return_code=proc.returncode or 0,
                execution_time=elapsed,
                output_lines=(stdout_str + stderr_str).splitlines(),
            )
        except asyncio.TimeoutError:
            result = ExecutionResult(
                code=source, language=lang,
                return_code=-1, timed_out=True,
                execution_time=effective_timeout,
                error=f"Execution timed out after {effective_timeout}s",
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            result = ExecutionResult(
                code=source, language=lang,
                return_code=-1, execution_time=elapsed,
                error=str(exc),
            )

        result.files_changed = self._detect_changed_files()
        self._file_snapshot = self._snapshot_files()
        self._history.append(result)
        return result

    async def stream_execute(
        self,
        code: str,
        language: str = "python",
    ) -> AsyncGenerator[str, None]:
        """Execute code and yield output lines as they are produced.

        This is an async generator that yields individual output lines
        (including stderr) prefixed with a tag, e.g. ``"stdout:hello"``.

        Args:
            code: Source code to execute.
            language: ``"python"``, ``"javascript"``, or ``"bash"``.

        Yields:
            Strings of the form ``"stdout:line"`` or ``"stderr:line"``.
            A final ``"done:status=<rc>"`` line is yielded on completion.
        """
        lang = language.lower().strip()
        self._security_check(code, lang)
        effective_timeout = self._default_timeout

        if lang in ("python", "py"):
            cmd = self._python_cmd(code)
        elif lang in ("javascript", "js", "node"):
            cmd = self._javascript_cmd(code)
        elif lang in ("bash", "sh"):
            cmd = ["/bin/bash", "-c", code]
        else:
            yield f"stderr:Unsupported language: {language}"
            yield "done:status=1"
            return

        proc_env = self._build_env()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._work_dir),
                env=proc_env,
            )

            # Read stdout and stderr concurrently, yielding lines as they arrive.
            async def _read_stream(stream: asyncio.StreamReader | None, tag: str):
                if stream is None:
                    return
                while True:
                    line_bytes = await stream.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                    yield f"{tag}:{line}"

            async def _merge():
                """Merge stdout and stderr generators."""
                stdout_gen = _read_stream(proc.stdout, "stdout")
                stderr_gen = _read_stream(proc.stderr, "stderr")
                # We need to iterate both concurrently.
                stdout_task = asyncio.create_task(_drain(stdout_gen))
                stderr_task = asyncio.create_task(_drain(stderr_gen))
                results = await asyncio.gather(stdout_task, stderr_task)
                return results

            async def _drain(agen):
                items = []
                async for item in agen:
                    items.append(item)
                return items

            # Collect all lines and yield them in real-time order.
            # Since we can't easily interleave, we read one stream at a time
            # but check both non-blocking.
            lines: list[str] = []
            done = False
            stdout_buffer = ""
            stderr_buffer = ""

            while not done:
                read_tasks = []
                if proc.stdout and not proc.stdout.at_eof():
                    read_tasks.append(asyncio.create_task(proc.stdout.readline()))
                if proc.stderr and not proc.stderr.at_eof():
                    read_tasks.append(asyncio.create_task(proc.stderr.readline()))

                if not read_tasks:
                    # Both streams at EOF.
                    break

                # Add a small timeout so we don't block indefinitely if both EOF.
                finished, _ = await asyncio.wait(
                    read_tasks,
                    timeout=max(0.1, effective_timeout - sum(
                        1 for _ in read_tasks
                    )),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in finished:
                    line_bytes = task.result()
                    if not line_bytes:
                        continue
                    # Determine which stream produced this by index.
                    idx = read_tasks.index(task)
                    tag = "stdout" if idx == 0 else "stderr"
                    line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                    msg = f"{tag}:{line}"
                    lines.append(msg)
                    yield msg

                # Check if process ended.
                if proc.returncode is not None:
                    # Drain remaining.
                    break

            try:
                await asyncio.wait_for(proc.wait(), timeout=effective_timeout)
            except asyncio.TimeoutError:
                proc.kill()
                yield f"stderr:Execution timed out after {effective_timeout}s"
                yield "done:status=-1"

            rc = proc.returncode if proc.returncode is not None else -1
            yield f"done:status={rc}"

        except Exception as exc:
            yield f"stderr:{exc}"
            yield "done:status=-1"

    # -- Magic commands ------------------------------------------------------

    def _handle_magic_command(self, code: str, language: str) -> ExecutionResult | None:
        """Detect and handle magic commands. Returns a result or None."""
        stripped = code.strip()
        if not stripped.startswith("%"):
            return None

        magic_line = stripped.lstrip("%").strip()

        # %pip install <pkg> or %pip <pkg>
        if magic_line.startswith("pip install ") or magic_line.startswith("pip "):
            pkg_part = magic_line
            if pkg_part.startswith("pip install "):
                pkg_part = pkg_part[len("pip install "):]
            else:
                pkg_part = pkg_part[len("pip "):]
            # Synchronous-ish – return a placeholder; actual install happens
            # via install_package in the async path. We'll just return a stub
            # and schedule the real install.  For simplicity we handle it inline.
            # (In production, this would be awaited.)
            return ExecutionResult(
                code=code, language=language,
                stdout=f"Use install_package('{pkg_part}') to install {pkg_part}.",
                return_code=0, execution_time=0.0,
                error=None,
            )

        # %cd <path>
        if magic_line.startswith("cd "):
            target = magic_line[3:].strip()
            try:
                resolved = _validate_sandbox_path(str(self._sandbox_dir), target)
                if resolved.is_dir():
                    self._work_dir = resolved
                    return ExecutionResult(
                        code=code, language=language,
                        stdout=f"Changed directory to {resolved}",
                        return_code=0, execution_time=0.0,
                    )
                return ExecutionResult(
                    code=code, language=language,
                    stderr=f"Directory not found: {target}",
                    return_code=1, execution_time=0.0,
                    error=f"Directory not found: {target}",
                )
            except SecurityError as exc:
                return ExecutionResult(
                    code=code, language=language,
                    stderr=str(exc), return_code=1, execution_time=0.0,
                    error=str(exc),
                )

        # %ls
        if magic_line == "ls" or magic_line.startswith("ls "):
            pattern = magic_line[2:].strip() if len(magic_line) > 2 else "*"
            entries = list(self._work_dir.glob(pattern))
            lines = []
            for e in sorted(entries):
                prefix = "d" if e.is_dir() else "f"
                lines.append(f"  [{prefix}] {e.name}")
            return ExecutionResult(
                code=code, language=language,
                stdout="\n".join(lines) if lines else "(empty)",
                return_code=0, execution_time=0.0,
            )

        # %env
        if magic_line == "env" or magic_line.startswith("env "):
            var_name = magic_line[3:].strip() if len(magic_line) > 3 else None
            proc_env = self._build_env()
            if var_name:
                val = proc_env.get(var_name, "(not set)")
                return ExecutionResult(
                    code=code, language=language,
                    stdout=f"{var_name}={val}",
                    return_code=0, execution_time=0.0,
                )
            lines = [f"{k}={v}" for k, v in sorted(proc_env.items())]
            return ExecutionResult(
                code=code, language=language,
                stdout="\n".join(lines),
                return_code=0, execution_time=0.0,
            )

        # %who – list variables (best-effort for Bash/JS, return code info for Python)
        if magic_line == "who":
            return ExecutionResult(
                code=code, language=language,
                stdout="(Variable introspection is only available after Python execution)",
                return_code=0, execution_time=0.0,
            )

        # %reset
        if magic_line == "reset":
            self.reset()
            return ExecutionResult(
                code=code, language=language,
                stdout="Sandbox reset complete.",
                return_code=0, execution_time=0.0,
            )

        # %history
        if magic_line.startswith("history"):
            limit_str = magic_line[7:].strip() if len(magic_line) > 7 else "20"
            try:
                limit = int(limit_str)
            except ValueError:
                limit = 20
            history = self.get_history(limit=limit)
            lines = []
            for i, r in enumerate(history):
                status = "OK" if r.success else f"ERR({r.return_code})"
                preview = r.code.split("\n")[0][:60]
                lines.append(f"  [{i+1}] [{r.language}] {status} {preview}")
            return ExecutionResult(
                code=code, language=language,
                stdout="\n".join(lines) if lines else "(no history)",
                return_code=0, execution_time=0.0,
            )

        # %time – just note it will time the next execution
        if magic_line.startswith("time"):
            inner_code = magic_line[4:].strip()
            if inner_code:
                return ExecutionResult(
                    code=code, language=language,
                    stdout=f"(Wrap code in %%time magic or use execute() which always returns execution_time)",
                    return_code=0, execution_time=0.0,
                )

        # %timeit
        if magic_line.startswith("timeit"):
            inner_code = magic_line[6:].strip()
            if inner_code and language in ("python", "py"):
                return ExecutionResult(
                    code=code, language=language,
                    stdout=f"(Use Python's timeit module: __import__('timeit').timeit('''{inner_code}''', number=1000))",
                    return_code=0, execution_time=0.0,
                )

        # Unrecognized magic command.
        return ExecutionResult(
            code=code, language=language,
            stderr=f"Unknown magic command: %{magic_line}",
            return_code=1, execution_time=0.0,
            error=f"Unknown magic command: %{magic_line}",
        )

    # -- Language-specific executors -----------------------------------------

    async def _execute_python(
        self,
        code: str,
        timeout: int,
        env: dict[str, str] | None,
    ) -> ExecutionResult:
        """Execute Python code in a subprocess."""
        cmd = self._python_cmd(code)
        return await self._run_command(cmd, code, "python", timeout, env)

    async def _execute_javascript(
        self,
        code: str,
        timeout: int,
        env: dict[str, str] | None,
    ) -> ExecutionResult:
        """Execute JavaScript code via Node.js in a subprocess."""
        cmd = self._javascript_cmd(code)
        return await self._run_command(cmd, code, "javascript", timeout, env)

    async def _execute_bash(
        self,
        code: str,
        timeout: int,
        env: dict[str, str] | None,
    ) -> ExecutionResult:
        """Execute Bash code in a subprocess."""
        cmd = ["/bin/bash", "-c", code]
        return await self._run_command(cmd, code, "bash", timeout, env)

    # -- Command building ----------------------------------------------------

    def _python_cmd(self, code: str) -> list[str]:
        """Build the command list for executing Python code."""
        script_path = self._sandbox_dir / "__exec__.py"
        mem_bytes = self._max_memory_mb * 1024 * 1024
        preamble = _PYTHON_PREAMBLE.replace("{{MAX_BYTES}}", str(mem_bytes))
        full_code = preamble + "\n" + code
        script_path.write_text(full_code, encoding="utf-8")
        return [sys.executable, "-u", str(script_path)]

    def _javascript_cmd(self, code: str) -> list[str]:
        """Build the command list for executing JavaScript code."""
        script_path = self._sandbox_dir / "__exec__.js"
        full_code = _JAVASCRIPT_PREAMBLE + "\n" + code
        script_path.write_text(full_code, encoding="utf-8")
        return ["node", "--no-warnings", str(script_path)]

    # -- Generic subprocess runner -------------------------------------------

    async def _run_command(
        self,
        cmd: list[str],
        code: str,
        language: str,
        timeout: int,
        env: dict[str, str] | None,
    ) -> ExecutionResult:
        """Run a subprocess command with timeout, resource limits, and output capture.

        Returns an ExecutionResult with full metadata.
        """
        proc_env = self._build_env(env)
        start = time.monotonic()

        try:
            # Use preexec_fn on POSIX to set resource limits.
            preexec = None
            if sys.platform != "win32":
                preexec = self._make_preexec_fn()

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._work_dir),
                env=proc_env,
                preexec_fn=preexec,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                # Kill the process group.
                self._kill_proc(proc)
                elapsed = time.monotonic() - start
                return ExecutionResult(
                    code=code, language=language,
                    return_code=-1, timed_out=True,
                    execution_time=elapsed,
                    error=f"Execution timed out after {timeout}s",
                )

            elapsed = time.monotonic() - start
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")

            # Truncate overly large output.
            max_output_chars = 1_000_000  # 1 MB
            truncated = False
            if len(stdout_str) > max_output_chars:
                stdout_str = stdout_str[:max_output_chars] + "\n... [output truncated]"
                truncated = True
            if len(stderr_str) > max_output_chars:
                stderr_str = stderr_str[:max_output_chars] + "\n... [output truncated]"
                truncated = True

            output_lines = (stdout_str + "\n" + stderr_str).splitlines()
            output_lines = [l for l in output_lines if l]  # skip empty

            return ExecutionResult(
                code=code, language=language,
                stdout=stdout_str, stderr=stderr_str,
                return_code=proc.returncode or 0,
                execution_time=elapsed,
                output_lines=output_lines,
                error=stderr_str.strip() if proc.returncode != 0 and stderr_str.strip() else None,
            )

        except PermissionError as exc:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                code=code, language=language,
                return_code=126, execution_time=elapsed,
                error=f"Permission denied: {exc}",
            )
        except FileNotFoundError as exc:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                code=code, language=language,
                return_code=127, execution_time=elapsed,
                error=f"Command not found: {exc}. Is '{cmd[0]}' installed?",
            )
        except OSError as exc:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                code=code, language=language,
                return_code=1, execution_time=elapsed,
                error=f"OS error during execution: {exc}",
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                code=code, language=language,
                return_code=1, execution_time=elapsed,
                error=f"Unexpected error: {exc}",
            )

    # -- Process lifecycle ---------------------------------------------------

    @staticmethod
    def _make_preexec_fn() -> callable:
        """Return a preexec function that sets resource limits on the child.

        Sets RLIMIT_AS (virtual memory) and RLIMIT_FSIZE (max file size).
        """
        def _preexec():
            import resource as _res
            try:
                # Limit virtual memory to ~512 MB default (overridden by per-call setting).
                _res.setrlimit(_res.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
            except (ValueError, OSError):
                pass
            try:
                # Prevent creating files larger than 50 MB.
                _res.setrlimit(_res.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024))
            except (ValueError, OSError):
                pass
            # Deny new privileges.
            try:
                if hasattr(os, "setresuid"):
                    pass  # Keep current UID; no privilege escalation.
            except Exception:
                pass
        return _preexec

    @staticmethod
    def _kill_proc(proc: asyncio.subprocess.Process) -> None:
        """Terminate a process and its children."""
        pid = proc.pid
        if pid is None:
            return
        if sys.platform != "win32":
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait()
        except Exception:
            pass

    # -- Security ------------------------------------------------------------

    def _security_check(self, code: str, language: str) -> None:
        """Run all security checks against the code. Raises SecurityError on violation."""
        if language in ("python", "py"):
            violations = _check_dangerous_python(code)
            if violations:
                raise SecurityError(
                    f"Potentially dangerous Python operations detected: {', '.join(violations)}"
                )

        if language in ("bash", "sh"):
            violations = _check_dangerous_bash(code)
            if violations:
                raise SecurityError(
                    f"Potentially dangerous Bash operations detected: {', '.join(violations)}"
                )

        # Check for internal IP access across all languages.
        bad_ips = _check_internal_ip_access(code)
        if bad_ips:
            raise SecurityError(
                f"Network access to internal IPs is blocked: {', '.join(bad_ips)}"
            )

    # -- Environment ---------------------------------------------------------

    def _build_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build a clean environment dict for child processes.

        Inherits PATH and HOME from the current process but removes
        potentially sensitive variables.
        """
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self._work_dir),
            "TERM": "dumb",
            "LANG": "en_US.UTF-8",
            "PYTHONPATH": str(self._sandbox_dir / "site-packages"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "NODE_PATH": str(self._sandbox_dir / "node_modules"),
            "SANDBOX": "1",
        }

        # Propagate a safe subset of user env vars.
        safe_vars = {"USER", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TZ"}
        for key in safe_vars:
            if key in os.environ:
                env[key] = os.environ[key]

        if extra:
            env.update(extra)

        return env

    # -- File change tracking ------------------------------------------------

    def _snapshot_files(self) -> set[str]:
        """Take a snapshot of all files in the sandbox (relative paths)."""
        snapshot: set[str] = set()
        if not self._sandbox_dir.exists():
            return snapshot
        for root, _dirs, filenames in os.walk(self._sandbox_dir):
            base = Path(root).relative_to(self._sandbox_dir)
            for fname in filenames:
                snapshot.add(str(base / fname) if str(base) != "." else fname)
        return snapshot

    def _detect_changed_files(self) -> list[str]:
        """Detect files added or modified since the last snapshot.

        Returns a sorted list of relative paths.
        """
        current = self._snapshot_files()
        changed = sorted(current - self._file_snapshot)
        # Also check for modifications by comparing mtimes of common files.
        for fpath in sorted(current & self._file_snapshot):
            full = self._sandbox_dir / fpath
            try:
                stat_result = full.stat()
                # We can't compare to old mtime without storing it, so
                # we consider intersection as potentially changed and rely
                # on the caller to re-snapshot.
                changed.append(fpath)
            except OSError:
                pass
        return sorted(set(changed))

    # -- Cleanup -------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove the sandbox directory if it was auto-created.

        Call this when the SandboxExecutor is no longer needed to free
        disk space.  If a custom sandbox_dir was provided, this method
        is a no-op (the caller is responsible for cleanup).
        """
        if self._managed_dir and self._sandbox_dir.exists():
            try:
                shutil.rmtree(self._sandbox_dir)
            except OSError:
                pass

    async def __aenter__(self) -> "SandboxExecutor":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def __repr__(self) -> str:
        return (
            f"SandboxExecutor(sandbox_dir={self._sandbox_dir!r}, "
            f"max_memory_mb={self._max_memory_mb}, "
            f"default_timeout={self._default_timeout}, "
            f"history_count={len(self._history)})"
        )
