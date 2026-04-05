"""
Hermes Code Execution — sandboxed code execution in multiple languages.

Features:
- Python, JavaScript, Bash execution
- Resource limits (memory, CPU time)
- Output capture and error handling
- Package auto-installation (pip, npm)
"""

from __future__ import annotations

import asyncio
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

_MAX_OUTPUT = 50_000  # chars


async def _exec_subprocess(
    cmd: list,
    cwd: str = None,
    timeout: int = 30,
    env: dict = None,
) -> Dict[str, Any]:
    """Execute a subprocess with timeout and capture."""
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or None,
            env=proc_env,
            preexec_fn=_set_resource_limits,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "returncode": -1,
            "timed_out": True,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "timed_out": False,
        }

    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

    if len(stdout) > _MAX_OUTPUT:
        stdout = stdout[:_MAX_OUTPUT] + f"\n[... truncated, {len(stdout)} total chars]"
    if len(stderr) > _MAX_OUTPUT:
        stderr = stderr[:_MAX_OUTPUT] + f"\n[... truncated, {len(stderr)} total chars]"

    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": proc.returncode or 0,
        "timed_out": False,
    }


def _set_resource_limits() -> None:
    """Set resource limits for child processes (memory, CPU)."""
    try:
        # Limit memory to 512MB
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    except (ValueError, resource.error):
        pass
    try:
        # Limit CPU time to 60 seconds
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    except (ValueError, resource.error):
        pass


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def hermes_exec_python(
    code: str,
    packages: str = "",
    timeout: int = 30,
) -> str:
    """Execute Python code in a sandboxed environment.

    param code (str): — Python code to execute.
    param packages (str): — Comma-separated packages to auto-install.
    param timeout (int): — Timeout in seconds. Default: 30.
    """
    # Auto-install packages
    if packages:
        pkg_list = [p.strip() for p in packages.split(",") if p.strip()]
        for pkg in pkg_list:
            install_result = await _exec_subprocess(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                timeout=60,
            )
            if install_result["returncode"] != 0:
                return f"Error installing package '{pkg}': {install_result['stderr']}"

    # Write code to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = await _exec_subprocess(
            [sys.executable, tmp_path],
            timeout=timeout,
        )

        parts = []
        if result["stdout"]:
            parts.append(result["stdout"])
        if result["stderr"]:
            parts.append(f"[stderr]\n{result['stderr']}")
        if result["timed_out"]:
            parts.append(f"[timed out after {timeout}s]")
        if result["returncode"] != 0:
            parts.append(f"[exit code: {result['returncode']}]")

        return "\n".join(parts) if parts else "(no output)"

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def hermes_exec_javascript(
    code: str,
    timeout: int = 30,
) -> str:
    """Execute JavaScript code using Node.js.

    param code (str): — JavaScript code to execute.
    param timeout (int): — Timeout in seconds. Default: 30.
    """
    # Check if node is available
    which_result = await _exec_subprocess(["which", "node"], timeout=5)
    if which_result["returncode"] != 0:
        return "Error: Node.js is not installed. Install it from https://nodejs.org/"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = await _exec_subprocess(
            ["node", tmp_path],
            timeout=timeout,
        )

        parts = []
        if result["stdout"]:
            parts.append(result["stdout"])
        if result["stderr"]:
            parts.append(f"[stderr]\n{result['stderr']}")
        if result["timed_out"]:
            parts.append(f"[timed out after {timeout}s]")
        if result["returncode"] != 0:
            parts.append(f"[exit code: {result['returncode']}]")

        return "\n".join(parts) if parts else "(no output)"

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def hermes_exec_bash(
    code: str,
    timeout: int = 30,
) -> str:
    """Execute Bash code in a shell.

    param code (str): — Bash commands to execute.
    param timeout (int): — Timeout in seconds. Default: 30.
    """
    result = await _exec_subprocess(
        ["bash", "-c", code],
        timeout=timeout,
    )

    parts = []
    if result["stdout"]:
        parts.append(result["stdout"])
    if result["stderr"]:
        parts.append(f"[stderr]\n{result['stderr']}")
    if result["timed_out"]:
        parts.append(f"[timed out after {timeout}s]")
    if result["returncode"] != 0:
        parts.append(f"[exit code: {result['returncode']}]")

    return "\n".join(parts) if parts else "(no output)"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="hermes_exec_python",
    func=hermes_exec_python,
    description="Execute Python code in a sandboxed environment with optional auto-install.",
    toolset="execution",
)

ToolRegistry.instance().register(
    name="hermes_exec_javascript",
    func=hermes_exec_javascript,
    description="Execute JavaScript code using Node.js.",
    toolset="execution",
)

ToolRegistry.instance().register(
    name="hermes_exec_bash",
    func=hermes_exec_bash,
    description="Execute Bash commands in a shell.",
    toolset="execution",
)
