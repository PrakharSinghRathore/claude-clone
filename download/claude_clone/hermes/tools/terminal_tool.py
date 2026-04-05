"""
Hermes Terminal Tool — shell command execution with multiple backends.

Features:
- Execute shell commands with timeout, cwd, env support
- Multiple backends: local, SSH (via paramiko if available)
- Output capture (stdout, stderr, return code)
- Command history and replay
- Security: command validation, blacklist
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Security blacklist — patterns that should never be executed without override
# ---------------------------------------------------------------------------

_BLACKLIST_PATTERNS = [
    r"rm\s+(-[rfRF]+\s+)?/",          # rm -rf /
    r"mkfs\.",                          # format filesystem
    r"dd\s+.*of=/dev/",                 # dd to block device
    r">\s*/dev/",                       # redirect to block device
    r"chmod\s+(-R\s+)?777\s+/",        # world-writable root
    r"shutdown",
    r"reboot",
    r"halt",
    r"init\s+[06]",
    r":\(\)\{\s*:\|:\s*&\s*\};:",     # fork bomb
]

_BLACKLIST_RE = [re.compile(p, re.IGNORECASE) for p in _BLACKLIST_PATTERNS]

# Command history (in-memory)
_COMMAND_HISTORY: List[Dict[str, Any]] = []
_MAX_HISTORY = 500


def _is_command_safe(command: str) -> tuple[bool, str]:
    """Return (is_safe, reason) for a command string."""
    for pattern in _BLACKLIST_RE:
        if pattern.search(command):
            return False, f"Command matches forbidden pattern: {pattern.pattern}"
    return True, ""


async def _run_subprocess(
    cmd,
    shell: bool = True,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Run a subprocess and capture output."""
    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or None,
            env=process_env,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
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

    # Truncate very large output
    max_len = 100_000
    if len(stdout) > max_len:
        stdout = stdout[:max_len] + f"\n[... truncated, {len(stdout)} total chars]"
    if len(stderr) > max_len:
        stderr = stderr[:max_len] + f"\n[... truncated, {len(stderr)} total chars]"

    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": proc.returncode or 0,
        "timed_out": False,
    }


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def hermes_run_command(
    command: str,
    cwd: str = "",
    timeout: int = 30,
    env: dict = None,
) -> str:
    """Execute a shell command and return output.

    param command (str): — Shell command to execute.
    param cwd (str): — Working directory. Default: current directory.
    param timeout (int): — Timeout in seconds. Default: 30.
    param env (dict): — Additional environment variables.
    """
    safe, reason = _is_command_safe(command)
    if not safe:
        return f"Error: Command blocked for safety — {reason}"

    work_dir = str(Path(cwd).expanduser().resolve()) if cwd else None

    start = time.time()
    result = await _run_subprocess(
        command, shell=True, cwd=work_dir, env=env, timeout=timeout
    )
    elapsed = time.time() - start

    # Record in history
    entry = {
        "command": command,
        "cwd": work_dir or os.getcwd(),
        "returncode": result["returncode"],
        "elapsed": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(),
    }
    _COMMAND_HISTORY.append(entry)
    if len(_COMMAND_HISTORY) > _MAX_HISTORY:
        _COMMAND_HISTORY.pop(0)

    parts = []
    if result["stdout"]:
        parts.append(result["stdout"])
    if result["stderr"]:
        parts.append(f"[stderr]\n{result['stderr']}")
    if result["timed_out"]:
        parts.append(f"[timed out after {timeout}s]")
    if result["returncode"] != 0:
        parts.append(f"[exit code: {result['returncode']}]")

    parts.append(f"[elapsed: {elapsed:.2f}s]")
    return "\n".join(parts) if parts else "(no output)"


async def hermes_run_command_ssh(
    command: str,
    host: str,
    username: str = "",
    port: int = 22,
    key_path: str = "",
    timeout: int = 30,
) -> str:
    """Execute a command on a remote host via SSH (requires paramiko).

    param command (str): — Shell command to execute remotely.
    param host (str): — Remote hostname or IP address.
    param username (str): — SSH username. Default: current user.
    param port (int): — SSH port. Default: 22.
    param key_path (str): — Path to SSH private key.
    param timeout (int): — Timeout in seconds. Default: 30.
    """
    try:
        import paramiko
    except ImportError:
        return "Error: paramiko is required for SSH. Install with: pip install paramiko"

    safe, reason = _is_command_safe(command)
    if not safe:
        return f"Error: Command blocked for safety — {reason}"

    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: Dict[str, Any] = {"hostname": host, "port": port, "timeout": 10}
        if username:
            connect_kwargs["username"] = username
        if key_path:
            connect_kwargs["key_filename"] = key_path
        else:
            # Try default keys
            connect_kwargs["allow_agent"] = True
            connect_kwargs["look_for_keys"] = True

        client.connect(**connect_kwargs)
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)

        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()

        client.close()

        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        if exit_code != 0:
            parts.append(f"[exit code: {exit_code}]")

        return "\n".join(parts) if parts else "(no output)"

    except Exception as e:
        return f"SSH error: {e}"


async def hermes_command_history(limit: int = 20) -> str:
    """Show recent command execution history.

    param limit (int): — Number of history entries to show. Default: 20.
    """
    if not _COMMAND_HISTORY:
        return "No command history yet."

    entries = _COMMAND_HISTORY[-limit:]
    lines = [f"Recent command history (last {len(entries)} of {_MAX_HISTORY}):"]
    for i, entry in enumerate(entries, 1):
        lines.append(
            f"  {i}. [{entry['timestamp']}] ({entry['elapsed']}s) "
            f"exit={entry['returncode']} — {entry['command'][:120]}"
        )
    return "\n".join(lines)


async def hermes_replay_command(index: int) -> str:
    """Replay a command from history by its index.

    param index (int): — 1-based index from command history.
    """
    if index < 1 or index > len(_COMMAND_HISTORY):
        return f"Error: Invalid index {index}. History has {len(_COMMAND_HISTORY)} entries."

    entry = _COMMAND_HISTORY[-index]
    return await hermes_run_command(
        command=entry["command"],
        cwd=entry["cwd"],
    )


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="hermes_run_command",
    func=hermes_run_command,
    description="Execute a shell command locally with timeout and output capture.",
    toolset="terminal",
)

ToolRegistry.instance().register(
    name="hermes_run_command_ssh",
    func=hermes_run_command_ssh,
    description="Execute a shell command on a remote host via SSH.",
    toolset="terminal",
)

ToolRegistry.instance().register(
    name="hermes_command_history",
    func=hermes_command_history,
    description="Show recent command execution history.",
    toolset="terminal",
)

ToolRegistry.instance().register(
    name="hermes_replay_command",
    func=hermes_replay_command,
    description="Replay a previously executed command from history.",
    toolset="terminal",
)
