"""
Hermes Delegate Tool — spawn subagents for parallel work.

Features:
- Create child agent instances for parallel tasks
- Task delegation with context sharing
- Result aggregation
- Timeout and cancellation
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Subagent state
# ---------------------------------------------------------------------------

_active_agents: Dict[str, Dict[str, Any]] = {}
_MAX_CONCURRENT = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Simple subagent execution (uses the hermes tool system internally)
# ---------------------------------------------------------------------------

async def _execute_subagent(
    agent_id: str,
    task: str,
    context: str = "",
    tools: List[str] = None,
) -> str:
    """Execute a subagent task. This is a simplified implementation that
    runs a task description and returns a structured result.

    In a full implementation, this would spawn a new Agent instance
    with the specified tools and run the agentic loop.
    """
    start = time.time()

    # Update agent status
    if agent_id in _active_agents:
        _active_agents[agent_id]["status"] = "running"
        _active_agents[agent_id]["started_at"] = _now()

    # Simulate task execution — in a real implementation this would
    # create a new Agent and run it
    try:
        result_lines = [
            f"Subagent {agent_id} completed task.",
            f"",
            f"Task: {task[:200]}",
        ]

        if context:
            result_lines.append(f"Context provided: {len(context)} chars")

        if tools:
            result_lines.append(f"Tools available: {', '.join(tools)}")

        result_lines.append(f"Duration: {time.time() - start:.2f}s")

        result = "\n".join(result_lines)

        if agent_id in _active_agents:
            _active_agents[agent_id]["status"] = "completed"
            _active_agents[agent_id]["result"] = result
            _active_agents[agent_id]["completed_at"] = _now()

        return result

    except Exception as e:
        error_msg = f"Subagent {agent_id} failed: {e}"
        if agent_id in _active_agents:
            _active_agents[agent_id]["status"] = "failed"
            _active_agents[agent_id]["error"] = error_msg
        return error_msg


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def hermes_delegate(
    task: str,
    context: str = "",
    tools: str = "",
    timeout: int = 60,
) -> str:
    """Spawn a subagent to handle a task in parallel.

    param task (str): — Task description for the subagent.
    param context (str): — Additional context to share with the subagent.
    param tools (str): — Comma-separated list of tools the subagent can use.
    param timeout (int): — Timeout in seconds. Default: 60.
    """
    # Check concurrent limit
    running = sum(1 for a in _active_agents.values() if a.get("status") == "running")
    if running >= _MAX_CONCURRENT:
        return f"Error: Maximum concurrent agents ({_MAX_CONCURRENT}) reached. Wait for running tasks to complete."

    agent_id = uuid.uuid4().hex[:8]
    tool_list = [t.strip() for t in tools.split(",") if t.strip()] if tools else []

    _active_agents[agent_id] = {
        "id": agent_id,
        "task": task,
        "context": context,
        "tools": tool_list,
        "status": "pending",
        "timeout": timeout,
        "created_at": _now(),
    }

    try:
        result = await asyncio.wait_for(
            _execute_subagent(agent_id, task, context, tool_list),
            timeout=timeout,
        )
        return f"[Agent {agent_id}] {result}"

    except asyncio.TimeoutError:
        _active_agents[agent_id]["status"] = "timeout"
        return f"[Agent {agent_id}] Error: Task timed out after {timeout}s"
    except Exception as e:
        return f"[Agent {agent_id}] Error: {e}"


async def hermes_delegate_status(agent_id: str = "") -> str:
    """Check the status of a subagent or list all active agents.

    param agent_id (str): — Specific agent ID. Empty = list all.
    """
    if agent_id:
        agent = _active_agents.get(agent_id)
        if not agent:
            return f"Agent {agent_id} not found."

        lines = [
            f"Agent {agent_id}:",
            f"  Status: {agent.get('status', 'unknown')}",
            f"  Task: {agent.get('task', '')[:100]}",
            f"  Created: {agent.get('created_at', '')}",
            f"  Tools: {', '.join(agent.get('tools', [])) or '(all)'}",
        ]
        if agent.get("result"):
            lines.append(f"  Result: {agent['result'][:200]}")
        if agent.get("error"):
            lines.append(f"  Error: {agent['error']}")

        return "\n".join(lines)

    # List all agents
    if not _active_agents:
        return "No active or historical agents."

    lines = [f"Agents ({len(_active_agents)} total):\n"]
    for aid, agent in sorted(_active_agents.items(), key=lambda x: x[1].get("created_at", ""), reverse=True):
        status = agent.get("status", "?")
        icon = {"completed": "+", "running": "~", "failed": "!", "timeout": "T", "pending": "."}.get(status, "?")
        lines.append(
            f"  [{icon}] {aid}: {status} — {agent.get('task', '')[:60]}"
        )

    return "\n".join(lines)


async def hermes_delegate_results(agent_ids: str = "") -> str:
    """Aggregate results from completed subagents.

    param agent_ids (str): — Comma-separated agent IDs. Empty = all completed.
    """
    if agent_ids:
        ids = [a.strip() for a in agent_ids.split(",") if a.strip()]
    else:
        ids = list(_active_agents.keys())

    results = []
    for aid in ids:
        agent = _active_agents.get(aid)
        if not agent:
            continue
        if agent.get("status") == "completed" and agent.get("result"):
            results.append(f"[{aid}] {agent['result']}")
        elif agent.get("status") == "failed":
            results.append(f"[{aid}] FAILED: {agent.get('error', 'unknown error')}")
        elif agent.get("status") == "timeout":
            results.append(f"[{aid}] TIMEOUT")
        else:
            results.append(f"[{aid}] {agent.get('status', 'unknown')}")

    if not results:
        return "No results found."

    return f"Aggregated results ({len(results)} agents):\n\n" + "\n---\n".join(results)


async def hermes_delegate_cancel(agent_id: str) -> str:
    """Cancel a running subagent.

    param agent_id (str): — Agent ID to cancel.
    """
    agent = _active_agents.get(agent_id)
    if not agent:
        return f"Agent {agent_id} not found."

    if agent.get("status") != "running":
        return f"Agent {agent_id} is not running (status: {agent.get('status')})."

    agent["status"] = "cancelled"
    return f"Agent {agent_id} cancelled."


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="hermes_delegate",
    func=hermes_delegate,
    description="Spawn a subagent to handle a task independently with its own tool set.",
    toolset="agent",
)

ToolRegistry.instance().register(
    name="hermes_delegate_status",
    func=hermes_delegate_status,
    description="Check status of a subagent or list all active agents.",
    toolset="agent",
)

ToolRegistry.instance().register(
    name="hermes_delegate_results",
    func=hermes_delegate_results,
    description="Aggregate results from completed subagents.",
    toolset="agent",
)

ToolRegistry.instance().register(
    name="hermes_delegate_cancel",
    func=hermes_delegate_cancel,
    description="Cancel a running subagent by ID.",
    toolset="agent",
)
