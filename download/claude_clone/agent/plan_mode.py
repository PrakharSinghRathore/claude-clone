"""
Plan Mode with User Approval system.

The agent creates a step-by-step plan BEFORE executing any work.  The plan is
presented to the user who can approve, reject, modify, or reorder individual
steps.  Execution follows the approved plan step-by-step with post-step
checkpoints that let the user continue, stop, retry, skip, or abort.

Usage::

    pm = PlanMode()
    plan = await pm.create_plan("Refactor auth module", steps=[...])
    rendered = pm.present_plan(plan)
    await pm.approve_step(plan, step_number=1)
    result = await pm.execute_plan(plan)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from agent.session_recorder import SessionRecorder


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    """A single step within an execution plan."""

    number: int
    description: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    files_affected: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low | medium | high | critical
    estimated_time: str = "< 1 min"
    depends_on: List[int] = field(default_factory=list)
    status: str = "pending"  # pending | approved | rejected | running | completed | failed | skipped
    result: str = ""
    approved: bool = False
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0


@dataclass
class ExecutionPlan:
    """A full execution plan consisting of ordered steps."""

    id: str
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: str = "draft"  # draft | awaiting_approval | executing | completed | failed | cancelled


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _generate_id() -> str:
    return uuid.uuid4().hex[:16]

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

_RISK_EMOJI = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🟠",
    "critical": "🔴",
}

_STATUS_EMOJI = {
    "pending": "⬜",
    "approved": "✅",
    "rejected": "❌",
    "running": "🔄",
    "completed": "✔️",
    "failed": "⚠️",
    "skipped": "⏭️",
}


# ──────────────────────────────────────────────────────────────────────────────
# PlanMode
# ──────────────────────────────────────────────────────────────────────────────

class PlanMode:
    """
    Plan Mode with User Approval.

    Parameters
    ----------
    tool_executor:
        An async callable ``async def executor(tool_name, tool_input) -> Any``
        that executes a single tool call.  Required for :meth:`execute_plan`.
    recorder:
        Optional :class:`SessionRecorder` used to log plan events.
    """

    def __init__(
        self,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        recorder: Optional[SessionRecorder] = None,
    ) -> None:
        self._tool_executor = tool_executor
        self._recorder = recorder
        self._plans: Dict[str, ExecutionPlan] = {}
        self._on_step_complete: Optional[Callable[[PlanStep, ExecutionPlan], Any]] = None

    # ── Plan creation ─────────────────────────────────────────────────────

    async def create_plan(
        self,
        goal: str,
        steps: Optional[List[Dict[str, Any]]] = None,
    ) -> ExecutionPlan:
        """
        Create a new execution plan.

        Parameters
        ----------
        goal:
            Description of the overall objective.
        steps:
            List of dicts with keys matching :class:`PlanStep` fields
            (at minimum ``description``).  If *None*, an empty plan is
            created and steps can be added later via :meth:`add_step`.

        Returns
        -------
        ExecutionPlan
        """
        plan_id = _generate_id()
        plan = ExecutionPlan(id=plan_id, goal=goal)

        if steps:
            for i, step_data in enumerate(steps, start=1):
                plan_step = PlanStep(
                    number=i,
                    description=step_data.get("description", f"Step {i}"),
                    tool_calls=step_data.get("tool_calls", []),
                    files_affected=step_data.get("files_affected", []),
                    risk_level=step_data.get("risk_level", "low"),
                    estimated_time=step_data.get("estimated_time", "< 1 min"),
                    depends_on=step_data.get("depends_on", []),
                )
                plan.steps.append(plan_step)

        self._plans[plan_id] = plan

        if self._recorder:
            try:
                await self._recorder.record_event(
                    "__plan__", "system",
                    {"action": "plan_created", "plan_id": plan_id, "goal": goal, "step_count": len(plan.steps)},
                )
            except Exception:
                pass

        return plan

    async def add_step(self, plan_id: str, step_data: Dict[str, Any]) -> PlanStep:
        """Append a new step to an existing plan."""
        plan = self._get_plan(plan_id)
        number = len(plan.steps) + 1
        step = PlanStep(
            number=number,
            description=step_data.get("description", f"Step {number}"),
            tool_calls=step_data.get("tool_calls", []),
            files_affected=step_data.get("files_affected", []),
            risk_level=step_data.get("risk_level", "low"),
            estimated_time=step_data.get("estimated_time", "< 1 min"),
            depends_on=step_data.get("depends_on", []),
        )
        plan.steps.append(step)
        return step

    # ── Plan presentation ─────────────────────────────────────────────────

    def present_plan(self, plan: ExecutionPlan) -> str:
        """
        Render a human-readable plan summary suitable for displaying to the
        user before approval.

        Returns a multi-line string with step details, risk indicators, and
        dependency information.
        """
        lines: List[str] = [
            f"# Execution Plan: {plan.goal}",
            f"**Plan ID:** `{plan.id}`  |  **Steps:** {len(plan.steps)}  |  **Status:** {plan.status}",
            "",
        ]

        if not plan.steps:
            lines.append("(empty plan — no steps defined)")
            return "\n".join(lines)

        # Aggregate risk and files.
        all_files: List[str] = []
        risk_counts: Dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for step in plan.steps:
            all_files.extend(step.files_affected)
            risk_counts[step.risk_level] = risk_counts.get(step.risk_level, 0) + 1

        unique_files = sorted(set(all_files))
        if unique_files:
            lines.append(f"**Files affected:** {', '.join(f'`{f}`' for f in unique_files)}")
            lines.append("")

        risk_parts = [f"{_RISK_EMOJI.get(k, k)} {k}: {v}" for k, v in risk_counts.items() if v > 0]
        if risk_parts:
            lines.append(f"**Risk profile:** {'  |  '.join(risk_parts)}")
            lines.append("")

        lines.append("---")
        lines.append("")

        for step in plan.steps:
            status_icon = _STATUS_EMOJI.get(step.status, "⬜")
            risk_icon = _RISK_EMOJI.get(step.risk_level, "⚪")
            approval = " (approved)" if step.approved else ""

            dep_str = ""
            if step.depends_on:
                dep_str = f"  ← depends on step(s) {', '.join(str(d) for d in step.depends_on)}"

            lines.append(
                f"### {status_icon} Step {step.number}: {step.description}{approval}"
            )
            lines.append(f"   {risk_icon} Risk: **{step.risk_level}**  |  ⏱ Est: **{step.estimated_time}**{dep_str}"
            )

            if step.files_affected:
                lines.append(f"   📁 Files: {', '.join(f'`{f}`' for f in step.files_affected)}")

            tool_names = [tc.get("name", "unknown") for tc in step.tool_calls]
            if tool_names:
                lines.append(f"   🔧 Tools: {', '.join(f'`{n}`' for n in tool_names)}")

            if step.result:
                result_preview = step.result[:200] + ("..." if len(step.result) > 200 else "")
                lines.append(f"   📋 Result: {result_preview}")

            if step.error:
                lines.append(f"   ⚠️ Error: {step.error}")

            lines.append("")

        return "\n".join(lines)

    # ── Step approval ─────────────────────────────────────────────────────

    def _get_plan(self, plan_id: str) -> ExecutionPlan:
        if plan_id not in self._plans:
            raise KeyError(f"Plan '{plan_id}' not found")
        return self._plans[plan_id]

    def _get_step(self, plan: ExecutionPlan, step_number: int) -> PlanStep:
        for step in plan.steps:
            if step.number == step_number:
                return step
        raise ValueError(f"Step {step_number} not found in plan '{plan.id}'")

    def approve_step(self, plan_id: str, step_number: int) -> PlanStep:
        """Mark a single step as approved."""
        plan = self._get_plan(plan_id)
        step = self._get_step(plan, step_number)
        step.approved = True
        step.status = "approved"
        return step

    def approve_all(self, plan_id: str) -> List[PlanStep]:
        """Approve every step in the plan."""
        plan = self._get_plan(plan_id)
        for step in plan.steps:
            step.approved = True
            step.status = "approved"
        return plan.steps

    def reject_step(self, plan_id: str, step_number: int, reason: str = "") -> PlanStep:
        """Mark a single step as rejected."""
        plan = self._get_plan(plan_id)
        step = self._get_step(plan, step_number)
        step.approved = False
        step.status = "rejected"
        step.error = reason or "Rejected by user"
        return step

    def modify_step(self, plan_id: str, step_number: int, **kwargs: Any) -> PlanStep:
        """
        Modify properties of an existing step.

        Accepted keyword arguments match :class:`PlanStep` fields:
        ``description``, ``tool_calls``, ``files_affected``, ``risk_level``,
        ``estimated_time``, ``depends_on``.
        """
        plan = self._get_plan(plan_id)
        step = self._get_step(plan, step_number)
        for key, value in kwargs.items():
            if hasattr(step, key):
                setattr(step, key, value)
            else:
                raise AttributeError(f"PlanStep has no field '{key}'")
        return step

    def reorder_steps(self, plan_id: str, new_order: List[int]) -> ExecutionPlan:
        """
        Reorder plan steps according to ``new_order``, a list of step numbers
        in the desired sequence.

        Dependencies are *not* automatically updated; the caller is responsible
        for ensuring the new order is consistent.
        """
        plan = self._get_plan(plan_id)
        existing_numbers = {s.number for s in plan.steps}
        if set(new_order) != existing_numbers:
            raise ValueError("new_order must contain exactly the same step numbers as the plan")

        step_map = {s.number: s for s in plan.steps}
        reordered: List[PlanStep] = []
        for i, num in enumerate(new_order, start=1):
            step = step_map[num]
            step.number = i
            reordered.append(step)

        plan.steps = reordered
        return plan

    # ── Plan execution ────────────────────────────────────────────────────

    def set_step_callback(self, callback: Callable[[PlanStep, ExecutionPlan], Any]) -> None:
        """
        Set a callback invoked after each step completes (or fails).

        The callback receives ``(step, plan)`` and can return ``"continue"``,
        ``"stop"``, ``"retry"``, ``"skip"``, or ``"abort"`` to control
        execution flow.
        """
        self._on_step_complete = callback

    async def execute_plan(
        self,
        plan_id: str,
        auto_approve_remaining: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute an approved plan step by step.

        Parameters
        ----------
        plan_id:
            The plan to execute.
        auto_approve_remaining:
            If ``True``, execute all pending steps without pausing for
            approval (useful for fully pre-approved plans).

        Returns
        -------
        dict
            Summary with ``status``, ``steps_completed``, ``steps_failed``,
            ``steps_skipped``, ``total_duration_ms``, and ``results``.
        """
        plan = self._get_plan(plan_id)
        plan.status = "executing"

        if self._tool_executor is None:
            plan.status = "failed"
            raise RuntimeError("No tool_executor configured. Pass one to PlanMode().")

        results: Dict[int, str] = {}
        completed = 0
        failed = 0
        skipped = 0
        start_time = time.monotonic()

        for step in plan.steps:
            # Handle rejected / already-completed steps.
            if step.status == "rejected":
                skipped += 1
                continue
            if step.status in ("completed", "skipped"):
                completed += 1
                continue

            # Wait for approval unless auto-approve or already approved.
            if not step.approved and not auto_approve_remaining:
                step.status = "pending"
                if self._on_step_complete:
                    decision = await self._safe_callback(step, plan)
                    if decision == "abort":
                        plan.status = "cancelled"
                        break
                    elif decision == "skip":
                        step.status = "skipped"
                        skipped += 1
                        continue
                    elif decision == "stop":
                        plan.status = "completed"
                        break
                    # "continue" or "retry" → fall through

                # Check again after callback (user may have approved).
                if not step.approved:
                    # Mark remaining as pending and stop.
                    plan.status = "awaiting_approval"
                    break

            # Check dependencies.
            unmet = [d for d in step.depends_on if self._get_step(plan, d).status != "completed"]
            if unmet:
                step.status = "skipped"
                step.error = f"Dependency not met: step(s) {unmet}"
                skipped += 1
                continue

            # Execute the step.
            step.status = "running"
            step.started_at = time.time()

            step_outputs: List[str] = []
            step_ok = True

            for tool_call in step.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_input = tool_call.get("input", {})
                try:
                    raw_result = await self._tool_executor(tool_name, tool_input)
                    if isinstance(raw_result, dict):
                        result_str = json.dumps(raw_result, indent=2, default=str)
                    else:
                        result_str = str(raw_result)
                    step_outputs.append(f"[{tool_name}] OK: {result_str[:500]}")
                except Exception as exc:
                    step_ok = False
                    step_outputs.append(f"[{tool_name}] ERROR: {exc}")
                    step.error = str(exc)
                    break

            step.completed_at = time.time()
            step.result = "\n".join(step_outputs)

            if step_ok:
                step.status = "completed"
                completed += 1
                results[step.number] = step.result
            else:
                step.status = "failed"
                failed += 1

            # Post-step callback.
            if self._on_step_complete:
                decision = await self._safe_callback(step, plan)
                if decision == "abort":
                    plan.status = "failed"
                    break
                elif decision == "stop":
                    plan.status = "completed"
                    break
                elif decision == "retry" and not step_ok:
                    step.status = "pending"
                    step.error = ""
                    step.approved = True
                    # Re-execute on next loop iteration by re-entering.
                    # To avoid infinite loops, decrement failed and continue.
                    failed -= 1
                    # Reset step for re-execution.
                    step.started_at = 0.0
                    step.completed_at = 0.0
                    step.result = ""
                    continue
                elif decision == "skip":
                    if not step_ok:
                        step.status = "skipped"
                        failed -= 1
                        skipped += 1

        total_duration = int((time.monotonic() - start_time) * 1000)

        if plan.status == "executing":
            plan.status = "completed"

        return {
            "plan_id": plan.id,
            "status": plan.status,
            "steps_completed": completed,
            "steps_failed": failed,
            "steps_skipped": skipped,
            "total_duration_ms": total_duration,
            "results": results,
        }

    async def _safe_callback(self, step: PlanStep, plan: ExecutionPlan) -> str:
        """Invoke the step callback, returning ``'continue'`` on error."""
        if self._on_step_complete is None:
            return "continue"
        try:
            result = await self._on_step_complete(step, plan)
            return str(result) if result else "continue"
        except Exception:
            return "continue"

    # ── Status & export ───────────────────────────────────────────────────

    def get_status(self, plan_id: str) -> Dict[str, Any]:
        """Return a status summary for a plan."""
        plan = self._get_plan(plan_id)
        completed = sum(1 for s in plan.steps if s.status == "completed")
        failed = sum(1 for s in plan.steps if s.status == "failed")
        skipped = sum(1 for s in plan.steps if s.status == "skipped")
        approved = sum(1 for s in plan.steps if s.approved)
        total = len(plan.steps)

        return {
            "plan_id": plan.id,
            "goal": plan.goal,
            "status": plan.status,
            "total_steps": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "approved": approved,
            "progress": f"{completed}/{total}" if total else "0/0",
        }

    def export_plan(self, plan_id: str, format: str = "json") -> str:  # noqa: A002
        """
        Export a plan as JSON.

        Parameters
        ----------
        plan_id:
            The plan to export.
        format:
            Currently only ``"json"`` is supported.

        Returns
        -------
        str
            JSON-encoded plan.
        """
        plan = self._get_plan(plan_id)
        if format == "json":
            return json.dumps(asdict(plan), indent=2, default=str, ensure_ascii=False)
        raise ValueError(f"Unknown format: {format!r}")

    def list_plans(self) -> List[Dict[str, Any]]:
        """Return a summary of all tracked plans."""
        return [
            self.get_status(pid) for pid in self._plans
        ]

    def remove_plan(self, plan_id: str) -> None:
        """Remove a plan from memory."""
        self._get_plan(plan_id)  # validate existence
        del self._plans[plan_id]
