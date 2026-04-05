"""
Skill Executor — Execute skill instructions with parameter substitution,
template rendering, step-by-step progress, and error recovery.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional

from .loader import Skill

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Status of a single execution step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionStep:
    """A single step in skill execution."""

    index: int
    instruction: str
    status: StepStatus = StepStatus.PENDING
    output: str = ""
    error: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_ms: float = 0.0


@dataclass
class ExecutionResult:
    """Result of a complete skill execution."""

    skill_name: str
    success: bool
    steps: list[ExecutionStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    error: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "success": self.success,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "steps": [
                {
                    "index": s.index,
                    "instruction": s.instruction[:200],
                    "status": s.status.value,
                    "output": s.output[:500] if s.output else "",
                    "error": s.error,
                    "duration_ms": round(s.duration_ms, 2),
                }
                for s in self.steps
            ],
            "error": self.error,
        }


class SkillExecutor:
    """
    Executes skills by rendering their instructions, substituting
    parameters, and processing steps with progress tracking.
    """

    def __init__(
        self,
        step_handler: Optional[Callable[[str, dict], Any]] = None,
        max_retries: int = 2,
    ) -> None:
        """
        Args:
            step_handler: Optional async callback for each step. Receives
                          (instruction_text, context_dict) and should return
                          the step result.
            max_retries: Number of retries for failed steps.
        """
        self._step_handler = step_handler
        self._max_retries = max_retries

    async def execute(
        self,
        skill: Skill,
        params: Optional[dict] = None,
        context: Optional[dict] = None,
    ) -> ExecutionResult:
        """
        Execute a skill with parameter substitution.

        The skill's instructions are parsed into steps (separated by
        numbered markers or newlines) and executed sequentially.

        Args:
            skill: The Skill to execute.
            params: Parameters to substitute into the instructions.
            context: Additional context for template rendering.

        Returns:
            ExecutionResult with all step details.
        """
        start_time = time.monotonic()
        result = ExecutionResult(
            skill_name=skill.name,
            success=False,
        )

        try:
            # Render instructions with Jinja2 if available, else simple substitution
            rendered = self._render_instructions(skill.instructions, params or {}, context or {})

            # Parse into steps
            steps = self._parse_steps(rendered)

            logger.info(
                "Executing skill %r v%s (%d steps)",
                skill.name, skill.version, len(steps),
            )

            # Execute steps
            for step_def in steps:
                step_result = await self._execute_step(step_def, params or {}, context or {})
                result.steps.append(step_result)

                if step_result.status == StepStatus.FAILED:
                    result.error = f"Step {step_result.index} failed: {step_result.error}"
                    break

            # Update skill stats
            skill.execution_count += 1
            skill.last_executed = time.monotonic()
            if result.steps and all(s.status == StepStatus.COMPLETED for s in result.steps):
                result.success = True
                skill.success_count += 1

        except Exception as exc:
            result.error = f"Execution error: {exc}"
            logger.exception("Skill %r execution failed", skill.name)

        result.total_duration_ms = (time.monotonic() - start_time) * 1000
        return result

    async def execute_stream(
        self,
        skill: Skill,
        params: Optional[dict] = None,
        context: Optional[dict] = None,
    ) -> AsyncGenerator[ExecutionStep, None]:
        """
        Execute a skill and yield results for each step as they complete.

        Useful for real-time progress updates.
        """
        rendered = self._render_instructions(skill.instructions, params or {}, context or {})
        steps = self._parse_steps(rendered)

        for step_def in steps:
            step_result = await self._execute_step(step_def, params or {}, context or {})
            yield step_result

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render_instructions(
        self,
        instructions: str,
        params: dict,
        context: dict,
    ) -> str:
        """
        Render instructions using Jinja2 if available, falling back
        to simple {{variable}} substitution.
        """
        # Merge context
        template_vars = {**context, **params}

        # Try Jinja2 first
        try:
            from jinja2 import Environment, BaseLoader, select_autoescape
            env = Environment(loader=BaseLoader(), autoescape=select_autoescape(default=False))
            template = env.from_string(instructions)
            return template.render(**template_vars)
        except Exception:
            pass

        # Fallback: simple substitution
        for key, value in template_vars.items():
            if isinstance(value, (str, int, float, bool)):
                instructions = instructions.replace(f"{{{{{key}}}}}", str(value))

        return instructions

    # ------------------------------------------------------------------
    # Step parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_steps(instructions: str) -> list[ExecutionStep]:
        """
        Parse instruction text into discrete steps.

        Recognises the following formats:
        - Numbered steps: "1. Do this" / "1) Do this"
        - Markdown headers: "## Step 1: Do this"
        - Separator-based: "---" between steps
        - Line-based: each paragraph is a step
        """
        steps: list[ExecutionStep] = []

        # Try numbered format first
        numbered_pattern = re.compile(
            r"(?:^|\n)\s*(?:#{1,4}\s*)?(?:step\s*)?(\d+)[\.\)]\s*(.+?)(?=\n\s*(?:#{1,4}\s*)?(?:step\s*)?\d+[\.\)]|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        matches = numbered_pattern.findall(instructions)
        if matches:
            for idx, (_, instruction) in enumerate(matches):
                step_text = instruction.strip()
                if step_text:
                    steps.append(ExecutionStep(index=idx, instruction=step_text))
            if steps:
                return steps

        # Try separator-based
        separator_pattern = re.compile(r"\n---+\n")
        parts = separator_pattern.split(instructions)
        if len(parts) > 1:
            for idx, part in enumerate(parts):
                part = part.strip()
                if part:
                    steps.append(ExecutionStep(index=idx, instruction=part))
            return steps

        # Fall back to paragraph-based
        paragraphs = re.split(r"\n{2,}", instructions.strip())
        for idx, para in enumerate(paragraphs):
            para = para.strip()
            if para and len(para) > 10:
                steps.append(ExecutionStep(index=idx, instruction=para))

        return steps

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    async def _execute_step(
        self,
        step: ExecutionStep,
        params: dict,
        context: dict,
    ) -> ExecutionStep:
        """Execute a single step with retry logic."""
        step.status = StepStatus.RUNNING
        step.started_at = time.monotonic()

        for attempt in range(self._max_retries + 1):
            try:
                if self._step_handler:
                    output = await self._step_handler(step.instruction, {**context, **params})
                    step.output = str(output) if output is not None else ""
                else:
                    # No handler — the instructions are meant for the agent
                    step.output = f"[Instruction: {step.instruction[:200]}]"
                step.status = StepStatus.COMPLETED
                break
            except Exception as exc:
                if attempt == self._max_retries:
                    step.status = StepStatus.FAILED
                    step.error = str(exc)
                    logger.error(
                        "Step %d failed after %d attempts: %s",
                        step.index, attempt + 1, exc,
                    )
                else:
                    logger.warning(
                        "Step %d attempt %d failed: %s; retrying...",
                        step.index, attempt + 1, exc,
                    )

        step.finished_at = time.monotonic()
        step.duration_ms = (step.finished_at - step.started_at) * 1000
        return step

    # ------------------------------------------------------------------
    # Self-improving skill creation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_skill_from_task(
        task_description: str,
        steps_taken: list[str],
        outcome: str,
        name: Optional[str] = None,
    ) -> str:
        """
        Generate SKILL.md content from a completed complex task.

        This is used by the self-improving loop when the agent
        completes a multi-step task and wants to codify it as
        a reusable skill.

        Returns the SKILL.md content string.
        """
        import re
        skill_name = name or re.sub(r"[^a-z0-9_]", "_", task_description.lower()[:30]).strip("_")

        instructions_parts = []
        for i, step in enumerate(steps_taken, 1):
            instructions_parts.append(f"{i}. {step}")

        instructions = "\n".join(instructions_parts)
        instructions += f"\n\n## Expected Outcome\n{outcome}\n"

        content = f"""---
name: {skill_name}
description: Skill auto-generated from task: {task_description[:100]}
version: 1.0.0
tags: [auto-generated, learned]
dependencies: []
author: Agent (self-improving)
---

# {skill_name.replace('_', ' ').title()}

## Context
This skill was auto-generated from a successfully completed task.

## Steps
{instructions}
"""
        return content
