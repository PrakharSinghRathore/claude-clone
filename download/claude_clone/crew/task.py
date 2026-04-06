"""
Task definitions for the crew orchestration system.

A **Task** is a unit of work assigned to an agent. It carries a description,
an expected output format, optional context from other tasks, and optional
tools. The **TaskOutput** model captures the result of a completed task.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from crew.guardrails import GuardrailResult, process_guardrail

if TYPE_CHECKING:
    from crew.agent import CrewAgent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# TaskOutput
# ──────────────────────────────────────────────

class TaskOutput(BaseModel):
    """
    Result produced by a completed task.

    Attributes:
        name:           Human-readable name of the task.
        description:    Original task description.
        expected_output: The expected output format/description.
        raw:            The raw string output from the agent.
        pydantic:       If the task's output was parsed into a Pydantic
                        model, this holds the parsed instance.
        json_dict:      If the output was parsed as JSON, this holds the
                        resulting dictionary.
        agent:          Name/role of the agent that executed this task.
        output_format:  Format of the output (``"raw"``, ``"json"``, ``"pydantic"``).
        messages:       Optional list of conversation messages exchanged
                        during task execution.
    """

    name: str = ""
    description: str = ""
    expected_output: str = ""
    raw: str = ""
    pydantic: Optional[Any] = None
    json_dict: Optional[Dict[str, Any]] = None
    agent: str = ""
    output_format: str = "raw"
    messages: Optional[List[Dict[str, Any]]] = None

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this output to a plain dictionary."""
        result: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "expected_output": self.expected_output,
            "raw": self.raw,
            "agent": self.agent,
            "output_format": self.output_format,
        }
        if self.json_dict is not None:
            result["json_dict"] = self.json_dict
        if self.pydantic is not None:
            if isinstance(self.pydantic, BaseModel):
                result["pydantic"] = self.pydantic.model_dump()
            else:
                result["pydantic"] = str(self.pydantic)
        if self.messages is not None:
            result["messages"] = self.messages
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __str__(self) -> str:
        return self.raw

    def __repr__(self) -> str:
        return (
            f"TaskOutput(name={self.name!r}, agent={self.agent!r}, "
            f"format={self.output_format!r}, len(raw)={len(self.raw)})"
        )


# ──────────────────────────────────────────────
# Task
# ──────────────────────────────────────────────

class Task(BaseModel):
    """
    A unit of work to be executed by an agent within a crew.

    Tasks are the building blocks of a crew pipeline. Each task has a
    description of what needs to be done, an expected output format, and
    optionally a reference to the agent that should execute it. Tasks can
    depend on other tasks via the ``context`` field.

    Attributes:
        name:                Optional human-readable name.
        description:         A clear description of what the task requires.
        expected_output:     A description of what the output should look like.
        agent:               The agent assigned to execute this task. Can be
                             ``None`` if assigned later by the crew.
        context:             List of tasks whose outputs serve as context.
        async_execution:     If ``True``, the task will run in a separate
                             thread/asyncio task.
        callback:            Optional callable invoked with the task output
                             after execution completes.
        output_json:         A Pydantic model class; the raw output will be
                             parsed as JSON and validated against this schema.
        output_pydantic:     Alias for ``output_json`` — kept for API
                             compatibility.
        response_model:      Alias for ``output_json`` — semantic alias.
        output_file:         If set, write the raw output to this file path.
        tools:               List of tools available only for this task.
        human_input:         If ``True``, pause and collect human input
                             before finalising the output.
        markdown:            If ``True``, format the expected output prompt
                             hint to request Markdown formatting.
        guardrail:           A single guardrail (string prompt, callable,
                             or :class:`~crew.guardrails.BaseGuardrail`).
        guardrails:          A list of guardrails to apply sequentially.
        guardrail_max_retries: Maximum number of attempts after guardrail
                             failure (with correction).
        id:                  Auto-generated UUID.
        start_time:          Set when execution begins.
        end_time:            Set when execution finishes.
    """

    name: Optional[str] = None
    description: str = ""
    expected_output: str = ""
    agent: Optional[Any] = Field(default=None, exclude=True, repr=False)
    context: Optional[List["Task"]] = Field(default=None, exclude=True, repr=False)
    async_execution: bool = False
    callback: Optional[Callable] = Field(default=None, exclude=True, repr=False)
    output_json: Optional[Type[BaseModel]] = Field(default=None, exclude=True, repr=False)
    output_pydantic: Optional[Type[BaseModel]] = Field(default=None, exclude=True, repr=False)
    response_model: Optional[Type[BaseModel]] = Field(default=None, exclude=True, repr=False)
    output_file: Optional[str] = None
    tools: Optional[List[Any]] = Field(default=None, exclude=True, repr=False)
    human_input: bool = False
    markdown: bool = False
    guardrail: Optional[Union[str, Callable, Any]] = Field(default=None, exclude=True, repr=False)
    guardrails: Optional[List[Union[str, Callable, Any]]] = Field(default=None, exclude=True, repr=False)
    guardrail_max_retries: int = Field(default=3, ge=0)
    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}

    @property
    def key(self) -> str:
        """Return a deterministic key for this task (uses id or name)."""
        return self.name or self.id

    @property
    def execution_duration(self) -> Optional[float]:
        """Return the execution duration in seconds, or ``None`` if not executed."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def _get_output_model(self) -> Optional[Type[BaseModel]]:
        """Resolve the effective output model (checks all aliases)."""
        return self.output_json or self.output_pydantic or self.response_model

    @property
    def prompt(self) -> str:
        """Build the full prompt for this task, including expected output hints."""
        parts = [self.description]

        if self.expected_output:
            format_hint = ""
            if self._get_output_model() is not None:
                schema_name = self._get_output_model().__name__
                format_hint = (
                    f"\n\nYou MUST respond with valid JSON matching the "
                    f"{schema_name} schema. Do NOT wrap in markdown code blocks."
                )
            elif self.markdown:
                format_hint = "\n\nFormat your response in Markdown."
            parts.append(f"\n\nExpected output: {self.expected_output}{format_hint}")

        return "".join(parts)

    def interpolate_inputs(self, inputs: Dict[str, Any]) -> "Task":
        """
        Replace ``{key}`` placeholders in the description and expected_output
        with values from *inputs*.

        Args:
            inputs: Dictionary of key-value pairs for interpolation.

        Returns:
            ``self`` (modified in-place for convenience).
        """
        if not inputs:
            return self

        def _replace(text: str) -> str:
            def _replacer(match: re.Match) -> str:
                key = match.group(1)
                if key in inputs:
                    return str(inputs[key])
                return match.group(0)

            return re.sub(r"\{(\w+)\}", _replacer, text)

        self.description = _replace(self.description)
        self.expected_output = _replace(self.expected_output)
        return self

    def _build_context_string(self, context_outputs: List[TaskOutput]) -> str:
        """Build a context block from previous task outputs."""
        if not context_outputs:
            return ""
        parts = []
        for co in context_outputs:
            label = co.name or co.agent or "previous task"
            parts.append(f"--- Result from {label} ---\n{co.raw}")
        return "\n\n".join(parts)

    def _parse_structured_output(self, raw: str) -> TaskOutput:
        """
        Try to parse raw output into the configured structured model.

        Returns a TaskOutput with ``pydantic`` and/or ``json_dict`` populated.
        """
        output_model = self._get_output_model()
        if output_model is None:
            return TaskOutput(raw=raw, output_format="raw")

        # Try to extract JSON from the raw output
        json_str = raw
        # Strip markdown code fences if present
        if "```json" in raw:
            match = re.search(r"```json\s*\n(.*?)```", raw, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
        elif "```" in raw:
            match = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
            if match:
                json_str = match.group(1).strip()

        try:
            parsed_dict = json.loads(json_str)
            if isinstance(parsed_dict, dict):
                parsed_instance = output_model.model_validate(parsed_dict)
                return TaskOutput(
                    raw=raw,
                    json_dict=parsed_dict,
                    pydantic=parsed_instance,
                    output_format="pydantic",
                )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                "Failed to parse structured output for task '%s': %s",
                self.name, e,
            )

        return TaskOutput(raw=raw, output_format="raw")

    def _apply_guardrails(
        self,
        raw_output: str,
        agent_role: str = "",
    ) -> GuardrailResult:
        """Run all configured guardrails against the output."""
        all_guardrails: List[Any] = []
        if self.guardrail is not None:
            all_guardrails.append(self.guardrail)
        if self.guardrails:
            all_guardrails.extend(self.guardrails)

        for gr in all_guardrails:
            result = process_guardrail(gr, raw_output, agent_role)
            if not result.passed:
                return result

        return GuardrailResult(passed=True, reason="All guardrails passed")

    async def execute_async(
        self,
        agent: Optional["CrewAgent"] = None,
        context: Optional[List[TaskOutput]] = None,
        tools: Optional[List[Any]] = None,
    ) -> TaskOutput:
        """
        Execute this task asynchronously.

        Args:
            agent:   The agent to use. Falls back to ``self.agent``.
            context: Outputs from prior tasks to use as context.
            tools:   Additional tools for this execution.

        Returns:
            A :class:`TaskOutput` with the result.
        """
        self.start_time = datetime.now()
        effective_agent = agent or self.agent
        context_str = self._build_context_string(context or [])

        # Build full prompt
        full_prompt = self.prompt
        if context_str:
            full_prompt = f"{context_str}\n\n{full_prompt}"

        # Execute via agent
        raw_output = ""
        agent_name = ""
        messages: List[Dict[str, Any]] = []

        if effective_agent is not None:
            # Use CrewAgent's execute_task
            from crew.agent import CrewAgent
            if isinstance(effective_agent, CrewAgent):
                task_output = await effective_agent.execute_task(
                    task_description=full_prompt,
                    expected_output=self.expected_output,
                    tools=tools or self.tools,
                )
                raw_output = task_output.raw if isinstance(task_output, TaskOutput) else str(task_output)
                agent_name = effective_agent.role
        else:
            # Fallback: direct LLM call
            raw_output = await self._direct_llm_call(full_prompt)
            agent_name = "direct_llm"

        # Apply guardrails with retry
        guardrail_result = self._apply_guardrails(raw_output, agent_name)
        retries = 0
        while not guardrail_result.passed and retries < self.guardrail_max_retries:
            if guardrail_result.corrected_output:
                logger.info(
                    "Guardrail failed for task '%s' (attempt %d/%d), using corrected output",
                    self.name, retries + 1, self.guardrail_max_retries,
                )
                raw_output = guardrail_result.corrected_output
                guardrail_result = self._apply_guardrails(raw_output, agent_name)
            else:
                logger.warning(
                    "Guardrail failed for task '%s' (attempt %d/%d): %s",
                    self.name, retries + 1, self.guardrail_max_retries,
                    guardrail_result.reason,
                )
                break
            retries += 1

        # Parse structured output if configured
        parsed_output = self._parse_structured_output(raw_output)
        parsed_output.name = self.name or ""
        parsed_output.description = self.description
        parsed_output.expected_output = self.expected_output
        parsed_output.agent = agent_name
        parsed_output.messages = messages

        # Write to file if configured
        if self.output_file:
            try:
                path = Path(self.output_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(raw_output, encoding="utf-8")
                logger.info("Task output written to %s", path)
            except Exception as e:
                logger.error("Failed to write output to file %s: %s", self.output_file, e)

        # Invoke callback
        if self.callback:
            try:
                self.callback(parsed_output)
            except Exception as e:
                logger.error("Task callback error: %s", e)

        self.end_time = datetime.now()
        return parsed_output

    def execute_sync(
        self,
        agent: Optional["CrewAgent"] = None,
        context: Optional[List[TaskOutput]] = None,
        tools: Optional[List[Any]] = None,
    ) -> TaskOutput:
        """
        Execute this task synchronously (blocking wrapper around async).

        Args:
            agent:   The agent to use. Falls back to ``self.agent``.
            context: Outputs from prior tasks to use as context.
            tools:   Additional tools for this execution.

        Returns:
            A :class:`TaskOutput` with the result.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an event loop — use nest_asyncio or create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.execute_async(agent=agent, context=context, tools=tools),
                )
                return future.result()
        else:
            return asyncio.run(
                self.execute_async(agent=agent, context=context, tools=tools)
            )

    async def _direct_llm_call(self, prompt: str) -> str:
        """
        Fallback: make a direct LLM call when no agent is available.

        Uses the Anthropic SDK via the project's existing integration pattern.
        """
        import os

        api_key = (
            os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        if not api_key:
            return "Error: No API key available for direct LLM call."

        model = os.environ.get("CLAUDE_MODEL", "anthropic/claude-sonnet-4-20250514")

        try:
            import anthropic

            client_kwargs: dict[str, Any] = {"api_key": api_key}
            base_url = os.environ.get("OPENROUTER_BASE_URL")
            if base_url and os.environ.get("OPENROUTER_API_KEY"):
                client_kwargs["base_url"] = base_url
                client_kwargs["default_headers"] = {
                    "HTTP-Referer": "https://github.com/claude-clone",
                    "X-Title": "Claude Clone",
                }

            client = anthropic.AsyncAnthropic(**client_kwargs)
            response = await client.messages.create(
                model=model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text
            return text.strip()
        except Exception as e:
            logger.error("Direct LLM call failed: %s", e)
            return f"Error during LLM call: {e}"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Task):
            return self.id == other.id
        return NotImplemented

    def __repr__(self) -> str:
        name_part = f", name={self.name!r}" if self.name else ""
        return f"Task(id={self.id[:8]}{name_part})"
