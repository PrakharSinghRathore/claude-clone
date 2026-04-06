"""
CrewAgent — the agent abstraction used within a Crew.

A ``CrewAgent`` defines a role, goal, and backstory that shape its behaviour.
It integrates with the project's existing :class:`agent.core.Agent` for the
actual LLM calls, but adds crew-specific features like delegation, planning,
guardrails, caching, and knowledge sources.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from crew.cache import CacheHandler
from crew.guardrails import GuardrailResult, process_guardrail
from crew.rpm_controller import RPMController
from crew.usage_metrics import UsageMetrics

logger = logging.getLogger(__name__)


class CrewAgent(BaseModel):
    """
    An autonomous agent that performs tasks within a crew.

    Each agent has a **role** (its job title), a **goal** (what it wants to
    achieve), and a **backstory** (context that shapes its personality and
    approach). The agent uses an LLM to reason and optionally invoke tools.

    Attributes:
        role:                The agent's job title (e.g. ``"Senior Researcher"``).
        goal:                What the agent is trying to achieve.
        backstory:           Context/personality that guides the agent's behaviour.
        llm:                 LLM model identifier (e.g.
                             ``"anthropic/claude-sonnet-4-20250514"``).
        function_calling_llm: Model to use specifically for tool/function calling.
        max_iter:            Maximum agentic loop iterations per task.
        max_rpm:             Maximum requests per minute (rate limiting).
        verbose:             Whether to emit detailed log messages.
        allow_delegation:    Whether this agent can delegate tasks to other agents.
        tools:               List of tools available to this agent.
        step_callback:       Callable invoked after each agentic step.
        max_execution_time:  Maximum execution time per task in seconds.
        max_retry_limit:     Maximum retries on tool call errors.
        planning:            Whether to engage in task planning before execution.
        planning_config:     Configuration dict for planning behaviour.
        guardrail:           Output guardrail (prompt, callable, or Guardrail).
        guardrail_max_retries: Retries after guardrail failure.
        cache:               Whether to cache tool results.
        memory:              Whether to maintain conversation memory across tasks.
        embedder:            Configuration for the embedding model.
        knowledge_sources:   External knowledge sources to query.
        skills:              Named skill functions the agent can invoke.
        id:                  Auto-generated unique identifier.
    """

    role: str
    goal: str
    backstory: str
    llm: Optional[str] = None
    function_calling_llm: Optional[str] = None
    max_iter: int = Field(default=25, ge=1)
    max_rpm: Optional[int] = Field(default=None, ge=1)
    verbose: bool = False
    allow_delegation: bool = False
    tools: Optional[List[Any]] = Field(default=None, exclude=True, repr=False)
    step_callback: Optional[Callable] = Field(default=None, exclude=True, repr=False)
    max_execution_time: Optional[int] = Field(default=None, ge=1)
    max_retry_limit: int = Field(default=2, ge=0)
    planning: bool = False
    planning_config: Optional[Dict[str, Any]] = None
    guardrail: Optional[Union[str, Callable, Any]] = Field(default=None, exclude=True, repr=False)
    guardrail_max_retries: int = Field(default=3, ge=0)
    cache: bool = True
    memory: bool = False
    embedder: Optional[Dict[str, Any]] = None
    knowledge_sources: Optional[List[Any]] = None
    skills: Optional[List[Any]] = None
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}

    # ── Internal state (not part of the model) ──
    _cache_handler: Optional[CacheHandler] = None
    _rpm_controller: Optional[RPMController] = None
    _usage_metrics: Optional[UsageMetrics] = None

    @model_validator(mode="after")
    def _init_internal_state(self) -> "CrewAgent":
        """Initialise internal state objects after model construction."""
        object.__setattr__(self, "_cache_handler", CacheHandler() if self.cache else None)
        object.__setattr__(self, "_rpm_controller", RPMController(self.max_rpm) if self.max_rpm else None)
        object.__setattr__(self, "_usage_metrics", UsageMetrics())
        return self

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CrewAgent):
            return self.id == other.id
        return NotImplemented

    def __repr__(self) -> str:
        return f"CrewAgent(role={self.role!r}, id={self.id[:8]})"

    # ──────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────

    @property
    def cache_handler(self) -> Optional[CacheHandler]:
        """Return the agent's cache handler, or ``None`` if caching is disabled."""
        return self._cache_handler

    @property
    def rpm_controller(self) -> Optional[RPMController]:
        """Return the agent's RPM controller, or ``None`` if no limit is set."""
        return self._rpm_controller

    @property
    def usage_metrics(self) -> UsageMetrics:
        """Return the agent's accumulated usage metrics."""
        return self._usage_metrics or UsageMetrics()

    @property
    def effective_llm(self) -> str:
        """Return the LLM model to use, defaulting to the project default."""
        return self.llm or os.environ.get(
            "CLAUDE_MODEL", "anthropic/claude-sonnet-4-20250514"
        )

    @property
    def system_prompt(self) -> str:
        """Build the full system prompt for this agent from role, goal, backstory."""
        parts = [
            f"You are a {self.role}.",
            f"\n## Goal\n{self.goal}",
            f"\n## Backstory\n{self.backstory}",
        ]
        if self.allow_delegation:
            parts.append(
                "\n## Delegation\nYou may delegate tasks to other agents when "
                "appropriate. Consider whether another agent could handle part "
                "of the work more efficiently."
            )
        if self.planning:
            parts.append(
                "\n## Planning\nBefore executing, create a step-by-step plan. "
                "Outline your approach, then execute each step methodically."
            )
        return "".join(parts)

    # ──────────────────────────────────────────────
    # Execution
    # ──────────────────────────────────────────────

    async def execute_task(
        self,
        task_description: str,
        expected_output: str = "",
        context: str = "",
        tools: Optional[List[Any]] = None,
    ) -> "TaskOutput":
        """
        Execute a task using this agent's LLM integration.

        Creates a system prompt from the agent's role/goal/backstory, sends
        the task description, and returns the result as a :class:`TaskOutput`.

        Args:
            task_description: The task prompt to send to the LLM.
            expected_output:  Description of the expected output format.
            context:          Additional context string from prior tasks.
            tools:            Tools available for this specific task.

        Returns:
            A :class:`TaskOutput` containing the raw result.
        """
        from crew.task import TaskOutput

        if self._rpm_controller:
            self._rpm_controller.check_or_wait()

        # Check tool cache
        effective_tools = tools or self.tools
        if effective_tools and self._cache_handler:
            cache_key_args = {"task": task_description[:200]}
            cached = self._cache_handler.get("_task_cache", cache_key_args)
            if cached:
                logger.debug("Cache hit for task on agent %s", self.role)
                return TaskOutput(raw=cached, agent=self.role, output_format="raw")

        # Build the user prompt
        user_prompt = task_description
        if context:
            user_prompt = f"{context}\n\n{user_prompt}"
        if expected_output:
            user_prompt += f"\n\nExpected output format: {expected_output}"

        # Make the LLM call
        raw_result = await self._call_llm(user_prompt)

        # Update usage metrics (best-effort)
        self._record_usage()

        # Cache result
        if effective_tools and self._cache_handler:
            self._cache_handler.set("_task_cache", cache_key_args, raw_result)

        # Apply guardrail
        if self.guardrail:
            gr = process_guardrail(self.guardrail, raw_result, self.role)
            retries = 0
            while not gr.passed and retries < self.guardrail_max_retries:
                if gr.corrected_output:
                    raw_result = gr.corrected_output
                    gr = process_guardrail(self.guardrail, raw_result, self.role)
                else:
                    logger.warning(
                        "Agent %s guardrail failed: %s",
                        self.role, gr.reason,
                    )
                    break
                retries += 1

        # Invoke step callback
        if self.step_callback:
            try:
                self.step_callback(raw_result)
            except Exception as e:
                logger.error("Step callback error for agent %s: %s", self.role, e)

        if self.verbose:
            logger.info(
                "Agent %s completed task. Output length: %d chars",
                self.role, len(raw_result),
            )

        return TaskOutput(raw=raw_result, agent=self.role, output_format="raw")

    async def _call_llm(self, user_message: str) -> str:
        """
        Call the LLM using the project's Anthropic integration.

        Uses either OpenRouter or direct Anthropic depending on available
        environment variables, following the same pattern as
        :class:`agent.core.Agent`.
        """
        api_key = (
            os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        if not api_key:
            return "Error: No API key available. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY."

        model = self.effective_llm

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
            elif "/" in model and not model.startswith("anthropic/"):
                client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
                client_kwargs["default_headers"] = {
                    "HTTP-Referer": "https://github.com/claude-clone",
                    "X-Title": "Claude Clone",
                }

            client = anthropic.AsyncAnthropic(**client_kwargs)

            # Determine the model name to send
            effective_model = model
            if "/" in model and not client_kwargs.get("base_url"):
                effective_model = model.split("/")[-1]

            response = await client.messages.create(
                model=effective_model,
                max_tokens=8192,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)

            result = "".join(text_parts).strip()

            # Track usage
            if hasattr(response, "usage") and response.usage:
                self._usage_metrics.prompt_tokens += response.usage.input_tokens
                self._usage_metrics.completion_tokens += response.usage.output_tokens
                self._usage_metrics.total_tokens += (
                    response.usage.input_tokens + response.usage.output_tokens
                )
                self._usage_metrics.successful_requests += 1
            else:
                self._usage_metrics.successful_requests += 1

            return result

        except Exception as e:
            self._usage_metrics.failed_requests += 1
            logger.error("LLM call failed for agent %s: %s", self.role, e)
            return f"Error during LLM call: {e}"

    def _record_usage(self) -> None:
        """Log usage metrics if verbose mode is on."""
        if self.verbose and self._usage_metrics:
            logger.info(
                "Agent %s usage: %s", self.role, self._usage_metrics.to_dict(),
            )

    def create_executor(
        self,
        tools: Optional[List[Any]] = None,
        task: Optional[Any] = None,
    ) -> "CrewAgent":
        """
        Create a copy of this agent with optional tool/task overrides.

        This is useful when an agent needs to be customised for a specific
        task without mutating the original.

        Args:
            tools: Override tools for the executor.
            task:  Optional task reference for context.

        Returns:
            A new :class:`CrewAgent` instance (deep copy).
        """
        return self.model_copy(update={
            "tools": tools if tools is not None else self.tools,
        })

    def get_delegation_tools(self, agents: List["CrewAgent"]) -> List[Dict[str, Any]]:
        """
        Build delegation tool schemas for other agents.

        When ``allow_delegation`` is ``True``, this method generates tool
        definitions that let the agent delegate subtasks to other agents
        in the crew.

        Args:
            agents: List of other agents this one can delegate to.

        Returns:
            A list of tool schema dictionaries compatible with the Anthropic
            tool-use format.
        """
        if not self.allow_delegation:
            return []

        delegation_tools = []
        for other in agents:
            if other.id == self.id:
                continue
            delegation_tools.append({
                "name": f"delegate_to_{other.role.lower().replace(' ', '_')}",
                "description": (
                    f"Delegate a task to {other.role}. Use this when you need "
                    f"help with: {other.goal}"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task description to delegate",
                        },
                    },
                    "required": ["task"],
                },
            })

        return delegation_tools
