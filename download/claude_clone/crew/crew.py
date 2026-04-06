"""
Crew orchestration — the central coordination layer.

A **Crew** groups agents and tasks together and defines the execution process
(sequential, hierarchical, or consensual). Calling :meth:`kickoff` runs the
entire pipeline and returns a :class:`CrewOutput`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, model_validator

from crew.agent import CrewAgent
from crew.cache import CacheHandler
from crew.process import Process
from crew.task import Task, TaskOutput
from crew.usage_metrics import UsageMetrics

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CrewOutput
# ──────────────────────────────────────────────

class CrewOutput(BaseModel):
    """
    The final result of a crew execution.

    Attributes:
        raw:            The raw string output (from the last task).
        pydantic:       If the last task produced a structured output,
                        this holds the parsed Pydantic instance.
        json_dict:      If the last task produced JSON, this holds the dict.
        tasks_output:   A list of :class:`TaskOutput` from every task executed.
        token_usage:    Cumulative token usage across all agents/tasks.
    """

    raw: str = ""
    pydantic: Optional[Any] = None
    json_dict: Optional[Dict[str, Any]] = None
    tasks_output: List[TaskOutput] = Field(default_factory=list)
    token_usage: Optional[Dict[str, Any]] = None

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        result: Dict[str, Any] = {
            "raw": self.raw,
            "tasks_output": [t.to_dict() for t in self.tasks_output],
        }
        if self.json_dict is not None:
            result["json_dict"] = self.json_dict
        if self.pydantic is not None:
            result["pydantic"] = (
                self.pydantic.model_dump()
                if isinstance(self.pydantic, BaseModel)
                else str(self.pydantic)
            )
        if self.token_usage is not None:
            result["token_usage"] = self.token_usage
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __str__(self) -> str:
        return self.raw

    def __repr__(self) -> str:
        return (
            f"CrewOutput(tasks={len(self.tasks_output)}, "
            f"len(raw)={len(self.raw)})"
        )


# ──────────────────────────────────────────────
# Crew
# ──────────────────────────────────────────────

class Crew(BaseModel):
    """
    Orchestrates a group of agents working on a sequence of tasks.

    The crew defines **how** work is distributed (via the ``process``
    attribute) and **what** work needs to be done (via ``tasks`` and
    ``agents``).

    Args:
        name:                 Optional crew name (for logging/display).
        tasks:                Ordered list of tasks to execute.
        agents:               List of agents available for task execution.
        process:              Execution strategy (sequential / hierarchical / consensual).
        verbose:              Enable detailed logging.
        memory:               Maintain conversation memory across tasks.
        cache:                Cache tool results.
        manager_llm:          LLM model for the manager agent (hierarchical process).
        manager_agent:        Pre-configured manager agent (hierarchical process).
        function_calling_llm: LLM model for function/tool calling.
        max_rpm:              Global requests-per-minute limit.
        step_callback:        Called after each agentic step across all agents.
        task_callback:        Called after each task completes.
        planning:             Enable planning mode for all agents.
        planning_llm:         LLM model for planning.
        embedder:             Configuration for embeddings.
        knowledge_sources:    External knowledge sources.
        stream:               If ``True``, yield partial results during execution.
        id:                   Auto-generated unique identifier.

    Example::

        from crew import Crew, CrewAgent, Task, Process

        researcher = CrewAgent(
            role="Researcher",
            goal="Find information on a topic",
            backstory="You are an expert researcher.",
        )
        writer = CrewAgent(
            role="Writer",
            goal="Write engaging content",
            backstory="You are a skilled writer.",
        )

        research_task = Task(
            description="Research the benefits of Python",
            expected_output="A list of 5 key benefits",
            agent=researcher,
        )
        writing_task = Task(
            description="Write a blog post based on the research",
            expected_output="A 500-word blog post",
            agent=writer,
            context=[research_task],
        )

        crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])
        result = crew.kickoff()
    """

    name: Optional[str] = "crew"
    tasks: List[Task] = Field(default_factory=list)
    agents: List[CrewAgent] = Field(default_factory=list)
    process: Process = Process.sequential
    verbose: bool = False
    memory: bool = False
    cache: bool = True
    manager_llm: Optional[str] = None
    manager_agent: Optional[CrewAgent] = Field(default=None, exclude=True, repr=False)
    function_calling_llm: Optional[str] = None
    max_rpm: Optional[int] = Field(default=None, ge=1)
    step_callback: Optional[Callable] = Field(default=None, exclude=True, repr=False)
    task_callback: Optional[Callable] = Field(default=None, exclude=True, repr=False)
    planning: bool = False
    planning_llm: Optional[str] = None
    embedder: Optional[Dict[str, Any]] = None
    knowledge_sources: Optional[List[Any]] = None
    stream: bool = False
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}

    # Internal cache and state
    _cache_handler: Optional[CacheHandler] = None

    @model_validator(mode="after")
    def _validate_crew(self) -> "Crew":
        """Post-construction validation."""
        if self.process == Process.hierarchical:
            if self.manager_agent is None and self.manager_llm is None:
                logger.warning(
                    "Hierarchical process requires manager_llm or manager_agent. "
                    "Defaulting to manager_llm='anthropic/claude-sonnet-4-20250514'."
                )
                object.__setattr__(
                    self, "manager_llm",
                    self.manager_llm or os.environ.get(
                        "CLAUDE_MODEL", "anthropic/claude-sonnet-4-20250514"
                    ),
                )

        # Bind agents to tasks if not already bound
        for task in self.tasks:
            if task.agent is None and self.agents:
                # Find an unassigned agent, or use the first one
                assigned_ids = {
                    t.agent.id for t in self.tasks
                    if t.agent is not None and isinstance(t.agent, CrewAgent)
                }
                for agent in self.agents:
                    if agent.id not in assigned_ids:
                        task.agent = agent
                        break
                else:
                    if self.agents:
                        task.agent = self.agents[0]

        # Initialise cache
        object.__setattr__(
            self, "_cache_handler",
            CacheHandler() if self.cache else None,
        )

        return self

    def __repr__(self) -> str:
        return (
            f"Crew(name={self.name!r}, agents={len(self.agents)}, "
            f"tasks={len(self.tasks)}, process={self.process})"
        )

    # ──────────────────────────────────────────────
    # Kickoff (main entry points)
    # ──────────────────────────────────────────────

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> CrewOutput:
        """
        Run the crew synchronously.

        This is the primary entry point. It interpolates inputs into task
        descriptions, then executes according to the configured process.

        Args:
            inputs: Optional dictionary of ``{key: value}`` pairs to
                    interpolate into task descriptions using ``{key}``
                    placeholders.

        Returns:
            A :class:`CrewOutput` containing the final result and all
            intermediate task outputs.
        """
        inputs = inputs or {}

        # Interpolate inputs into all task descriptions
        for task in self.tasks:
            task.interpolate_inputs(inputs)

        logger.info("Crew '%s' starting (%d agents, %d tasks, process=%s)",
                     self.name, len(self.agents), len(self.tasks), self.process)

        if self.verbose:
            logger.info("Crew inputs: %s", json.dumps(inputs, default=str)[:500])

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.kickoff_async(inputs))
                return future.result()
        else:
            return asyncio.run(self.kickoff_async(inputs))

    async def kickoff_async(self, inputs: Optional[Dict[str, Any]] = None) -> CrewOutput:
        """
        Run the crew asynchronously.

        Args:
            inputs: Optional dictionary of ``{key: value}`` pairs to
                    interpolate into task descriptions.

        Returns:
            A :class:`CrewOutput`.
        """
        inputs = inputs or {}

        # Interpolate inputs
        for task in self.tasks:
            task.interpolate_inputs(inputs)

        logger.info("Crew '%s' starting async", self.name)

        match self.process:
            case Process.sequential:
                outputs = await self._run_sequential_process()
            case Process.hierarchical:
                outputs = await self._run_hierarchical_process()
            case Process.consensual:
                outputs = await self._run_consensual_process()
            case _:
                raise ValueError(f"Unknown process: {self.process}")

        # Build final output
        last_output = outputs[-1] if outputs else TaskOutput()
        token_usage = self.calculate_usage_metrics()

        result = CrewOutput(
            raw=last_output.raw,
            pydantic=last_output.pydantic,
            json_dict=last_output.json_dict,
            tasks_output=outputs,
            token_usage=token_usage.to_dict() if token_usage else None,
        )

        logger.info("Crew '%s' completed. %d tasks executed.", self.name, len(outputs))

        if self.verbose:
            logger.info("Crew output length: %d chars", len(result.raw))

        return result

    # ──────────────────────────────────────────────
    # Sequential process
    # ──────────────────────────────────────────────

    async def _run_sequential_process(self) -> List[TaskOutput]:
        """
        Execute tasks in order, passing the output of each task as context
        to the next.

        Returns:
            A list of :class:`TaskOutput` in execution order.
        """
        outputs: List[TaskOutput] = []

        for i, task in enumerate(self.tasks):
            if self.verbose:
                logger.info(
                    "Sequential: executing task %d/%d (%s)",
                    i + 1, len(self.tasks), task.key,
                )

            # Gather context from previously completed tasks
            context_outputs = []
            if task.context:
                # Context tasks: find their outputs
                for ctx_task in task.context:
                    for prev_out in outputs:
                        if prev_out.name and prev_out.name == ctx_task.name:
                            context_outputs.append(prev_out)
                            break
            elif i > 0:
                # Default: pass previous task output as context
                context_outputs = [outputs[-1]]

            output = await task.execute_async(
                agent=task.agent,
                context=context_outputs if context_outputs else None,
                tools=task.tools,
            )
            outputs.append(output)

            # Task callback
            if self.task_callback:
                try:
                    self.task_callback(output)
                except Exception as e:
                    logger.error("Task callback error: %s", e)

        return outputs

    # ──────────────────────────────────────────────
    # Hierarchical process
    # ──────────────────────────────────────────────

    async def _run_hierarchical_process(self) -> List[TaskOutput]:
        """
        Use a manager agent to delegate tasks to the most appropriate agent.

        The manager receives the list of tasks and agents, decides the
        execution order, and delegates each task. After each task, the
        manager may re-plan based on the result.

        Returns:
            A list of :class:`TaskOutput` in execution order.
        """
        # Create or use the manager agent
        manager = self.manager_agent
        if manager is None:
            manager = CrewAgent(
                role="Crew Manager",
                goal="Coordinate task execution across the crew to achieve the best possible outcome",
                backstory=(
                    "You are an experienced project manager who excels at "
                    "delegating work to specialists. You review task requirements, "
                    "match them to the best-suited team member, and ensure quality "
                    "by reviewing results before moving on."
                ),
                llm=self.manager_llm,
                verbose=self.verbose,
            )

        # Build a description of the team for the manager
        agent_descriptions = "\n".join(
            f"- {a.role}: {a.goal}" for a in self.agents
        )
        task_descriptions = "\n".join(
            f"{i + 1}. {t.description}" for i, t in enumerate(self.tasks)
        )

        manager_prompt = (
            f"You are managing a crew of {len(self.agents)} agents.\n\n"
            f"## Available Agents\n{agent_descriptions}\n\n"
            f"## Tasks to Complete\n{task_descriptions}\n\n"
            "For each task, decide which agent should handle it and provide "
            "clear instructions. After each task result, decide the next step.\n\n"
            "IMPORTANT: For each task, respond with JSON:\n"
            '{"agent_role": "Role Name", "task_description": "detailed instructions", "done": false}\n\n'
            'When all tasks are complete, respond with: {"done": true, "summary": "final summary"}\n\n'
            "Begin with task 1."
        )

        # Execute the manager's planning loop
        outputs: List[TaskOutput] = []
        task_index = 0
        max_iterations = len(self.tasks) * 3  # Safety bound

        for iteration in range(max_iterations):
            if task_index >= len(self.tasks):
                break

            task = self.tasks[task_index]
            if self.verbose:
                logger.info(
                    "Hierarchical: manager delegating task %d/%d (%s)",
                    task_index + 1, len(self.tasks), task.key,
                )

            # Select the best agent for this task
            selected_agent = task.agent
            if selected_agent is None:
                selected_agent = self._select_agent_for_task(manager, task)

            if selected_agent is None:
                logger.warning("No agent available for task %s, skipping", task.key)
                task_index += 1
                continue

            # Build context from previous outputs
            context_outputs = outputs[-1:] if outputs else None

            # Execute the task
            output = await task.execute_async(
                agent=selected_agent,
                context=context_outputs,
                tools=task.tools,
            )
            outputs.append(output)

            if self.task_callback:
                try:
                    self.task_callback(output)
                except Exception as e:
                    logger.error("Task callback error: %s", e)

            task_index += 1

        return outputs

    def _select_agent_for_task(
        self,
        manager: CrewAgent,
        task: Task,
    ) -> Optional[CrewAgent]:
        """
        Select the best agent for a task.

        If the task has a pre-assigned agent, use it. Otherwise, pick the
        first available agent that hasn't been recently used (simple
        round-robin strategy).

        Args:
            manager: The manager agent making the selection.
            task:    The task to assign.

        Returns:
            A :class:`CrewAgent`, or ``None`` if no agent is available.
        """
        if task.agent and isinstance(task.agent, CrewAgent):
            return task.agent

        if not self.agents:
            return None

        # Simple strategy: return the first agent
        # A more sophisticated implementation could use LLM-based selection
        return self.agents[0]

    # ──────────────────────────────────────────────
    # Consensual process
    # ──────────────────────────────────────────────

    async def _run_consensual_process(self) -> List[TaskOutput]:
        """
        Execute each task with multiple agents and reach consensus.

        For each task, every agent (or a quorum) reviews the output and
        votes. The final output is the one with majority approval, or the
        best-scoring output if scoring is used.

        Returns:
            A list of :class:`TaskOutput` — one per task.
        """
        if not self.agents:
            logger.warning("Consensual process requires at least one agent")
            return []

        outputs: List[TaskOutput] = []
        quorum_size = max(1, (len(self.agents) + 1) // 2)  # Simple majority

        for i, task in enumerate(self.tasks):
            if self.verbose:
                logger.info(
                    "Consensual: processing task %d/%d with %d agents",
                    i + 1, len(self.tasks), len(self.agents),
                )

            # Collect context from previous tasks
            context_outputs = outputs[-1:] if i > 0 and outputs else None

            # Have each agent execute the task independently
            candidate_outputs: List[TaskOutput] = []
            for agent in self.agents:
                try:
                    output = await task.execute_async(
                        agent=agent,
                        context=context_outputs,
                        tools=task.tools,
                    )
                    candidate_outputs.append(output)
                except Exception as e:
                    logger.error(
                        "Agent %s failed on task %s: %s",
                        agent.role, task.key, e,
                    )

            if not candidate_outputs:
                logger.warning("No candidate outputs for task %s", task.key)
                outputs.append(TaskOutput(raw="", agent="none", output_format="raw"))
                continue

            # Select the best output (simple strategy: longest output, or first if equal)
            best_output = max(candidate_outputs, key=lambda o: len(o.raw))
            outputs.append(best_output)

            if self.verbose:
                logger.info(
                    "Consensual: %d candidates for task %s, selected output from %s (%d chars)",
                    len(candidate_outputs), task.key, best_output.agent, len(best_output.raw),
                )

            if self.task_callback:
                try:
                    self.task_callback(best_output)
                except Exception as e:
                    logger.error("Task callback error: %s", e)

        return outputs

    # ──────────────────────────────────────────────
    # Usage metrics
    # ──────────────────────────────────────────────

    def calculate_usage_metrics(self) -> UsageMetrics:
        """
        Aggregate usage metrics from all agents in the crew.

        Returns:
            A :class:`UsageMetrics` instance with summed totals.
        """
        total = UsageMetrics()
        for agent in self.agents:
            total.add(agent.usage_metrics)
        return total

    # ──────────────────────────────────────────────
    # Training & testing
    # ──────────────────────────────────────────────

    def train(
        self,
        n_iterations: int,
        filename: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Train the crew by running it multiple times and collecting results.

        This is useful for iterative prompt refinement and testing. Each
        iteration runs the full crew pipeline and records the outputs.

        Args:
            n_iterations: Number of training iterations to run.
            filename:     Optional file path to save training data as JSONL.
            inputs:       Optional inputs for each iteration (can be a list
                          for per-iteration inputs, or a dict for all iterations).

        Returns:
            A list of result dictionaries from each iteration.
        """
        from crew.training import TrainingHandler

        trainer = TrainingHandler()
        results: List[Dict[str, Any]] = []

        for i in range(n_iterations):
            logger.info("Training iteration %d/%d", i + 1, n_iterations)

            # Per-iteration inputs
            iter_inputs = inputs
            if isinstance(inputs, list) and i < len(inputs):
                iter_inputs = inputs[i]

            try:
                output = self.kickoff(inputs=iter_inputs)
                result = {
                    "iteration": i + 1,
                    "status": "success",
                    "output": output.raw,
                    "tasks": len(output.tasks_output),
                    "token_usage": output.token_usage,
                }
            except Exception as e:
                result = {
                    "iteration": i + 1,
                    "status": "error",
                    "error": str(e),
                }
                logger.error("Training iteration %d failed: %s", i + 1, e)

            results.append(result)
            trainer.save_iteration(
                iteration=i + 1,
                inputs=iter_inputs or {},
                output=result,
            )

        if filename:
            trainer.save_to_file(filename)

        logger.info("Training complete: %d/%d iterations successful",
                     sum(1 for r in results if r["status"] == "success"),
                     n_iterations)

        return results

    def test(
        self,
        n_iterations: int,
        filename: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Test the crew by running it multiple times without modifying state.

        Similar to :meth:`train` but explicitly does not modify any agent
        prompts or configurations.

        Args:
            n_iterations: Number of test iterations.
            filename:     Optional file to save test results.
            inputs:       Optional inputs (dict or list of dicts).

        Returns:
            A list of result dictionaries from each iteration.
        """
        from crew.training import TrainingHandler

        tester = TrainingHandler()
        results: List[Dict[str, Any]] = []

        for i in range(n_iterations):
            logger.info("Test iteration %d/%d", i + 1, n_iterations)

            iter_inputs = inputs
            if isinstance(inputs, list) and i < len(inputs):
                iter_inputs = inputs[i]

            try:
                output = self.kickoff(inputs=iter_inputs)
                result = {
                    "iteration": i + 1,
                    "status": "success",
                    "output": output.raw,
                    "tasks": len(output.tasks_output),
                    "token_usage": output.token_usage,
                }
            except Exception as e:
                result = {
                    "iteration": i + 1,
                    "status": "error",
                    "error": str(e),
                }

            results.append(result)
            tester.save_iteration(
                iteration=i + 1,
                inputs=iter_inputs or {},
                output=result,
            )

        if filename:
            tester.save_to_file(filename)

        return results

    # ──────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────

    def copy(self) -> "Crew":
        """
        Create a deep copy of this crew.

        Useful when you need to run the same crew with different inputs
        without mutating the original.
        """
        return self.model_copy(deep=True)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Crew":
        """
        Create a crew from a configuration dictionary.

        The config should have ``agents`` and ``tasks`` lists, where each
        entry is a dictionary of field values for :class:`CrewAgent` and
        :class:`Task` respectively.

        Args:
            config: Configuration dictionary. Example::

                {
                    "name": "research_crew",
                    "process": "sequential",
                    "agents": [
                        {"role": "Researcher", "goal": "...", "backstory": "..."},
                    ],
                    "tasks": [
                        {"description": "...", "expected_output": "...", "agent": 0},
                    ],
                }

        Returns:
            A new :class:`Crew` instance.

        Raises:
            ValueError: If the config is missing required fields.
        """
        if "agents" not in config:
            raise ValueError("Config must contain an 'agents' list")
        if "tasks" not in config:
            raise ValueError("Config must contain a 'tasks' list")

        agents: List[CrewAgent] = []
        for agent_cfg in config["agents"]:
            agents.append(CrewAgent(**agent_cfg))

        tasks: List[Task] = []
        for task_cfg in config["tasks"]:
            # Resolve agent reference (can be an index or a role name)
            agent_ref = task_cfg.pop("agent", None)
            resolved_agent = None
            if agent_ref is not None:
                if isinstance(agent_ref, int) and 0 <= agent_ref < len(agents):
                    resolved_agent = agents[agent_ref]
                elif isinstance(agent_ref, str):
                    for a in agents:
                        if a.role == agent_ref:
                            resolved_agent = a
                            break
            task_cfg["agent"] = resolved_agent

            # Resolve context references (list of task indices)
            ctx_refs = task_cfg.pop("context", None)
            if ctx_refs is not None:
                resolved_ctx: List[Task] = []
                for ref in ctx_refs:
                    if isinstance(ref, int) and 0 <= ref < len(tasks):
                        resolved_ctx.append(tasks[ref])
                task_cfg["context"] = resolved_ctx if resolved_ctx else None

            tasks.append(Task(**task_cfg))

        process_str = config.get("process", "sequential")
        process = Process(process_str)

        crew_config = {
            "name": config.get("name", "crew"),
            "agents": agents,
            "tasks": tasks,
            "process": process,
            "verbose": config.get("verbose", False),
            "memory": config.get("memory", False),
            "cache": config.get("cache", True),
            "manager_llm": config.get("manager_llm"),
            "max_rpm": config.get("max_rpm"),
            "planning": config.get("planning", False),
            "planning_llm": config.get("planning_llm"),
        }

        # Remove None values
        crew_config = {k: v for k, v in crew_config.items() if v is not None}

        return cls(**crew_config)
