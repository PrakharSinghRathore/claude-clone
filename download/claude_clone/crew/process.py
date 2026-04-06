"""
Crew process enumeration.

Defines the different execution strategies that a Crew can use to orchestrate
its agents and tasks. Each process represents a distinct coordination pattern.
"""

from enum import Enum


class Process(str, Enum):
    """
    Execution strategy for a Crew's task orchestration.

    Attributes:
        sequential:    Execute tasks one after another in the order they were
                       added. The output of each task is automatically passed
                       as context to the next task. This is the simplest and
                       most predictable process — ideal for linear pipelines
                       where each task builds on the result of the previous one.

        hierarchical:  A manager agent is created (or the user-supplied
                       ``manager_agent`` is used) that delegates tasks to
                       the most appropriate agent. The manager decides which
                       agent should handle each task and in what order,
                       allowing dynamic re-planning. Requires ``manager_llm``
                       or ``manager_agent`` to be set.

        consensual:    Every agent in the crew reviews and votes on every
                       task output. The final result is determined by majority
                       consensus or a scoring mechanism. Useful for quality-
                       critical workflows where multiple perspectives improve
                       reliability. This is the most expensive process in terms
                       of token usage.
    """

    sequential = "sequential"
    hierarchical = "hierarchical"
    consensual = "consensual"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"Process.{self.name}"
