# agent/__init__.py
from agent.core import Agent, AgentEvent, ThinkingEvent, TextEvent, ToolCallEvent, ToolResultEvent, ErrorEvent, DoneEvent
from agent.tools import TOOLS_REGISTRY, generate_tool_schemas
from agent.mcp import MCPClient
from agent.teams import (
    AGENT_REGISTRY, get_agent_config, list_agents, get_categories,
    get_tools_for_agent, get_category_label, build_team_for_task, print_agent_table,
)
from agent.session_recorder import SessionRecorder, SessionEvent, Session
from agent.plan_mode import PlanMode, PlanStep, ExecutionPlan
from agent.task_queue import TaskQueue, TaskPriority, TaskStatus, BackgroundTask
from agent.feedback import FeedbackCollector, FeedbackRating, FeedbackEntry

__all__ = [
    "Agent", "AgentEvent", "ThinkingEvent", "TextEvent",
    "ToolCallEvent", "ToolResultEvent", "ErrorEvent", "DoneEvent",
    "TOOLS_REGISTRY", "generate_tool_schemas", "MCPClient",
    "AGENT_REGISTRY", "get_agent_config", "list_agents", "get_categories",
    "get_tools_for_agent", "get_category_label", "build_team_for_task", "print_agent_table",
    "SessionRecorder", "SessionEvent", "Session",
    "PlanMode", "PlanStep", "ExecutionPlan",
    "TaskQueue", "TaskPriority", "TaskStatus", "BackgroundTask",
    "FeedbackCollector", "FeedbackRating", "FeedbackEntry",
]
