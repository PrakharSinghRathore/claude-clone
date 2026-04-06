# agent/__init__.py
import logging

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
from agent.self_improving import SelfImprovingOrchestrator

# Hermes Agent Integration
try:
    from hermes.core.prompt_builder import PromptBuilder
    from hermes.core.context_compressor import ContextCompressor
    from hermes.core.memory_manager import MemoryManager
    from hermes.core.builtin_memory import BuiltinMemoryProvider
    from hermes.core.smart_routing import SmartRouter
    from hermes.core.credential_pool import CredentialPool
    from hermes.core.insights import InsightsManager
    from hermes.core.trajectory import TrajectoryRecorder
    from hermes.core.model_metadata import ModelMetadata
    from hermes.core.usage_pricing import UsagePricing
    from hermes.core.auxiliary_client import AuxiliaryClient
    from hermes.core.redact import PIIRedactor
    from hermes.tools.registry import ToolRegistry
    from hermes.skills.manager import SkillManager
    from hermes.cron.scheduler import CronScheduler
    from hermes.cron.jobs import JobManager
    from hermes.plugins.memory.registry import MemoryPluginRegistry
    from hermes.acp.server import create_acp_app
    from hermes.gateway.runner import GatewayRunner
    from hermes.gateway.config import GatewayConfig
    HERMES_AVAILABLE = True
except ImportError as e:
    HERMES_AVAILABLE = False
    logging.getLogger(__name__).debug("Hermes packages not available: %s", e)

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
    "SelfImprovingOrchestrator",
    # Hermes exports
    "HERMES_AVAILABLE",
    "PromptBuilder", "ContextCompressor", "MemoryManager", "BuiltinMemoryProvider",
    "SmartRouter", "CredentialPool", "InsightsManager", "TrajectoryRecorder",
    "ModelMetadata", "UsagePricing", "AuxiliaryClient", "PIIRedactor",
    "ToolRegistry", "SkillManager", "CronScheduler", "JobManager",
    "MemoryPluginRegistry", "create_acp_app", "GatewayRunner", "GatewayConfig",
]
