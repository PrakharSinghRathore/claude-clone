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

# Atlas Agent Integration
try:
    from atlas.core.prompt_builder import PromptBuilder
    from atlas.core.context_compressor import ContextCompressor
    from atlas.core.memory_manager import MemoryManager
    from atlas.core.builtin_memory import BuiltinMemoryProvider
    from atlas.core.smart_routing import SmartRouter
    from atlas.core.credential_pool import CredentialPool
    from atlas.core.insights import InsightsManager
    from atlas.core.trajectory import TrajectoryRecorder
    from atlas.core.model_metadata import ModelMetadata
    from atlas.core.usage_pricing import UsagePricing
    from atlas.core.auxiliary_client import AuxiliaryClient
    from atlas.core.redact import PIIRedactor
    from atlas.tools.registry import ToolRegistry
    from atlas.skills.manager import SkillManager
    from atlas.cron.scheduler import CronScheduler
    from atlas.cron.jobs import JobManager
    from atlas.plugins.memory.registry import MemoryPluginRegistry
    from atlas.acp.server import create_acp_app
    from atlas.gateway.runner import GatewayRunner
    from atlas.gateway.config import GatewayConfig
    from atlas.plugin_sdk import PluginLoader, PluginRegistry as AtlasPluginRegistry
    from atlas.hooks.system import HookSystem, HookPoint, HookContext
    from atlas.i18n.loader import I18nManager
    ATLAS_AVAILABLE = True
except ImportError as e:
    ATLAS_AVAILABLE = False
    logging.getLogger(__name__).debug("Atlas packages not available: %s", e)

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
    # Atlas exports
    "ATLAS_AVAILABLE",
    "PromptBuilder", "ContextCompressor", "MemoryManager", "BuiltinMemoryProvider",
    "SmartRouter", "CredentialPool", "InsightsManager", "TrajectoryRecorder",
    "ModelMetadata", "UsagePricing", "AuxiliaryClient", "PIIRedactor",
    "ToolRegistry", "SkillManager", "CronScheduler", "JobManager",
    "MemoryPluginRegistry", "create_acp_app", "GatewayRunner", "GatewayConfig",
    "PluginLoader", "AtlasPluginRegistry", "HookSystem", "HookPoint", "HookContext",
    "I18nManager",
]
