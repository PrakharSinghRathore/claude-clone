"""
Atlas Core — Exports all public classes and utilities.

This package provides the foundational building blocks for an intelligent
AI agent system, including prompt construction, context management, memory
orchestration, model routing, cost tracking, and more.
"""

from atlas.core.prompt_builder import PromptBuilder, PromptSection
from atlas.core.context_compressor import ContextCompressor, CompressionStrategy
from atlas.core.memory_provider import MemoryProvider, MemoryEntry as CoreMemoryEntry
from atlas.core.memory_manager import MemoryManager
from atlas.core.builtin_memory import BuiltinMemoryProvider
from atlas.core.model_metadata import ModelMetadata, ModelInfo, get_model_info
from atlas.core.title_generator import TitleGenerator
from atlas.core.trajectory import TrajectoryRecorder, TrajectoryTurn
from atlas.core.usage_pricing import UsagePricing, CostEntry
from atlas.core.insights import InsightsManager, UsageSnapshot
from atlas.core.smart_routing import SmartRouter, RoutingDecision
from atlas.core.credential_pool import CredentialPool, CredentialEntry
from atlas.core.context_references import ContextReferenceManager, FileReference
from atlas.core.display import (
    Spinner,
    ProgressBar,
    ToolPreview,
    ColorCode,
    render_emoji,
    colorize,
)
from atlas.core.auxiliary_client import AuxiliaryClient
from atlas.core.redact import PIIRedactor, RedactionConfig

__all__ = [
    # Prompt building
    "PromptBuilder",
    "PromptSection",
    # Context management
    "ContextCompressor",
    "CompressionStrategy",
    # Memory system
    "MemoryProvider",
    "CoreMemoryEntry",
    "MemoryManager",
    "BuiltinMemoryProvider",
    # Model metadata
    "ModelMetadata",
    "ModelInfo",
    "get_model_info",
    # Title generation
    "TitleGenerator",
    # Trajectory recording
    "TrajectoryRecorder",
    "TrajectoryTurn",
    # Usage & pricing
    "UsagePricing",
    "CostEntry",
    # Insights & analytics
    "InsightsManager",
    "UsageSnapshot",
    # Smart routing
    "SmartRouter",
    "RoutingDecision",
    # Credential management
    "CredentialPool",
    "CredentialEntry",
    # Context references
    "ContextReferenceManager",
    "FileReference",
    # Display helpers
    "Spinner",
    "ProgressBar",
    "ToolPreview",
    "ColorCode",
    "render_emoji",
    "colorize",
    # Auxiliary model client
    "AuxiliaryClient",
    # PII redaction
    "PIIRedactor",
    "RedactionConfig",
]
