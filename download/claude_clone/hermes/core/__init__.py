"""
Hermes Core — Exports all public classes and utilities.

This package provides the foundational building blocks for an intelligent
AI agent system, including prompt construction, context management, memory
orchestration, model routing, cost tracking, and more.
"""

from hermes.core.prompt_builder import PromptBuilder, PromptSection
from hermes.core.context_compressor import ContextCompressor, CompressionStrategy
from hermes.core.memory_provider import MemoryProvider, MemoryEntry as CoreMemoryEntry
from hermes.core.memory_manager import MemoryManager
from hermes.core.builtin_memory import BuiltinMemoryProvider
from hermes.core.model_metadata import ModelMetadata, ModelInfo, get_model_info
from hermes.core.title_generator import TitleGenerator
from hermes.core.trajectory import TrajectoryRecorder, TrajectoryTurn
from hermes.core.usage_pricing import UsagePricing, CostEntry
from hermes.core.insights import InsightsManager, UsageSnapshot
from hermes.core.smart_routing import SmartRouter, RoutingDecision
from hermes.core.credential_pool import CredentialPool, CredentialEntry
from hermes.core.context_references import ContextReferenceManager, FileReference
from hermes.core.display import (
    Spinner,
    ProgressBar,
    ToolPreview,
    ColorCode,
    render_emoji,
    colorize,
)
from hermes.core.auxiliary_client import AuxiliaryClient
from hermes.core.redact import PIIRedactor, RedactionConfig

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
