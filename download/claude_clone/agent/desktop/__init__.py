"""
Desktop AI Assistant Module
===========================
Real-time PC awareness, voice interaction, desktop control,
permission management, and orchestration.
"""

from agent.desktop.awareness import (
    DesktopAwareness,
    DesktopEventType,
    DesktopEvent,
    SystemSnapshot,
    WindowInfo,
    ProcessInfo,
    NetworkConnection,
)
from agent.desktop.voice import (
    VoiceEngine,
    VoiceEventType,
    VoiceEvent,
    STTConfig,
    TTSConfig,
    TranscriptEntry,
)
from agent.desktop.controller import (
    DesktopController,
    AutomationMacro,
    ScreenRegion,
    MouseAction,
    KeyAction,
    WindowAction,
    AutomationStep,
)
from agent.desktop.permissions import (
    PermissionManager,
    PermissionGuard,
    PermissionLevel,
    PermissionStatus,
    PermissionRule,
    ActionCategory,
    ApprovalRequest,
    AuditEntry,
    ApprovalDecision,
)
from agent.desktop.orchestrator import (
    DesktopOrchestrator,
    OrchestratorMode,
    Intent,
    ProactiveSuggestion,
    TaskAutomation,
    TaskStep,
)

__all__ = [
    # Awareness
    "DesktopAwareness",
    "DesktopEventType",
    "DesktopEvent",
    "SystemSnapshot",
    "WindowInfo",
    "ProcessInfo",
    "NetworkConnection",
    # Voice
    "VoiceEngine",
    "VoiceEventType",
    "VoiceEvent",
    "STTConfig",
    "TTSConfig",
    "TranscriptEntry",
    # Controller
    "DesktopController",
    "AutomationMacro",
    "ScreenRegion",
    "MouseAction",
    "KeyAction",
    "WindowAction",
    "AutomationStep",
    # Permissions
    "PermissionManager",
    "PermissionGuard",
    "PermissionLevel",
    "PermissionStatus",
    "PermissionRule",
    "ActionCategory",
    "ApprovalRequest",
    "AuditEntry",
    "ApprovalDecision",
    # Orchestrator
    "DesktopOrchestrator",
    "OrchestratorMode",
    "Intent",
    "ProactiveSuggestion",
    "TaskAutomation",
    "TaskStep",
]
