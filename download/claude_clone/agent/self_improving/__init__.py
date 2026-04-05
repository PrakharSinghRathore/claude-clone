"""
Self-Improving System — The AI that improves itself.

A modular system that enables the Claude Clone agent to analyze its own code,
fix bugs, optimize performance, extend capabilities, and learn from user feedback.

Submodules:
- safety: Guardrails, approval gates, backup/rollback, quarantine
- evaluator: Deep static analysis, code quality scoring, bug detection
- patcher: Bug fix generation and verified application
- extender: New tool generation for capability gaps
- optimizer: Performance profiling and bottleneck optimization
- learner: User preference learning and behavior adaptation
- evolution: Timeline tracking, improvement metrics, lineage

Usage:
    from agent.self_improving import SelfImprovingOrchestrator

    orchestrator = SelfImprovingOrchestrator(
        agent=agent,
        project_root="/path/to/claude_clone",
    )
    await orchestrator.initialize()
    report = await orchestrator.run_full_cycle()
"""

from agent.self_improving.safety import (
    SafetyGuardrails,
    SafetyEvaluation,
    ChangeType,
    ApprovalLevel,
    GateResult,
    ChangeHistory,
    PROTECTED_FILES,
    CHANGE_SIZE_LIMIT,
    DAILY_CHANGE_LIMIT,
)

from agent.self_improving.evaluator import (
    SelfEvaluator,
    FileAnalysis,
    ProjectAnalysis,
    CodeIssue,
    Severity,
    IssueCategory,
    FunctionMetrics,
    ClassMetrics,
)

from agent.self_improving.patcher import (
    SelfPatcher,
    PatchResult,
    PatchSession,
)

from agent.self_improving.extender import (
    SelfExtender,
    CapabilityGap,
    GeneratedTool,
    ExtensionResult,
)

from agent.self_improving.optimizer import (
    SelfOptimizer,
    ToolProfile,
    Bottleneck,
    OptimizationResult,
)

from agent.self_improving.learner import (
    SelfLearner,
    InteractionRecord,
    UserPreference,
    UserProfile,
    Adaptation,
)

from agent.self_improving.evolution import (
    EvolutionTracker,
    EvolutionEvent,
    EvolutionScore,
    ChangeLineage,
    Generation,
)

__all__ = [
    # Safety
    "SafetyGuardrails", "SafetyEvaluation", "ChangeType", "ApprovalLevel",
    "GateResult", "ChangeHistory",
    # Evaluator
    "SelfEvaluator", "FileAnalysis", "ProjectAnalysis", "CodeIssue",
    "Severity", "IssueCategory", "FunctionMetrics", "ClassMetrics",
    # Patcher
    "SelfPatcher", "PatchResult", "PatchSession",
    # Extender
    "SelfExtender", "CapabilityGap", "GeneratedTool", "ExtensionResult",
    # Optimizer
    "SelfOptimizer", "ToolProfile", "Bottleneck", "OptimizationResult",
    # Learner
    "SelfLearner", "InteractionRecord", "UserPreference", "UserProfile", "Adaptation",
    # Evolution
    "EvolutionTracker", "EvolutionEvent", "EvolutionScore", "ChangeLineage", "Generation",
]
