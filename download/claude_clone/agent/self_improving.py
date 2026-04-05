"""
Self-Improving Orchestrator — The Brain Behind Self-Evolution.

Coordinates all self-improvement subsystems into a unified improvement cycle:
1. EVALUATE: Analyze the codebase for issues and metrics
2. LEARN: Review user feedback and adapt preferences
3. PATCH: Fix identified bugs and quality issues
4. EXTEND: Fill capability gaps with new tools
5. OPTIMIZE: Improve performance bottlenecks
6. TRACK: Record everything in the evolution timeline

The orchestrator can run in two modes:
- Manual: Trigger improvements on demand
- Autonomous: Run periodic improvement cycles automatically

Every change passes through safety gates. The orchestrator respects
rate limits, daily caps, and file cooldowns.

Usage:
    orchestrator = SelfImprovingOrchestrator(agent, project_root="/path/to/claude_clone")
    await orchestrator.initialize()
    report = await orchestrator.run_full_cycle()
    # Or run a specific phase:
    analysis = await orchestrator.run_evaluation()
    patches = await orchestrator.run_patching()
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.self_improving.safety import SafetyGuardrails, ChangeType
from agent.self_improving.evaluator import SelfEvaluator, CodeIssue
from agent.self_improving.patcher import SelfPatcher, PatchSession
from agent.self_improving.extender import SelfExtender
from agent.self_improving.optimizer import SelfOptimizer
from agent.self_improving.learner import SelfLearner
from agent.self_improving.evolution import EvolutionTracker


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ImprovementCycle:
    """Results of a complete improvement cycle."""
    cycle_id: str
    timestamp: str
    generation: int
    duration: float
    phases_completed: List[str]
    evaluation_score_before: float
    evaluation_score_after: float
    evaluation_grade_before: str
    evaluation_grade_after: str
    patches_attempted: int
    patches_applied: int
    patches_blocked: int
    extensions_generated: int
    extensions_integrated: int
    optimizations_attempted: int
    optimizations_applied: int
    rollbacks: int
    evolution_score_before: float
    evolution_score_after: float
    safety_stats: Dict[str, Any]
    learner_insights: Dict[str, Any]
    errors: List[str]


@dataclass
class SystemStatus:
    """Current status of the self-improving system."""
    initialized: bool
    safety_enabled: bool
    evaluator_ready: bool
    patcher_ready: bool
    extender_ready: bool
    optimizer_ready: bool
    learner_ready: bool
    evolution_ready: bool
    total_files_analyzed: int
    total_issues_found: int
    total_patches_applied: int
    total_tools_added: int
    total_optimizations: int
    evolution_score: float
    generation: int
    daily_changes_remaining: int
    last_cycle_time: str


# ──────────────────────────────────────────────────────────────────────────────
# SelfImprovingOrchestrator
# ──────────────────────────────────────────────────────────────────────────────

class SelfImprovingOrchestrator:
    """
    Central coordinator for all self-improvement activities.

    This is the main entry point for the self-improving system. It manages
    the lifecycle of all sub-modules and orchestrates improvement cycles.

    The improvement cycle follows this sequence:
    1. Analyze the codebase (SelfEvaluator)
    2. Learn from user feedback (SelfLearner)
    3. Fix bugs and quality issues (SelfPatcher + SafetyGuardrails)
    4. Fill capability gaps (SelfExtender + SafetyGuardrails)
    5. Optimize bottlenecks (SelfOptimizer + SafetyGuardrails)
    6. Track evolution (EvolutionTracker)

    Each phase can also run independently.
    """

    def __init__(
        self,
        agent: Any,
        project_root: str,
        auto_improve: bool = False,
        improvement_interval: int = 3600,  # seconds
        max_patches_per_cycle: int = 10,
        max_extensions_per_cycle: int = 3,
        max_optimizations_per_cycle: int = 5,
    ):
        self.agent = agent
        self.project_root = str(Path(project_root).resolve())
        self.auto_improve = auto_improve
        self.improvement_interval = improvement_interval
        self.max_patches = max_patches_per_cycle
        self.max_extensions = max_extensions_per_cycle
        self.max_optimizations = max_optimizations_per_cycle

        # Sub-modules (initialized in initialize())
        self.safety: Optional[SafetyGuardrails] = None
        self.evaluator: Optional[SelfEvaluator] = None
        self.patcher: Optional[SelfPatcher] = None
        self.extender: Optional[SelfExtender] = None
        self.optimizer: Optional[SelfOptimizer] = None
        self.learner: Optional[SelfLearner] = None
        self.evolution: Optional[EvolutionTracker] = None

        # State
        self._initialized = False
        self._current_generation = 0
        self._last_analysis: Optional[Any] = None
        self._auto_task: Optional[asyncio.Task] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize all sub-modules."""
        if self._initialized:
            return

        print("[Self-Improving] Initializing safety guardrails...")
        self.safety = SafetyGuardrails(project_root=self.project_root)
        await self.safety.initialize()

        print("[Self-Improving] Initializing self-evaluator...")
        self.evaluator = SelfEvaluator(project_root=self.project_root)
        await self.evaluator.initialize()

        print("[Self-Improving] Initializing self-patcher...")
        self.patcher = SelfPatcher(agent=self.agent, safety=self.safety, project_root=self.project_root)
        await self.patcher.initialize()

        print("[Self-Improving] Initializing self-extender...")
        self.extender = SelfExtender(agent=self.agent, safety=self.safety, evaluator=self.evaluator, project_root=self.project_root)
        await self.extender.initialize()

        print("[Self-Improving] Initializing self-optimizer...")
        self.optimizer = SelfOptimizer(agent=self.agent, safety=self.safety, project_root=self.project_root)
        await self.optimizer.initialize()

        print("[Self-Improving] Initializing self-learner...")
        self.learner = SelfLearner(project_root=self.project_root)
        await self.learner.initialize()

        print("[Self-Improving] Initializing evolution tracker...")
        self.evolution = EvolutionTracker(project_root=self.project_root)
        await self.evolution.initialize()

        self._initialized = True
        print(f"[Self-Improving] All subsystems initialized. Ready to evolve.")

    async def close(self) -> None:
        """Close all sub-modules."""
        if self._auto_task:
            self._auto_task.cancel()
            self._auto_task = None
        if self.safety:
            await self.safety.close()
        if self.optimizer:
            await self.optimizer.close()
        if self.learner:
            await self.learner.close()
        self._initialized = False

    # ── Phase 1: Evaluation ───────────────────────────────────────────────

    async def run_evaluation(self) -> Any:
        """Run codebase analysis and return the project analysis."""
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized")

        print("[Self-Improving] Phase 1: Evaluating codebase...")
        start = time.time()

        analysis = await self.evaluator.analyze_project()
        self._last_analysis = analysis

        # Save snapshot for evolution tracking
        snapshot_id = await self.evaluator.save_snapshot(analysis)

        # Record evolution event
        await self.evolution.record_event(
            event_type="evaluation",
            file_path="*",
            description=f"Full project analysis: {analysis.total_files} files, {len(analysis.top_issues)} issues",
            before_score=0,
            after_score=analysis.overall_score,
            generation=self._current_generation,
            metadata={
                "total_files": analysis.total_files,
                "total_lines": analysis.total_lines,
                "issues_count": len(analysis.top_issues),
                "quality_grade": analysis.quality_grade,
                "snapshot_id": snapshot_id,
            },
        )

        elapsed = time.time() - start
        print(f"[Self-Improving] Evaluation complete in {elapsed:.1f}s. Score: {analysis.overall_score:.2f} ({analysis.quality_grade})")
        return analysis

    # ── Phase 2: Learning ─────────────────────────────────────────────────

    async def run_learning(self) -> Dict[str, Any]:
        """Analyze user feedback and generate adaptations."""
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized")

        print("[Self-Improving] Phase 2: Learning from feedback...")
        start = time.time()

        adaptations = await self.learner.generate_adaptations()
        prompt_additions = await self.learner.generate_system_prompt_additions()
        learner_stats = await self.learner.get_stats()

        insights = {
            "adaptations_count": len(adaptations),
            "adaptations": [
                {"category": a.category, "description": a.description, "priority": a.priority}
                for a in adaptations[:10]
            ],
            "prompt_additions": prompt_additions[:500] if prompt_additions else "",
            "stats": learner_stats,
        }

        await self.evolution.record_event(
            event_type="learning",
            file_path="*",
            description=f"Processed {learner_stats['total_interactions']} interactions, generated {len(adaptations)} adaptations",
            generation=self._current_generation,
        )

        elapsed = time.time() - start
        print(f"[Self-Improving] Learning complete in {elapsed:.1f}s. {len(adaptations)} adaptations generated.")
        return insights

    # ── Phase 3: Patching ─────────────────────────────────────────────────

    async def run_patching(self, auto_apply: bool = True) -> PatchSession:
        """Identify and fix bugs and quality issues."""
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized")

        # Get issues from evaluator
        if self._last_analysis is None:
            await self.run_evaluation()

        issues = []
        for fa in self._last_analysis.file_analyses.values():
            issues.extend(fa.issues)

        # Filter to actionable issues (bug risks and high-severity)
        actionable = [
            i for i in issues
            if i.category.value in ("bug_risk", "complexity", "performance")
            and i.severity.value in ("critical", "high", "medium")
            and i.auto_fixable or i.confidence > 0.5
        ]

        # Deduplicate by file + line
        seen = set()
        unique_issues = []
        for issue in actionable:
            key = (issue.file_path, issue.line, issue.title)
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)

        print(f"[Self-Improving] Phase 3: Patching {len(unique_issues)} issues (out of {len(issues)} total)...")
        start = time.time()

        session = await self.patcher.patch_all(
            issues=unique_issues,
            auto_apply=auto_apply,
            max_patches=self.max_patches,
        )

        # Record evolution events
        for result in session.results:
            if result.patch_applied:
                await self.evolution.record_event(
                    event_type="patch",
                    file_path=result.file_path,
                    description=f"Fixed: {result.issue_title}",
                    before_score=0,
                    after_score=result.safety_score,
                    generation=self._current_generation,
                )
            elif result.safety_approved and not result.patch_applied:
                await self.evolution.record_event(
                    event_type="patch",
                    file_path=result.file_path,
                    description=f"Patch generated (not applied): {result.issue_title}",
                    generation=self._current_generation,
                )

        elapsed = time.time() - start
        print(f"[Self-Improving] Patching complete in {elapsed:.1f}s. Applied: {session.patches_applied}, Blocked: {session.patches_blocked}")
        return session

    # ── Phase 4: Extension ───────────────────────────────────────────────

    async def run_extension(self) -> List[Any]:
        """Detect capability gaps and generate new tools."""
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized")

        print("[Self-Improving] Phase 4: Detecting capability gaps...")
        start = time.time()

        gaps = await self.extender.detect_capability_gap()

        if not gaps:
            print("[Self-Improving] No capability gaps detected.")
            return []

        results = []
        for gap in gaps[:self.max_extensions]:
            print(f"[Self-Improving] Generating tool for: {gap.description}")
            tool = await self.extender.generate_tool(gap)
            if tool:
                ext_result = await self.extender.integrate_tool(tool)
                results.append(ext_result)

                await self.evolution.record_event(
                    event_type="extend" if ext_result.tool_integrated else "evaluation",
                    file_path=tool.file_path,
                    description=f"{'Integrated' if ext_result.tool_integrated else 'Generated'} tool: {tool.tool_name} — {tool.description}",
                    before_score=0,
                    after_score=ext_result.safety_score,
                    generation=self._current_generation,
                )

                await asyncio.sleep(1)

        elapsed = time.time() - start
        integrated = sum(1 for r in results if r.tool_integrated)
        print(f"[Self-Improving] Extension complete in {elapsed:.1f}s. Generated: {len(results)}, Integrated: {integrated}")
        return results

    # ── Phase 5: Optimization ────────────────────────────────────────────

    async def run_optimization(self) -> List[Any]:
        """Profile tools and optimize bottlenecks."""
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized")

        print("[Self-Improving] Phase 5: Profiling and optimizing...")
        start = time.time()

        bottlenecks = await self.optimizer.identify_bottlenecks()

        if not bottlenecks:
            print("[Self-Improving] No performance bottlenecks detected.")
            return []

        results = []
        for bottleneck in bottlenecks[:self.max_optimizations]:
            print(f"[Self-Improving] Optimizing: {bottleneck.tool_name} ({bottleneck.avg_time:.2f}s avg)")
            result = await self.optimizer.optimize_bottleneck(bottleneck)
            results.append(result)

            if result.optimization_applied:
                await self.evolution.record_event(
                    event_type="optimize",
                    file_path=bottleneck.file_path,
                    description=f"Optimized {bottleneck.tool_name}: {bottleneck.suggested_strategy}",
                    before_score=result.before_avg_time,
                    after_score=result.after_avg_time,
                    generation=self._current_generation,
                )

            await asyncio.sleep(1)

        elapsed = time.time() - start
        applied = sum(1 for r in results if r.optimization_applied)
        print(f"[Self-Improving] Optimization complete in {elapsed:.1f}s. Applied: {applied}")
        return results

    # ── Full Cycle ────────────────────────────────────────────────────────

    async def run_full_cycle(self) -> ImprovementCycle:
        """Run all phases of the improvement cycle."""
        if not self._initialized:
            await self.initialize()

        cycle_start = time.time()
        cycle_id = f"cycle_{int(time.time())}"

        print(f"\n{'='*60}")
        print(f"[Self-Improving] Starting improvement cycle {cycle_id}")
        print(f"{'='*60}\n")

        # Get evolution score before
        evo_score_before = await self.evolution.get_evolution_score()
        gen_id = await self.evolution.start_generation()

        phases_completed = []
        errors = []

        # Phase 1: Evaluate
        try:
            analysis = await self.run_evaluation()
            score_before = analysis.overall_score
            grade_before = analysis.quality_grade
            phases_completed.append("evaluation")
        except Exception as e:
            errors.append(f"Evaluation failed: {e}")
            score_before = 0.0
            grade_before = "N/A"
            analysis = None

        # Phase 2: Learn
        try:
            learner_insights = await self.run_learning()
            phases_completed.append("learning")
        except Exception as e:
            errors.append(f"Learning failed: {e}")
            learner_insights = {}

        # Phase 3: Patch
        try:
            patch_session = await self.run_patching(auto_apply=True)
            phases_completed.append("patching")
        except Exception as e:
            errors.append(f"Patching failed: {e}")
            patch_session = None

        # Phase 4: Extend
        try:
            ext_results = await self.run_extension()
            phases_completed.append("extension")
        except Exception as e:
            errors.append(f"Extension failed: {e}")
            ext_results = []

        # Phase 5: Optimize
        try:
            opt_results = await self.run_optimization()
            phases_completed.append("optimization")
        except Exception as e:
            errors.append(f"Optimization failed: {e}")
            opt_results = []

        # Re-evaluate to measure improvement
        try:
            analysis_after = await self.run_evaluation()
            score_after = analysis_after.overall_score
            grade_after = analysis_after.quality_grade
        except Exception:
            score_after = score_before
            grade_after = grade_before

        # Get final evolution score
        evo_score_after = await self.evolution.get_evolution_score()
        await self.evolution.end_generation(gen_id, evo_score_before.overall_score, evo_score_after.overall_score)
        self._current_generation += 1

        # Get safety stats
        safety_stats = await self.safety.get_stats()

        # Build cycle report
        cycle = ImprovementCycle(
            cycle_id=cycle_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            generation=self._current_generation,
            duration=time.time() - cycle_start,
            phases_completed=phases_completed,
            evaluation_score_before=score_before,
            evaluation_score_after=score_after,
            evaluation_grade_before=grade_before,
            evaluation_grade_after=grade_after,
            patches_attempted=patch_session.patches_generated if patch_session else 0,
            patches_applied=patch_session.patches_applied if patch_session else 0,
            patches_blocked=patch_session.patches_blocked if patch_session else 0,
            extensions_generated=len(ext_results),
            extensions_integrated=sum(1 for r in ext_results if hasattr(r, 'tool_integrated') and r.tool_integrated),
            optimizations_attempted=len(opt_results),
            optimizations_applied=sum(1 for r in opt_results if hasattr(r, 'optimization_applied') and r.optimization_applied),
            rollbacks=0,
            evolution_score_before=evo_score_before.overall_score,
            evolution_score_after=evo_score_after.overall_score,
            safety_stats=safety_stats,
            learner_insights=learner_insights,
            errors=errors,
        )

        # Save evolution snapshot
        await self.evolution.save_evolution_snapshot(evo_score_after)
        await self.optimizer.save_performance_snapshot()

        print(f"\n{'='*60}")
        print(f"[Self-Improving] Cycle {cycle_id} complete")
        print(f"  Duration: {cycle.duration:.1f}s")
        print(f"  Quality: {cycle.evaluation_grade_before} → {cycle.evaluation_grade_after} ({cycle.evaluation_score_before:.2f} → {cycle.evaluation_score_after:.2f})")
        print(f"  Patches: {cycle.patches_applied} applied, {cycle.patches_blocked} blocked")
        print(f"  Extensions: {cycle.extensions_integrated} integrated")
        print(f"  Optimizations: {cycle.optimizations_applied} applied")
        print(f"  Evolution: {cycle.evolution_score_before:.3f} → {cycle.evolution_score_after:.3f}")
        if errors:
            print(f"  Errors: {len(errors)}")
        print(f"{'='*60}\n")

        return cycle

    # ── Status & Reports ─────────────────────────────────────────────────

    async def get_status(self) -> SystemStatus:
        """Get the current status of the self-improving system."""
        if not self._initialized:
            return SystemStatus(
                initialized=False,
                safety_enabled=False,
                evaluator_ready=False,
                patcher_ready=False,
                extender_ready=False,
                optimizer_ready=False,
                learner_ready=False,
                evolution_ready=False,
                total_files_analyzed=0,
                total_issues_found=0,
                total_patches_applied=0,
                total_tools_added=0,
                total_optimizations=0,
                evolution_score=0.0,
                generation=0,
                daily_changes_remaining=0,
                last_cycle_time="never",
            )

        evo_score = await self.evolution.get_evolution_score()
        safety_stats = await self.safety.get_stats()

        return SystemStatus(
            initialized=True,
            safety_enabled=True,
            evaluator_ready=True,
            patcher_ready=True,
            extender_ready=True,
            optimizer_ready=True,
            learner_ready=True,
            evolution_ready=True,
            total_files_analyzed=self._last_analysis.total_files if self._last_analysis else 0,
            total_issues_found=len(self._last_analysis.top_issues) if self._last_analysis else 0,
            total_patches_applied=safety_stats.get("applied_changes", 0),
            total_tools_added=len(self._last_analysis.total_functions) if self._last_analysis else 0,
            total_optimizations=safety_stats.get("total_changes", 0),
            evolution_score=evo_score.overall_score,
            generation=evo_score.generation,
            daily_changes_remaining=safety_stats.get("daily_limit_remaining", 0),
            last_cycle_time=evo_score.timestamp,
        )

    async def generate_full_report(self) -> str:
        """Generate a comprehensive report from all subsystems."""
        parts = []

        # Evolution overview
        evo_report = await self.evolution.generate_evolution_report()
        parts.append(evo_report)

        # Safety report
        safety_report = await self.safety.safety_report()
        parts.append("\n\n" + safety_report)

        # Evaluation report
        if self._last_analysis:
            eval_report = await self.evaluator.generate_report(self._last_analysis)
            parts.append("\n\n" + eval_report)

        # Learner report
        learner_report = await self.learner.generate_report()
        parts.append("\n\n" + learner_report)

        # Optimizer report
        opt_report = await self.optimizer.generate_report()
        parts.append("\n\n" + opt_report)

        return "".join(parts)

    # ── Autonomous Mode ───────────────────────────────────────────────────

    async def start_autonomous_mode(self) -> None:
        """Start periodic autonomous improvement cycles."""
        if not self._initialized:
            await self.initialize()

        self.auto_improve = True
        self._auto_task = asyncio.create_task(self._auto_improve_loop())
        print(f"[Self-Improving] Autonomous mode started. Running every {self.improvement_interval}s.")

    async def stop_autonomous_mode(self) -> None:
        """Stop autonomous improvement cycles."""
        self.auto_improve = False
        if self._auto_task:
            self._auto_task.cancel()
            self._auto_task = None
        print("[Self-Improving] Autonomous mode stopped.")

    async def _auto_improve_loop(self) -> None:
        """Background loop for autonomous improvement."""
        while self.auto_improve:
            try:
                await asyncio.sleep(self.improvement_interval)
                if self.auto_improve:
                    await self.run_full_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Self-Improving] Auto-cycle error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    # ── Quick Actions ────────────────────────────────────────────────────

    async def quick_scan(self) -> str:
        """Quick scan: evaluate only, no modifications."""
        if not self._initialized:
            await self.initialize()

        analysis = await self.run_evaluation()
        lines = [
            f"Quick Scan Results (Grade: {analysis.quality_grade}, Score: {analysis.overall_score:.2f})",
            f"Files: {analysis.total_files} | Lines: {analysis.total_lines:,} | Issues: {len(analysis.top_issues)}",
        ]

        if analysis.top_issues:
            lines.append(f"\nTop Issues ({len(analysis.top_issues)}):")
            for issue in analysis.top_issues[:5]:
                lines.append(f"  [{issue.severity.value}] {issue.file_path}:{issue.line} — {issue.title}")

        if analysis.improvement_suggestions:
            lines.append(f"\nSuggestions:")
            for s in analysis.improvement_suggestions[:3]:
                lines.append(f"  - {s}")

        return "\n".join(lines)

    async def record_feedback(self, prompt: str, response: str, accepted: bool, task_type: str = "general") -> str:
        """Record user feedback for learning."""
        if not self._initialized:
            return "Learner not initialized"

        signal = "accept" if accepted else "reject"
        record_id = await self.learner.record_interaction(
            prompt=prompt,
            response=response,
            signal=signal,
            task_type=task_type,
        )

        # Also log for gap detection
        self.extender.log_user_request(prompt, response, success=accepted)

        return f"Feedback recorded (id: {record_id})"
