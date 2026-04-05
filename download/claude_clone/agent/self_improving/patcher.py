"""
Self-Patcher — The AI that fixes its own bugs.

Takes analysis results from the evaluator and generates verified patches:
- Reads current file content and identified issues
- Generates targeted fixes using the agent's own AI capabilities
- Verifies each patch passes safety gates (syntax, imports, patterns)
- Applies patches atomically with backup/rollback
- Tracks patch success rate and learns from failures
- Supports batch patching (fix all issues of a type at once)
- Generates patch descriptions and changelog entries

Usage:
    patcher = SelfPatcher(agent, safety, project_root="/path/to/claude_clone")
    await patcher.initialize()
    results = await patcher.patch_issue(issue, auto_apply=True)
    batch = await patcher.patch_all(issues, auto_apply=False)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.self_improving.safety import (
    SafetyGuardrails, ChangeType, ApprovalLevel, SafetyEvaluation,
    ChangeHistory,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PatchResult:
    """Result of a single patch attempt."""
    issue_title: str
    file_path: str
    line: int
    patch_generated: bool
    patch_applied: bool
    safety_approved: bool
    safety_score: float
    backup_id: str = ""
    change_id: str = ""
    original_code: str = ""
    patched_code: str = ""
    error: str = ""
    verification_passed: bool = False
    time_taken: float = 0.0


@dataclass
class PatchSession:
    """A batch patching session with aggregated results."""
    session_id: str
    timestamp: str
    total_issues: int
    patches_generated: int
    patches_applied: int
    patches_failed: int
    patches_blocked: int
    results: List[PatchResult]
    total_time: float = 0.0
    auto_applied: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Fix Templates
# ──────────────────────────────────────────────────────────────────────────────

# Pre-built fix templates for common issues that don't need AI intervention
FIX_TEMPLATES = {
    "Mutable default argument": {
        "pattern": r"(def\s+\w+\s*\([^)]*=)(\[[\s\S]*?\]|(\{[\s\S]*?\}))",
        "replacement": r"\1None",
        "insert_after_def": "    if {param} is None:\n        {param} = {default}",
        "description": "Replace mutable default with None and initialize in body",
    },
    "Bare except clause": {
        "pattern": r"except\s*:",
        "replacement": "except Exception:",
        "description": "Replace bare except with except Exception",
    },
    "Comparison with None using ==": {
        "pattern": r"(\w+)\s*==\s*None",
        "replacement": r"\1 is None",
        "description": "Replace == None with 'is None'",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# SelfPatcher
# ──────────────────────────────────────────────────────────────────────────────

class SelfPatcher:
    """
    Generates and applies patches to fix issues found by the SelfEvaluator.

    Uses a two-phase approach:
    1. Template-based fixes for simple, well-known patterns
    2. AI-generated fixes for complex issues (using the agent's capabilities)

    Every patch passes through the SafetyGuardrails system before being applied.
    """

    def __init__(
        self,
        agent: Any,
        safety: SafetyGuardrails,
        project_root: str,
    ):
        self.agent = agent
        self.safety = safety
        self.project_root = Path(project_root).resolve()
        self._patch_history: List[PatchResult] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Prepare the patcher for use."""
        self._initialized = True

    # ── Single Issue Patching ─────────────────────────────────────────────

    async def patch_issue(
        self,
        issue: Any,
        auto_apply: bool = True,
    ) -> PatchResult:
        """
        Attempt to patch a single issue.

        Parameters
        ----------
        issue:
            A CodeIssue from the SelfEvaluator.
        auto_apply:
            If True, automatically apply approved patches. If False, only generate.

        Returns
        -------
        PatchResult with full details of the attempt.
        """
        start_time = time.time()
        result = PatchResult(
            issue_title=issue.title,
            file_path=issue.file_path,
            line=issue.line,
            patch_generated=False,
            patch_applied=False,
            safety_approved=False,
            safety_score=0.0,
            time_taken=0.0,
        )

        try:
            # Read the current file
            full_path = self.project_root / issue.file_path
            if not full_path.exists():
                result.error = f"File not found: {issue.file_path}"
                return result

            original_code = full_path.read_text(encoding="utf-8")
            result.original_code = original_code

            # Try template-based fix first
            patched_code = self._apply_template_fix(original_code, issue)

            # If template didn't work, use AI-generated fix
            if patched_code is None:
                patched_code = await self._generate_ai_fix(original_code, issue)

            if patched_code is None or patched_code == original_code:
                result.error = "Could not generate a patch"
                return result

            result.patched_code = patched_code
            result.patch_generated = True

            # Run through safety gates
            safety_result = await self.safety.evaluate_change(
                file_path=issue.file_path,
                original_code=original_code,
                proposed_code=patched_code,
                change_type=ChangeType.PATCH,
                reason=f"Fix: {issue.title} ({issue.category.value})",
            )

            result.safety_approved = safety_result.approved
            result.safety_score = safety_result.overall_score

            if not safety_result.approved:
                result.error = f"Safety blocked: {', '.join(safety_result.blockers)}"
                # Quarantine if high risk but potentially valuable
                if safety_result.estimated_risk > 0.3 and safety_result.estimated_risk < 0.7:
                    await self.safety.quarantine_change(
                        file_path=issue.file_path,
                        proposed_code=patched_code,
                        reason=f"{issue.title}: {', '.join(safety_result.blockers)}",
                        risk_score=safety_result.estimated_risk,
                    )
                return result

            if not auto_apply:
                return result

            # Apply the patch
            import hashlib
            backup_id = await self.safety.backup_file(
                issue.file_path,
                change_type=ChangeType.PATCH,
                reason=f"Fix: {issue.title}",
            )
            result.backup_id = backup_id

            new_hash = await self.safety.apply_change(issue.file_path, patched_code)

            # Verify the patch was applied correctly
            verification = full_path.read_text(encoding="utf-8")
            result.verification_passed = (verification == patched_code)

            # Record in change history
            change_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
            await self.safety.record_change(ChangeHistory(
                change_id=change_id,
                file_path=issue.file_path,
                change_type=ChangeType.PATCH,
                original_hash=hashlib.sha256(original_code.encode()).hexdigest()[:16],
                new_hash=new_hash,
                backup_id=backup_id,
                approval_level=safety_result.approval_level,
                safety_score=safety_result.overall_score,
                reason=f"Fix: {issue.title}",
                applied=True,
                rolled_back=False,
                timestamp=datetime.now(timezone.utc).isoformat(),
                gates_summary={g.gate_name: g.result.value for g in safety_result.gates},
            ))
            result.change_id = change_id

            result.patch_applied = True
            await self.safety._record_change_rate(issue.file_path)

        except Exception as e:
            result.error = f"Patching error: {e}"
            # Attempt rollback if we got far enough
            if result.backup_id and not result.patch_applied:
                await self.safety.rollback(result.backup_id)

        result.time_taken = time.time() - start_time
        self._patch_history.append(result)
        return result

    def _apply_template_fix(self, original_code: str, issue: Any) -> Optional[str]:
        """Try to fix an issue using a pre-built template."""
        if issue.title not in FIX_TEMPLATES:
            return None

        template = FIX_TEMPLATES[issue.title]
        pattern = template.get("pattern", "")
        replacement = template.get("replacement", "")

        if not pattern:
            return None

        new_code = re.sub(pattern, replacement, original_code, count=1)
        if new_code != original_code:
            return new_code

        return None

    async def _generate_ai_fix(self, original_code: str, issue: Any) -> Optional[str]:
        """
        Use the agent to generate a fix for a complex issue.

        The agent is given the file content and issue description,
        and asked to produce only the modified file content.
        """
        try:
            # Extract the relevant lines around the issue
            lines = original_code.split("\n")
            issue_line = max(0, issue.line - 1)
            context_start = max(0, issue_line - 10)
            context_end = min(len(lines), issue_line + 15)
            context_lines = lines[context_start:context_end]
            context = "\n".join(
                f"{i+1}: {line}" for i, line in enumerate(context_lines)
            )

            prompt = (
                f"You are fixing a bug in your own codebase. Here is the context:\n\n"
                f"File: {issue.file_path}\n"
                f"Line {issue.line}: {issue.title}\n"
                f"Severity: {issue.severity.value}\n"
                f"Category: {issue.category.value}\n\n"
                f"Description: {issue.description}\n\n"
                f"Context (lines {context_start+1}-{context_end}):\n"
                f"{context}\n\n"
                f"Full file content:\n"
                f"```\n{original_code}\n```\n\n"
                f"Suggestion: {issue.suggestion}\n\n"
                f"IMPORTANT: Output ONLY the complete modified file content. "
                f"Do not include explanations, markdown fences, or comments about the change. "
                f"Fix the specific issue while preserving all other code exactly as-is."
            )

            # Use the agent to generate the fix
            self.agent.reset()
            full_response = ""
            async for event in self.agent.run(prompt):
                from agent.core import TextEvent
                if isinstance(event, TextEvent):
                    full_response += event.data

            # Extract code from response (handle markdown fences)
            code = self._extract_code_from_response(full_response)
            if not code:
                code = full_response.strip()

            if code and code != original_code:
                return code

            return None

        except Exception:
            return None

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from an agent response that may contain markdown fences."""
        # Try fenced code blocks
        patterns = [
            r"```python\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                # Return the longest match (most likely the complete file)
                return max(matches, key=len).strip()

        return None

    # ── Batch Patching ────────────────────────────────────────────────────

    async def patch_all(
        self,
        issues: List[Any],
        auto_apply: bool = False,
        max_patches: int = 20,
    ) -> PatchSession:
        """
        Attempt to patch multiple issues in a batch.

        Parameters
        ----------
        issues:
            List of CodeIssue objects from SelfEvaluator.
        auto_apply:
            If True, auto-apply approved patches.
        max_patches:
            Maximum number of patches to attempt.
        """
        import hashlib
        session_id = hashlib.sha256(os.urandom(32)).hexdigest()[:12]
        start_time = time.time()

        session = PatchSession(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_issues=len(issues),
            patches_generated=0,
            patches_applied=0,
            patches_failed=0,
            patches_blocked=0,
            results=[],
            auto_applied=auto_apply,
        )

        # Sort by severity (fix critical/high first)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        sorted_issues = sorted(issues, key=lambda i: (severity_order.get(i.severity.value, 5), i.line))

        applied_count = 0
        for issue in sorted_issues:
            if applied_count >= max_patches:
                break

            result = await self.patch_issue(issue, auto_apply=auto_apply)
            session.results.append(result)

            if result.patch_applied:
                session.patches_applied += 1
                applied_count += 1
            elif result.patch_generated and not result.safety_approved:
                session.patches_blocked += 1
            elif not result.patch_generated:
                session.patches_failed += 1

            if result.patch_generated:
                session.patches_generated += 1

            # Brief pause between patches
            await asyncio.sleep(0.5)

        session.total_time = time.time() - start_time
        return session

    # ── Rollback ──────────────────────────────────────────────────────────

    async def rollback_last(self, count: int = 1) -> int:
        """Rollback the last N applied patches."""
        rolled_back = 0
        for result in reversed(self._patch_history):
            if result.patch_applied and result.backup_id:
                success = await self.safety.rollback(result.backup_id)
                if success:
                    rolled_back += 1
                    if rolled_back >= count:
                        break
        return rolled_back

    # ── History ───────────────────────────────────────────────────────────

    async def get_patch_history(self) -> List[PatchResult]:
        """Get the history of all patch attempts in this session."""
        return list(self._patch_history)

    async def patch_success_rate(self) -> float:
        """Calculate the success rate of patches."""
        if not self._patch_history:
            return 0.0
        applied = sum(1 for r in self._patch_history if r.patch_applied)
        return applied / len(self._patch_history)

    # ── Report ────────────────────────────────────────────────────────────

    async def generate_report(self, session: PatchSession = None) -> str:
        """Generate a human-readable patching report."""
        history = self._patch_history
        applied = sum(1 for r in history if r.patch_applied)
        blocked = sum(1 for r in history if r.patch_generated and not r.patch_applied)
        failed = sum(1 for r in history if not r.patch_generated)
        avg_score = (
            sum(r.safety_score for r in history if r.safety_score > 0)
            / max(1, sum(1 for r in history if r.safety_score > 0))
        )

        lines = [
            "# Self-Patch Report",
            "",
            f"**Total Attempts:** {len(history)}",
            f"**Applied:** {applied}",
            f"**Blocked by Safety:** {blocked}",
            f"**Failed to Generate:** {failed}",
            f"**Success Rate:** {self.patch_success_rate():.0%}" if history else "**Success Rate:** N/A",
            f"**Avg Safety Score:** {avg_score:.2f}" if history else "**Avg Safety Score:** N/A",
            "",
        ]

        if session:
            lines.append("## Session Summary")
            lines.append(f"- Session ID: {session.session_id}")
            lines.append(f"- Time: {session.total_time:.1f}s")
            lines.append(f"- Auto-applied: {session.auto_applied}")
            lines.append("")

        if history:
            lines.append("## Recent Patches")
            for r in reversed(history[-20:]):
                status = "APPLIED" if r.patch_applied else ("BLOCKED" if r.safety_approved else "FAILED")
                lines.append(
                    f"- [{status}] {r.file_path}:{r.line} — {r.issue_title} "
                    f"(score={r.safety_score:.2f}, time={r.time_taken:.1f}s)"
                )
                if r.error:
                    lines.append(f"  Error: {r.error}")

        return "\n".join(lines)
