"""
Self-Extender — The AI that grows its own capabilities.

Monitors for capability gaps and automatically generates new tools/functions:
- Detects when the agent encounters tasks it cannot handle
- Generates new tool implementations for common unmet needs
- Creates wrapper functions for frequently used multi-step patterns
- Builds new agent team roles based on observed workflows
- Generates tests and documentation for new capabilities
- Integrates new tools into the registry automatically

The extender follows the same safety protocol as the patcher:
all new code passes through SafetyGuardrails before being added.

Usage:
    extender = SelfExtender(agent, safety, evaluator, project_root="/path/to/claude_clone")
    await extender.initialize()
    gap = await extender.detect_capability_gap("compress files", "The user asked to compress files but no tool exists")
    tool = await extender.generate_tool(gap)
    result = await extender.integrate_tool(tool)
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TOOL_FILE = "agent/tools.py"

# Common capability patterns that trigger new tool generation
CAPABILITY_PATTERNS = {
    "file_compression": {
        "keywords": ["compress", "zip", "tar", "gzip", "archive", "extract", "unzip"],
        "description": "Compress/extract files and directories",
        "suggested_tools": ["compress_files", "extract_archive"],
    },
    "file_encryption": {
        "keywords": ["encrypt", "decrypt", "cipher", "aes", "password protect"],
        "description": "Encrypt and decrypt files",
        "suggested_tools": ["encrypt_file", "decrypt_file"],
    },
    "database_operations": {
        "keywords": ["database", "sql", "query", "table", "schema", "migration"],
        "description": "Database operations and management",
        "suggested_tools": ["query_database", "list_tables", "describe_schema"],
    },
    "api_testing": {
        "keywords": ["api test", "endpoint", "http request", "rest api", "curl"],
        "description": "API endpoint testing and interaction",
        "suggested_tools": ["test_api_endpoint", "api_request"],
    },
    "docker": {
        "keywords": ["docker", "container", "image", "dockerfile", "compose"],
        "description": "Docker container management",
        "suggested_tools": ["docker_build", "docker_run", "docker_ps"],
    },
    "cron_jobs": {
        "keywords": ["schedule", "cron", "periodic", "recurring", "timer"],
        "description": "Schedule periodic tasks",
        "suggested_tools": ["schedule_task", "list_schedules"],
    },
    "image_processing": {
        "keywords": ["image", "resize", "crop", "thumbnail", "screenshot"],
        "description": "Image file manipulation",
        "suggested_tools": ["resize_image", "crop_image", "convert_image"],
    },
    "csv_processing": {
        "keywords": ["csv", "spreadsheet", "excel", "xlsx", "data export"],
        "description": "CSV and spreadsheet processing",
        "suggested_tools": ["read_csv", "write_csv", "query_csv"],
    },
    "git_advanced": {
        "keywords": ["rebase", "cherry-pick", "stash", "bisect", "blame"],
        "description": "Advanced git operations",
        "suggested_tools": ["git_rebase", "git_stash", "git_cherry_pick"],
    },
    "code_metrics": {
        "keywords": ["lines of code", "loc", "complexity", "coverage", "technical debt"],
        "description": "Code quality metrics and reporting",
        "suggested_tools": ["code_metrics", "complexity_report"],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CapabilityGap:
    """A detected gap in the agent's capabilities."""
    gap_id: str
    category: str
    description: str
    evidence: str  # What triggered the detection
    frequency: int  # How often this gap was encountered
    suggested_tool_name: str
    suggested_tool_description: str
    priority: float  # 0.0 to 1.0
    timestamp: str


@dataclass
class GeneratedTool:
    """A newly generated tool ready for integration."""
    tool_name: str
    code: str
    description: str
    file_path: str
    parameters: List[Dict[str, str]]
    dependencies: List[str]
    safety_score: float
    test_code: str = ""
    doc_generated: bool = False
    integrated: bool = False


@dataclass
class ExtensionResult:
    """Result of an extension operation."""
    gap: CapabilityGap
    tool_generated: bool
    tool_integrated: bool
    safety_approved: bool
    safety_score: float
    error: str = ""
    time_taken: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# SelfExtender
# ──────────────────────────────────────────────────────────────────────────────

class SelfExtender:
    """
    Detects capability gaps and generates new tools to fill them.

    Workflow:
    1. Monitor user requests and agent failures for unmet capabilities
    2. When a gap is detected frequently, generate a new tool
    3. Pass the new tool through safety evaluation
    4. Integrate into the tools registry if approved
    5. Generate tests and documentation
    """

    def __init__(
        self,
        agent: Any,
        safety: Any,
        evaluator: Any,
        project_root: str,
    ):
        self.agent = agent
        self.safety = safety
        self.evaluator = evaluator
        self.project_root = Path(project_root).resolve()
        self._detected_gaps: Dict[str, CapabilityGap] = {}
        self._generated_tools: List[GeneratedTool] = []
        self._request_log: List[Dict[str, Any]] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Scan existing tools to build the capability baseline."""
        await self._scan_existing_tools()
        self._initialized = True

    async def _scan_existing_tools(self) -> Set[str]:
        """Scan the existing tools registry to know what's available."""
        try:
            tools_file = self.project_root / TOOL_FILE
            if tools_file.exists():
                content = tools_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
                tool_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name.startswith("_"):
                            continue
                        tool_names.add(node.name)
                self._existing_tools = tool_names
            else:
                self._existing_tools = set()
        except Exception:
            self._existing_tools = set()

        return self._existing_tools

    # ── Gap Detection ─────────────────────────────────────────────────────

    def log_user_request(self, user_message: str, agent_response: str = "", success: bool = True) -> None:
        """Log a user request for gap analysis."""
        self._request_log.append({
            "message": user_message.lower(),
            "response": agent_response.lower() if agent_response else "",
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Keep log bounded
        if len(self._request_log) > 1000:
            self._request_log = self._request_log[-500:]

    async def detect_capability_gap(
        self, category_hint: str = "", evidence: str = ""
    ) -> List[CapabilityGap]:
        """
        Analyze request logs to detect capability gaps.

        Returns a list of detected gaps, sorted by priority.
        """
        gaps: List[CapabilityGap] = []

        # Analyze request log for patterns
        keyword_freq: Dict[str, int] = {}
        for entry in self._request_log:
            msg = entry["message"]
            if not entry["success"]:
                for cat_name, cat_info in CAPABILITY_PATTERNS.items():
                    for keyword in cat_info["keywords"]:
                        if keyword in msg:
                            key = f"{cat_name}:{keyword}"
                            keyword_freq[key] = keyword_freq.get(key, 0) + 1

        # Check for gaps where keyword frequency is high but no tool exists
        for cat_name, cat_info in CAPABILITY_PATTERNS.items():
            total_freq = sum(
                freq for key, freq in keyword_freq.items()
                if key.startswith(f"{cat_name}:")
            )
            if total_freq < 3:
                continue

            # Check if tool already exists
            for suggested in cat_info["suggested_tools"]:
                if suggested not in self._existing_tools:
                    gap_id = hashlib.sha256(f"{cat_name}_{suggested}".encode()).hexdigest()[:12]
                    priority = min(1.0, total_freq / 10.0)

                    gap = CapabilityGap(
                        gap_id=gap_id,
                        category=cat_name,
                        description=cat_info["description"],
                        evidence=f"Keyword matched {total_freq} times in failed requests",
                        frequency=total_freq,
                        suggested_tool_name=suggested,
                        suggested_tool_description=f"Tool for {cat_info['description']}",
                        priority=round(priority, 2),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    gaps.append(gap)
                    self._detected_gaps[gap_id] = gap

        # Also check explicit hints
        if category_hint:
            for cat_name, cat_info in CAPABILITY_PATTERNS.items():
                if cat_name == category_hint or category_hint in cat_info["description"].lower():
                    for suggested in cat_info["suggested_tools"]:
                        if suggested not in self._existing_tools:
                            gap_id = hashlib.sha256(f"{cat_name}_{suggested}".encode()).hexdigest()[:12]
                            if gap_id not in self._detected_gaps:
                                gap = CapabilityGap(
                                    gap_id=gap_id,
                                    category=cat_name,
                                    description=cat_info["description"],
                                    evidence=evidence or "Explicit category hint provided",
                                    frequency=1,
                                    suggested_tool_name=suggested,
                                    suggested_tool_description=f"Tool for {cat_info['description']}",
                                    priority=0.8,
                                    timestamp=datetime.now(timezone.utc).isoformat(),
                                )
                                gaps.append(gap)
                                self._detected_gaps[gap_id] = gap

        gaps.sort(key=lambda g: -g.priority)
        return gaps

    # ── Tool Generation ───────────────────────────────────────────────────

    async def generate_tool(self, gap: CapabilityGap) -> Optional[GeneratedTool]:
        """
        Use the agent to generate a new tool implementation for a detected gap.

        Returns a GeneratedTool if successful, None otherwise.
        """
        try:
            # Read existing tools for context/style reference
            tools_file = self.project_root / TOOL_FILE
            existing_code = ""
            if tools_file.exists():
                content = tools_file.read_text(encoding="utf-8")
                # Get the first 100 lines for style reference
                lines = content.split("\n")[:100]
                existing_code = "\n".join(lines)

            prompt = (
                f"You are extending your own toolset. You need to create a new async tool function.\n\n"
                f"Tool name: {gap.suggested_tool_name}\n"
                f"Description: {gap.suggested_tool_description}\n"
                f"Category: {gap.category}\n"
                f"Evidence of need: {gap.evidence}\n\n"
                f"STYLE REFERENCE (follow this exact style):\n"
                f"```\n{existing_code}\n```\n\n"
                f"REQUIREMENTS:\n"
                f"1. Function must be async (def with async keyword)\n"
                f"2. Follow the docstring format with param_name (type): — description\n"
                f"3. Use Path from pathlib for file operations\n"
                f"4. Return string results (not dicts unless it's a run_command-like tool)\n"
                f"5. Include proper error handling with try/except\n"
                f"6. Include a _is_path_safe check for any file path parameters\n"
                f"7. Output ONLY the function code — no markdown, no explanation\n"
                f"8. Include the docstring and function definition\n"
            )

            self.agent.reset()
            full_response = ""
            async for event in self.agent.run(prompt):
                from agent.core import TextEvent
                if isinstance(event, TextEvent):
                    full_response += event.data

            code = self._extract_code(full_response)
            if not code or "async def" not in code:
                return None

            # Extract parameters from the function signature
            parameters = self._extract_parameters(code)

            # Extract dependencies (imports)
            dependencies = self._extract_dependencies(code)

            tool = GeneratedTool(
                tool_name=gap.suggested_tool_name,
                code=code,
                description=gap.suggested_tool_description,
                file_path=TOOL_FILE,
                parameters=parameters,
                dependencies=dependencies,
                safety_score=0.0,
            )

            self._generated_tools.append(tool)
            return tool

        except Exception as e:
            return None

    async def generate_wrapper_tool(
        self,
        pattern_name: str,
        tool_sequence: List[str],
        description: str,
    ) -> Optional[GeneratedTool]:
        """
        Generate a wrapper tool that chains multiple existing tools into one.

        This is for commonly used multi-step workflows.
        """
        try:
            prompt = (
                f"Create a new async tool function that combines these existing tools "
                f"into a single convenient operation.\n\n"
                f"Wrapper name: {pattern_name}\n"
                f"Tools to chain: {', '.join(tool_sequence)}\n"
                f"Description: {description}\n\n"
                f"IMPORTANT: This is a WRAPPER — it should internally call the existing "
                f"tool functions (via tool_func(**kwargs) pattern). The wrapper should:\n"
                f"1. Accept high-level parameters from the user\n"
                f"2. Break the task into steps using the chained tools\n"
                f"3. Return a combined result string\n"
                f"4. Include proper error handling\n\n"
                f"Output ONLY the function code with docstring."
            )

            self.agent.reset()
            full_response = ""
            async for event in self.agent.run(prompt):
                from agent.core import TextEvent
                if isinstance(event, TextEvent):
                    full_response += event.data

            code = self._extract_code(full_response)
            if not code or "async def" not in code:
                return None

            tool = GeneratedTool(
                tool_name=pattern_name,
                code=code,
                description=description,
                file_path=TOOL_FILE,
                parameters=self._extract_parameters(code),
                dependencies=tool_sequence,
                safety_score=0.0,
            )

            self._generated_tools.append(tool)
            return tool

        except Exception:
            return None

    # ── Integration ───────────────────────────────────────────────────────

    async def integrate_tool(self, tool: GeneratedTool) -> ExtensionResult:
        """
        Integrate a generated tool into the codebase.

        Passes through safety gates, then appends to the tools file
        and updates the registry.
        """
        from agent.self_improving.safety import ChangeType, ChangeHistory

        start_time = time.time()
        result = ExtensionResult(
            gap=CapabilityGap(
                gap_id="", category="", description="", evidence="",
                frequency=0, suggested_tool_name=tool.tool_name,
                suggested_tool_description=tool.description,
                priority=0.0, timestamp="",
            ),
            tool_generated=True,
            tool_integrated=False,
            safety_approved=False,
            safety_score=0.0,
            time_taken=0.0,
        )

        try:
            # Read current tools file
            tools_path = self.project_root / tool.file_path
            if not tools_path.exists():
                result.error = f"Tools file not found: {tool.file_path}"
                return result

            original_code = tools_path.read_text(encoding="utf-8")

            # Check if tool already exists
            if tool.tool_name in original_code:
                result.error = f"Tool '{tool.tool_name}' already exists in {tool.file_path}"
                return result

            # Proposed code = original + new tool
            proposed_code = original_code.rstrip() + "\n\n\n" + tool.code + "\n"

            # Safety evaluation
            safety_result = await self.safety.evaluate_change(
                file_path=tool.file_path,
                original_code=original_code,
                proposed_code=proposed_code,
                change_type=ChangeType.EXTENSION,
                reason=f"New tool: {tool.tool_name} — {tool.description}",
            )

            result.safety_approved = safety_result.approved
            result.safety_score = safety_result.overall_score
            tool.safety_score = safety_result.overall_score

            if not safety_result.approved:
                result.error = f"Safety blocked: {', '.join(safety_result.blockers)}"
                return result

            # Backup and apply
            backup_id = await self.safety.backup_file(
                tool.file_path,
                change_type=ChangeType.EXTENSION,
                reason=f"New tool: {tool.tool_name}",
            )

            await self.safety.apply_change(tool.file_path, proposed_code)

            change_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
            await self.safety.record_change(ChangeHistory(
                change_id=change_id,
                file_path=tool.file_path,
                change_type=ChangeType.EXTENSION,
                original_hash=hashlib.sha256(original_code.encode()).hexdigest()[:16],
                new_hash=hashlib.sha256(proposed_code.encode()).hexdigest()[:16],
                backup_id=backup_id,
                approval_level=safety_result.approval_level,
                safety_score=safety_result.overall_score,
                reason=f"New tool: {tool.tool_name} — {tool.description}",
                applied=True,
                rolled_back=False,
                timestamp=datetime.now(timezone.utc).isoformat(),
                gates_summary={g.gate_name: g.result.value for g in safety_result.gates},
            ))

            tool.integrated = True
            result.tool_integrated = True
            self._existing_tools.add(tool.tool_name)

        except Exception as e:
            result.error = f"Integration error: {e}"

        result.time_taken = time.time() - start_time
        return result

    # ── Helper Methods ───────────────────────────────────────────────────

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from a response."""
        patterns = [
            r"```python\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return max(matches, key=len).strip()
        return None

    def _extract_parameters(self, code: str) -> List[Dict[str, str]]:
        """Extract parameter info from a function definition."""
        params = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for arg in node.args.args:
                        if arg.arg in ("self", "cls"):
                            continue
                        param_type = "str"
                        hint = node.returns
                        params.append({
                            "name": arg.arg,
                            "type": param_type,
                        })
                    break
        except SyntaxError:
            pass
        return params

    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract import dependencies from code."""
        deps = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        deps.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        deps.append(node.module)
        except SyntaxError:
            pass
        return deps

    # ── Queries ───────────────────────────────────────────────────────────

    async def get_detected_gaps(self) -> List[CapabilityGap]:
        """Get all detected capability gaps."""
        return sorted(self._detected_gaps.values(), key=lambda g: -g.priority)

    async def get_generated_tools(self) -> List[GeneratedTool]:
        """Get all generated tools (including non-integrated ones)."""
        return list(self._generated_tools)

    async def get_integration_stats(self) -> Dict[str, Any]:
        """Get statistics about tool extension."""
        total = len(self._generated_tools)
        integrated = sum(1 for t in self._generated_tools if t.integrated)
        avg_safety = (
            sum(t.safety_score for t in self._generated_tools if t.safety_score > 0)
            / max(1, sum(1 for t in self._generated_tools if t.safety_score > 0))
        )
        return {
            "total_generated": total,
            "integrated": integrated,
            "pending": total - integrated,
            "average_safety_score": round(avg_safety, 2),
            "existing_tools_count": len(self._existing_tools),
            "detected_gaps": len(self._detected_gaps),
        }
