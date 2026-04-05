"""
Self-Evaluator — The AI that analyzes its own code.

Implements deep static analysis of the claude_clone codebase:
- AST-based code quality scoring (complexity, coupling, coherence)
- Bug pattern detection (common Python anti-patterns, edge cases)
- Architecture health monitoring (dependency graphs, circular imports)
- Dead code detection (unused functions, unreachable branches)
- Documentation coverage analysis
- Consistency checking (naming conventions, error handling patterns)
- Comparison scoring against previous snapshots (improvement tracking)

This module does NOT modify code — it only reads and analyzes.
Modifications are handled by the patcher module.

Usage:
    evaluator = SelfEvaluator(project_root="/path/to/claude_clone")
    await evaluator.initialize()
    report = await evaluator.analyze_file("agent/tools.py")
    full_report = await evaluator.analyze_project()
    issues = await evaluator.find_bug_patterns()
    health = await evaluator.architecture_health()
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "~/.claude_clone/self_improve_eval.db"

# File extensions to analyze
PYTHON_EXTENSIONS = {".py"}

# Directories to skip
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "env", ".env",
}

# Complexity thresholds
MAX_FUNCTION_COMPLEXITY = 15
MAX_CLASS_COMPLEXITY = 50
MAX_FUNCTION_LINES = 100
MAX_FILE_LINES = 1500


# ──────────────────────────────────────────────────────────────────────────────
# Enums & Data Classes
# ──────────────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(str, Enum):
    BUG_RISK = "bug_risk"
    COMPLEXITY = "complexity"
    DEAD_CODE = "dead_code"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CONSISTENCY = "consistency"


@dataclass
class CodeIssue:
    """A single issue found during self-analysis."""
    file_path: str
    line: int
    severity: Severity
    category: IssueCategory
    title: str
    description: str
    suggestion: str = ""
    auto_fixable: bool = False
    confidence: float = 0.0  # 0.0 to 1.0


@dataclass
class FileAnalysis:
    """Analysis result for a single file."""
    file_path: str
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    functions: int
    classes: int
    avg_function_complexity: float
    max_function_complexity: int
    docstring_coverage: float  # 0.0 to 1.0
    issues: List[CodeIssue]
    overall_score: float  # 0.0 to 1.0
    quality_grade: str  # A, B, C, D, F


@dataclass
class ProjectAnalysis:
    """Analysis result for the entire project."""
    total_files: int
    total_lines: int
    total_functions: int
    total_classes: int
    avg_complexity: float
    max_complexity: int
    avg_docstring_coverage: float
    issues_by_severity: Dict[str, int]
    issues_by_category: Dict[str, int]
    top_issues: List[CodeIssue]
    file_analyses: Dict[str, FileAnalysis]
    overall_score: float
    quality_grade: str
    architecture_health: Dict[str, Any]
    improvement_suggestions: List[str]
    timestamp: str


@dataclass
class FunctionMetrics:
    """Metrics for a single function."""
    name: str
    file_path: str
    line_start: int
    line_end: int
    complexity: int
    lines: int
    parameters: int
    has_docstring: bool
    is_async: bool
    calls: List[str]
    called_by: List[str]
    nest_depth: int


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    name: str
    file_path: str
    line_start: int
    line_end: int
    methods: int
    complexity: int
    has_docstring: bool
    bases: List[str]
    decorators: List[str]


@dataclass
class AnalysisSnapshot:
    """A point-in-time snapshot of project analysis for comparison."""
    snapshot_id: str
    timestamp: str
    overall_score: float
    total_issues: int
    total_files: int
    total_lines: int
    avg_complexity: float
    docstring_coverage: float
    file_scores: Dict[str, float]
    issues_by_category: Dict[str, int]


# ──────────────────────────────────────────────────────────────────────────────
# SelfEvaluator
# ──────────────────────────────────────────────────────────────────────────────

class SelfEvaluator:
    """
    Deep static analysis engine for the claude_clone codebase.

    Reads and analyzes all Python files in the project to identify:
    - Bug risks and anti-patterns
    - Complexity hotspots
    - Dead code and unused symbols
    - Documentation gaps
    - Architecture issues (circular imports, tight coupling)
    - Consistency violations
    - Performance anti-patterns
    """

    def __init__(self, project_root: str, db_path: str = DEFAULT_DB_PATH):
        self.project_root = Path(project_root).resolve()
        self.db_path = Path(db_path).expanduser().resolve()
        self._conn: Optional[sqlite3.Connection] = None
        self._file_cache: Dict[str, Tuple[str, ast.AST]] = {}
        self._function_registry: Dict[str, FunctionMetrics] = {}
        self._class_registry: Dict[str, ClassMetrics] = {}
        self._import_graph: Dict[str, Set[str]] = defaultdict(set)
        self._call_graph: Dict[str, Set[str]] = defaultdict(set)
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Scan the project, build registries, and prepare for analysis."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(self._connect)
        await self._run_sync(self._create_tables)
        await self._scan_project()
        await self._build_registries()
        self._initialized = True

    async def close(self) -> None:
        if self._conn is not None:
            await self._run_sync(self._conn.close)
            self._conn = None

    def _connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id    TEXT PRIMARY KEY,
                timestamp      TEXT NOT NULL,
                overall_score  REAL DEFAULT 0.0,
                total_issues   INTEGER DEFAULT 0,
                total_files    INTEGER DEFAULT 0,
                total_lines    INTEGER DEFAULT 0,
                avg_complexity REAL DEFAULT 0.0,
                doc_coverage   REAL DEFAULT 0.0,
                file_scores    TEXT DEFAULT '{}',
                issues_by_cat  TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS issues_log (
                id          TEXT PRIMARY KEY,
                file_path   TEXT NOT NULL,
                line        INTEGER DEFAULT 0,
                severity    TEXT NOT NULL,
                category    TEXT NOT NULL,
                title       TEXT NOT NULL,
                description TEXT DEFAULT '',
                suggestion  TEXT DEFAULT '',
                timestamp   TEXT NOT NULL,
                resolved    INTEGER DEFAULT 0,
                snapshot_id TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_issues_file ON issues_log(file_path);
            CREATE INDEX IF NOT EXISTS idx_issues_severity ON issues_log(severity);
            CREATE INDEX IF NOT EXISTS idx_issues_category ON issues_log(category);
            CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(timestamp);
        """)
        self._conn.commit()

    async def _run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SelfEvaluator not initialized.")
        return self._conn

    # ── Project Scanning ──────────────────────────────────────────────────

    async def _scan_project(self) -> None:
        """Discover and parse all Python files in the project."""
        self._file_cache = {}

        def _scan() -> None:
            for py_file in self.project_root.rglob("*.py"):
                # Skip non-project directories
                rel = py_file.relative_to(self.project_root)
                if any(part in SKIP_DIRS for part in rel.parts):
                    continue
                try:
                    content = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(content, filename=str(py_file))
                    self._file_cache[str(rel)] = (content, tree)
                except (SyntaxError, UnicodeDecodeError, OSError):
                    pass  # Skip unparseable files

        await self._run_sync(_scan)

    async def _build_registries(self) -> None:
        """Build function, class, import, and call registries."""
        def _build() -> None:
            self._function_registry = {}
            self._class_registry = {}
            self._import_graph = defaultdict(set)
            self._call_graph = defaultdict(set)

            for file_path, (content, tree) in self._file_cache.items():
                # Extract imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module = alias.name.split(".")[0]
                            self._import_graph[file_path].add(module)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.level == 0:
                            module = node.module.split(".")[0]
                            self._import_graph[file_path].add(module)
                        elif node.level > 0:
                            # Relative import — resolve relative path
                            parts = file_path.replace("/", ".").split(".")
                            parent = ".".join(parts[:-1 - node.level]) if node.level < len(parts) else ""
                            if node.module:
                                parent = f"{parent}.{node.module}" if parent else node.module
                            self._import_graph[file_path].add(parent)

                # Extract functions and classes
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self._register_function(file_path, node, content)
                    elif isinstance(node, ast.ClassDef):
                        self._register_class(file_path, node, content)

        await self._run_sync(_build)

    def _register_function(self, file_path: str, node: ast.FunctionDef, content: str) -> None:
        """Register a function with its metrics."""
        key = f"{file_path}::{node.name}"
        complexity = self._compute_complexity(node)
        lines = node.end_lineno - node.lineno + 1 if hasattr(node, "end_lineno") and node.end_lineno else 0
        params = len(node.args.args) + len(node.args.posonlyargs or []) + len(node.args.kwonlyargs or [])
        if node.args.vararg:
            params += 1
        if node.args.kwarg:
            params += 1

        calls = self._extract_calls(node)
        docstring = ast.get_docstring(node) or ""

        self._function_registry[key] = FunctionMetrics(
            name=node.name,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            complexity=complexity,
            lines=lines,
            parameters=params,
            has_docstring=bool(docstring),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            calls=calls,
            called_by=[],
            nest_depth=self._compute_nest_depth(node),
        )

        # Update call graph
        for call in calls:
            self._call_graph[key].add(call)

    def _register_class(self, file_path: str, node: ast.ClassDef, content: str) -> None:
        """Register a class with its metrics."""
        key = f"{file_path}::{node.name}"
        methods = sum(
            1 for child in ast.iter_child_nodes(node)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        complexity = sum(
            self._compute_complexity(child)
            for child in ast.iter_child_nodes(node)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)
        docstring = ast.get_docstring(node) or ""

        self._class_registry[key] = ClassMetrics(
            name=node.name,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            methods=methods,
            complexity=complexity,
            has_docstring=bool(docstring),
            bases=bases,
            decorators=decorators,
        )

    # ── Complexity & Metrics ──────────────────────────────────────────────

    def _compute_complexity(self, node: ast.AST) -> int:
        """Compute McCabe cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
                if child.ifs:
                    complexity += len(child.ifs)

        return complexity

    def _compute_nest_depth(self, node: ast.AST) -> int:
        """Compute maximum nesting depth in a function."""
        max_depth = 0

        def _walk(n, depth):
            nonlocal max_depth
            if depth > max_depth:
                max_depth = depth
            for child in ast.iter_child_nodes(n):
                if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)):
                    _walk(child, depth + 1)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue  # Don't recurse into nested definitions
                else:
                    _walk(child, depth)

        _walk(node, 0)
        return max_depth

    def _extract_calls(self, node: ast.AST) -> List[str]:
        """Extract all function calls from a node."""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return calls

    # ── Bug Pattern Detection ─────────────────────────────────────────────

    def _detect_bug_patterns(self, file_path: str, tree: ast.AST, content: str) -> List[CodeIssue]:
        """Detect common bug patterns and anti-patterns."""
        issues = []
        lines = content.split("\n")

        for node in ast.walk(tree):
            # Mutable default arguments
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + (node.args.kw_defaults or []):
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line=node.lineno,
                            severity=Severity.HIGH,
                            category=IssueCategory.BUG_RISK,
                            title="Mutable default argument",
                            description=(
                                f"Function '{node.name}' has a mutable default argument. "
                                f"This is a common Python bug — the default is shared across all calls."
                            ),
                            suggestion="Use None as default and initialize inside the function body",
                            auto_fixable=True,
                            confidence=0.95,
                        ))

                # Check for bare except
                for child in ast.walk(node):
                    if isinstance(child, ast.ExceptHandler) and child.type is None:
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line=child.lineno,
                            severity=Severity.MEDIUM,
                            category=IssueCategory.BUG_RISK,
                            title="Bare except clause",
                            description="Catching all exceptions with bare 'except:' can hide bugs.",
                            suggestion="Use 'except Exception:' or catch specific exceptions",
                            auto_fixable=True,
                            confidence=0.9,
                        ))

                # Check for return in finally (swallows exceptions)
                for child in ast.walk(node):
                    if isinstance(child, ast.Try):
                        for finalizer in child.finalbody:
                            if isinstance(finalizer, ast.Return):
                                issues.append(CodeIssue(
                                    file_path=file_path,
                                    line=finalizer.lineno,
                                    severity=Severity.HIGH,
                                    category=IssueCategory.BUG_RISK,
                                    title="Return in finally block",
                                    description=(
                                        "Returning from a finally block silently swallows any exception "
                                        "raised in the try or except blocks."
                                    ),
                                    suggestion="Move the return logic outside the finally block",
                                    confidence=0.85,
                                ))

            # String concatenation in loop
            if isinstance(node, ast.AugAssign):
                if isinstance(node.op, ast.Add) and isinstance(node.target, ast.Name):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        # Check if inside a loop
                        for parent in ast.walk(tree):
                            if isinstance(parent, (ast.For, ast.AsyncFor, ast.While)):
                                issues.append(CodeIssue(
                                    file_path=file_path,
                                    line=node.lineno,
                                    severity=Severity.LOW,
                                    category=IssueCategory.PERFORMANCE,
                                    title="String concatenation in loop",
                                    description="Using += for string concatenation in a loop creates new strings each iteration.",
                                    suggestion="Use a list and ''.join() instead",
                                    auto_fixable=True,
                                    confidence=0.7,
                                ))
                                break

            # Comparison with None using ==
            if isinstance(node, ast.Compare):
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant) and comparator.value is None:
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line=node.lineno,
                            severity=Severity.LOW,
                            category=IssueCategory.STYLE,
                            title="Comparison with None using ==",
                            description="PEP 8 recommends using 'is' or 'is not' when comparing to None.",
                            suggestion="Replace == None with 'is None'",
                            auto_fixable=True,
                            confidence=0.95,
                        ))

            # Unused variable (simple heuristic: assignment but never used)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("_"):
                        continue  # _ prefixed is intentionally unused
                    if isinstance(target, ast.Name):
                        # Count uses of this name in the same scope
                        name = target.id
                        uses = sum(
                            1 for child in ast.walk(tree)
                            if isinstance(child, ast.Name) and child.id == name
                        )
                        if uses <= 1:  # Only the assignment itself
                            issues.append(CodeIssue(
                                file_path=file_path,
                                line=node.lineno,
                                severity=Severity.INFO,
                                category=IssueCategory.DEAD_CODE,
                                title=f"Potentially unused variable '{name}'",
                                description=f"Variable '{name}' is assigned but never used in this scope.",
                                suggestion="Remove the unused variable or prefix with '_'",
                                confidence=0.5,  # Low confidence — could be used dynamically
                            ))

        return issues

    # ── Single File Analysis ──────────────────────────────────────────────

    async def analyze_file(self, file_path: str) -> FileAnalysis:
        """Perform deep analysis on a single file."""
        key = str(file_path)
        if key not in self._file_cache:
            raise FileNotFoundError(f"File not found in project: {file_path}")

        content, tree = self._file_cache[key]
        lines = content.split("\n")
        total_lines = len(lines)

        # Count code, comment, blank lines
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        in_multiline_string = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_multiline_string:
                    in_multiline_string = False
                else:
                    in_multiline_string = True
                comment_lines += 1
            elif stripped.startswith("#"):
                comment_lines += 1
            elif not stripped:
                blank_lines += 1
            else:
                code_lines += 1

        # Count functions and classes
        functions = []
        classes = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)
            elif isinstance(node, ast.ClassDef):
                classes.append(node)

        # Complexity analysis
        func_complexities = []
        for func in functions:
            c = self._compute_complexity(func)
            func_complexities.append(c)
        class_complexities = []
        for cls in classes:
            total_c = sum(
                self._compute_complexity(child)
                for child in ast.iter_child_nodes(cls)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            class_complexities.append(total_c)

        avg_func_complexity = sum(func_complexities) / len(func_complexities) if func_complexities else 0
        max_complexity = max(func_complexities + class_complexities) if (func_complexities or class_complexities) else 0

        # Docstring coverage
        docable_items = functions + classes
        documented = sum(1 for item in docable_items if ast.get_docstring(item))
        doc_coverage = documented / len(docable_items) if docable_items else 1.0

        # Detect issues
        issues = self._detect_bug_patterns(key, tree, content)
        issues.extend(self._detect_complexity_issues(key, functions, classes))
        issues.extend(self._detect_style_issues(key, tree, content, lines))

        # Calculate overall score
        score = self._calculate_file_score(
            total_lines=total_lines,
            avg_complexity=avg_func_complexity,
            max_complexity=max_complexity,
            doc_coverage=doc_coverage,
            issue_count=len(issues),
        )

        grade = self._score_to_grade(score)

        return FileAnalysis(
            file_path=key,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            functions=len(functions),
            classes=len(classes),
            avg_function_complexity=round(avg_func_complexity, 1),
            max_function_complexity=max_complexity,
            docstring_coverage=round(doc_coverage, 2),
            issues=issues,
            overall_score=round(score, 3),
            quality_grade=grade,
        )

    def _detect_complexity_issues(self, file_path: str, functions: list, classes: list) -> List[CodeIssue]:
        """Detect complexity-related issues."""
        issues = []

        for func in functions:
            c = self._compute_complexity(func)
            if c > MAX_FUNCTION_COMPLEXITY:
                issues.append(CodeIssue(
                    file_path=file_path,
                    line=func.lineno,
                    severity=Severity.HIGH if c > MAX_FUNCTION_COMPLEXITY * 2 else Severity.MEDIUM,
                    category=IssueCategory.COMPLEXITY,
                    title=f"High complexity function: {func.name} (CC={c})",
                    description=(
                        f"Function '{func.name}' has cyclomatic complexity {c}, "
                        f"exceeding the threshold of {MAX_FUNCTION_COMPLEXITY}. "
                        f"High complexity makes code harder to test and maintain."
                    ),
                    suggestion="Break into smaller functions or simplify control flow",
                    confidence=0.9,
                ))

            lines = (func.end_lineno or func.lineno) - func.lineno + 1
            if lines > MAX_FUNCTION_LINES:
                issues.append(CodeIssue(
                    file_path=file_path,
                    line=func.lineno,
                    severity=Severity.MEDIUM,
                    category=IssueCategory.COMPLEXITY,
                    title=f"Long function: {func.name} ({lines} lines)",
                    description=f"Function '{func.name}' is {lines} lines long (limit: {MAX_FUNCTION_LINES}).",
                    suggestion="Split into smaller, focused functions",
                    confidence=0.85,
                ))

        for cls in classes:
            total_c = sum(
                self._compute_complexity(child)
                for child in ast.iter_child_nodes(cls)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            if total_c > MAX_CLASS_COMPLEXITY:
                issues.append(CodeIssue(
                    file_path=file_path,
                    line=cls.lineno,
                    severity=Severity.HIGH if total_c > MAX_CLASS_COMPLEXITY * 2 else Severity.MEDIUM,
                    category=IssueCategory.COMPLEXITY,
                    title=f"High complexity class: {cls.name} (CC={total_c})",
                    description=f"Class '{cls.name}' has total complexity {total_c}.",
                    suggestion="Consider splitting into multiple classes or using composition",
                    confidence=0.85,
                ))

        return issues

    def _detect_style_issues(self, file_path: str, tree: ast.AST, content: str, lines: list) -> List[CodeIssue]:
        """Detect style and consistency issues."""
        issues = []

        # Check file length
        if len(lines) > MAX_FILE_LINES:
            issues.append(CodeIssue(
                file_path=file_path,
                line=1,
                severity=Severity.MEDIUM,
                category=IssueCategory.COMPLEXITY,
                title=f"Long file ({len(lines)} lines)",
                description=f"File has {len(lines)} lines (limit: {MAX_FILE_LINES}).",
                suggestion="Consider splitting into multiple modules",
                confidence=0.8,
            ))

        # Check module docstring
        if not ast.get_docstring(tree):
            issues.append(CodeIssue(
                file_path=file_path,
                line=1,
                severity=Severity.INFO,
                category=IssueCategory.DOCUMENTATION,
                title="Missing module docstring",
                description="File lacks a module-level docstring.",
                suggestion='Add a docstring at the top: """Module description."""',
                auto_fixable=True,
                confidence=0.7,
            ))

        # Check for TODO/FIXME/HACK comments
        for i, line in enumerate(lines, 1):
            stripped = line.strip().upper()
            if any(marker in stripped for marker in ("TODO", "FIXME", "HACK", "XXX", "BUG")):
                marker = next(m for m in ["TODO", "FIXME", "HACK", "XXX", "BUG"] if m in stripped)
                severity = Severity.MEDIUM if marker in ("FIXME", "BUG", "HACK") else Severity.INFO
                issues.append(CodeIssue(
                    file_path=file_path,
                    line=i,
                    severity=severity,
                    category=IssueCategory.STYLE,
                    title=f"{marker} comment found",
                    description=f"Line {i} contains a '{marker}' comment",
                    suggestion="Resolve the TODO/FIXME or create an issue to track it",
                    confidence=0.95,
                ))

        return issues

    def _calculate_file_score(
        self, total_lines: int, avg_complexity: float, max_complexity: int,
        doc_coverage: float, issue_count: int,
    ) -> float:
        """Calculate an overall quality score for a file (0.0 to 1.0)."""
        # Complexity score (0-1, lower complexity is better)
        complexity_score = max(0.0, 1.0 - (avg_complexity / MAX_FUNCTION_COMPLEXITY) * 0.5)
        if max_complexity > MAX_FUNCTION_COMPLEXITY:
            complexity_score *= 0.7

        # Documentation score
        doc_score = doc_coverage

        # Size penalty
        size_score = max(0.5, 1.0 - (total_lines / MAX_FILE_LINES) * 0.3)

        # Issue penalty
        issue_penalty = min(0.3, issue_count * 0.03)

        raw = (complexity_score * 0.3 + doc_score * 0.3 + size_score * 0.2) - issue_penalty
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.65:
            return "C"
        elif score >= 0.5:
            return "D"
        return "F"

    # ── Full Project Analysis ─────────────────────────────────────────────

    async def analyze_project(self) -> ProjectAnalysis:
        """Analyze the entire project and produce a comprehensive report."""
        file_analyses: Dict[str, FileAnalysis] = {}
        all_issues: List[CodeIssue] = []

        # Analyze each file
        for file_path in sorted(self._file_cache.keys()):
            analysis = await self.analyze_file(file_path)
            file_analyses[file_path] = analysis
            all_issues.extend(analysis.issues)

        # Aggregate metrics
        total_files = len(file_analyses)
        total_lines = sum(fa.total_lines for fa in file_analyses.values())
        total_functions = sum(fa.functions for fa in file_analyses.values())
        total_classes = sum(fa.classes for fa in file_analyses.values())

        all_complexities = [
            fa.avg_function_complexity
            for fa in file_analyses.values()
            if fa.functions > 0
        ]
        avg_complexity = sum(all_complexities) / len(all_complexities) if all_complexities else 0
        max_complexity = max(
            (fa.max_function_complexity for fa in file_analyses.values()), default=0
        )

        all_doc_coverages = [
            fa.docstring_coverage for fa in file_analyses.values() if fa.functions + fa.classes > 0
        ]
        avg_doc_coverage = sum(all_doc_coverages) / len(all_doc_coverages) if all_doc_coverages else 1.0

        # Issue breakdown
        issues_by_severity = Counter(i.severity.value for i in all_issues)
        issues_by_category = Counter(i.category.value for i in all_issues)

        # Top issues (by severity, then confidence)
        severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3, Severity.INFO: 4}
        top_issues = sorted(all_issues, key=lambda i: (severity_order.get(i.severity, 5), -i.confidence))[:20]

        # Overall score (weighted average of file scores)
        overall_score = sum(fa.overall_score for fa in file_analyses.values()) / total_files if total_files else 0

        # Architecture health
        arch_health = self._compute_architecture_health(file_analyses)

        # Generate improvement suggestions
        suggestions = self._generate_suggestions(
            avg_complexity, avg_doc_coverage, issues_by_category,
            total_lines, total_files, arch_health,
        )

        return ProjectAnalysis(
            total_files=total_files,
            total_lines=total_lines,
            total_functions=total_functions,
            total_classes=total_classes,
            avg_complexity=round(avg_complexity, 1),
            max_complexity=max_complexity,
            avg_docstring_coverage=round(avg_doc_coverage, 2),
            issues_by_severity=dict(issues_by_severity),
            issues_by_category=dict(issues_by_category),
            top_issues=top_issues,
            file_analyses=file_analyses,
            overall_score=round(overall_score, 3),
            quality_grade=self._score_to_grade(overall_score),
            architecture_health=arch_health,
            improvement_suggestions=suggestions,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _compute_architecture_health(self, file_analyses: Dict[str, FileAnalysis]) -> Dict[str, Any]:
        """Compute architecture health metrics."""
        # Dependency graph analysis
        circular_deps = self._detect_circular_imports()

        # Coupling analysis
        avg_imports = 0
        import_counts = [len(deps) for deps in self._import_graph.values()]
        avg_imports = sum(import_counts) / len(import_counts) if import_counts else 0

        # File sizes
        file_sizes = {fp: fa.total_lines for fp, fa in file_analyses.items()}
        large_files = [fp for fp, size in file_sizes.items() if size > MAX_FILE_LINES]

        # Module cohesion (files that import each other)
        tightly_coupled = self._detect_tight_coupling()

        return {
            "circular_imports": circular_deps,
            "avg_imports_per_file": round(avg_imports, 1),
            "large_files": large_files,
            "tightly_coupled_modules": tightly_coupled,
            "total_modules": len(file_analyses),
            "import_graph_edges": sum(len(deps) for deps in self._import_graph.values()),
        }

    def _detect_circular_imports(self) -> List[List[str]]:
        """Detect circular import chains."""
        # Convert to absolute module paths
        abs_imports = {}
        for file_path, deps in self._import_graph.items():
            abs_deps = set()
            for dep in deps:
                # Try to resolve dep to a file path
                for fp in self._file_cache:
                    if fp.replace("/", ".").startswith(dep) or dep in fp.replace("/", "."):
                        abs_deps.add(fp)
                        break
            abs_imports[file_path] = abs_deps

        # DFS for cycles
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def _dfs(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in abs_imports.get(node, set()):
                if neighbor not in visited:
                    _dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    if cycle not in cycles:
                        cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for node in list(abs_imports.keys()):
            if node not in visited:
                _dfs(node)

        return cycles

    def _detect_tight_coupling(self) -> List[Dict[str, Any]]:
        """Detect pairs of modules that import each other."""
        coupled = []
        checked = set()

        for file_path, deps in self._import_graph.items():
            for dep in deps:
                pair = tuple(sorted([file_path, dep]))
                if pair in checked:
                    continue
                checked.add(pair)

                # Check if dep also imports file_path
                dep_deps = self._import_graph.get(dep, set())
                if file_path in dep_deps or any(file_path in str(d) for d in dep_deps):
                    coupled.append({
                        "module_a": file_path,
                        "module_b": dep,
                        "type": "bidirectional",
                    })

        return coupled

    def _generate_suggestions(
        self, avg_complexity: float, doc_coverage: float,
        issues_by_category: Counter, total_lines: int, total_files: int,
        arch_health: Dict[str, Any],
    ) -> List[str]:
        """Generate prioritized improvement suggestions."""
        suggestions = []

        if avg_complexity > 10:
            suggestions.append(
                f"Average function complexity is {avg_complexity:.1f}. "
                f"Refactor complex functions to improve maintainability."
            )

        if doc_coverage < 0.7:
            suggestions.append(
                f"Docstring coverage is {doc_coverage:.0%}. "
                f"Add docstrings to undocumented functions and classes."
            )

        if arch_health.get("circular_imports"):
            suggestions.append(
                f"Found {len(arch_health['circular_imports'])} circular import chain(s). "
                f"Restructure imports to break cycles."
            )

        if arch_health.get("large_files"):
            suggestions.append(
                f"{len(arch_health['large_files'])} file(s) exceed {MAX_FILE_LINES} lines. "
                f"Consider splitting large modules."
            )

        if issues_by_category.get(IssueCategory.BUG_RISK.value, 0) > 0:
            suggestions.append(
                f"Found {issues_by_category[IssueCategory.BUG_RISK.value]} potential bug(s). "
                f"Review and fix these issues first."
            )

        if arch_health.get("tightly_coupled_modules"):
            suggestions.append(
                f"Found {len(arch_health['tightly_coupled_modules'])} tightly coupled module pair(s). "
                f"Consider introducing an interface or event system to reduce coupling."
            )

        return suggestions

    # ── Snapshot & Comparison ─────────────────────────────────────────────

    async def save_snapshot(self, analysis: ProjectAnalysis) -> str:
        """Save a project analysis snapshot for later comparison."""
        import uuid
        snapshot_id = uuid.uuid4().hex[:12]

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO snapshots "
                "(snapshot_id, timestamp, overall_score, total_issues, total_files, "
                "total_lines, avg_complexity, doc_coverage, file_scores, issues_by_cat) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    analysis.timestamp,
                    analysis.overall_score,
                    len(analysis.top_issues),
                    analysis.total_files,
                    analysis.total_lines,
                    analysis.avg_complexity,
                    analysis.avg_docstring_coverage,
                    json.dumps({fp: fa.overall_score for fp, fa in analysis.file_analyses.items()}),
                    json.dumps(dict(analysis.issues_by_category)),
                ),
            )
            conn.commit()

        await self._run_sync(_do)
        return snapshot_id

    async def compare_snapshots(self, snapshot_id_a: str, snapshot_id_b: str) -> Dict[str, Any]:
        """Compare two snapshots and report improvements/regressions."""
        def _do() -> Tuple[Optional[dict], Optional[dict]]:
            conn = self._ensure_conn()
            a = conn.execute("SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id_a,)).fetchone()
            b = conn.execute("SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id_b,)).fetchone()
            return (dict(a) if a else None, dict(b) if b else None)

        snap_a, snap_b = await self._run_sync(_do)
        if snap_a is None or snap_b is None:
            return {"error": "One or both snapshots not found"}

        score_delta = snap_b["overall_score"] - snap_a["overall_score"]
        issue_delta = snap_b["total_issues"] - snap_a["total_issues"]
        complexity_delta = snap_b["avg_complexity"] - snap_a["avg_complexity"]
        doc_delta = snap_b["doc_coverage"] - snap_a["doc_coverage"]

        # Per-file comparison
        scores_a = json.loads(snap_a.get("file_scores", "{}"))
        scores_b = json.loads(snap_b.get("file_scores", "{}"))
        improved_files = []
        regressed_files = []
        for fp in set(list(scores_a.keys()) + list(scores_b.keys())):
            sa = scores_a.get(fp, 0)
            sb = scores_b.get(fp, 0)
            delta = sb - sa
            if delta > 0.05:
                improved_files.append({"file": fp, "before": round(sa, 3), "after": round(sb, 3), "delta": round(delta, 3)})
            elif delta < -0.05:
                regressed_files.append({"file": fp, "before": round(sa, 3), "after": round(sb, 3), "delta": round(delta, 3)})

        improved_files.sort(key=lambda x: -x["delta"])
        regressed_files.sort(key=lambda x: x["delta"])

        return {
            "snapshot_a": snapshot_id_a,
            "snapshot_b": snapshot_id_b,
            "timestamp_a": snap_a["timestamp"],
            "timestamp_b": snap_b["timestamp"],
            "score_delta": round(score_delta, 3),
            "issue_delta": issue_delta,
            "complexity_delta": round(complexity_delta, 1),
            "doc_coverage_delta": round(doc_delta, 2),
            "improved_files": improved_files,
            "regressed_files": regressed_files,
            "verdict": "improved" if score_delta > 0.02 else ("regressed" if score_delta < -0.02 else "stable"),
        }

    async def get_snapshots(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of saved snapshots."""
        def _do() -> List[Dict[str, Any]]:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

        return await self._run_sync(_do)

    # ── Focused Analysis Methods ──────────────────────────────────────────

    async def find_bug_patterns(self) -> List[CodeIssue]:
        """Find all potential bugs across the project."""
        all_issues = []
        for file_path in self._file_cache:
            content, tree = self._file_cache[file_path]
            issues = self._detect_bug_patterns(file_path, tree, content)
            bug_issues = [i for i in issues if i.category == IssueCategory.BUG_RISK]
            all_issues.extend(bug_issues)
        return sorted(all_issues, key=lambda i: (i.severity.value, -i.confidence))

    async def find_dead_code(self) -> List[CodeIssue]:
        """Find potentially dead code (unused functions, variables)."""
        all_issues = []
        for file_path in self._file_cache:
            content, tree = self._file_cache[file_path]
            issues = self._detect_bug_patterns(file_path, tree, content)
            dead_issues = [i for i in issues if i.category == IssueCategory.DEAD_CODE]
            all_issues.extend(dead_issues)
        return all_issues

    async def get_function_metrics(self) -> List[FunctionMetrics]:
        """Get metrics for all functions in the project, sorted by complexity."""
        return sorted(self._function_registry.values(), key=lambda f: -f.complexity)

    async def get_class_metrics(self) -> List[ClassMetrics]:
        """Get metrics for all classes in the project, sorted by complexity."""
        return sorted(self._class_registry.values(), key=lambda c: -c.complexity)

    async def generate_report(self, analysis: ProjectAnalysis = None) -> str:
        """Generate a human-readable analysis report."""
        if analysis is None:
            analysis = await self.analyze_project()

        lines = [
            "# Claude Clone — Self-Analysis Report",
            "",
            f"**Generated:** {analysis.timestamp}",
            f"**Overall Grade:** {analysis.quality_grade} (score: {analysis.overall_score:.2f})",
            f"**Files:** {analysis.total_files} | **Lines:** {analysis.total_lines:,} | **Functions:** {analysis.total_functions} | **Classes:** {analysis.total_classes}",
            f"**Avg Complexity:** {analysis.avg_complexity:.1f} | **Max:** {analysis.max_complexity}",
            f"**Doc Coverage:** {analysis.avg_docstring_coverage:.0%}",
            "",
            "## Issues Summary",
        ]

        for sev in ("critical", "high", "medium", "low", "info"):
            count = analysis.issues_by_severity.get(sev, 0)
            if count > 0:
                lines.append(f"- **{sev.upper()}:** {count}")

        lines.append("")

        if analysis.top_issues:
            lines.append("## Top Issues")
            for issue in analysis.top_issues[:15]:
                lines.append(
                    f"- [{issue.severity.value.upper()}] {issue.file_path}:{issue.line} — "
                    f"{issue.title}"
                )
                if issue.suggestion:
                    lines.append(f"  → {issue.suggestion}")
            lines.append("")

        if analysis.improvement_suggestions:
            lines.append("## Improvement Suggestions")
            for suggestion in analysis.improvement_suggestions:
                lines.append(f"- {suggestion}")
            lines.append("")

        lines.append("## Per-File Scores")
        sorted_files = sorted(analysis.file_analyses.items(), key=lambda x: x[1].overall_score)
        for fp, fa in sorted_files:
            bar = "\u2588" * int(fa.overall_score * 20)
            issues_count = len(fa.issues)
            lines.append(
                f"- [{fa.quality_grade}] {fp}: {fa.overall_score:.2f} "
                f"({fa.functions}f, {fa.classes}c, {issues_count} issues) {bar}"
            )

        return "\n".join(lines)
