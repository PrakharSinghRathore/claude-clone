"""
Deep Project Analyzer Module for Claude Code Clone.

Provides comprehensive codebase analysis including complexity metrics,
dependency graphs, quality scoring, dead code detection, and more.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class FileAnalysis:
    """Detailed analysis results for a single source file."""
    path: str
    language: str
    lines: int = 0
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    complexity: dict[str, int] = field(default_factory=dict)
    imports: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class DependencyNode:
    """A node in the project dependency graph."""
    name: str
    imports: list[str] = field(default_factory=list)
    imported_by: list[str] = field(default_factory=list)
    is_circular: bool = False


@dataclass
class AnalysisResult:
    """Top-level analysis result for an entire project."""
    project_name: str = ""
    total_files: int = 0
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
    languages: dict[str, dict] = field(default_factory=dict)
    complexity_score: float = 0.0
    quality_score: float = 0.0
    doc_coverage: dict[str, float] = field(default_factory=dict)
    dependencies: list[DependencyNode] = field(default_factory=list)
    circular_deps: list[list[str]] = field(default_factory=list)
    dead_code: list[dict] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        data = asdict(self)
        return data


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_IGNORE_PATTERNS: list[str] = [
    ".git",
    ".gitignore",
    ".hg",
    ".svn",
    ".DS_Store",
    ".env",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    "*.egg-info",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "htmlcov",
    ".coverage",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.dylib",
    "target",
    ".gradle",
    ".idea",
    ".vscode",
    "vendor",
    "Pods",
    ".terraform",
    "*.min.js",
    "*.min.css",
    "*.bundle.js",
    "*.map",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Pipfile.lock",
    "poetry.lock",
    ".next",
    ".nuxt",
    ".output",
    ".svelte-kit",
    "coverage",
    ".nyc_output",
    "*.snap",
    "*.log",
]

LANGUAGE_MAP: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript (JSX)",
    ".ts": "TypeScript",
    ".tsx": "TypeScript (TSX)",
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".scala": "Scala",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".c": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".swift": "Swift",
    ".m": "Objective-C",
    ".mm": "Objective-C++",
    ".r": "R",
    ".R": "R",
    ".jl": "Julia",
    ".lua": "Lua",
    ".pl": "Perl",
    ".pm": "Perl",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".fish": "Shell",
    ".ps1": "PowerShell",
    ".bat": "Batch",
    ".cmd": "Batch",
    ".sql": "SQL",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sass": "SCSS",
    ".less": "LESS",
    ".vue": "Vue",
    ".svelte": "Svelte",
    ".dart": "Dart",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".erl": "Erlang",
    ".hrl": "Erlang",
    ".clj": "Clojure",
    ".cljs": "ClojureScript",
    ".hs": "Haskell",
    ".lhs": "Haskell",
    ".ml": "OCaml",
    ".mli": "OCaml",
    ".fs": "F#",
    ".fsx": "F#",
    ".zig": "Zig",
    ".nim": "Nim",
    ".v": "V",
    ".tf": "HCL",
    ".proto": "Protocol Buffers",
    ".graphql": "GraphQL",
    ".gql": "GraphQL",
    ".dockerfile": "Dockerfile",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".toml": "TOML",
    ".json": "JSON",
    ".xml": "XML",
    ".md": "Markdown",
    ".rst": "reStructuredText",
    ".txt": "Plain Text",
    ".ini": "INI",
    ".cfg": "Config",
    ".conf": "Config",
}

# Regex patterns for heuristic complexity in non-Python languages.
# Each pattern adds 1 to cyclomatic complexity.
HEURISTIC_COMPLEXITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bif\b"),
    re.compile(r"\belse\s+if\b|\belif\b"),
    re.compile(r"\belse\b"),
    re.compile(r"\bfor\b"),
    re.compile(r"\bwhile\b"),
    re.compile(r"\bcase\b"),
    re.compile(r"\bcatch\b"),
    re.compile(r"\bexcept\b"),
    re.compile(r"\?\?"),                  # nullish coalescing
    re.compile(r"\?[^.]"),                # ternary
    re.compile(r"&&"),
    re.compile(r"\|\|"),
    re.compile(r"\band\b"),
    re.compile(r"\bor\b"),
    re.compile(r"\btry\b"),
    re.compile(r"\bfinally\b"),
    re.compile(r"\bswitch\b"),
    re.compile(r"\bdefault\b"),
    re.compile(r"\bawait\b"),
    re.compile(r"\basync\b"),
]

# Patterns that *subtract* from heuristic matches (avoid double-counting).
HEURISTIC_EXCLUSION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"#\s*if\b"),               # preprocessor comment
    re.compile(r"//\s*if\b"),              # JS comment
    re.compile(r"\/\*.*?\*\/", re.DOTALL), # block comment
    re.compile(r"\"[^\"]*\""),             # string literal (simple)
    re.compile(r"'[^']*'"),                # string literal (simple)
    re.compile(r"f\"[^\"]*\""),            # f-string
    re.compile(r"f'[^']*'"),
    re.compile(r"r\"[^\"]*\""),            # raw string
    re.compile(r"r'[^']*'"),
]

SNAPSHOTS_DIR = ".claude_snapshots"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _should_ignore(path: Path, ignore_patterns: list[str]) -> bool:
    """Return True if *path* matches any of the ignore patterns."""
    name = path.name
    parts = path.parts
    for pattern in ignore_patterns:
        if pattern.startswith("*"):
            suffix = pattern[1:]
            if name.endswith(suffix):
                return True
        elif pattern in parts:
            return True
        elif name == pattern:
            return True
    return False


def _detect_language(filepath: str) -> str:
    """Detect language from file extension."""
    name = Path(filepath).name.lower()
    if name in ("dockerfile", "makefile", "jenkinsfile", "vagrantfile", "gemfile", "rakefile"):
        return name.capitalize()
    ext = Path(filepath).suffix.lower()
    return LANGUAGE_MAP.get(ext, "Unknown")


def _strip_comments_and_strings(source: str, language: str) -> str:
    """Remove comments and string literals to avoid false complexity counts."""
    # Block comments
    if language in ("Python",):
        source = re.sub(r"(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\')", '""', source)
        source = re.sub(r"#.*$", "", source, flags=re.MULTILINE)
    elif language in ("JavaScript", "TypeScript", "JavaScript (JSX)", "TypeScript (TSX)",
                       "Java", "Kotlin", "C#", "Go", "Dart", "Swift", "Rust"):
        source = re.sub(r"//.*$", "", source, flags=re.MULTILINE)
        source = re.sub(r"/\*[\s\S]*?\*/", "", source)
    elif language in ("C", "C++", "C/C++ Header", "C++ Header", "Objective-C"):
        source = re.sub(r"//.*$", "", source, flags=re.MULTILINE)
        source = re.sub(r"/\*[\s\S]*?\*/", "", source)
    elif language in ("Ruby", "PHP", "Perl"):
        source = re.sub(r"#.*$", "", source, flags=re.MULTILINE)
    elif language in ("Shell", "PowerShell", "Batch"):
        source = re.sub(r"#.*$", "", source, flags=re.MULTILINE)
    # String literals (simple – single/double quoted)
    source = re.sub(r"(f?r?b?\"\"\"[\s\S]*?\"\"\"|f?r?b?\'\'\'[\s\S]*?\'\'\')", '""', source)
    source = re.sub(r'f?r?b?"[^"\\]*(?:\\.[^"\\]*)*"', '""', source)
    source = re.sub(r"f?r?b?'[^'\\]*(?:\\.[^'\\]*)*'", "''", source)
    return source


def _cleaned_name(filepath: str) -> str:
    """Return a dotted module-style name from a file path."""
    p = Path(filepath)
    stem = p.stem
    if stem == "__init__":
        stem = p.parent.name
    return stem


# ---------------------------------------------------------------------------
# Python AST-based complexity & structure analysis
# ---------------------------------------------------------------------------

class _CyclomaticVisitor(ast.NodeVisitor):
    """AST visitor that computes per-function/class cyclomatic complexity."""

    def __init__(self, source: str) -> None:
        self._source = source
        self.function_complexity: dict[str, int] = {}
        self.class_names: list[str] = []
        self.function_names: list[str] = []
        self.imports: list[str] = []
        self.docstrings: dict[str, bool] = {}

    # -- helpers --

    def _complexity(self, node: ast.AST) -> int:
        """Compute decision-point count for *node*."""
        count = 1  # base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While)):
                count += 1
            elif isinstance(child, ast.For):
                count += 1
            elif isinstance(child, ast.ExceptHandler):
                count += 1
            elif isinstance(child, (ast.And, ast.Or)):
                count += 1
            elif isinstance(child, ast.comprehension):
                count += sum(1 for _ in child.ifs)
            elif isinstance(child, ast.IfExp):  # ternary
                count += 1
            elif isinstance(child, ast.With):
                count += 1
            elif isinstance(child, ast.Assert):
                count += 1
        return count

    def _get_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from a node, if any."""
        if (node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant, ast.Str))):
            return ast.get_docstring(node)
        return None

    # -- visitors --

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        name = node.name
        self.function_names.append(name)
        c = self._complexity(node)
        self.function_complexity[name] = c
        self.docstrings[name] = self._get_docstring(node) is not None
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_names.append(node.name)
        self.docstrings[node.name] = self._get_docstring(node) is not None
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}" if module else alias.name)
        self.generic_visit(node)


def _analyze_python(filepath: str, source: str) -> FileAnalysis:
    """Full AST-based analysis of a Python file."""
    fa = FileAnalysis(path=filepath, language="Python")
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        fa.issues.append(f"Syntax error: {exc}")
        fa.lines = source.count("\n") + 1
        return fa

    fa.lines = source.count("\n") + 1
    visitor = _CyclomaticVisitor(source)
    visitor.visit(tree)
    fa.functions = list(visitor.function_names)
    fa.classes = list(visitor.class_names)
    fa.complexity = visitor.function_complexity
    fa.imports = visitor.imports

    # Name-quality checks
    for func in fa.functions:
        if not re.match(r"^[a-z_][a-z0-9_]*$", func):
            fa.issues.append(f"Function '{func}' does not follow snake_case naming.")
    for cls in fa.classes:
        if not re.match(r"^[A-Z][a-zA-Z0-9]*$", cls):
            fa.issues.append(f"Class '{cls}' does not follow PascalCase naming.")
    return fa


# ---------------------------------------------------------------------------
# Heuristic complexity for non-Python files
# ---------------------------------------------------------------------------

def _analyze_generic(filepath: str, source: str, language: str) -> FileAnalysis:
    """Heuristic analysis for non-Python source files."""
    fa = FileAnalysis(path=filepath, language=language)
    fa.lines = source.count("\n") + 1

    cleaned = _strip_comments_and_strings(source, language)

    # -- Functions --
    func_patterns: list[re.Pattern[str]] = []
    if language in ("JavaScript", "TypeScript", "JavaScript (JSX)", "TypeScript (TSX)"):
        func_patterns = [
            re.compile(r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[a-zA-Z_]\w*)\s*=>)"),
            re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function"),
        ]
    elif language in ("Go",):
        func_patterns = [re.compile(r"func\s+(?:\([^)]*\)\s+)?(\w+)")]
    elif language in ("Rust",):
        func_patterns = [
            re.compile(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)"),
            re.compile(r"impl\s+(\w+)"),
        ]
    elif language in ("Java", "Kotlin", "C#"):
        func_patterns = [
            re.compile(r"(?:public|private|protected|static|abstract|final|override|\s)*\s+\w+\s+(\w+)\s*\("),
            re.compile(r"(?:class|interface|enum|object|data\s+class)\s+(\w+)"),
        ]
    elif language in ("C", "C++", "C/C++ Header", "C++ Header", "Objective-C"):
        func_patterns = [
            re.compile(r"(?:(?:static|inline|extern|virtual|const)\s+)*\w+(?:\s*\*+)?\s+(\w+)\s*\("),
            re.compile(r"(?:class|struct|union|enum)\s+(\w+)"),
        ]
    elif language in ("Ruby",):
        func_patterns = [re.compile(r"def\s+(\w+)")]
    elif language in ("PHP",):
        func_patterns = [re.compile(r"function\s+(\w+)")]
    elif language in ("Swift",):
        func_patterns = [re.compile(r"(?:func|class|struct|enum|protocol)\s+(\w+)")]
    elif language in ("Shell", "PowerShell", "Batch"):
        func_patterns = [re.compile(r"(?:function\s+)?(\w+)\s*\(\)")]
    elif language in ("Dart",):
        func_patterns = [re.compile(r"(?:(?:static|final)\s+)?(?:Future\s*<[^>]*>\s+)?\w+\s+(\w+)\s*\(")]

    names_seen: set[str] = set()
    for pat in func_patterns:
        for m in pat.finditer(cleaned):
            name = next((g for g in m.groups() if g is not None), None)
            if name and name not in names_seen:
                names_seen.add(name)
                if re.match(r"^[A-Z]", name) and language not in ("Java", "Kotlin", "C#", "Swift", "Dart"):
                    fa.classes.append(name)
                else:
                    fa.functions.append(name)

    # -- Imports --
    import_patterns: list[re.Pattern[str]] = []
    if language in ("JavaScript", "TypeScript", "JavaScript (JSX)", "TypeScript (TSX)"):
        import_patterns = [
            re.compile(r"import\s+.*?from\s+['\"](.+?)['\"]"),
            re.compile(r"(?:const|let|var)\s+\w+\s*=\s*require\(['\"](.+?)['\"]\)"),
        ]
    elif language in ("Java", "Kotlin"):
        import_patterns = [re.compile(r"import\s+(?:static\s+)?([\w.]+)")]
    elif language in ("C#",):
        import_patterns = [re.compile(r"using\s+([\w.]+)")]
    elif language in ("Go",):
        import_patterns = [re.compile(r"\"([\w./]+)")]
    elif language in ("Rust",):
        import_patterns = [
            re.compile(r"use\s+([\w:]+)"),
            re.compile(r"mod\s+(\w+)"),
        ]
    elif language in ("C", "C++", "C/C++ Header", "C++ Header", "Objective-C"):
        import_patterns = [re.compile(r"#include\s+[<\"]([\w./]+)[>\"]")]
    elif language in ("Ruby",):
        import_patterns = [
            re.compile(r"require\s+['\"]([\w./]+)['\"]"),
            re.compile(r"(?:gem|load)\s+['\"]([\w./]+)['\"]"),
        ]
    elif language in ("PHP",):
        import_patterns = [
            re.compile(r"(?:use|require|include)(_once)?\s+['\"]?([\w\\]+)['\"]?"),
        ]
    elif language in ("Swift",):
        import_patterns = [re.compile(r"import\s+(\w+)")]
    elif language in ("Dart",):
        import_patterns = [re.compile(r"import\s+['\"](?:package:)?([\w./]+)['\"]")]

    import_names: set[str] = set()
    for pat in import_patterns:
        for m in pat.finditer(cleaned):
            imp = m.group(1) if m.group(1) else m.group(2)
            if imp and imp not in import_names:
                import_names.add(imp)
                fa.imports.append(imp)

    # -- Per-function heuristic complexity --
    for func in fa.functions:
        pattern = re.compile(
            rf"(?:def|function|func|fn|fn\s)\s+{re.escape(func)}\s*\([^)]*\)"
            rf"(?:(?:(?!def |function |func |fn ).)*?)"
            r"(?=\n(?:def |function |func |fn |class |struct |enum |interface |impl )|\Z)",
            re.DOTALL,
        )
        body_match = pattern.search(cleaned)
        body = body_match.group(0) if body_match else ""
        c = 1
        for cp in HEURISTIC_COMPLEXITY_PATTERNS:
            c += len(cp.findall(body))
        # Subtract exclusions
        for ep in HEURISTIC_EXCLUSION_PATTERNS:
            c -= len(ep.findall(body))
        fa.complexity[func] = max(c, 1)

    return fa


# ---------------------------------------------------------------------------
# Main ProjectAnalyzer
# ---------------------------------------------------------------------------

class ProjectAnalyzer:
    """Deep analyser for source-code projects."""

    def __init__(self, project_path: str, ignore_patterns: list[str] | None = None) -> None:
        self.project_path = Path(project_path).resolve()
        self.ignore_patterns: list[str] = ignore_patterns if ignore_patterns is not None else list(DEFAULT_IGNORE_PATTERNS)
        self._file_analyses: dict[str, FileAnalysis] = {}
        self._source_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ core

    def _source_files(self) -> list[Path]:
        """Walk the project tree and return analysable source files."""
        files: list[Path] = []
        for root, _dirs, filenames in os.walk(self.project_path):
            root_path = Path(root)
            if _should_ignore(root_path, self.ignore_patterns):
                _dirs.clear()
                continue
            for fn in filenames:
                fp = root_path / fn
                if _should_ignore(fp, self.ignore_patterns):
                    continue
                if fp.suffix.lower() in LANGUAGE_MAP or fp.name.lower() in (
                    "dockerfile", "makefile", "jenkinsfile", "vagrantfile", "gemfile", "rakefile",
                ):
                    files.append(fp)
        return sorted(files)

    def _read_source(self, filepath: Path) -> str:
        if str(filepath) in self._source_cache:
            return self._source_cache[str(filepath)]
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        self._source_cache[str(filepath)] = text
        return text

    # -------------------------------------------------------------- analyse

    async def analyze(self) -> AnalysisResult:
        """Run full analysis of the project."""
        files = self._source_files()

        # Parallel file analysis
        tasks = [self.analyze_file(str(f)) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        self._file_analyses.clear()
        for fa in results:
            if isinstance(fa, Exception):
                continue
            self._file_analyses[fa.path] = fa

        dep_graph = await self.get_dependency_graph()
        circular = await self.detect_circular_deps()
        dead = await self.find_dead_code()
        doc_cov = await self.get_documentation_coverage()
        quality = await self.get_quality_score()
        langs = await self.get_language_breakdown()
        recs = await self.get_recommendations()

        total_lines = sum(fa.lines for fa in self._file_analyses.values())
        total_funcs = sum(len(fa.functions) for fa in self._file_analyses.values())
        total_classes = sum(len(fa.classes) for fa in self._file_analyses.values())

        # Weighted complexity score (average across functions)
        all_cx: list[int] = []
        for fa in self._file_analyses.values():
            all_cx.extend(fa.complexity.values())
        avg_cx = statistics.mean(all_cx) if all_cx else 0.0

        return AnalysisResult(
            project_name=self.project_path.name,
            total_files=len(self._file_analyses),
            total_lines=total_lines,
            total_functions=total_funcs,
            total_classes=total_classes,
            languages=langs,
            complexity_score=avg_cx,
            quality_score=quality,
            doc_coverage=doc_cov,
            dependencies=dep_graph,
            circular_deps=circular,
            dead_code=dead,
            recommendations=recs,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def analyze_file(self, filepath: str) -> FileAnalysis:
        """Analyse a single file and return a FileAnalysis."""
        fp = Path(filepath)
        if not fp.is_absolute():
            fp = self.project_path / fp
        fp = fp.resolve()

        source = self._read_source(fp)
        language = _detect_language(str(fp))

        if language == "Python":
            return _analyze_python(str(fp), source)
        return _analyze_generic(str(fp), source, language)

    # ---------------------------------------------------------- complexity

    async def get_complexity(self, filepath: str) -> dict[str, int]:
        """Return per-function cyclomatic complexity for a file."""
        if filepath not in self._file_analyses:
            await self.analyze_file(filepath)
        fa = self._file_analyses.get(filepath)
        if fa is None:
            return {}
        return dict(fa.complexity)

    # ------------------------------------------------------------- deps

    async def get_dependency_graph(self) -> list[DependencyNode]:
        """Build the inter-module dependency graph."""
        if not self._file_analyses:
            files = self._source_files()
            tasks = [self.analyze_file(str(f)) for f in files]
            await asyncio.gather(*tasks, return_exceptions=True)

        # Build import -> target mapping
        node_map: dict[str, DependencyNode] = {}

        # Seed nodes for every analysed file
        for path, fa in self._file_analyses.items():
            name = _cleaned_name(path)
            node_map[name] = DependencyNode(name=name, imports=[])

        for path, fa in self._file_analyses.items():
            from_name = _cleaned_name(path)
            for imp in fa.imports:
                top = imp.split(".")[0]
                if top in node_map and top != from_name:
                    node_map[from_name].imports.append(top)
                    if from_name not in node_map[top].imported_by:
                        node_map[top].imported_by.append(from_name)

        return list(node_map.values())

    # ------------------------------------------------- circular deps

    async def detect_circular_deps(self) -> list[list[str]]:
        """Find circular dependency chains using DFS."""
        graph = await self.get_dependency_graph()
        adj: dict[str, list[str]] = {n.name: list(n.imports) for n in graph}

        cycles: list[list[str]] = []
        visited: set[str] = set()
        stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            stack.add(node)
            path.append(node)
            for neighbour in adj.get(node, []):
                if neighbour in stack:
                    idx = path.index(neighbour)
                    cycle = path[idx:] + [neighbour]
                    canonical = min(cycle[:-1])  # normalise
                    rotated = cycle[cycle.index(canonical):-1] + cycle[:cycle.index(canonical)] + [canonical]
                    if not any(
                        set(c[:-1]) == set(rotated[:-1]) for c in cycles
                    ):
                        cycles.append(rotated)
                elif neighbour not in visited:
                    dfs(neighbour)
            path.pop()
            stack.discard(node)

        for node in sorted(adj):
            if node not in visited:
                dfs(node)

        return cycles

    # -------------------------------------------------------- dead code

    async def find_dead_code(self) -> list[dict]:
        """Detect potentially dead code: unused imports, unreachable functions."""
        findings: list[dict] = []

        for path, fa in self._file_analyses.items():
            source = self._read_source(Path(path))

            # -- Unused imports (Python) --
            if fa.language == "Python":
                for imp in fa.imports:
                    mod_name = imp.split(".")[0]
                    local_name = mod_name
                    # Check for `as` aliases in source
                    as_pat = re.compile(
                        rf"(?:from\s+{re.escape(imp)}\s+import\s+)|(?:import\s+{re.escape(imp)}(?:\s+as\s+(\w+))?)",
                    )
                    as_match = as_pat.search(source)
                    if as_match and as_match.group(1):
                        local_name = as_match.group(1)
                    if local_name == "*" or local_name == "__future__":
                        continue
                    # Count usages in source (skip the import line itself)
                    usage_pat = re.compile(rf"\b{re.escape(local_name)}\b")
                    lines_no_import = [
                        ln for ln in source.splitlines()
                        if not ln.strip().startswith(("import", "from"))
                    ]
                    usage_count = sum(len(usage_pat.findall(ln)) for ln in lines_no_import)
                    if usage_count == 0:
                        findings.append({
                            "type": "unused_import",
                            "file": path,
                            "name": imp,
                            "severity": "low",
                            "message": f"Import '{imp}' appears unused.",
                        })

            # -- Unused functions --
            for func_name in fa.functions:
                pattern = re.compile(rf"\b{re.escape(func_name)}\b")
                uses = 0
                for ln in source.splitlines():
                    stripped = ln.strip()
                    # Skip definition lines
                    if re.match(rf"(?:def|async\s+def|function|func|fn)\s+{re.escape(func_name)}", stripped):
                        continue
                    uses += len(pattern.findall(stripped))
                if uses == 0:
                    # Check if it's a dunder / override
                    if func_name.startswith("__") and func_name.endswith("__"):
                        continue
                    if func_name.startswith("_") and not func_name.startswith("__"):
                        findings.append({
                            "type": "unused_function",
                            "file": path,
                            "name": func_name,
                            "severity": "info",
                            "message": f"Private function '{func_name}' has no internal references.",
                        })
                    else:
                        findings.append({
                            "type": "unused_function",
                            "file": path,
                            "name": func_name,
                            "severity": "medium",
                            "message": f"Function '{func_name}' appears to be unused.",
                        })

        return findings

    # ------------------------------------------------------- doc coverage

    async def get_documentation_coverage(self) -> dict[str, float]:
        """Compute documentation coverage per language."""
        stats: dict[str, dict[str, int]] = {}

        for path, fa in self._file_analyses.items():
            lang = fa.language
            if lang not in stats:
                stats[lang] = {"documented": 0, "total": 0}
            total_items = len(fa.functions) + len(fa.classes)
            if fa.language == "Python":
                # Use AST-derived docstrings
                doc_count = sum(1 for v in self._get_py_docstring_map(fa).values() if v)
            else:
                # Heuristic: look for docblock comments before def/class
                doc_count = self._heuristic_doc_count(path, fa)
            stats[lang]["total"] += total_items
            stats[lang]["documented"] += doc_count

        result: dict[str, float] = {}
        for lang, s in stats.items():
            if s["total"] > 0:
                result[lang] = round((s["documented"] / s["total"]) * 100, 1)
            else:
                result[lang] = 100.0
        return result

    def _get_py_docstring_map(self, fa: FileAnalysis) -> dict[str, bool]:
        """Re-run AST visitor to extract docstring presence (cached)."""
        source = self._read_source(Path(fa.path))
        try:
            tree = ast.parse(source, filename=fa.path)
        except SyntaxError:
            return {k: False for k in fa.functions + fa.classes}
        visitor = _CyclomaticVisitor(source)
        visitor.visit(tree)
        return visitor.docstrings

    def _heuristic_doc_count(self, path: str, fa: FileAnalysis) -> int:
        """Count docblock-style comments for non-Python files."""
        source = self._read_source(Path(path))
        count = 0
        all_names = fa.functions + fa.classes
        for name in all_names:
            pat = re.compile(
                rf"(?:/\*\*[\s\S]*?\*/|///[^\n]*(?:\n///[^\n]*)*|#\s*-{3,}[\s\S]*?#\s*-{3,})"
                rf"[\s\n]*"
                rf"(?:def|function|func|fn|class|struct|enum|interface|impl)\s+{re.escape(name)}",
                re.MULTILINE,
            )
            if pat.search(source):
                count += 1
        return count

    # --------------------------------------------------------- quality

    async def get_quality_score(self) -> float:
        """Compute a composite quality score from 0 to 100."""
        if not self._file_analyses:
            files = self._source_files()
            await asyncio.gather(*[self.analyze_file(str(f)) for f in files], return_exceptions=True)

        # -- Factor 1: Complexity (max 25 pts) --
        all_cx: list[int] = []
        for fa in self._file_analyses.values():
            all_cx.extend(fa.complexity.values())
        if all_cx:
            avg_cx = statistics.mean(all_cx)
            # Score inversely: lower complexity = higher score
            # avg=1 → 25, avg=20+ → 0
            cx_score = max(0.0, 25.0 - (avg_cx - 1) * 1.25)
        else:
            cx_score = 25.0

        # -- Factor 2: Documentation coverage (max 25 pts) --
        doc_cov = await self.get_documentation_coverage()
        if doc_cov:
            avg_doc = statistics.mean(doc_cov.values())
            doc_score = (avg_doc / 100.0) * 25.0
        else:
            doc_score = 25.0

        # -- Factor 3: Naming convention adherence (max 20 pts) --
        total_issues = sum(len(fa.issues) for fa in self._file_analyses.values())
        total_symbols = sum(len(fa.functions) + len(fa.classes) for fa in self._file_analyses.values())
        if total_symbols > 0:
            naming_ratio = 1.0 - (total_issues / total_symbols)
            naming_score = max(0.0, naming_ratio) * 20.0
        else:
            naming_score = 20.0

        # -- Factor 4: Type hints (Python files, max 15 pts) --
        py_files = [fa for fa in self._file_analyses.values() if fa.language == "Python"]
        if py_files:
            hinted, total = 0, 0
            for fa in py_files:
                source = self._read_source(Path(fa.path))
                for func in fa.functions:
                    total += 1
                    pat = re.compile(
                        rf"def\s+{re.escape(func)}\s*\([^)]*\)\s*(?:->\s*\w+)?\s*:",
                    )
                    # Check for return annotation
                    ret_pat = re.compile(rf"def\s+{re.escape(func)}\s*\([^)]*\)\s*->")
                    # Check for type annotations in params
                    params_pat = re.compile(rf"def\s+{re.escape(func)}\s*\([^)]*:\s*\w")
                    if ret_pat.search(source) or params_pat.search(source):
                        hinted += 1
            type_score = (hinted / total * 15.0) if total > 0 else 15.0
        else:
            type_score = 15.0

        # -- Factor 5: Dead code (max 15 pts) --
        dead = await self.find_dead_code()
        dead_score = max(0.0, 15.0 - len(dead) * 0.5)

        total_score = cx_score + doc_score + naming_score + type_score + dead_score
        return round(min(100.0, max(0.0, total_score)), 1)

    # --------------------------------------------------------- languages

    async def get_language_breakdown(self) -> dict[str, dict]:
        """Return per-language statistics (file count, line count, percentage)."""
        if not self._file_analyses:
            files = self._source_files()
            await asyncio.gather(*[self.analyze_file(str(f)) for f in files], return_exceptions=True)

        lang_data: dict[str, dict[str, int]] = {}
        for fa in self._file_analyses.values():
            lang = fa.language
            if lang not in lang_data:
                lang_data[lang] = {"files": 0, "lines": 0}
            lang_data[lang]["files"] += 1
            lang_data[lang]["lines"] += fa.lines

        total_lines = sum(d["lines"] for d in lang_data.values())
        total_files = sum(d["files"] for d in lang_data.values())

        result: dict[str, dict] = {}
        for lang, d in sorted(lang_data.items(), key=lambda x: -x[1]["lines"]):
            pct_lines = round((d["lines"] / total_lines * 100), 1) if total_lines > 0 else 0.0
            pct_files = round((d["files"] / total_files * 100), 1) if total_files > 0 else 0.0
            result[lang] = {
                "files": d["files"],
                "lines": d["lines"],
                "percent_lines": pct_lines,
                "percent_files": pct_files,
            }
        return result

    # -------------------------------------------------------- tree view

    async def generate_tree(self, max_depth: int = 4) -> str:
        """Return an ASCII tree of the project structure."""
        lines: list[str] = [self.project_path.name + "/"]
        all_files = self._source_files()

        def _build_tree(
            directory: Path, prefix: str = "", depth: int = 0
        ) -> list[str]:
            if depth >= max_depth:
                return []
            entries: list[str] = []
            try:
                items = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError:
                return entries

            visible = [
                p for p in items
                if not _should_ignore(p, self.ignore_patterns)
            ]

            for idx, item in enumerate(visible):
                is_last = idx == len(visible) - 1
                connector = "└── " if is_last else "├── "
                child_prefix = "    " if is_last else "│   "

                if item.is_dir():
                    entries.append(f"{prefix}{connector}{item.name}/")
                    entries.extend(_build_tree(item, prefix + child_prefix, depth + 1))
                elif item in all_files:
                    lang = _detect_language(str(item))
                    entries.append(f"{prefix}{connector}{item.name}  ({lang})")
            return entries

        lines.extend(_build_tree(self.project_path))
        return "\n".join(lines)

    # -------------------------------------------------------- report

    async def generate_report(self, fmt: str = "markdown") -> str:
        """Generate a full analysis report in the specified format."""
        result = await self.analyze()

        if fmt == "json":
            return json.dumps(result.to_dict(), indent=2, default=str)

        # Markdown report
        sections: list[str] = []

        sections.append(f"# 📊 Project Analysis: {result.project_name}")
        sections.append(f"\n> Generated: {result.timestamp}\n")

        # -- Overview --
        sections.append("## Overview")
        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|-------|")
        sections.append(f"| Total Files | {result.total_files} |")
        sections.append(f"| Total Lines | {result.total_lines} |")
        sections.append(f"| Functions | {result.total_functions} |")
        sections.append(f"| Classes | {result.total_classes} |")
        sections.append(f"| Avg Complexity | {result.complexity_score:.1f} |")
        sections.append(f"| Quality Score | **{result.quality_score:.1f}/100** |")
        sections.append("")

        # -- Quality badge --
        if result.quality_score >= 80:
            badge = "🟢 Excellent"
        elif result.quality_score >= 60:
            badge = "🟡 Good"
        elif result.quality_score >= 40:
            badge = "🟠 Needs Improvement"
        else:
            badge = "🔴 Poor"
        sections.append(f"**Overall Quality:** {badge} ({result.quality_score:.1f}/100)\n")

        # -- Languages --
        sections.append("## Languages")
        sections.append(f"| Language | Files | Lines | % Lines |")
        sections.append(f"|----------|-------|-------|---------|")
        for lang, stats in result.languages.items():
            sections.append(
                f"| {lang} | {stats['files']} | {stats['lines']} | {stats['percent_lines']}% |"
            )
        sections.append("")

        # -- Complexity --
        sections.append("## Complexity (Top 10 Most Complex Functions)")
        all_funcs: list[tuple[str, str, int]] = []
        for path, fa in self._file_analyses.items():
            for fn, cx in fa.complexity.items():
                all_funcs.append((path, fn, cx))
        all_funcs.sort(key=lambda x: -x[2])
        if all_funcs:
            sections.append(f"| File | Function | Complexity |")
            sections.append(f"|------|----------|------------|")
            for path, fn, cx in all_funcs[:10]:
                short_path = path.replace(str(self.project_path) + "/", "")
                sections.append(f"| `{short_path}` | `{fn}` | {cx} |")
        else:
            sections.append("No functions found.")
        sections.append("")

        # -- Documentation --
        sections.append("## Documentation Coverage")
        sections.append(f"| Language | Coverage |")
        sections.append(f"|----------|----------|")
        for lang, pct in result.doc_coverage.items():
            bar_len = int(pct / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            sections.append(f"| {lang} | {pct}% {bar} |")
        sections.append("")

        # -- Circular Dependencies --
        sections.append("## Circular Dependencies")
        if result.circular_deps:
            for i, cycle in enumerate(result.circular_deps, 1):
                chain = " → ".join(cycle)
                sections.append(f"{i}. {chain}")
        else:
            sections.append("✅ No circular dependencies detected.")
        sections.append("")

        # -- Dead Code --
        sections.append("## Dead Code Detection")
        if result.dead_code:
            sections.append(f"| Type | File | Name | Severity |")
            sections.append(f"|------|------|------|----------|")
            for item in result.dead_code[:20]:
                short = item["file"].replace(str(self.project_path) + "/", "")
                sections.append(
                    f"| {item['type']} | `{short}` | `{item['name']}` | {item['severity']} |"
                )
            if len(result.dead_code) > 20:
                sections.append(f"| ... and {len(result.dead_code) - 20} more | | | |")
        else:
            sections.append("✅ No dead code detected.")
        sections.append("")

        # -- Dependency Graph (text) --
        sections.append("## Dependency Graph")
        if result.dependencies:
            for node in result.dependencies:
                imports_str = ", ".join(node.imports) if node.imports else "—"
                imported_by_str = ", ".join(node.imported_by) if node.imported_by else "—"
                circ = " ⚠️" if node.is_circular else ""
                sections.append(f"- **{node.name}**{circ}")
                sections.append(f"  - imports: {imports_str}")
                sections.append(f"  - imported by: {imported_by_str}")
        else:
            sections.append("No dependency information available.")
        sections.append("")

        # -- Recommendations --
        sections.append("## Recommendations")
        if result.recommendations:
            for i, rec in enumerate(result.recommendations, 1):
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec["priority"], "⚪")
                sections.append(f"{i}. {icon} **[{rec['priority'].upper()}]** {rec['message']}")
                if rec.get("details"):
                    sections.append(f"   - {rec['details']}")
        else:
            sections.append("No recommendations at this time.")
        sections.append("")

        # -- Tree --
        tree = await self.generate_tree(max_depth=3)
        sections.append("## Project Structure")
        sections.append("```")
        sections.append(tree)
        sections.append("```")

        return "\n".join(sections)

    # -------------------------------------------------------- snapshots

    async def save_snapshot(self) -> str:
        """Save the current analysis to disk. Returns the snapshot ID."""
        result = await self.analyze()
        snap_dir = self.project_path / SNAPSHOTS_DIR
        snap_dir.mkdir(parents=True, exist_ok=True)

        snap_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snap_path = snap_dir / f"{snap_id}.json"
        snap_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        return snap_id

    async def compare_snapshots(self, old_id: str, new_id: str) -> dict:
        """Compare two analysis snapshots and return a diff."""
        snap_dir = self.project_path / SNAPSHOTS_DIR
        old_path = snap_dir / f"{old_id}.json"
        new_path = snap_dir / f"{new_id}.json"

        if not old_path.exists():
            raise FileNotFoundError(f"Snapshot '{old_id}' not found.")
        if not new_path.exists():
            raise FileNotFoundError(f"Snapshot '{new_id}' not found.")

        old_data = json.loads(old_path.read_text(encoding="utf-8"))
        new_data = json.loads(new_path.read_text(encoding="utf-8"))

        diff: dict[str, Any] = {
            "old_snapshot": old_id,
            "new_snapshot": new_id,
            "metrics": {},
            "improvements": [],
            "regressions": [],
        }

        metric_keys = [
            ("total_files", "Total Files"),
            ("total_lines", "Total Lines"),
            ("total_functions", "Total Functions"),
            ("total_classes", "Total Classes"),
            ("complexity_score", "Avg Complexity"),
            ("quality_score", "Quality Score"),
        ]

        for key, label in metric_keys:
            old_val = old_data.get(key, 0)
            new_val = new_data.get(key, 0)
            delta = new_val - old_val
            diff["metrics"][label] = {
                "old": old_val,
                "new": new_val,
                "delta": delta,
            }
            if key == "quality_score":
                if delta > 0:
                    diff["improvements"].append(f"Quality score improved by +{delta:.1f}")
                elif delta < 0:
                    diff["regressions"].append(f"Quality score decreased by {delta:.1f}")
            elif key == "complexity_score":
                if delta < 0:
                    diff["improvements"].append(f"Average complexity reduced by {abs(delta):.1f}")
                elif delta > 0:
                    diff["regressions"].append(f"Average complexity increased by +{delta:.1f}")

        # Compare language breakdown
        old_langs = old_data.get("languages", {})
        new_langs = new_data.get("languages", {})
        all_langs = sorted(set(list(old_langs.keys()) + list(new_langs.keys())))
        for lang in all_langs:
            old_lines = old_langs.get(lang, {}).get("lines", 0)
            new_lines = new_langs.get(lang, {}).get("lines", 0)
            delta = new_lines - old_lines
            if abs(delta) > 0:
                direction = "increased" if delta > 0 else "decreased"
                diff["metrics"][f"{lang} lines"] = {
                    "old": old_lines,
                    "new": new_lines,
                    "delta": delta,
                }
                if delta > 0:
                    diff["improvements"].append(f"{lang}: {abs(delta)} lines added")
                else:
                    diff["regressions"].append(f"{lang}: {abs(delta)} lines removed")

        return diff

    # ---------------------------------------------------- recommendations

    async def get_recommendations(self) -> list[dict]:
        """Generate prioritised improvement recommendations."""
        recs: list[dict] = []

        if not self._file_analyses:
            files = self._source_files()
            await asyncio.gather(*[self.analyze_file(str(f)) for f in files], return_exceptions=True)

        # -- High complexity --
        for path, fa in self._file_analyses.items():
            for fn, cx in fa.complexity.items():
                if cx >= 15:
                    recs.append({
                        "priority": "high",
                        "category": "complexity",
                        "file": path,
                        "name": fn,
                        "message": f"Function '{fn}' has very high complexity ({cx}).",
                        "details": (
                            "Consider breaking it into smaller functions, "
                            "reducing nested conditionals, or extracting helper methods."
                        ),
                    })
                elif cx >= 10:
                    recs.append({
                        "priority": "medium",
                        "category": "complexity",
                        "file": path,
                        "name": fn,
                        "message": f"Function '{fn}' has elevated complexity ({cx}).",
                        "details": "Consider simplifying control flow or extracting sub-functions.",
                    })

        # -- Missing documentation --
        doc_cov = await self.get_documentation_coverage()
        for lang, pct in doc_cov.items():
            if pct < 30:
                recs.append({
                    "priority": "high",
                    "category": "documentation",
                    "message": f"{lang} has very low documentation coverage ({pct}%).",
                    "details": "Add docstrings/comments to all public functions and classes.",
                })
            elif pct < 60:
                recs.append({
                    "priority": "medium",
                    "category": "documentation",
                    "message": f"{lang} documentation coverage is below 60% ({pct}%).",
                    "details": "Consider documenting undocumented public APIs.",
                })

        # -- Dead code --
        dead = await self.find_dead_code()
        dead_imports = [d for d in dead if d["type"] == "unused_import"]
        dead_funcs = [d for d in dead if d["type"] == "unused_function"]
        if len(dead_imports) > 10:
            recs.append({
                "priority": "medium",
                "category": "dead_code",
                "message": f"Found {len(dead_imports)} unused imports across the project.",
                "details": "Remove unused imports to improve readability and reduce load time.",
            })
        if len(dead_funcs) > 5:
            recs.append({
                "priority": "high",
                "category": "dead_code",
                "message": f"Found {len(dead_funcs)} potentially unused functions.",
                "details": "Review and remove unreachable code to reduce maintenance burden.",
            })

        # -- Circular dependencies --
        circular = await self.detect_circular_deps()
        if circular:
            recs.append({
                "priority": "high",
                "category": "architecture",
                "message": f"Detected {len(circular)} circular dependency chain(s).",
                "details": (
                    "Refactor module boundaries to eliminate cycles. "
                    "Consider introducing an abstraction layer or using dependency injection."
                ),
            })

        # -- Naming issues --
        naming_issues = []
        for path, fa in self._file_analyses.items():
            for issue in fa.issues:
                naming_issues.append(issue)
        if len(naming_issues) > 5:
            recs.append({
                "priority": "low",
                "category": "naming",
                "message": f"Found {len(naming_issues)} naming convention violations.",
                "details": "Adopt consistent naming conventions (snake_case for functions, PascalCase for classes).",
            })

        # -- Large files --
        for path, fa in self._file_analyses.items():
            if fa.lines > 500:
                recs.append({
                    "priority": "medium",
                    "category": "structure",
                    "file": path,
                    "message": f"File '{Path(path).name}' is very large ({fa.lines} lines).",
                    "details": "Consider splitting into smaller, focused modules.",
                })

        # -- Type hints (Python) --
        py_files = [fa for fa in self._file_analyses.values() if fa.language == "Python"]
        if py_files:
            hinted, total = 0, 0
            for fa in py_files:
                source = self._read_source(Path(fa.path))
                for func in fa.functions:
                    total += 1
                    ret_pat = re.compile(rf"def\s+{re.escape(func)}\s*\([^)]*\)\s*->")
                    params_pat = re.compile(rf"def\s+{re.escape(func)}\s*\([^)]*:\s*\w")
                    if ret_pat.search(source) or params_pat.search(source):
                        hinted += 1
            if total > 0 and (hinted / total) < 0.5:
                recs.append({
                    "priority": "medium",
                    "category": "type_hints",
                    "message": f"Only {round(hinted / total * 100)}% of Python functions have type hints.",
                    "details": "Adding type hints improves maintainability and enables better static analysis.",
                })

        # Sort: high → medium → low
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recs.sort(key=lambda r: priority_order.get(r["priority"], 99))

        return recs


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

async def analyze_project(project_path: str, ignore_patterns: list[str] | None = None) -> AnalysisResult:
    """Analyze a project and return the result. Convenience wrapper."""
    analyzer = ProjectAnalyzer(project_path, ignore_patterns)
    return await analyzer.analyze()


async def generate_project_report(project_path: str, fmt: str = "markdown") -> str:
    """Generate and return a formatted analysis report."""
    analyzer = ProjectAnalyzer(project_path)
    return await analyzer.generate_report(fmt)
