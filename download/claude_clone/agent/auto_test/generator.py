"""
Auto-Test Generator — Analyzes source code and generates comprehensive test suites.

Uses AST analysis to understand:
- Class structure, methods, inheritance
- Function signatures, parameters, return types
- Import dependencies
- Decorators and special patterns

Generates pytest-compatible test files with:
- Unit tests for each function/method
- Edge case tests (empty inputs, None, boundary values)
- Type error tests
- Integration tests for class interactions
- Fixture suggestions for common setups

Usage:
    gen = TestGenerator(project_root="/path/to/project")
    await gen.initialize()
    result = await gen.generate_for_file("src/utils.py")
    print(result.test_code)
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Directories to skip during project scanning
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "env", ".env", ".mypy", ".ruff_cache",
}

# Standard library modules that don't need special mocking
STDLIB_MODULES = {
    "os", "sys", "re", "json", "time", "datetime", "pathlib", "typing",
    "collections", "itertools", "functools", "copy", "math", "random",
    "hashlib", "io", "subprocess", "threading", "asyncio", "logging",
    "dataclasses", "enum", "abc", "contextlib", "tempfile", "shutil",
    "glob", "fnmatch", "textwrap", "string", "unicodedata", "struct",
    "socket", "ssl", "http", "urllib", "email", "html", "xml", "csv",
    "sqlite3", "uuid", "decimal", "fractions", "statistics", "traceback",
    "inspect", "dis", "ast", "token", "tokenize", "pickle", "shelve",
    "dbm", "gzip", "bz2", "lzma", "zipfile", "tarfile", "configparser",
    "argparse", "optparse", "getopt", "warnings", "contextvars",
}

# Type mapping for generating test values
TYPE_TEST_VALUES: Dict[str, list] = {
    "int": ["0", "1", "-1", "42", "999999"],
    "float": ["0.0", "1.5", "-1.0", "3.14159", "1e10"],
    "str": ['""', '"hello"', '"a" * 100', '"special chars: \\\\n\\\\t"'],
    "bool": ["True", "False"],
    "list": ["[]", "[1, 2, 3]", '["a", "b"]'],
    "dict": ["{}", '{"key": "value"}', '{"a": 1, "b": 2}'],
    "set": ["set()", "{1, 2, 3}"],
    "tuple": ["()", "(1, 2)", '("a", "b")'],
    "bytes": ['b""', 'b"data"'],
    "None": ["None"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FunctionInfo:
    """Extracted metadata about a single function or method."""

    name: str
    args: List[str] = field(default_factory=list)
    defaults: Dict[str, str] = field(default_factory=dict)
    return_annotation: str = ""
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    class_name: str = ""
    docstring: str = ""
    line_number: int = 0
    complexity: int = 1
    complexity_score: str = "low"  # low / medium / high

    def qualified_name(self) -> str:
        """Return the fully-qualified name (e.g. 'MyClass.method_name')."""
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name


@dataclass
class ClassInfo:
    """Extracted metadata about a single class."""

    name: str
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)
    docstring: str = ""
    line_number: int = 0
    has_init: bool = False
    is_abstract: bool = False
    decorators: List[str] = field(default_factory=list)


@dataclass
class GeneratedTest:
    """Result of test generation for a single source file."""

    file_path: str
    test_code: str = ""
    test_file_path: str = ""
    language: str = "python"
    framework: str = "pytest"
    functions_covered: List[str] = field(default_factory=list)
    classes_covered: List[str] = field(default_factory=list)
    total_tests: int = 0
    estimated_coverage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# TestGenerator
# ──────────────────────────────────────────────────────────────────────────────

class TestGenerator:
    """
    AST-based test suite generator.

    Scans Python source files, extracts function and class metadata via AST
    parsing, and generates runnable pytest test code including:
    - Basic unit tests (Arrange / Act / Assert)
    - Edge-case tests (empty inputs, None, boundary values, type errors)
    - Class construction and method tests
    - Pytest fixture suggestions
    """

    def __init__(self, project_root: str) -> None:
        """
        Parameters
        ----------
        project_root:
            Absolute or relative path to the project root directory.
        """
        self.project_root = Path(project_root).resolve()
        self._file_cache: Dict[str, Tuple[str, ast.AST]] = {}
        self._python_files: List[str] = []
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Scan the project for Python files and build the file cache."""
        await self._scan_project()
        self._initialized = True
        logger.info(
            "TestGenerator initialized: %d Python files found in %s",
            len(self._python_files),
            self.project_root,
        )

    async def _scan_project(self) -> None:
        """Discover all Python files in the project, skipping excluded dirs."""

        def _scan() -> None:
            self._file_cache = {}
            self._python_files = []
            for py_file in sorted(self.project_root.rglob("*.py")):
                rel = py_file.relative_to(self.project_root)
                if any(part in SKIP_DIRS for part in rel.parts):
                    continue
                # Skip test files — we don't generate tests for tests
                if any(part == "tests" or part.startswith("test_") for part in rel.parts):
                    continue
                try:
                    content = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(content, filename=str(py_file))
                    self._file_cache[str(rel)] = (content, tree)
                    self._python_files.append(str(rel))
                except (SyntaxError, UnicodeDecodeError, OSError):
                    pass

        await self._run_sync(_scan)

    async def _run_sync(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous function in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: func(*args, **kwargs)
        )

    # ── File Analysis ─────────────────────────────────────────────────────

    async def analyze_file(self, file_path: str) -> Tuple[List[FunctionInfo], List[ClassInfo]]:
        """
        Parse a source file and extract function and class metadata.

        Parameters
        ----------
        file_path:
            Relative path to the Python file within the project root.

        Returns
        -------
        Tuple of (functions, classes) with extracted metadata.
        """
        key = str(file_path)
        if key not in self._file_cache:
            raise FileNotFoundError(f"File not found in project cache: {file_path}")

        content, tree = self._file_cache[key]

        def _extract() -> Tuple[List[FunctionInfo], List[ClassInfo]]:
            functions: List[FunctionInfo] = []
            classes: List[ClassInfo] = []

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(self._extract_function(node, class_name=""))
                elif isinstance(node, ast.ClassDef):
                    cls = self._extract_class(node)
                    classes.append(cls)
                    functions.extend(cls.methods)

            return functions, classes

        return await self._run_sync(_extract)

    def _extract_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str) -> FunctionInfo:
        """Extract metadata from a function or method AST node."""
        # Argument names (skip 'self' and 'cls')
        args: List[str] = []
        defaults: Dict[str, str] = {}
        all_args = (
            node.args.posonlyargs or []
            + node.args.args
            + node.args.kwonlyargs or []
        )
        all_defaults = (
            [None] * (len(node.args.args) - len(node.args.defaults))
            + list(node.args.defaults)
            + [None] * (len(node.args.kwonlyargs) - len(node.args.kw_defaults or []))
            + list(node.args.kw_defaults or [])
        )

        # Build args list with defaults
        regular_args = node.args.posonlyargs or [] + node.args.args
        for i, arg in enumerate(regular_args):
            if arg.arg in ("self", "cls"):
                continue
            args.append(arg.arg)
            default_idx = i - (1 if regular_args[0].arg in ("self", "cls") and class_name else 0)
            if default_idx >= 0 and default_idx < len(node.args.defaults):
                defaults[arg.arg] = ast.dump(node.args.defaults[-(len(regular_args) - default_idx)])

        for i, arg in enumerate(node.args.kwonlyargs or []):
            args.append(arg.arg)
            if node.args.kw_defaults and node.args.kw_defaults[i] is not None:
                defaults[arg.arg] = self._ast_value_to_str(node.args.kw_defaults[i])

        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        # Return annotation
        return_annotation = ""
        if node.returns:
            return_annotation = self._infer_type_from_annotation(node.returns)

        # Decorators
        decorators: List[str] = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"{ast.dump(dec.value)}.{dec.attr}")
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(dec.func.attr)

        complexity = self._calculate_complexity(node)
        if complexity <= 5:
            complexity_score = "low"
        elif complexity <= 10:
            complexity_score = "medium"
        else:
            complexity_score = "high"

        return FunctionInfo(
            name=node.name,
            args=args,
            defaults=defaults,
            return_annotation=return_annotation,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=bool(class_name),
            class_name=class_name,
            docstring=ast.get_docstring(node) or "",
            line_number=node.lineno,
            complexity=complexity,
            complexity_score=complexity_score,
        )

    def _extract_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extract metadata from a class AST node."""
        # Bases
        bases: List[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
            elif isinstance(base, ast.Subscript):
                if isinstance(base.value, ast.Name):
                    bases.append(base.value.id)

        # Methods
        methods: List[FunctionInfo] = []
        has_init = False
        is_abstract = False

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._extract_function(child, class_name=node.name)
                methods.append(method)
                if child.name == "__init__":
                    has_init = True

        # Check for abstract
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and "abstract" in dec.id.lower():
                is_abstract = True

        # Also check methods for @abstractmethod
        for method in methods:
            if "abstractmethod" in method.decorators:
                is_abstract = True

        # Class decorators
        decorators: List[str] = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)

        return ClassInfo(
            name=node.name,
            bases=bases,
            methods=methods,
            docstring=ast.get_docstring(node) or "",
            line_number=node.lineno,
            has_init=has_init,
            is_abstract=is_abstract,
            decorators=decorators,
        )

    # ── Test Generation ───────────────────────────────────────────────────

    async def generate_for_file(self, file_path: str, style: str = "pytest") -> GeneratedTest:
        """
        Generate a complete test suite for a single source file.

        Parameters
        ----------
        file_path:
            Relative path to the Python file.
        style:
            Test framework style (currently only 'pytest' is supported).

        Returns
        -------
        GeneratedTest containing the full test source code and metadata.
        """
        if not self._initialized:
            raise RuntimeError("TestGenerator not initialized. Call initialize() first.")

        if style != "pytest":
            raise ValueError(f"Unsupported test style: {style}. Only 'pytest' is supported.")

        key = str(file_path)
        if key not in self._file_cache:
            raise FileNotFoundError(f"File not found in project: {file_path}")

        functions, classes = await self.analyze_file(file_path)

        def _build() -> GeneratedTest:
            # Separate top-level functions from class methods
            top_functions = [f for f in functions if not f.is_method]

            # Module import
            module_import = self._generate_module_import(file_path)

            # Fixture suggestions
            fixture_code = self._suggest_fixtures(classes)

            # Mock imports for external dependencies
            mock_imports = self._generate_mock_imports(classes, top_functions)

            # Unit tests
            unit_tests = self._generate_unit_tests(top_functions, classes)

            # Class tests
            class_tests = ""
            for cls in classes:
                class_tests += self._generate_class_tests(cls)

            # Assemble the full test file
            parts: List[str] = []
            parts.append('"""')
            parts.append(f"Auto-generated tests for {file_path}")
            parts.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
            parts.append(f"Framework: pytest")
            parts.append(f"Functions covered: {len(top_functions)}")
            parts.append(f"Classes covered: {len(classes)}")
            parts.append('"""')
            parts.append("")
            parts.append("from __future__ import annotations")
            parts.append("")
            parts.append("import pytest")
            parts.append(module_import)
            parts.append(mock_imports)
            parts.append("")

            if fixture_code:
                parts.append(fixture_code)
                parts.append("")

            if unit_tests:
                parts.append(unit_tests)

            if class_tests:
                parts.append(class_tests)

            test_code = "\n".join(parts)

            # Count generated test functions
            test_count = len(re.findall(r"^\s*def test_", test_code, re.MULTILINE))

            # Estimate coverage (rough heuristic)
            total_items = len(top_functions) + sum(len(c.methods) for c in classes)
            tested_items = min(test_count, total_items * 3)  # roughly 3 tests per item
            estimated_cov = min(1.0, (tested_items / max(1, total_items * 4)))

            # Suggested test file path
            path = Path(file_path)
            test_path = f"tests/test_{path.stem}.py"

            return GeneratedTest(
                file_path=key,
                test_code=test_code,
                test_file_path=test_path,
                language="python",
                framework="pytest",
                functions_covered=[f.name for f in top_functions],
                classes_covered=[c.name for c in classes],
                total_tests=test_count,
                estimated_coverage=round(estimated_cov, 2),
                metadata={
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source_file": key,
                    "total_functions": len(top_functions),
                    "total_classes": len(classes),
                    "total_methods": sum(len(c.methods) for c in classes),
                    "generator_version": "1.0.0",
                },
            )

        return await self._run_sync(_build)

    async def generate_for_directory(
        self,
        dir_path: str,
        pattern: str = "*.py",
    ) -> List[GeneratedTest]:
        """
        Generate test suites for all Python files in a directory.

        Parameters
        ----------
        dir_path:
            Relative path to the directory.
        pattern:
            Glob pattern to match source files.

        Returns
        -------
        List of GeneratedTest results.
        """
        if not self._initialized:
            raise RuntimeError("TestGenerator not initialized. Call initialize() first.")

        target_dir = self.project_root / dir_path
        if not target_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        results: List[GeneratedTest] = []
        for py_file in sorted(target_dir.rglob(pattern)):
            rel = str(py_file.relative_to(self.project_root))
            if any(part in SKIP_DIRS for part in py_file.parts):
                continue
            if rel not in self._file_cache:
                continue
            try:
                result = await self.generate_for_file(rel)
                results.append(result)
            except Exception as e:
                logger.warning("Failed to generate tests for %s: %s", rel, e)

        return results

    # ── Core Generation Logic ─────────────────────────────────────────────

    def _generate_unit_tests(
        self,
        functions: List[FunctionInfo],
        classes: List[ClassInfo],
    ) -> str:
        """Generate unit tests for top-level functions."""
        parts: List[str] = []

        for func in functions:
            # Skip private/dunder functions
            if func.name.startswith("_"):
                continue

            parts.append(f"# ── Tests for {func.name} ────────────────────────────")
            parts.append("")

            # Basic test
            parts.append(self._generate_basic_test(func))

            # Edge cases
            if func.args:
                parts.append(self._generate_edge_case_tests(func))

            # Type error tests
            if func.args:
                parts.append(self._generate_type_error_tests(func))

            parts.append("")

        return "\n".join(parts)

    def _generate_basic_test(self, func: FunctionInfo) -> str:
        """Generate a basic Arrange/Act/Assert test for a function."""
        safe_name = func.name
        lines: List[str] = []
        lines.append(f"def test_{safe_name}_basic():")
        lines.append(f'    """Basic test for {func.qualified_name()}."""')

        if func.is_async:
            lines.append("")

        # Build sample arguments
        if not func.args:
            lines.append("    # Arrange")
            lines.append("    # (no arguments required)")
            lines.append("")
            lines.append("    # Act")
            if func.is_async:
                lines.append(f"    result = asyncio.run({safe_name}())")
            else:
                lines.append(f"    result = {safe_name}()")
            lines.append("")
            lines.append("    # Assert")
            if func.return_annotation in ("bool",):
                lines.append("    assert isinstance(result, bool)")
            elif func.return_annotation in ("int",):
                lines.append("    assert isinstance(result, int)")
            elif func.return_annotation in ("float",):
                lines.append("    assert isinstance(result, float)")
            elif func.return_annotation in ("str",):
                lines.append("    assert isinstance(result, str)")
            elif func.return_annotation in ("list", "List"):
                lines.append("    assert isinstance(result, list)")
            elif func.return_annotation in ("dict", "Dict"):
                lines.append("    assert isinstance(result, dict)")
            elif func.return_annotation:
                lines.append(f"    # result should be of type {func.return_annotation}")
                lines.append("    assert result is not None")
            else:
                lines.append("    # Verify result is returned")
                lines.append("    assert result is not None")
        else:
            lines.append("    # Arrange")
            sample_args = self._generate_sample_args(func)
            for arg_name, arg_value in sample_args.items():
                lines.append(f"    {arg_name} = {arg_value}")
            lines.append("")
            lines.append("    # Act")
            args_str = ", ".join(func.args)
            if func.is_async:
                lines.append(f"    result = asyncio.run({safe_name}({args_str}))")
            else:
                lines.append(f"    result = {safe_name}({args_str})")
            lines.append("")
            lines.append("    # Assert")
            if func.return_annotation:
                lines.append(f"    # result should be of type {func.return_annotation}")
            lines.append("    assert result is not None")

        lines.append("")
        return "\n".join(lines)

    def _generate_edge_case_tests(self, func: FunctionInfo) -> str:
        """Generate edge case tests for a function with arguments."""
        lines: List[str] = []
        safe_name = func.name

        # Test with empty string
        if len(func.args) >= 1:
            arg0 = func.args[0]
            lines.append(f"def test_{safe_name}_empty_string():")
            lines.append(f'    """Test {safe_name} with empty string input."""')
            lines.append(f"    {arg0} = ''")
            if len(func.args) > 1:
                for extra_arg in func.args[1:]:
                    lines.append(f"    {extra_arg} = ''")
            args_str = ", ".join(func.args)
            lines.append(f"    result = {safe_name}({args_str})")
            lines.append(f"    # Should handle empty string gracefully")
            lines.append(f"    assert result is not None")
            lines.append("")

        # Test with None
        lines.append(f"def test_{safe_name}_none_input():")
        lines.append(f'    """Test {safe_name} with None input."""')
        for arg in func.args:
            lines.append(f"    {arg} = None")
        args_str = ", ".join(func.args)
        lines.append(f"    try:")
        lines.append(f"        result = {safe_name}({args_str})")
        lines.append(f"        # Should handle None input gracefully")
        lines.append(f"        assert result is not None")
        lines.append(f"    except (TypeError, AttributeError, ValueError):")
        lines.append(f"        pass  # Expected for None input on strict functions")
        lines.append("")

        # Test with boundary value 0
        lines.append(f"def test_{safe_name}_zero_value():")
        lines.append(f'    """Test {safe_name} with zero/boundary values."""')
        for arg in func.args:
            lines.append(f"    {arg} = 0")
        args_str = ", ".join(func.args)
        lines.append(f"    try:")
        lines.append(f"        result = {safe_name}({args_str})")
        lines.append(f"        assert result is not None")
        lines.append(f"    except (TypeError, ValueError, ZeroDivisionError):")
        lines.append(f"        pass  # Some functions cannot accept 0")
        lines.append("")

        # Test with negative value
        lines.append(f"def test_{safe_name}_negative_value():")
        lines.append(f'    """Test {safe_name} with negative value."""')
        for arg in func.args:
            lines.append(f"    {arg} = -1")
        args_str = ", ".join(func.args)
        lines.append(f"    try:")
        lines.append(f"        result = {safe_name}({args_str})")
        lines.append(f"        assert result is not None")
        lines.append(f"    except (TypeError, ValueError):")
        lines.append(f"        pass  # Some functions cannot accept negative values")
        lines.append("")

        # Test with empty collection
        lines.append(f"def test_{safe_name}_empty_collection():")
        lines.append(f'    """Test {safe_name} with empty collection."""')
        for arg in func.args:
            lines.append(f"    {arg} = []")
        args_str = ", ".join(func.args)
        lines.append(f"    try:")
        lines.append(f"        result = {safe_name}({args_str})")
        lines.append(f"        assert result is not None")
        lines.append(f"    except (TypeError, IndexError, ValueError):")
        lines.append(f"        pass  # Some functions cannot accept empty collections")
        lines.append("")

        return "\n".join(lines)

    def _generate_type_error_tests(self, func: FunctionInfo) -> str:
        """Generate type error tests for a function."""
        lines: List[str] = []
        safe_name = func.name

        lines.append(f"def test_{safe_name}_type_mismatch():")
        lines.append(f'    """Test {safe_name} with incorrect argument types."""')
        # Pass a string where an int might be expected (or vice versa)
        for arg in func.args:
            lines.append(f"    {arg} = 'unexpected_string_value'")
        args_str = ", ".join(func.args)
        lines.append(f"    try:")
        lines.append(f"        result = {safe_name}({args_str})")
        lines.append(f"        # If it doesn't raise, verify result")
        lines.append(f"        assert result is not None")
        lines.append(f"    except (TypeError, ValueError, AttributeError):")
        lines.append(f"        pass  # Expected for type-mismatched input")
        lines.append("")

        return "\n".join(lines)

    def _generate_class_tests(self, class_info: ClassInfo) -> str:
        """Generate tests for a class, including construction and method tests."""
        lines: List[str] = []
        cls = class_info
        safe_cls = cls.name

        lines.append(f"# ── Tests for {safe_cls} ─────────────────────────────")
        lines.append("")

        # Construction test
        lines.append(f"class Test{safe_cls}:")
        lines.append(f'    """Tests for {safe_cls}."""')
        lines.append("")

        # Test instantiation
        if not cls.is_abstract:
            lines.append(f"    def test_instantiation(self):")
            lines.append(f'        """Test basic instantiation of {safe_cls}."""')

            if cls.has_init:
                init_method = next(
                    (m for m in cls.methods if m.name == "__init__"), None
                )
                if init_method and init_method.args:
                    sample_args = self._generate_sample_args(init_method)
                    args_str = ", ".join(sample_args.keys())
                    lines.append(f"        # Arrange")
                    for arg_name, arg_value in sample_args.items():
                        lines.append(f"        {arg_name} = {arg_value}")
                    lines.append(f"        # Act")
                    lines.append(f"        instance = {safe_cls}({args_str})")
                    lines.append(f"        # Assert")
                    lines.append(f"        assert isinstance(instance, {safe_cls})")
                else:
                    lines.append(f"        instance = {safe_cls}()")
                    lines.append(f"        assert isinstance(instance, {safe_cls})")
            else:
                lines.append(f"        instance = {safe_cls}()")
                lines.append(f"        assert isinstance(instance, {safe_cls})")
            lines.append("")

            # Test with no arguments (should fail if __init__ requires args)
            if cls.has_init:
                init_method = next(
                    (m for m in cls.methods if m.name == "__init__"), None
                )
                if init_method and init_method.args:
                    lines.append(f"    def test_instantiation_no_args_raises(self):")
                    lines.append(f'        """Test that {safe_cls} raises without required args."""')
                    lines.append(f"        with pytest.raises(TypeError):")
                    lines.append(f"            {safe_cls}()")
                    lines.append("")

        # Method tests
        for method in cls.methods:
            if method.name.startswith("_"):
                continue
            lines.append(self._generate_method_test(cls, method))

        # Test dunder methods
        if any(m.name == "__eq__" for m in cls.methods):
            lines.append(self._generate_dunder_eq_test(cls))

        if any(m.name == "__repr__" for m in cls.methods) or any(
            m.name == "__str__" for m in cls.methods
        ):
            lines.append(self._generate_dunder_str_test(cls))

        lines.append("")
        return "\n".join(lines)

    def _generate_method_test(self, cls: ClassInfo, method: FunctionInfo) -> str:
        """Generate a test for a class method."""
        lines: List[str] = []
        safe_name = f"{cls.name}_{method.name}"

        lines.append(f"    def test_{safe_name}(self):")
        lines.append(f'        """Test {cls.name}.{method.name}."""')

        # Build instance first
        if cls.has_init and not cls.is_abstract:
            init_method = next(
                (m for m in cls.methods if m.name == "__init__"), None
            )
            if init_method and init_method.args:
                sample_args = self._generate_sample_args(init_method)
                args_str = ", ".join(sample_args.keys())
                lines.append(f"        # Arrange")
                for arg_name, arg_value in sample_args.items():
                    lines.append(f"        {arg_name} = {arg_value}")
                lines.append(f"        instance = {cls.name}({args_str})")
            else:
                lines.append(f"        instance = {cls.name}()")
        else:
            lines.append(f"        # Arrange (abstract class — create mock)")
            lines.append(f"        instance = type('{cls.name}', (), {{}})()")

        # Call method
        method_args = [a for a in method.args if a not in ("self", "cls")]
        if method_args:
            for arg in method_args:
                lines.append(f"        {arg} = ''")
            args_str = ", ".join(method_args)
            lines.append(f"        # Act")
            if method.is_async:
                lines.append(f"        result = asyncio.run(instance.{method.name}({args_str}))")
            else:
                lines.append(f"        result = instance.{method.name}({args_str})")
        else:
            lines.append(f"        # Act")
            if method.is_async:
                lines.append(f"        result = asyncio.run(instance.{method.name}())")
            else:
                lines.append(f"        result = instance.{method.name}()")

        lines.append(f"        # Assert")
        lines.append(f"        assert result is not None")
        lines.append("")

        return "\n".join(lines)

    def _generate_dunder_eq_test(self, cls: ClassInfo) -> str:
        """Generate test for __eq__ method."""
        lines: List[str] = []
        lines.append(f"    def test_{cls.name}_equality(self):")
        lines.append(f'        """Test {cls.name} equality comparison."""')

        if cls.has_init:
            init_method = next((m for m in cls.methods if m.name == "__init__"), None)
            if init_method and init_method.args:
                sample_args = self._generate_sample_args(init_method)
                args_str = ", ".join(sample_args.keys())
                for arg_name, arg_value in sample_args.items():
                    lines.append(f"        {arg_name} = {arg_value}")
                lines.append(f"        a = {cls.name}({args_str})")
                lines.append(f"        b = {cls.name}({args_str})")
            else:
                lines.append(f"        a = {cls.name}()")
                lines.append(f"        b = {cls.name}()")
        else:
            lines.append(f"        a = {cls.name}()")
            lines.append(f"        b = {cls.name}()")

        lines.append(f"        assert a == b")
        lines.append(f"        assert not (a != b)")
        lines.append("")
        return "\n".join(lines)

    def _generate_dunder_str_test(self, cls: ClassInfo) -> str:
        """Generate test for __str__/__repr__ method."""
        lines: List[str] = []
        lines.append(f"    def test_{cls.name}_string_representation(self):")
        lines.append(f'        """Test {cls.name} string representation."""')

        if cls.has_init:
            init_method = next((m for m in cls.methods if m.name == "__init__"), None)
            if init_method and init_method.args:
                sample_args = self._generate_sample_args(init_method)
                args_str = ", ".join(sample_args.keys())
                for arg_name, arg_value in sample_args.items():
                    lines.append(f"        {arg_name} = {arg_value}")
                lines.append(f"        instance = {cls.name}({args_str})")
            else:
                lines.append(f"        instance = {cls.name}()")
        else:
            lines.append(f"        instance = {cls.name}()")

        lines.append(f'        repr_str = repr(instance)')
        lines.append(f'        assert isinstance(repr_str, str)')
        lines.append(f'        assert len(repr_str) > 0')
        lines.append("")
        return "\n".join(lines)

    # ── Import Generation ─────────────────────────────────────────────────

    def _generate_module_import(self, file_path: str) -> str:
        """Generate the import statement for the module under test."""
        path = Path(file_path)
        # Convert file path to module path
        parts = list(path.parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        module_name = ".".join(parts)
        return f"from {module_name} import *"

    def _generate_mock_imports(
        self,
        classes: List[ClassInfo],
        functions: List[FunctionInfo],
    ) -> str:
        """Generate mock/patch import statements for external dependencies."""
        parts: List[str] = []

        # Collect all names that might need mocking
        external_deps: Set[str] = set()
        for func in functions:
            for dec in func.decorators:
                if dec not in ("staticmethod", "classmethod", "property") and not dec.startswith("_"):
                    external_deps.add(dec)
        for cls in classes:
            for base in cls.bases:
                if base not in STDLIB_MODULES:
                    external_deps.add(base)
            for dec in cls.decorators:
                if dec not in ("staticmethod", "classmethod", "property") and not dec.startswith("_"):
                    external_deps.add(dec)

        if external_deps:
            parts.append("")
            parts.append("# External dependencies that may need mocking")
            for dep in sorted(external_deps):
                parts.append(f"# from unittest.mock import patch  # {dep}")

        return "\n".join(parts)

    def _suggest_fixtures(self, classes: List[ClassInfo]) -> str:
        """Suggest pytest fixtures for common class setups."""
        if not classes:
            return ""

        lines: List[str] = []
        lines.append("# ── Fixtures ────────────────────────────────────────────")
        lines.append("")

        for cls in classes:
            if cls.is_abstract:
                continue
            if not cls.has_init:
                continue
            init_method = next(
                (m for m in cls.methods if m.name == "__init__"), None
            )
            if init_method and init_method.args:
                lines.append(f"@pytest.fixture")
                lines.append(f"def {cls.name.lower()}_instance():")
                lines.append(f'    """Provide a default {cls.name} instance for tests."""')
                sample_args = self._generate_sample_args(init_method)
                args_str = ", ".join(sample_args.keys())
                lines.append(f"    # Arrange fixture arguments")
                for arg_name, arg_value in sample_args.items():
                    lines.append(f"    {arg_name} = {arg_value}")
                lines.append(f"    return {cls.name}({args_str})")
                lines.append("")
            else:
                lines.append(f"@pytest.fixture")
                lines.append(f"def {cls.name.lower()}_instance():")
                lines.append(f'    """Provide a default {cls.name} instance for tests."""')
                lines.append(f"    return {cls.name}()")
                lines.append("")

        return "\n".join(lines)

    # ── Helper Methods ────────────────────────────────────────────────────

    def _generate_sample_args(self, func: FunctionInfo) -> Dict[str, str]:
        """Generate sample argument values for a function call."""
        sample_args: Dict[str, str] = {}
        for arg in func.args:
            if arg in ("self", "cls"):
                continue
            if arg.startswith("*"):
                continue
            if arg in func.defaults:
                continue  # Has a default, skip
            sample_args[arg] = "''"
        return sample_args

    def _calculate_complexity(self, node: ast.AST) -> int:
        """
        Compute McCabe cyclomatic complexity of a function/method.

        Counts decision points: if, for, while, except, with, assert,
        boolean operators, comprehensions.
        """
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

    def _infer_type_from_annotation(self, annotation: ast.AST) -> str:
        """Convert an AST annotation node to a human-readable type string."""
        if annotation is None:
            return ""
        if isinstance(annotation, ast.Constant):
            return str(annotation.value)
        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Attribute):
            return f"{self._infer_type_from_annotation(annotation.value)}.{annotation.attr}"
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id
            return self._infer_type_from_annotation(annotation.value)
        if isinstance(annotation, ast.BinOp):
            # Union types like str | int
            left = self._infer_type_from_annotation(annotation.left)
            right = self._infer_type_from_annotation(annotation.right)
            return f"{left} | {right}"
        if isinstance(annotation, ast.BitOr):
            left = self._infer_type_from_annotation(annotation.left)
            right = self._infer_type_from_annotation(annotation.right)
            return f"{left} | {right}"
        if isinstance(annotation, ast.Call):
            # Optional[str], List[int], Dict[str, int], etc.
            if isinstance(annotation.func, ast.Name):
                return annotation.func.id
            if isinstance(annotation.func, ast.Attribute):
                return annotation.func.attr
        return ""

    def _ast_value_to_str(self, node: ast.AST) -> str:
        """Convert an AST constant node to its Python source representation."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return repr(node.value)
            return repr(node.value)
        if isinstance(node, ast.NameConstant):  # Python 3.7 compat
            return repr(node.value)
        if isinstance(node, ast.Name):
            return node.id
        return "None"
