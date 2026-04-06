"""
Auto-Test Generation — Automatically generate, run, and report on test suites.

Analyzes source code via AST to understand structure, signatures, and patterns,
then produces runnable pytest-compatible test files. Includes a test runner
with subprocess execution, result persistence (SQLite), and rich reporting.

Submodules:
- generator: AST-based test code generation (unit, edge-case, class, fixtures)
- runner: Pytest subprocess execution, result tracking, coverage, retries
- reporter: Console, Markdown, JSON, and trend reports

Usage:
    from agent.auto_test import TestGenerator, TestRunner, TestReporter

    gen = TestGenerator(project_root="/path/to/project")
    await gen.initialize()
    test = await gen.generate_for_file("src/utils.py")

    runner = TestRunner(project_root="/path/to/project")
    await runner.initialize()
    result = await runner.run_tests(test.test_file_path)

    reporter = TestReporter(project_root="/path/to/project")
    print(reporter.generate_console_report([result]))
"""

from agent.auto_test.generator import (
    TestGenerator,
    GeneratedTest,
    FunctionInfo,
    ClassInfo,
)

from agent.auto_test.runner import (
    TestRunner,
    TestResult,
    TestSuiteResult,
    CoverageReport,
)

from agent.auto_test.reporter import TestReporter

__all__ = [
    # Generator
    "TestGenerator",
    "GeneratedTest",
    "FunctionInfo",
    "ClassInfo",
    # Runner
    "TestRunner",
    "TestResult",
    "TestSuiteResult",
    "CoverageReport",
    # Reporter
    "TestReporter",
]
