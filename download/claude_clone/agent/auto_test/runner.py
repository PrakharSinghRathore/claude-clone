"""
Auto-Test Runner — Executes generated tests and tracks results.

Supports:
- Running pytest programmatically via subprocess
- Capturing stdout/stderr and exit codes
- Tracking pass/fail/skip/error counts
- Coverage estimation
- Parallel test execution for multiple test files
- Test result persistence (SQLite)
- Retry on failure with configurable attempts

Usage:
    runner = TestRunner(project_root="/path/to/project")
    await runner.initialize()
    result = await runner.run_tests(test_file_path="tests/test_utils.py")
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "~/.claude_clone/test_results.db"

# Pytest exit codes
PYTEST_EXIT_OK = 0
PYTEST_EXIT_TESTS_FAILED = 1
PYTEST_EXIT_INTERRUPTED = 2
PYTEST_EXIT_INTERNAL_ERROR = 3
PYTEST_EXIT_NO_TESTS = 5

# Status labels
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_ERROR = "error"


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """Result of a single test function execution."""

    id: str = ""
    test_name: str = ""
    test_file: str = ""
    status: str = STATUS_PASSED
    duration: float = 0.0
    error_message: str = ""
    traceback: str = ""
    stdout: str = ""
    timestamp: str = ""


@dataclass
class TestSuiteResult:
    """Result of running an entire test file (suite)."""

    id: str = ""
    test_file: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    results: List[TestResult] = field(default_factory=list)
    timestamp: str = ""

    def summary(self) -> str:
        """Return a human-readable summary string."""
        status_icon = "PASSED" if self.failed == 0 and self.errors == 0 else "FAILED"
        parts = [
            f"[{status_icon}] {self.test_file}",
            f"  Total: {self.total}  Passed: {self.passed}  Failed: {self.failed}"
            f"  Skipped: {self.skipped}  Errors: {self.errors}",
            f"  Duration: {self.duration:.2f}s",
        ]
        return "\n".join(parts)

    def pass_rate(self) -> float:
        """Return the pass rate as a float between 0.0 and 1.0."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "test_file": self.test_file,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "duration": round(self.duration, 3),
            "pass_rate": round(self.pass_rate(), 4),
            "timestamp": self.timestamp,
            "results": [
                {
                    "id": r.id,
                    "test_name": r.test_name,
                    "test_file": r.test_file,
                    "status": r.status,
                    "duration": round(r.duration, 3),
                    "error_message": r.error_message,
                    "traceback": r.traceback,
                    "timestamp": r.timestamp,
                }
                for r in self.results
            ],
        }


@dataclass
class CoverageReport:
    """Coverage report for a test run."""

    total_files: int = 0
    covered_files: int = 0
    total_lines: int = 0
    covered_lines: int = 0
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    file_reports: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        pct = f"{self.line_coverage:.1%}"
        parts = [
            f"Coverage: {self.line_coverage:.1%} ({self.covered_lines}/{self.total_lines} lines)",
            f"Files covered: {self.covered_files}/{self.total_files}",
        ]
        if self.branch_coverage > 0:
            parts.append(f"Branch coverage: {self.branch_coverage:.1%}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "total_files": self.total_files,
            "covered_files": self.covered_files,
            "total_lines": self.total_lines,
            "covered_lines": self.covered_lines,
            "line_coverage": round(self.line_coverage, 4),
            "branch_coverage": round(self.branch_coverage, 4),
            "file_reports": self.file_reports,
        }


# ──────────────────────────────────────────────────────────────────────────────
# TestRunner
# ──────────────────────────────────────────────────────────────────────────────

class TestRunner:
    """
    Executes generated test suites and persists results.

    Uses subprocess to invoke pytest with verbose output, then parses
    the results into structured TestSuiteResult objects. Supports:
    - Single file and batch execution
    - Parallel execution with configurable concurrency
    - Coverage measurement via coverage.py
    - Retry-on-failure with configurable attempts
    - SQLite-backed result history for trend analysis
    """

    def __init__(
        self,
        project_root: str,
        db_path: str = DEFAULT_DB_PATH,
        python_path: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        project_root:
            Absolute or relative path to the project root directory.
        db_path:
            Path to the SQLite database for storing test results.
        python_path:
            Optional path to the Python interpreter to use. Defaults to
            ``sys.executable``.
        """
        self.project_root = Path(project_root).resolve()
        self.db_path = Path(db_path).expanduser().resolve()
        self.python_path = python_path or sys.executable
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create the database and prepare the runner."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(self._connect)
        await self._run_sync(self._create_tables)
        self._initialized = True
        logger.info("TestRunner initialized (db=%s)", self.db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._run_sync(self._conn.close)
            self._conn = None

    def _connect(self) -> None:
        """Open the SQLite database connection."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

    def _create_tables(self) -> None:
        """Create the result storage tables."""
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS test_suite_results (
                id            TEXT PRIMARY KEY,
                test_file     TEXT NOT NULL,
                total         INTEGER DEFAULT 0,
                passed        INTEGER DEFAULT 0,
                failed        INTEGER DEFAULT 0,
                skipped       INTEGER DEFAULT 0,
                errors        INTEGER DEFAULT 0,
                duration      REAL DEFAULT 0.0,
                pass_rate     REAL DEFAULT 0.0,
                timestamp     TEXT NOT NULL,
                raw_output    TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS test_results (
                id            TEXT PRIMARY KEY,
                suite_id      TEXT NOT NULL,
                test_name     TEXT NOT NULL,
                test_file     TEXT NOT NULL,
                status        TEXT NOT NULL,
                duration      REAL DEFAULT 0.0,
                error_message TEXT DEFAULT '',
                traceback     TEXT DEFAULT '',
                timestamp     TEXT NOT NULL,
                FOREIGN KEY (suite_id) REFERENCES test_suite_results(id)
            );

            CREATE TABLE IF NOT EXISTS coverage_reports (
                id              TEXT PRIMARY KEY,
                suite_id        TEXT,
                total_files     INTEGER DEFAULT 0,
                covered_files   INTEGER DEFAULT 0,
                total_lines     INTEGER DEFAULT 0,
                covered_lines   INTEGER DEFAULT 0,
                line_coverage   REAL DEFAULT 0.0,
                branch_coverage REAL DEFAULT 0.0,
                timestamp       TEXT NOT NULL,
                raw_output      TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_suite_file ON test_suite_results(test_file);
            CREATE INDEX IF NOT EXISTS idx_suite_ts ON test_suite_results(timestamp);
            CREATE INDEX IF NOT EXISTS idx_results_suite ON test_results(suite_id);
            CREATE INDEX IF NOT EXISTS idx_results_status ON test_results(status);
            CREATE INDEX IF NOT EXISTS idx_coverage_ts ON coverage_reports(timestamp);
        """)
        self._conn.commit()

    def _ensure_conn(self) -> sqlite3.Connection:
        """Return the active DB connection or raise."""
        if self._conn is None:
            raise RuntimeError("TestRunner not initialized.")
        return self._conn

    async def _run_sync(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous function in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: func(*args, **kwargs)
        )

    # ── Single File Execution ─────────────────────────────────────────────

    async def run_tests(
        self,
        test_file_path: str,
        retries: int = 0,
        timeout: int = 60,
    ) -> TestSuiteResult:
        """
        Run a single test file with pytest.

        Parameters
        ----------
        test_file_path:
            Path to the test file (relative to project root or absolute).
        retries:
            Number of retry attempts on failure.
        timeout:
            Maximum execution time in seconds.

        Returns
        -------
        TestSuiteResult with detailed results for every test collected.
        """
        if not self._initialized:
            raise RuntimeError("TestRunner not initialized. Call initialize() first.")

        # Resolve the test file path
        test_path = Path(test_file_path)
        if not test_path.is_absolute():
            test_path = self.project_root / test_path
        if not test_path.exists():
            return TestSuiteResult(
                id=str(uuid.uuid4())[:12],
                test_file=str(test_file_path),
                timestamp=datetime.now(timezone.utc).isoformat(),
                errors=1,
            )

        # Build the pytest command
        command = [
            self.python_path,
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "--tb=short",
            "--no-header",
            "-q",
        ]

        attempt = 0
        max_attempts = 1 + retries

        while attempt < max_attempts:
            attempt += 1
            try:
                stdout, stderr, exit_code, duration = await self._run_subprocess(
                    command, timeout
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Test file %s timed out after %ds (attempt %d/%d)",
                    test_file_path, timeout, attempt, max_attempts,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(1)
                    continue
                return TestSuiteResult(
                    id=str(uuid.uuid4())[:12],
                    test_file=str(test_file_path),
                    total=0,
                    errors=1,
                    duration=timeout,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    results=[TestResult(
                        id=str(uuid.uuid4())[:12],
                        test_name="[timeout]",
                        test_file=str(test_file_path),
                        status=STATUS_ERROR,
                        error_message=f"Test execution timed out after {timeout}s",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )],
                )

            # Parse output
            results = self._parse_pytest_output(stdout, exit_code)

            suite_result = TestSuiteResult(
                id=str(uuid.uuid4())[:12],
                test_file=str(test_file_path),
                total=len(results),
                passed=sum(1 for r in results if r.status == STATUS_PASSED),
                failed=sum(1 for r in results if r.status == STATUS_FAILED),
                skipped=sum(1 for r in results if r.status == STATUS_SKIPPED),
                errors=sum(1 for r in results if r.status == STATUS_ERROR),
                duration=duration,
                results=results,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # If all passed or no retries left, return
            if suite_result.failed == 0 or attempt >= max_attempts:
                # Persist results
                await self._persist_result(suite_result, stdout)
                return suite_result

            logger.info(
                "Retrying %s (attempt %d/%d, %d failures)",
                test_file_path, attempt, max_attempts, suite_result.failed,
            )
            await asyncio.sleep(1)

        # Should not reach here, but just in case
        return TestSuiteResult(
            id=str(uuid.uuid4())[:12],
            test_file=str(test_file_path),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ── Batch Execution ───────────────────────────────────────────────────

    async def run_test_suite(
        self,
        test_files: List[str],
        max_concurrent: int = 2,
    ) -> List[TestSuiteResult]:
        """
        Run multiple test files in parallel.

        Parameters
        ----------
        test_files:
            List of test file paths.
        max_concurrent:
            Maximum number of concurrent test executions.

        Returns
        -------
        List of TestSuiteResult, one per file.
        """
        if not self._initialized:
            raise RuntimeError("TestRunner not initialized. Call initialize() first.")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run_one(test_file: str) -> TestSuiteResult:
            async with semaphore:
                return await self.run_tests(test_file)

        tasks = [_run_one(tf) for tf in test_files]
        return await asyncio.gather(*tasks)

    # ── Coverage ──────────────────────────────────────────────────────────

    async def run_coverage(
        self,
        test_file_path: str,
    ) -> Optional[CoverageReport]:
        """
        Run tests with coverage measurement.

        Requires the ``coverage`` package to be installed. Falls back
        gracefully if unavailable.

        Parameters
        ----------
        test_file_path:
            Path to the test file.

        Returns
        -------
        CoverageReport if coverage.py is available, else None.
        """
        if not self._initialized:
            raise RuntimeError("TestRunner not initialized. Call initialize() first.")

        test_path = Path(test_file_path)
        if not test_path.is_absolute():
            test_path = self.project_root / test_path
        if not test_path.exists():
            return None

        # Try using coverage module
        command = [
            self.python_path,
            "-m",
            "coverage",
            "run",
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "--tb=short",
            "--no-header",
        ]

        try:
            stdout, stderr, exit_code, duration = await self._run_subprocess(
                command, timeout=120
            )
        except subprocess.TimeoutExpired:
            logger.warning("Coverage run timed out for %s", test_file_path)
            return None
        except FileNotFoundError:
            logger.info("coverage module not installed — skipping coverage")
            return None

        # Try to get coverage report as JSON
        report_command = [
            self.python_path,
            "-m",
            "coverage",
            "report",
            "--format=json",
            "--show-missing",
        ]

        try:
            report_stdout, _, _, _ = await self._run_subprocess(
                report_command, timeout=30
            )
            coverage_data = json.loads(report_stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Could not parse coverage JSON: %s", e)
            # Fall back to text parsing
            text_command = [
                self.python_path, "-m", "coverage", "report",
            ]
            try:
                text_stdout, _, _, _ = await self._run_subprocess(text_command, timeout=30)
                return self._parse_coverage_text(text_stdout)
            except Exception:
                return None

        # Parse JSON coverage
        report = CoverageReport(
            total_files=coverage_data.get("num_statements", 0),
            covered_files=0,
            total_lines=coverage_data.get("num_statements", 0),
            covered_lines=coverage_data.get("covered_lines", 0),
            line_coverage=coverage_data.get("percent_covered", 0.0) / 100.0,
            branch_coverage=0.0,
            file_reports=coverage_data.get("files", {}),
        )

        # Count covered files
        files = coverage_data.get("files", {})
        report.covered_files = sum(
            1 for f in files.values()
            if f.get("summary", {}).get("percent_covered", 0) > 0
        )
        report.file_reports = [
            {
                "path": path,
                "lines": info.get("summary", {}).get("num_statements", 0),
                "covered": info.get("summary", {}).get("covered_lines", 0),
                "percent": info.get("summary", {}).get("percent_covered", 0.0),
            }
            for path, info in files.items()
        ]

        # Persist
        await self._persist_coverage(report, str(test_file_path))
        return report

    # ── History ───────────────────────────────────────────────────────────

    async def get_history(self, limit: int = 50) -> List[TestSuiteResult]:
        """
        Retrieve past test results from the database.

        Parameters
        ----------
        limit:
            Maximum number of results to return.

        Returns
        -------
        List of historical TestSuiteResult objects, newest first.
        """
        if not self._initialized:
            raise RuntimeError("TestRunner not initialized. Call initialize() first.")

        def _query() -> List[TestSuiteResult]:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT * FROM test_suite_results ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            results: List[TestSuiteResult] = []
            for row in rows:
                # Load individual test results
                test_results = conn.execute(
                    "SELECT * FROM test_results WHERE suite_id = ? ORDER BY test_name",
                    (row["id"],),
                ).fetchall()

                suite = TestSuiteResult(
                    id=row["id"],
                    test_file=row["test_file"],
                    total=row["total"],
                    passed=row["passed"],
                    failed=row["failed"],
                    skipped=row["skipped"],
                    errors=row["errors"],
                    duration=row["duration"],
                    timestamp=row["timestamp"],
                    results=[
                        TestResult(
                            id=tr["id"],
                            test_name=tr["test_name"],
                            test_file=tr["test_file"],
                            status=tr["status"],
                            duration=tr["duration"],
                            error_message=tr["error_message"],
                            traceback=tr["traceback"],
                            timestamp=tr["timestamp"],
                        )
                        for tr in test_results
                    ],
                )
                results.append(suite)
            return results

        return await self._run_sync(_query)

    async def get_test_trends(self, test_file: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get pass-rate trends for a specific test file.

        Parameters
        ----------
        test_file:
            The test file to query.
        limit:
            Maximum number of historical runs to analyze.

        Returns
        -------
        Dictionary with trend data including pass rates over time.
        """
        if not self._initialized:
            raise RuntimeError("TestRunner not initialized. Call initialize() first.")

        def _query() -> Dict[str, Any]:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT * FROM test_suite_results WHERE test_file = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (test_file, limit),
            ).fetchall()

            if not rows:
                return {
                    "test_file": test_file,
                    "total_runs": 0,
                    "avg_pass_rate": 0.0,
                    "latest_pass_rate": 0.0,
                    "trend": "no_data",
                    "history": [],
                }

            pass_rates = [row["pass_rate"] for row in rows]
            avg_rate = sum(pass_rates) / len(pass_rates)

            # Determine trend
            if len(pass_rates) >= 3:
                recent = pass_rates[:len(pass_rates) // 2]
                older = pass_rates[len(pass_rates) // 2:]
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                diff = recent_avg - older_avg
                if diff > 0.05:
                    trend = "improving"
                elif diff < -0.05:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"

            return {
                "test_file": test_file,
                "total_runs": len(rows),
                "avg_pass_rate": round(avg_rate, 4),
                "latest_pass_rate": round(pass_rates[0], 4),
                "trend": trend,
                "history": [
                    {
                        "timestamp": row["timestamp"],
                        "pass_rate": row["pass_rate"],
                        "total": row["total"],
                        "passed": row["passed"],
                        "failed": row["failed"],
                    }
                    for row in rows
                ],
            }

        return await self._run_sync(_query)

    # ── Subprocess Execution ──────────────────────────────────────────────

    async def _run_subprocess(
        self,
        command: List[str],
        timeout: int = 60,
    ) -> Tuple[str, str, int, float]:
        """
        Execute a command as a subprocess and capture output.

        Returns
        -------
        Tuple of (stdout, stderr, exit_code, duration_seconds).
        """
        start = time.monotonic()

        def _execute() -> Tuple[str, str, int, float]:
            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(self.project_root),
                    env={
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "0",
                    },
                )
                elapsed = time.monotonic() - start
                return (
                    proc.stdout or "",
                    proc.stderr or "",
                    proc.returncode,
                    elapsed,
                )
            except subprocess.TimeoutExpired as e:
                elapsed = time.monotonic() - start
                return (
                    e.stdout or "" if isinstance(e.stdout, str) else "",
                    e.stderr or "" if isinstance(e.stderr, str) else "",
                    -1,
                    elapsed,
                )

        return await self._run_sync(_execute)

    # ── Output Parsing ────────────────────────────────────────────────────

    def _parse_pytest_output(self, output: str, exit_code: int) -> List[TestResult]:
        """
        Parse pytest verbose output into structured TestResult objects.

        Handles the standard pytest -v output format:
            tests/test_foo.py::test_bar PASSED                           [ 50%]
            tests/test_foo.py::test_baz FAILED                           [100%]
        """
        results: List[TestResult] = []
        now = datetime.now(timezone.utc).isoformat()

        # Pattern for pytest verbose output lines
        # Matches: path/to/test_file.py::TestClass::test_method STATUS
        pattern = re.compile(
            r"^([\w/.\\:-]+)::([\w.-]+)(?:::(test_\w+))?\s+"
            r"(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\s",
            re.MULTILINE,
        )

        for match in pattern.finditer(output):
            file_path = match.group(1)
            class_name = match.group(2) or ""
            test_name = match.group(3) or match.group(2) or ""
            status_raw = match.group(4)

            # Map pytest status to our status
            if status_raw == "PASSED":
                status = STATUS_PASSED
            elif status_raw == "FAILED":
                status = STATUS_FAILED
            elif status_raw == "SKIPPED":
                status = STATUS_SKIPPED
            elif status_raw == "ERROR":
                status = STATUS_ERROR
            elif status_raw == "XFAIL":
                status = STATUS_SKIPPED  # expected failure
            elif status_raw == "XPASS":
                status = STATUS_PASSED  # unexpected pass — still passed
            else:
                status = STATUS_ERROR

            full_name = f"{test_name}" if not class_name.startswith("test_") else test_name
            if class_name and class_name.startswith("test_"):
                full_name = class_name
            elif class_name and test_name:
                full_name = f"{class_name}::{test_name}"

            results.append(TestResult(
                id=str(uuid.uuid4())[:12],
                test_name=full_name,
                test_file=file_path,
                status=status,
                duration=0.0,
                timestamp=now,
            ))

        # If no tests were parsed but output exists, create a synthetic result
        if not results and output.strip():
            if exit_code == PYTEST_EXIT_NO_TESTS:
                results.append(TestResult(
                    id=str(uuid.uuid4())[:12],
                    test_name="[no tests collected]",
                    test_file="",
                    status=STATUS_SKIPPED,
                    error_message="No tests were collected",
                    timestamp=now,
                ))
            elif exit_code != PYTEST_EXIT_OK:
                results.append(TestResult(
                    id=str(uuid.uuid4())[:12],
                    test_name="[collection error]",
                    test_file="",
                    status=STATUS_ERROR,
                    error_message=output.strip()[-500:] if len(output) > 500 else output.strip(),
                    timestamp=now,
                ))

        return results

    def _parse_coverage_text(self, output: str) -> Optional[CoverageReport]:
        """Parse text coverage output when JSON is unavailable."""
        # Look for the summary line: "TOTAL   NNN NNN N%"
        total_match = re.search(
            r"(TOTAL|total)\s+(\d+)\s+(\d+)\s+(\d+)%?", output
        )
        if not total_match:
            return None

        total_lines = int(total_match.group(2))
        covered_lines = int(total_match.group(3))
        pct = int(total_match.group(4))

        # Parse per-file lines
        file_reports: List[Dict[str, Any]] = []
        file_pattern = re.compile(r"^([\w/.\\:-]+\.py)\s+(\d+)\s+(\d+)\s+(\d+)%?", re.MULTILINE)
        for match in file_pattern.finditer(output):
            if match.group(1) in ("TOTAL", "total"):
                continue
            file_reports.append({
                "path": match.group(1),
                "lines": int(match.group(2)),
                "covered": int(match.group(3)),
                "percent": float(match.group(4)),
            })

        return CoverageReport(
            total_files=len(file_reports),
            covered_files=sum(1 for f in file_reports if f["percent"] > 0),
            total_lines=total_lines,
            covered_lines=covered_lines,
            line_coverage=pct / 100.0,
            file_reports=file_reports,
        )

    # ── Persistence ───────────────────────────────────────────────────────

    async def _persist_result(
        self,
        suite: TestSuiteResult,
        raw_output: str,
    ) -> None:
        """Store a test suite result and its individual test results."""

        def _store() -> None:
            conn = self._ensure_conn()
            conn.execute(
                """INSERT OR REPLACE INTO test_suite_results
                   (id, test_file, total, passed, failed, skipped, errors,
                    duration, pass_rate, timestamp, raw_output)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    suite.id, suite.test_file, suite.total, suite.passed,
                    suite.failed, suite.skipped, suite.errors, suite.duration,
                    suite.pass_rate(), suite.timestamp,
                    raw_output[-2000:] if len(raw_output) > 2000 else raw_output,
                ),
            )
            for r in suite.results:
                conn.execute(
                    """INSERT OR REPLACE INTO test_results
                       (id, suite_id, test_name, test_file, status, duration,
                        error_message, traceback, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        r.id, suite.id, r.test_name, r.test_file, r.status,
                        r.duration, r.error_message, r.traceback, r.timestamp,
                    ),
                )
            conn.commit()

        await self._run_sync(_store)

    async def _persist_coverage(
        self,
        report: CoverageReport,
        test_file: str,
    ) -> None:
        """Store a coverage report."""

        def _store() -> None:
            conn = self._ensure_conn()
            conn.execute(
                """INSERT INTO coverage_reports
                   (id, suite_id, total_files, covered_files, total_lines,
                    covered_lines, line_coverage, branch_coverage, timestamp,
                    raw_output)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4())[:12], "", report.total_files,
                    report.covered_files, report.total_lines, report.covered_lines,
                    report.line_coverage, report.branch_coverage,
                    datetime.now(timezone.utc).isoformat(), "",
                ),
            )
            conn.commit()

        await self._run_sync(_store)
