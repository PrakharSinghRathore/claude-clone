"""
Test Reporter — Generates human-readable and machine-readable test reports.

Produces:
- Console summaries with color-coded pass/fail/skip
- Markdown reports for documentation
- JSON reports for CI/CD integration
- Trend analysis comparing runs over time
- Recommendations for improving coverage

Usage:
    reporter = TestReporter(project_root="/path/to/project")
    report = reporter.generate_report(suite_results)
    print(report)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# ANSI color codes for console output
_COLOR_RESET = "\033[0m"
_COLOR_BOLD = "\033[1m"
_COLOR_RED = "\033[31m"
_COLOR_GREEN = "\033[32m"
_COLOR_YELLOW = "\033[33m"
_COLOR_BLUE = "\033[34m"
_COLOR_MAGENTA = "\033[35m"
_COLOR_CYAN = "\033[36m"
_COLOR_DIM = "\033[2m"

# Trend thresholds
_TREND_IMPROVING_THRESHOLD = 0.05
_TREND_DECLINING_THRESHOLD = -0.05


# ──────────────────────────────────────────────────────────────────────────────
# TestReporter
# ──────────────────────────────────────────────────────────────────────────────

class TestReporter:
    """
    Generates formatted test reports from TestSuiteResult and CoverageReport data.

    Supports multiple output formats:
    - Console: Colorized summary for terminal display
    - Markdown: Structured report for documentation / PR comments
    - JSON: Machine-readable report for CI/CD pipelines
    - Trend: Historical analysis comparing multiple runs
    """

    def __init__(self, project_root: str) -> None:
        """
        Parameters
        ----------
        project_root:
            Absolute or relative path to the project root directory.
        """
        self.project_root = project_root

    # ── Console Report ────────────────────────────────────────────────────

    def generate_console_report(
        self,
        results: List[Any],
    ) -> str:
        """
        Generate a colorized console summary for test results.

        Parameters
        ----------
        results:
            List of TestSuiteResult objects.

        Returns
        -------
            Colorized console report string.
        """
        if not results:
            return f"{_COLOR_DIM}No test results to report.{_COLOR_RESET}"

        lines: List[str] = []
        lines.append("")
        lines.append(f"{_COLOR_BOLD}{'═' * 60}")
        lines.append(f"  TEST REPORT — {len(results)} suite(s)")
        lines.append(f"{'═' * 60}{_COLOR_RESET}")
        lines.append("")

        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0
        total_tests = 0
        total_duration = 0.0

        for suite in results:
            status = suite.status if hasattr(suite, "status") else ""
            if suite.failed > 0 or suite.errors > 0:
                icon = f"{_COLOR_RED}✗ FAILED{_COLOR_RESET}"
            elif suite.total == 0:
                icon = f"{_COLOR_YELLOW}⚠ NO TESTS{_COLOR_RESET}"
            else:
                icon = f"{_COLOR_GREEN}✓ PASSED{_COLOR_RESET}"

            lines.append(f"  {icon}  {_COLOR_BOLD}{suite.test_file}{_COLOR_RESET}")
            lines.append(
                f"         {suite.passed} passed · {suite.failed} failed"
                f" · {suite.skipped} skipped · {suite.errors} errors"
                f"  {_COLOR_DIM}({self._format_duration(suite.duration)}){_COLOR_RESET}"
            )

            # Show individual failures
            for r in suite.results:
                if r.status == "failed":
                    lines.append(
                        f"         {_COLOR_RED}  ✗ {r.test_name}{_COLOR_RESET}"
                    )
                    if r.error_message:
                        msg = r.error_message.split("\n")[0][:80]
                        lines.append(f"           {_COLOR_DIM}{msg}{_COLOR_RESET}")
                elif r.status == "error":
                    lines.append(
                        f"         {_COLOR_MAGENTA}  ! {r.test_name}{_COLOR_RESET}"
                    )

            lines.append("")

            total_passed += suite.passed
            total_failed += suite.failed
            total_skipped += suite.skipped
            total_errors += suite.errors
            total_tests += suite.total
            total_duration += suite.duration

        # Summary line
        if total_tests > 0:
            pass_rate = total_passed / total_tests
            if pass_rate == 1.0:
                rate_color = _COLOR_GREEN
            elif pass_rate >= 0.8:
                rate_color = _COLOR_YELLOW
            else:
                rate_color = _COLOR_RED

            lines.append(f"{_COLOR_BOLD}{'─' * 60}{_COLOR_RESET}")
            lines.append(
                f"  Total: {total_tests} tests  "
                f"{_COLOR_GREEN}{total_passed} passed{_COLOR_RESET}  "
                f"{_COLOR_RED}{total_failed} failed{_COLOR_RESET}  "
                f"{_COLOR_YELLOW}{total_skipped} skipped{_COLOR_RESET}  "
                f"{_COLOR_MAGENTA}{total_errors} errors{_COLOR_RESET}"
            )
            lines.append(
                f"  Pass rate: {rate_color}{pass_rate:.1%}{_COLOR_RESET}  "
                f"{_COLOR_DIM}({self._format_duration(total_duration)}){_COLOR_RESET}"
            )
            lines.append(f"{_COLOR_BOLD}{'═' * 60}{_COLOR_RESET}")
        else:
            lines.append(f"{_COLOR_YELLOW}  No tests were executed.{_COLOR_RESET}")

        lines.append("")
        return "\n".join(lines)

    # ── Markdown Report ───────────────────────────────────────────────────

    def generate_markdown_report(
        self,
        results: List[Any],
        coverage: Optional[Any] = None,
    ) -> str:
        """
        Generate a Markdown-formatted test report.

        Parameters
        ----------
        results:
            List of TestSuiteResult objects.
        coverage:
            Optional CoverageReport for coverage data.

        Returns
        -------
        Markdown report string.
        """
        if not results:
            return "*No test results to report.*"

        lines: List[str] = []
        lines.append("# Test Report")
        lines.append("")

        # Summary table
        total_passed = sum(s.passed for s in results)
        total_failed = sum(s.failed for s in results)
        total_skipped = sum(s.skipped for s in results)
        total_errors = sum(s.errors for s in results)
        total_tests = sum(s.total for s in results)
        total_duration = sum(s.duration for s in results)
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        if total_tests > 0:
            status_badge = (
                f"✅ **{pass_rate:.1f}% pass rate**"
                if pass_rate == 100
                else f"⚠️ **{pass_rate:.1f}% pass rate**"
                if pass_rate >= 80
                else f"❌ **{pass_rate:.1f}% pass rate**"
            )
        else:
            status_badge = "⚠️ *No tests collected*"

        lines.append(f"## Summary {status_badge}")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total tests | {total_tests} |")
        lines.append(f"| Passed | ✅ {total_passed} |")
        lines.append(f"| Failed | ❌ {total_failed} |")
        lines.append(f"| Skipped | ⏭️ {total_skipped} |")
        lines.append(f"| Errors | 💥 {total_errors} |")
        lines.append(f"| Duration | {self._format_duration(total_duration)} |")
        lines.append(f"| Suites | {len(results)} |")
        lines.append("")

        # Coverage section
        if coverage:
            lines.append("## Coverage")
            lines.append("")
            lines.append(f"**Line coverage:** {coverage.line_coverage:.1%}")
            lines.append(f"**Files covered:** {coverage.covered_files}/{coverage.total_files}")
            if coverage.branch_coverage > 0:
                lines.append(f"**Branch coverage:** {coverage.branch_coverage:.1%}")
            lines.append("")

            if coverage.file_reports:
                lines.append("| File | Lines | Covered | % |")
                lines.append("|------|-------|---------|---|")
                for fr in coverage.file_reports:
                    pct = fr.get("percent", 0)
                    if isinstance(pct, (int, float)):
                        pct_str = f"{pct:.0f}%"
                    else:
                        pct_str = str(pct)
                    lines.append(
                        f"| `{fr.get('path', '?')}` | {fr.get('lines', '?')} "
                        f"| {fr.get('covered', '?')} | {pct_str} |"
                    )
                lines.append("")

        # Per-suite results
        lines.append("## Suite Details")
        lines.append("")

        for suite in results:
            suite_status = "✅" if suite.failed == 0 and suite.errors == 0 else "❌"
            lines.append(f"### {suite_status} `{suite.test_file}`")
            lines.append("")
            lines.append(f"- **Passed:** {suite.passed}/{suite.total}")
            lines.append(f"- **Failed:** {suite.failed}")
            lines.append(f"- **Skipped:** {suite.skipped}")
            lines.append(f"- **Errors:** {suite.errors}")
            lines.append(f"- **Duration:** {self._format_duration(suite.duration)}")
            lines.append("")

            if suite.failed > 0:
                failed_tests = [r for r in suite.results if r.status == "failed"]
                if failed_tests:
                    lines.append("**Failed tests:**")
                    lines.append("")
                    for r in failed_tests:
                        lines.append(f"- `{r.test_name}`")
                        if r.error_message:
                            lines.append(f"  ```")
                            lines.append(f"  {r.error_message[:300]}")
                            lines.append(f"  ```")
                    lines.append("")

        return "\n".join(lines)

    # ── JSON Report ───────────────────────────────────────────────────────

    def generate_json_report(
        self,
        results: List[Any],
        coverage: Optional[Any] = None,
    ) -> dict:
        """
        Generate a machine-readable JSON report.

        Parameters
        ----------
        results:
            List of TestSuiteResult objects.
        coverage:
            Optional CoverageReport for coverage data.

        Returns
        -------
        Dictionary suitable for ``json.dumps()``.
        """
        total_passed = sum(s.passed for s in results)
        total_failed = sum(s.failed for s in results)
        total_skipped = sum(s.skipped for s in results)
        total_errors = sum(s.errors for s in results)
        total_tests = sum(s.total for s in results)
        total_duration = sum(s.duration for s in results)
        pass_rate = (total_passed / total_tests) if total_tests > 0 else 0.0

        report: Dict[str, Any] = {
            "summary": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_skipped": total_skipped,
                "total_errors": total_errors,
                "pass_rate": round(pass_rate, 4),
                "total_duration": round(total_duration, 3),
                "total_suites": len(results),
            },
            "suites": [s.to_dict() for s in results if hasattr(s, "to_dict")],
        }

        if coverage:
            report["coverage"] = (
                coverage.to_dict() if hasattr(coverage, "to_dict") else {}
            )

        return report

    # ── Trend Report ──────────────────────────────────────────────────────

    def generate_trend_report(self, history: List[Any]) -> str:
        """
        Generate a trend analysis report comparing multiple runs.

        Parameters
        ----------
        history:
            List of TestSuiteResult objects ordered chronologically
            (newest first).

        Returns
        -------
        Human-readable trend analysis string.
        """
        if not history:
            return "No historical data available for trend analysis."

        # Group by test file
        file_history: Dict[str, List[Any]] = {}
        for suite in history:
            key = suite.test_file
            if key not in file_history:
                file_history[key] = []
            file_history[key].append(suite)

        lines: List[str] = []
        lines.append("")
        lines.append(f"{_COLOR_BOLD}{'═' * 60}")
        lines.append(f"  TREND ANALYSIS — {len(history)} run(s), {len(file_history)} file(s)")
        lines.append(f"{'═' * 60}{_COLOR_RESET}")
        lines.append("")

        for test_file, runs in sorted(file_history.items()):
            # Runs are newest first; reverse for chronological order
            runs_chrono = list(reversed(runs))
            pass_rates = [
                (r.passed / r.total) if r.total > 0 else 0.0 for r in runs_chrono
            ]

            trend = self._calculate_trend(pass_rates)

            if trend == "improving":
                trend_icon = f"{_COLOR_GREEN}↑{_COLOR_RESET}"
            elif trend == "declining":
                trend_icon = f"{_COLOR_RED}↓{_COLOR_RESET}"
            else:
                trend_icon = f"{_COLOR_YELLOW}→{_COLOR_RESET}"

            avg_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
            latest = pass_rates[-1] if pass_rates else 0.0

            lines.append(
                f"  {trend_icon} {_COLOR_BOLD}{test_file}{_COLOR_RESET}"
            )
            lines.append(
                f"       Trend: {trend} | "
                f"Avg: {avg_rate:.1%} | "
                f"Latest: {latest:.1%} | "
                f"Runs: {len(runs)}"
            )

            # Visual bar chart of last 10 runs
            recent = pass_rates[-10:]
            bar_parts = []
            for rate in recent:
                filled = int(rate * 20)
                bar = "█" * filled + "░" * (20 - filled)
                if rate >= 0.9:
                    bar_color = _COLOR_GREEN
                elif rate >= 0.7:
                    bar_color = _COLOR_YELLOW
                else:
                    bar_color = _COLOR_RED
                bar_parts.append(f"{bar_color}{bar}{_COLOR_RESET}")

            lines.append(f"       {''.join(bar_parts)}")
            lines.append("")

        lines.append(f"{_COLOR_DIM}  Each bar represents one run (latest 10). "
                      f"█ = passed, ░ = failed.{_COLOR_RESET}")
        lines.append("")

        return "\n".join(lines)

    # ── Recommendations ───────────────────────────────────────────────────

    def generate_recommendations(
        self,
        results: List[Any],
        coverage: Optional[Any] = None,
    ) -> List[str]:
        """
        Generate actionable recommendations for improving tests and coverage.

        Parameters
        ----------
        results:
            List of TestSuiteResult objects.
        coverage:
            Optional CoverageReport for coverage data.

        Returns
        -------
        List of recommendation strings.
        """
        recommendations: List[str] = []

        total_passed = sum(s.passed for s in results)
        total_failed = sum(s.failed for s in results)
        total_skipped = sum(s.skipped for s in results)
        total_tests = sum(s.total for s in results)
        total_errors = sum(s.errors for s in results)

        # Overall pass rate
        if total_tests > 0:
            pass_rate = total_passed / total_tests
            if pass_rate < 0.5:
                recommendations.append(
                    "🔴 CRITICAL: Overall pass rate is below 50%. "
                    "Prioritize fixing failing tests before adding new ones."
                )
            elif pass_rate < 0.8:
                recommendations.append(
                    "🟡 WARNING: Overall pass rate is below 80%. "
                    "Consider focusing on stabilizing existing tests."
                )
            elif pass_rate == 1.0:
                recommendations.append(
                    "🟢 EXCELLENT: All tests pass! Consider adding edge cases "
                    "to increase coverage."
                )

        # No tests
        if total_tests == 0:
            recommendations.append(
                "⚠️ No tests were executed. Ensure test files exist and "
                "are discoverable by pytest."
            )

        # Errors
        if total_errors > 0:
            recommendations.append(
                f"💥 {total_errors} test(s) encountered errors (not test failures "
                "but collection/fixture errors). Check import paths and fixtures."
            )

        # Skipped tests
        if total_tests > 0 and total_skipped > 0:
            skip_rate = total_skipped / total_tests
            if skip_rate > 0.2:
                recommendations.append(
                    f"⏭️ {total_skipped} tests ({skip_rate:.1%}) were skipped. "
                    "Review skip conditions — high skip rates mask real issues."
                )
            elif skip_rate > 0:
                recommendations.append(
                    f"ℹ️ {total_skipped} test(s) were skipped. "
                    "Verify skip conditions are still relevant."
                )

        # Coverage recommendations
        if coverage:
            if coverage.line_coverage < 0.3:
                recommendations.append(
                    f"📊 Coverage is very low ({coverage.line_coverage:.1%}). "
                    "Focus on testing core business logic paths first."
                )
            elif coverage.line_coverage < 0.6:
                recommendations.append(
                    f"📊 Coverage is below 60% ({coverage.line_coverage:.1%}). "
                    "Add tests for uncovered functions and error paths."
                )
            elif coverage.line_coverage < 0.8:
                recommendations.append(
                    f"📊 Coverage is decent ({coverage.line_coverage:.1%}). "
                    "Target edge cases and error handling to reach 80%+."
                )
            elif coverage.line_coverage >= 0.8:
                recommendations.append(
                    f"📊 Coverage is good ({coverage.line_coverage:.1%})! "
                    "Focus on mutation testing or property-based tests for "
                    "additional confidence."
                )

            # File-level coverage gaps
            if coverage.file_reports:
                low_coverage_files = [
                    fr for fr in coverage.file_reports
                    if isinstance(fr.get("percent"), (int, float))
                    and fr["percent"] < 50
                ]
                if low_coverage_files:
                    file_names = [fr.get("path", "?") for fr in low_coverage_files[:5]]
                    recommendations.append(
                        f"📁 Low coverage files: {', '.join(file_names)}. "
                        "Consider adding targeted tests for these modules."
                    )

        # Slow tests
        slow_suites = [
            s for s in results if s.duration > 10.0
        ]
        if slow_suites:
            slow_names = [s.test_file for s in slow_suites[:3]]
            recommendations.append(
                f"🐢 Slow test suites: {', '.join(slow_names)}. "
                "Consider optimizing setup/teardown or running in parallel."
            )

        # Specific failing test patterns
        for suite in results:
            if suite.failed > 0:
                failed_names = [
                    r.test_name for r in suite.results if r.status == "failed"
                ]
                if failed_names:
                    recommendations.append(
                        f"❌ Fix failing tests in {suite.test_file}: "
                        f"{', '.join(failed_names[:5])}"
                    )

        if not recommendations:
            recommendations.append(
                "✅ Everything looks good! Keep writing tests."
            )

        return recommendations

    # ── Helper Methods ────────────────────────────────────────────────────

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a duration in seconds to a human-readable string."""
        if seconds < 0.001:
            return "< 1ms"
        if seconds < 1.0:
            return f"{seconds * 1000:.0f}ms"
        if seconds < 60.0:
            return f"{seconds:.2f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"

    @staticmethod
    def _calculate_trend(values: List[float]) -> str:
        """
        Analyze a sequence of pass rates and determine the trend.

        Returns
        -------
        One of: "improving", "stable", "declining", "no_data".
        """
        if len(values) < 2:
            return "no_data"

        # Split into first half and second half
        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]

        first_avg = sum(first_half) / len(first_half) if first_half else 0.0
        second_avg = sum(second_half) / len(second_half) if second_half else 0.0
        diff = second_avg - first_avg

        if diff > _TREND_IMPROVING_THRESHOLD:
            return "improving"
        elif diff < _TREND_DECLINING_THRESHOLD:
            return "declining"
        return "stable"
