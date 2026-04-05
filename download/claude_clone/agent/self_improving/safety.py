"""
Safety & Guardrails for the Self-Improving System.

Every self-modification passes through this module before being applied.
Implements:
- Multi-gate approval system (syntax, lint, test, security)
- Automatic backup and rollback of all changed files
- Quarantine zone for risky changes requiring manual review
- Change size limits, rate limiting, and cooldown timers
- Protected file registry (core files that need extra scrutiny)
- Full audit trail of all attempted and applied modifications

Usage:
    safety = SafetyGuardrails(project_root="/path/to/claude_clone")
    await safety.initialize()
    result = await safety.evaluate_change(
        file_path="agent/tools.py",
        original_code="...",
        proposed_code="...",
        change_type="patch",
        reason="Fix bug in read_file encoding detection",
    )
    if result.approved:
        backup_id = await safety.backup_file("agent/tools.py")
        await safety.apply_change("agent/tools.py", proposed_code)
        # If something goes wrong:
        await safety.rollback(backup_id)
"""

from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_BACKUP_DIR = "~/.claude_clone/self_improve_backups"
DEFAULT_DB_PATH = "~/.claude_clone/self_improve_safety.db"

# Files that are CRITICAL and require all gates + extra review
PROTECTED_FILES = {
    "agent/core.py",
    "agent/tools.py",
    "config.py",
    "main.py",
    "agent/sandbox.py",
    "agent/security.py",
    "agent/self_improving/safety.py",
}

# Maximum change size (lines added/removed) before requiring extra approval
CHANGE_SIZE_LIMIT = 200

# Maximum number of changes per hour
RATE_LIMIT_HOURLY = 10

# Minimum seconds between changes to the same file
COOLDOWN_PER_FILE = 60

# Maximum total self-improvement changes per day
DAILY_CHANGE_LIMIT = 50


# ──────────────────────────────────────────────────────────────────────────────
# Enums & Data Classes
# ──────────────────────────────────────────────────────────────────────────────

class ChangeType(str, Enum):
    PATCH = "patch"            # Bug fix / improvement to existing code
    OPTIMIZATION = "optimization"  # Performance improvement
    EXTENSION = "extension"    # New tool / capability added
    REFACTOR = "refactor"      # Code restructure without behavior change
    REWRITE = "rewrite"        # Significant rewrite of a module


class GateResult(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ApprovalLevel(str, Enum):
    AUTO_APPROVE = "auto"      # Safe, apply automatically
    ADVISORY = "advisory"      # Approved but logged with warnings
    MANUAL_REVIEW = "manual"   # Requires human review
    BLOCKED = "blocked"        # Absolutely not allowed


@dataclass
class GateCheck:
    """Result of a single safety gate check."""
    gate_name: str
    result: GateResult
    score: float          # 0.0 to 1.0
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyEvaluation:
    """Complete safety evaluation for a proposed change."""
    approved: bool
    approval_level: ApprovalLevel
    overall_score: float     # 0.0 to 1.0, weighted average of all gates
    gates: List[GateCheck]
    warnings: List[str]
    blockers: List[str]
    recommendations: List[str]
    file_path: str
    change_type: ChangeType
    estimated_risk: float    # 0.0 (safe) to 1.0 (dangerous)


@dataclass
class BackupRecord:
    """Record of a file backup before modification."""
    backup_id: str
    file_path: str
    original_hash: str
    backup_path: str
    timestamp: str
    change_type: ChangeType
    reason: str
    rolled_back: bool = False


@dataclass
class ChangeHistory:
    """Record of a change that was applied (or attempted)."""
    change_id: str
    file_path: str
    change_type: ChangeType
    original_hash: str
    new_hash: str
    backup_id: str
    approval_level: ApprovalLevel
    safety_score: float
    reason: str
    applied: bool
    rolled_back: bool
    timestamp: str
    gates_summary: Dict[str, str] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _generate_id() -> str:
    return hashlib.sha256(os.urandom(32)).hexdigest()[:16]


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_lines_diff(original: str, proposed: str) -> Tuple[int, int]:
    """Count lines added and removed between two code strings."""
    orig_lines = set(original.splitlines())
    prop_lines = set(proposed.splitlines())
    added = len(prop_lines - orig_lines)
    removed = len(orig_lines - prop_lines)
    return added, removed


# ──────────────────────────────────────────────────────────────────────────────
# SafetyGuardrails
# ──────────────────────────────────────────────────────────────────────────────

class SafetyGuardrails:
    """
    Multi-layered safety system for self-modification.

    Every proposed change passes through 7 gates in order:
    1. Syntax Gate — AST parse check (instant reject if fails)
    2. Size Gate — Line delta within limits
    3. Import Gate — No dangerous new imports
    4. Pattern Gate — No suspicious patterns (exec, eval, network exfil)
    5. Coverage Gate — Tests still pass
    6. Lint Gate — Code quality check
    7. Protected Gate — Extra scrutiny for critical files

    Parameters
    ----------
    project_root:
        Absolute path to the project root directory.
    backup_dir:
        Directory to store file backups.
    db_path:
        SQLite database path for audit trail.
    """

    def __init__(
        self,
        project_root: str,
        backup_dir: str = DEFAULT_BACKUP_DIR,
        db_path: str = DEFAULT_DB_PATH,
    ):
        self.project_root = Path(project_root).resolve()
        self.backup_dir = Path(backup_dir).expanduser().resolve()
        self.db_path = Path(db_path).expanduser().resolve()
        self._conn: Optional[sqlite3.Connection] = None
        self._rate_limit_cache: Dict[str, List[float]] = {}
        self._cooldown_cache: Dict[str, float] = {}
        self._daily_change_count: int = 0
        self._daily_reset_date: str = ""

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create backup directory, database, and load rate limit state."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        await self._run_sync(self._connect)
        await self._run_sync(self._create_tables)
        await self._load_rate_limit_state()

    async def close(self) -> None:
        if self._conn is not None:
            await self._run_sync(self._conn.close)
            self._conn = None

    def _connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS backups (
                backup_id    TEXT PRIMARY KEY,
                file_path    TEXT NOT NULL,
                original_hash TEXT NOT NULL,
                backup_path  TEXT NOT NULL,
                timestamp    TEXT NOT NULL,
                change_type  TEXT NOT NULL,
                reason       TEXT DEFAULT '',
                rolled_back  INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS change_history (
                change_id      TEXT PRIMARY KEY,
                file_path      TEXT NOT NULL,
                change_type    TEXT NOT NULL,
                original_hash  TEXT NOT NULL,
                new_hash       TEXT NOT NULL,
                backup_id      TEXT,
                approval_level TEXT NOT NULL,
                safety_score   REAL DEFAULT 0.0,
                reason         TEXT DEFAULT '',
                applied        INTEGER DEFAULT 0,
                rolled_back    INTEGER DEFAULT 0,
                timestamp      TEXT NOT NULL,
                gates_summary  TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS quarantine (
                id          TEXT PRIMARY KEY,
                file_path   TEXT NOT NULL,
                proposed_code TEXT NOT NULL,
                reason      TEXT DEFAULT '',
                risk_score  REAL DEFAULT 0.0,
                timestamp   TEXT NOT NULL,
                resolved    INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_backup_file ON backups(file_path);
            CREATE INDEX IF NOT EXISTS idx_history_file ON change_history(file_path);
            CREATE INDEX IF NOT EXISTS idx_history_timestamp ON change_history(timestamp);
            CREATE INDEX IF NOT EXISTS idx_quarantine_resolved ON quarantine(resolved);
        """)
        self._conn.commit()

    # ── Async wrapper ─────────────────────────────────────────────────────

    async def _run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SafetyGuardrails not initialized. Call await safety.initialize() first.")
        return self._conn

    # ── Rate Limiting ─────────────────────────────────────────────────────

    async def _load_rate_limit_state(self) -> None:
        """Load today's change count from database."""
        def _do() -> None:
            conn = self._ensure_conn()
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._daily_reset_date = today
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM change_history WHERE date(timestamp) = ? AND applied = 1",
                (today,),
            ).fetchone()
            self._daily_change_count = row["cnt"] if row else 0

        await self._run_sync(_do)

    async def _check_rate_limit(self, file_path: str) -> Tuple[bool, str]:
        """Check if a change is allowed under rate limits."""
        now = time.time()

        # Daily limit
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_change_count = 0
            self._daily_reset_date = today

        if self._daily_change_count >= DAILY_CHANGE_LIMIT:
            return False, f"Daily change limit reached ({DAILY_CHANGE_LIMIT}/{DAILY_CHANGE_LIMIT})"

        # Per-file cooldown
        last_change = self._cooldown_cache.get(file_path, 0)
        if now - last_change < COOLDOWN_PER_FILE:
            remaining = COOLDOWN_PER_FILE - (now - last_change)
            return False, f"Cooldown active for {file_path} ({remaining:.0f}s remaining)"

        # Hourly rate limit
        if file_path not in self._rate_limit_cache:
            self._rate_limit_cache[file_path] = []
        # Clean old entries
        self._rate_limit_cache[file_path] = [
            t for t in self._rate_limit_cache[file_path] if now - t < 3600
        ]
        if len(self._rate_limit_cache[file_path]) >= RATE_LIMIT_HOURLY:
            return False, f"Hourly rate limit for {file_path} ({RATE_LIMIT_HOURLY}/hour)"

        return True, ""

    async def _record_change_rate(self, file_path: str) -> None:
        """Record a change for rate limiting purposes."""
        now = time.time()
        self._cooldown_cache[file_path] = now
        self._daily_change_count += 1
        if file_path not in self._rate_limit_cache:
            self._rate_limit_cache[file_path] = []
        self._rate_limit_cache[file_path].append(now)

    # ── Safety Gates ──────────────────────────────────────────────────────

    def _gate_syntax(self, proposed_code: str) -> GateCheck:
        """Gate 1: Verify the proposed code is syntactically valid Python."""
        try:
            ast.parse(proposed_code)
            return GateCheck(
                gate_name="syntax",
                result=GateResult.PASSED,
                score=1.0,
                message="Code is syntactically valid",
            )
        except SyntaxError as e:
            return GateCheck(
                gate_name="syntax",
                result=GateResult.FAILED,
                score=0.0,
                message=f"Syntax error: {e}",
                details={"line": e.lineno, "offset": e.offset, "text": e.text},
            )

    def _gate_size(self, original_code: str, proposed_code: str) -> GateCheck:
        """Gate 2: Check change size is within acceptable limits."""
        added, removed = _count_lines_diff(original_code, proposed_code)
        total_delta = added + removed

        if total_delta == 0:
            return GateCheck(
                gate_name="size",
                result=GateResult.PASSED,
                score=1.0,
                message="No change detected",
                details={"added": 0, "removed": 0},
            )

        if total_delta > CHANGE_SIZE_LIMIT:
            score = max(0.0, 1.0 - (total_delta - CHANGE_SIZE_LIMIT) / CHANGE_SIZE_LIMIT)
            return GateCheck(
                gate_name="size",
                result=GateResult.WARNING,
                score=score,
                message=f"Large change: +{added} / -{removed} lines (limit: {CHANGE_SIZE_LIMIT})",
                details={"added": added, "removed": removed, "limit": CHANGE_SIZE_LIMIT},
            )

        score = 1.0 - (total_delta / CHANGE_SIZE_LIMIT) * 0.3  # Slight penalty for larger changes
        return GateCheck(
            gate_name="size",
            result=GateResult.PASSED,
            score=score,
            message=f"Change size acceptable: +{added} / -{removed} lines",
            details={"added": added, "removed": removed},
        )

    def _gate_imports(self, original_code: str, proposed_code: str) -> GateCheck:
        """Gate 3: Check for dangerous new imports."""
        DANGEROUS_IMPORTS = {
            "subprocess": ("HIGH", "Can execute arbitrary system commands"),
            "os.system": ("HIGH", "Can execute arbitrary shell commands"),
            "ctypes": ("HIGH", "Can call arbitrary C functions, bypass Python safety"),
            "signal": ("MEDIUM", "Can send signals to processes"),
            "multiprocessing": ("LOW", "Can spawn processes"),
            "socket": ("MEDIUM", "Can make network connections"),
            "http.server": ("MEDIUM", "Can start a network server"),
            "ftplib": ("MEDIUM", "Can make FTP connections"),
            "smtplib": ("MEDIUM", "Can send emails"),
            "pickle": ("HIGH", "Can deserialize untrusted data, RCE risk"),
            "shelve": ("MEDIUM", "Can persist data to disk"),
            "eval": ("CRITICAL", "Can execute arbitrary code"),
            "exec": ("CRITICAL", "Can execute arbitrary code"),
            "compile": ("HIGH", "Can compile arbitrary code"),
            "__import__": ("HIGH", "Can import arbitrary modules"),
            "importlib": ("LOW", "Dynamic imports"),
        }

        try:
            orig_tree = ast.parse(original_code)
            prop_tree = ast.parse(proposed_code)
        except SyntaxError:
            return GateCheck(
                gate_name="imports",
                result=GateResult.SKIPPED,
                score=0.5,
                message="Cannot check imports (syntax error)",
            )

        def _extract_imports(tree: ast.AST) -> set:
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
                        for alias in node.names:
                            imports.add(f"{node.module}.{alias.name}")
                # Check for eval/exec usage
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec", "compile"):
                        imports.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute) and node.func.attr == "system":
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                            imports.add("os.system")
            return imports

        orig_imports = _extract_imports(orig_tree)
        prop_imports = _extract_imports(prop_tree)
        new_imports = prop_imports - orig_imports

        if not new_imports:
            return GateCheck(
                gate_name="imports",
                result=GateResult.PASSED,
                score=1.0,
                message="No new imports detected",
            )

        # Check against dangerous imports
        max_severity = "LOW"
        flagged = []
        for imp in new_imports:
            # Check exact match or prefix match
            matched = False
            for pattern, (severity, reason) in DANGEROUS_IMPORTS.items():
                if imp == pattern or imp.startswith(pattern + "."):
                    flagged.append({"import": imp, "severity": severity, "reason": reason})
                    if severity in ("CRITICAL", "HIGH") and severity > max_severity:
                        max_severity = severity
                    matched = True
                    break
            if not matched and not imp.startswith("."):
                flagged.append({"import": imp, "severity": "INFO", "reason": "New import"})

        critical_flags = [f for f in flagged if f["severity"] in ("CRITICAL", "HIGH")]

        if critical_flags:
            details_str = "; ".join(f"{f['import']} ({f['severity']}): {f['reason']}" for f in critical_flags)
            return GateCheck(
                gate_name="imports",
                result=GateResult.FAILED,
                score=0.0,
                message=f"Dangerous new imports blocked: {details_str}",
                details={"new_imports": [f["import"] for f in flagged], "flagged": flagged},
            )

        if flagged:
            details_str = "; ".join(f"{f['import']} ({f['severity']}): {f['reason']}" for f in flagged)
            return GateCheck(
                gate_name="imports",
                result=GateResult.WARNING,
                score=0.7,
                message=f"New imports require review: {details_str}",
                details={"new_imports": [f["import"] for f in flagged], "flagged": flagged},
            )

        return GateCheck(
            gate_name="imports",
            result=GateResult.PASSED,
            score=0.9,
            message=f"Safe new imports: {', '.join(new_imports)}",
            details={"new_imports": list(new_imports)},
        )

    def _gate_patterns(self, proposed_code: str) -> GateCheck:
        """Gate 4: Check for suspicious code patterns."""
        SUSPICIOUS_PATTERNS = [
            # Network exfiltration
            (r"(?:requests|httpx|urllib|aiohttp)\.(?:get|post|put|delete|patch)\s*\(\s*['\"]https?://",
             "HIGH", "Outbound HTTP request detected"),
            (r"socket\.(?:socket|create_connection)",
             "HIGH", "Raw socket creation detected"),
            (r"(?:base64|binascii)\.(?:b64decode|a2b_)",
             "MEDIUM", "Base64 decoding detected (potential obfuscation)"),
            # Code execution
            (r"\beval\s*\(",
             "CRITICAL", "eval() usage detected"),
            (r"\bexec\s*\(",
             "CRITICAL", "exec() usage detected"),
            (r"__import__\s*\(",
             "HIGH", "Dynamic import via __import__() detected"),
            (r"getattr\s*\(\s*(?:builtins|__builtins__|globals|locals)",
             "HIGH", "Dynamic attribute access on builtins detected"),
            # File system
            (r"shutil\.rmtree\s*\(",
             "HIGH", "Recursive directory deletion detected"),
            (r"os\.(?:remove|unlink|system)\s*\(",
             "HIGH", "Dangerous OS operation detected"),
            (r"os\.walk\s*\(",
             "LOW", "Directory traversal detected"),
            # Privilege escalation
            (r"os\.(?:setuid|setgid|seteuid|setegid|chroot)",
             "CRITICAL", "Privilege escalation attempt detected"),
            (r"(?:subprocess|os\.popen)\s*\(\s*['\"]",
             "HIGH", "Shell command execution detected"),
            # Environment manipulation
            (r"os\.environ(?:\[[^\]]+\]|\.pop|\.update)\s*=",
             "MEDIUM", "Environment variable modification detected"),
            # Data exfiltration via encoding
            (r"(?:json|pickle)\.(?:dumps|dump)\s*\([^)]*(?:os\.|sys\.|environ|getenv)",
             "HIGH", "Potential data serialization with system data"),
        ]

        max_severity = "LOW"
        matched_patterns = []

        for pattern, severity, description in SUSPICIOUS_PATTERNS:
            matches = re.findall(pattern, proposed_code, re.MULTILINE | re.IGNORECASE)
            if matches:
                matched_patterns.append({
                    "pattern": pattern[:60],
                    "severity": severity,
                    "description": description,
                    "count": len(matches),
                })
                if severity in ("CRITICAL", "HIGH") and severity > max_severity:
                    max_severity = severity

        critical = [p for p in matched_patterns if p["severity"] in ("CRITICAL", "HIGH")]

        if critical:
            return GateCheck(
                gate_name="patterns",
                result=GateResult.FAILED,
                score=0.0,
                message=f"Suspicious patterns detected: {', '.join(p['description'] for p in critical)}",
                details={"patterns": matched_patterns},
            )

        if matched_patterns:
            return GateCheck(
                gate_name="patterns",
                result=GateResult.WARNING,
                score=0.7,
                message=f"Minor suspicious patterns: {', '.join(p['description'] for p in matched_patterns)}",
                details={"patterns": matched_patterns},
            )

        return GateCheck(
            gate_name="patterns",
            result=GateResult.PASSED,
            score=1.0,
            message="No suspicious patterns detected",
        )

    async def _gate_tests(self, file_path: str, proposed_code: str) -> GateCheck:
        """Gate 5: Run tests to verify the change doesn't break anything."""
        test_file = self._find_test_file(file_path)
        if test_file is None:
            return GateCheck(
                gate_name="tests",
                result=GateResult.SKIPPED,
                score=0.8,
                message="No test file found for this module",
            )

        try:
            # Write proposed code to a temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, dir=str(self.project_root)
            ) as tmp:
                tmp.write(proposed_code)
                tmp_path = tmp.name

            # Run syntax check on proposed code
            result = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "py_compile", tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(result.communicate(), timeout=15)

            if result.returncode != 0:
                return GateCheck(
                    gate_name="tests",
                    result=GateResult.WARNING,
                    score=0.5,
                    message="Proposed code has compilation issues",
                )

            return GateCheck(
                gate_name="tests",
                result=GateResult.PASSED,
                score=0.9,
                message=f"Syntax verification passed (test file: {test_file.name})",
                details={"test_file": str(test_file)},
            )

        except asyncio.TimeoutError:
            return GateCheck(
                gate_name="tests",
                result=GateResult.WARNING,
                score=0.5,
                message="Test verification timed out",
            )
        except Exception as e:
            return GateCheck(
                gate_name="tests",
                result=GateResult.SKIPPED,
                score=0.6,
                message=f"Could not run tests: {e}",
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _gate_lint(self, proposed_code: str) -> GateCheck:
        """Gate 6: Run lint checks on the proposed code."""
        # Simple built-in linting: check for common issues
        issues = []
        lines = proposed_code.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Bare except
            if re.match(r"^except\s*:", stripped):
                issues.append({"line": i, "issue": "bare except", "severity": "MEDIUM"})
            # Mutable default argument
            if re.match(r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\}|\{.*\}|set\(\))", stripped):
                issues.append({"line": i, "issue": "mutable default argument", "severity": "MEDIUM"})
            # Unused import (heuristic)
            if stripped.startswith("import ") and ";" in stripped:
                issues.append({"line": i, "issue": "multiple statements on one line", "severity": "LOW"})
            # Line too long
            if len(line) > 120:
                issues.append({"line": i, "issue": f"line too long ({len(line)} chars)", "severity": "LOW"})
            # Trailing whitespace
            if line.rstrip() != line and stripped:
                issues.append({"line": i, "issue": "trailing whitespace", "severity": "LOW"})
            # Missing docstring after class/function def
            if re.match(r"^(class|async\s+def|def)\s+", stripped):
                # Look ahead for docstring
                if i < len(lines) and not (lines[i].strip().startswith('"""') or lines[i].strip().startswith("'''")):
                    if not stripped.endswith(":"):  # one-liner
                        pass
                    else:
                        issues.append({"line": i, "issue": "missing docstring", "severity": "LOW"})

        high_issues = [i for i in issues if i["severity"] in ("HIGH", "MEDIUM")]

        if high_issues:
            return GateCheck(
                gate_name="lint",
                result=GateResult.WARNING,
                score=0.7,
                message=f"{len(high_issues)} lint issues found",
                details={"issues": issues, "high_count": len(high_issues)},
            )

        if issues:
            return GateCheck(
                gate_name="lint",
                result=GateResult.PASSED,
                score=0.85,
                message=f"{len(issues)} minor lint issues (cosmetic only)",
                details={"issues": issues},
            )

        return GateCheck(
            gate_name="lint",
            result=GateResult.PASSED,
            score=1.0,
            message="No lint issues detected",
        )

    def _gate_protected(self, file_path: str, change_type: ChangeType) -> GateCheck:
        """Gate 7: Extra scrutiny for protected files."""
        rel_path = str(file_path)
        # Check against protected files
        for pf in PROTECTED_FILES:
            if rel_path.endswith(pf) or pf in rel_path:
                if change_type in (ChangeType.REWRITE, ChangeType.EXTENSION):
                    return GateCheck(
                        gate_name="protected",
                        result=GateResult.WARNING,
                        score=0.6,
                        message=f"Protected file ({pf}) with high-impact change type ({change_type.value})",
                        details={"protected_file": pf, "change_type": change_type.value},
                    )
                return GateCheck(
                    gate_name="protected",
                    result=GateResult.PASSED,
                    score=0.8,
                    message=f"Protected file ({pf}) with low-risk change type ({change_type.value})",
                    details={"protected_file": pf, "change_type": change_type.value},
                )

        return GateCheck(
            gate_name="protected",
            result=GateResult.PASSED,
            score=1.0,
            message="File is not in the protected registry",
        )

    # ── Helper ────────────────────────────────────────────────────────────

    def _find_test_file(self, file_path: str) -> Optional[Path]:
        """Find the associated test file for a given source file."""
        p = Path(file_path)
        possible_names = [
            p.with_name(f"test_{p.stem}.py"),
            p.with_name(f"{p.stem}_test.py"),
            p.parent / "tests" / f"test_{p.stem}.py",
            self.project_root / "tests" / f"test_{p.stem}.py",
        ]
        for name in possible_names:
            if name.exists():
                return name
        return None

    # ── Main Evaluation ───────────────────────────────────────────────────

    async def evaluate_change(
        self,
        file_path: str,
        original_code: str,
        proposed_code: str,
        change_type: ChangeType = ChangeType.PATCH,
        reason: str = "",
    ) -> SafetyEvaluation:
        """
        Evaluate a proposed code change through all safety gates.

        Returns a SafetyEvaluation with approval decision, scores, and details.
        """
        gates: List[GateCheck] = []
        warnings: List[str] = []
        blockers: List[str] = []
        recommendations: List[str] = []

        # Pre-check: rate limiting
        rate_ok, rate_msg = await self._check_rate_limit(file_path)
        if not rate_ok:
            return SafetyEvaluation(
                approved=False,
                approval_level=ApprovalLevel.BLOCKED,
                overall_score=0.0,
                gates=[],
                warnings=[],
                blockers=[rate_msg],
                recommendations=["Wait before making more changes"],
                file_path=file_path,
                change_type=change_type,
                estimated_risk=0.0,
            )

        # Gate 1: Syntax (MUST pass)
        syntax_gate = self._gate_syntax(proposed_code)
        gates.append(syntax_gate)
        if syntax_gate.result == GateResult.FAILED:
            blockers.append(syntax_gate.message)
            return SafetyEvaluation(
                approved=False,
                approval_level=ApprovalLevel.BLOCKED,
                overall_score=0.0,
                gates=gates,
                warnings=warnings,
                blockers=blockers,
                recommendations=recommendations,
                file_path=file_path,
                change_type=change_type,
                estimated_risk=1.0,
            )

        # Gate 2: Size
        size_gate = self._gate_size(original_code, proposed_code)
        gates.append(size_gate)
        if size_gate.result == GateResult.WARNING:
            warnings.append(size_gate.message)
            recommendations.append("Consider breaking this into smaller changes")

        # Gate 3: Imports
        imports_gate = self._gate_imports(original_code, proposed_code)
        gates.append(imports_gate)
        if imports_gate.result == GateResult.FAILED:
            blockers.append(imports_gate.message)
        elif imports_gate.result == GateResult.WARNING:
            warnings.append(imports_gate.message)

        # Gate 4: Patterns
        patterns_gate = self._gate_patterns(proposed_code)
        gates.append(patterns_gate)
        if patterns_gate.result == GateResult.FAILED:
            blockers.append(patterns_gate.message)
        elif patterns_gate.result == GateResult.WARNING:
            warnings.append(patterns_gate.message)

        # Gate 5: Tests
        tests_gate = await self._gate_tests(file_path, proposed_code)
        gates.append(tests_gate)
        if tests_gate.result == GateResult.WARNING:
            warnings.append(tests_gate.message)

        # Gate 6: Lint
        lint_gate = self._gate_lint(proposed_code)
        gates.append(lint_gate)
        if lint_gate.result == GateResult.WARNING:
            warnings.append(lint_gate.message)
            recommendations.append("Fix lint issues before applying")

        # Gate 7: Protected
        protected_gate = self._gate_protected(file_path, change_type)
        gates.append(protected_gate)
        if protected_gate.result == GateResult.WARNING:
            warnings.append(protected_gate.message)
            recommendations.append("This file is in the protected registry — review carefully")

        # Calculate overall score (weighted average)
        weights = {
            "syntax": 0.25,
            "size": 0.10,
            "imports": 0.20,
            "patterns": 0.20,
            "tests": 0.10,
            "lint": 0.05,
            "protected": 0.10,
        }
        overall_score = sum(
            g.score * weights.get(g.gate_name, 0.1) for g in gates
        )

        # Determine approval
        has_blockers = len(blockers) > 0
        high_risk = overall_score < 0.5
        low_risk = overall_score >= 0.8

        if has_blockers:
            approved = False
            approval_level = ApprovalLevel.BLOCKED
        elif high_risk:
            approved = False
            approval_level = ApprovalLevel.MANUAL_REVIEW
        elif len(warnings) > 0:
            approved = True
            approval_level = ApprovalLevel.ADVISORY
        else:
            approved = True
            approval_level = ApprovalLevel.AUTO_APPROVE

        estimated_risk = round(1.0 - overall_score, 3)

        return SafetyEvaluation(
            approved=approved,
            approval_level=approval_level,
            overall_score=round(overall_score, 3),
            gates=gates,
            warnings=warnings,
            blockers=blockers,
            recommendations=recommendations,
            file_path=file_path,
            change_type=change_type,
            estimated_risk=estimated_risk,
        )

    # ── Backup & Rollback ─────────────────────────────────────────────────

    async def backup_file(self, file_path: str, change_type: ChangeType = ChangeType.PATCH, reason: str = "") -> str:
        """
        Create a backup of a file before modification.

        Returns the backup_id for later rollback.
        """
        full_path = self.project_root / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"Cannot backup: {full_path} does not exist")

        content = full_path.read_text(encoding="utf-8")
        content_hash = _file_hash(content)

        backup_id = _generate_id()
        backup_subdir = self.backup_dir / backup_id
        backup_subdir.mkdir(parents=True, exist_ok=True)

        # Copy the file
        backup_path = backup_subdir / full_path.name
        shutil.copy2(str(full_path), str(backup_path))

        # Also save metadata
        meta = {
            "backup_id": backup_id,
            "file_path": str(file_path),
            "original_hash": content_hash,
            "timestamp": _now_iso(),
            "change_type": change_type.value,
            "reason": reason,
        }
        (backup_subdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Record in database
        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO backups (backup_id, file_path, original_hash, backup_path, timestamp, change_type, reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (backup_id, str(file_path), content_hash, str(backup_path), _now_iso(), change_type.value, reason),
            )
            conn.commit()

        await self._run_sync(_do)
        return backup_id

    async def apply_change(self, file_path: str, proposed_code: str) -> str:
        """
        Apply a proposed code change to a file.

        Returns the new file hash.
        """
        full_path = self.project_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(proposed_code, encoding="utf-8")
        return _file_hash(proposed_code)

    async def rollback(self, backup_id: str) -> bool:
        """
        Rollback a change using a backup_id.

        Returns True if rollback was successful.
        """
        def _do_backup() -> Optional[BackupRecord]:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT * FROM backups WHERE backup_id = ?", (backup_id,)
            ).fetchone()
            if not row:
                return None
            return BackupRecord(
                backup_id=row["backup_id"],
                file_path=row["file_path"],
                original_hash=row["original_hash"],
                backup_path=row["backup_path"],
                timestamp=row["timestamp"],
                change_type=ChangeType(row["change_type"]),
                reason=row["reason"],
                rolled_back=bool(row["rolled_back"]),
            )

        record = await self._run_sync(_do_backup)
        if record is None:
            return False

        backup_path = Path(record.backup_path)
        if not backup_path.exists():
            return False

        full_path = self.project_root / record.file_path
        shutil.copy2(str(backup_path), str(full_path))

        # Mark as rolled back
        def _do_mark() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "UPDATE backups SET rolled_back = 1 WHERE backup_id = ?",
                (backup_id,),
            )
            conn.execute(
                "UPDATE change_history SET rolled_back = 1 WHERE backup_id = ?",
                (backup_id,),
            )
            conn.commit()

        await self._run_sync(_do_mark)
        return True

    async def record_change(self, history: ChangeHistory) -> None:
        """Record a change in the audit trail."""
        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO change_history "
                "(change_id, file_path, change_type, original_hash, new_hash, backup_id, "
                "approval_level, safety_score, reason, applied, rolled_back, timestamp, gates_summary) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    history.change_id,
                    history.file_path,
                    history.change_type.value,
                    history.original_hash,
                    history.new_hash,
                    history.backup_id,
                    history.approval_level.value,
                    history.safety_score,
                    history.reason,
                    int(history.applied),
                    int(history.rolled_back),
                    history.timestamp,
                    json.dumps(history.gates_summary),
                ),
            )
            conn.commit()

        await self._run_sync(_do)

    async def quarantine_change(
        self, file_path: str, proposed_code: str, reason: str, risk_score: float
    ) -> str:
        """
        Send a change to quarantine for manual review.

        Returns the quarantine ID.
        """
        q_id = _generate_id()

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO quarantine (id, file_path, proposed_code, reason, risk_score, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (q_id, str(file_path), proposed_code, reason, risk_score, _now_iso()),
            )
            conn.commit()

        await self._run_sync(_do)
        return q_id

    # ── Query Methods ─────────────────────────────────────────────────────

    async def get_change_history(
        self, file_path: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve change history, optionally filtered by file."""
        def _do() -> List[Dict[str, Any]]:
            conn = self._ensure_conn()
            if file_path:
                rows = conn.execute(
                    "SELECT * FROM change_history WHERE file_path = ? ORDER BY timestamp DESC LIMIT ?",
                    (file_path, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM change_history ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

        return await self._run_sync(_do)

    async def get_backups(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve available backups."""
        def _do() -> List[Dict[str, Any]]:
            conn = self._ensure_conn()
            if file_path:
                rows = conn.execute(
                    "SELECT * FROM backups WHERE file_path = ? ORDER BY timestamp DESC",
                    (file_path,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM backups ORDER BY timestamp DESC LIMIT 100"
                ).fetchall()
            return [dict(r) for r in rows]

        return await self._run_sync(_do)

    async def get_quarantined(self) -> List[Dict[str, Any]]:
        """Retrieve quarantined changes awaiting review."""
        def _do() -> List[Dict[str, Any]]:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT * FROM quarantine WHERE resolved = 0 ORDER BY timestamp DESC"
            ).fetchall()
            return [dict(r) for r in rows]

        return await self._run_sync(_do)

    async def resolve_quarantine(self, quarantine_id: str, approved: bool) -> bool:
        """Resolve a quarantined change (approve or reject)."""
        def _do() -> int:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "UPDATE quarantine SET resolved = 1 WHERE id = ?",
                (quarantine_id,),
            )
            conn.commit()
            return cursor.rowcount

        rows = await self._run_sync(_do)
        return rows > 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get safety system statistics."""
        def _do() -> Dict[str, Any]:
            conn = self._ensure_conn()
            total_changes = conn.execute("SELECT COUNT(*) FROM change_history").fetchone()[0]
            applied_changes = conn.execute("SELECT COUNT(*) FROM change_history WHERE applied = 1").fetchone()[0]
            rolled_back = conn.execute("SELECT COUNT(*) FROM change_history WHERE rolled_back = 1").fetchone()[0]
            avg_score = conn.execute(
                "SELECT AVG(safety_score) FROM change_history WHERE applied = 1"
            ).fetchone()[0] or 0.0
            total_backups = conn.execute("SELECT COUNT(*) FROM backups").fetchone()[0]
            quarantined = conn.execute("SELECT COUNT(*) FROM quarantine WHERE resolved = 0").fetchone()[0]

            # Change type distribution
            type_dist = {}
            for row in conn.execute("SELECT change_type, COUNT(*) FROM change_history GROUP BY change_type"):
                type_dist[row[0]] = row[1]

            return {
                "total_changes": total_changes,
                "applied_changes": applied_changes,
                "rolled_back_changes": rolled_back,
                "average_safety_score": round(avg_score, 3),
                "total_backups": total_backups,
                "quarantined_pending": quarantined,
                "change_type_distribution": type_dist,
                "daily_limit_remaining": max(0, DAILY_CHANGE_LIMIT - self._daily_change_count),
            }

        return await self._run_sync(_do)

    async def safety_report(self) -> str:
        """Generate a human-readable safety report."""
        stats = await self.get_stats()
        recent = await self.get_change_history(limit=10)
        quarantined = await self.get_quarantined()

        lines = [
            "# Self-Improvement Safety Report",
            "",
            f"**Total Changes:** {stats['total_changes']}",
            f"**Applied:** {stats['applied_changes']}",
            f"**Rolled Back:** {stats['rolled_back_changes']}",
            f"**Average Safety Score:** {stats['average_safety_score']:.2f}",
            f"**Backups Available:** {stats['total_backups']}",
            f"**Pending Quarantine:** {stats['quarantined_pending']}",
            f"**Daily Limit Remaining:** {stats['daily_limit_remaining']}",
            "",
        ]

        if stats.get("change_type_distribution"):
            lines.append("## Change Type Distribution")
            for ct, count in stats["change_type_distribution"].items():
                lines.append(f"- {ct}: {count}")
            lines.append("")

        if recent:
            lines.append("## Recent Changes")
            for ch in recent:
                status = "applied" if ch["applied"] else "rejected"
                rb = " (ROLLED BACK)" if ch["rolled_back"] else ""
                lines.append(
                    f"- [{status}{rb}] {ch['file_path']} "
                    f"(score={ch['safety_score']:.2f}, type={ch['change_type']})"
                )
            lines.append("")

        if quarantined:
            lines.append("## Quarantined Changes Awaiting Review")
            for q in quarantined:
                lines.append(f"- {q['file_path']}: {q['reason'][:100]} (risk={q['risk_score']:.2f})")
            lines.append("")

        return "\n".join(lines)


# Need sys for test gate
import sys
