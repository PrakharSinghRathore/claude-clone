"""
Security Vulnerability Scanner Module for Claude Code Clone.

Provides comprehensive source code security scanning including OWASP Top 10
detection, secret/credential leak detection, dependency vulnerability checking,
file permission analysis, and dangerous code pattern identification.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SeverityLevel(Enum):
    """Severity levels for vulnerability findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    def sort_weight(self) -> int:
        weights = {self.CRITICAL: 5, self.HIGH: 4, self.MEDIUM: 3, self.LOW: 2, self.INFO: 1}
        return weights[self]


class VulnerabilityCategory(Enum):
    """Categories of security vulnerabilities."""
    SECRET_LEAK = "secret_leak"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    BROKEN_AUTH = "broken_auth"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    DANGEROUS_FUNCTION = "dangerous_function"
    INSECURE_CONFIG = "insecure_config"
    PERMISSION_ISSUE = "permission_issue"
    DEPENDENCY_VULN = "dependency_vuln"
    CODE_SMELL = "code_smell"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Vulnerability:
    """Represents a single security vulnerability finding."""
    id: str
    title: str
    severity: SeverityLevel
    file_path: str
    line_number: Optional[int]
    code_snippet: Optional[str]
    category: VulnerabilityCategory
    description: str
    remediation: str
    cwe_id: Optional[str] = None
    confidence: float = 0.8

    def __hash__(self) -> int:
        fingerprint = f"{self.file_path}:{self.line_number}:{self.category.value}:{self.title}"
        return int(hashlib.sha256(fingerprint.encode()).hexdigest(), 16) % (2**63)


@dataclass
class ScanResult:
    """Aggregated result of a security scan."""
    total_findings: int
    by_severity: dict[str, int]
    by_category: dict[str, int]
    vulnerabilities: list[Vulnerability]
    scan_time: datetime
    files_scanned: int
    project_path: str
    scan_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.scan_id is None:
            raw = f"{self.project_path}:{self.scan_time.isoformat()}"
            self.scan_id = hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class ScanHistoryEntry:
    """A single entry in the scan history."""
    scan_id: str
    timestamp: datetime
    project_path: str
    total_findings: int
    critical_count: int
    high_count: int
    files_scanned: int


# ---------------------------------------------------------------------------
# Known CVE Database (simulated)
# ---------------------------------------------------------------------------

_KNOWN_CVES: dict[str, list[dict]] = {
    "requests": [
        {"version": "<2.28.0", "cve": "CVE-2023-32681", "severity": SeverityLevel.MEDIUM,
         "title": "Unintended leak of Proxy-Authorization header",
         "description": "Requests would leak the Proxy-Authorization header to destination servers when following HTTPS redirects.",
         "remediation": "Upgrade requests to >=2.28.0", "cwe": "CWE-200"},
    ],
    "flask": [
        {"version": "<2.3.2", "cve": "CVE-2023-30861", "severity": SeverityLevel.HIGH,
         "title": "Open redirect vulnerability in Flask URL routing",
         "description": "Flask is vulnerable to open redirects through the url_for function when URL routing is used.",
         "remediation": "Upgrade Flask to >=2.3.2", "cwe": "CWE-601"},
    ],
    "django": [
        {"version": "<4.2.5", "cve": "CVE-2023-46695", "severity": SeverityLevel.HIGH,
         "title": "Potential DoS in django.utils.text.Truncator",
         "description": "A DoS issue was discovered in Django's text truncation utilities.",
         "remediation": "Upgrade Django to >=4.2.5", "cwe": "CWE-400"},
        {"version": "<4.2.4", "cve": "CVE-2023-41164", "severity": SeverityLevel.CRITICAL,
         "title": "SQL injection in QuerySet.values()/values_list()",
         "description": "SQL injection vulnerability in QuerySet.order_by() when paired with values()/values_list().",
         "remediation": "Upgrade Django to >=4.2.4", "cwe": "CWE-89"},
    ],
    "pillow": [
        {"version": "<9.5.0", "cve": "CVE-2023-44271", "severity": SeverityLevel.HIGH,
         "title": "Uncontrolled resource consumption in Pillow",
         "description": "Pillow is susceptible to an out-of-bounds read in J2K image files.",
         "remediation": "Upgrade Pillow to >=9.5.0", "cwe": "CWE-125"},
    ],
    "pyyaml": [
        {"version": "<6.0.1", "cve": "CVE-2020-14343", "severity": SeverityLevel.HIGH,
         "title": "Arbitrary code execution via yaml.load",
         "description": "PyYAML's yaml.load function can instantiate arbitrary Python objects.",
         "remediation": "Use yaml.safe_load() or upgrade PyYAML to >=6.0.1", "cwe": "CWE-502"},
    ],
    "urllib3": [
        {"version": "<2.0.4", "cve": "CVE-2023-43804", "severity": SeverityLevel.HIGH,
         "title": "Request body not stripped on 303 redirect",
         "description": "urllib3 does not strip the request body on 303 redirects.",
         "remediation": "Upgrade urllib3 to >=2.0.4", "cwe": "CWE-639"},
    ],
    "cryptography": [
        {"version": "<41.0.3", "cve": "CVE-2023-49083", "severity": SeverityLevel.MEDIUM,
         "title": "Memory corruption in PKCS#7 padding handling",
         "description": "A memory corruption issue exists in the decryption of ciphertexts.",
         "remediation": "Upgrade cryptography to >=41.0.3", "cwe": "CWE-120"},
    ],
    "numpy": [
        {"version": "<1.25.2", "cve": "CVE-2023-41105", "severity": SeverityLevel.HIGH,
         "title": "Buffer overflow in numpy.sort",
         "description": "Buffer overflow in the array_from_pyobj function of numpy.sort.",
         "remediation": "Upgrade numpy to >=1.25.2", "cwe": "CWE-120"},
    ],
    "jinja2": [
        {"version": "<3.1.3", "cve": "CVE-2024-22195", "severity": SeverityLevel.HIGH,
         "title": "ReDoS in the urlize filter",
         "description": "The urlize filter in Jinja2 is vulnerable to Regular Expression Denial of Service.",
         "remediation": "Upgrade Jinja2 to >=3.1.3", "cwe": "CWE-1333"},
    ],
    "fastapi": [
        {"version": "<0.104.1", "cve": "CVE-2023-46146", "severity": SeverityLevel.MEDIUM,
         "title": "Open redirect in FastAPI",
         "description": "FastAPI is vulnerable to open redirects through the get_swagger_ui_html function.",
         "remediation": "Upgrade FastAPI to >=0.104.1", "cwe": "CWE-601"},
    ],
    "log4j": [
        {"version": "<2.17.1", "cve": "CVE-2021-45105", "severity": SeverityLevel.CRITICAL,
         "title": "Log4Shell - RCE via JNDI lookups",
         "description": "Log4j2 allows JNDI lookups which can be exploited for remote code execution.",
         "remediation": "Upgrade log4j to >=2.17.1", "cwe": "CWE-917"},
    ],
    "lodash": [
        {"version": "<4.17.21", "cve": "CVE-2021-23337", "severity": SeverityLevel.HIGH,
         "title": "Command injection in lodash template function",
         "description": "Lodash templates can be exploited for command injection.",
         "remediation": "Upgrade lodash to >=4.17.21", "cwe": "CWE-78"},
    ],
}


def _version_satisfies(installed: str, constraint: str) -> bool:
    """Check if *installed* version satisfies a pip-style *constraint* like '<2.28.0'."""
    try:
        def _parse(v: str) -> list[int]:
            parts = re.split(r"[.+-]", v)
            return [int(p) for p in parts if p.isdigit()]

        inst = _parse(installed)
        match = re.match(r"(<=?|>=?|==|!=)\s*(.+)", constraint.strip())
        if not match:
            return False
        op, ver = match.groups()
        target = _parse(ver)
        # normalise to same length
        maxlen = max(len(inst), len(target))
        inst.extend([0] * (maxlen - len(inst)))
        target.extend([0] * (maxlen - len(target)))
        if op == "<":
            return inst < target
        if op == "<=":
            return inst <= target
        if op == ">":
            return inst > target
        if op == ">=":
            return inst >= target
        if op == "==":
            return inst == target
        if op == "!=":
            return inst != target
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class SecurityScanner:
    """Comprehensive security vulnerability scanner for source code projects."""

    # Languages we can meaningfully scan for code-level vulnerabilities
    CODE_EXTENSIONS: frozenset[str] = frozenset({
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb", ".php",
        ".c", ".cpp", ".h", ".hpp", ".rs", ".cs", ".swift", ".kt", ".scala",
        ".sh", ".bash", ".zsh", ".fish", ".ps1", ".lua", ".r", ".pl", ".ex",
        ".exs", ".erl", ".hs", ".ml", ".mli", ".vim",
    })

    CONFIG_EXTENSIONS: frozenset[str] = frozenset({
        ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".conf", ".xml",
        ".env", ".properties",
    })

    # Files whose permissions should be audited
    SENSITIVE_FILES: frozenset[str] = frozenset({
        ".env", ".env.local", ".env.production", ".env.staging",
        "id_rsa", "id_ed25519", "id_dsa", "id_ecdsa",
        ".pem", ".key", ".p12", ".pfx", ".jks", ".keystore",
        "credentials.json", ".npmrc", ".pypirc",
        ".gitconfig", ".netrc", ".ssh/config",
        "wallet.dat", "keystore.json",
    })

    def __init__(self, project_path: str, ignore_file: str = ".claudescanignore") -> None:
        self.project_path = Path(project_path).resolve()
        self.ignore_file = ignore_file
        self._ignore_rules: list[str] = []
        self._ignore_patterns: list[re.Pattern] = []
        self._history: list[ScanHistoryEntry] = []
        self._secrets_cache: list[tuple[str, str, str]] | None = None
        self._dangerous_cache: list[tuple[str, str, str, SeverityLevel]] | None = None
        self._load_ignore_rules()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scan(self, paths: list[str] | None = None) -> ScanResult:
        """Run a full security scan over the project (or specific *paths*).

        Returns a :class:`ScanResult` with all aggregated findings.
        """
        scan_start = datetime.now(timezone.utc)
        all_vulns: list[Vulnerability] = []
        files_scanned = 0

        # Determine target files
        target_files: list[Path] = []
        if paths:
            for p in paths:
                pp = Path(p)
                if pp.is_dir():
                    target_files.extend(self._walk_dir(pp))
                elif pp.is_file():
                    target_files.append(pp)
        else:
            target_files = self._walk_dir(self.project_path)

        # Filter ignored
        target_files = [f for f in target_files if not self._is_ignored(f)]

        # Scan files concurrently (bounded)
        semaphore = asyncio.Semaphore(50)
        async def _scan_one(fp: Path) -> list[Vulnerability]:
            async with semaphore:
                try:
                    return await self.scan_file(str(fp))
                except Exception:
                    return []

        results = await asyncio.gather(*[_scan_one(f) for f in target_files])
        for vulns in results:
            all_vulns.extend(vulns)
            if vulns:
                files_scanned += 1
            else:
                files_scanned += 1

        # Cross-cutting scans
        dep_vulns = await self.scan_dependencies()
        all_vulns.extend(dep_vulns)

        perm_vulns = await self.scan_permissions()
        all_vulns.extend(perm_vulns)

        # Deduplicate
        seen_ids: set[int] = set()
        unique: list[Vulnerability] = []
        for v in all_vulns:
            vid = hash(v)
            if vid not in seen_ids:
                seen_ids.add(vid)
                unique.append(v)

        unique.sort(key=lambda v: v.severity.sort_weight(), reverse=True)

        by_severity: dict[str, int] = {s.value: 0 for s in SeverityLevel}
        by_category: dict[str, int] = {c.value: 0 for c in VulnerabilityCategory}
        for v in unique:
            by_severity[v.severity.value] += 1
            by_category[v.category.value] += 1

        result = ScanResult(
            total_findings=len(unique),
            by_severity=by_severity,
            by_category=by_category,
            vulnerabilities=unique,
            scan_time=scan_start,
            files_scanned=files_scanned,
            project_path=str(self.project_path),
        )

        # Persist history
        self._history.append(ScanHistoryEntry(
            scan_id=result.scan_id,
            timestamp=scan_start,
            project_path=str(self.project_path),
            total_findings=result.total_findings,
            critical_count=by_severity.get("critical", 0),
            high_count=by_severity.get("high", 0),
            files_scanned=files_scanned,
        ))

        return result

    async def scan_file(self, filepath: str) -> list[Vulnerability]:
        """Scan a single file for all vulnerability types."""
        path = Path(filepath)
        if not path.is_file():
            return []

        vulns: list[Vulnerability] = []
        ext = path.suffix.lower()
        stem = path.name.lower()

        # Config files: secrets only
        if ext in self.CONFIG_EXTENSIONS or stem in self.SENSITIVE_FILES or stem.endswith(".env"):
            vulns.extend(await self.scan_secrets(filepath))
            return vulns

        # Code files: all applicable checks
        if ext in self.CODE_EXTENSIONS or path.stat().st_size < 5_000_000:
            try:
                vulns.extend(await self.scan_secrets(filepath))
                vulns.extend(await self.scan_sql_injection(filepath))
                vulns.extend(await self.scan_xss(filepath))
                vulns.extend(await self.scan_dangerous_functions(filepath))
                vulns.extend(await self.scan_insecure_deserialization(filepath))
                vulns.extend(await self.scan_broken_auth(filepath))
            except Exception:
                pass

        return vulns

    async def scan_secrets(self, filepath: str) -> list[Vulnerability]:
        """Detect hardcoded secrets, API keys, tokens, and credentials."""
        vulns: list[Vulnerability] = []
        try:
            content = Path(filepath).read_text(errors="ignore")
        except Exception:
            return vulns

        patterns = self._get_secrets_patterns()
        for idx, line in enumerate(content.splitlines(), start=1):
            for regex, name, remediation in patterns:
                match = re.search(regex, line)
                if match:
                    vuln_id = f"SECRET-{name.upper()}-{filepath}:{idx}"
                    snippet = line.strip()
                    # Mask the actual secret value in the snippet for safety
                    masked = self._mask_secret(snippet, match)
                    vulns.append(Vulnerability(
                        id=vuln_id,
                        title=f"Hardcoded {name} detected",
                        severity=SeverityLevel.CRITICAL,
                        file_path=filepath,
                        line_number=idx,
                        code_snippet=masked,
                        category=VulnerabilityCategory.SECRET_LEAK,
                        description=(
                            f"A hardcoded {name} was found in {filepath} at line {idx}. "
                            "Hardcoded secrets in source code can be extracted by anyone "
                            "with repository access and should be stored in environment "
                            "variables or a secrets manager."
                        ),
                        remediation=remediation,
                        cwe_id="CWE-798",
                        confidence=0.9,
                    ))
        return vulns

    async def scan_sql_injection(self, filepath: str) -> list[Vulnerability]:
        """Detect potential SQL injection vulnerabilities."""
        vulns: list[Vulnerability] = []
        try:
            content = Path(filepath).read_text(errors="ignore")
        except Exception:
            return vulns

        ext = Path(filepath).suffix.lower()
        # Focus on Python, PHP, Java, JS, Go, Ruby, C/C++
        sql_relevant = {".py", ".php", ".java", ".js", ".ts", ".go", ".rb", ".c", ".cpp", ".rs"}
        if ext not in sql_relevant:
            return vulns

        sql_patterns: list[tuple[str, str, SeverityLevel]] = [
            # Python f-string / format with SQL
            (r'(f["\'].*?SELECT\s.*?\{.*?\}|f["\'].*?INSERT\s.*?\{.*?\}|f["\'].*?UPDATE\s.*?\{.*?\}|f["\'].*?DELETE\s.*?\{.*?\}|f["\'].*?WHERE\s.*?\{.*?\})',
             "SQL query built with f-string interpolation", SeverityLevel.HIGH),
            # Python .format() with SQL
            (r'["\'].*?(?:SELECT|INSERT|UPDATE|DELETE|WHERE).*?%s.*?["\']\s*%\s*\(',
             "SQL query built with %-formatting", SeverityLevel.HIGH),
            (r'["\'].*?(?:SELECT|INSERT|UPDATE|DELETE|WHERE).*?\{.*?\}.*?["\']\s*\.format',
             "SQL query built with .format()", SeverityLevel.HIGH),
            # String concatenation with SQL keywords
            (r'(?:SELECT|INSERT|UPDATE|DELETE|DROP)\s+.*?\+\s*(?:request\.|params\.|args|input|data|form|query)',
             "SQL query built via string concatenation with user input", SeverityLevel.CRITICAL),
            # PHP / general
            (r'\$_(?:GET|POST|REQUEST|COOKIE)\[.*?\].*?(?:SELECT|INSERT|UPDATE|DELETE)',
             "Direct use of superglobal in SQL query", SeverityLevel.CRITICAL),
            (r'mysql_query\s*\(\s*\$',
             "Direct mysql_query call with potential injection", SeverityLevel.HIGH),
            # Java / JDBC
            (r'(?:Statement|createQuery|createNativeQuery)\s*\(\s*["\'].*?\+',
             "JPA/SQL query built via string concatenation", SeverityLevel.HIGH),
            # Go
            (r'db\.Query\s*\(\s*["\'].*?%s',
             "Go database query with unsafe formatting", SeverityLevel.HIGH),
            # Raw execute
            (r'(?:cursor\.)?execute\s*\(\s*f["\']',
             "Cursor execute with f-string (no parameterization)", SeverityLevel.HIGH),
            (r'raw\s*\(\s*f["\']',
             "Raw SQL query with f-string", SeverityLevel.HIGH),
            # ORM extra / raw
            (r'\.extra\s*\(\s*where\s*=\s*\[.*?f["\']',
             "Django .extra() with f-string", SeverityLevel.HIGH),
            (r'\.raw\s*\(\s*["\'].*?\%',
             "ORM raw query with % formatting", SeverityLevel.HIGH),
        ]

        for idx, line in enumerate(content.splitlines(), start=1):
            for regex, desc, severity in sql_patterns:
                if re.search(regex, line, re.IGNORECASE):
                    vulns.append(Vulnerability(
                        id=f"SQLI-{hashlib.sha256(f'{filepath}:{idx}:{desc}'.encode()).hexdigest()[:10]}",
                        title=f"Potential SQL Injection: {desc}",
                        severity=severity,
                        file_path=filepath,
                        line_number=idx,
                        code_snippet=line.strip(),
                        category=VulnerabilityCategory.SQL_INJECTION,
                        description=(
                            f"Potential SQL injection vulnerability detected at {filepath}:{idx}. "
                            "User input is being incorporated directly into a SQL query without "
                            "parameterization, which could allow an attacker to manipulate the query."
                        ),
                        remediation=(
                            "Use parameterized queries / prepared statements instead of string "
                            "interpolation. For Python, use cursor.execute('SELECT * FROM t WHERE id = %s', (user_id,)). "
                            "For ORMs, use their built-in query builder with bound parameters."
                        ),
                        cwe_id="CWE-89",
                        confidence=0.8,
                    ))
        return vulns

    async def scan_xss(self, filepath: str) -> list[Vulnerability]:
        """Detect potential Cross-Site Scripting (XSS) vulnerabilities."""
        vulns: list[Vulnerability] = []
        try:
            content = Path(filepath).read_text(errors="ignore")
        except Exception:
            return vulns

        ext = Path(filepath).suffix.lower()
        xss_relevant = {".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".htm", ".php", ".rb", ".java", ".go"}
        if ext not in xss_relevant:
            return vulns

        xss_patterns: list[tuple[str, str, SeverityLevel]] = [
            # HTML rendered without escaping
            (r'(?:innerHTML\s*=|outerHTML\s*=|document\.write\s*\(|\.html\s*\(|dangerouslySetInnerHTML)',
             "Direct HTML rendering without escaping", SeverityLevel.HIGH),
            # Template literals with HTML
            (r'(?:template|render).*?`.*?<.*?>.*?\$\{.*?(?:req|request|params|query|user|input)',
             "Template literal with HTML and user input", SeverityLevel.MEDIUM),
            # Django / Jinja autoescape off
            (r'\{\%\s*autoescape\s+off\s*\%\}',
             "Jinja2/Django autoescape disabled", SeverityLevel.HIGH),
            (r'\|\s*safe\s*\}\}',
             "Jinja2 |safe filter (disables auto-escaping)", SeverityLevel.MEDIUM),
            (r'Markup\s*\(',
             "Markup() wraps value as safe HTML", SeverityLevel.MEDIUM),
            # Flask
            (r'render_template_string\s*\(\s*f["\']',
             "render_template_string with f-string", SeverityLevel.HIGH),
            # React
            (r'dangerouslySetInnerHTML\s*=\s*\{\{?\s*__html',
             "dangerouslySetInnerHTML with dynamic content", SeverityLevel.HIGH),
            # General echo/print of user input in PHP
            (r'<\?php\s*echo\s*\$',
             "Direct echo of variable (potential XSS)", SeverityLevel.MEDIUM),
            # JS eval
            (r'document\.location\s*=.*?\+',
             "document.location set with concatenation", SeverityLevel.MEDIUM),
            # URL redirect with user input
            (r'(?:redirect|location\.href)\s*=\s*(?:req|request|params|query)',
             "Open redirect via user-controlled parameter", SeverityLevel.MEDIUM),
        ]

        for idx, line in enumerate(content.splitlines(), start=1):
            for regex, desc, severity in xss_patterns:
                if re.search(regex, line, re.IGNORECASE):
                    vulns.append(Vulnerability(
                        id=f"XSS-{hashlib.sha256(f'{filepath}:{idx}:{desc}'.encode()).hexdigest()[:10]}",
                        title=f"Potential XSS: {desc}",
                        severity=severity,
                        file_path=filepath,
                        line_number=idx,
                        code_snippet=line.strip(),
                        category=VulnerabilityCategory.XSS,
                        description=(
                            f"Potential Cross-Site Scripting vulnerability at {filepath}:{idx}. "
                            "User-controlled data may be rendered as HTML without proper escaping, "
                            "allowing attackers to inject malicious scripts."
                        ),
                        remediation=(
                            "Always escape user input before rendering in HTML. Use frameworks' "
                            "built-in auto-escaping. For React, avoid dangerouslySetInnerHTML. "
                            "For Django/Jinja2, keep autoescape enabled. For raw HTML, use "
                            "DOMPurify or bleach for sanitization."
                        ),
                        cwe_id="CWE-79",
                        confidence=0.75,
                    ))
        return vulns

    async def scan_dangerous_functions(self, filepath: str) -> list[Vulnerability]:
        """Detect calls to dangerous functions and code patterns."""
        vulns: list[Vulnerability] = []
        try:
            content = Path(filepath).read_text(errors="ignore")
        except Exception:
            return vulns

        patterns = self._get_dangerous_patterns()
        for idx, line in enumerate(content.splitlines(), start=1):
            for regex, desc, remediation, severity in patterns:
                if re.search(regex, line):
                    vulns.append(Vulnerability(
                        id=f"DANG-{hashlib.sha256(f'{filepath}:{idx}:{desc}'.encode()).hexdigest()[:10]}",
                        title=f"Dangerous function: {desc}",
                        severity=severity,
                        file_path=filepath,
                        line_number=idx,
                        code_snippet=line.strip(),
                        category=VulnerabilityCategory.DANGEROUS_FUNCTION,
                        description=(
                            f"Dangerous function or pattern detected at {filepath}:{idx}: {desc}. "
                            "This can lead to arbitrary code execution, command injection, or "
                            "other security issues if user input reaches this call."
                        ),
                        remediation=remediation,
                        cwe_id="CWE-78",
                        confidence=0.85,
                    ))
        return vulns

    async def scan_insecure_deserialization(self, filepath: str) -> list[Vulnerability]:
        """Detect insecure deserialization patterns."""
        vulns: list[Vulnerability] = []
        try:
            content = Path(filepath).read_text(errors="ignore")
        except Exception:
            return vulns

        deser_patterns: list[tuple[str, str, str, SeverityLevel]] = [
            (r'pickle\.loads?\s*\(',
             "pickle.loads() - arbitrary code execution risk",
             "Use JSON or msgpack for serialization. If pickle is required, never deserialize untrusted data.",
             SeverityLevel.CRITICAL),
            (r'yaml\.load\s*\((?!.*Loader\s*=\s*SafeLoader)(?!.*safe_load)',
             "yaml.load() without SafeLoader - arbitrary code execution",
             "Use yaml.safe_load() or yaml.load(data, Loader=SafeLoader) instead.",
             SeverityLevel.CRITICAL),
            (r'marshal\.loads?\s*\(',
             "marshal.loads() - arbitrary code execution risk",
             "Avoid marshal for untrusted data. Use safer serialization formats.",
             SeverityLevel.CRITICAL),
            (r'shelve\.open\s*\(',
             "shelve.open() uses pickle internally",
             "Ensure shelved data cannot be tampered with by untrusted parties.",
             SeverityLevel.HIGH),
            (r'jsonpickle\.decode\s*\(',
             "jsonpickle.decode() on untrusted data",
             "Validate and sanitize data before decoding, or use JSON directly.",
             SeverityLevel.HIGH),
            (r'ObjectInputStream',
             "Java ObjectInputStream - deserialization of untrusted data",
             "Use SerialKiller or JEP 290 filters to restrict deserializable classes.",
             SeverityLevel.CRITICAL),
            (r'readObject\s*\(',
             "Custom readObject() implementation",
             "Implement input validation in readObject(). Consider using Serializable proxies.",
             SeverityLevel.MEDIUM),
            (r'eval\s*\(\s*(?:json|request|data|input|user)',
             "eval() on potentially user-controlled data",
             "Use ast.literal_eval() for JSON-like data, or json.loads() for JSON.",
             SeverityLevel.CRITICAL),
            (r'unserialize\s*\(',
             "PHP unserialize() - potential object injection",
             "Use json_decode() instead. If unserialize is required, implement allowed_classes.",
             SeverityLevel.CRITICAL),
            (r'serial\.unserialize',
             "Node.js node-serialize or similar - arbitrary code execution",
             "Use JSON.parse() instead of custom serialization for untrusted data.",
             SeverityLevel.CRITICAL),
        ]

        for idx, line in enumerate(content.splitlines(), start=1):
            for regex, desc, remediation, severity in deser_patterns:
                if re.search(regex, line):
                    vulns.append(Vulnerability(
                        id=f"DESER-{hashlib.sha256(f'{filepath}:{idx}:{desc}'.encode()).hexdigest()[:10]}",
                        title=f"Insecure Deserialization: {desc}",
                        severity=severity,
                        file_path=filepath,
                        line_number=idx,
                        code_snippet=line.strip(),
                        category=VulnerabilityCategory.INSECURE_DESERIALIZATION,
                        description=(
                            f"Insecure deserialization pattern at {filepath}:{idx}: {desc}. "
                            "Deserializing untrusted data can lead to remote code execution."
                        ),
                        remediation=remediation,
                        cwe_id="CWE-502",
                        confidence=0.85,
                    ))
        return vulns

    async def scan_broken_auth(self, filepath: str) -> list[Vulnerability]:
        """Detect broken authentication patterns."""
        vulns: list[Vulnerability] = []
        try:
            content = Path(filepath).read_text(errors="ignore")
        except Exception:
            return vulns

        auth_patterns: list[tuple[str, str, str, SeverityLevel]] = [
            (r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']+["\']',
             "Hardcoded password in source code",
             "Use environment variables, a secrets manager, or configuration files outside version control.",
             SeverityLevel.CRITICAL),
            (r'AUTH_SECRET\s*=\s*["\'][^"\']+["\']',
             "Hardcoded authentication secret",
             "Store AUTH_SECRET in environment variables, not in source code.",
             SeverityLevel.CRITICAL),
            (r'SECRET_KEY\s*=\s*["\'][^"\']+["\']',
             "Hardcoded secret key (Django/Flask)",
             "Move SECRET_KEY to environment variables or a secrets manager.",
             SeverityLevel.CRITICAL),
            (r'JWT_SECRET\s*=\s*["\'][^"\']+["\']',
             "Hardcoded JWT secret",
             "Store JWT signing key in a secure secret manager or environment variable.",
             SeverityLevel.CRITICAL),
            (r'md5\s*\(\s*(?:password|passwd|pwd)',
             "MD5 used for password hashing (insecure)",
             "Use bcrypt, scrypt, or Argon2 for password hashing via passlib or similar.",
             SeverityLevel.HIGH),
            (r'sha1\s*\(\s*(?:password|passwd|pwd)',
             "SHA-1 used for password hashing (insecure)",
             "Use bcrypt, scrypt, or Argon2 for password hashing.",
             SeverityLevel.HIGH),
            (r'session\.secret\s*=\s*["\'][^"\']+["\']',
             "Hardcoded session secret (Express)",
             "Store session secret in environment variable SESSION_SECRET.",
             SeverityLevel.HIGH),
            (r'compare\s*\(\s*(?:password|passwd|pwd|token)',
             "Non-constant-time string comparison (timing attack)",
             "Use hmac.compare_digest() or secrets.compare_digest() for constant-time comparison.",
             SeverityLevel.MEDIUM),
            (r'(?:password|token|secret)\s*==\s*(?:request|user|input|data)',
             "Direct equality check on secret (timing attack risk)",
             "Use constant-time comparison: hmac.compare_digest() or secrets.compare_digest().",
             SeverityLevel.MEDIUM),
            (r'Basic\s+Authorization.*?(?:base64|b64)',
             "Basic Auth with hardcoded credentials",
             "Use OAuth 2.0 or API key authentication with proper token management.",
             SeverityLevel.HIGH),
        ]

        for idx, line in enumerate(content.splitlines(), start=1):
            for regex, desc, remediation, severity in auth_patterns:
                if re.search(regex, line, re.IGNORECASE):
                    vulns.append(Vulnerability(
                        id=f"AUTH-{hashlib.sha256(f'{filepath}:{idx}:{desc}'.encode()).hexdigest()[:10]}",
                        title=f"Broken Authentication: {desc}",
                        severity=severity,
                        file_path=filepath,
                        line_number=idx,
                        code_snippet=line.strip(),
                        category=VulnerabilityCategory.BROKEN_AUTH,
                        description=(
                            f"Authentication-related vulnerability at {filepath}:{idx}: {desc}. "
                            "This may allow attackers to compromise user accounts or bypass authentication."
                        ),
                        remediation=remediation,
                        cwe_id="CWE-287",
                        confidence=0.85,
                    ))
        return vulns

    async def scan_dependencies(self) -> list[Vulnerability]:
        """Check project dependencies for known CVEs."""
        vulns: list[Vulnerability] = []
        # Check requirements.txt
        req_file = self.project_path / "requirements.txt"
        if req_file.is_file():
            vulns.extend(await self._check_python_deps(req_file))

        # Check Pipfile.lock
        pipfile_lock = self.project_path / "Pipfile.lock"
        if pipfile_lock.is_file():
            vulns.extend(await self._check_pipfile_lock(pipfile_lock))

        # Check setup.cfg / pyproject.toml
        pyproject = self.project_path / "pyproject.toml"
        if pyproject.is_file():
            vulns.extend(await self._check_toml_deps(pyproject))

        # Check package.json
        pkg_file = self.project_path / "package.json"
        if pkg_file.is_file():
            vulns.extend(await self._check_js_deps(pkg_file))

        # Check package-lock.json for exact versions
        lock_file = self.project_path / "package-lock.json"
        if lock_file.is_file():
            vulns.extend(await self._check_npm_lock(lock_file))

        return vulns

    async def scan_permissions(self) -> list[Vulnerability]:
        """Check file permissions for sensitive files."""
        vulns: list[Vulnerability] = []

        for root, dirs, files in os.walk(str(self.project_path)):
            # Skip hidden and ignored directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and not self._is_ignored(Path(root) / d)]
            for fname in files:
                fpath = Path(root) / fname
                try:
                    st = fpath.stat()
                except OSError:
                    continue

                # Check if it matches a sensitive filename pattern
                fname_lower = fname.lower()
                is_sensitive = fname_lower in self.SENSITIVE_FILES or any(
                    fname_lower.endswith(s) for s in [".pem", ".key", ".p12", ".pfx", ".env"]
                )

                if not is_sensitive:
                    continue

                mode = st.st_mode
                # Check if group or others have read/write/execute
                if mode & stat.S_IROTH or mode & stat.S_IWOTH:
                    perms = stat.filemode(mode)
                    vulns.append(Vulnerability(
                        id=f"PERM-{hashlib.sha256(f'{fpath}'.encode()).hexdigest()[:10]}",
                        title=f"Overly permissive file: {fname} ({perms})",
                        severity=SeverityLevel.HIGH,
                        file_path=str(fpath),
                        line_number=None,
                        code_snippet=None,
                        category=VulnerabilityCategory.PERMISSION_ISSUE,
                        description=(
                            f"Sensitive file {fpath} has permissions {perms}, which may allow "
                            "other users on the system to read or modify it. This could expose "
                            "private keys, credentials, or environment variables."
                        ),
                        remediation=(
                            f"Restrict permissions: chmod 600 {fpath} (for files) or chmod 700 "
                            f"{fpath} (for directories). Sensitive files should only be readable "
                            "by their owner."
                        ),
                        cwe_id="CWE-732",
                        confidence=0.95,
                    ))

                # Also warn if group has write
                if mode & stat.S_IWGRP:
                    perms = stat.filemode(mode)
                    if not any(v.file_path == str(fpath) and "Overly permissive" in v.title for v in vulns):
                        vulns.append(Vulnerability(
                            id=f"PERM-G-{hashlib.sha256(f'{fpath}'.encode()).hexdigest()[:10]}",
                            title=f"Group-writable sensitive file: {fname} ({perms})",
                            severity=SeverityLevel.MEDIUM,
                            file_path=str(fpath),
                            line_number=None,
                            code_snippet=None,
                            category=VulnerabilityCategory.PERMISSION_ISSUE,
                            description=(
                                f"Sensitive file {fpath} is writable by group members ({perms}). "
                                "This increases the attack surface for credential theft."
                            ),
                            remediation=f"Run: chmod o-w,g-w {fpath} to remove group/other write permissions.",
                            cwe_id="CWE-732",
                            confidence=0.9,
                        ))

        return vulns

    async def generate_report(self, result: ScanResult, format: str = "markdown") -> str:
        """Generate a human-readable report from scan results.

        Supported formats: ``markdown``, ``json``, ``text``.
        """
        if format == "json":
            return self._report_json(result)
        if format == "text":
            return self._report_text(result)
        return self._report_markdown(result)

    async def get_history(self) -> list[ScanResult]:
        """Return scan history entries.

        Note: In-memory only for this implementation. A production system
        would persist to a database.
        """
        return self._history  # type: ignore[return-value]

    def load_ignore_rules(self) -> list[str]:
        """Load ignore rules from the configured ignore file."""
        self._load_ignore_rules()
        return self._ignore_rules

    # ------------------------------------------------------------------
    # Ignore rules
    # ------------------------------------------------------------------

    def _load_ignore_rules(self) -> None:
        """Load and compile ignore rules from the project's ignore file."""
        ignore_path = self.project_path / self.ignore_file
        self._ignore_rules = []
        self._ignore_patterns = []

        if not ignore_path.is_file():
            return

        try:
            lines = ignore_path.read_text(errors="ignore").splitlines()
        except Exception:
            return

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            self._ignore_rules.append(stripped)
            # Convert glob-like patterns to regex
            pattern = stripped.replace(".", r"\.").replace("*", ".*").replace("?", ".")
            try:
                self._ignore_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass

    def _is_ignored(self, path: Path) -> bool:
        """Check if a path matches any ignore rule."""
        rel = path.relative_to(self.project_path).as_posix()
        abs_path = str(path)
        for rule in self._ignore_rules:
            if rel.startswith(rule) or abs_path.endswith(rule):
                return True
        for pat in self._ignore_patterns:
            if pat.search(rel) or pat.search(abs_path):
                return True
        # Always ignore common non-source directories
        parts = rel.split("/")
        for skip in {"node_modules", ".git", "__pycache__", ".tox", ".eggs",
                      "venv", ".venv", ".mypy_cache", ".pytest_cache", "dist", "build",
                      ".next", ".nuxt", "coverage", ".coverage", ".idea", ".vscode"}:
            if skip in parts:
                return True
        return False

    # ------------------------------------------------------------------
    # Secret patterns (30+)
    # ------------------------------------------------------------------

    def _get_secrets_patterns(self) -> list[tuple[str, str, str]]:
        """Return (regex, human_name, remediation) for secret detection."""
        if self._secrets_cache is not None:
            return self._secrets_cache

        patterns: list[tuple[str, str, str]] = [
            # ---- Cloud Provider Keys ----
            (r'(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}',
             "AWS Access Key ID",
             "Rotate the key immediately in AWS IAM. Store in environment variables or a secrets manager."),

            (r'aws(.{0,20})?(?-i:secret(.{0,20})?access(.{0,20})?key)["\s:=]+([A-Za-z0-9/+=]{40})',
             "AWS Secret Access Key",
             "Rotate the key in AWS IAM. Never store secret keys in source code."),

            (r'aws_session_token["\s:=]+["\']?([A-Za-z0-9/+=]{200,})',
             "AWS Session Token",
             "AWS session tokens are temporary but still sensitive. Rotate and store securely."),

            (r'"type":\s*"service_account"[\s\S]{0,1000}"private_key":\s*"-----BEGIN',
             "GCP Service Account Private Key",
             "Rotate the service account key in GCP Console. Store in Google Secret Manager."),

            (r'AIza[0-9A-Za-z_-]{35}',
             "Google API Key",
             "Restrict the API key in Google Cloud Console and store in environment variables."),

            (r'ya29\.[0-9A-Za-z_-]+',
             "Google OAuth Access Token",
             "Revoke and regenerate the OAuth token. Store in a secure token store."),

            (r'default[A-Za-z0-9_-]{39}',
             "Google Cloud default token",
             "Rotate the default application credentials."),

            # ---- Azure ----
            (r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.(?:snapshot|vault|blob|file|table|queue)',
             "Azure Storage Account Key",
             "Regenerate the storage account key in the Azure Portal."),

            # ---- GitHub ----
            (r'(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}',
             "GitHub Personal Access Token / OAuth / App Token",
             "Revoke the token at GitHub Settings > Developer settings > Personal access tokens."),

            (r'gho_[A-Za-z0-9]{36}',
             "GitHub OAuth Access Token",
             "Revoke the OAuth application's access at GitHub Developer Settings."),

            # ---- GitLab ----
            (r'glpat-[A-Za-z0-9_-]{20,}',
             "GitLab Personal Access Token",
             "Revoke the token at GitLab > User Settings > Access Tokens."),

            (r'glrt-[A-Za-z0-9_-]{20,}',
             "GitLab Runtime Token",
             "Rotate the GitLab runner registration token."),

            # ---- Slack ----
            (r'xox[baprs]-[0-9]{10,13}-[0-9A-Za-z]{24,}',
             "Slack Token (Bot, App, User, or Refresh)",
             "Revoke the token at Slack API > Your Apps > OAuth & Permissions."),

            # ---- Stripe ----
            (r'(?:sk|pk|rk)_(?:test|live)_[0-9A-Za-z]{24,}',
             "Stripe API Key (Secret or Publishable)",
             "Rotate the key at Stripe Dashboard > Developers > API Keys."),

            # ---- Twilio ----
            (r'SK[0-9A-Fa-f]{32}',
             "Twilio API Key SID",
             "Rotate the Twilio API key at twilio.com/console."),

            (r'[0-9A-Fa-f]{32}:(?:[0-9A-Fa-f]{8}){2}',
             "Twilio Auth Token / API Key Secret",
             "Rotate the Twilio credentials at twilio.com/console."),

            # ---- SendGrid ----
            (r'SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}',
             "SendGrid API Key",
             "Rotate the SendGrid API key at app.sendgrid.com/settings/api_keys."),

            # ---- Mailgun ----
            (r'key-[0-9A-Za-z]{32}',
             "Mailgun API Key",
             "Rotate the Mailgun API key at app.mailgun.com/app/account."),

            # ---- Other SaaS tokens ----
            (r'hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{24,}',
             "Slack Webhook URL",
             "Revoke the webhook at Slack App settings. Store webhook URLs in environment variables."),

            (r'https://hooks\.slack\.com/services/[A-Za-z0-9/]+',
             "Slack Webhook URL (generic)",
             "Revoke and recreate the webhook. Store the URL in a secrets manager."),

            (r'https://discord(?:app)?\.com/api/webhooks/[0-9]+/[A-Za-z0-9_-]+',
             "Discord Webhook URL",
             "Revoke the webhook at Discord server/channel settings."),

            (r'HEROKU_API_KEY(?:\s*[:=]\s*|:\s*)([A-Za-z0-9_-]{36,})',
             "Heroku API Key",
             "Rotate the Heroku API key at dashboard.heroku.com/account."),

            (r'hk_live_[0-9A-Za-z]{32,}',
             "Heroku Pipeline API Key",
             "Rotate the pipeline key at Heroku Dashboard."),

            # ---- Database URLs (containing passwords) ----
            (r'(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis|amqp)://[^\s:]+:[^\s@]+@[^\s]+',
             "Database URL with embedded credentials",
             "Use environment variables for database credentials. Do not embed passwords in URLs."),

            # ---- Private Keys ----
            (r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
             "Private Key (PEM)",
             "Move the private key outside the repository. Use a secrets manager or deploy keys."),

            (r'-----BEGIN CERTIFICATE-----',
             "Certificate (PEM)",
             "Certificates in source code can be extracted. Move to a secure certificate store."),

            (r'-----BEGIN PGP PRIVATE KEY BLOCK-----',
             "PGP Private Key",
             "PGP private keys should never be in source code. Use a key management service."),

            # ---- JWT ----
            (r'eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
             "JSON Web Token (JWT)",
             "JWTs in source code may contain valid sessions. Revoke if compromised and store in HTTP-only cookies."),

            # ---- General tokens / secrets ----
            (r'(?i)(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?key|auth[_-]?token|private[_-]?key)["\s]*[:=]\s*["\']([A-Za-z0-9_\-/+=!@#$%^&*]{20,})["\']',
             "Generic API Key / Secret Key",
             "Store API keys in environment variables or a secrets manager. Rotate immediately."),

            (r'(?i)password["\s]*[:=]\s*["\']([^"\']{8,})["\']',
             "Hardcoded Password",
             "Use environment variables, a secrets manager, or secure credential storage."),

            (r'(?i)(?:token|bearer)["\s]*[:=]\s*["\']([A-Za-z0-9_\-.]{20,})["\']',
             "Hardcoded Bearer Token",
             "Store tokens securely in environment variables. Use short-lived tokens when possible."),

            (r'(?i)(?:CONSUMER|CLIENT)_(?:SECRET|KEY)["\s]*[:=]\s*["\']([A-Za-z0-9_\-]{16,})["\']',
             "OAuth Consumer/Client Secret",
             "Store OAuth secrets in a secure secrets manager. Rotate regularly."),

            # ---- NPM / PyPI tokens ----
            (r'//registry\.npmjs\.org/:_authToken[=\s]+([A-Za-z0-9_-]+)',
             "NPM Registry Auth Token",
             "Remove from source code. Use .npmrc with environment variable references."),

            (r'pypi[-_]?(?:token|password|apikey)["\s]*[:=]\s*["\']([^"\']+)',
             "PyPI Token/Password",
             "Use twine with environment variables or keyring for PyPI uploads."),

            # ---- Docker / CI tokens ----
            (r'(?:docker|registry)\.io.*?(?:password|token)["\s]*[:=]\s*["\']([^"\']+)',
             "Docker Registry Credential",
             "Use Docker credential helpers or environment variables."),

            (r'(?:SONAR|SONARQUBE)_TOKEN["\s]*[:=]\s*["\']([^"\']+)',
             "SonarQube Token",
             "Store in CI/CD secret variables, not in source code."),

            (r'SENTRY[_-]?DSN\s*=\s*["\']https?://[^"\']+["\']',
             "Sentry DSN URL",
             "DSN URLs contain project keys. Move to environment variables."),

            (r'DATADOG[_-]?API[_-]?KEY["\s]*[:=]\s*["\']([^"\']+)',
             "Datadog API Key",
             "Store Datadog API keys in environment variables or secret management."),

            (r'CIRCLECI["\s]*[:=]\s*["\']([A-Za-z0-9_\-]{40,})',
             "CircleCI API Token",
             "Store in CircleCI project environment variables, not in source code."),

            (r'NPM_TOKEN["\s]*[:=]\s*["\']([A-Za-z0-9_\-]{20,})',
             "NPM Publish Token",
             "Store in .npmrc or CI secrets. Never commit to repository."),

            (r'ACTIONS_DEPLOY_KEY|ACTIONS_RUNTIME_TOKEN',
             "GitHub Actions Secret Reference",
             "Use GitHub Secrets instead of hardcoding values."),
        ]

        self._secrets_cache = patterns
        return patterns

    # ------------------------------------------------------------------
    # Dangerous code patterns (20+)
    # ------------------------------------------------------------------

    def _get_dangerous_patterns(self) -> list[tuple[str, str, str, SeverityLevel]]:
        """Return (regex, description, remediation, severity) for dangerous patterns."""
        if self._dangerous_cache is not None:
            return self._dangerous_cache

        patterns: list[tuple[str, str, str, SeverityLevel]] = [
            # Code execution
            (r'\beval\s*\(',
             "eval() allows arbitrary code execution",
             "Avoid eval(). Use ast.literal_eval() for literals, or specific parsers for your data format.",
             SeverityLevel.CRITICAL),

            (r'\bexec\s*\(',
             "exec() allows arbitrary code execution",
             "Avoid exec(). If dynamic code execution is necessary, use sandboxed environments.",
             SeverityLevel.CRITICAL),

            (r'\bcompile\s*\(',
             "compile() can be used for arbitrary code execution",
             "Avoid compile() on user input. Use static analysis or AST-based tools instead.",
             SeverityLevel.HIGH),

            (r'__import__\s*\(\s*["\']',
             "Dynamic import allows loading arbitrary modules",
             "Use explicit imports. If dynamic imports are needed, validate module names against an allowlist.",
             SeverityLevel.HIGH),

            # Command injection
            (r'subprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True',
             "subprocess with shell=True enables command injection",
             "Use shell=False (the default) and pass arguments as a list. Never use shell=True with user input.",
             SeverityLevel.CRITICAL),

            (r'os\.system\s*\(',
             "os.system() passes command to shell",
             "Use subprocess.run([...], shell=False) instead of os.system().",
             SeverityLevel.HIGH),

            (r'os\.popen\s*\(',
             "os.popen() passes command to shell",
             "Use subprocess.run([...], shell=False, capture_output=True) instead.",
             SeverityLevel.HIGH),

            (r'commands\.(?:getoutput|getstatusoutput)\s*\(',
             "commands module passes to shell (deprecated)",
             "Use subprocess module with shell=False.",
             SeverityLevel.HIGH),

            # Network / SSRF
            (r'urllib\.request\.urlopen\s*\(\s*(?:request|req|url)',
             "urlopen with potentially user-controlled URL",
             "Validate and sanitize URLs before fetching. Restrict to allowed protocols and domains.",
             SeverityLevel.MEDIUM),

            (r'requests\.(?:get|post|put|delete|patch)\s*\(\s*(?:url|link)',
             "HTTP request with potentially user-controlled URL (SSRF risk)",
             "Validate URLs against an allowlist. Block private/internal IP ranges.",
             SeverityLevel.MEDIUM),

            (r'socket\.connect\s*\(\s*\(.*?(?:request|input|user|param)',
             "Socket connection with potential user input (SSRF)",
             "Validate the target address. Use allowlists for permitted destinations.",
             SeverityLevel.HIGH),

            # File operations
            (r'open\s*\(\s*(?:request|input|user|param|data|args)',
             "File open with potentially user-controlled path (path traversal)",
             "Use os.path.abspath() and verify paths resolve within an allowed base directory.",
             SeverityLevel.HIGH),

            (r'\bos\.remove\b|\bos\.unlink\b|\bshutil\.rmtree\b',
             "File/directory deletion operation",
             "Validate paths before deletion. Use os.path.realpath() to resolve symlinks.",
             SeverityLevel.MEDIUM),

            (r'\bos\.symlink\b|\bos\.link\b',
             "Symlink creation (potential race condition)",
             "Use os.path.realpath() on the target before creating symlinks. Consider using hardlinks.",
             SeverityLevel.LOW),

            # Debug / info leaks
            (r'(?:print|console\.log|logging\.debug)\s*\(\s*(?:password|secret|token|api_key|credential)',
             "Sensitive data in debug/log output",
             "Never log sensitive data. Use structured logging with field redaction.",
             SeverityLevel.MEDIUM),

            (r'traceback\.print_exc\s*\(\s*\)',
             "Full traceback exposure to end users",
             "Use a custom error handler in production. Log tracebacks server-side only.",
             SeverityLevel.LOW),

            (r'debug\s*=\s*True',
             "Debug mode enabled in configuration",
             "Disable debug mode in production (e.g., DEBUG=False in Django).",
             SeverityLevel.MEDIUM),

            (r'ALLOWED_HOSTS\s*=\s*\[[^\]]*\*\]',
             "Django ALLOWED_HOSTS wildcard",
             "Set ALLOWED_HOSTS to specific hostnames. Wildcard '*' is insecure.",
             SeverityLevel.MEDIUM),

            (r'CORS.*?origin.*?\*',
             "CORS allowing all origins",
             "Restrict CORS origins to specific trusted domains in production.",
             SeverityLevel.MEDIUM),

            (r'(?:verify|ssl_verify|CERT_NONE|cert_reqs\s*=\s*ssl\.CERT_NONE)',
             "SSL/TLS verification disabled",
             "Never disable SSL verification in production. Use proper certificate bundles.",
             SeverityLevel.HIGH),

            (r'tempfile\.mktemp\s*\(',
             "tempfile.mktemp() has a race condition",
             "Use tempfile.mkstemp() or tempfile.NamedTemporaryFile() instead.",
             SeverityLevel.MEDIUM),

            (r'random\.(?:random|randint|choice)\s*\(',
             "random module used for security purposes",
             "Use secrets module for cryptographic randomness. random is not cryptographically secure.",
             SeverityLevel.LOW),

            (r'\bhashlib\.md5\b|\bhashlib\.sha1\b',
             "Weak hash algorithm (MD5 or SHA-1)",
             "Use hashlib.sha256() or stronger. MD5 and SHA-1 are cryptographically broken.",
             SeverityLevel.LOW),

            (r'ftp\.(?:FTP|TLS|SSL)\s*\(',
             "FTP connection (unencrypted protocol)",
             "Use SFTP (paramiko) or HTTPS for file transfers. FTP sends credentials in cleartext.",
             SeverityLevel.MEDIUM),

            (r'telnetlib\b',
             "telnetlib usage (unencrypted protocol)",
             "Use SSH (paramiko/fabric) instead of telnet. Telnet has no encryption.",
             SeverityLevel.MEDIUM),
        ]

        self._dangerous_cache = patterns
        return patterns

    # ------------------------------------------------------------------
    # Dependency checking helpers
    # ------------------------------------------------------------------

    async def _check_python_deps(self, filepath: Path) -> list[Vulnerability]:
        """Check Python requirements.txt for known CVEs."""
        vulns: list[Vulnerability] = []
        try:
            text = filepath.read_text(errors="ignore")
        except Exception:
            return vulns

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue

            # Parse: package[extras]>=1.2.3,<2.0; python_version>='3'
            match = re.match(
                r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)"
                r"(?:\[[^\]]+\])?"
                r"(?:\s*([><=!~]+)\s*([0-9][0-9A-Za-z.*-]*))?",
                line,
            )
            if not match:
                continue

            pkg_name = match.group(1).lower().replace("-", "_")
            version = match.group(4) or "0.0.0"

            pkg_key = pkg_name.replace("-", "").replace("_", "")
            for known_pkg, cves in _KNOWN_CVES.items():
                known_key = known_pkg.replace("-", "").replace("_", "")
                if pkg_key == known_key or pkg_name == known_pkg:
                    for cve in cves:
                        if _version_satisfies(version, cve["version"]):
                            vulns.append(Vulnerability(
                                id=f"DEP-{cve['cve']}",
                                title=cve["title"],
                                severity=cve["severity"],
                                file_path=str(filepath),
                                line_number=None,
                                code_snippet=line,
                                category=VulnerabilityCategory.DEPENDENCY_VULN,
                                description=cve["description"],
                                remediation=cve["remediation"],
                                cwe_id=cve.get("cwe"),
                                confidence=0.95,
                            ))
        return vulns

    async def _check_pipfile_lock(self, filepath: Path) -> list[Vulnerability]:
        """Check Pipfile.lock for known CVEs."""
        vulns: list[Vulnerability] = []
        try:
            data = json.loads(filepath.read_text(errors="ignore"))
        except Exception:
            return vulns

        for section in ("default", "develop"):
            deps = data.get(section, {})
            for pkg_name, info in deps.items():
                if not isinstance(info, dict):
                    continue
                version = info.get("version", "").lstrip("=")
                if not version:
                    continue

                pkg_key = pkg_name.lower().replace("-", "").replace("_", "")
                for known_pkg, cves in _KNOWN_CVES.items():
                    known_key = known_pkg.replace("-", "").replace("_", "")
                    if pkg_key == known_key:
                        for cve in cves:
                            if _version_satisfies(version, cve["version"]):
                                vulns.append(Vulnerability(
                                    id=f"DEP-{cve['cve']}-{pkg_name}",
                                    title=f"{cve['title']} (in {pkg_name})",
                                    severity=cve["severity"],
                                    file_path=str(filepath),
                                    line_number=None,
                                    code_snippet=f"{pkg_name} {version}",
                                    category=VulnerabilityCategory.DEPENDENCY_VULN,
                                    description=cve["description"],
                                    remediation=cve["remediation"],
                                    cwe_id=cve.get("cwe"),
                                    confidence=0.95,
                                ))
        return vulns

    async def _check_toml_deps(self, filepath: Path) -> list[Vulnerability]:
        """Check pyproject.toml for known CVEs."""
        vulns: list[Vulnerability] = []
        try:
            text = filepath.read_text(errors="ignore")
        except Exception:
            return vulns

        # Simple TOML parsing for [project.dependencies] and [tool.poetry.dependencies]
        in_deps = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[project.dependencies]") or stripped.startswith("[tool.poetry.dependencies]"):
                in_deps = True
                continue
            if stripped.startswith("[") and in_deps:
                in_deps = False
            if not in_deps:
                continue
            if stripped.startswith("#") or not stripped:
                continue

            # Parse: package = ">=1.2.3" or package = {version = "1.2.3"}
            match = re.match(
                r'^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)\s*=\s*["\']([^"\']+)["\']',
                stripped,
            )
            if match:
                pkg_name = match.group(1).lower()
                constraint = match.group(3)
                version_match = re.search(r"(\d+\.\d+(?:\.\d+)?)", constraint)
                version = version_match.group(1) if version_match else "0.0.0"

                pkg_key = pkg_name.replace("-", "").replace("_", "")
                for known_pkg, cves in _KNOWN_CVES.items():
                    known_key = known_pkg.replace("-", "").replace("_", "")
                    if pkg_key == known_key:
                        for cve in cves:
                            if _version_satisfies(version, cve["version"]):
                                vulns.append(Vulnerability(
                                    id=f"DEP-{cve['cve']}-toml",
                                    title=f"{cve['title']} (in {pkg_name})",
                                    severity=cve["severity"],
                                    file_path=str(filepath),
                                    line_number=None,
                                    code_snippet=stripped,
                                    category=VulnerabilityCategory.DEPENDENCY_VULN,
                                    description=cve["description"],
                                    remediation=cve["remediation"],
                                    cwe_id=cve.get("cwe"),
                                    confidence=0.9,
                                ))
        return vulns

    async def _check_js_deps(self, filepath: Path) -> list[Vulnerability]:
        """Check package.json dependencies for known CVEs."""
        vulns: list[Vulnerability] = []
        try:
            data = json.loads(filepath.read_text(errors="ignore"))
        except Exception:
            return vulns

        all_deps: dict[str, str] = {}
        for section in ("dependencies", "devDependencies", "peerDependencies"):
            all_deps.update(data.get(section, {}))

        for pkg_name, version in all_deps.items():
            if isinstance(version, str):
                version = version.lstrip("^~>=<=")
            else:
                version = "0.0.0"

            pkg_key = pkg_name.lower().replace("-", "").replace("_", "").replace("@", "")
            for known_pkg, cves in _KNOWN_CVES.items():
                known_key = known_pkg.replace("-", "").replace("_", "")
                if pkg_key == known_key:
                    for cve in cves:
                        if _version_satisfies(version, cve["version"]):
                            vulns.append(Vulnerability(
                                id=f"DEP-{cve['cve']}-js",
                                title=f"{cve['title']} (in {pkg_name})",
                                severity=cve["severity"],
                                file_path=str(filepath),
                                line_number=None,
                                code_snippet=f'"{pkg_name}": "{version}"',
                                category=VulnerabilityCategory.DEPENDENCY_VULN,
                                description=cve["description"],
                                remediation=cve["remediation"],
                                cwe_id=cve.get("cwe"),
                                confidence=0.9,
                            ))
        return vulns

    async def _check_npm_lock(self, filepath: Path) -> list[Vulnerability]:
        """Check package-lock.json for exact versions with CVEs."""
        vulns: list[Vulnerability] = []
        try:
            data = json.loads(filepath.read_text(errors="ignore"))
        except Exception:
            return vulns

        packages = data.get("packages", data.get("dependencies", {}))
        if not isinstance(packages, dict):
            return vulns

        for pkg_path, info in packages.items():
            if not isinstance(info, dict):
                continue
            version = info.get("version", "")
            if not version or not pkg_path.startswith("node_modules/"):
                continue
            pkg_name = pkg_path.replace("node_modules/", "").split("/")[0]

            pkg_key = pkg_name.lower().replace("-", "").replace("_", "").replace("@", "")
            for known_pkg, cves in _KNOWN_CVES.items():
                known_key = known_pkg.replace("-", "").replace("_", "")
                if pkg_key == known_key:
                    for cve in cves:
                        if _version_satisfies(version, cve["version"]):
                            vulns.append(Vulnerability(
                                id=f"DEP-{cve['cve']}-lock",
                                title=f"{cve['title']} (in {pkg_name})",
                                severity=cve["severity"],
                                file_path=str(filepath),
                                line_number=None,
                                code_snippet=f"{pkg_name}@{version}",
                                category=VulnerabilityCategory.DEPENDENCY_VULN,
                                description=cve["description"],
                                remediation=cve["remediation"],
                                cwe_id=cve.get("cwe"),
                                confidence=0.98,
                            ))
        return vulns

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _walk_dir(self, base: Path, max_depth: int = 30) -> list[Path]:
        """Recursively collect files from *base*, skipping ignored paths."""
        result: list[Path] = []
        parts_stack: list[tuple[Path, int]] = [(base, 0)]
        while parts_stack:
            current, depth = parts_stack.pop()
            if depth > max_depth:
                continue
            try:
                entries = list(current.iterdir())
            except PermissionError:
                continue
            for entry in sorted(entries):
                if entry.is_dir():
                    if not self._is_ignored(entry):
                        parts_stack.append((entry, depth + 1))
                elif entry.is_file():
                    if not self._is_ignored(entry):
                        result.append(entry)
        return result

    @staticmethod
    def _mask_secret(snippet: str, match: re.Match) -> str:
        """Mask the matched secret in a code snippet for safe display."""
        start, end = match.start(), match.end()
        value = match.group(0)
        if len(value) <= 8:
            masked_value = "****"
        else:
            masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:]
        return snippet[:start] + masked_value + snippet[end:]

    # ------------------------------------------------------------------
    # Report generators
    # ------------------------------------------------------------------

    def _report_markdown(self, result: ScanResult) -> str:
        """Generate a Markdown-formatted security report."""
        severity_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
            "info": "ℹ️",
        }

        lines: list[str] = [
            f"# Security Scan Report",
            f"",
            f"**Project:** `{result.project_path}`",
            f"**Scan ID:** `{result.scan_id}`",
            f"**Scan Time:** {result.scan_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Files Scanned:** {result.files_scanned}",
            f"**Total Findings:** {result.total_findings}",
            f"",
            f"## Summary",
            f"",
            f"| Severity | Count |",
            f"|----------|-------|",
        ]
        for sev in SeverityLevel:
            count = result.by_severity.get(sev.value, 0)
            emoji = severity_emoji.get(sev.value, "")
            lines.append(f"| {emoji} {sev.value.upper()} | {count} |")

        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for cat, count in sorted(result.by_category.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                lines.append(f"| {cat} | {count} |")

        if result.vulnerabilities:
            lines.append("")
            lines.append("## Findings")
            lines.append("")
            for i, v in enumerate(result.vulnerabilities, 1):
                emoji = severity_emoji.get(v.severity.value, "")
                loc = f"`{v.file_path}:{v.line_number}`" if v.line_number else f"`{v.file_path}`"
                lines.append(f"### {i}. {emoji} [{v.severity.value.upper()}] {v.title}")
                lines.append(f"")
                lines.append(f"- **Location:** {loc}")
                lines.append(f"- **Category:** `{v.category.value}`")
                if v.cwe_id:
                    lines.append(f"- **CWE:** {v.cwe_id}")
                lines.append(f"- **Confidence:** {v.confidence:.0%}")
                if v.code_snippet:
                    lines.append(f"- **Snippet:**")
                    lines.append(f"  ```")
                    lines.append(f"  {v.code_snippet[:200]}")
                    lines.append(f"  ```")
                lines.append(f"- **Description:** {v.description}")
                lines.append(f"- **Remediation:** {v.remediation}")
                lines.append("")

        lines.append("---")
        lines.append(f"*Generated by Claude Security Scanner*")
        return "\n".join(lines)

    def _report_json(self, result: ScanResult) -> str:
        """Generate a JSON-formatted security report."""
        vuln_dicts = []
        for v in result.vulnerabilities:
            vuln_dicts.append({
                "id": v.id,
                "title": v.title,
                "severity": v.severity.value,
                "file_path": v.file_path,
                "line_number": v.line_number,
                "code_snippet": v.code_snippet,
                "category": v.category.value,
                "description": v.description,
                "remediation": v.remediation,
                "cwe_id": v.cwe_id,
                "confidence": v.confidence,
            })
        output = {
            "scan_id": result.scan_id,
            "project_path": result.project_path,
            "scan_time": result.scan_time.isoformat(),
            "files_scanned": result.files_scanned,
            "total_findings": result.total_findings,
            "by_severity": result.by_severity,
            "by_category": result.by_category,
            "vulnerabilities": vuln_dicts,
        }
        return json.dumps(output, indent=2, default=str)

    def _report_text(self, result: ScanResult) -> str:
        """Generate a plain-text security report."""
        divider = "=" * 72
        lines: list[str] = [
            divider,
            "  SECURITY SCAN REPORT",
            divider,
            f"  Project:     {result.project_path}",
            f"  Scan ID:     {result.scan_id}",
            f"  Scan Time:   {result.scan_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"  Files:       {result.files_scanned}",
            f"  Findings:    {result.total_findings}",
            divider,
            "",
            "  SEVERITY SUMMARY",
            "  " + "-" * 40,
        ]
        for sev in SeverityLevel:
            count = result.by_severity.get(sev.value, 0)
            bar = "#" * count
            lines.append(f"  {sev.value.upper():<12} {count:>4}  {bar}")

        lines.append("")
        lines.append("  CATEGORY SUMMARY")
        lines.append("  " + "-" * 40)
        for cat, count in sorted(result.by_category.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                lines.append(f"  {cat:<30} {count:>4}")

        if result.vulnerabilities:
            lines.append("")
            lines.append(divider)
            lines.append("  DETAILED FINDINGS")
            lines.append(divider)
            for i, v in enumerate(result.vulnerabilities, 1):
                loc = f"{v.file_path}:{v.line_number}" if v.line_number else v.file_path
                lines.append("")
                lines.append(f"  [{i}] {v.severity.value.upper()}: {v.title}")
                lines.append(f"      Location:   {loc}")
                lines.append(f"      Category:   {v.category.value}")
                if v.cwe_id:
                    lines.append(f"      CWE:        {v.cwe_id}")
                if v.code_snippet:
                    lines.append(f"      Code:       {v.code_snippet[:120]}")
                lines.append(f"      Fix:        {v.remediation[:120]}")

        lines.append("")
        lines.append(divider)
        return "\n".join(lines)
