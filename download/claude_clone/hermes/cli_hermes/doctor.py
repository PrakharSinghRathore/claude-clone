"""
Diagnostic tool (doctor) for Hermes CLI.

System health check, dependency verification, API key validation,
configuration audit, performance benchmark, and issue fixing.
"""

import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes.cli_hermes.config_manager import ConfigManager, DEFAULT_CONFIG_DIR

try:
    import importlib.metadata as metadata
    HAS_METADATA = True
except ImportError:
    try:
        import importlib_metadata as metadata
        HAS_METADATA = True
    except ImportError:
        HAS_METADATA = False


class DiagnosticResult:
    """A single diagnostic check result."""

    def __init__(self, name: str, status: str, message: str, details: Optional[str] = None):
        self.name = name
        self.status = status  # "ok", "warning", "error", "skip"
        self.message = message
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


class Doctor:
    """System diagnostic tool for Hermes CLI."""

    REQUIRED_PACKAGES = {
        "anthropic": {"required": False, "description": "Anthropic SDK"},
        "httpx": {"required": True, "description": "HTTP client"},
        "prompt_toolkit": {"required": True, "description": "Terminal UI toolkit"},
        "rich": {"required": True, "description": "Rich text formatting"},
        "pyyaml": {"required": False, "description": "YAML support"},
        "chardet": {"required": False, "description": "Character encoding detection"},
    }

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.config.load()
        self._results: List[DiagnosticResult] = []

    def run_all(self) -> List[DiagnosticResult]:
        """Run all diagnostic checks."""
        self._results = []

        self.check_python_version()
        self.check_operating_system()
        self.check_dependencies()
        self.check_config_directory()
        self.check_config_file()
        self.check_api_keys()
        self.check_network_connectivity()
        self.check_model_access()
        self.check_terminal()
        self.check_disk_space()
        self.check_memory()
        self.check_performance()
        self.check_theme()

        return self._results

    def run_quick(self) -> List[DiagnosticResult]:
        """Run quick essential checks."""
        self._results = []
        self.check_python_version()
        self.check_dependencies()
        self.check_config_file()
        self.check_api_keys()
        return self._results

    def fix_issues(self) -> List[str]:
        """Attempt to fix common issues."""
        fixes = []

        # Ensure config directory exists
        try:
            DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            fixes.append("Created config directory")
        except Exception as e:
            fixes.append(f"Failed to create config directory: {e}")

        # Ensure hermes subdirectories exist
        hermes_dir = DEFAULT_CONFIG_DIR / "hermes"
        for subdir in ["profiles", "themes", "sessions", "cron", "skills"]:
            (hermes_dir / subdir).mkdir(parents=True, exist_ok=True)
        fixes.append("Created hermes subdirectories")

        # Validate and fix config
        self.config.load()
        errors, warnings = self.config.validate()
        if errors:
            self.config.save()
            fixes.append("Saved validated configuration")

        # Migrate old config if needed
        if self.config.migrate():
            fixes.append("Migrated configuration to new format")

        return fixes

    def generate_report(self) -> str:
        """Generate a comprehensive diagnostic report."""
        self.run_all()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "=" * 60,
            "  Hermes CLI Diagnostic Report",
            f"  Generated: {now}",
            "=" * 60,
            "",
        ]

        # Summary
        ok = sum(1 for r in self._results if r.status == "ok")
        warnings = sum(1 for r in self._results if r.status == "warning")
        errors = sum(1 for r in self._results if r.status == "error")
        skipped = sum(1 for r in self._results if r.status == "skip")

        lines.append(f"  Results: {ok} OK, {warnings} warnings, {errors} errors, {skipped} skipped")
        lines.append("")

        # System info
        lines.append("  System Information:")
        lines.append(f"    Python:   {sys.version}")
        lines.append(f"    OS:       {platform.system()} {platform.release()}")
        lines.append(f"    Machine:  {platform.machine()}")
        lines.append(f"    Terminal: {os.environ.get('TERM', 'unknown')}")
        lines.append("")

        # Check results
        lines.append("  Check Results:")
        lines.append("  " + "-" * 50)

        status_icons = {
            "ok": "\033[32m  \u2713\033[0m",
            "warning": "\033[33m  \u0021\033[0m",
            "error": "\033[31m  \u2717\033[0m",
            "skip": "\033[90m  \u2022\033[0m",
        }

        for result in self._results:
            icon = status_icons.get(result.status, "  ?")
            lines.append(f"  {icon} {result.name}: {result.message}")
            if result.details:
                for line in result.details.split("\n"):
                    lines.append(f"      {line}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    # ─── Individual checks ────────────────────

    def check_python_version(self):
        """Check Python version compatibility."""
        major = sys.version_info.major
        minor = sys.version_info.minor

        if major == 3 and minor >= 9:
            self._results.append(DiagnosticResult(
                "Python Version",
                "ok",
                f"Python {major}.{minor}.{sys.version_info.micro}",
            ))
        elif major == 3 and minor >= 8:
            self._results.append(DiagnosticResult(
                "Python Version",
                "warning",
                f"Python {major}.{minor} - upgrade to 3.9+ recommended",
            ))
        else:
            self._results.append(DiagnosticResult(
                "Python Version",
                "error",
                f"Python {major}.{minor} - requires 3.8+",
            ))

    def check_operating_system(self):
        """Check OS compatibility."""
        system = platform.system()
        supported = ["Linux", "Darwin", "Windows"]

        if system in supported:
            self._results.append(DiagnosticResult(
                "Operating System",
                "ok",
                f"{system} {platform.release()}",
            ))
        else:
            self._results.append(DiagnosticResult(
                "Operating System",
                "warning",
                f"{system} - may have limited support",
            ))

    def check_dependencies(self):
        """Check required Python packages."""
        installed = set()
        if HAS_METADATA:
            try:
                installed = {p.metadata["name"].lower() for p in metadata.distributions()}
            except Exception:
                try:
                    installed = {d.metadata["Name"].lower() for d in metadata.distributions()}
                except Exception:
                    pass

        for package, info in self.REQUIRED_PACKAGES.items():
            # Also check by import name
            try:
                __import__(package.replace("-", "_").replace("pyyaml", "yaml"))
                status = "ok"
                msg = f"{info['description']} installed"
            except ImportError:
                # Check metadata
                if package.lower() in installed or package.replace("-", "_").lower() in installed:
                    status = "ok"
                    msg = f"{info['description']} installed"
                elif info["required"]:
                    status = "error"
                    msg = f"{info['description']} NOT installed (required)"
                else:
                    status = "warning"
                    msg = f"{info['description']} NOT installed (optional)"

            self._results.append(DiagnosticResult(
                f"Dependency: {package}",
                status,
                msg,
            ))

    def check_config_directory(self):
        """Check config directory exists and is accessible."""
        if DEFAULT_CONFIG_DIR.exists():
            if os.access(DEFAULT_CONFIG_DIR, os.W_OK):
                self._results.append(DiagnosticResult(
                    "Config Directory",
                    "ok",
                    str(DEFAULT_CONFIG_DIR),
                ))
            else:
                self._results.append(DiagnosticResult(
                    "Config Directory",
                    "error",
                    f"Not writable: {DEFAULT_CONFIG_DIR}",
                ))
        else:
            self._results.append(DiagnosticResult(
                "Config Directory",
                "warning",
                f"Does not exist: {DEFAULT_CONFIG_DIR}",
                "Run /setup or /doctor fix to create it",
            ))

    def check_config_file(self):
        """Check config file is valid."""
        config_file = DEFAULT_CONFIG_DIR / "config.json"
        hermes_config = DEFAULT_CONFIG_DIR / "hermes" / "config.yaml"

        if config_file.exists() or hermes_config.exists():
            try:
                errors, warnings = self.config.validate()
                if errors:
                    self._results.append(DiagnosticResult(
                        "Config File",
                        "warning",
                        f"Config has {len(errors)} validation issues",
                        "\n".join(errors),
                    ))
                elif warnings:
                    self._results.append(DiagnosticResult(
                        "Config File",
                        "ok",
                        f"Config valid with {len(warnings)} warnings",
                    ))
                else:
                    self._results.append(DiagnosticResult(
                        "Config File",
                        "ok",
                        "Configuration valid",
                    ))
            except Exception as e:
                self._results.append(DiagnosticResult(
                    "Config File",
                    "error",
                    f"Failed to read config: {e}",
                ))
        else:
            self._results.append(DiagnosticResult(
                "Config File",
                "warning",
                "No config file found - run /setup",
            ))

    def check_api_keys(self):
        """Check API keys are configured."""
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        config_key = self.config.get("api_key", "")

        has_any = bool(openrouter_key or anthropic_key or config_key)

        if has_any:
            sources = []
            if openrouter_key:
                sources.append("OPENROUTER_API_KEY")
            if anthropic_key:
                sources.append("ANTHROPIC_API_KEY")
            if config_key and config_key not in (openrouter_key, anthropic_key):
                sources.append("config file")

            self._results.append(DiagnosticResult(
                "API Keys",
                "ok",
                f"Keys found: {', '.join(sources)}",
            ))
        else:
            self._results.append(DiagnosticResult(
                "API Keys",
                "error",
                "No API keys configured",
                "Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY, or run /setup",
            ))

    def check_network_connectivity(self):
        """Check basic network connectivity."""
        try:
            import httpx
            start = time.time()
            with httpx.Client(timeout=5) as client:
                resp = client.get("https://httpbin.org/get")
                elapsed = time.time() - start
                if resp.status_code == 200:
                    self._results.append(DiagnosticResult(
                        "Network",
                        "ok",
                        f"Connected ({elapsed*1000:.0f}ms)",
                    ))
                else:
                    self._results.append(DiagnosticResult(
                        "Network",
                        "warning",
                        f"HTTP {resp.status_code}",
                    ))
        except ImportError:
            self._results.append(DiagnosticResult(
                "Network",
                "skip",
                "httpx not installed",
            ))
        except Exception as e:
            self._results.append(DiagnosticResult(
                "Network",
                "warning",
                f"Cannot reach internet: {e}",
            ))

    def check_model_access(self):
        """Check if configured model is accessible."""
        api_key = self.config.get("api_key") or os.environ.get(
            "OPENROUTER_API_KEY", ""
        ) or os.environ.get("ANTHROPIC_API_KEY", ""
        )

        if not api_key:
            self._results.append(DiagnosticResult(
                "Model Access",
                "skip",
                "No API key to test",
            ))
            return

        model = self.config.get("model", "anthropic/claude-sonnet-4-20250514")
        base_url = self.config.get("base_url", "https://openrouter.ai/api/v1")

        try:
            import httpx
            start = time.time()
            with httpx.Client(timeout=15) as client:
                resp = client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 3,
                    },
                )
            elapsed = time.time() - start

            if resp.status_code == 200:
                self._results.append(DiagnosticResult(
                    "Model Access",
                    "ok",
                    f"{model} ({elapsed:.1f}s)",
                ))
            elif resp.status_code == 401:
                self._results.append(DiagnosticResult(
                    "Model Access",
                    "error",
                    f"Authentication failed for {model}",
                ))
            elif resp.status_code == 429:
                self._results.append(DiagnosticResult(
                    "Model Access",
                    "warning",
                    f"Rate limited for {model}",
                ))
            else:
                self._results.append(DiagnosticResult(
                    "Model Access",
                    "warning",
                    f"HTTP {resp.status_code} for {model}",
                ))
        except Exception as e:
            self._results.append(DiagnosticResult(
                "Model Access",
                "error",
                f"Cannot reach API: {e}",
            ))

    def check_terminal(self):
        """Check terminal capabilities."""
        term = os.environ.get("TERM", "")
        is_tty = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else False
        cols = os.environ.get("COLUMNS", "?")

        if is_tty:
            self._results.append(DiagnosticResult(
                "Terminal",
                "ok",
                f"TTY: {term}, cols: {cols}",
            ))
        else:
            self._results.append(DiagnosticResult(
                "Terminal",
                "warning",
                f"Not a TTY: {term}",
                "Some features may not work in non-interactive mode",
            ))

    def check_disk_space(self):
        """Check available disk space."""
        try:
            stat = os.statvfs(str(DEFAULT_CONFIG_DIR))
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            used_pct = (1 - free / total) * 100 if total > 0 else 100

            if free > 100_000_000:  # > 100MB
                self._results.append(DiagnosticResult(
                    "Disk Space",
                    "ok",
                    f"{free // 1_000_000}MB free ({used_pct:.0f}% used)",
                ))
            elif free > 10_000_000:  # > 10MB
                self._results.append(DiagnosticResult(
                    "Disk Space",
                    "warning",
                    f"Low space: {free // 1_000_000}MB free",
                ))
            else:
                self._results.append(DiagnosticResult(
                    "Disk Space",
                    "error",
                    f"Very low space: {free // 1_000_000}MB free",
                ))
        except Exception:
            self._results.append(DiagnosticResult(
                "Disk Space",
                "skip",
                "Cannot check disk space",
            ))

    def check_memory(self):
        """Check system memory."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            available_gb = mem.available / (1024 ** 3)
            used_pct = mem.percent

            if available_gb > 1:
                self._results.append(DiagnosticResult(
                    "Memory",
                    "ok",
                    f"{available_gb:.1f}GB available / {total_gb:.1f}GB total",
                ))
            elif available_gb > 0.5:
                self._results.append(DiagnosticResult(
                    "Memory",
                    "warning",
                    f"Low memory: {available_gb:.1f}GB available",
                ))
            else:
                self._results.append(DiagnosticResult(
                    "Memory",
                    "error",
                    f"Very low memory: {available_gb:.1f}GB available",
                ))
        except ImportError:
            self._results.append(DiagnosticResult(
                "Memory",
                "skip",
                "psutil not installed",
            ))

    def check_performance(self):
        """Run a quick performance benchmark."""
        start = time.time()

        # CPU benchmark: simple computation
        total = sum(i * i for i in range(100000))
        cpu_time = time.time() - start

        # IO benchmark: file write
        import tempfile
        tmp = Path(tempfile.mktemp())
        try:
            start_io = time.time()
            tmp.write_text("x" * 1000000)
            io_time = time.time() - start_io
            tmp.unlink()
        except Exception:
            io_time = -1

        if cpu_time < 0.1:
            self._results.append(DiagnosticResult(
                "Performance",
                "ok",
                f"CPU: {cpu_time*1000:.0f}ms, IO: {io_time*1000:.0f}ms",
            ))
        elif cpu_time < 0.5:
            self._results.append(DiagnosticResult(
                "Performance",
                "ok",
                f"CPU: {cpu_time*1000:.0f}ms (acceptable)",
            ))
        else:
            self._results.append(DiagnosticResult(
                "Performance",
                "warning",
                f"CPU: {cpu_time*1000:.0f}ms (slow)",
            ))

    def check_theme(self):
        """Check theme configuration."""
        theme = self.config.get("theme", "dark")
        skin = self.config.get("skin", "dark")

        if theme in ("dark", "light"):
            self._results.append(DiagnosticResult(
                "Theme",
                "ok",
                f"Theme: {theme}, Skin: {skin}",
            ))
        else:
            # Check if custom theme exists
            theme_file = Path.home() / ".claude_clone" / "hermes" / "themes" / f"{skin}.yaml"
            if theme_file.exists():
                self._results.append(DiagnosticResult(
                    "Theme",
                    "ok",
                    f"Custom theme: {skin}",
                ))
            else:
                self._results.append(DiagnosticResult(
                    "Theme",
                    "warning",
                    f"Unknown theme: {theme}",
                ))

    def format_results(self, results: Optional[List[DiagnosticResult]] = None) -> str:
        """Format diagnostic results as a table."""
        results = results or self._results

        lines = []
        status_markers = {
            "ok": "\033[32m\u2713\033[0m",
            "warning": "\033[33m!\033[0m",
            "error": "\033[31m\u2717\033[0m",
            "skip": "\033[90m\u2022\033[0m",
        }

        for r in results:
            marker = status_markers.get(r.status, "?")
            lines.append(f"  {marker} {r.name}: {r.message}")
            if r.details:
                for line in r.details.split("\n"):
                    lines.append(f"    {line}")

        return "\n".join(lines)
