"""
PII Redaction — Detection and reversible redaction of sensitive data.

Detects and redacts personally identifiable information (PII) including
API keys, email addresses, phone numbers, IP addresses, credit card numbers,
and other sensitive patterns. Supports reversible redaction with a mapping
storage.

Usage
-----
    redactor = PIIRedactor()
    text = "Contact me at john@example.com or call 555-123-4567"
    redacted, mapping = redactor.redact(text)
    # redacted = "Contact me at [REDACTED_EMAIL_0] or call [REDACTED_PHONE_0]"
    original = redactor.restore(redacted, mapping)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple

from hermes.constants import (
    HERMES_CONFIG_HOME,
    REDACTION_MAP_FILE,
    REDACTION_PLACEHOLDER_PREFIX,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class PIICategory(Enum):
    """Categories of personally identifiable information."""

    API_KEY = auto()
    EMAIL = auto()
    PHONE = auto()
    IP_ADDRESS = auto()
    CREDIT_CARD = auto()
    SSN = auto()
    DATE_OF_BIRTH = auto()
    ADDRESS = auto()
    PASSPORT = auto()
    CUSTOM = auto()


@dataclass
class RedactionPattern:
    """
    A configurable redaction pattern.

    Attributes
    ----------
    name:
        Human-readable name (used in placeholder).
    pattern:
        Compiled regex pattern.
    category:
        PII category.
    enabled:
        Whether this pattern is active.
    description:
        Description of what this pattern matches.
    """

    name: str
    pattern: Pattern
    category: PIICategory = PIICategory.CUSTOM
    enabled: bool = True
    description: str = ""


@dataclass
class RedactionConfig:
    """
    Configuration for the PII redactor.

    Attributes
    ----------
    enabled_categories:
        Set of PII categories to detect. If empty, all are enabled.
    custom_patterns:
        Additional custom patterns to include.
    disabled_patterns:
        Names of built-in patterns to disable.
    preserve_format:
        Whether to preserve format hints in placeholders.
    auto_save:
        Whether to auto-save the redaction map.
    max_pattern_length:
        Maximum match length to redact (prevents false positives on long text).
    """

    enabled_categories: set = field(default_factory=set)
    custom_patterns: List[RedactionPattern] = field(default_factory=list)
    disabled_patterns: List[str] = field(default_factory=set)
    preserve_format: bool = True
    auto_save: bool = True
    max_pattern_length: int = 200


# ──────────────────────────────────────────────────────────────────────────────
# Built-in patterns
# ──────────────────────────────────────────────────────────────────────────────

BUILTIN_PATTERNS: List[RedactionPattern] = [
    # API Keys
    RedactionPattern(
        name="openai_api_key",
        pattern=re.compile(r"(?i)sk-[a-zA-Z0-9]{20,}"),
        category=PIICategory.API_KEY,
        description="OpenAI API key",
    ),
    RedactionPattern(
        name="anthropic_api_key",
        pattern=re.compile(r"(?i)sk-ant-[a-zA-Z0-9-]{20,}"),
        category=PIICategory.API_KEY,
        description="Anthropic API key",
    ),
    RedactionPattern(
        name="generic_api_key",
        pattern=re.compile(r"(?i)(api[_-]?key|apikey|secret|token|password)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?"),
        category=PIICategory.API_KEY,
        description="Generic API key/secret in assignment",
    ),
    RedactionPattern(
        name="bearer_token",
        pattern=re.compile(r"(?i)Bearer\s+[a-zA-Z0-9_\-\.]{20,}"),
        category=PIICategory.API_KEY,
        description="Bearer token",
    ),

    # Email addresses
    RedactionPattern(
        name="email",
        pattern=re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
        category=PIICategory.EMAIL,
        description="Email address",
    ),

    # Phone numbers (US and international)
    RedactionPattern(
        name="phone_us",
        pattern=re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        category=PIICategory.PHONE,
        description="US phone number",
    ),
    RedactionPattern(
        name="phone_international",
        pattern=re.compile(r"\b\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"),
        category=PIICategory.PHONE,
        description="International phone number",
    ),

    # IP addresses
    RedactionPattern(
        name="ipv4",
        pattern=re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        category=PIICategory.IP_ADDRESS,
        description="IPv4 address",
    ),
    RedactionPattern(
        name="ipv6",
        pattern=re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"),
        category=PIICategory.IP_ADDRESS,
        description="IPv6 address (full)",
    ),

    # Credit card numbers (basic Luhn-like pattern)
    RedactionPattern(
        name="credit_card",
        pattern=re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        category=PIICategory.CREDIT_CARD,
        description="Potential credit card number",
    ),

    # SSN (US Social Security Number)
    RedactionPattern(
        name="ssn",
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        category=PIICategory.SSN,
        description="US Social Security Number",
    ),

    # AWS access keys
    RedactionPattern(
        name="aws_access_key",
        pattern=re.compile(r"(?i)(?:AKIA|ASIA)[A-Z0-9]{16}"),
        category=PIICategory.API_KEY,
        description="AWS access key",
    ),

    # Private keys
    RedactionPattern(
        name="private_key",
        pattern=re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----"),
        category=PIICategory.API_KEY,
        description="PEM private key block",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# PIIRedactor
# ──────────────────────────────────────────────────────────────────────────────

class PIIRedactor:
    """
    PII detection and reversible redaction engine.

    Scans text for sensitive data patterns, replaces matches with
    placeholder tokens, and maintains a mapping for restoration.

    Parameters
    ----------
    config:
        Redaction configuration. If ``None``, uses defaults (all categories enabled).
    persistence_path:
        Path to persist the redaction map. If ``None``, uses default.
    """

    def __init__(
        self,
        config: Optional[RedactionConfig] = None,
        persistence_path: Optional[str] = None,
    ) -> None:
        self._config = config or RedactionConfig()
        self._persistence_path = (
            Path(persistence_path) if persistence_path
            else HERMES_CONFIG_HOME / REDACTION_MAP_FILE
        )

        # Build active patterns
        self._patterns: List[RedactionPattern] = []
        self._build_patterns()

        # Counter for generating unique placeholders
        self._counter: Dict[str, int] = {}

    def _build_patterns(self) -> None:
        """Build the active pattern list from config."""
        enabled_categories = self._config.enabled_categories
        disabled = set(self._config.disabled_patterns)

        for pattern in BUILTIN_PATTERNS:
            if pattern.name in disabled:
                continue
            if enabled_categories and pattern.category not in enabled_categories:
                continue
            self._patterns.append(pattern)

        # Add custom patterns
        for pattern in self._config.custom_patterns:
            if pattern.name not in disabled and pattern.enabled:
                self._patterns.append(pattern)

    # ── Public API ────────────────────────────────────────────────────────

    def redact(
        self,
        text: str,
        mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """
        Detect and redact PII in text.

        Parameters
        ----------
        text:
            The text to scan and redact.
        mapping:
            Optional existing mapping to extend. If ``None``, creates a new one.

        Returns
        -------
        tuple[str, dict]
            (redacted_text, mapping_dict) where the mapping can be used
            with ``restore()`` to reverse the redaction.
        """
        if mapping is None:
            mapping = {}

        redacted = text
        for pattern_def in self._patterns:
            redacted = self._apply_pattern(redacted, pattern_def, mapping)

        return redacted, mapping

    def restore(
        self,
        redacted_text: str,
        mapping: Dict[str, str],
    ) -> str:
        """
        Restore original values from redacted text.

        Parameters
        ----------
        redacted_text:
            Text with placeholders.
        mapping:
            The mapping produced by ``redact()``.

        Returns
        -------
        str
            Text with original values restored.
        """
        restored = redacted_text
        for placeholder, original in mapping.items():
            restored = restored.replace(placeholder, original)
        return restored

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text without redacting.

        Parameters
        ----------
        text:
            The text to scan.

        Returns
        -------
        list[dict]
            List of detected PII items with ``category``, ``pattern_name``,
            ``match``, ``start``, and ``end`` keys.
        """
        detections: List[Dict[str, Any]] = []

        for pattern_def in self._patterns:
            for match in pattern_def.pattern.finditer(text):
                matched_text = match.group()
                # Skip very long matches (likely false positives)
                if len(matched_text) > self._config.max_pattern_length:
                    continue
                detections.append({
                    "category": pattern_def.category.value,
                    "pattern_name": pattern_def.name,
                    "description": pattern_def.description,
                    "match": matched_text,
                    "start": match.start(),
                    "end": match.end(),
                })

        return detections

    def is_sensitive(self, text: str) -> bool:
        """
        Check if text contains any PII.

        Parameters
        ----------
        text:
            The text to check.

        Returns
        -------
        bool
            ``True`` if any PII patterns were detected.
        """
        for pattern_def in self._patterns:
            if pattern_def.pattern.search(text):
                return True
        return False

    # ── Pattern management ────────────────────────────────────────────────

    def add_pattern(self, pattern: RedactionPattern) -> None:
        """Add a custom detection pattern."""
        self._patterns.append(pattern)

    def disable_pattern(self, name: str) -> None:
        """Disable a pattern by name."""
        self._patterns = [p for p in self._patterns if p.name != name]

    def enable_category(self, category: PIICategory) -> None:
        """Enable all patterns for a category."""
        for pattern in BUILTIN_PATTERNS:
            if pattern.category == category and pattern not in self._patterns:
                self._patterns.append(pattern)

    def disable_category(self, category: PIICategory) -> None:
        """Disable all patterns for a category."""
        self._patterns = [p for p in self._patterns if p.category != category]

    # ── Persistence ───────────────────────────────────────────────────────

    async def save_mapping(self, mapping: Dict[str, str]) -> None:
        """
        Persist a redaction mapping to disk.

        Note: Mappings contain the original sensitive values and should
        be stored securely. The file is written with restrictive permissions.
        """
        import asyncio

        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

        def _write():
            self._persistence_path.write_text(
                json.dumps(mapping, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # Set restrictive permissions (owner read/write only)
            try:
                import os
                os.chmod(str(self._persistence_path), 0o600)
            except (OSError, AttributeError):
                pass

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write)

    async def load_mapping(self) -> Dict[str, str]:
        """
        Load a redaction mapping from disk.

        Returns
        -------
        dict[str, str]
            The loaded mapping, or an empty dict if the file doesn't exist.
        """
        import asyncio

        if not self._persistence_path.exists():
            return {}

        def _read():
            return json.loads(
                self._persistence_path.read_text(encoding="utf-8")
            )

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _read)
        except Exception as e:
            logger.warning("Failed to load redaction mapping: %s", e)
            return {}

    # ── Internal ──────────────────────────────────────────────────────────

    def _apply_pattern(
        self,
        text: str,
        pattern_def: RedactionPattern,
        mapping: Dict[str, str],
    ) -> str:
        """
        Apply a single redaction pattern to text.

        Replaces all matches with unique placeholders and updates the mapping.
        """
        result = text

        for match in pattern_def.pattern.finditer(result):
            matched_text = match.group()

            # Skip very long matches
            if len(matched_text) > self._config.max_pattern_length:
                continue

            # Generate unique placeholder
            counter_key = pattern_def.category.value
            self._counter[counter_key] = self._counter.get(counter_key, 0)
            idx = self._counter[counter_key]
            self._counter[counter_key] += 1

            placeholder = f"{REDACTION_PLACEHOLDER_PREFIX}_{pattern_def.category.name}_{idx}]"

            # Only replace if this exact text hasn't been mapped yet
            if placeholder not in mapping:
                mapping[placeholder] = matched_text
                result = result.replace(matched_text, placeholder, 1)
            else:
                # Reuse existing placeholder
                existing_placeholder = next(
                    (k for k, v in mapping.items() if v == matched_text),
                    placeholder,
                )
                result = result.replace(matched_text, existing_placeholder, 1)

        return result
