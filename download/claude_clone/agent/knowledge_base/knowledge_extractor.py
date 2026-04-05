"""
Knowledge Extractor Module for Claude Code Clone.

Automatically mines knowledge entries from conversations, source code, and
arbitrary text content using regex-based pattern detection and heuristic
analysis.  Extracted knowledge is returned as structured
:class:`ExtractionResult` objects that can be persisted via a
:class:`KnowledgeStore`.

No machine-learning models are required — all extraction is performed with
deterministic pattern matching, making the module lightweight, fast, and
easy to debug.

Typical usage::

    from agent.knowledge_base.knowledge_extractor import KnowledgeExtractor
    from agent.knowledge_base.knowledge_store import KnowledgeStore

    store = KnowledgeStore()
    await store.initialize()
    extractor = KnowledgeExtractor(store)

    messages = [
        {"role": "user", "content": "How do I fix a KeyError in Python?"},
        {"role": "assistant", "content": "The fix is to use dict.get()..."},
    ]
    results = extractor.extract_from_conversation(messages)
    stats = await extractor.auto_extract_and_store(messages=messages)
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.knowledge_base.knowledge_store import KnowledgeEntry, KnowledgeStore

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class PatternMatch:
    """Represents a single pattern detected during extraction.

    Attributes
    ----------
    pattern_type:
        The category of pattern that was matched (e.g. ``"solution"``,
        ``"error_solution"``, ``"decision"``, ``"preference"``,
        ``"troubleshooting"``, ``"code_pattern"``).
    content:
        The primary content or text that was matched.
    context:
        Surrounding text that provides context for the matched content.
    confidence:
        A float from 0.0 to 1.0 indicating how confident the extractor is
        that this is a genuine knowledge entry.  Values above 0.7 are
        considered high-confidence.
    """

    pattern_type: str
    content: str
    context: str
    confidence: float


@dataclass
class ExtractionResult:
    """Aggregates one or more pattern matches into a coherent extraction result.

    Each result corresponds to a distinct piece of knowledge that can be
    stored as a :class:`KnowledgeEntry` in the knowledge base.

    Attributes
    ----------
    entries:
        A list of :class:`PatternMatch` objects found in this extraction.
    confidence:
        The average confidence across all contained pattern matches, or a
        single match confidence when only one pattern matched.
    source_type:
        The origin of the extraction — ``"conversation"``, ``"code"``,
        ``"text"``, or ``"markdown"``.
    metadata:
        Arbitrary key-value metadata attached to the extraction (e.g. file
        path, message indices, line numbers).
    """

    entries: List[PatternMatch] = field(default_factory=list)
    confidence: float = 0.0
    source_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- Convenience helpers --------------------------------------------------

    @property
    def summary(self) -> str:
        """Return a one-line summary built from the first entry's content."""
        if not self.entries:
            return ""
        text = self.entries[0].content
        if len(text) > 200:
            return text[:197] + "..."
        return text

    @property
    def tags(self) -> List[str]:
        """Derive tags from pattern types of contained entries."""
        tags: List[str] = []
        for entry in self.entries:
            tag = entry.pattern_type.replace("_", "-")
            if tag not in tags:
                tags.append(tag)
        return tags


# ---------------------------------------------------------------------------
# Pattern Definitions (regex + heuristic constants)
# ---------------------------------------------------------------------------

# -- Conversation patterns ---------------------------------------------------

# Phrases that signal the assistant is providing a solution or fix.
_SOLUTION_INDICATORS: List[re.Pattern[str]] = [
    re.compile(r"(?:here(?:'s| is) (?:how|the|a|your|one way|an?))", re.IGNORECASE),
    re.compile(r"(?:the (?:fix|solution|answer|issue|problem) (?:is|was|would be))", re.IGNORECASE),
    re.compile(r"(?:you (?:need to|should|can|must|have to|want to))", re.IGNORECASE),
    re.compile(r"(?:to (?:fix|resolve|solve|handle|address) (?:this|that|the|your))", re.IGNORECASE),
    re.compile(r"(?:try (?:using|adding|changing|replacing|setting))", re.IGNORECASE),
    re.compile(r"(?:the (?:correct|right|proper|best) (?:way|approach|method))", re.IGNORECASE),
    re.compile(r"(?:simply (?:use|add|remove|change|wrap))", re.IGNORECASE),
    re.compile(r"(?:a (?:common (?:fix|solution|approach|pattern))", re.IGNORECASE),
]

# Error message patterns (tracebacks, error names, etc.).
_ERROR_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(?:Traceback \(most recent call last\))", re.IGNORECASE),
    re.compile(r"(?:Error(?:\s+(?:at|in|on|\:))?\s*\:?\s*.+)", re.IGNORECASE),
    re.compile(r"(?:(?:Exception|TypeError|ValueError|KeyError|AttributeError|"
               r"IndexError|RuntimeError|ImportError|ModuleNotFoundError|"
               r"FileNotFoundError|PermissionError|ConnectionError|"
               r"TimeoutError|OSError|SyntaxError|NameError|"
               r"UnicodeDecodeError|UnicodeEncodeError|NotImplementedError|"
               r"StopIteration|RecursionError|MemoryError|ZeroDivisionError"
               r")\s*[\(:])"),
    re.compile(r"(?:(?:FAILED|ERROR|CRITICAL|WARNING)\s*\:.+)", re.IGNORECASE),
    re.compile(r"(?:assertion\s+(?:error|failed))", re.IGNORECASE),
    re.compile(r"(?:SEGFAULT|segfault|Segmentation fault)", re.IGNORECASE),
    re.compile(r"(?:undefined\s+(?:is\s+not|reference|symbol))", re.IGNORECASE),
    re.compile(r"(?:cannot\s+(?:read|find|import|resolve|open|connect))", re.IGNORECASE),
    re.compile(r"(?:no (?:such|module|file|directory|attribute|method))", re.IGNORECASE),
    re.compile(r"(?:unexpected\s+(?:token|indent|eof|end|character|symbol))", re.IGNORECASE),
]

# Decision/choice indicators.
_DECISION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(?:decided to|decision to|we decided|i decided)", re.IGNORECASE),
    re.compile(r"(?:going (?:to go |with ))", re.IGNORECASE),
    re.compile(r"(?:chose|choose to|chosen|choosing)\s+", re.IGNORECASE),
    re.compile(r"(?:using\s+\S+\s+instead\s+of\s+\S+)", re.IGNORECASE),
    re.compile(r"(?:switched\s+(?:to|from)\s+\S+)", re.IGNORECASE),
    re.compile(r"(?:opted\s+for\s+)", re.IGNORECASE),
    re.compile(r"(?:went\s+with\s+)", re.IGNORECASE),
    re.compile(r"(?:settled\s+on\s+)", re.IGNORECASE),
    re.compile(r"(?:selected\s+\S+\s+(?:because|over|for))", re.IGNORECASE),
    re.compile(r"(?:the\s+(?:choice|decision)\s+(?:was|is|has been)\s+)", re.IGNORECASE),
    re.compile(r"(?:architecture\s+(?:decision|choice)\s*:\s*)", re.IGNORECASE),
]

# Preference/style indicators.
_PREFERENCE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(?:always\s+(?:use|prefer|avoid|wrap|include|add))", re.IGNORECASE),
    re.compile(r"(?:i\s+(?:prefer|like|recommend|suggest))", re.IGNORECASE),
    re.compile(r"(?:make\s+sure\s+to)", re.IGNORECASE),
    re.compile(r"(?:it's\s+(?:better|best|preferred|recommended)\s+to)", re.IGNORECASE),
    re.compile(r"(?:best\s+practice(?:s)?\s+(?:is|are|dictate|recommend|suggest))", re.IGNORECASE),
    re.compile(r"(?:convention\s+(?:is|dictates|says|requires))", re.IGNORECASE),
    re.compile(r"(?:i\s+(?:usually|typically|generally|normally)\s+(?:use|do|prefer|write))", re.IGNORECASE),
    re.compile(r"(?:style\s+(?:guide|preference|choice)\s*(?:\:|is))", re.IGNORECASE),
    re.compile(r"(?:don't\s+(?:use|forget|omit|skip)|do\s+not\s+(?:use|forget|omit|skip))", re.IGNORECASE),
    re.compile(r"(?:avoid\s+(?:using|doing|calling|mixing))", re.IGNORECASE),
]

# Troubleshooting step indicators.
_TROUBLESHOOT_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(?:(?:first|second|third|fourth|fifth|next|last|finally)\s*,?\s+(?:try|check|verify|inspect|test|run|examine|look at|ensure|confirm))", re.IGNORECASE),
    re.compile(r"(?:step\s+\d+[\.:)])", re.IGNORECASE),
    re.compile(r"(?:debugging\s+steps?\s*:\s*)", re.IGNORECASE),
    re.compile(r"(?:troubleshoot(?:ing)?\s*(?:\:\s*|\s+steps?))", re.IGNORECASE),
    re.compile(r"(?:here(?:'s| is) (?:the|a|how to) (?:debugging|troubleshooting|diagnostic))", re.IGNORECASE),
    re.compile(r"(?:(?:to)\s+debug\s+this\s*,?\s*(?:try|check|first))", re.IGNORECASE),
    re.compile(r"(?:let(?:'s| us)\s+(?:start|begin)\s+(?:by|with)\s+)", re.IGNORECASE),
    re.compile(r"(?:common\s+(?:causes?|reasons?|issues?)\s+(?:include|are|:\s*))", re.IGNORECASE),
]

# -- Code patterns ------------------------------------------------------------

# Known design pattern names (matched case-insensitively).
_DESIGN_PATTERN_NAMES: List[str] = [
    "singleton", "factory", "abstract factory", "builder", "prototype",
    "adapter", "bridge", "composite", "decorator", "facade", "flyweight",
    "proxy", "chain of responsibility", "command", "interpreter", "iterator",
    "mediator", "memento", "observer", "state", "strategy", "template method",
    "visitor", "dependency injection", "repository", "unit of work",
    "active record", "data mapper", "mvc", "mvvm", "mvp", "dao",
    "service locator", "middleware", "pipeline", "event emitter",
    "publisher-subscriber", "pub-sub",
]

# API / library usage heuristics.
_API_USAGE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(?:import\s+(\w+)(?:\s+as\s+(\w+))?)"),
    re.compile(r"(?:from\s+([\w.]+)\s+import\s+)"),
    re.compile(r"(?:require\s*\(\s*['\"]([\w./-]+)['\"]\s*\))"),
    re.compile(r"(?:\b(\w+)\.\w+\s*\()"),
]

# Configuration / environment variable patterns.
_CONFIG_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(?:os\.environ(?:\.get)?\s*\(\s*['\"](\w+)['\"])"),
    re.compile(r"(?:os\.getenv\s*\(\s*['\"](\w+)['\"])"),
    re.compile(r"(?:(?:process\.env\.(\w+)))"),
    re.compile(r"(?:ENV(?:IRONMENT)?(?:_|\.)(\w+))"),
    re.compile(r"(?:\$\{?(\w+)\}?)"),
    re.compile(r"(?:(?:config|settings|options)\s*\[\s*['\"](\w+)['\"]\s*\])"),
    re.compile(r"(?:(?:config|settings|options)\.\w+)"),
    re.compile(r"(?:(?:BASE_URL|API_KEY|SECRET|TOKEN|HOST|PORT|DEBUG|LOG_LEVEL|DATABASE_URL|REDIS_URL)\b)"),
]

# Error handling patterns in code.
_ERROR_HANDLING_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(?:(?:try|except|catch|finally|raise|throw)\b)"),
    re.compile(r"(?:(?:Exception|Error|RuntimeError|ValueError|TypeError)\b)"),
    re.compile(r"(?:(?:logger?\.\w+|console\.\w+)\s*\()"),
    re.compile(r"(?:(?:assert|verify|validate|check)\s*\()"),
    re.compile(r"(?:(?:if\s+__name__\s*==\s*['\"]__main__['\"]))"),
    re.compile(r"(?:(?:\?\.|\?\?|\?\:))"),  # Optional chaining, nullish coalescing
]

# -- Text extraction patterns -------------------------------------------------

# Sentence-level pattern for step-by-step procedures.
_PROCEDURE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"^(?:\d+[\.\)]\s+)(.+)$", re.MULTILINE),
    re.compile(r"^(?:[-•▪▸]\s+)(.+)$", re.MULTILINE),
    re.compile(r"(?:first|then|next|after (?:that|this)|finally),?\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:step\s+\d+\s*:\s*)(.+)", re.IGNORECASE),
]

# Pattern for extracting key definitions (X is Y, X means Y).
_DEFINITION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(\w[\w\s]{2,40}?)\s+(?:is|are|was|were|means?|refers? to|stands for)\s+(.+?)(?:[.;!?]|\Z)", re.IGNORECASE),
    re.compile(r"(?:defined\s+as|known\s+as|called|termed)\s+(.+?)(?:[.;!?]|\Z)", re.IGNORECASE),
]

# Best practice indicator phrases.
_BEST_PRACTICE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(?:it(?:'s| is)\s+(?:a\s+)?best\s+practice\s+to)", re.IGNORECASE),
    re.compile(r"(?:best\s+practice(?:s)?\s+(?:recommend|suggest|dictate|include))", re.IGNORECASE),
    re.compile(r"(?:recommended\s+(?:approach|way|method|practice))", re.IGNORECASE),
    re.compile(r"(?:you\s+should\s+(?:always|never|generally|typically))", re.IGNORECASE),
    re.compile(r"(?:a\s+good\s+(?:rule|practice|habit|pattern)\s+is)", re.IGNORECASE),
]

# -- Markdown patterns --------------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_MD_CODE_BLOCK = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC = re.compile(r"\*(.+?)\*")
_MD_LIST_ITEM = re.compile(r"^\s*[-*+]\s+(.+)$", re.MULTILINE)
_MD_NUMBERED_LIST = re.compile(r"^\s*\d+[\.\)]\s+(.+)$", re.MULTILINE)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, max_length: int = 500) -> str:
    """Truncate *text* to *max_length* characters, appending ``...`` if cut."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _extract_surrounding_context(
    text: str,
    match_start: int,
    match_end: int,
    context_chars: int = 150,
) -> str:
    """Return up to *context_chars* of text around the matched region."""
    start = max(0, match_start - context_chars)
    end = min(len(text), match_end + context_chars)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def _compute_confidence_from_matches(matches: List[PatternMatch]) -> float:
    """Average the confidence of all matches, bounded to [0, 1]."""
    if not matches:
        return 0.0
    avg = sum(m.confidence for m in matches) / len(matches)
    return round(min(1.0, max(0.0, avg)), 3)


def _split_into_sentences(text: str) -> List[str]:
    """Split *text* into sentences, ignoring common abbreviations."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    sentences: List[str] = []
    for s in raw:
        s = s.strip()
        if len(s) < 5:
            continue
        sentences.append(s)
    return sentences


# ---------------------------------------------------------------------------
# KnowledgeExtractor
# ---------------------------------------------------------------------------


class KnowledgeExtractor:
    """Automatically extracts knowledge entries from conversations, code,
    and arbitrary text.

    The extractor uses deterministic regex-based pattern detection combined
    with heuristic scoring to identify actionable knowledge: solutions to
    problems, user preferences, error-solution pairs, project decisions,
    troubleshooting steps, design patterns in code, API usage conventions,
    configuration patterns, and best practices.

    Parameters
    ----------
    store:
        A :class:`KnowledgeStore` instance used for persisting extracted
        knowledge.  The store is not accessed during extraction itself —
        only when :meth:`auto_extract_and_store` is called.

    Attributes
    ----------
    store:
        The provided :class:`KnowledgeStore` reference.
    extraction_stats:
        A running tally of extraction calls and match counts since the
        instance was created.
    """

    def __init__(self, store: Optional["KnowledgeStore"] = None) -> None:
        self.store = store
        self.extraction_stats: Dict[str, int] = {
            "conversation_extractions": 0,
            "code_extractions": 0,
            "text_extractions": 0,
            "markdown_extractions": 0,
            "total_patterns_found": 0,
            "total_entries_stored": 0,
        }

    # ------------------------------------------------------------------ #
    #  Conversation extraction                                             #
    # ------------------------------------------------------------------ #

    def extract_from_conversation(
        self,
        messages: List[Dict[str, str]],
    ) -> List[ExtractionResult]:
        """Analyse a conversation and extract knowledge entries.

        Walks through *messages* (a list of dicts with ``"role"`` and
        ``"content"`` keys) looking for:

        * **Solutions** — when the assistant provides working code or a fix.
        * **User preferences** — repeated requests, style preferences.
        * **Error-solution pairs** — user reports an error, assistant fixes it.
        * **Project decisions** — architecture choices, tech stack decisions.
        * **Troubleshooting steps** — debugging sequences.

        Parameters
        ----------
        messages:
            A list of message dicts, each with at least ``"role"`` and
            ``"content"`` keys.  Roles are typically ``"user"``,
            ``"assistant"``, or ``"system"``.

        Returns
        -------
        list[ExtractionResult]
            Zero or more extraction results, one per distinct knowledge
            entry found in the conversation.
        """
        if not messages:
            return []

        self.extraction_stats["conversation_extractions"] += 1
        results: List[ExtractionResult] = []

        for idx in range(len(messages)):
            msg = messages[idx]
            role = msg.get("role", "").lower()
            content = msg.get("content", "")

            if not content or not content.strip():
                continue

            if role == "assistant":
                # -- Solutions ------------------------------------------------
                solution_matches = self._detect_solution_pattern(content)
                if solution_matches:
                    # Enrich context with the preceding user message if available
                    user_context = ""
                    if idx > 0 and messages[idx - 1].get("role", "").lower() == "user":
                        user_context = _truncate(messages[idx - 1]["content"], 200)

                    result = ExtractionResult(
                        entries=solution_matches,
                        confidence=_compute_confidence_from_matches(solution_matches),
                        source_type="conversation",
                        metadata={
                            "message_index": idx,
                            "role": role,
                            "user_context": user_context,
                            "extracted_at": _now_iso(),
                        },
                    )
                    results.append(result)

                # -- Error-solution pairs (assistant provides fix after error) --
                error_solution = self._detect_error_solution_pair(messages, idx)
                if error_solution:
                    result = ExtractionResult(
                        entries=[error_solution],
                        confidence=error_solution.confidence,
                        source_type="conversation",
                        metadata={
                            "message_index": idx,
                            "role": role,
                            "pair_type": "error_solution",
                            "extracted_at": _now_iso(),
                        },
                    )
                    results.append(result)

                # -- Troubleshooting steps ------------------------------------
                trouble_matches = self._detect_troubleshooting(content)
                if trouble_matches:
                    result = ExtractionResult(
                        entries=trouble_matches,
                        confidence=_compute_confidence_from_matches(trouble_matches),
                        source_type="conversation",
                        metadata={
                            "message_index": idx,
                            "role": role,
                            "extracted_at": _now_iso(),
                        },
                    )
                    results.append(result)

            elif role == "user":
                # -- Decisions ------------------------------------------------
                decision_matches = self._detect_decisions(content)
                if decision_matches:
                    result = ExtractionResult(
                        entries=decision_matches,
                        confidence=_compute_confidence_from_matches(decision_matches),
                        source_type="conversation",
                        metadata={
                            "message_index": idx,
                            "role": role,
                            "extracted_at": _now_iso(),
                        },
                    )
                    results.append(result)

                # -- Preferences ----------------------------------------------
                preference_matches = self._detect_preferences(content)
                if preference_matches:
                    result = ExtractionResult(
                        entries=preference_matches,
                        confidence=_compute_confidence_from_matches(preference_matches),
                        source_type="conversation",
                        metadata={
                            "message_index": idx,
                            "role": role,
                            "extracted_at": _now_iso(),
                        },
                    )
                    results.append(result)

                # -- Error messages (reported by user) -------------------------
                error_matches = self._detect_errors(content)
                if error_matches:
                    # Only store if there's no paired solution already captured
                    already_paired = any(
                        r.metadata.get("pair_type") == "error_solution"
                        and r.metadata.get("message_index", -1) == idx + 1
                        for r in results
                    )
                    if not already_paired:
                        result = ExtractionResult(
                            entries=error_matches,
                            confidence=_compute_confidence_from_matches(error_matches),
                            source_type="conversation",
                            metadata={
                                "message_index": idx,
                                "role": role,
                                "pair_type": "error_only",
                                "extracted_at": _now_iso(),
                            },
                        )
                        results.append(result)

        self.extraction_stats["total_patterns_found"] += sum(
            len(r.entries) for r in results
        )
        _logger.debug(
            "Extracted %d knowledge entries from %d conversation messages.",
            len(results),
            len(messages),
        )
        return results

    # ------------------------------------------------------------------ #
    #  Code extraction                                                      #
    # ------------------------------------------------------------------ #

    def extract_from_code(
        self,
        file_path: str,
        content: str,
    ) -> List[ExtractionResult]:
        """Analyse source code and extract knowledge entries.

        Detects the following patterns in code:

        * **Design patterns** — singleton, factory, observer, strategy, etc.
        * **API usage patterns** — how libraries/frameworks are imported and
          used.
        * **Configuration patterns** — environment variables, config files,
          settings objects.
        * **Error handling patterns** — try/except blocks, assertion usage,
          logging calls.
        * **Common utilities/helpers** — repeated utility functions,
          helper classes.

        Parameters
        ----------
        file_path:
            The path to the source file being analysed (used for metadata).
        content:
            The full text content of the source file.

        Returns
        -------
        list[ExtractionResult]
            Zero or more extraction results, one per distinct pattern found.
        """
        if not content or not content.strip():
            return []

        self.extraction_stats["code_extractions"] += 1
        results: List[ExtractionResult] = []

        # -- Design patterns -----------------------------------------------
        design_matches = self._detect_design_patterns(file_path, content)
        if design_matches:
            results.append(ExtractionResult(
                entries=design_matches,
                confidence=_compute_confidence_from_matches(design_matches),
                source_type="code",
                metadata={
                    "file_path": file_path,
                    "pattern_category": "design_pattern",
                    "extracted_at": _now_iso(),
                },
            ))

        # -- API usage patterns --------------------------------------------
        api_matches = self._detect_api_usage(content)
        if api_matches:
            results.append(ExtractionResult(
                entries=api_matches,
                confidence=_compute_confidence_from_matches(api_matches),
                source_type="code",
                metadata={
                    "file_path": file_path,
                    "pattern_category": "api_usage",
                    "extracted_at": _now_iso(),
                },
            ))

        # -- Configuration patterns ----------------------------------------
        config_matches = self._detect_config_patterns(content)
        if config_matches:
            results.append(ExtractionResult(
                entries=config_matches,
                confidence=_compute_confidence_from_matches(config_matches),
                source_type="code",
                metadata={
                    "file_path": file_path,
                    "pattern_category": "configuration",
                    "extracted_at": _now_iso(),
                },
            ))

        # -- Error handling patterns ---------------------------------------
        error_handling_matches = self._detect_error_handling(content)
        if error_handling_matches:
            results.append(ExtractionResult(
                entries=error_handling_matches,
                confidence=_compute_confidence_from_matches(error_handling_matches),
                source_type="code",
                metadata={
                    "file_path": file_path,
                    "pattern_category": "error_handling",
                    "extracted_at": _now_iso(),
                },
            ))

        # -- Utility/helper patterns ---------------------------------------
        util_matches = self._detect_utility_patterns(file_path, content)
        if util_matches:
            results.append(ExtractionResult(
                entries=util_matches,
                confidence=_compute_confidence_from_matches(util_matches),
                source_type="code",
                metadata={
                    "file_path": file_path,
                    "pattern_category": "utility",
                    "extracted_at": _now_iso(),
                },
            ))

        self.extraction_stats["total_patterns_found"] += sum(
            len(r.entries) for r in results
        )
        _logger.debug(
            "Extracted %d knowledge entries from code file: %s",
            len(results),
            file_path,
        )
        return results

    # ------------------------------------------------------------------ #
    #  Text extraction                                                      #
    # ------------------------------------------------------------------ #

    def extract_from_text(
        self,
        text: str,
        source: str = "manual",
    ) -> List[ExtractionResult]:
        """Extract knowledge entries from arbitrary prose text.

        Looks for:

        * **Key concepts and definitions** — sentences of the form "X is Y".
        * **Step-by-step procedures** — numbered or bulleted lists,
          sequential instructions.
        * **Facts and assertions** — declarative statements.
        * **Best practices mentioned** — prescriptive advice.

        Parameters
        ----------
        text:
            The prose text to analyse.
        source:
            A label identifying where the text came from (e.g.
            ``"manual"``, ``"clipboard"``, ``"documentation"``).  Stored
            in the result metadata.

        Returns
        -------
        list[ExtractionResult]
            Zero or more extraction results.
        """
        if not text or not text.strip():
            return []

        self.extraction_stats["text_extractions"] += 1
        results: List[ExtractionResult] = []
        sentences = _split_into_sentences(text)

        # -- Key concepts / definitions ------------------------------------
        definition_matches: List[PatternMatch] = []
        for sentence in sentences:
            for pat in _DEFINITION_PATTERNS:
                m = pat.search(sentence)
                if m:
                    definition_matches.append(PatternMatch(
                        pattern_type="definition",
                        content=m.group(0).strip(),
                        context=sentence,
                        confidence=0.65,
                    ))
        if definition_matches:
            results.append(ExtractionResult(
                entries=definition_matches,
                confidence=_compute_confidence_from_matches(definition_matches),
                source_type="text",
                metadata={"source": source, "extracted_at": _now_iso()},
            ))

        # -- Step-by-step procedures ---------------------------------------
        procedure_matches: List[PatternMatch] = []
        for pat in _PROCEDURE_PATTERNS:
            for m in pat.finditer(text):
                step_text = m.group(0).strip()
                if len(step_text) < 10:
                    continue
                procedure_matches.append(PatternMatch(
                    pattern_type="procedure",
                    content=step_text,
                    context=_extract_surrounding_context(
                        text, m.start(), m.end(), context_chars=100,
                    ),
                    confidence=0.75,
                ))
        if procedure_matches:
            results.append(ExtractionResult(
                entries=procedure_matches,
                confidence=_compute_confidence_from_matches(procedure_matches),
                source_type="text",
                metadata={"source": source, "extracted_at": _now_iso()},
            ))

        # -- Facts and assertions ------------------------------------------
        fact_matches: List[PatternMatch] = []
        assertion_patterns = [
            re.compile(r"^(?:it is|this is|that is|these are|those are)\s+", re.IGNORECASE),
            re.compile(r"^(?:the (?:fact|truth|reality|case) (?:is|was))\s+", re.IGNORECASE),
            re.compile(r"^(?:\w+\s+(?:has|have|had)\s+)", re.IGNORECASE),
            re.compile(r"^(?:\w+\s+(?:can|cannot|can't|could|would|will|should)\s+)", re.IGNORECASE),
            re.compile(r"^(?:there (?:is|are|was|were|will be))\s+", re.IGNORECASE),
        ]
        for sentence in sentences:
            for pat in assertion_patterns:
                m = pat.match(sentence)
                if m:
                    fact_matches.append(PatternMatch(
                        pattern_type="fact",
                        content=sentence,
                        context=sentence,
                        confidence=0.50,
                    ))
                    break  # One match per sentence is enough
        if fact_matches:
            results.append(ExtractionResult(
                entries=fact_matches,
                confidence=_compute_confidence_from_matches(fact_matches),
                source_type="text",
                metadata={"source": source, "extracted_at": _now_iso()},
            ))

        # -- Best practices ------------------------------------------------
        bp_matches: List[PatternMatch] = []
        for sentence in sentences:
            for pat in _BEST_PRACTICE_PATTERNS:
                m = pat.search(sentence)
                if m:
                    bp_matches.append(PatternMatch(
                        pattern_type="best_practice",
                        content=sentence,
                        context=sentence,
                        confidence=0.80,
                    ))
                    break
        if bp_matches:
            results.append(ExtractionResult(
                entries=bp_matches,
                confidence=_compute_confidence_from_matches(bp_matches),
                source_type="text",
                metadata={"source": source, "extracted_at": _now_iso()},
            ))

        self.extraction_stats["total_patterns_found"] += sum(
            len(r.entries) for r in results
        )
        _logger.debug(
            "Extracted %d knowledge entries from text (source=%s).",
            len(results),
            source,
        )
        return results

    # ------------------------------------------------------------------ #
    #  Markdown extraction                                                  #
    # ------------------------------------------------------------------ #

    def extract_from_markdown(self, filepath: str) -> List[ExtractionResult]:
        """Parse a Markdown file and extract structured knowledge entries.

        Extraction targets:

        * **Headings** — used as titles/labels for knowledge entries.
        * **Code blocks** — extracted as code snippets with the surrounding
          heading as context.
        * **Lists** — bulleted and numbered lists treated as procedures.
        * **Bold/italic terms** — key concepts highlighted by the author.

        Parameters
        ----------
        filepath:
            Path to the Markdown file to read and parse.

        Returns
        -------
        list[ExtractionResult]
            Zero or more extraction results.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {filepath}")

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _logger.error("Failed to read markdown file %s: %s", filepath, exc)
            return []

        if not raw.strip():
            return []

        self.extraction_stats["markdown_extractions"] += 1
        results: List[ExtractionResult] = []

        # -- Headings as titles --------------------------------------------
        heading_matches: List[PatternMatch] = []
        for m in _MD_HEADING.finditer(raw):
            level = len(m.group(1))
            title = m.group(2).strip()
            if len(title) < 3:
                continue
            heading_matches.append(PatternMatch(
                pattern_type="heading",
                content=title,
                context=f"Heading level {level}",
                confidence=0.90 if level <= 2 else 0.70,
            ))
        if heading_matches:
            results.append(ExtractionResult(
                entries=heading_matches,
                confidence=_compute_confidence_from_matches(heading_matches),
                source_type="markdown",
                metadata={"file_path": filepath, "extracted_at": _now_iso()},
            ))

        # -- Code blocks as snippets ---------------------------------------
        code_matches: List[PatternMatch] = []
        # Find the nearest preceding heading for context
        headings_list = list(_MD_HEADING.finditer(raw))
        for m in _MD_CODE_BLOCK.finditer(raw):
            lang = m.group(1) or "unknown"
            code = m.group(2).strip()
            if len(code) < 5:
                continue
            # Find the closest heading above this code block
            context_heading = ""
            for h in reversed(headings_list):
                if h.start() < m.start():
                    context_heading = h.group(2).strip()
                    break
            code_matches.append(PatternMatch(
                pattern_type="code_snippet",
                content=_truncate(code, 1000),
                context=context_heading or "No heading context",
                confidence=0.85,
                # Extra metadata about language is attached via the content
            ))
            # Also store the language as a tag hint in the metadata later
        if code_matches:
            results.append(ExtractionResult(
                entries=code_matches,
                confidence=_compute_confidence_from_matches(code_matches),
                source_type="markdown",
                metadata={
                    "file_path": filepath,
                    "pattern_category": "code_snippet",
                    "extracted_at": _now_iso(),
                },
            ))

        # -- Lists as procedures -------------------------------------------
        list_matches: List[PatternMatch] = []
        for m in _MD_LIST_ITEM.finditer(raw):
            item = m.group(1).strip()
            if len(item) < 5:
                continue
            list_matches.append(PatternMatch(
                pattern_type="list_item",
                content=item,
                context=_extract_surrounding_context(
                    raw, m.start(), m.end(), context_chars=100,
                ),
                confidence=0.65,
            ))
        for m in _MD_NUMBERED_LIST.finditer(raw):
            item = m.group(1).strip()
            if len(item) < 5:
                continue
            list_matches.append(PatternMatch(
                pattern_type="numbered_step",
                content=item,
                context=_extract_surrounding_context(
                    raw, m.start(), m.end(), context_chars=100,
                ),
                confidence=0.70,
            ))
        if list_matches:
            results.append(ExtractionResult(
                entries=list_matches,
                confidence=_compute_confidence_from_matches(list_matches),
                source_type="markdown",
                metadata={
                    "file_path": filepath,
                    "pattern_category": "procedure",
                    "extracted_at": _now_iso(),
                },
            ))

        # -- Bold/italic as key concepts -----------------------------------
        concept_matches: List[PatternMatch] = []
        for m in _MD_BOLD.finditer(raw):
            term = m.group(1).strip()
            if len(term) < 2 or len(term) > 100:
                continue
            concept_matches.append(PatternMatch(
                pattern_type="key_concept",
                content=term,
                context=_extract_surrounding_context(
                    raw, m.start(), m.end(), context_chars=80,
                ),
                confidence=0.75,
            ))
        for m in _MD_ITALIC.finditer(raw):
            term = m.group(1).strip()
            if len(term) < 2 or len(term) > 100:
                continue
            # Avoid duplicates from bold-italic (**_text_**)
            if any(c.content == term for c in concept_matches):
                continue
            concept_matches.append(PatternMatch(
                pattern_type="key_concept",
                content=term,
                context=_extract_surrounding_context(
                    raw, m.start(), m.end(), context_chars=80,
                ),
                confidence=0.55,
            ))
        if concept_matches:
            results.append(ExtractionResult(
                entries=concept_matches,
                confidence=_compute_confidence_from_matches(concept_matches),
                source_type="markdown",
                metadata={
                    "file_path": filepath,
                    "pattern_category": "key_concept",
                    "extracted_at": _now_iso(),
                },
            ))

        self.extraction_stats["total_patterns_found"] += sum(
            len(r.entries) for r in results
        )
        _logger.debug(
            "Extracted %d knowledge entries from markdown: %s",
            len(results),
            filepath,
        )
        return results

    # ------------------------------------------------------------------ #
    #  Auto-extract-and-store convenience method                            #
    # ------------------------------------------------------------------ #

    async def auto_extract_and_store(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        code_files: Optional[List[Dict[str, str]]] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract knowledge from all provided sources and store results.

        This is a convenience method that calls the appropriate extraction
        methods based on the provided arguments, then persists every
        :class:`ExtractionResult` via :attr:`store`.

        Parameters
        ----------
        messages:
            Conversation messages to pass to
            :meth:`extract_from_conversation`.
        code_files:
            A list of dicts, each with ``"file_path"`` and ``"content"``
            keys, passed to :meth:`extract_from_code`.
        text:
            Arbitrary text to pass to :meth:`extract_from_text`.

        Returns
        -------
        dict
            A statistics dict with keys:

            * ``"conversation_results"`` — count of results from conversations.
            * ``"code_results"`` — count of results from code analysis.
            * ``"text_results"`` — count of results from text extraction.
            * ``"total_extracted"`` — total results before deduplication.
            * ``"total_stored"`` — entries successfully stored.
            * ``"errors"`` — list of error messages encountered.

        Raises
        ------
        RuntimeError
            If no :class:`KnowledgeStore` was provided at init time.
        """
        if self.store is None:
            raise RuntimeError(
                "KnowledgeExtractor requires a KnowledgeStore instance to "
                "auto-extract and store.  Pass a store to the constructor."
            )

        errors: List[str] = []
        all_results: List[ExtractionResult] = []
        conv_count = 0
        code_count = 0
        text_count = 0

        # -- Conversation --------------------------------------------------
        if messages:
            try:
                conv_results = self.extract_from_conversation(messages)
                all_results.extend(conv_results)
                conv_count = len(conv_results)
            except Exception as exc:
                msg = f"Conversation extraction failed: {exc}"
                _logger.error(msg, exc_info=True)
                errors.append(msg)

        # -- Code files ----------------------------------------------------
        if code_files:
            for file_info in code_files:
                fpath = file_info.get("file_path", "")
                fcontent = file_info.get("content", "")
                if not fpath or not fcontent:
                    continue
                try:
                    code_results = self.extract_from_code(fpath, fcontent)
                    all_results.extend(code_results)
                    code_count += len(code_results)
                except Exception as exc:
                    msg = f"Code extraction failed for {fpath}: {exc}"
                    _logger.error(msg, exc_info=True)
                    errors.append(msg)

        # -- Text ----------------------------------------------------------
        if text:
            try:
                text_results = self.extract_from_text(text)
                all_results.extend(text_results)
                text_count = len(text_results)
            except Exception as exc:
                msg = f"Text extraction failed: {exc}"
                _logger.error(msg, exc_info=True)
                errors.append(msg)

        # -- Store ---------------------------------------------------------
        stored = 0
        for result in all_results:
            try:
                await self._store_result(result)
                stored += 1
            except Exception as exc:
                msg = (
                    f"Failed to store extraction result "
                    f"(type={result.source_type}): {exc}"
                )
                _logger.error(msg, exc_info=True)
                errors.append(msg)

        self.extraction_stats["total_entries_stored"] += stored

        stats = {
            "conversation_results": conv_count,
            "code_results": code_count,
            "text_results": text_count,
            "total_extracted": len(all_results),
            "total_stored": stored,
            "errors": errors,
        }
        _logger.info("Auto-extract-and-store complete: %s", stats)
        return stats

    # ------------------------------------------------------------------ #
    #  Internal pattern detectors                                          #
    # ------------------------------------------------------------------ #

    # -- Solution pattern detection -----------------------------------------

    def _detect_solution_pattern(self, text: str) -> List[PatternMatch]:
        """Detect when the assistant provides a solution or fix.

        A solution is identified when one or more solution-indicator phrases
        are found in the text.  If the text also contains a code block
        (fenced with triple backticks), confidence is boosted.
        """
        matches: List[PatternMatch] = []
        for pat in _SOLUTION_INDICATORS:
            m = pat.search(text)
            if m:
                confidence = 0.70
                # Boost if code block is present (stronger signal of a fix)
                if "```" in text:
                    confidence = 0.90
                # Extract a reasonable content window around the match
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 300)
                content = text[start:end].strip()
                matches.append(PatternMatch(
                    pattern_type="solution",
                    content=_truncate(content, 500),
                    context=_extract_surrounding_context(
                        text, m.start(), m.end(),
                    ),
                    confidence=confidence,
                ))
        return matches

    # -- Error detection ----------------------------------------------------

    def _detect_errors(self, text: str) -> List[PatternMatch]:
        """Detect error messages, tracebacks, and failure indicators."""
        matches: List[PatternMatch] = []
        for pat in _ERROR_PATTERNS:
            m = pat.search(text)
            if m:
                matches.append(PatternMatch(
                    pattern_type="error",
                    content=m.group(0).strip(),
                    context=_extract_surrounding_context(
                        text, m.start(), m.end(),
                    ),
                    confidence=0.85,
                ))
        return matches

    # -- Error-solution pair detection --------------------------------------

    def _detect_error_solution_pair(
        self,
        messages: List[Dict[str, str]],
        assistant_idx: int,
    ) -> Optional[PatternMatch]:
        """Look for an error in the preceding user message and a solution
        in the current assistant message, producing an error-solution pair.
        """
        if assistant_idx < 1:
            return None

        prev_msg = messages[assistant_idx - 1]
        if prev_msg.get("role", "").lower() != "user":
            return None

        user_text = prev_msg.get("content", "")
        assistant_text = messages[assistant_idx].get("content", "")

        # Check if user reported an error
        has_error = any(p.search(user_text) for p in _ERROR_PATTERNS)
        if not has_error:
            # Also accept implicit errors via keywords
            error_keywords = [
                "not working", "doesn't work", "broken", "failing", "crash",
                "error", "exception", "bug", "issue", "problem", "wrong",
                "failure", "fail", "traceback", "stack trace",
            ]
            has_error = any(kw in user_text.lower() for kw in error_keywords)
        if not has_error:
            return None

        # Check if assistant provides a solution
        has_solution = any(p.search(assistant_text) for p in _SOLUTION_INDICATORS)
        if not has_solution:
            # Accept code blocks as implicit solutions
            has_solution = "```" in assistant_text
        if not has_solution:
            return None

        # Extract the solution content
        solution_content = _truncate(assistant_text, 500)

        return PatternMatch(
            pattern_type="error_solution",
            content=solution_content,
            context=f"Error: {_truncate(user_text, 200)}",
            confidence=0.88,
        )

    # -- Decision detection -------------------------------------------------

    def _detect_decisions(self, text: str) -> List[PatternMatch]:
        """Detect project decisions and architecture choices."""
        matches: List[PatternMatch] = []
        for pat in _DECISION_PATTERNS:
            m = pat.search(text)
            if m:
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 200)
                content = text[start:end].strip()
                matches.append(PatternMatch(
                    pattern_type="decision",
                    content=_truncate(content, 400),
                    context=_extract_surrounding_context(
                        text, m.start(), m.end(),
                    ),
                    confidence=0.80,
                ))
        return matches

    # -- Preference detection -----------------------------------------------

    def _detect_preferences(self, text: str) -> List[PatternMatch]:
        """Detect user preferences and style choices."""
        matches: List[PatternMatch] = []
        for pat in _PREFERENCE_PATTERNS:
            m = pat.search(text)
            if m:
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 200)
                content = text[start:end].strip()
                matches.append(PatternMatch(
                    pattern_type="preference",
                    content=_truncate(content, 400),
                    context=_extract_surrounding_context(
                        text, m.start(), m.end(),
                    ),
                    confidence=0.72,
                ))
        return matches

    # -- Troubleshooting detection ------------------------------------------

    def _detect_troubleshooting(self, text: str) -> List[PatternMatch]:
        """Detect step-by-step troubleshooting and debugging sequences."""
        matches: List[PatternMatch] = []
        for pat in _TROUBLESHOOT_PATTERNS:
            for m in pat.finditer(text):
                step_text = m.group(0).strip()
                if len(step_text) < 15:
                    continue
                matches.append(PatternMatch(
                    pattern_type="troubleshooting",
                    content=step_text,
                    context=_extract_surrounding_context(
                        text, m.start(), m.end(),
                    ),
                    confidence=0.78,
                ))
        # Also detect numbered steps in code-free assistant messages
        numbered_steps = re.findall(
            r"(?:^\s*)(\d+)\.\s+(.{20,})", text, re.MULTILINE,
        )
        if len(numbered_steps) >= 2:
            combined = "\n".join(f"{n}. {s}" for n, s in numbered_steps)
            matches.append(PatternMatch(
                pattern_type="troubleshooting",
                content=_truncate(combined, 600),
                context="Numbered steps detected in response",
                confidence=0.70,
            ))
        return matches

    # -- Design pattern detection in code -----------------------------------

    def _detect_design_patterns(
        self, file_path: str, content: str,
    ) -> List[PatternMatch]:
        """Detect common design patterns by name mention and structural
        heuristics in the source code.
        """
        matches: List[PatternMatch] = []
        content_lower = content.lower()

        # -- By name --------------------------------------------------------
        for pattern_name in _DESIGN_PATTERN_NAMES:
            if pattern_name in content_lower:
                # Find the surrounding context
                idx = content_lower.index(pattern_name)
                context = _extract_surrounding_context(
                    content, idx, idx + len(pattern_name), context_chars=200,
                )
                matches.append(PatternMatch(
                    pattern_type="code_pattern",
                    content=f"Design pattern detected: {pattern_name.title()}",
                    context=context,
                    confidence=0.75,
                ))

        # -- Singleton: class with private constructor / instance check ------
        singleton_heuristic = re.search(
            r"class\s+(\w+)\s*[^:]*:(?:.*?)(?:_instance|__instance|_shared|__shared)\s*=\s*None",
            content, re.DOTALL,
        )
        if singleton_heuristic:
            matches.append(PatternMatch(
                pattern_type="code_pattern",
                content=f"Singleton pattern detected in class '{singleton_heuristic.group(1)}'",
                context=_truncate(singleton_heuristic.group(0), 300),
                confidence=0.85,
            ))

        # -- Factory: method that returns instances based on input ----------
        factory_heuristic = re.search(
            r"def\s+(?:create|make|build|factory|get_instance)\s*\([^)]*\)\s*(?:->\s*\w+)?:\s*\n\s*(?:.*?)(?:return\s+\w+\s*\()",
            content, re.DOTALL,
        )
        if factory_heuristic:
            matches.append(PatternMatch(
                pattern_type="code_pattern",
                content=f"Factory pattern detected: '{factory_heuristic.group(0)[:80]}'",
                context=_truncate(factory_heuristic.group(0), 400),
                confidence=0.80,
            ))

        # -- Observer: methods named on/notify/subscribe/emit ---------------
        observer_methods = re.findall(
            r"def\s+(?:on_\w+|notify|subscribe|unsubscribe|emit|dispatch|add_listener|remove_listener)\s*\(",
            content,
        )
        if len(observer_methods) >= 2:
            methods_str = ", ".join(observer_methods[:5])
            matches.append(PatternMatch(
                pattern_type="code_pattern",
                content=f"Observer pattern detected: methods [{methods_str}]",
                context=f"Found {len(observer_methods)} observer-like methods",
                confidence=0.78,
            ))

        # -- Strategy: classes with execute/run/process/handle + interface ---
        strategy_heuristic = re.findall(
            r"class\s+(\w+Strategy|\w+Handler|\w+Processor|\w+Executor)\b",
            content,
        )
        if strategy_heuristic:
            classes_str = ", ".join(strategy_heuristic[:5])
            matches.append(PatternMatch(
                pattern_type="code_pattern",
                content=f"Strategy pattern detected: classes [{classes_str}]",
                context=f"Found {len(strategy_heuristic)} strategy-like classes",
                confidence=0.82,
            ))

        # -- Decorator: functions that wrap and return another function -----
        decorator_heuristic = re.search(
            r"def\s+(\w+)\s*\([^)]*\)\s*:\s*\n\s*(?:.*?)(?:def\s+wrapper|def\s+inner|def\s+decorated|return\s+func|return\s+\w+)",
            content, re.DOTALL,
        )
        if decorator_heuristic:
            func_name = decorator_heuristic.group(1)
            matches.append(PatternMatch(
                pattern_type="code_pattern",
                content=f"Decorator pattern detected: function '{func_name}'",
                context=_truncate(decorator_heuristic.group(0), 400),
                confidence=0.80,
            ))

        return matches

    # -- API usage detection ------------------------------------------------

    def _detect_api_usage(self, content: str) -> List[PatternMatch]:
        """Detect how libraries and frameworks are imported and used."""
        matches: List[PatternMatch] = []
        imports_found: List[Tuple[str, int, int]] = []

        for pat in _API_USAGE_PATTERNS:
            for m in pat.finditer(content):
                module_name = m.group(1) if m.group(1) else ""
                if not module_name:
                    continue
                imports_found.append((module_name, m.start(), m.end()))

        # Count usage frequency for each imported module
        usage_counter: Counter = Counter()
        for module_name, _, _ in imports_found:
            # Count occurrences of module_name in the rest of the file
            usage_count = len(re.findall(rf"\b{re.escape(module_name)}\b", content))
            usage_counter[module_name] += usage_count

        # Report the most-used APIs
        for module_name, count in usage_counter.most_common(10):
            if count < 2:
                continue  # Skip single-use imports (less notable)
            matches.append(PatternMatch(
                pattern_type="api_usage",
                content=f"API: {module_name} (used {count} times)",
                context=f"Module '{module_name}' is imported and actively used",
                confidence=min(0.95, 0.50 + count * 0.05),
            ))

        # Detect framework-specific patterns
        framework_patterns = [
            ("FastAPI", r"(?:@\w+\.?(?:get|post|put|delete|patch|route)\s*\()", 0.88),
            ("Flask", r"(?:@\w+\.route\s*\()", 0.88),
            ("Django", r"(?:models\.Model|django\.|@api_view)", 0.88),
            ("SQLAlchemy", r"(?:session\.(?:add|query|execute|commit)|Column\s*\()", 0.85),
            ("Pydantic", r"(?:BaseModel|Field\s*\(|validator\s*\()", 0.85),
            ("React", r"(?:useEffect|useState|useRef|useCallback|jsx|tsx)", 0.85),
            ("Express", r"(?:app\.(?:get|post|put|delete|use|listen)\s*\()", 0.85),
            ("asyncio", r"(?:async\s+def|await\s+)", 0.60),
        ]
        for framework_name, pattern, confidence in framework_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                matches.append(PatternMatch(
                    pattern_type="api_usage",
                    content=f"Framework detected: {framework_name}",
                    context=f"Found {framework_name}-specific patterns in code",
                    confidence=confidence,
                ))

        return matches

    # -- Configuration pattern detection ------------------------------------

    def _detect_config_patterns(self, content: str) -> List[PatternMatch]:
        """Detect configuration and environment variable usage."""
        matches: List[PatternMatch] = []
        env_vars_seen: set = set()

        for pat in _CONFIG_PATTERNS:
            for m in pat.finditer(content):
                var_name = m.group(1) if m.lastindex and m.group(1) else ""
                if not var_name:
                    continue
                if var_name in env_vars_seen:
                    continue
                env_vars_seen.add(var_name)
                matches.append(PatternMatch(
                    pattern_type="configuration",
                    content=f"Config var: {var_name}",
                    context=_extract_surrounding_context(
                        content, m.start(), m.end(),
                    ),
                    confidence=0.82,
                ))

        # Detect config file patterns
        config_file_patterns = [
            (r"(?:\.env|\.yaml|\.yml|\.toml|\.ini|\.cfg|\.conf|\.json)", "config_file"),
            (r"(?:load_dotenv|read_config|ConfigParser|yaml\.safe_load|toml\.load)", "config_loader"),
        ]
        for pattern, label in config_file_patterns:
            m = re.search(pattern, content)
            if m:
                matches.append(PatternMatch(
                    pattern_type="configuration",
                    content=f"Config pattern: {label} ({m.group(0)})",
                    context=_extract_surrounding_context(
                        content, m.start(), m.end(),
                    ),
                    confidence=0.70,
                ))

        return matches

    # -- Error handling pattern detection -----------------------------------

    def _detect_error_handling(self, content: str) -> List[PatternMatch]:
        """Detect error handling constructs in code."""
        matches: List[PatternMatch] = []

        # Count try/except or try/catch blocks
        try_blocks = re.findall(
            r"(?:try\s*:|try\s*\{)", content,
        )
        if try_blocks:
            matches.append(PatternMatch(
                pattern_type="error_handling",
                content=f"Found {len(try_blocks)} try block(s) in file",
                context="Error handling via try/except or try/catch",
                confidence=0.90,
            ))

        # Detect specific exception handling patterns
        specific_except = re.findall(
            r"(?:except|catch)\s*\(\s*(\w+(?:Error|Exception|Fault|Failure)?)\s*\)",
            content,
        )
        if specific_except:
            exceptions_str = ", ".join(sorted(set(specific_except))[:8])
            matches.append(PatternMatch(
                pattern_type="error_handling",
                content=f"Specific exceptions handled: [{exceptions_str}]",
                context=f"Caught {len(set(specific_except))} distinct exception type(s)",
                confidence=0.85,
            ))

        # Detect logging patterns
        logging_calls = re.findall(
            r"(?:logger?\.(\w+)|console\.\w+|logging\.\w+)\s*\(",
            content,
        )
        if logging_calls:
            log_levels = Counter(logging_calls).most_common(5)
            levels_str = ", ".join(f"{lvl}({cnt})" for lvl, cnt in log_levels)
            matches.append(PatternMatch(
                pattern_type="error_handling",
                content=f"Logging: {levels_str}",
                context=f"Found {len(logging_calls)} logging call(s)",
                confidence=0.78,
            ))

        # Detect assertion usage
        assertions = re.findall(r"(?:assert\s+.+)", content)
        if len(assertions) >= 2:
            matches.append(PatternMatch(
                pattern_type="error_handling",
                content=f"Found {len(assertions)} assertion(s)",
                context="Defensive programming via assertions",
                confidence=0.72,
            ))

        return matches

    # -- Utility/helper pattern detection -----------------------------------

    def _detect_utility_patterns(
        self, file_path: str, content: str,
    ) -> List[PatternMatch]:
        """Detect utility functions and helper classes.

        Heuristics:
        * Functions with names starting with ``_`` (private helpers).
        * Functions in files named ``utils``, ``helpers``, ``common``, ``tools``.
        * Functions with generic, reusable signatures (e.g. ``def format_*``,
          ``def parse_*``, ``def validate_*``).
        """
        matches: List[PatternMatch] = []
        filename = Path(file_path).stem.lower()
        is_utility_file = filename in (
            "utils", "utility", "utilities", "helpers", "common",
            "tools", "toolkit", "shared", "base",
        )

        # Find function definitions
        func_defs = re.findall(
            r"def\s+((?:async\s+)?(\w+))\s*\(([^)]*)\)",
            content,
        )
        utility_funcs: List[str] = []
        utility_prefixes = ("format_", "parse_", "validate_", "sanitize_",
                           "normalize_", "convert_", "transform_", "encode_",
                           "decode_", "serialize_", "deserialize_", "hash_",
                           "generate_", "create_", "build_", "make_",
                           "is_", "has_", "check_", "get_", "set_", "compute_",
                           "calculate_", "clamp_", "truncate_", "merge_",
                           "flatten_", "dedupe_", "unique_")

        for full_name, name, params in func_defs:
            is_util = (
                name.startswith("_")
                or any(name.startswith(p) for p in utility_prefixes)
                or is_utility_file
            )
            if is_util:
                utility_funcs.append(name)

        if utility_funcs:
            funcs_str = ", ".join(utility_funcs[:10])
            match_content = f"Utility functions: [{funcs_str}]"
            if len(utility_funcs) > 10:
                match_content += f" ... and {len(utility_funcs) - 10} more"
            matches.append(PatternMatch(
                pattern_type="utility",
                content=match_content,
                context=f"File: {file_path}" + (
                    " (utility/helper file)" if is_utility_file else ""
                ),
                confidence=0.75 if is_utility_file else 0.60,
            ))

        # Detect helper classes
        helper_classes = re.findall(
            r"class\s+(\w+(?:Helper|Util|Utility|Mixin|Base|Abstract|Common|Shared|Wrapper)\w*)\b",
            content,
        )
        if helper_classes:
            classes_str = ", ".join(helper_classes[:5])
            matches.append(PatternMatch(
                pattern_type="utility",
                content=f"Helper classes: [{classes_str}]",
                context=f"Found {len(helper_classes)} utility class(es)",
                confidence=0.80,
            ))

        return matches

    # ------------------------------------------------------------------ #
    #  Storage helper                                                       #
    # ------------------------------------------------------------------ #

    async def _store_result(self, result: ExtractionResult) -> None:
        """Persist a single :class:`ExtractionResult` to the knowledge store.

        Each :class:`PatternMatch` inside the result is stored as an
        individual :class:`KnowledgeEntry`.  The store must be initialized
        before calling this method.

        Parameters
        ----------
        result:
            The extraction result to persist.

        Raises
        ------
        RuntimeError
            If no :class:`KnowledgeStore` was provided.
        """
        if self.store is None:
            raise RuntimeError("No KnowledgeStore provided.")

        for match in result.entries:
            try:
                entry = await self.store.add(
                    title=self._build_title(match, result),
                    content=match.content,
                    category=match.pattern_type,
                    tags=result.tags,
                    source=result.source_type,
                    confidence=match.confidence,
                    metadata={
                        **result.metadata,
                        "pattern_type": match.pattern_type,
                        "context": _truncate(match.context, 300),
                    },
                )
                _logger.debug("Stored knowledge entry: %s", entry)
            except Exception as exc:
                # Log but don't raise — one bad entry shouldn't block the rest
                _logger.warning(
                    "Failed to store individual entry (type=%s): %s",
                    match.pattern_type,
                    exc,
                )

    @staticmethod
    def _build_title(match: PatternMatch, result: ExtractionResult) -> str:
        """Build a short title for a knowledge entry from a match."""
        prefix_map = {
            "solution": "Solution",
            "error": "Error",
            "error_solution": "Error-Solution Pair",
            "decision": "Project Decision",
            "preference": "User Preference",
            "troubleshooting": "Troubleshooting Steps",
            "code_pattern": "Code Pattern",
            "api_usage": "API Usage",
            "configuration": "Configuration",
            "error_handling": "Error Handling",
            "utility": "Utility/Helper",
            "definition": "Definition",
            "procedure": "Procedure",
            "fact": "Fact",
            "best_practice": "Best Practice",
            "heading": "Heading",
            "code_snippet": "Code Snippet",
            "list_item": "List Item",
            "numbered_step": "Step",
            "key_concept": "Key Concept",
        }
        prefix = prefix_map.get(match.pattern_type, "Knowledge")
        content_preview = match.content[:60].replace("\n", " ")
        return f"[{prefix}] {content_preview}"

    # ------------------------------------------------------------------ #
    #  Statistics                                                           #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        """Return extraction statistics accumulated since this instance was
        created.

        Returns
        -------
        dict
            Keys mirror those in :attr:`extraction_stats` plus a
            ``"running_since"`` timestamp.
        """
        return {
            **self.extraction_stats,
            "running_since": _now_iso(),
        }

    def reset_stats(self) -> None:
        """Reset all extraction counters to zero."""
        for key in self.extraction_stats:
            self.extraction_stats[key] = 0
        _logger.info("Extraction stats reset.")
