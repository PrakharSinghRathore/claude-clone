"""
Knowledge Importer Module for Claude Code Clone.

Loads knowledge entries into the :class:`KnowledgeStore` from a wide range of
external sources: Markdown files (with YAML frontmatter), JSON / JSON Lines,
Obsidian vaults, code documentation, and plain text files.

All public import methods return an :class:`ImportResult` dataclass that
describes what was imported, how long it took, and any errors encountered.

The module ships its own lightweight YAML-like frontmatter parser so that no
external dependency (e.g. PyYAML) is required.

Typical usage::

    from agent.knowledge_base.knowledge_store import KnowledgeStore
    from agent.knowledge_base.knowledge_importer import KnowledgeImporter

    store = KnowledgeStore()
    await store.initialize()
    importer = KnowledgeImporter(store)

    result = await importer.import_markdown("docs/patterns.md")
    print(f"Imported {result.total_entries} entries in {result.duration_seconds:.2f}s")

    obsidian_result = await importer.import_obsidian_vault("~/my-vault")
    print(f"Vault: {obsidian_result.total_entries} notes, {len(obsidian_result.errors)} errors")

    json_result = await importer.import_json("knowledge-dump.json")

    await store.close()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.knowledge_base.knowledge_store import KnowledgeEntry, KnowledgeStore

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-exports from knowledge_store so callers only need to import from here
# ---------------------------------------------------------------------------

# Import the constants we need for validation.  Done inside __init__ methods
# rather than at module level to avoid circular imports at load time.
_VALID_CATEGORIES: Tuple[str, ...] = (
    "pattern",
    "solution",
    "concept",
    "troubleshooting",
    "reference",
    "snippet",
    "decision",
    "lesson",
)

_VALID_SOURCES: Tuple[str, ...] = (
    "conversation",
    "code",
    "web",
    "manual",
    "import",
)

# ---------------------------------------------------------------------------
# Regular expression patterns (compiled once, reused everywhere)
# ---------------------------------------------------------------------------

# YAML frontmatter: text between the first two "---" lines at the top of a file.
_RE_FRONTMATTER = re.compile(
    r"^---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)",
    re.DOTALL,
)

# First H1 heading (after frontmatter has been stripped).
_RE_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)

# Any heading (for section-based splitting).
_RE_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Inline ``#tag`` syntax — matches word-boundary-anchored hashtags that are
# NOT inside a code block (the parser strips code blocks first).
_RE_INLINE_TAG = re.compile(r"(?:^|\s)#([a-zA-Z][\w/-]*)")

# Obsidian-style ``[[wikilink]]`` — captures the link target, optionally
# followed by ``|display text``.
_RE_WIKILINK = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]")

# Obsidian-style ``[["double word tag"]]`` inline tag alias.
_RE_OBSIDIAN_INLINE_TAG = re.compile(r'\[\["([^"]+?)"\]\]')

# Markdown formatting to strip for clean content storage.
_RE_CODE_BLOCK = re.compile(r"```[\w]*\n.*?```", re.DOTALL)
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_ITALIC = re.compile(r"\*(.+?)\*")
_RE_LINK_MD = re.compile(r"\[([^\]]+?)\]\([^)]+?\)")
_RE_IMAGE_MD = re.compile(r"!\[([^\]]*?)\]\([^)]+?\)")
_RE_HORIZONTAL_RULE = re.compile(r"^---+$", re.MULTILINE)
_RE_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)

# Python docstring patterns.
_RE_PY_MODULE_DOCSTRING = re.compile(
    r'^(?:["\']{3})(.+?)(?:["\']{3})',
    re.DOTALL,
)
_RE_PY_CLASS_DOCSTRING = re.compile(
    r'class\s+(\w+)[^:]*:.*?"""(.*?)"""',
    re.DOTALL,
)
_RE_PY_FUNC_DOCSTRING = re.compile(
    r'(?:async\s+)?def\s+(\w+)\s*\([^)]*\)(?:\s*->[^:]*)?:.*?"""(.*?)"""',
    re.DOTALL,
)

# JSDoc comment blocks.
_RE_JSDOC = re.compile(r"/\*\*\s*(.*?)\s*\*/", re.DOTALL)

# Inline knowledge annotations: ``# knowledge:`` or ``// knowledge:``
_RE_KNOWLEDGE_ANNOTATION_PY = re.compile(r"#\s*knowledge:\s*(.+)$", re.MULTILINE)
_RE_KNOWLEDGE_ANNOTATION_JS = re.compile(r"//\s*knowledge:\s*(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ImportResult:
    """Summarises the outcome of a single import operation.

    Attributes
    ----------
    total_entries:
        Number of knowledge entries successfully imported and persisted.
    categories:
        Mapping of category name → count of entries in that category.
    tags:
        Mapping of tag name → count of entries carrying that tag.
    errors:
        Human-readable error messages for entries or files that failed.
    source:
        A short label identifying the import source (filepath, directory,
        vault name, etc.).
    duration_seconds:
        Wall-clock time the import took, in seconds.
    """

    total_entries: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    tags: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    source: str = ""
    duration_seconds: float = 0.0

    # -- Convenience helpers --------------------------------------------------

    def merge(self, other: "ImportResult") -> None:
        """Merge *other* into this result in-place.

        Counts are accumulated, errors are appended, and duration is summed.

        Parameters
        ----------
        other:
            Another :class:`ImportResult` to fold into ``self``.
        """
        self.total_entries += other.total_entries
        for cat, cnt in other.categories.items():
            self.categories[cat] = self.categories.get(cat, 0) + cnt
        for tag, cnt in other.tags.items():
            self.tags[tag] = self.tags.get(tag, 0) + cnt
        self.errors.extend(other.errors)
        self.duration_seconds += other.duration_seconds


@dataclass
class ImportedEntry:
    """Represents a single entry extracted from an external source, ready to be
    persisted as a :class:`KnowledgeEntry`.

    This is an intermediate representation: the importer produces
    :class:`ImportedEntry` objects, then converts them to
    :class:`KnowledgeEntry` instances before calling ``store.add()``.

    Attributes
    ----------
    title:
        Short human-readable title for the entry.
    content:
        Full body text (may contain Markdown).
    category:
        One of the valid category strings (default ``"reference"``).
    tags:
        Free-form list of tags.
    source:
        Origin label (filepath, URL, etc.).
    confidence:
        Reliability score in [0.0, 1.0].
    importance:
        Priority / relevance score in [0.0, 1.0].
    metadata:
        Arbitrary extra key-value data.
    """

    title: str = ""
    content: str = ""
    category: str = "reference"
    tags: List[str] = field(default_factory=list)
    source: str = ""
    confidence: float = 0.6
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Frontmatter parser (no PyYAML dependency)
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Extract and parse YAML-like frontmatter from the top of *text*.

    Frontmatter is the block between the first two ``---`` lines.  Supports:

    * Scalar values: ``key: value``
    * Quoted strings: ``key: "string value"`` or ``key: 'string value'``
    * Lists: ``key: [a, b, c]`` or ``key:\\n  - a\\n  - b``
    * Numbers: ``key: 42`` or ``key: 3.14``
    * Booleans: ``key: true`` / ``key: false``

    Parameters
    ----------
    text:
        Raw document text that may begin with ``---`` frontmatter.

    Returns
    -------
    tuple[dict, str]
        A 2-tuple of ``(metadata_dict, remaining_content)``.  If no
        frontmatter is found, returns ``({}, text)``.
    """

    match = _RE_FRONTMATTER.match(text)
    if not match:
        return {}, text

    raw_frontmatter = match.group(1)
    remaining = text[match.end():]

    metadata: Dict[str, Any] = {}
    for line in raw_frontmatter.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Split on first colon.
        colon_idx = line.find(":")
        if colon_idx < 0:
            continue

        key = line[:colon_idx].strip().lower()
        value_str = line[colon_idx + 1:].strip()

        # Skip keys with no name.
        if not key:
            continue

        metadata[key] = _parse_yaml_value(value_str)

    return metadata, remaining


def _parse_yaml_value(value_str: str) -> Any:
    """Parse a single YAML-style value string into a Python object.

    Handles quoted strings, booleans, numbers, and bracket lists.

    Parameters
    ----------
    value_str:
        The raw value portion of a ``key: value`` line.

    Returns
    -------
    str | int | float | bool | list
        The parsed Python value.
    """
    if not value_str:
        return ""

    # Quoted string (double or single quotes).
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return value_str[1:-1]

    # Boolean.
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # Integer.
    try:
        return int(value_str)
    except ValueError:
        pass

    # Float.
    try:
        return float(value_str)
    except ValueError:
        pass

    # Bracket-delimited list: ``[tag1, tag2, tag3]``
    if value_str.startswith("[") and value_str.endswith("]"):
        inner = value_str[1:-1].strip()
        if not inner:
            return []
        items: List[str] = []
        # Split on commas, strip whitespace and quotes from each item.
        for item in inner.split(","):
            item = item.strip().strip('"').strip("'")
            if item:
                items.append(item)
        return items

    # Comma-separated list (no brackets).
    # Only treated as a list if it contains at least one comma.
    if "," in value_str:
        items = [v.strip().strip('"').strip("'") for v in value_str.split(",")]
        items = [v for v in items if v]
        if len(items) > 1:
            return items

    # Fallback: plain string.
    return value_str


# ---------------------------------------------------------------------------
# Tag extraction helpers
# ---------------------------------------------------------------------------


def _extract_inline_tags(text: str) -> List[str]:
    """Extract ``#tag`` references from the body text.

    Matches hashtags that begin a line or follow whitespace, consisting of
    a letter followed by word characters, hyphens, or forward slashes.
    Common heading markers (``# heading``) are excluded by requiring the
    tag to start with a letter.

    Parameters
    ----------
    text:
        Markdown body text to scan for tags.

    Returns
    -------
    list[str]
        Deduplicated list of lowercase tags found in the text.
    """
    # Strip code blocks first so tags inside code aren't extracted.
    clean = _RE_CODE_BLOCK.sub("", text)
    matches = _RE_INLINE_TAG.findall(clean)
    seen: Set[str] = set()
    tags: List[str] = []
    for tag in matches:
        low = tag.lower()
        if low not in seen:
            seen.add(low)
            tags.append(low)
    return tags


def _extract_obsidian_inline_tags(text: str) -> List[str]:
    """Extract Obsidian-style ``[["double word tag"]]`` inline tag aliases.

    Parameters
    ----------
    text:
        Markdown body text to scan.

    Returns
    -------
    list[str]
        Deduplicated list of lowercase tags.
    """
    matches = _RE_OBSIDIAN_INLINE_TAG.findall(text)
    seen: Set[str] = set()
    tags: List[str] = []
    for tag in matches:
        low = tag.lower().strip()
        if low not in seen:
            seen.add(low)
            tags.append(low)
    return tags


def _extract_frontmatter_tags(metadata: Dict[str, Any]) -> List[str]:
    """Extract tags from frontmatter ``tags`` and ``keywords`` fields.

    Handles both list values (``[tag1, tag2]``) and comma-separated strings
    (``"tag1, tag2"``).

    Parameters
    ----------
    metadata:
        The parsed frontmatter dictionary.

    Returns
    -------
    list[str]
        Combined, deduplicated, lowercase tags.
    """
    tags: List[str] = []
    seen: Set[str] = set()

    for key in ("tags", "keywords"):
        value = metadata.get(key)
        if value is None:
            continue

        if isinstance(value, list):
            items = [str(v) for v in value]
        elif isinstance(value, str):
            items = [v.strip() for v in value.split(",")]
        else:
            continue

        for item in items:
            item = item.strip().strip('"').strip("'").lower()
            if item and item not in seen:
                seen.add(item)
                tags.append(item)

    return tags


def _extract_wikilinks(text: str) -> List[str]:
    """Parse ``[[link name]]`` and ``[[link name|display text]]`` wikilinks.

    Handles:

    * ``[[Simple Link]]`` → ``"simple link"``
    * ``[[link name|Display Text]]`` → ``"link name"``
    * ``[[folder/note]]`` → ``"folder/note"``

    Parameters
    ----------
    text:
        Markdown body text to scan.

    Returns
    -------
    list[str]
        Deduplicated list of wikilink targets (lowercased).
    """
    matches = _RE_WIKILINK.findall(text)
    seen: Set[str] = set()
    links: List[str] = []
    for link in matches:
        link = link.strip()
        if "/" in link:
            # Preserve folder structure as a tag-like path.
            low = link.lower()
        else:
            low = link.lower()
        if low not in seen:
            seen.add(low)
            links.append(low)
    return links


def _infer_category_from_content(title: str, content: str) -> str:
    """Heuristically infer a knowledge category from title and content text.

    Looks for keyword signals in the title (highest priority) and then in
    the first 500 characters of content.

    Parameters
    ----------
    title:
        The entry title.
    content:
        The entry body text.

    Returns
    -------
    str
        One of the valid category strings, defaulting to ``"reference"``.
    """
    text = (title + "\n" + content[:500]).lower()

    category_signals: Dict[str, List[str]] = {
        "pattern": ["pattern", "design pattern", "idiom", "template method"],
        "solution": ["solution", "fix", "how to fix", "how to solve", "resolved"],
        "concept": ["concept", "definition", "what is", "overview", "introduction"],
        "troubleshooting": [
            "troubleshoot", "debug", "error", "issue", "problem",
            "bug", "workaround", "diagnose",
        ],
        "snippet": ["snippet", "example", "code sample", "quick reference"],
        "decision": ["decision", "adr", "architecture decision", "chose", "we decided"],
        "lesson": ["lesson", "learned", "takeaway", "insight", "retrospective"],
    }

    # Score each category.
    scores: Dict[str, int] = {}
    for cat, signals in category_signals.items():
        score = sum(1 for s in signals if s in text)
        if score > 0:
            scores[cat] = score

    if not scores:
        return "reference"

    return max(scores, key=scores.get)  # type: ignore[arg-type]


def _infer_category_from_tags(tags: List[str]) -> str:
    """Map common tags to knowledge categories.

    Parameters
    ----------
    tags:
        List of tags assigned to the entry.

    Returns
    -------
    str
        Inferred category, or ``"reference"`` if no mapping applies.
    """
    tag_map: Dict[str, str] = {
        "pattern": "pattern",
        "design-pattern": "pattern",
        "solution": "solution",
        "fix": "solution",
        "concept": "concept",
        "definition": "concept",
        "troubleshooting": "troubleshooting",
        "debugging": "troubleshooting",
        "snippet": "snippet",
        "code": "snippet",
        "decision": "decision",
        "adr": "decision",
        "lesson": "lesson",
        "learned": "lesson",
        "reference": "reference",
        "docs": "reference",
        "documentation": "reference",
    }

    tag_set = {t.lower() for t in tags}
    for tag, cat in tag_map.items():
        if tag in tag_set:
            return cat

    return "reference"


def _infer_category_from_folders(relative_path: Path) -> str:
    """Use folder structure as a hint for the note category.

    Mapping examples:

    * ``patterns/...`` → ``"pattern"``
    * ``solutions/...`` → ``"solution"``
    * ``concepts/...`` → ``"concept"``
    * ``troubleshooting/...`` → ``"troubleshooting"``
    * ``snippets/...`` → ``"snippet"``
    * ``decisions/...`` → ``"decision"``

    Parameters
    ----------
    relative_path:
        Path relative to the vault / import root.

    Returns
    -------
    str
        Inferred category, or ``"reference"`` if no match.
    """
    folder_map: Dict[str, str] = {
        "pattern": "pattern",
        "patterns": "pattern",
        "design-patterns": "pattern",
        "solution": "solution",
        "solutions": "solution",
        "concept": "concept",
        "concepts": "concept",
        "troubleshoot": "troubleshooting",
        "troubleshooting": "troubleshooting",
        "debug": "troubleshooting",
        "debugging": "troubleshooting",
        "snippet": "snippet",
        "snippets": "snippet",
        "examples": "snippet",
        "decision": "decision",
        "decisions": "decision",
        "adr": "decision",
        "lesson": "lesson",
        "lessons": "lesson",
        "learnings": "lesson",
        "reference": "reference",
        "references": "reference",
        "docs": "reference",
    }

    parts = relative_path.parent.parts
    for part in parts:
        if part.lower() in folder_map:
            return folder_map[part.lower()]

    return "reference"


# ---------------------------------------------------------------------------
# Markdown content cleaning
# ---------------------------------------------------------------------------


def _strip_markdown_formatting(text: str) -> str:
    """Remove Markdown formatting artefacts to produce clean prose text.

    Performs the following transformations in order:

    1. Remove fenced code blocks entirely.
    2. Convert inline code to its content without backticks.
    3. Strip image syntax (keep alt text).
    4. Strip link syntax (keep display text).
    5. Remove bold / italic markers.
    6. Remove blockquote prefixes.
    7. Remove horizontal rules.
    8. Collapse excessive whitespace.

    Parameters
    ----------
    text:
        Raw Markdown text.

    Returns
    -------
    str
        Cleaned text suitable for storage or search indexing.
    """
    result = _RE_CODE_BLOCK.sub("", text)
    result = _RE_INLINE_CODE.sub(r"\1", result)
    result = _RE_IMAGE_MD.sub(r"\1", result)
    result = _RE_LINK_MD.sub(r"\1", result)
    result = _RE_BOLD.sub(r"\1", result)
    result = _RE_ITALIC.sub(r"\1", result)
    result = _RE_BLOCKQUOTE.sub("", result)
    result = _RE_HORIZONTAL_RULE.sub("", result)

    # Collapse multiple blank lines into at most two.
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def _folder_path_to_tags(relative_path: Path) -> List[str]:
    """Convert a relative file path into tags derived from folder names.

    For example, ``programming/python/async.md`` yields
    ``["programming", "python"]``.

    Parameters
    ----------
    relative_path:
        Path relative to the root / vault directory.

    Returns
    -------
    list[str]
        Lowercase folder names as tags.
    """
    tags: List[str] = []
    for part in relative_path.parent.parts:
        tag = part.lower().strip()
        if tag and tag not in (".", ".."):
            tags.append(tag)
    return tags


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_category(category: str) -> str:
    """Return *category* if valid, else fall back to ``"reference"``.

    Logs a warning on invalid input.

    Parameters
    ----------
    category:
        Proposed category string.

    Returns
    -------
    str
        A valid category string.
    """
    if category in _VALID_CATEGORIES:
        return category
    _logger.warning(
        "Invalid category %r; falling back to 'reference'. "
        "Valid categories: %s",
        category,
        _VALID_CATEGORIES,
    )
    return "reference"


# ---------------------------------------------------------------------------
# KnowledgeImporter
# ---------------------------------------------------------------------------


class KnowledgeImporter:
    """Import knowledge entries from external files and directories into a
    :class:`KnowledgeStore`.

    Supports Markdown (with YAML frontmatter), JSON, JSON Lines, Obsidian
    vaults, code documentation (Python docstrings, JSDoc, inline annotations),
    and generic plain-text files.

    Every public ``import_*`` method is ``async`` and returns an
    :class:`ImportResult` summarising the outcome.

    Parameters
    ----------
    store:
        An initialised :class:`KnowledgeStore` instance that entries will be
        persisted into.
    default_confidence:
        Default confidence score for imported entries that do not specify one.
        Must be in [0.0, 1.0].  Defaults to ``0.6``.
    default_importance:
        Default importance score for imported entries that do not specify one.
        Must be in [0.0, 1.0].  Defaults to ``0.5``.
    """

    def __init__(
        self,
        store: "KnowledgeStore",
        default_confidence: float = 0.6,
        default_importance: float = 0.5,
    ) -> None:
        self.store = store
        self.default_confidence = max(0.0, min(1.0, default_confidence))
        self.default_importance = max(0.0, min(1.0, default_importance))

        # Running statistics across all imports performed by this instance.
        self._import_stats: Dict[str, Any] = {
            "total_imports": 0,
            "total_entries_imported": 0,
            "total_errors": 0,
            "sources": [],
            "files_processed": 0,
        }

    # ================================================================== #
    #  Async bridge helper                                                #
    # ================================================================== #

    async def _persist_entry(self, imported: ImportedEntry) -> Optional[str]:
        """Convert an :class:`ImportedEntry` to a :class:`KnowledgeEntry` and
        persist it via the store.

        Parameters
        ----------
        imported:
            The intermediate entry to store.

        Returns
        -------
        str | None
            The entry id if persisted successfully, or ``None`` on error.
        """
        from agent.knowledge_base.knowledge_store import KnowledgeEntry

        category = _validate_category(imported.category)
        confidence = max(0.0, min(1.0, imported.confidence))
        importance = max(0.0, min(1.0, imported.importance))

        entry = KnowledgeEntry(
            title=imported.title,
            content=imported.content,
            category=category,
            tags=imported.tags,
            source="import",
            confidence=confidence,
            importance=importance,
            metadata=imported.metadata,
        )

        try:
            entry_id = await self.store.add(entry)
            return entry_id
        except Exception as exc:
            _logger.error(
                "Failed to persist entry %r: %s",
                imported.title[:80],
                exc,
            )
            return None

    # ================================================================== #
    #  Markdown import                                                    #
    # ================================================================== #

    async def import_markdown(
        self,
        filepath: str,
        category: str = "reference",
    ) -> ImportResult:
        """Import a single Markdown file into the knowledge base.

        Processing steps:

        1. Read the file contents.
        2. Parse YAML frontmatter (``---`` block) for metadata, tags, and
           an optional category override.
        3. Extract the title from the first ``# Heading`` line, falling back
           to the filename (without extension) if no H1 is found.
        4. Extract tags from frontmatter ``tags``/``keywords`` fields AND
           from inline ``#tag`` syntax in the body.
        5. Infer category from frontmatter or content heuristics.
        6. Strip heavy Markdown formatting for clean content storage.
        7. Persist the entry.

        Parameters
        ----------
        filepath:
            Path to the ``.md`` file to import.
        category:
            Default category to use if none is found in frontmatter or
            inferred from content.  Defaults to ``"reference"``.

        Returns
        -------
        ImportResult
            Summary of the import operation.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        """
        start = time.monotonic()
        result = ImportResult(source=filepath)

        path = Path(filepath).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {filepath}")

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result.errors.append(f"Failed to read {filepath}: {exc}")
            result.duration_seconds = time.monotonic() - start
            return result

        if not raw.strip():
            result.errors.append(f"File is empty: {filepath}")
            result.duration_seconds = time.monotonic() - start
            return result

        # Parse frontmatter.
        metadata, body = _parse_frontmatter(raw)

        # Extract tags.
        tags: List[str] = []
        tags.extend(_extract_frontmatter_tags(metadata))
        tags.extend(_extract_inline_tags(body))
        # Deduplicate while preserving order.
        seen: Set[str] = set()
        unique_tags: List[str] = []
        for t in tags:
            low = t.lower()
            if low not in seen:
                seen.add(low)
                unique_tags.append(low)
        tags = unique_tags

        # Extract title.
        h1_match = _RE_H1.search(body)
        if h1_match:
            title = h1_match.group(1).strip()
            # Remove the H1 line from the body content.
            body = body[h1_match.end():].strip()
        else:
            title = metadata.get("title", path.stem)
            if not isinstance(title, str):
                title = str(title)

        # Determine category.
        fm_category = metadata.get("category", "")
        if isinstance(fm_category, str) and fm_category.strip():
            resolved_category = _validate_category(fm_category.strip())
        else:
            resolved_category = category
            # Try inferring from tags first, then from content.
            inferred = _infer_category_from_tags(tags)
            if inferred != "reference":
                resolved_category = inferred
            else:
                content_inferred = _infer_category_from_content(title, body)
                if content_inferred != "reference":
                    resolved_category = content_inferred

        # Clean content for storage (keep original Markdown in metadata).
        clean_content = _strip_markdown_formatting(body)

        # Build metadata dict.
        entry_metadata: Dict[str, Any] = {
            "import_file": str(path),
            "import_filename": path.name,
            "original_format": "markdown",
        }
        if metadata:
            # Store the original frontmatter under a separate key.
            entry_metadata["frontmatter"] = metadata

        # Build imported entry.
        confidence = self.default_confidence
        if "confidence" in metadata:
            try:
                confidence = float(metadata["confidence"])
            except (ValueError, TypeError):
                pass

        importance = self.default_importance
        if "importance" in metadata:
            try:
                importance = float(metadata["importance"])
            except (ValueError, TypeError):
                pass

        imported = ImportedEntry(
            title=title,
            content=clean_content if clean_content else body,
            category=resolved_category,
            tags=tags,
            source=str(path),
            confidence=confidence,
            importance=importance,
            metadata=entry_metadata,
        )

        entry_id = await self._persist_entry(imported)
        if entry_id:
            result.total_entries = 1
            result.categories[resolved_category] = 1
            for t in tags:
                result.tags[t] = result.tags.get(t, 0) + 1
            _logger.info("Imported markdown entry %s: %s", entry_id, title)
        else:
            result.errors.append(f"Failed to persist entry from {filepath}")

        # Update instance stats.
        self._import_stats["total_imports"] += 1
        self._import_stats["total_entries_imported"] += result.total_entries
        self._import_stats["total_errors"] += len(result.errors)
        self._import_stats["files_processed"] += 1
        self._import_stats["sources"].append(filepath)

        result.duration_seconds = time.monotonic() - start
        return result

    # ================================================================== #
    #  Markdown directory import                                          #
    # ================================================================== #

    async def import_markdown_directory(
        self,
        dirpath: str,
        recursive: bool = True,
    ) -> ImportResult:
        """Import all Markdown files from a directory into the knowledge base.

        Processing details:

        * Walks the directory (optionally recursively) looking for ``.md``
          and ``.markdown`` files.
        * Skips hidden files and directories (names starting with ``.``).
        * Respects a ``.markdown-import-ignore`` file in the root directory
          — each line is a glob pattern for files/dirs to skip.
        * Continues importing remaining files even if some fail, collecting
          errors for reporting.

        Parameters
        ----------
        dirpath:
            Path to the directory containing Markdown files.
        recursive:
            If ``True`` (default), recurse into subdirectories.

        Returns
        -------
        ImportResult
            Aggregated summary across all files processed.

        Raises
        ------
        FileNotFoundError
            If *dirpath* does not exist or is not a directory.
        """
        start = time.monotonic()
        result = ImportResult(source=dirpath)

        root = Path(dirpath).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {dirpath}")

        # Load ignore patterns.
        ignore_file = root / ".markdown-import-ignore"
        ignore_patterns: List[str] = []
        if ignore_file.exists():
            try:
                ignore_patterns = [
                    line.strip()
                    for line in ignore_file.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("#")
                ]
            except OSError as exc:
                _logger.warning("Could not read %s: %s", ignore_file, exc)

        # Collect .md files.
        md_files: List[Path] = []
        if recursive:
            for dirpath_curr, dirnames, filenames in os.walk(root):
                current = Path(dirpath_curr)
                # Filter out hidden directories in-place (affects os.walk).
                dirnames[:] = [
                    d for d in dirnames
                    if not d.startswith(".")
                    and not self._matches_ignore_patterns(
                        str(current / d), ignore_patterns, root
                    )
                ]
                for fname in sorted(filenames):
                    if fname.startswith("."):
                        continue
                    if not fname.lower().endswith((".md", ".markdown")):
                        continue
                    full_path = current / fname
                    if not self._matches_ignore_patterns(
                        str(full_path), ignore_patterns, root
                    ):
                        md_files.append(full_path)
        else:
            for fname in sorted(os.listdir(root)):
                if fname.startswith("."):
                    continue
                if not fname.lower().endswith((".md", ".markdown")):
                    continue
                full_path = root / fname
                if full_path.is_file() and not self._matches_ignore_patterns(
                    str(full_path), ignore_patterns, root
                ):
                    md_files.append(full_path)

        _logger.info(
            "Found %d markdown files in %s (recursive=%s)",
            len(md_files),
            dirpath,
            recursive,
        )

        # Import each file, merging results.
        for md_file in md_files:
            try:
                file_result = await self.import_markdown(str(md_file))
                result.merge(file_result)
            except Exception as exc:
                result.errors.append(f"Error importing {md_file}: {exc}")
                _logger.error("Error importing %s: %s", md_file, exc)

        # Update instance stats.
        self._import_stats["total_imports"] += 1
        self._import_stats["total_entries_imported"] += result.total_entries
        self._import_stats["total_errors"] += len(result.errors)
        self._import_stats["files_processed"] += len(md_files)
        self._import_stats["sources"].append(dirpath)

        result.duration_seconds = time.monotonic() - start
        return result

    @staticmethod
    def _matches_ignore_patterns(
        path_str: str,
        patterns: List[str],
        root: Path,
    ) -> bool:
        """Check if *path_str* matches any of the gitignore-style *patterns*.

        Matching is performed against the path relative to *root*.

        Parameters
        ----------
        path_str:
            Absolute path to check.
        patterns:
            List of glob patterns from the ignore file.
        root:
            Root directory for relative path calculation.

        Returns
        -------
        bool
            ``True`` if the path matches any ignore pattern.
        """
        if not patterns:
            return False

        try:
            rel = str(Path(path_str).resolve().relative_to(root))
        except ValueError:
            return False

        from fnmatch import fnmatch

        for pat in patterns:
            # Support both matching the filename and the full relative path.
            if fnmatch(Path(path_str).name, pat) or fnmatch(rel, pat):
                return True

        return False

    # ================================================================== #
    #  Obsidian vault import                                              #
    # ================================================================== #

    async def import_obsidian_vault(self, vault_path: str) -> ImportResult:
        """Import an entire Obsidian vault into the knowledge base.

        This is the most feature-rich importer, handling:

        * **Frontmatter**: Parses the full YAML-like block for ``title``,
          ``tags``, ``aliases``, ``created``, ``modified``, ``category``,
          ``type``, and custom metadata fields.
        * **Inline tags**: ``#single-word-tags`` anywhere in note content.
        * **Obsidian inline tag aliases**: ``[["double word tags"]]``.
        * **Wikilinks**: ``[[link name]]`` and ``[[link name|display]]``
          are extracted as tags/references in the metadata.
        * **Aliases**: The ``aliases:`` frontmatter field provides
          alternative titles; the first alias is used as a metadata tag.
        * **Timestamps**: ``created:`` and ``modified:`` from frontmatter
          are passed through to the entry metadata.
        * **Note type detection**: Tags and folder names are used to infer
          note types (pattern, reference, concept, etc.).
        * **Folder-as-category**: Parent folder names become category
          hints and tags (e.g. ``programming/python/`` → tags ``["programming",
          "python"]``).

        Parameters
        ----------
        vault_path:
            Path to the root directory of the Obsidian vault.

        Returns
        -------
        ImportResult
            Aggregated summary across all notes in the vault.

        Raises
        ------
        FileNotFoundError
            If *vault_path* does not exist or is not a directory.
        """
        start = time.monotonic()
        result = ImportResult(source=vault_path)

        vault_root = Path(vault_path).resolve()
        if not vault_root.is_dir():
            raise FileNotFoundError(f"Obsidian vault not found: {vault_path}")

        # Collect all .md files (skip .obsidian and .trash directories).
        md_files: List[Path] = []
        for dirpath_curr, dirnames, filenames in os.walk(vault_root):
            current = Path(dirpath_curr)
            # Filter out Obsidian-internal and hidden directories.
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".")
            ]
            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                if not fname.lower().endswith((".md", ".markdown")):
                    continue
                md_files.append(current / fname)

        _logger.info(
            "Obsidian vault: found %d notes in %s",
            len(md_files),
            vault_path,
        )

        for md_file in md_files:
            try:
                file_result = await self._import_obsidian_note(md_file, vault_root)
                result.merge(file_result)
            except Exception as exc:
                result.errors.append(f"Error importing note {md_file}: {exc}")
                _logger.error("Error importing Obsidian note %s: %s", md_file, exc)

        # Update instance stats.
        self._import_stats["total_imports"] += 1
        self._import_stats["total_entries_imported"] += result.total_entries
        self._import_stats["total_errors"] += len(result.errors)
        self._import_stats["files_processed"] += len(md_files)
        self._import_stats["sources"].append(vault_path)

        result.duration_seconds = time.monotonic() - start
        return result

    async def _import_obsidian_note(
        self,
        note_path: Path,
        vault_root: Path,
    ) -> ImportResult:
        """Import a single Obsidian note file.

        Parameters
        ----------
        note_path:
            Absolute path to the note file.
        vault_root:
            Absolute path to the vault root directory.

        Returns
        -------
        ImportResult
            Summary for this single note import.
        """
        result = ImportResult(source=str(note_path))

        try:
            raw = note_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result.errors.append(f"Failed to read {note_path}: {exc}")
            return result

        if not raw.strip():
            return result

        relative_path = note_path.relative_to(vault_root)

        # Parse frontmatter.
        metadata, body = _parse_frontmatter(raw)

        # --- Title ---------------------------------------------------------
        title = metadata.get("title", "")
        if not isinstance(title, str) or not title.strip():
            # Fall back to first H1, then filename.
            h1_match = _RE_H1.search(body)
            if h1_match:
                title = h1_match.group(1).strip()
                body = body[h1_match.end():].strip()
            else:
                title = note_path.stem

        # --- Aliases -------------------------------------------------------
        aliases: List[str] = metadata.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [a.strip() for a in aliases.split(",") if a.strip()]
        elif not isinstance(aliases, list):
            aliases = []

        # --- Tags ----------------------------------------------------------
        tags: List[str] = []

        # From frontmatter.
        tags.extend(_extract_frontmatter_tags(metadata))

        # From inline #tag syntax.
        tags.extend(_extract_inline_tags(body))

        # From Obsidian [["inline tag"]] syntax.
        tags.extend(_extract_obsidian_inline_tags(body))

        # From folder structure.
        tags.extend(_folder_path_to_tags(relative_path))

        # Deduplicate.
        seen_tags: Set[str] = set()
        unique_tags: List[str] = []
        for t in tags:
            low = t.lower().strip()
            if low and low not in seen_tags:
                seen_tags.add(low)
                unique_tags.append(low)
        tags = unique_tags

        # --- Wikilinks (stored in metadata, not as tags) -------------------
        wikilinks = _extract_wikilinks(body)

        # --- Category inference --------------------------------------------
        fm_category = metadata.get("category", "")
        note_type = metadata.get("type", "")

        if isinstance(fm_category, str) and fm_category.strip():
            category = _validate_category(fm_category.strip())
        elif isinstance(note_type, str) and note_type.strip():
            category = _validate_category(note_type.strip())
        else:
            # Infer from folder, then tags, then content.
            folder_cat = _infer_category_from_folders(relative_path)
            tag_cat = _infer_category_from_tags(tags)
            content_cat = _infer_category_from_content(title, body)

            # Priority: folder > tags > content.
            if folder_cat != "reference":
                category = folder_cat
            elif tag_cat != "reference":
                category = tag_cat
            else:
                category = content_cat

        # --- Timestamps ----------------------------------------------------
        created_ts = metadata.get("created", "")
        modified_ts = metadata.get("modified", "")

        # --- Clean content -------------------------------------------------
        clean_content = _strip_markdown_formatting(body)

        # --- Build metadata ------------------------------------------------
        entry_metadata: Dict[str, Any] = {
            "import_file": str(note_path),
            "import_filename": note_path.name,
            "vault_path": str(vault_root),
            "vault_relative_path": str(relative_path),
            "original_format": "obsidian_markdown",
            "aliases": aliases,
        }
        if wikilinks:
            entry_metadata["wikilinks"] = wikilinks
        if created_ts:
            entry_metadata["obsidian_created"] = str(created_ts)
        if modified_ts:
            entry_metadata["obsidian_modified"] = str(modified_ts)
        if metadata:
            entry_metadata["frontmatter"] = metadata

        # --- Confidence / importance ---------------------------------------
        confidence = self.default_confidence
        if "confidence" in metadata:
            try:
                confidence = float(metadata["confidence"])
            except (ValueError, TypeError):
                pass
        importance = self.default_importance
        if "importance" in metadata:
            try:
                importance = float(metadata["importance"])
            except (ValueError, TypeError):
                pass

        # --- Persist -------------------------------------------------------
        imported = ImportedEntry(
            title=title,
            content=clean_content if clean_content else body,
            category=category,
            tags=tags,
            source=str(note_path),
            confidence=confidence,
            importance=importance,
            metadata=entry_metadata,
        )

        entry_id = await self._persist_entry(imported)
        if entry_id:
            result.total_entries = 1
            result.categories[category] = result.categories.get(category, 0) + 1
            for t in tags:
                result.tags[t] = result.tags.get(t, 0) + 1
            _logger.debug("Imported Obsidian note %s: %s", entry_id, title)
        else:
            result.errors.append(f"Failed to persist note: {note_path}")

        return result

    # ================================================================== #
    #  JSON import                                                        #
    # ================================================================== #

    async def import_json(self, filepath: str) -> ImportResult:
        """Import knowledge entries from a JSON file.

        Supports two formats:

        **Object format** (preferred):

        .. code-block:: json

            {
                "entries": [
                    {
                        "title": "Entry title",
                        "content": "Entry body text",
                        "category": "pattern",
                        "tags": ["python", "async"],
                        "confidence": 0.9,
                        "importance": 0.7
                    }
                ]
            }

        **Array format** (top-level list):

        .. code-block:: json

            [
                {
                    "title": "Entry title",
                    "content": "Entry body text"
                }
            ]

        Validation rules:

        * Each entry must have a non-empty ``title`` and ``content`` field.
        * ``category`` defaults to ``"reference"`` if missing or invalid.
        * ``tags`` defaults to an empty list.
        * ``confidence`` and ``importance`` default to the importer's
          configured defaults.
        * Unknown fields are silently ignored.

        Parameters
        ----------
        filepath:
            Path to the JSON file to import.

        Returns
        -------
        ImportResult
            Summary of the import operation.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        """
        start = time.monotonic()
        result = ImportResult(source=filepath)

        path = Path(filepath).resolve()
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            result.errors.append(f"Invalid JSON in {filepath}: {exc}")
            result.duration_seconds = time.monotonic() - start
            return result
        except OSError as exc:
            result.errors.append(f"Failed to read {filepath}: {exc}")
            result.duration_seconds = time.monotonic() - start
            return result

        # Normalise to a list of entry dicts.
        entries_data: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            # Object format: look for "entries" key.
            entries_data = data.get("entries", [])
            if not isinstance(entries_data, list):
                result.errors.append(
                    f"JSON 'entries' field must be a list, got {type(entries_data).__name__}"
                )
                result.duration_seconds = time.monotonic() - start
                return result
        elif isinstance(data, list):
            # Array format.
            entries_data = data
        else:
            result.errors.append(
                f"JSON root must be an object or array, got {type(data).__name__}"
            )
            result.duration_seconds = time.monotonic() - start
            return result

        # Process each entry.
        for idx, entry_dict in enumerate(entries_data):
            if not isinstance(entry_dict, dict):
                result.errors.append(f"Entry {idx}: expected object, got {type(entry_dict).__name__}")
                continue

            entry_result = self._process_json_entry(entry_dict, idx, filepath)
            if entry_result is not None:
                imported, tags_list, category = entry_result
                entry_id = await self._persist_entry(imported)
                if entry_id:
                    result.total_entries += 1
                    result.categories[category] = result.categories.get(category, 0) + 1
                    for t in tags_list:
                        result.tags[t] = result.tags.get(t, 0) + 1
                else:
                    result.errors.append(
                        f"Entry {idx}: failed to persist '{entry_dict.get('title', '<no title>')}'"
                    )
            # else: validation error already added to result.errors.

        # Update instance stats.
        self._import_stats["total_imports"] += 1
        self._import_stats["total_entries_imported"] += result.total_entries
        self._import_stats["total_errors"] += len(result.errors)
        self._import_stats["files_processed"] += 1
        self._import_stats["sources"].append(filepath)

        result.duration_seconds = time.monotonic() - start
        return result

    def _process_json_entry(
        self,
        entry_dict: Dict[str, Any],
        index: int,
        source: str,
    ) -> Optional[Tuple[ImportedEntry, List[str], str]]:
        """Validate and convert a single JSON entry dict to an
        :class:`ImportedEntry`.

        Parameters
        ----------
        entry_dict:
            The parsed JSON object for one entry.
        index:
            Zero-based index of this entry in the source file (for error
            messages).
        source:
            Source filepath string (for metadata).

        Returns
        -------
        tuple[ImportedEntry, list[str], str] | None
            A 3-tuple of ``(imported_entry, tags, category)`` on success,
            or ``None`` on validation failure (caller should not count it).
        """
        title = str(entry_dict.get("title", "")).strip()
        content = str(entry_dict.get("content", "")).strip()

        if not title:
            _logger.warning("JSON entry %d: missing or empty title", index)
            return None
        if not content:
            _logger.warning("JSON entry %d: missing or empty content", index)
            return None

        raw_category = str(entry_dict.get("category", "reference")).strip()
        category = _validate_category(raw_category)

        raw_tags = entry_dict.get("tags", [])
        if isinstance(raw_tags, list):
            tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()]
        elif isinstance(raw_tags, str):
            tags = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]
        else:
            tags = []

        confidence = self.default_confidence
        if "confidence" in entry_dict:
            try:
                confidence = float(entry_dict["confidence"])
            except (ValueError, TypeError):
                pass

        importance = self.default_importance
        if "importance" in entry_dict:
            try:
                importance = float(entry_dict["importance"])
            except (ValueError, TypeError):
                pass

        # Collect any extra fields into metadata.
        reserved_keys = {"title", "content", "category", "tags", "confidence", "importance"}
        extra_metadata: Dict[str, Any] = {
            k: v for k, v in entry_dict.items() if k not in reserved_keys
        }

        entry_metadata: Dict[str, Any] = {
            "import_file": source,
            "original_format": "json",
        }
        if extra_metadata:
            entry_metadata["extra_fields"] = extra_metadata

        imported = ImportedEntry(
            title=title,
            content=content,
            category=category,
            tags=tags,
            source=source,
            confidence=confidence,
            importance=importance,
            metadata=entry_metadata,
        )

        return imported, tags, category

    # ================================================================== #
    #  JSON Lines import                                                  #
    # ================================================================== #

    async def import_jsonl(self, filepath: str) -> ImportResult:
        """Import knowledge entries from a JSON Lines file (one JSON object
        per line).

        Each line must be a valid JSON object with at least ``title`` and
        ``content`` fields.  Blank lines and lines starting with ``#`` are
        skipped.  Malformed lines are logged as errors but do not abort the
        import.

        Parameters
        ----------
        filepath:
            Path to the ``.jsonl`` (or ``.ndjson``) file.

        Returns
        -------
        ImportResult
            Summary of the import operation.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        """
        start = time.monotonic()
        result = ImportResult(source=filepath)

        path = Path(filepath).resolve()
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {filepath}")

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            result.errors.append(f"Failed to read {filepath}: {exc}")
            result.duration_seconds = time.monotonic() - start
            return result

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                entry_dict = json.loads(line)
            except json.JSONDecodeError as exc:
                result.errors.append(f"Line {line_num}: invalid JSON: {exc}")
                continue

            if not isinstance(entry_dict, dict):
                result.errors.append(
                    f"Line {line_num}: expected JSON object, got {type(entry_dict).__name__}"
                )
                continue

            entry_result = self._process_json_entry(entry_dict, line_num, filepath)
            if entry_result is not None:
                imported, tags_list, category = entry_result
                entry_id = await self._persist_entry(imported)
                if entry_id:
                    result.total_entries += 1
                    result.categories[category] = result.categories.get(category, 0) + 1
                    for t in tags_list:
                        result.tags[t] = result.tags.get(t, 0) + 1
                else:
                    result.errors.append(
                        f"Line {line_num}: failed to persist "
                        f"'{entry_dict.get('title', '<no title>')}'"
                    )

        # Update instance stats.
        self._import_stats["total_imports"] += 1
        self._import_stats["total_entries_imported"] += result.total_entries
        self._import_stats["total_errors"] += len(result.errors)
        self._import_stats["files_processed"] += 1
        self._import_stats["sources"].append(filepath)

        result.duration_seconds = time.monotonic() - start
        return result

    # ================================================================== #
    #  Code documentation import                                          #
    # ================================================================== #

    async def import_code_documentation(
        self,
        file_path: str,
        content: str,
    ) -> ImportResult:
        """Extract knowledge entries from code comments and docstrings.

        Detects and extracts the following:

        * **Python module docstrings** — the first triple-quoted string at
          the top of the file.
        * **Python class docstrings** — triple-quoted strings immediately
          following ``class`` definitions.
        * **Python function/method docstrings** — triple-quoted strings
          immediately following ``def`` / ``async def`` definitions.
        * **JSDoc comment blocks** — ``/** ... */`` comments (typically in
          JavaScript / TypeScript files).
        * **Inline knowledge annotations** — lines starting with
          ``# knowledge:`` (Python) or ``// knowledge:`` (JS/TS).

        Each extraction becomes a separate :class:`ImportedEntry` with the
        category ``"snippet"`` or ``"reference"`` depending on context.

        Parameters
        ----------
        file_path:
            Path to the source file (used for metadata and title generation).
        content:
            The full text content of the source file.

        Returns
        -------
        ImportResult
            Summary of all extracted entries.
        """
        start = time.monotonic()
        result = ImportResult(source=file_path)

        if not content or not content.strip():
            result.duration_seconds = time.monotonic() - start
            return result

        path = Path(file_path)
        suffix = path.suffix.lower()

        extracted_entries: List[ImportedEntry] = []

        # --- Python docstrings ---------------------------------------------
        if suffix == ".py":
            extracted_entries.extend(self._extract_python_docstrings(content, file_path))

        # --- JSDoc blocks --------------------------------------------------
        if suffix in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
            extracted_entries.extend(self._extract_jsdoc_blocks(content, file_path))

        # --- Inline knowledge annotations (both Python and JS) -------------
        extracted_entries.extend(self._extract_knowledge_annotations(content, file_path, suffix))

        # --- Persist all extracted entries ---------------------------------
        for imp in extracted_entries:
            entry_id = await self._persist_entry(imp)
            if entry_id:
                result.total_entries += 1
                cat = imp.category
                result.categories[cat] = result.categories.get(cat, 0) + 1
                for t in imp.tags:
                    result.tags[t] = result.tags.get(t, 0) + 1
            else:
                result.errors.append(
                    f"Failed to persist code doc entry: {imp.title[:80]}"
                )

        # Update instance stats.
        self._import_stats["total_imports"] += 1
        self._import_stats["total_entries_imported"] += result.total_entries
        self._import_stats["total_errors"] += len(result.errors)
        self._import_stats["files_processed"] += 1
        self._import_stats["sources"].append(file_path)

        result.duration_seconds = time.monotonic() - start
        return result

    # -- Python docstring extraction ---------------------------------------

    @staticmethod
    def _extract_python_docstrings(content: str, file_path: str) -> List[ImportedEntry]:
        """Extract module, class, and function docstrings from Python source.

        Parameters
        ----------
        content:
            Full Python source text.
        file_path:
            Source file path (for metadata).

        Returns
        -------
        list[ImportedEntry]
        """
        entries: List[ImportedEntry] = []
        stem = Path(file_path).stem

        # Module docstring (first triple-quote block).
        module_match = _RE_PY_MODULE_DOCSTRING.match(content.lstrip())
        if module_match:
            doc = module_match.group(1).strip()
            if len(doc) > 20:
                # Clean up the docstring (remove leading/trailing whitespace per line).
                lines = [line.strip() for line in doc.splitlines() if line.strip()]
                clean_doc = "\n".join(lines)
                entries.append(ImportedEntry(
                    title=f"Module: {stem}",
                    content=clean_doc,
                    category="reference",
                    tags=["python", "module", stem],
                    source=file_path,
                    confidence=0.9,
                    importance=0.7,
                    metadata={
                        "import_file": file_path,
                        "original_format": "python_docstring",
                        "docstring_type": "module",
                        "language": "python",
                    },
                ))

        # Class docstrings.
        for m in _RE_PY_CLASS_DOCSTRING.finditer(content):
            class_name = m.group(1)
            doc = m.group(2).strip() if m.group(2) else ""
            doc = doc.strip('\'"').strip()
            if len(doc) > 15:
                lines = [line.strip() for line in doc.splitlines() if line.strip()]
                clean_doc = "\n".join(lines)
                entries.append(ImportedEntry(
                    title=f"Class: {class_name}",
                    content=clean_doc,
                    category="reference",
                    tags=["python", "class", class_name, stem],
                    source=file_path,
                    confidence=0.85,
                    importance=0.6,
                    metadata={
                        "import_file": file_path,
                        "original_format": "python_docstring",
                        "docstring_type": "class",
                        "class_name": class_name,
                        "language": "python",
                    },
                ))

        # Function/method docstrings.
        for m in _RE_PY_FUNC_DOCSTRING.finditer(content):
            func_name = m.group(1)
            doc = m.group(2).strip() if m.group(2) else ""
            doc = doc.strip('\'"').strip()
            if len(doc) > 15:
                lines = [line.strip() for line in doc.splitlines() if line.strip()]
                clean_doc = "\n".join(lines)
                entries.append(ImportedEntry(
                    title=f"Function: {func_name}",
                    content=clean_doc,
                    category="snippet",
                    tags=["python", "function", func_name, stem],
                    source=file_path,
                    confidence=0.8,
                    importance=0.5,
                    metadata={
                        "import_file": file_path,
                        "original_format": "python_docstring",
                        "docstring_type": "function",
                        "function_name": func_name,
                        "language": "python",
                    },
                ))

        return entries

    # -- JSDoc extraction --------------------------------------------------

    @staticmethod
    def _extract_jsdoc_blocks(content: str, file_path: str) -> List[ImportedEntry]:
        """Extract JSDoc comment blocks from JavaScript/TypeScript source.

        Parameters
        ----------
        content:
            Full source text.
        file_path:
            Source file path (for metadata).

        Returns
        -------
        list[ImportedEntry]
        """
        entries: List[ImportedEntry] = []
        stem = Path(file_path).stem

        for m in _RE_JSDOC.finditer(content):
            doc = m.group(1).strip()
            if len(doc) < 15:
                continue

            # Clean up JSDoc markers (* at line starts).
            lines = []
            for line in doc.splitlines():
                cleaned = re.sub(r"^\s*\*\s?", "", line).strip()
                if cleaned:
                    lines.append(cleaned)

            if not lines:
                continue

            clean_doc = "\n".join(lines)

            # Try to extract a title from @name, @class, or the first line.
            title = stem
            name_match = re.search(r"@(?:name|class|function)\s+(\S+)", clean_doc)
            if name_match:
                title = f"JSDoc: {name_match.group(1)}"
            elif lines:
                # Use the first line (truncated) as a title.
                first_line = lines[0][:80]
                title = f"JSDoc: {first_line}"

            # Extract @tags as knowledge tags.
            jsdoc_tags = re.findall(r"@(\w+)", clean_doc)
            knowledge_tags: List[str] = ["javascript", "jsdoc", stem]
            for jt in jsdoc_tags:
                if jt.lower() not in ("param", "return", "returns", "see", "example",
                                       "throws", "deprecated", "since", "author",
                                       "version", "license", "default"):
                    knowledge_tags.append(jt.lower())

            # Detect if this describes a class or function.
            category = "reference"
            if "@class" in clean_doc or "@constructor" in clean_doc:
                category = "reference"
            elif "@function" in clean_doc or "@callback" in clean_doc:
                category = "snippet"

            entries.append(ImportedEntry(
                title=title,
                content=clean_doc,
                category=category,
                tags=knowledge_tags,
                source=file_path,
                confidence=0.8,
                importance=0.5,
                metadata={
                    "import_file": file_path,
                    "original_format": "jsdoc",
                    "language": "javascript",
                },
            ))

        return entries

    # -- Inline knowledge annotations --------------------------------------

    @staticmethod
    def _extract_knowledge_annotations(
        content: str,
        file_path: str,
        suffix: str,
    ) -> List[ImportedEntry]:
        """Extract ``# knowledge:`` / ``// knowledge:`` inline annotations.

        Each annotation line becomes a separate entry.

        Parameters
        ----------
        content:
            Full source text.
        file_path:
            Source file path.
        suffix:
            File extension (to pick the right comment syntax).

        Returns
        -------
        list[ImportedEntry]
        """
        entries: List[ImportedEntry] = []
        stem = Path(file_path).stem

        # Pick the right pattern based on file type.
        if suffix == ".py":
            pattern = _RE_KNOWLEDGE_ANNOTATION_PY
            lang_tag = "python"
        else:
            pattern = _RE_KNOWLEDGE_ANNOTATION_JS
            lang_tag = "javascript"

        for m in pattern.finditer(content):
            annotation_text = m.group(1).strip()
            if len(annotation_text) < 5:
                continue

            # Truncate title but keep full content.
            title_text = annotation_text[:100]
            if len(annotation_text) > 100:
                title_text = title_text + "..."

            entries.append(ImportedEntry(
                title=f"Knowledge: {title_text}",
                content=annotation_text,
                category="snippet",
                tags=[lang_tag, "knowledge-annotation", stem],
                source=file_path,
                confidence=0.85,
                importance=0.6,
                metadata={
                    "import_file": file_path,
                    "original_format": "inline_annotation",
                    "language": lang_tag,
                    "annotation_line": m.start(),
                },
            ))

        return entries

    # ================================================================== #
    #  Generic text file import                                           #
    # ================================================================== #

    async def import_text_file(
        self,
        filepath: str,
        category: str = "reference",
    ) -> ImportResult:
        """Import a generic plain-text file as a single knowledge entry.

        The entire file content becomes the entry body.  The filename
        (without extension) is used as the title.

        Parameters
        ----------
        filepath:
            Path to the text file.
        category:
            Category to assign the entry (default ``"reference"``).

        Returns
        -------
        ImportResult
            Summary of the import operation.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        """
        start = time.monotonic()
        result = ImportResult(source=filepath)

        path = Path(filepath).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {filepath}")

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result.errors.append(f"Failed to read {filepath}: {exc}")
            result.duration_seconds = time.monotonic() - start
            return result

        if not content.strip():
            result.errors.append(f"File is empty: {filepath}")
            result.duration_seconds = time.monotonic() - start
            return result

        title = path.stem

        validated_category = _validate_category(category)

        imported = ImportedEntry(
            title=title,
            content=content.strip(),
            category=validated_category,
            tags=[],
            source=str(path),
            confidence=self.default_confidence,
            importance=self.default_importance,
            metadata={
                "import_file": str(path),
                "import_filename": path.name,
                "original_format": "text",
                "file_size_bytes": path.stat().st_size if path.exists() else 0,
            },
        )

        entry_id = await self._persist_entry(imported)
        if entry_id:
            result.total_entries = 1
            result.categories[validated_category] = 1
            _logger.info("Imported text file entry %s: %s", entry_id, title)
        else:
            result.errors.append(f"Failed to persist text file entry: {filepath}")

        # Update instance stats.
        self._import_stats["total_imports"] += 1
        self._import_stats["total_entries_imported"] += result.total_entries
        self._import_stats["total_errors"] += len(result.errors)
        self._import_stats["files_processed"] += 1
        self._import_stats["sources"].append(filepath)

        result.duration_seconds = time.monotonic() - start
        return result

    # ================================================================== #
    #  Export                                                              #
    # ================================================================== #

    async def export_all(self, filepath: str) -> None:
        """Export the entire knowledge base to a JSON file.

        The output file contains a single JSON object with an ``"entries"``
        key holding an array of entry objects, each serialised with all
        fields from the corresponding :class:`KnowledgeEntry`.

        Parameters
        ----------
        filepath:
            Destination path for the exported JSON file.  Parent
            directories are created automatically if they do not exist.

        Raises
        ------
        RuntimeError
            If the store fails to retrieve entries.
        """
        from agent.knowledge_base.knowledge_store import VALID_CATEGORIES as VALID_CATS

        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        all_entries: List[Dict[str, Any]] = []

        for category in VALID_CATS:
            try:
                entries = await self.store.get_by_category(category, limit=99999)
                for entry in entries:
                    entry_dict = {
                        "id": entry.id,
                        "title": entry.title,
                        "content": entry.content,
                        "category": entry.category,
                        "tags": entry.tags,
                        "source": entry.source,
                        "confidence": entry.confidence,
                        "importance": entry.importance,
                        "references": entry.references,
                        "metadata": entry.metadata,
                        "created_at": entry.created_at,
                        "updated_at": entry.updated_at,
                        "access_count": entry.access_count,
                        "version": entry.version,
                    }
                    all_entries.append(entry_dict)
            except Exception as exc:
                _logger.error("Failed to export category %s: %s", category, exc)

        export_data = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(all_entries),
            "entries": all_entries,
        }

        try:
            path.write_text(
                json.dumps(export_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            _logger.info(
                "Exported %d entries to %s",
                len(all_entries),
                filepath,
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to write export file {filepath}: {exc}") from exc

    # ================================================================== #
    #  Statistics                                                          #
    # ================================================================== #

    def get_import_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics for all import operations performed
        by this importer instance.

        Keys:

        * ``total_imports`` — number of ``import_*`` calls made.
        * ``total_entries_imported`` — entries successfully persisted.
        * ``total_errors`` — cumulative error count.
        * ``files_processed`` — total files examined across all calls.
        * ``sources`` — list of source paths/labels from each import call.

        Returns
        -------
        dict[str, Any]
            A snapshot of the running statistics.
        """
        return {
            "total_imports": self._import_stats["total_imports"],
            "total_entries_imported": self._import_stats["total_entries_imported"],
            "total_errors": self._import_stats["total_errors"],
            "files_processed": self._import_stats["files_processed"],
            "sources": list(self._import_stats["sources"]),
        }
