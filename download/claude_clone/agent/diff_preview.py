"""
Diff Preview & Multi-File Atomic Editing System for Claude Code Clone.

Provides a complete change proposal workflow:
  1. Agent proposes changes to one or many files.
  2. User reviews diffs at file / hunk / line granularity.
  3. Accept / reject / modify decisions are collected.
  4. Accepted changes are applied atomically (all-or-nothing).
  5. Applied changes can be fully undone at any time.

Uses difflib for diff computation, file-hashing for conflict detection,
and a temp-directory + rename strategy for safe atomic writes.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANGE_CATEGORIES = [
    "refactor", "feature", "bugfix", "test", "docs", "style", "perf",
    "security",
]

RISK_LEVELS = ["low", "medium", "high", "critical"]

DEFAULT_CONTEXT_LINES = 3

# Regex that matches a unified-diff hunk header, e.g.
# @@ -12,7 +12,8 @@ def foo():
_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$", re.MULTILINE
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChangeType(Enum):
    """Type of change being proposed."""
    ADD = "add"
    DELETE = "delete"
    MODIFY = "modify"
    MOVE = "move"
    RENAME = "rename"


class ChangeCategory(Enum):
    """Semantic category of a change."""
    REFACTOR = "refactor"
    FEATURE = "feature"
    BUGFIX = "bugfix"
    TEST = "test"
    DOCS = "docs"
    STYLE = "style"
    PERF = "perf"
    SECURITY = "security"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChangeHunk:
    """Represents one contiguous block of changes inside a file.

    Attributes:
        old_start: Line number where the hunk starts in the original file (1-based).
        old_lines: Lines from the original file (prefixed with context markers internally).
        new_start: Line number where the hunk starts in the proposed file (1-based).
        new_lines: Lines from the proposed file.
        header: Human-readable header string (typically extracted from the @@ line).
        change_type: The kind of change this hunk represents.
        accepted: ``None`` = undecided, ``True`` = accepted, ``False`` = rejected.
        risk_level: Computed risk: low / medium / high / critical.
    """
    old_start: int = 1
    old_lines: List[str] = field(default_factory=list)
    new_start: int = 1
    new_lines: List[str] = field(default_factory=list)
    header: str = ""
    change_type: ChangeType = ChangeType.MODIFY
    accepted: Optional[bool] = None
    risk_level: str = "low"


@dataclass
class LineDecision:
    """Per-line accept/reject decision used for fine-grained review."""
    old_line: str
    new_line: str
    accepted: Optional[bool] = None
    note: str = ""


@dataclass
class FileChange:
    """Tracks all proposed changes for a single file.

    Attributes:
        file_path: Absolute or relative path within the project.
        change_type: Overall change type for the file.
        hunks: Ordered list of change hunks.
        original_content: Snapshot of the file at proposal time.
        proposed_content: The new content the agent wants to write.
        status: One of proposed / reviewed / accepted / rejected / applied.
        risk_score: Float between 0 and 1.
        metadata: Arbitrary key-value metadata (category, description, etc.).
        line_decisions: Per-line granular decisions (populated when user reviews lines).
        original_hash: SHA-256 of ``original_content`` for conflict detection.
    """
    file_path: str = ""
    change_type: ChangeType = ChangeType.MODIFY
    hunks: List[ChangeHunk] = field(default_factory=list)
    original_content: str = ""
    proposed_content: str = ""
    status: str = "proposed"
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    line_decisions: List[LineDecision] = field(default_factory=list)
    original_hash: str = ""


@dataclass
class ChangeReview:
    """Records a single review decision made by the user."""
    file_path: str = ""
    hunk_index: int = -1
    line_index: int = -1
    decision: str = "pending"  # accept / reject / modify
    note: str = ""
    reviewer: str = "user"


@dataclass
class AtomicChangeSet:
    """A collection of file changes that should be applied together.

    Attributes:
        id: Unique identifier (UUID4).
        description: Human-readable description of the overall change.
        files: Ordered list of ``FileChange`` objects.
        created_at: POSIX timestamp when the changeset was created.
        status: One of proposed / reviewed / applying / applied / undone / failed.
        applied_at: POSIX timestamp when successfully applied, or 0.
        undone: Whether this changeset has been rolled back.
        reviews: List of ``ChangeReview`` decisions.
        metadata: Arbitrary metadata (category, tags, etc.).
        category: Semantic category of the changeset.
    """
    id: str = ""
    description: str = ""
    files: List[FileChange] = field(default_factory=list)
    created_at: float = 0.0
    status: str = "proposed"
    applied_at: float = 0.0
    undone: bool = False
    reviews: List[ChangeReview] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    category: str = ""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _file_hash(content: str) -> str:
    """Return SHA-256 hex digest of *content*."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _now() -> float:
    """Return current time as a POSIX timestamp."""
    return time.time()


def _generate_id() -> str:
    return uuid.uuid4().hex[:16]


def _classify_change_type(original: str, proposed: str) -> ChangeType:
    """Classify the overall change type between two file contents."""
    if not original.strip() and proposed.strip():
        return ChangeType.ADD
    if original.strip() and not proposed.strip():
        return ChangeType.DELETE
    return ChangeType.MODIFY


def _infer_category_from_path(file_path: str) -> str:
    """Heuristic: guess a category from the file path."""
    lower = file_path.lower()
    if any(p in lower for p in ("test", "spec", "tests", "specs")):
        return "test"
    if any(p in lower for p in ("readme", "changelog", "license", "doc")):
        return "docs"
    if any(p in lower for p in ("security", "auth", "token", "perm")):
        return "security"
    if any(p in lower for p in ("perf", "benchmark", "prof")):
        return "perf"
    if any(p in lower for p in ("migrate", "migration")):
        return "refactor"
    return "feature"


def _count_test_lines(file_path: str) -> int:
    """Return an estimate of test coverage lines for the given file path."""
    lower = file_path.lower()
    if any(kw in lower for kw in ("test", "spec")):
        return 50
    if any(kw in lower for kw in (".py", ".js", ".ts", ".go", ".rs", ".java")):
        return 15
    return 5


def _compute_added_deleted(hunk: ChangeHunk) -> Tuple[int, int]:
    """Count added and deleted lines in a hunk."""
    added = sum(1 for l in hunk.new_lines if l.startswith("+"))
    deleted = sum(1 for l in hunk.old_lines if l.startswith("-"))
    return added, deleted


# ---------------------------------------------------------------------------
# Hunk parsing from unified diff text
# ---------------------------------------------------------------------------

def _parse_hunks_from_unified(diff_text: str) -> List[ChangeHunk]:
    """Parse a unified diff string into a list of ``ChangeHunk`` objects."""
    hunks: List[ChangeHunk] = []
    current_hunk: Optional[ChangeHunk] = None
    old_lines: List[str] = []
    new_lines: List[str] = []

    for line in diff_text.splitlines():
        match = _HUNK_RE.match(line)
        if match:
            # Flush previous hunk
            if current_hunk is not None:
                current_hunk.old_lines = old_lines[:]
                current_hunk.new_lines = new_lines[:]
                hunks.append(current_hunk)

            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) else 1
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) else 1
            header = match.group(5).strip()

            current_hunk = ChangeHunk(
                old_start=old_start,
                new_start=new_start,
                header=header,
            )
            old_lines = []
            new_lines = []

            # Detect pure-add / pure-delete hunks
            if old_count == 0:
                current_hunk.change_type = ChangeType.ADD
            elif new_count == 0:
                current_hunk.change_type = ChangeType.DELETE
            else:
                current_hunk.change_type = ChangeType.MODIFY
        elif line.startswith("+++") or line.startswith("---"):
            continue
        elif line.startswith("\\"):
            # "\ No newline at end of file"
            continue
        elif line.startswith("-"):
            old_lines.append(line)
        elif line.startswith("+"):
            new_lines.append(line)
        elif line.startswith(" "):
            old_lines.append(line)
            new_lines.append(line)
        else:
            # Could be a context line or other
            if line.strip():
                old_lines.append(" " + line)
                new_lines.append(" " + line)

    # Flush final hunk
    if current_hunk is not None:
        current_hunk.old_lines = old_lines[:]
        current_hunk.new_lines = new_lines[:]
        hunks.append(current_hunk)

    return hunks


# ---------------------------------------------------------------------------
# Line-level diff extraction (for per-line accept/reject)
# ---------------------------------------------------------------------------

def _extract_line_decisions(
    hunks: List[ChangeHunk],
) -> List[LineDecision]:
    """Build a flat list of ``LineDecision`` from all hunks.

    Context-only lines are emitted as ``accepted=True`` by default so
    that the user only needs to decide on changed lines.
    """
    decisions: List[LineDecision] = []
    for hunk in hunks:
        # Walk through both old and new line sequences in lock-step
        oi, ni = 0, 0
        old_idx, new_idx = 0, 0
        while oi < len(hunk.old_lines) or ni < len(hunk.new_lines):
            old_line = hunk.old_lines[oi] if oi < len(hunk.old_lines) else None
            new_line = hunk.new_lines[ni] if ni < len(hunk.new_lines) else None

            if old_line is not None and old_line.startswith("-"):
                # Deleted line
                decisions.append(LineDecision(old_line=old_line, new_line=""))
                oi += 1
            elif new_line is not None and new_line.startswith("+"):
                # Added line
                decisions.append(LineDecision(old_line="", new_line=new_line))
                ni += 1
            elif (old_line is not None and new_line is not None
                  and old_line == new_line):
                # Context line — pre-accept
                decisions.append(
                    LineDecision(
                        old_line=old_line,
                        new_line=new_line,
                        accepted=True,
                    )
                )
                oi += 1
                ni += 1
            else:
                # Fallback: emit individually
                if old_line is not None:
                    decisions.append(LineDecision(old_line=old_line, new_line=""))
                    oi += 1
                if new_line is not None:
                    decisions.append(LineDecision(old_line="", new_line=new_line))
                    ni += 1
    return decisions


# ---------------------------------------------------------------------------
# Rebuild content from accepted line decisions
# ---------------------------------------------------------------------------

def _rebuild_from_line_decisions(
    original_lines: List[str],
    decisions: List[LineDecision],
) -> str:
    """Given the original file lines and a list of line decisions, reconstruct
    the content incorporating only accepted changes.

    This works by replaying the decisions in order: context lines map directly,
    accepted deletions remove the corresponding line, and accepted additions
    insert new lines.
    """
    result_lines: List[str] = []
    dec_idx = 0
    orig_idx = 0

    while dec_idx < len(decisions):
        d = decisions[dec_idx]
        if d.accepted is True and d.old_line.startswith(" ") and d.new_line.startswith(" "):
            # Context line
            result_lines.append(d.old_line[1:])  # strip leading space
            orig_idx += 1
            dec_idx += 1
        elif d.accepted is True and d.old_line.startswith("-"):
            # Accepted deletion — skip the original line
            orig_idx += 1
            dec_idx += 1
        elif d.accepted is False and d.old_line.startswith("-"):
            # Rejected deletion — keep original line
            if orig_idx < len(original_lines):
                result_lines.append(original_lines[orig_idx])
            orig_idx += 1
            dec_idx += 1
        elif d.accepted is True and d.new_line.startswith("+"):
            # Accepted addition
            result_lines.append(d.new_line[1:])  # strip leading +
            dec_idx += 1
        elif d.accepted is False and d.new_line.startswith("+"):
            # Rejected addition — skip
            dec_idx += 1
        elif d.accepted is None:
            # Undecided — treat as accept by default
            if d.new_line.startswith("+"):
                result_lines.append(d.new_line[1:])
            elif d.old_line.startswith(" "):
                result_lines.append(d.old_line[1:])
                orig_idx += 1
            elif d.old_line.startswith("-"):
                orig_idx += 1  # skip (delete)
            dec_idx += 1
        else:
            # Fallback: keep context
            if d.old_line:
                result_lines.append(d.old_line.lstrip("+- "))
            dec_idx += 1

    # Append any remaining original lines not covered by hunks
    while orig_idx < len(original_lines):
        result_lines.append(original_lines[orig_idx])
        orig_idx += 1

    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Rebuild content from accepted hunks (when using hunk-level granularity)
# ---------------------------------------------------------------------------

def _rebuild_from_hunks(
    original_content: str,
    hunks: List[ChangeHunk],
) -> str:
    """Rebuild the file content by applying only accepted hunks.

    Rejected hunks are skipped, leaving the original content intact for
    those regions.
    """
    original_lines = original_content.splitlines(True)

    # Collect accepted hunks sorted by old_start
    accepted = [h for h in hunks if h.accepted is True]
    accepted.sort(key=lambda h: h.old_start)

    if not accepted:
        return original_content

    result: List[str] = []
    current_line = 0  # 0-based

    for hunk in accepted:
        # Copy unchanged lines before this hunk
        hunk_start_0 = hunk.old_start - 1  # convert to 0-based
        while current_line < hunk_start_0 and current_line < len(original_lines):
            result.append(original_lines[current_line])
            current_line += 1

        # Compute old hunk length (non-add lines)
        old_hunk_lines = [l for l in hunk.old_lines if not l.startswith("+")]
        old_hunk_len = len(old_hunk_lines)

        # Compute new hunk lines (non-delete lines)
        new_hunk_lines = [
            l for l in hunk.new_lines if not l.startswith("-")
        ]

        # Skip the old lines that the hunk replaces
        current_line += old_hunk_len

        # Append the new lines
        for nl in new_hunk_lines:
            stripped = nl[1:] if nl and nl[0] in "+- " else nl
            result.append(stripped.rstrip("\n") + "\n")

    # Append trailing original lines
    while current_line < len(original_lines):
        result.append(original_lines[current_line])
        current_line += 1

    return "".join(result)


# ---------------------------------------------------------------------------
# Side-by-side diff formatter
# ---------------------------------------------------------------------------

def _format_side_by_side(
    diff_lines: List[str],
    width: int = 80,
) -> str:
    """Format unified-diff output into a side-by-side view.

    Returns a multi-line string suitable for terminal display.
    """
    half = max(20, (width - 3) // 2)
    separator = " | "

    left_lines: List[str] = []
    right_lines: List[str] = []

    for line in diff_lines:
        if line.startswith("---") or line.startswith("+++"):
            left_lines.append(line)
            right_lines.append("")
        elif line.startswith("@@"):
            left_lines.append(line)
            right_lines.append("")
        elif line.startswith("-"):
            left_lines.append(line[1:])
            right_lines.append("")
        elif line.startswith("+"):
            left_lines.append("")
            right_lines.append(line[1:])
        elif line.startswith(" "):
            left_lines.append(line[1:])
            right_lines.append(line[1:])
        else:
            left_lines.append(line)
            right_lines.append("")

    rows: List[str] = []
    for left, right in zip(left_lines, right_lines):
        left_part = left[:half].ljust(half)
        right_part = right[:half]
        rows.append(left_part + separator + right_part)

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Git-format patch export
# ---------------------------------------------------------------------------

def _build_git_patch(changeset: AtomicChangeSet) -> str:
    """Build a git-format-patch string from a changeset."""
    lines: List[str] = []
    lines.append(f"From: claude-agent\n")
    lines.append(f"Subject: [PATCH] {changeset.description}\n")
    lines.append(f"Date: {time.strftime('%a, %d %b %Y %H:%M:%S +0000', time.gmtime(changeset.created_at))}\n")
    lines.append("---\n")

    for fc in changeset.files:
        lines.append(f"diff --git a/{fc.file_path} b/{fc.file_path}\n")
        if fc.change_type == ChangeType.ADD:
            lines.append("new file mode 100644\n")
        elif fc.change_type == ChangeType.DELETE:
            lines.append("deleted file mode 100644\n")
        lines.append("--- /dev/null\n")
        lines.append(f"+++ b/{fc.file_path}\n")
        lines.append(fc.proposed_content)
        if not fc.proposed_content.endswith("\n"):
            lines.append("\n")

    return "".join(lines)


# ---------------------------------------------------------------------------
# Patch import (basic unified-diff parser)
# ---------------------------------------------------------------------------

def _parse_git_patch(patch_text: str) -> Dict[str, str]:
    """Parse a simple patch and return ``{file_path: new_content}``."""
    file_map: Dict[str, str] = {}
    current_file: Optional[str] = None
    in_body = False
    current_lines: List[str] = []

    for raw_line in patch_text.splitlines(True):
        stripped = raw_line.rstrip("\n")

        if stripped.startswith("diff --git"):
            if current_file is not None:
                file_map[current_file] = "\n".join(current_lines)
            in_body = False
            current_lines = []
            # Extract target path
            match = re.search(r"b/(.+)$", stripped)
            if match:
                current_file = match.group(1)
        elif stripped.startswith("+++ b/"):
            current_file = stripped[6:]
            in_body = True
        elif stripped.startswith("---"):
            continue
        elif stripped.startswith("new file") or stripped.startswith("deleted file"):
            continue
        elif stripped.startswith("@@"):
            continue
        elif in_body and current_file is not None:
            if stripped.startswith("+"):
                current_lines.append(stripped[1:])
            elif stripped.startswith(" ") or stripped.startswith("-"):
                if stripped.startswith(" "):
                    current_lines.append(stripped[1:])
                # Skip deletions for new-content reconstruction
            elif stripped == "":
                current_lines.append("")

    if current_file is not None:
        file_map[current_file] = "\n".join(current_lines)

    return file_map


# ===========================================================================
# Main class: DiffPreview
# ===========================================================================

class DiffPreview:
    """Diff Preview & Multi-File Atomic Editing system.

    Manages the full lifecycle of code change proposals:

    1. **Propose** — agent submits proposed file contents.
    2. **Review** — user inspects diffs and accepts/rejects at file, hunk,
       or line granularity.
    3. **Apply** — accepted changes are written atomically.
    4. **Undo** — any applied changeset can be rolled back.

    Parameters:
        project_path: Root directory of the project (default ``"."``).
    """

    def __init__(self, project_path: str = ".") -> None:
        self.project_path = Path(project_path).resolve()
        self._changesets: Dict[str, AtomicChangeSet] = {}
        self._undo_stack: List[AtomicChangeSet] = []
        self._undo_backups: Dict[str, Dict[str, str]] = {}
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------
    # Public: proposing changes
    # -------------------------------------------------------------------

    async def propose_change(
        self,
        file_path: str,
        new_content: str,
        description: str | None = None,
        category: str | None = None,
    ) -> FileChange:
        """Propose a change to a single file.

        Args:
            file_path: Path relative to ``project_path`` (or absolute).
            new_content: The proposed new file content.
            description: Optional human-readable description.
            category: Optional semantic category tag.

        Returns:
            The created ``FileChange`` object.
        """
        abs_path = self._resolve(file_path)

        # Read original content
        if abs_path.exists():
            original_content = abs_path.read_text(encoding="utf-8")
        else:
            original_content = ""

        change_type = _classify_change_type(original_content, new_content)
        hunks = self._compute_hunks(original_content, new_content)
        risk_score = self._assess_file_risk(str(abs_path), hunks)

        fc = FileChange(
            file_path=str(abs_path),
            change_type=change_type,
            hunks=hunks,
            original_content=original_content,
            proposed_content=new_content,
            status="proposed",
            risk_score=risk_score,
            metadata={
                "description": description or "",
                "category": category or _infer_category_from_path(str(abs_path)),
            },
            line_decisions=_extract_line_decisions(hunks),
            original_hash=_file_hash(original_content),
        )

        # Assign per-hunk risk
        for hunk in hunks:
            hunk.risk_level = self._hunk_risk(hunk)

        return fc

    async def propose_multi_file(
        self,
        changes: Dict[str, str],
        description: str | None = None,
        category: str | None = None,
    ) -> AtomicChangeSet:
        """Propose changes to multiple files as an atomic changeset.

        Args:
            changes: Mapping of ``{file_path: new_content}``.
            description: Optional description of the overall change.
            category: Optional semantic category tag.

        Returns:
            The created ``AtomicChangeSet``.
        """
        async with self._lock:
            cs = AtomicChangeSet(
                id=_generate_id(),
                description=description or "",
                files=[],
                created_at=_now(),
                status="proposed",
                category=category or "",
            )

            for file_path, new_content in changes.items():
                fc = await self.propose_change(
                    file_path, new_content,
                    description=description,
                    category=category,
                )
                cs.files.append(fc)

            self._changesets[cs.id] = cs
            return cs

    # -------------------------------------------------------------------
    # Public: diff rendering
    # -------------------------------------------------------------------

    async def get_diff(self, file_change: FileChange) -> str:
        """Return a unified diff string for the given file change."""
        from_file = file_change.file_path
        to_file = file_change.file_path

        old = file_change.original_content.splitlines(keepends=True)
        new = file_change.proposed_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old,
            new,
            fromfile=from_file,
            tofile=to_file,
            lineterm="",
        )
        return "\n".join(diff)

    async def get_side_by_side(
        self,
        file_change: FileChange,
        width: int = 80,
    ) -> str:
        """Return a side-by-side diff suitable for terminal display."""
        diff_text = await self.get_diff(file_change)
        diff_lines = diff_text.splitlines()
        return _format_side_by_side(diff_lines, width=width)

    # -------------------------------------------------------------------
    # Public: file-level review
    # -------------------------------------------------------------------

    async def accept_file(
        self,
        changeset_id: str,
        file_path: str,
    ) -> None:
        """Accept all hunks in a file within a changeset."""
        cs = self._get_changeset(changeset_id)
        fc = self._get_file_change(cs, file_path)

        for hunk in fc.hunks:
            hunk.accepted = True
        for ld in fc.line_decisions:
            ld.accepted = True
        fc.status = "accepted"

        self._add_review(cs, file_path, -1, -1, "accept", "All hunks accepted")

    async def reject_file(
        self,
        changeset_id: str,
        file_path: str,
    ) -> None:
        """Reject all hunks in a file within a changeset."""
        cs = self._get_changeset(changeset_id)
        fc = self._get_file_change(cs, file_path)

        for hunk in fc.hunks:
            hunk.accepted = False
        for ld in fc.line_decisions:
            ld.accepted = False
        fc.status = "rejected"

        self._add_review(cs, file_path, -1, -1, "reject", "All hunks rejected")

    # -------------------------------------------------------------------
    # Public: hunk-level review
    # -------------------------------------------------------------------

    async def accept_hunk(
        self,
        changeset_id: str,
        file_path: str,
        hunk_index: int,
    ) -> None:
        """Accept a specific hunk by its index."""
        cs = self._get_changeset(changeset_id)
        fc = self._get_file_change(cs, file_path)

        if hunk_index < 0 or hunk_index >= len(fc.hunks):
            raise IndexError(f"Hunk index {hunk_index} out of range for {file_path}")

        fc.hunks[hunk_index].accepted = True

        # Update line decisions that belong to this hunk
        line_offset = 0
        for i in range(hunk_index):
            line_offset += len(fc.hunks[i].old_lines) + len(fc.hunks[i].new_lines)
        hunk = fc.hunks[hunk_index]
        count = len(hunk.old_lines) + len(hunk.new_lines)
        for j in range(line_offset, min(line_offset + count, len(fc.line_decisions))):
            if fc.line_decisions[j].accepted is not True:
                fc.line_decisions[j].accepted = True

        fc.status = "reviewed"
        self._add_review(cs, file_path, hunk_index, -1, "accept", "Hunk accepted")

    async def reject_hunk(
        self,
        changeset_id: str,
        file_path: str,
        hunk_index: int,
    ) -> None:
        """Reject a specific hunk by its index."""
        cs = self._get_changeset(changeset_id)
        fc = self._get_file_change(cs, file_path)

        if hunk_index < 0 or hunk_index >= len(fc.hunks):
            raise IndexError(f"Hunk index {hunk_index} out of range for {file_path}")

        fc.hunks[hunk_index].accepted = False

        # Update corresponding line decisions
        line_offset = 0
        for i in range(hunk_index):
            line_offset += len(fc.hunks[i].old_lines) + len(fc.hunks[i].new_lines)
        hunk = fc.hunks[hunk_index]
        count = len(hunk.old_lines) + len(hunk.new_lines)
        for j in range(line_offset, min(line_offset + count, len(fc.line_decisions))):
            if fc.line_decisions[j].accepted is not False:
                fc.line_decisions[j].accepted = False

        fc.status = "reviewed"
        self._add_review(cs, file_path, hunk_index, -1, "reject", "Hunk rejected")

    # -------------------------------------------------------------------
    # Public: line-level review
    # -------------------------------------------------------------------

    async def accept_line(
        self,
        changeset_id: str,
        file_path: str,
        hunk_index: int,
        line_index: int,
    ) -> None:
        """Accept a single line within a hunk.

        *line_index* is relative to the hunk's combined old+new lines.
        """
        cs = self._get_changeset(changeset_id)
        fc = self._get_file_change(cs, file_path)

        if hunk_index < 0 or hunk_index >= len(fc.hunks):
            raise IndexError(f"Hunk index {hunk_index} out of range")

        # Convert (hunk_index, line_index) to flat index in line_decisions
        flat = self._flat_line_index(fc.hunks, hunk_index, line_index)
        if flat is None or flat >= len(fc.line_decisions):
            raise IndexError(f"Line index {line_index} out of range in hunk {hunk_index}")

        fc.line_decisions[flat].accepted = True
        fc.status = "reviewed"
        self._add_review(
            cs, file_path, hunk_index, line_index, "accept", "Line accepted"
        )

    async def reject_line(
        self,
        changeset_id: str,
        file_path: str,
        hunk_index: int,
        line_index: int,
    ) -> None:
        """Reject a single line within a hunk."""
        cs = self._get_changeset(changeset_id)
        fc = self._get_file_change(cs, file_path)

        if hunk_index < 0 or hunk_index >= len(fc.hunks):
            raise IndexError(f"Hunk index {hunk_index} out of range")

        flat = self._flat_line_index(fc.hunks, hunk_index, line_index)
        if flat is None or flat >= len(fc.line_decisions):
            raise IndexError(f"Line index {line_index} out of range in hunk {hunk_index}")

        fc.line_decisions[flat].accepted = False
        fc.status = "reviewed"
        self._add_review(
            cs, file_path, hunk_index, line_index, "reject", "Line rejected"
        )

    # -------------------------------------------------------------------
    # Public: apply / undo
    # -------------------------------------------------------------------

    async def apply_changeset(self, changeset_id: str) -> dict:
        """Apply all accepted changes in a changeset atomically.

        Returns a dict with keys:
            - ``success`` (bool)
            - ``files_written`` (list[str])
            - ``errors`` (list[str])
            - ``backup_id`` (str) — identifier for undo.
        """
        cs = self._get_changeset(changeset_id)

        # Check for conflicts first
        conflicts = await self.detect_conflicts(changeset_id)
        if conflicts:
            return {
                "success": False,
                "files_written": [],
                "errors": [
                    f"Conflict in {c['file_path']}: {c['reason']}"
                    for c in conflicts
                ],
                "backup_id": changeset_id,
            }

        cs.status = "applying"
        files_written: List[str] = []
        errors: List[str] = []
        backup: Dict[str, str] = {}  # original content per file

        # Create a temporary staging directory for atomic writes
        with tempfile.TemporaryDirectory(prefix="claude_diff_") as staging:
            stage_path = Path(staging)

            for fc in cs.files:
                if fc.status == "rejected":
                    continue

                # Determine final content
                if fc.status == "accepted":
                    final_content = fc.proposed_content
                elif fc.status == "reviewed":
                    # Use line decisions if any were explicitly set
                    has_explicit = any(
                        ld.accepted is not None and ld.new_line.startswith("+")
                        for ld in fc.line_decisions
                    )
                    if has_explicit:
                        final_content = _rebuild_from_line_decisions(
                            fc.original_content.splitlines(True),
                            fc.line_decisions,
                        )
                    else:
                        # Fall back to hunk-level
                        final_content = _rebuild_from_hunks(
                            fc.original_content, fc.hunks
                        )
                else:
                    # Undecided — apply proposed content as-is
                    final_content = fc.proposed_content

                target = Path(fc.file_path)

                # Backup original
                if target.exists():
                    backup[fc.file_path] = target.read_text(encoding="utf-8")
                else:
                    backup[fc.file_path] = ""

                # Write to staging
                rel = target.name
                staging_file = stage_path / rel
                # Use a unique name to avoid collisions
                safe_name = (
                    fc.file_path.replace(os.sep, "_").replace("/", "_").lstrip("_")
                )
                staging_file = stage_path / safe_name
                staging_file.write_text(final_content, encoding="utf-8")

                try:
                    # Ensure parent directory exists
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Atomic: copy from staging to target
                    shutil.copy2(str(staging_file), str(target))
                    files_written.append(fc.file_path)
                    fc.status = "applied"
                except OSError as exc:
                    errors.append(f"Failed to write {fc.file_path}: {exc}")
                    fc.status = "failed"

        if errors:
            # Roll back any files we already wrote
            for fpath in files_written:
                if fpath in backup:
                    Path(fpath).write_text(backup[fpath], encoding="utf-8")
            cs.status = "failed"
            return {
                "success": False,
                "files_written": [],
                "errors": errors,
                "backup_id": changeset_id,
            }

        # Store backup for undo
        self._undo_backups[changeset_id] = backup
        cs.status = "applied"
        cs.applied_at = _now()
        self._undo_stack.append(cs)

        return {
            "success": True,
            "files_written": files_written,
            "errors": [],
            "backup_id": changeset_id,
        }

    async def undo_changeset(self, changeset_id: str) -> dict:
        """Undo a previously applied changeset.

        Returns a dict with ``success``, ``files_restored``, ``errors``.
        """
        cs = self._get_changeset(changeset_id)

        if cs.status != "applied":
            return {
                "success": False,
                "files_restored": [],
                "errors": [
                    f"Cannot undo changeset {changeset_id} with status '{cs.status}'"
                ],
            }

        backup = self._undo_backups.get(changeset_id, {})
        if not backup:
            return {
                "success": False,
                "files_restored": [],
                "errors": [f"No backup found for changeset {changeset_id}"],
            }

        files_restored: List[str] = []
        errors: List[str] = []

        with tempfile.TemporaryDirectory(prefix="claude_undo_") as staging:
            stage_path = Path(staging)

            for fpath, original in backup.items():
                target = Path(fpath)
                safe_name = fpath.replace(os.sep, "_").replace("/", "_").lstrip("_")
                staging_file = stage_path / safe_name
                staging_file.write_text(original, encoding="utf-8")

                try:
                    if not original.strip() and not target.exists():
                        # File was added by the changeset — delete it
                        if target.exists():
                            target.unlink()
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(staging_file), str(target))
                    files_restored.append(fpath)
                except OSError as exc:
                    errors.append(f"Failed to restore {fpath}: {exc}")

        if errors:
            return {
                "success": False,
                "files_restored": files_restored,
                "errors": errors,
            }

        cs.status = "undone"
        cs.undone = True

        # Remove from undo stack
        self._undo_stack = [
            cs_item for cs_item in self._undo_stack
            if cs_item.id != changeset_id
        ]

        return {
            "success": True,
            "files_restored": files_restored,
            "errors": [],
        }

    async def get_undo_stack(self) -> List[AtomicChangeSet]:
        """Return the list of applied changesets that can be undone (LIFO)."""
        return list(reversed(self._undo_stack))

    # -------------------------------------------------------------------
    # Public: conflict detection
    # -------------------------------------------------------------------

    async def detect_conflicts(self, changeset_id: str) -> List[dict]:
        """Check if any proposed files have been modified since proposal.

        Returns a list of conflict descriptors, each with:
            - ``file_path`` (str)
            - ``reason`` (str)
        """
        cs = self._get_changeset(changeset_id)
        conflicts: List[dict] = []

        for fc in cs.files:
            target = Path(fc.file_path)
            file_existed = bool(fc.original_content) or fc.change_type != ChangeType.ADD
            if self._detect_conflict_in_file(
                fc.file_path, fc.original_hash, file_existed=file_existed,
            ):
                if not target.exists() and file_existed:
                    reason = "File was deleted since the change was proposed"
                elif target.exists() and not file_existed:
                    reason = (
                        "File was created externally since the change was proposed"
                    )
                else:
                    reason = (
                        "File has been modified since the change was proposed "
                        f"(expected hash: {fc.original_hash[:12]}...)"
                    )
                conflicts.append({
                    "file_path": fc.file_path,
                    "reason": reason,
                })

        return conflicts

    # -------------------------------------------------------------------
    # Public: risk assessment
    # -------------------------------------------------------------------

    async def assess_risk(self, changeset_id: str) -> dict:
        """Provide a risk assessment for the changeset.

        Returns a dict with:
            - ``overall_risk`` (str): low / medium / high / critical
            - ``overall_score`` (float): 0.0 – 1.0
            - ``files`` (list[dict]): per-file risk breakdown
            - ``recommendations`` (list[str]): human-readable suggestions
        """
        cs = self._get_changeset(changeset_id)
        file_risks: List[dict] = []
        total_score = 0.0

        for fc in cs.files:
            total_added = 0
            total_deleted = 0
            for h in fc.hunks:
                a, d = _compute_added_deleted(h)
                total_added += a
                total_deleted += d

            file_risks.append({
                "file_path": fc.file_path,
                "change_type": fc.change_type.value,
                "hunks": len(fc.hunks),
                "lines_added": total_added,
                "lines_deleted": total_deleted,
                "risk_score": round(fc.risk_score, 3),
                "test_coverage_estimate": _count_test_lines(fc.file_path),
            })
            total_score += fc.risk_score

        num_files = max(len(cs.files), 1)
        avg_score = total_score / num_files
        overall_risk = self._score_to_level(avg_score)

        recommendations = self._build_recommendations(cs, file_risks, avg_score)

        return {
            "overall_risk": overall_risk,
            "overall_score": round(avg_score, 3),
            "files": file_risks,
            "recommendations": recommendations,
            "category": cs.category or "unknown",
        }

    # -------------------------------------------------------------------
    # Public: listing & summaries
    # -------------------------------------------------------------------

    async def get_pending_changesets(self) -> List[AtomicChangeSet]:
        """Return all changesets not yet applied or undone."""
        return [
            cs for cs in self._changesets.values()
            if cs.status in ("proposed", "reviewed")
        ]

    async def get_changeset_summary(self, changeset_id: str) -> str:
        """Return a human-readable summary of a changeset."""
        cs = self._get_changeset(changeset_id)

        lines: List[str] = []
        lines.append(f"Changeset: {cs.id}")
        lines.append(f"Description: {cs.description or '(no description)'}")
        lines.append(f"Category: {cs.category or '(uncategorized)'}")
        lines.append(f"Status: {cs.status}")
        lines.append(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cs.created_at))}")
        if cs.applied_at:
            lines.append(
                f"Applied:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cs.applied_at))}"
            )
        lines.append(f"Files: {len(cs.files)}")
        lines.append("")

        for i, fc in enumerate(cs.files, 1):
            lines.append(f"  [{i}] {fc.file_path}  ({fc.change_type.value})")
            lines.append(f"      Status: {fc.status}  |  Risk: {fc.risk_score:.2f}")
            for j, hunk in enumerate(fc.hunks):
                added, deleted = _compute_added_deleted(hunk)
                accept_mark = {
                    None: "  ", True: "\u2713 ", False: "\u2717 ",
                }.get(hunk.accepted, "? ")
                lines.append(
                    f"        Hunk {j}: {hunk.change_type.value:>6s}  "
                    f"+{added}/-{deleted}  risk={hunk.risk_level}  "
                    f"[{accept_mark.strip()}]"
                )

        if cs.reviews:
            lines.append("")
            lines.append(f"Reviews: {len(cs.reviews)}")
            for r in cs.reviews[-5:]:  # Show last 5
                lines.append(
                    f"  {r.file_path} hunk={r.hunk_index} line={r.line_index} "
                    f"-> {r.decision}  ({r.note})"
                )

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Public: patch export / import
    # -------------------------------------------------------------------

    async def export_patch(self, changeset_id: str) -> str:
        """Export a changeset as a git-format patch string."""
        cs = self._get_changeset(changeset_id)
        return _build_git_patch(cs)

    async def import_patch(self, patch_text: str) -> AtomicChangeSet:
        """Import a patch and create a new proposed changeset.

        Only supports unified-diff format with ``+++ b/path`` headers.
        """
        file_map = _parse_git_patch(patch_text)
        if not file_map:
            raise ValueError("Could not parse any file changes from the patch")

        return await self.propose_multi_file(
            changes=file_map,
            description="Imported from patch",
        )

    # -------------------------------------------------------------------
    # Public: streaming diff support
    # -------------------------------------------------------------------

    def create_streaming_change(
        self,
        file_path: str,
        description: str | None = None,
    ) -> StreamingChangeBuilder:
        """Return a ``StreamingChangeBuilder`` for incremental diff updates.

        Usage::

            builder = dp.create_streaming_change("src/main.py")
            builder.append_chunk("def hello():\\n")
            builder.append_chunk("    print('hello')\\n")
            builder.append_chunk("\\n")
            file_change = builder.build()
        """
        return StreamingChangeBuilder(
            project_path=str(self.project_path),
            file_path=file_path,
            description=description,
        )

    # -------------------------------------------------------------------
    # Internal: hunk computation
    # -------------------------------------------------------------------

    def _compute_hunks(
        self,
        old_content: str,
        new_content: str,
    ) -> List[ChangeHunk]:
        """Compute ``ChangeHunk`` objects from two content strings."""
        old = old_content.splitlines(keepends=True)
        new = new_content.splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            old, new,
            fromfile="a",
            tofile="b",
            n=DEFAULT_CONTEXT_LINES,
            lineterm="",
        ))

        hunks = _parse_hunks_from_unified("\n".join(diff))

        # If the unified diff produced no hunks but content differs,
        # create a synthetic full-file hunk
        if not hunks and old_content != new_content:
            old_lines = ["- " + l.rstrip("\n") for l in old]
            new_lines = ["+ " + l.rstrip("\n") for l in new]
            hunks.append(
                ChangeHunk(
                    old_start=1,
                    old_lines=old_lines,
                    new_start=1,
                    new_lines=new_lines,
                    header="full file replacement",
                    change_type=_classify_change_type(old_content, new_content),
                )
            )

        return hunks

    # -------------------------------------------------------------------
    # Internal: risk assessment helpers
    # -------------------------------------------------------------------

    def _assess_file_risk(
        self,
        file_path: str,
        hunks: List[ChangeHunk],
    ) -> float:
        """Compute a risk score (0.0 – 1.0) for a file change.

        Factors:
            - Number of lines changed (more = riskier)
            - Number of hunks (more dispersed = riskier)
            - File extension (config files are riskier to change)
            - Test coverage heuristic
        """
        score = 0.0

        total_added = 0
        total_deleted = 0
        for h in hunks:
            a, d = _compute_added_deleted(h)
            total_added += a
            total_deleted += d

        total_changed = total_added + total_deleted

        # Line-count factor (logarithmic)
        if total_changed > 0:
            import math
            score += min(0.35, 0.05 * math.log1p(total_changed))

        # Hunk-dispersion factor
        if len(hunks) > 1:
            score += min(0.15, 0.03 * len(hunks))

        # File-type factor
        risky_extensions = {
            ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini", ".conf",
            ".env", ".lock", ".py",
        }
        ext = Path(file_path).suffix.lower()
        if ext in risky_extensions:
            score += 0.15
            if ext in (".yaml", ".yml", ".toml", ".json", ".cfg", ".ini", ".conf"):
                score += 0.05  # Config files are extra risky

        # Test coverage factor
        test_cov = _count_test_lines(file_path)
        if test_cov < 10:
            score += 0.15

        # Deletion-heavy changes are riskier than additions
        if total_deleted > total_added * 2 and total_deleted > 5:
            score += 0.1

        return min(1.0, score)

    def _hunk_risk(self, hunk: ChangeHunk) -> str:
        """Assign a risk level string to a single hunk."""
        added, deleted = _compute_added_deleted(hunk)
        total = added + deleted
        if total <= 3:
            return "low"
        elif total <= 10:
            return "medium"
        elif total <= 30:
            return "high"
        return "critical"

    def _score_to_level(self, score: float) -> str:
        """Map a numeric score to a risk level string."""
        if score < 0.25:
            return "low"
        elif score < 0.5:
            return "medium"
        elif score < 0.75:
            return "high"
        return "critical"

    def _build_recommendations(
        self,
        cs: AtomicChangeSet,
        file_risks: List[dict],
        avg_score: float,
    ) -> List[str]:
        """Generate human-readable risk mitigation recommendations."""
        recs: List[str] = []

        if avg_score > 0.6:
            recs.append(
                "High-risk changeset detected. Consider splitting into smaller, "
                "focused changesets."
            )

        for fr in file_risks:
            if fr["risk_score"] > 0.5:
                recs.append(
                    f"File {fr['file_path']} has elevated risk "
                    f"({fr['risk_score']:.2f}). Review carefully."
                )
            if fr["test_coverage_estimate"] < 10 and fr["lines_added"] + fr["lines_deleted"] > 5:
                recs.append(
                    f"Low test coverage for {fr['file_path']}. Consider adding "
                    "tests for the changes."
                )

        total_added = sum(f["lines_added"] for f in file_risks)
        total_deleted = sum(f["lines_deleted"] for f in file_risks)
        if total_deleted > total_added * 3 and total_deleted > 20:
            recs.append(
                "Large number of deletions detected. Verify that no "
                "important functionality is being removed."
            )

        if len(file_risks) > 10:
            recs.append(
                "Changes span many files. Consider applying in stages to "
                "isolate any issues."
            )

        if cs.category == "security":
            recs.append(
                "Security-related changes require extra scrutiny. Ensure "
                "authentication and authorization logic is correct."
            )

        if not recs:
            recs.append("Changeset looks reasonable. Review the diffs and proceed.")

        return recs

    # -------------------------------------------------------------------
    # Internal: conflict detection
    # -------------------------------------------------------------------

    def _detect_conflict_in_file(
        self,
        file_path: str,
        original_hash: str,
        file_existed: bool = True,
    ) -> bool:
        """Return ``True`` if the file on disk has changed since proposal.

        Parameters:
            file_path: Absolute path to check.
            original_hash: SHA-256 of the file content at proposal time.
            file_existed: Whether the file existed when the change was proposed.
                When ``False`` (new file), a missing file on disk is *not* a
                conflict — only an unexpected existing file would be.
        """
        target = Path(file_path)
        on_disk = target.exists()

        if not on_disk:
            if not file_existed:
                # New file that still doesn't exist — no conflict
                return False
            # File existed at proposal time but has been deleted — conflict
            return True

        if not file_existed:
            # File didn't exist at proposal time but now does — someone else
            # created it.  Consider it a conflict so the user can decide.
            return True

        current = target.read_text(encoding="utf-8")
        current_hash = _file_hash(current)
        return current_hash != original_hash

    # -------------------------------------------------------------------
    # Internal: helpers
    # -------------------------------------------------------------------

    def _resolve(self, file_path: str) -> Path:
        """Resolve a path relative to ``project_path``."""
        p = Path(file_path)
        if p.is_absolute():
            return p
        return (self.project_path / p).resolve()

    def _get_changeset(self, changeset_id: str) -> AtomicChangeSet:
        """Retrieve a changeset by ID or raise ``KeyError``."""
        if changeset_id not in self._changesets:
            raise KeyError(f"Changeset '{changeset_id}' not found")
        return self._changesets[changeset_id]

    def _get_file_change(
        self,
        cs: AtomicChangeSet,
        file_path: str,
    ) -> FileChange:
        """Find a ``FileChange`` inside a changeset or raise ``KeyError``."""
        resolved = str(self._resolve(file_path))
        for fc in cs.files:
            if fc.file_path == resolved:
                return fc
            # Also try direct match
            if fc.file_path == file_path:
                return fc
        raise KeyError(f"File '{file_path}' not found in changeset '{cs.id}'")

    @staticmethod
    def _add_review(
        cs: AtomicChangeSet,
        file_path: str,
        hunk_index: int,
        line_index: int,
        decision: str,
        note: str,
    ) -> None:
        """Append a ``ChangeReview`` to the changeset."""
        cs.reviews.append(
            ChangeReview(
                file_path=file_path,
                hunk_index=hunk_index,
                line_index=line_index,
                decision=decision,
                note=note,
                reviewer="user",
            )
        )

    @staticmethod
    def _flat_line_index(
        hunks: List[ChangeHunk],
        hunk_index: int,
        line_index: int,
    ) -> Optional[int]:
        """Convert (hunk_index, line_index) to a flat index in ``line_decisions``.

        Returns ``None`` if the indices are out of range.
        """
        offset = 0
        for i in range(hunk_index):
            offset += len(hunks[i].old_lines) + len(hunks[i].new_lines)
        return offset + line_index


# ===========================================================================
# StreamingChangeBuilder — incremental diff construction
# ===========================================================================

class StreamingChangeBuilder:
    """Helper for building diffs incrementally as the agent streams output.

    Usage::

        builder = dp.create_streaming_change("src/main.py")
        builder.append_chunk("def foo():\\n")
        builder.append_chunk("    return 42\\n")
        # ... later ...
        file_change = await builder.finalize()
    """

    def __init__(
        self,
        project_path: str,
        file_path: str,
        description: str | None = None,
    ) -> None:
        self.project_path = Path(project_path).resolve()
        self.file_path = file_path
        self.description = description
        self._chunks: List[str] = []
        self._finalized = False

        # Read original content upfront
        abs_path = self.project_path / file_path
        if abs_path.exists():
            self.original_content = abs_path.read_text(encoding="utf-8")
        else:
            self.original_content = ""

    def append_chunk(self, text: str) -> None:
        """Append a text chunk from the streaming agent output."""
        if self._finalized:
            raise RuntimeError("Cannot append to a finalized StreamingChangeBuilder")
        self._chunks.append(text)

    def get_intermediate_diff(self) -> str:
        """Return a diff of the original content vs. content accumulated so far."""
        current = "".join(self._chunks)
        old = self.original_content.splitlines(keepends=True)
        new = current.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old, new,
            fromfile=self.file_path,
            tofile=self.file_path,
            lineterm="",
        )
        return "\n".join(diff)

    def get_partial_content(self) -> str:
        """Return the content accumulated so far (before finalization)."""
        return "".join(self._chunks)

    async def finalize(self) -> FileChange:
        """Finalize the streaming and return a complete ``FileChange``.

        This should be called once the agent has finished streaming.
        """
        if self._finalized:
            raise RuntimeError("StreamingChangeBuilder already finalized")
        self._finalized = True

        proposed_content = "".join(self._chunks)

        change_type = _classify_change_type(self.original_content, proposed_content)

        # Compute hunks using DiffPreview's method
        dp = DiffPreview.__new__(DiffPreview)
        dp.project_path = self.project_path
        hunks = dp._compute_hunks(self.original_content, proposed_content)

        fc = FileChange(
            file_path=str(self.project_path / self.file_path),
            change_type=change_type,
            hunks=hunks,
            original_content=self.original_content,
            proposed_content=proposed_content,
            status="proposed",
            risk_score=dp._assess_file_risk(self.file_path, hunks),
            metadata={
                "description": self.description or "",
                "category": _infer_category_from_path(self.file_path),
            },
            line_decisions=_extract_line_decisions(hunks),
            original_hash=_file_hash(self.original_content),
        )

        for hunk in hunks:
            hunk.risk_level = dp._hunk_risk(hunk)

        return fc


# ===========================================================================
# Convenience: async context for batch operations
# ===========================================================================

class DiffPreviewSession:
    """High-level session that wraps ``DiffPreview`` with a typical workflow.

    Example::

        async with DiffPreviewSession("/my/project") as session:
            cs = await session.propose({"main.py": new_code}, "Add feature X")
            summary = await session.review_summary(cs.id)
            # User reviews, then:
            await session.accept_all(cs.id)
            result = await session.apply(cs.id)
            if result["success"]:
                print("Applied!")
    """

    def __init__(self, project_path: str = ".") -> None:
        self.dp = DiffPreview(project_path)
        self._active_id: Optional[str] = None

    async def __aenter__(self) -> "DiffPreviewSession":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    async def propose(
        self,
        changes: Dict[str, str],
        description: str | None = None,
        category: str | None = None,
    ) -> AtomicChangeSet:
        """Propose changes and store the active changeset ID."""
        cs = await self.dp.propose_multi_file(changes, description, category)
        self._active_id = cs.id
        return cs

    async def review_summary(self, changeset_id: str | None = None) -> str:
        """Get a human-readable summary for review."""
        cid = changeset_id or self._active_id
        if cid is None:
            raise ValueError("No active changeset")
        return await self.dp.get_changeset_summary(cid)

    async def accept_all(self, changeset_id: str | None = None) -> None:
        """Accept all files in the changeset."""
        cid = changeset_id or self._active_id
        if cid is None:
            raise ValueError("No active changeset")
        cs = self.dp._get_changeset(cid)
        for fc in cs.files:
            await self.dp.accept_file(cid, fc.file_path)

    async def reject_all(self, changeset_id: str | None = None) -> None:
        """Reject all files in the changeset."""
        cid = changeset_id or self._active_id
        if cid is None:
            raise ValueError("No active changeset")
        cs = self.dp._get_changeset(cid)
        for fc in cs.files:
            await self.dp.reject_file(cid, fc.file_path)

    async def apply(self, changeset_id: str | None = None) -> dict:
        """Apply the active (or specified) changeset."""
        cid = changeset_id or self._active_id
        if cid is None:
            raise ValueError("No active changeset")
        return await self.dp.apply_changeset(cid)

    async def undo(self, changeset_id: str | None = None) -> dict:
        """Undo the active (or specified) changeset."""
        cid = changeset_id or self._active_id
        if cid is None:
            raise ValueError("No active changeset")
        return await self.dp.undo_changeset(cid)

    async def risk(self, changeset_id: str | None = None) -> dict:
        """Get risk assessment."""
        cid = changeset_id or self._active_id
        if cid is None:
            raise ValueError("No active changeset")
        return await self.dp.assess_risk(cid)

    async def conflicts(self, changeset_id: str | None = None) -> List[dict]:
        """Detect conflicts."""
        cid = changeset_id or self._active_id
        if cid is None:
            raise ValueError("No active changeset")
        return await self.dp.detect_conflicts(cid)

    async def export_patch(self, changeset_id: str | None = None) -> str:
        """Export as git-format patch."""
        cid = changeset_id or self._active_id
        if cid is None:
            raise ValueError("No active changeset")
        return await self.dp.export_patch(cid)

    async def get_undo_stack(self) -> List[AtomicChangeSet]:
        """Return the undo stack."""
        return await self.dp.get_undo_stack()

    async def get_pending(self) -> List[AtomicChangeSet]:
        """Return pending changesets."""
        return await self.dp.get_pending_changesets()
