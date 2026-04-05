"""
Advanced Diff Engine with 3-way merge support.

Provides unified diff generation, patch application/reversal, word-level
intra-line diffing, side-by-side formatting, and conflict-aware 3-way merging.
Built on top of Python's difflib with significant enhancements.
"""

from __future__ import annotations

import re
import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class DiffLineType(Enum):
    """Type of a line inside a diff hunk."""
    CONTEXT = "context"
    ADDED = "added"
    REMOVED = "removed"


@dataclass
class DiffLine:
    """A single line within a diff hunk."""
    type: DiffLineType
    content: str
    old_line_no: int = -1
    new_line_no: int = -1


@dataclass
class Hunk:
    """A contiguous group of changed lines in a diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str = ""
    lines: list[DiffLine] = field(default_factory=list)

    @property
    def is_pure_addition(self) -> bool:
        return self.old_count == 0 and self.new_count > 0

    @property
    def is_pure_deletion(self) -> bool:
        return self.new_count == 0 and self.old_count > 0


@dataclass
class DiffResult:
    """Result of comparing two texts."""
    hunks: list[Hunk] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    changes: list[Hunk] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return self.additions + self.deletions


@dataclass
class MergeConflict:
    """A single conflict detected during a 3-way merge."""
    base_section: str
    ours_section: str
    theirs_section: str
    start_line: int = 0

    @property
    def marker(self) -> str:
        ours = self.ours_section.strip()
        theirs = self.theirs_section.strip()
        return (
            f"<<<<<<< OURS\n{ours}\n=======\n{theirs}\n>>>>>>> THEIRS"
        )


@dataclass
class MergeResult:
    """Result of a 3-way merge operation."""
    merged_text: str = ""
    conflicts: list[MergeConflict] = field(default_factory=list)
    success: bool = True
    auto_resolved: int = 0


@dataclass
class WordChange:
    """Represents a word-level change within a single line."""
    type: DiffLineType
    value: str
    position: int = 0
    old_position: int = -1


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

class _Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    BOLD = "\033[1m"
    GREY = "\033[2m"


# ---------------------------------------------------------------------------
# DiffEngine
# ---------------------------------------------------------------------------

class DiffEngine:
    """Advanced diff engine with unified-diff, patch, and merge support."""

    def __init__(self) -> None:
        self._matcher_options: dict = {}

    # ------------------------------------------------------------------
    # Core diff
    # ------------------------------------------------------------------

    def diff(
        self,
        text_a: str,
        text_b: str,
        context_lines: int = 3,
    ) -> DiffResult:
        """Produce a unified diff between *text_a* and *text_b*."""
        lines_a = text_a.splitlines(keepends=True)
        lines_b = text_b.splitlines(keepends=True)

        raw = list(
            difflib.unified_diff(
                lines_a,
                lines_b,
                fromfile="a",
                tofile="b",
                n=context_lines,
            )
        )

        result = DiffResult()
        if not raw:
            return result

        result.hunks = self.get_hunks_from_raw(raw)
        result.changes = [h for h in result.hunks if not (
            all(ln.type == DiffLineType.CONTEXT for ln in h.lines)
        )]
        result.additions = sum(
            1 for h in result.hunks for ln in h.lines if ln.type == DiffLineType.ADDED
        )
        result.deletions = sum(
            1 for h in result.hunks for ln in h.lines if ln.type == DiffLineType.REMOVED
        )
        return result

    def diff_files(self, file_a: str, file_b: str) -> DiffResult:
        """Diff two files on disk."""
        with open(file_a, "r", encoding="utf-8", errors="replace") as fh:
            text_a = fh.read()
        with open(file_b, "r", encoding="utf-8", errors="replace") as fh:
            text_b = fh.read()

        result = self.diff(text_a, text_b)
        result.files = [file_a, file_b]
        return result

    # ------------------------------------------------------------------
    # 3-way merge
    # ------------------------------------------------------------------

    def three_way_merge(
        self,
        base: str,
        ours: str,
        theirs: str,
    ) -> MergeResult:
        """Perform a 3-way merge.  Returns *MergeResult* with conflict info."""
        base_lines = base.splitlines(True)
        ours_lines = ours.splitlines(True)
        theirs_lines = theirs.splitlines(True)

        matcher_ours = difflib.SequenceMatcher(None, base_lines, ours_lines)
        matcher_theirs = difflib.SequenceMatcher(None, base_lines, theirs_lines)

        ops_ours = matcher_ours.get_opcodes()
        ops_theirs = matcher_theirs.get_opcodes()

        # Build change maps keyed by base-line ranges
        ours_changes = {
            (i1, i2): (tag, ours_lines[j1:j2])
            for tag, i1, i2, j1, j2 in ops_ours
            if tag != "equal"
        }
        theirs_changes = {
            (i1, i2): (tag, theirs_lines[j1:j2])
            for tag, i1, i2, j1, j2 in ops_theirs
            if tag != "equal"
        }

        merged: list[str] = []
        conflicts: list[MergeConflict] = []
        auto_resolved = 0
        pos = 0

        while pos < len(base_lines):
            # Check for overlap between ours and theirs
            ours_hit = [
                (rng, v) for rng, v in ours_changes.items()
                if rng[0] <= pos < rng[1] or rng[0] == pos
            ]
            theirs_hit = [
                (rng, v) for rng, v in theirs_changes.items()
                if rng[0] <= pos < rng[1] or rng[0] == pos
            ]

            if not ours_hit and not theirs_hit:
                merged.append(base_lines[pos])
                pos += 1
                continue

            # Determine which ranges to consume
            ours_range, ours_val = ours_hit[0] if ours_hit else (None, None)
            theirs_range, theirs_val = theirs_hit[0] if theirs_hit else (None, None)

            if ours_range and theirs_range and ours_range == theirs_range:
                # Same base region changed by both sides
                ours_section = "".join(ours_val[1])
                theirs_section = "".join(theirs_val[1])
                if ours_section == theirs_section:
                    merged.append(ours_section)
                    auto_resolved += 1
                else:
                    base_section = "".join(
                        base_lines[ours_range[0]:ours_range[1]]
                    )
                    conflicts.append(MergeConflict(
                        base_section=base_section,
                        ours_section=ours_section,
                        theirs_section=theirs_section,
                        start_line=len(merged) + 1,
                    ))
                    merged.append(
                        f"<<<<<<< OURS\n{ours_section}"
                        f"=======\n{theirs_section}>>>>>>> THEIRS\n"
                    )
                pos = max(ours_range[1], theirs_range[1])
            elif ours_range:
                tag, new_lines = ours_val
                if tag == "insert":
                    merged.extend(new_lines)
                elif tag == "replace":
                    merged.extend(new_lines)
                elif tag == "delete":
                    pass  # deletions remove lines
                auto_resolved += 1
                pos = ours_range[1]
            elif theirs_range:
                tag, new_lines = theirs_val
                if tag == "insert":
                    merged.extend(new_lines)
                elif tag == "replace":
                    merged.extend(new_lines)
                elif tag == "delete":
                    pass
                auto_resolved += 1
                pos = theirs_range[1]
            else:
                merged.append(base_lines[pos])
                pos += 1

        # Handle trailing insertions
        for rng, (tag, new_lines) in ours_changes.items():
            if rng[0] >= len(base_lines) and tag == "insert":
                merged.extend(new_lines)
                auto_resolved += 1
        for rng, (tag, new_lines) in theirs_changes.items():
            if rng[0] >= len(base_lines) and tag == "insert":
                ours_overlap = [
                    v for r, v in ours_changes.items() if r == rng
                ]
                if not ours_overlap:
                    merged.extend(new_lines)
                    auto_resolved += 1

        return MergeResult(
            merged_text="".join(merged),
            conflicts=conflicts,
            success=len(conflicts) == 0,
            auto_resolved=auto_resolved,
        )

    # ------------------------------------------------------------------
    # Patch generation / application / reversal
    # ------------------------------------------------------------------

    def generate_patch(
        self,
        original: str,
        modified: str,
        filename: str = "file",
    ) -> str:
        """Return a unified-diff patch string."""
        lines_a = original.splitlines(keepends=True)
        lines_b = modified.splitlines(keepends=True)

        diff_lines = list(
            difflib.unified_diff(
                lines_a,
                lines_b,
                fromfile=f"{filename}\t",
                tofile=f"{filename}\t",
                n=3,
            )
        )
        return "".join(diff_lines)

    def apply_patch(self, original: str, patch: str) -> str:
        """Apply a unified diff *patch* to *original* text and return result."""
        hunks = self._parse_hunks(patch)
        lines = original.splitlines(True)

        # Sort hunks in reverse so offsets stay valid
        hunks.sort(key=lambda h: h.old_start, reverse=True)

        for hunk in hunks:
            self._apply_hunk(lines, hunk)

        return "".join(lines)

    def reverse_patch(self, patched: str, patch: str) -> str:
        """Reverse a unified diff patch — restore the original text."""
        hunks = self._parse_hunks(patch)
        lines = patched.splitlines(True)

        hunks.sort(key=lambda h: h.new_start, reverse=True)

        for hunk in hunks:
            rev = Hunk(
                old_start=hunk.new_start,
                old_count=hunk.new_count,
                new_start=hunk.old_start,
                new_count=hunk.old_count,
                header=hunk.header,
            )
            for ln in hunk.lines:
                if ln.type == DiffLineType.ADDED:
                    rev.lines.append(DiffLine(
                        type=DiffLineType.REMOVED,
                        content=ln.content,
                        old_line_no=ln.new_line_no,
                        new_line_no=ln.old_line_no,
                    ))
                elif ln.type == DiffLineType.REMOVED:
                    rev.lines.append(DiffLine(
                        type=DiffLineType.ADDED,
                        content=ln.content,
                        old_line_no=ln.new_line_no,
                        new_line_no=ln.old_line_no,
                    ))
                else:
                    rev.lines.append(ln)
            self._apply_hunk(lines, rev)

        return "".join(lines)

    # ------------------------------------------------------------------
    # Hunk / diff parsing
    # ------------------------------------------------------------------

    def get_hunks(self, diff_text: str) -> list[Hunk]:
        """Parse a unified diff string into a list of *Hunk* objects."""
        return self.get_hunks_from_raw(diff_text.splitlines(True))

    def is_binary_diff(self, diff_text: str) -> bool:
        """Return *True* if the diff text indicates binary files."""
        binary_markers = (
            "Binary files",
            "Files a/ and b/ differ",
            "differ\n",
        )
        return any(m in diff_text for m in binary_markers)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_side_by_side(
        self,
        diff_result: DiffResult,
        width: int = 80,
    ) -> str:
        """Render a side-by-side diff view with a gutter in the middle."""
        half = max(2, (width - 3) // 2)
        sep = " | "
        out: list[str] = []

        header = (
            f"{'— Old —':^{half}}{sep}{'— New —':^{width - half - len(sep)}}"
        )
        out.append(header)
        out.append("—" * width)

        for hunk in diff_result.hunks:
            if hunk.header:
                out.append(f"{hunk.header:^{width}}")
                continue

            left_buf: list[str] = []
            right_buf: list[str] = []
            for ln in hunk.lines:
                text = ln.content.rstrip("\r\n")
                lno = str(ln.old_line_no) if ln.old_line_no >= 0 else ""
                rno = str(ln.new_line_no) if ln.new_line_no >= 0 else ""

                if ln.type == DiffLineType.CONTEXT:
                    left_buf.append(f"{lno:>4} {text}")
                    right_buf.append(f"{rno:>4} {text}")
                elif ln.type == DiffLineType.REMOVED:
                    left_buf.append(f"{lno:>4}- {text}")
                    right_buf.append("")
                elif ln.type == DiffLineType.ADDED:
                    left_buf.append("")
                    right_buf.append(f"{rno:>4}+ {text}")

            # Pad shorter side
            max_len = max(len(left_buf), len(right_buf))
            left_buf.extend([""] * (max_len - len(left_buf)))
            right_buf.extend([""] * (max_len - len(right_buf)))

            for l, r in zip(left_buf, right_buf):
                lp = l.ljust(half)
                rp = r.ljust(width - half - len(sep))
                out.append(lp + sep + rp)

        return "\n".join(out)

    def format_colorized(self, diff_result: DiffResult) -> str:
        """Return terminal-colorized unified diff output."""
        C = _Colors
        out: list[str] = []

        for hunk in diff_result.hunks:
            if hunk.header:
                out.append(f"{C.BOLD}{C.CYAN}{hunk.header.rstrip()}{C.RESET}")
                continue
            for ln in hunk.lines:
                text = ln.content.rstrip("\r\n")
                lno = str(ln.old_line_no) if ln.old_line_no >= 0 else ""
                rno = str(ln.new_line_no) if ln.new_line_no >= 0 else ""
                prefix = " "
                if ln.type == DiffLineType.REMOVED:
                    prefix = "-"
                    out.append(
                        f"{C.GREY}{lno:>4} {C.RED}{prefix} {text}{C.RESET}"
                    )
                elif ln.type == DiffLineType.ADDED:
                    prefix = "+"
                    out.append(
                        f"{C.GREY}{rno:>4} {C.GREEN}{prefix} {text}{C.RESET}"
                    )
                else:
                    out.append(
                        f"{C.GREY}{lno:>4} {C.GREY}{prefix} {text}{C.RESET}"
                    )

        return "\n".join(out)

    # ------------------------------------------------------------------
    # Word-level diff
    # ------------------------------------------------------------------

    def word_diff(
        self,
        text_a: str,
        text_b: str,
    ) -> list[WordChange]:
        """Perform intra-line word-level diffing between two texts.

        Returns a flat list of *WordChange* objects.
        """
        changes: list[WordChange] = []
        lines_a = text_a.splitlines()
        lines_b = text_b.splitlines()
        max_lines = max(len(lines_a), len(lines_b))

        for idx in range(max_lines):
            la = lines_a[idx] if idx < len(lines_a) else None
            lb = lines_b[idx] if idx < len(lines_b) else None

            if la is None:
                changes.append(WordChange(
                    type=DiffLineType.ADDED,
                    value=lb,
                    position=idx,
                ))
                continue
            if lb is None:
                changes.append(WordChange(
                    type=DiffLineType.REMOVED,
                    value=la,
                    position=idx,
                    old_position=idx,
                ))
                continue
            if la == lb:
                changes.append(WordChange(
                    type=DiffLineType.CONTEXT,
                    value=la,
                    position=idx,
                    old_position=idx,
                ))
                continue

            # Intra-line word diff using SequenceMatcher on words
            words_a = re.findall(r"\S+|\s+", la)
            words_b = re.findall(r"\S+|\s+", lb)
            sm = difflib.SequenceMatcher(None, words_a, words_b)

            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                segment_a = "".join(words_a[i1:i2])
                segment_b = "".join(words_b[j1:j2])
                if tag == "equal":
                    changes.append(WordChange(
                        type=DiffLineType.CONTEXT,
                        value=segment_a,
                        position=idx,
                        old_position=idx,
                    ))
                elif tag == "delete":
                    changes.append(WordChange(
                        type=DiffLineType.REMOVED,
                        value=segment_a,
                        position=idx,
                        old_position=idx,
                    ))
                elif tag == "insert":
                    changes.append(WordChange(
                        type=DiffLineType.ADDED,
                        value=segment_b,
                        position=idx,
                    ))
                elif tag == "replace":
                    changes.append(WordChange(
                        type=DiffLineType.REMOVED,
                        value=segment_a,
                        position=idx,
                        old_position=idx,
                    ))
                    changes.append(WordChange(
                        type=DiffLineType.ADDED,
                        value=segment_b,
                        position=idx,
                    ))

        return changes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_hunks_from_raw(raw_lines: list[str]) -> list[Hunk]:
        """Parse raw unified-diff lines into *Hunk* objects."""
        hunks: list[Hunk] = []
        current_hunk: Optional[Hunk] = None
        old_no = 0
        new_no = 0

        # Regular expression for @@ hunk header @@
        hunk_re = re.compile(
            r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)"
        )

        for line in raw_lines:
            line = line.rstrip("\n").rstrip("\r")

            # File header lines
            if line.startswith("--- ") or line.startswith("+++ "):
                continue

            m = hunk_re.match(line)
            if m:
                if current_hunk is not None:
                    hunks.append(current_hunk)
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                new_start = int(m.group(3))
                new_count = int(m.group(4)) if m.group(4) is not None else 1
                header_tail = m.group(5).strip()
                current_hunk = Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    header=f"@@ -{old_start},{old_count} +{new_start},{new_count} @@ {header_tail}",
                )
                old_no = old_start
                new_no = new_start
                continue

            if current_hunk is None:
                continue

            if line.startswith("+"):
                current_hunk.lines.append(DiffLine(
                    type=DiffLineType.ADDED,
                    content=line[1:] + "\n",
                    old_line_no=-1,
                    new_line_no=new_no,
                ))
                new_no += 1
            elif line.startswith("-"):
                current_hunk.lines.append(DiffLine(
                    type=DiffLineType.REMOVED,
                    content=line[1:] + "\n",
                    old_line_no=old_no,
                    new_line_no=-1,
                ))
                old_no += 1
            elif line.startswith(" "):
                current_hunk.lines.append(DiffLine(
                    type=DiffLineType.CONTEXT,
                    content=line[1:] + "\n",
                    old_line_no=old_no,
                    new_line_no=new_no,
                ))
                old_no += 1
                new_no += 1
            else:
                # No-newline-at-end-of-file marker or unexpected
                current_hunk.lines.append(DiffLine(
                    type=DiffLineType.CONTEXT,
                    content=line + "\n",
                ))

        if current_hunk is not None:
            hunks.append(current_hunk)

        return hunks

    @staticmethod
    def _parse_hunks(patch: str) -> list[Hunk]:
        """Parse a unified patch into hunks (convenience wrapper)."""
        return DiffEngine.get_hunks_from_raw(patch.splitlines(True))

    @staticmethod
    def _apply_hunk(lines: list[str], hunk: Hunk) -> None:
        """Apply a single hunk *in place* to *lines* (reverse-sorted by caller)."""
        start = max(0, hunk.old_start - 1)
        remove_count = hunk.old_count

        # Verify context if possible
        context_before: list[str] = []
        add_lines: list[str] = []
        remove_seen = 0

        for ln in hunk.lines:
            if ln.type == DiffLineType.CONTEXT:
                context_before.append(ln.content)
            elif ln.type == DiffLineType.ADDED:
                add_lines.append(ln.content)
            elif ln.type == DiffLineType.REMOVED:
                remove_seen += 1

        # Replace the old block with the new block
        replacement: list[str] = []
        for ln in hunk.lines:
            if ln.type == DiffLineType.CONTEXT or ln.type == DiffLineType.ADDED:
                replacement.append(ln.content)

        # Splice
        end = start + remove_count
        lines[start:end] = replacement


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def diff_text(text_a: str, text_b: str, context: int = 3) -> DiffResult:
    """Quick wrapper: produce a *DiffResult* between two strings."""
    return DiffEngine().diff(text_a, text_b, context)


def merge_text(base: str, ours: str, theirs: str) -> MergeResult:
    """Quick wrapper: perform a 3-way merge."""
    return DiffEngine().three_way_merge(base, ours, theirs)
