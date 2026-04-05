"""
Hermes File Tools — enhanced file system operations.

Features:
- Read, write, edit, append, delete, move, copy files
- Directory listing and creation
- File search with glob patterns
- File metadata extraction (size, type, permissions, modified time)
- Batch operations
"""

from __future__ import annotations

import glob
import os
import shutil
import stat
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_path(p: Path) -> Path:
    """Resolve a path safely."""
    return p.expanduser().resolve()


def _format_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


def _format_permissions(mode: int) -> str:
    """Unix-style permission string."""
    parts = []
    for who in ("USR", "GRP", "OTH"):
        for what, letter in ((stat.S_IR, "r"), (stat.S_IW, "w"), (stat.S_IX, "x")):
            parts.append(letter if mode & getattr(stat, f"S_I{what}{who}") else "-")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def hermes_file_read(path: str, offset: int = 0, limit: int = 0) -> str:
    """Read a file and return its contents.

    param path (str): — Path to the file.
    param offset (int): — Line offset to start reading from (0-based). Default: 0.
    param limit (int): — Max lines to read. 0 = entire file. Default: 0.
    """
    try:
        p = _safe_path(Path(path))
        if not p.exists():
            return f"Error: File not found: {p}"
        if not p.is_file():
            return f"Error: Not a file: {p}"

        raw = p.read_bytes()
        if not raw:
            return "(empty file)"

        # Detect encoding
        try:
            import chardet
            detection = chardet.detect(raw)
            encoding = detection.get("encoding") or "utf-8"
        except ImportError:
            encoding = "utf-8"

        try:
            content = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            content = raw.decode("utf-8", errors="replace")

        lines = content.split("\n")
        total_lines = len(lines)

        if offset > 0 or limit > 0:
            lines = lines[offset:]
            if limit > 0:
                lines = lines[:limit]

        header = f"[{len(raw)} bytes, {total_lines} lines, encoding: {encoding}]"
        if offset > 0 or limit > 0:
            header += f" [showing lines {offset+1}-{offset+len(lines)}]"

        return header + "\n" + "\n".join(lines)

    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"


async def hermes_file_write(path: str, content: str) -> str:
    """Write content to a file, creating directories as needed.

    param path (str): — Path to the file.
    param content (str): — Content to write.
    """
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        lines = content.count("\n") + (0 if content.endswith("\n") else 1)
        return f"Wrote {len(content)} bytes ({lines} lines) to {p.resolve()}"
    except Exception as e:
        return f"Error writing to {path}: {e}"


async def hermes_file_edit(path: str, old_str: str, new_str: str) -> str:
    """Find and replace text within a file.

    param path (str): — Path to the file.
    param old_str (str): — Exact text to find.
    param new_str (str): — Replacement text.
    """
    try:
        p = _safe_path(Path(path))
        if not p.exists():
            return f"Error: File not found: {p}"

        content = p.read_text(encoding="utf-8")
        occurrences = content.count(old_str)

        if occurrences == 0:
            return f"Error: old_str not found in {p}. Ensure exact match including whitespace."
        if occurrences > 1:
            return f"Error: old_str found {occurrences} times. Provide more context for a unique match."

        new_content = content.replace(old_str, new_str, 1)
        p.write_text(new_content, encoding="utf-8")
        delta = len(new_str) - len(old_str)
        return f"Edited {p} (delta: {delta:+d} bytes)"

    except Exception as e:
        return f"Error editing {path}: {e}"


async def hermes_file_append(path: str, content: str) -> str:
    """Append content to a file.

    param path (str): — Path to the file.
    param content (str): — Content to append.
    """
    try:
        p = _safe_path(Path(path))
        if p.exists():
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
            return f"Appended {len(content)} bytes to {p}"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Created {p} ({len(content)} bytes)"
    except Exception as e:
        return f"Error appending to {path}: {e}"


async def hermes_file_delete(path: str) -> str:
    """Delete a file.

    param path (str): — Path to the file to delete.
    """
    try:
        p = _safe_path(Path(path))
        if not p.exists():
            return f"Error: Not found: {p}"
        if not p.is_file():
            return f"Error: Not a file: {p}"
        p.unlink()
        return f"Deleted: {p}"
    except Exception as e:
        return f"Error deleting {path}: {e}"


async def hermes_file_move(src: str, dst: str) -> str:
    """Move or rename a file or directory.

    param src (str): — Source path.
    param dst (str): — Destination path.
    """
    try:
        s = _safe_path(Path(src))
        d = _safe_path(Path(dst))
        if not s.exists():
            return f"Error: Source not found: {s}"
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        return f"Moved {s} → {d}"
    except Exception as e:
        return f"Error moving {src} → {dst}: {e}"


async def hermes_file_copy(src: str, dst: str) -> str:
    """Copy a file or directory.

    param src (str): — Source path.
    param dst (str): — Destination path.
    """
    try:
        s = _safe_path(Path(src))
        d = _safe_path(Path(dst))
        if not s.exists():
            return f"Error: Source not found: {s}"
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(str(s), str(d))
            return f"Copied directory {s} → {d}"
        else:
            shutil.copy2(str(s), str(d))
            return f"Copied file {s} → {d}"
    except Exception as e:
        return f"Error copying {src} → {dst}: {e}"


async def hermes_file_list(path: str = ".", show_hidden: bool = False) -> str:
    """List directory contents.

    param path (str): — Directory path. Default: current directory.
    param show_hidden (bool): — Show hidden files. Default: False.
    """
    try:
        p = _safe_path(Path(path))
        if not p.is_dir():
            return f"Error: Not a directory: {p}"

        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]

        if not entries:
            return f"Directory {p} is empty."

        lines = [f"Directory: {p} ({len(entries)} entries)"]
        for entry in entries:
            icon = "dir/" if entry.is_dir() else ""
            try:
                size = f" ({_format_size(entry.stat().st_size)})" if entry.is_file() else ""
            except OSError:
                size = ""
            lines.append(f"  {entry.name}{icon}{size}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing {path}: {e}"


async def hermes_file_search(pattern: str, directory: str = ".") -> str:
    """Search for files matching a glob pattern.

    param pattern (str): — Glob pattern (e.g., '*.py', '**/*.js').
    param directory (str): — Directory to search in.
    """
    try:
        d = _safe_path(Path(directory))
        if not d.is_dir():
            return f"Error: Not a directory: {d}"

        matches = sorted(d.glob(pattern))
        if not matches:
            return f"No files matching '{pattern}' in {d}"

        lines = [f"Found {len(matches)} file(s):"]
        for m in matches[:200]:
            rel = m.relative_to(d)
            try:
                size = f" ({_format_size(m.stat().st_size)})"
            except OSError:
                size = ""
            lines.append(f"  {rel}{size}")

        if len(matches) > 200:
            lines.append(f"  ... and {len(matches) - 200} more")
        return "\n".join(lines)

    except Exception as e:
        return f"Error searching for '{pattern}': {e}"


async def hermes_file_metadata(path: str) -> str:
    """Get detailed metadata for a file or directory.

    param path (str): — Path to the file or directory.
    """
    try:
        p = _safe_path(Path(path))
        if not p.exists():
            return f"Error: Not found: {p}"

        st = p.stat()
        lines = [
            f"Path: {p}",
            f"Type: {'directory' if p.is_dir() else 'file'}",
            f"Size: {_format_size(st.st_size)} ({st.st_size:,} bytes)",
            f"Permissions: {_format_permissions(st.st_mode)}",
            f"Modified: {datetime.fromtimestamp(st.st_mtime).isoformat()}",
            f"Accessed: {datetime.fromtimestamp(st.st_atime).isoformat()}",
            f"Created: {datetime.fromtimestamp(st.st_ctime).isoformat()}",
        ]

        if p.is_file():
            suffix = p.suffix or "(none)"
            lines.append(f"Extension: {suffix}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error getting metadata for {path}: {e}"


async def hermes_file_batch_delete(paths: str) -> str:
    """Delete multiple files at once.

    param paths (str): — JSON array of file paths to delete.
    """
    import json
    try:
        path_list = json.loads(paths) if isinstance(paths, str) else paths
    except json.JSONDecodeError:
        return f"Error: paths must be a JSON array. Got: {paths}"

    results = []
    for p_str in path_list:
        try:
            p = _safe_path(Path(p_str))
            if p.exists() and p.is_file():
                p.unlink()
                results.append(f"  Deleted: {p}")
            else:
                results.append(f"  Skipped (not found): {p_str}")
        except Exception as e:
            results.append(f"  Error ({p_str}): {e}")

    return f"Batch delete ({len(results)} items):\n" + "\n".join(results)


async def hermes_mkdir(path: str) -> str:
    """Create a directory including parent directories.

    param path (str): — Directory path to create.
    """
    try:
        p = Path(path).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return f"Created directory: {p.resolve()}"
    except Exception as e:
        return f"Error creating directory {path}: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="hermes_file_read",
    func=hermes_file_read,
    description="Read a file with optional line offset and limit.",
    toolset="file",
)

ToolRegistry.instance().register(
    name="hermes_file_write",
    func=hermes_file_write,
    description="Write content to a file, creating parent directories as needed.",
    toolset="file",
)

ToolRegistry.instance().register(
    name="hermes_file_edit",
    func=hermes_file_edit,
    description="Find and replace exact text within a file.",
    toolset="file",
)

ToolRegistry.instance().register(
    name="hermes_file_append",
    func=hermes_file_append,
    description="Append content to the end of a file.",
    toolset="file",
)

ToolRegistry.instance().register(
    name="hermes_file_delete",
    func=hermes_file_delete,
    description="Delete a file.",
    toolset="file",
)

ToolRegistry.instance().register(
    name="hermes_file_move",
    func=hermes_file_move,
    description="Move or rename a file or directory.",
    toolset="file",
)

ToolRegistry.instance().register(
    name="hermes_file_copy",
    func=hermes_file_copy,
    description="Copy a file or directory.",
    toolset="file",
)

ToolRegistry.instance().register(
    name="hermes_file_list",
    func=hermes_file_list,
    description="List directory contents.",
    toolset="file",
)

ToolRegistry.instance().register(
    name="hermes_file_search",
    func=hermes_file_search,
    description="Search for files matching a glob pattern.",
    toolset="file",
)

ToolRegistry.instance().register(
    name="hermes_file_metadata",
    func=hermes_file_metadata,
    description="Get detailed metadata for a file or directory.",
    toolset="file",
)

ToolRegistry.instance().register(
    name="hermes_file_batch_delete",
    func=hermes_file_batch_delete,
    description="Delete multiple files at once (JSON array of paths).",
    toolset="file",
)

ToolRegistry.instance().register(
    name="hermes_mkdir",
    func=hermes_mkdir,
    description="Create a directory including parent directories.",
    toolset="file",
)
