"""
Context References — File/context reference management for conversations.

Tracks file paths referenced in conversations, handles resolution, content
injection, and deduplication. Integrates with the agent's context system
to provide seamless file content management.

Usage
-----
    manager = ContextReferenceManager()
    ref = manager.add_reference("/path/to/file.py")
    content = await manager.resolve(ref)
    all_refs = manager.get_all_references()
    injected = await manager.build_context_block()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FileReference:
    """
    A reference to a file in the conversation context.

    Attributes
    ----------
    path:
        Absolute file path.
    alias:
        Optional display name.
    content_hash:
        SHA-256 hash of the file content (for change detection).
    size_bytes:
        File size in bytes.
    added_at:
        ISO-8601 timestamp when the reference was added.
    last_resolved_at:
        ISO-8601 timestamp of last resolution.
    resolved_content:
        Cached resolved content.
    max_chars:
        Maximum characters to include from this file.
    metadata:
        Additional metadata (file type, encoding, etc.).
    """

    path: str
    alias: str = ""
    content_hash: str = ""
    size_bytes: int = 0
    added_at: str = ""
    last_resolved_at: str = ""
    resolved_content: str = ""
    max_chars: int = 50_000
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.alias:
            self.alias = Path(self.path).name
        if not self.added_at:
            from datetime import datetime, timezone
            self.added_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_resolved(self) -> bool:
        """Whether this reference has been resolved (content loaded)."""
        return bool(self.resolved_content)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "path": self.path,
            "alias": self.alias,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "added_at": self.added_at,
            "last_resolved_at": self.last_resolved_at,
            "is_resolved": self.is_resolved,
            "max_chars": self.max_chars,
            "metadata": self.metadata,
        }


# ──────────────────────────────────────────────────────────────────────────────
# ContextReferenceManager
# ──────────────────────────────────────────────────────────────────────────────

class ContextReferenceManager:
    """
    Manages file/context references in conversations.

    Handles file path resolution, content injection, deduplication, and
    change detection for files referenced during agent conversations.

    Parameters
    ----------
    max_file_chars:
        Default maximum characters per file.
    max_total_chars:
        Maximum total characters across all files.
    """

    DEFAULT_MAX_FILE_CHARS = 50_000
    DEFAULT_MAX_TOTAL_CHARS = 500_000

    def __init__(
        self,
        max_file_chars: int = DEFAULT_MAX_FILE_CHARS,
        max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
    ) -> None:
        self._max_file_chars = max_file_chars
        self._max_total_chars = max_total_chars
        self._references: List[FileReference] = []
        self._path_index: Dict[str, FileReference] = {}  # normalized path → ref

    @property
    def reference_count(self) -> int:
        """Number of tracked references."""
        return len(self._references)

    @property
    def total_size(self) -> int:
        """Total resolved content size in characters."""
        return sum(len(r.resolved_content) for r in self._references)

    # ── Reference management ──────────────────────────────────────────────

    def add_reference(
        self,
        path: str,
        alias: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> FileReference:
        """
        Add a file reference to track.

        Parameters
        ----------
        path:
            File path (can be relative or absolute).
        alias:
            Optional display name override.
        max_chars:
            Maximum characters to include from this file.

        Returns
        -------
        FileReference
            The created (or existing) reference.
        """
        # Normalize path
        normalized = self._normalize_path(path)
        if normalized in self._path_index:
            return self._path_index[normalized]

        ref = FileReference(
            path=str(normalized),
            alias=alias or "",
            max_chars=max_chars or self._max_file_chars,
        )
        self._references.append(ref)
        self._path_index[normalized] = ref
        logger.debug("Added reference: %s", normalized)
        return ref

    def remove_reference(self, path: str) -> bool:
        """
        Remove a file reference.

        Parameters
        ----------
        path:
            File path to remove.

        Returns
        -------
        bool
            ``True`` if the reference was found and removed.
        """
        normalized = self._normalize_path(path)
        ref = self._path_index.pop(normalized, None)
        if ref:
            self._references.remove(ref)
            return True
        return False

    def has_reference(self, path: str) -> bool:
        """Check if a path is already tracked."""
        return self._normalize_path(path) in self._path_index

    def get_reference(self, path: str) -> Optional[FileReference]:
        """Get a reference by path."""
        return self._path_index.get(self._normalize_path(path))

    def get_all_references(self) -> List[FileReference]:
        """Return all tracked references."""
        return list(self._references)

    def clear(self) -> None:
        """Remove all references."""
        self._references.clear()
        self._path_index.clear()

    # ── Resolution ────────────────────────────────────────────────────────

    async def resolve(self, ref: FileReference, force: bool = False) -> str:
        """
        Resolve a file reference by reading its content.

        Parameters
        ----------
        ref:
            The file reference to resolve.
        force:
            Force re-read even if already resolved.

        Returns
        -------
        str
            The file content (truncated to max_chars).
        """
        if not force and ref.is_resolved:
            return ref.resolved_content

        path = Path(ref.path)
        if not path.exists():
            logger.warning("File not found: %s", ref.path)
            ref.resolved_content = f"[Error: File not found: {ref.path}]"
            return ref.resolved_content

        if not path.is_file():
            ref.resolved_content = f"[Error: Not a file: {ref.path}]"
            return ref.resolved_content

        try:
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(
                None,
                lambda: path.read_text(encoding="utf-8", errors="replace"),
            )

            # Update metadata
            ref.size_bytes = len(content.encode("utf-8"))
            ref.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            # Detect file type
            suffix = path.suffix.lower()
            ref.metadata["file_type"] = suffix.lstrip(".")
            ref.metadata["suffix"] = suffix

            # Truncate if needed
            if len(content) > ref.max_chars:
                original_len = len(content)
                content = content[:ref.max_chars]
                content += f"\n\n[... truncated, {original_len} total chars]"
                ref.metadata["truncated"] = True
                ref.metadata["original_chars"] = original_len

            ref.resolved_content = content

            from datetime import datetime, timezone
            ref.last_resolved_at = datetime.now(timezone.utc).isoformat()

            return content

        except Exception as e:
            logger.error("Error resolving %s: %s", ref.path, e)
            ref.resolved_content = f"[Error reading file: {e}]"
            return ref.resolved_content

    async def resolve_all(self, force: bool = False) -> Dict[str, str]:
        """
        Resolve all tracked references.

        Parameters
        ----------
        force:
            Force re-read all files.

        Returns
        -------
        dict[str, str]
            Mapping of file paths to their resolved content.
        """
        results: Dict[str, str] = {}
        for ref in self._references:
            content = await self.resolve(ref, force=force)
            results[ref.path] = content
        return results

    async def check_changes(self) -> List[Dict[str, Any]]:
        """
        Check if any resolved files have changed since last resolution.

        Returns
        -------
        list[dict]
            List of change records with ``path``, ``changed``, and
            ``old_hash``/``new_hash`` keys.
        """
        changes: List[Dict[str, Any]] = []

        for ref in self._references:
            if not ref.is_resolved:
                continue

            path = Path(ref.path)
            if not path.exists():
                changes.append({
                    "path": ref.path,
                    "changed": True,
                    "reason": "file_deleted",
                    "old_hash": ref.content_hash,
                })
                continue

            try:
                loop = asyncio.get_running_loop()
                content = await loop.run_in_executor(
                    None,
                    lambda: path.read_text(encoding="utf-8", errors="replace"),
                )
                new_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                if new_hash != ref.content_hash:
                    changes.append({
                        "path": ref.path,
                        "changed": True,
                        "reason": "content_modified",
                        "old_hash": ref.content_hash,
                        "new_hash": new_hash,
                    })
            except Exception:
                changes.append({
                    "path": ref.path,
                    "changed": True,
                    "reason": "read_error",
                    "old_hash": ref.content_hash,
                })

        return changes

    # ── Context building ──────────────────────────────────────────────────

    async def build_context_block(
        self,
        max_total_chars: Optional[int] = None,
    ) -> str:
        """
        Build a formatted context block with all resolved file contents.

        Parameters
        ----------
        max_total_chars:
            Maximum total characters. Defaults to instance setting.

        Returns
        -------
        str
            Formatted context block suitable for prompt injection.
        """
        budget = max_total_chars or self._max_total_chars
        parts: List[str] = []
        used_chars = 0

        for ref in self._references:
            if not ref.is_resolved:
                content = await self.resolve(ref)
            else:
                content = ref.resolved_content

            # Try to get a relative path for display
            try:
                display_path = Path(ref.path).relative_to(Path.cwd())
            except (ValueError, RuntimeError):
                display_path = Path(ref.path).name

            header = f"--- File: {display_path} ---"
            footer = f"--- End of {display_path} ---"
            block = f"{header}\n{content}\n{footer}\n"
            block_chars = len(block)

            if used_chars + block_chars > budget:
                remaining = budget - used_chars
                if remaining < 100:
                    break
                # Include a truncated version
                trunc_content = content[:remaining - len(header) - len(footer) - 20]
                block = f"{header}\n{trunc_content}\n[... truncated due to context limit]\n{footer}\n"
                parts.append(block)
                break

            parts.append(block)
            used_chars += block_chars

        return "\n".join(parts)

    def build_reference_list(self) -> str:
        """
        Build a simple list of referenced file paths.

        Returns
        -------
        str
            Comma-separated list of file names.
        """
        names = [ref.alias for ref in self._references]
        return ", ".join(names)

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize a file path to an absolute, resolved path."""
        try:
            return str(Path(path).expanduser().resolve())
        except Exception:
            return str(Path(path).expanduser())
