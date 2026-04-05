"""
@ Mention System for context injection in Claude Code clone.

Provides intelligent resolution of @mentions in chat messages, including files,
folders, symbols, URLs, git refs, terminal history, clipboard, images (OCR),
environment variables, and glob patterns. Supports autocomplete, token budgeting,
smart truncation, caching, and mention chaining.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_TOKENS_PER_MENTION = 5000
_DEFAULT_TOTAL_BUDGET = 50000
_CACHE_TTL_SECONDS = 120
_TERMINAL_HISTORY_PATH = os.path.expanduser("~/.bash_history")
_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".zip", ".tar", ".gz",
    ".rar", ".7z", ".exe", ".dll", ".so", ".dylib", ".bin", ".wasm",
    ".pyc", ".whl", ".egg", ".parquet", ".sqlite", ".db",
})
_DEFAULT_LINE_COUNT = 50
_MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB
_TOKEN_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " \t\n\r"
    "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
    "áéíóúàèìòùâêîôûäëïöüñçåø"
)

# ---------------------------------------------------------------------------
# Mention type enum
# ---------------------------------------------------------------------------

class MentionType(Enum):
    FILE = "file"
    FOLDER = "folder"
    SYMBOL = "symbol"
    URL = "url"
    GIT_COMMIT = "git_commit"
    GIT_DIFF = "git_diff"
    TERMINAL = "terminal"
    CLIPBOARD = "clipboard"
    IMAGE = "image"
    ENV = "env"
    SELECTION = "selection"
    GLOB = "glob"
    LINE_RANGE = "line_range"
    WEB_SEARCH = "web_search"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Mention:
    type: MentionType
    raw_text: str
    resolved_path: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)
    token_count: int = 0
    cache_key: str = ""
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "raw_text": self.raw_text,
            "resolved_path": self.resolved_path,
            "content": self.content,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "cache_key": self.cache_key,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Mention:
        data = dict(data)
        data["type"] = MentionType(data["type"])
        return cls(**data)


@dataclass
class MentionResult:
    mentions: list[Mention] = field(default_factory=list)
    total_tokens: int = 0
    budget_remaining: int = _DEFAULT_TOTAL_BUDGET
    context_string: str = ""

    def to_dict(self) -> dict:
        return {
            "mentions": [m.to_dict() for m in self.mentions],
            "total_tokens": self.total_tokens,
            "budget_remaining": self.budget_remaining,
            "context_string": self.context_string,
        }


@dataclass
class CompletionItem:
    label: str
    detail: str = ""
    type: MentionType = MentionType.FILE
    insert_text: str = ""

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "detail": self.detail,
            "type": self.type.value,
            "insert_text": self.insert_text,
        }


# ---------------------------------------------------------------------------
# Mention history tracker
# ---------------------------------------------------------------------------

@dataclass
class MentionTurn:
    turn_index: int = 0
    mentions: list[Mention] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class MentionHistory:
    """Tracks which mentions were used across conversation turns."""

    def __init__(self) -> None:
        self._turns: list[MentionTurn] = []
        self._all_cache_keys: set[str] = set()

    def record(self, turn_index: int, mentions: list[Mention]) -> None:
        turn = MentionTurn(turn_index=turn_index, mentions=list(mentions))
        self._turns.append(turn)
        for m in mentions:
            if m.cache_key:
                self._all_cache_keys.add(m.cache_key)

    def get_recent(self, n: int = 5) -> list[Mention]:
        recent: list[Mention] = []
        for turn in reversed(self._turns):
            for m in reversed(turn.mentions):
                if m not in recent:
                    recent.append(m)
                if len(recent) >= n:
                    return recent
        return recent

    def get_turn(self, turn_index: int) -> list[Mention]:
        for turn in self._turns:
            if turn.turn_index == turn_index:
                return turn.mentions
        return []

    def has_been_used(self, cache_key: str) -> bool:
        return cache_key in self._all_cache_keys

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def clear(self) -> None:
        self._turns.clear()
        self._all_cache_keys.clear()


# ---------------------------------------------------------------------------
# Mention resolver
# ---------------------------------------------------------------------------

class MentionResolver:
    """Resolves @mentions in chat messages to actual content for LLM context injection."""

    def __init__(
        self,
        project_path: str = ".",
        max_tokens_per_mention: int = _DEFAULT_MAX_TOKENS_PER_MENTION,
        total_budget: int = _DEFAULT_TOTAL_BUDGET,
    ) -> None:
        self.project_path = os.path.abspath(project_path)
        self.max_tokens_per_mention = max_tokens_per_mention
        self.total_budget = total_budget
        self._cache: dict[str, tuple[float, Mention]] = {}
        self._history = MentionHistory()
        self._turn_counter = 0
        self._terminal_history_cache: Optional[list[str]] = None
        self._terminal_history_mtime: float = 0.0
        self._symbol_index_cache: Optional[dict[str, dict]] = None
        self._symbol_index_mtime: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve(self, text: str) -> MentionResult:
        """Find all @ mentions in *text*, resolve them, and return expanded context.

        Parses the text for mention patterns, resolves each one (using the cache
        when possible), enforces the token budget, and builds a formatted context
        string ready for injection into the LLM prompt.
        """
        parsed = self._parse_mentions(text)
        mentions: list[Mention] = []
        budget_used = 0

        for mention_type, raw in parsed:
            if budget_used >= self.total_budget:
                break

            try:
                mention = await self.resolve_single(raw)
            except Exception:
                continue

            remaining = self.total_budget - budget_used
            effective_max = min(self.max_tokens_per_mention, remaining)

            if mention.token_count > effective_max:
                mention.content = self._truncate_to_tokens(
                    mention.content, effective_max
                )
                mention.token_count = self._token_estimate(mention.content)
                mention.truncated = True

            budget_used += mention.token_count
            mentions.append(mention)

        context_string = await self.build_context_string(mentions)
        self._turn_counter += 1
        self._history.record(self._turn_counter, mentions)

        return MentionResult(
            mentions=mentions,
            total_tokens=budget_used,
            budget_remaining=self.total_budget - budget_used,
            context_string=context_string,
        )

    async def resolve_single(self, mention_text: str) -> Mention:
        """Resolve a single mention string to its content."""
        mention_text = mention_text.strip()
        if not mention_text.startswith("@"):
            mention_text = "@" + mention_text

        # Check cache first
        cache_key = self._make_cache_key(mention_text)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        mention_type, raw_body = self._classify_mention(mention_text)
        mention: Optional[Mention] = None

        if mention_type == MentionType.FILE or mention_type == MentionType.LINE_RANGE:
            mention = await self.resolve_file(raw_body)
        elif mention_type == MentionType.FOLDER:
            mention = await self.resolve_folder(raw_body)
        elif mention_type == MentionType.SYMBOL:
            mention = await self.resolve_symbol(raw_body)
        elif mention_type == MentionType.URL:
            mention = await self.resolve_url(raw_body)
        elif mention_type == MentionType.GIT_COMMIT:
            mention = await self.resolve_git(raw_body)
        elif mention_type == MentionType.GIT_DIFF:
            mention = await self.resolve_git(raw_body, diff=True)
        elif mention_type == MentionType.TERMINAL:
            mention = await self.resolve_terminal(raw_body)
        elif mention_type == MentionType.CLIPBOARD:
            mention = await self._resolve_clipboard()
        elif mention_type == MentionType.IMAGE:
            mention = await self.resolve_image(raw_body)
        elif mention_type == MentionType.ENV:
            mention = await self.resolve_env(raw_body)
        elif mention_type == MentionType.SELECTION:
            mention = await self._resolve_selection()
        elif mention_type == MentionType.GLOB:
            mention = await self._resolve_glob(raw_body)
        elif mention_type == MentionType.WEB_SEARCH:
            mention = await self._resolve_web_search(raw_body)
        else:
            # Fallback: treat as file path
            mention = await self.resolve_file(raw_body)

        if mention is None:
            mention = Mention(
                type=mention_type,
                raw_text=mention_text,
                content=f"<error>Could not resolve mention: {mention_text}</error>",
                metadata={"error": True},
            )

        mention.token_count = self._token_estimate(mention.content)
        mention.cache_key = cache_key
        self._put_in_cache(cache_key, mention)
        return mention

    async def get_completions(self, partial: str, limit: int = 20) -> list[CompletionItem]:
        """Return autocomplete suggestions for a partial @mention string."""
        items: list[CompletionItem] = []

        if not partial.startswith("@"):
            partial = "@" + partial

        body = partial[1:]

        # --- Special prefixes ---
        if body.startswith("git:") and len(body) > 4:
            items.extend(await self._git_completions(body[4:], limit))
        elif body.startswith("term:"):
            items.append(CompletionItem(
                label="@term:50",
                detail="Last 50 terminal commands",
                type=MentionType.TERMINAL,
                insert_text="@term:50",
            ))
            items.append(CompletionItem(
                label="@term:100",
                detail="Last 100 terminal commands",
                type=MentionType.TERMINAL,
                insert_text="@term:100",
            ))
        elif body.startswith("clip"):
            items.append(CompletionItem(
                label="@clip",
                detail="Clipboard contents",
                type=MentionType.CLIPBOARD,
                insert_text="@clip",
            ))
        elif body.startswith("img:"):
            items.extend(await self._image_completions(body[4:], limit))
        elif body.startswith("env"):
            items.append(CompletionItem(
                label="@env",
                detail="All environment variables",
                type=MentionType.ENV,
                insert_text="@env",
            ))
            items.extend(self._env_completions(body[4:], limit))
        elif body.startswith("http://") or body.startswith("https://"):
            # Already a URL, just confirm
            items.append(CompletionItem(
                label=partial,
                detail="Web page",
                type=MentionType.URL,
                insert_text=partial,
            ))
        else:
            # General completion: mix of files, folders, symbols, and globs
            items.extend(await self._file_completions(body, limit))
            items.extend(await self._symbol_completions(body, limit))
            items.extend(await self._folder_completions(body, limit))

        # Deduplicate and truncate
        seen_labels: set[str] = set()
        unique: list[CompletionItem] = []
        for item in items:
            if item.label not in seen_labels:
                seen_labels.add(item.label)
                unique.append(item)
        return unique[:limit]

    # ------------------------------------------------------------------
    # Individual resolvers
    # ------------------------------------------------------------------

    async def resolve_file(self, path: str, line_range: Optional[tuple[int, int]] = None) -> Mention:
        """Resolve a file mention, optionally starting/ending at specific lines."""
        line_start, line_end = None, None

        # Support chaining: file.py:42 or file.py:10-50
        if line_range is None:
            line_range = self._extract_line_range(path)
            path = path.split(":")[0] if ":" in path and not path.startswith(":") else path
            if line_range:
                line_start, line_end = line_range

        abs_path = self._resolve_path(path)
        if abs_path is None or not os.path.isfile(abs_path):
            return Mention(
                type=MentionType.FILE,
                raw_text=f"@{path}",
                resolved_path=path,
                content=f"<error>File not found: {path}</error>",
                metadata={"error": True, "path": path},
            )

        try:
            content = await self._read_file(abs_path, line_start, line_end)
            stat = os.stat(abs_path)
            metadata = {
                "path": abs_path,
                "relative_path": os.path.relpath(abs_path, self.project_path),
                "size": stat.st_size,
                "extension": os.path.splitext(abs_path)[1],
                "lines": content.count("\n") + 1 if content else 0,
            }
            if line_start is not None:
                metadata["line_start"] = line_start
            if line_end is not None:
                metadata["line_end"] = line_end

            mention_type = MentionType.LINE_RANGE if (line_start or line_end) else MentionType.FILE
            return Mention(
                type=mention_type,
                raw_text=f"@{path}",
                resolved_path=abs_path,
                content=content,
                metadata=metadata,
            )
        except Exception as exc:
            return Mention(
                type=MentionType.FILE,
                raw_text=f"@{path}",
                resolved_path=abs_path or path,
                content=f"<error>Error reading file {path}: {exc}</error>",
                metadata={"error": True, "path": path},
            )

    async def resolve_folder(self, path: str, pattern: Optional[str] = None) -> Mention:
        """Resolve a folder mention by reading all (matching) files in the directory."""
        abs_path = self._resolve_path(path.rstrip("/"))
        if abs_path is None or not os.path.isdir(abs_path):
            return Mention(
                type=MentionType.FOLDER,
                raw_text=f"@{path}",
                resolved_path=path,
                content=f"<error>Folder not found: {path}</error>",
                metadata={"error": True, "path": path},
            )

        parts: list[str] = []
        file_count = 0
        total_size = 0
        errors = 0

        entries = sorted(os.listdir(abs_path))
        for entry in entries:
            if entry.startswith("."):
                continue

            entry_path = os.path.join(abs_path, entry)
            ext = os.path.splitext(entry)[1].lower()

            if pattern and not self._matches_glob_pattern(entry, pattern):
                continue

            if os.path.isfile(entry_path) and ext not in _BINARY_EXTENSIONS:
                try:
                    content = await self._read_file(entry_path)
                    rel = os.path.relpath(entry_path, self.project_path)
                    header = f"--- {rel} ---"
                    parts.append(f"{header}\n{content}")
                    file_count += 1
                    total_size += os.path.getsize(entry_path)
                except Exception:
                    errors += 1

            if file_count >= 100:
                parts.append("\n<note>Truncated at 100 files.</note>")
                break

        combined = "\n\n".join(parts)
        if not combined:
            combined = "<empty>No readable files found in folder.</empty>"

        return Mention(
            type=MentionType.FOLDER,
            raw_text=f"@{path}",
            resolved_path=abs_path,
            content=combined,
            metadata={
                "path": abs_path,
                "relative_path": os.path.relpath(abs_path, self.project_path),
                "file_count": file_count,
                "total_size": total_size,
                "errors": errors,
                "pattern": pattern,
            },
        )

    async def resolve_symbol(self, symbol_name: str, file_filter: Optional[str] = None) -> Mention:
        """Resolve a symbol (function/class/variable) mention using AST index."""
        # Strip any @ prefix that may have been passed through
        clean_name = symbol_name.lstrip("@")
        clean_name = clean_name.rstrip("()")
        colon_idx = clean_name.find(":")
        if colon_idx != -1 and colon_idx > 0 and not clean_name[colon_idx - 1].isdigit():
            file_filter_hint = clean_name[:colon_idx]
            clean_name = clean_name[colon_idx + 1:]
            if file_filter is None:
                file_filter = file_filter_hint

        index = await self._get_symbol_index()
        matches = index.get(clean_name, [])

        if file_filter:
            matches = [
                m for m in matches
                if file_filter in m.get("file", "")
            ]

        if not matches:
            # Try partial match
            partial_matches = []
            for sym_name, entries in index.items():
                if clean_name.lower() in sym_name.lower():
                    for entry in entries:
                        if file_filter is None or file_filter in entry.get("file", ""):
                            partial_matches.append((sym_name, entry))
            if len(partial_matches) == 1:
                _, best = partial_matches[0]
                matches = [best]
            elif len(partial_matches) > 1:
                names = [name for name, _ in partial_matches[:5]]
                return Mention(
                    type=MentionType.SYMBOL,
                    raw_text=f"@{symbol_name}",
                    content=(
                        f"<ambiguous>Multiple symbols match '{clean_name}': "
                        + ", ".join(names)
                        + "</ambiguous>"
                    ),
                    metadata={"candidates": names, "partial": True},
                )

        if not matches:
            return Mention(
                type=MentionType.SYMBOL,
                raw_text=f"@{symbol_name}",
                content=f"<error>Symbol not found: {clean_name}</error>",
                metadata={"error": True, "symbol": clean_name},
            )

        # Use the best match (first one)
        best = matches[0]
        file_path = best["file"]
        line_start = best["line_start"]
        line_end = best["line_end"]

        try:
            content = await self._read_file(file_path, line_start, line_end)
            return Mention(
                type=MentionType.SYMBOL,
                raw_text=f"@{symbol_name}",
                resolved_path=file_path,
                content=content,
                metadata={
                    "symbol": clean_name,
                    "kind": best.get("kind", "unknown"),
                    "file": file_path,
                    "line_start": line_start,
                    "line_end": line_end,
                    "relative_path": os.path.relpath(file_path, self.project_path),
                },
            )
        except Exception as exc:
            return Mention(
                type=MentionType.SYMBOL,
                raw_text=f"@{symbol_name}",
                resolved_path=file_path,
                content=f"<error>Error reading symbol {clean_name}: {exc}</error>",
                metadata={"error": True},
            )

    async def resolve_url(self, url: str) -> Mention:
        """Fetch and resolve a URL mention."""
        url = url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        try:
            content = await self._fetch_url_content(url)
            return Mention(
                type=MentionType.URL,
                raw_text=f"@{url}",
                resolved_path=url,
                content=content,
                metadata={
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "fetched_at": time.time(),
                },
            )
        except Exception as exc:
            return Mention(
                type=MentionType.URL,
                raw_text=f"@{url}",
                resolved_path=url,
                content=f"<error>Failed to fetch URL {url}: {exc}</error>",
                metadata={"error": True, "url": url},
            )

    async def resolve_git(self, ref: str, diff: bool = False) -> Mention:
        """Resolve a git commit, diff, or branch mention."""
        if not self._is_git_repo():
            return Mention(
                type=MentionType.GIT_COMMIT,
                raw_text=f"@git:{ref}",
                content="<error>Not a git repository</error>",
                metadata={"error": True},
            )

        if diff or ".." in ref:
            mention_type = MentionType.GIT_DIFF
            content = await self._run_git_command(["git", "diff", ref])
        elif ref in ("staged", "--staged", "cached"):
            mention_type = MentionType.GIT_DIFF
            content = await self._run_git_command(["git", "diff", "--staged"])
        elif ref in ("stat",):
            mention_type = MentionType.GIT_DIFF
            content = await self._run_git_command(["git", "diff", "--stat", "HEAD~1"])
        else:
            mention_type = MentionType.GIT_COMMIT
            content = await self._run_git_command(["git", "show", ref])

        if content is None:
            content = f"<error>Failed to resolve git ref: {ref}</error>"

        return Mention(
            type=mention_type,
            raw_text=f"@git:{ref}",
            resolved_path=ref,
            content=content,
            metadata={"ref": ref, "repo": self.project_path},
        )

    async def resolve_terminal(self, n: Optional[str] = None) -> Mention:
        """Resolve the last *n* terminal commands."""
        try:
            count = int(n) if n and n.isdigit() else _DEFAULT_LINE_COUNT
        except (ValueError, TypeError):
            count = _DEFAULT_LINE_COUNT

        count = max(1, min(count, 500))

        history = await self._get_terminal_history()
        recent = history[-count:]

        if not recent:
            content = "<empty>No terminal history found.</empty>"
        else:
            content = "\n".join(f"{i + 1}: {cmd}" for i, cmd in enumerate(recent))

        return Mention(
            type=MentionType.TERMINAL,
            raw_text=f"@term:{n or _DEFAULT_LINE_COUNT}",
            content=content,
            metadata={
                "command_count": len(recent),
                "requested": count,
                "source": _TERMINAL_HISTORY_PATH,
            },
        )

    async def resolve_image(self, path: str) -> Mention:
        """Resolve an image mention by attempting OCR on the image file."""
        abs_path = self._resolve_path(path)
        if abs_path is None or not os.path.isfile(abs_path):
            return Mention(
                type=MentionType.IMAGE,
                raw_text=f"@img:{path}",
                content=f"<error>Image file not found: {path}</error>",
                metadata={"error": True, "path": path},
            )

        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}:
            return Mention(
                type=MentionType.IMAGE,
                raw_text=f"@img:{path}",
                content=f"<error>Not a supported image format: {ext}</error>",
                metadata={"error": True, "path": path, "extension": ext},
            )

        ocr_text = await self._perform_ocr(abs_path)
        stat = os.stat(abs_path)

        return Mention(
            type=MentionType.IMAGE,
            raw_text=f"@img:{path}",
            resolved_path=abs_path,
            content=ocr_text,
            metadata={
                "path": abs_path,
                "relative_path": os.path.relpath(abs_path, self.project_path),
                "size": stat.st_size,
                "extension": ext,
                "ocr_performed": True,
            },
        )

    async def resolve_env(self, pattern: Optional[str] = None) -> Mention:
        """Resolve environment variables, optionally filtered by a glob pattern."""
        env_vars = dict(os.environ)

        if pattern:
            regex = self._glob_to_regex(pattern)
            env_vars = {
                k: v for k, v in env_vars.items()
                if re.match(regex, k, re.IGNORECASE)
            }

        if not env_vars:
            content = f"<empty>No environment variables matching '{pattern or '*'}'</empty>"
        else:
            lines = [f"{k}={v}" for k, v in sorted(env_vars.items())]
            content = "\n".join(lines)

        return Mention(
            type=MentionType.ENV,
            raw_text=f"@env:{pattern}" if pattern else "@env",
            content=content,
            metadata={
                "pattern": pattern,
                "count": len(env_vars),
                "masked_keys": [
                    k for k in env_vars
                    if any(s in k.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASS"))
                ],
            },
        )

    async def build_context_string(self, mentions: list[Mention]) -> str:
        """Build a formatted context string from resolved mentions for LLM injection."""
        if not mentions:
            return ""

        parts: list[str] = []
        for i, mention in enumerate(mentions, 1):
            header = self._format_mention_header(mention, i)
            body = mention.content
            if mention.truncated:
                body += "\n\n[... content truncated due to token budget ...]"
            parts.append(f"{header}\n{body}")

        separator = "\n" + "=" * 72 + "\n"
        return (
            "\n<mentioned-context>\n"
            + separator.join(parts)
            + "\n</mentioned-context>\n"
        )

    # ------------------------------------------------------------------
    # Mention history
    # ------------------------------------------------------------------

    @property
    def history(self) -> MentionHistory:
        return self._history

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_mentions(self, text: str) -> list[tuple[MentionType, str]]:
        """Extract @mention patterns from *text*.

        Returns a list of ``(MentionType, raw_body)`` tuples.  The raw_body
        excludes the leading ``@`` but includes any qualifier (e.g. ``git:abc``).
        """
        results: list[tuple[MentionType, str]] = []
        seen_spans: list[tuple[int, int]] = []

        # Pattern order matters: more specific patterns must come first.
        patterns = [
            # @git:ref or @git:range
            (r"@git:([^\s]+)", MentionType.GIT_COMMIT),
            # @term:N
            (r"@term:(\d+)", MentionType.TERMINAL),
            # @clip
            (r"@clip\b", MentionType.CLIPBOARD),
            # @img:path
            (r"@img:([^\s]+)", MentionType.IMAGE),
            # @env or @env:pattern
            (r"@env(?::([^\s]+))?", MentionType.ENV),
            # @sel or @selection
            (r"@(?:sel(?:ection)?)\b", MentionType.SELECTION),
            # @search:query
            (r"@search:([^\s]+)", MentionType.WEB_SEARCH),
            # @https://... or @http://...
            (r"@(https?://[^\s]+)", MentionType.URL),
            # @path/to/file.py:line or @path/to/file.py:start-end
            (r"@([\w./\-]+\.\w+:\d+(?:-\d+)?)", MentionType.LINE_RANGE),
            # @path/*.py (glob)
            (r"@([\w./\-]+\*\w*)", MentionType.GLOB),
            # @path/to/file.ext  (MUST come before symbol patterns)
            (r"@([\w./\-]+\.\w+)", MentionType.FILE),
            # @path/to/folder/ (trailing slash)
            (r"@([\w./\-]+/)\s", MentionType.FOLDER),
            # @ClassName  (PascalCase → symbol)
            (r"@([A-Z][a-zA-Z0-9_]*)\b", MentionType.SYMBOL),
            # @function_name or @variable_name  (snake_case → symbol)
            (r"@([a-z_][a-zA-Z0-9_]*)\b", MentionType.SYMBOL),
            # @path/to/folder  (bare path without extension)
            (r"@([\w./\-]+)", MentionType.FILE),
        ]

        for pattern, mention_type in patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                # Skip overlapping matches
                overlaps = any(s <= start < e or s < end <= e for s, e in seen_spans)
                if overlaps:
                    continue
                seen_spans.append((start, end))
                raw_body = match.group(1) if match.lastindex else ""
                # Re-classify git diffs
                if mention_type == MentionType.GIT_COMMIT and raw_body and ".." in raw_body:
                    mention_type = MentionType.GIT_DIFF
                results.append((mention_type, raw_body))

        results.sort(key=lambda r: text.find("@" + r[1]))
        return results

    def _classify_mention(self, mention_text: str) -> tuple[MentionType, str]:
        """Classify a full mention string (with @) into a MentionType and its body."""
        body = mention_text.lstrip("@")
        if body.startswith("git:"):
            ref = body[4:]
            if ".." in ref or ref in ("staged", "--staged", "stat"):
                return MentionType.GIT_DIFF, ref
            return MentionType.GIT_COMMIT, ref
        if body.startswith("term:"):
            return MentionType.TERMINAL, body[5:]
        if body in ("clip", "clipboard"):
            return MentionType.CLIPBOARD, body
        if body.startswith("img:"):
            return MentionType.IMAGE, body[4:]
        if body.startswith("env"):
            pattern = body[3:].lstrip(":") or None
            return MentionType.ENV, pattern or ""
        if body in ("sel", "selection"):
            return MentionType.SELECTION, ""
        if body.startswith("search:"):
            return MentionType.WEB_SEARCH, body[7:]
        if body.startswith("http://") or body.startswith("https://"):
            return MentionType.URL, body
        if "*" in body:
            return MentionType.GLOB, body
        if body.endswith("/"):
            return MentionType.FOLDER, body
        if ":" in body:
            return MentionType.LINE_RANGE, body
        if self._looks_like_symbol(body):
            return MentionType.SYMBOL, body
        if os.path.isfile(self._resolve_path(body) or ""):
            return MentionType.FILE, body
        if os.path.isdir(self._resolve_path(body) or ""):
            return MentionType.FOLDER, body
        # Detect file-like names by extension even if file doesn't exist yet
        if "." in os.path.basename(body) and not body.startswith("."):
            return MentionType.FILE, body
        return MentionType.SYMBOL, body

    # ------------------------------------------------------------------
    # Token estimation & truncation
    # ------------------------------------------------------------------

    def _token_estimate(self, text: str) -> int:
        """Rough token count using ~4 chars per token heuristic."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Smartly truncate text to fit within *max_tokens*.

        Strategy:
        1. If already within budget, return as-is.
        2. Try to truncate at the last complete paragraph.
        3. Fall back to line-level truncation.
        4. Final fallback: character-level truncation.
        """
        if not text:
            return text

        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        # Try paragraph-level
        paragraphs = text.split("\n\n")
        truncated_parts: list[str] = []
        current_len = 0
        for para in paragraphs:
            if current_len + len(para) + 2 > max_chars and truncated_parts:
                break
            truncated_parts.append(para)
            current_len += len(para) + 2

        result = "\n\n".join(truncated_parts)
        if len(result) <= max_chars:
            return result

        # Line-level
        lines = text.split("\n")
        truncated_lines: list[str] = []
        current_len = 0
        for line in lines:
            if current_len + len(line) + 1 > max_chars and truncated_lines:
                break
            truncated_lines.append(line)
            current_len += len(line) + 1

        result = "\n".join(truncated_lines)
        if len(result) <= max_chars:
            return result

        # Hard character truncation at word boundary
        result = text[:max_chars]
        last_space = result.rfind(" ")
        if last_space > max_chars * 0.7:
            result = result[:last_space]
        return result

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[Mention]:
        if key in self._cache:
            ts, mention = self._cache[key]
            if time.time() - ts < _CACHE_TTL_SECONDS:
                return mention
            del self._cache[key]
        return None

    def _put_in_cache(self, key: str, mention: Mention) -> None:
        self._cache[key] = (time.time(), mention)
        # Evict old entries to keep memory bounded
        if len(self._cache) > 200:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]

    def clear_cache(self) -> None:
        self._cache.clear()
        self._terminal_history_cache = None
        self._symbol_index_cache = None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str) -> Optional[str]:
        """Resolve a possibly-relative path against the project root."""
        if not path:
            return None
        p = Path(path)
        if p.is_absolute() and p.exists():
            return str(p)
        joined = os.path.join(self.project_path, path)
        if os.path.exists(joined):
            return os.path.abspath(joined)
        return None

    def _extract_line_range(self, path: str) -> Optional[tuple[int, int]]:
        """Extract a ``start`` or ``start-end`` line range from a path string.

        E.g. ``"auth.py:42"`` → ``(42, None)``, ``"auth.py:10-50"`` → ``(10, 50)``.
        """
        colon_match = re.search(r":(\d+)(?:-(\d+))?$", path)
        if colon_match:
            start = int(colon_match.group(1))
            end = int(colon_match.group(2)) if colon_match.group(2) else None
            return (start, end)
        return None

    # ------------------------------------------------------------------
    # File I/O helpers
    # ------------------------------------------------------------------

    async def _read_file(
        self,
        path: str,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
    ) -> str:
        """Read a file, optionally returning only a range of lines (1-indexed)."""
        def _read() -> str:
            stat = os.stat(path)
            if stat.st_size > _MAX_FILE_SIZE_BYTES:
                return f"<error>File too large ({stat.st_size} bytes, max {_MAX_FILE_SIZE_BYTES})</error>"

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if line_start is None and line_end is None:
                    return f.read()

                lines = f.readlines()
                start_idx = max(0, (line_start or 1) - 1)
                end_idx = min(len(lines), line_end or len(lines))
                selected = lines[start_idx:end_idx]
                # Add line numbers
                numbered = []
                for i, line in enumerate(selected, start=start_idx + 1):
                    numbered.append(f"{i:>6}\t{line}")
                return "".join(numbered)

        return await asyncio.get_event_loop().run_in_executor(None, _read)

    # ------------------------------------------------------------------
    # URL fetching
    # ------------------------------------------------------------------

    async def _fetch_url_content(self, url: str) -> str:
        """Fetch a URL and extract its main text content.

        Uses a lightweight approach without heavy dependencies: attempts
        ``urllib`` first, then falls back to subprocess ``curl``.
        """
        import urllib.request
        import urllib.error

        loop = asyncio.get_event_loop()

        def _urllib_fetch() -> str:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ClaudeCode-MentionBot/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read(500_000).decode("utf-8", errors="replace")
            return self._strip_html(raw)

        try:
            return await loop.run_in_executor(None, _urllib_fetch)
        except Exception:
            pass

        # Fallback: curl
        try:
            proc = await asyncio.create_subprocess_exec(
                "curl", "-sL", "--max-time", "15", url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
            html = stdout.decode("utf-8", errors="replace")
            return self._strip_html(html)
        except Exception as exc:
            raise RuntimeError(f"All fetch methods failed: {exc}") from exc

    @staticmethod
    def _strip_html(html: str) -> str:
        """Crude HTML-to-text extraction: remove tags, scripts, and styles."""
        # Remove script and style blocks
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode common entities
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        text = text.replace("&nbsp;", " ")
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Limit length
        if len(text) > 50_000:
            text = text[:50_000] + "\n[... content truncated ...]"
        return text

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    def _is_git_repo(self) -> bool:
        return os.path.isdir(os.path.join(self.project_path, ".git"))

    async def _run_git_command(self, cmd: list[str]) -> Optional[str]:
        """Run a git command and return stdout."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace").strip()
                return f"<error>git command failed: {err}</error>"
            output = stdout.decode("utf-8", errors="replace")
            if len(output) > 100_000:
                output = output[:100_000] + "\n[... git output truncated ...]"
            return output
        except asyncio.TimeoutError:
            return "<error>git command timed out</error>"
        except Exception as exc:
            return f"<error>git command error: {exc}</error>"

    # ------------------------------------------------------------------
    # Terminal history
    # ------------------------------------------------------------------

    async def _get_terminal_history(self) -> list[str]:
        """Read the terminal history file, caching the result."""
        try:
            mtime = os.path.getmtime(_TERMINAL_HISTORY_PATH)
        except OSError:
            # Try zsh
            zsh_path = os.path.expanduser("~/.zsh_history")
            try:
                mtime = os.path.getmtime(zsh_path)
                history_path = zsh_path
            except OSError:
                return []

        history_path = _TERMINAL_HISTORY_PATH if os.path.exists(_TERMINAL_HISTORY_PATH) else os.path.expanduser("~/.zsh_history")
        if not os.path.exists(history_path):
            return []

        if (self._terminal_history_cache is not None
                and self._terminal_history_mtime == mtime):
            return self._terminal_history_cache

        def _read():
            with open(history_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            cleaned = []
            for line in lines:
                stripped = line.strip()
                if stripped:
                    # Handle zsh extended_history format: ": timestamp:command"
                    if stripped.startswith(":") and ":" in stripped[1:]:
                        parts = stripped.split(":", 2)
                        if len(parts) == 3 and parts[2].strip():
                            cleaned.append(parts[2].strip())
                        continue
                    cleaned.append(stripped)
            return cleaned

        self._terminal_history_cache = await asyncio.get_event_loop().run_in_executor(None, _read)
        self._terminal_history_mtime = mtime
        return self._terminal_history_cache

    # ------------------------------------------------------------------
    # Symbol index (AST-based)
    # ------------------------------------------------------------------

    async def _get_symbol_index(self) -> dict[str, list[dict]]:
        """Build (or return cached) index of top-level symbols in Python files."""
        # Check mtime of project directory
        try:
            mtime = os.path.getmtime(self.project_path)
        except OSError:
            mtime = 0.0

        if (self._symbol_index_cache is not None
                and self._symbol_index_mtime == mtime):
            return self._symbol_index_cache

        def _build_index() -> dict[str, list[dict]]:
            index: dict[str, list[dict]] = {}
            for root, _dirs, files in os.walk(self.project_path):
                # Skip common non-source directories
                skip = {".git", "__pycache__", "node_modules", ".venv", "venv",
                        ".tox", ".eggs", "dist", "build", ".mypy_cache"}
                rel_root = os.path.relpath(root, self.project_path)
                if any(part in skip for part in rel_root.split(os.sep)):
                    continue

                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        self._index_python_file(fpath, index)
                    except Exception:
                        continue
            return index

        self._symbol_index_cache = await asyncio.get_event_loop().run_in_executor(None, _build_index)
        self._symbol_index_mtime = mtime
        return self._symbol_index_cache

    def _index_python_file(self, filepath: str, index: dict[str, list[dict]]) -> None:
        """Parse a Python file with AST and add top-level symbols to the index."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()

        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError:
            return

        for node in ast.iter_child_nodes(tree):
            name = None
            kind = "unknown"
            if isinstance(node, ast.FunctionDef):
                name = node.name
                kind = "function"
            elif isinstance(node, ast.AsyncFunctionDef):
                name = node.name
                kind = "async_function"
            elif isinstance(node, ast.ClassDef):
                name = node.name
                kind = "class"
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        kind = "variable"
                        break
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    kind = "import"
                    self._add_symbol(index, name, kind, filepath, node.lineno, node.end_lineno or node.lineno)
                continue
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    kind = "import_from"
                    self._add_symbol(index, name, kind, filepath, node.lineno, node.end_lineno or node.lineno)
                continue

            if name:
                end = node.end_lineno or node.lineno
                self._add_symbol(index, name, kind, filepath, node.lineno, end)

                # Also index methods inside classes
                if isinstance(node, ast.ClassDef):
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_name = f"{name}.{child.name}"
                            child_end = child.end_lineno or child.lineno
                            self._add_symbol(
                                index, method_name,
                                "method" if isinstance(child, ast.FunctionDef) else "async_method",
                                filepath, child.lineno, child_end,
                            )

    @staticmethod
    def _add_symbol(
        index: dict[str, list[dict]],
        name: str,
        kind: str,
        filepath: str,
        line_start: int,
        line_end: int,
    ) -> None:
        entry = {
            "kind": kind,
            "file": filepath,
            "line_start": line_start,
            "line_end": line_end,
        }
        index.setdefault(name, []).append(entry)

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    async def _perform_ocr(self, image_path: str) -> str:
        """Attempt OCR on an image file.

        Tries ``tesseract`` via subprocess, falls back to a placeholder message
        if tesseract is not available.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "tesseract", image_path, "stdout", "--psm", "6",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0:
                text = stdout.decode("utf-8", errors="replace").strip()
                if text:
                    return text
        except (FileNotFoundError, asyncio.TimeoutError):
            pass
        except Exception:
            pass

        # Try pytesseract if installed
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore

            def _ocr():
                img = Image.open(image_path)
                return pytesseract.image_to_string(img)

            text = await asyncio.get_event_loop().run_in_executor(None, _ocr)
            if text and text.strip():
                return text.strip()
        except ImportError:
            pass
        except Exception:
            pass

        return (
            f"<image path=\"{image_path}\">\n"
            f"  Note: OCR could not be performed on this image.\n"
            f"  Install tesseract or pytesseract+Pillow for text extraction.\n"
            f"  File size: {os.path.getsize(image_path)} bytes\n"
            f"</image>"
        )

    # ------------------------------------------------------------------
    # Clipboard (cross-platform)
    # ------------------------------------------------------------------

    async def _resolve_clipboard(self) -> Mention:
        """Resolve clipboard contents."""
        content = await self._read_clipboard()

        return Mention(
            type=MentionType.CLIPBOARD,
            raw_text="@clip",
            content=content or "<empty>Clipboard is empty or inaccessible.</empty>",
            metadata={
                "length": len(content) if content else 0,
                "source": "clipboard",
            },
        )

    async def _read_clipboard(self) -> Optional[str]:
        """Read the system clipboard using platform-appropriate methods."""
        platform = os.sys.platform

        if platform == "darwin":
            try:
                proc = await asyncio.create_subprocess_exec(
                    "pbpaste",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                return stdout.decode("utf-8", errors="replace")
            except Exception:
                pass

        elif platform.startswith("linux"):
            for cmd in [["xclip", "-selection", "clipboard", "-o"],
                        ["xsel", "--clipboard", "--output"]]:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                    if proc.returncode == 0:
                        return stdout.decode("utf-8", errors="replace")
                except Exception:
                    continue

        elif platform == "win32":
            try:
                proc = await asyncio.create_subprocess_exec(
                    "powershell", "-command", "Get-Clipboard",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                return stdout.decode("utf-8", errors="replace")
            except Exception:
                pass

        # Try python clipboard libraries
        try:
            import pyperclip  # type: ignore
            return pyperclip.paste()
        except ImportError:
            pass

        return None

    # ------------------------------------------------------------------
    # Selection (editor integration stub)
    # ------------------------------------------------------------------

    async def _resolve_selection(self) -> Mention:
        """Resolve the current text selection in the editor.

        This is an integration point: in a real implementation this would
        communicate with the editor (VS Code, Neovim, etc.) via LSP or IPC.
        """
        return Mention(
            type=MentionType.SELECTION,
            raw_text="@sel",
            content=(
                "<empty>No active selection detected. "
                "Select text in your editor and use @sel to include it.</empty>"
            ),
            metadata={"source": "editor_selection"},
        )

    # ------------------------------------------------------------------
    # Glob resolver
    # ------------------------------------------------------------------

    async def _resolve_glob(self, pattern: str) -> Mention:
        """Resolve a glob pattern like ``src/*.py`` to matching files."""
        import glob as glob_mod

        abs_pattern = os.path.join(self.project_path, pattern)
        matching = glob_mod.glob(abs_pattern, recursive=True)

        # Filter out directories and binary files
        files = [
            f for f in matching
            if os.path.isfile(f) and os.path.splitext(f)[1].lower() not in _BINARY_EXTENSIONS
        ]
        files.sort()

        if not files:
            return Mention(
                type=MentionType.GLOB,
                raw_text=f"@{pattern}",
                content=f"<empty>No files match pattern: {pattern}</empty>",
                metadata={"pattern": pattern, "match_count": 0},
            )

        parts: list[str] = []
        total_size = 0
        for fpath in files[:100]:
            try:
                content = await self._read_file(fpath)
                rel = os.path.relpath(fpath, self.project_path)
                parts.append(f"--- {rel} ---\n{content}")
                total_size += os.path.getsize(fpath)
            except Exception:
                parts.append(f"--- {fpath} ---\n<error>Could not read file.</error>")

        if len(files) > 100:
            parts.append(f"\n<note>{len(files) - 100} more files not shown.</note>")

        return Mention(
            type=MentionType.GLOB,
            raw_text=f"@{pattern}",
            content="\n\n".join(parts),
            metadata={
                "pattern": pattern,
                "match_count": len(files),
                "shown": min(len(files), 100),
                "total_size": total_size,
            },
        )

    # ------------------------------------------------------------------
    # Web search (stub)
    # ------------------------------------------------------------------

    async def _resolve_web_search(self, query: str) -> Mention:
        """Resolve a @search: query mention.

        In a production system this would call a search API. Here we provide
        a structured placeholder and attempt a basic web fetch of a search URL.
        """
        encoded = query.replace(" ", "+")
        search_url = f"https://www.google.com/search?q={encoded}"

        try:
            preview = await self._fetch_url_content(search_url)
            return Mention(
                type=MentionType.WEB_SEARCH,
                raw_text=f"@search:{query}",
                resolved_path=search_url,
                content=preview[:10_000] if len(preview) > 10_000 else preview,
                metadata={"query": query, "url": search_url},
            )
        except Exception as exc:
            return Mention(
                type=MentionType.WEB_SEARCH,
                raw_text=f"@search:{query}",
                content=f"<error>Search failed: {exc}</error>",
                metadata={"error": True, "query": query},
            )

    # ------------------------------------------------------------------
    # Autocomplete helpers
    # ------------------------------------------------------------------

    async def _file_completions(self, partial: str, limit: int) -> list[CompletionItem]:
        """Suggest file paths matching *partial*."""
        items: list[CompletionItem] = []
        dir_part = os.path.dirname(partial) or "."
        prefix = os.path.basename(partial)

        abs_dir = self._resolve_path(dir_part)
        if abs_dir is None or not os.path.isdir(abs_dir):
            return items

        try:
            entries = os.listdir(abs_dir)
        except PermissionError:
            return items

        for entry in sorted(entries):
            if entry.startswith(".") and not prefix.startswith("."):
                continue
            if prefix and not entry.lower().startswith(prefix.lower()):
                continue

            entry_path = os.path.join(abs_dir, entry)
            rel_path = os.path.relpath(entry_path, self.project_path)

            if os.path.isfile(entry_path):
                ext = os.path.splitext(entry)[1].lower()
                if ext not in _BINARY_EXTENSIONS:
                    items.append(CompletionItem(
                        label=f"@{rel_path}",
                        detail=f"File ({ext})",
                        type=MentionType.FILE,
                        insert_text=f"@{rel_path}",
                    ))
            elif os.path.isdir(entry_path):
                items.append(CompletionItem(
                    label=f"@{rel_path}/",
                    detail="Folder",
                    type=MentionType.FOLDER,
                    insert_text=f"@{rel_path}/",
                ))

            if len(items) >= limit:
                break
        return items

    async def _folder_completions(self, partial: str, limit: int) -> list[CompletionItem]:
        """Suggest folders matching *partial*."""
        items: list[CompletionItem] = []
        abs_path = self._resolve_path(partial.rstrip("/"))
        if abs_path and os.path.isdir(abs_path):
            items.append(CompletionItem(
                label=f"@{partial}",
                detail="Folder",
                type=MentionType.FOLDER,
                insert_text=f"@{partial}",
            ))
        return items[:limit]

    async def _symbol_completions(self, partial: str, limit: int) -> list[CompletionItem]:
        """Suggest symbols matching *partial*."""
        items: list[CompletionItem] = []
        if not partial or not partial[0].isalpha() and partial[0] != "_":
            return items

        index = await self._get_symbol_index()
        lower_partial = partial.lower()

        matches = [
            (name, entries[0])
            for name, entries in index.items()
            if lower_partial in name.lower()
        ]

        matches.sort(key=lambda x: self._symbol_sort_key(x[0], lower_partial))

        for name, entry in matches[:limit]:
            rel = os.path.relpath(entry["file"], self.project_path)
            items.append(CompletionItem(
                label=f"@{name}",
                detail=f"{entry['kind']} in {rel}:{entry['line_start']}",
                type=MentionType.SYMBOL,
                insert_text=f"@{name}",
            ))

        return items

    @staticmethod
    def _symbol_sort_key(name: str, lower_query: str) -> tuple[int, int, str]:
        """Sort symbols: exact match first, then starts-with, then alphabetical."""
        lower_name = name.lower()
        if lower_name == lower_query:
            return (0, 0, name)
        if lower_name.startswith(lower_query):
            return (1, 0, name)
        return (2, 0, name)

    async def _git_completions(self, partial: str, limit: int) -> list[CompletionItem]:
        """Suggest git refs."""
        items: list[CompletionItem] = []
        if not self._is_git_repo():
            return items

        # Try to get recent branches/commits
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "branch", "--list", f"*{partial}*" if partial else "",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            if proc.returncode == 0:
                branches = stdout.decode("utf-8", errors="replace").strip().split("\n")
                for branch in branches[:limit]:
                    branch = branch.strip().lstrip("* ")
                    if branch:
                        items.append(CompletionItem(
                            label=f"@git:{branch}",
                            detail=f"Branch: {branch}",
                            type=MentionType.GIT_COMMIT,
                            insert_text=f"@git:{branch}",
                        ))
        except Exception:
            pass

        if not items:
            items.append(CompletionItem(
                label=f"@git:{partial or 'HEAD'}",
                detail="Git ref / commit / diff range",
                type=MentionType.GIT_COMMIT,
                insert_text=f"@git:{partial or 'HEAD'}",
            ))
            items.append(CompletionItem(
                label="@git:HEAD~3..HEAD",
                detail="Git diff range",
                type=MentionType.GIT_DIFF,
                insert_text="@git:HEAD~3..HEAD",
            ))
            items.append(CompletionItem(
                label="@git:staged",
                detail="Staged changes",
                type=MentionType.GIT_DIFF,
                insert_text="@git:staged",
            ))

        return items[:limit]

    async def _image_completions(self, partial: str, limit: int) -> list[CompletionItem]:
        """Suggest image files matching *partial*."""
        items: list[CompletionItem] = []
        dir_part = os.path.dirname(partial) or "."
        prefix = os.path.basename(partial)

        abs_dir = self._resolve_path(dir_part)
        if abs_dir is None or not os.path.isdir(abs_dir):
            return items

        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".svg"}

        try:
            entries = os.listdir(abs_dir)
        except PermissionError:
            return items

        for entry in sorted(entries):
            if entry.startswith(".") and not prefix.startswith("."):
                continue
            if prefix and not entry.lower().startswith(prefix.lower()):
                continue
            if os.path.splitext(entry)[1].lower() not in image_exts:
                continue

            rel = os.path.relpath(os.path.join(abs_dir, entry), self.project_path)
            items.append(CompletionItem(
                label=f"@img:{rel}",
                detail=f"Image ({os.path.splitext(entry)[1]})",
                type=MentionType.IMAGE,
                insert_text=f"@img:{rel}",
            ))
            if len(items) >= limit:
                break
        return items

    def _env_completions(self, partial: str, limit: int) -> list[CompletionItem]:
        """Suggest environment variables matching *partial*."""
        items: list[CompletionItem] = []
        partial = partial.lstrip(":")

        for key in sorted(os.environ):
            if partial and partial.replace("*", "").lower() not in key.lower():
                continue
            is_sensitive = any(
                s in key.upper()
                for s in ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASS")
            )
            display_val = "***" if is_sensitive else os.environ[key][:50]
            items.append(CompletionItem(
                label=f"@env:{key}",
                detail=f"{display_val}{' (sensitive)' if is_sensitive else ''}",
                type=MentionType.ENV,
                insert_text=f"@env:{key}",
            ))
            if len(items) >= limit:
                break
        return items

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _looks_like_symbol(self, name: str) -> bool:
        """Heuristic: does *name* look like a code symbol rather than a file path?"""
        if not name:
            return False
        # Contains path separators → not a symbol
        if "/" in name or "\\" in name:
            return False
        # Has a file extension → not a symbol
        if "." in name and not name[0].isalpha():
            return False
        # Python-like symbol: starts with letter/underscore, contains only identifier chars
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            # But exclude common file-like names without extensions
            if name in ("src", "lib", "test", "tests", "bin", "docs", "build"):
                return False
            return True
        return False

    @staticmethod
    def _matches_glob_pattern(filename: str, pattern: str) -> bool:
        """Check if *filename* matches a simple glob *pattern* (e.g. ``*.py``)."""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)

    @staticmethod
    def _glob_to_regex(pattern: str) -> str:
        """Convert a simple glob pattern to a regex string."""
        result = ""
        for char in pattern:
            if char == "*":
                result += ".*"
            elif char == "?":
                result += "."
            elif char in ".+^${}()|[]\\":
                result += "\\" + char
            else:
                result += char
        return f"^{result}$"

    @staticmethod
    def _format_mention_header(mention: Mention, index: int) -> str:
        """Format a header line for a mention in the context string."""
        path_info = mention.resolved_path or mention.raw_text
        meta_parts = []
        if "relative_path" in mention.metadata:
            meta_parts.append(mention.metadata["relative_path"])
        if "kind" in mention.metadata:
            meta_parts.append(mention.metadata["kind"])
        if "line_start" in mention.metadata:
            line_info = f"line {mention.metadata['line_start']}"
            if "line_end" in mention.metadata:
                line_info += f"-{mention.metadata['line_end']}"
            meta_parts.append(line_info)
        if "symbol" in mention.metadata:
            meta_parts.insert(0, mention.metadata["symbol"])
        if "url" in mention.metadata:
            meta_parts.append(mention.metadata["url"])

        type_label = mention.type.value
        detail = " | ".join(meta_parts) if meta_parts else path_info
        return f"[{index}] @{type_label}: {detail}"
