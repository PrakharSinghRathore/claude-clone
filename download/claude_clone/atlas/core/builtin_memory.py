"""
Built-in Memory Provider — File-based memory storage using MEMORY.md and USER.md.

Implements the ``MemoryProvider`` interface using flat Markdown files for
persistent storage. Provides session search capability, auto-save, and
auto-summarization for long conversation histories.

Storage Layout
--------------
```
<atlas_data_home>/memory/
    MEMORY.md        — Agent's persistent memory (project knowledge, decisions)
    USER.md          — User preferences and profile information
    sessions/
        <session_id>.json  — Per-session conversation summaries
```

Usage
-----
    provider = BuiltinMemoryProvider()
    await provider.initialize()

    # Store a memory
    entry = MemoryEntry(id="abc", content="User prefers dark theme", role="user")
    await provider.store(entry)

    # Search
    results = await provider.search("theme preference")
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from atlas.constants import (
    ATLAS_DATA_HOME,
    MEMORY_DIR,
    MEMORY_FILE,
    SESSIONS_DIR,
    USER_PROFILE_FILE,
)
from atlas.core.memory_provider import MemoryEntry, MemoryProvider

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def _generate_id() -> str:
    """Generate a short unique identifier."""
    return uuid.uuid4().hex[:16]


def _now_iso() -> str:
    """Current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _tokenize_simple(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokenizer for search.

    Splits text into lowercase tokens, removing stop words and short tokens.
    """
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than", "too",
        "very", "just", "because", "if", "when", "where", "how", "what", "which",
        "who", "whom", "this", "that", "these", "those", "i", "me", "my", "we",
        "our", "you", "your", "he", "him", "his", "she", "her", "it", "its",
        "they", "them", "their",
    }
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in stop_words]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ──────────────────────────────────────────────────────────────────────────────
# In-memory entry store
# ──────────────────────────────────────────────────────────────────────────────

class _EntryStore:
    """
    In-memory store for memory entries with optional persistence.

    Provides TF-IDF–style search using a simple bag-of-words approach.
    Entries can be persisted to and loaded from JSON files.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, MemoryEntry] = {}
        self._tfidf_vocabulary: Dict[str, int] = {}  # token → index
        self._doc_freq: Dict[str, int] = {}  # token → document frequency

    def add(self, entry: MemoryEntry) -> None:
        """Add or update an entry."""
        self._entries[entry.id] = entry
        self._update_index(entry)

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve an entry by ID."""
        return self._entries.get(entry_id)

    def remove(self, entry_id: str) -> bool:
        """Remove an entry. Returns True if it existed."""
        return self._entries.pop(entry_id, None) is not None

    def list_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        session_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """List entries sorted by timestamp (newest first)."""
        entries = list(self._entries.values())
        if session_id is not None:
            entries = [e for e in entries if e.session_id == session_id]
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[offset: offset + limit]

    def search(
        self,
        query: str,
        limit: int = 10,
        tags: Optional[List[str]] = None,
    ) -> List[MemoryEntry]:
        """Search entries by relevance to the query."""
        query_tokens = _tokenize_simple(query)
        if not query_tokens:
            return []

        # Score each entry
        scored: List[tuple[float, MemoryEntry]] = []
        for entry in self._entries.values():
            # Tag filtering
            if tags:
                if not all(t in entry.tags for t in tags):
                    continue

            # Keyword matching
            entry_tokens = _tokenize_simple(entry.content)
            overlap = len(set(query_tokens) & set(entry_tokens))
            if overlap > 0:
                # Simple scoring: keyword overlap + recency bonus
                score = float(overlap)
                # Boost recent entries slightly
                try:
                    entry_time = datetime.fromisoformat(entry.timestamp)
                    age_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                    recency_bonus = max(0, 1.0 - age_hours / 720)  # Decays over 30 days
                    score *= (1.0 + recency_bonus * 0.5)
                except (ValueError, TypeError):
                    pass
                # Boost by importance
                score *= (0.5 + entry.importance)
                scored.append((score, entry))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def _update_index(self, entry: MemoryEntry) -> None:
        """Update the search index with a new entry."""
        tokens = set(_tokenize_simple(entry.content))
        for token in tokens:
            self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the store to a dict."""
        return {
            "entries": {eid: e.to_dict() for eid, e in self._entries.items()},
            "doc_freq": self._doc_freq,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load the store from a dict."""
        entries_data = data.get("entries", {})
        for eid, edata in entries_data.items():
            self._entries[eid] = MemoryEntry.from_dict(edata)
        self._doc_freq = data.get("doc_freq", {})

    @property
    def count(self) -> int:
        """Number of stored entries."""
        return len(self._entries)


# ──────────────────────────────────────────────────────────────────────────────
# BuiltinMemoryProvider
# ──────────────────────────────────────────────────────────────────────────────

class BuiltinMemoryProvider(MemoryProvider):
    """
    File-based memory provider using MEMORY.md and USER.md.

    Provides persistent memory storage with:
    - MEMORY.md: Agent's persistent knowledge about the project/user
    - USER.md: User profile and preferences
    - Session summaries stored as JSON files
    - In-memory entry store with TF-IDF–style search
    - Auto-save and auto-summarize capabilities

    Parameters
    ----------
    memory_dir:
        Optional custom directory for memory files. If ``None``, uses the
        default Atlas data directory.
    auto_save:
        Whether to automatically persist changes to disk.
    auto_summarize:
        Whether to auto-summarize long session histories.
    """

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        auto_save: bool = True,
        auto_summarize: bool = True,
    ) -> None:
        if memory_dir:
            self._base_dir = Path(memory_dir)
        else:
            self._base_dir = ATLAS_DATA_HOME / MEMORY_DIR
        self._sessions_dir = self._base_dir / SESSIONS_DIR
        self._memory_file = self._base_dir / MEMORY_FILE
        self._user_file = self._base_dir / USER_PROFILE_FILE
        self._entries_file = self._base_dir / "entries.json"

        self._auto_save = auto_save
        self._auto_summarize = auto_summarize
        self._store = _EntryStore()
        self._memory_content: str = ""
        self._user_content: str = ""

    # ── MemoryProvider interface ──────────────────────────────────────────

    async def initialize(self) -> None:
        """Set up directories, load existing data, and build the index."""
        # Ensure directories exist
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

        # Load existing files
        self._memory_content = await self._read_file(self._memory_file)
        self._user_content = await self._read_file(self._user_file)

        # Load entry store
        await self._load_entries()

        logger.info(
            "Built-in memory initialized: %d entries, memory=%d chars, user=%d chars",
            self._store.count,
            len(self._memory_content),
            len(self._user_content),
        )

    async def close(self) -> None:
        """Persist data and release resources."""
        if self._auto_save:
            await self._persist()

    async def search(
        self,
        query: str,
        limit: int = 10,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[MemoryEntry]:
        """Search memory entries and memory files."""
        results = self._store.search(query, limit=limit, tags=tags)

        # Also search MEMORY.md content if query matches
        memory_entries = self._search_file_content(
            self._memory_content, "MEMORY.md", query, limit=3,
        )
        results = memory_entries + results

        # Search USER.md if relevant
        user_entries = self._search_file_content(
            self._user_content, "USER.md", query, limit=2,
        )
        results = user_entries + results

        # Deduplicate by ID
        seen: set = set()
        deduped: List[MemoryEntry] = []
        for entry in results:
            if entry.id not in seen:
                seen.add(entry.id)
                deduped.append(entry)

        return deduped[:limit]

    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        if not entry.id:
            entry.id = _generate_id()
        if not entry.timestamp:
            entry.timestamp = _now_iso()

        self._store.add(entry)

        if self._auto_save:
            await self._save_entries()

        return entry.id

    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific entry by ID."""
        return self._store.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        deleted = self._store.remove(entry_id)
        if deleted and self._auto_save:
            await self._save_entries()
        return deleted

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        session_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """List all memory entries."""
        return self._store.list_entries(limit=limit, offset=offset, session_id=session_id)

    # ── MEMORY.md management ──────────────────────────────────────────────

    async def get_memory_content(self) -> str:
        """Return the current MEMORY.md content."""
        return self._memory_content

    async def append_to_memory(self, content: str) -> None:
        """Append content to MEMORY.md."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        entry = f"\n## {timestamp}\n{content}\n"
        self._memory_content += entry
        if self._auto_save:
            await self._write_file(self._memory_file, self._memory_content)

    async def update_memory(self, content: str) -> None:
        """Replace the entire MEMORY.md content."""
        self._memory_content = content
        if self._auto_save:
            await self._write_file(self._memory_file, self._memory_content)

    # ── USER.md management ────────────────────────────────────────────────

    async def get_user_profile(self) -> str:
        """Return the current USER.md content."""
        return self._user_content

    async def update_user_profile(self, content: str) -> None:
        """Replace the entire USER.md content."""
        self._user_content = content
        if self._auto_save:
            await self._write_file(self._user_file, self._user_content)

    async def append_to_user_profile(self, content: str) -> None:
        """Append content to USER.md."""
        self._user_content += f"\n{content}\n"
        if self._auto_save:
            await self._write_file(self._user_file, self._user_content)

    # ── Session search ────────────────────────────────────────────────────

    async def search_sessions(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search across saved session summaries.

        Returns
        -------
        list[dict]
            Matching sessions with ``session_id``, ``title``, ``summary``, and
            ``relevance_score`` keys.
        """
        sessions = await self._list_session_files()
        query_tokens = set(_tokenize_simple(query))

        results: List[tuple[float, Dict[str, Any]]] = []
        for session_data in sessions:
            summary = session_data.get("summary", "")
            summary_tokens = set(_tokenize_simple(summary))
            overlap = len(query_tokens & summary_tokens)
            if overlap > 0:
                session_data["relevance_score"] = overlap
                results.append((float(overlap), session_data))

        results.sort(key=lambda x: x[0], reverse=True)
        return [data for _, data in results[:limit]]

    async def save_session_summary(
        self,
        session_id: str,
        summary: str,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a session summary to disk."""
        session_file = self._sessions_dir / f"{session_id}.json"
        data = {
            "session_id": session_id,
            "title": title or f"Session {session_id[:8]}",
            "summary": summary,
            "timestamp": _now_iso(),
            "metadata": metadata or {},
        }
        await self._write_json_file(session_file, data)

    # ── Auto-summarize ────────────────────────────────────────────────────

    async def auto_summarize(self) -> int:
        """
        Auto-summarize old memory entries.

        Scans the entry store and generates summaries for old entries,
        replacing their content with condensed versions to save context space.

        Returns
        -------
        int
            Number of entries that were summarized.
        """
        if not self._auto_summarize:
            return 0

        entries = self._store.list_entries(limit=1000)
        now = datetime.now(timezone.utc)
        summarized = 0

        for entry in entries:
            try:
                entry_time = datetime.fromisoformat(entry.timestamp)
                age_days = (now - entry_time).days
            except (ValueError, TypeError):
                continue

            if age_days < 7 or len(entry.content) < 200:
                continue

            # Create a summarized version
            summary = self._summarize_entry(entry)
            if summary != entry.content:
                entry.content = summary
                entry.metadata["summarized"] = True
                entry.metadata["original_length"] = len(entry.content)
                summarized += 1

        if summarized > 0 and self._auto_save:
            await self._save_entries()

        return summarized

    @staticmethod
    def _summarize_entry(entry: MemoryEntry) -> str:
        """Summarize a single entry by extracting key sentences."""
        content = entry.content.strip()
        if len(content) <= 300:
            return content

        # Extract the first 2-3 sentences as a summary
        sentences = re.split(r"(?<=[.!?])\s+", content)
        summary_sentences = sentences[:3]
        summary = " ".join(summary_sentences)

        if len(summary) > 300:
            summary = summary[:300] + "..."
        return summary

    # ── Context for prompt ────────────────────────────────────────────────

    async def get_context_for_prompt(
        self,
        query: str,
        max_tokens: int = 4000,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Build a context string from memory files and entries.

        Includes relevant MEMORY.md content, USER.md profile, and searched
        memory entries within the token budget.
        """
        parts: List[str] = []
        max_chars = max_tokens * 4
        used_chars = 0

        # Include MEMORY.md if non-empty
        if self._memory_content.strip():
            memory_block = f"<agent_memory>\n{self._memory_content[:2000]}\n</agent_memory>"
            if used_chars + len(memory_block) <= max_chars:
                parts.append(memory_block)
                used_chars += len(memory_block)

        # Include USER.md if non-empty
        if self._user_content.strip():
            user_block = f"<user_profile>\n{self._user_content[:1000]}\n</user_profile>"
            if used_chars + len(user_block) <= max_chars:
                parts.append(user_block)
                used_chars += len(user_block)

        # Search for relevant entries
        remaining_chars = max_chars - used_chars
        if remaining_chars > 200:
            search_limit = min(10, remaining_chars // 200)
            entries = await self.search(query, limit=search_limit, session_id=session_id)
            entry_parts: List[str] = []
            entry_used = 0
            for entry in entries:
                block = f"[{entry.role}] {entry.content[:300]}"
                if entry_used + len(block) <= remaining_chars:
                    entry_parts.append(block)
                    entry_used += len(block)
            if entry_parts:
                parts.append(
                    "<memory_entries>\n"
                    + "\n---\n".join(entry_parts)
                    + "\n</memory_entries>"
                )

        return "\n\n".join(parts)

    # ── Diagnostics ───────────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        """Check health of the built-in memory provider."""
        return {
            "status": "healthy",
            "base_dir": str(self._base_dir),
            "entry_count": self._store.count,
            "memory_file_size": len(self._memory_content),
            "user_file_size": len(self._user_content),
            "auto_save": self._auto_save,
            "auto_summarize": self._auto_summarize,
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        session_files = list(self._sessions_dir.glob("*.json"))
        return {
            "total_entries": self._store.count,
            "memory_file_chars": len(self._memory_content),
            "user_file_chars": len(self._user_content),
            "session_count": len(session_files),
            "base_dir": str(self._base_dir),
            "sessions_dir": str(self._sessions_dir),
        }

    # ── File I/O helpers ─────────────────────────────────────────────────

    async def _read_file(self, path: Path) -> str:
        """Read a file's content, returning empty string on error."""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: path.read_text(encoding="utf-8", errors="replace"),
            )
        except Exception:
            return ""

    async def _write_file(self, path: Path, content: str) -> None:
        """Write content to a file, creating directories as needed."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: path.write_text(content, encoding="utf-8"),
            )
        except Exception as e:
            logger.warning("Failed to write %s: %s", path, e)

    async def _write_json_file(self, path: Path, data: Dict[str, Any]) -> None:
        """Write data as JSON to a file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: path.write_text(
                    json.dumps(data, indent=2, default=str, ensure_ascii=False),
                    encoding="utf-8",
                ),
            )
        except Exception as e:
            logger.warning("Failed to write JSON to %s: %s", path, e)

    async def _load_entries(self) -> None:
        """Load the entry store from disk."""
        if not self._entries_file.exists():
            return
        try:
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(
                None, lambda: self._entries_file.read_text(encoding="utf-8"),
            )
            data = json.loads(content)
            self._store.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load entries: %s", e)

    async def _save_entries(self) -> None:
        """Persist the entry store to disk."""
        try:
            loop = asyncio.get_running_loop()
            data = self._store.to_dict()
            content = json.dumps(data, indent=2, default=str, ensure_ascii=False)
            await loop.run_in_executor(
                None,
                lambda: self._entries_file.write_text(content, encoding="utf-8"),
            )
        except Exception as e:
            logger.warning("Failed to save entries: %s", e)

    async def _persist(self) -> None:
        """Persist all data to disk."""
        await asyncio.gather(
            self._write_file(self._memory_file, self._memory_content),
            self._write_file(self._user_file, self._user_content),
            self._save_entries(),
        )

    async def _list_session_files(self) -> List[Dict[str, Any]]:
        """List and load all session summary files."""
        sessions: List[Dict[str, Any]] = []
        try:
            for path in self._sessions_dir.glob("*.json"):
                try:
                    loop = asyncio.get_running_loop()
                    content = await loop.run_in_executor(
                        None, lambda p=path: p.read_text(encoding="utf-8"),
                    )
                    data = json.loads(content)
                    sessions.append(data)
                except Exception:
                    pass
        except Exception:
            pass
        return sessions

    def _search_file_content(
        self,
        content: str,
        source: str,
        query: str,
        limit: int,
    ) -> List[MemoryEntry]:
        """Search within a file's content for relevant sections."""
        if not content.strip() or not query.strip():
            return []

        query_tokens = set(_tokenize_simple(query))
        lines = content.split("\n")

        # Find paragraphs that match query tokens
        paragraphs: List[str] = []
        current_para: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("---"):
                if current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
            elif stripped:
                current_para.append(stripped)
        if current_para:
            paragraphs.append(" ".join(current_para))

        scored: List[tuple[float, str]] = []
        for para in paragraphs:
            para_tokens = set(_tokenize_simple(para))
            overlap = len(query_tokens & para_tokens)
            if overlap > 0:
                scored.append((float(overlap), para))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            MemoryEntry(
                id=f"{source}_{i}",
                content=para[:500],
                role="system",
                source=source,
                importance=0.3,
            )
            for i, (_, para) in enumerate(scored[:limit])
        ]
