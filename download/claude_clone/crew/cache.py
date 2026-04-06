"""
Tool result cache with TTL support.

Provides an in-memory caching layer for tool execution results. Cache keys
are derived from the tool name and a hash of the arguments, making lookups
deterministic and fast.
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheHandler:
    """
    In-memory cache for tool execution results with TTL support.

    The cache stores results keyed by a SHA-256 hash of the tool name and
    its arguments (JSON-serialised). Each entry has an optional time-to-live
    (TTL) after which it is considered stale and evicted on next access.

    Args:
        ttl: Default time-to-live for cache entries in seconds. Entries
            without an explicit TTL use this value. Set to ``0`` for
            no expiry.

    Example::

        cache = CacheHandler(ttl=300)

        # Check cache before executing
        result = cache.get("search", {"query": "python"})
        if result is None:
            result = do_search("python")
            cache.set("search", {"query": "python"}, result)
    """

    def __init__(self, ttl: int = 300) -> None:
        self.ttl = ttl
        self._store: Dict[str, Dict[str, Any]] = {}

    def _make_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Build a deterministic cache key from a tool name and its arguments.

        The key is the hex digest of ``SHA-256(tool_name + json(args))``.
        Dictionaries are sorted before serialisation so key order does not
        matter.
        """
        serialised = json.dumps(args, sort_keys=True, default=str)
        payload = f"{tool_name}:{serialised}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """
        Retrieve a cached result for a tool invocation.

        Args:
            tool_name: Name of the tool that produced the result.
            args: Dictionary of arguments passed to the tool.

        Returns:
            The cached result string, or ``None`` if not found or expired.
        """
        key = self._make_key(tool_name, args)
        entry = self._store.get(key)
        if entry is None:
            return None

        # Check TTL
        if entry.get("ttl", self.ttl) > 0:
            elapsed = time.monotonic() - entry.get("created_at", 0)
            if elapsed > entry["ttl"]:
                logger.debug("Cache expired for tool %s (key=%s)", tool_name, key[:12])
                del self._store[key]
                return None

        logger.debug("Cache hit for tool %s (key=%s)", tool_name, key[:12])
        return entry["result"]

    def set(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: str,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a tool result in the cache.

        Args:
            tool_name: Name of the tool that produced the result.
            args: Dictionary of arguments passed to the tool.
            result: The result string to cache.
            ttl: Optional per-entry TTL in seconds. Falls back to the
                instance default ``self.ttl``.
        """
        key = self._make_key(tool_name, args)
        self._store[key] = {
            "result": result,
            "created_at": time.monotonic(),
            "ttl": ttl if ttl is not None else self.ttl,
        }
        logger.debug("Cached result for tool %s (key=%s, ttl=%ss)", tool_name, key[:12], ttl or self.ttl)

    def clear(self) -> None:
        """Remove all expired entries from the cache."""
        now = time.monotonic()
        expired_keys = [
            key for key, entry in self._store.items()
            if entry.get("ttl", self.ttl) > 0 and (now - entry.get("created_at", 0)) > entry["ttl"]
        ]
        for key in expired_keys:
            del self._store[key]
        if expired_keys:
            logger.debug("Evicted %d expired cache entries", len(expired_keys))

    def reset(self) -> None:
        """Remove all entries from the cache, regardless of TTL."""
        self._store.clear()
        logger.debug("Cache reset — all entries cleared")

    @property
    def size(self) -> int:
        """Return the current number of cached entries (including expired)."""
        return len(self._store)

    def __len__(self) -> int:
        return self.size

    def __contains__(self, item: str) -> bool:
        # Convenience: allow ``tool_name in cache`` checks
        return any(key.startswith(item) for key in self._store)
