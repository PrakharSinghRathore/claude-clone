"""
Flow context — state management across flow steps.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FlowContext:
    """
    Manages state shared across flow steps.

    Provides a dictionary-like interface for storing and retrieving
    data that flows between steps. Supports immutable snapshots
    and type-safe access.

    Attributes:
        data: The underlying state dictionary.
        step_count: Number of steps executed so far.
        current_step: Name of the currently executing step.
    """

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        self.data: Dict[str, Any] = initial_data or {}
        self.step_count: int = 0
        self.current_step: Optional[str] = None
        self._history: list[Dict[str, Any]] = []

    def get(self, key: str, default: T = None) -> Any:
        """Get a value from the context."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""
        self.data[key] = value

    def update(self, data: Dict[str, Any]) -> None:
        """Merge a dictionary into the context."""
        self.data.update(data)

    def has(self, key: str) -> bool:
        """Check if a key exists in the context."""
        return key in self.data

    def delete(self, key: str) -> bool:
        """Delete a key from the context. Returns True if deleted."""
        if key in self.data:
            del self.data[key]
            return True
        return False

    def snapshot(self) -> Dict[str, Any]:
        """Return an immutable copy of the current state."""
        return copy.deepcopy(self.data)

    def restore(self, state: Dict[str, Any]) -> None:
        """Restore the context to a previous snapshot."""
        self.data = copy.deepcopy(state)

    def save_checkpoint(self) -> None:
        """Save the current state as a checkpoint."""
        self._history.append(self.snapshot())

    def rollback(self) -> bool:
        """Roll back to the last checkpoint. Returns True if rolled back."""
        if self._history:
            self.data = self._history.pop()
            return True
        return False

    def reset(self) -> None:
        """Clear all context data and history."""
        self.data.clear()
        self.step_count = 0
        self.current_step = None
        self._history.clear()

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __repr__(self) -> str:
        keys = list(self.data.keys())[:5]
        more = f"... +{len(self.data) - 5} more" if len(self.data) > 5 else ""
        return f"FlowContext(steps={self.step_count}, keys={keys}{more})"
