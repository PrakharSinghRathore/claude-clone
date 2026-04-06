"""
Base persistence backend for flows.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BasePersistenceBackend(ABC):
    """
    Abstract base class for flow persistence backends.

    Subclasses must implement save and load methods for
    persisting flow execution state.
    """

    @abstractmethod
    def save(self, flow_id: str, state: Dict[str, Any]) -> None:
        """Save flow execution state."""
        ...

    @abstractmethod
    def load(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Load flow execution state. Returns None if not found."""
        ...

    @abstractmethod
    def delete(self, flow_id: str) -> bool:
        """Delete flow execution state. Returns True if deleted."""
        ...

    @abstractmethod
    def list_flows(self) -> list[Dict[str, Any]]:
        """List all persisted flows with metadata."""
        ...
