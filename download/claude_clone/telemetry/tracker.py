"""
Telemetry tracker — lightweight usage metrics collection.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricEvent:
    """A single telemetry event."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TelemetryTracker:
    """
    Lightweight in-process telemetry tracker.
    
    Tracks usage metrics like LLM calls, token consumption, tool usage,
    and custom events. Data is kept in memory and can be exported.
    
    Args:
        enabled: Whether telemetry collection is active.
        max_events: Maximum events to keep in memory.
    """
    
    def __init__(self, enabled: bool = True, max_events: int = 10000):
        self.enabled = enabled
        self._max_events = max_events
        self._events: List[MetricEvent] = []
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._start_time = time.time()
    
    def track(self, name: str, value: float = 1.0, metadata: Optional[Dict] = None) -> None:
        """Record a telemetry event."""
        if not self.enabled:
            return
        event = MetricEvent(name=name, value=value, metadata=metadata or {})
        self._events.append(event)
        self._counters[name] += value
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
    
    def increment(self, name: str, amount: float = 1.0) -> None:
        """Increment a named counter."""
        if not self.enabled:
            return
        self._counters[name] += amount
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        if not self.enabled:
            return
        self._gauges[name] = value
    
    def get_counter(self, name: str) -> float:
        """Get a counter value."""
        return self._counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> float:
        """Get a gauge value."""
        return self._gauges.get(name, 0.0)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked metrics."""
        uptime = time.time() - self._start_time
        return {
            "uptime_seconds": uptime,
            "events_recorded": len(self._events),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
        }
    
    def get_events(
        self,
        name: Optional[str] = None,
        limit: int = 100,
    ) -> List[MetricEvent]:
        """Get recent events, optionally filtered by name."""
        events = self._events
        if name:
            events = [e for e in events if e.name == name]
        return list(reversed(events[-limit:]))
    
    def reset(self) -> None:
        """Clear all tracked data."""
        self._events.clear()
        self._counters.clear()
        self._gauges.clear()
        self._start_time = time.time()
    
    @property
    def uptime(self) -> float:
        return time.time() - self._start_time
    
    def __repr__(self) -> str:
        return (
            f"TelemetryTracker(enabled={self.enabled}, "
            f"events={len(self._events)}, counters={len(self._counters)})"
        )


# Global singleton
tracker = TelemetryTracker()
