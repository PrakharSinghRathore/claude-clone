"""
Gateway status reporting and health checks.

Provides health checks for all platforms, uptime tracking, message
statistics, error reporting, and a JSON status endpoint.

Usage::

    status = GatewayStatus()
    status.platform_healthy("telegram")
    status.record_message_received("telegram")
    report = status.get_report()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger("hermes.gateway.status")


# ──────────────────────────────────────────────────────────────────────────────
# Platform Health
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PlatformHealth:
    """Health status for a single platform."""

    platform: str
    is_connected: bool = False
    last_check: Optional[str] = None
    last_connected: Optional[str] = None
    last_disconnected: Optional[str] = None
    consecutive_failures: int = 0
    total_reconnects: int = 0
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "status": "connected" if self.is_connected else "disconnected",
            "is_connected": self.is_connected,
            "last_check": self.last_check,
            "last_connected": self.last_connected,
            "last_disconnected": self.last_disconnected,
            "consecutive_failures": self.consecutive_failures,
            "total_reconnects": self.total_reconnects,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Error Record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ErrorRecord:
    """A recorded error event."""

    platform: str
    error_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Gateway Status
# ──────────────────────────────────────────────────────────────────────────────

class GatewayStatus:
    """
    Gateway health monitoring and status reporting.

    Features:
    - Per-platform health tracking
    - Uptime and downtime tracking
    - Message statistics (sent, received, failed)
    - Error recording and reporting
    - Periodic health check execution
    - JSON status endpoint data
    """

    def __init__(
        self,
        health_check_interval: int = 60,
        max_error_history: int = 1000,
        health_check_callback: Optional[Callable] = None,
    ):
        self._started_at = time.time()
        self._started_at_iso = datetime.now(timezone.utc).isoformat()
        self._health_check_interval = health_check_interval
        self._health_check_callback = health_check_callback
        self._max_error_history = max_error_history

        # Platform health
        self._platform_health: Dict[str, PlatformHealth] = {}

        # Message statistics
        self._messages_received: Dict[str, int] = defaultdict(int)
        self._messages_sent: Dict[str, int] = defaultdict(int)
        self._messages_failed: Dict[str, int] = defaultdict(int)
        self._total_bytes_sent: Dict[str, int] = defaultdict(int)

        # Error history
        self._errors: List[ErrorRecord] = []
        self._error_counts: Dict[str, int] = defaultdict(int)

        # Session statistics
        self._active_sessions: int = 0
        self._total_sessions_created: int = 0
        self._sessions_reset: int = 0

        # Streaming statistics
        self._streams_started: int = 0
        self._streams_completed: int = 0
        self._streams_cancelled: int = 0

        # Health check task
        self._health_task: Optional[asyncio.Task] = None

    # ── Platform Health ───────────────────────────────────────────────────

    def register_platform(self, name: str) -> None:
        """Register a platform for health tracking."""
        if name not in self._platform_health:
            self._platform_health[name] = PlatformHealth(platform=name)

    def platform_connected(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark a platform as connected."""
        health = self._platform_health.setdefault(name, PlatformHealth(platform=name))
        health.is_connected = True
        health.last_connected = datetime.now(timezone.utc).isoformat()
        health.last_check = datetime.now(timezone.utc).isoformat()
        health.consecutive_failures = 0
        health.last_error = None
        if metadata:
            health.metadata.update(metadata)

    def platform_disconnected(self, name: str, error: Optional[str] = None) -> None:
        """Mark a platform as disconnected."""
        health = self._platform_health.setdefault(name, PlatformHealth(platform=name))
        was_connected = health.is_connected
        health.is_connected = False
        health.last_disconnected = datetime.now(timezone.utc).isoformat()
        health.last_check = datetime.now(timezone.utc).isoformat()

        if error:
            health.last_error = error
            health.consecutive_failures += 1
            self.record_error(name, "disconnection", error)

        if was_connected:
            health.total_reconnects += 1

    def platform_healthy(self, name: str, latency_ms: Optional[float] = None) -> None:
        """Record a successful health check for a platform."""
        health = self._platform_health.setdefault(name, PlatformHealth(platform=name))
        health.last_check = datetime.now(timezone.utc).isoformat()
        health.is_connected = True
        health.consecutive_failures = 0
        if latency_ms is not None:
            health.latency_ms = latency_ms

    def platform_unhealthy(self, name: str, error: str) -> None:
        """Record a failed health check for a platform."""
        health = self._platform_health.setdefault(name, PlatformHealth(platform=name))
        health.last_check = datetime.now(timezone.utc).isoformat()
        health.consecutive_failures += 1
        health.last_error = error

    def get_platform_health(self, name: str) -> Optional[PlatformHealth]:
        """Get health status for a specific platform."""
        return self._platform_health.get(name)

    # ── Message Statistics ────────────────────────────────────────────────

    def record_message_received(self, platform: str) -> None:
        """Record an incoming message."""
        self._messages_received[platform] += 1

    def record_message_sent(self, platform: str, bytes_sent: int = 0) -> None:
        """Record an outgoing message."""
        self._messages_sent[platform] += 1
        self._total_bytes_sent[platform] += bytes_sent

    def record_message_failed(self, platform: str) -> None:
        """Record a failed message delivery."""
        self._messages_failed[platform] += 1

    # ── Session Statistics ────────────────────────────────────────────────

    def set_active_sessions(self, count: int) -> None:
        """Set the current number of active sessions."""
        self._active_sessions = count

    def record_session_created(self) -> None:
        """Record a new session creation."""
        self._total_sessions_created += 1

    def record_session_reset(self) -> None:
        """Record a session reset."""
        self._sessions_reset += 1

    # ── Streaming Statistics ──────────────────────────────────────────────

    def record_stream_started(self) -> None:
        """Record a stream start."""
        self._streams_started += 1

    def record_stream_completed(self) -> None:
        """Record a stream completion."""
        self._streams_completed += 1

    def record_stream_cancelled(self) -> None:
        """Record a stream cancellation."""
        self._streams_cancelled += 1

    # ── Error Recording ───────────────────────────────────────────────────

    def record_error(
        self,
        platform: str,
        error_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an error event."""
        record = ErrorRecord(
            platform=platform,
            error_type=error_type,
            message=message,
            metadata=metadata or {},
        )
        self._errors.append(record)
        self._error_counts[f"{platform}:{error_type}"] += 1

        # Trim error history
        if len(self._errors) > self._max_error_history:
            self._errors = self._errors[-self._max_error_history:]

    def get_recent_errors(
        self, platform: Optional[str] = None, limit: int = 50,
    ) -> List[ErrorRecord]:
        """Get recent errors, optionally filtered by platform."""
        errors = self._errors
        if platform:
            errors = [e for e in errors if e.platform == platform]
        return errors[-limit:]

    # ── Health Check Loop ─────────────────────────────────────────────────

    async def start_health_checks(self) -> None:
        """Start the periodic health check loop."""
        if self._health_task is not None:
            return

        async def _loop():
            try:
                while True:
                    await self._run_health_checks()
                    await asyncio.sleep(self._health_check_interval)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("Health check loop error: %s", e)

        self._health_task = asyncio.create_task(_loop())

    async def stop_health_checks(self) -> None:
        """Stop the periodic health check loop."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

    async def _run_health_checks(self) -> None:
        """Run health checks on all registered platforms."""
        if self._health_check_callback:
            try:
                results = await self._health_check_callback()
                if isinstance(results, dict):
                    for platform, is_healthy in results.items():
                        if is_healthy:
                            self.platform_healthy(platform)
                        else:
                            self.platform_unhealthy(platform, "Health check failed")
            except Exception as e:
                logger.error("Health check callback failed: %s", e)

    # ── Status Report ─────────────────────────────────────────────────────

    def get_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive status report.

        Returns a dictionary suitable for JSON serialization as a
        status endpoint response.
        """
        now = time.time()
        uptime_seconds = now - self._started_at

        # Calculate uptime percentage
        total_uptime = uptime_seconds
        platform_downtime = 0.0
        for health in self._platform_health.values():
            if health.last_disconnected:
                try:
                    disc_time = datetime.fromisoformat(health.last_disconnected).timestamp()
                    platform_downtime += max(0, now - disc_time)
                except (ValueError, TypeError):
                    pass

        uptime_pct = min(100.0, max(0.0, (
            (total_uptime - platform_downtime) / max(total_uptime, 1)
        ) * 100))

        # Aggregate message stats
        total_received = sum(self._messages_received.values())
        total_sent = sum(self._messages_sent.values())
        total_failed = sum(self._messages_failed.values())
        total_bytes = sum(self._total_bytes_sent.values())

        # Error summary
        error_summary: Dict[str, int] = {}
        for key, count in self._error_counts.most_common(10):
            error_summary[key] = count

        return {
            "status": "healthy" if self._is_healthy() else "degraded",
            "uptime_seconds": round(uptime_seconds, 1),
            "uptime_percentage": round(uptime_pct, 2),
            "started_at": self._started_at_iso,
            "current_time": datetime.now(timezone.utc).isoformat(),

            "platforms": {
                name: health.to_dict()
                for name, health in self._platform_health.items()
            },

            "messages": {
                "total_received": total_received,
                "total_sent": total_sent,
                "total_failed": total_failed,
                "total_bytes_sent": total_bytes,
                "by_platform": {
                    name: {
                        "received": self._messages_received.get(name, 0),
                        "sent": self._messages_sent.get(name, 0),
                        "failed": self._messages_failed.get(name, 0),
                        "bytes_sent": self._total_bytes_sent.get(name, 0),
                    }
                    for name in set(list(self._messages_received.keys()) + list(self._messages_sent.keys()))
                },
            },

            "sessions": {
                "active": self._active_sessions,
                "total_created": self._total_sessions_created,
                "total_reset": self._sessions_reset,
            },

            "streaming": {
                "started": self._streams_started,
                "completed": self._streams_completed,
                "cancelled": self._streams_cancelled,
                "active": self._streams_started - self._streams_completed - self._streams_cancelled,
            },

            "errors": {
                "total": len(self._errors),
                "recent_count": len(self._errors[-50:]),
                "summary": error_summary,
            },
        }

    def _is_healthy(self) -> bool:
        """Determine overall gateway health."""
        if not self._platform_health:
            return True

        connected = sum(1 for h in self._platform_health.values() if h.is_connected)
        total = len(self._platform_health)

        # Healthy if at least half the platforms are connected
        # and no platform has excessive consecutive failures
        if total > 0 and connected / total < 0.5:
            return False

        for health in self._platform_health.values():
            if health.consecutive_failures > 5:
                return False

        return True

    def get_json_status(self) -> str:
        """Return the status report as a JSON string."""
        import json
        return json.dumps(self.get_report(), indent=2, default=str)
