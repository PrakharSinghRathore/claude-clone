"""
Atlas Security — Security Audit System.

Provides comprehensive security event logging, querying, and export
capabilities. Maintains an immutable audit trail of all security-relevant
operations including tool calls, file access, and network requests.

Inspired by OpenClaw's security audit architecture.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SecuritySeverity(Enum):
    """Severity levels for security audit events."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_string(cls, value: str) -> "SecuritySeverity":
        """Parse severity from string, case-insensitive.

        Args:
            value: String severity identifier.

        Returns:
            Matching SecuritySeverity enum member.

        Raises:
            ValueError: If no matching severity is found.
        """
        normalized = value.lower().strip()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unknown severity: {value!r}. "
            f"Valid: {[m.value for m in cls]}"
        )


class AuditEventType(Enum):
    """Types of security-relevant events."""

    # General
    SYSTEM = "system"
    AUTH = "auth"
    CONFIG_CHANGE = "config_change"

    # Tool access
    TOOL_CALL = "tool_call"
    TOOL_DENIED = "tool_denied"
    TOOL_ERROR = "tool_error"

    # File access
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    FILE_DENIED = "file_denied"

    # Network
    NETWORK_REQUEST = "network_request"
    NETWORK_DENIED = "network_denied"
    NETWORK_ERROR = "network_error"

    # Channel
    CHANNEL_CONNECT = "channel_connect"
    CHANNEL_DISCONNECT = "channel_disconnect"
    CHANNEL_MESSAGE = "channel_message"

    # Sandbox
    SANDBOX_EXEC = "sandbox_exec"
    SANDBOX_VIOLATION = "sandbox_violation"

    # Pairing
    PAIRING_REQUEST = "pairing_request"
    PAIRING_SUCCESS = "pairing_success"
    PAIRING_FAILURE = "pairing_failure"
    PAIRING_EXPIRED = "pairing_expired"

    # Secrets
    SECRET_ACCESS = "secret_access"
    SECRET_MODIFY = "secret_modify"

    # Policy
    POLICY_VIOLATION = "policy_violation"
    POLICY_UPDATE = "policy_update"

    # Custom
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SecurityAuditEvent:
    """A single security audit event record.

    Attributes:
        timestamp: When the event occurred (UTC).
        event_type: The type/category of the event.
        severity: Severity level of the event.
        source: What triggered this event (module, function, etc.).
        description: Human-readable description of the event.
        metadata: Additional structured data about the event.
        user: User or agent that triggered the event.
        session_id: Session identifier.
        id: Unique event identifier (auto-generated).
    """

    id: str = field(default="")
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    event_type: AuditEventType = AuditEventType.SYSTEM
    severity: SecuritySeverity = SecuritySeverity.INFO
    source: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    user: str = ""
    session_id: str = ""

    def __post_init__(self) -> None:
        """Auto-generate ID if not provided."""
        if not self.id:
            import hashlib
            raw = (
                f"{self.timestamp.isoformat()}:{self.event_type.value}:"
                f"{self.source}:{self.description}"
            )
            self.id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "source": self.source,
            "description": self.description,
            "metadata": self.metadata,
            "user": self.user,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityAuditEvent":
        """Deserialize event from dictionary."""
        data = data.copy()
        if "event_type" in data and isinstance(data["event_type"], str):
            data["event_type"] = AuditEventType(data["event_type"])
        if "severity" in data and isinstance(data["severity"], str):
            data["severity"] = SecuritySeverity(data["severity"])
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class AuditFilter:
    """Filter criteria for querying the audit log.

    Attributes:
        event_type: Filter by event type.
        severity: Filter by minimum severity level.
        source: Filter by source (substring match).
        user: Filter by user identifier.
        session_id: Filter by session ID.
        start_time: Include events from this time onwards.
        end_time: Include events up to this time.
        description_contains: Filter by description substring.
        limit: Maximum number of results to return.
        offset: Number of results to skip (for pagination).
    """

    event_type: Optional[AuditEventType] = None
    severity: Optional[SecuritySeverity] = None
    source: str = ""
    user: str = ""
    session_id: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    description_contains: str = ""
    limit: int = 100
    offset: int = 0

    def matches(self, event: SecurityAuditEvent) -> bool:
        """Check if an event matches this filter.

        Args:
            event: The event to evaluate.

        Returns:
            True if the event matches all non-empty filter criteria.
        """
        if self.event_type and event.event_type != self.event_type:
            return False
        if self.severity:
            severity_order = [
                SecuritySeverity.INFO,
                SecuritySeverity.LOW,
                SecuritySeverity.MEDIUM,
                SecuritySeverity.HIGH,
                SecuritySeverity.CRITICAL,
            ]
            try:
                if severity_order.index(event.severity) < severity_order.index(self.severity):
                    return False
            except ValueError:
                return False
        if self.source and self.source.lower() not in event.source.lower():
            return False
        if self.user and self.user.lower() not in event.user.lower():
            return False
        if self.session_id and self.session_id != event.session_id:
            return False
        if self.start_time and event.timestamp < self.start_time:
            return False
        if self.end_time and event.timestamp > self.end_time:
            return False
        if self.description_contains and self.description_contains.lower() not in event.description.lower():
            return False
        return True


# ---------------------------------------------------------------------------
# Security Auditor
# ---------------------------------------------------------------------------

class SecurityAuditor:
    """Security audit logging and query system.

    Provides thread-safe, persistent audit logging for all security-relevant
    operations. Supports filtering, export to JSON/CSV, log rotation, and
    configurable retention policies.

    Usage::

        auditor = SecurityAuditor("/path/to/audit.log")

        # Log events
        auditor.audit_tool_call("bash", {"command": "ls"}, user="admin")
        auditor.audit_file_access("/tmp/data.csv", "read", user="agent_1")
        auditor.audit_network_request("https://api.example.com", "GET", user="agent_1")

        # Query events
        events = auditor.get_audit_log(
            AuditFilter(severity=SecuritySeverity.HIGH, limit=50)
        )

        # Export
        auditor.export_log("json", "/path/to/export.json")
    """

    DEFAULT_MAX_EVENTS = 10000
    DEFAULT_RETENTION_DAYS = 90

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        max_events: int = DEFAULT_MAX_EVENTS,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        auto_flush: bool = True,
        flush_interval: int = 30,
    ) -> None:
        """Initialize the security auditor.

        Args:
            persistence_path: Path to the JSON audit log file.
            max_events: Maximum events to keep in memory before trimming.
            retention_days: Days to retain events before pruning.
            auto_flush: Whether to automatically flush to disk.
            flush_interval: Seconds between automatic flushes.
        """
        self._events: List[SecurityAuditEvent] = []
        self._max_events = max_events
        self._retention_days = retention_days
        self._auto_flush = auto_flush
        self._flush_interval = flush_interval
        self._lock = threading.Lock()
        self._stats = {
            "total_events": 0,
            "by_severity": {},
            "by_type": {},
            "flush_count": 0,
            "last_flush": None,
        }

        if persistence_path:
            self._persistence_path = Path(persistence_path)
        else:
            self._persistence_path = Path("security_audit.json")

        self._load()
        self._prune_retention()

        logger.info(
            "SecurityAuditor initialized (%d events, max=%d, retention=%dd)",
            len(self._events), max_events, retention_days,
        )

    # ------------------------------------------------------------------
    # Event Logging
    # ------------------------------------------------------------------

    def log_event(self, event: SecurityAuditEvent) -> str:
        """Record a security audit event.

        Thread-safe. Automatically updates statistics and triggers
        auto-flush if enabled.

        Args:
            event: The audit event to record.

        Returns:
            The event ID.
        """
        with self._lock:
            self._events.append(event)
            self._update_stats(event)

            # Trim if over max
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

        if self._auto_flush:
            self._maybe_flush()

        logger.debug(
            "Audit: [%s] %s — %s",
            event.severity.value.upper(),
            event.event_type.value,
            event.description,
        )
        return event.id

    def log(
        self,
        event_type: AuditEventType,
        severity: SecuritySeverity,
        description: str,
        source: str = "",
        user: str = "",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience method to log a security event directly.

        Args:
            event_type: Type of the event.
            severity: Severity level.
            description: Event description.
            source: Event source.
            user: User/agent identifier.
            session_id: Session identifier.
            metadata: Additional metadata.

        Returns:
            The event ID.
        """
        event = SecurityAuditEvent(
            event_type=event_type,
            severity=severity,
            description=description,
            source=source,
            user=user,
            session_id=session_id,
            metadata=metadata or {},
        )
        return self.log_event(event)

    # ------------------------------------------------------------------
    # Specialized Audit Methods
    # ------------------------------------------------------------------

    def audit_tool_call(
        self,
        tool_name: str,
        tool_input: Any,
        user: str = "",
        session_id: str = "",
        allowed: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Audit a tool usage event.

        Args:
            tool_name: Name of the tool being called.
            tool_input: Input/parameters passed to the tool.
            user: User or agent invoking the tool.
            session_id: Session identifier.
            allowed: Whether the call was permitted.
            metadata: Additional metadata.

        Returns:
            The event ID.
        """
        event_type = (
            AuditEventType.TOOL_DENIED if not allowed
            else AuditEventType.TOOL_CALL
        )
        severity = (
            SecuritySeverity.HIGH if not allowed
            else SecuritySeverity.INFO
        )

        # Sanitize tool input for logging (mask secrets)
        sanitized_input = self._sanitize_input(tool_input)

        extra = metadata or {}
        extra["tool_name"] = tool_name
        extra["allowed"] = allowed
        extra["input"] = sanitized_input

        return self.log(
            event_type=event_type,
            severity=severity,
            description=(
                f"Tool call: {tool_name} "
                f"({'denied' if not allowed else 'executed'})"
            ),
            source="tool_executor",
            user=user,
            session_id=session_id,
            metadata=extra,
        )

    def audit_file_access(
        self,
        path: str,
        operation: str,
        user: str = "",
        session_id: str = "",
        allowed: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Audit a file system access event.

        Args:
            path: File or directory path being accessed.
            operation: Operation type (read, write, delete, etc.).
            user: User or agent performing the access.
            session_id: Session identifier.
            allowed: Whether the access was permitted.
            metadata: Additional metadata.

        Returns:
            The event ID.
        """
        if not allowed:
            event_type = AuditEventType.FILE_DENIED
            severity = SecuritySeverity.HIGH
        elif operation == "delete":
            event_type = AuditEventType.FILE_DELETE
            severity = SecuritySeverity.MEDIUM
        elif operation == "write":
            event_type = AuditEventType.FILE_WRITE
            severity = SecuritySeverity.LOW
        else:
            event_type = AuditEventType.FILE_READ
            severity = SecuritySeverity.INFO

        extra = metadata or {}
        extra["path"] = path
        extra["operation"] = operation
        extra["allowed"] = allowed

        return self.log(
            event_type=event_type,
            severity=severity,
            description=(
                f"File {operation}: {path} "
                f"({'denied' if not allowed else 'allowed'})"
            ),
            source="file_system",
            user=user,
            session_id=session_id,
            metadata=extra,
        )

    def audit_network_request(
        self,
        url: str,
        method: str,
        user: str = "",
        session_id: str = "",
        allowed: bool = True,
        status_code: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Audit a network access event.

        Args:
            url: The URL being accessed.
            method: HTTP method (GET, POST, etc.).
            user: User or agent making the request.
            session_id: Session identifier.
            allowed: Whether the request was permitted.
            status_code: HTTP response status code (if available).
            metadata: Additional metadata.

        Returns:
            The event ID.
        """
        if not allowed:
            event_type = AuditEventType.NETWORK_DENIED
            severity = SecuritySeverity.HIGH
        elif status_code and status_code >= 400:
            event_type = AuditEventType.NETWORK_ERROR
            severity = SecuritySeverity.LOW
        else:
            event_type = AuditEventType.NETWORK_REQUEST
            severity = SecuritySeverity.INFO

        extra = metadata or {}
        extra["url"] = url
        extra["method"] = method
        extra["allowed"] = allowed
        if status_code:
            extra["status_code"] = status_code

        return self.log(
            event_type=event_type,
            severity=severity,
            description=(
                f"Network {method} {url} "
                f"({'denied' if not allowed else 'allowed'})"
                + (f" — status {status_code}" if status_code else "")
            ),
            source="network",
            user=user,
            session_id=session_id,
            metadata=extra,
        )

    def audit_policy_violation(
        self,
        policy: str,
        reason: str,
        user: str = "",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Audit a security policy violation.

        Args:
            policy: Name of the violated policy.
            reason: Description of the violation.
            user: User or agent involved.
            session_id: Session identifier.
            metadata: Additional metadata.

        Returns:
            The event ID.
        """
        extra = metadata or {}
        extra["policy"] = policy

        return self.log(
            event_type=AuditEventType.POLICY_VIOLATION,
            severity=SecuritySeverity.HIGH,
            description=f"Policy violation: {policy} — {reason}",
            source="policy_engine",
            user=user,
            session_id=session_id,
            metadata=extra,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_audit_log(
        self,
        filters: Optional[AuditFilter] = None,
    ) -> List[SecurityAuditEvent]:
        """Query the audit log with optional filters.

        Args:
            filters: Filter criteria to apply.

        Returns:
            List of matching events.
        """
        with self._lock:
            events = list(self._events)

        if filters is None:
            events = events[-filters.limit:] if filters else events
            return events

        # Apply filters
        matched = [e for e in events if filters.matches(e)]

        # Apply pagination
        if filters.offset:
            matched = matched[filters.offset:]
        if filters.limit < len(matched):
            matched = matched[:filters.limit]

        return matched

    def get_audit_log_dict(
        self,
        filters: Optional[AuditFilter] = None,
    ) -> List[Dict[str, Any]]:
        """Query audit log and return as list of dictionaries.

        Args:
            filters: Filter criteria.

        Returns:
            List of event dictionaries.
        """
        events = self.get_audit_log(filters)
        return [e.to_dict() for e in events]

    def get_recent(
        self,
        limit: int = 50,
        min_severity: Optional[SecuritySeverity] = None,
    ) -> List[SecurityAuditEvent]:
        """Get the most recent audit events.

        Args:
            limit: Maximum events to return.
            min_severity: Minimum severity level.

        Returns:
            List of recent events.
        """
        filters = AuditFilter(
            severity=min_severity,
            limit=limit,
        )
        events = self.get_audit_log(filters)
        return list(reversed(events))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get audit log statistics.

        Returns:
            Dictionary with event counts, severity breakdown, etc.
        """
        with self._lock:
            return {
                "total_events": self._stats["total_events"],
                "current_events": len(self._events),
                "by_severity": self._stats["by_severity"].copy(),
                "by_type": self._stats["by_type"].copy(),
                "flush_count": self._stats["flush_count"],
                "last_flush": self._stats["last_flush"],
                "max_events": self._max_events,
                "retention_days": self._retention_days,
            }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_log(
        self,
        format: str,  # noqa: A002 — 'format' is intentional
        path: str,
        filters: Optional[AuditFilter] = None,
    ) -> str:
        """Export the audit log to a file.

        Args:
            format: Export format — 'json' or 'csv'.
            path: Output file path.
            filters: Optional filters to apply before export.

        Returns:
            The path to the exported file.

        Raises:
            ValueError: If format is not 'json' or 'csv'.
        """
        events = self.get_audit_log(filters)
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            self._export_json(events, export_path)
        elif format.lower() == "csv":
            self._export_csv(events, export_path)
        else:
            raise ValueError(
                f"Unsupported export format: {format!r}. Use 'json' or 'csv'."
            )

        logger.info(
            "Exported %d audit events to %s (%s)",
            len(events), export_path, format,
        )
        return str(export_path)

    def _export_json(
        self,
        events: List[SecurityAuditEvent],
        path: Path,
    ) -> None:
        """Export events as JSON."""
        data = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_events": len(events),
            "events": [e.to_dict() for e in events],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_csv(
        self,
        events: List[SecurityAuditEvent],
        path: Path,
    ) -> None:
        """Export events as CSV."""
        fieldnames = [
            "id", "timestamp", "event_type", "severity",
            "source", "description", "user", "session_id",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in events:
                writer.writerow({
                    "id": event.id,
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "severity": event.severity.value,
                    "source": event.source,
                    "description": event.description,
                    "user": event.user,
                    "session_id": event.session_id,
                })

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write current events to the persistence file."""
        with self._lock:
            events_copy = list(self._events)
            self._stats["flush_count"] += 1
            self._stats["last_flush"] = datetime.now(timezone.utc).isoformat()

        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": "1.0",
                "events": [e.to_dict() for e in events_copy],
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "count": len(events_copy),
            }

            temp_path = str(self._persistence_path) + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, str(self._persistence_path))

            logger.debug(
                "Flushed %d audit events to %s",
                len(events_copy), self._persistence_path,
            )
        except Exception as exc:
            logger.error("Audit flush failed: %s", exc)

    def _maybe_flush(self) -> None:
        """Conditionally flush based on event count threshold."""
        if len(self._events) >= self._max_events:
            self.flush()

    def _load(self) -> None:
        """Load events from the persistence file."""
        if not self._persistence_path.exists():
            return
        try:
            with open(self._persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            events_data = data.get("events", [])
            for event_data in events_data:
                try:
                    event = SecurityAuditEvent.from_dict(event_data)
                    self._events.append(event)
                    self._update_stats(event)
                except Exception as exc:
                    logger.warning("Failed to load audit event: %s", exc)

            logger.info(
                "Loaded %d audit events from %s",
                len(self._events), self._persistence_path,
            )
        except Exception as exc:
            logger.error("Failed to load audit log: %s", exc)

    def _prune_retention(self) -> None:
        """Remove events older than the retention period."""
        if self._retention_days <= 0:
            return

        cutoff = datetime.now(timezone.utc).timestamp() - (self._retention_days * 86400)
        original_len = len(self._events)
        self._events = [
            e for e in self._events
            if e.timestamp.timestamp() >= cutoff
        ]
        removed = original_len - len(self._events)
        if removed > 0:
            logger.info("Pruned %d expired audit events", removed)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_stats(self, event: SecurityAuditEvent) -> None:
        """Update statistics counters for an event.

        Args:
            event: The event to count.
        """
        self._stats["total_events"] += 1

        sev = event.severity.value
        self._stats["by_severity"][sev] = (
            self._stats["by_severity"].get(sev, 0) + 1
        )

        evt = event.event_type.value
        self._stats["by_type"][evt] = (
            self._stats["by_type"].get(evt, 0) + 1
        )

    def _sanitize_input(self, data: Any) -> Any:
        """Sanitize tool input for safe logging.

        Masks common secret patterns and truncates large values.

        Args:
            data: Input data to sanitize.

        Returns:
            Sanitized copy of the data.
        """
        import re
        if isinstance(data, str):
            # Mask common secret patterns
            masked = re.sub(
                r'(api[_-]?key|token|password|secret)["\s]*[:=]["\s]*["\']?'
                r'([^\s"\'\}]{8,})',
                r'\1=****',
                data,
                flags=re.IGNORECASE,
            )
            # Truncate long strings
            if len(masked) > 500:
                masked = masked[:500] + "...[truncated]"
            return masked
        elif isinstance(data, dict):
            return {k: self._sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_input(item) for item in data[:10]]
        return data

    def clear(self) -> int:
        """Clear all audit events.

        Returns:
            Number of events cleared.
        """
        with self._lock:
            count = len(self._events)
            self._events.clear()
            self._stats = {
                "total_events": 0,
                "by_severity": {},
                "by_type": {},
                "flush_count": 0,
                "last_flush": None,
            }
        return count

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        return f"<SecurityAuditor events={len(self._events)}>"
