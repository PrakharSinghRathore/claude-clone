"""
Permission & Approval System for AI Desktop Assistant.

Controls what the AI can and cannot do on the user's PC with fine-grained
action categories, per-action approval workflows, time-based restrictions,
scope limits, and comprehensive audit logging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PermissionLevel(Enum):
    """Overall permission level controlling the default behaviour."""
    FULL_ACCESS = "full_access"
    STANDARD = "standard"
    LIMITED = "limited"
    READ_ONLY = "read_only"
    CUSTOM = "custom"


class ActionCategory(Enum):
    """Granular action categories the permission system recognises."""
    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    WINDOW = "window"
    APPLICATION = "application"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    NETWORK = "network"
    SYSTEM = "system"
    CLIPBOARD = "clipboard"
    SCREENSHOT = "screenshot"
    INSTALL = "install"
    SHELL_COMMAND = "shell_command"
    NOTIFICATION = "notification"


class PermissionStatus(Enum):
    """Result of a permission check."""
    ALLOWED = "allowed"
    DENIED = "denied"
    REQUIRES_APPROVAL = "requires_approval"
    NOT_CONFIGURED = "not_configured"


class ApprovalDecision(Enum):
    """Outcome of a user approval decision."""
    APPROVED = "approved"
    DENIED = "denied"
    TIMED_OUT = "timed_out"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PermissionRule:
    """Rule governing a single *ActionCategory*."""
    category: ActionCategory
    status: PermissionStatus = PermissionStatus.NOT_CONFIGURED
    allowed_paths: list[str] = field(default_factory=list)
    denied_paths: list[str] = field(default_factory=list)
    allowed_apps: list[str] = field(default_factory=list)
    max_frequency: int = 0  # 0 = unlimited, >0 = max per minute
    time_restriction: dict = field(default_factory=dict)
    # time_restriction example: {"start_hour": 9, "end_hour": 17}

    # ------------------------------------------------------------------
    # Frequency tracking (runtime only, not persisted)
    # ------------------------------------------------------------------
    _action_timestamps: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (excluding private runtime fields)."""
        d: dict[str, Any] = {
            "category": self.category.value,
            "status": self.status.value,
            "allowed_paths": self.allowed_paths,
            "denied_paths": self.denied_paths,
            "allowed_apps": self.allowed_apps,
            "max_frequency": self.max_frequency,
            "time_restriction": self.time_restriction,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PermissionRule:
        """Deserialise from a plain dict."""
        return cls(
            category=ActionCategory(data["category"]),
            status=PermissionStatus(data.get("status", "not_configured")),
            allowed_paths=data.get("allowed_paths", []),
            denied_paths=data.get("denied_paths", []),
            allowed_apps=data.get("allowed_apps", []),
            max_frequency=data.get("max_frequency", 0),
            time_restriction=data.get("time_restriction", {}),
        )


@dataclass
class ApprovalRequest:
    """Represents a single pending approval request."""
    id: str
    action: str
    category: ActionCategory
    description: str
    risk_level: str
    timestamp: float
    decision: ApprovalDecision = ApprovalDecision.TIMED_OUT
    response_time: float = 0.0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action": self.action,
            "category": self.category.value,
            "description": self.description,
            "risk_level": self.risk_level,
            "timestamp": self.timestamp,
            "decision": self.decision.value,
            "response_time": self.response_time,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ApprovalRequest:
        return cls(
            id=data["id"],
            action=data["action"],
            category=ActionCategory(data["category"]),
            description=data["description"],
            risk_level=data["risk_level"],
            timestamp=data["timestamp"],
            decision=ApprovalDecision(data.get("decision", "timed_out")),
            response_time=data.get("response_time", 0.0),
            details=data.get("details", {}),
        )


@dataclass
class AuditEntry:
    """Immutable log entry for a single executed action."""
    timestamp: float
    action: str
    category: ActionCategory
    approved: bool
    outcome: str
    duration_ms: float = 0.0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "category": self.category.value,
            "approved": self.approved,
            "outcome": self.outcome,
            "duration_ms": self.duration_ms,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AuditEntry:
        return cls(
            timestamp=data["timestamp"],
            action=data["action"],
            category=ActionCategory(data["category"]),
            approved=data["approved"],
            outcome=data["outcome"],
            duration_ms=data.get("duration_ms", 0.0),
            details=data.get("details", {}),
        )


# ---------------------------------------------------------------------------
# Risk heuristics
# ---------------------------------------------------------------------------

_HIGH_RISK_CATEGORIES = frozenset({
    ActionCategory.FILE_DELETE,
    ActionCategory.INSTALL,
    ActionCategory.SHELL_COMMAND,
    ActionCategory.SYSTEM,
})

_MEDIUM_RISK_CATEGORIES = frozenset({
    ActionCategory.FILE_WRITE,
    ActionCategory.NETWORK,
    ActionCategory.APPLICATION,
    ActionCategory.KEYBOARD,
})

_HIGH_RISK_PATTERNS = (
    "sudo", "rm ", "rmdir", "format", "del ", "regsvr32",
    "powershell -command", "cmd /c", "mkfs", "dd if=",
    ":(){ :|:& };:", "chmod 777", "DROP TABLE", "DROP DATABASE",
)

_CRITICAL_TARGETS = (
    "/etc/passwd", "/etc/shadow", "/boot/", "/usr/bin/", "/System/",
    "C:\\Windows\\System32", "HKLM\\",
)


# ---------------------------------------------------------------------------
# PermissionManager
# ---------------------------------------------------------------------------

class PermissionManager:
    """
    Central authority for deciding what the AI desktop assistant may do.

    Supports per-category rules, time-based windows, scope restrictions,
    approval queues, and a full audit trail.
    """

    # Maximum entries kept in the in-memory audit log (ring-buffer size).
    MAX_AUDIT_ENTRIES: int = 50_000

    def __init__(self, config_path: str = "~/.claude_clone/permissions.json") -> None:
        self._config_path = Path(os.path.expanduser(config_path))
        self._level: PermissionLevel = PermissionLevel.STANDARD
        self._rules: dict[ActionCategory, PermissionRule] = {}
        # category -> list of ApprovalRequest (pending = not yet decided)
        self._approval_queue: dict[str, ApprovalRequest] = {}
        self._approval_futures: dict[str, asyncio.Future[ApprovalDecision]] = {}
        # Ring-buffer audit log
        self._audit_log: list[AuditEntry] = []
        self._initialised: bool = False
        # Lock for thread-safe internal mutations
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load persisted permissions from disk (if any)."""
        async with self._lock:
            if self._initialised:
                return

            defaults = self._get_default_rules()
            self._rules = {cat: rule for cat, rule in defaults.items()}

            if self._config_path.exists():
                try:
                    raw = self._config_path.read_text(encoding="utf-8")
                    data = json.loads(raw) if raw.strip() else {}
                    self._apply_persisted_data(data)
                    logger.info("Permissions loaded from %s", self._config_path)
                except Exception:
                    logger.exception("Failed to load permissions config; using defaults")
                    self._rules = {cat: rule for cat, rule in defaults.items()}
            else:
                self._level = PermissionLevel.STANDARD

            self._initialised = True

    async def save(self) -> None:
        """Persist current rules and level to the config file."""
        async with self._lock:
            data: dict[str, Any] = {
                "level": self._level.value,
                "rules": {
                    cat.value: rule.to_dict()
                    for cat, rule in self._rules.items()
                    if rule.status != PermissionStatus.NOT_CONFIGURED
                },
            }
            try:
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._config_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self._config_path)
                logger.info("Permissions saved to %s", self._config_path)
            except Exception:
                logger.exception("Failed to save permissions config")

    # ------------------------------------------------------------------
    # Permission checking
    # ------------------------------------------------------------------

    async def check_permission(
        self,
        action: str,
        category: ActionCategory,
        details: dict | None = None,
    ) -> PermissionStatus:
        """
        Determine whether *action* under *category* is currently allowed.

        Returns ``REQUIRES_APPROVAL`` for categories configured as such
        (e.g. FILE_DELETE, SHELL_COMMAND at STANDARD level).
        """
        details = details or {}
        async with self._lock:
            rule = self._rules.get(category)

            if rule is None or rule.status == PermissionStatus.NOT_CONFIGURED:
                return self._fallback_status(category)

            # Explicitly denied → hard deny
            if rule.status == PermissionStatus.DENIED:
                return PermissionStatus.DENIED

            # Check time-based restrictions
            if not self._check_time_restriction(rule):
                return PermissionStatus.DENIED

            # Check scope restrictions (paths)
            if details.get("path"):
                if not self._check_path_scope(details["path"], rule):
                    return PermissionStatus.DENIED

            # Check scope restrictions (apps)
            if details.get("app"):
                if not self._check_app_scope(details["app"], rule):
                    return PermissionStatus.DENIED

            # Check frequency limits
            if not self._check_frequency(rule):
                return PermissionStatus.DENIED

            if rule.status == PermissionStatus.REQUIRES_APPROVAL:
                return PermissionStatus.REQUIRES_APPROVAL

            return PermissionStatus.ALLOWED

    # ------------------------------------------------------------------
    # Approval workflow
    # ------------------------------------------------------------------

    async def request_approval(
        self,
        action: str,
        category: ActionCategory,
        description: str,
        details: dict | None = None,
        timeout: float = 60.0,
    ) -> ApprovalDecision:
        """
        Submit an action for user approval.

        The coroutine blocks until the user approves, denies, or *timeout*
        seconds elapse.
        """
        details = details or {}
        request_id = str(uuid.uuid4())
        risk_level = self._assess_risk(action, category, details)
        timestamp = time.time()

        request = ApprovalRequest(
            id=request_id,
            action=action,
            category=category,
            description=description,
            risk_level=risk_level,
            timestamp=timestamp,
            details=details,
        )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[ApprovalDecision] = loop.create_future()

        async with self._lock:
            self._approval_queue[request_id] = request
            self._approval_futures[request_id] = future

        logger.info(
            "Approval requested [%s] %s – risk=%s timeout=%.1fs",
            request_id, action, risk_level, timeout,
        )

        try:
            decision = await asyncio.wait_for(future, timeout=timeout)
            return decision
        except asyncio.TimeoutError:
            async with self._lock:
                request.decision = ApprovalDecision.TIMED_OUT
                request.response_time = time.time() - timestamp
                self._approval_queue.pop(request_id, None)
                self._approval_futures.pop(request_id, None)
            logger.warning("Approval timed out [%s] %s", request_id, action)
            return ApprovalDecision.TIMED_OUT

    async def approve(self, request_id: str) -> None:
        """Approve a pending approval request by its id."""
        async with self._lock:
            future = self._approval_futures.get(request_id)
            request = self._approval_queue.get(request_id)
        if future is None or request is None:
            logger.warning("approve() called for unknown request %s", request_id)
            return
        if future.done():
            return
        request.decision = ApprovalDecision.APPROVED
        request.response_time = time.time() - request.timestamp
        future.set_result(ApprovalDecision.APPROVED)
        async with self._lock:
            self._approval_queue.pop(request_id, None)
            self._approval_futures.pop(request_id, None)
        logger.info("Approved [%s] %s", request_id, request.action)

    async def deny(self, request_id: str) -> None:
        """Deny a pending approval request by its id."""
        async with self._lock:
            future = self._approval_futures.get(request_id)
            request = self._approval_queue.get(request_id)
        if future is None or request is None:
            logger.warning("deny() called for unknown request %s", request_id)
            return
        if future.done():
            return
        request.decision = ApprovalDecision.DENIED
        request.response_time = time.time() - request.timestamp
        future.set_result(ApprovalDecision.DENIED)
        async with self._lock:
            self._approval_queue.pop(request_id, None)
            self._approval_futures.pop(request_id, None)
        logger.info("Denied [%s] %s", request_id, request.action)

    async def approve_all(self) -> None:
        """Approve every currently pending request."""
        async with self._lock:
            request_ids = list(self._approval_queue.keys())
        for rid in request_ids:
            await self.approve(rid)

    async def deny_all(self) -> None:
        """Deny every currently pending request."""
        async with self._lock:
            request_ids = list(self._approval_queue.keys())
        for rid in request_ids:
            await self.deny(rid)

    async def get_pending_approvals(self) -> list[ApprovalRequest]:
        """Return a snapshot of all pending (undecided) approval requests."""
        async with self._lock:
            return list(self._approval_queue.values())

    # ------------------------------------------------------------------
    # Rule / level management
    # ------------------------------------------------------------------

    async def set_permission(
        self,
        category: ActionCategory,
        status: PermissionStatus,
        scope: dict | None = None,
    ) -> None:
        """Set the permission status for *category*, optionally with scope."""
        scope = scope or {}
        async with self._lock:
            rule = self._rules.get(category)
            if rule is None:
                rule = PermissionRule(category=category)
                self._rules[category] = rule
            rule.status = status
            rule.allowed_paths = scope.get("allowed_paths", rule.allowed_paths)
            rule.denied_paths = scope.get("denied_paths", rule.denied_paths)
            rule.allowed_apps = scope.get("allowed_apps", rule.allowed_apps)
            rule.max_frequency = scope.get("max_frequency", rule.max_frequency)
            rule.time_restriction = scope.get("time_restriction", rule.time_restriction)

        # Upgrade to CUSTOM when the user changes individual rules
        if self._level not in (PermissionLevel.CUSTOM, PermissionLevel.FULL_ACCESS):
            self._level = PermissionLevel.CUSTOM

    async def set_level(self, level: PermissionLevel) -> None:
        """
        Set the overall permission level.

        This replaces all rules with the preset rules for the chosen level
        while preserving any user-added CUSTOM rules that are not part of
        the standard presets.
        """
        async with self._lock:
            self._level = level
            if level == PermissionLevel.FULL_ACCESS:
                for rule in self._rules.values():
                    rule.status = PermissionStatus.ALLOWED
            elif level == PermissionLevel.STANDARD:
                self._rules = self._get_default_rules()
            elif level == PermissionLevel.LIMITED:
                self._rules = self._get_limited_rules()
            elif level == PermissionLevel.READ_ONLY:
                self._rules = self._get_readonly_rules()
            # CUSTOM: do not touch existing rules

    async def get_level(self) -> PermissionLevel:
        return self._level

    async def add_rule(self, rule: PermissionRule) -> None:
        async with self._lock:
            self._rules[rule.category] = rule
        if self._level != PermissionLevel.CUSTOM:
            self._level = PermissionLevel.CUSTOM

    async def remove_rule(self, category: ActionCategory) -> None:
        async with self._lock:
            self._rules.pop(category, None)

    async def get_rules(self) -> dict[ActionCategory, PermissionRule]:
        return dict(self._rules)

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    async def add_audit_entry(self, entry: AuditEntry) -> None:
        async with self._lock:
            self._audit_log.append(entry)
            if len(self._audit_log) > self.MAX_AUDIT_ENTRIES:
                self._audit_log = self._audit_log[-(self.MAX_AUDIT_ENTRIES // 2):]

    async def get_audit_log(
        self,
        limit: int = 100,
        category: ActionCategory | None = None,
    ) -> list[AuditEntry]:
        async with self._lock:
            if category is not None:
                filtered = [e for e in self._audit_log if e.category == category]
            else:
                filtered = list(self._audit_log)
        # Return most recent first
        return list(reversed(filtered[-limit:]))

    async def get_stats(self) -> dict:
        """Return aggregate statistics over the audit trail."""
        async with self._lock:
            total = len(self._audit_log)
            approved = sum(1 for e in self._audit_log if e.approved)
            denied = total - approved

            by_category: dict[str, int] = {}
            for e in self._audit_log:
                key = e.category.value
                by_category[key] = by_category.get(key, 0) + 1

            by_outcome: dict[str, int] = {}
            for e in self._audit_log:
                by_outcome[e.outcome] = by_outcome.get(e.outcome, 0) + 1

            pending_approvals = len(self._approval_queue)

        return {
            "total_actions": total,
            "approved": approved,
            "denied": denied,
            "by_category": by_category,
            "by_outcome": by_outcome,
            "pending_approvals": pending_approvals,
            "permission_level": self._level.value,
        }

    # ------------------------------------------------------------------
    # Reset / import / export
    # ------------------------------------------------------------------

    async def reset(self) -> None:
        """Reset all rules and audit log to fresh defaults."""
        async with self._lock:
            # Cancel any pending approval futures
            for future in self._approval_futures.values():
                if not future.done():
                    future.set_result(ApprovalDecision.TIMED_OUT)
            self._approval_queue.clear()
            self._approval_futures.clear()
            self._audit_log.clear()
            self._level = PermissionLevel.STANDARD
            self._rules = self._get_default_rules()

    async def export_rules(self, filepath: str) -> None:
        """Export current rules to a JSON file."""
        path = Path(filepath)
        async with self._lock:
            data = {
                "level": self._level.value,
                "rules": {
                    cat.value: rule.to_dict()
                    for cat, rule in self._rules.items()
                },
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Rules exported to %s", path)

    async def import_rules(self, filepath: str) -> None:
        """Import rules from a JSON file, replacing current rules."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Rules file not found: {path}")
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        async with self._lock:
            self._apply_persisted_data(data)
        logger.info("Rules imported from %s", path)

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def _assess_risk(
        self,
        action: str,
        category: ActionCategory,
        details: dict | None,
    ) -> str:
        """Return one of LOW, MEDIUM, HIGH, CRITICAL."""
        details = details or {}

        # CRITICAL: destructive commands targeting system paths
        if category in _HIGH_RISK_CATEGORIES:
            if any(pattern in action.lower() for pattern in _HIGH_RISK_PATTERNS):
                return "CRITICAL"
            if any(target.lower() in details.get("path", "").lower() for target in _CRITICAL_TARGETS):
                return "CRITICAL"

        if category in _HIGH_RISK_CATEGORIES:
            return "HIGH"

        if category in _MEDIUM_RISK_CATEGORIES:
            # Moderate risk if interacting with system directories
            path = details.get("path", "")
            if any(target.lower() in path.lower() for target in _CRITICAL_TARGETS):
                return "HIGH"
            return "MEDIUM"

        # FILE_READ with sensitive-looking paths bumps to MEDIUM
        if category == ActionCategory.FILE_READ:
            path = details.get("path", "")
            sensitive = ("/etc/shadow", "/etc/passwd", ".ssh/", ".env", "credentials")
            if any(s in path.lower() for s in sensitive):
                return "MEDIUM"

        return "LOW"

    # ------------------------------------------------------------------
    # Default rule sets
    # ------------------------------------------------------------------

    @staticmethod
    def _get_default_rules() -> dict[ActionCategory, PermissionRule]:
        """
        STANDARD level defaults.

        - Read-only categories are auto-approved.
        - FILE_DELETE, INSTALL, SHELL_COMMAND require approval.
        - Everything else is allowed.
        """
        return {
            ActionCategory.MOUSE: PermissionRule(
                category=ActionCategory.MOUSE,
                status=PermissionStatus.ALLOWED,
            ),
            ActionCategory.KEYBOARD: PermissionRule(
                category=ActionCategory.KEYBOARD,
                status=PermissionStatus.ALLOWED,
            ),
            ActionCategory.WINDOW: PermissionRule(
                category=ActionCategory.WINDOW,
                status=PermissionStatus.ALLOWED,
            ),
            ActionCategory.APPLICATION: PermissionRule(
                category=ActionCategory.APPLICATION,
                status=PermissionStatus.ALLOWED,
            ),
            ActionCategory.FILE_READ: PermissionRule(
                category=ActionCategory.FILE_READ,
                status=PermissionStatus.ALLOWED,
            ),
            ActionCategory.FILE_WRITE: PermissionRule(
                category=ActionCategory.FILE_WRITE,
                status=PermissionStatus.ALLOWED,
            ),
            ActionCategory.FILE_DELETE: PermissionRule(
                category=ActionCategory.FILE_DELETE,
                status=PermissionStatus.REQUIRES_APPROVAL,
            ),
            ActionCategory.NETWORK: PermissionRule(
                category=ActionCategory.NETWORK,
                status=PermissionStatus.ALLOWED,
            ),
            ActionCategory.SYSTEM: PermissionRule(
                category=ActionCategory.SYSTEM,
                status=PermissionStatus.REQUIRES_APPROVAL,
            ),
            ActionCategory.CLIPBOARD: PermissionRule(
                category=ActionCategory.CLIPBOARD,
                status=PermissionStatus.ALLOWED,
            ),
            ActionCategory.SCREENSHOT: PermissionRule(
                category=ActionCategory.SCREENSHOT,
                status=PermissionStatus.ALLOWED,
            ),
            ActionCategory.INSTALL: PermissionRule(
                category=ActionCategory.INSTALL,
                status=PermissionStatus.REQUIRES_APPROVAL,
            ),
            ActionCategory.SHELL_COMMAND: PermissionRule(
                category=ActionCategory.SHELL_COMMAND,
                status=PermissionStatus.REQUIRES_APPROVAL,
            ),
            ActionCategory.NOTIFICATION: PermissionRule(
                category=ActionCategory.NOTIFICATION,
                status=PermissionStatus.ALLOWED,
            ),
        }

    @staticmethod
    def _get_limited_rules() -> dict[ActionCategory, PermissionRule]:
        """
        LIMITED level: only read-oriented actions and clipboard/notifications
        are allowed; everything else requires approval.
        """
        allowed_categories = {
            ActionCategory.FILE_READ,
            ActionCategory.CLIPBOARD,
            ActionCategory.SCREENSHOT,
            ActionCategory.NOTIFICATION,
        }
        rules: dict[ActionCategory, PermissionRule] = {}
        for cat in ActionCategory:
            status = (
                PermissionStatus.ALLOWED
                if cat in allowed_categories
                else PermissionStatus.REQUIRES_APPROVAL
            )
            rules[cat] = PermissionRule(category=cat, status=status)
        return rules

    @staticmethod
    def _get_readonly_rules() -> dict[ActionCategory, PermissionRule]:
        """
        READ_ONLY level: only truly passive/read actions are allowed;
        all mutating actions are denied.
        """
        read_only_categories = {
            ActionCategory.FILE_READ,
            ActionCategory.SCREENSHOT,
            ActionCategory.CLIPBOARD,
            ActionCategory.NOTIFICATION,
            ActionCategory.WINDOW,
        }
        rules: dict[ActionCategory, PermissionRule] = {}
        for cat in ActionCategory:
            if cat in read_only_categories:
                rules[cat] = PermissionRule(category=cat, status=PermissionStatus.ALLOWED)
            else:
                rules[cat] = PermissionRule(category=cat, status=PermissionStatus.DENIED)
        return rules

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_persisted_data(self, data: dict) -> None:
        """Merge persisted config data into current state (must hold lock)."""
        level_str = data.get("level", "standard")
        try:
            self._level = PermissionLevel(level_str)
        except ValueError:
            self._level = PermissionLevel.STANDARD

        persisted_rules = data.get("rules", {})
        for cat_key, rule_dict in persisted_rules.items():
            try:
                cat = ActionCategory(cat_key)
                self._rules[cat] = PermissionRule.from_dict(rule_dict)
            except ValueError:
                logger.warning("Unknown action category '%s' in config; skipping", cat_key)

    @staticmethod
    def _fallback_status(category: ActionCategory) -> PermissionStatus:
        """Status returned when no rule is configured for *category*."""
        if category in _HIGH_RISK_CATEGORIES:
            return PermissionStatus.REQUIRES_APPROVAL
        if category in _MEDIUM_RISK_CATEGORIES:
            return PermissionStatus.ALLOWED
        return PermissionStatus.ALLOWED

    @staticmethod
    def _check_time_restriction(rule: PermissionRule) -> bool:
        """Return True if the current time falls within the rule's window."""
        tr = rule.time_restriction
        if not tr:
            return True
        start = tr.get("start_hour")
        end = tr.get("end_hour")
        if start is None or end is None:
            return True
        now = datetime.now(timezone.utc).hour
        if start <= end:
            return start <= now < end
        # Wraps around midnight, e.g. 22–06
        return now >= start or now < end

    @staticmethod
    def _check_path_scope(path: str, rule: PermissionRule) -> bool:
        """Return True if *path* is not explicitly denied and is allowed (if allowed_paths is set)."""
        # Normalise path separators for cross-platform comparison
        norm = path.replace("\\", "/")

        # Denied paths take priority
        for denied in rule.denied_paths:
            denied_norm = denied.replace("\\", "/")
            if norm.startswith(denied_norm):
                return False

        # If allowed_paths is empty → everything is allowed (no restriction)
        if not rule.allowed_paths:
            return True

        for allowed in rule.allowed_paths:
            allowed_norm = allowed.replace("\\", "/")
            if norm.startswith(allowed_norm):
                return True

        return False

    @staticmethod
    def _check_app_scope(app: str, rule: PermissionRule) -> bool:
        """Return True if *app* is allowed by the rule's allowed_apps list."""
        if not rule.allowed_apps:
            return True
        app_lower = app.lower()
        return any(allowed.lower() in app_lower for allowed in rule.allowed_apps)

    @staticmethod
    def _check_frequency(rule: PermissionRule) -> bool:
        """Return True if the action has not exceeded max_frequency (per minute)."""
        if rule.max_frequency <= 0:
            return True
        now = time.time()
        one_minute_ago = now - 60.0
        # Prune old entries
        rule._action_timestamps = [
            ts for ts in rule._action_timestamps if ts > one_minute_ago
        ]
        if len(rule._action_timestamps) >= rule.max_frequency:
            return False
        rule._action_timestamps.append(now)
        return True


# ---------------------------------------------------------------------------
# Convenience: decorate an action call with permission checking + approval
# ---------------------------------------------------------------------------

class PermissionGuard:
    """
    Context-manager style guard that checks permission before executing an
    action and, if required, requests user approval.

    Usage::

        guard = PermissionGuard(manager, action="rm -rf /tmp/foo",
                                category=ActionCategory.FILE_DELETE,
                                description="Delete temp files in /tmp/foo")
        if await guard.check():
            # action is approved – go ahead
            await guard.record_audit(outcome="success")
    """

    def __init__(
        self,
        manager: PermissionManager,
        action: str,
        category: ActionCategory,
        description: str = "",
        details: dict | None = None,
        approval_timeout: float = 60.0,
    ) -> None:
        self._manager = manager
        self._action = action
        self._category = category
        self._description = description or action
        self._details = details or {}
        self._approval_timeout = approval_timeout
        self._start_time: float = 0.0
        self._status: PermissionStatus | None = None
        self._decision: ApprovalDecision | None = None

    async def check(self) -> bool:
        """
        Run the permission check.  Returns ``True`` if the action may proceed.

        This may block while waiting for user approval.
        """
        self._start_time = time.time()
        status = await self._manager.check_permission(
            self._action, self._category, self._details,
        )
        self._status = status

        if status == PermissionStatus.ALLOWED:
            self._decision = ApprovalDecision.APPROVED
            return True

        if status == PermissionStatus.REQUIRES_APPROVAL:
            decision = await self._manager.request_approval(
                action=self._action,
                category=self._category,
                description=self._description,
                details=self._details,
                timeout=self._approval_timeout,
            )
            self._decision = decision
            return decision == ApprovalDecision.APPROVED

        # DENIED or NOT_CONFIGURED (treated as deny)
        self._decision = ApprovalDecision.DENIED
        return False

    async def record_audit(self, outcome: str) -> None:
        """Append an audit entry for the completed action."""
        duration_ms = (time.time() - self._start_time) * 1000.0 if self._start_time else 0.0
        entry = AuditEntry(
            timestamp=time.time(),
            action=self._action,
            category=self._category,
            approved=self._decision == ApprovalDecision.APPROVED,
            outcome=outcome,
            duration_ms=duration_ms,
            details=self._details,
        )
        await self._manager.add_audit_entry(entry)
