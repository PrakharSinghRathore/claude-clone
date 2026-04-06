"""
Atlas Security — Comprehensive Security Module.

Provides security audit logging, policy enforcement, sandboxed execution,
secret management, allowlists, and DM pairing for the Atlas integration
engine.

Modules:
    audit — SecurityAuditor for event logging and export
    policy — SecurityPolicy for rule evaluation (tool, file, network, DM, sandbox)
    sandbox — SandboxExecutor for isolated code execution
    secrets — SecretManager for secure credential storage
    allowlist — AllowlistManager for access control
    pairing — PairingManager for DM pairing security

Usage::

    from atlas.security import (
        # Audit
        SecurityAuditor, SecurityAuditEvent, AuditEventType,
        SecuritySeverity, AuditFilter,
        # Policy
        SecurityPolicy, ToolPolicy, FilePolicy, NetworkPolicy,
        DMPolicy, SandboxPolicy, PolicyDecision,
        # Sandbox
        SandboxExecutor, SandboxType, ResourceLimits, ExecutionResult,
        # Secrets
        SecretManager, SecretEntry,
        # Allowlist
        AllowlistManager, AllowlistEntryType,
        # Pairing
        PairingManager, PendingPairingCode, PairedContact,
    )
"""

from atlas.security.audit import (
    AuditEventType,
    AuditFilter,
    SecurityAuditEvent,
    SecurityAuditor,
    SecuritySeverity,
)
from atlas.security.policy import (
    DMPolicy,
    FilePolicy,
    NetworkPolicy,
    PolicyDecision,
    SandboxPolicy,
    SecurityPolicy,
    ToolPolicy,
)
from atlas.security.sandbox import (
    ExecutionResult,
    ExecutionStatus,
    ResourceLimits,
    SandboxExecutor,
    SandboxType,
)
from atlas.security.secrets import SecretEntry, SecretManager
from atlas.security.allowlist import AllowlistEntryType, AllowlistManager
from atlas.security.pairing import PairedContact, PairingManager, PendingPairingCode

__all__ = [
    # Audit
    "SecurityAuditor",
    "SecurityAuditEvent",
    "AuditEventType",
    "SecuritySeverity",
    "AuditFilter",
    # Policy
    "SecurityPolicy",
    "ToolPolicy",
    "FilePolicy",
    "NetworkPolicy",
    "DMPolicy",
    "SandboxPolicy",
    "PolicyDecision",
    # Sandbox
    "SandboxExecutor",
    "SandboxType",
    "ResourceLimits",
    "ExecutionResult",
    "ExecutionStatus",
    # Secrets
    "SecretManager",
    "SecretEntry",
    # Allowlist
    "AllowlistManager",
    "AllowlistEntryType",
    # Pairing
    "PairingManager",
    "PendingPairingCode",
    "PairedContact",
]
