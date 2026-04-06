"""
Atlas Sessions — Comprehensive session lifecycle management.

This module provides session creation, persistence, transcript management,
and deterministic key derivation for conversation sessions across multiple
messaging platforms.

Exports:
    SessionManager        – Session lifecycle management with activation modes.
    Session               – Session data model.
    SessionStatus         – Session status enum.
    SessionStore          – Persistent session storage.
    TranscriptEntry       – Transcript entry data model.
    TranscriptRole        – Transcript role enum.
    TranscriptStore       – Conversation transcript management.
    TranscriptCompactor   – Transcript compaction and summarization.
    SessionKeyDerivation  – Deterministic session key generation.
    ChannelNormalizer     – Channel identifier normalization.
    ActivationMode        – Session activation strategy.
    QueueMode             – Session queue ordering strategy.
    SessionError          – Base exception for session errors.
    SessionNotFoundError  – Session not found exception.
    SessionLimitError     – Session limit exceeded exception.
    derive_session_key    – Convenience function for session key derivation.
    derive_group_key      – Convenience function for group key derivation.
"""

from .manager import (
    SessionManager,
    ActivationMode,
    QueueMode,
    SessionCallbacks,
    QueuedSession,
    SessionError,
    SessionNotFoundError,
    SessionLimitError,
    SessionAlreadyActiveError,
)
from .store import Session, SessionStatus, SessionStore
from .transcript import (
    TranscriptEntry,
    TranscriptRole,
    TranscriptStore,
    TranscriptCompactor,
)
from .keys import (
    SessionKeyDerivation,
    ChannelNormalizer,
    KeyScope,
    derive_session_key,
    derive_group_key,
)

__all__ = [
    # Manager
    "SessionManager",
    "ActivationMode",
    "QueueMode",
    "SessionCallbacks",
    "QueuedSession",
    "SessionError",
    "SessionNotFoundError",
    "SessionLimitError",
    "SessionAlreadyActiveError",
    # Store
    "Session",
    "SessionStatus",
    "SessionStore",
    # Transcript
    "TranscriptEntry",
    "TranscriptRole",
    "TranscriptStore",
    "TranscriptCompactor",
    # Keys
    "SessionKeyDerivation",
    "ChannelNormalizer",
    "KeyScope",
    "derive_session_key",
    "derive_group_key",
]
