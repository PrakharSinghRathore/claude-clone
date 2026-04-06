"""
Atlas Polls — channel and group poll management.

Provides poll creation, voting, result calculation, and lifecycle management
including auto-expiration, duplicate prevention, and multi-choice support.

Classes
-------
PollManager
    Create and manage polls within channels and groups.
Poll
    A poll with options, votes, and lifecycle status.
PollOption
    A single option within a poll.
PollResults
    Aggregated results with per-option percentages.
PollStatus
    Poll lifecycle status enum (ACTIVE, CLOSED, EXPIRED).
"""

from .manager import (
    DuplicateVoteError,
    InvalidVoteError,
    Poll,
    PollClosedError,
    PollError,
    PollManager,
    PollNotFoundError,
    PollOption,
    PollResults,
    PollStatus,
)

__all__ = [
    "PollManager",
    "Poll",
    "PollOption",
    "PollResults",
    "PollStatus",
    "PollError",
    "PollNotFoundError",
    "PollClosedError",
    "DuplicateVoteError",
    "InvalidVoteError",
]
