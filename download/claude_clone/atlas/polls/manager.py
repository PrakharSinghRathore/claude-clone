"""
Poll management system for channels and groups.

Provides creation, voting, result calculation, and lifecycle management
for polls including auto-expiration, duplicate prevention, and
multi-choice support.

Usage::

    manager = PollManager()
    poll = manager.create(
        question="What framework should we use?",
        options=["Next.js", "Nuxt", "SvelteKit", "Astro"],
        creator="alice",
        channel="general",
        expires_at=datetime.now(tz=timezone.utc) + timedelta(hours=24),
    )
    manager.vote(poll_id=poll.id, voter_id="bob", option_ids=[0, 2])
    results = manager.get_results(poll.id)
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("atlas.polls.manager")


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class PollStatus(str, Enum):
    """Lifecycle status of a poll."""

    ACTIVE = "active"
    CLOSED = "closed"
    EXPIRED = "expired"


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PollOption:
    """A single option within a poll."""

    id: str
    text: str
    votes: int = 0
    voter_ids: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "votes": self.votes,
            "voter_ids": list(self.voter_ids),
        }


@dataclass
class Poll:
    """
    A poll with one or more options for voting.

    Attributes
    ----------
    id:
        Unique poll identifier.
    question:
        The question being asked.
    options:
        List of PollOption instances.
    created_by:
        User ID of the poll creator.
    channel:
        Channel or group where the poll was created.
    created_at:
        ISO-8601 timestamp of creation.
    expires_at:
        ISO-8601 timestamp when the poll expires (optional).
    is_anonymous:
        Whether voter identities are hidden.
    is_multi_choice:
        Whether voters can select multiple options.
    status:
        Current PollStatus.
    """

    id: str
    question: str
    options: List[PollOption]
    created_by: str
    channel: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    expires_at: Optional[str] = None
    is_anonymous: bool = False
    is_multi_choice: bool = False
    status: PollStatus = PollStatus.ACTIVE
    max_votes_per_voter: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "options": [opt.to_dict() for opt in self.options],
            "created_by": self.created_by,
            "channel": self.channel,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_anonymous": self.is_anonymous,
            "is_multi_choice": self.is_multi_choice,
            "status": self.status.value,
            "max_votes_per_voter": self.max_votes_per_voter,
            "total_votes": self.total_votes,
        }

    @property
    def total_votes(self) -> int:
        """Total votes across all options (counting duplicates per voter)."""
        return sum(opt.votes for opt in self.options)

    @property
    def unique_voters(self) -> int:
        """Count of unique voters across all options."""
        all_voters: Set[str] = set()
        for opt in self.options:
            all_voters |= opt.voter_ids
        return len(all_voters)

    @property
    def is_expired(self) -> bool:
        """Check if the poll has passed its expiration time."""
        if self.expires_at is None:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return datetime.now(timezone.utc) > exp
        except (ValueError, TypeError):
            return False

    @property
    def leading_option(self) -> Optional[PollOption]:
        """Return the option with the most votes, or None if tied/empty."""
        if not self.options:
            return None
        sorted_opts = sorted(self.options, key=lambda o: o.votes, reverse=True)
        if sorted_opts[0].votes == 0:
            return None
        # Check for tie
        if len(sorted_opts) > 1 and sorted_opts[0].votes == sorted_opts[1].votes:
            return None
        return sorted_opts[0]


@dataclass
class PollResults:
    """Aggregated results for a poll."""

    poll_id: str
    question: str
    status: PollStatus
    options: List[Dict[str, Any]]
    total_votes: int
    unique_voters: int
    created_at: str
    expires_at: Optional[str]
    percentages: Dict[str, float]
    is_tied: bool
    leader: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "poll_id": self.poll_id,
            "question": self.question,
            "status": self.status.value,
            "options": self.options,
            "total_votes": self.total_votes,
            "unique_voters": self.unique_voters,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "percentages": self.percentages,
            "is_tied": self.is_tied,
            "leader": self.leader,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class PollError(Exception):
    """Base exception for poll operations."""


class PollNotFoundError(PollError):
    """Raised when a poll ID is not found."""


class PollClosedError(PollError):
    """Raised when trying to vote on a closed or expired poll."""


class DuplicateVoteError(PollError):
    """Raised when a voter has already voted."""


class InvalidVoteError(PollError):
    """Raised when vote parameters are invalid."""


# ──────────────────────────────────────────────────────────────────────────────
# Poll Manager
# ──────────────────────────────────────────────────────────────────────────────

class PollManager:
    """
    Create and manage polls within channels and groups.

    Features:
    - Poll creation with flexible options
    - Single and multi-choice voting
    - Duplicate vote prevention
    - Auto-expiration handling
    - Results calculation with percentages
    - Poll lifecycle management (create, vote, close, delete)
    - Anonymous and named voting modes
    - Channel-scoped poll listing
    - Event callbacks for vote/close/expire

    Parameters
    ----------
    default_expires_in:
        Default duration until polls expire, in seconds. None = no expiration.
    max_options:
        Maximum number of options per poll. Default 20.
    min_options:
        Minimum number of options per poll. Default 2.
    max_question_length:
        Maximum question length in characters. Default 500.
    max_option_length:
        Maximum option text length in characters. Default 200.
    """

    def __init__(
        self,
        default_expires_in: Optional[float] = None,
        max_options: int = 20,
        min_options: int = 2,
        max_question_length: int = 500,
        max_option_length: int = 200,
    ) -> None:
        self._default_expires_in = default_expires_in
        self._max_options = max_options
        self._min_options = min_options
        self._max_question_length = max_question_length
        self._max_option_length = max_option_length

        # Poll storage: poll_id -> Poll
        self._polls: Dict[str, Poll] = {}

        # Channel index: channel -> [poll_ids]
        self._channel_index: Dict[str, List[str]] = defaultdict(list)

        # Voter index for duplicate prevention: (poll_id, voter_id) -> True
        self._voter_index: Dict[Tuple[str, str], bool] = {}

        # Event callbacks
        self._vote_callbacks: List[Any] = []
        self._close_callbacks: List[Any] = []
        self._expire_callbacks: List[Any] = []

    # ── Poll Creation ────────────────────────────────────────────────────

    def create(
        self,
        question: str,
        options: List[str],
        creator: str,
        channel: str,
        *,
        is_anonymous: bool = False,
        is_multi_choice: bool = False,
        expires_at: Optional[str] = None,
        expires_in: Optional[float] = None,
        max_votes_per_voter: Optional[int] = None,
    ) -> Poll:
        """
        Create a new poll.

        Parameters
        ----------
        question:
            The poll question.
        options:
            List of option text strings.
        creator:
            User ID of the poll creator.
        channel:
            Channel or group where the poll lives.
        is_anonymous:
            Whether to hide voter identities. Default False.
        is_multi_choice:
            Whether voters can select multiple options. Default False.
        expires_at:
            ISO-8601 timestamp for expiration. Overrides expires_in.
        expires_in:
            Seconds until expiration. Overrides default_expires_in.
        max_votes_per_voter:
            Max votes a single voter can cast. Default 1 (single choice).

        Returns
        -------
        Poll
            The newly created poll.

        Raises
        ------
        ValueError
            If validation fails (empty question, too few/many options, etc.).
        """
        # Validate question
        question = question.strip()
        if not question:
            raise ValueError("Poll question cannot be empty")
        if len(question) > self._max_question_length:
            raise ValueError(
                f"Question exceeds max length ({self._max_question_length} characters)"
            )

        # Validate options
        if len(options) < self._min_options:
            raise ValueError(
                f"Poll must have at least {self._min_options} options"
            )
        if len(options) > self._max_options:
            raise ValueError(
                f"Poll cannot have more than {self._max_options} options"
            )

        cleaned_options: List[str] = []
        seen_texts: Set[str] = set()
        for opt in options:
            text = opt.strip()
            if not text:
                raise ValueError("Poll option text cannot be empty")
            if len(text) > self._max_option_length:
                raise ValueError(
                    f"Option exceeds max length ({self._max_option_length} characters)"
                )
            text_lower = text.lower()
            if text_lower in seen_texts:
                raise ValueError(f"Duplicate option text: '{text}'")
            seen_texts.add(text_lower)
            cleaned_options.append(text)

        # Compute expiration
        if expires_at is None:
            expiry_seconds = (
                expires_in
                if expires_in is not None
                else self._default_expires_in
            )
            if expiry_seconds is not None:
                expiry_dt = (
                    datetime.now(timezone.utc) + timedelta(seconds=expiry_seconds)
                )
                expires_at = expiry_dt.isoformat()

        # Determine max votes per voter
        if max_votes_per_voter is None:
            max_votes_per_voter = len(cleaned_options) if is_multi_choice else 1
        elif max_votes_per_voter < 1:
            max_votes_per_voter = 1

        # Build PollOption list
        poll_options = [
            PollOption(id=uuid.uuid4().hex[:8], text=text)
            for text in cleaned_options
        ]

        # Create poll
        poll_id = uuid.uuid4().hex[:12]
        poll = Poll(
            id=poll_id,
            question=question,
            options=poll_options,
            created_by=creator,
            channel=channel,
            expires_at=expires_at,
            is_anonymous=is_anonymous,
            is_multi_choice=is_multi_choice,
            max_votes_per_voter=max_votes_per_voter,
        )

        # Store
        self._polls[poll_id] = poll
        self._channel_index[channel].append(poll_id)

        logger.info(
            "Created poll %s in channel '%s': '%s' (%d options, creator=%s)",
            poll_id, channel, question[:60], len(poll_options), creator,
        )

        return poll

    # ── Voting ───────────────────────────────────────────────────────────

    def vote(
        self,
        poll_id: str,
        voter_id: str,
        option_ids: Optional[List[str]] = None,
        option_indices: Optional[List[int]] = None,
    ) -> Poll:
        """
        Cast vote(s) on a poll.

        Can specify options by ID (option_ids) or by index (option_indices).
        At least one must be provided.

        Parameters
        ----------
        poll_id:
            The poll to vote on.
        voter_id:
            User ID of the voter.
        option_ids:
            Option IDs to vote for.
        option_indices:
            Option indices to vote for (0-based).

        Returns
        -------
        Poll
            The updated poll.

        Raises
        ------
        PollNotFoundError
            If the poll does not exist.
        PollClosedError
            If the poll is closed or expired.
        DuplicateVoteError
            If the voter has already voted.
        InvalidVoteError
            If the vote parameters are invalid.
        """
        poll = self._polls.get(poll_id)
        if poll is None:
            raise PollNotFoundError(f"Poll not found: {poll_id}")

        # Check status
        if poll.status == PollStatus.CLOSED:
            raise PollClosedError(f"Poll {poll_id} has been closed")
        if poll.is_expired:
            poll.status = PollStatus.EXPIRED
            raise PollClosedError(f"Poll {poll_id} has expired")

        # Check for duplicate vote
        voter_key = (poll_id, voter_id)
        if voter_key in self._voter_index:
            raise DuplicateVoteError(
                f"Voter {voter_id} has already voted on poll {poll_id}"
            )

        # Resolve option IDs
        target_options: List[PollOption] = []
        if option_ids:
            option_id_set = set(option_ids)
            for opt in poll.options:
                if opt.id in option_id_set:
                    target_options.append(opt)
                    option_id_set.discard(opt.id)
            if option_id_set:
                raise InvalidVoteError(
                    f"Unknown option IDs: {option_id_set}"
                )

        if option_indices is not None:
            for idx in option_indices:
                if idx < 0 or idx >= len(poll.options):
                    raise InvalidVoteError(
                        f"Option index out of range: {idx}"
                    )
                opt = poll.options[idx]
                if opt not in target_options:
                    target_options.append(opt)

        if not target_options:
            raise InvalidVoteError("No valid options specified for vote")

        # Check multi-choice constraints
        if not poll.is_multi_choice and len(target_options) > 1:
            raise InvalidVoteError(
                f"Poll is single-choice; got {len(target_options)} selections"
            )

        if len(target_options) > poll.max_votes_per_voter:
            raise InvalidVoteError(
                f"Exceeds max votes per voter ({poll.max_votes_per_voter})"
            )

        # Record votes
        for opt in target_options:
            opt.votes += 1
            opt.voter_ids.add(voter_id if not poll.is_anonymous else "__anon__")

        # Mark voter as having voted
        self._voter_index[voter_key] = True

        logger.info(
            "Vote recorded on poll %s by %s (%d option(s))",
            poll_id, voter_id, len(target_options),
        )

        # Fire callbacks
        for cb in self._vote_callbacks:
            try:
                cb(poll_id, voter_id, [opt.id for opt in target_options])
            except Exception as e:
                logger.error("Vote callback error: %s", e)

        return poll

    # ── Poll Lifecycle ───────────────────────────────────────────────────

    def close(self, poll_id: str) -> Poll:
        """
        Close a poll, preventing further votes.

        Parameters
        ----------
        poll_id:
            The poll to close.

        Returns
        -------
        Poll
            The closed poll.

        Raises
        ------
        PollNotFoundError
            If the poll does not exist.
        """
        poll = self._polls.get(poll_id)
        if poll is None:
            raise PollNotFoundError(f"Poll not found: {poll_id}")

        if poll.status != PollStatus.ACTIVE:
            logger.warning(
                "Attempted to close non-active poll %s (status=%s)",
                poll_id, poll.status.value,
            )
            return poll

        poll.status = PollStatus.CLOSED
        logger.info("Poll %s closed", poll_id)

        for cb in self._close_callbacks:
            try:
                cb(poll_id)
            except Exception as e:
                logger.error("Close callback error: %s", e)

        return poll

    def delete(self, poll_id: str) -> bool:
        """
        Delete a poll entirely.

        Parameters
        ----------
        poll_id:
            The poll to delete.

        Returns
        -------
        bool
            True if the poll was found and deleted, False otherwise.
        """
        poll = self._polls.get(poll_id)
        if poll is None:
            return False

        # Remove from channel index
        channel_list = self._channel_index.get(poll.channel, [])
        if poll_id in channel_list:
            channel_list.remove(poll_id)
            if not channel_list:
                del self._channel_index[poll.channel]

        # Remove voter index entries
        voter_keys_to_remove = [
            key for key in self._voter_index if key[0] == poll_id
        ]
        for key in voter_keys_to_remove:
            del self._voter_index[key]

        # Remove poll
        del self._polls[poll_id]

        logger.info("Poll %s deleted", poll_id)
        return True

    # ── Results ──────────────────────────────────────────────────────────

    def get_results(self, poll_id: str) -> PollResults:
        """
        Calculate and return poll results with percentages.

        Parameters
        ----------
        poll_id:
            The poll to get results for.

        Returns
        -------
        PollResults
            Aggregated results with per-option percentages.

        Raises
        ------
        PollNotFoundError
            If the poll does not exist.
        """
        poll = self._polls.get(poll_id)
        if poll is None:
            raise PollNotFoundError(f"Poll not found: {poll_id}")

        total = poll.total_votes
        percentages: Dict[str, float] = {}
        options_data: List[Dict[str, Any]] = []

        for opt in poll.options:
            pct = (opt.votes / total * 100.0) if total > 0 else 0.0
            percentages[opt.id] = round(pct, 2)
            opt_dict = opt.to_dict()
            opt_dict["percentage"] = round(pct, 2)
            if poll.is_anonymous:
                opt_dict.pop("voter_ids", None)
            options_data.append(opt_dict)

        # Determine if tied
        sorted_by_votes = sorted(
            poll.options, key=lambda o: o.votes, reverse=True
        )
        is_tied = (
            len(sorted_by_votes) > 1
            and sorted_by_votes[0].votes > 0
            and sorted_by_votes[0].votes == sorted_by_votes[1].votes
        )

        leader = None
        if not is_tied and sorted_by_votes[0].votes > 0:
            leader_opt = sorted_by_votes[0]
            leader = {
                "id": leader_opt.id,
                "text": leader_opt.text,
                "votes": leader_opt.votes,
                "percentage": percentages[leader_opt.id],
            }

        return PollResults(
            poll_id=poll.id,
            question=poll.question,
            status=poll.status,
            options=options_data,
            total_votes=total,
            unique_voters=poll.unique_voters,
            created_at=poll.created_at,
            expires_at=poll.expires_at,
            percentages=percentages,
            is_tied=is_tied,
            leader=leader,
        )

    # ── Listing & Querying ───────────────────────────────────────────────

    def list_polls(
        self,
        channel: Optional[str] = None,
        status: Optional[PollStatus] = None,
        created_by: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Poll]:
        """
        List polls with optional filtering and pagination.

        Parameters
        ----------
        channel:
            Filter by channel. None returns all channels.
        status:
            Filter by poll status. None returns all statuses.
        created_by:
            Filter by creator user ID. None returns all.
        limit:
            Maximum results. Default 50.
        offset:
            Skip this many results. Default 0.

        Returns
        -------
        List[Poll]
            Matching polls sorted by creation time (newest first).
        """
        # Start with all polls or channel-scoped
        if channel:
            poll_ids = self._channel_index.get(channel, [])
            polls = [self._polls[pid] for pid in poll_ids if pid in self._polls]
        else:
            polls = list(self._polls.values())

        # Apply filters
        if status is not None:
            polls = [p for p in polls if p.status == status]
        if created_by is not None:
            polls = [p for p in polls if p.created_by == created_by]

        # Sort by creation time (newest first)
        polls.sort(key=lambda p: p.created_at, reverse=True)

        return polls[offset: offset + limit]

    def get_poll(self, poll_id: str) -> Optional[Poll]:
        """Get a poll by ID. Returns None if not found."""
        return self._polls.get(poll_id)

    # ── Auto-Expiration ──────────────────────────────────────────────────

    def check_expirations(self) -> List[Poll]:
        """
        Check all active polls and expire any that have passed their deadline.

        Returns
        -------
        List[Poll]
            Polls that were just expired.
        """
        now = datetime.now(timezone.utc)
        newly_expired: List[Poll] = []

        for poll in self._polls.values():
            if poll.status != PollStatus.ACTIVE:
                continue
            if poll.expires_at is None:
                continue
            try:
                exp = datetime.fromisoformat(poll.expires_at)
                if now > exp:
                    poll.status = PollStatus.EXPIRED
                    newly_expired.append(poll)
                    logger.info("Poll %s expired", poll.id)

                    for cb in self._expire_callbacks:
                        try:
                            cb(poll.id)
                        except Exception as e:
                            logger.error("Expire callback error: %s", e)

            except (ValueError, TypeError):
                continue

        return newly_expired

    # ── Event Callbacks ──────────────────────────────────────────────────

    def on_vote(self, callback: Any) -> None:
        """
        Register a callback for vote events.

        Callback signature: (poll_id: str, voter_id: str, option_ids: List[str])
        """
        self._vote_callbacks.append(callback)

    def on_close(self, callback: Any) -> None:
        """
        Register a callback for poll close events.

        Callback signature: (poll_id: str)
        """
        self._close_callbacks.append(callback)

    def on_expire(self, callback: Any) -> None:
        """
        Register a callback for poll expiration events.

        Callback signature: (poll_id: str)
        """
        self._expire_callbacks.append(callback)

    # ── Statistics ───────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return poll manager statistics."""
        polls = list(self._polls.values())
        return {
            "total_polls": len(polls),
            "active": sum(1 for p in polls if p.status == PollStatus.ACTIVE),
            "closed": sum(1 for p in polls if p.status == PollStatus.CLOSED),
            "expired": sum(1 for p in polls if p.status == PollStatus.EXPIRED),
            "total_votes_cast": sum(p.total_votes for p in polls),
            "unique_voters_total": len(self._voter_index),
            "channels": len(self._channel_index),
        }
