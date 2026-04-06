"""
Session Manager — Comprehensive session lifecycle management.

Manages the full lifecycle of conversation sessions including creation,
activation, deactivation, closing, and pruning. Supports multiple activation
modes, queue modes, concurrent session limits, and timeout management.
Thread-safe with asyncio locks for use in async environments.

Usage::

    from atlas.sessions.manager import SessionManager, ActivationMode

    mgr = SessionManager(max_concurrent=5)
    session = await mgr.create(
        agent_id="agent-1",
        channel="whatsapp",
        peer_id="+1234567890",
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .keys import SessionKeyDerivation
from .store import Session, SessionStatus, SessionStore

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class ActivationMode(Enum):
    """
    How sessions are activated for an agent.

    EXCLUSIVE:
        Activating a new session deactivates all others for that agent.
    SHARED:
        Multiple sessions can be active simultaneously.
    QUEUED:
        Sessions are queued and activated one at a time in order.
    """

    EXCLUSIVE = "exclusive"
    SHARED = "shared"
    QUEUED = "queued"


class QueueMode(Enum):
    """
    How queued sessions are ordered for activation.

    FIFO:
        First-in, first-out ordering.
    PRIORITY:
        Sessions with higher priority are activated first.
    ROUND_ROBIN:
        Sessions take turns in circular order.
    """

    FIFO = "fifo"
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"


class SessionError(Exception):
    """Base exception for session manager errors."""
    pass


class SessionNotFoundError(SessionError):
    """Raised when a requested session does not exist."""
    pass


class SessionLimitError(SessionError):
    """Raised when the maximum concurrent session limit is exceeded."""
    pass


class SessionAlreadyActiveError(SessionError):
    """Raised when attempting to activate an already-active session."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Session Queue Entry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class QueuedSession:
    """
    A session waiting in the activation queue.

    Attributes
    ----------
    priority:
        Lower number = higher priority. Used for PRIORITY queue mode.
    session_id:
        The session identifier.
    enqueued_at:
        Timestamp when the session was added to the queue.
    metadata:
        Optional metadata for queue processing.
    """

    priority: int = field(default=10)
    session_id: str = field(default="", compare=False)
    enqueued_at: float = field(default_factory=time.time, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)


# ──────────────────────────────────────────────────────────────────────────────
# Session Callbacks
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionCallbacks:
    """
    Optional callbacks for session lifecycle events.

    All callbacks are optional. If provided, they are invoked at the
    corresponding lifecycle event.
    """

    on_create: Optional[Callable[[Session], Awaitable[None]]] = None
    """Called after a new session is created."""

    on_activate: Optional[Callable[[Session], Awaitable[None]]] = None
    """Called when a session is activated."""

    on_deactivate: Optional[Callable[[Session], Awaitable[None]]] = None
    """Called when a session is deactivated."""

    on_close: Optional[Callable[[Session], Awaitable[None]]] = None
    """Called when a session is closed."""

    on_timeout: Optional[Callable[[Session], Awaitable[None]]] = None
    """Called when a session times out."""

    on_prune: Optional[Callable[[Session], Awaitable[None]]] = None
    """Called when a session is pruned."""


# ──────────────────────────────────────────────────────────────────────────────
# Session Manager
# ──────────────────────────────────────────────────────────────────────────────

class SessionManager:
    """
    Comprehensive session lifecycle manager.

    Manages creation, activation, deactivation, closing, and pruning of
    conversation sessions. Supports configurable activation modes, queue
    modes, concurrent session limits, and timeout management.

    Thread-safe with asyncio locks for concurrent access.

    Parameters
    ----------
    max_concurrent:
        Maximum concurrent active sessions per agent (default 10).
    activation_mode:
        How sessions are activated (default ``SHARED``).
    queue_mode:
        How queued sessions are ordered (default ``FIFO``).
    session_timeout:
        Default session timeout in seconds (default 3600 = 1 hour).
    data_dir:
        Directory for persistent session storage.
    backend:
        Storage backend (``"json"`` or ``"sqlite"``).
    callbacks:
        Optional lifecycle callbacks.
    auto_save:
        If ``True`` (default), automatically persist sessions on changes.

    Example
    -------
    >>> import asyncio
    >>> async def main():
    ...     mgr = SessionManager(max_concurrent=5)
    ...     s = await mgr.create("agent-1", "whatsapp", "+1234")
    ...     print(s.id)
    >>> asyncio.run(main())  # doctest: +SKIP
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        activation_mode: ActivationMode = ActivationMode.SHARED,
        queue_mode: QueueMode = QueueMode.FIFO,
        session_timeout: int = 3600,
        data_dir: str | Path = Path.home() / ".claude_clone" / "atlas_sessions",
        backend: str = "json",
        callbacks: Optional[SessionCallbacks] = None,
        auto_save: bool = True,
        key_derivation: Optional[SessionKeyDerivation] = None,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.activation_mode = activation_mode
        self.queue_mode = queue_mode
        self.session_timeout = session_timeout
        self._callbacks = callbacks or SessionCallbacks()
        self._auto_save = auto_save
        self._key_derivation = key_derivation or SessionKeyDerivation()

        # In-memory session index: session_id -> Session
        self._sessions: Dict[str, Session] = {}

        # Active session index: agent_id -> list of active session_ids
        self._active_by_agent: Dict[str, List[str]] = {}

        # Activation queue per agent: agent_id -> list of QueuedSession
        self._queues: Dict[str, List[QueuedSession]] = {}

        # Round-robin state per agent: agent_id -> current index
        self._rr_index: Dict[str, int] = {}

        # Thread safety
        self._lock = asyncio.Lock()

        # Persistent storage
        self._store = SessionStore(
            backend=backend,
            data_dir=data_dir,
            auto_save=auto_save,
        )

        # Load existing sessions from storage
        self._load_existing()

        # Timeout checker task
        self._timeout_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            "SessionManager initialized (mode=%s, queue=%s, max=%d, timeout=%ds)",
            self.activation_mode.value,
            self.queue_mode.value,
            self.max_concurrent,
            self.session_timeout,
        )

    # ── Lifecycle Management ───────────────────────────────────────────

    async def create(
        self,
        agent_id: str,
        channel: str,
        peer_id: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        session_id: Optional[str] = None,
        priority: int = 10,
    ) -> Session:
        """
        Create a new session.

        If the activation mode is ``EXCLUSIVE`` or ``QUEUED``, the new
        session may deactivate or queue existing active sessions.

        Parameters
        ----------
        agent_id:
            The agent's identifier.
        channel:
            The messaging channel type.
        peer_id:
            The remote peer's identifier.
        metadata:
            Optional metadata to attach to the session.
        title:
            Optional human-readable title.
        session_id:
            Optional explicit session ID. If not provided, one is derived
            using the key derivation system.
        priority:
            Queue priority for QUEUED mode (lower = higher priority).

        Returns
        -------
        Session
            The newly created session.

        Raises
        ------
        SessionLimitError
            If the total session limit would be exceeded.
        """
        async with self._lock:
            # Check if we already have a session for this agent+channel+peer
            sid = session_id or self._key_derivation.derive(channel, agent_id, peer_id)
            existing = self._sessions.get(sid)

            if existing and existing.status_enum != SessionStatus.CLOSED:
                logger.debug(
                    "Existing session found for %s/%s/%s: %s",
                    agent_id, channel, peer_id, sid,
                )
                # Update the existing session
                existing.touch()
                existing.set_status(SessionStatus.ACTIVE)
                if metadata:
                    existing.metadata.update(metadata)
                if title:
                    existing.title = title
                self._save(existing)
                return existing

            # Check concurrent limit
            agent_sessions = self._get_agent_sessions(agent_id)
            active_count = sum(
                1 for s in agent_sessions
                if s.status_enum in (SessionStatus.ACTIVE, SessionStatus.INACTIVE)
            )
            if active_count >= self.max_concurrent:
                raise SessionLimitError(
                    f"Agent {agent_id!r} has reached the maximum of "
                    f"{self.max_concurrent} concurrent sessions"
                )

            # Create new session
            session = Session(
                id=sid,
                agent_id=agent_id,
                channel=channel,
                peer_id=peer_id,
                metadata=metadata or {},
                title=title,
            )

            self._sessions[sid] = session
            self._save(session)

            # Handle activation based on mode
            if self.activation_mode == ActivationMode.EXCLUSIVE:
                await self._deactivate_all_for_agent(agent_id, exclude=sid)
                self._add_active(agent_id, sid)
            elif self.activation_mode == ActivationMode.SHARED:
                self._add_active(agent_id, sid)
            elif self.activation_mode == ActivationMode.QUEUED:
                await self._enqueue(agent_id, sid, priority)

            # Invoke callback
            if self._callbacks.on_create:
                try:
                    await self._callbacks.on_create(session)
                except Exception:
                    logger.exception("Error in on_create callback")

            logger.info(
                "Session created: %s (agent=%s, channel=%s, peer=%s)",
                sid, agent_id, channel, peer_id,
            )
            return session

    async def get(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by ID.

        Parameters
        ----------
        session_id:
            The session identifier.

        Returns
        -------
        Session or None
            The session, or ``None`` if not found.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session

            # Try loading from persistent storage
            session = self._store.load(session_id)
            if session:
                self._sessions[session_id] = session
            return session

    async def close(
        self,
        session_id: str,
        *,
        archive: bool = False,
    ) -> bool:
        """
        Close a session.

        Parameters
        ----------
        session_id:
            The session identifier.
        archive:
            If ``True``, archive the session instead of marking it closed.

        Returns
        -------
        bool
            ``True`` if the session was found and closed.

        Raises
        ------
        SessionNotFoundError
            If the session does not exist.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f"Session {session_id!r} not found")

            # Remove from active index
            agent_id = session.agent_id
            active_list = self._active_by_agent.get(agent_id, [])
            if session_id in active_list:
                active_list.remove(session_id)

            # Remove from queue
            if agent_id in self._queues:
                self._queues[agent_id] = [
                    q for q in self._queues[agent_id]
                    if q.session_id != session_id
                ]

            # Update status
            if archive:
                session.set_status(SessionStatus.ARCHIVED)
            else:
                session.set_status(SessionStatus.CLOSED)

            self._save(session)

            # Activate next in queue if applicable
            if self.activation_mode == ActivationMode.QUEUED and self._queues.get(agent_id):
                await self._activate_next_in_queue(agent_id)

            # Invoke callback
            callback = self._callbacks.on_close if not archive else None
            if callback:
                try:
                    await callback(session)
                except Exception:
                    logger.exception("Error in on_close callback")

            logger.info("Session %s closed (archived=%s)", session_id, archive)
            return True

    async def activate(self, session_id: str) -> Session:
        """
        Activate a session.

        Parameters
        ----------
        session_id:
            The session identifier.

        Returns
        -------
        Session
            The activated session.

        Raises
        ------
        SessionNotFoundError
            If the session does not exist.
        SessionAlreadyActiveError
            If the session is already active and mode is EXCLUSIVE.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f"Session {session_id!r} not found")

            agent_id = session.agent_id
            active_list = self._active_by_agent.get(agent_id, [])

            if session_id in active_list:
                if self.activation_mode == ActivationMode.EXCLUSIVE:
                    raise SessionAlreadyActiveError(
                        f"Session {session_id!r} is already active"
                    )
                return session

            # Handle activation based on mode
            if self.activation_mode == ActivationMode.EXCLUSIVE:
                await self._deactivate_all_for_agent(agent_id, exclude=session_id)

            session.set_status(SessionStatus.ACTIVE)
            self._add_active(agent_id, session_id)
            self._save(session)

            # Invoke callback
            if self._callbacks.on_activate:
                try:
                    await self._callbacks.on_activate(session)
                except Exception:
                    logger.exception("Error in on_activate callback")

            logger.info("Session %s activated", session_id)
            return session

    async def deactivate(self, session_id: str) -> bool:
        """
        Deactivate a session without closing it.

        Parameters
        ----------
        session_id:
            The session identifier.

        Returns
        -------
        bool
            ``True`` if the session was deactivated.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False

            agent_id = session.agent_id
            active_list = self._active_by_agent.get(agent_id, [])
            if session_id in active_list:
                active_list.remove(session_id)

            session.set_status(SessionStatus.INACTIVE)
            self._save(session)

            # Activate next in queue
            if self.activation_mode == ActivationMode.QUEUED and self._queues.get(agent_id):
                await self._activate_next_in_queue(agent_id)

            # Invoke callback
            if self._callbacks.on_deactivate:
                try:
                    await self._callbacks.on_deactivate(session)
                except Exception:
                    logger.exception("Error in on_deactivate callback")

            logger.info("Session %s deactivated", session_id)
            return True

    # ── Query Operations ───────────────────────────────────────────────

    def list_sessions(
        self,
        agent_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        channel: Optional[str] = None,
        limit: int = 100,
    ) -> List[Session]:
        """
        List sessions with optional filters.

        Parameters
        ----------
        agent_id:
            Filter by agent identifier.
        status:
            Filter by session status.
        channel:
            Filter by channel type.
        limit:
            Maximum number of sessions to return.

        Returns
        -------
        list[Session]
            Matching sessions ordered by most recently updated.
        """
        sessions = list(self._sessions.values())

        if agent_id:
            sessions = [s for s in sessions if s.agent_id == agent_id]
        if status:
            sessions = [s for s in sessions if s.status_enum == status]
        if channel:
            sessions = [s for s in sessions if s.channel == channel]

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at_dt, reverse=True)
        return sessions[:limit]

    def get_active(self, agent_id: str) -> Optional[Session]:
        """
        Get the currently active session for an agent.

        In EXCLUSIVE mode, returns the single active session.
        In SHARED mode, returns the most recently activated session.
        In QUEUED mode, returns the currently dequeued session.

        Parameters
        ----------
        agent_id:
            The agent identifier.

        Returns
        -------
        Session or None
        """
        active_ids = self._active_by_agent.get(agent_id, [])
        if not active_ids:
            return None

        # Return the most recently activated
        for sid in reversed(active_ids):
            session = self._sessions.get(sid)
            if session and session.status_enum == SessionStatus.ACTIVE:
                return session
        return None

    def get_active_all(self, agent_id: str) -> List[Session]:
        """
        Get all active sessions for an agent.

        Parameters
        ----------
        agent_id:
            The agent identifier.

        Returns
        -------
        list[Session]
        """
        active_ids = self._active_by_agent.get(agent_id, [])
        return [
            self._sessions[sid]
            for sid in active_ids
            if sid in self._sessions and self._sessions[sid].status_enum == SessionStatus.ACTIVE
        ]

    def get_queue(self, agent_id: str) -> List[QueuedSession]:
        """
        Get the activation queue for an agent.

        Parameters
        ----------
        agent_id:
            The agent identifier.

        Returns
        -------
        list[QueuedSession]
        """
        return list(self._queues.get(agent_id, []))

    # ── Maintenance ────────────────────────────────────────────────────

    async def prune(
        self,
        max_age_seconds: Optional[int] = None,
        *,
        status: Optional[SessionStatus] = SessionStatus.CLOSED,
    ) -> int:
        """
        Prune old or closed sessions from memory and storage.

        Parameters
        ----------
        max_age_seconds:
            Maximum age of sessions to keep. If ``None``, only removes
            sessions matching ``status``.
        status:
            Status of sessions to prune. Default: CLOSED.

        Returns
        -------
        int
            Number of sessions pruned.
        """
        async with self._lock:
            cutoff = max_age_seconds or 0
            now = time.time()
            pruned = 0

            to_remove: List[str] = []

            for sid, session in self._sessions.items():
                should_prune = False

                if max_age_seconds and session.age_seconds > max_age_seconds:
                    should_prune = True

                if status and session.status_enum == status:
                    should_prune = True

                if should_prune:
                    to_remove.append(sid)

            for sid in to_remove:
                session = self._sessions.pop(sid, None)
                if session:
                    # Invoke callback
                    if self._callbacks.on_prune:
                        try:
                            await self._callbacks.on_prune(session)
                        except Exception:
                            logger.exception("Error in on_prune callback")

                    self._store.delete(sid)
                    pruned += 1

            if pruned:
                logger.info("Pruned %d sessions", pruned)
            return pruned

    async def check_timeouts(self) -> List[Session]:
        """
        Check for timed-out sessions and deactivate them.

        A session times out if it has been idle for longer than
        ``session_timeout`` seconds.

        Returns
        -------
        list[Session]
            Sessions that were timed out.
        """
        async with self._lock:
            timed_out: List[Session] = []

            for sid, session in self._sessions.items():
                if session.status_enum != SessionStatus.ACTIVE:
                    continue
                if session.idle_seconds > self.session_timeout:
                    timed_out.append(session)

            for session in timed_out:
                agent_id = session.agent_id
                active_list = self._active_by_agent.get(agent_id, [])
                if session.id in active_list:
                    active_list.remove(session.id)

                session.set_status(SessionStatus.INACTIVE)
                self._save(session)

                # Activate next in queue
                if self.activation_mode == ActivationMode.QUEUED:
                    if self._queues.get(agent_id):
                        await self._activate_next_in_queue(agent_id)

                if self._callbacks.on_timeout:
                    try:
                        await self._callbacks.on_timeout(session)
                    except Exception:
                        logger.exception("Error in on_timeout callback")

                logger.info(
                    "Session %s timed out (idle=%.0fs)",
                    session.id,
                    session.idle_seconds,
                )

            return timed_out

    # ── Session Updates ────────────────────────────────────────────────

    async def update_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Update a session's metadata.

        Parameters
        ----------
        session_id:
            The session identifier.
        metadata:
            Metadata to merge into the session.

        Returns
        -------
        bool
            ``True`` if the session was found and updated.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.metadata.update(metadata)
            session.touch()
            self._save(session)
            return True

    async def record_message(
        self,
        session_id: str,
        token_count: int = 0,
    ) -> bool:
        """
        Record a message event on a session.

        Updates the message count, token count, and last activity timestamp.

        Parameters
        ----------
        session_id:
            The session identifier.
        token_count:
            Estimated tokens for this message.

        Returns
        -------
        bool
            ``True`` if the session was found and updated.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.increment_messages()
            if token_count:
                session.add_tokens(token_count)
            self._save(session)
            return True

    async def set_title(
        self,
        session_id: str,
        title: str,
    ) -> bool:
        """
        Set a session's title.

        Parameters
        ----------
        session_id:
            The session identifier.
        title:
            The new title.

        Returns
        -------
        bool
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.title = title
            session.touch()
            self._save(session)
            return True

    # ── Background Task ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background timeout checker."""
        if self._running:
            return
        self._running = True
        self._timeout_task = asyncio.create_task(self._timeout_loop())
        logger.info("SessionManager timeout checker started")

    async def stop(self) -> None:
        """Stop the background timeout checker."""
        self._running = False
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
        logger.info("SessionManager timeout checker stopped")

    # ── Statistics ─────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """
        Return session manager statistics.

        Returns
        -------
        dict
            Keys: ``total``, ``active``, ``inactive``, ``closed``,
            ``archived``, ``by_agent``, ``by_channel``, ``queued``.
        """
        status_counts: Dict[str, int] = {}
        agent_counts: Dict[str, int] = {}
        channel_counts: Dict[str, int] = {}
        total_queued = 0

        for session in self._sessions.values():
            status_counts[session.status] = status_counts.get(session.status, 0) + 1
            agent_counts[session.agent_id] = agent_counts.get(session.agent_id, 0) + 1
            channel_counts[session.channel] = channel_counts.get(session.channel, 0) + 1

        for agent_id, queue in self._queues.items():
            total_queued += len(queue)

        return {
            "total": len(self._sessions),
            "active": status_counts.get("active", 0),
            "inactive": status_counts.get("inactive", 0),
            "closed": status_counts.get("closed", 0),
            "archived": status_counts.get("archived", 0),
            "by_agent": agent_counts,
            "by_channel": channel_counts,
            "queued": total_queued,
            "activation_mode": self.activation_mode.value,
            "queue_mode": self.queue_mode.value,
            "max_concurrent": self.max_concurrent,
            "session_timeout": self.session_timeout,
        }

    # ── Internal: Queue Management ─────────────────────────────────────

    async def _enqueue(
        self,
        agent_id: str,
        session_id: str,
        priority: int = 10,
    ) -> None:
        """Add a session to the activation queue."""
        if agent_id not in self._queues:
            self._queues[agent_id] = []

        entry = QueuedSession(
            priority=priority,
            session_id=session_id,
        )

        if self.queue_mode == QueueMode.PRIORITY:
            # Insert in sorted position
            self._queues[agent_id].append(entry)
            self._queues[agent_id].sort(key=lambda q: q.priority)
        elif self.queue_mode == QueueMode.ROUND_ROBIN:
            self._queues[agent_id].append(entry)
        else:  # FIFO
            self._queues[agent_id].append(entry)

        logger.debug(
            "Session %s queued for agent %s (priority=%d, queue_size=%d)",
            session_id, agent_id, priority, len(self._queues[agent_id]),
        )

    async def _activate_next_in_queue(self, agent_id: str) -> Optional[Session]:
        """Activate the next session in the queue."""
        queue = self._queues.get(agent_id, [])
        if not queue:
            return None

        # Check if we can activate more sessions
        active_count = len(self._active_by_agent.get(agent_id, []))
        if active_count >= self.max_concurrent:
            return None

        if self.queue_mode == QueueMode.ROUND_ROBIN:
            idx = self._rr_index.get(agent_id, 0)
            if idx >= len(queue):
                idx = 0
            entry = queue.pop(idx)
            self._rr_index[agent_id] = idx
        else:
            entry = queue.pop(0)

        session = self._sessions.get(entry.session_id)
        if session is None or session.status_enum == SessionStatus.CLOSED:
            # Skip invalid entries, try next
            if queue:
                return await self._activate_next_in_queue(agent_id)
            return None

        session.set_status(SessionStatus.ACTIVE)
        self._add_active(agent_id, session.id)
        self._save(session)

        if self._callbacks.on_activate:
            try:
                await self._callbacks.on_activate(session)
            except Exception:
                logger.exception("Error in on_activate callback")

        logger.info(
            "Session %s activated from queue for agent %s",
            session.id, agent_id,
        )
        return session

    async def _deactivate_all_for_agent(
        self,
        agent_id: str,
        exclude: Optional[str] = None,
    ) -> int:
        """Deactivate all active sessions for an agent, optionally excluding one."""
        active_ids = self._active_by_agent.get(agent_id, [])
        deactivated = 0

        for sid in list(active_ids):
            if sid == exclude:
                continue
            session = self._sessions.get(sid)
            if session:
                session.set_status(SessionStatus.INACTIVE)
                self._save(session)

                if self._callbacks.on_deactivate:
                    try:
                        await self._callbacks.on_deactivate(session)
                    except Exception:
                        logger.exception("Error in on_deactivate callback")

                deactivated += 1

        self._active_by_agent[agent_id] = [
            sid for sid in active_ids if sid == exclude
        ]
        return deactivated

    def _add_active(self, agent_id: str, session_id: str) -> None:
        """Add a session to the active list for an agent."""
        if agent_id not in self._active_by_agent:
            self._active_by_agent[agent_id] = []
        if session_id not in self._active_by_agent[agent_id]:
            self._active_by_agent[agent_id].append(session_id)

    def _get_agent_sessions(self, agent_id: str) -> List[Session]:
        """Get all sessions for a given agent."""
        return [
            s for s in self._sessions.values()
            if s.agent_id == agent_id
        ]

    def _save(self, session: Session) -> None:
        """Save a session to persistent storage."""
        if self._auto_save:
            self._store.save(session)

    def _load_existing(self) -> None:
        """Load existing sessions from persistent storage."""
        try:
            existing = self._store.list_all()
            for session in existing:
                self._sessions[session.id] = session

                if session.status_enum == SessionStatus.ACTIVE:
                    self._add_active(session.agent_id, session.id)

            if existing:
                logger.info("Loaded %d existing sessions from storage", len(existing))
        except Exception:
            logger.exception("Failed to load existing sessions")

    async def _timeout_loop(self) -> None:
        """Background loop that periodically checks for timed-out sessions."""
        check_interval = max(self.session_timeout // 10, 30)
        while self._running:
            try:
                await asyncio.sleep(check_interval)
                if not self._running:
                    break
                timed_out = await self.check_timeouts()
                if timed_out:
                    logger.info(
                        "%d sessions timed out in this check cycle",
                        len(timed_out),
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in timeout check loop")
