"""
Requests-per-minute rate limiter for crew agents.

Provides thread-safe RPM limiting to prevent exceeding API rate limits.
Uses a sliding window approach to track request timestamps.
"""

import threading
import time
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class RPMController:
    """
    Thread-safe requests-per-minute rate limiter.

    Uses a sliding window of request timestamps to enforce a maximum number
    of requests per minute. If the limit would be exceeded, ``check_or_wait``
    blocks until the oldest request falls outside the one-minute window.

    Args:
        max_rpm: Maximum number of requests allowed per minute. Set to
            ``None`` to disable rate limiting entirely.

    Example::

        rpm = RPMController(max_rpm=30)

        for task in tasks:
            rpm.check_or_wait()   # blocks if needed
            await execute(task)
    """

    def __init__(self, max_rpm: Optional[int] = None) -> None:
        self.max_rpm = max_rpm
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def check_or_wait(self) -> None:
        """
        Check if a new request can be made immediately.

        If the RPM limit would be exceeded, this method blocks (sleeps) until
        the oldest request timestamp falls outside the 60-second sliding window,
        freeing up a slot.

        When ``max_rpm`` is ``None``, this method returns immediately with no
        rate limiting.
        """
        if self.max_rpm is None:
            return

        with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0

            # Evict timestamps older than 60 seconds
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.max_rpm:
                # Calculate how long to wait for the oldest slot to free up
                sleep_time = self._timestamps[0] - cutoff + 0.05  # small buffer
                if sleep_time > 0:
                    logger.debug(
                        "RPM limit reached (%d/%d). Sleeping %.2fs",
                        len(self._timestamps), self.max_rpm, sleep_time,
                    )
                    # Release the lock while sleeping
                    self._lock.release()
                    try:
                        time.sleep(sleep_time)
                    finally:
                        self._lock.acquire()
                    # After waking up, prune again
                    now = time.monotonic()
                    cutoff = now - 60.0
                    while self._timestamps and self._timestamps[0] <= cutoff:
                        self._timestamps.popleft()

            # Record this request
            self._timestamps.append(time.monotonic())

    @property
    def current_rpm(self) -> int:
        """Return the number of requests made in the last 60 seconds."""
        with self._lock:
            cutoff = time.monotonic() - 60.0
            return sum(1 for ts in self._timestamps if ts > cutoff)

    @property
    def remaining(self) -> Optional[int]:
        """Return how many requests can still be made, or ``None`` if unlimited."""
        if self.max_rpm is None:
            return None
        with self._lock:
            cutoff = time.monotonic() - 60.0
            active = sum(1 for ts in self._timestamps if ts > cutoff)
            return max(0, self.max_rpm - active)

    def reset(self) -> None:
        """Clear all tracked timestamps."""
        with self._lock:
            self._timestamps.clear()
