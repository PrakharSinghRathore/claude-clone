"""
Human-in-the-loop feedback for flows.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FeedbackStatus(str, Enum):
    """Status of human feedback."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class HumanFeedbackResult:
    """
    Result from human-in-the-loop feedback.

    Attributes:
        status: Whether the output was approved, rejected, or modified.
        feedback: Human-provided feedback text.
        modified_output: If modified, the corrected output.
        metadata: Additional metadata from the feedback process.
    """

    def __init__(
        self,
        status: FeedbackStatus = FeedbackStatus.PENDING,
        feedback: str = "",
        modified_output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.status = status
        self.feedback = feedback
        self.modified_output = modified_output
        self.metadata = metadata or {}

    @property
    def is_approved(self) -> bool:
        return self.status == FeedbackStatus.APPROVED

    @property
    def is_rejected(self) -> bool:
        return self.status == FeedbackStatus.REJECTED

    @property
    def is_modified(self) -> bool:
        return self.status == FeedbackStatus.MODIFIED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "feedback": self.feedback,
            "modified_output": self.modified_output,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"HumanFeedbackResult(status={self.status.value}, feedback={self.feedback!r:.50})"


def human_feedback(
    message: str = "Please review the output and provide feedback.",
    timeout: Optional[int] = None,
) -> HumanFeedbackResult:
    """
    Request human feedback during flow execution.

    This function pauses flow execution and requests input from a human
    reviewer. The flow will resume once feedback is provided.

    Args:
        message: The message to display to the human reviewer.
        timeout: Optional timeout in seconds. None means wait indefinitely.

    Returns:
        A HumanFeedbackResult with the reviewer's response.
    """
    logger.info("Human feedback requested: %s", message)

    print(f"\n{'='*60}")
    print("HUMAN FEEDBACK REQUIRED")
    print(f"{'='*60}")
    print(f"\n{message}\n")
    print("Options:")
    print("  1. Type your feedback and press Enter (to approve with notes)")
    print("  2. Type 'reject: <reason>' to reject")
    print("  3. Type 'modify: <corrected output>' to provide a correction")
    print("  4. Press Enter with empty input to approve")
    print(f"{'='*60}")

    try:
        response = input("\nYour feedback: ").strip()
    except (EOFError, KeyboardInterrupt):
        return HumanFeedbackResult(
            status=FeedbackStatus.APPROVED,
            feedback="Auto-approved due to interrupt",
        )

    if not response:
        return HumanFeedbackResult(
            status=FeedbackStatus.APPROVED,
            feedback="Approved without changes",
        )

    if response.lower().startswith("reject:"):
        reason = response[7:].strip()
        return HumanFeedbackResult(
            status=FeedbackStatus.REJECTED,
            feedback=reason,
        )

    if response.lower().startswith("modify:"):
        modified = response[7:].strip()
        return HumanFeedbackResult(
            status=FeedbackStatus.MODIFIED,
            feedback="Output modified by reviewer",
            modified_output=modified,
        )

    return HumanFeedbackResult(
        status=FeedbackStatus.APPROVED,
        feedback=response,
    )
