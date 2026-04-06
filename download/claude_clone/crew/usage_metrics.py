"""
Token usage tracking for crew executions.

Provides a Pydantic model for accumulating and summarising token usage
across multiple agent calls within a crew run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class UsageMetrics(BaseModel):
    """
    Cumulative token usage metrics for a crew execution.

    Individual agent calls produce usage data; this model aggregates them
    so the crew can report total cost and efficiency.

    Attributes:
        total_tokens:       Sum of prompt + completion tokens.
        prompt_tokens:      Total tokens sent as input (system + user + context).
        completion_tokens:  Total tokens generated as output.
        successful_requests: Number of API calls that completed without error.
        failed_requests:    Number of API calls that raised an exception.
    """

    total_tokens: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    successful_requests: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)

    model_config = {"frozen": False}

    def add(self, other: UsageMetrics) -> None:
        """
        Add another :class:`UsageMetrics` into this one (in-place).

        Args:
            other: Another usage metrics instance to merge.
        """
        self.total_tokens += other.total_tokens
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.successful_requests += other.successful_requests
        self.failed_requests += other.failed_requests

    @classmethod
    def from_api_response(cls, usage_dict: Optional[Dict[str, int]]) -> UsageMetrics:
        """
        Build a :class:`UsageMetrics` from a raw API response usage dict.

        Accepts common key names: ``input_tokens``, ``prompt_tokens``,
        ``output_tokens``, ``completion_tokens``.

        Args:
            usage_dict: Dictionary from an API response containing token
                counts, e.g. ``{"input_tokens": 100, "output_tokens": 50}``.

        Returns:
            A new :class:`UsageMetrics` instance.
        """
        if not usage_dict:
            return cls()

        prompt = (
            usage_dict.get("input_tokens")
            or usage_dict.get("prompt_tokens")
            or 0
        )
        completion = (
            usage_dict.get("output_tokens")
            or usage_dict.get("completion_tokens")
            or 0
        )

        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            successful_requests=1,
        )

    def to_dict(self) -> Dict[str, int]:
        """Return the metrics as a plain dictionary."""
        return self.model_dump()

    def summary(self) -> str:
        """Return a human-readable summary string."""
        total_requests = self.successful_requests + self.failed_requests
        lines = [
            f"Usage Summary",
            f"  Total tokens:       {self.total_tokens:,}",
            f"  Prompt tokens:      {self.prompt_tokens:,}",
            f"  Completion tokens:  {self.completion_tokens:,}",
            f"  Successful calls:   {self.successful_requests}",
            f"  Failed calls:       {self.failed_requests}",
            f"  Total requests:     {total_requests}",
        ]
        if self.total_tokens > 0:
            pct = (self.prompt_tokens / self.total_tokens) * 100
            lines.append(f"  Prompt/completion:  {pct:.1f}% / {100 - pct:.1f}%")
        return "\n".join(lines)

    def __add__(self, other: UsageMetrics) -> UsageMetrics:
        """Return a new instance with combined metrics."""
        result = self.model_copy()
        result.add(other)
        return result

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"UsageMetrics(total={self.total_tokens}, "
            f"prompt={self.prompt_tokens}, completion={self.completion_tokens})"
        )
