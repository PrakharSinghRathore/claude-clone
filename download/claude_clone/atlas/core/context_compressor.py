"""
Context Compressor — Token-aware conversation history compression.

Compresses long conversation histories to stay within model context limits
while preserving key information. Supports multiple strategies including
summarization-based compression, sliding window truncation, and smart
preservation of system messages, tool definitions, and recent context.

Integrates with tiktoken when available for accurate token counting, falling
back to a character-based heuristic otherwise.

Usage
-----
    compressor = ContextCompressor(max_context_tokens=180_000)
    compressed = await compressor.compress(
        messages=full_history,
        system_prompt=current_system_prompt,
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Token counting
# ──────────────────────────────────────────────────────────────────────────────

def _try_tiktoken() -> Any:
    """Attempt to import tiktoken; return the module or None."""
    try:
        import tiktoken  # type: ignore
        return tiktoken
    except ImportError:
        return None


# Lazy-loaded tokenizer singleton
_tiktoken_module: Any = None
_tiktoken_encoding: Any = None


def _get_tokenizer():
    """Get a tiktoken encoding (cl100k_base), or None if unavailable."""
    global _tiktoken_module, _tiktoken_encoding
    if _tiktoken_encoding is not None:
        return _tiktoken_encoding
    _tiktoken_module = _try_tiktoken()
    if _tiktoken_module is not None:
        try:
            _tiktoken_encoding = _tiktoken_module.get_encoding("cl100k_base")
        except Exception:
            logger.debug("tiktoken available but encoding load failed, using heuristic")
    return _tiktoken_encoding


def count_tokens(text: str) -> int:
    """
    Count tokens in a text string.

    Uses tiktoken (cl100k_base) when available, otherwise falls back to a
    character-based heuristic (~4 characters per token).

    Parameters
    ----------
    text:
        The text to count tokens for.

    Returns
    -------
    int
        Estimated token count.
    """
    if not text:
        return 0
    encoding = _get_tokenizer()
    if encoding is not None:
        try:
            return len(encoding.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def count_message_tokens(message: Dict[str, Any]) -> int:
    """
    Count the approximate tokens used by a single message dict.

    Handles both simple string content and complex content blocks (lists
    of dicts with ``type`` and ``text``/``input`` keys, as used by Anthropic
    and OpenAI APIs).

    Parameters
    ----------
    message:
        A message dict with ``role`` and ``content`` fields.

    Returns
    -------
    int
        Estimated token count for the message.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        tokens = count_tokens(content)
    elif isinstance(content, list):
        tokens = 0
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    tokens += count_tokens(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tokens += count_tokens(str(block.get("input", {})))
                elif block.get("type") == "tool_result":
                    tokens += count_tokens(str(block.get("content", "")))
                elif block.get("type") == "thinking":
                    tokens += count_tokens(block.get("thinking", ""))
                else:
                    tokens += count_tokens(str(block))
            else:
                tokens += count_tokens(str(block))
    else:
        tokens = count_tokens(str(content))
    # Account for role overhead (~4 tokens for the role + formatting)
    return tokens + 4


def count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Count total tokens across a list of messages."""
    return sum(count_message_tokens(m) for m in messages)


# ──────────────────────────────────────────────────────────────────────────────
# Compression strategies
# ──────────────────────────────────────────────────────────────────────────────

class CompressionStrategy(Enum):
    """Available compression strategies."""

    AUTO = auto()
    """Automatically choose the best strategy based on context."""

    SUMMARIZE = auto()
    """Summarize older turns while preserving recent context."""

    SLIDING_WINDOW = auto()
    """Keep only the most recent N turns within the token budget."""

    HYBRID = auto()
    """Summarize old turns, preserve recent turns, keep tool definitions."""


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    messages: List[Dict[str, Any]]
    """The compressed message list."""

    original_tokens: int
    """Token count before compression."""

    compressed_tokens: int
    """Token count after compression."""

    summary: str
    """Human-readable summary of what was compressed."""

    turns_removed: int = 0
    """Number of conversation turns that were removed or summarized."""

    strategy_used: CompressionStrategy = CompressionStrategy.AUTO
    """The strategy that was actually applied."""


# ──────────────────────────────────────────────────────────────────────────────
# ContextCompressor
# ──────────────────────────────────────────────────────────────────────────────

class ContextCompressor:
    """
    Token-aware conversation history compressor.

    Compresses long conversation histories to fit within a specified token
    budget while preserving critical information: system messages, tool
    definitions, and the most recent conversation turns.

    Parameters
    ----------
    max_context_tokens:
        Maximum total context tokens (default 200,000).
    reserve_tokens:
        Tokens to reserve for the system prompt and model response
        (default 8,000).
    preserve_recent_turns:
        Number of recent turns to always keep intact (default 6).
    summarize_fn:
        Optional async callable that takes a list of messages and returns
        a summary string. If ``None``, uses built-in extractive summarization.
    """

    def __init__(
        self,
        max_context_tokens: int = 200_000,
        reserve_tokens: int = 8_000,
        preserve_recent_turns: int = 6,
        summarize_fn: Optional[Any] = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.reserve_tokens = reserve_tokens
        self.preserve_recent_turns = preserve_recent_turns
        self._summarize_fn = summarize_fn

    @property
    def available_tokens(self) -> int:
        """Tokens available for conversation history."""
        return self.max_context_tokens - self.reserve_tokens

    # ── Public API ────────────────────────────────────────────────────────

    async def compress(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        strategy: CompressionStrategy = CompressionStrategy.AUTO,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> CompressionResult:
        """
        Compress a conversation history to fit within the context budget.

        Parameters
        ----------
        messages:
            The full conversation message list.
        system_prompt:
            The current system prompt (counted against the budget).
        strategy:
            Compression strategy to use. ``AUTO`` selects the best strategy.
        tools:
            Optional tool definitions that must be preserved.

        Returns
        -------
        CompressionResult
            Contains the compressed messages, token counts, and metadata.
        """
        system_tokens = count_tokens(system_prompt or "")
        tools_tokens = self._count_tools_tokens(tools or [])
        overhead = system_tokens + tools_tokens

        original_tokens = count_messages_tokens(messages)
        total_budget = self.max_context_tokens - overhead

        if original_tokens <= total_budget:
            return CompressionResult(
                messages=list(messages),
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                summary="No compression needed — within budget.",
                turns_removed=0,
                strategy_used=CompressionStrategy.AUTO,
            )

        # Choose strategy
        if strategy == CompressionStrategy.AUTO:
            strategy = self._choose_strategy(messages, total_budget)

        # Separate system messages and conversation messages
        system_msgs, conv_msgs = self._split_messages(messages)

        # Separate tool definitions from conversation
        tool_msgs: List[Dict[str, Any]] = []

        # Apply compression
        if strategy == CompressionStrategy.SLIDING_WINDOW:
            compressed_conv, turns_removed = self._sliding_window(conv_msgs, total_budget)
            summary = f"Sliding window: kept last {len(compressed_conv)} messages."
        elif strategy in (CompressionStrategy.SUMMARIZE, CompressionStrategy.HYBRID):
            compressed_conv, turns_removed, summary = await self._summarize(
                conv_msgs, total_budget, strategy
            )
        else:
            compressed_conv, turns_removed = self._sliding_window(conv_msgs, total_budget)
            summary = f"Fallback sliding window: kept {len(compressed_conv)} messages."

        # Reassemble: system messages + tool definitions + compressed conversation
        final_messages = system_msgs + compressed_conv
        compressed_tokens = count_messages_tokens(final_messages) + overhead

        return CompressionResult(
            messages=final_messages,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            summary=summary,
            turns_removed=turns_removed,
            strategy_used=strategy,
        )

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate total tokens for a message list."""
        return count_messages_tokens(messages)

    # ── Strategy selection ────────────────────────────────────────────────

    def _choose_strategy(
        self,
        messages: List[Dict[str, Any]],
        budget: int,
    ) -> CompressionStrategy:
        """Choose the best compression strategy based on the situation."""
        total = count_messages_tokens(messages)
        ratio = total / budget if budget > 0 else float("inf")

        if ratio < 1.5:
            # Mildly over budget — sliding window is fast and sufficient
            return CompressionStrategy.SLIDING_WINDOW
        elif ratio < 3.0:
            # Moderately over budget — summarization keeps more info
            return CompressionStrategy.SUMMARIZE
        else:
            # Significantly over budget — hybrid for maximum compression
            return CompressionStrategy.HYBRID

    # ── Message splitting ─────────────────────────────────────────────────

    @staticmethod
    def _split_messages(
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split messages into system messages and conversation messages.

        Returns
        -------
        tuple[list, list]
            (system_messages, conversation_messages)
        """
        system_msgs: List[Dict[str, Any]] = []
        conv_msgs: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                conv_msgs.append(msg)

        return system_msgs, conv_msgs

    # ── Sliding window ────────────────────────────────────────────────────

    def _sliding_window(
        self,
        messages: List[Dict[str, Any]],
        budget: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Keep the most recent messages that fit within the token budget.

        Always preserves the last ``preserve_recent_turns`` turns if possible.

        Returns
        -------
        tuple[list, int]
            (compressed_messages, turns_removed)
        """
        if not messages:
            return [], 0

        # Work backwards from the end
        result: List[Dict[str, Any]] = []
        used_tokens = 0
        preserved_recent = 0

        for msg in reversed(messages):
            msg_tokens = count_message_tokens(msg)
            if used_tokens + msg_tokens > budget:
                # But always try to preserve recent turns
                if preserved_recent < self.preserve_recent_turns:
                    result.insert(0, msg)
                    used_tokens += msg_tokens
                    preserved_recent += 1
                continue

            result.insert(0, msg)
            used_tokens += msg_tokens
            if msg.get("role") in ("user", "assistant"):
                preserved_recent += 1

        turns_removed = len(messages) - len(result)
        return result, max(0, turns_removed)

    # ── Summarization ─────────────────────────────────────────────────────

    async def _summarize(
        self,
        messages: List[Dict[str, Any]],
        budget: int,
        strategy: CompressionStrategy,
    ) -> Tuple[List[Dict[str, Any]], int, str]:
        """
        Summarize older messages while preserving recent turns.

        Parameters
        ----------
        messages:
            Conversation messages (no system messages).
        budget:
            Available token budget for the compressed output.
        strategy:
            SUMMARIZE or HYBRID.

        Returns
        -------
        tuple[list, int, str]
            (compressed_messages, turns_removed, summary_text)
        """
        if len(messages) <= self.preserve_recent_turns:
            return list(messages), 0, "Too few messages to summarize."

        # Split into old (summarizable) and recent (preserve)
        split_idx = max(0, len(messages) - self.preserve_recent_turns)
        old_messages = messages[:split_idx]
        recent_messages = messages[split_idx:]

        # Generate summary of old messages
        if self._summarize_fn is not None:
            try:
                summary_text = await self._summarize_fn(old_messages)
            except Exception as e:
                logger.warning("Custom summarize_fn failed: %s — using built-in", e)
                summary_text = self._extractive_summary(old_messages)
        else:
            summary_text = self._extractive_summary(old_messages)

        # Build a summary message
        summary_msg = {
            "role": "system",
            "content": (
                "[Previous conversation summary]\n"
                f"{summary_text}\n"
                f"[End of summary — {len(old_messages)} earlier messages compressed]"
            ),
        }

        # Check if summary + recent fits
        result = [summary_msg] + recent_messages
        result_tokens = count_messages_tokens(result)

        if result_tokens > budget:
            # Further compress by sliding window on recent messages
            trimmed, extra_removed = self._sliding_window(recent_messages, budget - count_message_tokens(summary_msg))
            result = [summary_msg] + trimmed
            turns_removed = len(messages) - len(result) + extra_removed
        else:
            turns_removed = len(old_messages)

        return result, turns_removed, f"Summarized {len(old_messages)} older turns into a summary."

    @staticmethod
    def _extractive_summary(messages: List[Dict[str, Any]]) -> str:
        """
        Built-in extractive summarization.

        Picks the most representative sentences from user and assistant
        messages without requiring an external LLM call.

        Parameters
        ----------
        messages:
            The messages to summarize.

        Returns
        -------
        str
            A concise summary string.
        """
        import re

        user_points: List[str] = []
        assistant_points: List[str] = []

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text blocks from complex content
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        texts.append(block)
                content = " ".join(texts)

            if not isinstance(content, str) or not content.strip():
                continue

            role = msg.get("role", "")
            # Take first 200 chars of each message
            snippet = content.strip()[:200]
            if role == "user":
                user_points.append(snippet)
            elif role == "assistant":
                assistant_points.append(snippet)

        parts: List[str] = []

        # Take up to 5 user points and 5 assistant points
        for point in user_points[-5:]:
            parts.append(f"User: {point}")
        for point in assistant_points[-5:]:
            parts.append(f"Assistant: {point}")

        summary = " | ".join(parts)
        max_len = 1500
        if len(summary) > max_len:
            summary = summary[:max_len] + "..."
        return summary if summary else "Earlier conversation not available for summarization."

    # ── Tool token counting ───────────────────────────────────────────────

    @staticmethod
    def _count_tools_tokens(tools: List[Dict[str, Any]]) -> int:
        """Estimate token count for tool definitions."""
        if not tools:
            return 0
        import json
        tools_json = json.dumps(tools, default=str)
        return count_tokens(tools_json)
