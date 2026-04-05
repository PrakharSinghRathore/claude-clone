"""
Title Generator — Auto-generates conversation titles using a cheaper/faster model.

Uses an auxiliary model (typically Claude Haiku or GPT-4o Mini) to generate
concise, descriptive titles from the first few messages of a conversation.
Titles are used for session labeling and history display.

Usage
-----
    generator = TitleGenerator(api_key="sk-...", model="claude-3-5-haiku-20241022")
    title = await generator.generate(
        messages=[
            {"role": "user", "content": "How do I deploy a FastAPI app?"},
            {"role": "assistant", "content": "Here's how to deploy..."},
        ]
    )
    # title ≈ "FastAPI Deployment Guide"
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from hermes.constants import MAX_TITLE_LENGTH, MIN_TITLE_LENGTH, TITLE_GENERATION_MAX_TOKENS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Title Generator
# ──────────────────────────────────────────────────────────────────────────────

class TitleGenerator:
    """
    Generates concise conversation titles from message history.

    Uses a smaller, cheaper model via the ``AuxiliaryClient`` to generate
    titles without consuming expensive tokens from the primary model.

    Parameters
    ----------
    api_key:
        API key for the model provider.
    model:
        Model name to use for title generation (should be a fast/cheap model).
    base_url:
        Optional custom API base URL.
    provider:
        API provider (``"anthropic"``, ``"openai"``, or ``"openrouter"``).
    max_length:
        Maximum title length in characters (default: 60).
    min_length:
        Minimum title length in characters (default: 10).
    max_tokens:
        Maximum tokens for the generation request (default: 30).
    fallback_fn:
        Optional sync callable that takes messages and returns a title string.
        Used when the model API is unavailable.
    """

    DEFAULT_MODEL = "claude-3-5-haiku-20241022"
    TITLE_SYSTEM_PROMPT = (
        "You are a title generator. Given a conversation, generate a single, "
        "concise title (3-8 words) that captures the main topic or purpose. "
        "Output ONLY the title text, with no quotes, prefixes, or extra text. "
        "Use title case for the first letter of each major word."
    )

    def __init__(
        self,
        api_key: str = "",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openrouter",
        max_length: int = MAX_TITLE_LENGTH,
        min_length: int = MIN_TITLE_LENGTH,
        max_tokens: int = TITLE_GENERATION_MAX_TOKENS,
        fallback_fn: Optional[Any] = None,
    ) -> None:
        self._api_key = api_key
        self._model = model or self.DEFAULT_MODEL
        self._base_url = base_url
        self._provider = provider
        self._max_length = max_length
        self._min_length = min_length
        self._max_tokens = max_tokens
        self._fallback_fn = fallback_fn
        self._client: Any = None

    # ── Public API ────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 2,
    ) -> str:
        """
        Generate a title from conversation messages.

        Parameters
        ----------
        messages:
            List of message dicts with ``role`` and ``content`` keys.
        max_retries:
            Number of retries on API failure before falling back.

        Returns
        -------
        str
            A concise conversation title. Falls back to an extractive
            summary if the API call fails.
        """
        if not messages:
            return "New Conversation"

        # Extract conversation text
        conversation_text = self._extract_conversation_text(messages)

        if not conversation_text.strip():
            return "New Conversation"

        # Try the model-based approach
        for attempt in range(max_retries + 1):
            try:
                title = await self._generate_with_model(conversation_text)
                if title and self._validate_title(title):
                    return self._clean_title(title)
            except Exception as e:
                logger.debug(
                    "Title generation attempt %d failed: %s", attempt + 1, e
                )

        # Try fallback function
        if self._fallback_fn is not None:
            try:
                title = self._fallback_fn(messages)
                if title and self._validate_title(title):
                    return self._clean_title(title)
            except Exception as e:
                logger.debug("Fallback title generation failed: %s", e)

        # Extractive fallback
        return self._extractive_title(messages)

    async def generate_from_query(self, query: str) -> str:
        """
        Generate a title from a single user query.

        Parameters
        ----------
        query:
            The user's first message.

        Returns
        -------
        str
            A concise title derived from the query.
        """
        messages = [{"role": "user", "content": query}]
        return await self.generate(messages)

    # ── Model-based generation ────────────────────────────────────────────

    async def _generate_with_model(self, conversation_text: str) -> str:
        """Generate a title using the configured model."""
        from hermes.core.auxiliary_client import AuxiliaryClient

        if self._client is None:
            self._client = AuxiliaryClient(
                api_key=self._api_key,
                model=self._model,
                base_url=self._base_url,
                provider=self._provider,
            )

        response = await self._client.complete(
            system_prompt=self.TITLE_SYSTEM_PROMPT,
            user_message=f"Generate a title for this conversation:\n\n{conversation_text[:2000]}",
            max_tokens=self._max_tokens,
            temperature=0.3,
        )

        return response.strip()

    # ── Text extraction ───────────────────────────────────────────────────

    @staticmethod
    def _extract_conversation_text(messages: List[Dict[str, str]]) -> str:
        """Extract a text summary from conversation messages."""
        parts: List[str] = []
        for msg in messages[:6]:  # First 6 messages max
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}: {content[:500]}")
            elif isinstance(content, list):
                # Handle complex content blocks
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", "")[:200])
                if texts:
                    parts.append(f"{role}: {' '.join(texts)}")
        return "\n".join(parts)

    # ── Extractive fallback ───────────────────────────────────────────────

    @staticmethod
    def _extractive_title(messages: List[Dict[str, str]]) -> str:
        """
        Generate a title without an API call by extracting key phrases.

        Uses heuristic extraction from the first user message.
        """
        # Get the first user message
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    return TitleGenerator._extract_title_from_text(content)

        return "New Conversation"

    @staticmethod
    def _extract_title_from_text(text: str) -> str:
        """Extract a concise title from a single text."""
        # Remove common prefixes
        text = re.sub(r"^(please|can you|could you|i want|i need|help)\s+", "", text, flags=re.IGNORECASE)
        text = text.strip().rstrip("?!.;,")

        # Take first meaningful phrase (up to 8 words)
        words = text.split()
        title_words = words[:8]
        title = " ".join(title_words)

        # Capitalize first letter
        if title:
            title = title[0].upper() + title[1:] if len(title) > 1 else title.upper()

        # Enforce length limits
        if len(title) > MAX_TITLE_LENGTH:
            title = title[:MAX_TITLE_LENGTH].rsplit(" ", 1)[0]

        return title or "New Conversation"

    # ── Validation and cleaning ───────────────────────────────────────────

    def _validate_title(self, title: str) -> bool:
        """Check if a generated title meets quality criteria."""
        if not title or not title.strip():
            return False
        if len(title) < self._min_length:
            return False
        if len(title) > self._max_length * 2:  # Allow some slack
            return False
        # Reject titles that are just the prompt echoed back
        if title.lower().startswith("here is") or title.lower().startswith("the title"):
            return False
        # Reject if it contains newlines (should be a single line)
        if "\n" in title:
            title = title.split("\n")[0]
        return True

    def _clean_title(self, title: str) -> str:
        """Clean and normalize a generated title."""
        # Remove surrounding quotes
        title = title.strip().strip('"').strip("'").strip("`")

        # Remove common prefixes that models sometimes add
        prefixes_to_strip = [
            "Title:", "title:", "Title -", "title -",
            "#", "##", "###", "-", "*",
        ]
        for prefix in prefixes_to_strip:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()

        # Remove trailing punctuation
        title = title.rstrip("?!.;:")

        # Enforce length
        if len(title) > self._max_length:
            # Try to cut at a word boundary
            title = title[:self._max_length].rsplit(" ", 1)[0]

        # Ensure non-empty
        if not title.strip():
            return "New Conversation"

        # Capitalize first letter
        title = title[0].upper() + title[1:]

        return title
