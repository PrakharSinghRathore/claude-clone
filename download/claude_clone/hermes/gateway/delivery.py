"""
DeliveryRouter — Routes messages and outputs to appropriate platforms.

Handles multi-platform delivery with fallback, per-platform message
formatting (Markdown, HTML, plain text), media attachment handling,
delivery confirmation, and retry logic.

Usage::

    router = DeliveryRouter(config, session_store, adapters)
    await router.deliver(
        user_id="alice",
        platform="telegram",
        chat_id="12345",
        text="Hello, world!",
        attachments=["/path/to/file.pdf"],
    )
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hermes.gateway.config import GatewayConfig, PlatformConfig
from hermes.gateway.session import SessionContext, SessionStore

logger = logging.getLogger("hermes.gateway.delivery")


# ──────────────────────────────────────────────────────────────────────────────
# Message Format
# ──────────────────────────────────────────────────────────────────────────────

class MessageFormat:
    """Supported message formatting modes."""

    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"


# ──────────────────────────────────────────────────────────────────────────────
# Delivery Result
# ──────────────────────────────────────────────────────────────────────────────

class DeliveryResult:
    """Result of a message delivery attempt."""

    def __init__(
        self,
        success: bool,
        platform: str,
        chat_id: str,
        message_id: Optional[str] = None,
        error: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        self.success = success
        self.platform = platform
        self.chat_id = chat_id
        self.message_id = message_id
        self.error = error
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.delivery_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delivery_id": self.delivery_id,
            "success": self.success,
            "platform": self.platform,
            "chat_id": self.chat_id,
            "message_id": self.message_id,
            "error": self.error,
            "timestamp": self.timestamp,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Format Converters
# ──────────────────────────────────────────────────────────────────────────────

class FormatConverter:
    """Convert messages between Markdown, HTML, and plain text."""

    @staticmethod
    def markdown_to_html(text: str) -> str:
        """
        Convert Markdown to HTML.

        Handles basic Markdown: headers, bold, italic, code blocks,
        inline code, links, lists, and paragraphs.
        """
        lines = text.split("\n")
        html_lines: List[str] = []
        in_code_block = False
        in_list = False
        in_paragraph = False

        for line in lines:
            # Code blocks
            if line.startswith("```"):
                if in_code_block:
                    html_lines.append("</code></pre>")
                    in_code_block = False
                else:
                    lang = line[3:].strip()
                    html_lines.append(f'<pre><code class="language-{lang}">')
                    in_code_block = True
                in_paragraph = False
                continue

            if in_code_block:
                html_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
                continue

            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                if in_paragraph:
                    html_lines.append("</p>")
                    in_paragraph = False
                continue

            # Headers
            if stripped.startswith("# "):
                html_lines.append(f"<h1>{FormatConverter._inline_md_to_html(stripped[2:])}</h1>")
                continue
            elif stripped.startswith("## "):
                html_lines.append(f"<h2>{FormatConverter._inline_md_to_html(stripped[3:])}</h2>")
                continue
            elif stripped.startswith("### "):
                html_lines.append(f"<h3>{FormatConverter._inline_md_to_html(stripped[4:])}</h3>")
                continue
            elif stripped.startswith("#### "):
                html_lines.append(f"<h4>{FormatConverter._inline_md_to_html(stripped[5:])}</h4>")
                continue

            # Unordered lists
            if stripped.startswith("- ") or stripped.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                content = stripped[2:]
                html_lines.append(f"<li>{FormatConverter._inline_md_to_html(content)}</li>")
                continue

            # Ordered lists
            if stripped and stripped[0].isdigit() and ". " in stripped[:4]:
                if not in_list:
                    html_lines.append("<ol>")
                    in_list = True
                idx = stripped.index(". ")
                content = stripped[idx + 2:]
                html_lines.append(f"<li>{FormatConverter._inline_md_to_html(content)}</li>")
                continue

            if in_list:
                html_lines.append("</ul>")
                in_list = False

            # Paragraphs
            if not in_paragraph:
                html_lines.append("<p>")
                in_paragraph = True
            html_lines.append(FormatConverter._inline_md_to_html(stripped))

        if in_list:
            html_lines.append("</ul>")
        if in_paragraph:
            html_lines.append("</p>")

        return "\n".join(html_lines)

    @staticmethod
    def _inline_md_to_html(text: str) -> str:
        """Convert inline Markdown formatting to HTML."""
        import re

        # Inline code
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

        # Bold
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__([^_]+)__', r'<strong>\1</strong>', text)

        # Italic
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        text = re.sub(r'_([^_]+)_', r'<em>\1</em>', text)

        # Strikethrough
        text = re.sub(r'~~([^~]+)~~', r'<del>\1</del>', text)

        # Links
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

        return text

    @staticmethod
    def html_to_plain(text: str) -> str:
        """Strip HTML tags to produce plain text."""
        import re
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</h[1-6]>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&#39;', "'", text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def markdown_to_plain(text: str) -> str:
        """Convert Markdown to plain text."""
        import re
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        text = re.sub(r'~~([^~]+)~~', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)
        text = re.sub(r'^```[\s\S]*?```', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[-*]\s+', '- ', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def convert(text: str, source_format: str, target_format: str) -> str:
        """Convert text between any two formats."""
        if source_format == target_format:
            return text
        if source_format == MessageFormat.MARKDOWN:
            if target_format == MessageFormat.HTML:
                return FormatConverter.markdown_to_html(text)
            return FormatConverter.markdown_to_plain(text)
        if source_format == MessageFormat.HTML:
            if target_format == MessageFormat.PLAIN_TEXT:
                return FormatConverter.html_to_plain(text)
            return FormatConverter.html_to_plain(text)
        return text


# ──────────────────────────────────────────────────────────────────────────────
# Platform Format Preferences
# ──────────────────────────────────────────────────────────────────────────────

# Preferred message format per platform
PLATFORM_FORMAT_PREFERENCES: Dict[str, str] = {
    "telegram": MessageFormat.HTML,
    "discord": MessageFormat.MARKDOWN,
    "slack": MessageFormat.MARKDOWN,
    "whatsapp": MessageFormat.PLAIN_TEXT,
    "signal": MessageFormat.PLAIN_TEXT,
    "matrix": MessageFormat.HTML,
    "email": MessageFormat.HTML,
    "sms": MessageFormat.PLAIN_TEXT,
    "webhook": MessageFormat.PLAIN_TEXT,
    "api": MessageFormat.MARKDOWN,
    "dingtalk": MessageFormat.MARKDOWN,
    "feishu": MessageFormat.PLAIN_TEXT,
    "wecom": MessageFormat.PLAIN_TEXT,
    "mattermost": MessageFormat.MARKDOWN,
}

# Message length limits per platform
PLATFORM_MESSAGE_LIMITS: Dict[str, int] = {
    "telegram": 4096,
    "discord": 2000,
    "slack": 40000,
    "whatsapp": 65536,
    "signal": 65536,
    "matrix": 65536,
    "email": 1000000,
    "sms": 1600,
    "webhook": 65536,
    "api": 65536,
    "dingtalk": 20000,
    "feishu": 30000,
    "wecom": 2048,
    "mattermost": 16384,
}


# ──────────────────────────────────────────────────────────────────────────────
# Delivery Router
# ──────────────────────────────────────────────────────────────────────────────

class DeliveryRouter:
    """
    Routes messages to appropriate platform adapters.

    Features:
    - Per-platform message formatting and length truncation
    - Media attachment handling
    - Delivery retry with exponential backoff
    - Fallback to alternative platforms on failure
    - Delivery confirmation tracking
    """

    def __init__(
        self,
        config: GatewayConfig,
        session_store: Optional[SessionStore] = None,
        adapters: Optional[Dict[str, Any]] = None,
    ):
        self._config = config
        self._session_store = session_store
        self._adapters = adapters or {}

        # Delivery tracking
        self._pending: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, List[DeliveryResult]] = {}

        # Format converter
        self._converter = FormatConverter()

    def register_adapter(self, name: str, adapter: Any) -> None:
        """Register a platform adapter."""
        self._adapters[name] = adapter

    def unregister_adapter(self, name: str) -> None:
        """Unregister a platform adapter."""
        self._adapters.pop(name, None)

    # ── Message Delivery ──────────────────────────────────────────────────

    async def deliver(
        self,
        user_id: str,
        platform: str,
        chat_id: str,
        text: str,
        attachments: Optional[List[str]] = None,
        format: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        fallback_platforms: Optional[List[str]] = None,
    ) -> DeliveryResult:
        """
        Deliver a message to a specific platform and chat.

        Parameters
        ----------
        user_id:
            Target user identifier.
        platform:
            Target platform name.
        chat_id:
            Target chat identifier.
        text:
            Message text content.
        attachments:
            Optional list of file paths to attach.
        format:
            Message format override. Defaults to platform preference.
        reply_to:
            Optional message ID to reply to.
        metadata:
            Optional message metadata.
        fallback_platforms:
            List of fallback platform names if primary fails.

        Returns
        -------
        DeliveryResult
            The result of the delivery attempt.
        """
        # Format message for target platform
        target_format = format or PLATFORM_FORMAT_PREFERENCES.get(
            platform, MessageFormat.PLAIN_TEXT
        )
        source_format = self._config.message_format_default
        formatted_text = self._converter.convert(text, source_format, target_format)

        # Truncate to platform limit
        limit = PLATFORM_MESSAGE_LIMITS.get(platform, 4096)
        formatted_text = self._truncate_message(formatted_text, limit)

        # Get adapter
        adapter = self._adapters.get(platform)
        if adapter is None:
            result = DeliveryResult(
                success=False, platform=platform, chat_id=chat_id,
                error=f"No adapter registered for platform '{platform}'",
            )
            logger.error("No adapter for platform '%s': %s", platform, result.error)
            return result

        # Attempt delivery with retries
        result = await self._deliver_with_retry(
            adapter=adapter,
            platform=platform,
            chat_id=chat_id,
            text=formatted_text,
            attachments=attachments,
            reply_to=reply_to,
            metadata=metadata,
        )

        # If failed and fallback platforms are provided, try them
        if not result.success and fallback_platforms:
            for fb_platform in fallback_platforms:
                fb_adapter = self._adapters.get(fb_platform)
                if fb_adapter is None:
                    continue

                fb_format = PLATFORM_FORMAT_PREFERENCES.get(
                    fb_platform, MessageFormat.PLAIN_TEXT
                )
                fb_text = self._converter.convert(text, source_format, fb_format)
                fb_limit = PLATFORM_MESSAGE_LIMITS.get(fb_platform, 4096)
                fb_text = self._truncate_message(fb_text, fb_limit)

                fb_result = await self._deliver_with_retry(
                    adapter=fb_adapter,
                    platform=fb_platform,
                    chat_id=chat_id,
                    text=fb_text,
                    attachments=attachments,
                    metadata=metadata,
                )

                if fb_result.success:
                    return fb_result

        return result

    async def deliver_to_all_linked(
        self,
        user_id: str,
        source_platform: str,
        text: str,
        attachments: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DeliveryResult]:
        """
        Deliver a message to all platforms linked to a user.

        Useful for cross-platform notification broadcasting.
        """
        if self._session_store is None:
            return []

        # Find all linked platforms for this user
        linked: List[Tuple[str, str]] = []
        for cache_key, ctx in self._session_store._cache.items():
            if ctx.user_id == user_id:
                for lp in ctx.linked_platforms:
                    lp_platform = lp["platform"]
                    lp_chat_id = lp.get("chat_id", ctx.chat_id)
                    if lp_platform != source_platform:
                        linked.append((lp_platform, lp_chat_id))

        # Also check the source platform session
        source_ctx = await self._session_store.get(user_id, source_platform)
        if source_ctx:
            for lp in source_ctx.linked_platforms:
                lp_platform = lp["platform"]
                lp_chat_id = lp.get("chat_id", source_ctx.chat_id)
                if lp_platform != source_platform:
                    linked.append((lp_platform, lp_chat_id))

        # Deduplicate
        seen = set()
        unique_linked = []
        for platform, chat_id in linked:
            key = f"{platform}:{chat_id}"
            if key not in seen:
                seen.add(key)
                unique_linked.append((platform, chat_id))

        # Deliver to each
        results = []
        for platform, chat_id in unique_linked:
            result = await self.deliver(
                user_id=user_id,
                platform=platform,
                chat_id=chat_id,
                text=text,
                attachments=attachments,
                metadata=metadata,
            )
            results.append(result)

        return results

    async def deliver_agent_response(
        self,
        user_id: str,
        platform: str,
        chat_id: str,
        response_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeliveryResult:
        """
        Deliver an agent response to a user.

        Convenience method that handles formatting and truncation
        specifically for agent output.
        """
        return await self.deliver(
            user_id=user_id,
            platform=platform,
            chat_id=chat_id,
            text=response_text,
            metadata=metadata or {"source": "agent"},
        )

    # ── Retry Logic ───────────────────────────────────────────────────────

    async def _deliver_with_retry(
        self,
        adapter: Any,
        platform: str,
        chat_id: str,
        text: str,
        attachments: Optional[List[str]] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeliveryResult:
        """
        Attempt message delivery with configurable retries.

        Uses exponential backoff between retries.
        """
        retry_count = self._config.delivery_retry_count
        retry_delay = self._config.delivery_retry_delay
        last_error: Optional[str] = None

        for attempt in range(retry_count + 1):
            try:
                # Send text
                kwargs = {}
                if reply_to:
                    kwargs["reply_to"] = reply_to
                if metadata:
                    kwargs["metadata"] = metadata

                message_id = await adapter.send_message(chat_id, text, **kwargs)

                # Send attachments if any
                if attachments:
                    for file_path in attachments:
                        try:
                            await adapter.send_file(chat_id, file_path, **kwargs)
                        except Exception as att_err:
                            logger.warning(
                                "Failed to send attachment %s to %s:%s: %s",
                                file_path, platform, chat_id, att_err,
                            )

                return DeliveryResult(
                    success=True,
                    platform=platform,
                    chat_id=chat_id,
                    message_id=message_id,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Delivery attempt %d/%d failed for %s:%s: %s",
                    attempt + 1, retry_count + 1, platform, chat_id, e,
                )

                if attempt < retry_count:
                    delay = retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        return DeliveryResult(
            success=False,
            platform=platform,
            chat_id=chat_id,
            error=last_error,
        )

    # ── Message Splitting ─────────────────────────────────────────────────

    def _truncate_message(self, text: str, limit: int) -> str:
        """
        Truncate a message to fit within a platform's character limit.

        If truncation is needed, adds a continuation indicator.
        """
        if len(text) <= limit:
            return text

        # Try to truncate at a sentence or word boundary
        truncated = text[:limit - 50]
        # Find last sentence end
        for sep in [".\n", ". ", "\n", " "]:
            idx = truncated.rfind(sep)
            if idx > limit * 0.5:
                truncated = truncated[:idx + len(sep)]
                break

        return truncated.strip() + "\n\n[... message truncated]"

    async def split_and_deliver(
        self,
        user_id: str,
        platform: str,
        chat_id: str,
        text: str,
        attachments: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DeliveryResult]:
        """
        Split a long message into chunks and deliver each chunk separately.

        Useful for platforms with strict message length limits.
        """
        limit = PLATFORM_MESSAGE_LIMITS.get(platform, 4096)
        target_format = PLATFORM_FORMAT_PREFERENCES.get(
            platform, MessageFormat.PLAIN_TEXT
        )
        formatted = self._converter.convert(
            text, self._config.message_format_default, target_format,
        )

        if len(formatted) <= limit:
            result = await self.deliver(
                user_id=user_id, platform=platform, chat_id=chat_id,
                text=formatted, attachments=attachments, metadata=metadata,
            )
            return [result]

        # Split into chunks at paragraph boundaries
        chunks = self._split_message(formatted, limit)
        results = []

        for i, chunk in enumerate(chunks):
            prefix = f"[{i + 1}/{len(chunks)}]\n" if len(chunks) > 1 else ""
            result = await self.deliver(
                user_id=user_id, platform=platform, chat_id=chat_id,
                text=prefix + chunk,
                attachments=attachments if i == 0 else None,
                metadata=metadata,
            )
            results.append(result)

        return results

    def _split_message(self, text: str, limit: int) -> List[str]:
        """Split a message into chunks at paragraph boundaries."""
        chunks: List[str] = []
        current = ""
        max_chunk = limit - 20  # Leave room for chunk prefix

        paragraphs = text.split("\n\n")

        for para in paragraphs:
            if not para:
                continue

            if len(current) + len(para) + 2 <= max_chunk:
                current = current + "\n\n" + para if current else para
            else:
                if current:
                    chunks.append(current)
                # Handle individual paragraph that exceeds limit
                if len(para) > max_chunk:
                    sentences = para.replace(". ", ".\n").split("\n")
                    sub_current = ""
                    for sentence in sentences:
                        if len(sub_current) + len(sentence) + 1 <= max_chunk:
                            sub_current = sub_current + " " + sentence if sub_current else sentence
                        else:
                            if sub_current:
                                chunks.append(sub_current)
                            sub_current = sentence
                    if sub_current:
                        current = sub_current
                    else:
                        current = ""
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks if chunks else [text]

    # ── Status ────────────────────────────────────────────────────────────

    def get_pending_count(self) -> int:
        """Return the number of pending (in-flight) deliveries."""
        return sum(1 for t in self._pending.values() if not t.done())

    def get_registered_adapters(self) -> List[str]:
        """Return list of registered platform adapter names."""
        return list(self._adapters.keys())
