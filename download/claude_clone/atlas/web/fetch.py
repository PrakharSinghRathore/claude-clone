"""
atlas.web.fetch - Robust web page fetching with content extraction.

Implements a comprehensive web fetching system with support for
HTML parsing, JSON handling, content extraction, metadata retrieval,
robots.txt respect, rate limiting, and proxy support.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import html
import io
import json
import logging
import mimetypes
import re
import ssl
import time
import urllib.parse
import urllib.request
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)


@dataclass
class ContentMetadata:
    """Metadata extracted from a web page.

    Attributes:
        title: The page title.
        description: The meta description.
        og_title: Open Graph title.
        og_description: Open Graph description.
        og_image: Open Graph image URL.
        og_type: Open Graph type.
        og_url: Open Graph canonical URL.
        canonical_url: Canonical URL from link tag.
        author: Page author.
        keywords: Page keywords.
        publish_date: Publication date.
        modified_date: Last modification date.
        language: Page language.
        site_name: Site name from og:site_name.
        favicon: Favicon URL.
        twitter_card: Twitter card type.
        twitter_title: Twitter card title.
        twitter_description: Twitter card description.
        twitter_image: Twitter card image.
        extra: Additional metadata.
    """
    title: str = ""
    description: str = ""
    og_title: str = ""
    og_description: str = ""
    og_image: str = ""
    og_type: str = ""
    og_url: str = ""
    canonical_url: str = ""
    author: str = ""
    keywords: str = ""
    publish_date: Optional[str] = None
    modified_date: Optional[str] = None
    language: str = ""
    site_name: str = ""
    favicon: str = ""
    twitter_card: str = ""
    twitter_title: str = ""
    twitter_description: str = ""
    twitter_image: str = ""
    extra: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None and v != ""
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContentMetadata:
        """Create from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class FetchResult:
    """Result of a web fetch operation.

    Attributes:
        url: The final URL after any redirects.
        original_url: The originally requested URL.
        status_code: HTTP status code.
        content_type: Content-Type header value.
        content: Raw response content as bytes.
        text: Decoded text content (if applicable).
        encoding: Detected or declared character encoding.
        headers: Response headers as dictionary.
        elapsed_ms: Request duration in milliseconds.
        success: Whether the fetch was successful.
        error: Error message if the fetch failed.
        metadata: Extracted page metadata.
    """
    url: str = ""
    original_url: str = ""
    status_code: int = 0
    content_type: str = ""
    content: bytes = b""
    text: str = ""
    encoding: str = "utf-8"
    headers: Dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    success: bool = False
    error: Optional[str] = None
    metadata: Optional[ContentMetadata] = None

    @property
    def is_html(self) -> bool:
        """Check if the content is HTML."""
        return "text/html" in self.content_type.lower()

    @property
    def is_json(self) -> bool:
        """Check if the content is JSON."""
        return "application/json" in self.content_type.lower()

    @property
    def is_text(self) -> bool:
        """Check if the content is text-based."""
        ct = self.content_type.lower()
        return ct.startswith("text/") or ct in (
            "application/json",
            "application/xml",
            "application/javascript",
            "application/xhtml+xml",
        )

    @property
    def content_length(self) -> int:
        """Get content length in bytes."""
        return len(self.content)

    @property
    def text_length(self) -> int:
        """Get text content length in characters."""
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes raw content)."""
        return {
            "url": self.url,
            "original_url": self.original_url,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "encoding": self.encoding,
            "text_length": self.text_length,
            "content_length": self.content_length,
            "headers": self.headers,
            "elapsed_ms": self.elapsed_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }


class RobotsChecker:
    """robots.txt compliance checker with caching.

    Implements RFC-compliant robots.txt parsing and checking.
    Results are cached per domain with a configurable TTL.
    """

    def __init__(self, cache_ttl: float = 3600.0, user_agent: str = "*") -> None:
        """Initialize the robots checker.

        Args:
            cache_ttl: Cache TTL in seconds (default: 1 hour).
            user_agent: User-agent string for robots.txt checks.
        """
        self._cache: OrderedDict[str, Tuple[float, RobotFileParser]] = OrderedDict()
        self._cache_ttl = cache_ttl
        self._user_agent = user_agent
        self._max_cache_size = 1000

    def is_allowed(self, url: str, user_agent: Optional[str] = None) -> bool:
        """Check if a URL is allowed by robots.txt.

        Args:
            url: The URL to check.
            user_agent: Optional user-agent override.

        Returns:
            True if the URL is allowed, False if disallowed.
        """
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return True

            domain = parsed.netloc
            robots_url = f"{parsed.scheme}://{domain}/robots.txt"

            rp = self._get_or_fetch_robots(robots_url)
            if rp is None:
                return True  # If we can't fetch robots.txt, allow by default

            agent = user_agent or self._user_agent
            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"

            return rp.can_fetch(agent, path)

        except Exception as e:
            logger.debug("Robots.txt check failed for %s: %s", url, e)
            return True  # Fail open

    def _get_or_fetch_robots(
        self, robots_url: str
    ) -> Optional[RobotFileParser]:
        """Get cached robots.txt or fetch a new one.

        Args:
            robots_url: The robots.txt URL to fetch.

        Returns:
            A RobotFileParser, or None if unavailable.
        """
        now = time.monotonic()

        # Check cache
        if robots_url in self._cache:
            ts, rp = self._cache[robots_url]
            if now - ts < self._cache_ttl:
                # Move to end (most recently used)
                self._cache.move_to_end(robots_url)
                return rp
            del self._cache[robots_url]

        # Fetch robots.txt
        try:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            # Use a short timeout for robots.txt
            rp.read()
            self._cache[robots_url] = (now, rp)

            # Enforce max cache size
            while len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)

            return rp

        except Exception as e:
            logger.debug("Failed to fetch robots.txt from %s: %s", robots_url, e)
            # Cache the failure for a shorter time to avoid repeated attempts
            rp = RobotFileParser()
            self._cache[robots_url] = (now, rp)
            return None

    def clear_cache(self) -> None:
        """Clear the robots.txt cache."""
        self._cache.clear()


class RateLimiter:
    """Domain-based rate limiter for web fetching.

    Implements a sliding window rate limiter per domain
    to avoid overwhelming servers.
    """

    def __init__(
        self,
        max_requests_per_domain: int = 5,
        window_seconds: float = 10.0,
        max_requests_global: int = 30,
        global_window_seconds: float = 60.0,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            max_requests_per_domain: Max requests per domain per window.
            window_seconds: Domain rate limit window.
            max_requests_global: Max global requests per window.
            global_window_seconds: Global rate limit window.
        """
        self._max_per_domain = max_requests_per_domain
        self._domain_window = window_seconds
        self._max_global = max_requests_global
        self._global_window = global_window_seconds
        self._domain_timestamps: Dict[str, List[float]] = {}
        self._global_timestamps: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self, domain: str) -> None:
        """Wait until a request can be made to the given domain.

        Args:
            domain: The target domain.
        """
        async with self._lock:
            now = time.monotonic()

            # Clean up global timestamps
            self._global_timestamps = [
                ts for ts in self._global_timestamps
                if now - ts < self._global_window
            ]

            # Clean up domain timestamps
            if domain in self._domain_timestamps:
                self._domain_timestamps[domain] = [
                    ts for ts in self._domain_timestamps[domain]
                    if now - ts < self._domain_window
                ]

            # Check global rate limit
            if len(self._global_timestamps) >= self._max_global:
                sleep_time = (
                    self._global_timestamps[0]
                    + self._global_window
                    - now
                )
                if sleep_time > 0:
                    logger.debug(
                        "Global rate limit reached, waiting %.1fs", sleep_time
                    )
                    await asyncio.sleep(sleep_time)

            # Check domain rate limit
            domain_ts = self._domain_timestamps.get(domain, [])
            if len(domain_ts) >= self._max_per_domain:
                sleep_time = (
                    domain_ts[0] + self._domain_window - now
                )
                if sleep_time > 0:
                    logger.debug(
                        "Domain rate limit reached for %s, waiting %.1fs",
                        domain, sleep_time,
                    )
                    await asyncio.sleep(sleep_time)

            # Record this request
            self._global_timestamps.append(time.monotonic())
            self._domain_timestamps.setdefault(domain, []).append(
                time.monotonic()
            )


class WebFetcher:
    """Robust web page fetching with content extraction.

    Provides a comprehensive set of methods for fetching web pages,
    parsing HTML, extracting content and metadata, and handling
    various edge cases like redirects, encoding, and rate limiting.

    Example::

        fetcher = WebFetcher()

        # Simple fetch
        result = await fetcher.fetch("https://example.com")
        print(result.text)

        # Fetch with HTML parsing
        result = await fetcher.fetch_html("https://example.com")
        print(result.metadata.title)
        print(result.metadata.description)

        # Extract main content
        content = await fetcher.extract_content_from_url("https://example.com")
    """

    DEFAULT_USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
        "Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) "
        "Gecko/20100101 Firefox/121.0",
    ]

    # Common content encodings
    CONTENT_ENCODINGS = {"gzip", "deflate", "br"}

    def __init__(
        self,
        timeout: float = 30.0,
        max_content_size: int = 10 * 1024 * 1024,  # 10 MB
        max_redirects: int = 10,
        respect_robots_txt: bool = True,
        user_agents: Optional[List[str]] = None,
        default_headers: Optional[Dict[str, str]] = None,
        rate_limiter: Optional[RateLimiter] = None,
        proxy: Optional[Dict[str, str]] = None,
        ssl_verify: bool = True,
    ) -> None:
        """Initialize the WebFetcher.

        Args:
            timeout: Default request timeout in seconds.
            max_content_size: Maximum content size to download.
            max_redirects: Maximum number of redirects to follow.
            respect_robots_txt: Whether to check robots.txt.
            user_agents: Custom User-Agent strings for rotation.
            default_headers: Default HTTP headers.
            rate_limiter: Custom RateLimiter instance.
            proxy: Proxy configuration dict with 'http' and 'https' keys.
            ssl_verify: Whether to verify SSL certificates.
        """
        self.timeout = timeout
        self.max_content_size = max_content_size
        self.max_redirects = max_redirects
        self.respect_robots_txt = respect_robots_txt
        self.user_agents = user_agents or self.DEFAULT_USER_AGENTS
        self.default_headers = default_headers or {}
        self.rate_limiter = rate_limiter or RateLimiter()
        self.proxy = proxy
        self.ssl_verify = ssl_verify
        self._robots_checker = RobotsChecker()
        self._ua_index = 0
        self._content_cache: Dict[str, Tuple[float, FetchResult]] = {}
        self._cache_ttl = 300.0  # 5 minutes

    def _get_user_agent(self) -> str:
        """Get the next User-Agent string using round-robin."""
        ua = self.user_agents[self._ua_index % len(self.user_agents)]
        self._ua_index += 1
        return ua

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create an SSL context.

        Returns:
            An SSL context, or None if verification is disabled.
        """
        if not self.ssl_verify:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        return None  # Use default

    def _make_request(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Make an HTTP request.

        Args:
            url: The URL to request.
            method: HTTP method.
            headers: Additional headers.
            data: Request body.
            timeout: Timeout override.

        Returns:
            Tuple of (status_code, headers_dict, body_bytes).

        Raises:
            urllib.error.URLError: If the request fails.
        """
        req = urllib.request.Request(url, data=data, method=method)

        # Set headers
        req.add_header("User-Agent", self._get_user_agent())
        req.add_header("Accept", "*/*")
        req.add_header("Accept-Encoding", "gzip, deflate")
        req.add_header("Accept-Language", "en-US,en;q=0.9,*;q=0.8")
        req.add_header("Connection", "keep-alive")

        for key, value in self.default_headers.items():
            req.add_header(key, value)

        if headers:
            for key, value in headers.items():
                req.add_header(key, value)

        timeout_val = timeout or self.timeout
        ssl_ctx = self._create_ssl_context()

        # Handle proxy
        proxy_handler = None
        if self.proxy:
            proxies = {}
            if "http" in self.proxy:
                proxies["http"] = self.proxy["http"]
            if "https" in self.proxy:
                proxies["https"] = self.proxy["https"]
            if proxies:
                proxy_handler = urllib.request.ProxyHandler(proxies)

        opener = urllib.request.build_opener(proxy_handler) if proxy_handler else urllib.request.build_opener()

        with opener.open(req, timeout=timeout_val, context=ssl_ctx) as resp:
            status_code = resp.getcode() or 200
            resp_headers: Dict[str, str] = {}
            for key, value in resp.getheaders():
                resp_headers[key.lower()] = value

            # Read content with size limit
            content = b""
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                content += chunk
                if len(content) > self.max_content_size:
                    logger.warning(
                        "Content size exceeded limit (%d bytes) for %s",
                        self.max_content_size, url,
                    )
                    break

            # Decompress if needed
            content_encoding = resp_headers.get("content-encoding", "").lower()
            content = self._decompress(content, content_encoding)

            return status_code, resp_headers, content

    def _decompress(self, data: bytes, encoding: str) -> bytes:
        """Decompress response data based on content encoding.

        Args:
            data: The compressed data.
            encoding: The content encoding (gzip, deflate, etc.).

        Returns:
            Decompressed data.
        """
        try:
            if encoding == "gzip":
                return gzip.decompress(data)
            elif encoding == "deflate":
                return zlib.decompress(data)
            elif encoding == "br":
                # Brotli not in stdlib; return as-is
                return data
        except Exception as e:
            logger.debug("Decompression failed: %s", e)
        return data

    def _detect_encoding(
        self, content: bytes, headers: Dict[str, str]
    ) -> str:
        """Detect character encoding from content and headers.

        Args:
            content: The raw content bytes.
            headers: Response headers.

        Returns:
            Detected encoding string.
        """
        # Check Content-Type header
        ct = headers.get("content-type", "")
        charset_match = re.search(
            r"charset=([^\s;]+)", ct, re.IGNORECASE
        )
        if charset_match:
            return charset_match.group(1).strip().replace('"', "")

        # Check HTML meta tag
        if b"<meta" in content[:4096]:
            meta_match = re.search(
                rb'<meta[^>]+charset=["\']?([^"\'\s>]+)',
                content[:4096],
                re.IGNORECASE,
            )
            if meta_match:
                enc = meta_match.group(1).decode("ascii", errors="ignore")
                return enc.strip()

        # Check for BOM
        if content.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        elif content.startswith(b"\xff\xfe"):
            return "utf-16-le"
        elif content.startswith(b"\xfe\xff"):
            return "utf-16-be"

        # Default to UTF-8
        return "utf-8"

    def _decode_content(self, content: bytes, encoding: str) -> str:
        """Decode content bytes to string with fallback.

        Tries the specified encoding first, then falls back to
        common encodings if decoding fails.

        Args:
            content: The content bytes.
            encoding: The primary encoding to try.

        Returns:
            Decoded text string.
        """
        encodings_to_try = [encoding, "utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for enc in encodings_to_try:
            try:
                return content.decode(enc, errors="strict")
            except (UnicodeDecodeError, LookupError):
                continue

        # Last resort: decode with error replacement
        return content.decode("utf-8", errors="replace")

    async def fetch(
        self,
        url: str,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        method: str = "GET",
        data: Optional[bytes] = None,
        skip_robots: bool = False,
    ) -> FetchResult:
        """Fetch a URL and return the raw content.

        Args:
            url: The URL to fetch.
            timeout: Request timeout override.
            headers: Additional request headers.
            method: HTTP method.
            data: Request body.
            skip_robots: Skip robots.txt check.

        Returns:
            A FetchResult containing the response data.
        """
        start_time = time.monotonic()
        original_url = url
        result = FetchResult(
            original_url=original_url,
            url=url,
        )

        try:
            # Validate URL
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                result.error = f"Unsupported URL scheme: {parsed.scheme}"
                return result

            # Check robots.txt
            if self.respect_robots_txt and not skip_robots:
                if not self._robots_checker.is_allowed(url):
                    result.error = "Disallowed by robots.txt"
                    return result

            # Rate limiting
            domain = parsed.netloc
            await self.rate_limiter.acquire(domain)

            # Make request
            loop = asyncio.get_event_loop()
            status_code, resp_headers, content = await loop.run_in_executor(
                None,
                lambda: self._make_request(
                    url, method, headers, data, timeout
                ),
            )

            result.status_code = status_code
            result.headers = resp_headers
            result.content = content
            result.content_type = resp_headers.get("content-type", "")
            result.encoding = self._detect_encoding(content, resp_headers)

            if result.is_text:
                result.text = self._decode_content(content, result.encoding)

            # Check for redirect (some servers don't use proper redirects)
            if status_code in (301, 302, 303, 307, 308):
                location = resp_headers.get("location", "")
                if location:
                    result.url = urllib.parse.urljoin(url, location)

            result.success = 200 <= status_code < 400

            if not result.success:
                result.error = f"HTTP {status_code}"

        except urllib.error.HTTPError as e:
            result.status_code = e.code
            result.error = f"HTTP {e.code}: {e.reason}"
            result.headers = dict(e.headers) if e.headers else {}
            try:
                result.content = e.read()
                result.encoding = self._detect_encoding(
                    result.content, result.headers
                )
                if result.is_text:
                    result.text = self._decode_content(
                        result.content, result.encoding
                    )
            except Exception:
                pass

        except TimeoutError:
            result.error = f"Request timed out after {timeout or self.timeout}s"
        except urllib.error.URLError as e:
            result.error = f"URL error: {e.reason}"
        except Exception as e:
            result.error = f"Fetch error: {e}"

        result.elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.debug(
            "Fetch %s -> %d (%.0fms, %d bytes)",
            url, result.status_code, result.elapsed_ms, result.content_length,
        )

        return result

    async def fetch_html(
        self,
        url: str,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> FetchResult:
        """Fetch a URL and parse it as HTML.

        Automatically extracts metadata from the page.

        Args:
            url: The URL to fetch.
            timeout: Request timeout override.
            headers: Additional request headers.

        Returns:
            A FetchResult with text and metadata populated.
        """
        result = await self.fetch(url, timeout, headers)

        if result.success and result.is_html and result.text:
            result.metadata = self.extract_metadata(result.text, result.url)

        return result

    async def fetch_json(
        self,
        url: str,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[Any], FetchResult]:
        """Fetch a URL and parse it as JSON.

        Args:
            url: The URL to fetch.
            timeout: Request timeout override.
            headers: Additional request headers.

        Returns:
            A tuple of (parsed_json, FetchResult).
        """
        fetch_headers = headers or {}
        fetch_headers.setdefault("Accept", "application/json")

        result = await self.fetch(url, timeout, fetch_headers)

        json_data = None
        if result.success and result.text:
            try:
                json_data = json.loads(result.text)
            except json.JSONDecodeError as e:
                result.error = f"JSON parse error: {e}"
                result.success = False

        return json_data, result

    async def extract_content_from_url(
        self,
        url: str,
        timeout: Optional[float] = None,
    ) -> str:
        """Fetch a URL and extract its main content.

        Convenience method that combines fetch_html and extract_content.

        Args:
            url: The URL to fetch.
            timeout: Request timeout override.

        Returns:
            The extracted main content as text.
        """
        result = await self.fetch_html(url, timeout)
        if result.success and result.text:
            return self.extract_content(result.text)
        return ""

    # ── Content Extraction ────────────────────────────────────────────

    def extract_content(self, html_text: str) -> str:
        """Extract main content from HTML.

        Uses a heuristic-based approach to identify and extract the
        main content of a web page, removing navigation, sidebars,
        footers, ads, and other boilerplate content.

        Args:
            html_text: The HTML text to process.

        Returns:
            The extracted main content as clean text.
        """
        if not html_text:
            return ""

        # Step 1: Remove script and style blocks
        text = re.sub(
            r"<script[^>]*>.*?</script>", "", html_text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r"<noscript[^>]*>.*?</noscript>", "", text, flags=re.DOTALL | re.IGNORECASE
        )

        # Step 2: Remove comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

        # Step 3: Remove common boilerplate elements
        boilerplate_selectors = [
            r"<nav[^>]*>.*?</nav>",
            r"<header[^>]*>.*?</header>",
            r"<footer[^>]*>.*?</footer>",
            r"<aside[^>]*>.*?</aside>",
            r'<div[^>]*class="[^"]*(?:sidebar|ad|advertisement|banner|popup|modal|cookie|newsletter|social)[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*id="[^"]*(?:sidebar|ad|advertisement|banner|popup|modal|cookie|newsletter|social|nav|menu|header|footer)[^"]*"[^>]*>.*?</div>',
            r'<form[^>]*class="[^"]*search[^"]*"[^>]*>.*?</form>',
        ]
        for pattern in boilerplate_selectors:
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

        # Step 4: Try to find main content area
        main_content = self._find_main_content(text)

        # Step 5: Convert HTML to text
        text = self._html_to_text(main_content or text)

        # Step 6: Clean up whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = text.strip()

        return text

    def _find_main_content(self, html: str) -> Optional[str]:
        """Try to find the main content area of an HTML page.

        Looks for common content containers like <main>, <article>,
        and div elements with content-related classes.

        Args:
            html: The HTML text.

        Returns:
            The main content HTML, or None if not found.
        """
        # Priority order for content containers
        patterns = [
            (r"<main[^>]*>(.*?)</main>", 10),
            (r"<article[^>]*>(.*?)</article>", 9),
            (r'<div[^>]*class="[^"]*(?:content|article|post|entry|story|body|main|text)[^"]*"[^>]*>(.*?)</div>', 8),
            (r'<div[^>]*id="[^"]*(?:content|article|post|entry|story|body|main|text)[^"]*"[^>]*>(.*?)</div>', 7),
            (r'<div[^>]*role="main"[^>]*>(.*?)</div>', 8),
            (r'<div[^>]*class="[^"]*(?:markdown|prose)[^"]*"[^>]*>(.*?)</div>', 6),
        ]

        best_match: Optional[str] = None
        best_score = 0

        for pattern, score in patterns:
            match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1)
                text_length = len(re.sub(r"<[^>]+>", "", content).strip())
                adjusted_score = score * (1 + text_length / 1000)

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_match = content

        return best_match

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text.

        Handles common HTML elements and converts them to
        readable plain text with appropriate formatting.

        Args:
            html: The HTML text.

        Returns:
            Plain text representation.
        """
        text = html

        # Convert headings
        text = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n## \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h3[^>]*>(.*?)</h3>", r"\n### \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h4[^>]*>(.*?)</h4>", r"\n#### \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h5[^>]*>(.*?)</h5>", r"\n##### \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h6[^>]*>(.*?)</h6>", r"\n###### \1\n", text, flags=re.DOTALL | re.IGNORECASE)

        # Convert paragraphs
        text = re.sub(
            r"<p[^>]*>(.*?)</p>", r"\n\1\n", text, flags=re.DOTALL | re.IGNORECASE
        )

        # Convert line breaks
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<hr\s*/?>", "\n---\n", text, flags=re.IGNORECASE)

        # Convert list items
        text = re.sub(r"<li[^>]*>(.*?)</li>", r"  • \1\n", text, flags=re.DOTALL | re.IGNORECASE)

        # Convert links (keep text, add URL in brackets)
        text = re.sub(
            r'<a[^>]+href="([^"]*)"[^>]*>(.*?)</a>',
            r"\2 [\1]",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Convert bold and italic
        text = re.sub(r"<(?:strong|b)[^>]*>(.*?)</(?:strong|b)>", r"**\1**", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<(?:em|i)[^>]*>(.*?)</(?:em|i)>", r"*\1*", text, flags=re.DOTALL | re.IGNORECASE)

        # Convert code blocks
        text = re.sub(
            r"<pre[^>]*><code[^>]*>(.*?)</code></pre>",
            r"\n```\n\1\n```\n",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(
            r"<code[^>]*>(.*?)</code>",
            r"`\1`",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Convert blockquotes
        text = re.sub(
            r"<blockquote[^>]*>(.*?)</blockquote>",
            lambda m: "\n" + "\n".join(f"> {line}" for line in m.group(1).strip().split("\n")) + "\n",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Convert table cells
        text = re.sub(r"<t[hd][^>]*>(.*?)</t[hd]>", r" \1 |", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<tr[^>]*>", "\n|", text, flags=re.IGNORECASE)
        text = re.sub(r"</tr>", "", text, flags=re.IGNORECASE)

        # Remove all remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Decode HTML entities
        text = html.unescape(text)

        # Clean up excessive newlines
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        return text.strip()

    # ── Metadata Extraction ───────────────────────────────────────────

    def extract_metadata(self, html_text: str, url: str = "") -> ContentMetadata:
        """Extract metadata from an HTML page.

        Extracts title, description, Open Graph tags, Twitter cards,
        canonical URL, and other metadata from HTML.

        Args:
            html_text: The HTML text.
            url: The page URL (for resolving relative URLs).

        Returns:
            A ContentMetadata object with extracted metadata.
        """
        metadata = ContentMetadata()

        # Extract title
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>", html_text, re.DOTALL | re.IGNORECASE
        )
        if title_match:
            metadata.title = html.unescape(title_match.group(1).strip())

        # Extract meta tags
        meta_patterns = [
            (r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']*)["\']', "description"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+name=["\']description["\']', "description"),
            (r'<meta\s+name=["\']author["\']\s+content=["\']([^"\']*)["\']', "author"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+name=["\']author["\']', "author"),
            (r'<meta\s+name=["\']keywords["\']\s+content=["\']([^"\']*)["\']', "keywords"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+name=["\']keywords["\']', "keywords"),
            (r'<meta\s+name=["\']date["\']\s+content=["\']([^"\']*)["\']', "publish_date"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+name=["\']date["\']', "publish_date"),
            (r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']*)["\']', "og_title"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+property=["\']og:title["\']', "og_title"),
            (r'<meta\s+property=["\']og:description["\']\s+content=["\']([^"\']*)["\']', "og_description"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+property=["\']og:description["\']', "og_description"),
            (r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']*)["\']', "og_image"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+property=["\']og:image["\']', "og_image"),
            (r'<meta\s+property=["\']og:type["\']\s+content=["\']([^"\']*)["\']', "og_type"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+property=["\']og:type["\']', "og_type"),
            (r'<meta\s+property=["\']og:url["\']\s+content=["\']([^"\']*)["\']', "og_url"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+property=["\']og:url["\']', "og_url"),
            (r'<meta\s+property=["\']og:site_name["\']\s+content=["\']([^"\']*)["\']', "site_name"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+property=["\']og:site_name["\']', "site_name"),
            (r'<meta\s+name=["\']twitter:card["\']\s+content=["\']([^"\']*)["\']', "twitter_card"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+name=["\']twitter:card["\']', "twitter_card"),
            (r'<meta\s+name=["\']twitter:title["\']\s+content=["\']([^"\']*)["\']', "twitter_title"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+name=["\']twitter:title["\']', "twitter_title"),
            (r'<meta\s+name=["\']twitter:description["\']\s+content=["\']([^"\']*)["\']', "twitter_description"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+name=["\']twitter:description["\']', "twitter_description"),
            (r'<meta\s+name=["\']twitter:image["\']\s+content=["\']([^"\']*)["\']', "twitter_image"),
            (r'<meta\s+content=["\']([^"\']*)["\']\s+name=["\']twitter:image["\']', "twitter_image"),
        ]

        for pattern, attr_name in meta_patterns:
            match = re.search(pattern, html_text, re.IGNORECASE)
            if match:
                value = html.unescape(match.group(1).strip())
                setattr(metadata, attr_name, value)

        # Extract canonical URL
        canonical_match = re.search(
            r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']*)["\']',
            html_text, re.IGNORECASE,
        )
        if not canonical_match:
            canonical_match = re.search(
                r'<link[^>]+href=["\']([^"\']*)["\'][^>]+rel=["\']canonical["\']',
                html_text, re.IGNORECASE,
            )
        if canonical_match:
            metadata.canonical_url = canonical_match.group(1)

        # Extract favicon
        favicon_match = re.search(
            r'<link[^>]+rel=["\'](?:shortcut )?icon["\'][^>]+href=["\']([^"\']*)["\']',
            html_text, re.IGNORECASE,
        )
        if not favicon_match:
            favicon_match = re.search(
                r'<link[^>]+href=["\']([^"\']*)["\'][^>]+rel=["\'](?:shortcut )?icon["\']',
                html_text, re.IGNORECASE,
            )
        if favicon_match:
            favicon_url = favicon_match.group(1)
            if url:
                metadata.favicon = urllib.parse.urljoin(url, favicon_url)
            else:
                metadata.favicon = favicon_url

        # Extract language from html tag
        lang_match = re.search(
            r"<html[^>]+lang=[\"']([^\"']+)[\"']", html_text, re.IGNORECASE
        )
        if lang_match:
            metadata.language = lang_match.group(1)

        # Fallback: use og:title as title if no title found
        if not metadata.title and metadata.og_title:
            metadata.title = metadata.og_title
        if not metadata.description and metadata.og_description:
            metadata.description = metadata.og_description

        return metadata

    # ── Link Extraction ───────────────────────────────────────────────

    def extract_links(
        self,
        html_text: str,
        base_url: str = "",
        resolve_relative: bool = True,
        unique: bool = True,
    ) -> List[Dict[str, str]]:
        """Extract all links from HTML.

        Extracts href attributes from anchor tags, optionally
        resolving relative URLs and deduplicating.

        Args:
            html_text: The HTML text.
            base_url: Base URL for resolving relative URLs.
            resolve_relative: Whether to resolve relative URLs.
            unique: Whether to return only unique URLs.

        Returns:
            A list of dicts with 'url' and 'text' keys.
        """
        links: List[Dict[str, str]] = []
        seen: Set[str] = set()

        # Match anchor tags with href
        pattern = re.compile(
            r'<a[^>]+href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )

        for match in pattern.finditer(html_text):
            href = match.group(1).strip()
            text = self._html_to_text(match.group(2)).strip()

            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            # Resolve relative URLs
            if resolve_relative and base_url:
                href = urllib.parse.urljoin(base_url, href)

            if unique:
                if href in seen:
                    continue
                seen.add(href)

            links.append({"url": href, "text": text})

        return links

    # ── URL Reachability ──────────────────────────────────────────────

    async def is_reachable(
        self,
        url: str,
        timeout: Optional[float] = None,
        method: str = "HEAD",
    ) -> bool:
        """Check if a URL is reachable.

        Args:
            url: The URL to check.
            timeout: Timeout override.
            method: HTTP method to use (HEAD is faster).

        Returns:
            True if the URL is reachable.
        """
        try:
            result = await self.fetch(
                url, timeout=timeout or 10.0, method=method
            )
            return result.success
        except Exception:
            return False

    def detect_content_type(self, url: str) -> str:
        """Detect the likely content type of a URL.

        Uses the URL path extension to guess the content type.

        Args:
            url: The URL to check.

        Returns:
            A MIME type string.
        """
        parsed = urlparse(url)
        path = parsed.path.lower()

        # Check for common extensions
        extension_map = {
            ".html": "text/html",
            ".htm": "text/html",
            ".xhtml": "application/xhtml+xml",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".xml": "application/xml",
            ".txt": "text/plain",
            ".csv": "text/csv",
            ".md": "text/markdown",
            ".rss": "application/rss+xml",
            ".atom": "application/atom+xml",
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".webp": "image/webp",
            ".ico": "image/x-icon",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".zip": "application/zip",
            ".tar": "application/x-tar",
            ".gz": "application/gzip",
            ".rar": "application/vnd.rar",
            ".7z": "application/x-7z-compressed",
            ".py": "text/x-python",
            ".js": "application/javascript",
            ".ts": "application/typescript",
            ".java": "text/x-java-source",
            ".c": "text/x-c",
            ".cpp": "text/x-c++",
            ".h": "text/x-c",
            ".rb": "text/x-ruby",
            ".go": "text/x-go",
            ".rs": "text/x-rust",
            ".php": "text/x-php",
            ".sh": "text/x-shellscript",
        }

        for ext, mime_type in extension_map.items():
            if path.endswith(ext):
                return mime_type

        # Fall back to mimetypes module
        guessed_type, _ = mimetypes.guess_type(url)
        return guessed_type or "application/octet-stream"

    def clear_cache(self) -> None:
        """Clear the content cache."""
        self._content_cache.clear()
        self._robots_checker.clear_cache()
