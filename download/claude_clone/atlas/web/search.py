"""
atlas.web.search - Multi-provider web search engine.

Implements a robust web search system with support for multiple search
providers, automatic failover, rate limiting, result deduplication,
and ranking. Supports standard web search, news search, image search,
and contextual search.

Providers:
    - DuckDuckGo (default, no API key required)
    - Tavily (API key required, excellent for AI agents)
    - Exa (API key required, neural search)
    - Brave (API key required, privacy-focused)
    - SearXNG (self-hosted)
    - Google Builtin (uses urllib, no API key but limited)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SearchProvider(Enum):
    """Available search providers."""
    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"
    EXA = "exa"
    BRAVE = "brave"
    SEARXNG = "searxng"
    GOOGLE_BUILTIN = "google_builtin"


@dataclass
class SearchResult:
    """Represents a single search result.

    Attributes:
        url: The URL of the result.
        title: The title of the result.
        snippet: A brief description or snippet of the result.
        domain: The domain name extracted from the URL.
        rank: The rank position of this result (1-based).
        date: Optional publication/modification date string.
        favicon: Optional URL to the site's favicon.
        provider: Which provider returned this result.
        score: Relevance score (0.0 - 1.0), higher is more relevant.
        extra: Additional provider-specific metadata.
    """
    url: str
    title: str
    snippet: str
    domain: str = ""
    rank: int = 0
    date: Optional[str] = None
    favicon: Optional[str] = None
    provider: str = ""
    score: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Extract domain from URL if not provided."""
        if not self.domain and self.url:
            try:
                parsed = urlparse(self.url)
                self.domain = parsed.netloc or ""
            except Exception:
                self.domain = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "domain": self.domain,
            "rank": self.rank,
            "date": self.date,
            "favicon": self.favicon,
            "provider": self.provider,
            "score": self.score,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SearchResult:
        """Create a SearchResult from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def __hash__(self) -> int:
        """Hash based on URL for deduplication."""
        return hash(self.url.lower().rstrip("/"))

    def __eq__(self, other: object) -> bool:
        """Equality based on normalized URL."""
        if not isinstance(other, SearchResult):
            return NotImplemented
        return self.url.lower().rstrip("/") == other.url.lower().rstrip("/")


class SearchFilter:
    """Filters for search results.

    Attributes:
        date_from: Only include results after this date.
        date_to: Only include results before this date.
        domains: Only include results from these domains.
        exclude_domains: Exclude results from these domains.
        language: Language code (e.g. 'en', 'zh', 'de').
        region: Region code for localized results.
        safe_search: Whether to enable safe search.
        max_age_days: Maximum age of results in days.
    """

    def __init__(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        language: Optional[str] = None,
        region: Optional[str] = None,
        safe_search: bool = True,
        max_age_days: Optional[int] = None,
    ) -> None:
        self.date_from = date_from
        self.date_to = date_to
        self.domains = set(d.lower() for d in (domains or []))
        self.exclude_domains = set(d.lower() for d in (exclude_domains or []))
        self.language = language
        self.region = region
        self.safe_search = safe_search
        self.max_age_days = max_age_days

        if max_age_days and not date_from:
            self.date_from = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    def matches(self, result: SearchResult) -> bool:
        """Check if a search result matches all filters."""
        # Domain inclusion filter
        if self.domains:
            result_domain = result.domain.lower()
            if not any(
                result_domain == d or result_domain.endswith(f".{d}")
                for d in self.domains
            ):
                return False

        # Domain exclusion filter
        if self.exclude_domains:
            result_domain = result.domain.lower()
            if any(
                result_domain == d or result_domain.endswith(f".{d}")
                for d in self.exclude_domains
            ):
                return False

        # Date range filter
        if self.date_from and result.date:
            try:
                result_dt = self._parse_date(result.date)
                if result_dt and result_dt < self.date_from:
                    return False
            except (ValueError, TypeError):
                pass

        if self.date_to and result.date:
            try:
                result_dt = self._parse_date(result.date)
                if result_dt and result_dt > self.date_to:
                    return False
            except (ValueError, TypeError):
                pass

        return True

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse various date formats into datetime."""
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d %b %Y",
            "%d %B %Y",
            "%b %d, %Y",
            "%B %d, %Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
        ]
        date_str = date_str.strip()
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        # Try relative date parsing
        now = datetime.now(timezone.utc)
        day_match = re.match(r"(\d+)\s*(day|days|hour|hours|week|weeks|month|months)\s*ago", date_str, re.IGNORECASE)
        if day_match:
            value = int(day_match.group(1))
            unit = day_match.group(2).lower()
            if unit.startswith("day"):
                return now - timedelta(days=value)
            elif unit.startswith("hour"):
                return now - timedelta(hours=value)
            elif unit.startswith("week"):
                return now - timedelta(weeks=value)
            elif unit.startswith("month"):
                return now - timedelta(days=value * 30)
        return None


class RateLimiter:
    """Token bucket rate limiter for search providers.

    Attributes:
        max_requests: Maximum requests per window.
        window_seconds: Time window in seconds.
    """

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request can be made."""
        async with self._lock:
            now = time.monotonic()
            # Remove timestamps outside the window
            self._timestamps = [
                ts for ts in self._timestamps
                if now - ts < self.window_seconds
            ]
            if len(self._timestamps) >= self.max_requests:
                # Wait until the oldest timestamp expires
                sleep_time = self._timestamps[0] + self.window_seconds - now
                if sleep_time > 0:
                    logger.debug(
                        "Rate limit reached, waiting %.1fs", sleep_time
                    )
                    await asyncio.sleep(sleep_time)
                    # Clean up again after sleeping
                    self._timestamps = [
                        ts for ts in self._timestamps
                        if time.monotonic() - ts < self.window_seconds
                    ]
            self._timestamps.append(time.monotonic())

    def reset(self) -> None:
        """Reset the rate limiter."""
        self._timestamps.clear()


class ProviderConfig:
    """Configuration for a search provider.

    Attributes:
        provider: The provider type.
        api_key: Optional API key.
        base_url: Base URL for the API.
        enabled: Whether this provider is enabled.
        priority: Priority order (lower = preferred).
        rate_limit: Rate limit config (max_requests, window_seconds).
        timeout: Request timeout in seconds.
        extra: Provider-specific configuration.
    """

    def __init__(
        self,
        provider: SearchProvider,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enabled: bool = True,
        priority: int = 0,
        rate_limit: Tuple[int, float] = (10, 60.0),
        timeout: float = 30.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.enabled = enabled
        self.priority = priority
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.extra = extra or {}

    @classmethod
    def from_env(cls, provider: SearchProvider) -> ProviderConfig:
        """Create a ProviderConfig from environment variables."""
        import os

        env_map = {
            SearchProvider.TAVILY: ("TAVILY_API_KEY", "https://api.tavily.com"),
            SearchProvider.EXA: ("EXA_API_KEY", "https://api.exa.ai"),
            SearchProvider.BRAVE: ("BRAVE_API_KEY", "https://api.search.brave.com"),
            SearchProvider.SEARXNG: ("SEARXNG_URL", None),
        }

        env_key, default_url = env_map.get(provider, (None, None))

        api_key = None
        base_url = default_url

        if env_key:
            api_key = os.environ.get(env_key, "")
            # If the env var is a URL (like SearXNG), treat it as base_url
            if provider == SearchProvider.SEARXNG:
                base_url = api_key or "http://localhost:8888"
                api_key = None
            elif not api_key:
                logger.debug(
                    "No API key found for %s (env: %s)",
                    provider.value, env_key,
                )

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )


class WebSearchEngine:
    """Multi-provider web search engine with failover and ranking.

    This class provides a unified interface for searching the web using
    multiple providers. It implements automatic failover, rate limiting,
    result deduplication, and smart ranking.

    Example::

        engine = WebSearchEngine()
        engine.configure_provider(SearchProvider.DUCKDUCKGO)
        results = await engine.search("Python async programming")
        for result in results:
            print(f"{result.title}: {result.url}")
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
    ]

    def __init__(
        self,
        default_provider: SearchProvider = SearchProvider.DUCKDUCKGO,
        default_num_results: int = 10,
        timeout: float = 30.0,
        user_agents: Optional[List[str]] = None,
    ) -> None:
        """Initialize the WebSearchEngine.

        Args:
            default_provider: The default search provider to use.
            default_num_results: Default number of results to return.
            timeout: Default request timeout in seconds.
            user_agents: List of User-Agent strings to rotate through.
        """
        self.default_provider = default_provider
        self.default_num_results = default_num_results
        self.timeout = timeout
        self.user_agents = user_agents or self.DEFAULT_USER_AGENTS
        self._provider_configs: Dict[SearchProvider, ProviderConfig] = {}
        self._rate_limiters: Dict[SearchProvider, RateLimiter] = {}
        self._ua_index = 0
        self._search_cache: Dict[str, Tuple[float, List[SearchResult]]] = {}
        self._cache_ttl = 300.0  # 5 minutes
        self._provider_failures: Dict[SearchProvider, int] = {}
        self._max_failures_before_skip = 3

    def configure_provider(
        self,
        provider: SearchProvider,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        priority: int = 0,
        rate_limit: Optional[Tuple[int, float]] = None,
        timeout: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Configure a search provider.

        Args:
            provider: The provider to configure.
            api_key: API key for the provider (if required).
            base_url: Custom base URL for the provider.
            priority: Priority order (lower = preferred).
            rate_limit: (max_requests, window_seconds) tuple.
            timeout: Request timeout override.
            **extra: Provider-specific configuration.
        """
        config = ProviderConfig(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            priority=priority,
            rate_limit=rate_limit or (10, 60.0),
            timeout=timeout or self.timeout,
            extra=extra,
        )
        self._provider_configs[provider] = config
        self._rate_limiters[provider] = RateLimiter(
            max_requests=config.rate_limit[0],
            window_seconds=config.rate_limit[1],
        )
        logger.info(
            "Configured search provider: %s (priority=%d)",
            provider.value, priority,
        )

    def configure_from_env(self) -> None:
        """Auto-configure all providers from environment variables."""
        for provider in SearchProvider:
            config = ProviderConfig.from_env(provider)
            if provider == SearchProvider.DUCKDUCKGO or provider == SearchProvider.GOOGLE_BUILTIN:
                config.enabled = True
                config.priority = 0 if provider == SearchProvider.DUCKDUCKGO else 100
            elif config.api_key or (config.base_url and provider == SearchProvider.SEARXNG):
                config.enabled = True
            else:
                config.enabled = False
                continue
            self._provider_configs[provider] = config
            self._rate_limiters[provider] = RateLimiter(
                max_requests=config.rate_limit[0],
                window_seconds=config.rate_limit[1],
            )
        logger.info(
            "Auto-configured %d providers from environment",
            len(self._provider_configs),
        )

    def set_cache_ttl(self, ttl: float) -> None:
        """Set the cache time-to-live.

        Args:
            ttl: Time-to-live in seconds.
        """
        self._cache_ttl = ttl

    def clear_cache(self) -> None:
        """Clear the search result cache."""
        self._search_cache.clear()

    def _get_user_agent(self) -> str:
        """Get the next User-Agent string using round-robin."""
        ua = self.user_agents[self._ua_index % len(self.user_agents)]
        self._ua_index += 1
        return ua

    def _cache_key(self, query: str, provider: str, num: int) -> str:
        """Generate a cache key for a search query."""
        raw = f"{query}|{provider}|{num}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cache(self, key: str) -> Optional[List[SearchResult]]:
        """Get cached results if still valid."""
        if key in self._search_cache:
            ts, results = self._search_cache[key]
            if time.monotonic() - ts < self._cache_ttl:
                return results
            del self._search_cache[key]
        return None

    def _set_cache(self, key: str, results: List[SearchResult]) -> None:
        """Store results in cache."""
        self._search_cache[key] = (time.monotonic(), results)
        # Prune old entries if cache is too large
        if len(self._search_cache) > 500:
            now = time.monotonic()
            expired = [
                k for k, (ts, _) in self._search_cache.items()
                if now - ts >= self._cache_ttl
            ]
            for k in expired:
                del self._search_cache[k]

    def _deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL."""
        seen: Set[int] = set()
        unique = []
        for result in results:
            url_hash = hash(result.url.lower().rstrip("/"))
            if url_hash not in seen:
                seen.add(url_hash)
                unique.append(result)
        return unique

    def _rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank and sort results by relevance.

        Uses a combination of original rank, snippet quality,
        domain authority heuristics, and provider preference.
        """
        def score(result: SearchResult) -> float:
            """Calculate a composite relevance score."""
            s = 0.0

            # Original rank score (lower rank = higher score)
            if result.rank > 0:
                s += (1.0 / result.rank) * 10.0

            # Provider score
            provider_weights = {
                "duckduckgo": 1.0,
                "tavily": 1.2,
                "exa": 1.1,
                "brave": 1.0,
                "searxng": 0.9,
                "google_builtin": 0.8,
            }
            s *= provider_weights.get(result.provider, 1.0)

            # Snippet quality (longer, more descriptive snippets are better)
            snippet_len = len(result.snippet.strip())
            if snippet_len > 50:
                s += 0.5
            if snippet_len > 100:
                s += 0.5

            # Title quality
            title_len = len(result.title.strip())
            if 10 < title_len < 100:
                s += 0.3

            # HTTPS boost
            if result.url.startswith("https://"):
                s += 0.2

            # Known authoritative domains
            authoritative_domains = {
                "wikipedia.org", "github.com", "stackoverflow.com",
                "docs.python.org", "developer.mozilla.org", "arxiv.org",
                "medium.com", "dev.to", "hackernews.com",
            }
            for auth_domain in authoritative_domains:
                if auth_domain in result.domain.lower():
                    s += 1.0
                    break

            # Date freshness bonus (if date is available)
            if result.date:
                try:
                    dt = SearchFilter._parse_date(result.date)
                    if dt:
                        age_days = (datetime.now(timezone.utc) - dt).days
                        if age_days < 7:
                            s += 2.0
                        elif age_days < 30:
                            s += 1.0
                        elif age_days < 90:
                            s += 0.5
                except (ValueError, TypeError):
                    pass

            result.score = s
            return s

        results.sort(key=score, reverse=True)

        # Re-assign ranks after sorting
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def _get_sorted_providers(
        self,
        preferred: Optional[SearchProvider] = None,
    ) -> List[ProviderConfig]:
        """Get providers sorted by priority, excluding failed ones."""
        providers = []
        for provider, config in self._provider_configs.items():
            if not config.enabled:
                continue
            if self._provider_failures.get(provider, 0) >= self._max_failures_before_skip:
                logger.debug(
                    "Skipping provider %s due to too many failures",
                    provider.value,
                )
                continue
            providers.append(config)

        providers.sort(key=lambda p: p.priority)

        # Move preferred provider to front
        if preferred:
            for i, config in enumerate(providers):
                if config.provider == preferred:
                    providers.insert(0, providers.pop(i))
                    break

        return providers

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Make an HTTP GET request.

        Args:
            url: The URL to request.
            params: Query parameters.
            headers: Request headers.
            timeout: Timeout in seconds.

        Returns:
            Response body as bytes.

        Raises:
            urllib.error.URLError: If the request fails.
            TimeoutError: If the request times out.
        """
        if params:
            encoded_params = urllib.parse.urlencode(params)
            url = f"{url}?{encoded_params}"

        req = urllib.request.Request(url)
        req.add_header(
            "User-Agent",
            self._get_user_agent(),
        )
        req.add_header("Accept", "text/html,application/json,application/xml")

        if headers:
            for key, value in headers.items():
                req.add_header(key, value)

        timeout_val = timeout or self.timeout
        with urllib.request.urlopen(req, timeout=timeout_val) as resp:
            return resp.read()

    def _parse_json_response(self, data: bytes) -> Any:
        """Parse a JSON response body."""
        import json
        return json.loads(data.decode("utf-8"))

    # ── DuckDuckGo Implementation ─────────────────────────────────────

    async def _search_duckduckgo(
        self,
        query: str,
        num_results: int,
        filters: Optional[SearchFilter],
        search_type: str = "general",
    ) -> List[SearchResult]:
        """Search using DuckDuckGo HTML lite version.

        Args:
            query: The search query.
            num_results: Number of results to return.
            filters: Optional search filters.
            search_type: 'general', 'news', or 'images'.

        Returns:
            List of search results.
        """
        results: List[SearchResult] = []

        # DuckDuckGo lite endpoint
        base_url = "https://lite.duckduckgo.com/lite/"
        params: Dict[str, str] = {"q": query, "kl": "wt-wt"}

        if filters and filters.language:
            params["kl"] = f"{filters.language}-{filters.region or 'wt'}"

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, lambda: self._make_request(base_url, params)
            )
            html = data.decode("utf-8", errors="replace")

            # Parse DuckDuckGo lite HTML results
            # Results are in <tr> elements with specific class patterns
            results = self._parse_ddg_html(html, num_results)

            # Fallback: try the HTML version
            if not results:
                base_url = "https://html.duckduckgo.com/html/"
                data = await loop.run_in_executor(
                    None, lambda: self._make_request(base_url, params)
                )
                html = data.decode("utf-8", errors="replace")
                results = self._parse_ddg_html(html, num_results)

        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)

        return results[:num_results]

    def _parse_ddg_html(self, html: str, num_results: int) -> List[SearchResult]:
        """Parse DuckDuckGo HTML search results.

        Args:
            html: The HTML response from DuckDuckGo.
            num_results: Maximum number of results to parse.

        Returns:
            List of parsed SearchResult objects.
        """
        results: List[SearchResult] = []

        # Pattern for DuckDuckGo lite results
        # Links are in <a> tags within result rows
        link_pattern = re.compile(
            r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>',
            re.IGNORECASE,
        )
        snippet_pattern = re.compile(
            r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
            re.IGNORECASE | re.DOTALL,
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        # Also try alternative parsing for DuckDuckGo HTML version
        if not links:
            link_pattern2 = re.compile(
                r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                re.IGNORECASE | re.DOTALL,
            )
            snippet_pattern2 = re.compile(
                r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                re.IGNORECASE | re.DOTALL,
            )
            links = link_pattern2.findall(html)
            links = [(url, title.strip()) for url, title in links]
            raw_snippets = snippet_pattern2.findall(html)
            snippets = [s.strip() for s in raw_snippets]

        for i, (url, title) in enumerate(links):
            if i >= num_results:
                break

            # Clean HTML entities from title
            title = self._strip_html(title)
            url = self._decode_redirect_url(url)

            if not url or not title:
                continue

            snippet = ""
            if i < len(snippets):
                snippet = self._strip_html(snippets[i])

            results.append(SearchResult(
                url=url,
                title=title,
                snippet=snippet or "No description available.",
                provider="duckduckgo",
                rank=i + 1,
            ))

        return results

    @staticmethod
    def _decode_redirect_url(url: str) -> str:
        """Decode DuckDuckGo redirect URLs.

        DuckDuckGo wraps external URLs in redirect URLs like:
        https://duckduckgo.com/l/?uddg=<encoded_url>
        """
        if "uddg=" in url:
            try:
                parsed = urlparse(url)
                params = urllib.parse.parse_qs(parsed.query)
                if "uddg" in params:
                    return urllib.parse.unquote(params["uddg"][0])
            except Exception:
                pass
        if "//duckduckgo.com" in url and "/l/" in url:
            return ""
        return url

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags and decode entities."""
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", "", text)
        # Decode common HTML entities
        clean = clean.replace("&amp;", "&")
        clean = clean.replace("&lt;", "<")
        clean = clean.replace("&gt;", ">")
        clean = clean.replace("&quot;", '"')
        clean = clean.replace("&#39;", "'")
        clean = clean.replace("&nbsp;", " ")
        # Normalize whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    # ── Google Builtin Implementation ─────────────────────────────────

    async def _search_google_builtin(
        self,
        query: str,
        num_results: int,
        filters: Optional[SearchFilter],
    ) -> List[SearchResult]:
        """Search using Google via urllib (limited results).

        Args:
            query: The search query.
            num_results: Number of results to return.
            filters: Optional search filters.

        Returns:
            List of search results.
        """
        results: List[SearchResult] = []

        params: Dict[str, str] = {
            "q": query,
            "num": str(min(num_results, 10)),
            "gl": "us",
            "hl": "en",
        }

        if filters:
            if filters.language:
                params["hl"] = filters.language
            if filters.safe_search:
                params["safe"] = "active"

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: self._make_request(
                    "https://www.google.com/search", params
                ),
            )
            html = data.decode("utf-8", errors="replace")
            results = self._parse_google_html(html, num_results)

        except Exception as e:
            logger.warning("Google builtin search failed: %s", e)

        return results

    def _parse_google_html(
        self, html: str, num_results: int
    ) -> List[SearchResult]:
        """Parse Google HTML search results.

        Args:
            html: The HTML response from Google.
            num_results: Maximum number of results to parse.

        Returns:
            List of parsed SearchResult objects.
        """
        results: List[SearchResult] = []

        # Google result pattern
        result_blocks = re.findall(
            r'<div[^>]*class="[^"]*g[^"]*"[^>]*>.*?</div>',
            html,
            re.DOTALL,
        )

        for i, block in enumerate(result_blocks):
            if i >= num_results:
                break

            # Extract URL
            url_match = re.search(r'<a[^>]+href="(https?://[^"]+)"', block)
            if not url_match:
                continue
            url = url_match.group(1)

            # Skip Google internal URLs
            if url.startswith("https://www.google.com/"):
                continue

            # Extract title
            title_match = re.search(
                r'<h3[^>]*>(.*?)</h3>', block, re.DOTALL
            )
            if not title_match:
                continue
            title = self._strip_html(title_match.group(1))

            # Extract snippet
            snippet_match = re.search(
                r'<span[^>]*class="[^"]*aCOpRe[^"]*"[^>]*>(.*?)</span>',
                block,
                re.DOTALL,
            )
            if not snippet_match:
                snippet_match = re.search(
                    r'<div[^>]*class="[^"]*VwiC3b[^"]*"[^>]*>(.*?)</div>',
                    block,
                    re.DOTALL,
                )
            snippet = ""
            if snippet_match:
                snippet = self._strip_html(snippet_match.group(1))

            results.append(SearchResult(
                url=url,
                title=title,
                snippet=snippet or "No description available.",
                provider="google_builtin",
                rank=i + 1,
            ))

        return results

    # ── Tavily Implementation ─────────────────────────────────────────

    async def _search_tavily(
        self,
        query: str,
        num_results: int,
        filters: Optional[SearchFilter],
        search_type: str = "general",
    ) -> List[SearchResult]:
        """Search using the Tavily API.

        Args:
            query: The search query.
            num_results: Number of results to return.
            filters: Optional search filters.
            search_type: 'general' or 'news'.

        Returns:
            List of search results.
        """
        config = self._provider_configs.get(SearchProvider.TAVILY)
        if not config or not config.api_key:
            logger.debug("Tavily: No API key configured")
            return []

        results: List[SearchResult] = []

        payload: Dict[str, Any] = {
            "api_key": config.api_key,
            "query": query,
            "max_results": num_results,
            "include_answer": False,
            "include_raw_content": False,
        }

        if search_type == "news":
            payload["topic"] = "news"
        elif search_type == "images":
            payload["include_images"] = True
            payload["topic"] = "general"

        if filters:
            if filters.date_from:
                payload["search_depth"] = "advanced"
            if filters.domains:
                payload["include_domains"] = list(filters.domains)
            if filters.exclude_domains:
                payload["exclude_domains"] = list(filters.exclude_domains)

        try:
            loop = asyncio.get_event_loop()
            json_payload = self._json_dumps(payload)
            data = await loop.run_in_executor(
                None,
                lambda: self._make_post_request(
                    config.base_url + "/search" if config.base_url else "https://api.tavily.com/search",
                    json_payload,
                    {"Content-Type": "application/json"},
                    config.timeout,
                ),
            )
            response = self._parse_json_response(data)

            for i, item in enumerate(response.get("results", [])):
                results.append(SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    score=item.get("score", 1.0),
                    date=item.get("published_date"),
                    provider="tavily",
                    rank=i + 1,
                ))

            # Handle image results
            if search_type == "images":
                for i, img_url in enumerate(response.get("images", [])):
                    results.append(SearchResult(
                        url=img_url,
                        title=f"Image result {i + 1}",
                        snippet="",
                        provider="tavily",
                        rank=len(results) + 1,
                    ))

        except Exception as e:
            logger.warning("Tavily search failed: %s", e)

        return results[:num_results]

    def _make_post_request(
        self,
        url: str,
        data: bytes,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Make an HTTP POST request.

        Args:
            url: The URL to request.
            data: Request body as bytes.
            headers: Request headers.
            timeout: Timeout in seconds.

        Returns:
            Response body as bytes.
        """
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("User-Agent", self._get_user_agent())
        if headers:
            for key, value in headers.items():
                req.add_header(key, value)

        timeout_val = timeout or self.timeout
        with urllib.request.urlopen(req, timeout=timeout_val) as resp:
            return resp.read()

    @staticmethod
    def _json_dumps(obj: Any) -> bytes:
        """Serialize an object to JSON bytes."""
        import json
        return json.dumps(obj).encode("utf-8")

    # ── Brave Implementation ──────────────────────────────────────────

    async def _search_brave(
        self,
        query: str,
        num_results: int,
        filters: Optional[SearchFilter],
    ) -> List[SearchResult]:
        """Search using the Brave Search API.

        Args:
            query: The search query.
            num_results: Number of results to return.
            filters: Optional search filters.

        Returns:
            List of search results.
        """
        config = self._provider_configs.get(SearchProvider.BRAVE)
        if not config or not config.api_key:
            logger.debug("Brave: No API key configured")
            return []

        results: List[SearchResult] = []
        params: Dict[str, str] = {"q": query, "count": str(num_results)}

        if filters:
            if filters.language:
                params["search_lang"] = filters.language
            if filters.country:
                params["country"] = filters.country

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": config.api_key,
        }

        try:
            loop = asyncio.get_event_loop()
            base = config.base_url or "https://api.search.brave.com"
            data = await loop.run_in_executor(
                None,
                lambda: self._make_request(
                    f"{base}/res/v1/web/search", params, headers, config.timeout
                ),
            )
            response = self._parse_json_response(data)

            for i, item in enumerate(response.get("web", {}).get("results", [])):
                results.append(SearchResult(
                    url=item.get("url", ""),
                    title=self._strip_html(item.get("title", "")),
                    snippet=item.get("description", ""),
                    date=item.get("age"),
                    provider="brave",
                    rank=i + 1,
                    extra={
                        "language": item.get("language"),
                        "family_friendly": item.get("family_friendly"),
                    },
                ))

        except Exception as e:
            logger.warning("Brave search failed: %s", e)

        return results

    # ── Exa Implementation ────────────────────────────────────────────

    async def _search_exa(
        self,
        query: str,
        num_results: int,
        filters: Optional[SearchFilter],
    ) -> List[SearchResult]:
        """Search using the Exa (formerly Metaphor) neural search API.

        Args:
            query: The search query.
            num_results: Number of results to return.
            filters: Optional search filters.

        Returns:
            List of search results.
        """
        config = self._provider_configs.get(SearchProvider.EXA)
        if not config or not config.api_key:
            logger.debug("Exa: No API key configured")
            return []

        results: List[SearchResult] = []

        payload: Dict[str, Any] = {
            "query": query,
            "num_results": num_results,
            "use_autoprompt": True,
            "type": "auto",
        }

        if filters:
            if filters.date_from:
                payload.setdefault("start_published_date", filters.date_from.strftime("%Y-%m-%d"))
            if filters.domains:
                payload["domain_include"] = list(filters.domains)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": config.api_key,
        }

        try:
            loop = asyncio.get_event_loop()
            base = config.base_url or "https://api.exa.ai"
            json_payload = self._json_dumps(payload)
            data = await loop.run_in_executor(
                None,
                lambda: self._make_post_request(
                    f"{base}/search", json_payload, headers, config.timeout
                ),
            )
            response = self._parse_json_response(data)

            for i, item in enumerate(response.get("results", [])):
                results.append(SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("text", item.get("snippet", "")),
                    date=item.get("published_date"),
                    score=item.get("score", 1.0),
                    provider="exa",
                    rank=i + 1,
                    extra={
                        "author": item.get("author"),
                        "id": item.get("id"),
                    },
                ))

        except Exception as e:
            logger.warning("Exa search failed: %s", e)

        return results

    # ── SearXNG Implementation ────────────────────────────────────────

    async def _search_searxng(
        self,
        query: str,
        num_results: int,
        filters: Optional[SearchFilter],
    ) -> List[SearchResult]:
        """Search using a self-hosted SearXNG instance.

        Args:
            query: The search query.
            num_results: Number of results to return.
            filters: Optional search filters.

        Returns:
            List of search results.
        """
        config = self._provider_configs.get(SearchProvider.SEARXNG)
        if not config or not config.base_url:
            logger.debug("SearXNG: No base URL configured")
            return []

        results: List[SearchResult] = []
        params: Dict[str, str] = {
            "q": query,
            "format": "json",
            "safesearch": "1" if (filters and filters.safe_search) else "0",
        }

        if filters and filters.language:
            params["language"] = filters.language

        try:
            loop = asyncio.get_event_loop()
            base = config.base_url.rstrip("/")
            data = await loop.run_in_executor(
                None,
                lambda: self._make_request(
                    f"{base}/search", params, timeout=config.timeout
                ),
            )
            response = self._parse_json_response(data)

            for i, item in enumerate(response.get("results", [])):
                if i >= num_results:
                    break
                results.append(SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    engine=item.get("engine", ""),
                    provider="searxng",
                    rank=i + 1,
                    extra={
                        "engines": item.get("engines", []),
                        "category": item.get("category"),
                        "parsed_url": item.get("parsed_url"),
                    },
                ))

        except Exception as e:
            logger.warning("SearXNG search failed: %s", e)

        return results

    # ── Unified Search Methods ────────────────────────────────────────

    async def search(
        self,
        query: str,
        num_results: Optional[int] = None,
        provider: Optional[SearchProvider] = None,
        filters: Optional[SearchFilter] = None,
    ) -> List[SearchResult]:
        """Search the web using the specified or default provider.

        Implements multi-provider failover: if the primary provider
        fails, it automatically tries the next available provider.

        Args:
            query: The search query string.
            num_results: Maximum number of results to return.
                Defaults to the engine's default_num_results.
            provider: Specific provider to use. If None, uses
                the default_provider with failover to others.
            filters: Optional search filters for date, domain, etc.

        Returns:
            A list of SearchResult objects, ranked by relevance.

        Raises:
            ValueError: If the query is empty.
            RuntimeError: If all providers fail.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        query = query.strip()
        num = num_results or self.default_num_results

        # Check cache
        cache_key = self._cache_key(
            query,
            provider.value if provider else "multi",
            num,
        )
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.debug("Returning cached results for: %s", query)
            return cached

        # Get provider search functions
        search_fns = {
            SearchProvider.DUCKDUCKGO: self._search_duckduckgo,
            SearchProvider.GOOGLE_BUILTIN: self._search_google_builtin,
            SearchProvider.TAVILY: self._search_tavily,
            SearchProvider.BRAVE: self._search_brave,
            SearchProvider.EXA: self._search_exa,
            SearchProvider.SEARXNG: self._search_searxng,
        }

        if provider:
            # Use specific provider
            if provider not in search_fns:
                raise ValueError(f"Unknown provider: {provider}")

            rate_limiter = self._rate_limiters.get(provider)
            if rate_limiter:
                await rate_limiter.acquire()

            try:
                results = await search_fns[provider](query, num, filters)
                self._provider_failures[provider] = 0
            except Exception as e:
                logger.error(
                    "Provider %s failed: %s", provider.value, e
                )
                self._provider_failures[provider] = (
                    self._provider_failures.get(provider, 0) + 1
                )
                results = []
        else:
            # Multi-provider failover
            sorted_providers = self._get_sorted_providers(self.default_provider)

            if not sorted_providers:
                # Fall back to DuckDuckGo if no providers configured
                logger.info("No providers configured, using DuckDuckGo")
                sorted_providers = [
                    ProviderConfig(
                        provider=SearchProvider.DUCKDUCKGO,
                        priority=0,
                        rate_limit=(10, 60.0),
                        timeout=self.timeout,
                    )
                ]

            all_results: List[SearchResult] = []

            for config in sorted_providers:
                rate_limiter = self._rate_limiters.get(config.provider)
                if rate_limiter:
                    await rate_limiter.acquire()

                try:
                    fn = search_fns[config.provider]
                    results = await fn(query, num, filters)
                    self._provider_failures[config.provider] = 0

                    if results:
                        all_results.extend(results)
                        logger.info(
                            "Provider %s returned %d results",
                            config.provider.value,
                            len(results),
                        )
                        # If we have enough results from one provider, stop
                        if len(all_results) >= num:
                            break

                except Exception as e:
                    logger.error(
                        "Provider %s failed: %s",
                        config.provider.value,
                        e,
                    )
                    self._provider_failures[config.provider] = (
                        self._provider_failures.get(config.provider, 0) + 1
                    )
                    continue

            results = all_results

        # Post-process results
        results = self._deduplicate(results)
        results = self._rank_results(results)
        results = results[:num]

        # Apply filters
        if filters:
            results = [r for r in results if filters.matches(r)]
            results = results[:num]

        # Cache results
        self._set_cache(cache_key, results)

        logger.info(
            "Search '%s' returned %d results", query, len(results)
        )
        return results

    async def search_news(
        self,
        query: str,
        num_results: Optional[int] = None,
        provider: Optional[SearchProvider] = None,
        filters: Optional[SearchFilter] = None,
    ) -> List[SearchResult]:
        """Search for news articles.

        Args:
            query: The news search query.
            num_results: Maximum number of results.
            provider: Specific provider to use.
            filters: Optional search filters.

        Returns:
            A list of news-related SearchResult objects.
        """
        num = num_results or self.default_num_results

        # Ensure date filter for news (last 7 days by default)
        if not filters:
            filters = SearchFilter(max_age_days=7)
        elif not filters.date_from:
            filters = SearchFilter(
                date_from=filters.date_from,
                date_to=filters.date_to,
                domains=filters.domains if filters.domains else None,
                exclude_domains=filters.exclude_domains if filters.exclude_domains else None,
                language=filters.language,
                region=filters.region,
                safe_search=filters.safe_search,
                max_age_days=7,
            )

        if provider:
            if provider == SearchProvider.TAVILY:
                return await self._search_tavily(
                    query, num, filters, search_type="news"
                )
            # For other providers, add news-related keywords
            news_query = f"{query} news latest"
            return await self.search(news_query, num, provider, filters)
        else:
            # Try Tavily news first (if configured), then DuckDuckGo
            providers_to_try = []
            if SearchProvider.TAVILY in self._provider_configs:
                providers_to_try.append(SearchProvider.TAVILY)
            providers_to_try.append(SearchProvider.DUCKDUCKGO)

            all_results: List[SearchResult] = []
            for p in providers_to_try:
                config = self._provider_configs.get(p)
                if config and not config.enabled:
                    continue
                try:
                    if p == SearchProvider.TAVILY:
                        results = await self._search_tavily(
                            query, num, filters, search_type="news"
                        )
                    else:
                        news_query = f"{query} news latest"
                        results = await self.search(
                            news_query, num, p, filters
                        )
                    all_results.extend(results)
                    if len(all_results) >= num:
                        break
                except Exception as e:
                    logger.warning("News search with %s failed: %s", p.value, e)

            all_results = self._deduplicate(all_results)
            all_results = self._rank_results(all_results)
            return all_results[:num]

    async def search_images(
        self,
        query: str,
        num_results: Optional[int] = None,
        provider: Optional[SearchProvider] = None,
    ) -> List[SearchResult]:
        """Search for images.

        Args:
            query: The image search query.
            num_results: Maximum number of results.
            provider: Specific provider to use.

        Returns:
            A list of image-related SearchResult objects.
        """
        num = num_results or self.default_num_results

        # Use DuckDuckGo image search (most reliable without API)
        if not provider or provider == SearchProvider.DUCKDUCKGO:
            params: Dict[str, str] = {
                "q": query,
                "iax": "images",
                "ia": "images",
            }
            try:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None,
                    lambda: self._make_request(
                        "https://lite.duckduckgo.com/lite/", params
                    ),
                )
                html = data.decode("utf-8", errors="replace")

                # Extract image URLs
                img_pattern = re.compile(
                    r'<img[^>]+src="(https?://[^"]+)"',
                    re.IGNORECASE,
                )
                img_urls = list(set(img_pattern.findall(html)))[:num]

                results = []
                for i, img_url in enumerate(img_urls):
                    results.append(SearchResult(
                        url=img_url,
                        title=f"Image result for: {query}",
                        snippet="",
                        provider="duckduckgo",
                        rank=i + 1,
                    ))
                return results
            except Exception as e:
                logger.warning("DuckDuckGo image search failed: %s", e)

        # Try Tavily with image inclusion
        if provider == SearchProvider.TAVILY or (
            not provider and SearchProvider.TAVILY in self._provider_configs
        ):
            config = self._provider_configs.get(SearchProvider.TAVILY)
            if config and config.api_key:
                results = await self._search_tavily(
                    query, num, None, search_type="images"
                )
                if results:
                    return results[:num]

        return []

    async def search_with_context(
        self,
        query: str,
        context: str,
        num_results: Optional[int] = None,
        provider: Optional[SearchProvider] = None,
    ) -> List[SearchResult]:
        """Search with additional context for better relevance.

        Enhances the search query with the provided context to improve
        result quality. Useful when searching for information related
        to a specific topic or conversation.

        Args:
            query: The primary search query.
            context: Additional context or topic description.
            num_results: Maximum number of results.
            provider: Specific provider to use.

        Returns:
            A list of contextually relevant SearchResult objects.
        """
        # Combine query with context keywords
        # Extract key phrases from context (simple keyword extraction)
        context_keywords = self._extract_keywords(context)

        # Build enhanced query
        if context_keywords:
            enhanced_query = f"{query} {' '.join(context_keywords[:5])}"
        else:
            enhanced_query = query

        logger.debug(
            "Context-enhanced query: '%s' -> '%s'",
            query,
            enhanced_query,
        )

        # Perform the search with the enhanced query
        results = await self.search(
            enhanced_query, num_results, provider
        )

        # Re-rank based on context relevance
        if context_keywords:
            for result in results:
                relevance = self._calculate_context_relevance(
                    result, context_keywords
                )
                result.score = result.score * (1.0 + relevance)

            results = self._rank_results(results)

        return results

    @staticmethod
    def _extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text.

        Uses a simple TF-based approach to identify the most
        significant words in the text.

        Args:
            text: The text to extract keywords from.
            max_keywords: Maximum number of keywords to return.

        Returns:
            A list of keyword strings.
        """
        # Common stop words to exclude
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "because",
            "but", "and", "or", "if", "while", "about", "this", "that",
            "these", "those", "it", "its", "i", "me", "my", "we", "our",
            "you", "your", "he", "him", "his", "she", "her", "they",
            "them", "their", "what", "which", "who", "whom",
        }

        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        word_freq: Dict[str, int] = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(
            word_freq.items(), key=lambda x: x[1], reverse=True
        )
        return [word for word, _ in sorted_words[:max_keywords]]

    @staticmethod
    def _calculate_context_relevance(
        result: SearchResult, keywords: List[str]
    ) -> float:
        """Calculate how relevant a result is to the context keywords.

        Args:
            result: The search result to score.
            keywords: List of context keywords.

        Returns:
            A relevance score between 0.0 and 1.0.
        """
        if not keywords:
            return 0.0

        combined_text = f"{result.title} {result.snippet}".lower()
        matches = sum(1 for kw in keywords if kw.lower() in combined_text)
        return matches / len(keywords)

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all configured providers.

        Returns:
            A dictionary mapping provider names to their status info.
        """
        status = {}
        for provider, config in self._provider_configs.items():
            failures = self._provider_failures.get(provider, 0)
            status[provider.value] = {
                "enabled": config.enabled,
                "configured": bool(config.api_key or config.base_url or
                                   provider in (SearchProvider.DUCKDUCKGO, SearchProvider.GOOGLE_BUILTIN)),
                "priority": config.priority,
                "consecutive_failures": failures,
                "healthy": failures < self._max_failures_before_skip,
                "api_key_set": bool(config.api_key),
                "base_url": config.base_url,
            }
        return status

    def reset_provider_failures(self, provider: Optional[SearchProvider] = None) -> None:
        """Reset failure counters for one or all providers.

        Args:
            provider: Specific provider to reset. If None, resets all.
        """
        if provider:
            self._provider_failures.pop(provider, None)
        else:
            self._provider_failures.clear()
