"""
Atlas Web Tools — web search and page reading.

Features:
- Web search using DuckDuckGo (primary) and fallback engines
- Page content extraction and cleaning
- URL metadata extraction (title, description, og tags)
- Rate limiting and in-memory caching
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse, unquote

from atlas.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Rate limiter & cache
# ---------------------------------------------------------------------------

_cache: Dict[str, Dict[str, Any]] = {}
_MAX_CACHE = 500
_CACHE_TTL = 600  # 10 minutes

_last_request_time: float = 0.0
_MIN_REQUEST_INTERVAL: float = 1.0  # 1s between requests


async def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        await asyncio.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
        return entry["data"]
    return None


def _cache_set(key: str, data: Any) -> None:
    if len(_cache) >= _MAX_CACHE:
        # Evict oldest
        oldest_key = min(_cache, key=lambda k: _cache[k]["ts"])
        _cache.pop(oldest_key, None)
    _cache[key] = {"data": data, "ts": time.time()}


def _cache_key(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class _HTMLTextExtractor(HTMLParser):
    """Strip all HTML tags, keeping only visible text."""

    def __init__(self) -> None:
        super().__init__()
        self._result: List[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "noscript", "svg", "head"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "svg", "head"):
            self._skip = False
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "tr", "td", "blockquote"):
            self._result.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = data.strip()
            if text:
                self._result.append(text + " ")

    def get_text(self) -> str:
        raw = "".join(self._result)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r" {2,}", " ", raw)
        return raw.strip()


class _MetaExtractor(HTMLParser):
    """Extract metadata from <meta>, <title>, <og:*> tags."""

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.description = ""
        self.og_title = ""
        self.og_description = ""
        self.og_image = ""
        self.canonical = ""
        self._in_title = False
        self._title_buf: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attrs_dict = dict(attrs)
        if tag == "title":
            self._in_title = True
            self._title_buf = []
        elif tag == "meta":
            name = attrs_dict.get("name", "").lower()
            prop = attrs_dict.get("property", "").lower()
            content = attrs_dict.get("content", "")
            if name == "description":
                self.description = content
            elif prop == "og:title":
                self.og_title = content
            elif prop == "og:description":
                self.og_description = content
            elif prop == "og:image":
                self.og_image = content
        elif tag == "link":
            rel = attrs_dict.get("rel", "").lower()
            href = attrs_dict.get("href", "")
            if rel == "canonical":
                self.canonical = href

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            self.title = "".join(self._title_buf).strip()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_buf.append(data)

    def summary(self) -> Dict[str, str]:
        return {
            "title": self.og_title or self.title,
            "description": self.og_description or self.description,
            "og_image": self.og_image,
            "canonical": self.canonical,
        }


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def atlas_web_search(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo HTML (no API key needed).

    param query (str): — Search query string.
    param num_results (int): — Number of results to return. Default: 5.
    """
    cache_k = _cache_key("search", query, str(num_results))
    cached = _cache_get(cache_k)
    if cached:
        return cached

    try:
        import httpx
    except ImportError:
        return "Error: httpx is required. Install with: pip install httpx"

    await _rate_limit()

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=_HEADERS,
            )
            resp.raise_for_status()

        html = resp.text
        results = _parse_ddg_html(html)

        # Fallback: lite version
        if not results:
            await _rate_limit()
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp2 = await client.get(
                    "https://lite.duckduckgo.com/lite/",
                    params={"q": query},
                    headers=_HEADERS,
                )
                resp2.raise_for_status()
            results = _parse_ddg_lite(resp2.text)

        if not results:
            msg = f"No results found for: {query}"
            _cache_set(cache_k, msg)
            return msg

        lines = [f"Web search results for '{query}':\n"]
        for i, r in enumerate(results[:num_results], 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['url']}")
            if r.get("snippet"):
                lines.append(f"   {r['snippet']}")
            lines.append("")

        output = "\n".join(lines).strip()
        _cache_set(cache_k, output)
        return output

    except Exception as e:
        return f"Error searching the web: {e}"


def _parse_ddg_html(html: str) -> List[Dict[str, str]]:
    results = []
    blocks = re.findall(
        r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a class="result__snippet"[^>]*>(.*?)</a>',
        html, re.DOTALL,
    )
    if not blocks:
        blocks = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'(?:result__snippet[^>]*>(.*?)</a>)?',
            html, re.DOTALL,
        )

    seen = set()
    for url, title, snippet in blocks:
        url = re.sub(r"^//", "https://", url)
        title = re.sub(r"<[^>]+>", "", title).strip()
        snippet = re.sub(r"<[^>]+>", "", snippet or "").strip()
        if url not in seen and title:
            seen.add(url)
            results.append({"url": url, "title": title, "snippet": snippet})
    return results


def _parse_ddg_lite(html: str) -> List[Dict[str, str]]:
    results = []
    links = re.findall(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
    seen = set()
    for url, text in links:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            url = unquote(qs["uddg"][0])
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text and url not in seen:
            seen.add(url)
            results.append({"url": url, "title": text, "snippet": ""})
    return results


async def atlas_fetch_url(url: str, max_chars: int = 20000) -> str:
    """Fetch a URL and return cleaned page content.

    param url (str): — URL to fetch.
    param max_chars (int): — Maximum characters to return. Default: 20000.
    """
    cache_k = _cache_key("fetch", url)
    cached = _cache_get(cache_k)
    if cached:
        return cached

    try:
        import httpx
    except ImportError:
        return "Error: httpx is required. Install with: pip install httpx"

    await _rate_limit()

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=_HEADERS)
            resp.raise_for_status()

        # Extract metadata
        meta = _MetaExtractor()
        try:
            meta.feed(resp.text)
        except Exception:
            pass

        # Extract text
        extractor = _HTMLTextExtractor()
        try:
            extractor.feed(resp.text)
        except Exception:
            pass

        text = extractor.get_text()
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... truncated, total {len(text)} chars]"

        parts = [f"Title: {meta.summary()['title'] or '(no title)'}"]
        parts.append(f"URL: {url}")
        desc = meta.summary()["description"]
        if desc:
            parts.append(f"Description: {desc}")
        parts.append(f"\n{text}")

        output = "\n".join(parts)
        _cache_set(cache_k, output)
        return output

    except Exception as e:
        return f"Error fetching URL {url}: {e}"


async def atlas_url_metadata(url: str) -> str:
    """Extract metadata from a URL (title, description, OG tags) without fetching full content.

    param url (str): — URL to extract metadata from.
    """
    cache_k = _cache_key("meta", url)
    cached = _cache_get(cache_k)
    if cached:
        return cached

    try:
        import httpx
    except ImportError:
        return "Error: httpx is required. Install with: pip install httpx"

    await _rate_limit()

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=_HEADERS)
            resp.raise_for_status()

        meta = _MetaExtractor()
        try:
            meta.feed(resp.text[:100_000])  # Only parse first 100KB for metadata
        except Exception:
            pass

        info = meta.summary()
        lines = [
            f"URL: {url}",
            f"Title: {info['title'] or '(none)'}",
            f"Description: {info['description'] or '(none)'}",
            f"OG Title: {info['og_title'] or '(none)'}",
            f"OG Description: {info['og_description'] or '(none)'}",
            f"OG Image: {info['og_image'] or '(none)'}",
            f"Canonical: {info['canonical'] or '(none)'}",
        ]

        output = "\n".join(lines)
        _cache_set(cache_k, output)
        return output

    except Exception as e:
        return f"Error extracting metadata from {url}: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="atlas_web_search",
    func=atlas_web_search,
    description="Search the web using DuckDuckGo. Returns titles, URLs, and snippets.",
    toolset="web",
)

ToolRegistry.instance().register(
    name="atlas_fetch_url",
    func=atlas_fetch_url,
    description="Fetch a URL and return cleaned page text content with metadata.",
    toolset="web",
)

ToolRegistry.instance().register(
    name="atlas_url_metadata",
    func=atlas_url_metadata,
    description="Extract metadata (title, description, OG tags) from a URL without fetching full content.",
    toolset="web",
)
