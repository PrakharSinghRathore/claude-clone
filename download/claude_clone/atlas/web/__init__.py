"""
atlas.web - Web search, fetching, and link analysis utilities.

Provides multi-provider web search, robust page fetching with content
extraction, and intelligent URL/link analysis and categorization.
"""

from atlas.web.search import (
    SearchProvider,
    SearchResult,
    WebSearchEngine,
)
from atlas.web.fetch import (
    WebFetcher,
    FetchResult,
    ContentMetadata,
)
from atlas.web.links import (
    LinkAnalyzer,
    LinkCategory,
    URLAnalysis,
    RepoInfo,
    VideoInfo,
)

__all__ = [
    # Search
    "SearchProvider",
    "SearchResult",
    "WebSearchEngine",
    # Fetch
    "WebFetcher",
    "FetchResult",
    "ContentMetadata",
    # Links
    "LinkAnalyzer",
    "LinkCategory",
    "URLAnalysis",
    "RepoInfo",
    "VideoInfo",
]
