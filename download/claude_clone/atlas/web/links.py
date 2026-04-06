"""
atlas.web.links - URL and link analysis utilities.

Provides intelligent URL analysis, categorization, and information
extraction for various platforms and URL types. Includes detection
of URL shorteners, code repositories, video platforms, and more.
"""

from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class LinkCategory(Enum):
    """Categories for URL classification."""
    CODE_REPO = "code_repo"
    DOCUMENTATION = "documentation"
    ARTICLE = "article"
    BLOG = "blog"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    FORUM = "forum"
    WIKI = "wiki"
    ECOMMERCE = "ecommerce"
    EMAIL = "email"
    FILE = "file"
    SEARCH_ENGINE = "search_engine"
    SHORTENER = "shortener"
    GOVERNMENT = "government"
    EDUCATION = "education"
    COMPANY = "company"
    PORTFOLIO = "portfolio"
    FEED = "feed"
    API = "api"
    DATABASE = "database"
    PACKAGE = "package"
    DATASET = "dataset"
    UNKNOWN = "unknown"


@dataclass
class URLAnalysis:
    """Comprehensive analysis of a URL.

    Attributes:
        url: The original URL.
        normalized_url: The normalized version of the URL.
        domain: The domain name.
        tld: The top-level domain.
        subdomain: The subdomain (if any).
        path: The URL path.
        query_params: Query parameters as a dictionary.
        fragment: The URL fragment.
        scheme: The URL scheme.
        category: The detected category.
        platform: The detected platform name (e.g. 'github', 'youtube').
        is_https: Whether the URL uses HTTPS.
        is_shortened: Whether the URL appears to be shortened.
        expanded_url: The expanded URL (if shortened).
        is_valid: Whether the URL is syntactically valid.
        security_issues: List of potential security concerns.
        tags: Additional tags describing the URL.
    """
    url: str = ""
    normalized_url: str = ""
    domain: str = ""
    tld: str = ""
    subdomain: str = ""
    path: str = ""
    query_params: Dict[str, str] = field(default_factory=dict)
    fragment: str = ""
    scheme: str = ""
    category: LinkCategory = LinkCategory.UNKNOWN
    platform: str = ""
    is_https: bool = False
    is_shortened: bool = False
    expanded_url: Optional[str] = None
    is_valid: bool = False
    security_issues: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "url": self.url,
            "normalized_url": self.normalized_url,
            "domain": self.domain,
            "tld": self.tld,
            "subdomain": self.subdomain,
            "path": self.path,
            "query_params": self.query_params,
            "fragment": self.fragment,
            "scheme": self.scheme,
            "category": self.category.value,
            "platform": self.platform,
            "is_https": self.is_https,
            "is_shortened": self.is_shortened,
            "expanded_url": self.expanded_url,
            "is_valid": self.is_valid,
            "security_issues": self.security_issues,
            "tags": self.tags,
        }


@dataclass
class RepoInfo:
    """Information extracted from a code repository URL.

    Attributes:
        url: The repository URL.
        platform: The platform (github, gitlab, bitbucket, etc.).
        owner: The repository owner or organization.
        name: The repository name.
        full_name: The full repository name (owner/name).
        is_fork: Whether the repository is a fork.
        branch: The branch or reference.
        path: A specific file or directory path.
        is_raw: Whether the URL points to raw content.
        is_archive: Whether the URL points to an archive download.
        commit: A specific commit hash.
        issue_or_pr: Issue or pull request number (if applicable).
        extra: Additional platform-specific information.
    """
    url: str = ""
    platform: str = ""
    owner: str = ""
    name: str = ""
    full_name: str = ""
    is_fork: bool = False
    branch: str = ""
    path: str = ""
    is_raw: bool = False
    is_archive: bool = False
    commit: str = ""
    issue_or_pr: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None and v != ""
        }


@dataclass
class VideoInfo:
    """Information extracted from a video URL.

    Attributes:
        url: The video URL.
        platform: The video platform (youtube, vimeo, etc.).
        video_id: The platform-specific video ID.
        title: The video title (if available).
        is_embed: Whether this is an embed URL.
        is_short: Whether this is a short-form video.
        is_live: Whether this is a live stream.
        timestamp: Optional timestamp (in seconds) for time-offset URLs.
        playlist_id: Playlist ID (if applicable).
        channel_id: Channel ID (if applicable).
        extra: Additional platform-specific information.
    """
    url: str = ""
    platform: str = ""
    video_id: str = ""
    title: str = ""
    is_embed: bool = False
    is_short: bool = False
    is_live: bool = False
    timestamp: Optional[int] = None
    playlist_id: str = ""
    channel_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None and v != ""
        }


# Known URL shortener domains
SHORTENER_DOMAINS: Set[str] = {
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "bit.do", "mcaf.ee",
    "su.pr", "cli.gs", "lnkd.in", "db.tt", "qr.ae",
    "j.mp", "v.gd", "tr.im", "lnkd.in", "dft.ba",
    "tiny.cc", "shorte.st", "cutt.ly", "rebrand.ly",
    "rb.gy", "soo.gd", "clicky.me", "shorturl.at",
    "tny.im", "u.nu", "1pt.co", "shrtco.de",
    "short.io", "surl.li", "clever.ly", "env.sh",
}

# Platform detection patterns
PLATFORM_PATTERNS: Dict[str, Tuple[str, re.Pattern]] = {}

# Code repository platform patterns
REPO_PATTERNS: Dict[str, Tuple[re.Pattern, str]] = {}

# Video platform patterns
VIDEO_PATTERNS: Dict[str, Tuple[re.Pattern, str]] = {}


def _init_patterns() -> None:
    """Initialize all regex patterns. Called once on first use."""
    if PLATFORM_PATTERNS:
        return

    # Social media platforms
    social_patterns = {
        "twitter": r"(?:twitter\.com|x\.com)/(?:#!/)?(\w+)(?:/status(?:es)?/(\d+))?",
        "facebook": r"(?:facebook\.com|fb\.com)/(?:profile\.php\?id=|pages/|groups/)?([\w.-]+)",
        "instagram": r"instagram\.com/(?:p/|reel/|tv/)?(\w+)",
        "linkedin": r"linkedin\.com/in/([\w-]+)",
        "reddit": r"reddit\.com/r/([\w]+)/?(?:comments/(\w+))?",
        "mastodon": r"([\w.-]+)/@(\w+)",
        "discord": r"discord(?:\.com|\.gg)/(?:invite/|channels/)([\w-]+)",
        "slack": r"([\w-]+)\.slack\.com",
        "telegram": r"t\.me/(?:share/url\?)?(\w+)",
        "whatsapp": r"wa\.me/(\d+)",
        "tiktok": r"tiktok\.com/@([\w.-]+)/video/(\d+)",
        "threads": r"threads\.net/@(\w+)/post/(\w+)",
        "bluesky": r"bsky\.app/profile/([\w.]+)/post/(\w+)",
    }

    for platform, pattern_str in social_patterns.items():
        PLATFORM_PATTERNS[platform] = (
            LinkCategory.SOCIAL_MEDIA,
            re.compile(pattern_str, re.IGNORECASE),
        )

    # Code repository platforms
    repo_pattern_strs = {
        "github": (
            r"github\.com/([\w.-]+)/([\w.-]+)"
            r"(?:/(tree|blob|raw|commits|issues|pull)/([^/]+)/(.+))?"
            r"(?:/(issues|pull)/(\d+))?"
            r"(?:/archive/(.+))?",
            "github",
        ),
        "gitlab": (
            r"(?:gitlab\.com|[\w.-]+\.gitlab\.[\w]+)/([\w.-]+)/([\w.-]+)"
            r"(?:/-/([^/]+)/(.+))?"
            r"(?:/(issues)/(\d+))?",
            "gitlab",
        ),
        "bitbucket": (
            r"bitbucket\.org/([\w.-]+)/([\w.-]+)"
            r"(?:/(src|raw)/([^/]+)/(.+))?"
            r"(?:/(issues)/(\d+))?",
            "bitbucket",
        ),
        "codeberg": (
            r"codeberg\.org/([\w.-]+)/([\w.-]+)"
            r"(?:/(src|raw|commit|issues|pulls)/([^/]+)/(.+))?",
            "codeberg",
        ),
        "gitea": (
            r"gitea\.com/([\w.-]+)/([\w.-]+)",
            "gitea",
        ),
        "sourcehut": (
            r"sr\.ht/~([\w]+)/([\w]+)",
            "sourcehut",
        ),
    }

    for platform, (pattern_str, name) in repo_pattern_strs.items():
        REPO_PATTERNS[platform] = (
            re.compile(pattern_str, re.IGNORECASE),
            name,
        )

    # Video platforms
    video_pattern_strs = {
        "youtube": (
            r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/|live/|v/)|youtu\.be/)([\w-]+)"
            r"(?:\?(?:.*&)?t=(\d+))?"
            r"(?:\?(?:.*&)?list=([\w-]+))?",
            "youtube",
        ),
        "vimeo": (
            r"(?:vimeo\.com/|player\.vimeo\.com/video/)(\d+)",
            "vimeo",
        ),
        "twitch": (
            r"twitch\.tv/([\w]+)(?:/video/(\d+))?",
            "twitch",
        ),
        "dailymotion": (
            r"dailymotion\.com/video/([\w]+)",
            "dailymotion",
        ),
        "peertube": (
            r"([\w.-]+)/videos/watch/([\w-]+)",
            "peertube",
        ),
        "bilibili": (
            r"bilibili\.com/video/([\w]+)",
            "bilibili",
        ),
    }

    for platform, (pattern_str, name) in video_pattern_strs.items():
        VIDEO_PATTERNS[platform] = (
            re.compile(pattern_str, re.IGNORECASE),
            name,
        )

    # Documentation platforms
    doc_patterns = {
        "readthedocs": r"([\w.-]+)\.readthedocs\.io",
        "docs_rs": r"docs\.rs/([\w-]+)",
        "mdn": r"developer\.mozilla\.org",
        "python_docs": r"docs\.python\.org",
        "numpy_docs": r"numpy\.org/doc",
        "pytorch_docs": r"pytorch\.org/docs",
    }

    for platform, pattern_str in doc_patterns.items():
        PLATFORM_PATTERNS[platform] = (
            LinkCategory.DOCUMENTATION,
            re.compile(pattern_str, re.IGNORECASE),
        )

    # Wiki platforms
    wiki_patterns = {
        "wikipedia": r"([\w]+)\.wikipedia\.org",
        "wikimedia": r"([\w]+)\.wikimedia\.org",
        "fandom": r"([\w]+)\.fandom\.com",
        "wikihow": r"wikihow\.com",
    }

    for platform, pattern_str in wiki_patterns.items():
        PLATFORM_PATTERNS[platform] = (
            LinkCategory.WIKI,
            re.compile(pattern_str, re.IGNORECASE),
        )

    # News platforms
    news_patterns = {
        "reuters": r"reuters\.com",
        "ap_news": r"apnews\.com",
        "bbc": r"bbc\.(?:com|co\.uk)",
        "cnn": r"cnn\.com",
        "nytimes": r"nytimes\.com",
        "guardian": r"theguardian\.com",
        "washington_post": r"washingtonpost\.com",
        "wsj": r"wsj\.com",
        "bloomberg": r"bloomberg\.com",
    }

    for platform, pattern_str in news_patterns.items():
        PLATFORM_PATTERNS[platform] = (
            LinkCategory.NEWS,
            re.compile(pattern_str, re.IGNORECASE),
        )

    # E-commerce platforms
    ecommerce_patterns = {
        "amazon": r"amazon\.(?:com|co\.uk|de|fr|jp|ca|com\.au|com\.br|com\.mx|in|it|es|nl|se|sg)",
        "ebay": r"ebay\.(?:com|co\.uk|de|fr|ca|com\.au)",
        "etsy": r"etsy\.com",
        "shopify": r"myshopify\.com",
        "walmart": r"walmart\.com",
    }

    for platform, pattern_str in ecommerce_patterns.items():
        PLATFORM_PATTERNS[platform] = (
            LinkCategory.ECOMMERCE,
            re.compile(pattern_str, re.IGNORECASE),
        )

    # Package/registry platforms
    package_patterns = {
        "npm": r"npmjs\.com/package/([\w@.-]+)",
        "pypi": r"pypi\.org/project/([\w.-]+)",
        "crates": r"crates\.io/crates/([\w-]+)",
        "maven": r"mvnrepository\.com/artifact/([\w.]+)",
        "docker_hub": r"hub\.docker\.com/r/([\w/]+)",
        "homebrew": r"formulae\.brew\.sh/formula/([\w@.-]+)",
    }

    for platform, pattern_str in package_patterns.items():
        PLATFORM_PATTERNS[platform] = (
            LinkCategory.PACKAGE,
            re.compile(pattern_str, re.IGNORECASE),
        )

    # Feed patterns
    feed_patterns = {
        "rss": r"feed(?:s)?\.rss",
        "atom": r"atom\.xml",
    }

    for platform, pattern_str in feed_patterns.items():
        PLATFORM_PATTERNS[platform] = (
            LinkCategory.FEED,
            re.compile(pattern_str, re.IGNORECASE),
        )

    # Education platforms
    education_patterns = {
        "coursera": r"coursera\.org",
        "udemy": r"udemy\.com",
        "edx": r"edx\.org",
        "khan_academy": r"khanacademy\.org",
        "mit_ocw": r"ocw\.mit\.edu",
    }

    for platform, pattern_str in education_patterns.items():
        PLATFORM_PATTERNS[platform] = (
            LinkCategory.EDUCATION,
            re.compile(pattern_str, re.IGNORECASE),
        )


class LinkAnalyzer:
    """URL and link analysis utility.

    Provides comprehensive URL analysis including categorization,
    platform detection, repository info extraction, video info
    extraction, and URL shortener detection.

    Example::

        analyzer = LinkAnalyzer()

        # Analyze a URL
        analysis = analyzer.analyze("https://github.com/user/repo")
        print(analysis.category)  # LinkCategory.CODE_REPO
        print(analysis.platform)  # "github"

        # Categorize a URL
        category = analyzer.categorize("https://youtube.com/watch?v=abc123")
        print(category)  # LinkCategory.VIDEO

        # Extract repo info
        repo_info = analyzer.extract_repo_info("https://github.com/python/cpython")
        print(repo_info.owner)  # "python"
        print(repo_info.name)   # "cpython"
    """

    # File extensions by content type
    IMAGE_EXTENSIONS: Set[str] = {
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
        ".ico", ".bmp", ".tiff", ".tif", ".avif", ".heic",
    }
    AUDIO_EXTENSIONS: Set[str] = {
        ".mp3", ".wav", ".ogg", ".flac", ".aac", ".wma",
        ".m4a", ".opus", ".mid", ".midi",
    }
    VIDEO_EXTENSIONS: Set[str] = {
        ".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv",
        ".wmv", ".m4v", ".3gp",
    }
    DOCUMENT_EXTENSIONS: Set[str] = {
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt",
        ".pptx", ".odt", ".ods", ".odp", ".rtf", ".txt",
    }
    ARCHIVE_EXTENSIONS: Set[str] = {
        ".zip", ".tar", ".gz", ".rar", ".7z", ".bz2",
        ".xz", ".tgz", ".tbz2",
    }
    CODE_EXTENSIONS: Set[str] = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c",
        ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".rb",
        ".php", ".swift", ".kt", ".scala", ".r", ".lua",
        ".perl", ".sh", ".bash", ".zsh", ".fish", ".ps1",
        ".sql", ".html", ".css", ".scss", ".sass", ".less",
        ".vue", ".svelte", ".dart", ".zig", ".nim", ".julia",
    }

    # Government TLDs
    GOVERNMENT_TLDS: Set[str] = {
        ".gov", ".mil", ".gouv", ".govt", ".gov.au",
        ".gc.ca", ".gov.uk", ".gov.br", ".gov.in",
        ".go.jp", ".gov.cn", ".gov.ru", ".gov.za",
    }

    # Education TLDs
    EDUCATION_TLDS: Set[str] = {
        ".edu", ".ac", ".academy", ".school", ".university",
        ".college", ".education",
    }

    def __init__(self, expand_short_urls: bool = True) -> None:
        """Initialize the LinkAnalyzer.

        Args:
            expand_short_urls: Whether to automatically expand
                shortened URLs during analysis.
        """
        self.expand_short_urls = expand_short_urls
        self._expanded_cache: Dict[str, str] = {}
        _init_patterns()

    def analyze(self, url: str) -> URLAnalysis:
        """Perform a comprehensive analysis of a URL.

        Extracts all available information from the URL including
        category, platform, security issues, and metadata.

        Args:
            url: The URL to analyze.

        Returns:
            A URLAnalysis object with all extracted information.
        """
        _init_patterns()

        analysis = URLAnalysis(url=url)

        # Validate and parse URL
        parsed = self._parse_url(url)
        if not parsed:
            analysis.is_valid = False
            analysis.security_issues.append("Invalid URL syntax")
            return analysis

        analysis.is_valid = True
        analysis.scheme = parsed.scheme
        analysis.domain = parsed.netloc
        analysis.path = parsed.path
        analysis.fragment = parsed.fragment
        analysis.is_https = parsed.scheme == "https"

        # Parse query parameters
        if parsed.query:
            analysis.query_params = dict(
                urllib.parse.parse_qsl(parsed.query)
            )

        # Extract TLD, subdomain, etc.
        analysis.tld = self._extract_tld(analysis.domain)
        analysis.subdomain = self._extract_subdomain(analysis.domain)

        # Normalize URL
        analysis.normalized_url = self._normalize_url(url)

        # Check for shorteners
        analysis.is_shortened = self.is_shortened(url)

        # Categorize
        analysis.category = self.categorize(url)

        # Detect platform
        analysis.platform = self._detect_platform(url)

        # Security analysis
        analysis.security_issues = self._check_security(url, parsed)

        # Generate tags
        analysis.tags = self._generate_tags(url, analysis)

        # Expand short URL if enabled
        if self.expand_short_urls and analysis.is_shortened:
            try:
                expanded = self.expand_short_url(url)
                if expanded and expanded != url:
                    analysis.expanded_url = expanded
                    # Also categorize the expanded URL
                    expanded_analysis = self.analyze(expanded)
                    if expanded_analysis.category != LinkCategory.SHORTENER:
                        analysis.category = expanded_analysis.category
                        analysis.platform = expanded_analysis.platform
            except Exception as e:
                logger.debug("Failed to expand short URL %s: %s", url, e)

        return analysis

    def categorize(self, url: str) -> LinkCategory:
        """Categorize a URL into a known category.

        Uses pattern matching against known platforms and URL
        structure heuristics to determine the category.

        Args:
            url: The URL to categorize.

        Returns:
            The detected LinkCategory.
        """
        _init_patterns()

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        # Check by file extension first
        ext = self._get_extension(path)
        if ext in self.IMAGE_EXTENSIONS:
            return LinkCategory.IMAGE
        elif ext in self.AUDIO_EXTENSIONS:
            return LinkCategory.AUDIO
        elif ext in self.VIDEO_EXTENSIONS:
            return LinkCategory.VIDEO
        elif ext in self.DOCUMENT_EXTENSIONS:
            return LinkCategory.FILE
        elif ext in self.ARCHIVE_EXTENSIONS:
            return LinkCategory.FILE
        elif ext in self.CODE_EXTENSIONS:
            return LinkCategory.CODE_REPO

        # Check platform patterns
        for platform_name, (category, pattern) in PLATFORM_PATTERNS.items():
            if pattern.search(url):
                return category

        # Check repo patterns
        for platform_name, (pattern, name) in REPO_PATTERNS.items():
            if pattern.search(url):
                return LinkCategory.CODE_REPO

        # Check video patterns
        for platform_name, (pattern, name) in VIDEO_PATTERNS.items():
            if pattern.search(url):
                return LinkCategory.VIDEO

        # Check for URL shorteners
        if self.is_shortened(url):
            return LinkCategory.SHORTENER

        # Check for email links
        if url.startswith("mailto:"):
            return LinkCategory.EMAIL

        # Check for RSS/Atom feeds
        if ext in (".rss", ".xml") or "feed" in path or "rss" in path:
            return LinkCategory.FEED

        # Check for API endpoints
        if "/api/" in path or path.endswith("/api"):
            return LinkCategory.API

        # Check for datasets
        dataset_indicators = {"dataset", "data", "csv", "json", "download"}
        if any(ind in domain for ind in dataset_indicators):
            return LinkCategory.DATASET

        # Check by TLD
        tld = self._extract_tld(domain)
        if tld in self.GOVERNMENT_TLDS:
            return LinkCategory.GOVERNMENT
        elif tld in self.EDUCATION_TLDS:
            return LinkCategory.EDUCATION

        # Heuristic: check for blog indicators
        blog_indicators = {
            "blog", "medium.com", "substack.com", "dev.to",
            "hashnode.com", "ghost.io", "wordpress.com",
        }
        if any(ind in domain or ind in path for ind in blog_indicators):
            return LinkCategory.BLOG

        # Heuristic: check for forum indicators
        forum_indicators = {
            "forum", "discourse", "community", "stackexchange",
            "stack overflow", "quora", "reddit",
        }
        if any(ind in domain for ind in forum_indicators):
            return LinkCategory.FORUM

        # Heuristic: check for documentation indicators
        doc_indicators = {
            "docs", "documentation", "readthedocs", "wiki",
            "api-docs", "reference", "guide",
        }
        if any(ind in path for ind in doc_indicators):
            return LinkCategory.DOCUMENTATION

        # Heuristic: article detection
        article_indicators = {
            "article", "story", "post", "news", "journal",
        }
        if any(ind in path for ind in article_indicators):
            return LinkCategory.ARTICLE

        # Check for search engines
        search_engines = {
            "google.com/search", "bing.com/search", "duckduckgo.com",
            "yahoo.com/search", "baidu.com/s",
        }
        if any(se in url for se in search_engines):
            return LinkCategory.SEARCH_ENGINE

        # Check for company pages
        company_indicators = {"corp", "inc", "ltd", "company", "enterprise"}
        if any(ind in domain for ind in company_indicators):
            return LinkCategory.COMPANY

        # Check for portfolio pages
        portfolio_indicators = {"portfolio", "cv", "resume", "about-me"}
        if any(ind in domain or ind in path for ind in portfolio_indicators):
            return LinkCategory.PORTFOLIO

        return LinkCategory.UNKNOWN

    def extract_repo_info(self, url: str) -> Optional[RepoInfo]:
        """Extract repository information from a URL.

        Supports GitHub, GitLab, Bitbucket, Codeberg, Gitea,
        and SourceHut URLs.

        Args:
            url: The repository URL.

        Returns:
            A RepoInfo object, or None if the URL is not a
            recognized repository URL.
        """
        _init_patterns()

        for platform_name, (pattern, name) in REPO_PATTERNS.items():
            match = pattern.search(url)
            if not match:
                continue

            groups = match.groups()
            info = RepoInfo(
                url=url,
                platform=name,
            )

            # Parse based on platform structure
            if platform_name == "github":
                info.owner = groups[0] if len(groups) > 0 else ""
                info.name = groups[1] if len(groups) > 1 else ""
                info.full_name = f"{info.owner}/{info.name}"

                # Check for specific paths
                if len(groups) > 2 and groups[2]:
                    action = groups[2].lower()
                    if action in ("tree", "blob", "raw", "commits"):
                        info.branch = groups[3] if len(groups) > 3 else ""
                        info.path = groups[4] if len(groups) > 4 else ""
                        info.is_raw = action == "raw"
                    elif action in ("issues", "pull"):
                        try:
                            info.issue_or_pr = int(groups[3]) if len(groups) > 3 else None
                        except (ValueError, TypeError):
                            pass

                if len(groups) > 5 and groups[5]:
                    try:
                        info.issue_or_pr = int(groups[5])
                    except (ValueError, TypeError):
                        pass

                if "archive" in url:
                    info.is_archive = True

                # Detect forks
                if "/fork" in url or "fork" in (urlparse(url).fragment or ""):
                    info.is_fork = True

            elif platform_name == "gitlab":
                info.owner = groups[0] if len(groups) > 0 else ""
                info.name = groups[1] if len(groups) > 1 else ""
                info.full_name = f"{info.owner}/{info.name}"
                if len(groups) > 2 and groups[2]:
                    info.branch = groups[2]
                    info.path = groups[3] if len(groups) > 3 else ""

            elif platform_name == "bitbucket":
                info.owner = groups[0] if len(groups) > 0 else ""
                info.name = groups[1] if len(groups) > 1 else ""
                info.full_name = f"{info.owner}/{info.name}"
                if len(groups) > 2 and groups[2]:
                    info.branch = groups[3] if len(groups) > 3 else ""
                    info.path = groups[4] if len(groups) > 4 else ""
                    if groups[2] == "raw":
                        info.is_raw = True

            elif platform_name == "codeberg":
                info.owner = groups[0] if len(groups) > 0 else ""
                info.name = groups[1] if len(groups) > 1 else ""
                info.full_name = f"{info.owner}/{info.name}"
                if len(groups) > 2 and groups[2]:
                    info.branch = groups[3] if len(groups) > 3 else ""
                    info.path = groups[4] if len(groups) > 4 else ""
                    if groups[2] == "raw":
                        info.is_raw = True

            elif platform_name == "sourcehut":
                info.owner = groups[0] if len(groups) > 0 else ""
                info.name = groups[1] if len(groups) > 1 else ""
                info.full_name = f"~{info.owner}/{info.name}"

            elif platform_name == "gitea":
                info.owner = groups[0] if len(groups) > 0 else ""
                info.name = groups[1] if len(groups) > 1 else ""
                info.full_name = f"{info.owner}/{info.name}"

            return info

        return None

    def extract_video_info(self, url: str) -> Optional[VideoInfo]:
        """Extract video information from a URL.

        Supports YouTube, Vimeo, Twitch, Dailymotion, PeerTube,
        and Bilibili URLs.

        Args:
            url: The video URL.

        Returns:
            A VideoInfo object, or None if the URL is not a
            recognized video URL.
        """
        _init_patterns()

        for platform_name, (pattern, name) in VIDEO_PATTERNS.items():
            match = pattern.search(url)
            if not match:
                continue

            groups = match.groups()
            info = VideoInfo(
                url=url,
                platform=name,
            )

            if platform_name == "youtube":
                info.video_id = groups[0] if len(groups) > 0 else ""
                if len(groups) > 1 and groups[1]:
                    try:
                        info.timestamp = int(groups[1])
                    except (ValueError, TypeError):
                        pass
                if len(groups) > 2 and groups[2]:
                    info.playlist_id = groups[2]

                # Check for specific YouTube URL types
                parsed = urlparse(url)
                if "/embed/" in parsed.path:
                    info.is_embed = True
                elif "/shorts/" in parsed.path:
                    info.is_short = True
                elif "/live/" in parsed.path:
                    info.is_live = True

            elif platform_name == "vimeo":
                info.video_id = groups[0] if len(groups) > 0 else ""
                if "player.vimeo.com" in url:
                    info.is_embed = True

            elif platform_name == "twitch":
                if len(groups) > 0 and groups[0]:
                    info.channel_id = groups[0]
                if len(groups) > 1 and groups[1]:
                    info.video_id = groups[1]

            elif platform_name == "dailymotion":
                info.video_id = groups[0] if len(groups) > 0 else ""

            elif platform_name == "bilibili":
                info.video_id = groups[0] if len(groups) > 0 else ""

            elif platform_name == "peertube":
                if len(groups) > 0 and groups[0]:
                    info.extra["instance"] = groups[0]
                info.video_id = groups[1] if len(groups) > 1 else ""

            return info

        return None

    def is_shortened(self, url: str) -> bool:
        """Check if a URL is a known shortened URL.

        Checks the URL against a comprehensive list of known
        URL shortener services.

        Args:
            url: The URL to check.

        Returns:
            True if the URL appears to be a shortened URL.
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Check exact match
            if domain in SHORTENER_DOMAINS:
                return True

            # Check if domain is a shortener subdomain
            for shortener in SHORTENER_DOMAINS:
                if domain.endswith(f".{shortener}"):
                    return True

            # Heuristic: very short paths on well-known domains
            # could indicate short URLs
            if len(parsed.path) < 15 and parsed.path.count("/") == 1:
                # Check for single-segment short URLs
                path_segment = parsed.path.strip("/")
                if path_segment and len(path_segment) <= 10:
                    if re.match(r"^[\w-]+$", path_segment):
                        # Likely a short URL, but we can't be sure
                        # without checking against known services
                        pass

            return False

        except Exception:
            return False

    def expand_short_url(self, url: str, timeout: float = 10.0) -> Optional[str]:
        """Expand a shortened URL to its original destination.

        Follows HTTP redirects to find the final destination URL.

        Args:
            url: The shortened URL to expand.
            timeout: Request timeout in seconds.

        Returns:
            The expanded URL, or the original URL if expansion fails.
        """
        # Check cache
        if url in self._expanded_cache:
            return self._expanded_cache[url]

        try:
            req = urllib.request.Request(url, method="HEAD")
            req.add_header(
                "User-Agent",
                "Mozilla/5.0 (compatible; LinkAnalyzer/1.0)",
            )

            # Don't follow redirects automatically
            class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    raise urllib.error.HTTPError(
                        newurl, code, msg, headers, fp
                    )

            opener = urllib.request.build_opener(NoRedirectHandler)

            try:
                opener.open(req, timeout=timeout)
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308):
                    expanded = e.url
                    self._expanded_cache[url] = expanded
                    logger.debug("Expanded %s -> %s", url, expanded)
                    return expanded

            # If no redirect, the URL itself is the final destination
            self._expanded_cache[url] = url
            return url

        except Exception as e:
            logger.debug("Failed to expand URL %s: %s", url, e)
            return None

    async def expand_short_url_async(
        self, url: str, timeout: float = 10.0
    ) -> Optional[str]:
        """Async version of expand_short_url.

        Args:
            url: The shortened URL to expand.
            timeout: Request timeout in seconds.

        Returns:
            The expanded URL, or None if expansion fails.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.expand_short_url(url, timeout)
        )

    # ── Private Helper Methods ────────────────────────────────────────

    @staticmethod
    def _parse_url(url: str) -> Optional[urlparse]:
        """Parse a URL and validate it.

        Args:
            url: The URL to parse.

        Returns:
            A urlparse result, or None if invalid.
        """
        try:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                return parsed
            return None
        except Exception:
            return None

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize a URL for deduplication.

        Removes trailing slashes, sorts query parameters,
        and removes common tracking parameters.

        Args:
            url: The URL to normalize.

        Returns:
            The normalized URL string.
        """
        try:
            parsed = urlparse(url)

            # Remove tracking parameters
            tracking_params = {
                "utm_source", "utm_medium", "utm_campaign",
                "utm_term", "utm_content", "utm_id",
                "fbclid", "gclid", "gclsrc", "dclid",
                "ref", "referrer", "source",
                "mc_cid", "mc_eid",
                "_ga", "_gl", "_hsenc", "_hsmi",
                "hsCtaTracking", "vero_id",
                "oly_anon_id", "oly_enc_id",
            }

            query_params = urllib.parse.parse_qsl(parsed.query)
            filtered_params = [
                (k, v) for k, v in query_params
                if k.lower() not in tracking_params
            ]
            filtered_params.sort(key=lambda x: x[0])

            # Rebuild URL
            normalized = parsed._replace(
                path=parsed.path.rstrip("/"),
                query=urllib.parse.urlencode(filtered_params),
                fragment="",
            ).geturl()

            # Remove trailing slash (again, after rebuild)
            if normalized.endswith("/") and not normalized.endswith("://"):
                normalized = normalized.rstrip("/")

            return normalized

        except Exception:
            return url

    @staticmethod
    def _extract_tld(domain: str) -> str:
        """Extract the top-level domain from a domain name.

        Args:
            domain: The domain name.

        Returns:
            The TLD including the dot prefix.
        """
        parts = domain.rsplit(".", 2)
        if len(parts) >= 2:
            return f".{parts[-1]}"
        return ""

    @staticmethod
    def _extract_subdomain(domain: str) -> str:
        """Extract the subdomain from a domain name.

        Args:
            domain: The domain name.

        Returns:
            The subdomain, or empty string if none.
        """
        parts = domain.split(".")
        if len(parts) > 2:
            return ".".join(parts[:-2])
        return ""

    @staticmethod
    def _get_extension(path: str) -> str:
        """Get the file extension from a URL path.

        Args:
            path: The URL path.

        Returns:
            The file extension including the dot.
        """
        # Get the last segment of the path
        segments = path.rstrip("/").split("/")
        if not segments:
            return ""

        filename = segments[-1].split("?")[0].split("#")[0]

        # Handle compound extensions like .tar.gz
        if filename.endswith(".tar.gz"):
            return ".tar.gz"
        elif filename.endswith(".tar.bz2"):
            return ".tar.bz2"
        elif filename.endswith(".tar.xz"):
            return ".tar.xz"

        dot_pos = filename.rfind(".")
        if dot_pos >= 0:
            return filename[dot_pos:].lower()
        return ""

    def _detect_platform(self, url: str) -> str:
        """Detect the platform from a URL.

        Args:
            url: The URL to check.

        Returns:
            The platform name, or empty string if unknown.
        """
        _init_patterns()

        # Check platform patterns
        for platform_name, (_, pattern) in PLATFORM_PATTERNS.items():
            if pattern.search(url):
                return platform_name

        # Check repo patterns
        for platform_name, (pattern, name) in REPO_PATTERNS.items():
            if pattern.search(url):
                return name

        # Check video patterns
        for platform_name, (pattern, name) in VIDEO_PATTERNS.items():
            if pattern.search(url):
                return name

        # Check shorteners
        if self.is_shortened(url):
            return "shortener"

        return ""

    @staticmethod
    def _check_security(url: str, parsed: urlparse) -> List[str]:
        """Check for potential security issues in a URL.

        Args:
            url: The URL to check.
            parsed: The parsed URL.

        Returns:
            A list of security issue descriptions.
        """
        issues: List[str] = []

        # Check for HTTP (non-HTTPS)
        if parsed.scheme == "http":
            issues.append("Uses HTTP instead of HTTPS")

        # Check for suspicious domains
        suspicious_tlds = {".tk", ".ml", ".ga", ".cf", ".gq"}
        domain = parsed.netloc.lower()
        for tld in suspicious_tlds:
            if domain.endswith(tld):
                issues.append(f"Uses suspicious TLD: {tld}")
                break

        # Check for IP address URLs
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain):
            issues.append("Uses IP address instead of domain name")

        # Check for very long URLs (potential phishing)
        if len(url) > 500:
            issues.append("Unusually long URL (potential phishing)")

        # Check for excessive subdomains
        parts = domain.split(".")
        if len(parts) > 5:
            issues.append("Excessive subdomains")

        # Check for URL encoding abuse
        encoded_chars = url.count("%")
        if encoded_chars > 20:
            issues.append("Excessive URL encoding (potential obfuscation)")

        # Check for credential in URL
        if "@" in parsed.netloc:
            issues.append("Contains credentials in URL")

        return issues

    @staticmethod
    def _generate_tags(url: str, analysis: URLAnalysis) -> List[str]:
        """Generate descriptive tags for a URL.

        Args:
            url: The URL.
            analysis: The URL analysis.

        Returns:
            A list of tag strings.
        """
        tags: List[str] = []

        tags.append(analysis.category.value)
        if analysis.platform:
            tags.append(analysis.platform)
        if analysis.is_https:
            tags.append("https")
        if analysis.is_shortened:
            tags.append("shortened")

        # Add domain-based tags
        if analysis.tld:
            tags.append(f"tld:{analysis.tld}")

        # Add extension-based tags
        ext = LinkAnalyzer._get_extension(analysis.path)
        if ext:
            tags.append(f"ext:{ext}")

        # Add security-related tags
        if analysis.security_issues:
            tags.append("security-issues")

        return tags

    def batch_analyze(
        self, urls: List[str], expand_short: bool = False
    ) -> List[URLAnalysis]:
        """Analyze multiple URLs at once.

        Args:
            urls: List of URLs to analyze.
            expand_short: Whether to expand short URLs.

        Returns:
            List of URLAnalysis objects.
        """
        original_expand = self.expand_short_urls
        self.expand_short_urls = expand_short
        results = [self.analyze(url) for url in urls]
        self.expand_short_urls = original_expand
        return results

    def get_platform_stats(
        self, urls: List[str]
    ) -> Dict[str, int]:
        """Count URLs by platform.

        Args:
            urls: List of URLs to analyze.

        Returns:
            A dictionary mapping platform names to counts.
        """
        stats: Dict[str, int] = {}
        for url in urls:
            platform = self._detect_platform(url)
            if platform:
                stats[platform] = stats.get(platform, 0) + 1
            else:
                stats["unknown"] = stats.get("unknown", 0) + 1
        return stats

    def get_category_stats(
        self, urls: List[str]
    ) -> Dict[str, int]:
        """Count URLs by category.

        Args:
            urls: List of URLs to analyze.

        Returns:
            A dictionary mapping category names to counts.
        """
        stats: Dict[str, int] = {}
        for url in urls:
            category = self.categorize(url)
            cat_name = category.value
            stats[cat_name] = stats.get(cat_name, 0) + 1
        return stats
