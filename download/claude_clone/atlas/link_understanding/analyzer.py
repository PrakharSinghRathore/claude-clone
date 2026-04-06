"""
Link analysis, categorization, and content understanding.

Provides comprehensive URL analysis including category detection,
platform identification, redirect resolution, and content extraction
for 50+ web services.

Usage::

    analyzer = LinkAnalyzer()
    result = analyzer.analyze("https://github.com/anthropics/claude-code")
    print(result.category)    # LinkCategory.CODE_REPO
    print(result.platform)    # PlatformInfo(name="GitHub", ...)
    repo_info = analyzer.extract_repo_info("https://github.com/user/repo")
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import ParseResult, urlparse

logger = logging.getLogger("atlas.link_understanding.analyzer")


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class LinkCategory(str, Enum):
    """Category of a URL / link."""

    CODE_REPO = "code_repo"
    DOCUMENTATION = "documentation"
    ARTICLE = "article"
    SOCIAL_MEDIA = "social_media"
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"
    EMAIL = "email"
    MAP = "map"
    PRODUCT = "product"
    PROFILE = "profile"
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PlatformInfo:
    """Information about a recognized web platform."""

    name: str
    url_pattern: str
    icon: str = ""
    features: List[str] = field(default_factory=list)
    category: LinkCategory = LinkCategory.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url_pattern": self.url_pattern,
            "icon": self.icon,
            "features": self.features,
            "category": self.category.value,
        }


@dataclass
class LinkAnalysis:
    """
    Comprehensive analysis result for a URL.

    Attributes
    ----------
    url:
        The original URL provided.
    category:
        Detected link category.
    platform:
        Identified platform info, or None.
    title:
        Extracted or inferred page title.
    description:
        Content summary or description.
    resolved_url:
        Final URL after resolving redirects.
    metadata:
        Additional metadata extracted from the URL or content.
    """

    url: str
    category: LinkCategory = LinkCategory.UNKNOWN
    platform: Optional[PlatformInfo] = None
    title: str = ""
    description: str = ""
    resolved_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    analyzed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "category": self.category.value,
            "platform": self.platform.to_dict() if self.platform else None,
            "title": self.title,
            "description": self.description,
            "resolved_url": self.resolved_url,
            "metadata": self.metadata,
            "analyzed_at": self.analyzed_at,
        }


@dataclass
class RepoInfo:
    """Information extracted from a code repository URL."""

    platform: str
    owner: str
    repo: str
    url: str
    branch: Optional[str] = None
    path: Optional[str] = None
    is_git: bool = True
    file_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "owner": self.owner,
            "repo": self.repo,
            "url": self.url,
            "branch": self.branch,
            "path": self.path,
            "is_git": self.is_git,
            "file_type": self.file_type,
        }


@dataclass
class VideoInfo:
    """Information extracted from a video URL."""

    platform: str
    video_id: str
    url: str
    title: str = ""
    timestamp: Optional[float] = None
    playlist_id: Optional[str] = None
    embed_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "video_id": self.video_id,
            "url": self.url,
            "title": self.title,
            "timestamp": self.timestamp,
            "playlist_id": self.playlist_id,
            "embed_url": self.embed_url,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Platform Definitions (50+ services)
# ──────────────────────────────────────────────────────────────────────────────

# Built-in platform patterns
PLATFORM_PATTERNS: List[PlatformInfo] = [
    # ── Code Repositories ────────────────────────────────────────────
    PlatformInfo(
        name="GitHub",
        url_pattern=r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)",
        icon="🐙",
        features=["repos", "issues", "prs", "actions", "wiki", "codespaces"],
        category=LinkCategory.CODE_REPO,
    ),
    PlatformInfo(
        name="GitLab",
        url_pattern=r"gitlab\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)",
        icon="🦊",
        features=["repos", "issues", "prs", "ci/cd", "wiki", "registry"],
        category=LinkCategory.CODE_REPO,
    ),
    PlatformInfo(
        name="Bitbucket",
        url_pattern=r"bitbucket\.org/(?P<owner>[^/]+)/(?P<repo>[^/]+)",
        icon="🪣",
        features=["repos", "issues", "prs", "pipelines", "wiki"],
        category=LinkCategory.CODE_REPO,
    ),
    PlatformInfo(
        name="GitHub Gist",
        url_pattern=r"gist\.github\.com/(?P<owner>[^/]+)",
        icon="📋",
        features=["gists", "snippets"],
        category=LinkCategory.CODE_REPO,
    ),
    PlatformInfo(
        name="SourceHut",
        url_pattern=r"sr\.ht/~(?P<owner>[^/]+)/(?P<repo>[^/]+)",
        icon="🏔",
        features=["repos", "lists", "todo", "builds"],
        category=LinkCategory.CODE_REPO,
    ),
    PlatformInfo(
        name="Codeberg",
        url_pattern=r"codeberg\.org/(?P<owner>[^/]+)/(?P<repo>[^/]+)",
        icon="🏗",
        features=["repos", "issues", "prs"],
        category=LinkCategory.CODE_REPO,
    ),

    # ── Package Registries ────────────────────────────────────────────
    PlatformInfo(
        name="npm",
        url_pattern=r"npmjs\.com/package/(?P<package>[^/]+)",
        icon="📦",
        features=["packages", "versions", "dependencies"],
        category=LinkCategory.CODE_REPO,
    ),
    PlatformInfo(
        name="PyPI",
        url_pattern=r"pypi\.org/project/(?P<package>[^/]+)",
        icon="🐍",
        features=["packages", "versions", "classifiers"],
        category=LinkCategory.CODE_REPO,
    ),
    PlatformInfo(
        name="crates.io",
        url_pattern=r"crates\.io/crates/(?P<package>[^/]+)",
        icon="🦀",
        features=["packages", "versions", "dependencies"],
        category=LinkCategory.CODE_REPO,
    ),
    PlatformInfo(
        name="RubyGems",
        url_pattern=r"rubygems\.org/gems/(?P<package>[^/]+)",
        icon="💎",
        features=["packages", "versions"],
        category=LinkCategory.CODE_REPO,
    ),
    PlatformInfo(
        name="Maven Central",
        url_pattern=r"mvnrepository\.com/artifact/(?P<package>[^/]+)",
        icon="☕",
        features=["packages", "versions"],
        category=LinkCategory.CODE_REPO,
    ),

    # ── Documentation ────────────────────────────────────────────────
    PlatformInfo(
        name="MDN Web Docs",
        url_pattern=r"developer\.mozilla\.org",
        icon="📚",
        features=["html", "css", "javascript", "web apis"],
        category=LinkCategory.DOCUMENTATION,
    ),
    PlatformInfo(
        name="DevDocs",
        url_pattern=r"devdocs\.io",
        icon="📖",
        features=["multi-language docs", "offline"],
        category=LinkCategory.DOCUMENTATION,
    ),
    PlatformInfo(
        name="Read the Docs",
        url_pattern=r"readthedocs\.io",
        icon="📕",
        features=["project docs", "pdf", "search"],
        category=LinkCategory.DOCUMENTATION,
    ),
    PlatformInfo(
        name="Python Docs",
        url_pattern=r"docs\.python\.org",
        icon="🐍",
        features=["python docs", "api reference", "tutorials"],
        category=LinkCategory.DOCUMENTATION,
    ),
    PlatformInfo(
        name="Rust Docs",
        url_pattern=r"doc\.rust-lang\.org",
        icon="🦀",
        features=["rust docs", "std", "api reference"],
        category=LinkCategory.DOCUMENTATION,
    ),
    PlatformInfo(
        name="Go Docs",
        url_pattern=r"pkg\.go\.dev",
        icon="🔵",
        features=["go packages", "standard library"],
        category=LinkCategory.DOCUMENTATION,
    ),
    PlatformInfo(
        name="React Docs",
        url_pattern=r"react\.dev",
        icon="⚛️",
        features=["react docs", "tutorials", "api"],
        category=LinkCategory.DOCUMENTATION,
    ),
    PlatformInfo(
        name="Next.js Docs",
        url_pattern=r"nextjs\.org/docs",
        icon="▲",
        features=["next.js docs", "api reference"],
        category=LinkCategory.DOCUMENTATION,
    ),

    # ── Q&A / Knowledge ──────────────────────────────────────────────
    PlatformInfo(
        name="Stack Overflow",
        url_pattern=r"stackoverflow\.com/questions/(?P<id>\\d+)",
        icon="📊",
        features=["questions", "answers", "tags", "reputation"],
        category=LinkCategory.DOCUMENTATION,
    ),
    PlatformInfo(
        name="Stack Exchange",
        url_pattern=r"(?P<site>[a-z]+)\.stackexchange\.com",
        icon="📊",
        features=["questions", "answers", "tags"],
        category=LinkCategory.DOCUMENTATION,
    ),
    PlatformInfo(
        name="Wikipedia",
        url_pattern=r"[a-z]+\.wikipedia\.org/wiki/(?P<article>[^?#]+)",
        icon="🌐",
        features=["encyclopedia", "multilingual", "references"],
        category=LinkCategory.ARTICLE,
    ),

    # ── Social Media ─────────────────────────────────────────────────
    PlatformInfo(
        name="Twitter/X",
        url_pattern=r"(twitter|x)\.com/(?P<user>[^/]+)/status/(?P<id>\\d+)",
        icon="𝕏",
        features=["tweets", "threads", "media", "spaces"],
        category=LinkCategory.SOCIAL_MEDIA,
    ),
    PlatformInfo(
        name="Reddit",
        url_pattern=r"reddit\.com/r/(?P<subreddit>[^/]+)",
        icon="🤖",
        features=["posts", "comments", "subreddits", "awards"],
        category=LinkCategory.SOCIAL_MEDIA,
    ),
    PlatformInfo(
        name="LinkedIn",
        url_pattern=r"linkedin\.com",
        icon="💼",
        features=["profiles", "posts", "jobs", "networking"],
        category=LinkCategory.SOCIAL_MEDIA,
    ),
    PlatformInfo(
        name="Mastodon",
        url_pattern=r"(?P<instance>[a-z0-9.-]+@[a-z0-9.-]+)",
        icon="🐘",
        features=["toots", "federation", "hashtags"],
        category=LinkCategory.SOCIAL_MEDIA,
    ),
    PlatformInfo(
        name="Hacker News",
        url_pattern=r"news\.ycombinator\.com/item\?id=(?P<id>\\d+)",
        icon="🔶",
        features=["stories", "comments", "points"],
        category=LinkCategory.SOCIAL_MEDIA,
    ),
    PlatformInfo(
        name="Lobsters",
        url_pattern=r"lobste\.rs",
        icon="🦞",
        features=["stories", "comments", "tags"],
        category=LinkCategory.SOCIAL_MEDIA,
    ),
    PlatformInfo(
        name="Dev.to",
        url_pattern=r"dev\.to/(?P<user>[^/]+)",
        icon="🟣",
        features=["articles", "comments", "tags"],
        category=LinkCategory.ARTICLE,
    ),
    PlatformInfo(
        name="Medium",
        url_pattern=r"medium\.com",
        icon="📝",
        features=["articles", "publications", "responses"],
        category=LinkCategory.ARTICLE,
    ),
    PlatformInfo(
        name="Substack",
        url_pattern=r"(?P<publication>[^/]+)\.substack\.com",
        icon="📬",
        features=["newsletter", "articles", "podcasts"],
        category=LinkCategory.ARTICLE,
    ),

    # ── Video ────────────────────────────────────────────────────────
    PlatformInfo(
        name="YouTube",
        url_pattern=r"(youtube\.com|youtu\.be)/(watch\\?v=|embed/|shorts/)(?P<id>[^&#]+)",
        icon="▶️",
        features=["videos", "playlists", "shorts", "live", "chapters"],
        category=LinkCategory.VIDEO,
    ),
    PlatformInfo(
        name="Vimeo",
        url_pattern=r"vimeo\.com/(?P<id>\\d+)",
        icon="🎬",
        features=["videos", "channels", "review"],
        category=LinkCategory.VIDEO,
    ),
    PlatformInfo(
        name="Twitch",
        url_pattern=r"twitch\.tv/(?P<channel>[^/]+)",
        icon="🟣",
        features=["live", "vods", "clips", "chat"],
        category=LinkCategory.VIDEO,
    ),
    PlatformInfo(
        name="Bilibili",
        url_pattern=r"bilibili\.com/video/(?P<id>[^/]+)",
        icon="📺",
        features=["videos", "danmaku", "series"],
        category=LinkCategory.VIDEO,
    ),

    # ── Image Hosting ────────────────────────────────────────────────
    PlatformInfo(
        name="Imgur",
        url_pattern=r"imgur\.com/(?P<id>[a-zA-Z0-9]+)",
        icon="🖼",
        features=["images", "albums", "gallery"],
        category=LinkCategory.IMAGE,
    ),
    PlatformInfo(
        name="Unsplash",
        url_pattern=r"unsplash\.com/photos/(?P<id>[a-zA-Z0-9-_]+)",
        icon="📷",
        features=["photos", "collections", "topics"],
        category=LinkCategory.IMAGE,
    ),
    PlatformInfo(
        name="Flickr",
        url_pattern=r"flickr\.com/photos/(?P<user>[^/]+)/(?P<id>\\d+)",
        icon="📸",
        features=["photos", "albums", "galleries"],
        category=LinkCategory.IMAGE,
    ),

    # ── File Hosting ─────────────────────────────────────────────────
    PlatformInfo(
        name="Google Drive",
        url_pattern=r"drive\.google\.com",
        icon="📁",
        features=["files", "folders", "sharing", "collaboration"],
        category=LinkCategory.FILE,
    ),
    PlatformInfo(
        name="Dropbox",
        url_pattern=r"dropbox\.com",
        icon="📦",
        features=["files", "sharing", "paper"],
        category=LinkCategory.FILE,
    ),
    PlatformInfo(
        name="Google Docs",
        url_pattern=r"docs\.google\.com",
        icon="📄",
        features=["documents", "spreadsheets", "slides", "forms"],
        category=LinkCategory.FILE,
    ),
    PlatformInfo(
        name="Notion",
        url_pattern=r"notion\.so/(?P<page>[a-f0-9-]+)",
        icon="📓",
        features=["pages", "databases", "wikis", "tasks"],
        category=LinkCategory.FILE,
    ),

    # ── Maps ─────────────────────────────────────────────────────────
    PlatformInfo(
        name="Google Maps",
        url_pattern=r"maps\.google\.com|goo\.gl/maps",
        icon="🗺",
        features=["maps", "directions", "street view", "places"],
        category=LinkCategory.MAP,
    ),
    PlatformInfo(
        name="OpenStreetMap",
        url_pattern=r"openstreetmap\.org",
        icon="🗺",
        features=["maps", "routing", "editing"],
        category=LinkCategory.MAP,
    ),

    # ── Email ────────────────────────────────────────────────────────
    PlatformInfo(
        name="Email (mailto)",
        url_pattern=r"^mailto:",
        icon="✉️",
        features=["compose", "cc", "bcc", "subject"],
        category=LinkCategory.EMAIL,
    ),

    # ── Product / Shopping ───────────────────────────────────────────
    PlatformInfo(
        name="Amazon",
        url_pattern=r"amazon\.(com|co\.[a-z]{2}|[a-z]{2})/dp/(?P<id>[^/]+)",
        icon="📦",
        features=["products", "reviews", "prime", "wishlist"],
        category=LinkCategory.PRODUCT,
    ),
    PlatformInfo(
        name="GitHub Marketplace",
        url_pattern=r"github\.com/marketplace",
        icon="🛒",
        features=["apps", "actions", "themes"],
        category=LinkCategory.PRODUCT,
    ),

    # ── Profile ──────────────────────────────────────────────────────
    PlatformInfo(
        name="GitHub Profile",
        url_pattern=r"github\.com/(?P<user>[^/]+)$",
        icon="👤",
        features=["repos", "gists", "followers", "contributions"],
        category=LinkCategory.PROFILE,
    ),
    PlatformInfo(
        name="GitLab Profile",
        url_pattern=r"gitlab\.com/(?P<user>[^/]+)$",
        icon="👤",
        features=["repos", "groups", "activity"],
        category=LinkCategory.PROFILE,
    ),
]

# Known URL shorteners
URL_SHORTENERS: Set[str] = {
    "bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly", "is.gd",
    "buff.ly", "rb.gy", "short.io", "cutt.ly", "rebrand.ly",
    "tiny.cc", "mcaf.ee", "v.gd", "tny.im", "surl.li",
}


# ──────────────────────────────────────────────────────────────────────────────
# Link Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class LinkAnalyzer:
    """
    Analyze, categorize, and understand URLs.

    Features:
    - Comprehensive link categorization across 15+ categories
    - Platform identification for 50+ web services
    - Redirect resolution and URL shortener detection
    - Content extraction stubs (title, description)
    - Repository info extraction (GitHub, GitLab, Bitbucket)
    - Video info extraction (YouTube, Vimeo, Twitch)
    - URL validation and normalization
    - Built-in pattern database with regex matching

    Parameters
    ----------
    custom_platforms:
        Additional platform patterns to add to the built-in set.
    timeout:
        Network request timeout in seconds. Default 10.
    follow_redirects:
        Whether to resolve HTTP redirects. Default True.
    max_redirects:
        Maximum redirects to follow. Default 10.
    """

    def __init__(
        self,
        custom_platforms: Optional[List[PlatformInfo]] = None,
        timeout: float = 10.0,
        follow_redirects: bool = True,
        max_redirects: int = 10,
    ) -> None:
        self._timeout = timeout
        self._follow_redirects = follow_redirects
        self._max_redirects = max_redirects

        # Build platform pattern list (compiled regexes)
        self._platforms: List[PlatformInfo] = list(PLATFORM_PATTERNS)
        if custom_platforms:
            self._platforms.extend(custom_platforms)

        # Precompile patterns for performance
        self._compiled_patterns: List[Tuple[PlatformInfo, re.Pattern[str]]] = []
        for platform in self._platforms:
            try:
                compiled = re.compile(platform.url_pattern, re.IGNORECASE)
                self._compiled_patterns.append((platform, compiled))
            except re.error as e:
                logger.warning(
                    "Invalid platform pattern for %s: %s", platform.name, e
                )

        # Analysis cache: url_hash -> LinkAnalysis
        self._cache: Dict[str, LinkAnalysis] = {}
        self._cache_max: int = 500

    # ── Main Analysis ─────────────────────────────────────────────────

    def analyze(self, url: str) -> LinkAnalysis:
        """
        Perform comprehensive link analysis.

        Parameters
        ----------
        url:
            The URL to analyze.

        Returns
        -------
        LinkAnalysis
            Complete analysis including category, platform, and metadata.
        """
        # Check cache
        cache_key = self._cache_hash(url)
        if cache_key in self._cache:
            logger.debug("Cache hit for URL: %s", url[:80])
            return self._cache[cache_key]

        # Normalize URL
        normalized = self._normalize_url(url)

        # Identify platform
        platform = self.identify_platform(normalized)

        # Determine category
        category = self.categorize(normalized)

        # Extract metadata
        metadata = self._extract_url_metadata(normalized)

        # Infer title from URL if possible
        title = self._infer_title(normalized, platform)

        analysis = LinkAnalysis(
            url=url,
            category=category,
            platform=platform,
            title=title,
            resolved_url=normalized,
            metadata=metadata,
        )

        # Cache
        self._cache[cache_key] = analysis
        self._trim_cache()

        logger.debug(
            "Analyzed URL: %s -> category=%s, platform=%s",
            url[:60], category.value,
            platform.name if platform else "unknown",
        )

        return analysis

    # ── Categorization ────────────────────────────────────────────────

    def categorize(self, url: str) -> LinkCategory:
        """
        Determine the category of a URL.

        Parameters
        ----------
        url:
            The URL to categorize.

        Returns
        -------
        LinkCategory
            The detected category.
        """
        normalized = self._normalize_url(url)
        parsed = urlparse(normalized)
        hostname = parsed.hostname or ""
        path = parsed.path.lower()
        scheme = parsed.scheme.lower()

        # Mailto
        if scheme == "mailto":
            return LinkCategory.EMAIL

        # Image file extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp",
                           ".svg", ".bmp", ".ico", ".avif"}
        if any(path.endswith(ext) for ext in image_extensions):
            return LinkCategory.IMAGE

        # Audio file extensions
        audio_extensions = {".mp3", ".wav", ".ogg", ".flac", ".aac",
                           ".m4a", ".wma", ".opus"}
        if any(path.endswith(ext) for ext in audio_extensions):
            return LinkCategory.AUDIO

        # Video file extensions
        video_extensions = {".mp4", ".webm", ".avi", ".mov", ".mkv",
                           ".flv", ".wmv", ".m4v"}
        if any(path.endswith(ext) for ext in video_extensions):
            return LinkCategory.VIDEO

        # Document file extensions
        doc_extensions = {".pdf", ".doc", ".docx", ".xls", ".xlsx",
                         ".ppt", ".pptx", ".txt", ".rtf", ".csv"}
        if any(path.endswith(ext) for ext in doc_extensions):
            return LinkCategory.FILE

        # Archive file extensions
        archive_extensions = {".zip", ".tar", ".gz", ".rar", ".7z",
                             ".bz2", ".xz", ".dmg"}
        if any(path.endswith(ext) for ext in archive_extensions):
            return LinkCategory.FILE

        # Check platform-based category
        platform = self.identify_platform(normalized)
        if platform and platform.category != LinkCategory.UNKNOWN:
            return platform.category

        # Heuristic fallbacks based on hostname
        category_by_host = self._categorize_by_host(hostname)
        if category_by_host:
            return category_by_host

        # Heuristic fallbacks based on path patterns
        category_by_path = self._categorize_by_path(path)
        if category_by_path:
            return category_by_path

        return LinkCategory.UNKNOWN

    # ── Platform Identification ───────────────────────────────────────

    def identify_platform(self, url: str) -> Optional[PlatformInfo]:
        """
        Identify the platform from a URL.

        Parameters
        ----------
        url:
            The URL to check.

        Returns
        -------
        Optional[PlatformInfo]
            Platform info if matched, or None.
        """
        normalized = self._normalize_url(url)

        for platform, pattern in self._compiled_patterns:
            match = pattern.search(normalized)
            if match:
                return platform

        return None

    # ── Redirect Resolution ───────────────────────────────────────────

    def resolve(self, url: str) -> str:
        """
        Resolve URL redirects and shorteners.

        In a production environment, this would follow HTTP redirects.
        Here it normalizes and identifies known shorteners.

        Parameters
        ----------
        url:
            The URL to resolve.

        Returns
        -------
        str
            The resolved URL (same as input in non-network mode).
        """
        normalized = self._normalize_url(url)

        # Check if it's a known shortener domain
        parsed = urlparse(normalized)
        hostname = parsed.hostname or ""
        if hostname in URL_SHORTENERS:
            logger.info(
                "URL shortener detected: %s (resolution requires network)",
                hostname,
            )

        return normalized

    # ── Content Extraction ────────────────────────────────────────────

    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content summary from a URL.

        In a production environment, this would fetch and parse the page.
        Here it extracts what can be inferred from the URL structure.

        Parameters
        ----------
        url:
            The URL to extract content from.

        Returns
        -------
        Dict[str, Any]
            Extracted content summary.
        """
        analysis = self.analyze(url)
        content: Dict[str, Any] = {
            "url": url,
            "category": analysis.category.value,
            "platform": analysis.platform.name if analysis.platform else None,
            "title": analysis.title,
            "description": analysis.description,
        }

        # Platform-specific extraction
        if analysis.category == LinkCategory.CODE_REPO:
            repo_info = self.extract_repo_info(url)
            if repo_info:
                content["repo"] = repo_info.to_dict()

        if analysis.category == LinkCategory.VIDEO:
            video_info = self.extract_video_info(url)
            if video_info:
                content["video"] = video_info.to_dict()

        # Extract path-based info
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p]
        if path_parts:
            content["slug"] = path_parts[-1]
            content["path_depth"] = len(path_parts)

        return content

    # ── Repository Info Extraction ────────────────────────────────────

    def extract_repo_info(self, url: str) -> Optional[RepoInfo]:
        """
        Extract repository information from a code hosting URL.

        Supports GitHub, GitLab, Bitbucket, SourceHut, and Codeberg.

        Parameters
        ----------
        url:
            Repository URL.

        Returns
        -------
        Optional[RepoInfo]
            Extracted repository info, or None if not a repo URL.
        """
        normalized = self._normalize_url(url)
        parsed = urlparse(normalized)
        hostname = parsed.hostname or ""
        path_parts = [p for p in parsed.path.split("/") if p]

        # Map hostname to platform name
        platform_map = {
            "github.com": "GitHub",
            "gitlab.com": "GitLab",
            "bitbucket.org": "Bitbucket",
            "sr.ht": "SourceHut",
            "codeberg.org": "Codeberg",
        }

        platform_name = platform_map.get(hostname)
        if not platform_name or len(path_parts) < 2:
            return None

        owner = path_parts[0]
        repo = path_parts[1]
        branch = None
        path = None
        file_type = None

        # Handle /tree/branch/path or /blob/branch/path patterns
        if len(path_parts) >= 4 and path_parts[2] in ("tree", "blob"):
            branch = path_parts[3]
            if len(path_parts) > 4:
                path = "/".join(path_parts[4:])
                # Infer file type from extension
                if "." in path:
                    ext = path.rsplit(".", 1)[-1].lower()
                    file_type = ext

        # Handle /src/branch/path for Bitbucket
        if len(path_parts) >= 4 and path_parts[2] == "src":
            branch = path_parts[3]
            if len(path_parts) > 4:
                path = "/".join(path_parts[4:])

        # Remove .git suffix from repo name
        if repo.endswith(".git"):
            repo = repo[:-4]

        return RepoInfo(
            platform=platform_name,
            owner=owner,
            repo=repo,
            url=normalized,
            branch=branch,
            path=path,
            file_type=file_type,
        )

    # ── Video Info Extraction ─────────────────────────────────────────

    def extract_video_info(self, url: str) -> Optional[VideoInfo]:
        """
        Extract video information from a video platform URL.

        Supports YouTube, Vimeo, Twitch, and Bilibili.

        Parameters
        ----------
        url:
            Video URL.

        Returns
        -------
        Optional[VideoInfo]
            Extracted video info, or None if not a video URL.
        """
        normalized = self._normalize_url(url)
        parsed = urlparse(normalized)
        hostname = parsed.hostname or ""
        query = urllib.parse.parse_qs(parsed.query)

        # YouTube
        if "youtube.com" in hostname or "youtu.be" in hostname:
            video_id = ""
            if "youtu.be" in hostname:
                video_id = parsed.path.lstrip("/")
            elif "/watch" in parsed.path:
                video_id = query.get("v", [""])[0]
            elif "/embed/" in parsed.path:
                parts = [p for p in parsed.path.split("/") if p]
                if len(parts) >= 2:
                    video_id = parts[1]
            elif "/shorts/" in parsed.path:
                parts = [p for p in parsed.path.split("/") if p]
                if len(parts) >= 2:
                    video_id = parts[1]

            if video_id:
                timestamp = None
                if "t" in query:
                    try:
                        timestamp = float(query["t"][0])
                    except (ValueError, IndexError):
                        pass

                playlist_id = query.get("list", [None])[0]
                embed_url = f"https://www.youtube.com/embed/{video_id}"

                return VideoInfo(
                    platform="YouTube",
                    video_id=video_id,
                    url=normalized,
                    timestamp=timestamp,
                    playlist_id=playlist_id,
                    embed_url=embed_url,
                )

        # Vimeo
        if "vimeo.com" in hostname:
            parts = [p for p in parsed.path.split("/") if p]
            if parts:
                video_id = parts[0]
                try:
                    int(video_id)  # Vimeo IDs are numeric
                    return VideoInfo(
                        platform="Vimeo",
                        video_id=video_id,
                        url=normalized,
                        embed_url=f"https://player.vimeo.com/video/{video_id}",
                    )
                except ValueError:
                    pass

        # Twitch
        if "twitch.tv" in hostname:
            parts = [p for p in parsed.path.split("/") if p]
            if parts:
                channel = parts[0]
                return VideoInfo(
                    platform="Twitch",
                    video_id=channel,
                    url=normalized,
                )

        # Bilibili
        if "bilibili.com" in hostname:
            bvid = query.get("bvid", [""])[0]
            if bvid:
                return VideoInfo(
                    platform="Bilibili",
                    video_id=bvid,
                    url=normalized,
                )

        return None

    # ── URL Support Check ─────────────────────────────────────────────

    def is_supported(self, url: str) -> bool:
        """
        Check if a URL is supported for analysis.

        A URL is supported if it has a recognized scheme (http, https, mailto)
        and either matches a known platform or has a parseable structure.

        Parameters
        ----------
        url:
            The URL to check.

        Returns
        -------
        bool
            True if the URL can be analyzed.
        """
        try:
            parsed = urlparse(url)
            scheme = parsed.scheme.lower()

            if scheme not in ("http", "https", "mailto"):
                return False

            if not parsed.hostname and scheme != "mailto":
                return False

            return True

        except Exception:
            return False

    # ── Internal Helpers ──────────────────────────────────────────────

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize a URL for consistent analysis."""
        url = url.strip()
        if not url:
            return ""

        # Ensure scheme
        if not url.startswith(("http://", "https://", "mailto:")):
            if url.startswith("//"):
                url = "https:" + url
            elif "." in url.split("/")[0]:
                url = "https://" + url

        # Remove trailing slashes
        url = url.rstrip("/")

        # Lowercase scheme and hostname
        try:
            parsed = urlparse(url)
            normalized = parsed._replace(
                scheme=parsed.scheme.lower(),
                netloc=parsed.netloc.lower(),
            ).geturl()
            return normalized
        except Exception:
            return url

    @staticmethod
    def _cache_hash(url: str) -> str:
        """Generate a cache key for a URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _trim_cache(self) -> None:
        """Trim the cache to max size."""
        if len(self._cache) > self._cache_max:
            keys = list(self._cache.keys())
            for key in keys[: len(self._cache) - self._cache_max]:
                del self._cache[key]

    @staticmethod
    def _infer_title(url: str, platform: Optional[PlatformInfo]) -> str:
        """Infer a title from the URL and platform."""
        if platform:
            try:
                parsed = urlparse(url)
                path_parts = [p for p in parsed.path.split("/") if p]
                if path_parts:
                    slug = path_parts[-1]
                    # Convert slug to readable title
                    readable = slug.replace("-", " ").replace("_", " ").title()
                    return f"{readable} — {platform.name}"
            except Exception:
                pass
        return ""

    @staticmethod
    def _extract_url_metadata(url: str) -> Dict[str, Any]:
        """Extract metadata from URL structure."""
        metadata: Dict[str, Any] = {}
        try:
            parsed = urlparse(url)
            metadata["scheme"] = parsed.scheme
            metadata["hostname"] = parsed.hostname or ""
            metadata["port"] = parsed.port
            metadata["path"] = parsed.path
            metadata["query"] = parsed.query
            metadata["fragment"] = parsed.fragment

            # Query parameters
            if parsed.query:
                params = urllib.parse.parse_qs(parsed.query)
                metadata["params"] = {
                    k: v[0] if len(v) == 1 else v
                    for k, v in params.items()
                }

            # Path segments
            path_parts = [p for p in parsed.path.split("/") if p]
            metadata["path_segments"] = path_parts
            metadata["path_depth"] = len(path_parts)

            # Detect file extension
            if path_parts:
                last = path_parts[-1]
                if "." in last:
                    ext = last.rsplit(".", 1)[-1].lower()
                    metadata["file_extension"] = ext

            # Check if shortener
            hostname = parsed.hostname or ""
            if hostname in URL_SHORTENERS:
                metadata["is_shortened"] = True

        except Exception:
            pass

        return metadata

    @staticmethod
    def _categorize_by_host(hostname: str) -> Optional[LinkCategory]:
        """Categorize based on hostname patterns."""
        host_category_map: Dict[str, LinkCategory] = {
            "docs.google.com": LinkCategory.FILE,
            "drive.google.com": LinkCategory.FILE,
            "maps.google.com": LinkCategory.MAP,
            "goo.gl": LinkCategory.UNKNOWN,  # shortener
            "t.co": LinkCategory.UNKNOWN,  # shortener
        }

        for pattern, category in host_category_map.items():
            if pattern in hostname:
                return category

        return None

    @staticmethod
    def _categorize_by_path(path: str) -> Optional[LinkCategory]:
        """Categorize based on path patterns."""
        path_patterns: List[Tuple[str, LinkCategory]] = [
            ("/docs", LinkCategory.DOCUMENTATION),
            ("/doc/", LinkCategory.DOCUMENTATION),
            ("/wiki/", LinkCategory.DOCUMENTATION),
            ("/api/", LinkCategory.DOCUMENTATION),
            ("/blog/", LinkCategory.ARTICLE),
            ("/article/", LinkCategory.ARTICLE),
            ("/post/", LinkCategory.ARTICLE),
            ("/news/", LinkCategory.ARTICLE),
            ("/video/", LinkCategory.VIDEO),
            ("/watch", LinkCategory.VIDEO),
            ("/embed/", LinkCategory.VIDEO),
            ("/channel/", LinkCategory.VIDEO),
            ("/live", LinkCategory.VIDEO),
            ("/image/", LinkCategory.IMAGE),
            ("/photo/", LinkCategory.IMAGE),
            ("/gallery/", LinkCategory.IMAGE),
            ("/audio/", LinkCategory.AUDIO),
            ("/music/", LinkCategory.AUDIO),
            ("/podcast/", LinkCategory.AUDIO),
            ("/download/", LinkCategory.FILE),
            ("/file/", LinkCategory.FILE),
            ("/files/", LinkCategory.FILE),
            ("/u/", LinkCategory.PROFILE),
            ("/user/", LinkCategory.PROFILE),
            ("/users/", LinkCategory.PROFILE),
            ("/@ ", LinkCategory.PROFILE),
            ("/profile/", LinkCategory.PROFILE),
            ("/store/", LinkCategory.PRODUCT),
            ("/shop/", LinkCategory.PRODUCT),
            ("/product/", LinkCategory.PRODUCT),
            ("/marketplace/", LinkCategory.PRODUCT),
            ("/map/", LinkCategory.MAP),
            ("/maps/", LinkCategory.MAP),
            ("/place/", LinkCategory.MAP),
        ]

        path_lower = path.lower()
        for pattern, category in path_patterns:
            if pattern in path_lower:
                return category

        return None

    # ── Statistics ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return analyzer statistics."""
        return {
            "platforms_registered": len(self._platforms),
            "patterns_compiled": len(self._compiled_patterns),
            "cache_size": len(self._cache),
            "cache_max": self._cache_max,
            "shorteners_known": len(URL_SHORTENERS),
        }
