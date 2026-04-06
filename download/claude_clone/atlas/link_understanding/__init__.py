"""
Atlas Link Understanding — URL analysis, categorization, and content understanding.

Provides comprehensive link analysis including category detection for 15+ categories,
platform identification for 50+ web services, redirect resolution, and structured
information extraction for code repos, videos, and more.

Classes
-------
LinkAnalyzer
    Analyze, categorize, and understand URLs with a built-in platform database.
LinkCategory
    URL category enum (CODE_REPO, DOCUMENTATION, ARTICLE, VIDEO, etc.).
PlatformInfo
    Information about a recognized web platform.
LinkAnalysis
    Complete analysis result for a URL.
RepoInfo
    Structured information extracted from a code repository URL.
VideoInfo
    Structured information extracted from a video URL.
"""

from .analyzer import (
    LinkAnalysis,
    LinkAnalyzer,
    LinkCategory,
    PlatformInfo,
    RepoInfo,
    VideoInfo,
)

__all__ = [
    "LinkAnalyzer",
    "LinkCategory",
    "PlatformInfo",
    "LinkAnalysis",
    "RepoInfo",
    "VideoInfo",
]
