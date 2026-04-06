"""
atlas.i18n - Internationalization and localization support.

Provides comprehensive i18n support with JSON-based translation
files, fallback chains, template variable interpolation, and
plural form handling.
"""

from atlas.i18n.loader import I18nManager

__all__ = [
    "I18nManager",
]
