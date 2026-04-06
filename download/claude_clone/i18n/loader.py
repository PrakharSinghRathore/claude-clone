"""
Internationalization loader.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default locale data
_DEFAULT_LOCALE: Dict[str, str] = {
    "memory": "Relevant memories:\n{memory}",
    "lite_agent_system_prompt_with_tools": (
        "You are {role}.\n\n## Backstory\n{backstory}\n\n"
        "## Goal\n{goal}\n\n## Tools\nYou have access to the following tools:\n{tools}\n\n"
        "Use the right format to call a tool: {tool_names}"
    ),
    "lite_agent_system_prompt_without_tools": (
        "You are {role}.\n\n## Backstory\n{backstory}\n\n## Goal\n{goal}"
    ),
    "lite_agent_response_format": (
        "\n\nYou MUST respond with valid JSON matching the following schema:\n{response_format}"
    ),
    "formatted_task_instructions": (
        "\n\nFormat your response as JSON matching this schema:\n{output_format}"
    ),
}


class I18N:
    """
    Internationalization helper.
    
    Manages locale strings and provides formatted slices.
    
    Args:
        locale: Locale code (e.g., 'en', 'es').
        locale_dir: Directory containing locale JSON files.
    """
    
    def __init__(self, locale: str = "en", locale_dir: Optional[str] = None):
        self.locale = locale
        self._strings: Dict[str, str] = dict(_DEFAULT_LOCALE)
        
        if locale_dir:
            self._load_locale(locale_dir, locale)
    
    def _load_locale(self, locale_dir: str, locale: str) -> None:
        """Load locale strings from a JSON file."""
        locale_file = Path(locale_dir) / f"{locale}.json"
        if locale_file.exists():
            try:
                content = locale_file.read_text(encoding="utf-8")
                data = json.loads(content)
                if isinstance(data, dict):
                    self._strings.update(data)
                    logger.info("Loaded %d locale strings from %s", len(data), locale_file)
            except Exception as e:
                logger.error("Failed to load locale %s: %s", locale, e)
    
    def get(self, key: str, default: str = "") -> str:
        """Get a locale string by key."""
        return self._strings.get(key, default)
    
    def set(self, key: str, value: str) -> None:
        """Set a locale string."""
        self._strings[key] = value
    
    def slice(self, key: str) -> str:
        """Get a locale string (alias for get)."""
        return self._strings.get(key, "")
    
    def format(self, key: str, **kwargs: Any) -> str:
        """Get and format a locale string."""
        template = self._strings.get(key, "")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning("Missing format key %s in locale string '%s'", e, key)
            return template
    
    @property
    def available_keys(self) -> list[str]:
        return list(self._strings.keys())


# Default singleton
_default_i18n: Optional[I18N] = None


def get_i18n() -> I18N:
    """Get the global I18N instance."""
    global _default_i18n
    if _default_i18n is None:
        _default_i18n = I18N()
    return _default_i18n
