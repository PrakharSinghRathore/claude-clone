"""
atlas.i18n.loader - Internationalization manager.

Implements a comprehensive i18n system with JSON-based translation
files, locale fallback chains, template variable interpolation,
plural form handling, and automatic locale detection.

Translation files are loaded from a directory structure like::

    locales/
    ├── en.json
    ├── en_US.json
    ├── zh.json
    ├── zh_CN.json
    ├── zh_TW.json
    ├── ja.json
    ├── de.json
    ├── fr.json
    ├── es.json
    └── pt_BR.json

Each JSON file contains nested key-value pairs::

    {
        "common": {
            "yes": "Yes",
            "no": "No",
            "cancel": "Cancel",
            "confirm": "Confirm",
            "save": "Save",
            "delete": "Delete"
        },
        "messages": {
            "welcome": "Welcome, {name}!",
            "goodbye": "Goodbye, {name}. See you {time}!",
            "item_count": "You have {count, plural, one{# item} other{# items}}."
        },
        "errors": {
            "not_found": "The requested resource was not found.",
            "permission_denied": "You do not have permission to perform this action.",
            "timeout": "The operation timed out after {seconds} seconds."
        }
    }
"""

from __future__ import annotations

import json
import logging
import locale
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Default locale used when no locale is set
DEFAULT_LOCALE = "en"

# Common locale aliases (mapping short codes to full codes)
LOCALE_ALIASES: Dict[str, str] = {
    "en": "en",
    "en_US": "en_US",
    "en_GB": "en_GB",
    "en_AU": "en_AU",
    "zh": "zh",
    "zh_CN": "zh_CN",
    "zh_TW": "zh_TW",
    "zh_HK": "zh_HK",
    "ja": "ja",
    "ja_JP": "ja_JP",
    "ko": "ko",
    "ko_KR": "ko_KR",
    "de": "de",
    "de_DE": "de_DE",
    "de_AT": "de_AT",
    "fr": "fr",
    "fr_FR": "fr_FR",
    "fr_CA": "fr_CA",
    "es": "es",
    "es_ES": "es_ES",
    "es_MX": "es_MX",
    "pt": "pt",
    "pt_BR": "pt_BR",
    "pt_PT": "pt_PT",
    "it": "it",
    "it_IT": "it_IT",
    "nl": "nl",
    "nl_NL": "nl_NL",
    "ru": "ru",
    "ru_RU": "ru_RU",
    "ar": "ar",
    "ar_SA": "ar_SA",
    "hi": "hi",
    "hi_IN": "hi_IN",
    "th": "th",
    "th_TH": "th_TH",
    "vi": "vi",
    "vi_VN": "vi_VN",
    "tr": "tr",
    "tr_TR": "tr_TR",
    "pl": "pl",
    "pl_PL": "pl_PL",
    "uk": "uk",
    "uk_UA": "uk_UA",
    "sv": "sv",
    "sv_SE": "sv_SE",
    "da": "da",
    "da_DK": "da_DK",
    "fi": "fi",
    "fi_FI": "fi_FI",
    "nb": "nb",
    "nb_NO": "nb_NO",
    "cs": "cs",
    "cs_CZ": "cs_CZ",
    "el": "el",
    "el_GR": "el_GR",
    "he": "he",
    "he_IL": "he_IL",
    "id": "id",
    "id_ID": "id_ID",
    "ms": "ms",
    "ms_MY": "ms_MY",
    "ro": "ro",
    "ro_RO": "ro_RO",
    "hu": "hu",
    "hu_HU": "hu_HU",
}

# Plural rules by language family
# Maps locale prefix to a plural category function
# Categories: zero, one, two, few, many, other
PLURAL_RULES: Dict[str, Callable[[int], str]] = {}


def _get_plural_category_en(count: int) -> str:
    """English/Czech/Danish/Dutch/Finnish/Greek/etc plural rules."""
    if count == 1:
        return "one"
    return "other"


def _get_plural_category_fr(count: int) -> str:
    """French/Portuguese plural rules."""
    if count == 0 or count == 1:
        return "one"
    return "other"


def _get_plural_category_ru(count: int) -> str:
    """Russian/Ukrainian/Polish/Croatian/Serbian plural rules."""
    mod10 = count % 10
    mod100 = count % 100

    if mod10 == 1 and mod100 != 11:
        return "one"
    if 2 <= mod10 <= 4 and not (12 <= mod100 <= 14):
        return "few"
    if mod10 == 0 or (5 <= mod10 <= 9) or (11 <= mod100 <= 14):
        return "many"
    return "other"


def _get_plural_category_ar(count: int) -> str:
    """Arabic plural rules (6 forms)."""
    mod100 = count % 100

    if count == 0:
        return "zero"
    if count == 1:
        return "one"
    if count == 2:
        return "two"
    if 3 <= mod100 <= 10:
        return "few"
    if 11 <= mod100 <= 99:
        return "many"
    return "other"


def _get_plural_category_zh(count: int) -> str:
    """Chinese/Japanese/Korean/Vietnamese/Thai plural rules."""
    return "other"


def _get_plural_category_sl(count: int) -> str:
    """Slovenian plural rules."""
    mod100 = count % 100
    if mod100 == 1:
        return "one"
    if mod100 == 2:
        return "two"
    if 3 <= mod100 <= 4:
        return "few"
    return "other"


# Initialize plural rules
for _prefix in ("en", "de", "nl", "sv", "da", "fi", "el", "he", "hu", "it",
                "es", "pt", "cs", "nb", "id", "tr", "ro", "hi"):
    PLURAL_RULES[_prefix] = _get_plural_category_en

for _prefix in ("fr",):
    PLURAL_RULES[_prefix] = _get_plural_category_fr

for _prefix in ("ru", "uk", "pl", "hr", "sr", "sl"):
    PLURAL_RULES[_prefix] = _get_plural_category_ru

for _prefix in ("ar",):
    PLURAL_RULES[_prefix] = _get_plural_category_ar

for _prefix in ("zh", "ja", "ko", "vi", "th"):
    PLURAL_RULES[_prefix] = _get_plural_category_zh

# Default plural rule for unknown languages
_default_plural_rule = _get_plural_category_en


class I18nManager:
    """Internationalization manager with JSON-based translation files.

    Provides translation lookup with fallback chains, template
    variable interpolation, plural form handling, and automatic
    locale detection from the system.

    Example::

        i18n = I18nManager()
        i18n.load_translations(Path("locales"))

        i18n.set_locale("en_US")
        print(i18n.t("common.welcome", name="Alice"))
        # => "Welcome, Alice!"

        print(i18n.t("messages.item_count", count=5))
        # => "You have 5 items."

        i18n.set_locale("zh_CN")
        print(i18n.t("common.yes"))
        # => "是"
    """

    def __init__(
        self,
        default_locale: str = DEFAULT_LOCALE,
        fallback_locale: Optional[str] = None,
    ) -> None:
        """Initialize the I18nManager.

        Args:
            default_locale: The default locale to use.
            fallback_locale: The fallback locale when a key is not found.
                If None, defaults to DEFAULT_LOCALE.
        """
        self._default_locale = default_locale
        self._fallback_locale = fallback_locale or DEFAULT_LOCALE
        self._current_locale = default_locale
        self._translations: Dict[str, Dict[str, str]] = {}
        self._loaded_locales: Set[str] = set()
        self._missing_keys: Set[str] = set()
        self._lock = threading.RLock()
        self._on_missing_callback: Optional[Callable[[str, str], None]] = None
        self._locale_change_callbacks: List[Callable[[str, str], None]] = []
        self._locale_aliases = dict(LOCALE_ALIASES)

        logger.info(
            "I18nManager initialized (default: %s, fallback: %s)",
            default_locale,
            self._fallback_locale,
        )

    def set_locale(self, locale: str) -> None:
        """Set the active locale.

        Args:
            locale: The locale code (e.g. 'en', 'en_US', 'zh_CN').
        """
        with self._lock:
            old_locale = self._current_locale
            normalized = self._normalize_locale(locale)
            self._current_locale = normalized

            # Auto-load if not already loaded
            if normalized not in self._loaded_locales:
                logger.debug(
                    "Locale %s not loaded yet; will use fallback chain",
                    normalized,
                )

            logger.info("Locale changed: %s -> %s", old_locale, normalized)

            # Notify callbacks
            for callback in self._locale_change_callbacks:
                try:
                    callback(old_locale, normalized)
                except Exception as e:
                    logger.error("Locale change callback error: %s", e)

    def get_locale(self) -> str:
        """Get the current active locale.

        Returns:
            The current locale code.
        """
        return self._current_locale

    def get_fallback_locales(self, locale: Optional[str] = None) -> List[str]:
        """Get the fallback chain for a locale.

        The fallback chain is: specific locale -> language code -> default -> fallback.

        For example, 'zh_CN' -> ['zh_CN', 'zh', 'en'].

        Args:
            locale: The locale to get the chain for. Uses current locale if None.

        Returns:
            A list of locale codes in order of preference.
        """
        loc = locale or self._current_locale
        chain: List[str] = []
        seen: Set[str] = set()

        # 1. Full locale (e.g. 'zh_CN')
        normalized = self._normalize_locale(loc)
        if normalized not in seen:
            chain.append(normalized)
            seen.add(normalized)

        # 2. Language code only (e.g. 'zh')
        lang_code = normalized.split("_")[0]
        if lang_code != normalized and lang_code not in seen:
            chain.append(lang_code)
            seen.add(lang_code)

        # 3. Default locale
        if self._default_locale not in seen:
            chain.append(self._default_locale)
            seen.add(self._default_locale)

        # 4. Fallback locale
        if self._fallback_locale not in seen:
            chain.append(self._fallback_locale)
            seen.add(self._fallback_locale)

        return chain

    def t(self, key: str, **kwargs: Any) -> str:
        """Translate a key with optional template variables.

        Looks up the translation for the given key using the
        fallback chain. Supports template variable interpolation
        using {variable} syntax and ICU MessageFormat plural syntax.

        Args:
            key: The translation key (dot-separated for nesting).
            **kwargs: Template variables for interpolation.

        Returns:
            The translated string, or the key itself if not found.
        """
        with self._lock:
            translated = self._lookup(key)

            if translated is None:
                self._missing_keys.add(key)
                if self._on_missing_callback:
                    self._on_missing_callback(key, self._current_locale)
                # Return the key as fallback, with interpolation if kwargs provided
                result = key
                if kwargs:
                    try:
                        result = result.format(**kwargs)
                    except (KeyError, IndexError):
                        pass
                return result

            # Interpolate template variables
            if kwargs:
                translated = self._interpolate(translated, **kwargs)

            return translated

    def t_plural(
        self,
        key: str,
        count: int,
        **kwargs: Any,
    ) -> str:
        """Translate a key with plural form handling.

        Supports ICU MessageFormat plural syntax in the translation
        string::

            "item_count": "You have {count, plural, one{# item} other{# items}}."

        Or simple fallback using the count directly.

        Args:
            key: The translation key.
            count: The count for plural form selection.
            **kwargs: Additional template variables.

        Returns:
            The translated string with the correct plural form.
        """
        with self._lock:
            translated = self._lookup(key)

            if translated is None:
                self._missing_keys.add(key)
                if self._on_missing_callback:
                    self._on_missing_callback(key, self._current_locale)
                return key

            kwargs["count"] = count
            result = self._process_plural(translated, count, **kwargs)
            return result

    def t_exists(self, key: str) -> bool:
        """Check if a translation key exists.

        Args:
            key: The translation key to check.

        Returns:
            True if the key exists in the current locale or fallbacks.
        """
        with self._lock:
            return self._lookup(key) is not None

    def t_choices(
        self, key: str, choices: Dict[str, str], **kwargs: Any
    ) -> str:
        """Translate with explicit choice mapping.

        Similar to Django's {% blocktrans count %} with choices.

        Args:
            key: The translation key.
            choices: A mapping of choice values to template strings.
            **kwargs: Template variables (must include the choice variable).

        Returns:
            The translated string with the selected choice.
        """
        # Find the choice variable (usually 'choice' or first non-count kwarg)
        choice_var = kwargs.pop("choice_var", "choice")
        choice_value = kwargs.get(choice_var, "")

        translated = self._lookup(key)
        if translated is None:
            return choices.get(str(choice_value), str(choice_value))

        # Process choices in the translated string
        result = self._process_choices(translated, choices, choice_value, **kwargs)
        return result

    def add_translation(
        self, locale: str, key: str, value: str
    ) -> None:
        """Add or update a single translation entry.

        Args:
            locale: The locale code.
            key: The translation key (dot-separated).
            value: The translation value.
        """
        with self._lock:
            normalized = self._normalize_locale(locale)
            if normalized not in self._translations:
                self._translations[normalized] = {}

            self._translations[normalized][key] = value
            self._loaded_locales.add(normalized)

            # Remove from missing keys if present
            self._missing_keys.discard(key)

            logger.debug("Added translation: %s:%s = %s", locale, key, value)

    def add_translations(
        self, locale: str, translations: Dict[str, str]
    ) -> None:
        """Add multiple translations for a locale.

        Args:
            locale: The locale code.
            translations: A dictionary of key-value translations.
        """
        with self._lock:
            normalized = self._normalize_locale(locale)
            if normalized not in self._translations:
                self._translations[normalized] = {}

            self._translations[normalized].update(translations)
            self._loaded_locales.add(normalized)

            logger.debug(
                "Added %d translations for locale: %s",
                len(translations),
                locale,
            )

    def load_translations(self, directory: Union[str, Path]) -> int:
        """Load all translation files from a directory.

        Looks for JSON files in the specified directory. Each file
        should be named with the locale code (e.g. 'en.json', 'zh_CN.json').

        Args:
            directory: Path to the translations directory.

        Returns:
            The number of translation files loaded.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning("Translation directory does not exist: %s", directory)
            return 0

        if not dir_path.is_dir():
            logger.warning("Translation path is not a directory: %s", directory)
            return 0

        loaded_count = 0

        for file_path in sorted(dir_path.glob("*.json")):
            try:
                locale_name = file_path.stem  # e.g. 'en', 'zh_CN'

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Flatten nested JSON to dot-separated keys
                flat_translations = self._flatten_dict(data)

                self.add_translations(locale_name, flat_translations)
                loaded_count += 1

                logger.info(
                    "Loaded %d translations from %s (%s)",
                    len(flat_translations),
                    file_path.name,
                    locale_name,
                )

            except json.JSONDecodeError as e:
                logger.error(
                    "Failed to parse %s: %s", file_path.name, e
                )
            except Exception as e:
                logger.error(
                    "Failed to load %s: %s", file_path.name, e
                )

        # If no files loaded, try subdirectories
        if loaded_count == 0:
            for subdir in sorted(dir_path.iterdir()):
                if subdir.is_dir():
                    json_file = subdir / f"{subdir.name}.json"
                    if json_file.exists():
                        try:
                            with open(json_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            flat_translations = self._flatten_dict(data)
                            self.add_translations(subdir.name, flat_translations)
                            loaded_count += 1
                            logger.info(
                                "Loaded %d translations from %s",
                                len(flat_translations),
                                json_file.name,
                            )
                        except Exception as e:
                            logger.error("Failed to load %s: %s", json_file, e)

        logger.info(
            "Total: loaded %d translation files, %d locales available",
            loaded_count,
            len(self._loaded_locales),
        )
        return loaded_count

    def load_translation_file(
        self, file_path: Union[str, Path], locale: Optional[str] = None
    ) -> bool:
        """Load a single translation file.

        Args:
            file_path: Path to the JSON translation file.
            locale: Locale code. If None, derived from filename.

        Returns:
            True if the file was loaded successfully.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("Translation file not found: %s", file_path)
            return False

        loc = locale or path.stem

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            flat_translations = self._flatten_dict(data)
            self.add_translations(loc, flat_translations)

            logger.info(
                "Loaded %d translations from %s for locale %s",
                len(flat_translations),
                path.name,
                loc,
            )
            return True

        except json.JSONDecodeError as e:
            logger.error("Failed to parse %s: %s", path.name, e)
            return False
        except Exception as e:
            logger.error("Failed to load %s: %s", path.name, e)
            return False

    def supported_locales(self) -> List[str]:
        """Get a list of all loaded/supported locales.

        Returns:
            A sorted list of locale codes.
        """
        with self._lock:
            return sorted(self._loaded_locales)

    def get_missing_keys(self) -> Set[str]:
        """Get the set of translation keys that were requested but not found.

        Useful for identifying which translations need to be added.

        Returns:
            A set of missing translation keys.
        """
        with self._lock:
            return set(self._missing_keys)

    def clear_missing_keys(self) -> None:
        """Clear the set of missing translation keys."""
        with self._lock:
            self._missing_keys.clear()

    def on_missing(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for missing translation keys.

        The callback receives (key, locale) when a translation
        key is not found.

        Args:
            callback: A function receiving (key, locale).
        """
        self._on_missing_callback = callback

    def on_locale_change(
        self, callback: Callable[[str, str], None]
    ) -> None:
        """Register a callback for locale changes.

        The callback receives (old_locale, new_locale).

        Args:
            callback: A function receiving (old_locale, new_locale).
        """
        self._locale_change_callbacks.append(callback)

    def detect_locale(self) -> str:
        """Auto-detect the locale from the system.

        Checks environment variables, system locale settings,
        and falls back to DEFAULT_LOCALE.

        Returns:
            The detected locale code.
        """
        # Check environment variables
        env_vars = [
            "LANG", "LANGUAGE", "LC_ALL", "LC_MESSAGES", "LC_CTYPE",
        ]

        for var in env_vars:
            value = os.environ.get(var, "")
            if value:
                # Parse locale from value (e.g. "en_US.UTF-8" -> "en_US")
                locale_str = value.split(".")[0]
                locale_str = locale_str.replace("-", "_")

                if locale_str in self._locale_aliases:
                    return self._locale_aliases[locale_str]

                # Try to extract language code
                lang_code = locale_str.split("_")[0]
                if lang_code in self._locale_aliases:
                    return self._locale_aliases[lang_code]

        # Try Python's locale module
        try:
            sys_locale = locale.getdefaultlocale()[0]
            if sys_locale:
                normalized = sys_locale.replace("-", "_")
                if normalized in self._locale_aliases:
                    return self._locale_aliases[normalized]
                lang_code = normalized.split("_")[0]
                if lang_code in self._locale_aliases:
                    return self._locale_aliases[lang_code]
        except Exception:
            pass

        return DEFAULT_LOCALE

    def register_locale_alias(
        self, alias: str, target: str
    ) -> None:
        """Register a custom locale alias.

        Args:
            alias: The alias locale code.
            target: The target locale code.
        """
        self._locale_aliases[alias] = target

    def get_info(self) -> Dict[str, Any]:
        """Get information about the current i18n state.

        Returns:
            A dictionary with i18n state information.
        """
        with self._lock:
            return {
                "current_locale": self._current_locale,
                "default_locale": self._default_locale,
                "fallback_locale": self._fallback_locale,
                "loaded_locales": sorted(self._loaded_locales),
                "fallback_chain": self.get_fallback_locales(),
                "total_keys_per_locale": {
                    loc: len(self._translations.get(loc, {}))
                    for loc in self._loaded_locales
                },
                "missing_keys_count": len(self._missing_keys),
                "supported_locales": self.supported_locales(),
            }

    def export_translations(
        self, locale: Optional[str] = None
    ) -> Dict[str, str]:
        """Export all translations for a locale.

        Args:
            locale: The locale to export. Uses current locale if None.

        Returns:
            A flat dictionary of key-value translations.
        """
        with self._lock:
            loc = locale or self._current_locale
            return dict(self._translations.get(loc, {}))

    def export_nested(
        self, locale: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export translations as a nested dictionary.

        Args:
            locale: The locale to export.

        Returns:
            A nested dictionary of translations.
        """
        with self._lock:
            loc = locale or self._current_locale
            flat = self._translations.get(loc, {})
            return self._unflatten_dict(flat)

    # ── Private Methods ───────────────────────────────────────────────

    def _normalize_locale(self, locale: str) -> str:
        """Normalize a locale code.

        Handles common variations and applies aliases.

        Args:
            locale: The locale code to normalize.

        Returns:
            The normalized locale code.
        """
        # Replace hyphens with underscores
        normalized = locale.replace("-", "_")

        # Convert to lowercase (except country code)
        parts = normalized.split("_")
        if len(parts) >= 2:
            normalized = f"{parts[0].lower()}_{parts[1].upper()}"
        else:
            normalized = normalized.lower()

        # Apply alias if available
        if normalized in self._locale_aliases:
            return self._locale_aliases[normalized]

        return normalized

    def _lookup(self, key: str) -> Optional[str]:
        """Look up a translation key using the fallback chain.

        Args:
            key: The translation key.

        Returns:
            The translated string, or None if not found.
        """
        chain = self.get_fallback_locales()

        for loc in chain:
            translations = self._translations.get(loc, {})
            if key in translations:
                return translations[key]

        return None

    def _interpolate(self, template: str, **kwargs: Any) -> str:
        """Interpolate template variables into a string.

        Supports both simple {variable} and ICU MessageFormat
        plural/select syntax.

        Args:
            template: The template string with placeholders.
            **kwargs: Variable values.

        Returns:
            The interpolated string.
        """
        # Handle ICU plural syntax: {count, plural, one{...} other{...}}
        result = self._interpolate_plural_icu(template, **kwargs)

        # Handle ICU select syntax: {gender, select, male{...} female{...} other{...}}
        result = self._interpolate_select_icu(result, **kwargs)

        # Handle simple {variable} interpolation
        try:
            result = result.format(**kwargs)
        except (KeyError, IndexError, ValueError) as e:
            logger.debug("Template interpolation warning: %s", e)

        return result

    def _interpolate_plural_icu(
        self, template: str, **kwargs: Any
    ) -> str:
        """Interpolate ICU MessageFormat plural expressions.

        Pattern: {var, plural, one{text} other{text}}

        Args:
            template: The template string.
            **kwargs: Variable values.

        Returns:
            The interpolated string.
        """
        pattern = re.compile(
            r"\{(\w+),\s*plural,\s*"
            r"(zero\{([^}]*)\}\s*)?"
            r"(one\{([^}]*)\}\s*)?"
            r"(two\{([^}]*)\}\s*)?"
            r"(few\{([^}]*)\}\s*)?"
            r"(many\{([^}]*)\}\s*)?"
            r"(other\{([^}]*)\})?"
            r"\}",
            re.DOTALL,
        )

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            count = kwargs.get(var_name, 0)
            if not isinstance(count, int):
                try:
                    count = int(count)
                except (ValueError, TypeError):
                    count = 0

            # Get plural category for current locale
            category = self._get_plural_category(count)

            # Extract forms from the match groups
            forms = {
                "zero": match.group(3),
                "one": match.group(5),
                "two": match.group(7),
                "few": match.group(9),
                "many": match.group(11),
                "other": match.group(13),
            }

            # Select the appropriate form
            selected = forms.get(category) or forms.get("other") or str(count)

            # Replace # with the count value
            result = selected.replace("#", str(count))

            # Recursively interpolate remaining variables
            try:
                result = result.format(**kwargs)
            except (KeyError, IndexError, ValueError):
                pass

            return result

        return pattern.sub(replacer, template)

    def _interpolate_select_icu(
        self, template: str, **kwargs: Any
    ) -> str:
        """Interpolate ICU MessageFormat select expressions.

        Pattern: {var, select, option1{text1} option2{text2} other{default}}

        Args:
            template: The template string.
            **kwargs: Variable values.

        Returns:
            The interpolated string.
        """
        pattern = re.compile(
            r"\{(\w+),\s*select,\s*"
            r"((?:\w+\{[^}]*\}\s*)+)"
            r"\}",
            re.DOTALL,
        )

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            value = str(kwargs.get(var_name, ""))
            options_str = match.group(2)

            # Parse options
            options: Dict[str, str] = {}
            option_pattern = re.compile(r"(\w+)\{([^}]*)\}")
            for opt_match in option_pattern.finditer(options_str):
                options[opt_match.group(1)] = opt_match.group(2)

            # Select the matching option
            result = options.get(value) or options.get("other") or value

            # Recursively interpolate remaining variables
            try:
                result = result.format(**kwargs)
            except (KeyError, IndexError, ValueError):
                pass

            return result

        return pattern.sub(replacer, template)

    def _process_plural(
        self, template: str, count: int, **kwargs: Any
    ) -> str:
        """Process a plural translation.

        Args:
            template: The translation template.
            count: The item count.
            **kwargs: Additional variables.

        Returns:
            The translated string with the correct plural form.
        """
        kwargs["count"] = count
        return self._interpolate(template, **kwargs)

    def _process_choices(
        self,
        template: str,
        choices: Dict[str, str],
        choice_value: Any,
        **kwargs: Any,
    ) -> str:
        """Process a translation with choice mapping.

        Args:
            template: The translation template.
            choices: The choices mapping.
            choice_value: The selected choice value.
            **kwargs: Additional variables.

        Returns:
            The translated string.
        """
        str_value = str(choice_value)
        if str_value in choices:
            selected = choices[str_value]
        else:
            # Fall back to 'other' if available
            selected = choices.get("other", str_value)

        # Interpolate variables
        try:
            selected = selected.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            pass

        return selected

    def _get_plural_category(self, count: int) -> str:
        """Get the plural category for a count in the current locale.

        Args:
            count: The item count.

        Returns:
            A plural category string (zero, one, two, few, many, other).
        """
        # Get the language prefix from the current locale
        lang_prefix = self._current_locale.split("_")[0].lower()

        # Look up the plural rule
        rule_func = PLURAL_RULES.get(lang_prefix, _default_plural_rule)
        return rule_func(count)

    @staticmethod
    def _flatten_dict(
        data: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, str]:
        """Flatten a nested dictionary into dot-separated keys.

        Args:
            data: The nested dictionary.
            parent_key: Prefix for keys (used in recursion).
            sep: Separator between key levels.

        Returns:
            A flat dictionary with dot-separated keys.
        """
        items: List[Tuple[str, str]] = []

        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

            if isinstance(value, dict):
                items.extend(
                    I18nManager._flatten_dict(value, new_key, sep).items()
                )
            elif isinstance(value, str):
                items.append((new_key, value))
            else:
                items.append((new_key, str(value)))

        return dict(items)

    @staticmethod
    def _unflatten_dict(
        flat: Dict[str, str], sep: str = "."
    ) -> Dict[str, Any]:
        """Unflatten a dot-separated key dictionary into nested dict.

        Args:
            flat: The flat dictionary with dot-separated keys.
            sep: Separator between key levels.

        Returns:
            A nested dictionary.
        """
        result: Dict[str, Any] = {}

        for key, value in flat.items():
            parts = key.split(sep)
            current = result

            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

        return result
