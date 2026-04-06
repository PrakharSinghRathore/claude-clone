"""
Manifest Parser — Reads, validates, and serializes Atlas plugin manifests.

The manifest is the single source of truth for a plugin's identity,
permissions, capabilities, and dependencies.  Manifests can be authored
in either JSON or YAML and follow a versioned schema.

Supported schema versions:

* **v1** — Original minimal format (name, version, entry_point).
* **v2** — Extended format (adds tags, icon, min/max atlas version, homepage).

File discovery
--------------
The parser looks for files named ``plugin.json``, ``plugin.yaml``, or
``plugin.yml`` inside a plugin directory.  JSON is preferred when both
exist.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from atlas.plugin_sdk.core import (
    DEFAULT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    PluginCapability,
    PluginManifest,
    PluginPermission,
    PluginValidationError,
    normalize_plugin_name,
    parse_semver,
    validate_plugin_name,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Filenames to look for (in priority order)
MANIFEST_FILENAMES: Tuple[str, ...] = ("plugin.json", "plugin.yaml", "plugin.yml")

# v1 fields
_V1_REQUIRED_FIELDS: Set[str] = {"name", "version"}
_V1_OPTIONAL_FIELDS: Set[str] = {
    "description",
    "author",
    "license",
    "entry_point",
    "permissions",
    "capabilities",
    "dependencies",
}

# v2 adds
_V2_EXTRA_OPTIONAL_FIELDS: Set[str] = {
    "homepage",
    "tags",
    "icon",
    "min_atlas_version",
    "max_atlas_version",
    "schema_version",
}

# Well-known SPDX license identifiers (subset for validation)
_KNOWN_LICENSES: Set[str] = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "GPL-2.0",
    "GPL-3.0",
    "LGPL-2.1",
    "LGPL-3.0",
    "ISC",
    "MPL-2.0",
    "Unlicense",
    "0BSD",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "Proprietary",
}

# Default values for generating new manifests
_DEFAULT_AUTHOR = "Unknown"
_DEFAULT_LICENSE = "MIT"
_DEFAULT_DESCRIPTION = "An Atlas plugin."


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of a manifest validation pass.

    Attributes:
        valid: Whether the manifest is valid.
        errors: List of human-readable error messages.
        warnings: List of non-fatal warnings.
        schema_version: Detected schema version.
    """

    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_version: str = DEFAULT_SCHEMA_VERSION

    def add_error(self, msg: str) -> None:
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: ValidationResult) -> None:
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False


def _validate_name(data: Dict[str, Any], result: ValidationResult) -> str:
    """Extract and validate the plugin name."""
    name = data.get("name", "")
    if not name or not isinstance(name, str):
        result.add_error("'name' is required and must be a non-empty string")
        return ""
    normalized = normalize_plugin_name(name)
    if not validate_plugin_name(normalized):
        result.add_error(
            f"Invalid plugin name {name!r}: must start with a letter, "
            "contain only lowercase letters, digits, hyphens, and underscores"
        )
    if normalized != name:
        result.add_warning(
            f"Plugin name {name!r} was normalized to {normalized!r}"
        )
    return normalized


def _validate_version(data: Dict[str, Any], result: ValidationResult) -> str:
    """Extract and validate the version string."""
    version = data.get("version", "")
    if not version or not isinstance(version, str):
        result.add_error("'version' is required and must be a string")
        return ""
    try:
        parse_semver(version)
    except ValueError:
        result.add_error(
            f"Invalid version {version!r}: must follow semver (e.g. '1.2.3')"
        )
    return version


def _validate_entry_point(data: Dict[str, Any], result: ValidationResult) -> str:
    """Validate the entry_point format (module:callable or module)."""
    ep = data.get("entry_point", "")
    if not ep:
        result.add_warning("'entry_point' is empty — plugin cannot be loaded")
        return ""
    if ":" not in ep:
        result.add_warning(
            f"'entry_point' {ep!r} has no callable separator (':'). "
            "Expected format: 'module.submodule:callable'"
        )
    return str(ep)


def _validate_permissions(
    data: Dict[str, Any], result: ValidationResult
) -> List[PluginPermission]:
    """Parse and validate permission strings."""
    raw_perms = data.get("permissions", [])
    if not isinstance(raw_perms, list):
        result.add_error("'permissions' must be a list of strings")
        return []
    valid_perms: List[PluginPermission] = []
    seen: Set[str] = set()
    for p in raw_perms:
        if not isinstance(p, str):
            result.add_error(f"Permission {p!r} must be a string")
            continue
        if p in seen:
            result.add_warning(f"Duplicate permission {p!r}")
            continue
        seen.add(p)
        try:
            valid_perms.append(PluginPermission(p))
        except ValueError:
            result.add_error(f"Unknown permission {p!r}")
    return valid_perms


def _validate_capabilities(
    data: Dict[str, Any], result: ValidationResult
) -> List[PluginCapability]:
    """Parse and validate capability strings."""
    raw_caps = data.get("capabilities", [])
    if not isinstance(raw_caps, list):
        result.add_error("'capabilities' must be a list of strings")
        return []
    valid_caps: List[PluginCapability] = []
    seen: Set[str] = set()
    for c in raw_caps:
        if not isinstance(c, str):
            result.add_error(f"Capability {c!r} must be a string")
            continue
        if c in seen:
            result.add_warning(f"Duplicate capability {c!r}")
            continue
        seen.add(c)
        try:
            valid_caps.append(PluginCapability(c))
        except ValueError:
            result.add_error(f"Unknown capability {c!r}")
    return valid_caps


def _validate_dependencies(
    data: Dict[str, Any], result: ValidationResult
) -> Dict[str, str]:
    """Validate the dependencies mapping."""
    deps = data.get("dependencies", {})
    if not isinstance(deps, dict):
        result.add_error("'dependencies' must be an object")
        return {}
    validated: Dict[str, str] = {}
    for dep_name, constraint in deps.items():
        if not isinstance(dep_name, str) or not dep_name:
            result.add_error(f"Invalid dependency name {dep_name!r}")
            continue
        if not isinstance(constraint, str):
            result.add_error(
                f"Version constraint for {dep_name!r} must be a string, got {type(constraint)}"
            )
            continue
        validated[dep_name] = constraint
    return validated


def _validate_atlas_version_constraints(
    data: Dict[str, Any], result: ValidationResult
) -> Tuple[str, str]:
    """Validate min/max atlas version fields."""
    min_v = str(data.get("min_atlas_version", ""))
    max_v = str(data.get("max_atlas_version", ""))
    for label, v in (("min_atlas_version", min_v), ("max_atlas_version", max_v)):
        if v:
            try:
                parse_semver(v)
            except ValueError:
                result.add_error(f"{label} {v!r} is not a valid semver string")
    if min_v and max_v:
        try:
            if parse_semver(min_v) > parse_semver(max_v):
                result.add_error(
                    f"min_atlas_version ({min_v}) > max_atlas_version ({max_v})"
                )
        except ValueError:
            pass
    return min_v, max_v


def _validate_schema_version(data: Dict[str, Any], result: ValidationResult) -> str:
    """Detect and validate the manifest schema version."""
    raw = data.get("schema_version", "")
    if not raw:
        result.add_warning(
            "No 'schema_version' specified — assuming v2"
        )
        return DEFAULT_SCHEMA_VERSION
    if raw not in SUPPORTED_SCHEMA_VERSIONS:
        result.add_error(
            f"Unsupported schema_version {raw!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SCHEMA_VERSIONS))}"
        )
        return raw
    return str(raw)


def _validate_license(data: Dict[str, Any], result: ValidationResult) -> str:
    """Validate the license field."""
    lic = str(data.get("license", _DEFAULT_LICENSE))
    if lic and lic not in _KNOWN_LICENSES:
        result.add_warning(
            f"License {lic!r} is not a well-known SPDX identifier. "
            "See https://spdx.org/licenses/"
        )
    return lic


# ---------------------------------------------------------------------------
# ManifestParser
# ---------------------------------------------------------------------------

class ManifestParser:
    """Parses, validates, and serializes Atlas plugin manifests.

    Usage::

        parser = ManifestParser()

        # Read from file
        manifest = parser.parse(Path("my_plugin/plugin.json"))

        # Validate standalone
        result = parser.validate(data_dict)
        if not result.valid:
            for err in result.errors:
                print(err)

        # Generate default
        default = parser.generate_default(name="hello", version="1.0.0")
    """

    def __init__(self, strict: bool = False) -> None:
        """Initialise the parser.

        Args:
            strict: If ``True``, warnings are promoted to errors.
        """
        self.strict = strict

    # -- High-level API ----------------------------------------------------

    def parse(self, file_path: Union[str, Path]) -> PluginManifest:
        """Read a manifest file and return a validated :class:`PluginManifest`.

        Args:
            file_path: Path to a ``plugin.json``, ``plugin.yaml``, or
                       ``plugin.yml`` file.

        Returns:
            A validated, frozen :class:`PluginManifest` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            PluginValidationError: If validation fails (in strict mode) or
                the file format is unrecognised.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")

        data = self._read_file(path)
        logger.debug("Raw manifest data from %s: %s", path, data)

        result = self.validate(data)
        if self.strict and not result.valid:
            raise PluginValidationError(
                f"Manifest validation failed for {path}",
                plugin_name=data.get("name", ""),
                errors=result.errors,
            )

        for warn in result.warnings:
            logger.warning("[%s] %s", data.get("name", "<unknown>"), warn)

        manifest = self.from_dict(data, result)
        logger.info(
            "Parsed manifest %s@%s (schema=%s) from %s",
            manifest.name,
            manifest.version,
            result.schema_version,
            path,
        )
        return manifest

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate a manifest dictionary without parsing it.

        Returns a :class:`ValidationResult` with errors and warnings.
        The validation is *pure* — it does not modify *data*.
        """
        if not isinstance(data, dict):
            result = ValidationResult(valid=False)
            result.add_error("Manifest must be a JSON object / YAML mapping")
            return result

        result = ValidationResult()

        # Schema version
        result.schema_version = _validate_schema_version(data, result)

        # Required fields
        name = _validate_name(data, result)
        version = _validate_version(data, result)

        # Optional fields
        _validate_entry_point(data, result)
        _validate_permissions(data, result)
        _validate_capabilities(data, result)
        _validate_dependencies(data, result)
        _validate_license(data, result)

        # v2 fields
        if result.schema_version == "v2":
            _validate_atlas_version_constraints(data, result)
            tags = data.get("tags", [])
            if not isinstance(tags, list):
                result.add_error("'tags' must be a list")
            else:
                for t in tags:
                    if not isinstance(t, str):
                        result.add_error(f"Tag {t!r} must be a string")
            homepage = data.get("homepage", "")
            if homepage and not isinstance(homepage, str):
                result.add_error("'homepage' must be a string")

        # v1-specific checks
        if result.schema_version == "v1":
            v1_only_fields = {"homepage", "tags", "icon", "min_atlas_version", "max_atlas_version"}
            for f in v1_only_fields:
                if f in data and data[f]:
                    result.add_warning(f"Field {f!r} is a v2 feature (current schema: v1)")

        # Check for unknown fields
        all_known = (
            _V1_REQUIRED_FIELDS
            | _V1_OPTIONAL_FIELDS
            | _V2_EXTRA_OPTIONAL_FIELDS
        )
        for key in data:
            if key not in all_known:
                result.add_warning(f"Unknown field {key!r} — will be ignored")

        if self.strict:
            for w in result.warnings:
                result.add_error(w)

        return result

    def from_dict(
        self,
        data: Dict[str, Any],
        validation_result: Optional[ValidationResult] = None,
    ) -> PluginManifest:
        """Deserialize a dictionary into a :class:`PluginManifest`.

        If *validation_result* is provided, the parser uses the normalized
        name from validation.  Otherwise, the raw values are used.
        """
        if validation_result is not None and validation_result.errors:
            name = normalize_plugin_name(data.get("name", "unknown"))
        else:
            name = normalize_plugin_name(data.get("name", ""))

        return PluginManifest(
            schema_version=data.get("schema_version", DEFAULT_SCHEMA_VERSION),
            name=name,
            version=data.get("version", "0.0.0"),
            description=data.get("description", _DEFAULT_DESCRIPTION),
            author=data.get("author", _DEFAULT_AUTHOR),
            license=data.get("license", _DEFAULT_LICENSE),
            entry_point=data.get("entry_point", ""),
            permissions=data.get("permissions", []),
            capabilities=data.get("capabilities", []),
            dependencies=data.get("dependencies", {}),
            homepage=data.get("homepage", ""),
            tags=data.get("tags", []),
            icon=data.get("icon", ""),
            min_atlas_version=data.get("min_atlas_version", ""),
            max_atlas_version=data.get("max_atlas_version", ""),
        )

    def to_dict(self, manifest: PluginManifest) -> Dict[str, Any]:
        """Serialize a :class:`PluginManifest` to a plain dictionary.

        The output is suitable for ``json.dump()`` or ``yaml.dump()``.
        """
        return manifest.to_dict()

    def to_json(self, manifest: PluginManifest, indent: int = 2) -> str:
        """Serialize to a formatted JSON string."""
        return json.dumps(self.to_dict(manifest), indent=indent, sort_keys=False)

    def to_yaml(self, manifest: PluginManifest) -> str:
        """Serialize to a YAML string.

        Falls back to JSON if PyYAML is not installed.
        """
        try:
            import yaml
            return yaml.dump(self.to_dict(manifest), default_flow_style=False, sort_keys=False)
        except ImportError:
            logger.warning("PyYAML not installed; falling back to JSON output")
            return self.to_json(manifest)

    # -- Discovery ---------------------------------------------------------

    def discover_manifest(self, directory: Union[str, Path]) -> Optional[Path]:
        """Find the manifest file inside *directory*.

        Searches for ``plugin.json``, ``plugin.yaml``, ``plugin.yml`` in that
        priority order.

        Returns:
            Path to the found manifest, or ``None``.
        """
        dir_path = Path(directory).resolve()
        if not dir_path.is_dir():
            return None
        for filename in MANIFEST_FILENAMES:
            candidate = dir_path / filename
            if candidate.exists():
                logger.debug("Found manifest at %s", candidate)
                return candidate
        return None

    def scan_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
    ) -> List[Path]:
        """Scan a directory tree for plugin manifests.

        Args:
            directory: Root directory to scan.
            recursive: If ``True``, recurse into subdirectories.

        Returns:
            List of manifest file paths found.
        """
        root = Path(directory).resolve()
        if not root.is_dir():
            logger.warning("Cannot scan non-directory: %s", root)
            return []

        found: List[Path] = []
        if recursive:
            for entry in root.rglob("*"):
                if entry.name in MANIFEST_FILENAMES:
                    found.append(entry)
        else:
            for entry in root.iterdir():
                if entry.is_dir():
                    manifest = self.discover_manifest(entry)
                    if manifest:
                        found.append(manifest)
                elif entry.name in MANIFEST_FILENAMES:
                    found.append(entry)

        logger.info(
            "Scanned %s (recursive=%s): found %d manifest(s)",
            root,
            recursive,
            len(found),
        )
        return sorted(found)

    # -- Generation --------------------------------------------------------

    def generate_default(
        self,
        name: str,
        version: str = "1.0.0",
        *,
        description: str = _DEFAULT_DESCRIPTION,
        author: str = _DEFAULT_AUTHOR,
        license: str = _DEFAULT_LICENSE,
        entry_point: str = "",
        capabilities: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None,
    ) -> PluginManifest:
        """Generate a new manifest with sensible defaults.

        This is useful for scaffolding new plugins from the CLI.

        Args:
            name: Plugin name (will be normalized).
            version: Initial version (default ``"1.0.0"``).
            description: One-line description.
            author: Author name.
            license: SPDX license identifier.
            entry_point: Dotted import path.
            capabilities: List of capability strings.
            permissions: List of permission strings.

        Returns:
            A validated :class:`PluginManifest`.
        """
        normalized = normalize_plugin_name(name)
        if not validate_plugin_name(normalized):
            raise ValueError(f"Invalid plugin name: {name!r}")

        data: Dict[str, Any] = {
            "schema_version": DEFAULT_SCHEMA_VERSION,
            "name": normalized,
            "version": version,
            "description": description,
            "author": author,
            "license": license,
            "entry_point": entry_point or f"{normalized}.plugin:setup",
            "permissions": permissions or [],
            "capabilities": capabilities or [],
            "dependencies": {},
            "homepage": "",
            "tags": [],
            "icon": "",
            "min_atlas_version": "",
            "max_atlas_version": "",
        }

        result = self.validate(data)
        for w in result.warnings:
            logger.warning("Default manifest warning: %s", w)
        if not result.valid:
            raise PluginValidationError(
                "Generated manifest failed validation",
                plugin_name=normalized,
                errors=result.errors,
            )

        return self.from_dict(data, result)

    def write_manifest(
        self,
        manifest: PluginManifest,
        directory: Union[str, Path],
        format: str = "json",
    ) -> Path:
        """Write a manifest to disk.

        Args:
            manifest: The manifest to write.
            directory: Target directory (created if it does not exist).
            format: ``"json"`` or ``"yaml"``.

        Returns:
            Path to the written file.
        """
        dir_path = Path(directory).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        if format == "yaml":
            content = self.to_yaml(manifest)
            filename = "plugin.yaml"
        elif format == "yml":
            content = self.to_yaml(manifest)
            filename = "plugin.yml"
        else:
            content = self.to_json(manifest)
            filename = "plugin.json"

        target = dir_path / filename
        target.write_text(content, encoding="utf-8")
        logger.info("Wrote manifest to %s", target)
        return target

    # -- Internals ---------------------------------------------------------

    def _read_file(self, path: Path) -> Dict[str, Any]:
        """Read a JSON or YAML file and return its data as a dict.

        Raises:
            ValueError: If the file format is not recognised or the
                content cannot be parsed.
        """
        suffix = path.suffix.lower()
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Cannot read {path}: {exc}") from exc

        if suffix == ".json":
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        elif suffix in (".yaml", ".yml"):
            data = self._parse_yaml(raw, path)
        else:
            raise ValueError(
                f"Unrecognised manifest format: {suffix}. "
                f"Expected one of: .json, .yaml, .yml"
            )

        if not isinstance(data, dict):
            raise ValueError(f"Manifest root in {path} must be an object/mapping")

        return data

    @staticmethod
    def _parse_yaml(raw: str, path: Path) -> Dict[str, Any]:
        """Parse YAML content, with a fallback error message."""
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(
                f"PyYAML is required to parse {path}. "
                "Install it with: pip install pyyaml"
            ) from exc

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Manifest root in {path} must be an object/mapping")
        return data
