"""
ACP Permission Management — Tool permission levels, per-session
overrides, permission persistence, and permission templates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path.home() / ".claude_clone" / "acp"
_PERMISSIONS_FILE = "permissions.json"


class PermissionLevel(str, Enum):
    """Permission level for tool access."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"      # Prompt user for confirmation


@dataclass
class ToolPermission:
    """Permission configuration for a single tool."""

    tool_name: str
    level: PermissionLevel = PermissionLevel.ASK
    reason: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class PermissionTemplate:
    """
    A named set of tool permissions that can be applied to sessions.

    Templates allow quick permission configuration for common scenarios.
    """

    name: str
    description: str
    permissions: dict[str, PermissionLevel] = field(default_factory=dict)
    is_default: bool = False


# Pre-defined permission templates
BUILTIN_TEMPLATES: dict[str, PermissionTemplate] = {
    "readonly": PermissionTemplate(
        name="readonly",
        description="Read-only access: no file writes or command execution.",
        permissions={
            "read_file": PermissionLevel.ALLOW,
            "list_directory": PermissionLevel.ALLOW,
            "search_files": PermissionLevel.ALLOW,
            "grep": PermissionLevel.ALLOW,
            "find_definition": PermissionLevel.ALLOW,
            "get_environment": PermissionLevel.ALLOW,
            "get_git_status": PermissionLevel.ALLOW,
            "git_log": PermissionLevel.ALLOW,
            "git_diff": PermissionLevel.ALLOW,
            "web_search": PermissionLevel.ALLOW,
            "fetch_url": PermissionLevel.ALLOW,
            "write_file": PermissionLevel.DENY,
            "edit_file": PermissionLevel.DENY,
            "delete_file": PermissionLevel.DENY,
            "run_command": PermissionLevel.DENY,
            "run_python": PermissionLevel.DENY,
            "run_script": PermissionLevel.DENY,
        },
    ),
    "standard": PermissionTemplate(
        name="standard",
        description="Standard access with confirmation for dangerous operations.",
        permissions={
            "read_file": PermissionLevel.ALLOW,
            "list_directory": PermissionLevel.ALLOW,
            "search_files": PermissionLevel.ALLOW,
            "grep": PermissionLevel.ALLOW,
            "find_definition": PermissionLevel.ALLOW,
            "get_environment": PermissionLevel.ALLOW,
            "get_git_status": PermissionLevel.ALLOW,
            "git_log": PermissionLevel.ALLOW,
            "git_diff": PermissionLevel.ALLOW,
            "web_search": PermissionLevel.ALLOW,
            "fetch_url": PermissionLevel.ALLOW,
            "write_file": PermissionLevel.ASK,
            "edit_file": PermissionLevel.ASK,
            "delete_file": PermissionLevel.ASK,
            "run_command": PermissionLevel.ASK,
            "run_python": PermissionLevel.ASK,
            "run_script": PermissionLevel.ASK,
        },
        is_default=True,
    ),
    "unrestricted": PermissionTemplate(
        name="unrestricted",
        description="Full access to all tools without confirmation.",
        permissions={
            "read_file": PermissionLevel.ALLOW,
            "list_directory": PermissionLevel.ALLOW,
            "search_files": PermissionLevel.ALLOW,
            "grep": PermissionLevel.ALLOW,
            "find_definition": PermissionLevel.ALLOW,
            "get_environment": PermissionLevel.ALLOW,
            "get_git_status": PermissionLevel.ALLOW,
            "git_log": PermissionLevel.ALLOW,
            "git_diff": PermissionLevel.ALLOW,
            "web_search": PermissionLevel.ALLOW,
            "fetch_url": PermissionLevel.ALLOW,
            "write_file": PermissionLevel.ALLOW,
            "edit_file": PermissionLevel.ALLOW,
            "delete_file": PermissionLevel.ALLOW,
            "run_command": PermissionLevel.ALLOW,
            "run_python": PermissionLevel.ALLOW,
            "run_script": PermissionLevel.ALLOW,
        },
    ),
    "code_assist": PermissionTemplate(
        name="code_assist",
        description="Code assistance: allow reads and edits, deny destructive operations.",
        permissions={
            "read_file": PermissionLevel.ALLOW,
            "list_directory": PermissionLevel.ALLOW,
            "search_files": PermissionLevel.ALLOW,
            "grep": PermissionLevel.ALLOW,
            "find_definition": PermissionLevel.ALLOW,
            "lint_python": PermissionLevel.ALLOW,
            "format_python": PermissionLevel.ALLOW,
            "get_git_status": PermissionLevel.ALLOW,
            "git_log": PermissionLevel.ALLOW,
            "git_diff": PermissionLevel.ALLOW,
            "write_file": PermissionLevel.ASK,
            "edit_file": PermissionLevel.ASK,
            "delete_file": PermissionLevel.DENY,
            "run_command": PermissionLevel.ASK,
            "run_python": PermissionLevel.ASK,
            "run_script": PermissionLevel.ASK,
        },
    ),
}


class PermissionManager:
    """
    Manages tool permissions with per-session overrides,
    persistence, and template support.
    """

    def __init__(
        self,
        data_dir: str | Path = _DEFAULT_DATA_DIR,
        default_template: str = "standard",
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._default_template_name = default_template
        self._templates: dict[str, PermissionTemplate] = dict(BUILTIN_TEMPLATES)
        self._session_overrides: dict[str, dict[str, PermissionLevel]] = {}  # session_id -> {tool: level}
        self._global_permissions: dict[str, PermissionLevel] = {}
        self._load()

    # ------------------------------------------------------------------
    # Permission checking
    # ------------------------------------------------------------------

    def check_permission(
        self,
        tool_name: str,
        session_id: Optional[str] = None,
    ) -> PermissionLevel:
        """
        Check the permission level for a tool.

        Resolution order:
        1. Session-specific override
        2. Global permission override
        3. Default template
        4. Default to ASK
        """
        # Check session override
        if session_id and session_id in self._session_overrides:
            session_perms = self._session_overrides[session_id]
            if tool_name in session_perms:
                return session_perms[tool_name]

        # Check global override
        if tool_name in self._global_permissions:
            return self._global_permissions[tool_name]

        # Check default template
        template = self._templates.get(self._default_template_name)
        if template and tool_name in template.permissions:
            return template.permissions[tool_name]

        # Fallback to ASK
        return PermissionLevel.ASK

    def is_allowed(self, tool_name: str, session_id: Optional[str] = None) -> bool:
        """Return True if the tool is explicitly allowed."""
        return self.check_permission(tool_name, session_id) == PermissionLevel.ALLOW

    def requires_ask(self, tool_name: str, session_id: Optional[str] = None) -> bool:
        """Return True if the tool requires user confirmation."""
        return self.check_permission(tool_name, session_id) == PermissionLevel.ASK

    def is_denied(self, tool_name: str, session_id: Optional[str] = None) -> bool:
        """Return True if the tool is explicitly denied."""
        return self.check_permission(tool_name, session_id) == PermissionLevel.DENY

    # ------------------------------------------------------------------
    # Permission modification
    # ------------------------------------------------------------------

    def set_permission(
        self,
        tool_name: str,
        level: PermissionLevel,
        session_id: Optional[str] = None,
    ) -> None:
        """Set permission for a tool, optionally scoped to a session."""
        if session_id:
            self._session_overrides.setdefault(session_id, {})[tool_name] = level
        else:
            self._global_permissions[tool_name] = level
        self._save()

    def reset_permission(self, tool_name: str, session_id: Optional[str] = None) -> None:
        """Reset a tool permission to the template default."""
        if session_id and session_id in self._session_overrides:
            self._session_overrides[session_id].pop(tool_name, None)
        elif tool_name in self._global_permissions:
            del self._global_permissions[tool_name]
        self._save()

    def set_session_permissions(
        self,
        session_id: str,
        permissions: dict[str, PermissionLevel],
    ) -> None:
        """Set multiple permissions for a session at once."""
        self._session_overrides[session_id] = permissions
        self._save()

    def clear_session_permissions(self, session_id: str) -> None:
        """Remove all session-specific permission overrides."""
        self._session_overrides.pop(session_id, None)
        self._save()

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def get_template(self, name: str) -> Optional[PermissionTemplate]:
        """Get a permission template by name."""
        return self._templates.get(name)

    def list_templates(self) -> dict[str, PermissionTemplate]:
        """List all available templates."""
        return dict(self._templates)

    def create_template(self, template: PermissionTemplate) -> None:
        """Create or update a custom template."""
        self._templates[template.name] = template
        if template.is_default:
            self._default_template_name = template.name
        self._save()

    def delete_template(self, name: str) -> bool:
        """Delete a custom template. Built-in templates cannot be deleted."""
        if name in BUILTIN_TEMPLATES:
            logger.warning("Cannot delete built-in template %r", name)
            return False
        return self._templates.pop(name, None) is not None

    def apply_template(self, template_name: str, session_id: Optional[str] = None) -> bool:
        """Apply a template's permissions globally or to a session."""
        template = self._templates.get(template_name)
        if template is None:
            return False
        if session_id:
            self._session_overrides[session_id] = dict(template.permissions)
        else:
            self._global_permissions = dict(template.permissions)
        self._save()
        logger.info("Applied template %r to %s", template_name, session_id or "global")
        return True

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_permission_summary(self, session_id: Optional[str] = None) -> dict[str, str]:
        """Get a summary of all tool permissions."""
        # Collect all known tool names from templates
        all_tools: set[str] = set()
        for template in self._templates.values():
            all_tools.update(template.permissions.keys())
        all_tools.update(self._global_permissions.keys())
        for session_perms in self._session_overrides.values():
            all_tools.update(session_perms.keys())

        summary: dict[str, str] = {}
        for tool in sorted(all_tools):
            level = self.check_permission(tool, session_id)
            summary[tool] = level.value
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist permissions to disk."""
        perms_file = self.data_dir / _PERMISSIONS_FILE
        try:
            data = {
                "default_template": self._default_template_name,
                "global_permissions": {k: v.value for k, v in self._global_permissions.items()},
                "session_overrides": {
                    sid: {k: v.value for k, v in perms.items()}
                    for sid, perms in self._session_overrides.items()
                },
                "custom_templates": {
                    name: {
                        "name": t.name,
                        "description": t.description,
                        "permissions": {k: v.value for k, v in t.permissions.items()},
                        "is_default": t.is_default,
                    }
                    for name, t in self._templates.items()
                    if name not in BUILTIN_TEMPLATES
                },
            }
            perms_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            logger.exception("Failed to save permissions")

    def _load(self) -> None:
        """Load permissions from disk."""
        perms_file = self.data_dir / _PERMISSIONS_FILE
        if not perms_file.exists():
            return
        try:
            data = json.loads(perms_file.read_text(encoding="utf-8"))
            self._default_template_name = data.get("default_template", "standard")

            for tool, level_str in data.get("global_permissions", {}).items():
                self._global_permissions[tool] = PermissionLevel(level_str)

            for sid, perms in data.get("session_overrides", {}).items():
                self._session_overrides[sid] = {
                    tool: PermissionLevel(level_str) for tool, level_str in perms.items()
                }

            for name, tdata in data.get("custom_templates", {}).items():
                self._templates[name] = PermissionTemplate(
                    name=tdata["name"],
                    description=tdata.get("description", ""),
                    permissions={
                        k: PermissionLevel(v) for k, v in tdata.get("permissions", {}).items()
                    },
                    is_default=tdata.get("is_default", False),
                )

        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load permissions")
