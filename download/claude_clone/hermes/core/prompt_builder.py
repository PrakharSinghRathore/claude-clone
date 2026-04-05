"""
Dynamic System Prompt Builder for Hermes Agent.

Constructs system prompts by composing togglable sections including agent
identity, platform hints, memory guidance, session search instructions,
skills guidance, tool enforcement, context files, and custom overrides.

Designed to integrate with the existing Claude Clone ``Agent`` context system
while providing a more flexible, section-based approach.

Usage
-----
    builder = PromptBuilder()
    builder.set_section(PromptSection.IDENTITY, "You are Hermes...")
    builder.set_section(PromptSection.CONTEXT, context_str)
    prompt = builder.build()
"""

from __future__ import annotations

import os
import platform
import sys
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional


class PromptSection(Enum):
    """Identifies each togglable section of the system prompt."""

    IDENTITY = auto()
    BEHAVIOR = auto()
    CONTEXT = auto()
    PLATFORM_HINTS = auto()
    MEMORY_GUIDANCE = auto()
    SESSION_SEARCH = auto()
    SKILLS_GUIDANCE = auto()
    TOOL_ENFORCEMENT = auto()
    CONTEXT_FILES = auto()
    SECURITY_RULES = auto()
    CUSTOM_OVERRIDES = auto()
    KNOWLEDGE_BASE = auto()
    PLUGINS = auto()


@dataclass
class PromptSectionData:
    """Holds the content and metadata for a single prompt section."""

    content: str = ""
    enabled: bool = True
    priority: int = 0  # Higher = rendered later (closer to the end)
    separator: str = "\n\n"
    prefix: str = ""
    suffix: str = ""


class PromptBuilder:
    """
    Composable system prompt builder with togglable sections.

    Each section can be independently enabled/disabled, reordered by priority,
    and given custom prefixes/suffixes. The builder integrates with Claude
    Clone's existing context system by accepting a ``context_dict`` that maps
    section names to their content.

    Parameters
    ----------
    agent_name:
        The display name of the agent (default: "Hermes Agent").
    context_dict:
        Optional dict mapping ``PromptSection`` values (or their string names)
        to pre-built content strings.
    cwd:
        Current working directory for context detection (default: ``os.getcwd()``).
    """

    DEFAULT_IDENTITY = (
        "You are an expert agentic AI assistant running inside a local Python environment. "
        "You have full access to the user's file system and terminal via tools. "
        "You think step by step before acting, prefer doing over asking, and chain "
        "tool calls together for efficiency."
    )

    DEFAULT_BEHAVIOR = (
        "## BEHAVIOR\n"
        "- Think step by step before acting.\n"
        "- Prefer doing over asking. Attempt the task, then report.\n"
        "- Chain tool calls together — do not stop and ask after every step.\n"
        "- After a tool call, review output and decide the next action.\n"
        "- If a command fails, diagnose and retry with corrections.\n"
        "- Only ask the user when you are truly blocked."
    )

    DEFAULT_SECURITY = (
        "## RULES\n"
        "- Never delete files without confirmation.\n"
        "- Never expose secrets, API keys, or credentials.\n"
        "- Never run destructive commands without approval.\n"
        "- Always read a file before editing it.\n"
        "- After writing code, run it to verify it works.\n"
        "- Provide clear, concise responses with actionable information."
    )

    def __init__(
        self,
        agent_name: str = "Hermes Agent",
        context_dict: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._agent_name = agent_name
        self._sections: Dict[PromptSection, PromptSectionData] = {}

        # Initialize default sections
        self._init_default_sections()

        # Apply any provided context overrides
        if context_dict:
            for key, value in context_dict.items():
                try:
                    section = PromptSection[key.upper()] if isinstance(key, str) else key
                    self.set_section(section, value)
                except (KeyError, ValueError):
                    self.set_custom_override(key, value)

    def _init_default_sections(self) -> None:
        """Initialize all sections with their defaults."""
        section_defaults = {
            PromptSection.IDENTITY: (self.DEFAULT_IDENTITY, 100),
            PromptSection.BEHAVIOR: (self.DEFAULT_BEHAVIOR, 90),
            PromptSection.CONTEXT: ("", 80),
            PromptSection.PLATFORM_HINTS: ("", 70),
            PromptSection.MEMORY_GUIDANCE: ("", 60),
            PromptSection.SESSION_SEARCH: ("", 55),
            PromptSection.SKILLS_GUIDANCE: ("", 50),
            PromptSection.TOOL_ENFORCEMENT: ("", 45),
            PromptSection.CONTEXT_FILES: ("", 40),
            PromptSection.SECURITY_RULES: (self.DEFAULT_SECURITY, 30),
            PromptSection.CUSTOM_OVERRIDES: ("", 20),
            PromptSection.KNOWLEDGE_BASE: ("", 15),
            PromptSection.PLUGINS: ("", 10),
        }
        for section, (content, priority) in section_defaults.items():
            self._sections[section] = PromptSectionData(
                content=content,
                enabled=bool(content),  # Disabled if empty by default
                priority=priority,
            )

    # ── Section management ────────────────────────────────────────────────

    def set_section(
        self,
        section: PromptSection,
        content: str,
        enabled: bool = True,
        priority: Optional[int] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> None:
        """
        Set the content for a prompt section.

        Parameters
        ----------
        section:
            The section identifier.
        content:
            The text content for this section.
        enabled:
            Whether this section should be included in the final prompt.
        priority:
            Rendering priority (higher = later in output). If ``None``, keeps
            the existing priority.
        prefix:
            Text prepended before this section's content.
        suffix:
            Text appended after this section's content.
        """
        if section not in self._sections:
            self._sections[section] = PromptSectionData()
        data = self._sections[section]
        data.content = content
        data.enabled = enabled
        if priority is not None:
            data.priority = priority
        if prefix is not None:
            data.prefix = prefix
        if suffix is not None:
            data.suffix = suffix

    def enable_section(self, section: PromptSection) -> None:
        """Enable a specific section for inclusion in the prompt."""
        if section in self._sections:
            self._sections[section].enabled = True

    def disable_section(self, section: PromptSection) -> None:
        """Disable a specific section from inclusion in the prompt."""
        if section in self._sections:
            self._sections[section].enabled = False

    def toggle_section(self, section: PromptSection, enabled: Optional[bool] = None) -> None:
        """Toggle a section on/off. If ``enabled`` is None, flips the current state."""
        if section in self._sections:
            if enabled is None:
                self._sections[section].enabled = not self._sections[section].enabled
            else:
                self._sections[section].enabled = enabled

    def remove_section(self, section: PromptSection) -> None:
        """Remove a section entirely from the builder."""
        self._sections.pop(section, None)

    def is_enabled(self, section: PromptSection) -> bool:
        """Check if a section is currently enabled."""
        return self._sections.get(section, PromptSectionData()).enabled

    def get_section(self, section: PromptSection) -> Optional[PromptSectionData]:
        """Retrieve the data for a section, or ``None`` if it does not exist."""
        return self._sections.get(section)

    # ── Convenience setters ───────────────────────────────────────────────

    def set_custom_override(self, key: str, value: str) -> None:
        """Add a custom key-value override that gets appended to CUSTOM_OVERRIDES."""
        current = self._sections.get(PromptSection.CUSTOM_OVERRIDES)
        existing_content = current.content if current else ""
        new_content = f"{existing_content}\n- {key}: {value}".strip()
        self.set_section(PromptSection.CUSTOM_OVERRIDES, new_content, enabled=True)

    def set_context_from_dict(self, context_dict: Dict[str, Any]) -> None:
        """
        Populate the CONTEXT section from a dictionary of key-value pairs.

        Each entry is rendered as ``- key: value`` on its own line.
        """
        lines = []
        for key, value in context_dict.items():
            if isinstance(value, (list, tuple)):
                value_str = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                value_str = ", ".join(f"{k}={v}" for k, v in value.items())
            else:
                value_str = str(value)
            lines.append(f"- {key}: {value_str}")
        self.set_section(
            PromptSection.CONTEXT,
            "## CONTEXT\n" + "\n".join(lines),
            enabled=bool(lines),
        )

    def set_platform_hints(self) -> None:
        """Auto-detect and set platform-specific hints."""
        hints = self._detect_platform_hints()
        self.set_section(
            PromptSection.PLATFORM_HINTS,
            "## PLATFORM HINTS\n" + hints,
            enabled=True,
        )

    def set_memory_guidance(
        self,
        memory_available: bool = True,
        search_available: bool = True,
        max_context_tokens: int = 4000,
    ) -> None:
        """
        Set the memory guidance section.

        Parameters
        ----------
        memory_available:
            Whether persistent memory is active.
        search_available:
            Whether session search is available.
        max_context_tokens:
            Maximum tokens for injected memory context.
        """
        parts = ["## MEMORY GUIDANCE"]
        if memory_available:
            parts.append(
                "- You have access to persistent memory that stores information "
                "across sessions. Use it to remember user preferences, project "
                "details, and important decisions."
            )
        if search_available:
            parts.append(
                "- You can search past conversations and session history. "
                "Use this to find relevant context from earlier interactions."
            )
        parts.append(f"- Memory context is limited to ~{max_context_tokens} tokens per turn.")
        self.set_section(
            PromptSection.MEMORY_GUIDANCE,
            "\n".join(parts),
            enabled=memory_available or search_available,
        )

    def set_session_search_guidance(
        self,
        available: bool = True,
        max_results: int = 10,
    ) -> None:
        """Set session search instructions."""
        if not available:
            self.set_section(PromptSection.SESSION_SEARCH, "", enabled=False)
            return
        guidance = (
            "## SESSION SEARCH\n"
            "- You can search previous sessions to find relevant past conversations.\n"
            "- Use session search when the user references earlier work or asks about history.\n"
            f"- Search returns up to {max_results} relevant results ranked by similarity.\n"
            "- Use brief, keyword-focused queries for best search results."
        )
        self.set_section(PromptSection.SESSION_SEARCH, guidance, enabled=True)

    def set_tool_enforcement(self, tool_names: List[str]) -> None:
        """
        Set the tool enforcement section listing available tools.

        Parameters
        ----------
        tool_names:
            List of available tool function names.
        """
        if not tool_names:
            self.set_section(PromptSection.TOOL_ENFORCEMENT, "", enabled=False)
            return
        tools_str = ", ".join(f"`{name}`" for name in tool_names)
        section = (
            "## TOOLS\n"
            f"You have access to the following tools: {tools_str}\n"
            "- Always use the appropriate tool when available instead of guessing.\n"
            "- Chain tool calls when multiple steps are needed.\n"
            "- Review tool output before deciding the next action."
        )
        self.set_section(PromptSection.TOOL_ENFORCEMENT, section, enabled=True)

    def set_context_files_content(self, files_content: str) -> None:
        """
        Set the context files section with the raw content of referenced files.

        Parameters
        ----------
        files_content:
            Pre-formatted string containing file contents (as produced by
            ``Agent._build_context_files_content()`` or similar).
        """
        if not files_content.strip():
            self.set_section(PromptSection.CONTEXT_FILES, "", enabled=False)
            return
        section = f"## FILES IN CONTEXT\n{files_content}"
        self.set_section(PromptSection.CONTEXT_FILES, section, enabled=True)

    def set_knowledge_base_context(self, kb_context: str) -> None:
        """Inject knowledge base retrieval results into the prompt."""
        if not kb_context.strip():
            self.set_section(PromptSection.KNOWLEDGE_BASE, "", enabled=False)
            return
        section = f"## KNOWLEDGE BASE\n{kb_context}"
        self.set_section(PromptSection.KNOWLEDGE_BASE, section, enabled=True)

    def set_plugins_info(self, active_plugins: List[str]) -> None:
        """Set the plugins section listing active plugins."""
        if not active_plugins:
            self.set_section(PromptSection.PLUGINS, "", enabled=False)
            return
        plugins_str = ", ".join(active_plugins)
        section = (
            "## ACTIVE PLUGINS\n"
            f"The following plugins are active: {plugins_str}\n"
            "- Plugin-provided tools are available alongside built-in tools.\n"
            "- Plugin hooks may augment or filter tool calls and responses."
        )
        self.set_section(PromptSection.PLUGINS, section, enabled=True)

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self) -> str:
        """
        Assemble and return the final system prompt.

        Sections are rendered in priority order (lowest first). Disabled
        sections and sections with empty content (after applying prefix/suffix)
        are skipped.

        Returns
        -------
        str
            The complete system prompt string.
        """
        # Collect enabled sections with non-empty content
        active: List[tuple[int, PromptSectionData]] = []
        for section, data in self._sections.items():
            if not data.enabled:
                continue
            full_content = data.prefix + data.content + data.suffix
            if full_content.strip():
                active.append((data.priority, data))

        # Sort by priority (ascending = rendered first)
        active.sort(key=lambda x: x[0])

        # Join with the separator of each section
        parts: List[str] = []
        for _, data in active:
            full_content = data.prefix + data.content + data.suffix
            parts.append(full_content.strip())

        return "\n\n".join(parts)

    def estimate_tokens(self) -> int:
        """
        Estimate the token count of the current prompt.

        Uses a simple heuristic of ~4 characters per token, which is
        reasonable for English text with common sub-word tokenizers.

        Returns
        -------
        int
            Estimated token count.
        """
        prompt = self.build()
        return max(1, len(prompt) // 4)

    # ── Platform detection ────────────────────────────────────────────────

    def _detect_platform_hints(self) -> str:
        """Auto-detect platform information for the PLATFORM_HINTS section."""
        hints: List[str] = []
        try:
            cwd = self._cwd or os.getcwd()
            hints.append(f"- CWD: {cwd}")
        except Exception:
            hints.append("- CWD: unknown")

        hints.append(f"- OS: {platform.system()} {platform.release()} ({platform.machine()})")
        hints.append(f"- Python: {sys.version.split()[0]}")
        hints.append(f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Git status
        git_info = self._get_git_info()
        if git_info:
            hints.append(f"- Git: {git_info}")

        # Project type detection
        project_types = self._detect_project_types()
        if project_types:
            hints.append(f"- Project type: {', '.join(project_types)}")

        return "\n".join(hints)

    @staticmethod
    def _get_git_info() -> str:
        """Detect git branch and working tree status."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return ""
            branch = result.stdout.strip()
            result2 = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            )
            if result2.returncode == 0:
                dirty = len(result2.stdout.strip().split("\n"))
                if dirty > 0:
                    return f"branch={branch}, {dirty} dirty file(s)"
                return f"branch={branch}, clean"
            return f"branch={branch}"
        except Exception:
            return ""

    @staticmethod
    def _detect_project_types() -> List[str]:
        """Detect project type from files in the current directory."""
        cwd = Path.cwd()
        indicators = {
            "Python": ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"],
            "Node.js": ["package.json", "yarn.lock", "pnpm-lock.yaml"],
            "Rust": ["Cargo.toml"],
            "Go": ["go.mod"],
            "Java/Kotlin": ["pom.xml", "build.gradle", "build.gradle.kts"],
            "Ruby": ["Gemfile"],
            "PHP": ["composer.json"],
            "C/C++": ["CMakeLists.txt", "Makefile"],
        }
        detected: List[str] = []
        for project, files in indicators.items():
            for f in files:
                if (cwd / f).exists():
                    detected.append(project)
                    break
        return detected
