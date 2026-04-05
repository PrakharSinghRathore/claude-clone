"""
Desktop Orchestrator — the brain that ties together awareness, voice, controller,
and permissions for an AI desktop assistant.

This module provides:
- OrchestratorMode enum (SLEEP, PASSIVE, ACTIVE, VOICE, AUTONOMOUS)
- Intent / ProactiveSuggestion / TaskStep / TaskAutomation dataclasses
- DesktopOrchestrator class that manages all desktop interaction
- Rule-based intent parser with 30+ common intent patterns
- Context building from current PC state
- Proactive suggestions based on desktop activity
- Multi-step task automation
- Multi-modal input (text + voice)
- Screen understanding via screenshot + OCR
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Enums & Data Classes
# ──────────────────────────────────────────────────────────────

class OrchestratorMode(str, Enum):
    """Operating modes for the desktop orchestrator."""
    SLEEP = "sleep"
    PASSIVE = "passive"
    ACTIVE = "active"
    VOICE = "voice"
    AUTONOMOUS = "autonomous"


@dataclass
class Intent:
    """Parsed representation of a user command."""
    action: str
    params: dict = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "text"
    raw_text: str = ""


@dataclass
class ProactiveSuggestion:
    """A suggestion proactively offered to the user."""
    title: str
    description: str
    action: str
    priority: str = "medium"  # low, medium, high, critical
    auto_generated: bool = True


@dataclass
class TaskStep:
    """A single step within a multi-step automated task."""
    step_number: int
    description: str
    intent: Intent = field(default_factory=lambda: Intent(action="unknown"))
    status: str = "pending"  # pending, confirmed, executing, completed, failed, skipped
    result: str = ""


@dataclass
class TaskAutomation:
    """A multi-step task that the orchestrator can execute."""
    name: str
    description: str
    steps: List[TaskStep] = field(default_factory=list)
    status: str = "created"  # created, running, completed, failed, cancelled
    created_at: float = field(default_factory=time.time)


# ──────────────────────────────────────────────────────────────
# Intent rule definitions
# ──────────────────────────────────────────────────────────────

# Each entry: (compiled_regex, action_name, param_extractor, confidence, description)
IntentRule = tuple[re.Pattern, str, Callable[[re.Match], dict], float, str]

APP_ALIASES: Dict[str, str] = {
    "chrome": "chrome", "browser": "chrome", "firefox": "firefox",
    "edge": "msedge", "safari": "safari", "brave": "brave",
    "vscode": "code", "code": "code", "visual studio code": "code",
    "notepad": "notepad", "notepad++": "notepad++", "sublime": "sublime_text",
    "terminal": "terminal", "cmd": "cmd", "powershell": "powershell",
    "explorer": "explorer", "file explorer": "explorer",
    "slack": "slack", "discord": "discord", "teams": "teams",
    "zoom": "zoom", "spotify": "spotify", "vlc": "vlc",
    "word": "winword", "excel": "excel", "powerpoint": "powerpnt",
    "outlook": "outlook", "thunderbird": "thunderbird",
}

SYSTEM_COMMANDS: Dict[str, str] = {
    "volume up": "volume_up",
    "volume down": "volume_down",
    "mute": "mute",
    "unmute": "unmute",
    "brightness up": "brightness_up",
    "brightness down": "brightness_down",
    "lock screen": "lock_screen",
    "lock": "lock_screen",
    "sleep": "lock_screen",
    "hibernate": "hibernate",
    "shutdown": "shutdown",
    "restart": "restart",
}


def _build_intent_rules() -> List[IntentRule]:
    """
    Build the complete list of intent parsing rules.
    Each rule is a tuple: (regex_pattern, action, param_extractor, confidence, description).
    At least 30 rules covering common desktop commands.
    """
    rules: List[IntentRule] = []

    # ── 1. open [app] → launch_app ──
    rules.append((
        re.compile(r"^(?:open|launch|start|run)\s+(?:the\s+)?(?:app\s+)?(.+?)$", re.IGNORECASE),
        "launch_app",
        lambda m: {"app": _resolve_app_alias(m.group(1).strip())},
        0.90,
        "Open or launch an application",
    ))

    # ── 2. close [app/window] → close_window ──
    rules.append((
        re.compile(r"^(?:close|quit|exit|kill\s+app|stop)\s+(?:the\s+)?(?:app\s+)?(.+?)$", re.IGNORECASE),
        "close_window",
        lambda m: {"app": _resolve_app_alias(m.group(1).strip())},
        0.88,
        "Close an application or window",
    ))

    # ── 3. type [text] → type_text ──
    rules.append((
        re.compile(r"^type\s+(?:the\s+)?(?:text\s+)?[\"']?(.+?)[\"']?\s*$", re.IGNORECASE),
        "type_text",
        lambda m: {"text": m.group(1).strip()},
        0.85,
        "Type text into the active window",
    ))

    # ── 4. click [position/element] → mouse_click ──
    rules.append((
        re.compile(r"^click\s+(?:at\s+)?(?:the\s+)?(.+?)$", re.IGNORECASE),
        "mouse_click",
        lambda m: {"target": m.group(1).strip()},
        0.82,
        "Click at a position or on an element",
    ))

    # ── 5. right click → right_click ──
    rules.append((
        re.compile(r"^(?:right[\s-]?click|rclick)\s+(?:at\s+)?(?:the\s+)?(.+?)$", re.IGNORECASE),
        "right_click",
        lambda m: {"target": m.group(1).strip()},
        0.82,
        "Right click at a position or on an element",
    ))

    # ── 6. screenshot / take screenshot → take_screenshot ──
    rules.append((
        re.compile(r"^(?:take\s+)?(?:a\s+)?screenshot(?:\s+(?:of\s+)?(.+))?$", re.IGNORECASE),
        "take_screenshot",
        lambda m: {"region": m.group(1).strip() if m.group(1) else "full"},
        0.95,
        "Take a screenshot",
    ))

    # ── 7. what's my cpu/ram/disk → query_system ──
    rules.append((
        re.compile(
            r"^(?:what'?s?\s+)?(?:what\s+is\s+)?(?:my\s+)?"
            r"(cpu|ram|memory|disk|network|system)\s*(?:usage|use|load|utilization|status|info)?\s*\??$",
            re.IGNORECASE,
        ),
        "query_system",
        lambda m: {"target": m.group(1).strip().lower()},
        0.92,
        "Query system resource usage",
    ))

    # ── 8. kill [process] → kill_process ──
    rules.append((
        re.compile(r"^kill\s+(?:process\s+)?(.+?)$", re.IGNORECASE),
        "kill_process",
        lambda m: {"process": m.group(1).strip()},
        0.88,
        "Kill a running process",
    ))

    # ── 9. maximize/minimize [window] → window_action ──
    rules.append((
        re.compile(r"^(maximize|max)\s+(?:the\s+)?(?:window\s+)?(.+?)$", re.IGNORECASE),
        "window_action",
        lambda m: {"action": "maximize", "window": m.group(1).strip()},
        0.87,
        "Maximize a window",
    ))
    rules.append((
        re.compile(r"^(minimize|min)\s+(?:the\s+)?(?:window\s+)?(.+?)$", re.IGNORECASE),
        "window_action",
        lambda m: {"action": "minimize", "window": m.group(1).strip()},
        0.87,
        "Minimize a window",
    ))

    # ── 10. switch to [window/app] → focus_window ──
    rules.append((
        re.compile(r"^(?:switch\s+(?:to|window)|focus|bring\s+(?:up|to\s+front))\s+(?:the\s+)?(?:window\s+)?(.+?)$", re.IGNORECASE),
        "focus_window",
        lambda m: {"window": m.group(1).strip()},
        0.88,
        "Switch to or focus a window",
    ))

    # ── 11. copy/paste → clipboard_action ──
    rules.append((
        re.compile(r"^(?:copy|copy\s+all|copy\s+to\s+clipboard)\s*(?:\s+(.+))?$", re.IGNORECASE),
        "clipboard_action",
        lambda m: {"action": "copy", "text": m.group(1).strip() if m.group(1) else ""},
        0.85,
        "Copy text to clipboard",
    ))
    rules.append((
        re.compile(r"^(?:paste|paste\s+from\s+clipboard)\s*$", re.IGNORECASE),
        "clipboard_action",
        lambda m: {"action": "paste"},
        0.85,
        "Paste from clipboard",
    ))

    # ── 12. open url / go to [url] → open_url ──
    rules.append((
        re.compile(r"^(?:open\s+url|go\s+to|navigate\s+to|visit|browse)\s+(.+?)$", re.IGNORECASE),
        "open_url",
        lambda m: {"url": _ensure_url(m.group(1).strip())},
        0.90,
        "Open a URL in the default browser",
    ))

    # ── 13. search [query] → web_search ──
    rules.append((
        re.compile(r"^(?:search|google|look\s+up|find)\s+(?:for\s+)?(?:the\s+)?(.+?)$", re.IGNORECASE),
        "web_search",
        lambda m: {"query": m.group(1).strip()},
        0.88,
        "Perform a web search",
    ))

    # ── 14. volume up/down/mute, brightness, lock → system_action ──
    rules.append((
        re.compile(
            r"^(" + "|".join(re.escape(k) for k in SYSTEM_COMMANDS) + r")\s*$",
            re.IGNORECASE,
        ),
        "system_action",
        lambda m: {"action": SYSTEM_COMMANDS[m.group(1).strip().lower()]},
        0.92,
        "Perform a system action (volume, brightness, lock, etc.)",
    ))

    # ── 15. what am I looking at / describe screen → screen_describe ──
    rules.append((
        re.compile(
            r"^(?:what\s+am\s+i\s+looking\s+at|describe\s+(?:the\s+)?(?:current\s+)?screen|"
            r"what'?s?\s+on\s+(?:my\s+)?screen|read\s+(?:the\s+)?screen)\s*\??$",
            re.IGNORECASE,
        ),
        "screen_describe",
        lambda m: {},
        0.95,
        "Describe what is currently on screen",
    ))

    # ── 16. what's on my clipboard → query_clipboard ──
    rules.append((
        re.compile(
            r"^(?:what'?s?\s+on\s+my\s+clipboard|show\s+(?:my\s+)?clipboard|"
            r"clipboard\s+(?:content|contents|text))\s*\??$",
            re.IGNORECASE,
        ),
        "query_clipboard",
        lambda m: {},
        0.94,
        "Show clipboard contents",
    ))

    # ── 17. list windows / show windows → list_windows ──
    rules.append((
        re.compile(r"^(?:list|show|display|get)\s+(?:all\s+)?(?:open\s+)?(?:the\s+)?windows?\s*$", re.IGNORECASE),
        "list_windows",
        lambda m: {},
        0.93,
        "List all open windows",
    ))

    # ── 18. open terminal / open cmd → open_terminal ──
    rules.append((
        re.compile(r"^(?:open|launch|start)\s+(?:the\s+)?(?:terminal|cmd|command\s+prompt|powershell|shell|console)\s*$", re.IGNORECASE),
        "open_terminal",
        lambda m: {"terminal_type": "default"},
        0.94,
        "Open a terminal",
    ))

    # ── 19. start recording / stop recording → toggle_recording ──
    rules.append((
        re.compile(r"^(?:start|begin)\s+(?:a\s+)?(?:screen\s+)?recording\s*$", re.IGNORECASE),
        "toggle_recording",
        lambda m: {"action": "start"},
        0.90,
        "Start screen recording",
    ))
    rules.append((
        re.compile(r"^(?:stop|end)\s+(?:the\s+)?(?:screen\s+)?recording\s*$", re.IGNORECASE),
        "toggle_recording",
        lambda m: {"action": "stop"},
        0.90,
        "Stop screen recording",
    ))

    # ── 20. show notification → show_notification ──
    rules.append((
        re.compile(r"^(?:show|send|display|create)\s+(?:a\s+)?notification(?:\s+(.+))?$", re.IGNORECASE),
        "show_notification",
        lambda m: {"message": m.group(1).strip() if m.group(1) else "Notification from Desktop Assistant"},
        0.85,
        "Show a desktop notification",
    ))

    # ── 21. install [package] → install_package ──
    rules.append((
        re.compile(r"^(?:install|pip\s+install|npm\s+install|apt\s+install)\s+(.+?)$", re.IGNORECASE),
        "install_package",
        lambda m: {"package": m.group(1).strip()},
        0.90,
        "Install a software package",
    ))

    # ── 22. run [command] → run_command ──
    rules.append((
        re.compile(r"^(?:run|execute)\s+(?:the\s+)?(?:command\s+)?[\"']?(.+?)[\"']?\s*$", re.IGNORECASE),
        "run_command",
        lambda m: {"command": m.group(1).strip()},
        0.88,
        "Run a shell command",
    ))

    # ── 23. read file [path] → read_file ──
    rules.append((
        re.compile(r"^(?:read|show|display|cat|view|open)\s+(?:the\s+)?(?:file\s+)?(.+?)$", re.IGNORECASE),
        "read_file",
        lambda m: {"path": m.group(1).strip()},
        0.80,
        "Read a file's contents",
    ))

    # ── 24. what time is it → query_time ──
    rules.append((
        re.compile(r"^(?:what\s+(?:time|'?s?\s+the\s+time)|current\s+time|time\s+(?:is\s+it|now)|tell\s+(?:me\s+)?the\s+time)\s*\??$", re.IGNORECASE),
        "query_time",
        lambda m: {},
        0.98,
        "Query the current time",
    ))

    # ── 25. what day is it → query_date ──
    rules.append((
        re.compile(r"^(?:what\s+(?:day|date|'?s?\s+the\s+date)|today'?s?\s+date|what\s+day\s+is\s+it)\s*\??$", re.IGNORECASE),
        "query_date",
        lambda m: {},
        0.98,
        "Query the current date",
    ))

    # ── 26. how long has pc been on → query_uptime ──
    rules.append((
        re.compile(
            r"^(?:how\s+long\s+(?:has\s+)?(?:my\s+)?(?:pc|computer|system|machine)\s+been\s+(?:on|up|running)|"
            r"system\s+uptime|pc\s+uptime|computer\s+uptime)\s*\??$",
            re.IGNORECASE,
        ),
        "query_uptime",
        lambda m: {},
        0.95,
        "Query system uptime",
    ))

    # ── 27. clean up / free space → system_cleanup ──
    rules.append((
        re.compile(r"^(?:clean\s+up|cleanup|free\s+(?:up\s+)?(?:some\s+)?(?:disk\s+)?space|clear\s+(?:temp|cache))\s*$", re.IGNORECASE),
        "system_cleanup",
        lambda m: {},
        0.88,
        "Clean up system and free disk space",
    ))

    # ── 28. check internet / check connection → check_network ──
    rules.append((
        re.compile(
            r"^(?:check\s+(?:my\s+)?(?:internet|network|connection|wifi|wi[-]?fi)|"
            r"am\s+i\s+(?:connected|online)|is\s+(?:the\s+)?(?:internet|network)\s+(?:up|working|down))\s*\??$",
            re.IGNORECASE,
        ),
        "check_network",
        lambda m: {},
        0.94,
        "Check network connectivity",
    ))

    # ── 29. scroll [up/down] → scroll_action ──
    rules.append((
        re.compile(r"^scroll\s+(up|down|left|right)(?:\s+(\d+))?\s*$", re.IGNORECASE),
        "scroll_action",
        lambda m: {"direction": m.group(1).lower(), "amount": int(m.group(2)) if m.group(2) else 3},
        0.86,
        "Scroll in a direction",
    ))

    # ── 30. press [key] → press_key ──
    rules.append((
        re.compile(r"^(?:press|hit|type)\s+(?:the\s+)?(?:key\s+)?(.+?)$", re.IGNORECASE),
        "press_key",
        lambda m: {"key": m.group(1).strip().lower()},
        0.84,
        "Press a keyboard key",
    ))

    # ── 31. drag [from] to [to] → drag_action ──
    rules.append((
        re.compile(r"^drag\s+(?:from\s+)?(.+?)\s+to\s+(.+?)$", re.IGNORECASE),
        "drag_action",
        lambda m: {"from": m.group(1).strip(), "to": m.group(2).strip()},
        0.80,
        "Drag from one position to another",
    ))

    # ── 32. resize [window] → resize_window ──
    rules.append((
        re.compile(r"^resize\s+(?:the\s+)?(?:window\s+)?(.+?)\s+to\s+(.+?)$", re.IGNORECASE),
        "resize_window",
        lambda m: {"window": m.group(1).strip(), "size": m.group(2).strip()},
        0.78,
        "Resize a window",
    ))

    # ── 33. move [window] to [position] → move_window ──
    rules.append((
        re.compile(r"^move\s+(?:the\s+)?(?:window\s+)?(.+?)\s+to\s+(.+?)$", re.IGNORECASE),
        "move_window",
        lambda m: {"window": m.group(1).strip(), "position": m.group(2).strip()},
        0.78,
        "Move a window to a position",
    ))

    return rules


def _resolve_app_alias(name: str) -> str:
    """Resolve common app name aliases to canonical names."""
    key = name.strip().lower()
    return APP_ALIASES.get(key, key)


def _ensure_url(text: str) -> str:
    """Ensure the text is a valid URL, prepending https:// if needed."""
    text = text.strip()
    if not text:
        return text
    if not re.match(r"^https?://", text, re.IGNORECASE):
        # Heuristic: if it looks like a domain (contains a dot with no spaces), prepend https
        if "." in text and " " not in text:
            text = "https://" + text
        else:
            # Treat as a search query
            text = f"https://www.google.com/search?q={text}"
    return text


# ──────────────────────────────────────────────────────────────
# Proactive suggestion generators
# ──────────────────────────────────────────────────────────────

async def _suggest_disk_cleanup(system_info: dict) -> Optional[ProactiveSuggestion]:
    """Suggest cleanup if disk usage is high."""
    disk = system_info.get("disk", {})
    for mount, info in disk.items() if isinstance(disk, dict) else []:
        usage = info.get("percent", 0)
        if isinstance(usage, str):
            usage = float(usage.rstrip("%"))
        if usage >= 85:
            return ProactiveSuggestion(
                title="Disk space running low",
                description=f"Drive {mount} is {usage}% full. Want me to clean up temporary files, caches, and downloads?",
                action="system_cleanup",
                priority="high" if usage >= 93 else "medium",
            )
    return None


async def _suggest_close_tabs(window_info: dict) -> Optional[ProactiveSuggestion]:
    """Suggest closing tabs if too many browser windows are open."""
    windows = window_info.get("windows", [])
    browser_count = sum(1 for w in windows if any(
        b in w.get("title", "").lower() for b in ("chrome", "firefox", "edge", "brave", "safari")
    ))
    if browser_count >= 8:
        return ProactiveSuggestion(
            title="Many browser tabs open",
            description=f"You have {browser_count} browser windows/tabs open. Want me to identify and close unused ones?",
            action="list_windows",
            priority="low",
        )
    return None


async def _suggest_break(activity_info: dict) -> Optional[ProactiveSuggestion]:
    """Suggest a break if the user has been active for too long."""
    session_duration = activity_info.get("session_minutes", 0)
    if session_duration >= 120:
        return ProactiveSuggestion(
            title="Time for a break?",
            description=f"You've been active for {int(session_duration // 60)}h {int(session_duration % 60)}m. "
                        "Consider taking a short break to rest your eyes.",
            action="system_action",
            priority="low",
        )
    return None


async def _suggest_memory_pressure(system_info: dict) -> Optional[ProactiveSuggestion]:
    """Suggest closing apps if RAM is under pressure."""
    ram = system_info.get("ram", {})
    usage = ram.get("percent", 0)
    if isinstance(usage, str):
        usage = float(usage.rstrip("%"))
    if usage >= 90:
        return ProactiveSuggestion(
            title="High memory usage",
            description=f"RAM usage is at {usage}%. Want me to list the most memory-hungry processes so you can close some?",
            action="query_system",
            priority="high",
        )
    return None


async def _suggest_network_issue(system_info: dict) -> Optional[ProactiveSuggestion]:
    """Suggest checking network if latency is high."""
    network = system_info.get("network", {})
    latency = network.get("latency_ms", 0)
    if latency > 500:
        return ProactiveSuggestion(
            title="Network seems slow",
            description=f"Network latency is {latency}ms. Would you like me to run a network diagnostic?",
            action="check_network",
            priority="medium",
        )
    return None


# ──────────────────────────────────────────────────────────────
# Task decomposition patterns
# ──────────────────────────────────────────────────────────────

TASK_PATTERNS: List[tuple[re.Pattern, Callable[[re.Match], List[dict]]]] = [
    (
        re.compile(
            r"(?:set\s+up|create|make)\s+(?:a\s+)?(?:new\s+)?(?:python|node|react|web)\s+(?:project|app)",
            re.IGNORECASE,
        ),
        lambda m: [
            {"description": "Create project directory", "action": "run_command",
             "params": {"command": "mkdir -p ~/new-project && cd ~/new-project"}},
            {"description": "Initialize project", "action": "run_command",
             "params": {"command": "python -m venv venv && source venv/bin/activate && pip install -e ."}},
            {"description": "Create basic project structure", "action": "run_command",
             "params": {"command": "mkdir -p src tests docs && touch src/__init__.py tests/__init__.py"}},
            {"description": "Create README", "action": "run_command",
             "params": {"command": 'echo "# New Project" > README.md'}},
        ],
    ),
    (
        re.compile(
            r"(?:clean\s+up|tidy)\s+(?:my\s+)?(?:desktop|downloads|documents)",
            re.IGNORECASE,
        ),
        lambda m: [
            {"description": "List files on desktop", "action": "run_command",
             "params": {"command": "ls -la ~/Desktop/"}},
            {"description": "Move old files to archive", "action": "run_command",
             "params": {"command": "mkdir -p ~/Desktop/Archive && find ~/Desktop -maxdepth 1 -mtime +30 -exec mv {} ~/Desktop/Archive/ \\;"}},
            {"description": "Clean up temporary files", "action": "run_command",
             "params": {"command": "rm -rf /tmp/* 2>/dev/null; echo 'Temp files cleaned'"}},
        ],
    ),
    (
        re.compile(
            r"(?:backup|back\s+up)\s+(?:my\s+)?(?:files|documents|project|data)",
            re.IGNORECASE,
        ),
        lambda m: [
            {"description": "Create backup directory", "action": "run_command",
             "params": {"command": "mkdir -p ~/Backups/$(date +%Y-%m-%d)"}},
            {"description": "Copy important files", "action": "run_command",
             "params": {"command": "cp -r ~/Documents ~/Backups/$(date +%Y-%m-%d)/Documents 2>/dev/null; echo 'Done'"}},
            {"description": "Verify backup integrity", "action": "run_command",
             "params": {"command": "du -sh ~/Backups/$(date +%Y-%m-%d)"}},
        ],
    ),
]


# ──────────────────────────────────────────────────────────────
# Desktop Orchestrator
# ──────────────────────────────────────────────────────────────

class DesktopOrchestrator:
    """
    The central orchestrator for AI desktop interaction.

    Manages mode switching, intent parsing, action execution, proactive
    suggestions, task automation, and multi-modal input handling.

    Usage::

        orchestrator = DesktopOrchestrator(agent, awareness, voice, controller, permissions)
        await orchestrator.initialize()
        response = await orchestrator.process_input("Open Chrome")
        await orchestrator.shutdown()
    """

    SYSTEM_PROMPT = (
        "You are a helpful AI desktop assistant. You can see the user's screen, "
        "control their computer, and proactively offer help. Be concise and actionable. "
        "When you take actions, explain what you're doing. If unsure, ask for confirmation."
    )

    def __init__(
        self,
        agent: Any = None,
        awareness: Any = None,
        voice: Any = None,
        controller: Any = None,
        permissions: Any = None,
    ) -> None:
        self._agent = agent
        self._awareness = awareness
        self._voice = voice
        self._controller = controller
        self._permissions = permissions

        self._mode: OrchestratorMode = OrchestratorMode.ACTIVE
        self._initialized: bool = False
        self._shutting_down: bool = False
        self._session_start: float = time.time()

        # Stats tracking
        self._stats: Dict[str, int] = {
            "inputs_processed": 0,
            "intents_parsed": 0,
            "actions_executed": 0,
            "suggestions_given": 0,
            "tasks_completed": 0,
            "errors": 0,
        }

        # Build intent rules
        self._intent_rules: List[IntentRule] = _build_intent_rules()

        # Active tasks
        self._active_tasks: Dict[str, TaskAutomation] = {}

        # Activity log for summaries
        self._activity_log: List[Dict[str, Any]] = []
        self._max_activity_entries: int = 500

        # Pending input queue for multi-modal handling
        self._input_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Suggestion generators
        self._suggestion_generators: List[Callable] = [
            _suggest_disk_cleanup,
            _suggest_close_tabs,
            _suggest_break,
            _suggest_memory_pressure,
            _suggest_network_issue,
        ]

        # Periodic task handle
        self._proactive_task: Optional[asyncio.Task] = None
        self._voice_event_task: Optional[asyncio.Task] = None
        self._input_queue_task: Optional[asyncio.Task] = None

        # Cached system state (refreshed periodically)
        self._cached_system_info: Dict[str, Any] = {}
        self._cached_window_info: Dict[str, Any] = {}
        self._cached_clipboard: str = ""
        self._last_context_refresh: float = 0.0
        self._context_refresh_interval: float = 30.0  # seconds

        # Last proactive suggestion timestamp to avoid spamming
        self._last_suggestion_time: float = 0.0
        self._suggestion_cooldown: float = 300.0  # 5 minutes

        # OCR engine reference (lazy loaded)
        self._ocr_available: Optional[bool] = None

    # ──────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Start all components and background tasks.
        Sets up awareness monitoring, voice listener, and the input queue processor.
        """
        if self._initialized:
            logger.warning("DesktopOrchestrator is already initialized")
            return

        logger.info("Initializing DesktopOrchestrator...")

        # Initialize awareness module
        if self._awareness and hasattr(self._awareness, "start"):
            try:
                await self._awareness.start()
                logger.info("Desktop awareness started")
            except Exception as e:
                logger.error(f"Failed to start awareness: {e}")

        # Initialize voice engine
        if self._voice and hasattr(self._voice, "start"):
            try:
                await self._voice.start()
                logger.info("Voice engine started")
            except Exception as e:
                logger.error(f"Failed to start voice engine: {e}")

        # Initialize controller
        if self._controller and hasattr(self._controller, "connect"):
            try:
                await self._controller.connect()
                logger.info("Desktop controller connected")
            except Exception as e:
                logger.error(f"Failed to connect controller: {e}")

        # Check OCR availability
        self._ocr_available = self._check_ocr_available()

        # Refresh initial context
        await self._refresh_cached_state()

        # Start background tasks
        loop = asyncio.get_event_loop()
        self._proactive_task = loop.create_task(self._proactive_suggestion_loop(), name="proactive_suggestions")
        self._voice_event_task = loop.create_task(self._voice_event_loop(), name="voice_events")
        self._input_queue_task = loop.create_task(self._input_queue_loop(), name="input_queue")

        self._initialized = True
        logger.info("DesktopOrchestrator initialized in %s mode", self._mode.value)

    async def shutdown(self) -> None:
        """
        Graceful shutdown: stop all background tasks and components.
        """
        if not self._initialized:
            return

        self._shutting_down = True
        logger.info("Shutting down DesktopOrchestrator...")

        # Cancel background tasks
        for task in (self._proactive_task, self._voice_event_task, self._input_queue_task):
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        # Stop awareness
        if self._awareness and hasattr(self._awareness, "stop"):
            try:
                await self._awareness.stop()
            except Exception as e:
                logger.error(f"Error stopping awareness: {e}")

        # Stop voice
        if self._voice and hasattr(self._voice, "stop"):
            try:
                await self._voice.stop()
            except Exception as e:
                logger.error(f"Error stopping voice: {e}")

        # Disconnect controller
        if self._controller and hasattr(self._controller, "disconnect"):
            try:
                await self._controller.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting controller: {e}")

        self._initialized = False
        logger.info("DesktopOrchestrator shut down complete")

    # ──────────────────────────────────────────────
    # Mode management
    # ──────────────────────────────────────────────

    async def set_mode(self, mode: OrchestratorMode) -> None:
        """
        Switch the orchestrator to a new operating mode.

        Mode behavior:
        - SLEEP: Minimal monitoring, no responses, low resource usage.
        - PASSIVE: Monitors desktop events, can answer queries but won't act.
        - ACTIVE: Monitors and can execute actions with user permission.
        - VOICE: Continuous voice conversation, hands-free interaction.
        - AUTONOMOUS: Can act without permission for trusted actions.
        """
        old_mode = self._mode
        self._mode = mode

        self._log_activity("mode_change", {
            "from": old_mode.value,
            "to": mode.value,
        })

        logger.info("Mode changed: %s → %s", old_mode.value, mode.value)

        # Mode-specific side effects
        if mode == OrchestratorMode.SLEEP:
            if self._voice and hasattr(self._voice, "pause"):
                await self._voice.pause()
        elif mode == OrchestratorMode.VOICE:
            if self._voice and hasattr(self._voice, "resume"):
                await self._voice.resume()

    async def get_mode(self) -> OrchestratorMode:
        """Return the current orchestrator mode."""
        return self._mode

    def _is_action_allowed(self, action: str) -> bool:
        """
        Check whether an action is allowed in the current mode.

        In SLEEP mode, nothing is allowed.
        In PASSIVE mode, only queries are allowed.
        In ACTIVE mode, actions require permission.
        In AUTONOMOUS mode, trusted actions proceed without permission.
        """
        if self._mode == OrchestratorMode.SLEEP:
            return False
        if self._mode == OrchestratorMode.PASSIVE:
            return action.startswith("query_") or action in ("list_windows", "screen_describe")
        if self._mode == OrchestratorMode.AUTONOMOUS:
            trusted_actions = {
                "query_system", "query_time", "query_date", "query_uptime",
                "query_clipboard", "list_windows", "screen_describe",
                "check_network", "take_screenshot",
            }
            return action in trusted_actions
        # ACTIVE mode — all actions allowed, but execute_intent will check permissions
        return True

    # ──────────────────────────────────────────────
    # Main input processing
    # ──────────────────────────────────────────────

    async def process_input(self, text: str, source: str = "text") -> str:
        """
        Main entry point for user input. Accepts text from CLI, GUI, or voice.

        Flow:
        1. Check if the mode allows processing.
        2. Parse the text into an Intent.
        3. If intent is recognized with high confidence, execute it.
        4. Otherwise, fall back to the AI agent with full context.
        5. Return the response string.
        """
        if not text or not text.strip():
            return "Please provide a command or question."

        self._stats["inputs_processed"] += 1
        raw = text.strip()
        self._log_activity("user_input", {"text": raw, "source": source})

        # In SLEEP mode, wake up on any input
        if self._mode == OrchestratorMode.SLEEP:
            await self.set_mode(OrchestratorMode.ACTIVE)
            return f"Woke up from sleep mode. You said: \"{raw}\". How can I help?"

        # Parse intent
        intent = await self.parse_intent(raw)
        intent.source = source

        if intent.confidence >= 0.75 and intent.action != "unknown":
            self._stats["intents_parsed"] += 1

            # Check if action is allowed in current mode
            if not self._is_action_allowed(intent.action):
                return (
                    f"I understand you want to {intent.action}, but that action is not available "
                    f"in {self._mode.value} mode. Switch to ACTIVE or AUTONOMOUS mode to enable it."
                )

            # Execute the intent
            result = await self.execute_intent(intent)
            self._log_activity("intent_executed", {
                "action": intent.action,
                "confidence": intent.confidence,
            })

            # In VOICE mode, also speak the response
            if self._mode == OrchestratorMode.VOICE and self._voice:
                await self._speak_response(result)

            return result

        # Low confidence — delegate to AI agent with full context
        context = await self.get_context()
        full_prompt = f"{context}\n\nUser: {raw}\n\nRespond concisely and helpfully."

        response = await self._query_agent(full_prompt)
        self._log_activity("agent_response", {"query": raw, "response_length": len(response)})

        if self._mode == OrchestratorMode.VOICE and self._voice:
            await self._speak_response(response)

        return response

    # ──────────────────────────────────────────────
    # Context building
    # ──────────────────────────────────────────────

    async def get_context(self) -> str:
        """
        Build a comprehensive context string from the current PC state.

        Gathers: active window, clipboard, system resources, running processes,
        recent files, screen description, time, and activity summary.
        """
        await self._refresh_cached_state()

        parts: List[str] = []

        # Timestamp
        now = datetime.now()
        parts.append(f"## Current Time: {now.strftime('%Y-%m-%d %H:%M:%S %A')}")

        # Operating system
        parts.append(f"## System: {platform.system()} {platform.release()} ({platform.machine()})")
        parts.append(f"## Hostname: {platform.node()}")
        parts.append(f"## Python: {platform.python_version()}")

        # Active window
        active_window = self._cached_window_info.get("active_window", "Unknown")
        if active_window and active_window != "Unknown":
            parts.append(f"## Active Window: {active_window}")

        # Open windows
        windows = self._cached_window_info.get("windows", [])
        if windows:
            win_names = [w.get("title", "Untitled") for w in windows[:20]]
            parts.append(f"## Open Windows ({len(windows)}): {', '.join(win_names)}")

        # Clipboard
        clipboard = self._cached_clipboard
        if clipboard:
            clipped = clipboard[:200] + ("..." if len(clipboard) > 200 else "")
            parts.append(f"## Clipboard: {clipped}")

        # System resources
        sys_info = self._cached_system_info
        if sys_info:
            cpu = sys_info.get("cpu", {})
            if cpu:
                parts.append(f"## CPU Usage: {cpu.get('percent', 'N/A')}%")
            ram = sys_info.get("ram", {})
            if ram:
                parts.append(f"## RAM Usage: {ram.get('percent', 'N/A')}% ({ram.get('used', '?')}/{ram.get('total', '?')} GB)")
            disk = sys_info.get("disk", {})
            if disk:
                for mount, info in (disk.items() if isinstance(disk, dict) else []):
                    parts.append(f"## Disk {mount}: {info.get('percent', 'N/A')}% used ({info.get('free', '?')} GB free)")

        # Running processes (top 10 by memory)
        processes = self._get_top_processes(10)
        if processes:
            proc_lines = [f"  - {p['name']} (PID {p['pid']}, {p['memory_percent']:.1f}% RAM)" for p in processes]
            parts.append(f"## Top Processes:\n" + "\n".join(proc_lines))

        # Recent files
        recent = self._get_recent_files(5)
        if recent:
            parts.append(f"## Recent Files: {', '.join(recent)}")

        # Orchestrator mode
        parts.append(f"## Orchestrator Mode: {self._mode.value}")

        # Session duration
        elapsed = time.time() - self._session_start
        mins = int(elapsed // 60)
        parts.append(f"## Session Duration: {mins} minutes")

        # Recent activity summary
        if self._activity_log:
            recent_activity = self._activity_log[-5:]
            act_lines = [f"  - [{a['type']}] {a.get('data', {})}" for a in recent_activity]
            parts.append(f"## Recent Activity:\n" + "\n".join(act_lines))

        return "\n".join(parts)

    async def _refresh_cached_state(self) -> None:
        """Refresh cached system, window, and clipboard state."""
        now = time.time()
        if now - self._last_context_refresh < self._context_refresh_interval:
            return
        self._last_context_refresh = now

        # System info
        self._cached_system_info = await self._gather_system_info()

        # Window info
        if self._awareness and hasattr(self._awareness, "get_window_info"):
            try:
                self._cached_window_info = await self._awareness.get_window_info()
            except Exception as e:
                logger.debug("Failed to get window info: %s", e)
        else:
            self._cached_window_info = self._get_window_info_fallback()

        # Clipboard
        if self._awareness and hasattr(self._awareness, "get_clipboard"):
            try:
                self._cached_clipboard = await self._awareness.get_clipboard() or ""
            except Exception:
                self._cached_clipboard = self._get_clipboard_fallback()

    async def _gather_system_info(self) -> Dict[str, Any]:
        """Gather CPU, RAM, disk, and network information."""
        info: Dict[str, Any] = {}

        # CPU
        try:
            import psutil
            info["cpu"] = {
                "percent": psutil.cpu_percent(interval=0.1),
                "count": psutil.cpu_count(),
            }
        except ImportError:
            info["cpu"] = {"percent": self._cpu_percent_fallback(), "count": os.cpu_count()}

        # RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["ram"] = {
                "total": round(mem.total / (1024 ** 3), 1),
                "used": round(mem.used / (1024 ** 3), 1),
                "percent": mem.percent,
            }
        except ImportError:
            info["ram"] = {"percent": "N/A"}

        # Disk
        try:
            import psutil
            disk = {}
            for part in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    disk[part.mountpoint] = {
                        "total": round(usage.total / (1024 ** 3), 1),
                        "used": round(usage.used / (1024 ** 3), 1),
                        "free": round(usage.free / (1024 ** 3), 1),
                        "percent": usage.percent,
                    }
                except PermissionError:
                    continue
            info["disk"] = disk
        except ImportError:
            info["disk"] = self._disk_usage_fallback()

        # Network
        try:
            latency = await self._ping_test("8.8.8.8", count=1, timeout=2)
            info["network"] = {"latency_ms": latency}
        except Exception:
            info["network"] = {"latency_ms": -1}

        return info

    def _cpu_percent_fallback(self) -> float:
        """Get CPU usage without psutil (Linux/macOS fallback)."""
        try:
            result = subprocess.run(
                ["top", "-bn1"], capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.split("\n"):
                if "Cpu" in line:
                    match = re.search(r"(\d+\.?\d*)\s*(?:%|id)", line)
                    if match:
                        return 100.0 - float(match.group(1))
            return -1.0
        except Exception:
            return -1.0

    def _disk_usage_fallback(self) -> Dict[str, Any]:
        """Get disk usage without psutil."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "logicaldisk", "get", "size,freespace,caption"],
                    capture_output=True, text=True, timeout=10,
                )
                disk = {}
                for line in result.stdout.strip().split("\n")[1:]:
                    parts = line.split()
                    if len(parts) >= 3:
                        total = int(parts[1]) / (1024 ** 3)
                        free = int(parts[2]) / (1024 ** 3)
                        disk[parts[0]] = {
                            "total": round(total, 1),
                            "free": round(free, 1),
                            "percent": round((1 - free / total) * 100, 1) if total > 0 else 0,
                        }
                return disk
            else:
                result = subprocess.run(
                    ["df", "-h"], capture_output=True, text=True, timeout=5,
                )
                disk = {}
                for line in result.stdout.strip().split("\n")[1:]:
                    parts = line.split()
                    if len(parts) >= 5:
                        disk[parts[5] if len(parts) > 5 else parts[0]] = {
                            "total": parts[1],
                            "used": parts[2],
                            "free": parts[3],
                            "percent": parts[4].rstrip("%"),
                        }
                return disk
        except Exception:
            return {}

    def _get_window_info_fallback(self) -> Dict[str, Any]:
        """Get window info without the awareness module."""
        try:
            active_window = "Unknown"
            if platform.system() == "Linux":
                try:
                    result = subprocess.run(
                        ["xdotool", "getwindowfocus", "getwindowname"],
                        capture_output=True, text=True, timeout=3,
                    )
                    if result.returncode == 0:
                        active_window = result.stdout.strip()
                except FileNotFoundError:
                    pass
            elif platform.system() == "Darwin":
                try:
                    result = subprocess.run(
                        ["osascript", "-e", 'tell application "System Events" to get name of first process whose frontmost is true'],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0:
                        active_window = result.stdout.strip()
                except Exception:
                    pass
            elif platform.system() == "Windows":
                try:
                    import ctypes
                    hwnd = ctypes.windll.user32.GetForegroundWindow()
                    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                    buf = ctypes.create_unicode_buffer(length + 1)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                    active_window = buf.value or "Unknown"
                except Exception:
                    pass
            return {"active_window": active_window, "windows": []}
        except Exception:
            return {"active_window": "Unknown", "windows": []}

    def _get_clipboard_fallback(self) -> str:
        """Get clipboard contents without the awareness module."""
        try:
            if platform.system() == "Linux":
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True, text=True, timeout=3,
                )
                return result.stdout if result.returncode == 0 else ""
            elif platform.system() == "Darwin":
                result = subprocess.run(
                    ["pbpaste"], capture_output=True, text=True, timeout=3,
                )
                return result.stdout if result.returncode == 0 else ""
            elif platform.system() == "Windows":
                import ctypes
                CF_UNICODETEXT = 13
                user32 = ctypes.windll.user32
                kernel32 = ctypes.windll.kernel32
                if not user32.OpenClipboard(0):
                    return ""
                try:
                    handle = user32.GetClipboardData(CF_UNICODETEXT)
                    if handle:
                        kernel32.GlobalLock.restype = ctypes.c_wchar_p
                        return kernel32.GlobalLock(handle) or ""
                finally:
                    user32.CloseClipboard()
            return ""
        except Exception:
            return ""

    def _get_top_processes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N processes by memory usage."""
        try:
            import psutil
            procs = []
            for proc in psutil.process_iter(["pid", "name", "memory_percent"]):
                try:
                    info = proc.info
                    if info["memory_percent"] is not None:
                        procs.append(info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            procs.sort(key=lambda p: p.get("memory_percent", 0), reverse=True)
            return procs[:n]
        except ImportError:
            try:
                if platform.system() != "Windows":
                    result = subprocess.run(
                        ["ps", "aux", "--sort=-%mem"], capture_output=True, text=True, timeout=5,
                    )
                    procs = []
                    for line in result.stdout.strip().split("\n")[1:n + 1]:
                        parts = line.split(None, 10)
                        if len(parts) >= 11:
                            procs.append({
                                "pid": int(parts[1]),
                                "name": parts[10].split("/")[-1],
                                "memory_percent": float(parts[3]),
                            })
                    return procs
            except Exception:
                pass
        return []

    def _get_recent_files(self, n: int = 5) -> List[str]:
        """Get the most recently modified files in common directories."""
        files: List[tuple[float, str]] = []
        search_dirs = [
            Path.home() / "Desktop",
            Path.home() / "Documents",
            Path.home() / "Downloads",
        ]
        for d in search_dirs:
            if d.exists():
                try:
                    for entry in d.iterdir():
                        if entry.is_file():
                            try:
                                mtime = entry.stat().st_mtime
                                files.append((mtime, str(entry)))
                            except OSError:
                                continue
                except PermissionError:
                    continue
        files.sort(reverse=True)
        return [f[1] for f in files[:n]]

    # ──────────────────────────────────────────────
    # Intent parsing
    # ──────────────────────────────────────────────

    async def parse_intent(self, text: str) -> Intent:
        """
        Parse a user command (text or voice transcript) into a structured Intent.

        Uses the rule-based intent parser with ~30 regex patterns.
        Falls back to an "unknown" intent with low confidence if no rule matches.
        """
        cleaned = text.strip()
        if not cleaned:
            return Intent(action="unknown", confidence=0.0, raw_text=text)

        intent = self._parse_intent_rules(cleaned)
        intent.raw_text = text
        return intent

    def _parse_intent_rules(self, text: str) -> Intent:
        """
        Rule-based intent parsing using regex + keyword matching.

        Iterates through all registered rules, picks the match with highest
        confidence. Returns an Intent with action, params, and confidence.
        """
        best_match: Optional[Intent] = None
        best_confidence = 0.0

        for pattern, action, param_extractor, confidence, description in self._intent_rules:
            match = pattern.fullmatch(text)
            if match:
                try:
                    params = param_extractor(match)
                except Exception:
                    params = {}

                intent = Intent(
                    action=action,
                    params=params,
                    confidence=confidence,
                    raw_text=text,
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = intent

        if best_match is None:
            # Check for some fuzzy patterns as a fallback
            best_match = self._parse_intent_fuzzy(text)

        return best_match

    def _parse_intent_fuzzy(self, text: str) -> Intent:
        """
        Fuzzy intent matching for patterns that don't require exact regex.

        Uses keyword presence and heuristics.
        """
        lower = text.lower()

        # Check for mode change commands
        mode_keywords = {
            "go to sleep": OrchestratorMode.SLEEP,
            "sleep mode": OrchestratorMode.SLEEP,
            "wake up": OrchestratorMode.ACTIVE,
            "passive mode": OrchestratorMode.PASSIVE,
            "active mode": OrchestratorMode.ACTIVE,
            "voice mode": OrchestratorMode.VOICE,
            "autonomous mode": OrchestratorMode.AUTONOMOUS,
        }
        for phrase, mode in mode_keywords.items():
            if phrase in lower:
                return Intent(action="set_mode", params={"mode": mode.value}, confidence=0.80, raw_text=text)

        # Check for help / greeting
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if any(g in lower for g in greetings):
            return Intent(action="greet", params={}, confidence=0.70, raw_text=text)

        if "help" in lower:
            return Intent(action="show_help", params={}, confidence=0.75, raw_text=text)

        # Check for thank you
        if any(k in lower for k in ["thanks", "thank you", "thx"]):
            return Intent(action="acknowledge", params={}, confidence=0.80, raw_text=text)

        # Check for cancel / stop
        if any(k in lower for k in ["cancel", "stop", "never mind", "forget it"]):
            return Intent(action="cancel", params={}, confidence=0.75, raw_text=text)

        # Check for write / save file
        write_match = re.search(
            r"(?:write|save|create)\s+(?:the\s+)?(?:file\s+)?[\"']?(\S+?)[\"']?\s+(?:with\s+)?(?:content|text|saying)\s+[\"'](.+?)[\"']",
            text, re.IGNORECASE,
        )
        if write_match:
            return Intent(
                action="write_file",
                params={"path": write_match.group(1), "content": write_match.group(2)},
                confidence=0.82,
                raw_text=text,
            )

        # Check for automation request (multi-step)
        if any(k in lower for k in ["create a task", "automate", "set up a workflow"]):
            return Intent(action="create_task", params={"description": text}, confidence=0.78, raw_text=text)

        # Default: unknown
        return Intent(action="unknown", confidence=0.0, raw_text=text)

    # ──────────────────────────────────────────────
    # Action execution
    # ──────────────────────────────────────────────

    async def execute_intent(self, intent: Intent) -> str:
        """
        Execute a parsed intent using the controller, with permission checking.

        Returns a human-readable result string.
        """
        self._stats["actions_executed"] += 1
        action = intent.action
        params = intent.params

        # Check permissions for non-query actions in ACTIVE mode
        if self._mode == OrchestratorMode.ACTIVE and not action.startswith("query_"):
            if self._permissions and hasattr(self._permissions, "request_permission"):
                granted = await self._permissions.request_permission(action, params)
                if not granted:
                    return f"Action '{action}' was denied by permission manager."

        # Route to appropriate handler
        handler = self._get_action_handler(action)
        if handler:
            try:
                result = await handler(params)
                self._log_activity("action_result", {"action": action, "success": True})
                return result
            except Exception as e:
                self._stats["errors"] += 1
                self._log_activity("action_error", {"action": action, "error": str(e)})
                return f"Error executing '{action}': {e}"

        return self._execute_intent_with_controller(intent)

    def _get_action_handler(self, action: str) -> Optional[Callable[[dict], Any]]:
        """Map an action name to its handler method."""
        handlers = {
            "launch_app": self._action_launch_app,
            "close_window": self._action_close_window,
            "type_text": self._action_type_text,
            "mouse_click": self._action_mouse_click,
            "right_click": self._action_right_click,
            "take_screenshot": self._action_take_screenshot,
            "query_system": self._action_query_system,
            "kill_process": self._action_kill_process,
            "window_action": self._action_window_action,
            "focus_window": self._action_focus_window,
            "clipboard_action": self._action_clipboard,
            "open_url": self._action_open_url,
            "web_search": self._action_web_search,
            "system_action": self._action_system_action,
            "screen_describe": self._action_screen_describe,
            "query_clipboard": self._action_query_clipboard,
            "list_windows": self._action_list_windows,
            "open_terminal": self._action_open_terminal,
            "toggle_recording": self._action_toggle_recording,
            "show_notification": self._action_show_notification,
            "install_package": self._action_install_package,
            "run_command": self._action_run_command,
            "read_file": self._action_read_file,
            "write_file": self._action_write_file,
            "query_time": self._action_query_time,
            "query_date": self._action_query_date,
            "query_uptime": self._action_query_uptime,
            "system_cleanup": self._action_system_cleanup,
            "check_network": self._action_check_network,
            "scroll_action": self._action_scroll,
            "press_key": self._action_press_key,
            "drag_action": self._action_drag,
            "resize_window": self._action_resize_window,
            "move_window": self._action_move_window,
            "set_mode": self._action_set_mode,
            "greet": self._action_greet,
            "show_help": self._action_show_help,
            "acknowledge": self._action_acknowledge,
            "cancel": self._action_cancel,
        }
        return handlers.get(action)

    def _execute_intent_with_controller(self, intent: Intent) -> str:
        """Fallback: try to execute using the controller if available."""
        if self._controller and hasattr(self._controller, "execute"):
            try:
                result = self._controller.execute(intent.action, intent.params)
                return str(result)
            except Exception as e:
                return f"Could not execute '{intent.action}' via controller: {e}"
        return f"Unknown action: '{intent.action}'. Type 'help' for available commands."

    # ── Individual action handlers ──

    async def _action_launch_app(self, params: dict) -> str:
        app = params.get("app", "")
        if not app:
            return "Please specify which application to open."
        if self._controller and hasattr(self._controller, "launch_app"):
            result = await self._controller.launch_app(app)
            return f"Launching {app}..." if result else f"Failed to launch {app}."
        # Fallback: use subprocess
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["start", app], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-a", app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen([app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Opened {app}."
        except Exception as e:
            return f"Failed to open {app}: {e}"

    async def _action_close_window(self, params: dict) -> str:
        app = params.get("app", "")
        if not app:
            return "Please specify which application to close."
        if self._controller and hasattr(self._controller, "close_window"):
            await self._controller.close_window(app)
            return f"Closed {app}."
        try:
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/f", "/im", f"{app}.exe"], capture_output=True, timeout=10)
            elif platform.system() == "Darwin":
                subprocess.run(["osascript", "-e", f'quit app "{app}"'], capture_output=True, timeout=10)
            else:
                subprocess.run(["pkill", "-f", app], capture_output=True, timeout=10)
            return f"Closed {app}."
        except Exception as e:
            return f"Failed to close {app}: {e}"

    async def _action_type_text(self, params: dict) -> str:
        text = params.get("text", "")
        if not text:
            return "No text provided to type."
        if self._controller and hasattr(self._controller, "type_text"):
            await self._controller.type_text(text)
            return f"Typed: \"{text[:50]}{'...' if len(text) > 50 else ''}\""
        return "Typing is not available without a desktop controller."

    async def _action_mouse_click(self, params: dict) -> str:
        target = params.get("target", "")
        if self._controller and hasattr(self._controller, "click"):
            if self._controller.click(target):
                return f"Clicked on: {target}"
        if self._controller and hasattr(self._controller, "click_position"):
            # Try to parse x,y from target
            coords = re.findall(r"(\d+)", target)
            if len(coords) >= 2:
                x, y = int(coords[0]), int(coords[1])
                await self._controller.click_position(x, y)
                return f"Clicked at position ({x}, {y})."
        return f"Could not click on: {target}. Provide coordinates like '100,200'."

    async def _action_right_click(self, params: dict) -> str:
        target = params.get("target", "")
        if self._controller and hasattr(self._controller, "right_click"):
            await self._controller.right_click(target)
            return f"Right-clicked on: {target}"
        return f"Could not right-click on: {target}."

    async def _action_take_screenshot(self, params: dict) -> str:
        region = params.get("region", "full")
        if self._controller and hasattr(self._controller, "take_screenshot"):
            path = await self._controller.take_screenshot(region)
            return f"Screenshot saved to: {path}" if path else "Failed to take screenshot."
        if self._awareness and hasattr(self._awareness, "take_screenshot"):
            path = await self._awareness.take_screenshot(region)
            return f"Screenshot saved to: {path}" if path else "Failed to take screenshot."
        # Fallback using pyautogui or PIL
        try:
            import pyautogui
            img = pyautogui.screenshot()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"screenshot_{timestamp}.png"
            img.save(path)
            return f"Screenshot saved to: {path}"
        except ImportError:
            return "Screenshot capability is not available (install pyautogui)."

    async def _action_query_system(self, params: dict) -> str:
        target = params.get("target", "cpu")
        await self._refresh_cached_state()
        info = self._cached_system_info
        if target in ("cpu",):
            cpu = info.get("cpu", {})
            return f"CPU usage: {cpu.get('percent', 'N/A')}% (cores: {cpu.get('count', 'N/A')})"
        if target in ("ram", "memory"):
            ram = info.get("ram", {})
            return (f"RAM: {ram.get('used', '?')}/{ram.get('total', '?')} GB "
                    f"({ram.get('percent', 'N/A')}% used)")
        if target == "disk":
            disk = info.get("disk", {})
            if not disk:
                return "No disk information available."
            lines = []
            for mount, d in (disk.items() if isinstance(disk, dict) else []):
                lines.append(f"  {mount}: {d.get('percent', '?')}% used ({d.get('free', '?')} GB free)")
            return "Disk usage:\n" + "\n".join(lines)
        if target in ("network",):
            net = info.get("network", {})
            latency = net.get("latency_ms", -1)
            if latency < 0:
                return "Network: unable to measure latency."
            return f"Network latency: {latency:.0f}ms {'(good)' if latency < 100 else '(slow)'}"
        if target == "system":
            cpu = info.get("cpu", {})
            ram = info.get("ram", {})
            return (
                f"System Overview:\n"
                f"  CPU: {cpu.get('percent', 'N/A')}% ({cpu.get('count', 'N/A')} cores)\n"
                f"  RAM: {ram.get('used', '?')}/{ram.get('total', '?')} GB ({ram.get('percent', 'N/A')}%)\n"
                f"  OS: {platform.system()} {platform.release()}"
            )
        return f"No information available for target: {target}"

    async def _action_kill_process(self, params: dict) -> str:
        process = params.get("process", "")
        if not process:
            return "Please specify a process name or PID to kill."
        try:
            import psutil
            killed = 0
            for proc in psutil.process_iter(["pid", "name"]):
                if process.lower() in (proc.info["name"] or "").lower() or process == str(proc.info["pid"]):
                    proc.kill()
                    killed += 1
            return f"Killed {killed} process(es) matching '{process}'." if killed else f"No processes found matching '{process}'."
        except ImportError:
            try:
                result = subprocess.run(
                    ["pkill", "-f", process], capture_output=True, text=True, timeout=10,
                )
                return f"Kill command executed for '{process}'." if result.returncode == 0 else f"Failed to kill '{process}'."
            except Exception as e:
                return f"Failed to kill process '{process}': {e}"

    async def _action_window_action(self, params: dict) -> str:
        action = params.get("action", "")
        window = params.get("window", "")
        if self._controller and hasattr(self._controller, "window_action"):
            await self._controller.window_action(action, window)
            return f"{action.title()}d window: {window}"
        return f"Window action '{action}' not available without a desktop controller."

    async def _action_focus_window(self, params: dict) -> str:
        window = params.get("window", "")
        if self._controller and hasattr(self._controller, "focus_window"):
            await self._controller.focus_window(window)
            return f"Switched to: {window}"
        try:
            if platform.system() == "Linux":
                subprocess.run(["wmctrl", "-a", window], capture_output=True, timeout=5)
            elif platform.system() == "Darwin":
                subprocess.run(["osascript", "-e", f'tell application "{window}" to activate'], capture_output=True, timeout=5)
            elif platform.system() == "Windows":
                subprocess.run(["powershell", "-Command", f"(New-Object -ComObject WScript.Shell).AppActivate('{window}')"], capture_output=True, timeout=5)
            return f"Focused window: {window}"
        except Exception as e:
            return f"Could not focus window '{window}': {e}"

    async def _action_clipboard(self, params: dict) -> str:
        action = params.get("action", "")
        text = params.get("text", "")
        if action == "copy" and text:
            try:
                import pyperclip
                pyperclip.copy(text)
                return f"Copied to clipboard: \"{text[:50]}{'...' if len(text) > 50 else ''}\""
            except ImportError:
                # Fallback
                try:
                    if platform.system() == "Linux":
                        proc = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
                        proc.communicate(text.encode())
                    elif platform.system() == "Darwin":
                        proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                        proc.communicate(text.encode())
                    elif platform.system() == "Windows":
                        import ctypes
                        CF_UNICODETEXT = 13
                        user32 = ctypes.windll.user32
                        kernel32 = ctypes.windll.kernel32
                        user32.OpenClipboard(0)
                        user32.EmptyClipboard()
                        buf = ctypes.c_wchar_p(text)
                        h = kernel32.GlobalAlloc(0x0042, len(text.encode("utf-16-le")) + 2)
                        ctypes.memmove(kernel32.GlobalLock(h), text.encode("utf-16-le"), len(text.encode("utf-16-le")))
                        kernel32.GlobalUnlock(h)
                        user32.SetClipboardData(CF_UNICODETEXT, h)
                        user32.CloseClipboard()
                    return f"Copied to clipboard."
                except Exception as e:
                    return f"Failed to copy: {e}"
        elif action == "paste":
            content = self._cached_clipboard or self._get_clipboard_fallback()
            if content:
                return f"Clipboard contents: \"{content[:200]}{'...' if len(content) > 200 else ''}\""
            return "Clipboard is empty."
        return "Clipboard action not recognized. Use 'copy <text>' or 'paste'."

    async def _action_open_url(self, params: dict) -> str:
        url = params.get("url", "")
        if not url:
            return "Please provide a URL to open."
        try:
            import webbrowser
            webbrowser.open(url)
            return f"Opened: {url}"
        except Exception as e:
            return f"Failed to open URL: {e}"

    async def _action_web_search(self, params: dict) -> str:
        query = params.get("query", "")
        if not query:
            return "Please provide a search query."
        url = f"https://www.google.com/search?q={query}"
        try:
            import webbrowser
            webbrowser.open(url)
            return f"Searched for: \"{query}\""
        except Exception as e:
            return f"Failed to open search: {e}"

    async def _action_system_action(self, params: dict) -> str:
        action = params.get("action", "")
        actions_map = {
            "volume_up": self._system_volume_up,
            "volume_down": self._system_volume_down,
            "mute": self._system_mute,
            "unmute": self._system_unmute,
            "brightness_up": self._system_brightness_up,
            "brightness_down": self._system_brightness_down,
            "lock_screen": self._system_lock_screen,
            "hibernate": self._system_hibernate,
            "shutdown": self._system_shutdown,
            "restart": self._system_restart,
        }
        handler = actions_map.get(action)
        if handler:
            return await handler()
        return f"Unknown system action: {action}"

    async def _system_volume_up(self) -> str:
        if platform.system() == "Darwin":
            subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"], capture_output=True, timeout=5)
            return "Volume increased."
        elif platform.system() == "Linux":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"], capture_output=True, timeout=5)
            return "Volume increased."
        return "Volume control not available on this platform."

    async def _system_volume_down(self) -> str:
        if platform.system() == "Darwin":
            subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"], capture_output=True, timeout=5)
            return "Volume decreased."
        elif platform.system() == "Linux":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"], capture_output=True, timeout=5)
            return "Volume decreased."
        return "Volume control not available on this platform."

    async def _system_mute(self) -> str:
        if platform.system() == "Darwin":
            subprocess.run(["osascript", "-e", "set volume output volume 0"], capture_output=True, timeout=5)
            return "Muted."
        elif platform.system() == "Linux":
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"], capture_output=True, timeout=5)
            return "Muted."
        return "Mute not available on this platform."

    async def _system_unmute(self) -> str:
        if platform.system() == "Darwin":
            subprocess.run(["osascript", "-e", "set volume output volume 50"], capture_output=True, timeout=5)
            return "Unmuted."
        elif platform.system() == "Linux":
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"], capture_output=True, timeout=5)
            return "Unmuted."
        return "Unmute not available on this platform."

    async def _system_brightness_up(self) -> str:
        if platform.system() == "Linux":
            subprocess.run(["brightnessctl", "set", "+10%"], capture_output=True, timeout=5)
            return "Brightness increased."
        elif platform.system() == "Darwin":
            subprocess.run(["brightness", "0.1"], capture_output=True, timeout=5)
            return "Brightness increased."
        return "Brightness control not available on this platform."

    async def _system_brightness_down(self) -> str:
        if platform.system() == "Linux":
            subprocess.run(["brightnessctl", "set", "10%-"], capture_output=True, timeout=5)
            return "Brightness decreased."
        elif platform.system() == "Darwin":
            subprocess.run(["brightness", "-0.1"], capture_output=True, timeout=5)
            return "Brightness decreased."
        return "Brightness control not available on this platform."

    async def _system_lock_screen(self) -> str:
        try:
            if platform.system() == "Linux":
                subprocess.run(["loginctl", "lock-session"], capture_output=True, timeout=5)
            elif platform.system() == "Darwin":
                subprocess.run(
                    ["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"],
                    capture_output=True, timeout=5,
                )
            elif platform.system() == "Windows":
                subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], capture_output=True, timeout=5)
            return "Screen locked."
        except Exception as e:
            return f"Failed to lock screen: {e}"

    async def _system_hibernate(self) -> str:
        return "Hibernate is not supported for safety reasons."

    async def _system_shutdown(self) -> str:
        return "Shutdown is not supported for safety reasons."

    async def _system_restart(self) -> str:
        return "Restart is not supported for safety reasons."

    async def _action_screen_describe(self, params: dict) -> str:
        """Describe what's currently on the screen using OCR and AI."""
        return await self.take_screenshot_and_describe()

    async def _action_query_clipboard(self, params: dict) -> str:
        await self._refresh_cached_state()
        content = self._cached_clipboard or self._get_clipboard_fallback()
        if not content:
            return "Clipboard is empty."
        display = content[:500] + ("..." if len(content) > 500 else "")
        return f"Clipboard contents ({len(content)} chars):\n{display}"

    async def _action_list_windows(self, params: dict) -> str:
        await self._refresh_cached_state()
        windows = self._cached_window_info.get("windows", [])
        active = self._cached_window_info.get("active_window", "Unknown")
        if not windows:
            return "No windows found."
        lines = [f"Active: {active}", ""]
        for i, w in enumerate(windows, 1):
            title = w.get("title", "Untitled")
            app = w.get("app", "")
            lines.append(f"  {i}. {title} [{app}]")
        return "Open windows:\n" + "\n".join(lines)

    async def _action_open_terminal(self, params: dict) -> str:
        terminal_type = params.get("terminal_type", "default")
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["cmd.exe", "/k"], creationflags=subprocess.CREATE_NEW_CONSOLE)
                return "Opened Command Prompt."
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-a", "Terminal"])
                return "Opened Terminal."
            else:
                term = os.environ.get("TERMINAL", "gnome-terminal")
                subprocess.Popen([term])
                return f"Opened {term}."
        except Exception as e:
            return f"Failed to open terminal: {e}"

    async def _action_toggle_recording(self, params: dict) -> str:
        action = params.get("action", "start")
        if action == "start":
            return "Screen recording is not available in this version."
        return "No active recording to stop."

    async def _action_show_notification(self, params: dict) -> str:
        message = params.get("message", "Desktop Assistant Notification")
        try:
            if platform.system() == "Linux":
                subprocess.run(
                    ["notify-send", "Desktop Assistant", message], capture_output=True, timeout=5,
                )
            elif platform.system() == "Darwin":
                script = f'display notification "{message}" with title "Desktop Assistant"'
                subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
            elif platform.system() == "Windows":
                try:
                    from ctypes import windll
                    windll.user32.MessageBoxW(0, message, "Desktop Assistant", 0x40)
                except Exception:
                    subprocess.run(
                        ["powershell", "-Command",
                         f"[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms'); "
                         f"[System.Windows.Forms.MessageBox]::Show('{message}', 'Desktop Assistant')"],
                        capture_output=True, timeout=10,
                    )
            return f"Notification shown: \"{message[:80]}\""
        except Exception as e:
            return f"Failed to show notification: {e}"

    async def _action_install_package(self, params: dict) -> str:
        package = params.get("package", "")
        if not package:
            return "Please specify a package to install."
        # Check if permission is needed
        if self._mode != OrchestratorMode.AUTONOMOUS:
            if self._permissions and hasattr(self._permissions, "request_permission"):
                granted = await self._permissions.request_permission("install_package", {"package": package})
                if not granted:
                    return f"Package installation of '{package}' was denied."
        # Detect package manager from context
        cmd = self._detect_install_command(package)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                return f"Successfully installed: {package}"
            return f"Failed to install {package}: {result.stderr[:200]}"
        except subprocess.TimeoutExpired:
            return f"Installation of {package} timed out."
        except Exception as e:
            return f"Failed to install {package}: {e}"

    def _detect_install_command(self, package: str) -> List[str]:
        """Detect the appropriate install command based on the system."""
        if package.startswith("pip "):
            return [sys.executable, "-m", "pip", "install"] + package.split()[1:]
        if package.startswith("npm "):
            return ["npm", "install"] + package.split()[1:]
        if platform.system() == "Linux":
            # Check for apt, dnf, pacman
            if shutil.which("apt"):
                return ["sudo", "apt", "install", "-y", package]
            if shutil.which("dnf"):
                return ["sudo", "dnf", "install", "-y", package]
            if shutil.which("pacman"):
                return ["sudo", "pacman", "-S", "--noconfirm", package]
        return [sys.executable, "-m", "pip", "install", package]

    async def _action_run_command(self, params: dict) -> str:
        command = params.get("command", "")
        if not command:
            return "Please provide a command to run."
        if self._mode != OrchestratorMode.AUTONOMOUS:
            if self._permissions and hasattr(self._permissions, "request_permission"):
                granted = await self._permissions.request_permission("run_command", {"command": command})
                if not granted:
                    return f"Command execution was denied: {command}"
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=60,
            )
            output = result.stdout or ""
            error = result.stderr or ""
            if result.returncode != 0:
                return f"Command exited with code {result.returncode}:\n{error[:500]}"
            return output[:1000] if output else "Command executed successfully (no output)."
        except subprocess.TimeoutExpired:
            return "Command timed out after 60 seconds."
        except Exception as e:
            return f"Failed to run command: {e}"

    async def _action_read_file(self, params: dict) -> str:
        path = params.get("path", "")
        if not path:
            return "Please provide a file path to read."
        expanded = os.path.expanduser(path)
        try:
            p = Path(expanded)
            if not p.exists():
                return f"File not found: {expanded}"
            if not p.is_file():
                return f"Not a file: {expanded}"
            content = p.read_text(encoding="utf-8", errors="replace")
            max_chars = 5000
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n[... truncated, {len(content)} total chars]"
            return content
        except Exception as e:
            return f"Failed to read file '{expanded}': {e}"

    async def _action_write_file(self, params: dict) -> str:
        path = params.get("path", "")
        content = params.get("content", "")
        if not path:
            return "Please provide a file path."
        expanded = os.path.expanduser(path)
        if self._mode != OrchestratorMode.AUTONOMOUS:
            if self._permissions and hasattr(self._permissions, "request_permission"):
                granted = await self._permissions.request_permission("write_file", {"path": expanded})
                if not granted:
                    return f"File write to '{expanded}' was denied."
        try:
            p = Path(expanded)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} characters to {expanded}."
        except Exception as e:
            return f"Failed to write file '{expanded}': {e}"

    async def _action_query_time(self, params: dict) -> str:
        now = datetime.now()
        return f"The current time is {now.strftime('%I:%M:%S %p')} ({now.strftime('%H:%M:%S')} 24h)."

    async def _action_query_date(self, params: dict) -> str:
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."

    async def _action_query_uptime(self, params: dict) -> str:
        try:
            if platform.system() == "Linux":
                result = subprocess.run(["uptime", "-p"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return f"System has been {result.stdout.strip()}."
            elif platform.system() == "Darwin":
                result = subprocess.run(["uptime"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return f"System uptime: {result.stdout.strip()}"
            elif platform.system() == "Windows":
                result = subprocess.run(
                    ["powershell", "-Command", "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    return f"Last boot: {result.stdout.strip()}"

            session_mins = (time.time() - self._session_start) / 60
            return f"Session uptime: {int(session_mins // 60)}h {int(session_mins % 60)}m. System uptime not available."
        except Exception as e:
            return f"Could not query uptime: {e}"

    async def _action_system_cleanup(self, params: dict) -> str:
        if self._mode != OrchestratorMode.AUTONOMOUS:
            if self._permissions and hasattr(self._permissions, "request_permission"):
                granted = await self._permissions.request_permission("system_cleanup", {})
                if not granted:
                    return "System cleanup was denied."
        cleaned = []
        try:
            # Temp files
            if platform.system() != "Windows":
                tmp = Path("/tmp")
                if tmp.exists():
                    count = sum(1 for _ in tmp.iterdir() if _.is_file())
                    subprocess.run(["find", "/tmp", "-type", "f", "-mtime", "+1", "-delete"],
                                   capture_output=True, timeout=30)
                    cleaned.append(f"Cleaned temp files ({count} files older than 1 day)")
        except Exception as e:
            cleaned.append(f"Temp cleanup error: {e}")
        try:
            # Python cache
            for cache_dir in [Path.home() / ".cache" / "pip", Path.cwd() / "__pycache__"]:
                if cache_dir.exists():
                    size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    cleaned.append(f"Removed {cache_dir} ({size / 1024:.1f} KB)")
        except Exception:
            pass
        return "Cleanup complete:\n" + "\n".join(cleaned) if cleaned else "Nothing to clean up."

    async def _action_check_network(self, params: dict) -> str:
        results = []
        # Ping test
        latency = await self._ping_test("8.8.8.8", count=3, timeout=3)
        if latency >= 0:
            status = "good" if latency < 100 else "slow" if latency < 500 else "very slow"
            results.append(f"Google DNS (8.8.8.8): {latency:.0f}ms — {status}")
        else:
            results.append("Google DNS (8.8.8.8): unreachable")
        # DNS test
        dns_latency = await self._ping_test("1.1.1.1", count=1, timeout=3)
        if dns_latency >= 0:
            results.append(f"Cloudflare DNS (1.1.1.1): {dns_latency:.0f}ms")
        # Web test
        web_ok = await self._http_test("https://www.google.com")
        results.append(f"Google web: {'reachable' if web_ok else 'unreachable'}")
        web_ok2 = await self._http_test("https://www.github.com")
        results.append(f"GitHub web: {'reachable' if web_ok2 else 'unreachable'}")
        return "Network status:\n" + "\n".join(results)

    async def _action_scroll(self, params: dict) -> str:
        direction = params.get("direction", "down")
        amount = params.get("amount", 3)
        if self._controller and hasattr(self._controller, "scroll"):
            await self._controller.scroll(direction, amount)
            return f"Scrolled {direction} {amount} times."
        return "Scroll action requires a desktop controller."

    async def _action_press_key(self, params: dict) -> str:
        key = params.get("key", "")
        if not key:
            return "Please specify a key to press."
        key_map = {
            "enter": "enter", "return": "enter",
            "tab": "tab", "escape": "escape", "esc": "escape",
            "space": "space", "spacebar": "space",
            "backspace": "backspace", "delete": "delete", "del": "delete",
            "up": "up", "down": "down", "left": "left", "right": "right",
            "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "shift": "shift",
            "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4",
            "f5": "f5", "f6": "f6", "f7": "f7", "f8": "f8",
            "f9": "f9", "f10": "f10", "f11": "f11", "f12": "f12",
        }
        mapped = key_map.get(key.lower(), key)
        if self._controller and hasattr(self._controller, "press_key"):
            await self._controller.press_key(mapped)
            return f"Pressed key: {mapped}"
        try:
            import pyautogui
            pyautogui.press(mapped)
            return f"Pressed key: {mapped}"
        except ImportError:
            return f"Key press not available (install pyautogui)."

    async def _action_drag(self, params: dict) -> str:
        from_pos = params.get("from", "")
        to_pos = params.get("to", "")
        if self._controller and hasattr(self._controller, "drag"):
            await self._controller.drag(from_pos, to_pos)
            return f"Dragged from {from_pos} to {to_pos}."
        return "Drag action requires a desktop controller."

    async def _action_resize_window(self, params: dict) -> str:
        window = params.get("window", "")
        size = params.get("size", "")
        if self._controller and hasattr(self._controller, "resize_window"):
            await self._controller.resize_window(window, size)
            return f"Resized {window} to {size}."
        return "Resize action requires a desktop controller."

    async def _action_move_window(self, params: dict) -> str:
        window = params.get("window", "")
        position = params.get("position", "")
        if self._controller and hasattr(self._controller, "move_window"):
            await self._controller.move_window(window, position)
            return f"Moved {window} to {position}."
        return "Move window action requires a desktop controller."

    async def _action_set_mode(self, params: dict) -> str:
        mode_str = params.get("mode", "active")
        try:
            mode = OrchestratorMode(mode_str)
            await self.set_mode(mode)
            return f"Switched to {mode.value} mode."
        except ValueError:
            return f"Unknown mode: {mode_str}. Available: {', '.join(m.value for m in OrchestratorMode)}"

    async def _action_greet(self, params: dict) -> str:
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"
        return f"{greeting}! I'm your desktop assistant. I can help you control your computer, answer questions, and automate tasks. What would you like to do?"

    async def _action_show_help(self, params: dict) -> str:
        return (
            "## Desktop Assistant Commands\n\n"
            "**App Control:**\n"
            "  open [app], close [app], switch to [window], list windows\n\n"
            "**System:**\n"
            "  what's my cpu/ram/disk, check internet, clean up\n\n"
            "**Interaction:**\n"
            "  type [text], click [position], screenshot, press [key]\n\n"
            "**Clipboard:**\n"
            "  copy [text], paste, what's on my clipboard\n\n"
            "**Web:**\n"
            "  go to [url], search [query]\n\n"
            "**System Actions:**\n"
            "  volume up/down/mute, brightness up/down, lock screen\n\n"
            "**Files:**\n"
            "  read file [path], run [command]\n\n"
            "**Info:**\n"
            "  what time is it, what day is it, how long has pc been on\n\n"
            "**Modes:**\n"
            "  sleep mode, passive mode, active mode, voice mode, autonomous mode\n\n"
            "**Automation:**\n"
            "  describe a multi-step task and I'll break it down for you\n\n"
            "Or just ask me anything in natural language!"
        )

    async def _action_acknowledge(self, params: dict) -> str:
        responses = [
            "You're welcome!",
            "Happy to help!",
            "Any time!",
            "Glad I could assist!",
        ]
        import random
        return random.choice(responses)

    async def _action_cancel(self, params: dict) -> str:
        return "Okay, cancelled. What else can I help with?"

    # ──────────────────────────────────────────────
    # Proactive suggestions
    # ──────────────────────────────────────────────

    async def get_proactive_suggestions(self) -> List[ProactiveSuggestion]:
        """
        Generate proactive suggestions based on the current desktop state.

        Runs through all suggestion generators and collects those that apply.
        Respects a cooldown period to avoid spamming the user.
        """
        now = time.time()
        if now - self._last_suggestion_time < self._suggestion_cooldown:
            return []

        await self._refresh_cached_state()
        suggestions: List[ProactiveSuggestion] = []

        for generator in self._suggestion_generators:
            try:
                result = await generator(self._cached_system_info)
                if result:
                    suggestions.append(result)
            except Exception as e:
                logger.debug("Suggestion generator error: %s", e)

        if suggestions:
            self._last_suggestion_time = now
            self._stats["suggestions_given"] += len(suggestions)

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 99))

        return suggestions

    async def _proactive_suggestion_loop(self) -> None:
        """Background task that periodically checks for proactive suggestions."""
        while not self._shutting_down:
            try:
                await asyncio.sleep(60)  # Check every 60 seconds
                if self._shutting_down or self._mode in (OrchestratorMode.SLEEP, OrchestratorMode.PASSIVE):
                    continue
                suggestions = await self.get_proactive_suggestions()
                if suggestions:
                    for suggestion in suggestions[:2]:  # Max 2 at a time
                        logger.info("Proactive suggestion: %s — %s", suggestion.title, suggestion.description)
                        self._log_activity("proactive_suggestion", {
                            "title": suggestion.title,
                            "action": suggestion.action,
                            "priority": suggestion.priority,
                        })
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Proactive suggestion loop error: %s", e)

    # ──────────────────────────────────────────────
    # Screen understanding (screenshot + OCR)
    # ──────────────────────────────────────────────

    async def take_screenshot_and_describe(self) -> str:
        """
        Capture a screenshot, run OCR, and provide a description.

        Returns a string describing what's on the screen with OCR text.
        """
        parts: List[str] = []

        # Take screenshot
        screenshot_path = None
        if self._controller and hasattr(self._controller, "take_screenshot"):
            screenshot_path = await self._controller.take_screenshot("full")
        elif self._awareness and hasattr(self._awareness, "take_screenshot"):
            screenshot_path = await self._awareness.take_screenshot("full")

        if not screenshot_path:
            # Fallback: take screenshot ourselves
            try:
                import pyautogui
                import tempfile
                fd, screenshot_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                img = pyautogui.screenshot()
                img.save(screenshot_path)
            except ImportError:
                return "Screenshot capability not available (install pyautogui)."
            except Exception as e:
                return f"Failed to take screenshot: {e}"

        parts.append(f"Screenshot captured: {screenshot_path}")

        # Run OCR
        ocr_text = await self._run_ocr(screenshot_path)
        if ocr_text:
            display = ocr_text[:1000] + ("..." if len(ocr_text) > 1000 else "")
            parts.append(f"\nOCR Text:\n{display}")
        else:
            parts.append("\nNo text detected via OCR.")

        # Get active window context
        await self._refresh_cached_state()
        active = self._cached_window_info.get("active_window", "Unknown")
        if active and active != "Unknown":
            parts.append(f"\nActive window: {active}")

        return "\n".join(parts)

    async def _run_ocr(self, image_path: str) -> str:
        """Run OCR on an image file. Returns extracted text or empty string."""
        if self._ocr_available is False:
            return ""

        # Try pytesseract
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            cleaned = text.strip()
            if cleaned:
                self._ocr_available = True
                return cleaned
        except ImportError:
            self._ocr_available = False
        except Exception:
            pass

        # Try easyocr as fallback
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            results = reader.readtext(image_path)
            text = " ".join(r[1] for r in results)
            cleaned = text.strip()
            if cleaned:
                self._ocr_available = True
                return cleaned
        except ImportError:
            self._ocr_available = False
        except Exception:
            pass

        return ""

    def _check_ocr_available(self) -> bool:
        """Check if any OCR engine is available."""
        try:
            import pytesseract
            return True
        except ImportError:
            pass
        try:
            import easyocr
            return True
        except ImportError:
            pass
        return False

    # ──────────────────────────────────────────────
    # Task automation
    # ──────────────────────────────────────────────

    async def create_task(self, description: str) -> TaskAutomation:
        """
        Create a multi-step task from a user description.

        Attempts to match known task patterns. If none match, delegates
        to the AI agent to decompose the task.
        """
        # Try pattern-based decomposition first
        for pattern, step_generator in TASK_PATTERNS:
            if pattern.search(description):
                steps = step_generator(pattern.search(description))
                task_steps = []
                for i, step_def in enumerate(steps, 1):
                    intent = Intent(
                        action=step_def["action"],
                        params=step_def.get("params", {}),
                        confidence=0.85,
                        raw_text=step_def["description"],
                    )
                    task_steps.append(TaskStep(
                        step_number=i,
                        description=step_def["description"],
                        intent=intent,
                        status="pending",
                    ))
                task = TaskAutomation(
                    name=f"Task: {description[:60]}",
                    description=description,
                    steps=task_steps,
                    status="created",
                )
                self._active_tasks[task.name] = task
                self._log_activity("task_created", {"name": task.name, "steps": len(task_steps)})
                return task

        # AI-based decomposition
        task_steps = await self._ai_decompose_task(description)
        task = TaskAutomation(
            name=f"Task: {description[:60]}",
            description=description,
            steps=task_steps,
            status="created",
        )
        self._active_tasks[task.name] = task
        self._log_activity("task_created", {"name": task.name, "steps": len(task_steps), "source": "ai"})
        return task

    async def _ai_decompose_task(self, description: str) -> List[TaskStep]:
        """Use the AI agent to decompose a task into steps."""
        prompt = (
            f"Break down the following task into concrete, executable steps. "
            f"For each step, provide a description and the type of action needed "
            f"(run_command, launch_app, type_text, open_url, read_file, etc.).\n\n"
            f"Task: {description}\n\n"
            f"Respond in this exact format (one per line):\n"
            f"STEP: <description> | ACTION: <action> | PARAMS: <key=value,key=value>\n"
        )
        response = await self._query_agent(prompt)
        steps: List[TaskStep] = []
        step_pattern = re.compile(
            r"STEP:\s*(.+?)\s*\|\s*ACTION:\s*(\w+)\s*\|\s*PARAMS:\s*(.+)",
            re.IGNORECASE,
        )
        for i, line in enumerate(response.split("\n"), 1):
            line = line.strip()
            if not line:
                continue
            match = step_pattern.search(line)
            if match:
                desc = match.group(1).strip()
                action = match.group(2).strip()
                params_str = match.group(3).strip()
                params: Dict[str, str] = {}
                for pair in params_str.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        params[k.strip()] = v.strip()
                intent = Intent(action=action, params=params, confidence=0.70, raw_text=desc)
            else:
                # Plain text step — treat as a general command
                intent = Intent(action="run_command", params={"command": line}, confidence=0.50, raw_text=line)
                desc = line
            steps.append(TaskStep(step_number=i, description=desc, intent=intent, status="pending"))
        return steps

    async def execute_task(self, task: TaskAutomation) -> str:
        """
        Execute a multi-step task step by step.

        In ACTIVE mode, asks for confirmation before each step.
        In AUTONOMOUS mode, executes all steps without confirmation.
        Returns a summary of results.
        """
        if not task.steps:
            return "Task has no steps to execute."

        task.status = "running"
        results: List[str] = [f"Executing task: {task.name}\n"]

        for step in task.steps:
            step.status = "executing"
            results.append(f"\n--- Step {step.step_number}: {step.description} ---")

            # Ask for confirmation in ACTIVE mode
            if self._mode == OrchestratorMode.ACTIVE:
                if self._permissions and hasattr(self._permissions, "request_permission"):
                    granted = await self._permissions.request_permission(
                        f"task_step_{step.step_number}",
                        {"description": step.description, "action": step.intent.action},
                    )
                    if not granted:
                        step.status = "skipped"
                        step.result = "Skipped by user"
                        results.append("  Skipped (permission denied)")
                        continue

            # Execute the step
            try:
                result = await self.execute_intent(step.intent)
                step.status = "completed"
                step.result = result
                results.append(f"  Result: {result[:200]}")
            except Exception as e:
                step.status = "failed"
                step.result = str(e)
                results.append(f"  Failed: {e}")
                task.status = "failed"
                self._log_activity("task_step_failed", {
                    "task": task.name,
                    "step": step.step_number,
                    "error": str(e),
                })
                break

        # Determine final status
        if all(s.status == "completed" for s in task.steps):
            task.status = "completed"
            self._stats["tasks_completed"] += 1
        elif all(s.status in ("completed", "skipped") for s in task.steps):
            task.status = "completed"
            self._stats["tasks_completed"] += 1

        self._log_activity("task_finished", {"name": task.name, "status": task.status})
        return "\n".join(results)

    # ──────────────────────────────────────────────
    # Voice handling
    # ──────────────────────────────────────────────

    async def _speak_response(self, text: str) -> None:
        """Speak a response using the voice engine."""
        if not self._voice or not hasattr(self._voice, "speak"):
            return
        try:
            # Truncate long responses for speech
            max_speech_chars = 500
            speech_text = text[:max_speech_chars]
            if len(text) > max_speech_chars:
                last_period = speech_text.rfind(".")
                if last_period > 0:
                    speech_text = speech_text[:last_period + 1]
            await self._voice.speak(speech_text)
        except Exception as e:
            logger.debug("Voice speak error: %s", e)

    async def handle_voice_event(self, event: Any) -> None:
        """
        Handle incoming voice events from the voice engine.

        Processes transcribed speech as user input.
        """
        if event is None:
            return

        # Extract text from various voice event formats
        text = ""
        if isinstance(event, dict):
            text = event.get("text", "") or event.get("transcript", "")
        elif isinstance(event, str):
            text = event
        elif hasattr(event, "text"):
            text = event.text
        elif hasattr(event, "transcript"):
            text = event.transcript

        if not text or not text.strip():
            return

        logger.info("Voice input: %s", text)
        self._log_activity("voice_input", {"text": text[:200]})

        if self._mode == OrchestratorMode.SLEEP:
            # Wake trigger
            wake_words = ["hey assistant", "wake up", "hello assistant"]
            if any(w in text.lower() for w in wake_words):
                await self.set_mode(OrchestratorMode.VOICE)
                await self._speak_response("I'm awake. How can I help?")
            return

        if self._mode in (OrchestratorMode.VOICE, OrchestratorMode.ACTIVE, OrchestratorMode.AUTONOMOUS):
            response = await self.process_input(text, source="voice")
            if self._mode == OrchestratorMode.VOICE:
                await self._speak_response(response)

    async def _voice_event_loop(self) -> None:
        """Background task that listens for voice events."""
        if not self._voice:
            return
        while not self._shutting_down:
            try:
                if hasattr(self._voice, "get_event"):
                    event = await asyncio.wait_for(self._voice.get_event(), timeout=1.0)
                    await self.handle_voice_event(event)
                elif hasattr(self._voice, "listen"):
                    text = await asyncio.wait_for(self._voice.listen(), timeout=1.0)
                    if text:
                        await self.handle_voice_event(text)
                else:
                    await asyncio.sleep(2.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Voice event loop error: %s", e)
                await asyncio.sleep(2.0)

    # ──────────────────────────────────────────────
    # Desktop event handling
    # ──────────────────────────────────────────────

    async def handle_desktop_event(self, event: Any) -> None:
        """
        Handle incoming desktop events from the awareness module.

        Processes window focus changes, clipboard updates, and other events.
        """
        if event is None:
            return

        if isinstance(event, dict):
            event_type = event.get("type", "unknown")
        elif hasattr(event, "type"):
            event_type = event.type
        else:
            event_type = str(event)

        self._log_activity("desktop_event", {"type": event_type, "data": str(event)[:200]})

        # Refresh relevant cached state based on event type
        if event_type in ("window_focus", "window_created", "window_closed"):
            self._last_context_refresh = 0  # Force refresh
        elif event_type == "clipboard_change":
            self._cached_clipboard = self._get_clipboard_fallback()

    # ──────────────────────────────────────────────
    # Input queue (multi-modal)
    # ──────────────────────────────────────────────

    async def _input_queue_loop(self) -> None:
        """Background task that processes queued inputs from all sources."""
        while not self._shutting_down:
            try:
                text, source = await asyncio.wait_for(self._input_queue.get(), timeout=1.0)
                await self.process_input(text, source=source)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Input queue loop error: %s", e)

    def queue_input(self, text: str, source: str = "text") -> bool:
        """
        Queue an input for async processing. Thread-safe.

        Returns True if queued successfully, False if the queue is full.
        """
        try:
            self._input_queue.put_nowait((text, source))
            return True
        except asyncio.QueueFull:
            logger.warning("Input queue full, dropping input from %s", source)
            return False

    # ──────────────────────────────────────────────
    # Status & activity
    # ──────────────────────────────────────────────

    async def get_status(self) -> dict:
        """
        Return a comprehensive status dict about the orchestrator.

        Includes: mode, component states, stats, active tasks, and uptime.
        """
        await self._refresh_cached_state()

        component_status = {}
        for name, comp in [("awareness", self._awareness), ("voice", self._voice),
                           ("controller", self._controller), ("permissions", self._permissions)]:
            if comp is None:
                component_status[name] = "not_configured"
            elif hasattr(comp, "is_running"):
                component_status[name] = "running" if comp.is_running else "stopped"
            elif hasattr(comp, "connected"):
                component_status[name] = "connected" if comp.connected else "disconnected"
            else:
                component_status[name] = "available"

        active_tasks_summary = []
        for name, task in self._active_tasks.items():
            completed = sum(1 for s in task.steps if s.status == "completed")
            active_tasks_summary.append({
                "name": task.name,
                "status": task.status,
                "steps": f"{completed}/{len(task.steps)}",
            })

        session_elapsed = time.time() - self._session_start

        return {
            "mode": self._mode.value,
            "initialized": self._initialized,
            "session_duration_minutes": round(session_elapsed / 60, 1),
            "components": component_status,
            "stats": dict(self._stats),
            "active_tasks": active_tasks_summary,
            "ocr_available": self._ocr_available,
            "cached_context_age_seconds": round(time.time() - self._last_context_refresh, 1),
            "system": {
                "cpu_percent": self._cached_system_info.get("cpu", {}).get("percent", "N/A"),
                "ram_percent": self._cached_system_info.get("ram", {}).get("percent", "N/A"),
                "network_latency_ms": self._cached_system_info.get("network", {}).get("latency_ms", -1),
            },
            "active_window": self._cached_window_info.get("active_window", "Unknown"),
            "input_queue_size": self._input_queue.qsize(),
        }

    async def get_activity_summary(self, minutes: int = 30) -> str:
        """
        Summarize the user's recent activity over the specified time window.

        Aggregates activity log entries from the last N minutes and produces
        a human-readable summary.
        """
        cutoff = time.time() - (minutes * 60)
        recent = [e for e in self._activity_log if e.get("timestamp", 0) >= cutoff]

        if not recent:
            return f"No activity recorded in the last {minutes} minutes."

        # Count by type
        type_counts: Dict[str, int] = {}
        for entry in recent:
            etype = entry.get("type", "unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1

        lines = [f"## Activity Summary (last {minutes} minutes)"]
        lines.append(f"Total events: {len(recent)}")
        lines.append("")

        for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {etype}: {count}")

        # Notable events
        notable = [e for e in recent if e.get("type") in ("user_input", "intent_executed", "proactive_suggestion")]
        if notable:
            lines.append("\nNotable events:")
            for entry in notable[-10:]:
                ts = datetime.fromtimestamp(entry.get("timestamp", 0)).strftime("%H:%M:%S")
                data = entry.get("data", {})
                if entry["type"] == "user_input":
                    lines.append(f"  [{ts}] Input: \"{data.get('text', '')[:80]}\" (via {data.get('source', '?')})")
                elif entry["type"] == "intent_executed":
                    lines.append(f"  [{ts}] Executed: {data.get('action', '?')} (confidence: {data.get('confidence', '?'):.0%})")
                elif entry["type"] == "proactive_suggestion":
                    lines.append(f"  [{ts}] Suggestion: {data.get('title', '?')} [{data.get('priority', '?')}]")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # Internal utilities
    # ──────────────────────────────────────────────

    def _log_activity(self, event_type: str, data: dict) -> None:
        """Add an entry to the activity log with a timestamp."""
        entry = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        self._activity_log.append(entry)
        # Trim the log if it exceeds the max
        if len(self._activity_log) > self._max_activity_entries:
            self._activity_log = self._activity_log[-self._max_activity_entries:]

    async def _query_agent(self, prompt: str) -> str:
        """
        Send a query to the AI agent and return the text response.

        Falls back gracefully if the agent is not configured.
        """
        if self._agent is None:
            return "AI agent is not configured. Available commands: type 'help' for a list."

        # Check if agent has async run method
        if hasattr(self._agent, "run"):
            try:
                chunks = []
                async for event in self._agent.run(prompt):
                    if hasattr(event, "data") and isinstance(event.data, str):
                        chunks.append(event.data)
                return "".join(chunks).strip()
            except Exception as e:
                logger.error("Agent run error: %s", e)
                return f"AI agent error: {e}"

        # Check if agent has a simple chat/send method
        if hasattr(self._agent, "chat"):
            try:
                return str(await self._agent.chat(prompt))
            except Exception as e:
                return f"AI agent error: {e}"

        return "AI agent is not configured properly."

    async def _ping_test(self, host: str, count: int = 3, timeout: int = 3) -> float:
        """Run a ping test and return average latency in ms, or -1 on failure."""
        try:
            if platform.system() == "Windows":
                cmd = ["ping", "-n", str(count), "-w", str(timeout * 1000), host]
            else:
                cmd = ["ping", "-c", str(count), "-W", str(timeout), host]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout * count + 5)
            if result.returncode != 0:
                return -1.0
            # Parse average latency
            match = re.search(r"(?:rtt|round-trip).*?=\s*[\d.]+/([\d.]+)", result.stdout)
            if match:
                return float(match.group(1))
            # Try simpler match
            match = re.search(r"time[=<]([\d.]+)\s*ms", result.stdout)
            if match:
                return float(match.group(1))
            return -1.0
        except Exception:
            return -1.0

    async def _http_test(self, url: str, timeout: int = 5) -> bool:
        """Test if a URL is reachable. Returns True if status < 400."""
        try:
            import urllib.request
            import urllib.error
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", "ClaudeClone/1.0")
            response = urllib.request.urlopen(req, timeout=timeout)
            return response.status < 400
        except Exception:
            return False
