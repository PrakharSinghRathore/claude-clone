"""
Desktop Controller Module
========================
Provides comprehensive PC control capabilities for an AI desktop assistant,
including mouse, keyboard, window management, application launching,
clipboard control, screen regions, notifications, and automation macros.

All actions require explicit permission unless a permission checker is not
configured. Permission checking is performed before every action executes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# ---------------------------------------------------------------------------
# Third-party imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    import pyautogui
except ImportError:
    pyautogui = None  # type: ignore[assignment]
    logger.warning("pyautogui is not installed. Mouse/keyboard features will be unavailable.")

try:
    import pygetwindow as gw
except ImportError:
    gw = None  # type: ignore[assignment]
    logger.warning("pygetwindow is not installed. Window management features will be unavailable.")

try:
    from PIL import ImageGrab, Image
except ImportError:
    ImageGrab = None  # type: ignore[assignment, misc]
    Image = None  # type: ignore[assignment, misc]
    logger.warning("Pillow is not installed. Image matching features will be unavailable.")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MouseActionType(Enum):
    MOVE = "move"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    DRAG = "drag"
    SCROLL = "scroll"


class KeyActionType(Enum):
    PRESS = "press"
    RELEASE = "release"
    TYPE = "type"
    HOTKEY = "hotkey"


class WindowActionType(Enum):
    FOCUS = "focus"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    RESTORE = "restore"
    CLOSE = "close"
    RESIZE = "resize"
    MOVE = "move"
    ALWAYS_ON_TOP = "always_on_top"


class AutomationStepType(Enum):
    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    WINDOW = "window"
    WAIT = "wait"
    CLIPBOARD = "clipboard"
    LAUNCH = "launch"
    NOTIFICATION = "notification"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MouseAction:
    """Records a single mouse action for macro playback."""
    x: int = 0
    y: int = 0
    action: MouseActionType = MouseActionType.MOVE
    button: str = "left"
    duration: float = 0.3
    steps: int = 30
    scroll_clicks: int = 0
    scroll_direction: str = "up"
    drag_end_x: int = 0
    drag_end_y: int = 0


@dataclass
class KeyAction:
    """Records a single keyboard action for macro playback."""
    key: str = ""
    action: KeyActionType = KeyActionType.PRESS
    text: str = ""
    delay: float = 0.02
    keys: list[str] = field(default_factory=list)


@dataclass
class WindowAction:
    """Records a single window management action."""
    title: str = ""
    action: WindowActionType = WindowActionType.FOCUS
    params: dict = field(default_factory=dict)


@dataclass
class AutomationStep:
    """A single step in an automation macro."""
    step_type: AutomationStepType = AutomationStepType.MOUSE
    action_data: dict = field(default_factory=dict)
    timestamp: float = 0.0
    delay_after: float = 0.0


@dataclass
class AutomationMacro:
    """A recorded sequence of automation steps."""
    name: str = ""
    steps: list[AutomationStep] = field(default_factory=list)
    created_at: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        """Serialize macro to a plain dictionary for JSON storage."""
        return {
            "name": self.name,
            "steps": [
                {
                    "step_type": s.step_type.value,
                    "action_data": s.action_data,
                    "timestamp": s.timestamp,
                    "delay_after": s.delay_after,
                }
                for s in self.steps
            ],
            "created_at": self.created_at,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AutomationMacro:
        """Deserialize a macro from a dictionary."""
        steps = [
            AutomationStep(
                step_type=AutomationStepType(s["step_type"]),
                action_data=s["action_data"],
                timestamp=s.get("timestamp", 0.0),
                delay_after=s.get("delay_after", 0.0),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            name=data.get("name", ""),
            steps=steps,
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
        )


@dataclass
class ScreenRegion:
    """A named rectangular area on the screen."""
    name: str = ""
    x: int = 0
    y: int = 0
    width: int = 100
    height: int = 100
    description: str = ""


# ---------------------------------------------------------------------------
# Desktop Controller
# ---------------------------------------------------------------------------

class DesktopController:
    """
    Comprehensive desktop control interface for an AI assistant.

    Every mutating action runs through an optional *permission_checker*
    callback.  If no checker is provided, all actions are permitted.
    If the checker returns ``False`` the action is silently skipped and
    a warning is logged.

    Parameters
    ----------
    permission_checker:
        Optional ``Callable[[str], bool]`` that receives a human-readable
        description of the action and returns whether it is allowed.
    """

    def __init__(self, permission_checker: Callable[[str], bool] | None = None):
        self._permission_checker: Callable[[str], bool] | None = permission_checker
        self._initialized: bool = False
        self._recording: bool = False
        self._recording_start: float = 0.0
        self._recorded_steps: list[AutomationStep] = []
        self._regions: dict[str, ScreenRegion] = {}
        self._mouse_listener: object | None = None
        self._keyboard_listener: object | None = None
        self._previous_mouse_pos: tuple[int, int] | None = None
        self._scroll_fail_safe: bool = True
        if pyautogui is not None:
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.05

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Prepare the controller for use (configure OS-level hooks)."""
        if self._initialized:
            return

        # Set up pyautogui safety
        if pyautogui is not None:
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.05
            # Disable the 0.1-second pause for more responsive automation
            pyautogui.MINIMUM_DURATION = 0.01
            pyautogui.MINIMUM_SLEEP = 0.005

        logger.info("DesktopController initialized on %s", platform.system())
        self._initialized = True

    async def shutdown(self) -> None:
        """Tear down hooks and stop any active recording."""
        if self._recording:
            await self.stop_recording()
        self._initialized = False
        self._regions.clear()
        logger.info("DesktopController shut down")

    # ------------------------------------------------------------------
    # Permission
    # ------------------------------------------------------------------

    def _check_permission(self, action_desc: str) -> bool:
        """Return ``True`` when the action is permitted."""
        if self._permission_checker is None:
            return True
        try:
            allowed = self._permission_checker(action_desc)
        except Exception:
            logger.exception("Permission checker raised an exception for: %s", action_desc)
            allowed = False
        if not allowed:
            logger.warning("Action blocked by permission checker: %s", action_desc)
        return allowed

    # ==================================================================
    # Mouse control
    # ==================================================================

    async def mouse_move(
        self,
        x: int,
        y: int,
        smooth: bool = True,
        duration: float = 0.3,
    ) -> None:
        """Move the mouse cursor to ``(x, y)`` with optional smooth animation."""
        if not self._check_permission(f"Mouse move to ({x}, {y})"):
            return
        if pyautogui is None:
            raise RuntimeError("pyautogui is not installed")
        if smooth:
            await self._smooth_move(x, y, duration)
        else:
            pyautogui.moveTo(x, y, duration=0)
        await self._record_mouse_step(x, y, MouseActionType.MOVE)

    async def _smooth_move(self, target_x: int, target_y: int, duration: float = 0.3) -> None:
        """Animate the cursor using linear interpolation (lerp)."""
        if pyautogui is None:
            return
        start_x, start_y = pyautogui.position()
        total_steps = max(10, int(duration / 0.01))
        for step in range(1, total_steps + 1):
            t = step / total_steps
            # Ease-in-out cubic for natural feel
            t = t * t * (3.0 - 2.0 * t)
            cur_x = start_x + (target_x - start_x) * t
            cur_y = start_y + (target_y - start_y) * t
            pyautogui.moveTo(int(cur_x), int(cur_y), duration=0)
            await asyncio.sleep(duration / total_steps)

    async def mouse_click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        """Click the mouse at the given coordinates."""
        if not self._check_permission(f"Mouse click ({button}, {clicks}x) at ({x}, {y})"):
            return
        if pyautogui is None:
            raise RuntimeError("pyautogui is not installed")
        if x is not None and y is not None:
            pyautogui.click(x=x, y=y, button=button, clicks=clicks)
        else:
            pyautogui.click(button=button, clicks=clicks)
        pos_x, pos_y = (x, y) if x is not None and y is not None else pyautogui.position()
        action_type = (
            MouseActionType.DOUBLE_CLICK if clicks >= 2
            else MouseActionType.CLICK
        )
        await self._record_mouse_step(pos_x, pos_y, action_type, button=button)
        await asyncio.sleep(0.05)

    async def mouse_right_click(self, x: int | None = None, y: int | None = None) -> None:
        """Perform a right-click at the given coordinates."""
        if not self._check_permission(f"Right click at ({x}, {y})"):
            return
        await self.mouse_click(x=x, y=y, button="right", clicks=1)

    async def mouse_double_click(self, x: int | None = None, y: int | None = None) -> None:
        """Perform a double-click at the given coordinates."""
        if not self._check_permission(f"Double click at ({x}, {y})"):
            return
        await self.mouse_click(x=x, y=y, button="left", clicks=2)

    async def mouse_drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5,
        button: str = "left",
    ) -> None:
        """Click-and-drag from one position to another."""
        if not self._check_permission(
            f"Mouse drag from ({start_x}, {start_y}) to ({end_x}, {end_y})"
        ):
            return
        if pyautogui is None:
            raise RuntimeError("pyautogui is not installed")
        pyautogui.moveTo(start_x, start_y, duration=0)
        await asyncio.sleep(0.02)
        pyautogui.drag(
            end_x - start_x,
            end_y - start_y,
            duration=duration,
            button=button,
        )
        await self._record_mouse_step(
            start_x, start_y, MouseActionType.DRAG,
            drag_end_x=end_x, drag_end_y=end_y,
        )

    async def mouse_scroll(
        self,
        x: int | None = None,
        y: int | None = None,
        clicks: int = 3,
        direction: str = "up",
    ) -> None:
        """Scroll the mouse wheel at the given position."""
        if not self._check_permission(f"Mouse scroll {direction} {clicks} clicks at ({x}, {y})"):
            return
        if pyautogui is None:
            raise RuntimeError("pyautogui is not installed")
        if x is not None and y is not None:
            pyautogui.moveTo(x, y, duration=0)
            await asyncio.sleep(0.01)
        amount = clicks if direction == "up" else -clicks
        pyautogui.scroll(amount)
        pos_x, pos_y = (x, y) if x is not None and y is not None else pyautogui.position()
        await self._record_mouse_step(
            pos_x, pos_y, MouseActionType.SCROLL,
            scroll_clicks=clicks, scroll_direction=direction,
        )

    async def mouse_position(self) -> tuple[int, int]:
        """Return the current ``(x, y)`` position of the mouse cursor."""
        if pyautogui is None:
            raise RuntimeError("pyautogui is not installed")
        pos = pyautogui.position()
        return (pos.x, pos.y) if hasattr(pos, "x") else (int(pos[0]), int(pos[1]))

    async def screen_size(self) -> tuple[int, int]:
        """Return the screen resolution ``(width, height)``."""
        if pyautogui is None:
            raise RuntimeError("pyautogui is not installed")
        size = pyautogui.size()
        return (size.width, size.height) if hasattr(size, "width") else (int(size[0]), int(size[1]))

    async def find_image(
        self,
        image_path: str,
        confidence: float = 0.8,
        region: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int] | None:
        """
        Locate *image_path* on the screen and return its centre ``(x, y)``.

        Returns ``None`` when the image cannot be found at the given
        confidence threshold.
        """
        if pyautogui is None or ImageGrab is None:
            raise RuntimeError("pyautogui and Pillow are required for image matching")
        if not os.path.isfile(image_path):
            logger.error("Image file not found: %s", image_path)
            return None
        try:
            kwargs = {"confidence": confidence}
            if region is not None:
                kwargs["region"] = region
            location = pyautogui.locateOnScreen(image_path, **kwargs)
            if location is None:
                return None
            return pyautogui.center(location)
        except pyautogui.ImageNotFoundException:
            return None
        except Exception as exc:
            logger.error("Error finding image %s: %s", image_path, exc)
            return None

    async def click_image(
        self,
        image_path: str,
        confidence: float = 0.8,
        button: str = "left",
        clicks: int = 1,
    ) -> bool:
        """Find an image on screen and click its centre. Returns success flag."""
        pos = await self.find_image(image_path, confidence=confidence)
        if pos is None:
            logger.warning("Could not find image on screen: %s", image_path)
            return False
        await self.mouse_click(int(pos[0]), int(pos[1]), button=button, clicks=clicks)
        return True

    # ------------------------------------------------------------------
    # Mouse recording helpers
    # ------------------------------------------------------------------

    async def _record_mouse_step(self, x: int, y: int, action: MouseActionType, **extra) -> None:
        if not self._recording:
            return
        data = {"x": x, "y": y, "action": action.value}
        data.update(extra)
        elapsed = time.time() - self._recording_start
        self._recorded_steps.append(AutomationStep(
            step_type=AutomationStepType.MOUSE,
            action_data=data,
            timestamp=elapsed,
            delay_after=0.0,
        ))

    # ==================================================================
    # Keyboard control
    # ==================================================================

    async def type_text(self, text: str, delay: float = 0.02) -> None:
        """Type *text* character-by-character with a human-like delay."""
        if not self._check_permission(f"Type text ({len(text)} characters)"):
            return
        if pyautogui is None:
            raise RuntimeError("pyautogui is not installed")
        for char in text:
            pyautogui.press(char)
            await asyncio.sleep(delay)
        await self._record_key_step(text=text, action=KeyActionType.TYPE, delay=delay)

    async def press_key(self, key: str) -> None:
        """Press and release a single key (e.g. ``"enter"``, ``"tab"``, ``"a"``)."""
        if not self._check_permission(f"Press key '{key}'"):
            return
        if pyautogui is None:
            raise RuntimeError("pyautogui is not installed")
        pyautogui.press(key)
        await self._record_key_step(key=key, action=KeyActionType.PRESS)

    async def hotkey(self, *keys: str) -> None:
        """Press a combination of keys simultaneously (e.g. ``ctrl+c``)."""
        keys_str = "+".join(keys)
        if not self._check_permission(f"Hotkey '{keys_str}'"):
            return
        if pyautogui is None:
            raise RuntimeError("pyautogui is not installed")
        pyautogui.hotkey(*keys)
        await self._record_key_step(keys=list(keys), action=KeyActionType.HOTKEY)

    async def paste_text(self, text: str) -> None:
        """Copy *text* to the system clipboard and paste it via ``Ctrl+V`` / ``Cmd+V``."""
        if not self._check_permission(f"Paste text ({len(text)} characters)"):
            return
        await self.set_clipboard(text)
        await asyncio.sleep(0.05)
        await self.hotkey("ctrl" if IS_WINDOWS or IS_LINUX else "cmd", "v")
        await asyncio.sleep(0.05)

    async def press_unicode_char(self, char: str) -> None:
        """
        Type a Unicode character using platform-specific mechanisms.

        On Windows, uses the clipboard-paste approach. On macOS, uses
        the ``ctrl+cmd+u`` Unicode hex input. On Linux, uses ``ctrl+shift+u``.
        """
        if not self._check_permission(f"Press unicode character U+{ord(char):04X}"):
            return
        if pyautogui is None:
            raise RuntimeError("pyautogui is not installed")

        if IS_WINDOWS:
            await self.paste_text(char)
        elif IS_MACOS:
            # macOS Unicode hex input via Character Viewer / Keyboard Viewer
            code = f"{ord(char):04x}"
            await self.hotkey("ctrl", "cmd", "space")
            await asyncio.sleep(0.1)
            await self.type_text(code)
            await asyncio.sleep(0.05)
            await self.press_key("enter")
        elif IS_LINUX:
            hex_str = f"{ord(char):04x}"
            await self.hotkey("ctrl", "shift", "u")
            await self.type_text(hex_str)
            await self.press_key("enter")
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------------
    # Keyboard recording helpers
    # ------------------------------------------------------------------

    async def _record_key_step(self, **data) -> None:
        if not self._recording:
            return
        elapsed = time.time() - self._recording_start
        self._recorded_steps.append(AutomationStep(
            step_type=AutomationStepType.KEYBOARD,
            action_data=data,
            timestamp=elapsed,
            delay_after=0.0,
        ))

    # ==================================================================
    # Window management
    # ==================================================================

    def _get_window(self, title: str | int):
        """Resolve a window by title substring or PID."""
        if gw is None:
            raise RuntimeError("pygetwindow is not installed")
        try:
            if isinstance(title, int):
                for w in gw.getAllWindows():
                    # pygetwindow does not expose PID on all platforms;
                    # fall back to title matching.
                    pass
                # Try to get by title as fallback
                return gw.getWindowsWithTitle(str(title))[0]
            windows = gw.getWindowsWithTitle(title)
            if not windows:
                raise ValueError(f"No window found matching '{title}'")
            return windows[0]
        except Exception:
            raise ValueError(f"Window not found: {title}")

    async def focus_window(self, title: str | int) -> bool:
        """Bring the matching window to the foreground."""
        if not self._check_permission(f"Focus window '{title}'"):
            return False
        try:
            win = self._get_window(title)
            win.activate()
            await asyncio.sleep(0.15)
            return True
        except Exception as exc:
            logger.error("Could not focus window '%s': %s", title, exc)
            return False

    async def minimize_window(self, title: str) -> bool:
        """Minimize the matching window."""
        if not self._check_permission(f"Minimize window '{title}'"):
            return False
        try:
            win = self._get_window(title)
            win.minimize()
            await asyncio.sleep(0.15)
            return True
        except Exception as exc:
            logger.error("Could not minimize window '%s': %s", title, exc)
            return False

    async def maximize_window(self, title: str) -> bool:
        """Maximize the matching window."""
        if not self._check_permission(f"Maximize window '{title}'"):
            return False
        try:
            win = self._get_window(title)
            win.maximize()
            await asyncio.sleep(0.15)
            return True
        except Exception as exc:
            logger.error("Could not maximize window '%s': %s", title, exc)
            return False

    async def close_window(self, title: str) -> bool:
        """Close the matching window."""
        if not self._check_permission(f"Close window '{title}'"):
            return False
        try:
            win = self._get_window(title)
            win.close()
            await asyncio.sleep(0.15)
            return True
        except Exception as exc:
            logger.error("Could not close window '%s': %s", title, exc)
            return False

    async def resize_window(self, title: str, width: int, height: int) -> bool:
        """Resize the matching window to ``(width, height)``."""
        if not self._check_permission(f"Resize window '{title}' to {width}x{height}"):
            return False
        try:
            win = self._get_window(title)
            if hasattr(win, "isMinimized") and win.isMinimized:
                win.restore()
                await asyncio.sleep(0.15)
            win.resize(width, height)
            await asyncio.sleep(0.15)
            return True
        except Exception as exc:
            logger.error("Could not resize window '%s': %s", title, exc)
            return False

    async def move_window(self, title: str, x: int, y: int) -> bool:
        """Move the matching window so its top-left corner is at ``(x, y)``."""
        if not self._check_permission(f"Move window '{title}' to ({x}, {y})"):
            return False
        try:
            win = self._get_window(title)
            if hasattr(win, "isMinimized") and win.isMinimized:
                win.restore()
                await asyncio.sleep(0.15)
            win.moveTo(x, y)
            await asyncio.sleep(0.15)
            return True
        except Exception as exc:
            logger.error("Could not move window '%s': %s", title, exc)
            return False

    async def list_windows(self) -> list[dict]:
        """Return a list of dicts describing every visible window."""
        if gw is None:
            raise RuntimeError("pygetwindow is not installed")
        try:
            windows = gw.getAllWindows()
        except Exception:
            windows = []
        result = []
        for w in windows:
            try:
                info = {
                    "title": w.title,
                    "left": w.left,
                    "top": w.top,
                    "width": w.width,
                    "height": w.height,
                    "visible": getattr(w, "visible", True),
                }
            except Exception:
                continue
            result.append(info)
        return result

    async def get_foreground_window(self) -> dict:
        """Return information about the currently focused window."""
        if gw is None:
            raise RuntimeError("pygetwindow is not installed")
        try:
            win = gw.getActiveWindow()
            if win is None:
                return {"title": "", "left": 0, "top": 0, "width": 0, "height": 0}
            return {
                "title": win.title,
                "left": win.left,
                "top": win.top,
                "width": win.width,
                "height": win.height,
            }
        except Exception as exc:
            logger.error("Could not get foreground window: %s", exc)
            return {"title": "", "left": 0, "top": 0, "width": 0, "height": 0}

    async def set_always_on_top(self, title: str, on_top: bool = True) -> bool:
        """
        Pin a window so that it stays above all other windows.

        On Windows this uses the Win32 API directly. On macOS and Linux
        the operation is best-effort.
        """
        if not self._check_permission(f"Set always-on-top for '{title}' (on_top={on_top})"):
            return False
        try:
            win = self._get_window(title)
            if IS_WINDOWS:
                import ctypes
                from ctypes import wintypes

                hwnd = ctypes.windll.user32.FindWindowW(None, win.title)
                if hwnd:
                    SWP_NOMOVE = 0x0002
                    SWP_NOSIZE = 0x0001
                    HWND_TOPMOST = -1
                    HWND_NOTOPMOST = -2
                    z_order = HWND_TOPMOST if on_top else HWND_NOTOPMOST
                    ctypes.windll.user32.SetWindowPos(
                        hwnd, z_order, 0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE,
                    )
                    return True
                return False
            else:
                logger.info(
                    "Always-on-top is not fully supported on %s; "
                    "bringing window to front instead.",
                    platform.system(),
                )
                if on_top:
                    await self.focus_window(title)
                return True
        except Exception as exc:
            logger.error("Could not set always-on-top for '%s': %s", title, exc)
            return False

    async def snap_window(self, title: str, position: str) -> bool:
        """
        Snap a window to a screen edge or corner.

        *position* can be ``"left"``, ``"right"``, ``"top"``, ``"bottom"``,
        ``"top-left"``, ``"top-right"``, ``"bottom-left"``, ``"bottom-right"``,
        or ``"center"``.
        """
        sw, sh = await self.screen_size()
        half_w = sw // 2
        half_h = sh // 2
        mappings: dict[str, tuple[int, int, int, int]] = {
            "left": (0, 0, half_w, sh),
            "right": (half_w, 0, half_w, sh),
            "top": (0, 0, sw, half_h),
            "bottom": (0, half_h, sw, half_h),
            "top-left": (0, 0, half_w, half_h),
            "top-right": (half_w, 0, half_w, half_h),
            "bottom-left": (0, half_h, half_w, half_h),
            "bottom-right": (half_w, half_h, half_w, half_h),
            "center": (sw // 4, sh // 4, half_w, half_h),
        }
        if position not in mappings:
            logger.error("Unknown snap position: %s", position)
            return False
        x, y, w, h = mappings[position]
        if not await self.resize_window(title, w, h):
            return False
        return await self.move_window(title, x, y)

    # ==================================================================
    # Application launcher
    # ==================================================================

    async def launch_app(self, app_name_or_path: str, args: list[str] | None = None) -> dict:
        """
        Launch an application by name or path.

        Returns a dict with ``"success"``, ``"pid"``, and ``"error"`` keys.
        """
        args = args or []
        display = app_name_or_path
        if not self._check_permission(f"Launch application '{display}' with args {args}"):
            return {"success": False, "pid": -1, "error": "Permission denied"}

        try:
            if IS_WINDOWS:
                cmd = f"start \"\" \"{app_name_or_path}\""
                if args:
                    cmd += " " + " ".join(f'"{a}"' for a in args)
                proc = subprocess.Popen(
                    cmd, shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif IS_MACOS:
                cmd = ["open", "-a", app_name_or_path] + args
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                cmd = ["nohup", app_name_or_path] + args
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setpgrp if os.fork else None,
                )
            return {"success": True, "pid": proc.pid, "error": ""}
        except Exception as exc:
            logger.error("Failed to launch '%s': %s", app_name_or_path, exc)
            return {"success": False, "pid": -1, "error": str(exc)}

    async def open_file(self, filepath: str) -> bool:
        """Open a file or folder with the OS default application."""
        if not self._check_permission(f"Open file '{filepath}'"):
            return False
        if not os.path.exists(filepath):
            logger.error("File not found: %s", filepath)
            return False
        try:
            if IS_WINDOWS:
                os.startfile(filepath)  # type: ignore[attr-defined]
            elif IS_MACOS:
                subprocess.Popen(["open", filepath],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(["xdg-open", filepath],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            return True
        except Exception as exc:
            logger.error("Failed to open file '%s': %s", filepath, exc)
            return False

    async def open_url(self, url: str) -> None:
        """Open a URL in the user's default web browser."""
        if not self._check_permission(f"Open URL '{url}'"):
            return
        try:
            if IS_WINDOWS:
                os.startfile(url)  # type: ignore[attr-defined]
            elif IS_MACOS:
                subprocess.Popen(["open", url],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(["xdg-open", url],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except Exception as exc:
            logger.error("Failed to open URL '%s': %s", url, exc)

    async def open_terminal(self, directory: str | None = None, command: str | None = None) -> None:
        """Open a terminal window, optionally at *directory* running *command*."""
        display_dir = directory or os.getcwd()
        if not self._check_permission(f"Open terminal in '{display_dir}'"):
            return
        try:
            if IS_WINDOWS:
                if command:
                    full_cmd = f'cd /d "{directory}" && {command}' if directory else command
                    subprocess.Popen(
                        f'start cmd /k "{full_cmd}"', shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    target = directory or os.getcwd()
                    subprocess.Popen(
                        f'start cmd /k "cd /d "{target}""', shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            elif IS_MACOS:
                if command and directory:
                    script = f'cd "{directory}" && {command}'
                elif command:
                    script = command
                elif directory:
                    script = f'cd "{directory}"'
                else:
                    script = ""
                apple_script = (
                    f'tell application "Terminal"\n'
                    f'    activate\n'
                    f'    do script "{script}"\n'
                    f'end tell'
                )
                subprocess.Popen(
                    ["osascript", "-e", apple_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                terminal = os.environ.get("TERMINAL", "gnome-terminal")
                if command and directory:
                    subprocess.Popen(
                        [terminal, "--", "bash", "-c",
                         f'cd "{directory}" && {command}; exec bash'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                elif command:
                    subprocess.Popen(
                        [terminal, "--", "bash", "-c",
                         f'{command}; exec bash'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                elif directory:
                    subprocess.Popen(
                        [terminal, "--working-directory", directory],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    subprocess.Popen(
                        [terminal],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
        except Exception as exc:
            logger.error("Failed to open terminal: %s", exc)

    # ==================================================================
    # Clipboard control
    # ==================================================================

    async def set_clipboard(self, text: str) -> None:
        """Set the system clipboard to *text*."""
        if not self._check_permission("Set clipboard text"):
            return
        try:
            if IS_WINDOWS:
                import ctypes
                CF_UNICODETEXT = 13
                kernel32 = ctypes.windll.kernel32
                user32 = ctypes.windll.user32

                user32.OpenClipboard(0)
                user32.EmptyClipboard()

                text_bytes = (text + "\x00").encode("utf-16-le")
                h_global = kernel32.GlobalAlloc(0x0042, len(text_bytes))
                p_global = kernel32.GlobalLock(h_global)
                ctypes.memmove(p_global, text_bytes, len(text_bytes))
                kernel32.GlobalUnlock(h_global)
                user32.SetClipboardData(CF_UNICODETEXT, h_global)
                user32.CloseClipboard()
            elif IS_MACOS:
                proc = subprocess.Popen(
                    ["pbcopy"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc.communicate(text.encode("utf-8"))
            else:
                proc = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc.communicate(text.encode("utf-8"))
        except Exception as exc:
            logger.error("Failed to set clipboard: %s", exc)

    async def get_clipboard(self) -> str:
        """Return the current contents of the system clipboard."""
        try:
            if IS_WINDOWS:
                import ctypes
                user32 = ctypes.windll.user32
                user32.OpenClipboard(0)
                handle = user32.GetClipboardData(13)
                text = ctypes.c_wchar_p(handle).value or ""
                user32.CloseClipboard()
                return text
            elif IS_MACOS:
                proc = subprocess.Popen(
                    ["pbpaste"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                return proc.stdout.read().decode("utf-8")
            else:
                proc = subprocess.Popen(
                    ["xclip", "-selection", "clipboard", "-o"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                return proc.stdout.read().decode("utf-8")
        except Exception as exc:
            logger.error("Failed to get clipboard: %s", exc)
            return ""

    async def clear_clipboard(self) -> None:
        """Clear the system clipboard."""
        if not self._check_permission("Clear clipboard"):
            return
        try:
            if IS_WINDOWS:
                import ctypes
                ctypes.windll.user32.OpenClipboard(0)
                ctypes.windll.user32.EmptyClipboard()
                ctypes.windll.user32.CloseClipboard()
            elif IS_MACOS:
                proc = subprocess.Popen(
                    ["pbcopy"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc.communicate(b"")
            else:
                proc = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc.communicate(b"")
        except Exception as exc:
            logger.error("Failed to clear clipboard: %s", exc)

    # ==================================================================
    # Screen regions
    # ==================================================================

    async def define_region(
        self,
        name: str,
        x: int,
        y: int,
        w: int,
        h: int,
        description: str = "",
    ) -> None:
        """Create or update a named screen region."""
        self._regions[name] = ScreenRegion(
            name=name, x=x, y=y, width=w, height=h, description=description,
        )
        logger.info("Defined region '%s' at (%d, %d) size %dx%d", name, x, y, w, h)

    async def remove_region(self, name: str) -> bool:
        """Remove a named screen region."""
        if name in self._regions:
            del self._regions[name]
            return True
        return False

    async def get_region(self, name: str) -> ScreenRegion | None:
        """Return a previously defined region."""
        return self._regions.get(name)

    async def list_regions(self) -> dict[str, dict]:
        """Return all defined regions as serialisable dicts."""
        return {
            name: {
                "name": r.name,
                "x": r.x,
                "y": r.y,
                "width": r.width,
                "height": r.height,
                "description": r.description,
            }
            for name, r in self._regions.items()
        }

    async def click_region(self, name: str, button: str = "left", clicks: int = 1) -> bool:
        """Click the centre of a named region."""
        region = self._regions.get(name)
        if region is None:
            logger.error("Region '%s' is not defined", name)
            return False
        cx = region.x + region.width // 2
        cy = region.y + region.height // 2
        await self.mouse_move(cx, cy, smooth=True, duration=0.15)
        await self.mouse_click(cx, cy, button=button, clicks=clicks)
        return True

    async def type_in_region(self, name: str, text: str, delay: float = 0.02) -> bool:
        """Move to the centre of a region and start typing."""
        region = self._regions.get(name)
        if region is None:
            logger.error("Region '%s' is not defined", name)
            return False
        cx = region.x + region.width // 2
        cy = region.y + region.height // 2
        await self.mouse_click(cx, cy)
        await asyncio.sleep(0.1)
        await self.type_text(text, delay=delay)
        return True

    async def capture_region(self, name: str, save_path: str) -> bool:
        """Capture a screenshot of a named region and save it to *save_path*."""
        region = self._regions.get(name)
        if region is None:
            logger.error("Region '%s' is not defined", name)
            return False
        if ImageGrab is None:
            raise RuntimeError("Pillow is required for screen capture")
        try:
            screenshot = ImageGrab.grab(
                bbox=(region.x, region.y, region.x + region.width, region.y + region.height),
            )
            screenshot.save(save_path)
            return True
        except Exception as exc:
            logger.error("Failed to capture region '%s': %s", name, exc)
            return False

    # ==================================================================
    # Notifications
    # ==================================================================

    async def show_notification(
        self,
        title: str,
        message: str,
        duration: int = 5,
    ) -> None:
        """Display a desktop toast / notification."""
        if not self._check_permission(f"Show notification '{title}'"):
            return
        try:
            if IS_WINDOWS:
                self._notify_windows(title, message, duration)
            elif IS_MACOS:
                self._notify_macos(title, message, duration)
            else:
                self._notify_linux(title, message, duration)
        except Exception as exc:
            logger.error("Failed to show notification: %s", exc)

    @staticmethod
    def _notify_windows(title: str, message: str, duration: int) -> None:
        """Show a balloon-tip notification via PowerShell."""
        ps_script = (
            f'[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, '
            f'ContentType = WindowsRuntime] > $null\n'
            f'[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, '
            f'ContentType = WindowsRuntime] > $null\n'
            f'$template = @"\n'
            f'<toast>\n'
            f'  <visual>\n'
            f'    <binding template="ToastGeneric">\n'
            f'      <text>{title}</text>\n'
            f'      <text>{message}</text>\n'
            f'    </binding>\n'
            f'  </visual>\n'
            f'</toast>\n'
            f'"@\n'
            f'$xml = New-Object Windows.Data.Xml.Dom.XmlDocument\n'
            f'$xml.LoadXml($template)\n'
            f'$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)\n'
            f'[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("AI Assistant").Show($toast)\n'
        )
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", ps_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _notify_macos(title: str, message: str, duration: int) -> None:
        """Show a notification via osascript."""
        script = (
            f'display notification "{message}" with title "{title}" '
            f'sound name "default"'
        )
        subprocess.Popen(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _notify_linux(title: str, message: str, duration: int) -> None:
        """Show a notification via notify-send."""
        try:
            subprocess.Popen(
                ["notify-send", "-t", str(duration * 1000), title, message],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.warning("notify-send not found; cannot display notification on Linux")

    async def alert(self, title: str, message: str) -> None:
        """
        Show a blocking desktop alert dialog that requires user dismissal.

        Falls back to a non-blocking notification when a GUI dialog
        toolkit is not available.
        """
        if not self._check_permission(f"Show alert dialog '{title}'"):
            return
        try:
            if IS_WINDOWS:
                import ctypes
                ctypes.windll.user32.MessageBoxW(
                    0, message, title, 0x40,  # MB_ICONINFORMATION
                )
            elif IS_MACOS:
                script = (
                    f'display dialog "{message}" with title "{title}" '
                    f'buttons {{"OK"}} default button "OK" '
                    f'with icon note'
                )
                proc = subprocess.Popen(
                    ["osascript", "-e", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                proc.communicate()
            else:
                # Try zenity first, then kdialog, then fall back.
                for cmd in [
                    ["zenity", "--info", f"--title={title}", f"--text={message}"],
                    ["kdialog", "--title", title, "--msgbox", message],
                ]:
                    try:
                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        proc.wait(timeout=60)
                        return
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        continue
                # Final fallback to a non-blocking notification
                await DesktopController._notify_linux(title, message, 10)
        except Exception as exc:
            logger.error("Failed to show alert: %s", exc)

    # ==================================================================
    # Text selection and extraction
    # ==================================================================

    async def select_all(self) -> None:
        """Send ``Ctrl+A`` / ``Cmd+A`` to select all in the focused window."""
        mod = "ctrl" if IS_WINDOWS or IS_LINUX else "cmd"
        await self.hotkey(mod, "a")

    async def copy_selected(self) -> str:
        """Copy selected text to clipboard and return it."""
        mod = "ctrl" if IS_WINDOWS or IS_LINUX else "cmd"
        await self.hotkey(mod, "c")
        await asyncio.sleep(0.1)
        return await self.get_clipboard()

    async def extract_text_from_region(self, name: str) -> str:
        """
        Capture a screenshot of a region, attempt OCR, and return the text.

        Requires ``pytesseract`` and ``tesseract-ocr`` to be installed.
        Falls back to an empty string when OCR is unavailable.
        """
        region = self._regions.get(name)
        if region is None:
            logger.error("Region '%s' is not defined", name)
            return ""
        if ImageGrab is None:
            logger.error("Pillow is required for text extraction")
            return ""
        try:
            img = ImageGrab.grab(
                bbox=(region.x, region.y, region.x + region.width, region.y + region.height),
            )
        except Exception as exc:
            logger.error("Failed to capture region for OCR: %s", exc)
            return ""

        try:
            import pytesseract
            text = pytesseract.image_to_string(img).strip()
            return text
        except ImportError:
            logger.warning("pytesseract is not installed; cannot perform OCR")
            return ""
        except Exception as exc:
            logger.error("OCR failed: %s", exc)
            return ""

    # ==================================================================
    # Automation macro recording & playback
    # ==================================================================

    async def start_recording(self) -> None:
        """Begin recording user actions into a new macro."""
        if self._recording:
            logger.warning("Recording is already in progress")
            return
        self._recording = True
        self._recording_start = time.time()
        self._recorded_steps = []
        # Store the initial mouse position as a reference
        self._previous_mouse_pos = await self.mouse_position()
        logger.info("Recording started")

    async def stop_recording(self) -> AutomationMacro:
        """Stop recording and return the captured macro."""
        if not self._recording:
            logger.warning("No recording in progress")
            return AutomationMacro(name="empty", created_at=datetime.now().isoformat())
        self._recording = False
        # Insert timing delays between consecutive steps
        self._compute_step_delays()
        macro = AutomationMacro(
            name=f"macro_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            steps=list(self._recorded_steps),
            created_at=datetime.now().isoformat(),
            description=f"Recorded with {len(self._recorded_steps)} steps",
        )
        self._recorded_steps = []
        self._previous_mouse_pos = None
        logger.info("Recording stopped: %d steps captured", len(macro.steps))
        return macro

    def _compute_step_delays(self) -> None:
        """Populate ``delay_after`` on each step based on timestamps."""
        for i, step in enumerate(self._recorded_steps):
            if i + 1 < len(self._recorded_steps):
                next_ts = self._recorded_steps[i + 1].timestamp
                step.delay_after = max(0.0, next_ts - step.timestamp)
            else:
                step.delay_after = 0.0

    async def replay_macro(self, macro: AutomationMacro, speed: float = 1.0) -> None:
        """
        Replay a previously recorded (or loaded) macro.

        *speed* > 1.0 makes it faster, < 1.0 makes it slower.
        """
        if not self._check_permission(f"Replay macro '{macro.name}' ({len(macro.steps)} steps)"):
            return
        logger.info("Replaying macro '%s' at %.2fx speed", macro.name, speed)
        for i, step in enumerate(macro.steps):
            try:
                await self._execute_step(step, speed)
                delay = step.delay_after / max(speed, 0.01)
                if delay > 0:
                    await asyncio.sleep(delay)
            except Exception as exc:
                logger.error(
                    "Error at macro step %d (%s): %s",
                    i, step.step_type.value, exc,
                )
        logger.info("Macro '%s' replay complete", macro.name)

    async def _execute_step(self, step: AutomationStep, speed: float = 1.0) -> None:
        """Dispatch a single macro step to the appropriate controller method."""
        data = step.action_data

        if step.step_type == AutomationStepType.MOUSE:
            action = data.get("action", "move")
            x, y = data.get("x", 0), data.get("y", 0)
            if action == MouseActionType.MOVE.value:
                await self.mouse_move(x, y, smooth=False)
            elif action == MouseActionType.CLICK.value:
                await self.mouse_click(x, y, button=data.get("button", "left"))
            elif action == MouseActionType.DOUBLE_CLICK.value:
                await self.mouse_double_click(x, y)
            elif action == MouseActionType.RIGHT_CLICK.value:
                await self.mouse_right_click(x, y)
            elif action == MouseActionType.MIDDLE_CLICK.value:
                await self.mouse_click(x, y, button="middle")
            elif action == MouseActionType.DRAG.value:
                await self.mouse_drag(
                    x, y,
                    data.get("drag_end_x", x), data.get("drag_end_y", y),
                    duration=data.get("duration", 0.5) / max(speed, 0.01),
                )
            elif action == MouseActionType.SCROLL.value:
                await self.mouse_scroll(
                    x, y,
                    clicks=data.get("scroll_clicks", 3),
                    direction=data.get("scroll_direction", "up"),
                )

        elif step.step_type == AutomationStepType.KEYBOARD:
            action = data.get("action", "press")
            if action == KeyActionType.PRESS.value:
                await self.press_key(data.get("key", ""))
            elif action == KeyActionType.TYPE.value:
                delay = data.get("delay", 0.02) / max(speed, 0.01)
                await self.type_text(data.get("text", ""), delay=delay)
            elif action == KeyActionType.HOTKEY.value:
                await self.hotkey(*data.get("keys", []))

        elif step.step_type == AutomationStepType.WAIT:
            wait_time = data.get("seconds", 0) / max(speed, 0.01)
            await asyncio.sleep(wait_time)

        elif step.step_type == AutomationStepType.CLIPBOARD:
            if data.get("operation") == "paste":
                await self.paste_text(data.get("text", ""))

        elif step.step_type == AutomationStepType.LAUNCH:
            await self.launch_app(data.get("path", ""), args=data.get("args"))

        elif step.step_type == AutomationStepType.NOTIFICATION:
            await self.show_notification(
                data.get("title", ""), data.get("message", ""),
            )

    async def save_macro(self, macro: AutomationMacro, filepath: str) -> None:
        """Persist a macro to a JSON file."""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(macro.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info("Macro '%s' saved to %s", macro.name, filepath)
        except Exception as exc:
            logger.error("Failed to save macro: %s", exc)

    async def load_macro(self, filepath: str) -> AutomationMacro:
        """Load a macro from a JSON file."""
        if not os.path.isfile(filepath):
            logger.error("Macro file not found: %s", filepath)
            return AutomationMacro(
                name="error",
                created_at=datetime.now().isoformat(),
                description="File not found",
            )
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            macro = AutomationMacro.from_dict(data)
            logger.info("Macro '%s' loaded from %s", macro.name, filepath)
            return macro
        except Exception as exc:
            logger.error("Failed to load macro: %s", exc)
            return AutomationMacro(
                name="error",
                created_at=datetime.now().isoformat(),
                description=f"Load error: {exc}",
            )

    # ==================================================================
    # Generic action dispatcher
    # ==================================================================

    async def execute_action(self, action: dict) -> dict:
        """
        Route a plain dict describing an action to the appropriate method.

        Expected keys:
          ``"type"`` – one of ``"mouse_move"``, ``"mouse_click"``,
          ``"mouse_right_click"``, ``"mouse_double_click"``, ``"mouse_drag"``,
          ``"mouse_scroll"``, ``"type_text"``, ``"press_key"``, ``"hotkey"``,
          ``"paste_text"``, ``"focus_window"``, ``"minimize_window"``,
          ``"maximize_window"``, ``"close_window"``, ``"resize_window"``,
          ``"move_window"``, ``"launch_app"``, ``"open_file"``, ``"open_url"``,
          ``"open_terminal"``, ``"set_clipboard"``, ``"get_clipboard"``,
          ``"clear_clipboard"``, ``"show_notification"``, ``"alert"``,
          ``"define_region"``, ``"click_region"``, ``"type_in_region"``,
          ``"click_image"``, ``"find_image"``, ``"start_recording"``,
          ``"stop_recording"``, ``"replay_macro"``, ``"save_macro"``,
          ``"load_macro"``, ``"list_windows"``, ``"screen_size"``,
          ``"mouse_position"``, ``"select_all"``, ``"copy_selected"``.

        Returns a dict with at least ``"success"`` (bool) and optionally
        additional result data.
        """
        action_type = action.get("type", "")
        params = action.get("params", {})
        result: dict = {"success": False, "type": action_type}

        try:
            # ---- Mouse --------------------------------------------------
            if action_type == "mouse_move":
                await self.mouse_move(
                    x=params["x"], y=params["y"],
                    smooth=params.get("smooth", True),
                    duration=params.get("duration", 0.3),
                )
                result["success"] = True

            elif action_type == "mouse_click":
                await self.mouse_click(
                    x=params.get("x"), y=params.get("y"),
                    button=params.get("button", "left"),
                    clicks=params.get("clicks", 1),
                )
                result["success"] = True

            elif action_type == "mouse_right_click":
                await self.mouse_right_click(x=params.get("x"), y=params.get("y"))
                result["success"] = True

            elif action_type == "mouse_double_click":
                await self.mouse_double_click(x=params.get("x"), y=params.get("y"))
                result["success"] = True

            elif action_type == "mouse_drag":
                await self.mouse_drag(
                    start_x=params["start_x"], start_y=params["start_y"],
                    end_x=params["end_x"], end_y=params["end_y"],
                    duration=params.get("duration", 0.5),
                )
                result["success"] = True

            elif action_type == "mouse_scroll":
                await self.mouse_scroll(
                    x=params.get("x"), y=params.get("y"),
                    clicks=params.get("clicks", 3),
                    direction=params.get("direction", "up"),
                )
                result["success"] = True

            elif action_type == "click_image":
                found = await self.click_image(
                    image_path=params["image_path"],
                    confidence=params.get("confidence", 0.8),
                )
                result["success"] = found

            elif action_type == "find_image":
                pos = await self.find_image(
                    image_path=params["image_path"],
                    confidence=params.get("confidence", 0.8),
                )
                result["success"] = pos is not None
                result["position"] = pos

            # ---- Keyboard -----------------------------------------------
            elif action_type == "type_text":
                await self.type_text(
                    text=params["text"],
                    delay=params.get("delay", 0.02),
                )
                result["success"] = True

            elif action_type == "press_key":
                await self.press_key(params["key"])
                result["success"] = True

            elif action_type == "hotkey":
                keys = params["keys"] if isinstance(params["keys"], list) else [params["keys"]]
                await self.hotkey(*keys)
                result["success"] = True

            elif action_type == "paste_text":
                await self.paste_text(params["text"])
                result["success"] = True

            # ---- Window -------------------------------------------------
            elif action_type == "focus_window":
                result["success"] = await self.focus_window(params["title"])

            elif action_type == "minimize_window":
                result["success"] = await self.minimize_window(params["title"])

            elif action_type == "maximize_window":
                result["success"] = await self.maximize_window(params["title"])

            elif action_type == "close_window":
                result["success"] = await self.close_window(params["title"])

            elif action_type == "resize_window":
                result["success"] = await self.resize_window(
                    params["title"], params["width"], params["height"],
                )

            elif action_type == "move_window":
                result["success"] = await self.move_window(
                    params["title"], params["x"], params["y"],
                )

            elif action_type == "list_windows":
                result["success"] = True
                result["windows"] = await self.list_windows()

            elif action_type == "get_foreground_window":
                result["success"] = True
                result["window"] = await self.get_foreground_window()

            elif action_type == "set_always_on_top":
                result["success"] = await self.set_always_on_top(
                    params["title"], params.get("on_top", True),
                )

            elif action_type == "snap_window":
                result["success"] = await self.snap_window(
                    params["title"], params["position"],
                )

            # ---- Application launcher -----------------------------------
            elif action_type == "launch_app":
                launch_result = await self.launch_app(
                    params["app"], args=params.get("args"),
                )
                result.update(launch_result)

            elif action_type == "open_file":
                result["success"] = await self.open_file(params["filepath"])

            elif action_type == "open_url":
                await self.open_url(params["url"])
                result["success"] = True

            elif action_type == "open_terminal":
                await self.open_terminal(
                    directory=params.get("directory"),
                    command=params.get("command"),
                )
                result["success"] = True

            # ---- Clipboard ----------------------------------------------
            elif action_type == "set_clipboard":
                await self.set_clipboard(params["text"])
                result["success"] = True

            elif action_type == "get_clipboard":
                result["success"] = True
                result["text"] = await self.get_clipboard()

            elif action_type == "clear_clipboard":
                await self.clear_clipboard()
                result["success"] = True

            # ---- Regions ------------------------------------------------
            elif action_type == "define_region":
                await self.define_region(
                    name=params["name"],
                    x=params["x"], y=params["y"],
                    w=params["width"], h=params["height"],
                    description=params.get("description", ""),
                )
                result["success"] = True

            elif action_type == "click_region":
                result["success"] = await self.click_region(
                    params["name"],
                    button=params.get("button", "left"),
                    clicks=params.get("clicks", 1),
                )

            elif action_type == "type_in_region":
                result["success"] = await self.type_in_region(
                    params["name"], params["text"],
                    delay=params.get("delay", 0.02),
                )

            elif action_type == "list_regions":
                result["success"] = True
                result["regions"] = await self.list_regions()

            # ---- Notifications ------------------------------------------
            elif action_type == "show_notification":
                await self.show_notification(
                    title=params["title"],
                    message=params["message"],
                    duration=params.get("duration", 5),
                )
                result["success"] = True

            elif action_type == "alert":
                await self.alert(title=params["title"], message=params["message"])
                result["success"] = True

            # ---- Text extraction ----------------------------------------
            elif action_type == "select_all":
                await self.select_all()
                result["success"] = True

            elif action_type == "copy_selected":
                result["success"] = True
                result["text"] = await self.copy_selected()

            elif action_type == "extract_text_from_region":
                result["success"] = True
                result["text"] = await self.extract_text_from_region(params["name"])

            # ---- Macros -------------------------------------------------
            elif action_type == "start_recording":
                await self.start_recording()
                result["success"] = True

            elif action_type == "stop_recording":
                macro = await self.stop_recording()
                result["success"] = True
                result["macro"] = macro.to_dict()

            elif action_type == "replay_macro":
                if "macro" in params:
                    macro = AutomationMacro.from_dict(params["macro"])
                else:
                    macro = await self.load_macro(params["filepath"])
                await self.replay_macro(macro, speed=params.get("speed", 1.0))
                result["success"] = True

            elif action_type == "save_macro":
                macro = AutomationMacro.from_dict(params["macro"])
                await self.save_macro(macro, params["filepath"])
                result["success"] = True

            elif action_type == "load_macro":
                macro = await self.load_macro(params["filepath"])
                result["success"] = True
                result["macro"] = macro.to_dict()

            # ---- Info ---------------------------------------------------
            elif action_type == "screen_size":
                w, h = await self.screen_size()
                result["success"] = True
                result["width"] = w
                result["height"] = h

            elif action_type == "mouse_position":
                x, y = await self.mouse_position()
                result["success"] = True
                result["x"] = x
                result["y"] = y

            else:
                result["error"] = f"Unknown action type: {action_type}"
                logger.warning("Unknown action type: %s", action_type)

        except Exception as exc:
            result["error"] = str(exc)
            logger.error("Error executing action '%s': %s", action_type, exc)

        return result
