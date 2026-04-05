"""
CLI event callbacks for Hermes TUI.

Provides pre/post message callbacks, tool call callbacks,
error handling callbacks, notification callbacks, and
sound/notification on completion.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import subprocess
    HAS_SUBPROCESS = True
except ImportError:
    HAS_SUBPROCESS = False


class CallbackEvent:
    """Represents a callback event."""

    def __init__(self, event_type: str, data: Any = None, timestamp: Optional[float] = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()
        self.handled = False

    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "handled": self.handled,
        }


class CallbackManager:
    """Manages event callbacks for the Hermes CLI."""

    EVENT_TYPES = [
        "pre_message",
        "post_message",
        "pre_tool_call",
        "post_tool_call",
        "error",
        "notification",
        "completion",
        "interrupt",
        "session_save",
        "session_load",
        "config_change",
        "model_change",
        "provider_change",
        "theme_change",
        "profile_change",
    ]

    def __init__(self, sound_enabled: bool = False, notification_enabled: bool = False):
        self._callbacks: Dict[str, List[Callable]] = {e: [] for e in self.EVENT_TYPES}
        self.sound_enabled = sound_enabled
        self.notification_enabled = notification_enabled
        self._event_log: List[CallbackEvent] = []
        self._max_log_size = 1000

    def register(self, event_type: str, callback: Callable) -> None:
        """Register a callback for an event type."""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def unregister(self, event_type: str, callback: Callable) -> bool:
        """Unregister a callback."""
        if event_type in self._callbacks:
            try:
                self._callbacks[event_type].remove(callback)
                return True
            except ValueError:
                pass
        return False

    def emit(self, event_type: str, data: Any = None) -> CallbackEvent:
        """Emit an event and trigger all registered callbacks."""
        event = CallbackEvent(event_type, data)
        self._event_log.append(event)

        # Trim log if too large
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size:]

        callbacks = self._callbacks.get(event_type, [])
        for callback in callbacks:
            try:
                result = callback(event)
                if result is False:
                    event.handled = True
            except Exception as e:
                # Silently handle callback errors
                pass

        return event

    def on(self, event_type: str) -> Callable:
        """Decorator to register a callback."""
        def decorator(func: Callable) -> Callable:
            self.register(event_type, func)
            return func
        return decorator

    def get_event_log(self, event_type: Optional[str] = None, limit: int = 50) -> List[CallbackEvent]:
        """Get event log, optionally filtered by type."""
        events = self._event_log
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def clear_log(self) -> None:
        """Clear event log."""
        self._event_log.clear()


# ──────────────────────────────────────────────
# Built-in callback implementations
# ──────────────────────────────────────────────

class StandardCallbacks:
    """Collection of standard built-in callbacks."""

    def __init__(self, manager: CallbackManager):
        self.manager = manager
        self._register_defaults()

    def _register_defaults(self):
        """Register all default callbacks."""
        self.manager.register("completion", self.on_completion)
        self.manager.register("error", self.on_error)
        self.manager.register("notification", self.on_notification)
        self.manager.register("pre_message", self.on_pre_message)
        self.manager.register("post_message", self.on_post_message)
        self.manager.register("interrupt", self.on_interrupt)
        self.manager.register("config_change", self.on_config_change)

    def on_completion(self, event: CallbackEvent) -> None:
        """Handle completion event - play sound or show notification."""
        if self.manager.sound_enabled:
            self._play_sound("complete")

        if self.manager.notification_enabled:
            self._show_notification("Hermes CLI", "Response completed")

    def on_error(self, event: CallbackEvent) -> None:
        """Handle error event."""
        if self.manager.sound_enabled:
            self._play_sound("error")

    def on_notification(self, event: CallbackEvent) -> None:
        """Handle notification event."""
        if self.manager.notification_enabled:
            data = event.data or {}
            title = data.get("title", "Hermes CLI")
            message = data.get("message", "")
            self._show_notification(title, message)

    def on_pre_message(self, event: CallbackEvent) -> None:
        """Called before sending a message to the AI."""
        # Could be used for logging, analytics, etc.
        pass

    def on_post_message(self, event: CallbackEvent) -> None:
        """Called after receiving a response from the AI."""
        data = event.data or {}
        if data.get("token_count"):
            tokens = data["token_count"]
            if tokens.get("total", 0) > 50000:
                self.manager.emit("notification", {
                    "title": "Token Usage Warning",
                    "message": f"Conversation uses {tokens['total']:,} tokens",
                })

    def on_interrupt(self, event: CallbackEvent) -> None:
        """Handle interrupt (Ctrl+C)."""
        pass

    def on_config_change(self, event: CallbackEvent) -> None:
        """Handle configuration changes."""
        pass

    @staticmethod
    def _play_sound(sound_type: str) -> None:
        """Play a system sound."""
        if not HAS_SUBPROCESS:
            return

        sounds = {
            "complete": "complete",
            "error": "error",
            "notification": "notification",
        }

        # Try different sound approaches based on platform
        system = sys.platform
        try:
            if system == "darwin":
                # macOS
                if sound_type == "error":
                    subprocess.run(["afplay", "/System/Library/Sounds/Basso.aiff"],
                                   capture_output=True, timeout=2)
                else:
                    subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"],
                                   capture_output=True, timeout=2)
            elif system == "linux":
                # Linux - try paplay or aplay
                subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"],
                               capture_output=True, timeout=2)
            elif system == "win32":
                # Windows
                import winsound
                if sound_type == "error":
                    winsound.Beep(440, 200)
                else:
                    winsound.Beep(880, 100)
        except Exception:
            pass

    @staticmethod
    def _show_notification(title: str, message: str) -> None:
        """Show a desktop notification."""
        if not HAS_SUBPROCESS:
            return

        system = sys.platform
        try:
            if system == "darwin":
                cmd = [
                    "osascript", "-e",
                    f'display notification "{message}" with title "{title}"'
                ]
                subprocess.run(cmd, capture_output=True, timeout=5)
            elif system == "linux":
                cmd = ["notify-send", title, message]
                subprocess.run(cmd, capture_output=True, timeout=5)
            elif system == "win32":
                from ctypes import windll
                windll.user32.MessageBoxW(0, message, title, 0x40)
        except Exception:
            pass


class ProgressCallback:
    """Callback for progress tracking during long operations."""

    def __init__(self, total: int = 0, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, current: int, description: str = ""):
        """Update progress."""
        self.current = current
        if description:
            self.description = description

    def get_progress_str(self) -> str:
        """Get formatted progress string."""
        if self.total == 0:
            return f"  {self.description} ({self.current})"

        percent = min(100, (self.current / self.total) * 100)
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = self._format_time(eta)
        else:
            eta_str = "?"

        bar_width = 30
        filled = int(bar_width * percent / 100)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

        return f"  {self.description}\n  [{bar}] {percent:.1f}% ETA: {eta_str}"

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
        else:
            return f"{seconds / 3600:.1f}h"
