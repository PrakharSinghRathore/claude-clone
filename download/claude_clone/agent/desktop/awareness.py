"""
Real-Time PC Awareness Module for AI Desktop Assistant.

Provides comprehensive system monitoring including CPU, memory, disk, GPU,
battery, network, processes, windows, clipboard, screen capture with OCR,
file system watching, and an event-driven architecture for change detection.

All public methods are async. Heavy I/O is offloaded to thread executors.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports – handled gracefully when missing
# ---------------------------------------------------------------------------

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    # Provide stubs so the module can still be imported without watchdog.
    Observer = object  # type: ignore[assignment,misc]
    class _DummyFSEventHandler:  # type: ignore[no-redef]
        pass
    class _DummyFSEvent:  # type: ignore[no-redef]
        event_type: str = ""
        src_path: str = ""
        dest_path: str = ""
        is_directory: bool = False
    FileSystemEventHandler = _DummyFSEventHandler  # type: ignore[misc,assignment]
    FileSystemEvent = _DummyFSEvent  # type: ignore[misc,assignment]


# ═══════════════════════════════════════════════════════════════════════════
# Data classes & enumerations
# ═══════════════════════════════════════════════════════════════════════════


class DesktopEventType(Enum):
    """Types of desktop events that can be emitted."""

    WINDOW_FOCUS_CHANGED = "window_focus_changed"
    CLIPBOARD_CHANGED = "clipboard_changed"
    PROCESS_STARTED = "process_started"
    PROCESS_STOPPED = "process_stopped"
    DISK_LOW = "disk_low"
    MEMORY_HIGH = "memory_high"
    CPU_HIGH = "cpu_high"
    NETWORK_CHANGE = "network_change"
    FILE_CHANGED = "file_changed"
    SCREEN_CHANGED = "screen_changed"


@dataclass
class DesktopEvent:
    """A single desktop awareness event."""

    event_type: DesktopEventType
    timestamp: datetime
    data: Dict[str, Any]
    description: str


@dataclass
class WindowInfo:
    """Information about a desktop window."""

    title: str
    app_name: str
    pid: int
    geometry: Tuple[int, int, int, int]  # x, y, width, height
    is_focused: bool = False
    is_minimized: bool = False
    is_maximized: bool = False


@dataclass
class ProcessInfo:
    """Information about a running process."""

    pid: int
    name: str
    exe_path: str
    cmd_line: str
    cpu_percent: float
    memory_percent: float
    status: str
    create_time: float
    parent_pid: int
    username: str


@dataclass
class SystemSnapshot:
    """Point-in-time snapshot of the entire system state."""

    timestamp: datetime
    cpu_percent: float
    cpu_per_core: List[float]
    memory_percent: float
    memory_used: int
    memory_total: int
    disk_usage: List[Dict[str, Any]]
    network_io: Dict[str, Any]
    active_window: Optional[WindowInfo]
    running_processes_count: int
    clipboard_text: str
    battery_percent: Optional[float]
    uptime: float
    gpu_info: List[Dict[str, Any]]


@dataclass
class NetworkConnection:
    """Information about a network connection."""

    local_ip: str
    local_port: int
    remote_ip: str
    remote_port: int
    status: str
    pid: int
    process_name: str
    protocol: str


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _run_blocking(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous blocking function in a thread executor."""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, func, *args, **kwargs)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════
# Watchdog file-system handler
# ═══════════════════════════════════════════════════════════════════════════


class _FSHandler(FileSystemEventHandler):
    """Bridges watchdog events into the asyncio event queue."""

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._queue = queue
        self._loop = loop

    def _emit(self, event: "FileSystemEvent") -> None:
        try:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                DesktopEvent(
                    event_type=DesktopEventType.FILE_CHANGED,
                    timestamp=_now(),
                    data={
                        "src_path": getattr(event, "src_path", ""),
                        "dest_path": getattr(event, "dest_path", ""),
                        "event_type": event.event_type,
                        "is_directory": event.is_directory,
                    },
                    description=f"File changed: {event.src_path}",
                ),
            )
        except Exception:
            pass

    def on_created(self, event: "FileSystemEvent") -> None:
        self._emit(event)

    def on_deleted(self, event: "FileSystemEvent") -> None:
        self._emit(event)

    def on_modified(self, event: "FileSystemEvent") -> None:
        self._emit(event)

    def on_moved(self, event: "FileSystemEvent") -> None:
        self._emit(event)


# ═══════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════


class DesktopAwareness:
    """Real-time PC awareness for an AI desktop assistant.

    Parameters
    ----------
    monitor_clipboard:
        Enable periodic clipboard change detection.
    monitor_windows:
        Enable periodic active-window change detection.
    snapshot_interval:
        Seconds between automatic system snapshots when the background
        monitor is running.
    clipboard_history_size:
        Maximum number of clipboard entries to retain.
    window_history_minutes:
        Minutes of window-focus history to retain.
    cpu_high_threshold:
        Percentage above which ``CPU_HIGH`` events fire.
    memory_high_threshold:
        Percentage above which ``MEMORY_HIGH`` events fire.
    disk_low_threshold:
        Percentage of free space below which ``DISK_LOW`` events fire.
    screen_change_threshold:
        Pixel-difference ratio above which ``SCREEN_CHANGED`` events fire.
    """

    def __init__(
        self,
        monitor_clipboard: bool = True,
        monitor_windows: bool = True,
        snapshot_interval: int = 5,
        clipboard_history_size: int = 100,
        window_history_minutes: int = 60,
        cpu_high_threshold: float = 90.0,
        memory_high_threshold: float = 90.0,
        disk_low_threshold: float = 10.0,
        screen_change_threshold: float = 0.02,
    ) -> None:
        self._monitor_clipboard = monitor_clipboard
        self._monitor_windows = monitor_windows
        self._snapshot_interval = snapshot_interval
        self._clipboard_history_size = clipboard_history_size
        self._window_history_minutes = window_history_minutes
        self._cpu_high_threshold = cpu_high_threshold
        self._memory_high_threshold = memory_high_threshold
        self._disk_low_threshold = disk_low_threshold
        self._screen_change_threshold = screen_change_threshold

        # State ----------------------------------------------------------
        self._running = False
        self._bg_task: Optional[asyncio.Task] = None
        self._event_queue: asyncio.Queue[DesktopEvent] = asyncio.Queue()
        self._event_handlers: Dict[DesktopEventType, List[Callable]] = {}

        # Clipboard ------------------------------------------------------
        self._clipboard_history: Deque[Dict[str, Any]] = deque(
            maxlen=self._clipboard_history_size
        )
        self._last_clipboard: str = ""

        # Window focus ---------------------------------------------------
        self._last_window_title: str = ""
        self._window_history: Deque[Dict[str, Any]] = deque()

        # Process tracking -----------------------------------------------
        self._known_pids: set = set()

        # Screen change detection ----------------------------------------
        self._last_screenshot_bytes: Optional[bytes] = None

        # Previous snapshot for threshold alerts -------------------------
        self._last_cpu: float = 0.0
        self._last_memory: float = 0.0

        # Network --------------------------------------------------------
        self._last_net_io: Dict[str, Tuple[int, int]] = {}

        # Filesystem watchers --------------------------------------------
        self._fs_observer: Optional[Observer] = None
        self._fs_watched_paths: set = set()

        # Loop reference (set during initialize) -------------------------
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Start all monitors and record the initial baseline state."""
        if self._running:
            return

        self._loop = asyncio.get_running_loop()
        self._running = True

        # Baseline process list
        if HAS_PSUTIL:
            for p in psutil.process_iter(["pid"]):
                try:
                    self._known_pids.add(p.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        # Baseline clipboard
        self._last_clipboard = await self.get_clipboard_text()

        # Baseline window
        try:
            win = await self.get_active_window()
            self._last_window_title = win.title
        except Exception:
            pass

        # Baseline network I/O
        if HAS_PSUTIL:
            net = psutil.net_io_counters(pernic=True)
            self._last_net_io = {
                nic: (io.bytes_sent, io.bytes_recv) for nic, io in net.items()
            }

        logger.info("DesktopAwareness initialized")

    async def shutdown(self) -> None:
        """Stop all monitors and release resources."""
        self._running = False

        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()
            try:
                await self._bg_task
            except asyncio.CancelledError:
                pass
            self._bg_task = None

        if self._fs_observer is not None:
            self._fs_observer.stop()
            self._fs_observer.join(timeout=5)
            self._fs_observer = None

        self._event_handlers.clear()
        logger.info("DesktopAwareness shut down")

    # ------------------------------------------------------------------
    # Event system
    # ------------------------------------------------------------------

    def register_event_handler(
        self, event_type: DesktopEventType, handler: Callable[[DesktopEvent], Any]
    ) -> None:
        """Register a callback for a specific event type."""
        self._event_handlers.setdefault(event_type, []).append(handler)

    async def _emit_event(self, event: DesktopEvent) -> None:
        """Deliver an event to registered handlers and the async queue."""
        await self._event_queue.put(event)
        for handler in self._event_handlers.get(event.event_type, []):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.warning("Event handler error for %s: %s", event.event_type, exc)

    async def watch_events(self) -> AsyncGenerator[DesktopEvent, None]:
        """Async generator that yields desktop events as they occur."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # Background monitor
    # ------------------------------------------------------------------

    async def start_background_monitor(self) -> None:
        """Start the periodic snapshot + change-detection loop."""

        async def _loop() -> None:
            while self._running:
                try:
                    await self._tick()
                except Exception as exc:
                    logger.debug("Background tick error: %s", exc)
                await asyncio.sleep(self._snapshot_interval)

        self._bg_task = asyncio.create_task(_loop())

    async def _tick(self) -> None:
        """One iteration of the background monitor."""
        # Clipboard
        if self._monitor_clipboard:
            try:
                text = await self.get_clipboard_text()
                if text and text != self._last_clipboard:
                    prev = self._last_clipboard
                    self._last_clipboard = text
                    self._clipboard_history.append(
                        {"text": text, "timestamp": _now().isoformat()}
                    )
                    await self._emit_event(
                        DesktopEvent(
                            event_type=DesktopEventType.CLIPBOARD_CHANGED,
                            timestamp=_now(),
                            data={"new_text": text, "previous_text": prev},
                            description=f"Clipboard changed ({len(text)} chars)",
                        )
                    )
            except Exception:
                pass

        # Window focus
        if self._monitor_windows:
            try:
                win = await self.get_active_window()
                if win.title != self._last_window_title:
                    prev = self._last_window_title
                    self._last_window_title = win.title
                    entry = {
                        "title": win.title,
                        "app_name": win.app_name,
                        "pid": win.pid,
                        "timestamp": _now().isoformat(),
                    }
                    self._window_history.append(entry)
                    await self._emit_event(
                        DesktopEvent(
                            event_type=DesktopEventType.WINDOW_FOCUS_CHANGED,
                            timestamp=_now(),
                            data={"new_window": entry, "previous_title": prev},
                            description=f"Focus → {win.app_name}: {win.title}",
                        )
                    )
            except Exception:
                pass

        # Process tracking
        await self._check_process_changes()

        # Threshold alerts
        await self._check_thresholds()

        # Screen change detection
        if HAS_PYAUTOGUI and HAS_PIL:
            await self._check_screen_change()

    async def _check_process_changes(self) -> None:
        if not HAS_PSUTIL:
            return
        current_pids: set = set()
        try:
            for p in psutil.process_iter(["pid", "name"]):
                try:
                    current_pids.add(p.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            return

        started = current_pids - self._known_pids
        stopped = self._known_pids - current_pids

        for pid in started:
            try:
                proc = psutil.Process(pid)
                await self._emit_event(
                    DesktopEvent(
                        event_type=DesktopEventType.PROCESS_STARTED,
                        timestamp=_now(),
                        data={"pid": pid, "name": proc.name()},
                        description=f"Process started: {proc.name()} (pid {pid})",
                    )
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        for pid in stopped:
            await self._emit_event(
                DesktopEvent(
                    event_type=DesktopEventType.PROCESS_STOPPED,
                    timestamp=_now(),
                    data={"pid": pid},
                    description=f"Process stopped: pid {pid}",
                )
            )

        self._known_pids = current_pids

    async def _check_thresholds(self) -> None:
        if not HAS_PSUTIL:
            return
        try:
            cpu = psutil.cpu_percent(interval=0)
            if cpu > self._cpu_high_threshold and self._last_cpu <= self._cpu_high_threshold:
                await self._emit_event(
                    DesktopEvent(
                        event_type=DesktopEventType.CPU_HIGH,
                        timestamp=_now(),
                        data={"cpu_percent": cpu, "threshold": self._cpu_high_threshold},
                        description=f"CPU usage high: {cpu:.1f}%",
                    )
                )
            self._last_cpu = cpu

            mem = psutil.virtual_memory()
            mem_pct = mem.percent
            if (
                mem_pct > self._memory_high_threshold
                and self._last_memory <= self._memory_high_threshold
            ):
                await self._emit_event(
                    DesktopEvent(
                        event_type=DesktopEventType.MEMORY_HIGH,
                        timestamp=_now(),
                        data={
                            "memory_percent": mem_pct,
                            "memory_used_gb": mem.used / (1024 ** 3),
                            "threshold": self._memory_high_threshold,
                        },
                        description=f"Memory usage high: {mem_pct:.1f}%",
                    )
                )
            self._last_memory = mem_pct

            for part in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    free_pct = (usage.free / usage.total) * 100
                    if free_pct < self._disk_low_threshold:
                        await self._emit_event(
                            DesktopEvent(
                                event_type=DesktopEventType.DISK_LOW,
                                timestamp=_now(),
                                data={
                                    "mountpoint": part.mountpoint,
                                    "device": part.device,
                                    "free_percent": free_pct,
                                    "free_gb": usage.free / (1024 ** 3),
                                    "total_gb": usage.total / (1024 ** 3),
                                },
                                description=(
                                    f"Disk space low on {part.mountpoint}: "
                                    f"{free_pct:.1f}% free"
                                ),
                            )
                        )
                except (PermissionError, OSError):
                    continue
        except Exception:
            pass

    async def _check_screen_change(self) -> None:
        """Compare current screenshot to the previous one pixel-by-pixel."""
        try:
            current = await self.take_screenshot()
            if current is None or self._last_screenshot_bytes is None:
                self._last_screenshot_bytes = current
                return
            if current == self._last_screenshot_bytes:
                return

            prev_img = Image.open(__import__("io").BytesIO(self._last_screenshot_bytes))
            curr_img = Image.open(__import__("io").BytesIO(current))
            if prev_img.size != curr_img.size:
                prev_img = prev_img.resize(curr_img.size)

            # Sample comparison – resize to small for speed
            small = (64, 64)
            p1 = prev_img.convert("L").resize(small)
            p2 = curr_img.convert("L").resize(small)
            px1 = list(p1.getdata())
            px2 = list(p2.getdata())
            if len(px1) != len(px2):
                self._last_screenshot_bytes = current
                return

            diff_count = sum(1 for a, b in zip(px1, px2) if abs(a - b) > 15)
            ratio = diff_count / len(px1)
            if ratio > self._screen_change_threshold:
                await self._emit_event(
                    DesktopEvent(
                        event_type=DesktopEventType.SCREEN_CHANGED,
                        timestamp=_now(),
                        data={"change_ratio": ratio},
                        description=f"Screen changed ({ratio:.2%} pixels differ)",
                    )
                )
            self._last_screenshot_bytes = current
        except Exception:
            pass

    # ------------------------------------------------------------------
    # System snapshot
    # ------------------------------------------------------------------

    async def get_system_snapshot(self) -> SystemSnapshot:
        """Build a point-in-time snapshot of the entire system state."""
        cpu_pct, cpu_cores = await self._get_cpu_raw()
        mem = await self._get_memory_raw()
        disks = await self.get_disk_info()
        net = await self._get_network_io_raw()
        gpu = await self.get_gpu_info()
        bat = await self.get_battery_info()
        uptime = await self._get_uptime_raw()
        active_win = await self._safe_active_window()
        proc_count = await self._process_count()
        clip = await self.get_clipboard_text()

        return SystemSnapshot(
            timestamp=_now(),
            cpu_percent=cpu_pct,
            cpu_per_core=cpu_cores,
            memory_percent=mem["percent"],
            memory_used=mem["used"],
            memory_total=mem["total"],
            disk_usage=disks,
            network_io=net,
            active_window=active_win,
            running_processes_count=proc_count,
            clipboard_text=clip,
            battery_percent=bat.get("percent"),
            uptime=uptime,
            gpu_info=gpu,
        )

    # ------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------

    async def _get_cpu_raw(self) -> Tuple[float, List[float]]:
        if not HAS_PSUTIL:
            return 0.0, []
        loop = asyncio.get_running_loop()

        def _read() -> Tuple[float, List[float]]:
            pct = psutil.cpu_percent(interval=0.5)
            cores = psutil.cpu_percent(interval=0, percpu=True)
            return pct, cores

        return await loop.run_in_executor(None, _read)

    async def get_cpu_info(self) -> dict:
        """Return CPU usage, per-core usage, frequency, and count."""
        if not HAS_PSUTIL:
            return {"error": "psutil not available"}
        pct, cores = await self._get_cpu_raw()
        freq = psutil.cpu_freq()
        return {
            "percent": pct,
            "per_core": cores,
            "count_logical": psutil.cpu_count(logical=True),
            "count_physical": psutil.cpu_count(logical=False),
            "freq_current": freq.current if freq else None,
            "freq_min": freq.min if freq else None,
            "freq_max": freq.max if freq else None,
        }

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    async def _get_memory_raw(self) -> Dict[str, Any]:
        if not HAS_PSUTIL:
            return {"percent": 0, "used": 0, "total": 1}
        loop = asyncio.get_running_loop()
        mem = await loop.run_in_executor(None, psutil.virtual_memory)
        return {
            "percent": mem.percent,
            "used": mem.used,
            "total": mem.total,
            "available": mem.available,
            "used_gb": mem.used / (1024 ** 3),
            "total_gb": mem.total / (1024 ** 3),
        }

    async def get_memory_info(self) -> dict:
        """Return RAM usage details including swap."""
        mem = await self._get_memory_raw()
        if HAS_PSUTIL:
            swap = psutil.swap_memory()
            mem["swap_percent"] = swap.percent
            mem["swap_used_gb"] = swap.used / (1024 ** 3)
            mem["swap_total_gb"] = swap.total / (1024 ** 3)
        return mem

    # ------------------------------------------------------------------
    # Disk
    # ------------------------------------------------------------------

    async def get_disk_info(self) -> List[dict]:
        """Return usage info for all mounted disks."""
        if not HAS_PSUTIL:
            return []

        def _read() -> List[dict]:
            result: List[dict] = []
            for part in psutil.disk_partitions(all=False):
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    result.append(
                        {
                            "device": part.device,
                            "mountpoint": part.mountpoint,
                            "fstype": part.fstype,
                            "total_gb": usage.total / (1024 ** 3),
                            "used_gb": usage.used / (1024 ** 3),
                            "free_gb": usage.free / (1024 ** 3),
                            "percent": usage.percent,
                        }
                    )
                except (PermissionError, OSError):
                    continue
            return result

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _read)

    # ------------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------------

    async def get_gpu_info(self) -> List[dict]:
        """Return GPU information (best-effort across platforms)."""
        gpus: List[dict] = []

        def _query_nvidia_smi() -> List[dict]:
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                lines = out.strip().splitlines()
                info: List[dict] = []
                for idx, line in enumerate(lines):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        info.append(
                            {
                                "index": idx,
                                "name": parts[0],
                                "memory_total_mb": float(parts[1]) if parts[1] else None,
                                "memory_used_mb": float(parts[2]) if parts[2] else None,
                                "memory_free_mb": float(parts[3]) if parts[3] else None,
                                "temperature_c": float(parts[4]) if parts[4] else None,
                                "utilization_percent": (
                                    float(parts[5]) if parts[5] else None
                                ),
                            }
                        )
                return info
            except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
                return []

        nvidia_gpus = await _run_blocking(_query_nvidia_smi) if HAS_PSUTIL else []
        gpus.extend(nvidia_gpus)

        if not gpus:
            gpus.append({"info": "No GPU information available"})

        return gpus

    # ------------------------------------------------------------------
    # Battery
    # ------------------------------------------------------------------

    async def get_battery_info(self) -> dict:
        """Return battery status if available."""
        if not HAS_PSUTIL:
            return {"percent": None, "plugged": None}
        loop = asyncio.get_running_loop()

        def _read() -> dict:
            if not hasattr(psutil, "sensors_battery"):
                return {"percent": None, "plugged": None}
            bat = psutil.sensors_battery()
            if bat is None:
                return {"percent": None, "plugged": None}
            return {
                "percent": bat.percent,
                "plugged": bat.power_plugged,
                "secs_left": bat.secsleft,
                "time_left": (
                    f"{bat.secsleft // 60}m {bat.secsleft % 60}s"
                    if bat.secsleft >= 0
                    else None
                ),
            }

        return await loop.run_in_executor(None, _read)

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    async def _get_network_io_raw(self) -> Dict[str, Any]:
        if not HAS_PSUTIL:
            return {}
        loop = asyncio.get_running_loop()

        def _read() -> Dict[str, Any]:
            io = psutil.net_io_counters(pernic=True)
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()
            result: Dict[str, Any] = {}
            for nic, counters in io.items():
                result[nic] = {
                    "bytes_sent": counters.bytes_sent,
                    "bytes_recv": counters.bytes_recv,
                    "packets_sent": counters.packets_sent,
                    "packets_recv": counters.packets_recv,
                    "addresses": [
                        {"address": a.address, "family": a.family.name}
                        for a in addrs.get(nic, [])
                    ],
                    "up": stats[nic].isup if nic in stats else None,
                    "speed": stats[nic].speed if nic in stats else None,
                }
            return result

        return await loop.run_in_executor(None, _read)

    async def get_network_info(self) -> dict:
        """Return network interfaces, IPs, and I/O counters."""
        data = await self._get_network_io_raw()
        if not HAS_PSUTIL:
            return {"error": "psutil not available", "interfaces": data}
        # Convenience: primary hostname and OS info
        return {
            "hostname": platform.node(),
            "interfaces": data,
        }

    # ------------------------------------------------------------------
    # Network connections / ports
    # ------------------------------------------------------------------

    async def get_active_connections(self) -> List[NetworkConnection]:
        """Return all active TCP/UDP network connections."""
        if not HAS_PSUTIL:
            return []
        loop = asyncio.get_running_loop()

        def _read() -> List[NetworkConnection]:
            result: List[NetworkConnection] = []
            try:
                for conn in psutil.net_connections(kind="inet"):
                    pname = ""
                    pid = conn.pid or 0
                    try:
                        if pid:
                            pname = psutil.Process(pid).name()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    result.append(
                        NetworkConnection(
                            local_ip=conn.laddr.ip if conn.laddr else "",
                            local_port=conn.laddr.port if conn.laddr else 0,
                            remote_ip=conn.raddr.ip if conn.raddr else "",
                            remote_port=conn.raddr.port if conn.raddr else 0,
                            status=conn.status,
                            pid=pid,
                            process_name=pname,
                            protocol=conn.type.name if conn.type else "UNKNOWN",
                        )
                    )
            except (psutil.AccessDenied, OSError):
                pass
            return result

        return await loop.run_in_executor(None, _read)

    async def get_listening_ports(self) -> List[dict]:
        """Return all ports in LISTEN state."""
        all_conns = await self.get_active_connections()
        return [
            {
                "port": c.local_port,
                "protocol": c.protocol,
                "pid": c.pid,
                "process_name": c.process_name,
                "local_ip": c.local_ip,
            }
            for c in all_conns
            if c.status == "LISTEN"
        ]

    # ------------------------------------------------------------------
    # Window tracking
    # ------------------------------------------------------------------

    async def get_active_window(self) -> WindowInfo:
        """Return information about the currently focused window."""
        if not HAS_PYAUTOGUI:
            return WindowInfo(
                title="",
                app_name="",
                pid=0,
                geometry=(0, 0, 0, 0),
            )

        def _read() -> WindowInfo:
            try:
                import pyscreeze  # type: ignore

                # pyautogui.getActiveWindow() may not be available on all platforms.
                # Fall back gracefully.
                if not hasattr(pyautogui, "getActiveWindow"):
                    return WindowInfo(
                        title="",
                        app_name="",
                        pid=0,
                        geometry=(0, 0, 0, 0),
                    )
                win = pyautogui.getActiveWindow()
                if win is None:
                    return WindowInfo(
                        title="",
                        app_name="",
                        pid=0,
                        geometry=(0, 0, 0, 0),
                    )
                return WindowInfo(
                    title=getattr(win, "title", "") or "",
                    app_name=getattr(win, "appName", "") or "",
                    pid=int(getattr(win, "processId", 0) or 0),
                    geometry=(
                        getattr(win, "left", 0),
                        getattr(win, "top", 0),
                        getattr(win, "width", 0),
                        getattr(win, "height", 0),
                    ),
                    is_focused=True,
                    is_minimized=bool(getattr(win, "isMinimized", False)),
                    is_maximized=bool(getattr(win, "isMaximized", False)),
                )
            except Exception:
                return WindowInfo(
                    title="",
                    app_name="",
                    pid=0,
                    geometry=(0, 0, 0, 0),
                )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _read)

    async def _safe_active_window(self) -> Optional[WindowInfo]:
        try:
            win = await self.get_active_window()
            if win.title:
                return win
        except Exception:
            pass
        return None

    async def get_all_windows(self) -> List[WindowInfo]:
        """Return information about all visible windows."""
        if not HAS_PYAUTOGUI or not hasattr(pyautogui, "getAllWindows"):
            return []

        def _read() -> List[WindowInfo]:
            try:
                windows = pyautogui.getAllWindows()
            except Exception:
                return []
            result: List[WindowInfo] = []
            for w in windows:
                try:
                    result.append(
                        WindowInfo(
                            title=getattr(w, "title", "") or "",
                            app_name=getattr(w, "appName", "") or "",
                            pid=int(getattr(w, "processId", 0) or 0),
                            geometry=(
                                getattr(w, "left", 0),
                                getattr(w, "top", 0),
                                getattr(w, "width", 0),
                                getattr(w, "height", 0),
                            ),
                            is_focused=bool(getattr(w, "isActive", False)),
                            is_minimized=bool(getattr(w, "isMinimized", False)),
                            is_maximized=bool(getattr(w, "isMaximized", False)),
                        )
                    )
                except Exception:
                    continue
            return result

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _read)

    async def get_window_history(self, minutes: int = 30) -> List[dict]:
        """Return window-focus history for the last *minutes* minutes."""
        cutoff = _now().timestamp() - (minutes * 60)
        return [
            entry
            for entry in self._window_history
            if datetime.fromisoformat(entry["timestamp"]).timestamp() >= cutoff
        ]

    # ------------------------------------------------------------------
    # Process monitoring
    # ------------------------------------------------------------------

    async def _process_count(self) -> int:
        if not HAS_PSUTIL:
            return 0
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(lambda: len(list(psutil.process_iter())))

    async def get_running_processes(
        self, name_filter: Optional[str] = None, sort_by: str = "cpu"
    ) -> List[ProcessInfo]:
        """Return a list of running processes, optionally filtered and sorted."""
        if not HAS_PSUTIL:
            return []
        loop = asyncio.get_running_loop()

        def _read() -> List[ProcessInfo]:
            procs: List[ProcessInfo] = []
            for p in psutil.process_iter(
                ["pid", "name", "exe", "cmdline", "cpu_percent", "memory_percent",
                 "status", "create_time", "ppid", "username"]
            ):
                try:
                    info = p.info
                    name = info["name"] or ""
                    if name_filter and name_filter.lower() not in name.lower():
                        continue
                    cmd = " ".join(info["cmdline"] or [])
                    procs.append(
                        ProcessInfo(
                            pid=info["pid"],
                            name=name,
                            exe_path=info["exe"] or "",
                            cmd_line=cmd,
                            cpu_percent=info["cpu_percent"] or 0.0,
                            memory_percent=info["memory_percent"] or 0.0,
                            status=info["status"] or "",
                            create_time=info["create_time"] or 0.0,
                            parent_pid=info["ppid"] or 0,
                            username=info["username"] or "",
                        )
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            reverse = True
            if sort_by == "cpu":
                procs.sort(key=lambda p: p.cpu_percent, reverse=reverse)
            elif sort_by == "memory":
                procs.sort(key=lambda p: p.memory_percent, reverse=reverse)
            elif sort_by == "name":
                procs.sort(key=lambda p: p.name.lower(), reverse=False)
            elif sort_by == "pid":
                procs.sort(key=lambda p: p.pid, reverse=False)
            return procs

        return await loop.run_in_executor(None, _read)

    async def get_process(self, pid: int) -> ProcessInfo:
        """Return information about a single process by PID."""
        if not HAS_PSUTIL:
            raise RuntimeError("psutil not available")
        loop = asyncio.get_running_loop()

        def _read() -> ProcessInfo:
            p = psutil.Process(pid)
            with p.oneshot():
                cmd = " ".join(p.cmdline())
                return ProcessInfo(
                    pid=p.pid,
                    name=p.name(),
                    exe_path=p.exe() or "",
                    cmd_line=cmd,
                    cpu_percent=p.cpu_percent(),
                    memory_percent=p.memory_percent(),
                    status=p.status(),
                    create_time=p.create_time(),
                    parent_pid=p.ppid(),
                    username=p.username() or "",
                )

        return await loop.run_in_executor(None, _read)

    async def kill_process(self, pid: int) -> bool:
        """Terminate a process by PID. Returns True on success."""
        if not HAS_PSUTIL:
            return False
        loop = asyncio.get_running_loop()

        def _kill() -> bool:
            try:
                p = psutil.Process(pid)
                p.terminate()
                p.wait(timeout=5)
                return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                try:
                    psutil.Process(pid).kill()
                    return True
                except Exception:
                    return False

        return await loop.run_in_executor(None, _kill)

    # ------------------------------------------------------------------
    # Clipboard
    # ------------------------------------------------------------------

    async def get_clipboard_text(self) -> str:
        """Read the current clipboard text content."""
        if not HAS_PYAUTOGUI:
            return ""
        loop = asyncio.get_running_loop()

        def _read() -> str:
            try:
                return pyautogui.paste() or ""
            except Exception:
                return ""

        return await loop.run_in_executor(None, _read)

    async def get_clipboard_history(self, limit: int = 20) -> List[dict]:
        """Return the last *limit* clipboard entries."""
        items = list(self._clipboard_history)
        return items[-limit:]

    # ------------------------------------------------------------------
    # Screen capture & OCR
    # ------------------------------------------------------------------

    async def take_screenshot(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        window_title: Optional[str] = None,
    ) -> Optional[bytes]:
        """Take a screenshot and return PNG bytes.

        Parameters
        ----------
        region:
            Crop to (left, top, width, height).
        window_title:
            If provided, attempt to capture only the window matching this
            title (substring match).
        """
        if not HAS_PYAUTOGUI or not HAS_PIL:
            return None
        loop = asyncio.get_running_loop()

        def _capture() -> Optional[bytes]:
            try:
                if window_title and hasattr(pyautogui, "getWindowsWithTitle"):
                    matches = pyautogui.getWindowsWithTitle(window_title)
                    if matches:
                        w = matches[0]
                        if hasattr(w, "activate"):
                            try:
                                w.activate()
                                time.sleep(0.15)
                            except Exception:
                                pass
                        left = getattr(w, "left", 0)
                        top = getattr(w, "top", 0)
                        width = getattr(w, "width", 0)
                        height = getattr(w, "height", 0)
                        if width > 0 and height > 0:
                            img = pyautogui.screenshot(region=(left, top, width, height))
                        else:
                            img = pyautogui.screenshot(region=region)
                    else:
                        img = pyautogui.screenshot(region=region)
                else:
                    img = pyautogui.screenshot(region=region)

                buf = __import__("io").BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
            except Exception:
                return None

        return await loop.run_in_executor(None, _capture)

    async def ocr_screenshot(
        self, region: Optional[Tuple[int, int, int, int]] = None
    ) -> str:
        """Take a screenshot and run OCR on it. Falls back gracefully."""
        png_bytes = await self.take_screenshot(region=region)
        if png_bytes is None:
            return "[Screenshot not available]"

        if not HAS_TESSERACT or not HAS_PIL:
            return "[OCR not available]"

        loop = asyncio.get_running_loop()

        def _ocr() -> str:
            try:
                buf = __import__("io").BytesIO(png_bytes)
                img = Image.open(buf)
                text = pytesseract.image_to_string(img).strip()
                return text or "[No text detected]"
            except Exception:
                return "[OCR error]"

        return await loop.run_in_executor(None, _ocr)

    # ------------------------------------------------------------------
    # File system monitoring
    # ------------------------------------------------------------------

    async def watch_directory(self, path: str) -> bool:
        """Start watching a directory for file-system changes."""
        if not HAS_WATCHDOG:
            logger.warning("watchdog not installed; cannot watch directories")
            return False
        if not self._loop:
            return False
        abs_path = str(Path(path).resolve())
        if abs_path in self._fs_watched_paths:
            return True
        if self._fs_observer is None:
            self._fs_observer = Observer()
            self._fs_observer.start()
        handler = _FSHandler(self._event_queue, self._loop)
        self._fs_observer.schedule(handler, abs_path, recursive=True)
        self._fs_watched_paths.add(abs_path)
        logger.info("Watching directory: %s", abs_path)
        return True

    async def unwatch_directory(self, path: str) -> bool:
        """Stop watching a directory."""
        if self._fs_observer is None:
            return False
        abs_path = str(Path(path).resolve())
        # watchdog does not expose an easy way to unschedule by path alone,
        # so we store watched paths for bookkeeping but need to recreate the
        # observer to truly stop watching.  For now just record the intent.
        self._fs_watched_paths.discard(abs_path)
        return True

    # ------------------------------------------------------------------
    # Installed applications
    # ------------------------------------------------------------------

    async def get_installed_apps(self) -> List[dict]:
        """Return a list of installed applications (platform-dependent)."""
        apps: List[dict] = []
        system = platform.system()

        if system == "Windows":
            apps = await self._get_installed_apps_windows()
        elif system == "Darwin":
            apps = await self._get_installed_apps_macos()
        elif system == "Linux":
            apps = await self._get_installed_apps_linux()
        return apps

    async def _get_installed_apps_windows(self) -> List[dict]:
        loop = asyncio.get_running_loop()

        def _read() -> List[dict]:
            import winreg  # type: ignore

            apps: List[dict] = []
            reg_paths = [
                winreg.HKEY_LOCAL_MACHINE,
                winreg.HKEY_CURRENT_USER,
            ]
            sub_keys = [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
            ]
            for hive in reg_paths:
                for sub in sub_keys:
                    try:
                        key = winreg.OpenKey(hive, sub)
                        for i in range(winreg.QueryInfoKey(key)[0]):
                            try:
                                app_key = winreg.OpenKey(key, winreg.EnumKey(key, i))
                                name = winreg.QueryValueEx(app_key, "DisplayName")[0]
                                version = ""
                                try:
                                    version = winreg.QueryValueEx(app_key, "DisplayVersion")[0]
                                except OSError:
                                    pass
                                install_loc = ""
                                try:
                                    install_loc = winreg.QueryValueEx(app_key, "InstallLocation")[0]
                                except OSError:
                                    pass
                                apps.append(
                                    {
                                        "name": name,
                                        "version": version,
                                        "install_location": install_loc,
                                    }
                                )
                                winreg.CloseKey(app_key)
                            except (OSError, ValueError):
                                continue
                        winreg.CloseKey(key)
                    except (OSError, FileNotFoundError):
                        continue
            return apps

        return await loop.run_in_executor(None, _read)

    async def _get_installed_apps_macos(self) -> List[dict]:
        loop = asyncio.get_running_loop()

        def _read() -> List[dict]:
            apps: List[dict] = []
            app_dirs = [
                "/Applications",
                os.path.expanduser("~/Applications"),
            ]
            for d in app_dirs:
                if not os.path.isdir(d):
                    continue
                for entry in os.scandir(d):
                    if entry.is_dir() and entry.name.endswith(".app"):
                        apps.append(
                            {
                                "name": entry.name,
                                "path": entry.path,
                            }
                        )
            return apps

        return await loop.run_in_executor(None, _read)

    async def _get_installed_apps_linux(self) -> List[dict]:
        loop = asyncio.get_running_loop()

        def _read() -> List[dict]:
            apps: List[dict] = []
            # Flatpak
            flatpak = shutil.which("flatpak")
            if flatpak:
                try:
                    out = subprocess.check_output(
                        [flatpak, "list", "--app", "--columns=name,version"],
                        text=True,
                        timeout=10,
                    )
                    for line in out.strip().splitlines():
                        parts = line.split("\t")
                        apps.append(
                            {
                                "name": parts[0] if parts else "",
                                "version": parts[1] if len(parts) > 1 else "",
                                "source": "flatpak",
                            }
                        )
                except Exception:
                    pass
            # Snap
            snap = shutil.which("snap")
            if snap:
                try:
                    out = subprocess.check_output(
                        ["snap", "list", "--app"],
                        text=True,
                        timeout=10,
                    )
                    for line in out.strip().splitlines()[1:]:
                        parts = line.split()
                        if parts:
                            apps.append(
                                {
                                    "name": parts[0],
                                    "version": parts[1] if len(parts) > 1 else "",
                                    "source": "snap",
                                }
                            )
                except Exception:
                    pass
            # Desktop entries
            desktop_dirs = [
                "/usr/share/applications",
                os.path.expanduser("~/.local/share/applications"),
                "/var/lib/flatpak/exports/share/applications",
            ]
            seen: set = set()
            for dd in desktop_dirs:
                if not os.path.isdir(dd):
                    continue
                for f in os.listdir(dd):
                    if f.endswith(".desktop") and f not in seen:
                        seen.add(f)
                        filepath = os.path.join(dd, f)
                        name = f[:-8]
                        try:
                            with open(filepath, "r", errors="ignore") as fh:
                                for line in fh:
                                    if line.startswith("Name="):
                                        name = line.strip().split("=", 1)[1]
                                        break
                        except OSError:
                            pass
                        apps.append({"name": name, "source": "desktop-entry"})
            return apps

        return await loop.run_in_executor(None, _read)

    # ------------------------------------------------------------------
    # Environment context
    # ------------------------------------------------------------------

    async def get_environment_context(self) -> dict:
        """Return a comprehensive context dict useful for AI reasoning."""
        cpu = await self.get_cpu_info()
        mem = await self.get_memory_info()
        disks = await self.get_disk_info()
        net = await self.get_network_info()
        bat = await self.get_battery_info()
        gpu = await self.get_gpu_info()
        active_win = await _safe_active_window()
        all_windows = await self.get_all_windows()
        procs = await self.get_running_processes(sort_by="memory")
        clip = await self.get_clipboard_text()
        uptime = await self._get_uptime_raw()

        # Current working directory of our process
        cwd = os.getcwd()

        # Find git repos in cwd and home
        git_repos = await self._find_git_repos()

        # Docker
        docker = await self._get_docker_info()

        # Recently opened files (platform-dependent)
        recent_files = await self._get_recent_files()

        # Open terminals / IDEs
        open_ides = await self._detect_open_ides(procs)

        return {
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "hostname": platform.node(),
                "processor": platform.processor(),
            },
            "cpu": cpu,
            "memory": mem,
            "disks": disks,
            "gpu": gpu,
            "battery": bat,
            "network": net,
            "uptime_seconds": uptime,
            "uptime_human": self._format_uptime(uptime),
            "active_window": {
                "title": active_win.title if active_win else "",
                "app_name": active_win.app_name if active_win else "",
                "pid": active_win.pid if active_win else 0,
            }
            if active_win
            else None,
            "open_windows": [
                {"title": w.title, "app_name": w.app_name, "pid": w.pid}
                for w in all_windows
            ],
            "top_processes_by_memory": [
                {
                    "pid": p.pid,
                    "name": p.name,
                    "cpu_percent": p.cpu_percent,
                    "memory_percent": p.memory_percent,
                }
                for p in procs[:15]
            ],
            "process_count": len(procs),
            "clipboard_preview": (clip[:500] + "...") if len(clip) > 500 else clip,
            "current_working_directory": cwd,
            "git_repos_found": git_repos,
            "docker": docker,
            "recent_files": recent_files,
            "open_ides": open_ides,
            "environment_variables": {
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", os.environ.get("USERPROFILE", "")),
                "USER": os.environ.get("USER", os.environ.get("USERNAME", "")),
                "SHELL": os.environ.get("SHELL", ""),
                "LANG": os.environ.get("LANG", ""),
                "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
                "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
            },
        }

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------

    async def _get_uptime_raw(self) -> float:
        if not HAS_PSUTIL:
            return 0.0
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(lambda: psutil.boot_time())

    @staticmethod
    def _format_uptime(boot_time: float) -> str:
        try:
            elapsed = time.time() - boot_time
            days = int(elapsed // 86400)
            hours = int((elapsed % 86400) // 3600)
            minutes = int((elapsed % 3600) // 60)
            parts = []
            if days:
                parts.append(f"{days}d")
            if hours:
                parts.append(f"{hours}h")
            parts.append(f"{minutes}m")
            return " ".join(parts)
        except Exception:
            return "unknown"

    async def _find_git_repos(self) -> List[str]:
        """Search cwd and home for git repositories (non-recursive depth)."""
        loop = asyncio.get_running_loop()

        def _search() -> List[str]:
            repos: List[str] = []
            search_paths = set()
            search_paths.add(os.getcwd())
            home = os.path.expanduser("~")
            search_paths.add(home)
            # Common project directories
            for extra in ["projects", "code", "dev", "repos", "workspace", "src"]:
                p = os.path.join(home, extra)
                if os.path.isdir(p):
                    search_paths.add(p)

            for base in search_paths:
                if not os.path.isdir(base):
                    continue
                try:
                    for entry in os.scandir(base):
                        if entry.is_dir():
                            git_dir = os.path.join(entry.path, ".git")
                            if os.path.isdir(git_dir):
                                repos.append(entry.path)
                except PermissionError:
                    continue
            return repos

        return await loop.run_in_executor(None, _search)

    async def _get_docker_info(self) -> dict:
        """Return running Docker containers if Docker is available."""
        docker = shutil.which("docker")
        if not docker:
            return {"available": False}

        loop = asyncio.get_running_loop()

        def _query() -> dict:
            try:
                out = subprocess.check_output(
                    ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"],
                    text=True,
                    timeout=10,
                )
                containers = []
                for line in out.strip().splitlines():
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        containers.append(
                            {
                                "name": parts[0],
                                "image": parts[1],
                                "status": parts[2],
                            }
                        )
                return {"available": True, "running_containers": containers}
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                return {"available": True, "running_containers": []}

        return await loop.run_in_executor(None, _query)

    async def _get_recent_files(self) -> List[dict]:
        """Return recently accessed files (platform-dependent, best-effort)."""
        system = platform.system()
        loop = asyncio.get_running_loop()

        def _macos_recent() -> List[dict]:
            recent_dir = os.path.expanduser(
                "~/Library/Application Support/com.apple.sharedfilelist/"
                "RecentApplications.sfl"
            )
            # Use fs_usage or Recent Items plist as a best-effort
            recent_items = os.path.expanduser("~/Library/Application Support/com.apple.sharedfilelist/")
            files: List[dict] = []
            # Simpler approach: read ~/Recent Items if it exists
            recent = os.path.expanduser("~/Recent Items")
            if os.path.exists(recent):
                try:
                    # It's a binary plist on newer macOS; skip parsing and return empty
                    pass
                except Exception:
                    pass
            return files

        def _windows_recent() -> List[dict]:
            import ctypes
            import ctypes.wintypes  # type: ignore

            files: List[dict] = []
            recent_dir = os.path.join(
                os.environ.get("APPDATA", ""), "Microsoft\\Windows\\Recent"
            )
            if not os.path.isdir(recent_dir):
                return files
            try:
                for entry in os.scandir(recent_dir):
                    if entry.is_file() and entry.name.endswith(".lnk"):
                        files.append(
                            {
                                "name": entry.name[:-4],
                                "path": entry.path,
                                "modified": datetime.fromtimestamp(
                                    entry.stat().st_mtime, tz=timezone.utc
                                ).isoformat(),
                            }
                        )
            except PermissionError:
                pass
            return files

        def _linux_recent() -> List[dict]:
            files: List[dict] = []
            recent_dir = os.path.expanduser("~/.local/share/recently-used.xbel")
            if os.path.isfile(recent_dir):
                try:
                    import xml.etree.ElementTree as ET

                    tree = ET.parse(recent_dir)
                    root = tree.getroot()
                    for bookmark in root.iter("bookmark"):
                        href = bookmark.get("href", "")
                        if href.startswith("file://"):
                            files.append({"path": href[7:]})
                except Exception:
                    pass
            return files

        if system == "Windows":
            return await loop.run_in_executor(None, _windows_recent)
        elif system == "Darwin":
            return await loop.run_in_executor(None, _macos_recent)
        else:
            return await loop.run_in_executor(None, _linux_recent)

    @staticmethod
    async def _detect_open_ides(procs: List[ProcessInfo]) -> List[dict]:
        """Detect running IDEs / editors from the process list."""
        ide_signatures: Dict[str, List[str]] = {
            "VS Code": ["code", "Code - Insiders", "code-explorer"],
            "JetBrains": [
                "idea", "webstorm", "pycharm", "goland", "clion",
                "rustrover", "datagrip", "rider", "phpstorm", "rubymine",
            ],
            "Vim/Neovim": ["vim", "nvim", "neovim"],
            "Emacs": ["emacs"],
            "Sublime Text": ["sublime_text", "subl"],
            "Notepad++": ["notepad++"],
            "Cursor": ["cursor"],
            "Zed": ["zed"],
        }
        seen: Dict[str, dict] = {}
        for proc in procs:
            pname = proc.name.lower().replace(".exe", "").replace(" ", "")
            for ide, patterns in ide_signatures.items():
                for pat in patterns:
                    if pat.lower().replace(" ", "") in pname and ide not in seen:
                        seen[ide] = {
                            "name": ide,
                            "pid": proc.pid,
                            "process_name": proc.name,
                        }
                        break
        return list(seen.values())

    # ------------------------------------------------------------------
    # Browser tab detection (best-effort)
    # ------------------------------------------------------------------

    async def get_browser_tabs(self) -> List[dict]:
        """Attempt to list open browser tabs via JSON debugging ports."""
        tabs: List[dict] = []
        system = platform.system()
        loop = asyncio.get_running_loop()

        # Chrome / Edge / Brave expose a JSON API on a debugging port.
        # We try common ports. This is only possible if the browser was
        # launched with --remote-debugging-port.
        debug_ports = [9222, 9229, 9333]
        for port in debug_ports:
            try:
                import urllib.request
                import json

                url = f"http://127.0.0.1:{port}/json/list"
                req = urllib.request.Request(url, timeout=2)
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read())
                    for tab in data:
                        if tab.get("type") == "page":
                            tabs.append(
                                {
                                    "title": tab.get("title", ""),
                                    "url": tab.get("url", ""),
                                    "browser_port": port,
                                }
                            )
            except Exception:
                continue
        return tabs


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: quick sync context for non-async callers
# ═══════════════════════════════════════════════════════════════════════════


async def _async_context_summary() -> dict:
    """Return a quick environment context without persistent monitoring."""
    da = DesktopAwareness()
    await da.initialize()
    try:
        return await da.get_environment_context()
    finally:
        await da.shutdown()


def get_context_summary() -> dict:
    """Blocking helper: run the async context summary from sync code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, _async_context_summary()
            ).result(timeout=30)
    else:
        return asyncio.run(_async_context_summary())
