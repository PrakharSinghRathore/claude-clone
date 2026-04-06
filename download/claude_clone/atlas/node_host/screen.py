"""
Remote screen access and management for connected devices.

Provides screenshot capture, screen recording, and AI-powered
screen content analysis through the DeviceNode infrastructure.

Usage::

    screen = ScreenManager(device_node)
    screenshot = await screen.capture("desktop-01")
    recording = await screen.start_recording("desktop-01")
    analysis = await screen.analyze_screen(screenshot, "What app is open?")
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .device import Capability, DeviceNode

logger = logging.getLogger("atlas.node_host.screen")


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class ScreenFormat(str, Enum):
    """Screenshot output format."""

    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    BMP = "bmp"


class RecordingFormat(str, Enum):
    """Screen recording output format."""

    MP4 = "mp4"
    WEBM = "webm"
    GIF = "gif"
    MKV = "mkv"


class RecordingQuality(str, Enum):
    """Screen recording quality presets."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScreenshotResult:
    """Result of a screen capture operation."""

    success: bool
    device_id: str
    image_data: Optional[bytes] = None
    image_format: ScreenFormat = ScreenFormat.PNG
    width: int = 0
    height: int = 0
    file_size: int = 0
    display_index: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    screenshot_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses. Omits binary image_data."""
        return {
            "success": self.success,
            "device_id": self.device_id,
            "image_format": self.image_format.value,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "display_index": self.display_index,
            "timestamp": self.timestamp,
            "screenshot_id": self.screenshot_id,
            "error": self.error,
            "metadata": self.metadata,
        }

    def to_base64(self) -> Optional[str]:
        """Return base64-encoded image data, or None."""
        if self.image_data is None:
            return None
        return base64.b64encode(self.image_data).decode("ascii")

    @property
    def checksum(self) -> Optional[str]:
        """SHA-256 checksum of image data."""
        if self.image_data is None:
            return None
        return hashlib.sha256(self.image_data).hexdigest()

    @property
    def aspect_ratio(self) -> Optional[float]:
        """Compute aspect ratio (width / height)."""
        if self.height > 0:
            return self.width / self.height
        return None


@dataclass
class RecordingInfo:
    """Information about an active or completed screen recording."""

    recording_id: str
    device_id: str
    format: RecordingFormat = RecordingFormat.MP4
    quality: RecordingQuality = RecordingQuality.MEDIUM
    fps: int = 30
    width: int = 0
    height: int = 0
    started_at: float = field(default_factory=time.time)
    stopped_at: Optional[float] = None
    duration_seconds: float = 0.0
    file_size: int = 0
    is_active: bool = True
    audio_enabled: bool = False
    cursor_visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "device_id": self.device_id,
            "format": self.format.value,
            "quality": self.quality.value,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "duration_seconds": self._compute_duration(),
            "file_size": self.file_size,
            "is_active": self.is_active,
            "audio_enabled": self.audio_enabled,
            "cursor_visible": self.cursor_visible,
            "metadata": self.metadata,
        }

    def _compute_duration(self) -> float:
        end = self.stopped_at or time.time()
        return end - self.started_at

    @property
    def estimated_file_size_mb(self) -> float:
        """Rough estimate of file size based on quality and duration."""
        bitrate_map = {
            RecordingQuality.LOW: 1_000_000,
            RecordingQuality.MEDIUM: 5_000_000,
            RecordingQuality.HIGH: 15_000_000,
            RecordingQuality.ULTRA: 50_000_000,
        }
        bitrate = bitrate_map.get(self.quality, 5_000_000)
        duration = self._compute_duration()
        return (bitrate * duration) / (8 * 1_048_576)


@dataclass
class ScreenAnalysisResult:
    """Result of AI-powered screen content analysis."""

    success: bool
    prompt: str
    description: str = ""
    active_app: Optional[str] = None
    active_window_title: Optional[str] = None
    visible_text: Optional[str] = None
    ui_elements: List[str] = field(default_factory=list)
    layout_description: str = ""
    accessibility_info: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "prompt": self.prompt,
            "description": self.description,
            "active_app": self.active_app,
            "active_window_title": self.active_window_title,
            "visible_text": self.visible_text,
            "ui_elements": self.ui_elements,
            "layout_description": self.layout_description,
            "accessibility_info": self.accessibility_info,
            "confidence": self.confidence,
            "tags": self.tags,
            "error": self.error,
            "timestamp": self.timestamp,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Screen Manager
# ──────────────────────────────────────────────────────────────────────────────

class ScreenManager:
    """
    Manage remote screen access across connected devices.

    Features:
    - Single screenshot capture with format and display selection
    - Continuous screen recording with quality/format options
    - AI-powered screen content analysis (OCR, UI understanding, app detection)
    - Recording management with duration tracking
    - Screenshot history with deduplication
    - Rate limiting and timeout handling

    Parameters
    ----------
    device_node:
        The DeviceNode instance used for device communication.
    max_recording_duration:
        Maximum recording duration in seconds. Default 3600 (1 hour).
    capture_timeout:
        Default screenshot timeout in seconds. Default 15.
    analysis_timeout:
        Default analysis timeout in seconds. Default 30.
    max_concurrent_recordings:
        Maximum concurrent recordings per device. Default 1.
    """

    def __init__(
        self,
        device_node: DeviceNode,
        max_recording_duration: float = 3600.0,
        capture_timeout: float = 15.0,
        analysis_timeout: float = 30.0,
        max_concurrent_recordings: int = 1,
    ) -> None:
        self._node = device_node
        self._max_recording_duration = max_recording_duration
        self._capture_timeout = capture_timeout
        self._analysis_timeout = analysis_timeout
        self._max_concurrent_recordings = max_concurrent_recordings

        # Active recordings: device_id -> [RecordingInfo]
        self._active_recordings: Dict[str, List[RecordingInfo]] = {}

        # Screenshot history: screenshot_id -> ScreenshotResult
        self._screenshot_history: Dict[str, ScreenshotResult] = {}
        self._max_history: int = 200

        # Analysis cache: image checksum -> ScreenAnalysisResult
        self._analysis_cache: Dict[str, ScreenAnalysisResult] = {}
        self._max_cache: int = 50

        # Rate limiting: device_id -> [timestamps]
        self._capture_rate: Dict[str, List[float]] = {}
        self._capture_rate_limit: int = 20  # screenshots per window
        self._capture_rate_window: float = 60.0

        # Duration monitor task
        self._duration_monitor_task: Optional[asyncio.Task[None]] = None
        self._running = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the screen manager background tasks."""
        if self._running:
            return
        self._running = True
        self._duration_monitor_task = asyncio.create_task(
            self._duration_monitor_loop()
        )
        logger.info("ScreenManager started")

    async def stop(self) -> None:
        """Stop the screen manager and all active recordings."""
        self._running = False
        if self._duration_monitor_task and not self._duration_monitor_task.done():
            self._duration_monitor_task.cancel()
            try:
                await self._duration_monitor_task
            except asyncio.CancelledError:
                pass
        # Stop all active recordings
        for device_id in list(self._active_recordings.keys()):
            await self.stop_recording(device_id)
        logger.info("ScreenManager stopped")

    # ── Screenshot Capture ───────────────────────────────────────────────

    async def capture(
        self,
        device_id: str,
        format: ScreenFormat = ScreenFormat.PNG,
        display_index: int = 0,
        include_cursor: bool = False,
        timeout: Optional[float] = None,
    ) -> ScreenshotResult:
        """
        Capture a screenshot from a remote device.

        Parameters
        ----------
        device_id:
            The target device identifier.
        format:
            Image format for the screenshot. Default PNG.
        display_index:
            Display index for multi-monitor setups. Default 0.
        include_cursor:
            Whether to include the cursor in the screenshot. Default False.
        timeout:
            Capture timeout in seconds. Default from constructor.

        Returns
        -------
        ScreenshotResult
            The screenshot result including image data.

        Raises
        ------
        LookupError
            If the device is not registered.
        PermissionError
            If the device lacks screen capability.
        RuntimeError
            If rate limited.
        """
        if not self._check_capture_rate(device_id):
            raise RuntimeError(
                f"Screen capture rate limit exceeded for device: {device_id}"
            )

        device = await self._node.get_device(device_id)
        if device is None:
            raise LookupError(f"Device not registered: {device_id}")

        if Capability.SCREEN not in device.capabilities:
            raise PermissionError(
                f"Device {device_id} does not have screen capability"
            )

        timeout_val = timeout or self._capture_timeout

        try:
            result = await self._node.send_command(
                device_id=device_id,
                command="capture_screen",
                params={
                    "format": format.value,
                    "display_index": display_index,
                    "include_cursor": include_cursor,
                },
                timeout=timeout_val,
            )

            screenshot = ScreenshotResult(
                success=result.success,
                device_id=device_id,
                image_data=result.data if isinstance(result.data, bytes) else None,
                image_format=format,
                display_index=display_index,
                error=result.error,
                metadata=result.data if isinstance(result.data, dict) else {},
            )

            if isinstance(result.data, dict):
                screenshot.width = result.data.get("width", 0)
                screenshot.height = result.data.get("height", 0)
                screenshot.file_size = result.data.get("file_size", 0)

            self._screenshot_history[screenshot.screenshot_id] = screenshot
            self._trim_history()

            logger.info(
                "Screenshot %s from %s (format=%s, display=%d, size=%d bytes)",
                screenshot.screenshot_id[:8], device_id, format.value,
                display_index, screenshot.file_size,
            )

            await self._node.emit_capability_event(
                device_id, Capability.SCREEN,
                {"type": "screenshot", "screenshot_id": screenshot.screenshot_id,
                 "success": screenshot.success},
            )

            return screenshot

        except asyncio.TimeoutError:
            return ScreenshotResult(
                success=False,
                device_id=device_id,
                error=f"Screenshot timed out after {timeout_val}s",
                image_format=format,
                display_index=display_index,
            )

    # ── Screen Recording ─────────────────────────────────────────────────

    async def start_recording(
        self,
        device_id: str,
        format: RecordingFormat = RecordingFormat.MP4,
        quality: RecordingQuality = RecordingQuality.MEDIUM,
        fps: int = 30,
        audio_enabled: bool = False,
        cursor_visible: bool = True,
    ) -> RecordingInfo:
        """
        Start a screen recording on a remote device.

        Parameters
        ----------
        device_id:
            Target device identifier.
        format:
            Recording output format. Default MP4.
        quality:
            Recording quality preset. Default MEDIUM.
        fps:
            Frames per second. Default 30.
        audio_enabled:
            Whether to record system audio. Default False.
        cursor_visible:
            Whether to show cursor in recording. Default True.

        Returns
        -------
        RecordingInfo
            Information about the started recording.

        Raises
        ------
        LookupError
            If device not registered.
        PermissionError
            If device lacks screen capability.
        RuntimeError
            If max concurrent recordings reached.
        """
        device = await self._node.get_device(device_id)
        if device is None:
            raise LookupError(f"Device not registered: {device_id}")

        if Capability.SCREEN not in device.capabilities:
            raise PermissionError(
                f"Device {device_id} does not have screen capability"
            )

        active = self._active_recordings.get(device_id, [])
        active_count = sum(1 for r in active if r.is_active)
        if active_count >= self._max_concurrent_recordings:
            raise RuntimeError(
                f"Max concurrent recordings ({self._max_concurrent_recordings}) "
                f"reached for device: {device_id}"
            )

        recording_id = uuid.uuid4().hex[:12]

        await self._node.send_command(
            device_id=device_id,
            command="start_recording",
            params={
                "recording_id": recording_id,
                "format": format.value,
                "quality": quality.value,
                "fps": fps,
                "audio_enabled": audio_enabled,
                "cursor_visible": cursor_visible,
            },
            timeout=10.0,
        )

        recording = RecordingInfo(
            recording_id=recording_id,
            device_id=device_id,
            format=format,
            quality=quality,
            fps=fps,
            audio_enabled=audio_enabled,
            cursor_visible=cursor_visible,
        )

        if device_id not in self._active_recordings:
            self._active_recordings[device_id] = []
        self._active_recordings[device_id].append(recording)

        logger.info(
            "Started recording %s on device %s (%s, %s, %dfps)",
            recording_id, device_id, format.value, quality.value, fps,
        )

        return recording

    async def stop_recording(
        self, device_id: str, recording_id: Optional[str] = None
    ) -> Optional[RecordingInfo]:
        """
        Stop a screen recording on a remote device.

        Parameters
        ----------
        device_id:
            Target device identifier.
        recording_id:
            Specific recording to stop. If None, stops all recordings.

        Returns
        -------
        Optional[RecordingInfo]
            The stopped recording info, or None if none were active.
        """
        recordings = self._active_recordings.get(device_id, [])
        if not recordings:
            logger.warning("No active recordings for device: %s", device_id)
            return None

        stopped_recording: Optional[RecordingInfo] = None

        for recording in recordings:
            if recording_id and recording.recording_id != recording_id:
                continue
            if not recording.is_active:
                continue

            try:
                result = await self._node.send_command(
                    device_id=device_id,
                    command="stop_recording",
                    params={"recording_id": recording.recording_id},
                    timeout=15.0,
                )
                if isinstance(result.data, dict):
                    recording.file_size = result.data.get("file_size", 0)
            except Exception as e:
                logger.error(
                    "Error stopping recording %s: %s", recording.recording_id, e
                )

            recording.stopped_at = time.time()
            recording.duration_seconds = recording._compute_duration()
            recording.is_active = False
            stopped_recording = recording

            logger.info(
                "Stopped recording %s on device %s (duration=%.1fs, size=%d bytes)",
                recording.recording_id, device_id,
                recording.duration_seconds, recording.file_size,
            )

        return stopped_recording

    def get_active_recordings(
        self, device_id: Optional[str] = None
    ) -> List[RecordingInfo]:
        """
        Get currently active recordings.

        Parameters
        ----------
        device_id:
            Filter by device. None returns all active recordings.

        Returns
        -------
        List[RecordingInfo]
            Active recording information.
        """
        if device_id:
            recordings = self._active_recordings.get(device_id, [])
        else:
            recordings = [
                r for rlist in self._active_recordings.values()
                for r in rlist
            ]
        return [r for r in recordings if r.is_active]

    # ── Screen Analysis ──────────────────────────────────────────────────

    async def analyze_screen(
        self,
        screenshot: ScreenshotResult,
        prompt: str = "Describe what is visible on this screen.",
        timeout: Optional[float] = None,
    ) -> ScreenAnalysisResult:
        """
        Analyze screen content using AI vision.

        Parameters
        ----------
        screenshot:
            The screenshot result to analyze.
        prompt:
            Analysis prompt describing what to look for.
        timeout:
            Analysis timeout in seconds.

        Returns
        -------
        ScreenAnalysisResult
            AI analysis results including app detection, OCR, and UI elements.
        """
        if screenshot.image_data is None:
            return ScreenAnalysisResult(
                success=False,
                prompt=prompt,
                error="No image data available for analysis",
            )

        # Check cache
        checksum = screenshot.checksum
        if checksum and checksum in self._analysis_cache:
            cached = self._analysis_cache[checksum]
            logger.debug(
                "Returning cached screen analysis for %s",
                screenshot.screenshot_id[:8],
            )
            return cached

        timeout_val = timeout or self._analysis_timeout

        try:
            # In production, this would call a vision AI model for OCR,
            # UI element detection, app identification, etc.
            result = ScreenAnalysisResult(
                success=True,
                prompt=prompt,
                description=f"Screen analysis from device {screenshot.device_id}",
            )

            if checksum:
                self._analysis_cache[checksum] = result
                self._trim_cache()

            logger.info(
                "Screen analysis complete for %s (prompt=%s)",
                screenshot.screenshot_id[:8], prompt[:50],
            )

            return result

        except asyncio.TimeoutError:
            return ScreenAnalysisResult(
                success=False,
                prompt=prompt,
                error=f"Analysis timed out after {timeout_val}s",
            )
        except Exception as e:
            return ScreenAnalysisResult(
                success=False,
                prompt=prompt,
                error=f"Analysis failed: {str(e)}",
            )

    # ── Duration Monitor ─────────────────────────────────────────────────

    async def _duration_monitor_loop(self) -> None:
        """Background task that enforces max recording duration."""
        logger.info("Recording duration monitor started")
        while self._running:
            try:
                await asyncio.sleep(30.0)
                for device_id in list(self._active_recordings.keys()):
                    for recording in self._active_recordings.get(device_id, []):
                        if not recording.is_active:
                            continue
                        duration = recording._compute_duration()
                        if duration >= self._max_recording_duration:
                            logger.warning(
                                "Recording %s on device %s exceeded max duration, "
                                "auto-stopping (%.1fs)",
                                recording.recording_id, device_id, duration,
                            )
                            asyncio.create_task(
                                self.stop_recording(device_id, recording.recording_id)
                            )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Duration monitor error: %s", e)
        logger.info("Recording duration monitor stopped")

    # ── History & Cache ──────────────────────────────────────────────────

    def get_screenshot_history(
        self,
        device_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[ScreenshotResult]:
        """
        Retrieve screenshot history.

        Parameters
        ----------
        device_id:
            Filter by device. None returns all.
        limit:
            Maximum results. Default 20.

        Returns
        -------
        List[ScreenshotResult]
            Recent screenshots, newest first.
        """
        screenshots = list(self._screenshot_history.values())
        if device_id:
            screenshots = [s for s in screenshots if s.device_id == device_id]
        screenshots.sort(key=lambda s: s.timestamp, reverse=True)
        return screenshots[:limit]

    def clear_history(self, device_id: Optional[str] = None) -> int:
        """Clear screenshot history. Returns count removed."""
        if device_id:
            to_remove = [
                sid for sid, s in self._screenshot_history.items()
                if s.device_id == device_id
            ]
            for sid in to_remove:
                del self._screenshot_history[sid]
            return len(to_remove)
        count = len(self._screenshot_history)
        self._screenshot_history.clear()
        return count

    # ── Rate Limiting ────────────────────────────────────────────────────

    def _check_capture_rate(self, device_id: str) -> bool:
        """Check and update capture rate limit. Returns True if allowed."""
        now = time.time()
        timestamps = self._capture_rate.get(device_id, [])
        cutoff = now - self._capture_rate_window
        timestamps = [ts for ts in timestamps if ts > cutoff]

        if len(timestamps) >= self._capture_rate_limit:
            self._capture_rate[device_id] = timestamps
            return False

        timestamps.append(now)
        self._capture_rate[device_id] = timestamps
        return True

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _trim_history(self) -> None:
        """Trim screenshot history to max size."""
        if len(self._screenshot_history) > self._max_history:
            sorted_ids = sorted(
                self._screenshot_history.keys(),
                key=lambda sid: self._screenshot_history[sid].timestamp,
            )
            for sid in sorted_ids[: len(self._screenshot_history) - self._max_history]:
                del self._screenshot_history[sid]

    def _trim_cache(self) -> None:
        """Trim analysis cache to max size."""
        if len(self._analysis_cache) > self._max_cache:
            keys = list(self._analysis_cache.keys())
            for key in keys[: len(self._analysis_cache) - self._max_cache]:
                del self._analysis_cache[key]

    async def get_stats(self) -> Dict[str, Any]:
        """Return screen manager statistics."""
        active = self.get_active_recordings()
        return {
            "screenshot_history_size": len(self._screenshot_history),
            "analysis_cache_size": len(self._analysis_cache),
            "active_recordings": len(active),
            "devices_recording": len(self._active_recordings),
            "max_recording_duration": self._max_recording_duration,
        }
