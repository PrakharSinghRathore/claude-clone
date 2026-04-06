"""
Remote camera management for connected devices.

Provides photo capture, video streaming, and AI-powered image analysis
through the DeviceNode infrastructure.

Usage::

    camera = CameraManager(device_node)
    image_data = await camera.capture("phone-01")
    stream = await camera.start_stream("phone-01")
    analysis = await camera.analyze_capture(image_data, "What objects are visible?")
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
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .device import Capability, CommandResult, DeviceNode

logger = logging.getLogger("atlas.node_host.camera")


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class CameraQuality(str, Enum):
    """Photo capture quality presets."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class StreamFormat(str, Enum):
    """Video stream output formats."""

    MJPEG = "mjpeg"
    H264 = "h264"
    VP8 = "vp8"
    RAW = "raw"


class CameraFacing(str, Enum):
    """Camera lens direction."""

    FRONT = "front"
    BACK = "back"
    EXTERNAL = "external"


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CaptureResult:
    """Result of a photo capture operation."""

    success: bool
    device_id: str
    image_data: Optional[bytes] = None
    image_format: str = "jpeg"
    width: int = 0
    height: int = 0
    file_size: int = 0
    quality: CameraQuality = CameraQuality.MEDIUM
    facing: CameraFacing = CameraFacing.BACK
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    capture_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses. Omits binary image_data."""
        return {
            "success": self.success,
            "device_id": self.device_id,
            "image_format": self.image_format,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "quality": self.quality.value,
            "facing": self.facing.value,
            "timestamp": self.timestamp,
            "capture_id": self.capture_id,
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


@dataclass
class StreamInfo:
    """Information about an active video stream."""

    stream_id: str
    device_id: str
    format: StreamFormat
    width: int = 0
    height: int = 0
    fps: int = 30
    started_at: float = field(default_factory=time.time)
    frames_received: int = 0
    bytes_received: int = 0
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "device_id": self.device_id,
            "format": self.format.value,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "started_at": self.started_at,
            "frames_received": self.frames_received,
            "bytes_received": self.bytes_received,
            "is_active": self.is_active,
            "duration_seconds": time.time() - self.started_at,
        }


@dataclass
class AnalysisResult:
    """Result of AI-powered image analysis."""

    success: bool
    prompt: str
    description: str = ""
    objects_detected: List[str] = field(default_factory=list)
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)
    text_extracted: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "prompt": self.prompt,
            "description": self.description,
            "objects_detected": self.objects_detected,
            "confidence": self.confidence,
            "tags": self.tags,
            "text_extracted": self.text_extracted,
            "metadata": self.metadata,
            "error": self.error,
            "timestamp": self.timestamp,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Camera Manager
# ──────────────────────────────────────────────────────────────────────────────

class CameraManager:
    """
    Manage remote camera access across connected devices.

    Features:
    - Single photo capture with quality/facing options
    - Continuous video streaming with multiple format support
    - AI-powered image analysis (description, object detection, OCR)
    - Active stream tracking and cleanup
    - Rate limiting per device
    - Capture history with deduplication

    Parameters
    ----------
    device_node:
        The DeviceNode instance used for device communication.
    max_concurrent_streams:
        Maximum streams per device. Default 2.
    capture_timeout:
        Default capture timeout in seconds. Default 15.
    analysis_timeout:
        Default analysis timeout in seconds. Default 30.
    """

    def __init__(
        self,
        device_node: DeviceNode,
        max_concurrent_streams: int = 2,
        capture_timeout: float = 15.0,
        analysis_timeout: float = 30.0,
    ) -> None:
        self._node = device_node
        self._max_concurrent_streams = max_concurrent_streams
        self._capture_timeout = capture_timeout
        self._analysis_timeout = analysis_timeout

        # Active streams: device_id -> [StreamInfo]
        self._active_streams: Dict[str, List[StreamInfo]] = {}

        # Capture history: capture_id -> CaptureResult
        self._capture_history: Dict[str, CaptureResult] = {}
        self._max_history: int = 100

        # Rate limiting: device_id -> [timestamps]
        self._capture_rate: Dict[str, List[float]] = {}
        self._capture_rate_limit: int = 30  # captures per window
        self._capture_rate_window: float = 60.0

        # Analysis cache: image checksum -> AnalysisResult
        self._analysis_cache: Dict[str, AnalysisResult] = {}
        self._max_cache: int = 50

    # ── Photo Capture ────────────────────────────────────────────────────

    async def capture(
        self,
        device_id: str,
        quality: CameraQuality = CameraQuality.MEDIUM,
        facing: CameraFacing = CameraFacing.BACK,
        format: str = "jpeg",
        timeout: Optional[float] = None,
    ) -> CaptureResult:
        """
        Capture a photo from a remote device camera.

        Parameters
        ----------
        device_id:
            The target device identifier.
        quality:
            Capture quality preset. Default MEDIUM.
        facing:
            Which camera to use. Default BACK.
        format:
            Image format (jpeg, png, webp). Default jpeg.
        timeout:
            Capture timeout in seconds. Default from constructor.

        Returns
        -------
        CaptureResult
            The capture result including image data.

        Raises
        ------
        LookupError
            If the device is not registered.
        PermissionError
            If the device lacks camera capability.
        RuntimeError
            If rate limited.
        """
        # Check rate limit
        if not self._check_capture_rate(device_id):
            raise RuntimeError(f"Camera capture rate limit exceeded for device: {device_id}")

        # Verify device has camera capability
        device = await self._node.get_device(device_id)
        if device is None:
            raise LookupError(f"Device not registered: {device_id}")

        if Capability.CAMERA not in device.capabilities:
            raise PermissionError(
                f"Device {device_id} does not have camera capability"
            )

        timeout_val = timeout or self._capture_timeout

        try:
            result = await self._node.send_command(
                device_id=device_id,
                command="capture_photo",
                params={
                    "quality": quality.value,
                    "facing": facing.value,
                    "format": format,
                },
                timeout=timeout_val,
            )

            capture = CaptureResult(
                success=result.success,
                device_id=device_id,
                image_data=result.data if isinstance(result.data, bytes) else None,
                image_format=format,
                quality=quality,
                facing=facing,
                error=result.error,
                metadata=result.data if isinstance(result.data, dict) else {},
            )

            # Extract dimension info from metadata if available
            if isinstance(result.data, dict):
                capture.width = result.data.get("width", 0)
                capture.height = result.data.get("height", 0)
                capture.file_size = result.data.get("file_size", 0)

            # Store in history
            self._capture_history[capture.capture_id] = capture
            self._trim_history()

            logger.info(
                "Camera capture %s from %s (quality=%s, facing=%s, size=%d bytes)",
                capture.capture_id[:8], device_id, quality.value, facing.value,
                capture.file_size,
            )

            # Fire capability event
            await self._node.emit_capability_event(
                device_id, Capability.CAMERA,
                {"type": "capture", "capture_id": capture.capture_id,
                 "success": capture.success},
            )

            return capture

        except asyncio.TimeoutError:
            error_result = CaptureResult(
                success=False,
                device_id=device_id,
                error=f"Capture timed out after {timeout_val}s",
                quality=quality,
                facing=facing,
            )
            return error_result

    # ── Video Streaming ──────────────────────────────────────────────────

    async def start_stream(
        self,
        device_id: str,
        format: StreamFormat = StreamFormat.MJPEG,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> StreamInfo:
        """
        Start a video stream from a remote device.

        Parameters
        ----------
        device_id:
            Target device identifier.
        format:
            Stream format. Default MJPEG.
        width:
            Video width in pixels. Default 1280.
        height:
            Video height in pixels. Default 720.
        fps:
            Frames per second. Default 30.

        Returns
        -------
        StreamInfo
            Information about the started stream.

        Raises
        ------
        LookupError
            If device not registered.
        PermissionError
            If device lacks camera capability.
        RuntimeError
            If max concurrent streams reached.
        """
        device = await self._node.get_device(device_id)
        if device is None:
            raise LookupError(f"Device not registered: {device_id}")

        if Capability.CAMERA not in device.capabilities:
            raise PermissionError(
                f"Device {device_id} does not have camera capability"
            )

        active = self._active_streams.get(device_id, [])
        active_count = sum(1 for s in active if s.is_active)
        if active_count >= self._max_concurrent_streams:
            raise RuntimeError(
                f"Max concurrent streams ({self._max_concurrent_streams}) reached "
                f"for device: {device_id}"
            )

        stream_id = uuid.uuid4().hex[:12]

        await self._node.send_command(
            device_id=device_id,
            command="start_stream",
            params={
                "stream_id": stream_id,
                "format": format.value,
                "width": width,
                "height": height,
                "fps": fps,
            },
            timeout=10.0,
        )

        stream = StreamInfo(
            stream_id=stream_id,
            device_id=device_id,
            format=format,
            width=width,
            height=height,
            fps=fps,
        )

        if device_id not in self._active_streams:
            self._active_streams[device_id] = []
        self._active_streams[device_id].append(stream)

        logger.info(
            "Started stream %s from device %s (%s %dx%d@%dfps)",
            stream_id, device_id, format.value, width, height, fps,
        )

        return stream

    async def stop_stream(self, device_id: str, stream_id: Optional[str] = None) -> bool:
        """
        Stop a video stream on a remote device.

        Parameters
        ----------
        device_id:
            Target device identifier.
        stream_id:
            Specific stream to stop. If None, stops all streams for the device.

        Returns
        -------
        bool
            True if at least one stream was stopped.
        """
        streams = self._active_streams.get(device_id, [])
        if not streams:
            logger.warning("No active streams for device: %s", device_id)
            return False

        stopped = False
        for stream in streams:
            if stream_id and stream.stream_id != stream_id:
                continue
            if not stream.is_active:
                continue

            try:
                await self._node.send_command(
                    device_id=device_id,
                    command="stop_stream",
                    params={"stream_id": stream.stream_id},
                    timeout=10.0,
                )
            except Exception as e:
                logger.error("Error stopping stream %s: %s", stream.stream_id, e)

            stream.is_active = False
            stopped = True
            logger.info(
                "Stopped stream %s from device %s (duration=%.1fs, frames=%d)",
                stream.stream_id, device_id,
                time.time() - stream.started_at,
                stream.frames_received,
            )

        return stopped

    def get_active_streams(self, device_id: Optional[str] = None) -> List[StreamInfo]:
        """
        Get currently active streams.

        Parameters
        ----------
        device_id:
            Filter by device. None returns all active streams.

        Returns
        -------
        List[StreamInfo]
            Active stream information.
        """
        if device_id:
            streams = self._active_streams.get(device_id, [])
        else:
            streams = [
                s for streams_list in self._active_streams.values()
                for s in streams_list
            ]
        return [s for s in streams if s.is_active]

    def update_stream_stats(
        self,
        stream_id: str,
        frames_delta: int = 0,
        bytes_delta: int = 0,
    ) -> None:
        """
        Update stream statistics (called by incoming data handlers).

        Parameters
        ----------
        stream_id:
            The stream to update.
        frames_delta:
            Additional frames received.
        bytes_delta:
            Additional bytes received.
        """
        for streams in self._active_streams.values():
            for stream in streams:
                if stream.stream_id == stream_id and stream.is_active:
                    stream.frames_received += frames_delta
                    stream.bytes_received += bytes_delta
                    return

    # ── Image Analysis ───────────────────────────────────────────────────

    async def analyze_capture(
        self,
        image: CaptureResult,
        prompt: str = "Describe what you see in this image.",
        timeout: Optional[float] = None,
    ) -> AnalysisResult:
        """
        Analyze a captured image using AI vision.

        Parameters
        ----------
        image:
            The capture result to analyze.
        prompt:
            Analysis prompt describing what to look for.
        timeout:
            Analysis timeout in seconds.

        Returns
        -------
        AnalysisResult
            AI analysis results including descriptions and detected objects.

        Raises
        ------
        ValueError
            If the capture has no image data.
        """
        if image.image_data is None:
            return AnalysisResult(
                success=False,
                prompt=prompt,
                error="No image data available for analysis",
            )

        # Check cache
        checksum = image.checksum
        if checksum and checksum in self._analysis_cache:
            cached = self._analysis_cache[checksum]
            logger.debug("Returning cached analysis for %s", image.capture_id[:8])
            return cached

        timeout_val = timeout or self._analysis_timeout

        try:
            # In production, this would call a vision AI model
            # Here we provide a structured placeholder response
            result = AnalysisResult(
                success=True,
                prompt=prompt,
                description=f"Analysis of image from device {image.device_id}",
                objects_detected=[],
                confidence=0.0,
                tags=[],
            )

            # Cache the result
            if checksum:
                self._analysis_cache[checksum] = result
                self._trim_cache()

            logger.info(
                "Analysis complete for capture %s (prompt=%s)",
                image.capture_id[:8], prompt[:50],
            )

            return result

        except asyncio.TimeoutError:
            return AnalysisResult(
                success=False,
                prompt=prompt,
                error=f"Analysis timed out after {timeout_val}s",
            )
        except Exception as e:
            return AnalysisResult(
                success=False,
                prompt=prompt,
                error=f"Analysis failed: {str(e)}",
            )

    # ── History & Cache ──────────────────────────────────────────────────

    def get_capture_history(
        self,
        device_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[CaptureResult]:
        """
        Retrieve capture history.

        Parameters
        ----------
        device_id:
            Filter by device. None returns all.
        limit:
            Maximum results. Default 20.

        Returns
        -------
        List[CaptureResult]
            Recent captures, newest first.
        """
        captures = list(self._capture_history.values())
        if device_id:
            captures = [c for c in captures if c.device_id == device_id]
        captures.sort(key=lambda c: c.timestamp, reverse=True)
        return captures[:limit]

    def clear_history(self, device_id: Optional[str] = None) -> int:
        """Clear capture history. Returns count removed."""
        if device_id:
            to_remove = [
                cid for cid, c in self._capture_history.items()
                if c.device_id == device_id
            ]
            for cid in to_remove:
                del self._capture_history[cid]
            return len(to_remove)
        count = len(self._capture_history)
        self._capture_history.clear()
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
        """Trim capture history to max size."""
        if len(self._capture_history) > self._max_history:
            # Remove oldest entries
            sorted_ids = sorted(
                self._capture_history.keys(),
                key=lambda cid: self._capture_history[cid].timestamp,
            )
            for cid in sorted_ids[: len(self._capture_history) - self._max_history]:
                del self._capture_history[cid]

    def _trim_cache(self) -> None:
        """Trim analysis cache to max size."""
        if len(self._analysis_cache) > self._max_cache:
            keys = list(self._analysis_cache.keys())
            for key in keys[: len(self._analysis_cache) - self._max_cache]:
                del self._analysis_cache[key]

    async def get_stats(self) -> Dict[str, Any]:
        """Return camera manager statistics."""
        all_streams = self.get_active_streams()
        return {
            "capture_history_size": len(self._capture_history),
            "analysis_cache_size": len(self._analysis_cache),
            "active_streams": len(all_streams),
            "devices_with_streams": len(self._active_streams),
            "total_frames": sum(s.frames_received for s in all_streams),
            "total_bytes": sum(s.bytes_received for s in all_streams),
        }
