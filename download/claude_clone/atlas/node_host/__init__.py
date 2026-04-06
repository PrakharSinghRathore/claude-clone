"""
Atlas Node Host — remote device management with camera and screen access.

This module provides infrastructure for managing remote device connections,
including device registration, capability negotiation, heartbeat monitoring,
camera capture/streaming, and screen capture/recording.

Classes
-------
DeviceNode
    Central device node for managing remote device connections.
CameraManager
    Remote camera access: photo capture, video streaming, image analysis.
ScreenManager
    Remote screen access: screenshots, recording, screen analysis.
"""

from .camera import (
    AnalysisResult,
    CameraFacing,
    CameraManager,
    CameraQuality,
    CaptureResult,
    StreamFormat,
    StreamInfo,
)
from .device import (
    Capability,
    CommandResult,
    DeviceInfo,
    DeviceNode,
    DeviceStatus,
    DiscoveredDevice,
)
from .screen import (
    RecordingFormat,
    RecordingInfo,
    RecordingQuality,
    ScreenAnalysisResult,
    ScreenFormat,
    ScreenManager,
    ScreenshotResult,
)

__all__ = [
    # Device
    "DeviceNode",
    "DeviceInfo",
    "DeviceStatus",
    "Capability",
    "CommandResult",
    "DiscoveredDevice",
    # Camera
    "CameraManager",
    "CameraQuality",
    "CameraFacing",
    "StreamFormat",
    "CaptureResult",
    "StreamInfo",
    "AnalysisResult",
    # Screen
    "ScreenManager",
    "ScreenFormat",
    "ScreenshotResult",
    "RecordingFormat",
    "RecordingQuality",
    "RecordingInfo",
    "ScreenAnalysisResult",
]
