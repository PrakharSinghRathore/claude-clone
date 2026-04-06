"""
atlas.realtime - Real-time voice and transcription services.

Provides real-time voice conversation support with full-duplex
communication, multiple TTS/STT provider support, and live
audio transcription with automatic failover.
"""

from atlas.realtime.voice import VoiceMode
from atlas.realtime.transcription import (
    TranscriptionProvider,
    TranscriptionResult,
    TranscriptionSegment,
    RealtimeTranscriber,
)

__all__ = [
    "VoiceMode",
    "TranscriptionProvider",
    "TranscriptionResult",
    "TranscriptionSegment",
    "RealtimeTranscriber",
]
