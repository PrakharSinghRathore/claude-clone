"""
Atlas Media Processing Pipeline.

A comprehensive media processing framework inspired by OpenClaw's media pipeline,
providing image, audio, video processing, vision analysis, and AI generation
capabilities with multi-provider support and graceful fallbacks.
"""

from .pipeline import MediaPipeline, PipelineStage, PipelineResult
from .images import ImageProcessor, ImageAnalysis
from .audio import AudioProcessor, AudioInfo, WaveformData
from .video import VideoProcessor, VideoInfo
from .vision import VisionAnalyzer, ObjectDetection, FaceDetection, TextExtraction
from .generation import (
    ImageGenerator,
    VideoGenerator,
    MusicGenerator,
    ImageGenerationResult,
    VideoGenerationResult,
    MusicGenerationResult,
)

__all__ = [
    # Pipeline
    "MediaPipeline",
    "PipelineStage",
    "PipelineResult",
    # Images
    "ImageProcessor",
    "ImageAnalysis",
    # Audio
    "AudioProcessor",
    "AudioInfo",
    "WaveformData",
    # Video
    "VideoProcessor",
    "VideoInfo",
    # Vision
    "VisionAnalyzer",
    "ObjectDetection",
    "FaceDetection",
    "TextExtraction",
    # Generation
    "ImageGenerator",
    "VideoGenerator",
    "MusicGenerator",
    "ImageGenerationResult",
    "VideoGenerationResult",
    "MusicGenerationResult",
]
