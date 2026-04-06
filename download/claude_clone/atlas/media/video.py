"""
Atlas Video Processor — Comprehensive video processing tools.

Provides video metadata extraction, audio extraction, frame extraction,
video creation from images, concatenation, trimming, subtitle addition,
compression, and thumbnail generation. Uses FFmpeg for all operations.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class VideoCodec(Enum):
    """Supported video codecs."""
    H264 = "libx264"
    H265 = "libx265"
    VP8 = "libvpx"
    VP9 = "libvpx-vp9"
    AV1 = "libaom-av1"
    MPEG4 = "mpeg4"
    COPY = "copy"


class AudioCodec(Enum):
    """Supported audio codecs."""
    AAC = "aac"
    MP3 = "libmp3lame"
    OPUS = "libopus"
    VORBIS = "libvorbis"
    FLAC = "flac"
    COPY = "copy"
    NONE = "anull"


class VideoProcessingError(Exception):
    """Raised when video processing fails."""
    pass


class FFmpegNotFoundError(Exception):
    """Raised when FFmpeg is required but not found."""
    pass


@dataclass
class VideoInfo:
    """Metadata about a video file."""
    path: Optional[Path] = None
    format: str = ""
    container: str = ""
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    video_codec: str = ""
    audio_codec: str = ""
    audio_sample_rate: int = 0
    audio_channels: int = 0
    bit_rate: int = 0
    file_size: int = 0
    has_audio: bool = False
    has_subtitle: bool = False
    rotation: int = 0
    pixel_format: str = ""
    aspect_ratio: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path) if self.path else None,
            "format": self.format,
            "container": self.container,
            "duration_seconds": round(self.duration_seconds, 3),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "bit_rate": self.bit_rate,
            "file_size": self.file_size,
            "has_audio": self.has_audio,
            "has_subtitle": self.has_subtitle,
            "rotation": self.rotation,
            "pixel_format": self.pixel_format,
            "aspect_ratio": round(self.aspect_ratio, 4),
        }

    @property
    def resolution(self) -> str:
        """Human-readable resolution string."""
        if self.width and self.height:
            # Map common resolutions to names
            res_map = {
                (3840, 2160): "4K UHD",
                (2560, 1440): "2K QHD",
                (1920, 1080): "1080p Full HD",
                (1280, 720): "720p HD",
                (854, 480): "480p SD",
                (640, 360): "360p",
                (426, 240): "240p",
            }
            return res_map.get(
                (self.width, self.height),
                f"{self.width}x{self.height}",
            )
        return "unknown"

    @property
    def duration_formatted(self) -> str:
        """Human-readable duration string (HH:MM:SS)."""
        total_sec = int(self.duration_seconds)
        hours, remainder = divmod(total_sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"


def _check_ffmpeg() -> bool:
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_ffmpeg(args: List[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Run an FFmpeg command and return the result."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    logger.debug("FFmpeg: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[:500]
            raise VideoProcessingError(f"FFmpeg error: {stderr}")
        return result
    except subprocess.TimeoutExpired:
        raise VideoProcessingError(f"FFmpeg timed out after {timeout}s")
    except FileNotFoundError:
        raise FFmpegNotFoundError("FFmpeg is not installed or not found in PATH")


def _run_ffprobe(path: Path) -> Optional[Dict[str, Any]]:
    """Run ffprobe and return parsed JSON."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", "-show_chapters", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


def _parse_fps(fps_str: str) -> float:
    """Parse FFmpeg frame rate string."""
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            return float(num) / float(den) if float(den) != 0 else 0.0
        return float(fps_str)
    except (ValueError, ZeroDivisionError):
        return 0.0


class VideoProcessor:
    """
    Comprehensive video processing class.

    Provides tools for extracting metadata, audio extraction, frame extraction,
    video creation from images, concatenation, trimming, subtitle addition,
    compression, and thumbnail generation. All operations use FFmpeg.

    Example:
        >>> processor = VideoProcessor()
        >>> info = await processor.get_info("video.mp4")
        >>> await processor.extract_frames("video.mp4", "frames/", fps=5)
        >>> await processor.trim("video.mp4", "clip.mp4", start=10, end=30)
    """

    def __init__(
        self,
        temp_dir: Optional[Union[str, Path]] = None,
        default_codec: VideoCodec = VideoCodec.H264,
        default_audio_codec: AudioCodec = AudioCodec.AAC,
    ) -> None:
        """
        Initialize the VideoProcessor.

        Args:
            temp_dir: Directory for temporary files.
            default_codec: Default video codec for encoding.
            default_audio_codec: Default audio codec for encoding.
        """
        self._temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="atlas_video_"))
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._default_codec = default_codec
        self._default_audio_codec = default_audio_codec
        self._ffmpeg_available = _check_ffmpeg()

        if not self._ffmpeg_available:
            logger.warning("FFmpeg is not available. Video processing is disabled.")

    @property
    def ffmpeg_available(self) -> bool:
        """Whether FFmpeg is available."""
        return self._ffmpeg_available

    @property
    def temp_dir(self) -> Path:
        """The temporary directory path."""
        return self._temp_dir

    def _ensure_ffmpeg(self) -> None:
        """Raise if FFmpeg is not available."""
        if not self._ffmpeg_available:
            raise FFmpegNotFoundError("FFmpeg is required for video processing")

    async def get_info(self, path: Union[str, Path]) -> VideoInfo:
        """
        Get detailed metadata about a video file.

        Args:
            path: Path to the video file.

        Returns:
            VideoInfo dataclass with all extracted metadata.
        """
        path = Path(path)
        if not path.exists():
            raise VideoProcessingError(f"Video file not found: {path}")

        info = VideoInfo(
            path=path,
            format=path.suffix.lstrip("."),
            file_size=path.stat().st_size,
        )

        probe = _run_ffprobe(path)
        if probe:
            # Parse format info
            if "format" in probe:
                fmt = probe["format"]
                info.container = fmt.get("format_name", "")
                try:
                    info.duration_seconds = float(fmt.get("duration", 0))
                except ValueError:
                    pass
                try:
                    info.bit_rate = int(fmt.get("bit_rate", 0))
                except ValueError:
                    pass
                info.has_subtitle = int(fmt.get("nb_streams", 0)) > 0

            # Parse stream info
            for stream in probe.get("streams", []):
                codec_type = stream.get("codec_type", "")

                if codec_type == "video":
                    info.video_codec = stream.get("codec_name", "")
                    info.width = int(stream.get("width", 0))
                    info.height = int(stream.get("height", 0))
                    info.fps = _parse_fps(stream.get("r_frame_rate", "0/1"))
                    info.pixel_format = stream.get("pix_fmt", "")

                    # Check rotation from side_data or tags
                    for sd in stream.get("side_data_list", []):
                        if sd.get("side_data_type") == "Display Matrix":
                            info.rotation = abs(int(sd.get("rotation", 0)))
                    tags = stream.get("tags", {})
                    if not info.rotation and "rotate" in tags:
                        info.rotation = int(tags["rotate"])

                    if info.height > 0:
                        info.aspect_ratio = info.width / info.height

                elif codec_type == "audio":
                    info.has_audio = True
                    info.audio_codec = stream.get("codec_name", "")
                    try:
                        info.audio_sample_rate = int(stream.get("sample_rate", 0))
                    except ValueError:
                        pass
                    try:
                        info.audio_channels = int(stream.get("channels", 0))
                    except ValueError:
                        pass

                elif codec_type == "subtitle":
                    info.has_subtitle = True

        return info

    async def extract_audio(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        format: str = "mp3",
        bitrate: str = "192k",
    ) -> Path:
        """
        Extract the audio track from a video file.

        Args:
            video_path: Path to the video file.
            output_path: Path for the extracted audio.
            format: Audio format (mp3, wav, aac, flac, ogg).
            bitrate: Bitrate for lossy formats.

        Returns:
            Path to the extracted audio file.
        """
        self._ensure_ffmpeg()
        video_path = Path(video_path)
        output_path = Path(output_path)

        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        args = [
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", self._get_audio_codec(format),
        ]

        if format != "wav" and format != "flac":
            args.extend(["-b:a", bitrate])

        args.append(str(output_path))
        _run_ffmpeg(args, timeout=300)

        logger.info("Extracted audio: %s → %s", video_path.name, output_path.name)
        return output_path

    async def extract_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        fps: float = 1.0,
        format: str = "jpg",
        quality: int = 2,
        max_frames: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Path]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the video file.
            output_dir: Directory to save extracted frames.
            fps: Frames per second to extract.
            format: Output image format (jpg, png, webp).
            quality: Image quality (1=best, 31=worst for JPEG).
            max_frames: Maximum number of frames to extract.
            start_time: Start time in seconds (None = from beginning).
            end_time: End time in seconds (None = to end).

        Returns:
            List of paths to extracted frames.
        """
        self._ensure_ffmpeg()
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        args = ["-i", str(video_path)]

        if start_time is not None:
            args.extend(["-ss", str(start_time)])
        if end_time is not None:
            args.extend(["-to", str(end_time)])

        args.extend([
            "-vf", f"fps={fps}",
            "-q:v", str(quality),
        ])

        if max_frames:
            args.extend(["-vframes", str(max_frames)])

        pattern = str(output_dir / f"frame_%04d.{format}")
        args.append(pattern)

        _run_ffmpeg(args, timeout=600)

        frames = sorted(output_dir.glob(f"frame_*.{format}"))
        logger.info("Extracted %d frames from %s", len(frames), video_path.name)
        return frames

    async def create_video(
        self,
        images_dir: Union[str, Path],
        output_path: Union[str, Path],
        fps: float = 24.0,
        codec: Optional[VideoCodec] = None,
        audio_path: Optional[Union[str, Path]] = None,
        duration: Optional[float] = None,
        pattern: str = "*.jpg",
    ) -> Path:
        """
        Create a video from a sequence of images.

        Args:
            images_dir: Directory containing sequential images.
            output_path: Path for the output video.
            fps: Frames per second.
            codec: Video codec. Uses default if None.
            audio_path: Optional audio file to add.
            duration: Fixed duration per image in seconds.
            pattern: Glob pattern for image files.

        Returns:
            Path to the created video.
        """
        self._ensure_ffmpeg()
        images_dir = Path(images_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not images_dir.exists():
            raise VideoProcessingError(f"Images directory not found: {images_dir}")

        images = sorted(images_dir.glob(pattern))
        if not images:
            raise VideoProcessingError(f"No images found matching {pattern} in {images_dir}")

        codec = codec or self._default_codec
        args = [
            "-framerate", str(fps),
            "-i", str(images_dir / f"frame_%04d{images[0].suffix}"),
            "-c:v", codec.value,
            "-pix_fmt", "yuv420p",
        ]

        if duration:
            args.extend(["-t", str(duration * len(images))])

        if audio_path:
            audio_path = Path(audio_path)
            if audio_path.exists():
                args.extend([
                    "-i", str(audio_path),
                    "-c:a", self._default_audio_codec.value,
                    "-shortest",
                ])

        args.append(str(output_path))
        _run_ffmpeg(args, timeout=600)

        logger.info("Created video from %d images: %s", len(images), output_path.name)
        return output_path

    async def concatenate(
        self,
        video_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        codec: Optional[VideoCodec] = None,
    ) -> Path:
        """
        Concatenate multiple video files.

        Args:
            video_paths: List of video file paths to concatenate.
            output_path: Path for the concatenated video.
            codec: Video codec. Uses "copy" if None for fast concat.

        Returns:
            Path to the concatenated video.
        """
        self._ensure_ffmpeg()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not video_paths:
            raise VideoProcessingError("No video files provided for concatenation")

        # Validate all input files
        for vp in video_paths:
            p = Path(vp)
            if not p.exists():
                raise VideoProcessingError(f"Video file not found: {p}")

        codec = codec or VideoCodec.COPY

        # Create concat list file
        concat_file = self._temp_dir / f"concat_{_uuid_hex()}.txt"
        with open(concat_file, "w") as f:
            for vp in video_paths:
                resolved = Path(vp).resolve()
                f.write(f"file '{resolved}'\n")

        args = [
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", codec.value,
            "-c:a", self._default_audio_codec.value if codec != VideoCodec.COPY else "copy",
            str(output_path),
        ]

        _run_ffmpeg(args, timeout=600)
        concat_file.unlink(missing_ok=True)

        logger.info("Concatenated %d videos: %s", len(video_paths), output_path.name)
        return output_path

    async def trim(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        start: float,
        end: float,
        codec: Optional[VideoCodec] = None,
    ) -> Path:
        """
        Trim a video to a time range.

        Args:
            video_path: Path to the source video.
            output_path: Path for the trimmed video.
            start: Start time in seconds.
            end: End time in seconds.
            codec: Video codec. Uses "copy" if None for fast trim.

        Returns:
            Path to the trimmed video.
        """
        self._ensure_ffmpeg()
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        if start < 0 or end <= start:
            raise VideoProcessingError(f"Invalid time range: {start}s - {end}s")

        duration = end - start
        codec = codec or VideoCodec.COPY

        args = [
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", codec.value,
            "-c:a", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]

        _run_ffmpeg(args, timeout=300)

        logger.info("Trimmed %s: %.1fs - %.1fs → %s", video_path.name, start, end, output_path.name)
        return output_path

    async def add_subtitle(
        self,
        video_path: Union[str, Path],
        subtitle_path: Union[str, Path],
        output_path: Union[str, Path],
        encoding: str = "utf-8",
        style: Optional[str] = None,
    ) -> Path:
        """
        Add subtitles to a video (soft subtitles or burned-in).

        Args:
            video_path: Path to the video file.
            subtitle_path: Path to the subtitle file (SRT, ASS, VTT).
            output_path: Path for the output video.
            encoding: Subtitle file encoding.
            style: Optional ASS/SSA style override string.

        Returns:
            Path to the video with subtitles.
        """
        self._ensure_ffmpeg()
        video_path = Path(video_path)
        subtitle_path = Path(subtitle_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")
        if not subtitle_path.exists():
            raise VideoProcessingError(f"Subtitle file not found: {subtitle_path}")

        ext = subtitle_path.suffix.lower()

        if ext in (".srt", ".ass", ".ssa", ".vtt"):
            # Burn-in subtitles using subtitles filter
            args = [
                "-i", str(video_path),
                "-vf", f"subtitles={subtitle_path}:charenc={encoding}",
                "-c:v", self._default_codec.value,
                "-c:a", "copy",
                str(output_path),
            ]

            if style and ext in (".ass", ".ssa"):
                args[3] = f"ass={subtitle_path}"
        else:
            raise VideoProcessingError(f"Unsupported subtitle format: {ext}")

        _run_ffmpeg(args, timeout=600)
        logger.info("Added subtitles: %s → %s", video_path.name, output_path.name)
        return output_path

    async def compress(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        crf: int = 23,
        preset: str = "medium",
        codec: Optional[VideoCodec] = None,
        max_bitrate: Optional[str] = None,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> Path:
        """
        Compress a video file.

        Args:
            video_path: Path to the source video.
            output_path: Path for the compressed video.
            crf: Constant Rate Factor (0=lossless, 23=default, 51=worst).
            preset: Encoding preset (ultrafast, superfast, fast, medium, slow, veryslow).
            codec: Video codec. Defaults to H264.
            max_bitrate: Maximum bitrate (e.g., "1M").
            resolution: Optional (width, height) to scale to.

        Returns:
            Path to the compressed video.
        """
        self._ensure_ffmpeg()
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        codec = codec or VideoCodec.H264
        crf = max(0, min(51, crf))

        args = [
            "-i", str(video_path),
            "-c:v", codec.value,
            "-crf", str(crf),
            "-preset", preset,
            "-c:a", self._default_audio_codec.value,
            "-b:a", "128k",
        ]

        if resolution:
            w, h = resolution
            args.extend(["-vf", f"scale={w}:{h}"])

        if max_bitrate:
            args.extend(["-maxrate", max_bitrate, "-bufsize", "2M"])

        args.append(str(output_path))
        _run_ffmpeg(args, timeout=1200)

        # Calculate compression ratio
        original_size = video_path.stat().st_size
        compressed_size = output_path.stat().st_size if output_path.exists() else 0
        ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

        logger.info(
            "Compressed %s: CRF=%d, ratio=%.1f%% (%d→%d bytes)",
            video_path.name, crf, ratio, original_size, compressed_size,
        )
        return output_path

    async def get_thumbnail(
        self,
        video_path: Union[str, Path],
        time_sec: float = 0.0,
        output_path: Optional[Union[str, Path]] = None,
        width: int = 320,
        height: int = -1,
    ) -> Path:
        """
        Extract a single frame as a thumbnail.

        Args:
            video_path: Path to the video file.
            time_sec: Time in seconds to extract the frame from.
            output_path: Path for the thumbnail. Auto-generated if None.
            width: Thumbnail width. -1 for auto.
            height: Thumbnail height. -1 for auto.

        Returns:
            Path to the thumbnail image.
        """
        self._ensure_ffmpeg()
        video_path = Path(video_path)

        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        if output_path is None:
            output_path = self._temp_dir / f"thumb_{_uuid_hex()}.jpg"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        args = [
            "-i", str(video_path),
            "-ss", str(time_sec),
            "-vframes", "1",
            "-vf", f"scale={width}:{height}",
            "-q:v", "2",
            str(output_path),
        ]

        _run_ffmpeg(args, timeout=60)
        logger.info("Created thumbnail from %s at %.1fs", video_path.name, time_sec)
        return output_path

    async def add_audio(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        replace: bool = False,
    ) -> Path:
        """
        Add or replace audio track in a video.

        Args:
            video_path: Path to the video file.
            audio_path: Path to the audio file.
            output_path: Path for the output video.
            replace: Whether to replace existing audio.

        Returns:
            Path to the video with audio.
        """
        self._ensure_ffmpeg()
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")
        if not audio_path.exists():
            raise VideoProcessingError(f"Audio file not found: {audio_path}")

        args = [
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", self._default_audio_codec.value,
            "-map", "0:v:0",
            "-map", "1:a:0",
        ]

        if replace:
            args.extend(["-shortest"])

        args.append(str(output_path))
        _run_ffmpeg(args, timeout=300)

        logger.info("Added audio to %s", output_path.name)
        return output_path

    async def reverse(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Path:
        """
        Reverse a video.

        Args:
            video_path: Path to the source video.
            output_path: Path for the reversed video.

        Returns:
            Path to the reversed video.
        """
        self._ensure_ffmpeg()
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        _run_ffmpeg([
            "-i", str(video_path),
            "-vf", "reverse",
            "-af", "areverse",
            "-c:v", self._default_codec.value,
            "-c:a", self._default_audio_codec.value,
            str(output_path),
        ], timeout=600)

        logger.info("Reversed video: %s", output_path.name)
        return output_path

    async def speed_change(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        factor: float = 1.0,
    ) -> Path:
        """
        Change video playback speed.

        Args:
            video_path: Path to the source video.
            output_path: Path for the output video.
            factor: Speed multiplier (2.0 = 2x faster, 0.5 = half speed).

        Returns:
            Path to the speed-adjusted video.
        """
        self._ensure_ffmpeg()
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        if factor <= 0:
            raise VideoProcessingError(f"Invalid speed factor: {factor}")

        # Video filter: setpts=PTS/factor
        vfilter = f"setpts={1/factor}*PTS"
        # Audio filter: atempo=factor (limited to 0.5-100 range)
        afilters = []
        remaining = factor
        while remaining > 2.0:
            afilters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            afilters.append("atempo=0.5")
            remaining /= 0.5
        afilters.append(f"atempo={remaining}")
        afilter = ",".join(afilters)

        _run_ffmpeg([
            "-i", str(video_path),
            "-vf", vfilter,
            "-af", afilter,
            "-c:v", self._default_codec.value,
            "-c:a", self._default_audio_codec.value,
            str(output_path),
        ], timeout=600)

        logger.info("Changed speed %.2fx: %s", factor, output_path.name)
        return output_path

    def _get_audio_codec(self, format: str) -> str:
        """Get the appropriate FFmpeg audio codec for a format."""
        codec_map = {
            "mp3": "libmp3lame",
            "wav": "pcm_s16le",
            "flac": "flac",
            "aac": "aac",
            "ogg": "libvorbis",
            "opus": "libopus",
            "m4a": "aac",
            "wma": "wmav2",
            "aiff": "pcm_s16be",
        }
        return codec_map.get(format.lower(), "aac")

    async def cleanup(self) -> None:
        """Clean up temporary files."""
        import shutil
        if self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.info("Cleaned up temp directory: %s", self._temp_dir)
            except Exception as e:
                logger.warning("Failed to clean up %s: %s", self._temp_dir, e)


def _uuid_hex() -> str:
    """Generate a short random hex string."""
    import uuid
    return uuid.uuid4().hex[:8]
