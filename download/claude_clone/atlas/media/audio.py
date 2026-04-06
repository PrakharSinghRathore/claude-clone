"""
Atlas Audio Processor — Comprehensive audio processing tools.

Provides audio loading, saving, format conversion, segment extraction,
waveform generation, volume normalization, resampling, and mixing.
Uses FFmpeg for format conversion and pydub/wave for processing when available.
"""

from __future__ import annotations

import asyncio
import logging
import math
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Try to import pydub for advanced audio processing
try:
    from pydub import AudioSegment
    from pydub.effects import normalize as pydub_normalize
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    logger.debug("pydub not available; audio processing will use FFmpeg fallback")

# Try to import numpy for waveform data generation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.debug("numpy not available; waveform generation will be limited")


class AudioFormat(Enum):
    """Supported audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    AAC = "aac"
    OGG = "ogg"
    M4A = "m4a"
    WMA = "wma"
    OPUS = "opus"
    AIFF = "aiff"


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""
    pass


class FFmpegNotFoundError(Exception):
    """Raised when FFmpeg is required but not found."""
    pass


@dataclass
class AudioInfo:
    """Information about an audio file."""
    path: Optional[Path] = None
    format: str = ""
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    bit_depth: int = 0
    bit_rate: int = 0
    file_size: int = 0
    codec: str = ""
    is_stereo: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path) if self.path else None,
            "format": self.format,
            "duration_seconds": round(self.duration_seconds, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "bit_rate": self.bit_rate,
            "file_size": self.file_size,
            "codec": self.codec,
            "is_stereo": self.is_stereo,
        }


@dataclass
class WaveformData:
    """Waveform data for visualization."""
    samples: List[float] = field(default_factory=list)
    sample_rate: int = 44100
    channels: int = 1
    duration_seconds: float = 0.0
    peaks: List[float] = field(default_factory=list)
    min_val: float = 0.0
    max_val: float = 0.0
    rms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes raw samples for size)."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": round(self.duration_seconds, 3),
            "peaks_count": len(self.peaks),
            "min_val": round(self.min_val, 4),
            "max_val": round(self.max_val, 4),
            "rms": round(self.rms, 4),
        }


def _check_ffmpeg() -> bool:
    """Check if FFmpeg is available on the system."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_ffmpeg(args: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run an FFmpeg command."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    logger.debug("FFmpeg: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[:500]
            raise AudioProcessingError(f"FFmpeg error: {stderr}")
        return result
    except subprocess.TimeoutExpired:
        raise AudioProcessingError(f"FFmpeg timed out after {timeout}s")
    except FileNotFoundError:
        raise FFmpegNotFoundError("FFmpeg is not installed or not found in PATH")


def _run_ffprobe(path: Path) -> Optional[Dict[str, Any]]:
    """Run ffprobe and return parsed JSON."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


class AudioProcessor:
    """
    Comprehensive audio processing class.

    Provides tools for loading, saving, format conversion, segment extraction,
    waveform generation, volume normalization, resampling, and mixing of audio.
    Uses FFmpeg for format conversion and pydub for processing when available.

    Example:
        >>> processor = AudioProcessor()
        >>> info = await processor.get_info("song.mp3")
        >>> await processor.convert_format("song.mp3", "song.wav", "wav")
        >>> segment = await processor.extract_segment("song.mp3", 10, 30)
    """

    def __init__(
        self,
        temp_dir: Optional[Union[str, Path]] = None,
        default_sample_rate: int = 44100,
        default_bitrate: str = "192k",
    ) -> None:
        """
        Initialize the AudioProcessor.

        Args:
            temp_dir: Directory for temporary files. Defaults to system temp.
            default_sample_rate: Default sample rate for processing.
            default_bitrate: Default bitrate for encoding.
        """
        self._temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="atlas_audio_"))
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._default_sample_rate = default_sample_rate
        self._default_bitrate = default_bitrate
        self._ffmpeg_available = _check_ffmpeg()

        if not self._ffmpeg_available:
            logger.warning("FFmpeg is not available. Audio processing will be limited.")

    @property
    def ffmpeg_available(self) -> bool:
        """Whether FFmpeg is available."""
        return self._ffmpeg_available

    @property
    def temp_dir(self) -> Path:
        """The temporary directory path."""
        return self._temp_dir

    async def load(self, path: Union[str, Path]) -> Any:
        """
        Load an audio file.

        Args:
            path: Path to the audio file.

        Returns:
            AudioSegment (pydub) or raw audio data dict.

        Raises:
            AudioProcessingError: If loading fails.
        """
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        if HAS_PYDUB:
            try:
                audio = AudioSegment.from_file(str(path))
                logger.info("Loaded audio: %s (%.1fs)", path.name, len(audio) / 1000)
                return audio
            except Exception as e:
                raise AudioProcessingError(f"Failed to load audio with pydub: {e}") from e
        else:
            # Fallback: return a dict with basic info and bytes
            info = await self.get_info(path)
            return {"path": path, "info": info, "bytes": path.read_bytes()}

    async def save(
        self,
        audio: Any,
        path: Union[str, Path],
        format: Optional[str] = None,
        bitrate: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ) -> Path:
        """
        Save an audio file.

        Args:
            audio: AudioSegment (pydub) or audio data dict.
            path: Output file path.
            format: Output format. Auto-detected from path if None.
            bitrate: Bitrate for lossy formats.
            sample_rate: Target sample rate.

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = path.suffix.lstrip(".")
            if not format:
                format = "wav"

        bitrate = bitrate or self._default_bitrate
        sr = sample_rate or self._default_sample_rate

        if HAS_PYDUB and isinstance(audio, AudioSegment):
            try:
                export_kwargs: Dict[str, Any] = {
                    "format": format,
                    "bitrate": bitrate,
                }
                # Set tags for supported formats
                audio.export(str(path), **export_kwargs)
                logger.info("Saved audio: %s (%s)", path, format)
                return path
            except Exception as e:
                raise AudioProcessingError(f"Failed to save audio: {e}") from e
        elif isinstance(audio, dict) and "bytes" in audio:
            # Direct byte write fallback
            if format == "wav":
                path.write_bytes(audio["bytes"])
            elif self._ffmpeg_available:
                # Use FFmpeg to convert
                temp_input = self._temp_dir / f"temp_input.{format}"
                temp_input.write_bytes(audio["bytes"])
                await self.convert_format(temp_input, path, format, bitrate=bitrate)
                temp_input.unlink(missing_ok=True)
            else:
                path.write_bytes(audio["bytes"])
            return path
        else:
            raise AudioProcessingError("Cannot save: unsupported audio data type")

    async def convert_format(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        format: str,
        bitrate: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ) -> Path:
        """
        Convert audio file to a different format.

        Args:
            input_path: Path to the source audio file.
            output_path: Path for the output file.
            format: Target format (mp3, wav, flac, aac, ogg, etc.).
            bitrate: Bitrate for lossy formats (e.g., "192k").
            sample_rate: Target sample rate.

        Returns:
            Path to the converted file.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise AudioProcessingError(f"Audio file not found: {input_path}")

        # Try pydub first
        if HAS_PYDUB and format.lower() in ("mp3", "wav", "ogg", "flac", "aac", "m4a"):
            try:
                audio = AudioSegment.from_file(str(input_path))
                export_kwargs: Dict[str, Any] = {"format": format.lower()}
                if bitrate:
                    export_kwargs["bitrate"] = bitrate
                if sample_rate:
                    audio = audio.set_frame_rate(sample_rate)
                audio.export(str(output_path), **export_kwargs)
                logger.info("Converted %s → %s (%s)", input_path.name, output_path.name, format)
                return output_path
            except Exception as e:
                logger.warning("pydub conversion failed, falling back to FFmpeg: %s", e)

        # FFmpeg fallback
        if self._ffmpeg_available:
            args = ["-i", str(input_path)]
            if sample_rate:
                args.extend(["-ar", str(sample_rate)])
            if bitrate:
                args.extend(["-b:a", bitrate])
            args.append(str(output_path))
            _run_ffmpeg(args)
            logger.info("Converted %s → %s via FFmpeg", input_path.name, output_path.name)
            return output_path

        raise AudioProcessingError(
            f"Cannot convert to {format}: neither pydub nor FFmpeg available"
        )

    async def extract_segment(
        self,
        audio_or_path: Union[Any, str, Path],
        start_sec: float,
        end_sec: float,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[Any, Path]:
        """
        Extract a segment from an audio file.

        Args:
            audio_or_path: AudioSegment or path to audio file.
            start_sec: Start time in seconds.
            end_sec: End time in seconds.
            output_path: Output file path. If None, returns AudioSegment.

        Returns:
            AudioSegment or path to saved segment.
        """
        if start_sec < 0 or end_sec <= start_sec:
            raise AudioProcessingError(f"Invalid time range: {start_sec}s - {end_sec}s")

        duration_ms = (end_sec - start_sec) * 1000
        start_ms = start_sec * 1000

        if HAS_PYDUB and isinstance(audio_or_path, AudioSegment):
            segment = audio_or_path[start_ms:start_ms + duration_ms]
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fmt = output_path.suffix.lstrip(".") or "wav"
                segment.export(str(output_path), format=fmt)
                return output_path
            return segment

        # File path input
        path = Path(audio_or_path) if isinstance(audio_or_path, (str, Path)) else None
        if path and path.exists():
            if HAS_PYDUB:
                audio = AudioSegment.from_file(str(path))
                segment = audio[start_ms:start_ms + duration_ms]
                if output_path:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    fmt = output_path.suffix.lstrip(".") or "wav"
                    segment.export(str(output_path), format=fmt)
                    return output_path
                return segment
            elif self._ffmpeg_available:
                if output_path is None:
                    output_path = self._temp_dir / f"segment_{uuid_hex()}.wav"
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                _run_ffmpeg([
                    "-i", str(path),
                    "-ss", str(start_sec),
                    "-t", str(end_sec - start_sec),
                    "-c", "copy",
                    str(output_path),
                ])
                return output_path

        raise AudioProcessingError("Cannot extract segment: no backend available")

    async def get_duration(self, path: Union[str, Path]) -> float:
        """
        Get the duration of an audio file in seconds.

        Args:
            path: Path to the audio file.

        Returns:
            Duration in seconds.
        """
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        # Try pydub first
        if HAS_PYDUB:
            try:
                audio = AudioSegment.from_file(str(path))
                return len(audio) / 1000.0
            except Exception:
                pass

        # Try wave module for WAV files
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    return frames / rate if rate > 0 else 0.0
            except Exception:
                pass

        # FFmpeg ffprobe fallback
        if self._ffmpeg_available:
            probe = _run_ffprobe(path)
            if probe and "format" in probe:
                duration = probe["format"].get("duration", "0")
                try:
                    return float(duration)
                except ValueError:
                    pass

        raise AudioProcessingError(f"Cannot determine duration for: {path}")

    async def get_info(self, path: Union[str, Path]) -> AudioInfo:
        """
        Get detailed information about an audio file.

        Args:
            path: Path to the audio file.

        Returns:
            AudioInfo dataclass with file metadata.
        """
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        info = AudioInfo(
            path=path,
            format=path.suffix.lstrip("."),
            file_size=path.stat().st_size,
        )

        # Try pydub for basic info
        if HAS_PYDUB:
            try:
                audio = AudioSegment.from_file(str(path))
                info.duration_seconds = len(audio) / 1000.0
                info.sample_rate = audio.frame_rate
                info.channels = audio.channels
                info.is_stereo = audio.channels == 2
                info.bit_depth = audio.sample_width * 8
            except Exception:
                pass

        # Try wave module for WAV
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as wf:
                    info.channels = wf.getnchannels()
                    info.sample_rate = wf.getframerate()
                    info.is_stereo = info.channels == 2
                    info.bit_depth = wf.getsampwidth() * 8
                    frames = wf.getnframes()
                    info.duration_seconds = frames / wf.getframerate() if wf.getframerate() > 0 else 0
            except Exception:
                pass

        # Use ffprobe for more detailed info
        if self._ffmpeg_available:
            probe = _run_ffprobe(path)
            if probe:
                for stream in probe.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        info.codec = stream.get("codec_name", "")
                        if not info.sample_rate:
                            info.sample_rate = int(stream.get("sample_rate", 0))
                        if not info.channels:
                            info.channels = int(stream.get("channels", 0))
                            info.is_stereo = info.channels == 2
                        if not info.bit_depth:
                            info.bit_depth = int(stream.get("bits_per_sample", 0))
                        break
                if "format" in probe:
                    fmt = probe["format"]
                    if not info.duration_seconds:
                        try:
                            info.duration_seconds = float(fmt.get("duration", 0))
                        except ValueError:
                            pass
                    if not info.bit_rate:
                        try:
                            info.bit_rate = int(fmt.get("bit_rate", 0))
                        except ValueError:
                            pass

        return info

    async def get_waveform(
        self,
        path: Union[str, Path],
        samples: int = 1000,
        window_size: Optional[float] = None,
    ) -> WaveformData:
        """
        Generate waveform data for visualization.

        Args:
            path: Path to the audio file.
            samples: Number of samples to generate.
            window_size: Duration of each sample window in seconds.

        Returns:
            WaveformData with peak values for visualization.
        """
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        # Get audio info first
        info = await self.get_info(path)
        duration = info.duration_seconds

        if duration <= 0:
            return WaveformData(sample_rate=info.sample_rate, channels=info.channels)

        # Calculate window size
        if window_size is None:
            window_size = duration / samples

        waveform = WaveformData(
            sample_rate=info.sample_rate,
            channels=info.channels,
            duration_seconds=duration,
        )

        # Try numpy + pydub for accurate waveform
        if HAS_PYDUB and HAS_NUMPY:
            try:
                audio = AudioSegment.from_file(str(path))
                raw_data = np.array(audio.get_array_of_samples())

                if audio.channels == 2:
                    # Convert stereo to mono by averaging
                    left = raw_data[0::2].astype(float)
                    right = raw_data[1::2].astype(float)
                    raw_data = (left + right) / 2
                else:
                    raw_data = raw_data.astype(float)

                # Normalize to -1.0 .. 1.0
                max_val = np.max(np.abs(raw_data)) if len(raw_data) > 0 else 1
                if max_val > 0:
                    raw_data = raw_data / max_val

                # Calculate window-based peaks
                samples_per_window = int(window_size * info.sample_rate)
                if samples_per_window <= 0:
                    samples_per_window = len(raw_data) // samples

                peaks = []
                for i in range(0, len(raw_data), max(1, samples_per_window)):
                    window = raw_data[i:i + samples_per_window]
                    if len(window) > 0:
                        peak = np.max(np.abs(window))
                        peaks.append(float(peak))

                waveform.peaks = peaks[:samples]
                waveform.min_val = float(np.min(raw_data))
                waveform.max_val = float(np.max(raw_data))
                waveform.rms = float(np.sqrt(np.mean(raw_data ** 2)))

                # Store downsampled raw data
                step = max(1, len(raw_data) // samples)
                waveform.samples = raw_data[::step].tolist()[:samples]

                return waveform

            except Exception as e:
                logger.warning("NumPy/pydub waveform generation failed: %s", e)

        # FFmpeg-based waveform extraction (using volumedetect)
        if self._ffmpeg_available:
            try:
                result = subprocess.run(
                    ["ffmpeg", "-i", str(path), "-af", "volumedetect",
                     "-f", "null", "-hide_banner", "-loglevel", "info", "/dev/null"],
                    capture_output=True, text=True, timeout=60,
                )
                stderr = result.stderr
                import re
                max_volume_match = re.search(r"max_volume:\s*([-\d.]+)\s*dB", stderr)
                mean_volume_match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", stderr)

                if max_volume_match:
                    db = float(max_volume_match.group(1))
                    waveform.max_val = 10 ** (db / 20)
                if mean_volume_match:
                    db = float(mean_volume_match.group(1))
                    waveform.rms = 10 ** (db / 20)

                # Generate approximate peaks
                waveform.peaks = [
                    waveform.rms + (waveform.max_val - waveform.rms) * abs(math.sin(i * 0.1))
                    for i in range(samples)
                ]
                return waveform

            except Exception as e:
                logger.warning("FFmpeg waveform generation failed: %s", e)

        # Final fallback: generate dummy waveform based on duration
        waveform.peaks = [0.5] * min(samples, int(duration * 10))
        return waveform

    async def normalize(
        self,
        audio_or_path: Union[Any, str, Path],
        target_dbfs: float = -20.0,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[Any, Path]:
        """
        Normalize audio volume.

        Args:
            audio_or_path: AudioSegment or path to audio file.
            target_dbfs: Target volume in dBFS (default -20.0).
            output_path: Output file path. If None, returns AudioSegment.

        Returns:
            Normalized AudioSegment or path to saved file.
        """
        if HAS_PYDUB:
            if isinstance(audio_or_path, AudioSegment):
                audio = audio_or_path
            elif isinstance(audio_or_path, (str, Path)):
                path = Path(audio_or_path)
                if not path.exists():
                    raise AudioProcessingError(f"Audio file not found: {path}")
                audio = AudioSegment.from_file(str(path))
            else:
                raise AudioProcessingError("Unsupported audio input type")

            change_in_dbfs = target_dbfs - audio.dBFS
            normalized = audio.apply_gain(change_in_dbfs)

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fmt = output_path.suffix.lstrip(".") or "wav"
                normalized.export(str(output_path), format=fmt)
                logger.info("Normalized audio to %.1f dBFS: %s", target_dbfs, output_path)
                return output_path

            return normalized

        # FFmpeg fallback
        if isinstance(audio_or_path, (str, Path)):
            path = Path(audio_or_path)
            if not path.exists():
                raise AudioProcessingError(f"Audio file not found: {path}")
            if output_path is None:
                output_path = self._temp_dir / f"normalized_{uuid_hex()}.wav"
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use loudnorm filter
            if self._ffmpeg_available:
                _run_ffmpeg([
                    "-i", str(path),
                    "-af", f"loudnorm=I={target_dbfs}:TP=-1.5:LRA=11",
                    str(output_path),
                ])
                logger.info("Normalized audio via FFmpeg: %s", output_path)
                return output_path

        raise AudioProcessingError("Cannot normalize: no backend available")

    async def resample(
        self,
        audio_or_path: Union[Any, str, Path],
        sample_rate: int,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[Any, Path]:
        """
        Resample audio to a different sample rate.

        Args:
            audio_or_path: AudioSegment or path to audio file.
            sample_rate: Target sample rate in Hz.
            output_path: Output file path. If None, returns AudioSegment.

        Returns:
            Resampled AudioSegment or path to saved file.
        """
        if HAS_PYDUB:
            if isinstance(audio_or_path, AudioSegment):
                audio = audio_or_path
            elif isinstance(audio_or_path, (str, Path)):
                path = Path(audio_or_path)
                if not path.exists():
                    raise AudioProcessingError(f"Audio file not found: {path}")
                audio = AudioSegment.from_file(str(path))
            else:
                raise AudioProcessingError("Unsupported audio input type")

            resampled = audio.set_frame_rate(sample_rate)

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fmt = output_path.suffix.lstrip(".") or "wav"
                resampled.export(str(output_path), format=fmt)
                logger.info("Resampled to %d Hz: %s", sample_rate, output_path)
                return output_path

            return resampled

        # FFmpeg fallback
        if isinstance(audio_or_path, (str, Path)):
            path = Path(audio_or_path)
            if not path.exists():
                raise AudioProcessingError(f"Audio file not found: {path}")
            if output_path is None:
                output_path = self._temp_dir / f"resampled_{uuid_hex()}.wav"
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if self._ffmpeg_available:
                _run_ffmpeg([
                    "-i", str(path),
                    "-ar", str(sample_rate),
                    str(output_path),
                ])
                logger.info("Resampled to %d Hz via FFmpeg: %s", sample_rate, output_path)
                return output_path

        raise AudioProcessingError("Cannot resample: no backend available")

    async def mix(
        self,
        audio1: Any,
        audio2: Any,
        volumes: Optional[Tuple[float, float]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[Any, Path]:
        """
        Mix two audio tracks together.

        Args:
            audio1: First AudioSegment or path.
            audio2: Second AudioSegment or path.
            volumes: (volume1, volume2) multipliers (0.0-2.0). None for equal mix.
            output_path: Output file path. If None, returns AudioSegment.

        Returns:
            Mixed AudioSegment or path to saved file.
        """
        if not HAS_PYDUB:
            raise AudioProcessingError("pydub is required for audio mixing")

        # Load if paths
        if isinstance(audio1, (str, Path)):
            audio1 = AudioSegment.from_file(str(Path(audio1)))
        if isinstance(audio2, (str, Path)):
            audio2 = AudioSegment.from_file(str(Path(audio2)))

        # Apply volume adjustments
        if volumes:
            audio1 = audio1 + (20 * math.log10(volumes[0])) if volumes[0] > 0 else audio1 - 60
            audio2 = audio2 + (20 * math.log10(volumes[1])) if volumes[1] > 0 else audio2 - 60

        # Overlay: audio2 on top of audio1
        mixed = audio1.overlay(audio2)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = output_path.suffix.lstrip(".") or "wav"
            mixed.export(str(output_path), format=fmt)
            logger.info("Mixed audio: %s", output_path)
            return output_path

        return mixed

    async def concatenate(
        self,
        audio_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        crossfade_ms: int = 0,
    ) -> Path:
        """
        Concatenate multiple audio files.

        Args:
            audio_paths: List of audio file paths.
            output_path: Output file path.
            crossfade_ms: Crossfade duration in milliseconds (0 = no crossfade).

        Returns:
            Path to the concatenated file.
        """
        if not audio_paths:
            raise AudioProcessingError("No audio files provided for concatenation")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_PYDUB:
            segments = []
            for p in audio_paths:
                p = Path(p)
                if not p.exists():
                    raise AudioProcessingError(f"Audio file not found: {p}")
                segments.append(AudioSegment.from_file(str(p)))

            combined = segments[0]
            for segment in segments[1:]:
                if crossfade_ms > 0:
                    combined = combined.append(segment, crossfade=crossfade_ms)
                else:
                    combined += segment

            fmt = output_path.suffix.lstrip(".") or "wav"
            combined.export(str(output_path), format=fmt)
            logger.info("Concatenated %d audio files: %s", len(audio_paths), output_path)
            return output_path

        # FFmpeg fallback
        if self._ffmpeg_available:
            # Create concat file list
            list_file = self._temp_dir / f"concat_list_{uuid_hex()}.txt"
            with open(list_file, "w") as f:
                for p in audio_paths:
                    f.write(f"file '{Path(p).resolve()}'\n")
            _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_file), str(output_path)])
            list_file.unlink(missing_ok=True)
            logger.info("Concatenated %d files via FFmpeg: %s", len(audio_paths), output_path)
            return output_path

        raise AudioProcessingError("Cannot concatenate: no backend available")

    async def get_loudness(self, path: Union[str, Path]) -> Dict[str, float]:
        """
        Measure audio loudness using FFmpeg loudnorm.

        Args:
            path: Path to the audio file.

        Returns:
            Dict with loudness metrics (input_i, input_tp, input_lra, input_thresh).
        """
        if not self._ffmpeg_available:
            raise AudioProcessingError("FFmpeg required for loudness measurement")

        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        try:
            result = subprocess.run(
                ["ffmpeg", "-i", str(path), "-af", "loudnorm=print_format=json",
                 "-f", "null", "-hide_banner", "-loglevel", "info", "/dev/null"],
                capture_output=True, text=True, timeout=120,
            )
            stderr = result.stderr
            import re
            json_match = re.search(r"\{.*\}", stderr, re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning("Loudness measurement failed: %s", e)

        return {}

    async def cleanup(self) -> None:
        """Clean up temporary files."""
        import shutil
        if self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.info("Cleaned up temp directory: %s", self._temp_dir)
            except Exception as e:
                logger.warning("Failed to clean up %s: %s", self._temp_dir, e)


def uuid_hex() -> str:
    """Generate a short random hex string."""
    import uuid
    return uuid.uuid4().hex[:8]
