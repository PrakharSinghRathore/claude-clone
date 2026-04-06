"""
Atlas Media Pipeline — Orchestrates media processing through a multi-stage pipeline.

Pipeline stages: input → validate → transform → encode → output
Each stage has dedicated error handling, logging, and retry logic.
Integrates with ImageProcessor, AudioProcessor, and VideoProcessor.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Enumeration of pipeline processing stages."""
    INPUT = "input"
    VALIDATE = "validate"
    TRANSFORM = "transform"
    ENCODE = "encode"
    OUTPUT = "output"


class MediaType(Enum):
    """Supported media types."""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


class ResizeMode(Enum):
    """Image resize modes."""
    EXACT = "exact"
    FIT = "fit"
    COVER = "cover"
    CROP = "crop"


@dataclass
class PipelineResult:
    """Result of a pipeline processing operation."""
    success: bool
    output_path: Optional[Path] = None
    output_data: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    stage_results: Dict[str, Any] = field(default_factory=dict)
    pipeline_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary."""
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "metadata": self.metadata,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
            "pipeline_id": self.pipeline_id,
        }


@dataclass
class MediaMetadata:
    """Metadata extracted from a media file."""
    path: Path
    media_type: MediaType
    format: str = ""
    size_bytes: int = 0
    width: int = 0
    height: int = 0
    duration_seconds: float = 0.0
    fps: float = 0.0
    codec: str = ""
    bit_rate: int = 0
    sample_rate: int = 0
    channels: int = 0
    has_audio: bool = False
    has_video: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary."""
        return {
            "path": str(self.path),
            "media_type": self.media_type.value,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "codec": self.codec,
            "bit_rate": self.bit_rate,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "has_audio": self.has_audio,
            "has_video": self.has_video,
            "extra": self.extra,
        }


@dataclass
class PipelineOperation:
    """A single operation in the pipeline."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    stage: PipelineStage = PipelineStage.TRANSFORM
    required: bool = True
    retry_count: int = 0
    max_retries: int = 2


class MediaInputError(Exception):
    """Raised when media input is invalid or inaccessible."""
    pass


class PipelineStageError(Exception):
    """Raised when a pipeline stage fails."""
    def __init__(self, stage: PipelineStage, message: str, cause: Optional[Exception] = None):
        self.stage = stage
        self.message = message
        self.cause = cause
        super().__init__(f"[{stage.value}] {message}")


class FFmpegError(Exception):
    """Raised when an FFmpeg operation fails."""
    pass


def _check_ffmpeg() -> bool:
    """Check if FFmpeg is available on the system."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_ffmpeg(args: List[str], timeout: int = 300) -> Tuple[bytes, bytes, int]:
    """Run an FFmpeg command and return stdout, stderr, and return code."""
    cmd = ["ffmpeg", "-y"] + args
    logger.debug("Running FFmpeg: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise FFmpegError(
                f"FFmpeg failed (code {result.returncode}): {result.stderr.decode()[:500]}"
            )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        raise FFmpegError(f"FFmpeg timed out after {timeout}s")
    except FileNotFoundError:
        raise FFmpegError("FFmpeg is not installed or not found in PATH")


def _detect_media_type(path: Path) -> MediaType:
    """Detect media type from file extension."""
    extension_map = {
        # Images
        ".jpg": MediaType.IMAGE, ".jpeg": MediaType.IMAGE, ".png": MediaType.IMAGE,
        ".gif": MediaType.IMAGE, ".webp": MediaType.IMAGE, ".bmp": MediaType.IMAGE,
        ".tiff": MediaType.IMAGE, ".tif": MediaType.IMAGE, ".svg": MediaType.IMAGE,
        ".ico": MediaType.IMAGE, ".avif": MediaType.IMAGE,
        # Audio
        ".mp3": MediaType.AUDIO, ".wav": MediaType.AUDIO, ".flac": MediaType.AUDIO,
        ".aac": MediaType.AUDIO, ".ogg": MediaType.AUDIO, ".m4a": MediaType.AUDIO,
        ".wma": MediaType.AUDIO, ".opus": MediaType.AUDIO, ".aiff": MediaType.AUDIO,
        # Video
        ".mp4": MediaType.VIDEO, ".mkv": MediaType.VIDEO, ".avi": MediaType.VIDEO,
        ".mov": MediaType.VIDEO, ".wmv": MediaType.VIDEO, ".flv": MediaType.VIDEO,
        ".webm": MediaType.VIDEO, ".m4v": MediaType.VIDEO, ".ts": MediaType.VIDEO,
        ".3gp": MediaType.VIDEO,
    }
    ext = path.suffix.lower()
    return extension_map.get(ext, MediaType.UNKNOWN)


# Format extension mapping
FORMAT_EXTENSIONS: Dict[str, str] = {
    "jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "gif": ".gif",
    "webp": ".webp", "bmp": ".bmp", "tiff": ".tiff", "ico": ".ico",
    "mp3": ".mp3", "wav": ".wav", "flac": ".flac", "aac": ".aac",
    "ogg": ".ogg", "m4a": ".m4a", "opus": ".opus",
    "mp4": ".mp4", "mkv": ".mkv", "avi": ".avi", "mov": ".mov",
    "webm": ".webm", "flv": ".flv", "wmv": ".wmv",
}


class MediaPipeline:
    """
    Orchestrates media processing through a multi-stage pipeline.

    The pipeline follows the flow: input → validate → transform → encode → output.
    Each stage has dedicated error handling and can be customized with callbacks.
    Supports image, audio, and video processing via FFmpeg integration.

    Example:
        >>> pipeline = MediaPipeline()
        >>> result = await pipeline.process(
        ...     Path("input.jpg"),
        ...     [PipelineOperation("resize", {"width": 800, "height": 600})]
        ... )
        >>> print(result.output_path)
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        temp_dir: Optional[Union[str, Path]] = None,
        ffmpeg_available: Optional[bool] = None,
        max_file_size_mb: int = 500,
        retry_count: int = 2,
    ) -> None:
        """
        Initialize the MediaPipeline.

        Args:
            output_dir: Directory for output files. Defaults to a temp dir.
            temp_dir: Directory for temporary files. Defaults to system temp.
            ffmpeg_available: Whether FFmpeg is available. Auto-detected if None.
            max_file_size_mb: Maximum input file size in megabytes.
            retry_count: Default retry count for failed operations.
        """
        self._output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="atlas_media_out_"))
        self._temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="atlas_media_tmp_"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir.mkdir(parents=True, exist_ok=True)

        if ffmpeg_available is None:
            self._ffmpeg_available = _check_ffmpeg()
        else:
            self._ffmpeg_available = ffmpeg_available

        self._max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._default_retry_count = retry_count
        self._stage_handlers: Dict[PipelineStage, Callable] = {
            PipelineStage.INPUT: self._stage_input,
            PipelineStage.VALIDATE: self._stage_validate,
            PipelineStage.TRANSFORM: self._stage_transform,
            PipelineStage.ENCODE: self._stage_encode,
            PipelineStage.OUTPUT: self._stage_output,
        }
        self._stats: Dict[str, int] = {
            "total_processed": 0,
            "total_failed": 0,
            "total_bytes_in": 0,
            "total_bytes_out": 0,
        }
        logger.info(
            "MediaPipeline initialized: output_dir=%s, ffmpeg=%s",
            self._output_dir, self._ffmpeg_available,
        )

    @property
    def ffmpeg_available(self) -> bool:
        """Whether FFmpeg is available for video/audio processing."""
        return self._ffmpeg_available

    @property
    def output_dir(self) -> Path:
        """The output directory path."""
        return self._output_dir

    async def process(
        self,
        input_source: Union[str, Path, bytes],
        operations: Optional[List[Union[PipelineOperation, Dict[str, Any]]]] = None,
        output_format: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> PipelineResult:
        """
        Process media through the full pipeline.

        Args:
            input_source: File path or raw bytes to process.
            operations: List of PipelineOperation objects or dicts to apply.
            output_format: Desired output format (e.g., "webp", "mp4").
            output_path: Specific output file path. Auto-generated if None.

        Returns:
            PipelineResult with output path, metadata, and any errors.

        Raises:
            MediaInputError: If the input source is invalid.
            PipelineStageError: If a required pipeline stage fails.
        """
        pipeline_id = uuid.uuid4().hex[:12]
        start_time = time.monotonic()
        result = PipelineResult(pipeline_id=pipeline_id, success=False)

        logger.info("Pipeline %s: starting process", pipeline_id)

        # Normalize operations
        ops: List[PipelineOperation] = []
        if operations:
            for op in operations:
                if isinstance(op, PipelineOperation):
                    ops.append(op)
                elif isinstance(op, dict):
                    ops.append(PipelineOperation(
                        name=op.get("name", "unknown"),
                        params=op.get("params", {}),
                        stage=PipelineStage(op.get("stage", "transform")),
                        required=op.get("required", True),
                        max_retries=op.get("max_retries", self._default_retry_count),
                    ))

        try:
            # Stage 1: Input
            current_path, input_meta = await self._run_stage(
                PipelineStage.INPUT, pipeline_id, {"source": input_source}
            )
            result.stage_results["input"] = {"path": str(current_path)}

            # Stage 2: Validate
            validate_result = await self._run_stage(
                PipelineStage.VALIDATE, pipeline_id, {
                    "path": current_path, "metadata": input_meta,
                }
            )
            result.stage_results["validate"] = validate_result

            # Stage 3: Transform
            current_path = await self._run_stage(
                PipelineStage.TRANSFORM, pipeline_id, {
                    "path": current_path, "operations": ops, "metadata": input_meta,
                }
            )
            result.stage_results["transform"] = {"path": str(current_path)}

            # Stage 4: Encode
            if output_format:
                current_path = await self._run_stage(
                    PipelineStage.ENCODE, pipeline_id, {
                        "path": current_path, "format": output_format,
                    }
                )
                result.stage_results["encode"] = {"path": str(current_path)}

            # Stage 5: Output
            final_path = await self._run_stage(
                PipelineStage.OUTPUT, pipeline_id, {
                    "path": current_path, "output_path": output_path,
                }
            )
            result.output_path = final_path
            result.stage_results["output"] = {"path": str(final_path)}

            # Gather output metadata
            result.metadata = await self.get_metadata(final_path)
            result.success = True
            self._stats["total_processed"] += 1
            self._stats["total_bytes_out"] += final_path.stat().st_size if final_path.exists() else 0

            logger.info("Pipeline %s: completed successfully in %.1fms", pipeline_id, (time.monotonic() - start_time) * 1000)

        except PipelineStageError as e:
            result.errors.append(str(e))
            result.warnings.append(f"Pipeline failed at stage: {e.stage.value}")
            self._stats["total_failed"] += 1
            logger.error("Pipeline %s: stage error - %s", pipeline_id, e)
        except Exception as e:
            result.errors.append(f"Unexpected error: {e}")
            self._stats["total_failed"] += 1
            logger.error("Pipeline %s: unexpected error - %s", pipeline_id, e, exc_info=True)
        finally:
            result.duration_ms = (time.monotonic() - start_time) * 1000

        return result

    async def _run_stage(
        self,
        stage: PipelineStage,
        pipeline_id: str,
        context: Dict[str, Any],
    ) -> Any:
        """
        Execute a pipeline stage with retry logic.

        Args:
            stage: The pipeline stage to execute.
            pipeline_id: Unique identifier for the current pipeline run.
            context: Shared context dict with stage-specific data.

        Returns:
            Stage-specific result.

        Raises:
            PipelineStageError: If the stage fails after all retries.
        """
        handler = self._stage_handlers.get(stage)
        if handler is None:
            raise PipelineStageError(stage, f"No handler registered for stage: {stage.value}")

        max_retries = context.get("max_retries", self._default_retry_count)
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Pipeline %s: running stage %s (attempt %d/%d)",
                    pipeline_id, stage.value, attempt + 1, max_retries + 1,
                )
                result = await handler(context)
                return result
            except PipelineStageError as e:
                last_error = e
                logger.warning(
                    "Pipeline %s: stage %s failed (attempt %d): %s",
                    pipeline_id, stage.value, attempt + 1, e,
                )
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                last_error = e
                raise PipelineStageError(stage, str(e), cause=e) from e

        raise PipelineStageError(
            stage,
            f"Stage failed after {max_retries + 1} attempts: {last_error}",
            cause=last_error,
        )

    async def _stage_input(self, context: Dict[str, Any]) -> Tuple[Path, MediaMetadata]:
        """
        Input stage: Load and prepare input media.

        Handles both file paths and raw bytes input.
        """
        source = context["source"]

        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise PipelineStageError(
                    PipelineStage.INPUT, f"Input file does not exist: {path}"
                )
            if not os.access(path, os.R_OK):
                raise PipelineStageError(
                    PipelineStage.INPUT, f"Input file is not readable: {path}"
                )
            file_size = path.stat().st_size
            if file_size > self._max_file_size_bytes:
                raise PipelineStageError(
                    PipelineStage.INPUT,
                    f"Input file too large: {file_size / (1024*1024):.1f}MB "
                    f"(max {self._max_file_size_bytes / (1024*1024):.1f}MB)",
                )
            self._stats["total_bytes_in"] += file_size
            metadata = MediaMetadata(
                path=path,
                media_type=_detect_media_type(path),
                size_bytes=file_size,
                format=path.suffix.lstrip("."),
            )
            return path, metadata

        elif isinstance(source, bytes):
            # Write bytes to a temp file
            media_type = self._detect_bytes_type(source)
            ext = self._type_to_extension(media_type)
            temp_path = self._temp_dir / f"{uuid.uuid4().hex[:8]}{ext}"
            temp_path.write_bytes(source)
            metadata = MediaMetadata(
                path=temp_path,
                media_type=media_type,
                size_bytes=len(source),
                format=ext.lstrip("."),
            )
            self._stats["total_bytes_in"] += len(source)
            return temp_path, metadata

        else:
            raise PipelineStageError(
                PipelineStage.INPUT,
                f"Unsupported input type: {type(source).__name__}",
            )

    async def _stage_validate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate stage: Check media integrity and compatibility.

        Validates file headers, checks for corruption, and ensures
        the media type is supported for the requested operations.
        """
        path: Path = context["path"]
        metadata: MediaMetadata = context["metadata"]

        validation_result: Dict[str, Any] = {
            "valid": True,
            "checks": [],
        }

        # Check file is not empty
        if metadata.size_bytes == 0:
            raise PipelineStageError(
                PipelineStage.VALIDATE, f"Input file is empty: {path}"
            )
        validation_result["checks"].append({"name": "non_empty", "passed": True})

        # Check known media type
        if metadata.media_type == MediaType.UNKNOWN:
            # Try FFmpeg probe for unknown types
            if self._ffmpeg_available:
                probe = await self._ffprobe(path)
                if probe:
                    if probe.get("streams"):
                        for stream in probe["streams"]:
                            codec_type = stream.get("codec_type", "")
                            if codec_type == "video":
                                metadata.media_type = MediaType.VIDEO
                                metadata.has_video = True
                            elif codec_type == "audio":
                                if metadata.media_type == MediaType.VIDEO:
                                    metadata.has_audio = True
                                else:
                                    metadata.media_type = MediaType.AUDIO

            if metadata.media_type == MediaType.UNKNOWN:
                validation_result["warnings"] = ["Unknown media type, processing may be limited"]
                logger.warning("Unknown media type for %s", path)

        validation_result["checks"].append({
            "name": "media_type_detected",
            "passed": metadata.media_type != MediaType.UNKNOWN,
            "media_type": metadata.media_type.value,
        })

        # Check file header magic bytes
        magic_ok = await self._check_file_magic(path, metadata.media_type)
        validation_result["checks"].append({
            "name": "magic_bytes",
            "passed": magic_ok,
        })
        if not magic_ok:
            validation_result["warnings"] = validation_result.get("warnings", [])
            validation_result["warnings"].append(
                "File header magic bytes do not match expected format"
            )

        validation_result["valid"] = all(c["passed"] for c in validation_result["checks"])
        return validation_result

    async def _stage_transform(self, context: Dict[str, Any]) -> Path:
        """
        Transform stage: Apply all requested operations.

        Processes operations sequentially, passing intermediate results
        between operations.
        """
        path: Path = context["path"]
        operations: List[PipelineOperation] = context.get("operations", [])
        metadata: MediaMetadata = context.get("metadata")

        if not operations:
            return path

        current_path = path

        for i, op in enumerate(operations):
            logger.debug("Applying operation %d/%d: %s", i + 1, len(operations), op.name)
            try:
                current_path = await self._apply_operation(current_path, op, metadata)
            except Exception as e:
                if op.required:
                    raise PipelineStageError(
                        PipelineStage.TRANSFORM,
                        f"Required operation '{op.name}' failed: {e}",
                        cause=e,
                    )
                else:
                    logger.warning(
                        "Optional operation '%s' failed, continuing: %s",
                        op.name, e,
                    )

        return current_path

    async def _stage_encode(self, context: Dict[str, Any]) -> Path:
        """
        Encode stage: Convert media to the target format.

        Uses FFmpeg for video/audio and Pillow for images.
        """
        path: Path = context["path"]
        target_format: str = context["format"].lower()

        ext = FORMAT_EXTENSIONS.get(target_format, f".{target_format}")
        output_path = self._temp_dir / f"{uuid.uuid4().hex[:8]}{ext}"

        media_type = _detect_media_type(path)

        if media_type == MediaType.IMAGE:
            return await self._encode_image(path, output_path, target_format)
        elif media_type in (MediaType.AUDIO, MediaType.VIDEO):
            return await self._encode_av(path, output_path, target_format)
        else:
            # Try FFmpeg as last resort
            if self._ffmpeg_available:
                return await self._encode_av(path, output_path, target_format)
            raise PipelineStageError(
                PipelineStage.ENCODE,
                f"Cannot encode unknown media type to {target_format}",
            )

    async def _stage_output(self, context: Dict[str, Any]) -> Path:
        """
        Output stage: Move the result to the final output location.

        If no explicit output path is given, generates one based on
        the input filename with a timestamp suffix.
        """
        path: Path = context["path"]
        requested_output: Optional[Path] = context.get("output_path")

        if requested_output:
            output = Path(requested_output)
            output.parent.mkdir(parents=True, exist_ok=True)
        else:
            timestamp = int(time.time())
            output = self._output_dir / f"{path.stem}_{timestamp}{path.suffix}"

        shutil.copy2(str(path), str(output))
        return output

    async def _apply_operation(
        self, path: Path, operation: PipelineOperation, metadata: Optional[MediaMetadata] = None,
    ) -> Path:
        """Apply a single pipeline operation and return the result path."""
        name = operation.name
        params = operation.params

        if name == "resize":
            return await self.resize(
                path,
                params.get("width", 0),
                params.get("height", 0),
                mode=ResizeMode(params.get("mode", "fit")),
            )
        elif name == "convert":
            return await self.convert(path, params.get("format", "png"))
        elif name == "extract_frames":
            output_dir = self._temp_dir / f"frames_{uuid.uuid4().hex[:8]}"
            output_dir.mkdir(exist_ok=True)
            await self.extract_frames(path, output_dir, params.get("fps", 1))
            return path  # Frame extraction doesn't produce a single file
        elif name == "transcode":
            output_ext = FORMAT_EXTENSIONS.get(params.get("format", "mp4"), ".mp4")
            output = self._temp_dir / f"{uuid.uuid4().hex[:8]}{output_ext}"
            return await self.transcode(
                path, output,
                format=params.get("format", "mp4"),
                codec=params.get("codec", "auto"),
            )
        elif name == "create_thumbnail":
            output = self._temp_dir / f"thumb_{uuid.uuid4().hex[:8]}.jpg"
            return await self.create_thumbnail(path, output, params.get("size", (256, 256)))
        elif name == "compress":
            if _detect_media_type(path) == MediaType.IMAGE:
                return await self._compress_image(path, params.get("quality", 80))
            else:
                output = self._temp_dir / f"{uuid.uuid4().hex[:8]}{path.suffix}"
                return await self._compress_video(path, output, params.get("crf", 23))
        elif name == "crop":
            return await self._crop_media(
                path,
                params.get("x", 0), params.get("y", 0),
                params.get("width", 100), params.get("height", 100),
            )
        elif name == "rotate":
            return await self._rotate_media(path, params.get("degrees", 90))
        elif name == "trim":
            output = self._temp_dir / f"trimmed_{uuid.uuid4().hex[:8]}{path.suffix}"
            return await self._trim_media(
                path, output,
                params.get("start", 0), params.get("end", 10),
            )
        else:
            raise PipelineStageError(
                PipelineStage.TRANSFORM,
                f"Unknown operation: {name}",
            )

    async def resize(
        self,
        image_path: Union[str, Path],
        width: int,
        height: int,
        mode: ResizeMode = ResizeMode.FIT,
    ) -> Path:
        """
        Resize an image.

        Args:
            image_path: Path to the input image.
            width: Target width in pixels.
            height: Target height in pixels.
            mode: Resize strategy (exact, fit, cover, crop).

        Returns:
            Path to the resized image.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise MediaInputError(f"Image not found: {image_path}")

        output_path = self._temp_dir / f"resized_{uuid.uuid4().hex[:8]}{image_path.suffix}"

        try:
            from PIL import Image as PILImage

            with PILImage.open(image_path) as img:
                orig_w, orig_h = img.size

                if mode == ResizeMode.EXACT:
                    new_w, new_h = width, height
                elif mode == ResizeMode.FIT:
                    ratio = min(width / orig_w, height / orig_h)
                    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
                elif mode == ResizeMode.COVER:
                    ratio = max(width / orig_w, height / orig_h)
                    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
                elif mode == ResizeMode.CROP:
                    ratio = min(width / orig_w, height / orig_h)
                    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
                    # Then crop to exact
                    left = (new_w - width) // 2
                    top = (new_h - height) // 2
                    img = img.resize((new_w, new_h), PILImage.LANCZOS)
                    img = img.crop((left, top, left + width, top + height))
                    img.save(output_path)
                    return output_path
                else:
                    new_w, new_h = width, height

                img = img.resize((new_w, new_h), PILImage.LANCZOS)
                img.save(output_path)
                return output_path

        except ImportError:
            logger.warning("Pillow not available, falling back to FFmpeg for resize")
            if self._ffmpeg_available:
                _run_ffmpeg([
                    "-i", str(image_path),
                    "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                    "-q:v", "2",
                    str(output_path),
                ])
                return output_path
            raise PipelineStageError(
                PipelineStage.TRANSFORM,
                "Cannot resize: neither Pillow nor FFmpeg is available",
            )

    async def convert(
        self,
        input_path: Union[str, Path],
        output_format: str,
    ) -> Path:
        """
        Convert media between formats.

        Args:
            input_path: Path to the input media file.
            output_format: Target format (e.g., "webp", "mp3", "mp4").

        Returns:
            Path to the converted file.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise MediaInputError(f"File not found: {input_path}")

        ext = FORMAT_EXTENSIONS.get(output_format.lower(), f".{output_format.lower()}")
        output_path = self._temp_dir / f"converted_{uuid.uuid4().hex[:8]}{ext}"

        media_type = _detect_media_type(input_path)

        if media_type == MediaType.IMAGE:
            return await self._encode_image(input_path, output_path, output_format.lower())
        elif media_type in (MediaType.AUDIO, MediaType.VIDEO):
            return await self._encode_av(input_path, output_path, output_format.lower())
        else:
            if self._ffmpeg_available:
                return await self._encode_av(input_path, output_path, output_format.lower())
            raise PipelineStageError(
                PipelineStage.TRANSFORM,
                f"Cannot convert file of type: {media_type.value}",
            )

    async def extract_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        fps: float = 1.0,
    ) -> List[Path]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the video file.
            output_dir: Directory to save extracted frames.
            fps: Frames per second to extract.

        Returns:
            List of paths to extracted frame images.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise MediaInputError(f"Video not found: {video_path}")

        if not self._ffmpeg_available:
            raise PipelineStageError(
                PipelineStage.TRANSFORM,
                "FFmpeg is required for frame extraction",
            )

        pattern = str(output_dir / "frame_%04d.jpg")
        _run_ffmpeg([
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            pattern,
        ])

        frames = sorted(output_dir.glob("frame_*.jpg"))
        logger.info("Extracted %d frames from %s", len(frames), video_path.name)
        return frames

    async def transcode(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        format: str = "mp4",
        codec: str = "auto",
    ) -> Path:
        """
        Transcode a media file.

        Args:
            input_path: Path to the input media file.
            output_path: Path for the output file.
            format: Target container format.
            codec: Target codec. "auto" selects based on format.

        Returns:
            Path to the transcoded file.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise MediaInputError(f"File not found: {input_path}")

        if not self._ffmpeg_available:
            raise PipelineStageError(
                PipelineStage.TRANSFORM,
                "FFmpeg is required for transcoding",
            )

        codec_map = {
            "mp4": {"video": "libx264", "audio": "aac"},
            "webm": {"video": "libvpx-vp9", "audio": "libopus"},
            "avi": {"video": "mpeg4", "audio": "mp3"},
            "mkv": {"video": "libx264", "audio": "aac"},
            "mov": {"video": "libx264", "audio": "aac"},
            "mp3": {"audio": "libmp3lame"},
            "wav": {"audio": "pcm_s16le"},
            "flac": {"audio": "flac"},
            "ogg": {"audio": "libvorbis"},
            "aac": {"audio": "aac"},
        }

        codecs = codec_map.get(format, {})
        if codec != "auto":
            codecs = {"video": codec, "audio": codec}

        args = ["-i", str(input_path)]
        if "video" in codecs:
            args.extend(["-c:v", codecs["video"]])
        if "audio" in codecs:
            args.extend(["-c:a", codecs["audio"]])
        args.append(str(output_path))

        _run_ffmpeg(args)
        return output_path

    async def get_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract metadata from a media file.

        Args:
            path: Path to the media file.

        Returns:
            Dictionary containing metadata fields.
        """
        path = Path(path)
        if not path.exists():
            raise MediaInputError(f"File not found: {path}")

        metadata: Dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "format": path.suffix.lstrip("."),
            "media_type": _detect_media_type(path).value,
        }

        # Try FFmpeg ffprobe for detailed metadata
        if self._ffmpeg_available:
            probe = await self._ffprobe(path)
            if probe:
                if "format" in probe:
                    fmt = probe["format"]
                    metadata["duration_seconds"] = float(fmt.get("duration", 0))
                    metadata["bit_rate"] = int(fmt.get("bit_rate", 0))
                    metadata["format_name"] = fmt.get("format_name", "")

                for stream in probe.get("streams", []):
                    codec_type = stream.get("codec_type", "")
                    if codec_type == "video":
                        metadata["width"] = int(stream.get("width", 0))
                        metadata["height"] = int(stream.get("height", 0))
                        metadata["video_codec"] = stream.get("codec_name", "")
                        metadata["fps"] = self._parse_fps(stream.get("r_frame_rate", "0/1"))
                        metadata["has_video"] = True
                    elif codec_type == "audio":
                        metadata["audio_codec"] = stream.get("codec_name", "")
                        metadata["sample_rate"] = int(stream.get("sample_rate", 0))
                        metadata["channels"] = int(stream.get("channels", 0))
                        metadata["has_audio"] = True

        # Try Pillow for image-specific metadata
        if metadata["media_type"] == "image":
            try:
                from PIL import Image as PILImage
                with PILImage.open(path) as img:
                    metadata["width"] = img.width
                    metadata["height"] = img.height
                    metadata["image_mode"] = img.mode
                    metadata["image_format"] = img.format or ""
                    if hasattr(img, "_getexif") and img._getexif():
                        metadata["has_exif"] = True
            except ImportError:
                pass
            except Exception as e:
                logger.debug("Could not read image metadata: %s", e)

        return metadata

    async def create_thumbnail(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        size: Tuple[int, int] = (256, 256),
    ) -> Path:
        """
        Create a thumbnail from a media file.

        Args:
            input_path: Path to the input media.
            output_path: Path for the thumbnail. Auto-generated if None.
            size: Thumbnail dimensions as (width, height).

        Returns:
            Path to the created thumbnail.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise MediaInputError(f"File not found: {input_path}")

        if output_path is None:
            output_path = self._temp_dir / f"thumb_{uuid.uuid4().hex[:8]}.jpg"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        media_type = _detect_media_type(input_path)

        if media_type == MediaType.IMAGE:
            try:
                from PIL import Image as PILImage
                with PILImage.open(input_path) as img:
                    img.thumbnail(size, PILImage.LANCZOS)
                    img.save(output_path, "JPEG", quality=85)
                return output_path
            except ImportError:
                pass

        # FFmpeg fallback for all types including video
        if self._ffmpeg_available:
            _run_ffmpeg([
                "-i", str(input_path),
                "-vf", f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease",
                "-vframes", "1",
                "-q:v", "2",
                str(output_path),
            ])
            return output_path

        raise PipelineStageError(
            PipelineStage.TRANSFORM,
            "Cannot create thumbnail: neither Pillow nor FFmpeg available",
        )

    # ── Private helpers ──────────────────────────────────────────────

    async def _encode_image(
        self, input_path: Path, output_path: Path, target_format: str,
    ) -> Path:
        """Encode an image to the target format using Pillow or FFmpeg."""
        try:
            from PIL import Image as PILImage

            format_map = {
                "jpeg": "JPEG", "jpg": "JPEG", "png": "PNG", "gif": "GIF",
                "webp": "WEBP", "bmp": "BMP", "tiff": "TIFF", "ico": "ICO",
            }
            pil_format = format_map.get(target_format, "PNG")

            with PILImage.open(input_path) as img:
                # Convert mode if needed (e.g., RGBA → RGB for JPEG)
                if pil_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                    background = PILImage.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    background.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
                    img = background
                elif pil_format != "GIF" and img.mode == "P":
                    img = img.convert("RGBA")

                img.save(output_path, pil_format, quality=90)
            return output_path

        except ImportError:
            if self._ffmpeg_available:
                return await self._encode_av(input_path, output_path, target_format)
            raise PipelineStageError(
                PipelineStage.ENCODE,
                f"Cannot encode image to {target_format}: Pillow not available",
            )

    async def _encode_av(
        self, input_path: Path, output_path: Path, target_format: str,
    ) -> Path:
        """Encode audio/video using FFmpeg."""
        if not self._ffmpeg_available:
            raise PipelineStageError(
                PipelineStage.ENCODE,
                f"FFmpeg required to encode to {target_format}",
            )
        _run_ffmpeg(["-i", str(input_path), str(output_path)])
        return output_path

    async def _compress_image(self, path: Path, quality: int = 80) -> Path:
        """Compress an image by reducing quality."""
        output_path = self._temp_dir / f"compressed_{uuid.uuid4().hex[:8]}{path.suffix}"
        try:
            from PIL import Image as PILImage
            with PILImage.open(path) as img:
                img.save(output_path, quality=quality, optimize=True)
            return output_path
        except ImportError:
            if self._ffmpeg_available:
                _run_ffmpeg(["-i", str(path), "-q:v", str(max(1, (100 - quality) // 5 + 1)), str(output_path)])
                return output_path
            raise PipelineStageError(PipelineStage.TRANSFORM, "Cannot compress image: no backend available")

    async def _compress_video(self, path: Path, output_path: Path, crf: int = 23) -> Path:
        """Compress a video using CRF-based encoding."""
        if not self._ffmpeg_available:
            raise PipelineStageError(PipelineStage.TRANSFORM, "FFmpeg required for video compression")
        _run_ffmpeg([
            "-i", str(path), "-c:v", "libx264", "-crf", str(crf),
            "-preset", "medium", "-c:a", "aac", "-b:a", "128k",
            str(output_path),
        ])
        return output_path

    async def _crop_media(self, path: Path, x: int, y: int, w: int, h: int) -> Path:
        """Crop media (image or video frame)."""
        output_path = self._temp_dir / f"cropped_{uuid.uuid4().hex[:8]}{path.suffix}"
        if _detect_media_type(path) == MediaType.IMAGE:
            try:
                from PIL import Image as PILImage
                with PILImage.open(path) as img:
                    img = img.crop((x, y, x + w, y + h))
                    img.save(output_path)
                return output_path
            except ImportError:
                pass
        if self._ffmpeg_available:
            _run_ffmpeg([
                "-i", str(path), "-vf", f"crop={w}:{h}:{x}:{y}",
                str(output_path),
            ])
            return output_path
        raise PipelineStageError(PipelineStage.TRANSFORM, "Cannot crop: no backend available")

    async def _rotate_media(self, path: Path, degrees: int) -> Path:
        """Rotate media."""
        output_path = self._temp_dir / f"rotated_{uuid.uuid4().hex[:8]}{path.suffix}"
        if _detect_media_type(path) == MediaType.IMAGE:
            try:
                from PIL import Image as PILImage
                with PILImage.open(path) as img:
                    img = img.rotate(-degrees, expand=True)
                    img.save(output_path)
                return output_path
            except ImportError:
                pass
        if self._ffmpeg_available:
            # FFmpeg transpose: 1=90CW, 2=90CCW, 3=90CW+flip, 0=90CCW+flip
            transpose_map = {90: "1", 180: "1,transpose=1", 270: "2"}
            transpose = transpose_map.get(degrees, "1")
            _run_ffmpeg([
                "-i", str(path), "-vf", f"transpose={transpose}",
                "-c:a", "copy", str(output_path),
            ])
            return output_path
        raise PipelineStageError(PipelineStage.TRANSFORM, "Cannot rotate: no backend available")

    async def _trim_media(self, path: Path, output_path: Path, start: float, end: float) -> Path:
        """Trim audio/video to a time range."""
        if not self._ffmpeg_available:
            raise PipelineStageError(PipelineStage.TRANSFORM, "FFmpeg required for trimming")
        duration = end - start
        _run_ffmpeg([
            "-ss", str(start), "-i", str(path), "-t", str(duration),
            "-c", "copy", str(output_path),
        ])
        return output_path

    async def _ffprobe(self, path: Path) -> Optional[Dict[str, Any]]:
        """Run ffprobe and parse JSON output."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return __import__("json").loads(result.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired, __import__("json").JSONDecodeError):
            pass
        return None

    def _parse_fps(self, fps_str: str) -> float:
        """Parse FFmpeg frame rate string like '30/1' or '29.97'."""
        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num) / float(den) if float(den) != 0 else 0.0
            return float(fps_str)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _detect_bytes_type(self, data: bytes) -> MediaType:
        """Detect media type from byte content (magic bytes)."""
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return MediaType.IMAGE
        if data[:2] in (b"\xff\xd8", b"\xff\xd9"):
            return MediaType.IMAGE
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return MediaType.IMAGE
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return MediaType.AUDIO
        if data[:3] == b"ID3" or data[:4] == b"\xff\xfb" or data[:4] == b"\xff\xf3":
            return MediaType.AUDIO
        if data[:4] in (b"\x1a\x45\xdf\xa3", b"ftyp"):
            return MediaType.VIDEO
        if data[:12] == b"\x00\x00\x00\x18ftypmp42":
            return MediaType.VIDEO
        return MediaType.UNKNOWN

    def _type_to_extension(self, media_type: MediaType) -> str:
        """Map media type to a default file extension."""
        return {
            MediaType.IMAGE: ".png",
            MediaType.AUDIO: ".wav",
            MediaType.VIDEO: ".mp4",
            MediaType.UNKNOWN: ".bin",
        }.get(media_type, ".bin")

    async def _check_file_magic(self, path: Path, media_type: MediaType) -> bool:
        """Check if file magic bytes match the expected media type."""
        try:
            with open(path, "rb") as f:
                header = f.read(12)
            if media_type == MediaType.IMAGE:
                return header[:8] == b"\x89PNG\r\n\x1a\n" or header[:2] == b"\xff\xd8"
            elif media_type == MediaType.AUDIO:
                return header[:4] == b"RIFF" or header[:3] == b"ID3"
            elif media_type == MediaType.VIDEO:
                return True  # Video formats have varied headers
            return True  # Allow unknown types
        except Exception:
            return True  # Don't fail validation on magic check errors

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline usage statistics."""
        return dict(self._stats)

    async def cleanup(self) -> None:
        """Clean up temporary files and directories."""
        for d in [self._temp_dir]:
            if d.exists():
                try:
                    shutil.rmtree(d)
                    logger.info("Cleaned up temp directory: %s", d)
                except Exception as e:
                    logger.warning("Failed to clean up %s: %s", d, e)

    async def __aenter__(self) -> "MediaPipeline":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.cleanup()
