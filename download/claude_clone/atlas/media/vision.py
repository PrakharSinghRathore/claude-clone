"""
Atlas Vision Analyzer — Image and video understanding with multi-provider support.

Provides image description, object detection, OCR text extraction, video analysis,
image comparison, face detection, demographic estimation, and auto-captioning.
Supports multiple AI providers (OpenAI Vision, Anthropic Vision, Google Vision)
with graceful fallback when providers are unavailable.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DetailLevel(Enum):
    """Level of detail for image descriptions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class VisionProvider(Enum):
    """Supported vision analysis providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    MOCK = "mock"


class VisionError(Exception):
    """Raised when vision analysis fails."""
    pass


class ProviderUnavailableError(VisionError):
    """Raised when a vision provider is not available."""
    pass


@dataclass
class ObjectDetection:
    """Result of object detection in an image."""
    label: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    count: int = 1
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "count": self.count,
            "attributes": self.attributes,
        }


@dataclass
class FaceDetection:
    """Result of face detection."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "landmarks": self.landmarks,
            "attributes": self.attributes,
        }


@dataclass
class TextExtraction:
    """Result of OCR text extraction."""
    text: str
    confidence: float = 0.0
    regions: List[Dict[str, Any]] = field(default_factory=list)
    language: str = ""
    blocks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "language": self.language,
            "region_count": len(self.regions),
            "block_count": len(self.blocks),
        }


@dataclass
class ImageComparison:
    """Result of comparing two images."""
    similarity: float
    structural_similarity: float = 0.0
    pixel_difference: float = 0.0
    histogram_difference: float = 0.0
    is_same: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "similarity": round(self.similarity, 4),
            "structural_similarity": round(self.structural_similarity, 4),
            "pixel_difference": round(self.pixel_difference, 4),
            "histogram_difference": round(self.histogram_difference, 4),
            "is_same": self.is_same,
            "description": self.description,
        }


@dataclass
class VideoAnalysis:
    """Result of analyzing video frames."""
    frames_analyzed: int = 0
    interval_seconds: float = 1.0
    descriptions: List[Dict[str, Any]] = field(default_factory=list)
    objects_detected: List[ObjectDetection] = field(default_factory=list)
    scenes: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames_analyzed": self.frames_analyzed,
            "interval_seconds": self.interval_seconds,
            "summary": self.summary,
            "duration_seconds": self.duration_seconds,
            "object_count": len(self.objects_detected),
            "scene_count": len(self.scenes),
        }


@dataclass
class AgeGenderEstimate:
    """Result of age and gender estimation."""
    age_range: Tuple[int, int] = (20, 40)
    apparent_age: float = 30.0
    gender: str = "unknown"
    gender_confidence: float = 0.0
    ethnicity: str = "unknown"
    emotion: str = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "age_range": list(self.age_range),
            "apparent_age": round(self.apparent_age, 1),
            "gender": self.gender,
            "gender_confidence": round(self.gender_confidence, 4),
            "ethnicity": self.ethnicity,
            "emotion": self.emotion,
        }


def _encode_image_to_base64(image_or_path: Union[Any, str, Path, bytes]) -> str:
    """Encode an image to base64 string."""
    if isinstance(image_or_path, bytes):
        return base64.b64encode(image_or_path).decode("utf-8")

    if isinstance(image_or_path, (str, Path)):
        path = Path(image_or_path)
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode("utf-8")
        raise VisionError(f"Image file not found: {path}")

    # PIL Image
    try:
        buffer = io.BytesIO()
        if hasattr(image_or_path, "save"):
            fmt = getattr(image_or_path, "format", "PNG") or "PNG"
            image_or_path.save(buffer, format=fmt)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        raise VisionError(f"Failed to encode image: {e}") from e

    raise VisionError(f"Cannot encode image from type: {type(image_or_path)}")


def _get_mime_type(image_or_path: Union[Any, str, Path, bytes]) -> str:
    """Determine the MIME type of an image."""
    if isinstance(image_or_path, (str, Path)):
        path = Path(image_or_path)
        ext_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".bmp": "image/bmp",
        }
        return ext_map.get(path.suffix.lower(), "image/png")
    return "image/png"


class VisionAnalyzer:
    """
    Comprehensive image and video understanding class.

    Provides AI-powered image description, object detection, OCR,
    video analysis, image comparison, face detection, demographic
    estimation, and auto-captioning with multi-provider support.

    Supports OpenAI Vision, Anthropic Vision, Google Vision, and
    local processing with graceful fallback when providers are unavailable.

    Example:
        >>> analyzer = VisionAnalyzer()
        >>> desc = await analyzer.describe("photo.jpg", DetailLevel.HIGH)
        >>> objects = await analyzer.detect_objects("photo.jpg")
        >>> text = await analyzer.extract_text("document.png")
    """

    def __init__(
        self,
        providers: Optional[List[VisionProvider]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        default_provider: Optional[VisionProvider] = None,
        max_retries: int = 2,
        timeout: int = 60,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialize the VisionAnalyzer.

        Args:
            providers: Ordered list of providers to try. None uses all available.
            api_keys: API keys keyed by provider name.
            default_provider: Preferred provider. Falls back to auto-detection.
            max_retries: Retry count for failed API calls.
            timeout: Request timeout in seconds.
            model: Override the default model for the provider.
        """
        self._api_keys = api_keys or {}
        self._max_retries = max_retries
        self._timeout = timeout
        self._model_override = model

        # Detect available providers
        self._available_providers = self._detect_providers()
        self._provider_order = providers or list(self._available_providers)
        self._default_provider = default_provider

        # Provider-specific configurations
        self._provider_models = {
            VisionProvider.OPENAI: model or "gpt-4o",
            VisionProvider.ANTHROPIC: model or "claude-sonnet-4-20250514",
            VisionProvider.GOOGLE: model or "gemini-2.0-flash",
        }

        logger.info(
            "VisionAnalyzer initialized: providers=%s, default=%s",
            [p.value for p in self._available_providers],
            self._default_provider.value if self._default_provider else "auto",
        )

    def _detect_providers(self) -> List[VisionProvider]:
        """Detect which vision providers are available."""
        available: List[VisionProvider] = []

        # Check OpenAI
        openai_key = self._api_keys.get("openai") or os.environ.get("OPENAI_API_KEY")
        if openai_key:
            available.append(VisionProvider.OPENAI)

        # Check Anthropic
        anthropic_key = self._api_keys.get("anthropic") or os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            available.append(VisionProvider.ANTHROPIC)

        # Check Google
        google_key = self._api_keys.get("google") or os.environ.get("GOOGLE_API_KEY")
        if google_key:
            available.append(VisionProvider.GOOGLE)

        # Always have local as fallback
        available.append(VisionProvider.LOCAL)
        available.append(VisionProvider.MOCK)

        return available

    def _get_api_key(self, provider: VisionProvider) -> Optional[str]:
        """Get the API key for a provider."""
        key = self._api_keys.get(provider.value) or os.environ.get(f"{provider.value.upper()}_API_KEY")
        return key

    def _get_model(self, provider: VisionProvider) -> str:
        """Get the model name for a provider."""
        if self._model_override:
            return self._model_override
        return self._provider_models.get(provider, "default")

    async def _call_provider(
        self,
        prompt: str,
        image_or_path: Union[Any, str, Path, bytes],
        providers_to_try: Optional[List[VisionProvider]] = None,
        detail: DetailLevel = DetailLevel.MEDIUM,
    ) -> str:
        """
        Call a vision provider with an image and prompt.

        Tries providers in order until one succeeds.
        """
        providers = providers_to_try or self._provider_order

        # If a default provider is set, try it first
        if self._default_provider and self._default_provider in providers:
            providers = [self._default_provider] + [p for p in providers if p != self._default_provider]

        last_error: Optional[Exception] = None

        for provider in providers:
            for attempt in range(self._max_retries + 1):
                try:
                    result = await self._call_single_provider(
                        provider, prompt, image_or_path, detail,
                    )
                    return result
                except ProviderUnavailableError as e:
                    logger.debug("Provider %s unavailable: %s", provider.value, e)
                    break  # Don't retry unavailable providers
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Provider %s attempt %d failed: %s",
                        provider.value, attempt + 1, e,
                    )
                    if attempt < self._max_retries:
                        await asyncio.sleep(0.5 * (attempt + 1))

        if last_error:
            raise VisionError(f"All providers failed: {last_error}") from last_error

        return "No vision provider available for analysis."

    async def _call_single_provider(
        self,
        provider: VisionProvider,
        prompt: str,
        image_or_path: Union[Any, str, Path, bytes],
        detail: DetailLevel,
    ) -> str:
        """Call a single provider."""
        if provider == VisionProvider.OPENAI:
            return await self._call_openai(prompt, image_or_path, detail)
        elif provider == VisionProvider.ANTHROPIC:
            return await self._call_anthropic(prompt, image_or_path, detail)
        elif provider == VisionProvider.GOOGLE:
            return await self._call_google(prompt, image_or_path, detail)
        elif provider == VisionProvider.MOCK:
            return self._mock_response(prompt, image_or_path)
        else:
            raise ProviderUnavailableError(f"Provider not implemented: {provider.value}")

    async def _call_openai(
        self,
        prompt: str,
        image_or_path: Union[Any, str, Path, bytes],
        detail: DetailLevel,
    ) -> str:
        """Call OpenAI Vision API."""
        api_key = self._get_api_key(VisionProvider.OPENAI)
        if not api_key:
            raise ProviderUnavailableError("OpenAI API key not configured")

        try:
            import httpx

            b64 = _encode_image_to_base64(image_or_path)
            mime = _get_mime_type(image_or_path)
            model = self._get_model(VisionProvider.OPENAI)

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                    "detail": detail.value,
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 1000,
            }

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

        except ImportError:
            raise ProviderUnavailableError("httpx not installed for OpenAI Vision")
        except Exception as e:
            raise VisionError(f"OpenAI Vision error: {e}") from e

    async def _call_anthropic(
        self,
        prompt: str,
        image_or_path: Union[Any, str, Path, bytes],
        detail: DetailLevel,
    ) -> str:
        """Call Anthropic Vision API."""
        api_key = self._get_api_key(VisionProvider.ANTHROPIC)
        if not api_key:
            raise ProviderUnavailableError("Anthropic API key not configured")

        try:
            import httpx

            b64 = _encode_image_to_base64(image_or_path)
            mime = _get_mime_type(image_or_path)
            model = self._get_model(VisionProvider.ANTHROPIC)

            payload = {
                "model": model,
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            }

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"]

        except ImportError:
            raise ProviderUnavailableError("httpx not installed for Anthropic Vision")
        except Exception as e:
            raise VisionError(f"Anthropic Vision error: {e}") from e

    async def _call_google(
        self,
        prompt: str,
        image_or_path: Union[Any, str, Path, bytes],
        detail: DetailLevel,
    ) -> str:
        """Call Google Gemini Vision API."""
        api_key = self._get_api_key(VisionProvider.GOOGLE)
        if not api_key:
            raise ProviderUnavailableError("Google API key not configured")

        try:
            import httpx

            b64 = _encode_image_to_base64(image_or_path)
            mime = _get_mime_type(image_or_path)
            model = self._get_model(VisionProvider.GOOGLE)

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": mime, "data": b64}},
                        ]
                    }
                ],
            }

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

        except ImportError:
            raise ProviderUnavailableError("httpx not installed for Google Vision")
        except Exception as e:
            raise VisionError(f"Google Vision error: {e}") from e

    @staticmethod
    def _mock_response(prompt: str, image_or_path: Union[Any, str, Path, bytes]) -> str:
        """Generate a mock response for testing."""
        if "describe" in prompt.lower() or "what" in prompt.lower():
            return "[Mock] This image appears to contain various visual elements. Detailed analysis requires a configured AI vision provider."
        elif "object" in prompt.lower() or "detect" in prompt.lower():
            return "[Mock] Detected objects: person (0.95), table (0.87), cup (0.82)"
        elif "text" in prompt.lower() or "ocr" in prompt.lower():
            return "[Mock] Extracted text would appear here with a configured provider."
        return "[Mock] Vision analysis result would appear here with a configured provider."

    async def describe(
        self,
        image_or_path: Union[Any, str, Path, bytes],
        detail_level: DetailLevel = DetailLevel.MEDIUM,
        context: Optional[str] = None,
    ) -> str:
        """
        Describe the content of an image.

        Args:
            image_or_path: PIL Image, file path, or bytes.
            detail_level: Level of detail for the description.
            context: Additional context to guide the description.

        Returns:
            Text description of the image content.
        """
        detail_prompts = {
            DetailLevel.LOW: "Briefly describe this image in 1-2 sentences.",
            DetailLevel.MEDIUM: "Describe this image in detail, including objects, scene, colors, and composition.",
            DetailLevel.HIGH: (
                "Provide a comprehensive description of this image including: "
                "1) Main subjects and objects, 2) Scene setting and environment, "
                "3) Colors and lighting, 4) Composition and perspective, "
                "5) Any text visible, 6) Emotional tone or mood, "
                "7) Notable details or unusual elements."
            ),
        }

        prompt = detail_prompts.get(detail_level, detail_prompts[DetailLevel.MEDIUM])
        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        return await self._call_provider(prompt, image_or_path)

    async def detect_objects(
        self,
        image_or_path: Union[Any, str, Path, bytes],
        labels: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
    ) -> List[ObjectDetection]:
        """
        Detect objects in an image.

        Args:
            image_or_path: PIL Image, file path, or bytes.
            labels: Optional list of specific labels to detect.
            confidence_threshold: Minimum confidence score (0-1).

        Returns:
            List of detected objects with confidence scores.
        """
        label_str = ""
        if labels:
            label_str = f"Focus on these categories: {', '.join(labels)}.\n"

        prompt = (
            f"{label_str}"
            "Analyze this image and identify all visible objects. "
            "For each object, provide: label, confidence (0-1), "
            "and approximate bounding box position (x, y, width, height as pixel percentages from top-left). "
            f"Only include objects with confidence above {confidence_threshold}. "
            "Format: JSON array of objects."
        )

        try:
            response = await self._call_provider(prompt, image_or_path)

            # Try to parse JSON from the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                detections = json.loads(json_match.group())
                return [
                    ObjectDetection(
                        label=d.get("label", d.get("name", "unknown")),
                        confidence=float(d.get("confidence", d.get("score", 0))),
                        bbox=tuple(d["bbox"]) if "bbox" in d else None,
                        count=int(d.get("count", 1)),
                    )
                    for d in detections
                    if float(d.get("confidence", d.get("score", 0))) >= confidence_threshold
                ]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse object detection response: %s", e)

        return []

    async def extract_text(
        self,
        image_or_path: Union[Any, str, Path, bytes],
        language: Optional[str] = None,
    ) -> TextExtraction:
        """
        Extract text from an image using OCR.

        Args:
            image_or_path: PIL Image, file path, or bytes.
            language: Expected language (e.g., "en", "zh", "ja").

        Returns:
            TextExtraction with extracted text and metadata.
        """
        lang_str = f" The text is in {language}." if language else ""
        prompt = (
            f"Extract all visible text from this image.{lang_str} "
            "Preserve the original formatting, line breaks, and structure. "
            "If the text is in a table, represent it as such. "
            "Provide the overall confidence of the extraction (0-1)."
        )

        try:
            response = await self._call_provider(prompt, image_or_path)
            return TextExtraction(
                text=response.strip(),
                confidence=0.85,  # Approximate when using LLM-based OCR
                language=language or "auto",
            )
        except Exception as e:
            logger.warning("Text extraction failed: %s", e)
            return TextExtraction(text="", confidence=0.0, language=language or "")

    async def analyze_video(
        self,
        video_path: Union[str, Path],
        interval_sec: float = 2.0,
        max_frames: int = 30,
    ) -> VideoAnalysis:
        """
        Analyze video by extracting frames at intervals.

        Args:
            video_path: Path to the video file.
            interval_sec: Seconds between frame analyses.
            max_frames: Maximum number of frames to analyze.

        Returns:
            VideoAnalysis with frame-by-frame descriptions and summary.
        """
        from atlas.media.video import VideoProcessor

        video_path = Path(video_path)
        if not video_path.exists():
            raise VisionError(f"Video file not found: {video_path}")

        analysis = VideoAnalysis(interval_seconds=interval_sec)

        # Get video info
        video_proc = VideoProcessor()
        info = await video_proc.get_info(video_path)
        analysis.duration_seconds = info.duration_seconds

        # Extract frames at intervals
        frames = await video_proc.extract_frames(
            video_path,
            video_proc.temp_dir / f"analysis_{_uuid_hex()}",
            fps=1.0 / interval_sec if interval_sec > 0 else 1,
            format="jpg",
            quality=3,
            max_frames=max_frames,
        )

        logger.info("Analyzing %d video frames", len(frames))

        # Analyze each frame
        for i, frame_path in enumerate(frames):
            timestamp = i * interval_sec
            try:
                description = await self.describe(
                    frame_path,
                    DetailLevel.LOW,
                    context=f"This is frame {i+1} of a video at timestamp {timestamp:.1f}s.",
                )
                analysis.descriptions.append({
                    "frame": i + 1,
                    "timestamp": timestamp,
                    "path": str(frame_path),
                    "description": description,
                })
                analysis.frames_analyzed += 1
            except Exception as e:
                logger.warning("Frame %d analysis failed: %s", i + 1, e)

        # Generate a summary of the video
        if analysis.descriptions:
            frame_texts = [
                f"[{d['timestamp']:.1f}s] {d['description']}"
                for d in analysis.descriptions[:10]  # Limit for context window
            ]
            summary_prompt = (
                "Based on these frame descriptions from a video, provide a concise "
                "summary of what happens in the video:\n\n"
                + "\n".join(frame_texts)
            )
            try:
                analysis.summary = await self._call_provider(summary_prompt, frames[0] if frames else "")
            except Exception as e:
                analysis.summary = f"Summary unavailable: {e}"

        # Cleanup temp frames
        await video_proc.cleanup()

        return analysis

    async def compare_images(
        self,
        image1: Union[Any, str, Path, bytes],
        image2: Union[Any, str, Path, bytes],
    ) -> ImageComparison:
        """
        Compare two images for similarity.

        Args:
            image1: First image (PIL Image, path, or bytes).
            image2: Second image (PIL Image, path, or bytes).

        Returns:
            ImageComparison with similarity metrics.
        """
        comparison = ImageComparison()

        # Calculate pixel-level difference
        try:
            from PIL import Image as PILImage
            import numpy as np

            def load_as_array(img_or_path):
                if isinstance(img_or_path, (str, Path)):
                    return np.array(PILImage.open(str(img_or_path)).convert("RGB"))
                elif isinstance(img_or_path, bytes):
                    return np.array(PILImage.open(io.BytesIO(img_or_path)).convert("RGB"))
                else:
                    return np.array(img_or_path.convert("RGB"))

            arr1 = load_as_array(image1)
            arr2 = load_as_array(image2)

            # Resize to match if needed
            if arr1.shape != arr2.shape:
                # Use LLM comparison for different-sized images
                b64_1 = _encode_image_to_base64(image1)
                b64_2 = _encode_image_to_base64(image2)

                prompt = (
                    "Compare these two images and rate their similarity from 0 to 1. "
                    "Consider: subject matter, composition, colors, and overall appearance. "
                    "Respond with a JSON: {\"similarity\": 0.XX, \"description\": \"...\"}"
                )

                # Use mock provider for base64 comparison prompt
                try:
                    response = await self._call_provider(prompt, image1)
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        comparison.similarity = float(data.get("similarity", 0.5))
                        comparison.description = data.get("description", "")
                        comparison.is_same = comparison.similarity > 0.9
                    return comparison
                except Exception:
                    comparison.similarity = 0.0
                    comparison.description = "Unable to compare images of different sizes."
                    return comparison

            # Pixel difference
            diff = np.abs(arr1.astype(float) - arr2.astype(float))
            comparison.pixel_difference = float(np.mean(diff)) / 255.0

            # Histogram difference
            def get_histogram(arr):
                hist = []
                for channel in range(3):
                    h, _ = np.histogram(arr[:, :, channel], bins=256, range=(0, 256))
                    hist.extend(h.tolist())
                return np.array(hist, dtype=float)

            hist1 = get_histogram(arr1)
            hist2 = get_histogram(arr2)
            if np.max(hist1) > 0 and np.max(hist2) > 0:
                hist1 = hist1 / np.sum(hist1)
                hist2 = hist2 / np.sum(hist2)
                comparison.histogram_difference = float(np.sum(np.abs(hist1 - hist2)))

            # Overall similarity (inverse of differences)
            comparison.similarity = max(0, 1.0 - (comparison.pixel_difference * 0.5 + comparison.histogram_difference * 0.5))
            comparison.is_same = comparison.similarity > 0.95

            if comparison.is_same:
                comparison.description = "Images appear identical or nearly identical."
            elif comparison.similarity > 0.8:
                comparison.description = "Images are very similar with minor differences."
            elif comparison.similarity > 0.5:
                comparison.description = "Images share some similarities but are noticeably different."
            else:
                comparison.description = "Images are significantly different."

        except ImportError:
            # No PIL/numpy available, use LLM comparison
            comparison.description = "Pixel-level comparison unavailable (missing Pillow/numpy)."

        return comparison

    async def detect_faces(
        self,
        image_or_path: Union[Any, str, Path, bytes],
        confidence_threshold: float = 0.5,
    ) -> List[FaceDetection]:
        """
        Detect faces in an image.

        Args:
            image_or_path: PIL Image, file path, or bytes.
            confidence_threshold: Minimum confidence score.

        Returns:
            List of detected faces with bounding boxes.
        """
        prompt = (
            "Detect all faces in this image. For each face, provide: "
            "bounding box (x, y, width, height as pixel percentages from top-left), "
            "confidence (0-1), and any visible facial landmarks "
            "(left_eye, right_eye, nose, mouth_left, mouth_right). "
            "Format: JSON array of face objects."
        )

        try:
            response = await self._call_provider(prompt, image_or_path)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                faces = json.loads(json_match.group())
                return [
                    FaceDetection(
                        bbox=tuple(f.get("bbox", [0, 0, 0, 0])),
                        confidence=float(f.get("confidence", 0)),
                        landmarks=f.get("landmarks"),
                    )
                    for f in faces
                    if float(f.get("confidence", 0)) >= confidence_threshold
                ]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse face detection response: %s", e)

        return []

    async def estimate_age_gender(
        self,
        image_or_path: Union[Any, str, Path, bytes],
    ) -> List[AgeGenderEstimate]:
        """
        Estimate age and gender of faces in an image.

        Args:
            image_or_path: PIL Image, file path, or bytes.

        Returns:
            List of demographic estimates, one per detected face.
        """
        prompt = (
            "For each face visible in this image, estimate: "
            "age range (min_age, max_age), apparent age, gender, "
            "gender confidence (0-1), dominant emotion, and ethnicity. "
            "Format: JSON array with estimates."
        )

        try:
            response = await self._call_provider(prompt, image_or_path)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                estimates = json.loads(json_match.group())
                return [
                    AgeGenderEstimate(
                        age_range=tuple(e.get("age_range", [20, 40])),
                        apparent_age=float(e.get("apparent_age", 30)),
                        gender=e.get("gender", "unknown"),
                        gender_confidence=float(e.get("gender_confidence", 0)),
                        ethnicity=e.get("ethnicity", "unknown"),
                        emotion=e.get("emotion", "neutral"),
                    )
                    for e in estimates
                ]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse age/gender estimation: %s", e)

        return []

    async def generate_caption(
        self,
        image_or_path: Union[Any, str, Path, bytes],
        max_length: int = 100,
        style: str = "descriptive",
    ) -> str:
        """
        Generate an automatic caption for an image.

        Args:
            image_or_path: PIL Image, file path, or bytes.
            max_length: Maximum caption length in characters.
            style: Caption style (descriptive, concise, creative, alt_text).

        Returns:
            Generated caption string.
        """
        style_prompts = {
            "descriptive": "Write a descriptive caption for this image.",
            "concise": "Write a short, concise caption (under 10 words) for this image.",
            "creative": "Write a creative and engaging caption for this image.",
            "alt_text": "Write accessibility alt-text for this image, describing its content for visually impaired users.",
        }

        prompt = style_prompts.get(style, style_prompts["descriptive"])
        prompt += f" Keep it under {max_length} characters."

        return await self._call_provider(prompt, image_or_path)

    async def analyze_colors(
        self,
        image_or_path: Union[Any, str, Path, bytes],
        num_colors: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Analyze the color palette of an image.

        Args:
            image_or_path: PIL Image, file path, or bytes.
            num_colors: Number of dominant colors to extract.

        Returns:
            List of color dictionaries with RGB, hex, and percentage.
        """
        try:
            from PIL import Image as PILImage
            from collections import Counter
            import colorsys

            def load_image(img_or_path):
                if isinstance(img_or_path, (str, Path)):
                    return PILImage.open(str(img_or_path)).convert("RGB")
                elif isinstance(img_or_path, bytes):
                    return PILImage.open(io.BytesIO(img_or_path)).convert("RGB")
                else:
                    return img_or_path.convert("RGB")

            img = load_image(image_or_path)
            img.thumbnail((128, 128))
            pixels = list(img.getdata())

            # Quantize and count
            quantized = Counter()
            for r, g, b in pixels:
                qr = (r // 32) * 32
                qg = (g // 32) * 32
                qb = (b // 32) * 32
                quantized[(qr, qg, qb)] += 1

            total = sum(quantized.values())
            colors = []
            for (r, g, b), count in quantized.most_common(num_colors):
                r = min(255, r)
                g = min(255, g)
                b = min(255, b)
                h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
                colors.append({
                    "rgb": [r, g, b],
                    "hex": f"#{r:02x}{g:02x}{b:02x}",
                    "percentage": round(count / total * 100, 1),
                    "hsl": [round(h * 360, 1), round(s * 100, 1), round(l * 100, 1)],
                })

            return colors

        except ImportError:
            return [{"error": "Pillow required for color analysis"}]
        except Exception as e:
            logger.warning("Color analysis failed: %s", e)
            return []


def _uuid_hex() -> str:
    """Generate a short random hex string."""
    import uuid
    return uuid.uuid4().hex[:8]
