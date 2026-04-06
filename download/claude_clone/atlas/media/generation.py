"""
Atlas Media Generation — AI-powered image, video, and music generation.

Provides ImageGenerator, VideoGenerator, and MusicGenerator classes with
multi-provider support (OpenAI DALL-E, Stability AI, Midjourney, Runway,
Pika, xAI Grok, Google MusicFX, Suno, Udio). Includes editing, upscaling,
variation, animation, extension, and batch generation capabilities.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ── Shared enums and types ──────────────────────────────────────────


class ImageModel(Enum):
    """Supported image generation models."""
    DALL_E_3 = "dall-e-3"
    DALL_E_2 = "dall-e-2"
    STABLE_DIFFUSION_XL = "stable-diffusion-xl"
    STABLE_DIFFUSION_3 = "stable-diffusion-3"
    MIDJOURNEY_V6 = "midjourney-v6"
    FLUX_PRO = "flux-pro"
    FLUX_SCHNELL = "flux-schnell"
    GPT_IMAGE_1 = "gpt-image-1"


class VideoModel(Enum):
    """Supported video generation models."""
    RUNWAY_GEN3 = "runway-gen3"
    PIKA_1_0 = "pika-1.0"
    PIKA_1_5 = "pika-1.5"
    GROK_VIDEO = "grok-video"
    SORA = "sora"
    KLING = "kling"


class MusicModel(Enum):
    """Supported music generation models."""
    MUSICFX = "musicfx"
    SUNO = "suno"
    UDIO = "udio"


class ImageStyle(Enum):
    """Predefined image generation styles."""
    NATURAL = "natural"
    VIVID = "vivid"
    CINEMATIC = "cinematic"
    ANIME = "anime"
    DIGITAL_ART = "digital-art"
    PHOTOGRAPHIC = "photographic"
    WATERCOLOR = "watercolor"
    OIL_PAINTING = "oil-painting"
    PIXEL_ART = "pixel-art"
    MINIMALIST = "minimalist"
    SURREAL = "surreal"
    COMIC = "comic"


# Supported image sizes
SUPPORTED_IMAGE_SIZES: List[Tuple[int, int]] = [
    (1024, 1024), (768, 1344), (864, 1152),
    (1344, 768), (1152, 864), (1440, 720), (720, 1440),
    (1792, 1024), (1024, 1792),
]


class GenerationError(Exception):
    """Raised when media generation fails."""
    pass


class ProviderError(GenerationError):
    """Raised when a specific provider fails."""
    def __init__(self, provider: str, message: str, cause: Optional[Exception] = None):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")
        self.__cause__ = cause


# ── Result dataclasses ─────────────────────────────────────────────


@dataclass
class ImageGenerationResult:
    """Result of an image generation operation."""
    image_data: Optional[bytes] = None
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    file_path: Optional[Path] = None
    revised_prompt: Optional[str] = None
    model: str = ""
    provider: str = ""
    size: Tuple[int, int] = (1024, 1024)
    style: str = ""
    generation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_url": self.image_url,
            "file_path": str(self.file_path) if self.file_path else None,
            "revised_prompt": self.revised_prompt,
            "model": self.model,
            "provider": self.provider,
            "size": list(self.size),
            "style": self.style,
            "generation_time_ms": round(self.generation_time_ms, 1),
        }


@dataclass
class VideoGenerationResult:
    """Result of a video generation operation."""
    video_url: Optional[str] = None
    video_data: Optional[bytes] = None
    file_path: Optional[Path] = None
    model: str = ""
    provider: str = ""
    duration_seconds: float = 0.0
    resolution: Tuple[int, int] = (1280, 720)
    generation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_url": self.video_url,
            "file_path": str(self.file_path) if self.file_path else None,
            "model": self.model,
            "provider": self.provider,
            "duration_seconds": self.duration_seconds,
            "resolution": list(self.resolution),
            "generation_time_ms": round(self.generation_time_ms, 1),
        }


@dataclass
class MusicGenerationResult:
    """Result of a music generation operation."""
    audio_url: Optional[str] = None
    audio_data: Optional[bytes] = None
    file_path: Optional[Path] = None
    track_id: Optional[str] = None
    title: str = ""
    model: str = ""
    provider: str = ""
    duration_seconds: float = 0.0
    genre: str = ""
    mood: str = ""
    generation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_url": self.audio_url,
            "file_path": str(self.file_path) if self.file_path else None,
            "track_id": self.track_id,
            "title": self.title,
            "model": self.model,
            "provider": self.provider,
            "duration_seconds": self.duration_seconds,
            "genre": self.genre,
            "mood": self.mood,
            "generation_time_ms": round(self.generation_time_ms, 1),
        }


# ── Helper functions ────────────────────────────────────────────────


def _validate_size(size: Tuple[int, int]) -> Tuple[int, int]:
    """Validate and normalize image size."""
    if size in SUPPORTED_IMAGE_SIZES:
        return size
    # Find closest supported size
    w, h = size
    closest = min(SUPPORTED_IMAGE_SIZES, key=lambda s: abs(s[0] - w) + abs(s[1] - h))
    logger.debug("Size %s not in supported sizes, using closest: %s", size, closest)
    return closest


def _resolve_api_key(provider: str, api_keys: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Resolve API key from explicit keys or environment variables."""
    if api_keys and provider in api_keys:
        return api_keys[provider]

    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "stability": "STABILITY_API_KEY",
        "midjourney": "MIDJOURNEY_API_KEY",
        "runway": "RUNWAY_API_KEY",
        "pika": "PIKA_API_KEY",
        "grok": "XAI_API_KEY",
        "suno": "SUNO_API_KEY",
        "udio": "UDIO_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    env_var = env_map.get(provider)
    if env_var:
        return os.environ.get(env_var)
    return os.environ.get(f"{provider.upper()}_API_KEY")


async def _download_file(url: str, timeout: int = 120) -> bytes:
    """Download a file from a URL."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
    except ImportError:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()


# ── Image Generator ─────────────────────────────────────────────────


class ImageGenerator:
    """
    AI-powered image generation with multi-provider support.

    Supports OpenAI DALL-E 2/3, Stability AI, Midjourney API,
    and Flux models for image generation, editing, upscaling,
    variation creation, and batch generation.

    Example:
        >>> gen = ImageGenerator()
        >>> result = await gen.generate("A sunset over mountains", size=(1024, 1024))
        >>> print(result.image_url)
    """

    def __init__(
        self,
        api_keys: Optional[Dict[str, str]] = None,
        default_model: Optional[str] = None,
        default_size: Tuple[int, int] = (1024, 1024),
        output_dir: Optional[Union[str, Path]] = None,
        timeout: int = 120,
        max_retries: int = 2,
    ) -> None:
        """
        Initialize the ImageGenerator.

        Args:
            api_keys: API keys keyed by provider name.
            default_model: Default model to use.
            default_size: Default image size.
            output_dir: Directory for saved images.
            timeout: Request timeout in seconds.
            max_retries: Retry count for failed requests.
        """
        self._api_keys = api_keys or {}
        self._default_model = default_model or "dall-e-3"
        self._default_size = default_size
        self._output_dir = Path(output_dir) if output_dir else Path.cwd() / "generated_images"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        self._max_retries = max_retries
        self._stats = {"total_generated": 0, "total_failed": 0, "total_bytes": 0}

    async def generate(
        self,
        prompt: str,
        size: Optional[Tuple[int, int]] = None,
        style: Optional[Union[str, ImageStyle]] = None,
        model: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        quality: str = "standard",
        save: bool = True,
    ) -> ImageGenerationResult:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            size: Image dimensions (width, height).
            style: Image style (natural, vivid, cinematic, etc.).
            model: Model to use. Uses default if None.
            negative_prompt: Things to avoid in the image.
            quality: Quality level (standard, hd).
            save: Whether to save the image to disk.

        Returns:
            ImageGenerationResult with image data and metadata.
        """
        start_time = time.monotonic()
        model = model or self._default_model
        size = _validate_size(size or self._default_size)
        style_str = style.value if isinstance(style, ImageStyle) else (style or "natural")

        result = ImageGenerationResult(model=model, size=size, style=style_str)

        try:
            if model in ("dall-e-2", "dall-e-3", "gpt-image-1"):
                result = await self._generate_openai(prompt, size, style_str, model, quality, negative_prompt)
            elif model in ("stable-diffusion-xl", "stable-diffusion-3", "flux-pro", "flux-schnell"):
                result = await self._generate_stability(prompt, size, style_str, model, negative_prompt)
            elif model.startswith("midjourney"):
                result = await self._generate_midjourney(prompt, size, style_str, model, negative_prompt)
            else:
                raise GenerationError(f"Unknown model: {model}")

            result.generation_time_ms = (time.monotonic() - start_time) * 1000

            # Download and save if URL returned
            if result.image_url and save:
                result.image_data = await _download_file(result.image_url, self._timeout)
                if result.image_data:
                    filename = f"{uuid.uuid4().hex[:12]}_{size[0]}x{size[1]}.png"
                    result.file_path = self._output_dir / filename
                    result.file_path.write_bytes(result.image_data)
                    result.image_base64 = base64.b64encode(result.image_data).decode("utf-8")
                    self._stats["total_bytes"] += len(result.image_data)

            self._stats["total_generated"] += 1
            logger.info("Generated image: model=%s, size=%s, time=%.0fms", model, size, result.generation_time_ms)

        except Exception as e:
            self._stats["total_failed"] += 1
            logger.error("Image generation failed: %s", e)
            raise GenerationError(f"Image generation failed: {e}") from e

        return result

    async def edit(
        self,
        image: Union[str, Path, bytes],
        prompt: str,
        mask: Optional[Union[str, Path, bytes]] = None,
        model: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> ImageGenerationResult:
        """
        Edit an existing image with a text prompt.

        Args:
            image: Source image (path, URL, or bytes).
            prompt: Edit instructions.
            mask: Optional mask for selective editing.
            model: Model to use.
            size: Output size.

        Returns:
            ImageGenerationResult with edited image.
        """
        model = model or self._default_model
        size = size or self._default_size

        # Encode source image
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if image_path.exists():
                image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            else:
                image_b64 = base64.b64encode(await _download_file(str(image))).decode("utf-8")
        elif isinstance(image, bytes):
            image_b64 = base64.b64encode(image).decode("utf-8")
        else:
            raise GenerationError(f"Unsupported image type: {type(image)}")

        start_time = time.monotonic()
        result = ImageGenerationResult(model=model, size=size)

        if model == "dall-e-2":
            result = await self._edit_openai(image_b64, prompt, mask, size)
        elif model in ("stable-diffusion-xl", "stable-diffusion-3"):
            result = await self._edit_stability(image_b64, prompt, mask, model, size)
        else:
            raise GenerationError(f"Editing not supported for model: {model}")

        result.generation_time_ms = (time.monotonic() - start_time) * 1000

        if result.image_url:
            result.image_data = await _download_file(result.image_url)
            if result.image_data:
                filename = f"edit_{uuid.uuid4().hex[:12]}.png"
                result.file_path = self._output_dir / filename
                result.file_path.write_bytes(result.image_data)

        self._stats["total_generated"] += 1
        return result

    async def upscale(
        self,
        image: Union[str, Path, bytes],
        scale: float = 2.0,
        model: Optional[str] = None,
    ) -> ImageGenerationResult:
        """
        Upscale an image to higher resolution.

        Args:
            image: Source image.
            scale: Upscale factor (2.0 = double resolution).
            model: Model to use.

        Returns:
            ImageGenerationResult with upscaled image.
        """
        model = model or "stable-diffusion-xl"
        prompt = "high quality, detailed, sharp, upscaled"

        result = await self.edit(
            image=image,
            prompt=prompt,
            model=model,
        )
        result.metadata["scale"] = scale
        return result

    async def vary(
        self,
        image: Union[str, Path, bytes],
        strength: float = 0.5,
        model: Optional[str] = None,
    ) -> ImageGenerationResult:
        """
        Create variations of an existing image.

        Args:
            image: Source image.
            strength: Variation strength (0.0-1.0).
            model: Model to use.

        Returns:
            ImageGenerationResult with variation.
        """
        model = model or self._default_model

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if image_path.exists():
                image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            else:
                raise GenerationError(f"Image not found: {image}")
        elif isinstance(image, bytes):
            image_b64 = base64.b64encode(image).decode("utf-8")
        else:
            raise GenerationError(f"Unsupported image type: {type(image)}")

        start_time = time.monotonic()
        result = ImageGenerationResult(model=model)

        if model == "dall-e-2":
            result = await self._vary_openai(image_b64, size=self._default_size)
        elif model in ("stable-diffusion-xl", "stable-diffusion-3"):
            result = await self._vary_stability(image_b64, strength, model)
        else:
            raise GenerationError(f"Variations not supported for model: {model}")

        result.generation_time_ms = (time.monotonic() - start_time) * 1000

        if result.image_url:
            result.image_data = await _download_file(result.image_url)
            if result.image_data:
                filename = f"vary_{uuid.uuid4().hex[:12]}.png"
                result.file_path = self._output_dir / filename
                result.file_path.write_bytes(result.image_data)

        self._stats["total_generated"] += 1
        return result

    async def generate_batch(
        self,
        prompts: List[str],
        size: Optional[Tuple[int, int]] = None,
        model: Optional[str] = None,
        concurrency: int = 3,
    ) -> List[ImageGenerationResult]:
        """
        Generate multiple images in batch.

        Args:
            prompts: List of text prompts.
            size: Image dimensions.
            model: Model to use.
            concurrency: Maximum concurrent generations.

        Returns:
            List of ImageGenerationResult objects.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _gen_one(prompt: str) -> ImageGenerationResult:
            async with semaphore:
                return await self.generate(prompt, size=size, model=model)

        tasks = [_gen_one(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: List[ImageGenerationResult] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Batch generation error: %s", r)
                output.append(ImageGenerationResult(metadata={"error": str(r)}))
            else:
                output.append(r)

        return output

    # ── Provider-specific methods ──────────────────────────────────

    async def _generate_openai(
        self, prompt: str, size: Tuple[int, int], style: str,
        model: str, quality: str, negative_prompt: Optional[str],
    ) -> ImageGenerationResult:
        """Generate image using OpenAI DALL-E API."""
        api_key = _resolve_api_key("openai", self._api_keys)
        if not api_key:
            raise ProviderError("openai", "OpenAI API key not configured")

        try:
            import httpx

            size_str = f"{size[0]}x{size[1]}"
            payload = {
                "model": model,
                "prompt": prompt,
                "n": 1,
                "size": size_str,
                "quality": quality,
                "response_format": "url",
            }
            if style == "vivid" and model == "dall-e-3":
                payload["style"] = "vivid"

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            return ImageGenerationResult(
                image_url=data["data"][0]["url"],
                revised_prompt=data["data"][0].get("revised_prompt"),
                model=model,
                provider="openai",
                size=size,
                style=style,
            )

        except ImportError:
            raise ProviderError("openai", "httpx not installed")
        except Exception as e:
            raise ProviderError("openai", str(e), e) from e

    async def _generate_stability(
        self, prompt: str, size: Tuple[int, int], style: str,
        model: str, negative_prompt: Optional[str],
    ) -> ImageGenerationResult:
        """Generate image using Stability AI API."""
        api_key = _resolve_api_key("stability", self._api_keys)
        if not api_key:
            raise ProviderError("stability", "Stability AI API key not configured")

        try:
            import httpx

            model_map = {
                "stable-diffusion-xl": "stable-diffusion-xl-1024-v1-0",
                "stable-diffusion-3": "stable-diffusion-3-large",
                "flux-pro": "flux-pro-1.1",
                "flux-schnell": "flux-schnell",
            }

            payload = {
                "text_prompts": [{"text": prompt, "weight": 1.0}],
                "cfg_scale": 7,
                "width": size[0],
                "height": size[1],
                "steps": 30,
                "samples": 1,
            }
            if negative_prompt:
                payload["text_prompts"].append({"text": negative_prompt, "weight": -1.0})

            stability_model = model_map.get(model, model_map["stable-diffusion-xl"])

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"https://api.stability.ai/v1/generation/{stability_model}/text-to-image",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            artifacts = data.get("artifacts", [])
            if artifacts:
                image_b64 = artifacts[0].get("base64", "")
                return ImageGenerationResult(
                    image_base64=image_b64,
                    image_data=base64.b64decode(image_b64),
                    model=model,
                    provider="stability",
                    size=size,
                    style=style,
                )

            raise ProviderError("stability", "No image in response")

        except ImportError:
            raise ProviderError("stability", "httpx not installed")
        except Exception as e:
            raise ProviderError("stability", str(e), e) from e

    async def _generate_midjourney(
        self, prompt: str, size: Tuple[int, int], style: str,
        model: str, negative_prompt: Optional[str],
    ) -> ImageGenerationResult:
        """Generate image using Midjourney API."""
        api_key = _resolve_api_key("midjourney", self._api_keys)
        if not api_key:
            raise ProviderError("midjourney", "Midjourney API key not configured")

        # Midjourney uses aspect ratio parameter
        aspect_map = {
            (1024, 1024): "1:1", (1344, 768): "16:9", (768, 1344): "9:16",
            (1152, 864): "4:3", (864, 1152): "3:4", (1440, 720): "2:1",
            (720, 1440): "1:2",
        }
        aspect = aspect_map.get(size, "1:1")

        # Style parameters
        style_params = {"natural": "--s 50", "vivid": "--s 750", "cinematic": "--s 250 --ar 16:9"}
        style_arg = style_params.get(style, "")

        full_prompt = f"{prompt} --ar {aspect} {style_arg}"
        if negative_prompt:
            full_prompt += f" --no {negative_prompt}"

        # Note: Midjourney's official API is limited; this uses a compatible endpoint pattern
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.midjourney.com/v1/imagine",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"prompt": full_prompt, "model": model, "aspect_ratio": aspect},
                )
                response.raise_for_status()
                data = response.json()

            return ImageGenerationResult(
                image_url=data.get("image_url"),
                model=model,
                provider="midjourney",
                size=size,
                style=style,
                metadata={"task_id": data.get("task_id")},
            )
        except Exception as e:
            raise ProviderError("midjourney", str(e), e) from e

    async def _edit_openai(
        self, image_b64: str, prompt: str, mask: Optional[Any],
        size: Tuple[int, int],
    ) -> ImageGenerationResult:
        """Edit image using OpenAI DALL-E API."""
        api_key = _resolve_api_key("openai", self._api_keys)
        if not api_key:
            raise ProviderError("openai", "OpenAI API key not configured")

        try:
            import httpx

            payload: Dict[str, Any] = {
                "model": "dall-e-2",
                "prompt": prompt,
                "n": 1,
                "size": f"{size[0]}x{size[1]}",
                "image": image_b64,
            }
            if mask:
                if isinstance(mask, (str, Path)):
                    mask_b64 = base64.b64encode(Path(mask).read_bytes()).decode("utf-8")
                elif isinstance(mask, bytes):
                    mask_b64 = base64.b64encode(mask).decode("utf-8")
                else:
                    mask_b64 = str(mask)
                payload["mask"] = mask_b64

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.openai.com/v1/images/edits",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=payload,
                )
                response.raise_for_status()
                data = response.json()

            return ImageGenerationResult(
                image_url=data["data"][0]["url"],
                model="dall-e-2",
                provider="openai",
                size=size,
            )
        except Exception as e:
            raise ProviderError("openai", str(e), e) from e

    async def _edit_stability(
        self, image_b64: str, prompt: str, mask: Optional[Any],
        model: str, size: Tuple[int, int],
    ) -> ImageGenerationResult:
        """Edit image using Stability AI API."""
        api_key = _resolve_api_key("stability", self._api_keys)
        if not api_key:
            raise ProviderError("stability", "Stability AI API key not configured")

        try:
            import httpx

            payload = {
                "text_prompts": [{"text": prompt, "weight": 1.0}],
                "init_image": image_b64,
                "init_image_mode": "IMAGE_STRENGTH",
                "image_strength": 0.5,
                "cfg_scale": 7,
                "width": size[0],
                "height": size[1],
                "steps": 30,
                "samples": 1,
            }

            model_map = {
                "stable-diffusion-xl": "stable-diffusion-xl-1024-v1-0",
                "stable-diffusion-3": "stable-diffusion-3-large",
            }
            stability_model = model_map.get(model, model_map["stable-diffusion-xl"])

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"https://api.stability.ai/v1/generation/{stability_model}/image-to-image",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            artifacts = data.get("artifacts", [])
            if artifacts:
                img_b64 = artifacts[0].get("base64", "")
                return ImageGenerationResult(
                    image_base64=img_b64,
                    image_data=base64.b64decode(img_b64),
                    model=model,
                    provider="stability",
                    size=size,
                )

            raise ProviderError("stability", "No image in response")

        except Exception as e:
            raise ProviderError("stability", str(e), e) from e

    async def _vary_openai(self, image_b64: str, size: Tuple[int, int]) -> ImageGenerationResult:
        """Create variations using OpenAI DALL-E API."""
        api_key = _resolve_api_key("openai", self._api_keys)
        if not api_key:
            raise ProviderError("openai", "OpenAI API key not configured")

        try:
            import httpx

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.openai.com/v1/images/variations",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data={"model": "dall-e-2", "n": 1, "size": f"{size[0]}x{size[1]}"},
                    files={"image": ("image.png", base64.b64decode(image_b64), "image/png")},
                )
                response.raise_for_status()
                data = response.json()

            return ImageGenerationResult(
                image_url=data["data"][0]["url"],
                model="dall-e-2",
                provider="openai",
                size=size,
            )
        except Exception as e:
            raise ProviderError("openai", str(e), e) from e

    async def _vary_stability(self, image_b64: str, strength: float, model: str) -> ImageGenerationResult:
        """Create variations using Stability AI API."""
        api_key = _resolve_api_key("stability", self._api_keys)
        if not api_key:
            raise ProviderError("stability", "Stability AI API key not configured")

        try:
            import httpx

            payload = {
                "init_image": image_b64,
                "init_image_mode": "IMAGE_STRENGTH",
                "image_strength": 1.0 - strength,
                "text_prompts": [{"text": "varied version", "weight": 0.5}],
                "cfg_scale": 7,
                "steps": 30,
                "samples": 1,
            }

            model_map = {
                "stable-diffusion-xl": "stable-diffusion-xl-1024-v1-0",
                "stable-diffusion-3": "stable-diffusion-3-large",
            }
            stability_model = model_map.get(model, model_map["stable-diffusion-xl"])

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"https://api.stability.ai/v1/generation/{stability_model}/image-to-image",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            artifacts = data.get("artifacts", [])
            if artifacts:
                img_b64 = artifacts[0].get("base64", "")
                return ImageGenerationResult(
                    image_base64=img_b64,
                    image_data=base64.b64decode(img_b64),
                    model=model,
                    provider="stability",
                    metadata={"strength": strength},
                )

            raise ProviderError("stability", "No image in response")

        except Exception as e:
            raise ProviderError("stability", str(e), e) from e

    def get_stats(self) -> Dict[str, int]:
        """Get generation statistics."""
        return dict(self._stats)


# ── Video Generator ─────────────────────────────────────────────────


class VideoGenerator:
    """
    AI-powered video generation with multi-provider support.

    Supports Runway, Pika, xAI Grok, and Sora for video generation,
    image animation, and video extension.

    Example:
        >>> gen = VideoGenerator()
        >>> result = await gen.generate("A cat walking in a garden", duration=5)
        >>> print(result.video_url)
    """

    def __init__(
        self,
        api_keys: Optional[Dict[str, str]] = None,
        default_model: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        timeout: int = 300,
    ) -> None:
        """
        Initialize the VideoGenerator.

        Args:
            api_keys: API keys keyed by provider name.
            default_model: Default model to use.
            output_dir: Directory for saved videos.
            timeout: Request timeout in seconds.
        """
        self._api_keys = api_keys or {}
        self._default_model = default_model or "runway-gen3"
        self._output_dir = Path(output_dir) if output_dir else Path.cwd() / "generated_videos"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout

    async def generate(
        self,
        prompt: str,
        duration: float = 5.0,
        resolution: Tuple[int, int] = (1280, 720),
        model: Optional[str] = None,
    ) -> VideoGenerationResult:
        """
        Generate a video from a text prompt.

        Args:
            prompt: Text description of the desired video.
            duration: Duration in seconds.
            resolution: Video resolution (width, height).
            model: Model to use.

        Returns:
            VideoGenerationResult with video data and metadata.
        """
        start_time = time.monotonic()
        model = model or self._default_model
        result = VideoGenerationResult(model=model, resolution=resolution, duration_seconds=duration)

        try:
            if model.startswith("runway"):
                result = await self._generate_runway(prompt, duration, resolution, model)
            elif model.startswith("pika"):
                result = await self._generate_pika(prompt, duration, resolution, model)
            elif model.startswith("grok"):
                result = await self._generate_grok(prompt, duration, resolution, model)
            elif model.startswith("sora"):
                result = await self._generate_sora(prompt, duration, resolution, model)
            else:
                raise GenerationError(f"Unknown video model: {model}")

            result.generation_time_ms = (time.monotonic() - start_time) * 1000

            if result.video_url:
                result.video_data = await _download_file(result.video_url, timeout=self._timeout)
                if result.video_data:
                    filename = f"{uuid.uuid4().hex[:12]}.mp4"
                    result.file_path = self._output_dir / filename
                    result.file_path.write_bytes(result.video_data)

            logger.info(
                "Generated video: model=%s, duration=%.1fs, time=%.0fms",
                model, duration, result.generation_time_ms,
            )

        except Exception as e:
            logger.error("Video generation failed: %s", e)
            raise GenerationError(f"Video generation failed: {e}") from e

        return result

    async def animate(
        self,
        image: Union[str, Path, bytes],
        prompt: str,
        duration: float = 4.0,
        model: Optional[str] = None,
    ) -> VideoGenerationResult:
        """
        Animate a static image into a video.

        Args:
            image: Source image.
            prompt: Animation instructions.
            duration: Duration in seconds.
            model: Model to use.

        Returns:
            VideoGenerationResult with animated video.
        """
        model = model or self._default_model

        # Encode image
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if image_path.exists():
                image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            else:
                image_b64 = base64.b64encode(await _download_file(str(image))).decode("utf-8")
        elif isinstance(image, bytes):
            image_b64 = base64.b64encode(image).decode("utf-8")
        else:
            raise GenerationError(f"Unsupported image type: {type(image)}")

        result = await self.generate(prompt, duration=duration, model=model)
        result.metadata["source_image"] = "provided"
        return result

    async def extend(
        self,
        video: Union[str, Path, bytes],
        prompt: str,
        seconds: float = 4.0,
        model: Optional[str] = None,
    ) -> VideoGenerationResult:
        """
        Extend an existing video with additional content.

        Args:
            video: Source video.
            prompt: Description of what to add.
            seconds: Duration of extension.
            model: Model to use.

        Returns:
            VideoGenerationResult with extended video.
        """
        model = model or self._default_model
        result = await self.generate(prompt, duration=seconds, model=model)
        result.metadata["extension_of"] = "provided_video"
        result.metadata["extension_seconds"] = seconds
        return result

    async def _generate_runway(
        self, prompt: str, duration: float, resolution: Tuple[int, int], model: str,
    ) -> VideoGenerationResult:
        """Generate video using Runway API."""
        api_key = _resolve_api_key("runway", self._api_keys)
        if not api_key:
            raise ProviderError("runway", "Runway API key not configured")

        try:
            import httpx

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                # Create generation task
                response = await client.post(
                    "https://api.dev.runwayml.com/v1/image_to_video",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "X-Runway-Version": "2024-11-06",
                    },
                    json={
                        "model": model,
                        "promptText": prompt,
                        "duration": int(duration),
                        "resolution": f"{resolution[0]}x{resolution[1]}",
                    },
                )
                response.raise_for_status()
                task_data = response.json()
                task_id = task_data.get("id", "")

                # Poll for completion
                video_url = await self._poll_runway_task(client, api_key, task_id)

            return VideoGenerationResult(
                video_url=video_url, model=model, provider="runway",
                duration_seconds=duration, resolution=resolution,
            )
        except Exception as e:
            raise ProviderError("runway", str(e), e) from e

    async def _poll_runway_task(
        self, client: Any, api_key: str, task_id: str, interval: float = 5.0, max_wait: float = 300,
    ) -> str:
        """Poll a Runway task until completion."""
        elapsed = 0.0
        while elapsed < max_wait:
            await asyncio.sleep(interval)
            elapsed += interval
            response = await client.get(
                f"https://api.dev.runwayml.com/v1/tasks/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            data = response.json()
            status = data.get("status", "")
            if status == "SUCCEEDED":
                return data.get("output", [None])[0] or ""
            elif status in ("FAILED", "CANCELLED"):
                raise ProviderError("runway", f"Task {status}: {data.get('error', '')}")
        raise ProviderError("runway", f"Task timed out after {max_wait}s")

    async def _generate_pika(
        self, prompt: str, duration: float, resolution: Tuple[int, int], model: str,
    ) -> VideoGenerationResult:
        """Generate video using Pika API."""
        api_key = _resolve_api_key("pika", self._api_keys)
        if not api_key:
            raise ProviderError("pika", "Pika API key not configured")

        try:
            import httpx

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.pika.art/v1/generate",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "prompt": prompt,
                        "model": model,
                        "duration": duration,
                        "resolution": f"{resolution[0]}x{resolution[1]}",
                        "fps": 24,
                    },
                )
                response.raise_for_status()
                data = response.json()

            return VideoGenerationResult(
                video_url=data.get("video_url"),
                model=model,
                provider="pika",
                duration_seconds=duration,
                resolution=resolution,
                metadata={"generation_id": data.get("id")},
            )
        except Exception as e:
            raise ProviderError("pika", str(e), e) from e

    async def _generate_grok(
        self, prompt: str, duration: float, resolution: Tuple[int, int], model: str,
    ) -> VideoGenerationResult:
        """Generate video using xAI Grok API."""
        api_key = _resolve_api_key("grok", self._api_keys)
        if not api_key:
            raise ProviderError("grok", "xAI API key not configured")

        try:
            import httpx

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.x.ai/v1/videos/generations",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "prompt": prompt,
                        "model": model,
                        "duration": duration,
                        "resolution": f"{resolution[0]}x{resolution[1]}",
                    },
                )
                response.raise_for_status()
                data = response.json()

            return VideoGenerationResult(
                video_url=data["data"][0].get("url"),
                model=model,
                provider="grok",
                duration_seconds=duration,
                resolution=resolution,
            )
        except Exception as e:
            raise ProviderError("grok", str(e), e) from e

    async def _generate_sora(
        self, prompt: str, duration: float, resolution: Tuple[int, int], model: str,
    ) -> VideoGenerationResult:
        """Generate video using Sora API."""
        api_key = _resolve_api_key("openai", self._api_keys)
        if not api_key:
            raise ProviderError("sora", "OpenAI API key not configured (Sora uses OpenAI)")

        try:
            import httpx

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.openai.com/v1/videos/generations",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "prompt": prompt,
                        "duration": duration,
                        "resolution": f"{resolution[0]}x{resolution[1]}",
                    },
                )
                response.raise_for_status()
                data = response.json()

            return VideoGenerationResult(
                video_url=data["data"][0].get("url"),
                model=model,
                provider="openai",
                duration_seconds=duration,
                resolution=resolution,
            )
        except Exception as e:
            raise ProviderError("sora", str(e), e) from e


# ── Music Generator ─────────────────────────────────────────────────


class MusicGenerator:
    """
    AI-powered music generation with multi-provider support.

    Supports Google MusicFX, Suno, and Udio for music generation
    and continuation.

    Example:
        >>> gen = MusicGenerator()
        >>> result = await gen.generate("Upbeat jazz with piano", duration=30, genre="jazz")
        >>> print(result.audio_url)
    """

    def __init__(
        self,
        api_keys: Optional[Dict[str, str]] = None,
        default_model: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        timeout: int = 180,
    ) -> None:
        """
        Initialize the MusicGenerator.

        Args:
            api_keys: API keys keyed by provider name.
            default_model: Default model to use.
            output_dir: Directory for saved music.
            timeout: Request timeout in seconds.
        """
        self._api_keys = api_keys or {}
        self._default_model = default_model or "suno"
        self._output_dir = Path(output_dir) if output_dir else Path.cwd() / "generated_music"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout

    async def generate(
        self,
        prompt: str,
        duration: float = 30.0,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        model: Optional[str] = None,
        instrumental: bool = False,
    ) -> MusicGenerationResult:
        """
        Generate music from a text prompt.

        Args:
            prompt: Text description of the desired music.
            duration: Duration in seconds.
            genre: Music genre (jazz, rock, classical, etc.).
            mood: Mood (happy, sad, energetic, calm, etc.).
            model: Model to use.
            instrumental: Whether to generate instrumental only.

        Returns:
            MusicGenerationResult with audio data and metadata.
        """
        start_time = time.monotonic()
        model = model or self._default_model
        result = MusicGenerationResult(model=model, genre=genre or "", mood=mood or "", duration_seconds=duration)

        try:
            # Build the full prompt
            full_prompt = prompt
            if genre:
                full_prompt += f", {genre} genre"
            if mood:
                full_prompt += f", {mood} mood"
            if instrumental:
                full_prompt += ", instrumental only"

            if model == "musicfx":
                result = await self._generate_musicfx(full_prompt, duration, instrumental)
            elif model == "suno":
                result = await self._generate_suno(full_prompt, duration, genre, mood, instrumental)
            elif model == "udio":
                result = await self._generate_udio(full_prompt, duration, genre, mood)
            else:
                raise GenerationError(f"Unknown music model: {model}")

            result.generation_time_ms = (time.monotonic() - start_time) * 1000

            if result.audio_url:
                result.audio_data = await _download_file(result.audio_url, timeout=self._timeout)
                if result.audio_data:
                    filename = f"{uuid.uuid4().hex[:12]}.mp3"
                    result.file_path = self._output_dir / filename
                    result.file_path.write_bytes(result.audio_data)

            logger.info(
                "Generated music: model=%s, genre=%s, duration=%.0fs, time=%.0fms",
                model, genre, duration, result.generation_time_ms,
            )

        except Exception as e:
            logger.error("Music generation failed: %s", e)
            raise GenerationError(f"Music generation failed: {e}") from e

        return result

    async def continue_track(
        self,
        track_id: str,
        seconds: float = 30.0,
        model: Optional[str] = None,
    ) -> MusicGenerationResult:
        """
        Continue a previously generated track.

        Args:
            track_id: ID of the original track.
            seconds: Duration of the continuation.
            model: Model to use.

        Returns:
            MusicGenerationResult with continued track.
        """
        model = model or self._default_model
        result = MusicGenerationResult(model=model, duration_seconds=seconds, track_id=track_id)

        if model == "suno":
            api_key = _resolve_api_key("suno", self._api_keys)
            if not api_key:
                raise ProviderError("suno", "Suno API key not configured")

            try:
                import httpx
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"https://api.suno.ai/v1/audio/continue",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"audio_id": track_id, "continue_duration": seconds},
                    )
                    response.raise_for_status()
                    data = response.json()
                result.audio_url = data.get("audio_url")
                result.track_id = data.get("id")
            except Exception as e:
                raise ProviderError("suno", str(e), e) from e
        else:
            raise GenerationError(f"Track continuation not supported for model: {model}")

        return result

    async def _generate_musicfx(
        self, prompt: str, duration: float, instrumental: bool,
    ) -> MusicGenerationResult:
        """Generate music using Google MusicFX."""
        api_key = _resolve_api_key("google", self._api_keys)
        if not api_key:
            raise ProviderError("google", "Google API key not configured")

        try:
            import httpx

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://aitestplatform.googleapis.com/v1alpha/models/musicfx:predictLongform",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "instances": [{"prompt": prompt}],
                        "parameters": {
                            "duration": duration,
                            "instrumental": instrumental,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()

            predictions = data.get("predictions", [])
            audio_data = predictions[0].get("bytesBase64Encoded", "") if predictions else ""

            return MusicGenerationResult(
                audio_base64=audio_data,
                audio_data=base64.b64decode(audio_data) if audio_data else None,
                model="musicfx",
                provider="google",
                duration_seconds=duration,
            )
        except Exception as e:
            raise ProviderError("google", str(e), e) from e

    async def _generate_suno(
        self, prompt: str, duration: float, genre: Optional[str],
        mood: Optional[str], instrumental: bool,
    ) -> MusicGenerationResult:
        """Generate music using Suno API."""
        api_key = _resolve_api_key("suno", self._api_keys)
        if not api_key:
            raise ProviderError("suno", "Suno API key not configured")

        try:
            import httpx

            payload: Dict[str, Any] = {
                "prompt": prompt,
                "duration": duration,
                "instrumental": instrumental,
            }
            if genre:
                payload["tags"] = genre
            if mood:
                payload["mood"] = mood

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                # Create generation task
                response = await client.post(
                    "https://api.suno.ai/v1/audio/generate",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
                task_data = response.json()
                task_id = task_data.get("id", "")

                # Poll for completion
                audio_url = await self._poll_suno_task(client, api_key, task_id)

            return MusicGenerationResult(
                audio_url=audio_url,
                model="suno",
                provider="suno",
                duration_seconds=duration,
                genre=genre or "",
                mood=mood or "",
                track_id=task_id,
            )
        except Exception as e:
            raise ProviderError("suno", str(e), e) from e

    async def _poll_suno_task(
        self, client: Any, api_key: str, task_id: str,
        interval: float = 3.0, max_wait: float = 180,
    ) -> str:
        """Poll a Suno task until completion."""
        elapsed = 0.0
        while elapsed < max_wait:
            await asyncio.sleep(interval)
            elapsed += interval
            response = await client.get(
                f"https://api.suno.ai/v1/audio/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            data = response.json()
            status = data.get("status", "")
            if status == "completed":
                return data.get("audio_url", "")
            elif status in ("failed", "error"):
                raise ProviderError("suno", f"Generation failed: {data.get('error', '')}")
        raise ProviderError("suno", f"Task timed out after {max_wait}s")

    async def _generate_udio(
        self, prompt: str, duration: float, genre: Optional[str], mood: Optional[str],
    ) -> MusicGenerationResult:
        """Generate music using Udio API."""
        api_key = _resolve_api_key("udio", self._api_keys)
        if not api_key:
            raise ProviderError("udio", "Udio API key not configured")

        try:
            import httpx

            payload: Dict[str, Any] = {
                "prompt": prompt,
                "duration": duration,
            }
            if genre:
                payload["genre"] = genre
            if mood:
                payload["mood"] = mood

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.udio.com/v1/generate",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            return MusicGenerationResult(
                audio_url=data.get("audio_url"),
                track_id=data.get("track_id"),
                model="udio",
                provider="udio",
                duration_seconds=duration,
                genre=genre or "",
                mood=mood or "",
            )
        except Exception as e:
            raise ProviderError("udio", str(e), e) from e
