"""
Atlas Image Processor — Comprehensive image processing tools.

Provides image loading, saving, resizing, cropping, rotation, compression,
format conversion, analysis, and blending. Uses Pillow when available
with graceful fallbacks.
"""

from __future__ import annotations

import asyncio
import colorsys
import io
import logging
import math
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Attempt to import Pillow
try:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False
    logger.debug("Pillow not available; image processing will be limited")


class ImageFormat(Enum):
    """Supported image output formats."""
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"
    BMP = "bmp"
    TIFF = "tiff"
    ICO = "ico"
    AVIF = "avif"


class BlendMode(Enum):
    """Image blending modes."""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"
    HARD_LIGHT = "hard_light"
    DIFFERENCE = "difference"
    ADDITION = "addition"
    SUBTRACT = "subtract"


class ImageLoadError(Exception):
    """Raised when image loading fails."""
    pass


class ImageProcessingError(Exception):
    """Raised when image processing fails."""
    pass


@dataclass
class ImageAnalysis:
    """Results of image analysis."""
    width: int = 0
    height: int = 0
    mode: str = ""
    format: str = ""
    file_size: int = 0
    aspect_ratio: float = 0.0
    is_transparent: bool = False
    is_animated: bool = False
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    brightness: float = 0.0
    contrast: float = 0.0
    color_count: int = 0
    has_alpha: bool = False
    exif_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to a dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "format": self.format,
            "file_size": self.file_size,
            "aspect_ratio": round(self.aspect_ratio, 4),
            "is_transparent": self.is_transparent,
            "is_animated": self.is_animated,
            "dominant_colors": self.dominant_colors,
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "color_count": self.color_count,
            "has_alpha": self.has_alpha,
            "has_exif": self.exif_data is not None,
        }


@dataclass
class ColorInfo:
    """Color information extracted from an image."""
    rgb: Tuple[int, int, int]
    hex: str
    percentage: float
    hsl: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        r, g, b = self.rgb
        self.hex = f"#{r:02x}{g:02x}{b:02x}"
        self.hsl = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)


class ImageProcessor:
    """
    Comprehensive image processing class.

    Provides tools for loading, saving, resizing, cropping, rotating,
    compressing, format conversion, analysis, and blending of images.
    Uses Pillow when available with graceful fallbacks for environments
    where Pillow is not installed.

    Example:
        >>> processor = ImageProcessor()
        >>> image = await processor.load("photo.jpg")
        >>> resized = await processor.resize(image, 800, 600)
        >>> await processor.save(resized, "output.webp", format="webp", quality=85)
    """

    def __init__(
        self,
        default_quality: int = 90,
        default_format: str = "PNG",
        max_dimension: int = 10000,
        cache_size: int = 50,
    ) -> None:
        """
        Initialize the ImageProcessor.

        Args:
            default_quality: Default quality for lossy formats (1-100).
            default_format: Default output format.
            max_dimension: Maximum allowed dimension for any side.
            cache_size: Number of recently loaded images to cache.
        """
        self._default_quality = max(1, min(100, default_quality))
        self._default_format = default_format
        self._max_dimension = max_dimension
        self._cache: Dict[str, Any] = {}
        self._cache_order: List[str] = []
        self._cache_size = cache_size

        if not HAS_PILLOW:
            logger.warning(
                "Pillow is not installed. Image processing will use FFmpeg "
                "fallback where available. Install with: pip install Pillow"
            )

    @property
    def pillow_available(self) -> bool:
        """Whether Pillow is available for image processing."""
        return HAS_PILLOW

    async def load(
        self,
        path_or_url: Union[str, Path],
        mode: Optional[str] = None,
    ) -> Any:
        """
        Load an image from a file path or URL.

        Args:
            path_or_url: Local file path or HTTP/HTTPS URL.
            mode: Target color mode (e.g., "RGB", "RGBA", "L"). None keeps original.

        Returns:
            PIL Image object.

        Raises:
            ImageLoadError: If the image cannot be loaded.
        """
        if not HAS_PILLOW:
            raise ImageProcessingError("Pillow is required for image loading")

        path_str = str(path_or_url)

        # Check cache
        if path_str in self._cache:
            logger.debug("Cache hit for: %s", path_str)
            cached_img = self._cache[path_str]
            if mode and cached_img.mode != mode:
                return cached_img.convert(mode)
            return cached_img

        image = None

        if path_str.startswith(("http://", "https://")):
            image = await self._load_from_url(path_str)
        else:
            path = Path(path_str)
            if not path.exists():
                raise ImageLoadError(f"Image file not found: {path}")
            try:
                image = PILImage.open(path)
            except Exception as e:
                raise ImageLoadError(f"Failed to open image: {e}") from e

        if image is None:
            raise ImageLoadError(f"Could not load image from: {path_str}")

        # Convert mode if requested
        if mode and image.mode != mode:
            image = image.convert(mode)

        # Update cache
        self._add_to_cache(path_str, image)

        return image

    async def _load_from_url(self, url: str) -> Any:
        """Load an image from a URL using httpx or urllib."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                return PILImage.open(io.BytesIO(response.content))
        except ImportError:
            # Fallback to urllib
            import urllib.request
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = resp.read()
                    return PILImage.open(io.BytesIO(data))
            except Exception as e:
                raise ImageLoadError(f"Failed to download image from URL: {e}") from e
        except Exception as e:
            raise ImageLoadError(f"Failed to download image from URL: {e}") from e

    async def save(
        self,
        image: Any,
        path: Union[str, Path],
        format: Optional[str] = None,
        quality: Optional[int] = None,
        optimize: bool = True,
        progressive: bool = False,
        **kwargs: Any,
    ) -> Path:
        """
        Save an image to a file.

        Args:
            image: PIL Image object.
            path: Output file path.
            format: Output format. Auto-detected from extension if None.
            quality: Quality for lossy formats (1-100). Uses default if None.
            optimize: Whether to optimize the output file.
            progressive: Create progressive JPEG if applicable.
            **kwargs: Additional format-specific options.

        Returns:
            Path to the saved file.

        Raises:
            ImageProcessingError: If saving fails.
        """
        if not HAS_PILLOW:
            raise ImageProcessingError("Pillow is required for image saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = self._default_format

        quality = quality if quality is not None else self._default_quality

        # Format-specific options
        save_kwargs: Dict[str, Any] = {"optimize": optimize}
        if format.upper() in ("JPEG", "JPG"):
            save_kwargs["quality"] = quality
            save_kwargs["progressive"] = progressive
            # JPEG doesn't support alpha
            if image.mode in ("RGBA", "LA", "P"):
                background = PILImage.new("RGB", image.size, (255, 255, 255))
                if image.mode == "P":
                    image = image.convert("RGBA")
                if image.mode in ("RGBA", "LA"):
                    background.paste(image, mask=image.split()[-1])
                else:
                    background.paste(image)
                image = background
        elif format.upper() == "WEBP":
            save_kwargs["quality"] = quality
            if image.mode == "RGBA":
                save_kwargs["lossless"] = kwargs.get("lossless", False)
        elif format.upper() == "PNG":
            save_kwargs["compress_level"] = kwargs.get("compress_level", 6)
        elif format.upper() == "GIF":
            save_kwargs["optimize"] = True
        elif format.upper() == "TIFF":
            save_kwargs["compression"] = kwargs.get("compression", "tiff_deflate")

        save_kwargs.update(kwargs)

        try:
            image.save(str(path), format=format.upper(), **save_kwargs)
            logger.info("Saved image: %s (%s, quality=%d)", path, format, quality)
            return path
        except Exception as e:
            raise ImageProcessingError(f"Failed to save image: {e}") from e

    async def resize(
        self,
        image: Any,
        width: int,
        height: int,
        maintain_aspect: bool = True,
        resample: str = "lanczos",
    ) -> Any:
        """
        Resize an image.

        Args:
            image: PIL Image object.
            width: Target width in pixels.
            height: Target height in pixels.
            maintain_aspect: Maintain aspect ratio (fit within bounds).
            resample: Resampling filter (lanczos, bilinear, nearest, bicubic).

        Returns:
            Resized PIL Image object.
        """
        if not HAS_PILLOW:
            raise ImageProcessingError("Pillow is required for resizing")

        resample_map = {
            "lanczos": PILImage.LANCZOS,
            "bilinear": PILImage.BILINEAR,
            "nearest": PILImage.NEAREST,
            "bicubic": PILImage.BICUBIC,
        }
        filter_type = resample_map.get(resample, PILImage.LANCZOS)

        orig_w, orig_h = image.size

        if maintain_aspect:
            ratio = min(width / orig_w, height / orig_h)
            new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        else:
            new_w, new_h = width, height

        # Clamp to max dimension
        if new_w > self._max_dimension or new_h > self._max_dimension:
            ratio = min(self._max_dimension / new_w, self._max_dimension / new_h)
            new_w, new_h = int(new_w * ratio), int(new_h * ratio)

        resized = image.resize((new_w, new_h), filter_type)
        logger.debug("Resized %dx%d → %dx%d", orig_w, orig_h, new_w, new_h)
        return resized

    async def crop(
        self,
        image: Any,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Any:
        """
        Crop an image.

        Args:
            image: PIL Image object.
            x: Left edge x coordinate.
            y: Top edge y coordinate.
            width: Width of the crop region.
            height: Height of the crop region.

        Returns:
            Cropped PIL Image object.
        """
        if not HAS_PILLOW:
            raise ImageProcessingError("Pillow is required for cropping")

        orig_w, orig_h = image.size
        # Clamp to image bounds
        x = max(0, min(x, orig_w - 1))
        y = max(0, min(y, orig_h - 1))
        width = min(width, orig_w - x)
        height = min(height, orig_h - y)

        cropped = image.crop((x, y, x + width, y + height))
        logger.debug("Cropped from (%d,%d) size %dx%d", x, y, width, height)
        return cropped

    async def rotate(
        self,
        image: Any,
        degrees: float,
        expand: bool = True,
        fill_color: Optional[Tuple[int, int, int]] = None,
    ) -> Any:
        """
        Rotate an image.

        Args:
            image: PIL Image object.
            degrees: Rotation angle in degrees (counter-clockwise).
            expand: Expand output to fit rotated image.
            fill_color: Fill color for uncovered areas. None uses transparent.

        Returns:
            Rotated PIL Image object.
        """
        if not HAS_PILLOW:
            raise ImageProcessingError("Pillow is required for rotation")

        if fill_color is not None and image.mode != "RGB":
            image = image.convert("RGB")

        rotated = image.rotate(degrees, expand=expand, fillcolor=fill_color)
        logger.debug("Rotated by %.1f degrees", degrees)
        return rotated

    async def compress(
        self,
        image: Any,
        quality: int = 75,
        format: Optional[str] = None,
    ) -> bytes:
        """
        Compress an image and return bytes.

        Args:
            image: PIL Image object.
            quality: Compression quality (1-100).
            format: Output format for compression.

        Returns:
            Compressed image bytes.
        """
        if not HAS_PILLOW:
            raise ImageProcessingError("Pillow is required for compression")

        if format is None:
            format = "JPEG"

        buffer = io.BytesIO()
        save_kwargs = {"format": format.upper(), "quality": quality, "optimize": True}

        if format.upper() == "JPEG" and image.mode in ("RGBA", "LA"):
            bg = PILImage.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[-1])
            bg.save(buffer, **save_kwargs)
        elif format.upper() == "WEBP":
            save_kwargs["quality"] = quality
            image.save(buffer, **save_kwargs)
        else:
            image.save(buffer, **save_kwargs)

        return buffer.getvalue()

    async def convert_format(
        self,
        image: Any,
        target_format: str,
        quality: Optional[int] = None,
    ) -> bytes:
        """
        Convert an image to a different format and return bytes.

        Args:
            image: PIL Image object.
            target_format: Target format (e.g., "webp", "png").
            quality: Quality for lossy formats.

        Returns:
            Image bytes in the target format.
        """
        return await self.compress(image, quality or self._default_quality, target_format)

    async def analyze(
        self,
        image_or_path: Union[Any, str, Path],
        color_samples: int = 10,
    ) -> ImageAnalysis:
        """
        Perform basic image analysis.

        Extracts dimensions, color mode, dominant colors, brightness,
        contrast, and other metadata from the image.

        Args:
            image_or_path: PIL Image object or path to image file.
            color_samples: Number of dominant colors to extract.

        Returns:
            ImageAnalysis dataclass with all extracted information.
        """
        if not HAS_PILLOW:
            raise ImageProcessingError("Pillow is required for image analysis")

        # Load if path provided
        if isinstance(image_or_path, (str, Path)):
            path = Path(image_or_path)
            file_size = path.stat().st_size if path.exists() else 0
            image = await self.load(path)
        else:
            image = image_or_path
            file_size = 0

        analysis = ImageAnalysis()
        analysis.width = image.width
        analysis.height = image.height
        analysis.mode = image.mode
        analysis.format = image.format or ""
        analysis.file_size = file_size
        analysis.aspect_ratio = image.width / image.height if image.height > 0 else 0
        analysis.has_alpha = "A" in image.mode

        # Check transparency
        if image.mode == "RGBA":
            extrema = image.getchannel("A").getextrema()
            analysis.is_transparent = extrema[0] < 255

        # Check animated (GIF, WEBP)
        if hasattr(image, "is_animated"):
            analysis.is_animated = getattr(image, "is_animated", False)
            if hasattr(image, "n_frames"):
                analysis.frame_count = getattr(image, "n_frames", 1)

        # Extract dominant colors
        analysis.dominant_colors = self._extract_dominant_colors(image, color_samples)
        analysis.color_count = len(analysis.dominant_colors)

        # Calculate brightness and contrast
        if image.mode in ("RGB", "RGBA", "L"):
            rgb_image = image.convert("RGB")
            pixels = list(rgb_image.getdata())
            if pixels:
                brightness_values = []
                for r, g, b in pixels:
                    brightness_values.append(0.299 * r + 0.587 * g + 0.114 * b)
                analysis.brightness = sum(brightness_values) / len(brightness_values) / 255
                if len(brightness_values) > 1:
                    mean_b = sum(brightness_values) / len(brightness_values)
                    variance = sum((b - mean_b) ** 2 for b in brightness_values) / len(brightness_values)
                    analysis.contrast = math.sqrt(variance) / 128

        # Extract EXIF data if available
        try:
            if hasattr(image, "_getexif"):
                exif_raw = image._getexif()
                if exif_raw:
                    from PIL.ExifTags import TAGS
                    analysis.exif_data = {
                        TAGS.get(tag, tag): value
                        for tag, value in exif_raw.items()
                        if isinstance(value, (str, int, float, bytes))
                    }
        except Exception as e:
            logger.debug("EXIF extraction failed: %s", e)

        return analysis

    def _extract_dominant_colors(
        self, image: Any, num_colors: int = 10,
    ) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from an image using pixel sampling.

        Uses a simple frequency-based approach rather than k-means
        for performance. Samples a grid of pixels and counts colors.
        """
        # Resize to small size for fast analysis
        small = image.copy()
        small.thumbnail((64, 64), PILImage.LANCZOS)
        rgb = small.convert("RGB")

        # Sample pixels and count colors (with quantization)
        pixel_counts: Counter = Counter()
        for pixel in rgb.getdata():
            # Quantize to reduce color space (round to nearest 16)
            quantized = (pixel[0] // 32 * 32, pixel[1] // 32 * 32, pixel[2] // 32 * 32)
            pixel_counts[quantized] += 1

        total = sum(pixel_counts.values())
        top_colors = pixel_counts.most_common(num_colors)

        result = []
        for (r, g, b), count in top_colors:
            # Clamp to valid range
            r = min(255, max(0, r))
            g = min(255, max(0, g))
            b = min(255, max(0, b))
            result.append((r, g, b))

        return result

    async def blend(
        self,
        image1: Any,
        image2: Any,
        mode: BlendMode = BlendMode.NORMAL,
        opacity: float = 0.5,
    ) -> Any:
        """
        Blend two images together.

        Args:
            image1: Base PIL Image object.
            image2: Overlay PIL Image object.
            mode: Blending mode.
            opacity: Opacity of the overlay (0.0-1.0).

        Returns:
            Blended PIL Image object.
        """
        if not HAS_PILLOW:
            raise ImageProcessingError("Pillow is required for blending")

        # Ensure same size
        if image1.size != image2.size:
            image2 = image2.resize(image1.size, PILImage.LANCZOS)

        # Ensure same mode
        if image1.mode != image2.mode:
            image1 = image1.convert("RGBA")
            image2 = image2.convert("RGBA")

        if mode == BlendMode.NORMAL:
            # PIL alpha composite
            overlay = image2.copy()
            if overlay.mode == "RGBA":
                overlay.putalpha(int(opacity * 255))
            else:
                overlay = overlay.convert("RGBA")
                overlay.putalpha(int(opacity * 255))
            base = image1.convert("RGBA")
            result = PILImage.alpha_composite(base, overlay)
        elif mode == BlendMode.MULTIPLY:
            result = PILImageChops.multiply(image1, image2)
        elif mode == BlendMode.SCREEN:
            result = PILImageChops.screen(image1, image2)
        elif mode == BlendMode.OVERLAY:
            result = PILImageChops.overlay(image1, image2)
        elif mode == BlendMode.DIFFERENCE:
            result = PILImageChops.difference(image1, image2)
        elif mode == BlendMode.ADDITION:
            result = PILImageChops.add(image1, image2)
        elif mode == BlendMode.SUBTRACT:
            result = PILImageChops.subtract(image1, image2)
        elif mode == BlendMode.SOFT_LIGHT:
            result = self._soft_light_blend(image1, image2)
        elif mode == BlendMode.HARD_LIGHT:
            result = self._hard_light_blend(image1, image2)
        else:
            result = PILImage.blend(image1, image2, opacity)

        logger.debug("Blended images with mode=%s, opacity=%.2f", mode.value, opacity)
        return result

    def _soft_light_blend(self, image1: Any, image2: Any) -> Any:
        """Apply soft light blending."""
        # Convert to float arrays for pixel-level blending
        import array
        img1 = image1.convert("RGB")
        img2 = image2.convert("RGB")
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)

        pixels1 = list(img1.getdata())
        pixels2 = list(img2.getdata())
        result_pixels = []

        for (r1, g1, b1), (r2, g2, b2) in zip(pixels1, pixels2):
            def soft_light(a, b):
                if b <= 128:
                    return a - (255 - 2 * b) * a * (255 - a) / (255 * 255)
                else:
                    return a + (2 * b - 255) * (self._sqrt_approx(a / 255) * 255 - a) / 255

            result_pixels.append((
                int(max(0, min(255, soft_light(r1, r2)))),
                int(max(0, min(255, soft_light(g1, g2)))),
                int(max(0, min(255, soft_light(b1, b2)))),
            ))

        result = PILImage.new("RGB", img1.size)
        result.putdata(result_pixels)
        return result

    def _hard_light_blend(self, image1: Any, image2: Any) -> Any:
        """Apply hard light blending."""
        img1 = image1.convert("RGB")
        img2 = image2.convert("RGB")
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)

        pixels1 = list(img1.getdata())
        pixels2 = list(img2.getdata())
        result_pixels = []

        for (r1, g1, b1), (r2, g2, b2) in zip(pixels1, pixels2):
            def hard_light(a, b):
                if b <= 128:
                    return 2 * a * b / 255
                else:
                    return 255 - 2 * (255 - a) * (255 - b) / 255

            result_pixels.append((
                int(max(0, min(255, hard_light(r1, r2)))),
                int(max(0, min(255, hard_light(g1, g2)))),
                int(max(0, min(255, hard_light(b1, b2)))),
            ))

        result = PILImage.new("RGB", img1.size)
        result.putdata(result_pixels)
        return result

    @staticmethod
    def _sqrt_approx(x: float) -> float:
        """Approximate square root for blending calculations."""
        if x <= 0:
            return 0
        # Newton's method approximation
        y = x
        for _ in range(8):
            y = (y + x / y) / 2
        return y

    async def add_text(
        self,
        image: Any,
        text: str,
        position: Tuple[int, int] = (10, 10),
        font_size: int = 24,
        color: Tuple[int, int, int] = (255, 255, 255),
        stroke_color: Optional[Tuple[int, int, int]] = None,
        stroke_width: int = 2,
        font_path: Optional[str] = None,
    ) -> Any:
        """
        Add text overlay to an image.

        Args:
            image: PIL Image object.
            text: Text to render.
            position: (x, y) position for the text.
            font_size: Font size in pixels.
            color: Text color as RGB tuple.
            stroke_color: Outline color. None for no outline.
            stroke_width: Outline width.
            font_path: Path to TTF font file. None uses default.

        Returns:
            Image with text overlay.
        """
        if not HAS_PILLOW:
            raise ImageProcessingError("Pillow is required for text overlay")

        img = image.copy().convert("RGBA")
        draw = ImageDraw.Draw(img)

        font = None
        if font_path and Path(font_path).exists():
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                pass
        if font is None:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except (IOError, OSError):
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except (IOError, OSError):
                    font = ImageFont.load_default()

        draw.text(
            position, text, fill=color, font=font,
            stroke_fill=stroke_color, stroke_width=stroke_width,
        )
        return img

    async def add_watermark(
        self,
        image: Any,
        watermark_text: str,
        opacity: float = 0.3,
        position: str = "center",
        font_size: int = 36,
        tile: bool = False,
    ) -> Any:
        """
        Add a watermark to an image.

        Args:
            image: PIL Image object.
            watermark_text: Text for the watermark.
            opacity: Watermark opacity (0.0-1.0).
            position: Position ("center", "bottom-right", "bottom-left", etc.).
            font_size: Font size in pixels.
            tile: Repeat watermark across the entire image.

        Returns:
            Watermarked image.
        """
        if not HAS_PILLOW:
            raise ImageProcessingError("Pillow is required for watermarking")

        img = image.copy().convert("RGBA")
        watermark_layer = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark_layer)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

        text_color = (255, 255, 255, int(opacity * 255))

        if tile:
            for y in range(0, img.height, font_size * 4):
                for x in range(0, img.width, font_size * 8):
                    draw.text((x, y), watermark_text, fill=text_color, font=font)
        else:
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            positions = {
                "center": ((img.width - text_w) // 2, (img.height - text_h) // 2),
                "bottom-right": (img.width - text_w - 20, img.height - text_h - 20),
                "bottom-left": (20, img.height - text_h - 20),
                "top-right": (img.width - text_w - 20, 20),
                "top-left": (20, 20),
            }
            pos = positions.get(position, positions["center"])
            draw.text(pos, watermark_text, fill=text_color, font=font)

        # Rotate watermark -45 degrees for style
        if not tile:
            watermark_layer = watermark_layer.rotate(45, expand=True, resample=PILImage.BICUBIC)
            # Center the rotated watermark
            offset_x = (img.width - watermark_layer.width) // 2
            offset_y = (img.height - watermark_layer.height) // 2

        result = PILImage.alpha_composite(img, watermark_layer)
        return result

    async def create_thumbnail(
        self,
        image: Any,
        size: Tuple[int, int] = (128, 128),
        fit: bool = True,
    ) -> Any:
        """
        Create a thumbnail of the image.

        Args:
            image: PIL Image object.
            size: Maximum (width, height) for the thumbnail.
            fit: Whether to maintain aspect ratio.

        Returns:
            Thumbnail PIL Image object.
        """
        if not HAS_PILLOW:
            raise ImageProcessingError("Pillow is required for thumbnail creation")

        thumb = image.copy()
        if fit:
            thumb.thumbnail(size, PILImage.LANCZOS)
        else:
            thumb = thumb.resize(size, PILImage.LANCZOS)
        return thumb

    async def to_base64(self, image: Any, format: str = "PNG") -> str:
        """
        Convert an image to a base64-encoded string.

        Args:
            image: PIL Image object.
            format: Image format for encoding.

        Returns:
            Base64-encoded image string.
        """
        import base64
        buffer = io.BytesIO()
        if image.mode in ("RGBA", "LA") and format.upper() == "JPEG":
            image = image.convert("RGB")
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _add_to_cache(self, key: str, image: Any) -> None:
        """Add an image to the LRU cache."""
        if key in self._cache:
            self._cache_order.remove(key)
        self._cache[key] = image
        self._cache_order.append(key)

        # Evict oldest entries
        while len(self._cache) > self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

    def clear_cache(self) -> None:
        """Clear the image cache."""
        self._cache.clear()
        self._cache_order.clear()
        logger.debug("Image cache cleared")


# PIL ImageChops import (used in blend method)
try:
    from PIL import ImageChops as PILImageChops
except ImportError:
    PILImageChops = None  # type: ignore[assignment]
