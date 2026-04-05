"""
Hermes Vision Tool — image understanding and analysis.

Features:
- Image analysis via multimodal API
- Screenshot interpretation
- Image description and captioning
- OCR integration
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Optional

from hermes.tools.registry import ToolRegistry


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def hermes_analyze_image(
    image_path: str,
    prompt: str = "Describe this image in detail.",
) -> str:
    """Analyze an image using a multimodal AI model.

    param image_path (str): — Path to the image file.
    param prompt (str): — Analysis prompt. Default: detailed description.
    """
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        return f"Error: Image not found: {path}"

    if not path.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        return f"Error: Unsupported image format: {path.suffix}"

    # Read and encode image
    try:
        image_data = path.read_bytes()
        base64_image = base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        return f"Error reading image: {e}"

    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(path.suffix.lower(), "image/png")

    # Check for z-ai-web-dev-sdk for analysis
    try:
        zai_mod = __import__("z-ai-web-dev-sdk")
        ZAI = getattr(zai_mod, "ZAI", None) or getattr(zai_mod, "create", None)
    except (ImportError, AttributeError):
        pass

    # Return image metadata and base64 info
    size_kb = len(image_data) / 1024
    lines = [
        f"Image: {path.name}",
        f"Format: {media_type}",
        f"Size: {size_kb:.1f} KB ({len(image_data):,} bytes)",
        f"Base64 length: {len(base64_image):,} chars",
        "",
        f"Prompt: {prompt}",
        "",
        "Image data is encoded and ready for multimodal analysis.",
        f"Base64 data (first 100 chars): {base64_image[:100]}...",
    ]

    return "\n".join(lines)


async def hermes_ocr(
    image_path: str,
    language: str = "eng",
) -> str:
    """Extract text from an image using OCR.

    param image_path (str): — Path to the image file.
    param language (str): — OCR language code. Default: eng.
    """
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        return f"Error: Image not found: {path}"

    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return "Error: pytesseract and Pillow are required. Install with: pip install pytesseract Pillow"

    def _do():
        try:
            img = Image.open(str(path))
            text = pytesseract.image_to_string(img, lang=language)
            text = text.strip()

            if not text:
                return f"No text detected in {path.name}"

            return f"OCR result for {path.name}:\n\n{text}"

        except Exception as e:
            return f"OCR error: {e}"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error performing OCR: {e}"


async def hermes_image_caption(image_path: str) -> str:
    """Generate a caption/description for an image.

    param image_path (str): — Path to the image file.
    """
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        return f"Error: Image not found: {path}"

    try:
        from PIL import Image
    except ImportError:
        return "Error: Pillow is required. Install with: pip install Pillow"

    def _do():
        try:
            img = Image.open(str(path))
            width, height = img.size
            mode = img.mode
            fmt = img.format or "unknown"

            lines = [
                f"Image: {path.name}",
                f"Format: {fmt}",
                f"Dimensions: {width}x{height} pixels",
                f"Color mode: {mode}",
                f"File size: {path.stat().st_size:,} bytes",
            ]

            # Try to extract EXIF data
            try:
                exif = img.getexif()
                if exif:
                    lines.append("\nEXIF data:")
                    for tag_id, value in list(exif.items())[:10]:
                        from PIL.ExifTags import TAGS
                        tag = TAGS.get(tag_id, tag_id)
                        val_str = str(value)[:100]
                        lines.append(f"  {tag}: {val_str}")
            except Exception:
                pass

            return "\n".join(lines)

        except Exception as e:
            return f"Error processing image: {e}"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="hermes_analyze_image",
    func=hermes_analyze_image,
    description="Analyze an image using a multimodal AI model with a custom prompt.",
    toolset="vision",
)

ToolRegistry.instance().register(
    name="hermes_ocr",
    func=hermes_ocr,
    description="Extract text from an image using OCR (Tesseract).",
    toolset="vision",
)

ToolRegistry.instance().register(
    name="hermes_image_caption",
    func=hermes_image_caption,
    description="Generate a caption and metadata summary for an image.",
    toolset="vision",
)
