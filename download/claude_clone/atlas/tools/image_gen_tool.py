"""
Atlas Image Generation Tool — AI-powered image generation.

Features:
- Text-to-image generation
- Size and style control
- Image variation and editing
- Gallery management
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from atlas.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Gallery storage
# ---------------------------------------------------------------------------

_GALLERY_DIR = Path.home() / ".claude_clone" / "gallery"


def _ensure_gallery() -> None:
    _GALLERY_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


_VALID_SIZES = [
    "1024x1024", "768x1344", "864x1152",
    "1344x768", "1152x864", "1440x720", "720x1440",
]


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def atlas_generate_image(
    prompt: str,
    size: str = "1024x1024",
    output_path: str = "",
    style: str = "",
) -> str:
    """Generate an image from a text prompt using AI.

    param prompt (str): — Text description of the image to generate.
    param size (str): — Image size. Options: 1024x1024, 768x1344, 864x1152, 1344x768, 1152x864. Default: 1024x1024.
    param output_path (str): — Output file path. Default: auto-generated.
    param style (str): — Optional style modifier (e.g., photorealistic, anime, oil painting).
    """
    if size not in _VALID_SIZES:
        return f"Error: Invalid size '{size}'. Valid sizes: {', '.join(_VALID_SIZES)}"

    full_prompt = prompt
    if style:
        full_prompt = f"{style} style: {prompt}"

    _ensure_gallery()

    if not output_path:
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in prompt[:30])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(_GALLERY_DIR / f"gen_{timestamp}_{safe_name}.png")

    try:
        # Try z-ai-generate CLI
        import subprocess

        proc = await asyncio.create_subprocess_exec(
            "z-ai-generate", "-p", full_prompt, "-o", output_path, "-s", size,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

        if proc.returncode == 0 and Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            return (
                f"Image generated successfully!\n"
                f"Prompt: {prompt}\n"
                f"Size: {size}\n"
                f"Output: {output_path} ({file_size:,} bytes)"
            )
        else:
            err = stderr.decode(errors="replace").strip() if stderr else "Unknown error"
            return f"Image generation failed: {err}"

    except asyncio.TimeoutError:
        return "Error: Image generation timed out (120s limit)"
    except FileNotFoundError:
        return (
            "Error: z-ai-generate CLI not found. "
            "The image generation tool requires the z-ai-web-dev-sdk CLI to be available."
        )
    except Exception as e:
        return f"Error generating image: {e}"


async def atlas_image_gallery(limit: int = 20) -> str:
    """List images in the generated image gallery.

    param limit (int): — Max images to list. Default: 20.
    """
    _ensure_gallery()

    images = sorted(_GALLERY_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not images:
        return "Gallery is empty. Use atlas_generate_image to create images."

    images = images[:limit]
    lines = [f"Gallery — {len(images)} images (of {len(list(_GALLERY_DIR.glob('*')))} total):\n"]

    for img in images:
        try:
            size = img.stat().st_size
            mtime = datetime.fromtimestamp(img.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  {img.name} ({size:,} bytes, {mtime})")
        except OSError:
            lines.append(f"  {img.name}")

    return "\n".join(lines)


async def atlas_image_info(image_path: str) -> str:
    """Get detailed information about an image file.

    param image_path (str): — Path to the image.
    """
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        return f"Error: File not found: {path}"

    try:
        from PIL import Image
    except ImportError:
        # Fallback to basic file info
        stat = path.stat()
        return (
            f"File: {path.name}\n"
            f"Path: {path}\n"
            f"Size: {stat.st_size:,} bytes\n"
            f"Modified: {datetime.fromtimestamp(stat.st_mtime).isoformat()}\n"
            f"(Install Pillow for detailed image info: pip install Pillow)"
        )

    def _do():
        try:
            img = Image.open(str(path))
            lines = [
                f"File: {path.name}",
                f"Path: {path}",
                f"Format: {img.format or 'unknown'}",
                f"Dimensions: {img.width}x{img.height}",
                f"Color mode: {img.mode}",
                f"File size: {path.stat().st_size:,} bytes",
            ]

            if hasattr(img, "info") and img.info:
                lines.append("Additional info:")
                for k, v in list(img.info.items())[:5]:
                    lines.append(f"  {k}: {str(v)[:100]}")

            return "\n".join(lines)

        except Exception as e:
            return f"Error reading image info: {e}"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error: {e}"


async def atlas_image_delete(image_path: str) -> str:
    """Delete an image from the gallery or filesystem.

    param image_path (str): — Path to the image to delete.
    """
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        return f"Error: File not found: {path}"

    try:
        path.unlink()
        return f"Deleted: {path}"
    except Exception as e:
        return f"Error deleting {image_path}: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="atlas_generate_image",
    func=atlas_generate_image,
    description="Generate an AI image from a text prompt with size and style control.",
    toolset="media",
)

ToolRegistry.instance().register(
    name="atlas_image_gallery",
    func=atlas_image_gallery,
    description="List generated images in the gallery.",
    toolset="media",
)

ToolRegistry.instance().register(
    name="atlas_image_info",
    func=atlas_image_info,
    description="Get detailed information about an image file.",
    toolset="media",
)

ToolRegistry.instance().register(
    name="atlas_image_delete",
    func=atlas_image_delete,
    description="Delete an image file from the gallery.",
    toolset="media",
)
