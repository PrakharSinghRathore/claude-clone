"""
Hermes TTS Tool — text-to-speech synthesis.

Features:
- Edge TTS (free, primary)
- Voice selection and customization
- Audio file output (MP3)
- Speed and volume control
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional

from hermes.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Known Edge TTS voices
# ---------------------------------------------------------------------------

_EDGE_VOICES = {
    "en-US": ["en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural", "en-US-DavisNeural"],
    "en-GB": ["en-GB-SoniaNeural", "en-GB-RyanNeural"],
    "es-ES": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"],
    "fr-FR": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
    "de-DE": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
    "ja-JP": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
    "zh-CN": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
}

_DEFAULT_VOICE = "en-US-AriaNeural"


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def hermes_tts(
    text: str,
    voice: str = "",
    output_path: str = "",
    rate: str = "+0%",
    volume: str = "+0%",
) -> str:
    """Convert text to speech and save as audio file.

    param text (str): — Text to convert to speech.
    param voice (str): — Voice name. Default: en-US-AriaNeural.
    param output_path (str): — Output MP3 file path. Default: auto-generated.
    param rate (str): — Speech rate adjustment (e.g., '+20%', '-10%'). Default: +0%.
    param volume (str): — Volume adjustment (e.g., '+50%', '-20%'). Default: +0%.
    """
    if not voice:
        voice = _DEFAULT_VOICE

    if not output_path:
        output_path = str(Path(tempfile.gettempdir()) / f"tts_{os.getpid()}_{id(text) % 10000}.mp3")

    try:
        import edge_tts
    except ImportError:
        return "Error: edge-tts is required. Install with: pip install edge-tts"

    def _do():
        # Create communicate instance
        communicate = edge_tts.Communicate(text, voice)

        # Build SSML for rate/volume
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis">'
            f'<voice name="{voice}">'
            f'<prosody rate="{rate}" volume="{volume}">'
            f'{text}'
            f'</prosody></voice></speak>'
        )

        communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)

        loop = asyncio.new_event_loop()

        async def _save():
            await communicate.save(output_path)

        loop.run_until_complete(_save())
        loop.close()

        if Path(output_path).exists():
            size = Path(output_path).stat().st_size
            return f"Audio saved to {output_path} ({size:,} bytes, voice: {voice}, rate: {rate}, volume: {volume})"
        return f"TTS completed but file not found at {output_path}"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error generating speech: {e}"


async def hermes_tts_voices(locale: str = "") -> str:
    """List available TTS voices.

    param locale (str): — Filter by locale (e.g., en-US, es-ES). Empty = all.
    """
    if locale:
        voices = _EDGE_VOICES.get(locale, [])
        if not voices:
            return f"No voices found for locale '{locale}'. Available locales: {', '.join(_EDGE_VOICES.keys())}"
        return f"Available voices for {locale}:\n" + "\n".join(f"  - {v}" for v in voices)

    lines = ["Available TTS voices by locale:\n"]
    for loc, voices in sorted(_EDGE_VOICES.items()):
        lines.append(f"  {loc}: {', '.join(voices)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="hermes_tts",
    func=hermes_tts,
    description="Convert text to speech and save as an MP3 audio file using Edge TTS.",
    toolset="audio",
)

ToolRegistry.instance().register(
    name="hermes_tts_voices",
    func=hermes_tts_voices,
    description="List available text-to-speech voices by locale.",
    toolset="audio",
)
