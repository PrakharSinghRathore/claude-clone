"""
Hermes Transcription Tool — speech-to-text audio transcription.

Features:
- Audio file transcription
- Language detection
- Timestamp alignment
- Speaker diarization (basic)
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Optional

from hermes.tools.registry import ToolRegistry


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def hermes_transcribe(
    audio_path: str,
    language: str = "",
) -> str:
    """Transcribe an audio file to text.

    param audio_path (str): — Path to the audio file (wav, mp3, flac, ogg).
    param language (str): — Language code (e.g., en-US). Empty = auto-detect.
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        return f"Error: File not found: {path}"

    try:
        import speech_recognition as sr
    except ImportError:
        return "Error: SpeechRecognition is required. Install with: pip install SpeechRecognition"

    def _do():
        recognizer = sr.Recognizer()

        audio_file = sr.AudioFile(str(path))
        with audio_file as source:
            audio_data = recognizer.record(source)

        # Try Google's free speech recognition
        try:
            if language:
                text = recognizer.recognize_google(audio_data, language=language)
            else:
                text = recognizer.recognize_google(audio_data)

            return f"Transcription of {path.name}:\n\n{text}"

        except sr.UnknownValueError:
            return "Error: Could not understand the audio content."
        except sr.RequestError as e:
            return f"Error: Speech recognition service error: {e}"
        except Exception as e:
            return f"Error during transcription: {e}"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error transcribing {audio_path}: {e}"


async def hermes_transcribe_with_timestamps(
    audio_path: str,
    language: str = "",
    segment_duration: int = 10,
) -> str:
    """Transcribe audio with timestamp segments.

    param audio_path (str): — Path to the audio file.
    param language (str): — Language code. Empty = auto-detect.
    param segment_duration (int): — Segment duration in seconds. Default: 10.
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        return f"Error: File not found: {path}"

    try:
        import speech_recognition as sr
    except ImportError:
        return "Error: SpeechRecognition is required."

    def _do():
        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile(str(path))

        segments = []
        offset = 0

        with audio_file as source:
            while True:
                try:
                    segment = recognizer.record(source, duration=segment_duration)
                except Exception:
                    break

                try:
                    if language:
                        text = recognizer.recognize_google(segment, language=language)
                    else:
                        text = recognizer.recognize_google(segment)

                    if text.strip():
                        end_time = offset + segment_duration
                        segments.append(f"[{offset:02d}:{end_time:02d}] {text}")
                except (sr.UnknownValueError, sr.RequestError):
                    pass

                offset += segment_duration

        if not segments:
            return "No speech detected in the audio file."

        return f"Timestamped transcription of {path.name}:\n\n" + "\n".join(segments)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error: {e}"


async def hermes_detect_language(audio_path: str) -> str:
    """Detect the language spoken in an audio file.

    param audio_path (str): — Path to the audio file.
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        return f"Error: File not found: {path}"

    try:
        import speech_recognition as sr
    except ImportError:
        return "Error: SpeechRecognition is required."

    def _do():
        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile(str(path))

        with audio_file as source:
            audio_data = recognizer.record(source, duration=15)

        # Try multiple languages and compare confidence
        languages = {
            "en-US": "English (US)",
            "en-GB": "English (UK)",
            "es-ES": "Spanish",
            "fr-FR": "French",
            "de-DE": "German",
            "it-IT": "Italian",
            "pt-BR": "Portuguese (BR)",
            "ja-JP": "Japanese",
            "zh-CN": "Chinese (Simplified)",
            "ko-KR": "Korean",
            "ru-RU": "Russian",
            "ar-SA": "Arabic",
        }

        results = []
        for lang_code, lang_name in languages.items():
            try:
                text = recognizer.recognize_google(audio_data, language=lang_code)
                if text and len(text) > 5:
                    results.append((lang_code, lang_name, len(text), text[:50]))
            except (sr.UnknownValueError, sr.RequestError):
                continue

        if not results:
            return "Could not detect language from the audio."

        # Sort by text length (longer = more likely correct)
        results.sort(key=lambda x: x[2], reverse=True)

        lines = ["Detected language candidates:\n"]
        for code, name, length, sample in results[:5]:
            lines.append(f"  {name} ({code}): {length} chars — \"{sample}...\"")

        return "\n".join(lines)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error detecting language: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="hermes_transcribe",
    func=hermes_transcribe,
    description="Transcribe an audio file to text using speech recognition.",
    toolset="audio",
)

ToolRegistry.instance().register(
    name="hermes_transcribe_with_timestamps",
    func=hermes_transcribe_with_timestamps,
    description="Transcribe audio with per-segment timestamps.",
    toolset="audio",
)

ToolRegistry.instance().register(
    name="hermes_detect_language",
    func=hermes_detect_language,
    description="Detect the language spoken in an audio file.",
    toolset="audio",
)
