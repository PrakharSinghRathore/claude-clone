"""
Voice Interaction Module for AI Desktop Assistant.

Provides speech-to-text (STT) and text-to-speech (TTS) capabilities with
multiple backend support, wake word detection, conversation mode, audio
recording/playback, sound effects, and voice commands.

Dependencies (all optional with graceful fallbacks):
    - speech_recognition  (Google, Whisper, browser STT)
    - pyttsx3            (offline TTS)
    - gTTS               (Google TTS)
    - pyaudio            (microphone capture & playback)
    - numpy              (audio preprocessing)
    - noisereduce        (noise reduction)
    - soundfile          (audio file I/O)
    - webrtcvad           (voice activity detection)
"""

from __future__ import annotations

import asyncio
import ctypes
import io
import logging
import math
import os
import platform
import re
import struct
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency checks – we import lazily and surface clear messages
# ---------------------------------------------------------------------------

def _try_import(name: str) -> Tuple[Optional[Any], Optional[str]]:
    """Try to import *name*; return (module | None, error_message | None)."""
    try:
        mod = __import__(name)
        for part in name.split(".")[1:]:
            mod = getattr(mod, part)
        return mod, None
    except Exception as exc:
        return None, str(exc)


_speech_recognition, _sr_err = _try_import("speech_recognition")
_pyaudio, _pa_err = _try_import("pyaudio")
_pyttsx3, _pt_err = _try_import("pyttsx3")
_gtts, _gt_err = _try_import("gtts")
_numpy, _np_err = _try_import("numpy")
_noisereduce, _nr_err = _try_import("noisereduce")
_soundfile, _sf_err = _try_import("soundfile")
_webrtcvad, _vad_err = _try_import("webrtcvad")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class VoiceEventType(Enum):
    """Events emitted by the voice engine."""
    WAKE_WORD_DETECTED = "wake_word_detected"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    TEXT_RECOGNIZED = "text_recognized"
    SPEECH_STARTED_TTS = "speech_started_tts"
    SPEECH_ENDED_TTS = "speech_ended_tts"
    ERROR = "error"
    RECORDING_STARTED = "recording_started"
    RECORDING_ENDED = "recording_ended"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class STTConfig:
    """Configuration for speech-to-text."""
    engine: str = "google"                       # google | whisper | browser
    language: str = "en-US"                      # BCP-47 language tag
    wake_word: str = "hey claude"                # phrase that activates the assistant
    wake_word_sensitivity: float = 0.8           # 0.0 (always match) – 1.0 (exact only)
    continuous: bool = True                      # keep listening after first result
    timeout: int = 10                            # seconds of silence before giving up
    noise_gate: float = 0.02                     # RMS threshold to ignore silence


@dataclass
class TTSConfig:
    """Configuration for text-to-speech."""
    engine: str = "pyttsx3"                      # pyttsx3 | gtts | system
    voice_id: str = ""                           # specific voice identifier
    rate: float = 1.0                            # speech rate multiplier (0.5 – 2.0)
    volume: float = 1.0                          # volume level 0.0 – 1.0
    pitch: float = 1.0                           # pitch multiplier (engine-dependent)
    queue_enabled: bool = True                   # queue utterances instead of dropping


@dataclass
class VoiceEvent:
    """An event dispatched by the voice engine."""
    event_type: VoiceEventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptEntry:
    """A single recognised speech segment."""
    text: str = ""
    confidence: float = 0.0
    language: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_wake_word: bool = False
    audio_duration: float = 0.0


# ---------------------------------------------------------------------------
# Helper – simple SSML strip / parse
# ---------------------------------------------------------------------------

_SSML_TAG_RE = re.compile(r"<(/?\w+)([^>]*)>([^<]*)</\1>", re.DOTALL)
_SSML_BREAK_RE = re.compile(r"<break\s+([^/]+)/?>")
_SSML_EMPHASIS_MAP = {"strong": "!!!", "moderate": "!!", "reduced": "!"}


def _process_ssml(text: str) -> Tuple[str, Dict[str, Any]]:
    """Strip SSML tags and extract prosody hints.

    Returns cleaned plain-text and a dict of settings derived from SSML.
    """
    settings: Dict[str, Any] = {}

    # Extract <prosody> attributes (rate, pitch, volume)
    for match in _SSML_TAG_RE.finditer(text):
        tag = match.group(1)
        attrs = match.group(2)
        content = match.group(3)
        if tag == "prosody":
            for attr in ("rate", "pitch", "volume"):
                value = re.search(rf'{attr}=["\']([^"\']+)["\']', attrs)
                if value:
                    settings[f"ssml_{attr}"] = value.group(1)

    # Convert <emphasis> to punctuation cues
    def _emphasis_repl(m: re.Match) -> str:
        level = re.search(r'level=["\']([^"\']+)["\']', m.group(2))
        lvl = level.group(1) if level else "moderate"
        return _SSML_EMPHASIS_MAP.get(lvl, content) + " " + m.group(3)

    text = _SSML_TAG_RE.sub(lambda m: m.group(3) if m.group(1) in ("speak", "voice") else _emphasis_repl(m), text)

    # Convert <break> to pauses (commas / ellipsis)
    def _break_repl(m: re.Match) -> str:
        attrs = m.group(1)
        time_m = re.search(r'time=["\'](\d+)ms["\']', attrs)
        ms = int(time_m.group(1)) if time_m else 500
        if ms >= 1000:
            return ". "
        if ms >= 500:
            return ", "
        return " "

    text = _SSML_BREAK_RE.sub(_break_repl, text)
    text = re.sub(r"<[^>]+>", "", text)  # strip remaining tags
    return text.strip(), settings


# ---------------------------------------------------------------------------
# Utility – generate simple notification WAV data
# ---------------------------------------------------------------------------

def _generate_notification_wav(
    frequency: int = 880,
    duration_ms: int = 200,
    sample_rate: int = 22050,
    volume: float = 0.5,
) -> bytes:
    """Synthesise a short sine-wave beep as a WAV byte-string."""
    n_samples = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            envelope = min(1.0, min(i, n_samples - i) / (sample_rate * 0.01))
            value = int(32767 * volume * envelope * math.sin(2 * math.pi * frequency * t))
            wf.writeframes(struct.pack("<h", value))
    return buf.getvalue()


_NOTIFICATION_SOUNDS: Dict[str, bytes] = {
    "default": _generate_notification_wav(880, 200),
    "success": _generate_notification_wav(1046, 150),
    "error": _generate_notification_wav(330, 400),
    "alert": _generate_notification_wav(1200, 100),
    "beep": _generate_notification_wav(1000, 80),
    "ping": _generate_notification_wav(1400, 60),
}


# ---------------------------------------------------------------------------
# Voice Commands
# ---------------------------------------------------------------------------

_VOICE_COMMANDS: Dict[str, Callable] = {}  # populated by VoiceEngine


def _register_voice_command(keyword: str):
    """Decorator that registers a method as a voice command handler."""
    def decorator(fn):
        _VOICE_COMMANDS[keyword] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# VoiceEngine
# ---------------------------------------------------------------------------

class VoiceEngine:
    """Unified engine for speech-to-text, text-to-speech, and voice interaction.

    Supports multiple STT/TTS backends with graceful fallback when optional
    dependencies are missing.
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        stt_config: STTConfig | None = None,
        tts_config: TTSConfig | None = None,
    ) -> None:
        self._stt_config = stt_config or STTConfig()
        self._tts_config = tts_config or TTSConfig()

        # Internal state
        self._initialized = False
        self._shutting_down = False
        self._listening = False
        self._speaking = False
        self._conversation_active = False
        self._recording = False

        # Backend instances (created during initialize)
        self._recognizer: Any | None = None
        self._microphone: Any | None = None
        self._pyaudio_instance: Any | None = None
        self._tts_engine: Any | None = None
        self._tts_lock = asyncio.Lock()
        self._speech_queue: asyncio.Queue[str] = asyncio.Queue()
        self._current_speech_task: asyncio.Task | None = None
        self._stop_speaking_event = asyncio.Event()

        # VAD
        self._vad: Any | None = None

        # Event handlers  {VoiceEventType: [callable, ...]}
        self._event_handlers: Dict[VoiceEventType, List[Callable]] = {
            et: [] for et in VoiceEventType
        }

        # Recording state
        self._recorded_frames: List[bytes] = []
        self._record_stream: Any | None = None

        # Threading primitives for pyaudio callbacks (runs in non-async context)
        self._listen_result: Optional[TranscriptEntry] = None
        self._listen_error: Optional[Exception] = None
        self._listen_event = threading.Event()

        # Register built-in voice commands
        self._register_builtin_commands()

    # --------------------------------------------------------- voice commands

    def _register_builtin_commands(self) -> None:
        """Set up built-in voice command handlers."""
        self._command_handlers: Dict[str, Callable[[], Awaitable[None]]] = {
            "stop": self._cmd_stop,
            "louder": self._cmd_louder,
            "quieter": self._cmd_quieter,
            "faster": self._cmd_faster,
            "slower": self._cmd_slower,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "mute": self._cmd_mute,
            "unmute": self._cmd_unmute,
        }

    async def _cmd_stop(self) -> None:
        await self.stop_speaking()

    async def _cmd_louder(self) -> None:
        new_vol = min(1.0, self._tts_config.volume + 0.15)
        await self.set_volume(new_vol)

    async def _cmd_quieter(self) -> None:
        new_vol = max(0.0, self._tts_config.volume - 0.15)
        await self.set_volume(new_vol)

    async def _cmd_faster(self) -> None:
        new_rate = min(2.0, self._tts_config.rate + 0.15)
        await self.set_rate(new_rate)

    async def _cmd_slower(self) -> None:
        new_rate = max(0.5, self._tts_config.rate - 0.15)
        await self.set_rate(new_rate)

    async def _cmd_pause(self) -> None:
        await self.stop_speaking()

    async def _cmd_resume(self) -> None:
        self._stop_speaking_event.clear()

    async def _cmd_mute(self) -> None:
        self._muted = True

    async def _cmd_unmute(self) -> None:
        self._muted = False

    def _try_handle_voice_command(self, text: str) -> bool:
        """Check *text* for a built-in voice command and execute it.

        Returns True if a command was recognised and dispatched.
        """
        lower = text.strip().lower()
        for keyword, handler in self._command_handlers.items():
            if lower == keyword or lower.startswith(keyword + " "):
                # Schedule in event loop (called potentially from a thread)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(handler(), loop=loop)
                    else:
                        loop.run_until_complete(handler())
                except RuntimeError:
                    pass
                return True
        return False

    # -------------------------------------------------------- event helpers

    def on_event(
        self, event_type: VoiceEventType, handler: Callable[[VoiceEvent], Any]
    ) -> None:
        """Register a callback for a specific event type."""
        self._event_handlers[event_type].append(handler)

    def _emit(self, event_type: VoiceEventType, **data: Any) -> None:
        event = VoiceEvent(event_type=event_type, data=data)
        for handler in self._event_handlers.get(event_type, []):
            try:
                handler(event)
            except Exception:
                logger.exception("Event handler error for %s", event_type)

    # --------------------------------------------------- lifecycle: init/shut

    async def initialize(self) -> None:
        """Set up all backends and resources."""
        if self._initialized:
            return

        logger.info("Initializing VoiceEngine …")

        # STT – speech_recognition
        if _speech_recognition is not None:
            self._recognizer = _speech_recognition.Recognizer()
            self._recognizer.energy_threshold = self._stt_config.noise_gate * 32768
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold = 0.6
            self._recognizer.phrase_threshold = 0.3
            self._recognizer.non_speaking_duration = 0.4
            logger.info("speech_recognition backend ready.")
        else:
            logger.warning(
                "speech_recognition not available (%s). STT disabled.", _sr_err
            )

        # Microphone
        if _speech_recognition is not None:
            try:
                self._microphone = _speech_recognition.Microphone()
                # Calibrate ambient noise in a thread to avoid blocking
                with self._microphone as src:
                    self._recognizer.adjust_for_ambient_noise(src, duration=0.5)
                logger.info("Microphone ready.")
            except (OSError, AttributeError, Exception) as exc:
                logger.warning("Microphone init failed: %s", exc)
                self._microphone = None

        # PyAudio instance (for raw audio record/play)
        if _pyaudio is not None:
            try:
                self._pyaudio_instance = _pyaudio.PyAudio()
                logger.info("PyAudio instance ready.")
            except Exception as exc:
                logger.warning("PyAudio init failed: %s", exc)
                self._pyaudio_instance = None

        # VAD (WebRTC)
        if _webrtcvad is not None:
            try:
                self._vad = _webrtcvad.Vad(2)  # mode 2 – balanced aggressiveness
                logger.info("WebRTC VAD ready.")
            except Exception as exc:
                logger.warning("VAD init failed: %s", exc)
                self._vad = None

        # TTS – pyttsx3
        if self._tts_config.engine == "pyttsx3":
            if _pyttsx3 is not None:
                try:
                    self._tts_engine = _pyttsx3.init()
                    self._apply_tts_settings()
                    logger.info("pyttsx3 TTS ready.")
                except Exception as exc:
                    logger.warning("pyttsx3 init failed: %s", exc)
                    self._tts_engine = None
                    self._tts_config.engine = "system"  # fallback
            else:
                logger.warning("pyttsx3 not available (%s).", _pt_err)
                self._tts_config.engine = "system"

        self._muted = False
        self._initialized = True
        logger.info("VoiceEngine initialized (STT=%s, TTS=%s).",
                     self._stt_config.engine, self._tts_config.engine)

    async def shutdown(self) -> None:
        """Release all resources."""
        self._shutting_down = True
        self._conversation_active = False

        await self.stop_speaking()
        self._listening = False

        if self._tts_engine is not None:
            try:
                self._tts_engine.stop()
            except Exception:
                pass
            self._tts_engine = None

        if self._pyaudio_instance is not None:
            try:
                self._pyaudio_instance.terminate()
            except Exception:
                pass
            self._pyaudio_instance = None

        self._initialized = False
        logger.info("VoiceEngine shut down.")

    # ---------------------------------------------------------- properties

    async def is_speaking(self) -> bool:
        return self._speaking

    async def is_listening(self) -> bool:
        return self._listening

    # ------------------------------------------------ STT: listen_once

    async def listen_once(self, timeout: int = 10) -> TranscriptEntry:
        """Listen for a single utterance and return a TranscriptEntry.

        If STT backend is not available returns an empty entry with an
        error description in *text*.
        """
        if self._recognizer is None or self._microphone is None:
            msg = "STT not available: install speech_recognition and pyaudio"
            logger.error(msg)
            self._emit(VoiceEventType.ERROR, message=msg)
            return TranscriptEntry(text=msg)

        self._listening = True
        self._emit(VoiceEventType.SPEECH_STARTED)
        self._listen_event.clear()
        self._listen_result = None
        self._listen_error = None

        loop = asyncio.get_event_loop()

        def _worker() -> None:
            try:
                with self._microphone as source:
                    audio = self._recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=None,
                    )

                # Preprocess audio
                raw = audio.get_raw_data()
                preprocessed = self._preprocess_audio(raw)
                # Reconstruct AudioData after preprocessing
                processed_audio = _speech_recognition.AudioData(
                    preprocessed,
                    audio.sample_rate,
                    audio.sample_width,
                )

                entry = self._recognize_audio(processed_audio)
                entry.audio_duration = len(preprocessed) / (
                    audio.sample_rate * audio.sample_width
                )
                self._listen_result = entry
            except _speech_recognition.WaitTimeoutError:
                self._listen_result = TranscriptEntry(text="", confidence=0.0)
            except _speech_recognition.UnknownValueError:
                self._listen_result = TranscriptEntry(text="", confidence=0.0)
            except Exception as exc:
                self._listen_error = exc
            finally:
                self._listen_event.set()

        try:
            await loop.run_in_executor(None, _worker)
        except asyncio.CancelledError:
            self._listening = False
            raise

        self._listening = False
        self._emit(VoiceEventType.SPEECH_ENDED)

        if self._listen_error is not None:
            err_msg = f"STT error: {self._listen_error}"
            logger.error(err_msg)
            self._emit(VoiceEventType.ERROR, message=err_msg)
            return TranscriptEntry(text=err_msg)

        entry = self._listen_result or TranscriptEntry()
        if entry.text:
            entry.is_wake_word = self._detect_wake_word(entry.text)
            self._emit(VoiceEventType.TEXT_RECOGNIZED, text=entry.text, confidence=entry.confidence)
        return entry

    # ----------------------------------------- STT: listen_continuous (async gen)

    async def listen_continuous(self) -> AsyncGenerator[TranscriptEntry, None]:
        """Continuously listen and yield TranscriptEntry objects.

        If ``continuous`` is True the generator runs until cancelled or
        shutdown.  When a wake word is configured the generator only yields
        entries that follow a wake word (or the wake word entry itself).
        """
        if self._recognizer is None or self._microphone is None:
            msg = "STT not available: install speech_recognition and pyaudio"
            logger.error(msg)
            self._emit(VoiceEventType.ERROR, message=msg)
            yield TranscriptEntry(text=msg)
            return

        self._listening = True
        await self._init_microphone_stream()

        try:
            while not self._shutting_down and self._conversation_active:
                entry = await self.listen_once(timeout=self._stt_config.timeout)
                if self._shutting_down:
                    break

                if not entry.text:
                    continue

                # If wake-word mode, only forward entries that contain it
                if self._stt_config.wake_word and self._stt_config.continuous:
                    if entry.is_wake_word:
                        self._emit(VoiceEventType.WAKE_WORD_DETECTED, text=entry.text)
                        await self.play_notification("ping")
                        # Strip wake word from text for downstream processing
                        stripped = self._strip_wake_word(entry.text)
                        if stripped:
                            entry.text = stripped
                            yield entry
                        # After wake word, capture one more follow-up phrase
                        follow = await self.listen_once(timeout=self._stt_config.timeout)
                        if follow.text and not self._shutting_down:
                            yield follow
                    # Voice command shortcut (no wake word needed)
                    elif self._try_handle_voice_command(entry.text):
                        continue
                else:
                    if self._try_handle_voice_command(entry.text):
                        continue
                    yield entry

        except asyncio.CancelledError:
            pass
        finally:
            self._listening = False
            self._close_microphone_stream()

    # ------------------------------------------- STT: recognition backends

    def _recognize_audio(self, audio: Any) -> TranscriptEntry:
        """Run recognition using the configured STT backend."""
        engine = self._stt_config.engine
        lang = self._stt_config.language

        if engine == "google":
            return self._recognize_google(audio, lang)
        elif engine == "whisper":
            return self._recognize_whisper(audio, lang)
        elif engine == "browser":
            return self._recognize_browser(audio)
        else:
            return self._recognize_google(audio, lang)

    def _recognize_google(self, audio: Any, language: str) -> TranscriptEntry:
        try:
            result = self._recognizer.recognize_google(audio, language=language)
            return TranscriptEntry(
                text=result,
                confidence=0.9,
                language=language,
            )
        except _speech_recognition.UnknownValueError:
            return TranscriptEntry(text="")
        except _speech_recognition.RequestError as exc:
            logger.warning("Google STT request error: %s", exc)
            return TranscriptEntry(text="")

    def _recognize_whisper(self, audio: Any, language: str) -> TranscriptEntry:
        try:
            result = self._recognizer.recognize_whisper(
                audio, model="base", language=language
            )
            return TranscriptEntry(
                text=result,
                confidence=0.85,
                language=language,
            )
        except _speech_recognition.UnknownValueError:
            return TranscriptEntry(text="")
        except _speech_recognition.RequestError as exc:
            logger.warning("Whisper STT error: %s", exc)
            return TranscriptEntry(text="")

    def _recognize_browser(self, audio: Any) -> TranscriptEntry:
        """Use the browser-based (Vosk/Nemo cloud) recognizer."""
        try:
            result = self._recognizer.recognize_vosk(audio)
            # vosk returns JSON
            import json
            parsed = json.loads(result)
            text = parsed.get("text", "")
            conf = parsed.get("confidence", 0.8)
            return TranscriptEntry(text=text, confidence=conf)
        except Exception as exc:
            logger.warning("Browser STT error: %s", exc)
            return TranscriptEntry(text="")

    # ------------------------------------------------- wake word helpers

    def _detect_wake_word(self, text: str) -> bool:
        """Return True if *text* contains the configured wake word."""
        if not self._stt_config.wake_word:
            return True  # no wake word configured → everything activates

        lower = text.strip().lower()
        wake = self._stt_config.wake_word.lower()

        # Exact match
        if lower == wake:
            return True

        # Starts with wake word
        if lower.startswith(wake):
            return True

        # Fuzzy: check if all wake-word tokens appear near the start
        wake_words = wake.split()
        text_words = lower.split()[: len(wake_words) + 2]
        wake_set = set(wake_words)
        text_set = set(text_words)
        overlap = wake_set & text_set
        if len(overlap) / len(wake_set) >= self._stt_config.wake_word_sensitivity:
            return True

        return False

    def _strip_wake_word(self, text: str) -> str:
        """Remove the wake word phrase from the beginning of *text*."""
        if not self._stt_config.wake_word:
            return text
        lower = text.strip().lower()
        wake = self._stt_config.wake_word.lower()
        if lower.startswith(wake):
            rest = text.strip()[len(wake):].strip()
            # Remove leading punctuation/comma
            return rest.lstrip(",:;. ").strip()
        return text

    # ----------------------------------------- audio preprocessing / noise

    def _preprocess_audio(self, audio_data: bytes) -> bytes:
        """Apply noise reduction and normalisation to raw PCM audio data.

        Falls back to returning the original data when numpy/noisereduce are
        not installed.
        """
        if _numpy is None or _noisereduce is None:
            return audio_data

        try:
            samples = _numpy.frombuffer(audio_data, dtype=_numpy.int16).astype(
                _numpy.float32
            )
            if len(samples) == 0:
                return audio_data
            cleaned = _noisereduce.reduce_noise(y=samples, sr=16000, prop_decrease=0.8)
            # Normalise to int16 range
            peak = _numpy.max(_numpy.abs(cleaned))
            if peak > 0:
                cleaned = cleaned / peak * 32767
            return cleaned.astype(_numpy.int16).tobytes()
        except Exception as exc:
            logger.debug("Audio preprocessing error: %s", exc)
            return audio_data

    def _vad_is_speech(self, frame: bytes, sample_rate: int = 16000) -> bool:
        """Return True if *frame* contains speech (VAD check)."""
        if self._vad is None:
            return True  # no VAD → assume speech
        try:
            return self._vad.is_speech(frame, sample_rate)
        except Exception:
            return True

    # ------------------------------------------ microphone stream helpers

    def _init_microphone_stream(self) -> None:
        """Open a raw PyAudio stream for continuous reading (used by
        ``listen_continuous`` for fine-grained VAD gating)."""
        # This is optional – the recognizer.listen() call handles its own
        # stream.  We keep this available for future VAD-driven listening.
        pass

    def _close_microphone_stream(self) -> None:
        pass

    # ============================================================ TTS

    def _apply_tts_settings(self) -> None:
        """Push current TTSConfig values into the pyttsx3 engine."""
        if self._tts_engine is None:
            return
        try:
            self._tts_engine.setProperty("rate", int(200 * self._tts_config.rate))
            self._tts_engine.setProperty("volume", self._tts_config.volume)
        except Exception as exc:
            logger.warning("Failed to apply TTS settings: %s", exc)

    async def speak(self, text: str, interrupt: bool = False) -> None:
        """Speak *text* aloud using the configured TTS backend.

        If *interrupt* is True the current utterance (if any) is stopped
        before the new one begins.
        """
        if not text or not text.strip():
            return

        if self._muted:
            return

        if interrupt:
            await self.stop_speaking()

        if self._tts_config.queue_enabled and self._speaking:
            self._speech_queue.put_nowait(text)
            return

        async with self._tts_lock:
            self._speaking = True
            self._stop_speaking_event.clear()
            self._emit(VoiceEventType.SPEECH_STARTED_TTS, text=text)

            # Process SSML
            plain_text, ssml_settings = _process_ssml(text)

            # Temporarily override settings from SSML if provided
            saved_rate = self._tts_config.rate
            saved_volume = self._tts_config.volume
            if "ssml_rate" in ssml_settings:
                self._tts_config.rate = float(ssml_settings["ssml_rate"].replace("x", ""))
            if "ssml_volume" in ssml_settings:
                self._tts_config.volume = float(ssml_settings["ssml_volume"].replace("%", "")) / 100

            try:
                engine = self._tts_config.engine
                if engine == "pyttsx3":
                    await self._speak_pyttsx3(plain_text)
                elif engine == "gtts":
                    await self._speak_gtts(plain_text)
                elif engine == "system":
                    await self._speak_system(plain_text)
                else:
                    await self._speak_pyttsx3(plain_text)
            finally:
                # Restore settings
                self._tts_config.rate = saved_rate
                self._tts_config.volume = saved_volume
                self._speaking = False
                self._emit(VoiceEventType.SPEECH_ENDED_TTS)

    async def _speak_pyttsx3(self, text: str) -> None:
        """Speak using pyttsx3 (runs in executor)."""
        if self._tts_engine is None:
            await self._speak_system(text)  # fallback
            return

        def _run() -> None:
            self._apply_tts_settings()
            self._tts_engine.say(text)
            self._tts_engine.runAndWait()

        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, _run)

        # Allow interruption via stop event
        done, _ = await asyncio.wait(
            {asyncio.ensure_future(task), asyncio.ensure_future(self._wait_stop_event())},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # If stop event fired, try to stop engine
        if self._stop_speaking_event.is_set():
            try:
                self._tts_engine.stop()
            except Exception:
                pass

    async def _speak_gtts(self, text: str) -> None:
        """Speak using Google TTS (gTTS).  Plays via pyaudio or system player."""
        if _gtts is None:
            await self._speak_system(text)
            return

        def _generate() -> bytes:
            buf = io.BytesIO()
            tts = _gtts.gTTS(text=text, lang=self._stt_config.language[:2])
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()

        try:
            mp3_data = await asyncio.get_event_loop().run_in_executor(None, _generate)
            await self._play_bytes(mp3_data, format_hint="mp3")
        except Exception as exc:
            logger.warning("gTTS failed: %s – falling back to system TTS", exc)
            await self._speak_system(text)

    async def _speak_system(self, text: str) -> None:
        """Speak using the OS's built-in speech synthesiser."""
        safe_text = text.replace('"', '\\"')
        system = platform.system()

        if system == "Darwin":
            cmd = ["say", "-r", str(int(200 * self._tts_config.rate)), "-v", "Alex", safe_text]
        elif system == "Windows":
            # Use PowerShell SAPI
            escaped = safe_text.replace("'", "''")
            ps_script = (
                f"Add-Type -AssemblyName System.Speech; "
                f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = {int((self._tts_config.rate - 1) * 10)}; "
                f"$s.Volume = {int(self._tts_config.volume * 100)}; "
                f"$s.Speak('{escaped}')"
            )
            cmd = ["powershell", "-Command", ps_script]
        else:
            # Linux: try espeak, then festival
            cmd = [
                "espeak",
                "-s", str(int(175 * self._tts_config.rate)),
                "-a", str(int(self._tts_config.volume * 100)),
                "-v", self._stt_config.language[:2],
                safe_text,
            ]

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: subprocess.run(cmd, check=True, timeout=60, capture_output=True)
            )
        except FileNotFoundError:
            logger.warning("System TTS command not found (%s).", cmd[0])
        except subprocess.TimeoutExpired:
            logger.warning("System TTS timed out.")
        except Exception as exc:
            logger.warning("System TTS error: %s", exc)

    # -------------------------------------------------- TTS helpers

    async def _wait_stop_event(self) -> None:
        """Future that resolves when ``_stop_speaking_event`` is set."""
        while not self._stop_speaking_event.is_set():
            await asyncio.sleep(0.05)

    async def _play_bytes(self, data: bytes, format_hint: str = "wav") -> None:
        """Play raw audio bytes (WAV/MP3) through pyaudio or system player."""
        if _pyaudio is not None and self._pyaudio_instance is not None and format_hint == "wav":
            try:
                await self._play_wav_pyaudio(data)
                return
            except Exception as exc:
                logger.debug("pyaudio playback failed: %s", exc)

        # Fallback: write to temp file and use system player
        with tempfile.NamedTemporaryFile(suffix=f".{format_hint}", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            system = platform.system()
            if system == "Darwin":
                cmd = ["afplay", tmp_path]
            elif system == "Windows":
                cmd = ["powershell", "-Command", f"(New-Object Media.SoundPlayer '{tmp_path}').PlaySync()"]
            else:
                cmd = ["aplay", "-q", tmp_path]
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: subprocess.run(cmd, check=True, timeout=60, capture_output=True)
            )
        except Exception as exc:
            logger.warning("System playback failed: %s", exc)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def _play_wav_pyaudio(self, wav_data: bytes) -> None:
        """Play WAV bytes via PyAudio with stop-event support."""
        wf = wave.open(io.BytesIO(wav_data), "rb")
        pa = self._pyaudio_instance
        stream = pa.open(
            format=pa.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        chunk_size = 1024
        loop = asyncio.get_event_loop()

        def _read_chunk() -> bytes | None:
            return wf.readframes(chunk_size)

        try:
            while True:
                if self._stop_speaking_event.is_set():
                    break
                chunk = await loop.run_in_executor(None, _read_chunk)
                if not chunk:
                    break
                await loop.run_in_executor(None, lambda c=chunk: stream.write(c))
        finally:
            stream.stop_stream()
            stream.close()
            wf.close()

    async def speak_many(self, texts: list[str]) -> None:
        """Queue multiple texts to be spoken in order."""
        if not texts:
            return
        if not self._tts_config.queue_enabled:
            for t in texts:
                await self.speak(t)
            return

        for t in texts:
            self._speech_queue.put_nowait(t)
        # Ensure the queue consumer is running
        if self._current_speech_task is None or self._current_speech_task.done():
            self._current_speech_task = asyncio.ensure_future(self._speech_queue_consumer())

    async def _speech_queue_consumer(self) -> None:
        """Drain the speech queue."""
        while not self._speech_queue.empty():
            if self._shutting_down:
                return
            text = await self._speech_queue.get()
            await self.speak(text)

    async def stop_speaking(self) -> None:
        """Interrupt the current speech."""
        self._stop_speaking_event.set()
        self._speech_queue.queue.clear()  # discard pending utterances
        if self._tts_engine is not None:
            try:
                self._tts_engine.stop()
            except Exception:
                pass
        self._speaking = False
        await asyncio.sleep(0.05)  # give time for engine to actually stop

    async def set_volume(self, volume: float) -> None:
        """Set TTS volume (0.0 – 1.0)."""
        self._tts_config.volume = max(0.0, min(1.0, volume))
        if self._tts_engine is not None:
            self._tts_engine.setProperty("volume", self._tts_config.volume)

    async def set_rate(self, rate: float) -> None:
        """Set TTS rate/speed (0.5 – 2.0)."""
        self._tts_config.rate = max(0.5, min(2.0, rate))
        if self._tts_engine is not None:
            self._tts_engine.setProperty("rate", int(200 * self._tts_config.rate))

    async def get_available_voices(self) -> list[dict]:
        """Return a list of available TTS voices."""
        voices: list[dict] = []

        if self._tts_engine is not None:
            try:
                for v in self._tts_engine.getProperty("voices"):
                    voices.append({
                        "id": v.id,
                        "name": v.name,
                        "languages": getattr(v, "languages", []),
                        "gender": getattr(v, "gender", "unknown"),
                        "backend": "pyttsx3",
                    })
            except Exception as exc:
                logger.warning("Could not list pyttsx3 voices: %s", exc)

        # System voices (platform-specific)
        system = platform.system()
        if system == "Darwin":
            try:
                result = subprocess.run(
                    ["say", "-v", "?"], capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.strip().splitlines():
                    parts = line.split(maxsplit=2)
                    if len(parts) >= 2:
                        voices.append({
                            "id": parts[0],
                            "name": parts[1] if len(parts) > 1 else parts[0],
                            "languages": [],
                            "gender": "unknown",
                            "backend": "system",
                        })
            except Exception:
                pass
        elif system == "Windows":
            try:
                ps = (
                    "Add-Type -AssemblyName System.Speech; "
                    "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices()"
                    " | ForEach-Object { $_.VoiceInfo.Name }"
                )
                result = subprocess.run(
                    ["powershell", "-Command", ps], capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.strip().splitlines():
                    if line.strip():
                        voices.append({
                            "id": line.strip(),
                            "name": line.strip(),
                            "languages": [],
                            "gender": "unknown",
                            "backend": "system",
                        })
            except Exception:
                pass
        elif system == "Linux":
            try:
                result = subprocess.run(
                    ["espeak", "--voices"], capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.strip().splitlines()[1:]:
                    parts = line.split()
                    if len(parts) >= 5:
                        voices.append({
                            "id": parts[4],
                            "name": " ".join(parts[3:]),
                            "languages": [parts[1]] if len(parts) > 1 else [],
                            "gender": "unknown",
                            "backend": "system",
                        })
            except Exception:
                pass

        return voices

    async def set_voice(self, voice_id: str) -> None:
        """Switch to a specific TTS voice."""
        if self._tts_engine is not None:
            try:
                self._tts_engine.setProperty("voice", voice_id)
                self._tts_config.voice_id = voice_id
                logger.info("Voice set to %s", voice_id)
            except Exception as exc:
                logger.warning("Failed to set voice: %s", exc)
        else:
            self._tts_config.voice_id = voice_id

    # ================================================ Conversation mode

    async def start_conversation_mode(self) -> AsyncGenerator[str, None]:
        """Continuous listen → respond loop.

        Yields recognised user text.  The caller is expected to *send* the
        AI response back into the generator (or simply call ``speak``).

        Usage::

            async for user_text in engine.start_conversation_mode():
                response = await llm.chat(user_text)
                await engine.speak(response)
        """
        if self._conversation_active:
            logger.warning("Conversation mode already active.")
            return

        self._conversation_active = True
        self._shutting_down = False

        await self.play_notification("ping")
        logger.info("Conversation mode started.")

        try:
            async for entry in self.listen_continuous():
                if not entry.text:
                    continue
                # Skip bare wake-word entries (the strip already handled them)
                if entry.is_wake_word and not entry.text.strip():
                    continue
                yield entry.text
                if not self._conversation_active:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self._conversation_active = False
            logger.info("Conversation mode ended.")

    # ================================================= Recording

    async def record_audio(self, duration: float) -> bytes:
        """Record *duration* seconds of microphone audio.

        Returns raw PCM 16-bit mono data at 16 kHz.
        Falls back to 44100 Hz mono 16-bit if PyAudio default is used.
        """
        if _pyaudio is None:
            logger.error("PyAudio not available for recording.")
            return b""

        pa = _pyaudio.PyAudio()

        sample_rate = 16000
        channels = 1
        sample_format = pa.get_format_from_width(2)  # 16-bit

        frames: list[bytes] = []
        frame_size = int(sample_rate / 1000 * 60)  # 60 ms frames
        n_frames = int(duration * 1000 / 60)

        self._recording = True
        self._emit(VoiceEventType.RECORDING_STARTED, duration=duration)

        def _record() -> None:
            try:
                stream = pa.open(
                    format=sample_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=frame_size,
                )
            except Exception as exc:
                logger.error("Could not open mic for recording: %s", exc)
                self._recording = False
                return

            try:
                for _ in range(n_frames):
                    if not self._recording:
                        break
                    data = stream.read(frame_size, exception_on_overflow=False)
                    frames.append(data)
            finally:
                stream.stop_stream()
                stream.close()

        try:
            await asyncio.get_event_loop().run_in_executor(None, _record)
        except asyncio.CancelledError:
            pass
        finally:
            pa.terminate()
            self._recording = False
            self._emit(VoiceEventType.RECORDING_ENDED)

        return b"".join(frames)

    async def play_audio(self, audio_data: bytes) -> None:
        """Play raw PCM 16-bit mono audio data at 16 kHz."""
        if not audio_data:
            return
        await self._play_pcm(audio_data, sample_rate=16000, channels=1, sample_width=2)

    async def _play_pcm(
        self,
        data: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> None:
        """Play raw PCM through pyaudio or system fallback."""
        if _pyaudio is not None:
            pa = _pyaudio.PyAudio()
            try:
                stream = pa.open(
                    format=pa.get_format_from_width(sample_width),
                    channels=channels,
                    rate=sample_rate,
                    output=True,
                )
                chunk_size = 1024
                for i in range(0, len(data), chunk_size):
                    if self._stop_speaking_event.is_set():
                        break
                    stream.write(data[i : i + chunk_size])
                stream.stop_stream()
                stream.close()
            except Exception as exc:
                logger.warning("PyAudio playback error: %s", exc)
            finally:
                pa.terminate()
        else:
            # Wrap in WAV and use system player
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(data)
            await self._play_bytes(buf.getvalue())

    # ========================================== Notification sounds

    async def play_notification(self, sound: str = "default") -> None:
        """Play a short notification sound.

        Built-in sounds: default, success, error, alert, beep, ping.
        Accepts a file path to a WAV file for custom sounds.
        """
        path = Path(sound)
        if path.is_file():
            try:
                with open(path, "rb") as f:
                    wav_data = f.read()
                await self._play_bytes(wav_data)
                return
            except Exception as exc:
                logger.warning("Could not play notification file: %s", exc)

        wav_data = _NOTIFICATION_SOUNDS.get(sound, _NOTIFICATION_SOUNDS["default"])
        if _pyaudio is not None:
            await self._play_bytes(wav_data)
        else:
            # Fallback: use system bell
            try:
                if platform.system() == "Windows":
                    import winsound  # type: ignore
                    winsound.Beep(880, 200)
                else:
                    print("\a", end="", flush=True)
            except Exception:
                pass

    # ========================================== Save / load recordings

    async def save_recording(self, audio_data: bytes, filepath: str | Path) -> None:
        """Save raw PCM audio data to a WAV file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)

        filepath.write_bytes(buf.getvalue())
        logger.info("Recording saved to %s", filepath)

    async def load_and_play(self, filepath: str | Path) -> None:
        """Load a WAV file and play it."""
        path = Path(filepath)
        if not path.is_file():
            logger.warning("File not found: %s", path)
            return
        data = path.read_bytes()
        await self._play_bytes(data)

    # ========================================== Utility

    def _sample_rate_from_microphone(self) -> int:
        """Return the sample rate of the default microphone via PyAudio."""
        if _pyaudio is None:
            return 16000
        try:
            pa = _pyaudio.PyAudio()
            info = pa.get_default_input_device_info()
            pa.terminate()
            return int(info.get("defaultSampleRate", 16000))
        except Exception:
            return 16000

    async def health_check(self) -> Dict[str, Any]:
        """Return a diagnostic dict indicating which backends are available."""
        return {
            "speech_recognition": _speech_recognition is not None,
            "pyaudio": _pyaudio is not None,
            "pyttsx3": _pyttsx3 is not None,
            "gtts": _gtts is not None,
            "numpy": _numpy is not None,
            "noisereduce": _noisereduce is not None,
            "soundfile": _soundfile is not None,
            "webrtcvad": _webrtcvad is not None,
            "stt_engine": self._stt_config.engine,
            "tts_engine": self._tts_config.engine,
            "microphone": self._microphone is not None,
            "initialized": self._initialized,
        }
