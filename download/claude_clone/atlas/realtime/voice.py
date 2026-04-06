"""
atlas.realtime.voice - Real-time voice conversation mode.

Implements a full-duplex voice conversation system with support
for multiple TTS/STT providers, echo cancellation, noise
suppression, and interrupt handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class VoiceProvider(Enum):
    """Supported voice service providers."""
    SYSTEM_DEFAULT = "system_default"
    WEBRTC = "webrtc"
    POLLY = "polly"
    GOOGLE_TTS = "google_tts"
    AZURE_TTS = "azure_tts"
    ELEVENLABS = "elevenlabs"
    OPENAI_TTS = "openai_tts"
    EDGE_TTS = "edge_tts"
    COQUI = "coqui"
    PIPER = "piper"


class AudioFormat(Enum):
    """Audio format specifications."""
    PCM_16KHZ_16BIT_MONO = "pcm_16khz_16bit_mono"
    PCM_8KHZ_16BIT_MONO = "pcm_8khz_16bit_mono"
    PCM_22KHZ_16BIT_MONO = "pcm_22khz_16bit_mono"
    PCM_44KHZ_16BIT_MONO = "pcm_44khz_16bit_mono"
    PCM_48KHZ_16BIT_MONO = "pcm_48khz_16bit_mono"
    MULAW_8KHZ = "mulaw_8khz"
    OPUS = "opus"
    AAC = "aac"
    MP3 = "mp3"
    WAV = "wav"

    @property
    def sample_rate(self) -> int:
        """Get the sample rate in Hz."""
        rates = {
            AudioFormat.PCM_8KHZ_16BIT_MONO: 8000,
            AudioFormat.PCM_16KHZ_16BIT_MONO: 16000,
            AudioFormat.PCM_22KHZ_16BIT_MONO: 22050,
            AudioFormat.PCM_44KHZ_16BIT_MONO: 44100,
            AudioFormat.PCM_48KHZ_16BIT_MONO: 48000,
            AudioFormat.MULAW_8KHZ: 8000,
            AudioFormat.OPUS: 48000,
            AudioFormat.AAC: 44100,
            AudioFormat.MP3: 44100,
            AudioFormat.WAV: 44100,
        }
        return rates.get(self, 16000)

    @property
    def bytes_per_sample(self) -> int:
        """Get the bytes per sample."""
        return 2  # 16-bit

    @property
    def is_compressed(self) -> bool:
        """Whether this format uses compression."""
        return self in (
            AudioFormat.OPUS, AudioFormat.AAC, AudioFormat.MP3
        )


@dataclass
class VoiceConfig:
    """Configuration for voice mode.

    Attributes:
        tts_provider: The TTS provider to use.
        stt_provider: The STT provider to use.
        input_format: Audio format for microphone input.
        output_format: Audio format for speaker output.
        language: Language code (e.g. 'en-US').
        voice_id: Voice identifier for TTS.
        sample_rate: Audio sample rate in Hz.
        echo_cancellation: Enable echo cancellation.
        noise_suppression: Enable noise suppression.
        auto_gain: Enable automatic gain control.
        vad_enabled: Enable voice activity detection.
        vad_threshold: VAD sensitivity threshold (0.0-1.0).
        interrupt_enabled: Enable interrupt handling.
        max_interruptions: Max consecutive interrupts.
        buffer_size: Audio buffer size in milliseconds.
    """
    tts_provider: VoiceProvider = VoiceProvider.SYSTEM_DEFAULT
    stt_provider: VoiceProvider = VoiceProvider.SYSTEM_DEFAULT
    input_format: AudioFormat = AudioFormat.PCM_16KHZ_16BIT_MONO
    output_format: AudioFormat = AudioFormat.PCM_16KHZ_16BIT_MONO
    language: str = "en-US"
    voice_id: str = "default"
    sample_rate: int = 16000
    echo_cancellation: bool = True
    noise_suppression: bool = True
    auto_gain: bool = True
    vad_enabled: bool = True
    vad_threshold: float = 0.3
    interrupt_enabled: bool = True
    max_interruptions: int = 5
    buffer_size: int = 100  # milliseconds

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VoiceConfig:
        """Create a VoiceConfig from a dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        # Handle enum conversions
        if "tts_provider" in filtered:
            filtered["tts_provider"] = VoiceProvider(filtered["tts_provider"])
        if "stt_provider" in filtered:
            filtered["stt_provider"] = VoiceProvider(filtered["stt_provider"])
        if "input_format" in filtered:
            filtered["input_format"] = AudioFormat(filtered["input_format"])
        if "output_format" in filtered:
            filtered["output_format"] = AudioFormat(filtered["output_format"])

        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tts_provider": self.tts_provider.value,
            "stt_provider": self.stt_provider.value,
            "input_format": self.input_format.value,
            "output_format": self.output_format.value,
            "language": self.language,
            "voice_id": self.voice_id,
            "sample_rate": self.sample_rate,
            "echo_cancellation": self.echo_cancellation,
            "noise_suppression": self.noise_suppression,
            "auto_gain": self.auto_gain,
            "vad_enabled": self.vad_enabled,
            "vad_threshold": self.vad_threshold,
            "interrupt_enabled": self.interrupt_enabled,
            "max_interruptions": self.max_interruptions,
            "buffer_size": self.buffer_size,
        }


@dataclass
class AudioChunk:
    """A chunk of audio data.

    Attributes:
        data: Raw audio bytes.
        timestamp: When the chunk was recorded.
        duration_ms: Duration of the chunk in milliseconds.
        sample_rate: Sample rate in Hz.
        format: Audio format.
        is_speech: Whether VAD detected speech.
        is_interrupt: Whether this is an interrupt signal.
        source: Source identifier (e.g. 'mic', 'speaker').
    """
    data: bytes = b""
    timestamp: float = 0.0
    duration_ms: float = 0.0
    sample_rate: int = 16000
    format: AudioFormat = AudioFormat.PCM_16KHZ_16BIT_MONO
    is_speech: bool = False
    is_interrupt: bool = False
    source: str = "mic"

    @property
    def num_samples(self) -> int:
        """Number of audio samples."""
        if self.format.bytes_per_sample > 0:
            return len(self.data) // self.format.bytes_per_sample
        return 0


@dataclass
class VoiceEvent:
    """An event emitted by the voice mode system.

    Attributes:
        event_type: Type of event.
        data: Event-specific data.
        timestamp: When the event occurred.
        session_id: Voice session identifier.
    """
    class Type(Enum):
        """Voice event types."""
        SESSION_STARTED = "session_started"
        SESSION_ENDED = "session_ended"
        SPEECH_STARTED = "speech_started"
        SPEECH_ENDED = "speech_ended"
        INTERRUPT = "interrupt"
        TTS_STARTED = "tts_started"
        TTS_ENDED = "tts_ended"
        STT_RESULT = "stt_result"
        STT_PARTIAL = "stt_partial"
        ERROR = "error"
        WARNING = "warning"
        PROVIDER_CHANGED = "provider_changed"
        CONFIG_CHANGED = "config_changed"
        VOLUME_CHANGED = "volume_changed"
        MUTE_CHANGED = "mute_changed"

    event_type: Type = Type.SESSION_STARTED
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    session_id: str = ""


@dataclass
class VoiceState:
    """Current state of the voice mode.

    Attributes:
        is_active: Whether voice mode is currently active.
        is_speaking: Whether TTS is currently speaking.
        is_listening: Whether STT is currently listening.
        is_muted: Whether the microphone is muted.
        is_paused: Whether the session is paused.
        session_id: Current session identifier.
        total_interruptions: Total interrupt count.
        current_provider_tts: Current TTS provider.
        current_provider_stt: Current STT provider.
        language: Current language setting.
        voice_id: Current voice identifier.
        uptime_seconds: Session uptime in seconds.
    """
    is_active: bool = False
    is_speaking: bool = False
    is_listening: bool = False
    is_muted: bool = False
    is_paused: bool = False
    session_id: str = ""
    total_interruptions: int = 0
    current_provider_tts: VoiceProvider = VoiceProvider.SYSTEM_DEFAULT
    current_provider_stt: VoiceProvider = VoiceProvider.SYSTEM_DEFAULT
    language: str = "en-US"
    voice_id: str = "default"
    uptime_seconds: float = 0.0


class AudioProcessor:
    """Audio signal processing utilities.

    Provides methods for echo cancellation, noise suppression,
    voice activity detection, and audio format conversion.
    """

    def __init__(self, config: VoiceConfig) -> None:
        """Initialize the audio processor.

        Args:
            config: Voice configuration.
        """
        self._config = config
        self._energy_threshold = config.vad_threshold
        self._silence_duration = 0.0
        self._max_silence = 1.0  # seconds of silence before speech end
        self._speech_active = False
        self._running_energy: float = 0.0
        self._alpha = 0.95  # smoothing factor for energy

    def compute_rms_energy(self, audio_data: bytes) -> float:
        """Compute the RMS energy of PCM audio data.

        Args:
            audio_data: Raw PCM 16-bit audio bytes.

        Returns:
            RMS energy value (0.0 - 1.0).
        """
        if not audio_data:
            return 0.0

        import struct

        # Unpack PCM 16-bit samples
        num_samples = len(audio_data) // 2
        if num_samples == 0:
            return 0.0

        samples = struct.unpack(f"<{num_samples}h", audio_data[: num_samples * 2])

        # Compute RMS
        sum_sq = sum(s * s for s in samples)
        rms = (sum_sq / num_samples) ** 0.5

        # Normalize to 0-1 range (max for 16-bit is 32767)
        normalized = rms / 32767.0

        # Apply smoothing
        self._running_energy = (
            self._alpha * self._running_energy
            + (1 - self._alpha) * normalized
        )

        return self._running_energy

    def detect_voice_activity(
        self, audio_data: bytes, chunk_duration_ms: float
    ) -> bool:
        """Detect whether audio chunk contains speech.

        Uses energy-based voice activity detection with
        adaptive thresholding.

        Args:
            audio_data: Raw PCM audio bytes.
            chunk_duration_ms: Duration of the chunk in ms.

        Returns:
            True if speech is detected.
        """
        energy = self.compute_rms_energy(audio_data)

        if energy > self._energy_threshold:
            self._silence_duration = 0.0
            self._speech_active = True
            return True
        else:
            self._silence_duration += chunk_duration_ms / 1000.0
            if self._silence_duration > self._max_silence:
                self._speech_active = False
            return self._speech_active

    def apply_noise_gate(
        self, audio_data: bytes, gate_threshold: float = 0.01
    ) -> bytes:
        """Apply a noise gate to reduce background noise.

        Args:
            audio_data: Raw PCM 16-bit audio bytes.
            gate_threshold: Energy threshold below which audio is silenced.

        Returns:
            Processed audio bytes.
        """
        energy = self.compute_rms_energy(audio_data)

        if energy < gate_threshold:
            # Below gate: output silence
            return b"\x00" * len(audio_data)

        return audio_data

    def normalize_volume(
        self, audio_data: bytes, target_rms: float = 0.5
    ) -> bytes:
        """Normalize audio volume to a target RMS level.

        Args:
            audio_data: Raw PCM 16-bit audio bytes.
            target_rms: Target RMS energy level.

        Returns:
            Normalized audio bytes.
        """
        if not audio_data:
            return audio_data

        import struct

        num_samples = len(audio_data) // 2
        if num_samples == 0:
            return audio_data

        samples = list(struct.unpack(f"<{num_samples}h", audio_data))

        # Compute current RMS
        sum_sq = sum(s * s for s in samples)
        current_rms = (sum_sq / num_samples) ** 0.5
        if current_rms == 0:
            return audio_data

        # Compute gain factor
        gain = target_rms * 32767.0 / current_rms

        # Apply gain with soft clipping
        max_val = 32767
        processed = []
        for s in samples:
            scaled = int(s * gain)
            # Soft clipping
            if abs(scaled) > max_val:
                scaled = int(max_val * (2.0 / 3.14159) * __import__("math").asin(scaled / max_val))
            processed.append(max(-max_val, min(max_val, scaled)))

        return struct.pack(f"<{num_samples}h", *processed)

    def detect_interrupt(
        self, audio_data: bytes, is_speaking: bool
    ) -> bool:
        """Detect if the user is trying to interrupt the TTS output.

        An interrupt is detected when speech energy exceeds a
        higher threshold while TTS is actively speaking.

        Args:
            audio_data: Raw PCM audio bytes.
            is_speaking: Whether TTS is currently speaking.

        Returns:
            True if an interrupt is detected.
        """
        if not is_speaking:
            return False

        # Use a higher threshold for interrupts (user must speak louder)
        energy = self.compute_rms_energy(audio_data)
        interrupt_threshold = self._energy_threshold * 2.5

        return energy > interrupt_threshold


class TTSProviderInterface:
    """Abstract base class for TTS providers.

    Each concrete provider implements text-to-speech conversion
    with provider-specific API calls and audio format handling.
    """

    def __init__(self, voice_id: str = "default", language: str = "en-US") -> None:
        """Initialize the TTS provider.

        Args:
            voice_id: Voice identifier.
            language: Language code.
        """
        self.voice_id = voice_id
        self.language = language
        self._is_available = False
        self._initialized = False

    @property
    def is_available(self) -> bool:
        """Whether this provider is available."""
        return self._is_available

    async def initialize(self) -> None:
        """Initialize the provider (load models, verify API keys, etc.)."""
        self._initialized = True

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio.

        Args:
            text: The text to synthesize.

        Returns:
            Raw audio bytes.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError

    async def get_available_voices(self) -> List[Dict[str, str]]:
        """Get list of available voices.

        Returns:
            List of voice info dicts with 'id', 'name', 'language' keys.
        """
        return [{"id": "default", "name": "Default", "language": self.language}]

    async def shutdown(self) -> None:
        """Clean up provider resources."""
        self._initialized = False


class SystemTTSProvider(TTSProviderInterface):
    """System-default TTS using platform TTS engines.

    Falls back to available platform speech synthesis.
    """

    def __init__(self, voice_id: str = "default", language: str = "en-US") -> None:
        super().__init__(voice_id, language)
        self._is_available = True

    async def synthesize(self, text: str) -> bytes:
        """Synthesize using platform TTS (placeholder implementation).

        Note: In production, this would use pyttsx3 or similar.
        This is a placeholder that returns silent audio.
        """
        # Placeholder: generate silence for the expected duration
        # (approximately 100 bytes per character at 16kHz 16-bit)
        duration_samples = max(int(len(text) * 60), 100)  # ~60 samples per char
        silence = b"\x00" * (duration_samples * 2)
        logger.debug("System TTS: synthesized %d chars -> %d bytes", len(text), len(silence))
        return silence


class VoiceMode:
    """Real-time voice conversation mode.

    Manages the full-duplex voice conversation pipeline including
    audio input/output, TTS/STT provider management, voice activity
    detection, echo cancellation, and interrupt handling.

    Example::

        voice = VoiceMode()
        voice.on_audio_input(lambda chunk: print(f"Got audio: {len(chunk.data)} bytes"))
        voice.on_audio_output(lambda chunk: print(f"Sending audio: {len(chunk.data)} bytes"))

        await voice.start()
        # ... conversation happens ...
        await voice.stop()
    """

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        """Initialize the voice mode.

        Args:
            config: Optional voice configuration. Uses defaults if not provided.
        """
        self._config = config or VoiceConfig()
        self._state = VoiceState(
            current_provider_tts=self._config.tts_provider,
            current_provider_stt=self._config.stt_provider,
            language=self._config.language,
            voice_id=self._config.voice_id,
        )
        self._audio_processor = AudioProcessor(self._config)
        self._tts_providers: Dict[VoiceProvider, TTSProviderInterface] = {}
        self._current_tts: Optional[TTSProviderInterface] = None
        self._audio_input_callbacks: List[Callable[[AudioChunk], Coroutine[Any, Any, None]]] = []
        self._audio_output_callbacks: List[Callable[[AudioChunk], Coroutine[Any, Any, None]]] = []
        self._event_callbacks: List[Callable[[VoiceEvent], Coroutine[Any, Any, None]]] = []
        self._interrupt_count = 0
        self._session_start_time: float = 0.0
        self._processing_loop: Optional[asyncio.Task] = None
        self._output_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        self._input_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        self._active = False
        self._lock = asyncio.Lock()
        self._pending_interrupts: Deque[AudioChunk] = deque()

        # Register default TTS provider
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        """Register the default TTS providers."""
        system_tts = SystemTTSProvider(
            voice_id=self._config.voice_id,
            language=self._config.language,
        )
        self._tts_providers[VoiceProvider.SYSTEM_DEFAULT] = system_tts
        self._current_tts = system_tts

    async def start(self) -> bool:
        """Start the voice mode session.

        Initializes all providers and begins the audio processing loop.

        Returns:
            True if the session started successfully.
        """
        async with self._lock:
            if self._active:
                logger.warning("Voice mode is already active")
                return False

            try:
                # Initialize current TTS provider
                if self._current_tts and not self._current_tts._initialized:
                    await self._current_tts.initialize()

                # Update state
                self._state.session_id = str(uuid.uuid4())[:8]
                self._state.is_active = True
                self._state.is_listening = True
                self._session_start_time = time.monotonic()
                self._active = True
                self._interrupt_count = 0

                # Start the processing loop
                self._processing_loop = asyncio.create_task(
                    self._audio_processing_loop()
                )

                # Emit session started event
                await self._emit_event(
                    VoiceEvent.Type.SESSION_STARTED,
                    {"session_id": self._state.session_id},
                )

                logger.info(
                    "Voice mode started (session: %s)", self._state.session_id
                )
                return True

            except Exception as e:
                logger.error("Failed to start voice mode: %s", e)
                await self._emit_event(
                    VoiceEvent.Type.ERROR,
                    {"error": str(e), "stage": "start"},
                )
                return False

    async def stop(self) -> None:
        """Stop the voice mode session.

        Gracefully stops all audio processing, flushes queues,
        and cleans up provider resources.
        """
        async with self._lock:
            if not self._active:
                return

            self._active = False
            self._state.is_active = False
            self._state.is_speaking = False
            self._state.is_listening = False

            # Cancel processing loop
            if self._processing_loop:
                self._processing_loop.cancel()
                try:
                    await self._processing_loop
                except asyncio.CancelledError:
                    pass
                self._processing_loop = None

            # Flush queues
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            while not self._output_queue.empty():
                try:
                    self._output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Shutdown TTS provider
            if self._current_tts and self._current_tts._initialized:
                await self._current_tts.shutdown()

            # Update uptime
            if self._session_start_time > 0:
                self._state.uptime_seconds = time.monotonic() - self._session_start_time

            # Emit session ended event
            await self._emit_event(
                VoiceEvent.Type.SESSION_ENDED,
                {
                    "session_id": self._state.session_id,
                    "uptime": self._state.uptime_seconds,
                    "interruptions": self._interrupt_count,
                },
            )

            logger.info("Voice mode stopped (session: %s)", self._state.session_id)

    def is_active(self) -> bool:
        """Check if voice mode is currently active.

        Returns:
            True if the voice session is active.
        """
        return self._active

    def get_state(self) -> VoiceState:
        """Get the current voice state.

        Returns:
            A copy of the current VoiceState.
        """
        state = VoiceState(
            is_active=self._active,
            is_speaking=self._state.is_speaking,
            is_listening=self._state.is_listening,
            is_muted=self._state.is_muted,
            is_paused=self._state.is_paused,
            session_id=self._state.session_id,
            total_interruptions=self._interrupt_count,
            current_provider_tts=self._state.current_provider_tts,
            current_provider_stt=self._state.current_provider_stt,
            language=self._state.language,
            voice_id=self._state.voice_id,
        )
        if self._session_start_time > 0 and self._active:
            state.uptime_seconds = time.monotonic() - self._session_start_time
        return state

    async def set_provider(
        self,
        provider: VoiceProvider,
        provider_type: str = "tts",
    ) -> bool:
        """Switch the TTS or STT provider.

        Args:
            provider: The provider to switch to.
            provider_type: Either 'tts' or 'stt'.

        Returns:
            True if the switch was successful.
        """
        try:
            if provider_type == "tts":
                if provider in self._tts_providers:
                    if self._current_tts and self._current_tts._initialized:
                        await self._current_tts.shutdown()
                    self._current_tts = self._tts_providers[provider]
                    if not self._current_tts._initialized:
                        await self._current_tts.initialize()
                    self._state.current_provider_tts = provider

                    await self._emit_event(
                        VoiceEvent.Type.PROVIDER_CHANGED,
                        {"provider": provider.value, "type": "tts"},
                    )
                    logger.info("TTS provider changed to: %s", provider.value)
                    return True
                else:
                    logger.warning("TTS provider not registered: %s", provider.value)
                    return False
            elif provider_type == "stt":
                self._state.current_provider_stt = provider
                await self._emit_event(
                    VoiceEvent.Type.PROVIDER_CHANGED,
                    {"provider": provider.value, "type": "stt"},
                )
                logger.info("STT provider changed to: %s", provider.value)
                return True
            else:
                logger.error("Invalid provider type: %s", provider_type)
                return False

        except Exception as e:
            logger.error("Failed to switch provider: %s", e)
            await self._emit_event(
                VoiceEvent.Type.ERROR,
                {"error": str(e), "stage": "provider_switch"},
            )
            return False

    def on_audio_input(
        self,
        callback: Callable[[AudioChunk], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a callback for audio input events.

        Called when new audio is captured from the microphone.

        Args:
            callback: Async callback receiving AudioChunk.
        """
        self._audio_input_callbacks.append(callback)

    def on_audio_output(
        self,
        callback: Callable[[AudioChunk], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a callback for audio output events.

        Called when audio is ready to be played through the speaker.

        Args:
            callback: Async callback receiving AudioChunk.
        """
        self._audio_output_callbacks.append(callback)

    def on_event(
        self,
        callback: Callable[[VoiceEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a callback for voice events.

        Args:
            callback: Async callback receiving VoiceEvent.
        """
        self._event_callbacks.append(callback)

    async def set_language(self, language: str) -> None:
        """Set the language for TTS and STT.

        Args:
            language: Language code (e.g. 'en-US', 'zh-CN', 'ja-JP').
        """
        self._config.language = language
        self._state.language = language

        if self._current_tts:
            self._current_tts.language = language

        await self._emit_event(
            VoiceEvent.Type.CONFIG_CHANGED,
            {"setting": "language", "value": language},
        )
        logger.info("Language set to: %s", language)

    async def set_voice(self, voice_id: str) -> None:
        """Set the voice identifier for TTS.

        Args:
            voice_id: Voice identifier string.
        """
        self._config.voice_id = voice_id
        self._state.voice_id = voice_id

        if self._current_tts:
            self._current_tts.voice_id = voice_id

        await self._emit_event(
            VoiceEvent.Type.CONFIG_CHANGED,
            {"setting": "voice_id", "value": voice_id},
        )
        logger.info("Voice set to: %s", voice_id)

    async def send_text_for_speech(self, text: str) -> Optional[bytes]:
        """Send text to be synthesized and played.

        Args:
            text: The text to speak.

        Returns:
            The synthesized audio bytes, or None if synthesis failed.
        """
        if not self._active:
            logger.warning("Cannot synthesize: voice mode not active")
            return None

        try:
            await self._emit_event(
                VoiceEvent.Type.TTS_STARTED,
                {"text_length": len(text)},
            )
            self._state.is_speaking = True

            if self._current_tts:
                audio = await self._current_tts.synthesize(text)
                if audio:
                    chunk = AudioChunk(
                        data=audio,
                        timestamp=time.monotonic(),
                        duration_ms=len(audio) / (self._config.sample_rate * 2) * 1000,
                        sample_rate=self._config.sample_rate,
                        format=self._config.output_format,
                        source="tts",
                    )
                    await self._output_queue.put(chunk)
                return audio

        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            await self._emit_event(
                VoiceEvent.Type.ERROR,
                {"error": str(e), "stage": "tts_synthesis"},
            )
        finally:
            self._state.is_speaking = False
            await self._emit_event(
                VoiceEvent.Type.TTS_ENDED,
                {"text_length": len(text)},
            )

        return None

    async def submit_audio_input(self, audio_data: bytes, source: str = "mic") -> None:
        """Submit raw audio data for processing.

        Args:
            audio_data: Raw PCM audio bytes.
            source: Source identifier.
        """
        if not self._active or self._state.is_muted:
            return

        chunk = AudioChunk(
            data=audio_data,
            timestamp=time.monotonic(),
            duration_ms=len(audio_data) / (self._config.sample_rate * 2) * 1000,
            sample_rate=self._config.sample_rate,
            format=self._config.input_format,
            source=source,
        )

        # Voice activity detection
        if self._config.vad_enabled:
            chunk.is_speech = self._audio_processor.detect_voice_activity(
                audio_data, chunk.duration_ms
            )

            # Check for interrupts
            if self._config.interrupt_enabled and self._state.is_speaking:
                if self._audio_processor.detect_interrupt(
                    audio_data, self._state.is_speaking
                ):
                    chunk.is_interrupt = True
                    self._interrupt_count += 1
                    self._state.total_interruptions = self._interrupt_count

                    await self._emit_event(
                        VoiceEvent.Type.INTERRUPT,
                        {"count": self._interrupt_count},
                    )

                    # Handle interrupt: stop current TTS output
                    # Flush the output queue
                    while not self._output_queue.empty():
                        try:
                            self._output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

        # Apply noise suppression if enabled
        if self._config.noise_suppression and not chunk.is_speech:
            chunk.data = self._audio_processor.apply_noise_gate(audio_data)

        await self._input_queue.put(chunk)

    async def set_muted(self, muted: bool) -> None:
        """Mute or unmute the microphone.

        Args:
            muted: True to mute, False to unmute.
        """
        self._state.is_muted = muted
        await self._emit_event(
            VoiceEvent.Type.MUTE_CHANGED,
            {"muted": muted},
        )
        logger.info("Microphone %s", "muted" if muted else "unmuted")

    async def pause(self) -> None:
        """Pause the voice session (stop processing but keep state)."""
        self._state.is_paused = True
        self._state.is_listening = False
        logger.info("Voice session paused")

    async def resume(self) -> None:
        """Resume a paused voice session."""
        self._state.is_paused = False
        self._state.is_listening = True
        logger.info("Voice session resumed")

    async def _audio_processing_loop(self) -> None:
        """Main audio processing loop.

        Processes audio from the input queue and dispatches to
        registered callbacks. Also handles output queue dispatch.
        """
        logger.debug("Audio processing loop started")

        try:
            while self._active:
                # Process input
                try:
                    chunk = await asyncio.wait_for(
                        self._input_queue.get(), timeout=0.1
                    )

                    # Dispatch to audio input callbacks
                    for callback in self._audio_input_callbacks:
                        try:
                            await callback(chunk)
                        except Exception as e:
                            logger.error(
                                "Audio input callback error: %s", e
                            )

                    # Track speech state
                    if chunk.is_speech and not self._state.is_listening:
                        self._state.is_listening = True
                        await self._emit_event(VoiceEvent.Type.SPEECH_STARTED, {})
                    elif not chunk.is_speech and self._state.is_listening:
                        # Will be set to False by silence timeout
                        pass

                except asyncio.TimeoutError:
                    # Check for speech end (no speech detected)
                    if not self._audio_processor._speech_active and self._state.is_listening:
                        self._state.is_listening = False
                        await self._emit_event(VoiceEvent.Type.SPEECH_ENDED, {})

                # Process output
                if not self._output_queue.empty():
                    try:
                        output_chunk = self._output_queue.get_nowait()
                        for callback in self._audio_output_callbacks:
                            try:
                                await callback(output_chunk)
                            except Exception as e:
                                logger.error(
                                    "Audio output callback error: %s", e
                                )
                    except asyncio.QueueEmpty:
                        pass

                # Small sleep to prevent busy loop
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            logger.debug("Audio processing loop cancelled")
        except Exception as e:
            logger.error("Audio processing loop error: %s", e)
            await self._emit_event(
                VoiceEvent.Type.ERROR,
                {"error": str(e), "stage": "processing_loop"},
            )

    async def _emit_event(
        self, event_type: VoiceEvent.Type, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Emit a voice event to all registered callbacks.

        Args:
            event_type: The type of event.
            data: Optional event data.
        """
        event = VoiceEvent(
            event_type=event_type,
            data=data or {},
            timestamp=time.monotonic(),
            session_id=self._state.session_id,
        )

        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error("Event callback error: %s", e)

    def register_tts_provider(
        self, provider: VoiceProvider, tts: TTSProviderInterface
    ) -> None:
        """Register a custom TTS provider.

        Args:
            provider: The provider enum value.
            tts: The TTS provider instance.
        """
        self._tts_providers[provider] = tts
        logger.info("Registered TTS provider: %s", provider.value)

    async def get_available_voices(self) -> List[Dict[str, str]]:
        """Get available voices from the current TTS provider.

        Returns:
            List of voice info dicts.
        """
        if self._current_tts:
            return await self._current_tts.get_available_voices()
        return []
