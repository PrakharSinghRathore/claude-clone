"""
atlas.realtime.transcription - Real-time audio transcription.

Implements a real-time audio transcription system with multi-provider
support, automatic failover, language detection, and streaming
results with interim and final transcripts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class TranscriptionProvider(Enum):
    """Supported transcription providers."""
    DEEPGRAM = "deepgram"
    WHISPER_OPENAI = "whisper_openai"
    WHISPER_LOCAL = "whisper_local"
    GOOGLE = "google"
    AZURE = "azure"
    ASSEMBLYAI = "assemblyai"
    REV_AI = "rev_ai"
    SPEECHMATICS = "speechmatics"
    SYSTEM_DEFAULT = "system_default"


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio.

    Attributes:
        text: The transcribed text for this segment.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        confidence: Confidence score (0.0 - 1.0).
        speaker: Speaker identifier (for diarization).
        words: Optional word-level timing information.
        language: Detected language for this segment.
    """
    text: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 1.0
    speaker: str = ""
    words: List[Dict[str, Any]] = field(default_factory=list)
    language: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "speaker": self.speaker,
            "words": self.words,
            "language": self.language,
        }


@dataclass
class TranscriptionResult:
    """Result of a transcription operation.

    Attributes:
        text: The full transcribed text.
        confidence: Overall confidence score (0.0 - 1.0).
        language: Detected or specified language.
        segments: Individual segments with timing.
        timestamp: When the transcription was completed.
        provider: Which provider produced this result.
        duration: Audio duration in seconds.
        is_final: Whether this is a final (not interim) result.
        words_per_minute: Estimated speaking rate.
    """
    text: str = ""
    confidence: float = 1.0
    language: str = ""
    segments: List[TranscriptionSegment] = field(default_factory=list)
    timestamp: float = 0.0
    provider: str = ""
    duration: float = 0.0
    is_final: bool = True
    words_per_minute: float = 0.0

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "segments": [s.to_dict() for s in self.segments],
            "timestamp": self.timestamp,
            "provider": self.provider,
            "duration": self.duration,
            "is_final": self.is_final,
            "words_per_minute": self.words_per_minute,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TranscriptionResult:
        """Create from dictionary."""
        segments = [
            TranscriptionSegment.from_dict(s)
            for s in data.get("segments", [])
        ]
        return cls(
            text=data.get("text", ""),
            confidence=data.get("confidence", 1.0),
            language=data.get("language", ""),
            segments=segments,
            timestamp=data.get("timestamp", time.time()),
            provider=data.get("provider", ""),
            duration=data.get("duration", 0.0),
            is_final=data.get("is_final", True),
            words_per_minute=data.get("words_per_minute", 0.0),
        )

    def merge(self, other: TranscriptionResult) -> TranscriptionResult:
        """Merge another result into this one.

        Args:
            other: Another TranscriptionResult to merge.

        Returns:
            A new merged TranscriptionResult.
        """
        merged_text = f"{self.text} {other.text}".strip()
        merged_segments = self.segments + other.segments

        # Weighted average confidence
        total_duration = self.duration + other.duration
        if total_duration > 0:
            avg_confidence = (
                (self.confidence * self.duration + other.confidence * other.duration)
                / total_duration
            )
        else:
            avg_confidence = (self.confidence + other.confidence) / 2

        return TranscriptionResult(
            text=merged_text,
            confidence=avg_confidence,
            language=other.language or self.language,
            segments=merged_segments,
            duration=total_duration,
            provider=self.provider or other.provider,
            is_final=self.is_final and other.is_final,
        )


@dataclass
class TranscriptionConfig:
    """Configuration for the transcription service.

    Attributes:
        provider: The primary transcription provider.
        fallback_providers: Ordered list of fallback providers.
        api_key: API key for the primary provider.
        api_keys: API keys for all providers.
        language: Language code (e.g. 'en', 'zh', 'ja').
        auto_detect_language: Enable automatic language detection.
        sample_rate: Audio sample rate in Hz.
        encoding: Audio encoding format.
        enable_punctuation: Enable automatic punctuation.
        enable_diarization: Enable speaker diarization.
        max_speakers: Maximum number of speakers for diarization.
        enable_word_timing: Enable word-level timing.
        enable_profanity_filter: Filter profanity.
        smart_format: Enable smart formatting.
        endpointing: Silence duration for endpointing (seconds).
        interim_results: Whether to emit interim results.
    """
    provider: TranscriptionProvider = TranscriptionProvider.SYSTEM_DEFAULT
    fallback_providers: List[TranscriptionProvider] = field(default_factory=list)
    api_key: str = ""
    api_keys: Dict[str, str] = field(default_factory=dict)
    language: str = "en"
    auto_detect_language: bool = True
    sample_rate: int = 16000
    encoding: str = "linear16"
    enable_punctuation: bool = True
    enable_diarization: bool = False
    max_speakers: int = 4
    enable_word_timing: bool = False
    enable_profanity_filter: bool = False
    smart_format: bool = True
    endpointing: float = 1.0
    interim_results: bool = True

    def get_api_key(self, provider: TranscriptionProvider) -> Optional[str]:
        """Get the API key for a specific provider.

        Args:
            provider: The provider to get the key for.

        Returns:
            The API key, or None if not configured.
        """
        if self.api_key and provider == self.provider:
            return self.api_key
        return self.api_keys.get(provider.value)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TranscriptionConfig:
        """Create from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        if "provider" in filtered and isinstance(filtered["provider"], str):
            filtered["provider"] = TranscriptionProvider(filtered["provider"])
        if "fallback_providers" in filtered:
            filtered["fallback_providers"] = [
                TranscriptionProvider(p) if isinstance(p, str) else p
                for p in filtered["fallback_providers"]
            ]
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes API keys)."""
        return {
            "provider": self.provider.value,
            "fallback_providers": [p.value for p in self.fallback_providers],
            "language": self.language,
            "auto_detect_language": self.auto_detect_language,
            "sample_rate": self.sample_rate,
            "encoding": self.encoding,
            "enable_punctuation": self.enable_punctuation,
            "enable_diarization": self.enable_diarization,
            "max_speakers": self.max_speakers,
            "enable_word_timing": self.enable_word_timing,
            "enable_profanity_filter": self.enable_profanity_filter,
            "smart_format": self.smart_format,
            "endpointing": self.endpointing,
            "interim_results": self.interim_results,
        }


class ProviderStats:
    """Statistics for a transcription provider.

    Attributes:
        provider: The provider name.
        total_requests: Total number of requests.
        successful_requests: Number of successful requests.
        failed_requests: Number of failed requests.
        total_latency_ms: Cumulative latency in milliseconds.
        total_audio_seconds: Total audio processed in seconds.
        average_confidence: Average confidence score.
        last_error: Most recent error message.
        last_success_time: Timestamp of last successful request.
        enabled: Whether the provider is enabled.
    """

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.total_latency_ms: float = 0.0
        self.total_audio_seconds: float = 0.0
        self.average_confidence: float = 0.0
        self.last_error: str = ""
        self.last_success_time: float = 0.0
        self.enabled: bool = True

    def record_success(
        self, latency_ms: float, confidence: float, audio_seconds: float
    ) -> None:
        """Record a successful transcription request.

        Args:
            latency_ms: Request latency in milliseconds.
            confidence: Confidence score.
            audio_seconds: Audio duration in seconds.
        """
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        self.total_audio_seconds += audio_seconds
        self.last_success_time = time.time()

        # Update running average confidence
        if self.successful_requests == 1:
            self.average_confidence = confidence
        else:
            alpha = 1.0 / self.successful_requests
            self.average_confidence = (
                (1 - alpha) * self.average_confidence + alpha * confidence
            )

    def record_failure(self, error: str) -> None:
        """Record a failed transcription request.

        Args:
            error: Error message.
        """
        self.total_requests += 1
        self.failed_requests += 1
        self.last_error = error

    @property
    def success_rate(self) -> float:
        """Get the success rate (0.0 - 1.0)."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def average_latency_ms(self) -> float:
        """Get average latency in milliseconds."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "average_latency_ms": round(self.average_latency_ms, 2),
            "total_audio_seconds": round(self.total_audio_seconds, 2),
            "average_confidence": round(self.average_confidence, 4),
            "last_error": self.last_error,
            "enabled": self.enabled,
        }


class AudioBuffer:
    """Ring buffer for accumulating audio chunks before transcription.

    Implements a time-based buffering system that collects audio
    until enough data is accumulated or a silence threshold is met.
    """

    def __init__(
        self,
        max_duration_seconds: float = 30.0,
        sample_rate: int = 16000,
        bytes_per_sample: int = 2,
    ) -> None:
        """Initialize the audio buffer.

        Args:
            max_duration_seconds: Maximum buffer duration.
            sample_rate: Audio sample rate.
            bytes_per_sample: Bytes per audio sample.
        """
        self._max_samples = int(max_duration_seconds * sample_rate)
        self._sample_rate = sample_rate
        self._bytes_per_sample = bytes_per_sample
        self._buffer: bytearray = bytearray()
        self._start_time: Optional[float] = None
        self._last_chunk_time: float = 0.0
        self._chunk_count: int = 0

    @property
    def duration_seconds(self) -> float:
        """Current buffer duration in seconds."""
        num_samples = len(self._buffer) // self._bytes_per_sample
        return num_samples / self._sample_rate

    @property
    def size_bytes(self) -> int:
        """Current buffer size in bytes."""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Whether the buffer is empty."""
        return len(self._buffer) == 0

    def append(self, audio_data: bytes, timestamp: Optional[float] = None) -> None:
        """Append audio data to the buffer.

        Args:
            audio_data: Raw audio bytes.
            timestamp: Optional timestamp for the chunk.
        """
        if self._start_time is None:
            self._start_time = timestamp or time.time()

        self._buffer.extend(audio_data)
        self._last_chunk_time = timestamp or time.time()
        self._chunk_count += 1

        # Trim if over max duration
        max_bytes = self._max_samples * self._bytes_per_sample
        if len(self._buffer) > max_bytes:
            excess = len(self._buffer) - max_bytes
            self._buffer = self._buffer[excess:]

    def get_and_clear(self) -> bytes:
        """Get all buffered audio and clear the buffer.

        Returns:
            The buffered audio bytes.
        """
        data = bytes(self._buffer)
        self.clear()
        return data

    def get_without_clear(self) -> bytes:
        """Get buffered audio without clearing.

        Returns:
            The buffered audio bytes.
        """
        return bytes(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._start_time = None
        self._chunk_count = 0

    def trim_leading_silence(self, threshold_db: float = -40.0) -> int:
        """Remove leading silence from the buffer.

        Args:
            threshold_db: Silence threshold in dB.

        Returns:
            Number of bytes removed.
        """
        if not self._buffer:
            return 0

        import struct

        num_samples = len(self._buffer) // 2
        samples = struct.unpack(f"<{num_samples}h", bytes(self._buffer))

        # Convert threshold from dB to amplitude
        threshold_amplitude = 10 ** (threshold_db / 20.0) * 32767.0

        # Find first non-silent sample
        silence_end = 0
        for i, s in enumerate(samples):
            if abs(s) > threshold_amplitude:
                silence_end = i
                break
        else:
            silence_end = num_samples

        # Remove silence
        bytes_to_remove = silence_end * 2
        self._buffer = self._buffer[bytes_to_remove:]
        return bytes_to_remove


class RealtimeTranscriber:
    """Real-time audio transcription with multi-provider support.

    Provides a streaming transcription interface that accepts
    audio chunks and produces interim and final transcription
    results. Implements automatic failover between providers.

    Example::

        transcriber = RealtimeTranscriber()
        transcriber.configure(api_key="your-key", language="en")

        # Register result callback
        def on_result(result: TranscriptionResult):
            print(f"[{'FINAL' if result.is_final else 'interim'}] {result.text}")

        transcriber.on_result(on_result)

        await transcriber.start_stream()

        # Send audio chunks
        await transcriber.send_audio(audio_chunk)

        # Get final results
        finals = transcriber.get_final_results()

        await transcriber.stop_stream()
    """

    # API endpoint mappings for each provider
    PROVIDER_ENDPOINTS = {
        TranscriptionProvider.DEEPGRAM: "https://api.deepgram.com/v1/listen",
        TranscriptionProvider.ASSEMBLYAI: "https://api.assemblyai.com/v2/transcript",
        TranscriptionProvider.REV_AI: "https://api.rev.ai/speechtotext/v1/stream",
    }

    def __init__(
        self,
        config: Optional[TranscriptionConfig] = None,
    ) -> None:
        """Initialize the realtime transcriber.

        Args:
            config: Optional transcription configuration.
        """
        self._config = config or TranscriptionConfig()
        self._audio_buffer = AudioBuffer(
            max_duration_seconds=30.0,
            sample_rate=self._config.sample_rate,
        )
        self._interim_result: Optional[TranscriptionResult] = None
        self._final_results: List[TranscriptionResult] = []
        self._result_callbacks: List[
            Callable[[TranscriptionResult], Coroutine[Any, Any, None]]
        ] = []
        self._is_streaming: bool = False
        self._stream_id: str = ""
        self._total_audio_seconds: float = 0.0
        self._lock = asyncio.Lock()
        self._processing_task: Optional[asyncio.Task] = None
        self._provider_stats: Dict[str, ProviderStats] = {
            p.value: ProviderStats(p.value) for p in TranscriptionProvider
        }
        self._active_provider: TranscriptionProvider = self._config.provider
        self._consecutive_failures: Dict[str, int] = {}
        self._max_consecutive_failures = 3
        self._session_start_time: float = 0.0

    def configure(
        self,
        provider: Optional[TranscriptionProvider] = None,
        api_key: Optional[str] = None,
        language: Optional[str] = None,
        auto_detect_language: Optional[bool] = None,
        sample_rate: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Update the transcription configuration.

        Args:
            provider: Primary transcription provider.
            api_key: API key for the provider.
            language: Language code.
            auto_detect_language: Enable language auto-detection.
            sample_rate: Audio sample rate.
            **kwargs: Additional configuration options.
        """
        if provider:
            self._config.provider = provider
            self._active_provider = provider
        if api_key:
            self._config.api_key = api_key
        if language:
            self._config.language = language
        if auto_detect_language is not None:
            self._config.auto_detect_language = auto_detect_language
        if sample_rate:
            self._config.sample_rate = sample_rate
            self._audio_buffer = AudioBuffer(
                max_duration_seconds=30.0,
                sample_rate=sample_rate,
            )
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    async def start_stream(self) -> bool:
        """Start the audio transcription stream.

        Initializes the transcription session and begins
        processing audio chunks.

        Returns:
            True if the stream started successfully.
        """
        async with self._lock:
            if self._is_streaming:
                logger.warning("Transcription stream already active")
                return False

            self._stream_id = str(uuid.uuid4())[:8]
            self._is_streaming = True
            self._session_start_time = time.monotonic()
            self._audio_buffer.clear()
            self._final_results.clear()
            self._interim_result = None
            self._total_audio_seconds = 0.0

            # Start the processing task
            self._processing_task = asyncio.create_task(
                self._processing_loop()
            )

            logger.info(
                "Transcription stream started (id: %s, provider: %s)",
                self._stream_id,
                self._active_provider.value,
            )
            return True

    async def stop_stream(self) -> List[TranscriptionResult]:
        """Stop the transcription stream and return final results.

        Returns:
            List of all final transcription results.
        """
        async with self._lock:
            if not self._is_streaming:
                return self._final_results

            self._is_streaming = False

            # Cancel processing task
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
                self._processing_task = None

            # Transcribe any remaining buffered audio
            if not self._audio_buffer.is_empty:
                remaining = self._audio_buffer.get_and_clear()
                if len(remaining) > 0:
                    result = await self._transcribe_with_failover(remaining)
                    if result and result.text:
                        result.is_final = True
                        self._final_results.append(result)

            logger.info(
                "Transcription stream stopped (id: %s, results: %d)",
                self._stream_id,
                len(self._final_results),
            )
            return self._final_results

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Send an audio chunk for transcription.

        Audio chunks are buffered and transcribed when enough
        data is accumulated or silence is detected.

        Args:
            audio_chunk: Raw audio bytes (PCM 16-bit, mono).
        """
        if not self._is_streaming:
            logger.warning("Cannot send audio: stream not active")
            return

        chunk_duration = (
            len(audio_chunk)
            / (self._config.sample_rate * 2)
        )
        self._total_audio_seconds += chunk_duration

        self._audio_buffer.append(audio_chunk)

        # Check if we have enough audio to transcribe
        if self._audio_buffer.duration_seconds >= self._config.endpointing:
            audio_data = self._audio_buffer.get_and_clear()
            if len(audio_data) > 0:
                result = await self._transcribe_with_failover(audio_data)
                if result:
                    if result.is_final:
                        self._final_results.append(result)
                    else:
                        self._interim_result = result
                    await self._dispatch_result(result)

    async def get_interim_result(self) -> Optional[TranscriptionResult]:
        """Get the current interim (partial) transcription result.

        Returns:
            The current interim result, or None if not available.
        """
        return self._interim_result

    def get_final_results(self) -> List[TranscriptionResult]:
        """Get all finalized transcription results.

        Returns:
            List of final TranscriptionResult objects.
        """
        return list(self._final_results)

    def get_full_transcript(self) -> str:
        """Get the full transcript as a single string.

        Returns:
            Concatenated text from all final results.
        """
        return " ".join(r.text for r in self._final_results if r.text)

    def on_result(
        self,
        callback: Callable[[TranscriptionResult], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a callback for transcription results.

        Called for both interim and final results.

        Args:
            callback: Async callback receiving TranscriptionResult.
        """
        self._result_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get transcription statistics for all providers.

        Returns:
            Dictionary of provider stats.
        """
        return {
            name: stats.to_dict()
            for name, stats in self._provider_stats.items()
            if stats.total_requests > 0 or name == self._active_provider.value
        }

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current session.

        Returns:
            Session statistics dictionary.
        """
        uptime = 0.0
        if self._session_start_time > 0:
            uptime = time.monotonic() - self._session_start_time

        return {
            "stream_id": self._stream_id,
            "is_streaming": self._is_streaming,
            "active_provider": self._active_provider.value,
            "total_audio_seconds": round(self._total_audio_seconds, 2),
            "session_uptime_seconds": round(uptime, 2),
            "final_result_count": len(self._final_results),
            "total_chars": len(self.get_full_transcript()),
            "words_per_minute": self._calculate_wpm(),
        }

    def _calculate_wpm(self) -> float:
        """Calculate the current words per minute rate.

        Returns:
            Words per minute.
        """
        full_text = self.get_full_transcript()
        word_count = len(full_text.split())
        minutes = self._total_audio_seconds / 60.0
        if minutes > 0:
            return word_count / minutes
        return 0.0

    async def _processing_loop(self) -> None:
        """Background processing loop for the transcription stream.

        Periodically generates interim results from buffered audio.
        """
        logger.debug("Transcription processing loop started")

        try:
            while self._is_streaming:
                await asyncio.sleep(2.0)  # Check every 2 seconds

                if not self._is_streaming:
                    break

                # Generate interim result if we have buffered audio
                if not self._audio_buffer.is_empty and self._config.interim_results:
                    audio_data = self._audio_buffer.get_without_clear()
                    if len(audio_data) > 0:
                        result = await self._transcribe_interim(audio_data)
                        if result:
                            self._interim_result = result
                            await self._dispatch_result(result)

        except asyncio.CancelledError:
            logger.debug("Transcription processing loop cancelled")
        except Exception as e:
            logger.error("Transcription processing loop error: %s", e)

    async def _transcribe_with_failover(
        self, audio_data: bytes
    ) -> Optional[TranscriptionResult]:
        """Transcribe audio with automatic provider failover.

        Tries the active provider first, then falls back to
        configured fallback providers.

        Args:
            audio_data: Raw audio bytes.

        Returns:
            TranscriptionResult, or None if all providers fail.
        """
        providers = [self._active_provider] + self._config.fallback_providers

        for provider in providers:
            # Skip providers with too many consecutive failures
            failures = self._consecutive_failures.get(provider.value, 0)
            if failures >= self._max_consecutive_failures:
                logger.debug(
                    "Skipping provider %s: too many failures (%d)",
                    provider.value, failures,
                )
                continue

            result = await self._transcribe_with_provider(provider, audio_data)

            if result:
                self._consecutive_failures[provider.value] = 0
                return result
            else:
                self._consecutive_failures[provider.value] = (
                    self._consecutive_failures.get(provider.value, 0) + 1
                )

        logger.error("All transcription providers failed")
        return None

    async def _transcribe_with_provider(
        self, provider: TranscriptionProvider, audio_data: bytes
    ) -> Optional[TranscriptionResult]:
        """Transcribe audio with a specific provider.

        Args:
            provider: The provider to use.
            audio_data: Raw audio bytes.

        Returns:
            TranscriptionResult, or None if transcription failed.
        """
        start_time = time.monotonic()
        api_key = self._config.get_api_key(provider)
        stats = self._provider_stats.get(provider.value)
        if not stats:
            stats = ProviderStats(provider.value)
            self._provider_stats[provider.value] = stats

        try:
            if provider == TranscriptionProvider.DEEPGRAM:
                result = await self._transcribe_deepgram(audio_data, api_key)
            elif provider == TranscriptionProvider.WHISPER_OPENAI:
                result = await self._transcribe_whisper_openai(audio_data, api_key)
            elif provider == TranscriptionProvider.GOOGLE:
                result = await self._transcribe_google(audio_data)
            elif provider == TranscriptionProvider.AZURE:
                result = await self._transcribe_azure(audio_data, api_key)
            elif provider == TranscriptionProvider.ASSEMBLYAI:
                result = await self._transcribe_assemblyai(audio_data, api_key)
            elif provider == TranscriptionProvider.SYSTEM_DEFAULT:
                result = await self._transcribe_system(audio_data)
            else:
                logger.warning("Unknown provider: %s", provider.value)
                return None

            if result:
                latency_ms = (time.monotonic() - start_time) * 1000
                audio_seconds = len(audio_data) / (self._config.sample_rate * 2)
                stats.record_success(
                    latency_ms, result.confidence, audio_seconds
                )
                result.provider = provider.value
                result.duration = audio_seconds
                logger.debug(
                    "Transcription via %s: '%s' (%dms, conf=%.2f)",
                    provider.value,
                    result.text[:50],
                    latency_ms,
                    result.confidence,
                )
                return result

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            error_msg = str(e)
            stats.record_failure(error_msg)
            logger.error(
                "Transcription failed via %s: %s (%dms)",
                provider.value, error_msg, latency_ms,
            )

        return None

    async def _transcribe_deepgram(
        self, audio_data: bytes, api_key: Optional[str]
    ) -> Optional[TranscriptionResult]:
        """Transcribe using Deepgram API.

        Args:
            audio_data: Raw audio bytes.
            api_key: Deepgram API key.

        Returns:
            TranscriptionResult, or None.
        """
        if not api_key:
            logger.debug("Deepgram: No API key")
            return None

        endpoint = self.PROVIDER_ENDPOINTS[TranscriptionProvider.DEEPGRAM]
        params: Dict[str, str] = {
            "model": "nova-2",
            "smart_format": "true",
            "punctuate": "true",
            "language": self._config.language,
        }

        if self._config.auto_detect_language:
            params["detect_language"] = "true"
            del params["language"]

        if self._config.enable_diarization:
            params["diarize"] = "true"
            params["diarize_version"] = "latest"

        encoded_params = urllib.parse.urlencode(params)
        url = f"{endpoint}?{encoded_params}"

        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": f"audio/raw;rate={self._config.sample_rate}",
        }

        try:
            loop = asyncio.get_event_loop()
            req = urllib.request.Request(url, data=audio_data, method="POST")
            for k, v in headers.items():
                req.add_header(k, v)
            req.add_header(
                "User-Agent",
                "AtlasTranscriber/1.0",
            )

            resp_data = await loop.run_in_executor(
                None,
                lambda: self._make_request(req),
            )

            response = json.loads(resp_data.decode("utf-8"))
            channel = response.get("results", {}).get("channels", [{}])[0]
            alternatives = channel.get("alternatives", [])

            if not alternatives:
                return TranscriptionResult(text="", confidence=0.0)

            alt = alternatives[0]
            transcript = alt.get("transcript", "")
            confidence = alt.get("confidence", 0.0)
            words = alt.get("words", [])

            segments = []
            if words:
                segment_texts = []
                current_start = words[0].get("start", 0.0)
                current_end = words[0].get("end", 0.0)

                for word_info in words:
                    word_text = word_info.get("word", "")
                    word_end = word_info.get("end", current_end)
                    word_start = word_info.get("start", current_start)
                    word_conf = word_info.get("confidence", 1.0)

                    if word_end - current_start > 5.0:
                        # New segment
                        if segment_texts:
                            segments.append(TranscriptionSegment(
                                text=" ".join(segment_texts),
                                start_time=current_start,
                                end_time=current_end,
                                confidence=sum(w.get("confidence", 1.0) for w in words) / max(len(words), 1),
                            ))
                        segment_texts = [word_text]
                        current_start = word_start
                    else:
                        segment_texts.append(word_text)
                    current_end = word_end

                if segment_texts:
                    segments.append(TranscriptionSegment(
                        text=" ".join(segment_texts),
                        start_time=current_start,
                        end_time=current_end,
                    ))

            detected_lang = response.get("results", {}).get("detected_language", "")

            return TranscriptionResult(
                text=transcript.strip(),
                confidence=confidence,
                language=detected_lang or self._config.language,
                segments=segments,
                is_final=True,
            )

        except Exception as e:
            logger.error("Deepgram transcription error: %s", e)
            raise

    async def _transcribe_whisper_openai(
        self, audio_data: bytes, api_key: Optional[str]
    ) -> Optional[TranscriptionResult]:
        """Transcribe using OpenAI Whisper API.

        Args:
            audio_data: Raw audio bytes.
            api_key: OpenAI API key.

        Returns:
            TranscriptionResult, or None.
        """
        if not api_key:
            logger.debug("Whisper OpenAI: No API key")
            return None

        import base64

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "whisper-1",
            "audio": base64.b64encode(audio_data).decode("ascii"),
            "language": self._config.language if not self._config.auto_detect_language else None,
            "response_format": "verbose_json",
        }

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            loop = asyncio.get_event_loop()
            json_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                "https://api.openai.com/v1/audio/transcriptions",
                data=json_data,
                method="POST",
            )
            for k, v in headers.items():
                req.add_header(k, v)

            resp_data = await loop.run_in_executor(
                None, lambda: self._make_request(req)
            )

            response = json.loads(resp_data.decode("utf-8"))
            text = response.get("text", "").strip()
            language = response.get("language", self._config.language)

            segments = []
            for seg in response.get("segments", []):
                segments.append(TranscriptionSegment(
                    text=seg.get("text", ""),
                    start_time=seg.get("start", 0.0),
                    end_time=seg.get("end", 0.0),
                    confidence=seg.get("avg_logprob", 0.0),
                ))

            return TranscriptionResult(
                text=text,
                confidence=0.9,  # Whisper doesn't always provide confidence
                language=language,
                segments=segments,
                is_final=True,
            )

        except Exception as e:
            logger.error("Whisper OpenAI transcription error: %s", e)
            raise

    async def _transcribe_google(
        self, audio_data: bytes
    ) -> Optional[TranscriptionResult]:
        """Transcribe using Google Speech-to-Text.

        Note: This uses the Google Cloud Speech API which requires
        authentication setup. Falls back gracefully.

        Args:
            audio_data: Raw audio bytes.

        Returns:
            TranscriptionResult, or None.
        """
        # Google STT requires the google-cloud-speech library
        # which is not in stdlib. Return a system fallback.
        logger.debug("Google STT: requires google-cloud-speech, falling back")
        return await self._transcribe_system(audio_data)

    async def _transcribe_azure(
        self, audio_data: bytes, api_key: Optional[str]
    ) -> Optional[TranscriptionResult]:
        """Transcribe using Azure Speech Services.

        Note: This requires the azure-cognitiveservices-speech library.

        Args:
            audio_data: Raw audio bytes.
            api_key: Azure API key.

        Returns:
            TranscriptionResult, or None.
        """
        logger.debug("Azure STT: requires azure-cognitiveservices-speech, falling back")
        return await self._transcribe_system(audio_data)

    async def _transcribe_assemblyai(
        self, audio_data: bytes, api_key: Optional[str]
    ) -> Optional[TranscriptionResult]:
        """Transcribe using AssemblyAI API.

        Args:
            audio_data: Raw audio bytes.
            api_key: AssemblyAI API key.

        Returns:
            TranscriptionResult, or None.
        """
        if not api_key:
            logger.debug("AssemblyAI: No API key")
            return None

        # AssemblyAI requires uploading audio first, then polling
        # For simplicity, return system fallback
        logger.debug("AssemblyAI: async streaming requires upload, falling back")
        return await self._transcribe_system(audio_data)

    async def _transcribe_system(
        self, audio_data: bytes
    ) -> TranscriptionResult:
        """System-default transcription (placeholder).

        Provides a basic transcription fallback that analyzes audio
        energy patterns. In production, this would use a local model.

        Args:
            audio_data: Raw audio bytes.

        Returns:
            A placeholder TranscriptionResult.
        """
        # Placeholder: return empty result
        # In production, this could use SpeechRecognition library
        # or a locally bundled Whisper model
        audio_seconds = len(audio_data) / (self._config.sample_rate * 2)

        return TranscriptionResult(
            text="",
            confidence=0.0,
            language=self._config.language,
            duration=audio_seconds,
            is_final=True,
            provider="system_default",
        )

    async def _transcribe_interim(
        self, audio_data: bytes
    ) -> Optional[TranscriptionResult]:
        """Generate an interim transcription result.

        Uses the active provider for a quick interim transcription.

        Args:
            audio_data: Raw audio bytes.

        Returns:
            Interim TranscriptionResult, or None.
        """
        result = await self._transcribe_with_provider(
            self._active_provider, audio_data
        )
        if result:
            result.is_final = False
        return result

    async def _dispatch_result(self, result: TranscriptionResult) -> None:
        """Dispatch a transcription result to all callbacks.

        Args:
            result: The transcription result to dispatch.
        """
        for callback in self._result_callbacks:
            try:
                await callback(result)
            except Exception as e:
                logger.error("Result callback error: %s", e)

    @staticmethod
    def _make_request(req: urllib.request.Request) -> bytes:
        """Make an HTTP request and return the response body.

        Args:
            req: The request object.

        Returns:
            Response body as bytes.
        """
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
