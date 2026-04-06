"""
Voice call channel adapter for the Atlas Gateway.

Supports voice call handling with speech-to-text transcription and
text-to-speech synthesis. Integrates with telephony backends, SIP
services, or VoIP platforms for bidirectional voice conversations.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.voice_call import VoiceCallAdapter

    config = PlatformConfig(
        name="voice_call",
        token="VOICE_API_KEY",
        enabled=True,
        extra={
            "provider": "twilio",
            "twilio_account_sid": "AC_xxx",
            "twilio_auth_token": "auth_token",
            "phone_number": "+1234567890",
            "tts_engine": "default",
            "stt_engine": "default",
        },
    )
    adapter = VoiceCallAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.voice_call")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class CallState(str, Enum):
    """Voice call lifecycle states."""

    IDLE = "idle"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    TRANSFERRING = "transferring"
    ENDED = "ended"
    FAILED = "failed"


class TTSProvider(str, Enum):
    """Supported text-to-speech providers."""

    DEFAULT = "default"
    AWS_POLLY = "aws_polly"
    GOOGLE_TTS = "google_tts"
    AZURE_TTS = "azure_tts"
    ELEVENLABS = "elevenlabs"


class STTProvider(str, Enum):
    """Supported speech-to-text providers."""

    DEFAULT = "default"
    WHISPER = "whisper"
    GOOGLE_STT = "google_stt"
    AWS_TRANSCRIBE = "aws_transcribe"
    AZURE_STT = "azure_stt"
    DEEPGRAM = "deepgram"


@dataclass
class VoiceCallConfig:
    """Configuration for the voice call adapter."""

    provider: str = "generic"
    account_sid: str = ""
    auth_token: str = ""
    phone_number: str = ""
    webhook_base_url: str = ""
    timeout: int = 30
    max_call_duration: int = 3600  # seconds
    tts_provider: str = "default"
    tts_voice: str = "default"
    tts_language: str = "en-US"
    stt_provider: str = "default"
    stt_language: str = "en-US"
    silence_timeout: float = 10.0  # seconds
    sample_rate: int = 16000
    recording_enabled: bool = False


@dataclass
class ActiveCall:
    """Represents an active voice call."""

    call_id: str
    caller_number: str
    callee_number: str
    state: CallState = CallState.IDLE
    started_at: float = 0.0
    duration: float = 0.0
    direction: str = "inbound"  # inbound or outbound
    metadata: Dict[str, Any] = field(default_factory=dict)
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    audio_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


class VoiceCallAdapter:
    """
    Voice call channel adapter.

    Provides an abstraction layer for handling voice calls through
    various telephony/VoIP providers. Supports speech-to-text
    transcription of caller speech, text-to-speech synthesis for
    responses, and full call lifecycle management.

    The adapter can work with webhook-based providers (Twilio, Vonage)
    or SIP-based backends.

    Parameters
    ----------
    config:
        Platform configuration. Provider-specific settings are in
        ``config.extra``.
    """

    MAX_TEXT_LENGTH = 2000  # Max TTS input length per request

    def __init__(self, config: Any):
        self._config = config
        extra = getattr(config, "extra", {}) or {}

        self._voice_config = VoiceCallConfig(
            provider=extra.get("provider", "generic"),
            account_sid=extra.get("account_sid") or os.environ.get("TWILIO_ACCOUNT_SID", ""),
            auth_token=extra.get("auth_token") or os.environ.get("TWILIO_AUTH_TOKEN", ""),
            phone_number=extra.get("phone_number") or os.environ.get("VOICE_PHONE_NUMBER", ""),
            webhook_base_url=extra.get("webhook_base_url", ""),
            timeout=config.timeout or 30,
            max_call_duration=extra.get("max_call_duration", 3600),
            tts_provider=extra.get("tts_engine", "default"),
            tts_voice=extra.get("tts_voice", "default"),
            tts_language=extra.get("tts_language", "en-US"),
            stt_provider=extra.get("stt_engine", "default"),
            stt_language=extra.get("stt_language", "en-US"),
            silence_timeout=extra.get("silence_timeout", 10.0),
            recording_enabled=extra.get("recording_enabled", False),
        )

        self._connected = False
        self._session: Optional[Any] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._active_calls: Dict[str, ActiveCall] = {}
        self._call_listeners: Dict[str, asyncio.Task] = {}
        self._tts_fn: Optional[Callable] = None
        self._stt_fn: Optional[Callable] = None

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize the voice call adapter."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._voice_config.timeout),
        )

        provider = self._voice_config.provider
        logger.info(
            "Voice call adapter initializing (provider=%s, number=%s)",
            provider, self._voice_config.phone_number,
        )

        # Provider-specific initialization
        if provider == "twilio":
            if not self._voice_config.account_sid:
                logger.warning("Twilio account SID not configured")
        elif provider == "generic":
            logger.info("Generic voice provider: expecting external webhook events")

        self._connected = True

    async def disconnect(self) -> None:
        """Shut down the voice call adapter and end all active calls."""
        self._connected = False

        # End all active calls
        for call_id, call in list(self._active_calls.items()):
            await self.end_call(call_id)

        # Cancel listener tasks
        for task in self._call_listeners.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._call_listeners.clear()

        if self._session:
            await self._session.close()
            self._session = None

    async def is_connected(self) -> bool:
        """Check if the adapter is connected."""
        return self._connected and self._session is not None

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a text message to an active call (spoken via TTS).

        Parameters
        ----------
        chat_id:
            Call ID of the active call.
        text:
            Text to be synthesized and spoken to the caller.
        """
        call = self._active_calls.get(chat_id)
        if not call:
            logger.error("No active call found: %s", chat_id)
            return None

        if call.state != CallState.IN_PROGRESS:
            logger.warning(
                "Cannot send message to call %s in state %s",
                chat_id, call.state.value,
            )
            return None

        text = self._truncate(text)

        # Synthesize speech
        audio_data = await self._synthesize_speech(text)

        if audio_data:
            await call.audio_queue.put(audio_data)
            logger.info(
                "Voice message queued for call %s (%d chars)",
                chat_id, len(text),
            )
        else:
            logger.error("TTS synthesis failed for call %s", chat_id)

        return f"voice_{int(time.time())}"

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a pre-recorded audio file to an active call.

        Parameters
        ----------
        chat_id:
            Call ID.
        file_path:
            Path to an audio file (WAV, MP3, etc.).
        """
        call = self._active_calls.get(chat_id)
        if not call or call.state != CallState.IN_PROGRESS:
            return None

        if not os.path.exists(file_path):
            logger.error("Audio file not found: %s", file_path)
            return None

        try:
            with open(file_path, "rb") as f:
                audio_data = f.read()
            await call.audio_queue.put(audio_data)
            logger.info("Audio file queued for call %s: %s", chat_id, file_path)
            return f"audio_{int(time.time())}"
        except Exception as e:
            logger.error("Failed to read audio file: %s", e)
            return None

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for transcribed messages from active calls."""
        messages: List[IncomingMessage] = []
        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    # ── Call Management ───────────────────────────────────────────────────

    async def start_call(
        self, to_number: str, from_number: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Initiate an outbound voice call.

        Parameters
        ----------
        to_number:
            Phone number to call.
        from_number:
            Caller ID number. Defaults to configured number.
        """
        call_id = str(uuid.uuid4())[:12]
        from_number = from_number or self._voice_config.phone_number

        call = ActiveCall(
            call_id=call_id,
            caller_number=from_number,
            callee_number=to_number,
            state=CallState.RINGING,
            started_at=time.time(),
            direction="outbound",
            metadata=kwargs.get("metadata", {}),
        )

        self._active_calls[call_id] = call
        logger.info("Outbound call initiated: %s -> %s (id=%s)", from_number, to_number, call_id)

        # Provider-specific call initiation
        if self._voice_config.provider == "twilio":
            success = await self._twilio_initiate_call(to_number, from_number, call_id)
            if not success:
                call.state = CallState.FAILED
                return call_id
        else:
            # Generic: transition to in_progress
            call.state = CallState.IN_PROGRESS

        # Start call listener
        listener = asyncio.create_task(self._call_listener(call_id))
        self._call_listeners[call_id] = listener

        return call_id

    async def handle_incoming_call(
        self, call_id: str, caller_number: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Register an incoming call from a webhook event.

        Parameters
        ----------
        call_id:
            Unique call identifier from the provider.
        caller_number:
            Caller's phone number.
        metadata:
            Additional call metadata from the webhook.
        """
        call = ActiveCall(
            call_id=call_id,
            caller_number=caller_number,
            callee_number=self._voice_config.phone_number,
            state=CallState.IN_PROGRESS,
            started_at=time.time(),
            direction="inbound",
            metadata=metadata or {},
        )

        self._active_calls[call_id] = call
        logger.info("Incoming call registered: %s -> %s (id=%s)", caller_number, call.callee_number, call_id)

        # Start listener
        listener = asyncio.create_task(self._call_listener(call_id))
        self._call_listeners[call_id] = listener

        return call_id

    async def end_call(self, call_id: str, reason: str = "") -> bool:
        """End an active call."""
        call = self._active_calls.get(call_id)
        if not call:
            return False

        call.state = CallState.ENDED
        call.duration = time.time() - call.started_at

        logger.info(
            "Call ended: id=%s, duration=%.1fs, reason=%s",
            call_id, call.duration, reason or "normal",
        )

        # Cancel listener
        task = self._call_listeners.pop(call_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        return True

    def get_active_calls(self) -> List[Dict[str, Any]]:
        """Get information about all active calls."""
        return [
            {
                "call_id": c.call_id,
                "caller": c.caller_number,
                "callee": c.callee_number,
                "state": c.state.value,
                "direction": c.direction,
                "duration": time.time() - c.started_at,
            }
            for c in self._active_calls.values()
            if c.state in (CallState.RINGING, CallState.IN_PROGRESS)
        ]

    # ── Transcription Handling ────────────────────────────────────────────

    async def handle_transcription(
        self, call_id: str, text: str, confidence: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Process a speech transcription and enqueue it as a message.

        Called by webhook handlers or STT engines when speech is
        recognized during a call.
        """
        call = self._active_calls.get(call_id)
        if not call:
            return

        call.transcript.append({
            "text": text,
            "confidence": confidence,
            "timestamp": time.time(),
            "direction": "caller",
        })

        metadata: Dict[str, Any] = {
            "call_id": call_id,
            "confidence": confidence,
            "call_direction": call.direction,
            "caller_number": call.caller_number,
            "callee_number": call.callee_number,
            "transcript_index": len(call.transcript),
        }

        msg = IncomingMessage(
            platform="voice_call",
            chat_id=call_id,
            user_id=call.caller_number,
            text=text,
            message_id=f"transcript_{int(time.time())}_{len(call.transcript)}",
            username=call.caller_number,
            metadata=metadata,
        )
        await self._message_queue.put(msg)

    async def handle_dtmf(
        self, call_id: str, digit: str,
    ) -> None:
        """Process a DTMF (keypad press) event during a call."""
        call = self._active_calls.get(call_id)
        if not call:
            return

        msg = IncomingMessage(
            platform="voice_call",
            chat_id=call_id,
            user_id=call.caller_number,
            text=f"[DTMF: {digit}]",
            message_id=f"dtmf_{int(time.time())}",
            username=call.caller_number,
            metadata={
                "event_type": "dtmf",
                "digit": digit,
                "call_direction": call.direction,
            },
        )
        await self._message_queue.put(msg)

    # ── Custom Engine Registration ────────────────────────────────────────

    def register_tts_engine(
        self, fn: Callable[[str, str, str], Any],
    ) -> None:
        """
        Register a custom TTS engine function.

        The function should accept (text, voice, language) and return
        audio bytes or an async coroutine that resolves to audio bytes.
        """
        self._tts_fn = fn
        logger.info("Custom TTS engine registered")

    def register_stt_engine(
        self, fn: Callable[[bytes, str], Any],
    ) -> None:
        """
        Register a custom STT engine function.

        The function should accept (audio_bytes, language) and return
        a transcription string or an async coroutine.
        """
        self._stt_fn = fn
        logger.info("Custom STT engine registered")

    # ── Twilio-Specific ───────────────────────────────────────────────────

    def parse_twilio_webhook(
        self, form_data: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a Twilio voice webhook into structured event data.

        Handles both the initial call webhook and speech transcription
        callbacks.
        """
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")

        event: Dict[str, Any] = {
            "call_id": call_sid,
            "status": call_status,
            "caller": form_data.get("From", ""),
            "callee": form_data.get("To", ""),
            "direction": form_data.get("Direction", "inbound"),
        }

        # Speech result
        speech_result = form_data.get("SpeechResult", "")
        if speech_result:
            event["transcription"] = speech_result
            confidence = form_data.get("Confidence", "1.0")
            event["confidence"] = float(confidence)

        # DTMF
        digits = form_data.get("Digits", "")
        if digits:
            event["dtmf"] = digits

        return event

    async def _twilio_initiate_call(
        self, to: str, from_number: str, call_id: str,
    ) -> bool:
        """Initiate a call via the Twilio REST API."""
        if not self._session or not self._voice_config.account_sid:
            return False

        account_sid = self._voice_config.account_sid
        auth_token = self._voice_config.auth_token
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json"
        auth = aiohttp.BasicAuth(account_sid, auth_token)

        webhook_url = self._voice_config.webhook_base_url
        twiml_url = f"{webhook_url}/voice/twiml" if webhook_url else ""

        data = {
            "To": to,
            "From": from_number,
            "Url": twiml_url,
            "Record": "true" if self._voice_config.recording_enabled else "false",
        }

        try:
            async with self._session.post(url, data=data, auth=auth) as resp:
                if resp.status in (200, 201):
                    result = await resp.json()
                    logger.info(
                        "Twilio call initiated: sid=%s, status=%s",
                        result.get("sid"), result.get("status"),
                    )
                    return True
                error = await resp.text()
                logger.error("Twilio call error: %s", error[:200])
                return False
        except Exception as e:
            logger.error("Twilio call initiation failed: %s", e)
            return False

    # ── Internal ──────────────────────────────────────────────────────────

    async def _call_listener(self, call_id: str) -> None:
        """Monitor an active call for events and timeouts."""
        call = self._active_calls.get(call_id)
        if not call:
            return

        last_activity = time.time()

        while self._connected and call.state == CallState.IN_PROGRESS:
            # Check max duration
            duration = time.time() - call.started_at
            if duration >= self._voice_config.max_call_duration:
                logger.info("Call %s exceeded max duration, ending", call_id)
                await self.end_call(call_id, "max_duration")
                return

            # Check for pending audio to play
            try:
                audio = await asyncio.wait_for(
                    call.audio_queue.get(), timeout=5.0,
                )
                last_activity = time.time()
                # In a real implementation, the audio would be streamed
                # to the call via the provider's API or SIP connection.
                logger.debug("Audio chunk ready for call %s (%d bytes)", call_id, len(audio) if isinstance(audio, bytes) else 0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                return

            # Silence timeout check
            silence = time.time() - last_activity
            if silence > self._voice_config.silence_timeout:
                logger.debug("Call %s silence detected (%.1fs)", call_id, silence)

            await asyncio.sleep(0.5)

    async def _synthesize_speech(self, text: str) -> Optional[bytes]:
        """
        Synthesize speech from text using the configured TTS provider.

        Returns raw audio bytes or None if synthesis fails.
        """
        # Check for custom engine
        if self._tts_fn:
            try:
                result = self._tts_fn(
                    text,
                    self._voice_config.tts_voice,
                    self._voice_config.tts_language,
                )
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except Exception as e:
                logger.error("Custom TTS error: %s", e)
                return None

        # Default: placeholder — returns silence audio
        logger.warning(
            "No TTS engine configured. Message will not be spoken: %s",
            text[:100],
        )
        return b""

    async def _transcribe_audio(
        self, audio_bytes: bytes,
    ) -> Optional[str]:
        """Transcribe audio using the configured STT provider."""
        if self._stt_fn:
            try:
                result = self._stt_fn(audio_bytes, self._voice_config.stt_language)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except Exception as e:
                logger.error("Custom STT error: %s", e)
                return None

        logger.warning("No STT engine configured — transcription unavailable")
        return None

    def _truncate(self, text: str) -> str:
        """Truncate text for TTS processing."""
        if len(text) <= self.MAX_TEXT_LENGTH:
            return text
        # Split at sentence boundary
        truncated = text[:self.MAX_TEXT_LENGTH]
        last_period = truncated.rfind(".")
        if last_period > self.MAX_TEXT_LENGTH // 2:
            return truncated[:last_period + 1]
        return truncated[:self.MAX_TEXT_LENGTH - 3] + "..."
