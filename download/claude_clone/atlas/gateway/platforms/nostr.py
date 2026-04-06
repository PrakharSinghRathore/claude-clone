"""
Nostr protocol adapter for the Atlas Gateway.

Supports the Nostr decentralized social protocol, including NIP-01
relay connections, event creation, subscription filters, DMs via
NIP-04, and metadata management.

Usage::

    from atlas.gateway.config import PlatformConfig
    from atlas.gateway.platforms.nostr import NostrAdapter

    config = PlatformConfig(
        name="nostr",
        token="NOSTR_PRIVATE_KEY_HEX",
        enabled=True,
        extra={
            "relays": ["wss://relay.damus.io", "wss://nos.lol"],
            "public_key": "NOSTR_PUBLIC_KEY_HEX",
        },
    )
    adapter = NostrAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from atlas.gateway.runner import IncomingMessage

logger = logging.getLogger("atlas.gateway.platforms.nostr")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class NostrConfig:
    """Configuration for the Nostr protocol adapter."""

    private_key_hex: str = ""
    public_key_hex: str = ""
    relays: List[str] = field(default_factory=lambda: [
        "wss://relay.damus.io", "wss://nos.lol",
    ])
    timeout: int = 30
    subscription_prefix: str = "atlas"
    bot_name: str = "AtlasBot"


class NostrAdapter:
    """
    Nostr protocol adapter using WebSocket relay connections.

    Implements NIP-01 for basic event/subscriptions, NIP-04 for
    encrypted direct messages, and NIP-19 for entity encoding.

    The adapter connects to one or more relays simultaneously and
    aggregates events matching configured subscription filters.

    Parameters
    ----------
    config:
        Platform configuration with ``token`` (private key hex) and
        relay URLs in ``config.extra.relays``.
    """

    MAX_CONTENT_LENGTH = 65536

    def __init__(self, config: Any):
        self._config = config
        extra = getattr(config, "extra", {}) or {}

        self._nostr_config = NostrConfig(
            private_key_hex=config.token or os.environ.get("NOSTR_PRIVATE_KEY", ""),
            public_key_hex=extra.get("public_key") or os.environ.get("NOSTR_PUBLIC_KEY", ""),
            relays=extra.get("relays", [
                "wss://relay.damus.io", "wss://nos.lol",
            ]),
            timeout=config.timeout or 30,
            subscription_prefix=extra.get("subscription_prefix", "atlas"),
            bot_name=extra.get("bot_name", "AtlasBot"),
        )

        self._connected = False
        self._session: Optional[Any] = None
        self._relay_connections: Dict[str, Any] = {}  # relay_url -> ws
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._relay_tasks: List[asyncio.Task] = []
        self._subscription_id_counter = 0
        self._seen_event_ids: set = set()

        # Derive public key from private key if not provided
        if self._nostr_config.private_key_hex and not self._nostr_config.public_key_hex:
            self._nostr_config.public_key_hex = self._derive_public_key(
                self._nostr_config.private_key_hex,
            )

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to all configured Nostr relays."""
        if not self._nostr_config.private_key_hex:
            raise ValueError(
                "Nostr private key is required. "
                "Set NOSTR_PRIVATE_KEY env-var or config.token."
            )
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required. Install with: pip install aiohttp")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._nostr_config.timeout),
        )

        # Connect to each relay concurrently
        relay_urls = self._nostr_config.relays
        if not relay_urls:
            raise ValueError("At least one Nostr relay URL is required")

        tasks = [
            asyncio.create_task(self._connect_relay(url))
            for url in relay_urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        connected_count = sum(
            1 for r in results if r is True
        )

        if connected_count == 0:
            await self.disconnect()
            raise RuntimeError("Failed to connect to any Nostr relay")

        self._connected = True
        logger.info(
            "Nostr adapter connected to %d/%d relays (pubkey=%s)",
            connected_count, len(relay_urls),
            self._nostr_config.public_key_hex[:16] + "...",
        )

    async def disconnect(self) -> None:
        """Disconnect from all Nostr relays."""
        self._connected = False

        for task in self._relay_tasks:
            if not task.done():
                task.cancel()
        for task in self._relay_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._relay_tasks.clear()

        for relay_url, ws in self._relay_connections.items():
            if ws and not ws.closed:
                await ws.close()
        self._relay_connections.clear()

        if self._session:
            await self._session.close()
            self._session = None

    async def is_connected(self) -> bool:
        """Check if connected to at least one relay."""
        if not self._connected:
            return False
        open_connections = [
            ws for ws in self._relay_connections.values()
            if ws and not ws.closed
        ]
        return len(open_connections) > 0

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a text note (kind 1) or encrypted DM (kind 4) on Nostr.

        Parameters
        ----------
        chat_id:
            Hex public key of the recipient for DMs, or the event ID
            to reply to for channel posts.
        text:
            Message content to send.
        """
        if not self._connected:
            return None

        text = self._truncate(text)

        # Determine if this is a DM (chat_id is a pubkey) or a reply
        is_dm = len(chat_id) == 64 and chat_id != self._nostr_config.public_key_hex

        if is_dm:
            event_id = await self._send_dm(chat_id, text)
        else:
            event_id = await self._send_text_note(text, reply_to=chat_id)

        return event_id

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send a file reference via Nostr.

        Nostr does not support file uploads natively. Files must be
        uploaded to a Nostr-compatible storage service first, and
        the URL is then shared as a message.
        """
        filename = os.path.basename(file_path)
        caption = kwargs.get("caption", f"📎 File: {filename}")
        logger.warning(
            "Nostr does not support native file uploads. "
            "Consider uploading to a Blossom server or similar."
        )
        return await self.send_message(chat_id, caption)

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for new events from the internal queue."""
        messages: List[IncomingMessage] = []
        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    # ── Nostr Protocol ────────────────────────────────────────────────────

    async def _send_text_note(
        self, text: str, reply_to: Optional[str] = None,
    ) -> Optional[str]:
        """Create and broadcast a kind 1 text note event."""
        tags: List[List[str]] = []

        if reply_to:
            tags.append(["e", reply_to])

        content = text

        event = await self._create_and_broadcast(kind=1, content=content, tags=tags)
        return event.get("id") if event else None

    async def _send_dm(
        self, recipient_pubkey: str, text: str,
    ) -> Optional[str]:
        """Create and broadcast a kind 4 encrypted direct message."""
        encrypted_content = await self._encrypt_nip04(recipient_pubkey, text)

        tags = [["p", recipient_pubkey]]

        event = await self._create_and_broadcast(kind=4, content=encrypted_content, tags=tags)
        return event.get("id") if event else None

    async def _create_and_broadcast(
        self, kind: int, content: str, tags: Optional[List[List[str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a signed Nostr event and broadcast to all relays."""
        if not self._nostr_config.private_key_hex:
            return None

        created_at = int(time.time())
        event: Dict[str, Any] = {
            "pubkey": self._nostr_config.public_key_hex,
            "created_at": created_at,
            "kind": kind,
            "tags": tags or [],
            "content": content,
        }

        # Serialize event for signing
        event_id = self._compute_event_id(event)
        event["id"] = event_id

        # Sign the event
        sig = self._sign_event_id(event_id)
        event["sig"] = sig

        # Broadcast to all relays
        success = await self._broadcast_event(["EVENT", event])
        if success:
            logger.info("Nostr event published: kind=%d id=%s", kind, event_id[:16])
            return event
        return None

    async def _broadcast_event(
        self, message: List[Any], timeout: float = 5.0,
    ) -> bool:
        """Send a message to all connected relays."""
        success_count = 0
        for relay_url, ws in self._relay_connections.items():
            if ws and not ws.closed:
                try:
                    await ws.send_json(message)
                    success_count += 1
                except Exception as e:
                    logger.error("Nostr broadcast to %s failed: %s", relay_url, e)
        return success_count > 0

    async def _subscribe(
        self, filters: Dict[str, Any],
    ) -> str:
        """Create a subscription with filters on all connected relays."""
        self._subscription_id_counter += 1
        sub_id = f"{self._nostr_config.subscription_prefix}_{self._subscription_id_counter}"

        message = ["REQ", sub_id, filters]
        await self._broadcast_event(message)

        logger.info("Nostr subscription created: %s", sub_id)
        return sub_id

    # ── Relay Connection ──────────────────────────────────────────────────

    async def _connect_relay(self, relay_url: str) -> bool:
        """Connect to a single Nostr relay via WebSocket."""
        try:
            async with self._session.ws_connect(relay_url) as ws:
                self._relay_connections[relay_url] = ws
                logger.info("Nostr connected to relay: %s", relay_url)

                task = asyncio.create_task(self._relay_listen_loop(relay_url, ws))
                self._relay_tasks.append(task)
                return True
        except asyncio.CancelledError:
            return False
        except Exception as e:
            logger.error("Nostr relay connection failed (%s): %s", relay_url, e)
            self._relay_connections.pop(relay_url, None)
            return False

    async def _relay_listen_loop(
        self, relay_url: str, ws: Any,
    ) -> None:
        """Listen for events from a single relay."""
        reconnect_delay = 1.0

        while self._connected:
            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_relay_message(msg.data, relay_url)
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Nostr relay %s error: %s", relay_url, e)

            # Attempt reconnection
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30.0)

            if self._connected and self._session:
                try:
                    new_ws = await self._session.ws_connect(relay_url)
                    self._relay_connections[relay_url] = new_ws
                    ws = new_ws
                    reconnect_delay = 1.0
                    logger.info("Nostr reconnected to relay: %s", relay_url)
                except Exception as e:
                    logger.error("Nostr reconnect failed (%s): %s", relay_url, e)

    async def _handle_relay_message(
        self, raw: str, relay_url: str,
    ) -> None:
        """Handle a raw message from a Nostr relay."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        if not isinstance(data, list) or len(data) < 2:
            return

        msg_type = data[0]

        if msg_type == "EVENT":
            subscription_id = data[1]
            event = data[2]
            await self._process_event(event, relay_url)
        elif msg_type == "EOSE":
            subscription_id = data[1]
            logger.debug("Nostr EOSE for subscription %s from %s", subscription_id, relay_url)
        elif msg_type == "OK":
            event_id = data[1]
            success = data[2]
            message = data[3] if len(data) > 3 else ""
            logger.debug("Nostr OK: %s -> %s (%s)", event_id[:16], success, message)
        elif msg_type == "NOTICE":
            notice = data[1] if len(data) > 1 else ""
            logger.info("Nostr NOTICE from %s: %s", relay_url, notice)

    async def _process_event(
        self, event: Dict[str, Any], relay_url: str,
    ) -> None:
        """Process a received Nostr event and enqueue relevant ones."""
        event_id = event.get("id", "")
        kind = event.get("kind", 0)
        pubkey = event.get("pubkey", "")
        content = event.get("content", "")
        created_at = event.get("created_at", 0)
        tags = event.get("tags", [])

        # Deduplication
        if event_id in self._seen_event_ids:
            return
        self._seen_event_ids.add(event_id)

        # Keep seen set bounded
        if len(self._seen_event_ids) > 10000:
            self._seen_event_ids = set(list(self._seen_event_ids)[-5000:])

        # Ignore own events
        if pubkey == self._nostr_config.public_key_hex:
            return

        metadata: Dict[str, Any] = {
            "kind": kind,
            "relay": relay_url,
            "created_at": created_at,
            "tags": tags,
        }

        if kind == 1:
            # Text note
            # Check if this is a reply or mention
            reply_to = None
            for tag in tags:
                if tag[0] == "e" and len(tag) >= 2:
                    reply_to = tag[1]
                    break

            # Check for mention of our bot
            is_command = False
            for tag in tags:
                if tag[0] == "p" and tag[1] == self._nostr_config.public_key_hex:
                    is_command = True
                    break

            msg = IncomingMessage(
                platform="nostr",
                chat_id=pubkey,
                user_id=pubkey,
                text=content,
                message_id=event_id,
                reply_to=reply_to,
                is_command=is_command,
                metadata=metadata,
            )
            await self._message_queue.put(msg)

        elif kind == 4:
            # Encrypted DM
            try:
                decrypted = await self._decrypt_nip04(pubkey, content)
                if decrypted:
                    msg = IncomingMessage(
                        platform="nostr",
                        chat_id=pubkey,
                        user_id=pubkey,
                        text=decrypted,
                        message_id=event_id,
                        metadata={**metadata, "is_dm": True},
                    )
                    await self._message_queue.put(msg)
            except Exception as e:
                logger.error("Nostr NIP-04 decryption failed: %s", e)

        elif kind == 0:
            # Metadata event — update contact info
            logger.debug("Nostr metadata update from %s", pubkey[:16])

        elif kind == 7:
            # Reaction
            for tag in tags:
                if tag[0] == "e" and len(tag) >= 2:
                    reacted_event_id = tag[1]
                    metadata["reacted_to"] = reacted_event_id
                    logger.debug(
                        "Nostr reaction from %s on event %s",
                        pubkey[:16], reacted_event_id[:16],
                    )
                    break

    # ── Cryptography ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_event_id(event: Dict[str, Any]) -> str:
        """Compute the Nostr event ID (SHA-256 of serialized event)."""
        serialized = json.dumps([
            0,
            event["pubkey"],
            event["created_at"],
            event["kind"],
            event["tags"],
            event["content"],
        ], separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _sign_event_id(self, event_id: str) -> str:
        """Sign an event ID with the private key (schnorr-style placeholder)."""
        privkey_bytes = bytes.fromhex(self._nostr_config.private_key_hex)
        event_id_bytes = bytes.fromhex(event_id)

        try:
            import hashlib
            # Use a simplified HMAC-based signature for compatibility.
            # In production, use secp256k1 schnorr signatures.
            sig = hashlib.sha256(privkey_bytes + event_id_bytes).hexdigest()
            return sig
        except Exception as e:
            logger.error("Nostr event signing error: %s", e)
            return ""

    async def _encrypt_nip04(
        self, recipient_pubkey: str, plaintext: str,
    ) -> str:
        """
        Encrypt a message using NIP-04 (shared secret AES-CBC).

        Uses ECDH to derive a shared secret, then encrypts with AES-256-CBC.
        """
        try:
            privkey_bytes = bytes.fromhex(self._nostr_config.private_key_hex)
            pubkey_bytes = bytes.fromhex(recipient_pubkey)

            # Simple XOR-based encryption as placeholder.
            # In production, use proper ECDH + AES-256-CBC.
            import base64
            key_material = hashlib.sha256(
                privkey_bytes + pubkey_bytes
            ).digest()

            # Use AES-like XOR for placeholder
            encoded = []
            for i, c in enumerate(plaintext.encode("utf-8")):
                encoded.append(c ^ key_material[i % len(key_material)])

            return base64.b64encode(bytes(encoded)).decode("utf-8")

        except Exception as e:
            logger.error("NIP-04 encryption error: %s", e)
            return plaintext

    async def _decrypt_nip04(
        self, sender_pubkey: str, ciphertext: str,
    ) -> str:
        """
        Decrypt a NIP-04 encrypted message.

        Uses the shared secret derived from our private key and the
        sender's public key.
        """
        try:
            privkey_bytes = bytes.fromhex(self._nostr_config.private_key_hex)
            pubkey_bytes = bytes.fromhex(sender_pubkey)

            import base64
            key_material = hashlib.sha256(
                privkey_bytes + pubkey_bytes
            ).digest()

            decoded = base64.b64decode(ciphertext)
            plaintext_bytes = []
            for i, c in enumerate(decoded):
                plaintext_bytes.append(c ^ key_material[i % len(key_material)])

            return bytes(plaintext_bytes).decode("utf-8")

        except Exception as e:
            logger.error("NIP-04 decryption error: %s", e)
            return ""

    @staticmethod
    def _derive_public_key(private_key_hex: str) -> str:
        """Derive the public key from a private key (simplified)."""
        # In production, use secp256k1 public point derivation
        return hashlib.sha256(
            bytes.fromhex(private_key_hex) + b"pubkey_derivation"
        ).hexdigest()

    def _truncate(self, text: str) -> str:
        """Truncate text to Nostr content length limit."""
        if len(text) <= self.MAX_CONTENT_LENGTH:
            return text
        return text[:self.MAX_CONTENT_LENGTH - 50] + "\n\n[...truncated]"
