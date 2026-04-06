"""
Device discovery for pairable devices.

Provides mDNS/Bonjour discovery simulation and QR code pairing support
for finding nearby pairable devices on the network.

Usage::

    discovery = DeviceDiscovery()
    await discovery.start_discovery()
    devices = await discovery.get_discovered()
    discovery.on_device_found(lambda d: print(f"Found: {d.device_name}"))
    await discovery.stop_discovery()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("atlas.pairing.discovery")


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DiscoveredDevice:
    """A device discovered during scanning."""

    device_id: str
    device_name: str
    device_type: str = "unknown"  # mobile, desktop, tablet, embedded, etc.
    ip_address: str = ""
    port: int = 0
    service_type: str = "_atlas-pair._tcp.local."
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: float = field(default_factory=time.time)
    signal_strength: int = 0  # 0-100, -1 if unknown
    is_pairable: bool = True
    version: str = ""
    fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "ip_address": self.ip_address,
            "port": self.port,
            "service_type": self.service_type,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
            "discovered_at": self.discovered_at,
            "age_seconds": time.time() - self.discovered_at,
            "signal_strength": self.signal_strength,
            "is_pairable": self.is_pairable,
            "version": self.version,
            "fingerprint": self.fingerprint,
        }

    def is_expired(self, ttl: float = 300.0) -> bool:
        """Check if the discovery entry has expired."""
        return time.time() - self.discovered_at > ttl


@dataclass
class QRPairingInfo:
    """Information encoded in a QR pairing code."""

    pairing_url: str
    device_id: str
    device_name: str
    service_type: str
    expires_at: float
    code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pairing_url": self.pairing_url,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "service_type": self.service_type,
            "expires_at": self.expires_at,
            "code": self.code,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Callback Types
# ──────────────────────────────────────────────────────────────────────────────

DeviceFoundCallback = Callable[[DiscoveredDevice], Any]


# ──────────────────────────────────────────────────────────────────────────────
# Device Discovery
# ──────────────────────────────────────────────────────────────────────────────

class DeviceDiscovery:
    """
    Discover pairable devices via mDNS/Bonjour simulation.

    Features:
    - Network scanning for pairable devices (mDNS/Bonjour simulation)
    - Configurable scan interval and device TTL
    - Device found callbacks
    - QR code pairing URL generation
    - Discovered device tracking with automatic expiry
    - Device filtering by type, capability, and name

    Parameters
    ----------
    scan_interval:
        Seconds between discovery scans. Default 5.
    device_ttl:
        How long a discovered device stays in the list (seconds). Default 300.
    service_types:
        List of mDNS service types to scan for.
    enable_mdns:
        Whether to enable mDNS/Bonjour simulation. Default True.
    """

    # Known mDNS service types for atlas devices
    DEFAULT_SERVICE_TYPES: List[str] = [
        "_atlas-pair._tcp.local.",
        "_atlas-device._tcp.local.",
        "_atlas-host._udp.local.",
    ]

    def __init__(
        self,
        scan_interval: float = 5.0,
        device_ttl: float = 300.0,
        service_types: Optional[List[str]] = None,
        enable_mdns: bool = True,
    ) -> None:
        self._scan_interval = scan_interval
        self._device_ttl = device_ttl
        self._service_types = service_types or list(self.DEFAULT_SERVICE_TYPES)
        self._enable_mdns = enable_mdns

        # Discovered devices: device_id -> DiscoveredDevice
        self._discovered: Dict[str, DiscoveredDevice] = {}

        # Callbacks
        self._device_found_callbacks: List[DeviceFoundCallback] = []

        # Scan state
        self._scanning = False
        self._scan_task: Optional[asyncio.Task[None]] = None
        self._scan_count = 0

        # Device name filter patterns
        self._name_filter: Optional[str] = None
        self._type_filter: Optional[str] = None

    # ── Discovery Lifecycle ──────────────────────────────────────────────

    async def start_discovery(
        self,
        name_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
    ) -> None:
        """
        Start scanning for pairable devices.

        Parameters
        ----------
        name_filter:
            Only report devices whose name contains this substring.
        type_filter:
            Only report devices of this type (mobile, desktop, etc.).
        """
        if self._scanning:
            logger.warning("Discovery is already running")
            return

        self._scanning = True
        self._name_filter = name_filter
        self._type_filter = type_filter
        self._scan_count = 0

        if self._enable_mdns:
            self._scan_task = asyncio.create_task(self._scan_loop())

        logger.info(
            "Device discovery started (interval=%.1fs, ttl=%.0fs, services=%s)",
            self._scan_interval, self._device_ttl,
            len(self._service_types),
        )

    async def stop_discovery(self) -> None:
        """Stop device discovery scanning."""
        self._scanning = False

        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "Device discovery stopped after %d scans",
            self._scan_count,
        )

    # ── Queries ──────────────────────────────────────────────────────────

    async def get_discovered(
        self,
        pairable_only: bool = True,
        type_filter: Optional[str] = None,
    ) -> List[DiscoveredDevice]:
        """
        List currently discovered devices.

        Parameters
        ----------
        pairable_only:
            Only return pairable devices. Default True.
        type_filter:
            Filter by device type. Default None.

        Returns
        -------
        List[DiscoveredDevice]
            Active discovered devices.
        """
        self._prune_expired()
        devices = list(self._discovered.values())

        if pairable_only:
            devices = [d for d in devices if d.is_pairable]
        if type_filter:
            devices = [d for d in devices if d.device_type == type_filter]
        if self._name_filter:
            devices = [
                d for d in devices
                if self._name_filter.lower() in d.device_name.lower()
            ]
        if self._type_filter:
            devices = [
                d for d in devices if d.device_type == self._type_filter
            ]

        # Sort by discovery time (most recent first)
        devices.sort(key=lambda d: d.discovered_at, reverse=True)
        return devices

    # ── Callbacks ────────────────────────────────────────────────────────

    def on_device_found(self, callback: DeviceFoundCallback) -> None:
        """
        Register a callback for when a new device is discovered.

        Parameters
        ----------
        callback:
            Function receiving a DiscoveredDevice instance.
        """
        self._device_found_callbacks.append(callback)

    # ── QR Code Pairing ──────────────────────────────────────────────────

    def generate_qr_pairing_url(
        self,
        device_id: str,
        device_name: str,
        service_type: str = "_atlas-pair._tcp.local.",
        expiry_seconds: float = 300.0,
        host: str = "localhost",
        port: int = 8080,
    ) -> QRPairingInfo:
        """
        Generate a QR-code-compatible pairing URL.

        Parameters
        ----------
        device_id:
            The device identifier.
        device_name:
            Human-readable device name.
        service_type:
            mDNS service type.
        expiry_seconds:
            How long the QR code is valid.
        host:
            Host address for the pairing endpoint.
        port:
            Port for the pairing endpoint.

        Returns
        -------
        QRPairingInfo
            The pairing information to encode in a QR code.
        """
        code = secrets.token_hex(4).upper()
        expires_at = time.time() + expiry_seconds

        # Build a simple pairing URL
        # Format: atlas-pair://{host}:{port}/pair?code={code}&device={device_id}
        pairing_url = (
            f"atlas-pair://{host}:{port}/pair"
            f"?code={code}&device={device_id}"
            f"&name={device_name}&expires={int(expires_at)}"
        )

        # Generate fingerprint
        fingerprint = hashlib.sha256(
            f"{device_id}:{code}:{expires_at}".encode()
        ).hexdigest()[:16]

        info = QRPairingInfo(
            pairing_url=pairing_url,
            device_id=device_id,
            device_name=device_name,
            service_type=service_type,
            expires_at=expires_at,
            code=code,
        )

        logger.info(
            "Generated QR pairing URL for device %s (expires in %.0fs)",
            device_id, expiry_seconds,
        )

        return info

    def parse_qr_pairing_url(self, url: str) -> Optional[QRPairingInfo]:
        """
        Parse a QR pairing URL back into structured data.

        Parameters
        ----------
        url:
            The QR pairing URL to parse.

        Returns
        -------
        Optional[QRPairingInfo]
            Parsed pairing info, or None if the URL is invalid.
        """
        try:
            if not url.startswith("atlas-pair://"):
                return None

            # Extract host:port
            rest = url[len("atlas-pair://"):]
            slash_idx = rest.find("/")
            if slash_idx < 0:
                return None

            host_port = rest[:slash_idx]
            if ":" in host_port:
                host, port_str = host_port.rsplit(":", 1)
                try:
                    int(port_str)
                except ValueError:
                    return None
            else:
                host = host_port

            # Parse query parameters
            query_string = rest[slash_idx + 1:]
            if "?" not in query_string:
                return None

            query = query_string.split("?", 1)[1]
            params: Dict[str, str] = {}
            for param in query.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    params[key] = value

            device_id = params.get("device", "")
            device_name = params.get("name", "")
            code = params.get("code", "")
            expires = float(params.get("expires", "0"))

            if not device_id or not code:
                return None

            if time.time() > expires:
                return None

            return QRPairingInfo(
                pairing_url=url,
                device_id=device_id,
                device_name=device_name,
                service_type="_atlas-pair._tcp.local.",
                expires_at=expires,
                code=code,
            )

        except Exception as e:
            logger.debug("Failed to parse QR URL: %s", e)
            return None

    # ── Manual Registration ──────────────────────────────────────────────

    async def add_discovered_device(self, device: DiscoveredDevice) -> None:
        """
        Manually add a discovered device (for testing or integration).

        Parameters
        ----------
        device:
            The discovered device entry.
        """
        existing = self._discovered.get(device.device_id)
        is_new = existing is None

        self._discovered[device.device_id] = device

        if is_new:
            logger.info(
                "New device discovered: %s (%s) at %s:%d",
                device.device_name, device.device_type,
                device.ip_address, device.port,
            )
            # Fire callbacks
            for cb in self._device_found_callbacks:
                try:
                    result = cb(device)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error("Device found callback error: %s", e)

    def remove_discovered_device(self, device_id: str) -> bool:
        """Remove a discovered device. Returns True if found."""
        if device_id in self._discovered:
            del self._discovered[device_id]
            return True
        return False

    # ── Internal ─────────────────────────────────────────────────────────

    async def _scan_loop(self) -> None:
        """Background discovery loop simulating mDNS scanning."""
        while self._scanning:
            try:
                await asyncio.sleep(self._scan_interval)
                self._scan_count += 1

                # Prune expired entries
                self._prune_expired()

                # In production, this would broadcast mDNS queries
                # and collect responses. The simulation simply maintains
                # existing entries and allows external add_discovered_device.
                logger.debug(
                    "Discovery scan #%d complete (%d devices tracked)",
                    self._scan_count, len(self._discovered),
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Discovery scan error: %s", e)

    def _prune_expired(self) -> int:
        """Remove expired discovery entries. Returns count pruned."""
        now = time.time()
        expired = [
            did for did, dev in self._discovered.items()
            if dev.is_expired(self._device_ttl)
        ]
        for did in expired:
            del self._discovered[did]
        return len(expired)

    # ── Statistics ───────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """Return discovery statistics."""
        self._prune_expired()
        devices = list(self._discovered.values())
        return {
            "scanning": self._scanning,
            "scan_count": self._scan_count,
            "devices_discovered": len(devices),
            "pairable_devices": sum(1 for d in devices if d.is_pairable),
            "by_type": self._count_by_type(devices),
            "scan_interval": self._scan_interval,
            "device_ttl": self._device_ttl,
        }

    @staticmethod
    def _count_by_type(devices: List[DiscoveredDevice]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in devices:
            counts[d.device_type] = counts.get(d.device_type, 0) + 1
        return counts
