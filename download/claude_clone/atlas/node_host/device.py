"""
Device node management for remote actions.

Implements a robust device registry with capability negotiation, heartbeat
monitoring, mDNS/Bonjour discovery simulation, and reconnection handling.
Inspired by OpenClaw's device-host architecture.

Usage::

    node = DeviceNode()
    await node.register("phone-01", [Capability.CAMERA, Capability.LOCATION])
    devices = await node.list_devices()
    result = await node.send_command("phone-01", "capture_photo", {"quality": "high"})
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger("atlas.node_host.device")


# ──────────────────────────────────────────────────────────────────────────────
# Device Capabilities
# ──────────────────────────────────────────────────────────────────────────────

class Capability(str, Enum):
    """Capabilities that a remote device can declare."""

    CAMERA = "camera"
    SCREEN = "screen"
    LOCATION = "location"
    NOTIFICATIONS = "notifications"
    CLIPBOARD = "clipboard"
    MICROPHONE = "microphone"
    SPEAKER = "speaker"
    FILES = "files"


# ──────────────────────────────────────────────────────────────────────────────
# Device Status
# ──────────────────────────────────────────────────────────────────────────────

class DeviceStatus(str, Enum):
    """Current status of a registered device."""

    ONLINE = "online"
    OFFLINE = "offline"
    RECONNECTING = "reconnecting"
    SUSPENDED = "suspended"


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DeviceInfo:
    """Complete information about a registered device."""

    device_id: str
    capabilities: List[Capability]
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: DeviceStatus = DeviceStatus.ONLINE
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_heartbeat: float = field(default_factory=time.time)
    last_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ip_address: Optional[str] = None
    hostname: Optional[str] = None
    os_name: Optional[str] = None
    os_version: Optional[str] = None
    app_version: Optional[str] = None
    reconnect_attempts: int = 0
    total_commands_sent: int = 0
    total_commands_failed: int = 0
    fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage or API responses."""
        return {
            "device_id": self.device_id,
            "capabilities": [c.value for c in self.capabilities],
            "status": self.status.value,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "last_seen": self.last_seen,
            "ip_address": self.ip_address,
            "hostname": self.hostname,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "app_version": self.app_version,
            "reconnect_attempts": self.reconnect_attempts,
            "total_commands_sent": self.total_commands_sent,
            "total_commands_failed": self.total_commands_failed,
            "metadata": self.metadata,
        }


@dataclass
class CommandResult:
    """Result of a command sent to a device."""

    success: bool
    device_id: str
    command: str
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "device_id": self.device_id,
            "command": self.command,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class DiscoveredDevice:
    """A device discovered via mDNS/Bonjour scanning."""

    device_name: str
    service_type: str
    ip_address: str
    port: int
    capabilities: List[Capability] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: float = field(default_factory=time.time)
    ttl: int = 300  # seconds before the entry expires


# ──────────────────────────────────────────────────────────────────────────────
# Type Aliases
# ──────────────────────────────────────────────────────────────────────────────

CapabilityCallback = Callable[
    [str, Capability, Dict[str, Any]], Awaitable[None]
]
DeviceEventCallback = Callable[[str, str, Dict[str, Any]], Awaitable[None]]
DeviceDiscoveryCallback = Callable[[DiscoveredDevice], Awaitable[None]]


# ──────────────────────────────────────────────────────────────────────────────
# Device Node
# ──────────────────────────────────────────────────────────────────────────────

class DeviceNode:
    """
    Central device node for managing remote device connections.

    Features:
    - Device registration and capability negotiation
    - Heartbeat monitoring with configurable intervals and timeouts
    - Device discovery via mDNS/Bonjour simulation
    - Automatic reconnection handling with exponential backoff
    - Command routing with capability validation
    - Event callbacks for capability changes and device events
    - Thread-safe in-memory device registry

    Parameters
    ----------
    heartbeat_interval:
        Seconds between expected heartbeats from devices. Default 30.
    heartbeat_timeout:
        Seconds before a device is marked offline after missed heartbeat. Default 90.
    max_reconnect_attempts:
        Maximum reconnection attempts before giving up. Default 5.
    reconnect_base_delay:
        Base delay in seconds for exponential backoff. Default 2.
    discovery_ttl:
        Default TTL for discovered devices in seconds. Default 300.
    """

    def __init__(
        self,
        heartbeat_interval: float = 30.0,
        heartbeat_timeout: float = 90.0,
        max_reconnect_attempts: int = 5,
        reconnect_base_delay: float = 2.0,
        discovery_ttl: int = 300,
    ) -> None:
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_base_delay = reconnect_base_delay
        self._discovery_ttl = discovery_ttl

        # Device registry: device_id -> DeviceInfo
        self._devices: Dict[str, DeviceInfo] = {}

        # Pending commands awaiting responses: request_id -> asyncio.Future
        self._pending_commands: Dict[str, asyncio.Future[CommandResult]] = {}

        # Capability callbacks: (device_id, capability) -> [callbacks]
        self._capability_callbacks: Dict[
            Tuple[str, Capability], List[CapabilityCallback]
        ] = defaultdict(list)

        # Generic event callbacks: device_id -> [callbacks]
        self._event_callbacks: Dict[str, List[DeviceEventCallback]] = defaultdict(list)

        # Discovery state
        self._discovered: Dict[str, DiscoveredDevice] = {}
        self._discovery_callbacks: List[DeviceDiscoveryCallback] = []
        self._discovery_running = False
        self._discovery_task: Optional[asyncio.Task[None]] = None

        # Heartbeat monitor task
        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._running = False

        # Rate limiting for commands per device
        self._command_rate: Dict[str, List[float]] = defaultdict(list)
        self._command_rate_limit: int = 60  # max commands per window
        self._command_rate_window: float = 60.0  # seconds

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the device node background tasks (heartbeat monitor, etc.)."""
        if self._running:
            logger.warning("DeviceNode is already running")
            return
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        logger.info("DeviceNode started (heartbeat_interval=%.1fs, timeout=%.1fs)",
                     self._heartbeat_interval, self._heartbeat_timeout)

    async def stop(self) -> None:
        """Stop the device node and cancel background tasks."""
        self._running = False
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        if self._discovery_task and not self._discovery_task.done():
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        # Cancel pending commands
        for future in self._pending_commands.values():
            if not future.done():
                future.cancel()
        self._pending_commands.clear()
        logger.info("DeviceNode stopped")

    # ── Device Registration ──────────────────────────────────────────────

    async def register(
        self,
        device_id: str,
        capabilities: List[Capability],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeviceInfo:
        """
        Register a new device or update an existing one.

        Parameters
        ----------
        device_id:
            Unique identifier for the device.
        capabilities:
            List of capabilities this device supports.
        metadata:
            Optional metadata dict with device details (ip, hostname, os, etc.).

        Returns
        -------
        DeviceInfo
            The registered device information.

        Raises
        ------
        ValueError
            If capabilities list is empty.
        """
        if not capabilities:
            raise ValueError("Device must declare at least one capability")

        meta = metadata or {}
        now = datetime.now(timezone.utc).isoformat()
        now_ts = time.time()

        if device_id in self._devices:
            # Update existing device (reconnection)
            device = self._devices[device_id]
            device.capabilities = list(capabilities)
            device.metadata.update(meta)
            device.status = DeviceStatus.ONLINE
            device.last_heartbeat = now_ts
            device.last_seen = now
            device.reconnect_attempts = 0
            device.ip_address = meta.get("ip_address", device.ip_address)
            device.hostname = meta.get("hostname", device.hostname)
            device.os_name = meta.get("os_name", device.os_name)
            device.os_version = meta.get("os_version", device.os_version)
            device.app_version = meta.get("app_version", device.app_version)
            logger.info("Device re-registered: %s (%d capabilities)",
                         device_id, len(capabilities))
        else:
            # New device
            fingerprint = self._generate_fingerprint(device_id, meta)
            device = DeviceInfo(
                device_id=device_id,
                capabilities=list(capabilities),
                metadata=meta,
                ip_address=meta.get("ip_address"),
                hostname=meta.get("hostname"),
                os_name=meta.get("os_name"),
                os_version=meta.get("os_version"),
                app_version=meta.get("app_version"),
                fingerprint=fingerprint,
            )
            self._devices[device_id] = device
            logger.info("New device registered: %s (%d capabilities)",
                         device_id, len(capabilities))

        # Fire event callbacks
        await self._fire_event(device_id, "registered", {
            "capabilities": [c.value for c in capabilities],
            "metadata": meta,
        })

        return device

    async def unregister(self, device_id: str) -> bool:
        """
        Unregister a device from the node.

        Parameters
        ----------
        device_id:
            The device to remove.

        Returns
        -------
        bool
            True if the device was found and removed, False otherwise.
        """
        if device_id not in self._devices:
            logger.warning("Cannot unregister unknown device: %s", device_id)
            return False

        del self._devices[device_id]
        self._capability_callbacks = {
            k: v for k, v in self._capability_callbacks.items()
            if k[0] != device_id
        }
        self._event_callbacks.pop(device_id, None)
        self._command_rate.pop(device_id, None)

        logger.info("Device unregistered: %s", device_id)
        await self._fire_event(device_id, "unregistered", {})
        return True

    # ── Device Queries ───────────────────────────────────────────────────

    async def list_devices(
        self,
        status: Optional[DeviceStatus] = None,
        capability: Optional[Capability] = None,
    ) -> List[DeviceInfo]:
        """
        List registered devices with optional filtering.

        Parameters
        ----------
        status:
            Filter by device status. None returns all.
        capability:
            Filter by required capability. None returns all.

        Returns
        -------
        List[DeviceInfo]
            Matching devices.
        """
        devices = list(self._devices.values())
        if status is not None:
            devices = [d for d in devices if d.status == status]
        if capability is not None:
            devices = [d for d in devices if capability in d.capabilities]
        return devices

    async def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """
        Get information about a specific device.

        Parameters
        ----------
        device_id:
            The device identifier.

        Returns
        -------
        Optional[DeviceInfo]
            Device information or None if not found.
        """
        return self._devices.get(device_id)

    # ── Command Sending ──────────────────────────────────────────────────

    async def send_command(
        self,
        device_id: str,
        command: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        """
        Send a command to a registered device.

        Parameters
        ----------
        device_id:
            Target device identifier.
        command:
            Command name (e.g., "capture_photo", "get_location").
        params:
            Optional command parameters.
        timeout:
            Maximum time to wait for a response in seconds.

        Returns
        -------
        CommandResult
            The command execution result.

        Raises
        ------
        LookupError
            If the device is not registered.
        ConnectionError
            If the device is offline.
        RuntimeError
            If rate limited.
        """
        # Validate device exists
        device = self._devices.get(device_id)
        if device is None:
            raise LookupError(f"Device not registered: {device_id}")

        # Check device status
        if device.status == DeviceStatus.OFFLINE:
            raise ConnectionError(f"Device is offline: {device_id}")

        # Rate limit check
        if not self._check_rate_limit(device_id):
            raise RuntimeError(f"Rate limit exceeded for device: {device_id}")

        # Validate capability for certain commands
        required_capability = self._command_capability_map(command)
        if required_capability and required_capability not in device.capabilities:
            raise PermissionError(
                f"Device {device_id} lacks required capability: {required_capability.value}"
            )

        request_id = secrets.token_hex(16)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[CommandResult] = loop.create_future()
        self._pending_commands[request_id] = future

        params = params or {}
        logger.info(
            "Sending command '%s' to device %s (request_id=%s, params=%s)",
            command, device_id, request_id[:8], list(params.keys()),
        )

        # In a real implementation this would send via WebSocket, gRPC, etc.
        # Here we simulate a successful dispatch that gets acknowledged
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            device.total_commands_sent += 1
            if not result.success:
                device.total_commands_failed += 1
            return result
        except asyncio.TimeoutError:
            device.total_commands_failed += 1
            del self._pending_commands[request_id]
            return CommandResult(
                success=False,
                device_id=device_id,
                command=command,
                error=f"Command timed out after {timeout}s",
            )

    async def acknowledge_command(
        self,
        request_id: str,
        success: bool,
        data: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> bool:
        """
        Acknowledge a pending command with its result.

        Called by device-side handlers to deliver command results back to
        the waiting caller.

        Parameters
        ----------
        request_id:
            The command request ID to acknowledge.
        success:
            Whether the command succeeded.
        data:
            Optional response data.
        error:
            Optional error message.

        Returns
        -------
        bool
            True if a pending command was found and resolved.
        """
        future = self._pending_commands.pop(request_id, None)
        if future is None or future.done():
            return False

        result = CommandResult(
            success=success,
            device_id="",
            command="",
            data=data,
            error=error,
        )
        future.set_result(result)
        return True

    # ── Capability Events ────────────────────────────────────────────────

    async def on_capability(
        self,
        device_id: str,
        capability: Capability,
        callback: CapabilityCallback,
    ) -> None:
        """
        Register a callback for a device capability event.

        Parameters
        ----------
        device_id:
            Device identifier, or "*" to listen to all devices.
        capability:
            The capability to subscribe to.
        callback:
            Async callback invoked when the capability event fires.
        """
        key = (device_id, capability)
        self._capability_callbacks[key].append(callback)
        logger.debug("Registered capability callback for %s:%s", device_id, capability.value)

    async def emit_capability_event(
        self,
        device_id: str,
        capability: Capability,
        event_data: Dict[str, Any],
    ) -> int:
        """
        Emit a capability event to registered listeners.

        Returns
        -------
        int
            Number of callbacks invoked.
        """
        count = 0

        # Wildcard listeners
        for cb in self._capability_callbacks.get(("*", capability), []):
            try:
                await cb(device_id, capability, event_data)
                count += 1
            except Exception as e:
                logger.error("Capability callback error (*:%s): %s", capability.value, e)

        # Device-specific listeners
        for cb in self._capability_callbacks.get((device_id, capability), []):
            try:
                await cb(device_id, capability, event_data)
                count += 1
            except Exception as e:
                logger.error("Capability callback error (%s:%s): %s",
                              device_id, capability.value, e)

        return count

    # ── Heartbeat ────────────────────────────────────────────────────────

    async def heartbeat(self, device_id: str) -> bool:
        """
        Record a heartbeat from a device.

        Parameters
        ----------
        device_id:
            The device sending the heartbeat.

        Returns
        -------
        bool
            True if the device is registered, False otherwise.
        """
        device = self._devices.get(device_id)
        if device is None:
            logger.warning("Heartbeat from unknown device: %s", device_id)
            return False

        now = time.time()
        device.last_heartbeat = now
        device.last_seen = datetime.now(timezone.utc).isoformat()

        # If device was reconnecting and now heartbeats, mark it online
        if device.status == DeviceStatus.RECONNECTING:
            device.status = DeviceStatus.ONLINE
            device.reconnect_attempts = 0
            logger.info("Device reconnected after heartbeat: %s", device_id)

        return True

    async def _heartbeat_monitor(self) -> None:
        """Background task that monitors device heartbeats."""
        logger.info("Heartbeat monitor started")
        while self._running:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                now = time.time()
                expired_devices: List[str] = []

                for device_id, device in self._devices.items():
                    if device.status == DeviceStatus.SUSPENDED:
                        continue

                    elapsed = now - device.last_heartbeat

                    if elapsed > self._heartbeat_timeout:
                        if device.status == DeviceStatus.ONLINE:
                            logger.warning(
                                "Device %s missed heartbeat (%.1fs), marking offline",
                                device_id, elapsed,
                            )
                            device.status = DeviceStatus.OFFLINE
                            await self._fire_event(device_id, "heartbeat_timeout", {
                                "elapsed": elapsed,
                            })
                            expired_devices.append(device_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat monitor error: %s", e)

        logger.info("Heartbeat monitor stopped")

    # ── Reconnection ─────────────────────────────────────────────────────

    async def request_reconnect(self, device_id: str) -> bool:
        """
        Request reconnection for an offline device.

        Implements exponential backoff with jitter.

        Parameters
        ----------
        device_id:
            The device to reconnect.

        Returns
        -------
        bool
            True if reconnection was initiated, False if not possible.
        """
        device = self._devices.get(device_id)
        if device is None:
            return False

        if device.reconnect_attempts >= self._max_reconnect_attempts:
            logger.warning(
                "Max reconnection attempts reached for device %s (%d/%d)",
                device_id, device.reconnect_attempts, self._max_reconnect_attempts,
            )
            device.status = DeviceStatus.OFFLINE
            await self._fire_event(device_id, "reconnect_failed", {
                "attempts": device.reconnect_attempts,
            })
            return False

        device.status = DeviceStatus.RECONNECTING
        device.reconnect_attempts += 1

        # Exponential backoff with jitter
        delay = self._reconnect_base_delay * (2 ** (device.reconnect_attempts - 1))
        jitter = delay * 0.3 * (2 * (0.5 - secrets.randbelow(100) / 100))
        delay = max(0.1, delay + jitter)

        logger.info(
            "Reconnection attempt %d/%d for device %s (next=%.1fs)",
            device.reconnect_attempts, self._max_reconnect_attempts,
            device_id, delay,
        )

        await self._fire_event(device_id, "reconnect_attempt", {
            "attempt": device.reconnect_attempts,
            "delay": delay,
        })

        # Schedule reconnect task
        asyncio.create_task(self._reconnect_after(device_id, delay))
        return True

    async def _reconnect_after(self, device_id: str, delay: float) -> None:
        """Wait and then attempt reconnection."""
        await asyncio.sleep(delay)
        device = self._devices.get(device_id)
        if device is None or device.status != DeviceStatus.RECONNECTING:
            return

        # In a real system this would send a reconnect probe
        # For now we simulate: if heartbeat arrives the status flips to ONLINE
        logger.debug("Reconnect probe sent to device %s", device_id)
        await self._fire_event(device_id, "reconnect_probe", {})

    # ── Device Discovery ─────────────────────────────────────────────────

    async def start_discovery(
        self,
        service_type: str = "_atlas-device._tcp.local.",
        scan_interval: float = 5.0,
    ) -> None:
        """
        Start discovering devices via mDNS/Bonjour simulation.

        Parameters
        ----------
        service_type:
            The mDNS service type to scan for.
        scan_interval:
            Seconds between discovery scans.
        """
        if self._discovery_running:
            logger.warning("Discovery is already running")
            return

        self._discovery_running = True
        self._discovery_task = asyncio.create_task(
            self._discovery_loop(service_type, scan_interval)
        )
        logger.info("Device discovery started (service=%s)", service_type)

    async def stop_discovery(self) -> None:
        """Stop device discovery scanning."""
        self._discovery_running = False
        if self._discovery_task and not self._discovery_task.done():
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        logger.info("Device discovery stopped")

    async def get_discovered(self) -> List[DiscoveredDevice]:
        """
        List currently discovered devices.

        Expired entries are pruned automatically.

        Returns
        -------
        List[DiscoveredDevice]
            Active discovered devices.
        """
        self._prune_discovered()
        return list(self._discovered.values())

    def on_device_found(self, callback: DeviceDiscoveryCallback) -> None:
        """
        Register a callback invoked when a new device is discovered.

        Parameters
        ----------
        callback:
            Async callback receiving the DiscoveredDevice.
        """
        self._discovery_callbacks.append(callback)

    async def _discovery_loop(
        self, service_type: str, scan_interval: float
    ) -> None:
        """Background discovery loop simulating mDNS scanning."""
        while self._discovery_running:
            try:
                # Simulate mDNS/Bonjour discovery
                # In production this would use zeroconf or pybonjour
                await asyncio.sleep(scan_interval)

                # Simulated devices for demonstration
                # Real implementation would broadcast and collect responses
                self._prune_discovered()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Discovery loop error: %s", e)

    def _prune_discovered(self) -> None:
        """Remove expired discovery entries."""
        now = time.time()
        expired = [
            key for key, dev in self._discovered.items()
            if now - dev.discovered_at > dev.ttl
        ]
        for key in expired:
            del self._discovered[key]

    async def add_discovered_device(self, device: DiscoveredDevice) -> None:
        """
        Manually add a discovered device (for testing or integration).

        Parameters
        ----------
        device:
            The discovered device entry.
        """
        self._discovered[device.device_name] = device
        logger.info("Discovered device: %s at %s:%d",
                     device.device_name, device.ip_address, device.port)

        for cb in self._discovery_callbacks:
            try:
                await cb(device)
            except Exception as e:
                logger.error("Discovery callback error: %s", e)

    # ── Event Helpers ────────────────────────────────────────────────────

    async def _fire_event(
        self, device_id: str, event_type: str, data: Dict[str, Any]
    ) -> int:
        """Fire event callbacks for a device."""
        count = 0
        for cb in self._event_callbacks.get(device_id, []):
            try:
                await cb(device_id, event_type, data)
                count += 1
            except Exception as e:
                logger.error("Event callback error (%s): %s", device_id, e)
        return count

    def on_event(
        self, device_id: str, callback: DeviceEventCallback
    ) -> None:
        """
        Register a callback for device events.

        Parameters
        ----------
        device_id:
            Device ID, or "*" for all devices.
        callback:
            Async callback receiving (device_id, event_type, data).
        """
        self._event_callbacks[device_id].append(callback)

    # ── Rate Limiting ────────────────────────────────────────────────────

    def _check_rate_limit(self, device_id: str) -> bool:
        """Check and update rate limit for a device. Returns True if allowed."""
        now = time.time()
        timestamps = self._command_rate[device_id]

        # Prune old entries
        cutoff = now - self._command_rate_window
        self._command_rate[device_id] = [
            ts for ts in timestamps if ts > cutoff
        ]
        timestamps = self._command_rate[device_id]

        if len(timestamps) >= self._command_rate_limit:
            return False

        timestamps.append(now)
        return True

    # ── Utility ──────────────────────────────────────────────────────────

    @staticmethod
    def _generate_fingerprint(device_id: str, metadata: Dict[str, Any]) -> str:
        """Generate a unique device fingerprint."""
        raw = f"{device_id}:{metadata.get('hostname', '')}:{metadata.get('os_name', '')}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _command_capability_map(command: str) -> Optional[Capability]:
        """Map commands to required capabilities."""
        mapping = {
            "capture_photo": Capability.CAMERA,
            "start_stream": Capability.CAMERA,
            "stop_stream": Capability.CAMERA,
            "capture_screen": Capability.SCREEN,
            "start_recording": Capability.SCREEN,
            "stop_recording": Capability.SCREEN,
            "get_location": Capability.LOCATION,
            "get_notifications": Capability.NOTIFICATIONS,
            "set_clipboard": Capability.CLIPBOARD,
            "get_clipboard": Capability.CLIPBOARD,
            "start_recording_audio": Capability.MICROPHONE,
            "play_audio": Capability.SPEAKER,
            "list_files": Capability.FILES,
            "read_file": Capability.FILES,
            "write_file": Capability.FILES,
        }
        return mapping.get(command)

    async def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the device node."""
        devices = list(self._devices.values())
        return {
            "total_devices": len(devices),
            "online": sum(1 for d in devices if d.status == DeviceStatus.ONLINE),
            "offline": sum(1 for d in devices if d.status == DeviceStatus.OFFLINE),
            "reconnecting": sum(1 for d in devices if d.status == DeviceStatus.RECONNECTING),
            "suspended": sum(1 for d in devices if d.status == DeviceStatus.SUSPENDED),
            "pending_commands": len(self._pending_commands),
            "discovered_devices": len(self._discovered),
            "capability_counts": self._count_capabilities(devices),
        }

    @staticmethod
    def _count_capabilities(devices: List[DeviceInfo]) -> Dict[str, int]:
        """Count how many devices have each capability."""
        counts: Dict[str, int] = defaultdict(int)
        for device in devices:
            for cap in device.capabilities:
                counts[cap.value] += 1
        return dict(counts)
