"""
Atlas Pairing — device and application pairing with discovery.

Provides secure pairing code generation, device discovery via mDNS/Bonjour
simulation, QR code pairing support, and device trust management.

Classes
-------
PairingManager
    Manage device pairing with 6-digit codes, expiration, and rate limiting.
DeviceDiscovery
    Discover pairable devices via mDNS/Bonjour simulation and QR codes.
"""

from .discovery import (
    DeviceDiscovery,
    DiscoveredDevice,
    QRPairingInfo,
)
from .manager import (
    AlreadyPairedError,
    ExpiredCodeError,
    InvalidCodeError,
    PairedDevice,
    PairingCode,
    PairingError,
    PairingManager,
    RateLimitError,
)

__all__ = [
    # Manager
    "PairingManager",
    "PairingCode",
    "PairedDevice",
    "PairingError",
    "InvalidCodeError",
    "ExpiredCodeError",
    "RateLimitError",
    "AlreadyPairedError",
    # Discovery
    "DeviceDiscovery",
    "DiscoveredDevice",
    "QRPairingInfo",
]
