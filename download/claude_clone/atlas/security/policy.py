"""
Atlas Security — Security Policy Engine.

Defines and enforces security policies for tool access, file operations,
network requests, DM pairing, and sandbox execution. Policies can be
loaded from/saved to YAML or JSON configuration files.

Inspired by OpenClaw's security policy architecture.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ToolPolicy:
    """Security policy for a specific tool.

    Attributes:
        name: Tool name or pattern (supports glob wildcards).
        allowed: Whether the tool is permitted.
        require_confirmation: Whether user confirmation is needed.
        max_calls_per_minute: Rate limit for tool calls.
        allowed_paths: File paths the tool is allowed to access.
        denied_paths: File paths the tool is explicitly denied.
        allowed_args: Allowed argument names (empty = all allowed).
        denied_args: Explicitly denied argument names.
        description: Human-readable policy description.
    """

    name: str = ""
    allowed: bool = True
    require_confirmation: bool = False
    max_calls_per_minute: int = 60
    allowed_paths: List[str] = field(default_factory=list)
    denied_paths: List[str] = field(default_factory=list)
    allowed_args: List[str] = field(default_factory=list)
    denied_args: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "allowed": self.allowed,
            "require_confirmation": self.require_confirmation,
            "max_calls_per_minute": self.max_calls_per_minute,
            "allowed_paths": self.allowed_paths,
            "denied_paths": self.denied_paths,
            "allowed_args": self.allowed_args,
            "denied_args": self.denied_args,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolPolicy":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FilePolicy:
    """Security policy for file system access.

    Attributes:
        allowed_roots: Directory roots that are permitted for access.
        denied_patterns: Glob patterns for files/directories to deny.
        max_file_size: Maximum file size in bytes (0 = unlimited).
        require_confirmation_for_delete: Whether delete ops need confirmation.
        require_confirmation_for_write: Whether write ops need confirmation.
        allowed_extensions: Allowed file extensions (empty = all).
        denied_extensions: Denied file extensions.
        read_only_paths: Paths that are read-only.
    """

    allowed_roots: List[str] = field(default_factory=lambda: ["/tmp", "."])
    denied_patterns: List[str] = field(default_factory=lambda: [
        "/etc/shadow", "/etc/passwd", "/etc/sudoers",
        "~/.ssh", "~/.gnupg", "~/.aws",
        "/proc", "/sys", "/dev",
    ])
    max_file_size: int = 100 * 1024 * 1024  # 100 MB
    require_confirmation_for_delete: bool = True
    require_confirmation_for_write: bool = False
    allowed_extensions: List[str] = field(default_factory=list)
    denied_extensions: List[str] = field(default_factory=list)
    read_only_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed_roots": self.allowed_roots,
            "denied_patterns": self.denied_patterns,
            "max_file_size": self.max_file_size,
            "require_confirmation_for_delete": self.require_confirmation_for_delete,
            "require_confirmation_for_write": self.require_confirmation_for_write,
            "allowed_extensions": self.allowed_extensions,
            "denied_extensions": self.denied_extensions,
            "read_only_paths": self.read_only_paths,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilePolicy":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class NetworkPolicy:
    """Security policy for network access.

    Attributes:
        allowed_hosts: Hostname patterns that are permitted.
        denied_hosts: Hostname patterns that are blocked.
        allowed_ports: Port numbers that are permitted (empty = all).
        denied_ports: Port numbers that are blocked.
        allowed_schemes: URL schemes allowed (http, https, etc.).
        denied_schemes: URL schemes that are blocked.
        require_confirmation: Whether all requests need confirmation.
        max_request_size: Maximum request body size in bytes.
        allowed_content_types: Allowed response content types.
        blocked_ip_ranges: IP address ranges to block.
        timeout_seconds: Default request timeout in seconds.
    """

    allowed_hosts: List[str] = field(default_factory=list)
    denied_hosts: List[str] = field(default_factory=lambda: [
        "localhost", "127.0.0.1", "0.0.0.0", "::1",
        "169.254.169.254",  # AWS metadata
        "metadata.google.internal",  # GCP metadata
    ])
    allowed_ports: List[int] = field(default_factory=lambda: [80, 443, 8080, 8443])
    denied_ports: List[int] = field(default_factory=lambda: [
        22, 23, 25, 3389, 5432, 6379, 27017, 3306,
    ])
    allowed_schemes: List[str] = field(default_factory=lambda: ["https", "http"])
    denied_schemes: List[str] = field(default_factory=list)
    require_confirmation: bool = False
    max_request_size: int = 10 * 1024 * 1024  # 10 MB
    allowed_content_types: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    timeout_seconds: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed_hosts": self.allowed_hosts,
            "denied_hosts": self.denied_hosts,
            "allowed_ports": self.allowed_ports,
            "denied_ports": self.denied_ports,
            "allowed_schemes": self.allowed_schemes,
            "denied_schemes": self.denied_schemes,
            "require_confirmation": self.require_confirmation,
            "max_request_size": self.max_request_size,
            "allowed_content_types": self.allowed_content_types,
            "blocked_ip_ranges": self.blocked_ip_ranges,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkPolicy":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DMPolicy:
    """Security policy for direct messaging.

    Attributes:
        allow_new_dm: Whether new (unpaired) DMs are allowed.
        require_pairing: Whether pairing is required before messaging.
        auto_approve_contacts: Whether to auto-approve known contacts.
        max_dm_per_minute: Rate limit for DMs per user.
        allowed_channels: Channel types allowed for DMs.
        blocked_channels: Channel types blocked for DMs.
        max_message_length: Maximum DM length in characters.
    """

    allow_new_dm: bool = False
    require_pairing: bool = True
    auto_approve_contacts: bool = False
    max_dm_per_minute: int = 30
    allowed_channels: List[str] = field(default_factory=list)
    blocked_channels: List[str] = field(default_factory=list)
    max_message_length: int = 10000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allow_new_dm": self.allow_new_dm,
            "require_pairing": self.require_pairing,
            "auto_approve_contacts": self.auto_approve_contacts,
            "max_dm_per_minute": self.max_dm_per_minute,
            "allowed_channels": self.allowed_channels,
            "blocked_channels": self.blocked_channels,
            "max_message_length": self.max_message_length,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DMPolicy":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SandboxPolicy:
    """Security policy for sandbox execution.

    Attributes:
        enabled: Whether sandboxing is active.
        sandbox_type: Type of sandbox (none, docker, process, restricted_path).
        resource_limits: Resource limit configuration.
        allowed_commands: Command patterns allowed in sandbox.
        denied_commands: Command patterns denied in sandbox.
        network_access: Whether sandbox has network access.
        environment_vars: Environment variables passed to sandbox.
        working_directory: Default working directory.
        timeout_seconds: Maximum execution time in seconds.
    """

    enabled: bool = True
    sandbox_type: str = "process"
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_memory_mb": 512,
        "max_cpu_percent": 50,
        "max_time_seconds": 300,
        "max_output_bytes": 10 * 1024 * 1024,
    })
    allowed_commands: List[str] = field(default_factory=list)
    denied_commands: List[str] = field(default_factory=lambda: [
        "rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:",  # Fork bomb
    ])
    network_access: bool = False
    environment_vars: Dict[str, str] = field(default_factory=dict)
    working_directory: str = "/tmp"
    timeout_seconds: int = 300

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enabled": self.enabled,
            "sandbox_type": self.sandbox_type,
            "resource_limits": self.resource_limits,
            "allowed_commands": self.allowed_commands,
            "denied_commands": self.denied_commands,
            "network_access": self.network_access,
            "environment_vars": self.environment_vars,
            "working_directory": self.working_directory,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SandboxPolicy":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Policy Evaluation Result
# ---------------------------------------------------------------------------

@dataclass
class PolicyDecision:
    """Result of a security policy evaluation.

    Attributes:
        allowed: Whether the operation is permitted.
        require_confirmation: Whether user confirmation is needed.
        reason: Human-readable explanation of the decision.
        policy_name: Name of the policy that made the decision.
        metadata: Additional decision context.
    """

    allowed: bool = True
    require_confirmation: bool = False
    reason: str = ""
    policy_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Security Policy
# ---------------------------------------------------------------------------

class SecurityPolicy:
    """Comprehensive security policy engine.

    Manages and evaluates security policies for tool access, file operations,
    network requests, DM pairing, and sandbox execution. Policies can be
    loaded from and saved to YAML or JSON configuration files.

    Usage::

        policy = SecurityPolicy()

        # Evaluate tool access
        decision = policy.evaluate_tool_access("bash", {"command": "ls"})
        if decision.require_confirmation:
            await confirm_with_user()

        # Evaluate file access
        decision = policy.evaluate_file_access("/home/user/data.csv", "read")

        # Evaluate network access
        decision = policy.evaluate_network_request("https://api.example.com/data", "GET")

        # Save/Load
        policy.save("security_policy.yaml")
        policy = SecurityPolicy.load("security_policy.yaml")
    """

    def __init__(
        self,
        tool_policies: Optional[List[ToolPolicy]] = None,
        file_policy: Optional[FilePolicy] = None,
        network_policy: Optional[NetworkPolicy] = None,
        dm_policy: Optional[DMPolicy] = None,
        sandbox_policy: Optional[SandboxPolicy] = None,
    ) -> None:
        """Initialize the security policy engine.

        Args:
            tool_policies: List of tool-specific policies.
            file_policy: File system access policy.
            network_policy: Network access policy.
            dm_policy: Direct messaging policy.
            sandbox_policy: Sandbox execution policy.
        """
        self._tool_policies: List[ToolPolicy] = tool_policies or []
        self._tool_policies_index: Dict[str, ToolPolicy] = {}
        for tp in self._tool_policies:
            self._tool_policies_index[tp.name.lower()] = tp

        self._file_policy = file_policy or FilePolicy()
        self._network_policy = network_policy or NetworkPolicy()
        self._dm_policy = dm_policy or DMPolicy()
        self._sandbox_policy = sandbox_policy or SandboxPolicy()

        self._tool_rate_limits: Dict[str, List[float]] = {}

        logger.info(
            "SecurityPolicy initialized (%d tool policies)",
            len(self._tool_policies),
        )

    # ------------------------------------------------------------------
    # Tool Policy Management
    # ------------------------------------------------------------------

    def set_tool_policy(self, policy: ToolPolicy) -> None:
        """Set or update a tool policy.

        Args:
            policy: The tool policy to set.
        """
        self._tool_policies_index[policy.name.lower()] = policy

        # Update or append in list
        found = False
        for i, existing in enumerate(self._tool_policies):
            if existing.name.lower() == policy.name.lower():
                self._tool_policies[i] = policy
                found = True
                break
        if not found:
            self._tool_policies.append(policy)

        logger.info("Set tool policy for '%s' (allowed=%s)", policy.name, policy.allowed)

    def get_tool_policy(self, tool_name: str) -> Optional[ToolPolicy]:
        """Get the policy for a specific tool.

        Supports glob pattern matching if no exact match is found.

        Args:
            tool_name: The tool name to look up.

        Returns:
            Matching ToolPolicy, or None.
        """
        # Exact match first
        policy = self._tool_policies_index.get(tool_name.lower())
        if policy:
            return policy

        # Pattern match
        for tp in self._tool_policies:
            if self._match_pattern(tool_name, tp.name):
                return tp

        return None

    def remove_tool_policy(self, tool_name: str) -> bool:
        """Remove a tool policy.

        Args:
            tool_name: The tool name to remove.

        Returns:
            True if found and removed.
        """
        key = tool_name.lower()
        if key in self._tool_policies_index:
            del self._tool_policies_index[key]
            self._tool_policies = [
                tp for tp in self._tool_policies
                if tp.name.lower() != key
            ]
            return True
        return False

    def list_tool_policies(self) -> List[Dict[str, Any]]:
        """List all tool policies.

        Returns:
            List of policy dictionaries.
        """
        return [tp.to_dict() for tp in self._tool_policies]

    # ------------------------------------------------------------------
    # Policy Evaluation
    # ------------------------------------------------------------------

    def evaluate_tool_access(
        self,
        tool_name: str,
        tool_input: Optional[Dict[str, Any]] = None,
        user: str = "",
    ) -> PolicyDecision:
        """Evaluate whether a tool call is allowed.

        Checks:
        1. Whether the tool has a policy and if it's allowed
        2. Rate limits for the tool
        3. Argument restrictions (allowed/denied args)
        4. Path restrictions if applicable

        Args:
            tool_name: Name of the tool being called.
            tool_input: Tool input parameters.
            user: User or agent making the call.

        Returns:
            PolicyDecision with the evaluation result.
        """
        import time
        policy = self.get_tool_policy(tool_name)

        # No specific policy — allow by default
        if policy is None:
            return PolicyDecision(
                allowed=True,
                reason=f"No specific policy for tool '{tool_name}', allowed by default",
                policy_name="default",
            )

        # Check if explicitly denied
        if not policy.allowed:
            return PolicyDecision(
                allowed=False,
                reason=f"Tool '{tool_name}' is explicitly denied by policy",
                policy_name=tool_name,
            )

        # Check rate limit
        now = time.time()
        window = 60.0
        if tool_name not in self._tool_rate_limits:
            self._tool_rate_limits[tool_name] = []

        self._tool_rate_limits[tool_name] = [
            t for t in self._tool_rate_limits[tool_name]
            if now - t < window
        ]

        if len(self._tool_rate_limits[tool_name]) >= policy.max_calls_per_minute:
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"Tool '{tool_name}' rate limit exceeded "
                    f"({len(self._tool_rate_limits[tool_name])}/{policy.max_calls_per_minute} per minute)"
                ),
                policy_name=tool_name,
            )

        self._tool_rate_limits[tool_name].append(now)

        # Check denied arguments
        if tool_input and policy.denied_args:
            for arg_name in policy.denied_args:
                if arg_name in tool_input:
                    return PolicyDecision(
                        allowed=False,
                        reason=f"Tool '{tool_name}': denied argument '{arg_name}'",
                        policy_name=tool_name,
                    )

        # Check allowed arguments (if specified, only these are allowed)
        if tool_input and policy.allowed_args:
            for arg_name in tool_input:
                if arg_name not in policy.allowed_args:
                    return PolicyDecision(
                        allowed=False,
                        reason=f"Tool '{tool_name}': argument '{arg_name}' not in allowed list",
                        policy_name=tool_name,
                    )

        # Check path restrictions
        if tool_input:
            input_str = json.dumps(tool_input)
            for denied_path in policy.denied_paths:
                if denied_path in input_str:
                    return PolicyDecision(
                        allowed=False,
                        reason=f"Tool '{tool_name}': references denied path '{denied_path}'",
                        policy_name=tool_name,
                    )

        return PolicyDecision(
            allowed=True,
            require_confirmation=policy.require_confirmation,
            reason=f"Tool '{tool_name}' access allowed",
            policy_name=tool_name,
        )

    def evaluate_file_access(
        self,
        path: str,
        operation: str,
        user: str = "",
    ) -> PolicyDecision:
        """Evaluate whether a file system operation is allowed.

        Checks:
        1. Denied patterns (blocklist)
        2. Allowed roots (allowlist)
        3. Read-only paths
        4. Extension restrictions
        5. Operation-specific confirmations

        Args:
            path: File or directory path.
            operation: Operation type (read, write, delete, etc.).
            user: User or agent performing the operation.

        Returns:
            PolicyDecision with the evaluation result.
        """
        from pathlib import Path as FilePath

        resolved = FilePath(path).resolve()

        # Check denied patterns
        for pattern in self._file_policy.denied_patterns:
            if self._match_pattern(str(resolved), pattern):
                return PolicyDecision(
                    allowed=False,
                    reason=f"Path '{path}' matches denied pattern '{pattern}'",
                    policy_name="file_policy",
                )

        # Check allowed roots
        if self._file_policy.allowed_roots:
            in_allowed = any(
                self._path_is_under(str(resolved), root)
                for root in self._file_policy.allowed_roots
            )
            if not in_allowed:
                return PolicyDecision(
                    allowed=False,
                    reason=(
                        f"Path '{path}' is not under any allowed root: "
                        f"{self._file_policy.allowed_roots}"
                    ),
                    policy_name="file_policy",
                )

        # Check read-only paths for write/delete
        if operation in ("write", "delete", "move"):
            for ro_path in self._file_policy.read_only_paths:
                if self._path_is_under(str(resolved), ro_path):
                    return PolicyDecision(
                        allowed=False,
                        reason=f"Path '{path}' is in read-only location",
                        policy_name="file_policy",
                    )

        # Check denied extensions
        if self._file_policy.denied_extensions:
            suffix = resolved.suffix.lower()
            if suffix and suffix in self._file_policy.denied_extensions:
                return PolicyDecision(
                    allowed=False,
                    reason=f"File extension '{suffix}' is denied",
                    policy_name="file_policy",
                )

        # Check allowed extensions (if specified)
        if self._file_policy.allowed_extensions:
            suffix = resolved.suffix.lower()
            if suffix and suffix not in self._file_policy.allowed_extensions:
                return PolicyDecision(
                    allowed=False,
                    reason=f"File extension '{suffix}' not in allowed list",
                    policy_name="file_policy",
                )

        # Operation-specific confirmation requirements
        require_confirm = False
        if operation == "delete" and self._file_policy.require_confirmation_for_delete:
            require_confirm = True
        elif operation == "write" and self._file_policy.require_confirmation_for_write:
            require_confirm = True

        return PolicyDecision(
            allowed=True,
            require_confirmation=require_confirm,
            reason=f"File {operation} on '{path}' allowed",
            policy_name="file_policy",
        )

    def evaluate_network_request(
        self,
        url: str,
        method: str,
        user: str = "",
    ) -> PolicyDecision:
        """Evaluate whether a network request is allowed.

        Checks:
        1. URL scheme (http vs https)
        2. Host-based allowlist/denylist
        3. Port restrictions
        4. Internal network blocking

        Args:
            url: The URL to access.
            method: HTTP method (GET, POST, etc.).
            user: User or agent making the request.

        Returns:
            PolicyDecision with the evaluation result.
        """
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
        except Exception:
            return PolicyDecision(
                allowed=False,
                reason=f"Invalid URL: {url}",
                policy_name="network_policy",
            )

        # Check scheme
        scheme = parsed.scheme.lower()
        if scheme in self._network_policy.denied_schemes:
            return PolicyDecision(
                allowed=False,
                reason=f"URL scheme '{scheme}' is denied",
                policy_name="network_policy",
            )

        if self._network_policy.allowed_schemes:
            if scheme not in self._network_policy.allowed_schemes:
                return PolicyDecision(
                    allowed=False,
                    reason=f"URL scheme '{scheme}' not in allowed list",
                    policy_name="network_policy",
                )

        # Check host
        hostname = parsed.hostname or ""
        if hostname:
            # Check denied hosts
            for denied in self._network_policy.denied_hosts:
                if self._match_pattern(hostname, denied):
                    return PolicyDecision(
                        allowed=False,
                        reason=f"Host '{hostname}' is denied",
                        policy_name="network_policy",
                    )

            # Check allowed hosts (if specified)
            if self._network_policy.allowed_hosts:
                host_allowed = any(
                    self._match_pattern(hostname, allowed)
                    for allowed in self._network_policy.allowed_hosts
                )
                if not host_allowed:
                    return PolicyDecision(
                        allowed=False,
                        reason=f"Host '{hostname}' not in allowed list",
                        policy_name="network_policy",
                    )

        # Check port
        port = parsed.port
        if port:
            if port in self._network_policy.denied_ports:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Port {port} is denied",
                    policy_name="network_policy",
                )
            if self._network_policy.allowed_ports:
                if port not in self._network_policy.allowed_ports:
                    return PolicyDecision(
                        allowed=False,
                        reason=f"Port {port} not in allowed list",
                        policy_name="network_policy",
                    )

        return PolicyDecision(
            allowed=True,
            require_confirmation=self._network_policy.require_confirmation,
            reason=f"Network {method} {url} allowed",
            policy_name="network_policy",
        )

    def evaluate_dm_access(
        self,
        channel_type: str,
        peer_id: str,
        message_length: int = 0,
    ) -> PolicyDecision:
        """Evaluate whether a DM is allowed.

        Args:
            channel_type: The channel type.
            peer_id: The peer/user identifier.
            message_length: Length of the message.

        Returns:
            PolicyDecision with the evaluation result.
        """
        # Check blocked channels
        if channel_type in self._dm_policy.blocked_channels:
            return PolicyDecision(
                allowed=False,
                reason=f"Channel type '{channel_type}' is blocked for DMs",
                policy_name="dm_policy",
            )

        # Check allowed channels (if specified)
        if self._dm_policy.allowed_channels:
            if channel_type not in self._dm_policy.allowed_channels:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Channel type '{channel_type}' not in allowed DM channels",
                    policy_name="dm_policy",
                )

        # Check message length
        if message_length > self._dm_policy.max_message_length:
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"DM message length {message_length} exceeds maximum "
                    f"{self._dm_policy.max_message_length}"
                ),
                policy_name="dm_policy",
            )

        require_pairing = self._dm_policy.require_pairing

        return PolicyDecision(
            allowed=True,
            require_confirmation=require_pairing,
            reason="DM access allowed" + (", pairing required" if require_pairing else ""),
            policy_name="dm_policy",
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all policies to a dictionary."""
        return {
            "version": "1.0",
            "tool_policies": [tp.to_dict() for tp in self._tool_policies],
            "file_policy": self._file_policy.to_dict(),
            "network_policy": self._network_policy.to_dict(),
            "dm_policy": self._dm_policy.to_dict(),
            "sandbox_policy": self._sandbox_policy.to_dict(),
        }

    def save(self, path: str) -> None:
        """Save all policies to a YAML or JSON file.

        Auto-detects format based on file extension.

        Args:
            path: Output file path (.yaml, .yml, or .json).
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()

        suffix = output.suffix.lower()
        if suffix in (".yaml", ".yml"):
            self._save_yaml(data, output)
        else:
            self._save_json(data, output)

        logger.info("Saved security policies to %s", output)

    @classmethod
    def load(cls, path: str) -> "SecurityPolicy":
        """Load security policies from a YAML or JSON file.

        Args:
            path: Input file path.

        Returns:
            Loaded SecurityPolicy instance.
        """
        input_path = Path(path)
        suffix = input_path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            data = cls._load_yaml(input_path)
        else:
            data = cls._load_json(input_path)

        # Parse tool policies
        tool_policies = []
        for tp_data in data.get("tool_policies", []):
            tool_policies.append(ToolPolicy.from_dict(tp_data))

        # Parse sub-policies
        file_data = data.get("file_policy", {})
        file_policy = FilePolicy.from_dict(file_data) if file_data else FilePolicy()

        net_data = data.get("network_policy", {})
        network_policy = NetworkPolicy.from_dict(net_data) if net_data else NetworkPolicy()

        dm_data = data.get("dm_policy", {})
        dm_policy = DMPolicy.from_dict(dm_data) if dm_data else DMPolicy()

        sandbox_data = data.get("sandbox_policy", {})
        sandbox_policy = SandboxPolicy.from_dict(sandbox_data) if sandbox_data else SandboxPolicy()

        instance = cls(
            tool_policies=tool_policies,
            file_policy=file_policy,
            network_policy=network_policy,
            dm_policy=dm_policy,
            sandbox_policy=sandbox_policy,
        )

        logger.info("Loaded security policies from %s", input_path)
        return instance

    # ------------------------------------------------------------------
    # YAML helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_yaml(data: Dict[str, Any], path: Path) -> None:
        """Save data as YAML."""
        try:
            import yaml
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            # Fallback to JSON if PyYAML not available
            logger.warning("PyYAML not available, saving as JSON instead")
            json_path = path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load data from YAML."""
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            raise ImportError("PyYAML is required to load YAML config files")

    @staticmethod
    def _save_json(data: Dict[str, Any], path: Path) -> None:
        """Save data as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Load data from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_pattern(text: str, pattern: str) -> bool:
        """Match text against a glob-like pattern.

        Supports * (any chars) and ? (single char) wildcards.

        Args:
            text: Text to match against.
            pattern: Pattern string.

        Returns:
            True if the text matches the pattern.
        """
        if not pattern:
            return True
        if "*" not in pattern and "?" not in pattern:
            return text.lower() == pattern.lower()

        # Convert glob to regex
        regex_pattern = ""
        for char in pattern:
            if char == "*":
                regex_pattern += ".*"
            elif char == "?":
                regex_pattern += "."
            elif char in r"\[](){}+^$.|":
                regex_pattern += "\\" + char
            else:
                regex_pattern += char

        try:
            return bool(re.search(f"^{regex_pattern}$", text, re.IGNORECASE))
        except re.error:
            return text.lower() == pattern.lower()

    @staticmethod
    def _path_is_under(path: str, root: str) -> bool:
        """Check if a path is under a root directory.

        Args:
            path: The path to check.
            root: The root directory.

        Returns:
            True if path is under root.
        """
        from pathlib import Path as FilePath
        try:
            resolved_path = FilePath(path).resolve()
            resolved_root = FilePath(root).resolve()
            return str(resolved_path).startswith(str(resolved_root))
        except (OSError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def file_policy(self) -> FilePolicy:
        """Current file access policy."""
        return self._file_policy

    @property
    def network_policy(self) -> NetworkPolicy:
        """Current network access policy."""
        return self._network_policy

    @property
    def dm_policy(self) -> DMPolicy:
        """Current DM policy."""
        return self._dm_policy

    @property
    def sandbox_policy(self) -> SandboxPolicy:
        """Current sandbox policy."""
        return self._sandbox_policy

    @file_policy.setter
    def file_policy(self, value: FilePolicy) -> None:
        self._file_policy = value

    @network_policy.setter
    def network_policy(self, value: NetworkPolicy) -> None:
        self._network_policy = value

    @dm_policy.setter
    def dm_policy(self, value: DMPolicy) -> None:
        self._dm_policy = value

    @sandbox_policy.setter
    def sandbox_policy(self, value: SandboxPolicy) -> None:
        self._sandbox_policy = value

    def __repr__(self) -> str:
        return (
            f"<SecurityPolicy "
            f"tools={len(self._tool_policies)} "
            f"file={'✓' if self._file_policy else '✗'} "
            f"network={'✓' if self._network_policy else '✗'}>"
        )
