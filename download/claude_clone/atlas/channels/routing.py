"""
Atlas Channels — Message Router.

Provides intelligent message routing that determines which agent, session,
or handler should process an incoming message. Supports pattern-based routing,
account-to-agent bindings, keyword matching, and consistent session key
derivation.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union

from atlas.channels.base import ChannelMessage, ChannelType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RouteMatchType(Enum):
    """Types of routing rules."""

    REGEX = "regex"
    KEYWORD = "keyword"
    CHANNEL = "channel"
    SENDER = "sender"
    COMMAND = "command"
    EXACT = "exact"
    PREFIX = "prefix"
    WILDCARD = "wildcard"


class RoutingDecision(Enum):
    """Outcome of a routing decision."""

    ROUTED = "routed"
    DROPPED = "dropped"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    NO_MATCH = "no_match"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RouteRule:
    """A single routing rule definition.

    Attributes:
        id: Unique rule identifier.
        name: Human-readable rule name.
        match_type: Type of pattern matching to apply.
        pattern: The pattern string or regex.
        compiled_pattern: Pre-compiled regex (for REGEX rules).
        handler_name: Name or ID of the handler to route to.
        priority: Rule priority (lower = higher priority).
        enabled: Whether this rule is active.
        description: Human-readable description of the rule.
        metadata: Additional rule metadata.
        match_count: Number of times this rule has matched.
        created_at: When this rule was created.
    """

    id: str = ""
    name: str = ""
    match_type: RouteMatchType = RouteMatchType.REGEX
    pattern: str = ""
    compiled_pattern: Optional[Pattern[str]] = None
    handler_name: str = ""
    priority: int = 100
    enabled: bool = True
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    match_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Compile regex patterns after initialization."""
        if self.match_type == RouteMatchType.REGEX and self.pattern and not self.compiled_pattern:
            try:
                self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
            except re.error as exc:
                logger.error("Invalid regex pattern %r: %s", self.pattern, exc)
                self.enabled = False

    def matches(self, message: ChannelMessage) -> bool:
        """Check if this rule matches a given message.

        Args:
            message: The message to evaluate against.

        Returns:
            True if the rule matches.
        """
        if not self.enabled:
            return False

        if self.match_type == RouteMatchType.REGEX:
            return self._match_regex(message)
        elif self.match_type == RouteMatchType.KEYWORD:
            return self._match_keyword(message)
        elif self.match_type == RouteMatchType.CHANNEL:
            return self._match_channel(message)
        elif self.match_type == RouteMatchType.SENDER:
            return self._match_sender(message)
        elif self.match_type == RouteMatchType.COMMAND:
            return self._match_command(message)
        elif self.match_type == RouteMatchType.EXACT:
            return self._match_exact(message)
        elif self.match_type == RouteMatchType.PREFIX:
            return self._match_prefix(message)
        elif self.match_type == RouteMatchType.WILDCARD:
            return True

        return False

    def _match_regex(self, message: ChannelMessage) -> bool:
        """Match using compiled regex pattern on message content."""
        if self.compiled_pattern is None:
            return False
        return bool(self.compiled_pattern.search(message.content))

    def _match_keyword(self, message: ChannelMessage) -> bool:
        """Match using keyword (space-separated, all must appear)."""
        keywords = self.pattern.lower().split()
        content = message.content.lower()
        return all(kw in content for kw in keywords)

    def _match_channel(self, message: ChannelMessage) -> bool:
        """Match based on channel type."""
        try:
            target = ChannelType(self.pattern)
            return message.channel_type == target
        except ValueError:
            return message.channel_type.value == self.pattern.lower()

    def _match_sender(self, message: ChannelMessage) -> bool:
        """Match based on sender identifier."""
        return message.sender == self.pattern

    def _match_command(self, message: ChannelMessage) -> bool:
        """Match if message starts with a slash command."""
        return message.content.strip().startswith(self.pattern)

    def _match_exact(self, message: ChannelMessage) -> bool:
        """Match exact message content."""
        return message.content.strip().lower() == self.pattern.lower()

    def _match_prefix(self, message: ChannelMessage) -> bool:
        """Match message content prefix."""
        return message.content.strip().lower().startswith(self.pattern.lower())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize rule to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "match_type": self.match_type.value,
            "pattern": self.pattern,
            "handler_name": self.handler_name,
            "priority": self.priority,
            "enabled": self.enabled,
            "description": self.description,
            "metadata": self.metadata,
            "match_count": self.match_count,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AccountBinding:
    """Binding between a messaging account and an agent/session.

    Attributes:
        account_id: Platform-specific user/account identifier.
        agent_id: The agent ID to handle this account's messages.
        channel_type: The channel type this binding applies to.
        session_key: Derived session key for conversation continuity.
        bound_at: When this binding was created.
        metadata: Additional binding metadata (e.g., user preferences).
    """

    account_id: str = ""
    agent_id: str = ""
    channel_type: ChannelType = ChannelType.WEBCHAT
    session_key: str = ""
    bound_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize binding to dictionary."""
        return {
            "account_id": self.account_id,
            "agent_id": self.agent_id,
            "channel_type": self.channel_type.value,
            "session_key": self.session_key,
            "bound_at": self.bound_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccountBinding":
        """Deserialize binding from dictionary."""
        data = data.copy()
        if "channel_type" in data and isinstance(data["channel_type"], str):
            data["channel_type"] = ChannelType(data["channel_type"])
        if "bound_at" in data and isinstance(data["bound_at"], str):
            data["bound_at"] = datetime.fromisoformat(data["bound_at"])
        return cls(**data)


@dataclass
class RoutingResult:
    """Result of a message routing operation.

    Attributes:
        decision: The routing decision made.
        handler_name: The selected handler name.
        session_key: The derived session key.
        rule_id: The ID of the matching rule (if any).
        confidence: Confidence score (0.0 to 1.0).
        reason: Human-readable explanation of the routing decision.
        metadata: Additional routing metadata.
    """

    decision: RoutingDecision = RoutingDecision.NO_MATCH
    handler_name: str = ""
    session_key: str = ""
    rule_id: str = ""
    confidence: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Handler Type
# ---------------------------------------------------------------------------

MessageHandler = Callable[
    [ChannelMessage, RoutingResult],
    Any,
]


# ---------------------------------------------------------------------------
# Message Router
# ---------------------------------------------------------------------------

class MessageRouter:
    """Intelligent message routing engine.

    Evaluates incoming messages against a priority-ordered set of routing
    rules and account bindings to determine the appropriate handler and
    session. Provides consistent session key derivation for conversation
    continuity across channels.

    Features:
    - Priority-based rule matching
    - Multiple match types (regex, keyword, channel, sender, command, etc.)
    - Account-to-agent binding lookup
    - Consistent session key derivation via SHA-256
    - Fallback/default handler support
    - Routing statistics and audit trail

    Usage::

        router = MessageRouter()

        # Add routing rules
        router.add_route(
            name="help_command",
            match_type=RouteMatchType.COMMAND,
            pattern="/help",
            handler_name="help_handler",
            priority=10,
        )

        # Add account binding
        router.add_account_binding(
            account_id="user_123",
            agent_id="agent_main",
            channel_type=ChannelType.TELEGRAM,
        )

        # Route messages
        result = router.route(message)
        if result.decision == RoutingDecision.ROUTED:
            handler = router.get_handler(result.handler_name)
            await handler(message, result)
    """

    DEFAULT_HANDLER = "default"
    FALLBACK_HANDLER = "fallback"

    def __init__(self, default_handler: str = "default") -> None:
        """Initialize the message router.

        Args:
            default_handler: Name of the default handler when no rules match.
        """
        self._rules: List[RouteRule] = []
        self._handlers: Dict[str, MessageHandler] = {}
        self._account_bindings: Dict[
            Tuple[ChannelType, str], AccountBinding
        ] = {}
        self._session_cache: Dict[str, str] = {}
        self._default_handler = default_handler
        self._stats = {
            "total_routed": 0,
            "total_dropped": 0,
            "total_deferred": 0,
            "total_no_match": 0,
            "rule_hits": {},
        }
        self._lock = None  # lazy init for non-async contexts

        logger.info("MessageRouter initialized (default_handler=%s)", default_handler)

    # ------------------------------------------------------------------
    # Rule Management
    # ------------------------------------------------------------------

    def add_route(
        self,
        name: str,
        match_type: Union[RouteMatchType, str],
        pattern: str,
        handler_name: str,
        priority: int = 100,
        description: str = "",
        rule_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RouteRule:
        """Add a routing rule.

        Args:
            name: Human-readable rule name.
            match_type: Type of matching (RouteMatchType enum or string).
            pattern: Pattern string to match against.
            handler_name: Handler to route matching messages to.
            priority: Priority (lower = higher priority).
            description: Rule description.
            rule_id: Optional explicit ID (auto-generated if None).
            metadata: Additional rule metadata.

        Returns:
            The created RouteRule.

        Raises:
            ValueError: If match_type is invalid.
        """
        if isinstance(match_type, str):
            try:
                match_type = RouteMatchType(match_type)
            except ValueError:
                raise ValueError(
                    f"Invalid match type: {match_type!r}. "
                    f"Valid: {[t.value for t in RouteMatchType]}"
                )

        if not rule_id:
            rule_id = hashlib.md5(
                f"{name}:{pattern}:{handler_name}".encode()
            ).hexdigest()[:12]

        rule = RouteRule(
            id=rule_id,
            name=name,
            match_type=match_type,
            pattern=pattern,
            handler_name=handler_name,
            priority=priority,
            description=description,
            metadata=metadata or {},
        )

        self._rules.append(rule)
        # Keep sorted by priority
        self._rules.sort(key=lambda r: r.priority)

        logger.info(
            "Added route rule '%s' (%s, pattern=%r, handler=%s, priority=%d)",
            name, match_type.value, pattern, handler_name, priority,
        )
        return rule

    def remove_route(self, rule_id: str) -> bool:
        """Remove a routing rule by ID.

        Args:
            rule_id: The rule ID to remove.

        Returns:
            True if the rule was found and removed.
        """
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.id != rule_id]
        removed = len(self._rules) < original_len
        if removed:
            logger.info("Removed route rule %s", rule_id)
        return removed

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a routing rule.

        Args:
            rule_id: The rule ID to enable.

        Returns:
            True if found and enabled.
        """
        for rule in self._rules:
            if rule.id == rule_id:
                rule.enabled = True
                return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a routing rule.

        Args:
            rule_id: The rule ID to disable.

        Returns:
            True if found and disabled.
        """
        for rule in self._rules:
            if rule.id == rule_id:
                rule.enabled = False
                return True
        return False

    def get_rule(self, rule_id: str) -> Optional[RouteRule]:
        """Get a rule by ID.

        Args:
            rule_id: The rule ID.

        Returns:
            The RouteRule, or None if not found.
        """
        for rule in self._rules:
            if rule.id == rule_id:
                return rule
        return None

    def list_rules(
        self,
        enabled_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """List all routing rules.

        Args:
            enabled_only: Only return enabled rules.

        Returns:
            List of rule dictionaries.
        """
        rules = self._rules
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        return [r.to_dict() for r in rules]

    # ------------------------------------------------------------------
    # Handler Management
    # ------------------------------------------------------------------

    def register_handler(
        self,
        name: str,
        handler: MessageHandler,
    ) -> None:
        """Register a message handler.

        Args:
            name: Handler name (referenced by routing rules).
            handler: Async or sync callable accepting (message, routing_result).
        """
        self._handlers[name] = handler
        logger.info("Registered handler '%s'", name)

    def unregister_handler(self, name: str) -> bool:
        """Unregister a message handler.

        Args:
            name: Handler name to remove.

        Returns:
            True if found and removed.
        """
        if name in self._handlers:
            del self._handlers[name]
            logger.info("Unregistered handler '%s'", name)
            return True
        return False

    def get_handler(self, name: str) -> Optional[MessageHandler]:
        """Get a registered handler by name.

        Args:
            name: Handler name.

        Returns:
            The handler callable, or None.
        """
        return self._handlers.get(name)

    def list_handlers(self) -> List[str]:
        """List all registered handler names.

        Returns:
            Sorted list of handler names.
        """
        return sorted(self._handlers.keys())

    # ------------------------------------------------------------------
    # Account Bindings
    # ------------------------------------------------------------------

    def add_account_binding(
        self,
        account_id: str,
        agent_id: str,
        channel_type: ChannelType,
        session_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AccountBinding:
        """Bind a messaging account to an agent.

        Args:
            account_id: Platform-specific user/account identifier.
            agent_id: The agent ID to handle this account's messages.
            channel_type: The channel type.
            session_key: Pre-derived session key (auto-generated if None).
            metadata: Additional binding metadata.

        Returns:
            The created AccountBinding.
        """
        if session_key is None:
            session_key = self.derive_session_key(channel_type, account_id)

        binding = AccountBinding(
            account_id=account_id,
            agent_id=agent_id,
            channel_type=channel_type,
            session_key=session_key,
            metadata=metadata or {},
        )

        key = (channel_type, account_id)
        self._account_bindings[key] = binding

        logger.info(
            "Bound account %s on %s to agent %s (session=%s)",
            account_id, channel_type.value, agent_id, session_key[:12],
        )
        return binding

    def remove_account_binding(
        self,
        channel_type: ChannelType,
        account_id: str,
    ) -> bool:
        """Remove an account binding.

        Args:
            channel_type: The channel type.
            account_id: The account identifier.

        Returns:
            True if found and removed.
        """
        key = (channel_type, account_id)
        if key in self._account_bindings:
            del self._account_bindings[key]
            logger.info(
                "Removed binding for %s on %s",
                account_id, channel_type.value,
            )
            return True
        return False

    def get_account_binding(
        self,
        channel_type: ChannelType,
        account_id: str,
    ) -> Optional[AccountBinding]:
        """Look up an account binding.

        Args:
            channel_type: The channel type.
            account_id: The account identifier.

        Returns:
            The AccountBinding, or None if not bound.
        """
        return self._account_bindings.get((channel_type, account_id))

    def list_account_bindings(
        self,
        channel_type: Optional[ChannelType] = None,
    ) -> List[Dict[str, Any]]:
        """List all account bindings.

        Args:
            channel_type: Filter by channel type (None for all).

        Returns:
            List of binding dictionaries.
        """
        bindings = list(self._account_bindings.values())
        if channel_type is not None:
            bindings = [
                b for b in bindings
                if b.channel_type == channel_type
            ]
        return [b.to_dict() for b in bindings]

    # ------------------------------------------------------------------
    # Session Key Derivation
    # ------------------------------------------------------------------

    def derive_session_key(
        self,
        channel_type: ChannelType,
        peer_id: str,
        extra_salt: str = "",
    ) -> str:
        """Derive a consistent session key from channel and peer info.

        Uses SHA-256 hashing to produce a deterministic session key that
        remains the same for the same channel+peer combination, enabling
        conversation continuity.

        Args:
            channel_type: The messaging channel type.
            peer_id: The peer/user identifier.
            extra_salt: Optional additional salt for uniqueness.

        Returns:
            Hex-encoded session key (16 characters).
        """
        cache_key = f"{channel_type.value}:{peer_id}:{extra_salt}"

        if cache_key in self._session_cache:
            return self._session_cache[cache_key]

        raw = f"{channel_type.value}:{peer_id}:{extra_salt}:atlas_session"
        if extra_salt:
            raw = f"{raw}:{extra_salt}"
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        self._session_cache[cache_key] = key
        return key

    def clear_session_cache(self) -> None:
        """Clear the session key cache."""
        self._session_cache.clear()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, message: ChannelMessage) -> RoutingResult:
        """Route a message to the appropriate handler.

        Evaluation order:
        1. Check account bindings for sender → direct agent mapping
        2. Evaluate rules in priority order → first match wins
        3. Fall back to default handler

        Args:
            message: The incoming message to route.

        Returns:
            RoutingResult with decision, handler, session key, etc.
        """
        self._stats["total_routed"] += 1

        # 1. Check account bindings first (highest priority routing)
        binding = self._account_bindings.get(
            (message.channel_type, message.sender)
        )
        if binding is not None:
            self._stats["total_routed"] -= 1  # Adjust for route below
            self._stats["total_routed"] += 1
            result = RoutingResult(
                decision=RoutingDecision.ROUTED,
                handler_name=binding.agent_id,
                session_key=binding.session_key,
                rule_id="account_binding",
                confidence=1.0,
                reason=f"Account {message.sender} bound to agent {binding.agent_id}",
                metadata={"source": "account_binding"},
            )
            self._stats["rule_hits"]["account_binding"] = (
                self._stats["rule_hits"].get("account_binding", 0) + 1
            )
            return result

        # 2. Evaluate rules in priority order
        for rule in self._rules:
            if not rule.enabled:
                continue
            if rule.matches(message):
                rule.match_count += 1
                session_key = self.derive_session_key(
                    message.channel_type, message.sender,
                )
                self._stats["rule_hits"][rule.id] = (
                    self._stats["rule_hits"].get(rule.id, 0) + 1
                )
                return RoutingResult(
                    decision=RoutingDecision.ROUTED,
                    handler_name=rule.handler_name,
                    session_key=session_key,
                    rule_id=rule.id,
                    confidence=0.9 if rule.match_type == RouteMatchType.REGEX else 0.8,
                    reason=f"Matched rule '{rule.name}' ({rule.match_type.value})",
                    metadata={
                        "source": "rule",
                        "rule_name": rule.name,
                        "rule_pattern": rule.pattern,
                    },
                )

        # 3. Fall back to default handler
        session_key = self.derive_session_key(
            message.channel_type, message.sender,
        )

        if self._default_handler and self._default_handler in self._handlers:
            return RoutingResult(
                decision=RoutingDecision.ROUTED,
                handler_name=self._default_handler,
                session_key=session_key,
                rule_id="default",
                confidence=0.3,
                reason="No rules matched, using default handler",
                metadata={"source": "default"},
            )

        # No handler available
        self._stats["total_no_match"] += 1
        return RoutingResult(
            decision=RoutingDecision.NO_MATCH,
            session_key=session_key,
            reason="No matching rules or handlers found",
            metadata={"source": "none"},
        )

    async def route_and_handle(self, message: ChannelMessage) -> RoutingResult:
        """Route a message and invoke the matched handler.

        Args:
            message: The incoming message.

        Returns:
            RoutingResult from the routing decision.
        """
        result = self.route(message)

        if result.decision == RoutingDecision.ROUTED and result.handler_name:
            handler = self._handlers.get(result.handler_name)
            if handler is not None:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message, result)
                    else:
                        handler(message, result)
                except Exception as exc:
                    logger.error(
                        "Handler '%s' error for message %s: %s",
                        result.handler_name, message.id, exc,
                    )
            else:
                logger.warning(
                    "No handler registered for '%s'", result.handler_name,
                )
                result.decision = RoutingDecision.DEFERRED

        return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics.

        Returns:
            Dictionary with routing stats and rule hit counts.
        """
        return {
            "total_routed": self._stats["total_routed"],
            "total_dropped": self._stats["total_dropped"],
            "total_deferred": self._stats["total_deferred"],
            "total_no_match": self._stats["total_no_match"],
            "rule_hits": self._stats["rule_hits"].copy(),
            "rules_count": len(self._rules),
            "enabled_rules_count": sum(1 for r in self._rules if r.enabled),
            "handlers_count": len(self._handlers),
            "bindings_count": len(self._account_bindings),
            "session_cache_size": len(self._session_cache),
        }

    def reset_stats(self) -> None:
        """Reset all routing statistics."""
        for key in self._stats:
            if key == "rule_hits":
                self._stats[key] = {}
            else:
                self._stats[key] = 0
        for rule in self._rules:
            rule.match_count = 0

    # ------------------------------------------------------------------
    # Bulk Operations
    # ------------------------------------------------------------------

    def load_rules(self, rules: List[Dict[str, Any]]) -> int:
        """Load multiple routing rules from a list of dictionaries.

        Args:
            rules: List of rule dictionaries (matching RouteRule.to_dict format).

        Returns:
            Number of rules successfully loaded.
        """
        count = 0
        for rule_data in rules:
            try:
                self.add_route(
                    name=rule_data.get("name", ""),
                    match_type=rule_data.get("match_type", "regex"),
                    pattern=rule_data.get("pattern", ""),
                    handler_name=rule_data.get("handler_name", ""),
                    priority=rule_data.get("priority", 100),
                    description=rule_data.get("description", ""),
                    rule_id=rule_data.get("id"),
                    metadata=rule_data.get("metadata"),
                )
                count += 1
            except Exception as exc:
                logger.error("Failed to load rule %r: %s", rule_data, exc)
        logger.info("Loaded %d routing rules", count)
        return count

    def load_bindings(self, bindings: List[Dict[str, Any]]) -> int:
        """Load multiple account bindings from a list of dictionaries.

        Args:
            bindings: List of binding dictionaries.

        Returns:
            Number of bindings successfully loaded.
        """
        count = 0
        for binding_data in bindings:
            try:
                binding = AccountBinding.from_dict(binding_data)
                self._account_bindings[
                    (binding.channel_type, binding.account_id)
                ] = binding
                count += 1
            except Exception as exc:
                logger.error("Failed to load binding %r: %s", binding_data, exc)
        logger.info("Loaded %d account bindings", count)
        return count

    def __repr__(self) -> str:
        return (
            f"<MessageRouter "
            f"rules={len(self._rules)} "
            f"handlers={len(self._handlers)} "
            f"bindings={len(self._account_bindings)}>"
        )
