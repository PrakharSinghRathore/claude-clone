"""
Agent fingerprinting — unique identity generation for agents.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from typing import Any, Dict, Optional


def generate_fingerprint(
    agent_role: str = "",
    agent_goal: str = "",
    extra_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a unique fingerprint for an agent configuration.
    
    The fingerprint is a SHA-256 hash derived from the agent's role,
    goal, and any extra identifying data. This enables tracking agents
    across sessions without storing personally identifiable information.
    
    Args:
        agent_role: The agent's role string.
        agent_goal: The agent's goal string.
        extra_data: Additional data to include in the fingerprint.
    
    Returns:
        A hex-encoded SHA-256 fingerprint string.
    """
    payload = {
        "role": agent_role,
        "goal": agent_goal,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "extra": extra_data or {},
    }
    
    payload_str = json.dumps(payload, sort_keys=True)
    fingerprint = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
    return fingerprint


def verify_fingerprint(
    fingerprint: str,
    agent_role: str = "",
    agent_goal: str = "",
    extra_data: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Verify if a fingerprint matches the given agent configuration.
    
    Args:
        fingerprint: The fingerprint to verify.
        agent_role: The expected agent role.
        agent_goal: The expected agent goal.
        extra_data: Additional expected data.
    
    Returns:
        True if the fingerprint matches.
    """
    expected = generate_fingerprint(agent_role, agent_goal, extra_data)
    return fingerprint == expected
