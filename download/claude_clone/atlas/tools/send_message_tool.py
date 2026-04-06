"""
Atlas Send Message Tool — cross-platform messaging.

Features:
- Send messages to Telegram, Discord, Slack, etc.
- Message formatting and media attachment
- Delivery status tracking
- Reply threading
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from atlas.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Config and history
# ---------------------------------------------------------------------------

_MSG_CONFIG_PATH = Path.home() / ".claude_clone" / "atlas_messaging.json"


def _load_config() -> Dict[str, Any]:
    if _MSG_CONFIG_PATH.exists():
        try:
            return json.loads(_MSG_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            pass
    return {"platforms": {}, "history": []}


def _save_config(data: Dict[str, Any]) -> None:
    _MSG_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Keep history manageable
    if len(data.get("history", [])) > 500:
        data["history"] = data["history"][-200:]
    _MSG_CONFIG_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Platform senders
# ---------------------------------------------------------------------------

async def _send_telegram(
    bot_token: str,
    chat_id: str,
    message: str,
    parse_mode: str = "Markdown",
) -> Dict[str, Any]:
    """Send a message via Telegram Bot API."""
    import httpx

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


async def _send_discord(
    webhook_url: str,
    message: str,
    username: str = "Atlas",
) -> Dict[str, Any]:
    """Send a message via Discord webhook."""
    import httpx

    payload = {
        "content": message,
        "username": username,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(webhook_url, json=payload)
        resp.raise_for_status()
        return {"status": resp.status_code, "ok": True}


async def _send_slack(
    webhook_url: str,
    message: str,
    channel: str = "",
) -> Dict[str, Any]:
    """Send a message via Slack webhook."""
    import httpx

    payload = {"text": message}
    if channel:
        payload["channel"] = channel

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(webhook_url, json=payload)
        resp.raise_for_status()
        return {"status": resp.status_code, "ok": True}


async def _send_generic_webhook(
    url: str,
    message: str,
    method: str = "POST",
    headers: str = "{}",
) -> Dict[str, Any]:
    """Send a message to a generic webhook URL."""
    import httpx

    try:
        hdrs = json.loads(headers) if isinstance(headers, str) else headers
    except json.JSONDecodeError:
        hdrs = {"Content-Type": "application/json"}

    hdrs.setdefault("Content-Type", "application/json")

    async with httpx.AsyncClient(timeout=15.0) as client:
        if method.upper() == "POST":
            resp = await client.post(url, json={"message": message}, headers=hdrs)
        else:
            resp = await client.get(url, params={"message": message}, headers=hdrs)
        resp.raise_for_status()
        return {"status": resp.status_code, "ok": True}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def atlas_send_message(
    platform: str,
    message: str,
    destination: str = "",
    config: str = "{}",
) -> str:
    """Send a message to a messaging platform.

    param platform (str): — Platform: telegram, discord, slack, webhook.
    param message (str): — Message text to send.
    param destination (str): — Chat ID, webhook URL, or channel name.
    param config (str): — JSON config with credentials (bot_token, etc.).
    """
    try:
        cfg = json.loads(config) if isinstance(config, str) else config
    except json.JSONDecodeError:
        return f"Error: config must be valid JSON."

    platform = platform.lower().strip()

    try:
        if platform == "telegram":
            bot_token = cfg.get("bot_token", "") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = destination or cfg.get("chat_id", "")
            if not bot_token or not chat_id:
                return "Error: Telegram requires bot_token and chat_id (or destination)."

            result = await _send_telegram(bot_token, chat_id, message)
            msg_id = result.get("result", {}).get("message_id", "?")
            return f"Message sent to Telegram (chat {chat_id}, msg_id: {msg_id})"

        elif platform == "discord":
            webhook = destination or cfg.get("webhook_url", "")
            if not webhook:
                return "Error: Discord requires a webhook URL (destination or config.webhook_url)."

            result = await _send_discord(webhook, message)
            return f"Message sent to Discord webhook (status: {result.get('status')})"

        elif platform == "slack":
            webhook = destination or cfg.get("webhook_url", "")
            if not webhook:
                return "Error: Slack requires a webhook URL."

            result = await _send_slack(webhook, message)
            return f"Message sent to Slack (status: {result.get('status')})"

        elif platform == "webhook":
            if not destination:
                return "Error: Webhook requires a destination URL."
            headers_json = json.dumps(cfg.get("headers", {}))
            result = await _send_generic_webhook(destination, message, headers=headers_json)
            return f"Message sent to webhook (status: {result.get('status')})"

        else:
            return f"Error: Unsupported platform '{platform}'. Supported: telegram, discord, slack, webhook."

    except Exception as e:
        return f"Error sending message via {platform}: {e}"


async def atlas_configure_platform(
    platform: str,
    credentials: str,
) -> str:
    """Store credentials for a messaging platform.

    param platform (str): — Platform name (telegram, discord, slack).
    param credentials (str): — JSON credentials (e.g., bot_token, webhook_url).
    """
    try:
        creds = json.loads(credentials) if isinstance(credentials, str) else credentials
    except json.JSONDecodeError:
        return "Error: credentials must be valid JSON."

    def _do():
        cfg = _load_config()
        cfg["platforms"][platform.lower()] = {
            "credentials": creds,
            "configured_at": _now(),
        }
        _save_config(cfg)
        return f"Configured {platform} messaging platform"

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error configuring platform: {e}"


async def atlas_message_history(platform: str = "", limit: int = 20) -> str:
    """View message sending history.

    param platform (str): — Filter by platform. Empty = all.
    param limit (int): — Max entries. Default: 20.
    """
    def _do():
        cfg = _load_config()
        history = cfg.get("history", [])

        if platform:
            history = [h for h in history if h.get("platform") == platform]

        if not history:
            return "No message history found."

        entries = history[-limit:]
        lines = [f"Message history ({len(entries)} most recent):\n"]
        for entry in reversed(entries):
            status = entry.get("status", "?")
            icon = "+" if status == "success" else "!"
            lines.append(
                f"  [{icon}] {entry.get('timestamp', '')[:16]} "
                f"{entry.get('platform', '?')} → {str(entry.get('destination', ''))[:40]}: {status}"
            )

        return "\n".join(lines)

    try:
        return await _run_sync(_do)
    except Exception as e:
        return f"Error reading history: {e}"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="atlas_send_message",
    func=atlas_send_message,
    description="Send a message to Telegram, Discord, Slack, or a generic webhook.",
    toolset="messaging",
)

ToolRegistry.instance().register(
    name="atlas_configure_platform",
    func=atlas_configure_platform,
    description="Store credentials for a messaging platform.",
    toolset="messaging",
)

ToolRegistry.instance().register(
    name="atlas_message_history",
    func=atlas_message_history,
    description="View message sending history with optional platform filter.",
    toolset="messaging",
)
