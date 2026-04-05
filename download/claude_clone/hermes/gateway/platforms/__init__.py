"""
Platform adapters for the Hermes Gateway.

Each adapter implements a common interface for connecting to, sending messages
through, and receiving updates from a messaging platform.

Common interface:
    - connect() — Start listening
    - disconnect() — Stop listening
    - send_message(chat_id, text, **kwargs) — Send message
    - send_file(chat_id, file_path, **kwargs) — Send file/attachment
    - get_updates() — Poll for new messages
    - is_connected() — Check connection status
"""

from hermes.gateway.platforms.telegram import TelegramAdapter
from hermes.gateway.platforms.discord import DiscordAdapter
from hermes.gateway.platforms.slack import SlackAdapter
from hermes.gateway.platforms.whatsapp import WhatsAppAdapter
from hermes.gateway.platforms.signal import SignalAdapter
from hermes.gateway.platforms.matrix import MatrixAdapter
from hermes.gateway.platforms.email_platform import EmailAdapter
from hermes.gateway.platforms.sms import SMSAdapter
from hermes.gateway.platforms.webhook import WebhookAdapter
from hermes.gateway.platforms.api_server import APIServerAdapter
from hermes.gateway.platforms.dingtalk import DingTalkAdapter
from hermes.gateway.platforms.feishu import FeishuAdapter
from hermes.gateway.platforms.wecom import WeComAdapter
from hermes.gateway.platforms.mattermost import MattermostAdapter

__all__ = [
    "TelegramAdapter",
    "DiscordAdapter",
    "SlackAdapter",
    "WhatsAppAdapter",
    "SignalAdapter",
    "MatrixAdapter",
    "EmailAdapter",
    "SMSAdapter",
    "WebhookAdapter",
    "APIServerAdapter",
    "DingTalkAdapter",
    "FeishuAdapter",
    "WeComAdapter",
    "MattermostAdapter",
]

PLATFORM_NAMES: list = [
    "telegram", "discord", "slack", "whatsapp", "signal",
    "matrix", "email", "sms", "webhook", "api",
    "dingtalk", "feishu", "wecom", "mattermost",
]
