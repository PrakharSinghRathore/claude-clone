"""
Platform adapters for the Atlas Gateway.

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

from atlas.gateway.platforms.telegram import TelegramAdapter
from atlas.gateway.platforms.discord import DiscordAdapter
from atlas.gateway.platforms.slack import SlackAdapter
from atlas.gateway.platforms.whatsapp import WhatsAppAdapter
from atlas.gateway.platforms.signal import SignalAdapter
from atlas.gateway.platforms.matrix import MatrixAdapter
from atlas.gateway.platforms.email_platform import EmailAdapter
from atlas.gateway.platforms.sms import SMSAdapter
from atlas.gateway.platforms.webhook import WebhookAdapter
from atlas.gateway.platforms.api_server import APIServerAdapter
from atlas.gateway.platforms.dingtalk import DingTalkAdapter
from atlas.gateway.platforms.feishu import FeishuAdapter
from atlas.gateway.platforms.wecom import WeComAdapter
from atlas.gateway.platforms.mattermost import MattermostAdapter
from atlas.gateway.platforms.irc import IRCAdapter
from atlas.gateway.platforms.google_chat import GoogleChatAdapter
from atlas.gateway.platforms.msteams import MSTeamsAdapter
from atlas.gateway.platforms.line import LINEAdapter
from atlas.gateway.platforms.nextcloud import NextcloudAdapter
from atlas.gateway.platforms.nostr import NostrAdapter
from atlas.gateway.platforms.twitch import TwitchAdapter
from atlas.gateway.platforms.zalo import ZaloAdapter
from atlas.gateway.platforms.bluebubbles import BlueBubblesAdapter
from atlas.gateway.platforms.voice_call import VoiceCallAdapter

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
    "IRCAdapter",
    "GoogleChatAdapter",
    "MSTeamsAdapter",
    "LINEAdapter",
    "NextcloudAdapter",
    "NostrAdapter",
    "TwitchAdapter",
    "ZaloAdapter",
    "BlueBubblesAdapter",
    "VoiceCallAdapter",
]

PLATFORM_NAMES: list = [
    "telegram", "discord", "slack", "whatsapp", "signal",
    "matrix", "email", "sms", "webhook", "api",
    "dingtalk", "feishu", "wecom", "mattermost",
    "irc", "google_chat", "msteams", "line",
    "nextcloud", "nostr", "twitch", "zalo",
    "bluebubbles", "voice_call",
]
