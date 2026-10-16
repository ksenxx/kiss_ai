# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Registry of every third-party channel agent class, shared by the contract tests.

A new channel agent is added here once and is then covered by
test_base_channel_agent.py and test_channel_agents_no_chat_session.py.
"""

from __future__ import annotations

import importlib

ALL_CHANNEL_AGENTS = [
    ("A2AAgent", "kiss.agents.third_party_agents.a2a.a2a_sea"),
    ("BlueBubblesAgent", "kiss.agents.third_party_agents.bluebubbles.bluebubbles_sea"),
    ("BraveSearchAgent", "kiss.agents.third_party_agents.brave.brave_sea"),
    ("DingTalkAgent", "kiss.agents.third_party_agents.dingtalk.dingtalk_sea"),
    ("DiscordAgent", "kiss.agents.third_party_agents.discord.discord_sea"),
    ("EmailAgent", "kiss.agents.third_party_agents.email.email_sea"),
    ("FeishuAgent", "kiss.agents.third_party_agents.feishu.feishu_sea"),
    ("FirecrawlAgent", "kiss.agents.third_party_agents.firecrawl.firecrawl_sea"),
    ("GitHubAgent", "kiss.agents.third_party_agents.github.github_sea"),
    ("GmailAgent", "kiss.agents.third_party_agents.gmail.gmail_sea"),
    ("GoogleCalendarAgent", "kiss.agents.third_party_agents.gcal.gcal_sea"),
    ("GoogleChatAgent", "kiss.agents.third_party_agents.googlechat.googlechat_sea"),
    ("GoogleDocsAgent", "kiss.agents.third_party_agents.gdocs.gdocs_sea"),
    ("GoogleDriveAgent", "kiss.agents.third_party_agents.gdrive.gdrive_sea"),
    ("GoogleSheetsAgent", "kiss.agents.third_party_agents.gsheets.gsheets_sea"),
    ("HomeAssistantAgent", "kiss.agents.third_party_agents.homeassistant.homeassistant_sea"),
    ("IMessageAgent", "kiss.agents.third_party_agents.imessage.imessage_sea"),
    ("IRCAgent", "kiss.agents.third_party_agents.irc.irc_sea"),
    ("LineAgent", "kiss.agents.third_party_agents.line.line_sea"),
    ("MatrixAgent", "kiss.agents.third_party_agents.matrix.matrix_sea"),
    ("MattermostAgent", "kiss.agents.third_party_agents.mattermost.mattermost_sea"),
    ("MSTeamsAgent", "kiss.agents.third_party_agents.msteams.msteams_sea"),
    ("NextcloudTalkAgent", "kiss.agents.third_party_agents.nextcloud.nextcloud_sea"),
    ("NostrAgent", "kiss.agents.third_party_agents.nostr.nostr_sea"),
    ("NotionAgent", "kiss.agents.third_party_agents.notion.notion_sea"),
    ("NtfyAgent", "kiss.agents.third_party_agents.ntfy.ntfy_sea"),
    ("OpenAICompatAgent", "kiss.agents.third_party_agents.oai.oai_sea"),
    ("OverleafAgent", "kiss.agents.third_party_agents.overleaf.overleaf_sea"),
    ("PhoneControlAgent", "kiss.agents.third_party_agents.phone.phone_sea"),
    ("PostgresAgent", "kiss.agents.third_party_agents.postgres.postgres_sea"),
    ("QQAgent", "kiss.agents.third_party_agents.qq.qq_sea"),
    ("SignalAgent", "kiss.agents.third_party_agents.signal.signal_sea"),
    ("SimpleXAgent", "kiss.agents.third_party_agents.simplex.simplex_sea"),
    ("SlackAgent", "kiss.agents.third_party_agents.slack.slack_sea"),
    ("SMSAgent", "kiss.agents.third_party_agents.sms.sms_sea"),
    ("SynologyChatAgent", "kiss.agents.third_party_agents.synology.synology_sea"),
    ("TelegramAgent", "kiss.agents.third_party_agents.telegram.telegram_sea"),
    ("TlonAgent", "kiss.agents.third_party_agents.tlon.tlon_sea"),
    ("TwitchAgent", "kiss.agents.third_party_agents.twitch.twitch_sea"),
    ("WebhookAgent", "kiss.agents.third_party_agents.webhook.webhook_sea"),
    ("WeComAgent", "kiss.agents.third_party_agents.wecom.wecom_sea"),
    ("WeixinAgent", "kiss.agents.third_party_agents.weixin.weixin_sea"),
    ("WhatsAppAgent", "kiss.agents.third_party_agents.whatsapp.whatsapp_sea"),
    ("ZaloAgent", "kiss.agents.third_party_agents.zalo.zalo_sea"),
]
"""``(class name, module path)`` of every channel agent, one per SEA folder."""

AGENT_IDS = [a[0] for a in ALL_CHANNEL_AGENTS]
"""Parametrize ids matching :data:`ALL_CHANNEL_AGENTS`."""


def get_agent_class(class_name: str, module_path: str) -> type:
    """Import *module_path* and return the agent class named *class_name*."""
    mod = importlib.import_module(module_path)
    cls: type = getattr(mod, class_name)
    return cls
