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

EXPECTED_AUTH_TOOLS = {
    "A2AAgent": ["check_a2a_auth", "authenticate_a2a", "clear_a2a_auth"],
    "BlueBubblesAgent": [
        "check_bluebubbles_auth",
        "authenticate_bluebubbles",
        "clear_bluebubbles_auth",
    ],
    "BraveSearchAgent": [
        "check_brave_search_auth",
        "authenticate_brave_search",
        "clear_brave_search_auth",
    ],
    "DingTalkAgent": ["check_dingtalk_auth", "authenticate_dingtalk", "clear_dingtalk_auth"],
    "DiscordAgent": [
        "check_discord_auth",
        "authenticate_discord",
        "finish_discord_auth",
        "clear_discord_auth",
    ],
    "EmailAgent": ["check_email_auth", "authenticate_email", "clear_email_auth"],
    "FeishuAgent": ["check_feishu_auth", "authenticate_feishu", "clear_feishu_auth"],
    "FirecrawlAgent": ["check_firecrawl_auth", "authenticate_firecrawl", "clear_firecrawl_auth"],
    "GitHubAgent": [
        "check_github_auth",
        "authenticate_github",
        "finish_github_auth",
        "clear_github_auth",
    ],
    "GmailAgent": [
        "check_gmail_auth",
        "authenticate_gmail",
        "clear_gmail_auth",
        "finish_gmail_auth",
    ],
    "GoogleCalendarAgent": [
        "check_google_calendar_auth",
        "authenticate_google_calendar",
        "clear_google_calendar_auth",
        "finish_google_calendar_auth",
    ],
    "GoogleChatAgent": [
        "check_googlechat_auth",
        "authenticate_googlechat",
        "clear_googlechat_auth",
        "finish_googlechat_auth",
        "authenticate_googlechat_service_account",
    ],
    "GoogleDocsAgent": [
        "check_google_docs_auth",
        "authenticate_google_docs",
        "clear_google_docs_auth",
        "finish_google_docs_auth",
    ],
    "GoogleDriveAgent": [
        "check_google_drive_auth",
        "authenticate_google_drive",
        "clear_google_drive_auth",
        "finish_google_drive_auth",
    ],
    "GoogleSheetsAgent": [
        "check_google_sheets_auth",
        "authenticate_google_sheets",
        "clear_google_sheets_auth",
        "finish_google_sheets_auth",
    ],
    "HomeAssistantAgent": [
        "check_homeassistant_auth",
        "authenticate_homeassistant",
        "clear_homeassistant_auth",
    ],
    "IMessageAgent": ["check_imessage_auth", "authenticate_imessage", "clear_imessage_auth"],
    "IRCAgent": ["check_irc_auth", "authenticate_irc", "clear_irc_auth"],
    "LineAgent": ["check_line_auth", "authenticate_line", "clear_line_auth"],
    "MatrixAgent": [
        "check_matrix_auth",
        "authenticate_matrix",
        "finish_matrix_auth",
        "clear_matrix_auth",
    ],
    "MattermostAgent": [
        "check_mattermost_auth",
        "authenticate_mattermost",
        "clear_mattermost_auth",
    ],
    "MSTeamsAgent": [
        "check_msteams_auth",
        "authenticate_msteams",
        "finish_msteams_auth",
        "clear_msteams_auth",
    ],
    "NextcloudTalkAgent": [
        "check_nextcloud_auth",
        "authenticate_nextcloud",
        "finish_nextcloud_auth",
        "clear_nextcloud_auth",
    ],
    "NostrAgent": ["check_nostr_auth", "authenticate_nostr", "clear_nostr_auth"],
    "NotionAgent": ["check_notion_auth", "authenticate_notion", "clear_notion_auth"],
    "NtfyAgent": ["check_ntfy_auth", "authenticate_ntfy", "clear_ntfy_auth"],
    "OpenAICompatAgent": [
        "check_openai_compat_auth",
        "authenticate_openai_compat",
        "clear_openai_compat_auth",
    ],
    "OverleafAgent": ["check_overleaf_auth", "authenticate_overleaf", "clear_overleaf_auth"],
    "PhoneControlAgent": ["check_phone_auth", "authenticate_phone", "clear_phone_auth"],
    "PostgresAgent": ["check_postgres_auth", "authenticate_postgres", "clear_postgres_auth"],
    "QQAgent": ["check_qq_auth", "authenticate_qq", "clear_qq_auth"],
    "SignalAgent": [
        "check_signal_auth",
        "authenticate_signal",
        "finish_signal_auth",
        "clear_signal_auth",
    ],
    "SimpleXAgent": ["check_simplex_auth", "authenticate_simplex", "clear_simplex_auth"],
    "SlackAgent": [
        "check_slack_auth",
        "authenticate_slack",
        "finish_slack_auth",
        "clear_slack_auth",
    ],
    "SMSAgent": ["check_sms_auth", "authenticate_sms", "clear_sms_auth"],
    "SynologyChatAgent": ["check_synology_auth", "authenticate_synology", "clear_synology_auth"],
    "TelegramAgent": ["check_telegram_auth", "authenticate_telegram", "clear_telegram_auth"],
    "TlonAgent": ["check_tlon_auth", "authenticate_tlon", "clear_tlon_auth"],
    "TwitchAgent": [
        "check_twitch_auth",
        "authenticate_twitch",
        "finish_twitch_auth",
        "clear_twitch_auth",
    ],
    "WebhookAgent": ["check_webhook_auth", "authenticate_webhook", "clear_webhook_auth"],
    "WeComAgent": ["check_wecom_auth", "authenticate_wecom", "clear_wecom_auth"],
    "WeixinAgent": ["check_weixin_auth", "authenticate_weixin", "clear_weixin_auth"],
    "WhatsAppAgent": [
        "check_whatsapp_auth",
        "authenticate_whatsapp",
        "start_whatsapp_bridge",
        "get_whatsapp_qr_code",
        "wait_for_whatsapp_pairing",
        "stop_whatsapp_bridge",
        "clear_whatsapp_auth",
    ],
    "ZaloAgent": ["check_zalo_auth", "authenticate_zalo", "clear_zalo_auth"],
}
"""Auth tool names, in order, of every agent in :data:`ALL_CHANNEL_AGENTS` (an empty ``KISS_HOME``).

The agent's ``_get_auth_tools()`` and its module's SEA ``tools([])`` both expose exactly these.
"""


def get_agent_class(class_name: str, module_path: str) -> type:
    """Import *module_path* and return the agent class named *class_name*."""
    mod = importlib.import_module(module_path)
    cls: type = getattr(mod, class_name)
    return cls
