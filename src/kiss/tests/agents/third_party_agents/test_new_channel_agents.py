# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for all new channel agents — no mocks or test doubles.

Tests agent instantiation, auth trio presence, config persistence,
and backend tool discovery for every new channel agent created by
the OpenClaw port.
"""

from __future__ import annotations

import importlib
import sys

import pytest

_CHANNEL_AGENTS: list[dict[str, str | bool]] = [
    {
        "module": "kiss.agents.third_party_agents.telegram.telegram_sea",
        "sea_class": "TelegramSea",
        "agent_class": "TelegramAgent",
        "backend_class": "TelegramChannelBackend",
        "agent_name": "Telegram Agent",
        "auth_check": "check_telegram_auth",
        "auth_set": "authenticate_telegram",
        "auth_clear": "clear_telegram_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.discord.discord_sea",
        "sea_class": "DiscordSea",
        "agent_class": "DiscordAgent",
        "backend_class": "DiscordChannelBackend",
        "agent_name": "Discord Agent",
        "auth_check": "check_discord_auth",
        "auth_set": "authenticate_discord",
        "auth_clear": "clear_discord_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.googlechat.googlechat_sea",
        "sea_class": "GooglechatSea",
        "agent_class": "GoogleChatAgent",
        "backend_class": "GoogleChatChannelBackend",
        "agent_name": "Google Chat Agent",
        "auth_check": "check_googlechat_auth",
        "auth_set": "authenticate_googlechat",
        "auth_clear": "clear_googlechat_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.signal.signal_sea",
        "sea_class": "SignalSea",
        "agent_class": "SignalAgent",
        "backend_class": "SignalChannelBackend",
        "agent_name": "Signal Agent",
        "auth_check": "check_signal_auth",
        "auth_set": "authenticate_signal",
        "auth_clear": "clear_signal_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.msteams.msteams_sea",
        "sea_class": "MsteamsSea",
        "agent_class": "MSTeamsAgent",
        "backend_class": "MSTeamsChannelBackend",
        "agent_name": "MS Teams Agent",
        "auth_check": "check_msteams_auth",
        "auth_set": "authenticate_msteams",
        "auth_clear": "clear_msteams_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.matrix.matrix_sea",
        "sea_class": "MatrixSea",
        "agent_class": "MatrixAgent",
        "backend_class": "MatrixChannelBackend",
        "agent_name": "Matrix Agent",
        "auth_check": "check_matrix_auth",
        "auth_set": "authenticate_matrix",
        "auth_clear": "clear_matrix_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.feishu.feishu_sea",
        "sea_class": "FeishuSea",
        "agent_class": "FeishuAgent",
        "backend_class": "FeishuChannelBackend",
        "agent_name": "Feishu Agent",
        "auth_check": "check_feishu_auth",
        "auth_set": "authenticate_feishu",
        "auth_clear": "clear_feishu_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.line.line_sea",
        "sea_class": "LineSea",
        "agent_class": "LineAgent",
        "backend_class": "LineChannelBackend",
        "agent_name": "LINE Agent",
        "auth_check": "check_line_auth",
        "auth_set": "authenticate_line",
        "auth_clear": "clear_line_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.mattermost.mattermost_sea",
        "sea_class": "MattermostSea",
        "agent_class": "MattermostAgent",
        "backend_class": "MattermostChannelBackend",
        "agent_name": "Mattermost Agent",
        "auth_check": "check_mattermost_auth",
        "auth_set": "authenticate_mattermost",
        "auth_clear": "clear_mattermost_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.irc.irc_sea",
        "sea_class": "IrcSea",
        "agent_class": "IRCAgent",
        "backend_class": "IRCChannelBackend",
        "agent_name": "IRC Agent",
        "auth_check": "check_irc_auth",
        "auth_set": "authenticate_irc",
        "auth_clear": "clear_irc_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.bluebubbles.bluebubbles_sea",
        "sea_class": "BluebubblesSea",
        "agent_class": "BlueBubblesAgent",
        "backend_class": "BlueBubblesChannelBackend",
        "agent_name": "BlueBubbles Agent",
        "auth_check": "check_bluebubbles_auth",
        "auth_set": "authenticate_bluebubbles",
        "auth_clear": "clear_bluebubbles_auth",
        "macos_only": True,
    },
    {
        "module": "kiss.agents.third_party_agents.imessage.imessage_sea",
        "sea_class": "ImessageSea",
        "agent_class": "IMessageAgent",
        "backend_class": "IMessageChannelBackend",
        "agent_name": "iMessage Agent",
        "auth_check": "check_imessage_auth",
        "auth_set": "authenticate_imessage",
        "auth_clear": "clear_imessage_auth",
        "macos_only": True,
    },
    {
        "module": "kiss.agents.third_party_agents.nextcloud.nextcloud_sea",
        "sea_class": "NextcloudSea",
        "agent_class": "NextcloudTalkAgent",
        "backend_class": "NextcloudTalkChannelBackend",
        "agent_name": "Nextcloud Talk Agent",
        "auth_check": "check_nextcloud_auth",
        "auth_set": "authenticate_nextcloud",
        "auth_clear": "clear_nextcloud_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.nostr.nostr_sea",
        "sea_class": "NostrSea",
        "agent_class": "NostrAgent",
        "backend_class": "NostrChannelBackend",
        "agent_name": "Nostr Agent",
        "auth_check": "check_nostr_auth",
        "auth_set": "authenticate_nostr",
        "auth_clear": "clear_nostr_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.synology.synology_sea",
        "sea_class": "SynologySea",
        "agent_class": "SynologyChatAgent",
        "backend_class": "SynologyChatChannelBackend",
        "agent_name": "Synology Chat Agent",
        "auth_check": "check_synology_auth",
        "auth_set": "authenticate_synology",
        "auth_clear": "clear_synology_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.tlon.tlon_sea",
        "sea_class": "TlonSea",
        "agent_class": "TlonAgent",
        "backend_class": "TlonChannelBackend",
        "agent_name": "Tlon Agent",
        "auth_check": "check_tlon_auth",
        "auth_set": "authenticate_tlon",
        "auth_clear": "clear_tlon_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.twitch.twitch_sea",
        "sea_class": "TwitchSea",
        "agent_class": "TwitchAgent",
        "backend_class": "TwitchChannelBackend",
        "agent_name": "Twitch Agent",
        "auth_check": "check_twitch_auth",
        "auth_set": "authenticate_twitch",
        "auth_clear": "clear_twitch_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.zalo.zalo_sea",
        "sea_class": "ZaloSea",
        "agent_class": "ZaloAgent",
        "backend_class": "ZaloChannelBackend",
        "agent_name": "Zalo Agent",
        "auth_check": "check_zalo_auth",
        "auth_set": "authenticate_zalo",
        "auth_clear": "clear_zalo_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.phone.phone_sea",
        "sea_class": "PhoneSea",
        "agent_class": "PhoneControlAgent",
        "backend_class": "PhoneControlChannelBackend",
        "agent_name": "Phone Control Agent",
        "auth_check": "check_phone_auth",
        "auth_set": "authenticate_phone",
        "auth_clear": "clear_phone_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.sms.sms_sea",
        "sea_class": "SmsSea",
        "agent_class": "SMSAgent",
        "backend_class": "SMSChannelBackend",
        "agent_name": "SMS Agent",
        "auth_check": "check_sms_auth",
        "auth_set": "authenticate_sms",
        "auth_clear": "clear_sms_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.email.email_sea",
        "sea_class": "EmailSea",
        "agent_class": "EmailAgent",
        "backend_class": "EmailChannelBackend",
        "agent_name": "Email Agent",
        "auth_check": "check_email_auth",
        "auth_set": "authenticate_email",
        "auth_clear": "clear_email_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.ntfy.ntfy_sea",
        "sea_class": "NtfySea",
        "agent_class": "NtfyAgent",
        "backend_class": "NtfyChannelBackend",
        "agent_name": "Ntfy Agent",
        "auth_check": "check_ntfy_auth",
        "auth_set": "authenticate_ntfy",
        "auth_clear": "clear_ntfy_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.webhook.webhook_sea",
        "sea_class": "WebhookSea",
        "agent_class": "WebhookAgent",
        "backend_class": "WebhookChannelBackend",
        "agent_name": "Webhook Agent",
        "auth_check": "check_webhook_auth",
        "auth_set": "authenticate_webhook",
        "auth_clear": "clear_webhook_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.a2a.a2a_sea",
        "sea_class": "A2aSea",
        "agent_class": "A2AAgent",
        "backend_class": "A2AChannelBackend",
        "agent_name": "A2A Agent",
        "auth_check": "check_a2a_auth",
        "auth_set": "authenticate_a2a",
        "auth_clear": "clear_a2a_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.homeassistant.homeassistant_sea",
        "sea_class": "HomeassistantSea",
        "agent_class": "HomeAssistantAgent",
        "backend_class": "HomeAssistantChannelBackend",
        "agent_name": "Home Assistant Agent",
        "auth_check": "check_homeassistant_auth",
        "auth_set": "authenticate_homeassistant",
        "auth_clear": "clear_homeassistant_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.dingtalk.dingtalk_sea",
        "sea_class": "DingtalkSea",
        "agent_class": "DingTalkAgent",
        "backend_class": "DingTalkChannelBackend",
        "agent_name": "DingTalk Agent",
        "auth_check": "check_dingtalk_auth",
        "auth_set": "authenticate_dingtalk",
        "auth_clear": "clear_dingtalk_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.wecom.wecom_sea",
        "sea_class": "WecomSea",
        "agent_class": "WeComAgent",
        "backend_class": "WeComChannelBackend",
        "agent_name": "WeCom Agent",
        "auth_check": "check_wecom_auth",
        "auth_set": "authenticate_wecom",
        "auth_clear": "clear_wecom_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.weixin.weixin_sea",
        "sea_class": "WeixinSea",
        "agent_class": "WeixinAgent",
        "backend_class": "WeixinChannelBackend",
        "agent_name": "Weixin Agent",
        "auth_check": "check_weixin_auth",
        "auth_set": "authenticate_weixin",
        "auth_clear": "clear_weixin_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.qq.qq_sea",
        "sea_class": "QqSea",
        "agent_class": "QQAgent",
        "backend_class": "QQChannelBackend",
        "agent_name": "QQ Agent",
        "auth_check": "check_qq_auth",
        "auth_set": "authenticate_qq",
        "auth_clear": "clear_qq_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.simplex.simplex_sea",
        "sea_class": "SimplexSea",
        "agent_class": "SimpleXAgent",
        "backend_class": "SimpleXChannelBackend",
        "agent_name": "SimpleX Agent",
        "auth_check": "check_simplex_auth",
        "auth_set": "authenticate_simplex",
        "auth_clear": "clear_simplex_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.oai.oai_sea",
        "sea_class": "OaiSea",
        "agent_class": "OpenAICompatAgent",
        "backend_class": "OpenAICompatChannelBackend",
        "agent_name": "OpenAI-compatible API Agent",
        "auth_check": "check_openai_compat_auth",
        "auth_set": "authenticate_openai_compat",
        "auth_clear": "clear_openai_compat_auth",
    },
]

_CHANNEL_IDS = [ch["agent_class"] for ch in _CHANNEL_AGENTS]

_TOOL_SEAS: list[tuple[str, str, set[str]]] = [
    (
        str(ch["module"]),
        str(ch["sea_class"]),
        {str(ch["auth_check"]), str(ch["auth_set"]), str(ch["auth_clear"])},
    )
    for ch in _CHANNEL_AGENTS
] + [
    (
        "kiss.agents.third_party_agents.notion.notion_sea",
        "NotionSea",
        {"check_notion_auth", "authenticate_notion", "clear_notion_auth"},
    ),
    (
        "kiss.agents.third_party_agents.postgres.postgres_sea",
        "PostgresSea",
        {"check_postgres_auth", "authenticate_postgres", "clear_postgres_auth"},
    ),
]
"""``(module, SEA class, required tool names)`` of each SEA whose ``tools()`` adds channel tools."""


@pytest.mark.parametrize("module,sea_class,auth_tools", _TOOL_SEAS, ids=[s[1] for s in _TOOL_SEAS])
def test_tools_module_function(module: str, sea_class: str, auth_tools: set[str]) -> None:
    """``<Sea>().tools([])`` returns callables including the check/authenticate/clear auth trio."""
    sea_cls = getattr(importlib.import_module(module), sea_class)
    tools = sea_cls().tools([])
    assert all(callable(t) for t in tools)
    assert auth_tools <= {t.__name__ for t in tools}
    assert sea_cls.tools.__doc__


def _load(info: dict) -> tuple:
    """Import and return (module, AgentClass, BackendClass)."""
    mod = importlib.import_module(info["module"])
    agent_cls = getattr(mod, info["agent_class"])
    backend_cls = getattr(mod, info["backend_class"])
    return mod, agent_cls, backend_cls


def _reset_config(mod) -> None:
    """Clear any leftover persisted config so the agent starts unauthenticated.

    Every channel agent module exposes its ``ChannelConfig`` as the
    module-level ``_config``; googlechat keeps no config and is skipped.
    """
    config = getattr(mod, "_config", None)
    if config is not None:
        config.clear()


@pytest.mark.parametrize("info", _CHANNEL_AGENTS, ids=_CHANNEL_IDS)
def test_check_auth_unauthenticated(info: dict) -> None:
    """check_*_auth() returns a helpful message when not configured."""
    if info.get("macos_only") and sys.platform != "darwin":
        pytest.skip("macOS-only agent")
    mod, agent_cls, _ = _load(info)
    _reset_config(mod)
    agent = agent_cls()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = tools[info["auth_check"]]()
    lower = result.lower()
    assert ("not authenticated" in lower
            or "not configured" in lower
            or "no " in lower
            or "authenticate" in lower), (
        f"Expected unauthenticated message, got: {result[:200]}"
    )


@pytest.mark.parametrize("info", _CHANNEL_AGENTS, ids=_CHANNEL_IDS)
def test_clear_auth_when_not_authenticated(info: dict) -> None:
    """clear_*_auth() works without error even when not authenticated."""
    mod, agent_cls, _ = _load(info)
    _reset_config(mod)
    agent = agent_cls()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = tools[info["auth_clear"]]()
    assert "cleared" in result.lower() or "removed" in result.lower(), (
        f"Expected cleared message, got: {result[:200]}"
    )


@pytest.mark.parametrize("info", _CHANNEL_AGENTS, ids=_CHANNEL_IDS)
def test_config_roundtrip(info: dict) -> None:
    """Config save/load/clear works on real filesystem."""
    mod, _, _ = _load(info)
    config = getattr(mod, "_config", None)
    if config is None:
        pytest.skip(f"{info['module']} keeps no module-level ChannelConfig")
    path = config.path
    load_fn = config.load
    clear_fn = config.clear
    backup = None
    if path.exists():
        backup = path.read_text()

    try:
        clear_fn()
        assert load_fn() is None

        if path.exists():
            path.unlink()
        assert load_fn() is None

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad json!!")
        assert load_fn() is None

        path.write_text('"just a string"')
        assert load_fn() is None

        if path.exists():
            path.unlink()
        clear_fn()

    finally:
        if backup is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(backup)
        elif path.exists():
            path.unlink()


@pytest.mark.parametrize("info", _CHANNEL_AGENTS, ids=_CHANNEL_IDS)
def test_main_exits_with_no_args(info: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() exits when called with no arguments."""
    mod, _, _ = _load(info)
    main_fn = getattr(mod, "main", None)
    if main_fn is None:
        pytest.skip(f"No main() in {info['module']}")

    monkeypatch.setattr(sys, "argv", ["test_agent"])
    with pytest.raises(SystemExit):
        main_fn()
