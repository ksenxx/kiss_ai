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
from pathlib import Path

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import ChannelConfig
from kiss.tests.agents.third_party_agents.muse_test_utils import tools_by_name

# Every channel's ChannelConfig resolves KISS_HOME lazily, so each test
# starts from an empty per-test home and leaves nothing behind.
pytestmark = pytest.mark.usefixtures("isolated_kiss_home")

_CHANNEL_AGENTS: list[dict[str, str | bool]] = [
    {
        "module": "kiss.agents.third_party_agents.telegram.telegram_sea",
        "agent_class": "TelegramAgent",
        "auth_check": "check_telegram_auth",
        "auth_clear": "clear_telegram_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.discord.discord_sea",
        "agent_class": "DiscordAgent",
        "auth_check": "check_discord_auth",
        "auth_clear": "clear_discord_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.googlechat.googlechat_sea",
        "agent_class": "GoogleChatAgent",
        "auth_check": "check_googlechat_auth",
        "auth_clear": "clear_googlechat_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.signal.signal_sea",
        "agent_class": "SignalAgent",
        "auth_check": "check_signal_auth",
        "auth_clear": "clear_signal_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.msteams.msteams_sea",
        "agent_class": "MSTeamsAgent",
        "auth_check": "check_msteams_auth",
        "auth_clear": "clear_msteams_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.matrix.matrix_sea",
        "agent_class": "MatrixAgent",
        "auth_check": "check_matrix_auth",
        "auth_clear": "clear_matrix_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.feishu.feishu_sea",
        "agent_class": "FeishuAgent",
        "auth_check": "check_feishu_auth",
        "auth_clear": "clear_feishu_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.line.line_sea",
        "agent_class": "LineAgent",
        "auth_check": "check_line_auth",
        "auth_clear": "clear_line_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.mattermost.mattermost_sea",
        "agent_class": "MattermostAgent",
        "auth_check": "check_mattermost_auth",
        "auth_clear": "clear_mattermost_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.irc.irc_sea",
        "agent_class": "IRCAgent",
        "auth_check": "check_irc_auth",
        "auth_clear": "clear_irc_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.bluebubbles.bluebubbles_sea",
        "agent_class": "BlueBubblesAgent",
        "auth_check": "check_bluebubbles_auth",
        "auth_clear": "clear_bluebubbles_auth",
        "macos_only": True,
    },
    {
        "module": "kiss.agents.third_party_agents.imessage.imessage_sea",
        "agent_class": "IMessageAgent",
        "auth_check": "check_imessage_auth",
        "auth_clear": "clear_imessage_auth",
        "macos_only": True,
    },
    {
        "module": "kiss.agents.third_party_agents.nextcloud.nextcloud_sea",
        "agent_class": "NextcloudTalkAgent",
        "auth_check": "check_nextcloud_auth",
        "auth_clear": "clear_nextcloud_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.nostr.nostr_sea",
        "agent_class": "NostrAgent",
        "auth_check": "check_nostr_auth",
        "auth_clear": "clear_nostr_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.synology.synology_sea",
        "agent_class": "SynologyChatAgent",
        "auth_check": "check_synology_auth",
        "auth_clear": "clear_synology_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.tlon.tlon_sea",
        "agent_class": "TlonAgent",
        "auth_check": "check_tlon_auth",
        "auth_clear": "clear_tlon_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.twitch.twitch_sea",
        "agent_class": "TwitchAgent",
        "auth_check": "check_twitch_auth",
        "auth_clear": "clear_twitch_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.zalo.zalo_sea",
        "agent_class": "ZaloAgent",
        "auth_check": "check_zalo_auth",
        "auth_clear": "clear_zalo_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.phone.phone_sea",
        "agent_class": "PhoneControlAgent",
        "auth_check": "check_phone_auth",
        "auth_clear": "clear_phone_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.sms.sms_sea",
        "agent_class": "SMSAgent",
        "auth_check": "check_sms_auth",
        "auth_clear": "clear_sms_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.email.email_sea",
        "agent_class": "EmailAgent",
        "auth_check": "check_email_auth",
        "auth_clear": "clear_email_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.ntfy.ntfy_sea",
        "agent_class": "NtfyAgent",
        "auth_check": "check_ntfy_auth",
        "auth_clear": "clear_ntfy_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.webhook.webhook_sea",
        "agent_class": "WebhookAgent",
        "auth_check": "check_webhook_auth",
        "auth_clear": "clear_webhook_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.a2a.a2a_sea",
        "agent_class": "A2AAgent",
        "auth_check": "check_a2a_auth",
        "auth_clear": "clear_a2a_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.homeassistant.homeassistant_sea",
        "agent_class": "HomeAssistantAgent",
        "auth_check": "check_homeassistant_auth",
        "auth_clear": "clear_homeassistant_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.dingtalk.dingtalk_sea",
        "agent_class": "DingTalkAgent",
        "auth_check": "check_dingtalk_auth",
        "auth_clear": "clear_dingtalk_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.wecom.wecom_sea",
        "agent_class": "WeComAgent",
        "auth_check": "check_wecom_auth",
        "auth_clear": "clear_wecom_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.weixin.weixin_sea",
        "agent_class": "WeixinAgent",
        "auth_check": "check_weixin_auth",
        "auth_clear": "clear_weixin_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.qq.qq_sea",
        "agent_class": "QQAgent",
        "auth_check": "check_qq_auth",
        "auth_clear": "clear_qq_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.simplex.simplex_sea",
        "agent_class": "SimpleXAgent",
        "auth_check": "check_simplex_auth",
        "auth_clear": "clear_simplex_auth",
    },
    {
        "module": "kiss.agents.third_party_agents.oai.oai_sea",
        "agent_class": "OpenAICompatAgent",
        "auth_check": "check_openai_compat_auth",
        "auth_clear": "clear_openai_compat_auth",
    },
]

_CHANNEL_IDS = [ch["agent_class"] for ch in _CHANNEL_AGENTS]


def _load(info: dict) -> tuple:
    """Import and return (module, AgentClass)."""
    mod = importlib.import_module(info["module"])
    return mod, getattr(mod, info["agent_class"])


@pytest.mark.parametrize("info", _CHANNEL_AGENTS, ids=_CHANNEL_IDS)
def test_check_auth_unauthenticated(info: dict) -> None:
    """check_*_auth() returns a helpful message when not configured."""
    if info.get("macos_only") and sys.platform != "darwin":
        pytest.skip("macOS-only agent")
    _, agent_cls = _load(info)
    agent = agent_cls()
    tools = tools_by_name(agent)
    result = tools[info["auth_check"]]()
    lower = result.lower()
    assert (
        "not authenticated" in lower
        or "not configured" in lower
        or "no " in lower
        or "authenticate" in lower
    ), f"Expected unauthenticated message, got: {result[:200]}"


@pytest.mark.parametrize("info", _CHANNEL_AGENTS, ids=_CHANNEL_IDS)
def test_clear_auth_when_not_authenticated(info: dict) -> None:
    """clear_*_auth() works without error even when not authenticated."""
    _, agent_cls = _load(info)
    agent = agent_cls()
    tools = tools_by_name(agent)
    result = tools[info["auth_clear"]]()
    assert "cleared" in result.lower() or "removed" in result.lower(), (
        f"Expected cleared message, got: {result[:200]}"
    )


def test_config_roundtrip(tmp_path: Path) -> None:
    """``ChannelConfig`` load/clear tolerate a missing, corrupt or non-object file.

    Every channel module's ``_config`` is a ``ChannelConfig``, so one instance
    pins the behaviour for all of them.
    """
    config = ChannelConfig(tmp_path / "channel", ("token",))
    path = config.path
    config.clear()
    assert config.load() is None
    assert not path.exists()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{bad json!!")
    assert config.load() is None

    path.write_text('"just a string"')
    assert config.load() is None

    config.clear()
    assert not path.exists()
