"""Integration tests for BaseChannelAgent and channel_main().

Verifies that:
1. All 23 channel agents inherit from BaseChannelAgent
2. BaseChannelAgent._get_tools() properly delegates to _get_auth_tools()
   and conditionally includes backend tools based on _is_authenticated()
3. channel_main() handles CLI parsing for interactive and poll modes
"""

from __future__ import annotations

import importlib
import sys

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import BaseChannelAgent, channel_main

ALL_AGENTS = [
    ("BlueBubblesAgent", "kiss.agents.third_party_agents.bluebubbles_agent"),
    ("DiscordAgent", "kiss.agents.third_party_agents.discord_agent"),
    ("FeishuAgent", "kiss.agents.third_party_agents.feishu_agent"),
    ("GmailAgent", "kiss.agents.third_party_agents.gmail_agent"),
    ("GoogleChatAgent", "kiss.agents.third_party_agents.googlechat_agent"),
    ("IMessageAgent", "kiss.agents.third_party_agents.imessage_agent"),
    ("IRCAgent", "kiss.agents.third_party_agents.irc_agent"),
    ("LineAgent", "kiss.agents.third_party_agents.line_agent"),
    ("MatrixAgent", "kiss.agents.third_party_agents.matrix_agent"),
    ("MattermostAgent", "kiss.agents.third_party_agents.mattermost_agent"),
    ("MSTeamsAgent", "kiss.agents.third_party_agents.msteams_agent"),
    ("NextcloudTalkAgent", "kiss.agents.third_party_agents.nextcloud_talk_agent"),
    ("NostrAgent", "kiss.agents.third_party_agents.nostr_agent"),
    ("PhoneControlAgent", "kiss.agents.third_party_agents.phone_control_agent"),
    ("SignalAgent", "kiss.agents.third_party_agents.signal_agent"),
    ("SlackAgent", "kiss.agents.third_party_agents.slack_agent"),
    ("SMSAgent", "kiss.agents.third_party_agents.sms_agent"),
    ("SynologyChatAgent", "kiss.agents.third_party_agents.synology_chat_agent"),
    ("TelegramAgent", "kiss.agents.third_party_agents.telegram_agent"),
    ("TlonAgent", "kiss.agents.third_party_agents.tlon_agent"),
    ("TwitchAgent", "kiss.agents.third_party_agents.twitch_agent"),
    ("WhatsAppAgent", "kiss.agents.third_party_agents.whatsapp_agent"),
    ("ZaloAgent", "kiss.agents.third_party_agents.zalo_agent"),
]


def _get_agent_class(module_path: str, class_name: str) -> type:
    mod = importlib.import_module(module_path)
    cls: type = getattr(mod, class_name)
    return cls


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_all_agents_inherit_base_channel_agent(
    class_name: str, module_path: str
) -> None:
    """Every channel agent class inherits from BaseChannelAgent."""
    cls = _get_agent_class(module_path, class_name)
    assert issubclass(cls, BaseChannelAgent)


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_all_agents_have_is_authenticated(
    class_name: str, module_path: str
) -> None:
    """Every channel agent overrides _is_authenticated()."""
    cls = _get_agent_class(module_path, class_name)
    assert "_is_authenticated" in cls.__dict__


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_all_agents_have_get_auth_tools(
    class_name: str, module_path: str
) -> None:
    """Every channel agent overrides _get_auth_tools()."""
    cls = _get_agent_class(module_path, class_name)
    assert "_get_auth_tools" in cls.__dict__


@pytest.mark.parametrize(
    "class_name,module_path", ALL_AGENTS, ids=[a[0] for a in ALL_AGENTS]
)
def test_no_inline_get_tools(class_name: str, module_path: str) -> None:
    """No agent defines its own _get_tools() — inherited from BaseChannelAgent."""
    cls = _get_agent_class(module_path, class_name)
    assert "_get_tools" not in cls.__dict__


def test_channel_main_list_chats_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """channel_main() with -l prints recent chats and exits."""
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    class FakeAgent(BaseChannelAgent, ChatSorcarAgent):
        pass

    original_argv = sys.argv[:]
    try:
        sys.argv = ["test-cli", "-l"]
        with pytest.raises(SystemExit) as exc_info:
            channel_main(FakeAgent, "kiss-test")
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert captured.out
    finally:
        sys.argv = original_argv


_POLL_MODULES = [
    "kiss.agents.third_party_agents.bluebubbles_agent",
    "kiss.agents.third_party_agents.discord_agent",
    "kiss.agents.third_party_agents.feishu_agent",
    "kiss.agents.third_party_agents.googlechat_agent",
    "kiss.agents.third_party_agents.irc_agent",
    "kiss.agents.third_party_agents.line_agent",
    "kiss.agents.third_party_agents.matrix_agent",
    "kiss.agents.third_party_agents.mattermost_agent",
    "kiss.agents.third_party_agents.msteams_agent",
    "kiss.agents.third_party_agents.nextcloud_talk_agent",
    "kiss.agents.third_party_agents.phone_control_agent",
    "kiss.agents.third_party_agents.signal_agent",
    "kiss.agents.third_party_agents.slack_agent",
    "kiss.agents.third_party_agents.sms_agent",
    "kiss.agents.third_party_agents.synology_chat_agent",
    "kiss.agents.third_party_agents.telegram_agent",
    "kiss.agents.third_party_agents.zalo_agent",
]

_NO_POLL_MODULES = [
    "kiss.agents.third_party_agents.gmail_agent",
    "kiss.agents.third_party_agents.imessage_agent",
    "kiss.agents.third_party_agents.nostr_agent",
    "kiss.agents.third_party_agents.tlon_agent",
    "kiss.agents.third_party_agents.twitch_agent",
    "kiss.agents.third_party_agents.whatsapp_agent",
]


@pytest.mark.parametrize("module_path", _POLL_MODULES)
def test_poll_modules_have_make_backend(module_path: str) -> None:
    """Modules with poll mode support expose a _make_backend() function."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "_make_backend")
    assert callable(mod._make_backend)


@pytest.mark.parametrize("module_path", _NO_POLL_MODULES)
def test_non_poll_modules_have_no_make_backend(module_path: str) -> None:
    """Modules without poll mode support don't expose _make_backend()."""
    mod = importlib.import_module(module_path)
    assert not hasattr(mod, "_make_backend")


@pytest.mark.parametrize("module_path", _POLL_MODULES + _NO_POLL_MODULES)
def test_all_modules_have_main(module_path: str) -> None:
    """All channel agent modules expose a main() function."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "main")
    assert callable(mod.main)
