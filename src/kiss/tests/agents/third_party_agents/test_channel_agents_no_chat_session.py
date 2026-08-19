# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests locking the channel-agent / chat-session contract.

Verifies the contract enforced by the
"third-party agents do not inherit SorcarAgent" cleanup:

1. No channel agent class is a subclass of :class:`SorcarAgent` (nor of
   :class:`ChatSorcarAgent`) — third-party agents are plain
   :class:`BaseChannelAgent` carriers whose tasks always execute on the
   kiss-web daemon through :func:`kiss.server.sorcar.run`.
2. ``channel_main()``'s parser rejects every chat-session CLI flag
   (``-n/--new``, ``-c/--chat-id``, ``-l/--list-chat-id``) — they no
   longer exist anywhere in the project's CLI surface.
3. The ``_apply_chat_args`` helper has been removed from the
   channel-agent CLI helpers
   (``kiss.agents.third_party_agents._channel_cli``).
"""

from __future__ import annotations

import importlib
import sys

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    channel_main,
)

ALL_CHANNEL_AGENTS = [
    ("A2AAgent", "kiss.agents.third_party_agents.a2a_agent"),
    ("BlueBubblesAgent", "kiss.agents.third_party_agents.bluebubbles_agent"),
    ("DingTalkAgent", "kiss.agents.third_party_agents.dingtalk_agent"),
    ("DiscordAgent", "kiss.agents.third_party_agents.discord_agent"),
    ("EmailAgent", "kiss.agents.third_party_agents.email_agent"),
    ("FeishuAgent", "kiss.agents.third_party_agents.feishu_agent"),
    ("GmailAgent", "kiss.agents.third_party_agents.gmail_agent"),
    ("GoogleChatAgent", "kiss.agents.third_party_agents.googlechat_agent"),
    ("HomeAssistantAgent", "kiss.agents.third_party_agents.homeassistant_agent"),
    ("IMessageAgent", "kiss.agents.third_party_agents.imessage_agent"),
    ("IRCAgent", "kiss.agents.third_party_agents.irc_agent"),
    ("LineAgent", "kiss.agents.third_party_agents.line_agent"),
    ("MatrixAgent", "kiss.agents.third_party_agents.matrix_agent"),
    ("MattermostAgent", "kiss.agents.third_party_agents.mattermost_agent"),
    ("MSTeamsAgent", "kiss.agents.third_party_agents.msteams_agent"),
    ("NextcloudTalkAgent", "kiss.agents.third_party_agents.nextcloud_talk_agent"),
    ("NostrAgent", "kiss.agents.third_party_agents.nostr_agent"),
    ("NtfyAgent", "kiss.agents.third_party_agents.ntfy_agent"),
    ("OpenAICompatAgent", "kiss.agents.third_party_agents.openai_compat_agent"),
    ("PhoneControlAgent", "kiss.agents.third_party_agents.phone_control_agent"),
    ("QQAgent", "kiss.agents.third_party_agents.qq_agent"),
    ("SignalAgent", "kiss.agents.third_party_agents.signal_agent"),
    ("SimpleXAgent", "kiss.agents.third_party_agents.simplex_agent"),
    ("SlackAgent", "kiss.agents.third_party_agents.slack_agent"),
    ("SMSAgent", "kiss.agents.third_party_agents.sms_agent"),
    ("SynologyChatAgent", "kiss.agents.third_party_agents.synology_chat_agent"),
    ("TelegramAgent", "kiss.agents.third_party_agents.telegram_agent"),
    ("TlonAgent", "kiss.agents.third_party_agents.tlon_agent"),
    ("TwitchAgent", "kiss.agents.third_party_agents.twitch_agent"),
    ("WebhookAgent", "kiss.agents.third_party_agents.webhook_agent"),
    ("WeComAgent", "kiss.agents.third_party_agents.wecom_agent"),
    ("WeixinAgent", "kiss.agents.third_party_agents.weixin_agent"),
    ("WhatsAppAgent", "kiss.agents.third_party_agents.whatsapp_agent"),
    ("ZaloAgent", "kiss.agents.third_party_agents.zalo_agent"),
]


def _get_agent_class(class_name: str, module_path: str) -> type:
    """Import *module_path* and return the agent class named *class_name*."""
    mod = importlib.import_module(module_path)
    cls: type = getattr(mod, class_name)
    return cls


@pytest.mark.parametrize(
    "class_name,module_path",
    ALL_CHANNEL_AGENTS,
    ids=[a[0] for a in ALL_CHANNEL_AGENTS],
)
def test_channel_agent_does_not_subclass_sorcar_agent(
    class_name: str, module_path: str,
) -> None:
    """No channel agent class inherits from ``SorcarAgent``.

    Channel agents are plain :class:`BaseChannelAgent` carriers; every
    task they run is submitted to the kiss-web daemon through
    :func:`kiss.server.sorcar.run`, so the local instance needs none of
    the ``SorcarAgent`` execution machinery.
    """
    cls = _get_agent_class(class_name, module_path)
    assert not issubclass(cls, SorcarAgent), (
        f"{class_name} still inherits SorcarAgent"
    )
    assert issubclass(cls, BaseChannelAgent)


@pytest.mark.parametrize(
    "class_name,module_path",
    ALL_CHANNEL_AGENTS,
    ids=[a[0] for a in ALL_CHANNEL_AGENTS],
)
def test_channel_agent_does_not_subclass_chat_sorcar_agent(
    class_name: str, module_path: str,
) -> None:
    """No channel agent inherits chat-session persistence.

    The third-party agents previously inherited from
    :class:`ChatSorcarAgent`, which silently added a ``chat_id``,
    ``new_chat()``, ``resume_chat()`` etc. surface to every channel
    agent (and to the on-disk ``sorcar.db`` ``task_history`` rows
    they produced).  After the cleanup they are plain
    :class:`BaseChannelAgent` carriers and have none of that surface.
    """
    cls = _get_agent_class(class_name, module_path)
    assert not issubclass(cls, ChatSorcarAgent), (
        f"{class_name} still inherits ChatSorcarAgent"
    )


@pytest.mark.parametrize(
    "flag",
    [
        "-n",
        "--new",
        "-c",
        "--chat-id",
        "-l",
        "--list-chat-id",
    ],
)
def test_channel_main_rejects_chat_session_flag(
    flag: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """``channel_main`` no longer exposes any chat-session CLI flag.

    Argparse exits with status 2 on an unrecognized flag (after
    printing a "unrecognized arguments" diagnostic to stderr).  Both
    are asserted so the contract is locked end-to-end.
    """

    class _FakeAgent(BaseChannelAgent):
        def _is_authenticated(self) -> bool:
            return False

        def _get_auth_tools(self) -> list:
            return []

    original_argv = sys.argv[:]
    try:
        sys.argv = ["test-cli", flag, "-t", "noop"]
        with pytest.raises(SystemExit) as exc_info:
            channel_main(_FakeAgent, "kiss-test")
        assert exc_info.value.code == 2
    finally:
        sys.argv = original_argv
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err or "unrecognized" in err


def test_apply_chat_args_helper_removed() -> None:
    """``_apply_chat_args`` no longer exists in the CLI helpers.

    Its only consumer was the channel-agent CLI, which is now a
    plain :class:`SorcarAgent` and has no use for chat-session
    routing.  The helper itself is removed so the chat-session
    surface really is gone from the project.
    """
    from kiss.agents.third_party_agents import _channel_cli

    assert not hasattr(_channel_cli, "_apply_chat_args")
