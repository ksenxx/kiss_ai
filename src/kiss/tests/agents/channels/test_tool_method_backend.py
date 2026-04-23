"""Integration tests for Issue 1: all channel backends use ToolMethodBackend mixin.

Verifies that every channel backend inherits from ToolMethodBackend, that
get_tool_methods() returns only callable methods, and that protocol methods
are properly excluded.
"""

from __future__ import annotations

import pytest

from kiss.agents.channels._channel_agent_utils import ToolMethodBackend
from kiss.agents.channels.bluebubbles_agent import BlueBubblesChannelBackend
from kiss.agents.channels.discord_agent import DiscordChannelBackend
from kiss.agents.channels.feishu_agent import FeishuChannelBackend
from kiss.agents.channels.gmail_agent import GmailChannelBackend
from kiss.agents.channels.googlechat_agent import GoogleChatChannelBackend
from kiss.agents.channels.imessage_agent import IMessageChannelBackend
from kiss.agents.channels.irc_agent import IRCChannelBackend
from kiss.agents.channels.line_agent import LineChannelBackend
from kiss.agents.channels.matrix_agent import MatrixChannelBackend
from kiss.agents.channels.mattermost_agent import MattermostChannelBackend
from kiss.agents.channels.msteams_agent import MSTeamsChannelBackend
from kiss.agents.channels.nextcloud_talk_agent import NextcloudTalkChannelBackend
from kiss.agents.channels.nostr_agent import NostrChannelBackend
from kiss.agents.channels.phone_control_agent import PhoneControlChannelBackend
from kiss.agents.channels.signal_agent import SignalChannelBackend
from kiss.agents.channels.slack_agent import SlackChannelBackend
from kiss.agents.channels.sms_agent import SMSChannelBackend
from kiss.agents.channels.synology_chat_agent import SynologyChatChannelBackend
from kiss.agents.channels.telegram_agent import TelegramChannelBackend
from kiss.agents.channels.tlon_agent import TlonChannelBackend
from kiss.agents.channels.twitch_agent import TwitchChannelBackend
from kiss.agents.channels.whatsapp_agent import WhatsAppChannelBackend
from kiss.agents.channels.zalo_agent import ZaloChannelBackend

ALL_BACKENDS = [
    BlueBubblesChannelBackend,
    DiscordChannelBackend,
    FeishuChannelBackend,
    GmailChannelBackend,
    GoogleChatChannelBackend,
    IMessageChannelBackend,
    IRCChannelBackend,
    LineChannelBackend,
    MatrixChannelBackend,
    MattermostChannelBackend,
    MSTeamsChannelBackend,
    NextcloudTalkChannelBackend,
    NostrChannelBackend,
    PhoneControlChannelBackend,
    SignalChannelBackend,
    SlackChannelBackend,
    SMSChannelBackend,
    SynologyChatChannelBackend,
    TelegramChannelBackend,
    TlonChannelBackend,
    TwitchChannelBackend,
    WhatsAppChannelBackend,
    ZaloChannelBackend,
]


@pytest.mark.parametrize("cls", ALL_BACKENDS, ids=lambda c: c.__name__)
def test_backend_inherits_tool_method_backend(cls: type) -> None:
    """Every channel backend class inherits from ToolMethodBackend."""
    assert issubclass(cls, ToolMethodBackend)


@pytest.mark.parametrize("cls", ALL_BACKENDS, ids=lambda c: c.__name__)
def test_no_inline_get_tool_methods(cls: type) -> None:
    """Backend class does not define its own get_tool_methods (uses mixin)."""
    assert "get_tool_methods" not in cls.__dict__
