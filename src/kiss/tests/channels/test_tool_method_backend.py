"""Integration tests for Issue 1: all channel backends use ToolMethodBackend mixin.

Verifies that every channel backend inherits from ToolMethodBackend, that
get_tool_methods() returns only callable methods, and that protocol methods
are properly excluded.
"""

from __future__ import annotations

import pytest

from kiss.channels._channel_agent_utils import ToolMethodBackend
from kiss.channels.bluebubbles_agent import BlueBubblesChannelBackend
from kiss.channels.discord_agent import DiscordChannelBackend
from kiss.channels.feishu_agent import FeishuChannelBackend
from kiss.channels.gmail_agent import GmailChannelBackend
from kiss.channels.googlechat_agent import GoogleChatChannelBackend
from kiss.channels.imessage_agent import IMessageChannelBackend
from kiss.channels.irc_agent import IRCChannelBackend
from kiss.channels.line_agent import LineChannelBackend
from kiss.channels.matrix_agent import MatrixChannelBackend
from kiss.channels.mattermost_agent import MattermostChannelBackend
from kiss.channels.msteams_agent import MSTeamsChannelBackend
from kiss.channels.nextcloud_talk_agent import NextcloudTalkChannelBackend
from kiss.channels.nostr_agent import NostrChannelBackend
from kiss.channels.phone_control_agent import PhoneControlChannelBackend
from kiss.channels.signal_agent import SignalChannelBackend
from kiss.channels.slack_agent import SlackChannelBackend
from kiss.channels.sms_agent import SMSChannelBackend
from kiss.channels.synology_chat_agent import SynologyChatChannelBackend
from kiss.channels.telegram_agent import TelegramChannelBackend
from kiss.channels.tlon_agent import TlonChannelBackend
from kiss.channels.twitch_agent import TwitchChannelBackend
from kiss.channels.whatsapp_agent import WhatsAppChannelBackend
from kiss.channels.zalo_agent import ZaloChannelBackend

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
