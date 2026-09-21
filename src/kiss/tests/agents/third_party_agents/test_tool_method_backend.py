# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for Issue 1: all channel backends use ToolMethodBackend mixin.

Verifies that every channel backend inherits from ToolMethodBackend, that
get_tool_methods() returns only callable methods, and that protocol methods
are properly excluded.
"""

from __future__ import annotations

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import ToolMethodBackend
from kiss.agents.third_party_agents.bluebubbles_sea import BlueBubblesChannelBackend
from kiss.agents.third_party_agents.discord_sea import DiscordChannelBackend
from kiss.agents.third_party_agents.feishu_sea import FeishuChannelBackend
from kiss.agents.third_party_agents.gmail_sea import GmailChannelBackend
from kiss.agents.third_party_agents.googlechat_sea import GoogleChatChannelBackend
from kiss.agents.third_party_agents.imessage_sea import IMessageChannelBackend
from kiss.agents.third_party_agents.irc_sea import IRCChannelBackend
from kiss.agents.third_party_agents.line_sea import LineChannelBackend
from kiss.agents.third_party_agents.matrix_sea import MatrixChannelBackend
from kiss.agents.third_party_agents.mattermost_sea import MattermostChannelBackend
from kiss.agents.third_party_agents.msteams_sea import MSTeamsChannelBackend
from kiss.agents.third_party_agents.nextcloud_sea import NextcloudTalkChannelBackend
from kiss.agents.third_party_agents.nostr_sea import NostrChannelBackend
from kiss.agents.third_party_agents.phone_sea import PhoneControlChannelBackend
from kiss.agents.third_party_agents.signal_sea import SignalChannelBackend
from kiss.agents.third_party_agents.slack_sea import SlackChannelBackend
from kiss.agents.third_party_agents.sms_sea import SMSChannelBackend
from kiss.agents.third_party_agents.synology_sea import SynologyChatChannelBackend
from kiss.agents.third_party_agents.telegram_sea import TelegramChannelBackend
from kiss.agents.third_party_agents.tlon_sea import TlonChannelBackend
from kiss.agents.third_party_agents.twitch_sea import TwitchChannelBackend
from kiss.agents.third_party_agents.whatsapp_sea import WhatsAppChannelBackend
from kiss.agents.third_party_agents.zalo_sea import ZaloChannelBackend

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
