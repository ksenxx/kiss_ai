# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The channel-listing tools say "channels", not the package name.

A mechanical rename of the ``channels`` package to ``third_party_agents``
also renamed the LLM-facing tools that list a service's channels
(``list_channels`` became ``list_third_party_agents`` in Discord, Teams,
Mattermost, Slack and Tlon; Twitch's ``search_channels`` became
``search_third_party_agents``) and the result key they return.  The
tests read the tool names the agents expose and call the tools against
one local HTTP server that answers like Graph, Helix, Discord and
Mattermost (Mattermost through the Muse-auth driver and a live daemon,
the one driver that does not need ``mattermostdriver``).  No mocks.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import ToolMethodBackend
from kiss.agents.third_party_agents.discord.discord_sea import DiscordChannelBackend
from kiss.agents.third_party_agents.mattermost.mattermost_sea import (
    MattermostChannelBackend,
)
from kiss.agents.third_party_agents.mattermost.mattermost_sea import _config as mm_config
from kiss.agents.third_party_agents.msteams.msteams_sea import MSTeamsChannelBackend
from kiss.agents.third_party_agents.slack.slack_sea import SlackChannelBackend
from kiss.agents.third_party_agents.tlon.tlon_sea import TlonChannelBackend
from kiss.agents.third_party_agents.twitch.twitch_sea import TwitchChannelBackend
from kiss.tests.agents.third_party_agents.muse_test_utils import loopback_policy
from kiss.tests.agents.third_party_agents.recording_http import (
    RecordingHandler,
    RecordingServer,
    recording_server,
)

# Mattermost binds its credential to the enrolled origin: no extra hosts.
MUSE_POLICY = loopback_policy()

_CHANNEL_LISTS = {
    "/teams/T1/channels": {
        "value": [{"id": "c1", "displayName": "General", "membershipType": "standard"}],
    },
    "/search/channels": {"data": [{"broadcaster_login": "streamer"}]},
    "/guilds/G1/channels": [{"id": "c1", "name": "general", "type": 0}],
    "/api/v4/users/me": {"id": "u1", "username": "kissbot"},
    "/api/v4/users/me/teams/T1/channels": [
        {"id": "c1", "name": "town-square", "display_name": "Town Square", "type": "O"},
    ],
}


class _ServicesHandler(RecordingHandler):
    """Answers each listing endpoint of the four services with one channel."""

    def do_GET(self) -> None:  # noqa: N802
        entry = self.record_request(None)
        body = _CHANNEL_LISTS.get(entry["path"])
        self._json(200 if body is not None else 404, body or {"error": "not found"})


@pytest.fixture()
def server() -> Iterator[RecordingServer]:
    """A recording server answering like Graph, Helix, Discord and Mattermost."""
    with recording_server(_ServicesHandler) as srv:
        yield srv


def _tool_names(backend: ToolMethodBackend) -> list[str]:
    return [tool.__name__ for tool in backend.get_tool_methods()]


def test_listing_tools_are_named_for_channels() -> None:
    backends: dict[str, list[ToolMethodBackend]] = {
        "list_channels": [
            DiscordChannelBackend(), MSTeamsChannelBackend(), MattermostChannelBackend(),
            SlackChannelBackend(), TlonChannelBackend(),
        ],
        "search_channels": [TwitchChannelBackend()],
    }
    for tool, owners in backends.items():
        for backend in owners:
            names = _tool_names(backend)
            assert tool in names, (type(backend).__name__, names)
            assert not [n for n in names if "third_party_agents" in n], names


def test_listing_results_carry_a_channels_key(server: RecordingServer) -> None:
    teams = MSTeamsChannelBackend(graph_base=server.base_url)
    teams._access_token = "graph-token"
    assert json.loads(teams.list_channels("T1"))["channels"] == [
        {"id": "c1", "name": "General", "type": "standard", "description": ""},
    ]

    twitch = TwitchChannelBackend(helix_base=server.base_url)
    twitch._client_id, twitch._access_token = "cid", "helix-token"
    assert json.loads(twitch.search_channels("streamer"))["channels"] == [
        {"broadcaster_login": "streamer"},
    ]

    discord = DiscordChannelBackend(api_base=server.base_url)
    discord._token = "bot-token"
    listed = json.loads(discord.list_channels("G1"))
    assert [c["name"] for c in listed["channels"]] == ["general"]


def test_mattermost_listing_result_carries_a_channels_key(
    muse_env: Path, server: RecordingServer,
) -> None:
    mm_config.save({
        "url": "127.0.0.1", "token": "mm-token",
        "port": str(server.server_address[1]), "scheme": "http",
    })
    mattermost = MattermostChannelBackend()
    assert mattermost.connect() is True, mattermost._connection_info
    listed = json.loads(mattermost.list_channels("T1"))
    assert [c["name"] for c in listed["channels"]] == ["town-square"]
