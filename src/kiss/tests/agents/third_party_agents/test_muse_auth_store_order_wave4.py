# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A vault store that fails leaves the connector's ``config.json`` untouched.

Mattermost, BlueBubbles, LINE, Zalo, Twitch and Nextcloud Talk probe a
candidate credential directly, then used to write its non-secret metadata
(server origin, webhook secret, OA id, client id) BEFORE ``store_credentials``.
When the vault then refused the store, the previous credential stayed enrolled
but ``config.json`` already described the candidate, so the next backend (each
cron tick builds one) paired the old credential with the new server, secret or
client id: a Mattermost or Nextcloud token bound to its old origin was sent to
the new one and refused, and LINE verified webhooks with the wrong secret.

Each test enrolls a working credential against the loopback emulator of
``test_muse_auth_messaging``, then rotates to new metadata with a credential the
service accepts and the vault refuses: one with a doubled space, which
``valid_credential_value`` rejects.  Nextcloud's ``Basic`` value is always well
formed, so its store fails because a directory occupies the vault entry's path
(the vault cannot write the new entry, as on a broken disk).
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.third_party_agents.bluebubbles.bluebubbles_sea import (
    BlueBubblesChannelBackend,
)
from kiss.agents.third_party_agents.bluebubbles.bluebubbles_sea import _config as bb_config
from kiss.agents.third_party_agents.bluebubbles.bluebubbles_sea import (
    _muse_authenticate as bb_authenticate,
)
from kiss.agents.third_party_agents.line.line_sea import LineAgent
from kiss.agents.third_party_agents.line.line_sea import _config as line_config
from kiss.agents.third_party_agents.mattermost.mattermost_sea import (
    MattermostAgent,
    MattermostChannelBackend,
)
from kiss.agents.third_party_agents.mattermost.mattermost_sea import _config as mm_config
from kiss.agents.third_party_agents.muse_auth._common import muse_auth_dir
from kiss.agents.third_party_agents.nextcloud.nextcloud_sea import NextcloudTalkAgent
from kiss.agents.third_party_agents.nextcloud.nextcloud_sea import _config as nc_config
from kiss.agents.third_party_agents.twitch.twitch_sea import TwitchAgent
from kiss.agents.third_party_agents.twitch.twitch_sea import _config as twitch_config
from kiss.agents.third_party_agents.zalo.zalo_sea import ZaloAgent
from kiss.agents.third_party_agents.zalo.zalo_sea import _config as zalo_config
from kiss.tests.agents.third_party_agents.muse_test_utils import auth_tools, loopback_policy
from kiss.tests.agents.third_party_agents.recording_http import serve_recording
from kiss.tests.agents.third_party_agents.test_muse_auth_messaging import (
    _REAL_BB_PASSWORD,
    _REAL_LINE_TOKEN,
    _REAL_MM_TOKEN,
    _REAL_NC_PASSWORD,
    _REAL_TWITCH_TOKEN,
    _REAL_ZALO_TOKEN,
    _MessagingApiHandler,
    _MessagingApiServer,
)

# The same policy as test_muse_auth_messaging: Twitch, Zalo and LINE reach
# the loopback emulator through an ``extra_hosts`` entry, the self-hosted
# services through the origin they were enrolled with.
MUSE_POLICY = loopback_policy("twitch", "zalo", "line")

_VAULT_REFUSED = "invalid credential token value"


@pytest.fixture()
def api_server() -> Iterator[_MessagingApiServer]:
    """The server the working credential is enrolled against."""
    yield from serve_recording(_MessagingApiHandler, _MessagingApiServer)


@pytest.fixture()
def other_server() -> Iterator[_MessagingApiServer]:
    """The server a failed rotation points at."""
    yield from serve_recording(_MessagingApiHandler, _MessagingApiServer)


def test_mattermost_failed_store_keeps_the_server(
    muse_env: Path, api_server: _MessagingApiServer, other_server: _MessagingApiServer
) -> None:
    """The enrolled token and the server it is bound to stay paired for the next backend."""
    tools = auth_tools(MattermostAgent())
    port = api_server.server_address[1]
    first = tools["authenticate_mattermost"]("127.0.0.1", _REAL_MM_TOKEN, port=port, scheme="http")
    assert json.loads(first)["ok"] is True
    enrolled = mm_config.path.read_text()
    result = json.loads(
        tools["authenticate_mattermost"](
            "127.0.0.1", "mm  spaced", port=other_server.server_address[1], scheme="http"
        )
    )
    assert result["ok"] is False
    assert _VAULT_REFUSED in result["error"]
    assert mm_config.path.read_text() == enrolled
    fresh = MattermostChannelBackend()
    assert fresh.connect() is True, fresh._connection_info
    assert api_server.header("Authorization") == f"Bearer {_REAL_MM_TOKEN}"


def test_bluebubbles_failed_store_keeps_the_server(
    muse_env: Path, api_server: _MessagingApiServer, other_server: _MessagingApiServer
) -> None:
    """The enrolled password keeps its ``server_url``; a fresh backend still reaches it."""
    backend = BlueBubblesChannelBackend()
    assert json.loads(bb_authenticate(backend, api_server.base_url, _REAL_BB_PASSWORD))["ok"]
    result = json.loads(bb_authenticate(backend, other_server.base_url, "bb  spaced"))
    assert result["ok"] is False
    assert _VAULT_REFUSED in result["error"]
    assert json.loads(bb_config.path.read_text()) == {"server_url": api_server.base_url}
    fresh = BlueBubblesChannelBackend()
    assert fresh._wire_muse() is True
    fresh.poll_messages("chat1", "")
    assert api_server.query().get("password") == [_REAL_BB_PASSWORD]


@pytest.mark.parametrize("new_secret", ["cs2", ""], ids=["replaced-secret", "dropped-secret"])
def test_line_failed_store_keeps_the_webhook_secret(
    muse_env: Path,
    api_server: _MessagingApiServer,
    monkeypatch: pytest.MonkeyPatch,
    new_secret: str,
) -> None:
    """The enrolled token keeps the channel secret its webhooks are signed with."""
    monkeypatch.setenv("LINE_API_BASE", api_server.base_url)
    tools = auth_tools(LineAgent())
    assert json.loads(tools["authenticate_line"](_REAL_LINE_TOKEN, "cs1"))["ok"] is True
    result = json.loads(tools["authenticate_line"]("line  spaced", new_secret))
    assert result["ok"] is False
    assert _VAULT_REFUSED in result["error"]
    assert json.loads(line_config.path.read_text()) == {"channel_secret": "cs1"}
    assert json.loads(auth_tools(LineAgent())["check_line_auth"]())["ok"] is True
    assert api_server.header("Authorization") == f"Bearer {_REAL_LINE_TOKEN}"


@pytest.mark.parametrize("new_oa_id", ["OA2", ""], ids=["replaced-oa", "dropped-oa"])
def test_zalo_failed_store_keeps_the_oa_id(
    muse_env: Path,
    api_server: _MessagingApiServer,
    monkeypatch: pytest.MonkeyPatch,
    new_oa_id: str,
) -> None:
    """The enrolled token keeps the Official Account id it belongs to."""
    monkeypatch.setenv("ZALO_API_BASE", api_server.base_url + "/v2.0/oa")
    tools = auth_tools(ZaloAgent())
    assert json.loads(tools["authenticate_zalo"](_REAL_ZALO_TOKEN, oa_id="OA1"))["ok"] is True
    result = json.loads(tools["authenticate_zalo"]("zalo  spaced", oa_id=new_oa_id))
    assert result["ok"] is False
    assert _VAULT_REFUSED in result["error"]
    assert json.loads(zalo_config.path.read_text()) == {"oa_id": "OA1"}
    assert json.loads(auth_tools(ZaloAgent())["check_zalo_auth"]())["ok"] is True
    assert api_server.header("access_token") == _REAL_ZALO_TOKEN


def test_twitch_failed_store_keeps_the_client_id(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """The enrolled token keeps the client id Helix checks it against."""
    agent = TwitchAgent()
    agent._backend._helix_base = api_server.base_url + "/helix"
    tools = auth_tools(agent)
    first = tools["authenticate_twitch"]("cid1", "cs-secret", _REAL_TWITCH_TOKEN, "kisscaster")
    assert json.loads(first)["ok"] is True
    result = json.loads(tools["authenticate_twitch"]("cid2", "", "twitch  spaced", "otherchan"))
    assert result["ok"] is False
    assert _VAULT_REFUSED in result["error"]
    assert json.loads(twitch_config.path.read_text()) == {
        "client_id": "cid1",
        "channel_name": "kisscaster",
    }
    fresh = TwitchAgent()
    fresh._backend._helix_base = api_server.base_url + "/helix"
    assert json.loads(auth_tools(fresh)["check_twitch_auth"]())["ok"] is True
    assert api_server.header("Client-ID") == "cid1"
    assert api_server.header("Authorization") == f"Bearer {_REAL_TWITCH_TOKEN}"


def test_nextcloud_failed_store_keeps_the_server(
    muse_env: Path, api_server: _MessagingApiServer, other_server: _MessagingApiServer
) -> None:
    """The enrolled login keeps its server URL and user name for the next backend."""
    tools = auth_tools(NextcloudTalkAgent())
    first = tools["authenticate_nextcloud"](api_server.base_url, "bot", _REAL_NC_PASSWORD)
    assert json.loads(first)["ok"] is True
    enrolled = nc_config.path.read_text()
    entry = muse_auth_dir() / "vault" / "nextcloud.json"
    credential = entry.read_bytes()
    entry.unlink()
    entry.mkdir()
    try:
        result = json.loads(
            tools["authenticate_nextcloud"](other_server.base_url, "bot2", "nc-other-secret")
        )
    finally:
        entry.rmdir()
        entry.write_bytes(credential)
        entry.chmod(0o600)
    assert result["ok"] is False
    assert "IsADirectoryError" in result["error"]
    assert nc_config.path.read_text() == enrolled
    assert json.loads(auth_tools(NextcloudTalkAgent())["check_nextcloud_auth"]())["ok"] is True
    basic = base64.b64encode(f"bot:{_REAL_NC_PASSWORD}".encode()).decode()
    assert api_server.header("Authorization") == f"Basic {basic}"
