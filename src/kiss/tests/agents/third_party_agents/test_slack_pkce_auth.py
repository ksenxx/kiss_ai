# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the Slack click-Allow sign-in (legacy storage mode).

``authenticate_slack()`` starts a PKCE authorization-code session on the
loopback redirect port; the tests play the user's browser by requesting
``http://127.0.0.1:53682/callback`` with the code, and a real local HTTP
server (``KISS_SLACK_BASE_URL``) plays Slack's token endpoint and Web
API.  Muse-auth mode is covered in ``test_muse_auth_channels.py``.
"""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from kiss.agents.third_party_agents import _oauth_apps
from kiss.agents.third_party_agents._device_auth import ConsentSession
from kiss.agents.third_party_agents._oauth_apps import LOOPBACK_PORT, LOOPBACK_REDIRECT_URI
from kiss.agents.third_party_agents.slack_sea import (
    _USER_SCOPES,
    SlackAgent,
    _legacy_client,
    _list_workspaces,
    _load_config,
    _make_backend,
)
from kiss.tests.agents.third_party_agents.slack_oauth_test_utils import (
    CLIENT_ID,
    SlackApiServer,
    SlackOAuthState,
    sign_in,
    start_sign_in,
)
from kiss.tests.conftest import hold_loopback_port


@pytest.fixture()
def slack(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[SlackOAuthState, SlackApiServer]]:
    """Run the emulated Slack in legacy (plaintext token file) mode.

    Reserves the fixed redirect port across pytest processes for the
    test's duration.
    """
    oauth = SlackOAuthState()
    server = SlackApiServer(oauth)
    monkeypatch.setenv("KISS_MUSE_AUTH", "0")
    monkeypatch.setenv("KISS_SLACK_BASE_URL", server.base_url + "/")
    monkeypatch.setenv("KISS_SLACK_CLIENT_ID", CLIENT_ID)
    with hold_loopback_port(LOOPBACK_PORT):
        yield oauth, server
        ConsentSession.cancel_active("slack")
    server.stop()


def _tools(agent: SlackAgent) -> dict[str, Any]:
    return {fn.__name__: fn for fn in agent._get_auth_tools()}


def test_missing_client_id_is_reported(
    slack: tuple[SlackOAuthState, SlackApiServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an embedded or overridden client ID nothing starts."""
    monkeypatch.delenv("KISS_SLACK_CLIENT_ID")
    monkeypatch.setitem(_oauth_apps.KISS_OAUTH_CLIENT_IDS, "slack", "")
    result = json.loads(_tools(SlackAgent())["authenticate_slack"]())
    assert result["ok"] is False
    assert "KISS_SLACK_CLIENT_ID" in result["error"]


def test_authorize_url_requests_user_scopes_with_pkce(
    slack: tuple[SlackOAuthState, SlackApiServer],
) -> None:
    """The sign-in URL targets Slack's authorize page with PKCE and user scopes."""
    oauth, server = slack
    tools = _tools(SlackAgent())
    started = start_sign_in(tools, oauth)
    assert started["status"] == "consent_required"
    assert "finish_slack_auth()" in started["instructions"]
    parts = urlsplit(started["verification_uri"])
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == f"{server.base_url}/oauth/v2/authorize"
    query = {k: v[0] for k, v in parse_qs(parts.query).items()}
    assert query["client_id"] == CLIENT_ID
    assert query["user_scope"] == _USER_SCOPES
    assert "scope" not in query  # bot scopes need an https redirect
    assert query["code_challenge_method"] == "S256"
    assert query["redirect_uri"] == LOOPBACK_REDIRECT_URI

    # Before the user approves, finishing reports pending; clearing
    # cancels the session and frees the redirect port.
    assert json.loads(tools["finish_slack_auth"]())["status"] == "pending"
    assert "cleared" in tools["clear_slack_auth"]()
    assert "authenticate_slack()" in json.loads(tools["finish_slack_auth"]())["error"]


def test_sign_in_stores_rotating_user_token_and_refreshes(
    slack: tuple[SlackOAuthState, SlackApiServer], capsys: pytest.CaptureFixture
) -> None:
    """The user token is validated, saved with its refresh token, and rotated."""
    oauth, server = slack
    agent = SlackAgent()
    tools = _tools(agent)
    assert "Not authenticated" in tools["check_slack_auth"]()

    done = sign_in(tools, oauth)
    assert done["ok"] is True
    assert (done["team"], done["user"]) == ("KISS", "alice")
    exchange = oauth.forms[-1]
    assert exchange["grant_type"] == "authorization_code"
    assert exchange["redirect_uri"] == LOOPBACK_REDIRECT_URI
    assert "client_secret" not in exchange
    assert server.requests[-1] == ("/api/auth.test", "Bearer xoxp-access-1")
    cfg = _load_config("default")
    assert cfg is not None
    assert cfg["access_token"] == "xoxp-access-1"
    assert cfg["refresh_token"] == "xoxe-1-refresh-1"
    assert cfg["client_id"] == CLIENT_ID
    assert abs(float(cfg["expires_at"]) - (time.time() + 43200)) < 60

    checked = json.loads(tools["check_slack_auth"]())
    assert checked["ok"] is True and checked["user_id"] == "U1"
    _list_workspaces()
    assert "✓ valid" in capsys.readouterr().out

    # A token about to expire is refreshed in-process before the call.
    oauth.expires_in = 30
    assert sign_in(tools, oauth)["ok"] is True  # issues xoxp-access-2
    assert json.loads(tools["check_slack_auth"]())["ok"] is True
    refresh = oauth.forms[-1]
    assert refresh == {
        "grant_type": "refresh_token",
        "refresh_token": "xoxe-1-refresh-2",
        "client_id": CLIENT_ID,
    }
    assert server.requests[-1] == ("/api/auth.test", "Bearer xoxp-access-3")
    cfg = _load_config("default")
    assert cfg is not None
    assert (cfg["access_token"], cfg["refresh_token"]) == ("xoxp-access-3", "xoxe-1-refresh-3")

    # Poll mode builds the same refreshing client from the file.
    backend = _make_backend("default")
    assert backend.connect()
    assert server.requests[-1] == ("/api/auth.test", "Bearer xoxp-access-4")

    # A revoked refresh token, then an unreachable Slack, surface as errors.
    oauth.refresh_token = "revoked"
    failed = json.loads(tools["check_slack_auth"]())
    assert failed["ok"] is False and "invalid_refresh_token" in failed["error"]
    server.stop()
    failed = json.loads(tools["check_slack_auth"]())
    assert failed["ok"] is False and "ConnectionError" in failed["error"]


def test_rejected_or_failed_sign_in_keeps_the_stored_token(
    slack: tuple[SlackOAuthState, SlackApiServer],
) -> None:
    """Failed sign-ins never replace the working credential."""
    oauth, server = slack
    agent = SlackAgent()
    tools = _tools(agent)
    assert "Slack sign-in failed" in json.loads(tools["finish_slack_auth"]())["error"]
    assert sign_in(tools, oauth)["ok"] is True

    # auth.test rejects the new token.
    oauth.invalid_next = True
    failed = sign_in(tools, oauth)
    assert failed["ok"] is False and "invalid_auth" in failed["error"]

    # The code exchange fails (PKCE verifier mismatch).
    started = start_sign_in(tools, oauth)
    oauth.challenge = "not-the-challenge"
    state = parse_qs(urlsplit(started["verification_uri"]).query)["state"][0]
    requests.get(
        f"http://127.0.0.1:{LOOPBACK_PORT}/callback",
        params={"code": "test-code", "state": state},
        timeout=10,
    )
    failed = json.loads(tools["finish_slack_auth"]())
    assert failed["ok"] is False and "invalid_code_verifier" in failed["error"]

    cfg = _load_config("default")
    assert cfg is not None and cfg["access_token"] == "xoxp-access-1"
    assert json.loads(tools["check_slack_auth"]())["ok"] is True
    assert server.requests[-1] == ("/api/auth.test", "Bearer xoxp-access-1")


def test_answer_without_user_token_is_refused(
    slack: tuple[SlackOAuthState, SlackApiServer],
) -> None:
    """An exchange answer lacking ``authed_user.access_token`` stores nothing."""
    oauth, _server = slack
    oauth.omit_user_token = True
    tools = _tools(SlackAgent())
    failed = sign_in(tools, oauth)
    assert failed["ok"] is False and "no user token" in failed["error"]
    assert _load_config("default") is None


def test_second_client_reuses_token_rotated_by_the_first(
    slack: tuple[SlackOAuthState, SlackApiServer],
) -> None:
    """Two clients built from one expiring token file refresh it only once.

    Slack refuses a consumed refresh token, so the second client must
    reload the file the first client rotated instead of replaying the
    refresh token it was constructed with.
    """
    oauth, server = slack
    tools = _tools(SlackAgent())
    oauth.expires_in = 30  # the stored token is inside the refresh margin
    assert sign_in(tools, oauth)["ok"] is True  # xoxp-access-1 / refresh-1
    oauth.expires_in = 43200
    api_base = server.base_url + "/api/"
    first = _legacy_client("default", api_base)
    second = _legacy_client("default", api_base)
    assert first is not None and second is not None
    assert first.token == second.token == "xoxp-access-1"

    assert first.auth_test()["ok"] is True
    assert server.requests[-1] == ("/api/auth.test", "Bearer xoxp-access-2")
    assert second.auth_test()["ok"] is True
    assert server.requests[-1] == ("/api/auth.test", "Bearer xoxp-access-2")

    refreshes = [f for f in oauth.forms if f.get("grant_type") == "refresh_token"]
    assert refreshes == [
        {"grant_type": "refresh_token", "refresh_token": "xoxe-1-refresh-1", "client_id": CLIENT_ID}
    ]
    cfg = _load_config("default")
    assert cfg is not None
    assert (cfg["access_token"], cfg["refresh_token"]) == ("xoxp-access-2", "xoxe-1-refresh-2")
    assert second.token == "xoxp-access-2"


def test_busy_redirect_port_is_reported(
    slack: tuple[SlackOAuthState, SlackApiServer],
) -> None:
    """Another program holding the redirect port makes the start fail cleanly."""
    blocker = socket.socket()
    # Earlier callbacks leave TIME_WAIT entries on the port.
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        blocker.bind(("127.0.0.1", LOOPBACK_PORT))
        blocker.listen(1)
        result = json.loads(_tools(SlackAgent())["authenticate_slack"]())
    finally:
        blocker.close()
    assert result["ok"] is False
    assert "Could not start the Slack sign-in" in result["error"]
