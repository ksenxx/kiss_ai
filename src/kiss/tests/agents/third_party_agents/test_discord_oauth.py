# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Discord's click-Allow OAuth sign-in.

A local ``http.server`` plays discord.com: the OAuth token endpoint
(``/api/oauth2/token``), the REST API (``/api/v10``) and the incoming
webhook the token answer hands out.  ``DISCORD_OAUTH_BASE`` and
``DISCORD_API_BASE`` point the agent at it.  The tests play the
browser by issuing a real GET to the loopback callback on the fixed
redirect port 53682 (Discord matches redirect URIs exactly, so the
port cannot vary; tests in one process never overlap because a new
session cancels the previous owner of the port, and ``discord_server``
reserves the port across pytest processes).

Unreachable without test doubles: the ``OSError`` branch when port
53682 is held by another process, and transport failures of the
``/users/@me`` probe (``requests.RequestException``) other than a
refused connection, which ``test_finish_reports_unreachable_api``
covers.
"""

from __future__ import annotations

import json
import stat
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlsplit

import pytest
import requests

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server
from kiss.agents.third_party_agents._device_auth import ConsentSession
from kiss.agents.third_party_agents._oauth_apps import LOOPBACK_PORT, LOOPBACK_REDIRECT_URI
from kiss.agents.third_party_agents.discord.discord_sea import (
    DiscordAgent,
    _config,
    _make_backend,
    _scrub_config_token,
    _webhook_config,
)
from kiss.tests.agents.third_party_agents.muse_test_utils import (
    auth_tools,
    setup_muse_env,
    teardown_muse_env,
)
from kiss.tests.conftest import hold_loopback_port

_USER_TOKEN = "user-access-token"
_BOT_TOKEN = "bot-secret-token"
_HOOK_PATH = "/api/webhooks/901/hook-secret-token"


class _DiscordHandler(BaseHTTPRequestHandler):
    """Emulates the Discord endpoints the agent uses."""

    server: _DiscordServer  # type: ignore[assignment]

    def _reply(self, status: int, body: Any) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        auth = self.headers.get("Authorization", "")
        self.server.auth_headers.append(auth)
        path = urlsplit(self.path).path
        if auth == "Bearer html-token":
            payload = b"<html>Bad Gateway</html>"
            self.send_response(502)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif auth not in (f"Bearer {_USER_TOKEN}", f"Bot {_BOT_TOKEN}"):
            self._reply(401, {"message": "401: Unauthorized", "code": 0})
        elif path == "/api/v10/users/@me":
            self._reply(200, {"id": "42", "username": "alice", "discriminator": "0"})
        elif path == "/api/v10/users/@me/guilds":
            self._reply(200, [{"id": "777", "name": "Guild"}])
        else:
            self._reply(404, {"message": "Unknown"})

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        url = urlsplit(self.path)
        if url.path == "/api/oauth2/token":
            self.server.token_forms.append(dict(parse_qsl(self._read().decode())))
            self._reply(200, self.server.token_answer)
        elif url.path == _HOOK_PATH:
            self.server.webhook_posts.append((url.query, json.loads(self._read())))
            self._reply(200, {"id": "m-1", "channel_id": "555"})
        else:
            self._reply(404, {"message": "Unknown"})

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence per-request logging."""


class _DiscordServer(ThreadedHTTPServer):
    """Local Discord emulator with recorded requests."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _DiscordHandler)
        self.base = f"http://127.0.0.1:{self.server_address[1]}"
        self.token_forms: list[dict[str, str]] = []
        self.webhook_posts: list[tuple[str, dict[str, Any]]] = []
        self.auth_headers: list[str] = []
        self.token_answer: dict[str, Any] = {
            "access_token": _USER_TOKEN,
            "refresh_token": "user-refresh-token",
            "expires_in": 604800,
            "token_type": "Bearer",
            "scope": "identify guilds webhook.incoming",
            "webhook": {
                "id": "901",
                "token": "hook-secret-token",
                "channel_id": "555",
                "guild_id": "777",
                "url": f"{self.base}{_HOOK_PATH}",
            },
        }


@pytest.fixture()
def discord_server(
    isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_DiscordServer]:
    """Run the emulator and point the agent at it (legacy, non-Muse mode).

    Reserves the fixed redirect port across pytest processes for the
    test's duration.
    """
    server = _DiscordServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("DISCORD_OAUTH_BASE", server.base)
    monkeypatch.setenv("DISCORD_API_BASE", f"{server.base}/api/v10")
    monkeypatch.setenv("KISS_DISCORD_CLIENT_ID", "kiss-discord-app")
    monkeypatch.setenv("KISS_HEADLESS", "1")
    monkeypatch.setenv("KISS_MUSE_AUTH", "0")
    with hold_loopback_port(LOOPBACK_PORT):
        yield server
        ConsentSession.cancel_active("discord")
    stop_http_server(server, thread)


def _start(tools: dict[str, Any]) -> str:
    """Call authenticate_discord(), waiting while a foreign process holds port 53682.

    ``discord_server`` already serialises the port across pytest
    processes; the retry covers a non-test owner (a real Discord sign-in
    on this machine), which the file lock cannot see.

    Args:
        tools: The agent's auth tools.

    Returns:
        The authenticate_discord() answer.
    """
    deadline = time.monotonic() + 120
    answer = str(tools["authenticate_discord"]())
    while "Address already in use" in answer and time.monotonic() < deadline:
        time.sleep(0.5)
        answer = str(tools["authenticate_discord"]())
    return answer


def _approve(answer: str, code: str = "the-code") -> dict[str, Any]:
    """Play the browser: approve the consent URL in *answer*.

    Args:
        answer: The JSON returned by authenticate_discord().
        code: Authorization code the "provider" hands back.

    Returns:
        The query parameters of the authorization URL.
    """
    started = json.loads(answer)
    assert started.get("status") == "consent_required", started
    query = {k: v[0] for k, v in parse_qs(urlsplit(started["verification_uri"]).query).items()}
    resp = requests.get(
        LOOPBACK_REDIRECT_URI, params={"code": code, "state": query["state"]}, timeout=10
    )
    assert resp.status_code == 200
    return query


def test_sign_in_stores_token_and_webhook(discord_server: _DiscordServer) -> None:
    """Consent URL, PKCE exchange, storage, and webhook posting work end to end."""
    agent = DiscordAgent()
    tools = auth_tools(agent)
    assert "start_discord_browser_auth" not in tools
    assert not agent._is_authenticated()
    assert "authenticate_discord()" in tools["check_discord_auth"]()

    query = _approve(_start(tools))
    assert urlsplit(query["redirect_uri"]).port == 53682
    assert query["client_id"] == "kiss-discord-app"
    assert query["scope"] == "identify guilds webhook.incoming"
    assert query["code_challenge_method"] == "S256"

    done = json.loads(tools["finish_discord_auth"]())
    assert done["ok"] is True and done["username"] == "alice"
    assert done["webhook_channel_id"] == "555" and done["webhook_guild_id"] == "777"
    assert "hook-secret-token" not in json.dumps(done)
    form = discord_server.token_forms[0]
    assert form["grant_type"] == "authorization_code" and form["code"] == "the-code"
    assert "code_verifier" in form and "client_secret" not in form

    # Legacy storage: the refreshable user token in config, webhook in
    # its own 0600 file.
    stored = json.loads(_config.path.read_text())
    assert stored["auth_mode"] == "user" and stored["access_token"] == _USER_TOKEN
    assert stored["refresh_token"] == "user-refresh-token"
    assert stored["client_id"] == "kiss-discord-app" and float(stored["expires_at"]) > time.time()
    hook = json.loads(_webhook_config.path.read_text())
    assert hook["url"].endswith(_HOOK_PATH) and hook["channel_id"] == "555"
    assert stat.S_IMODE(_webhook_config.path.stat().st_mode) == 0o600

    check = json.loads(tools["check_discord_auth"]())
    assert check["auth"] == "user" and check["webhook_channel_id"] == "555"
    assert "bot token" in check["note"]

    backend = agent._backend
    assert json.loads(backend.list_guilds()) == {
        "ok": True,
        "guilds": [{"id": "777", "name": "Guild"}],
    }
    assert discord_server.auth_headers[-1] == f"Bearer {_USER_TOKEN}"
    posted = json.loads(backend.post_message("555", "hello"))
    assert posted == {"ok": True, "id": "m-1"}
    assert discord_server.webhook_posts == [("wait=true", {"content": "hello", "tts": False})]

    # A fresh agent reloads the user sign-in from config.
    reloaded = DiscordAgent()
    assert reloaded._backend._user_auth and reloaded._backend.connect()
    assert reloaded._backend.connection_info.endswith("(user sign-in)")


def test_user_sign_in_refuses_bot_only_features(discord_server: _DiscordServer) -> None:
    """Bot-only tools, other channels, replies and poll mode say a bot token is needed."""
    agent = DiscordAgent()
    tools = auth_tools(agent)
    _approve(_start(tools))
    assert json.loads(tools["finish_discord_auth"]())["ok"] is True
    backend = agent._backend
    for result in (
        backend.list_third_party_agents("777"),
        backend.get_channel("555"),
        backend.get_channel_messages("555"),
        backend.edit_message("555", "1", "x"),
        backend.delete_message("555", "1"),
        backend.add_reaction("555", "1", "x"),
        backend.create_thread("555", "1", "t"),
        backend.list_guild_members("777"),
        backend.create_invite("555"),
    ):
        error = json.loads(result)
        assert error["ok"] is False and "authenticate_discord(bot_token=" in error["error"]
    other = json.loads(backend.post_message("556", "hi"))
    assert other["ok"] is False and "only to channel 555" in other["error"]
    reply = json.loads(backend.post_message("555", "hi", reply_to="9"))
    assert reply["ok"] is False and "cannot reply" in reply["error"]
    assert backend.find_channel("general") is None
    assert discord_server.webhook_posts == []

    with pytest.raises(SystemExit):
        _make_backend()

    _webhook_config.clear()
    missing = json.loads(backend.post_message("555", "hi"))
    assert missing["ok"] is False and "No Discord webhook" in missing["error"]


def test_finish_pending_denied_and_rejected(discord_server: _DiscordServer) -> None:
    """finish reports pending, provider denial, and a token Discord rejects."""
    tools = auth_tools(DiscordAgent())
    assert "failed" in json.loads(tools["finish_discord_auth"]())["error"]

    started = json.loads(_start(tools))
    assert json.loads(tools["finish_discord_auth"]())["status"] == "pending"
    state = parse_qs(urlsplit(started["verification_uri"]).query)["state"][0]
    requests.get(
        LOOPBACK_REDIRECT_URI, params={"error": "access_denied", "state": state}, timeout=10
    )
    denied = json.loads(tools["finish_discord_auth"]())
    assert denied["ok"] is False and "access_denied" in denied["error"]

    for bad in ("bad", "html-token"):
        discord_server.token_answer = {**discord_server.token_answer, "access_token": bad}
        _approve(_start(tools))
        rejected = json.loads(tools["finish_discord_auth"]())
        assert rejected["ok"] is False and "rejected the new token" in rejected["error"]
    assert not _config.path.exists() and not _webhook_config.path.exists()


def test_finish_reports_unreachable_api(
    discord_server: _DiscordServer, refusing_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused /users/@me probe is reported and nothing is stored."""
    monkeypatch.setenv("DISCORD_API_BASE", f"http://127.0.0.1:{refusing_port}/api/v10")
    tools = auth_tools(DiscordAgent())
    _approve(_start(tools))
    result = json.loads(tools["finish_discord_auth"]())
    assert result["ok"] is False and "unreachable" in result["error"]
    assert not _config.path.exists()


def test_webhook_failures_hide_the_url(discord_server: _DiscordServer, refusing_port: int) -> None:
    """Failed webhook posts report the failure without leaking the webhook URL."""
    agent = DiscordAgent()
    tools = auth_tools(agent)
    _approve(_start(tools))
    assert json.loads(tools["finish_discord_auth"]())["ok"] is True
    for url in (
        f"{discord_server.base}/api/webhooks/901/wrong-secret",
        f"http://127.0.0.1:{refusing_port}/api/webhooks/901/refused-secret",
    ):
        _webhook_config.save({"url": url, "channel_id": "555", "guild_id": "777"})
        result = agent._backend.post_message("555", "hi")
        assert json.loads(result)["ok"] is False and "webhook post failed" in result
        assert "secret" not in result


def test_storage_failure_is_reported(
    discord_server: _DiscordServer, isolated_kiss_home: Path
) -> None:
    """A config directory that cannot be created fails finish cleanly."""
    tools = auth_tools(DiscordAgent())
    _approve(_start(tools))
    blocker = _config.path.parent
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("not a directory")
    result = json.loads(tools["finish_discord_auth"]())
    assert result["ok"] is False and "storing the sign-in failed" in result["error"]


def test_sign_in_without_webhook_and_bot_token_path(discord_server: _DiscordServer) -> None:
    """A grant without a webhook stores no webhook; a bot token replaces the sign-in."""
    discord_server.token_answer = {
        k: v for k, v in discord_server.token_answer.items() if k != "webhook"
    }
    agent = DiscordAgent()
    tools = auth_tools(agent)
    _approve(_start(tools))
    assert json.loads(tools["finish_discord_auth"]())["webhook_channel_id"] == ""
    assert not _webhook_config.path.exists()
    assert "webhook_channel_id" not in json.loads(tools["check_discord_auth"]())

    # Starting a sign-in and then storing a bot token cancels the sign-in.
    json.loads(_start(tools))
    assert json.loads(tools["authenticate_discord"](bot_token=_BOT_TOKEN))["ok"] is True
    assert "no sign-in in progress" in ConsentSession.finish("discord")[1]
    assert json.loads(_config.path.read_text())["bot_token"] == _BOT_TOKEN
    assert not agent._backend._user_auth
    assert discord_server.auth_headers[-1] == f"Bot {_BOT_TOKEN}"
    assert json.loads(tools["check_discord_auth"]())["auth"] == "bot"
    assert _make_backend()._token == _BOT_TOKEN

    assert tools["clear_discord_auth"]() == "Discord authentication cleared."
    assert not _config.path.exists() and not agent._is_authenticated()


def _expire_stored_token() -> dict[str, str]:
    """Rewrite config.json so the stored user token is already expired.

    Returns:
        The rewritten config dict.
    """
    cfg = _config.load_metadata()
    assert cfg is not None and cfg["refresh_token"] == "user-refresh-token"
    cfg["expires_at"] = str(time.time() - 60)
    _config.save(cfg)
    return cfg


def test_legacy_expired_token_is_refreshed_on_load(discord_server: _DiscordServer) -> None:
    """Loading an expired legacy user token rotates it with the public client."""
    tools = auth_tools(DiscordAgent())
    _approve(_start(tools))
    assert json.loads(tools["finish_discord_auth"]())["ok"] is True
    _expire_stored_token()
    discord_server.token_answer = {
        "access_token": "rotated-user-token",
        "refresh_token": "user-refresh-token-2",
        "expires_in": 3600,
        "token_type": "Bearer",
    }

    agent = DiscordAgent()  # legacy mode: _load_legacy_config refreshes
    assert agent._backend._user_auth and agent._backend._token == "rotated-user-token"
    assert len(discord_server.token_forms) == 2
    refresh = discord_server.token_forms[1]
    assert refresh == {
        "grant_type": "refresh_token",
        "refresh_token": "user-refresh-token",
        "client_id": "kiss-discord-app",
    }
    stored = json.loads(_config.path.read_text())
    assert stored["auth_mode"] == "user" and stored["client_id"] == "kiss-discord-app"
    assert stored["access_token"] == "rotated-user-token"
    assert stored["refresh_token"] == "user-refresh-token-2"
    assert abs(float(stored["expires_at"]) - (time.time() + 3600)) < 60

    # A token that is still valid is not refreshed again.
    DiscordAgent()
    assert len(discord_server.token_forms) == 2


def test_legacy_refused_refresh_keeps_the_old_token(
    discord_server: _DiscordServer, refusing_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused or unreachable refresh leaves config.json and the token alone."""
    tools = auth_tools(DiscordAgent())
    _approve(_start(tools))
    assert json.loads(tools["finish_discord_auth"]())["ok"] is True
    expired = _expire_stored_token()

    # Discord answers the refresh grant with an error body.
    discord_server.token_answer = {"error": "invalid_grant"}
    agent = DiscordAgent()
    assert discord_server.token_forms[-1]["grant_type"] == "refresh_token"
    assert agent._backend._token == _USER_TOKEN
    assert json.loads(_config.path.read_text()) == expired

    # The token endpoint is unreachable.
    monkeypatch.setenv("DISCORD_OAUTH_BASE", f"http://127.0.0.1:{refusing_port}")
    agent = DiscordAgent()
    assert agent._backend._token == _USER_TOKEN
    assert json.loads(_config.path.read_text()) == expired
    # The still-stored token keeps working against the API.
    assert json.loads(auth_tools(agent)["check_discord_auth"]())["auth"] == "user"


def test_second_sign_in_without_webhook_clears_the_old_webhook(
    discord_server: _DiscordServer,
) -> None:
    """A new grant without a channel choice stops posting through the old webhook."""
    agent = DiscordAgent()
    tools = auth_tools(agent)
    _approve(_start(tools))
    assert json.loads(tools["finish_discord_auth"]())["webhook_channel_id"] == "555"
    assert _webhook_config.path.exists()

    discord_server.token_answer = {
        k: v for k, v in discord_server.token_answer.items() if k != "webhook"
    }
    _approve(_start(tools))
    done = json.loads(tools["finish_discord_auth"]())
    assert done["ok"] is True and done["webhook_channel_id"] == ""
    assert not _webhook_config.path.exists()
    missing = json.loads(agent._backend.post_message("555", "hi"))
    assert missing["ok"] is False and "No Discord webhook" in missing["error"]
    assert discord_server.webhook_posts == []
    assert "webhook_channel_id" not in json.loads(tools["check_discord_auth"]())


def test_scrub_drops_every_refresh_field(discord_server: _DiscordServer) -> None:
    """Scrubbing removes the refresh token and its expiry/client metadata too."""
    _config.save(
        {
            "auth_mode": "user",
            "access_token": _USER_TOKEN,
            "refresh_token": "user-refresh-token",
            "expires_at": str(time.time() + 3600),
            "client_id": "kiss-discord-app",
            "application_id": "app-1",
        }
    )
    _scrub_config_token()
    assert json.loads(_config.path.read_text()) == {"auth_mode": "user", "application_id": "app-1"}

    # Nothing but secrets stored: the file goes away.
    _config.save({"access_token": _USER_TOKEN, "refresh_token": "r", "expires_at": "1"})
    _scrub_config_token()
    assert not _config.path.exists()


def test_missing_client_id(discord_server: _DiscordServer, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a client ID the sign-in names the environment variable."""
    monkeypatch.delenv("KISS_DISCORD_CLIENT_ID")
    result = json.loads(auth_tools(DiscordAgent())["authenticate_discord"]())
    assert result["ok"] is False and "KISS_DISCORD_CLIENT_ID" in result["error"]
    with pytest.raises(SystemExit):
        _make_backend()


def test_prompt_describes_click_allow_sign_in() -> None:
    """The prompt drives the browser sign-in and names bot tokens once."""
    prompt = DiscordAgent.channel_system_prompt
    assert "authenticate_discord() with no arguments" in prompt
    assert "finish_discord_auth()" in prompt
    assert "webhook.incoming" in prompt
    assert prompt.count("authenticate_discord(bot_token=...)") == 1
    assert "Developer Portal" not in prompt and "developers/applications" not in prompt


def test_muse_sign_in_uses_vault(
    discord_server: _DiscordServer,
    isolated_kiss_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In Muse mode the grant goes to the vault and only a surrogate stays here."""
    from kiss.agents.third_party_agents.muse_auth.client import vault_has_credentials

    setup_muse_env(
        monkeypatch,
        {
            "defaults": {"read": "allow", "write": "ask"},
            "services": {"discord": {"extra_hosts": ["127.0.0.1"]}},
        },
    )
    try:
        # A legacy user sign-in migrates into the vault as a bearer.
        _config.save({"auth_mode": "user", "access_token": _USER_TOKEN})
        migrated = DiscordAgent()
        assert migrated._backend._user_auth and vault_has_credentials("discord")
        assert json.loads(_config.path.read_text()) == {"auth_mode": "user"}

        agent = DiscordAgent()
        tools = auth_tools(agent)
        _approve(_start(tools))
        assert json.loads(tools["finish_discord_auth"]())["ok"] is True
        assert vault_has_credentials("discord")
        assert json.loads(_config.path.read_text()) == {"auth_mode": "user"}
        backend = agent._backend
        assert backend._muse and backend._user_auth
        assert backend._token.startswith("muse-sgt.discord.")
        assert json.loads(backend.list_guilds())["ok"] is True
        # The daemon swapped the surrogate for the real user token.
        assert discord_server.auth_headers[-1] == f"Bearer {_USER_TOKEN}"

        rewired = DiscordAgent()
        assert rewired._backend._user_auth and rewired._backend._muse

        assert tools["clear_discord_auth"]() == "Discord authentication cleared."
        assert not vault_has_credentials("discord")
    finally:
        teardown_muse_env()


def test_muse_migrates_refreshable_legacy_sign_in(
    discord_server: _DiscordServer,
    isolated_kiss_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy sign-in with a refresh token becomes a daemon-refreshed vault entry."""
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_dir

    setup_muse_env(
        monkeypatch,
        {
            "defaults": {"read": "allow", "write": "ask"},
            "services": {"discord": {"extra_hosts": ["127.0.0.1"]}},
        },
    )
    try:
        expires_at = time.time() + 3600
        _config.save(
            {
                "auth_mode": "user",
                "access_token": _USER_TOKEN,
                "refresh_token": "user-refresh-token",
                "expires_at": str(expires_at),
                "client_id": "kiss-discord-app",
            }
        )
        agent = DiscordAgent()
        assert agent._backend._muse and agent._backend._user_auth
        stored = json.loads((muse_auth_dir() / "vault" / "discord.json").read_text())
        info = stored["authorized_user_info"]
        assert info["kind"] == "oauth2_refresh_token"
        assert info["token_url"] == f"{discord_server.base}/api/oauth2/token"
        assert info["client_id"] == "kiss-discord-app"
        assert info["access_token"] == _USER_TOKEN
        assert info["refresh_token"] == "user-refresh-token"
        assert abs(float(info["expires_at"]) - expires_at) < 1
        # Every secret and its refresh metadata left config.json.
        assert json.loads(_config.path.read_text()) == {"auth_mode": "user"}
        # The daemon spends the migrated access token for the surrogate.
        assert json.loads(agent._backend.list_guilds())["ok"] is True
        assert discord_server.auth_headers[-1] == f"Bearer {_USER_TOKEN}"
    finally:
        teardown_muse_env()
