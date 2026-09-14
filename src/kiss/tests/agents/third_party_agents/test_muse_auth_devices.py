# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Muse-auth on Discord, Home Assistant, ntfy, and Govee.

SEA style, mirroring ``test_muse_auth_channels.py``: a REAL Muse-auth
daemon subprocess plus a REAL local HTTP server (stdlib
``ThreadedHTTPServer``) emulating the Discord / Home Assistant / ntfy /
Govee REST APIs — no mocks, patches, or fakes.  The emulated API
asserts every request arriving at the "network" carries the REAL
credential in the right header and scheme (proving the boundary swap
for the two non-Bearer schemes: Discord's ``Authorization: Bot`` and
Govee's ``Govee-API-Key``), while the agent side only ever holds
``muse-sgt.*`` surrogates.

The Home Assistant tests additionally exercise the consent-scoped
*insecure host* mechanism: plain-HTTP egress to a non-loopback host is
allowed only when that host was enrolled from an ``http://`` base URL,
and refused when the host is merely allowlisted.

Branch-coverage notes (unreachable without test doubles, so documented
instead of mocked):

* ``discord_agent._muse_authenticate``'s ``except Exception`` rollback
  and ``homeassistant_agent``/``ntfy_agent``'s ``MuseAuthError``
  handlers in their authenticate tools need the daemon to die between
  the enrollment and the validation call (a cross-process race).
* ``ntfy_agent._wire_muse``'s empty-surrogate return after
  ``bearer_surrogate`` needs the vault cleared between the store and
  the mint (a cross-process race); the tokenless return is covered.
* ``govee._muse_session``'s cached-return branch is covered, but the
  process-lifetime cache means the "no key and no vault" exit can only
  fire on the first call of a process; both are exercised by resetting
  the module cache between tests.
* ``vault._payload``'s missing-file return is guarded by the daemon
  (surrogate binding is validated before hosts are consulted, so the
  vault file exists).
* ``vault.resolve_token``'s Google refresh branch (including its
  ``trust_env=False`` session) contacts ``oauth2.googleapis.com``;
  exercising it needs Google's real token endpoint (pre-existing note
  in ``test_muse_auth``).  The unrefreshable-credential branch IS
  covered there.
* ``daemon._boundary``'s post-generation-read surrogate re-check and
  ``_execute``'s header-name-change abort and 6-deep-redirect final
  ``return resp`` need the vault cleared/re-kinded between two hops of
  one in-flight request (cross-process races).
* ``govee._muse_request``'s second-attempt raise needs two consecutive
  stale-surrogate rejections (a daemon that keeps rejecting across a
  remint) — a cross-process timing race.
* ``channel_main``'s single call into ``channel_override_config`` sits
  on the poll-runner startup path, which no suite drives end-to-end
  (pre-existing); the helper itself is covered in both directions here.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest
import requests

from kiss.agents.third_party_agents import govee
from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server
from kiss.agents.third_party_agents.discord_agent import DiscordChannelBackend
from kiss.agents.third_party_agents.discord_agent import _config as discord_config
from kiss.agents.third_party_agents.homeassistant_agent import HomeAssistantChannelBackend
from kiss.agents.third_party_agents.homeassistant_agent import _config as ha_config
from kiss.agents.third_party_agents.muse_auth import __main__ as muse_cli
from kiss.agents.third_party_agents.muse_auth._common import (
    builtin_hosts,
    muse_auth_dir,
    socket_path,
)
from kiss.agents.third_party_agents.muse_auth.client import (
    MuseBoundarySession,
    clear_credentials,
    grant,
    mint_surrogate,
    stop_daemon,
    store_credentials,
    vault_has_credentials,
)
from kiss.agents.third_party_agents.ntfy_agent import NtfyChannelBackend
from kiss.agents.third_party_agents.ntfy_agent import _config as ntfy_config

_REAL_DISCORD_TOKEN = "discord-real-secret"
_REAL_HA_TOKEN = "ha-real-secret"
_REAL_NTFY_TOKEN = "tk_real_ntfy_secret"
_REAL_GOVEE_KEY = "govee-real-secret"


class _DeviceApiHandler(BaseHTTPRequestHandler):
    """Emulated REST API recording every request's method/path/headers."""

    server: Any

    def _serve(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": {k: v for k, v in self.headers.items()},
                "body": body.decode("utf-8", errors="replace"),
            }
        )
        auth = next((v for k, v in self.headers.items() if k.lower() == "authorization"), "")
        path = self.path.split("?", 1)[0]
        if path in self.server.drop_after_recording:
            # Simulate an ambiguous failure: the request was received
            # (and any side effect applied) but the connection dies
            # before a response, so the client cannot know the outcome.
            self.close_connection = True
            self.connection.close()
            return
        if self.server.rotate_before_redirect and self.command == "POST" \
                and path == "/t-rotate":
            # The vault is rotated (new generation) between the initial
            # request and the redirect hop the daemon is about to follow.
            self.server.rotate_before_redirect()
            self.send_response(302)
            self.send_header("Location", "/t-redirect-target")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path.endswith(("/t302", "/t307")) and self.command == "POST":
            # Same-host redirects: 302 downgrades the POST to GET, 307
            # keeps the method and body.
            self.send_response(302 if path.endswith("/t302") else 307)
            self.send_header("Location", "/t-redirect-target")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.command == "DELETE" or "/reactions/" in path:
            # Discord's delete_message / add_reaction contract: 204,
            # no body.
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        content_type = "application/json; charset=utf-8"
        if auth.endswith("-invalid"):
            payload = json.dumps({"message": "401: Unauthorized", "code": 0}).encode()
        elif path.endswith("/users/@me/guilds"):
            payload = json.dumps([{"id": "G1", "name": "kiss-guild"}]).encode()
        elif path.endswith("/users/@me"):
            payload = json.dumps(
                {"id": "B1", "username": "kissbot", "discriminator": "0"}
            ).encode()
        elif "/channels/" in path and path.endswith("/messages"):
            payload = json.dumps({"id": "M1", "content": "sent"}).encode()
        elif path.endswith("/api/states"):
            payload = json.dumps([{"entity_id": "light.kitchen", "state": "on"}]).encode()
        elif "/api/services/" in path:
            payload = json.dumps([{"entity_id": "light.kitchen", "state": "off"}]).encode()
        elif path.endswith("/json"):
            content_type = "application/x-ndjson"
            events = [
                {"event": "open", "id": "o1"},
                {
                    "event": "message",
                    "id": "m1",
                    "time": 1700000001,
                    "topic": "t1",
                    "message": "hello from ntfy",
                    "tags": [],
                },
            ]
            payload = ("\n".join(json.dumps(e) for e in events) + "\n").encode()
        elif path.endswith("/user/devices"):
            payload = json.dumps(
                {
                    "data": [
                        {"sku": "H6008", "device": "AA:BB", "deviceName": "Desk lamp"},
                        {"sku": "H705E", "device": "CC:DD", "deviceName": "String lights"},
                    ]
                }
            ).encode()
        elif path.endswith("/device/state"):
            payload = json.dumps(
                {"data": {"device": "AA:BB", "capabilities": [{"state": {"value": 1}}]}}
            ).encode()
        elif path.endswith("/device/control"):
            payload = json.dumps({"code": 200, "data": {}}).encode()
        else:
            payload = json.dumps({"id": "m1", "ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a GET request."""
        self._serve()

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a POST request."""
        self._serve()

    def do_DELETE(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a DELETE request."""
        self._serve()

    def do_PATCH(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a PATCH request."""
        self._serve()

    def do_PUT(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a PUT request."""
        self._serve()

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence request logging."""


class _DeviceApiServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records requests for verification."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _DeviceApiHandler)
        self.requests: list[dict[str, Any]] = []
        # Paths that record the request then kill the connection with no
        # response (ambiguous-failure / replay-safety tests).
        self.drop_after_recording: set[str] = set()
        # Optional callback invoked before a POST /t-rotate redirect, to
        # rotate the vault mid-request (generation-abort tests).
        self.rotate_before_redirect: Any = None

    def header(self, name: str, index: int = -1) -> str:
        """Return a recorded request header (case-insensitive).

        Args:
            name: Header name.
            index: Which recorded request to inspect (default: last).

        Returns:
            The header value, or ``""`` when absent.
        """
        headers = self.requests[index]["headers"]
        return next((v for k, v in headers.items() if k.lower() == name.lower()), "")


@pytest.fixture()
def api_server() -> Any:
    """Run the emulated device REST API on a loopback port."""
    server = _DeviceApiServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    stop_http_server(server, thread)


def _local_ip() -> str:
    """Return this machine's primary non-loopback IPv4 address.

    Returns:
        The address, or ``""`` when the machine has none (offline box).
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # connect() on UDP sends no packets; it only picks a route.
        probe.connect(("8.8.8.8", 80))
        ip = str(probe.getsockname()[0])
    except OSError:
        return ""
    finally:
        probe.close()
    return "" if ip.startswith("127.") else ip


@pytest.fixture()
def lan_api_server() -> Any:
    """Run the emulated API reachable via the machine's non-loopback IP."""
    ip = _local_ip()
    if not ip:
        pytest.skip("machine has no non-loopback IPv4 address")
    server = _DeviceApiServer(("0.0.0.0", 0))
    server.lan_ip = ip  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    stop_http_server(server, thread)


@pytest.fixture()
def muse_env(isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Enable Muse-auth inside an isolated ``KISS_HOME`` with a live daemon.

    Discord and Govee get a loopback ``extra_hosts`` policy entry
    (their real hosts are fixed); Home Assistant and ntfy deliberately
    get none, so their tests prove the enrollment-time host extension.
    """
    monkeypatch.setenv("KISS_MUSE_AUTH", "1")
    directory = muse_auth_dir()
    directory.mkdir(parents=True, exist_ok=True)
    policy = {
        "defaults": {"read": "allow", "write": "ask"},
        "services": {
            "discord": {"extra_hosts": ["127.0.0.1"]},
            "govee": {"extra_hosts": ["127.0.0.1"]},
        },
    }
    (directory / "policy.json").write_text(json.dumps(policy))
    yield isolated_kiss_home
    stop_daemon()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and socket_path().exists():
        time.sleep(0.05)


def _discord_backend(api_server: _DeviceApiServer) -> DiscordChannelBackend:
    """Build a Discord backend pointed at the emulator.

    Args:
        api_server: The emulated Discord API.

    Returns:
        An unconnected backend with the emulator as its API base.
    """
    return DiscordChannelBackend(
        api_base=f"http://127.0.0.1:{api_server.server_address[1]}/api/v10"
    )


def test_discord_bot_scheme_boundary_swap(muse_env: Path, api_server: _DeviceApiServer) -> None:
    """The Discord surrogate bearer becomes a real ``Authorization: Bot``."""
    discord_config.save(
        {"bot_token": _REAL_DISCORD_TOKEN, "application_id": "app1", "guild_ids": ""}
    )
    backend = _discord_backend(api_server)
    assert backend.connect()
    assert backend._bot_token.startswith("muse-sgt.discord.")
    assert vault_has_credentials("discord")
    # connect() validated /users/@me (a read) through the boundary with
    # the REAL Bot-scheme header; the surrogate never hit the network.
    assert api_server.requests[-1]["path"].endswith("/users/@me")
    assert api_server.header("Authorization") == f"Bot {_REAL_DISCORD_TOKEN}"
    # Migration scrubbed the plaintext token but kept the metadata.
    assert json.loads(discord_config.path.read_text()) == {"application_id": "app1"}

    # Reads (guild list) pass without grants.
    guilds = json.loads(backend.list_guilds())
    assert guilds["ok"] is True
    assert guilds["guilds"] == [{"id": "G1", "name": "kiss-guild"}]

    # The typing indicator is an ephemeral POST classified as a read,
    # so it can never burn the write grant meant for the message send.
    backend.send_typing("C1")
    assert api_server.requests[-1]["path"].endswith("/typing")

    # A write (post_message) is asked; the network never saw the attempt.
    before = len(api_server.requests)
    denied = json.loads(backend.post_message("C1", "hello"))
    assert denied["ok"] is False
    assert "MUSE_AUTH_DENIED" in denied["error"]
    assert "grant discord write" in denied["error"]
    assert len(api_server.requests) == before

    # A once-grant admits exactly one write.
    grant("discord", "write", "once")
    sent = json.loads(backend.post_message("C1", "hello"))
    assert sent["ok"] is True
    assert api_server.header("Authorization") == f"Bot {_REAL_DISCORD_TOKEN}"
    assert json.loads(backend.post_message("C1", "again"))["ok"] is False

    # Discord's 204-no-body DELETE contract survives the boundary.
    grant("discord", "write", "once")
    deleted = json.loads(backend.delete_message("C1", "M1"))
    assert deleted == {"ok": True}

    # PATCH (edit) and PUT (react, 204 no body) also run at the boundary.
    grant("discord", "write", "once")
    assert json.loads(backend.edit_message("C1", "M1", "edited"))["ok"] is True
    assert api_server.requests[-1]["method"] == "PATCH"
    grant("discord", "write", "once")
    assert json.loads(backend.add_reaction("C1", "M1", "👍")) == {"ok": True}
    assert api_server.requests[-1]["method"] == "PUT"
    assert api_server.header("Authorization") == f"Bot {_REAL_DISCORD_TOKEN}"

    # The surrogate never crossed the network boundary.
    for request in api_server.requests:
        assert "muse-sgt." not in json.dumps(request["headers"])


def test_discord_vault_first_and_make_backend(
    muse_env: Path, api_server: _DeviceApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vault-first connects survive config removal; poll mode is vault-aware."""
    from kiss.agents.third_party_agents.discord_agent import _make_backend

    monkeypatch.setenv(
        "DISCORD_API_BASE", f"http://127.0.0.1:{api_server.server_address[1]}/api/v10"
    )
    # Token-only config: the scrub deletes the whole file.
    discord_config.save({"bot_token": _REAL_DISCORD_TOKEN})
    backend = _discord_backend(api_server)
    assert backend.connect()
    assert not discord_config.path.exists()

    # The vault alone still connects (mint path, no config anywhere).
    backend2 = _discord_backend(api_server)
    assert backend2.connect()
    assert backend2._bot_token.startswith("muse-sgt.discord.")

    # Poll-mode backends wire the same surrogate.
    polled = _make_backend()
    assert polled._muse and polled._bot_token.startswith("muse-sgt.discord.")

    # Without vault or config there is nothing to connect with.
    clear_credentials("discord")
    backend3 = _discord_backend(api_server)
    assert not backend3.connect()
    assert "No Discord credential" in backend3._connection_info
    with pytest.raises(SystemExit):
        _make_backend()


def test_discord_authenticate_rotation_rollback_clear(
    muse_env: Path, api_server: _DeviceApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """authenticate_discord enrolls, validates, rolls back, and rotates."""
    from kiss.agents.third_party_agents.discord_agent import DiscordAgent

    monkeypatch.setenv(
        "DISCORD_API_BASE", f"http://127.0.0.1:{api_server.server_address[1]}/api/v10"
    )
    agent = DiscordAgent()
    assert agent._backend._bot_token == ""
    tools = {t.__name__: t for t in agent._get_auth_tools()}
    assert "Not authenticated" in tools["check_discord_auth"]()

    # Enrollment validates through the boundary and never writes the
    # token to disk (no metadata was given, so no config file at all).
    result = json.loads(tools["authenticate_discord"]("tok-first"))
    assert result["ok"] is True and result["username"] == "kissbot"
    assert not discord_config.path.exists()
    assert api_server.header("Authorization") == "Bot tok-first"
    audit = (muse_auth_dir() / "audit.jsonl").read_text()
    assert any(
        json.loads(line)["service"] == "discord" for line in audit.splitlines()
    )
    assert json.loads(tools["check_discord_auth"]())["ok"] is True

    # Rotation: the second token replaces the first at the boundary.
    assert json.loads(tools["authenticate_discord"]("tok-second", "app2", "G1,G2"))["ok"] is True
    assert json.loads(discord_config.path.read_text()) == {
        "application_id": "app2",
        "guild_ids": "G1,G2",
    }
    json.loads(tools["check_discord_auth"]())
    assert api_server.header("Authorization") == "Bot tok-second"

    # An invalid token is rolled back: nothing stays enrolled.
    failed = json.loads(tools["authenticate_discord"]("tok-invalid"))
    assert failed["ok"] is False
    assert not vault_has_credentials("discord")
    assert agent._backend._bot_token == ""

    # A token the daemon itself rejects (embedded newline) surfaces the
    # enrollment error and leaves the vault empty too.
    failed = json.loads(tools["authenticate_discord"]("bad\ntoken"))
    assert failed["ok"] is False
    assert "invalid credential token value" in failed["error"]
    assert not vault_has_credentials("discord")

    # Clearing wipes both the config metadata and the vault enrollment.
    assert json.loads(tools["authenticate_discord"]("tok-third"))["ok"] is True
    assert "cleared" in tools["clear_discord_auth"]()
    assert not vault_has_credentials("discord")
    assert mint_surrogate("discord") is None


def test_homeassistant_insecure_host_consent(muse_env: Path, lan_api_server: Any) -> None:
    """Plain-HTTP egress needs the consent-scoped insecure enrollment."""
    ip = lan_api_server.lan_ip
    base_url = f"http://{ip}:{lan_api_server.server_address[1]}"

    # An allowlisted-but-not-insecure host refuses plaintext: enroll the
    # host WITHOUT the insecure flag and watch the Sentinel deny http.
    store_credentials(
        "homeassistant", {"kind": "bearer", "token": _REAL_HA_TOKEN}, [], hosts=(ip,)
    )
    handle = mint_surrogate("homeassistant")
    assert handle is not None
    session = MuseBoundarySession("homeassistant")
    resp = session.get(
        f"{base_url}/api/states",
        headers={"Authorization": f"Bearer {handle.token}"},
    )
    assert resp.status_code == 403
    assert "non-HTTPS" in resp.text
    assert not lan_api_server.requests
    clear_credentials("homeassistant")

    # The same host enrolled from an http:// base URL is allowed.
    ha_config.save({"base_url": base_url, "token": _REAL_HA_TOKEN})
    backend = HomeAssistantChannelBackend()
    assert backend.connect()
    assert backend._token.startswith("muse-sgt.homeassistant.")
    assert "(Muse-auth)" in backend._connection_info
    # Migration scrubbed the token, keeping the non-secret base_url.
    assert json.loads(ha_config.path.read_text()) == {"base_url": base_url}

    states = json.loads(backend.ha_get_states())
    assert states["ok"] is True
    assert lan_api_server.requests[-1]["path"] == "/api/states"
    assert lan_api_server.header("Authorization") == f"Bearer {_REAL_HA_TOKEN}"

    # Writes (service calls) are asked, then admitted by a grant.
    denied = json.loads(backend.ha_call_service("light", "turn_off", "light.kitchen"))
    assert denied["ok"] is False and "grant homeassistant write" in denied["error"]
    grant("homeassistant", "write", "once")
    called = json.loads(backend.ha_call_service("light", "turn_off", "light.kitchen"))
    assert called["ok"] is True
    assert lan_api_server.requests[-1]["path"] == "/api/services/light/turn_off"

    # Vault-first: with the token scrubbed, a fresh backend reconnects.
    backend2 = HomeAssistantChannelBackend()
    assert backend2.connect()
    assert json.loads(backend2.ha_get_states("light.kitchen"))["ok"] is True

    # The surrogate never crossed the network boundary.
    for request in lan_api_server.requests:
        assert "muse-sgt." not in json.dumps(request["headers"])


def test_homeassistant_authenticate_and_clear_tools(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """authenticate_homeassistant re-enrolls rotated tokens; clear wipes."""
    from kiss.agents.third_party_agents.homeassistant_agent import HomeAssistantAgent

    base_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    agent = HomeAssistantAgent()
    tools = {t.__name__: t for t in agent._get_auth_tools()}
    assert "Not configured" in tools["check_homeassistant_auth"]()
    assert "cannot be empty" in tools["authenticate_homeassistant"]("", "")

    assert json.loads(tools["authenticate_homeassistant"](base_url, "ha-first"))["ok"] is True
    assert json.loads(ha_config.path.read_text()) == {"base_url": base_url}
    assert json.loads(tools["check_homeassistant_auth"]())["ok"] is True
    assert json.loads(agent._backend.ha_get_states())["ok"] is True
    assert api_server.header("Authorization") == "Bearer ha-first"

    # Rotation: the second token must actually be used.
    assert json.loads(tools["authenticate_homeassistant"](base_url, "ha-second"))["ok"] is True
    assert json.loads(agent._backend.ha_get_states())["ok"] is True
    assert api_server.header("Authorization") == "Bearer ha-second"

    # A fresh agent construction wires the vault surrogate offline.
    agent2 = HomeAssistantAgent()
    assert agent2._backend._token.startswith("muse-sgt.homeassistant.")

    assert "cleared" in tools["clear_homeassistant_auth"]()
    assert not vault_has_credentials("homeassistant")
    backend = HomeAssistantChannelBackend()
    assert not backend.connect()
    assert "No Home Assistant config found." in backend._connection_info


def test_ntfy_token_enrollment_and_scrub(muse_env: Path, api_server: _DeviceApiServer) -> None:
    """A configured ntfy token is enrolled with the self-hosted server host."""
    server_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    ntfy_config.save({"topic": "t1", "server": server_url, "token": _REAL_NTFY_TOKEN})
    backend = NtfyChannelBackend()
    assert backend.connect()
    assert backend._muse and backend._token.startswith("muse-sgt.ntfy.")
    assert "(Muse-auth)" in backend._connection_info
    # The scrub kept the non-secret config so load() still works.
    assert json.loads(ntfy_config.path.read_text()) == {"topic": "t1", "server": server_url}

    # Polling is a read: allowed unattended, with the REAL token at the
    # network (the host came from the enrollment, not from any policy).
    polled = json.loads(backend.poll_topic())
    assert polled["ok"] is True
    assert polled["messages"][0]["text"] == "hello from ntfy"
    assert api_server.header("Authorization") == f"Bearer {_REAL_NTFY_TOKEN}"

    # Publishing is a write: asked, then admitted by a grant.
    denied = json.loads(backend.publish_notification("build done"))
    assert denied["ok"] is False and "HTTP 403" in denied["error"]
    grant("ntfy", "write", "once")
    published = json.loads(backend.publish_notification("build done", title="CI"))
    assert published == {"ok": True, "id": "m1"}
    assert api_server.header("X-Title") == "CI"
    assert api_server.header("Authorization") == f"Bearer {_REAL_NTFY_TOKEN}"

    # send_message (the channel-runner path) also runs at the boundary.
    grant("ntfy", "write", "once")
    backend.send_message("t1", "channel text")
    assert api_server.requests[-1]["body"] == "channel text"

    # Vault-first reconnect after the scrub.
    backend2 = NtfyChannelBackend()
    assert backend2.connect() and backend2._muse

    for request in api_server.requests:
        assert "muse-sgt." not in json.dumps(request["headers"])


def test_ntfy_tokenless_stays_legacy_and_rotation(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """Tokenless ntfy has no credential to protect and skips the boundary."""
    from kiss.agents.third_party_agents.ntfy_agent import NtfyAgent

    server_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    agent = NtfyAgent()
    tools = {t.__name__: t for t in agent._get_auth_tools()}
    assert "Not configured" in tools["check_ntfy_auth"]()

    # Tokenless configuration: legacy direct path, no vault, no grants.
    assert json.loads(tools["authenticate_ntfy"]("t1", server_url))["ok"] is True
    assert not agent._backend._muse
    assert not vault_has_credentials("ntfy")
    assert json.loads(agent._backend.publish_notification("open topic"))["ok"] is True
    assert api_server.header("Authorization") == ""

    # Adding a token moves the backend onto the boundary.
    assert json.loads(tools["authenticate_ntfy"]("t1", server_url, "tk-first"))["ok"] is True
    assert agent._backend._muse
    grant("ntfy", "write", "once")
    assert json.loads(agent._backend.publish_notification("with token"))["ok"] is True
    assert api_server.header("Authorization") == "Bearer tk-first"

    # Rotation: the second token must actually be used.
    assert json.loads(tools["authenticate_ntfy"]("t1", server_url, "tk-second"))["ok"] is True
    grant("ntfy", "write", "once")
    assert json.loads(agent._backend.publish_notification("rotated"))["ok"] is True
    assert api_server.header("Authorization") == "Bearer tk-second"

    # Re-configuring tokenless drops the stale vault credential.
    assert json.loads(tools["authenticate_ntfy"]("t1", server_url))["ok"] is True
    assert not vault_has_credentials("ntfy")
    assert not agent._backend._muse

    # Clearing resets the backend to the legacy direct path.
    assert json.loads(tools["authenticate_ntfy"]("t1", server_url, "tk-third"))["ok"] is True
    assert "cleared" in tools["clear_ntfy_auth"]()
    assert not vault_has_credentials("ntfy")
    assert agent._backend._http is requests


def test_govee_header_kind_credential_and_action_classes(
    muse_env: Path, api_server: _DeviceApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Govee surrogate bearer becomes a real ``Govee-API-Key`` header."""
    monkeypatch.setattr(
        govee, "API", f"http://127.0.0.1:{api_server.server_address[1]}/router/api/v1"
    )
    monkeypatch.setattr(govee, "_MUSE_SESSION", None)
    monkeypatch.setenv("GOVEE_API_KEY", _REAL_GOVEE_KEY)

    # Device listing (GET) is a read; the excluded-name filter applies.
    devices = govee.list_devices()
    assert [d["deviceName"] for d in devices] == ["Desk lamp"]
    assert vault_has_credentials("govee")
    assert api_server.header("Govee-API-Key") == _REAL_GOVEE_KEY
    assert api_server.header("Authorization") == ""

    # The state query is a POST but classifies as a read.
    state = govee.state(devices[0])
    assert state["data"]["device"] == "AA:BB"
    assert api_server.requests[-1]["method"] == "POST"
    assert api_server.requests[-1]["path"].endswith("/device/state")

    # Actuation is a write: refused without a grant (the CLI exits with
    # the grant instructions), admitted with one.
    before = len(api_server.requests)
    with pytest.raises(SystemExit, match="grant govee write"):
        govee.control(devices[0], "devices.capabilities.on_off", "powerSwitch", 1)
    assert len(api_server.requests) == before
    grant("govee", "write", "once")
    result = govee.control(devices[0], "devices.capabilities.on_off", "powerSwitch", 1)
    assert result["code"] == 200
    assert api_server.header("Govee-API-Key") == _REAL_GOVEE_KEY

    # The real key was dropped from this process's environment at
    # enrollment, and the vault alone keeps the CLI working.
    assert "GOVEE_API_KEY" not in os.environ
    monkeypatch.setattr(govee, "_MUSE_SESSION", None)
    assert [d["deviceName"] for d in govee.list_devices()] == ["Desk lamp"]

    # Surrogates die with the daemon; the CLI re-mints and retries once.
    stop_daemon()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and socket_path().exists():
        time.sleep(0.05)
    state_after_restart = govee.state({"sku": "H6008", "device": "AA:BB"})
    assert state_after_restart["data"]["device"] == "AA:BB"
    assert api_server.header("Govee-API-Key") == _REAL_GOVEE_KEY

    # Without the env var and the vault there is nothing to enroll.
    clear_credentials("govee")
    monkeypatch.setattr(govee, "_MUSE_SESSION", None)
    with pytest.raises(SystemExit, match="GOVEE_API_KEY"):
        govee.list_devices()

    for request in api_server.requests:
        assert "muse-sgt." not in json.dumps(request["headers"])


def test_cli_import_devices(
    muse_env: Path,
    api_server: _DeviceApiServer,
    lan_api_server: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``muse_auth import`` migrates each device connector's credential."""
    # Discord: the stored header credential carries the Bot prefix, and
    # the import itself scrubs the plaintext (keeping the metadata,
    # which stays readable through load_metadata despite the missing
    # required bot_token key).
    assert muse_cli.main(["import", "discord"]) == 1  # no legacy config
    discord_config.save({"bot_token": _REAL_DISCORD_TOKEN, "application_id": "app9"})
    assert muse_cli.main(["import", "discord"]) == 0
    assert json.loads(discord_config.path.read_text()) == {"application_id": "app9"}
    assert discord_config.load() is None
    assert discord_config.load_metadata() == {"application_id": "app9"}
    backend = _discord_backend(api_server)
    assert backend.connect()
    assert api_server.header("Authorization") == f"Bot {_REAL_DISCORD_TOKEN}"

    # Home Assistant: the http:// base URL host imports as insecure.
    ip = lan_api_server.lan_ip
    base_url = f"http://{ip}:{lan_api_server.server_address[1]}"
    ha_config.save({"base_url": base_url, "token": _REAL_HA_TOKEN})
    assert muse_cli.main(["import", "homeassistant"]) == 0
    assert json.loads(ha_config.path.read_text()) == {"base_url": base_url}
    ha_backend = HomeAssistantChannelBackend()
    assert ha_backend.connect()
    assert json.loads(ha_backend.ha_get_states())["ok"] is True
    assert lan_api_server.header("Authorization") == f"Bearer {_REAL_HA_TOKEN}"

    # ntfy: the self-hosted server host is enrolled with the token.
    server_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    ntfy_config.save({"topic": "t1", "server": server_url, "token": _REAL_NTFY_TOKEN})
    assert muse_cli.main(["import", "ntfy"]) == 0
    assert json.loads(ntfy_config.path.read_text()) == {"topic": "t1", "server": server_url}
    ntfy_backend = NtfyChannelBackend()
    assert ntfy_backend.connect() and ntfy_backend._muse
    assert json.loads(ntfy_backend.poll_topic())["ok"] is True
    assert api_server.header("Authorization") == f"Bearer {_REAL_NTFY_TOKEN}"

    # Firecrawl cloud-only import (no base_url) must still enroll the
    # cloud origin, or the migrated key cannot reach api.firecrawl.dev.
    from kiss.agents.third_party_agents.firecrawl_agent import _config as firecrawl_config
    from kiss.agents.third_party_agents.muse_auth.daemon import MuseAuthDaemon

    firecrawl_config.save({"api_key": "fc-cloud-key"})
    assert muse_cli.main(["import", "firecrawl"]) == 0
    assert not firecrawl_config.path.exists()
    daemon = MuseAuthDaemon()
    assert daemon.sentinel.origin_allowed("firecrawl", "https://api.firecrawl.dev/v2/scrape")
    clear_credentials("firecrawl")

    # A userinfo-bearing self-hosted URL is refused (no password
    # migrated, nothing stored).
    firecrawl_config.save(
        {"api_key": "fc-secret", "base_url": "https://alice:pw@fc.example"}
    )
    assert muse_cli.main(["import", "firecrawl"]) == 1
    assert not vault_has_credentials("firecrawl")
    assert firecrawl_config.path.exists()  # plaintext left for the user to fix
    firecrawl_config.clear()

    # Govee: imported from the environment, not from a config file.
    monkeypatch.delenv("GOVEE_API_KEY", raising=False)
    assert muse_cli.main(["import", "govee"]) == 1
    monkeypatch.setenv("GOVEE_API_KEY", _REAL_GOVEE_KEY)
    assert muse_cli.main(["import", "govee"]) == 0
    assert vault_has_credentials("govee")

    # A config missing its token key is rejected.
    ha_config.save({"base_url": base_url, "token": ""})
    assert muse_cli.main(["import", "homeassistant"]) == 1
    out = capsys.readouterr()
    assert "no 'token' key" in out.err

    # Firecrawl plain-HTTP self-host bases enroll their consent-scoped
    # insecure host through the same import path.
    from kiss.agents.third_party_agents.firecrawl_agent import _insecure_extra_hosts

    assert _insecure_extra_hosts("http://firecrawl.lan:3002") == ("firecrawl.lan:3002",)
    assert _insecure_extra_hosts("https://firecrawl.lan:3002") == ()
    assert _insecure_extra_hosts("http://127.0.0.1:3002") == ()
    assert muse_cli._import_hosts("firecrawl", {"base_url": "http://firecrawl.lan:3002"}) == (
        ("firecrawl.lan:3002",),
        ("firecrawl.lan:3002",),
    )


def test_scrub_helpers_tolerate_malformed_configs(muse_env: Path) -> None:
    """The metadata readers and scrubbers never crash on odd files."""
    from kiss.agents.third_party_agents.discord_agent import (
        _scrub_config_token as discord_scrub,
    )
    from kiss.agents.third_party_agents.homeassistant_agent import _read_base_url
    from kiss.agents.third_party_agents.homeassistant_agent import (
        _scrub_config_token as ha_scrub,
    )
    from kiss.agents.third_party_agents.ntfy_agent import _scrub_config_token as ntfy_scrub

    # Absent files: every helper is a no-op.
    discord_scrub()
    ha_scrub()
    ntfy_scrub()
    assert _read_base_url() == ""

    # Non-dict JSON: tolerated everywhere.
    for config in (discord_config, ha_config, ntfy_config):
        config.path.parent.mkdir(parents=True, exist_ok=True)
        config.path.write_text("[1, 2]")
    discord_scrub()
    ha_scrub()
    ntfy_scrub()
    assert _read_base_url() == ""

    # Files without the secret key are left alone.
    discord_config.path.write_text(json.dumps({"application_id": "app1"}))
    discord_scrub()
    assert json.loads(discord_config.path.read_text()) == {"application_id": "app1"}
    ha_config.path.write_text(json.dumps({"base_url": "https://ha.example"}))
    ha_scrub()
    assert _read_base_url() == "https://ha.example"
    ntfy_config.path.write_text(json.dumps({"topic": "t1"}))
    ntfy_scrub()
    assert json.loads(ntfy_config.path.read_text()) == {"topic": "t1"}

    # A token-only Home Assistant config is deleted outright.
    ha_config.path.write_text(json.dumps({"token": "x"}))
    ha_scrub()
    assert not ha_config.path.exists()

    # The CLI's import-time scrubber tolerates the same odd files.
    muse_cli._scrub_imported_config("homeassistant")  # file absent
    ha_config.path.write_text("[1, 2]")
    muse_cli._scrub_imported_config("homeassistant")  # not a dict
    ha_config.path.write_text(json.dumps({"base_url": "https://ha.example"}))
    muse_cli._scrub_imported_config("homeassistant")  # no token key
    assert json.loads(ha_config.path.read_text()) == {"base_url": "https://ha.example"}
    ha_config.path.write_text(json.dumps({"token": "x"}))
    muse_cli._scrub_imported_config("homeassistant")  # token-only file
    assert not ha_config.path.exists()

    # load_metadata tolerates missing, malformed, and non-dict files.
    assert ha_config.load_metadata() is None
    ha_config.path.write_text("{not json")
    assert ha_config.load_metadata() is None
    ha_config.path.write_text("[1, 2]")
    assert ha_config.load_metadata() is None
    ha_config.path.write_text(json.dumps({"base_url": "https://ha.example"}))
    assert ha_config.load_metadata() == {"base_url": "https://ha.example"}

    # Poll-mode startup resolves model/budget overrides from a scrubbed
    # config through the same tolerant reader.
    from kiss.agents.third_party_agents._channel_agent_utils import channel_override_config
    from kiss.agents.third_party_agents.discord_agent import DiscordAgent

    overrides = {"application_id": "app9", "channel_model_name": "model-x"}
    discord_config.path.write_text(json.dumps(overrides))
    assert discord_config.load() is None
    assert channel_override_config(DiscordAgent) == overrides
    assert channel_override_config(_DeviceApiServer) is None  # module has no _config


def test_discord_legacy_mode_unchanged(
    isolated_kiss_home: Path, api_server: _DeviceApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With Muse-auth off, Discord keeps its plaintext-config behavior."""
    from kiss.agents.third_party_agents.discord_agent import DiscordAgent, _make_backend

    monkeypatch.setenv("KISS_MUSE_AUTH", "0")
    monkeypatch.setenv(
        "DISCORD_API_BASE", f"http://127.0.0.1:{api_server.server_address[1]}/api/v10"
    )
    backend = _discord_backend(api_server)
    assert not backend.connect()
    assert backend._connection_info == "No Discord token found."

    agent = DiscordAgent()
    tools = {t.__name__: t for t in agent._get_auth_tools()}
    assert json.loads(tools["authenticate_discord"](_REAL_DISCORD_TOKEN))["ok"] is True
    # Legacy mode stores the plaintext token and sends it directly.
    assert json.loads(discord_config.path.read_text())["bot_token"] == _REAL_DISCORD_TOKEN
    assert api_server.header("Authorization") == f"Bot {_REAL_DISCORD_TOKEN}"
    assert not (muse_auth_dir() / "vault").exists()

    agent2 = DiscordAgent()
    assert agent2._backend._bot_token == _REAL_DISCORD_TOKEN
    polled = _make_backend()
    assert polled._bot_token == _REAL_DISCORD_TOKEN and not polled._muse
    backend2 = _discord_backend(api_server)
    assert backend2.connect()
    assert backend2._connection_info == "Authenticated as kissbot#0"

    assert "cleared" in tools["clear_discord_auth"]()
    assert not discord_config.path.exists()


def test_discord_muse_connect_failure_paths(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """A vault-enrolled but invalid token fails connect() with the API error."""
    discord_config.save({"bot_token": "tok-invalid"})
    backend = _discord_backend(api_server)
    assert not backend.connect()
    assert "Discord auth failed" in backend._connection_info
    # The enrollment itself stays: connect() only validates, it is the
    # authenticate tool that rolls back bad tokens.
    assert vault_has_credentials("discord")


def test_homeassistant_no_credential_and_bad_host(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """Vault-less connects fail cleanly; malformed hosts fail enrollment."""
    from kiss.agents.third_party_agents.homeassistant_agent import HomeAssistantAgent

    base_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    # A base_url without any token (config or vault) cannot connect.
    ha_config.path.parent.mkdir(parents=True, exist_ok=True)
    ha_config.path.write_text(json.dumps({"base_url": base_url}))
    backend = HomeAssistantChannelBackend()
    assert not backend.connect()
    assert "No Home Assistant credential" in backend._connection_info

    # A base_url whose host is syntactically invalid is rejected up
    # front (before any vault or config change), not after a partial
    # migration.
    agent = HomeAssistantAgent()
    tools = {t.__name__: t for t in agent._get_auth_tools()}
    result = tools["authenticate_homeassistant"]("http://bad!:1", "tok")
    assert "http(s):// URL with a hostname" in result
    assert not vault_has_credentials("homeassistant")


def test_ntfy_enrollment_host_helpers_and_bad_host(muse_env: Path) -> None:
    """Host helpers skip the public server; bad hosts surface cleanly."""
    from kiss.agents.third_party_agents.ntfy_agent import (
        NtfyAgent,
        _extra_hosts,
        _insecure_extra_hosts,
    )

    # An ntfy token is origin-bound: exactly the configured server's
    # origin (host AND port) is enrolled (there is no built-in
    # allowlist, so a private token can never be spent against public
    # ntfy.sh), and only plain-HTTP non-loopback servers are insecure.
    assert builtin_hosts("ntfy") == ()
    assert _extra_hosts("https://ntfy.sh") == ("ntfy.sh:443",)
    assert _extra_hosts("not a url") == ()
    assert _extra_hosts("https://ntfy.example.com") == ("ntfy.example.com:443",)
    assert _extra_hosts("http://ntfy.example.com:8080") == ("ntfy.example.com:8080",)
    assert _insecure_extra_hosts("http://ntfy.example.com") == ("ntfy.example.com:80",)
    assert _insecure_extra_hosts("http://ntfy.example.com:8080") == ("ntfy.example.com:8080",)
    assert _insecure_extra_hosts("https://ntfy.example.com") == ()
    assert _insecure_extra_hosts("http://127.0.0.1:99") == ()

    agent = NtfyAgent()
    tools = {t.__name__: t for t in agent._get_auth_tools()}
    result = tools["authenticate_ntfy"]("t1", "http://bad!:1", "tok")
    assert "http(s):// URL with a hostname" in result
    assert not vault_has_credentials("ntfy")


def test_vault_direct_reads(muse_env: Path) -> None:
    """The daemon-side vault reads tolerate absent entries and no generation.

    Exercises ``CredentialVault`` directly (it is the daemon's own
    object, not a test double): a real credential is enrolled through
    the daemon, then read back by a fresh vault instance pointed at the
    same ``KISS_HOME``.
    """
    from kiss.agents.third_party_agents.muse_auth.vault import CredentialVault

    store_credentials(
        "govee", {"kind": "header", "header": "Govee-API-Key", "token": "gk"}, []
    )
    vault = CredentialVault()
    # No generation pin: the header credential resolves without a check.
    assert vault.resolve_credential("govee") == ("header", "Govee-API-Key", "gk")
    # A fresh generation nonce was written and is stable across reads.
    assert vault.generation("govee") == vault.generation("govee") != ""
    # Absent entries fail closed: no hosts, empty generation, no error.
    assert vault.generation("not-enrolled") == ""
    assert vault.enrolled_hosts("not-enrolled") == ()
    assert vault.enrolled_insecure_hosts("not-enrolled") == ()


def test_store_rejects_invalid_insecure_hosts(muse_env: Path) -> None:
    """The daemon validates insecure enrollment hosts like regular ones."""
    from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

    with pytest.raises(MuseAuthError, match="invalid enrollment host"):
        store_credentials(
            "homeassistant",
            {"kind": "bearer", "token": "x"},
            [],
            insecure_hosts=("bad host!",),
        )
    assert not vault_has_credentials("homeassistant")


def test_failed_enrollment_is_transactional(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """A rejected enrollment leaves the old credential and config intact."""
    from kiss.agents.third_party_agents.homeassistant_agent import HomeAssistantAgent
    from kiss.agents.third_party_agents.ntfy_agent import NtfyAgent

    base_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    ha_agent = HomeAssistantAgent()
    ha_tools = {t.__name__: t for t in ha_agent._get_auth_tools()}
    assert json.loads(ha_tools["authenticate_homeassistant"](base_url, "ha-good"))["ok"] is True

    # The daemon rejects a malformed token (embedded newline) at the
    # SAME valid base URL: the prior credential survives, no plaintext
    # reaches disk, and the config still points at the same origin.
    failed = json.loads(ha_tools["authenticate_homeassistant"](base_url, "ha\nevil"))
    assert failed["ok"] is False
    assert json.loads(ha_config.path.read_text()) == {"base_url": base_url}
    assert json.loads(ha_agent._backend.ha_get_states())["ok"] is True
    assert api_server.header("Authorization") == "Bearer ha-good"

    ntfy_agent_obj = NtfyAgent()
    ntfy_tools = {t.__name__: t for t in ntfy_agent_obj._get_auth_tools()}
    assert json.loads(ntfy_tools["authenticate_ntfy"]("t1", base_url, "tk-good"))["ok"] is True
    failed = json.loads(ntfy_tools["authenticate_ntfy"]("t1", base_url, "tk\nevil"))
    assert failed["ok"] is False
    # No plaintext token reached disk, the old enrollment still works,
    # and the backend was not downgraded to a direct-secret transport.
    assert "token" not in json.loads(ntfy_config.path.read_text())
    assert ntfy_agent_obj._backend._muse
    grant("ntfy", "write", "once")
    assert json.loads(ntfy_agent_obj._backend.publish_notification("still good"))["ok"] is True
    assert api_server.header("Authorization") == "Bearer tk-good"


def test_boundary_ignores_ambient_proxy_env(
    isolated_kiss_home: Path,
    api_server: _DeviceApiServer,
    refusing_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The daemon must not route credentialed requests via HTTP(S)_PROXY.

    The proxy variables point at a port that refuses every connection
    and are set BEFORE the daemon spawns (it inherits this process's
    environment), so a ``trust_env`` regression would fail the request
    instead of reaching the API host directly.
    """
    monkeypatch.setenv("KISS_MUSE_AUTH", "1")
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(var, f"http://127.0.0.1:{refusing_port}")
    server_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    ntfy_config.save({"topic": "t1", "server": server_url, "token": _REAL_NTFY_TOKEN})
    backend = NtfyChannelBackend()
    assert backend.connect() and backend._muse
    polled = json.loads(backend.poll_topic())
    assert polled["ok"] is True
    assert api_server.header("Authorization") == f"Bearer {_REAL_NTFY_TOKEN}"
    stop_daemon()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and socket_path().exists():
        time.sleep(0.05)


def test_rotation_invalidates_old_surrogates(
    muse_env: Path, api_server: _DeviceApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing a credential kills surrogates minted for the old one."""
    from kiss.agents.third_party_agents.discord_agent import DiscordAgent

    monkeypatch.setenv(
        "DISCORD_API_BASE", f"http://127.0.0.1:{api_server.server_address[1]}/api/v10"
    )
    discord_config.save({"bot_token": "tok-old"})
    old_backend = _discord_backend(api_server)
    assert old_backend.connect()

    agent = DiscordAgent()
    tools = {t.__name__: t for t in agent._get_auth_tools()}
    assert json.loads(tools["authenticate_discord"]("tok-new"))["ok"] is True

    # The old backend's surrogate is from the previous generation: it
    # can no longer spend the new credential.
    before = len(api_server.requests)
    stale = json.loads(old_backend.list_guilds())
    assert stale["ok"] is False and "stale" in stale["error"]
    assert len(api_server.requests) == before

    # A fresh connect uses the new token.
    new_backend = _discord_backend(api_server)
    assert new_backend.connect()
    assert api_server.header("Authorization") == "Bot tok-new"


def test_discord_config_token_replaces_stale_vault_entry(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """A token written while Muse was off wins over the vault on re-enable."""
    discord_config.save({"bot_token": "tok-vault"})
    backend = _discord_backend(api_server)
    assert backend.connect()
    assert api_server.header("Authorization") == "Bot tok-vault"

    # Simulate a legacy-mode rotation: a newer plaintext token appears
    # in config.json while the vault still holds the old one.
    discord_config.save({"bot_token": "tok-legacy-rotated"})
    backend2 = _discord_backend(api_server)
    assert backend2.connect()
    assert api_server.header("Authorization") == "Bot tok-legacy-rotated"
    # The newer token was migrated (and scrubbed), not discarded.
    assert not discord_config.path.exists()
    backend3 = _discord_backend(api_server)
    assert backend3.connect()
    assert api_server.header("Authorization") == "Bot tok-legacy-rotated"


def test_ntfy_private_token_never_reaches_public_server(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """A self-hosted enrollment is origin-bound: ntfy.sh is not allowed."""
    server_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    ntfy_config.save({"topic": "t1", "server": server_url, "token": _REAL_NTFY_TOKEN})
    backend = NtfyChannelBackend()
    assert backend.connect() and backend._muse

    # Re-point the (non-secret) metadata at the public server: Sentinel
    # refuses before any network contact because only the enrolled host
    # is in the allowlist.
    saved = json.loads(ntfy_config.path.read_text())
    saved["server"] = "https://ntfy.sh"
    ntfy_config.path.write_text(json.dumps(saved))
    backend2 = NtfyChannelBackend()
    assert backend2.connect() and backend2._muse
    before = len(api_server.requests)
    polled = json.loads(backend2.poll_topic())
    assert polled["ok"] is False and "HTTP 403" in polled["error"]
    assert len(api_server.requests) == before
    audit = (muse_auth_dir() / "audit.jsonl").read_text().splitlines()
    last = json.loads(audit[-1])
    assert last["host"] == "ntfy.sh" and last["verdict"] == "deny"


def test_store_rejects_malformed_credential_values(muse_env: Path) -> None:
    """Control characters or stray whitespace never enter the vault."""
    from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

    for bad in (" leading-space", "trail ", "new\nline", "tab\tchar", ""):
        with pytest.raises(MuseAuthError, match="invalid credential token value"):
            store_credentials("govee", {"kind": "header", "header": "Govee-API-Key",
                                        "token": bad}, [])
        with pytest.raises(MuseAuthError, match="invalid credential token value"):
            store_credentials("ntfy", {"kind": "bearer", "token": bad}, [])
    assert not vault_has_credentials("govee")
    assert not vault_has_credentials("ntfy")
    # A scheme-prefixed value with one internal space stays valid.
    store_credentials(
        "discord",
        {"kind": "header", "header": "Authorization", "token": "Bot tok"},
        [],
    )
    assert vault_has_credentials("discord")


def test_redirect_hops_reauthorized_per_generation(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """Redirect hops re-run Sentinel and re-resolve the credential."""
    server_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    store_credentials(
        "ntfy", {"kind": "bearer", "token": _REAL_NTFY_TOKEN}, [], hosts=("127.0.0.1",)
    )
    handle = mint_surrogate("ntfy")
    assert handle is not None
    session = MuseBoundarySession("ntfy")
    headers = {"Authorization": f"Bearer {handle.token}"}

    # A 302 downgrades the granted POST to a GET, which re-authorizes
    # as a read and carries a freshly resolved real token to the hop.
    grant("ntfy", "write", "once")
    resp = session.post(f"{server_url}/t302", headers=headers, data=b"payload")
    assert resp.status_code == 200
    assert [r["method"] for r in api_server.requests[-2:]] == ["POST", "GET"]
    assert api_server.requests[-1]["path"] == "/t-redirect-target"
    assert api_server.header("Authorization") == f"Bearer {_REAL_NTFY_TOKEN}"

    # A 307 keeps the POST: the same-host hop is re-decided as a write,
    # and with the once-grant already consumed it is refused before the
    # body can leave again.
    grant("ntfy", "write", "once")
    resp = session.post(f"{server_url}/t307", headers=headers, data=b"payload")
    assert resp.status_code == 403
    assert "grant ntfy write" in resp.text
    assert api_server.requests[-1]["path"].endswith("/t307")


def test_ntfy_credential_is_port_origin_bound(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """A token enrolled for one port is refused on another port of the host."""
    server_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    ntfy_config.save({"topic": "t1", "server": server_url, "token": _REAL_NTFY_TOKEN})
    backend = NtfyChannelBackend()
    assert backend.connect() and backend._muse

    # Re-point the metadata at the SAME host on a different port: a
    # different server, so the origin-bound credential is refused with
    # no network contact.
    other_port = api_server.server_address[1] + 1
    saved = json.loads(ntfy_config.path.read_text())
    saved["server"] = f"http://127.0.0.1:{other_port}"
    ntfy_config.path.write_text(json.dumps(saved))
    backend2 = NtfyChannelBackend()
    assert backend2.connect() and backend2._muse
    polled = json.loads(backend2.poll_topic())
    assert polled["ok"] is False and "HTTP 403" in polled["error"]

    # The original port still works.
    backend3 = NtfyChannelBackend()
    saved["server"] = server_url
    ntfy_config.path.write_text(json.dumps(saved))
    assert backend3.connect()
    assert json.loads(backend3.poll_topic())["ok"] is True


def test_rotation_mid_request_aborts_redirect_hop(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """A credential replaced between hops aborts the in-flight request."""
    server_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    store_credentials(
        "ntfy", {"kind": "bearer", "token": _REAL_NTFY_TOKEN}, [], hosts=("127.0.0.1",)
    )
    handle = mint_surrogate("ntfy")
    assert handle is not None
    session = MuseBoundarySession("ntfy")

    def rotate() -> None:
        store_credentials(
            "ntfy", {"kind": "bearer", "token": "rotated-token"}, [], hosts=("127.0.0.1",)
        )

    api_server.rotate_before_redirect = rotate
    grant("ntfy", "write", "session")
    resp = session.post(
        f"{server_url}/t-rotate",
        headers={"Authorization": f"Bearer {handle.token}"},
        data=b"payload",
    )
    # The redirect hop re-resolves the credential pinned to the original
    # generation, sees the rotation, and aborts before contacting the
    # redirect target; neither token reaches the target path.
    assert resp.status_code == 403
    assert "changed mid-request" in resp.text
    assert not [r for r in api_server.requests if r["path"] == "/t-redirect-target"]
    api_server.rotate_before_redirect = None


def test_rotation_mid_request_aborts_header_kind_hop(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """A header-kind credential replaced between hops also aborts."""
    server_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    store_credentials(
        "discord",
        {"kind": "header", "header": "Authorization", "token": "Bot tok-old"},
        [],
        hosts=("127.0.0.1",),
    )
    handle = mint_surrogate("discord")
    assert handle is not None
    session = MuseBoundarySession("discord")

    def rotate() -> None:
        store_credentials(
            "discord",
            {"kind": "header", "header": "Authorization", "token": "Bot tok-new"},
            [],
            hosts=("127.0.0.1",),
        )

    api_server.rotate_before_redirect = rotate
    grant("discord", "write", "session")
    resp = session.post(
        f"{server_url}/t-rotate",
        headers={"Authorization": f"Bearer {handle.token}"},
        data=b"payload",
    )
    assert resp.status_code == 403
    assert "changed mid-request" in resp.text
    assert not [r for r in api_server.requests if r["path"] == "/t-redirect-target"]
    api_server.rotate_before_redirect = None


def test_authenticate_rejects_malformed_ports(muse_env: Path) -> None:
    """A non-numeric or out-of-range port is refused before any state change."""
    from kiss.agents.third_party_agents.firecrawl_agent import FirecrawlAgent
    from kiss.agents.third_party_agents.homeassistant_agent import HomeAssistantAgent
    from kiss.agents.third_party_agents.ntfy_agent import NtfyAgent

    ha_tools = {t.__name__: t for t in HomeAssistantAgent()._get_auth_tools()}
    assert "valid port" in ha_tools["authenticate_homeassistant"](
        "http://localhost:not-a-port", "tok"
    )
    assert not vault_has_credentials("homeassistant")

    ntfy_tools = {t.__name__: t for t in NtfyAgent()._get_auth_tools()}
    assert "valid port" in ntfy_tools["authenticate_ntfy"]("t1", "http://localhost:99999", "tok")
    assert not vault_has_credentials("ntfy")

    # Firecrawl rejects the bad port BEFORE writing any plaintext key or
    # clearing a prior vault entry (no destructive partial migration).
    from kiss.agents.third_party_agents.firecrawl_agent import _config as firecrawl_config
    from kiss.agents.third_party_agents.muse_auth.client import store_credentials

    store_credentials("firecrawl", {"kind": "bearer", "token": "fc-old"}, [])
    fc_tools = {t.__name__: t for t in FirecrawlAgent()._get_auth_tools()}
    assert "valid port" in fc_tools["authenticate_firecrawl"]("fc-new", "http://host:not-a-port")
    assert vault_has_credentials("firecrawl")
    assert not firecrawl_config.path.exists()

    # A malformed api_key value (daemon-rejected) surfaces as a JSON
    # error from the Muse enrollment path.  The non-secret base_url
    # metadata is written first (transactional order) but the plaintext
    # key never reaches disk.
    failed = json.loads(fc_tools["authenticate_firecrawl"]("fc\nnew", "https://fc.example"))
    assert failed["ok"] is False and "invalid credential token value" in failed["error"]
    assert json.loads(firecrawl_config.path.read_text()) == {"base_url": "https://fc.example"}


def test_govee_legacy_mode_unchanged(
    isolated_kiss_home: Path, api_server: _DeviceApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With Muse-auth off, Govee sends the real key directly via urllib."""
    monkeypatch.setenv("KISS_MUSE_AUTH", "0")
    monkeypatch.setattr(
        govee, "API", f"http://127.0.0.1:{api_server.server_address[1]}/router/api/v1"
    )
    monkeypatch.setenv("GOVEE_API_KEY", _REAL_GOVEE_KEY)
    devices = govee.list_devices()
    assert [d["deviceName"] for d in devices] == ["Desk lamp"]
    # Legacy path: the real key travels directly (no surrogate, no vault).
    assert api_server.header("Govee-API-Key") == _REAL_GOVEE_KEY
    assert not (muse_auth_dir() / "vault").exists()


def test_fqdn_policy_and_enrollment_entries_match(muse_env: Path) -> None:
    """Trailing-dot FQDNs match whether they come via policy or enrollment."""
    from kiss.agents.third_party_agents.muse_auth.daemon import (
        MuseAuthDaemon,
        _invalid_hosts_reason,
    )

    # A port-pinned FQDN enrollment host validates and canonicalizes.
    assert _invalid_hosts_reason(["localhost.:8123"]) == ""
    store_credentials(
        "homeassistant",
        {"kind": "bearer", "token": _REAL_HA_TOKEN},
        [],
        hosts=("localhost.:8123",),
    )
    daemon = MuseAuthDaemon()
    assert daemon.sentinel.origin_allowed("homeassistant", "http://localhost.:8123/api/states")
    assert daemon.sentinel.origin_allowed("homeassistant", "http://localhost:8123/api/states")

    # An FQDN-spelled policy extra_hosts entry matches a request host
    # canonicalized the same way (was a round-2 regression).
    policy = muse_auth_dir() / "policy.json"
    policy.write_text(json.dumps({"services": {"govee": {"extra_hosts": ["Example.COM."]}}}))
    assert daemon.sentinel.origin_allowed("govee", "https://example.com/x")
    assert daemon.sentinel.origin_allowed("govee", "https://example.com./x")


def test_govee_daemon_reply_loss_not_replayed(
    muse_env: Path, api_server: _DeviceApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost daemon reply after egress is surfaced, not replayed as a write."""
    from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

    monkeypatch.setattr(
        govee, "API", f"http://127.0.0.1:{api_server.server_address[1]}/router/api/v1"
    )
    monkeypatch.setattr(govee, "_MUSE_SESSION", None)
    monkeypatch.setenv("GOVEE_API_KEY", _REAL_GOVEE_KEY)
    assert govee.list_devices()
    grant("govee", "write", "session")
    # The endpoint records the control POST then the daemon's reply is
    # lost (drop_after_recording kills the connection): a "daemon
    # transport failure" is NOT provably pre-egress, so it must surface.
    api_server.drop_after_recording = {"/router/api/v1/device/control"}
    before = len(api_server.requests)
    with pytest.raises((SystemExit, MuseAuthError)):
        govee.control({"sku": "H6008", "device": "AA:BB"},
                      "devices.capabilities.on_off", "powerSwitch", 1)
    controls = [r for r in api_server.requests[before:]
                if r["path"].endswith("/device/control")]
    assert len(controls) == 1


def test_ntfy_wiring_failure_hides_backend_tools(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """A failed Muse wiring hides the backend tools (no tokenless egress)."""
    from kiss.agents.third_party_agents.ntfy_agent import NtfyAgent

    base_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    ntfy_config.save({"topic": "t1", "server": base_url, "token": "bad\ntoken"})
    agent = NtfyAgent()
    assert agent._backend._muse_error
    # Not authenticated: the backend tools (publish/poll) are not
    # exposed, so nothing can reach the server on the direct transport.
    assert agent._is_authenticated() is False
    tool_names = {t.__name__ for t in agent._get_tools()}
    assert "publish_notification" not in tool_names
    assert "poll_topic" not in tool_names


def test_valid_http_url_rejects_userinfo_and_accepts_ipv4_mapped(muse_env: Path) -> None:
    """URL validation rejects userinfo and accepts IPv4-mapped IPv6 end to end."""
    from kiss.agents.third_party_agents.homeassistant_agent import HomeAssistantAgent
    from kiss.agents.third_party_agents.muse_auth._common import valid_http_url
    from kiss.agents.third_party_agents.muse_auth.daemon import _invalid_hosts_reason

    # Userinfo is refused before any state change: no password reaches
    # config.json and nothing is enrolled.
    assert not valid_http_url("http://alice:secret@127.0.0.1:9")
    tools = {t.__name__: t for t in HomeAssistantAgent()._get_auth_tools()}
    result = tools["authenticate_homeassistant"]("http://alice:secret@127.0.0.1:9", "tok")
    assert "http(s):// URL" in result
    assert not ha_config.path.exists()
    assert not vault_has_credentials("homeassistant")

    # An IPv4-mapped IPv6 literal is valid at BOTH the URL validator and
    # the daemon's enrollment-host parser (no cross-layer mismatch).
    assert valid_http_url("http://[::ffff:127.0.0.1]:8124")
    assert _invalid_hosts_reason(["[::ffff:127.0.0.1]:8124"]) == ""

    # A zone/scoped link-local IPv6 URL is consistent across layers: the
    # validator, the generated enrollment origin, and the daemon parser
    # all accept it, and the request-side origin matches the enrollment.
    from kiss.agents.third_party_agents.muse_auth._common import url_origin_entry
    from kiss.agents.third_party_agents.muse_auth.daemon import MuseAuthDaemon

    # ZoneIDs with RFC-6874 unreserved chars (-, _, ~) are accepted too.
    for scoped in (
        "http://[fe80::1%25eth0]:8123",
        "http://[fe80::1%25eth-0]:8123",
        "http://[fe80::1%25eth_0]:8123",
    ):
        assert valid_http_url(scoped)
        entry = url_origin_entry(scoped)
        assert _invalid_hosts_reason([entry]) == ""
        clear_credentials("homeassistant")
        store_credentials(
            "homeassistant", {"kind": "bearer", "token": _REAL_HA_TOKEN}, [], hosts=(entry,)
        )
        daemon = MuseAuthDaemon()
        assert daemon.sentinel.origin_allowed("homeassistant", f"{scoped}/api/states")

    # Malformed DNS labels (empty label) are rejected at both layers,
    # and a bracketed non-IP is refused as an enrollment host.
    assert not valid_http_url("http://bad..example:8080")
    assert _invalid_hosts_reason(["bad..example:80"]) != ""
    assert _invalid_hosts_reason(["[not-an-ip]:80"]) != ""


def test_firecrawl_is_origin_bound(muse_env: Path) -> None:
    """A self-hosted Firecrawl key is never authorized for the cloud API."""
    from kiss.agents.third_party_agents.firecrawl_agent import _extra_hosts
    from kiss.agents.third_party_agents.muse_auth._common import builtin_hosts
    from kiss.agents.third_party_agents.muse_auth.daemon import MuseAuthDaemon

    # No built-in host: the credential is bound only to its one origin.
    assert builtin_hosts("firecrawl") == ()
    assert _extra_hosts("https://firecrawl.private.example:7443") == (
        "firecrawl.private.example:7443",
    )
    assert _extra_hosts("https://api.firecrawl.dev") == ("api.firecrawl.dev:443",)

    store_credentials(
        "firecrawl",
        {"kind": "bearer", "token": "selfhost-key"},
        [],
        hosts=("firecrawl.private.example:7443",),
    )
    daemon = MuseAuthDaemon()
    assert daemon.sentinel.origin_allowed(
        "firecrawl", "https://firecrawl.private.example:7443/v2/scrape"
    )
    # The self-host key can NOT be spent against the public cloud API.
    assert not daemon.sentinel.origin_allowed(
        "firecrawl", "https://api.firecrawl.dev/v2/scrape"
    )


def test_ntfy_explicit_tokenless_overrides_stale_vault(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """A deliberate tokenless rotation is not revived from the vault."""
    server_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    # Enroll a token, then simulate a legacy-mode tokenless rotation:
    # config.json carries an explicit empty token while the vault still
    # holds the old one.
    ntfy_config.save({"topic": "t1", "server": server_url, "token": _REAL_NTFY_TOKEN})
    assert NtfyChannelBackend().connect()
    assert vault_has_credentials("ntfy")
    ntfy_config.path.write_text(
        json.dumps({"topic": "t1", "server": server_url, "token": ""})
    )

    backend = NtfyChannelBackend()
    assert backend.connect()
    # The explicit removal wins: the backend is tokenless/legacy, the
    # stale vault entry is gone, and no revived token is sent.
    assert not backend._muse
    assert not vault_has_credentials("ntfy")
    assert json.loads(backend.poll_topic())["ok"] is True
    assert api_server.header("Authorization") == ""


def test_leading_zero_and_fqdn_port_entries_match(muse_env: Path) -> None:
    """Leading-zero and FQDN port-pinned entries match the request origin."""
    from kiss.agents.third_party_agents.muse_auth.daemon import (
        MuseAuthDaemon,
        _invalid_hosts_reason,
    )

    assert _invalid_hosts_reason(["h:00080"]) == ""
    store_credentials(
        "homeassistant",
        {"kind": "bearer", "token": _REAL_HA_TOKEN},
        [],
        hosts=("127.0.0.1:00080",),
    )
    daemon = MuseAuthDaemon()
    # The leading-zero enrollment entry matches a request whose parsed
    # port is 80.
    assert daemon.sentinel.origin_allowed("homeassistant", "http://127.0.0.1/api/states")

    # A port-pinned FQDN policy entry matches its canonical request form.
    policy = muse_auth_dir() / "policy.json"
    policy.write_text(
        json.dumps({"services": {"govee": {"extra_hosts": ["Example.COM.:443"]}}})
    )
    assert daemon.sentinel.origin_allowed("govee", "https://example.com/x")
    assert daemon.sentinel.origin_allowed("govee", "https://example.com.:443/x")


def test_ipv6_origin_helpers(muse_env: Path) -> None:
    """Origin helpers bracket IPv6 literals and validate host:port entries."""
    from kiss.agents.third_party_agents.muse_auth._common import (
        host_port_entry,
        url_origin_entry,
        valid_hostname,
        valid_http_url,
    )
    from kiss.agents.third_party_agents.muse_auth.daemon import _invalid_hosts_reason

    assert host_port_entry("::1", 8080) == "[::1]:8080"
    assert host_port_entry("127.0.0.1", 80) == "127.0.0.1:80"
    assert url_origin_entry("http://[::1]:8123") == "[::1]:8123"
    assert url_origin_entry("https://[2001:db8::1]") == "[2001:db8::1]:443"
    # An IPv6 literal is a valid hostname/URL, and a malformed port is
    # treated as absent by url_origin_entry (validation happens up front).
    assert valid_hostname("::1") and valid_http_url("http://[::1]:8080")
    assert not valid_http_url("http://[::1]:notaport")
    assert url_origin_entry("http://h:99999") == "h:80"
    # The daemon accepts both bare and port-pinned IPv6 enrollment hosts.
    assert _invalid_hosts_reason(["[::1]:8123"]) == ""
    assert _invalid_hosts_reason(["::1"]) == ""
    assert _invalid_hosts_reason(["[::1]:99999"]) != ""
    # A port-pinned enrollment is honored end-to-end for a loopback
    # IPv6 Home Assistant origin.
    store_credentials(
        "homeassistant",
        {"kind": "bearer", "token": _REAL_HA_TOKEN},
        [],
        hosts=("[::1]:8123",),
    )
    handle = mint_surrogate("homeassistant")
    assert handle is not None
    from kiss.agents.third_party_agents.muse_auth.daemon import MuseAuthDaemon

    daemon = MuseAuthDaemon()
    assert daemon.sentinel.origin_allowed("homeassistant", "http://[::1]:8123/api/states")
    assert not daemon.sentinel.origin_allowed("homeassistant", "http://[::1]:9999/api/states")


def test_govee_does_not_replay_writes_on_ambiguous_failure(
    muse_env: Path, api_server: _DeviceApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ambiguous post-egress failure surfaces instead of replaying."""
    from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

    monkeypatch.setattr(
        govee, "API", f"http://127.0.0.1:{api_server.server_address[1]}/router/api/v1"
    )
    monkeypatch.setattr(govee, "_MUSE_SESSION", None)
    monkeypatch.setenv("GOVEE_API_KEY", _REAL_GOVEE_KEY)
    # Enroll first (a clean read), then make control fail post-egress.
    assert govee.list_devices()
    grant("govee", "write", "session")
    api_server.drop_after_recording = {"/router/api/v1/device/control"}
    before = len(api_server.requests)
    with pytest.raises((SystemExit, MuseAuthError)):
        govee.control({"sku": "H6008", "device": "AA:BB"},
                      "devices.capabilities.on_off", "powerSwitch", 1)
    # Exactly one control POST reached the endpoint: no replay.
    controls = [r for r in api_server.requests[before:]
                if r["path"].endswith("/device/control")]
    assert len(controls) == 1


def test_authenticate_accepts_terminal_dot_host(
    muse_env: Path, api_server: _DeviceApiServer
) -> None:
    """A fully-qualified hostname with a trailing dot is canonicalized."""
    from kiss.agents.third_party_agents.homeassistant_agent import HomeAssistantAgent

    # 127.0.0.1. is the loopback address written fully-qualified; it
    # must be accepted (canonicalized to 127.0.0.1), not rejected.
    port = api_server.server_address[1]
    agent = HomeAssistantAgent()
    tools = {t.__name__: t for t in agent._get_auth_tools()}
    result = json.loads(
        tools["authenticate_homeassistant"](f"http://127.0.0.1.:{port}", "ha-fqdn")
    )
    assert result["ok"] is True
    assert vault_has_credentials("homeassistant")


def test_wiring_failure_leaves_agents_constructible(
    muse_env: Path, api_server: _DeviceApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daemon-rejected config token fails closed but keeps tools alive."""
    from kiss.agents.third_party_agents.discord_agent import DiscordAgent
    from kiss.agents.third_party_agents.homeassistant_agent import HomeAssistantAgent
    from kiss.agents.third_party_agents.ntfy_agent import NtfyAgent

    base_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    monkeypatch.setenv("DISCORD_API_BASE", f"{base_url}/api/v10")
    # A token the daemon rejects at enrollment (embedded newline).
    discord_config.save({"bot_token": "bad\ntoken"})
    agent = DiscordAgent()
    assert agent._backend._bot_token == ""
    assert "Muse-auth wiring failed" in agent._backend._connection_info
    assert not vault_has_credentials("discord")

    ha_config.save({"base_url": base_url, "token": "bad\ntoken"})
    ha_agent = HomeAssistantAgent()
    assert ha_agent._backend._token == ""
    assert "Muse-auth wiring failed" in ha_agent._backend._connection_info

    ntfy_config.save({"topic": "t1", "server": base_url, "token": "bad\ntoken"})
    ntfy_agent_obj = NtfyAgent()
    assert ntfy_agent_obj._backend._token == ""
    assert not ntfy_agent_obj._backend._muse
    assert "wiring failed" in ntfy_agent_obj._backend._connection_info
    # The check tool is honest about the broken credential rather than
    # reporting the topic as authenticated, and connect() fails closed.
    ntfy_tools = {t.__name__: t for t in ntfy_agent_obj._get_auth_tools()}
    checked = json.loads(ntfy_tools["check_ntfy_auth"]())
    assert checked["ok"] is False and "wiring failed" in checked["error"]
    assert not NtfyChannelBackend().connect()


def test_authenticate_rejects_malformed_urls(muse_env: Path) -> None:
    """Base URLs without an http(s) scheme or hostname are refused early."""
    from kiss.agents.third_party_agents.homeassistant_agent import HomeAssistantAgent
    from kiss.agents.third_party_agents.ntfy_agent import NtfyAgent

    ha_tools = {t.__name__: t for t in HomeAssistantAgent()._get_auth_tools()}
    result = ha_tools["authenticate_homeassistant"]("homeassistant.local:8123", "tok")
    assert "http(s):// URL with a hostname" in result
    assert not vault_has_credentials("homeassistant")

    ntfy_tools = {t.__name__: t for t in NtfyAgent()._get_auth_tools()}
    result = ntfy_tools["authenticate_ntfy"]("t1", "ntfy.local:8080", "tok")
    assert "http(s):// URL with a hostname" in result
    assert not vault_has_credentials("ntfy")


def test_govee_boundary_transport_failure_raises(
    muse_env: Path, refusing_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistent transport failure surfaces after the single retry."""
    from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

    monkeypatch.setattr(govee, "API", f"http://127.0.0.1:{refusing_port}/router/api/v1")
    monkeypatch.setattr(govee, "_MUSE_SESSION", None)
    monkeypatch.setenv("GOVEE_API_KEY", _REAL_GOVEE_KEY)
    with pytest.raises(MuseAuthError, match="network boundary request failed"):
        govee.list_devices()


def test_network_failure_error_is_redacted(muse_env: Path, refusing_port: int) -> None:
    """Transport errors cross the boundary without the real credential."""
    store_credentials(
        "homeassistant",
        {"kind": "bearer", "token": _REAL_HA_TOKEN},
        [],
        hosts=("127.0.0.1",),
    )
    handle = mint_surrogate("homeassistant")
    assert handle is not None
    session = MuseBoundarySession("homeassistant")
    from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

    with pytest.raises(MuseAuthError, match="network boundary request failed") as info:
        session.get(
            f"http://127.0.0.1:{refusing_port}/api/states",
            headers={"Authorization": f"Bearer {handle.token}"},
        )
    assert _REAL_HA_TOKEN not in str(info.value)


def test_corrupted_vault_entry_fails_closed(muse_env: Path, api_server: _DeviceApiServer) -> None:
    """A corrupted vault file yields no enrolled hosts: requests are denied."""
    server_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    ntfy_config.save({"topic": "t1", "server": server_url, "token": _REAL_NTFY_TOKEN})
    backend = NtfyChannelBackend()
    assert backend.connect() and backend._muse

    # Corrupt the daemon-side vault entry while its surrogate is live.
    vault_file = muse_auth_dir() / "vault" / "ntfy.json"
    vault_file.write_text("{not json")
    polled = json.loads(backend.poll_topic())
    assert polled["ok"] is False
    assert "HTTP 403" in polled["error"]
    assert not [r for r in api_server.requests if r["path"].startswith("/t1/json")]
    # The Sentinel recorded the denial: the enrolled host list was gone.
    audit = (muse_auth_dir() / "audit.jsonl").read_text().splitlines()
    last = json.loads(audit[-1])
    assert last["verdict"] == "deny" and "allowlist" in last["reason"]


def test_multiple_trailing_dot_hosts_rejected(
    muse_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hosts with more than one terminal dot are malformed at every layer.

    ``example.com.`` is the fully-qualified spelling of ``example.com``,
    but a second terminal dot is an empty DNS label: canonicalization
    must strip exactly one dot so the malformed spelling cannot collapse
    into a valid name and reach urllib3 (where it fails unparseably).
    """
    from kiss.agents.third_party_agents.muse_auth._common import (
        canonical_host,
        canonical_host_entry,
        valid_hostname,
        valid_http_url,
    )
    from kiss.agents.third_party_agents.muse_auth.daemon import _invalid_hosts_reason

    assert canonical_host("Example.COM.") == "example.com"
    assert canonical_host("example.com..") == "example.com."
    assert canonical_host_entry("localhost.:8123") == "localhost:8123"
    assert valid_hostname("localhost.") and valid_hostname("127.0.0.1.")
    for bad in ("localhost..", "example.com..", "127.0.0.1.."):
        assert not valid_hostname(bad)
        assert not valid_http_url(f"http://{bad}:8123")
    assert _invalid_hosts_reason(["localhost..:8123"]) != ""
    assert _invalid_hosts_reason(["localhost.:8123"]) == ""

    # The CLI import refuses the malformed URL outright: nothing is
    # stored and the plaintext config is left for the user to fix.
    ha_config.path.parent.mkdir(parents=True, exist_ok=True)
    ha_config.path.write_text(
        json.dumps({"base_url": "http://localhost..:8123", "token": _REAL_HA_TOKEN})
    )
    assert muse_cli.main(["import", "homeassistant"]) == 1
    assert "not a valid http(s):// URL" in capsys.readouterr().err
    assert not vault_has_credentials("homeassistant")
    assert json.loads(ha_config.path.read_text())["token"] == _REAL_HA_TOKEN


def test_cli_import_url_requirements(
    muse_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Null/absent URLs default only where a default exists; falsy junk fails.

    A null/absent server for the optional ntfy URL selects the public
    ntfy.sh default (and must enroll its origin, or the migrated token
    is unusable); Home Assistant's base_url is required; non-string
    falsy JSON values (false, 0, []) are malformed configs, not "use
    the default".
    """
    from kiss.agents.third_party_agents.firecrawl_agent import _config as firecrawl_config
    from kiss.agents.third_party_agents.muse_auth.daemon import MuseAuthDaemon

    # ntfy with a null server: import succeeds and enrolls ntfy.sh:443,
    # the same origin the connector loader defaults to.
    assert muse_cli._import_hosts("ntfy", {}) == (("ntfy.sh:443",), ())
    assert muse_cli._import_hosts("ntfy", {"server": None}) == (("ntfy.sh:443",), ())
    ntfy_config.path.parent.mkdir(parents=True, exist_ok=True)
    ntfy_config.path.write_text(
        json.dumps({"topic": "t1", "server": None, "token": _REAL_NTFY_TOKEN})
    )
    assert muse_cli.main(["import", "ntfy"]) == 0
    assert vault_has_credentials("ntfy")
    daemon = MuseAuthDaemon()
    assert daemon.sentinel.origin_allowed("ntfy", "https://ntfy.sh/t1/json")
    # The scrubbed config (null server dropped) still wires in Muse
    # mode against the defaulted server.
    backend = NtfyChannelBackend()
    assert backend.connect() and backend._muse
    assert backend._server == "https://ntfy.sh"
    clear_credentials("ntfy")

    # Home Assistant requires base_url: null and absent are rejected
    # before any credential state changes.
    for cfg in ({"base_url": None, "token": _REAL_HA_TOKEN}, {"token": _REAL_HA_TOKEN}):
        ha_config.path.parent.mkdir(parents=True, exist_ok=True)
        ha_config.path.write_text(json.dumps(cfg))
        assert muse_cli.main(["import", "homeassistant"]) == 1
        assert "required" in capsys.readouterr().err
        assert not vault_has_credentials("homeassistant")
        assert json.loads(ha_config.path.read_text())["token"] == _REAL_HA_TOKEN

    # Non-string falsy values are rejected, not silently defaulted.
    falsy: Any
    for falsy in (False, 0, []):
        firecrawl_config.path.parent.mkdir(parents=True, exist_ok=True)
        firecrawl_config.path.write_text(
            json.dumps({"api_key": "fc-secret", "base_url": falsy})
        )
        assert muse_cli.main(["import", "firecrawl"]) == 1
        assert "not a valid http(s):// URL" in capsys.readouterr().err
        assert not vault_has_credentials("firecrawl")
        assert json.loads(firecrawl_config.path.read_text())["api_key"] == "fc-secret"


def test_connect_rejects_malformed_legacy_urls(muse_env: Path) -> None:
    """Muse-mode auto-migration validates the base URL before any change.

    A malformed legacy URL must not enroll the credential into a host
    scope Sentinel can never match, and must not scrub the plaintext
    copy: the backend fails closed with an actionable message instead.
    """
    from kiss.agents.third_party_agents.firecrawl_agent import FirecrawlChannelBackend
    from kiss.agents.third_party_agents.firecrawl_agent import _config as firecrawl_config

    ha_config.path.parent.mkdir(parents=True, exist_ok=True)
    ha_config.path.write_text(
        json.dumps({"base_url": "http://localhost..:8123", "token": _REAL_HA_TOKEN})
    )
    ha_backend = HomeAssistantChannelBackend()
    assert not ha_backend.connect()
    assert "not a valid http(s):// URL" in ha_backend._connection_info
    assert not vault_has_credentials("homeassistant")
    assert json.loads(ha_config.path.read_text())["token"] == _REAL_HA_TOKEN

    firecrawl_config.path.parent.mkdir(parents=True, exist_ok=True)
    firecrawl_config.path.write_text(
        json.dumps({"api_key": "fc-secret", "base_url": "http://bad..example"})
    )
    fc_backend = FirecrawlChannelBackend()
    assert not fc_backend.connect()
    assert "not a valid http(s):// URL" in fc_backend._connection_info
    assert not vault_has_credentials("firecrawl")
    assert json.loads(firecrawl_config.path.read_text())["api_key"] == "fc-secret"

    ntfy_config.path.parent.mkdir(parents=True, exist_ok=True)
    ntfy_config.path.write_text(
        json.dumps(
            {"topic": "t1", "server": "http://bad..example:8080", "token": _REAL_NTFY_TOKEN}
        )
    )
    ntfy_backend = NtfyChannelBackend()
    assert not ntfy_backend.connect()
    assert "not a valid http(s):// URL" in ntfy_backend._muse_error
    assert not vault_has_credentials("ntfy")
    assert json.loads(ntfy_config.path.read_text())["token"] == _REAL_NTFY_TOKEN
