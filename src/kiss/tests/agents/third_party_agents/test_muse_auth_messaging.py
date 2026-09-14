# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Muse-auth on the remaining messaging connectors.

SEA style, mirroring ``test_muse_auth_devices.py``: a REAL Muse-auth
daemon subprocess plus a REAL local HTTP server (stdlib
``ThreadedHTTPServer``) emulating the Mattermost / Twitch / Zalo /
LINE / Nextcloud Talk / BlueBubbles / Synology Chat APIs — no mocks,
patches, or fakes.  The emulated API records every request arriving at
the "network" so tests can prove the boundary swap for each credential
placement: bearer (Mattermost, Twitch, LINE), custom header (Zalo's
``access_token``), Basic (Nextcloud), and the NEW query-kind
credentials (BlueBubbles' ``password=`` and Synology's webhook
``token=`` URL parameters), while the agent side only ever holds
``muse-sgt.*`` surrogates.

Branch-coverage notes (unreachable without test doubles, so documented
instead of mocked):

* The connectors' ``_muse_authenticate`` ``except Exception`` rollback
  paths that are not driven by an invalid-credential response need the
  daemon to die between the enrollment and the validation call (a
  cross-process race); the invalid-credential rollback IS covered.
* The ``# pragma: no cover`` defense-in-depth branches after
  ``store_credentials`` + ``_wire_muse`` (wire failing although the
  credential was just stored) need the vault cleared between the store
  and the mint by another process.
* ``BlueBubblesChannelBackend.connect``'s Muse branch and the
  BlueBubbles tool methods are macOS-gated (``sys.platform``); on this
  Linux CI they are unreachable without faking the platform, so the
  Muse plumbing is exercised through ``_wire_muse`` + the ungated
  ``poll_messages``/``send_message`` paths instead.  This includes
  ``get_server_info``'s strict HTTP-status + ``status == 200`` envelope
  check (an HTTP 401 body no longer reads as success), which sits
  behind the same gate.
* ``_SdkLineApi``'s method bodies require the ``linebot`` SDK, which is
  not installed in this environment; its constructor's ImportError path
  (legacy mode) IS covered.
* ``_MuseMattermostDriver.get_users``/``get_channel``/
  ``create_direct_message_channel``/``get_channels_for_user`` are thin
  ``_call`` wrappers exercised via the same boundary path as the
  covered methods.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server
from kiss.agents.third_party_agents.bluebubbles_agent import BlueBubblesChannelBackend
from kiss.agents.third_party_agents.bluebubbles_agent import _config as bb_config
from kiss.agents.third_party_agents.line_agent import LineChannelBackend
from kiss.agents.third_party_agents.line_agent import _config as line_config
from kiss.agents.third_party_agents.mattermost_agent import MattermostChannelBackend
from kiss.agents.third_party_agents.mattermost_agent import _config as mm_config
from kiss.agents.third_party_agents.muse_auth import __main__ as muse_cli
from kiss.agents.third_party_agents.muse_auth._common import (
    PROTOCOL_VERSION,
    muse_auth_dir,
    socket_path,
)
from kiss.agents.third_party_agents.muse_auth.client import (
    MuseAuthError,
    MuseBoundarySession,
    _daemon_protocol,
    clear_credentials,
    ensure_daemon,
    grant,
    mint_surrogate,
    stop_daemon,
    store_credentials,
    vault_has_credentials,
)
from kiss.agents.third_party_agents.nextcloud_talk_agent import NextcloudTalkChannelBackend
from kiss.agents.third_party_agents.nextcloud_talk_agent import _config as nc_config
from kiss.agents.third_party_agents.synology_chat_agent import SynologyChatChannelBackend
from kiss.agents.third_party_agents.synology_chat_agent import _config as syno_config
from kiss.agents.third_party_agents.twitch_agent import TwitchChannelBackend
from kiss.agents.third_party_agents.twitch_agent import _config as twitch_config
from kiss.agents.third_party_agents.zalo_agent import ZaloChannelBackend
from kiss.agents.third_party_agents.zalo_agent import _config as zalo_config

_REAL_MM_TOKEN = "mm-real-secret"
_REAL_TWITCH_TOKEN = "twitch-real-secret"
_REAL_ZALO_TOKEN = "zalo-real-secret"
_REAL_LINE_TOKEN = "line-real-secret"
_REAL_NC_PASSWORD = "nc-real-secret"
_REAL_BB_PASSWORD = "bb-real-secret"
_REAL_SYNO_TOKEN = "syno-real-secret"


class _MessagingApiHandler(BaseHTTPRequestHandler):
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
        path = self.path.split("?", 1)[0]
        if path == "/redirect-offsite":
            # Cross-origin redirect target configured by the test.
            self.send_response(302)
            self.send_header("Location", self.server.offsite_location)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/chain-middle":
            # Same-origin second hop used to prove a foreign allowlisted
            # origin cannot regain a query credential via A -> B -> B.
            self.send_response(302)
            self.send_header("Location", "/api/v1/server/info")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/redirect-onsite":
            # Same-origin redirect: the boundary must re-authorize the
            # hop and re-inject the (query-kind) credential.
            self.send_response(302)
            self.send_header("Location", "/api/v1/server/info")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        auth = next((v for k, v in self.headers.items() if k.lower() == "authorization"), "")
        zalo_token = next(
            (v for k, v in self.headers.items() if k.lower() == "access_token"), ""
        )
        credential = auth or zalo_token
        if credential.startswith("Basic "):
            credential = base64.b64decode(credential[6:]).decode("utf-8", errors="replace")
        status = 200
        if credential.endswith("-nonjson"):
            # A proxy-style failure: non-JSON body with a 5xx status
            # (drives the strict validators' JSON-decode fallbacks).
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "11")
            self.end_headers()
            self.wfile.write(b"bad gateway")
            return
        if credential.endswith("-invalid"):
            status = 401
            if "/ocs/" in path:
                # Nextcloud wraps even auth failures in an "ocs" envelope
                # (envelope presence must not read as success).
                payload = json.dumps(
                    {"ocs": {"meta": {"status": "failure", "statuscode": 401}, "data": []}}
                ).encode()
            else:
                payload = json.dumps({"message": "invalid credentials"}).encode()
        elif path.endswith("/api/v4/users/me"):
            payload = json.dumps({"id": "U1", "username": "mmbot"}).encode()
        elif path.endswith("/api/v4/teams"):
            payload = json.dumps([{"id": "T1", "name": "team", "display_name": "Team"}]).encode()
        elif path.endswith("/api/v4/channels/C1/posts"):
            payload = json.dumps(
                {
                    "order": ["P1"],
                    "posts": {
                        "P1": {
                            "id": "P1",
                            "message": "hello",
                            "user_id": "U2",
                            "create_at": 1700000000000,
                        }
                    },
                }
            ).encode()
        elif path.endswith("/api/v4/posts") and self.command == "POST":
            payload = json.dumps({"id": "P2"}).encode()
        elif path.endswith("/api/v4/users/me/typing"):
            payload = b"{}"
        elif path.endswith("/helix/users"):
            payload = json.dumps({"data": [{"id": "T1", "login": "kisscaster"}]}).encode()
        elif path.endswith("/helix/streams"):
            payload = json.dumps({"data": []}).encode()
        elif path.endswith("/helix/chat/messages"):
            payload = json.dumps({"data": [{"message_id": "tm1"}]}).encode()
        elif path.endswith("/getoa"):
            payload = json.dumps(
                {"error": 0, "data": {"name": "kiss-oa", "oa_id": "OA1"}}
            ).encode()
        elif path.endswith("/message/text") and "/v2.0/oa" in path:
            payload = json.dumps({"error": 0, "data": {"message_id": "Z1"}}).encode()
        elif path.endswith("/upload/image"):
            payload = json.dumps({"error": 0, "data": {"attachment_id": "ATT1"}}).encode()
        elif "/v2.0/oa/" in path:
            # Generic Zalo OA endpoints (profile, followers, chats, ...).
            payload = json.dumps({"error": 0, "data": {}}).encode()
        elif path.endswith("/v2/bot/message/quota"):
            payload = json.dumps({"type": "limited", "value": 500}).encode()
        elif "/v2/bot/profile/" in path:
            payload = json.dumps(
                {
                    "displayName": "Ann",
                    "userId": "U1",
                    "pictureUrl": "https://line.example/p.png",
                    "statusMessage": "hi",
                }
            ).encode()
        elif path.startswith("/v2/bot/"):
            payload = b"{}"
        elif path.endswith("/spreed/api/v4/room") and self.command == "GET":
            payload = json.dumps(
                {
                    "ocs": {
                        "meta": {"statuscode": 200},
                        "data": [
                            {
                                "token": "r1",
                                "displayName": "Room",
                                "type": 3,
                                "participantCount": 2,
                            }
                        ],
                    }
                }
            ).encode()
        elif path.endswith("/participants/active"):
            payload = json.dumps({"ocs": {"meta": {"statuscode": 200}, "data": {}}}).encode()
        elif "/spreed/api/v4/chat/" in path and self.command == "POST":
            payload = json.dumps(
                {"ocs": {"meta": {"statuscode": 201}, "data": {"id": 42}}}
            ).encode()
        elif "/spreed/api/v4/chat/" in path:
            payload = json.dumps(
                {
                    "ocs": {
                        "meta": {"statuscode": 200},
                        "data": [
                            {
                                "id": 7,
                                "actorId": "ann",
                                "message": "hello",
                                "timestamp": 1700000000,
                            }
                        ],
                    }
                }
            ).encode()
        elif path.endswith("/api/v1/server/info"):
            payload = json.dumps({"status": 200, "data": {"os_version": "14.0"}}).encode()
        elif path.endswith("/api/v1/message/query"):
            payload = json.dumps(
                {
                    "status": 200,
                    "data": [
                        {
                            "guid": "g1",
                            "text": "hi",
                            "dateCreated": 1700000000000,
                            "isFromMe": False,
                            "chats": [{"guid": "chat1"}],
                            "sender": {"address": "+15550000000"},
                        }
                    ],
                }
            ).encode()
        elif path.endswith("/api/v1/message/text"):
            payload = json.dumps({"status": 200, "data": {}}).encode()
        elif path.endswith("/webapi/entry.cgi"):
            query = parse_qs(urlsplit(self.path).query)
            if query.get("token", [""])[0] == _REAL_SYNO_TOKEN:
                payload = json.dumps({"success": True}).encode()
            else:
                payload = json.dumps({"success": False, "error": {"code": 404}}).encode()
        else:
            payload = json.dumps({"ok": True}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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

    def do_PUT(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a PUT request."""
        self._serve()

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence request logging."""


class _MessagingApiServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records requests for verification."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _MessagingApiHandler)
        self.requests: list[dict[str, Any]] = []
        # Redirect target served by /redirect-offsite.
        self.offsite_location: str = ""

    @property
    def port(self) -> int:
        """Return the bound TCP port."""
        return int(self.server_address[1])

    def base(self, suffix: str = "") -> str:
        """Return the server's loopback base URL plus *suffix*."""
        return f"http://127.0.0.1:{self.port}{suffix}"

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

    def query(self, index: int = -1) -> dict[str, list[str]]:
        """Return a recorded request's parsed query parameters.

        Args:
            index: Which recorded request to inspect (default: last).

        Returns:
            ``parse_qs`` of the recorded URL's query string.
        """
        return parse_qs(urlsplit(self.requests[index]["path"]).query)


@pytest.fixture()
def api_server() -> Any:
    """Run the emulated messaging REST API on a loopback port."""
    server = _MessagingApiServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    stop_http_server(server, thread)


@pytest.fixture()
def rogue_server() -> Any:
    """Run a second, off-allowlist emulator for redirect tests."""
    server = _MessagingApiServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    stop_http_server(server, thread)


@pytest.fixture()
def muse_env(isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Enable Muse-auth inside an isolated ``KISS_HOME`` with a live daemon.

    Twitch, Zalo, and LINE get a loopback ``extra_hosts`` policy entry
    (their real hosts are fixed cloud endpoints); Mattermost, Nextcloud,
    BlueBubbles, and Synology deliberately get none, so their tests
    prove the enrollment-time origin binding.
    """
    monkeypatch.setenv("KISS_MUSE_AUTH", "1")
    directory = muse_auth_dir()
    directory.mkdir(parents=True, exist_ok=True)
    policy = {
        "defaults": {"read": "allow", "write": "ask"},
        "services": {
            "twitch": {"extra_hosts": ["127.0.0.1"]},
            "zalo": {"extra_hosts": ["127.0.0.1"]},
            "line": {"extra_hosts": ["127.0.0.1"]},
        },
    }
    (directory / "policy.json").write_text(json.dumps(policy))
    yield isolated_kiss_home
    stop_daemon()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and socket_path().exists():
        time.sleep(0.05)


def _auth_tools(agent: Any) -> dict[str, Any]:
    """Return an agent's auth tools keyed by function name."""
    return {tool.__name__: tool for tool in agent._get_auth_tools()}


# ---------------------------------------------------------------- Mattermost


def _mm_config(api_server: _MessagingApiServer, token: str = _REAL_MM_TOKEN) -> None:
    """Save a legacy Mattermost config pointing at the emulator."""
    mm_config.save(
        {"url": "127.0.0.1", "token": token, "port": str(api_server.port), "scheme": "http"}
    )


def test_mattermost_muse_driver_swap_and_scrub(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """The Muse driver shim spends the real PAT only at the boundary."""
    _mm_config(api_server)
    backend = MattermostChannelBackend()
    assert backend.connect() is True
    assert backend._muse is True
    # The emulated server saw the REAL bearer; the agent holds a surrogate.
    assert api_server.header("Authorization") == f"Bearer {_REAL_MM_TOKEN}"
    assert backend._token.startswith("muse-sgt.mattermost.")
    assert "Authenticated as mmbot" in backend._connection_info
    # The plaintext token is scrubbed; non-secret metadata survives.
    stored = json.loads(mm_config.path.read_text())
    assert "token" not in stored
    assert stored["url"] == "127.0.0.1"
    assert stored["scheme"] == "http"
    assert vault_has_credentials("mattermost")
    # Reads flow without grants: polling and every GET namespace method.
    messages, cursor = backend.poll_messages("C1", "")
    assert [m["text"] for m in messages] == ["hello"]
    assert cursor == "1700000000000"
    assert json.loads(backend.list_teams())["ok"] is True
    driver = backend._driver
    assert driver.get_users(params={"page": 0}) == {"ok": True}
    assert driver.get_channel("C9") == {"ok": True}
    assert driver.get_channels_for_user("me", "T1") == {"ok": True}
    # Creating a DM channel is a write.
    grant("mattermost", "write", "once")
    assert driver.create_direct_message_channel(["U1", "U2"]) == {"ok": True}


def test_mattermost_write_needs_grant_and_typing_is_read(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """create_post asks for a write grant; the typing POST stays a read."""
    _mm_config(api_server)
    backend = MattermostChannelBackend()
    assert backend.connect() is True
    api_server.requests.clear()
    # The ephemeral typing indicator must not need (or burn) a grant.
    backend.send_typing("C1", thread_ts="P1")
    assert api_server.requests[-1]["path"].endswith("/users/me/typing")
    assert api_server.header("Authorization") == f"Bearer {_REAL_MM_TOKEN}"
    # A post is a write: ask -> grant once -> allowed.
    denied = json.loads(backend.create_post("C1", "hi"))
    assert denied["ok"] is False
    assert "requires user approval" in denied["error"]
    grant("mattermost", "write", "once")
    assert json.loads(backend.create_post("C1", "hi"))["ok"] is True
    assert api_server.requests[-1]["path"].endswith("/api/v4/posts")
    # The one-shot grant is spent.
    assert json.loads(backend.delete_post("P1"))["ok"] is False


def test_mattermost_muse_authenticate_rotation_rollback_clear(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """The authenticate tool enrolls/rotates/rolls back; clear empties the vault."""
    from kiss.agents.third_party_agents.mattermost_agent import MattermostAgent

    agent = MattermostAgent()
    tools = _auth_tools(agent)
    result = json.loads(
        tools["authenticate_mattermost"](
            "127.0.0.1", _REAL_MM_TOKEN, port=api_server.port, scheme="http"
        )
    )
    assert result["ok"] is True
    assert result["username"] == "mmbot"
    assert vault_has_credentials("mattermost")
    # config.json never saw the token.
    stored = json.loads(mm_config.path.read_text())
    assert "token" not in stored
    assert json.loads(tools["check_mattermost_auth"]())["ok"] is True
    # A bad rotation rolls the vault back to empty.
    result = json.loads(
        tools["authenticate_mattermost"](
            "127.0.0.1", "mm-rotated-invalid", port=api_server.port, scheme="http"
        )
    )
    assert result["ok"] is False
    assert not vault_has_credentials("mattermost")
    # A malformed server URL is rejected before any state change.
    result = json.loads(
        tools["authenticate_mattermost"](
            "bad..host", _REAL_MM_TOKEN, port=api_server.port, scheme="http"
        )
    )
    assert result["ok"] is False
    assert "not a valid" in result["error"]
    # Re-enroll, then clear.
    assert json.loads(
        tools["authenticate_mattermost"](
            "127.0.0.1", _REAL_MM_TOKEN, port=api_server.port, scheme="http"
        )
    )["ok"] is True
    assert "cleared" in tools["clear_mattermost_auth"]()
    assert not vault_has_credentials("mattermost")
    assert agent._backend._muse is False


def test_mattermost_wire_failure_paths(muse_env: Path) -> None:
    """Missing config, malformed URL, and empty vault each fail closed."""
    backend = MattermostChannelBackend()
    assert backend._wire_muse() is False
    assert "No Mattermost config found" in backend._connection_info
    mm_config.save({"url": "bad..host", "token": "t", "port": "443", "scheme": "https"})
    assert backend._wire_muse() is False
    assert "not a valid" in backend._connection_info
    # The malformed config was NOT scrubbed (fail closed).
    assert json.loads(mm_config.path.read_text())["token"] == "t"
    mm_config.clear()
    mm_config.save({"url": "127.0.0.1", "port": "443", "scheme": "https"})
    assert backend._wire_muse() is False
    assert "No Mattermost credential" in backend._connection_info


# ------------------------------------------------------------------- Twitch


def test_twitch_bearer_swap_and_secret_scrub(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """The Twitch surrogate becomes the real bearer; client_secret is scrubbed."""
    twitch_config.save(
        {
            "client_id": "cid1",
            "client_secret": "cs-secret",
            "access_token": _REAL_TWITCH_TOKEN,
            "channel_name": "kisscaster",
        }
    )
    backend = TwitchChannelBackend(helix_base=api_server.base("/helix"))
    assert backend.connect() is True
    assert backend._muse is True
    assert api_server.header("Authorization") == f"Bearer {_REAL_TWITCH_TOKEN}"
    assert api_server.header("Client-ID") == "cid1"
    assert backend._access_token.startswith("muse-sgt.twitch.")
    stored = json.loads(twitch_config.path.read_text())
    assert "access_token" not in stored
    assert "client_secret" not in stored
    assert stored["client_id"] == "cid1"
    # Reads flow without grants; a chat send is a write.
    assert json.loads(backend.get_stream_info("kisscaster"))["ok"] is True
    denied = json.loads(backend.send_chat_message("T1", "T1", "hi"))
    assert denied.get("ok") is False
    grant("twitch", "write", "once")
    assert json.loads(backend.send_chat_message("T1", "T1", "hi"))["ok"] is True


def test_twitch_muse_authenticate_rollback_and_clear(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """authenticate_twitch never persists secrets and rolls back bad tokens."""
    from kiss.agents.third_party_agents.twitch_agent import TwitchAgent

    agent = TwitchAgent()
    agent._backend._helix_base = api_server.base("/helix")
    tools = _auth_tools(agent)
    result = json.loads(
        tools["authenticate_twitch"]("cid1", "cs-secret", _REAL_TWITCH_TOKEN, "kisscaster")
    )
    assert result["ok"] is True
    assert result["login"] == "kisscaster"
    stored = json.loads(twitch_config.path.read_text())
    assert stored == {"client_id": "cid1", "channel_name": "kisscaster"}
    assert vault_has_credentials("twitch")
    assert json.loads(tools["check_twitch_auth"]())["ok"] is True
    result = json.loads(
        tools["authenticate_twitch"]("cid1", "", "twitch-rotated-invalid", "")
    )
    assert result["ok"] is False
    assert not vault_has_credentials("twitch")
    assert "Not authenticated" in tools["check_twitch_auth"]()
    assert json.loads(
        tools["authenticate_twitch"]("cid1", "", _REAL_TWITCH_TOKEN, "")
    )["ok"] is True
    assert "cleared" in tools["clear_twitch_auth"]()
    assert not vault_has_credentials("twitch")


def test_twitch_wire_failure_and_off_allowlist(
    muse_env: Path, api_server: _MessagingApiServer, rogue_server: Any
) -> None:
    """An empty vault fails closed; an off-allowlist origin is denied."""
    backend = TwitchChannelBackend(helix_base=api_server.base("/helix"))
    assert backend.connect() is False
    assert "No Twitch credential" in backend._connection_info
    # Enroll, then aim the backend at a host missing from the twitch
    # allowlist (the policy allows 127.0.0.1; use localhost instead).
    twitch_config.save({"client_id": "cid1", "access_token": _REAL_TWITCH_TOKEN})
    backend = TwitchChannelBackend(
        helix_base=f"http://localhost:{rogue_server.port}/helix"
    )
    assert backend.connect() is False
    assert "not in the 'twitch' connector's allowlist" in backend._connection_info
    assert not rogue_server.requests


# --------------------------------------------------------------------- Zalo


def test_zalo_header_kind_swap_and_scrub(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """The surrogate bearer becomes Zalo's real ``access_token`` header."""
    zalo_config.save({"access_token": _REAL_ZALO_TOKEN, "oa_id": "OA1"})
    backend = ZaloChannelBackend(api_base=api_server.base("/v2.0/oa"))
    assert backend._wire_muse() is True
    assert backend._access_token.startswith("muse-sgt.zalo.")
    result = json.loads(backend.get_oa_info())
    assert result["ok"] is True
    # The server saw the REAL token in the custom header and no
    # Authorization header at all (the surrogate bearer was consumed).
    assert api_server.header("access_token") == _REAL_ZALO_TOKEN
    assert api_server.header("Authorization") == ""
    stored = json.loads(zalo_config.path.read_text())
    assert stored == {"oa_id": "OA1"}
    assert vault_has_credentials("zalo")


def test_zalo_write_grant_and_multipart_upload(
    muse_env: Path, api_server: _MessagingApiServer, tmp_path: Path
) -> None:
    """Sends/uploads are writes; multipart bodies survive the boundary."""
    zalo_config.save({"access_token": _REAL_ZALO_TOKEN})
    backend = ZaloChannelBackend(api_base=api_server.base("/v2.0/oa"))
    assert backend._wire_muse() is True
    denied = json.loads(backend.send_text_message("U9", "hello"))
    assert denied["ok"] is False
    grant("zalo", "write", "session")
    assert json.loads(backend.send_text_message("U9", "hello"))["ok"] is True
    assert api_server.header("access_token") == _REAL_ZALO_TOKEN
    image = tmp_path / "pic.png"
    image.write_bytes(b"png-bytes-123")
    result = json.loads(backend.upload_image(str(image)))
    assert result == {"ok": True, "attachment_id": "ATT1"}
    recorded = api_server.requests[-1]
    assert recorded["path"].endswith("/upload/image")
    assert "multipart/form-data" in api_server.header("Content-Type")
    assert "png-bytes-123" in recorded["body"]


def test_zalo_muse_authenticate_rollback_and_clear(
    muse_env: Path, api_server: _MessagingApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """authenticate_zalo enrolls into the vault and rolls back bad tokens."""
    from kiss.agents.third_party_agents.zalo_agent import ZaloAgent

    monkeypatch.setenv("ZALO_API_BASE", api_server.base("/v2.0/oa"))
    agent = ZaloAgent()
    tools = _auth_tools(agent)
    result = json.loads(tools["authenticate_zalo"](_REAL_ZALO_TOKEN, oa_id="OA1"))
    assert result["ok"] is True
    assert vault_has_credentials("zalo")
    assert json.loads(zalo_config.path.read_text()) == {"oa_id": "OA1"}
    assert json.loads(tools["check_zalo_auth"]())["ok"] is True
    result = json.loads(tools["authenticate_zalo"]("zalo-rotated-invalid"))
    assert result["ok"] is False
    assert not vault_has_credentials("zalo")
    assert "Not authenticated" in tools["check_zalo_auth"]()
    assert json.loads(tools["authenticate_zalo"](_REAL_ZALO_TOKEN))["ok"] is True
    assert "cleared" in tools["clear_zalo_auth"]()
    assert not vault_has_credentials("zalo")


# --------------------------------------------------------------------- LINE


def test_line_muse_adapter_reads_and_writes(
    muse_env: Path, api_server: _MessagingApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LINE Muse adapter spends the real bearer only at the boundary."""
    monkeypatch.setenv("LINE_API_BASE", api_server.base(""))
    line_config.save({"channel_access_token": _REAL_LINE_TOKEN, "channel_secret": "cs1"})
    backend = LineChannelBackend()
    assert backend._wire_muse() is True
    # Reads: quota and profile (camelCase JSON mapped to snake_case).
    assert json.loads(backend.get_quota()) == {"ok": True, "type": "limited", "value": 500}
    assert api_server.header("Authorization") == f"Bearer {_REAL_LINE_TOKEN}"
    profile = json.loads(backend.get_profile("U1"))
    assert profile["display_name"] == "Ann"
    assert profile["user_id"] == "U1"
    # The channel access token is scrubbed; the inbound webhook secret
    # (never egressed) survives.
    assert json.loads(line_config.path.read_text()) == {"channel_secret": "cs1"}
    # Writes ask for a grant.
    denied = json.loads(backend.push_text_message("U1", "hello"))
    assert denied["ok"] is False
    assert "requires user approval" in denied["error"]
    grant("line", "write", "session")
    assert json.loads(backend.push_text_message("U1", "hello"))["ok"] is True
    sent = json.loads(api_server.requests[-1]["body"])
    assert sent == {"to": "U1", "messages": [{"type": "text", "text": "hello"}]}
    assert json.loads(
        backend.reply_message("rt1", '[{"type":"text","text":"pong"}]')
    )["ok"] is True
    assert json.loads(backend.push_image_message("U1", "http://i/1.png", "http://i/2.png"))[
        "ok"
    ] is True
    assert json.loads(backend.leave_group("G1"))["ok"] is True
    assert api_server.requests[-1]["path"].endswith("/v2/bot/group/G1/leave")


def test_line_muse_authenticate_rollback_and_clear(
    muse_env: Path, api_server: _MessagingApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """authenticate_line enrolls into the vault and rolls back bad tokens."""
    from kiss.agents.third_party_agents.line_agent import LineAgent

    monkeypatch.setenv("LINE_API_BASE", api_server.base(""))
    agent = LineAgent()
    tools = _auth_tools(agent)
    result = json.loads(tools["authenticate_line"](_REAL_LINE_TOKEN, "cs1"))
    assert result["ok"] is True
    assert vault_has_credentials("line")
    assert json.loads(line_config.path.read_text()) == {"channel_secret": "cs1"}
    assert json.loads(tools["check_line_auth"]())["ok"] is True
    result = json.loads(tools["authenticate_line"]("line-rotated-invalid"))
    assert result["ok"] is False
    assert not vault_has_credentials("line")
    assert agent._backend._api is None
    assert json.loads(tools["authenticate_line"](_REAL_LINE_TOKEN))["ok"] is True
    assert "cleared" in tools["clear_line_auth"]()
    assert not vault_has_credentials("line")
    assert agent._backend._api is None


def test_line_wire_failure_without_credential(muse_env: Path) -> None:
    """An empty vault and config fail closed."""
    backend = LineChannelBackend()
    assert backend._wire_muse() is False
    assert "No LINE credential" in backend._connection_info


# ----------------------------------------------------------- Nextcloud Talk


def _nc_expected_basic() -> str:
    """Return the Basic header value the emulator must receive."""
    return "Basic " + base64.b64encode(f"bot:{_REAL_NC_PASSWORD}".encode()).decode()


def test_nextcloud_basic_swap_and_scrub(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """The surrogate bearer becomes the real ``Authorization: Basic``."""
    nc_config.save(
        {"url": api_server.base(), "username": "bot", "password": _REAL_NC_PASSWORD}
    )
    backend = NextcloudTalkChannelBackend()
    assert backend.connect() is True
    assert backend._muse is True
    assert api_server.header("Authorization") == _nc_expected_basic()
    assert api_server.header("OCS-APIRequest") == "true"
    assert backend._surrogate.startswith("muse-sgt.nextcloud.")
    # The password is scrubbed; the username (needed for is_from_bot)
    # and URL survive.
    stored = json.loads(nc_config.path.read_text())
    assert "password" not in stored
    assert stored["username"] == "bot"
    assert backend.is_from_bot({"user": "bot"}) is True
    assert backend.is_from_bot({"user": "ann"}) is False
    assert vault_has_credentials("nextcloud")


def test_nextcloud_join_is_read_and_writes_need_grants(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """participants/active is a state-changing join: a write, like posting."""
    nc_config.save(
        {"url": api_server.base(), "username": "bot", "password": _REAL_NC_PASSWORD}
    )
    backend = NextcloudTalkChannelBackend()
    assert backend.connect() is True
    api_server.requests.clear()
    # Joining a room creates an active participant session on the
    # server; without a write grant it must never reach the API.
    backend.join_channel("r1")
    assert api_server.requests == []
    # Reads: room listing and message polling.
    messages, cursor = backend.poll_messages("r1", "")
    assert [m["text"] for m in messages] == ["hello"]
    assert cursor == "7"
    # Writes: sending raises on the Sentinel denial, then a grant opens
    # both the message send and the participant join.
    with pytest.raises(RuntimeError, match="send failed"):
        backend.send_message("r1", "hi there")
    grant("nextcloud", "write", "session")
    backend.join_channel("r1")
    assert api_server.requests[-1]["path"].endswith("/room/r1/participants/active")
    assert api_server.header("Authorization") == _nc_expected_basic()
    backend.send_message("r1", "hi there")
    assert json.loads(backend.post_message("r1", "again"))["ok"] is True
    assert json.loads(backend.set_room_name("r1", "Renamed"))["ok"] is True
    assert json.loads(backend.delete_message("r1", 7))["ok"] is True


def test_nextcloud_muse_authenticate_rollback_and_clear(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """authenticate_nextcloud enrolls the Basic pair and rolls back."""
    from kiss.agents.third_party_agents.nextcloud_talk_agent import NextcloudTalkAgent

    agent = NextcloudTalkAgent()
    tools = _auth_tools(agent)
    result = json.loads(
        tools["authenticate_nextcloud"](api_server.base(), "bot", _REAL_NC_PASSWORD)
    )
    assert result["ok"] is True
    assert vault_has_credentials("nextcloud")
    stored = json.loads(nc_config.path.read_text())
    assert "password" not in stored
    assert json.loads(tools["check_nextcloud_auth"]())["ok"] is True
    result = json.loads(
        tools["authenticate_nextcloud"](api_server.base(), "bot", "nc-rotated-invalid")
    )
    assert result["ok"] is False
    assert not vault_has_credentials("nextcloud")
    result = json.loads(
        tools["authenticate_nextcloud"]("https://bad..host", "bot", _REAL_NC_PASSWORD)
    )
    assert result["ok"] is False
    assert "not a valid" in result["error"]
    assert json.loads(
        tools["authenticate_nextcloud"](api_server.base(), "bot", _REAL_NC_PASSWORD)
    )["ok"] is True
    assert "cleared" in tools["clear_nextcloud_auth"]()
    assert not vault_has_credentials("nextcloud")


def test_nextcloud_wire_failure_paths(muse_env: Path) -> None:
    """Missing config, malformed URL, and empty vault each fail closed."""
    backend = NextcloudTalkChannelBackend()
    assert backend._wire_muse() is False
    assert "No Nextcloud config found" in backend._connection_info
    nc_config.save({"url": "https://bad..host", "username": "bot", "password": "pw"})
    assert backend._wire_muse() is False
    assert "not a valid" in backend._connection_info
    # The malformed config was NOT scrubbed (fail closed).
    assert json.loads(nc_config.path.read_text())["password"] == "pw"
    nc_config.clear()
    nc_config.save({"url": "https://cloud.example.com", "username": "bot", "password": ""})
    assert backend._wire_muse() is False
    assert "No Nextcloud credential" in backend._connection_info


# -------------------------------------------------------------- BlueBubbles


def _bb_backend(api_server: _MessagingApiServer) -> BlueBubblesChannelBackend:
    """Enroll and wire a BlueBubbles backend against the emulator."""
    bb_config.save({"server_url": api_server.base(), "password": _REAL_BB_PASSWORD})
    backend = BlueBubblesChannelBackend()
    assert backend._wire_muse() is True
    return backend


def test_bluebubbles_query_kind_password_injection(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """The real password travels only as a boundary-injected query param."""
    backend = _bb_backend(api_server)
    assert backend._muse is True
    assert backend._surrogate.startswith("muse-sgt.bluebubbles.")
    # The password is scrubbed from the config; server_url survives.
    assert json.loads(bb_config.path.read_text()) == {"server_url": api_server.base()}
    assert vault_has_credentials("bluebubbles")
    # message/query is a POST that only retrieves data: a read.
    messages, cursor = backend.poll_messages("chat1", "")
    assert [m["text"] for m in messages] == ["hi"]
    assert cursor == "1700000000000"
    recorded = api_server.query()
    assert recorded.get("password") == [_REAL_BB_PASSWORD]
    # The surrogate bearer was consumed at the boundary; nothing
    # Authorization-shaped reaches the server for query-kind services.
    assert api_server.header("Authorization") == ""
    # Sending is a write: denial raises, a grant opens it.
    with pytest.raises(RuntimeError, match="send failed"):
        backend.send_message("chat1", "hello")
    grant("bluebubbles", "write", "once")
    backend.send_message("chat1", "hello")
    assert api_server.query().get("password") == [_REAL_BB_PASSWORD]


def test_bluebubbles_caller_password_param_is_stripped(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """A caller-supplied ``password=`` never survives next to the real one."""
    backend = _bb_backend(api_server)
    session = MuseBoundarySession("bluebubbles")
    resp = session.get(
        api_server.base("/api/v1/server/info"),
        params={"password": "attacker-chosen", "limit": "5"},
        headers={"Authorization": f"Bearer {backend._surrogate}"},
    )
    assert resp.status_code == 200
    recorded = api_server.query()
    assert recorded.get("password") == [_REAL_BB_PASSWORD]
    assert recorded.get("limit") == ["5"]


def test_bluebubbles_cross_origin_redirect_drops_password(
    muse_env: Path, api_server: _MessagingApiServer, rogue_server: Any
) -> None:
    """A redirect off the enrolled origin is followed without the credential."""
    backend = _bb_backend(api_server)
    api_server.offsite_location = rogue_server.base("/api/v1/server/info")
    session = MuseBoundarySession("bluebubbles")
    resp = session.get(
        api_server.base("/redirect-offsite"),
        headers={"Authorization": f"Bearer {backend._surrogate}"},
    )
    assert resp.status_code == 200
    assert len(rogue_server.requests) == 1
    assert rogue_server.query().get("password") is None
    assert rogue_server.header("Authorization") == ""
    # The on-origin hop DID carry the credential.
    assert api_server.query(index=-1).get("password") == [_REAL_BB_PASSWORD]


def test_bluebubbles_rotation_invalidates_surrogates(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """Replacing a query-kind credential kills outstanding surrogates."""
    backend = _bb_backend(api_server)
    old_surrogate = backend._surrogate
    store_credentials(
        "bluebubbles",
        {"kind": "query", "param": "password", "token": "bb-rotated"},
        [],
        hosts=(f"127.0.0.1:{api_server.port}",),
    )
    session = MuseBoundarySession("bluebubbles")
    with pytest.raises(MuseAuthError, match="stale surrogate"):
        session.get(
            api_server.base("/api/v1/server/info"),
            headers={"Authorization": f"Bearer {old_surrogate}"},
        )
    # A fresh wire uses the rotated credential.
    assert backend._wire_muse() is True
    backend.poll_messages("chat1", "")
    assert api_server.query().get("password") == ["bb-rotated"]


def test_bluebubbles_wire_failure_paths(muse_env: Path) -> None:
    """Missing config, malformed URL, and empty vault each fail closed."""
    backend = BlueBubblesChannelBackend()
    assert backend._wire_muse() is False
    assert "No BlueBubbles config found" in backend._connection_info
    bb_config.save({"server_url": "http://bad..host:1234", "password": "pw"})
    assert backend._wire_muse() is False
    assert "not a valid" in backend._connection_info
    assert json.loads(bb_config.path.read_text())["password"] == "pw"
    bb_config.clear()
    bb_config.save({"server_url": "http://127.0.0.1:9", "password": ""})
    assert backend._wire_muse() is False
    assert "No BlueBubbles credential" in backend._connection_info


# ------------------------------------------------------------ Synology Chat


def test_synology_webhook_token_extraction_and_send(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """The URL-embedded token moves to the vault; sends re-inject it."""
    webhook = api_server.base(
        f"/webapi/entry.cgi?api=SYNO.Chat.External&method=incoming&version=2"
        f"&token={_REAL_SYNO_TOKEN}"
    )
    syno_config.save({"webhook_url": webhook, "token": "outgoing-verify"})
    backend = SynologyChatChannelBackend()
    assert backend._wire_muse() is True
    assert backend._muse is True
    # The stored URL lost its token; the outgoing verification token
    # (inbound-only) survives.
    stored = json.loads(syno_config.path.read_text())
    assert "token=" not in stored["webhook_url"]
    assert stored["token"] == "outgoing-verify"
    assert vault_has_credentials("synology")
    assert "token=" not in backend._webhook_url
    # Sending is a write: denial raises, a grant opens it; the emulator
    # only reports success when the REAL token arrives in the query.
    with pytest.raises(RuntimeError, match="send failed"):
        backend.send_message("", "hello team")
    grant("synology", "write", "once")
    backend.send_message("", "hello team")
    recorded = api_server.query()
    assert recorded.get("token") == [_REAL_SYNO_TOKEN]
    assert recorded.get("api") == ["SYNO.Chat.External"]
    assert "payload" in parse_qs(api_server.requests[-1]["body"])
    grant("synology", "write", "session")
    assert json.loads(backend.post_message("hi", user_ids="u1,u2"))["ok"] is True
    assert json.loads(backend.send_file_message("hi", "http://f/1.png"))["ok"] is True


def test_synology_tokenless_webhook_stays_legacy(muse_env: Path) -> None:
    """A webhook URL with no embedded token keeps the legacy direct path."""
    syno_config.save({"webhook_url": "http://127.0.0.1:9/webapi/entry.cgi?api=X"})
    backend = SynologyChatChannelBackend()
    assert backend._wire_muse() is True
    assert backend._muse is False
    assert backend._http is requests
    assert not vault_has_credentials("synology")


def test_synology_muse_authenticate_and_clear(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """authenticate_synology scrubs the URL and enrolls the vault."""
    from kiss.agents.third_party_agents.synology_chat_agent import SynologyChatAgent

    agent = SynologyChatAgent()
    tools = _auth_tools(agent)
    webhook = api_server.base(f"/webapi/entry.cgi?api=X&token={_REAL_SYNO_TOKEN}")
    result = json.loads(tools["authenticate_synology"](webhook, token="verify1"))
    assert result["ok"] is True
    assert vault_has_credentials("synology")
    stored = json.loads(syno_config.path.read_text())
    assert "token=" not in stored["webhook_url"]
    assert stored["token"] == "verify1"
    assert "ok" in tools["check_synology_auth"]()
    result = json.loads(tools["authenticate_synology"]("http://bad..host/x", token=""))
    assert result["ok"] is False
    assert "not a valid" in result["error"]
    assert "cleared" in tools["clear_synology_auth"]()
    assert not vault_has_credentials("synology")


def test_synology_wire_failure_paths(muse_env: Path) -> None:
    """Missing config and malformed URL fail closed without scrubbing."""
    backend = SynologyChatChannelBackend()
    assert backend._wire_muse() is False
    assert "No Synology Chat config found" in backend._connection_info
    bad = "http://bad..host/webapi/entry.cgi?token=s"
    syno_config.save({"webhook_url": bad})
    assert backend._wire_muse() is False
    assert "not a valid" in backend._connection_info
    assert json.loads(syno_config.path.read_text())["webhook_url"] == bad


# ----------------------------------------------- query-kind daemon contract


def test_query_kind_store_validation(muse_env: Path) -> None:
    """Malformed query-kind credentials are refused at enrollment."""
    ensure_daemon()
    with pytest.raises(MuseAuthError, match="query parameter name"):
        store_credentials(
            "bluebubbles", {"kind": "query", "param": "bad name!", "token": "pw"}, []
        )
    with pytest.raises(MuseAuthError, match="token value"):
        store_credentials(
            "bluebubbles", {"kind": "query", "param": "password", "token": "bad\nvalue"}, []
        )
    assert not vault_has_credentials("bluebubbles")


def test_query_kind_vault_resolution(muse_env: Path) -> None:
    """The vault resolves query-kind credentials with their placement."""
    from kiss.agents.third_party_agents.muse_auth.vault import CredentialVault

    vault = CredentialVault()
    vault.store("synology", {"kind": "query", "param": "token", "token": "qv"}, [])
    assert vault.resolve_credential("synology") == ("query", "token", "qv")
    vault.store("zalo", {"kind": "header", "header": "access_token", "token": "hv"}, [])
    assert vault.resolve_credential("zalo") == ("header", "access_token", "hv")
    vault.clear("synology")
    vault.clear("zalo")


def test_daemon_protocol_is_current(muse_env: Path) -> None:
    """A freshly ensured daemon reports the v4 (query-kind) protocol."""
    ensure_daemon()
    assert _daemon_protocol() == PROTOCOL_VERSION == 4


def test_underscore_header_kind_accepted(muse_env: Path) -> None:
    """Zalo's ``access_token`` header name (with underscore) enrolls fine."""
    store_credentials(
        "zalo", {"kind": "header", "header": "access_token", "token": "tok"}, []
    )
    assert mint_surrogate("zalo") is not None
    clear_credentials("zalo")


# -------------------------------------------------------------- CLI imports


def test_cli_import_messaging_services(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """``muse_auth import`` migrates every messaging connector's secrets."""
    mm_config.save(
        {"url": "127.0.0.1", "token": _REAL_MM_TOKEN, "port": str(api_server.port),
         "scheme": "http"}
    )
    assert muse_cli.main(["import", "mattermost"]) == 0
    assert vault_has_credentials("mattermost")
    assert "token" not in json.loads(mm_config.path.read_text())

    twitch_config.save(
        {"client_id": "cid1", "client_secret": "cs", "access_token": _REAL_TWITCH_TOKEN}
    )
    assert muse_cli.main(["import", "twitch"]) == 0
    stored = json.loads(twitch_config.path.read_text())
    assert "access_token" not in stored
    assert "client_secret" not in stored
    assert stored["client_id"] == "cid1"

    zalo_config.save({"access_token": _REAL_ZALO_TOKEN, "oa_id": "OA1"})
    assert muse_cli.main(["import", "zalo"]) == 0
    assert json.loads(zalo_config.path.read_text()) == {"oa_id": "OA1"}

    line_config.save({"channel_access_token": _REAL_LINE_TOKEN, "channel_secret": "cs1"})
    assert muse_cli.main(["import", "line"]) == 0
    assert json.loads(line_config.path.read_text()) == {"channel_secret": "cs1"}

    nc_config.save(
        {"url": api_server.base(), "username": "bot", "password": _REAL_NC_PASSWORD}
    )
    assert muse_cli.main(["import", "nextcloud"]) == 0
    stored = json.loads(nc_config.path.read_text())
    assert "password" not in stored
    assert stored["username"] == "bot"

    bb_config.save({"server_url": api_server.base(), "password": _REAL_BB_PASSWORD})
    assert muse_cli.main(["import", "bluebubbles"]) == 0
    assert json.loads(bb_config.path.read_text()) == {"server_url": api_server.base()}

    webhook = api_server.base(f"/webapi/entry.cgi?api=X&token={_REAL_SYNO_TOKEN}")
    syno_config.save({"webhook_url": webhook})
    assert muse_cli.main(["import", "synology"]) == 0
    stored = json.loads(syno_config.path.read_text())
    assert "token=" not in stored["webhook_url"]
    for service in (
        "mattermost", "twitch", "zalo", "line", "nextcloud", "bluebubbles", "synology"
    ):
        assert vault_has_credentials(service)
    # The imported credentials are actually spendable: a Synology send
    # succeeds against the emulator (which validates the real token).
    backend = SynologyChatChannelBackend()
    assert backend._wire_muse() is True
    grant("synology", "write", "once")
    backend.send_message("", "imported and working")
    assert api_server.query().get("token") == [_REAL_SYNO_TOKEN]


def test_cli_import_rejects_malformed_configs(muse_env: Path) -> None:
    """Imports fail closed on missing/invalid URLs and secrets."""
    # Mattermost: composed URL must validate.
    mm_config.save({"url": "bad..host", "token": "t", "port": "443", "scheme": "https"})
    assert muse_cli.main(["import", "mattermost"]) == 1
    assert not vault_has_credentials("mattermost")
    assert json.loads(mm_config.path.read_text())["token"] == "t"
    # BlueBubbles: server_url is required and must validate.
    bb_config.save({"password": "pw"})
    assert muse_cli.main(["import", "bluebubbles"]) == 1
    bb_config.save({"server_url": "http://bad..host", "password": "pw"})
    assert muse_cli.main(["import", "bluebubbles"]) == 1
    assert not vault_has_credentials("bluebubbles")
    # Nextcloud: needs url+username+password, and a valid URL.
    assert muse_cli.main(["import", "nextcloud"]) == 1
    nc_config.save({"url": "https://cloud.example.com", "username": "bot", "password": ""})
    assert muse_cli.main(["import", "nextcloud"]) == 1
    nc_config.save({"url": "https://u:p@cloud.example.com", "username": "b", "password": "pw"})
    assert muse_cli.main(["import", "nextcloud"]) == 1
    assert not vault_has_credentials("nextcloud")
    # Synology: needs a config, a valid URL, and an embedded token.
    assert muse_cli.main(["import", "synology"]) == 1
    syno_config.save({"webhook_url": "http://bad..host/webapi/entry.cgi?token=s"})
    assert muse_cli.main(["import", "synology"]) == 1
    syno_config.save({"webhook_url": "http://127.0.0.1:9/webapi/entry.cgi?api=X"})
    assert muse_cli.main(["import", "synology"]) == 1
    assert not vault_has_credentials("synology")


# --------------------------------------------------- legacy mode unchanged


def test_legacy_mode_unchanged(
    isolated_kiss_home: Path, api_server: _MessagingApiServer
) -> None:
    """With Muse off, every connector still sends its credential directly."""
    assert os.environ.get("KISS_MUSE_AUTH", "") in ("", "0")
    # Mattermost: the typing indicator posts the real bearer directly.
    backend_mm = MattermostChannelBackend(base_url=api_server.base(), token=_REAL_MM_TOKEN)
    backend_mm.send_typing("C1")
    assert api_server.header("Authorization") == f"Bearer {_REAL_MM_TOKEN}"
    assert backend_mm._http is requests
    # Twitch: direct connect with the stored token.
    twitch_config.save({"client_id": "cid1", "access_token": _REAL_TWITCH_TOKEN})
    backend_tw = TwitchChannelBackend(helix_base=api_server.base("/helix"))
    assert backend_tw.connect() is True
    assert backend_tw._muse is False
    assert backend_tw._access_token == _REAL_TWITCH_TOKEN
    assert api_server.header("Authorization") == f"Bearer {_REAL_TWITCH_TOKEN}"
    # Zalo: the real token rides the custom header directly.
    zalo_config.save({"access_token": _REAL_ZALO_TOKEN})
    backend_z = ZaloChannelBackend(api_base=api_server.base("/v2.0/oa"))
    backend_z._access_token = _REAL_ZALO_TOKEN
    assert json.loads(backend_z.get_oa_info())["ok"] is True
    assert api_server.header("access_token") == _REAL_ZALO_TOKEN
    assert backend_z._headers() == {"access_token": _REAL_ZALO_TOKEN}
    # Nextcloud: requests derives the Basic header from the auth tuple.
    nc_config.save(
        {"url": api_server.base(), "username": "bot", "password": _REAL_NC_PASSWORD}
    )
    backend_nc = NextcloudTalkChannelBackend()
    assert backend_nc.connect() is True
    assert backend_nc._muse is False
    assert api_server.header("Authorization") == _nc_expected_basic()
    # BlueBubbles: the password rides the query string directly.
    backend_bb = BlueBubblesChannelBackend()
    backend_bb._server_url = api_server.base()
    backend_bb._password = _REAL_BB_PASSWORD
    backend_bb.poll_messages("chat1", "")
    assert api_server.query().get("password") == [_REAL_BB_PASSWORD]
    assert backend_bb._headers() == {}
    # Synology: the token-bearing webhook URL is posted directly.
    backend_sy = SynologyChatChannelBackend()
    backend_sy._webhook_url = api_server.base(
        f"/webapi/entry.cgi?api=X&token={_REAL_SYNO_TOKEN}"
    )
    backend_sy.send_message("", "legacy hello")
    assert api_server.query().get("token") == [_REAL_SYNO_TOKEN]
    # LINE: the SDK is not installed here, so the legacy constructor
    # fails and the authenticate tool reports the error (no config write).
    from kiss.agents.third_party_agents.line_agent import LineAgent

    line_config.clear()
    agent = LineAgent()
    tools = _auth_tools(agent)
    result = json.loads(tools["authenticate_line"]("tok"))
    assert result["ok"] is False
    assert line_config.load() is None


def test_scrub_helpers_tolerate_malformed_configs(muse_env: Path) -> None:
    """Scrub helpers no-op on unreadable, non-dict, or secret-free files."""
    from kiss.agents.third_party_agents.bluebubbles_agent import _scrub_config_password
    from kiss.agents.third_party_agents.line_agent import _scrub_config_token
    from kiss.agents.third_party_agents.mattermost_agent import (
        _scrub_config_token as mm_scrub,
    )
    from kiss.agents.third_party_agents.synology_chat_agent import (
        _scrub_config_webhook_token,
    )
    from kiss.agents.third_party_agents.twitch_agent import _scrub_config_secrets
    from kiss.agents.third_party_agents.zalo_agent import (
        _scrub_config_token as zalo_scrub,
    )

    for cfg, scrub in (
        (bb_config, _scrub_config_password),
        (line_config, _scrub_config_token),
        (mm_config, mm_scrub),
        (syno_config, _scrub_config_webhook_token),
        (twitch_config, _scrub_config_secrets),
        (zalo_config, zalo_scrub),
    ):
        scrub()  # missing file
        cfg.path.parent.mkdir(parents=True, exist_ok=True)
        cfg.path.write_text("not json")
        scrub()  # unreadable
        assert cfg.path.read_text() == "not json"
        cfg.path.write_text('["a list"]')
        scrub()  # non-dict
        assert json.loads(cfg.path.read_text()) == ["a list"]
        cfg.path.write_text('{"other": "keep"}')
        scrub()  # nothing secret to remove
        assert json.loads(cfg.path.read_text()) == {"other": "keep"}
        cfg.clear()
    # A config holding ONLY the secret is deleted outright.
    mm_config.save({"url": "", "token": "t"})
    mm_scrub()
    assert not mm_config.path.exists()
    zalo_config.save({"access_token": "t"})
    zalo_scrub()
    assert not zalo_config.path.exists()
    twitch_config.save({"client_secret": "cs", "access_token": "t"})
    _scrub_config_secrets()
    assert not twitch_config.path.exists()
    line_config.save({"channel_access_token": "t", "channel_secret": ""})
    _scrub_config_token()
    assert not line_config.path.exists()
    bb_config.save({"server_url": "", "password": "pw"})
    _scrub_config_password()
    assert not bb_config.path.exists()


def test_base_url_from_config_edge_cases(muse_env: Path) -> None:
    """Mattermost's composed base URL tolerates junk config values."""
    from kiss.agents.third_party_agents.mattermost_agent import _base_url_from_config

    assert _base_url_from_config({}) == ""
    assert _base_url_from_config({"url": "mm.example.com"}) == "https://mm.example.com:443"
    assert (
        _base_url_from_config({"url": "mm.example.com", "scheme": "http", "port": "8065"})
        == "http://mm.example.com:8065"
    )
    assert _base_url_from_config({"url": "mm.example.com", "port": "junk"}) == ""


def test_embedded_token_helper(muse_env: Path) -> None:
    """Synology's embedded-token extraction handles all URL shapes."""
    from kiss.agents.third_party_agents.synology_chat_agent import _embedded_token

    assert _embedded_token("http://nas:5001/webapi/entry.cgi?api=X&token=abc") == "abc"
    assert _embedded_token("http://nas:5001/webapi/entry.cgi?api=X") == ""
    assert _embedded_token("not a url") == ""
    assert _embedded_token("http://nas:5001/x?token=%22quoted%22") == '"quoted"'


# ------------------------------------------------------ coverage completion


def test_connect_failure_paths_under_muse(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """connect() fails closed without credentials or with a bad token."""
    assert MattermostChannelBackend().connect() is False
    assert ZaloChannelBackend(api_base=api_server.base("/v2.0/oa")).connect() is False
    assert SynologyChatChannelBackend().connect() is False
    backend_line = LineChannelBackend()
    assert backend_line.connect() is False
    assert "No LINE credential" in backend_line._connection_info
    # A bad enrolled Mattermost token makes the boundary /users/me 401.
    _mm_config(api_server, token="mm-token-invalid")
    backend = MattermostChannelBackend()
    assert backend.connect() is False
    assert "Mattermost connection failed" in backend._connection_info


def test_make_backends_muse_mode(
    muse_env: Path, api_server: _MessagingApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every _make_backend wires the Muse boundary or exits when unenrolled."""
    from kiss.agents.third_party_agents import (
        bluebubbles_agent,
        line_agent,
        mattermost_agent,
        nextcloud_talk_agent,
        synology_chat_agent,
        zalo_agent,
    )

    for module in (
        bluebubbles_agent,
        line_agent,
        mattermost_agent,
        nextcloud_talk_agent,
        synology_chat_agent,
        zalo_agent,
    ):
        with pytest.raises(SystemExit):
            module._make_backend()
    monkeypatch.setenv("LINE_API_BASE", api_server.base(""))
    _mm_config(api_server)
    zalo_config.save({"access_token": _REAL_ZALO_TOKEN})
    line_config.save({"channel_access_token": _REAL_LINE_TOKEN})
    nc_config.save(
        {"url": api_server.base(), "username": "bot", "password": _REAL_NC_PASSWORD}
    )
    bb_config.save({"server_url": api_server.base(), "password": _REAL_BB_PASSWORD})
    syno_config.save(
        {"webhook_url": api_server.base(f"/webapi/entry.cgi?api=X&token={_REAL_SYNO_TOKEN}")}
    )
    assert mattermost_agent._make_backend()._muse is True
    assert zalo_agent._make_backend()._muse is True
    assert line_agent._make_backend()._api is not None
    assert nextcloud_talk_agent._make_backend()._muse is True
    assert bluebubbles_agent._make_backend()._muse is True
    assert synology_chat_agent._make_backend()._muse is True


def test_mattermost_login_and_reaction(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """The shim's login() read and create_reaction write work at the boundary."""
    _mm_config(api_server)
    backend = MattermostChannelBackend()
    assert backend.connect() is True
    assert backend._driver.login()["username"] == "mmbot"
    denied = json.loads(backend.add_reaction("U1", "P1", "thumbsup"))
    assert denied["ok"] is False
    grant("mattermost", "write", "once")
    assert json.loads(backend.add_reaction("U1", "P1", "thumbsup"))["ok"] is True
    assert api_server.requests[-1]["path"].endswith("/api/v4/reactions")


def test_zalo_remaining_reads_and_image_send(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """Every remaining Zalo tool method flows through the boundary."""
    zalo_config.save({"access_token": _REAL_ZALO_TOKEN})
    backend = ZaloChannelBackend(api_base=api_server.base("/v2.0/oa"))
    assert backend._wire_muse() is True
    assert json.loads(backend.get_follower_profile("U1"))["ok"] is True
    assert json.loads(backend.get_followers())["ok"] is True
    assert json.loads(backend.get_recent_messages())["ok"] is True
    assert json.loads(backend.get_conversation("U1"))["ok"] is True
    assert api_server.header("access_token") == _REAL_ZALO_TOKEN
    grant("zalo", "write", "once")
    assert json.loads(backend.send_image_message("U1", "http://i/1.png", "cap"))["ok"] is True
    assert api_server.requests[-1]["path"].endswith("/message")


def test_line_send_message_and_legacy_paths(
    muse_env: Path, api_server: _MessagingApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """send_message uses the adapter; legacy paths surface the missing SDK."""
    monkeypatch.setenv("LINE_API_BASE", api_server.base(""))
    line_config.save({"channel_access_token": _REAL_LINE_TOKEN})
    backend = LineChannelBackend()
    assert backend._wire_muse() is True
    grant("line", "write", "once")
    backend.send_message("U1", "direct send")
    sent = json.loads(api_server.requests[-1]["body"])
    assert sent["messages"][0]["text"] == "direct send"


def test_line_legacy_sdk_missing(
    isolated_kiss_home: Path, api_server: _MessagingApiServer
) -> None:
    """Without the SDK, legacy LINE construction fails soft (or raises in poll mode)."""
    from kiss.agents.third_party_agents import line_agent
    from kiss.agents.third_party_agents.line_agent import LineAgent

    line_config.save({"channel_access_token": "tok", "channel_secret": ""})
    agent = LineAgent()  # constructor swallows the ImportError
    assert agent._backend._api is None
    backend = LineChannelBackend()
    assert backend.connect() is False
    assert "LINE connection failed" in backend._connection_info
    with pytest.raises(Exception, match="linebot"):
        line_agent._make_backend()


def test_bluebubbles_muse_authenticate_rollbacks(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """_muse_authenticate validates the URL and rolls back on failure.

    On this Linux host the validating ``get_server_info`` read is
    macOS-gated and reports a platform error, so the enrollment is
    rolled back — which is exactly the rollback path under test.  The
    macOS-only success return is documented in the module docstring.
    """
    from kiss.agents.third_party_agents.bluebubbles_agent import _muse_authenticate

    backend = BlueBubblesChannelBackend()
    result = json.loads(_muse_authenticate(backend, "http://bad..host", "pw"))
    assert result["ok"] is False
    assert "not a valid" in result["error"]
    result = json.loads(_muse_authenticate(backend, api_server.base(), _REAL_BB_PASSWORD))
    assert result["ok"] is False
    assert not vault_has_credentials("bluebubbles")
    assert backend._muse is False
    assert backend._http is requests


def test_bluebubbles_agent_init_and_clear_tool(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """The agent wires Muse at construction; the clear tool empties the vault."""
    from kiss.agents.third_party_agents.bluebubbles_agent import BlueBubblesAgent

    bb_config.save({"server_url": api_server.base(), "password": _REAL_BB_PASSWORD})
    agent = BlueBubblesAgent()
    assert agent._backend._muse is True
    assert vault_has_credentials("bluebubbles")
    tools = _auth_tools(agent)
    assert "cleared" in tools["clear_bluebubbles_auth"]()
    assert not vault_has_credentials("bluebubbles")
    assert agent._backend._muse is False


def test_query_kind_same_origin_redirect_reinjects(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """A same-origin redirect hop re-resolves and re-injects the query credential."""
    backend = _bb_backend(api_server)
    session = MuseBoundarySession("bluebubbles")
    resp = session.get(
        api_server.base("/redirect-onsite"),
        headers={"Authorization": f"Bearer {backend._surrogate}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == 200
    # Both the initial hop and the redirected hop carried the credential.
    assert api_server.query(index=-2).get("password") == [_REAL_BB_PASSWORD]
    assert api_server.query(index=-1).get("password") == [_REAL_BB_PASSWORD]
    assert api_server.requests[-1]["path"].split("?")[0] == "/api/v1/server/info"


def test_insecure_origin_hosts_non_loopback(muse_env: Path) -> None:
    """Only plain-HTTP non-loopback bases enroll insecure origins."""
    from kiss.agents.third_party_agents.muse_auth._common import insecure_origin_hosts

    assert insecure_origin_hosts("http://nas.lan:5001") == ("nas.lan:5001",)
    assert insecure_origin_hosts("http://127.0.0.1:5001") == ()
    assert insecure_origin_hosts("https://nas.lan:5001") == ()
    assert insecure_origin_hosts("not a url") == ()


def test_cli_import_synology_empty_url(muse_env: Path) -> None:
    """A config whose webhook_url key is empty is rejected."""
    syno_config.path.parent.mkdir(parents=True, exist_ok=True)
    syno_config.path.write_text(json.dumps({"webhook_url": ""}))
    assert muse_cli.main(["import", "synology"]) == 1
    assert not vault_has_credentials("synology")


def test_nextcloud_scrub_edge_cases(muse_env: Path) -> None:
    """The Nextcloud scrub helper tolerates junk and deletes empty configs."""
    from kiss.agents.third_party_agents.nextcloud_talk_agent import _scrub_config_password

    _scrub_config_password()  # missing file
    nc_config.path.parent.mkdir(parents=True, exist_ok=True)
    nc_config.path.write_text("not json")
    _scrub_config_password()
    assert nc_config.path.read_text() == "not json"
    nc_config.path.write_text('{"url": "https://c.example"}')
    _scrub_config_password()  # nothing secret to remove
    assert json.loads(nc_config.path.read_text()) == {"url": "https://c.example"}
    nc_config.path.write_text('{"password": "pw", "username": ""}')
    _scrub_config_password()  # only the secret was stored
    assert not nc_config.path.exists()

# ------------------------------------------------ review round-1 regressions


def test_query_kind_allowlisted_cross_origin_redirect_drops_credential(
    muse_env: Path, api_server: _MessagingApiServer, rogue_server: Any
) -> None:
    """A cross-origin hop never carries a query credential, even allowlisted.

    A query credential binds to the exact origin it was consented for: a
    redirect to a *different* origin that happens to be on the service's
    policy ``extra_hosts`` may be followed, but with a credential-free
    URL.
    """
    backend = _bb_backend(api_server)
    policy = {
        "defaults": {"read": "allow", "write": "ask"},
        "services": {
            "bluebubbles": {"extra_hosts": [f"127.0.0.1:{rogue_server.port}"]}
        },
    }
    (muse_auth_dir() / "policy.json").write_text(json.dumps(policy))
    api_server.offsite_location = rogue_server.base("/api/v1/server/info")
    session = MuseBoundarySession("bluebubbles")
    resp = session.get(
        api_server.base("/redirect-offsite"),
        headers={"Authorization": f"Bearer {backend._surrogate}"},
    )
    assert resp.status_code == 200
    assert len(rogue_server.requests) == 1
    assert rogue_server.query().get("password") is None
    assert rogue_server.header("Authorization") == ""
    # The on-origin first hop DID carry the credential.
    assert api_server.query(index=0).get("password") == [_REAL_BB_PASSWORD]


def test_query_kind_error_text_hides_encoded_credential(muse_env: Path) -> None:
    """Transport errors never reflect the credential, raw or percent-encoded.

    The sent URL carries the percent-encoded credential and the HTTP
    stack embeds that URL verbatim in exception text (e.g. "Max retries
    exceeded with url: /x?password=a%2Fb"), so plain raw-value redaction
    is not enough.
    """
    from urllib.parse import quote, quote_plus

    secret = "slash/plus+question?percent%"
    store_credentials(
        "bluebubbles",
        {"kind": "query", "param": "password", "token": secret},
        [],
        hosts=("127.0.0.1:9",),
    )
    handle = mint_surrogate("bluebubbles")
    assert handle is not None
    session = MuseBoundarySession("bluebubbles")
    with pytest.raises(MuseAuthError) as err:
        # Port 9 (discard) refuses connections: a real transport error.
        session.get(
            "http://127.0.0.1:9/api/v1/server/info",
            headers={"Authorization": f"Bearer {handle.token}"},
        )
    text = str(err.value)
    assert secret not in text
    assert quote_plus(secret) not in text
    assert quote(secret, safe="") not in text
    # Only the exception class and the credential-free URL survive.
    assert "ConnectionError contacting http://127.0.0.1:9/api/v1/server/info" in text


def test_synology_tokenless_rotation_never_revives_vault_token(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """Rotating to an explicitly tokenless webhook clears the stale vault entry."""
    from kiss.agents.third_party_agents.synology_chat_agent import SynologyChatAgent

    agent = SynologyChatAgent()
    tools = _auth_tools(agent)
    webhook = api_server.base(f"/webapi/entry.cgi?api=X&token={_REAL_SYNO_TOKEN}")
    assert json.loads(tools["authenticate_synology"](webhook))["ok"] is True
    assert vault_has_credentials("synology")
    tokenless = api_server.base("/webapi/entry.cgi?api=X")
    result = json.loads(tools["authenticate_synology"](tokenless))
    assert result["ok"] is True
    assert not vault_has_credentials("synology")
    backend = agent._backend
    assert backend._muse is False
    api_server.requests.clear()
    # The emulator only accepts the REAL token, so the tokenless send is
    # rejected — proving the stale vault token never reached the wire.
    with pytest.raises(RuntimeError, match="send failed"):
        backend.send_message("", "no secret here")
    assert api_server.query().get("token") is None


def test_synology_wire_tokenless_config_ignores_stale_vault(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """A tokenless webhook URL without the migration marker stays direct.

    Even with a stale synology credential in the vault, a config whose
    URL carries no ``token=`` and no ``muse`` marker is explicit user
    intent to run without a secret: no surrogate is minted and the old
    token never reaches the new URL.
    """
    store_credentials(
        "synology",
        {"kind": "query", "param": "token", "token": "stale-synology-token"},
        [],
        hosts=(f"127.0.0.1:{api_server.port}",),
    )
    syno_config.save({"webhook_url": api_server.base("/webapi/entry.cgi?api=X")})
    backend = SynologyChatChannelBackend()
    assert backend._wire_muse() is True
    assert backend._muse is False
    assert backend._http is requests
    # The emulator only accepts the REAL token, so the tokenless send is
    # rejected — proving the stale vault token never reached the wire.
    with pytest.raises(RuntimeError, match="send failed"):
        backend.send_message("", "hello")
    assert api_server.query().get("token") is None
    clear_credentials("synology")


def test_synology_scrubbed_config_marker_rewires(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """A Muse-scrubbed config carries the marker and keeps re-wiring."""
    webhook = api_server.base(f"/webapi/entry.cgi?api=X&token={_REAL_SYNO_TOKEN}")
    syno_config.save({"webhook_url": webhook})
    first = SynologyChatChannelBackend()
    assert first._wire_muse() is True
    stored = json.loads(syno_config.path.read_text())
    assert stored["muse"] == "1"
    assert "token=" not in stored["webhook_url"]
    second = SynologyChatChannelBackend()
    assert second._wire_muse() is True
    assert second._muse is True
    grant("synology", "write", "once")
    second.send_message("", "hi")
    assert api_server.query().get("token") == [_REAL_SYNO_TOKEN]


def test_synology_authenticate_invalid_embedded_token_keeps_config(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """A doomed enrollment never mutates the stored configuration."""
    from kiss.agents.third_party_agents.synology_chat_agent import (
        _muse_authenticate as syno_auth,
    )

    prev = {"webhook_url": api_server.base("/webapi/entry.cgi?api=Y")}
    syno_config.save(prev)
    backend = SynologyChatChannelBackend()
    bad = api_server.base("/webapi/entry.cgi?api=X&token=line%0Abreak")
    result = json.loads(syno_auth(backend, bad, ""))
    assert result["ok"] is False
    assert "control characters" in result["error"]
    assert json.loads(syno_config.path.read_text()) == prev
    assert not vault_has_credentials("synology")


def test_nextcloud_ocs_error_envelope_is_rejected(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """An HTTP 401 with an OCS failure envelope is a failure, not a success."""
    from kiss.agents.third_party_agents.nextcloud_talk_agent import NextcloudTalkAgent

    nc_config.save(
        {"url": api_server.base(), "username": "bot", "password": "nc-bad-invalid"}
    )
    backend = NextcloudTalkChannelBackend()
    assert backend.connect() is False
    assert "HTTP 401" in backend._connection_info
    clear_credentials("nextcloud")
    nc_config.clear()
    agent = NextcloudTalkAgent()
    tools = _auth_tools(agent)
    result = json.loads(
        tools["authenticate_nextcloud"](api_server.base(), "bot", "nc-bad-invalid")
    )
    assert result["ok"] is False
    assert "401" in result["error"]
    assert not vault_has_credentials("nextcloud")


def test_cli_import_mattermost_rejects_non_string_url(muse_env: Path) -> None:
    """JSON ``true``/boolean fields never compose a Mattermost origin."""
    mm_config.path.parent.mkdir(parents=True, exist_ok=True)
    mm_config.path.write_text(json.dumps({"url": True, "token": "tok"}))
    assert muse_cli.main(["import", "mattermost"]) == 1
    assert not vault_has_credentials("mattermost")
    assert json.loads(mm_config.path.read_text())["token"] == "tok"
    mm_config.path.write_text(
        json.dumps({"url": "chat.example.com", "port": True, "token": "tok"})
    )
    assert muse_cli.main(["import", "mattermost"]) == 1
    assert not vault_has_credentials("mattermost")
    mm_config.path.write_text(
        json.dumps({"url": "chat.example.com", "scheme": "ftp", "token": "tok"})
    )
    assert muse_cli.main(["import", "mattermost"]) == 1
    assert not vault_has_credentials("mattermost")


def test_nextcloud_non_json_failure_is_rejected(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """A 5xx with a non-JSON body fails validation instead of crashing."""
    nc_config.save(
        {"url": api_server.base(), "username": "bot", "password": "nc-bad-nonjson"}
    )
    backend = NextcloudTalkChannelBackend()
    assert backend.connect() is False
    assert "HTTP 500" in backend._connection_info
    assert "None" in backend._connection_info  # no OCS statuscode at all


def test_cli_import_mattermost_default_scheme_port_and_bad_port(muse_env: Path) -> None:
    """Absent scheme/port default to https:443; a junk port is rejected."""
    mm_config.path.parent.mkdir(parents=True, exist_ok=True)
    mm_config.path.write_text(json.dumps({"url": "chat.example.com", "token": "tok"}))
    assert muse_cli.main(["import", "mattermost"]) == 0
    assert vault_has_credentials("mattermost")
    # The plaintext token was scrubbed after a successful import.
    assert "token" not in json.loads(mm_config.path.read_text())
    clear_credentials("mattermost")
    mm_config.path.write_text(
        json.dumps({"url": "chat.example.com", "port": "not-a-port", "token": "tok"})
    )
    assert muse_cli.main(["import", "mattermost"]) == 1
    assert not vault_has_credentials("mattermost")


def test_synology_marker_with_empty_vault_falls_back_direct(
    muse_env: Path, api_server: _MessagingApiServer
) -> None:
    """A scrubbed config whose vault entry is gone degrades to direct mode."""
    syno_config.save(
        {"webhook_url": api_server.base("/webapi/entry.cgi?api=X"), "muse": "1"}
    )
    backend = SynologyChatChannelBackend()
    assert backend._wire_muse() is True
    assert backend._muse is False
    assert backend._http is requests

# ------------------------------------------------ review round-2 regressions


def test_query_kind_two_hop_allowlisted_chain_never_regains_credential(
    muse_env: Path, api_server: _MessagingApiServer, rogue_server: Any
) -> None:
    """A -> allowlisted B -> B carries the query credential on NO B request.

    The credential is pinned to the origin of the initially authorized
    request for the whole redirect chain; a same-origin hop *within* the
    foreign origin must not re-qualify it.
    """
    backend = _bb_backend(api_server)
    policy = {
        "defaults": {"read": "allow", "write": "ask"},
        "services": {
            "bluebubbles": {"extra_hosts": [f"127.0.0.1:{rogue_server.port}"]}
        },
    }
    (muse_auth_dir() / "policy.json").write_text(json.dumps(policy))
    api_server.offsite_location = rogue_server.base("/chain-middle")
    session = MuseBoundarySession("bluebubbles")
    resp = session.get(
        api_server.base("/redirect-offsite"),
        headers={"Authorization": f"Bearer {backend._surrogate}"},
    )
    assert resp.status_code == 200
    assert len(rogue_server.requests) == 2
    assert rogue_server.query(index=0).get("password") is None
    assert rogue_server.query(index=1).get("password") is None
    assert rogue_server.header("Authorization", index=0) == ""
    assert rogue_server.header("Authorization", index=1) == ""
    # The initial on-origin hop DID carry the credential.
    assert api_server.query(index=0).get("password") == [_REAL_BB_PASSWORD]


def test_header_kind_allowlisted_cross_origin_redirect_keeps_header(
    muse_env: Path, api_server: _MessagingApiServer, rogue_server: Any
) -> None:
    """Header-kind behavior is unchanged: an allowlisted hop keeps the header."""
    zalo_config.save({"access_token": _REAL_ZALO_TOKEN})
    backend = ZaloChannelBackend()
    assert backend._wire_muse() is True
    policy = {
        "defaults": {"read": "allow", "write": "ask"},
        "services": {
            "zalo": {
                "extra_hosts": [
                    "127.0.0.1",
                    f"127.0.0.1:{rogue_server.port}",
                ]
            }
        },
    }
    (muse_auth_dir() / "policy.json").write_text(json.dumps(policy))
    api_server.offsite_location = rogue_server.base("/v2.0/oa/getoa")
    session = MuseBoundarySession("zalo")
    resp = session.get(
        api_server.base("/redirect-offsite"),
        headers={"Authorization": f"Bearer {backend._access_token}"},
    )
    assert resp.status_code == 200
    assert len(rogue_server.requests) == 1
    # The real Zalo header credential crossed to the allowlisted origin,
    # exactly as before the query-kind pinning fix.
    assert rogue_server.header("access_token") == _REAL_ZALO_TOKEN


def test_cli_import_mattermost_rejects_float_port(muse_env: Path) -> None:
    """JSON float ports are malformed, never silently truncated."""
    mm_config.path.parent.mkdir(parents=True, exist_ok=True)
    for bad_port in (443.9, 443.0):
        mm_config.path.write_text(
            json.dumps({"url": "chat.example.com", "port": bad_port, "token": "tok"})
        )
        assert muse_cli.main(["import", "mattermost"]) == 1
        assert not vault_has_credentials("mattermost")
        assert json.loads(mm_config.path.read_text())["token"] == "tok"
    # A plain JSON integer port stays importable.
    mm_config.path.write_text(
        json.dumps({"url": "chat.example.com", "port": 8065, "token": "tok"})
    )
    assert muse_cli.main(["import", "mattermost"]) == 0
    assert vault_has_credentials("mattermost")
    clear_credentials("mattermost")
