# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Muse-auth on the token-exchange connectors.

SEA style, mirroring ``test_muse_auth_messaging.py``: a REAL Muse-auth
daemon subprocess plus a REAL local HTTP server (stdlib
``ThreadedHTTPServer``) emulating the Azure AD token endpoint, the
Microsoft Graph API, and the Telegram Bot API — no mocks, patches, or
fakes.  The emulated APIs record every request arriving at the
"network" so tests can prove the two NEW mechanisms:

* ``oauth2_client_credentials`` vault entries (MS Teams): the DAEMON
  runs the client-credentials exchange at the boundary and caches the
  acquired Graph token; the agent process holds only a surrogate and
  never sees the client_secret or the Graph token.
* path-kind credentials (Telegram): the agent's request URL embeds the
  surrogate in the token path segment (``/bot<surrogate>/<Method>``)
  and the daemon splices in the real bot token just before each send,
  re-authorizing every redirect hop with pinned-origin re-injection.

Branch-coverage notes (unreachable without test doubles, so documented
instead of mocked):

* The connectors' ``_muse_authenticate`` ``except Exception`` rollback
  branches not driven by an invalid-credential response need the
  daemon to die between the enrollment and the validation call (a
  cross-process race); the invalid-credential rollback IS covered.
* The ``# pragma: no cover`` defense-in-depth branches after
  ``store_credentials`` + ``_wire_muse`` (wire failing although the
  credential was just stored) need the vault cleared between the store
  and the mint by another process.
* The Telegram connector has no SDK dependency: both the Muse branches
  and the legacy ``__init__``/``connect``/``_make_backend``/
  ``authenticate_telegram`` bodies drive the in-house ``_TelegramBot``
  adapter, and the legacy direction is covered flag-off against the
  same emulator (``test_telegram_legacy_tools_without_sdk``; the
  transport itself is exercised in ``test_telegram_legacy_transport.py``).
* ``_get_access_token`` and the legacy MS Teams ``_token`` refresh
  contact the real ``login.microsoftonline.com`` (hardcoded legacy
  URL), so the legacy directions of the ``_token``/``connect``/
  ``authenticate_msteams`` mode branches — which would perform a real
  Azure exchange — are exercised only up to their offline guards here.
* Non-path/query/client-credentials directions of the shared daemon,
  vault, and CLI branches (header/bearer kinds, query CLI imports) are
  exercised by the sibling ``test_muse_auth*.py`` suites.
* Both connectors' ``_muse_authenticate`` now validate the candidate
  against a scratch ``-pending`` enrollment BEFORE mutating the live
  config or vault, so the post-validation swap (config save + vault
  store + ``_wire_muse``) and its ``except``/config-restore rollback
  need ``store_credentials``/``_wire_muse`` to fail AFTER a candidate
  the daemon just accepted — a cross-process race (the daemon dying
  between the probe and the store).  The rejected-candidate path (which
  returns before any live mutation) and the successful swap ARE
  covered; the mid-swap failure branch is documented here instead of
  simulated with a fault-injecting double.
* The daemon's transport-write gate (``daemon._GatedSendMixin``)
  re-checks the pinned credential generation UNDER the vault lock at
  the instant each hop's request head is written to the
  already-connected socket, so a rotation completing during DNS/TCP/
  TLS setup (or during the exchange's network round-trip) aborts the
  request before any credential byte is emitted; the lock is never
  held across a response wait, so a peer-triggered rotation cannot
  deadlock.  ``test_rotation_during_connection_setup_never_emits_...``
  reproduces this with a real backlog-blocked TCP connect (no
  monkeypatching): the rotation lands while the daemon is inside the
  kernel connect, and the accepted socket provably receives zero
  bytes.  The gate's no-generation direction (``generation is None``)
  is unreachable from ``_boundary`` (it always pins a generation) and
  is documented rather than driven through a double.
* ``sweep_stale_pending``'s ``except OSError`` (a scratch file removed
  by another sweeper between the glob and the stat/unlink) is a
  cross-process race documented rather than simulated.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote

import pytest
import requests

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server
from kiss.agents.third_party_agents.msteams_sea import MSTeamsAgent, MSTeamsChannelBackend
from kiss.agents.third_party_agents.msteams_sea import _config as ms_config
from kiss.agents.third_party_agents.msteams_sea import _make_backend as ms_make_backend
from kiss.agents.third_party_agents.muse_auth import __main__ as muse_cli
from kiss.agents.third_party_agents.muse_auth._common import (
    muse_auth_dir,
    request_action,
    socket_path,
    valid_credential_path_value,
    valid_token_endpoint,
)
from kiss.agents.third_party_agents.muse_auth.client import (
    MuseAuthError,
    MuseBoundarySession,
    clear_credentials,
    ensure_daemon,
    grant,
    mint_surrogate,
    stop_daemon,
    store_credentials,
    vault_has_credentials,
)
from kiss.agents.third_party_agents.telegram_sea import TelegramAgent, TelegramChannelBackend
from kiss.agents.third_party_agents.telegram_sea import _config as tg_config
from kiss.agents.third_party_agents.telegram_sea import _make_backend as tg_make_backend
from kiss.tests.agents.third_party_agents.muse_test_utils import (
    auth_tools,
    setup_muse_env,
    teardown_muse_env,
    wait_daemon_stopped,
)

_REAL_TG_TOKEN = "7000000001:AAtelegram-real-secret_x"
_REAL_MS_SECRET = "msteams-real-client-secret"
_MS_TENANT = "11111111-2222-3333-4444-555555555555"
_MS_CLIENT_ID = "app-client-id-1"


class _TokenXApiHandler(BaseHTTPRequestHandler):
    """Emulated Azure AD + Graph + Telegram APIs, recording every request."""

    server: Any

    def _reply(self, status: int, payload: bytes, location: str = "") -> None:
        """Send one JSON (or redirect) response."""
        self.send_response(status)
        if location:
            self.send_header("Location", location)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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
        if path.endswith("/oauth2/v2.0/token"):
            form = {k: v[0] for k, v in parse_qs(body.decode()).items()}
            self.server.token_requests.append(form)
            if self.server.token_redirect_to:
                # A redirecting token endpoint: the daemon must refuse
                # to forward the secret-bearing POST body.
                self._reply(307, b"", location=self.server.token_redirect_to)
                return
            if self.server.token_error_echo:
                # An endpoint reflecting the received secret in a
                # reversible encoding inside its error field.
                encoded = quote(form.get("client_secret", ""), safe="")
                self._reply(400, json.dumps({"error": encoded}).encode())
                return
            if form.get("client_secret") != _REAL_MS_SECRET:
                self._reply(
                    401,
                    json.dumps(
                        {"error": "invalid_client", "error_description": "AADSTS7000215"}
                    ).encode(),
                )
                return
            token = f"graph-tok-{len(self.server.token_requests)}"
            self._reply(
                200,
                json.dumps(
                    {
                        "access_token": token,
                        "token_type": "Bearer",
                        "expires_in": self.server.token_expires_in,
                    }
                ).encode(),
            )
            return
        if "/v1.0/" in path:
            auth = next(
                (v for k, v in self.headers.items() if k.lower() == "authorization"), ""
            )
            if not auth.startswith("Bearer graph-tok-") or self.server.graph_always_401:
                self._reply(
                    401, json.dumps({"error": {"code": "InvalidAuthenticationToken"}}).encode()
                )
                return
            if self.server.graph_nonjson:
                # A proxy-style failure: non-JSON body with a 5xx status.
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", "11")
                self.end_headers()
                self.wfile.write(b"bad gateway")
                return
            if "bad" in path:
                self._reply(400, json.dumps({"error": {"code": "BadRequest"}}).encode())
                return
            if self.command == "POST":
                self._reply(200, json.dumps({"id": "M1"}).encode())
                return
            self._reply(
                200, json.dumps({"value": [{"id": "T1", "displayName": "Team"}]}).encode()
            )
            return
        segments = [s for s in path.split("/") if s]
        if not (segments and segments[0].startswith("bot")):
            self._reply(404, json.dumps({"ok": False, "description": "not found"}).encode())
            return
        token = segments[0][3:]
        api_method = segments[1] if len(segments) > 1 else ""
        if api_method == "redirectEcho":
            # Server-echoed absolute-path Location carrying the REAL
            # token: the boundary must scrub it to the surrogate and
            # re-inject only because the hop stays on the pinned origin.
            self._reply(302, b"", location=f"/bot{_REAL_TG_TOKEN}/getMe")
            return
        if api_method == "redirectRelative":
            self._reply(302, b"", location="getMe")
            return
        if api_method == "redirectOffsite":
            self._reply(302, b"", location=self.server.offsite_location)
            return
        if api_method == "redirectLowercase":
            # Absolute Location echoing the REAL token with a
            # lowercase percent escape for its colon (%3a).
            self._reply(302, b"", location=self.server.lowercase_location)
            return
        if api_method == "redirectOddEncoding":
            # Absolute Location echoing the token with a normally-safe
            # character percent-encoded (A -> %41): no whole-string
            # spelling matches, only a decoded per-segment comparison
            # catches it.
            self._reply(302, b"", location=self.server.odd_location)
            return
        if api_method == "echoBody":
            # An ordinary API error that reflects the request path.
            self._reply(
                404,
                json.dumps(
                    {"ok": False, "description": f"invalid request path {self.path}"}
                ).encode(),
            )
            return
        if api_method == "echoOdd":
            # Reflect the real token with an unreserved char encoded as
            # %41 in the body, a header, and the reason phrase.
            odd = f"/bot{token.replace('A', '%41', 1)}/echoOdd"
            self.send_response(418, f"path {odd} bad")
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Echoed-Path", odd)
            payload = json.dumps({"ok": False, "description": f"bad {odd}"}).encode()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if api_method == "oddChain1":
            # A chain longer than the five-hop limit whose terminal
            # Location echoes the token with a %41-encoded char.
            self._reply(302, b"", location=f"/bot{token}/oddChain2")
            return
        if api_method.startswith("oddChain"):
            hop = int(api_method[8:])
            odd = token.replace("A", "%41", 1)
            target = f"/bot{token}/oddChain{hop + 1}" if hop < 6 else f"/bot{odd}/getMe"
            self._reply(302, b"", location=target)
            return
        if api_method.startswith("chain"):
            # chain1 .. chain6: a redirect chain longer than the
            # daemon's five-hop limit, echoing the real token.
            hop = int(api_method[5:])
            target = f"/bot{token}/chain{hop + 1}" if hop < 6 else f"/bot{token}/getMe"
            self._reply(302, b"", location=target)
            return
        if token != _REAL_TG_TOKEN:
            self._reply(
                401,
                json.dumps(
                    {"ok": False, "error_code": 401, "description": "Unauthorized"}
                ).encode(),
            )
            return
        if api_method == "badJson" or (api_method == "getMe" and self.server.getme_nonjson):
            self._reply(200, b"{not json")
            return
        if api_method == "listJson":
            self._reply(200, b"[]")
            return
        results: dict[str, Any] = {
            "getMe": {"id": 99, "is_bot": True, "username": "kissbot", "first_name": "KISS"},
            "getUpdates": [
                {
                    "update_id": 5,
                    "message": {
                        "message_id": 11,
                        "date": 1700000000,
                        "text": "hello",
                        "chat": {"id": 777},
                        "from": {"id": 42},
                    },
                }
            ],
            "sendMessage": {"message_id": 12},
            "sendPhoto": {"message_id": 13},
            "sendDocument": {"message_id": 13},
            "sendChatAction": True,
            "getChat": {
                "id": 777,
                "title": "Room",
                "type": "group",
                "username": "room",
                "description": "d",
            },
            "getChatMemberCount": 3,
            "getChatMember": {
                "user": {"id": 42, "username": "ann", "first_name": "Ann"},
                "status": "member",
            },
            "sendPoll": {"message_id": 14},
            "forwardMessage": {"message_id": 15},
        }
        result = results.get(api_method, True)
        self._reply(200, json.dumps({"ok": True, "result": result}).encode())

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a GET request."""
        self._serve()

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve a POST request."""
        self._serve()

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence request logging."""


class _TokenXApiServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records requests for verification."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _TokenXApiHandler)
        self.requests: list[dict[str, Any]] = []
        self.token_requests: list[dict[str, str]] = []
        # May be set to a non-numeric value to exercise the vault's
        # malformed-expires_in fallback.
        self.token_expires_in: Any = 3600
        self.offsite_location: str = ""
        self.lowercase_location: str = ""
        self.odd_location: str = ""
        self.graph_nonjson: bool = False
        self.graph_always_401: bool = False
        self.getme_nonjson: bool = False
        self.token_redirect_to: str = ""
        self.token_error_echo: bool = False

    @property
    def port(self) -> int:
        """Return the bound TCP port."""
        return int(self.server_address[1])

    def base(self, suffix: str = "") -> str:
        """Return the server's loopback base URL plus *suffix*."""
        return f"http://127.0.0.1:{self.port}{suffix}"


@pytest.fixture()
def api_server() -> Any:
    """Run the emulated token/Graph/Telegram API on a loopback port."""
    server = _TokenXApiServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    stop_http_server(server, thread)


@pytest.fixture()
def rogue_server() -> Any:
    """Run a second emulator on a different origin for redirect tests."""
    server = _TokenXApiServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    stop_http_server(server, thread)


@pytest.fixture()
def muse_env(isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Enable Muse-auth inside an isolated ``KISS_HOME`` with a live daemon.

    Both services have fixed cloud hosts, so the loopback emulators are
    reached through per-service ``extra_hosts`` policy entries (any
    port — the redirect tests rely on the second emulator's different
    port being a different, but still allowlisted, origin).
    """
    policy = {
        "defaults": {"read": "allow", "write": "ask"},
        "services": {
            # The scratch candidate-validation enrollments
            # (``msteams-pending-<hex>``, ``telegram-pending-<hex>``)
            # inherit these root ``extra_hosts`` via service_root.
            "msteams": {"extra_hosts": ["127.0.0.1"]},
            "telegram": {"extra_hosts": ["127.0.0.1"]},
        },
    }
    setup_muse_env(monkeypatch, policy)
    yield isolated_kiss_home
    teardown_muse_env()


def _ms_env(monkeypatch: pytest.MonkeyPatch, api_server: _TokenXApiServer) -> None:
    """Point the composed Azure token URL at the emulator."""
    monkeypatch.setenv("MSTEAMS_LOGIN_BASE", api_server.base())


def _ms_config(secret: str = _REAL_MS_SECRET) -> None:
    """Save a legacy MS Teams config."""
    ms_config.save(
        {
            "tenant_id": _MS_TENANT,
            "client_id": _MS_CLIENT_ID,
            "client_secret": secret,
            "bot_id": "B1",
        }
    )


def _ms_backend(api_server: _TokenXApiServer) -> MSTeamsChannelBackend:
    """Create an MS Teams backend whose Graph base is the emulator."""
    return MSTeamsChannelBackend(graph_base=api_server.base("/v1.0"))


def _client_credential_info_from_env(api_server: _TokenXApiServer) -> dict[str, str]:
    """Build the client-credentials vault payload pointed at the emulator."""
    from kiss.agents.third_party_agents.msteams_sea import _client_credential_info

    return _client_credential_info(_MS_TENANT, _MS_CLIENT_ID, _REAL_MS_SECRET)


def _tg_backend(api_server: _TokenXApiServer) -> TelegramChannelBackend:
    """Create a Telegram backend pointed at the emulator."""
    backend = TelegramChannelBackend()
    backend._api_base = api_server.base()
    return backend


def _audit_text() -> str:
    """Return the raw Sentinel audit log contents."""
    return (muse_auth_dir() / "audit.jsonl").read_text()


# ---------------------------------------------------------------- MS Teams


def test_msteams_daemon_side_token_exchange_and_scrub(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon exchanges the vaulted secret; the agent holds a surrogate."""
    _ms_env(monkeypatch, api_server)
    _ms_config()
    backend = _ms_backend(api_server)
    assert backend.connect() is True
    assert backend._muse is True
    assert backend._access_token.startswith("muse-sgt.msteams.")
    assert "Muse-auth" in backend._connection_info
    # The exchange happened daemon-side, once, with the real secret.
    assert len(api_server.token_requests) == 1
    exchange = api_server.token_requests[0]
    assert exchange["grant_type"] == "client_credentials"
    assert exchange["client_id"] == _MS_CLIENT_ID
    assert exchange["client_secret"] == _REAL_MS_SECRET
    assert exchange["scope"] == "https://graph.microsoft.com/.default"
    # The Graph call carried the acquired token, not the surrogate.
    graph = [r for r in api_server.requests if "/v1.0/" in r["path"]]
    auth = next(v for k, v in graph[-1]["headers"].items() if k.lower() == "authorization")
    assert auth == "Bearer graph-tok-1"
    # The secret is scrubbed; non-secret metadata survives.
    stored = json.loads(ms_config.path.read_text())
    assert "client_secret" not in stored
    assert stored == {"tenant_id": _MS_TENANT, "client_id": _MS_CLIENT_ID, "bot_id": "B1"}
    assert vault_has_credentials("msteams")
    # Reads flow without grants and reuse the cached token: no second
    # exchange happens for the next Graph call.
    assert backend.list_teams()
    assert len(api_server.token_requests) == 1


def test_msteams_token_cache_expiry_forces_reexchange(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expired cached token is re-acquired (60s skew honored)."""
    _ms_env(monkeypatch, api_server)
    api_server.token_expires_in = 61  # 1s of effective validity after skew
    _ms_config()
    backend = _ms_backend(api_server)
    assert backend.connect() is True
    assert len(api_server.token_requests) == 1
    time.sleep(1.2)
    assert json.loads(backend.list_teams())["ok"] is True
    assert len(api_server.token_requests) == 2
    graph = [r for r in api_server.requests if "/v1.0/" in r["path"]]
    auth = next(v for k, v in graph[-1]["headers"].items() if k.lower() == "authorization")
    assert auth == "Bearer graph-tok-2"


def test_msteams_bad_secret_fails_without_leaking_it(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused exchange surfaces a secret-free resolution error."""
    _ms_env(monkeypatch, api_server)
    bad_secret = "msteams-wrong-secret-value"
    _ms_config(secret=bad_secret)
    backend = _ms_backend(api_server)
    assert backend.connect() is False
    assert "credential resolution failed" in backend._connection_info
    assert "invalid_client" in backend._connection_info
    assert bad_secret not in backend._connection_info
    # The write side needs the exchange too: same secret-free error
    # (granted first, so the failure comes from resolution, not policy).
    grant("msteams", "write", "once")
    posted = json.loads(backend.post_channel_message("T1", "C1", "hi"))
    assert posted["ok"] is False
    assert bad_secret not in posted["error"]


def test_msteams_token_endpoint_down_is_a_safe_error(
    muse_env: Path,
    api_server: _TokenXApiServer,
    refusing_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead token endpoint reports class + URL, never form contents."""
    monkeypatch.setenv("MSTEAMS_LOGIN_BASE", f"http://127.0.0.1:{refusing_port}")
    _ms_config()
    backend = _ms_backend(api_server)
    assert backend.connect() is False
    assert "ConnectionError" in backend._connection_info
    assert _REAL_MS_SECRET not in backend._connection_info


def test_msteams_store_validation_pins_the_token_endpoint(muse_env: Path) -> None:
    """Non-pinned/userinfo token URLs and malformed values are refused."""
    good = {
        "kind": "oauth2_client_credentials",
        "token_url": f"https://login.microsoftonline.com/{_MS_TENANT}/oauth2/v2.0/token",
        "client_id": _MS_CLIENT_ID,
        "client_secret": _REAL_MS_SECRET,
        "token_scope": "https://graph.microsoft.com/.default",
    }
    store_credentials("msteams", good, [])
    clear_credentials("msteams")
    with pytest.raises(MuseAuthError, match="unpinned OAuth token endpoint"):
        store_credentials("msteams", {**good, "token_url": "https://evil.example/token"}, [])
    with pytest.raises(MuseAuthError, match="unpinned OAuth token endpoint"):
        # The pinned host must be https; only loopback may be plain.
        store_credentials(
            "msteams", {**good, "token_url": "http://login.microsoftonline.com/t/token"}, []
        )
    with pytest.raises(MuseAuthError, match="unpinned OAuth token endpoint"):
        store_credentials(
            "msteams",
            {**good, "token_url": f"https://u:p@login.microsoftonline.com/{_MS_TENANT}/t"},
            [],
        )
    with pytest.raises(MuseAuthError, match="invalid client_secret"):
        store_credentials("msteams", {**good, "client_secret": "bad\nsecret"}, [])
    with pytest.raises(MuseAuthError, match="invalid client_id"):
        store_credentials("msteams", {**good, "client_id": ""}, [])
    with pytest.raises(MuseAuthError, match="invalid token_scope"):
        store_credentials("msteams", {**good, "token_scope": "a\tb"}, [])
    assert not vault_has_credentials("msteams")
    # The endpoint validator itself is total over junk input.
    assert valid_token_endpoint("msteams", None) is False
    assert valid_token_endpoint("msteams", "https://login.microsoftonline.com/t/x") is True
    assert valid_token_endpoint("msteams", "http://127.0.0.1:9/t") is True


def test_msteams_wire_rejects_malformed_tenant_before_any_state_change(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A traversal-shaped tenant_id neither enrolls nor scrubs anything."""
    _ms_env(monkeypatch, api_server)
    ms_config.save(
        {
            "tenant_id": "evil/../../tenant",
            "client_id": _MS_CLIENT_ID,
            "client_secret": _REAL_MS_SECRET,
        }
    )
    backend = _ms_backend(api_server)
    assert backend.connect() is False
    assert "invalid tenant_id" in backend._connection_info
    assert not vault_has_credentials("msteams")
    assert json.loads(ms_config.path.read_text())["client_secret"] == _REAL_MS_SECRET


def test_msteams_authenticate_tool_success_and_rollback(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """authenticate_msteams stores secrets vault-only; failures roll back."""
    _ms_env(monkeypatch, api_server)
    agent = MSTeamsAgent.__new__(MSTeamsAgent)
    agent._backend = _ms_backend(api_server)
    tools = auth_tools(agent)
    # Tenant/value validation happens before any state change.
    bad_tenant = tools["authenticate_msteams"]("bad/tenant", "c", "s")
    assert "tenant_id must be" in bad_tenant
    assert not ms_config.path.exists()
    bad_value = json.loads(tools["authenticate_msteams"](_MS_TENANT, "c\nid", "s"))
    assert bad_value["ok"] is False
    # A refused exchange rolls back the vault and restores the config.
    ms_config.save({"tenant_id": "old-tenant", "client_id": "old-client"})
    prev_raw = ms_config.path.read_text()
    refused = json.loads(
        tools["authenticate_msteams"](_MS_TENANT, _MS_CLIENT_ID, "msteams-wrong-secret")
    )
    assert refused["ok"] is False
    assert "msteams-wrong-secret" not in refused["error"]
    assert not vault_has_credentials("msteams")
    assert ms_config.path.read_text() == prev_raw
    assert agent._is_authenticated() is False
    # A good secret enrolls, probes through the boundary, and never
    # writes the secret to config.json.
    saved = json.loads(
        tools["authenticate_msteams"](_MS_TENANT, _MS_CLIENT_ID, _REAL_MS_SECRET, "B1")
    )
    assert saved["ok"] is True
    assert vault_has_credentials("msteams")
    stored = json.loads(ms_config.path.read_text())
    assert "client_secret" not in stored
    assert stored["tenant_id"] == _MS_TENANT
    assert agent._is_authenticated() is True
    checked = json.loads(tools["check_msteams_auth"]())
    assert checked["ok"] is True
    # Clearing removes the vault entry and the wired session.
    assert "cleared" in tools["clear_msteams_auth"]()
    assert not vault_has_credentials("msteams")
    assert agent._backend._muse is False
    assert agent._is_authenticated() is False
    assert "Not authenticated" in tools["check_msteams_auth"]()


def test_msteams_write_needs_grant_and_agent_wires_from_vault(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Graph POSTs classify as writes; a fresh agent wires vault-first."""
    _ms_env(monkeypatch, api_server)
    _ms_config()
    backend = _ms_backend(api_server)
    assert backend.connect() is True
    denied = json.loads(backend.post_channel_message("T1", "C1", "hi"))
    assert denied["ok"] is False
    assert "requires user approval" in denied["error"]
    grant("msteams", "write", "once")
    posted = json.loads(backend.post_channel_message("T1", "C1", "hi"))
    assert posted == {"ok": True, "id": "M1"}
    # A fresh agent (config already scrubbed) wires from the vault.
    agent = MSTeamsAgent.__new__(MSTeamsAgent)
    agent._backend = _ms_backend(api_server)
    assert agent._backend._wire_muse() is True
    assert agent._is_authenticated() is True
    # Poll mode wires the same way; the default-base backend still
    # mints (its Graph base is the real cloud host).
    wired = ms_make_backend()
    assert wired._muse is True
    assert wired._access_token.startswith("muse-sgt.msteams.")


def test_msteams_make_backend_exits_when_unenrolled(muse_env: Path) -> None:
    """Poll mode refuses to start with no vault entry and no config."""
    with pytest.raises(SystemExit):
        ms_make_backend()


def test_msteams_cli_import(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`muse_auth import msteams` migrates and scrubs the client_secret."""
    _ms_env(monkeypatch, api_server)
    assert muse_cli.main(["import", "msteams"]) == 1  # no config yet
    ms_config.save({"tenant_id": _MS_TENANT, "client_id": _MS_CLIENT_ID})
    assert muse_cli.main(["import", "msteams"]) == 1  # no secret
    ms_config.save(
        {"tenant_id": "bad/tenant", "client_id": _MS_CLIENT_ID, "client_secret": "s1"}
    )
    assert muse_cli.main(["import", "msteams"]) == 1  # malformed tenant
    _ms_config()
    assert muse_cli.main(["import", "msteams"]) == 0
    assert vault_has_credentials("msteams")
    stored = json.loads(ms_config.path.read_text())
    assert "client_secret" not in stored
    backend = _ms_backend(api_server)
    assert backend.connect() is True
    assert api_server.token_requests[0]["client_secret"] == _REAL_MS_SECRET


# ---------------------------------------------------------------- Telegram


def test_telegram_path_credential_swap_scrub_and_audit(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """The path surrogate is swapped for the real token only on the wire."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    assert backend._muse is True
    assert "Authenticated as @kissbot" in backend._connection_info
    surrogate = backend._bot.token
    assert surrogate.startswith("muse-sgt.telegram.")
    # The wire saw the REAL token in the path; the config is gone
    # (it held nothing but the token).
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/getMe"
    assert not tg_config.path.exists()
    assert vault_has_credentials("telegram")
    # Reads flow without grants: polling confirms the cursor contract.
    messages, cursor = backend.poll_messages("777", "0")
    assert [m["text"] for m in messages] == ["hello"]
    assert cursor == "6"
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/getUpdates"
    # The typing indicator is a read too (sendChatAction).
    backend.send_typing("777")
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/sendChatAction"
    # Neither the real token nor the surrogate ever reaches the audit
    # log: path credentials are redacted before Sentinel sees the URL.
    audit = _audit_text()
    assert _REAL_TG_TOKEN not in audit
    assert surrogate not in audit
    assert "muse-path-credential" in audit


def test_telegram_write_needs_grant_and_adapter_tools(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """sendMessage asks for a write grant; adapter methods round-trip."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    denied = json.loads(backend.send_text("777", "hi"))
    assert denied["ok"] is False
    assert "requires user approval" in denied["error"]
    grant("telegram", "write", "session")
    assert json.loads(backend.send_text("777", "hi", reply_to_message_id="11")) == {
        "ok": True,
        "message_id": 12,
    }
    sent_body = json.loads(api_server.requests[-1]["body"])
    assert sent_body == {"chat_id": 777, "text": "hi", "reply_to_message_id": 11}
    # send_message (channel-poll seam) threads through the adapter too.
    backend.send_message("777", "tick", thread_ts="11")
    # Reads: chat metadata, membership, updates.
    chat = json.loads(backend.get_chat("777"))
    assert chat == {
        "ok": True,
        "id": 777,
        "title": "Room",
        "type": "group",
        "username": "room",
        "description": "d",
    }
    assert json.loads(backend.get_chat_members_count("777")) == {"ok": True, "count": 3}
    member = json.loads(backend.get_chat_member("777", "42"))
    assert member["username"] == "ann"
    assert member["status"] == "member"
    updates = json.loads(backend.get_updates(offset="5", limit=5))
    assert updates["ok"] is True
    assert updates["updates"][0] == {
        "update_id": 5,
        "chat_id": "777",
        "user_id": "42",
        "text": "hello",
        "message_id": "11",
    }
    # Writes under the session grant: every remaining adapter method.
    assert json.loads(backend.edit_message_text("777", "12", "edited"))["ok"] is True
    assert json.loads(backend.delete_message("777", "12"))["ok"] is True
    assert json.loads(backend.pin_message("777", "12"))["ok"] is True
    assert json.loads(backend.unpin_message("777", "12"))["ok"] is True
    assert json.loads(backend.unpin_message("777"))["ok"] is True
    assert json.loads(backend.ban_chat_member("777", "42"))["ok"] is True
    assert json.loads(backend.unban_chat_member("777", "42"))["ok"] is True
    poll = json.loads(backend.send_poll("777", "q?", '["a", "b"]'))
    assert poll == {"ok": True, "message_id": 14}
    fwd = json.loads(backend.forward_message("888", "777", "11"))
    assert fwd == {"ok": True, "message_id": 15}
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/forwardMessage"


def test_telegram_uploads_flow_through_the_boundary(
    muse_env: Path, api_server: _TokenXApiServer, tmp_path: Path
) -> None:
    """Multipart photo/document uploads carry the real token path."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    grant("telegram", "write", "session")
    assert json.loads(backend.send_photo("777", "https://example.com/p.png", "cap")) == {
        "ok": True,
        "message_id": 13,
    }
    photo = tmp_path / "p.png"
    photo.write_bytes(b"PNG-BYTES")
    assert json.loads(backend.send_photo("777", str(photo)))["ok"] is True
    sent = api_server.requests[-1]
    assert sent["path"] == f"/bot{_REAL_TG_TOKEN}/sendPhoto"
    assert "PNG-BYTES" in sent["body"]
    doc = tmp_path / "d.txt"
    doc.write_text("DOC-CONTENT")
    assert json.loads(backend.send_document("777", str(doc), "cap"))["ok"] is True
    sent = api_server.requests[-1]
    assert sent["path"] == f"/bot{_REAL_TG_TOKEN}/sendDocument"
    assert "DOC-CONTENT" in sent["body"]
    # And without a caption (the field is simply omitted).
    assert json.loads(backend.send_document("777", str(doc)))["ok"] is True
    assert "caption" not in api_server.requests[-1]["body"]


def test_telegram_redirects_scrub_and_pin_the_path_credential(
    muse_env: Path, api_server: _TokenXApiServer, rogue_server: _TokenXApiServer
) -> None:
    """Echoed tokens are scrubbed; only pinned-origin hops re-inject."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    grant("telegram", "write", "session")
    session = MuseBoundarySession("telegram")
    surrogate = backend._bot.token
    headers = {"Authorization": f"Bearer {surrogate}"}
    # Same-origin absolute Location echoing the REAL token: followed,
    # re-injected (still the real token on the wire), one origin only.
    resp = session.request(
        "GET", f"{api_server.base()}/bot{surrogate}/redirectEcho", headers=headers
    )
    assert resp.status_code == 200
    assert json.loads(resp.content)["result"]["username"] == "kissbot"
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/getMe"
    # Same-origin relative Location: resolved against the credential-
    # free URL, then re-injected.
    resp = session.request(
        "GET", f"{api_server.base()}/bot{surrogate}/redirectRelative", headers=headers
    )
    assert resp.status_code == 200
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/getMe"
    # Cross-origin hop to a separately allowlisted origin (same host,
    # different port): followed WITHOUT the credential — the rogue
    # origin sees a redacted path, never the token or the surrogate.
    api_server.offsite_location = f"{rogue_server.base()}/bot{_REAL_TG_TOKEN}/getMe"
    resp = session.request(
        "GET", f"{api_server.base()}/bot{surrogate}/redirectOffsite", headers=headers
    )
    assert resp.status_code == 401  # rogue emulator rejects the redacted token
    assert rogue_server.requests[-1]["path"] == "/botmuse-redacted/getMe"
    rogue_paths = "".join(r["path"] for r in rogue_server.requests)
    assert _REAL_TG_TOKEN not in rogue_paths
    assert surrogate not in rogue_paths
    # The audit trail for all hops stays capability-free.
    audit = _audit_text()
    assert _REAL_TG_TOKEN not in audit
    assert surrogate not in audit


def test_telegram_boundary_requires_the_surrogate_in_the_path(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """A path-credential request whose URL lacks the surrogate is refused."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    surrogate = backend._bot.token
    session = MuseBoundarySession("telegram")
    with pytest.raises(MuseAuthError, match="does not reference the surrogate"):
        session.request(
            "GET",
            f"{api_server.base()}/botSOMETHINGELSE/getMe",
            headers={"Authorization": f"Bearer {surrogate}"},
        )
    # A surrogate only in the query is not a path reference either.
    with pytest.raises(MuseAuthError, match="does not reference the surrogate"):
        session.request(
            "GET",
            f"{api_server.base()}/bot/getMe?x={surrogate}",
            headers={"Authorization": f"Bearer {surrogate}"},
        )


def test_telegram_rotation_invalidates_path_surrogates(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """Re-storing the credential kills surrogates minted before it."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    store_credentials("telegram", {"kind": "path", "token": "7:rotated-token"}, [])
    with pytest.raises(RuntimeError, match="stale surrogate"):
        backend._bot.get_me()
    # Re-wiring from the vault mints against the new generation: the
    # rotated token reaches the wire (the emulator rejects it with 401,
    # which the adapter surfaces as an API error, proving the swap).
    assert backend._wire_muse() is True
    with pytest.raises(RuntimeError, match="HTTP 401"):
        backend._bot.get_me()
    assert api_server.requests[-1]["path"] == "/bot7:rotated-token/getMe"


def test_telegram_store_validation_rejects_unsafe_path_tokens(muse_env: Path) -> None:
    """Path-splice-unsafe tokens are refused at enrollment."""
    for bad in ("a/b", "a b", "a%3Ab", "", "x" * 257):
        with pytest.raises(MuseAuthError, match="invalid path credential token value"):
            store_credentials("telegram", {"kind": "path", "token": bad}, [])
    assert not vault_has_credentials("telegram")
    assert valid_credential_path_value(_REAL_TG_TOKEN) is True


def test_telegram_authenticate_tool_success_and_rollback(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """authenticate_telegram validates via the boundary and rolls back."""
    agent = TelegramAgent.__new__(TelegramAgent)
    agent._backend = _tg_backend(api_server)
    tools = auth_tools(agent)
    unsafe = json.loads(tools["authenticate_telegram"]("bad token"))
    assert unsafe["ok"] is False
    assert "URL-path-safe" in unsafe["error"]
    # A token the API rejects rolls the vault back and restores the
    # pre-call config bytes.
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    prev_raw = tg_config.path.read_text()
    rejected = json.loads(tools["authenticate_telegram"]("1:wrong-token"))
    assert rejected["ok"] is False
    assert "Unauthorized" in rejected["error"]
    assert not vault_has_credentials("telegram")
    assert tg_config.path.read_text() == prev_raw
    assert agent._is_authenticated() is False
    # A valid token enrolls vault-only and validates through getMe.
    saved = json.loads(tools["authenticate_telegram"](_REAL_TG_TOKEN))
    assert saved == {
        "ok": True,
        "message": "Telegram token saved and validated (Muse-auth).",
        "username": "kissbot",
        "id": 99,
    }
    assert vault_has_credentials("telegram")
    assert not tg_config.path.exists()
    assert agent._is_authenticated() is True
    checked = json.loads(tools["check_telegram_auth"]())
    assert checked["ok"] is True
    assert checked["username"] == "kissbot"
    assert "cleared" in tools["clear_telegram_auth"]()
    assert not vault_has_credentials("telegram")
    assert agent._backend._muse is False
    assert "Not authenticated" in tools["check_telegram_auth"]()


def test_telegram_agent_and_make_backend_wire_from_vault(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """Agent construction and poll mode wire vault-first without the SDK."""
    store_credentials("telegram", {"kind": "path", "token": _REAL_TG_TOKEN}, [])
    agent = TelegramAgent.__new__(TelegramAgent)
    agent._backend = TelegramChannelBackend()
    assert agent._backend._wire_muse() is True
    assert agent._is_authenticated() is True
    wired = tg_make_backend()
    assert wired._muse is True
    wired._api_base = api_server.base()
    messages, cursor = wired.poll_messages("", "0")
    assert messages and cursor == "6"


def test_telegram_make_backend_exits_when_unenrolled(muse_env: Path) -> None:
    """Poll mode refuses to start with no vault entry and no config."""
    with pytest.raises(SystemExit):
        tg_make_backend()


def test_telegram_cli_import(muse_env: Path, api_server: _TokenXApiServer) -> None:
    """`muse_auth import telegram` migrates the token and deletes the file."""
    assert muse_cli.main(["import", "telegram"]) == 1  # no config yet
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    assert muse_cli.main(["import", "telegram"]) == 0
    assert vault_has_credentials("telegram")
    assert not tg_config.path.exists()
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/getMe"


def test_telegram_action_classification() -> None:
    """GET sendMessage is a write; the file namespace is a read."""
    assert request_action("telegram", "GET", "/botX/sendMessage") == "write"
    assert request_action("telegram", "POST", "/botX/getUpdates") == "read"
    assert request_action("telegram", "POST", "/botX/sendChatAction") == "read"
    assert request_action("telegram", "GET", "/file/botX/documents/f.txt") == "read"
    assert request_action("telegram", "GET", "/botX/GetMe/") == "read"
    assert request_action("telegram", "GET", "") == "write"


def test_telegram_legacy_direct_path_unchanged(
    isolated_kiss_home: Path, api_server: _TokenXApiServer
) -> None:
    """Flag off: polling/typing hit the API directly with the URL token."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend._muse is False
    assert backend._http is requests
    messages, cursor = backend.poll_messages("777", "0")
    assert [m["text"] for m in messages] == ["hello"]
    assert cursor == "6"
    backend.send_typing("777")
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/sendChatAction"
    # No daemon socket appeared and the config kept its token.
    assert not socket_path().exists()
    assert json.loads(tg_config.path.read_text())["bot_token"] == _REAL_TG_TOKEN


def test_msteams_probe_policy_denial_and_nonjson_graph(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe distinguishes policy denials from proof of exchange."""
    _ms_env(monkeypatch, api_server)
    _ms_config()
    backend = _ms_backend(api_server)
    assert backend.connect() is True
    agent = MSTeamsAgent.__new__(MSTeamsAgent)
    agent._backend = backend
    tools = auth_tools(agent)
    # Sentinel reloads policy per decision: flip Graph reads to "ask".
    policy_path = muse_auth_dir() / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy["services"]["msteams"]["read"] = "ask"
    policy_path.write_text(json.dumps(policy))
    checked = json.loads(tools["check_msteams_auth"]())
    assert checked["ok"] is False
    assert "requires user approval" in checked["error"]
    assert backend.connect() is False
    assert "MS Teams auth failed" in backend._connection_info
    policy["services"]["msteams"].pop("read")
    policy_path.write_text(json.dumps(policy))
    # A non-JSON Graph response still proves the exchange succeeded.
    api_server.graph_nonjson = True
    ok, message = backend._muse_probe()
    assert ok is True and "Muse-auth" in message


def test_msteams_graph_http_errors_surface_in_tools(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Graph 4xx envelopes surface as ok=False from the posting tools."""
    _ms_env(monkeypatch, api_server)
    _ms_config()
    backend = _ms_backend(api_server)
    assert backend.connect() is True
    grant("msteams", "write", "session")
    bad = json.loads(backend.post_channel_message("T1", "bad1", "hi"))
    assert bad["ok"] is False and "BadRequest" in bad["error"]
    assert json.loads(backend.reply_to_message("T1", "C1", "M1", "re")) == {
        "ok": True,
        "id": "M1",
    }
    bad = json.loads(backend.reply_to_message("T1", "bad1", "M1", "re"))
    assert bad["ok"] is False and "BadRequest" in bad["error"]
    assert json.loads(backend.post_chat_message("chat1", "hi")) == {"ok": True, "id": "M1"}
    bad = json.loads(backend.post_chat_message("chatbad", "hi"))
    assert bad["ok"] is False and "BadRequest" in bad["error"]
    # GET-side 4xx envelopes get the same ok=False marking.
    listed = json.loads(backend.list_channel_messages("badT", "C1"))
    assert listed == {"ok": True, "messages": []}  # tolerant reader
    assert backend._get("/teams/badT")["ok"] is False


def test_msteams_scrub_edge_cases_and_legacy_paths(
    isolated_kiss_home: Path, api_server: _TokenXApiServer
) -> None:
    """The secret scrub tolerates odd configs; legacy paths stay direct."""
    from kiss.agents.third_party_agents.msteams_sea import _scrub_config_secret

    _scrub_config_secret()  # no config file: a no-op
    ms_config.save({"tenant_id": _MS_TENANT, "client_id": _MS_CLIENT_ID})
    before = ms_config.path.read_text()
    _scrub_config_secret()  # no client_secret key: untouched
    assert ms_config.path.read_text() == before
    ms_config.path.write_text(json.dumps({"client_secret": "only-secret"}))
    _scrub_config_secret()  # nothing but the secret: file removed
    assert not ms_config.path.exists()
    # Legacy poll mode builds the direct backend without any daemon.
    _ms_config()
    backend = ms_make_backend()
    assert backend._muse is False
    assert backend._http is requests
    assert backend._client_secret == _REAL_MS_SECRET
    # Legacy agent construction and a config-less connect stay offline.
    legacy_agent = MSTeamsAgent()
    assert legacy_agent._backend._client_secret == _REAL_MS_SECRET
    ms_config.clear()
    fresh = MSTeamsChannelBackend()
    assert fresh.connect() is False
    assert "No MS Teams config found" in fresh._connection_info
    # The legacy clear tool needs no daemon either.
    agent = MSTeamsAgent.__new__(MSTeamsAgent)
    agent._backend = backend
    assert "cleared" in auth_tools(agent)["clear_msteams_auth"]()
    assert not ms_config.path.exists()
    assert not socket_path().exists()


def test_msteams_authenticate_unpinned_login_base_rolls_back(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store-time rejection (unpinned endpoint) restores a clean state."""
    monkeypatch.setenv("MSTEAMS_LOGIN_BASE", "https://attacker.example")
    agent = MSTeamsAgent.__new__(MSTeamsAgent)
    agent._backend = _ms_backend(api_server)
    result = json.loads(
        auth_tools(agent)["authenticate_msteams"](_MS_TENANT, _MS_CLIENT_ID, _REAL_MS_SECRET)
    )
    assert result["ok"] is False
    assert "unpinned OAuth token endpoint" in result["error"]
    assert not vault_has_credentials("msteams")
    # There was no pre-call config, so none survives the rollback.
    assert not ms_config.path.exists()


def test_agents_construct_and_fail_closed(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full agent construction wires vault-first and fails closed."""
    _ms_env(monkeypatch, api_server)
    # Happy path: enrolled vault, no config.
    store_credentials("telegram", {"kind": "path", "token": _REAL_TG_TOKEN}, [])
    tg_agent = TelegramAgent()
    assert tg_agent._backend._muse is True
    assert tg_agent._is_authenticated() is True
    _ms_config()
    ms_agent = MSTeamsAgent()
    assert ms_agent._backend._muse is True
    assert ms_agent._is_authenticated() is True
    # A credential the daemon rejects at enrollment (embedded newline)
    # fails closed but keeps the agents constructible.
    clear_credentials("telegram")
    clear_credentials("msteams")
    tg_config.save({"bot_token": "bad\ntoken"})
    tg_agent = TelegramAgent()
    assert tg_agent._backend._bot is None
    assert "Muse-auth wiring failed" in tg_agent._backend._connection_info
    assert not vault_has_credentials("telegram")
    ms_config.save(
        {"tenant_id": _MS_TENANT, "client_id": _MS_CLIENT_ID, "client_secret": "bad\nsecret"}
    )
    ms_agent = MSTeamsAgent()
    assert ms_agent._backend._access_token == ""
    assert "Muse-auth wiring failed" in ms_agent._backend._connection_info
    assert not vault_has_credentials("msteams")


def test_telegram_adapter_and_scrub_edges(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """Non-JSON/non-dict envelopes raise; scrub keeps non-secret keys."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN, "note": "keep-me"})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    # The scrub kept the non-secret key while removing the token.
    assert json.loads(tg_config.path.read_text()) == {"note": "keep-me"}
    grant("telegram", "write", "session")
    with pytest.raises(RuntimeError, match="badJson failed"):
        backend._bot._call("badJson")
    with pytest.raises(RuntimeError, match="listJson failed"):
        backend._bot._call("listJson")
    # get_updates without an offset omits the parameter entirely.
    assert json.loads(backend.get_updates())["ok"] is True
    assert "offset" not in json.loads(api_server.requests[-1]["body"])
    # Caption permutations for both upload methods.
    assert json.loads(backend.send_photo("777", "https://example.com/p.png"))["ok"] is True
    doc = json.loads(tg_config.path.read_text())  # config survives ops
    assert doc == {"note": "keep-me"}
    # The scrub itself tolerates odd configs.
    from kiss.agents.third_party_agents.telegram_sea import _scrub_config_token

    tg_config.path.write_text("not json")
    _scrub_config_token()  # unreadable: a no-op
    assert tg_config.path.read_text() == "not json"
    tg_config.path.write_text(json.dumps({"note": "n"}))
    _scrub_config_token()  # no bot_token key: untouched
    assert json.loads(tg_config.path.read_text()) == {"note": "n"}
    tg_config.path.unlink()
    _scrub_config_token()  # no file: a no-op


def test_telegram_connect_failure_paths(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """connect() fails closed on missing credentials and rejected tokens."""
    backend = _tg_backend(api_server)
    assert backend.connect() is False
    assert "No Telegram credential" in backend._connection_info
    store_credentials("telegram", {"kind": "path", "token": "1:wrong-token"}, [])
    backend = _tg_backend(api_server)
    assert backend.connect() is False
    assert "Telegram auth failed" in backend._connection_info
    assert "Unauthorized" in backend._connection_info


def test_telegram_authenticate_without_precall_config(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """A rejected token with no prior config leaves no config behind."""
    agent = TelegramAgent.__new__(TelegramAgent)
    agent._backend = _tg_backend(api_server)
    rejected = json.loads(auth_tools(agent)["authenticate_telegram"]("1:wrong-token"))
    assert rejected["ok"] is False
    assert not vault_has_credentials("telegram")
    assert not tg_config.path.exists()


def test_telegram_legacy_tools_without_sdk(
    isolated_kiss_home: Path, api_server: _TokenXApiServer
) -> None:
    """Flag off: legacy tools talk to the Bot API directly, no SDK, no daemon.

    The real token travels in the URL path segment and no bearer header
    is sent (that header only identifies a Muse surrogate).
    """
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    assert backend._connection_info == "Authenticated as @kissbot"
    assert backend._muse is False
    probe = api_server.requests[-1]
    assert probe["path"] == f"/bot{_REAL_TG_TOKEN}/getMe"
    assert not any(k.lower() == "authorization" for k in probe["headers"])
    # Legacy agent construction wires the bot from the persisted config.
    agent = TelegramAgent()
    assert agent._backend._bot is not None
    agent._backend = backend
    tools = auth_tools(agent)
    status = json.loads(tools["check_telegram_auth"]())
    assert status == {"ok": True, "username": "kissbot", "first_name": "KISS", "id": 99}
    rejected = json.loads(tools["authenticate_telegram"]("wrong-token"))
    assert rejected["ok"] is False and "401" in rejected["error"]
    assert tg_config.load() == {"bot_token": _REAL_TG_TOKEN}  # config untouched
    accepted = json.loads(tools["authenticate_telegram"](_REAL_TG_TOKEN))
    assert accepted["ok"] is True and accepted["username"] == "kissbot"
    polled = tg_make_backend()
    assert polled._muse is False
    polled._api_base = api_server.base()
    messages, cursor = polled.poll_messages("", "0")
    assert messages and cursor == "6"
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/getUpdates"
    assert "cleared" in tools["clear_telegram_auth"]()
    assert not tg_config.path.exists()
    assert agent._backend._bot is None
    with pytest.raises(SystemExit):
        tg_make_backend()
    assert not socket_path().exists()


def test_daemon_refuses_a_tampered_unsafe_path_credential(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """Defense in depth: a vault file edited to an unsafe value is refused."""
    store_credentials("telegram", {"kind": "path", "token": _REAL_TG_TOKEN}, [])
    handle = mint_surrogate("telegram")
    assert handle is not None
    vault_file = muse_auth_dir() / "vault" / "telegram.json"
    payload = json.loads(vault_file.read_text())
    payload["authorized_user_info"]["token"] = "un/safe"
    vault_file.write_text(json.dumps(payload))
    session = MuseBoundarySession("telegram")
    with pytest.raises(MuseAuthError, match="not path-splice-safe"):
        session.request(
            "GET",
            f"{api_server.base()}/bot{handle.token}/getMe",
            headers={"Authorization": f"Bearer {handle.token}"},
        )


def test_transport_failures_redact_url_credentials(
    muse_env: Path, refusing_port: int
) -> None:
    """Refused connections report class + credential-free URL only."""
    port = refusing_port
    # Path placement: the message shows the redacted path form.
    store_credentials("telegram", {"kind": "path", "token": _REAL_TG_TOKEN}, [])
    handle = mint_surrogate("telegram")
    assert handle is not None
    session = MuseBoundarySession("telegram")
    with pytest.raises(MuseAuthError) as err:
        session.request(
            "GET",
            f"http://127.0.0.1:{port}/bot{handle.token}/getMe",
            headers={"Authorization": f"Bearer {handle.token}"},
        )
    message = str(err.value)
    assert "ConnectionError" in message
    assert "muse-path-credential" in message
    assert _REAL_TG_TOKEN not in message
    assert handle.token not in message
    # Query placement: the message shows the param-stripped URL.
    store_credentials(
        "synology",
        {"kind": "query", "param": "token", "token": "syno-secret-q"},
        [],
        hosts=(f"127.0.0.1:{port}",),
    )
    q_handle = mint_surrogate("synology")
    assert q_handle is not None
    q_session = MuseBoundarySession("synology")
    with pytest.raises(MuseAuthError) as q_err:
        q_session.request(
            "GET",
            f"http://127.0.0.1:{port}/webapi/entry.cgi?token=junk",
            headers={"Authorization": f"Bearer {q_handle.token}"},
        )
    q_message = str(q_err.value)
    assert "ConnectionError" in q_message
    assert "syno-secret-q" not in q_message and "token=" not in q_message


def test_offsite_redirect_to_unallowlisted_host_is_audited_and_redacted(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """A hop to a host off every allowlist follows bodyless and redacted."""
    rogue = _TokenXApiServer(("127.0.0.2", 0))
    thread = threading.Thread(target=rogue.serve_forever, daemon=True)
    thread.start()
    try:
        tg_config.save({"bot_token": _REAL_TG_TOKEN})
        backend = _tg_backend(api_server)
        assert backend.connect() is True
        grant("telegram", "write", "session")
        surrogate = backend._bot.token
        # 127.0.0.2 is NOT in the telegram allowlist (only 127.0.0.1 is).
        api_server.offsite_location = (
            f"http://127.0.0.2:{rogue.port}/bot{_REAL_TG_TOKEN}/getMe"
        )
        resp = MuseBoundarySession("telegram").request(
            "GET",
            f"{api_server.base()}/bot{surrogate}/redirectOffsite",
            headers={"Authorization": f"Bearer {surrogate}"},
        )
        # The bodyless GET hop is followed without any credential.
        assert resp.status_code == 401
        assert rogue.requests[-1]["path"] == "/botmuse-redacted/getMe"
        audit = _audit_text()
        assert "127.0.0.2" in audit
        assert _REAL_TG_TOKEN not in audit and surrogate not in audit
    finally:
        stop_http_server(rogue, thread)


def test_vault_client_credentials_cache_edges(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed cache/expiry values fall back safely; scope is optional."""
    _ms_env(monkeypatch, api_server)
    # A scope-less enrollment sends no scope field to the endpoint.
    info = {
        "kind": "oauth2_client_credentials",
        "token_url": f"{api_server.base()}/{_MS_TENANT}/oauth2/v2.0/token",
        "client_id": _MS_CLIENT_ID,
        "client_secret": _REAL_MS_SECRET,
    }
    store_credentials("msteams", info, [])
    backend = _ms_backend(api_server)
    assert backend._wire_muse() is True
    assert json.loads(backend.list_teams())["ok"] is True
    assert "scope" not in api_server.token_requests[-1]
    # A tampered (non-numeric) cached expiry forces a clean re-exchange.
    vault_file = muse_auth_dir() / "vault" / "msteams.json"
    payload = json.loads(vault_file.read_text())
    payload["cached_token"]["expires_at"] = "bogus"
    vault_file.write_text(json.dumps(payload))
    exchanges = len(api_server.token_requests)
    assert json.loads(backend.list_teams())["ok"] is True
    assert len(api_server.token_requests) == exchanges + 1
    # A malformed expires_in from the endpoint defaults to one hour.
    api_server.token_expires_in = "abc"
    payload = json.loads(vault_file.read_text())
    payload.pop("cached_token", None)
    vault_file.write_text(json.dumps(payload))
    assert json.loads(backend.list_teams())["ok"] is True
    cached = json.loads(vault_file.read_text())["cached_token"]
    assert cached["expires_at"] > time.time() + 3000


# --------------------------------------------- review round-1 regressions


def test_token_exchange_refuses_redirecting_endpoint(
    muse_env: Path, api_server: _TokenXApiServer, rogue_server: _TokenXApiServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 307 from the token endpoint never forwards the secret-bearing POST."""
    _ms_env(monkeypatch, api_server)
    api_server.token_redirect_to = rogue_server.base("/stolen/oauth2/v2.0/token")
    _ms_config()
    backend = _ms_backend(api_server)
    assert backend.connect() is False
    assert "answered with a redirect" in backend._connection_info
    assert _REAL_MS_SECRET not in backend._connection_info
    # The redirect target never saw any request at all.
    assert rogue_server.requests == []
    assert rogue_server.token_requests == []


def test_token_exchange_error_never_relays_endpoint_text(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reflected (encoded) secret text in an error field is suppressed."""
    _ms_env(monkeypatch, api_server)
    api_server.token_error_echo = True
    _ms_config()
    backend = _ms_backend(api_server)
    assert backend.connect() is False
    message = backend._connection_info
    assert "HTTP 400" in message
    assert _REAL_MS_SECRET not in message
    assert quote(_REAL_MS_SECRET, safe="") not in message


def test_token_exchange_rejects_nonfinite_expiry(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-finite expires_in must not create an immortal cache entry."""
    _ms_env(monkeypatch, api_server)
    api_server.token_expires_in = "Infinity"
    _ms_config()
    backend = _ms_backend(api_server)
    assert backend.connect() is True
    cached = json.loads((muse_auth_dir() / "vault" / "msteams.json").read_text())[
        "cached_token"
    ]
    expires_at = float(cached["expires_at"])
    assert expires_at < time.time() + 3700  # finite, bounded fallback


def test_msteams_probe_rejects_graph_401(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Graph 401 (unusable token) is not reported as authenticated."""
    _ms_env(monkeypatch, api_server)
    api_server.graph_always_401 = True
    _ms_config()
    backend = _ms_backend(api_server)
    assert backend.connect() is False
    assert "HTTP 401" in backend._connection_info


def test_failed_rotations_keep_the_prior_credential(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected replacement leaves the working enrollment untouched."""
    _ms_env(monkeypatch, api_server)
    # Telegram: enroll a good token, then attempt a bad rotation.
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    tg_backend = _tg_backend(api_server)
    assert tg_backend.connect() is True
    tg_agent = TelegramAgent.__new__(TelegramAgent)
    tg_agent._backend = tg_backend
    rejected = json.loads(
        auth_tools(tg_agent)["authenticate_telegram"]("1:rotated-but-wrong")
    )
    assert rejected["ok"] is False
    assert vault_has_credentials("telegram")
    assert not vault_has_credentials("telegram-pending")
    # The previously wired backend still works end to end.
    assert json.loads(auth_tools(tg_agent)["check_telegram_auth"]())["ok"] is True
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/getMe"
    # MS Teams: same contract.
    _ms_config()
    ms_backend = _ms_backend(api_server)
    assert ms_backend.connect() is True
    ms_agent = MSTeamsAgent.__new__(MSTeamsAgent)
    ms_agent._backend = ms_backend
    refused = json.loads(
        auth_tools(ms_agent)["authenticate_msteams"](
            _MS_TENANT, _MS_CLIENT_ID, "msteams-rotated-wrong"
        )
    )
    assert refused["ok"] is False
    assert vault_has_credentials("msteams")
    assert not vault_has_credentials("msteams-pending")
    assert json.loads(ms_backend.list_teams())["ok"] is True


def test_lowercase_percent_escapes_cannot_smuggle_the_token(
    muse_env: Path, api_server: _TokenXApiServer, rogue_server: _TokenXApiServer
) -> None:
    """A %3a-encoded echoed token is normalized before any cross-origin hop."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    grant("telegram", "write", "session")
    surrogate = backend._bot.token
    encoded = _REAL_TG_TOKEN.replace(":", "%3a")
    api_server.lowercase_location = f"{rogue_server.base()}/bot{encoded}/getMe"
    resp = MuseBoundarySession("telegram").request(
        "GET",
        f"{api_server.base()}/bot{surrogate}/redirectLowercase",
        headers={"Authorization": f"Bearer {surrogate}"},
    )
    assert resp.status_code == 401  # rogue origin got no usable credential
    rogue_paths = "".join(r["path"] for r in rogue_server.requests)
    assert "muse-redacted" in rogue_paths
    for spelling in (_REAL_TG_TOKEN, encoded, _REAL_TG_TOKEN.replace(":", "%3A")):
        assert spelling not in rogue_paths
    audit = _audit_text()
    for spelling in (_REAL_TG_TOKEN, encoded, _REAL_TG_TOKEN.replace(":", "%3A")):
        assert spelling not in audit


def test_odd_percent_encoding_is_scrubbed_per_segment(
    muse_env: Path, api_server: _TokenXApiServer, rogue_server: _TokenXApiServer
) -> None:
    """A token spelled with a non-standard escape is caught by segment decode."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    grant("telegram", "write", "session")
    surrogate = backend._bot.token
    # Encode a normally-safe 'A' as %41 so no whole-string spelling of
    # the token matches; only decoding each path segment reveals it.
    odd = _REAL_TG_TOKEN.replace("A", "%41", 1)
    api_server.odd_location = f"{rogue_server.base()}/bot{odd}/getMe"
    resp = MuseBoundarySession("telegram").request(
        "GET",
        f"{api_server.base()}/bot{surrogate}/redirectOddEncoding",
        headers={"Authorization": f"Bearer {surrogate}"},
    )
    assert resp.status_code == 401
    rogue_paths = "".join(r["path"] for r in rogue_server.requests)
    assert _REAL_TG_TOKEN not in rogue_paths
    # The daemon's decoded target no longer contains the real token.
    assert "muse-redacted" in rogue_paths
    audit = _audit_text()
    assert _REAL_TG_TOKEN not in audit


def test_reply_echoes_are_normalized_to_the_surrogate(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """Echoed request paths in bodies/Locations never reach the agent raw."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    grant("telegram", "write", "session")
    surrogate = backend._bot.token
    session = MuseBoundarySession("telegram")
    headers = {"Authorization": f"Bearer {surrogate}"}
    # An ordinary API error echoing the credentialed request path.
    resp = session.request(
        "GET", f"{api_server.base()}/bot{surrogate}/echoBody", headers=headers
    )
    assert resp.status_code == 404
    assert _REAL_TG_TOKEN not in resp.text
    assert quote(_REAL_TG_TOKEN, safe="") not in resp.text
    assert surrogate in resp.text  # the echo is normalized, not dropped
    # A chain longer than the redirect limit: the returned (still
    # redirecting) response's Location is normalized too.
    resp = session.request(
        "GET", f"{api_server.base()}/bot{surrogate}/chain1", headers=headers
    )
    assert resp.status_code == 302
    location = resp.headers.get("Location", "")
    assert _REAL_TG_TOKEN not in location
    assert surrogate in location


def test_telegram_authenticate_handles_nonjson_probe(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """A non-JSON getMe during candidate validation is a clean failure."""
    api_server.getme_nonjson = True
    agent = TelegramAgent.__new__(TelegramAgent)
    agent._backend = _tg_backend(api_server)
    result = json.loads(auth_tools(agent)["authenticate_telegram"](_REAL_TG_TOKEN))
    assert result["ok"] is False
    assert "getMe failed" in result["error"]
    # Nothing was enrolled and no scratch entry lingered.
    assert not vault_has_credentials("telegram")
    assert not vault_has_credentials("telegram-pending")


def test_cli_import_msteams_rejects_non_string_values(muse_env: Path) -> None:
    """JSON booleans are a malformed config, not importable credentials."""
    ms_config.path.parent.mkdir(parents=True, exist_ok=True)
    ms_config.path.write_text(
        json.dumps({"tenant_id": True, "client_id": True, "client_secret": True})
    )
    before = ms_config.path.read_text()
    assert muse_cli.main(["import", "msteams"]) == 1
    assert ms_config.path.read_text() == before  # nothing scrubbed
    assert not vault_has_credentials("msteams")
    # The generic token importers are type-strict too.
    tg_config.path.parent.mkdir(parents=True, exist_ok=True)
    tg_config.path.write_text(json.dumps({"bot_token": True}))
    assert muse_cli.main(["import", "telegram"]) == 1
    assert not vault_has_credentials("telegram")


def test_flag_off_msteams_matches_head_semantics(
    isolated_kiss_home: Path, api_server: _TokenXApiServer
) -> None:
    """Legacy responses carry no injected ok-marking and auth is client_id-only."""
    backend = _ms_backend(api_server)
    # Prime a token client-side so no real Azure exchange happens.
    backend._access_token = "graph-tok-legacy"
    backend._token_expiry = time.time() + 3600
    result = backend._get("/teams/badT")  # emulator answers HTTP 400
    assert result == {"error": {"code": "BadRequest"}}  # no ok=False injected
    posted = backend._post("/teams/badT/channels/C1/messages", {"body": {}})
    assert posted == {"error": {"code": "BadRequest"}}
    agent = MSTeamsAgent.__new__(MSTeamsAgent)
    agent._backend = backend
    assert agent._is_authenticated() is False  # token alone is not auth
    backend._client_id = _MS_CLIENT_ID
    assert agent._is_authenticated() is True
    assert not socket_path().exists()


# --------------------------------------------- review round-2 regressions


def test_auto_migration_never_clobbers_a_working_credential(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad config token during auto-migration must not destroy the vault."""
    # Telegram: enroll a good token, then simulate a stale/bad config
    # value appearing (a rotation typed into config while Muse was off).
    store_credentials("telegram", {"kind": "path", "token": _REAL_TG_TOKEN}, [])
    tg_config.save({"bot_token": "9:stale-bad-rotation"})
    backend = _tg_backend(api_server)
    assert backend.connect() is True  # the GOOD vault token still works
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/getMe"
    stored = json.loads((muse_auth_dir() / "vault" / "telegram.json").read_text())
    assert stored["authorized_user_info"]["token"] == _REAL_TG_TOKEN
    # MS Teams: same contract with a stale config secret.
    _ms_env(monkeypatch, api_server)
    store_credentials(
        "msteams",
        _client_credential_info_from_env(api_server),
        [],
    )
    ms_config.save(
        {"tenant_id": _MS_TENANT, "client_id": _MS_CLIENT_ID, "client_secret": "stale-bad"}
    )
    ms_backend = _ms_backend(api_server)
    assert ms_backend.connect() is True
    assert api_server.token_requests[-1]["client_secret"] == _REAL_MS_SECRET


def test_first_migration_still_seeds_the_vault(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """With an empty vault, a config token seeds and is scrubbed."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    assert vault_has_credentials("telegram")
    assert not tg_config.path.exists()


def test_auto_migration_rejects_non_string_config_values(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JSON boolean in config is refused, not coerced, and never scrubbed."""
    _ms_env(monkeypatch, api_server)
    ms_config.path.parent.mkdir(parents=True, exist_ok=True)
    ms_config.path.write_text(
        json.dumps({"tenant_id": True, "client_id": _MS_CLIENT_ID, "client_secret": "s"})
    )
    before = ms_config.path.read_text()
    backend = _ms_backend(api_server)
    assert backend.connect() is False
    assert "non-string credentials" in backend._connection_info
    assert not vault_has_credentials("msteams")
    assert ms_config.path.read_text() == before  # nothing scrubbed
    # Telegram: a boolean bot_token is likewise refused (mint returns
    # None because nothing was enrolled).
    tg_config.path.parent.mkdir(parents=True, exist_ok=True)
    tg_config.path.write_text(json.dumps({"bot_token": True}))
    tg_before = tg_config.path.read_text()
    tg_backend = _tg_backend(api_server)
    assert tg_backend.connect() is False
    assert not vault_has_credentials("telegram")
    assert tg_config.path.read_text() == tg_before


def test_concurrent_scratch_validation_is_isolated(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """Two simultaneous authenticate calls never validate each other."""
    import threading

    results: dict[str, Any] = {}

    def authenticate(key: str, token: str) -> None:
        agent = TelegramAgent.__new__(TelegramAgent)
        agent._backend = _tg_backend(api_server)
        results[key] = json.loads(auth_tools(agent)["authenticate_telegram"](token))

    good = threading.Thread(target=authenticate, args=("good", _REAL_TG_TOKEN))
    bad = threading.Thread(target=authenticate, args=("bad", "1:concurrent-bad-token"))
    good.start()
    bad.start()
    # Bounded: pytest-timeout is disabled repo-wide, so an authd hang
    # must fail this test instead of stalling the whole run.
    good.join(timeout=60)
    bad.join(timeout=60)
    assert not good.is_alive() and not bad.is_alive(), "authenticate did not finish"
    # The bad candidate is always rejected; the good one always wins.
    assert results["bad"]["ok"] is False
    assert results["good"]["ok"] is True
    stored = json.loads((muse_auth_dir() / "vault" / "telegram.json").read_text())
    assert stored["authorized_user_info"]["token"] == _REAL_TG_TOKEN
    # No scratch entries leaked.
    vault_dir = muse_auth_dir() / "vault"
    assert not list(vault_dir.glob("telegram-pending-*.json"))


def test_scratch_validation_obeys_root_deny_policy(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """An explicit telegram read-deny also denies candidate validation."""
    policy_path = muse_auth_dir() / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy["services"]["telegram"]["read"] = "deny"
    policy_path.write_text(json.dumps(policy))
    agent = TelegramAgent.__new__(TelegramAgent)
    agent._backend = _tg_backend(api_server)
    result = json.loads(auth_tools(agent)["authenticate_telegram"](_REAL_TG_TOKEN))
    assert result["ok"] is False
    assert "policy denies" in result["error"]
    assert not vault_has_credentials("telegram")


def test_odd_encoded_response_echoes_are_scrubbed(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """Partial percent-encoded echoes never expose the token to the agent."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    grant("telegram", "write", "session")
    surrogate = backend._bot.token
    session = MuseBoundarySession("telegram")
    headers = {"Authorization": f"Bearer {surrogate}"}
    # Body, header, and reason all echo the token with a %41 escape.
    resp = session.request(
        "GET", f"{api_server.base()}/bot{surrogate}/echoOdd", headers=headers
    )
    assert resp.status_code == 418
    odd = _REAL_TG_TOKEN.replace("A", "%41", 1)
    for surface in (resp.text, resp.reason, resp.headers.get("X-Echoed-Path", "")):
        assert _REAL_TG_TOKEN not in surface
        assert odd not in surface
    # The terminal Location of an over-limit chain is scrubbed too.
    resp = session.request(
        "GET", f"{api_server.base()}/bot{surrogate}/oddChain1", headers=headers
    )
    assert resp.status_code == 302
    location = resp.headers.get("Location", "")
    assert _REAL_TG_TOKEN not in location
    assert odd not in location


# --------------------------------------------- review round-3 regressions


def test_ask_policy_grant_is_inherited_by_scratch_validation(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """A root ``read: ask`` grant approves candidate validation."""
    policy_path = muse_auth_dir() / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy["services"]["telegram"]["read"] = "ask"
    policy_path.write_text(json.dumps(policy))
    agent = TelegramAgent.__new__(TelegramAgent)
    agent._backend = _tg_backend(api_server)
    # Without a grant, validation cannot proceed (read is ask).
    denied = json.loads(auth_tools(agent)["authenticate_telegram"](_REAL_TG_TOKEN))
    assert denied["ok"] is False
    assert "requires user approval" in denied["error"]
    # A telegram read grant (the root identity) approves the scratch
    # probe, so authentication succeeds.
    grant("telegram", "read", "perpetual")
    saved = json.loads(auth_tools(agent)["authenticate_telegram"](_REAL_TG_TOKEN))
    assert saved["ok"] is True
    assert vault_has_credentials("telegram")


def test_double_encoded_echo_cannot_smuggle_the_token(
    muse_env: Path, api_server: _TokenXApiServer, rogue_server: _TokenXApiServer
) -> None:
    """A %253A (two-level) echo is normalized on every surface."""
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    grant("telegram", "write", "session")
    surrogate = backend._bot.token
    double = _REAL_TG_TOKEN.replace(":", "%253A")
    api_server.lowercase_location = f"{rogue_server.base()}/bot{double}/getMe"
    resp = MuseBoundarySession("telegram").request(
        "GET",
        f"{api_server.base()}/bot{surrogate}/redirectLowercase",
        headers={"Authorization": f"Bearer {surrogate}"},
    )
    assert resp.status_code == 401
    rogue_paths = "".join(r["path"] for r in rogue_server.requests)
    assert double not in rogue_paths
    assert _REAL_TG_TOKEN not in rogue_paths
    assert "muse-redacted" in rogue_paths
    audit = _audit_text()
    assert double not in audit and _REAL_TG_TOKEN not in audit


def test_atomic_store_if_absent_never_clobbers(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """store-if-absent leaves an already-present vault credential intact."""
    from kiss.agents.third_party_agents.muse_auth.client import store_credentials

    # First store creates the entry.
    assert store_credentials("telegram", {"kind": "path", "token": _REAL_TG_TOKEN}, [],
                             only_if_absent=True) is True
    # A second store-if-absent with a different token is a no-op.
    assert store_credentials("telegram", {"kind": "path", "token": "9:other"}, [],
                             only_if_absent=True) is False
    stored = json.loads((muse_auth_dir() / "vault" / "telegram.json").read_text())
    assert stored["authorized_user_info"]["token"] == _REAL_TG_TOKEN
    # Auto-migration therefore cannot overwrite it: a stale config
    # token is ignored while the vault credential wins.
    tg_config.save({"bot_token": "9:stale"})
    backend = _tg_backend(api_server)
    assert backend.connect() is True
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/getMe"


def test_prefix_infinity_cache_is_not_trusted(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persisted expires_at=inf (from an older build) forces re-exchange."""
    _ms_env(monkeypatch, api_server)
    store_credentials("msteams", _client_credential_info_from_env(api_server), [])
    vault_file = muse_auth_dir() / "vault" / "msteams.json"
    payload = json.loads(vault_file.read_text())
    payload["cached_token"] = {"access_token": "stale-immortal", "expires_at": float("inf")}
    vault_file.write_text(json.dumps(payload))
    backend = _ms_backend(api_server)
    assert backend._wire_muse() is True
    assert json.loads(backend.list_teams())["ok"] is True
    # A fresh exchange happened; the immortal cache was not trusted.
    assert len(api_server.token_requests) == 1
    graph = [r for r in api_server.requests if "/v1.0/" in r["path"]]
    auth = next(v for k, v in graph[-1]["headers"].items() if k.lower() == "authorization")
    assert auth == "Bearer graph-tok-1"


def test_slack_workspace_named_pending_stays_isolated(muse_env: Path) -> None:
    """A slack-<ws>-<hash> name containing 'pending' is not a scratch service."""
    from kiss.agents.third_party_agents.muse_auth._common import (
        policy_service,
        scratch_root,
    )

    # The exact token-exchange scratch grammar resolves to its root.
    assert scratch_root("telegram-pending-0123456789abcdef") == "telegram"
    assert policy_service("telegram-pending-0123456789abcdef") == "telegram"
    # A Slack workspace whose slug is "pending" is NOT a scratch service:
    # it governs its own policy and is never swept.
    slack_ws = "slack-pending-0123456789abcdef"
    assert scratch_root(slack_ws) == ""
    assert policy_service(slack_ws) == slack_ws
    # Wrong root, wrong hex length, or uppercase hex do not match.
    assert scratch_root("slack-pending-0123456789abcdef") == ""
    assert scratch_root("telegram-pending-short") == ""
    assert scratch_root("telegram-pending-0123456789ABCDEF") == ""


def test_abandoned_pending_file_is_swept(muse_env: Path) -> None:
    """A stale scratch file is swept when the next scratch entry is stored."""
    from kiss.agents.third_party_agents.muse_auth.client import store_credentials

    ensure_daemon()
    vault_dir = muse_auth_dir() / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)
    stale = vault_dir / "telegram-pending-0123456789abcdef.json"
    stale.write_text(json.dumps({"authorized_user_info": {"kind": "path", "token": "x"}}))
    # Backdate it well past the sweep window.
    old = time.time() - 3600
    os.utime(stale, (old, old))
    # A legitimate slack-pending-<hash> workspace file must NOT be swept.
    slack_ws = vault_dir / "slack-pending-fedcba9876543210.json"
    slack_ws.write_text(json.dumps({"authorized_user_info": {"kind": "bearer", "token": "y"}}))
    os.utime(slack_ws, (old, old))
    # Storing a new scratch entry triggers the sweep.
    store_credentials("msteams-pending-abcdef0123456789", {"kind": "bearer", "token": "z"}, [])
    assert not stale.exists()
    assert slack_ws.exists()


def test_malformed_first_migration_connect_returns_false(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daemon-rejected first-migration credential yields False, not a raise."""
    tg_config.save({"bot_token": "not path safe"})
    tg_backend = _tg_backend(api_server)
    assert tg_backend.connect() is False
    assert "auth failed" in tg_backend._connection_info
    assert not vault_has_credentials("telegram")
    # Poll mode exits cleanly rather than crashing with MuseAuthError.
    with pytest.raises(SystemExit):
        tg_make_backend()
    _ms_env(monkeypatch, api_server)
    ms_config.save(
        {"tenant_id": _MS_TENANT, "client_id": _MS_CLIENT_ID, "client_secret": "bad\nsecret"}
    )
    ms_backend = _ms_backend(api_server)
    assert ms_backend.connect() is False
    assert "auth failed" in ms_backend._connection_info
    with pytest.raises(SystemExit):
        ms_make_backend()


# --------------------------------------------- review round-4 regressions


def test_ask_remediation_names_the_grantable_root(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """The ask instruction names the grantable root, not the scratch id."""
    policy_path = muse_auth_dir() / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy["services"]["telegram"]["read"] = "ask"
    policy_path.write_text(json.dumps(policy))
    agent = TelegramAgent.__new__(TelegramAgent)
    agent._backend = _tg_backend(api_server)
    denied = json.loads(auth_tools(agent)["authenticate_telegram"](_REAL_TG_TOKEN))
    assert denied["ok"] is False
    # The printed command grants ``telegram`` (grantable), never a
    # random ``telegram-pending-<hex>`` (which a retry would replace).
    assert "grant telegram read" in denied["error"]
    assert "pending" not in denied["error"]
    assert "on 'telegram'" in denied["error"]


def test_malformed_stale_config_does_not_block_the_vault(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed stale config value never fails an authoritative connect."""
    # Telegram: authoritative vault credential + a malformed stale config.
    store_credentials("telegram", {"kind": "path", "token": _REAL_TG_TOKEN}, [])
    tg_config.save({"bot_token": "not path safe"})
    tg_backend = _tg_backend(api_server)
    assert tg_backend.connect() is True
    assert api_server.requests[-1]["path"] == f"/bot{_REAL_TG_TOKEN}/getMe"
    # MS Teams: authoritative vault credential + a malformed stale secret.
    _ms_env(monkeypatch, api_server)
    store_credentials("msteams", _client_credential_info_from_env(api_server), [])
    ms_config.save(
        {"tenant_id": _MS_TENANT, "client_id": _MS_CLIENT_ID, "client_secret": "bad\nsecret"}
    )
    ms_backend = _ms_backend(api_server)
    assert ms_backend.connect() is True
    assert api_server.token_requests[-1]["client_secret"] == _REAL_MS_SECRET


def test_compare_and_scrub_keeps_a_newer_config_value(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """The scrub removes only the migrated token, never a newer one."""
    from kiss.agents.third_party_agents.telegram_sea import _scrub_config_token

    # A newer token was written to config after the migrated one; the
    # compare-and-scrub must leave it in place.
    tg_config.save({"bot_token": "9:newer-token-from-another-writer"})
    _scrub_config_token(expected=_REAL_TG_TOKEN)
    assert json.loads(tg_config.path.read_text())["bot_token"] == (
        "9:newer-token-from-another-writer"
    )
    # When the config still holds the migrated token, it IS scrubbed.
    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    _scrub_config_token(expected=_REAL_TG_TOKEN)
    assert not tg_config.path.exists()
    # MS Teams compare-and-scrub behaves the same.
    from kiss.agents.third_party_agents.msteams_sea import _scrub_config_secret

    ms_config.save(
        {"tenant_id": _MS_TENANT, "client_id": _MS_CLIENT_ID, "client_secret": "newer-secret"}
    )
    _scrub_config_secret(expected=_REAL_MS_SECRET)
    assert json.loads(ms_config.path.read_text())["client_secret"] == "newer-secret"


def test_huge_finite_cache_expiry_is_not_trusted(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persisted expires_at=1e308 forces a fresh exchange."""
    _ms_env(monkeypatch, api_server)
    store_credentials("msteams", _client_credential_info_from_env(api_server), [])
    vault_file = muse_auth_dir() / "vault" / "msteams.json"
    payload = json.loads(vault_file.read_text())
    payload["cached_token"] = {"access_token": "stale-huge", "expires_at": 1e308}
    vault_file.write_text(json.dumps(payload))
    backend = _ms_backend(api_server)
    assert backend._wire_muse() is True
    assert json.loads(backend.list_teams())["ok"] is True
    assert len(api_server.token_requests) == 1
    graph = [r for r in api_server.requests if "/v1.0/" in r["path"]]
    auth = next(v for k, v in graph[-1]["headers"].items() if k.lower() == "authorization")
    assert auth == "Bearer graph-tok-1"


def test_scratch_files_are_swept_at_daemon_startup(muse_env: Path) -> None:
    """A stale scratch file is swept when the daemon (re)starts."""
    ensure_daemon()
    vault_dir = muse_auth_dir() / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)
    stale = vault_dir / "telegram-pending-0123456789abcdef.json"
    stale.write_text(json.dumps({"authorized_user_info": {"kind": "path", "token": "x"}}))
    old = time.time() - 3600
    os.utime(stale, (old, old))
    # Restart the daemon (protocol handshake tears down and respawns).
    stop_daemon()
    wait_daemon_stopped()
    ensure_daemon()
    # Give the startup sweep a moment.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and stale.exists():
        time.sleep(0.05)
    assert not stale.exists()


def test_stale_boundary_surrogate_reports_reconnect(
    muse_env: Path, api_server: _TokenXApiServer
) -> None:
    """Direct boundary calls with a cleared vault name the remedy."""
    store_credentials("telegram", {"kind": "path", "token": _REAL_TG_TOKEN}, [])
    handle = mint_surrogate("telegram")
    assert handle is not None
    clear_credentials("telegram")
    session = MuseBoundarySession("telegram")
    with pytest.raises(MuseAuthError, match="re-connect the agent backend"):
        session.request(
            "GET",
            f"{api_server.base()}/bot{handle.token}/getMe",
            headers={"Authorization": f"Bearer {handle.token}"},
        )


def test_local_msteams_validation_never_blocks_the_vault(
    muse_env: Path, api_server: _TokenXApiServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Locally-rejected stale config never disables the vault credential.

    Round-5 regression: ``_wire_muse`` used to run the type/tenant
    validation of the legacy config BEFORE minting, so a stale
    non-string or malformed-tenant config failed the connect even
    though the authoritative vault credential was present and mintable.
    """
    _ms_env(monkeypatch, api_server)
    store_credentials("msteams", _client_credential_info_from_env(api_server), [])
    # Non-string credentials (JSON booleans/numbers) in the stale
    # config, all truthy so the migration guard is entered and the
    # type check (not the falsy short-circuit) is what rejects them.
    ms_config.path.parent.mkdir(parents=True, exist_ok=True)
    ms_config.path.write_text(
        json.dumps({"tenant_id": True, "client_id": 7, "client_secret": 3})
    )
    backend = _ms_backend(api_server)
    assert backend.connect() is True
    assert api_server.token_requests[-1]["client_secret"] == _REAL_MS_SECRET
    # The stale config was neither applied nor scrubbed (it never
    # migrated; rotation goes through authenticate_msteams).
    assert json.loads(ms_config.path.read_text())["tenant_id"] is True
    # A malformed tenant string in the stale config.
    ms_config.save(
        {"tenant_id": "bad/tenant", "client_id": _MS_CLIENT_ID, "client_secret": "s3"}
    )
    backend2 = _ms_backend(api_server)
    assert backend2.connect() is True
    assert json.loads(backend2.list_teams())["ok"] is True
    # With an EMPTY vault the same stale configs are genuinely needed,
    # so their local rejection now surfaces (fail closed).
    clear_credentials("msteams")
    backend3 = _ms_backend(api_server)
    assert backend3.connect() is False
    assert "invalid tenant_id" in backend3._connection_info
    ms_config.path.write_text(
        json.dumps({"tenant_id": True, "client_id": 7, "client_secret": 3})
    )
    backend4 = _ms_backend(api_server)
    assert backend4.connect() is False
    assert "non-string credentials" in backend4._connection_info


def test_store_if_absent_presence_and_validation_are_atomic(muse_env: Path) -> None:
    """A concurrent authoritative store makes a malformed candidate moot.

    Round-5 regression: the daemon's presence check, candidate
    validation, and conditional write are ONE vault critical section.
    The test holds the (reentrant) vault lock, lets a malformed
    store-if-absent candidate block at the section entrance, stores the
    authoritative credential while still holding the lock, and releases:
    the candidate must observe the credential and report
    ``created=False`` instead of failing validation — so the automatic
    connect that issued it succeeds.  This runs the REAL daemon handler
    and vault in-process (no test doubles); the interleaving cannot be
    scheduled from outside a subprocess daemon precisely because the
    section is now atomic.
    """
    from kiss.agents.third_party_agents.muse_auth.daemon import MuseAuthDaemon

    daemon = MuseAuthDaemon()
    result: dict[str, Any] = {}

    def candidate() -> None:
        result["reply"] = daemon._handle(
            {
                "op": "store_credentials",
                "service": "telegram",
                "authorized_user_info": {"kind": "path", "token": "not path safe"},
                "scopes": [],
                "only_if_absent": True,
            }
        )

    with daemon.vault.locked():
        thread = threading.Thread(target=candidate)
        thread.start()
        # The candidate is blocked at (or will block at) the critical
        # section; the authoritative store lands first either way.
        time.sleep(0.2)
        assert "reply" not in result
        daemon.vault.store("telegram", {"kind": "path", "token": _REAL_TG_TOKEN}, [])
    thread.join(10.0)
    assert result["reply"] == {"ok": True, "created": False}
    payload = json.loads((muse_auth_dir() / "vault" / "telegram.json").read_text())
    assert payload["authorized_user_info"]["token"] == _REAL_TG_TOKEN


def test_config_lock_makes_scrub_a_compare_and_swap(muse_env: Path) -> None:
    """The scrub's read-compare-replace cycle excludes concurrent writers.

    Round-5 regression: the scrub used to compare an unlocked snapshot,
    so a newer credential written between its read and its file
    replacement was deleted.  Now the whole cycle holds
    ``config_file_lock``, which every config writer shares: a writer
    that lands while the scrub is pending is observed by the scrub's
    read (the comparison backs off), and one that lands after the
    replacement survives it — the in-between interleaving no longer
    exists.  The test proves the serialization end-to-end: the scrub
    demonstrably BLOCKS while a writer holds the lock, then honors the
    value that writer landed.
    """
    from kiss.agents.third_party_agents._channel_agent_utils import (
        config_file_lock,
        write_private_file,
    )
    from kiss.agents.third_party_agents.msteams_sea import _scrub_config_secret
    from kiss.agents.third_party_agents.telegram_sea import _scrub_config_token

    tg_config.save({"bot_token": _REAL_TG_TOKEN})
    scrubbed = threading.Event()

    def scrub() -> None:
        _scrub_config_token(expected=_REAL_TG_TOKEN)
        scrubbed.set()

    with config_file_lock(tg_config.path):
        thread = threading.Thread(target=scrub)
        thread.start()
        # The scrub is blocked on the lock, BEFORE its read.
        assert not scrubbed.wait(0.4)
        # A concurrent writer (holding the lock, as all writers do)
        # lands a newer token.
        write_private_file(
            tg_config.path, json.dumps({"bot_token": "9:newer-token"}, indent=2)
        )
    thread.join(10.0)
    assert scrubbed.is_set()
    # The scrub read the newer value under the lock and backed off.
    assert json.loads(tg_config.path.read_text())["bot_token"] == "9:newer-token"
    # MS Teams: same mechanism, newer secret written while the scrub is
    # pending survives.
    ms_config.save(
        {"tenant_id": _MS_TENANT, "client_id": _MS_CLIENT_ID, "client_secret": "old-sec"}
    )
    ms_scrubbed = threading.Event()

    def ms_scrub() -> None:
        _scrub_config_secret(expected="old-sec")
        ms_scrubbed.set()

    with config_file_lock(ms_config.path):
        ms_thread = threading.Thread(target=ms_scrub)
        ms_thread.start()
        assert not ms_scrubbed.wait(0.4)
        write_private_file(
            ms_config.path,
            json.dumps(
                {
                    "tenant_id": _MS_TENANT,
                    "client_id": _MS_CLIENT_ID,
                    "client_secret": "newer-sec",
                },
                indent=2,
            ),
        )
    ms_thread.join(10.0)
    assert json.loads(ms_config.path.read_text())["client_secret"] == "newer-sec"


def _wait_for_blocked_connect(port: int, timeout: float = 10.0) -> bool:
    """Poll ``/proc/net/tcp`` until a connect to ``port`` sits in SYN-SENT.

    A socket whose remote address is ``127.0.0.1:port`` and whose state is
    SYN-SENT (``02``) proves the daemon has authorized the request and
    resolved the credential -- pinning the vault generation, which happens
    strictly before the transport connect -- and is now blocked in the
    kernel on the saturated accept backlog.  On platforms without
    ``/proc/net/tcp`` a fixed grace sleep is the best approximation.

    Args:
        port: The listener port the daemon's connect is aimed at.
        timeout: Deadline in seconds for the half-open connect to appear.

    Returns:
        True once the half-open connect is visible (or after the
        fallback sleep), False if the deadline passed without one.
    """
    proc_tcp = Path("/proc/net/tcp")
    if not proc_tcp.exists():  # pragma: no cover - non-Linux fallback
        time.sleep(1.0)
        return True
    want_remote = f"0100007F:{port:04X}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for line in proc_tcp.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) > 3 and fields[2] == want_remote and fields[3] == "02":
                return True
        time.sleep(0.02)
    return False


def test_rotation_during_connection_setup_never_emits_old_credential(
    muse_env: Path,
) -> None:
    """A rotation completing during real connect I/O aborts the request.

    Round-5 regression: ``requests`` performs DNS/TCP/TLS setup between
    the daemon's credential resolution and the moment the credential
    bytes are written, and a rotation used to be able to complete
    inside that window, after which the OLD-generation token was sent.
    The transport-write gate now re-checks the generation under the
    vault lock at the head-write boundary (after the connection is
    established), so the rotation aborts the request instead.

    The window is held open with REAL kernel behavior, no scheduling
    hooks: a loopback listener with a saturated accept backlog makes
    the daemon's TCP connect block (SYN drops + client retransmit); the
    rotation lands through the daemon socket while the connect is
    blocked; draining the backlog then lets the connect complete — and
    the accepted socket must receive ZERO bytes.
    """
    import socket

    store_credentials("telegram", {"kind": "path", "token": "1:old-generation-token"}, [])
    handle = mint_surrogate("telegram")
    assert handle is not None
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(0)
    port = listener.getsockname()[1]
    saturator = socket.socket()
    saturator.settimeout(5.0)
    saturator.connect(("127.0.0.1", port))  # fills the accept queue
    result: dict[str, Any] = {}

    def blocked_request() -> None:
        # The boundary surfaces the abort as a synthetic 403 with a
        # MUSE_AUTH_DENIED envelope (the same shape Sentinel denials
        # use), so agents see an ordinary HTTP failure.
        resp = MuseBoundarySession("telegram").request(
            "POST",
            f"http://127.0.0.1:{port}/bot{handle.token}/getMe",
            headers={"Authorization": f"Bearer {handle.token}"},
            json={},
            timeout=25,
        )
        result["status"] = resp.status_code
        result["error"] = resp.text

    try:
        thread = threading.Thread(target=blocked_request)
        thread.start()
        # Wait until the boundary has authorized, resolved the credential
        # (pinning the generation), and entered the kernel connect, which
        # blocks on the saturated backlog with the socket in SYN-SENT.
        assert _wait_for_blocked_connect(port), "daemon connect never blocked on the backlog"
        assert "error" not in result
        # The rotation completes while the connect is still blocked.
        store_credentials("telegram", {"kind": "path", "token": "2:new-generation-token"}, [])
        # Drain the backlog so the blocked connect can complete.
        listener.settimeout(15.0)
        accepted = []
        for _ in range(2):
            try:
                accepted.append(listener.accept()[0])
            except TimeoutError:  # pragma: no cover - kernel timing
                break
        thread.join(30.0)
        assert not thread.is_alive()
        assert result["status"] == 403
        assert "changed mid-request" in result["error"]
        assert "MUSE_AUTH_DENIED" in result["error"]
        # No byte of either generation's credential reached the wire.
        received = b""
        for conn in accepted:
            conn.settimeout(1.0)
            with contextlib.suppress(OSError):
                received += conn.recv(65536)
            conn.close()
        assert received == b""
    finally:
        saturator.close()
        listener.close()
