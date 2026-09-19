# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Connect-style consent flows.

Exercises the Muse-app-like sign-in of the four connectors whose
providers support a poll-based grant — GitHub, Twitch and Microsoft
Teams (OAuth 2.0 device authorization grant, RFC 8628) and Nextcloud
Talk (Login Flow v2) — against real loopback emulators of the
authorization servers and APIs, with the real Muse-auth daemon storing
and refreshing the resulting credentials.  No mocks: the agent code
talks HTTP to the emulators exactly as it would to the providers, and
the daemon runs as a separate process.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
import requests

from kiss.agents.third_party_agents import _device_auth
from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server
from kiss.agents.third_party_agents._device_auth import (
    ConsentSession,
    DeviceFlowProvider,
    DeviceFlowSession,
    NextcloudLoginSession,
    TokenGrant,
    consent_instructions,
)
from kiss.agents.third_party_agents.github_agent import GitHubAgent
from kiss.agents.third_party_agents.github_agent import _config as gh_config
from kiss.agents.third_party_agents.matrix_agent import MatrixAgent, MatrixChannelBackend
from kiss.agents.third_party_agents.matrix_agent import _config as mx_config
from kiss.agents.third_party_agents.msteams_agent import MSTeamsAgent
from kiss.agents.third_party_agents.msteams_agent import _config as ms_config
from kiss.agents.third_party_agents.muse_auth._common import muse_auth_dir
from kiss.agents.third_party_agents.muse_auth.client import (
    MuseAuthError,
    MuseBoundarySession,
    mint_surrogate,
    store_credentials,
    vault_has_credentials,
)
from kiss.agents.third_party_agents.nextcloud_talk_agent import NextcloudTalkAgent
from kiss.agents.third_party_agents.nextcloud_talk_agent import _config as nc_config
from kiss.agents.third_party_agents.signal_agent import SignalAgent, SignalLinkSession
from kiss.agents.third_party_agents.signal_agent import _config as sg_config
from kiss.agents.third_party_agents.twitch_agent import TwitchAgent
from kiss.agents.third_party_agents.twitch_agent import _config as tw_config
from kiss.tests.agents.third_party_agents.muse_test_utils import (
    auth_tools,
    setup_muse_env,
    teardown_muse_env,
)
from kiss.tests.conftest import IS_WINDOWS, install_cli_script

_MS_TENANT = "contoso.onmicrosoft.com"

# --------------------------------------------------------------- emulators


class _DeviceState:
    """One pending device authorization on the emulated server."""

    def __init__(self, dialect: str, scope: str) -> None:
        self.dialect = dialect
        self.scope = scope
        self.approved = False
        self.denied = False
        self.polls = 0


class _AuthServer(ThreadedHTTPServer):
    """Emulates GitHub / Twitch / Microsoft device flows plus their APIs.

    Behaviour knobs (set by tests before the flow runs):

    * ``expiring``: token answers carry ``refresh_token``/``expires_in``.
    * ``expires_in``: access-token lifetime announced to the client.
    * ``slow_down_once``: the first poll answers ``slow_down``.
    * ``refresh_error``: refresh grants fail with this OAuth error code.
    * ``omit_code``: device answers carry no ``verification_uri``.
    """

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _AuthHandler)
        self.devices: dict[str, _DeviceState] = {}
        self.access_tokens: dict[str, str] = {}
        self.refresh_tokens: dict[str, str] = {}
        self.expiring = False
        self.expires_in = 3600
        self.slow_down_once = False
        self.refresh_error = ""
        self.refresh_status = 400
        self.omit_code = False
        self.complete_uri = False
        self.plain_text_device = False
        self.bad_expires_in = False
        self.bad_access_token = False
        self.rotate = True
        self.refreshes = 0
        # Graph probe status (Teams); 200 answers the team list.
        self.graph_status = 200
        # Matrix homeserver knobs: OAuth API present, registrations made,
        # tokens revoked, whoami/joined_rooms token check.
        self.matrix_oauth = True
        self.matrix_insecure_revocation = False
        self.refresh_hook: Any = None
        self.matrix_registrations: list[dict[str, Any]] = []
        self.revoked: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def base(self) -> str:
        """Return the emulator's base URL."""
        return f"http://127.0.0.1:{self.server_port}"

    def next_id(self, prefix: str) -> str:
        """Return a fresh unique identifier."""
        with self._lock:
            self._counters[prefix] = self._counters.get(prefix, 0) + 1
            return f"{prefix}-{self._counters[prefix]}"

    def pending_codes(self) -> list[str]:
        """Return the device codes not yet approved or denied."""
        return [c for c, d in self.devices.items() if not d.approved and not d.denied]

    def approve(self, device_code: str | None = None) -> None:
        """Approve the given (or the only pending) device code."""
        code = device_code or self.pending_codes()[0]
        self.devices[code].approved = True

    def deny(self) -> None:
        """Deny the only pending device code."""
        self.devices[self.pending_codes()[0]].denied = True

    def issue(
        self, scope: str, refresh_scope: str = "", expiring: bool | None = None
    ) -> dict[str, Any]:
        """Mint a token answer for *scope* (``expiring`` overrides the knob)."""
        access = "bad\ntoken" if self.bad_access_token else self.next_id("access")
        self.access_tokens[access] = scope
        body: dict[str, Any] = {"access_token": access, "token_type": "bearer"}
        if self.expiring if expiring is None else expiring:
            body["expires_in"] = "later" if self.bad_expires_in else self.expires_in
            if self.rotate or not self.refresh_tokens:
                refresh = self.next_id("refresh")
                self.refresh_tokens[refresh] = refresh_scope or scope
                body["refresh_token"] = refresh
        return body


class _AuthHandler(BaseHTTPRequestHandler):
    """Request handler for :class:`_AuthServer`."""

    server: Any

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence request logging."""

    def _json(self, status: int, body: Any) -> None:
        if urlsplit(
            self.path
        ).path == "/login/oauth/access_token" and "application/json" not in self.headers.get(
            "Accept", ""
        ):
            # GitHub's real default: form-encoded unless JSON is requested.
            data = urlencode({k: str(v) for k, v in body.items()}).encode()
            content_type = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(body).encode()
            content_type = "application/json"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length") or 0)
        self._raw_body = self.rfile.read(length).decode() if length else ""
        return {k: v[0] for k, v in parse_qs(self._raw_body).items()}

    def _record(self, form: dict[str, str]) -> None:
        self.server.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "form": form,
                "raw_body": getattr(self, "_raw_body", ""),
            }
        )

    # ---- device authorization ------------------------------------------

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        """Serve the device, token and login endpoints."""
        path = urlsplit(self.path).path
        form = self._form()
        self._record(form)
        if path == "/login/device/code":
            self._device("github", form, "https://github.com/login/device", scope_key="scope")
        elif path == "/oauth2/device":
            self._device("twitch", form, None, scope_key="scopes")
        elif path == "/mas/oauth2/device":
            self._device("matrix", form, "https://account.example/link", scope_key="scope")
        elif path == "/mas/oauth2/registration":
            self._register_matrix_client()
        elif path == "/mas/oauth2/revoke":
            token = form.get("token", "")
            self.server.access_tokens.pop(token, None)
            self.server.revoked.append(token)
            self._json(200, {})
        elif path == f"/{_MS_TENANT}/oauth2/v2.0/devicecode":
            self._device("ms", form, "https://microsoft.com/devicelogin", scope_key="scope")
        elif path in (
            "/login/oauth/access_token",
            "/oauth2/token",
            "/mas/oauth2/token",
            f"/{_MS_TENANT}/oauth2/v2.0/token",
        ):
            self._token(form)
        else:
            self._json(404, {"error": "not_found"})

    def _register_matrix_client(self) -> None:
        """Dynamic client registration (Matrix spec): JSON in, 201 + client_id out."""
        raw = self.server.requests[-1]["raw_body"]
        try:
            metadata = json.loads(raw)
        except ValueError:
            metadata = None
        if not isinstance(metadata, dict) or not str(metadata.get("client_uri", "")).startswith(
            "https://"
        ):
            self._json(400, {"error": "invalid_client_metadata"})
            return
        self.server.matrix_registrations.append(metadata)
        client_id = self.server.next_id("mas-client")
        self._json(201, {"client_id": client_id, **metadata})

    def _matrix_metadata(self) -> dict[str, Any]:
        base = self.server.base()
        return {
            "issuer": f"{base}/mas/",
            "authorization_endpoint": f"{base}/mas/authorize",
            "token_endpoint": f"{base}/mas/oauth2/token",
            "registration_endpoint": f"{base}/mas/oauth2/registration",
            "device_authorization_endpoint": f"{base}/mas/oauth2/device",
            "revocation_endpoint": (
                "http://revoke.example/oauth2/revoke"
                if self.server.matrix_insecure_revocation
                else f"{base}/mas/oauth2/revoke"
            ),
            "grant_types_supported": [
                "authorization_code",
                "refresh_token",
                _device_auth.DEVICE_CODE_GRANT,
            ],
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
        }

    def _device(self, dialect: str, form: dict[str, str], uri: str | None, scope_key: str) -> None:
        if not form.get("client_id"):
            self._json(400, {"error": "invalid_request"})
            return
        if form.get("client_id") == "rejected-client":
            self._json(401, {"error": "invalid_client", "error_description": "unknown app"})
            return
        if self.server.plain_text_device:
            data = b"<html>maintenance</html>"
            self.send_response(503)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        code = self.server.next_id("device")
        self.server.devices[code] = _DeviceState(dialect, form.get(scope_key, ""))
        user_code = f"CODE-{code[-1]}"
        body: dict[str, Any] = {
            "device_code": code,
            "user_code": user_code,
            "expires_in": 120,
            "interval": 1,
        }
        if self.server.complete_uri:
            body["verification_uri_complete"] = f"{uri}?user_code={user_code}"
        if not self.server.omit_code:
            if dialect == "twitch":
                body["verification_uri"] = (
                    f"https://www.twitch.tv/activate?public=true&device-code={user_code}"
                )
            else:
                body["verification_uri"] = uri
        self._json(200, body)

    def _token(self, form: dict[str, str]) -> None:
        grant = form.get("grant_type")
        if grant == "refresh_token":
            self._refresh(form)
            return
        if grant != _device_auth.DEVICE_CODE_GRANT:
            self._json(400, {"error": "unsupported_grant_type"})
            return
        state = self.server.devices.get(form.get("device_code", ""))
        if state is None:
            self._json(400, {"error": "incorrect_device_code"})
            return
        state.polls += 1
        if state.dialect == "twitch" and form.get("scopes") != state.scope:
            self._json(400, {"status": 400, "message": "missing scopes"})
            return
        if self.server.slow_down_once and state.polls == 1:
            self._json(400, {"error": "slow_down", "interval": 6})
            return
        if state.denied:
            code = "authorization_declined" if state.dialect == "ms" else "access_denied"
            self._json(400, {"error": code})
            return
        if not state.approved:
            if state.dialect == "twitch":
                self._json(400, {"status": 400, "message": "authorization_pending"})
            else:
                self._json(400, {"error": "authorization_pending"})
            return
        # A Matrix homeserver MUST issue a refresh token (short-lived tokens).
        body = self.server.issue(
            state.scope, form.get("scope", ""), True if state.dialect == "matrix" else None
        )
        if state.dialect == "twitch":
            body["scope"] = state.scope.split()
        elif state.dialect == "github":
            # Real GitHub answers with the granted scopes COMMA-separated
            # ("read:org,read:user,repo") although the request format is
            # space-separated — verified against live GitHub 2026-09-15.
            body["scope"] = ",".join(state.scope.split())
        else:
            body["scope"] = state.scope
        del self.server.devices[form["device_code"]]
        self._json(200, body)

    def _refresh(self, form: dict[str, str]) -> None:
        self.server.refreshes += 1
        if self.server.refresh_hook is not None:
            # Something happens on the client while the grant is in flight.
            self.server.refresh_hook()
        if self.server.refresh_error == "plain":
            data = b"Bad Request"
            self.send_response(self.server.refresh_status)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.server.refresh_error:
            self._json(self.server.refresh_status, {"error": self.server.refresh_error})
            return
        refresh_token = form.get("refresh_token", "")
        scope = self.server.refresh_tokens.get(refresh_token)
        if scope is None or not form.get("client_id") or form.get("client_secret"):
            self._json(400, {"error": "invalid_grant"})
            return
        if self.server.rotate:
            del self.server.refresh_tokens[refresh_token]
        self._json(200, self.server.issue(scope))

    # ---- protected APIs --------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        """Serve the GitHub, Twitch Helix and Graph probe endpoints."""
        path = urlsplit(self.path).path
        self._record({})
        if path == "/_matrix/client/v1/auth_metadata":
            if self.server.matrix_oauth:
                self._json(200, self._matrix_metadata())
            else:
                self._json(404, {"errcode": "M_UNRECOGNIZED", "error": "Unrecognized request"})
            return
        auth = self.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        if token not in self.server.access_tokens:
            if path.startswith("/_matrix/"):
                self._json(401, {"errcode": "M_UNKNOWN_TOKEN", "error": "Unrecognised token"})
                return
            self._json(401, {"error": {"code": "InvalidAuthenticationToken"}})
            return
        if path == "/_matrix/client/v3/account/whoami":
            scope = self.server.access_tokens[token]
            device = next(
                (
                    s.rsplit(":", 1)[1]
                    for s in scope.split()
                    if s.startswith("urn:matrix:client:device:")
                ),
                "LEGACYDEV",
            )
            self._json(200, {"user_id": "@alice:example.org", "device_id": device})
        elif path == "/_matrix/client/v3/joined_rooms":
            self._json(200, {"joined_rooms": ["!room:example.org"]})
        elif path == "/user":
            self._json(200, {"login": "octocat", "id": 1})
        elif path == "/helix/users":
            if not self.headers.get("Client-ID"):
                self._json(401, {"error": "Unauthorized", "message": "Client-ID missing"})
                return
            self._json(200, {"data": [{"id": "42", "login": "kisscaster"}]})
        elif path == "/v1.0/teams":
            if self.server.graph_status != 200:
                self._json(self.server.graph_status, {"error": {"code": "ServiceUnavailable"}})
                return
            self._json(200, {"value": [{"id": "team-1", "displayName": "KISS"}]})
        else:
            self._json(404, {"error": "not_found"})


class _NextcloudServer(ThreadedHTTPServer):
    """Emulates Nextcloud Login Flow v2 and the Talk ``/room`` read."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _NextcloudHandler)
        self.flows: dict[str, bool] = {}
        self.app_passwords: dict[str, str] = {}
        self.login_name = "alice@example.com"
        self.disabled = False
        self.foreign_endpoint = False
        self.plain_text = False
        self.poll_error = False
        # ``server`` reported by the poll answer (defaults to the base URL).
        self.reported_server = ""
        self.revoked: list[str] = []
        self.requests: list[dict[str, Any]] = []

    def base(self) -> str:
        """Return the emulator's base URL."""
        return f"http://127.0.0.1:{self.server_port}"

    def grant(self) -> None:
        """Simulate the user clicking "Grant access" on the only open flow."""
        token = next(t for t, done in self.flows.items() if not done)
        self.flows[token] = True


class _NextcloudHandler(BaseHTTPRequestHandler):
    """Request handler for :class:`_NextcloudServer`."""

    server: Any

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence request logging."""

    def _json(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        """Serve ``/index.php/login/v2`` and ``/login/v2/poll``."""
        path = urlsplit(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode()).items()}
        self.server.requests.append({"path": path, "headers": dict(self.headers), "form": form})
        if path == "/index.php/login/v2":
            if self.server.disabled:
                self._json(404, {"message": "not found"})
                return
            if self.server.plain_text:
                data = b"<html>login page</html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            token = f"poll-{len(self.server.flows) + 1}"
            self.server.flows[token] = False
            endpoint_base = self.server.base()
            if self.server.foreign_endpoint:
                endpoint_base = "http://nextcloud.evil.example"
            self._json(
                200,
                {
                    "poll": {"token": token, "endpoint": f"{endpoint_base}/login/v2/poll"},
                    "login": f"{self.server.base()}/login/v2/flow/{token}",
                },
            )
        elif path == "/login/v2/poll":
            token = form.get("token", "")
            if self.server.poll_error:
                self._json(500, {"message": "database down"})
                return
            if not self.server.flows.get(token):
                self._json(404, {})
                return
            del self.server.flows[token]
            app_password = f"app-password-{token}"
            self.server.app_passwords[app_password] = self.server.login_name
            self._json(
                200,
                {
                    "server": self.server.reported_server or self.server.base(),
                    "loginName": self.server.login_name,
                    "appPassword": app_password,
                },
            )
        else:
            self._json(404, {})

    def _basic_password(self) -> str | None:
        """Return the app password of a valid Basic credential, else None."""
        import base64

        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return None
        try:
            user, _, password = base64.b64decode(auth[6:]).decode().partition(":")
        except Exception:
            return None
        return password if self.server.app_passwords.get(password) == user else None

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        """Serve the OCS ``/room`` read with Basic-auth checking."""
        path = urlsplit(self.path).path
        self.server.requests.append({"path": path, "headers": dict(self.headers), "form": {}})
        ok = self._basic_password() is not None
        meta = {"status": "ok" if ok else "failure", "statuscode": 200 if ok else 997}
        rooms = [{"token": "r1", "displayName": "General"}] if ok else []
        body = {"ocs": {"meta": meta, "data": rooms}}
        self._json(200 if ok else 401, body)

    def do_DELETE(self) -> None:  # noqa: N802 - http.server API
        """Serve ``DELETE /ocs/v2.php/core/apppassword`` (revoke current)."""
        path = urlsplit(self.path).path
        self.server.requests.append({"path": path, "headers": dict(self.headers), "form": {}})
        password = self._basic_password()
        if path != "/ocs/v2.php/core/apppassword" or password is None:
            self._json(401, {"ocs": {"meta": {"status": "failure", "statuscode": 997}}})
            return
        del self.server.app_passwords[password]
        self.server.revoked.append(password)
        self._json(200, {"ocs": {"meta": {"status": "ok", "statuscode": 200}, "data": []}})


# ---------------------------------------------------------------- fixtures


@pytest.fixture()
def auth_server() -> Any:
    """Run the device-flow emulator on a loopback port."""
    server = _AuthServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    stop_http_server(server, thread)


@pytest.fixture()
def nextcloud_server() -> Any:
    """Run the Nextcloud emulator on a loopback port."""
    server = _NextcloudServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    stop_http_server(server, thread)


@pytest.fixture()
def muse_env(isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Enable Muse-auth inside an isolated ``KISS_HOME`` with a live daemon.

    GitHub, Twitch and MS Teams have fixed cloud API hosts, so their
    loopback emulators are reached through ``extra_hosts``; Nextcloud is
    origin-bound at enrollment and needs no policy entry.
    """
    policy = {
        "defaults": {"read": "allow", "write": "ask"},
        "services": {
            "github": {"extra_hosts": ["127.0.0.1"]},
            "twitch": {"extra_hosts": ["127.0.0.1"]},
            "msteams": {"extra_hosts": ["127.0.0.1"]},
            # Clearing a login-flow connection revokes the app password
            # with a DELETE (a write) through the boundary.
            "nextcloud": {"write": "allow"},
        },
    }
    setup_muse_env(monkeypatch, policy)
    yield isolated_kiss_home
    teardown_muse_env()


@pytest.fixture(autouse=True)
def _no_leftover_sessions() -> Any:
    """Cancel any consent session a test left behind."""
    yield
    for service in list(ConsentSession._active):
        ConsentSession.cancel_active(service)


def _finish(tool: Any, attempts: int = 30) -> dict[str, Any]:
    """Call a finish tool until it stops reporting ``pending``."""
    for _ in range(attempts):
        result: dict[str, Any] = json.loads(tool())
        if result.get("status") != "pending":
            return result
    raise AssertionError("finish tool kept reporting pending")


def _vault_entry(service: str) -> dict[str, Any]:
    """Read a vault file (test-side inspection of the daemon's storage)."""
    return dict(json.loads((muse_auth_dir() / "vault" / f"{service}.json").read_text()))


# ------------------------------------------------------------ GitHub (Muse)


def test_github_device_flow_enrolls_bearer_and_connects(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub: consent_required → user approves → token validated and vaulted."""
    monkeypatch.setenv("GITHUB_OAUTH_BASE", auth_server.base())
    agent = GitHubAgent()
    agent._backend._base_url = auth_server.base()
    tools = auth_tools(agent)

    # No client ID anywhere: the agent is told how to get one, no network call.
    missing = json.loads(tools["authenticate_github"]())
    assert missing["ok"] is False
    assert "settings/applications/new" in missing["error"]
    assert not auth_server.requests

    started = json.loads(tools["authenticate_github"](client_id="kiss-app", read_only=True))
    assert started["status"] == "consent_required"
    assert started["verification_uri"] == "https://github.com/login/device"
    assert started["user_code"].startswith("CODE-")
    assert "OWN browser" in started["instructions"]
    assert started["user_code"] in started["instructions"]
    assert "finish_github_auth" in started["instructions"]
    device_request = auth_server.requests[0]
    assert device_request["form"] == {"client_id": "kiss-app", "scope": "repo read:org read:user"}
    assert device_request["headers"]["Accept"] == "application/json"
    # Only the (public) client ID is remembered at this point: nothing
    # else changes until the sign-in lands.
    saved = json.loads(gh_config.path.read_text())
    assert saved == {"oauth_client_id": "kiss-app"}

    pending = json.loads(tools["finish_github_auth"]())
    assert pending == {
        "ok": False,
        "status": "pending",
        "error": "The user has not approved yet; ask them to finish the sign-in, then call "
        "this tool again.",
    }
    auth_server.approve()
    done = _finish(tools["finish_github_auth"])
    assert done == {
        "ok": True,
        "message": "GitHub connected.",
        "login": "octocat",
        "read_only": True,
        "scope": "repo read:org read:user",
    }
    assert agent._is_authenticated() is True
    assert agent._backend._read_only is True
    # Non-expiring token → plain bearer credential in the vault.
    assert _vault_entry("github")["authorized_user_info"] == {
        "kind": "bearer",
        "token": "access-1",
    }
    assert agent._backend._token.startswith("muse-sgt.github.")
    # Real GitHub reports granted scopes comma-separated; the adapter
    # splits them into individual vault scope entries.
    assert _vault_entry("github")["scopes"] == ["repo", "read:org", "read:user"]
    assert "token" not in json.loads(gh_config.path.read_text())
    assert json.loads(tools["check_github_auth"]()) == {"ok": True, "read_only": True}
    # The client ID is remembered for the next sign-in.
    again = json.loads(tools["authenticate_github"]())
    assert again["status"] == "consent_required"
    assert auth_server.requests[-1]["form"]["client_id"] == "kiss-app"
    assert tools["clear_github_auth"]() == "GitHub configuration cleared."
    assert not vault_has_credentials("github")
    assert "github" not in ConsentSession._active


def test_github_expiring_token_is_refreshed_by_daemon(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expiring grant becomes an oauth2_refresh_token entry the daemon renews."""
    monkeypatch.setenv("GITHUB_OAUTH_BASE", auth_server.base())
    monkeypatch.setenv("KISS_GITHUB_CLIENT_ID", "env-app")
    auth_server.expiring = True
    auth_server.expires_in = 30  # inside the 60 s skew: refresh on first use
    agent = GitHubAgent()
    agent._backend._base_url = auth_server.base()
    tools = auth_tools(agent)
    started = json.loads(tools["authenticate_github"](scope="repo"))
    assert started["status"] == "consent_required"
    assert auth_server.requests[0]["form"]["client_id"] == "env-app"
    auth_server.approve()
    done = _finish(tools["finish_github_auth"])
    assert done["ok"] is True and done["login"] == "octocat"
    entry = _vault_entry("github")["authorized_user_info"]
    assert entry["kind"] == "oauth2_refresh_token"
    assert entry["token_url"] == f"{auth_server.base()}/login/oauth/access_token"
    assert entry["client_id"] == "env-app"
    assert "token_scope" not in entry
    # The /user validation used the freshly issued token directly (no
    # refresh yet); the first boundary call forces one (30 s lifetime).
    assert auth_server.refreshes == 0
    data, error = agent._backend._api("GET", "/user")
    assert error == "" and data["login"] == "octocat"
    assert auth_server.refreshes == 1
    refresh_request = next(
        r for r in auth_server.requests if r["form"].get("grant_type") == "refresh_token"
    )
    assert refresh_request["form"] == {
        "grant_type": "refresh_token",
        "refresh_token": "refresh-1",
        "client_id": "env-app",
    }
    # GitHub answers form-encoded unless JSON is asked for explicitly.
    assert refresh_request["headers"]["Accept"] == "application/json"
    # The rotated refresh token replaced the stored one; every later
    # call refreshes again because the lifetime stays within the skew.
    assert _vault_entry("github")["authorized_user_info"]["refresh_token"] == "refresh-2"
    data, error = agent._backend._api("GET", "/user")
    assert error == "" and data["login"] == "octocat"
    assert auth_server.refreshes == 2
    # A refresh refusal surfaces as a boundary error naming the OAuth code only.
    auth_server.refresh_error = "invalid_grant"
    with pytest.raises(MuseAuthError, match="invalid_grant"):
        agent._backend._api("GET", "/user")


def test_github_denied_and_rejected_client(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Denied consent and a refused client ID report errors and enroll nothing."""
    monkeypatch.setenv("GITHUB_OAUTH_BASE", auth_server.base())
    agent = GitHubAgent()
    tools = auth_tools(agent)
    rejected = json.loads(tools["authenticate_github"](client_id="rejected-client"))
    assert rejected == {
        "ok": False,
        "error": "device authorization refused (invalid_client: unknown app)",
    }
    assert "github" not in ConsentSession._active

    assert json.loads(tools["authenticate_github"](client_id="kiss-app"))["status"] == (
        "consent_required"
    )
    auth_server.deny()
    denied = _finish(tools["finish_github_auth"])
    assert denied == {
        "ok": False,
        "error": "GitHub sign-in failed: sign-in refused (access_denied)",
    }
    assert not vault_has_credentials("github")
    assert agent._is_authenticated() is False
    # Without a session, finish explains what to do.
    assert json.loads(tools["finish_github_auth"]()) == {
        "ok": False,
        "error": "GitHub sign-in failed: no sign-in in progress; call authenticate_github() first",
    }


def test_github_api_rejecting_new_token_rolls_back(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token the API refuses is cleared again instead of staying enrolled."""
    monkeypatch.setenv("GITHUB_OAUTH_BASE", auth_server.base())
    agent = GitHubAgent()
    agent._backend._base_url = auth_server.base()
    tools = auth_tools(agent)
    tools["authenticate_github"](client_id="kiss-app")
    session = ConsentSession._active["github"]
    auth_server.approve()
    session._thread.join(timeout=10.0)
    assert session.result is not None
    # Forget the issued token server-side so /user answers 401.
    auth_server.access_tokens.clear()
    result = _finish(tools["finish_github_auth"])
    assert result["ok"] is False
    assert "GitHub rejected the new token" in result["error"]
    assert not vault_has_credentials("github")
    assert agent._is_authenticated() is False


def test_github_token_path_unchanged(muse_env: Path, auth_server: _AuthServer) -> None:
    """A personal access token still configures GitHub directly (Muse vault)."""
    agent = GitHubAgent()
    tools = auth_tools(agent)
    assert json.loads(tools["authenticate_github"]("ghp_direct"))["ok"] is True
    assert _vault_entry("github")["authorized_user_info"] == {
        "kind": "bearer",
        "token": "ghp_direct",
    }
    assert json.loads(gh_config.path.read_text()) == {"read_only": "false"}
    assert agent._is_authenticated() is True


# ---------------------------------------------------------- GitHub (legacy)


def test_github_device_flow_legacy_mode(
    isolated_kiss_home: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With Muse-auth off the device-flow token lands in config.json."""
    assert os.environ.get("KISS_MUSE_AUTH") == "0"  # pinned by tests/conftest.py
    monkeypatch.setenv("GITHUB_OAUTH_BASE", auth_server.base())
    agent = GitHubAgent()
    agent._backend._base_url = auth_server.base()
    tools = auth_tools(agent)
    unauth = tools["check_github_auth"]()
    assert "finish_github_auth" in unauth and "KISS_GITHUB_CLIENT_ID" in unauth
    started = json.loads(tools["authenticate_github"](client_id="kiss-app"))
    assert started["status"] == "consent_required"
    auth_server.approve()
    done = _finish(tools["finish_github_auth"])
    assert done["ok"] is True and done["read_only"] is False
    assert json.loads(gh_config.path.read_text()) == {
        "token": "access-1",
        "read_only": "false",
        "oauth_client_id": "kiss-app",
    }
    assert agent._backend._token == "access-1"
    assert json.loads(tools["check_github_auth"]()) == {"ok": True, "read_only": False}


# ------------------------------------------------------------------- Twitch


def test_twitch_device_code_grant_and_refresh(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Twitch: pre-filled activate link, scopes repeated on poll, daemon refresh."""
    monkeypatch.setenv("TWITCH_OAUTH_BASE", auth_server.base())
    auth_server.expiring = True
    auth_server.expires_in = 30
    agent = TwitchAgent()
    agent._backend._helix_base = f"{auth_server.base()}/helix"
    tools = auth_tools(agent)
    assert tools["authenticate_twitch"]("  ") == "client_id cannot be empty."
    unauth = tools["check_twitch_auth"]()
    assert "twitch.tv/activate" in unauth and "finish_twitch_auth" in unauth

    started = json.loads(tools["authenticate_twitch"]("cid1", channel_name="kisscaster"))
    assert started["status"] == "consent_required"
    assert started["verification_uri"].startswith(
        "https://www.twitch.tv/activate?public=true&device-code=CODE-"
    )
    assert auth_server.requests[0]["form"]["scopes"].startswith("user:read:chat user:write:chat")
    # Nothing is written until the sign-in lands.
    assert not tw_config.path.exists()
    auth_server.approve()
    done = _finish(tools["finish_twitch_auth"])
    assert done == {
        "ok": True,
        "message": "Twitch credentials saved (Muse-auth).",
        "login": "kisscaster",
    }
    polls = [r["form"] for r in auth_server.requests if r["path"] == "/oauth2/token"]
    assert polls[0]["grant_type"] == _device_auth.DEVICE_CODE_GRANT
    assert polls[0]["scopes"] == auth_server.requests[0]["form"]["scopes"]
    entry = _vault_entry("twitch")["authorized_user_info"]
    assert entry["kind"] == "oauth2_refresh_token"
    assert entry["token_url"] == f"{auth_server.base()}/oauth2/token"
    # The pre-store probe used the fresh token directly; the first
    # boundary read forces a refresh (30 s lifetime).
    assert auth_server.refreshes == 0
    assert agent._backend._client_id == "cid1"
    assert agent._backend._muse is True
    assert json.loads(tools["check_twitch_auth"]())["login"] == "kisscaster"
    assert auth_server.refreshes == 1
    assert json.loads(tw_config.path.read_text()) == {
        "client_id": "cid1",
        "channel_name": "kisscaster",
    }
    assert tools["clear_twitch_auth"]() == "Twitch authentication cleared."
    assert not vault_has_credentials("twitch")


def test_twitch_slow_down_and_token_path(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow_down answer backs off; the direct access-token path still works."""
    monkeypatch.setenv("TWITCH_OAUTH_BASE", auth_server.base())
    auth_server.slow_down_once = True
    agent = TwitchAgent()
    agent._backend._helix_base = f"{auth_server.base()}/helix"
    tools = auth_tools(agent)
    tools["authenticate_twitch"]("cid1", scopes="clips:edit")
    session = ConsentSession._active["twitch"]
    assert isinstance(session, DeviceFlowSession)
    auth_server.approve()
    done = _finish(tools["finish_twitch_auth"])
    assert done["ok"] is True
    assert session._interval == 6.0  # 1 s + the 5 s slow-down step
    assert _vault_entry("twitch")["authorized_user_info"] == {
        "kind": "bearer",
        "token": "access-1",
    }
    # Direct token: enrolled as bearer, validated, secret never persisted.
    auth_server.access_tokens["hand-token"] = "clips:edit"
    direct = json.loads(
        tools["authenticate_twitch"]("cid1", "unused-secret", "hand-token", "kisscaster")
    )
    assert direct["ok"] is True and direct["login"] == "kisscaster"
    assert _vault_entry("twitch")["authorized_user_info"] == {
        "kind": "bearer",
        "token": "hand-token",
    }
    assert "unused-secret" not in tw_config.path.read_text()


def test_twitch_legacy_mode_device_flow(
    isolated_kiss_home: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With Muse-auth off the device-code access token is stored in config.json."""
    monkeypatch.setenv("TWITCH_OAUTH_BASE", auth_server.base())
    auth_server.expiring = True
    agent = TwitchAgent()
    agent._backend._helix_base = f"{auth_server.base()}/helix"
    tools = auth_tools(agent)
    assert json.loads(tools["authenticate_twitch"]("cid1"))["status"] == "consent_required"
    auth_server.approve()
    done = _finish(tools["finish_twitch_auth"])
    assert done["ok"] is True and done["login"] == "kisscaster"
    saved = json.loads(tw_config.path.read_text())
    assert saved["access_token"] == "access-1" and saved["client_id"] == "cid1"
    assert "refresh-1" not in tw_config.path.read_text()
    # Legacy direct path keeps its behaviour.
    auth_server.access_tokens["hand-token"] = ""
    direct = json.loads(tools["authenticate_twitch"]("cid1", "sec", "hand-token", "chan"))
    assert direct["ok"] is True
    assert json.loads(tw_config.path.read_text())["client_secret"] == "sec"


# ----------------------------------------------------------------- MS Teams


def test_msteams_device_code_delegated_token(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MS Teams: device code → refresh-token entry with scope, probed via Graph."""
    monkeypatch.setenv("MSTEAMS_LOGIN_BASE", auth_server.base())
    auth_server.expiring = True
    agent = MSTeamsAgent()
    agent._backend._graph_base = f"{auth_server.base()}/v1.0"
    tools = auth_tools(agent)
    unauth = tools["check_msteams_auth"]()
    assert "microsoft.com/devicelogin" in unauth and "finish_msteams_auth" in unauth
    assert tools["authenticate_msteams"]("", "c") == "tenant_id cannot be empty."
    assert "GUID or verified domain" in tools["authenticate_msteams"]("bad/tenant", "c")
    assert "control characters" in tools["authenticate_msteams"](_MS_TENANT, "c\nid")

    started = json.loads(tools["authenticate_msteams"](_MS_TENANT, "app-1", bot_id="B1"))
    assert started["status"] == "consent_required"
    assert started["verification_uri"] == "https://microsoft.com/devicelogin"
    assert started["user_code"]
    device = auth_server.requests[0]
    assert device["path"] == f"/{_MS_TENANT}/oauth2/v2.0/devicecode"
    assert device["form"]["scope"].startswith("offline_access User.Read")
    # Nothing is written until the sign-in lands.
    assert not ms_config.path.exists()
    auth_server.approve()
    done = _finish(tools["finish_msteams_auth"])
    assert done == {"ok": True, "message": "MS Teams credentials saved (Muse-auth)."}
    assert json.loads(ms_config.path.read_text()) == {
        "tenant_id": _MS_TENANT,
        "client_id": "app-1",
        "bot_id": "B1",
    }
    entry = _vault_entry("msteams")["authorized_user_info"]
    assert entry["kind"] == "oauth2_refresh_token"
    assert entry["token_url"] == f"{auth_server.base()}/{_MS_TENANT}/oauth2/v2.0/token"
    assert entry["client_id"] == "app-1"
    assert entry["token_scope"] == device["form"]["scope"]
    assert agent._backend._muse is True and agent._backend._bot_id == "B1"
    # The scratch validation entry is gone; only the live one remains.
    vault_files = sorted(p.name for p in (muse_auth_dir() / "vault").glob("*.json"))
    assert vault_files == ["msteams.json"]
    checked = json.loads(tools["check_msteams_auth"]())
    assert checked == {"ok": True, "message": "MS Teams authenticated (Muse-auth)."}
    # Force a refresh: rewrite the expiry (test-side) and read again.
    entry_file = muse_auth_dir() / "vault" / "msteams.json"
    payload = json.loads(entry_file.read_text())
    payload["authorized_user_info"]["expires_at"] = time.time() - 1
    entry_file.write_text(json.dumps(payload))
    assert json.loads(tools["check_msteams_auth"]())["ok"] is True
    refresh = next(
        r["form"] for r in auth_server.requests if r["form"].get("grant_type") == "refresh_token"
    )
    assert refresh["client_id"] == "app-1" and refresh["scope"] == device["form"]["scope"]
    assert "client_secret" not in refresh
    assert tools["clear_msteams_auth"]() == "MS Teams authentication cleared."


def test_msteams_device_code_requires_refresh_token_and_muse(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without offline_access (no refresh token) the sign-in is rejected."""
    monkeypatch.setenv("MSTEAMS_LOGIN_BASE", auth_server.base())
    agent = MSTeamsAgent()
    agent._backend._graph_base = f"{auth_server.base()}/v1.0"
    tools = auth_tools(agent)
    assert json.loads(tools["authenticate_msteams"](_MS_TENANT, "app-1"))["status"] == (
        "consent_required"
    )
    auth_server.approve()
    result = _finish(tools["finish_msteams_auth"])
    assert result["ok"] is False and "offline_access" in result["error"]
    assert not vault_has_credentials("msteams")
    # The tenant rides on the session, so a config.json edited while the
    # sign-in is pending cannot redirect the finish step.
    assert json.loads(tools["authenticate_msteams"](_MS_TENANT, "app-1"))["status"] == (
        "consent_required"
    )
    ms_config.path.parent.mkdir(parents=True, exist_ok=True)
    ms_config.path.write_text(json.dumps({"tenant_id": "bad/tenant", "client_id": "app-1"}))
    auth_server.approve()
    result = _finish(tools["finish_msteams_auth"])
    assert result["ok"] is False and "offline_access" in result["error"]
    assert not vault_has_credentials("msteams")


def test_msteams_legacy_mode_rejects_device_code(
    isolated_kiss_home: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy mode has no vault to refresh in; the tool says so and does nothing."""
    monkeypatch.setenv("MSTEAMS_LOGIN_BASE", auth_server.base())
    agent = MSTeamsAgent()
    tools = auth_tools(agent)
    result = json.loads(tools["authenticate_msteams"](_MS_TENANT, "app-1"))
    assert result["ok"] is False and "KISS_MUSE_AUTH=0" in result["error"]
    assert not auth_server.requests
    assert "msteams" not in ConsentSession._active


# ----------------------------------------------------------------- Nextcloud


def test_nextcloud_login_flow_v2_enrolls_app_password(
    muse_env: Path, nextcloud_server: _NextcloudServer
) -> None:
    """Nextcloud: URL only → login link → grant → app password in the vault."""
    agent = NextcloudTalkAgent()
    tools = auth_tools(agent)
    unauth = tools["check_nextcloud_auth"]()
    assert "authenticate_nextcloud(url=...)" in unauth and "finish_nextcloud_auth" in unauth
    assert tools["authenticate_nextcloud"]("  ") == "url cannot be empty."
    assert json.loads(tools["authenticate_nextcloud"]("https://bad..host"))["ok"] is False
    assert "together" in tools["authenticate_nextcloud"](nextcloud_server.base(), "alice")

    started = json.loads(tools["authenticate_nextcloud"](nextcloud_server.base() + "/"))
    assert started["status"] == "consent_required"
    assert started["verification_uri"] == f"{nextcloud_server.base()}/login/v2/flow/poll-1"
    assert started["user_code"] == ""
    assert "enter the code" not in started["instructions"]
    assert nextcloud_server.requests[0]["headers"]["User-Agent"] == "KISS Sorcar"
    pending = json.loads(tools["finish_nextcloud_auth"]())
    assert pending["status"] == "pending"
    nextcloud_server.grant()
    done = _finish(tools["finish_nextcloud_auth"])
    assert done == {"ok": True, "message": "Nextcloud credentials saved (Muse-auth)."}
    import base64

    expected = "Basic " + base64.b64encode(b"alice@example.com:app-password-poll-1").decode()
    entry = _vault_entry("nextcloud")
    assert entry["authorized_user_info"] == {
        "kind": "header",
        "header": "Authorization",
        "token": expected,
    }
    assert entry["hosts"] == [f"127.0.0.1:{nextcloud_server.server_port}"]
    assert json.loads(nc_config.path.read_text()) == {
        "url": nextcloud_server.base(),
        "username": "alice@example.com",
        "login_flow": "true",
    }
    assert agent._backend._muse is True and agent._backend._auth == ("alice@example.com", "")
    checked = json.loads(tools["check_nextcloud_auth"]())
    assert checked == {"ok": True, "room_count": 1}
    # A password revoked server-side is reported, not hidden behind an
    # empty room list.
    nextcloud_server.app_passwords.clear()
    revoked = json.loads(tools["check_nextcloud_auth"]())
    assert revoked == {"ok": False, "error": "Authentication failed: HTTP 401, OCS statuscode 997"}
    nextcloud_server.app_passwords["app-password-poll-1"] = "alice@example.com"
    # Clearing a login-flow connection revokes its app password on the server.
    assert tools["clear_nextcloud_auth"]() == (
        "Nextcloud authentication cleared; the app password was revoked."
    )
    assert nextcloud_server.revoked == ["app-password-poll-1"]
    assert not vault_has_credentials("nextcloud")
    assert not nc_config.path.exists()


def test_nextcloud_login_flow_failure_paths(
    muse_env: Path, nextcloud_server: _NextcloudServer
) -> None:
    """Servers without Login Flow v2 or with a foreign poll endpoint are refused."""
    agent = NextcloudTalkAgent()
    tools = auth_tools(agent)
    nextcloud_server.disabled = True
    refused = json.loads(tools["authenticate_nextcloud"](nextcloud_server.base()))
    assert refused == {
        "ok": False,
        "error": f"{nextcloud_server.base()} did not offer Nextcloud Login Flow v2 (HTTP 404)",
    }
    nextcloud_server.disabled = False
    nextcloud_server.foreign_endpoint = True
    foreign = json.loads(tools["authenticate_nextcloud"](nextcloud_server.base()))
    assert foreign["ok"] is False and "another origin" in foreign["error"]
    assert "nextcloud" not in ConsentSession._active
    assert json.loads(tools["finish_nextcloud_auth"]()) == {
        "ok": False,
        "error": "Nextcloud sign-in failed: no sign-in in progress; call "
        "authenticate_nextcloud() first",
    }
    # Starting a new flow cancels the previous pending one.
    nextcloud_server.foreign_endpoint = False
    tools["authenticate_nextcloud"](nextcloud_server.base())
    first = ConsentSession._active["nextcloud"]
    tools["authenticate_nextcloud"](nextcloud_server.base())
    second = ConsentSession._active["nextcloud"]
    assert first is not second and first._cancelled is True
    first._thread.join(timeout=3.0)
    assert not first._thread.is_alive()
    # Direct credentials still work (and reject a wrong app password).
    nextcloud_server.app_passwords["manual-pw"] = "bob"
    direct = json.loads(
        tools["authenticate_nextcloud"](nextcloud_server.base(), "bob", "manual-pw")
    )
    assert direct["ok"] is True
    bad = json.loads(tools["authenticate_nextcloud"](nextcloud_server.base(), "bob", "wrong"))
    assert bad == {"ok": False, "error": "Authentication failed: HTTP 401, OCS statuscode 997"}
    # The rejected attempt left bob's working credential in place.
    assert vault_has_credentials("nextcloud")
    assert json.loads(tools["check_nextcloud_auth"]()) == {"ok": True, "room_count": 1}
    # A hand-supplied app password is not revoked when cleared.
    assert tools["clear_nextcloud_auth"]() == "Nextcloud authentication cleared."
    assert nextcloud_server.revoked == []


def test_nextcloud_login_flow_legacy_mode(
    isolated_kiss_home: Path, nextcloud_server: _NextcloudServer
) -> None:
    """With Muse-auth off the app password is written to config.json."""
    agent = NextcloudTalkAgent()
    tools = auth_tools(agent)
    assert json.loads(tools["authenticate_nextcloud"](nextcloud_server.base()))["status"] == (
        "consent_required"
    )
    nextcloud_server.grant()
    done = _finish(tools["finish_nextcloud_auth"])
    assert done == {"ok": True, "message": "Nextcloud credentials saved."}
    assert json.loads(nc_config.path.read_text()) == {
        "url": nextcloud_server.base(),
        "username": "alice@example.com",
        "password": "app-password-poll-1",
        "login_flow": "true",
    }
    assert json.loads(tools["check_nextcloud_auth"]()) == {"ok": True, "room_count": 1}
    # A rejected app password (legacy) reports the failure and leaves
    # the working credential in place.
    bad = json.loads(tools["authenticate_nextcloud"](nextcloud_server.base(), "bob", "nope"))
    assert bad == {"ok": False, "error": "Authentication failed: HTTP 401, OCS statuscode 997"}
    assert agent._is_authenticated() is True
    assert agent._backend._auth == ("alice@example.com", "app-password-poll-1")
    # A server that cannot be reached is reported, not raised.
    down = json.loads(tools["authenticate_nextcloud"]("http://127.0.0.1:9", "bob", "nope"))
    assert down["ok"] is False and "Authentication failed: ConnectionError" in down["error"]
    # Clearing revokes the login-flow password (legacy Basic auth).
    assert tools["clear_nextcloud_auth"]() == (
        "Nextcloud authentication cleared; the app password was revoked."
    )
    assert nextcloud_server.revoked == ["app-password-poll-1"]


# ------------------------------------------------------- daemon validation


def test_daemon_validates_refresh_token_entries(muse_env: Path, auth_server: _AuthServer) -> None:
    """The daemon refuses malformed or unpinned oauth2_refresh_token payloads."""
    good = {
        "kind": "oauth2_refresh_token",
        "token_url": f"{auth_server.base()}/login/oauth/access_token",
        "client_id": "app",
        "access_token": "a",
        "refresh_token": "r",
        "expires_at": time.time() + 100,
    }
    store_credentials("github", good, [])
    assert vault_has_credentials("github")
    with pytest.raises(MuseAuthError, match="unpinned OAuth token endpoint"):
        store_credentials("github", {**good, "token_url": "https://evil.example/token"}, [])
    with pytest.raises(MuseAuthError, match="invalid refresh_token value"):
        store_credentials("github", {**good, "refresh_token": "bad\nvalue"}, [])
    with pytest.raises(MuseAuthError, match="invalid expires_at value"):
        store_credentials("github", {**good, "expires_at": "soon"}, [])
    with pytest.raises(MuseAuthError, match="invalid expires_at value"):
        store_credentials("github", {**good, "expires_at": float("inf")}, [])
    with pytest.raises(MuseAuthError, match="invalid token_scope value"):
        store_credentials("github", {**good, "token_scope": "a\x00b"}, [])
    # Twitch's pinned host is id.twitch.tv; github.com is not accepted there.
    with pytest.raises(MuseAuthError, match="unpinned"):
        store_credentials(
            "twitch", {**good, "token_url": "https://github.com/login/oauth/access_token"}, []
        )
    # A malformed persisted expiry is treated as expired: the daemon refreshes.
    auth_server.expiring = True
    auth_server.refresh_tokens["r"] = "repo"
    entry_file = muse_auth_dir() / "vault" / "github.json"
    payload = json.loads(entry_file.read_text())
    payload["authorized_user_info"]["expires_at"] = 1e308
    entry_file.write_text(json.dumps(payload))
    handle = mint_surrogate("github")
    assert handle is not None
    resp = MuseBoundarySession("github").get(
        f"{auth_server.base()}/user", headers={"Authorization": f"Bearer {handle.token}"}
    )
    assert resp.status_code == 200 and auth_server.refreshes == 1


# ------------------------------------------------------- session mechanics


def test_device_session_expiry_and_bounds(auth_server: _AuthServer) -> None:
    """A never-approved flow expires; announced lifetimes are clamped."""
    provider = DeviceFlowProvider(
        device_url=f"{auth_server.base()}/login/device/code",
        token_url=f"{auth_server.base()}/login/oauth/access_token",
    )
    session = DeviceFlowSession("github", provider, "app", "repo")
    # Shorten the deadline so the expiry branch runs quickly.
    session._deadline = time.monotonic() + 1.5
    session.register()
    session._thread.join(timeout=10.0)
    assert session.result is None
    assert session.error == "the sign-in request expired before it was approved"
    assert ConsentSession.finish("github") == (None, session.error)
    assert _device_auth._bounded("nan", 5.0, 1.0, 60.0) == 5.0
    assert _device_auth._bounded(float("inf"), 5.0, 1.0, 60.0) == 5.0
    assert _device_auth._bounded(0, 5.0, 1.0, 60.0) == 1.0
    assert _device_auth._bounded(10**9, 5.0, 1.0, 60.0) == 60.0
    # A device answer without a verification URI is a refusal.
    auth_server.omit_code = True
    with pytest.raises(RuntimeError, match="device authorization refused \\(HTTP 200\\)"):
        DeviceFlowSession("github", provider, "app", "repo")


def test_token_grant_and_instructions() -> None:
    """TokenGrant maps token bodies to vault payloads; instructions mention the code."""
    grant = TokenGrant.from_response(
        {"access_token": "a", "refresh_token": "r", "expires_in": "oops", "scope": ["x", "y"]}
    )
    assert grant.scope == "x y" and grant.expires_in == 0.0
    info = grant.vault_credential("https://id.twitch.tv/oauth2/token", "cid", refresh_scope="x y")
    assert info["kind"] == "oauth2_refresh_token" and info["token_scope"] == "x y"
    assert 3500 < info["expires_at"] - time.time() <= 3600
    plain = TokenGrant.from_response({"access_token": "a", "expires_in": 100})
    assert plain.vault_credential("https://x/t", "cid") == {"kind": "bearer", "token": "a"}

    session = NextcloudLoginSession.__new__(NextcloudLoginSession)
    ConsentSession.__init__(session, "nextcloud", 120.0, 1.0)
    session.verification_uri = "https://cloud.example/login/v2/flow/abc"
    text = consent_instructions("nextcloud", "Nextcloud", session)
    assert "https://cloud.example/login/v2/flow/abc" in text
    assert "OWN browser" in text and "finish_nextcloud_auth()" in text
    assert "password" in text and "enter the code" not in text
    assert "about 2 minutes" in text
    session.user_code = "WDJB-MJHT"
    assert "enter the code WDJB-MJHT" in consent_instructions("nextcloud", "Nextcloud", session)


# ------------------------------------------------------- prompt contract


@pytest.mark.parametrize(
    ("agent_cls", "service"),
    [
        (GitHubAgent, "github"),
        (TwitchAgent, "twitch"),
        (MSTeamsAgent, "msteams"),
        (NextcloudTalkAgent, "nextcloud"),
    ],
    ids=["github", "twitch", "msteams", "nextcloud"],
)
def test_channel_prompt_pins_connect_hand_off(
    isolated_kiss_home: Path, agent_cls: type[Any], service: str
) -> None:
    """Every Connect-style agent's prompt names real tools and the safe hand-off."""
    import re

    prompt = agent_cls.channel_system_prompt
    for phrase in (
        f"check_{service}_auth()",
        f"finish_{service}_auth()",
        "consent_required",
        "OWN browser",
        "never ask for or type the user's password or 2FA code",
        "Do NOT open the URL or any sign-in page in your built-in browser",
        "Nothing is pasted back",
        "Muse app",
    ):
        assert phrase in prompt, f"{service}: missing {phrase!r}"
    for banned in ("curl", "paste back the", "stuck on login or captcha", "autonomously"):
        assert banned not in prompt, f"{service}: stale wording {banned!r}"
    tools = {t.__name__ for t in agent_cls()._get_auth_tools()}
    named = set(re.findall(r"\b((?:check|authenticate|finish|clear)_[a-z_]+)\(", prompt))
    assert named and named <= tools, f"{service}: prompt names unknown tools {named - tools}"
    assert {f"check_{service}_auth", f"authenticate_{service}", f"finish_{service}_auth"} <= tools


# ---------------------------------------------------- remaining branches


def test_daemon_refresh_edge_cases(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tampered expiry strings, odd expires_in and non-rotating providers."""
    monkeypatch.setenv("GITHUB_OAUTH_BASE", auth_server.base())
    auth_server.expiring = True
    auth_server.rotate = False
    auth_server.bad_expires_in = True
    agent = GitHubAgent()
    agent._backend._base_url = auth_server.base()
    tools = auth_tools(agent)
    tools["authenticate_github"](client_id="kiss-app")
    auth_server.approve()
    assert _finish(tools["finish_github_auth"])["ok"] is True
    entry_file = muse_auth_dir() / "vault" / "github.json"
    info = json.loads(entry_file.read_text())["authorized_user_info"]
    # An unparseable expires_in from the grant defaults to one hour.
    assert 3500 < info["expires_at"] - time.time() <= 3600
    assert auth_server.refreshes == 0
    # A non-numeric persisted expiry counts as expired → refresh; the
    # provider answers without a new refresh token, so the old one stays.
    payload = json.loads(entry_file.read_text())
    payload["authorized_user_info"]["expires_at"] = "soon"
    entry_file.write_text(json.dumps(payload))
    data, error = agent._backend._api("GET", "/user")
    assert error == "" and data["login"] == "octocat"
    assert auth_server.refreshes == 1
    refreshed = json.loads(entry_file.read_text())["authorized_user_info"]
    assert refreshed["refresh_token"] == info["refresh_token"] == "refresh-1"
    assert refreshed["access_token"] == "access-2"
    assert 3500 < refreshed["expires_at"] - time.time() <= 3600


def test_device_flow_complete_uri_and_non_json_answers(auth_server: _AuthServer) -> None:
    """verification_uri_complete pre-fills the code (still shown); non-JSON = refusal."""
    provider = DeviceFlowProvider(
        device_url=f"{auth_server.base()}/login/device/code",
        token_url=f"{auth_server.base()}/login/oauth/access_token",
    )
    auth_server.complete_uri = True
    session = DeviceFlowSession("github", provider, "app", "repo")
    session.cancel()
    assert session.verification_uri.startswith("https://github.com/login/device?user_code=CODE-")
    # RFC 8628 section 3.3.1: the code is still displayed for confirmation.
    assert session.user_code == "CODE-1" and session.code_prefilled is True
    assert "confirm the code shown is CODE-1" in consent_instructions("github", "GitHub", session)
    auth_server.plain_text_device = True
    with pytest.raises(RuntimeError, match=r"device authorization refused \(HTTP 503\)"):
        DeviceFlowSession("github", provider, "app", "repo")


def test_nextcloud_non_json_and_poll_error(nextcloud_server: _NextcloudServer) -> None:
    """A non-JSON login answer is refused; a failing poll ends the session."""
    nextcloud_server.plain_text = True
    with pytest.raises(RuntimeError, match=r"did not offer Nextcloud Login Flow v2 \(HTTP 200\)"):
        NextcloudLoginSession("nextcloud", nextcloud_server.base())
    nextcloud_server.plain_text = False
    nextcloud_server.poll_error = True
    session = NextcloudLoginSession("nextcloud", nextcloud_server.base())
    session.register()
    session._thread.join(timeout=10.0)
    assert ConsentSession.finish("nextcloud") == (
        None,
        "Nextcloud login poll failed (HTTP 500)",
    )


def test_failure_paths_after_approval(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused clients, missing sessions, vault refusals and API rejections."""
    monkeypatch.setenv("TWITCH_OAUTH_BASE", auth_server.base())
    monkeypatch.setenv("MSTEAMS_LOGIN_BASE", auth_server.base())
    monkeypatch.setenv("GITHUB_OAUTH_BASE", auth_server.base())
    twitch = TwitchAgent()
    twitch._backend._helix_base = f"{auth_server.base()}/helix"
    tw_tools = auth_tools(twitch)
    ms_tools = auth_tools(MSTeamsAgent())
    assert json.loads(tw_tools["authenticate_twitch"]("rejected-client")) == {
        "ok": False,
        "error": "device authorization refused (invalid_client: unknown app)",
    }
    assert json.loads(ms_tools["authenticate_msteams"](_MS_TENANT, "rejected-client")) == {
        "ok": False,
        "error": "device authorization refused (invalid_client: unknown app)",
    }
    assert "no sign-in in progress" in json.loads(tw_tools["finish_twitch_auth"]())["error"]
    assert "no sign-in in progress" in json.loads(ms_tools["finish_msteams_auth"]())["error"]

    # Twitch: the API rejects the freshly issued token → nothing enrolled.
    tw_tools["authenticate_twitch"]("cid1")
    session = ConsentSession._active["twitch"]
    auth_server.approve()
    session._thread.join(timeout=10.0)
    auth_server.access_tokens.clear()
    rejected = _finish(tw_tools["finish_twitch_auth"])
    assert rejected == {"ok": False, "error": "Twitch rejected the new token: HTTP 401"}
    assert not vault_has_credentials("twitch") and twitch._is_authenticated() is False
    assert not tw_config.path.exists()

    # GitHub: a malformed token value fails the probe → no enrollment.
    auth_server.bad_access_token = True
    github = GitHubAgent()
    github._backend._base_url = auth_server.base()
    gh_tools = auth_tools(github)
    gh_tools["authenticate_github"](client_id="kiss-app")
    auth_server.approve()
    result = _finish(gh_tools["finish_github_auth"])
    assert result == {
        "ok": False,
        "error": "GitHub rejected the new token: InvalidHeader while validating the token",
    }
    assert not vault_has_credentials("github") and github._is_authenticated() is False


def test_twitch_legacy_failure_paths(
    isolated_kiss_home: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy mode reports API rejections and unreachable servers."""
    monkeypatch.setenv("TWITCH_OAUTH_BASE", auth_server.base())
    agent = TwitchAgent()
    agent._backend._helix_base = f"{auth_server.base()}/helix"
    tools = auth_tools(agent)
    bad = json.loads(tools["authenticate_twitch"]("cid1", "", "unknown-token"))
    assert bad["ok"] is False and "InvalidAuthenticationToken" in bad["error"]
    agent._backend._helix_base = "http://127.0.0.1:9/helix"
    down = json.loads(tools["authenticate_twitch"]("cid1", "", "unknown-token"))
    assert down["ok"] is False and "Connection" in down["error"]
    assert not tw_config.path.exists()


# ------------------------------------------------- review-driven regressions


class _ScriptedSession(ConsentSession):
    """A consent session whose polls are scripted by the test."""

    def __init__(self, service: str, steps: list[Any], interval: float = 0.2) -> None:
        super().__init__(service, 60.0, interval)
        self.steps = list(steps)
        self.polls = 0

    def _poll_once(self) -> dict[str, Any] | None:
        self.polls += 1
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step()  # type: ignore[no-any-return]
        return step  # type: ignore[no-any-return]


def test_session_registry_is_race_safe_and_backs_off_on_timeouts() -> None:
    """A newer session survives a stale finish; timeouts slow the poll down."""
    # Compare-and-pop: the stale session A completes while finish() waits
    # on it, but session B took A's place meanwhile → A is not handed out.
    # Inexhaustibly pending: even if this thread stalls >= first's 0.3 s
    # interval and finish() snapshots ``late``, it stays pending throughout.
    late = _ScriptedSession("scripted", [None] * 200)

    def approve_after_replacement() -> dict[str, Any]:
        late.register()
        return {"access_token": "stale"}

    first = _ScriptedSession("scripted", [approve_after_replacement], 0.3)
    first.register()
    assert ConsentSession.finish("scripted", wait_seconds=10.0) == (None, "pending")
    assert first.result == {"access_token": "stale"} and not first._thread.is_alive()
    assert ConsentSession._active["scripted"] is late
    ConsentSession.cancel_active("scripted")
    late._thread.join(timeout=5.0)
    assert not late._thread.is_alive() and "scripted" not in ConsentSession._active

    # RFC 8628 section 3.5: a connection timeout is not terminal; the
    # interval grows by 5 s (capped) and polling continues.
    flaky = _ScriptedSession(
        "flaky",
        [requests.ConnectionError("boom"), requests.Timeout("slow"), {"access_token": "t"}],
        0.1,
    )
    flaky._interval = 0.1
    flaky.register()
    flaky._thread.join(timeout=30.0)
    assert flaky.result == {"access_token": "t"} and flaky.polls == 3
    assert flaky._interval == pytest.approx(10.1)
    assert flaky.result_at > 0
    got, status = ConsentSession.finish("flaky")
    assert got is flaky and status == "ok"

    # Any other exception ends the session with its message.
    broken = _ScriptedSession("broken", [RuntimeError("sign-in refused (access_denied)")])
    broken.register()
    broken._thread.join(timeout=10.0)
    assert ConsentSession.finish("broken") == (None, "sign-in refused (access_denied)")


def test_token_lifetime_counts_from_acquisition_and_origins_ignore_default_ports() -> None:
    """A late finish must not extend the access token; default ports compare equal."""
    session = _ScriptedSession(
        "late", [{"access_token": "a", "refresh_token": "r", "expires_in": 100}], 0.1
    )
    session.register()
    session._thread.join(timeout=10.0)
    time.sleep(1.2)
    got, status = ConsentSession.finish("late")
    assert got is session and status == "ok"
    grant = TokenGrant.from_session(session)  # type: ignore[arg-type]
    assert grant.acquired_at == session.result_at
    info = grant.vault_credential("https://id.example/token", "cid")
    assert info["expires_at"] == pytest.approx(session.result_at + 100)
    assert info["expires_at"] < time.time() + 100 - 1.0

    origin = _device_auth._origin
    assert origin("https://cloud.example.com") == origin("https://cloud.example.com:443/x")
    assert origin("http://cloud.example.com") == origin("HTTP://Cloud.Example.com:80")
    assert origin("https://cloud.example.com") != origin("https://cloud.example.com:8443")
    assert origin("https://cloud.example.com") != origin("http://cloud.example.com")


def test_github_sign_in_never_disturbs_the_working_credential(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starting, failing or superseding a sign-in leaves the current token intact."""
    monkeypatch.setenv("GITHUB_OAUTH_BASE", auth_server.base())
    auth_server.access_tokens["ghp_old"] = "repo"
    agent = GitHubAgent()
    agent._backend._base_url = auth_server.base()
    tools = auth_tools(agent)
    assert json.loads(tools["authenticate_github"]("ghp_old", read_only=True))["ok"] is True
    assert json.loads(tools["check_github_auth"]()) == {"ok": True, "read_only": True}

    # Starting a sign-in changes nothing but the remembered client ID.
    started = json.loads(tools["authenticate_github"](client_id="kiss-app"))
    assert started["status"] == "consent_required"
    assert json.loads(gh_config.path.read_text()) == {
        "read_only": "true",
        "oauth_client_id": "kiss-app",
    }
    assert _vault_entry("github")["authorized_user_info"] == {"kind": "bearer", "token": "ghp_old"}
    data, error = agent._backend._api("GET", "/user")
    assert error == "" and data["login"] == "octocat"

    # The API rejects the new token → the old one stays enrolled and usable.
    auth_server.approve()
    ConsentSession._active["github"]._thread.join(timeout=10.0)
    auth_server.access_tokens.pop("access-1")
    rejected = _finish(tools["finish_github_auth"])
    assert rejected == {"ok": False, "error": "GitHub rejected the new token: HTTP 401"}
    assert _vault_entry("github")["authorized_user_info"] == {"kind": "bearer", "token": "ghp_old"}
    assert json.loads(tools["check_github_auth"]()) == {"ok": True, "read_only": True}

    # A hand-supplied token supersedes a pending sign-in: the late approval
    # has nowhere to land.
    assert json.loads(tools["authenticate_github"](client_id="kiss-app"))["status"] == (
        "consent_required"
    )
    pending = ConsentSession._active["github"]
    auth_server.access_tokens["ghp_new"] = "repo"
    assert json.loads(tools["authenticate_github"]("ghp_new"))["ok"] is True
    assert pending._cancelled is True and "github" not in ConsentSession._active
    auth_server.approve()
    assert "no sign-in in progress" in json.loads(tools["finish_github_auth"]())["error"]
    assert _vault_entry("github")["authorized_user_info"] == {"kind": "bearer", "token": "ghp_new"}
    assert json.loads(tools["check_github_auth"]()) == {"ok": True, "read_only": False}
    # The direct path never writes the plaintext token into the config:
    # it goes straight into the vault (atomic replace).
    assert "token" not in json.loads(gh_config.path.read_text())

    # A direct token the daemon refuses (embedded control character) must
    # not disturb the enrolled credential or the config: store() replaces
    # atomically, so there is no clear-then-store window.  (Regression:
    # the old path cleared the vault before migrating the candidate, so a
    # refused candidate lost the working credential.)
    bad = json.loads(tools["authenticate_github"]("bad\ttoken"))
    assert bad["ok"] is False and "failed to save GitHub config" in bad["error"]
    assert _vault_entry("github")["authorized_user_info"] == {"kind": "bearer", "token": "ghp_new"}
    assert json.loads(tools["check_github_auth"]()) == {"ok": True, "read_only": False}
    assert "token" not in json.loads(gh_config.path.read_text())

    # A successful sign-in applies the options chosen when it was started.
    assert json.loads(tools["authenticate_github"](read_only=True))["status"] == (
        "consent_required"
    )
    auth_server.approve()
    done = _finish(tools["finish_github_auth"])
    assert done["ok"] is True and done["read_only"] is True
    assert json.loads(gh_config.path.read_text()) == {
        "read_only": "true",
        "oauth_client_id": "kiss-app",
    }
    assert _vault_entry("github")["authorized_user_info"] == {"kind": "bearer", "token": "access-2"}


def test_device_polling_ignores_ambient_proxy_settings(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A proxy in the environment never sees the device code or the token."""
    monkeypatch.setenv("GITHUB_OAUTH_BASE", auth_server.base())
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    agent = GitHubAgent()
    agent._backend._base_url = auth_server.base()
    tools = auth_tools(agent)
    assert json.loads(tools["authenticate_github"](client_id="kiss-app"))["status"] == (
        "consent_required"
    )
    auth_server.approve()
    done = _finish(tools["finish_github_auth"])
    assert done["ok"] is True and done["login"] == "octocat"


def test_msteams_graph_permission_verdicts_and_superseding(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 401 from Graph rejects the token; 403 proves the exchange; secrets supersede."""
    monkeypatch.setenv("MSTEAMS_LOGIN_BASE", auth_server.base())
    auth_server.expiring = True
    auth_server.graph_status = 401
    agent = MSTeamsAgent()
    agent._backend._graph_base = f"{auth_server.base()}/v1.0"
    tools = auth_tools(agent)
    assert json.loads(tools["authenticate_msteams"](_MS_TENANT, "app-1"))["status"] == (
        "consent_required"
    )
    auth_server.approve()
    result = _finish(tools["finish_msteams_auth"])
    assert result == {
        "ok": False,
        "error": "Microsoft Graph rejected the acquired token (HTTP 401)",
    }
    assert not vault_has_credentials("msteams") and not ms_config.path.exists()
    assert agent._is_authenticated() is False
    # A 403 (token fine, permission missing) counts as proof of the exchange.
    auth_server.graph_status = 403
    assert json.loads(tools["authenticate_msteams"](_MS_TENANT, "app-1"))["status"] == (
        "consent_required"
    )
    auth_server.approve()
    assert _finish(tools["finish_msteams_auth"])["ok"] is True
    assert vault_has_credentials("msteams")
    # App credentials supersede a pending sign-in.
    assert json.loads(tools["authenticate_msteams"](_MS_TENANT, "app-1"))["status"] == (
        "consent_required"
    )
    pending = ConsentSession._active["msteams"]
    tools["authenticate_msteams"](_MS_TENANT, "app-1", client_secret="s3cret")
    assert pending._cancelled is True and "msteams" not in ConsentSession._active


def test_twitch_token_supersedes_pending_sign_in(
    muse_env: Path, auth_server: _AuthServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-supplied Twitch token cancels the pending device-code session."""
    monkeypatch.setenv("TWITCH_OAUTH_BASE", auth_server.base())
    agent = TwitchAgent()
    agent._backend._helix_base = f"{auth_server.base()}/helix"
    tools = auth_tools(agent)
    assert json.loads(tools["authenticate_twitch"]("cid1"))["status"] == "consent_required"
    pending = ConsentSession._active["twitch"]
    auth_server.access_tokens["direct"] = "clips:edit"
    # A hand-supplied token is probed before it is stored.
    bad = json.loads(tools["authenticate_twitch"]("cid1", access_token="unknown"))
    assert bad == {"ok": False, "error": "Twitch rejected the token: HTTP 401"}
    assert not vault_has_credentials("twitch")
    direct = json.loads(tools["authenticate_twitch"]("cid1", access_token="direct"))
    assert direct == {
        "ok": True,
        "message": "Twitch credentials saved (Muse-auth).",
        "login": "kisscaster",
    }
    assert pending._cancelled is True and "twitch" not in ConsentSession._active
    assert _vault_entry("twitch")["authorized_user_info"] == {"kind": "bearer", "token": "direct"}

    # A rejected browser sign-in leaves the working token untouched.
    assert json.loads(tools["authenticate_twitch"]("cid1"))["status"] == "consent_required"
    auth_server.approve(auth_server.pending_codes()[-1])  # the cancelled flow's code lingers
    ConsentSession._active["twitch"]._thread.join(timeout=10.0)
    auth_server.access_tokens.clear()
    rejected = _finish(tools["finish_twitch_auth"])
    assert rejected == {"ok": False, "error": "Twitch rejected the new token: HTTP 401"}
    assert _vault_entry("twitch")["authorized_user_info"] == {"kind": "bearer", "token": "direct"}
    auth_server.access_tokens["direct"] = "clips:edit"
    assert json.loads(tools["check_twitch_auth"]()) == {"ok": True, "login": "kisscaster"}


def test_nextcloud_uses_the_server_url_the_login_flow_reports(
    muse_env: Path, nextcloud_server: _NextcloudServer
) -> None:
    """The credential is bound to the canonical server URL from the poll answer."""
    nextcloud_server.reported_server = f"{nextcloud_server.base()}/cloud/"
    agent = NextcloudTalkAgent()
    tools = auth_tools(agent)
    assert json.loads(tools["authenticate_nextcloud"](nextcloud_server.base()))["status"] == (
        "consent_required"
    )
    nextcloud_server.grant()
    assert _finish(tools["finish_nextcloud_auth"])["ok"] is True
    assert json.loads(nc_config.path.read_text())["url"] == f"{nextcloud_server.base()}/cloud"
    assert agent._backend._url == f"{nextcloud_server.base()}/cloud"
    assert json.loads(tools["check_nextcloud_auth"]()) == {"ok": True, "room_count": 1}
    room_reads = [r["path"] for r in nextcloud_server.requests if r["path"].endswith("/room")]
    assert room_reads and all(p.startswith("/cloud/ocs/") for p in room_reads)
    # An unusable reported server falls back to the URL the user gave.
    session = NextcloudLoginSession.__new__(NextcloudLoginSession)
    ConsentSession.__init__(session, "nextcloud", 120.0, 1.0)
    session.base_url = "https://cloud.example.com"
    for bad in ("", "ftp://cloud.example.com", "not a url", "https:///x"):
        session.result = {"server": bad}
        assert session.server_url() == "https://cloud.example.com"
    session.result = {"server": "https://Cloud.example.com/nextcloud/"}
    assert session.server_url() == "https://Cloud.example.com/nextcloud"


# ------------------------------------------------------------------- Matrix


class _MatrixCredentials:
    """nio ``AsyncClient`` data contract (matrix-nio is not installed here)."""

    def __init__(self, homeserver: str, access_token: str) -> None:
        self.homeserver = homeserver
        self.access_token = access_token
        self.user_id = ""
        self.device_id = ""


def test_matrix_oauth_device_flow_registers_client_and_refreshes(
    isolated_kiss_home: Path, auth_server: _AuthServer
) -> None:
    """Matrix: discovery → dynamic registration → device grant → self-refresh → revoke."""
    auth_server.expiring = True
    auth_server.expires_in = 300
    agent = MatrixAgent()
    tools = auth_tools(agent)
    unauth = tools["check_matrix_auth"]()
    assert "finish_matrix_auth" in unauth and "OWN browser" in unauth
    assert tools["authenticate_matrix"]("  ") == "homeserver_url cannot be empty."
    assert json.loads(tools["authenticate_matrix"]("not a url"))["ok"] is False

    started = json.loads(tools["authenticate_matrix"](auth_server.base() + "/"))
    assert started["status"] == "consent_required"
    assert started["verification_uri"] == "https://account.example/link"
    assert started["user_code"] == "CODE-1"
    # Public client registered with the spec's metadata; no config yet.
    assert len(auth_server.matrix_registrations) == 1
    registration = auth_server.matrix_registrations[0]
    assert registration["token_endpoint_auth_method"] == "none"
    assert registration["grant_types"] == [_device_auth.DEVICE_CODE_GRANT, "refresh_token"]
    assert registration["client_uri"] == "https://kisssorcar.github.io/"
    assert "redirect_uris" not in registration
    device = next(r for r in auth_server.requests if r["path"] == "/mas/oauth2/device")
    assert device["form"]["client_id"] == "mas-client-1"
    scope_tokens = device["form"]["scope"].split()
    assert scope_tokens[0] == "urn:matrix:client:api:*"
    assert scope_tokens[1].startswith("urn:matrix:client:device:")
    device_id = scope_tokens[1].rsplit(":", 1)[1]
    assert len(device_id) == 10 and device_id.isalnum()
    assert not mx_config.path.exists()

    assert json.loads(tools["finish_matrix_auth"]())["status"] == "pending"
    auth_server.approve()
    done = _finish(tools["finish_matrix_auth"])
    assert done["ok"] is True and done["user_id"] == "@alice:example.org"
    assert done["device_id"] == device_id
    assert "matrix-nio is not installed" in done["warning"]
    cfg = json.loads(mx_config.path.read_text())
    assert cfg["access_token"] == "access-1" and cfg["refresh_token"] == "refresh-1"
    assert cfg["oauth_client_id"] == "mas-client-1"
    assert cfg["token_url"] == f"{auth_server.base()}/mas/oauth2/token"
    assert cfg["revocation_url"] == f"{auth_server.base()}/mas/oauth2/revoke"
    assert "oauth_issuer" not in cfg
    assert cfg["device_id"] == device_id and cfg["user_id"] == "@alice:example.org"
    assert 200 < float(cfg["expires_at"]) - time.time() <= 300

    # The backend renews the short-lived token by itself before it expires.
    backend = MatrixChannelBackend()
    backend._client = _MatrixCredentials(auth_server.base(), cfg["access_token"])
    backend.refresh_if_needed()
    assert backend._client.access_token == "access-1" and auth_server.refreshes == 0
    cfg["expires_at"] = str(time.time() + 30)  # inside the skew
    mx_config.save(cfg)

    async def _noop() -> str:
        return "ran"

    assert backend._run(_noop()) == "ran"
    assert auth_server.refreshes == 1
    refresh = next(
        r for r in auth_server.requests if r["form"].get("grant_type") == "refresh_token"
    )
    assert refresh["form"] == {
        "grant_type": "refresh_token",
        "refresh_token": "refresh-1",
        "client_id": "mas-client-1",
    }
    assert backend._client.access_token == "access-2"
    renewed = json.loads(mx_config.path.read_text())
    assert renewed["access_token"] == "access-2" and renewed["refresh_token"] == "refresh-2"
    assert float(renewed["expires_at"]) > time.time() + 200

    # Matrix spec: every authorisation flow registers the client anew.
    assert json.loads(tools["authenticate_matrix"](auth_server.base()))["status"] == (
        "consent_required"
    )
    assert len(auth_server.matrix_registrations) == 2
    ConsentSession.cancel_active("matrix")

    # Clearing revokes the session on the authorization server.
    assert tools["clear_matrix_auth"]() == "Matrix authentication cleared; the session was revoked."
    assert auth_server.revoked == ["access-2"]
    assert not mx_config.path.exists()


def test_matrix_refresh_failures_follow_the_spec(
    isolated_kiss_home: Path, auth_server: _AuthServer
) -> None:
    """5xx/network → keep the old token and retry later; 4xx → logged out."""
    mx_config.save(
        {
            "homeserver_url": auth_server.base(),
            "access_token": "old",
            "refresh_token": "refresh-x",
            "expires_at": "0",
            "token_url": f"{auth_server.base()}/mas/oauth2/token",
            "oauth_client_id": "mas-client-9",
        }
    )
    backend = MatrixChannelBackend()
    backend._client = _MatrixCredentials(auth_server.base(), "old")
    # The emulator refuses unknown refresh tokens with 400 invalid_grant.
    auth_server.refresh_error = "server_error"
    auth_server.refresh_status = 503

    async def _noop() -> str:
        return "ran"

    assert backend._run(_noop()) == "ran"  # 5xx: session kept
    assert backend._client.access_token == "old"
    backend2 = MatrixChannelBackend()
    backend2._client = _MatrixCredentials("http://127.0.0.1:9", "old")
    cfg = mx_config.load() or {}
    cfg["token_url"] = "http://127.0.0.1:9/token"
    mx_config.save(cfg)
    assert backend2._run(_noop()) == "ran"  # unreachable: session kept
    cfg["token_url"] = f"{auth_server.base()}/mas/oauth2/token"
    mx_config.save(cfg)
    # A 4xx without a JSON body is still a logout, not a transient failure.
    auth_server.refresh_error = "plain"
    auth_server.refresh_status = 400
    with pytest.raises(RuntimeError, match=r"logged out \(HTTP 400\)"):
        backend._run(_noop())
    auth_server.refresh_error = ""
    with pytest.raises(RuntimeError, match=r"logged out \(invalid_grant\)"):
        backend._run(_noop())
    # A legacy (hand-supplied) token never refreshes.
    mx_config.save({"homeserver_url": auth_server.base(), "access_token": "legacy"})
    assert backend._run(_noop()) == "ran"
    assert auth_server.refreshes == 3


def test_matrix_refresh_never_resurrects_a_cleared_or_replaced_session(
    isolated_kiss_home: Path, auth_server: _AuthServer
) -> None:
    """A clear (or direct re-auth) that lands mid-refresh wins; minted tokens are revoked."""
    auth_server.expiring = True
    auth_server.refresh_tokens["refresh-old"] = "urn:matrix:client:api:*"
    auth_server.refresh_hook = lambda: mx_config.clear()
    mx_config.save(
        {
            "homeserver_url": auth_server.base(),
            "access_token": "old",
            "refresh_token": "refresh-old",
            "expires_at": "0",
            "token_url": f"{auth_server.base()}/mas/oauth2/token",
            "oauth_client_id": "mas-client-9",
            "revocation_url": f"{auth_server.base()}/mas/oauth2/revoke",
        }
    )
    backend = MatrixChannelBackend()
    backend._client = _MatrixCredentials(auth_server.base(), "old")
    backend.refresh_if_needed()
    assert not mx_config.path.exists()
    assert backend._client.access_token == "old"
    assert auth_server.revoked == ["access-1"]
    # Replaced by a direct re-authentication mid-refresh: the new config stays.
    auth_server.refresh_tokens["refresh-old2"] = "urn:matrix:client:api:*"
    replacement = {"homeserver_url": auth_server.base(), "access_token": "hand-supplied"}
    auth_server.refresh_hook = lambda: mx_config.save(replacement)
    mx_config.save(
        {
            "homeserver_url": auth_server.base(),
            "access_token": "old2",
            "refresh_token": "refresh-old2",
            "expires_at": "0",
            "token_url": f"{auth_server.base()}/mas/oauth2/token",
            "oauth_client_id": "mas-client-9",
        }
    )
    backend.refresh_if_needed()
    assert mx_config.load() == replacement
    assert auth_server.revoked == ["access-1"]  # no revocation URL configured this time


def test_matrix_without_oauth_api_and_failure_paths(
    isolated_kiss_home: Path, auth_server: _AuthServer
) -> None:
    """No OAuth API → explanatory error; refusals and rejected tokens enroll nothing."""
    agent = MatrixAgent()
    tools = auth_tools(agent)
    auth_server.matrix_oauth = False
    result = json.loads(tools["authenticate_matrix"](auth_server.base()))
    assert result["ok"] is False and "does not offer the Matrix OAuth 2.0 API" in result["error"]
    assert "access_token=..." in result["error"]
    down = json.loads(tools["authenticate_matrix"]("http://127.0.0.1:9"))
    assert down["ok"] is False and "ConnectionError" in down["error"]
    auth_server.matrix_oauth = True
    assert "no sign-in in progress" in json.loads(tools["finish_matrix_auth"]())["error"]

    # The homeserver rejects the freshly issued token → nothing stored.
    assert json.loads(tools["authenticate_matrix"](auth_server.base()))["status"] == (
        "consent_required"
    )
    auth_server.approve()
    ConsentSession._active["matrix"]._thread.join(timeout=10.0)
    auth_server.access_tokens.clear()
    rejected = _finish(tools["finish_matrix_auth"])
    assert rejected == {
        "ok": False,
        "error": "the homeserver rejected the new token: M_UNKNOWN_TOKEN",
    }
    assert not mx_config.path.exists()

    # Denied consent.
    assert json.loads(tools["authenticate_matrix"](auth_server.base()))["status"] == (
        "consent_required"
    )
    auth_server.deny()
    assert _finish(tools["finish_matrix_auth"]) == {
        "ok": False,
        "error": "Matrix sign-in failed: sign-in refused (access_denied)",
    }

    # An insecure (http) revocation endpoint never receives a token.
    auth_server.matrix_insecure_revocation = True
    auth_server.expiring = True
    assert json.loads(tools["authenticate_matrix"](auth_server.base()))["status"] == (
        "consent_required"
    )
    auth_server.approve()
    assert _finish(tools["finish_matrix_auth"])["ok"] is True
    assert json.loads(mx_config.path.read_text())["revocation_url"] == ""
    assert tools["clear_matrix_auth"]() == "Matrix authentication cleared."
    assert auth_server.revoked == []
    auth_server.matrix_insecure_revocation = False

    # A hand-supplied token supersedes a pending sign-in (legacy path; nio
    # is not installed here, so the pre-existing ImportError surfaces).
    assert json.loads(tools["authenticate_matrix"](auth_server.base()))["status"] == (
        "consent_required"
    )
    pending = ConsentSession._active["matrix"]
    legacy = json.loads(tools["authenticate_matrix"](auth_server.base(), "tok"))
    assert legacy["ok"] is False and "nio" in legacy["error"]
    assert pending._cancelled is True and "matrix" not in ConsentSession._active
    assert tools["clear_matrix_auth"]() == "Matrix authentication cleared."


# ------------------------------------------------------------------- Signal

# A Python program (not a shell script) so the same stand-in runs on
# Windows, where ``install_cli_script`` adds the ``.cmd`` shim.
_FAKE_SIGNAL_CLI = """#!/usr/bin/env python3
# Scripted stand-in for signal-cli: ``link`` prints the provisioning URI,
# waits for the phone (a marker file), then announces the linked account.
import os
import sys
import time

state_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
args = sys.argv[1:]


def marker(name):
    return os.path.exists(os.path.join(state_dir, name))


if args[:1] == ["link"]:
    if args[1:2] != ["-n"]:
        print("expected -n", file=sys.stderr)
        sys.exit(2)
    with open(os.path.join(state_dir, "device-name"), "w") as fh:
        fh.write(args[2] + "\\n")
    if marker("no-uri"):
        print("Failed to link", file=sys.stderr)
        sys.exit(3)
    print("sgnl://linkdevice?uuid=abc-123&pub_key=BQ%2Fkey", flush=True)
    if marker("noisy"):
        sys.stderr.write("x" * 2097152)
        sys.stderr.flush()
    while not marker("scanned"):
        if marker("fail"):
            print("Link request timed out", file=sys.stderr)
            sys.exit(1)
        time.sleep(0.1)
    if not marker("silent"):
        print("Associated with: +15550001111")
    sys.exit(0)
if args[:1] == ["listAccounts"]:
    if marker("silent"):
        print("Number: +15550009999")
    sys.exit(0)
print("unknown command " + " ".join(args[:1]), file=sys.stderr)
sys.exit(4)
"""


@pytest.fixture()
def fake_signal_cli(tmp_path: Path) -> Path:
    """Install the scripted signal-cli stand-in and return its path."""
    binary = tmp_path / "signal-cli"
    install_cli_script(binary, _FAKE_SIGNAL_CLI)
    return binary


def test_signal_link_flow_renders_qr_and_records_linked_account(
    isolated_kiss_home: Path, fake_signal_cli: Path
) -> None:
    """Signal: ``signal-cli link`` → QR → phone scans → account recorded."""
    agent = SignalAgent()
    tools = auth_tools(agent)
    unauth = tools["check_signal_auth"]()
    assert "finish_signal_auth" in unauth and "Linked devices" in unauth
    assert "verify CODE" not in unauth and "register &&" not in unauth
    assert tools["authenticate_signal"]("  ") == "phone_number cannot be empty."

    started = json.loads(tools["authenticate_signal"](signal_cli_path=str(fake_signal_cli)))
    assert started["status"] == "consent_required"
    uri = "sgnl://linkdevice?uuid=abc-123&pub_key=BQ%2Fkey"
    assert started["verification_uri"] == uri
    assert (fake_signal_cli.parent / "device-name").read_text().strip() == "KISS Sorcar"
    # The QR is rendered as half-block text and as an SVG page (0600).
    qr_text = started["qr_text"]
    assert len(qr_text.splitlines()) >= 10 and set(qr_text) <= set(" \u2580\u2584\u2588\n")
    assert qr_text in started["instructions"]
    page = Path(started["qr_page"])
    assert page.name == "link-qr.html"
    # NTFS has no POSIX mode bits: chmod(0o600) is a no-op there.
    assert IS_WINDOWS or (page.stat().st_mode & 0o777) == 0o600
    html = page.read_text()
    assert "<svg" in html and "uuid=abc-123&amp;pub_key=BQ%2Fkey" in html
    assert "Linked devices" in started["instructions"]
    assert "PIN" in started["instructions"] and "verification code" in started["instructions"]
    # The QR decodes back to the URI (module matrix round trip).
    from kiss.agents.third_party_agents.signal_agent import _qr_rows, _qr_text

    assert _qr_text(_qr_rows(uri)) == qr_text
    assert not sg_config.path.exists()

    pending = json.loads(tools["finish_signal_auth"]())
    assert pending["status"] == "pending" and "scan" in pending["error"]
    (fake_signal_cli.parent / "scanned").touch()
    done = _finish(tools["finish_signal_auth"], attempts=60)
    assert done == {"ok": True, "message": "Signal linked.", "phone_number": "+15550001111"}
    assert json.loads(sg_config.path.read_text()) == {
        "phone_number": "+15550001111",
        "signal_cli_path": str(fake_signal_cli),
    }
    assert agent._backend._phone_number == "+15550001111"
    assert agent._is_authenticated() is True
    assert tools["clear_signal_auth"]() == "Signal configuration cleared."
    assert not sg_config.path.exists()


def test_signal_link_failure_paths(isolated_kiss_home: Path, fake_signal_cli: Path) -> None:
    """Missing binary, no URI, link failure, silent success, cancel, and superseding."""
    agent = SignalAgent()
    tools = auth_tools(agent)
    state = fake_signal_cli.parent
    missing = json.loads(tools["authenticate_signal"](signal_cli_path=str(state / "nope")))
    assert missing["ok"] is False and "could not start" in missing["error"]
    (state / "no-uri").touch()
    no_uri = json.loads(tools["authenticate_signal"](signal_cli_path=str(fake_signal_cli)))
    assert no_uri == {"ok": False, "error": "signal-cli link failed: Failed to link"}
    (state / "no-uri").unlink()
    assert "no sign-in in progress" in json.loads(tools["finish_signal_auth"]())["error"]

    # The phone never scans and signal-cli gives up.
    assert (
        json.loads(tools["authenticate_signal"](signal_cli_path=str(fake_signal_cli)))["status"]
        == "consent_required"
    )
    (state / "fail").touch()
    failed = _finish(tools["finish_signal_auth"], attempts=60)
    assert failed == {
        "ok": False,
        "error": "Signal linking failed: linking failed: Link request timed out",
    }
    (state / "fail").unlink()

    # Older signal-cli prints no "Associated with" line → listAccounts.
    (state / "silent").touch()
    assert (
        json.loads(tools["authenticate_signal"](signal_cli_path=str(fake_signal_cli)))["status"]
        == "consent_required"
    )
    (state / "scanned").touch()
    done = _finish(tools["finish_signal_auth"], attempts=60)
    assert done["ok"] is True and done["phone_number"] == "+15550009999"
    (state / "scanned").unlink()
    (state / "silent").unlink()

    # Cancelling ends the signal-cli process; a recorded number supersedes
    # a pending link.
    assert (
        json.loads(tools["authenticate_signal"](signal_cli_path=str(fake_signal_cli)))["status"]
        == "consent_required"
    )
    session = ConsentSession._active["signal"]
    assert isinstance(session, SignalLinkSession)
    assert session._process.poll() is None
    direct = json.loads(tools["authenticate_signal"]("+15550002222", str(fake_signal_cli)))
    assert direct["ok"] is True and "signal" not in ConsentSession._active
    session._thread.join(timeout=10)
    assert not session._thread.is_alive()
    # Reaped (no zombie: returncode collected) and the QR page removed.
    assert session._process.returncode is not None
    assert not Path(session.page).exists()
    assert json.loads(sg_config.path.read_text())["phone_number"] == "+15550002222"

    # A chatty signal-cli (2 MiB on stderr) cannot stall the linking.
    (state / "noisy").touch()
    started = json.loads(tools["authenticate_signal"](signal_cli_path=str(fake_signal_cli)))
    assert started["status"] == "consent_required"
    (state / "scanned").touch()
    done = _finish(tools["finish_signal_auth"], attempts=100)
    assert done["ok"] is True and done["phone_number"] == "+15550001111"
    (state / "scanned").unlink()
    (state / "noisy").unlink()

    # The provisioning URI expires: the session reports it and reaps signal-cli.
    expiring = json.loads(tools["authenticate_signal"](signal_cli_path=str(fake_signal_cli)))
    assert expiring["status"] == "consent_required"
    session = ConsentSession._active["signal"]
    assert isinstance(session, SignalLinkSession)
    session._deadline = time.monotonic()
    session._thread.join(timeout=15)
    assert not session._thread.is_alive()
    assert session._process.returncode is not None and not Path(session.page).exists()
    assert _finish(tools["finish_signal_auth"]) == {
        "ok": False,
        "error": "Signal linking failed: the sign-in request expired before it was approved",
    }
