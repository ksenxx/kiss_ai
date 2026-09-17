# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the shared Google Workspace OAuth helpers.

Exercises ``_google_workspace_utils`` against the real filesystem and
real ``google.oauth2.credentials.Credentials`` objects — no mocks,
patches, or fakes.  Each test isolates its state by pointing
``KISS_HOME`` at a fresh temporary directory (the helpers resolve
``KISS_HOME`` lazily on every call).  Token refreshes are exercised
against a REAL local OAuth token endpoint (stdlib HTTP server).

Deliberately untested branches (unreachable without test doubles):

* ``load_google_credentials``'s expired-token refresh: credentials
  loaded via ``Credentials.from_authorized_user_file`` carry the
  hard-coded ``https://oauth2.googleapis.com/token`` endpoint, so
  exercising that refresh would require real network access to Google.
* ``run_google_oauth_flow`` with a credentials.json present: it starts
  ``InstalledAppFlow.run_local_server``, which blocks on an interactive
  browser consent that no automated test can complete.  The
  authenticate-success branch of ``make_google_auth_tools`` depends on
  that same interactive flow returning credentials.
"""

from __future__ import annotations

import json
import stat
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, cast

from google.oauth2.credentials import Credentials

from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents._google_workspace_utils import (
    clear_google_credentials,
    credentials_path,
    fresh_access_token,
    google_service_dir,
    load_google_credentials,
    make_google_auth_tools,
    run_google_oauth_flow,
    save_google_credentials,
    token_path,
)
from kiss.agents.third_party_agents.google_calendar_agent import (
    _SCOPES,
    _SERVICE,
    GoogleCalendarAgent,
)

_SYNTHETIC_INFO = {
    "token": "synthetic-access-token",
    "refresh_token": "synthetic-refresh-token",
    "client_id": "synthetic-client-id",
    "client_secret": "synthetic-client-secret",
    "scopes": _SCOPES,
    "expiry": "2099-01-01T00:00:00Z",
}


def _synthetic_creds() -> Credentials:
    """Build valid (non-expired) synthetic Google OAuth2 credentials."""
    return cast(
        Credentials, Credentials.from_authorized_user_info(dict(_SYNTHETIC_INFO), _SCOPES)
    )


def _write_token(service: str, text: str) -> Path:
    """Write raw *text* as the service's token.json and return its path."""
    path = token_path(service)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_service_dir_and_token_path_follow_kiss_home(isolated_kiss_home) -> None:
    """google_service_dir and token_path live under $KISS_HOME."""
    assert google_service_dir("gsvc") == isolated_kiss_home / "third_party_agents" / "gsvc"
    assert token_path("gsvc") == isolated_kiss_home / "third_party_agents" / "gsvc" / "token.json"


def test_credentials_path_fallback_order(isolated_kiss_home) -> None:
    """credentials_path prefers service dir > google dir > gmail dir > own path."""
    own = google_service_dir(_SERVICE) / "credentials.json"
    shared = google_service_dir("google") / "credentials.json"
    gmail = google_service_dir("gmail") / "credentials.json"

    # No candidate exists: the service's own (not yet created) path is returned.
    assert credentials_path(_SERVICE) == own
    assert not own.exists()

    gmail.parent.mkdir(parents=True, exist_ok=True)
    gmail.write_text("{}", encoding="utf-8")
    assert credentials_path(_SERVICE) == gmail

    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_text("{}", encoding="utf-8")
    assert credentials_path(_SERVICE) == shared

    own.parent.mkdir(parents=True, exist_ok=True)
    own.write_text("{}", encoding="utf-8")
    assert credentials_path(_SERVICE) == own


def test_load_credentials_missing_file(isolated_kiss_home) -> None:
    """load_google_credentials returns None when no token.json exists."""
    assert load_google_credentials(_SERVICE, _SCOPES) is None


def test_load_credentials_corrupt_file(isolated_kiss_home) -> None:
    """load_google_credentials returns None for unparseable token files."""
    _write_token(_SERVICE, "this is not json {")
    assert load_google_credentials(_SERVICE, _SCOPES) is None
    _write_token(_SERVICE, json.dumps({"token": "x"}))  # missing required OAuth keys
    assert load_google_credentials(_SERVICE, _SCOPES) is None


def test_load_credentials_wrong_shape_json_returns_none(isolated_kiss_home) -> None:
    """Valid JSON of the wrong shape ([], null, bare string) yields None, not a crash.

    Credentials.from_authorized_user_file raises AttributeError on
    non-dict JSON; the loader must swallow that and return None.
    """
    for wrong_shape in ("[]", "null", '"just-a-string"'):
        _write_token(_SERVICE, wrong_shape)
        assert load_google_credentials(_SERVICE, _SCOPES) is None


def test_tools_survives_wrong_shape_token_files(isolated_kiss_home) -> None:
    """Each Google agent module's tools() works with a wrong-shape token.json."""
    import kiss.agents.third_party_agents.google_calendar_agent as gcal_mod
    import kiss.agents.third_party_agents.google_docs_agent as gdocs_mod
    import kiss.agents.third_party_agents.google_drive_agent as gdrive_mod

    modules = {
        "google_calendar": gcal_mod,
        "google_drive": gdrive_mod,
        "google_docs": gdocs_mod,
    }
    for wrong_shape in ("[]", "null", '"just-a-string"'):
        for service, module in modules.items():
            _write_token(service, wrong_shape)
            tools = module.tools()
            assert tools, f"{service} tools() returned no tools for {wrong_shape!r}"
            assert all(callable(t) for t in tools)


def test_authenticate_with_malformed_credentials_json_returns_ok_false(isolated_kiss_home) -> None:
    """A malformed credentials.json makes authenticate return ok:false — no raise."""
    agent = GoogleCalendarAgent()
    creds_file = google_service_dir(_SERVICE) / "credentials.json"
    creds_file.parent.mkdir(parents=True, exist_ok=True)
    for malformed in ("not json", "[]"):
        creds_file.write_text(malformed, encoding="utf-8")
        tools = {t.__name__: t for t in agent._get_auth_tools()}
        result = json.loads(tools["authenticate_google_calendar"]())
        assert result["ok"] is False
        assert result["error"]
        assert "OAuth flow failed" in result["error"]


def test_load_credentials_valid_synthetic_token(isolated_kiss_home) -> None:
    """A synthetic non-expired token.json loads as valid credentials."""
    _write_token(_SERVICE, json.dumps(_SYNTHETIC_INFO))
    creds = load_google_credentials(_SERVICE, _SCOPES)
    assert creds is not None
    assert creds.valid is True
    assert creds.token == "synthetic-access-token"


def test_save_credentials_writes_0600(isolated_kiss_home) -> None:
    """save_google_credentials persists the token with owner-only permissions."""
    save_google_credentials(_SERVICE, _synthetic_creds())
    path = token_path(_SERVICE)
    assert path.exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    reloaded = load_google_credentials(_SERVICE, _SCOPES)
    assert reloaded is not None
    assert reloaded.token == "synthetic-access-token"


def test_clear_credentials_removes_and_tolerates_absence(isolated_kiss_home) -> None:
    """clear_google_credentials deletes the token and is a no-op when absent."""
    _write_token(_SERVICE, json.dumps(_SYNTHETIC_INFO))
    clear_google_credentials(_SERVICE)
    assert not token_path(_SERVICE).exists()
    clear_google_credentials(_SERVICE)  # second call must not raise
    assert not token_path(_SERVICE).exists()


def test_fresh_access_token(isolated_kiss_home) -> None:
    """fresh_access_token returns '' for None and the token for valid creds."""
    assert fresh_access_token(None) == ""
    assert fresh_access_token(_synthetic_creds()) == "synthetic-access-token"


class _TokenEndpointHandler(BaseHTTPRequestHandler):
    """Emulates the OAuth2 token endpoint for refresh-token grants."""

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        payload = json.dumps(
            {"access_token": "refreshed-token", "expires_in": 3600, "token_type": "Bearer"}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


def _expired_creds(token_uri: str) -> Credentials:
    """Build real expired credentials whose refresh hits *token_uri*."""
    return Credentials(
        token="stale-token",
        refresh_token="synthetic-refresh-token",
        token_uri=token_uri,
        client_id="synthetic-client-id",
        client_secret="synthetic-client-secret",
        scopes=_SCOPES,
        expiry=datetime(2000, 1, 1),
    )


def test_fresh_access_token_refreshes_expired_creds(isolated_kiss_home) -> None:
    """Expired creds are refreshed against a real local token endpoint."""
    server = ThreadedHTTPServer(("127.0.0.1", 0), _TokenEndpointHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        creds = _expired_creds(f"http://127.0.0.1:{server.server_address[1]}/token")
        assert creds.valid is False
        assert fresh_access_token(creds) == "refreshed-token"
    finally:
        stop_http_server(server, thread)


def test_fresh_access_token_returns_empty_when_refresh_fails(isolated_kiss_home) -> None:
    """A refresh against an unreachable token endpoint yields '' — no exception."""
    creds = _expired_creds("http://127.0.0.1:9/token")  # discard port; nothing listens
    assert fresh_access_token(creds) == ""


def test_run_google_oauth_flow_without_credentials_json(isolated_kiss_home) -> None:
    """run_google_oauth_flow returns None when no credentials.json exists."""
    assert run_google_oauth_flow(_SERVICE, _SCOPES) is None


def test_make_google_auth_tools_names_and_docstrings(isolated_kiss_home) -> None:
    """make_google_auth_tools builds the 5 named tools with real docstrings."""
    agent = GoogleCalendarAgent()

    def on_credentials(creds) -> None:
        agent._backend._creds = creds

    tools = make_google_auth_tools(
        agent, _SERVICE, "Google Calendar", _SCOPES, on_credentials=on_credentials
    )
    names = [t.__name__ for t in tools]
    assert names == [
        "check_google_calendar_auth",
        "authenticate_google_calendar",
        "clear_google_calendar_auth",
        "start_google_calendar_browser_setup",
        "finish_google_calendar_auth",
    ]
    for tool in tools:
        assert tool.__doc__ is not None
        assert "Returns:" in tool.__doc__
        assert "Google Calendar" in tool.__doc__


def test_auth_tools_end_to_end_flow(isolated_kiss_home) -> None:
    """The generated auth tools work end-to-end on a real agent."""
    agent = GoogleCalendarAgent()
    tools = {t.__name__: t for t in agent._get_auth_tools()}

    # Unauthenticated, no credentials.json anywhere: check points at browser setup.
    msg = tools["check_google_calendar_auth"]()
    assert "start_google_calendar_browser_setup" in msg

    # authenticate without credentials.json explains where to put the file.
    result = tools["authenticate_google_calendar"]()
    assert "credentials.json not found" in result
    assert str(google_service_dir(_SERVICE) / "credentials.json") in result

    # With a credentials.json present, check names its path instead.
    creds_file = google_service_dir(_SERVICE) / "credentials.json"
    creds_file.parent.mkdir(parents=True, exist_ok=True)
    creds_file.write_text("{}", encoding="utf-8")
    msg = tools["check_google_calendar_auth"]()
    assert "credentials.json exists" in msg
    assert "authenticate_google_calendar" in msg

    # Authenticated agent: check reports ok.
    _write_token(_SERVICE, json.dumps(_SYNTHETIC_INFO))
    agent._backend._creds = load_google_credentials(_SERVICE, _SCOPES)
    assert json.loads(tools["check_google_calendar_auth"]())["ok"] is True

    # start_browser_setup returns Google Cloud Console instructions.
    setup = tools["start_google_calendar_browser_setup"]()
    assert "console.cloud.google.com" in setup
    assert "authenticate_google_calendar" in setup

    # clear removes the token and detaches the backend credentials.
    cleared = tools["clear_google_calendar_auth"]()
    assert "cleared" in cleared.lower()
    assert not token_path(_SERVICE).exists()
    assert agent._backend._creds is None
    assert agent._is_authenticated() is False
