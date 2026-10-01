# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.agents.third_party_agents._composio_google``.

A local HTTP server speaks the subset of the Composio v3.1 REST API the
installed ``composio`` SDK calls (paths and JSON shapes taken from
``composio_client/resources``); ``COMPOSIO_BASE_URL`` points the real SDK
at it, so every request goes through the real client code.

Unreachable branch: the ``except Exception`` around ``_client()`` in
``clear_connection`` only fires when the ``Composio`` constructor itself
fails after an API key was found, which the real SDK does not do for a
well-formed key and base URL.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import httplib2  # type: ignore[import-untyped]
import pytest
import requests

from kiss.agents.third_party_agents import _composio_google as cg
from kiss.tests.conftest import IS_WINDOWS

_API = "/api/v3.1"


class _Request:
    """One request the fake Composio API received."""

    def __init__(self, method: str, path: str, query: dict[str, list[str]], body: Any) -> None:
        self.method = method
        self.path = path
        self.query = query
        self.body = body


class _ComposioHandler(BaseHTTPRequestHandler):
    def _handle(self) -> None:
        parts = urlsplit(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        body = json.loads(raw) if raw else None
        server = cast(_FakeComposio, self.server)
        server.requests.append(_Request(self.command, parts.path, parse_qs(parts.query), body))
        status, reply = server.route(self.command, parts.path)
        if isinstance(reply, bytes):
            payload, ctype = reply, "application/octet-stream"
        else:
            payload, ctype = json.dumps(reply).encode(), "application/json"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_DELETE = _handle  # noqa: N815 - http.server naming

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence per-request logging."""


class _FakeComposio(ThreadingHTTPServer):
    """In-memory Composio project: auth configs, connected accounts, proxy."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _ComposioHandler)
        self.requests: list[_Request] = []
        self.auth_configs: list[dict[str, Any]] = []
        # account id -> statuses returned by successive GETs (last one sticks)
        self.accounts: dict[str, list[str]] = {}
        self.undeletable: set[str] = set()
        self.next_account = "ca_new"
        self.proxy_reply: dict[str, Any] = {"status": 200, "data": {"ok": True}, "headers": {}}
        self.download: tuple[int, bytes] = (200, b"%PDF-1.7 bytes")

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def route(self, method: str, path: str) -> tuple[int, Any]:
        if path == f"{_API}/auth_configs" and method == "GET":
            return 200, {"items": self.auth_configs, "total_pages": 1, "current_page": 1}
        if path == f"{_API}/auth_configs" and method == "POST":
            created = {"id": "ac_created", "auth_scheme": "OAUTH2", "is_composio_managed": True}
            self.auth_configs.append(created)
            return 201, {"auth_config": created, "toolkit": {"slug": "gmail"}}
        if path == f"{_API}/connected_accounts" and method == "GET":
            return 200, {"items": [], "total_pages": 1, "current_page": 1}
        if path == f"{_API}/connected_accounts/link":
            self.accounts.setdefault(self.next_account, ["INITIATED"])
            return 201, {
                "connected_account_id": self.next_account,
                "redirect_url": "https://connect.composio.test/link/abc",
                "link_token": "lt_1",
                "expires_at": "2030-01-01T00:00:00Z",
            }
        if path.startswith(f"{_API}/connected_accounts/"):
            account = path.rsplit("/", 1)[1]
            if account not in self.accounts or (method == "DELETE" and account in self.undeletable):
                return 404, {"error": {"message": "not found", "status": 404}}
            if method == "DELETE":
                del self.accounts[account]
                return 200, {"success": True}
            statuses = self.accounts[account]
            status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
            return 200, {"id": account, "status": status}
        if path == f"{_API}/tools/execute/proxy":
            return 200, self.proxy_reply
        if path == "/download":
            return self.download
        return 404, {"error": {"message": f"no route {method} {path}"}}

    def paths(self, method: str) -> list[str]:
        return [r.path for r in self.requests if r.method == method]


@pytest.fixture
def composio(isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeComposio]:
    server = _FakeComposio()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COMPOSIO_BASE_URL", server.base)
    monkeypatch.setenv("COMPOSIO_API_KEY", "test-project-key")
    # A developer's shell may pin auth configs for any service
    # (KISS_COMPOSIO_AUTH_CONFIG_GOOGLE_CALENDAR=ac_...); every override
    # must go so the tests exercise the list/create path.
    monkeypatch.delenv("KISS_COMPOSIO_USER_ID", raising=False)
    for name in [n for n in os.environ if n.startswith("KISS_COMPOSIO_AUTH_CONFIG_")]:
        monkeypatch.delenv(name)
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _state(service: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((cg.service_dir(service) / "composio.json").read_text())
    return data


def _connect(service: str, account: str) -> None:
    """Persist *account* as *service*'s active connection."""
    cg.service_dir(service).mkdir(parents=True, exist_ok=True)
    (cg.service_dir(service) / "composio.json").write_text(
        json.dumps({"connected_account_id": account})
    )


# ---------------------------------------------------------------------------
# API key, user id, state files
# ---------------------------------------------------------------------------


def test_service_dir_honors_kiss_home(isolated_kiss_home: Path) -> None:
    assert cg.service_dir("gmail") == isolated_kiss_home / "third_party_agents" / "gmail"


def test_api_key_env_then_saved_file(
    isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    assert cg.composio_api_key() == ""
    cg.save_api_key("  saved-key \n")
    assert cg.composio_api_key() == "saved-key"
    key_file = isolated_kiss_home / "third_party_agents" / "google" / "composio_api_key.json"
    assert json.loads(key_file.read_text()) == {"api_key": "saved-key"}
    assert IS_WINDOWS or key_file.stat().st_mode & 0o777 == 0o600  # NTFS has no mode bits
    monkeypatch.setenv("COMPOSIO_API_KEY", " env-key ")
    assert cg.composio_api_key() == "env-key"


def test_user_id_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KISS_COMPOSIO_USER_ID", raising=False)
    assert cg.composio_user_id() == "kiss-default"
    monkeypatch.setenv("KISS_COMPOSIO_USER_ID", " alice ")
    assert cg.composio_user_id() == "alice"


def test_invalid_state_files_read_as_empty(isolated_kiss_home: Path) -> None:
    path = cg.service_dir("gmail") / "composio.json"
    path.parent.mkdir(parents=True)
    path.write_text("not json")
    assert cg.connected_account_id("gmail") == ""
    path.write_text("[1, 2]")
    assert cg.connected_account_id("gmail") == ""


# ---------------------------------------------------------------------------
# start_connect
# ---------------------------------------------------------------------------


def test_start_connect_reuses_newest_auth_config(
    composio: _FakeComposio, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KISS_COMPOSIO_USER_ID", "bob")
    composio.auth_configs = [
        {"id": "ac_old", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "ac_newest", "created_at": "2025-06-01T00:00:00Z"},
        {"id": "ac_undated"},
    ]
    answer = cg.start_connect("gmail", "Gmail")
    assert answer["ok"] is True
    assert answer["status"] == "consent_required"
    assert answer["verification_uri"] == "https://connect.composio.test/link/abc"
    assert answer["browser_opened"] is False
    assert "finish_gmail_auth()" in answer["instructions"]
    assert "https://connect.composio.test/link/abc" in answer["instructions"]

    listing = composio.requests[0]
    assert (listing.method, listing.path) == ("GET", f"{_API}/auth_configs")
    assert listing.query["toolkit_slug"] == ["gmail"]
    assert f"{_API}/auth_configs" not in composio.paths("POST")
    link = next(r for r in composio.requests if r.path.endswith("/link"))
    assert link.body == {"auth_config_id": "ac_newest", "user_id": "bob"}
    assert _state("gmail") == {"pending_id": "ca_new"}


def test_start_connect_creates_auth_config_when_none(composio: _FakeComposio) -> None:
    answer = cg.start_connect("google_calendar", "Google Calendar")
    assert answer["ok"] is True
    create = next(
        r for r in composio.requests if r.method == "POST" and r.path.endswith("/auth_configs")
    )
    assert create.body["toolkit"] == {"slug": "googlecalendar"}
    assert create.body["auth_config"]["type"] == "use_composio_managed_auth"
    link = next(r for r in composio.requests if r.path.endswith("/link"))
    assert link.body["auth_config_id"] == "ac_created"


def test_start_connect_auth_config_override(
    composio: _FakeComposio, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KISS_COMPOSIO_AUTH_CONFIG_GMAIL", " ac_pinned ")
    _connect("gmail", "ca_existing")
    answer = cg.start_connect("gmail", "Gmail")
    assert answer["ok"] is True
    assert not [r for r in composio.requests if r.path.endswith("/auth_configs")]
    link = next(r for r in composio.requests if r.path.endswith("/link"))
    assert link.body["auth_config_id"] == "ac_pinned"
    # The existing connection is kept until the new one becomes active.
    assert _state("gmail") == {"connected_account_id": "ca_existing", "pending_id": "ca_new"}


def test_start_connect_uses_saved_api_key(
    composio: _FakeComposio, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COMPOSIO_API_KEY")
    cg.save_api_key("saved-key")
    assert cg.start_connect("gmail", "Gmail")["ok"] is True


def test_start_connect_missing_api_key(
    composio: _FakeComposio, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COMPOSIO_API_KEY")
    answer = cg.start_connect("gmail", "Gmail")
    assert answer["ok"] is False
    assert answer["error"].startswith("Could not start the Gmail connection: No Composio API key")
    assert composio.requests == []
    assert not (cg.service_dir("gmail") / "composio.json").exists()


# ---------------------------------------------------------------------------
# finish_connect
# ---------------------------------------------------------------------------


def test_finish_connect_without_pending(composio: _FakeComposio) -> None:
    answer = cg.finish_connect("gmail", "Gmail")
    assert answer == {
        "ok": False,
        "error": "no sign-in in progress; call authenticate_gmail() first",
    }


def test_finish_connect_pending_timeout(composio: _FakeComposio) -> None:
    assert cg.start_connect("gmail", "Gmail")["ok"]
    answer = cg.finish_connect("gmail", "Gmail")
    assert answer["ok"] is False
    assert answer["status"] == "pending"
    assert _state("gmail") == {"pending_id": "ca_new"}


def test_finish_connect_active_replaces_and_deletes_previous(composio: _FakeComposio) -> None:
    composio.accounts["ca_prev"] = ["ACTIVE"]
    _connect("gmail", "ca_prev")
    assert cg.start_connect("gmail", "Gmail")["ok"]
    composio.accounts["ca_new"] = ["INITIATED", "INITIATED", "ACTIVE"]
    answer = cg.finish_connect("gmail", "Gmail")
    assert answer == {"ok": True, "message": "Gmail connected through Composio."}
    assert _state("gmail") == {"connected_account_id": "ca_new"}
    assert composio.paths("DELETE") == [f"{_API}/connected_accounts/ca_prev"]
    assert "ca_prev" not in composio.accounts


def test_finish_connect_active_same_account_not_deleted(composio: _FakeComposio) -> None:
    cg.service_dir("gmail").mkdir(parents=True)
    (cg.service_dir("gmail") / "composio.json").write_text(
        json.dumps({"connected_account_id": "ca_same", "pending_id": "ca_same"})
    )
    composio.accounts["ca_same"] = ["ACTIVE"]
    assert cg.finish_connect("gmail", "Gmail")["ok"] is True
    assert composio.paths("DELETE") == []


def test_finish_connect_previous_delete_failure_is_ignored(composio: _FakeComposio) -> None:
    composio.accounts["ca_prev"] = ["ACTIVE"]
    composio.undeletable.add("ca_prev")
    _connect("gmail", "ca_prev")
    cg.start_connect("gmail", "Gmail")
    composio.accounts["ca_new"] = ["ACTIVE"]
    assert cg.finish_connect("gmail", "Gmail")["ok"] is True
    assert _state("gmail") == {"connected_account_id": "ca_new"}


def test_finish_connect_failed_state(composio: _FakeComposio) -> None:
    _connect("gmail", "ca_prev")
    cg.start_connect("gmail", "Gmail")
    composio.accounts["ca_new"] = ["FAILED"]
    answer = cg.finish_connect("gmail", "Gmail")
    assert answer == {"ok": False, "error": "Gmail connection ended in state FAILED"}
    assert _state("gmail") == {"connected_account_id": "ca_prev"}


def test_finish_connect_fails_while_waiting(composio: _FakeComposio) -> None:
    cg.start_connect("gmail", "Gmail")
    composio.accounts["ca_new"] = ["INITIATED", "FAILED"]
    answer = cg.finish_connect("gmail", "Gmail")
    assert answer["ok"] is False
    assert answer["error"].startswith("Gmail connection failed:")
    assert "FAILED" in answer["error"]


def test_finish_connect_unknown_account_error(composio: _FakeComposio) -> None:
    cg.service_dir("gmail").mkdir(parents=True)
    (cg.service_dir("gmail") / "composio.json").write_text(json.dumps({"pending_id": "ca_gone"}))
    answer = cg.finish_connect("gmail", "Gmail")
    assert answer["ok"] is False
    assert answer["error"].startswith("Gmail connection failed:")


def test_finish_connect_keeps_waiting_through_inactive(composio: _FakeComposio) -> None:
    """INACTIVE is not terminal: the wait continues until the account activates."""
    assert cg.start_connect("gmail", "Gmail")["ok"]
    # The fake replies with one status per poll: INACTIVE first, then ACTIVE.
    composio.accounts["ca_new"] = ["INACTIVE", "ACTIVE"]
    answer = cg.finish_connect("gmail", "Gmail")
    assert answer == {"ok": True, "message": "Gmail connected through Composio."}
    assert _state("gmail") == {"connected_account_id": "ca_new"}
    # The status was polled more than once before it turned ACTIVE.
    assert composio.paths("GET").count(f"{_API}/connected_accounts/ca_new") >= 2


def test_start_connect_twice_deletes_the_superseded_pending_account(
    composio: _FakeComposio,
) -> None:
    """A second Connect Link deletes the first, unfinished account at Composio."""
    assert cg.start_connect("gmail", "Gmail")["ok"]
    assert _state("gmail") == {"pending_id": "ca_new"}
    composio.next_account = "ca_second"
    assert cg.start_connect("gmail", "Gmail")["ok"]
    assert composio.paths("DELETE") == [f"{_API}/connected_accounts/ca_new"]
    assert "ca_new" not in composio.accounts and "ca_second" in composio.accounts
    assert _state("gmail") == {"pending_id": "ca_second"}

    # Re-issuing the same pending account deletes nothing.
    assert cg.start_connect("gmail", "Gmail")["ok"]
    assert len(composio.paths("DELETE")) == 1


def test_finish_connect_superseded_while_waiting_does_not_record_the_old_account(
    composio: _FakeComposio,
) -> None:
    """A sign-in started during the wait owns the state; the stale one is deleted.

    ``finish_connect`` reads ``pending_id`` (A), waits for A, then
    re-reads the state.  Here a real ``start_connect`` for B runs while A
    is still INACTIVE.  Its own clean-up of A is refused by Composio
    (``undeletable``), the way a lost delete request would be, so A is
    still there to turn ACTIVE; the re-read then finds B pending, A is
    deleted again and nothing is recorded for it.
    """
    assert cg.start_connect("gmail", "Gmail")["ok"]  # A = ca_new
    composio.accounts["ca_new"] = ["INACTIVE"]
    composio.undeletable.add("ca_new")

    def start_b_then_activate_a() -> None:
        composio.next_account = "ca_b"
        assert cg.start_connect("gmail", "Gmail")["ok"]
        composio.accounts["ca_new"] = ["ACTIVE"]

    timer = threading.Timer(1.5, start_b_then_activate_a)
    timer.start()
    try:
        answer = cg.finish_connect("gmail", "Gmail")
    finally:
        timer.cancel()
        timer.join()
    assert answer == {
        "ok": False,
        "error": "a newer Gmail sign-in was started; finish that one instead",
    }
    # start_connect(B) tried to delete A, then finish_connect(A) did too.
    assert composio.paths("DELETE") == [f"{_API}/connected_accounts/ca_new"] * 2
    assert _state("gmail") == {"pending_id": "ca_b"}
    assert cg.connected_account_id("gmail") == ""

    # Only B, once ACTIVE, gets recorded.
    composio.accounts["ca_b"] = ["ACTIVE"]
    assert cg.finish_connect("gmail", "Gmail")["ok"] is True
    assert _state("gmail") == {"connected_account_id": "ca_b"}


def test_overlapping_finish_calls_keep_the_recorded_account(composio: _FakeComposio) -> None:
    """A finish that loses the race to another finish must not delete the account.

    Both calls wait on the same pending account A.  The second one sees
    A ACTIVE first and records it; the first one, re-reading the state
    afterwards, finds A already connected (not superseded) and answers
    ok without deleting anything.
    """
    assert cg.start_connect("gmail", "Gmail")["ok"]  # A = ca_new
    composio.accounts["ca_new"] = ["INACTIVE"]
    second: dict[str, Any] = {}

    def activate_and_finish() -> None:
        composio.accounts["ca_new"] = ["ACTIVE"]
        second.update(cg.finish_connect("gmail", "Gmail"))

    timer = threading.Timer(1.5, activate_and_finish)
    timer.start()
    try:
        first = cg.finish_connect("gmail", "Gmail")
    finally:
        timer.cancel()
        timer.join()
    assert second == {"ok": True, "message": "Gmail connected through Composio."}
    assert first == {"ok": True, "message": "Gmail connected through Composio."}
    assert composio.paths("DELETE") == []
    assert _state("gmail") == {"connected_account_id": "ca_new"}


# ---------------------------------------------------------------------------
# clear_connection
# ---------------------------------------------------------------------------


def test_clear_connection_deletes_remote_account(composio: _FakeComposio) -> None:
    composio.accounts["ca_live"] = ["ACTIVE"]
    _connect("gmail", "ca_live")
    cg.clear_connection("gmail")
    assert composio.paths("DELETE") == [f"{_API}/connected_accounts/ca_live"]
    assert not (cg.service_dir("gmail") / "composio.json").exists()
    cg.clear_connection("gmail")  # nothing connected: no request, no error
    assert len(composio.requests) == 1


def test_clear_connection_without_api_key_only_forgets(
    composio: _FakeComposio, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COMPOSIO_API_KEY")
    _connect("gmail", "ca_live")
    cg.clear_connection("gmail")
    assert composio.requests == []
    assert cg.connected_account_id("gmail") == ""


def test_clear_connection_deletes_a_pending_only_account(composio: _FakeComposio) -> None:
    """An unfinished sign-in is deleted at Composio too, not just forgotten."""
    assert cg.start_connect("gmail", "Gmail")["ok"]
    assert _state("gmail") == {"pending_id": "ca_new"}
    cg.clear_connection("gmail")
    assert composio.paths("DELETE") == [f"{_API}/connected_accounts/ca_new"]
    assert "ca_new" not in composio.accounts
    assert not (cg.service_dir("gmail") / "composio.json").exists()
    assert cg.finish_connect("gmail", "Gmail")["ok"] is False


# ---------------------------------------------------------------------------
# Proxy: ComposioSession, ComposioHttp, ComposioResponse
# ---------------------------------------------------------------------------


def _proxy_bodies(server: _FakeComposio) -> list[Any]:
    return [r.body for r in server.requests if r.path == f"{_API}/tools/execute/proxy"]


def test_session_get_params_and_headers(composio: _FakeComposio) -> None:
    _connect("gmail", "ca_live")
    composio.proxy_reply = {"status": 200, "data": {"messages": [1]}, "headers": {"X-Up": "1"}}
    response = cg.ComposioSession("gmail").get(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages?q=is%3Aunread&empty=",
        params={
            "includeSpamTrash": True,
            "labelIds": ["INBOX", "UNREAD"],
            "maxResults": 5,
            "x": False,
        },
        headers={"Authorization": "Bearer leaked", "User-Agent": "ua", "X-Goog-Trace": "t"},
    )
    body = _proxy_bodies(composio)[0]
    assert body["connected_account_id"] == "ca_live"
    assert body["endpoint"] == "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    assert body["method"] == "GET"
    assert "body" not in body or body["body"] is None
    assert body["parameters"] == [
        {"name": "q", "type": "query", "value": "is:unread"},
        {"name": "empty", "type": "query", "value": ""},
        {"name": "includeSpamTrash", "type": "query", "value": "true"},
        {"name": "labelIds", "type": "query", "value": "INBOX"},
        {"name": "labelIds", "type": "query", "value": "UNREAD"},
        {"name": "maxResults", "type": "query", "value": "5"},
        {"name": "x", "type": "query", "value": "false"},
        {"name": "X-Goog-Trace", "type": "header", "value": "t"},
    ]
    assert response.status_code == 200
    assert response.ok
    assert response.json() == {"messages": [1]}
    assert response.headers["content-type"] == "application/json"
    assert response.headers["x-up"] == "1"
    assert response.url.startswith("https://gmail.googleapis.com/gmail/v1/users/me/messages?q=")
    response.raise_for_status()


def test_session_bodies_and_methods(composio: _FakeComposio) -> None:
    _connect("gmail", "ca_live")
    session = cg.ComposioSession("gmail")
    url = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
    session.post(url, json={"summary": "x"})
    session.put(url, data='{"a": 1}')
    session.patch(url, data=b"[1, 2]")
    session.delete(url, data="")
    session.request("POST", url, json={"j": 1}, data="ignored", timeout=30)
    bodies = _proxy_bodies(composio)
    assert [b["method"] for b in bodies] == ["POST", "PUT", "PATCH", "DELETE", "POST"]
    assert [b.get("body") for b in bodies] == [{"summary": "x"}, {"a": 1}, [1, 2], None, {"j": 1}]
    assert all(b.get("parameters") is None for b in bodies)


def test_session_sends_non_json_bodies_as_binary(composio: _FakeComposio) -> None:
    """Non-JSON bodies travel base64-encoded in ``binary_body`` with their content type."""
    _connect("gmail", "ca_live")
    session = cg.ComposioSession("gmail")
    session.post(
        "https://www.googleapis.com/upload",
        data="--boundary\r\nmultipart",
        headers={"Content-Type": "multipart/related; boundary=boundary"},
    )
    session.post("https://www.googleapis.com/upload", data=b"\xff\xfe\x00binary")
    # JSON text under a non-JSON content type is still binary; under a JSON
    # content type that is not valid JSON, it is binary too.
    session.post(
        "https://www.googleapis.com/upload", data='{"a": 1}', headers={"Content-Type": "text/plain"}
    )
    session.post(
        "https://www.googleapis.com/upload",
        data="not json",
        headers={"Content-Type": "application/json"},
    )
    bodies = _proxy_bodies(composio)
    assert [b.get("body") for b in bodies] == [None, None, None, None]
    assert [base64.b64decode(b["binary_body"]["base64"]) for b in bodies] == [
        b"--boundary\r\nmultipart", b"\xff\xfe\x00binary", b'{"a": 1}', b"not json"
    ]
    assert [b["binary_body"]["content_type"] for b in bodies] == [
        "multipart/related; boundary=boundary", "application/octet-stream", "text/plain",
        "application/json",
    ]


def test_http_method_override_becomes_a_get(composio: _FakeComposio) -> None:
    """googleapiclient's over-long-GET-as-POST is sent to the proxy as the real GET."""
    _connect("gmail", "ca_live")
    cg.ComposioHttp("gmail").request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages",
        method="POST",
        body="q=subject%3Ax&maxResults=5",
        headers={
            "x-http-method-override": "GET",
            "content-type": "application/x-www-form-urlencoded",
            "content-length": "26",
        },
    )
    body = _proxy_bodies(composio)[0]
    assert body["method"] == "GET" and body.get("body") is None
    assert body.get("binary_body") is None
    assert [(p["name"], p["value"]) for p in body["parameters"]] == [
        ("q", "subject:x"), ("maxResults", "5")
    ]


def test_session_not_connected(composio: _FakeComposio) -> None:
    with pytest.raises(RuntimeError, match="gmail is not connected; call authenticate_gmail"):
        cg.ComposioSession("gmail").get("https://gmail.googleapis.com/x")


def test_proxy_text_and_empty_data(composio: _FakeComposio) -> None:
    _connect("gmail", "ca_live")
    composio.proxy_reply = {
        "status": 201,
        "data": "plain text",
        "headers": {"Content-Type": "text/plain"},
    }
    response = cg.ComposioSession("gmail").get("https://gmail.googleapis.com/x")
    assert (response.status_code, response.text, response.headers["Content-Type"]) == (
        201,
        "plain text",
        "text/plain",
    )
    composio.proxy_reply = {"status": 204}
    response = cg.ComposioSession("gmail").delete("https://gmail.googleapis.com/x")
    assert response.status_code == 204
    assert response.content == b""
    assert response.json() is None
    assert dict(response.headers) == {}


def test_proxy_binary_data_download(composio: _FakeComposio) -> None:
    _connect("google_drive", "ca_drive")
    composio.proxy_reply = {
        "status": 200,
        "headers": {"X-Drive": "1"},
        "binary_data": {
            "url": f"{composio.base}/download",
            "content_type": "application/pdf",
            "size": 14,
        },
    }
    response = cg.ComposioSession("google_drive").get(
        "https://www.googleapis.com/drive/v3/files/f1", params={"alt": "media"}
    )
    assert response.content == b"%PDF-1.7 bytes"
    assert response.headers["Content-Type"] == "application/pdf"
    assert response.headers["X-Drive"] == "1"

    composio.download = (404, b"gone")
    with pytest.raises(requests.HTTPError):
        cg.ComposioSession("google_drive").get("https://www.googleapis.com/drive/v3/files/f1")


def test_composio_http_returns_httplib2_response(composio: _FakeComposio) -> None:
    _connect("google_docs", "ca_docs")
    composio.proxy_reply = {
        "status": 404,
        "data": {"error": {"code": 404}},
        "headers": {"X-Goog-Thing": "v"},
    }
    response, content = cg.ComposioHttp("google_docs").request(
        "https://docs.googleapis.com/v1/documents/d1?fields=title",
        method="POST",
        body='{"requests": []}',
        headers={"authorization": "Bearer x", "content-type": "application/json"},
        redirections=5,
    )
    assert isinstance(response, httplib2.Response)
    assert response.status == 404
    assert response["x-goog-thing"] == "v"
    assert response["content-type"] == "application/json"
    assert json.loads(content) == {"error": {"code": 404}}
    body = _proxy_bodies(composio)[0]
    assert body["body"] == {"requests": []}
    assert body["parameters"] == [
        {"name": "fields", "type": "query", "value": "title"},
        {"name": "content-type", "type": "header", "value": "application/json"},
    ]


def test_composio_response_raise_for_status() -> None:
    ok = cg.ComposioResponse(200, {"A": "b"}, b'{"x": 1}', "https://g.test/ok")
    ok.raise_for_status()
    assert (ok.ok, ok.reason, ok.json(), ok.headers["a"]) == (True, "OK", {"x": 1}, "b")

    bad = cg.ComposioResponse(404, {}, b"missing \xff", "https://g.test/missing")
    assert bad.ok is False
    assert bad.reason == "NOT_FOUND"
    with pytest.raises(requests.HTTPError) as info:
        bad.raise_for_status()
    assert info.value.response is bad
    assert "404 Error for url: https://g.test/missing: missing" in str(info.value)

    assert cg.ComposioResponse(799, {}, b"", "u").reason == ""
