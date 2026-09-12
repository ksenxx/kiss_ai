# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing verified bugs in gmail_agent and whatsapp_agent.

No mocks, patches, or fakes of kiss classes: WhatsApp tests run the real
``WhatsAppChannelBackend`` against a real local HTTP server speaking the
whatsapp-mcp bridge REST protocol and a real bridge-schema SQLite database;
Gmail tests use the real OAuth flow (headless, real dummy credentials file)
and a real googleapiclient service built from the bundled static discovery
document.

Bugs covered (the WhatsApp ones re-targeted at the QR-paired bridge backend):
  (A) gmail: ``flow.run_console()`` removed in google-auth-oauthlib >= 1.0.
  (C) gmail: ``send_message`` addressed mail to a label ID (e.g. "INBOX").
  (E) whatsapp: ``send_message`` must surface bridge send failures.
  (G) whatsapp: ``poll_messages`` must honour ``channel_id``/limit/cursor.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from kiss.agents.third_party_agents import gmail_agent
from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents.gmail_agent import GmailChannelBackend
from kiss.agents.third_party_agents.whatsapp_agent import WhatsAppChannelBackend

_DUMMY_CLIENT_SECRETS = {
    "installed": {
        "client_id": "test-client-id.apps.googleusercontent.com",
        "client_secret": "test-secret",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"],
    }
}


class _BridgeHandler(BaseHTTPRequestHandler):
    """Records POST requests and replies with the server's canned JSON body.

    Speaks the whatsapp-mcp bridge REST protocol (POST-only /api/send).
    """

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server.recorded_requests.append(  # type: ignore[attr-defined]
            (self.path, json.loads(body or b"{}"))
        )
        payload = json.dumps(self.server.response_body).encode()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


def _start_bridge_server(response_body: dict[str, Any]) -> tuple[ThreadedHTTPServer, int]:
    """Start a local HTTP server standing in for the whatsapp-mcp bridge."""
    server = ThreadedHTTPServer(("127.0.0.1", 0), _BridgeHandler)
    server.response_body = response_body  # type: ignore[attr-defined]
    server.recorded_requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def _make_db(repo_dir: Path) -> None:
    """Create a bridge-schema messages.db with two senders' messages."""
    store = repo_dir / "whatsapp-bridge" / "store"
    store.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(store / "messages.db")
    conn.executescript(
        "CREATE TABLE chats (jid TEXT PRIMARY KEY, name TEXT,"
        " last_message_time TIMESTAMP);"
        "CREATE TABLE messages (id TEXT, chat_jid TEXT, sender TEXT, content TEXT,"
        " timestamp TIMESTAMP, is_from_me BOOLEAN, media_type TEXT, filename TEXT,"
        " url TEXT, media_key BLOB, file_sha256 BLOB, file_enc_sha256 BLOB,"
        " file_length INTEGER, PRIMARY KEY (id, chat_jid))"
    )
    conn.executemany(
        "INSERT INTO chats VALUES (?,?,?)",
        [
            ("111@s.whatsapp.net", "One", "2026-01-01 00:00:02+00:00"),
            ("222@s.whatsapp.net", "Two", "2026-01-01 00:00:03+00:00"),
        ],
    )
    conn.executemany(
        "INSERT INTO messages (id, chat_jid, sender, content, timestamp, is_from_me,"
        " media_type) VALUES (?,?,?,?,?,?,?)",
        [
            ("m1", "111@s.whatsapp.net", "111@s.whatsapp.net", "from-111-a",
             "2026-01-01 00:00:01+00:00", 0, ""),
            ("m2", "222@s.whatsapp.net", "222@s.whatsapp.net", "from-222",
             "2026-01-01 00:00:02+00:00", 0, ""),
            ("m3", "111@s.whatsapp.net", "111@s.whatsapp.net", "from-111-b",
             "2026-01-01 00:00:03+00:00", 0, ""),
        ],
    )
    conn.commit()
    conn.close()


class TestWhatsAppSendMessage:
    """Bug (E): send_message must raise when the bridge reports failure."""

    def test_send_message_raises_on_bridge_error(self, tmp_path: Path) -> None:
        server, port = _start_bridge_server({"success": False, "message": "bad recipient"})
        try:
            backend = WhatsAppChannelBackend(repo_dir=str(tmp_path), bridge_port=port)
            with pytest.raises(RuntimeError, match="bad recipient"):
                backend.send_message("+14155238886", "hello")
        finally:
            stop_http_server(server, None)

    def test_send_message_succeeds_without_error(self, tmp_path: Path) -> None:
        server, port = _start_bridge_server({"success": True, "message": "sent"})
        try:
            backend = WhatsAppChannelBackend(repo_dir=str(tmp_path), bridge_port=port)
            backend.send_message("+14155238886", "hello")
            path, body = server.recorded_requests[0]  # type: ignore[attr-defined]
            assert path == "/api/send"
            assert body == {"recipient": "14155238886", "message": "hello"}
        finally:
            stop_http_server(server, None)


class TestWhatsAppPollMessages:
    """Bug (G): poll_messages must honour channel_id, limit, and the cursor.

    (Bug (F) — racy hand-rolled webhook-queue draining — no longer has an
    equivalent: the QR-paired backend reads the bridge's SQLite database,
    which has no shared in-process queue to race on.)
    """

    def test_poll_messages_filters_to_channel_id(self, tmp_path: Path) -> None:
        _make_db(tmp_path)
        backend = WhatsAppChannelBackend(repo_dir=str(tmp_path))
        messages, cursor = backend.poll_messages("111", "0", limit=10)
        assert [m["user"] for m in messages] == ["111@s.whatsapp.net"] * 2
        assert [m["text"] for m in messages] == ["from-111-a", "from-111-b"]
        assert cursor == "2026-01-01 00:00:03+00:00"

    def test_poll_messages_empty_channel_id_returns_all_senders(
        self, tmp_path: Path
    ) -> None:
        _make_db(tmp_path)
        backend = WhatsAppChannelBackend(repo_dir=str(tmp_path))
        messages, _ = backend.poll_messages("", "0", limit=10)
        assert [m["text"] for m in messages] == ["from-111-a", "from-222", "from-111-b"]

    def test_poll_messages_respects_limit(self, tmp_path: Path) -> None:
        _make_db(tmp_path)
        backend = WhatsAppChannelBackend(repo_dir=str(tmp_path))
        messages, _ = backend.poll_messages("111", "0", limit=1)
        assert len(messages) == 1

    def test_poll_messages_cursor_excludes_seen(self, tmp_path: Path) -> None:
        _make_db(tmp_path)
        backend = WhatsAppChannelBackend(repo_dir=str(tmp_path))
        messages, _ = backend.poll_messages("111", "2026-01-01 00:00:01+00:00", limit=10)
        assert [m["text"] for m in messages] == ["from-111-b"]


def _wait_for_redirect_port(
    caplog: pytest.LogCaptureFixture, flow_thread: threading.Thread, timeout: float = 10.0
) -> int:
    """Return the local redirect port announced by ``run_local_server``.

    The flow logs its authorization URL (INFO on ``google_auth_oauthlib.flow``)
    once the redirect server is listening; the ``redirect_uri`` query
    parameter carries the ephemeral port.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for record in caplog.records:
            match = re.search(
                r"redirect_uri=http%3A%2F%2Flocalhost%3A(\d+)", record.getMessage()
            )
            if match:
                return int(match.group(1))
        if not flow_thread.is_alive():
            break
        time.sleep(0.02)
    raise AssertionError("OAuth flow never announced its local redirect server")


class TestGmailOAuthFlow:
    """Bug (A): headless OAuth flow must not call the removed run_console()."""

    def test_run_console_removed_from_installed_dependency(self) -> None:
        assert not hasattr(InstalledAppFlow, "run_console")

    def test_headless_oauth_flow_does_not_raise_attribute_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The headless flow starts a real local redirect server, not run_console().

        ``run_local_server`` blocks in ``handle_request()`` until the
        browser redirect arrives, so the test plays the browser: it reads
        the redirect port from the flow's INFO log line and sends a bogus
        ``?state=...&code=...`` redirect. ``fetch_token`` then fails on the
        CSRF state check before any network I/O, which lets the flow
        thread and its listening socket exit instead of leaking.
        """
        creds_path = gmail_agent._credentials_path()
        backup = creds_path.read_text() if creds_path.exists() else None
        creds_path.parent.mkdir(parents=True, exist_ok=True)
        creds_path.write_text(json.dumps(_DUMMY_CLIENT_SECRETS))
        monkeypatch.setenv("KISS_HEADLESS", "1")
        result: dict[str, BaseException] = {}

        def run_flow() -> None:
            try:
                gmail_agent._run_oauth_flow()
            except BaseException as exc:
                result["exc"] = exc

        thread = threading.Thread(target=run_flow, daemon=True)
        try:
            with caplog.at_level(logging.INFO, logger="google_auth_oauthlib.flow"):
                thread.start()
                port = _wait_for_redirect_port(caplog, thread)
            with urllib.request.urlopen(
                f"http://localhost:{port}/?state=bogus&code=bogus", timeout=10
            ) as resp:
                assert resp.status == 200
            thread.join(timeout=10.0)
            assert not thread.is_alive(), "OAuth flow thread did not exit"
            exc = result.get("exc")
            assert not isinstance(exc, AttributeError), f"run_console still used: {exc}"
            # oauthlib ships no type stubs, so match the CSRF-check error by name.
            assert type(exc).__name__ == "MismatchingStateError", f"unexpected outcome: {exc!r}"
        finally:
            if backup is not None:
                creds_path.write_text(backup)
            elif creds_path.exists():
                creds_path.unlink()


class TestGmailSendMessage:
    """Bug (C): send_message must not address mail to a non-email channel_id."""

    @staticmethod
    def _backend() -> GmailChannelBackend:
        backend = GmailChannelBackend()
        backend._service = build("gmail", "v1", developerKey="test", static_discovery=True)
        return backend

    def test_send_message_rejects_label_id_recipient(self) -> None:
        with pytest.raises(ValueError, match="email address"):
            self._backend().send_message("INBOX", "hello")

    def test_send_message_rejects_label_id_when_thread_unresolvable(self) -> None:
        with pytest.raises(ValueError, match="email address"):
            self._backend().send_message("INBOX", "hello", thread_ts="nonexistent-thread")
