# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing verified bugs in gmail_agent and whatsapp_agent.

No mocks, patches, or fakes of kiss classes: WhatsApp tests run the real
``WhatsAppChannelBackend`` against a real local HTTP server and the backend's
real webhook queue; Gmail tests use the real OAuth flow (headless, real dummy
credentials file) and a real googleapiclient service built from the bundled
static discovery document.

Bugs covered:
  (A) gmail: ``flow.run_console()`` removed in google-auth-oauthlib >= 1.0.
  (C) gmail: ``send_message`` addressed mail to a label ID (e.g. "INBOX").
  (E) whatsapp: ``send_message`` silently swallowed Graph API errors.
  (F) whatsapp: hand-rolled queue draining raced (``queue.Empty`` escapes).
  (G) whatsapp: ``poll_messages`` ignored ``channel_id`` (no sender filter).
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler
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


class _GraphAPIHandler(BaseHTTPRequestHandler):
    """Records POST requests and replies with the server's canned JSON body."""

    def do_POST(self) -> None:
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


def _start_graph_server(response_body: dict[str, Any]) -> tuple[ThreadedHTTPServer, str]:
    """Start a local HTTP server standing in for the Meta Graph API."""
    server = ThreadedHTTPServer(("127.0.0.1", 0), _GraphAPIHandler)
    server.response_body = response_body  # type: ignore[attr-defined]
    server.recorded_requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _make_backend(base_url: str) -> WhatsAppChannelBackend:
    """Build a WhatsApp backend pointed at a local Graph API server."""
    backend = WhatsAppChannelBackend(graph_api_base=base_url)
    backend._access_token = "test-token"
    backend._phone_number_id = "1234567890"
    return backend


class TestWhatsAppSendMessage:
    """Bug (E): send_message must raise when the Graph API returns an error."""

    def test_send_message_raises_on_api_error(self) -> None:
        server, base = _start_graph_server({"error": {"message": "bad token", "code": 190}})
        try:
            backend = _make_backend(base)
            with pytest.raises(RuntimeError, match="bad token"):
                backend.send_message("+14155238886", "hello")
        finally:
            stop_http_server(server, None)

    def test_send_message_succeeds_without_error(self) -> None:
        server, base = _start_graph_server({"messages": [{"id": "wamid.OK"}]})
        try:
            backend = _make_backend(base)
            backend.send_message("+14155238886", "hello")
            path, body = server.recorded_requests[0]  # type: ignore[attr-defined]
            assert path == "/1234567890/messages"
            assert body["to"] == "+14155238886"
            assert body["text"]["body"] == "hello"
        finally:
            stop_http_server(server, None)


class TestWhatsAppPollMessages:
    """Bugs (F)+(G): safe queue draining and channel_id sender filtering."""

    @staticmethod
    def _raw(sender: str, text: str, msg_id: str) -> dict[str, Any]:
        return {
            "from": sender,
            "id": msg_id,
            "timestamp": "1700000000",
            "type": "text",
            "text": {"body": text},
        }

    def test_poll_messages_filters_to_channel_id(self) -> None:
        backend = WhatsAppChannelBackend()
        backend._message_queue.put(self._raw("111", "from-111-a", "m1"))
        backend._message_queue.put(self._raw("222", "from-222", "m2"))
        backend._message_queue.put(self._raw("111", "from-111-b", "m3"))
        messages, cursor = backend.poll_messages("111", "0", limit=10)
        assert cursor == "0"
        assert [m["user"] for m in messages] == ["111", "111"]
        assert [m["text"] for m in messages] == ["from-111-a", "from-111-b"]
        assert messages[0]["ts"] == "1700000000"
        assert messages[0]["id"] == "m1"

    def test_poll_messages_empty_channel_id_returns_all_senders(self) -> None:
        backend = WhatsAppChannelBackend()
        backend._message_queue.put(self._raw("111", "a", "m1"))
        backend._message_queue.put(self._raw("222", "b", "m2"))
        messages, _ = backend.poll_messages("", "0", limit=10)
        assert [m["user"] for m in messages] == ["111", "222"]

    def test_poll_messages_respects_limit(self) -> None:
        backend = WhatsAppChannelBackend()
        for i in range(5):
            backend._message_queue.put(self._raw("111", f"t{i}", f"m{i}"))
        messages, _ = backend.poll_messages("111", "0", limit=2)
        assert len(messages) == 2

    def test_poll_messages_non_text_message_gets_placeholder(self) -> None:
        backend = WhatsAppChannelBackend()
        backend._message_queue.put(
            {"from": "111", "id": "m1", "timestamp": "1", "type": "image", "image": {}}
        )
        messages, _ = backend.poll_messages("111", "0", limit=10)
        assert messages[0]["text"] == "[image message]"

    def test_poll_messages_survives_concurrent_consumer(self) -> None:
        """Bug (F): a competing consumer must not make poll_messages raise."""
        backend = WhatsAppChannelBackend()
        stop = threading.Event()

        def compete() -> None:
            import queue as _q

            while not stop.is_set():
                try:
                    backend._message_queue.get_nowait()
                except _q.Empty:
                    pass

        thief = threading.Thread(target=compete, daemon=True)
        thief.start()
        try:
            for i in range(300):
                backend._message_queue.put(self._raw("111", "x", f"m{i}"))
                messages, _ = backend.poll_messages("111", "0", limit=10)
                assert isinstance(messages, list)
        finally:
            stop.set()
            thief.join(timeout=5.0)


class TestWhatsAppWaitForReply:
    """wait_for_reply must use safe draining and match the expected sender."""

    def test_wait_for_reply_returns_matching_text(self) -> None:
        backend = WhatsAppChannelBackend()
        backend._message_queue.put(
            {"from": "999", "id": "m1", "timestamp": "1", "type": "text", "text": {"body": "yes"}}
        )
        assert backend.wait_for_reply("", "", "999", timeout_seconds=3.0) == "yes"

    def test_wait_for_reply_times_out_without_match(self) -> None:
        backend = WhatsAppChannelBackend()
        backend._message_queue.put(
            {"from": "888", "id": "m1", "timestamp": "1", "type": "text", "text": {"body": "no"}}
        )
        assert backend.wait_for_reply("", "", "999", timeout_seconds=0.2) is None


class TestGmailOAuthFlow:
    """Bug (A): headless OAuth flow must not call the removed run_console()."""

    def test_run_console_removed_from_installed_dependency(self) -> None:
        assert not hasattr(InstalledAppFlow, "run_console")

    def test_headless_oauth_flow_does_not_raise_attribute_error(self) -> None:
        creds_path = gmail_agent._credentials_path()
        backup = creds_path.read_text() if creds_path.exists() else None
        old_headless = os.environ.get("KISS_HEADLESS")
        creds_path.parent.mkdir(parents=True, exist_ok=True)
        creds_path.write_text(json.dumps(_DUMMY_CLIENT_SECRETS))
        os.environ["KISS_HEADLESS"] = "1"
        result: dict[str, BaseException] = {}

        def run_flow() -> None:
            try:
                gmail_agent._run_oauth_flow()
            except BaseException as exc:
                result["exc"] = exc

        thread = threading.Thread(target=run_flow, daemon=True)
        try:
            thread.start()
            thread.join(timeout=3.0)
            exc = result.get("exc")
            assert not isinstance(exc, AttributeError), f"run_console still used: {exc}"
        finally:
            if old_headless is None:
                os.environ.pop("KISS_HEADLESS", None)
            else:
                os.environ["KISS_HEADLESS"] = old_headless
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
