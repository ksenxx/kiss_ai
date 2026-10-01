# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Gmail message headers and in-thread drafts.

Both bugs these tests pin down were observed on a live account:

* ``list_messages`` returned empty subject/from/to because Composio's
  proxy keeps only the last value of a repeated query parameter, so of
  ``metadataHeaders=Subject&metadataHeaders=From&...`` only ``Date``
  reached Gmail.  The query now travels in the endpoint URL.
* ``create_draft`` could not attach a draft to an existing thread.  It
  now takes ``thread_id`` / ``in_reply_to`` / ``references`` and
  resolves the reply headers from the thread.

The Gmail API calls go through the real googleapiclient and Composio
SDK to the local Composio emulator, whose proxy forwards them to the
local Gmail endpoint below (which, like Gmail, returns only the
requested ``metadataHeaders``).
"""

from __future__ import annotations

import base64
import json
from email import message_from_bytes
from email.message import Message
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import pytest

from kiss.agents.third_party_agents.gmail.gmail_sea import _SERVICE, GmailChannelBackend
from kiss.tests.agents.third_party_agents.composio_test_utils import (
    connect,
    reset_state,
    start_fake_composio,
)
from kiss.tests.agents.third_party_agents.recording_http import RecordingServer, recording_server

_MESSAGES: dict[str, dict[str, Any]] = {
    "m1": {
        "id": "m1",
        "threadId": "t1",
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": "first",
        "headers": {
            "Subject": "Quals schedule",
            "From": "Alice <alice@example.com>",
            "To": "me@example.com",
            "Date": "Thu, 1 Oct 2026 10:00:00 +0000",
            "Message-ID": "<alice-1@example.com>",
        },
    },
    "m2": {
        "id": "m2",
        "threadId": "t2",
        "labelIds": ["INBOX"],
        "snippet": "second",
        "headers": {
            "Subject": "Re: Budget",
            "From": "Bob <bob@example.com>",
            "To": "me@example.com, carol@example.com",
            "Date": "Thu, 1 Oct 2026 11:00:00 +0000",
            "Message-Id": "<bob-2@example.com>",
            "References": "<me-1@example.com>",
        },
    },
}
_THREADS = {"t1": ["m1"], "t2": ["m2"]}


def _api_message(msg_id: str, wanted: list[str]) -> dict[str, Any]:
    """Shape a stored message like ``messages.get(format=metadata)`` does.

    Like Gmail, ``metadataHeaders`` matches case-insensitively and the
    header names come back as the sender wrote them.
    """
    msg = _MESSAGES[msg_id]
    wanted_lower = {w.lower() for w in wanted}
    headers = [
        {"name": name, "value": value}
        for name, value in msg["headers"].items()
        if not wanted_lower or name.lower() in wanted_lower
    ]
    return {
        "id": msg["id"],
        "threadId": msg["threadId"],
        "labelIds": msg["labelIds"],
        "snippet": msg["snippet"],
        "payload": {"headers": headers},
    }


class _GmailHandler(BaseHTTPRequestHandler):
    """Local Gmail endpoint: profile, messages.list/get, threads.get, drafts.create."""

    def _send(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=UTF-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        """Serve profile, message list/get and thread get."""
        parts = urlsplit(self.path)
        query = parse_qs(parts.query, keep_blank_values=True)
        wanted = query.get("metadataHeaders", [])
        cast(RecordingServer, self.server).requests.append(
            {"method": "GET", "path": parts.path, "query": query}
        )
        tail = parts.path.split("/users/me/", 1)[1]
        if tail == "profile":
            self._send(200, {"emailAddress": "me@example.com"})
        elif tail == "messages":
            self._send(
                200,
                {
                    "messages": [{"id": "m1", "threadId": "t1"}, {"id": "m2", "threadId": "t2"}],
                    "resultSizeEstimate": 2,
                },
            )
        elif tail.startswith("messages/"):
            self._send(200, _api_message(tail.split("/", 1)[1], wanted))
        elif tail.startswith("threads/"):
            thread_id = tail.split("/", 1)[1]
            if thread_id not in _THREADS:
                self._send(404, {"error": {"code": 404, "message": "Requested entity not found."}})
                return
            self._send(
                200,
                {
                    "id": thread_id,
                    "messages": [_api_message(m, wanted) for m in _THREADS[thread_id]],
                },
            )
        else:
            self._send(404, {"error": {"code": 404, "message": f"no route {tail}"}})

    def do_POST(self) -> None:  # noqa: N802
        """Record ``drafts.create`` and answer with a draft in the requested thread."""
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else {}
        parts = urlsplit(self.path)
        cast(RecordingServer, self.server).requests.append(
            {"method": "POST", "path": parts.path, "body": body}
        )
        message = body.get("message", {})
        thread_id = message.get("threadId", "dm-1")
        self._send(200, {"id": "d-1", "message": {"id": "dm-1", "threadId": thread_id}})

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


@pytest.fixture()
def backend(monkeypatch):
    """A connected GmailChannelBackend talking to the local Gmail endpoint.

    Yields:
        ``(backend, server)``.
    """
    reset_state(_SERVICE)
    with recording_server(_GmailHandler) as server:
        for composio in start_fake_composio(monkeypatch):
            composio.upstream_overrides["https://gmail.googleapis.com"] = server.base_url
            connect(composio, _SERVICE)
            gmail = GmailChannelBackend()
            assert gmail.connect() is True
            yield gmail, server
    reset_state(_SERVICE)


def _draft_mime(server: RecordingServer) -> tuple[str | None, Message, Message]:
    """Return ``(threadId, MIME message, its text part)`` of the recorded drafts.create call."""
    posts = [r for r in server.requests if r["method"] == "POST"]
    assert len(posts) == 1 and posts[0]["path"].endswith("/users/me/drafts")
    message = posts[0]["body"]["message"]
    mime = message_from_bytes(base64.urlsafe_b64decode(message["raw"]))
    text_part = cast(list[Message], mime.get_payload())[0]
    return message.get("threadId"), mime, text_part


def test_list_messages_returns_all_requested_headers(backend) -> None:
    """Every metadataHeaders value reaches Gmail, so subject/from/to/date are filled."""
    gmail, server = backend
    result = json.loads(gmail.list_messages(query="is:unread", label_ids="INBOX,UNREAD"))
    assert result["ok"] is True
    assert [m["subject"] for m in result["messages"]] == ["Quals schedule", "Re: Budget"]
    assert result["messages"][0]["from"] == "Alice <alice@example.com>"
    assert result["messages"][1]["to"] == "me@example.com, carol@example.com"
    assert result["messages"][0]["date"] == "Thu, 1 Oct 2026 10:00:00 +0000"
    assert result["messages"][0]["thread_id"] == "t1"
    assert result["result_size_estimate"] == 2
    listing = next(r for r in server.requests if r["path"].endswith("/users/me/messages"))
    assert listing["query"]["labelIds"] == ["INBOX", "UNREAD"]
    assert listing["query"]["q"] == ["is:unread"]
    fetch = next(r for r in server.requests if r["path"].endswith("/messages/m1"))
    assert sorted(fetch["query"]["metadataHeaders"]) == ["Date", "From", "Subject", "To"]


def test_create_draft_in_thread_resolves_reply_headers(backend) -> None:
    """thread_id alone: draft addressed to the last sender with Re: subject and reply headers."""
    gmail, server = backend
    result = json.loads(gmail.create_draft("", "", "Thanks, works for me.", thread_id="t1"))
    assert result == {"ok": True, "draft_id": "d-1", "message_id": "dm-1", "thread_id": "t1"}
    thread_id, mime, text_part = _draft_mime(server)
    assert thread_id == "t1"
    assert mime["to"] == "Alice <alice@example.com>"
    assert mime["subject"] == "Re: Quals schedule"
    assert mime["In-Reply-To"] == "<alice-1@example.com>"
    assert mime["References"] == "<alice-1@example.com>"
    assert "cc" not in mime and "bcc" not in mime
    assert text_part.get_content_type() == "text/plain"
    assert "Thanks, works for me." in str(text_part.get_payload())


def test_create_draft_in_thread_extends_references_and_keeps_re_subject(backend) -> None:
    """References chain extended, Re: subject kept, ``Message-Id`` spelling accepted."""
    gmail, server = backend
    result = json.loads(
        gmail.create_draft(
            "",
            "",
            "<p>ok</p>",
            cc="dan@example.com",
            bcc="eve@example.com",
            html=True,
            thread_id="t2",
        )
    )
    assert result["ok"] is True and result["thread_id"] == "t2"
    thread_id, mime, text_part = _draft_mime(server)
    assert thread_id == "t2"
    assert mime["to"] == "Bob <bob@example.com>"
    assert mime["subject"] == "Re: Budget"
    assert mime["cc"] == "dan@example.com" and mime["bcc"] == "eve@example.com"
    assert mime["In-Reply-To"] == "<bob-2@example.com>"
    assert mime["References"] == "<me-1@example.com> <bob-2@example.com>"
    assert text_part.get_content_type() == "text/html"


def test_create_draft_explicit_values_override_thread_lookup(backend) -> None:
    """Caller-supplied to/subject/in_reply_to/references are sent unchanged."""
    gmail, server = backend
    result = json.loads(
        gmail.create_draft(
            "carol@example.com",
            "Re: Budget (revised)",
            "see attached",
            thread_id="t2",
            in_reply_to="<custom@example.com>",
            references="<a@example.com> <custom@example.com>",
        )
    )
    assert result["ok"] is True
    thread_id, mime, _ = _draft_mime(server)
    assert thread_id == "t2"
    assert mime["to"] == "carol@example.com"
    assert mime["subject"] == "Re: Budget (revised)"
    assert mime["In-Reply-To"] == "<custom@example.com>"
    assert mime["References"] == "<a@example.com> <custom@example.com>"


def test_create_draft_without_thread_is_a_standalone_message(backend) -> None:
    """No thread_id: no threadId, no reply headers, and the draft starts its own thread."""
    gmail, server = backend
    result = json.loads(gmail.create_draft("frank@example.com", "Hello", "plain body"))
    assert result == {"ok": True, "draft_id": "d-1", "message_id": "dm-1", "thread_id": "dm-1"}
    thread_id, mime, _ = _draft_mime(server)
    assert thread_id is None
    assert mime["to"] == "frank@example.com" and mime["subject"] == "Hello"
    assert "In-Reply-To" not in mime and "References" not in mime
    assert not any(r["path"].endswith("/threads/t1") for r in server.requests)


def test_create_draft_without_recipient_fails(backend) -> None:
    """Neither a recipient nor a resolvable thread: a clear error, and nothing is created."""
    gmail, server = backend
    assert json.loads(gmail.create_draft("", "Subject", "body")) == {
        "ok": False,
        "error": "create_draft needs a recipient: pass `to` or a `thread_id`",
    }
    missing = json.loads(gmail.create_draft("", "", "body", thread_id="no-such-thread"))
    assert missing["ok"] is False and "recipient" in missing["error"]
    assert not any(r["method"] == "POST" for r in server.requests)
