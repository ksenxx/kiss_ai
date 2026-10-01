# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing verified bugs in gmail_sea and whatsapp_sea.

No mocks, patches, or fakes of kiss classes: WhatsApp tests run the real
``WhatsAppChannelBackend`` against a real local HTTP server speaking the
whatsapp-mcp bridge REST protocol and a real bridge-schema SQLite database;
Gmail tests use a real googleapiclient service built from the bundled
static discovery document.

Bugs covered (the WhatsApp ones re-targeted at the QR-paired bridge backend):
  (C) gmail: ``send_message`` addressed mail to a label ID (e.g. "INBOX").
  (E) whatsapp: ``send_message`` must surface bridge send failures.
  (G) whatsapp: ``poll_messages`` must honour ``channel_id``/limit/cursor.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import httplib2  # type: ignore[import-untyped]
import pytest
from googleapiclient.discovery import build

from kiss.agents.third_party_agents.gmail.gmail_sea import GmailChannelBackend
from kiss.agents.third_party_agents.whatsapp.whatsapp_sea import WhatsAppChannelBackend
from kiss.tests.agents.third_party_agents.whatsapp_bridge import bridge_server


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
            (
                "m1",
                "111@s.whatsapp.net",
                "111@s.whatsapp.net",
                "from-111-a",
                "2026-01-01 00:00:01+00:00",
                0,
                "",
            ),
            (
                "m2",
                "222@s.whatsapp.net",
                "222@s.whatsapp.net",
                "from-222",
                "2026-01-01 00:00:02+00:00",
                0,
                "",
            ),
            (
                "m3",
                "111@s.whatsapp.net",
                "111@s.whatsapp.net",
                "from-111-b",
                "2026-01-01 00:00:03+00:00",
                0,
                "",
            ),
        ],
    )
    conn.commit()
    conn.close()


class TestWhatsAppSendMessage:
    """Bug (E): send_message must raise when the bridge reports failure."""

    def test_send_message_raises_on_bridge_error(self, tmp_path: Path) -> None:
        with bridge_server({"success": False, "message": "bad recipient"}) as server:
            port = server.server_address[1]
            backend = WhatsAppChannelBackend(repo_dir=str(tmp_path), bridge_port=port)
            with pytest.raises(RuntimeError, match="bad recipient"):
                backend.send_message("+14155238886", "hello")

    def test_send_message_succeeds_without_error(self, tmp_path: Path) -> None:
        with bridge_server({"success": True, "message": "sent"}) as server:
            port = server.server_address[1]
            backend = WhatsAppChannelBackend(repo_dir=str(tmp_path), bridge_port=port)
            backend.send_message("+14155238886", "hello")
            assert server.requests == [
                {"path": "/api/send", "json": {"recipient": "14155238886", "message": "hello"}}
            ]


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

    def test_poll_messages_empty_channel_id_returns_all_senders(self, tmp_path: Path) -> None:
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


class TestGmailSendMessage:
    """Bug (C): send_message must not address mail to a non-email channel_id."""

    @staticmethod
    def _backend(refusing_port: int) -> GmailChannelBackend:
        """Real Gmail service whose API endpoint refuses every connection.

        ``send_message`` resolves ``thread_ts`` with a live
        ``threads().get`` call before validating the recipient; pointing
        the service at a refusing loopback port (with a short socket
        timeout) keeps that call off the network and instant.
        """
        backend = GmailChannelBackend()
        backend._service = build(
            "gmail",
            "v1",
            developerKey="test",
            static_discovery=True,
            client_options={"api_endpoint": f"http://127.0.0.1:{refusing_port}/"},
            http=httplib2.Http(timeout=2),
        )
        return backend

    def test_send_message_rejects_label_id_recipient(self, refusing_port: int) -> None:
        with pytest.raises(ValueError, match="email address"):
            self._backend(refusing_port).send_message("INBOX", "hello")

    def test_send_message_rejects_label_id_when_thread_unresolvable(
        self, refusing_port: int
    ) -> None:
        with pytest.raises(ValueError, match="email address"):
            self._backend(refusing_port).send_message(
                "INBOX", "hello", thread_ts="nonexistent-thread"
            )
