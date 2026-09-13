# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for whatsapp_agent — no mocks or test doubles.

The agent wraps the lharries/whatsapp-mcp Go bridge (QR-paired personal
WhatsApp).  These tests exercise the real code paths end to end:

- SQLite-backed tools run against a real ``messages.db`` created with the
  bridge's exact schema.
- REST-backed tools (send, download) run against a real local HTTP server
  that speaks the bridge's ``/api/send`` / ``/api/download`` protocol
  (POST-only, 405 on GET — exactly like Go's http mux).
- QR pairing helpers run against real bridge-log text (the half-block
  qrterminal format captured from a live bridge run).

Building and launching the actual Go bridge requires a Go toolchain and
network access to WhatsApp servers, so those two subprocess branches are
covered up to their guard conditions (missing binary / missing go).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents.whatsapp_agent import (
    WhatsAppAgent,
    WhatsAppChannelBackend,
    _bridge_log_path,
    _bridge_pid_path,
    _config,
    _extract_last_qr,
    _is_qr_line,
    _qr_html_path,
    _rest_recipient,
    _to_jid,
    _write_qr_html,
    main,
    tools,
)

# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    jid TEXT PRIMARY KEY,
    name TEXT,
    last_message_time TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT,
    chat_jid TEXT,
    sender TEXT,
    content TEXT,
    timestamp TIMESTAMP,
    is_from_me BOOLEAN,
    media_type TEXT,
    filename TEXT,
    url TEXT,
    media_key BLOB,
    file_sha256 BLOB,
    file_enc_sha256 BLOB,
    file_length INTEGER,
    PRIMARY KEY (id, chat_jid),
    FOREIGN KEY (chat_jid) REFERENCES chats(jid)
);
"""

_ALICE = "14155550001@s.whatsapp.net"
_BOB = "14155550002@s.whatsapp.net"
_GROUP = "120363000000000001@g.us"


def _make_db(repo_dir: Path) -> Path:
    """Create a messages.db with the bridge's schema and sample data."""
    store = repo_dir / "whatsapp-bridge" / "store"
    store.mkdir(parents=True, exist_ok=True)
    db = store / "messages.db"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    chats = [
        (_ALICE, "Alice", "2026-09-10 10:00:03+00:00"),
        (_BOB, "Bob", "2026-09-10 09:00:00+00:00"),
        (_GROUP, "Family Group", "2026-09-10 11:00:00+00:00"),
    ]
    conn.executemany("INSERT INTO chats (jid, name, last_message_time) VALUES (?,?,?)", chats)
    messages = [
        ("m1", _ALICE, _ALICE, "hi there", "2026-09-10 10:00:01+00:00", 0, ""),
        ("m2", _ALICE, "me", "hello alice", "2026-09-10 10:00:02+00:00", 1, ""),
        ("m3", _ALICE, _ALICE, "photo for you", "2026-09-10 10:00:03+00:00", 0, "image"),
        ("m4", _BOB, _BOB, "lunch tomorrow?", "2026-09-10 09:00:00+00:00", 0, ""),
        ("m5", _GROUP, _ALICE, "group ping", "2026-09-10 11:00:00+00:00", 0, ""),
        ("m6", _BOB, _BOB, "", "2026-09-10 09:30:00+00:00", 0, "audio"),
    ]
    conn.executemany(
        "INSERT INTO messages (id, chat_jid, sender, content, timestamp, is_from_me,"
        " media_type) VALUES (?,?,?,?,?,?,?)",
        messages,
    )
    conn.commit()
    conn.close()
    return db


def _make_paired_session(repo_dir: Path, paired: bool = True) -> Path:
    """Create a whatsmeow session DB; *paired* adds a device row.

    The bridge creates ``store/whatsapp.db`` before the QR is ever
    scanned, so pairing is a row in ``whatsmeow_device``, not the file.
    """
    store = repo_dir / "whatsapp-bridge" / "store"
    store.mkdir(parents=True, exist_ok=True)
    db = store / "whatsapp.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE IF NOT EXISTS whatsmeow_device (jid TEXT PRIMARY KEY)")
    if paired:
        conn.execute(
            "INSERT OR REPLACE INTO whatsmeow_device (jid) "
            "VALUES ('14155550000.0:1@s.whatsapp.net')"
        )
    conn.commit()
    conn.close()
    return db


class _BridgeHandler(BaseHTTPRequestHandler):
    """Real HTTP handler speaking the whatsapp-mcp bridge REST protocol."""

    def do_GET(self) -> None:  # noqa: N802 – http.server API
        # Go's mux answers GET on the POST-only /api/send with 405.
        self.send_response(405)
        self.end_headers()
        self.wfile.write(b"Method not allowed\n")

    def do_POST(self) -> None:  # noqa: N802 – http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.recorded.append((self.path, body))  # type: ignore[attr-defined]
        payload = json.dumps(self.server.response_body).encode()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


def _start_bridge_server(
    response_body: dict[str, Any],
) -> tuple[ThreadingHTTPServer, int]:
    """Start a local HTTP server standing in for the bridge REST API."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BridgeHandler)
    server.response_body = response_body  # type: ignore[attr-defined]
    server.recorded = []  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


@contextmanager
def _bridge_server(response_body: dict[str, Any]) -> Iterator[tuple[ThreadingHTTPServer, int]]:
    """Run a bridge stand-in server for the block, shutting it down afterwards."""
    server, port = _start_bridge_server(response_body)
    try:
        yield server, port
    finally:
        server.shutdown()
        server.server_close()


# A genuine qrterminal.GenerateHalfBlock-shaped block (charset {█ ▀ ▄ space}).
_QR_BLOCK = "\n".join(
    ["█" * 69] * 2
    + ["████ ▄▄▄▄▄ ██▀ ▄▄█ ▀▄▄ ▄█▄▄▄▄ ▀█▄█▄▄▄▀ █▄ ▀ █▀▀█ █ ▄▄▄▄▄ ████".ljust(69, "█")] * 30
    + ["█" * 69]
    + ["▀" * 69]
)

_BRIDGE_LOG_WITH_QR = (
    "14:59:03.322\x1b[36m [Client INFO] Starting WhatsApp client...\x1b[0m\n"
    "\nScan this QR code with your WhatsApp app:\n" + _QR_BLOCK + "\n"
    "14:59:05.414\x1b[36m [Client INFO] waiting\x1b[0m\n"
)


def _backup_and_clear() -> str | None:
    """Back up existing config file and remove it."""
    path = _config.path
    backup = None
    if path.exists():
        backup = path.read_text()
        path.unlink()
    return backup


def _restore(backup: str | None) -> None:
    """Restore a previously backed-up config file."""
    path = _config.path
    if backup is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(backup)
    elif path.exists():
        path.unlink()


class _ChannelState:
    """Backs up and cleans the channel dir state files between tests."""

    def __init__(self) -> None:
        self._cfg = _backup_and_clear()
        for p in (_bridge_log_path(), _bridge_pid_path(), _qr_html_path()):
            p.unlink(missing_ok=True)

    def restore(self) -> None:
        _restore(self._cfg)
        for p in (_bridge_log_path(), _bridge_pid_path(), _qr_html_path()):
            p.unlink(missing_ok=True)


@pytest.fixture()
def channel_state() -> Any:
    """Isolate config/log/pid/QR files for a test."""
    state = _ChannelState()
    yield state
    state.restore()


@pytest.fixture()
def db_backend(tmp_path: Path) -> WhatsAppChannelBackend:
    """Backend with a real populated messages.db under tmp_path."""
    _make_db(tmp_path)
    return WhatsAppChannelBackend(repo_dir=str(tmp_path))


# ----------------------------------------------------------------------
# Recipient normalization
# ----------------------------------------------------------------------


class TestRecipientNormalization:
    def test_to_jid_passes_jids_through(self) -> None:
        assert _to_jid(_GROUP) == _GROUP
        assert _to_jid(" 123@s.whatsapp.net ") == "123@s.whatsapp.net"

    def test_to_jid_normalizes_phone_numbers(self) -> None:
        assert _to_jid("+1 (415) 555-0001") == "14155550001@s.whatsapp.net"

    def test_rest_recipient_strips_symbols(self) -> None:
        assert _rest_recipient("+1 415-555.0001") == "14155550001"
        assert _rest_recipient(_GROUP) == _GROUP


# ----------------------------------------------------------------------
# QR extraction and rendering
# ----------------------------------------------------------------------


class TestQRExtraction:
    def test_is_qr_line(self) -> None:
        assert _is_qr_line("█" * 30)
        assert _is_qr_line("████ ▄▄▄▄▄ █▀▄ ▀▄▄ ▄█▄▄▄▄ ████")
        assert not _is_qr_line("█" * 10)  # too short
        assert not _is_qr_line(" " * 30)  # no block glyph at all
        assert not _is_qr_line("14:59:03 [Client INFO] Starting...")

    def test_extract_last_qr_from_real_log(self) -> None:
        qr = _extract_last_qr(_BRIDGE_LOG_WITH_QR)
        assert qr == _QR_BLOCK

    def test_extract_last_qr_returns_latest_block(self) -> None:
        older = "\n".join(["▄" * 40] * 12)
        log = f"noise\n{older}\nrotating...\n{_QR_BLOCK}\ntail"
        assert _extract_last_qr(log) == _QR_BLOCK

    def test_extract_ignores_short_blocks(self) -> None:
        log = "line\n" + "\n".join(["█" * 40] * 3) + "\nline"
        assert _extract_last_qr(log) == ""

    def test_extract_from_empty_log(self) -> None:
        assert _extract_last_qr("") == ""

    def test_write_qr_html_renders_black_on_white_glyphs(self, channel_state: Any) -> None:
        page = _write_qr_html(_QR_BLOCK)
        html = page.read_text(encoding="utf-8")
        assert _QR_BLOCK in html
        assert "background:#000" in html and "color:#fff" in html
        assert "Linked devices" in html
        assert "http-equiv='refresh'" in html


# ----------------------------------------------------------------------
# SQLite-backed tools (real messages.db, bridge schema)
# ----------------------------------------------------------------------


class TestDatabaseTools:
    def test_search_contacts_excludes_groups(self, db_backend: WhatsAppChannelBackend) -> None:
        data = json.loads(db_backend.search_whatsapp_contacts("alice"))
        assert data["ok"] is True
        assert [c["name"] for c in data["contacts"]] == ["Alice"]
        assert data["contacts"][0]["phone_number"] == "14155550001"
        everyone = json.loads(db_backend.search_whatsapp_contacts(""))
        assert {c["name"] for c in everyone["contacts"]} == {"Alice", "Bob"}

    def test_list_chats_sorted_by_last_active(
        self, db_backend: WhatsAppChannelBackend
    ) -> None:
        data = json.loads(db_backend.list_whatsapp_chats())
        assert data["ok"] is True
        assert [c["name"] for c in data["chats"]] == ["Family Group", "Alice", "Bob"]
        assert data["chats"][1]["last_message"] == "photo for you"

    def test_list_chats_by_name_without_last_message(
        self, db_backend: WhatsAppChannelBackend
    ) -> None:
        data = json.loads(
            db_backend.list_whatsapp_chats(include_last_message=False, sort_by="name")
        )
        assert [c["name"] for c in data["chats"]] == ["Alice", "Bob", "Family Group"]
        assert "last_message" not in data["chats"][0]

    def test_list_chats_query_filter_and_pagination(
        self, db_backend: WhatsAppChannelBackend
    ) -> None:
        data = json.loads(db_backend.list_whatsapp_chats(query="bob"))
        assert [c["name"] for c in data["chats"]] == ["Bob"]
        page2 = json.loads(db_backend.list_whatsapp_chats(limit=1, page=1))
        assert [c["name"] for c in page2["chats"]] == ["Alice"]

    def test_get_chat(self, db_backend: WhatsAppChannelBackend) -> None:
        data = json.loads(db_backend.get_whatsapp_chat(_GROUP))
        assert data["ok"] is True and data["chat"]["name"] == "Family Group"
        missing = json.loads(db_backend.get_whatsapp_chat("nope@g.us"))
        assert missing["ok"] is False

    def test_get_direct_chat_by_contact(self, db_backend: WhatsAppChannelBackend) -> None:
        data = json.loads(db_backend.get_whatsapp_direct_chat_by_contact("+1-415-555-0002"))
        assert data["ok"] is True and data["chat"]["name"] == "Bob"
        missing = json.loads(db_backend.get_whatsapp_direct_chat_by_contact("99999"))
        assert missing["ok"] is False

    def test_get_contact_chats(self, db_backend: WhatsAppChannelBackend) -> None:
        data = json.loads(db_backend.get_whatsapp_contact_chats(_ALICE))
        names = [c["name"] for c in data["chats"]]
        assert names == ["Family Group", "Alice"]

    def test_get_last_interaction(self, db_backend: WhatsAppChannelBackend) -> None:
        data = json.loads(db_backend.get_whatsapp_last_interaction(_ALICE))
        assert data["ok"] is True
        assert data["message"]["content"] == "group ping"
        missing = json.loads(db_backend.get_whatsapp_last_interaction("0@s.whatsapp.net"))
        assert missing["ok"] is False

    def test_list_messages_filters(self, db_backend: WhatsAppChannelBackend) -> None:
        by_chat = json.loads(db_backend.list_whatsapp_messages(chat_jid=_ALICE))
        assert [m["id"] for m in by_chat["messages"]] == ["m3", "m2", "m1"]
        by_query = json.loads(db_backend.list_whatsapp_messages(query="LUNCH"))
        assert [m["id"] for m in by_query["messages"]] == ["m4"]
        by_sender = json.loads(
            db_backend.list_whatsapp_messages(sender_phone_number="14155550001")
        )
        assert {m["id"] for m in by_sender["messages"]} == {"m1", "m3", "m5"}
        windowed = json.loads(
            db_backend.list_whatsapp_messages(
                after="2026-09-10 10:00:01+00:00", before="2026-09-10 10:00:03+00:00"
            )
        )
        assert [m["id"] for m in windowed["messages"]] == ["m2"]
        paged = json.loads(db_backend.list_whatsapp_messages(chat_jid=_ALICE, limit=1, page=1))
        assert [m["id"] for m in paged["messages"]] == ["m2"]
        t_form = json.loads(
            db_backend.list_whatsapp_messages(after="2026-09-10T10:00:01+00:00")
        )
        assert {m["id"] for m in t_form["messages"]} == {"m2", "m3", "m5"}
        bad = json.loads(db_backend.list_whatsapp_messages(after="not-a-date"))
        assert bad["ok"] is False and "Invalid date" in bad["error"]

    def test_list_messages_sender_matches_both_stored_forms(
        self, db_backend: WhatsAppChannelBackend
    ) -> None:
        # Live events store bare digits; history sync stores full JIDs.
        conn = sqlite3.connect(db_backend.messages_db)
        conn.execute(
            "INSERT INTO messages (id, chat_jid, sender, content, timestamp,"
            " is_from_me, media_type) VALUES ('m7', ?, '14155550001', 'live form',"
            " '2026-09-10 12:00:00+00:00', 0, '')",
            (_GROUP,),
        )
        conn.commit()
        conn.close()
        data = json.loads(
            db_backend.list_whatsapp_messages(sender_phone_number="+1 (415) 555-0001")
        )
        assert {m["id"] for m in data["messages"]} == {"m1", "m3", "m5", "m7"}
        by_jid = json.loads(db_backend.list_whatsapp_messages(sender_phone_number=_ALICE))
        assert {m["id"] for m in by_jid["messages"]} == {"m1", "m3", "m5", "m7"}

    def test_long_results_stay_valid_json(self, db_backend: WhatsAppChannelBackend) -> None:
        conn = sqlite3.connect(db_backend.messages_db)
        for i in range(40):
            conn.execute(
                "INSERT INTO messages (id, chat_jid, sender, content, timestamp,"
                " is_from_me, media_type) VALUES (?, ?, ?, ?, ?, 0, '')",
                (f"big{i}", _ALICE, _ALICE, "x" * 3000, f"2026-09-11 00:00:{i:02d}+00:00"),
            )
        conn.commit()
        conn.close()
        text = db_backend.list_whatsapp_messages(chat_jid=_ALICE, limit=40)
        data = json.loads(text)  # must never be truncated mid-token
        assert data["ok"] is True and data.get("truncated") is True
        assert all(len(m["content"]) <= 2020 for m in data["messages"])

    def test_get_message_context(self, db_backend: WhatsAppChannelBackend) -> None:
        data = json.loads(db_backend.get_whatsapp_message_context("m2"))
        assert data["ok"] is True
        assert data["message"]["id"] == "m2"
        assert [m["id"] for m in data["before"]] == ["m1"]
        assert [m["id"] for m in data["after"]] == ["m3"]
        missing = json.loads(db_backend.get_whatsapp_message_context("zzz"))
        assert missing["ok"] is False

    def test_missing_db_reports_pairing_needed(self, tmp_path: Path) -> None:
        backend = WhatsAppChannelBackend(repo_dir=str(tmp_path / "empty"))
        data = json.loads(backend.search_whatsapp_contacts("x"))
        assert data["ok"] is False
        assert "Pair WhatsApp" in data["error"]


# ----------------------------------------------------------------------
# Channel protocol: poll, send, is_from_bot, connect
# ----------------------------------------------------------------------


class TestChannelProtocol:
    def test_poll_all_messages_and_cursor(self, db_backend: WhatsAppChannelBackend) -> None:
        messages, cursor = db_backend.poll_messages("", "", limit=10)
        assert [m["id"] for m in messages] == ["m4", "m6", "m1", "m2", "m3", "m5"]
        assert cursor == "2026-09-10 11:00:00+00:00"

    def test_poll_filters_by_channel_and_advances_cursor(
        self, db_backend: WhatsAppChannelBackend
    ) -> None:
        messages, cursor = db_backend.poll_messages("+1 415 555 0001", "", limit=10)
        assert [m["id"] for m in messages] == ["m1", "m2", "m3"]
        assert cursor == "2026-09-10 10:00:03+00:00"
        newer, cursor2 = db_backend.poll_messages(_ALICE, cursor, limit=10)
        assert newer == [] and cursor2 == cursor
        some, _ = db_backend.poll_messages(_ALICE, "2026-09-10 10:00:01+00:00", limit=10)
        assert [m["id"] for m in some] == ["m2", "m3"]

    def test_poll_respects_limit_and_media_placeholder(
        self, db_backend: WhatsAppChannelBackend
    ) -> None:
        messages, _ = db_backend.poll_messages(_ALICE, "", limit=1)
        assert len(messages) == 1
        assert messages[0]["id"] == "m3"
        assert messages[0]["text"] == "photo for you"
        media_only, _ = db_backend.poll_messages(_BOB, "", limit=1)
        assert media_only[0]["id"] == "m6"
        assert media_only[0]["text"] == "[audio message]"

    def test_poll_burst_larger_than_limit_is_lossless(
        self, db_backend: WhatsAppChannelBackend
    ) -> None:
        # 3 unseen messages, polled 1 at a time: each tick must deliver the
        # OLDEST unseen message so nothing is skipped.
        cursor = "2026-09-10 09:59:59+00:00"
        seen = []
        for _ in range(3):
            messages, cursor = db_backend.poll_messages(_ALICE, cursor, limit=1)
            seen += [m["id"] for m in messages]
        assert seen == ["m1", "m2", "m3"]
        rest, _ = db_backend.poll_messages(_ALICE, cursor, limit=1)
        assert rest == []

    def test_poll_without_db_returns_empty(self, tmp_path: Path) -> None:
        backend = WhatsAppChannelBackend(repo_dir=str(tmp_path / "none"))
        assert backend.poll_messages("x", "1", limit=5) == ([], "1")

    def test_is_from_bot_uses_is_from_me(self, db_backend: WhatsAppChannelBackend) -> None:
        messages, _ = db_backend.poll_messages(_ALICE, "", limit=10)
        flags = {m["id"]: db_backend.is_from_bot(m) for m in messages}
        assert flags == {"m1": False, "m2": True, "m3": False}

    def test_send_message_posts_to_bridge(self, tmp_path: Path) -> None:
        with _bridge_server({"success": True, "message": "sent"}) as (server, port):
            backend = WhatsAppChannelBackend(repo_dir=str(tmp_path), bridge_port=port)
            backend.send_message("+1 (415) 555-0001", "hello")
            path, body = server.recorded[0]  # type: ignore[attr-defined]
            assert path == "/api/send"
            assert body == {"recipient": "14155550001", "message": "hello"}

    def test_send_message_raises_on_bridge_error(self, tmp_path: Path) -> None:
        with _bridge_server({"success": False, "message": "not on WhatsApp"}) as (server, port):
            backend = WhatsAppChannelBackend(repo_dir=str(tmp_path), bridge_port=port)
            with pytest.raises(RuntimeError, match="not on WhatsApp"):
                backend.send_message("14155550001", "hello")

    def test_send_message_raises_when_bridge_down(self, tmp_path: Path) -> None:
        backend = WhatsAppChannelBackend(repo_dir=str(tmp_path), bridge_port=1)
        with pytest.raises(RuntimeError, match="start_whatsapp_bridge"):
            backend.send_message("14155550001", "hello")

    def test_connect_unpaired(self, channel_state: Any, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path / "nowhere"), "bridge_port": "1"})
        backend = WhatsAppChannelBackend()
        assert backend.connect() is False
        assert "not paired" in backend.connection_info

    def test_connect_paired_but_bridge_down(self, channel_state: Any, tmp_path: Path) -> None:
        _make_db(tmp_path)
        _make_paired_session(tmp_path)
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        backend = WhatsAppChannelBackend()
        assert backend.connect() is False
        assert "not running" in backend.connection_info

    def test_connect_paired_and_running(self, channel_state: Any, tmp_path: Path) -> None:
        _make_db(tmp_path)
        _make_paired_session(tmp_path)
        with _bridge_server({"success": True, "message": ""}) as (server, port):
            _config.save({"repo_dir": str(tmp_path), "bridge_port": str(port)})
            backend = WhatsAppChannelBackend()
            assert backend.connect() is True
            assert "connected" in backend.connection_info

    def test_connect_ignores_invalid_port_config(
        self, channel_state: Any, tmp_path: Path
    ) -> None:
        _config.save({"repo_dir": str(tmp_path / "nowhere"), "bridge_port": "abc"})
        backend = WhatsAppChannelBackend()
        assert backend.connect() is False  # unpaired; invalid port only logs


# ----------------------------------------------------------------------
# REST tools: send / download via a real local bridge-shaped server
# ----------------------------------------------------------------------


class TestRestTools:
    def setup_method(self) -> None:
        self.server, self.port = _start_bridge_server({"success": True, "message": "ok"})
        self.backend = WhatsAppChannelBackend(repo_dir="/nonexistent", bridge_port=self.port)

    def teardown_method(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def test_send_whatsapp_message(self) -> None:
        data = json.loads(self.backend.send_whatsapp_message("+14155550001", "yo"))
        assert data == {"ok": True, "message": "ok"}
        path, body = self.server.recorded[0]  # type: ignore[attr-defined]
        assert path == "/api/send"
        assert body == {"recipient": "14155550001", "message": "yo"}

    def test_send_whatsapp_message_requires_recipient(self) -> None:
        data = json.loads(self.backend.send_whatsapp_message("  ", "yo"))
        assert data["ok"] is False and "recipient" in data["error"]

    def test_send_whatsapp_file(self, tmp_path: Path) -> None:
        f = tmp_path / "pic.png"
        f.write_bytes(b"png")
        data = json.loads(self.backend.send_whatsapp_file(_GROUP, str(f)))
        assert data["ok"] is True
        _, body = self.server.recorded[0]  # type: ignore[attr-defined]
        assert body == {"recipient": _GROUP, "media_path": str(f)}

    def test_send_whatsapp_file_missing(self) -> None:
        data = json.loads(self.backend.send_whatsapp_file(_GROUP, "/no/such/file"))
        assert data["ok"] is False and "not found" in data["error"].lower()

    def test_send_whatsapp_file_requires_recipient(self, tmp_path: Path) -> None:
        f = tmp_path / "pic.png"
        f.write_bytes(b"png")
        data = json.loads(self.backend.send_whatsapp_file("", str(f)))
        assert data["ok"] is False

    def test_send_whatsapp_audio_ogg_passthrough(self, tmp_path: Path) -> None:
        f = tmp_path / "note.ogg"
        f.write_bytes(b"OggS")
        data = json.loads(self.backend.send_whatsapp_audio_message("14155550001", str(f)))
        assert data["ok"] is True
        _, body = self.server.recorded[0]  # type: ignore[attr-defined]
        assert body["media_path"] == str(f)

    def test_send_whatsapp_audio_missing_file(self) -> None:
        data = json.loads(self.backend.send_whatsapp_audio_message("1", "/no/file.mp3"))
        assert data["ok"] is False and "not found" in data["error"].lower()

    def test_send_whatsapp_audio_requires_recipient(self) -> None:
        data = json.loads(self.backend.send_whatsapp_audio_message(" ", "/no/file.mp3"))
        assert data["ok"] is False

    def test_send_whatsapp_audio_bad_conversion(self, tmp_path: Path) -> None:
        # Real ffmpeg (when installed) fails on a non-audio file; without
        # ffmpeg the OSError branch reports the same guidance.
        f = tmp_path / "not_audio.mp3"
        f.write_bytes(b"garbage")
        data = json.loads(self.backend.send_whatsapp_audio_message("1", str(f)))
        assert data["ok"] is False
        assert "send_whatsapp_file" in data["error"]

    def test_download_whatsapp_media(self) -> None:
        self.server.response_body = {  # type: ignore[attr-defined]
            "success": True,
            "message": "Successfully downloaded image media",
            "filename": "photo.jpg",
            "path": "/abs/photo.jpg",
        }
        data = json.loads(self.backend.download_whatsapp_media("m3", _ALICE))
        assert data["ok"] is True and data["file_path"] == "/abs/photo.jpg"
        path, body = self.server.recorded[0]  # type: ignore[attr-defined]
        assert path == "/api/download"
        assert body == {"message_id": "m3", "chat_jid": _ALICE}

    def test_download_whatsapp_media_bridge_down(self) -> None:
        backend = WhatsAppChannelBackend(repo_dir="/nonexistent", bridge_port=1)
        data = json.loads(backend.download_whatsapp_media("m3", _ALICE))
        assert data["ok"] is False

    def test_bridge_running_requires_405(self) -> None:
        assert self.backend._bridge_running() is True
        down = WhatsAppChannelBackend(repo_dir="/nonexistent", bridge_port=1)
        assert down._bridge_running() is False


# ----------------------------------------------------------------------
# Agent, auth tools, and tools() contract
# ----------------------------------------------------------------------

_AUTH_TOOL_NAMES = [
    "check_whatsapp_auth",
    "authenticate_whatsapp",
    "start_whatsapp_bridge",
    "get_whatsapp_qr_code",
    "wait_for_whatsapp_pairing",
    "stop_whatsapp_bridge",
    "clear_whatsapp_auth",
]


def _auth_tools(agent: WhatsAppAgent) -> dict[str, Any]:
    """Return the agent's auth tools keyed by name."""
    return {t.__name__: t for t in agent._get_auth_tools()}


class TestAgentAndAuthTools:
    def setup_method(self) -> None:
        self._state = _ChannelState()

    def teardown_method(self) -> None:
        self._state.restore()

    def test_unauthenticated_agent_exposes_only_auth_tools(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path / "nowhere")})
        agent = WhatsAppAgent()
        assert agent._is_authenticated() is False
        assert [t.__name__ for t in agent._get_tools()] == _AUTH_TOOL_NAMES

    def test_paired_agent_exposes_backend_tools(self, tmp_path: Path) -> None:
        _make_db(tmp_path)
        _make_paired_session(tmp_path)
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "18099"})
        agent = WhatsAppAgent()
        assert agent._is_authenticated() is True
        assert agent._backend._bridge_port == 18099
        names = [t.__name__ for t in agent._get_tools()]
        assert "send_whatsapp_message" in names
        assert "list_whatsapp_chats" in names
        assert "download_whatsapp_media" in names

    def test_tools_module_contract(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path / "nowhere")})
        names = [t.__name__ for t in tools()]
        assert names == _AUTH_TOOL_NAMES
        assert callable(main)

    def test_check_auth_reports_setup_needed(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path / "nowhere"), "bridge_port": "1"})
        status = json.loads(_auth_tools(WhatsAppAgent())["check_whatsapp_auth"]())
        assert status["repo_cloned"] is False
        assert status["paired"] is False
        assert "authenticate_whatsapp" in status["next_step"]

    def test_check_auth_reports_start_needed(self, tmp_path: Path) -> None:
        (tmp_path / "whatsapp-bridge").mkdir(parents=True)
        (tmp_path / "whatsapp-bridge" / "main.go").write_text("package main")
        (tmp_path / "whatsapp-bridge" / "kiss-whatsapp-bridge").write_text("bin")
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        status = json.loads(_auth_tools(WhatsAppAgent())["check_whatsapp_auth"]())
        assert status["repo_cloned"] is True and status["bridge_built"] is True
        assert status["next_step"] == "Call start_whatsapp_bridge()."

    def test_check_auth_reports_ready(self, tmp_path: Path) -> None:
        _make_db(tmp_path)
        bridge = tmp_path / "whatsapp-bridge"
        (bridge / "main.go").write_text("package main")
        (bridge / "kiss-whatsapp-bridge").write_text("bin")
        _make_paired_session(tmp_path)
        with _bridge_server({"success": True, "message": ""}) as (server, port):
            _config.save({"repo_dir": str(tmp_path), "bridge_port": str(port)})
            status = json.loads(_auth_tools(WhatsAppAgent())["check_whatsapp_auth"]())
            assert status["bridge_running"] is True and status["paired"] is True
            assert status["next_step"].startswith("Ready")

    def test_check_auth_reports_pairing_needed(self, tmp_path: Path) -> None:
        bridge = tmp_path / "whatsapp-bridge"
        bridge.mkdir(parents=True)
        (bridge / "main.go").write_text("package main")
        (bridge / "kiss-whatsapp-bridge").write_text("bin")
        with _bridge_server({"success": True, "message": ""}) as (server, port):
            _config.save({"repo_dir": str(tmp_path), "bridge_port": str(port)})
            status = json.loads(_auth_tools(WhatsAppAgent())["check_whatsapp_auth"]())
            assert status["bridge_running"] is True and status["paired"] is False
            assert "get_whatsapp_qr_code" in status["next_step"]

    def test_authenticate_rejects_bad_port(self, tmp_path: Path) -> None:
        result = json.loads(
            _auth_tools(WhatsAppAgent())["authenticate_whatsapp"](
                repo_dir=str(tmp_path), bridge_port="not-a-port"
            )
        )
        assert result["ok"] is False and "bridge_port" in result["error"]

    def test_authenticate_without_go_reports_install_steps(self, tmp_path: Path) -> None:
        # Pre-cloned repo (main.go present) but no binary and no `go` on
        # an emptied PATH: the tool must explain how to install Go.
        bridge = tmp_path / "whatsapp-bridge"
        bridge.mkdir(parents=True)
        (bridge / "main.go").write_text("package main")
        old_path = os.environ["PATH"]
        os.environ["PATH"] = str(tmp_path / "no-tools")
        try:
            result = json.loads(
                _auth_tools(WhatsAppAgent())["authenticate_whatsapp"](repo_dir=str(tmp_path))
            )
        finally:
            os.environ["PATH"] = old_path
        assert result["ok"] is False
        assert "go.dev" in result["error"]

    def test_authenticate_skips_build_when_binary_exists(self, tmp_path: Path) -> None:
        bridge = tmp_path / "whatsapp-bridge"
        bridge.mkdir(parents=True)
        (bridge / "main.go").write_text("package main")
        (bridge / "kiss-whatsapp-bridge").write_text("bin")
        result = json.loads(
            _auth_tools(WhatsAppAgent())["authenticate_whatsapp"](
                repo_dir=str(tmp_path), bridge_port="18042"
            )
        )
        assert result["ok"] is True
        cfg = _config.load()
        assert cfg is not None
        assert cfg["repo_dir"] == str(tmp_path)
        assert cfg["bridge_port"] == "18042"

    def test_authenticate_clone_failure_reported(self, tmp_path: Path) -> None:
        # Cloning into a path whose parent is an existing *file* makes the
        # real `git clone` fail; the error must be surfaced.
        parent = tmp_path / "blocker"
        parent.write_text("i am a file")
        result = json.loads(
            _auth_tools(WhatsAppAgent())["authenticate_whatsapp"](
                repo_dir=str(parent / "repo")
            )
        )
        assert result["ok"] is False

    def test_start_bridge_requires_build(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path / "nowhere"), "bridge_port": "1"})
        result = json.loads(_auth_tools(WhatsAppAgent())["start_whatsapp_bridge"]())
        assert result["ok"] is False and "authenticate_whatsapp" in result["error"]

    def test_start_bridge_short_circuits_when_running(self, tmp_path: Path) -> None:
        with _bridge_server({"success": True, "message": ""}) as (server, port):
            _config.save({"repo_dir": str(tmp_path), "bridge_port": str(port)})
            result = json.loads(_auth_tools(WhatsAppAgent())["start_whatsapp_bridge"]())
            assert result == {"ok": True, "message": "Bridge already running."}

    def test_qr_code_without_log(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        result = json.loads(_auth_tools(WhatsAppAgent())["get_whatsapp_qr_code"]())
        assert result["ok"] is False and "start_whatsapp_bridge" in result["error"]

    def test_qr_code_from_bridge_log(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        _bridge_log_path().parent.mkdir(parents=True, exist_ok=True)
        _bridge_log_path().write_text(_BRIDGE_LOG_WITH_QR, encoding="utf-8")
        result = json.loads(_auth_tools(WhatsAppAgent())["get_whatsapp_qr_code"]())
        assert result["ok"] is True
        page = Path(result["qr_page"])
        assert page.exists() and _QR_BLOCK in page.read_text(encoding="utf-8")

    def test_qr_code_when_no_qr_yet(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        _bridge_log_path().parent.mkdir(parents=True, exist_ok=True)
        _bridge_log_path().write_text("starting up...", encoding="utf-8")
        result = json.loads(_auth_tools(WhatsAppAgent())["get_whatsapp_qr_code"]())
        assert result["ok"] is False and "No QR code" in result["error"]

    def test_qr_code_when_already_paired(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        _bridge_log_path().parent.mkdir(parents=True, exist_ok=True)
        _bridge_log_path().write_text(
            "\u2713 Connected to WhatsApp! Type 'help' for commands.", encoding="utf-8"
        )
        result = json.loads(_auth_tools(WhatsAppAgent())["get_whatsapp_qr_code"]())
        assert result["ok"] is True and "Already paired" in result["message"]

    def test_wait_for_pairing_success_rewrites_page(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        _bridge_log_path().parent.mkdir(parents=True, exist_ok=True)
        _write_qr_html(_QR_BLOCK)
        _bridge_log_path().write_text(
            _BRIDGE_LOG_WITH_QR + "\nSuccessfully connected and authenticated!\n",
            encoding="utf-8",
        )
        result = json.loads(
            _auth_tools(WhatsAppAgent())["wait_for_whatsapp_pairing"](timeout=3)
        )
        assert result["ok"] is True and "paired" in result["message"]
        assert "linked successfully" in _qr_html_path().read_text(encoding="utf-8")

    def test_wait_for_pairing_bridge_timeout(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        _bridge_log_path().parent.mkdir(parents=True, exist_ok=True)
        _bridge_log_path().write_text(
            "Timeout waiting for QR code scan", encoding="utf-8"
        )
        result = json.loads(
            _auth_tools(WhatsAppAgent())["wait_for_whatsapp_pairing"](timeout=3)
        )
        assert result["ok"] is False and "fresh QR" in result["error"]

    def test_wait_for_pairing_still_waiting_refreshes_qr(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        _bridge_log_path().parent.mkdir(parents=True, exist_ok=True)
        _bridge_log_path().write_text(_BRIDGE_LOG_WITH_QR, encoding="utf-8")
        _bridge_pid_path().write_text(str(os.getpid()), encoding="utf-8")
        result = json.loads(
            _auth_tools(WhatsAppAgent())["wait_for_whatsapp_pairing"](timeout=1)
        )
        assert result["ok"] is False and "Still waiting" in result["error"]
        assert _QR_BLOCK in _qr_html_path().read_text(encoding="utf-8")

    def test_wait_for_pairing_without_log(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        result = json.loads(
            _auth_tools(WhatsAppAgent())["wait_for_whatsapp_pairing"](timeout=1)
        )
        assert result["ok"] is False and "start_whatsapp_bridge" in result["error"]

    def test_stop_bridge_without_pid(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        result = json.loads(_auth_tools(WhatsAppAgent())["stop_whatsapp_bridge"]())
        assert result["ok"] is False and "PID" in result["error"]

    def test_stop_bridge_with_stale_pid(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        _bridge_pid_path().parent.mkdir(parents=True, exist_ok=True)
        _bridge_pid_path().write_text("999999999", encoding="utf-8")
        result = json.loads(_auth_tools(WhatsAppAgent())["stop_whatsapp_bridge"]())
        assert result["ok"] is True and "not running" in result["message"]

    def test_clear_auth_removes_session(self, tmp_path: Path) -> None:
        _make_db(tmp_path)
        store = tmp_path / "whatsapp-bridge" / "store"
        _make_paired_session(tmp_path)
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        _write_qr_html(_QR_BLOCK)
        agent = WhatsAppAgent()
        assert agent._is_authenticated() is True
        message = _auth_tools(agent)["clear_whatsapp_auth"]()
        assert "Linked devices" in message
        assert not store.exists()
        assert not _qr_html_path().exists()
        assert _config.load() is None
        assert agent._is_authenticated() is False

    def test_unpaired_session_file_is_not_authenticated(self, tmp_path: Path) -> None:
        # The bridge creates whatsapp.db before the QR scan; a device table
        # without rows (or an empty file) must NOT count as paired.
        _make_paired_session(tmp_path, paired=False)
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        agent = WhatsAppAgent()
        assert agent._is_authenticated() is False
        (tmp_path / "whatsapp-bridge" / "store" / "whatsapp.db").unlink()
        (tmp_path / "whatsapp-bridge" / "store" / "whatsapp.db").touch()
        assert WhatsAppAgent()._is_authenticated() is False

    def test_clear_refuses_unowned_running_bridge(self, tmp_path: Path) -> None:
        _make_db(tmp_path)
        _make_paired_session(tmp_path)
        with _bridge_server({"success": True, "message": ""}) as (server, port):
            _config.save({"repo_dir": str(tmp_path), "bridge_port": str(port)})
            message = _auth_tools(WhatsAppAgent())["clear_whatsapp_auth"]()
        assert "Refusing to clear" in message
        assert (tmp_path / "whatsapp-bridge" / "store" / "whatsapp.db").exists()

    def test_start_bridge_guards_against_double_start(self, tmp_path: Path) -> None:
        # A live PID without a REST answer means the first bridge is still
        # pairing; a second start must not launch a duplicate.
        bridge = tmp_path / "whatsapp-bridge"
        bridge.mkdir(parents=True)
        (bridge / "main.go").write_text("package main")
        (bridge / "kiss-whatsapp-bridge").write_text("bin")
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        _bridge_pid_path().parent.mkdir(parents=True, exist_ok=True)
        _bridge_pid_path().write_text(str(os.getpid()), encoding="utf-8")
        result = json.loads(_auth_tools(WhatsAppAgent())["start_whatsapp_bridge"]())
        assert result["ok"] is True
        assert "already starting or waiting" in result["message"]

    def test_wait_for_pairing_detects_dead_bridge(self, tmp_path: Path) -> None:
        _config.save({"repo_dir": str(tmp_path), "bridge_port": "1"})
        _bridge_log_path().parent.mkdir(parents=True, exist_ok=True)
        _bridge_log_path().write_text(_BRIDGE_LOG_WITH_QR, encoding="utf-8")
        _bridge_pid_path().write_text("999999999", encoding="utf-8")
        result = json.loads(
            _auth_tools(WhatsAppAgent())["wait_for_whatsapp_pairing"](timeout=5)
        )
        assert result["ok"] is False and "no longer running" in result["error"]

    def test_main_without_args_exits(self) -> None:
        import sys

        old_argv = sys.argv
        sys.argv = ["kiss-whatsapp"]
        try:
            with pytest.raises(SystemExit):
                main()
        finally:
            sys.argv = old_argv

    def test_channel_system_prompt_mentions_pairing_flow(self) -> None:
        prompt = WhatsAppAgent.channel_system_prompt
        assert "QR" in prompt and "wait_for_whatsapp_pairing" in prompt
