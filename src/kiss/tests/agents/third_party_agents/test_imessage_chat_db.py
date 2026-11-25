"""End-to-end tests of the iMessage list_conversations / list_messages tools.

Each test builds a real SQLite database with the Messages ``chat.db``
tables the tools read, points a real ``IMessageChannelBackend`` at it and
checks the JSON the tools return.  The tools return a platform error off
macOS, so the tests are skipped there.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from kiss.agents.third_party_agents.imessage import chat_db
from kiss.agents.third_party_agents.imessage.imessage_sea import IMessageChannelBackend

_SCHEMA = """
CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, chat_identifier TEXT,
    display_name TEXT, service_name TEXT, style INTEGER);
CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
CREATE TABLE message (ROWID INTEGER PRIMARY KEY, date INTEGER, is_from_me INTEGER,
    text TEXT, attributedBody BLOB, associated_message_type INTEGER, service TEXT,
    handle_id INTEGER);
CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER, message_date INTEGER);
CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY, transfer_name TEXT, mime_type TEXT);
CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
"""

# 2026-11-22T05:00:00Z as a Messages date (nanoseconds since 2001-01-01 UTC).
_BASE = chat_db.iso_to_apple_time("2026-11-22T05:00:00Z")
_HOUR = 3600 * 10**9


def _body(text: str) -> bytes:
    """Build an attributedBody typedstream blob holding *text*."""
    data = text.encode()
    if len(data) < 0x81:
        length = bytes([len(data)])
    elif len(data) < 0x10000:
        length = b"\x81" + len(data).to_bytes(2, "little")
    else:
        length = b"\x82" + len(data).to_bytes(4, "little")
    return (
        b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x12NSAttributedString"
        b"\x00\x84\x84\x08NSObject\x00\x85\x92\x84\x84\x84\x08NSString\x01\x94\x84\x01+"
        + length + data + b"\x86\x84\x02iI"
    )


def _make_db(path: Path) -> None:
    """Create a small chat.db with one direct chat and one group chat."""
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.executemany(
        "INSERT INTO chat VALUES (?,?,?,?,?,?)",
        [
            (1, "any;-;+15550001", "+15550001", "", "iMessage", 45),
            (2, "any;+;chat42", "chat42", "Family", "iMessage", 43),
        ],
    )
    conn.executemany("INSERT INTO handle VALUES (?,?)", [(1, "+15550001"), (2, "a@b.com")])
    conn.executemany("INSERT INTO chat_handle_join VALUES (?,?)", [(1, 1), (2, 2), (2, 1)])
    messages = [
        # rowid, date, from_me, text, body, reaction, handle, chat
        (1, _BASE - _HOUR, 1, "old hello", None, 0, 0, 1),
        (2, _BASE + _HOUR, 0, None, _body("body only"), 3, 1, 1),  # app message
        (3, _BASE + 2 * _HOUR, 0, "Liked “old hello”", None, 2000, 1, 1),
        (4, _BASE + 3 * _HOUR, 0, "\ufffc", None, 0, 2, 2),
        (5, _BASE + 4 * _HOUR, 1, None, _body("x" * 200), 0, 0, 2),
        (6, _BASE + 5 * _HOUR, 0, None, _body("y" * 70000), 0, 2, 2),
        (7, _BASE + 6 * _HOUR, 0, None, b"no class name", 0, 2, 2),
        (8, _BASE + 7 * _HOUR, 0, None, b"NSString", 0, 2, 2),
        (9, _BASE + 8 * _HOUR, 0, None, b"NSString+", 0, 2, 2),
        (10, _BASE + 9 * _HOUR, 0, None, None, 0, 2, 2),
    ]
    for rowid, date, from_me, text, body, reaction, handle, chat in messages:
        conn.execute(
            "INSERT INTO message VALUES (?,?,?,?,?,?,?,?)",
            (rowid, date, from_me, text, body, reaction, "iMessage", handle),
        )
        conn.execute("INSERT INTO chat_message_join VALUES (?,?,?)", (chat, rowid, date))
    conn.execute("INSERT INTO attachment VALUES (1, 'IMG_1.png', 'image/png')")
    conn.execute("INSERT INTO message_attachment_join VALUES (4, 1)")
    conn.commit()
    conn.close()


@unittest.skipUnless(sys.platform == "darwin", "iMessage tools are macOS only")
class TestIMessageChatDbTools(unittest.TestCase):
    """list_conversations and list_messages read the Messages database."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.backend = IMessageChannelBackend()
        self.backend._chat_db = Path(self._tmp.name) / "chat db.sqlite"
        _make_db(self.backend._chat_db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _messages(self, **kwargs: object) -> list[dict]:
        result = json.loads(self.backend.list_messages(**kwargs))  # type: ignore[arg-type]
        self.assertTrue(result["ok"], result)
        return list(result["messages"])

    def test_list_conversations_newest_first(self) -> None:
        result = json.loads(self.backend.list_conversations())
        self.assertTrue(result["ok"], result)
        group, direct = result["conversations"]
        self.assertEqual(group["id"], "any;+;chat42")
        self.assertEqual(group["name"], "Family")
        self.assertTrue(group["is_group"])
        self.assertEqual(group["participants"], ["+15550001", "a@b.com"])
        self.assertEqual(direct["identifier"], "+15550001")
        self.assertFalse(direct["is_group"])
        self.assertEqual(direct["participants"], ["+15550001"])
        self.assertEqual(
            direct["last_message_at"],
            chat_db.apple_time_to_iso(_BASE + 2 * _HOUR),
        )
        limited = json.loads(self.backend.list_conversations(limit=1))["conversations"]
        self.assertEqual([c["id"] for c in limited], ["any;+;chat42"])

    def test_list_messages_of_one_chat(self) -> None:
        by_number = self._messages(chat_id="+15550001")
        self.assertEqual(by_number, self._messages(chat_id="any;-;+15550001"))
        self.assertEqual([m["id"] for m in by_number], [1, 2, 3])
        first, second, reaction = by_number
        self.assertEqual((first["sender"], first["is_from_me"]), ("me", True))
        self.assertEqual(first["text"], "old hello")
        self.assertEqual((second["sender"], second["text"]), ("+15550001", "body only"))
        self.assertFalse(second["is_reaction"])
        self.assertTrue(reaction["is_reaction"])
        self.assertEqual(first["chat_name"], "")
        self.assertEqual(first["date"], chat_db.apple_time_to_iso(_BASE - _HOUR))

    def test_list_messages_decodes_bodies_and_attachments(self) -> None:
        texts = {m["id"]: m for m in self._messages(chat_id="chat42")}
        self.assertEqual(texts[4]["attachments"], [{"name": "IMG_1.png", "mime_type": "image/png"}])
        self.assertEqual(texts[4]["chat_name"], "Family")
        self.assertEqual(texts[5]["text"], "x" * 200)
        self.assertEqual(texts[6]["text"], "y" * 70000)
        self.assertEqual(texts[7]["text"], "")
        self.assertEqual(texts[8]["text"], "")
        self.assertEqual(texts[9]["text"], "")
        self.assertEqual(texts[10]["text"], "")

    def test_list_messages_since_and_limit(self) -> None:
        self.assertEqual(len(self._messages()), 10)
        since_utc = self._messages(since="2026-11-22T05:00:00Z")
        self.assertEqual([m["id"] for m in since_utc], [2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(since_utc, self._messages(since="2026-11-22T05:00:00"))
        self.assertEqual(since_utc, self._messages(since="2026-11-21T21:00:00-08:00"))
        newest = self._messages(since="2026-11-22T05:00:00Z", limit=2)
        self.assertEqual([m["id"] for m in newest], [9, 10])

    def test_unreadable_database_reports_full_disk_access(self) -> None:
        self.backend._chat_db = Path(self._tmp.name) / "missing" / "chat.db"
        for raw in (self.backend.list_messages(), self.backend.list_conversations()):
            result = json.loads(raw)
            self.assertFalse(result["ok"])
            self.assertIn("Full Disk Access", result["error"])
        bad_date = json.loads(self.backend.list_messages(since="yesterday"))
        self.assertFalse(bad_date["ok"])
        self.assertIn("yesterday", bad_date["error"])
        self.assertNotIn("Full Disk Access", bad_date["error"])


if __name__ == "__main__":
    unittest.main()
