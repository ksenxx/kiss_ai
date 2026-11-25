# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Read-only access to the macOS Messages database (``~/Library/Messages/chat.db``).

The Messages app's AppleScript dictionary cannot read message history, so
the iMessage tools that list conversations and messages query the SQLite
database directly, opened read-only.  Reading it requires Full Disk Access
for the process (Terminal, the IDE or the Python binary running the agent).
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHAT_DB = Path.home() / "Library" / "Messages" / "chat.db"

# Messages stores dates as nanoseconds since 2001-01-01 00:00:00 UTC.
_APPLE_EPOCH = 978307200
_GROUP_CHAT_STYLE = 43

_CONVERSATIONS_SQL = """
SELECT c.ROWID, c.guid, c.chat_identifier, c.display_name, c.service_name, c.style,
       MAX(cmj.message_date) AS last_date
FROM chat c JOIN chat_message_join cmj ON cmj.chat_id = c.ROWID
GROUP BY c.ROWID
ORDER BY last_date DESC
LIMIT ?
"""

_PARTICIPANTS_SQL = """
SELECT h.id FROM chat_handle_join chj JOIN handle h ON h.ROWID = chj.handle_id
WHERE chj.chat_id = ? ORDER BY h.id
"""

_MESSAGES_SQL = """
SELECT m.ROWID, m.date, m.is_from_me, m.text, m.attributedBody,
       m.associated_message_type, m.service, h.id AS sender,
       c.guid, c.chat_identifier, c.display_name
FROM message m
JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
JOIN chat c ON c.ROWID = cmj.chat_id
LEFT JOIN handle h ON h.ROWID = m.handle_id
WHERE (? = '' OR c.guid = ? OR c.chat_identifier = ?) AND m.date >= ?
ORDER BY m.date DESC, m.ROWID DESC
LIMIT ?
"""

_ATTACHMENTS_SQL = """
SELECT a.transfer_name, a.mime_type FROM message_attachment_join maj
JOIN attachment a ON a.ROWID = maj.attachment_id WHERE maj.message_id = ?
"""


FULL_DISK_ACCESS_HINT = (
    "Reading the Messages database needs Full Disk Access for the app running "
    "the agent (System Settings > Privacy & Security > Full Disk Access)."
)


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open *db_path* read-only."""
    return sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)


def apple_time_to_iso(value: int) -> str:
    """Convert a Messages date (nanoseconds since 2001-01-01 UTC) to local ISO 8601."""
    seconds = value / 1e9 + _APPLE_EPOCH
    return datetime.fromtimestamp(seconds, UTC).astimezone().isoformat(timespec="seconds")


def iso_to_apple_time(value: str) -> int:
    """Convert an ISO 8601 date (naive means UTC) to a Messages date in nanoseconds."""
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int((moment.timestamp() - _APPLE_EPOCH) * 1e9)


def _is_tapback(associated_message_type: int) -> bool:
    """Return True for a tapback: types 2000-2999 add one, 3000-3999 remove one.

    Types 0, 2 and 3 are ordinary messages (2 and 3 are app messages).
    """
    return 2000 <= associated_message_type < 4000


def decode_attributed_body(blob: bytes | None) -> str:
    """Extract the plain text from a message's ``attributedBody`` typedstream blob.

    Recent macOS versions leave ``message.text`` NULL and store the text
    only in this NSAttributedString archive: after the ``NSString`` class
    name comes a ``+`` marker, a length (one byte, or ``0x81`` followed by
    two little-endian bytes, or ``0x82`` followed by four) and the UTF-8 text.

    Args:
        blob: The raw ``attributedBody`` column value.

    Returns:
        The decoded text, or "" when the blob holds none.
    """
    if not blob:
        return ""
    start = blob.find(b"NSString")
    if start < 0:
        return ""
    i = blob.find(b"+", start) + 1
    if i == 0 or i >= len(blob):
        return ""
    length = blob[i]
    i += 1
    if length == 0x81:
        length = int.from_bytes(blob[i : i + 2], "little")
        i += 2
    elif length == 0x82:
        length = int.from_bytes(blob[i : i + 4], "little")
        i += 4
    return blob[i : i + length].decode("utf-8", errors="replace")


def list_conversations(db_path: Path, limit: int) -> list[dict[str, Any]]:
    """Return the *limit* most recently active chats, newest first.

    Args:
        db_path: Path of the Messages ``chat.db``.
        limit: Maximum number of chats.

    Returns:
        One dict per chat: id (the chat guid), identifier, name,
        service, is_group, participants and last_message_at.
    """
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(_CONVERSATIONS_SQL, (limit,)).fetchall()
        return [
            {
                "id": guid,
                "identifier": identifier,
                "name": name or "",
                "service": service,
                "is_group": style == _GROUP_CHAT_STYLE,
                "participants": [
                    r[0] for r in conn.execute(_PARTICIPANTS_SQL, (rowid,)).fetchall()
                ],
                "last_message_at": apple_time_to_iso(last_date),
            }
            for rowid, guid, identifier, name, service, style, last_date in rows
        ]


def list_messages(db_path: Path, chat_id: str, since: str, limit: int) -> list[dict[str, Any]]:
    """Return the *limit* newest messages, in chronological order.

    Args:
        db_path: Path of the Messages ``chat.db``.
        chat_id: A chat guid or chat identifier (phone number, email or
            ``chatNNN`` for groups); "" for every chat.
        since: ISO 8601 date; only messages at or after it are returned.
            "" for no lower bound.
        limit: Maximum number of messages.

    Returns:
        One dict per message: id, date, chat_id, chat_identifier,
        chat_name, sender ("me" for your own messages), is_from_me,
        service, text, is_reaction and attachments.
    """
    since_value = iso_to_apple_time(since) if since else 0
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            _MESSAGES_SQL, (chat_id, chat_id, chat_id, since_value, limit)
        ).fetchall()
        messages = []
        for (
            rowid, date, is_from_me, text, body, reaction, service, sender,
            guid, identifier, name,
        ) in reversed(rows):
            attachments = conn.execute(_ATTACHMENTS_SQL, (rowid,)).fetchall()
            messages.append(
                {
                    "id": rowid,
                    "date": apple_time_to_iso(date),
                    "chat_id": guid,
                    "chat_identifier": identifier,
                    "chat_name": name or "",
                    "sender": "me" if is_from_me else (sender or ""),
                    "is_from_me": bool(is_from_me),
                    "service": service,
                    "text": text or decode_attributed_body(body),
                    "is_reaction": _is_tapback(reaction),
                    "attachments": [
                        {"name": a_name or "", "mime_type": mime or ""}
                        for a_name, mime in attachments
                    ],
                }
            )
        return messages
