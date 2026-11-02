"""A failed ``make_messages_db`` leaves the fake bridge database writable.

pytest keeps the traceback of a failed fixture or test, and the traceback
keeps the helper's frame locals alive.  A connection left open there with
its half-done insert transaction held SQLite's write lock, so every later
writer of the same database got ``database is locked``.  The helper must
close its connection on the failure path too.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from kiss.tests.agents.third_party_agents.whatsapp_bridge import make_messages_db


def test_failed_setup_releases_the_database(tmp_path: Path) -> None:
    """After a duplicate-chat failure, a fresh connection can write at once."""
    with pytest.raises(sqlite3.IntegrityError) as info:
        make_messages_db(tmp_path, [("same", "first", "1"), ("same", "duplicate", "2")], [])
    # ``info`` keeps the traceback, and with it the helper's ``conn``, as pytest does.
    db = tmp_path / "whatsapp-bridge" / "store" / "messages.db"
    # ``closing``: the connection is closed even when the insert fails (the very
    # defect under test), not left in the retained traceback like the helper's was.
    with closing(sqlite3.connect(db, timeout=0.05)) as other, other:
        other.execute("INSERT INTO chats VALUES (?, ?, ?)", ("other", "name", "3"))
    assert "chats.jid" in str(info.value)
