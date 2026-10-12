# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A failed legacy-journal adoption must not break ``_get_db``.

``_get_db`` runs :func:`_adopt_legacy_journal_snapshots` on every NEW
connection, and that helper writes to ``replayed_journals``.  When the
write fails (in production: the 30 s busy timeout expires behind another
process's write transaction, or the database is in a ``disk I/O error``
state) the exception used to escape ``_get_db`` AFTER the connection was
opened and registered in ``_open_conns`` but BEFORE it was cached on the
thread, so every call leaked one open connection and no caller ever got
a handle.  The write is made to fail here with a ``RAISE(ABORT)`` trigger
installed through a separate connection — database state, not a patch of
the code under test — because a real busy-timeout expiry takes 30 s.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th


@pytest.fixture
def kiss_dir(monkeypatch: pytest.MonkeyPatch):
    tmp = tempfile.mkdtemp()
    home = Path(tmp) / ".kiss"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    th._KISS_DIR = home
    th._db_conn = None
    try:
        yield home
    finally:
        th._close_db()
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved
        shutil.rmtree(tmp, ignore_errors=True)


def _open_conns_for(path: Path) -> int:
    return sum(1 for _c, _o, p in th._open_conns.values() if p == str(path))


def test_adoption_write_failure_neither_leaks_nor_withholds_the_connection(
    kiss_dir: Path,
) -> None:
    db_path = kiss_dir / "history.db"
    th._DB_PATH = db_path
    task_id, _chat = th._add_task("legacy replay")
    legacy = f"sorcar.db.failed_events.jsonl.consumed-{1:020d}-{os.getpid()}-" + "a" * 32
    # The snapshot was committed under its legacy name by a replayer
    # that died before unlinking it: the marker must be re-keyed, which
    # is the write the trigger below refuses.
    th._get_db().execute(
        "INSERT INTO replayed_journals (snapshot, timestamp) VALUES (?, 2.0)",
        (legacy,),
    )
    th._close_db()
    (kiss_dir / legacy).write_text(
        json.dumps(
            {
                "task_id": task_id,
                "event_json": json.dumps({"type": "result", "text": "x"}),
                "timestamp": 2.0,
                "origin_db_path": str(kiss_dir / "sorcar.db"),
            }
        )
        + "\n"
    )
    blocker = sqlite3.connect(str(db_path))
    blocker.execute(
        "CREATE TRIGGER audit_block BEFORE INSERT ON replayed_journals "
        "BEGIN SELECT RAISE(ABORT, 'adoption refused'); END"
    )
    blocker.commit()
    blocker.close()

    before = _open_conns_for(db_path)
    first = th._get_db()
    second = th._get_db()
    assert first is second, "the thread must cache the connection it opened"
    assert _open_conns_for(db_path) == before + 1, "exactly one connection is registered"
    assert first.execute("SELECT 1").fetchone()[0] == 1
    # The snapshot is left under its legacy name for a later connection.
    assert (kiss_dir / legacy).exists()

    # Once the write succeeds again, the next NEW connection adopts it.
    th._close_db()
    blocker = sqlite3.connect(str(db_path))
    blocker.execute("DROP TRIGGER audit_block")
    blocker.commit()
    blocker.close()
    th._get_db()
    assert not (kiss_dir / legacy).exists()
    adopted = list(kiss_dir.glob("history.db.failed_events.jsonl.consumed-*"))
    assert len(adopted) == 1
    # The marker followed the rename, so the committed rows are not inserted again.
    th._replay_failed_events()
    assert not adopted[0].exists()
    rows = (
        th._get_db()
        .execute(
            "SELECT event_json FROM events WHERE task_id = ?",
            (task_id,),
        )
        .fetchall()
    )
    assert rows == []
