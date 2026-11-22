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

Also here: a snapshot is not adopted while a process of the previous
version replays it.  That process (kept working by the ``sorcar.db``
symlink the database rename leaves) commits the snapshot's rows with a
marker under the legacy name; a snapshot renamed between its read and
that commit carried no marker to the new name, and the next replay under
the new name inserted the same rows again.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.tests.conftest import posix_only
from kiss.tests.persistence_helpers import close_db

# A process of the version before 2026.10.2: its database is ``sorcar.db`` (argv[1]).
# It replays its journal and stops where a snapshot is claimed and read but not yet
# committed, until its stdin has a line (a barrier on a function's entry, not a patch).
_LEGACY_REPLAYER = """
import sys
from pathlib import Path

from kiss.agents.sorcar import persistence as th


def park(code, offset):
    print("claimed", flush=True)
    sys.stdin.readline()
    return sys.monitoring.DISABLE


th._DB_PATH = Path(sys.argv[1])
th._get_db()
sys.monitoring.use_tool_id(4, "kiss-tests")
sys.monitoring.register_callback(4, sys.monitoring.events.PY_START, park)
sys.monitoring.set_local_events(
    4, th._write_event_batch.__code__, sys.monitoring.events.PY_START
)
th._replay_failed_events()
"""


@pytest.fixture
def kiss_dir(monkeypatch: pytest.MonkeyPatch):
    tmp = tempfile.mkdtemp()
    home = Path(tmp) / ".kiss"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    saved = (th._DB_PATH, th._KISS_DIR)
    th._KISS_DIR = home
    try:
        yield home
    finally:
        close_db()
        (th._DB_PATH, th._KISS_DIR) = saved
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
    close_db()
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
    close_db()
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


@posix_only("the legacy database name is kept as a symlink, and the journal lock is flock")
@pytest.mark.parametrize("replayer_dies", [False, True])
def test_a_snapshot_a_previous_version_is_replaying_is_left_to_it(
    kiss_dir: Path, replayer_dies: bool
) -> None:
    """A new connection that meets a legacy snapshot mid-replay waits: its rows land once.

    A replayer killed before its commit never lands them: the lock it held is free, and the
    next flush of the connection that waited adopts and replays the snapshot (S9-3).
    """
    th._DB_PATH = kiss_dir / "history.db"
    task_id, _chat = th._add_task("legacy replay")
    close_db()
    legacy_db = kiss_dir / "sorcar.db"
    legacy_db.symlink_to("history.db")  # what the rename leaves for a running old version
    legacy = f"sorcar.db.failed_events.jsonl.consumed-{1:020d}-{os.getpid()}-" + "c" * 32
    event = {
        "task_id": task_id,
        "event_json": json.dumps({"type": "result", "text": "x"}),
        "timestamp": 2.0,
        "origin_db_path": str(legacy_db),
    }
    (kiss_dir / legacy).write_text(json.dumps(event) + "\n")
    with subprocess.Popen(
        [sys.executable, "-c", _LEGACY_REPLAYER, str(legacy_db)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as replayer:
        assert replayer.stdout is not None
        watchdog = threading.Timer(60, replayer.kill)
        watchdog.start()
        try:
            assert replayer.stdout.readline() == "claimed\n"  # "" once it died or was killed
            waited = th._get_db()  # this version's new connection, the snapshot uncommitted
            if replayer_dies:
                replayer.kill()  # where it is parked; the kernel frees the lock it held
            replayer.communicate("\n")
        finally:
            watchdog.cancel()
    if replayer_dies:
        assert (kiss_dir / legacy).exists()
        th._flush_chat_events()  # no new connection: the one that waited is still the thread's
        assert th._get_db() is waited
    else:
        assert replayer.returncode == 0
        close_db()
        th._replay_failed_events()  # on a new connection, which adopts what is left to adopt
    db = th._get_db()
    stored = db.execute("SELECT COUNT(*) FROM events WHERE task_id = ?", (task_id,)).fetchone()
    assert stored[0] == 1
    assert list(kiss_dir.glob("*.failed_events.jsonl.consumed-*")) == []
    assert db.execute("SELECT snapshot FROM replayed_journals").fetchall() == []


def test_a_flush_that_cannot_open_the_database_leaves_the_snapshot_to_the_next(
    kiss_dir: Path,
) -> None:
    """The flush's adoption needs the thread's connection; without one it waits, not raises."""
    db_path = kiss_dir / "history.db"
    th._DB_PATH = db_path
    task_id, _chat = th._add_task("legacy replay")
    close_db()
    legacy = kiss_dir / (
        f"sorcar.db.failed_events.jsonl.consumed-{1:020d}-{os.getpid()}-" + "d" * 32
    )
    event = {
        "task_id": task_id,
        "event_json": json.dumps({"type": "result", "text": "x"}),
        "timestamp": 2.0,
        "origin_db_path": str(kiss_dir / "sorcar.db"),
    }
    legacy.write_text(json.dumps(event) + "\n")
    db_path.rename(kiss_dir / "elsewhere.db")
    db_path.mkdir()  # nothing SQLite can open

    th._flush_chat_events()  # an exception here would also end the event writer

    assert legacy.exists()
    db_path.rmdir()
    (kiss_dir / "elsewhere.db").rename(db_path)
    th._flush_chat_events()
    stored = th._get_db().execute("SELECT COUNT(*) FROM events WHERE task_id = ?", (task_id,))
    assert stored.fetchone()[0] == 1
    assert list(kiss_dir.glob("*.failed_events.jsonl.consumed-*")) == []


def test_a_flush_of_a_database_under_the_legacy_name_keeps_its_markers(kiss_dir: Path) -> None:
    """Nothing is adopted from a database to itself: a committed snapshot is not replayed again."""
    th._DB_PATH = kiss_dir / "sorcar.db"
    task_id, _chat = th._add_task("committed replay")
    committed = f"sorcar.db.failed_events.jsonl.consumed-{1:020d}-{os.getpid()}-" + "e" * 32
    db = th._get_db()
    db.execute("INSERT INTO replayed_journals (snapshot, timestamp) VALUES (?, 2.0)", (committed,))
    event = {
        "task_id": task_id,
        "event_json": json.dumps({"type": "result", "text": "twice"}),
        "timestamp": 2.0,
        "origin_db_path": str(th._DB_PATH),
    }
    (kiss_dir / committed).write_text(json.dumps(event) + "\n")

    th._flush_chat_events()

    stored = db.execute("SELECT COUNT(*) FROM events WHERE task_id = ?", (task_id,)).fetchone()
    assert stored[0] == 0
    assert not (kiss_dir / committed).exists()
