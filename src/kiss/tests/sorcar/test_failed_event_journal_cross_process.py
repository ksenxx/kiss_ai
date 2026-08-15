# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The failed-event journal is shared by every Sorcar process.

The history database and its ``<db>.failed_events.jsonl`` sidecar are
written by the ``kiss-web`` daemon, by ``kiss`` CLI runs and by VS Code
reloads at the same time.  A process replaying the sidecar must never
delete a batch another process journalled while the replay was in
flight — that batch is a chat transcript nobody can recover.

Everything here is real: a real second OS process, a real SQLite
database with its real write lock, a real sidecar file and the real
replay path.  Nothing is mocked or patched.
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
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as persistence

# How long the parent waits for the child to finish appending before it
# releases the database lock.  A process that is free to append does so
# in milliseconds; one that correctly waits for the replay to consume
# the sidecar cannot, and the wait simply expires.
_APPEND_GRACE_S = 3.0

_CHILD_SCRIPT = """
import json
import sys

from kiss.agents.sorcar import persistence

started, appended, task_id = sys.argv[1], sys.argv[2], sys.argv[3]
origin = str(persistence._DB_PATH)
open(started, "w").close()
persistence._journal_failed_events(
    [(task_id, json.dumps({"type": "text", "text": "from the second process"}),
      1234.5, origin)],
    4,
)
open(appended, "w").close()
"""


@pytest.fixture
def kiss_home() -> Iterator[Path]:
    """Point persistence at a throwaway KISS_HOME for this test."""
    home = Path(tempfile.mkdtemp(prefix="kiss-journal-xproc-"))
    saved_env = os.environ.get("KISS_HOME")
    saved = (persistence._DB_PATH, persistence._db_conn, persistence._KISS_DIR)
    os.environ["KISS_HOME"] = str(home)
    persistence._KISS_DIR = home
    persistence._DB_PATH = home / "sorcar.db"
    persistence._db_conn = None
    try:
        yield home
    finally:
        if persistence._db_conn is not None:
            try:
                persistence._db_conn.close()
            except Exception:  # pragma: no cover — cleanup best-effort
                pass
        (
            persistence._DB_PATH,
            persistence._db_conn,
            persistence._KISS_DIR,
        ) = saved
        if saved_env is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = saved_env
        shutil.rmtree(home, ignore_errors=True)


def _journalled_event(task_id: str, text: str) -> tuple[str, str, float, str]:
    """Build one journal row for *task_id*."""
    return (
        task_id,
        json.dumps({"type": "text", "text": text}),
        1234.5,
        str(persistence._DB_PATH),
    )


def _event_texts(task_id: str) -> list[str]:
    """Return the ``text`` of every persisted event of *task_id*."""
    db = persistence._get_db()
    rows = db.execute(
        "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq",
        (task_id,),
    ).fetchall()
    return [json.loads(row[0]).get("text", "") for row in rows]


def _journal_residue(home: Path) -> list[str]:
    """Return leftover journal files, ignoring the persistent lock file."""
    return sorted(
        p.name
        for p in home.glob("sorcar.db.failed*")
        if not p.name.endswith(".lock")
    )


def _wait_for(path: Path, timeout: float) -> bool:
    """Wait until *path* exists, returning whether it appeared."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return path.exists()


def test_replay_never_deletes_another_processs_journalled_events(
    kiss_home: Path,
) -> None:
    """A batch journalled during a replay is still recoverable."""
    local_task, _chat = persistence._add_task("local task")
    other_task, _chat2 = persistence._add_task("other process task")
    persistence._journal_failed_events(
        [_journalled_event(local_task, "from this process")], 4,
    )
    sidecar = Path(persistence._failed_events_path(str(persistence._DB_PATH)))
    assert sidecar.exists()

    started = kiss_home / "child-started"
    appended = kiss_home / "child-appended"

    # Hold the database's write lock so the replay blocks *after* it has
    # taken its snapshot of the sidecar — the exact window in which
    # another process appends.
    holder = sqlite3.connect(str(persistence._DB_PATH), timeout=30)
    holder.execute("PRAGMA busy_timeout=30000")
    holder.execute("BEGIN IMMEDIATE")
    holder.execute(
        "INSERT INTO events (task_id, seq, event_json, timestamp) "
        "VALUES (?, ?, ?, ?)",
        (local_task, 900, json.dumps({"type": "text", "text": "lock"}), 1.0),
    )

    replay_thread = threading.Thread(
        target=persistence._replay_failed_events,
        name="journal-replay",
        daemon=True,
    )
    replay_thread.start()

    child = subprocess.Popen(
        [
            sys.executable, "-c", _CHILD_SCRIPT,
            str(started), str(appended), other_task,
        ],
        env={**os.environ, "KISS_HOME": str(kiss_home)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_for(started, timeout=60), "the second process never ran"
        _wait_for(appended, timeout=_APPEND_GRACE_S)
    finally:
        holder.rollback()
        holder.close()
        out, err = child.communicate(timeout=120)
        assert child.returncode == 0, f"child failed: {out}\n{err}"
        replay_thread.join(timeout=120)
        assert not replay_thread.is_alive()

    # Whatever the interleaving was, nothing may be lost: drain the
    # journal until it is empty and both batches must be in the events
    # table.
    for _attempt in range(5):
        persistence._replay_failed_events()

    assert _event_texts(local_task) == ["from this process"], (
        "this process's batch was replayed more than once: two "
        "processes consumed the same journal snapshot"
    )
    assert _event_texts(other_task) == ["from the second process"], (
        "the replay deleted the batch the other process journalled "
        "while it was running"
    )
    assert _journal_residue(kiss_home) == []


def test_replay_recovers_a_snapshot_left_by_a_crashed_replayer(
    kiss_home: Path,
) -> None:
    """A consumed-but-unwritten snapshot is picked up by the next replay."""
    task_id, _chat = persistence._add_task("crashed replayer task")
    persistence._journal_failed_events(
        [_journalled_event(task_id, "stranded by a crash")], 4,
    )
    sidecar = Path(persistence._failed_events_path(str(persistence._DB_PATH)))
    # Exactly what a replayer that died mid-write leaves behind.
    sidecar.rename(sidecar.with_name(sidecar.name + ".consumed-4242-abcd"))
    assert not sidecar.exists()

    persistence._replay_failed_events()

    assert _event_texts(task_id) == ["stranded by a crash"]
    assert _journal_residue(kiss_home) == []


def test_nothing_is_lost_while_the_database_keeps_refusing_writes(
    kiss_home: Path,
) -> None:
    """Pending rows stay under the well-known journal name, then land."""
    live_task, _chat = persistence._add_task("live journal task")
    stale_task, _chat2 = persistence._add_task("stale snapshot task")
    persistence._journal_failed_events(
        [_journalled_event(live_task, "waiting in the sidecar")], 4,
    )
    sidecar = Path(persistence._failed_events_path(str(persistence._DB_PATH)))
    # A snapshot a replayer died holding, alongside the live sidecar.
    stale = sidecar.with_name(sidecar.name + ".consumed-0000-stale")
    stale.write_text(
        json.dumps({
            "task_id": stale_task,
            "event_json": json.dumps(
                {"type": "text", "text": "waiting in a snapshot"},
            ),
            "timestamp": 99.0,
            "origin_db_path": str(persistence._DB_PATH),
        }) + "\n",
        encoding="utf-8",
    )

    db = persistence._get_db()
    db.execute(
        "CREATE TRIGGER reject_events BEFORE INSERT ON events "
        "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
    )
    try:
        persistence._replay_failed_events()
        assert sidecar.is_file(), (
            "a failed replay hid the pending rows under a snapshot name; "
            "the journal must stay where operators and the next replay "
            "look for it"
        )
        assert len(_journal_residue(kiss_home)) == 2, _journal_residue(kiss_home)
    finally:
        db.execute("DROP TRIGGER reject_events")

    for _attempt in range(3):
        persistence._replay_failed_events()

    assert _event_texts(live_task) == ["waiting in the sidecar"]
    assert _event_texts(stale_task) == ["waiting in a snapshot"]
    assert _journal_residue(kiss_home) == []


def test_replay_without_a_journal_is_a_no_op(kiss_home: Path) -> None:
    """The common case touches nothing and raises nothing."""
    task_id, _chat = persistence._add_task("no journal task")
    persistence._replay_failed_events()
    assert _event_texts(task_id) == []
