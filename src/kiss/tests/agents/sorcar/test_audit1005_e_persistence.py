# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-05 scope E: persistence races.

``_adopt_legacy_journal_snapshots`` renames a ``sorcar.db.*.consumed-*``
snapshot after ``history.db`` and re-keys its exactly-once marker in
``replayed_journals``.  Until this audit it renamed FIRST and re-keyed
SECOND: a replay (:func:`_replay_failed_events`) that claimed the renamed
file in between found no marker under the new name and inserted the
already-committed rows a second time.  The marker is now copied to the
new name before the rename, so a replayer that can see the file can see
its marker.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th

_SNAPSHOT_TAIL = f".failed_events.jsonl.consumed-{1:020d}-{os.getpid()}-" + "b" * 32


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


def _committed_legacy_snapshot(kiss_dir: Path) -> str:
    """Create a sorcar.db database with one task and one committed-but-not-unlinked snapshot.

    Returns the task id.  The snapshot's rows are already in
    ``replayed_journals`` (the replayer died between COMMIT and unlink),
    so a correct replay must never insert them.
    """
    th._DB_PATH = kiss_dir / "sorcar.db"
    task_id, _chat = th._add_task("interrupted replay")
    committed = "sorcar.db" + _SNAPSHOT_TAIL
    th._get_db().execute(
        "INSERT INTO replayed_journals (snapshot, timestamp) VALUES (?, 2.0)", (committed,),
    )
    th._close_db()
    (kiss_dir / committed).write_text(json.dumps({
        "task_id": task_id, "event_json": json.dumps({"type": "result", "text": "twice"}),
        "timestamp": 2.0, "origin_db_path": str(th._DB_PATH),
    }) + "\n")
    return task_id


def test_a_replay_racing_the_adoption_never_duplicates_committed_rows(
    kiss_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replayer claims the renamed snapshot the instant it appears, and skips it."""
    monkeypatch.setenv("KISS_RACE_DELAY", "0.1")  # widens the window after the rename
    task_id = _committed_legacy_snapshot(kiss_dir)
    th._DB_PATH = kiss_dir / "history.db"

    adopter = threading.Thread(target=th._get_db)  # a first connection adopts
    adopter.start()
    renamed = kiss_dir / ("history.db" + _SNAPSHOT_TAIL)
    deadline = time.monotonic() + 10
    while not renamed.exists():
        assert time.monotonic() < deadline, "snapshot was never adopted"
        time.sleep(0.001)
    # The invariant the replayer relies on, sampled the instant the renamed
    # file is visible: its marker is visible too (the old order published the
    # file first and re-keyed the marker afterwards).
    with closing(sqlite3.connect(str(th._DB_PATH), timeout=10)) as raw:
        visible = raw.execute(
            "SELECT 1 FROM replayed_journals WHERE snapshot = ?", (renamed.name,),
        ).fetchone()
    assert visible is not None, "renamed snapshot is visible before its marker"
    th._replay_failed_events()  # runs inside the adopter's window
    adopter.join(timeout=10)
    assert not adopter.is_alive()

    events = th._get_db().execute(
        "SELECT event_json FROM events WHERE task_id = ?", (task_id,),
    ).fetchall()
    assert events == []
    assert not list(kiss_dir.glob("*.failed_events.jsonl.consumed-*"))
    markers = th._get_db().execute("SELECT snapshot FROM replayed_journals").fetchall()
    assert [row["snapshot"] for row in markers] == []


def test_adoption_then_replay_skips_committed_rows_and_leaves_no_marker(
    kiss_dir: Path,
) -> None:
    """Sequential adoption + replay: the marker follows the file and is pruned with it."""
    task_id = _committed_legacy_snapshot(kiss_dir)
    th._DB_PATH = kiss_dir / "history.db"
    th._get_db()
    markers = th._get_db().execute("SELECT snapshot FROM replayed_journals").fetchall()
    assert [row["snapshot"] for row in markers] == ["history.db" + _SNAPSHOT_TAIL]
    th._replay_failed_events()
    events = th._get_db().execute(
        "SELECT event_json FROM events WHERE task_id = ?", (task_id,),
    ).fetchall()
    assert events == []
    assert not list(kiss_dir.glob("*.failed_events.jsonl.consumed-*"))
    assert th._get_db().execute("SELECT snapshot FROM replayed_journals").fetchall() == []
