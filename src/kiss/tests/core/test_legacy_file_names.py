# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An install upgraded across the 2026.10.2 renames keeps its data.

``~/.kiss/sorcar.db`` became ``history.db`` and ``~/.kiss/SORCAR.md``
became ``AGENTS.md``.  :func:`kiss.core.config.adopt_legacy_file` renames
a file still under its old name the first time the new name is looked
up: the persistence layer does it before opening the database, and
:func:`kiss.core.config.agents_md_path` does it for the instructions.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.core import config
from kiss.core.file_lock import exclusive_file_lock


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


def test_sorcar_db_is_renamed_to_history_db_with_its_sidecars(kiss_dir: Path) -> None:
    """A database written as sorcar.db is read back as history.db, rows intact."""
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._add_task("written under the old name")
    assert (kiss_dir / "sorcar.db").exists()

    th._DB_PATH = kiss_dir / "history.db"
    rows = th._get_db().execute("SELECT task FROM task_history").fetchall()
    assert [row["task"] for row in rows] == ["written under the old name"]
    assert (kiss_dir / "history.db").is_file()
    assert os.readlink(kiss_dir / "sorcar.db") == "history.db"
    assert not (kiss_dir / "sorcar.db-wal").exists()
    assert not (kiss_dir / "sorcar.db-shm").exists()


def test_a_process_of_the_old_version_keeps_writing_to_the_same_file(kiss_dir: Path) -> None:
    """Opened through the old name after the rename, SQLite shares the new file's WAL."""
    th._DB_PATH = kiss_dir / "history.db"
    th._add_task("new version")
    config.adopt_legacy_file(kiss_dir / "history.db", "sorcar.db")  # no-op: nothing legacy
    (kiss_dir / "sorcar.db").symlink_to("history.db")  # what the rename leaves behind

    old_version = sqlite3.connect(kiss_dir / "sorcar.db")
    old_version.execute(
        "INSERT INTO task_history (timestamp, task, chat_id) VALUES (1.0, 'old version', 'c')",
    )
    old_version.commit()
    assert (kiss_dir / "history.db-wal").exists()
    assert not (kiss_dir / "sorcar.db-wal").exists()
    old_version.close()
    rows = th._get_db().execute("SELECT task FROM task_history").fetchall()
    assert sorted(row["task"] for row in rows) == ["new version", "old version"]


def test_events_journalled_against_the_old_path_are_replayed(kiss_dir: Path) -> None:
    """The journals move with the database and their rows still name the active database."""
    th._DB_PATH = kiss_dir / "sorcar.db"
    task_id, _chat = th._add_task("journalled under the old name")
    th._journal_failed_events(
        [(task_id, json.dumps({"type": "result", "text": "kept"}), 1.0, str(th._DB_PATH))], 3,
    )
    th._journal_final_result(task_id, "<p>kept</p>")
    th._close_db()

    th._DB_PATH = kiss_dir / "history.db"
    th._get_db()  # the first open renames the database and its journals
    th._replay_failed_events()
    assert (kiss_dir / "history.db.final_results.jsonl").is_file()
    assert not (kiss_dir / "sorcar.db.failed_events.jsonl").exists()
    events = th._get_db().execute(
        "SELECT event_json FROM events WHERE task_id = ?", (task_id,),
    ).fetchall()
    assert [json.loads(row["event_json"])["text"] for row in events] == ["kept"]
    assert th._load_final_results(th._final_results_path(str(th._DB_PATH))) == {
        task_id: "<p>kept</p>",
    }


def test_claimed_journal_snapshots_follow_the_database_and_keep_their_markers(
    kiss_dir: Path,
) -> None:
    """A replay interrupted before the upgrade resumes; one already committed is not repeated."""
    th._DB_PATH = kiss_dir / "sorcar.db"
    task_id, _chat = th._add_task("interrupted replay")
    claimed = f"sorcar.db.failed_events.jsonl.consumed-{1:020d}-{os.getpid()}-"
    pending, committed = claimed + "a" * 32, claimed + "b" * 32
    th._get_db().execute(
        "INSERT INTO replayed_journals (snapshot, timestamp) VALUES (?, 2.0)", (committed,),
    )
    th._get_db().commit()
    th._close_db()  # the replayer died after committing one snapshot and claiming the other
    for name, text in ((pending, "pending"), (committed, "twice")):
        (kiss_dir / name).write_text(json.dumps({
            "task_id": task_id, "event_json": json.dumps({"type": "result", "text": text}),
            "timestamp": 2.0, "origin_db_path": str(th._DB_PATH),
        }) + "\n")

    th._DB_PATH = kiss_dir / "history.db"
    th._get_db()
    assert not list(kiss_dir.glob("sorcar.db.failed_events.jsonl.consumed-*"))
    assert len(list(kiss_dir.glob("history.db.failed_events.jsonl.consumed-*"))) == 2
    th._replay_failed_events()
    events = th._get_db().execute(
        "SELECT event_json FROM events WHERE task_id = ?", (task_id,),
    ).fetchall()
    assert [json.loads(row["event_json"])["text"] for row in events] == ["pending"]
    assert not list(kiss_dir.glob("*.failed_events.jsonl.consumed-*"))


def test_sidecars_move_before_the_main_file_and_keep_their_bytes(tmp_path: Path) -> None:
    """Every part of the trio is renamed; a missing sidecar is simply skipped."""
    (tmp_path / "old.db").write_bytes(b"main")
    (tmp_path / "old.db-wal").write_bytes(b"wal pages")
    config.adopt_legacy_file(tmp_path / "new.db", "old.db", ("-wal", "-shm", ""))
    assert (tmp_path / "new.db").read_bytes() == b"main"
    assert (tmp_path / "new.db-wal").read_bytes() == b"wal pages"
    assert not (tmp_path / "new.db-shm").exists()
    assert os.readlink(tmp_path / "old.db") == "new.db"
    assert not (tmp_path / "old.db-wal").exists()


def test_nothing_happens_without_a_legacy_file_or_with_both(tmp_path: Path) -> None:
    """No legacy file: no-op.  New name already present: the old file is left alone."""
    config.adopt_legacy_file(tmp_path / "new.db", "old.db")
    assert list(tmp_path.iterdir()) == []

    (tmp_path / "old.db").write_bytes(b"old")
    (tmp_path / "new.db").write_bytes(b"new")
    config.adopt_legacy_file(tmp_path / "new.db", "old.db")
    assert (tmp_path / "new.db").read_bytes() == b"new"
    assert (tmp_path / "old.db").read_bytes() == b"old"


def test_a_concurrent_process_that_renamed_first_wins(tmp_path: Path) -> None:
    """The second arrival finds the rename done under the lock and returns."""
    (tmp_path / "old.md").write_text("rule")
    lock_path = tmp_path / "new.md.rename.lock"
    lock_taken = threading.Event()
    release = threading.Event()

    def other_process() -> None:
        with exclusive_file_lock(lock_path):
            lock_taken.set()
            release.wait(timeout=10)
            (tmp_path / "old.md").rename(tmp_path / "new.md")

    other = threading.Thread(target=other_process)
    other.start()
    assert lock_taken.wait(timeout=10)
    adopter = threading.Thread(
        target=config.adopt_legacy_file, args=(tmp_path / "new.md", "old.md"),
    )
    adopter.start()
    time.sleep(0.3)  # the adopter has seen the old file and is waiting for the lock
    release.set()
    other.join(timeout=10)
    adopter.join(timeout=10)
    assert not adopter.is_alive()
    assert (tmp_path / "new.md").read_text() == "rule"
    # The other adopter left no link, and this one found nothing to do.
    assert not (tmp_path / "old.md").exists()


def test_sorcar_md_becomes_agents_md(kiss_dir: Path) -> None:
    """The standing instructions written to SORCAR.md are served from AGENTS.md."""
    (kiss_dir / "SORCAR.md").write_text("# User instructions\n\n- Always be brief\n")
    path = config.agents_md_path()
    assert path == kiss_dir / "AGENTS.md"
    assert path.read_text() == "# User instructions\n\n- Always be brief\n"
    assert os.readlink(kiss_dir / "SORCAR.md") == "AGENTS.md"
    assert config.agents_md_path() == path
