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
from kiss.tests.conftest import posix_only


def assert_old_name_left_behind(old: Path, new_name: str) -> None:
    """The old name is a link to the new one, except on Windows, where nothing is left.

    SQLite on Windows names the WAL after the path it was given, so a link
    would have an old-version process write a second WAL over the database.
    """
    if os.name == "nt":
        assert not old.is_symlink() and not old.exists()
    else:
        assert os.readlink(old) == new_name


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
    assert_old_name_left_behind(kiss_dir / "sorcar.db", "history.db")
    assert not (kiss_dir / "sorcar.db-wal").exists()
    assert not (kiss_dir / "sorcar.db-shm").exists()


@posix_only("the rename leaves a link behind only where SQLite resolves it (not Windows)")
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


def test_rows_journalled_against_an_unrelated_sorcar_db_are_dropped(kiss_dir: Path) -> None:
    """A sorcar.db still beside history.db is another database, not its former name."""
    th._DB_PATH = kiss_dir / "history.db"
    task_id, _chat = th._add_task("new version")
    th._close_db()
    other = sqlite3.connect(kiss_dir / "sorcar.db")  # e.g. created by an old version run later
    other.execute("CREATE TABLE t (x)")
    other.commit()
    other.close()
    (kiss_dir / "sorcar.db.failed_events.jsonl").write_text(json.dumps({
        "task_id": task_id, "event_json": json.dumps({"type": "result", "text": "stray"}),
        "timestamp": 1.0, "origin_db_path": str(kiss_dir / "sorcar.db"),
    }) + "\n")

    th._get_db()  # adopts the journal; the database itself stays, history.db exists
    th._replay_failed_events()
    assert (kiss_dir / "sorcar.db").is_file() and not (kiss_dir / "sorcar.db").is_symlink()
    assert not (kiss_dir / "history.db.failed_events.jsonl").exists()
    events = th._get_db().execute(
        "SELECT event_json FROM events WHERE task_id = ?", (task_id,),
    ).fetchall()
    assert events == []


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
    # The rows name the old path on their own merits, not through the link
    # the rename leaves behind (there is none on Windows, and a user may
    # have deleted it).
    (kiss_dir / "sorcar.db").unlink(missing_ok=True)
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
    assert_old_name_left_behind(tmp_path / "old.db", "new.db")
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


#: ``task_history`` as the versions of March/April 2026 created it, when
#: the database was already called ``history.db``: integer ids, no
#: ``extra`` column yet (that came 2026-04-13), none of today's columns.
_MARCH_2026_SCHEMA = """
    CREATE TABLE task_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        task TEXT NOT NULL,
        has_events INTEGER DEFAULT 0,
        result TEXT DEFAULT '',
        chat_id TEXT DEFAULT ''
    );
    CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER NOT NULL REFERENCES task_history(id),
        seq INTEGER NOT NULL,
        event_json TEXT NOT NULL,
        timestamp REAL NOT NULL
    );
    INSERT INTO task_history (timestamp, task, has_events, chat_id)
        VALUES (1.0, 'from march 2026', 1, 'c1');
    INSERT INTO events (task_id, seq, event_json, timestamp)
        VALUES (1, 0, '{"type": "start"}', 1.0);
"""


def write_march_2026_history_db(path: Path) -> None:
    """Leave at *path* the ``history.db`` an install of March 2026 wrote."""
    conn = sqlite3.connect(path)
    conn.executescript(_MARCH_2026_SCHEMA)
    conn.commit()
    conn.close()


def test_a_history_db_abandoned_in_april_2026_does_not_hide_sorcar_db(kiss_dir: Path) -> None:
    """The real database is adopted even when its new name is taken by a pre-rename leftover.

    The database was ``history.db`` until 2026-04-24, when it became
    ``sorcar.db`` and the old file was left in place.  An install that
    old has both names when it upgrades to 2026.10.2: the stale
    ``history.db`` must not be taken for the current database (its
    schema cannot even be opened: ``no such column: parent_task_id``),
    so it is set aside and ``sorcar.db`` is renamed as usual.
    """
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._add_task("the real history")
    (kiss_dir / "sorcar.db.final_results.jsonl").write_text("")
    th._close_db()
    write_march_2026_history_db(kiss_dir / "history.db")
    (kiss_dir / "history.db-wal").write_bytes(b"stale wal")

    th._DB_PATH = kiss_dir / "history.db"
    rows = th._get_db().execute("SELECT task FROM task_history").fetchall()
    assert [row["task"] for row in rows] == ["the real history"]
    assert_old_name_left_behind(kiss_dir / "sorcar.db", "history.db")
    assert_old_name_left_behind(
        kiss_dir / "sorcar.db.final_results.jsonl", "history.db.final_results.jsonl",
    )
    assert not (kiss_dir / "sorcar.db-wal").exists()
    # The leftover is kept, complete with its sidecar, under a name that says what it is.
    [stale_db] = [
        p for p in kiss_dir.iterdir()
        if p.name.startswith("history.db.stale-") and not p.name.endswith(("-wal", "-shm"))
    ]
    assert stale_db.with_name(stale_db.name + "-wal").read_bytes() == b"stale wal"
    kept = sqlite3.connect(stale_db).execute("SELECT task FROM task_history").fetchall()
    assert kept == [("from march 2026",)]


def test_a_current_history_db_is_never_set_aside(kiss_dir: Path) -> None:
    """Both names present and history.db current: sorcar.db is an unrelated file, left alone."""
    th._DB_PATH = kiss_dir / "history.db"
    th._add_task("current")
    th._close_db()
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._add_task("written by a downgrade")
    th._close_db()

    th._DB_PATH = kiss_dir / "history.db"
    rows = th._get_db().execute("SELECT task FROM task_history").fetchall()
    assert [row["task"] for row in rows] == ["current"]
    assert not (kiss_dir / "sorcar.db").is_symlink()
    assert not [p for p in kiss_dir.iterdir() if ".stale-" in p.name]


def test_sorcar_db_linked_to_history_db_is_not_a_second_database(kiss_dir: Path) -> None:
    """After an adoption the old name is a link; the next open must not inspect it as a rival."""
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._add_task("adopted")
    th._DB_PATH = kiss_dir / "history.db"
    assert [r["task"] for r in th._get_db().execute("SELECT task FROM task_history")] == [
        "adopted",
    ]
    th._close_db()
    assert [r["task"] for r in th._get_db().execute("SELECT task FROM task_history")] == [
        "adopted",
    ]
    assert not [p for p in kiss_dir.iterdir() if ".stale-" in p.name]


def test_a_history_db_from_march_2026_on_its_own_is_migrated(kiss_dir: Path) -> None:
    """Without a sorcar.db the leftover IS the history: it is ported to the current schema.

    Its ``task_history`` predates the ``extra`` column, which the
    migration used to require, so opening it failed on the first index
    over a column the table never had.
    """
    th._DB_PATH = kiss_dir / "history.db"
    write_march_2026_history_db(kiss_dir / "history.db")

    conn = th._get_db()
    rows = conn.execute("SELECT id, task, chat_id, parent_task_id FROM task_history").fetchall()
    assert len(rows) == 1
    assert rows[0]["task"] == "from march 2026"
    assert rows[0]["chat_id"] == "c1"
    assert rows[0]["parent_task_id"] == ""
    assert th._TASK_ID_RE.fullmatch(rows[0]["id"])
    events = conn.execute("SELECT task_id, event_json FROM events").fetchall()
    assert [(e["task_id"], e["event_json"]) for e in events] == [
        (rows[0]["id"], '{"type": "start"}'),
    ]
    th._add_task("after the migration")
    assert conn.execute("SELECT COUNT(*) FROM task_history").fetchone()[0] == 2


@posix_only("the lock is a flock and the rename leaves a link only where SQLite resolves it")
def test_whether_history_db_is_stale_is_decided_under_the_lock(tmp_path: Path) -> None:
    """A second adopter waits for the first instead of judging a file that is being moved.

    Judged outside the lock, a history.db that the first adopter is
    setting aside at that instant reads as unreadable, hence "not stale,
    nothing to do" -- and the second adopter would go on to create an
    empty history.db that the first then renames sorcar.db over.
    """
    write_march_2026_history_db(tmp_path / "history.db")
    (tmp_path / "sorcar.db").write_bytes(b"the real database")
    judged: list[Path] = []

    def stale(path: Path) -> bool:
        judged.append(path)
        return th.is_pre_2026_04_db(path)

    lock_taken = threading.Event()
    release = threading.Event()

    def first_adopter() -> None:
        with exclusive_file_lock(tmp_path / "history.db.rename.lock"):
            lock_taken.set()
            release.wait(timeout=10)
            (tmp_path / "history.db").rename(tmp_path / "history.db.stale-first")
            (tmp_path / "sorcar.db").rename(tmp_path / "history.db")
            os.symlink("history.db", tmp_path / "sorcar.db")

    first = threading.Thread(target=first_adopter)
    first.start()
    assert lock_taken.wait(timeout=10)
    second = threading.Thread(
        target=config.adopt_legacy_file,
        args=(tmp_path / "history.db", "sorcar.db", ("-wal", "-shm", "")),
        kwargs={"stale": stale},
    )
    second.start()
    second.join(timeout=0.5)
    assert second.is_alive()  # waiting for the lock ...
    assert judged == []  # ... without having judged the file outside it
    release.set()
    first.join(timeout=10)
    second.join(timeout=10)
    assert not second.is_alive()
    assert judged == []  # under the lock, sorcar.db was already the link: nothing to judge
    assert (tmp_path / "history.db").read_bytes() == b"the real database"
    assert [p.name for p in tmp_path.iterdir() if ".stale-" in p.name] == ["history.db.stale-first"]


def test_sorcar_md_becomes_agents_md(kiss_dir: Path) -> None:
    """The standing instructions written to SORCAR.md are served from AGENTS.md."""
    (kiss_dir / "SORCAR.md").write_text("# User instructions\n\n- Always be brief\n")
    path = config.agents_md_path()
    assert path == kiss_dir / "AGENTS.md"
    assert path.read_text() == "# User instructions\n\n- Always be brief\n"
    assert_old_name_left_behind(kiss_dir / "SORCAR.md", "AGENTS.md")
    assert config.agents_md_path() == path
