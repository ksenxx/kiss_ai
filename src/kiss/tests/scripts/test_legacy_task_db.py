# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``src/kiss/scripts/legacy_task_db.py``: which file holds the history, and the rename.

Run as the deploy scripts run it -- a stdlib-only script fed to
``python3 -`` with the ``~/.kiss`` directory as its argument -- against
every arrangement of ``history.db`` and ``sorcar.db`` an upgrade can meet.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from kiss.core.file_lock import exclusive_file_lock, lock_exclusive, unlock
from kiss.scripts import legacy_task_db
from kiss.tests.conftest import posix_only
from kiss.tests.core.test_legacy_file_names import _PauseAfterEveryFileOperation

_SCRIPT = Path(legacy_task_db.__file__)


def run_script(kiss_dir: Path | str, *args: str) -> subprocess.CompletedProcess[str]:
    """Pipe the script into ``python3 -`` the way ``rsorcar`` does over ssh; it must succeed."""
    with open(_SCRIPT, encoding="utf-8") as source:
        result = subprocess.run(
            [sys.executable, "-", str(kiss_dir), *args],
            stdin=source, capture_output=True, text=True, timeout=60,
        )
    assert result.returncode == 0, result.stderr
    return result


def start_script(kiss_dir: Path, *args: str) -> subprocess.Popen[str]:
    """Start the script fed the way :func:`run_script` feeds it, and leave it running."""
    with open(_SCRIPT, encoding="utf-8") as source:
        return subprocess.Popen(
            [sys.executable, "-", str(kiss_dir), *args],
            stdin=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )


def outcome(script: subprocess.Popen[str]) -> tuple[int, str, str]:
    """Wait (60 s at most) for *script* to end; return its exit status and what it printed."""
    out, err = script.communicate(timeout=60)
    return script.returncode, out, err


@contextmanager
def lookup_and_rename_waiting(kiss_dir: Path) -> Iterator[list[subprocess.Popen[str]]]:
    """Start the lookup and the rename while the caller holds the rename lock.

    Yields the two processes once both have waited a second; kills what is left of them.
    """
    scripts = [start_script(kiss_dir), start_script(kiss_dir, "adopt")]
    try:
        time.sleep(1.0)
        assert [script.poll() for script in scripts] == [None, None]  # both wait for the lock
        yield scripts
    finally:
        for script in scripts:
            script.kill()
            script.communicate(timeout=30)


def write_db(path: Path, id_type: str) -> None:
    """Create a task database whose ``task_history.id`` has *id_type*."""
    con = sqlite3.connect(path)
    con.execute(f"CREATE TABLE task_history (id {id_type} PRIMARY KEY, task TEXT)")
    row_id = 1 if id_type == "INTEGER" else "a"
    con.execute("INSERT INTO task_history (id, task) VALUES (?, 'row')", (row_id,))
    con.commit()
    con.close()


def test_no_database_at_all(tmp_path: Path) -> None:
    assert run_script(tmp_path).stdout.strip() == "history.db"
    assert run_script(tmp_path, "adopt").stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_only_sorcar_db_is_adopted(tmp_path: Path) -> None:
    write_db(tmp_path / "sorcar.db", "TEXT")
    (tmp_path / "sorcar.db-shm").write_bytes(b"shm")
    assert run_script(tmp_path).stdout.strip() == "sorcar.db"
    assert run_script(tmp_path, "adopt").stdout == "sorcar.db renamed to history.db\n"
    assert (tmp_path / "history.db-shm").read_bytes() == b"shm"
    assert run_script(tmp_path).stdout.strip() == "history.db"


def test_a_current_history_db_keeps_its_place(tmp_path: Path) -> None:
    """history.db with the current schema beside an unrelated sorcar.db: nothing moves."""
    write_db(tmp_path / "history.db", "TEXT")
    write_db(tmp_path / "sorcar.db", "INTEGER")
    assert run_script(tmp_path).stdout.strip() == "history.db"
    assert run_script(tmp_path, "adopt").stdout == ""
    assert not (tmp_path / "sorcar.db").is_symlink()


def test_a_history_db_from_spring_2026_is_set_aside(tmp_path: Path) -> None:
    write_db(tmp_path / "history.db", "INTEGER")
    (tmp_path / "history.db-wal").write_bytes(b"old wal")
    write_db(tmp_path / "sorcar.db", "TEXT")
    assert run_script(tmp_path).stdout.strip() == "sorcar.db"
    out = run_script(tmp_path, "adopt").stdout.splitlines()
    assert out[0].startswith(
        "history.db is a leftover from before 2026-04-24; kept as history.db.stale-"
    )
    assert out[1] == "sorcar.db renamed to history.db"
    kept = tmp_path / out[0].rsplit(" ", 1)[1]
    assert kept.with_name(kept.name + "-wal").read_bytes() == b"old wal"
    con = sqlite3.connect(tmp_path / "history.db")
    assert con.execute("SELECT id FROM task_history").fetchall() == [("a",)]
    con.close()
    assert not (tmp_path / "history.db-wal").exists()


def test_a_relative_directory_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``KISS_HOME=.kiss`` reaches the script as a relative path; the schema probe must cope."""
    write_db(tmp_path / "history.db", "INTEGER")
    write_db(tmp_path / "sorcar.db", "TEXT")
    monkeypatch.chdir(tmp_path.parent)
    assert run_script(tmp_path.name).stdout.strip() == "sorcar.db"
    assert run_script(tmp_path.name, "adopt").stdout.endswith("sorcar.db renamed to history.db\n")


@posix_only("the lock is the web app's flock, and the link exists only where SQLite resolves it")
def test_the_rename_waits_for_a_web_app_adopting_at_the_same_time(tmp_path: Path) -> None:
    """A web app holding the rename lock finishes first; the script then finds nothing to do.

    Without the lock the script, having judged history.db stale a moment
    earlier, would set aside the database the web app has just adopted.
    """
    write_db(tmp_path / "history.db", "INTEGER")
    write_db(tmp_path / "sorcar.db", "TEXT")
    with exclusive_file_lock(tmp_path / "history.db.rename.lock"):
        script = start_script(tmp_path, "adopt")
        with pytest.raises(subprocess.TimeoutExpired):
            script.wait(timeout=1.0)
        # What the web app's adoption does under the lock it holds.
        os.replace(tmp_path / "history.db", tmp_path / "history.db.stale-by-the-web-app")
        os.replace(tmp_path / "sorcar.db", tmp_path / "history.db")
        os.symlink("history.db", tmp_path / "sorcar.db")
    assert outcome(script) == (0, "", "")
    stale = [p for p in tmp_path.iterdir() if ".stale-" in p.name]
    assert len(stale) == 1, stale  # the web app's set-aside copy, and no second one
    con = sqlite3.connect(tmp_path / "history.db")
    assert con.execute("SELECT id FROM task_history").fetchall() == [("a",)]
    con.close()


def test_a_rename_lock_that_is_never_released_ends_the_wait(tmp_path: Path) -> None:
    """A holder that never lets go costs the script its timeout; ``flock`` waited for good."""
    with exclusive_file_lock(tmp_path / "history.db.rename.lock"):  # the web app, wedged
        with pytest.raises(TimeoutError, match="history.db.rename.lock still locked"):
            with legacy_task_db._rename_lock(tmp_path, timeout=0.2):
                pytest.fail("the lock was given to two holders")


def test_a_database_renamed_during_the_wait_is_found_under_its_new_name(tmp_path: Path) -> None:
    """Both forms saw ``sorcar.db`` and waited for the web app; they find that name gone.

    That is what an adoption leaves on Windows (plain renames, no link), done by hand
    here under the web app's lock: the lookup answers ``history.db``, the rename does
    nothing.  The lookup took no lock and answered ``sorcar.db`` on the spot.
    """
    write_db(tmp_path / "sorcar.db", "TEXT")
    with open(tmp_path / "history.db.rename.lock", "w") as web_app_lock:
        lock_exclusive(web_app_lock)
        with lookup_and_rename_waiting(tmp_path) as scripts:
            os.replace(tmp_path / "sorcar.db", tmp_path / "history.db")
            unlock(web_app_lock)
            assert [outcome(script) for script in scripts] == [(0, "history.db\n", ""), (0, "", "")]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["history.db", "history.db.rename.lock"]


def _adopt_pausing(kiss_dir: Path, hook: _PauseAfterEveryFileOperation, done: list[object]) -> None:
    """Adopt on this thread with *hook* installed; leave the result or the error in *done*."""
    sys.setprofile(hook)
    try:
        done.append(legacy_task_db.adopt_legacy_db(kiss_dir))
    except BaseException as e:  # noqa: BLE001 - reported by the test
        done.append(e)
    finally:
        sys.setprofile(None)
        hook.paused.set()


def _resume_to_the_end(adopter: threading.Thread, hook: _PauseAfterEveryFileOperation) -> None:
    """Give the paused *adopter* the turn back until it is done (30 s at most)."""
    deadline = time.monotonic() + 30
    while adopter.is_alive() and time.monotonic() < deadline:
        hook.resume.set()
        adopter.join(0.05)


def test_a_lookup_and_a_second_rename_wait_for_the_rename_under_way(tmp_path: Path) -> None:
    """Whoever arrives while a rename is half done waits for it, then sees its result.

    The rename is paused right after it moved the leftover ``history.db`` aside, when
    no file has that name.  The lookup took no lock: it answered from that half-done
    state, and one that had seen the leftover a moment earlier died on it
    (``FileNotFoundError``).  On Windows the script took no lock at all, so a second
    rename ran beside the first and set the database the first had just renamed aside
    as the leftover.  (Windows locks through ``msvcrt``, which exists only there, so
    that branch cannot run here without a test double; the wait around it is the one
    the ``flock`` of this run goes through.)
    """
    write_db(tmp_path / "history.db", "INTEGER")
    write_db(tmp_path / "sorcar.db", "TEXT")
    hook = _PauseAfterEveryFileOperation()
    done: list[object] = []
    adopter = threading.Thread(target=_adopt_pausing, args=(tmp_path, hook, done), daemon=True)
    adopter.start()
    try:
        assert hook.paused.wait(30) and not done, done
        assert not (tmp_path / "history.db").exists()  # the leftover has just been moved aside
        with lookup_and_rename_waiting(tmp_path) as scripts:
            _resume_to_the_end(adopter, hook)
            assert [outcome(script) for script in scripts] == [(0, "history.db\n", ""), (0, "", "")]
    finally:
        _resume_to_the_end(adopter, hook)
    assert isinstance(done[0], list) and done[0][1:] == ["sorcar.db renamed to history.db"], done
    assert len(list(tmp_path.glob("history.db.stale-*"))) == 1  # set aside once, by the first
    con = sqlite3.connect(tmp_path / "history.db")
    assert con.execute("SELECT id FROM task_history").fetchall() == [("a",)]
    con.close()


@posix_only("an old-version process keeps the old name only where it becomes a link")
def test_an_old_version_writer_loses_nothing_at_any_step_of_the_rename(tmp_path: Path) -> None:
    """An old-version web app opens ``sorcar.db`` per operation, at any moment of the rename.

    After every file operation of the adoption its last connection closes (a WAL
    checkpoint that deletes the sidecars unless something holds them) and a new
    one commits a row through the old name.  A plain rename left no file under
    that name for a moment: the writer created an empty one, the symlink could
    not be made, and the row was stranded in a second database.  (The Windows
    branch, a plain rename with no link, cannot run here.)
    """
    old = tmp_path / "sorcar.db"
    schema = "CREATE TABLE IF NOT EXISTS task_history (id TEXT PRIMARY KEY, task TEXT)"
    writer = sqlite3.connect(old)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute(schema)
    committed: list[str] = []
    hook = _PauseAfterEveryFileOperation()
    done: list[object] = []
    adopter = threading.Thread(target=_adopt_pausing, args=(tmp_path, hook, done), daemon=True)
    adopter.start()
    while hook.paused.wait(30) and not done:
        hook.paused.clear()
        writer.close()
        writer = sqlite3.connect(old)
        writer.execute(schema)  # what the web app does on every open
        committed.append(f"{len(committed)} after {hook.operations[-1]}")
        writer.execute("INSERT INTO task_history VALUES (?, 'row')", (committed[-1],))
        writer.commit()
        hook.resume.set()
    adopter.join(30)
    writer.close()
    assert done == [["sorcar.db renamed to history.db"]] and len(committed) > 3, (done, committed)
    assert os.readlink(old) == "history.db"
    for name in ("history.db", "sorcar.db"):
        reader = sqlite3.connect(tmp_path / name)
        assert reader.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        rows = reader.execute("SELECT id FROM task_history ORDER BY rowid").fetchall()
        assert [row[0] for row in rows] == committed, name
        reader.close()
    assert not list(tmp_path.glob("sorcar.db-*")) and not list(tmp_path.glob("*.link"))


@posix_only("two hard links are what an adoption that died half-way leaves behind")
def test_a_rename_that_stopped_before_the_link_is_finished(tmp_path: Path) -> None:
    """Both names hard links of one file: the old one has yet to become the symlink."""
    write_db(tmp_path / "sorcar.db", "TEXT")
    os.link(tmp_path / "sorcar.db", tmp_path / "history.db")
    assert run_script(tmp_path).stdout.strip() == "sorcar.db"
    assert run_script(tmp_path, "adopt").stdout == "sorcar.db renamed to history.db\n"
    assert os.readlink(tmp_path / "sorcar.db") == "history.db"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "history.db", "history.db.rename.lock", "sorcar.db",
    ]


@posix_only("a WAL is kept in place only where the old name becomes a link")
def test_a_sorcar_db_that_sqlite_cannot_read_keeps_its_name(tmp_path: Path) -> None:
    """No connection can keep its WAL in place, so nothing is renamed; no database, no WAL."""
    old = tmp_path / "sorcar.db"
    old.write_bytes(b"SQLite format 3\x00" + b"\xff" * 100)
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(tmp_path), "adopt"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1 and "sqlite3." in result.stderr, result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["history.db.rename.lock", "sorcar.db"]
    old.write_bytes(b"not sqlite")
    assert run_script(tmp_path, "adopt").stdout == "sorcar.db renamed to history.db\n"
    assert (tmp_path / "history.db").read_bytes() == b"not sqlite" and old.is_symlink()


@posix_only("a file mode keeps its owner out only on POSIX")
def test_a_sorcar_db_this_user_cannot_read_is_renamed_unkept(tmp_path: Path) -> None:
    """No header to look at and no WAL this user could keep: renamed, as the web app does."""
    old = tmp_path / "sorcar.db"
    write_db(old, "TEXT")
    old.chmod(0)
    assert run_script(tmp_path, "adopt").stdout == "sorcar.db renamed to history.db\n"
    assert old.is_symlink() and (tmp_path / "history.db").stat().st_mode & 0o777 == 0


@posix_only("the link the rename leaves behind exists only where SQLite resolves it")
def test_the_link_left_behind_is_not_a_second_database(tmp_path: Path) -> None:
    write_db(tmp_path / "history.db", "INTEGER")  # the deploy scripts' simplified schema
    os.symlink("history.db", tmp_path / "sorcar.db")
    assert run_script(tmp_path).stdout.strip() == "history.db"
    assert run_script(tmp_path, "adopt").stdout == ""


def test_a_file_that_is_not_a_database_is_left_alone(tmp_path: Path) -> None:
    (tmp_path / "history.db").write_bytes(b"not sqlite")
    write_db(tmp_path / "sorcar.db", "TEXT")
    assert legacy_task_db.is_pre_2026_04_db(tmp_path / "history.db") is False
    assert run_script(tmp_path).stdout.strip() == "history.db"


@pytest.mark.parametrize("args", [(), ("a", "b", "c"), ("x", "drop")])
def test_usage_errors(tmp_path: Path, args: tuple[str, ...]) -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "usage: legacy_task_db.py KISS_DIR [adopt]"
