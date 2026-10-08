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
from pathlib import Path

import pytest

from kiss.core.file_lock import exclusive_file_lock
from kiss.scripts import legacy_task_db
from kiss.tests.conftest import posix_only

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
        with open(_SCRIPT, encoding="utf-8") as source:
            script = subprocess.Popen(
                [sys.executable, "-", str(tmp_path), "adopt"],
                stdin=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        with pytest.raises(subprocess.TimeoutExpired):
            script.wait(timeout=1.0)
        # What the web app's adoption does under the lock it holds.
        os.replace(tmp_path / "history.db", tmp_path / "history.db.stale-by-the-web-app")
        os.replace(tmp_path / "sorcar.db", tmp_path / "history.db")
        os.symlink("history.db", tmp_path / "sorcar.db")
    out, err = script.communicate(timeout=30)
    assert (script.returncode, out, err) == (0, "", "")
    stale = [p for p in tmp_path.iterdir() if ".stale-" in p.name]
    assert len(stale) == 1, stale  # the web app's set-aside copy, and no second one
    con = sqlite3.connect(tmp_path / "history.db")
    assert con.execute("SELECT id FROM task_history").fetchall() == [("a",)]
    con.close()


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
