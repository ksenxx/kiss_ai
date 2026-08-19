# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the two-way half of ``scripts/sync-task-db.sh``.

A deploy used to push ``~/.kiss/sorcar.db`` one way only, so every task an
agent ran *on the server* stayed there: it showed up in the deployment's
History panel and nowhere else, and the moment the push had to fall back to
replacing the remote database wholesale it was gone for good.

These tests run the real script against a sandbox "remote": a fake ``ssh``
on ``PATH`` executes the remote half locally with ``HOME`` pointed at
another directory.  Unlike the fake ssh of ``test_sync_task_db.py`` this one
understands the options ``sync_db.py`` passes (``-o BatchMode=yes``), so the
delta merge — the path a real deploy takes — is what is exercised here, in
both directions, over databases with the real schema.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import kiss.agents.sorcar.persistence as th

_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = _ROOT / "scripts" / "sync-task-db.sh"
_RELOCATE = _ROOT / "src" / "kiss" / "scripts" / "relocate_work_dir.py"

# ssh accepts its options before the destination; sync_db.py passes
# ``-o BatchMode=yes``, and a caller may add ``-p`` or ``-i``.
_FAKE_SSH = """#!/bin/bash
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-p|-i|-l|-F|-c) shift 2 ;;
        -*) shift ;;
        *) break ;;
    esac
done
shift                       # drop the user@host argument
export HOME="$REMOTE_HOME"
exec bash -c "$*"
"""

_LAPTOP = "/Users/me/work/kiss"
_SERVER = "/home/ubuntu/kiss"


def _make_db(
    path: Path,
    task_ids: list[str],
    work_dir: str = "",
    steps: int = 0,
    events: int = 0,
) -> None:
    """Create a task database with the real schema and some tasks in it.

    Args:
        path: Database file to create.
        task_ids: Identifier of each task to insert.
        work_dir: Directory every task is recorded as having run in.
        steps: Step count of every task.
        events: Number of event rows to add per task.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    th._init_tables(con)
    for task_id in task_ids:
        con.execute(
            "INSERT INTO task_history(id, timestamp, task, work_dir, steps,"
            " has_events) VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, 1.0, f"task {task_id}", work_dir, steps, int(events > 0)),
        )
        for seq in range(events):
            con.execute(
                "INSERT INTO events(task_id, seq, event_json, timestamp)"
                " VALUES (?, ?, ?, ?)",
                (task_id, seq, json.dumps({"task": task_id, "seq": seq}), 1.0),
            )
    con.commit()
    con.close()


def _tasks(path: Path) -> dict[str, str]:
    """Return every task's id mapped to its recorded work directory."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {
            str(task_id): str(work_dir)
            for task_id, work_dir in con.execute(
                "SELECT id, work_dir FROM task_history"
            )
        }
    finally:
        con.close()


def _backups(kiss_dir: Path) -> list[Path]:
    """Return the databases a wholesale replacement moved aside, oldest first."""
    return sorted(
        p for p in kiss_dir.glob("sorcar.db.replaced-*") if not p.name.endswith("-wal")
    )


def _steps(path: Path, task_id: str) -> int:
    """Return the step count recorded for *task_id*."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return int(
            con.execute(
                "SELECT steps FROM task_history WHERE id = ?", (task_id,)
            ).fetchone()[0]
        )
    finally:
        con.close()


def _event_seqs(path: Path, task_id: str) -> list[int]:
    """Return the sequence numbers of *task_id*'s events, in order."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [
            int(row[0])
            for row in con.execute(
                "SELECT seq FROM events WHERE task_id = ? ORDER BY seq", (task_id,)
            )
        ]
    finally:
        con.close()


class SyncTaskDbBothWaysTest(unittest.TestCase):
    """A deploy must leave both machines holding both machines' tasks."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.local_kiss = Path(self.tmp) / "laptop" / ".kiss"
        self.remote_home = Path(self.tmp) / "server"
        self.remote_kiss = self.remote_home / ".kiss"
        self.remote_kiss.mkdir(parents=True)
        self.local_kiss.mkdir(parents=True)
        self.local_db = self.local_kiss / "sorcar.db"
        self.remote_db = self.remote_kiss / "sorcar.db"
        bindir = Path(self.tmp) / "bin"
        bindir.mkdir()
        fake_ssh = bindir / "ssh"
        fake_ssh.write_text(_FAKE_SSH)
        fake_ssh.chmod(0o755)
        # The fake ssh runs the "remote" half locally, and changing HOME does
        # not give it a private systemd — without this stand-in the tests
        # would stop and start the developer's own kiss-web service.
        systemctl = bindir / "systemctl"
        systemctl.write_text("#!/bin/bash\nexit 0\n")
        systemctl.chmod(0o755)
        self.env = dict(os.environ)
        self.env["PATH"] = f"{bindir}:{self.env['PATH']}"
        self.env["KISS_HOME"] = str(self.local_kiss)
        self.env["REMOTE_HOME"] = str(self.remote_home)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _sync(self, *dirs: str) -> subprocess.CompletedProcess[str]:
        """Run the real script against the sandbox remote."""
        return subprocess.run(
            ["bash", str(_SCRIPT), "user@example.com", *dirs],
            env=self.env, capture_output=True, text=True, timeout=300,
        )

    def test_tasks_from_both_machines_end_up_on_both(self) -> None:
        """The point of the exercise: neither side loses, both sides gain."""
        _make_db(self.local_db, ["L1", "L2"])
        _make_db(self.remote_db, ["R1", "R2"])

        result = self._sync(_LAPTOP, _SERVER)
        self.assertEqual(result.returncode, 0, result.stderr)

        self.assertEqual(sorted(_tasks(self.local_db)), ["L1", "L2", "R1", "R2"])
        self.assertEqual(sorted(_tasks(self.remote_db)), ["L1", "L2", "R1", "R2"])
        # Merged in place, so no database was replaced wholesale.
        self.assertEqual(_backups(self.remote_kiss), [])

    def test_a_live_local_task_is_not_mistaken_for_remote_work(self) -> None:
        """The deploy task itself must not make the final restart refuse.

        ``rsorcar`` can be launched by a Sorcar task.  Its database sync runs
        while that task is still writing events.  Shipping the unfinished row
        to the server makes the final remote safety probe report a new live
        *remote* task, even though the task is running on this machine, and
        every such deployment aborts just before restarting the service.
        Finished work should travel now; live work can travel on the next run.
        """
        _make_db(self.local_db, ["DONE"], work_dir=_LAPTOP, events=1)
        con = sqlite3.connect(self.local_db)
        con.execute("UPDATE task_history SET end_ts = ? WHERE id = 'DONE'", (time.time(),))
        con.execute(
            "INSERT INTO task_history(id, timestamp, task, work_dir, has_events)"
            " VALUES ('LIVE', ?, 'the deploy itself', ?, 1)",
            (time.time(), _LAPTOP),
        )
        con.execute(
            "INSERT INTO events(task_id, seq, event_json, timestamp)"
            " VALUES ('LIVE', 0, '{}', ?)",
            (time.time(),),
        )
        con.commit()
        con.close()
        _make_db(self.remote_db, ["REMOTE"], work_dir=_SERVER)

        result = self._sync(_LAPTOP, _SERVER)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(sorted(_tasks(self.remote_db)), ["DONE", "REMOTE"])
        self.assertEqual(sorted(_tasks(self.local_db)), ["DONE", "LIVE", "REMOTE"])
        self.assertIn("live local task", result.stdout)

    def test_the_remotes_tasks_arrive_pointing_at_this_checkout(self) -> None:
        """Otherwise the panel's Workspace chip hides the whole import.

        Both machines have the project at their own path, so each side's
        tasks have to be translated on the way in — the checkout itself
        and everything under it (the agent does most of its work in
        ``.kiss-worktrees/``), while directories outside the project are
        left alone.  Neither live database may be rewritten in the
        process: each keeps recording its own paths.
        """
        _make_db(self.local_db, ["L1"], work_dir=f"{_LAPTOP}/.kiss-worktrees/w")
        _make_db(self.remote_db, ["R1"], work_dir=_SERVER)
        con = sqlite3.connect(self.remote_db)
        con.execute(
            "INSERT INTO task_history(id, timestamp, task, work_dir)"
            " VALUES ('R2', 1.0, 'elsewhere', '/home/ubuntu/other')"
        )
        con.execute(
            "INSERT INTO task_history(id, timestamp, task, work_dir)"
            " VALUES ('R3', 1.0, 'lookalike', '/home/ubuntu/kiss_ai')"
        )
        con.commit()
        con.close()

        self.assertEqual(self._sync(_LAPTOP, _SERVER).returncode, 0)

        here = _tasks(self.local_db)
        self.assertEqual(here["R1"], _LAPTOP)
        self.assertEqual(here["R2"], "/home/ubuntu/other")
        self.assertEqual(here["R3"], "/home/ubuntu/kiss_ai")
        self.assertEqual(here["L1"], f"{_LAPTOP}/.kiss-worktrees/w")
        there = _tasks(self.remote_db)
        self.assertEqual(there["L1"], f"{_SERVER}/.kiss-worktrees/w")
        self.assertEqual(there["R1"], _SERVER, "the remote's own row was rewritten")

    def test_the_events_of_the_remotes_tasks_travel_back(self) -> None:
        """A task without its events replays as an empty transcript."""
        _make_db(self.local_db, ["L1"], events=2)
        _make_db(self.remote_db, ["R1"], events=5)

        self.assertEqual(self._sync(_LAPTOP, _SERVER).returncode, 0)

        self.assertEqual(_event_seqs(self.local_db, "R1"), [0, 1, 2, 3, 4])
        self.assertEqual(_event_seqs(self.local_db, "L1"), [0, 1])
        self.assertEqual(_event_seqs(self.remote_db, "L1"), [0, 1])
        self.assertEqual(_event_seqs(self.remote_db, "R1"), [0, 1, 2, 3, 4])

    def test_a_second_sync_moves_nothing(self) -> None:
        """Re-pointed paths must not make every row look new every time."""
        _make_db(self.local_db, ["L1"], work_dir=_LAPTOP, events=3)
        _make_db(self.remote_db, ["R1"], work_dir=_SERVER, events=3)

        self.assertEqual(self._sync(_LAPTOP, _SERVER).returncode, 0)
        again = self._sync(_LAPTOP, _SERVER)
        self.assertEqual(again.returncode, 0, again.stderr)

        idle = "0 task row(s) added, 0 updated, 0 event row(s) added"
        self.assertEqual(again.stdout.count(idle), 2, again.stdout)
        self.assertEqual(sorted(_tasks(self.local_db)), ["L1", "R1"])
        self.assertEqual(sorted(_tasks(self.remote_db)), ["L1", "R1"])

    def test_a_task_that_got_further_on_the_remote_wins(self) -> None:
        """The same task on both sides: the run that progressed is kept."""
        _make_db(self.local_db, ["T"], steps=3)
        _make_db(self.remote_db, ["T"], steps=17)

        self.assertEqual(self._sync(_LAPTOP, _SERVER).returncode, 0)

        self.assertEqual(_steps(self.local_db, "T"), 17)
        self.assertEqual(_steps(self.remote_db, "T"), 17)

    def test_a_favourite_marked_here_survives_the_sync(self) -> None:
        """A flag the user set is not progress, and must not be reverted."""
        _make_db(self.local_db, ["T"], steps=5)
        _make_db(self.remote_db, ["T"], steps=5)
        con = sqlite3.connect(self.local_db)
        con.execute("UPDATE task_history SET is_favorite = 1 WHERE id = 'T'")
        con.commit()
        con.close()

        self.assertEqual(self._sync(_LAPTOP, _SERVER).returncode, 0)

        for database in (self.local_db, self.remote_db):
            con = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            favourite = con.execute(
                "SELECT is_favorite FROM task_history WHERE id = 'T'"
            ).fetchone()[0]
            con.close()
            self.assertEqual(favourite, 1, database)

    def test_neither_side_overwrites_the_other_at_equal_progress(self) -> None:
        """Two copies of one task that differ without either being ahead.

        Whatever each machine last recorded about the task stays there:
        a sync that cannot tell which version is newer must not pick one
        and destroy the other.
        """
        _make_db(self.local_db, ["T"], steps=4)
        _make_db(self.remote_db, ["T"], steps=4)
        for database, note in ((self.local_db, "here"), (self.remote_db, "there")):
            con = sqlite3.connect(database)
            con.execute("UPDATE task_history SET result = ? WHERE id = 'T'", (note,))
            con.commit()
            con.close()

        self.assertEqual(self._sync(_LAPTOP, _SERVER).returncode, 0)

        for database, note in ((self.local_db, "here"), (self.remote_db, "there")):
            con = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            kept = con.execute(
                "SELECT result FROM task_history WHERE id = 'T'"
            ).fetchone()[0]
            con.close()
            self.assertEqual(kept, note, database)

    def test_a_failed_pull_stops_the_sync_instead_of_replacing_anything(
        self,
    ) -> None:
        """The remote's rows must be here before its database is replaced.

        The snapshot the pull reads cannot be written (a directory sits
        in its place), so nothing comes back.  A run that reported
        success and pushed on would leave the server's tasks nowhere but
        on the server — and the next fallback would bury them.
        """
        _make_db(self.local_db, ["L1"])
        _make_db(self.remote_db, ["R1"])
        (self.remote_kiss / "sorcar.db.outgoing").mkdir()

        result = self._sync(_LAPTOP, _SERVER)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Could not snapshot", result.stdout)
        self.assertEqual(sorted(_tasks(self.local_db)), ["L1"])
        self.assertEqual(_backups(self.remote_kiss), [])

    def test_an_unreachable_remote_is_not_reported_as_synced(self) -> None:
        """Not knowing what is there is not the same as there being nothing.

        The probe cannot run, so the remote may well hold tasks this run
        should have brought back — including when this machine has no
        database of its own to show for the failure.
        """
        broken = Path(self.tmp) / "bin" / "ssh"
        broken.write_text("#!/bin/bash\nexit 255\n")
        broken.chmod(0o755)

        result = self._sync(_LAPTOP, _SERVER)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not in sync yet", result.stderr)
        self.assertFalse(self.local_db.exists())

    def test_a_half_answered_probe_counts_as_no_answer(self) -> None:
        """A connection that dies mid-command must not look like a reply.

        ssh can print a line and then fail, which used to leave the
        probe's answer and the failure marker concatenated — a value no
        branch recognised, and so one that quietly meant "nothing to
        bring back".
        """
        _make_db(self.remote_db, ["R1"])
        half = Path(self.tmp) / "bin" / "ssh"
        half.write_text("#!/bin/bash\necho ok\nexit 255\n")
        half.chmod(0o755)

        result = self._sync(_LAPTOP, _SERVER)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Could not read the task database", result.stdout)
        self.assertEqual(sorted(_tasks(self.remote_db)), ["R1"])

    def test_a_shape_the_relocation_cannot_read_changes_nothing(self) -> None:
        """A translation that fails must not let raw paths through.

        ``WITHOUT ROWID`` leaves ``task_history`` with no rowid to
        address rows by, so the relocation fails on both machines.
        Merging the untranslated snapshots would write each machine's
        paths over the other's; the sync stops with both databases as
        they were.
        """
        for database, task_id, work_dir in (
            (self.local_db, "L1", _LAPTOP), (self.remote_db, "R1", _SERVER)
        ):
            con = sqlite3.connect(database)
            con.execute(
                "CREATE TABLE task_history(id TEXT PRIMARY KEY, work_dir TEXT)"
                " WITHOUT ROWID"
            )
            con.execute("CREATE TABLE events(task_id TEXT, seq INTEGER)")
            con.execute(
                "INSERT INTO task_history(id, work_dir) VALUES (?, ?)",
                (task_id, work_dir),
            )
            con.commit()
            con.close()

        result = self._sync(_LAPTOP, _SERVER)

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(_tasks(self.local_db), {"L1": _LAPTOP})
        self.assertEqual(_tasks(self.remote_db), {"R1": _SERVER})
        self.assertEqual(_backups(self.remote_kiss), [])

    def test_this_machines_database_is_never_replaced(self) -> None:
        """The pull merges in place, so a local web app keeps its file."""
        _make_db(self.local_db, ["L1"])
        _make_db(self.remote_db, ["R1"])
        inode = self.local_db.stat().st_ino

        self.assertEqual(self._sync(_LAPTOP, _SERVER).returncode, 0)

        self.assertEqual(self.local_db.stat().st_ino, inode)

    def test_no_snapshot_is_left_behind_on_the_remote(self) -> None:
        """A second copy of the database must not squat on the server."""
        _make_db(self.local_db, ["L1"])
        _make_db(self.remote_db, ["R1"])

        self.assertEqual(self._sync(_LAPTOP, _SERVER).returncode, 0)

        self.assertFalse((self.remote_kiss / "sorcar.db.outgoing").exists())
        self.assertFalse((self.remote_kiss / "sorcar.db.incoming").exists())

    def test_a_first_sync_here_downloads_the_remotes_database(self) -> None:
        """A machine with no history yet still ends up with the server's."""
        _make_db(self.remote_db, [f"R{i}" for i in range(30)], work_dir=_SERVER)
        self.assertFalse(self.local_db.exists())

        result = self._sync(_LAPTOP, _SERVER)
        self.assertEqual(result.returncode, 0, result.stderr)

        here = _tasks(self.local_db)
        self.assertEqual(len(here), 30)
        self.assertEqual(here["R7"], _LAPTOP)
        self.assertEqual(len(_tasks(self.remote_db)), 30)
        self.assertEqual(_backups(self.remote_kiss), [])

    def test_a_remote_without_a_database_gets_this_machines(self) -> None:
        """First deploy: there is nothing to bring back, only to send."""
        _make_db(self.local_db, ["L1", "L2"], work_dir=_LAPTOP)

        result = self._sync(_LAPTOP, _SERVER)
        self.assertEqual(result.returncode, 0, result.stderr)

        self.assertEqual(_tasks(self.remote_db), {"L1": _SERVER, "L2": _SERVER})
        self.assertEqual(_backups(self.remote_kiss), [])

    def test_neither_machine_having_a_database_is_not_an_error(self) -> None:
        """Nothing to sync is a fine outcome, not a failed deploy."""
        result = self._sync(_LAPTOP, _SERVER)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nothing to send", result.stdout)
        self.assertFalse(self.remote_db.exists())

    def test_an_unreadable_remote_database_still_receives_this_machines(
        self,
    ) -> None:
        """There is nothing to pull from a corrupt file, but plenty to push."""
        self.remote_db.write_bytes(b"this is not a database")
        _make_db(self.local_db, ["L1"])

        result = self._sync(_LAPTOP, _SERVER)
        self.assertEqual(result.returncode, 0, result.stderr)

        self.assertEqual(sorted(_tasks(self.remote_db)), ["L1"])
        # The unreadable file was moved aside, never deleted.
        kept = _backups(self.remote_kiss)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].read_bytes(), b"this is not a database")

    def test_a_second_replacement_keeps_the_first_backup_too(self) -> None:
        """A backup may be the only copy left, so it is never written over."""
        _make_db(self.local_db, ["L1"])
        self.remote_db.write_bytes(b"corrupt one")
        self.assertEqual(self._sync(_LAPTOP, _SERVER).returncode, 0)
        self.remote_db.write_bytes(b"corrupt two")

        self.assertEqual(self._sync(_LAPTOP, _SERVER).returncode, 0)

        self.assertEqual(
            sorted(p.read_bytes() for p in _backups(self.remote_kiss)),
            [b"corrupt one", b"corrupt two"],
        )

    def test_a_replacement_keeps_the_previous_databases_wal_as_well(self) -> None:
        """The -wal holds committed pages the main file does not have yet."""
        _make_db(self.local_db, ["L1"])
        _make_db(self.remote_db, [], work_dir=_SERVER)
        # A web app on the server, holding the database open — which is why
        # the -wal is there at all: SQLite folds it away on the last close.
        # Dropping ``events`` makes the remote unmergeable, which is what
        # sends the push down its wholesale path.
        live = sqlite3.connect(self.remote_db)
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("DROP TABLE events")
        live.execute(
            "INSERT INTO task_history(id, timestamp, task, work_dir)"
            " VALUES ('R1', 1.0, 'only in the wal', ?)",
            (_SERVER,),
        )
        live.commit()
        self.assertTrue((self.remote_kiss / "sorcar.db-wal").exists())

        try:
            result = self._sync(_LAPTOP, _SERVER)
            self.assertEqual(result.returncode, 0, result.stderr)
        finally:
            live.close()

        self.assertEqual(sorted(_tasks(self.remote_db)), ["L1"])
        kept = _backups(self.remote_kiss)
        self.assertEqual(len(kept), 1)
        self.assertEqual(sorted(_tasks(kept[0])), ["R1"])
        self.assertFalse((self.remote_kiss / "sorcar.db-wal").exists())

    def test_a_remote_whose_rows_cannot_be_taken_is_left_alone(self) -> None:
        """Refusing beats replacing: the rows are still only on the server.

        A remote whose ``task_history`` has a column this machine's does
        not can be neither pulled from nor merged into.  Replacing it
        would leave its tasks in a backup file and in neither History
        panel, so the sync stops instead, changes nothing, and says so.
        """
        _make_db(self.local_db, ["L1"])
        _make_db(self.remote_db, ["R1"])
        con = sqlite3.connect(self.remote_db)
        con.execute("ALTER TABLE task_history ADD COLUMN drifted TEXT")
        con.commit()
        con.close()

        result = self._sync(_LAPTOP, _SERVER)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to replace", result.stdout)
        self.assertEqual(sorted(_tasks(self.remote_db)), ["R1"])
        self.assertEqual(sorted(_tasks(self.local_db)), ["L1"])
        self.assertEqual(_backups(self.remote_kiss), [])

    def test_the_history_panel_here_lists_the_remotes_tasks(self) -> None:
        """The pulled database is what this machine's History panel reads."""
        saved = (th._KISS_DIR, th._DB_PATH, th._db_conn)
        try:
            th._KISS_DIR = self.remote_kiss
            th._DB_PATH = self.remote_db
            th._db_conn = None
            th._close_db()
            for i in range(7):
                th._add_task(f"server task {i}", "", {"work_dir": _SERVER})

            th._KISS_DIR = self.local_kiss
            th._DB_PATH = self.local_db
            th._db_conn = None
            th._close_db()
            th._add_task("laptop task", "", {"work_dir": _LAPTOP})
            th._close_db()

            self.assertEqual(self._sync(_LAPTOP, _SERVER).returncode, 0)

            th._db_conn = None
            th._close_db()
            listed = {str(e["task"]): str(e["work_dir"]) for e in th._load_history()}
        finally:
            th._KISS_DIR, th._DB_PATH, th._db_conn = saved
            th._close_db()
        self.assertEqual(len(listed), 8)
        self.assertEqual(listed["server task 3"], _LAPTOP)
        self.assertEqual(listed["laptop task"], _LAPTOP)


class RelocateWorkDirTest(unittest.TestCase):
    """The path translation both directions of the sync depend on."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.db = Path(self.tmp) / "sorcar.db"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run the relocation script as the sync script runs it."""
        return subprocess.run(
            ["python3", str(_RELOCATE), *args],
            capture_output=True, text=True, timeout=120,
        )

    def test_relocating_onto_the_same_directory_changes_nothing(self) -> None:
        """A deployment at the same path as this checkout is a no-op."""
        _make_db(self.db, ["T"], work_dir=_LAPTOP)

        result = self._run(str(self.db), _LAPTOP, _LAPTOP + "/")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0")
        self.assertEqual(_tasks(self.db)["T"], _LAPTOP)

    def test_metadata_that_is_not_an_object_is_left_alone(self) -> None:
        """The legacy ``extra`` column may hold anything at all."""
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE task_history(id INTEGER PRIMARY KEY, extra TEXT)")
        con.executemany(
            "INSERT INTO task_history(id, extra) VALUES (?, ?)",
            [(1, "[1, 2]"), (2, "{not json"), (3, json.dumps({"work_dir": _SERVER}))],
        )
        con.commit()
        con.close()

        result = self._run(str(self.db), _SERVER, _LAPTOP)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1")

        con = sqlite3.connect(self.db)
        rows = dict(con.execute("SELECT id, extra FROM task_history"))
        con.close()
        self.assertEqual(rows[1], "[1, 2]")
        self.assertEqual(rows[2], "{not json")
        self.assertEqual(json.loads(rows[3])["work_dir"], _LAPTOP)

    def test_a_database_recording_no_work_directory_is_left_alone(self) -> None:
        """Neither column present: nothing to translate, no failure."""
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE task_history(id INTEGER PRIMARY KEY, task TEXT)")
        con.execute("INSERT INTO task_history(id, task) VALUES (1, 't')")
        con.commit()
        con.close()

        result = self._run(str(self.db), _SERVER, _LAPTOP)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0")

    def test_a_file_that_is_not_a_database_fails_loudly(self) -> None:
        """A torn snapshot must be reported, not silently accepted."""
        self.db.write_bytes(b"not a database at all")

        result = self._run(str(self.db), _SERVER, _LAPTOP)
        self.assertEqual(result.returncode, 1)
        self.assertIn("relocate_work_dir", result.stderr)

    def test_the_wrong_number_of_arguments_is_a_usage_error(self) -> None:
        """The script is called from shell code; misuse must not be quiet."""
        result = self._run(str(self.db), _SERVER)

        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
