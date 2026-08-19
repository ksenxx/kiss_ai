# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the sending half of ``scripts/sync-task-db.sh``.

Reproduces the reported defect: after ``./sorcar-cloud`` the remote web
app's History panel showed only a handful of tasks instead of the whole
list from the laptop's ``~/.kiss/sorcar.db``.

The cause is not the transfer being incomplete, it is *who else has the
file open*.  ``scp`` writes into an existing destination in place —
``O_WRONLY|O_CREAT|O_TRUNC``, same inode — so a ``kiss-web`` left running
on the remote from an earlier deploy keeps a connection to the very bytes
that were just replaced.  Its page cache and WAL still describe the small,
empty database it created, and the first checkpoint after the swap (a
clean shutdown is enough) writes that view back, truncating the freshly
uploaded database to a couple of pages.
``test_in_place_overwrite_under_a_live_reader_destroys_the_database``
demonstrates exactly that with plain SQLite.

The remaining tests run the real script against a sandbox "remote": a
fake ``ssh`` on ``PATH`` executes the remote half locally with ``HOME``
pointed at another directory.  The database that lands there is then read
back through the real persistence layer — the same code the History panel
is a view of.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

import kiss.agents.sorcar.persistence as th

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "sync-task-db.sh"

_FAKE_SSH = """#!/bin/bash
# Stand-in for ssh: run the command locally with HOME inside the sandbox.
shift                       # drop the user@host argument
export HOME="$REMOTE_HOME"
exec bash -c "$*"
"""


def _make_db(path: Path, tasks: int, first_id: int = 0) -> None:
    """Create a task database at *path* holding *tasks* rows."""
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE task_history(id INTEGER PRIMARY KEY, task TEXT)")
    con.executemany(
        "INSERT INTO task_history(id, task) VALUES (?, ?)",
        [(first_id + i, f"task {first_id + i} {'x' * 400}") for i in range(tasks)],
    )
    con.commit()
    con.close()


def _count(path: Path) -> int:
    """Return the number of rows in *path*'s ``task_history`` table."""
    con = sqlite3.connect(path)
    try:
        return int(con.execute("SELECT count(*) FROM task_history").fetchone()[0])
    finally:
        con.close()


class SyncTaskDbPushTest(unittest.TestCase):
    """The whole task list must survive the trip to the remote host."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.local_home = Path(self.tmp) / "laptop" / ".kiss"
        self.remote_home = Path(self.tmp) / "server"
        (self.remote_home / ".kiss").mkdir(parents=True)
        self.local_home.mkdir(parents=True)
        bindir = Path(self.tmp) / "bin"
        bindir.mkdir()
        fake_ssh = bindir / "ssh"
        fake_ssh.write_text(_FAKE_SSH)
        fake_ssh.chmod(0o755)
        # The fake ssh runs the "remote" half locally, and changing HOME does
        # not give it a private systemd — without this stand-in the tests
        # would stop and start the developer's own kiss-web service.
        self.systemctl_log = Path(self.tmp) / "systemctl.log"
        systemctl = bindir / "systemctl"
        systemctl.write_text(
            f'#!/bin/bash\necho "$@" >> {self.systemctl_log}\nexit 0\n')
        systemctl.chmod(0o755)
        self.env = dict(os.environ)
        self.env["PATH"] = f"{bindir}:{self.env['PATH']}"
        self.env["KISS_HOME"] = str(self.local_home)
        self.env["REMOTE_HOME"] = str(self.remote_home)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _push(self, *relocate: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(_SCRIPT), "user@example.com", *relocate],
            env=self.env, capture_output=True, text=True, timeout=300,
        )

    def test_in_place_overwrite_under_a_live_reader_destroys_the_database(
        self,
    ) -> None:
        """The defect itself: what a plain ``scp`` did to the remote."""
        remote_db = self.remote_home / ".kiss" / "sorcar.db"
        big = Path(self.tmp) / "laptop.db"
        _make_db(big, 2000)

        # The remote's leftover kiss-web, holding its own small database open.
        live = sqlite3.connect(remote_db)
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("CREATE TABLE task_history(id INTEGER PRIMARY KEY, task TEXT)")
        live.execute("INSERT INTO task_history(task) VALUES ('leftover deploy')")
        live.commit()
        inode = remote_db.stat().st_ino

        # scp truncates the destination in place, reusing that inode.
        with open(big, "rb") as src, open(remote_db, "r+b") as dst:
            dst.truncate(0)
            shutil.copyfileobj(src, dst)
        self.assertEqual(remote_db.stat().st_ino, inode)
        uploaded_bytes = remote_db.stat().st_size

        # The leftover -wal still describes the old, empty database, so it is
        # replayed over the upload and readers see one task instead of 2000.
        self.assertEqual(_count(remote_db), 1)

        # sorcar-cloud stops the old service only much later; that clean
        # shutdown checkpoints the stale view, truncating the file for good.
        live.close()
        self.assertLess(remote_db.stat().st_size, uploaded_bytes)
        self.assertEqual(_count(remote_db), 1)

    def test_every_task_reaches_the_remote(self) -> None:
        """The full list arrives, replacing whatever was there before."""
        _make_db(self.remote_home / ".kiss" / "sorcar.db", 3)
        _make_db(self.local_home / "sorcar.db", 500)

        result = self._push()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(_count(self.remote_home / ".kiss" / "sorcar.db"), 500)

    def test_survives_a_live_reader_on_the_remote(self) -> None:
        """A leftover web app holding the old database loses the race."""
        remote_db = self.remote_home / ".kiss" / "sorcar.db"
        _make_db(remote_db, 2)
        live = sqlite3.connect(remote_db)
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("INSERT INTO task_history(task) VALUES ('leftover')")
        live.commit()
        _make_db(self.local_home / "sorcar.db", 400)

        self.assertEqual(self._push().returncode, 0)
        live.close()  # the stale checkpoint now has nothing to damage
        self.assertEqual(_count(remote_db), 400)

    def test_uncommitted_wal_pages_travel_too(self) -> None:
        """Tasks living only in the -wal must not be left behind."""
        local_db = self.local_home / "sorcar.db"
        _make_db(local_db, 10)
        writer = sqlite3.connect(local_db)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.executemany(
            "INSERT INTO task_history(id, task) VALUES (?, ?)",
            [(1000 + i, f"wal-only {i}") for i in range(90)],
        )
        writer.commit()
        self.assertTrue((self.local_home / "sorcar.db-wal").exists())

        self.assertEqual(self._push().returncode, 0)
        self.assertEqual(_count(self.remote_home / ".kiss" / "sorcar.db"), 100)

    def test_no_wal_or_shm_is_left_on_the_remote(self) -> None:
        """The -shm index is machine-local and must never be shipped."""
        kiss = self.remote_home / ".kiss"
        _make_db(kiss / "sorcar.db", 2)
        (kiss / "sorcar.db-wal").write_bytes(b"stale wal from an older deploy")
        (kiss / "sorcar.db-shm").write_bytes(b"stale shm")
        _make_db(self.local_home / "sorcar.db", 20)

        self.assertEqual(self._push().returncode, 0)
        self.assertFalse((kiss / "sorcar.db-wal").exists())
        self.assertFalse((kiss / "sorcar.db-shm").exists())
        self.assertFalse((kiss / "sorcar.db.incoming").exists())

    def test_a_truncated_upload_keeps_the_previous_database(self) -> None:
        """A cut-short transfer must not replace a good database."""
        remote_db = self.remote_home / ".kiss" / "sorcar.db"
        _make_db(remote_db, 7)
        _make_db(self.local_home / "sorcar.db", 300)
        # Truncate the stream the way a dropped connection would.
        truncating = Path(self.tmp) / "bin" / "gzip"
        truncating.write_text(
            "#!/bin/bash\n"
            'if [ "$1" = "-dc" ]; then head -c 4096 > /dev/null; '
            'printf "" > "$HOME/.kiss/sorcar.db.incoming"; exit 0; fi\n'
            'exec /usr/bin/env -i PATH=/usr/bin:/bin gzip "$@"\n'
        )
        truncating.chmod(0o755)

        self.assertNotEqual(self._push().returncode, 0)
        self.assertEqual(_count(remote_db), 7)

    def test_work_dirs_are_relocated_to_the_remote_checkout(self) -> None:
        """Otherwise the panel's Workspace chip hides the whole import.

        Every task remembers where it ran.  The deployment is this
        project at another path, so the recorded paths are re-pointed
        at it — the checkout itself and everything under it (the
        agent does most of its work in ``.kiss-worktrees/``), while
        directories outside the project are left alone.
        """
        local_db = self.local_home / "sorcar.db"
        _make_db(local_db, 0)
        con = sqlite3.connect(local_db)
        con.execute("ALTER TABLE task_history ADD COLUMN work_dir TEXT")
        con.executemany(
            "INSERT INTO task_history(id, task, work_dir) VALUES (?, ?, ?)",
            [
                (1, "in the checkout", "/Users/me/work/kiss"),
                (2, "in a worktree",
                 "/Users/me/work/kiss/.kiss-worktrees/kiss_wt-1"),
                (3, "a lookalike sibling", "/Users/me/work/kiss_ai"),
                (4, "another project", "/Users/me/3rdparty/skydiscover"),
                (5, "no directory at all", ""),
            ],
        )
        con.commit()
        con.close()

        result = self._push("/Users/me/work/kiss", "/home/ubuntu/kiss")
        self.assertEqual(result.returncode, 0, result.stderr)

        con = sqlite3.connect(self.remote_home / ".kiss" / "sorcar.db")
        shipped = dict(con.execute("SELECT task, work_dir FROM task_history"))
        con.close()
        self.assertEqual(shipped["in the checkout"], "/home/ubuntu/kiss")
        self.assertEqual(
            shipped["in a worktree"],
            "/home/ubuntu/kiss/.kiss-worktrees/kiss_wt-1",
        )
        self.assertEqual(shipped["a lookalike sibling"], "/Users/me/work/kiss_ai")
        self.assertEqual(shipped["another project"], "/Users/me/3rdparty/skydiscover")
        self.assertEqual(shipped["no directory at all"], "")

    def test_relocation_leaves_this_machines_database_untouched(self) -> None:
        """Shipping must never rewrite the laptop's own history."""
        local_db = self.local_home / "sorcar.db"
        _make_db(local_db, 0)
        con = sqlite3.connect(local_db)
        con.execute("ALTER TABLE task_history ADD COLUMN work_dir TEXT")
        con.execute(
            "INSERT INTO task_history(id, task, work_dir) VALUES (1, 't', ?)",
            ("/Users/me/work/kiss",),
        )
        con.commit()
        con.close()

        self.assertEqual(
            self._push("/Users/me/work/kiss", "/home/ubuntu/kiss").returncode, 0)

        con = sqlite3.connect(local_db)
        try:
            self.assertEqual(
                con.execute("SELECT work_dir FROM task_history").fetchone()[0],
                "/Users/me/work/kiss",
            )
        finally:
            con.close()

    def test_a_database_without_a_work_dir_column_still_ships(self) -> None:
        """Databases predating the flat ``work_dir`` column must not abort."""
        _make_db(self.local_home / "sorcar.db", 12)  # no work_dir column

        result = self._push("/Users/me/work/kiss", "/home/ubuntu/kiss")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(_count(self.remote_home / ".kiss" / "sorcar.db"), 12)

    def test_legacy_databases_keep_their_paths_in_extra_json(self) -> None:
        """The pre-migration schema stores work_dir inside ``extra``.

        The migration that runs on the remote copies that value across
        unchanged, so it has to be relocated here or the imported
        history arrives complete and invisible.
        """
        local_db = self.local_home / "sorcar.db"
        _make_db(local_db, 0)
        con = sqlite3.connect(local_db)
        con.execute("ALTER TABLE task_history ADD COLUMN extra TEXT")
        con.executemany(
            "INSERT INTO task_history(id, task, extra) VALUES (?, ?, ?)",
            [
                (1, "in the checkout",
                 json.dumps({"model": "m", "work_dir": "/Users/me/work/kiss"})),
                (2, "another project",
                 json.dumps({"work_dir": "/Users/me/3rdparty/skydiscover"})),
                (3, "unreadable metadata", "{not json"),
                (4, "no metadata at all", None),
            ],
        )
        con.commit()
        con.close()

        result = self._push("/Users/me/work/kiss", "/home/ubuntu/kiss")
        self.assertEqual(result.returncode, 0, result.stderr)

        con = sqlite3.connect(self.remote_home / ".kiss" / "sorcar.db")
        shipped = dict(con.execute("SELECT task, extra FROM task_history"))
        con.close()
        self.assertEqual(
            json.loads(shipped["in the checkout"])["work_dir"],
            "/home/ubuntu/kiss",
        )
        self.assertEqual(
            json.loads(shipped["another project"])["work_dir"],
            "/Users/me/3rdparty/skydiscover",
        )
        self.assertEqual(shipped["unreadable metadata"], "{not json")
        self.assertIsNone(shipped["no metadata at all"])

    def test_the_root_directory_relocates_nothing(self) -> None:
        """"/" is not a project, so no path may be rewritten against it."""
        local_db = self.local_home / "sorcar.db"
        _make_db(local_db, 0)
        con = sqlite3.connect(local_db)
        con.execute("ALTER TABLE task_history ADD COLUMN work_dir TEXT")
        con.executemany(
            "INSERT INTO task_history(id, task, work_dir) VALUES (?, ?, ?)",
            [(1, "somewhere", "/Users/me/work/kiss"), (2, "nowhere", "")],
        )
        con.commit()
        con.close()

        self.assertEqual(self._push("/", "/home/ubuntu").returncode, 0)

        con = sqlite3.connect(self.remote_home / ".kiss" / "sorcar.db")
        shipped = dict(con.execute("SELECT task, work_dir FROM task_history"))
        con.close()
        self.assertEqual(shipped, {"somewhere": "/Users/me/work/kiss",
                                   "nowhere": ""})

    def test_the_remote_web_app_is_restarted_even_when_shipping_fails(
        self,
    ) -> None:
        """A failed deploy must not leave the remote without a web app."""
        _make_db(self.local_home / "sorcar.db", 5)
        # Break the upload so the script takes its failure path.
        gzip = Path(self.tmp) / "bin" / "gzip"
        gzip.write_text('#!/bin/bash\nexit 3\n')
        gzip.chmod(0o755)

        self.assertNotEqual(self._push().returncode, 0)
        recorded = self.systemctl_log.read_text()
        self.assertIn("stop kiss-web.service", recorded)
        self.assertIn("start kiss-web.service", recorded)

    def test_the_remote_web_app_is_restarted_when_the_stop_itself_fails(
        self,
    ) -> None:
        """The connection can drop after the service is already stopped."""
        _make_db(self.local_home / "sorcar.db", 5)
        # The last command of the stop step; failing it aborts that step
        # after ``systemctl stop`` has already run.
        rm = Path(self.tmp) / "bin" / "rm"
        rm.write_text('#!/bin/bash\nexit 1\n')
        rm.chmod(0o755)

        self.assertNotEqual(self._push().returncode, 0)
        recorded = self.systemctl_log.read_text()
        self.assertIn("stop kiss-web.service", recorded)
        self.assertIn("start kiss-web.service", recorded)

    def test_shipping_never_kills_a_web_app_outside_the_remote_home(
        self,
    ) -> None:
        """The stop step must only ever match the remote's own web app."""
        _make_db(self.local_home / "sorcar.db", 4)
        killed = Path(self.tmp) / "pkill.log"
        fake = Path(self.tmp) / "bin" / "pkill"
        fake.write_text(f'#!/bin/bash\necho "$@" >> {killed}\nexit 0\n')
        fake.chmod(0o755)

        self.assertEqual(self._push().returncode, 0)
        pattern = killed.read_text().split()[-1]
        self.assertRegex(
            f"{self.remote_home}/.venv/bin/kiss-web --workdir {self.remote_home}",
            pattern,
        )
        self.assertNotRegex(
            "/Users/dev/work/kiss/.venv/bin/kiss-web --workdir /Users/dev/work/kiss",
            pattern,
        )

    def test_history_panel_lists_the_shipped_tasks(self) -> None:
        """The shipped database is what the History panel reads."""
        saved = (th._KISS_DIR, th._DB_PATH, th._db_conn)
        try:
            th._KISS_DIR = self.local_home
            th._DB_PATH = self.local_home / "sorcar.db"
            th._db_conn = None
            th._close_db()
            for i in range(25):
                th._add_task(f"local task {i}", "", {"model": "m"})

            self.assertEqual(self._push().returncode, 0)

            th._KISS_DIR = self.remote_home / ".kiss"
            th._DB_PATH = th._KISS_DIR / "sorcar.db"
            th._db_conn = None
            th._close_db()
            titles = [str(e["task"]) for e in th._load_history()]
            self.assertEqual(len(titles), 25)
            self.assertIn("local task 24", titles)
        finally:
            th._KISS_DIR, th._DB_PATH, th._db_conn = saved
            th._close_db()


if __name__ == "__main__":
    unittest.main()
