# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for what a deploy must not destroy.

``./sorcar-cloud`` writes to a machine somebody else is using: it copies
ssh keys into a home directory that has keys of its own, edits a
``~/.bashrc`` that was written by hand, rewrites a ``config.json`` that
holds every setting the web app remembers, and -- when a merge is
impossible -- replaces the task database wholesale.  Each of those steps
used to have a way of taking something with it that nobody could get
back.  The tests here drive the real scripts over real files and real
databases and assert that nothing is lost:

* ``scripts/install-api-keys.sh``          -- ~/.bashrc is edited, not overwritten;
* ``src/kiss/scripts/remote_config.py``    -- other settings survive, and a file
                                              json cannot read is kept;
* ``src/kiss/scripts/running_tasks.py``    -- a running task is recognised, so a
                                              deploy can refuse to kill it;
* ``src/kiss/scripts/carry_over_tables.py``-- the tables no sync moves come back
                                              from the database that was replaced;
* ``scripts/sync-task-db.sh``              -- a replacement never leaves the
                                              machine without a database, and
                                              never runs while a task does.
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
_INSTALL_KEYS = _ROOT / "scripts" / "install-api-keys.sh"
_SYNC_TASK_DB = _ROOT / "scripts" / "sync-task-db.sh"
_RUNNING_TASKS = _ROOT / "src" / "kiss" / "scripts" / "running_tasks.py"
_REMOTE_CONFIG = _ROOT / "src" / "kiss" / "scripts" / "remote_config.py"
_CARRY_OVER = _ROOT / "src" / "kiss" / "scripts" / "carry_over_tables.py"
_FINGERPRINT = _ROOT / "src" / "kiss" / "scripts" / "db_fingerprint.py"
# Everything scripts/sync-task-db.sh reaches for, so that a test can run it
# from a copy of the tree with one of them left out.
_HELPERS = ("sync_db.py", "relocate_work_dir.py", "carry_over_tables.py",
            "running_tasks.py", "db_fingerprint.py")

_FAKE_SSH = """#!/bin/bash
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-p|-i|-l|-F|-c) shift 2 ;;
        -*) shift ;;
        *) break ;;
    esac
done
shift
export HOME="$REMOTE_HOME"
exec bash -c "$*"
"""

_LAPTOP = "/Users/me/work/kiss"
_SERVER = "/home/ubuntu/kiss"


def _run(
    *command: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a command and capture its output as text.

    Args:
        command: Program and arguments.
        env: Environment for the command; the caller's when None.

    Returns:
        The completed process.
    """
    return subprocess.run(
        list(command), capture_output=True, text=True, timeout=300, env=env, check=False
    )


class InstallApiKeysTest(unittest.TestCase):
    """The remote's own ~/.bashrc has to survive being wired up."""

    def setUp(self) -> None:
        self.home = Path(tempfile.mkdtemp())
        self.kiss = self.home / ".kiss"
        self.kiss.mkdir()
        (self.kiss / "api_keys.env").write_text("export FOO_API_KEY=secret\n")
        (self.kiss / "api_keys.systemd.env").write_text("FOO_API_KEY=secret\n")
        self.rc = self.home / ".bashrc"
        self.env = {**os.environ, "HOME": str(self.home)}

    def tearDown(self) -> None:
        shutil.rmtree(self.home, ignore_errors=True)

    def _install(self) -> subprocess.CompletedProcess[str]:
        """Run the real script with HOME pointed at the sandbox."""
        done = _run("bash", str(_INSTALL_KEYS), env=self.env)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done

    def test_the_existing_bashrc_is_kept_and_extended(self) -> None:
        """Somebody wrote that file; a deploy adds to it and nothing more."""
        self.rc.write_text("export PS1='server$ '\nalias ll='ls -l'\n")

        self._install()

        text = self.rc.read_text()
        self.assertIn("export PS1='server$ '", text)
        self.assertIn("alias ll='ls -l'", text)
        self.assertIn(".kiss/api_keys.env", text)

    def test_the_file_as_it_was_is_kept_once(self) -> None:
        """A second deploy must not overwrite the pristine copy with an edited one."""
        self.rc.write_text("original\n")

        self._install()
        kept = list(self.kiss.glob("bashrc-before-sorcar-*"))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].read_text(), "original\n")

        self._install()
        self.assertEqual([p.read_text() for p in
                          self.kiss.glob("bashrc-before-sorcar-*")], ["original\n"])

    def test_a_second_run_does_not_add_a_second_block(self) -> None:
        """Deploys are repeated; the block is replaced, not accumulated."""
        self.rc.write_text("original\n")

        self._install()
        self._install()
        self._install()

        text = self.rc.read_text()
        self.assertEqual(text.count("sorcar-cloud API keys >>>"), 1)
        self.assertEqual(text.count("sorcar-cloud API keys <<<"), 1)
        self.assertEqual(text.count('. "$HOME/.kiss/api_keys.env"'), 1)
        self.assertEqual(text.count("original"), 1)

    def test_nothing_is_left_behind_and_the_keys_are_private(self) -> None:
        """A temporary file beside ~/.bashrc is somebody's next surprise."""
        self._install()

        self.assertFalse((self.home / ".bashrc.sorcar-new").exists())
        self.assertFalse((self.home / ".bashrc.sorcar-tmp").exists())
        for name in ("api_keys.env", "api_keys.systemd.env"):
            self.assertEqual((self.kiss / name).stat().st_mode & 0o077, 0)

    def test_the_permissions_of_the_bashrc_are_preserved(self) -> None:
        """The rewrite goes through a copy, which must not change the mode."""
        self.rc.write_text("original\n")
        self.rc.chmod(0o600)

        self._install()

        self.assertEqual(self.rc.stat().st_mode & 0o777, 0o600)

    def test_an_unterminated_block_does_not_swallow_the_rest_of_the_file(self) -> None:
        """An interrupted earlier deploy must not cost somebody their shell.

        Skipping everything after the begin marker throws away the whole
        tail of the file when the end marker is missing -- and, because
        the marker is there, the file looks like one this script wrote,
        so no copy of it would be kept either.
        """
        self.rc.write_text(
            "export PS1='server$ '\n"
            "# >>> sorcar-cloud API keys >>>\n"
            "alias ll='ls -l'\n"
            "export EDITOR=vim\n"
        )

        done = self._install()

        text = self.rc.read_text()
        self.assertIn("alias ll='ls -l'", text)
        self.assertIn("export EDITOR=vim", text)
        self.assertIn("export PS1='server$ '", text)
        # And the file as it was is kept, because it is not one this script wrote.
        kept = list(self.kiss.glob("bashrc-before-sorcar-*"))
        self.assertEqual(len(kept), 1)
        self.assertIn("alias ll='ls -l'", kept[0].read_text())
        self.assertIn("begin and", done.stdout)

    def test_the_second_deploy_over_an_unterminated_block_keeps_them_too(self) -> None:
        """The first deploy is not the one that loses them.

        After the first deploy the file holds the stray begin marker and,
        further down, a complete block of its own.  Pairing that marker
        with the *later* block's end marker deletes everything between
        the two -- somebody's aliases, on the second deploy, in a file the
        first deploy handled correctly.
        """
        self.rc.write_text(
            "export PS1='server$ '\n"
            "# >>> sorcar-cloud API keys >>>\n"
            "alias ll='ls -l'\n"
            "export EDITOR=vim\n"
        )

        self._install()
        self._install()

        text = self.rc.read_text()
        self.assertIn("alias ll='ls -l'", text)
        self.assertIn("export EDITOR=vim", text)
        self.assertIn("export PS1='server$ '", text)
        # And still exactly one working block, not two.
        self.assertEqual(text.count('. "$HOME/.kiss/api_keys.env"'), 1)

    def test_a_bashrc_that_is_a_link_is_edited_where_it_lives(self) -> None:
        """Replacing the link would disconnect a dotfiles repository."""
        real = self.home / "dotfiles" / "bashrc"
        real.parent.mkdir()
        real.write_text("from the dotfiles repo\n")
        self.rc.unlink(missing_ok=True)
        self.rc.symlink_to(real)

        self._install()

        self.assertTrue(self.rc.is_symlink())
        self.assertEqual(self.rc.resolve(), real.resolve())
        self.assertIn("from the dotfiles repo", real.read_text())
        self.assertIn(".kiss/api_keys.env", real.read_text())
        self.assertFalse((real.parent / "bashrc.sorcar-new").exists())


class RemoteConfigTest(unittest.TestCase):
    """config.json holds everything the web app remembers between runs."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.config = self.tmp / "config.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _configure(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run the real script over the sandbox configuration file."""
        done = _run("python3", str(_REMOTE_CONFIG), str(self.config), *args)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done

    def test_every_other_setting_survives(self) -> None:
        """A deploy sets two keys; the rest is not its business."""
        self.config.write_text(json.dumps(
            {"last_model": "claude-opus-5", "notify": {"ntfy": True},
             "remote_password": "keepme"}))

        self._configure(_SERVER)

        saved = json.loads(self.config.read_text())
        self.assertEqual(saved["last_model"], "claude-opus-5")
        self.assertEqual(saved["notify"], {"ntfy": True})
        self.assertEqual(saved["remote_password"], "keepme")
        self.assertEqual(saved["work_dir"], _SERVER)

    def test_a_file_json_cannot_read_is_kept(self) -> None:
        """It may be the only copy of settings that took a while to get right."""
        self.config.write_text('{"last_model": "claude-opus-5",')

        done = self._configure(_SERVER)

        kept = list(self.tmp.glob("config.json.unreadable-*"))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].read_text(), '{"last_model": "claude-opus-5",')
        self.assertIn("kept the previous", done.stderr)
        self.assertEqual(json.loads(self.config.read_text())["work_dir"], _SERVER)

    def test_json_that_is_not_an_object_is_kept_too(self) -> None:
        """It parses, but it is no more usable -- and no less somebody's file."""
        self.config.write_text('["a list nobody expected"]')

        self._configure(_SERVER)

        kept = list(self.tmp.glob("config.json.unreadable-*"))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].read_text(), '["a list nobody expected"]')

    def test_a_config_that_is_a_link_is_written_where_it_lives(self) -> None:
        """Renaming over the link would leave the real file behind.

        A ``~/.kiss/config.json`` that points into a dotfiles repository
        is followed by the web app, so a deploy that replaces the link
        with a regular file leaves the maintained file holding the old
        settings and the web app reading a copy nobody maintains.
        """
        real = self.tmp / "dotfiles" / "config.json"
        real.parent.mkdir()
        real.write_text(json.dumps({"last_model": "claude-opus-5"}))
        self.config.symlink_to(real)

        self._configure(_SERVER, "chosen")

        self.assertTrue(self.config.is_symlink())
        self.assertEqual(self.config.resolve(), real.resolve())
        saved = json.loads(real.read_text())
        self.assertEqual(saved["last_model"], "claude-opus-5")
        self.assertEqual(saved["remote_password"], "chosen")
        self.assertEqual(saved["work_dir"], _SERVER)
        self.assertFalse((real.parent / "config.json.sorcar-new").exists())

    def test_a_file_that_cannot_be_kept_is_not_replaced_either(self) -> None:
        """Stopping the deploy beats destroying the only copy of the settings."""
        self.config.write_text("{not json")
        self.tmp.chmod(0o500)                      # nothing new can be created here
        try:
            done = _run("python3", str(_REMOTE_CONFIG), str(self.config), _SERVER)
        finally:
            self.tmp.chmod(0o700)

        self.assertEqual(done.returncode, 1)
        self.assertEqual(self.config.read_text(), "{not json")

    def test_a_given_password_replaces_the_configured_one(self) -> None:
        """SORCAR_PASSWORD is how the password is set on purpose."""
        self.config.write_text(json.dumps({"remote_password": "old"}))

        done = self._configure(_SERVER, "new-one")

        self.assertIn("SORCAR_PASSWORD=new-one", done.stdout)
        self.assertEqual(
            json.loads(self.config.read_text())["remote_password"], "new-one")

    def test_a_machine_without_a_password_gets_one(self) -> None:
        """The web app only opens a public tunnel when a password is set."""
        done = self._configure(_SERVER)

        password = done.stdout.strip().split("=", 1)[1]
        self.assertGreaterEqual(len(password), 12)
        self.assertEqual(
            json.loads(self.config.read_text())["remote_password"], password)
        self.assertEqual(self.config.stat().st_mode & 0o077, 0)

    def test_no_half_written_file_is_left_behind(self) -> None:
        """The new content is renamed over the old one, never poured into it."""
        self._configure(_SERVER)
        self._configure(_SERVER)

        self.assertEqual(
            sorted(p.name for p in self.tmp.iterdir()), ["config.json"])


class RunningTasksTest(unittest.TestCase):
    """A deploy restarts the web app, so it has to see a task running."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "sorcar.db"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make(self, tasks: list[tuple[str, int, float | None]]) -> None:
        """Create a database of ``(task_id, end_ts, newest_event_age)`` tasks.

        Args:
            tasks: One triple per task; an age of None means the task has
                no events at all.
        """
        con = sqlite3.connect(self.db)
        th._init_tables(con)
        now = time.time()
        for task_id, end_ts, age in tasks:
            con.execute(
                "INSERT INTO task_history(id, timestamp, task, end_ts)"
                " VALUES (?, ?, ?, ?)", (task_id, now, f"task {task_id}", end_ts))
            if age is not None:
                con.execute(
                    "INSERT INTO events(task_id, seq, event_json, timestamp)"
                    " VALUES (?, 0, '{}', ?)", (task_id, now - age))
        con.commit()
        con.close()

    def _count(self, *args: str) -> tuple[int, str]:
        """Run the real script and return the count it printed and its output."""
        done = _run("python3", str(_RUNNING_TASKS), str(self.db), *args)
        self.assertEqual(done.returncode, 0, done.stderr)
        return int(done.stdout.splitlines()[0]), done.stdout

    def test_a_task_writing_events_now_is_running(self) -> None:
        """That is the one a deploy would kill."""
        self._make([("LIVE", 0, 2.0)])

        count, output = self._count("300")

        self.assertEqual(count, 1)
        self.assertIn("LIVE", output)

    def test_the_rows_old_crashes_left_are_not_running(self) -> None:
        """A database holds thousands of those; refusing on them is refusing always."""
        self._make([("CRASHED", 0, 90000.0), ("NO_EVENTS", 0, None),
                    ("FINISHED", int(time.time()), 1.0)])

        self.assertEqual(self._count("300")[0], 0)

    def test_the_window_is_what_decides(self) -> None:
        """An event two minutes old is a heartbeat for one deploy, not another."""
        self._make([("RECENT", 0, 120.0)])

        self.assertEqual(self._count("300")[0], 1)
        self.assertEqual(self._count("60")[0], 0)
        self.assertEqual(self._count("0")[0], 0)

    def test_a_machine_with_no_history_has_nothing_running(self) -> None:
        """A first deploy: no database, and therefore nothing to lose."""
        self.assertEqual(self._count("300")[0], 0)

    def test_a_file_that_is_not_a_database_stops_nothing(self) -> None:
        """It cannot be holding a running task, whatever else is wrong with it."""
        self.db.write_bytes(b"not a database")

        self.assertEqual(self._count("300")[0], 0)

    def test_a_database_that_cannot_be_read_is_not_reported_as_idle(self) -> None:
        """"I could not look" must not read as "nothing is running"."""
        # An SQLite file whose content is unusable: it opens, and then every
        # query fails -- so nothing can be said about what runs out of it.
        self.db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 512)

        done = _run("python3", str(_RUNNING_TASKS), str(self.db), "300")

        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stdout.splitlines()[0], "unknown")

    def test_the_wrong_number_of_arguments_is_a_usage_error(self) -> None:
        """A silent 0 would read as "nothing is running"."""
        done = _run("python3", str(_RUNNING_TASKS))
        self.assertEqual(done.returncode, 2)
        done = _run("python3", str(_RUNNING_TASKS), str(self.db), "soon")
        self.assertEqual(done.returncode, 2)


class FingerprintTest(unittest.TestCase):
    """Counting tasks does not notice a task getting further along."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "sorcar.db"
        con = sqlite3.connect(self.db)
        th._init_tables(con)
        con.execute(
            "INSERT INTO task_history(id, timestamp, task, steps, tokens)"
            " VALUES ('T1', 1.0, 'a task', 3, 100)")
        con.execute(
            "INSERT INTO events(task_id, seq, event_json, timestamp)"
            " VALUES ('T1', 0, '{}', 1.0)")
        con.commit()
        con.close()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fingerprint(self) -> str:
        """Return the fingerprint the real script prints."""
        done = _run("python3", str(_FINGERPRINT), str(self.db))
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout.strip()

    def _execute(self, sql: str) -> None:
        """Run one statement against the database."""
        con = sqlite3.connect(self.db)
        con.execute(sql)
        con.commit()
        con.close()

    def test_the_same_database_gives_the_same_answer(self) -> None:
        """Otherwise every comparison would look like a change."""
        self.assertEqual(self._fingerprint(), self._fingerprint())

    def test_an_event_written_by_a_task_changes_it(self) -> None:
        """The task count would not have moved at all."""
        before = self._fingerprint()
        self._execute(
            "INSERT INTO events(task_id, seq, event_json, timestamp)"
            " VALUES ('T1', 1, '{}', 2.0)")

        self.assertNotEqual(self._fingerprint(), before)
        self.assertTrue(before.startswith("1 "))

    def test_a_task_getting_further_along_changes_it(self) -> None:
        """A step or a token spent on a task nobody here knows about."""
        before = self._fingerprint()
        self._execute("UPDATE task_history SET steps = 4 WHERE id = 'T1'")
        after = self._fingerprint()

        self.assertNotEqual(after, before)
        # Same number of tasks in both, which is the whole point.
        self.assertEqual(after.split()[0], before.split()[0])

    def test_a_task_finishing_changes_it(self) -> None:
        """A task can end without any count moving at all.

        No task is added, no step is taken and no event is written when a
        task records the moment it finished -- and a fingerprint that does
        not notice would let an older copy of that task be written over
        it.
        """
        before = self._fingerprint()
        self._execute("UPDATE task_history SET end_ts = 1700000000 WHERE id = 'T1'")
        after = self._fingerprint()

        self.assertNotEqual(after, before)
        self.assertEqual(after.split()[:6], before.split()[:6])

    def test_what_a_task_cost_changes_it(self) -> None:
        """Cost is money spent on the server that this machine never saw."""
        before = self._fingerprint()
        self._execute("UPDATE task_history SET cost = 0.42 WHERE id = 'T1'")

        self.assertNotEqual(self._fingerprint(), before)

    def test_a_favourite_marked_there_changes_it(self) -> None:
        """Somebody starred a task on the server between the two passes."""
        before = self._fingerprint()
        self._execute("UPDATE task_history SET is_favorite = 1 WHERE id = 'T1'")

        self.assertNotEqual(self._fingerprint(), before)

    def test_a_result_written_there_changes_it(self) -> None:
        """The summary of a task is the whole point of having run it."""
        before = self._fingerprint()
        self._execute("UPDATE task_history SET result = 'what it found' WHERE id = 'T1'")

        self.assertNotEqual(self._fingerprint(), before)

    def test_a_new_task_changes_it(self) -> None:
        """The obvious case still has to work."""
        before = self._fingerprint()
        self._execute(
            "INSERT INTO task_history(id, timestamp, task) VALUES ('T2', 2.0, 'x')")

        self.assertNotEqual(self._fingerprint(), before)

    def test_a_database_it_cannot_read_is_not_an_answer(self) -> None:
        """An empty answer must never read as "nothing changed"."""
        self.db.write_bytes(b"not a database")

        done = _run("python3", str(_FINGERPRINT), str(self.db))

        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stdout.strip(), "")

    def test_the_wrong_number_of_arguments_is_a_usage_error(self) -> None:
        """A blank line would compare equal to another blank line."""
        self.assertEqual(_run("python3", str(_FINGERPRINT)).returncode, 2)


class CarryOverTablesTest(unittest.TestCase):
    """What a wholesale replacement would otherwise take with it."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.old = self.tmp / "old.db"
        self.new = self.tmp / "new.db"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make(self, path: Path, models: dict[str, int],
              files: dict[str, int] | None = None) -> None:
        """Create a database whose usage counters hold the given tallies.

        Args:
            path: Database file to create.
            models: How often each model was chosen.
            files: How often each file was opened.
        """
        con = sqlite3.connect(path)
        th._init_tables(con)
        for model, count in models.items():
            con.execute(
                "INSERT INTO model_usage(model, count) VALUES (?, ?)", (model, count))
        for name, count in (files or {}).items():
            con.execute(
                "INSERT INTO file_usage(path, count, last_used)"
                " VALUES (?, ?, ?)", (name, count, float(count)))
        con.commit()
        con.close()

    def _carry(self) -> subprocess.CompletedProcess[str]:
        """Run the real script and require it to succeed."""
        done = _run("python3", str(_CARRY_OVER), str(self.old), str(self.new))
        self.assertEqual(done.returncode, 0, done.stderr)
        return done

    def _models(self, path: Path) -> dict[str, int]:
        """Return the model tallies of a database."""
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return {str(m): int(c) for m, c in
                    con.execute("SELECT model, count FROM model_usage")}
        finally:
            con.close()

    def test_the_tallies_only_the_replaced_database_had_come_back(self) -> None:
        """They are the deployment's own; nothing else records them."""
        self._make(self.old, {"claude-opus-5": 40, "gpt-5.6-sol": 7},
                   files={"server.py": 3})
        self._make(self.new, {"claude-opus-5": 2})

        self._carry()

        self.assertEqual(self._models(self.new),
                         {"claude-opus-5": 40, "gpt-5.6-sol": 7})
        con = sqlite3.connect(f"file:{self.new}?mode=ro", uri=True)
        self.assertEqual(
            con.execute("SELECT path, count FROM file_usage").fetchall(),
            [("server.py", 3)])
        con.close()

    def test_a_larger_tally_in_the_new_database_is_not_lowered(self) -> None:
        """Counters are raised, never copied: neither side loses a count."""
        self._make(self.old, {"claude-opus-5": 3})
        self._make(self.new, {"claude-opus-5": 90})

        self._carry()

        self.assertEqual(self._models(self.new), {"claude-opus-5": 90})

    def test_running_it_twice_changes_nothing_the_second_time(self) -> None:
        """A deploy is repeated; a counter must not creep."""
        self._make(self.old, {"claude-opus-5": 40})
        self._make(self.new, {})

        self._carry()
        first = self._models(self.new)
        self._carry()

        self.assertEqual(self._models(self.new), first)

    def test_a_replaced_file_that_cannot_be_read_changes_nothing(self) -> None:
        """Being unreadable is exactly why such a file was replaced."""
        self.old.write_bytes(b"not a database")
        self._make(self.new, {"claude-opus-5": 5})

        self._carry()

        self.assertEqual(self._models(self.new), {"claude-opus-5": 5})

    def test_a_missing_file_changes_nothing(self) -> None:
        """The remote may have had no database at all."""
        self._make(self.new, {"claude-opus-5": 5})

        self._carry()

        self.assertEqual(self._models(self.new), {"claude-opus-5": 5})

    def test_a_tally_of_nothing_is_left_where_it_is(self) -> None:
        """A NULL key matches nothing, so copying it would repeat for ever.

        Today's schema forbids it, an older one did not, and the file
        being carried over is by definition an older machine's.
        """
        self._make(self.old, {})
        self._make(self.new, {"claude-opus-5": 1})
        con = sqlite3.connect(self.old)
        con.execute("DROP TABLE model_usage")
        con.execute(
            "CREATE TABLE model_usage (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " model TEXT UNIQUE, count INTEGER DEFAULT 0, is_last INTEGER DEFAULT 0)")
        con.execute("INSERT INTO model_usage(model, count) VALUES (NULL, 7)")
        con.commit()
        con.close()

        self._carry()
        first = self._models(self.new)
        self._carry()

        self.assertEqual(first, {"claude-opus-5": 1})
        self.assertEqual(self._models(self.new), first)

    def test_a_carry_over_that_fails_says_so(self) -> None:
        """Reporting success would leave the counters only in the backup."""
        self._make(self.old, {"claude-opus-5": 40})
        self._make(self.new, {})
        self.new.chmod(0o444)
        try:
            done = _run("python3", str(_CARRY_OVER), str(self.old), str(self.new))
        finally:
            self.new.chmod(0o644)

        self.assertEqual(done.returncode, 1)

    def test_the_wrong_number_of_arguments_is_a_usage_error(self) -> None:
        """Silence would look like a successful carry-over."""
        self.assertEqual(_run("python3", str(_CARRY_OVER)).returncode, 2)


class ReplacementNeverLosesTest(unittest.TestCase):
    """The one destructive step of a deploy, held to its promise."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.local_kiss = Path(self.tmp) / "laptop" / ".kiss"
        self.remote_home = Path(self.tmp) / "server"
        self.remote_kiss = self.remote_home / ".kiss"
        self.local_kiss.mkdir(parents=True)
        self.remote_kiss.mkdir(parents=True)
        self.local_db = self.local_kiss / "sorcar.db"
        self.remote_db = self.remote_kiss / "sorcar.db"
        bindir = Path(self.tmp) / "bin"
        bindir.mkdir()
        (bindir / "ssh").write_text(_FAKE_SSH)
        (bindir / "ssh").chmod(0o755)
        # HOME does not give the fake remote a systemd of its own, and these
        # tests must not stop the developer's own kiss-web service.
        (bindir / "systemctl").write_text("#!/bin/bash\nexit 0\n")
        (bindir / "systemctl").chmod(0o755)
        self.env = dict(os.environ)
        self.env["PATH"] = f"{bindir}:{self.env['PATH']}"
        self.env["KISS_HOME"] = str(self.local_kiss)
        self.env["REMOTE_HOME"] = str(self.remote_home)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_db(self, path: Path, task_ids: list[str], *, work_dir: str = "",
                 models: dict[str, int] | None = None,
                 running: str = "", drop_events: bool = False) -> None:
        """Create a task database the way a machine that has run tasks has one.

        Args:
            path: Database file to create.
            task_ids: Identifier of each finished task to insert.
            work_dir: Directory every task is recorded as having run in.
            models: How often each model was chosen.
            running: Identifier of a task that is running right now, if any.
            drop_events: Drop the ``events`` table, which is what makes a
                database unmergeable and sends a push down its wholesale path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
        th._init_tables(con)
        now = time.time()
        for task_id in task_ids:
            con.execute(
                "INSERT INTO task_history(id, timestamp, task, work_dir, end_ts)"
                " VALUES (?, ?, ?, ?, ?)",
                (task_id, now, f"task {task_id}", work_dir, int(now)))
        if running:
            con.execute(
                "INSERT INTO task_history(id, timestamp, task, work_dir, end_ts,"
                " has_events) VALUES (?, ?, ?, ?, 0, 1)",
                (running, now, "running right now", work_dir))
            con.execute(
                "INSERT INTO events(task_id, seq, event_json, timestamp)"
                " VALUES (?, 0, '{}', ?)", (running, now))
        for model, count in (models or {}).items():
            con.execute(
                "INSERT INTO model_usage(model, count) VALUES (?, ?)", (model, count))
        if drop_events:
            con.execute("DROP TABLE events")
        con.commit()
        con.close()

    def _insert_task(self, con: sqlite3.Connection, task_id: str) -> None:
        """Add one finished task over an open connection.

        Args:
            con: Connection to insert through; committed before returning.
            task_id: Identifier of the task to add.
        """
        now = time.time()
        con.execute(
            "INSERT INTO task_history(id, timestamp, task, work_dir, end_ts)"
            " VALUES (?, ?, ?, ?, ?)",
            (task_id, now, f"task {task_id}", _SERVER, int(now)))
        con.commit()

    def _leave_rows_in_the_wal(self, path: Path, task_id: str) -> None:
        """Commit a task and leave it where only a checkpoint will find it.

        The pages of a committed transaction sit in the ``-wal`` until a
        checkpoint folds them into the database, and closing the last
        connection folds them in -- so the pair is copied into place while
        a connection still holds it open.  This is the state a database
        that was being written up to the moment the web app stopped is
        actually in.

        Args:
            path: Database to add the task to.
            task_id: Identifier of the task that will live in the -wal.
        """
        staging = Path(self.tmp) / "staging" / "sorcar.db"
        staging.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, staging)
        con = sqlite3.connect(staging)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            now = time.time()
            con.execute(
                "INSERT INTO task_history(id, timestamp, task, work_dir, end_ts)"
                " VALUES (?, ?, ?, ?, ?)",
                (task_id, now, f"task {task_id}", _SERVER, int(now)))
            con.commit()
            for suffix in ("", "-wal"):
                shutil.copy(f"{staging}{suffix}", f"{path}{suffix}")
        finally:
            con.close()
        shutil.rmtree(staging.parent, ignore_errors=True)
        # The main file on its own does not have the task: that is the point.
        con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        try:
            alone = [str(row[0]) for row in con.execute("SELECT id FROM task_history")]
        finally:
            con.close()
        self.assertNotIn(task_id, alone)

    def _sync(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        """Run the real script against the sandbox remote."""
        return _run("bash", str(_SYNC_TASK_DB), "user@example.com", _LAPTOP, _SERVER,
                    env={**self.env, **overrides})

    def _tasks(self, path: Path) -> list[str]:
        """Return the sorted task ids a database holds."""
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return sorted(str(row[0]) for row in
                          con.execute("SELECT id FROM task_history"))
        finally:
            con.close()

    def _backups(self) -> list[Path]:
        """Return the databases a replacement moved aside, oldest first."""
        return sorted(p for p in self.remote_kiss.glob("sorcar.db.replaced-*")
                      if not p.name.endswith("-wal"))

    def test_the_replaced_database_stays_readable_all_along(self) -> None:
        """It is kept as a second link, so there is never a moment without one."""
        self._make_db(self.local_db, ["L1"])
        self._make_db(self.remote_db, ["R1"], work_dir=_SERVER, drop_events=True)

        result = self._sync()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        self.assertTrue(self.remote_db.is_file())
        self.assertEqual(self._tasks(self.remote_db), ["L1"])
        kept = self._backups()
        self.assertEqual(len(kept), 1)
        self.assertEqual(self._tasks(kept[0]), ["R1"])
        # Nothing of the old database's own journal is left beside the new one.
        self.assertFalse((self.remote_kiss / "sorcar.db-wal").exists())
        self.assertFalse((self.remote_kiss / "sorcar.db-shm").exists())

    def test_the_backup_holds_what_was_only_in_the_wal(self) -> None:
        """The last thing the server did is the likeliest to be only there.

        A copy of the main file taken while committed pages are still out
        in the -wal is a copy of an older database than the one being
        replaced, and the -wal is deleted a moment later -- so those rows
        would survive in neither the backup nor the replacement.
        """
        self._make_db(self.local_db, ["L1"])
        self._make_db(self.remote_db, ["R1"], work_dir=_SERVER, drop_events=True)
        self._leave_rows_in_the_wal(self.remote_db, "RWAL")

        result = self._sync()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        kept = self._backups()
        self.assertEqual(len(kept), 1)
        self.assertEqual(self._tasks(kept[0]), ["R1", "RWAL"])
        self.assertEqual(self._tasks(self.remote_db), ["L1"])
        self.assertFalse((self.remote_kiss / "sorcar.db-wal").exists())
        self.assertFalse((self.remote_kiss / "sorcar.db-shm").exists())

    def test_a_wal_a_reader_holds_open_is_kept_beside_the_backup(self) -> None:
        """A checkpoint can run without error and fold nothing in.

        Something else on the machine -- a reader that opened the database
        before the last commits and has not finished -- keeps the
        checkpoint from truncating the log, and SQLite reports that by
        returning "busy", not by failing.  A caller that only asks whether
        the checkpoint raised deletes a -wal holding rows that are in no
        other file.
        """
        self._make_db(self.local_db, ["L1"])
        self._make_db(self.remote_db, ["R1"], work_dir=_SERVER, drop_events=True)
        writer = sqlite3.connect(self.remote_db, isolation_level=None)
        reader = sqlite3.connect(self.remote_db)
        try:
            writer.execute("PRAGMA journal_mode=WAL")
            self._insert_task(writer, "RPRE")
            # Read here, so this connection is looking at the database as it
            # was before the commit below: the checkpointer may not copy past
            # what a reader can still see.
            reader.execute("BEGIN")
            reader.execute("SELECT count(*) FROM task_history").fetchone()
            self._insert_task(writer, "RWAL")
            writer.close()

            result = self._sync()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        finally:
            reader.close()

        kept = self._backups()
        self.assertEqual(len(kept), 1)
        self.assertTrue(Path(f"{kept[0]}-wal").is_file())
        self.assertEqual(self._tasks(kept[0]), ["R1", "RPRE", "RWAL"])
        self.assertEqual(self._tasks(self.remote_db), ["L1"])
        self.assertFalse((self.remote_kiss / "sorcar.db-wal").exists())

    def test_the_usage_counters_of_the_replaced_database_come_back(self) -> None:
        """No sync moves them, so a replacement has to carry them over."""
        self._make_db(self.local_db, ["L1"], models={"gpt-5.6-sol": 2})
        self._make_db(self.remote_db, ["R1"], work_dir=_SERVER, drop_events=True,
                      models={"claude-opus-5": 40, "gpt-5.6-sol": 1})

        result = self._sync()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        con = sqlite3.connect(f"file:{self.remote_db}?mode=ro", uri=True)
        tallies = {str(m): int(c) for m, c in
                   con.execute("SELECT model, count FROM model_usage")}
        con.close()
        self.assertEqual(tallies, {"claude-opus-5": 40, "gpt-5.6-sol": 2})

    def test_a_running_task_stops_the_replacement(self) -> None:
        """Stopping the web app for the swap would kill it mid-step.

        The remote's database is readable -- so its tasks come back here --
        but cannot be written, which is what sends the push down the
        wholesale path.  That path stops the web app, so it must not run
        while a task is writing events.
        """
        self._make_db(self.local_db, ["L1"])
        self._make_db(self.remote_db, ["R1"], work_dir=_SERVER, running="LIVE")
        self.remote_db.chmod(0o444)

        result = self._sync()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("running on", result.stdout)
        self.assertEqual(self._backups(), [])
        self.assertEqual(self._tasks(self.remote_db), ["LIVE", "R1"])
        # Pass 1 still did its job: the server's tasks are here.
        self.assertEqual(self._tasks(self.local_db), ["L1", "LIVE", "R1"])

    def _tree_without(self, helper: str) -> Path:
        """Copy the script and its helpers, leaving one of them out.

        Args:
            helper: File name under ``src/kiss/scripts`` to omit.

        Returns:
            The path of the copied ``sync-task-db.sh``.
        """
        root = Path(self.tmp) / "checkout"
        (root / "scripts").mkdir(parents=True)
        (root / "src" / "kiss" / "scripts").mkdir(parents=True)
        shutil.copy(_SYNC_TASK_DB, root / "scripts" / "sync-task-db.sh")
        for name in _HELPERS:
            if name != helper:
                shutil.copy(_ROOT / "src" / "kiss" / "scripts" / name,
                            root / "src" / "kiss" / "scripts" / name)
        return root / "scripts" / "sync-task-db.sh"

    def test_a_replacement_it_cannot_vouch_for_does_not_happen(self) -> None:
        """Not knowing what the server held is not the same as it holding nothing.

        The remote's database is readable but not writable, which is what
        sends the push down its wholesale path.  Without a way to check
        that the server has not moved on since its rows were brought
        here, replacing its database could throw away whatever it did in
        the meantime -- so it is left exactly as it is.
        """
        self._make_db(self.local_db, ["L1"])
        self._make_db(self.remote_db, ["R1"], work_dir=_SERVER)
        self.remote_db.chmod(0o444)
        script = self._tree_without("db_fingerprint.py")

        result = _run("bash", str(script), "user@example.com", _LAPTOP, _SERVER,
                      env=self.env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("left as it is", result.stdout)
        self.assertEqual(self._backups(), [])
        self.assertEqual(self._tasks(self.remote_db), ["R1"])
        # Pass 1 still brought the server's task here.
        self.assertEqual(self._tasks(self.local_db), ["L1", "R1"])

    def test_the_replacement_can_be_forced_when_the_task_can_wait(self) -> None:
        """The refusal is a default, not a wall."""
        self._make_db(self.local_db, ["L1"])
        self._make_db(self.remote_db, ["R1"], work_dir=_SERVER, running="LIVE")
        self.remote_db.chmod(0o444)

        result = self._sync(SORCAR_FORCE_RESTART="1")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self._backups()), 1)
        self.assertEqual(self._tasks(self.remote_db), ["L1", "LIVE", "R1"])


if __name__ == "__main__":
    unittest.main()
