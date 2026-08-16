# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for the asyncTearDown orphan-sweep guard.

The root ``tests/conftest.py`` joins every live ``orphan-task-sweep``
thread before a test's teardown closes ``persistence._db_conn``.  That
guard originally wrapped only the sync ``tearDown`` — but CPython's
``IsolatedAsyncioTestCase._callTearDown`` awaits ``asyncTearDown``
FIRST, and dozens of async server test files close the connection
inside ``asyncTearDown``.  A sweep still inside ``db.execute(...)`` at
that moment dereferences a freed connection and kills the whole pytest
process with SIGSEGV (observed order-dependently in large
``tests/agents/vscode`` runs, e.g. ``test_per_window_reply_isolation``
tearing down while ``_recover_orphaned_tasks`` was mid-SQL).

These tests drive the real race in a pytest subprocess.  The scenario
seeds enough genuine orphan sentinel rows that the real sweep spawned
by ``VSCodeServer.__init__`` demonstrably outlives the test body, then
tears down exactly like the production fixtures do:

* run WITH the root conftest guard loaded — the sweep must already be
  finished when the ``asyncTearDown`` body runs, so the run passes;
* run WITHOUT the guard — the scenario's teardown observes the sweep
  still alive (or the interpreter dies outright), proving the scenario
  reproduces the pre-fix crash window and the guard is what closes it.

No mocks: the subprocess uses the real ``VSCodeServer``, the real
sweep thread, and the real SQLite database.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_RACE_MARKER = "orphan-task-sweep still running when asyncTearDown body started"

_SCENARIO_SOURCE = f'''
"""Race scenario: async fixture teardown vs. the real orphan sweep."""
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.server.server import VSCodeServer

# Enough sentinel rows that the sweep (per-row liveness probe, per-row
# forensic queries and log lines, then one bulk UPDATE) runs for around
# a second — the gap between an instant test body and its teardown is
# a few milliseconds, so without the conftest guard the sweep is still
# mid-SQL when asyncTearDown runs.
N_SENTINEL_ROWS = 60_000


class SweepTeardownRace(unittest.IsolatedAsyncioTestCase):
    """Mirror of the production fixture pattern that used to SIGSEGV."""

    async def asyncSetUp(self) -> None:
        self._saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(tempfile.mkdtemp(prefix="kiss_sweep_race_"))
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        db = th._get_db()
        old_ts = time.time() - 3600.0
        rows = [
            (
                uuid.uuid4().hex,
                old_ts,
                "orphaned task %d" % i,
                "Agent Failed Abruptly",
                "dead-owner-%d" % i,
            )
            for i in range(N_SENTINEL_ROWS)
        ]
        with th._rw_lock.write_lock(), th._immediate_txn(db):
            db.executemany(
                "INSERT INTO task_history "
                "(id, timestamp, task, result, owner) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        # Spawns the real "orphan-task-sweep" daemon thread.
        self.server = VSCodeServer()

    async def test_teardown_races_the_sweep(self) -> None:
        """Finish immediately, while the sweep is still walking rows."""

    async def asyncTearDown(self) -> None:
        alive = [
            t
            for t in threading.enumerate()
            if t.name == "orphan-task-sweep" and t.is_alive()
        ]
        assert not alive, "{_RACE_MARKER}"
        # The exact pattern the production async fixtures use — closing
        # the module-global connection, which may belong to the sweep
        # thread.  Safe only because the guard joined the sweep above.
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved
'''


def _run_scenario(guarded: bool) -> subprocess.CompletedProcess[str]:
    """Run the race scenario in a pytest subprocess.

    Args:
        guarded: When True the root ``kiss.tests.conftest`` (which
            carries the sweep-join guard) is loaded as a plugin,
            exactly as it is in normal test runs; when False the
            scenario runs bare, as pytest ran before the guard existed.

    Returns:
        The completed subprocess, with combined stdout/stderr text.
    """
    tmp = Path(tempfile.mkdtemp(prefix="kiss_sweep_guard_"))
    scenario = tmp / "test_sweep_race_scenario.py"
    scenario.write_text(_SCENARIO_SOURCE, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(scenario),
        "-q",
        "--show-capture=no",
        "-p",
        "no:cacheprovider",
    ]
    if guarded:
        cmd += ["-p", "kiss.tests.conftest"]
    env = dict(os.environ)
    env["KISS_HOME"] = str(tmp / "kiss_home")
    env.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(
        cmd,
        cwd=tmp,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


class TestAsyncTeardownSweepGuard(unittest.TestCase):
    """The conftest guard must join sweeps before asyncTearDown runs."""

    def test_unguarded_scenario_detects_the_race(self) -> None:
        """Without the guard the sweep outlives the test body.

        The scenario either fails its liveness assertion (the sweep is
        provably still running when ``asyncTearDown`` starts — the
        pre-fix crash window) or, if the close wins the race at the C
        level, kills the interpreter outright.  Either way the exit
        status is non-zero, proving the scenario genuinely exercises
        the race the guard exists to close.
        """
        result = _run_scenario(guarded=False)
        self.assertNotEqual(
            result.returncode,
            0,
            "scenario passed without the guard — the race window "
            f"was not reproduced:\n{result.stdout}\n{result.stderr}",
        )
        crashed = result.returncode < 0
        self.assertTrue(
            crashed or _RACE_MARKER in result.stdout + result.stderr,
            "scenario failed for an unrelated reason:\n"
            f"rc={result.returncode}\n{result.stdout}\n{result.stderr}",
        )

    def test_guarded_scenario_passes(self) -> None:
        """With the guard loaded the same scenario must pass cleanly."""
        result = _run_scenario(guarded=True)
        self.assertEqual(
            result.returncode,
            0,
            "guarded scenario did not pass:\n"
            f"rc={result.returncode}\n{result.stdout}\n{result.stderr}",
        )
