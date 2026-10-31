# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A failed set-up of the ``/sync`` machines test must not leak its git environment.

:class:`~kiss.tests.agents.seas.test_sync_sea_machines.TestSyncOnMachines`
puts a hermetic git configuration and a fixed identity into
``os.environ`` before it starts its two daemons.  ``unittest`` runs
``asyncTearDown`` only when ``asyncSetUp`` succeeded, so a set-up that
fails (a daemon that cannot start) must restore the variables through a
cleanup, or every later test of the same pytest process runs with a
hidden git config and the identity ``t <t@t>``.

The scenario runs in a pytest child: the real ``asyncSetUp`` with
``KISS_HOME`` pointing at a regular file, so the base class fails as soon
as it writes its machine registry, followed by a test that checks the
six variables are back to what they were.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SCENARIO = '''
import os

# The module, not the class: pytest collects every TestCase subclass bound
# in a test module, whatever its name, and the real test must not run here.
from kiss.tests.agents.seas import test_sync_sea_machines

KEYS = (
    "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL",
    "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
)
BEFORE = {key: os.environ.get(key) for key in KEYS}


class TestSetUpFails(test_sync_sea_machines.TestSyncOnMachines):
    """The real set-up, with a KISS_HOME the daemons cannot write to."""

    def setUp(self) -> None:
        self.addCleanup(os.environ.__setitem__, "KISS_HOME", os.environ["KISS_HOME"])
        os.environ["KISS_HOME"] = __file__
        super().setUp()


def test_git_environment_is_back_after_the_failed_set_up() -> None:
    assert {key: os.environ.get(key) for key in KEYS} == BEFORE
'''


def test_failed_async_set_up_restores_the_git_environment(tmp_path: Path) -> None:
    """The scenario's set-up errors out, its follow-up test passes: nothing leaked."""
    scenario = tmp_path / "test_sync_set_up_fails.py"
    scenario.write_text(_SCENARIO, encoding="utf-8")
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "kiss.tests.conftest",
            str(scenario),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    report = f"{proc.stdout}\n{proc.stderr}"
    # pytest reports a unittest set-up failure as a failed test, not an error.  Every test
    # TestSetUpFails inherits fails in its set-up, however many TestSyncOnMachines holds.
    failed = [line for line in proc.stdout.splitlines() if line.startswith("FAILED ")]
    assert failed and all("::TestSetUpFails::" in line for line in failed), report
    assert "FileExistsError" in proc.stdout, report
    assert f"{len(failed)} failed, 1 passed" in proc.stdout.splitlines()[-1], report
