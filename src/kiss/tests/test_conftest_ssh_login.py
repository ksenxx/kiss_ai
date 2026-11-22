# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A suite started from an ssh login runs its shells as one started locally does.

``sshd`` exports ``SSH_CLIENT``, and a top-level bash started under it reads
``~/.bashrc`` before its command.  That file may redefine ``sleep``, so the
sleep guard spends no long ``sleep`` on a running job in such an environment
(``useful_tools._runs_environment_text``).  A login shell that exports
``SHLVL`` (bash, zsh) hides the login from the suite.  PowerShell under
OpenSSH on Windows exports none: the tests of that wait slept their 60 s out
and failed there, only in a suite started straight from the ssh session
(four tests on 2026-11-22).  ``conftest.py`` drops the login's markers for
the session.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path

from kiss.agents.sorcar.useful_tools import UsefulTools


def _let_the_job_end(marker: Path) -> str:
    """Create *marker*; as ``UsefulTools.idle_children``, which only a wait on a job asks here.

    The job runs until the marker exists, so it ends once the wait on it has begun,
    however late the call that waits starts.  The empty answer lets that wait go on.
    """
    marker.touch()
    return ""


def test_long_sleep_is_spent_on_the_running_job(tmp_path: Path) -> None:
    """The wait the tests of the sleep guard rely on, in the environment of this session."""
    tools = UsefulTools(work_dir=str(tmp_path))
    marker = tmp_path / "waited"
    tools.idle_children = functools.partial(_let_the_job_end, marker)
    tools.Bash("until [ -e waited ]; do sleep 0.1; done", "job", background=True)
    try:
        out = tools.Bash("sleep 31; echo after", "nap", timeout_seconds=40)
    finally:
        marker.touch()  # a wait that was not taken leaves the job running
    assert out.startswith("after\n") and "Waited on job " in out, out


def test_a_suite_started_from_an_ssh_login_spends_it_too() -> None:
    """The same test in a pytest started as sshd starts a command with no shell above it."""
    env = {name: value for name, value in os.environ.items() if name != "SHLVL"}
    env.update(SSH_CLIENT="203.0.113.7 50000 22", SSH2_CLIENT="203.0.113.7 50000 22")
    test = f"{__file__}::test_long_sleep_is_spent_on_the_running_job"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
