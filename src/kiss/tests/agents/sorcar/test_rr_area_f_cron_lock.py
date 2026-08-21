# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the deduplicated cron job-store lock (F-R4).

``cron_agent._jobs_lock`` used to re-implement
``useful_tools._file_lock`` solely to add a non-blocking mode; it is
now a thin wrapper over it (gaining the 0600 lock-file mode and the
Windows branch).  These tests pin the wrapper's contract — the exact
semantics the scheduler tick and the ``cron_job`` tool rely on — on
the real lock file under an isolated ``KISS_HOME``.  The tick-skip
behavior built on top is covered end-to-end by
``test_cron_agent.test_tick_skips_when_lock_held``.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from kiss.agents.sorcar import cron_agent


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KISS_HOME at a per-test temp dir so the lock is isolated."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    return tmp_path


def test_blocking_lock_yields_truthy_and_creates_0600_file(
    tmp_path: Path,
) -> None:
    with cron_agent._jobs_lock(blocking=True) as held:
        assert held
    lock_path = cron_agent._jobs_path().with_suffix(".lock")
    assert lock_path.is_file()
    assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o600


def test_nonblocking_lock_yields_none_while_held() -> None:
    # flock treats separate descriptors of one file as independent
    # holders, so nesting exercises real contention (this is exactly
    # how an overlapping scheduler tick sees a running tool edit).
    with cron_agent._jobs_lock(blocking=True) as held:
        assert held
        with cron_agent._jobs_lock(blocking=False) as inner:
            assert inner is None
    # Released: the non-blocking tick path acquires immediately.
    with cron_agent._jobs_lock(blocking=False) as reacquired:
        assert reacquired
