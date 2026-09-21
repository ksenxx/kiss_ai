# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The RLIMIT_NPROC test helpers must leave the pytest process able to spawn.

``nproc_limit_lowered_to_one`` / ``thread_start_can_be_starved`` (conftest)
lower the soft per-user process limit to 1 so tests can provoke a real
``Thread.start`` failure.  The earlier hand-rolled version restored the
*originally reported* hard limit; macOS reports ``kern.maxproc`` but
clamps the hard limit to ``kern.maxprocperuid`` on the first
``setrlimit``, so the restore raised ``ValueError: not allowed to raise
maximum limit`` and the whole pytest process stayed at soft NPROC=1 --
every later ``subprocess`` spawn in that process failed with
``BlockingIOError``, taking hundreds of unrelated tests down with it.

These tests run the helpers and then prove the process is intact: the
soft limit is back where it started and a child process can be spawned.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from kiss.tests.conftest import (
    is_root,
    nproc_limit_lowered_to_one,
    posix_only,
    thread_start_can_be_starved,
)

pytestmark = posix_only("RLIMIT_NPROC")


def _nproc_limits() -> tuple[int, int]:
    import resource

    return resource.getrlimit(resource.RLIMIT_NPROC)


def _spawn_child_output() -> str:
    """Spawn a child interpreter and return what it printed."""
    child = [sys.executable, "-c", "print('alive')"]
    return subprocess.run(child, capture_output=True, text=True).stdout.strip()


def test_lowering_block_restores_soft_limit_and_spawning() -> None:
    """After the block the soft limit is restored and subprocesses spawn."""
    if is_root():
        pytest.skip("root is not bound by RLIMIT_NPROC")
    soft_before, _ = _nproc_limits()
    with nproc_limit_lowered_to_one():
        assert _nproc_limits()[0] == 1
    soft_after, hard_after = _nproc_limits()
    assert soft_after == min(soft_before, hard_after)
    # The whole point: a later spawn in this process must not fail.
    assert _spawn_child_output() == "alive"


def test_lowering_block_restores_when_body_raises() -> None:
    """The restore also happens when the guarded block raises."""
    if is_root():
        pytest.skip("root is not bound by RLIMIT_NPROC")
    soft_before, _ = _nproc_limits()
    with pytest.raises(RuntimeError, match="boom"), nproc_limit_lowered_to_one():
        raise RuntimeError("boom")
    soft_after, hard_after = _nproc_limits()
    assert soft_after == min(soft_before, hard_after)


def test_probe_never_poisons_the_process() -> None:
    """``thread_start_can_be_starved`` returns a bool and leaves spawning intact."""
    soft_before, _ = _nproc_limits()
    verdict = thread_start_can_be_starved()
    assert isinstance(verdict, bool)
    soft_after, hard_after = _nproc_limits()
    assert soft_after == min(soft_before, hard_after)
    assert _spawn_child_output() == "alive"
