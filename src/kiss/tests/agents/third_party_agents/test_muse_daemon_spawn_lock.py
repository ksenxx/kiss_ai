# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Concurrent ``ensure_daemon`` callers start exactly one Muse-auth daemon.

Every caller used to probe the socket, find nothing and spawn its own
``muse_auth.daemon`` process.  The daemon's bind lock kept all but one
of them from listening, but each loser was still a full interpreter
start that probed and exited.  ``ensure_daemon`` now serializes the
check-and-spawn step on ``spawn.lock``, so waiters find the winner's
daemon instead of spawning.

The test releases several caller processes at once and records every
daemon process that carries this test's ``KISS_HOME`` while they run.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.third_party_agents.muse_auth._common import (
    PROTOCOL_VERSION,
    muse_auth_dir,
)
from kiss.agents.third_party_agents.muse_auth.client import (
    _daemon_protocol,
    ensure_daemon,
)
from kiss.tests.agents.third_party_agents.muse_test_utils import (
    _process_running,
    muse_daemons_of,
    setup_muse_env,
    teardown_muse_env,
)

_CALLERS = 8
MUSE_POLICY = {"defaults": {"read": "allow", "write": "ask"}}
# Upper bound on the whole race: a hung caller fails the test instead of
# hanging the suite (pytest timeouts are disabled repo-wide).
_DEADLINE_S = 60.0

_CALLER_SCRIPT = """
import os, sys, time
from kiss.agents.third_party_agents.muse_auth.client import ensure_daemon
print("ready", flush=True)
while not os.path.exists(sys.argv[1]):
    time.sleep(0.001)
ensure_daemon()
print("ok", flush=True)
"""


def _watch_daemons(
    kiss_home: str, workers: list[subprocess.Popen[str]] | list[threading.Thread]
) -> set[int]:
    """Record every daemon pid while ``workers`` run.

    A losing daemon lives for at least an interpreter start, so a tight
    ``/proc`` poll catches every spawned daemon.  ``ensure_daemon`` spawns
    only under ``spawn.lock`` and waits for its daemon before returning,
    so once every worker has finished no daemon can still be starting;
    one last poll covers a spawn between the final two polls.

    Args:
        kiss_home: The ``KISS_HOME`` value that identifies this test's daemons.
        workers: Caller processes or threads running ``ensure_daemon``.

    Returns:
        The pids of all daemons observed.

    Raises:
        AssertionError: When the workers are still running after
            ``_DEADLINE_S`` seconds.
    """
    seen: set[int] = set()
    deadline = time.monotonic() + _DEADLINE_S
    while not all(_finished(worker) for worker in workers):
        assert time.monotonic() < deadline, "ensure_daemon callers hung"
        seen |= muse_daemons_of(kiss_home)
        time.sleep(0.002)
    return seen | muse_daemons_of(kiss_home)


def _finished(worker: subprocess.Popen[str] | threading.Thread) -> bool:
    """Return whether a caller process or thread has finished.

    Args:
        worker: The caller to check.

    Returns:
        True once the process has exited or the thread has ended.
    """
    if isinstance(worker, threading.Thread):
        return not worker.is_alive()
    return worker.poll() is not None


def _kill_daemons(kiss_home: str) -> None:
    """SIGKILL any daemon of ``kiss_home`` still alive and wait for it to exit.

    The daemons are grandchildren of pytest (the callers spawned them),
    so the subprocess reaper does not track them, and
    ``teardown_muse_env`` can only stop a daemon that already listens.
    After a normal run nothing is left; after a failure this removes
    daemons that never bound or that a killed caller left behind.

    Args:
        kiss_home: The ``KISS_HOME`` value that identifies this test's daemons.
    """
    pids = muse_daemons_of(kiss_home)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and any(_process_running(pid) for pid in pids):
        time.sleep(0.02)


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="needs Linux /proc")
def test_concurrent_callers_spawn_one_daemon(
    isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Callers released together spawn a single daemon and all get served."""
    setup_muse_env(monkeypatch, MUSE_POLICY)
    kiss_home = os.environ["KISS_HOME"]
    barrier = tmp_path / "go"
    # Spawning happens inside the try so a failure while spawning still kills
    # the callers already started, which would otherwise wait on the barrier.
    callers: list[subprocess.Popen[str]] = []
    try:
        for _ in range(_CALLERS):
            callers.append(
                subprocess.Popen(
                    [sys.executable, "-c", _CALLER_SCRIPT, str(barrier)],
                    stdout=subprocess.PIPE,
                    text=True,
                )
            )
        deadline = time.monotonic() + _DEADLINE_S
        for caller in callers:
            assert caller.stdout is not None
            remaining = max(0.0, deadline - time.monotonic())
            assert select.select([caller.stdout], [], [], remaining)[0], "caller never got ready"
            assert caller.stdout.readline().strip() == "ready"
        barrier.touch()
        seen = _watch_daemons(kiss_home, callers)
        outputs = [caller.communicate(timeout=30)[0] for caller in callers]
    finally:
        for caller in callers:
            if caller.poll() is None:
                caller.kill()
            caller.wait()
            if caller.stdout is not None:
                caller.stdout.close()
        running = _daemon_protocol() == PROTOCOL_VERSION
        teardown_muse_env()
        _kill_daemons(kiss_home)

    assert all(caller.returncode == 0 for caller in callers), outputs
    assert all(out.strip() == "ok" for out in outputs), outputs
    assert running
    assert len(seen) == 1, f"expected one daemon, saw pids {sorted(seen)}"
    assert (muse_auth_dir() / "spawn.lock").exists()


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="needs Linux /proc")
def test_concurrent_threads_spawn_one_daemon(
    isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Threads of one process also serialize on ``spawn.lock``.

    ``flock`` locks belong to the open file, and every caller opens the
    lock file itself, so threads exclude each other just like processes.
    The threads that wait take the re-check-under-the-lock return.
    """
    setup_muse_env(monkeypatch, MUSE_POLICY)
    kiss_home = os.environ["KISS_HOME"]
    start = threading.Barrier(_CALLERS)
    errors: list[BaseException] = []
    threads = [
        # Daemon threads: a hung caller fails the deadline assertion and
        # must not then keep the interpreter alive after pytest ends.
        threading.Thread(target=_call_after_barrier, args=(start, errors), daemon=True)
        for _ in range(_CALLERS)
    ]
    try:
        for thread in threads:
            thread.start()
        seen = _watch_daemons(kiss_home, threads)
        running = _daemon_protocol() == PROTOCOL_VERSION
    finally:
        for thread in threads:
            thread.join(timeout=30)
        teardown_muse_env()
        _kill_daemons(kiss_home)

    assert errors == []
    assert running
    assert len(seen) == 1, f"expected one daemon, saw pids {sorted(seen)}"


def _call_after_barrier(start: threading.Barrier, errors: list[BaseException]) -> None:
    """Wait for every thread to arrive, then call ``ensure_daemon``.

    Args:
        start: Barrier that releases all callers together.
        errors: Collects any exception a caller raises, a broken barrier included.
    """
    try:
        start.wait(timeout=_DEADLINE_S)
        ensure_daemon()
    except BaseException as e:  # reported by the test's assertion
        errors.append(e)
