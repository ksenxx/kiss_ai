# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the area-F ``useful_tools`` fixes.

Covers:

* F-RC2 — ``_kill_process_group`` returns without signalling when the
  subprocess is already reaped.  The hazardous interleaving itself is
  unreachable without test doubles: it needs (1) the shell to exit and
  be reaped by ``Popen.wait``, (2) the OS to recycle the shell's PID as
  a NEW process-group leader, and (3) ``_stop_monitor`` to fire inside
  that window (its post-``stop_event`` grace is <=0.2s) — PID recycling
  cannot be forced deterministically from user space.  Per the testing
  policy the guard is therefore tested directly: a reaped process must
  make ``_kill_process_group`` a no-op, while a live process group must
  still be killed.
* F-R1 — ``Bash`` has one uniform error contract whether or not a
  ``stream_callback`` is installed (the redundant pre-``try`` branch
  that bypassed the ``except`` is gone).
* F-R4 — ``_file_lock`` supports ``blocking=False`` (yields ``None``
  when the lock is held elsewhere, a truthy value when acquired) and
  creates the lock file mode ``0600``; ``flock`` treats separate
  descriptors of one file as independent lockers, so contention is
  exercised for real without a second process.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import time
from pathlib import Path

from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
    _file_lock,
    _kill_process_group,
)


class TestKillProcessGroupReapedGuard:
    """F-RC2: no ``killpg`` after the shell has been reaped."""

    def test_reaped_process_is_left_alone(self) -> None:
        # After wait() the PID/PGID may already belong to an unrelated
        # process; _kill_process_group must not signal it.
        process = subprocess.Popen(
            ["/bin/sh", "-c", "exit 0"], start_new_session=True,
        )
        assert process.wait(timeout=10) == 0
        returncode = process.returncode
        start = time.monotonic()
        _kill_process_group(process)  # must be a silent no-op
        assert time.monotonic() - start < 1.0
        assert process.returncode == returncode, (
            "the reaped process's recorded exit status must be untouched"
        )

    def test_live_process_group_is_still_killed(self) -> None:
        process = subprocess.Popen(
            ["/bin/sh", "-c", "sleep 300"], start_new_session=True,
        )
        assert process.poll() is None
        _kill_process_group(process)
        assert process.wait(timeout=10) == -signal.SIGKILL


class TestBashUniformErrorContract:
    """F-R1: streaming and non-streaming Bash behave identically."""

    def test_output_parity_with_and_without_stream_callback(
        self, tmp_path: Path,
    ) -> None:
        streamed: list[str] = []
        with_cb = UsefulTools(
            work_dir=str(tmp_path), stream_callback=streamed.append,
        )
        without_cb = UsefulTools(work_dir=str(tmp_path))
        cmd = "echo out-line; echo err-line >&2; exit 3"
        out_a = with_cb.Bash(cmd, "probe")
        out_b = without_cb.Bash(cmd, "probe")
        assert out_a == out_b
        assert out_a.startswith("Error (exit code 3):")
        assert "out-line" in out_a and "err-line" in out_a
        assert any("out-line" in chunk for chunk in streamed)


class TestFileLockNonBlocking:
    """F-R4: ``_file_lock``'s non-blocking mode and lock-file mode."""

    def test_blocking_acquire_yields_truthy_and_sets_mode(
        self, tmp_path: Path,
    ) -> None:
        lock_path = tmp_path / "locks" / "probe.lock"
        with _file_lock(lock_path) as held:
            assert held
        assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o600

    def test_nonblocking_yields_none_while_held_elsewhere(
        self, tmp_path: Path,
    ) -> None:
        lock_path = tmp_path / "probe.lock"
        with _file_lock(lock_path) as outer:
            assert outer
            # A second descriptor of the same file is an independent
            # flock holder: this is real contention, not a double.
            with _file_lock(lock_path, blocking=False) as inner:
                assert inner is None, (
                    "non-blocking acquire must report contention, "
                    "not wait"
                )
        # Released: the non-blocking path must acquire immediately.
        with _file_lock(lock_path, blocking=False) as held:
            assert held
