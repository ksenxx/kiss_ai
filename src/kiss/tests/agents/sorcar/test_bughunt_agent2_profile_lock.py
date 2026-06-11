# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt integration tests for ``_is_profile_in_use`` (web_use_tool).

``_is_profile_in_use`` decides whether a Chromium profile directory is
locked by a *live* process by sending ``os.kill(pid, 0)`` to the PID
recorded in the ``SingletonLock`` symlink target (``hostname-pid``).

POSIX ``kill(pid, 0)`` semantics:

* raises ``ProcessLookupError`` (ESRCH)  -> the process does NOT exist;
* raises ``PermissionError``    (EPERM)  -> the process DOES exist, the
  caller merely lacks permission to signal it (e.g. it is owned by
  another user or by root);
* returns normally                       -> the process exists.

The buggy code caught ``OSError`` wholesale (``PermissionError`` is a
subclass of ``OSError``) and returned ``False``, i.e. it misreported a
profile locked by a live-but-unsignalable Chromium (e.g. one started by
another user on a shared machine, or by a root-owned session) as FREE.
``_resolve_user_data_dir`` would then happily reuse that locked profile
directory and ``_clean_singleton_locks`` would delete the live browser's
``Singleton*`` files — exactly the corruption the lock exists to prevent.

These tests use only real filesystem symlinks and real PIDs (PID 1 is
the init/launchd process: always alive on every Unix and never
signalable by a non-root test run). No mocks, patches, or fakes.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool, _is_profile_in_use

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="SingletonLock symlinks are Unix-only"
)


def _make_lock(profile_dir, pid: int) -> None:
    """Create a Chromium-style ``SingletonLock`` symlink pointing at *pid*."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(f"testhost-{pid}", str(profile_dir / "SingletonLock"))


def test_profile_locked_by_unsignalable_live_process_is_in_use(tmp_path) -> None:
    """A lock held by a live process we cannot signal must read as in-use.

    PID 1 (init/launchd) always exists.  When the test runs as a normal
    user, ``os.kill(1, 0)`` raises ``PermissionError`` — the kernel's
    way of saying "that process exists but is not yours".  The profile
    is therefore locked and ``_is_profile_in_use`` must return True.
    """
    profile = tmp_path / "profile_eperm"
    _make_lock(profile, 1)
    if os.geteuid() == 0:  # pragma: no cover — root CI can signal PID 1
        pytest.skip("running as root: cannot provoke EPERM from os.kill")
    assert _is_profile_in_use(str(profile)) is True


def test_profile_locked_by_own_live_process_is_in_use(tmp_path) -> None:
    """A lock held by our own live process (signalable) reads as in-use."""
    profile = tmp_path / "profile_alive"
    _make_lock(profile, os.getpid())
    assert _is_profile_in_use(str(profile)) is True


def test_profile_locked_by_dead_process_is_free(tmp_path) -> None:
    """A lock whose PID no longer exists reads as free (stale lock)."""
    # Spawn-and-reap a real child so the PID is guaranteed dead.
    proc = subprocess.Popen(["true"])
    proc.wait()
    pid = proc.pid
    profile = tmp_path / "profile_dead"
    _make_lock(profile, pid)
    assert _is_profile_in_use(str(profile)) is False


def test_profile_with_garbage_lock_target_is_free(tmp_path) -> None:
    """A lock whose symlink target has no parsable PID reads as free."""
    profile = tmp_path / "profile_garbage"
    profile.mkdir(parents=True, exist_ok=True)
    os.symlink("not-a-pid-at-all", str(profile / "SingletonLock"))
    assert _is_profile_in_use(str(profile)) is False


def test_profile_without_lock_is_free(tmp_path) -> None:
    """No ``SingletonLock`` symlink at all reads as free."""
    profile = tmp_path / "profile_empty"
    profile.mkdir(parents=True, exist_ok=True)
    assert _is_profile_in_use(str(profile)) is False


def test_resolve_user_data_dir_skips_unsignalable_locked_profile(tmp_path) -> None:
    """End-to-end: ``_resolve_user_data_dir`` must NOT hand out a profile
    directory whose lock is held by a live-but-unsignalable process."""
    if os.geteuid() == 0:  # pragma: no cover — root CI can signal PID 1
        pytest.skip("running as root: cannot provoke EPERM from os.kill")
    profile = tmp_path / "profile_busy"
    _make_lock(profile, 1)
    tool = WebUseTool(user_data_dir=str(profile))
    resolved = tool._resolve_user_data_dir()
    assert resolved == f"{profile}_1", (
        "locked profile was handed out for reuse instead of a numbered variant"
    )
