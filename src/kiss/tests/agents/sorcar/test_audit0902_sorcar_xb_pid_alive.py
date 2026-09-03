# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (cross-boundary): one shared ``pid_alive`` helper.

Three modules carried their own "is this pid alive" probe with
divergent semantics:

* ``git_worktree.GitWorktreeOps._pid_alive`` — ``pid <= 0`` -> False,
  ``ProcessLookupError`` -> False, any other ``OSError`` -> True;
* ``web_use_tool._pid_alive`` — no ``pid <= 0`` guard (so ``os.kill(0, 0)``
  signalled the caller's own process group and answered True),
  ``PermissionError`` -> True, other ``OSError`` -> False;
* ``web_server._is_pid_alive`` — ``pid <= 0`` -> False,
  ``ProcessLookupError`` -> False, ``PermissionError`` -> True, other
  ``OSError`` -> False.

The single :func:`kiss.agents.sorcar._concurrency.pid_alive` now
carries the ``web_server`` semantics and the three call sites delegate
to it.  These tests probe real processes: a live child, a reaped
child, pid ``0`` / ``-1``, and (when not root) pid ``1`` for the
``PermissionError`` branch.

The ``except OSError`` fallthrough (an ``OSError`` that is neither
``ProcessLookupError`` nor ``PermissionError``, e.g. ``EINVAL`` from
an exotic platform) cannot be triggered from user space on Linux
without test doubles, so it is left uncovered here on purpose.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from kiss.agents.sorcar import web_use_tool
from kiss.agents.sorcar._concurrency import pid_alive
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.server import web_server


def test_live_child_is_alive() -> None:
    """A sleeping child we own is reported alive; after reaping it is not."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert pid_alive(proc.pid) is True
    finally:
        proc.kill()
        proc.wait()
    assert pid_alive(proc.pid) is False


def test_reaped_child_is_dead() -> None:
    """A child that exited and was waited on is a ``ProcessLookupError``."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    assert pid_alive(proc.pid) is False


def test_self_is_alive() -> None:
    """The calling process is trivially alive."""
    assert pid_alive(os.getpid()) is True


def test_non_positive_pids_are_dead() -> None:
    """``0`` (own process group) and ``-1`` (every process) are never 'alive'."""
    assert pid_alive(0) is False
    assert pid_alive(-1) is False


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root may signal pid 1, so no PermissionError branch to exercise",
)
def test_foreign_owned_process_is_alive() -> None:
    """pid 1 belongs to another user: ``PermissionError`` still means alive."""
    with pytest.raises(PermissionError):
        os.kill(1, 0)
    assert pid_alive(1) is True


def test_call_sites_share_the_helper() -> None:
    """All three former helpers now answer exactly like ``pid_alive``.

    Real processes again: the observable contract is identical for a
    live child, a reaped child, the non-positive guards, and (when
    reachable) the foreign-owned pid 1.
    """
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        live = proc.pid
        for probe in (
            GitWorktreeOps._pid_alive,
            web_use_tool._pid_alive,
            web_server._is_pid_alive,
        ):
            assert probe(live) is True
            assert probe(0) is False
            assert probe(-1) is False
    finally:
        proc.kill()
        proc.wait()
    for probe in (
        GitWorktreeOps._pid_alive,
        web_use_tool._pid_alive,
        web_server._is_pid_alive,
    ):
        assert probe(proc.pid) is False
    if os.geteuid() != 0:
        for probe in (
            GitWorktreeOps._pid_alive,
            web_use_tool._pid_alive,
            web_server._is_pid_alive,
        ):
            assert probe(1) is True
