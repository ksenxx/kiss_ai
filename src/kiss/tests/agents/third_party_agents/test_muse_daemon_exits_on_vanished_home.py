# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The Muse-auth daemon exits once the ``KISS_HOME`` it serves is deleted.

Bug report: tasks that ran tests or servers with ``KISS_HOME`` pointing
inside their worktree (``<worktree>/tmp/khome``) left a detached
``muse_auth.daemon`` behind.  Every 300 s its serve loop swept the vault
scratch files, and ``Vault._vault_dir`` re-created
``<KISS_HOME>/muse_auth/vault`` — inside the worktree directory that had
already been removed, resurrecting it as an unregistered husk under
``.kiss-worktrees/``.

The daemon now checks its state directory on every pass of the accept
loop (0.5 s) and stops when it is gone.  This test spawns the real daemon
through ``ensure_daemon``, deletes the whole ``KISS_HOME`` and verifies
the daemon process exits and nothing re-creates the directory.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from kiss.agents.third_party_agents.muse_auth._common import muse_auth_dir, socket_path
from kiss.agents.third_party_agents.muse_auth.client import enrolled_services, ensure_daemon
from kiss.tests.agents.third_party_agents.muse_test_utils import (
    _daemon_pid,
    _process_running,
    setup_muse_env,
    teardown_muse_env,
)


def test_daemon_exits_when_its_home_is_deleted(
    isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_muse_env(monkeypatch, {"defaults": {"read": "allow", "write": "ask"}})
    try:
        ensure_daemon()
        pid = _daemon_pid()
        assert pid is not None, "no daemon is listening after ensure_daemon()"
        assert muse_auth_dir().is_dir()

        shutil.rmtree(isolated_kiss_home)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _process_running(pid):
            time.sleep(0.05)
        assert not _process_running(pid), (
            "the daemon kept running after its KISS_HOME was deleted"
        )
        assert not socket_path().exists(), "the daemon left its socket behind"
        # Long enough for a periodic sweep in the (now gone) loop to have
        # run had the daemon survived the accept timeout.
        time.sleep(0.6)
        assert not isolated_kiss_home.exists(), (
            "something re-created the deleted KISS_HOME"
        )
    finally:
        teardown_muse_env()


def test_client_recovers_right_after_home_is_recreated(
    isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delete ``KISS_HOME``, recreate it, and use the client at once.

    With a long ``KISS_HOME`` path the socket lives in ``/tmp`` and
    survives the deletion, so ``ensure_daemon`` can still reach the old
    daemon during the half-second before it notices its state directory
    is gone.  ``ensure_daemon`` restores the directory before reusing
    any daemon, so the old one either keeps serving (it sees the
    directory again) or has already left and a fresh one is spawned —
    either way the call succeeds instead of dying under the request.
    """
    setup_muse_env(monkeypatch, {"defaults": {"read": "allow", "write": "ask"}})
    try:
        ensure_daemon()
        for _ in range(5):
            shutil.rmtree(isolated_kiss_home)
            isolated_kiss_home.mkdir()
            assert enrolled_services() == []
            assert muse_auth_dir().is_dir(), "ensure_daemon did not restore the state dir"
    finally:
        teardown_muse_env()
