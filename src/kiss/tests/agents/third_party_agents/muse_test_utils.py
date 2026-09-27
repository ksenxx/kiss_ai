# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared Muse-auth helpers for the ``test_muse_*`` modules.

Each Muse test module keeps its own tiny ``muse_env`` fixture (the
per-module policy documents which services get loopback ``extra_hosts``
entries) but delegates the common setup and teardown here:
``setup_muse_env`` enables Muse-auth and writes the policy file inside
the isolated ``KISS_HOME``, and ``teardown_muse_env`` stops the daemon
subprocess and waits for its socket to vanish and its process to exit so
the next test (or the next daemon in the same test) starts from a clean
slate and the subprocess reaper finds nothing left running.
"""

from __future__ import annotations

import contextlib
import json
import socket
import struct
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents.muse_auth._common import (
    muse_auth_dir,
    platform_supports_muse_daemon,
    socket_path,
)
from kiss.agents.third_party_agents.muse_auth.client import stop_daemon

_DAEMON_SKIP_REASON = "Linux-only: the Muse-auth daemon authenticates clients with SO_PEERCRED"

requires_muse_daemon = pytest.mark.skipif(
    not platform_supports_muse_daemon(), reason=_DAEMON_SKIP_REASON,
)
"""Skip marker for tests that spawn the real daemon outside ``muse_env``."""


def setup_muse_env(monkeypatch: pytest.MonkeyPatch, policy: dict[str, Any]) -> None:
    """Enable Muse-auth and install ``policy`` inside the current ``KISS_HOME``.

    Sets ``KISS_MUSE_AUTH=1`` via ``monkeypatch`` (so it is undone on
    teardown) and writes ``policy`` as ``policy.json`` in the Muse-auth
    directory, creating the directory first.  Callers must already have
    pointed ``KISS_HOME`` at an isolated location (the
    ``isolated_kiss_home`` fixture).

    The daemon's only transport is a Unix-domain socket authenticated
    with ``SO_PEERCRED``, which exists on Linux only (macOS has
    ``LOCAL_PEERCRED`` instead, Windows has no ``AF_UNIX`` at all), so
    the product deliberately does not run it elsewhere
    (:func:`platform_supports_muse_daemon`).  Every daemon-backed test
    is therefore skipped here, in the one place all six ``muse_env``
    fixtures pass through; without the skip each one waited ~40 s for
    a daemon that cannot start.

    Args:
        monkeypatch: The test's monkeypatch, used for the env var.
        policy: The Muse-auth policy document to write.
    """
    if not platform_supports_muse_daemon():
        pytest.skip(_DAEMON_SKIP_REASON)
    monkeypatch.setenv("KISS_MUSE_AUTH", "1")
    directory = muse_auth_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "policy.json").write_text(json.dumps(policy))


def teardown_muse_env() -> None:
    """Stop the Muse-auth daemon and wait until its process has exited.

    ``stop_daemon`` only asks the daemon to exit.  The daemon unlinks its
    socket as its serve loop ends and then still needs tens of
    milliseconds (more under coverage) to shut its interpreter down, so
    waiting for the socket alone let the subprocess reaper find the
    daemon alive at test teardown and ``SIGTERM`` it.  The daemon's pid
    is therefore read before the stop and awaited afterwards; waiting
    for the socket to vanish also keeps a follow-up ``ensure_daemon``
    from handshaking with the dying daemon.
    """
    pid = _daemon_pid()
    stop_daemon()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and (
        socket_path().exists() or (pid is not None and _process_running(pid))
    ):
        time.sleep(0.02)


def _daemon_pid() -> int | None:
    """Return the pid of the daemon listening on the Muse-auth socket.

    Returns:
        The listener's pid from the kernel's ``SO_PEERCRED`` record, or
        None when no daemon accepts connections.
    """
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        # Bounded: a listener with a full backlog must not stall teardown.
        sock.settimeout(5.0)
        try:
            sock.connect(str(socket_path()))
        except OSError:
            return None
        peercred = getattr(socket, "SO_PEERCRED")  # Linux-only constant
        creds = sock.getsockopt(socket.SOL_SOCKET, peercred, struct.calcsize("3i"))
    pid: int = struct.unpack("3i", creds)[0]
    return pid


def _process_running(pid: int) -> bool:
    """Return whether ``pid`` exists and has not exited (a zombie has exited).

    Args:
        pid: Process id to check.

    Returns:
        True while the process is running.
    """
    with contextlib.suppress(OSError):
        stat = Path(f"/proc/{pid}/stat").read_text()
        # The state letter follows the parenthesised command name.
        return stat[stat.rindex(")") + 2] != "Z"
    return False


def auth_tools(agent: Any) -> dict[str, Any]:
    """Return ``agent``'s auth tools keyed by function name.

    Args:
        agent: Any channel agent exposing ``_get_auth_tools``.

    Returns:
        A mapping from tool function name to the tool callable.
    """
    return {tool.__name__: tool for tool in agent._get_auth_tools()}
