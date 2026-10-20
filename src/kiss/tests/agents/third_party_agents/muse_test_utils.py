# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared Muse-auth helpers for the ``test_muse_*`` modules.

Each Muse test module declares a ``MUSE_POLICY`` (built with
``loopback_policy``: it documents which services get loopback
``extra_hosts`` entries) that the shared ``muse_env`` fixture in
``conftest.py`` installs through the helpers here:
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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents.muse_auth._common import (
    muse_auth_dir,
    platform_supports_muse_daemon,
    socket_path,
)
from kiss.agents.third_party_agents.muse_auth.client import stop_daemon
from kiss.tests.agents.third_party_agents.recording_http import RecordingHandler

_DAEMON_SKIP_REASON = "Linux-only: the Muse-auth daemon authenticates clients with SO_PEERCRED"

requires_muse_daemon = pytest.mark.skipif(
    not platform_supports_muse_daemon(),
    reason=_DAEMON_SKIP_REASON,
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
    is therefore skipped here, the one place the shared ``muse_env``
    fixture passes through; without the skip each test waited ~40 s for
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


def teardown_muse_env(timeout: float = 10.0) -> None:
    """Stop the Muse-auth daemon and wait until its process has exited.

    ``stop_daemon`` only asks the daemon to exit.  The daemon unlinks its
    socket as its serve loop ends and then still needs tens of
    milliseconds (more under coverage) to shut its interpreter down, so
    waiting for the socket alone let the subprocess reaper find the
    daemon alive at test teardown and ``SIGTERM`` it.  The daemon's pid
    is therefore read before the stop and awaited afterwards; waiting
    for the socket to vanish also keeps a follow-up ``ensure_daemon``
    from handshaking with the dying daemon.

    A daemon that is still there when *timeout* runs out is reported as
    a failure of the test that leaked it: silently returning would let
    the next daemon start on the same socket path while the old one is
    alive, and the subprocess reaper would blame the next test.

    Args:
        timeout: Seconds to wait for the socket and the process to go.

    Raises:
        AssertionError: When the socket or the daemon process outlives
            *timeout*.
    """
    pid = _daemon_pid()
    stop_daemon()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and (
        socket_path().exists() or (pid is not None and _process_running(pid))
    ):
        time.sleep(0.02)
    assert not socket_path().exists(), f"Muse daemon socket still present after {timeout}s"
    assert pid is None or not _process_running(pid), (
        f"Muse daemon pid={pid} still running after {timeout}s"
    )


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


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll ``predicate`` every 50 ms until it is true or ``timeout`` seconds elapse.

    Args:
        predicate: Condition to wait for.
        timeout: Seconds to keep polling.

    Returns:
        True as soon as the predicate holds (it is checked once more after
        the deadline, so a condition met during the last sleep counts),
        False when the timeout expires first.
    """
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


class MuseApiHandler(RecordingHandler):
    """Base for the loopback API emulators the Muse-auth daemon is pointed at.

    Every request is recorded as ``{"method", "path", "headers", "body"}``
    (the full header dict, so a test can prove which credential crossed
    the wire); :func:`bearer_of` reads the credential back out of a
    recorded entry.
    """

    server: Any

    def record_full_request(self) -> bytes:
        """Read the body, append this request to ``server.requests`` and return the raw body."""
        body = self._read_raw()
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": body.decode("utf-8", errors="replace"),
            }
        )
        return body

    def authorization(self) -> str:
        """Return the request's ``Authorization`` header (``""`` when absent)."""
        return self.headers.get("Authorization", "")

    def redirect(self, status: int, location: str) -> None:
        """Send an empty-bodied *status* redirect to *location*."""
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


def bearer_of(request: dict[str, Any]) -> str:
    """Return the ``Authorization`` header of a request recorded by :class:`MuseApiHandler`.

    Args:
        request: One entry of ``server.requests``.

    Returns:
        The header value, or ``""`` when the request carried none.
    """
    return next((v for k, v in request["headers"].items() if k.lower() == "authorization"), "")


def auth_tools(agent: Any) -> dict[str, Any]:
    """Return ``agent``'s auth tools keyed by function name.

    Args:
        agent: Any channel agent exposing ``_get_auth_tools``.

    Returns:
        A mapping from tool function name to the tool callable.
    """
    return {tool.__name__: tool for tool in agent._get_auth_tools()}


def tools_by_name(agent: Any) -> dict[str, Any]:
    """Return ``agent``'s tools (``_get_tools()``) keyed by function name.

    Args:
        agent: Any agent exposing ``_get_tools``.

    Returns:
        A mapping from tool function name to the tool callable.
    """
    return {tool.__name__: tool for tool in agent._get_tools()}


def loopback_policy(*services: str) -> dict[str, Any]:
    """Return the Muse-auth policy a test module's ``MUSE_POLICY`` is built from.

    Reads are allowed, writes ask the user, and every *service* gets an
    ``extra_hosts`` entry for ``127.0.0.1`` so a loopback emulator of
    that service (whose real host is a fixed cloud endpoint) passes the
    daemon's host allow-list.  Services left out are origin-bound at
    enrollment, which their tests then prove.

    Args:
        services: Muse-auth service names whose hosts include loopback.

    Returns:
        The policy document the shared ``muse_env`` fixture installs.
    """
    return {
        "defaults": {"read": "allow", "write": "ask"},
        "services": {service: {"extra_hosts": ["127.0.0.1"]} for service in services},
    }


def set_policy(service: str, **fields: Any) -> None:
    """Merge *fields* into *service*'s entry of the installed ``policy.json``.

    Sentinel reloads the policy per decision, so a test flips a service
    rule (``read="ask"``) or adds a loopback host after the daemon is up.

    Args:
        service: Muse-auth service name (added when not in the policy).
        fields: Policy keys to set on that service.
    """
    policy_path = muse_auth_dir() / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy["services"].setdefault(service, {}).update(fields)
    policy_path.write_text(json.dumps(policy))
