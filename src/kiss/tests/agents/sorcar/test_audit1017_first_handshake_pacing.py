"""``_first_handshake`` retries a failed enrollment ping once a second, not in a tight loop.

The 2026-10-16 restructuring of :func:`kiss.agents.sorcar.machines._first_handshake`
moved its ``time.sleep(1.0)`` out of the retry loop into the function's outer
``finally``: a daemon that did not answer yet (its trusted-client watcher
polls every 2 s after an enrollment) was then hammered with TLS handshakes
until ``ENROLL_TIMEOUT``, and every successful registration paid a pointless
one-second sleep on return.  These tests pin the documented pacing with a
real listener and a real daemon; nothing is mocked.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import machines as mach
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon


def _serve_closing(listener: socket.socket, accepted: list[float], stop: threading.Event) -> None:
    """Accept connections on *listener* and close each at once, logging the time."""
    while not stop.is_set():
        try:
            conn, _ = listener.accept()
        except TimeoutError:  # the 0.2 s accept timeout: look at *stop* again
            continue
        except OSError:  # the listener was closed
            return
        accepted.append(time.monotonic())
        conn.close()


@pytest.fixture
def daemon_ca(tmp_path: Path) -> Iterator[tuple[RemoteDaemon, Path]]:
    """A real password-protected daemon; yields ``(daemon, its ca.pem)``."""
    daemon = start_remote_daemon(tmp_path / "home", password="secret-1")
    try:
        yield daemon, daemon.home / "tls" / "ca.pem"
    finally:
        daemon.stop()


def test_failed_handshakes_are_retried_once_a_second(daemon_ca: tuple[RemoteDaemon, Path]) -> None:
    """A peer that drops every connection is dialled about once a second until the deadline."""
    _, ca_pem = daemon_ca
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.settimeout(0.2)
    accepted: list[float] = []
    stop = threading.Event()
    server = threading.Thread(target=_serve_closing, args=(listener, accepted, stop), daemon=True)
    server.start()
    machine = mach.Machine("dropper", f"https://127.0.0.1:{listener.getsockname()[1]}", "secret-1")
    budget = 2.5
    started = time.monotonic()
    try:
        with pytest.raises(mach.MachineError):
            mach._first_handshake(machine, ca_pem, started + budget)
    finally:
        stop.set()
        listener.close()
        server.join(timeout=5.0)
    elapsed = time.monotonic() - started
    assert elapsed >= budget, f"gave up after {elapsed:.2f}s, before the {budget}s deadline"
    # Attempts at ~0 s, ~1 s, ~2 s and the one that crosses the deadline:
    # a tight loop makes dozens in the same budget.
    assert 2 <= len(accepted) <= 4, f"{len(accepted)} dials in {elapsed:.2f}s: {accepted}"
    gaps = [b - a for a, b in zip(accepted, accepted[1:], strict=False)]
    assert all(gap >= 0.9 for gap in gaps), f"dials closer than a second apart: {gaps}"


def test_successful_handshake_returns_without_raising(daemon_ca: tuple[RemoteDaemon, Path]) -> None:
    """A daemon that answers the ping is accepted on the first attempt."""
    daemon, ca_pem = daemon_ca
    machine = mach.Machine("box", daemon.url, "secret-1")
    mach._first_handshake(machine, ca_pem, time.monotonic() + 30.0)
