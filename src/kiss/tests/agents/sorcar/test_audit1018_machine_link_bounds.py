# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: every wait of a machine link is bounded (audit 2026-10-18).

Against a "black hole" listener (accepts the TCP connection and never
answers, as a host that drops packets does), and a real remote daemon:

* a request's wait for the link's turn is bounded by its own timeout:
  one queued behind a slow request fails ``busy`` when its budget is up
  and starts no back-off;
* the handshake of a long request (an app-probe poll's 53 s) has
  ``REQUEST_TIMEOUT`` at most, so its thread is not held for the whole
  budget by a dial that cannot succeed;
* a machine found down on one link lane is reported down on the others
  without being dialled again;
* ``retire()`` does not wait for a request in flight: the request's
  read is woken and fails at once;
* ``registry_lock`` gives up on a held lock after its timeout with a
  ``MachineError`` (the callers' error text) instead of waiting forever.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import machines as mach
from kiss.core.file_lock import exclusive_file_lock
from kiss.tests.local_ws import make_test_tls
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon


@pytest.fixture(autouse=True)
def _empty_registry() -> Iterator[None]:
    """Every test starts and ends with no registered machine and no cached link."""
    mach.save_machines([])
    yield
    mach.save_machines([])


@pytest.fixture
def black_hole(tmp_path: Path) -> Iterator[mach.Machine]:
    """A registered-looking machine whose port accepts connections and never answers."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    accepted: list[socket.socket] = []
    stop = threading.Event()

    def accept_forever() -> None:
        listener.settimeout(0.2)
        while not stop.is_set():
            try:
                accepted.append(listener.accept()[0])
            except TimeoutError:
                continue

    thread = threading.Thread(target=accept_forever, daemon=True)
    thread.start()
    machine = mach.Machine(
        name="hole", url=f"https://127.0.0.1:{listener.getsockname()[1]}", password="pw",
    )
    cert = machine.cert_file()
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_bytes(make_test_tls(tmp_path)[0].read_bytes())
    yield machine
    stop.set()
    thread.join(5)
    for conn in accepted:
        conn.close()
    listener.close()
    cert.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def remote(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RemoteDaemon]:
    """A password-protected daemon with its own local CA."""
    daemon = start_remote_daemon(tmp_path_factory.mktemp("remote-home"), password="secret-1")
    yield daemon
    daemon.stop()


def test_turn_wait_and_handshake_are_bounded_by_the_timeouts(black_hole: mach.Machine) -> None:
    """A queued request fails ``busy`` within its timeout; a long one dials for REQUEST_TIMEOUT."""
    link = mach.link_for(black_hole)
    outcome: dict[str, object] = {}

    def slow_request() -> None:
        started = time.monotonic()
        try:
            link.request({"type": "ping"}, "pong", timeout=mach.REQUEST_TIMEOUT + 45)
        except mach.MachineError as exc:
            outcome["error"] = str(exc)
        outcome["elapsed"] = time.monotonic() - started

    slow = threading.Thread(target=slow_request, daemon=True)
    slow.start()
    time.sleep(0.3)
    assert link._lock.acquire(timeout=0) is False, "the slow request holds the link"
    started = time.monotonic()
    with pytest.raises(mach.MachineError, match="is busy: no pong within 0.5s"):
        link.request({"type": "ping"}, "pong", timeout=0.5)
    assert time.monotonic() - started < 2.0
    # A busy link is not a down machine: no back-off was started.
    assert link._retry_at == 0.0
    slow.join(mach.REQUEST_TIMEOUT + 10)
    assert not slow.is_alive(), "the slow request's handshake outlived REQUEST_TIMEOUT"
    assert "unreachable" in str(outcome["error"])
    elapsed = outcome["elapsed"]
    assert isinstance(elapsed, float)
    assert mach.REQUEST_TIMEOUT - 1 < elapsed < mach.REQUEST_TIMEOUT + 5


def test_lanes_share_the_back_off(black_hole: mach.Machine) -> None:
    """A machine found down on one lane is reported down on another without a dial."""
    default = mach.link_for(black_hole)
    with pytest.raises(mach.MachineError, match="unreachable") as first:
        default.request({"type": "ping"}, "pong", timeout=1.0)
    assert default._retry_at > time.monotonic()
    started = time.monotonic()
    panels = mach.link_for(black_hole, "panels")
    assert panels is not default
    with pytest.raises(mach.MachineError) as again:
        panels.request({"type": "ping"}, "pong", timeout=1.0)
    assert time.monotonic() - started < 0.5
    assert str(again.value) == str(first.value)
    # The back-off is one record: clearing it on either lane dials again.
    panels._health.retry_at = 0.0
    assert default._retry_at == 0.0
    # A lane made after the links were retired starts afresh.
    mach.close_links()
    assert mach.link_for(black_hole, "panels")._health is not panels._health


def test_retire_does_not_wait_for_a_request_in_flight(remote: RemoteDaemon) -> None:
    """Retiring a link with a request in flight returns at once; the request fails at once."""
    machine = mach.add_machine("box", remote.url, "secret-1")
    link = mach.link_for(machine)
    assert link.request({"type": "ping"}, "pong")["type"] == "pong"
    outcome: dict[str, object] = {}

    def waiting_request() -> None:
        started = time.monotonic()
        try:
            link.request({"type": "ping"}, "never-sent", timeout=20.0)
        except mach.MachineError as exc:
            outcome["error"] = str(exc)
        outcome["elapsed"] = time.monotonic() - started

    waiting = threading.Thread(target=waiting_request, daemon=True)
    waiting.start()
    time.sleep(0.5)
    started = time.monotonic()
    link.retire()
    assert time.monotonic() - started < 5.0
    waiting.join(5.0)
    assert not waiting.is_alive(), "the request in flight was not woken by the retire"
    assert "registry changed" in str(outcome["error"])
    with pytest.raises(mach.MachineError, match="registry changed"):
        link.request({"type": "ping"}, "pong")


def test_registry_lock_gives_up_on_a_held_lock() -> None:
    """A held registry lock makes ``registry_lock`` fail after its timeout, not wait forever."""
    lock_path = mach.machines_file().with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(lock_path):
        started = time.monotonic()
        with pytest.raises(mach.MachineError, match="registry is busy"):
            with mach.registry_lock(timeout=0.3):
                raise AssertionError("the lock was taken while held elsewhere")
        assert 0.2 < time.monotonic() - started < 3.0
    with mach.registry_lock(timeout=0.3):
        pass
