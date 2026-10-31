# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: machine links after the host slept, and the growing back-off.

A laptop back from sleep found its registered machines "unreachable"
for a minute or two: every cached socket to a remote daemon was dead
(the daemon closed it when its pings went unanswered), a request hung
on it for its whole budget, the failure backed the machine off for a
flat 30 s, and the MACHINES subpanel cached the red rows for 60 s.

* a first failure backs the machine off for ``FIRST_BACKOFF`` seconds,
  doubling with every failure in a row up to ``DOWN_BACKOFF``; a success
  starts the ladder over; a rejected password waits ``DOWN_BACKOFF``
  from the first failure;
* a sleep (the wall clock ran ahead of the monotonic one by
  ``SLEEP_GAP`` or more) is noticed by the next request and by the
  watchdog's ``forget_after_sleep``: the cached socket is shut down,
  the back-off a poll during the sleep left is cleared, and the request
  answers over a fresh socket;
* an ssh forward is closed by the wake (its ssh process may sit on a
  dead connection for 45 s) and opened again on the next request, while
  one being opened at that moment is left alone.

The sleep itself is staged by moving the detector's last wall-clock
stamp back: the host cannot be put to sleep from a test.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection

from kiss.agents.sorcar import machines as mach
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon
from kiss.tests.ssh_server import SSHD, SshServer, start_sshd

needs_sshd = pytest.mark.skipif(not SSHD, reason="no sshd on this host")


@pytest.fixture(autouse=True)
def _empty_registry() -> Iterator[None]:
    """Every test starts and ends with no registered machine, no cached link, no sleep."""
    mach.save_machines([])
    mach.forget_after_sleep()
    yield
    mach.save_machines([])


@pytest.fixture(scope="module")
def remote(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RemoteDaemon]:
    """A password-protected daemon with its own local CA."""
    daemon = start_remote_daemon(
        tmp_path_factory.mktemp("remote-home"), password="secret-1",
        seed=[{"chat_id": "r", "task": "remote alpha"}],
    )
    yield daemon
    daemon.stop()


@pytest.fixture(scope="module")
def sshd(remote: RemoteDaemon, tmp_path_factory: pytest.TempPathFactory) -> Iterator[SshServer]:
    """An sshd whose logins see the password-protected daemon's home as KISS_HOME."""
    if not SSHD:
        pytest.skip("no sshd on this host")
    server = start_sshd(tmp_path_factory.mktemp("sshd"), remote.home)
    env = pytest.MonkeyPatch()
    env.setenv(mach.SSH_CONFIG_ENV, str(server.config))
    yield server
    env.undo()
    server.stop()


def _closed_port() -> int:
    """Return a loopback port nothing listens on (a dial is refused at once)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stage_sleep(seconds: float) -> None:
    """Make the next ``forget_after_sleep`` see the wall clock *seconds* ahead of the monotonic."""
    wall, mono = mach._clocks
    mach._clocks = (wall - seconds, mono)


def _assert_closed(ws: ClientConnection) -> None:
    """Fail unless *ws* is closed (frames the daemon pushed before the close are read past)."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            ws.recv(timeout=1.0)
        except ConnectionClosed:
            return
    raise AssertionError("the socket is still open")


def _back_off(link: mach.MachineLink) -> float:
    """Seconds *link* waits before dialling again, as of now."""
    return link._health.retry_at - time.monotonic()


def _fail_once(link: mach.MachineLink) -> str:
    """Make one request fail on *link* (its back-off cleared first) and return the error."""
    link._health.retry_at = 0.0
    with pytest.raises(mach.MachineError) as failure:
        link.request({"type": "ping"}, "pong", timeout=2.0)
    return str(failure.value)


def test_the_back_off_grows_with_the_failures_in_a_row(remote: RemoteDaemon) -> None:
    """5, 10, 20, 30, 30 s after failures in a row; a success starts over; a bad password: 30 s."""
    machine = mach.add_machine("box", remote.url, "secret-1")
    gone = mach.Machine("gone", f"https://127.0.0.1:{_closed_port()}", password="pw")
    gone.cert_file().write_bytes(machine.cert_file().read_bytes())
    mach.save_machines([machine, gone])
    link = mach.link_for(gone)
    waits = []
    for _ in range(5):
        assert "unreachable" in _fail_once(link)
        waits.append(round(_back_off(link)))
    assert waits == [5, 10, 20, 30, 30]
    assert link._health.failures == 5
    # A machine down for days: the ladder stays at the top (the
    # doubling is capped, so the count never overflows the power).
    for _ in range(1030):
        link._fail("still down")
    assert link._health.failures == 1035
    assert round(_back_off(link)) == 30
    assert "unreachable" in _fail_once(link)
    assert round(_back_off(link)) == 30
    # The other lanes share the ladder: their failure counts here too.
    assert "unreachable" in _fail_once(mach.link_for(gone, "panels"))
    assert link._health.failures == 1037

    # A machine that answers starts its ladder over.
    box = mach.link_for(machine)
    # The request's timeout also bounds the connect: 0.3 s let a TLS
    # handshake on a loaded full-suite host time out ("unreachable").
    with pytest.raises(mach.MachineError, match="did not answer"):
        box.request({"type": "ping"}, "never-sent", timeout=2.0)
    assert round(_back_off(box)) == 5 and box._health.failures == 1
    box._health.retry_at = 0.0
    assert box.request({"type": "ping"}, "pong")["type"] == "pong"
    assert box._health.failures == 0

    # A rejected password is not tried again sooner: the remote daemon
    # locks a client out after a few wrong passwords in a row.
    wrong = mach.Machine("box", remote.url, password="not-it")
    mach.save_machines([wrong])
    assert "rejected the password" in _fail_once(mach.link_for(wrong))
    assert round(_back_off(mach.link_for(wrong))) == 30


def test_a_sleep_drops_the_cached_socket_and_the_back_off(remote: RemoteDaemon) -> None:
    """The first request after a sleep answers over a fresh socket, back-off or not."""
    machine = mach.add_machine("box", remote.url, "secret-1")
    link = mach.link_for(machine)
    assert link.request({"type": "ping"}, "pong")["type"] == "pong"
    kept = link._ws
    assert kept is not None
    # A poll during the sleep failed and backed the machine off on
    # another lane (the record is shared).
    panels = mach.link_for(machine, "panels")
    with pytest.raises(mach.MachineError, match="did not answer"):
        panels.request({"type": "ping"}, "never-sent", timeout=0.3)
    assert _back_off(link) > 0

    # Less than SLEEP_GAP is no sleep: the socket and the back-off stay.
    _stage_sleep(mach.SLEEP_GAP - 1)
    assert mach.forget_after_sleep() == 0.0
    assert link._ws is kept and _back_off(link) > 0

    _stage_sleep(100.0)
    started = time.monotonic()
    assert link.request({"type": "ping"}, "pong")["type"] == "pong"
    assert time.monotonic() - started < mach.REQUEST_TIMEOUT
    assert link._ws is not None and link._ws is not kept
    _assert_closed(kept)
    assert link._health.failures == 0 and _back_off(link) <= 0
    # The next check sees no new sleep.
    assert mach.forget_after_sleep() == 0.0


def test_the_watchdog_check_abandons_the_sockets_before_a_request(remote: RemoteDaemon) -> None:
    """``forget_after_sleep`` (the watchdog's call) shuts every cached socket down at once."""
    machine = mach.add_machine("box", remote.url, "secret-1")
    links = [mach.link_for(machine), mach.link_for(machine, "panels")]
    sockets = []
    for link in links:
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        sockets.append(link._ws)
    _stage_sleep(100.0)
    started = time.monotonic()
    assert mach.forget_after_sleep() > 90.0
    # No close handshake was waited for on either socket.
    assert time.monotonic() - started < 1.0
    for ws in sockets:
        assert ws is not None
        _assert_closed(ws)
    for link in links:
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        assert link._ws not in sockets


@needs_sshd
def test_a_sleep_closes_the_ssh_forward_unless_it_is_being_opened(
    remote: RemoteDaemon, sshd: SshServer,
) -> None:
    """The wake kills the kept ssh; the next request opens a new one; a login in progress stays."""
    machine = mach.add_machine("near", sshd.login)
    assert mach.machine_status(machine) == "reachable"
    tunnel = mach._tunnels[sshd.login]
    assert tunnel.process is not None and tunnel.alive()
    pid = tunnel.process.pid

    _stage_sleep(100.0)
    assert mach.forget_after_sleep() > 90.0
    assert tunnel.process is None and not tunnel.alive()
    assert mach._tunnels[sshd.login] is tunnel  # kept in the cache, opened again on use
    assert mach.machine_status(machine) == "reachable"
    assert tunnel.process is not None and tunnel.process.pid != pid and tunnel.alive()

    # A forward being opened when the wake is noticed is fresh: the wake
    # does not wait for the login and does not kill its ssh.
    tunnel.close()
    opening = threading.Event()
    opened: list[int] = []

    def open_slowly() -> None:
        def body() -> int:
            opening.set()
            time.sleep(1.5)
            return tunnel._open_locked(mach.SSH_TIMEOUT)

        opened.append(mach.locked_call(tunnel._lock, body))

    opener = threading.Thread(target=open_slowly, daemon=True)
    opener.start()
    assert opening.wait(5.0)
    _stage_sleep(100.0)
    started = time.monotonic()
    assert mach.forget_after_sleep() > 90.0
    assert time.monotonic() - started < 1.0, "the wake waited for the login"
    opener.join(mach.SSH_TIMEOUT + 5)
    assert not opener.is_alive()
    assert tunnel.alive() and tunnel.local_port == opened[0]
    mach.close_links()
    assert mach._tunnels == {}
