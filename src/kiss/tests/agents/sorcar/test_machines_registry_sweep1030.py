# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Registry edits leave nothing behind: no staged pin, no link or forward for a removed machine.

Audit 2026-10-30 (S9).

``add_machine`` stages the machine's certificate in a hidden
``.<name>.pem.<pid>.<hex>.new`` beside the pins and removed it only when
the handshake raised ``MachineError``.  A Stop (an injected
``KeyboardInterrupt``: the ``/machine add`` tool runs on the task thread)
during the handshake, or any error after it (the registry lock busy for
a minute), left the staged file in ``$KISS_HOME/machines/`` for good.

A :class:`Machine` read just before a ``/machine remove`` (a CHATS
refresh's pool thread, a ``/machine list`` status check) re-created
the removed machine's cached link, with an authenticated socket, and
its ssh port-forward after the removal's sweep (:func:`close_links`)
had run, and nothing closed them until the next registry write or the
daemon's exit.  ``link_for`` and ``tunnel_port`` now refuse a machine
(an ssh login) the registry no longer holds; the registry is read after
the link (tunnel) is cached and the sweep runs after the registry is
written, so a removal either shows in that read or sweeps what was
cached.

The windows are landed deterministically with :class:`Pause`
(instrumentation that holds a thread at a function's entry; nothing is
replaced) against a REAL remote daemon in its own process.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from types import CodeType
from typing import Any
from urllib.parse import urlsplit

import pytest

from kiss.agents.sorcar import machines as mach
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.agents.sorcar.test_machines import _install_fake_ssh, fake_ssh_only
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon
from kiss.tests.server.pause import Pause


@pytest.fixture(scope="module")
def remote(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RemoteDaemon]:
    """A password-protected daemon in its own process."""
    daemon = start_remote_daemon(tmp_path_factory.mktemp("remote-home"), password="secret-1")
    yield daemon
    daemon.stop()


@pytest.fixture(autouse=True)
def _empty_registry() -> Iterator[None]:
    """Every test starts and ends with no registered machine, link or forward."""
    mach.save_machines([])
    yield
    mach.save_machines([])


def _on_thread(name: str) -> Any:
    """Return a :class:`Pause` matcher that holds only the thread called *name*."""
    return lambda _local: threading.current_thread().name == name


def _run(outcome: dict[str, object], fn: Any, *args: Any) -> None:
    """Call *fn*, recording its result or the exception it ended with."""
    try:
        outcome["result"] = fn(*args)
    except BaseException as exc:  # noqa: BLE001 - the interrupt is the outcome
        outcome["error"] = exc


def _staged(name: str) -> list[Path]:
    """Return the staged certificates of machine *name* still on disk."""
    return sorted((mach.kiss_home() / mach.MACHINE_CERTS_DIR).glob(f".{name}.pem.*.new"))


@pytest.mark.parametrize(
    "where",
    [mach._first_handshake.__code__, mach.registry_lock.__wrapped__.__code__],  # type: ignore[attr-defined]
    ids=["during the handshake", "waiting for the registry lock"],
)
def test_a_stop_during_add_machine_leaves_no_staged_pin(
    remote: RemoteDaemon, where: CodeType,
) -> None:
    outcome: dict[str, object] = {}
    adder = threading.Thread(
        target=_run, args=(outcome, mach.add_machine, "stagedbox", remote.url, "secret-1"),
        name="adder",
    )
    with Pause(where, _on_thread("adder")) as held:
        adder.start()
        assert held.paused.wait(30), "add_machine never got there"
        assert len(_staged("stagedbox")) == 1, "the certificate is staged by now"
        assert adder.ident is not None
        assert inject_keyboard_interrupt(adder.ident) == 1
    adder.join(30)
    assert not adder.is_alive()
    assert isinstance(outcome.get("error"), KeyboardInterrupt), outcome
    assert _staged("stagedbox") == []
    assert mach.find_machine("stagedbox") is None
    assert not mach.Machine("stagedbox", remote.url).cert_file().exists()


def test_a_machine_read_before_its_removal_gets_no_link_after_the_sweep(
    remote: RemoteDaemon,
) -> None:
    machine = mach.add_machine("stalebox", remote.url, "secret-1")
    mach.link_for(machine).request({"type": "ping"}, "pong")
    outcome: dict[str, object] = {}
    remover = threading.Thread(
        target=_run, args=(outcome, mach.remove_machine, "stalebox"), name="remover",
    )
    # remove_machine's first cert_file() call is the pin's unlink, after
    # the registry write and its sweep: the audit's window.
    with Pause(mach.Machine.cert_file.__code__, _on_thread("remover")) as held:
        remover.start()
        assert held.paused.wait(30), "remove_machine never reached the pin's unlink"
        assert mach.load_machines() == []
        assert machine.cert_file().is_file(), "the pin is still there"
        link = mach.link_for(machine)
        with pytest.raises(mach.MachineError, match="the machine registry changed, try again"):
            link.request({"type": "ping"}, "pong")
        assert link._ws is None
        assert [key for key in mach._links if key[0] == "stalebox"] == []
    remover.join(30)
    assert not remover.is_alive()
    assert outcome == {"result": True}
    assert not machine.cert_file().exists()


@fake_ssh_only
def test_a_status_check_of_a_removed_ssh_machine_opens_no_forward(
    remote: RemoteDaemon, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ssh(tmp_path, monkeypatch, tmp_path / "far-home")
    pidfile = tmp_path / "forward.pid"
    monkeypatch.setenv("FAKE_SSH_PIDFILE", str(pidfile))
    far = mach.Machine(
        "far", f"https://far.example.org:{urlsplit(remote.url).port}", ssh="ksen@far.example.org",
    )
    far.cert_file().parent.mkdir(parents=True, exist_ok=True)
    far.cert_file().write_text(mach.pin_certificate(remote.url), encoding="utf-8")
    try:
        mach.save_machines([far])
        # Removed (another thread's /machine remove) after `far` was read.
        mach.save_machines([])
        status = mach.machine_status(far)
        assert status.endswith(
            "is unreachable: ssh ksen@far.example.org: the machine registry changed, try again"
        ), status
        assert far.ssh not in mach._tunnels
        assert not pidfile.exists(), "an ssh forward was started for the removed login"
    finally:
        mach.close_tunnels()
        far.cert_file().unlink(missing_ok=True)
