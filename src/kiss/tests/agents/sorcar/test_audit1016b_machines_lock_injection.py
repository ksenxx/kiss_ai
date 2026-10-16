# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``machines``' locks survive the runner's Stop injection on a task thread.

The ``/machine`` SEA's tools run on the agent's task thread:
``add_machine`` and ``machine_status`` take a :class:`MachineLink`'s
lock (``request``, ``close``), ``add_machine`` / ``remove_machine``
write the registry, which retires every cached link (``_links_lock``)
and closes every ssh forward (``_tunnels_lock``, each
:class:`SshTunnel`'s lock).  The Stop watchdog delivers
``KeyboardInterrupt`` to that thread with ``PyThreadState_SetAsyncExc``;
a thread blocked in a contended ``acquire`` receives it at the check
point right after ``__enter__`` returns, outside the ``with`` block
(CPython gh-74174), so a plain ``with lock:`` there leaked the lock for
good: a link's lock hung that task's ``finally: link.close()`` (the
task never ended), ``_links_lock`` hung every later CHATS refresh and
remote tab, ``_tunnels_lock`` every later connection to an SSH
machine.  Each site therefore goes through
:func:`~kiss.core.locked.locked_call` on a reentrant lock.

Each hammer runs in a child process (a leaked lock would wedge the
pytest process): a contender thread holds the lock half the time so
the worker's ``acquire`` really blocks, and the worker loops one real
path while hundreds of interrupts are injected into it.  The machine
is an unreachable address (connection refused, no network, no ssh
process), so the paths are the lock sites and nothing else.  The test
fails on the pre-fix ``with`` sites and passes on the fixed ones.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from kiss.agents.sorcar import machines as mach
from kiss.core.locked import locked_call
from kiss.tests.lock_injection import hammer

MACHINE = mach.Machine(name="x", url="https://127.0.0.1:1")
LOGIN = "me@127.0.0.1"


class _Request:
    """``MachineLink.request`` then ``close`` on a cached link: ``machine_status``'s path."""

    def __init__(self) -> None:
        self.link = mach.link_for(MACHINE)

    def __call__(self) -> None:
        try:
            self.link.request({"type": "ping"}, "pong", timeout=1.0)
        except mach.MachineError:
            pass
        finally:
            self.link.close()


class _Status:
    """``machine_status`` itself: a private link per call, so nothing can contend it."""

    def __call__(self) -> None:
        mach.machine_status(MACHINE)


class _Retire:
    """``MachineLink.retire`` on a cached link: ``close_links``'s per-link step."""

    def __init__(self) -> None:
        self.link = mach.link_for(MACHINE)

    def __call__(self) -> None:
        self.link.retire()


class _LinkFor:
    """``link_for``: the links registry lock."""

    def __call__(self) -> None:
        mach.link_for(MACHINE)


class _CloseLinks:
    """``link_for`` then ``close_links``: both registry locks and the link's."""

    def __call__(self) -> None:
        mach.link_for(MACHINE)
        mach.close_links()


class _CloseTunnels:
    """A cached tunnel (no process) closed through ``close_tunnels``: both tunnel locks."""

    def __init__(self) -> None:
        self.tunnel = mach.SshTunnel(LOGIN, 1)

    def __call__(self) -> None:
        locked_call(mach._tunnels_lock, mach._tunnels.setdefault, LOGIN, self.tunnel)
        mach.close_tunnels()


class _TunnelClose:
    """``SshTunnel.close`` on a tunnel without a process: the tunnel's own lock."""

    def __init__(self) -> None:
        self.tunnel = mach.SshTunnel(LOGIN, 1)

    def __call__(self) -> None:
        self.tunnel.close()


_DRIVERS: dict[str, type] = {
    "request": _Request, "status": _Status, "retire": _Retire, "link_for": _LinkFor,
    "close_links": _CloseLinks, "close_tunnels": _CloseTunnels, "tunnel_close": _TunnelClose,
}


def run_hammer(name: str) -> None:
    """Child-process body: hammer the path *name* and check the locks stay free."""
    driver = _DRIVERS[name]()
    locks: dict[str, object] = {
        "_links_lock": mach._links_lock, "_tunnels_lock": mach._tunnels_lock,
    }
    contended: object | None
    if name in ("request", "retire"):
        contended = locks["link"] = driver.link._lock
    elif name == "status":
        contended = None
    elif name in ("link_for", "close_links"):
        contended = mach._links_lock
    elif name == "close_tunnels":
        contended = mach._tunnels_lock
        locks["tunnel"] = driver.tunnel._lock
    else:
        contended = locks["tunnel"] = driver.tunnel._lock
    # The locks are checked once the contender is gone: it holds them most of the time.
    hammer(driver, locks, contend=contended)
    # The registry and the links still work afterwards: the next task's
    # ``/machine list`` gets its answer.
    assert mach.machine_status(MACHINE).startswith("x (https://127.0.0.1:1) ")
    mach.link_for(MACHINE)
    mach.close_links()
    assert mach._links == {} and mach._tunnels == {}


@pytest.mark.parametrize("name", sorted(_DRIVERS))
def test_machines_locks_survive_injection(name: str) -> None:
    """No lock site a task thread reaches in ``machines`` leaks under injection."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer({name!r})"],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0, f"{name}: {proc.stderr[-3000:]}"


if __name__ == "__main__":
    pytest.main([__file__])
