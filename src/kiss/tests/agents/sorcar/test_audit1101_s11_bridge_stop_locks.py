# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A task Stop landing on a Chrome bridge lock wait leaks neither the lock nor the command.

The runner's Stop injects ``KeyboardInterrupt`` into the task thread with
``PyThreadState_SetAsyncExc``.  A thread blocked in a contended
``acquire`` receives it at the check point right after the acquire
returns, outside a ``with`` block's exception region (CPython gh-74174),
so the thread dies owning the lock.  ``ChromeBridge.instance`` took the
singleton lock, and ``call`` the connection's send lock, that way: a
Stop while another thread held either left every later ``instance()``
hanging for good, or every later command failing "stayed busy", and the
cancelled command's future stayed registered in ``conn.pending``.  Every
task-thread section now goes through :func:`kiss.core.locked.locked_call`
(sliced acquires: the Stop lands within half a second even while the
holder keeps the lock), and the command's registration is cleaned up on
any exit.

Each scenario runs in a child process: on the old code the stopped
thread keeps the lock, which would wedge the pytest process at the next
``ChromeBridge.shutdown()``.  The real bridge, a real scripted
extension peer over a loopback socket and the real injection are used;
nothing is replaced.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import NoReturn

import pytest

from kiss.agents.sorcar.chrome_bridge import ChromeBridge
from kiss.tests.agents.sorcar.test_chrome_bridge import ExtensionPeer
from kiss.tests.lock_injection import inject, lock_is_free

_LOCK_FILES = ("chrome_bridge.py", "locked.py")


def _waiting_on_a_bridge_lock(thread: threading.Thread) -> bool:
    """Whether *thread*'s innermost Python frame is a bridge lock acquisition."""
    assert thread.ident is not None
    frame = sys._current_frames().get(thread.ident)
    if frame is None:
        return False
    top = traceback.extract_stack(frame)[-1]
    line = top.line or ""
    return top.filename.endswith(_LOCK_FILES) and ("acquire(" in line or line.startswith("with "))


def _fail(message: str) -> NoReturn:
    """End the child process at once: the leak just found would hang the orderly cleanup."""
    print(message, file=sys.stderr, flush=True)
    os._exit(1)


def _stop_while_blocked(
    lock: object, call: Callable[[], object], gone_while_held: bool = True
) -> list[str]:
    """Hold *lock*, run *call* on a thread, Stop it once it waits on a lock; return what it raised.

    With *gone_while_held* the thread must have exited while the lock is
    still held (the Stop reached it inside one acquire slice); without
    it the thread may wait for the lock once more on its way out (the
    command's cleanup takes ``pending_lock`` again).  Either way the lock
    is free once released here: a stopped thread that keeps it is the leak.
    """
    caught: list[str] = []

    def worker() -> None:
        try:
            call()
        except BaseException as exc:  # noqa: BLE001 - the Stop is the expected outcome
            caught.append(type(exc).__name__)

    lock.acquire()  # type: ignore[attr-defined]
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not _waiting_on_a_bridge_lock(thread):
        if time.monotonic() > deadline:
            _fail("the worker never reached the lock")
        time.sleep(0.005)
    assert inject(thread) == 1
    if gone_while_held:
        thread.join(3)
        if thread.is_alive():
            _fail("the Stop did not reach the thread while the lock was held")
    else:
        time.sleep(1)  # longer than an acquire slice: the Stop has been delivered
    lock.release()  # type: ignore[attr-defined]
    thread.join(3)
    if thread.is_alive():
        _fail("the stopped thread never exited")
    if not lock_is_free(lock):
        _fail("the stopped thread left the lock held")
    return caught


def run_singleton_lock() -> None:
    """Child body: a Stop while ``instance()`` waits for the singleton lock."""
    bridge = ChromeBridge(write_config=False)
    ChromeBridge._instance = bridge  # a live singleton: instance() only has to read it
    caught = _stop_while_blocked(ChromeBridge._instance_lock, ChromeBridge.instance)
    assert caught == ["KeyboardInterrupt"], caught
    assert ChromeBridge.instance() is bridge  # the singleton is still reachable


def _answer_list_tabs(command: dict[str, object]) -> list[object]:
    return []


def _send_lock(bridge: ChromeBridge) -> None:
    conn = bridge._conn
    assert conn is not None
    caught = _stop_while_blocked(conn.lock, functools.partial(bridge.call, "list_tabs", timeout=5))
    assert caught == ["KeyboardInterrupt"], caught
    assert conn.pending == {}, "the stopped command stayed registered"
    assert bridge.extension_connected
    assert bridge.call("list_tabs", timeout=5) == []  # the connection still works


def _pending_lock(bridge: ChromeBridge) -> None:
    conn = bridge._conn
    assert conn is not None
    caught = _stop_while_blocked(
        conn.pending_lock, functools.partial(bridge.call, "list_tabs", timeout=5),
        gone_while_held=False,
    )
    assert caught == ["KeyboardInterrupt"], caught
    assert conn.pending == {}, "the stopped command stayed registered"
    assert bridge.call("list_tabs", timeout=5) == []


def _connection_lock(bridge: ChromeBridge) -> None:
    # The peer never answers: the command's timeout path drops the
    # connection, which takes the bridge's connection lock.
    conn = bridge._conn
    assert conn is not None
    caught = _stop_while_blocked(
        bridge._conn_lock, functools.partial(bridge.call, "list_tabs", timeout=0.3)
    )
    assert caught == ["KeyboardInterrupt"], caught
    assert conn.pending == {}, "the stopped command stayed registered"


# Scenario name -> (its body, whether the scripted extension answers commands).
_CONNECTED_SCENARIOS: dict[str, tuple[Callable[[ChromeBridge], None], bool]] = {
    "send_lock": (_send_lock, True),
    "pending_lock": (_pending_lock, True),
    "connection_lock": (_connection_lock, False),
}
_SCENARIOS = ("singleton_lock", *_CONNECTED_SCENARIOS)


def run_scenario(name: str) -> None:
    """Child-process body: run the scenario *name* against a real bridge and peer."""
    if name == "singleton_lock":
        run_singleton_lock()
        return
    scenario, answers = _CONNECTED_SCENARIOS[name]
    bridge = ChromeBridge(write_config=False)
    bridge.start()
    try:
        peer = ExtensionPeer(bridge.port, bridge.token, _answer_list_tabs if answers else None)
        try:
            assert bridge.wait_for_extension(5)
            scenario(bridge)
        finally:
            peer.close()
    finally:
        bridge.stop()


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_a_stop_on_a_bridge_lock_wait_leaks_nothing(name: str) -> None:
    """The stopped thread exits within a slice, leaving the lock free and no command behind."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_scenario({name!r})"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, f"{name}: {proc.stderr[-3000:]}"


if __name__ == "__main__":
    pytest.main([__file__])
