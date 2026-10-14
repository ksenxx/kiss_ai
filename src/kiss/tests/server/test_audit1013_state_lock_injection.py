# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``agent_state.STATE_LOCK`` survives the runner's Stop injection on a task thread.

``STATE_LOCK`` is the daemon-wide registry lock: every display event a
task broadcasts files itself for persistence through ``agent_state.get``,
and the runner's own task-thread code (state resolution, the
cancellation verdict, the end-of-run release) takes it too.  The Stop
watchdog delivers ``KeyboardInterrupt`` to the task thread with
``PyThreadState_SetAsyncExc``; an interrupt raised at the check point
right after a plain ``with STATE_LOCK:``'s ``__enter__`` returned skips
``__exit__`` (CPython gh-74174) and leaks the lock for the daemon's
life: every later task, ``VSCodeServer()`` and every timer flush then
hang on it.  Every ``STATE_LOCK`` site a task thread can reach therefore
goes through :func:`~kiss.agents.sorcar._concurrency.locked_call`.

Each test hammers one real path from a worker thread with hundreds of
injections, in a child process (a leaked ``STATE_LOCK`` would wedge
the rest of the pytest session), and asserts the lock is free after.
"""

from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from kiss.agents.sorcar._concurrency import locked_call
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer
from kiss.server.task_runner import _stop_interrupt_wrapped, _TaskRunnerMixin
from kiss.server.web_server import WebPrinter
from kiss.tests.lock_injection import hammer


class _Registry:
    """Every ``agent_state`` registry function, in a run's order, on the worker."""

    def __init__(self) -> None:
        self.agent = object()

    def __call__(self) -> None:
        state = AgentState("audit1013-reg", tab_id="tab-reg", server_owned=True)
        state.agent = self.agent  # type: ignore[assignment]
        state.task_thread = threading.current_thread()
        agent_state.register(state)
        agent_state.rekey(state, "audit1013-reg-2")
        assert agent_state.get("audit1013-reg-2") is state
        assert agent_state.find_by_tab("tab-reg") is state
        assert agent_state.find_by_agent(self.agent) is state
        assert agent_state.current_agent() is self.agent
        assert state in agent_state.snapshot()
        state.is_merging = True
        state.merge_thread = threading.current_thread()
        assert state.merge_in_progress()
        state.is_merging = False
        assert not state.busy() or state.thread_alive()
        agent_state.unregister("audit1013-reg-2", state)


class _Runner(_TaskRunnerMixin):
    """The runner mixin over a bare ``JsonPrinter`` with the server's tab maps."""

    def __init__(self) -> None:
        self.printer = JsonPrinter()
        self._state_lock = agent_state.STATE_LOCK
        self._tab_models = {"tab-run": "m0", "tab-viewer": "m0"}
        self._tab_chat_views: dict[str, str] = {}
        self._tab_opened_task_ids: dict[str, str] = {}
        self._default_model = "m0"


class _CancelVerdict:
    """The stop verdict and its acknowledgement: the re-injection window."""

    def __init__(self, runner: _Runner) -> None:
        self.runner = runner
        self.state = AgentState("audit1013-cancel", stop_event=threading.Event())
        self.state.stop_event.set()  # type: ignore[union-attr]

    def __call__(self) -> None:
        assert _stop_interrupt_wrapped(KeyboardInterrupt(), self.state)
        assert self.runner._cancel_outcome(self.state)[1] == {"type": "task_stopped"}


class _RunStateCycle:
    """``_resolve_run_state`` then the end-of-run release and picker restore."""

    def __init__(self, runner: _Runner) -> None:
        self.runner = runner
        self.runner.printer.subscribe_tab("audit1013-run", "tab-viewer")

    def __call__(self) -> None:
        cmd: dict[str, object] = {"_state_key": "", "tabId": "tab-run"}
        state = self.runner._resolve_run_state(cmd)
        assert state.task_thread is threading.current_thread()
        locked_call(
            self.runner._state_lock, self.runner._release_run_state, state, {}, "tab-run", "",
        )
        self.runner._broadcast_status_end_to_viewers("audit1013-run", "tab-run")


_DRIVERS: dict[str, type] = {
    "registry": _Registry,
    "cancel": _CancelVerdict,
    "run_state": _RunStateCycle,
}


class _Server:
    """A real ``VSCodeServer`` over the daemon's ``WebPrinter``, its boot threads joined."""

    def __init__(self) -> None:
        self.printer = WebPrinter()
        self.server = VSCodeServer(self.printer)
        if self.server._orphan_sweep_thread is not None:
            self.server._orphan_sweep_thread.join(20)
        for thread in threading.enumerate():
            if thread.name == "task-dependency-prewarm":
                thread.join(20)

    def locks(self) -> dict[str, object]:
        return {
            "printer._lock": self.printer._lock,
            "printer._ws_lock": self.printer._ws_lock,
            "delivery_lock": self.printer.delivery_lock,
        }


class _ViewerSubscriber(_Server):
    """``_subscribe_chat_viewers`` at task startup: the viewer scan and the subscribe."""

    def __init__(self) -> None:
        super().__init__()
        self.server._tab_chat_views["tab-viewer"] = "chat-1"

    def __call__(self) -> None:
        self.server._subscribe_chat_viewers(
            "audit1013-sub", "chat-1", source_tab_id="tab-launcher", start_ms=0,
        )


class _TabDisplacer(_Server):
    """A task's tab taking over a chat another tab shows: the displaced tab's drop.

    ``_registry_update_tab`` prunes the displaced tab's local interest
    (``WebPrinter.prune_local_tab``, ``_ws_lock``) and drops its state
    (``_drop_tab_state`` and the teardown tail, ``STATE_LOCK``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.server._registry_update_tab("tab-a", chat_id="chat-d", create=True)
        self.next_tab = "tab-b"

    def __call__(self) -> None:
        self.server._registry_update_tab(self.next_tab, chat_id="chat-d", create=True)
        self.next_tab = "tab-a" if self.next_tab == "tab-b" else "tab-b"


class _ClosedTabDisposer(_Server):
    """``_dispose_if_closed`` at a lifecycle transition: the claim and the teardown."""

    def __call__(self) -> None:
        state = AgentState("audit1013-closed", tab_id="tab-closed", server_owned=True)
        agent_state.register(state)
        state.frontend_closed = True
        self.server._dispose_if_closed("tab-closed")
        agent_state.unregister("audit1013-closed", state)


class _TalkBroadcaster(_Server):
    """A ``talk`` event's per-device arbitration: ``_local_tab_shown`` on the task thread."""

    def __init__(self) -> None:
        super().__init__()
        self.printer.register_local_tab("local-peer", "tab-headless", set())
        self.printer.subscribe_tab("audit1013-talk", "tab-headless")

    def __call__(self) -> None:
        self.printer._thread_local.task_id = "audit1013-talk"
        self.printer.broadcast({"type": "talk", "text": "t", "taskId": "audit1013-talk"})


_SERVER_DRIVERS: dict[str, tuple[type, str]] = {
    "subscribe_viewers": (_ViewerSubscriber, "STATE_LOCK"),
    "displace_tab": (_TabDisplacer, "STATE_LOCK"),
    "displace_tab_ws": (_TabDisplacer, "printer._ws_lock"),
    "dispose_closed": (_ClosedTabDisposer, "STATE_LOCK"),
    "talk": (_TalkBroadcaster, "STATE_LOCK"),
}


def run_hammer(name: str) -> None:
    """Child-process body: hammer the path *name* and check ``STATE_LOCK`` stays free.

    The ``_SERVER_DRIVERS`` paths run over a real ``VSCodeServer`` with
    the lock they are named for kept contended, so the worker's
    ``acquire()`` blocks and the interrupt lands on its return.
    """
    if name in _SERVER_DRIVERS:
        driver_type, contended = _SERVER_DRIVERS[name]
        driver = driver_type()
        locks = driver.locks()
        contend = agent_state.STATE_LOCK if contended == "STATE_LOCK" else locks[contended]
        hammer(driver, locks, contend=contend)
        return
    runner = _Runner()
    call = _Registry() if name == "registry" else _DRIVERS[name](runner)
    hammer(call, {"printer._lock": runner.printer._lock})


def _hammer_isolated(name: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer({name!r})"],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0, f"{name}: {proc.stderr[-3000:]}"


@pytest.mark.parametrize("name", sorted(_DRIVERS))
def test_state_lock_paths(name: str) -> None:
    """No ``STATE_LOCK`` site a task thread runs leaks the lock under injection."""
    _hammer_isolated(name)


@pytest.mark.parametrize("name", sorted(_SERVER_DRIVERS))
def test_server_task_thread_paths(name: str) -> None:
    """The server's task-thread paths (viewer subscription, tab displacement,
    deferred disposal, talk arbitration) leak neither ``STATE_LOCK`` nor
    the printer's ``_ws_lock`` under injection on a blocking acquire."""
    _hammer_isolated(name)
