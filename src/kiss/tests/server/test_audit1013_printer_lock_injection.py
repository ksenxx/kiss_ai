# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The daemon's shared printer locks survive the runner's Stop injection.

The runner's Stop delivers ``KeyboardInterrupt`` to a task thread
with ``PyThreadState_SetAsyncExc``, which CPython raises at the next
check point of the bytecode, including the one right after a
``with lock:``'s ``__enter__`` returned (gh-74174): a plain ``with``
on a task thread leaks the lock, and the daemon has ONE ``WebPrinter``
for every task, so a leaked printer lock wedges printing for the
daemon's life.  Every printer lock a task thread can take therefore
goes through :func:`~kiss.agents.sorcar._concurrency.locked_call`.

Each test here hammers a real printer path from a worker thread with
hundreds of injections and then asserts that every lock the path takes
is free.  Before the conversion, ``JsonPrinter.tokens_offset``'s
setter leaked ``_lock`` in roughly one of ten injections.

The hammers run in a child process (:func:`run_hammer`): the printer
paths also take the daemon-wide ``agent_state.STATE_LOCK`` (a display
event is filed for persistence through ``agent_state.get``) and
persistence's own lock, and a lock the worker leaked there would wedge
the rest of the pytest session.  ``STATE_LOCK`` is checked by every
hammer; ``test_audit1013_state_lock_injection`` hammers the registry
and the runner's own task-thread sites directly.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Callable

import pytest

from kiss.agents.sorcar._concurrency import locked_call
from kiss.agents.sorcar.persistence import TASK_USAGE_LOCK
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.task_runner import _TaskRunnerMixin
from kiss.server.web_server import WebPrinter
from kiss.tests.lock_injection import INJECTIONS, hammer, lock_is_free

# The core and sorcar hammers still import the helpers under these
# names; they live in ``kiss.tests.lock_injection`` now.
_hammer = hammer
_lock_is_free = lock_is_free


def _printer_locks(printer: JsonPrinter) -> dict[str, object]:
    return {
        "_lock": printer._lock,
        "_bash_lock": printer._bash_lock,
        "_model_pick_lock": printer._model_pick_lock,
        "delivery_lock": printer.delivery_lock,
    }


class _OffsetWriter:
    """``printer.tokens_offset = n`` on the worker, under its task key."""

    def __init__(self, printer: JsonPrinter) -> None:
        self.printer = printer

    def __call__(self) -> None:
        self.printer._thread_local.task_id = "audit1013-offsets"
        self.printer.tokens_offset = 1


class _Broadcaster:
    """``printer.broadcast`` of a task event on the worker: the delivery path.

    The event is a non-display one (``status``), so the path stays
    within the printer: a display event is also filed for persistence
    through ``agent_state.get``.  A subscribed tab makes the fan-out
    reach the send path (``_ws_lock`` in a ``WebPrinter``).
    """

    def __init__(self, printer: JsonPrinter) -> None:
        self.printer = printer
        printer.subscribe_tab("audit1013-broadcast", "tab-1")

    def __call__(self) -> None:
        self.printer._thread_local.task_id = "audit1013-broadcast"
        self.printer.broadcast({"type": "status", "running": True})


class _DisplayBroadcaster(_Broadcaster):
    """``broadcast`` of a display event: the delivery path plus persistence filing."""

    def __call__(self) -> None:
        self.printer._thread_local.task_id = "audit1013-broadcast"
        self.printer.broadcast({"type": "text_delta", "text": "x"})


class _BashStreamer:
    """``printer.print(type="bash_stream")`` on the worker: the buffering path."""

    def __init__(self, printer: JsonPrinter) -> None:
        self.printer = printer

    def __call__(self) -> None:
        self.printer._thread_local.task_id = "audit1013-bash"
        self.printer.print("line\n", type="bash_stream")
        self.printer.print("ls", type="tool_call", tool_input={})
        self.printer.print("done", type="tool_result", tool_name="Bash", is_error=False)


class _ModelPicker:
    """An agent's ``set_model`` and the end-of-run restore on the worker."""

    def __init__(self, printer: JsonPrinter) -> None:
        self.printer = printer

    def __call__(self) -> None:
        self.printer._thread_local.task_id = "audit1013-pick"
        self.printer.subscribe_tab("audit1013-pick", "tab-1")
        self.printer.broadcast_agent_model_pick("m1", "tab-1")
        self.printer.restore_model_pick("m0", "tab-1")
        self.printer.cleanup_tab("tab-1")


class _Lifecycle:
    """A run's recording lifecycle plus the runner's ``cleanup_task``."""

    def __init__(self, printer: JsonPrinter) -> None:
        self.printer = printer

    def __call__(self) -> None:
        self.printer._thread_local.task_id = "audit1013-life"
        self.printer.start_recording()
        self.printer.ensure_recording_for_task("audit1013-life")
        self.printer.peek_recording()
        self.printer.stop_recording()
        self.printer.cleanup_task("audit1013-life", subscriber_linger_seconds=0)


class _Runner(_TaskRunnerMixin):
    """The runner mixin over a bare ``JsonPrinter``: what ``_persist_subtask_row`` needs."""

    def __init__(self) -> None:
        self.printer = JsonPrinter()


class _SubtaskPersister:
    """``_persist_subtask_row`` on the worker, between two sub-tasks of a run."""

    def __init__(self, runner: _Runner) -> None:
        self.runner = runner
        self.state = AgentState("audit1013-row")

    def __call__(self) -> None:
        self.runner._persist_subtask_row(
            self.state, task_id="audit1013-row", result_summary="r",
            model="m", work_dir="/tmp", sub_start_ms=0,
        )


_DRIVERS: dict[str, type] = {
    "offset": _OffsetWriter,
    "broadcast": _Broadcaster,
    "broadcast_display": _DisplayBroadcaster,
    "bash_stream": _BashStreamer,
    "model_pick": _ModelPicker,
    "lifecycle": _Lifecycle,
}


def run_hammer(name: str) -> None:
    """Hammer the printer path *name* (``web:<path>`` drives a ``WebPrinter``).

    The child-process body of every printer test: builds the real
    printer and driver, runs :func:`hammer` and checks the per-task
    ``flush_lock``s too.  ``persist`` drives the runner's
    ``_persist_subtask_row`` instead.
    """
    kind, _, path = name.rpartition(":")
    if path == "persist":
        runner = _Runner()
        printer: JsonPrinter = runner.printer
        call: Callable[[], object] = _SubtaskPersister(runner)
    else:
        printer = WebPrinter() if kind == "web" else JsonPrinter()
        call = _DRIVERS[path](printer)
    locks = _printer_locks(printer)
    if isinstance(printer, WebPrinter):
        locks["_ws_lock"] = printer._ws_lock
    if path == "persist":
        locks["TASK_USAGE_LOCK"] = TASK_USAGE_LOCK
    hammer(call, locks)
    for state in list(printer._bash_states.values()):
        assert lock_is_free(state.flush_lock), "an interrupt leaked flush_lock"


def _hammer_isolated(name: str) -> None:
    """Run :func:`run_hammer` for *name* in a child process and relay its verdict."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer({name!r})"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]


class TestJsonPrinterLocks:
    """No ``JsonPrinter`` path a task thread runs can leak a printer lock."""

    @pytest.mark.parametrize("path", sorted(_DRIVERS))
    def test_path(self, path: str) -> None:
        _hammer_isolated(path)


class TestWebPrinterLocks:
    """The daemon's one ``WebPrinter``: delivery, worktree tracking and send paths."""

    @pytest.mark.parametrize("path", ["broadcast", "broadcast_display", "bash_stream"])
    def test_path(self, path: str) -> None:
        _hammer_isolated("web:" + path)


class TestTaskRunnerLocks:
    def test_persist_subtask_row(self) -> None:
        _hammer_isolated("persist")


class _ReentrantSection:
    """An outer ``locked_call`` whose body loops inner ``locked_call``s on the same lock.

    The shape of the runner's setup-failure path, which holds
    ``delivery_lock`` while ``broadcast`` takes it again.  After every
    inner call, interrupted or not, the outer hold must still be the
    thread's: an inner release may only undo the inner acquisition.
    """

    def __init__(self, lock: threading.RLock) -> None:
        self.lock = lock
        self.inner_calls = 0
        self.outer_lost = 0

    def __call__(self) -> None:
        locked_call(self.lock, self._inner_loop)

    def _inner_loop(self) -> None:
        for _ in range(1000):
            try:
                locked_call(self.lock, _noop)
            except KeyboardInterrupt:
                pass
            self.inner_calls += 1
            if not self.lock._is_owned():  # type: ignore[attr-defined]
                self.outer_lost += 1


def _noop() -> None:
    pass


class TestReentrantHold:
    """``delivery_lock`` is held re-entrantly by the runner's setup-failure
    broadcast: an interrupt inside the inner ``locked_call`` never
    releases the outer hold."""

    def test_inner_interrupts_keep_outer_hold(self) -> None:
        lock = threading.RLock()
        section = _ReentrantSection(lock)
        hammer(section, {"lock": lock})
        assert section.inner_calls > INJECTIONS
        assert section.outer_lost == 0
