# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The shutdown sweep re-checks ownership before injecting an interrupt.

Concurrency audit 2026 (C2): ``RemoteAccessServer._stop_active_agent_tasks``
— the graceful SIGTERM sweep — set every active state's stop event,
joined each worker for one second and then injected
``KeyboardInterrupt`` with NO re-check, unlike the user-Stop watchdog
(``_force_stop_thread``), which evaluates ``_state_owns_thread`` under
``STATE_LOCK`` immediately before every injection.  Consequences:

* a worker that honoured the cooperative stop and was already inside
  its legitimate cleanup ``finally`` (persisting the interrupted row,
  presenting the worktree — the very work the sweep's docstring says
  it exists to let finish) was interrupted AGAIN, aborting that
  cleanup;
* a worker performing the state's post-task merge was interrupted
  mid-merge, violating the "a merge is awaited, never stopped"
  invariant the sibling ``_await_active_merges`` enforces;
* in a narrow window a recycled thread ident could route the interrupt
  into an unrelated freshly spawned thread (closed by the
  ``task_thread is thread`` re-check under the lock, since the run's
  ``finally`` clears ``task_thread`` under the same lock before the
  thread can exit).

The fix guards the sweep's injection with the same
``_state_owns_thread`` predicate.  Tested with a REAL
``RemoteAccessServer``, real registered states and real threads; the
only substitution is the worker body itself, which stands in for an
agent run (external to the code under test).
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.server import agent_state
from kiss.server.agent_state import AgentState


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Run with an empty agent-state registry and leave it empty."""
    agent_state.agent_states.clear()
    try:
        yield
    finally:
        agent_state.agent_states.clear()


def _make_remote_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.server.web_server import RemoteAccessServer

    tmp = tempfile.mkdtemp(prefix="kiss-conc2026-sweep-")
    return RemoteAccessServer(
        use_tunnel=False,
        url_file=os.path.join(tmp, "url.json"),
        uds_path=os.path.join(tmp, "sorcar.sock"),
    )


class _CooperativeWorker:
    """Stands in for a run that honours the stop, then cleans up.

    On the stop event it acknowledges the cancellation through the
    REAL ``_cancel_outcome`` (exactly what ``_run_task_inner``'s
    interrupt handlers do) and then spends *cleanup_seconds* in its
    "cleanup finally" — longer than the sweep's 1 s pre-injection
    join, like a SQLite persist waiting out the busy timeout.  Records
    any ``KeyboardInterrupt`` landing in that cleanup.
    """

    def __init__(
        self, vscode: Any, state: AgentState, cleanup_seconds: float,
    ) -> None:
        self.vscode = vscode
        self.state = state
        self.cleanup_seconds = cleanup_seconds
        self.interrupted = threading.Event()
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            stop = self.state.stop_event
            assert stop is not None
            stop.wait(30)
            self.vscode._cancel_outcome(self.state)  # acknowledges the stop
            deadline = time.monotonic() + self.cleanup_seconds
            while time.monotonic() < deadline:
                time.sleep(0.005)
        except KeyboardInterrupt:
            self.interrupted.set()
        finally:
            self.finished.set()


class _StubbornWorker:
    """Stands in for a run wedged in a call that ignores the stop event."""

    def __init__(self, lifetime: float = 60.0) -> None:
        self.lifetime = lifetime
        self.interrupted = threading.Event()
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        try:
            deadline = time.monotonic() + self.lifetime
            while time.monotonic() < deadline:
                time.sleep(0.005)
        except KeyboardInterrupt:
            self.interrupted.set()
        finally:
            self.finished.set()


def _register(task_id: str, tab_id: str, thread: threading.Thread) -> AgentState:
    state = AgentState(
        task_id,
        tab_id=tab_id,
        server_owned=True,
        stop_event=threading.Event(),
        task_thread=thread,
        is_task_active=True,
    )
    agent_state.register(state)
    return state


def test_sweep_does_not_reinterrupt_acknowledged_cleanup(
    clean_registry: None,
) -> None:
    """A cooperatively stopped worker's slow cleanup runs to completion."""
    remote = _make_remote_server()
    vscode = remote._vscode_server
    state = _register("sweep-acked", "TAB-SWEEP-ACKED", threading.Thread())
    worker = _CooperativeWorker(vscode, state, cleanup_seconds=3.0)
    state.task_thread = worker.thread
    worker.thread.start()

    remote._stop_active_agent_tasks(timeout=10.0)

    assert worker.finished.wait(15), "the worker never finished its cleanup"
    assert not worker.interrupted.is_set(), (
        "BUG: the shutdown sweep injected KeyboardInterrupt into the "
        "cleanup of an already-acknowledged stop"
    )
    assert state.stop_acknowledged, "the cooperative ack must have happened"


def test_sweep_still_interrupts_a_wedged_worker(
    clean_registry: None,
) -> None:
    """The guarded injection still stops a worker ignoring the event."""
    remote = _make_remote_server()
    state = _register("sweep-wedged", "TAB-SWEEP-WEDGED", threading.Thread())
    worker = _StubbornWorker(lifetime=60.0)
    state.task_thread = worker.thread
    worker.thread.start()

    t0 = time.monotonic()
    remote._stop_active_agent_tasks(timeout=10.0)
    elapsed = time.monotonic() - t0

    assert worker.finished.wait(15), "the wedged worker was never interrupted"
    assert worker.interrupted.is_set(), (
        "BUG: the guarded sweep no longer interrupts a wedged worker"
    )
    assert elapsed < 10.5, elapsed


def test_sweep_awaits_a_merging_task_thread(
    clean_registry: None,
) -> None:
    """A task thread inside its post-task merge is joined, not injected."""
    remote = _make_remote_server()
    state = _register("sweep-merge", "TAB-SWEEP-MERGE", threading.Thread())
    worker = _StubbornWorker(lifetime=3.0)  # the "merge": outlives the 1 s join
    state.task_thread = worker.thread
    with agent_state.STATE_LOCK:
        state.is_merging = True
        state.merge_thread = worker.thread
    worker.thread.start()

    remote._stop_active_agent_tasks(timeout=10.0)

    assert worker.finished.wait(15), "the merging worker never finished"
    assert not worker.interrupted.is_set(), (
        "BUG: the shutdown sweep interrupted an in-flight merge"
    )
