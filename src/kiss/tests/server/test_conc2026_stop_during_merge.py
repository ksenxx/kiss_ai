# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A user Stop must never interrupt an in-flight post-task worktree merge.

Concurrency audit 2026 (M-C2): the codebase's stated invariant is that
a merge is *awaited, never stopped* — the shutdown path
(``_await_active_merges``) refuses to interrupt ``merge_thread``s
because aborting a stash → checkout → merge → pop sequence midway
leaves the user's repository in a state they did not ask for.  The
user-Stop watchdog did not honour that invariant: its ownership guard
``_state_owns_thread`` checked ``task_thread`` identity and
``stop_acknowledged`` but not ``is_merging``, so a Stop clicked while
the task thread ran the post-task auto-merge injected
``KeyboardInterrupt`` into ``wt.merge()`` (swallowed by the
presentation handler's ``except BaseException``, silently abandoning
the merge while the run still reported success).

The fix makes ``_state_owns_thread`` answer ``False`` while the TARGET
thread itself holds the state's merge claim (``is_merging`` and
``merge_thread is thread``, both published under ``STATE_LOCK`` by
``_handle_worktree_action``).  Both directions matter and are tested
with the REAL ``_stop_task`` + REAL watchdog + real threads:

* a merging task thread is not injected (the merge finishes), and
* a claim held by a DIFFERENT thread does not shield a wedged task
  thread — the stop is still enforced, never stranded.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer
from kiss.server.task_runner import _state_owns_thread
from kiss.tests.server._memory_printer import MemoryPrinter


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Run with an empty agent-state registry and leave it empty."""
    agent_state.agent_states.clear()
    try:
        yield
    finally:
        agent_state.agent_states.clear()


class _Worker:
    """A real thread that ignores the cooperative stop event.

    Spins in Python for *lifetime* seconds (so an injected
    ``KeyboardInterrupt`` lands at the next bytecode boundary) and
    records whether it was interrupted.
    """

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


def _register(
    task_id: str,
    tab_id: str,
    worker: _Worker,
) -> AgentState:
    """Register a server-owned state that owns *worker*'s thread."""
    state = AgentState(
        task_id,
        tab_id=tab_id,
        server_owned=True,
        stop_event=threading.Event(),
        task_thread=worker.thread,
    )
    agent_state.register(state)
    return state


def test_predicate_refuses_only_the_merging_thread(
    clean_registry: None,
) -> None:
    """``_state_owns_thread`` is False exactly while *thread* merges."""
    worker = _Worker(lifetime=2.0)
    state = _register("m-c2-pred", "TAB-M-C2-PRED", worker)
    worker.thread.start()
    other = threading.Thread(target=lambda: None)
    try:
        with agent_state.STATE_LOCK:
            # Baseline: owned, unacknowledged, not merging.
            assert _state_owns_thread(state, worker.thread)
            # The thread itself holds the merge claim: refuse.
            state.is_merging = True
            state.merge_thread = worker.thread
            assert not _state_owns_thread(state, worker.thread)
            # A FOREIGN claim (another thread, or a stale one with no
            # thread) must not shield the task thread from a stop.
            state.merge_thread = other
            assert _state_owns_thread(state, worker.thread)
            state.merge_thread = None
            assert _state_owns_thread(state, worker.thread)
            # Claim released after the merge: stop enforcement resumes.
            state.is_merging = False
            state.merge_thread = None
            assert _state_owns_thread(state, worker.thread)
    finally:
        worker.finished.wait(10)


def test_stop_does_not_interrupt_the_post_task_merge(
    clean_registry: None,
) -> None:
    """Stop during the task thread's own merge: awaited, not injected."""
    printer = MemoryPrinter()
    server = VSCodeServer(printer=printer)
    tab_id = "TAB-M-C2-MERGE"
    # The "merge" outlives both watchdog injection points (1 s and
    # 1 s + 5 s), so a regression that injects at either point is
    # caught deterministically.
    worker = _Worker(lifetime=7.5)
    state = _register("m-c2-merge", tab_id, worker)
    # Publish the merge claim under STATE_LOCK exactly like
    # ``_handle_worktree_action``'s internal post-task call does on
    # the task thread.
    with agent_state.STATE_LOCK:
        state.is_merging = True
        state.merge_thread = worker.thread
    worker.thread.start()

    server._stop_task(tab_id)

    assert worker.finished.wait(20), "the merging worker never finished"
    assert not worker.interrupted.is_set(), (
        "BUG: the Stop watchdog injected KeyboardInterrupt into the "
        "task thread's in-flight post-task merge"
    )
    # The stop click itself was acknowledged to the UI.
    acks = [e for e in printer.emitted if e.get("type") == "stop_ack"]
    assert acks and acks[-1].get("accepted") is True, printer.emitted


def test_foreign_merge_claim_does_not_strand_the_stop(
    clean_registry: None,
) -> None:
    """A claim held by ANOTHER thread must not block stopping the task."""
    printer = MemoryPrinter()
    server = VSCodeServer(printer=printer)
    tab_id = "TAB-M-C2-FOREIGN"
    worker = _Worker(lifetime=60.0)
    state = _register("m-c2-foreign", tab_id, worker)
    with agent_state.STATE_LOCK:
        state.is_merging = True
        state.merge_thread = None  # stale / foreign holder
    worker.thread.start()

    server._stop_task(tab_id)

    assert worker.finished.wait(15), "the worker was never interrupted"
    assert worker.interrupted.is_set(), (
        "BUG: a foreign is_merging claim stranded the user's Stop"
    )
