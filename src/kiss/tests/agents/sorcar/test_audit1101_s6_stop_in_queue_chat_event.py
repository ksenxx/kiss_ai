"""An injected Stop inside ``_queue_chat_event`` never wedges event persistence.

The daemon stops a task by injecting ``KeyboardInterrupt`` into its thread
(``PyThreadState_SetAsyncExc``); the task thread may be inside
``persistence._queue_chat_event`` at that moment.  Before the fix the
event queue was a stdlib ``queue.Queue`` whose ``put`` runs Python-level
``Condition.__enter__``/``__exit__`` code: a Stop delivered right after
the queue mutex was taken leaked it for good (every later producer and
the writer blocked forever, ``_flush_chat_events`` spun), a Stop between
``deque.append`` and ``unfinished_tasks += 1`` killed the writer thread
with ``ValueError('task_done() called too many times')``, and a Stop
between the per-task reservation and the ``try`` stranded the
reservation so every later ``_flush_chat_events(task_id)`` hung.

This test reproduces the daemon's mechanism (real injections at random
moments into a producer thread, no monkeypatching) and asserts the three
invariants: the writer thread never dies, ``_pending_cond`` is always
acquirable, and every event whose ``_queue_chat_event`` returned is
persisted by a flush that returns promptly.

An injected exception that lands inside a ``__del__`` is reported as
"unraisable" and dropped by CPython; the flat-out producer triggers the
cyclic collector, which runs the finalizers of objects earlier tests of
the same process left behind.  The collector is therefore off during the
injection loop, and a Stop that still goes missing is re-injected once,
as the daemon's watchdog (``_force_stop_thread``) does, before the test
calls the producer wedged.

The producer's flat-out loop lives in a frame of its own (``spin``):
CPython 3.13 leaves a ``while`` loop's back-edge outside the exception
table of the handlers of its own frame (cpython gh-130279), so a Stop
raised on the back-edge of a loop written inside the ``try`` would
escape the ``except KeyboardInterrupt`` and kill the producer; raised
in ``spin``, it reaches the producer at its covered call.
"""

from __future__ import annotations

import gc
import json
import random
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.persistence_helpers import close_db

_INJECTIONS = 250


@pytest.fixture
def no_cyclic_gc() -> Iterator[None]:
    """Run the test with the cyclic collector off (collected once up front)."""
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point persistence at a fresh database under *tmp_path*."""
    monkeypatch.setattr(th, "_KISS_DIR", tmp_path)
    monkeypatch.setattr(th, "_DB_PATH", tmp_path / "history.db")
    close_db()
    yield tmp_path
    close_db()


@pytest.mark.usefixtures("no_cyclic_gc")
def test_injected_stops_inside_queue_chat_event_never_wedge_the_writer(
    isolated_db: Path,
) -> None:
    """Hundreds of real Stops landing inside ``_queue_chat_event`` leave the
    writer alive, the condition free, and every returned event persisted."""
    task_id, _chat = th._add_task("stop inside _queue_chat_event")
    event = json.dumps({"type": "text", "text": "x"})
    th._queue_chat_event(event, task_id)
    th._flush_chat_events(task_id)
    writer = th._event_writer_thread
    assert writer is not None and writer.is_alive()

    stop = threading.Event()
    in_loop = threading.Event()
    paused = threading.Event()
    resume = threading.Event()
    returned = [0]

    def spin() -> None:
        """Queue events flat out until stopped (its own frame, see above)."""
        in_loop.set()
        while not stop.is_set():
            th._queue_chat_event(event, task_id)
            returned[0] += 1

    def producer() -> None:
        """Queue events until stopped; park after every injected Stop."""
        while not stop.is_set():
            try:
                spin()
            except KeyboardInterrupt:
                in_loop.clear()
                paused.set()
                resume.wait()
                resume.clear()

    prod = threading.Thread(target=producer, name="s6-producer", daemon=True)
    prod.start()
    assert prod.ident is not None
    problems: list[str] = []
    for n in range(_INJECTIONS):
        assert in_loop.wait(5), f"injection {n}: the producer never re-entered its loop"
        time.sleep(random.uniform(0.0, 0.0004))
        assert inject_keyboard_interrupt(prod.ident) == 1
        if not paused.wait(5) and not paused.is_set():
            # Lost (landed in a finalizer): the watchdog's retry.
            assert inject_keyboard_interrupt(prod.ident) == 1
        if not paused.wait(5):
            problems.append(
                f"injection {n}: the producer did not receive the Stop within 5 s "
                "(blocked in a leaked lock)"
            )
            break
        if th._pending_cond.acquire(timeout=0.5):
            th._pending_cond.release()
        else:
            problems.append(f"injection {n}: _pending_cond stayed held after the Stop")
            break
        if th._event_writer_thread is not writer or not writer.is_alive():
            problems.append(f"injection {n}: the writer thread died or was replaced")
            break
        # Let the writer catch up while the producer is parked (the flat-out
        # producer starves it of the GIL), so the final flush stays short.
        while th._pending_count(None) > 5000:
            time.sleep(0.005)
        paused.clear()
        resume.set()
    stop.set()
    resume.set()
    prod.join(5)
    assert not prod.is_alive(), "the producer did not stop"
    assert not problems, problems

    flushed = threading.Event()

    def flush() -> None:
        """Wait for the task's events; must return promptly."""
        th._flush_chat_events(task_id)
        flushed.set()

    flusher = threading.Thread(target=flush, name="s6-flusher", daemon=True)
    flusher.start()
    assert flushed.wait(10), "_flush_chat_events(task_id) did not return within 10 s"
    assert th._pending_count(task_id) == 0
    assert th._pending_count(None) == 0
    row = th._get_db().execute(
        "SELECT COUNT(*) AS n FROM events WHERE task_id = ?", (task_id,),
    ).fetchone()
    persisted = int(row["n"]) - 1  # minus the warm-up event
    # An event whose call returned is persisted; a Stop landing after the
    # publication but before the call returned may add one more per injection.
    assert returned[0] <= persisted <= returned[0] + _INJECTIONS, (returned[0], persisted)
    assert th._event_writer_thread is writer and writer.is_alive()
