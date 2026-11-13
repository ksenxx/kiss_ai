# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review round 2 (sorcar_b F1/F2): the event writer under concurrent producers and a writer death.

F1: ``_flush_chat_events(task_id)`` scanned the live deque for the task's
newest event while producers appended to it without the condition (the
append is atomic by design), so the scan could raise ``RuntimeError:
deque mutated during iteration`` into the flushing caller.  The goal is
now computed on a snapshot taken in one C call.

F2: the writer popped its batch into a separate in-flight list before
persisting it.  A writer that died right after taking the batch left
that list behind: its replacement found an empty deque and returned no
batch (a task flush waited forever), and a later take re-counted the
orphan, leaving the watermark short for good.  The batch now stays in
the deque until it is persisted, so a replacement writer simply takes it
again, and a flusher waits for its own target event to leave the deque
rather than for a count of the events written: a writer dying between
persisting a batch and popping it (the round-2 reviewer's case) only
makes the replacement persist that batch again; it can no longer release
a flusher whose event is still queued.

2026-11-10: a waiting flusher looked for its event in the whole queue on
every 50 ms tick.  Behind a blocked writer some twenty flushers scanned
400,000 queued events each, under the GIL, and starved the thread whose
progress would have unblocked the writer.  A flusher now fixes, once,
the number of pops after which its event has left the queue, and the
writer counts each pop right after making it: a writer that dies between
the two leaves the count behind, never ahead, and an empty queue
releases the flusher whatever the count says.

The tests use real threads, the real writer and a real sqlite database;
the writer deaths are produced the way the daemon stops a thread, by an
injected ``KeyboardInterrupt``, delivered at the point where the writer
holds its batch or is popping it (a trace function parks the writer
there).
"""

from __future__ import annotations

import inspect
import json
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.core.locked import locked_call
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.persistence_helpers import close_db


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point persistence at a fresh database under *tmp_path*."""
    monkeypatch.setattr(th, "_KISS_DIR", tmp_path)
    monkeypatch.setattr(th, "_DB_PATH", tmp_path / "history.db")
    close_db()
    yield tmp_path
    close_db()


def _event_count(task_id: str) -> int:
    """Return the number of persisted events of *task_id*."""
    row = th._get_db().execute(
        "SELECT COUNT(*) AS n FROM events WHERE task_id = ?", (task_id,),
    ).fetchone()
    return int(row["n"])


def test_a_task_scoped_flush_survives_a_concurrent_append(isolated_db: Path) -> None:
    """A producer appending while another task's flush scans the queue does not break the flush."""
    task_a, _chat = th._add_task("flushed task")
    task_b, _chat = th._add_task("busy task")
    event = json.dumps({"type": "text", "text": "x"})
    # The producer serialises its events itself, as the runner does: the
    # work between two appends is where a flush's scan starts overlapping.
    dict_event: dict[str, object] = {"type": "text", "text": "y" * 1024}
    stop = threading.Event()
    errors: list[BaseException] = []

    def producer() -> None:
        """Append events for task B flat out until stopped."""
        while not stop.is_set():
            th._queue_chat_event(dict_event, task_b)

    def flusher() -> None:
        """Flush task A; the defect raised RuntimeError out of this call."""
        try:
            th._flush_chat_events(task_a)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    prod = threading.Thread(target=producer, name="r2-producer", daemon=True)
    flush = threading.Thread(target=flusher, name="r2-flusher", daemon=True)
    # The writer blocks on the write lock, so the queue stays long enough
    # for the flush's scan of it to overlap the producer's appends.
    with th._rw_lock.write_lock():
        th._queue_chat_event(event, task_a)
        for _ in range(300_000):
            th._queue_chat_event(event, task_b)
        prod.start()
        flush.start()
        flush.join(0.5)  # the flush has computed its goal (or raised) by now
        stop.set()
    prod.join(10)
    assert not prod.is_alive(), "the producer did not stop"
    flush.join(60)
    assert not flush.is_alive(), "the flush did not return"
    assert not errors, errors
    assert _event_count(task_a) == 1


def test_a_stop_inside_the_writers_start_leaves_one_writer(isolated_db: Path) -> None:
    """A Stop cutting the lazy start once the writer's OS thread exists leaves one writer.

    The starter starts a writer again while it still holds the
    interpreter (hence the switch interval), before the first thread
    reports itself started: two writers would each persist the queued
    event and each pop it, the second pop taking the next event unwritten.
    """
    task_id, _chat = th._add_task("stopped start")

    def stop_once_the_thread_exists(_frame: FrameType, kind: str, arg: Any) -> None:
        if kind == "c_return" and getattr(arg, "__name__", "") == "start_joinable_thread":
            raise KeyboardInterrupt

    def flush() -> None:
        flusher = threading.Thread(target=th._flush_chat_events, args=(task_id,), daemon=True)
        flusher.start()
        flusher.join(30)
        assert not flusher.is_alive(), "the flush hung"

    interval = sys.getswitchinterval()
    sys.setswitchinterval(5.0)
    sys.setprofile(stop_once_the_thread_exists)
    try:
        with pytest.raises(KeyboardInterrupt):
            th._queue_chat_event({"type": "text", "text": "once"}, task_id)
        th._start_event_writer()
    finally:
        sys.setprofile(None)
        sys.setswitchinterval(interval)
    flush()
    th._queue_chat_event({"type": "text", "text": "next"}, task_id)
    flush()
    rows = th._get_db().execute(
        "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq", (task_id,),
    ).fetchall()
    assert [json.loads(row[0])["text"] for row in rows] == ["once", "next"]


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_writer_that_dies_holding_a_batch_leaves_it_to_its_replacement(
    isolated_db: Path,
) -> None:
    """Events a dead writer had taken are persisted by the next writer; flushes return.

    The injected Stop ends the writer thread with an unhandled
    ``KeyboardInterrupt`` by design, hence the filtered warning.
    """
    task_id, _chat = th._add_task("writer death")
    event = json.dumps({"type": "text", "text": "x"})
    th._stop_event_writer()
    parked = threading.Event()
    release = threading.Event()

    def trace(frame: FrameType, kind: str, _arg: Any) -> Any:
        """Park the writer at its first line after it took a non-empty batch."""
        if frame.f_code is not th._event_writer_loop.__code__:
            return None
        if kind == "line" and frame.f_locals.get("batch") and not parked.is_set():
            parked.set()
            release.wait(10)
        return trace

    threading.settrace(trace)
    try:
        th._queue_chat_event(event, task_id)
        assert parked.wait(10), "the writer never took the batch"
        writer = th._event_writer_thread
        assert writer is not None and writer.ident is not None
        assert inject_keyboard_interrupt(writer.ident) == 1
        release.set()
        writer.join(10)
    finally:
        threading.settrace(None)
    assert not writer.is_alive(), "the injected Stop did not end the writer"

    flushed = threading.Event()

    def flush() -> None:
        """Wait for the task's events; a replacement writer must persist them."""
        th._flush_chat_events(task_id)
        flushed.set()

    threading.Thread(target=flush, name="r2-flush-1", daemon=True).start()
    assert flushed.wait(10), "the flush hung after the writer died"
    assert _event_count(task_id) == 1
    # The watermark is intact: a later event is flushed by the usual path.
    th._queue_chat_event(event, task_id)
    flushed.clear()
    threading.Thread(target=flush, name="r2-flush-2", daemon=True).start()
    assert flushed.wait(10), "the flush hung after the replacement writer took over"
    assert _event_count(task_id) == 2
    assert th._pending_count(None) == 0


def test_flushes_waiting_behind_a_long_queue_do_not_rescan_it(isolated_db: Path) -> None:
    """Flushes parked behind a long queue leave the CPU to the rest of the process.

    Sixteen flushes wait for events queued behind 200,000 others while the
    writer is blocked.  Each used to search the whole queue for its event
    on every tick, which together took the GIL for most of the second
    measured here; a comparison of two counts takes next to none of it.
    """
    bulk, _chat = th._add_task("bulk task")
    waiting = [th._add_task(f"flushed task {index}")[0] for index in range(16)]
    event = json.dumps({"type": "text", "text": "x"})
    flushers = [
        threading.Thread(
            target=th._flush_chat_events, args=(task,), name=f"r2-waiting-{task}", daemon=True,
        )
        for task in waiting
    ]
    # The writer blocks on the write lock, so the queue stays long while
    # the flushes wait behind it.
    with th._rw_lock.write_lock():
        for _ in range(200_000):
            th._queue_chat_event(event, bulk)
        for task in waiting:
            th._queue_chat_event(event, task)
        for flusher in flushers:
            flusher.start()
        time.sleep(0.5)  # every flush has fixed its target and waits by now
        cpu = time.process_time()
        time.sleep(1.0)
        cpu = time.process_time() - cpu
    for flusher in flushers:
        flusher.join(60)
    assert not any(flusher.is_alive() for flusher in flushers), "a flush did not return"
    assert cpu < 0.25, f"sixteen waiting flushes used {cpu:.2f}s of CPU in one second"
    assert [_event_count(task) for task in waiting] == [1] * len(waiting)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
@pytest.mark.parametrize(
    ("boundary", "pops"),
    [
        ("_event_deque.popleft()", 0),
        ("_event_deque.popleft()", 1),
        ("_event_deque.popleft()", th._BATCH_MAX - 1),
        ("_pending_cond.notify_all()", th._BATCH_MAX),
        ("_events_popped += 1", 0),
        ("_events_popped += 1", th._BATCH_MAX - 1),
    ],
)
def test_a_writer_that_dies_acknowledging_a_batch_cannot_release_a_later_flush(
    isolated_db: Path, boundary: str, pops: int,
) -> None:
    """A flush waiting for an event behind a persisted batch returns only once its own
    event is persisted, whichever step of the acknowledgement the writer dies at.

    Task A fills the writer's first batch and task B's event is next.  The
    writer dies inside ``_release_batch_locked`` at the line *boundary* of
    the loop pass *pops*: before that pass's pop (*pops* of A's events
    have left the queue), before its count (one more has left, uncounted),
    or after the last pass, before the wake-up.  The round-2 writer counted
    the batch as written before popping it, so the replacement's
    re-persisting of A satisfied B's flusher with B still queued (the
    round-3 review's case).  The writer now counts a pop only after making
    it, and the flusher waits for the count of B's own pop: while the
    replacement is held before persisting the batch holding B, the flusher
    is still waiting and B is still queued; it returns once B is persisted,
    also when a pop the dead writer never counted leaves the count one
    short of B's.  The handshakes are trace events of the real functions.
    """
    task_a, _chat = th._add_task("bulk task")
    task_b, _chat = th._add_task("flushed task")
    event = json.dumps({"type": "text", "text": "x"})
    th._stop_event_writer()
    popped = th._events_popped  # the queue is empty and nothing pops until the batch below
    source, first_line = inspect.getsourcelines(th._release_batch_locked)
    boundary_line = first_line + next(
        index for index, line in enumerate(source) if line.strip() == boundary
    )
    first_taken, release_first = threading.Event(), threading.Event()
    dying, release_death = threading.Event(), threading.Event()
    tail_taken, release_tail = threading.Event(), threading.Event()
    target_fixed, rechecked = threading.Event(), threading.Event()
    targets: list[Any] = []
    rechecks: list[bool] = []

    def trace(frame: FrameType, kind: str, arg: Any) -> Any:
        """Park the writer at its first persist, at the death boundary and at the persist
        of the batch holding B; record the flusher's target and its re-checks."""
        code = frame.f_code
        if code is th._persist_batch_with_retry.__code__ and kind == "call":
            if not first_taken.is_set():
                first_taken.set()
                release_first.wait(15)
            elif not tail_taken.is_set() and any(
                item[0] == task_b for item in frame.f_locals["batch"]
            ):
                tail_taken.set()
                release_tail.wait(15)
        elif code is th._flush_target.__code__ and kind == "return":
            if frame.f_locals["task_id"] == task_b:
                targets.append(arg)
                target_fixed.set()
        elif code is th._release_batch_locked.__code__ and kind == "line":
            if (frame.f_lineno == boundary_line and not dying.is_set()
                    and (pops == th._BATCH_MAX or frame.f_locals.get("_") == pops)):
                dying.set()
                release_death.wait(15)
        elif code is th._left_the_queue.__code__ and kind == "return" and tail_taken.is_set():
            rechecks.append(arg)
            rechecked.set()
        return trace

    counts: list[int] = []
    errors: list[BaseException] = []

    def flush_b() -> None:
        """Flush task B and record how many of its events were persisted at that moment."""
        try:
            th._flush_chat_events(task_b)
            counts.append(_event_count(task_b))
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            errors.append(exc)

    def enqueue() -> None:
        """Queue one full batch of A, then B, in one go: B is in the writer's second batch."""
        for _ in range(th._BATCH_MAX):
            th._queue_chat_event(event, task_a)
        th._queue_chat_event(event, task_b)

    flusher = threading.Thread(target=flush_b, name="r2-flush-b", daemon=True)
    threading.settrace(trace)
    try:
        locked_call(th._pending_cond, enqueue)
        assert first_taken.wait(10), "the writer never took its first batch"
        flusher.start()
        assert target_fixed.wait(10), "the flush did not fix its target"
        # B is the 257th event queued and none has been popped yet.
        assert targets == [popped + th._BATCH_MAX + 1]
        release_first.set()
        assert dying.wait(10), "the writer never reached the death boundary"
        writer = th._event_writer_thread
        assert writer is not None and writer.ident is not None
        assert inject_keyboard_interrupt(writer.ident) == 1
        release_death.set()
        writer.join(10)
        assert not writer.is_alive(), "the injected Stop did not end the writer"
        assert tail_taken.wait(10), "no replacement writer took the batch holding B"
        assert rechecked.wait(10), "the flusher did not re-check its target"
        # B is persisted by nobody yet: the flusher must still be waiting.
        assert counts == [] and not errors, (counts, errors)
        assert _event_count(task_b) == 0 and th._pending_count(task_b) == 1
        assert rechecks and not any(rechecks), rechecks
    finally:
        release_first.set()
        release_death.set()
        release_tail.set()
        threading.settrace(None)
    flusher.join(20)
    assert not flusher.is_alive(), "the flush of task B hung"
    assert counts == [1], f"the flush returned with task B's events persisted: {counts}"
    assert not errors, errors
    assert _event_count(task_a) >= th._BATCH_MAX  # re-persisted by the replacement writer
    th._flush_chat_events()
    assert th._pending_count(None) == 0
