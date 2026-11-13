# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The finish save scans a task's trajectory for secrets before it locks the database.

``_save_task_extra`` with an ``endTs`` derives the finished task's tags,
and the ``secret`` tag needs a regex scan of every persisted event of the
task (seconds for a long task).  The scan ran inside the save's
``_rw_lock.write_lock()`` and ``BEGIN IMMEDIATE`` transaction, so for its
whole duration every reader and writer of the daemon, and every other
process writing ``history.db``, waited.  The tests probe both locks at
the moment the scan's ``SELECT`` starts, and check that the verdict
still reaches the ``tags`` column.

Scanning outside the transaction opens a window: an event committed
after the scan and before the save's ``BEGIN IMMEDIATE`` (a journal
replay by another thread or process) was never scanned, and the
finished row kept a stale clean verdict.  The save therefore records
the newest event id before the scan and, in its transaction, checks
that no event landed after it; the late-event tests drive that window
with a second connection.  The events that did land (a whole deferred
transcript when a journal replay was waiting behind a peer's lock) are
scanned with both locks free as well, and the transaction retried: the
replay test journals a multi-megabyte transcript and replays it, for
real, between the finalizer's scan and its write.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from collections.abc import Callable
from contextlib import closing
from functools import partial
from types import FrameType

import kiss.agents.sorcar.persistence as th
from kiss.core.file_lock import lock_exclusive, unlock

_SCAN_PREFIX = "SELECT event_json FROM events"
_SCAN_SUFFIX = "ORDER BY seq, id"
_TOKEN = "ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"


def _insert_events(task_id: str, contents: list[str]) -> None:
    """Persist one ``tool_result`` event per entry of *contents* for *task_id*."""
    with closing(sqlite3.connect(th._DB_PATH)) as conn, conn:
        for seq, content in enumerate(contents):
            conn.execute(
                "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, ?)",
                (task_id, seq, json.dumps({"type": "tool_result", "content": content}), 1.0),
            )


def _read_lock_once(done: threading.Event) -> None:
    """Take and release the daemon's read lock, then set *done*."""
    with th._rw_lock.read_lock():
        done.set()


def _probe_locks(other: sqlite3.Connection, seen: list[str], statement: str) -> None:
    """Trace callback: when the secret scan starts, record which locks are free.

    Records ``writer free``/``writer locked`` for another connection's
    ``BEGIN IMMEDIATE`` (another process) and ``reader free``/``reader
    blocked`` for a thread taking ``_rw_lock.read_lock()`` (the daemon's
    history panel, transcript loads).  Every read of the task's events
    counts: the scan of the whole transcript and the scan of the events
    that landed after it alike.
    """
    if not statement.startswith(_SCAN_PREFIX):
        return
    try:
        other.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError:
        seen.append("writer locked")
    else:
        other.execute("ROLLBACK")
        seen.append("writer free")
    done = threading.Event()
    reader = threading.Thread(target=_read_lock_once, args=(done,), daemon=True)
    reader.start()
    seen.append("reader free" if done.wait(2.0) else "reader blocked")


def _tags(task_id: str) -> list[str]:
    """Return the stored ``tags`` of *task_id*."""
    with closing(sqlite3.connect(th._DB_PATH)) as conn:
        row = conn.execute("SELECT tags FROM task_history WHERE id = ?", (task_id,)).fetchone()
    return str(row[0] or "").split(",")


def _traced(run: Callable[[], object]) -> list[str]:
    """Run *run* on this thread with the lock probe on its connection; return what it saw."""
    seen: list[str] = []
    db = th._get_db()
    with closing(sqlite3.connect(th._DB_PATH, timeout=0, isolation_level=None)) as other:
        db.set_trace_callback(partial(_probe_locks, other, seen))
        try:
            run()
        finally:
            db.set_trace_callback(None)
    return seen


def _finish_traced(task_id: str | None, extra: dict[str, object]) -> list[str]:
    """Run ``_save_task_extra(extra, task_id)`` with the lock probe; return what it saw."""
    return _traced(partial(th._save_task_extra, extra, task_id))


def _commit_between_scan_and_txn(
    peer: sqlite3.Connection, task_id: str, seen: list[str], statement: str
) -> None:
    """Trace callback: land a secret-bearing event after the scan, before the write.

    A peer (a journal replay in another thread or process) begins its
    write when the scan's ``SELECT`` starts — its uncommitted row is
    invisible to the scan's WAL snapshot — and commits it when the
    save's own ``BEGIN IMMEDIATE`` starts, so the event is in the
    database before the transaction that writes the tags.  The save's
    scan of that event (a second ``SELECT``) lands nothing more.
    """
    if not seen and statement.startswith(_SCAN_PREFIX) and statement.endswith(_SCAN_SUFFIX):
        peer.execute("BEGIN IMMEDIATE")
        peer.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, ?)",
            (task_id, 99, json.dumps({"type": "tool_result", "content": f"key {_TOKEN}"}), 2.0),
        )
        seen.append("scan")
    elif statement == "BEGIN IMMEDIATE" and seen == ["scan"]:
        peer.execute("COMMIT")
        seen.append("committed before txn")


def _run_with_late_event(task_id: str, finish: Callable[[], object]) -> list[str]:
    """Run *finish* here while a peer commits a secret event between its scan and its write."""
    seen: list[str] = []
    db = th._get_db()
    with closing(sqlite3.connect(th._DB_PATH, timeout=5, isolation_level=None)) as peer:
        db.set_trace_callback(partial(_commit_between_scan_and_txn, peer, task_id, seen))
        try:
            finish()
        finally:
            db.set_trace_callback(None)
    return seen


def _pause_after_scan(
    scanned: threading.Event, resume: threading.Event, frame: FrameType, event: str, arg: object
) -> None:
    """Profile hook of the finalizer thread: pause once its unlocked scan has returned."""
    if event == "return" and frame.f_code is th._scan_finished_trajectory.__code__:
        scanned.set()
        assert resume.wait(30)


def _finish_paused_after_scan(
    task_id: str, scanned: threading.Event, resume: threading.Event, seen: list[str]
) -> None:
    """Finalizer thread: save ``endTs`` with the lock probe, pausing after the scan."""
    sys.setprofile(partial(_pause_after_scan, scanned, resume))
    try:
        seen.extend(_finish_traced(task_id, {"endTs": 1_900_000_000_000}))
    finally:
        sys.setprofile(None)
        th._close_thread_db()


def test_a_transcript_replayed_after_the_scan_is_rescanned_with_both_locks_free() -> None:
    """A journal replay landing between the scan and the write is not scanned under the locks."""
    task_id, _ = th._add_task("list the files")
    th._save_task_result("done", task_id, success=True)
    big = json.dumps({"type": "tool_result", "content": "ordinary output " * 16_384})
    secret = json.dumps({"type": "tool_result", "content": f"export GITHUB_TOKEN={_TOKEN}"})
    th._journal_failed_events(
        [(task_id, big, float(seq), str(th._DB_PATH)) for seq in range(32)]
        + [(task_id, secret, 32.0, str(th._DB_PATH))],
        4,
    )
    scanned, resume = threading.Event(), threading.Event()
    seen: list[str] = []
    finalizer = threading.Thread(
        target=_finish_paused_after_scan, args=(task_id, scanned, resume, seen), daemon=True
    )
    # A peer process holds the journal: the save's own flush cannot replay the
    # journal before its scan (it gives up after _JOURNAL_LOCK_TIMEOUT_S), so
    # the transcript lands only once the scan is over and the lock released.
    held = open(th._failed_events_path(str(th._DB_PATH)) + ".lock", "a+")  # noqa: SIM115
    assert lock_exclusive(held, timeout=1)
    finalizer.start()
    try:
        assert scanned.wait(30)
        unlock(held)
        held.close()
        th._replay_failed_events()  # lands the 33 events; the row is not finished yet
        with closing(sqlite3.connect(th._DB_PATH)) as conn:
            count, end_ts = conn.execute(
                "SELECT (SELECT COUNT(*) FROM events WHERE task_id = ?), end_ts "
                "FROM task_history WHERE id = ?",
                (task_id, task_id),
            ).fetchone()
        assert (count, end_ts) == (33, 0)
    finally:
        resume.set()
        finalizer.join(60)
        held.close()
    assert not finalizer.is_alive()

    # The scan of the replayed events held neither lock, like the first one.
    assert seen == ["writer free", "reader free"] * 2
    assert "secret" in _tags(task_id)


def _pause_after_each_scan(
    scanned: threading.Event, resume: threading.Event, frame: FrameType, event: str, arg: object
) -> None:
    """Profile hook of the finalizer thread: pause after each scan that found no secret.

    The first scan (``_scan_finished_trajectory``) and every scan of the
    events that landed since (``_secret_since_scan``) pause; the scan
    that returns a ``secret`` verdict is the last one, so it does not.
    """
    if event != "return" or frame.f_code not in (
        th._scan_finished_trajectory.__code__, th._secret_since_scan.__code__
    ):
        return
    if isinstance(arg, tuple) and not arg[1]:
        resume.clear()
        scanned.set()
        assert resume.wait(30)


def _finalize_paused_after_each_scan(
    finalize: Callable[[], object], scanned: threading.Event, resume: threading.Event,
    seen: list[str],
) -> None:
    """Finalizer thread: run *finalize* with the lock probe, pausing after each clean scan."""
    sys.setprofile(partial(_pause_after_each_scan, scanned, resume))
    try:
        seen.extend(_traced(finalize))
    finally:
        sys.setprofile(None)
        th._close_thread_db()


def _overtake_every_scan(task_id: str, finalize: Callable[[], object]) -> list[str]:
    """Run *finalize* on a thread and replay a transcript after each of its clean scans.

    Three replays of one clean event each overtake the first scan and
    its first two rescans; the fourth replay, 33 events of 8 MB with
    the secret last, overtakes the third rescan.  Returns what the lock
    probe saw at every scan of the finalizer's connection.
    """
    scanned, resume = threading.Event(), threading.Event()
    seen: list[str] = []
    worker = threading.Thread(
        target=_finalize_paused_after_each_scan, args=(finalize, scanned, resume, seen),
        daemon=True,
    )
    batches = [["clean output"]] * 3 + [
        ["ordinary output " * 16_384] * 32 + [f"export GITHUB_TOKEN={_TOKEN}"]
    ]
    worker.start()
    try:
        for round_, contents in enumerate(batches):
            assert scanned.wait(30)
            scanned.clear()
            th._journal_failed_events(
                [
                    (task_id, json.dumps({"type": "tool_result", "content": content}),
                     float(round_ * 100 + seq), str(th._DB_PATH))
                    for seq, content in enumerate(contents)
                ],
                4,
            )
            th._replay_failed_events()
            resume.set()
    finally:
        resume.set()
        worker.join(120)
    assert not worker.is_alive()
    return seen


def test_replays_overtaking_every_rescan_of_a_final_save_are_scanned_with_both_locks_free() -> None:
    """However many replays overtake the save's rescans, none is scanned under a lock."""
    task_id, _ = th._add_task("list the files")
    th._save_task_result("done", task_id, success=True)

    seen = _overtake_every_scan(
        task_id, partial(th._save_task_extra, {"endTs": 1_900_000_000_000}, task_id)
    )

    # The first scan, three rescans of one event and the rescan of the 8 MB
    # transcript: every one of them held neither lock.
    assert seen == ["writer free", "reader free"] * 5
    assert "secret" in _tags(task_id)


def test_replays_overtaking_every_rescan_of_a_refresh_are_scanned_with_both_locks_free() -> None:
    """``_refresh_finished_rows`` retries the same way, without a locked last scan."""
    task_id, _ = th._add_task("list the files")
    th._save_task_result("done", task_id, success=True)
    th._save_task_extra({"endTs": 1_900_000_000_000}, task_id)
    assert _tags(task_id) == ["work"]

    seen = _overtake_every_scan(task_id, partial(th._refresh_finished_rows, {task_id}))

    assert seen == ["writer free", "reader free"] * 5
    assert "secret" in _tags(task_id)


def test_an_event_committed_between_the_scan_and_the_write_still_tags_secret() -> None:
    """A replay landing after the unlocked scan must not leave a stale clean verdict."""
    task_id, _ = th._add_task("list the files")
    _insert_events(task_id, ["plain output with no credential"] * 5)
    th._save_task_result("done", task_id, success=True)

    seen = _run_with_late_event(
        task_id, partial(th._save_task_extra, {"endTs": 1_900_000_000_000}, task_id)
    )

    assert seen == ["scan", "committed before txn"]
    assert "secret" in _tags(task_id)


def test_a_replay_refresh_rescans_events_committed_after_its_scan() -> None:
    """``_refresh_finished_rows`` (after a journal replay) guards its scan the same way."""
    task_id, _ = th._add_task("list the files")
    _insert_events(task_id, ["plain output with no credential"] * 5)
    th._save_task_result("done", task_id, success=True)
    th._save_task_extra({"endTs": 1_900_000_000_000}, task_id)
    assert _tags(task_id) == ["work"]

    seen = _run_with_late_event(task_id, partial(th._refresh_finished_rows, {task_id}))

    assert seen == ["scan", "committed before txn"]
    assert "secret" in _tags(task_id)


def test_secret_scan_runs_with_both_locks_free_and_tags_the_task() -> None:
    task_id, _ = th._add_task("list the files")
    _insert_events(task_id, ["x" * 200] * 20 + [f"export GITHUB_TOKEN={_TOKEN}"])
    th._save_task_result("done", task_id, success=True)

    seen = _finish_traced(task_id, {"endTs": 1_900_000_000_000, "steps": 3})

    assert seen == ["writer free", "reader free"]
    assert "secret" in _tags(task_id)


def test_a_clean_trajectory_is_scanned_once_and_not_tagged_secret() -> None:
    task_id, _ = th._add_task("list the files")
    _insert_events(task_id, ["plain output with no credential"] * 5)
    th._save_task_result("done", task_id, success=True)

    seen = _finish_traced(task_id, {"endTs": 1_900_000_000_000})

    assert seen == ["writer free", "reader free"]
    assert _tags(task_id) == ["work"]


def test_saves_without_end_ts_or_without_a_row_scan_nothing() -> None:
    task_id, _ = th._add_task("list the files")
    _insert_events(task_id, [f"token {_TOKEN}"])

    assert _finish_traced(task_id, {"steps": 2}) == []
    assert _finish_traced(None, {"endTs": 1_900_000_000_000}) == []
    assert _finish_traced("0" * 32, {"endTs": 1_900_000_000_000}) == []
    with closing(sqlite3.connect(th._DB_PATH)) as conn:
        row = conn.execute(
            "SELECT steps, end_ts, tags FROM task_history WHERE id = ?", (task_id,)
        ).fetchone()
    assert row == (2, 0, "")
