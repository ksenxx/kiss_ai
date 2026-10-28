"""The digest loaders and ``_add_task`` keep their slow work outside ``_rw_lock``.

``persistence._rw_lock`` is writer-preferring: while one writer waits,
every new reader waits too, so anything slow done under the lock stalls
the whole persistence layer of the process.  Two invariants of
``persistence.py`` are pinned here for the callers that used to break
them:

* ``_get_db()`` runs BEFORE the lock is taken.  On a thread's first
  call it opens a connection, runs the migration check (which can sit
  in SQLite's 30 s busy wait behind another process) and, after the
  ``-wal``/``-shm`` sidecars were replaced, the orphaned-sidecar
  recovery.  ``task_digest.load_task`` and ``child_tasks`` run on fresh
  ``/ask`` and task-update threads, so that is their every call.  The
  test holds the write lock from the main thread, runs the loader on a
  fresh thread and waits for the loader's connection to be registered
  in ``persistence._open_conns`` while the lock is still held: a loader
  that takes the lock first never gets there.
* The transcript is decoded AFTER the read lock is released.  The
  probe synchronises with the decoder itself: a corrupt event row makes
  ``_decode_event_rows`` log its DEBUG "Exception caught" record, and a
  real ``logging.Handler`` on the persistence logger parks the loader
  thread inside that call.  While it is parked a writer must get the
  lock at once; a decode under the read lock keeps the writer out until
  the decoder is released.  (A writer-acquisition count over whole
  loads would also pass when something slow BEFORE the lock — e.g. a
  migration check — gave the writer its turns.)
* ``_add_task`` publishes the owner liveness marker (``mkdir``,
  ``open``, ``flock``, ``write`` under ``<KISS_HOME>/task-owners/``)
  before taking the write lock.  Same shape as the first probe: the
  marker must appear while the main thread still holds the lock.

Everything is real: the real database, lock, connection registry,
marker directory and loaders.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence, task_digest
from kiss.tests.agents.sorcar.test_persistence_read_lock_stall import EVENTS, add_heavy_task

#: Upper bound on how long a probe waits for the step it expects to see
#: before the lock is taken.  Normally a few milliseconds.
PROBE_TIMEOUT_S = 10.0


@pytest.fixture
def kiss_home(tmp_path: Path) -> Iterator[Path]:
    """Point persistence at a throwaway KISS_HOME (no owner marker yet) and restore it."""
    saved = (persistence._KISS_DIR, persistence._DB_PATH)
    persistence._KISS_DIR = tmp_path
    persistence._DB_PATH = tmp_path / "history.db"
    try:
        yield tmp_path
    finally:
        persistence._close_db()
        persistence._KISS_DIR, persistence._DB_PATH = saved


def _threads_with_open_conns() -> set[threading.Thread]:
    """Return the threads that currently own a registered connection."""
    return {owner for _conn, owner, _path in list(persistence._open_conns.values())}


def _collect(body: Callable[[], Any], result: list[Any]) -> None:
    """Thread body: append the return value of *body* to *result*."""
    result.append(body())


def _run_while_write_locked(
    body: Callable[[], Any], seen: Callable[[threading.Thread], bool],
) -> tuple[bool, Any]:
    """Run *body* on a fresh thread while this thread holds the write lock.

    Returns ``(observed, result)``: *observed* is whether ``seen(thread)``
    became true while the write lock was still held, i.e. whether *body*
    did the step *seen* watches for BEFORE blocking on the lock.
    """
    result: list[Any] = []
    worker = threading.Thread(target=_collect, args=(body, result))
    with persistence._rw_lock.write_lock():
        worker.start()
        deadline = time.monotonic() + PROBE_TIMEOUT_S
        observed = False
        while time.monotonic() < deadline:
            if seen(worker):
                observed = True
                break
            time.sleep(0.005)
    worker.join(timeout=PROBE_TIMEOUT_S)
    assert not worker.is_alive(), "the worker never got the lock after it was released"
    return observed, result[0]


def _owns_a_conn(worker: threading.Thread) -> bool:
    """Probe: *worker* has opened and registered its database connection."""
    return worker in _threads_with_open_conns()


def test_load_task_opens_its_connection_before_the_read_lock(kiss_home: Path) -> None:
    """A fresh thread's ``load_task`` has its connection before it waits for the lock."""
    task_id, _ = persistence._add_task("digest me", "")
    persistence._append_chat_event({"type": "agent_text", "text": "hi"}, task_id=task_id)

    observed, task = _run_while_write_locked(
        partial(task_digest.load_task, task_id), _owns_a_conn,
    )

    assert observed, "load_task took the read lock before opening its connection"
    assert task is not None and task["id"] == task_id
    assert [e["text"] for e in task["events"]] == ["hi"]


def test_load_task_of_unknown_id_opens_its_connection_before_the_read_lock(
    kiss_home: Path,
) -> None:
    """The ``None`` branch of ``load_task`` sits inside the lock and still returns."""
    observed, task = _run_while_write_locked(
        partial(task_digest.load_task, "nope"), _owns_a_conn,
    )

    assert observed
    assert task is None


def test_child_tasks_opens_its_connection_before_the_read_lock(kiss_home: Path) -> None:
    """A fresh thread's ``child_tasks`` has its connection before it waits for the lock."""
    parent_id, chat_id = persistence._add_task("parent", "")
    child_id, _ = persistence._add_task("child", chat_id, {"parent_task_id": parent_id})

    observed, rows = _run_while_write_locked(
        partial(task_digest.child_tasks, parent_id), _owns_a_conn,
    )

    assert observed, "child_tasks took the read lock before opening its connection"
    assert [r["id"] for r in rows] == [child_id]


class _ParkDecoder(logging.Handler):
    """Park the thread that logs the decoder's "Exception caught" record.

    ``entered`` is set from inside ``_decode_event_rows`` (the corrupt
    row's DEBUG record is emitted there and nowhere else on the load
    path); the thread then waits for ``resume``.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.entered = threading.Event()
        self.resume = threading.Event()

    def emit(self, record: logging.LogRecord) -> None:
        """Signal decoder entry and block until the test resumes the thread."""
        if record.getMessage() == "Exception caught":
            self.entered.set()
            self.resume.wait(PROBE_TIMEOUT_S)


def _take_write_lock(acquired: threading.Event) -> None:
    """Thread body: take and release the write lock, flagging the acquisition."""
    with persistence._rw_lock.write_lock():
        acquired.set()


def writer_acquires_while_decoding(load: Callable[[], Any]) -> tuple[bool, Any]:
    """Run *load* on a thread parked inside the decoder; try the write lock meanwhile.

    Returns ``(writer_got_lock, result)``.  The caller's task must carry
    one undecodable event row so the decoder logs its DEBUG record.
    """
    park = _ParkDecoder()
    level_before = persistence.logger.level
    persistence.logger.setLevel(logging.DEBUG)
    persistence.logger.addHandler(park)
    result: list[Any] = []
    loader = threading.Thread(target=_collect, args=(load, result))
    acquired = threading.Event()
    writer = threading.Thread(target=_take_write_lock, args=(acquired,))
    try:
        loader.start()
        assert park.entered.wait(PROBE_TIMEOUT_S), "the corrupt row never reached the decoder"
        writer.start()
        writer_got_lock = acquired.wait(2.0)
    finally:
        park.resume.set()
        loader.join(PROBE_TIMEOUT_S)
        if writer.ident is not None:
            writer.join(PROBE_TIMEOUT_S)
        persistence.logger.removeHandler(park)
        persistence.logger.setLevel(level_before)
    assert not loader.is_alive() and not writer.is_alive()
    return writer_got_lock, result[0]


def test_load_task_decodes_outside_the_read_lock(kiss_home: Path) -> None:
    """A writer gets the lock while ``load_task`` is inside the decoder."""
    task_id, _chat_id = add_heavy_task("")
    persistence._get_db().execute(
        "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, -1, ?, ?)",
        (task_id, "{not json", time.time()),
    )

    writer_got_lock, task = writer_acquires_while_decoding(
        partial(task_digest.load_task, task_id),
    )

    assert writer_got_lock, "the read lock was still held while the transcript was decoded"
    assert task is not None and task["id"] == task_id
    assert len(task["events"]) == EVENTS, "the corrupt row is dropped, the rest decoded"


def _owner_marker_published(kiss_home: Path, _worker: threading.Thread) -> bool:
    """Probe: this process's liveness marker exists under *kiss_home*."""
    owner_dir = kiss_home / persistence._OWNER_DIR_NAME
    return owner_dir.is_dir() and any(owner_dir.glob("*.lock"))


def test_add_task_publishes_the_owner_marker_before_the_write_lock(kiss_home: Path) -> None:
    """The first ``_add_task`` of a home mints the marker before it waits for the lock."""
    assert not (kiss_home / persistence._OWNER_DIR_NAME).exists()

    observed, (task_id, chat_id) = _run_while_write_locked(
        partial(persistence._add_task, "first of this home", ""),
        partial(_owner_marker_published, kiss_home),
    )

    assert observed, "_add_task took the write lock before publishing the owner marker"
    assert chat_id
    row = persistence._get_db().execute(
        "SELECT owner FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    assert row["owner"] and (
        kiss_home / persistence._OWNER_DIR_NAME / f"{row['owner']}.lock"
    ).is_file()
