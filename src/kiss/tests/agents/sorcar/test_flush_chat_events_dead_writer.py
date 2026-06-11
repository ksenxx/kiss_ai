# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: ``_flush_chat_events`` must not hang on a dead writer.

``_queue_chat_event`` (re)starts the background writer whenever the
writer thread is ``None`` OR no longer alive::

    if _event_writer_thread is None or not _event_writer_thread.is_alive():
        _start_event_writer()

``_flush_chat_events`` historically only checked ``is None`` before
calling ``_event_queue.join()``.  When the writer thread had died (for
example, it was stopped during a DB swap by ``_close_db`` /
``_stop_event_writer`` while items were still queued) the thread
reference is non-``None`` but ``is_alive()`` is ``False``.  In that state
``join()`` waits forever for a ``task_done()`` that the dead writer can
never call, hanging the calling worker thread.

This test reconstructs that exact state — a dead writer reference plus a
pending queued event — and asserts ``_flush_chat_events`` still returns
promptly (it must restart the writer to drain the backlog).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _flush_chat_events,
    _load_chat_events_by_task_id,
)


def _redirect(tmpdir: str) -> tuple:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore(saved: tuple) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


class TestFlushChatEventsDeadWriter:
    """``_flush_chat_events`` recovers when the writer thread has died."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        # Start from a clean writer/queue state.
        th._stop_event_writer()

    def teardown_method(self) -> None:
        th._stop_event_writer()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_flush_does_not_hang_when_writer_thread_dead(self) -> None:
        task_id, _ = _add_task("flush dead writer task")

        # Reconstruct the broken state: a non-None but dead writer thread
        # reference plus an event still queued (unfinished).  This is what
        # remains after the writer stops while items are in flight.
        dead = threading.Thread(target=lambda: None, name="dead-writer")
        dead.start()
        dead.join()
        assert not dead.is_alive()

        th._event_writer_stop.clear()
        th._event_writer_thread = dead
        event = {"type": "result", "content": "hello"}
        th._event_queue.put((
            task_id, json.dumps(event), time.time(), th._current_db_path(),
        ))
        assert th._event_queue.unfinished_tasks == 1

        # Without the fix this join() blocks forever; run it in a helper
        # thread and require it to finish within a generous timeout.
        done = threading.Event()

        def _run_flush() -> None:
            _flush_chat_events()
            done.set()

        worker = threading.Thread(target=_run_flush, name="flush-worker")
        worker.start()
        worker.join(timeout=10)

        assert done.is_set(), "_flush_chat_events hung on a dead writer thread"
        assert not worker.is_alive()

        # The previously-stranded event must now be persisted.
        loaded = _load_chat_events_by_task_id(task_id)
        assert loaded is not None
        events = cast("list[dict[str, object]]", loaded["events"])
        assert any(e.get("type") == "result" for e in events)
