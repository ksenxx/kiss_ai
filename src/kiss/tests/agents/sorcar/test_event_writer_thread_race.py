# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Stress test: event-writer globals must be read via a snapshot.

``_queue_chat_event`` / ``_flush_chat_events`` historically read the
module-global ``_event_writer_thread`` twice::

    if _event_writer_thread is None or not _event_writer_thread.is_alive():

A concurrent ``_stop_event_writer()`` nulling the global between the
``None``-check and the ``.is_alive()`` attribute access raised
``AttributeError: 'NoneType' object has no attribute 'is_alive'`` on the
hot event path.  Additionally ``_flush_chat_events`` returned early
(events still queued) when the writer thread had not started yet,
breaking its "block until persisted" contract.

This test hammers ``_queue_chat_event`` + ``_flush_chat_events`` from
five producer threads while a disruptor thread repeatedly stops the
writer, then verifies no thread raised and every queued event was
persisted.
"""

from __future__ import annotations

import random
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
    _queue_chat_event,
)

_EVENTS_PER_THREAD = 40
_PRODUCER_THREADS = 5


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


class TestEventWriterThreadRace:
    """Concurrent queue/flush/stop must never raise or lose events."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        th._stop_event_writer()

    def teardown_method(self) -> None:
        th._stop_event_writer()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_five_thread_stress_no_attribute_error_no_lost_events(self) -> None:
        task_id, _ = _add_task("event writer race stress task")
        errors: list[BaseException] = []
        stop_disruptor = threading.Event()

        def producer(worker: int) -> None:
            try:
                for i in range(_EVENTS_PER_THREAD):
                    time.sleep(random.uniform(0, 0.005))
                    _queue_chat_event(
                        {"type": "text_delta", "text": f"w{worker}-e{i}"},
                        task_id,
                    )
                    if i % 10 == 0:
                        _flush_chat_events()
                _flush_chat_events()
            except BaseException as exc:  # noqa: BLE001 - collected for assert
                errors.append(exc)

        def disruptor() -> None:
            try:
                while not stop_disruptor.is_set():
                    time.sleep(random.uniform(0, 0.01))
                    th._stop_event_writer()
            except BaseException as exc:  # noqa: BLE001 - collected for assert
                errors.append(exc)

        threads = [
            threading.Thread(target=producer, args=(w,))
            for w in range(_PRODUCER_THREADS)
        ]
        d = threading.Thread(target=disruptor)
        for t in threads:
            t.start()
        d.start()
        for t in threads:
            t.join(timeout=60)
        stop_disruptor.set()
        d.join(timeout=60)

        assert not errors, f"threads raised: {errors!r}"
        _flush_chat_events()
        chat = _load_chat_events_by_task_id(task_id)
        assert chat is not None
        events = cast(list[dict[str, object]], chat["events"])
        assert len(events) == _PRODUCER_THREADS * _EVENTS_PER_THREAD

    def test_flush_blocks_until_persisted_when_writer_not_started(self) -> None:
        """Events queued before the writer ever started must still be
        persisted by ``_flush_chat_events`` (no early return)."""
        task_id, _ = _add_task("flush before writer start task")
        th._stop_event_writer()
        # Queue directly, bypassing the auto-start in ``_queue_chat_event``,
        # to reconstruct the "writer never started" state.
        import json as _json

        th._event_queue.put(
            (task_id, _json.dumps({"type": "text_delta", "text": "x"}),
             time.time(), th._current_db_path())
        )
        assert th._event_writer_thread is None
        _flush_chat_events()
        chat = _load_chat_events_by_task_id(task_id)
        assert chat is not None
        events = cast(list[dict[str, object]], chat["events"])
        assert len(events) == 1
