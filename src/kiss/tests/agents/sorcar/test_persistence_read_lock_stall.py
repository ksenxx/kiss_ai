"""Transcript loaders must not hold the read lock while decoding JSON.

``_rw_lock`` is writer-preferring: while the event-writer thread (or a
synchronous ``_append_chat_event``) waits for the write lock, every new
reader waits too.  The transcript loaders used to run ``json.loads`` on
every event row — and ``json.dumps`` again for the byte budget — inside
``_rw_lock.read_lock()``, so one load of a large transcript stalled
every write in the process for its whole Python-side decode, not just
for the SQL fetch.

Each test below builds one task whose events are cheap to fetch but
expensive to decode, runs a loader back to back on a helper thread,
and counts from the main thread how often a writer gets the write lock
while the loads run.  With the decode under the read lock a writer only
gets in between two loads (at most two acquisitions per gap, so at most
``2 * LOADS``); with the decode outside it the writer gets in hundreds
of times during every load.  A count, not a wall-time ratio, so the
test holds under a loaded CI box.

Everything is real: the real database, lock, writer thread and loaders.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence

EVENTS = 1500
LOADS = 3
#: Writer acquisitions that prove the lock was free during the loads:
#: a decode under the lock allows at most ``2 * LOADS``.
MIN_WRITER_ACQUISITIONS = 5 * LOADS


@pytest.fixture
def kiss_home(tmp_path: Path) -> Iterator[Path]:
    """Point persistence at a throwaway KISS_HOME and restore it afterwards."""
    saved = (persistence._KISS_DIR, persistence._DB_PATH)
    persistence._KISS_DIR = tmp_path
    persistence._DB_PATH = tmp_path / "history.db"
    try:
        yield tmp_path
    finally:
        persistence._close_db()
        persistence._KISS_DIR, persistence._DB_PATH = saved


def heavy_event(i: int) -> dict[str, object]:
    """An event that is a few KB to transfer but hundreds of objects to decode."""
    return {
        "type": "tool_result",
        "text": f"result {i} " + "x" * 64,
        "data": [{"k": j, "v": "abc", "w": [1, 2, 3]} for j in range(120)],
    }


def add_heavy_task(chat_id: str, extra: dict[str, object] | None = None) -> tuple[str, str]:
    """Insert one task with ``EVENTS`` heavy events and return ``(task_id, chat_id)``."""
    task_id, chat_id = persistence._add_task("heavy task", chat_id, extra)
    for i in range(EVENTS):
        persistence._queue_chat_event(heavy_event(i), task_id)
    persistence._flush_chat_events(task_id)
    return task_id, chat_id


def run_loads(load: Callable[[], Any], loaded: list[Any]) -> None:
    """Thread body: run *load* ``LOADS`` times back to back, collecting the results."""
    for _ in range(LOADS):
        loaded.append(load())


def writer_acquisitions_during(load: Callable[[], Any]) -> tuple[int, float, list[Any]]:
    """Return ``(acquisitions, longest wait, results)`` of a probing writer while *load* runs."""
    loaded: list[Any] = []
    loader = threading.Thread(target=run_loads, args=(load, loaded))
    loader.start()
    acquisitions = 0
    longest = 0.0
    while loader.is_alive():
        start = time.perf_counter()
        with persistence._rw_lock.write_lock():
            acquisitions += 1
        longest = max(longest, time.perf_counter() - start)
        time.sleep(0.001)
    loader.join()
    return acquisitions, longest, loaded


def assert_lock_released_for_decode(load: Callable[[], Any]) -> list[Any]:
    """Run *load* under a probing writer and assert the lock covers only the SQL."""
    acquisitions, longest, loaded = writer_acquisitions_during(load)
    assert acquisitions >= MIN_WRITER_ACQUISITIONS, (
        f"a writer got the lock only {acquisitions} times during {LOADS} loads "
        f"(longest wait {longest:.3f}s): the read lock is held while the "
        "transcript is decoded"
    )
    return loaded


def test_load_all_chat_events_decodes_outside_the_read_lock(kiss_home: Path) -> None:
    """``_load_all_chat_events_by_chat_id`` (with a byte budget) decodes after the SELECTs."""
    task_id, chat_id = add_heavy_task("")

    def load() -> Any:
        return persistence._load_all_chat_events_by_chat_id(chat_id, max_json_bytes=10**9)

    for tasks, truncated in assert_lock_released_for_decode(load):
        assert not truncated
        assert [t["task_id"] for t in tasks] == [task_id]
        assert len(tasks[0]["events"]) == EVENTS
        assert tasks[0]["events"][7]["text"] == heavy_event(7)["text"]


def test_load_by_task_id_decodes_outside_the_read_lock(kiss_home: Path) -> None:
    """``_load_chat_events_by_task_id`` decodes after the SELECTs."""
    task_id, _chat_id = add_heavy_task("")

    def load() -> Any:
        return persistence._load_chat_events_by_task_id(task_id)

    for session in assert_lock_released_for_decode(load):
        assert session is not None and session["task_id"] == task_id
        assert len(session["events"]) == EVENTS


def test_load_subagent_rows_decodes_outside_the_read_lock(kiss_home: Path) -> None:
    """``_load_subagent_rows_by_parent_task_id`` decodes after the SELECTs."""
    parent_id, chat_id = persistence._add_task("parent", "")
    child_id, _ = add_heavy_task(chat_id, {"parent_task_id": parent_id})

    def load() -> Any:
        return persistence._load_subagent_rows_by_parent_task_id(parent_id)

    for rows in assert_lock_released_for_decode(load):
        assert [r["task_id"] for r in rows] == [child_id]
        assert len(rows[0]["events"]) == EVENTS


def test_adjacent_task_decodes_outside_the_read_lock(kiss_home: Path) -> None:
    """``_get_adjacent_task_by_chat_id`` decodes after the SELECTs."""
    heavy_id, chat_id = add_heavy_task("")
    time.sleep(0.01)
    current_id, _ = persistence._add_task("current", chat_id)

    def load() -> Any:
        return persistence._get_adjacent_task_by_chat_id(chat_id, current_id, "prev")

    for session in assert_lock_released_for_decode(load):
        assert session is not None and session["task_id"] == heavy_id
        assert len(session["events"]) == EVENTS
