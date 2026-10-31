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
and counts from the main thread how often a writer holds the write lock
while the loader thread is inside ``json.loads`` (a ``sys.setprofile``
hook on that thread tells).  With the decode under the read lock that
never happens, however the threads are scheduled; with the decode
outside it the writer gets in hundreds of times.  The writer only yields
between tries (``time.sleep(0)``) and the GIL switch interval is cut to
:data:`SWITCH_INTERVAL` while it probes: on an idle Mac timer coalescing
stretches every timed wait ninefold, so ``sleep(0.001)`` lasts 9 ms and
the default 5 ms switch interval hands the GIL back only every 45 ms,
which leaves the writer about one try per load.  A count, not a
wall-time ratio, so the test holds under a loaded CI box.

Everything is real: the real database, lock, writer thread and loaders.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import persistence

EVENTS = 1500
LOADS = 3
#: Writer acquisitions during a ``json.loads`` of the loader that prove the
#: lock was free during the decode: a decode under the lock allows none.
MIN_WRITER_ACQUISITIONS = 5 * LOADS
#: GIL switch interval while the writer probes (the default is 5 ms).
SWITCH_INTERVAL = 0.0005


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


class DecodeWatch:
    """``sys.setprofile`` hook of the loader thread: ``active`` while ``json.loads`` runs."""

    def __init__(self) -> None:
        self.active = False

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        """Track the ``call`` and ``return`` events of ``json.loads`` frames."""
        if frame.f_code is json.loads.__code__ and event in ("call", "return"):
            self.active = event == "call"


def run_loads(load: Callable[[], Any], loaded: list[Any], watch: DecodeWatch) -> None:
    """Thread body: run *load* ``LOADS`` times back to back under *watch*, collecting results."""
    sys.setprofile(watch)
    try:
        for _ in range(LOADS):
            loaded.append(load())
    finally:
        sys.setprofile(None)


def writer_acquisitions_during(load: Callable[[], Any]) -> tuple[int, float, list[Any]]:
    """Return ``(acquisitions, longest wait, results)`` of a probing writer while *load* runs.

    Only the acquisitions made while the loader is inside ``json.loads``
    count: a loader that decodes under the read lock keeps the writer out
    for the whole decode, so for it the count is zero whatever the timing.
    """
    loaded: list[Any] = []
    watch = DecodeWatch()
    loader = threading.Thread(target=run_loads, args=(load, loaded, watch))
    acquisitions = 0
    longest = 0.0
    default_interval = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_INTERVAL)
    try:
        loader.start()
        while loader.is_alive():
            start = time.perf_counter()
            with persistence._rw_lock.write_lock():
                acquisitions += watch.active
            longest = max(longest, time.perf_counter() - start)
            time.sleep(0)
        loader.join()
    finally:
        sys.setswitchinterval(default_interval)
    return acquisitions, longest, loaded


def assert_lock_released_for_decode(load: Callable[[], Any]) -> list[Any]:
    """Run *load* under a probing writer and assert the lock covers only the SQL."""
    acquisitions, longest, loaded = writer_acquisitions_during(load)
    assert acquisitions >= MIN_WRITER_ACQUISITIONS, (
        f"a writer got the lock only {acquisitions} times while {LOADS} loads decoded JSON "
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
