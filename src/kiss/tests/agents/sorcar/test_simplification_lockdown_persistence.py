# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Simplification-lockdown tests for ``kiss.agents.sorcar.persistence``.

These are characterization tests that pin the CURRENT externally-observable
behavior of ``persistence.py`` so that the planned simplifications (see
``tmp/findings-2.md`` sections A4, A5, A6, A7, A8, A10, A12) cannot silently
break it.  Locked-down contracts:

1. ``_close_db()`` reconnect path: rows written before and after a
   ``_close_db()`` are all visible, and the event seq / has_events caches
   do not leak across the close (seqs stay strictly increasing and unique).
2. Interleaved ``_queue_chat_event`` + ``_append_chat_event`` for one task
   yields strictly increasing unique seqs with no losses, and
   ``_append_chat_event`` persists previously queued events before returning.
3. ``_save_task_result`` overwrites the ``"Agent Failed Abruptly"`` sentinel;
   ``_save_task_extra`` round-trips JSON; both honor the flush-before-write
   contract with respect to queued events.
4. ``_get_task_chat_id(task_id)`` returns the row's chat_id and ``""``
   for missing rows.
5. ``has_events`` is set to 1 via BOTH the synchronous ``_append_chat_event``
   path and the queued ``_queue_chat_event`` + ``_flush_chat_events`` path.
6. ``_load_chat_context_text`` cache freshness: writes after a cached read
   are reflected on the next read, including after a global invalidation.
7. ``_delete_task`` removes the row AND its events and returns False for a
   missing id; ``_delete_frequent_task`` returns True only when the row
   existed.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th


def _redirect(tmpdir: str):
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


def _event_seqs(task_id: int) -> list[int]:
    """Return the raw ``seq`` values for *task_id* in insertion order."""
    with th._rw_lock.read_lock():
        db = th._get_db()
        rows = db.execute(
            "SELECT seq FROM events WHERE task_id = ? ORDER BY seq",
            (task_id,),
        ).fetchall()
        return [r["seq"] for r in rows]


def _history_row(task_id: str) -> dict:
    """Return the ``_load_history`` entry matching *task_id*."""
    rows = [r for r in th._load_history() if r["id"] == task_id]
    assert len(rows) == 1
    return rows[0]


def _session_events(session: dict[str, object] | None) -> list[dict[str, object]]:
    """Return the ``events`` list of a non-None replay session dict."""
    assert session is not None
    events = session["events"]
    assert isinstance(events, list)
    return events


def _assert_strictly_increasing_unique(seqs: list[int]) -> None:
    assert len(seqs) == len(set(seqs)), f"duplicate seqs: {seqs}"
    assert seqs == sorted(seqs), f"seqs not increasing: {seqs}"
    for a, b in zip(seqs, seqs[1:]):
        assert b > a, f"seqs not strictly increasing: {seqs}"


class _PersistenceTestBase:
    """Shared tempdir DB redirection (mirrors test_favorite_task.py)."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        th._close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestCloseDbReconnect(_PersistenceTestBase):
    """(1) ``_close_db()`` reconnect path and cache hygiene."""

    def test_rows_survive_close_db_and_reconnect(self) -> None:
        th._add_task("before close")
        th._close_db()
        th._add_task("after close")

        rows = th._load_history()
        tasks = {r["task"] for r in rows}
        assert tasks == {"before close", "after close"}
        # Most-recent-first ordering preserved across the reconnect.
        assert rows[0]["task"] == "after close"

    def test_seq_cache_does_not_leak_across_close_db(self) -> None:
        task_id, _ = th._add_task("seq cache task")
        th._queue_chat_event({"type": "agent_text", "text": "e0"}, task_id)
        th._flush_chat_events()
        th._close_db()
        # Synchronous append after the close must re-seed the seq counter
        # from the database, not restart at 0 (which would collide).
        th._append_chat_event(
            {"type": "agent_text", "text": "e1"}, task_id=task_id
        )
        th._queue_chat_event({"type": "agent_text", "text": "e2"}, task_id)
        th._flush_chat_events()

        seqs = _event_seqs(task_id)
        assert len(seqs) == 3
        _assert_strictly_increasing_unique(seqs)

        events = _session_events(th._load_chat_events_by_task_id(task_id))
        assert [e["text"] for e in events] == ["e0", "e1", "e2"]


class TestInterleavedQueueAndAppend(_PersistenceTestBase):
    """(2) Interleaved async queue + sync append seq contract."""

    def test_interleaved_seqs_strictly_increasing_no_losses(self) -> None:
        task_id, _ = th._add_task("interleave task")
        total = 24
        for i in range(total):
            event: dict[str, object] = {"type": "agent_text", "text": f"ev{i}"}
            if i % 2 == 0:
                th._queue_chat_event(event, task_id)
            else:
                th._append_chat_event(event, task_id=task_id)
        th._flush_chat_events()

        seqs = _event_seqs(task_id)
        assert len(seqs) == total
        _assert_strictly_increasing_unique(seqs)

        events = _session_events(th._load_chat_events_by_task_id(task_id))
        texts = [e["text"] for e in events]
        assert texts == [f"ev{i}" for i in range(total)]
        # Replay injects a ``_timestamp`` on every event.
        assert all("_timestamp" in e for e in events)

    def test_append_persists_previously_queued_events(self) -> None:
        task_id, _ = th._add_task("append flushes queue")
        for i in range(5):
            th._queue_chat_event(
                {"type": "agent_text", "text": f"queued{i}"}, task_id
            )
        # No explicit flush: _append_chat_event itself must drain the
        # queue so the sync event lands AFTER all queued ones.
        th._append_chat_event(
            {"type": "agent_text", "text": "sync"}, task_id=task_id
        )

        with th._rw_lock.read_lock():
            db = th._get_db()
            events = th._fetch_events_for_task_id(db, task_id)
        texts = [e["text"] for e in events]
        assert texts == ["queued0", "queued1", "queued2", "queued3",
                         "queued4", "sync"]
        _assert_strictly_increasing_unique(_event_seqs(task_id))


class TestSaveResultAndExtra(_PersistenceTestBase):
    """(3) Sentinel overwrite, extra JSON round-trip, flush-before-write."""

    def test_add_task_writes_sentinel_and_save_result_overwrites(self) -> None:
        task_id, _ = th._add_task("sentinel task")
        assert _history_row(task_id)["result"] == "Agent Failed Abruptly"

        th._save_task_result("All done.", task_id=task_id)
        assert _history_row(task_id)["result"] == "All done."

    def test_save_result_after_queued_events_flushes_first(self) -> None:
        task_id, _ = th._add_task("flush before result")
        for i in range(4):
            th._queue_chat_event(
                {"type": "agent_text", "text": f"q{i}"}, task_id
            )
        th._save_task_result("result after queue", task_id=task_id)

        # The queued events must already be persisted (no explicit flush).
        seqs = _event_seqs(task_id)
        assert len(seqs) == 4
        _assert_strictly_increasing_unique(seqs)
        assert _history_row(task_id)["result"] == "result after queue"

    def test_save_task_extra_round_trips_json(self) -> None:
        task_id, _ = th._add_task("extra task")
        th._queue_chat_event({"type": "agent_text", "text": "q"}, task_id)
        # Only known flat-column keys round-trip in the new schema; an
        # arbitrary "nested" key is silently dropped by ``_save_task_extra``.
        extra = {"model": "m1", "tokens": 123, "cost": 0.5}
        th._save_task_extra(extra, task_id=task_id)

        # Queued event persisted before the extra write.
        assert len(_event_seqs(task_id)) == 1
        session = th._load_chat_events_by_task_id(task_id)
        assert session is not None
        # r3-H3: ``_row_to_extra_json`` emits every typed column
        # consistently.  Pop the defaulted ones; the assertion is
        # that the explicitly-written keys round-trip.
        loaded = json.loads(str(session["extra"]))
        for k in (
            "auto_commit_mode", "is_parallel", "is_worktree",
            "work_dir", "version", "steps", "startTs", "endTs",
            "is_favorite",
        ):
            loaded.pop(k, None)
        assert loaded == extra
        # Row in history table mirrors the same flat-column shape.
        # r3-H3: pop every defaulted key consistently.
        row_extra = json.loads(str(_history_row(task_id)["extra"]))
        for k in (
            "auto_commit_mode", "is_parallel", "is_worktree",
            "work_dir", "version", "steps", "startTs", "endTs",
            "is_favorite",
        ):
            row_extra.pop(k, None)
        assert row_extra == extra


class TestChatIdLookups(_PersistenceTestBase):
    """(4) ``_get_task_chat_id`` contract."""

    def test_get_task_chat_id_returns_row_chat(self) -> None:
        old_id, old_chat = th._add_task("repeated task")
        time.sleep(0.02)  # distinct timestamps for ORDER BY timestamp DESC
        new_id, new_chat = th._add_task("repeated task")
        assert old_chat != new_chat

        assert th._get_task_chat_id(old_id) == old_chat
        assert th._get_task_chat_id(new_id) == new_chat

    def test_lookups_return_empty_for_missing(self) -> None:
        th._add_task("present task")
        assert th._get_task_chat_id(999_999) == ""


class TestHasEventsFlag(_PersistenceTestBase):
    """(5) ``has_events`` set via both write paths."""

    def test_sync_append_sets_has_events(self) -> None:
        task_id, _ = th._add_task("sync has_events")
        assert _history_row(task_id)["has_events"] == 0
        th._append_chat_event(
            {"type": "agent_text", "text": "x"}, task_id=task_id
        )
        assert _history_row(task_id)["has_events"] == 1

    def test_queued_path_sets_has_events(self) -> None:
        task_id, _ = th._add_task("queued has_events")
        assert _history_row(task_id)["has_events"] == 0
        th._queue_chat_event({"type": "agent_text", "text": "x"}, task_id)
        th._flush_chat_events()
        assert _history_row(task_id)["has_events"] == 1


class TestChatContextCacheFreshness(_PersistenceTestBase):
    """(6) ``_load_chat_context_text`` never serves stale text."""

    def test_writes_after_cached_read_are_visible(self) -> None:
        task_id, chat_id = th._add_task("ctx task one")
        first = th._load_chat_context_text(chat_id)
        assert "ctx task one" in first
        # Result written AFTER the cached read must appear on re-read.
        th._save_task_result("ctx result one", task_id=task_id)
        second = th._load_chat_context_text(chat_id)
        assert "ctx result one" in second
        # A new task in the same chat must also invalidate the cache.
        time.sleep(0.02)
        th._add_task("ctx task two", chat_id=chat_id)
        third = th._load_chat_context_text(chat_id)
        assert "ctx task one" in third
        assert "ctx result one" in third
        assert "ctx task two" in third

    def test_fresh_after_global_invalidation(self) -> None:
        task_id, chat_id = th._add_task("ctx global task")
        assert "ctx global task" in th._load_chat_context_text(chat_id)
        th._invalidate_chat_context_cache("")
        # Cache cleared globally: next read recomputes from the DB.
        assert "ctx global task" in th._load_chat_context_text(chat_id)
        # Read (re-caches) → global invalidate → write → read: fresh.
        th._invalidate_chat_context_cache("")
        th._save_task_result("ctx global result", task_id=task_id)
        assert "ctx global result" in th._load_chat_context_text(chat_id)
        # Empty chat_id always short-circuits to "".
        assert th._load_chat_context_text("") == ""


class TestDeletions(_PersistenceTestBase):
    """(7) ``_delete_task`` / ``_delete_frequent_task`` contracts."""

    def test_delete_task_removes_row_and_events(self) -> None:
        task_id, _ = th._add_task("doomed task")
        th._append_chat_event(
            {"type": "agent_text", "text": "ev"}, task_id=task_id
        )
        th._queue_chat_event({"type": "agent_text", "text": "ev2"}, task_id)

        assert th._delete_task(task_id) is True
        assert th._load_chat_events_by_task_id(task_id) is None
        assert _event_seqs(task_id) == []
        assert all(r["id"] != task_id for r in th._load_history())

    def test_delete_task_missing_returns_false(self) -> None:
        assert th._delete_task(424_242) is False
        task_id, _ = th._add_task("delete twice")
        assert th._delete_task(task_id) is True
        assert th._delete_task(task_id) is False

    def test_delete_frequent_task_true_only_when_existed(self) -> None:
        th._record_frequent_task("freq task")
        assert th._delete_frequent_task("freq task") is True
        assert th._delete_frequent_task("freq task") is False
        assert th._delete_frequent_task("never recorded") is False
        assert th._delete_frequent_task("") is False
        assert th._load_frequent_tasks() == []
