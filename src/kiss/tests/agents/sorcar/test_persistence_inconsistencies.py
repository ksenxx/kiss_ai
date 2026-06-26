# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing four persistence.py inconsistencies.

Each test class targets one confirmed bug (see ``tmp/findings-1.md``):

* **A1** — ``_write_event_batch``: an event whose ``task_id`` row was
  deleted raised ``sqlite3.IntegrityError`` (FK violation) that aborted
  the ENTIRE batch, silently dropping events of every other task in
  that batch.  The code comment claimed dangling events are skipped but
  no validation happened.
* **A2** — ``_delete_task`` never invalidated the chat-context cache,
  so ghost-text autocomplete kept serving deleted task/result text.
* **A3** — ``_save_task_extra`` blind-overwrote the ``extra`` JSON
  column, destroying the ``is_favorite`` flag that
  ``_set_task_favorite`` merge-updates.
* **A4** — ``_prefix_match_task`` lacked the
  sub-agent exclusion filter used by ``_load_history`` /
  ``_search_history``, so their row indexing disagreed with the UI list
  and sub-agent internal task text leaked into autocomplete.

All tests run against a real SQLite database redirected to a temp dir
(same pattern as ``test_event_persistence_no_duplicate.py``).  No
mocks, patches, fakes, or test doubles are used.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _delete_task,
    _flush_chat_events,
    _load_chat_context_text,
    _load_chat_events_by_task_id,
    _load_history,
    _prefix_match_task,
    _queue_chat_event,
    _save_task_extra,
    _save_task_result,
    _set_task_favorite,
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


class _TempDbTestBase:
    """Shared fixture: fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._invalidate_chat_context_cache("")
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


def _read_extra(task_id: int) -> dict[str, object]:
    """Load the parsed ``extra`` JSON of a history row via the public reader."""
    for entry in _load_history():
        if entry["id"] == task_id:
            raw = entry["extra"] or "{}"
            parsed: dict[str, object] = json.loads(str(raw))
            return parsed
    raise AssertionError(f"task_id {task_id} not found in history")


class TestA1DanglingEventDoesNotDropBatch(_TempDbTestBase):
    """[A1] A deleted-task event must not abort the whole event batch."""

    def test_live_task_events_survive_dangling_event_in_same_batch(self) -> None:
        tid_live, _ = _add_task("live task")
        tid_doomed, _ = _add_task("doomed task")
        assert _delete_task(tid_doomed) is True

        # Queue a dangling event (deleted task) immediately followed by a
        # valid event — both land in the same <=20ms writer batch.
        _queue_chat_event({"type": "ev_doomed"}, tid_doomed)
        _queue_chat_event({"type": "ev_live"}, tid_live)
        _flush_chat_events()

        loaded = _load_chat_events_by_task_id(tid_live)
        assert loaded is not None
        events = loaded["events"]
        assert isinstance(events, list)
        assert [e["type"] for e in events] == ["ev_live"]

        # The dangling event must be skipped, not persisted.
        assert _load_chat_events_by_task_id(tid_doomed) is None

    def test_seq_cache_stays_consistent_after_skipped_event(self) -> None:
        tid_live, _ = _add_task("live task 2")
        tid_doomed, _ = _add_task("doomed task 2")
        assert _delete_task(tid_doomed) is True

        _queue_chat_event({"type": "ev_doomed"}, tid_doomed)
        _queue_chat_event({"type": "first"}, tid_live)
        _flush_chat_events()
        # A later batch for the live task must keep gapless ordering.
        _queue_chat_event({"type": "second"}, tid_live)
        _flush_chat_events()

        loaded = _load_chat_events_by_task_id(tid_live)
        assert loaded is not None
        events = loaded["events"]
        assert isinstance(events, list)
        assert [e["type"] for e in events] == ["first", "second"]
        # The deleted task must not have been seeded into the seq cache.
        assert tid_doomed not in th._next_seq_cache


class TestA2DeleteTaskInvalidatesChatContextCache(_TempDbTestBase):
    """[A2] ``_delete_task`` must invalidate the chat-context cache."""

    def test_deleted_task_text_disappears_from_chat_context(self) -> None:
        tid, chat_id = _add_task("secret task")
        _save_task_result("secret result", task_id=tid)

        # Populate the cache.
        text_before = _load_chat_context_text(chat_id)
        assert "secret task" in text_before
        assert "secret result" in text_before

        assert _delete_task(tid) is True

        text_after = _load_chat_context_text(chat_id)
        assert "secret task" not in text_after
        assert "secret result" not in text_after
        assert text_after == ""


class TestA3SaveTaskExtraPreservesFavorite(_TempDbTestBase):
    """[A3] ``_save_task_extra`` must not clobber ``is_favorite``."""

    def test_completion_extra_write_preserves_favorite_star(self) -> None:
        tid, _ = _add_task("fav task")
        assert _set_task_favorite(tid, True) is True

        # Task-completion write (tokens/cost) must keep the star.
        _save_task_extra({"tokens": 5, "cost": 0.25}, task_id=tid)

        extra = _read_extra(tid)
        assert extra.get("is_favorite") is True
        assert extra.get("tokens") == 5
        assert extra.get("cost") == 0.25

    def test_explicit_is_favorite_in_payload_wins(self) -> None:
        tid, _ = _add_task("unfav task")
        assert _set_task_favorite(tid, True) is True

        # A payload explicitly carrying is_favorite must be honored as-is.
        _save_task_extra({"tokens": 1, "is_favorite": False}, task_id=tid)

        extra = _read_extra(tid)
        assert extra.get("is_favorite") is False
        assert extra.get("tokens") == 1

    def test_extra_without_prior_favorite_written_verbatim(self) -> None:
        tid, _ = _add_task("plain task")
        _save_task_extra({"tokens": 7}, task_id=tid)

        extra = _read_extra(tid)
        assert extra == {"tokens": 7}


class TestA4SubagentFilterConsistency(_TempDbTestBase):
    """[A4] history-entry indexing and prefix matching must skip sub-agents."""

    def test_prefix_match_skips_subagent_rows(self) -> None:
        _add_task("visible parent task")
        time.sleep(0.01)
        _add_task(
            "subagent internal task",
            extra={"subagent": {"parent_task_id": 1}},
        )

        assert _prefix_match_task("subagent") == ""
        assert _prefix_match_task("visible") == "visible parent task"
