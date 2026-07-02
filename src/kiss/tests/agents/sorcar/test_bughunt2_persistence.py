# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 2: writes addressed to a DELETED task must not hit another task.

``_resolve_task_id`` falls back to ``_most_recent_task_id(db, task)``
whenever the given ``task_id`` does not resolve to a row.  That
fallback exists for LEGACY callers whose ``task_id`` is malformed (an
old integer id, a non-UUID string) — see
``test_review_round4_bugs.py``.  But the fallback also fired when the
caller supplied a perfectly well-formed ``uuid4().hex`` id whose row
had simply been DELETED (user removes a running task from the history
sidebar via ``VSCodeServer._handle_delete_task``, or a stale id from a
finished tab).  Because those callers pass ``task=None``, the fallback
resolved to the MOST RECENT row overall — an unrelated task — so:

* ``_save_task_result(result, task_id=<deleted>)`` (invoked from
  ``ChatSorcarAgent.run``'s cleanup ``finally``) clobbered the
  unrelated task's result text,
* ``_save_task_extra({...}, task_id=<deleted>)`` overwrote the
  unrelated task's tokens/cost/steps metadata, and
* ``_append_chat_event(ev, task_id=<deleted>)`` (e.g. the server's
  fire-and-forget ``followup_suggestion`` thread) appended event rows
  into the unrelated task's replay stream.

The fix: when ``task_id`` is shaped like a canonical
``task_history.id`` (``is_task_history_id``) but no row exists, the
write is dropped (``_resolve_task_id`` returns ``None``) instead of
being redirected.  The legacy malformed-id fallback is preserved.

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _append_chat_event,
    _delete_task,
    _load_chat_events_by_task_id,
    _resolve_task_id,
    _save_task_extra,
    _save_task_result,
)


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._invalidate_chat_context_cache("")
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestDeletedTaskIdWritesAreDropped(_TempDbTestBase):
    """Writes with a valid-but-deleted UUID id must be no-ops."""

    def test_save_task_result_does_not_clobber_unrelated_task(self) -> None:
        victim_id, _ = _add_task("victim task")
        _save_task_result("victim finished fine", task_id=victim_id)
        doomed_id, _ = _add_task("doomed task")
        assert _delete_task(doomed_id) is True

        # The doomed task's cleanup ``finally`` fires after deletion.
        _save_task_result("Task failed: agent crashed", task_id=doomed_id)

        entry = _load_chat_events_by_task_id(victim_id)
        assert entry is not None
        row = th._get_db().execute(
            "SELECT result FROM task_history WHERE id = ?", (victim_id,)
        ).fetchone()
        assert row["result"] == "victim finished fine", (
            "result of a DELETED task was written onto an unrelated "
            "(most recent) task row"
        )

    def test_save_task_extra_does_not_overwrite_unrelated_task(self) -> None:
        victim_id, _ = _add_task(
            "victim task", extra={"tokens": 111, "cost": 1.5, "steps": 7},
        )
        doomed_id, _ = _add_task("doomed task")
        assert _delete_task(doomed_id) is True

        _save_task_extra(
            {"tokens": 99999, "cost": 42.0, "steps": 3}, task_id=doomed_id,
        )

        row = th._get_db().execute(
            "SELECT tokens, cost, steps FROM task_history WHERE id = ?",
            (victim_id,),
        ).fetchone()
        assert (row["tokens"], row["cost"], row["steps"]) == (111, 1.5, 7), (
            "extra metadata of a DELETED task overwrote an unrelated "
            "(most recent) task row"
        )

    def test_append_chat_event_does_not_pollute_unrelated_task(self) -> None:
        victim_id, _ = _add_task("victim task")
        doomed_id, _ = _add_task("doomed task")
        assert _delete_task(doomed_id) is True

        # e.g. the fire-and-forget followup-suggestion thread landing
        # after the user deleted the task.
        _append_chat_event(
            {"type": "followup_suggestion", "text": "stale"},
            task_id=doomed_id,
        )

        entry = _load_chat_events_by_task_id(victim_id)
        assert entry is not None
        events = entry["events"]
        assert isinstance(events, list)
        types = [ev.get("type") for ev in events]
        assert "followup_suggestion" not in types, (
            "event addressed to a DELETED task was appended to an "
            "unrelated (most recent) task's replay stream"
        )

    def test_resolve_returns_none_for_missing_valid_uuid(self) -> None:
        _add_task("some task")
        missing = "0123456789abcdef0123456789abcdef"
        with th._rw_lock.read_lock():
            resolved = _resolve_task_id(th._get_db(), missing, None)
        assert resolved is None

    def test_legacy_malformed_id_fallback_preserved(self) -> None:
        real_id, _ = _add_task("legacy task")
        with th._rw_lock.read_lock():
            db = th._get_db()
            assert _resolve_task_id(db, "not-a-uuid", "legacy task") == real_id
            assert _resolve_task_id(db, None, None) == real_id


class TestCorruptNumericColumnsDoNotCrashHistory(_TempDbTestBase):
    """One corrupt numeric column must not take down every history reader.

    ``_row_to_extra_json`` claims (in its own comments) to tolerate
    hand-edited / 3rd-party-source databases — it routes the payload
    through ``_dumps_extra`` precisely so a non-finite ``cost`` from
    such a DB cannot break strict JSON consumers.  Yet the same
    function did ``float(row["cost"] or 0.0)`` / ``int(row["tokens"]
    or 0)`` with only ``(KeyError, IndexError)`` caught: SQLite's
    dynamic typing happily stores TEXT like ``'abc'`` in the REAL
    ``cost`` (or INTEGER ``tokens``) column, and the resulting
    ``ValueError`` propagated out of ``_load_history`` /
    ``_search_history`` / ``_load_chat_events_by_task_id`` — one
    corrupt row blanked the entire history sidebar and made every
    session unloadable.
    """

    def _corrupt(self, task_id: str, column: str, value: str) -> None:
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                f"UPDATE task_history SET {column} = ? WHERE id = ?",
                (value, task_id),
            )
            db.commit()

    def test_load_history_survives_text_in_cost_column(self) -> None:
        good_id, _ = _add_task("good task")
        bad_id, _ = _add_task("bad task")
        self._corrupt(bad_id, "cost", "not-a-number")

        entries = th._load_history()

        ids = {e["id"] for e in entries}
        assert ids == {good_id, bad_id}
        bad = next(e for e in entries if e["id"] == bad_id)
        # The corrupt field degrades to its default instead of
        # crashing the whole listing.
        assert '"cost": 0.0' in str(bad["extra"]) or bad["extra"] == ""

    def test_load_chat_events_survives_text_in_tokens_column(self) -> None:
        task_id, _ = _add_task("task with corrupt tokens")
        _append_chat_event({"type": "text_delta", "text": "x"}, task_id=task_id)
        self._corrupt(task_id, "tokens", "NaNops")

        loaded = _load_chat_events_by_task_id(task_id)

        assert loaded is not None
        assert loaded["task"] == "task with corrupt tokens"
        loaded_events = loaded["events"]
        assert isinstance(loaded_events, list)
        assert len(loaded_events) == 1
