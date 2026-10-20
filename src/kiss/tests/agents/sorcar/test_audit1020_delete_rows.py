# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Row deletion and the shutdown rewrite in ``persistence`` (audit 2026-10-20, slice B1).

* ``_delete_task_rows`` drops BOTH per-task event caches of a deleted
  id (B1-F1: ``_marked_has_events`` used to survive, so a reused id
  never got ``has_events`` set again) and removes a chat's summary row
  in the same transaction as its tasks (B1-F2: it used to be a second
  transaction after the commit).
* ``_shutdown_persist_in_flight_results`` still refreshes the chat
  context cache after dropping its extra ``SELECT`` (B1-F5).
* ``_delete_task`` keeps its per-task flush: queued events of a deleted
  sub-task are discarded by the writer, not orphaned (C4-F7 shown moot).
"""

from __future__ import annotations

import shutil
import tempfile
import time

import kiss.agents.sorcar.persistence as th
from kiss.tests.server.conftest import (
    clear_agent_states,
    redirect_persistence,
    restore_persistence,
)


def _count(sql: str, *params: object) -> int:
    row = th._get_db().execute(sql, params).fetchone()
    return int(row[0])


def _queue_events(task_id: str, n: int) -> None:
    for i in range(n):
        th._queue_chat_event({"type": "system_output", "text": f"line {i}"}, task_id)


class TestDeleteRows:
    def setup_method(self) -> None:
        clear_agent_states()
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)

    def teardown_method(self) -> None:
        clear_agent_states()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_delete_task_drops_both_event_caches(self) -> None:
        task_id, _ = th._add_task("with events")
        _queue_events(task_id, 2)
        th._flush_chat_events(task_id)
        assert task_id in th._next_seq_cache
        assert task_id in th._marked_has_events

        assert th._delete_task(task_id) == 1

        assert task_id not in th._next_seq_cache
        assert task_id not in th._marked_has_events

    def test_deleted_id_reused_gets_has_events_again(self) -> None:
        """The observable effect of a stale ``_marked_has_events`` entry."""
        task_id, chat_id = th._add_task("first life")
        _queue_events(task_id, 1)
        th._flush_chat_events(task_id)
        th._delete_task(task_id)

        db = th._get_db()
        with th._rw_lock.write_lock(), th._immediate_txn(db):
            db.execute(
                "INSERT INTO task_history (id, chat_id, timestamp, task, result, has_events) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (task_id, chat_id, time.time(), "second life", "Agent Failed Abruptly"),
            )
        _queue_events(task_id, 1)
        th._flush_chat_events(task_id)

        assert _count("SELECT has_events FROM task_history WHERE id = ?", task_id) == 1

    def test_delete_chat_removes_summary_in_the_tasks_transaction(self) -> None:
        task_id, chat_id = th._add_task("summarised")
        assert th._set_chat_summary(chat_id, "a summary")
        assert _count("SELECT COUNT(*) FROM chat_summaries WHERE chat_id = ?", chat_id) == 1

        statements: list[str] = []
        db = th._get_db()
        db.set_trace_callback(statements.append)
        try:
            assert th._delete_chat(chat_id) == 1
        finally:
            db.set_trace_callback(None)

        assert _count("SELECT COUNT(*) FROM task_history WHERE id = ?", task_id) == 0
        assert _count("SELECT COUNT(*) FROM chat_summaries WHERE chat_id = ?", chat_id) == 0
        # The summary delete is inside the transaction that drops the rows.
        begin = [i for i, s in enumerate(statements) if s.startswith("BEGIN")]
        commit = [i for i, s in enumerate(statements) if s.startswith("COMMIT")]
        summary = [i for i, s in enumerate(statements) if "DELETE FROM chat_summaries" in s]
        tasks = [i for i, s in enumerate(statements) if "DELETE FROM task_history" in s]
        assert len(summary) == 1 and len(tasks) == 1
        assert begin[-1] < tasks[0] < summary[0] < commit[-1], statements

    def test_delete_chat_with_no_tasks_still_removes_summary(self) -> None:
        chat_id = "chat-no-tasks"
        assert th._set_chat_summary(chat_id, "stale")

        assert th._delete_chat(chat_id) == 0

        assert _count("SELECT COUNT(*) FROM chat_summaries WHERE chat_id = ?", chat_id) == 0

    def test_delete_task_leaves_no_rows_for_a_subtask_with_queued_events(self) -> None:
        parent, chat_id = th._add_task("parent")
        child, _ = th._add_task(
            "child",
            chat_id,
            extra={"subagent": {"parent_task_id": parent}},
        )
        _queue_events(child, 3)  # still queued, not flushed

        assert th._delete_task(parent) == 2
        th._flush_chat_events()

        assert _count("SELECT COUNT(*) FROM task_history WHERE id IN (?, ?)", parent, child) == 0
        assert _count("SELECT COUNT(*) FROM events WHERE task_id IN (?, ?)", parent, child) == 0

    def test_shutdown_rewrite_refreshes_chat_context(self) -> None:
        task_id, chat_id = th._add_task("in flight")
        assert "Agent Failed Abruptly" in th._load_chat_context_text(chat_id)

        assert th._shutdown_persist_in_flight_results({task_id}) == 1

        assert "Task interrupted by server restart/shutdown" in th._load_chat_context_text(chat_id)

    def test_shutdown_rewrite_of_nothing_touches_nothing(self) -> None:
        done_id, chat_id = th._add_task("done")
        th._save_task_result("finished", task_id=done_id)
        text = th._load_chat_context_text(chat_id)

        assert th._shutdown_persist_in_flight_results({done_id}) == 0
        assert th._shutdown_persist_in_flight_results(set()) == 0

        assert th._load_chat_context_text(chat_id) == text

    def test_connection_busy_timeout_is_the_pragma(self) -> None:
        row = th._get_db().execute("PRAGMA busy_timeout").fetchone()
        assert int(row[0]) == 30000
