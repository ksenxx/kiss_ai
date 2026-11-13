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
import sqlite3
import tempfile
import threading
import time

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_summary import summarize_chat
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


def _delete_task_signalling_begin(
    task_id: str, begun: threading.Event, removed: list[int],
) -> None:
    """Run ``_delete_task`` on this thread's connection; set *begun* at its ``BEGIN``."""
    db = th._get_db()

    def trace(statement: str) -> None:
        if statement.startswith("BEGIN"):
            begun.set()

    db.set_trace_callback(trace)
    try:
        removed.append(th._delete_task(task_id))
    finally:
        db.set_trace_callback(None)


class TestDeleteRows:
    def setup_method(self) -> None:
        clear_agent_states()
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)

    def teardown_method(self) -> None:
        clear_agent_states()
        th._close_thread_db()
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

    def test_delete_task_refreshes_the_summary_row_of_its_chat(self) -> None:
        # A11-4 (astra): the derived summary fields of the chat followed the
        # deleted task (its launch and completion instants, its text in the
        # summary) because only a whole-chat deletion touched the row.
        first_id, chat_id = th._add_task(
            "Write the paper introduction", extra={"startTs": 1_000})
        th._save_task_result("done", task_id=first_id)
        th._save_task_extra({"endTs": 2_000}, task_id=first_id)
        second_id, _ = th._add_task(
            "Polish the abstract", chat_id=chat_id, extra={"startTs": 3_000})
        th._save_task_result("done", task_id=second_id)
        th._save_task_extra({"endTs": 4_000}, task_id=second_id)
        assert th._set_chat_summary(chat_id, "My title")
        before = th._chat_summaries([chat_id])[chat_id]
        assert (before["last_launched"], before["last_finished"]) == (3_000, 4_000)

        statements: list[str] = []
        db = th._get_db()
        db.set_trace_callback(statements.append)
        try:
            assert th._delete_task(second_id) == 1
        finally:
            db.set_trace_callback(None)

        # The stamps are the remaining task's, the summary is derived from
        # it alone, the user's title survives, all in the deletion's
        # transaction.
        after = th._chat_summaries([chat_id])[chat_id]
        assert (after["last_launched"], after["last_finished"]) == (1_000, 2_000)
        assert after["summary"] == "My title"
        computed = th._get_db().execute(
            "SELECT summary FROM chat_summaries WHERE chat_id = ?", (chat_id,)).fetchone()[0]
        assert computed == summarize_chat(["Write the paper introduction"])
        begin = [i for i, s in enumerate(statements) if s.startswith("BEGIN")]
        commit = [i for i, s in enumerate(statements) if s.startswith("COMMIT")]
        refresh = [i for i, s in enumerate(statements) if "INSERT INTO chat_summaries" in s]
        assert len(refresh) == 1 and begin[-1] < refresh[0] < commit[-1], statements

        # The last listable task gone: nothing derived is left on the row,
        # the title is (the chat may run its next task).
        assert th._delete_task(first_id) == 1
        assert th._chat_summaries([chat_id])[chat_id] == {
            "summary": "My title", "last_launched": 0, "last_finished": 0}
        assert _count(
            "SELECT COUNT(*) FROM chat_summaries WHERE chat_id = ? AND summary = ''", chat_id
        ) == 1

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

    def test_every_deletion_is_recorded_for_the_sync(self) -> None:
        """A deleted task would come back from the other machine without a record.

        The task database is synced between machines by adding the rows
        one lacks, so a deleted task, chat or sub-task leaves its id in
        ``deleted_tasks`` with the moment it went, in the deleting
        transaction; a chat deletion records each of the chat's tasks.
        """
        before = time.time()
        parent, chat_id = th._add_task("parent")
        child, _ = th._add_task("child", chat_id, extra={"subagent": {"parent_task_id": parent}})
        other, other_chat = th._add_task("other")
        th._save_task_result("finished", task_id=other)
        assert _count("SELECT COUNT(*) FROM deleted_tasks") == 0

        assert th._delete_task(parent) == 2
        assert th._delete_chat(other_chat) == 1

        rows = th._get_db().execute("SELECT id, deleted_ts FROM deleted_tasks").fetchall()
        assert sorted(r[0] for r in rows) == sorted([parent, child, other])
        assert all(before <= r[1] <= time.time() for r in rows)

    def test_delete_drops_the_remote_subtask_records_of_the_deleted_tasks(self) -> None:
        """A ``remote_subtasks`` record goes with the parent it belongs to.

        The record is the parent's side of the account of a sub-agent run
        on another machine; nothing else deletes it, and every spend
        computation scans the whole table, so a deleted parent must not
        leave its records behind.  A record of a surviving task stays.
        """
        parent, chat_id = th._add_task("parent")
        child, _ = th._add_task("child", chat_id, extra={"subagent": {"parent_task_id": parent}})
        other, other_chat = th._add_task("other")
        th._record_remote_subtask("remote-of-parent", parent, "box", "m", 1.0, 10, 2)
        th._record_remote_subtask("remote-of-child", child, "box", "m", 0.5, 5, 1)
        th._record_remote_subtask("remote-of-other", other, "box", "m", 0.25, 2, 1)

        assert th._delete_task(parent) == 2

        assert _count(
            "SELECT COUNT(*) FROM remote_subtasks WHERE parent_task_id IN (?, ?)", parent, child,
        ) == 0
        assert _count("SELECT COUNT(*) FROM remote_subtasks WHERE parent_task_id = ?", other) == 1

        assert th._delete_chat(other_chat) == 1

        assert _count("SELECT COUNT(*) FROM remote_subtasks") == 0

    def test_delete_task_takes_a_sub_task_another_process_commits_meanwhile(self) -> None:
        """The tree is discovered under the database's write lock.

        ``_rw_lock`` orders this process only.  A peer process (a second
        connection here) holding the write lock while it inserts a
        sub-task commits once the deletion has asked for ``BEGIN
        IMMEDIATE``; the sub-task is then in the tree the deletion
        removes, not left behind pointing at a parent that is gone.
        """
        parent, chat_id = th._add_task("parent")
        peer = sqlite3.connect(str(th._DB_PATH), timeout=30)
        peer.execute("BEGIN IMMEDIATE")
        peer.execute(
            "INSERT INTO task_history (id, chat_id, timestamp, task, result, parent_task_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("late-child", chat_id, time.time(), "child", "Agent Failed Abruptly", parent),
        )
        begun = threading.Event()
        removed: list[int] = []
        deleter = threading.Thread(
            target=_delete_task_signalling_begin, args=(parent, begun, removed),
        )
        try:
            deleter.start()
            assert begun.wait(20), "the deletion never issued its BEGIN"
            peer.commit()
            deleter.join(40)
            assert not deleter.is_alive()
        finally:
            peer.close()  # rolls back whatever is still open

        assert removed == [2]
        assert _count(
            "SELECT COUNT(*) FROM task_history WHERE id IN (?, 'late-child')", parent,
        ) == 0
        assert _count("SELECT COUNT(*) FROM deleted_tasks WHERE id = 'late-child'") == 1

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
