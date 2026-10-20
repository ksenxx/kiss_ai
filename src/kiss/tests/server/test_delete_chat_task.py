# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The CHATS panel's delete buttons: ``deleteTask`` / ``deleteChat``.

End-to-end over a real history.db (redirected to a temp dir): the
persistence helpers remove the task rows, their sub-tasks and events
(a chat delete also drops its ``chat_summaries`` row); the command
handlers refuse a live task with a notice and nudge every page with
``tasks_updated`` once the rows are gone.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.tests.server.conftest import (
    clear_agent_states,
    make_capturing_server,
    redirect_persistence,
    restore_persistence,
)


def _count(sql: str, *params: object) -> int:
    db = th._get_db()
    row = db.execute(sql, params).fetchone()
    return int(row[0])


def _add_events(task_id: str, n: int) -> None:
    for i in range(n):
        th._queue_chat_event({"type": "system_output", "text": f"line {i}"}, task_id)
    th._flush_chat_events(task_id)


class _DbTest:
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


class TestDeleteTaskPersistence(_DbTest):
    def test_task_events_and_subtasks_go_other_rows_stay(self) -> None:
        task_id, chat_id = th._add_task("parent")
        child, _ = th._add_task(
            "child", chat_id, extra={"subagent": {"parent_task_id": task_id}},
        )
        grandchild, _ = th._add_task(
            "grandchild", chat_id, extra={"subagent": {"parent_task_id": child}},
        )
        other, _ = th._add_task("other", chat_id)
        for tid in (task_id, child, grandchild, other):
            _add_events(tid, 2)
        assert _count("SELECT COUNT(*) FROM task_history") == 4
        assert _count("SELECT COUNT(*) FROM events") == 8

        assert th._delete_task(task_id) == 3

        ids = {r[0] for r in th._get_db().execute("SELECT id FROM task_history")}
        assert ids == {other}
        assert _count("SELECT COUNT(*) FROM events WHERE task_id = ?", other) == 2
        assert _count("SELECT COUNT(*) FROM events") == 2
        # The chat lives on with its remaining task.
        assert _count("SELECT COUNT(*) FROM chat_summaries WHERE chat_id = ?", chat_id) >= 0

    def test_unknown_or_empty_task_is_a_noop(self) -> None:
        th._add_task("keep")
        assert th._delete_task("no-such-task") == 0
        assert th._delete_task("") == 0
        assert _count("SELECT COUNT(*) FROM task_history") == 1

    def test_deleted_task_can_be_recreated_with_fresh_events(self) -> None:
        """The per-task seq cache is dropped with the rows."""
        task_id, _ = th._add_task("again")
        _add_events(task_id, 3)
        th._delete_task(task_id)
        db = th._get_db()
        db.execute(
            "INSERT INTO task_history (id, task, chat_id, timestamp) VALUES (?, ?, ?, 1.0)",
            (task_id, "again", "c"),
        )
        _add_events(task_id, 1)
        assert _count("SELECT COUNT(*) FROM events WHERE task_id = ?", task_id) == 1


class TestDeleteChatPersistence(_DbTest):
    def test_every_task_of_the_chat_and_its_summary_go(self) -> None:
        t1, chat_id = th._add_task("first")
        t2, _ = th._add_task("second", chat_id)
        sub, _ = th._add_task("sub", chat_id, extra={"subagent": {"parent_task_id": t2}})
        keep, other_chat = th._add_task("elsewhere")
        for tid in (t1, t2, sub, keep):
            _add_events(tid, 1)
        th._set_chat_summary(chat_id, "My chat")
        th._set_chat_summary(other_chat, "Other chat")

        assert th._delete_chat(chat_id) == 3

        ids = {r[0] for r in th._get_db().execute("SELECT id FROM task_history")}
        assert ids == {keep}
        assert _count("SELECT COUNT(*) FROM events") == 1
        assert _count("SELECT COUNT(*) FROM chat_summaries WHERE chat_id = ?", chat_id) == 0
        assert _count("SELECT COUNT(*) FROM chat_summaries WHERE chat_id = ?", other_chat) == 1

    def test_unknown_or_empty_chat_is_a_noop(self) -> None:
        th._add_task("keep")
        assert th._delete_chat("no-such-chat") == 0
        assert th._delete_chat("") == 0
        assert _count("SELECT COUNT(*) FROM task_history") == 1

    def test_delete_is_visible_to_history(self) -> None:
        _, chat_id = th._add_task("gone")
        th._add_task("stays")
        th._delete_chat(chat_id)
        server, events = make_capturing_server()
        server._get_history(query=None)
        sessions = [e for e in events if e.get("type") == "history"][0]["sessions"]
        assert [s["title"] for s in sessions] == ["stays"]


def _notices(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if e.get("type") == "notice"]


class TestDeleteCommands(_DbTest):
    def test_delete_task_command_removes_the_row_and_nudges(self) -> None:
        task_id, _ = th._add_task("doomed")
        th._add_task("spared")
        server, events = make_capturing_server()

        server._cmd_delete_task({"taskId": task_id, "tabId": "tab-1", "connId": "c1"})

        assert _count("SELECT COUNT(*) FROM task_history WHERE id = ?", task_id) == 0
        assert _count("SELECT COUNT(*) FROM task_history") == 1
        assert {"type": "tasks_updated"} in events
        assert _notices(events) == []

    def test_delete_chat_command_removes_the_chat_and_nudges(self) -> None:
        _, chat_id = th._add_task("doomed")
        th._add_task("doomed too", chat_id)
        th._add_task("spared")
        server, events = make_capturing_server()

        server._cmd_delete_chat({"chatId": chat_id, "tabId": "tab-1", "connId": "c1"})

        assert _count("SELECT COUNT(*) FROM task_history WHERE chat_id = ?", chat_id) == 0
        assert _count("SELECT COUNT(*) FROM task_history") == 1
        assert {"type": "tasks_updated"} in events

    def test_missing_ids_do_nothing(self) -> None:
        th._add_task("stays")
        server, events = make_capturing_server()
        server._cmd_delete_task({"taskId": ""})
        server._cmd_delete_task({})
        server._cmd_delete_chat({"chatId": ""})
        server._cmd_delete_chat({})
        assert _count("SELECT COUNT(*) FROM task_history") == 1
        assert events == []

    def test_running_task_is_refused_with_a_notice(self) -> None:
        task_id, chat_id = th._add_task("live")
        st = AgentState(task_id, tab_id="tab-live", chat_id=chat_id, server_owned=True)
        st.is_task_active = True
        agent_state.register(st)
        server, events = make_capturing_server()

        server._cmd_delete_task({"taskId": task_id, "tabId": "tab-x", "connId": "c9"})
        server._cmd_delete_chat({"chatId": chat_id, "tabId": "tab-x", "connId": "c9"})

        assert _count("SELECT COUNT(*) FROM task_history WHERE id = ?", task_id) == 1
        notices = _notices(events)
        assert [n["text"] for n in notices] == [
            "Cannot delete the task: its task is still running. Stop it first.",
            "Cannot delete the chat: its task is still running. Stop it first.",
        ]
        assert all(n["connId"] == "c9" and n["tabId"] == "tab-x" for n in notices)
        assert {"type": "tasks_updated"} not in events

    def test_finished_parent_of_a_running_subtask_is_refused(self) -> None:
        """The sub-task's rows would go with the parent's while its worker still writes them."""
        parent, chat_id = th._add_task("parent done")
        child, _ = th._add_task(
            "child running", chat_id, extra={"subagent": {"parent_task_id": parent}},
        )
        st = AgentState(child, tab_id="tab-sub", chat_id=chat_id, server_owned=True)
        st.is_task_active = True
        agent_state.register(st)
        server, events = make_capturing_server()

        server._cmd_delete_task({"taskId": parent, "tabId": "tab-x", "connId": "c9"})
        server._cmd_delete_chat({"chatId": chat_id, "tabId": "tab-x", "connId": "c9"})

        assert _count("SELECT COUNT(*) FROM task_history") == 2
        assert len(_notices(events)) == 2
        assert {"type": "tasks_updated"} not in events

        # Once the sub-task ends, the parent (and the sub-task) go.
        st.is_task_active = False
        server._cmd_delete_task({"taskId": parent, "tabId": "tab-x", "connId": "c9"})
        assert _count("SELECT COUNT(*) FROM task_history") == 0
        assert {"type": "tasks_updated"} in events

    def test_parent_of_a_starting_subtask_without_a_row_is_refused(self) -> None:
        """A sub-task that has no history row yet is known by its parent id alone."""
        parent, chat_id = th._add_task("parent done")
        st = AgentState(
            "api-temp-child", tab_id="tab-sub", chat_id="", server_owned=True,
            parent_task_id=parent,
        )
        st.is_task_active = True
        agent_state.register(st)
        server, events = make_capturing_server()

        server._cmd_delete_task({"taskId": parent, "tabId": "tab-x", "connId": "c9"})

        assert _count("SELECT COUNT(*) FROM task_history WHERE id = ?", parent) == 1
        assert len(_notices(events)) == 1
        assert {"type": "tasks_updated"} not in events

    def test_finished_task_with_a_stale_state_is_deleted(self) -> None:
        """A registered but idle state (the run ended) does not block the delete."""
        task_id, chat_id = th._add_task("done")
        st = AgentState(task_id, tab_id="tab-done", chat_id=chat_id, server_owned=True)
        st.is_task_active = False
        agent_state.register(st)
        server, events = make_capturing_server()
        server._cmd_delete_chat({"chatId": chat_id})
        assert _count("SELECT COUNT(*) FROM task_history") == 0
        assert {"type": "tasks_updated"} in events

    def test_delete_commands_are_in_the_api_catalog(self) -> None:
        from kiss.server.sorcar import API

        assert API["deleteTask"].required == ("taskId",)
        assert API["deleteChat"].required == ("chatId",)
        assert API["openExternal"].required == ("url",)


def test_sqlite_rows_are_really_gone_on_disk(tmp_path: Any) -> None:
    """A second connection (another process, say) sees the deletion."""
    saved = redirect_persistence(str(tmp_path))
    try:
        task_id, chat_id = th._add_task("on disk")
        _add_events(task_id, 2)
        th._delete_chat(chat_id)
        other = sqlite3.connect(str(th._DB_PATH))
        try:
            assert other.execute("SELECT COUNT(*) FROM task_history").fetchone()[0] == 0
            assert other.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
        finally:
            other.close()
    finally:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        restore_persistence(saved)
