# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""``deleteTask`` / ``deleteChat`` versus a run starting in the same chat.

The liveness check and the delete form one critical section under
``STATE_LOCK`` (audit 2026-10-20, findings B1-N1 / C4-F8).  Before the
fix the check ran lock-free: a ``run`` registering its state and
inserting its ``task_history`` row between the check and the delete
lost that row, so the task ran on with its events dropped and the
chats panel never showing it.

The run side is driven with the same primitives ``_cmd_run`` and the
worker use (``agent_state.register`` under ``STATE_LOCK``, then
``_add_task``).  The delete handler is parked INSIDE its critical
section, at the ``_race_delay`` hook between the check and the delete,
by an :class:`EntryGate` (a profile hook; nothing is replaced), so the
run provably arrives while the section is open: it must wait at the
lock and keep its row.  No sleep decides the interleaving.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar._concurrency import _race_delay
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.tests.server.conftest import (
    EntryGate,
    clear_agent_states,
    make_capturing_server,
    redirect_persistence,
    restore_persistence,
)


def _count(sql: str, *params: object) -> int:
    row = th._get_db().execute(sql, params).fetchone()
    return int(row[0])


class TestDeleteVersusStartingRun:
    def setup_method(self) -> None:
        clear_agent_states()
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
        self.gate = EntryGate(_race_delay.__code__)

    def teardown_method(self) -> None:
        threading.setprofile(None)
        self.gate.release.set()
        clear_agent_states()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _start_run(self, chat_id: str, parent: str = "") -> dict[str, Any]:
        """Register a running state under ``STATE_LOCK`` and insert its row, as a run does."""
        started: dict[str, Any] = {"at_lock": threading.Event()}

        def run() -> None:
            st = AgentState(
                "late-run",
                tab_id="tab-late",
                chat_id=chat_id,
                server_owned=True,
                parent_task_id=parent,
            )
            st.is_task_active = True
            started["at_lock"].set()
            with agent_state.STATE_LOCK:
                agent_state.register(st)
            extra: dict[str, object] | None = (
                {"subagent": {"parent_task_id": parent}} if parent else None
            )
            started["task_id"], _ = th._add_task("late run", chat_id, extra=extra)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        started["thread"] = thread
        return started

    def _run_arrives_inside_the_critical_section(
        self,
        delete: Any,
        cmd: dict[str, Any],
        chat_id: str,
        parent: str = "",
    ) -> dict[str, Any]:
        """Park *delete* between its check and its delete, then start a run.

        Returns the run's record once both threads have finished.
        """
        deleter = self.gate.start(delete, cmd)
        assert self.gate.reached.wait(10), "the delete never passed its liveness check"
        late = self._start_run(chat_id, parent)
        assert late["at_lock"].wait(10)
        # The section is open and holds ``STATE_LOCK``: the run cannot
        # register (let alone insert its row) until it closes.
        late["thread"].join(timeout=0.2)
        assert late["thread"].is_alive(), (
            "BUG: the run registered and inserted its row inside the delete's critical section"
        )
        self.gate.release.set()
        deleter.join(timeout=10)
        late["thread"].join(timeout=10)
        assert not deleter.is_alive() and not late["thread"].is_alive()
        return late

    def test_delete_chat_does_not_drop_a_run_that_starts_after_the_check(self) -> None:
        _, chat_id = th._add_task("finished")
        server, events = make_capturing_server()

        late = self._run_arrives_inside_the_critical_section(
            server._cmd_delete_chat,
            {"chatId": chat_id, "tabId": "tab-1", "connId": "c1"},
            chat_id,
        )

        # The finished task went; the run that started meanwhile kept its row.
        assert _count("SELECT COUNT(*) FROM task_history WHERE id = ?", late["task_id"]) == 1
        assert _count("SELECT COUNT(*) FROM task_history") == 1
        assert {"type": "tasks_updated"} in events

    def test_delete_task_does_not_drop_a_subtask_that_starts_after_the_check(self) -> None:
        parent, chat_id = th._add_task("parent done")
        server, events = make_capturing_server()

        late = self._run_arrives_inside_the_critical_section(
            server._cmd_delete_task,
            {"taskId": parent, "tabId": "tab-1", "connId": "c1"},
            chat_id,
            parent,
        )

        assert _count("SELECT COUNT(*) FROM task_history WHERE id = ?", late["task_id"]) == 1
        assert _count("SELECT COUNT(*) FROM task_history WHERE id = ?", parent) == 0
        assert {"type": "tasks_updated"} in events

    def test_run_registered_before_the_check_is_still_refused(self) -> None:
        """The critical section does not change the refusal path."""
        _, chat_id = th._add_task("finished")
        late = self._start_run(chat_id)
        late["thread"].join(timeout=5)
        server, events = make_capturing_server()

        server._cmd_delete_chat({"chatId": chat_id, "tabId": "tab-1", "connId": "c1"})

        assert _count("SELECT COUNT(*) FROM task_history") == 2
        assert [e["type"] for e in events] == ["notice"]
