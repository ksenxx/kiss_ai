# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""``deleteTask`` / ``deleteChat`` versus a run starting in the same chat.

Audit 2026-10-20 (findings B1-N1 / C4-F8): the liveness check used to
run lock-free, so a ``run`` registering its state and inserting its
``task_history`` row between the check and the delete lost that row.
The check moved under ``STATE_LOCK`` — and the delete with it, which
made every command of every connection wait behind the history
delete's event-writer flush, reader drain and SQLite busy timeout
(audit 2026-10-25, server-main).  Now only the liveness check holds
``STATE_LOCK``; the database delete runs outside it while a counter
(``_history_deletes``) keeps ``_cmd_run`` from admitting a fresh task
until the delete is over, so the run's row is never swept away.

The run side is the real ``_cmd_run`` with a ``/ask help`` prompt: it
registers its state, starts its worker and inserts its row without
ever calling a model.  The delete handler is parked INSIDE the
database delete (``persistence._delete_chat`` / ``_delete_task``) by
an :class:`EntryGate` (a profile hook; nothing is replaced), so the
run provably arrives while the delete is in flight: ``STATE_LOCK``
must be free, the run must wait, and its row must survive.  No sleep
decides the interleaving.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import sea_commands
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.commands import _CommandsMixin
from kiss.tests.server.conftest import (
    EntryGate,
    clear_agent_states,
    make_capturing_server,
    redirect_persistence,
    restore_persistence,
)

TAB = "tab-a1020-late"


def _count(sql: str, *params: object) -> int:
    row = th._get_db().execute(sql, params).fetchone()
    return int(row[0])


class _AfterClock(EntryGate):
    """An :class:`EntryGate` that parks the thread once ``time.time()`` returned in the function."""

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        if event == "c_return" and arg is time.time and frame.f_code is self.code:
            self.reached.set()
            self.release.wait(30)


def _next_tick() -> float:
    """A wall-clock reading strictly later than the one taken on entry."""
    start, deadline = time.time(), time.monotonic() + 5
    while (now := time.time()) <= start:
        assert time.monotonic() < deadline, "the wall clock does not advance"
    return now


class TestDeleteVersusStartingRun:
    def setup_method(self) -> None:
        clear_agent_states()
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
        sea_commands.refresh_registry()
        self.gate: EntryGate | None = None

    def teardown_method(self) -> None:
        threading.setprofile(None)
        if self.gate is not None:
            self.gate.release.set()
        clear_agent_states()
        th._close_thread_db()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_cmd(self, chat_id: str) -> dict[str, Any]:
        return {
            "type": "run",
            "tabId": TAB,
            "prompt": "/ask help",
            "chatId": chat_id,
            "workDir": self.tmpdir,
            "useWorktree": False,
            "autoCommit": False,
            "classifyTasks": False,
        }

    def _run_arrives_during_the_delete(
        self,
        server: Any,
        delete: Any,
        cmd: dict[str, Any],
        gated: Any,
        chat_id: str,
    ) -> None:
        """Park *delete* inside the database delete *gated*, then submit a run.

        Returns once the delete, the run's handler and its worker have
        all finished.
        """
        self.gate = EntryGate(gated.__code__)
        deleter = self.gate.start(delete, cmd)
        assert self.gate.reached.wait(10), "the delete never reached the database"
        # The database delete runs OUTSIDE ``STATE_LOCK``: every other
        # command keeps going while it waits on the event writer, the
        # readers and SQLite.
        assert agent_state.STATE_LOCK.acquire(timeout=1), (
            "BUG: the history delete holds STATE_LOCK across the database write"
        )
        agent_state.STATE_LOCK.release()
        runner = threading.Thread(target=server._cmd_run, args=(self._run_cmd(chat_id),))
        runner.start()
        # The run is held back until the delete is over: nothing is
        # registered for its tab and no row is inserted meanwhile.
        runner.join(timeout=0.3)
        assert runner.is_alive(), "BUG: the run was admitted while the delete was in flight"
        with agent_state.STATE_LOCK:
            assert agent_state.find_by_tab(TAB) is None
        self.gate.release.set()
        deleter.join(timeout=10)
        runner.join(timeout=10)
        assert not deleter.is_alive() and not runner.is_alive()
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(TAB)
            worker = state.task_thread if state is not None else None
        if worker is not None:
            worker.join(timeout=10)
        th._flush_chat_events()

    def test_delete_chat_does_not_drop_a_run_that_starts_meanwhile(self) -> None:
        _, chat_id = th._add_task("finished")
        server, events = make_capturing_server()

        self._run_arrives_during_the_delete(
            server,
            server._cmd_delete_chat,
            {"chatId": chat_id, "tabId": "tab-1", "connId": "c1"},
            th._delete_chat,
            chat_id,
        )

        # The finished task went; the run that started meanwhile kept its row.
        assert _count("SELECT COUNT(*) FROM task_history WHERE task = 'finished'") == 0
        assert (
            _count("SELECT COUNT(*) FROM task_history WHERE chat_id = ? AND task = ?",
                   chat_id, "/ask help") == 1
        )
        assert _count("SELECT COUNT(*) FROM task_history") == 1
        assert {"type": "tasks_updated"} in events
        assert server._history_deletes == 0

    def test_delete_task_does_not_drop_a_run_that_starts_meanwhile(self) -> None:
        parent, chat_id = th._add_task("parent done")
        server, events = make_capturing_server()

        self._run_arrives_during_the_delete(
            server,
            server._cmd_delete_task,
            {"taskId": parent, "tabId": "tab-1", "connId": "c1"},
            th._delete_task,
            chat_id,
        )

        assert _count("SELECT COUNT(*) FROM task_history WHERE id = ?", parent) == 0
        assert (
            _count("SELECT COUNT(*) FROM task_history WHERE chat_id = ? AND task = ?",
                   chat_id, "/ask help") == 1
        )
        assert {"type": "tasks_updated"} in events
        assert server._history_deletes == 0

    def test_run_registered_before_the_check_is_still_refused(self) -> None:
        """The liveness check (still under ``STATE_LOCK``) refuses as before."""
        _, chat_id = th._add_task("finished")
        live = AgentState("live-run", tab_id="tab-live", chat_id=chat_id, server_owned=True)
        live.is_task_active = True
        with agent_state.STATE_LOCK:
            agent_state.register(live)
        server, events = make_capturing_server()

        server._cmd_delete_chat({"chatId": chat_id, "tabId": "tab-1", "connId": "c1"})

        assert _count("SELECT COUNT(*) FROM task_history") == 1
        assert [e["type"] for e in events] == ["notice"]
        assert server._history_deletes == 0

    def test_an_earlier_delete_storing_last_does_not_move_the_cut_back(self) -> None:
        """The first delete has read the clock; the second must not get its cut in before it."""
        older, _ = th._add_task("older prompt")
        newer, _ = th._add_task("newer prompt")
        th._add_task("kept prompt")
        server, events = make_capturing_server()
        self.gate = _AfterClock(_CommandsMixin._delete_history.__code__)
        first = self.gate.start(server._cmd_delete_task, {"taskId": older})
        second = threading.Thread(
            target=server._cmd_delete_task, args=({"taskId": newer},), daemon=True,
        )
        assert self.gate.reached.wait(30), "the first delete never read the clock"
        threading.setprofile(None)  # only ``first`` carries the gate
        held = _next_tick()  # a page's watermark: after the first delete's time ...
        _next_tick()  # ... and before the second's
        second.start()
        second.join(0.5)  # over before the release only when the cut is stored without the lock
        self.gate.release.set()
        first.join(timeout=30)
        second.join(timeout=30)
        assert not first.is_alive() and not second.is_alive()

        server._get_input_history("c1", since=held)

        reply = [e for e in events if e["type"] == "inputHistory"][-1]
        assert "since" not in reply and reply["tasks"] == ["kept prompt"], (
            f"BUG: a list older than the last deletion was refreshed with a delta: {reply}"
        )
