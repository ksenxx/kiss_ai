# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A run that fails before it has a history row must not write onto other rows.

``_run_task_inner`` finishes every run through ``_persist_subtask_row``.
A run that unwinds before ``ChatSorcarAgent.run`` allocated its
``task_history`` row (an unknown ``toolProfile`` is rejected first)
has no id of its own.  The persistence helpers resolve a missing id by
falling back to "the latest row with the same prompt text" or, with no
text, "the latest row of the whole table", so that run's ``task_error``
event, its "Task failed" result and its zeroed metrics used to land on
an EARLIER successful run of the same prompt and on whichever row was
written last, in any chat.

Everything here is real: a real :class:`~kiss.server.server.AgentServer`
driving ``_cmd_run``, a real worker thread, a real SQLite history
database under a throwaway ``KISS_HOME`` and a local stand-in model
endpoint.  No mocks, patches or test doubles.
"""

from __future__ import annotations

import time
import unittest
from typing import Any

import kiss.agents.sorcar.persistence as persistence
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.server.task_runner import build_task_extra_payload
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    wait_for,
)
from kiss.tests.server.pause import Pause
from kiss.tests.server.test_autocommit_wire_toggle import _FakeCredentials

_PROMPT = "v4-1 hello: finish at once"
_ROW_COLUMNS = "task, result, success, model, tokens, cost, steps, work_dir"


def _row(task_id: str) -> tuple[Any, ...]:
    """Return the persisted columns of history row *task_id*."""
    row = persistence._get_db().execute(
        f"SELECT {_ROW_COLUMNS} FROM task_history WHERE id = ?", (task_id,)
    ).fetchone()
    assert row is not None, task_id
    return tuple(row)


class TestRunWithoutRowPersistsNothing(unittest.TestCase):
    """The failed run leaves the earlier rows exactly as they were."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-v4-1-")
        self.addCleanup(self.home.cleanup)
        self.home.write_config(max_budget=5.0, use_web_browser=False)
        self.credentials = _FakeCredentials()
        self.addCleanup(self.credentials.restore)
        self.standin = StandInModelServer(self._respond)
        self.addCleanup(self.standin.stop)
        self.addCleanup(self._join_workers)
        self.printer = CapturePrinter()
        self.server = AgentServer(printer=self.printer)
        self.server.work_dir = str(self.home.repo)

    def _join_workers(self) -> None:
        with agent_state.STATE_LOCK:
            threads = [
                state.task_thread
                for state in agent_state.agent_states.values()
                if state.task_thread is not None
            ]
        for thread in threads:
            thread.join(timeout=30)

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        return finish_response("v4-1 done")

    def _run_cmd(self, **extra: Any) -> dict[str, Any]:
        return {
            "type": "run",
            "tabId": "v4-1-tab",
            "prompt": _PROMPT,
            "model": STANDIN_MODEL,
            "workDir": str(self.home.repo),
            "isParallel": False,
            "useWebTools": False,
            "maxBudget": 5.0,
            "modelConfig": self.standin.model_config,
            **extra,
        }

    def _terminal_events(self, event_type: str) -> list[dict[str, Any]]:
        return [
            e for e in self.printer.events_of_type(event_type)
            if e.get("tabId") == "v4-1-tab"
        ]

    def test_failed_run_without_a_row_leaves_other_rows_untouched(self) -> None:
        # Row A: a real, completed run of the prompt.
        self.server._cmd_run(self._run_cmd())
        self.assertTrue(
            wait_for(lambda: bool(self._terminal_events("task_done")), timeout=60),
            "the first run never finished",
        )
        self._join_workers()
        persistence._flush_chat_events()
        rows = persistence._get_db().execute(
            "SELECT id FROM task_history WHERE task = ?", (_PROMPT,)
        ).fetchall()
        self.assertEqual(len(rows), 1, "expected exactly one row for the prompt")
        row_a = str(rows[0]["id"])
        # Row B: the latest row of the table, an unrelated task of another chat.
        row_b, _chat_b = persistence._add_task("v4-1 unrelated", chat_id="", extra=None)
        persistence._save_task_result("fine", task_id=row_b, success=True)
        persistence._save_task_extra(
            build_task_extra_payload(
                model="other-model", work_dir="/elsewhere", version="v",
                tokens=99, cost=0.5, steps=3, is_parallel=True, start_ms=3, end_ms=4,
            ),
            task_id=row_b,
        )
        before_a = _row(row_a)
        before_b = _row(row_b)
        events_a = persistence._load_chat_events_by_task_id(row_a)
        self.assertIn("v4-1 done", str(before_a[1]))
        self.assertEqual(before_a[2], 1)
        self.assertEqual(before_b[1:4], ("fine", 1, "other-model"))

        # The same prompt again, rejected before any row is allocated.
        self.server._cmd_run(self._run_cmd(toolProfile="bogus-profile"))
        self.assertTrue(
            wait_for(lambda: bool(self._terminal_events("task_error")), timeout=60),
            "the rejected run never reported its failure",
        )
        self._join_workers()
        persistence._flush_chat_events()
        results = [
            e for e in self._terminal_events("result")
            if "tool_profile must be one of" in str(e.get("text", ""))
        ]
        self.assertEqual(len(results), 1, self.printer.events_of_type("result"))

        self.assertEqual(_row(row_a), before_a)
        self.assertEqual(_row(row_b), before_b)
        self.assertEqual(persistence._load_chat_events_by_task_id(row_a), events_a)
        rows = persistence._get_db().execute(
            "SELECT id FROM task_history WHERE task = ?", (_PROMPT,)
        ).fetchall()
        self.assertEqual(len(rows), 1, "the rejected run must not allocate a row")

    def test_stop_before_a_later_subtask_allocates_leaves_the_previous_row(self) -> None:
        """A Stop that lands before the second ``<task>`` has its row persists nothing.

        The agent is reused across the subtasks of one submission and
        its ``last_task_id`` names the first subtask's (completed,
        persisted) row until the second one allocates its own.  A stop
        injected while the second subtask is still inside ``_add_task``
        must end the run on the tab, not rewrite the first row as
        "Task stopped by user" with zeroed metrics.
        """
        second = "v4-1 second subtask"
        prompt = f"<task>{_PROMPT}</task><task>{second}</task>"
        with Pause(
            persistence._add_task.__code__, lambda args: args.get("task") == second,
        ) as hold:
            self.server._cmd_run(self._run_cmd(prompt=prompt))
            self.assertTrue(hold.paused.wait(30), "the second subtask never reached _add_task")
            persistence._flush_chat_events()
            rows = persistence._get_db().execute(
                "SELECT id FROM task_history WHERE task = ?", (_PROMPT,)
            ).fetchall()
            self.assertEqual(len(rows), 1)
            row_a = str(rows[0]["id"])
            before_a = _row(row_a)
            events_a = persistence._load_chat_events_by_task_id(row_a)
            self.assertIn("v4-1 done", str(before_a[1]))
            self.server._stop_task("v4-1-tab")
            # The real Stop watchdog injects its KeyboardInterrupt one
            # second after the stop event; the held worker takes it the
            # moment it is released, still inside ``_add_task``.
            time.sleep(3)
            hold.release()
            self._join_workers()
        persistence._flush_chat_events()
        stopped = [
            e for e in self._terminal_events("result")
            if e.get("text") == "Task stopped by user"
        ]
        self.assertEqual(len(stopped), 1, self.printer.events_of_type("result"))
        self.assertEqual(len(self._terminal_events("task_stopped")), 1)
        self.assertEqual(_row(row_a), before_a)
        self.assertEqual(persistence._load_chat_events_by_task_id(row_a), events_a)
        rows = persistence._get_db().execute(
            "SELECT id FROM task_history WHERE task = ?", (second,)
        ).fetchall()
        self.assertEqual(rows, [], "the stopped subtask must not allocate a row")


if __name__ == "__main__":  # pragma: no cover — manual runs
    unittest.main()
