# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Each subtask's outcome log names its own row (sweep 2026-10-30, V4).

``_run_task_inner`` logged a subtask's outcome ("Agent returned: ...",
"Task stopped by user: ...", "Task failed: ...") with
``task_history_id`` before the subtask's ``finally`` assigned it, so the
first subtask logged ``task_id=None`` and every later one the previous
subtask's row.  An operator matching a stop or failure in the log to a
row in the history database looked at the wrong row.

Reproduced end to end: real multi-``<task>`` runs through ``_cmd_run``
against a local stand-in model endpoint, the log captured with
``assertLogs``, the rows read from the history database, a real Stop
landing while the second subtask waits on the model, and a real SEA
whose ``system_prompt()`` raises inside ``agent.run``.  No mocks,
patches or test doubles.
"""

from __future__ import annotations

import re
import textwrap
import threading
from typing import Any

from kiss.tests.server.parallel_agent_harness import finish_response
from kiss.tests.server.test_chat_viewer_fanout_sweep1030 import _rows_of, _RunHarness

_TAB = "sweep1030-log-tab"
_PROMPT = "<task>first sweep1030 logged subtask</task><task>second sweep1030 logged subtask</task>"

# A genuinely broken SEA: ``system_prompt()`` runs inside ``agent.run``
# after the run's row is allocated, and the ``SeaError`` it becomes
# propagates to the per-subtask ``except Exception`` handler.
_BROKEN_SYSTEM_PROMPT_SEA = textwrap.dedent(
    """
    from kiss.agents.seas.base.base_sea import BaseSea


    class Sea(BaseSea):
        def system_prompt(self, system_prompt):
            raise ValueError("sweep1030 broken system_prompt")
    """
)


class TestSubtaskOutcomeLogs(_RunHarness):
    """The outcome log of every subtask carries that subtask's row id."""

    def setUp(self) -> None:
        super().setUp()
        self.calls = 0
        self.calls_lock = threading.Lock()
        self.park_call = 0
        self.parked = threading.Event()

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        # One ``finish`` call per subtask: agent-loop call N (it offers
        # tools; a task classifier's call does not) is subtask N's.
        if not request.get("tools"):
            return finish_response("sweep1030 logged")
        with self.calls_lock:
            self.calls += 1
            park = self.calls == self.park_call
        if park:
            self.parked.set()
            self.release.wait(60)
        return finish_response("sweep1030 logged")

    def _logged_rows(self, output: list[str], outcome: str) -> list[str]:
        pattern = rf"{outcome}: tab_id={_TAB} task_id=(\S+)"
        return [m.group(1) for line in output if (m := re.search(pattern, line))]

    def test_each_finished_subtask_logs_its_own_row(self) -> None:
        chat_id = "chat-sweep1030-logs"
        with self.assertLogs("kiss.server.task_runner", level="INFO") as logs:
            self._finish(self._start(_TAB, chat_id, _PROMPT))
        self.assertEqual(
            self._logged_rows(logs.output, "Agent returned"),
            _rows_of(chat_id),
            "BUG: a subtask's outcome log names another subtask's row",
        )

    def test_a_stopped_subtask_logs_its_own_row(self) -> None:
        chat_id = "chat-sweep1030-logs-stop"
        self.park_call = 2
        with self.assertLogs("kiss.server.task_runner", level="INFO") as logs:
            worker = self._start(_TAB, chat_id, _PROMPT)
            self.assertTrue(self.parked.wait(60), "the second subtask never called the model")
            self.server._stop_task(_TAB)
            self._finish(worker)
        self.release.set()
        rows = _rows_of(chat_id)
        self.assertEqual(len(rows), 2)
        stopped = self._logged_rows(logs.output, r"Task stopped by user(?: \(agent raised \w+\))?")
        self.assertEqual(
            stopped, [rows[1]], "BUG: the stopped subtask's log names the previous subtask's row"
        )

    def test_a_failed_subtask_logs_its_own_row(self) -> None:
        # An SEA run is one atomic task, so the failing subtask is the
        # run's first: before the fix its log said ``task_id=None``.
        chat_id = "chat-sweep1030-logs-fail"
        script = self.home.tmpdir / "broken_system_prompt_sea.py"
        script.write_text(_BROKEN_SYSTEM_PROMPT_SEA, encoding="utf-8")
        with self.assertLogs("kiss.server.task_runner", level="INFO") as logs:
            self._finish(self._start(_TAB, chat_id, _PROMPT, seaPath=str(script)))
        rows = _rows_of(chat_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            self._logged_rows(logs.output, "Task failed"),
            rows,
            "BUG: the failed subtask's log does not name its own row",
        )
