# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A stopped/failed task's terminal ``result`` event must be persisted.

Reproduces the "no Result panel in the last task" bug: task_runner's
failure/stop paths broadcast their terminal ``{"type": "result"}``
event stamped with an explicit ``tabId``.  ``WebPrinter.broadcast``
treats every ``tabId``-stamped event as a transient targeted system
event — sent to connected clients but never recorded or persisted.
The Result panel therefore shows up live but VANISHES from the task's
persisted event stream: reloading the webview, loading the task from
history, or scrolling to it as an adjacent task shows no Result panel
at all (observed in ``~/.kiss/sorcar.db``: a user-stopped task's
stream ends ``tool_call -> task_stopped -> followup_suggestion`` with
no ``result`` row).

Fix (two layers, both tested here):

* task_runner broadcasts its terminal failure result ``taskId``-only
  whenever the task row id is known, so it takes the standard
  record -> persist -> per-subscriber fan-out path a SUCCESS result
  takes (``TestTaskRunnerFailureResultPersistedE2E`` drives the REAL
  ``_run_task_inner``); with no row id it falls back to the transient
  ``tabId``-scoped broadcast.
* ``WebPrinter.broadcast`` keeps a defensive net mirroring the
  injected-prompt exception: a ``tabId``-stamped ``result`` event that
  ALSO carries a ``taskId`` gets a tabId-stripped copy recorded +
  persisted under that task (``TestFailedTaskResultEventPersisted``).
"""

import json
import os
import queue
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.web_server import WebPrinter


class _CapturingWebPrinter(WebPrinter):
    """A real ``WebPrinter`` that captures WS payloads instead of sending."""

    def __init__(self) -> None:
        super().__init__()
        self.wire: list[dict[str, Any]] = []
        self._wire_lock = threading.Lock()

    def _send_to_ws_clients(self, data: str) -> None:
        with self._wire_lock:
            self.wire.append(json.loads(data))


def _redirect(tmpdir: str):
    """Redirect the DB to a temp dir and reset the singleton connection."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved):
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class TestFailedTaskResultEventPersisted:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        th._flush_chat_events()
        self.saved = _redirect(self.tmpdir)
        self.printer = _CapturingWebPrinter()

    def teardown_method(self):
        agent_state.agent_states.clear()
        th._flush_chat_events()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _register_task(self, task: str) -> str:
        """Create a task row and register its agent in the state registry."""
        task_id, _ = th._add_task(task, chat_id="chat-1")
        th._flush_chat_events()
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        agent._last_task_id = task_id
        agent_state.register(AgentState(str(task_id), agent=agent))
        return str(task_id)

    def _persisted_events(self, task_id: str) -> list:
        th._flush_chat_events()
        loaded = th._load_chat_events_by_task_id(task_id)
        if not loaded:
            return []
        events = loaded["events"]
        assert isinstance(events, list)
        return events

    def _stop_result_event(self, task_id: str) -> dict[str, Any]:
        """The exact terminal event task_runner broadcasts on user stop."""
        return {
            "type": "result",
            "text": "Task stopped by user",
            "success": False,
            "total_tokens": 0,
            "cost": "$0.0000",
            "step_count": 0,
            "tabId": "tab-1",
            "taskId": task_id,
        }

    def test_stopped_task_result_event_is_persisted(self):
        """A tabId+taskId terminal result must survive into the DB."""
        task_id = self._register_task("stopped task")
        self.printer.broadcast(self._stop_result_event(task_id))
        events = self._persisted_events(task_id)
        results = [e for e in events if e.get("type") == "result"]
        assert results, (
            "terminal result event of a stopped/failed task was not "
            "persisted — the task's replay shows no Result panel"
        )
        rec = results[0]
        assert rec["text"] == "Task stopped by user"
        assert rec["success"] is False
        assert "tabId" not in rec
        assert rec.get("taskId") == task_id

    def test_stopped_task_result_event_still_sent_live(self):
        """The live wire copy must keep its tabId stamp (unchanged)."""
        task_id = self._register_task("stopped task live")
        self.printer.broadcast(self._stop_result_event(task_id))
        sent = [e for e in self.printer.wire if e.get("type") == "result"]
        assert len(sent) == 1
        assert sent[0]["tabId"] == "tab-1"
        assert sent[0]["text"] == "Task stopped by user"

    def test_tabid_result_without_taskid_stays_transient(self):
        """No resolvable task row -> nothing to persist (e.g. the
        no-model failure fires before any task row exists)."""
        task_id = self._register_task("unrelated task")
        ev = self._stop_result_event(task_id)
        del ev["taskId"]
        self.printer.broadcast(ev)
        events = self._persisted_events(task_id)
        assert [e for e in events if e.get("type") == "result"] == []
        sent = [e for e in self.printer.wire if e.get("type") == "result"]
        assert len(sent) == 1

    def test_recorded_transcript_includes_failure_result(self):
        """The in-memory recording (tab-restore transcript) gets the
        tabId-stripped copy too, mirroring the prompt-echo contract.

        Note: in the production stop path the recording has already
        been popped by ``ChatSorcarAgent.run``'s ``stop_recording`` by
        the time task_runner broadcasts the terminal result, so there
        this is a no-op — this asserts the recording contract for
        emitters that broadcast a tab-stamped result WHILE recording
        is live.  Recording is keyed by the printer's thread-local
        ``task_id`` (set by the agent thread at task allocation), so
        this test stamps it exactly as ``ChatSorcarAgent.run`` does.
        """
        task_id = self._register_task("recorded stop")
        self.printer._thread_local.task_id = task_id
        try:
            self.printer.start_recording()
            self.printer.broadcast(self._stop_result_event(task_id))
            recorded = self.printer.stop_recording()
        finally:
            self.printer._thread_local.task_id = None
        results = [e for e in recorded if e.get("type") == "result"]
        assert results and "tabId" not in results[0]


class _NonExceptionBase(BaseException):
    """A ``BaseException`` that is neither ``Exception`` nor
    ``KeyboardInterrupt`` — lands in ``_run_task_inner``'s outer
    ``except BaseException`` block (like ``SystemExit`` /
    ``asyncio.CancelledError``) without killing the test runner."""


class TestTaskRunnerFailureResultPersistedE2E:
    """Drive the REAL ``_run_task_inner`` failure/stop paths end-to-end.

    A stub agent ``run`` allocates a real ``task_history`` row in a
    temp sqlite DB and registers itself with the REAL ``WebPrinter``
    exactly as ``ChatSorcarAgent.run`` does (persist-agent map +
    launcher-tab subscription), then raises.  The tests assert the
    terminal ``result`` event of a stopped / failed task:

    * is persisted under the task (survives replay — THE bug), and
    * is fanned out live to the subscribed launcher tab, and
    * falls back to a transient tabId-scoped broadcast when the
      failure happened before any task row existed.
    """

    def setup_method(self):
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        self.tmpdir = tempfile.mkdtemp()
        th._flush_chat_events()
        self.saved = _redirect(self.tmpdir)
        self.work_dir = str(Path(self.tmpdir) / "wd")
        Path(self.work_dir).mkdir()
        from kiss.server.server import VSCodeServer

        printer = _CapturingWebPrinter()
        self.server = VSCodeServer(printer=printer)
        self.wire = printer.wire

    def teardown_method(self):
        agent_state.agent_states.clear()
        th._flush_chat_events()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_failing_task(
        self,
        tab_id: str,
        exc: BaseException,
        *,
        allocate_row: bool,
    ) -> str | None:
        """Run a task whose agent raises *exc*; return its row id.

        When *allocate_row* is set the stub mirrors
        ``ChatSorcarAgent.run``'s allocation-time wiring (task row,
        ``_last_task_id``, printer-bridge registration, launcher-tab
        subscription via ``_on_task_id_allocated``) before raising.
        """
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        state = AgentState(
            f"pre-{tab_id}",
            agent=agent,
            tab_id=tab_id,
            server_owned=True,
            stop_event=threading.Event(),
        )
        state.user_answer_queue = queue.Queue()
        agent_state.register(state)
        printer = self.server.printer
        allocated: list[str] = []

        def fake_run(**kwargs: Any) -> str:
            if allocate_row:
                task_id, _ = th._add_task("failing task", chat_id="chat-e2e")
                allocated.append(str(task_id))
                agent._last_task_id = task_id
                printer._thread_local.task_id = str(task_id)
                printer.agent_task_allocated(agent, task_id, "chat-e2e")
                on_allocated = kwargs.get("_on_task_id_allocated")
                if on_allocated is not None:
                    on_allocated(task_id, "chat-e2e")
            raise exc

        agent.run = fake_run  # type: ignore[method-assign, assignment]
        self.server._run_task_inner({
            "type": "run",
            "prompt": "failing task",
            "tabId": tab_id,
            "workDir": self.work_dir,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
        })
        return allocated[0] if allocated else None

    def _persisted_results(self, task_id: str) -> list[dict[str, Any]]:
        th._flush_chat_events()
        loaded = th._load_chat_events_by_task_id(task_id)
        events = loaded["events"] if loaded else []
        assert isinstance(events, list)
        return [e for e in events if e.get("type") == "result"]

    def _wire_results(self) -> list[dict[str, Any]]:
        return [e for e in self.wire if e.get("type") == "result"]

    def test_user_stop_result_persisted_and_fanned_out(self):
        """Inner ``KeyboardInterrupt`` path with a task row: the
        terminal result must be persisted AND reach the subscribed
        launcher tab live (the exact "no Result panel in the last
        task" bug)."""
        task_id = self._run_failing_task(
            "e2e-stop-1", KeyboardInterrupt("Stopped by user"),
            allocate_row=True,
        )
        assert task_id is not None
        persisted = self._persisted_results(task_id)
        assert persisted, (
            "stopped task's terminal result was not persisted — its "
            "replay shows no Result panel"
        )
        assert persisted[0]["text"] == "Task stopped by user"
        assert persisted[0]["success"] is False
        assert "tabId" not in persisted[0]
        live = self._wire_results()
        assert any(
            e.get("tabId") == "e2e-stop-1"
            and e.get("taskId") == task_id
            and e.get("text") == "Task stopped by user"
            for e in live
        ), f"no live fan-out copy reached the launcher tab: {live}"

    def test_outer_baseexception_result_persisted(self):
        """Outer ``except BaseException`` path with a task row: same
        durability guarantee."""
        task_id = self._run_failing_task(
            "e2e-outer-1", _NonExceptionBase("simulated cancel"),
            allocate_row=True,
        )
        assert task_id is not None
        persisted = self._persisted_results(task_id)
        assert persisted, (
            "outer-BaseException task's terminal result was not persisted"
        )
        assert "_NonExceptionBase" in persisted[0]["text"]
        assert persisted[0]["success"] is False
        assert "tabId" not in persisted[0]

    def test_inner_failure_without_row_falls_back_to_tab_scoped(self):
        """Inner ``Exception`` path with NO task row: nothing to
        persist under — the result must still reach the launcher tab
        as a transient tabId-scoped broadcast without a taskId."""
        task_id = self._run_failing_task(
            "e2e-norow-1", RuntimeError("boom before row"),
            allocate_row=False,
        )
        assert task_id is None
        live = self._wire_results()
        assert any(
            e.get("tabId") == "e2e-norow-1"
            and "taskId" not in e
            and "boom before row" in e.get("text", "")
            for e in live
        ), f"no tab-scoped fallback result reached the launcher: {live}"

    def test_outer_baseexception_without_row_falls_back_to_tab_scoped(self):
        """Outer ``except BaseException`` path with NO task row: same
        transient tabId-scoped fallback."""
        task_id = self._run_failing_task(
            "e2e-norow-2", _NonExceptionBase("early cancel"),
            allocate_row=False,
        )
        assert task_id is None
        live = self._wire_results()
        assert any(
            e.get("tabId") == "e2e-norow-2"
            and "taskId" not in e
            and "_NonExceptionBase" in e.get("text", "")
            for e in live
        ), f"no tab-scoped fallback result reached the launcher: {live}"
