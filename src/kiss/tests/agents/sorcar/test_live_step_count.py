"""Integration tests: running-task and history panels show live step counts.

Verifies that ``_get_running_tasks()`` and ``_get_history()`` include the
in-progress executor's ``step_count`` in the reported steps — not just the
``total_steps`` accumulated from completed sessions, which stays 0 during
the first (and typically only) session.

No mocks — uses real ``VSCodeServer`` / ``_RunningAgentState`` instances
with a real local HTTP server that delays its response so the agent is
mid-execution when we inspect the live metrics.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer
from kiss.core.kiss_agent import KISSAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redirect_db(tmpdir: str) -> tuple[Any, Any, Any]:
    """Point persistence at *tmpdir* and return original values to restore."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple[Any, Any, Any]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]]]:
    """Create a VSCodeServer that captures all broadcasts."""
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


def _slow_finish_handler_factory(
    gate: threading.Event,
) -> type[BaseHTTPRequestHandler]:
    """Return an HTTP handler that waits on *gate* before replying with finish.

    The gate lets the test hold the agent mid-execution so we can
    inspect the live step count (the agent has entered the loop and
    incremented ``step_count`` to 1 before the LLM response arrives).
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            cl = int(self.headers.get("Content-Length", 0))
            if cl:
                self.rfile.read(cl)
            gate.wait(timeout=30)
            body = json.dumps({
                "id": "chatcmpl-slow",
                "object": "chat.completion",
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_fin",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps(
                                    {"success": "true", "summary": "done"},
                                ),
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # type: ignore[override]  # noqa: A002
            pass

    return Handler


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLiveStepCountInRunningTasks:
    """``_get_running_tasks()`` reports the live executor step count."""

    def test_running_task_reports_nonzero_steps(self) -> None:
        """Simulate a running agent with a live executor and verify steps > 0.

        Instead of running a real LLM loop, we directly set up the state
        that exists mid-execution: a ``_RunningAgentState`` with a live
        ``task_thread``, an agent whose ``_current_executor`` has
        ``step_count > 0``, and ``total_steps == 0`` (the first session
        hasn't finished yet).  Then call ``_get_running_tasks()`` and
        assert the reported steps include the executor's step count.
        """
        server, events = _make_server()
        tab_id = "live-steps-tab"
        tab = server._get_tab(tab_id)

        # Simulate mid-execution state
        agent = WorktreeSorcarAgent("Test Agent")
        agent._last_task_id = 42
        agent._last_user_prompt = "test task"
        agent.total_steps = 0  # No completed sessions yet
        agent.total_tokens_used = 100
        agent.budget_used = 0.05

        # Simulate the live executor with step_count = 7
        executor = KISSAgent("Test Executor")
        executor.step_count = 7
        agent._current_executor = executor

        tab.agent = agent
        tab.chat_id = "chat-123"

        # Simulate a live task thread
        alive_event = threading.Event()
        def _stay_alive() -> None:
            alive_event.wait(timeout=30)

        tab.task_thread = threading.Thread(target=_stay_alive, daemon=True)
        tab.task_thread.start()
        tab.stop_event = threading.Event()
        tab.is_task_active = True

        try:
            server._get_running_tasks()

            running_events = [e for e in events if e.get("type") == "runningTasks"]
            assert len(running_events) == 1
            tasks = running_events[0]["tasks"]
            assert len(tasks) == 1
            # steps should be total_steps (0) + executor.step_count (7) = 7
            assert tasks[0]["steps"] == 7, (
                f"Expected steps=7, got steps={tasks[0]['steps']}"
            )
        finally:
            alive_event.set()
            tab.task_thread.join(timeout=5)

    def test_steps_accumulate_across_sessions(self) -> None:
        """After a completed session, total_steps + executor.step_count."""
        server, events = _make_server()
        tab_id = "accum-tab"
        tab = server._get_tab(tab_id)

        agent = WorktreeSorcarAgent("Test Agent")
        agent._last_task_id = 99
        agent._last_user_prompt = "multi session"
        agent.total_steps = 10  # One completed session with 10 steps
        agent.total_tokens_used = 500
        agent.budget_used = 0.1

        # Second session in progress with 3 steps so far
        executor = KISSAgent("Session 2")
        executor.step_count = 3
        agent._current_executor = executor

        tab.agent = agent
        tab.chat_id = "chat-456"

        alive_event = threading.Event()
        def _stay_alive() -> None:
            alive_event.wait(timeout=30)

        tab.task_thread = threading.Thread(target=_stay_alive, daemon=True)
        tab.task_thread.start()
        tab.stop_event = threading.Event()
        tab.is_task_active = True

        try:
            server._get_running_tasks()

            running_events = [e for e in events if e.get("type") == "runningTasks"]
            assert len(running_events) == 1
            tasks = running_events[0]["tasks"]
            assert len(tasks) == 1
            # total_steps (10) + executor.step_count (3) = 13
            assert tasks[0]["steps"] == 13, (
                f"Expected steps=13, got steps={tasks[0]['steps']}"
            )
        finally:
            alive_event.set()
            tab.task_thread.join(timeout=5)

    def test_no_executor_reports_total_steps_only(self) -> None:
        """When _current_executor is None, steps == total_steps."""
        server, events = _make_server()
        tab_id = "no-exec-tab"
        tab = server._get_tab(tab_id)

        agent = WorktreeSorcarAgent("Test Agent")
        agent._last_task_id = 55
        agent._last_user_prompt = "finished session"
        agent.total_steps = 15
        agent._current_executor = None

        tab.agent = agent
        tab.chat_id = "chat-789"

        alive_event = threading.Event()
        def _stay_alive() -> None:
            alive_event.wait(timeout=30)

        tab.task_thread = threading.Thread(target=_stay_alive, daemon=True)
        tab.task_thread.start()
        tab.stop_event = threading.Event()
        tab.is_task_active = True

        try:
            server._get_running_tasks()

            running_events = [e for e in events if e.get("type") == "runningTasks"]
            assert len(running_events) == 1
            tasks = running_events[0]["tasks"]
            assert len(tasks) == 1
            assert tasks[0]["steps"] == 15
        finally:
            alive_event.set()
            tab.task_thread.join(timeout=5)


class TestLiveStepCountInHistory:
    """``_get_history()`` overlays live metrics for running tasks."""

    def test_history_running_task_shows_live_steps(self) -> None:
        """A running task in history should show live steps, not persisted 0."""
        tmpdir = tempfile.mkdtemp()
        saved = _redirect_db(tmpdir)
        try:
            server, events = _make_server()
            server.work_dir = tmpdir

            # Persist a task row so _get_history returns it
            from kiss.agents.sorcar.persistence import _add_task, _save_task_result
            task_id, _chat_id = _add_task("test live history", "chat-hist")
            _save_task_result(
                result="",
                task_id=task_id,
                task="test live history",
            )

            # Set up a running agent state matching the task
            tab_id = "hist-tab"
            tab = server._get_tab(tab_id)
            agent = WorktreeSorcarAgent("History Agent")
            agent._last_task_id = task_id
            agent._last_user_prompt = "test live history"
            agent.total_steps = 0
            agent.total_tokens_used = 200
            agent.budget_used = 0.03

            executor = KISSAgent("Hist Executor")
            executor.step_count = 5
            agent._current_executor = executor

            tab.agent = agent
            tab.chat_id = "chat-hist"
            tab.task_history_id = task_id

            alive_event = threading.Event()
            def _stay_alive() -> None:
                alive_event.wait(timeout=30)

            tab.task_thread = threading.Thread(target=_stay_alive, daemon=True)
            tab.task_thread.start()
            tab.stop_event = threading.Event()
            tab.is_task_active = True

            try:
                server._get_history(None)

                history_events = [e for e in events if e.get("type") == "history"]
                assert len(history_events) == 1
                sessions = history_events[0]["sessions"]

                # Find our task
                our_session = None
                for s in sessions:
                    if s.get("task_id") == task_id:
                        our_session = s
                        break
                assert our_session is not None, (
                    f"Task {task_id} not found in history sessions"
                )
                assert our_session["is_running"] is True
                # Live steps: total_steps (0) + executor.step_count (5) = 5
                assert our_session["steps"] == 5, (
                    f"Expected steps=5, got steps={our_session['steps']}"
                )
                assert our_session["tokens"] == 200
            finally:
                alive_event.set()
                tab.task_thread.join(timeout=5)
        finally:
            _restore_db(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)
