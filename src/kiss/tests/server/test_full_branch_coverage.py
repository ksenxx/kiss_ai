# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for 100% branch coverage of sorcar/ and vscode/ modules.

No mocks, patches, fakes, or test doubles. All tests use real objects.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path

from kiss.agents.sorcar import persistence as th
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _git(tmpdir: str, *args: str) -> None:
    """Run a git command in tmpdir."""
    subprocess.run(["git", *args], cwd=tmpdir, capture_output=True, check=True)


class TestUserAnswerDrain:
    """Cover drain-stale-answers path in userAnswer handler (lines 169-170)."""

    def test_user_answer_drains_stale(self) -> None:
        """Pre-filling a task queue before userAnswer should drain stale item."""
        import queue as queue_mod

        server = VSCodeServer()
        q: queue_mod.Queue[str] = queue_mod.Queue(maxsize=1)
        q.put("stale")
        state = agent_state.AgentState(
            "task-7", tab_id="7", server_owned=True, is_task_active=True,
        )
        state.user_answer_queue = q
        agent_state.register(state)
        try:
            server._handle_command(
                {"type": "userAnswer", "answer": "new", "tabId": "7"}
            )
            answer = q.get_nowait()
        finally:
            agent_state.unregister(state.task_id, state)
        assert answer == "new"


class TestResumeSessionWithTask:
    """Cover resumeSession handler calling _replay_session (line 179)."""

    def test_resume_session_with_task(self) -> None:
        """resumeSession with a non-empty chatId calls _replay_session."""
        server = VSCodeServer()
        events: list[dict[str, object]] = []
        orig = server.printer.broadcast

        def capture(ev: dict[str, object]) -> None:
            events.append(ev)
            orig(ev)

        server.printer.broadcast = capture  # type: ignore[assignment]
        server._handle_command(
            {"type": "resumeSession", "chatId": "999999"}
        )
        err = [e for e in events if e.get("type") == "error"]
        assert len(err) == 0


class TestReplaySessionWithEvents:
    """Cover successful _replay_session path (lines 554-555)."""

    def test_replay_session_with_recorded_events(self, tmp_path: Path) -> None:
        """_replay_session broadcasts task_events when events exist."""
        orig_dir = th._KISS_DIR
        orig_db = th._DB_PATH
        orig_conn = th._db_conn
        try:
            th._db_conn = None
            th._KISS_DIR = tmp_path
            th._DB_PATH = tmp_path / "sorcar.db"

            task_text = "test-replay-session-task"
            task_id, chat_id = th._add_task(task_text, chat_id="0")
            test_events: list[dict[str, object]] = [
                {"type": "text_delta", "text": "hello"},
                {"type": "result", "summary": "done"},
            ]
            for ev in test_events:
                th._append_chat_event(ev, task_id=task_id)

            server = VSCodeServer()
            captured: list[dict[str, object]] = []
            orig_broadcast = server.printer.broadcast

            def capture(ev: dict[str, object]) -> None:
                captured.append(ev)
                orig_broadcast(ev)

            server.printer.broadcast = capture  # type: ignore[assignment]

            server._replay_session(chat_id, tab_id="tab-replay")

            task_ev = [e for e in captured if e.get("type") == "task_events"]
            assert len(task_ev) == 1
            ev_list = task_ev[0].get("events", [])
            assert isinstance(ev_list, list)
            # The synthesized task_settings event leads the replay.
            assert len(ev_list) == 3
            assert ev_list[0].get("type") == "task_settings"
        finally:
            th._close_db()
            th._db_conn = orig_conn
            th._KISS_DIR = orig_dir
            th._DB_PATH = orig_db


class TestAwaitUserResponseLoop:
    """Cover _await_user_response loop continuing (466->462)."""

    def test_await_user_response_delayed(self) -> None:
        """Answer arriving after first timeout iteration covers loop branch."""
        import queue as queue_mod

        server = VSCodeServer()
        stop_event = threading.Event()
        server.printer._thread_local.stop_event = stop_event
        server.printer._thread_local.task_id = "42"
        q: queue_mod.Queue[str] = queue_mod.Queue(maxsize=1)
        state = agent_state.AgentState(
            "42", tab_id="42", server_owned=True, is_task_active=True,
        )
        state.user_answer_queue = q
        agent_state.register(state)
        server.printer.subscribe_tab("42", "42")

        def delayed_answer() -> None:
            time.sleep(1.0)
            q.put("delayed")

        t = threading.Thread(target=delayed_answer, daemon=True)
        t.start()
        try:
            result = server._await_user_response()
        finally:
            agent_state.unregister(state.task_id, state)
            server.printer._thread_local.task_id = ""
        assert result == "delayed"
        t.join(timeout=2)


class TestRunTaskDrain:
    """Cover drain of stale answers at start of _run_task_inner (lines 268-269)."""

    def test_run_task_creates_fresh_queue(self) -> None:
        """Each task gets a fresh user_answer queue (RC8 fix)."""
        server = VSCodeServer()
        captured: list[dict[str, object]] = []
        orig = server.printer.broadcast

        def cap(ev: dict[str, object]) -> None:
            captured.append(ev)
            orig(ev)

        server.printer.broadcast = cap  # type: ignore[assignment]

        tab_id = "1"
        stale_q: queue.Queue[str] = queue.Queue(maxsize=1)
        stale_q.put("stale-answer")
        prev = agent_state.AgentState(
            "task-stale", tab_id=tab_id, server_owned=True,
        )
        prev.user_answer_queue = stale_q
        agent_state.register(prev)

        try:
            server._handle_command({
                "type": "run",
                "prompt": "test drain",
                "model": "nonexistent-model",
                "tabId": tab_id,
            })

            state = agent_state.find_by_tab(tab_id)
            thread = state.task_thread if state is not None else None
            if thread:
                thread.join(timeout=30)
        finally:
            with agent_state.STATE_LOCK:
                for st in agent_state.snapshot():
                    if st.tab_id == tab_id:
                        agent_state.unregister(st.task_id, st)

        status_events = [
            e for e in captured
            if e.get("type") == "status" and e.get("running") is False
        ]
        assert len(status_events) >= 1
