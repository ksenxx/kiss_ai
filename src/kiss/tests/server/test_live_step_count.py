# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: the history panel shows the live step count.

Verifies that ``_get_history()`` includes the in-progress executor's
``step_count`` in the reported steps — not just the ``total_steps``
accumulated from completed sessions, which stays 0 during the first
(and typically only) session.

No mocks — uses real ``VSCodeServer`` / ``AgentState`` instances.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.kiss_agent import KISSAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer


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



class TestLiveStepCountInHistory:
    """``_get_history()`` overlays live metrics for running tasks."""

    def test_history_running_task_shows_live_steps(self) -> None:
        """A running task in history should show live steps, not persisted 0."""
        tmpdir = tempfile.mkdtemp()
        saved = _redirect_db(tmpdir)
        try:
            server, events = _make_server()
            server.work_dir = tmpdir

            from kiss.agents.sorcar.persistence import _add_task, _save_task_result
            task_id, _chat_id = _add_task("test live history", "chat-hist")
            _save_task_result(
                result="",
                task_id=task_id,
                task="test live history",
            )

            tab_id = "hist-tab"
            agent = WorktreeSorcarAgent("History Agent")
            agent._last_task_id = task_id
            agent._last_user_prompt = "test live history"
            agent.total_steps = 0
            agent.total_tokens_used = 200
            agent.budget_used = 0.03

            executor = KISSAgent("Hist Executor")
            executor.step_count = 5
            agent._current_executor = executor

            alive_event = threading.Event()
            def _stay_alive() -> None:
                alive_event.wait(timeout=30)

            task_thread = threading.Thread(target=_stay_alive, daemon=True)
            task_thread.start()
            state = AgentState(
                str(task_id),
                agent=agent,
                chat_id="chat-hist",
                tab_id=tab_id,
                server_owned=True,
                stop_event=threading.Event(),
                task_thread=task_thread,
                is_task_active=True,
            )
            agent_state.register(state)

            try:
                server._get_history(None)

                history_events = [e for e in events if e.get("type") == "history"]
                assert len(history_events) == 1
                sessions = history_events[0]["sessions"]

                our_session = None
                for s in sessions:
                    if s.get("task_id") == task_id:
                        our_session = s
                        break
                assert our_session is not None, (
                    f"Task {task_id} not found in history sessions"
                )
                assert our_session["is_running"] is True
                assert our_session["steps"] == 5, (
                    f"Expected steps=5, got steps={our_session['steps']}"
                )
                assert our_session["tokens"] == 200
            finally:
                alive_event.set()
                task_thread.join(timeout=5)
                agent_state.unregister(str(task_id), state)
        finally:
            _restore_db(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)
