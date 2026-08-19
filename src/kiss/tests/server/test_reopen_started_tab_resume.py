# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Reproduction probe: a tab that STARTED a task and then resumes its
OWN still-running chat (the close+reopen / reload path) must, on the
real ``VSCodeServer``:

* broadcast ``status running:true`` for that tab during the resume so
  the re-opened webview re-learns the task is running, and
* still accept ``appendUserMessage`` for that tab (inject it, not drop).

This drives the REAL daemon (no stubs of project logic).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _redirect(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class TestStartedTabResumeOwnRunningChat:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        agent_state.agent_states.clear()

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resume_own_running_tab_broadcasts_running_and_accepts_input(
        self,
    ) -> None:
        chat_id = "chat-started"
        tab_id = "tab-started"
        task_id, _ = th._add_task("do a long task", chat_id=chat_id)
        th._append_chat_event(
            {"type": "text_delta", "text": "partial"}, task_id=task_id,
        )

        server = VSCodeServer()
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        subs: list[tuple[Any, str]] = []
        server.printer.subscribe_tab = (  # type: ignore[assignment]
            lambda task_id, tab_id: subs.append((task_id, tab_id))
        )

        state = agent_state.AgentState(
            task_id,
            chat_id=chat_id,
            tab_id=tab_id,
            server_owned=True,
            is_task_active=True,
        )
        agent_state.register(state)

        server._replay_session(chat_id=chat_id, tab_id=tab_id, task_id=task_id)

        running_events = [
            e
            for e in events
            if e.get("type") == "status"
            and e.get("running") is True
            and e.get("tabId") == tab_id
        ]
        assert running_events, (
            "resume of a started tab's OWN running chat must broadcast "
            "status running:true so the re-opened webview re-learns the "
            f"task is running; broadcasts were: {[e.get('type') for e in events]}"
        )

        server._cmd_append_user_message({"tabId": tab_id, "prompt": "hi there"})
        assert state.pending_user_messages == ["hi there"]

    def test_run_for_busy_tab_injects_prompt_instead_of_dropping(
        self,
    ) -> None:
        """A ``run`` command for a tab that already has a live task must
        NOT be silently dropped: the prompt must be injected into the
        running agent's ``pending_user_messages`` (and echoed back as a
        ``prompt`` event), exactly like ``appendUserMessage``.

        This is the root cause of "input ignored during the task" after a
        close+reopen: the re-opened webview can momentarily believe the
        task is NOT running (before the resume's ``status running:true``
        arrives) and therefore send the typed text as a ``submit`` (→
        ``run``) rather than an ``appendUserMessage``.  The daemon — the
        source of truth for whether a task is live — must route that
        ``run`` into the live agent instead of discarding it.
        """
        chat_id = "chat-busy"
        tab_id = "tab-busy"

        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.start()

        server = VSCodeServer()
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]

        state = agent_state.AgentState(
            "task-busy",
            chat_id=chat_id,
            tab_id=tab_id,
            server_owned=True,
            is_task_active=True,
            task_thread=worker,
        )
        agent_state.register(state)

        try:
            server._cmd_run(
                {
                    "tabId": tab_id,
                    "prompt": "also update the docs",
                    "model": "test-model",
                },
            )
        finally:
            release.set()
            worker.join(timeout=5)

        assert state.pending_user_messages == ["also update the docs"], (
            "a run for a tab with a live task must inject the prompt as a "
            "follow-up user message, not silently drop it"
        )
        prompt_echoes = [
            e
            for e in events
            if e.get("type") == "prompt"
            and e.get("tabId") == tab_id
            and e.get("text") == "also update the docs"
        ]
        assert prompt_echoes, "the injected prompt must be echoed to the webview"
        assert state.task_thread is worker
        assert not any(e.get("type") == "clear" for e in events)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
