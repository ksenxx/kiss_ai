"""Regression tests: a sub-agent task reopened from the history panel
must render with the **same indicator color and icon** as the tab the
backend originally created when the sub-task was launched.

Contract
--------
``_replay_session`` consults
:attr:`kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent.running_agents`
— a task-id-keyed map of currently-running sub-agents — to decide
whether the reopened tab is still running or already done.  The
result is broadcast as ``isDone`` on ``openSubagentTab``.

Tests
-----
1. Completed sub-agent (task id NOT in ``running_agents``) →
   ``openSubagentTab`` event has ``isDone=true``.
2. Currently-running sub-agent (task id IN ``running_agents``) →
   ``openSubagentTab`` event has ``isDone=false``.
3. Frontend handler (static check on ``main.js``) reads ``ev.isDone``
   and sets ``subTab.isDone`` / ``subTab.isRunning`` accordingly.
4. Frontend handler default (no ``isDone`` field) is still "running"
   — preserves the existing fresh-launch path
   (``_run_tasks_parallel``) which doesn't send ``isDone``.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.vscode.server import VSCodeServer

_MAIN_JS = (
    Path(__file__).resolve().parents[4]
    / "kiss" / "agents" / "vscode" / "media" / "main.js"
)


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


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        ev = server.printer._inject_tab_id(event)
        with server.printer._lock:
            server.printer._record_event(ev)
        with lock:
            events.append(ev)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


def _seed_subagent_row(
    *,
    parent_task_id: int,
    chat_id: str,
    description: str,
) -> int:
    task_id, _ = th._add_task(description, chat_id=chat_id)
    th._append_chat_event(
        {"type": "text_delta", "text": "x"}, task_id=task_id,
    )
    th._save_task_extra(
        {
            "model": "m",
            "work_dir": "/tmp",
            "version": "v",
            "tokens": 0,
            "cost": 0.0,
            "is_parallel": False,
            "is_worktree": False,
            "subagent": {"parent_task_id": parent_task_id},
        },
        task_id=task_id,
    )
    return task_id


class TestBackendIsDoneSignal:
    """``_replay_session`` decides ``isDone`` from task-id-keyed
    :attr:`ChatSorcarAgent.running_agents` membership."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        # Ensure no leaked entries from prior tests poison this one.
        ChatSorcarAgent.running_agents.clear()

    def teardown_method(self) -> None:
        ChatSorcarAgent.running_agents.clear()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_completed_subagent_emits_is_done_true(self) -> None:
        chat_id = "chat-done"
        parent_id, _ = th._add_task("parent", chat_id=chat_id)
        task_id = _seed_subagent_row(
            parent_task_id=parent_id,
            chat_id=chat_id,
            description="Completed sub-task",
        )
        server, events = _make_server()
        # running_agents is empty → sub-agent is NOT running.
        assert task_id not in ChatSorcarAgent.running_agents

        server._replay_session(
            chat_id=chat_id,
            tab_id="tab-history-click",
            task_id=task_id,
        )

        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 1, events
        assert opens[0]["isDone"] is True

    def test_running_subagent_emits_is_done_false(self) -> None:
        chat_id = "chat-running"
        parent_id, _ = th._add_task("parent", chat_id=chat_id)
        task_id = _seed_subagent_row(
            parent_task_id=parent_id,
            chat_id=chat_id,
            description="Running sub-task",
        )
        server, events = _make_server()
        # Simulate "sub-agent thread is running": register a dummy
        # entry in running_agents under the sub-agent's own task id.
        ChatSorcarAgent.running_agents[task_id] = object()  # type: ignore[assignment]
        try:
            server._replay_session(
                chat_id=chat_id,
                tab_id="tab-history-click",
                task_id=task_id,
            )
        finally:
            ChatSorcarAgent.running_agents.pop(task_id, None)

        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 1, events
        assert opens[0]["isDone"] is False


class TestFrontendHandlerHonorsIsDone:
    """Static checks on ``media/main.js`` ``case 'openSubagentTab'``."""

    def _handler_source(self) -> str:
        src = _MAIN_JS.read_text(encoding="utf-8")
        idx = src.index("case 'openSubagentTab':")
        end = src.index("case 'subagentDone':", idx)
        return src[idx:end]

    def test_handler_reads_ev_is_done(self) -> None:
        body = self._handler_source()
        assert "ev.isDone" in body, body

    def test_handler_sets_is_done_and_is_running_consistently(
        self,
    ) -> None:
        body = self._handler_source()
        m_done = re.search(
            r"subTab\.isDone\s*=\s*([^;]+);", body,
        )
        m_running = re.search(
            r"subTab\.isRunning\s*=\s*([^;]+);", body,
        )
        assert m_done is not None, body
        assert m_running is not None, body
        done_expr = m_done.group(1).strip()
        running_expr = m_running.group(1).strip()
        assert "subDone" in done_expr or "ev.isDone" in done_expr
        assert running_expr.startswith("!"), running_expr
        assert (
            "subDone" in running_expr or "ev.isDone" in running_expr
        )

    def test_handler_default_is_running_when_is_done_missing(self) -> None:
        body = self._handler_source()
        coerce = (
            "!!ev.isDone" in body
            or "Boolean(ev.isDone)" in body
            or "ev.isDone === true" in body
        )
        assert coerce, body


class TestSubagentTabClassesUnchanged:
    """The rendered tab classes for done vs running come from the
    same ``isDone``/``isRunning`` flags the handler sets, so the
    indicator color (purple ◉ vs green ✓) tracks the backend signal
    automatically."""

    def test_render_tab_bar_branches_on_is_done(self) -> None:
        src = _MAIN_JS.read_text(encoding="utf-8")
        assert "'subagent-indicator' + (tab.isDone ? ' done' : '')" in src
        assert "tab.isDone ? '✓' : '◉'" in src
