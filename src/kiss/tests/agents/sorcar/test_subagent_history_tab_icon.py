"""Regression tests: a sub-agent task reopened from the history panel
must render with the **same indicator color and icon** as the tab the
backend originally created when the sub-task was launched.

Bug
---
``VSCodeServer._replay_session`` (server.py) broadcast an
``openSubagentTab`` event with no ``isDone`` field.  The frontend
handler (``case 'openSubagentTab'`` in ``media/main.js``)
unconditionally set ``subTab.isDone = false`` and
``subTab.isRunning = true``, so the tab's indicator pulsed the running
``◉`` purple glyph forever.  No later ``subagentDone`` event ever
arrived for an already-finished sub-agent (the agent thread is gone),
so the tab never flipped to the completed ``✓`` green icon — the
indicator color and shape did NOT match the originally-launched tab
(which ended on ``✓`` green via the ``subagentDone`` event from
``ChatSorcarAgent._run_tasks_parallel``).

Fix
---
``_replay_session`` now consults ``printer._persist_agents`` for the
original ``sub_tab_id`` from the persisted ``extra.subagent`` payload.
An entry there means the sub-agent thread is still running
(registered just before ``agent.run()``, popped in the ``finally``);
absent means the sub-agent has finished.  The result is broadcast as
``isDone`` on ``openSubagentTab``, and the frontend handler honors
it: ``isDone=true`` → ``subTab.isDone=true`` (✓ green, no
animation), ``isDone=false`` → ``subTab.isRunning=true`` (◉ purple
pulse).

Tests
-----
1. Completed sub-agent → ``openSubagentTab`` event has
   ``isDone=true``.
2. Currently-running sub-agent (entry in ``_persist_agents``) →
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
    parent_tab_id: str,
    sub_tab_id: str,
    task_index: int,
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
            "subagent": {
                "tab_id": sub_tab_id,
                "parent_tab_id": parent_tab_id,
                "task_index": task_index,
                "description": description[:200],
            },
        },
        task_id=task_id,
    )
    return task_id


class TestBackendIsDoneSignal:
    """``_replay_session`` decides ``isDone`` from
    ``printer._persist_agents`` membership for the original
    sub_tab_id."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_completed_subagent_emits_is_done_true(self) -> None:
        chat_id = "chat-done"
        original_sub_tab_id = "tab-parent__sub_0"
        task_id = _seed_subagent_row(
            parent_tab_id="tab-parent",
            sub_tab_id=original_sub_tab_id,
            task_index=0,
            chat_id=chat_id,
            description="Completed sub-task",
        )
        server, events = _make_server()
        # _persist_agents is empty → sub-agent is NOT running.
        assert (
            original_sub_tab_id not in server.printer._persist_agents
        )

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
        original_sub_tab_id = "tab-parent__sub_1"
        task_id = _seed_subagent_row(
            parent_tab_id="tab-parent",
            sub_tab_id=original_sub_tab_id,
            task_index=1,
            chat_id=chat_id,
            description="Running sub-task",
        )
        server, events = _make_server()
        # Simulate "sub-agent thread is running": register a dummy
        # entry in _persist_agents under the original sub_tab_id.
        server.printer._persist_agents[original_sub_tab_id] = object()

        server._replay_session(
            chat_id=chat_id,
            tab_id="tab-history-click",
            task_id=task_id,
        )

        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 1, events
        assert opens[0]["isDone"] is False


class TestFrontendHandlerHonorsIsDone:
    """Static checks on ``media/main.js`` ``case 'openSubagentTab'``.

    The webview JS is evaluated inside the VS Code webview so we
    cannot execute it here, but the handler's logic is a small chunk
    of source we can lock down with patterns.
    """

    def _handler_source(self) -> str:
        src = _MAIN_JS.read_text(encoding="utf-8")
        # Slice out the openSubagentTab case body for focused checks.
        idx = src.index("case 'openSubagentTab':")
        end = src.index("case 'subagentDone':", idx)
        return src[idx:end]

    def test_handler_reads_ev_is_done(self) -> None:
        body = self._handler_source()
        # The handler must consult ev.isDone — exact name, not a
        # neighbor.  Allow either ``ev.isDone`` or destructuring.
        assert "ev.isDone" in body, body

    def test_handler_sets_is_done_and_is_running_consistently(
        self,
    ) -> None:
        body = self._handler_source()
        # Capture the assignments: isDone and isRunning are coupled —
        # done implies not-running and vice versa.  Match the exact
        # pattern the fix introduces.
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
        # done_expr is sourced from ev.isDone (boolean-coerced).
        assert "subDone" in done_expr or "ev.isDone" in done_expr
        # running_expr is the negation of the done flag.
        assert running_expr.startswith("!"), running_expr
        assert (
            "subDone" in running_expr or "ev.isDone" in running_expr
        )

    def test_handler_default_is_running_when_is_done_missing(self) -> None:
        """Fresh-launch path (``_run_tasks_parallel``) does NOT send
        ``isDone`` — the handler must default to "running" (◉).
        """
        body = self._handler_source()
        # Boolean coercion via !! treats missing fields as false →
        # subDone=false → isRunning=true, isDone=false.  Look for the
        # !! coercion (or an equivalent ``=== true`` / ``Boolean()``).
        coerce = (
            "!!ev.isDone" in body
            or "Boolean(ev.isDone)" in body
            or "ev.isDone === true" in body
        )
        assert coerce, body


class TestSubagentTabClassesUnchanged:
    """Sanity: the rendered tab classes for done vs running come from
    the same ``isDone``/``isRunning`` flags the handler now sets, so
    the indicator color (purple ◉ vs green ✓) tracks the backend
    signal automatically — no separate CSS path."""

    def test_render_tab_bar_branches_on_is_done(self) -> None:
        src = _MAIN_JS.read_text(encoding="utf-8")
        # The renderTabBar branch that distinguishes done vs running
        # for a sub-agent tab.
        assert "'subagent-indicator' + (tab.isDone ? ' done' : '')" in src
        assert "tab.isDone ? '✓' : '◉'" in src
