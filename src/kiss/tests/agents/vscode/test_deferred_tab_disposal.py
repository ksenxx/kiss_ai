"""Integration tests for deferred ``_RunningAgentState`` disposal.

When the frontend issues ``closeTab`` for a tab whose backend agent
is still running a task (or whose merge view is open), the tab state
must be kept alive so the in-flight work can finish.  Once the last
lifecycle flag (``is_task_active`` / ``is_merging`` /
``task_thread.is_alive()``) drops to false, the state must be
disposed automatically — the frontend does not (and cannot) issue a
second ``closeTab``.

This contract is exercised by the tests below:

1. ``closeTab`` during a running task marks the tab
   ``frontend_closed=True`` but does NOT pop ``_running_agent_states``.
2. When the task ends (``_run_task`` finally), ``_dispose_if_closed``
   pops the tab AND ``printer.cleanup_tab`` is called (so per-tab
   printer state is torn down too).
3. ``closeTab`` during an open merge view defers in the same way and
   ``_finish_merge`` triggers the disposal.
4. ``closeTab`` on an idle tab still disposes immediately (the legacy
   path is preserved).
5. ``_dispose_if_closed`` is a no-op when the frontend has not yet
   closed the tab, even if all lifecycle flags are clear.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.server import VSCodeServer


def _redirect(tmpdir: str) -> tuple[Path, object, Path]:
    """Redirect the persistence DB to a temp dir; return saved state."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old  # type: ignore[return-value]


def _restore(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _silent_server() -> VSCodeServer:
    """``VSCodeServer`` whose printer.broadcast is a no-op (for tests).

    Avoids polluting test stdout with JSON event lines while still
    exercising the real ``_running_agent_states`` / ``_subscribers`` /
    ``cleanup_tab`` machinery.
    """
    server = VSCodeServer()
    server.printer.broadcast = lambda event: None  # type: ignore[assignment]
    return server


class TestDeferredDisposal:
    """``closeTab`` mid-task defers disposal until the task ends."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_close_tab_during_running_task_defers(self) -> None:
        server = _silent_server()
        tab_id = "tab-defer"
        task_id = "task-defer"
        tab = server._get_tab(tab_id)
        server.printer.subscribe_tab(task_id, tab_id)
        # Simulate "task running" — same flags _run_task sets.
        release = threading.Event()
        tab.is_task_active = True

        def fake() -> None:
            release.wait(timeout=5)

        thr = threading.Thread(target=fake, daemon=True)
        tab.task_thread = thr
        thr.start()

        # Also pre-register a printer-side entry so we can verify
        # cleanup_task is called at task-end time.
        server.printer._persist_agents[task_id] = tab.agent  # type: ignore[assignment]

        # User closes the tab while the task is still running.
        server._close_tab(tab_id)

        # Deferred: tab state is still present and flagged.  The
        # per-task printer state lives on because the agent thread
        # has not yet called ``cleanup_task``.
        assert tab_id in server._running_agent_states
        assert tab.frontend_closed is True
        assert task_id in server.printer._persist_agents

        # Disposal MUST NOT happen while the task is still active.
        server._dispose_if_closed(tab_id)
        assert tab_id in server._running_agent_states

        # Task ends — mirror _run_task's finally block exactly.
        release.set()
        thr.join(timeout=5)
        with server._state_lock:
            tab.task_thread = None
            tab.is_task_active = False
        # The agent thread's finally block calls ``cleanup_task``.
        server.printer.cleanup_task(task_id)
        server._dispose_if_closed(tab_id)

        # Now the tab state must be gone, and per-task printer state
        # too.
        assert tab_id not in server._running_agent_states
        assert task_id not in server.printer._persist_agents

    def test_close_tab_during_merge_defers(self) -> None:
        server = _silent_server()
        tab_id = "tab-merge-defer"
        tab = server._get_tab(tab_id)
        tab.is_merging = True

        server._close_tab(tab_id)
        # Deferred — tab is still here, just flagged.
        assert tab_id in server._running_agent_states
        assert tab.frontend_closed is True

        # Merge ends.
        with server._state_lock:
            tab.is_merging = False
        server._dispose_if_closed(tab_id)
        assert tab_id not in server._running_agent_states

    def test_close_tab_idle_disposes_immediately(self) -> None:
        server = _silent_server()
        tab_id = "tab-idle"
        server._get_tab(tab_id)
        # No lifecycle flags raised.
        server._close_tab(tab_id)
        assert tab_id not in server._running_agent_states

    def test_dispose_if_closed_noop_when_not_flagged(self) -> None:
        server = _silent_server()
        tab_id = "tab-still-open"
        tab = server._get_tab(tab_id)
        assert tab.frontend_closed is False

        # Frontend never closed the tab — disposal must NOT happen
        # even though every lifecycle flag is clear.
        server._dispose_if_closed(tab_id)
        assert tab_id in server._running_agent_states

    def test_dispose_if_closed_idempotent_and_unknown_tab_safe(self) -> None:
        server = _silent_server()
        # No-op for unknown tabs.
        server._dispose_if_closed("tab-never-existed")
        server._dispose_if_closed("")

        # No-op for an already disposed tab.
        tab_id = "tab-X"
        server._get_tab(tab_id)
        server._close_tab(tab_id)
        assert tab_id not in server._running_agent_states
        server._dispose_if_closed(tab_id)
        assert tab_id not in server._running_agent_states

    def test_subscribers_pruned_on_deferred_disposal(self) -> None:
        """When the source tab is closed, ``cleanup_tab`` removes it
        from every task subscriber set so no events leak to it."""
        server = _silent_server()
        source = "tab-src"
        viewer = "tab-viewer"
        task_id = "task-src"
        src_tab = server._get_tab(source)
        server._get_tab(viewer)
        server.printer.subscribe_tab(task_id, source)
        server.printer.subscribe_tab(task_id, viewer)
        assert source in server.printer._subscribers.get(task_id, set())
        assert viewer in server.printer._subscribers.get(task_id, set())

        src_tab.is_task_active = True
        server._close_tab(source)
        assert source in server._running_agent_states  # deferred

        with server._state_lock:
            src_tab.is_task_active = False
        server._dispose_if_closed(source)

        assert source not in server._running_agent_states
        # cleanup_tab dropped the source tab from the subscriber set.
        # The viewer tab remains subscribed.
        assert source not in server.printer._subscribers.get(task_id, set())
        assert viewer in server.printer._subscribers.get(task_id, set())
