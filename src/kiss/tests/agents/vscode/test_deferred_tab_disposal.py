# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for deferred agent-state disposal on ``closeTab``.

When the frontend issues ``closeTab`` for a tab whose backend agent
is still running a task (or whose merge view is open), the agent
state must be kept alive so the in-flight work can finish.  Once the
last lifecycle flag (``is_task_active`` / ``is_merging`` /
``task_thread.is_alive()``) drops to false, the state must be
disposed automatically — the frontend does not (and cannot) issue a
second ``closeTab``.

This contract is exercised by the tests below:

1. ``closeTab`` during a running task marks the state
   ``frontend_closed=True`` but does NOT pop the registry entry.
2. When the task ends, ``_dispose_if_closed`` pops the state AND
   ``printer.cleanup_tab`` is called (so per-tab printer state is
   torn down too).
3. ``closeTab`` during an open merge view defers in the same way and
   clearing ``is_merging`` triggers the disposal.
4. ``closeTab`` on an idle tab still disposes immediately.
5. ``_dispose_if_closed`` is a no-op when the frontend has not yet
   closed the tab, even if all lifecycle flags are clear.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


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
    exercising the real registry / ``_subscribers`` / ``cleanup_tab``
    machinery.
    """
    server = VSCodeServer()
    server.printer.broadcast = lambda event: None  # type: ignore[assignment]
    return server


def _register(
    task_id: str,
    tab_id: str,
    *,
    is_task_active: bool = False,
) -> agent_state.AgentState:
    """Register a server-owned state the way a UI-launched run leaves it."""
    state = agent_state.AgentState(
        task_id,
        tab_id=tab_id,
        server_owned=True,
        is_task_active=is_task_active,
    )
    agent_state.register(state)
    return state


class TestDeferredDisposal:
    """``closeTab`` mid-task defers disposal until the task ends."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_close_tab_during_running_task_defers(self) -> None:
        server = _silent_server()
        tab_id = "tab-defer"
        task_id = "task-defer"
        state = _register(task_id, tab_id, is_task_active=True)
        server.printer.subscribe_tab(task_id, tab_id)
        release = threading.Event()

        def fake() -> None:
            release.wait(timeout=5)

        thr = threading.Thread(target=fake, daemon=True)
        state.task_thread = thr
        thr.start()

        server._close_tab(tab_id)

        assert agent_state.get(task_id) is state
        assert state.frontend_closed is True

        server._dispose_if_closed(tab_id)
        assert agent_state.get(task_id) is state

        release.set()
        thr.join(timeout=5)
        with server._state_lock:
            state.task_thread = None
            state.is_task_active = False
        server.printer.cleanup_task(task_id)
        server._dispose_if_closed(tab_id)

        assert agent_state.get(task_id) is None

    def test_close_tab_during_merge_defers(self) -> None:
        server = _silent_server()
        tab_id = "tab-merge-defer"
        task_id = "task-merge-defer"
        state = _register(task_id, tab_id)
        state.is_merging = True

        server._close_tab(tab_id)
        assert agent_state.get(task_id) is state
        assert state.frontend_closed is True

        with server._state_lock:
            state.is_merging = False
        server._dispose_if_closed(tab_id)
        assert agent_state.get(task_id) is None

    def test_close_tab_idle_disposes_immediately(self) -> None:
        server = _silent_server()
        tab_id = "tab-idle"
        _register("task-idle", tab_id)
        server._close_tab(tab_id)
        assert agent_state.get("task-idle") is None

    def test_dispose_if_closed_noop_when_not_flagged(self) -> None:
        server = _silent_server()
        tab_id = "tab-still-open"
        state = _register("task-still-open", tab_id)
        assert state.frontend_closed is False

        server._dispose_if_closed(tab_id)
        assert agent_state.get("task-still-open") is state

    def test_dispose_if_closed_idempotent_and_unknown_tab_safe(self) -> None:
        server = _silent_server()
        server._dispose_if_closed("tab-never-existed")
        server._dispose_if_closed("")

        tab_id = "tab-X"
        _register("task-X", tab_id)
        server._close_tab(tab_id)
        assert agent_state.get("task-X") is None
        server._dispose_if_closed(tab_id)
        assert agent_state.get("task-X") is None

    def test_subscribers_pruned_on_deferred_disposal(self) -> None:
        """When the source tab is closed, ``cleanup_tab`` removes it
        from every task subscriber set so no events leak to it."""
        server = _silent_server()
        source = "tab-src"
        viewer = "tab-viewer"
        task_id = "task-src"
        src_state = _register(task_id, source, is_task_active=True)
        server.printer.subscribe_tab(task_id, source)
        server.printer.subscribe_tab(task_id, viewer)
        assert source in server.printer._subscribers.get(task_id, set())
        assert viewer in server.printer._subscribers.get(task_id, set())

        server._close_tab(source)
        assert agent_state.get(task_id) is src_state

        with server._state_lock:
            src_state.is_task_active = False
        server._dispose_if_closed(source)

        assert agent_state.get(task_id) is None
        assert source not in server.printer._subscribers.get(task_id, set())
        assert viewer in server.printer._subscribers.get(task_id, set())
