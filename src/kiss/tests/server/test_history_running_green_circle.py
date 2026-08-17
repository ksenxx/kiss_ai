# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Server-only tests extracted from ``kiss.tests.agents.vscode.test_history_running_green_circle``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).
"""


from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as th
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _history_event_from_real_backend(
    *,
    result: str = "",
    fake_running_task_id: str | None = None,
) -> dict[str, Any]:
    """Persist one task and return the real ``getHistory`` broadcast.

    When *fake_running_task_id* is set, a synthetic
    :class:`agent_state.AgentState` keyed by the persisted task's row
    id and whose ``task_thread`` is an alive daemon thread
    is registered before the ``getHistory`` call.  This drives the
    real :meth:`VSCodeServer._get_running_task_ids` to flag the row as
    running — proving the backend → ``is_running`` plumbing works
    against a real DB row instead of fabricating ``is_running`` in
    the browser-side test.

    When *fake_running_task_id* equals ``-1``, the helper resolves the
    sentinel to the just-persisted task's auto-assigned id so callers
    don't have to predict it.
    """
    tmp = tempfile.mkdtemp(prefix="kiss-history-running-test-")
    orig_db_path = th._DB_PATH  # type: ignore[attr-defined]
    th._close_db()
    th._DB_PATH = Path(tmp) / "sorcar.db"  # type: ignore[attr-defined]

    stop = threading.Event()
    worker: threading.Thread | None = None
    fake_tab_id = "__test_running_tab__"
    try:
        server = VSCodeServer()
        server.work_dir = tmp
        events: list[dict[str, Any]] = []
        lock = threading.Lock()
        orig_broadcast = server.printer.broadcast

        def capture(event: dict[str, Any]) -> None:
            with lock:
                events.append(dict(event))
            orig_broadcast(event)

        server.printer.broadcast = capture  # type: ignore[assignment]

        task_id, _ = th._add_task("running task from real backend")
        if result:
            th._save_task_result(result=result, task_id=task_id)

        if fake_running_task_id is not None:
            resolved = (
                task_id if fake_running_task_id == "-1"
                else fake_running_task_id
            )
            state = agent_state.AgentState(
                str(resolved), tab_id=fake_tab_id, server_owned=True,
            )
            worker = threading.Thread(
                target=stop.wait, name="kiss-test-fake-worker", daemon=True,
            )
            worker.start()
            for _ in range(50):
                if worker.is_alive():
                    break
                time.sleep(0.01)
            state.task_thread = worker
            agent_state.register(state)

        server._handle_command({"type": "getHistory"})

        with lock:
            for event in reversed(events):
                if event.get("type") == "history":
                    return dict(event)
        raise AssertionError("getHistory did not broadcast a history event")
    finally:
        if worker is not None:
            stop.set()
            worker.join(timeout=2.0)
        agent_state.agent_states.clear()
        th._close_db()
        th._DB_PATH = orig_db_path  # type: ignore[attr-defined]
        shutil.rmtree(tmp, ignore_errors=True)


def test_backend_marks_alive_thread_as_running() -> None:
    """Backend ``_get_running_task_ids`` flags a task whose worker
    thread is alive, and ``_get_history`` surfaces ``is_running=True``
    on its row."""
    tmp = tempfile.mkdtemp(prefix="kiss-history-running-test-")
    orig_db_path = th._DB_PATH  # type: ignore[attr-defined]
    th._close_db()
    th._DB_PATH = Path(tmp) / "sorcar.db"  # type: ignore[attr-defined]
    stop = threading.Event()
    worker: threading.Thread | None = None
    fake_tab_id = "__alive_thread_test__"
    try:
        server = VSCodeServer()
        server.work_dir = tmp
        task_id, _ = th._add_task("alive thread task")

        state = agent_state.AgentState(
            str(task_id), tab_id=fake_tab_id, server_owned=True,
        )
        worker = threading.Thread(
            target=stop.wait, name="kiss-test-fake-worker", daemon=True,
        )
        worker.start()
        state.task_thread = worker
        agent_state.register(state)

        running = server._get_running_task_ids()
        assert str(task_id) in running, (
            f"backend must flag alive-thread row {task_id} as running; "
            f"got: {running}"
        )

        stop.set()
        worker.join(timeout=2.0)
        assert not worker.is_alive(), "test worker must have stopped"
        running_after = server._get_running_task_ids()
        assert str(task_id) not in running_after, (
            f"backend must drop row {task_id} once its thread dies; "
            f"got: {running_after}"
        )
    finally:
        if worker is not None:
            stop.set()
            worker.join(timeout=2.0)
        agent_state.agent_states.clear()
        th._close_db()
        th._DB_PATH = orig_db_path  # type: ignore[attr-defined]
        shutil.rmtree(tmp, ignore_errors=True)


def test_backend_overrides_failed_sentinel_for_running_task() -> None:
    """A row whose persisted result is ``"Agent Failed Abruptly"`` but
    whose worker thread is still alive must broadcast as
    ``is_running=True, failed=False``.

    This is the "crash-then-resume" path: the previous run left a
    failure sentinel in ``task_history.result``, but the agent has
    been reattached and is now actively running.  The History row
    must show the green pulsing dot, NOT the red failed dot.
    """
    event = _history_event_from_real_backend(
        result="Agent Failed Abruptly",
        fake_running_task_id="-1",
    )
    sessions = event["sessions"]
    row = next(
        s for s in sessions
        if s["preview"] == "running task from real backend"
    )
    assert row["is_running"] is True, (
        "alive-thread row must broadcast is_running=True even when the "
        "persisted result is a failed sentinel; got: "
        f"{row}"
    )
    assert row["failed"] is False, (
        "alive-thread row must NOT broadcast failed=True; got: "
        f"{row}"
    )
