# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: server-shutdown cancellation must not masquerade as a user stop.

Production failure (task 3025 in the bundled ``sorcar.db``): a task was
running in its worker thread (step 3 of 100) when ``kiss-web`` received a
routine restart ``SIGTERM`` — a daemon / LaunchAgent restart triggered by
an in-progress KISS Sorcar extension update.  The graceful-shutdown path
(:meth:`RemoteAccessServer._stop_active_agent_tasks`) injects a
``KeyboardInterrupt`` into the worker exactly like the user clicking
"Stop".  Both paths were therefore indistinguishable and the row was
persisted as ``"Task stopped by user"`` (event ``task_stopped``) — so the
post-mortem analysis wrongly told the user that *they* stopped the task.

The fix marks the tab with
:attr:`_RunningAgentState.interrupted_by_shutdown` before the shutdown
helper injects the interrupt; the task-runner's ``except KeyboardInterrupt``
handler then persists ``"Task interrupted by server restart/shutdown"``
(event ``task_interrupted``) for the shutdown path while leaving the
genuine "Stop" button path at ``"Task stopped by user"`` (event
``task_stopped``).

Both tests stand up a *real* worker thread running :meth:`_run_task` with
a stub agent blocked in a non-cooperative ``sleep`` loop (mimicking an
agent wedged in an LLM API call), then drive the two cancellation paths
and assert the persisted ``task_history.result`` AND the persisted
lifecycle event type.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
from typing import Any, cast
from unittest import TestCase

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_remote_server() -> Any:
    """Build a :class:`RemoteAccessServer` with no tunnel / sockets bound.

    Constructing the remote server also constructs its owned
    :class:`VSCodeServer`, whose ``__init__`` *clears* the process-global
    ``_RunningAgentState.running_agent_states`` registry.  Tests must
    therefore build the remote server FIRST and then run their task
    through ``remote._vscode_server`` so the worker registers into the
    same registry the shutdown helper later scans — exactly as in
    production where a single server owns both.
    """
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.agents.vscode.web_server import RemoteAccessServer

    tmp = tempfile.mkdtemp(prefix="kiss-interrupt-test-")
    return RemoteAccessServer(
        use_tunnel=False,
        url_file=os.path.join(tmp, "url.json"),
        uds_path=os.path.join(tmp, "sorcar.sock"),
    )


def _start_blocked_worker(vscode: Any, tab_id: str) -> tuple[Any, str]:
    """Start a worker whose stub agent blocks in a non-cooperative sleep.

    Mirrors a real agent wedged inside an LLM API call: it registers a
    ``task_history`` row, signals readiness, then sleeps without polling
    the cooperative stop event, so only an injected ``KeyboardInterrupt``
    can break it.

    Args:
        vscode: The :class:`VSCodeServer` that owns the worker.
        tab_id: The tab id to run under.

    Returns:
        ``(worker_thread, task_id)`` once the worker is inside the agent
        run and ``is_task_active`` has been set.
    """
    tab = vscode._get_tab(tab_id)
    agent = WorktreeSorcarAgent("Sorcar VS Code")
    tab.agent = agent
    tab.chat_id = ""

    captured: dict[str, str] = {}
    entered = threading.Event()

    def fake_run(**kwargs: Any) -> None:
        agent.total_tokens_used = 123
        agent.budget_used = 0.01
        agent.step_count = 3
        agent._chat_id = agent._chat_id or f"interrupt-chat-{tab_id}"
        task_id, _ = _persistence._add_task(
            kwargs.get("prompt_template", ""),
            chat_id=agent._chat_id,
            extra={
                "model": kwargs.get("model_name", ""),
                "work_dir": kwargs.get("work_dir", ""),
                "version": "test",
                "is_parallel": False,
                "is_worktree": False,
            },
        )
        agent._last_task_id = task_id
        captured["id"] = task_id
        entered.set()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            time.sleep(0.05)

    agent.run = fake_run  # type: ignore[assignment]

    tab.stop_event = threading.Event()
    tab.user_answer_queue = queue.Queue()

    worker = threading.Thread(
        target=vscode._run_task,
        args=({
            "type": "run",
            "prompt": f"cancel-while-running-{tab_id}",
            "tabId": tab_id,
            "workDir": "/tmp",
            "useParallel": False,
            "useWorktree": False,
            "autoCommit": False,
        },),
        daemon=True,
    )
    tab.task_thread = worker
    worker.start()

    assert entered.wait(timeout=10), "worker never entered agent.run"
    for _ in range(100):
        if tab.is_task_active:
            break
        time.sleep(0.02)
    assert tab.is_task_active, "is_task_active was never set on the tab"
    return worker, captured["id"]


def _persisted_result_and_event(task_id: str) -> tuple[str, list[str]]:
    """Return ``(result, event_types)`` persisted for *task_id*.

    Flushes any buffered chat events first so the lifecycle end event is
    queryable.

    Args:
        task_id: The ``task_history`` primary key to read.

    Returns:
        The persisted ``result`` string and the ordered list of event
        ``type`` strings recorded for the task.
    """
    _persistence._flush_chat_events()
    db = _persistence._get_db()
    row = db.execute(
        "SELECT result FROM task_history WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row is not None
    loaded = _persistence._load_chat_events_by_task_id(task_id)
    events = (
        cast(list[dict[str, Any]], loaded["events"]) if loaded else []
    )
    types = [str(e.get("type")) for e in events]
    return row["result"], types


class TestTaskInterruptedVsStopped(TestCase):
    """SIGTERM-shutdown cancellation and a user "Stop" must be distinct."""

    def test_shutdown_persists_task_interrupted_label(self) -> None:
        """The shutdown path persists the restart/shutdown-specific outcome.

        ``_stop_active_agent_tasks`` (reached only on ``SIGTERM`` / daemon
        restart) must persist ``"Task interrupted by server
        restart/shutdown"`` and emit a ``task_interrupted`` lifecycle
        event — NOT the user-stop label.
        """
        tab_id = "interrupt-shutdown-1"
        remote = _make_remote_server()
        vscode = remote._vscode_server
        worker, task_id = _start_blocked_worker(vscode, tab_id)

        remote._stop_active_agent_tasks(timeout=12.0)
        worker.join(timeout=5)
        assert not worker.is_alive(), "shutdown left the worker running"

        result, types = _persisted_result_and_event(task_id)
        assert result == "Task interrupted by server restart/shutdown", (
            "shutdown cancellation must not be mislabelled as a user stop; "
            f"got result {result!r}"
        )
        assert "task_interrupted" in types, (
            f"expected a task_interrupted event; got {types!r}"
        )
        assert "task_stopped" not in types, (
            f"shutdown must not emit a task_stopped event; got {types!r}"
        )

    def test_manual_stop_persists_task_stopped_label(self) -> None:
        """The user "Stop" button keeps the original stopped outcome.

        ``_stop_task`` must persist ``"Task stopped by user"`` and emit a
        ``task_stopped`` event — unchanged by the shutdown-labelling fix.
        """
        tab_id = "interrupt-userstop-1"
        remote = _make_remote_server()
        vscode = remote._vscode_server
        worker, task_id = _start_blocked_worker(vscode, tab_id)

        vscode._stop_task(tab_id)
        worker.join(timeout=10)
        assert not worker.is_alive(), "manual stop left the worker running"

        result, types = _persisted_result_and_event(task_id)
        assert result == "Task stopped by user", (
            "a user Stop click must persist the user-stop label; "
            f"got result {result!r}"
        )
        assert "task_stopped" in types, (
            f"expected a task_stopped event; got {types!r}"
        )
        assert "task_interrupted" not in types, (
            f"a user stop must not emit task_interrupted; got {types!r}"
        )
