# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: closing a chat tab does NOT stop a running agent
task, and resuming a still-running task from history subscribes the
newly opened tab to the live agent's stream so events flow there
dynamically — without stealing the stream from the original client.

Spec
----
1. When the frontend sends ``closeTab`` for a tab whose backend agent
   is still running a task, the backend must **not** kill the task —
   the agent must keep running to completion.

2. When the user clicks a still-running task in the task history panel
   and the chat tab for it is not open (or is open in a different
   web client), the frontend allocates a new ``tabId`` and sends
   ``resumeSession``.  The backend must:

   a. Load and broadcast the persisted ``task_events`` for the chat
      (so the new tab shows the work-so-far history).

   b. Detect that an agent is still running for ``chat_id`` under
      some *other* tab id and subscribe the new tab id to that
      agent's event stream via
      :meth:`JsonPrinter.subscribe_tab` so every subsequent
      broadcast is duplicated with ``tabId=new_id``.  The source
      ``_RunningAgentState`` is NOT moved — both the original tab id and the
      new tab id receive the stream, supporting multiple concurrent
      viewers of the same running task.

   c. Broadcast ``{"type": "status", "running": True, "tabId": new_id}``
      so the new tab shows the running spinner immediately.

3. Resuming a *finished* task (no live thread) must NOT subscribe —
   it should just load history into the new tab.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.json_printer import JsonPrinter
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


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer whose broadcasts go into an in-memory list."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    real_broadcast = JsonPrinter.broadcast

    def capture(event: dict) -> None:
        # Mirror :meth:`WebPrinter.broadcast` exactly:
        #   * Events with explicit ``tabId`` go through verbatim.
        #   * Other events are tagged with ``taskId`` and fanned out
        #     to every subscriber tab, with each copy stamped with
        #     its own ``tabId``.
        if "tabId" in event:
            with lock:
                events.append(event)
            return
        ev = server.printer._inject_task_id(event)
        with server.printer._lock:
            server.printer._record_event(ev)
        for tab_id in server.printer._fanout_targets(ev.get("taskId")):
            with lock:
                events.append({**ev, "tabId": tab_id})

    # Bind capture in place of the JSON-stdout-writing broadcast.
    server.printer.broadcast = capture  # type: ignore[assignment]
    # Keep a reference so we don't shadow unused-imports.
    _ = real_broadcast
    return server, events


def _start_fake_running_task(
    server: VSCodeServer, tab_id: str, chat_id: str,
    task_id: int | None = None,
) -> tuple[threading.Event, threading.Event, threading.Thread]:
    """Install a fake task thread on ``tab_id`` that emits one event,
    then blocks on ``release_event`` until the test releases it.

    The thread sets ``self.printer._thread_local.task_id`` to the
    string form of the task_id exactly like the real ``_run_task``
    does, so subsequent ``broadcast()`` calls pick up the task id
    from thread-local storage and fan out to every subscribed tab.

    Returns:
        ``(started_event, release_event, thread)``
    """
    tab = server._get_tab(tab_id)
    tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
    tab.agent._chat_id = chat_id
    if task_id is not None:
        tab.agent._last_task_id = task_id
        tab.task_history_id = task_id
    tab.chat_id = chat_id
    tab.is_task_active = True
    tab.stop_event = threading.Event()

    started_event = threading.Event()
    release_event = threading.Event()
    pre_release_emitted = threading.Event()
    post_release_emitted = threading.Event()

    task_key = str(task_id) if task_id is not None else tab_id
    # Subscribe the source tab to the running task so its own client
    # receives the broadcasts (the agent thread does this inside
    # ``ChatSorcarAgent.run`` in production; the fake task skips that
    # path so we wire it up here).
    server.printer.subscribe_tab(task_key, tab_id)

    def fake_run() -> None:
        server.printer._thread_local.task_id = task_key
        server.printer.broadcast({"type": "text_delta", "text": "pre"})
        pre_release_emitted.set()
        started_event.set()
        release_event.wait(timeout=10)
        server.printer.broadcast({"type": "text_delta", "text": "post"})
        post_release_emitted.set()
        with server._state_lock:
            tab.is_task_active = False
            tab.task_thread = None

    thread = threading.Thread(target=fake_run, daemon=True)
    tab.task_thread = thread
    thread.start()
    started_event.wait(timeout=5)
    # Stash markers on the server so tests can inspect them.
    setattr(server, "_pre_emitted", pre_release_emitted)
    setattr(server, "_post_emitted", post_release_emitted)
    return started_event, release_event, thread


class TestCloseTabDoesNotStopRunningTask:
    """``closeTab`` must NOT kill a running agent task."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_close_tab_keeps_thread_alive_and_finishes_task(self) -> None:
        server, _events = _make_server()
        # ``tab_id`` and ``chat_id`` are orthogonal — the tab's
        # routing key need not match the chat persistence key.
        chat_id = "chat-running-1"
        tab_id_a = "tab-A"
        started, release, thread = _start_fake_running_task(
            server, tab_id_a, chat_id,
        )
        assert started.is_set()
        assert thread.is_alive()

        # Simulate the frontend closing the tab while the task runs.
        server._handle_command({"type": "closeTab", "tabId": tab_id_a})

        # Backend MUST keep the _RunningAgentState so the task can finish.
        assert tab_id_a in server._running_agent_states
        assert thread.is_alive(), (
            "Agent thread must continue to run after closeTab"
        )

        # Release the task; it must run to its natural completion.
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive(), (
            "Task thread must complete on its own; closeTab must not "
            "have signalled stop."
        )
        post_emitted: threading.Event = getattr(server, "_post_emitted")
        assert post_emitted.is_set(), (
            "Post-release broadcast was not emitted — the task did "
            "not finish."
        )


class TestResumeRunningTaskReattachesLiveEvents:
    """Resuming a still-running task in a new tab rebinds the live
    agent so its subsequent events show up in the new tab."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resume_running_task_subscribes_and_fans_out_events(
        self,
    ) -> None:
        # Seed history: one task row + one persisted event.
        task_id, chat_id = th._add_task("running task")
        th._append_chat_event(
            {"type": "text_delta", "text": "hi-from-history"},
            task_id=task_id,
        )

        server, events = _make_server()
        # ``tab_id`` and ``chat_id`` are orthogonal — the source tab
        # carries its own routing key, and the viewer that joins from
        # history allocates a fresh tab id; the chat lookup is routed
        # by the persisted chat id.
        tab_id_a = "tab-A"
        tab_id_b = "tab-B"

        started, release, thread = _start_fake_running_task(
            server, tab_id_a, chat_id, task_id=task_id,
        )

        # Close tab A; backend retains _RunningAgentState because task active.
        server._handle_command({"type": "closeTab", "tabId": tab_id_a})
        assert tab_id_a in server._running_agent_states

        # User clicks the running task in history → frontend allocates
        # a new tab id and sends resumeSession.
        server._handle_command({
            "type": "resumeSession",
            "chatId": chat_id,
            "taskId": task_id,
            "tabId": tab_id_b,
        })

        # Both tab states exist: the source (still owns the running
        # thread) AND the new viewer tab.  Multi-viewer fan-out keeps
        # both alive instead of moving the stream.
        assert tab_id_a in server._running_agent_states
        assert tab_id_b in server._running_agent_states
        assert server._running_agent_states[tab_id_a].task_thread is thread
        # The new viewer tab does not own a task thread.
        assert server._running_agent_states[tab_id_b].task_thread is None

        # Replay broadcast went to the new tab id.
        replays = [e for e in events if e.get("type") == "task_events"]
        assert len(replays) == 1
        assert replays[0]["tabId"] == tab_id_b
        assert replays[0]["chat_id"] == chat_id

        # Running-status broadcast was sent for the new tab so the
        # spinner shows up immediately.
        running_statuses = [
            e for e in events
            if e.get("type") == "status"
            and e.get("running") is True
            and e.get("tabId") == tab_id_b
        ]
        assert len(running_statuses) >= 1, (
            f"expected status running=True for {tab_id_b}, got {events}"
        )

        # Capture the position so we can isolate post-subscribe events.
        post_replay_idx = len(events)

        # Release the fake task; the agent thread emits one 'post'
        # event with thread-local tab id == tab_id_a.  Because tab_id_b
        # is subscribed, the printer must fan out an additional copy
        # tagged with tab_id_b.
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

        post_events = events[post_replay_idx:]
        post_deltas = [
            e for e in post_events
            if e.get("type") == "text_delta" and e.get("text") == "post"
        ]
        post_delta_tab_ids = sorted(
            {str(e.get("tabId") or "") for e in post_deltas},
        )
        assert tab_id_a in post_delta_tab_ids, (
            "Source-tagged 'post' delta missing — original client "
            f"would not see it.  Got: {post_deltas}"
        )
        assert tab_id_b in post_delta_tab_ids, (
            "Fan-out 'post' delta tagged with the new viewer tab id "
            f"missing — multi-viewer broken.  Got: {post_deltas}"
        )

    def test_resume_finished_task_does_not_rebind(self) -> None:
        """Replaying a completed task only loads history; no rebind."""
        task_id, chat_id = th._add_task("done task")
        th._append_chat_event(
            {"type": "text_delta", "text": "done"}, task_id=task_id,
        )
        server, events = _make_server()

        tab_id_b = "tab-B"
        server._handle_command({
            "type": "resumeSession",
            "chatId": chat_id,
            "taskId": task_id,
            "tabId": tab_id_b,
        })

        # No status:running=True broadcast for finished tasks.
        running = [
            e for e in events
            if e.get("type") == "status" and e.get("running") is True
        ]
        assert running == []

        # History is broadcast.
        replays = [e for e in events if e.get("type") == "task_events"]
        assert len(replays) == 1
        assert replays[0]["tabId"] == tab_id_b



if __name__ == "__main__":
    import unittest
    unittest.main()
