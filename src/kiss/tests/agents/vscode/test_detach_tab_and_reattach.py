"""Integration tests: closing a chat tab does NOT stop a running agent
task, and resuming a still-running task from history reattaches the
live agent to the newly opened tab so events flow there dynamically.

Spec
----
1. When the frontend sends ``closeTab`` for a tab whose backend agent
   is still running a task, the backend must **not** kill the task —
   the agent must keep running to completion.

2. When the user clicks a still-running task in the task history panel
   and the chat tab for it is not open, the frontend allocates a new
   ``tabId`` and sends ``resumeSession``.  The backend must:

   a. Load and broadcast the persisted ``task_events`` for the chat
      (so the new tab shows the work-so-far history).

   b. Detect that an agent is still running for ``chat_id`` under some
      *other* tab id and re-key the ``_TabState`` from the old id to
      the new id.

   c. Tell the printer's per-tab event routing to forward every
      subsequent event emitted by the running agent to the new tab id
      (via ``printer.rebind_tab(old, new)`` which migrates per-tab
      state and installs an alias used by ``_inject_tab_id`` and
      ``_record_event``).

   d. Broadcast ``{"type": "status", "running": True, "tabId": new_id}``
      so the new tab shows the running spinner immediately.

3. Resuming a *finished* task (no live thread) must NOT rebind — it
   should just load history into the new tab.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
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

    real_broadcast = BaseBrowserPrinter.broadcast

    def capture(event: dict) -> None:
        # Run the real machinery (tab-id injection, recording) so
        # alias resolution is exercised in tests, then snapshot the
        # injected event.
        ev = server.printer._inject_tab_id(event)
        with server.printer._lock:
            server.printer._record_event(ev)
        with lock:
            events.append(ev)

    # Bind capture in place of the JSON-stdout-writing broadcast.
    server.printer.broadcast = capture  # type: ignore[assignment]
    # Keep a reference so we don't shadow unused-imports.
    _ = real_broadcast
    return server, events


def _start_fake_running_task(
    server: VSCodeServer, tab_id: str, chat_id: str,
) -> tuple[threading.Event, threading.Event, threading.Thread]:
    """Install a fake task thread on ``tab_id`` that emits one event,
    then blocks on ``release_event`` until the test releases it.

    The thread sets ``self.printer._thread_local.tab_id`` exactly like
    the real ``_run_task`` does, so subsequent ``broadcast()`` calls
    pick up the tab id from thread-local storage.

    Returns:
        ``(started_event, release_event, thread)``
    """
    tab = server._get_tab(tab_id)
    tab.agent._chat_id = chat_id
    tab.is_task_active = True
    tab.stop_event = threading.Event()

    started_event = threading.Event()
    release_event = threading.Event()
    pre_release_emitted = threading.Event()
    post_release_emitted = threading.Event()

    def fake_run() -> None:
        server.printer._thread_local.tab_id = tab_id
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
        chat_id = "chat-running-1"
        tab_id_a = "tab-A"
        started, release, thread = _start_fake_running_task(
            server, tab_id_a, chat_id,
        )
        assert started.is_set()
        assert thread.is_alive()

        # Simulate the frontend closing the tab while the task runs.
        server._handle_command({"type": "closeTab", "tabId": tab_id_a})

        # Backend MUST keep the _TabState so the task can finish.
        assert tab_id_a in server._tab_states
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

    def test_resume_running_task_rebinds_tab_state_and_routes_events(
        self,
    ) -> None:
        # Seed history: one task row + one persisted event.
        task_id, chat_id = th._add_task("running task")
        th._append_chat_event(
            {"type": "text_delta", "text": "hi-from-history"},
            task_id=task_id,
        )

        server, events = _make_server()
        tab_id_a = "tab-A"
        tab_id_b = "tab-B"

        started, release, thread = _start_fake_running_task(
            server, tab_id_a, chat_id,
        )

        # Close tab A; backend retains _TabState because task active.
        server._handle_command({"type": "closeTab", "tabId": tab_id_a})
        assert tab_id_a in server._tab_states

        # User clicks the running task in history → frontend allocates
        # a new tab id and sends resumeSession.
        server._handle_command({
            "type": "resumeSession",
            "chatId": chat_id,
            "taskId": task_id,
            "tabId": tab_id_b,
        })

        # _TabState moved from A to B.
        assert tab_id_b in server._tab_states
        assert tab_id_a not in server._tab_states
        assert server._tab_states[tab_id_b].task_thread is thread

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

        # Capture the position so we can isolate post-rebind events.
        post_replay_idx = len(events)

        # Release the fake task; the second broadcast must be routed
        # to tab_id_b (the new tab) even though the agent thread set
        # its thread-local tab_id to tab_id_a originally.
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

        post_events = events[post_replay_idx:]
        post_deltas = [
            e for e in post_events
            if e.get("type") == "text_delta" and e.get("text") == "post"
        ]
        assert len(post_deltas) == 1, (
            f"expected one 'post' delta after rebind, got {post_events}"
        )
        assert post_deltas[0]["tabId"] == tab_id_b, (
            "Live event from running agent thread must carry the new "
            f"tab id {tab_id_b}; got {post_deltas[0]}"
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


class TestPrinterRebindTab:
    """Unit tests for the alias map / rebind helper on the printer."""

    def test_rebind_aliases_old_to_new_tab_id(self) -> None:
        printer = BaseBrowserPrinter()
        printer._thread_local.tab_id = "old"
        printer.rebind_tab("old", "new")

        ev = printer._inject_tab_id({"type": "text_delta", "text": "x"})
        assert ev["tabId"] == "new", (
            "Events broadcast from a thread whose thread-local tab_id "
            "is the OLD id must resolve to the NEW id via the alias."
        )

    def test_rebind_migrates_per_tab_state(self) -> None:
        printer = BaseBrowserPrinter()
        printer._thread_local.tab_id = "old"
        printer.start_recording()
        with printer._lock:
            printer._record_event(
                {"type": "text_delta", "text": "a", "tabId": "old"},
            )
        with printer._bash_lock:
            printer._bash_state.buffer.append("buffered")
        printer.tokens_offset = 7
        printer.budget_offset = 1.25
        printer.steps_offset = 3

        printer.rebind_tab("old", "new")

        # Per-tab state moved under "new".
        assert "old" not in printer._recordings
        assert "new" in printer._recordings
        assert "old" not in printer._bash_states
        assert "new" in printer._bash_states
        assert printer._bash_states["new"].buffer == ["buffered"]
        assert printer._tokens_offsets.get("new") == 7
        assert printer._budget_offsets.get("new") == 1.25
        assert printer._steps_offsets.get("new") == 3
        assert "old" not in printer._tokens_offsets
        assert "old" not in printer._budget_offsets
        assert "old" not in printer._steps_offsets

    def test_alias_chain_resolves_to_final(self) -> None:
        printer = BaseBrowserPrinter()
        printer._thread_local.tab_id = "a"
        printer.rebind_tab("a", "b")
        printer.rebind_tab("b", "c")
        ev = printer._inject_tab_id({"type": "text_delta", "text": "x"})
        assert ev["tabId"] == "c"

    def test_recording_lookup_follows_alias(self) -> None:
        """An event tagged with the OLD tab id must still be appended
        to the recording keyed by the NEW tab id after a rebind."""
        printer = BaseBrowserPrinter()
        printer._thread_local.tab_id = "old"
        printer.start_recording()
        printer.rebind_tab("old", "new")

        with printer._lock:
            printer._record_event(
                {"type": "text_delta", "text": "after", "tabId": "old"},
            )

        # Switch to new tab id to read the recording.
        printer._thread_local.tab_id = "new"
        events = printer.peek_recording()
        types = [e.get("type") for e in events]
        assert "text_delta" in types


if __name__ == "__main__":
    import unittest
    unittest.main()
