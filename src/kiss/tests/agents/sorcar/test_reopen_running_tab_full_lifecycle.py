# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Full-lifecycle integration test reproducing the close+reopen bug.

The bug under repair: when a Sorcar tab that STARTED a task is closed
while the task is running and then re-opened while the task is still
running, user input typed into the chat textbox is ignored — during
the task AND after it finishes (so the user cannot start a follow-up
task in the same tab).  A tab where a task started must behave
identically to a tab that loads the task: typed input must always
reach the daemon and trigger an injection (during running) or a new
``run`` (after finish).

Approach: drive the REAL :class:`VSCodeServer` with a stubbed
``_run_task_inner`` that blocks on a release event so the test holds
the lifecycle in each phase deterministically.  No mocks of routing /
state machinery.  Then verify EVERY layer the bug touches:

* ``resumeSession`` for the re-opened tab broadcasts
  ``status running:true`` and routes through ``_reattach_running_chat``.
* ``appendUserMessage`` during the task injects into
  ``pending_user_messages`` (does NOT drop with "no live task").
* A second ``run`` while busy is injected (matches existing fix).
* After ``_run_task`` finishes its ``finally`` block, the tab is
  re-runnable: a fresh ``run`` starts a new thread, clears state, and
  broadcasts a new ``clear`` + ``status running:true``.
* A second ``appendUserMessage`` during the second task is injected
  too — guards against state pollution across the close+reopen.
"""

from __future__ import annotations

import threading
import time
import unittest

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer


class TestReopenRunningTabFullLifecycle(unittest.TestCase):
    """End-to-end lifecycle: start → close+reopen → input → end → run again."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict] = []
        self._evt_lock = threading.Lock()

        def capture(ev: dict) -> None:
            with self._evt_lock:
                self.events.append(ev)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        # Replace ``_run_task_inner`` with a blocking stub: the test
        # uses ``release`` to deterministically end the task.
        self.started = threading.Event()
        self.release = threading.Event()
        self.runs: list[dict] = []

        def fake_inner(cmd: dict) -> None:
            tab_id = cmd.get("tabId", "")
            tab = _RunningAgentState.running_agent_states[tab_id]
            tab.is_task_active = True
            self.runs.append(cmd)
            self.started.set()
            # Block until the test signals task completion.
            self.release.wait(timeout=10)

        self.server._run_task_inner = fake_inner  # type: ignore[assignment]

    def tearDown(self) -> None:
        self.release.set()
        # Join any worker threads on lingering tabs.
        for tab in list(_RunningAgentState.running_agent_states.values()):
            th = tab.task_thread
            if th is not None:
                th.join(timeout=2)
        _RunningAgentState.running_agent_states.clear()

    def _events_of(self, ev_type: str, tab_id: str | None = None) -> list[dict]:
        with self._evt_lock:
            out = [e for e in self.events if e.get("type") == ev_type]
        if tab_id is None:
            return out
        return [e for e in out if e.get("tabId") == tab_id]

    def test_reopen_during_run_then_run_again_after_finish(self) -> None:
        tab_id = "tab-XYZ"

        # ---- Phase 1: user starts the task in tab_id ----
        self.server._handle_command({
            "type": "run",
            "prompt": "do a long task",
            "model": "test-model",
            "tabId": tab_id,
        })
        assert self.started.wait(timeout=3), "task did not start"
        tab = _RunningAgentState.running_agent_states[tab_id]
        chat_id = tab.chat_id
        assert chat_id, "chat_id must be allocated"

        # Initial clear was broadcast.
        assert self._events_of("clear", tab_id)

        # ---- Phase 2: user closes the tab while task is running ----
        # In the extension the webview is destroyed (onDidDispose) but
        # the daemon-side ``_RunningAgentState`` and task_thread stay
        # alive.  The extension does NOT send ``closeTab`` for a
        # webview-view close — only chat-tab closes do.  So the daemon
        # state is unchanged here; no command is sent.

        # ---- Phase 3: user reopens the view → resumeSession ----
        with self._evt_lock:
            self.events.clear()
        self.server._handle_command({
            "type": "resumeSession",
            "chatId": chat_id,
            "tabId": tab_id,
        })

        running_events = [
            e
            for e in self._events_of("status", tab_id)
            if e.get("running") is True
        ]
        assert running_events, (
            "BUG: resume of the started tab's OWN running chat must "
            "broadcast status running:true so the re-opened webview "
            f"re-learns the task state.  Got events: "
            f"{[e.get('type') for e in self.events]}"
        )

        # ---- Phase 4: user types DURING the running task ----
        with self._evt_lock:
            self.events.clear()
        self.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "also update the docs",
            "tabId": tab_id,
        })
        assert tab.pending_user_messages == ["also update the docs"], (
            "BUG: appendUserMessage during the task must be queued, "
            f"got pending={tab.pending_user_messages}"
        )
        # User text is echoed back so the chat surface shows it.
        prompt_echoes = [
            e
            for e in self._events_of("prompt", tab_id)
            if e.get("text") == "also update the docs"
        ]
        assert prompt_echoes, "queued prompt must be echoed to the webview"

        # Also: a ``run`` sent while busy (because the re-opened
        # webview can momentarily still think it is idle and send
        # ``submit`` → ``run``) is injected too, not dropped.
        with self._evt_lock:
            self.events.clear()
        self.server._handle_command({
            "type": "run",
            "prompt": "and add tests",
            "model": "test-model",
            "tabId": tab_id,
        })
        assert tab.pending_user_messages == ["also update the docs", "and add tests"], (
            "BUG: run-while-busy must inject the prompt rather than drop it"
        )

        # ---- Phase 5: task ends → release the worker ----
        self.release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if tab.task_thread is None:
                break
            time.sleep(0.05)
        assert tab.task_thread is None, (
            "BUG: task_thread must be cleared by _run_task's finally"
        )
        # Daemon broadcast status running:false in the finally block.
        status_off = [
            e for e in self._events_of("status", tab_id) if e.get("running") is False
        ]
        assert status_off, "task finish must broadcast status running:false"

        # ---- Phase 6: user types AFTER the task ended → submit/run ----
        # In the real webview, isRunning is now false (set by the
        # status:false event), so the webview sends ``submit`` which
        # the extension translates to a ``run`` command.  Verify the
        # daemon starts a NEW task — does NOT silently drop / reuse.
        # Re-arm the worker bookkeeping for the second run.
        self.started.clear()
        self.release.clear()
        prev_runs = len(self.runs)

        with self._evt_lock:
            self.events.clear()
        self.server._handle_command({
            "type": "run",
            "prompt": "now do a follow-up task",
            "model": "test-model",
            "tabId": tab_id,
        })
        assert self.started.wait(timeout=3), (
            "BUG: 'I cannot run any other task when the task ends' — a "
            "fresh run after the prior task finished must start a new "
            "task thread in the same (re-opened) tab.  No new task "
            "was started."
        )
        assert len(self.runs) == prev_runs + 1
        # A fresh clear was broadcast for the new task.
        assert self._events_of("clear", tab_id), (
            "new task must broadcast a fresh ``clear`` event"
        )

        # ---- Phase 7: input during the SECOND task must also be queued ----
        self.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "tweak the result",
            "tabId": tab_id,
        })
        assert "tweak the result" in tab.pending_user_messages, (
            "BUG: appendUserMessage during the second task must be queued; "
            f"got pending={tab.pending_user_messages}"
        )

        # Release the second task for clean teardown.
        self.release.set()


if __name__ == "__main__":
    unittest.main()
