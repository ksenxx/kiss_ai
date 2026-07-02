# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 2: ``_cmd_complete`` active-file snapshot clobber.

``_cmd_complete`` keeps a per-connection snapshot of the last-reported
active editor file (``_last_active_file``) and its content
(``_last_active_content``) so a ``complete`` command that carries no
``activeFile`` (e.g. focus is inside the webview) can still harvest
completions from the file the user was just editing.

Malformed (non-string) values are coerced before use — but the two
fields were coerced asymmetrically:

* junk ``activeFile``    -> coerced to ``""``; storage guard is
  ``if active_file:`` so the empty string is filtered and the previous
  file snapshot is correctly RETAINED;
* junk ``activeFileContent`` -> coerced to ``""``; storage guard is
  ``if active_content is not None:`` (deliberately, so a genuine empty
  file can be stored) — the coerced ``""`` passes the guard and
  OVERWRITES the previous content snapshot.

So a single malformed ``complete`` command (e.g. ``activeFileContent``
sent as a number/list by a buggy client) silently pairs the retained
file path with EMPTY content, and every subsequent ghost-text request
completes against an empty active file until the user re-focuses the
editor.  A junk value must be treated as "not supplied" (``None``),
exactly like a junk ``activeFile``.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with broadcast capture (no stdout)."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


class TestCompleteSnapshotClobber(unittest.TestCase):
    """Junk ``activeFileContent`` must not clobber the content snapshot."""

    def test_junk_content_retains_previous_snapshot(self) -> None:
        """A non-string ``activeFileContent`` is 'not supplied', not ''."""
        server, _events = _make_server()
        conn_id = "win-a"
        # 1. A well-formed complete command establishes the snapshot.
        server._handle_command({
            "type": "complete",
            "query": "",
            "activeFile": "pkg/mod.py",
            "activeFileContent": "unique_snapshot_token = 1\n",
            "connId": conn_id,
            "tabId": "t1",
        })
        self.assertEqual(
            server._last_active_file.get(conn_id), "pkg/mod.py",
        )
        self.assertEqual(
            server._last_active_content.get(conn_id),
            "unique_snapshot_token = 1\n",
        )
        # 2. A malformed command (numeric content, junk file) arrives —
        #    e.g. from a buggy or hostile client.  It must not corrupt
        #    either half of the snapshot.
        server._handle_command({
            "type": "complete",
            "query": "",
            "activeFile": 12345,
            "activeFileContent": 67890,
            "connId": conn_id,
            "tabId": "t1",
        })
        self.assertEqual(
            server._last_active_file.get(conn_id),
            "pkg/mod.py",
            "junk activeFile clobbered the file snapshot",
        )
        self.assertEqual(
            server._last_active_content.get(conn_id),
            "unique_snapshot_token = 1\n",
            "junk activeFileContent clobbered the content snapshot: "
            "ghost text now completes against an empty active file",
        )

    def test_genuine_empty_content_is_still_stored(self) -> None:
        """A real empty-string content (empty file) must still be stored."""
        server, _events = _make_server()
        conn_id = "win-b"
        server._handle_command({
            "type": "complete",
            "query": "",
            "activeFile": "pkg/full.py",
            "activeFileContent": "x = 1\n",
            "connId": conn_id,
            "tabId": "t1",
        })
        # The user switches to a genuinely empty file.
        server._handle_command({
            "type": "complete",
            "query": "",
            "activeFile": "pkg/empty.py",
            "activeFileContent": "",
            "connId": conn_id,
            "tabId": "t1",
        })
        self.assertEqual(server._last_active_file.get(conn_id), "pkg/empty.py")
        self.assertEqual(server._last_active_content.get(conn_id), "")


class TestRunStartWindowPromptLoss(unittest.TestCase):
    """A submit landing in the run start window must not lose its prompt.

    ``_cmd_run``'s busy branch routes a ``run`` that arrives while a
    task is in flight into the live agent as a follow-up user message
    ("The user's text must NOT be silently dropped").  But the guard
    requires ``tab.is_task_active`` — which is only set once the
    worker thread has begun executing ``_run_task``.  The winning
    submit assigns ``tab.task_thread`` under ``_state_lock``, then
    broadcasts ``clear`` (network I/O — a wide window), and only then
    calls ``thread.start()``.  A second submit landing in that window
    sees ``task_thread is not None`` (busy) and ``is_task_active ==
    False`` — and silently drops the user's prompt: it is neither
    queued into ``pending_user_messages`` nor echoed, and no new task
    is started.  A fast double-submit therefore loses the second
    message, contradicting the invariant stated in the code itself.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt2-race-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self.first_clear_entered = threading.Event()
        self.release = threading.Event()
        self._blocked_once = False
        self._events_lock = threading.Lock()

        def blocking_broadcast(event: dict[str, Any]) -> None:
            do_block = False
            with self._events_lock:
                self.events.append(event)
                if event.get("type") == "clear" and not self._blocked_once:
                    self._blocked_once = True
                    do_block = True
            if do_block:
                self.first_clear_entered.set()
                self.release.wait(timeout=30)

        self.server.printer.broadcast = blocking_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        self.release.set()
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_second_submit_in_start_window_is_queued(self) -> None:
        """The racing prompt must be queued as a follow-up, not dropped."""
        work_dir = str(Path(self.tmpdir) / "plain")
        Path(work_dir).mkdir()
        tab_id = "bughunt2-race-tab"
        first_cmd = {
            "type": "run",
            "prompt": "bughunt2 first task",
            "tabId": tab_id,
            "workDir": work_dir,
            "useWorktree": False,
            "autoCommit": False,
            # A model that cannot resolve: the real worker thread runs,
            # fails fast inside ``_run_task``'s guarded setup (no LLM
            # call is ever attempted), and its ``finally`` cleans up —
            # exactly the production error path.
            "model": "kiss-bughunt2-no-such-model",
        }
        t1 = threading.Thread(
            target=self.server._handle_command, args=(dict(first_cmd),),
            daemon=True,
        )
        t1.start()
        assert self.first_clear_entered.wait(timeout=30), (
            "first run never reached its clear broadcast"
        )
        # First submit is mid-``clear``-broadcast: ``task_thread`` is
        # assigned but NOT yet started and ``is_task_active`` is still
        # False — the exact start window.
        second_cmd = dict(first_cmd)
        second_cmd["prompt"] = "bughunt2 follow-up during start window"
        self.server._handle_command(second_cmd)

        with self.server._state_lock:
            tab = _RunningAgentState.running_agent_states[tab_id]
            queued = list(tab.pending_user_messages)
        with self._events_lock:
            echoes = [
                e for e in self.events
                if e.get("type") == "prompt"
                and e.get("text") == "bughunt2 follow-up during start window"
            ]
        self.release.set()
        t1.join(timeout=30)
        # Let the real worker finish and clean up its state.
        deadline = time.time() + 30
        while time.time() < deadline:
            state = _RunningAgentState.running_agent_states.get(tab_id)
            if state is not None and state.task_thread is None:
                break
            time.sleep(0.02)

        with self._events_lock:
            clears = [e for e in self.events if e.get("type") == "clear"]
        assert len(clears) == 1, (
            f"{len(clears)} clear events — a second concurrent submit "
            "passed the busy guard and started a second task"
        )
        assert queued == ["bughunt2 follow-up during start window"], (
            "BUG: a submit landing in the run start window "
            "(task_thread assigned, thread not yet started, "
            "is_task_active still False) silently dropped the user's "
            f"prompt — pending_user_messages={queued!r}"
        )
        assert echoes, (
            "BUG: the queued follow-up was never echoed back as a "
            "'prompt' event, so the user's message vanished from the UI"
        )


if __name__ == "__main__":
    unittest.main()
