# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 2: ``_cmd_complete`` active-file snapshot clobber.

``_cmd_complete`` keeps a per-connection snapshot of the last-reported
active editor file (``_last_active_file``) and its content
(``_last_active_content``) so a ``complete`` command that carries no
``activeFile`` (e.g. focus is inside the chat page) can still harvest
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

from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import make_capturing_server
from kiss.tests.server.pause import Pause


class TestCompleteSnapshotClobber(unittest.TestCase):
    """Junk ``activeFileContent`` must not clobber the content snapshot."""

    def test_junk_content_retains_previous_snapshot(self) -> None:
        """A non-string ``activeFileContent`` is 'not supplied', not ''."""
        server, _events = make_capturing_server()
        conn_id = "win-a"
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
        server, _events = make_capturing_server()
        conn_id = "win-b"
        server._handle_command({
            "type": "complete",
            "query": "",
            "activeFile": "pkg/full.py",
            "activeFileContent": "x = 1\n",
            "connId": conn_id,
            "tabId": "t1",
        })
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
    publishes the tab (a file write and a broadcast — a wide window),
    and only then announces the run and calls ``thread.start()``.  A
    second submit landing in that window
    sees ``task_thread is not None`` (busy) and ``is_task_active ==
    False`` — and silently drops the user's prompt: it is neither
    queued into ``pending_user_messages`` nor echoed, and no new task
    is started.  A fast double-submit therefore loses the second
    message, contradicting the invariant stated in the code itself.
    Echoed before the run's ``clear``, the message is wiped off the
    page again: its echo follows the ``clear``.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt2-race-")
        self.server, self.events = make_capturing_server()

    def tearDown(self) -> None:
        agent_state.agent_states.clear()
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
            "model": "kiss-bughunt2-no-such-model",
        }
        t1 = threading.Thread(
            target=self.server._handle_command, args=(dict(first_cmd),),
            daemon=True,
        )
        with Pause(AgentServer._publish_tabs_state.__code__, bool) as gate:
            t1.start()
            assert gate.paused.wait(timeout=30), "first run never published its tab"
            second_cmd = dict(first_cmd)
            second_cmd["prompt"] = "bughunt2 follow-up during start window"
            self.server._handle_command(second_cmd)

            with self.server._state_lock:
                tab_state = agent_state.find_by_tab(tab_id)
                assert tab_state is not None
                queued = list(tab_state.pending_user_messages)
            assert not [e for e in list(self.events) if e.get("type") == "clear"], (
                "a second concurrent submit passed the busy guard and "
                "started a second task"
            )
        t1.join(timeout=30)
        assert queued == ["bughunt2 follow-up during start window"], (
            "BUG: a submit landing in the run start window "
            "(task_thread assigned, thread not yet started, "
            "is_task_active still False) silently dropped the user's "
            f"prompt — pending_user_messages={queued!r}"
        )
        frames = [
            "clear" if e.get("type") == "clear" else "echo"
            for e in list(self.events)
            if e.get("type") == "clear" or e.get("steer")
        ]
        assert frames[:2] == ["clear", "echo"] and frames.count("echo") == 1, (
            "the follow-up must be echoed once, after the run's clear "
            f"(which empties the tab): {frames}"
        )
        # The first run exits at its "No model available" check without
        # ever draining the queued prompt, so the runner re-submits it
        # as the tab's next run once the first worker has torn down
        # (a message typed during teardown is never dropped).  Wait for
        # that second run to finish.
        deadline = time.time() + 30
        state = None
        while time.time() < deadline:
            state = agent_state.find_by_tab(tab_id)
            if (
                state is not None
                and state.task_thread is None
                and state.last_user_prompt
                == "bughunt2 follow-up during start window"
            ):
                break
            time.sleep(0.02)
        assert state is not None and state.task_thread is None, (
            "the queued prompt was never re-run after the first run ended"
        )


if __name__ == "__main__":
    unittest.main()
