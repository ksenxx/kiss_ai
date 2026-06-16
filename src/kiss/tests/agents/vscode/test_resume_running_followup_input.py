# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: loading a still-running task into a freshly-opened tab
from the history panel must accept follow-up user input WHILE the task
is running (the typed text must reach the live agent, not be silently
dropped), and the viewer tab must transition out of the running state
when the task finishes (so the pulsing green circle in the tab title
disappears and the next user message starts a NEW task instead of
being lost forever).

Reproduces and pins two bugs in the multi-viewer routing layer:

1. **Follow-up input dropped on viewer tab.**
   ``_cmd_append_user_message`` queues the prompt onto
   ``_RunningAgentState[tab_id].pending_user_messages``.  A viewer tab
   created by a history-resume click has a registry entry, but the
   live task runs in a different tab's state (the launcher tab) and
   the viewer's ``is_task_active`` is ``False``.  Without the routing
   fix, ``_cmd_append_user_message`` decides the viewer's tab has
   no live task and silently drops the prompt — the user types a
   follow-up while the task runs, sees their text disappear from the
   input box, but the agent never sees the message.

2. **No ``status running=false`` reaches the viewer tab when the task
   ends.**  ``_run_task``'s ``finally`` broadcasts ``status
   running=false`` stamped with the launcher's ``tabId``.  Because the
   event already carries a ``tabId`` the transport routes it verbatim
   without consulting the per-task subscriber map, so the viewer tab
   never observes the running→idle transition.  Symptoms: the pulsing
   green circle in the viewer tab's title pulses forever, the input
   box stays in "queue follow-up" mode, and the next user message is
   routed as an ``appendUserMessage`` against an already-finished
   task — getting dropped again.

The test drives the real ``VSCodeServer`` end-to-end through
``_handle_command``, the real ``_replay_session`` /
``_reattach_running_chat`` paths, and the real ``_TaskRunnerMixin``
worker thread.  Only the innermost LLM-driven ``run`` (the
grandparent of :class:`SorcarAgent`) is stubbed: it broadcasts a
sentinel ``text_delta`` so the test can observe live fan-out, then
blocks on a release event so we can perform follow-up actions WHILE
the task is provably still running.

The captured ``broadcast`` mirrors :class:`WebPrinter` fan-out
exactly: events with an explicit ``tabId`` pass verbatim and events
that go through the per-task subscriber map are duplicated once per
subscribed tab with the viewer's tab id stamped.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

import yaml

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.server import VSCodeServer
from kiss.core.models.model_info import get_available_models

_LIVE_TEXT = "live-delta-marker"


def _redirect_db(tmpdir: str) -> tuple:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _init_git_repo(tmpdir: str) -> None:
    subprocess.run(["git", "init", tmpdir], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmpdir,
                   capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir,
                   capture_output=True)
    Path(tmpdir, ".gitkeep").touch()
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir,
                   capture_output=True)


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]], threading.Lock]:
    """Create a ``VSCodeServer`` whose broadcasts mirror ``WebPrinter``."""
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    printer = server.printer

    def capture(event: dict[str, Any]) -> None:
        if "tabId" in event:
            with lock:
                events.append(event)
            return
        ev = printer._inject_task_id(event)
        if not ev.get("taskId"):
            with lock:
                events.append(ev)
            return
        with printer._lock:
            printer._record_event(ev)
        printer._persist_event(ev)
        for tab_id in printer._fanout_targets(ev.get("taskId")):
            with lock:
                events.append({**ev, "tabId": tab_id})

    printer.broadcast = capture  # type: ignore[assignment]
    return server, events, lock


def _patch_grandparent_run_blocking(
    release: threading.Event, started: threading.Event,
) -> Any:
    """Replace the LLM ``run`` with one that streams a delta then waits.

    Broadcasts a single sentinel ``text_delta`` so the test can
    observe live fan-out, signals *started* so the test can proceed
    knowing the worker thread is provably inside the run loop, then
    blocks on *release* so follow-up actions occur WHILE the task is
    still running.  Returns a minimal YAML result so the post-run
    bookkeeping completes cleanly.
    """
    parent = cast(Any, SorcarAgent.__mro__[1])
    original = parent.run

    def _run_proxy(self_agent: Any, **kwargs: Any) -> str:
        printer = kwargs.get("printer") or getattr(self_agent, "printer", None)
        if printer is not None:
            printer.broadcast({"type": "text_delta", "text": _LIVE_TEXT})
        # Drain any user-message queue that the test may have populated
        # while the task was running — mirrors the production drain
        # hook that ``SorcarAgent`` runs in its pre-step.
        drain = getattr(self_agent, "_drain_pending_user_messages", None)
        started.set()
        release.wait(timeout=30)
        # Re-drain after release so any prompt the test queued after
        # ``started.set()`` but before ``release.set()`` is observable
        # via the recorded ``add_message_to_conversation`` calls.
        if drain is not None:
            try:
                drain()
            except Exception:  # pragma: no cover — best-effort drain
                pass
        return str(yaml.dump({"success": True, "summary": "done"}))

    parent.run = _run_proxy
    return original


def _unpatch_grandparent_run(original: Any) -> None:
    cast(Any, SorcarAgent.__mro__[1]).run = original


def _wait_until(predicate: Any, timeout: float = 10.0) -> bool:
    """Poll *predicate* every 10 ms up to *timeout* seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestResumeRunningFollowupInput(unittest.TestCase):
    """End-to-end: history-resumed viewer tab accepts input during AND
    after a still-running task, and the running indicator clears when
    the task finishes."""

    def setUp(self) -> None:
        models = get_available_models()
        if not models:
            self.skipTest("no model API key configured")
        self.model = models[0]
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)
        self.server, self.events, self.lock = _make_server()
        self.release = threading.Event()
        self.started = threading.Event()
        self.original_run = _patch_grandparent_run_blocking(
            self.release, self.started,
        )

    def tearDown(self) -> None:
        # Always release so the worker thread can unwind cleanly even
        # when the test fails mid-flight.
        self.release.set()
        _unpatch_grandparent_run(self.original_run)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _start_blocking_task(self, tab_id: str, prompt: str) -> None:
        """Launch a task in *tab_id* and wait until it is provably
        inside the patched run loop."""
        self.server._handle_command({
            "type": "run", "prompt": prompt, "model": self.model,
            "workDir": self.tmpdir, "tabId": tab_id, "autoCommit": True,
        })
        assert self.started.wait(timeout=30), (
            "task worker thread never entered the patched run loop"
        )

    def _events(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.events)

    def test_append_user_message_from_viewer_tab_reaches_live_agent(
        self,
    ) -> None:
        """The user types a follow-up into the viewer tab's input box
        WHILE the task is running — the prompt must reach the launcher
        tab's live agent, not be silently dropped."""
        tab_launcher, tab_viewer = "tab-launcher", "tab-viewer"
        self._start_blocking_task(tab_launcher, "long running task")
        launcher_state = self.server._get_tab(tab_launcher)
        chat_id = launcher_state.chat_id
        assert chat_id

        # Viewer tab opens the chat from history while the task runs.
        self.server._handle_command(
            {"type": "newChat", "tabId": tab_viewer},
        )
        self.server._handle_command({
            "type": "resumeSession", "chatId": chat_id, "tabId": tab_viewer,
        })

        # Sanity check: the viewer received a ``status running=true``
        # stamped with ITS OWN tabId (the running-indicator on).
        run_true_viewer = [
            e for e in self._events()
            if e.get("type") == "status"
            and e.get("running") is True
            and e.get("tabId") == tab_viewer
        ]
        assert run_true_viewer, (
            "viewer tab should have received status running=True "
            "stamped with its own tabId after resumeSession"
        )

        # Type a follow-up into the viewer tab's input box and submit.
        # The frontend in the viewer tab observes ``isRunning=true``
        # (from the status event above) and therefore forwards the
        # text as an ``appendUserMessage`` carrying the viewer's tabId.
        self.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "viewer follow-up",
            "tabId": tab_viewer,
        })

        # The prompt must reach the LIVE agent — i.e. land in the
        # launcher tab's pending_user_messages — not be dropped just
        # because the viewer tab has no active task of its own.
        assert "viewer follow-up" in launcher_state.pending_user_messages, (
            "appendUserMessage from a history-resumed viewer tab was "
            "silently dropped instead of being routed to the live "
            f"agent on the launcher tab: launcher pending="
            f"{launcher_state.pending_user_messages!r}"
        )

        # The prompt echo must appear in the viewer's chat surface
        # (stamped with the viewer's tabId) so the user sees their
        # typed message in the viewer's transcript.
        prompt_echoes_viewer = [
            e for e in self._events()
            if e.get("type") == "prompt"
            and e.get("text") == "viewer follow-up"
            and e.get("tabId") == tab_viewer
        ]
        assert prompt_echoes_viewer, (
            "viewer tab missing prompt echo for follow-up message"
        )

    def test_viewer_tab_receives_running_false_when_task_ends(
        self,
    ) -> None:
        """When the live task finishes, the viewer tab must receive a
        ``status running=false`` stamped with the viewer's own tabId so
        the pulsing green indicator in its tab title turns off."""
        tab_launcher, tab_viewer = "tab-launcher", "tab-viewer"
        self._start_blocking_task(tab_launcher, "long running task")
        chat_id = self.server._get_tab(tab_launcher).chat_id

        self.server._handle_command(
            {"type": "newChat", "tabId": tab_viewer},
        )
        self.server._handle_command({
            "type": "resumeSession", "chatId": chat_id, "tabId": tab_viewer,
        })

        # Let the task complete.
        mark = len(self.events)
        self.release.set()
        task_thread = self.server._get_tab(tab_launcher).task_thread
        if task_thread is not None:
            task_thread.join(timeout=30)
        # The worker thread's outer finally clears ``task_thread`` to
        # None — poll for that as the canonical "task ended" signal.
        assert _wait_until(
            lambda: self.server._get_tab(tab_launcher).task_thread is None,
            timeout=30,
        ), "task worker thread never completed"

        after_end = self._events()[mark:]
        running_false_viewer = [
            e for e in after_end
            if e.get("type") == "status"
            and e.get("running") is False
            and e.get("tabId") == tab_viewer
        ]
        assert running_false_viewer, (
            "viewer tab missing status running=False at task end — "
            "the pulsing green indicator in the tab title would pulse "
            "forever and the input box would stay in 'queue follow-up' "
            f"mode.  Got events for viewer tab: "
            f"{[e for e in after_end if e.get('tabId') == tab_viewer]!r}"
        )

    def test_new_run_from_viewer_tab_after_task_ends(self) -> None:
        """After the task ends, a fresh ``run`` command from the viewer
        tab must start a new task (not be silently dropped).  This
        mirrors the user typing a new task into the viewer's input
        textbox after the running task finished."""
        tab_launcher, tab_viewer = "tab-launcher", "tab-viewer"
        self._start_blocking_task(tab_launcher, "first task")
        chat_id = self.server._get_tab(tab_launcher).chat_id

        self.server._handle_command(
            {"type": "newChat", "tabId": tab_viewer},
        )
        self.server._handle_command({
            "type": "resumeSession", "chatId": chat_id, "tabId": tab_viewer,
        })

        # End the first task.
        self.release.set()
        assert _wait_until(
            lambda: self.server._get_tab(tab_launcher).task_thread is None,
            timeout=30,
        )

        # Re-arm the patched run for the next task: it will signal
        # ``started`` and then return immediately (release stays set).
        self.started.clear()

        # Fresh task from the viewer tab.  The viewer's frontend now
        # observes ``isRunning=false`` and sends a ``run`` command.
        self.server._handle_command({
            "type": "run", "prompt": "second task from viewer",
            "model": self.model, "workDir": self.tmpdir,
            "tabId": tab_viewer, "autoCommit": True,
        })
        assert self.started.wait(timeout=30), (
            "second task from the viewer tab never reached the agent "
            "run loop — the user's typed text was lost"
        )

        viewer_state = self.server._get_tab(tab_viewer)
        assert _wait_until(
            lambda: viewer_state.task_thread is None,
            timeout=30,
        ), "second task did not complete"


if __name__ == "__main__":
    unittest.main()
