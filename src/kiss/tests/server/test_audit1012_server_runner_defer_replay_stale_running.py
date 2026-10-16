# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-12 (server-runner): deferred replay racing the task's end.

``AgentServer._defer_replay`` (the ``ready``-time attach of a
reconnecting client to a tab whose chat runs a live task) attaches the
viewer tab under ``_state_lock``, releases the lock, and broadcasts an
optimistic ``status running=true`` to EVERY client (the tab is a
registry tab mirrored on every surface).  The task can end inside that
window: its end-of-run fan-out already sent ``running=false`` for the
viewer tab, so the optimistic ``running=true`` lands AFTER it.  The
only correction ``_defer_replay`` sent was the ``idle`` ``running=false``
scoped with ``connId`` to the one reconnecting client — every other
surface showing the tab kept its spinner forever.

The window is made deterministic with ``sys.settrace`` on the deferring
thread: it parks on the call into ``_broadcast_viewer_running`` (after
the attach, before the optimistic broadcast) while a real run,
submitted through the real ``_cmd_run``, ends in its SEA getter.
Everything else is real: a real :class:`AgentServer`, a real worker
thread, the real end-of-run broadcasts, the real
:class:`~kiss.tests.server._memory_printer.MemoryPrinter` capture.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any
from unittest import TestCase

from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server._memory_printer import MemoryPrinter

_BLOCKING_SCRIPT = textwrap.dedent(
    """
    import pathlib
    import time

    from kiss.agents.seas.base.base_sea import BaseSea

    _DIR = pathlib.Path(__file__).resolve().parent


    class Sea(BaseSea):
        def settings(self, settings):
            \"\"\"Block until released, then raise (the task ends in setup).\"\"\"
            (_DIR / "entered").write_text("1", encoding="utf-8")
            deadline = time.time() + 60
            while time.time() < deadline:
                if (_DIR / "release").exists():
                    raise RuntimeError("released")
                time.sleep(0.02)
            raise RuntimeError("timed out waiting for the release")
    """
)

_RECONNECTING_CONN = "conn-reconnecting"
_OTHER_CONN = "conn-other-surface"


class TestDeferReplayStaleRunning(TestCase):
    """A deferred attach racing the task's end must not strand other surfaces."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-audit1012-defer-"))
        # Cleanups run after tearDown, which releases and joins the worker
        # first, so the directory is removed only once nothing uses it.
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        self.script = self.tmp / "agent.py"
        self.script.write_text(_BLOCKING_SCRIPT, encoding="utf-8")
        self.printer = MemoryPrinter()
        self.server = AgentServer(printer=self.printer)
        self.server.work_dir = str(self.work_dir)
        self.parked = threading.Event()
        self.release_park = threading.Event()

    def tearDown(self) -> None:
        (self.tmp / "release").write_text("1", encoding="utf-8")
        self.release_park.set()
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and any(
            s.task_thread is not None and s.task_thread.is_alive()
            for s in agent_state.agent_states.values()
        ):
            time.sleep(0.05)
        agent_state.agent_states.clear()

    def _statuses_seen_by(self, tab_id: str, conn_id: str) -> list[bool]:
        """The ``status`` booleans connection *conn_id* receives for *tab_id*."""
        return [
            bool(ev.get("running"))
            for ev in list(self.printer.emitted)
            if ev.get("type") == "status"
            and ev.get("tabId") == tab_id
            and ev.get("connId", conn_id) == conn_id
        ]

    def _wait(self, predicate: Any, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def _start_parked_run(self, tab_id: str, chat_id: str) -> None:
        self.server._cmd_run(
            {
                "type": "run",
                "prompt": "deferred attach race",
                "tabId": tab_id,
                "taskId": f"tok-{tab_id}",
                "chatId": chat_id,
                "workDir": str(self.work_dir),
                "useWorktree": False,
                "isParallel": False,
                "autoCommit": False,
                "seaPath": str(self.script),
            }
        )
        self.assertTrue(
            self._wait((self.tmp / "entered").exists, 30.0),
            "the run never reached the SEA getter",
        )

    def _defer_parked_before_running_broadcast(
        self,
        chat_id: str,
        tab_id: str,
    ) -> None:
        """Run ``_defer_replay``, parking on the call into the optimistic broadcast."""

        def hook(frame: FrameType, event: str, arg: Any) -> None:
            if frame.f_code.co_name == "_broadcast_viewer_running" and not self.parked.is_set():
                self.parked.set()
                if not self.release_park.wait(timeout=30):
                    raise TimeoutError("parked too long before the running broadcast")

        sys.settrace(hook)
        try:
            self.server._defer_replay(chat_id, tab_id, None, _RECONNECTING_CONN)
        finally:
            sys.settrace(None)

    def test_task_ending_inside_deferred_attach_window(self) -> None:
        launcher, viewer, chat_id = "defer-launcher", "defer-viewer", "chat-defer"
        self._start_parked_run(launcher, chat_id)

        deferrer = threading.Thread(
            target=self._defer_parked_before_running_broadcast,
            args=(chat_id, viewer),
            daemon=True,
        )
        deferrer.start()
        self.assertTrue(
            self.parked.wait(timeout=30),
            "the deferred replay never attached the viewer",
        )

        # The task ends INSIDE the window: the viewer is subscribed, so
        # the end-of-run fan-out sends it ``running=false`` — before
        # the parked thread's optimistic ``running=true``.
        (self.tmp / "release").write_text("1", encoding="utf-8")
        self.assertTrue(
            self._wait(
                lambda: False in self._statuses_seen_by(viewer, _OTHER_CONN),
                30.0,
            ),
            "the end-of-run fan-out never reached the attached viewer",
        )

        self.release_park.set()
        deferrer.join(timeout=30)
        self.assertFalse(deferrer.is_alive())

        self.assertIs(
            self._statuses_seen_by(viewer, _RECONNECTING_CONN)[-1],
            False,
            "the reconnecting client was left running: "
            f"{self._statuses_seen_by(viewer, _RECONNECTING_CONN)}",
        )
        self.assertIs(
            self._statuses_seen_by(viewer, _OTHER_CONN)[-1],
            False,
            "BUG: a surface other than the reconnecting one was left with "
            "running=true forever — the optimistic running=true was "
            "broadcast to every client after the task ended, but the "
            "correction was scoped to the reconnecting connection: "
            f"{self._statuses_seen_by(viewer, _OTHER_CONN)}",
        )

    def test_idle_chat_answers_only_the_reconnecting_client(self) -> None:
        """No live task: the ``running=false`` answer stays connection-scoped.

        Nothing optimistic was broadcast, so the other surfaces (which
        already know the tab is idle) must not be re-told.
        """
        viewer, chat_id = "idle-viewer", "chat-idle"
        self.server._defer_replay(chat_id, viewer, None, _RECONNECTING_CONN)
        self.assertEqual(self._statuses_seen_by(viewer, _RECONNECTING_CONN), [False])
        self.assertEqual(self._statuses_seen_by(viewer, _OTHER_CONN), [])
