# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a machine stream whose relay thread cannot start closes its socket and says why.

``MachineStream.__init__`` opens the socket to the machine's daemon and
then starts the thread relaying its frames.  Under thread exhaustion
``Thread.start()`` raised ``RuntimeError``: nothing closed the socket
(the websockets client keeps it alive with its own receive thread, and
the machine kept its per-connection state), and the error escaped
``RemoteViews.route`` (which answers only ``MachineError``), so the
page's ``terminalOpen`` got no reply.

Real local daemon, real remote daemon (a child process) registered as
the machine ``box``, a real client connection.  ``RLIMIT_NPROC`` is
lowered to 1 for exactly the relay thread's start: :class:`Pause` hooks
(instrumentation, nothing replaced) lower it when that thread is built
and restore it when the refusal is sent.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from websockets.protocol import State

from kiss.agents.sorcar import machines as mach
from kiss.server.machine_stream import MachineStream
from kiss.server.remote_views import RemoteViews
from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.pause import Pause, Pauses
from kiss.tests.server.two_client_harness import TwoClientHarness


@unittest.skipUnless(
    thread_start_can_be_starved(),
    "needs a host where RLIMIT_NPROC makes Thread.start fail (non-root Linux)",
)
class MachineStreamStartFailureTest(TwoClientHarness):
    """One local daemon with one page, one remote daemon as the machine ``box``."""

    def setUp(self) -> None:
        super().setUp()
        mach.save_machines([])
        self.remote_dir = tempfile.mkdtemp(prefix="stream_start_")
        self.remote = start_remote_daemon(Path(self.remote_dir) / "home", password="pw-remote")
        self.a = self.open_client("a")
        self.a.send({
            "type": "machineAction", "action": "add", "name": "box",
            "address": self.remote.url, "machinePassword": "pw-remote",
        })
        self.assertEqual(self.wait_event(self.a, "machines")["error"], "")

    def tearDown(self) -> None:
        super().tearDown()
        self.remote.stop()
        mach.save_machines([])
        shutil.rmtree(self.remote_dir, ignore_errors=True)

    @staticmethod
    def _close_sockets(streams: list[MachineStream]) -> None:
        """Close what a failed start left open, so a failing run leaks nothing."""
        for stream in streams:
            stream._ws.close()

    def test_terminal_open_is_refused_and_the_socket_closed(self) -> None:
        """The opener gets a ``terminalError``; the socket is closed; the next open works."""
        if sys.platform == "win32":
            self.skipTest("terminal tabs need a pty")
        streams: list[MachineStream] = []
        starved = contextlib.ExitStack()
        self.addCleanup(starved.close)
        self.addCleanup(self._close_sockets, streams)

        def starve_the_relay(frame_locals: dict[str, Any]) -> bool:
            if frame_locals.get("name") == "remote-views-box":
                streams.append(frame_locals["target"].__self__)
                starved.enter_context(nproc_limit_lowered_to_one())
            return False

        def restore_at_refusal(_frame_locals: dict[str, Any]) -> bool:
            starved.close()
            return False

        with Pauses(
            Pause(threading.Thread.__init__.__code__, starve_the_relay),
            Pause(RemoteViews._refuse.__code__, restore_at_refusal),
        ):
            self.a.send({
                "type": "terminalOpen", "tab_id": "term-x", "cols": 80, "rows": 24,
                "workDir": "", "machine": "box",
            })
            err = self.wait_event(self.a, "terminalError", timeout=20, tab_id="term-x")
        starved.close()
        self.assertIn("box: the stream could not start", err["text"])
        (stream,) = streams
        self.assertIs(stream._ws.protocol.state, State.CLOSED)
        self.assertEqual(self.server._server_api.remote_views._streams, {})

        self.a.send({
            "type": "terminalOpen", "tab_id": "term-y", "cols": 80, "rows": 24,
            "workDir": "", "machine": "box",
        })
        opened = self.wait_event(self.a, "terminalOpened", timeout=60, tab_id="term-y")
        self.assertEqual(opened["machine"], "box")
        self.a.send({"type": "terminalClose", "tab_id": "term-y"})
