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
import logging
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
from kiss.tests.server.remote_tabs_probe import machine_of
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

    def test_a_bind_whose_warm_up_cannot_start_is_refused_and_keeps_no_bookkeeping(self) -> None:
        """The view stream's warm-up thread is starved: the run is refused, the next bind works."""
        tabs = self.server._server_api.remote_tabs
        starved = contextlib.ExitStack()
        self.addCleanup(starved.close)

        def starve_the_warm_up(frame_locals: dict[str, Any]) -> bool:
            if frame_locals.get("name") == "remote-views-warm-box":
                starved.enter_context(nproc_limit_lowered_to_one())
            return False

        def restore_at_the_next_warning(_frame_locals: dict[str, Any]) -> bool:
            starved.close()  # the failed start is logged (so is an error that escapes)
            return False

        resume = {"type": "resumeSession", "tabId": "chat", "machine": "box"}
        with Pauses(
            Pause(threading.Thread.__init__.__code__, starve_the_warm_up),
            Pause(logging.Logger.warning.__code__, restore_at_the_next_warning),
        ):
            self.a.send(resume)
            refused = self.wait_event(self.a, "error", tabId="chat")
        self.assertIn("box: the stream could not start", refused["text"])
        self.a.send(resume)
        self.wait_event(self.a, "tabMachine", tabId="chat", machine="box")
        self.a.send({"type": "closeTab", "tabId": "chat"})
        self.wait_for(
            lambda: not (machine_of(tabs, "chat") or tabs._connecting or tabs._unbinds),
            what="the tab to unbind with no bind bookkeeping left",
        )

    def test_a_close_whose_send_thread_cannot_start_still_ends_the_stream(self) -> None:
        """The helper thread of a bound tab's forwarded ``closeTab`` is starved: the unbind ends."""
        api = self.server._server_api
        self.a.send({"type": "resumeSession", "tabId": "chat", "machine": "box"})
        self.wait_event(self.a, "tabMachine", tabId="chat", machine="box")
        self.wait_for(lambda: api.remote_views._streams, what="the machine's view stream")
        with api.remote_tabs._lock:
            stream = api.remote_tabs._tabs["chat"]
        starved = contextlib.ExitStack()
        self.addCleanup(starved.close)
        sender: list[int] = []  # the thread forwarding the closeTab

        def note_the_sender(frame_locals: dict[str, Any]) -> bool:
            if frame_locals["cmd"].get("type") == "closeTab":
                sender.append(threading.get_ident())
            return False

        def starve_its_helper(frame_locals: dict[str, Any]) -> bool:
            helper = frame_locals.get("name") == "sorcar-local-send"
            if helper and sender == [threading.get_ident()]:
                starved.enter_context(nproc_limit_lowered_to_one())
            return False

        def restore(frame_locals: dict[str, Any]) -> bool:
            # The send closes the socket on its own thread; an error that escapes it is logged.
            escaped = "Error handling" in str(frame_locals.get("msg"))
            if escaped or sender == [threading.get_ident()]:
                starved.close()
            return False

        with Pauses(
            Pause(MachineStream.send.__code__, note_the_sender),
            Pause(threading.Thread.__init__.__code__, starve_its_helper),
            Pause(MachineStream._close_socket.__code__, restore),
            Pause(logging.Logger.warning.__code__, restore),
        ):
            self.a.send({"type": "closeTab", "tabId": "chat"})
            # The unbind goes on past the failed send: the tab's stream is closed and
            # the machine's view stream, kept open for the binding, is released.
            self.wait_for(lambda: not api.remote_views._streams, what="the view stream to close")
        self.assertIs(stream._ws.protocol.state, State.CLOSED)

    def test_a_dropped_page_whose_grace_timer_cannot_start_has_its_shell_hung_up_at_once(
        self,
    ) -> None:
        """The grace timer of a dropped page's remote shell is starved: the shell ends now.

        Detached with no timer, nothing would ever hang the shell up, and
        the stream would stay open for a terminal no page owns.
        """
        if sys.platform == "win32":
            self.skipTest("terminal tabs need a pty")
        views = self.server._server_api.remote_views
        self.a.send({
            "type": "terminalOpen", "tab_id": "term-g", "cols": 80, "rows": 24,
            "workDir": "", "machine": "box",
        })
        self.wait_event(self.a, "terminalOpened", timeout=60, tab_id="term-g")
        starved = contextlib.ExitStack()
        self.addCleanup(starved.close)

        def starve_the_grace_timer(frame_locals: dict[str, Any]) -> bool:
            if frame_locals.get("function") == views._hang_up_detached:
                starved.enter_context(nproc_limit_lowered_to_one())
            return False

        def restore_at_the_hang_up(_frame_locals: dict[str, Any]) -> bool:
            starved.close()
            return False

        try:
            with Pauses(
                Pause(threading.Timer.__init__.__code__, starve_the_grace_timer),
                Pause(RemoteViews._hang_up_detached.__code__, restore_at_the_hang_up),
            ):
                self.a.close()
                self.wait_for(
                    lambda: not views._streams, timeout=20,
                    what="the shell hung up and its stream closed (the grace is 60 s)",
                )
        finally:
            starved.close()

    def test_a_dropped_page_whose_clean_up_thread_cannot_start_is_still_cleaned_up(self) -> None:
        """The thread of a dropped page's clean-up is starved: the clean-up is done all the same.

        Left undone (the failed start was only logged), the page stayed
        the owner of its remote shell for good: no grace period ever
        ran and the stream never closed.  Still starved, the grace
        timer cannot start either, so the shell ends at once (the test
        above).
        """
        if sys.platform == "win32":
            self.skipTest("terminal tabs need a pty")
        views = self.server._server_api.remote_views
        self.a.send({
            "type": "terminalOpen", "tab_id": "term-g", "cols": 80, "rows": 24,
            "workDir": "", "machine": "box",
        })
        self.wait_event(self.a, "terminalOpened", timeout=60, tab_id="term-g")
        starved = contextlib.ExitStack()
        self.addCleanup(starved.close)

        def starve_the_clean_up(frame_locals: dict[str, Any]) -> bool:
            if str(frame_locals.get("name")).startswith("remote-views-gone-"):
                starved.enter_context(nproc_limit_lowered_to_one())
            return False

        def restore_at_the_hang_up(_frame_locals: dict[str, Any]) -> bool:
            starved.close()
            return False

        def restore_at_a_clean_up_left_undone(frame_locals: dict[str, Any]) -> bool:
            if "teardown step" in str(frame_locals.get("msg")):
                starved.close()
            return False

        try:
            with Pauses(
                Pause(threading.Thread.__init__.__code__, starve_the_clean_up),
                Pause(RemoteViews._hang_up_detached.__code__, restore_at_the_hang_up),
                Pause(logging.Logger.warning.__code__, restore_at_a_clean_up_left_undone),
            ):
                self.a.close()
                self.wait_for(
                    lambda: not views._streams, timeout=20,
                    what="the dropped page's shell hung up and its stream closed",
                )
        finally:
            starved.close()
