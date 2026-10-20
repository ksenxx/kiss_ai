# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of terminal and browser tabs on a remote machine.

A real local daemon (:class:`TwoClientHarness`, several real local
connections), a real remote daemon (:mod:`kiss.tests.remote_daemon`)
registered as the machine ``box``, and the real wire: a
``terminalOpen``/``browserOpen`` stamped ``machine: "box"`` runs the
shell, or opens the page, on the remote daemon
(:mod:`kiss.server.remote_views`), whose events come back to this
daemon's clients stamped with the machine.

Not covered here, as they need a failure no real pair of daemons
produces on demand (and test doubles are not used): a write that
fails on a stream the relay thread has not yet seen die
(``RemoteViewStream.send`` raising, ``_refuse`` after a send), the
remote browser failing to launch (a ``browserError`` without a tab),
a frame that is not JSON, a ``browserState`` for a page never
announced, the ``retired`` checks of the commands that follow an open
(an open meeting a stream closed as idle just too late is covered), and
an SSH-registered machine (no password on the frames).  The pages a
chat tab's binding brings here (the stream's ``browserList``) are
covered by :mod:`kiss.tests.server.test_remote_bound_tab_browser_pages`.
"""

from __future__ import annotations

import http.server
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar import web_stealth
from kiss.server.remote_views import RemoteViews, RemoteViewStream, namespaced, split_tab_id
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.pause import Pause
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

_PLAYWRIGHT_CACHE = Path.home() / ".cache" / "ms-playwright"


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        title = "Remote Home" if self.path == "/" else self.path
        body = f"<html><head><title>{title}</title></head><body>{title}</body></html>"
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return


class RemoteViewsTest(TwoClientHarness):
    """One local daemon with two pages, one remote daemon as the machine ``box``."""

    def setUp(self) -> None:
        super().setUp()
        mach.save_machines([])
        self.remote_dir = tempfile.mkdtemp(prefix="remote_views_")
        self.remote_work = Path(self.remote_dir) / "work"
        self.remote_work.mkdir()
        self.remote = start_remote_daemon(
            Path(self.remote_dir) / "home", password="pw-remote",
        )
        self.a = self.open_client("a")
        self.b = self.open_client("b")
        self.a.send({
            "type": "machineAction", "action": "add", "name": "box",
            "address": self.remote.url, "machinePassword": "pw-remote",
        })
        self.assertEqual(self.wait_event(self.a, "machines")["error"], "")

    def tearDown(self) -> None:
        # The daemon stops first, its streams still open (close_all).
        super().tearDown()
        self.remote.stop()
        mach.save_machines([])
        shutil.rmtree(self.remote_dir, ignore_errors=True)

    # -- helpers -----------------------------------------------------------

    def _views(self) -> Any:
        return self.server._server_api.remote_views

    def _output(self, client: Client, tab_id: str) -> str:
        return "".join(
            str(e.get("data", "")) for e in client.of_type("terminalData", tab_id=tab_id)
        )

    def _show(self, client: Client, tab_id: str) -> None:
        """Tell the daemon *client* shows the tab at 400x300 (frames wanted)."""
        client.send({
            "type": "browserViewport", "tab_id": tab_id, "width": 400, "height": 300,
            "visible": True,
        })

    def _hide(self, client: Client, tab_id: str) -> None:
        client.send({
            "type": "browserViewport", "tab_id": tab_id, "width": 0, "height": 0,
            "visible": False,
        })

    def _viewers(self, tab_id: str) -> set[str]:
        """The local connections the stream of ``box`` streams the page to."""
        stream = self._views()._streams["box"]
        with stream.lock:
            return set(stream.viewers.get(split_tab_id(tab_id)[0], ()))

    @staticmethod
    def _frames(tab_id: str, *clients: Client) -> int:
        """How many frames of the page *clients* got between them."""
        return sum(len(c.of_type("browserFrame", tab_id=tab_id)) for c in clients)

    def _more_frames(self, client: Client, tab_id: str) -> None:
        """Reload the page and wait for *client* to get a fresh frame."""
        before = len(client.of_type("browserFrame", tab_id=tab_id))
        client.send({"type": "browserNavigate", "tab_id": tab_id, "action": "reload", "url": ""})
        self.wait_for(
            lambda: len(client.of_type("browserFrame", tab_id=tab_id)) > before, timeout=60,
            what=f"{client.name}: more frames",
        )

    def _open_terminal(
        self, client: Client, tab_id: str, work_dir: str = "", machine: str = "box",
    ) -> dict[str, Any]:
        if sys.platform == "win32":
            self.skipTest("terminal tabs need a pty")
        client.send({
            "type": "terminalOpen", "tab_id": tab_id, "cols": 100, "rows": 30,
            "workDir": work_dir, "machine": machine,
        })
        return self.wait_event(client, "terminalOpened", timeout=60, tab_id=tab_id)

    # -- terminal ------------------------------------------------------------

    def test_terminal_runs_on_the_machine_for_its_opener_only(self) -> None:
        a, b = self.a, self.b
        opened = self._open_terminal(a, "term-1", work_dir=str(self.remote_work))
        self.assertEqual(opened["machine"], "box")
        self.assertFalse(opened["attached"])
        # The shell runs in the folder named on the machine.
        self.assertEqual(opened["cwd"], str(self.remote_work))

        a.send({
            "type": "terminalInput", "tab_id": "term-1",
            "data": "echo marker-$((40+2)); stty size\n",
        })
        self.wait_for(
            lambda: "marker-42" in self._output(a, "term-1")
            and "30 100" in self._output(a, "term-1"),
            timeout=60, what="the remote shell's echo",
        )
        a.send({"type": "terminalResize", "tab_id": "term-1", "cols": 120, "rows": 40})
        a.send({"type": "terminalInput", "tab_id": "term-1", "data": "stty size\n"})
        self.wait_for(lambda: "40 120" in self._output(a, "term-1"), timeout=60, what="resize")
        # Every event of the shell went to the page that opened it.
        self.assertEqual(b.of_type("terminalOpened") + b.of_type("terminalData"), [])
        for ev in a.of_type("terminalData", tab_id="term-1"):
            self.assertEqual(ev["machine"], "box")
        # The local daemon ran no shell of its own.
        self.assertEqual(self.server._agent_server.terminals.session_count(), 0)

        a.send({"type": "terminalInput", "tab_id": "term-1", "data": "exit 3\n"})
        exited = self.wait_event(a, "terminalExit", timeout=60, tab_id="term-1")
        self.assertEqual(exited["code"], 3)
        # Closing the tab ends the machine's stream: nothing is left on it.
        a.send({"type": "terminalClose", "tab_id": "term-1"})
        self.wait_for(lambda: not self._views()._streams, timeout=30, what="idle stream closed")

    def test_terminal_on_an_unknown_machine_is_refused_to_the_sender(self) -> None:
        a = self.a
        a.send({
            "type": "terminalOpen", "tab_id": "term-x", "cols": 80, "rows": 24,
            "workDir": "", "machine": "nowhere",
        })
        err = self.wait_event(a, "terminalError", tab_id="term-x")
        self.assertIn("nowhere is not a registered machine", err["text"])
        self.assertEqual(self.b.of_type("terminalError"), [])
        self.assertEqual(self._views()._streams, {})
        # Without a machine the shell is this daemon's, as before; a
        # browser command with a plain id is this daemon's too.
        self._open_terminal(a, "term-local", machine="")
        self.assertEqual(self.server._agent_server.terminals.session_count(), 1)
        a.send({"type": "browserViewport", "tab_id": "browser__9", "visible": False})
        a.send({"type": "terminalClose", "tab_id": "term-local"})
        self.wait_for(
            lambda: self.server._agent_server.terminals.session_count() == 0, what="hung up",
        )
        self.assertEqual(self._views()._streams, {})

    def test_losing_the_machine_ends_its_terminal(self) -> None:
        a = self.a
        self._open_terminal(a, "term-lost")
        self.remote.stop()
        err = self.wait_event(a, "terminalError", timeout=60, tab_id="term-lost")
        self.assertIn("lost the connection to box", err["text"])
        self.assertEqual(self._views()._streams, {})
        # A command for the gone terminal is nobody's: not the local daemon's either.
        a.send({"type": "terminalInput", "tab_id": "term-lost", "data": "echo x\n"})
        a.send({"type": "terminalClose", "tab_id": "term-lost"})
        time.sleep(0.3)
        self.assertEqual(self.server._agent_server.terminals.session_count(), 0)

    def test_a_dropped_page_s_terminal_is_reclaimed_by_its_reload(self) -> None:
        a = self.a
        self._open_terminal(a, "term-r")
        a.send({"type": "terminalInput", "tab_id": "term-r", "data": "MARK=kept-$$; echo $MARK\n"})
        self.wait_for(lambda: "kept-" in self._output(a, "term-r"), timeout=60, what="echo")
        a.close()
        # The reloaded page claims the shell (within the grace period).
        c = self.open_client("c")
        opened = self._open_terminal(c, "term-r")
        self.assertTrue(opened["attached"])
        c.send({"type": "terminalInput", "tab_id": "term-r", "data": "echo again-$MARK\n"})
        self.wait_for(
            lambda: "again-kept-" in self._output(c, "term-r"), timeout=60, what="reattach",
        )
        c.send({"type": "terminalClose", "tab_id": "term-r"})
        self.wait_for(lambda: not self._views()._streams, timeout=30, what="idle stream closed")

    def test_a_dropped_page_s_terminal_is_hung_up_after_the_grace_period(self) -> None:
        a = self.a
        self._open_terminal(a, "term-g")
        # A second terminal stays open until the daemon stops (close_all).
        self._open_terminal(self.b, "term-kept")
        stream = self._views()._streams["box"]
        a.close()
        self.wait_for(lambda: a.closed, what="page gone")
        # Nobody reclaimed the shell: the grace timer hangs it up (run
        # here directly, the real timer fires after GRACE_SECONDS).
        self._views()._hang_up_detached(stream, "term-g", "no-such-conn")
        self.assertIn("term-g", stream.terminals)  # another connection's claim is kept
        with stream.lock:
            conn_id = stream.terminals["term-g"]
        self._views()._hang_up_detached(stream, "term-g", conn_id)
        self.assertNotIn("term-g", stream.terminals)
        # The stream stays for the other terminal.
        time.sleep(0.3)
        self.assertIs(self._views()._streams.get("box"), stream)

    def test_a_terminal_obeys_the_page_that_opened_it_only(self) -> None:
        """Input, resize and close from another page are ignored (as locally)."""
        a, b = self.a, self.b
        self._open_terminal(a, "term-own")
        b.send({
            "type": "terminalInput", "tab_id": "term-own",
            "data": "echo unexpected-$((7*13))\n",
        })
        b.send({"type": "terminalResize", "tab_id": "term-own", "cols": 10, "rows": 10})
        b.send({"type": "terminalClose", "tab_id": "term-own"})
        time.sleep(1.0)
        a.send({
            "type": "terminalInput", "tab_id": "term-own",
            "data": "echo mine-$((6*7)); stty size\n",
        })
        self.wait_for(
            lambda: "mine-42" in self._output(a, "term-own"), timeout=60, what="a's own echo",
        )
        out = self._output(a, "term-own")
        self.assertNotIn("unexpected-91", out)
        self.assertIn("30 100", out)
        self.assertEqual(a.of_type("terminalExit"), [])
        self.assertEqual(b.of_type("terminalData") + b.of_type("terminalError"), [])
        # Neither daemon treated b's commands as its own.
        self.assertEqual(self.server._agent_server.terminals.session_count(), 0)
        self.assertIn("term-own", self._views()._streams["box"].terminals)
        a.send({"type": "terminalClose", "tab_id": "term-own"})
        self.wait_for(lambda: not self._views()._streams, timeout=30, what="idle stream closed")

    def test_the_grace_hang_up_never_kills_a_shell_a_reload_reclaimed(self) -> None:
        """The hang-up's owner check and its ``terminalClose`` are one step.

        The hang-up is held just before it sends (the window a reload's
        ``terminalOpen`` could slip into); the reload's open either waits
        for the close (and starts a fresh shell) or the hang-up finds
        the shell claimed: the reclaimed shell is never closed under
        the page that holds it.
        """
        a = self.a
        self._open_terminal(a, "term-race")
        views, stream = self._views(), self._views()._streams["box"]
        with stream.lock:
            conn_a = stream.terminals["term-race"]
        a.close()
        self.wait_for(lambda: a.closed, what="page gone")
        c = self.open_client("c")
        with Pause(
            RemoteViewStream.send.__code__,
            lambda local: local["cmd"].get("type") == "terminalClose",
        ) as pause:
            hang_up = threading.Thread(
                target=views._hang_up_detached, args=(stream, "term-race", conn_a),
            )
            hang_up.start()
            self.assertTrue(pause.paused.wait(10), "the hang-up reached its send")
            c.send({
                "type": "terminalOpen", "tab_id": "term-race", "cols": 80, "rows": 24,
                "workDir": "", "machine": "box",
            })
            time.sleep(1.0)
            pause.release()
            hang_up.join(30)
        opened = self.wait_event(c, "terminalOpened", timeout=60, tab_id="term-race")
        c.send({"type": "terminalInput", "tab_id": "term-race", "data": "echo alive-$((6*7))\n"})
        self.wait_for(
            lambda: "alive-42" in self._output(c, "term-race") or c.of_type("terminalExit"),
            timeout=60, what="c's shell answering",
        )
        self.assertEqual(c.of_type("terminalExit"), [])
        self.assertFalse(opened["attached"])  # the open waited for the hang-up
        c.send({"type": "terminalClose", "tab_id": "term-race"})
        self.wait_for(lambda: not self._views()._streams, timeout=30, what="idle stream closed")

    def test_losing_the_machine_fails_a_pending_browser_open(self) -> None:
        """A ``browserOpen`` the machine never answered gets a ``browserError``."""
        a, b = self.a, self.b
        self._open_terminal(a, "term-keep")  # keeps the stream open
        stream = self._views()._streams["box"]
        # The machine's daemon stops reading: b's open stays pending.
        os.kill(self.remote.pid, signal.SIGSTOP)
        try:
            b.send({"type": "browserOpen", "url": "", "machine": "box"})
            self.wait_for(lambda: stream.openers, what="b's open pending")
        finally:
            self.remote.stop()
        self.wait_event(a, "terminalError", timeout=60, tab_id="term-keep")
        err = self.wait_event(b, "browserError", timeout=60, tab_id="")
        self.assertIn("lost the connection to box", err["text"])
        self.assertEqual(err["machine"], "box")
        self.assertEqual(a.of_type("browserError"), [])
        self.assertEqual(self._views()._streams, {})

    def test_an_open_that_twice_meets_a_retiring_stream_is_refused_to_its_sender(self) -> None:
        """A stream retiring under a ``terminalOpen`` is retried once; the second time is refused.

        The window is between ``_stream_for`` handing out the registered
        stream and ``_book`` recording the open on it: the stream is
        closed as idle (``close_if_idle``) at the entry of ``_forward``,
        on both attempts.  The sender must be told (audit 2026-10-16
        server-rest-a F1: it got nothing and its opening tab stayed
        stuck), and the streams that retired are gone.
        """
        a, b, views = self.a, self.b, self._views()
        retired: list[RemoteViewStream] = []

        def retire_under(local: dict[str, Any]) -> bool:
            # Runs on the routing thread as ``_forward`` is entered: the
            # real ``close_if_idle`` retires the stream the command was
            # just handed, before ``_book`` sees it.  Never pauses.
            views.close_if_idle(local["stream"])
            retired.append(local["stream"])
            return False

        with Pause(RemoteViews._forward.__code__, retire_under):
            a.send({
                "type": "terminalOpen", "tab_id": "term-retired", "cols": 80, "rows": 24,
                "workDir": "", "machine": "box",
            })
            err = self.wait_event(a, "terminalError", timeout=60, tab_id="term-retired")
        self.assertIn("lost the connection to box", err["text"])
        self.assertEqual(len(retired), 2)
        self.assertIsNot(retired[0], retired[1])
        self.assertTrue(all(s.retired for s in retired))
        self.assertEqual(views._streams, {})
        self.assertEqual(a.of_type("terminalOpened"), [])
        self.assertEqual(b.of_type("terminalError"), [])
        # The machine is fine: the next open gets a fresh stream and a shell.
        self._open_terminal(a, "term-after")
        a.send({"type": "terminalClose", "tab_id": "term-after"})
        self.wait_for(lambda: not views._streams, timeout=30, what="idle stream closed")

    # -- browser -------------------------------------------------------------

    def test_browser_tab_ids(self) -> None:
        self.assertEqual(namespaced("browser__3", "box"), "browser__3@box")
        self.assertEqual(split_tab_id("browser__3@box"), ("browser__3", "box"))
        self.assertEqual(split_tab_id("browser__3"), ("browser__3", ""))

    def _with_headless_remote(self, check: Callable[[str], None]) -> None:
        """Run *check(base_url)* against ``box`` restarted with a headless browser.

        A small HTTP server (titles ``Remote Home`` and the path) is the
        remote browser's home page; the remote daemon inherits the
        environment that makes its browser headless, so it is restarted.
        """
        if not _PLAYWRIGHT_CACHE.is_dir():
            self.skipTest("Playwright browsers are not installed")
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        saved = {k: os.environ.get(k) for k in ("KISS_HEADLESS", "KISS_BROWSER_HOME")}
        os.environ["KISS_HEADLESS"] = "1"
        os.environ["KISS_BROWSER_HOME"] = base + "/"
        self.remote.stop()
        self.remote = start_remote_daemon(
            Path(self.remote_dir) / "home2", password="pw-remote",
        )
        self.a.send({
            "type": "machineAction", "action": "add", "name": "box",
            "address": self.remote.url, "machinePassword": "pw-remote",
        })
        self.wait_for(lambda: len(self.a.of_type("machines")) >= 2, what="box re-added")
        try:
            check(base)
        finally:
            web_stealth.stop_virtual_display()
            httpd.shutdown()
            httpd.server_close()
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _open_page(self, client: Client, url: str) -> str:
        """Open *url* on ``box`` from *client*; the namespaced id of the focused tab."""
        client.send({"type": "browserOpen", "url": url, "machine": "box"})
        return str(self.wait_event(client, "openBrowserTab", timeout=120, focus=True)["tab_id"])

    @staticmethod
    def _live_remote_tabs(client: Client) -> set[str]:
        """The remote tabs *client* ends up with, replaying its events in order."""
        live: set[str] = set()
        for ev in list(client.events):
            if ev.get("type") == "browserTabs" and ev.get("remote"):
                live = {str(t["tab_id"]) for t in ev["tabs"]}
            elif ev.get("type") == "openBrowserTab" and ev.get("machine"):
                live.add(str(ev["tab_id"]))
            elif ev.get("type") == "closeBrowserTab" and ev.get("machine"):
                live.discard(str(ev["tab_id"]))
        return live

    def test_browser_page_on_the_machine_is_a_tab_on_every_page_here(self) -> None:
        self._with_headless_remote(self._check_browser)

    def test_concurrent_viewport_and_open_commands_reach_the_machine_in_order(self) -> None:
        self._with_headless_remote(self._check_ordering)

    def _check_ordering(self, base: str) -> None:
        """What a command meant here and when it reaches the machine agree.

        The last viewer's hide is held just before it is sent while
        another page shows the tab: the show must not be overtaken by
        the stale hide (which would stop the frames for the new viewer).
        Then one page's ``browserOpen`` is held likewise while another's
        goes: the focused reply of each page's own open comes back to it.
        """
        a, b = self.a, self.b
        tab_id = self._open_page(a, base + "/")
        self.wait_event(b, "openBrowserTab", timeout=60, tab_id=tab_id)
        self._show(a, tab_id)
        self.wait_event(a, "browserFrame", timeout=60, tab_id=tab_id)
        with Pause(
            RemoteViewStream.send.__code__,
            lambda local: local["cmd"].get("type") == "browserViewport"
            and not local["cmd"].get("visible"),
        ) as pause:
            self._hide(a, tab_id)
            self.assertTrue(pause.paused.wait(10), "a's hide reached its send")
            self._show(b, tab_id)
            time.sleep(1.0)
            pause.release()
        self.wait_event(b, "browserFrame", timeout=60, tab_id=tab_id)
        self.assertEqual(len(self._viewers(tab_id)), 1)
        before = len(b.of_type("browserFrame", tab_id=tab_id))
        b.send({"type": "browserNavigate", "tab_id": tab_id, "action": "reload", "url": ""})
        self.wait_for(
            lambda: len(b.of_type("browserFrame", tab_id=tab_id)) > before, timeout=15,
            what="frames for the new viewer after the stale hide",
        )

        with Pause(
            RemoteViewStream.send.__code__,
            lambda local: local["cmd"].get("type") == "browserOpen"
            and str(local["cmd"].get("url", "")).endswith("/pageA"),
        ) as pause:
            a.send({"type": "browserOpen", "url": base + "/pageA", "machine": "box"})
            self.assertTrue(pause.paused.wait(10), "a's open reached its send")
            b.send({"type": "browserOpen", "url": base + "/pageB", "machine": "box"})
            time.sleep(1.0)
            pause.release()
        # The focused reply precedes the page's navigation: the page each
        # reply names is told by the state that follows.
        for client, page in ((a, "/pageA"), (b, "/pageB")):
            focused = self.wait_for(
                lambda: [
                    e for e in client.of_type("openBrowserTab", focus=True)
                    if e["tab_id"] != tab_id
                ],
                timeout=120, what=f"the focused reply to {page}",
            )
            self.assertEqual(len(focused), 1)
            self.assertEqual(focused[0]["machine"], "box")
            self.wait_event(
                client, "browserState", timeout=60, tab_id=focused[0]["tab_id"], url=base + page,
            )

    def test_a_reconnecting_page_is_told_the_remote_tabs_as_they_are(self) -> None:
        self._with_headless_remote(self._check_replay)

    def _check_replay(self, base: str) -> None:
        """The ``ready`` snapshot and the closes around it arrive in one order.

        The snapshot's delivery is held after it copied a page that
        another client then closes: the reconnecting page must not end
        up with the closed page (snapshot after close).  And once the
        machine's last page is closed while a page is away, its stream
        gone, the page's ``ready`` still gets an (empty) remote snapshot
        that drops the tab it kept.
        """
        a, b = self.a, self.b
        tab_id = self._open_page(a, base + "/")
        self.wait_event(b, "openBrowserTab", timeout=60, tab_id=tab_id)
        c = self.open_client("c")
        with Pause(
            RemoteViews.deliver_to.__code__,
            lambda local: '"browserTabs"' in str(local.get("text", "")),
        ) as pause:
            c.ready()
            self.assertTrue(pause.paused.wait(10), "the replay reached its delivery")
            b.send({"type": "browserClose", "tab_id": tab_id})
            time.sleep(1.0)
            pause.release()
        self.wait_event(c, "closeBrowserTab", timeout=60, tab_id=tab_id)
        self.wait_event(c, "browserTabs", timeout=30, remote=True)
        self.wait_for(lambda: not self._views()._streams, timeout=30, what="idle stream closed")
        self.assertEqual(self._live_remote_tabs(c), set())
        self.assertEqual(self._live_remote_tabs(b), set())

        # A page away while the last remote page closes and its stream
        # goes: its ready gets an empty remote snapshot.
        again = self._open_page(a, base + "/again")
        self.wait_event(b, "openBrowserTab", timeout=60, tab_id=again)
        b.close()
        a.send({"type": "browserClose", "tab_id": again})
        self.wait_event(a, "closeBrowserTab", timeout=60, tab_id=again)
        self.wait_for(lambda: not self._views()._streams, timeout=30, what="idle stream closed")
        d = self.open_client("d")
        d.ready()
        snapshot = self.wait_event(d, "browserTabs", timeout=10, remote=True)
        self.assertEqual(snapshot["tabs"], [])

    def _check_browser(self, base: str) -> None:
        a, b = self.a, self.b
        a.send({"type": "browserOpen", "url": base + "/", "machine": "box"})
        focused = self.wait_event(a, "openBrowserTab", timeout=120, focus=True)
        tab_id = str(focused["tab_id"])
        self.assertEqual(split_tab_id(tab_id)[1], "box")
        self.assertEqual(focused["machine"], "box")
        # Every page shows the tab; only the one that asked switches to it.
        self.wait_event(b, "openBrowserTab", timeout=60, tab_id=tab_id, focus=False)
        self.assertEqual(b.of_type("openBrowserTab", focus=True), [])
        state = self.wait_event(b, "browserState", timeout=60, tab_id=tab_id, title="Remote Home")
        self.assertEqual(state["machine"], "box")
        # No local browser was launched.
        self.assertEqual(self.server._agent_server.browser_tabs.open_events(), [])

        # A page connecting now is told the tab, unfocused and stamped,
        # in the remote snapshot.
        c = self.open_client("c")
        c.ready()
        snapshot = self.wait_event(c, "browserTabs", remote=True)
        self.assertEqual([t["tab_id"] for t in snapshot["tabs"]], [tab_id])
        self.assertFalse(snapshot["tabs"][0]["focus"])
        self.assertEqual(snapshot["tabs"][0]["machine"], "box")
        # The local daemon's own snapshot lists no remote tab.
        local = self.wait_for(
            lambda: [e for e in c.of_type("browserTabs") if not e.get("remote")],
            what="the local daemon's own browserTabs",
        )
        self.assertEqual(local[0]["tabs"], [])

        # Frames go to the page showing the tab and nowhere else.
        self._show(b, tab_id)
        frame = self.wait_event(b, "browserFrame", timeout=60, tab_id=tab_id)
        self.assertEqual(frame["machine"], "box")
        self.assertEqual(a.of_type("browserFrame"), [])
        self.assertEqual(c.of_type("browserFrame"), [])

        # Navigation is forwarded; the new title comes back to everyone.
        a.send({
            "type": "browserNavigate", "tab_id": tab_id, "action": "go", "url": base + "/second",
        })
        self.wait_event(a, "browserState", timeout=60, tab_id=tab_id, title="/second")
        self.wait_event(c, "browserState", timeout=60, tab_id=tab_id, title="/second")

        # A second viewer: hiding the tab on one keeps the frames flowing
        # to the other (the machine is told nothing until nobody looks),
        # and so does a viewer's page going away.
        self._show(c, tab_id)
        self.wait_event(c, "browserFrame", timeout=60, tab_id=tab_id)
        self._hide(b, tab_id)
        self._more_frames(c, tab_id)
        self._show(a, tab_id)
        self.wait_event(a, "browserFrame", timeout=60, tab_id=tab_id)
        self.assertEqual(len(self._viewers(tab_id)), 2)
        a.close()
        self.wait_for(
            lambda: len(self._viewers(tab_id)) == 1, what="the dropped page watches no more",
        )
        self._more_frames(c, tab_id)
        # Hiding on the only viewer is forwarded: the frames stop; a
        # page showing the tab again gets frames again.
        self._hide(c, tab_id)
        self.wait_for(lambda: not self._viewers(tab_id), what="nobody watching")
        time.sleep(0.5)
        seen = self._frames(tab_id, b, c)
        time.sleep(1.0)
        self.assertEqual(self._frames(tab_id, b, c), seen)
        before = len(b.of_type("browserFrame", tab_id=tab_id))
        self._show(b, tab_id)
        self.wait_for(
            lambda: len(b.of_type("browserFrame", tab_id=tab_id)) > before, timeout=60,
            what="frames again",
        )
        # The last viewer's page going away stops the cast too.
        d = self.open_client("d")
        self._show(d, tab_id)
        self.wait_event(d, "browserFrame", timeout=60, tab_id=tab_id)
        self._hide(b, tab_id)
        d.close()
        self.wait_for(lambda: not self._viewers(tab_id), what="nobody watching after the drop")
        time.sleep(0.5)
        seen = len(b.of_type("browserFrame", tab_id=tab_id))
        time.sleep(1.0)
        self.assertEqual(len(b.of_type("browserFrame", tab_id=tab_id)), seen)


        # Closing from one page closes it on every page, and the idle stream goes.
        c.send({"type": "browserClose", "tab_id": tab_id})
        for client in (b, c):
            self.wait_event(client, "closeBrowserTab", timeout=60, tab_id=tab_id)
        self.wait_for(lambda: not self._views()._streams, timeout=30, what="idle stream closed")
        # A late command for the gone tab is swallowed, not run locally.
        b.send({"type": "browserClose", "tab_id": tab_id})
        time.sleep(0.3)
        self.assertEqual(self.server._agent_server.browser_tabs.open_events(), [])

        # Losing the machine closes its pages on every surface.
        c.send({"type": "browserOpen", "url": base + "/again", "machine": "box"})
        again = str(self.wait_event(c, "openBrowserTab", timeout=120, focus=True)["tab_id"])
        self.wait_event(b, "openBrowserTab", timeout=60, tab_id=again)
        self.remote.stop()
        for client in (b, c):
            self.wait_event(client, "closeBrowserTab", timeout=60, tab_id=again, machine="box")
        self.wait_for(lambda: not self._views()._streams, timeout=30, what="lost stream dropped")
