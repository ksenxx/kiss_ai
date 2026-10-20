# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: the browser pages of a machine a chat tab is bound to are shown here.

A real local daemon with two pages (:class:`TwoClientHarness`), a real
remote daemon registered as ``box`` whose model is a fake endpoint
that first calls ``show_browser`` and then ``finish``.  A ``submit``
stamped ``machine: "box"`` binds the tab; the agent's call for the
user's attention (an ``openBrowserTab`` the remote broadcasts,
focused) must reach every page of this daemon although no page here
ever pressed the Browser button of the machine, and the pages the
machine's browser had open before the binding must show up too
(:mod:`kiss.server.remote_views`).

Not covered, as no real pair of daemons produces it on demand: a
machine whose view stream cannot be opened right after its tab stream
could (``RemoteViews.bind`` logging the failure), a ``browserList``
lost on a stream that died in the same instant, a snapshot entry that
is not a dict, a snapshot the machine answers after a close it relayed
earlier (the ``seen`` hold-back), and an older daemon answering
``browserList`` with an ``error``.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar import web_stealth
from kiss.server.remote_tabs import RemoteTab, RemoteTabs
from kiss.server.remote_views import REOPEN_DELAY, RemoteViewStream
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.pause import Pause
from kiss.tests.server.two_client_harness import TwoClientHarness

_PLAYWRIGHT_CACHE = Path.home() / ".cache" / "ms-playwright"
SUMMARY = "<p>shown</p>"


class _ShowBrowserModel(http.server.BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint: ``show_browser`` first, then ``finish`` once released."""

    released = threading.Event()
    calls = 0

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        _ShowBrowserModel.calls += 1
        if _ShowBrowserModel.calls == 1:
            function = {"name": "show_browser", "arguments": json.dumps({"visible": True})}
        else:
            _ShowBrowserModel.released.wait(60)
            function = {
                "name": "finish",
                "arguments": json.dumps({"success": True, "summary_in_html": SUMMARY}),
            }
        call = {"index": 0, "id": f"call-{_ShowBrowserModel.calls}", "type": "function",
                "function": function}
        base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "gpt-4o-mini"}
        chunks = [
            {**base, "choices": [{
                "index": 0, "delta": {"role": "assistant", "tool_calls": [call]},
                "finish_reason": None,
            }]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {**base, "choices": [], "usage": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            }},
        ]
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass


class _Home(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        data = b"<html><head><title>Remote Home</title></head><body>home</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return


class RemoteBoundTabBrowserPagesTest(TwoClientHarness):
    """Two local pages; ``box`` runs a headless browser and the fake model."""

    def setUp(self) -> None:
        if not _PLAYWRIGHT_CACHE.is_dir():
            self.skipTest("Playwright browsers are not installed")
        super().setUp()
        # Every resource is released by its own cleanup (after tearDown,
        # last registered first), whether or not the setup got further.
        mach.save_machines([])
        self.addCleanup(mach.save_machines, [])
        _ShowBrowserModel.released = threading.Event()
        _ShowBrowserModel.calls = 0
        self.model_server = self._serve(_ShowBrowserModel)
        self.home = self._serve(_Home)
        self.home_url = f"http://127.0.0.1:{self.home.server_address[1]}/"
        saved_env = {k: os.environ.get(k) for k in ("KISS_HEADLESS", "KISS_BROWSER_HOME")}
        self.addCleanup(self._restore_env, saved_env)
        os.environ["KISS_HEADLESS"] = "1"
        os.environ["KISS_BROWSER_HOME"] = self.home_url
        self.remote_dir = tempfile.mkdtemp(prefix="remote_bound_")
        self.addCleanup(shutil.rmtree, self.remote_dir, ignore_errors=True)
        self.remote = start_remote_daemon(
            Path(self.remote_dir) / "home", password="pw-remote",
            config={
                "custom_endpoint": f"http://127.0.0.1:{self.model_server.server_address[1]}/v1",
                "custom_api_key": "k", "max_budget": 100,
            },
        )
        # The daemon stops first (tearDown, its streams still open), the machine after it.
        self.addCleanup(web_stealth.stop_virtual_display)
        self.addCleanup(self.remote.stop)
        self.a = self.open_client("a")
        self.b = self.open_client("b")
        self.a.send({
            "type": "machineAction", "action": "add", "name": "box",
            "address": self.remote.url, "machinePassword": "pw-remote",
        })
        self.assertEqual(self.wait_event(self.a, "machines")["error"], "")

    def tearDown(self) -> None:
        _ShowBrowserModel.released.set()
        super().tearDown()

    def _serve(self, handler: type[http.server.BaseHTTPRequestHandler]) -> Any:
        """A local HTTP server of *handler* on a free port, shut down at cleanup."""
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    @staticmethod
    def _restore_env(saved: dict[str, str | None]) -> None:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _views(self) -> Any:
        return self.server._server_api.remote_views

    def _submit(self, tab_id: str) -> None:
        self.a.send({
            "type": "submit", "prompt": "Show me", "tabId": tab_id, "machine": "box",
            "model": "gpt-4o-mini", "workDir": self.remote_dir,
        })

    def _finish_task(self, tab_id: str, done: int = 1) -> None:
        """Let the fake model finish and wait for the task of *tab_id* to end (*done* times)."""
        _ShowBrowserModel.released.set()
        self.wait_for(
            lambda: len(self.a.of_type("status", tabId=tab_id, running=False)) >= done,
            timeout=120, what=f"task {done} of {tab_id} done",
        )

    def _open_on_box_directly(self, url: str) -> str:
        """Open *url* on ``box`` as one of its own pages would; the remote's own tab id.

        Returns once the page has loaded (its ``browserState`` carries
        the title), so a snapshot taken after this call shows the title.
        """
        box = next(m for m in mach.load_machines() if m.name == "box")
        ws = mach.link_for(box).open_connection()
        tab_id = ""
        try:
            frame = {"type": "browserOpen", "url": url, "password": "pw-remote"}
            ws.send(json.dumps(frame))
            for _ in range(500):
                msg = json.loads(ws.recv(timeout=120))
                if msg.get("type") == "openBrowserTab" and msg.get("focus"):
                    tab_id = str(msg["tab_id"])
                elif (
                    msg.get("type") == "browserState" and msg.get("tab_id") == tab_id
                    and msg.get("title") == "Remote Home"
                ):
                    return tab_id
        finally:
            ws.close()
        raise AssertionError("box never opened the page")

    def _close_remote_pages(self) -> None:
        """Close every page of ``box`` shown here from page a; the idle stream goes."""
        live: set[str] = set()
        for ev in list(self.a.events):
            if ev.get("type") == "browserTabs" and ev.get("remote"):
                live = {str(t["tab_id"]) for t in ev["tabs"]}
            elif ev.get("type") == "openBrowserTab" and ev.get("machine"):
                live.add(str(ev["tab_id"]))
            elif ev.get("type") == "closeBrowserTab" and ev.get("machine"):
                live.discard(str(ev["tab_id"]))
        for tab_id in live:
            self.a.send({"type": "browserClose", "tab_id": tab_id})
        self.wait_for(lambda: not self._views()._streams, timeout=60, what="idle stream closed")

    def test_the_agent_s_page_on_the_machine_is_a_focused_tab_on_every_page_here(self) -> None:
        a, b = self.a, self.b
        self._submit("tab-1")
        self.wait_event(a, "tabMachine", tabId="tab-1", machine="box")
        # Binding the tab opened the machine's view stream, without any
        # Terminal or Browser button pressed.
        self.wait_for(lambda: "box" in self._views()._streams, what="the view stream of box")

        # The agent's show_browser: every page here switches to the tab.
        shown = self.wait_event(a, "openBrowserTab", timeout=120, focus=True)
        self.assertEqual(shown["machine"], "box")
        self.assertTrue(shown["tab_id"].endswith("@box"), shown["tab_id"])
        self.wait_event(b, "openBrowserTab", timeout=60, tab_id=shown["tab_id"], focus=True)

        # A page connecting now is told the tab with the machine's others.
        c = self.open_client("c")
        c.ready()
        snapshot = self.wait_event(c, "browserTabs", remote=True)
        self.assertEqual([t["tab_id"] for t in snapshot["tabs"]], [shown["tab_id"]])
        self.assertEqual(snapshot["tabs"][0]["machine"], "box")

        self._finish_task("tab-1")
        self.assertIn("tab-1", self._views()._bound)
        # The stream outlives the task while the tab is bound to the machine.
        first = self._views()._streams["box"]

        # A view socket the machine drops (its tab stream alive) is
        # opened again for the binding, after the delay (no connect
        # loop against a machine that drops every socket): the next
        # task's page shows too.
        lost_at = time.monotonic()
        first._ws.close()
        self.wait_for(
            lambda: self._views()._streams.get("box") not in (None, first),
            timeout=60, what="the view stream reopened",
        )
        self.assertGreaterEqual(time.monotonic() - lost_at, REOPEN_DELAY)
        _ShowBrowserModel.released = threading.Event()
        _ShowBrowserModel.calls = 0
        self._submit("tab-1")
        again = self.wait_for(
            lambda: [e for e in b.of_type("openBrowserTab", focus=True)
                     if e["tab_id"] != shown["tab_id"]],
            timeout=120, what="the second task's page",
        )[0]
        self.assertEqual(again["machine"], "box")
        self._finish_task("tab-1", done=2)

        # Closing the tab unbinds it: the stream stays for its pages
        # only, and goes with the last of them.
        a.send({"type": "closeTab", "tabId": "tab-1"})
        self.wait_event(a, "tabMachine", tabId="tab-1", machine="")
        self.assertNotIn("tab-1", self._views()._bound)
        self._close_remote_pages()

    def test_pages_the_machine_already_had_open_are_shown_when_a_tab_binds(self) -> None:
        a, b = self.a, self.b
        remote_id = self._open_on_box_directly(self.home_url)
        self.assertEqual(self._views()._streams, {}, "no stream before the binding")

        self._submit("tab-2")
        self.wait_event(a, "tabMachine", tabId="tab-2", machine="box")
        # The page open before the stream is announced here, unfocused.
        for client in (a, b):
            earlier = self.wait_event(
                client, "openBrowserTab", timeout=120, tab_id=f"{remote_id}@box",
            )
            self.assertFalse(earlier["focus"])
            self.assertEqual(earlier["machine"], "box")
            self.assertEqual(earlier["title"], "Remote Home")
        # The agent's page comes focused, after it.
        shown = self.wait_event(a, "openBrowserTab", timeout=120, focus=True)
        self.assertNotEqual(shown["tab_id"], f"{remote_id}@box")
        self._finish_task("tab-2")

        # Re-binding the tab to the same machine keeps the one stream and
        # announces nothing twice.
        _ShowBrowserModel.calls = 0
        self._submit("tab-2")
        self._finish_task("tab-2", done=2)
        self.assertEqual(
            len(a.of_type("openBrowserTab", tab_id=f"{remote_id}@box")), 1,
        )
        self.assertEqual(list(self._views()._streams), ["box"])

    def test_the_browser_button_shows_the_machine_s_earlier_pages_too(self) -> None:
        """A stream opened by a page's own ``browserOpen`` asks for the earlier pages as well."""
        a, b = self.a, self.b
        remote_id = self._open_on_box_directly(self.home_url)
        a.send({"type": "browserOpen", "url": self.home_url, "machine": "box"})
        pressed = self.wait_event(a, "openBrowserTab", timeout=120, focus=True)
        self.assertNotEqual(pressed["tab_id"], f"{remote_id}@box")
        for client in (a, b):
            earlier = self.wait_event(
                client, "openBrowserTab", timeout=60, tab_id=f"{remote_id}@box",
            )
            self.assertFalse(earlier["focus"])
            self.assertEqual(earlier["title"], "Remote Home")
        self.assertEqual(self._views()._bound, {})
        self._close_remote_pages()

    def test_an_unbind_under_a_bind_still_announcing_wins(self) -> None:
        """A bind's announcement delayed past the tab's close is dropped.

        The binding thread is held as it enters ``_announce`` (the tab
        registered, the registry updated); the tab is closed meanwhile,
        which announces the unbinding.  Released, the bind must not
        announce a machine the registry no longer has: no client ends
        on ``box``, no view stream is opened for it.
        """
        a, b, views = self.a, self.b, self._views()
        with Pause(RemoteTabs._announce.__code__, lambda local: local["tab"] is not None) as held:
            self._submit("tab-4")
            self.assertTrue(held.paused.wait(60), "the bind announcing")
            b.send({"type": "closeTab", "tabId": "tab-4"})
            self.wait_event(b, "tabMachine", tabId="tab-4", machine="")
        time.sleep(1.0)
        for client in (a, b):
            self.assertEqual(
                [e["machine"] for e in client.of_type("tabMachine", tabId="tab-4")], [""],
            )
        self.assertEqual(views._bound, {})
        self.assertEqual(views._streams, {})

    def test_a_ready_s_replay_never_lands_after_a_later_unbind(self) -> None:
        """A reconnecting page is not told a binding the registry moved past meanwhile.

        The replay is held between copying the bindings and sending
        one; the tab is closed meanwhile (every page told).  Released,
        the replay must not put the reconnecting page back on ``box``.
        """
        a, b = self.a, self.b
        self._submit("tab-7")
        self.wait_event(a, "tabMachine", tabId="tab-7", machine="box")
        c = self.open_client("c")
        c.tab_id = "tab-7"  # the reloading page shows the tab (its frames reach the page)
        with Pause(RemoteTabs._replay_binding.__code__, lambda local: True) as held:
            c.ready()
            self.assertTrue(held.paused.wait(60), "the replay about to tell the binding")
            b.send({"type": "closeTab", "tabId": "tab-7"})
            self.wait_event(b, "tabMachine", tabId="tab-7", machine="")
            self.wait_event(c, "tabMachine", tabId="tab-7", machine="")
        time.sleep(1.0)
        self.assertEqual(
            [e["machine"] for e in c.of_type("tabMachine", tabId="tab-7")], [""],
        )
        self.assertEqual(self._views()._bound, {})

    def test_an_unbind_while_the_view_stream_connects_closes_it(self) -> None:
        """The view stream a binding opens is closed again when the binding went meanwhile.

        Held before it connects (nothing registered yet, so the close's
        own release finds no stream), the stream meets the tab's close;
        released, it connects and registers, and the open finds it
        idle and unbound: it goes (the machine's snapshot answer runs
        the same idle check, so the stream goes either way).
        """
        b, views = self.b, self._views()
        created: list[RemoteViewStream] = []

        def about_to_connect(local: dict[str, Any]) -> bool:
            created.append(local["self"])
            return True

        with Pause(RemoteViewStream.__init__.__code__, about_to_connect) as held:
            self._submit("tab-5")
            self.assertTrue(held.paused.wait(60), "the view stream about to connect")
            self.assertEqual(views._streams, {})
            b.send({"type": "closeTab", "tabId": "tab-5"})
            self.wait_event(b, "tabMachine", tabId="tab-5", machine="")
            self.assertEqual(views._bound, {})
        # The stream that then connected is the one that must go
        # (``retired`` exists once it is connected).
        self.wait_for(
            lambda: getattr(created[0], "retired", False), timeout=60,
            what="the unbound stream closed",
        )
        self.assertEqual(views._streams, {})
        self.assertEqual(views._bound, {})

    def test_a_bind_that_finds_the_tab_registered_waits_for_its_stream_too(self) -> None:
        """A second bind of a tab another bind registered forwards nothing before the stream.

        The first bind is held as it announces (the tab registered);
        a second ``submit`` for the tab from another page finds it
        bound to the same machine and announces nothing itself, but
        must still open the machine's stream before its run goes: the
        views learned the binding when the tab was registered.
        """
        a, b, views = self.a, self.b, self._views()
        with Pause(RemoteTabs._announce.__code__, lambda local: local["tab"] is not None) as held:
            self._submit("tab-8")
            self.assertTrue(held.paused.wait(60), "the first bind announcing")
            self.assertEqual(views._bound, {"tab-8": "box"})
            b.send({
                "type": "submit", "prompt": "Show me too", "tabId": "tab-8", "machine": "box",
                "model": "gpt-4o-mini", "workDir": self.remote_dir,
            })
            self.wait_for(lambda: "box" in views._streams, timeout=60, what="the view stream")
            self.assertEqual(a.of_type("tabMachine", tabId="tab-8"), [])
        self.wait_event(a, "tabMachine", tabId="tab-8", machine="box")
        # The agent's call for the user reached every page, focused.
        for client in (a, b):
            self.wait_event(client, "openBrowserTab", timeout=120, focus=True, machine="box")
        self._finish_task("tab-8")

    def test_a_bind_displaced_by_a_later_one_releases_its_machine_s_stream(self) -> None:
        """The view stream of a machine a tab was re-bound away from, while connecting, goes.

        The same daemon is registered twice (``box``, ``box2``).  A
        bind to ``box`` is held before it connects; a bind of the same
        tab to ``box2`` completes meanwhile (a resume of a chat the
        machine does not have: its view stream opened, no page on it).
        Released, the first displaces the second: the tab ends bound
        to ``box``, with ``box``'s stream alone.
        """
        a, b, views = self.a, self.b, self._views()
        a.send({
            "type": "machineAction", "action": "add", "name": "box2",
            "address": self.remote.url, "machinePassword": "pw-remote",
        })
        self.wait_for(lambda: len(a.of_type("machines")) >= 2, what="box2 added")
        with Pause(
            RemoteTab.__init__.__code__, lambda local: local["machine"].name == "box",
        ) as held:
            self._submit("tab-9")
            self.assertTrue(held.paused.wait(60), "the bind to box about to connect")
            b.send({
                "type": "resumeSession", "chatId": "no-such-chat", "tabId": "tab-9",
                "machine": "box2",
            })
            self.wait_event(b, "tabMachine", tabId="tab-9", machine="box2")
            self.wait_for(lambda: "box2" in views._streams, timeout=60, what="box2's stream")
        self.wait_event(b, "tabMachine", tabId="tab-9", machine="box")
        self.wait_for(
            lambda: list(views._streams) == ["box"], timeout=60, what="box2's stream released",
        )
        self.assertEqual(views._bound, {"tab-9": "box"})
        self._finish_task("tab-9")

    def test_the_daemon_stopping_opens_no_stream(self) -> None:
        """After ``close_all`` neither a reopen nor a command opens a stream.

        The bound machine's socket is dropped and the daemon's views
        close before the reopen's delay is up: the reopen finds the
        daemon closing.  A ``browserOpen`` after that is refused.
        """
        a, views = self.a, self._views()
        self._submit("tab-6")
        self.wait_event(a, "tabMachine", tabId="tab-6", machine="box")
        first = self.wait_for(lambda: views._streams.get("box"), what="the view stream")
        first._ws.close()
        self.wait_for(lambda: first.retired, what="the lost stream retired")
        views.close_all()
        self.assertEqual(views._bound, {})
        time.sleep(REOPEN_DELAY + 1.0)
        self.assertEqual(views._streams, {})
        a.send({"type": "browserOpen", "url": self.home_url, "machine": "box"})
        err = self.wait_event(a, "browserError", timeout=60)
        self.assertIn("this daemon is stopping", err["text"])
        self.assertEqual(views._streams, {})

    def test_losing_the_machine_closes_its_pages_here(self) -> None:
        a, b = self.a, self.b
        self._submit("tab-3")
        shown = self.wait_event(a, "openBrowserTab", timeout=120, focus=True)
        self.remote.stop()
        for client in (a, b):
            self.wait_event(client, "closeBrowserTab", timeout=60, tab_id=shown["tab_id"])
            self.wait_event(client, "tabMachine", timeout=60, tabId="tab-3", machine="")
        self.wait_for(lambda: not self._views()._streams, what="the lost stream forgotten")
        self.assertNotIn("tab-3", self._views()._bound)
