# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the daemon machine's browser is a tab on EVERY client.

The daemon may run on a machine the user reaches only over ssh.  The
"Browser" entry of the composer's "..." menu launches that machine's
browser (``kiss/server/browser_tab.py``) and every page of it becomes a
tab that is open on every connected client, streams JPEG frames to the
clients showing it, and is closed on every client when the page closes.

This test runs the REAL daemon on its local WSS endpoint with several
real client connections (``two_client_harness``) and a REAL
Chromium-family browser (Playwright's bundled Chromium, headless)
against a local HTTP server.  Checked on the wire, per client:

1. ``browserOpen`` from one client announces the tab (``openBrowserTab``)
   to every connected client unfocused, plus a focused copy to the
   requester only; the page's title and URL follow (``browserState``).
2. Frames reach the client whose viewport shows the tab; none reach a
   client that does not show it.
3. A client connecting MID-SESSION gets the tab from its ``browserTabs``
   snapshot, unfocused.
4. A client whose connection drops and comes back gets a snapshot without
   a tab that was closed while it was away, with the surviving tab's
   current title, and streams again once it shows the tab.
5. ``browserClose`` from ONE client closes the tab on ALL clients
   (``closeBrowserTab``); the last close shuts the browser down, and a
   client connecting afterwards sees no browser tab.
"""

from __future__ import annotations

import http.server
import os
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import web_stealth
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

_PLAYWRIGHT_CACHE = Path.home() / ".cache" / "ms-playwright"


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        title = "Home Page" if self.path == "/" else self.path
        body = f"<html><head><title>{title}</title></head><body>{title}</body></html>"
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return


class BrowserTabTwoClientsTest(TwoClientHarness):
    """Open everywhere while the page lives, closed everywhere when it closes."""

    def setUp(self) -> None:
        if not _PLAYWRIGHT_CACHE.is_dir():
            self.skipTest("Playwright browsers are not installed")
        self._saved_env = {
            k: os.environ.get(k) for k in ("KISS_HEADLESS", "KISS_BROWSER_HOME")
        }
        os.environ["KISS_HEADLESS"] = "1"
        # Threaded: Chromium opens speculative connections it never
        # writes to, and a single-threaded server would sit in that
        # handler's readline while shutdown() waited for it.
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        os.environ["KISS_BROWSER_HOME"] = self.base + "/"
        super().setUp()
        self.service = self.server._agent_server.browser_tabs

    def tearDown(self) -> None:
        self.service.shutdown()
        web_stealth.stop_virtual_display()
        self.httpd.shutdown()
        self.httpd.server_close()
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        super().tearDown()

    # -- helpers -----------------------------------------------------------

    def _announced(self, client: Client) -> list[dict[str, Any]]:
        """Every ``openBrowserTab`` *client* got, live or via a snapshot."""
        found: list[dict[str, Any]] = []
        for e in list(client.events):
            if e.get("type") == "openBrowserTab":
                found.append(e)
            elif e.get("type") == "browserTabs":
                found.extend(e["tabs"])
        return found

    def _tab_ids(self, client: Client) -> list[str]:
        ids: list[str] = []
        for e in self._announced(client):
            if e["tab_id"] not in ids:
                ids.append(e["tab_id"])
        return ids

    def _show(self, client: Client, tab_id: str) -> None:
        """Tell the daemon *client* shows the tab at 400x300 (frames wanted)."""
        client.send({
            "type": "browserViewport",
            "tab_id": tab_id,
            "width": 400,
            "height": 300,
            "visible": True,
        })

    def _frames(self, client: Client, tab_id: str) -> int:
        return len(client.of_type("browserFrame", tab_id=tab_id))

    def _wait_title(self, client: Client, tab_id: str, title: str) -> dict[str, Any]:
        return self.wait_event(client, "browserState", timeout=60, tab_id=tab_id, title=title)

    # -- the test ------------------------------------------------------------

    def test_browser_tab_open_everywhere_closed_everywhere(self) -> None:
        a = self.open_client("a")
        a.ready()
        b = self.open_client("b")
        b.ready()
        for client in (a, b):
            self.assertEqual(
                self.wait_event(client, "browserTabs")["tabs"], [],
                "no browser tab before anyone opens one",
            )

        # 1. "Browser" on ONE client: announced to everyone unfocused,
        #    focused for the requester only; the page's state follows.
        a.send({"type": "browserOpen", "url": self.base + "/"})
        focused = self.wait_event(a, "openBrowserTab", timeout=60, focus=True)
        tab_id = str(focused["tab_id"])
        self.wait_event(a, "openBrowserTab", tab_id=tab_id, focus=False)
        self.wait_event(b, "openBrowserTab", tab_id=tab_id, focus=False)
        time.sleep(0.5)
        self.assertEqual(
            b.of_type("openBrowserTab", focus=True), [],
            "a client that did not click must not be switched to the tab",
        )
        self.assertEqual(self._tab_ids(a), [tab_id])
        self.assertEqual(self._tab_ids(b), [tab_id])
        for ev in a.of_type("openBrowserTab") + b.of_type("openBrowserTab"):
            self.assertTrue(ev["browser"], "the announcement names the browser")
            self.assertFalse(ev["popup"])
        for client in (a, b):
            state = self._wait_title(client, tab_id, "Home Page")
            self.assertEqual(state["url"], self.base + "/")

        # 2. Frames reach the client showing the tab; none reach b.
        self._show(a, tab_id)
        self.wait_for(lambda: self._frames(a, tab_id) > 0, what="a frame on a")
        time.sleep(0.5)
        self.assertEqual(self._frames(b, tab_id), 0, "b shows no tab: no frames")
        self._show(b, tab_id)
        self.wait_for(lambda: self._frames(b, tab_id) > 0, what="a frame on b")

        # 3. A client connecting mid-session gets the tab from its
        #    snapshot, unfocused.
        c = self.open_client("c")
        c.ready()
        snapshot = self.wait_event(c, "browserTabs")
        self.assertEqual([t["tab_id"] for t in snapshot["tabs"]], [tab_id])
        self.assertFalse(any(t["focus"] for t in snapshot["tabs"]))
        self.assertEqual(snapshot["tabs"][0]["title"], "Home Page")
        time.sleep(0.5)
        self.assertEqual(self._frames(c, tab_id), 0, "an unshown tab streams nothing")

        # 4. b drops its connection.  While it is away a second tab is
        #    opened and closed again, and the first page moves on.  The
        #    reconnected b gets a snapshot with only the first tab, its
        #    current title, and streams again once it shows the tab.
        viewers_before = dict(self.service._pages[tab_id].viewers)
        b.close()
        # Only a's viewer is left (c never showed the tab).
        self.wait_for(
            lambda: len(self.service._pages[tab_id].viewers) == 1,
            what=f"b's viewer gone (before={viewers_before}, "
            f"now={self.service._pages[tab_id].viewers})",
            timeout=10,
        )
        a.send({"type": "browserOpen", "url": self.base + "/second"})
        second = self.wait_for(
            lambda: [
                e for e in a.of_type("openBrowserTab", focus=True)
                if e["tab_id"] != tab_id
            ],
            timeout=60,
            what="a: the second tab, focused",
        )[0]
        second_id = str(second["tab_id"])
        self.wait_event(c, "openBrowserTab", tab_id=second_id, focus=False)
        a.send({"type": "browserClose", "tab_id": second_id})
        for client in (a, c):
            self.wait_event(client, "closeBrowserTab", tab_id=second_id)
        a.send({
            "type": "browserNavigate",
            "tab_id": tab_id,
            "action": "go",
            "url": self.base + "/moved",
        })
        for client in (a, c):
            self._wait_title(client, tab_id, "/moved")
        b2 = self.open_client("b2")
        b2.ready()
        snapshot = self.wait_event(b2, "browserTabs")
        self.assertEqual([t["tab_id"] for t in snapshot["tabs"]], [tab_id])
        self.assertEqual(snapshot["tabs"][0]["title"], "/moved")
        self.assertEqual(snapshot["tabs"][0]["url"], self.base + "/moved")
        self.assertEqual(b2.of_type("openBrowserTab", tab_id=second_id), [])
        self._show(b2, tab_id)
        self.wait_for(lambda: self._frames(b2, tab_id) > 0, what="frames resume on b2")

        # 5. Closing on ONE client closes on ALL; the last close shuts
        #    the browser down, and a later client sees no tab.
        c.send({"type": "browserClose", "tab_id": tab_id})
        for client in (a, b2, c):
            self.wait_event(client, "closeBrowserTab", tab_id=tab_id)
        self.wait_for(lambda: not self.service.open_events(), what="daemon forgets the tabs")
        self.wait_for(
            lambda: self.service._context is None,
            what="browser shut down after the last close",
        )
        late = self.open_client("late")
        late.ready()
        self.assertEqual(self.wait_event(late, "browserTabs")["tabs"], [])
        time.sleep(0.3)
        self.assertEqual(late.of_type("openBrowserTab"), [])
