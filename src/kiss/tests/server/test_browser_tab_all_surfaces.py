# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the daemon machine's browser is a tab on EVERY surface.

The daemon may run on a machine the user reaches only over ssh.  The
"Browser" entry of the composer's "..." menu launches that machine's
browser (``kiss/server/browser_tab.py``) and every page of it becomes a
tab that is open on every connected surface, streams JPEG frames to
the surfaces showing it, replays their input on the real page, and is
closed on every surface when the page closes.

This test runs the REAL daemon on its local WSS endpoint, REAL webviews under
jsdom (``test/multiSurfaceBridge.js``, one daemon connection each) and
a REAL Chromium-family browser (the machine's default browser when it
is Chromium-based, else Playwright's bundled Chromium) against a local
HTTP server.  Checked, from the rendered tab bars and address bars:

1. Clicking "Browser" on one surface opens the tab on every connected
   surface, with the page title and the browser's name; only the
   surface that clicked switches to it.
2. Every surface showing the tab receives frames; one that does not
   show it receives none.
3. A surface connecting MID-SESSION gets the tab from its ``ready``
   snapshot (unfocused, so it does not steal that surface's chat).
4. Keys typed on one surface reach the real page (the page navigates
   on Enter to a URL carrying the text) and every surface's address bar
   follows; a click on the streamed image follows a link.
5. A ``target=_blank`` link opens a second tab on every surface.
6. A surface whose connection drops and comes back (the remote webapp's
   shim re-authenticating) streams again, and drops a tab that was
   closed while it was away.
7. Closing a tab by hand on ONE surface closes it on ALL surfaces, and
   the last close shuts the browser down; a surface connecting
   afterwards sees no browser tab.
"""

from __future__ import annotations

import http.server
import os
import shutil
import socketserver
import threading
import time
from collections.abc import Callable
from functools import partial
from typing import Any, cast

from kiss.tests.conftest import PLAYWRIGHT_CHROMIUM_INSTALLED
from kiss.tests.server.test_run_agent_subagent_tab import DaemonLocalHarness
from kiss.tests.server.test_subagent_tabs_all_surfaces import (
    _JSDOM_PKG,
    SurfaceBridge,
)

_HOME_PAGE = """<!doctype html><html><head><title>Home Page</title></head>
<body style="margin:0;background:#fff">
<a id="big" href="/clicked" style="display:block;width:300px;height:300px;background:#48c">go</a>
<input id="q" autofocus style="display:block;width:300px;font-size:24px">
<a id="pop" href="/popup" target="_blank"
   style="display:block;width:300px;height:100px;background:#c84">pop</a>
<script>
document.getElementById('q').addEventListener('keydown', function (e) {
  if (e.key === 'Enter') location.href = '/typed?q=' + encodeURIComponent(this.value);
});
</script></body></html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/":
            body = _HOME_PAGE
        else:
            body = f"<html><head><title>{self.path}</title></head><body>{self.path}</body></html>"
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return


def _wait(pred: Callable[[], Any], what: str, timeout: float = 30) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = pred()
        if last:
            return last
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}; last={last!r}")


class BrowserTabAllSurfacesTest(DaemonLocalHarness):
    """Open everywhere while the page lives, closed everywhere when it closes."""

    def setUp(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest("jsdom is not installed under agents/vscode")
        if not PLAYWRIGHT_CHROMIUM_INSTALLED:
            self.skipTest("Playwright browsers are not installed")
        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self._saved_home = os.environ.get("KISS_BROWSER_HOME")
        os.environ["KISS_BROWSER_HOME"] = self.base + "/"
        super().setUp()
        self.bridge = SurfaceBridge(str(self.endpoint_file))

    def tearDown(self) -> None:
        self.bridge.quit()
        self.server._vscode_server.browser_tabs.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._saved_home is None:
            os.environ.pop("KISS_BROWSER_HOME", None)
        else:
            os.environ["KISS_BROWSER_HOME"] = self._saved_home
        super().tearDown()

    # -- helpers -----------------------------------------------------------

    def _announcements(self, surface: str) -> list[dict[str, Any]]:
        """Every ``openBrowserTab`` *surface* received, live or via a ``browserTabs`` snapshot."""
        found: list[dict[str, Any]] = []
        for e in self.bridge.events(surface):
            if not isinstance(e, dict):
                continue
            if e["type"] == "openBrowserTab":
                found.append(e)
            elif e["type"] == "browserTabs":
                found.extend(e["tabs"])
        return found

    def _browser_tab_ids(self, surface: str) -> list[str]:
        ids: list[str] = []
        for e in self._announcements(surface):
            if e["tab_id"] not in ids:
                ids.append(e["tab_id"])
        return ids

    def _info(self, surface: str, tab_id: str) -> dict[str, Any]:
        reply = self.bridge.call("browser", name=surface, tabId=tab_id)
        assert reply["op"] == "browser", reply
        return dict(reply["info"])

    def _has_tab(self, surface: str, tab_id: str) -> bool:
        return tab_id in self._browser_tab_ids(surface)

    def _in_tab_bar(self, surface: str, tab_id: str) -> bool:
        return bool(self._info(surface, tab_id)["inTabBar"])

    def _gone(self, surface: str, tab_id: str) -> bool:
        return not self._in_tab_bar(surface, tab_id)

    def _has_frame(self, surface: str, tab_id: str) -> bool:
        return bool(self._info(surface, tab_id)["hasFrame"])

    def _wait_info(
        self, surface: str, tab_id: str, ok: Callable[[dict[str, Any]], bool], what: str
    ) -> dict[str, Any]:
        """Poll the browser tab's view on *surface* until *ok* accepts it."""

        def _probe() -> dict[str, Any] | None:
            info = self._info(surface, tab_id)
            return info if ok(info) else None

        return cast("dict[str, Any]", _wait(_probe, f"{surface}: {what}"))

    def _wait_url(self, surface: str, tab_id: str, fragment: str) -> dict[str, Any]:
        return self._wait_info(
            surface, tab_id, lambda i: fragment in str(i.get("url")), f"url has {fragment!r}"
        )

    def _assert_no_webview_errors(self, *surfaces: str) -> None:
        for name in surfaces:
            self.bridge.tabs(name)  # raises on webview errors

    # -- the test ------------------------------------------------------------

    def test_browser_tab_open_everywhere_closed_everywhere(self) -> None:
        bridge = self.bridge
        bridge.open("sidebar")
        bridge.open("remote1")

        # 1. "Browser" in the ... menu of ONE surface.
        reply = bridge.call("click", name="sidebar", selector="#browser-btn")
        self.assertTrue(reply["found"], "chat.html has no #browser-btn")
        tab_id = str(
            _wait(
                lambda: next(iter(self._browser_tab_ids("sidebar")), None),
                "openBrowserTab on the sidebar",
                timeout=60,
            )
        )
        _wait(partial(self._has_tab, "remote1", tab_id), "openBrowserTab on remote1")
        for name in ("sidebar", "remote1"):
            info = self._wait_info(name, tab_id, lambda i: i["title"] == "Home Page", "tab title")
            self.assertTrue(info["inTabBar"] and info["isBrowserTab"], info)
            self.assertEqual(info["url"], self.base + "/")
            self.assertTrue(info["badge"], "the address bar names the streamed browser")
        self.assertTrue(self._info("sidebar", tab_id)["visible"], "the clicking surface switches")
        self.assertFalse(
            self._info("remote1", tab_id)["visible"], "other surfaces stay on their chat"
        )

        # 2. Frames reach the surface showing the tab; none reach remote1
        #    until the user switches to the tab there too.
        _wait(lambda: self._info("sidebar", tab_id)["hasFrame"], "a frame on the sidebar")
        time.sleep(0.5)
        self.assertEqual(self._info("remote1", tab_id)["frames"], 0)
        # The browser tab belongs to no chat; on a stacked surface (jsdom)
        # every content tab is listed on the group strip.
        reply = bridge.call("click", name="remote1", selector=f'#tab-list [data-tab-id="{tab_id}"]')
        self.assertTrue(reply["found"])
        _wait(partial(self._has_frame, "remote1", tab_id), "a frame on remote1")

        # 3. A surface connecting mid-session gets the tab from `ready`,
        #    unfocused.
        bridge.open("remote2")
        _wait(partial(self._has_tab, "remote2", tab_id), "replay on remote2")
        late = self._info("remote2", tab_id)
        self.assertTrue(late["inTabBar"] and late["isBrowserTab"], late)
        self.assertFalse(late["visible"], "a replayed browser tab must not steal focus")
        self.assertEqual(late["title"], "Home Page")
        self.assertFalse(any(e["focus"] for e in self._announcements("remote2")))
        # An unfocused surface asks for no frames.
        time.sleep(1.0)
        self.assertEqual(self._info("remote2", tab_id)["frames"], 0)

        # 4. Typing on remote1 reaches the real page (Enter navigates to
        #    /typed?q=...); every surface's address bar follows.
        reply = bridge.call("browserKeys", name="remote1", keys=["h", "i", "Enter"])
        self.assertTrue(reply["found"])
        for name in ("sidebar", "remote1", "remote2"):
            self._wait_url(name, tab_id, "/typed?q=hi")
        # Back to the home page via the toolbar, then click the big link.
        bridge.call(
            "post",
            name="sidebar",
            msg={"type": "browserNavigate", "tab_id": tab_id, "action": "back"},
        )
        self._wait_url("sidebar", tab_id, self.base + "/")
        self._wait_info("sidebar", tab_id, lambda i: i["title"] == "Home Page", "home title")
        reply = bridge.call("browserClick", name="sidebar", x=150, y=150)
        self.assertTrue(reply["found"])
        for name in ("sidebar", "remote1", "remote2"):
            self._wait_url(name, tab_id, "/clicked")

        # 5. A target=_blank link opens a second tab on every surface.
        bridge.call(
            "post",
            name="sidebar",
            msg={
                "type": "browserNavigate",
                "tab_id": tab_id,
                "action": "go",
                "url": self.base + "/",
            },
        )
        self._wait_info("sidebar", tab_id, lambda i: i["title"] == "Home Page", "home again")
        bridge.call("browserClick", name="sidebar", x=150, y=350)
        popup_id = str(
            _wait(
                lambda: next((t for t in self._browser_tab_ids("sidebar") if t != tab_id), None),
                "popup tab on the sidebar",
            )
        )
        for name in ("remote1", "remote2"):
            _wait(partial(self._has_tab, name, popup_id), f"popup tab on {name}")
        _wait(lambda: "/popup" in str(self._info("remote1", popup_id)["url"]), "popup url")

        # 6. remote1 (showing the first tab) drops its connection.  While it
        #    is away the popup is closed elsewhere.  On reconnect the
        #    `ready` snapshot drops the popup there and, since remote1 still
        #    shows the first tab, frames flow to its new connection.
        reply = bridge.call("disconnect", name="remote1")
        self.assertEqual(reply["op"], "disconnected")
        reply = bridge.call("closeTab", name="remote2", tabId=popup_id)
        self.assertTrue(reply["found"])
        for name in ("sidebar", "remote2"):
            _wait(partial(self._gone, name, popup_id), f"popup closed on {name}")
        self.assertTrue(self._info("remote1", popup_id)["inTabBar"], "away: still has the popup")
        # ... and the surviving page moves on while remote1 is away.
        bridge.call(
            "post",
            name="sidebar",
            msg={
                "type": "browserNavigate",
                "tab_id": tab_id,
                "action": "go",
                "url": self.base + "/moved",
            },
        )
        self._wait_url("sidebar", tab_id, "/moved")
        self.assertIn("Home Page", self._info("remote1", tab_id)["title"], "away: stale title")
        frames_before = self._info("remote1", tab_id)["frames"]
        reply = bridge.call("reconnect", name="remote1")
        self.assertEqual(reply["op"], "reconnected")
        _wait(partial(self._gone, "remote1", popup_id), "snapshot dropped the popup on remote1")
        moved = self._wait_url("remote1", tab_id, "/moved")
        self.assertEqual(moved["title"], "/moved", "snapshot refreshed the tab title")
        self.assertTrue(moved["visible"])
        _wait(
            lambda: self._info("remote1", tab_id)["frames"] > frames_before,
            "frames resume on the reconnected surface",
        )

        # 7. Closing by hand on ONE surface closes on ALL; the last close
        #    shuts the browser down.
        # The first tab survives the popup's close everywhere.
        for name in ("sidebar", "remote1", "remote2"):
            self.assertTrue(self._info(name, tab_id)["inTabBar"])
        reply = bridge.call("closeTab", name="remote1", tabId=tab_id)
        self.assertTrue(reply["found"])
        for name in ("sidebar", "remote1", "remote2"):
            _wait(partial(self._gone, name, tab_id), f"browser tab closed on {name}")
        service = self.server._vscode_server.browser_tabs
        _wait(lambda: not service.open_events(), "daemon forgets the tabs")
        _wait(lambda: service._context is None, "browser shut down after the last close")
        bridge.open("remote3")
        time.sleep(0.5)
        self.assertEqual(self._browser_tab_ids("remote3"), [])
        self._assert_no_webview_errors("sidebar", "remote1", "remote2", "remote3")
