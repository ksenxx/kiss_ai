# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""The remote webapp's WebSocket shim reconnects in place, never by reload.

The Playwright tests drive the production ``_WS_SHIM_JS`` string
verbatim inside a synthetic page (no vscode media assets) served from a
routed origin: an instrumented ``WebSocket`` stub lets the tests take
the shim through drop-then-reauth cycles and observe, from outside the
page via Playwright's ``framenavigated`` events, that no reload happens
— the page keeps its JavaScript state and is told the socket is back
through a ``daemonStatus`` event, which is what makes ``main.js`` send
``ready`` again and receive the server's updates.

The one reload the shim still performs is for a page the service
worker served from its offline cache (``<meta name="kiss-offline-shell">``):
its code may be older than the server's, so it reloads once, on its
first ``auth_ok``, and never again in the same browsing session.
"""


from __future__ import annotations

import pytest
from playwright.sync_api import sync_playwright

from kiss.server.web_server import _WS_SHIM_JS

PAGE_URL = "https://shim.test/"


def _build_test_page(offline_shell: bool = False) -> str:
    """Return an HTML page that loads the real shim with a mock WS.

    The page installs an instrumented ``WebSocket`` constructor
    BEFORE the shim's IIFE runs.  Test code then drives the shim
    through ``auth_ok → close → auth_ok`` cycles by toggling the mock
    socket.  Reloads are observed from OUTSIDE the page via
    Playwright's ``framenavigated`` events, because modern Chromium
    does not allow page JS to replace ``Location.prototype.reload``.

    The page also includes the ``#kiss-server-loading`` overlay node
    (faithful to ``media/chat.html``) and, when *offline_shell* is
    set, the meta tag ``media/sw.js`` stamps on a cached copy.
    """
    meta = '<meta name="kiss-offline-shell" content="1">' if offline_shell else ""
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  {meta}
  <title>shim test</title>
</head>
<body>
  <div id="kiss-server-loading" role="status" aria-live="polite">
    <div id="kiss-server-loading-msg">KISS Sorcar Server is starting ...</div>
  </div>
  <div id="app" style="display:none;"></div>
  <script>
    // Trace state exposed to Playwright.  A reload would wipe all of
    // it, which is exactly what the tests assert does not happen.
    window.__daemonStatusEvents = [];
    window.__sockets = [];
    window.__openSocket = null;
    window.__pageState = 'kept';

    window.addEventListener('message', function(e) {{
      var d = e && e.data;
      if (d && d.type === 'daemonStatus') {{
        window.__daemonStatusEvents.push({{
          connected: d.connected, reconnecting: d.reconnecting === true,
        }});
      }}
    }});

    // Mock WebSocket: every constructor call records the instance
    // and stores it on ``__openSocket`` so the test can fire
    // ``onopen`` / ``onmessage`` / ``onclose`` on demand.  The mock
    // also remembers everything ``send()`` was called with and, like
    // the server, answers the shim's ``ping`` probe with ``pong`` once
    // the earlier commands have been "taken" (next tick).
    var _MockWS = function(url) {{
      this.url = url;
      this.readyState = 0; // CONNECTING
      this.sent = [];
      this.onopen = null;
      this.onmessage = null;
      this.onclose = null;
      this.onerror = null;
      window.__sockets.push(this);
      window.__openSocket = this;
    }};
    _MockWS.prototype.send = function(data) {{
      this.sent.push(data);
      var ws = this;
      if (JSON.parse(data).type === 'ping') {{
        _origSetTimeout(function() {{
          if (ws.readyState === 1 && ws.onmessage) {{
            ws.onmessage({{data: JSON.stringify({{type: 'pong'}})}});
          }}
        }}, 0);
      }}
    }};
    _MockWS.prototype.close = function() {{
      this.readyState = 3;
      if (this.onclose) this.onclose({{}});
    }};
    _MockWS.CONNECTING = 0;
    _MockWS.OPEN = 1;
    _MockWS.CLOSING = 2;
    _MockWS.CLOSED = 3;
    window.WebSocket = _MockWS;

    window.__fireOpen = function() {{
      var ws = window.__openSocket;
      ws.readyState = 1;
      if (ws.onopen) ws.onopen({{}});
    }};
    window.__fireAuthOk = function() {{
      var ws = window.__openSocket;
      if (ws.onmessage) {{
        ws.onmessage({{data: JSON.stringify({{type: 'auth_ok'}})}});
      }}
    }};
    window.__fireClose = function() {{
      var ws = window.__openSocket;
      ws.readyState = 3;
      if (ws.onclose) ws.onclose({{}});
    }};
    window.__sentTypes = function() {{
      return window.__openSocket.sent.map(function(d) {{
        return JSON.parse(d).type;
      }});
    }};
    // Run the shim's deferred reconnect immediately so the test does
    // not have to wait for the backoff (250 ms doubling to 5 s).
    // Delays < 100 ms (the auth-modal focus call) are left alone.
    var _origSetTimeout = window.setTimeout;
    window.setTimeout = function(fn, ms) {{
      if (typeof fn === 'function' && ms >= 100 && ms <= 5000) {{
        try {{ fn(); }} catch (e) {{}}
        return 0;
      }}
      return _origSetTimeout(fn, ms);
    }};
  </script>
  <script>
{_WS_SHIM_JS}
  </script>
</body>
</html>
"""


@pytest.fixture(scope="module")
def _browser():
    """Module-scoped headless Chromium for the shim tests."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


def _load_shim_page(browser, offline_shell: bool = False):
    """Open the shim test page at a real origin; return context, page, navs.

    ``navs[0]`` counts main-frame navigations after the initial load:
    any increment proves the shim issued a real ``location.reload()``.
    The page is served through ``page.route`` so it has a proper
    origin (``sessionStorage`` works, unlike on ``about:blank``) and so
    a reload lands on the same page again.
    """
    context = browser.new_context()
    page = context.new_page()
    html = _build_test_page(offline_shell)
    page.route(
        PAGE_URL + "**",
        lambda route: route.fulfill(body=html, content_type="text/html"),
    )
    navs = [0]

    def _on_nav(frame) -> None:
        if frame == page.main_frame:
            navs[0] += 1

    page.on("framenavigated", _on_nav)
    page.goto(PAGE_URL, wait_until="load")
    navs[0] = 0
    return context, page, navs


def _authenticate_current_socket(page) -> None:
    page.evaluate("window.__fireOpen()")
    page.evaluate("window.__fireAuthOk()")


def test_reconnect_after_drop_resyncs_in_place(_browser):
    """A drop-then-reauth cycle keeps the page: no reload, state intact.

    The server goes away after the user authenticated (socket closes),
    the shim's retry succeeds and the new socket re-authenticates.
    The page must NOT be reloaded: its JavaScript state survives, the
    command posted during the outage goes out on the new socket, and
    the app is told the daemon is back (``daemonStatus connected``),
    which is what makes ``main.js`` re-send ``ready`` and receive the
    server's updates.
    """
    context, page, navs = _load_shim_page(_browser)
    try:
        _authenticate_current_socket(page)
        page.wait_for_timeout(100)
        assert navs[0] == 0, f"first auth_ok must not reload; saw {navs[0]}"
        page.evaluate("window.__pageState = 'set before the outage'")

        page.evaluate("window.__fireClose()")
        assert page.evaluate("window.__sockets.length") == 2, (
            "shim must reconnect after onclose"
        )
        page.evaluate(
            "window.acquireVsCodeApi().postMessage("
            "{type: 'saveConfig', config: {edited: 'during the outage'}})"
        )
        _authenticate_current_socket(page)
        page.wait_for_timeout(500)

        assert navs[0] == 0, (
            f"BUG: a reconnect reloaded the page ({navs[0]} navigation(s)); "
            "the app must keep its state and only receive updates"
        )
        assert page.evaluate("window.__pageState") == "set before the outage"
        assert page.evaluate("window.__sentTypes()") == ["auth", "saveConfig", "ping"], (
            "the command queued during the outage is flushed, then the server is probed"
        )
        events = page.evaluate("window.__daemonStatusEvents")
        assert events == [
            {"connected": True, "reconnecting": False},
            {"connected": False, "reconnecting": True},
            {"connected": True, "reconnecting": False},
        ], events
        assert page.evaluate(
            "document.getElementById('kiss-server-loading-msg').textContent"
        ) == "Reconnecting to KISS Sorcar Server ..."

        # A second cycle on the same document behaves the same way.
        page.evaluate("window.__fireClose()")
        _authenticate_current_socket(page)
        page.wait_for_timeout(300)
        assert navs[0] == 0
        assert page.evaluate("window.__pageState") == "set before the outage"
    finally:
        context.close()


@pytest.mark.parametrize(
    "wake_event", ["focus", "pageshow", "online", "visibilitychange"],
)
def test_wakeup_reconnect_while_old_socket_closing_keeps_page(
    _browser, wake_event: str,
) -> None:
    """Mobile-Safari ordering: a wake-up replaces a CLOSING socket.

    JS resumes with the dead socket still ``CLOSING`` and the wake-up
    event arrives before its queued ``onclose``.  ``connect()`` swaps
    the socket atomically; the fresh ``auth_ok`` must keep the page
    (no reload) and tell the app the daemon is back.
    """
    context, page, navs = _load_shim_page(_browser)
    try:
        _authenticate_current_socket(page)
        page.wait_for_timeout(100)
        page.evaluate("window.__pageState = 'before the app switch'")
        page.evaluate("window.__openSocket.readyState = 2")
        target = "document" if wake_event == "visibilitychange" else "window"
        page.evaluate(f"{target}.dispatchEvent(new Event({wake_event!r}))")
        assert page.evaluate("window.__sockets.length") == 2, (
            "wake-up listener must open a replacement socket"
        )
        _authenticate_current_socket(page)
        page.wait_for_timeout(300)
        assert navs[0] == 0, f"wake-up reconnect must not reload; saw {navs[0]}"
        assert page.evaluate("window.__pageState") == "before the app switch"
        events = page.evaluate("window.__daemonStatusEvents")
        assert events == [
            {"connected": True, "reconnecting": False},
            {"connected": False, "reconnecting": True},
            {"connected": True, "reconnecting": False},
        ], "the replacement must report the drop so main.js re-sends ready: " + repr(events)
    finally:
        context.close()


def test_offline_shell_page_reloads_once_on_first_auth(_browser):
    """A cached (offline-shell) page reloads once, then stays put.

    The reload happens on the first ``auth_ok`` and is recorded in
    ``sessionStorage``; the reloaded page — still the cached copy in
    this synthetic setup — sees the guard and keeps itself, and a
    later drop-then-reauth on it does not reload either.
    """
    context, page, navs = _load_shim_page(_browser, offline_shell=True)
    try:
        assert page.evaluate("sessionStorage.getItem('sorcar-offline-reloaded')") is None
        page.evaluate("window.__pageState = 'cached copy'")
        with page.expect_navigation(wait_until="load"):
            _authenticate_current_socket(page)
        assert navs[0] == 1, f"offline-shell page must reload once; saw {navs[0]}"
        assert page.evaluate("sessionStorage.getItem('sorcar-offline-reloaded')") == "1"
        assert page.evaluate("window.__pageState") == "kept", "fresh page after reload"

        _authenticate_current_socket(page)
        page.wait_for_timeout(200)
        assert navs[0] == 1, "the guard stops a second reload"
        events = page.evaluate("window.__daemonStatusEvents")
        assert events == [{"connected": True, "reconnecting": False}], events

        page.evaluate("window.__fireClose()")
        _authenticate_current_socket(page)
        page.wait_for_timeout(300)
        assert navs[0] == 1, "a kept cached page reconnects in place too"
    finally:
        context.close()
