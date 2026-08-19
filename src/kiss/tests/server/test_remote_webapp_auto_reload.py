# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Server-only tests from ``kiss.tests.agents.vscode.test_remote_webapp_auto_reload``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).

The Playwright tests drive the production ``_WS_SHIM_JS`` string
verbatim inside a SYNTHETIC page (no vscode media assets): an
instrumented ``WebSocket`` stub lets the tests assert that a
drop-then-reauth cycle triggers ``window.location.reload()`` while a
single first ``auth_ok`` never does.
"""


from __future__ import annotations

import re

import pytest
from playwright.sync_api import sync_playwright

from kiss.server.web_server import _WS_SHIM_JS


def test_shim_source_contains_reload_call():
    """Sanity check: the shipped shim source includes a reload call.

    A second, source-level guard so a future refactor that removes
    the reload (and is somehow missed by the browser-level tests
    below — e.g. an inadvertent rename of the mock helpers) still
    fails loudly.  This is intentionally narrow: it only asserts
    that ``location.reload`` appears somewhere in the shim string.
    """
    assert re.search(r"location\s*\.\s*reload\s*\(", _WS_SHIM_JS), (
        "BUG: _WS_SHIM_JS must call location.reload() to recover from "
        "a server restart"
    )


def _build_test_page() -> str:
    """Return an HTML page that loads the real shim with a mock WS.

    The page installs an instrumented ``WebSocket`` constructor
    BEFORE the shim's IIFE runs.  Test code then drives the shim
    through a real ``auth_ok → close → auth_ok`` cycle by toggling
    the mock socket; reloads are observed from OUTSIDE the page via
    Playwright's ``framenavigated`` events (see ``_load_shim_page``),
    because modern Chromium does not allow page JS to replace
    ``Location.prototype.reload``.

    The page also includes the ``#kiss-server-loading`` overlay node
    (faithful to ``media/chat.html``) so any future shim revision
    that toggles the overlay via the DOM rather than the
    ``daemonStatus`` event keeps working.
    """
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>shim test</title>
</head>
<body>
  <div id="kiss-server-loading" role="status" aria-live="polite">
    <div class="kiss-server-loading-msg">KISS Sorcar Server is starting ...</div>
  </div>
  <div id="app" style="display:none;"></div>
  <script>
    // Trace state exposed to Playwright.  Reload itself is observed
    // outside the page (via Playwright's ``framenavigated`` event)
    // because modern Chromium does not allow page JS to replace
    // ``Location.prototype.reload`` — letting the real reload run
    // is the only reliable way to assert the shim called it.
    window.__daemonStatusEvents = [];
    window.__sockets = [];
    window.__openSocket = null;

    // Capture every daemonStatus event the shim dispatches.
    window.addEventListener('message', function(e) {{
      var d = e && e.data;
      if (d && d.type === 'daemonStatus') {{
        window.__daemonStatusEvents.push({{
          connected: d.connected, t: Date.now(),
        }});
      }}
    }});

    // Mock WebSocket: every constructor call records the instance
    // and stores it on ``__openSocket`` so the test can fire
    // ``onopen`` / ``onmessage`` / ``onclose`` on demand.  The mock
    // also remembers everything ``send()`` was called with.
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
    _MockWS.prototype.send = function(data) {{ this.sent.push(data); }};
    _MockWS.prototype.close = function() {{
      this.readyState = 3;
      if (this.onclose) this.onclose({{}});
    }};
    _MockWS.CONNECTING = 0;
    _MockWS.OPEN = 1;
    _MockWS.CLOSING = 2;
    _MockWS.CLOSED = 3;
    window.WebSocket = _MockWS;

    // Helpers Playwright calls to drive the shim.
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
    // Run the shim's deferred reconnect immediately so the test
    // does not have to actually wait for the backoff.  The shim now
    // uses an exponential backoff starting at 250ms (was a fixed
    // 3000ms), so we intercept any setTimeout whose delay falls in
    // the range used by the reconnect path.  Delays < 100ms (e.g.
    // the 0ms auth-modal focus call) are left alone.
    var _origSetTimeout = window.setTimeout;
    window.setTimeout = function(fn, ms) {{
      if (typeof fn === 'function' && ms >= 100 && ms <= 5000) {{
        // Run synchronously, so __openSocket is replaced before we
        // return control to the test.
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


def _load_shim_page(browser):
    """Open the shim test page and return a Playwright ``Page``.

    Also returns a mutable ``nav_counter`` list whose ``[0]`` entry
    counts main-frame navigations.  After the initial ``set_content``
    returns, ``nav_counter[0]`` is reset to 0 — any subsequent
    increment proves the shim issued a real ``location.reload()``
    (modern Chromium does not allow tampering with
    ``Location.prototype.reload`` from page JS, so we observe the
    side effect directly through Playwright's frame navigation
    notifications).
    """
    context = browser.new_context()
    page = context.new_page()
    nav_counter = [0]

    def _on_nav(frame) -> None:
        if frame == page.main_frame:
            nav_counter[0] += 1

    page.on("framenavigated", _on_nav)
    page.set_content(_build_test_page(), wait_until="load")
    nav_counter[0] = 0
    return context, page, nav_counter


def test_reconnect_after_server_restart_triggers_page_reload(_browser):
    """A drop-then-reauth cycle must call ``window.location.reload()``.

    Simulates the real failure mode: server is up, the user
    authenticates, the server restarts (socket closes), then the
    shim's scheduled retry succeeds and the new socket
    re-authenticates.  The fix must reload the page on that second
    ``auth_ok`` so the page re-runs its load pipeline against the
    restored backend state.
    """
    context, page, navs = _load_shim_page(_browser)
    try:
        page.evaluate("window.__fireOpen()")
        page.evaluate("window.__fireAuthOk()")
        page.wait_for_timeout(100)
        assert navs[0] == 0, (
            f"shim must NOT reload on the very first successful auth; "
            f"saw {navs[0]} extra navigation(s)"
        )

        page.evaluate("window.__fireClose()")

        socket_count = page.evaluate("window.__sockets.length")
        assert socket_count == 2, (
            f"shim must reconnect after onclose; got {socket_count} sockets"
        )

        try:
            page.evaluate("window.__fireOpen()")
            page.evaluate("window.__fireAuthOk()")
        except Exception:
            pass

        page.wait_for_timeout(500)

        assert navs[0] >= 1, (
            "BUG: after server restart the remote webapp shim does not "
            "reload the page; the user sees 'KISS Sorcar Server is "
            "starting ...' forever and the in-flight task UI stays "
            "stale until the user manually refreshes"
        )
    finally:
        context.close()


def test_first_auth_alone_does_not_reload(_browser):
    """The very first ``auth_ok`` must NOT cause a reload.

    Guards against the obvious overcorrection where every
    successful auth reloads the page — which would yield an
    infinite reload loop on every fresh page load.
    """
    context, page, navs = _load_shim_page(_browser)
    try:
        page.evaluate("window.__fireOpen()")
        page.evaluate("window.__fireAuthOk()")
        page.wait_for_timeout(300)
        assert navs[0] == 0, (
            f"first auth_ok must NOT reload; saw {navs[0]} navigation(s)"
        )
        events = page.evaluate("window.__daemonStatusEvents")
        assert any(e.get("connected") is True for e in events), events
    finally:
        context.close()
