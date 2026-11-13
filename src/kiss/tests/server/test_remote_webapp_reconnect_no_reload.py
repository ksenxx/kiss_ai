# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""The remote webapp's WebSocket shim reconnects in place, never by reload.

The Playwright tests drive the production ``media/transport.js``
verbatim inside a synthetic page (no other media assets) served from a
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

The once-only test runs the shim against a real localhost WebSocket
server (:class:`_ScriptedDaemon`, speaking the daemon's handshake
frames) with the browser's own ``WebSocket`` and timers: a frame the
daemon must not receive twice (an Explorer copy) posted while the
socket is down is delivered once, after a ``pong`` confirmed the
reconnect's batch; one ``main.js`` gave up because the connection died
before that ``pong`` is not delivered at all, only the user's retry is.
"""


from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import sync_playwright
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from kiss.server.fs_actions import fs_action
from kiss.server.web_server import MEDIA_DIR, _http_response
from kiss.tests.conftest import goto_retrying_network_change

PAGE_URL = "https://shim.test/"
_TRANSPORT_JS = (MEDIA_DIR / "transport.js").read_text(encoding="utf-8")


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
{_TRANSPORT_JS}
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
    goto_retrying_network_change(page, PAGE_URL, wait_until="load")
    navs[0] = 0
    return context, page, navs


def _authenticate_current_socket(page) -> None:
    page.evaluate("window.__fireOpen()")
    page.evaluate("window.__fireAuthOk()")


def _real_shim_page() -> str:
    """The shim's page for a real WebSocket: the overlay nodes, a recorder
    of the ``daemonStatus`` events the shim posts to the app, the
    production shim and the app's handle on it (``window.__t``)."""
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>shim test</title></head>
<body>
  <div id="kiss-server-loading" role="status" aria-live="polite">
    <div id="kiss-server-loading-msg">KISS Sorcar Server is starting ...</div>
  </div>
  <div id="app" style="display:none;"></div>
  <script>
    window.__connected = [];
    window.addEventListener('message', function(e) {{
      var d = e && e.data;
      if (d && d.type === 'daemonStatus') window.__connected.push(d.connected);
    }});
  </script>
  <script>
{_TRANSPORT_JS}
  </script>
  <script>window.__t = window.acquireTransport();</script>
</body>
</html>
"""


class _ScriptedDaemon:
    """A localhost WebSocket server speaking the daemon's frames to the
    real shim, on a script: ``auth`` is answered with ``auth_ok``; the
    ``ping`` that ends a reconnect's batch is answered with ``pong`` on
    connection *pong_on* and, on every earlier connection, with the
    connection's close (the daemon died before confirming the batch);
    an ``fsAction`` is run through the daemon's own :func:`fs_action`.
    ``frames`` lists every frame received as ``(connection number,
    frame)``.  It serves the shim's page at ``/`` too, and the
    ``/api/daemon`` probe the shim makes on every ``auth_ok`` (one
    constant instance: the shim never reloads for it)."""

    def __init__(self, pong_on: int) -> None:
        self._pong_on = pong_on
        self.frames: list[tuple[int, dict[str, Any]]] = []
        self.results: list[dict[str, Any]] = []
        self.url = ""
        self._connections: dict[int, ServerConnection] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> _ScriptedDaemon:
        self._thread.start()
        assert self._ready.wait(10), "the scripted daemon failed to start"
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._loop is not None and self._stop is not None
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=30)
        assert not self._thread.is_alive(), "the scripted daemon failed to stop"

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(self._handle, "127.0.0.1", 0, process_request=self._http) as server:
            port = next(iter(server.sockets)).getsockname()[1]
            self.url = f"http://127.0.0.1:{port}/"
            self._ready.set()
            await self._stop.wait()

    def _http(self, connection: ServerConnection, request: Request) -> Response | None:
        if request.path == "/ws":
            return None
        if request.path == "/":
            return _http_response(200, "text/html; charset=utf-8", _real_shim_page().encode())
        if request.path == "/api/daemon":
            return _http_response(200, "application/json", b'{"instance": "scripted"}')
        return _http_response(404, "text/plain", b"")

    async def _handle(self, ws: ServerConnection) -> None:
        number = len(self._connections) + 1
        self._connections[number] = ws
        try:
            async for raw in ws:
                msg = json.loads(raw)
                self.frames.append((number, msg))
                if msg["type"] == "auth":
                    await ws.send(
                        json.dumps({"type": "auth_ok", "local": True, "echoesGeneration": True})
                    )
                elif msg["type"] == "ping":
                    if number < self._pong_on:
                        await ws.close()
                    else:
                        await ws.send(json.dumps({"type": "pong"}))
                elif msg["type"] == "fsAction":
                    self.results.append(
                        fs_action(
                            msg["action"],
                            msg["path"],
                            msg["dest"],
                            overwrite=msg["overwrite"],
                        )
                    )
        except ConnectionClosed:
            pass

    def close(self, number: int) -> None:
        """Close connection *number* from the server's side (the outage)."""
        assert self._loop is not None
        asyncio.run_coroutine_threadsafe(self._connections[number].close(), self._loop).result(10)

    def wait_for(self, number: int, frame_type: str, timeout: float = 30) -> None:
        """Block until connection *number* has delivered a *frame_type* frame."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(n == number and f["type"] == frame_type for n, f in self.frames):
                return
            time.sleep(0.05)
        raise AssertionError(f"no {frame_type} frame on connection {number}: {self.frames}")


@pytest.mark.timeout(120)
def test_fs_action_posted_while_disconnected_is_sent_once_after_the_pong(
    _browser,
    tmp_path: Path,
) -> None:
    """An Explorer copy posted while the socket is down goes out once,
    after a ``pong`` confirmed the reconnect's batch (W9 F1).  The batch
    is at-least-once: a connection lost before its ``pong`` re-sends it
    on the next, which for a replayable frame (``setModel``) means a
    repeated delivery and for a copy into the file's own folder would
    mean a second file (``notes copy 2.txt`` behind ``notes copy.txt``:
    the daemon names every such copy afresh).

    ``main.js`` gives a pending copy up each time it hears that the
    connection is gone ("the pending Explorer/Git action was not
    completed. Try again once it reconnects"), so the shim must not
    deliver it later (W4-011): the page pastes again at every drop, as
    that user does, connections 2 and 3 die before their ``pong``, and
    only the paste made after the last drop reaches the daemon; each of
    the two given up before it would have made one more copy.  The shim
    runs in Chromium with the browser's own ``WebSocket`` and timers
    against a real localhost server speaking the daemon's frames, which
    runs the copy it receives through the daemon's own ``fs_action``."""
    folder = tmp_path / "folder"
    folder.mkdir()
    src = folder / "notes.txt"
    src.write_text("kept once")
    with _ScriptedDaemon(pong_on=4) as daemon:
        context = _browser.new_context()
        page = context.new_page()
        try:
            goto_retrying_network_change(page, daemon.url, wait_until="load")
            page.wait_for_function("window.__connected.at(-1) === true", timeout=15000)
            page.evaluate(
                """args => {
                  let pastes = 0;
                  window.addEventListener('message', e => {
                    const d = e && e.data;
                    if (!d || d.type !== 'daemonStatus' || d.connected) return;
                    if (pastes === 0) {
                      window.__t.postMessage({type: 'setModel', model: 'replayable'});
                    }
                    pastes += 1;
                    window.__t.postMessage({
                      type: 'fsAction', action: 'copy', path: args.src, dest: args.dest,
                      name: '', query: '', overwrite: false, workDir: args.root,
                      tabId: 't1', token: 'fs-' + pastes,
                    });
                  });
                }""",
                {"src": str(src), "dest": str(folder), "root": str(tmp_path)},
            )
            daemon.close(1)
            daemon.wait_for(4, "fsAction")
            page.wait_for_function("window.__connected.at(-1) === true", timeout=15000)
        finally:
            context.close()
    by_type: dict[str, list[int]] = {}
    for number, frame in daemon.frames:
        by_type.setdefault(frame["type"], []).append(number)
    # The batch went out on every connection after the drop; the copy on
    # the confirmed one only, behind the batch's ping: the third paste,
    # the one the user still awaited.
    assert by_type["setModel"] == [2, 3, 4], daemon.frames
    assert by_type["fsAction"] == [4], daemon.frames
    assert by_type["ping"] == [2, 3, 4], daemon.frames
    copies = [frame["token"] for _, frame in daemon.frames if frame["type"] == "fsAction"]
    assert copies == ["fs-3"], daemon.frames
    kinds = [frame["type"] for number, frame in daemon.frames if number == 4]
    assert kinds.index("ping") < kinds.index("fsAction"), kinds
    assert daemon.results == [{"ok": True, "path": str(folder / "notes copy.txt")}]
    assert sorted(p.name for p in folder.iterdir()) == ["notes copy.txt", "notes.txt"]
    assert (folder / "notes copy.txt").read_text() == "kept once"


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
            "window.acquireTransport().postMessage("
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
