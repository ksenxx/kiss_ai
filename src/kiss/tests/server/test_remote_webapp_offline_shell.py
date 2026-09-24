# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the remote webapp keeps working on a flaky connection.

The remote webapp must stay usable when the connection to the server is
slow, flaky or gone, and pick up where it was — without reloading the
page — once the server is reachable again.  Three pieces cooperate:

* ``/sw.js`` (``media/sw.js`` rendered by ``_build_service_worker``) is
  a service worker that precaches the app shell — the page plus every
  cache-busted ``/media`` asset it references — and serves it when the
  network fails or stalls;
* the WebSocket shim registers that worker and, when the socket drops
  after the page was authenticated, posts ``daemonStatus
  {connected:false, reconnecting:true}``;
* ``main.js`` then keeps ``#app`` on screen under a slim "Reconnecting
  ..." banner instead of the full-screen overlay, holds prompts back,
  and on the re-authenticated socket sends ``ready`` again so the
  server pushes what changed meanwhile into the page it kept.

The one reload left is for a page the worker served from its cache
while the server was down: its code may be older than the server's, so
it reloads once when the server is back.

The live test boots the production ``RemoteAccessServer`` + headless
Chromium and really stops and restarts the server (Playwright's
``set_offline`` does not reach service-worker fetches) to observe the
banner, the in-place resync on reconnection, and the page served from
the worker while the server is down.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import re
import socket
import ssl
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from kiss.core.brand import PRODUCT_NAME

MEDIA_URL_RE = re.compile(r"/media/[A-Za-z0-9_.-]+\?v=[0-9a-f]+")


def _free_port() -> int:
    """Return a TCP port that is free right now."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _LiveServer:
    """The production ``RemoteAccessServer`` on a fixed port, in a thread.

    ``start()`` / ``stop()`` may be called repeatedly so a test can take
    the server away from a connected page and bring it back on the same
    port, as a server restart or a network outage would.
    """

    def __init__(self, tmp_path: Path, port: int) -> None:
        from kiss.server.web_server import _generate_self_signed_cert

        self.tmp_path = tmp_path
        self.port = port
        self.certfile = tmp_path / "cert.pem"
        self.keyfile = tmp_path / "key.pem"
        _generate_self_signed_cert(self.certfile, self.keyfile)
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._error: BaseException | None = None
        # The running server and its event loop, for tests that drive a
        # server-side action (the keep-alive watchdog) on demand.
        self.server: Any = None
        self.loop: asyncio.AbstractEventLoop | None = None
        # While set, the server answers the next page (``/``) request
        # only after PAGE_STALL_S — a link so slow the worker must give
        # up on it.  One-shot: the request clears it, so the reload the
        # cached page then performs is answered at once.
        self.stall_page = threading.Event()

    PAGE_STALL_S = 7.0

    def _run(self, ready: threading.Event) -> None:
        from urllib.parse import urlsplit

        from kiss.server.web_server import RemoteAccessServer

        live = self

        class _StallablePageServer(RemoteAccessServer):
            async def _process_request(self, connection, request):  # type: ignore[override]
                if live.stall_page.is_set() and urlsplit(
                    request.path
                ).path in ("", "/"):
                    live.stall_page.clear()
                    await asyncio.sleep(live.PAGE_STALL_S)
                return await super()._process_request(connection, request)

        async def scenario() -> None:
            server = _StallablePageServer(
                host="127.0.0.1",
                port=self.port,
                work_dir=str(self.tmp_path),
                certfile=str(self.certfile),
                keyfile=str(self.keyfile),
                url_file=self.tmp_path / "remote-url.json",
                uds_path=self.tmp_path / "sorcar.sock",
            )
            started = False
            try:
                await server.start_async()
                started = True
                self.server = server
                self.loop = asyncio.get_running_loop()
                ready.set()
                while not self._done.is_set():
                    await asyncio.sleep(0.02)
            except BaseException as exc:  # pragma: no cover - defensive
                self._error = exc
                ready.set()
            finally:
                if started:
                    await server.stop_async()

        asyncio.run(scenario())

    def start(self) -> None:
        """Boot the server and block until it accepts connections."""
        self._done.clear()
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(ready,), daemon=True
        )
        self._thread.start()
        assert ready.wait(30), "RemoteAccessServer failed to start"
        if self._error is not None:
            raise AssertionError(
                "RemoteAccessServer startup failed"
            ) from self._error

    def ping_clients(self) -> None:
        """Run one round of the server's keep-alive watchdog now."""
        assert self.loop is not None and self.server is not None
        asyncio.run_coroutine_threadsafe(
            self.server._watchdog_ping_clients(), self.loop
        ).result(timeout=30)

    def stop(self) -> None:
        """Shut the server down and block until its port is released."""
        self._done.set()
        self.server = None
        self.loop = None
        if self._thread is not None:
            self._thread.join(timeout=30)
            assert not self._thread.is_alive(), "server failed to stop"
            self._thread = None
        if self._error is not None:
            raise AssertionError(
                "RemoteAccessServer thread failed"
            ) from self._error

    def get(self, path: str) -> tuple[int, dict[str, str], bytes]:
        """HTTPS GET *path*; returns (status, lower-cased headers, body)."""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        conn = http.client.HTTPSConnection(
            "127.0.0.1", self.port, timeout=10, context=ctx
        )
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            headers = {k.lower(): v for k, v in resp.getheaders()}
            return resp.status, headers, resp.read()
        finally:
            conn.close()


@pytest.fixture
def live_server(tmp_path: Path) -> Iterator[_LiveServer]:
    """A running ``_LiveServer``; stopped at teardown if still up."""
    server = _LiveServer(tmp_path, _free_port())
    server.start()
    yield server
    if server._thread is not None:
        server.stop()


@pytest.mark.timeout(120)
def test_sw_script_precaches_exactly_the_page_assets(
    live_server: _LiveServer,
) -> None:
    """``/sw.js`` is served as JavaScript with a manifest that lists the
    page (``/``) and exactly the cache-busted ``/media`` URLs the rendered
    page references — so the worker can serve every asset the app loads."""
    status, headers, body = live_server.get("/sw.js")
    assert status == 200
    assert headers["content-type"].startswith("text/javascript")
    script = body.decode("utf-8")
    assert "__KISS_SW_SHELL__" not in script, "placeholder left unrendered"
    manifest = re.search(r"const SHELL = (\{.*?\});", script)
    assert manifest, script[:400]
    shell = json.loads(manifest.group(1))
    assert re.fullmatch(r"[0-9a-f]{16}", shell["version"])
    assert shell["urls"][0] == "/"

    page_status, _, page_body = live_server.get("/")
    assert page_status == 200
    page_urls = set(MEDIA_URL_RE.findall(page_body.decode("utf-8")))
    assert page_urls, "the page references no /media assets?"
    # ... plus the plain /media/<name> files the brand skin (brand.css)
    # pulls in through relative url() outside comments, which the browser
    # requests un-hashed.
    _, _, brand_css = live_server.get("/media/brand.css")
    css_no_comments = re.sub(r"/\*.*?\*/", "", brand_css.decode(), flags=re.DOTALL)
    css_assets = {
        f"/media/{name}"
        for name in re.findall(r"""url\(\s*["']?([A-Za-z0-9_.-]+)["']?\s*\)""", css_no_comments)
    }
    assert set(shell["urls"][1:]) == page_urls | css_assets
    assert any(u.startswith("/media/main.js?v=") for u in page_urls)

    # Every manifest URL is servable, so the install-time addAll succeeds.
    for url in shell["urls"]:
        asset_status, _, _ = live_server.get(url)
        assert asset_status == 200, url

    # Identical assets -> identical script: the browser keeps its worker.
    _, _, again = live_server.get("/sw.js")
    assert again == body


_UI_STATE_JS = r"""
(() => {
  const overlay = document.getElementById('kiss-server-loading');
  const app = document.getElementById('app');
  const msg = document.getElementById('kiss-server-loading-msg');
  const inp = document.getElementById('task-input');
  return {
    overlayShown: !!overlay && overlay.style.display !== 'none',
    banner: !!overlay && overlay.classList.contains('kiss-server-loading--banner'),
    overlayRect: overlay ? overlay.getBoundingClientRect().toJSON() : null,
    appShown: !!app && app.style.display !== 'none',
    msg: msg ? msg.textContent : null,
    input: inp ? inp.value : null,
    controlled: !!(navigator.serviceWorker && navigator.serviceWorker.controller),
    marker: window.__offline_shell_marker || null,
    navType: (performance.getEntriesByType('navigation')[0] || {}).type || null,
    offlineMeta: !!document.querySelector('meta[name="kiss-offline-shell"]'),
    offlineReloaded: sessionStorage.getItem('sorcar-offline-reloaded'),
  };
})()
"""

_ACTIVE_TAB_ID_JS = "window._testApi.getActiveTabId()"

# Ids of the chat tabs in the tab bar.
_CHAT_TAB_IDS_JS = (
    "Array.from(document.querySelectorAll('.chat-tab[data-tab-id]'))"
    ".map(t => t.getAttribute('data-tab-id'))"
)

# Open a tab in the shared registry through the shim (as main.js would).
_OPEN_TAB_JS = (
    "window.acquireVsCodeApi().postMessage("
    "{type: 'openTab', tabId: '%s', title: '%s'})"
)

# A second, raw WebSocket to the server (no shim in between): records the
# type of every frame after completing the empty-password handshake.
_RAW_WS_JS = r"""
(() => {
  const ws = new WebSocket('wss://' + location.host + '/ws');
  const raw = {authed: false, frames: []};
  window.__rawWs = raw;
  ws.onopen = () => ws.send(JSON.stringify({type: 'auth', password: ''}));
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'auth_ok') raw.authed = true;
    raw.frames.push(msg.type);
  };
})()
"""

_CACHE_KEYS_JS = r"""
(async () => {
  const names = (await caches.keys()).filter(n => n.startsWith('kiss-shell-'));
  if (names.length !== 1) return {names};
  const cache = await caches.open(names[0]);
  const keys = (await cache.keys()).map(r => new URL(r.url).pathname + new URL(r.url).search);
  return {names, keys: keys.sort()};
})()
"""

# The worker registration's lifecycle, for the failure message when the
# worker never takes control of the page: which worker slots are filled
# and in what state at the timeout (``found``), and what a fresh
# ``register()`` yields (``retried``) or throws.
_SW_REGISTRATION_STATE_JS = r"""
(async () => {
  const slots = reg => (reg
    ? {installing: reg.installing && reg.installing.state,
       waiting: reg.waiting && reg.waiting.state,
       active: reg.active && reg.active.state}
    : null);
  const found = slots(await navigator.serviceWorker.getRegistration());
  try {
    const reg = await navigator.serviceWorker.register(
      '/sw.js', {updateViaCache: 'none'});
    return {found, retried: slots(reg)};
  } catch (e) {
    return {found, registerError: String(e)};
  }
})()
"""


# True on a page that is the server's own copy (not the worker's cached
# one), reached by a reload, with the app on screen.
_RELOADED_FROM_SERVER_JS = (
    "(performance.getEntriesByType('navigation')[0] || {}).type === 'reload'"
    " && !document.querySelector('meta[name=\"kiss-offline-shell\"]')"
    " && document.getElementById('app')"
    " && document.getElementById('app').style.display === ''"
)


def _wait_for(page: Page, predicate: str, timeout_ms: int = 30_000) -> None:
    """Block until the JS *predicate* expression is truthy on *page*."""
    page.wait_for_function(predicate, timeout=timeout_ms)


@pytest.mark.timeout(300)
def test_live_app_survives_outage_and_resyncs_on_reconnect(
    live_server: _LiveServer,
) -> None:
    """Live remote page across a server outage:

    1. the worker installs and precaches the whole shell;
    2. the socket drops -> ``#app`` stays visible under the reconnect
       banner, and a prompt sent meanwhile stays in the composer;
    3. the server returns -> the page is NOT reloaded (same document,
       same JS state, draft still in the composer), the banner goes,
       the command issued during the outage has reached the server,
       and the server's keep-alive round delivers a ``heartbeat``
       frame to an authenticated client;
    4. with the server down, a fresh navigation is answered by the
       worker (page and assets from cache, tagged as the offline shell)
       and the page shows the full "Reconnecting ..." overlay, having
       nothing to show yet;
    5. the server returns -> that page reloads itself and shows the
       server's own copy of the app.
    """
    url = f"https://127.0.0.1:{live_server.port}/"
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--ignore-certificate-errors"])
        try:
            context = browser.new_context(
                ignore_https_errors=True,
                viewport={"width": 1200, "height": 800},
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded")
            # 1. Connected: the app is on screen and the worker has
            #    precached the shell and taken control of this page.
            _wait_for(
                page,
                "document.getElementById('app').style.display === ''",
            )
            try:
                _wait_for(
                    page,
                    "navigator.serviceWorker && !!navigator.serviceWorker.controller",
                )
            except PlaywrightTimeoutError as exc:
                # Registration is best effort in the page (the shim
                # swallows failures), so name the worker's state: an
                # install aborted by a host network change (Docker
                # veth churn -> ERR_NETWORK_CHANGED) leaves the
                # registration without an active worker.
                state = page.evaluate(_SW_REGISTRATION_STATE_JS)
                raise AssertionError(
                    f"worker never took control of the page: {state!r}"
                ) from exc
            _, _, sw_body = live_server.get("/sw.js")
            manifest = re.search(r"const SHELL = (\{.*?\});", sw_body.decode())
            assert manifest, sw_body[:400]
            shell = json.loads(manifest.group(1))
            cached = page.evaluate(_CACHE_KEYS_JS)
            assert cached.get("keys") == sorted(shell["urls"]), cached
            page.evaluate("window.__offline_shell_marker = 'before-outage'")
            # A registered tab (the boot placeholder is not in the shared
            # registry and would not come back after a reload).
            page.evaluate(_OPEN_TAB_JS % ("draft-tab", "draft"))
            _wait_for(page, _ACTIVE_TAB_ID_JS + " === 'draft-tab'")
            page.fill("#task-input", "typed while offline")

            # 2. Outage: the socket closes, the app stays visible under
            #    the banner, and sending is held back.
            live_server.stop()
            _wait_for(
                page,
                "document.getElementById('kiss-server-loading')"
                ".classList.contains('kiss-server-loading--banner')",
            )
            page.click("#send-btn")
            page.wait_for_timeout(300)
            during = page.evaluate(_UI_STATE_JS)
            assert during["appShown"], during
            assert during["overlayShown"] and during["banner"], during
            assert during["msg"] == f"Reconnecting to {PRODUCT_NAME} Server ...", during
            assert during["input"] == "typed while offline", during
            assert during["marker"] == "before-outage", during
            rect = during["overlayRect"]
            assert rect["height"] < 80 and rect["width"] < 600, (
                "banner must be a slim pill, not the full-screen overlay; "
                + repr(rect)
            )
            # A command the page issues during the outage (here: a tab
            # opened in the shared registry) is queued and flushed on the
            # new connection.
            page.evaluate(_OPEN_TAB_JS % ("opened-during-outage", "outage"))

            # 3. Reconnection: the same document carries on; the banner
            #    goes and the server's state comes in over the socket.
            live_server.start()
            _wait_for(
                page,
                "!document.getElementById('kiss-server-loading')"
                ".classList.contains('kiss-server-loading--banner') && "
                "document.getElementById('kiss-server-loading')"
                ".style.display === 'none'",
                timeout_ms=60_000,
            )
            after = page.evaluate(_UI_STATE_JS)
            assert after["navType"] == "navigate", (
                "a reconnect must not reload the page; " + repr(after)
            )
            assert after["marker"] == "before-outage", (
                "JS state must survive the reconnect; " + repr(after)
            )
            assert after["appShown"] and not after["overlayShown"], after
            assert not after["offlineMeta"], after
            # The tab opened during the outage reached the server (the
            # resync's registry snapshot lists it) and the draft never
            # left the composer.
            _wait_for(
                page,
                _CHAT_TAB_IDS_JS + ".includes('opened-during-outage')",
            )
            assert page.evaluate(_ACTIVE_TAB_ID_JS) == "draft-tab"
            assert page.input_value("#task-input") == "typed while offline", (
                "the composer draft must survive the reconnect; "
                + repr(page.evaluate(_UI_STATE_JS))
            )
            # A second outage on the very same document: still no reload.
            live_server.stop()
            _wait_for(
                page,
                "document.getElementById('kiss-server-loading')"
                ".classList.contains('kiss-server-loading--banner')",
            )
            live_server.start()
            _wait_for(
                page,
                "document.getElementById('kiss-server-loading')"
                ".style.display === 'none'",
                timeout_ms=60_000,
            )
            again = page.evaluate(_UI_STATE_JS)
            assert again["navType"] == "navigate", again
            assert again["marker"] == "before-outage", again

            # 3a. Keep-alive: a raw authenticated client gets the app-level
            #     heartbeat the shim uses to detect half-open sockets.
            page.evaluate(_RAW_WS_JS)
            _wait_for(page, "window.__rawWs && window.__rawWs.authed")
            live_server.ping_clients()
            _wait_for(
                page,
                "window.__rawWs.frames.includes('heartbeat')",
                timeout_ms=10_000,
            )

            # 3b. Runtime media: an asset outside the manifest is fetched
            #     once and cached (served offline in step 4); a 404 is not.
            icon_url = "/media/kiss-icon.svg?v=runtime"
            missing_url = "/media/no-such-asset.js?v=1"
            fetched = page.evaluate(
                "(urls) => Promise.all(urls.map(u => fetch(u)"
                ".then(r => r.status).catch(e => String(e))))",
                [icon_url, missing_url],
            )
            assert fetched == [200, 404], fetched

            # 3c. Slow link: the server stalls the page past the worker's
            #     timeout, so the navigation is answered from the cache
            #     well before the server would have replied.
            live_server.stall_page.set()
            started = time.monotonic()
            # ``commit``: the clock stops when the cached response is
            # committed.  Waiting for ``domcontentloaded`` measured the
            # wrong navigation whenever the cached page's socket
            # authenticated before its parser finished: the page then
            # reloads itself (below) before DCL, ``goto`` follows the
            # reload, and Chromium queues that second request for the
            # same URL behind the still-stalled first one, so the
            # measured time was the full stall.
            response = page.goto(url, wait_until="commit")
            elapsed = time.monotonic() - started
            assert response is not None and response.from_service_worker
            assert elapsed < live_server.PAGE_STALL_S - 1, (
                f"stalled page must be served from cache, took {elapsed:.1f}s"
            )
            # The cached page (tagged as the offline shell, see step 4
            # for the tag itself) connects — the socket is fine — and,
            # being the cached copy, reloads itself once into the
            # server's copy.  That happens within milliseconds, so it is
            # observed through the reloaded page.
            _wait_for(page, _RELOADED_FROM_SERVER_JS, timeout_ms=60_000)
            fresh = page.evaluate(_UI_STATE_JS)
            assert fresh["offlineReloaded"] is None, (
                "a server-served page clears the reload-once guard; "
                + repr(fresh)
            )

            # 4. Server down again: a navigation is served by the worker.
            live_server.stop()
            response = page.goto(url, wait_until="domcontentloaded")
            assert response is not None and response.ok, response
            assert response.from_service_worker, (
                "with the server down the page must come from the worker"
            )
            page.wait_for_timeout(500)
            offline = page.evaluate(_UI_STATE_JS)
            assert offline["overlayShown"] and not offline["banner"], offline
            assert not offline["appShown"], offline
            assert offline["msg"] == f"Reconnecting to {PRODUCT_NAME} Server ...", offline
            assert offline["offlineMeta"], offline
            # The one reload an offline copy gets is recorded when it
            # happens, not when the copy is parsed.
            assert offline["offlineReloaded"] is None, offline
            main_js = next(
                u for u in shell["urls"] if u.startswith("/media/main.js?v=")
            )
            asset = page.evaluate(
                "(u) => fetch(u).then(r => ({ok: r.ok, status: r.status}))"
                ".catch(e => ({error: String(e)}))",
                main_js,
            )
            assert asset.get("ok"), asset
            runtime = page.evaluate(
                "(urls) => Promise.all(urls.map(u => fetch(u)"
                ".then(r => r.status).catch(e => 'error')))",
                [icon_url, missing_url],
            )
            assert runtime == [200, "error"], (
                "an asset fetched while online is served offline; a 404 "
                "was never cached: " + repr(runtime)
            )
            uncached = page.evaluate(
                "fetch('/api/jobs').then(r => ({ok: r.ok}))"
                ".catch(e => ({error: String(e)}))"
            )
            assert "error" in uncached, (
                "non-shell paths must not be answered from the cache; "
                + repr(uncached)
            )

            # 5. Server back: the offline-loaded page reloads itself into
            #    the server's copy of the app.
            live_server.start()
            _wait_for(page, _RELOADED_FROM_SERVER_JS, timeout_ms=60_000)
            final = page.evaluate(_UI_STATE_JS)
            assert final["appShown"] and not final["overlayShown"], final
            assert final["offlineReloaded"] is None, final
        finally:
            browser.close()
