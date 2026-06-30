// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the "KISS Sorcar Server is starting ..."
// overlay in the REMOTE WEBAPP (browser tab) shell — counterpart of
// ``serverLoadingOverlay.test.js`` which covers the VS Code extension
// host path.
//
// Bug being locked in:
//
//   The remote webapp served by ``web_server.py`` reuses the same
//   ``media/chat.html`` template as the VS Code webview.  That
//   template ships a ``#kiss-server-loading`` overlay over a hidden
//   ``#app`` so the user never sees a non-functional tab bar before
//   the backend is up.  ``media/main.js`` toggles those nodes in
//   response to a ``daemonStatus`` window message.
//
//   In the VS Code path the extension host (``SorcarSidebarView.ts``)
//   posts those ``daemonStatus`` messages.  In the WEBAPP path the
//   ``acquireVsCodeApi`` shim defined inside ``web_server.py``
//   (``_WS_SHIM_JS``) is the equivalent transport — but it was never
//   dispatching a ``daemonStatus`` event.  Result: the overlay was
//   the ONLY thing the user saw, forever, on every remote-webapp
//   load.
//
//   The fix synthesises ``daemonStatus`` events from the WebSocket
//   lifecycle:
//     * ``auth_ok`` → dispatch ``daemonStatus connected:true``  so
//       ``setServerLoading(false)`` reveals ``#app``.
//     * ``onclose``  → dispatch ``daemonStatus connected:false`` so
//       the overlay reappears while the socket is down.
//
// This test:
//   1. Extracts the live ``_WS_SHIM_JS`` from ``web_server.py``.
//   2. Mounts a minimal jsdom DOM with the overlay / app nodes and
//      the auth modal nodes the shim looks up.
//   3. Stubs ``WebSocket`` with a controllable in-memory fake so the
//      test can simulate ``onopen`` / ``onmessage`` / ``onclose``.
//   4. Runs the shim and asserts the ``daemonStatus`` MessageEvents
//      reach ``window``.
//
// Run with:
//
//     node src/kiss/agents/vscode/test/webappServerLoadingOverlay.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const WEB_SERVER_PY = path.resolve(
  PROJECT_ROOT,
  '..', '..',
  'agents',
  'vscode',
  'web_server.py',
);
const CHAT_HTML = path.join(PROJECT_ROOT, 'media', 'chat.html');

function readShimJs() {
  // The shim lives inside a raw Python triple-quoted string named
  // ``_WS_SHIM_JS``.  Match from the first ``r"""`` after the
  // identifier through the closing ``"""``.
  const src = fs.readFileSync(WEB_SERVER_PY, 'utf-8');
  const re = /_WS_SHIM_JS\s*=\s*r"""([\s\S]*?)"""/;
  const m = src.match(re);
  assert.ok(m, 'could not locate _WS_SHIM_JS literal in web_server.py');
  return m[1];
}

function ok(msg) {
  console.log('  ok -', msg);
}

function fail(msg, err) {
  console.error('  FAIL -', msg);
  if (err) console.error('       ', err.message || err);
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Build a minimal DOM that mirrors the elements the shim and the
// daemonStatus contract touch: the overlay, the hidden #app, and the
// auth modal nodes.
// ---------------------------------------------------------------------------

function buildDom() {
  const html = `<!DOCTYPE html><html><head></head><body>
    <div id="kiss-server-loading" role="status"><div id="kiss-server-loading-msg" class="kiss-server-loading-msg">KISS Sorcar Server is starting ...</div></div>
    <div id="app" style="display:none;"></div>
    <div id="auth-modal" style="display:none;">
      <input id="auth-modal-input" type="password">
      <button id="auth-modal-ok"></button>
      <button id="auth-modal-cancel"></button>
    </div>
  </body></html>`;
  return new JSDOM(html, {
    url: 'https://example.test/',
    runScripts: 'outside-only',
  });
}

// ---------------------------------------------------------------------------
// Controllable fake WebSocket that records sends and lets the test
// drive ``onopen`` / ``onmessage`` / ``onclose`` manually.
// ---------------------------------------------------------------------------

function installFakeWebSocket(window, sockets) {
  function FakeWebSocket(url) {
    this.url = url;
    this.readyState = 0; // CONNECTING
    this.sent = [];
    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;
    sockets.push(this);
  }
  FakeWebSocket.CONNECTING = 0;
  FakeWebSocket.OPEN = 1;
  FakeWebSocket.CLOSING = 2;
  FakeWebSocket.CLOSED = 3;
  FakeWebSocket.prototype.send = function (data) {
    this.sent.push(data);
  };
  FakeWebSocket.prototype.close = function () {
    this.readyState = FakeWebSocket.CLOSED;
    if (typeof this.onclose === 'function') this.onclose();
  };
  // Drive the open / message hooks the shim attaches.
  FakeWebSocket.prototype.fireOpen = function () {
    this.readyState = FakeWebSocket.OPEN;
    if (typeof this.onopen === 'function') this.onopen();
  };
  FakeWebSocket.prototype.fireMessage = function (msg) {
    if (typeof this.onmessage === 'function') {
      this.onmessage({data: JSON.stringify(msg)});
    }
  };
  FakeWebSocket.prototype.fireClose = function () {
    this.readyState = FakeWebSocket.CLOSED;
    if (typeof this.onclose === 'function') this.onclose();
  };
  window.WebSocket = FakeWebSocket;
  return FakeWebSocket;
}

function setupSimulatedSetServerLoading(window) {
  // Faithful copy of media/main.js ``setServerLoading`` + handleEvent
  // gate.  Pinning this here keeps the test self-contained while still
  // exercising the CONTRACT main.js depends on: a window ``message``
  // event with ``data.type === 'daemonStatus'`` toggles overlay /
  // #app.
  function setServerLoading(loading) {
    const overlay = window.document.getElementById('kiss-server-loading');
    const app = window.document.getElementById('app');
    if (overlay) overlay.style.display = loading ? '' : 'none';
    if (app) app.style.display = loading ? 'none' : '';
  }
  window.addEventListener('message', (ev) => {
    const d = ev.data;
    if (d && d.type === 'daemonStatus') {
      setServerLoading(!d.connected);
    }
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

async function run() {
  const shimJs = readShimJs();

  // Sanity: chat.html ships the overlay markup we depend on so the
  // python -> chat.html -> webapp pipeline cannot drift.
  const tpl = fs.readFileSync(CHAT_HTML, 'utf-8');
  try {
    assert.ok(
      /id="kiss-server-loading"/.test(tpl),
      'chat.html must render the loading overlay element',
    );
    assert.ok(
      /KISS Sorcar Server is starting \.\.\./.test(tpl),
      'overlay must contain the "KISS Sorcar Server is starting ..." message',
    );
    assert.ok(
      /<div id="app" style="display:none;?"/.test(tpl),
      '#app must start hidden so the overlay is what the user sees',
    );
    ok('chat.html paints overlay over a hidden #app on first load');
  } catch (err) {
    fail('chat.html initial overlay assertions', err);
  }

  // ---------------------------------------------------------------
  // Test: ``auth_ok`` triggers ``daemonStatus connected:true``.
  // This is the regression that locks in the visible bug fix.
  // ---------------------------------------------------------------
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    setupSimulatedSetServerLoading(window);

    const seen = [];
    window.addEventListener('message', (ev) => {
      if (ev.data && ev.data.type === 'daemonStatus') {
        seen.push(ev.data);
      }
    });

    // Run the shim inside the jsdom realm.
    window.eval(shimJs);

    // Shim's connect() ran synchronously and created exactly one socket.
    assert.strictEqual(
      sockets.length,
      1,
      'shim must open exactly one WebSocket on load',
    );
    const sock = sockets[0];

    // Before any messages arrive the overlay must still cover #app:
    // this is the BUG STATE the user reported.
    const overlay = window.document.getElementById('kiss-server-loading');
    const app = window.document.getElementById('app');
    assert.notStrictEqual(
      overlay.style.display,
      'none',
      'overlay must still be visible before auth_ok arrives',
    );
    assert.strictEqual(
      app.style.display,
      'none',
      '#app must still be hidden before auth_ok arrives',
    );
    ok('webapp loads with the "Server is starting" overlay covering #app');

    // Simulate the server handshake.
    sock.fireOpen();
    // The shim sends an ``{type:'auth'}`` first.
    assert.ok(
      sock.sent.some((d) => /"type":\s*"auth"/.test(d)),
      'shim must send an auth frame after WebSocket open',
    );

    // Server replies with ``auth_ok`` — this is the moment the
    // overlay MUST go away in the remote webapp.
    sock.fireMessage({type: 'auth_ok'});

    try {
      assert.ok(
        seen.some((d) => d.type === 'daemonStatus' && d.connected === true),
        'shim must dispatch daemonStatus(connected:true) after auth_ok',
      );
      ok('auth_ok dispatches daemonStatus(connected:true) — overlay hides');
    } catch (err) {
      fail(
        'auth_ok -> daemonStatus(connected:true): the remote webapp will ' +
          'stay on "KISS Sorcar Server is starting ..." forever',
        err,
      );
    }

    assert.strictEqual(
      overlay.style.display,
      'none',
      'overlay must be hidden once daemonStatus(connected:true) fires',
    );
    assert.notStrictEqual(
      app.style.display,
      'none',
      '#app must be revealed once daemonStatus(connected:true) fires',
    );
    ok('setServerLoading(false) reveals #app after auth_ok');

    // Close cleanly so nothing dangles into the next test.
    window.close();
  }

  // ---------------------------------------------------------------
  // Test: ``auth_required`` ALSO triggers
  // ``daemonStatus connected:true`` so the auth modal — which lives
  // inside ``#app`` — becomes visible.  Without this dispatch a
  // password-protected webapp shows the loading overlay forever and
  // the password prompt is hidden by its display:none parent.
  // ---------------------------------------------------------------
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    setupSimulatedSetServerLoading(window);

    const seen = [];
    window.addEventListener('message', (ev) => {
      if (ev.data && ev.data.type === 'daemonStatus') {
        seen.push(ev.data);
      }
    });

    window.eval(shimJs);
    const sock = sockets[0];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_required'});

    try {
      assert.ok(
        seen.some((d) => d.type === 'daemonStatus' && d.connected === true),
        'shim must dispatch daemonStatus(connected:true) on auth_required',
      );
      ok('auth_required dispatches daemonStatus(connected:true) — modal becomes visible');
    } catch (err) {
      fail(
        'auth_required -> daemonStatus(connected:true): password-protected ' +
          'webapp cannot show its password prompt',
        err,
      );
    }

    const overlay = window.document.getElementById('kiss-server-loading');
    const app = window.document.getElementById('app');
    assert.strictEqual(
      overlay.style.display,
      'none',
      'overlay must be hidden on auth_required so the auth modal can render',
    );
    assert.notStrictEqual(
      app.style.display,
      'none',
      '#app (parent of #auth-modal) must be revealed on auth_required',
    );
    ok('#app is revealed so #auth-modal can render on top');

    window.close();
  }

  // ---------------------------------------------------------------
  // Test: socket ``onclose`` triggers
  // ``daemonStatus connected:false`` so the overlay reappears.
  // ---------------------------------------------------------------
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    setupSimulatedSetServerLoading(window);

    const seen = [];
    window.addEventListener('message', (ev) => {
      if (ev.data && ev.data.type === 'daemonStatus') {
        seen.push(ev.data);
      }
    });

    window.eval(shimJs);
    const sock = sockets[0];

    // Authenticate so we have a known ``connected:true`` baseline.
    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok'});
    assert.ok(
      seen.some((d) => d.connected === true),
      'precondition: auth_ok must have produced a connected:true event',
    );

    // Now drop the socket.  The shim must announce that.
    sock.fireClose();

    try {
      assert.ok(
        seen.some((d) => d.type === 'daemonStatus' && d.connected === false),
        'shim must dispatch daemonStatus(connected:false) on socket close',
      );
      ok('socket close dispatches daemonStatus(connected:false) — overlay returns');
    } catch (err) {
      fail('close -> daemonStatus(connected:false)', err);
    }

    const overlay = window.document.getElementById('kiss-server-loading');
    const app = window.document.getElementById('app');
    assert.notStrictEqual(
      overlay.style.display,
      'none',
      'overlay must be visible again after socket close',
    );
    assert.strictEqual(
      app.style.display,
      'none',
      '#app must be re-hidden after socket close',
    );
    ok('setServerLoading(true) re-hides #app after disconnect');

    window.close();
  }

  // ---------------------------------------------------------------
  // Test: After a successful auth, a socket close must:
  //   (a) set the sessionStorage flag ``sorcar-reconnect-pending``
  //       so the upcoming ``location.reload()`` lands on a page that
  //       knows it is reconnecting, NOT cold-starting; and
  //   (b) flip the overlay message from
  //       "KISS Sorcar Server is starting ..." to
  //       "Reconnecting to KISS Sorcar Server ...".
  //
  // Bug being locked in: when an iPhone user backgrounded Safari
  // and returned, the overlay said "Server is starting ..." even
  // though we knew the server was already up and the only thing
  // happening was a WebSocket reconnect.  That made the UX feel
  // broken — the user thought a fresh boot was in progress.
  // ---------------------------------------------------------------
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    setupSimulatedSetServerLoading(window);

    window.eval(shimJs);
    const sock = sockets[0];

    // Establish a real authenticated session.
    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok'});

    // Drop it.
    sock.fireClose();

    try {
      assert.strictEqual(
        window.sessionStorage.getItem('sorcar-reconnect-pending'),
        '1',
        'onclose after auth must set sorcar-reconnect-pending=1',
      );
      ok('onclose latches sessionStorage["sorcar-reconnect-pending"]="1"');
    } catch (err) {
      fail('onclose must persist the reconnect-pending flag', err);
    }

    const msg = window.document.getElementById('kiss-server-loading-msg');
    try {
      assert.ok(msg, 'overlay must have a #kiss-server-loading-msg node');
      assert.strictEqual(
        msg.textContent,
        'Reconnecting to KISS Sorcar Server ...',
        'overlay text must say "Reconnecting ..." after a post-auth close',
      );
      ok('onclose flips overlay text to "Reconnecting to KISS Sorcar Server ..."');
    } catch (err) {
      fail(
        'overlay text must say "Reconnecting ..." after a post-auth close',
        err,
      );
    }

    window.close();
  }

  // ---------------------------------------------------------------
  // Test: A fresh page load whose sessionStorage already carries
  // ``sorcar-reconnect-pending=1`` (the state left behind by the
  // previous tab's ``location.reload()`` during an iOS Safari
  // app-switch round-trip) must show "Reconnecting ..." IMMEDIATELY
  // on script start — BEFORE any WebSocket message arrives.
  //
  // Without this the user briefly sees "KISS Sorcar Server is
  // starting ..." after every backgrounding, which is the visible
  // regression we are fixing.
  // ---------------------------------------------------------------
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    setupSimulatedSetServerLoading(window);

    // Pre-seed the flag the prior page would have set.
    window.sessionStorage.setItem('sorcar-reconnect-pending', '1');

    window.eval(shimJs);

    // No WebSocket events have been delivered yet — the shim must
    // have read the flag synchronously and updated the overlay.
    const msg = window.document.getElementById('kiss-server-loading-msg');
    try {
      assert.ok(msg, 'overlay must have a #kiss-server-loading-msg node');
      assert.strictEqual(
        msg.textContent,
        'Reconnecting to KISS Sorcar Server ...',
        'overlay must say "Reconnecting ..." on load when the flag is set',
      );
      ok('reload with pending flag shows "Reconnecting ..." immediately');
    } catch (err) {
      fail(
        'shim must update overlay text on script start when ' +
          'sessionStorage["sorcar-reconnect-pending"]=="1"',
        err,
      );
    }

    // Sanity: the shim still opens its WebSocket.
    assert.strictEqual(
      sockets.length,
      1,
      'shim must still open exactly one WebSocket on load',
    );

    window.close();
  }

  // ---------------------------------------------------------------
  // Test: Reconnect speed.  After a post-auth ``onclose`` the shim
  // schedules a reconnect; the first retry must fire fast (<= ~400ms,
  // i.e. the 250ms initial backoff plus jitter) — NOT the old 3000ms
  // fixed delay that made the wait feel sluggish on mobile Safari.
  // ---------------------------------------------------------------
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    setupSimulatedSetServerLoading(window);

    window.eval(shimJs);
    const sock = sockets[0];

    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok'});
    assert.strictEqual(sockets.length, 1, 'precondition: 1 socket so far');

    const t0 = Date.now();
    sock.fireClose();

    // Poll until a second socket is created, capturing the wall-clock
    // moment it appeared.  Old shim used a fixed setTimeout(connect,
    // 3000), so anything substantially below that proves the speedup.
    let openedAt = null;
    while (Date.now() - t0 < 1500) {
      if (sockets.length >= 2) {
        openedAt = Date.now();
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 10));
    }

    try {
      assert.ok(
        openedAt !== null,
        'shim must open a second WebSocket within 1500ms after onclose',
      );
      const dt = openedAt - t0;
      assert.ok(
        dt < 500,
        `reconnect must happen within ~400ms (initial 250ms backoff); took ${dt}ms`,
      );
      ok(
        `reconnect is fast: 2nd socket opened in ${dt}ms (was 3000ms before)`,
      );
    } catch (err) {
      fail('reconnect speed regression: backoff is too slow', err);
    }

    window.close();
  }

  // ---------------------------------------------------------------
  // Test: When iOS Safari un-backgrounds the tab, the shim must
  // reconnect IMMEDIATELY (via the visibilitychange wake-up
  // listener) instead of waiting out the backoff timer.  We simulate
  // this by:
  //   1. driving auth_ok and onclose to enter the "scheduled
  //      reconnect" state,
  //   2. immediately firing visibilitychange with state=visible,
  //   3. asserting a second socket appears WITHOUT waiting the
  //      250ms backoff.
  // ---------------------------------------------------------------
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    setupSimulatedSetServerLoading(window);

    window.eval(shimJs);
    const sock = sockets[0];

    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok'});
    sock.fireClose();
    assert.strictEqual(
      sockets.length,
      1,
      'precondition: still only one socket right after onclose',
    );

    // Force visibilityState='visible' (jsdom default is already
    // 'visible' but we redefine to be explicit) and dispatch the
    // event the shim listens for.
    try {
      Object.defineProperty(window.document, 'visibilityState', {
        configurable: true,
        get() {
          return 'visible';
        },
      });
    } catch (e) {
      // jsdom may already define this — ignore.
    }
    window.document.dispatchEvent(
      new window.Event('visibilitychange', {bubbles: false}),
    );

    try {
      assert.ok(
        sockets.length >= 2,
        'visibilitychange must trigger an immediate reconnect',
      );
      ok('visibilitychange wakes the shim and reconnects immediately');
    } catch (err) {
      fail(
        'visibilitychange did not trigger an immediate reconnect — ' +
          'iOS Safari users will wait out the full backoff',
        err,
      );
    }

    window.close();
  }

  console.log('\nAll webapp server-loading-overlay tests passed.');
}

run().catch((err) => {
  console.error('FAIL:', err);
  process.exit(1);
});
