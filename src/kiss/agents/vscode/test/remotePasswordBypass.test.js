// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the REMOTE-PASSWORD BYPASS in the
// remote webapp shell (``_WS_SHIM_JS`` inside ``web_server.py``).
//
// Bug being locked in:
//
//   The remote webapp is served over the public cloudflared tunnel and
//   is protected by ``remote_password``.  The WebSocket handshake
//   enforces the password (``_authenticate_ws`` on the server, and the
//   shim only ``_ws.send``s commands once ``_authenticated`` is true).
//
//   BUT the password check was bypassable at the UI layer:
//
//     * On ``auth_required`` the shim dispatches
//       ``daemonStatus connected:true`` so ``media/main.js`` reveals
//       ``#app`` — necessary because the auth modal is a *child* of
//       ``#app`` and cannot render while its parent is display:none.
//     * The auth modal (``#auth-modal``) is a full-viewport backdrop
//       that blocks the revealed app WHILE IT IS OPEN.
//     * When the user CANCELS or presses ESCAPE, the modal hides
//       (display:none) — but ``#app`` stays revealed and the loading
//       overlay stays hidden.  The unauthenticated visitor now has the
//       full, interactive webapp in front of them without ever having
//       entered the remote password.
//
//   The fix: when the auth modal is dismissed without authenticating,
//   the shim re-gates the app by dispatching
//   ``daemonStatus connected:false`` (re-showing the loading overlay
//   over a re-hidden ``#app``).  A second, defense-in-depth fix drops
//   any server data frame that arrives while ``_authenticated`` is
//   false so an unauthenticated client can never act on backend data.
//
// This test drives the LIVE ``_WS_SHIM_JS`` extracted from
// ``web_server.py`` inside jsdom with a controllable fake WebSocket and
// a faithful copy of main.js's ``daemonStatus`` overlay contract.  The
// shim is eval'd with a ``//# sourceURL=ws-shim.js`` pragma so V8's
// built-in coverage attributes its execution; the companion
// ``remotePasswordBypass.coverage.js`` re-runs this file under
// ``NODE_V8_COVERAGE`` and enforces 100% line coverage of the shim.
//
// Run with:
//
//     node src/kiss/agents/vscode/test/remotePasswordBypass.test.js
//     node src/kiss/agents/vscode/test/remotePasswordBypass.coverage.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM, VirtualConsole} = require('jsdom');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const WEB_SERVER_PY = path.resolve(
  PROJECT_ROOT, '..', '..', 'server', 'web_server.py',
);

function readShimJs() {
  const src = fs.readFileSync(WEB_SERVER_PY, 'utf-8');
  const m = src.match(/_WS_SHIM_JS\s*=\s*r"""([\s\S]*?)"""/);
  assert.ok(m, 'could not locate _WS_SHIM_JS literal in web_server.py');
  return m[1];
}

// Evaluate the shim inside the jsdom window.  The sourceURL pragma
// names the eval'd script ``ws-shim.js`` in V8 coverage output so the
// coverage gate can find and measure it.
function evalShim(window, shimJs) {
  window.eval(shimJs + '\n//# sourceURL=ws-shim.js');
}

function ok(msg) {
  console.log('  ok -', msg);
}

function fail(msg, err) {
  console.error('  FAIL -', msg);
  if (err) console.error('       ', err.message || err);
  process.exit(1);
}

// A DOM that mirrors the elements the shim + daemonStatus contract
// touch: the loading overlay (with its message node), the hidden #app,
// and the auth modal (a child of #app, matching the chat.html
// template).  Options:
//   opaqueOrigin — no ``url``, so sessionStorage/localStorage access
//                  throws SecurityError (covers the storage catches).
//   noModal      — omit the auth modal AND the overlay message node
//                  (covers the prompt() fallback and the
//                  ``_updateLoadingMsg`` missing-node early return).
//   silent       — swallow jsdom "not implemented" noise (needed for
//                  the ``window.location.reload()`` reconnect path).
function buildDom(opts) {
  opts = opts || {};
  const modal = opts.noModal ? '' : `
      <div id="auth-modal" style="display:none;">
        <input id="auth-modal-input" type="password">
        <button id="auth-modal-ok"></button>
        <button id="auth-modal-cancel"></button>
      </div>`;
  const msgNode = opts.noModal
    ? ''
    : '<div id="kiss-server-loading-msg" class="kiss-server-loading-msg">' +
      'KISS Sorcar Server is starting ...</div>';
  const html = `<!DOCTYPE html><html><head></head><body>
    <div id="kiss-server-loading" role="status">${msgNode}</div>
    <div id="app" style="display:none;">${modal}
    </div>
  </body></html>`;
  const jsdomOpts = {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
  };
  if (!opts.opaqueOrigin) jsdomOpts.url = 'https://example.test/';
  if (opts.silent) jsdomOpts.virtualConsole = new VirtualConsole();
  return new JSDOM(html, jsdomOpts);
}

// Controllable fake WebSocket recording sends + driving lifecycle.
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

// Faithful copy of media/main.js's daemonStatus -> setServerLoading
// contract: a window 'message' event with data.type === 'daemonStatus'
// toggles the overlay and #app.
function wireOverlayContract(window) {
  function setServerLoading(loading) {
    const overlay = window.document.getElementById('kiss-server-loading');
    const app = window.document.getElementById('app');
    if (overlay) overlay.style.display = loading ? '' : 'none';
    if (app) app.style.display = loading ? 'none' : '';
  }
  window.addEventListener('message', (ev) => {
    const d = ev.data;
    if (d && d.type === 'daemonStatus') setServerLoading(!d.connected);
  });
}

function isVisible(el) {
  return el.style.display !== 'none';
}

// Wait one macrotask so the shim's ``_showAuthModal().then(...)``
// (a resolved Promise) settles before we assert.
function tick() {
  return new Promise((resolve) => setTimeout(resolve, 5));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function run() {
  const shimJs = readShimJs();

  // Sanity: the shim we are about to fix must actually enforce the
  // password on the SEND path (commands must queue while unauth).
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    const sock = sockets[0];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_required'});
    // Clear the auto-sent auth frame.
    sock.sent.length = 0;
    // An unauthenticated visitor issues a backend command.
    window.acquireVsCodeApi().postMessage({type: 'runTask', prompt: 'x'});
    try {
      assert.deepStrictEqual(
        sock.sent, [],
        'commands must NOT reach the socket before authentication',
      );
      ok('unauthenticated commands are queued, never sent to the socket');
    } catch (err) {
      fail('SEND-path password gate is broken', err);
    }
    window.close();
  }

  // ---------------------------------------------------------------
  // Core regression: CANCEL of the auth modal must NOT leave the
  // webapp exposed to an unauthenticated visitor.
  // ---------------------------------------------------------------
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    const sock = sockets[0];

    sock.fireOpen();
    sock.fireMessage({type: 'auth_required'});

    const overlay = window.document.getElementById('kiss-server-loading');
    const app = window.document.getElementById('app');
    const modal = window.document.getElementById('auth-modal');

    // While the modal is open, #app is revealed (so the modal, its
    // child, can render) and the modal's full-viewport backdrop blocks
    // interaction.  This matches the loading-overlay contract test.
    assert.ok(isVisible(app), 'app revealed while auth modal is open');
    assert.ok(isVisible(modal), 'auth modal is open on auth_required');
    ok('auth_required opens the modal over a revealed #app');

    // User dismisses the prompt (Cancel button).
    window.document.getElementById('auth-modal-cancel').click();
    await tick();

    try {
      assert.ok(
        !isVisible(app),
        'BYPASS: #app must be re-hidden after the password prompt is ' +
          'cancelled (unauthenticated visitor must not keep the app)',
      );
      assert.ok(
        isVisible(overlay),
        'the loading overlay must reappear after a cancelled password prompt',
      );
      assert.ok(!isVisible(modal), 'the modal is closed after cancel');
      ok('cancelling the password prompt re-gates the webapp');
    } catch (err) {
      fail('remote-password check is bypassed by cancelling the modal', err);
    }
    window.close();
  }

  // ---------------------------------------------------------------
  // ESCAPE key dismissal must re-gate identically to Cancel.
  // ---------------------------------------------------------------
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    const sock = sockets[0];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_required'});

    const app = window.document.getElementById('app');
    const overlay = window.document.getElementById('kiss-server-loading');
    const input = window.document.getElementById('auth-modal-input');

    const ev = new window.KeyboardEvent('keydown', {key: 'Escape', bubbles: true});
    input.dispatchEvent(ev);
    await tick();

    try {
      assert.ok(!isVisible(app), 'Escape must re-hide #app');
      assert.ok(isVisible(overlay), 'Escape must re-show the loading overlay');
      ok('pressing Escape on the password prompt re-gates the webapp');
    } catch (err) {
      fail('Escape-dismissing the modal bypasses the password check', err);
    }
    window.close();
  }

  // ---------------------------------------------------------------
  // Happy path: entering the correct password authenticates, reveals
  // the app for good, and flushes queued commands.
  // ---------------------------------------------------------------
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    const sock = sockets[0];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_required'});

    // Unauth visitor queues a command.
    window.acquireVsCodeApi().postMessage({type: 'runTask', prompt: 'hi'});

    // Enter the password and submit.
    const input = window.document.getElementById('auth-modal-input');
    input.value = 'hunter2';
    sock.sent.length = 0;
    window.document.getElementById('auth-modal-ok').click();
    await tick();

    try {
      assert.ok(
        sock.sent.some((d) => /"type":"auth"/.test(d) && /hunter2/.test(d)),
        'OK must send the typed password as an auth frame',
      );
      ok('submitting the modal sends the auth frame with the typed password');
    } catch (err) {
      fail('modal OK did not send the auth frame', err);
    }

    // Server accepts.
    sock.sent.length = 0;
    sock.fireMessage({type: 'auth_ok'});
    await tick();

    const app = window.document.getElementById('app');
    const overlay = window.document.getElementById('kiss-server-loading');
    try {
      assert.ok(isVisible(app), 'app stays revealed after auth_ok');
      assert.ok(!isVisible(overlay), 'overlay hidden after auth_ok');
      assert.ok(
        sock.sent.some((d) => /"type":"runTask"/.test(d)),
        'queued command must flush to the socket after auth_ok',
      );
      ok('auth_ok reveals the app and flushes the queued command');
    } catch (err) {
      fail('auth_ok did not complete the authenticated handshake', err);
    }
    window.close();
  }

  // ---------------------------------------------------------------
  // Defense in depth: a server data frame that somehow arrives while
  // unauthenticated must NOT be forwarded to the app.
  // ---------------------------------------------------------------
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);

    const forwarded = [];
    window.addEventListener('message', (ev) => {
      const d = ev.data;
      if (d && d.type && d.type !== 'daemonStatus') forwarded.push(d);
    });

    evalShim(window, shimJs);
    const sock = sockets[0];
    sock.fireOpen();
    // No auth_ok yet — deliver a data frame.
    sock.fireMessage({type: 'history', items: [{secret: 'leak'}]});
    await tick();
    try {
      assert.deepStrictEqual(
        forwarded, [],
        'data frames must not reach the app before authentication',
      );
      ok('pre-auth server data frames are dropped, not forwarded to the app');
    } catch (err) {
      fail('unauthenticated data frame was forwarded to the app', err);
    }

    // After auth_ok, the same data frame IS forwarded.
    sock.fireMessage({type: 'auth_ok'});
    await tick();
    sock.fireMessage({type: 'history', items: [{ok: 1}]});
    await tick();
    try {
      assert.ok(
        forwarded.some((d) => d.type === 'history'),
        'post-auth data frames must be forwarded to the app',
      );
      ok('post-auth server data frames are forwarded to the app');
    } catch (err) {
      fail('post-auth data frame was not forwarded', err);
    }
    window.close();
  }

  // ---------------------------------------------------------------
  // Full reconnect lifecycle: saved password, pinned work-dir replay,
  // state persistence, reconnect backoff + wake-up listeners, and the
  // reload-on-reauth path.  Together with the blocks above this
  // exercises every line of the shim (enforced by the coverage gate).
  // ---------------------------------------------------------------
  {
    const dom = buildDom({silent: true});
    const {window} = dom;
    // Simulate a page that reloaded mid-reconnect with a previously
    // saved password and persisted UI state.
    window.sessionStorage.setItem('sorcar-state', '{"foo":1}');
    window.sessionStorage.setItem('sorcar-reconnect-pending', '1');
    window.localStorage.setItem('sorcar-remote-pwd', 'savedpw');
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);

    const msgEl = window.document.getElementById('kiss-server-loading-msg');
    try {
      assert.strictEqual(
        msgEl.textContent, 'Reconnecting to KISS Sorcar Server ...',
        'surviving reconnect flag must relabel the overlay at load',
      );
      ok('reconnect flag from a prior page instance relabels the overlay');
    } catch (err) {
      fail('reconnect overlay label missing on load', err);
    }

    const api = window.acquireVsCodeApi();
    try {
      // JSON-compare: the state object comes from the jsdom realm, so
      // deepStrictEqual would reject its foreign Object.prototype.
      assert.strictEqual(JSON.stringify(api.getState()), '{"foo":1}');
      api.setState({bar: 2});
      assert.strictEqual(
        window.sessionStorage.getItem('sorcar-state'), '{"bar":2}');
      ok('getState/setState round-trip through sessionStorage');
    } catch (err) {
      fail('vscode-api state persistence broken', err);
    }

    // Pin a work dir while still unauthenticated: persisted AND queued.
    api.postMessage({type: 'setWorkDir', workDir: '/w'});
    assert.strictEqual(
      window.sessionStorage.getItem('sorcar-work-dir'), '/w');

    // Wake-up while the socket is still CONNECTING must not disturb it.
    const s0 = sockets[0];
    window.dispatchEvent(new window.Event('focus'));
    assert.strictEqual(sockets.length, 1, 'no reconnect while CONNECTING');

    s0.fireOpen();
    try {
      assert.ok(
        s0.sent.some((d) => /"type":"auth"/.test(d) && /savedpw/.test(d)),
        'saved password must be replayed on open',
      );
      ok('saved password from localStorage is replayed on connect');
    } catch (err) {
      fail('saved password not replayed', err);
    }

    s0.sent.length = 0;
    s0.fireMessage({type: 'auth_ok'});
    try {
      assert.strictEqual(
        window.sessionStorage.getItem('sorcar-reconnect-pending'), null,
        'auth_ok must clear the reconnect flag',
      );
      assert.strictEqual(s0.sent.length, 2, 'work-dir pin + queued frame');
      assert.ok(
        /"type":"setWorkDir"/.test(s0.sent[0]) && /\/w/.test(s0.sent[0]),
        'pinned work dir must be re-announced FIRST after auth_ok',
      );
      ok('auth_ok replays the pinned work dir before flushing the queue');
    } catch (err) {
      fail('work-dir replay on auth_ok broken', err);
    }

    // A pin with no workDir clears the persisted value (`|| ''`).
    api.postMessage({type: 'setWorkDir'});
    assert.strictEqual(
      window.sessionStorage.getItem('sorcar-work-dir'), '');

    // Wake-up listeners while OPEN are no-ops.
    window.dispatchEvent(new window.Event('pageshow'));
    window.dispatchEvent(new window.Event('online'));
    window.document.dispatchEvent(new window.Event('visibilitychange'));
    assert.strictEqual(sockets.length, 1, 'no reconnect while OPEN');

    // Server restart: authenticated socket drops.
    const overlay = window.document.getElementById('kiss-server-loading');
    s0.fireClose();
    try {
      assert.ok(isVisible(overlay), 'overlay re-shown on disconnect');
      assert.strictEqual(
        msgEl.textContent, 'Reconnecting to KISS Sorcar Server ...');
      assert.strictEqual(
        window.sessionStorage.getItem('sorcar-reconnect-pending'), '1',
        'authenticated disconnect persists the reconnect flag',
      );
      ok('authenticated disconnect re-gates the app and labels a reconnect');
    } catch (err) {
      fail('disconnect handling broken', err);
    }

    // A duplicate close must not schedule a second reconnect timer.
    s0.fireClose();
    assert.strictEqual(sockets.length, 1, 'no eager reconnect');

    // The backoff timer (250ms on the first attempt) fires -> reconnect.
    await sleep(400);
    assert.strictEqual(sockets.length, 2, 'backoff timer reconnects');
    const s1 = sockets[1];

    // Wake-up while the reconnect is in flight (CONNECTING) is a no-op.
    window.dispatchEvent(new window.Event('online'));
    assert.strictEqual(sockets.length, 2);

    // The reconnect attempt itself fails before authenticating ...
    s1.fireOpen();
    s1.fireClose();
    // Sabotage the dead socket and clearTimeout so the shim's
    // defensive catches (handler-nulling / close() in connect()'s
    // neutralisation, clearTimeout in _reconnectNowIfNeeded) are
    // exercised: hostile/broken environments must not break reconnect.
    Object.defineProperty(s1, 'onopen', {
      set() { throw new Error('handler nulling blocked'); },
      get() { return null; },
    });
    s1.close = function () { throw new Error('close blocked'); };
    window.clearTimeout = function () {
      throw new Error('clearTimeout blocked');
    };
    // ... and a wake-up event short-circuits the (longer) backoff.
    window.dispatchEvent(new window.Event('focus'));
    assert.strictEqual(sockets.length, 3, 'wake-up short-circuits backoff');
    ok('backoff timer and wake-up listeners drive reconnects');

    const s2 = sockets[2];
    s2.fireOpen();
    s2.onerror(); // the shim installs a no-op error handler
    s2.sent.length = 0;
    // Re-auth after a previously completed handshake must reload the
    // page (jsdom: "not implemented", swallowed) and NOT mark the
    // socket authenticated in the stale JS world.
    s2.fireMessage({type: 'auth_ok'});
    api.postMessage({type: 'runTask', prompt: 'stale'});
    try {
      assert.strictEqual(
        window.sessionStorage.getItem('sorcar-reconnect-pending'), '1',
        'reload path must keep the reconnect flag for the next page',
      );
      assert.ok(
        !s2.sent.some((d) => /"type":"runTask"/.test(d)),
        'stale page must not send commands after the reload-triggering auth_ok',
      );
      ok('re-auth after a real session triggers the reload path, not reuse');
    } catch (err) {
      fail('reload-on-reauth path broken', err);
    }
    window.close();
  }

  // ---------------------------------------------------------------
  // Isolation fallbacks: no modal nodes -> prompt(); no overlay msg
  // node -> label update is a no-op; opaque origin -> every
  // sessionStorage/localStorage access throws and is swallowed.
  // ---------------------------------------------------------------
  {
    const dom = buildDom({opaqueOrigin: true, noModal: true});
    const {window} = dom;
    window.prompt = function () { return 'promptpwd'; };
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    const sock = sockets[0];

    sock.fireOpen();
    try {
      assert.ok(
        sock.sent.some((d) => /"type":"auth"/.test(d) && /""/.test(d)),
        'unreadable localStorage must fall back to an empty password',
      );
      ok('SecurityError on localStorage falls back to empty password');
    } catch (err) {
      fail('opaque-origin open handshake broken', err);
    }

    sock.sent.length = 0;
    sock.fireMessage({type: 'auth_required'});
    await tick();
    try {
      assert.ok(
        sock.sent.some((d) => /"type":"auth"/.test(d) && /promptpwd/.test(d)),
        'missing modal nodes must fall back to prompt()',
      );
      ok('missing auth-modal nodes fall back to prompt()');
    } catch (err) {
      fail('prompt() fallback broken', err);
    }

    sock.sent.length = 0;
    sock.fireMessage({type: 'auth_ok'});
    const api = window.acquireVsCodeApi();
    // Unreadable sessionStorage at load left the persisted state null.
    assert.strictEqual(api.getState(), null, 'state starts null');
    api.postMessage({type: 'setWorkDir', workDir: '/x'});
    // A pin with no workDir falls back to '' (the `|| ''` operand).
    api.postMessage({type: 'setWorkDir'});
    api.setState({x: 1});
    try {
      assert.ok(
        sock.sent.some((d) => /"type":"setWorkDir"/.test(d)),
        'authenticated sends must survive throwing storage',
      );
      ok('throwing sessionStorage never breaks the message path');
    } catch (err) {
      fail('opaque-origin storage handling broken', err);
    }

    // Authenticated disconnect with no overlay message node.
    sock.fireClose();
    ok('disconnect with missing overlay message node is harmless');
    window.close();
  }

  // ---------------------------------------------------------------
  // Enter-key submit + submit racing a dead socket: the password is
  // saved for the next connect but nothing is sent on a closed socket.
  // Also: an unauthenticated disconnect keeps the cold-start label.
  // ---------------------------------------------------------------
  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    const sock = sockets[0];
    sock.fireOpen();
    // A throwing focus() (some embedded browsers) must not break the
    // modal: the shim's deferred-focus catch swallows it.
    const focusInput = window.document.getElementById('auth-modal-input');
    focusInput.focus = function () { throw new Error('focus blocked'); };
    sock.fireMessage({type: 'auth_required'});

    // The socket dies while the modal is open.
    sock.fireClose();
    const msgEl = window.document.getElementById('kiss-server-loading-msg');
    try {
      assert.strictEqual(
        msgEl.textContent, 'KISS Sorcar Server is starting ...',
        'never-authenticated disconnect keeps the cold-start label',
      );
      ok('unauthenticated disconnect keeps the cold-start overlay label');
    } catch (err) {
      fail('overlay label wrong on unauthenticated disconnect', err);
    }

    const input = window.document.getElementById('auth-modal-input');
    const modal = window.document.getElementById('auth-modal');
    input.value = 'pw2';
    sock.sent.length = 0;
    input.dispatchEvent(
      new window.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
    await tick();
    try {
      assert.ok(!isVisible(modal), 'Enter submits and closes the modal');
      assert.strictEqual(
        window.localStorage.getItem('sorcar-remote-pwd'), 'pw2',
        'Enter must save the typed password for the reconnect',
      );
      assert.deepStrictEqual(
        sock.sent, [],
        'nothing must be sent on a closed socket',
      );
      ok('Enter saves the password; a dead socket receives nothing');
    } catch (err) {
      fail('Enter-key submit path broken', err);
    }
    window.close();
  }

  console.log('\nAll remotePasswordBypass tests passed.');
}

run().catch((err) => fail('unexpected error', err));
