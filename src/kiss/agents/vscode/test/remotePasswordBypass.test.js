// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

function evalShim(window, shimJs) {
  // chat.html defines the brand before the shim script; the overlay text
  // is built from it (the bare-window fallback is exercised separately).
  window.__BRAND__ = {productName: 'KISS Sorcar', shortName: 'KISS'};
  window.eval(shimJs + '\n//# sourceURL=ws-shim.js');
  // The shim defers app-bound dispatches until the parser finishes
  // (DOMContentLoaded) so none are lost while main.js is still being
  // fetched; jsdom keeps readyState 'loading' until a later macrotask,
  // so fire it now to switch the shim to direct dispatch.
  window.document.dispatchEvent(
    new window.Event('DOMContentLoaded', {bubbles: true}),
  );
}

function ok(msg) {
  console.log('  ok -', msg);
}

function fail(msg, err) {
  console.error('  FAIL -', msg);
  if (err) console.error('       ', err.message || err);
  process.exit(1);
}

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
  const head = opts.offlineShell
    ? '<meta name="kiss-offline-shell" content="1">'
    : '';
  const html = `<!DOCTYPE html><html><head>${head}</head><body>
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
  if (opts.reloads) {
    // jsdom cannot navigate: every window.location.reload() surfaces
    // as a "Not implemented: navigation" jsdomError, which is how the
    // tests count reload requests.
    const vc = new VirtualConsole();
    vc.on('jsdomError', (err) => {
      if (/navigation/.test(String(err && err.message))) opts.reloads.push(1);
    });
    jsdomOpts.virtualConsole = vc;
  }
  return new JSDOM(html, jsdomOpts);
}

function installFakeWebSocket(window, sockets) {
  function FakeWebSocket(url) {
    this.url = url;
    this.readyState = 0;
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

function tick() {
  return new Promise((resolve) => setTimeout(resolve, 5));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Frames dispatched into the page arrive as jsdom objects whose
// prototype differs from a literal's; compare them as plain data.
function plain(x) {
  return JSON.parse(JSON.stringify(x));
}

// The offline app-shell service worker is registered best-effort: a
// rejected registration (self-signed certificate) and a throwing one
// (no secure context) must both leave the shim running.
function installFakeServiceWorker(window, mode, calls) {
  Object.defineProperty(window.navigator, 'serviceWorker', {
    configurable: true,
    value: {
      register(url, opts) {
        calls.push({url, opts});
        if (mode === 'throw') throw new Error('SecurityError');
        return Promise.reject(new Error('SSL certificate error'));
      },
    },
  });
}

async function run() {
  const shimJs = readShimJs();

  for (const mode of ['reject', 'throw']) {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    const swCalls = [];
    installFakeWebSocket(window, sockets);
    installFakeServiceWorker(window, mode, swCalls);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    await tick();
    try {
      // (objects cross the jsdom realm boundary: compare fields, not
      // prototypes)
      assert.strictEqual(swCalls.length, 1, 'exactly one registration');
      assert.strictEqual(swCalls[0].url, '/sw.js', 'root-scope worker URL');
      assert.strictEqual(
        swCalls[0].opts.updateViaCache, 'none',
        'update checks must bypass the HTTP cache',
      );
      assert.strictEqual(
        sockets.length, 1,
        'a failed registration must not stop the shim from connecting',
      );
      ok(`service worker registration failure (${mode}) is tolerated`);
    } catch (err) {
      fail(`service worker registration (${mode})`, err);
    }
    window.close();
  }

  {
    // The post-auth drop is flagged `reconnecting` so main.js keeps the
    // app on screen; a drop before any auth_ok is not.
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    const statuses = [];
    installFakeWebSocket(window, sockets);
    window.addEventListener('message', (ev) => {
      if (ev.data && ev.data.type === 'daemonStatus') statuses.push(ev.data);
    });
    evalShim(window, shimJs);
    sockets[0].fireOpen();
    sockets[0].fireClose();
    sockets[0].fireOpen();
    sockets[0].fireMessage({type: 'auth_ok'});
    sockets[0].fireClose();
    try {
      assert.deepStrictEqual(
        statuses.map((s) => [s.connected, s.reconnecting]),
        [[false, false], [true, undefined], [false, true]],
        'reconnecting is true only for a drop after auth_ok',
      );
      ok('post-auth drop posts daemonStatus reconnecting:true');
    } catch (err) {
      fail('daemonStatus reconnecting flag', err);
    }
    window.close();
  }

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
    sock.sent.length = 0;
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

    assert.ok(isVisible(app), 'app revealed while auth modal is open');
    assert.ok(isVisible(modal), 'auth modal is open on auth_required');
    ok('auth_required opens the modal over a revealed #app');

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

    window.acquireVsCodeApi().postMessage({type: 'runTask', prompt: 'hi'});

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

  {
    const reloads = [];
    const dom = buildDom({reloads});
    const {window} = dom;
    const appFrames = [];
    const statusFrames = [];
    window.addEventListener('message', (ev) => {
      if (!ev.data) return;
      if (ev.data.type === 'daemonStatus') statusFrames.push(ev.data);
      else appFrames.push(ev.data);
    });
    window.sessionStorage.setItem('sorcar-state', '{"foo":1}');
    window.localStorage.setItem('sorcar-remote-pwd', 'savedpw');
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);

    const msgEl = window.document.getElementById('kiss-server-loading-msg');
    try {
      assert.strictEqual(
        msgEl.textContent, 'KISS Sorcar Server is starting ...',
        'a fresh page starts with the cold-start label',
      );
      ok('a fresh page shows the cold-start overlay label');
    } catch (err) {
      fail('cold-start overlay label wrong on load', err);
    }

    const api = window.acquireVsCodeApi();
    try {
      assert.strictEqual(JSON.stringify(api.getState()), '{"foo":1}');
      api.setState({bar: 2});
      assert.strictEqual(
        window.sessionStorage.getItem('sorcar-state'), '{"bar":2}');
      ok('getState/setState round-trip through sessionStorage');
    } catch (err) {
      fail('vscode-api state persistence broken', err);
    }

    api.postMessage({type: 'setWorkDir', workDir: '/w'});
    assert.strictEqual(
      window.sessionStorage.getItem('sorcar-work-dir'), '/w');

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
      assert.deepStrictEqual(
        s0.sent.map((d) => JSON.parse(d).type), ['setWorkDir', 'setWorkDir', 'ping'],
        'work-dir pin, the queued frame, then the probe',
      );
      assert.ok(
        /\/w/.test(s0.sent[0]),
        'pinned work dir must be re-announced FIRST after auth_ok',
      );
      ok('auth_ok replays the pinned work dir before flushing the queue');
    } catch (err) {
      fail('work-dir replay on auth_ok broken', err);
    }
    // The server took the boot batch: nothing is owed to a later socket.
    s0.fireMessage({type: 'pong'});

    api.postMessage({type: 'setWorkDir'});
    assert.strictEqual(
      window.sessionStorage.getItem('sorcar-work-dir'), '');

    window.dispatchEvent(new window.Event('pageshow'));
    window.dispatchEvent(new window.Event('online'));
    window.document.dispatchEvent(new window.Event('visibilitychange'));
    assert.strictEqual(sockets.length, 1, 'no reconnect while OPEN');

    const overlay = window.document.getElementById('kiss-server-loading');
    s0.fireClose();
    try {
      assert.ok(isVisible(overlay), 'overlay re-shown on disconnect');
      assert.strictEqual(
        msgEl.textContent, 'Reconnecting to KISS Sorcar Server ...');
      assert.deepStrictEqual(
        plain(statusFrames[statusFrames.length - 1]),
        {type: 'daemonStatus', connected: false, reconnecting: true},
        'the app is told the socket dropped after it was authenticated',
      );
      ok('authenticated disconnect re-gates the app and labels a reconnect');
    } catch (err) {
      fail('disconnect handling broken', err);
    }

    s0.fireClose();
    assert.strictEqual(sockets.length, 1, 'no eager reconnect');

    await sleep(400);
    assert.strictEqual(sockets.length, 2, 'backoff timer reconnects');
    const s1 = sockets[1];

    window.dispatchEvent(new window.Event('online'));
    assert.strictEqual(sockets.length, 2);

    s1.fireOpen();
    s1.fireClose();
    Object.defineProperty(s1, 'onopen', {
      set() { throw new Error('handler nulling blocked'); },
      get() { return null; },
    });
    s1.close = function () { throw new Error('close blocked'); };
    window.clearTimeout = function () {
      throw new Error('clearTimeout blocked');
    };
    window.dispatchEvent(new window.Event('focus'));
    assert.strictEqual(sockets.length, 3, 'wake-up short-circuits backoff');
    ok('backoff timer and wake-up listeners drive reconnects');

    const s2 = sockets[2];
    s2.fireOpen();
    s2.onerror();
    s2.sent.length = 0;
    // Posted while the connection was down (a settings save, say): it
    // goes out on the new connection, not into the void.
    api.postMessage({type: 'saveConfig', config: {edited: 'during the outage'}});
    statusFrames.length = 0;
    s2.fireMessage({type: 'auth_ok'});
    try {
      // The page keeps its state: no reload.  main.js hears
      // `connected: true` and re-sends `ready` itself, which is what
      // resyncs it with the server.  The flushed batch is followed by
      // a `ping` whose `pong` confirms the server took it.
      assert.strictEqual(reloads.length, 0, 'a re-auth never reloads the page');
      const types = s2.sent.map((d) => JSON.parse(d).type);
      assert.deepStrictEqual(
        types, ['saveConfig', 'ping'],
        'the outage-queued command is flushed, then the server is probed: ' + types,
      );
      assert.deepStrictEqual(
        plain(statusFrames), [{type: 'daemonStatus', connected: true}],
        'the app is told the socket is back, after the flush',
      );
      s2.fireMessage({type: 'pong'});
      assert.ok(!appFrames.some((d) => d.type === 'pong'), 'pong never reaches the app');
      api.postMessage({type: 'runTask', prompt: 'after re-auth'});
      assert.ok(
        s2.sent.some((d) => /after re-auth/.test(d)),
        'the page stays usable after the re-auth',
      );
      assert.ok(
        isVisible(window.document.getElementById('app')),
        'the app is revealed again after the re-auth',
      );
      s2.fireMessage({type: 'jobs', jobs: []});
      assert.ok(appFrames.some((d) => d.type === 'jobs'), 'server frames reach the app');
      // A later drop of this (still authenticated) socket is again a
      // reconnect, and its re-auth again keeps the page.
      statusFrames.length = 0;
      s2.fireClose();
      assert.deepStrictEqual(
        plain(statusFrames), [{type: 'daemonStatus', connected: false, reconnecting: true}],
      );
      await sleep(400);
      const s3 = sockets[3];
      s3.fireOpen();
      s3.fireMessage({type: 'auth_ok'});
      assert.strictEqual(reloads.length, 0, 'the next lost session does not reload either');
      assert.deepStrictEqual(
        s3.sent.map((d) => JSON.parse(d).type), ['auth'],
        'a confirmed batch is not sent again, and an empty flush is not probed',
      );
      assert.deepStrictEqual(
        plain(statusFrames[statusFrames.length - 1]), {type: 'daemonStatus', connected: true},
      );
      ok('re-auth after a lost session resyncs in place, never reloads');
    } catch (err) {
      fail('re-auth path broken', err);
    }
    window.close();
  }

  {
    // Mobile-Safari ordering: the authenticated socket is already dead
    // (CLOSING) but its queued ``onclose`` has not run yet when a wake-up
    // listener fires.  connect() must record "had a session and lost it"
    // itself, otherwise a replacement socket that fails would show the
    // cold-start overlay label for a server that was merely dropped.
    const reloads = [];
    const dom = buildDom({reloads});
    const {window} = dom;
    const statusFrames = [];
    window.addEventListener('message', (ev) => {
      if (ev.data && ev.data.type === 'daemonStatus') statusFrames.push(ev.data);
    });
    const sockets = [];
    const FakeWebSocket = installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    const msgEl = window.document.getElementById('kiss-server-loading-msg');
    const s0 = sockets[0];
    s0.fireOpen();
    s0.fireMessage({type: 'auth_ok'});
    s0.readyState = FakeWebSocket.CLOSING;
    statusFrames.length = 0;
    window.dispatchEvent(new window.Event('focus'));
    try {
      assert.strictEqual(sockets.length, 2, 'CLOSING socket is replaced');
      // The replacement itself tells the app the session dropped (the
      // dead socket's onclose was discarded): without this the app
      // would never re-send `ready` on the replacement's auth_ok.
      assert.deepStrictEqual(
        plain(statusFrames), [{type: 'daemonStatus', connected: false, reconnecting: true}],
        'replacing an authenticated socket reports the drop to the app',
      );
      assert.strictEqual(
        msgEl.textContent, 'Reconnecting to KISS Sorcar Server ...',
        'the latch taken by connect() labels the overlay',
      );
      const s1 = sockets[1];
      // The replacement dies before it authenticates: still a reconnect.
      statusFrames.length = 0;
      s1.fireClose();
      assert.deepStrictEqual(
        plain(statusFrames), [{type: 'daemonStatus', connected: false, reconnecting: true}],
        'the app keeps its banner mode after a wake-up replacement',
      );
      window.acquireVsCodeApi().postMessage({type: 'runTask', prompt: 'queued'});
      await sleep(400);
      const s2 = sockets[2];
      s2.fireOpen();
      s2.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(
        s2.sent.map((d) => JSON.parse(d).type), ['auth', 'runTask', 'ping'],
        'the command queued during the outage goes out on the new socket',
      );
      // The OS kills this socket too before the pong; the wake-up
      // replacement carries the unconfirmed batch over as well.
      s2.readyState = FakeWebSocket.CLOSING;
      window.dispatchEvent(new window.Event('focus'));
      const s3 = sockets[3];
      s3.fireOpen();
      s3.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(
        s3.sent.map((d) => JSON.parse(d).type), ['auth', 'runTask', 'ping'],
        'a batch unconfirmed when the wake-up replaced its socket is re-sent',
      );
      assert.strictEqual(reloads.length, 0, 'no reload after a wake-up replacement');
      ok('wake-up replacing a CLOSING authenticated socket keeps the page');
    } catch (err) {
      fail('CLOSING-socket replacement lost the had-auth latch', err);
    }
    window.close();
  }

  {
    // Also an offline-cached page here: with sessionStorage unreadable
    // the reload-once guard cannot be recorded, so no reload is risked.
    const reloads = [];
    const dom = buildDom({opaqueOrigin: true, noModal: true, offlineShell: true, reloads});
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
    assert.strictEqual(
      reloads.length, 0,
      'no offline reload when sessionStorage cannot record the guard',
    );
    const api = window.acquireVsCodeApi();
    assert.strictEqual(api.getState(), null, 'state starts null');
    api.postMessage({type: 'setWorkDir', workDir: '/x'});
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

    sock.fireClose();
    ok('disconnect with missing overlay message node is harmless');
    window.close();
  }

  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    const sock = sockets[0];
    sock.fireOpen();
    const focusInput = window.document.getElementById('auth-modal-input');
    focusInput.focus = function () { throw new Error('focus blocked'); };
    sock.fireMessage({type: 'auth_required'});

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

  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    const sock = sockets[0];
    sock.fireOpen();

    const overlay = window.document.getElementById('kiss-server-loading');
    const app = window.document.getElementById('app');
    const msgEl = window.document.getElementById('kiss-server-loading-msg');

    sock.fireMessage({type: 'auth_locked', retry_after: 1});
    try {
      assert.strictEqual(
        msgEl.textContent,
        'Too many failed login attempts. ' +
          'Asking for the password again in 1s ...',
        'auth_locked must explain the lockout on the overlay',
      );
      assert.ok(isVisible(overlay), 'overlay shown while locked');
      assert.ok(!isVisible(app), 'app stays gated while locked');
      ok('auth_locked relabels the overlay with the lockout explanation');
    } catch (err) {
      fail('auth_locked overlay explanation missing', err);
    }

    window.clearTimeout = function () {
      throw new Error('clearTimeout blocked');
    };
    sock.fireClose();
    try {
      assert.strictEqual(
        msgEl.textContent,
        'Too many failed login attempts. ' +
          'Asking for the password again in 1s ...',
        'the close after auth_locked must not overwrite the explanation',
      );
      assert.ok(isVisible(overlay), 'overlay stays shown after the close');
      ok('the lockout label survives the server closing the socket');
    } catch (err) {
      fail('lockout label overwritten by the close handler', err);
    }

    await sleep(400);
    try {
      assert.strictEqual(
        sockets.length, 1,
        'locked shim must NOT reconnect on the fast backoff',
      );
      ok('locked shim skips the fast reconnect backoff');
    } catch (err) {
      fail('locked shim reconnected too early', err);
    }
    await sleep(900);
    try {
      assert.strictEqual(
        sockets.length, 2,
        'shim must reconnect once retry_after has elapsed',
      );
      ok('shim reconnects after the lockout expires');
    } catch (err) {
      fail('post-lockout reconnect missing', err);
    }

    const s1 = sockets[1];
    s1.fireOpen();
    s1.fireMessage({type: 'auth_required'});
    const modal = window.document.getElementById('auth-modal');
    try {
      assert.ok(isVisible(modal), 'password modal opens after the lockout');
      ok('the password prompt appears on the post-lockout reconnect');
    } catch (err) {
      fail('no password prompt after the lockout expired', err);
    }
    window.close();
  }

  {
    const dom = buildDom({noModal: true});
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    const sock = sockets[0];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_locked'});
    sock.fireClose();
    await sleep(400);
    try {
      assert.strictEqual(
        sockets.length, 1,
        'invalid retry_after must default to 60s, not instant retry',
      );
      ok('missing retry_after defaults to a 60s wait; no msg node is fine');
    } catch (err) {
      fail('invalid retry_after handling broken', err);
    }
    window.close();
  }

  {
    // App-bound events produced while the document is still parsing
    // (before main.js's message listener exists) must be queued and
    // flushed at DOMContentLoaded, not lost.  The lost dispatch left
    // the "KISS Sorcar Server is starting ..." overlay up forever
    // whenever auth_ok beat the main.js fetch.
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    // Eval the shim WITHOUT the parser-finished simulation: jsdom
    // keeps readyState 'loading' here, exactly like a real page whose
    // later body scripts are still being fetched.
    window.eval(shimJs + '\n//# sourceURL=ws-shim.js');
    const sock = sockets[0];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok'});

    const overlay = window.document.getElementById('kiss-server-loading');
    const app = window.document.getElementById('app');
    try {
      assert.ok(
        isVisible(overlay),
        'pre-DOMContentLoaded daemonStatus must be queued, not dispatched',
      );
      window.document.dispatchEvent(
        new window.Event('DOMContentLoaded', {bubbles: true}),
      );
      assert.ok(
        !isVisible(overlay),
        'DOMContentLoaded must flush the queued daemonStatus (overlay off)',
      );
      assert.ok(isVisible(app), '#app revealed by the flushed daemonStatus');
      // After the flush the queue is retired: later events dispatch
      // directly (and a second DOMContentLoaded is a no-op).
      window.document.dispatchEvent(
        new window.Event('DOMContentLoaded', {bubbles: true}),
      );
      sock.fireClose();
      assert.ok(
        isVisible(overlay),
        'post-flush events must dispatch immediately (overlay re-shown)',
      );
      ok('parse-time events are queued and flushed at DOMContentLoaded');
    } catch (err) {
      fail('pre-parse event queueing broken', err);
    }
    window.close();
  }

  {
    // Commands posted across several outages go out in the order they
    // were posted on whichever connection first authenticates, stay in
    // flight until the server's pong confirms them (a connection that
    // dies first re-sends them), and a confirmed batch is never sent
    // again; the page itself is never reloaded along the way.
    const reloads = [];
    const dom = buildDom({reloads});
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    const api = window.acquireVsCodeApi();
    const types = (sock) => sock.sent.map((d) => JSON.parse(d).type);
    sockets[0].fireOpen();
    sockets[0].fireMessage({type: 'auth_ok'});
    sockets[0].fireClose();
    api.postMessage({type: 'saveConfig', config: {edited: 'first outage'}});
    await sleep(400);
    const s1 = sockets[1];
    s1.fireOpen();
    // Dies before it authenticates; the user keeps working meanwhile.
    s1.fireClose();
    api.postMessage({type: 'saveMyModel', name: 'second outage'});
    // The backoff grows with every failed attempt; a wake-up listener
    // short-circuits it.
    window.dispatchEvent(new window.Event('focus'));
    const s2 = sockets[2];
    s2.fireOpen();
    s2.fireMessage({type: 'auth_ok'});
    try {
      assert.deepStrictEqual(types(s1), ['auth'], 'nothing goes out before auth_ok');
      assert.deepStrictEqual(
        types(s2), ['auth', 'saveConfig', 'saveMyModel', 'ping'],
        'every command queued during the outages goes out in order, then the probe',
      );
      // Dies before the pong: the batch is owed to the next connection.
      s2.fireClose();
      window.dispatchEvent(new window.Event('focus'));
      const s3 = sockets[3];
      s3.fireOpen();
      s3.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(
        types(s3), ['auth', 'saveConfig', 'saveMyModel', 'ping'],
        'an unconfirmed batch is sent again',
      );
      s3.fireMessage({type: 'pong'});
      s3.fireClose();
      window.dispatchEvent(new window.Event('focus'));
      const s4 = sockets[4];
      s4.fireOpen();
      s4.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(types(s4), ['auth'], 'a confirmed batch is not sent again');
      assert.strictEqual(reloads.length, 0, 'no outage reloads the page');
      ok('outage-queued commands stay in flight until the pong, never reload');
    } catch (err) {
      fail('in-flight batch handling broken', err);
    }
    window.close();
  }

  {
    // A page the service worker served from its offline cache (marked
    // with <meta name="kiss-offline-shell">) reloads on its FIRST
    // auth_ok — its code may predate the server's — but only once per
    // browsing session: when the page fetch keeps timing out while the
    // WebSocket works, the second cached load must not loop.
    const reloads = [];
    const dom = buildDom({offlineShell: true, reloads});
    const {window} = dom;
    const statusFrames = [];
    window.addEventListener('message', (ev) => {
      if (ev.data && ev.data.type === 'daemonStatus') statusFrames.push(ev.data);
    });
    const sockets = [];
    installFakeWebSocket(window, sockets);
    wireOverlayContract(window);
    evalShim(window, shimJs);
    const msgEl = window.document.getElementById('kiss-server-loading-msg');
    try {
      // The worker's copy exists because the server was reachable
      // before: the overlay is a reconnect, before and after a failed
      // attempt.
      assert.strictEqual(msgEl.textContent, 'Reconnecting to KISS Sorcar Server ...');
      sockets[0].fireClose();
      assert.strictEqual(msgEl.textContent, 'Reconnecting to KISS Sorcar Server ...');
      ok('an offline-cached page labels its overlay as a reconnect');
    } catch (err) {
      fail('offline-cached page overlay label wrong', err);
    }
    window.dispatchEvent(new window.Event('focus'));
    statusFrames.length = 0;
    const s0 = sockets[1];
    s0.fireOpen();
    assert.strictEqual(
      window.sessionStorage.getItem('sorcar-offline-reloaded'), null,
      'the one reload is not used up before it happens (a page that ' +
        'never gets this far leaves it to the next one)',
    );
    window.acquireVsCodeApi().postMessage({type: 'ready'});
    s0.fireMessage({type: 'auth_ok'});
    try {
      assert.strictEqual(reloads.length, 1, 'offline-cached page reloads on first auth_ok');
      assert.strictEqual(
        window.sessionStorage.getItem('sorcar-offline-reloaded'), '1',
        'the reload is recorded for the next page instance',
      );
      assert.deepStrictEqual(
        s0.sent.map((d) => JSON.parse(d).type), ['auth'],
        'the boot queue is left to the fresh page, not flushed to a client about to go',
      );
      assert.deepStrictEqual(statusFrames, [], 'the app is not revealed on a page that is going away');
      ok('offline-cached page reloads on its first auth_ok');
    } catch (err) {
      fail('offline shell reload broken', err);
    }
    window.close();

    // Storage that fails at reload time (quota) cannot stop the reload.
    const domQ = buildDom({offlineShell: true, reloads});
    const wq = domQ.window;
    const socketsQ = [];
    installFakeWebSocket(wq, socketsQ);
    wireOverlayContract(wq);
    evalShim(wq, shimJs);
    socketsQ[0].fireOpen();
    Object.defineProperty(wq.Storage.prototype, 'setItem', {
      configurable: true,
      value() { throw new Error('QuotaExceededError'); },
    });
    socketsQ[0].fireMessage({type: 'auth_ok'});
    try {
      assert.strictEqual(reloads.length, 2, 'the reload happens even when the guard cannot be written');
      ok('an unwritable guard does not stop the offline-shell reload');
    } catch (err) {
      fail('offline shell reload with failing storage broken', err);
    }
    wq.close();

    // Second cached load in the same session: the guard holds, and the
    // page is a normal live page from here on (a later drop resyncs it
    // in place like any other).
    const dom2 = buildDom({offlineShell: true, reloads});
    const w2 = dom2.window;
    w2.sessionStorage.setItem('sorcar-offline-reloaded', '1');
    const sockets2 = [];
    installFakeWebSocket(w2, sockets2);
    wireOverlayContract(w2);
    evalShim(w2, shimJs);
    sockets2[0].fireOpen();
    sockets2[0].fireMessage({type: 'auth_ok'});
    try {
      assert.strictEqual(reloads.length, 2, 'no second reload: the cached page is kept');
      assert.ok(isVisible(w2.document.getElementById('app')), 'the kept page is revealed');
      assert.strictEqual(w2.sessionStorage.getItem('sorcar-offline-reloaded'), '1');
      sockets2[0].fireClose();
      await sleep(400);
      sockets2[1].fireOpen();
      sockets2[1].fireMessage({type: 'auth_ok'});
      assert.strictEqual(reloads.length, 2, 'a kept cached page reconnects without reloading');
      ok('a repeated offline-cached load does not reload again');
    } catch (err) {
      fail('offline shell reload loop guard broken', err);
    }
    w2.close();

    // A page the server itself served clears the guard.
    const dom3 = buildDom({reloads});
    const w3 = dom3.window;
    w3.sessionStorage.setItem('sorcar-offline-reloaded', '1');
    installFakeWebSocket(w3, []);
    evalShim(w3, shimJs);
    try {
      assert.strictEqual(
        w3.sessionStorage.getItem('sorcar-offline-reloaded'), null,
        'a server-served page clears the offline-reload guard',
      );
      ok('a server-served page clears the offline-reload guard');
    } catch (err) {
      fail('offline-reload guard not cleared', err);
    }
    w3.close();
  }

  {
    // Half-open socket: no onclose ever fires, so the shim's stale check
    // (every 15 s; the server sends a heartbeat frame every 15 s) must
    // drop a socket that has been silent for 45 s and reconnect.
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    const statuses = [];
    installFakeWebSocket(window, sockets);
    window.addEventListener('message', (ev) => {
      if (ev.data && ev.data.type === 'daemonStatus') statuses.push(ev.data);
    });
    const timers = [];
    window.setTimeout = function (fn, ms) {
      timers.push({fn, ms});
      return timers.length;
    };
    window.clearTimeout = function () {};
    let now = 1000000;
    window.Date.now = () => now;
    evalShim(window, shimJs);
    const s0 = sockets[0];
    s0.fireOpen();
    s0.fireMessage({type: 'auth_ok'});
    const appFrames = [];
    window.addEventListener('message', (ev) => {
      if (ev.data && ev.data.type === 'heartbeat') appFrames.push(ev.data);
    });
    try {
      const armed = timers.filter((t) => t.ms === 15000);
      assert.strictEqual(armed.length, 1, 'auth_ok arms the stale check');
      // Heartbeats keep the socket alive and never reach the app.
      now += 14000;
      s0.fireMessage({type: 'heartbeat'});
      assert.strictEqual(appFrames.length, 0, 'heartbeat frames are swallowed');
      now += 15000;
      armed[0].fn();
      assert.strictEqual(sockets.length, 1, 'a socket heard from recently is kept');
      const rearmed = timers.filter((t) => t.ms === 15000);
      assert.strictEqual(rearmed.length, 2, 'the stale check re-arms itself');
      // Silence for 45 s: the socket is half-open.
      now += 45000;
      s0.close = function () { this.closed = true; };
      rearmed[1].fn();
      assert.ok(s0.closed, 'the stale socket is closed');
      assert.strictEqual(s0.onclose, null, 'its late events are neutralised');
      const last = statuses[statuses.length - 1];
      assert.deepStrictEqual(
        [last.connected, last.reconnecting], [false, true],
        'the app is told the connection is down (reconnecting banner)',
      );
      const backoff = timers[timers.length - 1];
      assert.notStrictEqual(backoff.ms, 15000, 'a reconnect is scheduled');
      backoff.fn();
      assert.strictEqual(sockets.length, 2, 'the backoff opens a new socket');
      // A stale-check callback that outlives its socket is a no-op.
      const stale = timers.filter((t) => t.ms === 15000);
      stale[stale.length - 1].fn();
      assert.strictEqual(sockets.length, 2);
      ok('a half-open socket is dropped by the stale check and reconnected');
    } catch (err) {
      fail('stale-socket detection broken', err);
    }
    window.close();
  }

  console.log('\nAll remotePasswordBypass tests passed.');
}

run().catch((err) => fail('unexpected error', err));
