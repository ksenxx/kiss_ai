// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const WEB_SERVER_PY = path.resolve(
  PROJECT_ROOT,
  '..', '..',
  'server',
  'web_server.py',
);
const CHAT_HTML = path.join(PROJECT_ROOT, 'media', 'chat.html');

function readShimJs() {
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

function setupSimulatedSetServerLoading(window) {
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

async function run() {
  const shimJs = readShimJs();

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

    assert.strictEqual(
      sockets.length,
      1,
      'shim must open exactly one WebSocket on load',
    );
    const sock = sockets[0];

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

    sock.fireOpen();
    assert.ok(
      sock.sent.some((d) => /"type":\s*"auth"/.test(d)),
      'shim must send an auth frame after WebSocket open',
    );

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

    window.close();
  }

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
    sock.fireMessage({type: 'auth_ok'});
    assert.ok(
      seen.some((d) => d.connected === true),
      'precondition: auth_ok must have produced a connected:true event',
    );

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

  {
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    setupSimulatedSetServerLoading(window);

    window.sessionStorage.setItem('sorcar-reconnect-pending', '1');

    window.eval(shimJs);

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

    assert.strictEqual(
      sockets.length,
      1,
      'shim must still open exactly one WebSocket on load',
    );

    window.close();
  }

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

    try {
      Object.defineProperty(window.document, 'visibilityState', {
        configurable: true,
        get() {
          return 'visible';
        },
      });
    } catch (e) {
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
