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
const CHAT_HTML = path.join(PROJECT_ROOT, 'media', 'chat.html');
const TRANSPORT_JS = path.join(PROJECT_ROOT, 'media', 'transport.js');

function readShimJs() {
  return fs.readFileSync(TRANSPORT_JS, 'utf-8');
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
      /\{\{PRODUCT_NAME\}\} Server is starting \.\.\./.test(tpl),
      'overlay must contain the "<product> Server is starting ..." message',
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
    // The shim defers app-bound dispatches until the parser finishes
    // (DOMContentLoaded) so none are lost while main.js is still being
    // fetched; jsdom keeps readyState 'loading' until a later macrotask,
    // so fire it now to switch the shim to direct dispatch.
    window.document.dispatchEvent(
      new window.Event('DOMContentLoaded', {bubbles: true}),
    );

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
    // The shim defers app-bound dispatches until the parser finishes
    // (DOMContentLoaded) so none are lost while main.js is still being
    // fetched; jsdom keeps readyState 'loading' until a later macrotask,
    // so fire it now to switch the shim to direct dispatch.
    window.document.dispatchEvent(
      new window.Event('DOMContentLoaded', {bubbles: true}),
    );
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
    // The shim defers app-bound dispatches until the parser finishes
    // (DOMContentLoaded) so none are lost while main.js is still being
    // fetched; jsdom keeps readyState 'loading' until a later macrotask,
    // so fire it now to switch the shim to direct dispatch.
    window.document.dispatchEvent(
      new window.Event('DOMContentLoaded', {bubbles: true}),
    );
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
    // The shim defers app-bound dispatches until the parser finishes
    // (DOMContentLoaded) so none are lost while main.js is still being
    // fetched; jsdom keeps readyState 'loading' until a later macrotask,
    // so fire it now to switch the shim to direct dispatch.
    window.document.dispatchEvent(
      new window.Event('DOMContentLoaded', {bubbles: true}),
    );
    const sock = sockets[0];

    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok'});

    sock.fireClose();

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
    // A page the service worker served from its cache (the server was
    // unreachable) is a reconnect from the user's point of view: the
    // worker only has the copy because the server was up before.
    const dom = buildDom();
    const {window} = dom;
    const sockets = [];
    installFakeWebSocket(window, sockets);
    setupSimulatedSetServerLoading(window);
    const meta = window.document.createElement('meta');
    meta.setAttribute('name', 'kiss-offline-shell');
    window.document.head.appendChild(meta);

    window.eval(shimJs);
    // The shim defers app-bound dispatches until the parser finishes
    // (DOMContentLoaded) so none are lost while main.js is still being
    // fetched; jsdom keeps readyState 'loading' until a later macrotask,
    // so fire it now to switch the shim to direct dispatch.
    window.document.dispatchEvent(
      new window.Event('DOMContentLoaded', {bubbles: true}),
    );

    const msg = window.document.getElementById('kiss-server-loading-msg');
    try {
      assert.ok(msg, 'overlay must have a #kiss-server-loading-msg node');
      assert.strictEqual(
        msg.textContent,
        'Reconnecting to KISS Sorcar Server ...',
        'overlay must say "Reconnecting ..." on a page served from the worker cache',
      );
      ok('offline-cached page shows "Reconnecting ..." immediately');
    } catch (err) {
      fail(
        'shim must label the overlay a reconnect on an offline-shell page',
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
    // The shim defers app-bound dispatches until the parser finishes
    // (DOMContentLoaded) so none are lost while main.js is still being
    // fetched; jsdom keeps readyState 'loading' until a later macrotask,
    // so fire it now to switch the shim to direct dispatch.
    window.document.dispatchEvent(
      new window.Event('DOMContentLoaded', {bubbles: true}),
    );
    const sock = sockets[0];

    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok'});
    assert.strictEqual(sockets.length, 1, 'precondition: 1 socket so far');

    sock.fireClose();

    // transport.js reconnects after onclose: _scheduleReconnect arms
    // the first backoff at 500-1000 ms (jittered), and the wake-up
    // listeners (pageshow, which jsdom fires right after load, and
    // visibilitychange / online) reconnect at once.  Poll for the
    // second socket with a deadline that covers the full backoff.
    const deadline = Date.now() + 1500;
    while (sockets.length < 2 && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }

    try {
      assert.strictEqual(
        sockets.length,
        2,
        'shim must open a second WebSocket within 1500ms after onclose',
      );
      ok('onclose reconnects: a 2nd socket is opened (was 3000ms before)');
    } catch (err) {
      fail('reconnect regression: no 2nd socket after onclose', err);
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
    // The shim defers app-bound dispatches until the parser finishes
    // (DOMContentLoaded) so none are lost while main.js is still being
    // fetched; jsdom keeps readyState 'loading' until a later macrotask,
    // so fire it now to switch the shim to direct dispatch.
    window.document.dispatchEvent(
      new window.Event('DOMContentLoaded', {bubbles: true}),
    );
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
