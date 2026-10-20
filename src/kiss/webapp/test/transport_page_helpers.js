// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// Pieces for booting chat.html on the REAL webapp transport
// (media/transport.js) over a fake WebSocket, so the frames the daemon
// would see are exactly the ones a suite inspects.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {newDom, waitFor} = require('./simplify2_harness');

const MEDIA = path.join(__dirname, '..', 'media');

/** Eval panelCopy.js, api.js and main.js into *win*, in that order. */
function loadPageScripts(win) {
  for (const f of ['panelCopy.js', 'api.js', 'main.js']) {
    win.eval(fs.readFileSync(path.join(MEDIA, f), 'utf8'));
  }
}

/** The source of media/transport.js (the webapp shim). */
function readShimJs() {
  return fs.readFileSync(path.join(MEDIA, 'transport.js'), 'utf8');
}

/**
 * The minimal page the shim needs outside chat.html: the loading overlay
 * with its message node, a hidden `#app`, and the password dialog.
 * Scripts run `outside-only`: the suite evals the shim itself.
 */
function newAuthDom(url) {
  const html = `<!DOCTYPE html><html><head></head><body>
    <div id="kiss-server-loading" role="status">
      <div id="kiss-server-loading-msg" class="kiss-server-loading-msg">KISS Sorcar Server is starting ...</div></div>
    <div id="app" style="display:none;">
      <div id="auth-modal" style="display:none;">
        <input id="auth-modal-input" type="password">
        <div id="auth-modal-error" role="alert"></div>
        <button id="auth-modal-ok"></button>
        <button id="auth-modal-cancel"></button>
      </div>
    </div></body></html>`;
  return new JSDOM(html, {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    url: url || 'https://example.test/',
  });
}

/**
 * Eval the shim into *win* and fire DOMContentLoaded: the shim defers
 * app-bound dispatches until the parser finishes so none are lost while
 * main.js is still being fetched, and jsdom keeps `document.readyState`
 * at 'loading' until a later macrotask.
 */
function bootShim(win) {
  win.__BRAND__ = {productName: 'KISS Sorcar', shortName: 'KISS'};
  win.eval(readShimJs() + '\n//# sourceURL=ws-shim.js');
  win.document.dispatchEvent(
    new win.Event('DOMContentLoaded', {bubbles: true}),
  );
}

/**
 * Replace *win*.WebSocket with a fake that records every socket it opens
 * into *sockets* and every frame sent on it into `socket.sent` (the raw
 * JSON strings; see `frames`).  A socket is born open, as the suites
 * that call `onopen()` directly expect; `fireOpen` / `fireMessage` /
 * `fireClose` play the server's side for the suites that drive the
 * handshake.
 *
 * @param {{connecting?: boolean, halfOpen?: boolean}} [opts]
 *   `connecting`: a socket is born CONNECTING and only `fireOpen` opens
 *   it, as the shim's reconnect logic sees a real socket.  `halfOpen`
 *   (implies `connecting`): `close()` only records `closed = true` and
 *   moves to CLOSING without ever firing `onclose`, so the shim's own
 *   stale-socket check is what must reconnect.
 */
function installFakeWebSocket(win, sockets, opts) {
  const {halfOpen = false, connecting = halfOpen} = opts || {};
  function FakeWebSocket(url) {
    this.url = url;
    this.readyState = connecting
      ? FakeWebSocket.CONNECTING
      : FakeWebSocket.OPEN;
    this.sent = [];
    this.closed = false;
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
    this.closed = true;
    if (halfOpen) {
      this.readyState = FakeWebSocket.CLOSING;
      return;
    }
    this.readyState = FakeWebSocket.CLOSED;
    if (typeof this.onclose === 'function') this.onclose();
  };
  FakeWebSocket.prototype.fireOpen = function () {
    this.readyState = FakeWebSocket.OPEN;
    if (typeof this.onopen === 'function') this.onopen();
  };
  FakeWebSocket.prototype.fireMessage = function (msg) {
    if (typeof this.onmessage === 'function')
      this.onmessage({data: JSON.stringify(msg)});
  };
  FakeWebSocket.prototype.fireClose = function () {
    this.readyState = FakeWebSocket.CLOSED;
    if (typeof this.onclose === 'function') this.onclose();
  };
  win.WebSocket = FakeWebSocket;
}

function parseFrame(data) {
  return JSON.parse(data);
}

/** The frames sent on a fake socket, parsed. */
function frames(sock) {
  return sock.sent.map(parseFrame);
}

/**
 * Wait until the fake WebSocket has opened *count* sockets.  The shim's
 * reconnect backoff is 0.5-1 s for the first attempt and doubles (with
 * jitter) on every further one within a page, so the socket a backoff
 * opens is awaited (up to 20 s) rather than a fixed delay.
 *
 * @param {Array<object>} sockets The ledger `installFakeWebSocket` fills.
 * @param {number} count How many sockets must have opened.
 */
async function waitForSocket(sockets, count) {
  await waitFor(
    () => sockets.length >= count,
    `socket ${count} never opened`,
    20000,
  );
}

/**
 * A chat.html page on the real shim with its handshake completed: the
 * fake socket is open and `auth_ok` has been received, so the commands
 * the app queued while connecting are flushed.  `drain()` parses every
 * frame sent so far into `posted` and returns it; the shim buffers
 * pre-auth frames, so call it each time the assertions look at `posted`.
 */
function makeWebappClient() {
  const win = newDom();
  const sockets = [];
  installFakeWebSocket(win, sockets);
  win.eval(readShimJs());
  assert.strictEqual(
    typeof win.acquireTransport,
    'function',
    'media/transport.js must define acquireTransport()',
  );
  loadPageScripts(win);
  const sock = sockets[0];
  assert.ok(sock, 'the webapp shim must open a WebSocket');
  sock.onopen();
  assert.strictEqual(
    parseFrame(sock.sent[0]).type,
    'auth',
    'the shim must authenticate before sending commands',
  );
  sock.onmessage({data: JSON.stringify({type: 'auth_ok'})});
  const posted = [];
  function drain() {
    while (posted.length < sock.sent.length) {
      posted.push(parseFrame(sock.sent[posted.length]));
    }
    return posted;
  }
  return {win, posted, drain, sock};
}

module.exports = {
  newDom,
  newAuthDom,
  bootShim,
  loadPageScripts,
  readShimJs,
  installFakeWebSocket,
  frames,
  waitForSocket,
  makeWebappClient,
};
