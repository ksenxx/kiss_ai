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
const {
  MEDIA,
  newDom,
  waitFor,
  loadPageScripts,
  settle,
} = require('./simplify2_harness');

/** The source of media/transport.js (the webapp shim). */
function readShimJs() {
  return fs.readFileSync(path.join(MEDIA, 'transport.js'), 'utf8');
}

/**
 * The minimal page the shim needs outside chat.html: the loading overlay
 * with its message node, a hidden `#app`, and the password dialog.
 * Scripts run `outside-only`: the suite evals the shim itself.
 *
 * @param {string} [url] The page URL (default `https://example.test/`).
 * @param {{virtualConsole?: object}} [opts] Extra JSDOM options, e.g. a
 *   `VirtualConsole` that observes the navigations jsdom cannot do.
 */
function newAuthDom(url, opts) {
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
    ...(opts || {}),
  });
}

/** Print a passed check of a transport suite. */
function ok(msg) {
  console.log('  ok -', msg);
}

/** Print a failed check of a transport suite and exit 1. */
function fail(msg, err) {
  console.error('  FAIL -', msg);
  if (err) console.error('       ', err.message || err);
  process.exit(1);
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

/** The frames sent on a fake socket, parsed. */
function frames(sock) {
  return sock.sent.map(data => JSON.parse(data));
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
    JSON.parse(sock.sent[0]).type,
    'auth',
    'the shim must authenticate before sending commands',
  );
  sock.onmessage({data: JSON.stringify({type: 'auth_ok'})});
  // The daemon answers the boot batch's ``ping`` with ``pong``
  // (sorcar.py), and transport.js holds later frames until it does.
  assert.strictEqual(JSON.parse(sock.sent[sock.sent.length - 1]).type, 'ping');
  sock.onmessage({data: JSON.stringify({type: 'pong'})});
  const posted = [];
  function drain() {
    while (posted.length < sock.sent.length) {
      posted.push(JSON.parse(sock.sent[posted.length]));
    }
    return posted;
  }
  return {win, posted, drain, sock};
}

/**
 * Replace *win*'s `setTimeout` / `clearTimeout` with a recorder, so the
 * shim's reconnect backoff (0.5-1 s, doubling) never sleeps for real:
 * the suite fires the recorded timer by hand (`openNextSocket`).
 * Install BEFORE `bootShim`.  Returns the ledger, one
 * `{fn, ms, cleared, ran}` per armed timer in arming order; a timer's
 * id is its 1-based index.
 *
 * With `{intervals, clock}` the shim's `setInterval` / `clearInterval`
 * are recorded the same way into `intervals` and `Date.now()` reads
 * `clock.now`, so a suite ages an outage by moving the clock and running
 * the recorded ticks.
 *
 * @param {object} win The jsdom window.
 * @param {{intervals?: Array<object>, clock?: {now: number}}} [opts]
 */
function installFakeTimers(win, opts) {
  const timers = [];
  win.setTimeout = function (fn, ms) {
    timers.push({fn, ms, cleared: false, ran: false});
    return timers.length;
  };
  win.clearTimeout = function (id) {
    if (timers[id - 1]) timers[id - 1].cleared = true;
  };
  const {intervals, clock} = opts || {};
  if (intervals) {
    win.setInterval = function (fn, ms) {
      intervals.push({fn, ms, cleared: false});
      return intervals.length;
    };
    win.clearInterval = function (id) {
      if (intervals[id - 1]) intervals[id - 1].cleared = true;
    };
  }
  if (clock) win.Date.now = () => clock.now;
  return timers;
}

/** The recorded timers that are neither cleared nor fired yet. */
function pendingTimers(timers) {
  return timers.filter(t => !t.cleared && !t.ran);
}

/** Fire a recorded timer, once. */
function fireTimer(timer) {
  timer.ran = true;
  timer.fn();
}

/**
 * Fire the one pending reconnect timer (the shim arms it at >= 500 ms;
 * the dialog's focus timer is 0 ms), so socket *n + 1* opens on the
 * fake WebSocket; return that socket.
 */
function openNextSocket(sockets, timers, n) {
  const pending = pendingTimers(timers).filter(t => t.ms >= 500);
  assert.strictEqual(pending.length, 1, 'one reconnect timer is pending');
  fireTimer(pending[0]);
  assert.strictEqual(sockets.length, n + 2, 'the timer opened the next socket');
  return sockets[n + 1];
}

module.exports = {
  newDom,
  newAuthDom,
  ok,
  fail,
  bootShim,
  loadPageScripts,
  readShimJs,
  installFakeWebSocket,
  frames,
  waitForSocket,
  makeWebappClient,
  installFakeTimers,
  pendingTimers,
  fireTimer,
  openNextSocket,
  settle,
};
