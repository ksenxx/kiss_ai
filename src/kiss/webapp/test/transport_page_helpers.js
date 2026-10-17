// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// Pieces for booting chat.html on the REAL webapp transport
// (media/transport.js) over a fake WebSocket, so the frames the daemon
// would see are exactly the ones a suite inspects.

const fs = require('fs');
const path = require('path');
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
 * Replace *win*.WebSocket with a fake that records every socket it opens
 * into *sockets* and every frame sent on it into `socket.sent`.
 */
function installFakeWebSocket(win, sockets) {
  function FakeWebSocket(url) {
    this.url = url;
    this.readyState = FakeWebSocket.OPEN;
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
  win.WebSocket = FakeWebSocket;
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

module.exports = {
  newDom,
  loadPageScripts,
  readShimJs,
  installFakeWebSocket,
  waitForSocket,
};
