// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// transport.js: the overlay's "Retry now" button must not throw away a
// reconnect attempt whose WebSocket handshake is still in flight.  The
// button is on screen after 8 s of outage, exactly when handshakes are
// slow, and connect() closes the current socket before opening a new
// one; a second click on a slow link used to restart the handshake
// every time, so none could ever finish.  The wake-up path
// (visibilitychange / online) already leaves a CONNECTING socket alone;
// the button now takes the same path.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.resolve(__dirname, '..', 'media');
const shimJs = fs.readFileSync(path.join(MEDIA, 'transport.js'), 'utf-8');

// The page's WebSocket: every instance stays CONNECTING until the test
// opens or closes it, and remembers whether the shim closed it.
function installFakeWebSocket(window, sockets) {
  function FakeWebSocket(url) {
    this.url = url;
    this.readyState = FakeWebSocket.CONNECTING;
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
    this.readyState = FakeWebSocket.CLOSED;
  };
  FakeWebSocket.prototype.fireOpen = function () {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen();
  };
  FakeWebSocket.prototype.fireClose = function () {
    this.readyState = FakeWebSocket.CLOSED;
    if (typeof this.onclose === 'function') this.onclose();
  };
  window.WebSocket = FakeWebSocket;
}

// A page with fake timers, fake intervals and a fake clock, so the
// outage can be aged past the escalation threshold without waiting.
function page() {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><head></head><body>' +
      '<div id="kiss-server-loading" role="status">' +
      '<div id="kiss-server-loading-msg"></div></div>' +
      '<div id="app" style="display:none;"></div></body></html>',
    {
      runScripts: 'outside-only',
      pretendToBeVisual: true,
      url: 'https://example.test/',
    },
  );
  const {window} = dom;
  const sockets = [];
  const timers = [];
  const intervals = [];
  const clock = {now: 1000000};
  installFakeWebSocket(window, sockets);
  window.setTimeout = function (fn, ms) {
    timers.push({fn, ms, cleared: false});
    return timers.length;
  };
  window.clearTimeout = function (id) {
    if (timers[id - 1]) timers[id - 1].cleared = true;
  };
  window.setInterval = function (fn, ms) {
    intervals.push({fn, ms, cleared: false});
    return intervals.length;
  };
  window.clearInterval = function (id) {
    if (intervals[id - 1]) intervals[id - 1].cleared = true;
  };
  window.Date.now = () => clock.now;
  window.__BRAND__ = {productName: 'KISS Sorcar', shortName: 'KISS'};
  window.eval(shimJs + '\n//# sourceURL=ws-shim.js');
  window.document.dispatchEvent(
    new window.Event('DOMContentLoaded', {bubbles: true}),
  );
  return {window, sockets, timers, intervals, clock};
}

function retryButton(window) {
  return window.document.getElementById('kiss-server-loading-action');
}

function runTicks(intervals) {
  for (const t of intervals) if (!t.cleared) t.fn();
}

function main() {
  const {window, sockets, timers, intervals, clock} = page();
  // The first socket opened and the session was authenticated, then
  // the server went away: an outage with a reconnect timer pending.
  sockets[0].fireOpen();
  sockets[0].onmessage({data: JSON.stringify({type: 'auth_ok'})});
  sockets[0].fireClose();
  const pending = timers.filter(t => !t.cleared && t.ms >= 500);
  assert.strictEqual(pending.length, 1, 'one reconnect timer is pending');
  clock.now += 9000;
  runTicks(intervals);
  const button = retryButton(window);
  assert.ok(button, 'Retry now shown after 8 s of outage');

  // First click: the backoff is dropped and a handshake starts.
  button.click();
  assert.ok(pending[0].cleared, 'the pending reconnect timer is cancelled');
  assert.strictEqual(sockets.length, 2, 'one socket opened by the click');
  const inFlight = sockets[1];
  assert.strictEqual(inFlight.readyState, window.WebSocket.CONNECTING);

  // Second click while that handshake is still running: nothing new.
  retryButton(window).click();
  retryButton(window).click();
  assert.strictEqual(
    sockets.length,
    2,
    'no socket is opened while the handshake is in flight',
  );
  assert.strictEqual(
    inFlight.closed,
    false,
    'the in-flight handshake is not thrown away',
  );

  // The handshake completes on the socket the first click opened.
  inFlight.fireOpen();
  inFlight.onmessage({data: JSON.stringify({type: 'auth_ok'})});
  assert.strictEqual(retryButton(window), null, 'auth_ok clears the button');

  // The button still works once the socket is closed again.
  inFlight.fireClose();
  clock.now += 9000;
  runTicks(intervals);
  retryButton(window).click();
  assert.strictEqual(sockets.length, 3, 'a closed socket is retried');
  assert.strictEqual(sockets[2].readyState, window.WebSocket.CONNECTING);
  console.log('transportRetryNowConnecting: ok');
}

main();
