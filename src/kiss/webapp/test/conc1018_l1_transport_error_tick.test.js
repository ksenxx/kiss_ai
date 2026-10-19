// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// transport.js: a pre-auth ``error`` frame (e.g. ``remote_disabled``,
// "Remote access is turned off.") is the server's answer, so the outage
// that preceded it is over.  Before the fix the branch wrote the
// explanation onto the overlay but left the outage clock and the 1 s
// overlay tick running: once the outage had lasted longer than the
// escalation threshold the next tick replaced the text with "Still
// trying to reach the KISS Sorcar server (N s) ..." and put the
// "Retry now" button back, so the user never saw the real reason for
// longer than a second.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.resolve(__dirname, '..', 'media');
const shimJs = fs.readFileSync(path.join(MEDIA, 'transport.js'), 'utf-8');

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
  FakeWebSocket.prototype.fireMessage = function (msg) {
    this.onmessage({data: JSON.stringify(msg)});
  };
  FakeWebSocket.prototype.fireClose = function () {
    this.readyState = FakeWebSocket.CLOSED;
    if (typeof this.onclose === 'function') this.onclose();
  };
  window.WebSocket = FakeWebSocket;
}

// A page with fake timers, fake intervals and a fake clock.
function page() {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><head></head><body>' +
      '<div id="kiss-server-loading" role="status">' +
      '<div id="kiss-server-loading-msg"></div></div>' +
      '<div id="app" style="display:none;">' +
      '<div id="auth-modal" style="display:none;">' +
      '<input id="auth-modal-input" type="password">' +
      '<div id="auth-modal-error" role="alert"></div>' +
      '<button id="auth-modal-ok"></button>' +
      '<button id="auth-modal-cancel"></button></div>' +
      '</div></body></html>',
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
    timers.push({fn, ms});
    return timers.length;
  };
  window.clearTimeout = function () {};
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

function overlayText(window) {
  return window.document.getElementById('kiss-server-loading-msg').textContent;
}

function retryButton(window) {
  return window.document.getElementById('kiss-server-loading-action');
}

// Run every armed, not yet cleared overlay tick once.
function runTicks(intervals) {
  for (const t of intervals) if (!t.cleared) t.fn();
}

function main() {
  {
    // Outage long enough to escalate, then the server comes back and
    // refuses the handshake because remote access is off.
    const {window, sockets, timers, intervals, clock} = page();
    sockets[0].fireClose();
    assert.strictEqual(intervals.length, 1, 'the overlay tick is armed');
    clock.now += 9000;
    runTicks(intervals);
    assert.ok(
      overlayText(window).startsWith('Still trying'),
      'escalated after 8 s: ' + overlayText(window),
    );
    assert.ok(retryButton(window), 'Retry now shown while escalated');

    timers[timers.length - 1].fn();
    const s1 = sockets[1];
    s1.fireOpen();
    s1.fireMessage({
      type: 'error',
      code: 'remote_disabled',
      text: 'Remote access is turned off.',
    });
    assert.strictEqual(overlayText(window), 'Remote access is turned off.');
    assert.strictEqual(
      retryButton(window),
      null,
      'the stale Retry now button is removed with the outage text',
    );
    assert.ok(intervals[0].cleared, 'the overlay tick is stopped');

    // One second later: whatever is still armed must not touch the text.
    clock.now += 1000;
    runTicks(intervals);
    assert.strictEqual(
      overlayText(window),
      'Remote access is turned off.',
      "the server's explanation survives the next tick",
    );
    assert.strictEqual(retryButton(window), null, 'no Retry now button');

    // The server closes the socket; the quiet reconnect and a repeated
    // refusal keep showing the explanation.
    s1.fireClose();
    timers[timers.length - 1].fn();
    const s2 = sockets[2];
    s2.fireOpen();
    s2.fireMessage({
      type: 'error',
      code: 'remote_disabled',
      text: 'Remote access is turned off.',
    });
    clock.now += 5000;
    runTicks(intervals);
    assert.strictEqual(overlayText(window), 'Remote access is turned off.');
    assert.strictEqual(retryButton(window), null);
  }
  {
    // A wrong password while the dialog is open keeps reporting inside
    // the dialog; the overlay tick from the preceding outage is stopped
    // as well.
    const {window, sockets, timers, intervals, clock} = page();
    sockets[0].fireClose();
    clock.now += 9000;
    runTicks(intervals);
    timers[timers.length - 1].fn();
    const s1 = sockets[1];
    s1.fireOpen();
    s1.fireMessage({type: 'auth_required'});
    assert.ok(intervals[0].cleared, 'auth_required stops the tick');
    const input = window.document.getElementById('auth-modal-input');
    input.value = 'wrong';
    window.document.getElementById('auth-modal-ok').click();
    s1.fireMessage({
      type: 'error',
      code: 'auth_failed',
      text: 'Wrong password.',
    });
    assert.strictEqual(
      window.document.getElementById('auth-modal-error').textContent,
      'Wrong password.',
    );
    assert.ok(
      intervals.every(t => t.cleared),
      'no overlay tick is left running after the server answered',
    );
    assert.strictEqual(retryButton(window), null, 'no Retry now button');
  }
  console.log('conc1018_l1_transport_error_tick: ok');
}

main();
