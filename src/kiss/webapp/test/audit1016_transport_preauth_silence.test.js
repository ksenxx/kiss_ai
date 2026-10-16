// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// transport.js: a socket that reaches OPEN but never gets an auth answer
// (the network path died right after the upgrade, or the server's
// auth_ok was lost) must be dropped by the half-open check and
// reconnected, exactly like a socket that goes silent after auth_ok.
// Before the fix the stale check was armed only on auth_ok, so a cold
// page load whose handshake never completed sat on "starting ..." for
// ever: no timer, no reconnect, and no "Retry now" button either (that
// one needs an onclose to have set the outage clock).

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
    this.readyState = 0;
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
  // A half-open socket: close() never yields an onclose.
  FakeWebSocket.prototype.close = function () {
    this.closed = true;
  };
  FakeWebSocket.prototype.fireOpen = function () {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen();
  };
  FakeWebSocket.prototype.fireMessage = function (msg) {
    this.onmessage({data: JSON.stringify(msg)});
  };
  window.WebSocket = FakeWebSocket;
}

// A page with fake timers and a fake clock: `timers` collects every
// setTimeout the shim arms, `clock.now` is what Date.now() returns.
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
    {runScripts: 'outside-only', pretendToBeVisual: true,
     url: 'https://example.test/'},
  );
  const {window} = dom;
  const sockets = [];
  const timers = [];
  const statuses = [];
  const clock = {now: 1000000};
  installFakeWebSocket(window, sockets);
  window.setTimeout = function (fn, ms) {
    timers.push({fn, ms});
    return timers.length;
  };
  window.clearTimeout = function () {};
  window.Date.now = () => clock.now;
  window.addEventListener('message', ev => {
    if (ev.data && ev.data.type === 'daemonStatus') statuses.push(ev.data);
  });
  window.__BRAND__ = {productName: 'KISS Sorcar', shortName: 'KISS'};
  window.eval(shimJs + '\n//# sourceURL=ws-shim.js');
  window.document.dispatchEvent(
    new window.Event('DOMContentLoaded', {bubbles: true}),
  );
  return {window, sockets, timers, statuses, clock};
}

function staleChecks(timers) {
  return timers.filter(t => t.ms === 15000);
}

// Run every stale check armed so far once, in order, as the clock
// would (each may arm a successor, which runs on the next call).
function runStaleChecks(timers) {
  const due = staleChecks(timers).filter(t => !t.ran);
  for (const t of due) {
    t.ran = true;
    t.fn();
  }
}

function main() {
  {
    // Cold load: open, then silence.  No auth answer ever arrives.
    const {window, sockets, timers, statuses, clock} = page();
    const s0 = sockets[0];
    s0.fireOpen();
    assert.strictEqual(
      staleChecks(timers).length, 1,
      'the half-open check is armed as soon as the socket opens',
    );
    clock.now += 15000;
    runStaleChecks(timers);
    clock.now += 15000;
    runStaleChecks(timers);
    assert.ok(!s0.closed, 'a socket silent for 30 s is still waited on');
    clock.now += 15000;
    runStaleChecks(timers);
    assert.ok(s0.closed, 'a socket silent for 45 s after open is dropped');
    assert.strictEqual(s0.onclose, null, 'its late events are neutralised');
    const last = statuses[statuses.length - 1];
    assert.deepStrictEqual(
      [last.connected, last.reconnecting], [false, false],
      'the app is told the daemon is not connected (pre-auth drop)',
    );
    const backoff = timers[timers.length - 1];
    assert.ok(
      backoff.ms >= 500 && backoff.ms < 1000,
      'a reconnect is scheduled with the first backoff delay: ' + backoff.ms,
    );
    backoff.fn();
    assert.strictEqual(sockets.length, 2, 'the backoff opens a new socket');
    // The replacement socket opens and completes the handshake: the
    // check armed at its open keeps running as the post-auth check.
    const s1 = sockets[1];
    s1.fireOpen();
    s1.fireMessage({type: 'auth_ok'});
    clock.now += 45000;
    runStaleChecks(timers);
    assert.ok(s1.closed, 'the post-auth half-open check still works');
    console.log('  ok - a socket silent since open is dropped and reconnected');
    window.close();
  }

  {
    // A slow but live handshake is not cut: the server's heartbeat
    // frames (sent to every connection, authenticated or not) count as
    // life before auth_ok exactly as after it.
    const {window, sockets, timers, clock} = page();
    const s0 = sockets[0];
    s0.fireOpen();
    for (let i = 0; i < 4; i++) {
      clock.now += 15000;
      s0.fireMessage({type: 'heartbeat'});
      runStaleChecks(timers);
    }
    assert.ok(!s0.closed, 'heartbeats keep a pre-auth socket alive');
    assert.strictEqual(sockets.length, 1, 'no reconnect while it is alive');
    s0.fireMessage({type: 'auth_ok'});
    assert.strictEqual(
      staleChecks(timers).filter(t => !t.ran).length, 1,
      'auth_ok leaves exactly one pending stale check (no duplicate chain)',
    );
    console.log('  ok - a slow handshake kept alive by heartbeats is not cut');
    window.close();
  }

  {
    // The dialog case: auth_required opened the password prompt and the
    // server then falls silent (it drops an idle unauthenticated socket
    // after 60 s anyway).  The client drop must take the quiet path:
    // no status churn, the prompt stays, a reconnect is scheduled.
    const {window, sockets, timers, statuses, clock} = page();
    const s0 = sockets[0];
    s0.fireOpen();
    s0.fireMessage({type: 'auth_required'});
    const before = statuses.length;
    clock.now += 45000;
    runStaleChecks(timers);
    assert.ok(s0.closed, 'the silent socket behind the prompt is dropped');
    assert.strictEqual(
      statuses.length, before,
      'no daemonStatus churn while the password prompt is up',
    );
    const backoff = timers[timers.length - 1];
    assert.notStrictEqual(backoff.ms, 15000, 'a reconnect is scheduled');
    backoff.fn();
    assert.strictEqual(sockets.length, 2, 'the prompt gets a fresh socket');
    console.log('  ok - a silent socket behind the password prompt is replaced quietly');
    window.close();
  }
  console.log('audit1016_transport_preauth_silence: all passed');
}

main();
