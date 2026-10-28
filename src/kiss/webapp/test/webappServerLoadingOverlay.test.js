// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// transport.js and the "<product> Server is starting ..." overlay: the
// shim dispatches `daemonStatus` on auth_ok / auth_required (connected)
// and on a socket close (disconnected), reconnects after a close (at
// once on visibilitychange), and labels the overlay "Reconnecting ..."
// after a post-auth close or on a page the service worker served from
// its cache.  What main.js does with `daemonStatus` (hide the overlay,
// show #app) is reconnectBannerKeepsAppVisible.test.js.

'use strict';

const assert = require('assert');
const {
  newAuthDom,
  bootShim,
  installFakeWebSocket,
  ok,
} = require('./transport_page_helpers');
const {captureTimers} = require('./simplify2_harness');

const RECONNECTING = 'Reconnecting to KISS Sorcar Server ...';

/** A booted shim page: its fake sockets and every daemonStatus it dispatched. */
function boot(prepare) {
  const {window} = newAuthDom();
  const sockets = [];
  installFakeWebSocket(window, sockets);
  const seen = [];
  window.addEventListener('message', ev => {
    if (ev.data && ev.data.type === 'daemonStatus') seen.push(ev.data);
  });
  if (prepare) prepare(window);
  bootShim(window);
  assert.strictEqual(sockets.length, 1, 'the shim opens one WebSocket on load');
  return {window, sockets, sock: sockets[0], seen};
}

function overlayText(window) {
  return window.document.getElementById('kiss-server-loading-msg').textContent;
}

async function run() {
  {
    const {window, sock, seen} = boot();
    assert.strictEqual(seen.length, 0, 'nothing is dispatched before auth');
    sock.fireOpen();
    assert.ok(
      sock.sent.some(d => /"type":\s*"auth"/.test(d)),
      'the shim sends an auth frame after the socket opens',
    );
    sock.fireMessage({type: 'auth_ok'});
    assert.ok(
      seen.some(d => d.connected === true),
      'auth_ok must dispatch daemonStatus(connected:true), or the remote ' +
        'webapp stays on "Server is starting ..." forever',
    );
    ok('auth_ok dispatches daemonStatus(connected:true)');
    sock.fireClose();
    assert.ok(
      seen.some(d => d.connected === false),
      'a socket close must dispatch daemonStatus(connected:false)',
    );
    assert.strictEqual(
      overlayText(window),
      RECONNECTING,
      'the overlay says "Reconnecting ..." after a post-auth close',
    );
    ok(
      'socket close dispatches daemonStatus(connected:false) and relabels the overlay',
    );
    window.close();
  }

  {
    const {window, sock, seen} = boot();
    sock.fireOpen();
    sock.fireMessage({type: 'auth_required'});
    assert.ok(
      seen.some(d => d.connected === true),
      'auth_required must dispatch daemonStatus(connected:true), or a ' +
        'password-protected webapp cannot show its password prompt',
    );
    ok('auth_required dispatches daemonStatus(connected:true)');
    window.close();
  }

  {
    // transport.js reconnects after onclose: _scheduleReconnect arms
    // the first backoff at 500-1000 ms (jittered).  The timer is
    // recorded and fired by hand rather than waited for.
    let timers;
    const {window, sockets, sock} = boot(window => {
      timers = captureTimers(window);
    });
    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok'});
    timers.length = 0;
    sock.fireClose();
    const backoff = timers.filter(t => t.ms >= 500 && t.ms < 1000);
    assert.strictEqual(backoff.length, 1, 'onclose arms one 0.5-1 s backoff');
    assert.strictEqual(sockets.length, 1, 'no socket before the backoff fires');
    backoff[0].cb();
    assert.strictEqual(sockets.length, 2, 'the backoff opens a 2nd socket');
    ok('onclose reconnects: a 2nd socket is opened');
    window.close();
  }

  {
    const {window, sockets, sock} = boot();
    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok'});
    sock.fireClose();
    assert.strictEqual(
      sockets.length,
      1,
      'only one socket right after onclose',
    );
    Object.defineProperty(window.document, 'visibilityState', {
      configurable: true,
      get() {
        return 'visible';
      },
    });
    window.document.dispatchEvent(new window.Event('visibilitychange'));
    assert.ok(
      sockets.length >= 2,
      'visibilitychange must reconnect at once, or iOS Safari users wait ' +
        'out the full backoff',
    );
    ok('visibilitychange wakes the shim and reconnects immediately');
    window.close();
  }

  {
    // A page the service worker served from its cache (the server was
    // unreachable) is a reconnect from the user's point of view: the
    // worker only has the copy because the server was up before.
    const {window} = boot(window => {
      const meta = window.document.createElement('meta');
      meta.setAttribute('name', 'kiss-offline-shell');
      window.document.head.appendChild(meta);
    });
    assert.strictEqual(
      overlayText(window),
      RECONNECTING,
      'the overlay says "Reconnecting ..." on a page served from the worker cache',
    );
    ok('offline-cached page shows "Reconnecting ..." immediately');
    window.close();
  }

  console.log('\nAll webapp server-loading-overlay tests passed.');
}

run().catch(err => {
  console.error('FAIL:', err);
  process.exit(1);
});
