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
//
// The server's answer also ends what the PREVIOUS answer left for the
// close that was to follow it (the refusal flag, the lockout delay): a
// close that never consumed them used to hand them to the next
// session, whose first drop was then taken for the refusal's or the
// lockout's.

'use strict';

const assert = require('assert');
const {
  newAuthDom,
  bootShim,
  installFakeWebSocket,
  installFakeTimers,
  openNextSocket,
} = require('./transport_page_helpers');

// A page with fake timers, fake intervals and a fake clock.
function page() {
  const {window} = newAuthDom();
  const sockets = [];
  const intervals = [];
  const clock = {now: 1000000};
  installFakeWebSocket(window, sockets, {halfOpen: true});
  const timers = installFakeTimers(window, {intervals, clock});
  bootShim(window);
  return {window, sockets, timers, intervals, clock};
}

function overlayText(window) {
  return window.document.getElementById('kiss-server-loading-msg').textContent;
}

function retryButton(window) {
  return window.document.getElementById('kiss-server-loading-action');
}

// The daemonStatus messages the shim dispatches to the app from now on.
function watchStatuses(window) {
  const statuses = [];
  window.addEventListener('message', ev => {
    if (ev.data && ev.data.type === 'daemonStatus') statuses.push(ev.data);
  });
  return statuses;
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

    const s1 = openNextSocket(sockets, timers, 0);
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
    const s2 = openNextSocket(sockets, timers, 1);
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
    const s1 = openNextSocket(sockets, timers, 0);
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
  {
    // Remote access is turned off while the user sits at the password
    // dialog: the refusal's close keeps the dialog (and never consumes
    // the refusal flag).  Once a session is up, its first drop must be
    // reported like any other.
    const {window, sockets, timers} = page();
    const statuses = watchStatuses(window);
    sockets[0].fireOpen();
    sockets[0].fireMessage({type: 'auth_required'});
    window.document.getElementById('auth-modal-input').value = 'pw';
    window.document.getElementById('auth-modal-ok').click();
    sockets[0].fireMessage({
      type: 'error',
      text: 'Remote access is turned off.',
    });
    sockets[0].fireClose();
    const s1 = openNextSocket(sockets, timers, 0);
    s1.fireOpen();
    s1.fireMessage({type: 'auth_ok'});
    statuses.length = 0;
    s1.fireClose();
    assert.deepStrictEqual(
      statuses.map(s => [s.connected, s.reconnecting]),
      [[false, true]],
      'the drop of the session that followed a refusal is reported',
    );
  }
  {
    // auth_locked, then a wake-up listener replaces the socket before
    // its onclose runs.  The new socket is admitted: its drop is an
    // ordinary reconnect, not the lockout's.
    const {window, sockets} = page();
    const statuses = watchStatuses(window);
    sockets[0].fireOpen();
    sockets[0].fireMessage({type: 'auth_locked', retry_after: 120});
    sockets[0].readyState = window.WebSocket.CLOSED;
    window.dispatchEvent(new window.Event('online'));
    assert.strictEqual(sockets.length, 2, 'the wake-up opened a new socket');
    sockets[1].fireOpen();
    sockets[1].fireMessage({type: 'auth_ok'});
    statuses.length = 0;
    sockets[1].fireClose();
    assert.deepStrictEqual(
      statuses.map(s => [s.connected, s.reconnecting]),
      [[false, true]],
      'the drop is a reconnect, not a lockout',
    );
  }
  console.log('conc1018_l1_transport_error_tick: ok');
}

main();
