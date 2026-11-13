// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// transport.js: a page that reconnects to a DIFFERENT daemon process
// reloads itself.  The shim resyncs in place after a drop, which is
// right for a network blip but leaves the page running the JavaScript
// it loaded before `install.sh` updated or reinstalled KISS Sorcar and
// restarted the daemon.  On every auth_ok the shim asks GET /api/daemon
// which process it reached (`DAEMON_INSTANCE` in web_server.py):
//
//   * the first answer is remembered, no reload;
//   * a reconnect answered with the same token (a blip) keeps the page;
//   * a reconnect answered with another token reloads the page, once
//     the batch flushed on that connection is confirmed by `pong`;
//   * a failed request (network error, 5xx, a throwing fetch) is
//     retried with a doubling backoff (5 s ... 60 s) while the
//     connection stays up; an answer without a token is final;
//   * an answer that lands after its connection dropped is ignored,
//     and a drop cancels the retry and the deferred reload.
//
// The reloads are counted by the harness's reloadCountingConsole.  fetch
// is replayed from a script so the test controls every answer; the
// page's timers are recorded and fired by hand.

'use strict';

const assert = require('assert');
const {reloadCountingConsole} = require('./simplify2_harness');
const {
  newAuthDom,
  bootShim,
  installFakeWebSocket,
  installFakeTimers,
  pendingTimers,
  fireTimer: fire,
  openNextSocket: openNext,
  settle,
  frames,
} = require('./transport_page_helpers');

function page() {
  const reloads = [];
  const {window} = newAuthDom('https://example.test/', {
    virtualConsole: reloadCountingConsole(reloads),
  });
  const sockets = [];
  installFakeWebSocket(window, sockets);
  const timers = installFakeTimers(window);
  // Every /api/daemon request is answered from the front of `answers`:
  // {instance} -> 200 JSON, 'fail' -> a network error, 'http500' ->
  // 500, 'http404' -> 404, 'throw' -> fetch itself throws, a function
  // (given the request's options) -> a deferred answer the test
  // resolves later.
  const answers = [];
  const requests = [];
  window.fetch = function (url, opts) {
    requests.push({url, opts});
    const next = answers.shift();
    if (next === 'throw') throw new TypeError('fetch is broken');
    if (next === 'fail') return Promise.reject(new TypeError('offline'));
    if (next === 'http500' || next === 'http404') {
      return Promise.resolve({
        ok: false,
        status: next === 'http500' ? 500 : 404,
        json: () => Promise.reject(new Error('no body')),
      });
    }
    if (typeof next === 'function') return next(opts);
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(next),
    });
  };
  bootShim(window);
  return {window, sockets, timers, reloads, answers, requests};
}

function authOk(sock) {
  sock.fireOpen();
  sock.fireMessage({type: 'auth_ok'});
}

/**
 * The pending daemon-probe retries (5 s doubling to 60 s); the shim's
 * stale check (15 s) and stable timer (30 s) are not among them.
 */
const RETRY_MS = [5000, 10000, 20000, 40000, 60000];
function retries(timers) {
  return pendingTimers(timers).filter(t => RETRY_MS.includes(t.ms));
}

/**
 * Drop socket number *n* and run the reconnect timer the shim armed for
 * it, so socket *n + 1* opens; return that socket.
 */
function reconnect(sockets, timers, n) {
  sockets[n].fireClose();
  return openNext(sockets, timers, n);
}

async function main() {
  const {window, sockets, timers, reloads, answers, requests} = page();
  assert.strictEqual(sockets.length, 1, 'the shim opens a socket');

  // First auth: the daemon's identity is learned, nothing reloads.
  answers.push({instance: 'daemon-A', version: '2026.10.21'});
  authOk(sockets[0]);
  await settle();
  assert.strictEqual(requests.length, 1, 'auth_ok asks /api/daemon once');
  assert.strictEqual(requests[0].url, '/api/daemon');
  assert.strictEqual(requests[0].opts.cache, 'no-store', 'never cached');
  assert.strictEqual(reloads.length, 0, 'the first answer is only remembered');
  assert.strictEqual(retries(timers).length, 0, 'a good answer arms no retry');

  // A blip: the socket drops and the same daemon answers the reconnect.
  reconnect(sockets, timers, 0);
  answers.push({instance: 'daemon-A', version: '2026.10.21'});
  authOk(sockets[1]);
  await settle();
  assert.strictEqual(requests.length, 2, 'every auth_ok asks again');
  assert.strictEqual(reloads.length, 0, 'the same daemon keeps the page');
  assert.ok(
    frames(sockets[1]).some(f => f.type === 'auth'),
    'the reconnect authenticated normally',
  );

  // Failures are retried with a doubling backoff, capped at 60 s, for
  // as long as the connection is up; a good answer ends the retries.
  reconnect(sockets, timers, 1);
  const failures = ['fail', 'http500', 'throw', 'fail', 'fail', 'fail'];
  answers.push(failures[0]);
  authOk(sockets[2]);
  await settle();
  const delays = [];
  for (let i = 1; i <= failures.length; i++) {
    const armed = retries(timers);
    assert.strictEqual(armed.length, 1, `failure ${i} arms one retry`);
    delays.push(armed[0].ms);
    answers.push(i < failures.length ? failures[i] : {instance: 'daemon-A'});
    fire(armed[0]);
    await settle();
  }
  assert.deepStrictEqual(delays, [5000, 10000, 20000, 40000, 60000, 60000]);
  assert.strictEqual(retries(timers).length, 0, 'the good answer ends it');
  assert.strictEqual(reloads.length, 0, 'still the same daemon');

  // A drop while a retry is pending cancels it.
  reconnect(sockets, timers, 2);
  answers.push('fail');
  authOk(sockets[3]);
  await settle();
  const dropped = retries(timers)[0];
  assert.ok(dropped, 'the failure armed a retry');
  sockets[3].fireClose();
  assert.ok(dropped.cleared, 'the drop cancelled the pending retry');
  assert.strictEqual(retries(timers).length, 0, 'no retry outlives the drop');

  // The same on a page whose clearTimeout throws (the shim's timer
  // cancels tolerate that): the reconnect still goes ahead, and the retry
  // the page could not cancel is harmless: fired for its retired
  // connection, it probes nothing and reloads nothing.
  openNext(sockets, timers, 3);
  answers.push('fail');
  authOk(sockets[4]);
  await settle();
  const orphan = retries(timers)[0];
  assert.ok(orphan, 'the failure armed a retry');
  const realClearTimeout = window.clearTimeout;
  const blocked = [];
  window.clearTimeout = function (id) {
    blocked.push(id);
    throw new Error('clearTimeout blocked');
  };
  const armedBefore = timers.length;
  sockets[4].fireClose();
  window.clearTimeout = realClearTimeout;
  assert.ok(
    blocked.includes(timers.indexOf(orphan) + 1),
    'the close path asked to cancel the retry',
  );
  assert.ok(!orphan.cleared, 'the blocked cancel left the retry armed');
  const reconnects = timers
    .slice(armedBefore)
    .filter(t => !t.cleared && !t.ran && t.ms >= 500);
  assert.strictEqual(
    reconnects.length,
    1,
    'the close path survived and armed the reconnect',
  );
  const probes = requests.length;
  fire(orphan);
  await settle();
  assert.strictEqual(requests.length, probes, 'the orphan retry probes nothing');
  assert.strictEqual(reloads.length, 0, 'and reloads nothing');
  // The other timers the blocked cancels left armed (the stale check,
  // the stable timer) are retired from the ledger: the page asked for
  // their cancellation, and the one whose firing matters is the retry.
  for (const id of blocked) if (timers[id - 1]) timers[id - 1].ran = true;
  fire(reconnects[0]);
  assert.strictEqual(sockets.length, 6, 'the reconnect opened the next socket');

  // An answer without a token is final: a 404 (a daemon without the
  // endpoint) or JSON naming no instance arms no retry.
  answers.push('http404');
  authOk(sockets[5]);
  await settle();
  assert.strictEqual(retries(timers).length, 0, 'a 404 is not retried');
  reconnect(sockets, timers, 5);
  answers.push({version: '2026.10.20'});
  authOk(sockets[6]);
  await settle();
  assert.strictEqual(retries(timers).length, 0, 'no token, no retry');
  assert.strictEqual(reloads.length, 0, 'and no reload either');

  // A late answer, landing after its connection dropped, is ignored:
  // a reload now would lose the batch just re-queued in memory.
  reconnect(sockets, timers, 6);
  let answerLate = null;
  answers.push(() => new Promise(r => (answerLate = r)));
  authOk(sockets[7]);
  await settle();
  assert.strictEqual(typeof answerLate, 'function', 'the probe is in flight');
  sockets[7].fireClose();
  answerLate({
    ok: true,
    status: 200,
    json: () => Promise.resolve({instance: 'daemon-B'}),
  });
  await settle();
  assert.strictEqual(reloads.length, 0, 'a late answer reloads nothing');
  assert.strictEqual(retries(timers).length, 0, 'and arms no retry');

  // A late FAILURE after the drop arms no retry for the dead connection.
  openNext(sockets, timers, 7);
  let failLate = null;
  answers.push(() => new Promise((_resolve, reject) => (failLate = reject)));
  authOk(sockets[8]);
  await settle();
  assert.strictEqual(typeof failLate, 'function', 'the probe is in flight');
  sockets[8].fireClose();
  failLate(new TypeError('offline'));
  await settle();
  assert.strictEqual(retries(timers).length, 0, 'no retry for a dead socket');
  assert.strictEqual(reloads.length, 0, 'and no reload');

  // install.sh restarted the daemon: the reconnect reaches a new
  // process and nothing is owed on the connection, so the page reloads.
  openNext(sockets, timers, 8);
  answers.push({instance: 'daemon-B', version: '2026.10.22'});
  authOk(sockets[9]);
  await settle();
  assert.strictEqual(reloads.length, 1, 'a different daemon reloads the page');
  // The new daemon is now the remembered one: no second reload.
  reconnect(sockets, timers, 9);
  answers.push({instance: 'daemon-B', version: '2026.10.22'});
  authOk(sockets[10]);
  await settle();
  assert.strictEqual(reloads.length, 1, 'the new daemon is remembered');

  // A command posted during an outage is flushed on the reconnect and
  // awaits the server's pong; a new daemon seen meanwhile waits for it.
  sockets[10].fireClose();
  window.acquireTransport().postMessage({type: 'setWorkDir', workDir: '/w'});
  openNext(sockets, timers, 10);
  answers.push({instance: 'daemon-C', version: '2026.10.23'});
  authOk(sockets[11]);
  await settle();
  let sent = frames(sockets[11]);
  assert.ok(
    sent.some(f => f.type === 'setWorkDir'),
    'the batch was flushed',
  );
  assert.ok(
    sent.some(f => f.type === 'ping'),
    'and its confirmation asked',
  );
  assert.strictEqual(reloads.length, 1, 'the reload waits for the pong');
  // The connection drops before the pong: the deferred reload goes
  // with it and the batch is re-queued ...
  sockets[11].fireClose();
  await settle();
  assert.strictEqual(reloads.length, 1, 'a drop cancels the deferred reload');
  // ... flushed again on the next connection, where the new daemon is
  // seen again and the pong lets the reload go ahead.
  openNext(sockets, timers, 11);
  answers.push({instance: 'daemon-C', version: '2026.10.23'});
  authOk(sockets[12]);
  await settle();
  sent = frames(sockets[12]);
  assert.ok(
    sent.some(f => f.type === 'setWorkDir'),
    'the batch is re-sent',
  );
  assert.strictEqual(reloads.length, 1, 'still waiting for the pong');
  sockets[12].fireMessage({type: 'pong'});
  assert.strictEqual(reloads.length, 2, 'the pong releases the reload');

  // A page without fetch at all: the check is skipped.
  reconnect(sockets, timers, 12);
  const realFetch = window.fetch;
  window.fetch = undefined;
  const asked = requests.length;
  authOk(sockets[13]);
  await settle();
  assert.strictEqual(requests.length, asked, 'no fetch, no request');
  assert.strictEqual(reloads.length, 2, 'no fetch, no reload');
  window.fetch = realFetch;

  // A probe that never answers (a proxy holding the request while the
  // socket stays up) is a lost probe too: its deadline aborts it, the
  // abort is retried like any failure, and the retry's answer from a
  // new daemon reloads the page.  Without the deadline nothing would
  // ever ask again on this connection.
  reconnect(sockets, timers, 13);
  answers.push(
    opts =>
      new Promise((_resolve, reject) => {
        opts.signal.addEventListener('abort', () =>
          reject(new Error('AbortError')),
        );
      }),
  );
  authOk(sockets[14]);
  await settle();
  const stalled = requests[requests.length - 1];
  assert.strictEqual(requests.length, asked + 1, 'the auth_ok probed');
  assert.ok(stalled.opts.signal, 'the probe carries an abort signal');
  const deadline = pendingTimers(timers).filter(t => t.ms === 12000);
  assert.strictEqual(deadline.length, 1, 'the probe armed its deadline');
  assert.strictEqual(retries(timers).length, 0, 'no retry while it waits');
  fire(deadline[0]);
  assert.ok(stalled.opts.signal.aborted, 'the deadline aborts the probe');
  await settle();
  const retry = retries(timers);
  assert.strictEqual(retry.length, 1, 'the aborted probe is retried');
  assert.strictEqual(retry[0].ms, 5000);
  answers.push({instance: 'daemon-D', version: '2026.10.24'});
  fire(retry[0]);
  await settle();
  assert.strictEqual(requests.length, asked + 2, 'the retry asked again');
  assert.strictEqual(
    pendingTimers(timers).filter(t => t.ms === 12000).length,
    0,
    'an answered probe clears its deadline',
  );
  sockets[14].fireMessage({type: 'pong'});
  assert.strictEqual(reloads.length, 3, 'the new daemon reloads the page');
  window.close();
  console.log('transportReloadOnNewDaemon.test.js: all tests passed');
}

main().catch(err => {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  // The page is left open (its fake sockets and listeners keep node
  // alive), so exit explicitly instead of hanging until the runner's cap.
  process.exit(1);
});
