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
// jsdom cannot navigate: window.location.reload() surfaces as a "Not
// implemented: navigation" jsdomError on the virtual console, which is
// how the reloads are counted.  fetch is replayed from a script so the
// test controls every answer; the page's timers are recorded and fired
// by hand.

'use strict';

const assert = require('assert');
const {JSDOM, VirtualConsole} = require('jsdom');
const {
  bootShim,
  installFakeWebSocket,
  frames,
} = require('./transport_page_helpers');

function page() {
  const reloads = [];
  const vc = new VirtualConsole();
  vc.on('jsdomError', err => {
    if (/navigation/.test(String(err && err.message))) reloads.push(1);
  });
  const dom = new JSDOM(
    `<!DOCTYPE html><html><head></head><body>
    <div id="kiss-server-loading" role="status">
      <div id="kiss-server-loading-msg" class="kiss-server-loading-msg">starting</div></div>
    <div id="app" style="display:none;">
      <div id="auth-modal" style="display:none;">
        <input id="auth-modal-input" type="password">
        <div id="auth-modal-error" role="alert"></div>
        <button id="auth-modal-ok"></button>
        <button id="auth-modal-cancel"></button>
      </div>
    </div></body></html>`,
    {
      runScripts: 'outside-only',
      pretendToBeVisual: true,
      url: 'https://example.test/',
      virtualConsole: vc,
    },
  );
  const {window} = dom;
  const sockets = [];
  const timers = [];
  installFakeWebSocket(window, sockets);
  window.setTimeout = function (fn, ms) {
    timers.push({fn, ms, cleared: false, ran: false});
    return timers.length;
  };
  window.clearTimeout = function (id) {
    if (timers[id - 1]) timers[id - 1].cleared = true;
  };
  // Every /api/daemon request is answered from the front of `answers`:
  // {instance} -> 200 JSON, 'fail' -> a network error, 'http500' ->
  // 500, 'http404' -> 404, 'throw' -> fetch itself throws, a function
  // -> a deferred answer the test resolves later.
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
    if (typeof next === 'function') return next();
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(next),
    });
  };
  bootShim(window);
  return {window, sockets, timers, reloads, answers, requests};
}

// Let the fetch promise chain settle (Node's own timer, not the page's).
function settle() {
  return new Promise(r => setTimeout(r, 20));
}

function authOk(sock) {
  sock.fireOpen();
  sock.fireMessage({type: 'auth_ok'});
}

function pendingTimers(timers) {
  return timers.filter(t => !t.cleared && !t.ran);
}

/**
 * The pending daemon-probe retries (5 s doubling to 60 s); the shim's
 * stale check (15 s) and stable timer (30 s) are not among them.
 */
const RETRY_MS = [5000, 10000, 20000, 40000, 60000];
function retries(timers) {
  return pendingTimers(timers).filter(t => RETRY_MS.includes(t.ms));
}

function fire(timer) {
  timer.ran = true;
  timer.fn();
}

/**
 * Drop socket number *n* and run the reconnect timer the shim armed for
 * it, so socket *n + 1* opens; return that socket.
 */
function reconnect(sockets, timers, n) {
  sockets[n].fireClose();
  return openNext(sockets, timers, n);
}

/** Run the pending reconnect timer, so socket *n + 1* opens. */
function openNext(sockets, timers, n) {
  const pending = pendingTimers(timers).filter(t => t.ms >= 500);
  assert.strictEqual(pending.length, 1, 'one reconnect timer is pending');
  fire(pending[0]);
  assert.strictEqual(sockets.length, n + 2, 'the timer opened the next socket');
  return sockets[n + 1];
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

  // A drop while a retry is pending cancels it, even on a page whose
  // clearTimeout throws (the shim's other timer cancels tolerate that
  // too): the reconnect still goes ahead.
  reconnect(sockets, timers, 2);
  answers.push('fail');
  authOk(sockets[3]);
  await settle();
  const orphan = retries(timers);
  assert.strictEqual(orphan.length, 1, 'the failure armed a retry');
  const realClearTimeout = window.clearTimeout;
  window.clearTimeout = function () {
    throw new Error('clearTimeout blocked');
  };
  const before = pendingTimers(timers);
  sockets[3].fireClose();
  window.clearTimeout = realClearTimeout;
  // The throwing clearTimeout could mark none of the timers the close
  // path cancels (the retry, the stale check, the stable timer).
  for (const t of before) t.ran = true;
  assert.strictEqual(
    pendingTimers(timers).filter(t => t.ms >= 500).length,
    1,
    'the close path survived and armed the reconnect',
  );
  openNext(sockets, timers, 3);
  answers.push({instance: 'daemon-A', version: '2026.10.21'});
  authOk(sockets[4]);
  await settle();
  assert.strictEqual(retries(timers).length, 0, 'the retry did not survive');

  // An answer without a token is final: a 404 (a daemon without the
  // endpoint) or JSON naming no instance arms no retry.
  reconnect(sockets, timers, 4);
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
  window.close();
  console.log('transportReloadOnNewDaemon.test.js: all tests passed');
}

main().catch(err => {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exitCode = 1;
});
