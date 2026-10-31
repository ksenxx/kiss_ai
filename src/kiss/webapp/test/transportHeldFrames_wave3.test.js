// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// media/transport.js re-sends the batch it flushed on a reconnect when
// the socket dies before the server's `pong` confirms it.  A frame the
// page posted between that flush and the `pong` (a `submit`, a `stop`,
// a tab close) used to go straight out AND join the batch, so the next
// connection sent it a second time: a prompt the daemon had already
// started ran twice.  Such a frame now waits behind the batch and goes
// out exactly once: after the `pong`, or after the `pong` that confirms
// the batch re-sent on the next connection (wave 4: it never joins the
// re-sent batch, see transportUnconfirmedBatch_wave4.test.js).
//
// The real shim over a fake WebSocket whose sockets are born CONNECTING
// (transport_page_helpers.js); the reconnect backoff timers are fired by
// hand.

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  newAuthDom,
  bootShim,
  installFakeWebSocket,
  installFakeTimers,
  openNextSocket,
  frames,
} = require('./transport_page_helpers');

const {test, report} = makeRunner();

/** A page on the real shim; sockets are born CONNECTING, timers recorded. */
function newPage() {
  const {window} = newAuthDom('https://example.test/');
  const sockets = [];
  installFakeWebSocket(window, sockets, {connecting: true});
  const timers = installFakeTimers(window);
  bootShim(window);
  return {window, sockets, timers, transport: window.acquireTransport()};
}

/** The frame types one socket carried after its auth frame. */
function sentTypes(sock) {
  return frames(sock)
    .slice(1)
    .map(f => f.type);
}

/** How many `submit` frames went out over every socket. */
function submitCount(sockets) {
  return sockets.flatMap(frames).filter(f => f.type === 'submit').length;
}

async function main() {
  await test(
    'a submit posted before the pong goes out once when the socket dies ' +
      'and the batch is re-sent',
    async () => {
      const {window, sockets, timers, transport} = newPage();
      transport.postMessage({type: 'saveConfig', config: {value: 'A'}});
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_ok'});
      transport.postMessage({type: 'submit', prompt: 'run once'});
      assert.deepStrictEqual(
        sentTypes(sockets[0]),
        ['saveConfig', 'ping'],
        'the submit waits behind the unconfirmed batch',
      );
      sockets[0].fireClose(); // died before the pong
      const next = openNextSocket(sockets, timers, 0);
      next.fireOpen();
      next.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(sentTypes(next), ['saveConfig', 'ping']);
      next.fireMessage({type: 'pong'});
      assert.deepStrictEqual(sentTypes(next), ['saveConfig', 'ping', 'submit']);
      assert.strictEqual(submitCount(sockets), 1, 'the prompt runs once');
      window.close();
    },
  );

  await test(
    'frames held behind the batch go out once, in order, when the pong ' +
      'arrives, and a later reconnect re-sends none of them',
    async () => {
      const {window, sockets, timers, transport} = newPage();
      transport.postMessage({type: 'saveConfig', config: {value: 'A'}});
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_ok'});
      transport.postMessage({type: 'submit', prompt: 'first'});
      transport.postMessage({type: 'stop'});
      sockets[0].fireMessage({type: 'pong'});
      assert.deepStrictEqual(sentTypes(sockets[0]), [
        'saveConfig',
        'ping',
        'submit',
        'stop',
      ]);
      // Confirmed: a frame posted now goes straight out.
      transport.postMessage({type: 'getConfig'});
      assert.strictEqual(sentTypes(sockets[0]).at(-1), 'getConfig');
      sockets[0].fireClose();
      const next = openNextSocket(sockets, timers, 0);
      next.fireOpen();
      next.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(sentTypes(next), [], 'nothing is owed any more');
      assert.strictEqual(submitCount(sockets), 1);
      window.close();
    },
  );

  await test('with no batch to confirm, a frame posted after auth_ok goes straight out', async () => {
    const {window, sockets, transport} = newPage();
    sockets[0].fireOpen();
    sockets[0].fireMessage({type: 'auth_ok'});
    transport.postMessage({type: 'submit', prompt: 'live'});
    assert.deepStrictEqual(sentTypes(sockets[0]), ['submit']);
    window.close();
  });

  report('transportHeldFrames_wave3');
}

main();
