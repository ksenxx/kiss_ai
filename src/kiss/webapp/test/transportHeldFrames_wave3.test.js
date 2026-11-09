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
  submitCount,
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

  await test(
    'a submit posted while the socket is CLOSING (main.js has not heard of ' +
      'the drop yet) never joins a re-sent batch: it goes out once, after ' +
      'a pong',
    async () => {
      const {window, sockets, timers, transport} = newPage();
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_ok'});
      // The server began the close handshake; onclose has not fired.
      sockets[0].readyState = window.WebSocket.CLOSING;
      transport.postMessage({type: 'submit', prompt: 'run once'});
      assert.deepStrictEqual(sentTypes(sockets[0]), []);
      sockets[0].fireClose();
      const second = openNextSocket(sockets, timers, 0);
      second.fireOpen();
      second.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(
        sentTypes(second),
        ['ping'],
        'the submit is not in the batch; the ping alone asks for the pong',
      );
      second.fireClose(); // died before the pong
      const third = openNextSocket(sockets, timers, 1);
      third.fireOpen();
      third.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(sentTypes(third), ['ping']);
      third.fireMessage({type: 'pong'});
      assert.deepStrictEqual(sentTypes(third), ['ping', 'submit']);
      assert.strictEqual(submitCount(sockets), 1, 'the prompt runs once');
      window.close();
    },
  );

  await test(
    'an Explorer paste (fsAction copy) posted while the socket is CLOSING ' +
      'never joins a re-sent batch: two lost pongs send it once, after a pong',
    async () => {
      const {window, sockets, timers, transport} = newPage();
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_ok'});
      sockets[0].readyState = window.WebSocket.CLOSING;
      transport.postMessage({
        type: 'fsAction',
        action: 'copy',
        path: '/workspace/a.txt',
        dest: '/workspace',
        token: 'one-user-paste',
      });
      assert.deepStrictEqual(sentTypes(sockets[0]), []);
      sockets[0].fireClose();
      const second = openNextSocket(sockets, timers, 0);
      second.fireOpen();
      second.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(
        sentTypes(second),
        ['ping'],
        'the copy is not in the batch; the ping alone asks for the pong',
      );
      second.fireClose(); // the daemon did the copy; the pong never came
      const third = openNextSocket(sockets, timers, 1);
      third.fireOpen();
      third.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(sentTypes(third), ['ping']);
      third.fireMessage({type: 'pong'});
      assert.deepStrictEqual(sentTypes(third), ['ping', 'fsAction']);
      assert.strictEqual(
        sockets.flatMap(frames).filter(f => f.type === 'fsAction').length,
        1,
        'one paste makes one copy',
      );
      window.close();
    },
  );

  await test(
    'a cron run_now posted while disconnected waits behind the batch with ' +
      'every frame after it, and a lost pong re-sends the batch alone',
    async () => {
      const {window, sockets, timers, transport} = newPage();
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_ok'});
      sockets[0].fireClose();
      transport.postMessage({type: 'saveConfig', config: {value: 'A'}});
      transport.postMessage({
        type: 'cronAction',
        action: 'run_now',
        jobId: 'j',
      });
      transport.postMessage({type: 'getCronJobs'}); // keeps its place
      const second = openNextSocket(sockets, timers, 0);
      second.fireOpen();
      second.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(sentTypes(second), ['saveConfig', 'ping']);
      second.fireClose(); // died before the pong
      const third = openNextSocket(sockets, timers, 1);
      third.fireOpen();
      third.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(sentTypes(third), ['saveConfig', 'ping']);
      third.fireMessage({type: 'pong'});
      assert.deepStrictEqual(sentTypes(third), [
        'saveConfig',
        'ping',
        'cronAction',
        'getCronJobs',
      ]);
      assert.strictEqual(
        sockets.flatMap(frames).filter(f => f.type === 'cronAction').length,
        1,
        'the job runs once',
      );
      window.close();
    },
  );

  report('transportHeldFrames_wave3');
}

main();
