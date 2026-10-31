// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// media/transport.js re-sends the batch it flushed on a reconnect until
// the server's `pong` confirms it (at-least-once).  A frame the page
// posts while such a batch is unconfirmed (a `submit`, a `stop`) is held
// back behind it.  The held frames used to join the batch of the next
// connection, so a second connection lost before ITS `pong` sent a held
// `submit` again: the daemon has no per-command dedupe, and the prompt
// ran twice.  Held frames now wait, across every reconnect, until a
// `pong` confirms the batch ahead of them, and go out once.
//
// Also pinned: a batch left unconfirmed by a password rotation (the
// server refuses a frame with `auth_failed` and closes) is re-sent with
// the password the user types next, ahead of the frames held behind it.
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
  settle,
  frames,
} = require('./transport_page_helpers');

const {test, report} = makeRunner();

/** A page on the real shim at *url*; sockets born CONNECTING, timers recorded. */
function newPage(url) {
  const {window} = newAuthDom(url);
  const sockets = [];
  installFakeWebSocket(window, sockets, {connecting: true});
  const timers = installFakeTimers(window);
  return {window, sockets, timers};
}

/** Boot the shim into *page*; return its transport. */
function boot(page) {
  bootShim(page.window);
  return page.window.acquireTransport();
}

/** The frame types one socket carried after its auth frame(s). */
function sentTypes(sock) {
  return frames(sock)
    .filter(f => f.type !== 'auth')
    .map(f => f.type);
}

/** How many `submit` frames went out over every socket. */
function submitCount(sockets) {
  return sockets.flatMap(frames).filter(f => f.type === 'submit').length;
}

/** Open the next socket of *page* (socket *n + 1*) and authenticate it. */
function reconnect(page, n) {
  const sock = openNextSocket(page.sockets, page.timers, n);
  sock.fireOpen();
  sock.fireMessage({type: 'auth_ok'});
  return sock;
}

async function main() {
  await test('a held submit runs once when two reconnects in a row lose the pong', async () => {
    const page = newPage('https://example.test/');
    const transport = boot(page);
    const {sockets} = page;
    transport.postMessage({type: 'saveConfig', config: {value: 'A'}});
    sockets[0].fireOpen();
    sockets[0].fireMessage({type: 'auth_ok'});
    transport.postMessage({
      type: 'submit',
      tabId: 'parent-sync',
      prompt: 'run exactly once',
    });
    sockets[0].fireClose(); // died before the pong
    const second = reconnect(page, 0);
    assert.deepStrictEqual(
      sentTypes(second),
      ['saveConfig', 'ping'],
      'the held submit does not join the re-sent batch',
    );
    second.fireClose(); // died before ITS pong
    const third = reconnect(page, 1);
    assert.deepStrictEqual(sentTypes(third), ['saveConfig', 'ping']);
    third.fireMessage({type: 'pong'});
    assert.deepStrictEqual(sentTypes(sockets[0]), ['saveConfig', 'ping']);
    assert.deepStrictEqual(sentTypes(third), ['saveConfig', 'ping', 'submit']);
    assert.strictEqual(submitCount(sockets), 1, 'the prompt runs once');
    page.window.close();
  });

  await test(
    'a frame posted while the socket is down stays behind the held ones, ' +
      'and none of them is re-sent once confirmed',
    async () => {
      const page = newPage('https://example.test/');
      const transport = boot(page);
      const {sockets} = page;
      transport.postMessage({type: 'saveConfig', config: {value: 'A'}});
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_ok'});
      transport.postMessage({type: 'submit', prompt: 'first'});
      sockets[0].fireClose();
      transport.postMessage({type: 'stop'}); // posted while disconnected
      const second = reconnect(page, 0);
      assert.deepStrictEqual(sentTypes(second), ['saveConfig', 'ping']);
      second.fireMessage({type: 'pong'});
      assert.deepStrictEqual(sentTypes(second), [
        'saveConfig',
        'ping',
        'submit',
        'stop',
      ]);
      // Confirmed: a frame posted now goes straight out.
      transport.postMessage({type: 'getConfig'});
      assert.strictEqual(sentTypes(second).at(-1), 'getConfig');
      second.fireClose();
      const third = reconnect(page, 1);
      assert.deepStrictEqual(sentTypes(third), [], 'nothing is owed any more');
      assert.strictEqual(submitCount(sockets), 1);
      page.window.close();
    },
  );

  await test(
    'a batch left unconfirmed by a password rotation is re-sent with the ' +
      'newly typed password, ahead of the frames held behind it',
    async () => {
      const page = newPage('http://127.0.0.1:8787/');
      page.window.localStorage.setItem(
        'sorcar-remote-pwd',
        'Secret20Chars0000000',
      );
      const transport = boot(page);
      const {window, sockets} = page;
      transport.postMessage({type: 'saveConfig', config: {value: 'A'}});
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_ok'});
      transport.postMessage({type: 'submit', prompt: 'after the save'});
      // The password was rotated on the daemon: the batch's frames are
      // refused and the socket closed, with no pong.
      sockets[0].fireMessage({
        type: 'error',
        code: 'auth_failed',
        text: 'Wrong password.',
      });
      sockets[0].close();
      const next = openNextSocket(sockets, page.timers, 0);
      next.fireOpen();
      next.fireMessage({type: 'auth_required'});
      window.document.getElementById('auth-modal-input').value =
        'NewPassword000000000';
      window.document.getElementById('auth-modal-ok').click();
      await settle(); // the dialog resolves a promise before the auth frame goes out
      transport.postMessage({type: 'getConfig'}); // posted while socket 2 awaits auth_ok
      next.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(frames(next).slice(1), [
        {type: 'auth', password: 'NewPassword000000000'},
        {
          password: 'NewPassword000000000',
          type: 'saveConfig',
          config: {value: 'A'},
        },
        {password: 'NewPassword000000000', type: 'ping'},
      ]);
      next.fireMessage({type: 'pong'});
      assert.deepStrictEqual(frames(next).slice(4), [
        {
          password: 'NewPassword000000000',
          type: 'submit',
          prompt: 'after the save',
        },
        {password: 'NewPassword000000000', type: 'getConfig'},
      ]);
      assert.deepStrictEqual(sentTypes(sockets[0]), ['saveConfig', 'ping']);
      assert.strictEqual(submitCount(sockets), 1);
      window.close();
    },
  );

  report('transportUnconfirmedBatch_wave4');
}

main();
