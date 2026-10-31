// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests of media/transport.js on the real shim over a fake WebSocket
// whose sockets are born CONNECTING (transport_page_helpers.js); the
// reconnect backoff timers are recorded and fired by hand.  Each test
// failed against the pre-fix code.
//
//  DIALOG  After Unlock the dialog stayed on screen with its listeners
//          detached.  When the reconnect that Unlock started failed (the
//          daemon unreachable), the close only scheduled another backoff
//          reconnect: Unlock, Cancel and Escape were dead until a later
//          reconnect reached the server, up to a minute after it was back.
//  BATCH   A frame posted between the reconnect flush and its `pong` went
//          straight out, untracked.  When the socket died before the
//          `pong`, the next connection re-sent the flushed batch, which
//          the server applied AFTER the newer frame: the newer setting was
//          silently reverted.  Such a frame now waits behind the
//          unconfirmed batch and goes out once, after it.

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  newAuthDom,
  bootShim,
  installFakeWebSocket,
  installFakeTimers,
  openNextSocket,
  frames,
  settle,
} = require('./transport_page_helpers');

const {test, report} = makeRunner();

/** A page on the real shim; sockets are born CONNECTING, timers recorded. */
function newPage() {
  const {window} = newAuthDom('https://example.test/');
  const sockets = [];
  installFakeWebSocket(window, sockets, {connecting: true});
  const timers = installFakeTimers(window);
  bootShim(window);
  return {window, doc: window.document, sockets, timers};
}

function dialogShown(doc) {
  return doc.getElementById('auth-modal').style.display === 'flex';
}

/**
 * The server asks for a password, drops the idle socket while the user
 * types, the user presses Unlock (which reconnects at once) and that
 * reconnect fails: the daemon is unreachable.
 */
async function unlockWhileServerUnreachable(page) {
  const {doc, sockets, timers} = page;
  sockets[0].fireOpen();
  sockets[0].fireMessage({type: 'auth_required'});
  assert.ok(dialogShown(doc), 'auth_required opens the dialog');
  sockets[0].fireClose(); // idle unauthenticated socket dropped
  openNextSocket(sockets, timers, 0).fireClose(); // the quiet reconnect fails
  doc.getElementById('auth-modal-input').value = 'typed-pw';
  doc.getElementById('auth-modal-ok').click();
  await settle(); // the dialog's promise hands the password on
  assert.strictEqual(sockets.length, 3, 'Unlock reconnects at once');
  sockets[2].fireClose(); // ... and that attempt fails too
}

/** The saveConfig values a fake server applied, over every socket in order. */
function appliedValues(sockets) {
  return sockets
    .flatMap(frames)
    .filter(f => f.type === 'saveConfig')
    .map(f => f.config.value);
}

async function main() {
  await test(
    'DIALOG: after a failed Unlock reconnect the dialog says why and ' +
      'Cancel closes it',
    async () => {
      const page = newPage();
      const {doc} = page;
      await unlockWhileServerUnreachable(page);
      assert.ok(dialogShown(doc), 'the dialog stays up with the typed text');
      assert.strictEqual(
        doc.getElementById('auth-modal-input').value,
        'typed-pw',
      );
      assert.strictEqual(
        doc.getElementById('auth-modal-error').textContent,
        'Cannot reach the server; retrying ...',
      );
      doc.getElementById('auth-modal-cancel').click();
      await settle();
      assert.ok(!dialogShown(doc), 'Cancel closes the dialog');
      assert.strictEqual(
        doc.getElementById('kiss-server-loading-msg').textContent,
        'A password is needed to use this server.',
      );
      page.window.close();
    },
  );

  await test('DIALOG: Escape closes the re-armed dialog', async () => {
    const page = newPage();
    const {doc, window} = page;
    await unlockWhileServerUnreachable(page);
    doc
      .getElementById('auth-modal-input')
      .dispatchEvent(new window.KeyboardEvent('keydown', {key: 'Escape'}));
    await settle();
    assert.ok(!dialogShown(doc), 'Escape closes the dialog');
    window.close();
  });

  await test('DIALOG: a second Unlock reconnects at once and sends the password', async () => {
    const page = newPage();
    const {doc, sockets} = page;
    await unlockWhileServerUnreachable(page);
    doc.getElementById('auth-modal-ok').click();
    await settle();
    assert.strictEqual(sockets.length, 4, 'the second Unlock reconnects');
    sockets[3].fireOpen();
    assert.deepStrictEqual(frames(sockets[3]), [
      {type: 'auth', password: 'typed-pw'},
    ]);
    sockets[3].fireMessage({type: 'auth_ok'});
    assert.ok(!dialogShown(doc), 'auth_ok closes the dialog');
    assert.strictEqual(doc.getElementById('auth-modal-error').textContent, '');
    page.window.close();
  });

  await test(
    'BATCH: a frame posted before the pong is never overwritten by the ' +
      're-sent batch',
    async () => {
      const page = newPage();
      const {window, sockets, timers} = page;
      const transport = window.acquireTransport();
      // Changed while offline: queued, flushed on the reconnect.
      transport.postMessage({type: 'saveConfig', config: {value: 'A'}});
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_ok'});
      // Changed again before the server's pong confirms the batch.
      transport.postMessage({type: 'saveConfig', config: {value: 'B'}});
      sockets[0].fireClose(); // died before the pong
      const next = openNextSocket(sockets, timers, 0);
      next.fireOpen();
      next.fireMessage({type: 'auth_ok'});
      // Wave 4: the held frame waits for the pong of the re-sent batch
      // too (it never joins that batch; see
      // transportUnconfirmedBatch_wave4.test.js).
      assert.deepStrictEqual(appliedValues([next]), ['A']);
      next.fireMessage({type: 'pong'});
      const applied = appliedValues(sockets);
      assert.strictEqual(
        applied[applied.length - 1],
        'B',
        'the newer setting must win: ' + JSON.stringify(applied),
      );
      assert.deepStrictEqual(
        appliedValues([next]),
        ['A', 'B'],
        'the next connection re-sends the batch, then the newer frame',
      );
      window.close();
    },
  );

  await test('BATCH: once the pong confirms the batch, nothing is re-sent', async () => {
    const page = newPage();
    const {window, sockets, timers} = page;
    const transport = window.acquireTransport();
    transport.postMessage({type: 'saveConfig', config: {value: 'A'}});
    sockets[0].fireOpen();
    sockets[0].fireMessage({type: 'auth_ok'});
    transport.postMessage({type: 'saveConfig', config: {value: 'B'}});
    sockets[0].fireMessage({type: 'pong'});
    transport.postMessage({type: 'saveConfig', config: {value: 'C'}});
    assert.deepStrictEqual(appliedValues(sockets), ['A', 'B', 'C']);
    sockets[0].fireClose();
    const next = openNextSocket(sockets, timers, 0);
    next.fireOpen();
    next.fireMessage({type: 'auth_ok'});
    assert.deepStrictEqual(
      frames(next).map(f => f.type),
      ['auth'],
      'a confirmed batch and the frames after it are not re-sent',
    );
    window.close();
  });

  report('transportAuthBatch_sweep1030');
}

main();
