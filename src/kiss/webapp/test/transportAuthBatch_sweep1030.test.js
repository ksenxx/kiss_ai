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
  unlock,
  overlayText,
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

function errorLine(doc) {
  return doc.getElementById('auth-modal-error').textContent;
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
  await unlock(doc, 'typed-pw');
  assert.strictEqual(sockets.length, 3, 'Unlock reconnects at once');
  sockets[2].fireClose(); // ... and that attempt fails too
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
      assert.strictEqual(errorLine(doc), 'Cannot reach the server; retrying ...');
      doc.getElementById('auth-modal-cancel').click();
      await settle();
      assert.ok(!dialogShown(doc), 'Cancel closes the dialog');
      assert.strictEqual(
        overlayText(page.window),
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
      {type: 'auth', password: 'typed-pw', parts: true},
    ]);
    sockets[3].fireMessage({type: 'auth_ok'});
    assert.ok(!dialogShown(doc), 'auth_ok closes the dialog');
    assert.strictEqual(errorLine(doc), '');
    page.window.close();
  });

  await test(
    'DIALOG: a typed password the server rejects as the first auth of a ' +
      'reconnect is reported, in place of the cannot-reach note',
    async () => {
      const page = newPage();
      const {doc, sockets, timers} = page;
      await unlockWhileServerUnreachable(page);
      const s = openNextSocket(sockets, timers, 2);
      s.fireOpen();
      assert.deepStrictEqual(frames(s), [
        {type: 'auth', password: 'typed-pw', parts: true},
      ]);
      s.fireMessage({type: 'auth_required'}); // how a wrong FIRST auth is answered
      assert.strictEqual(errorLine(doc), 'That password is not correct. Try again.');
      page.window.close();
    },
  );

  await test('DIALOG: the cannot-reach note goes when the server answers an empty Unlock', async () => {
    const page = newPage();
    const {doc, sockets} = page;
    await unlockWhileServerUnreachable(page);
    await unlock(doc, '');
    sockets[3].fireOpen();
    sockets[3].fireMessage({type: 'auth_required'}); // answers the empty probe
    assert.strictEqual(errorLine(doc), '');
    page.window.close();
  });

  await test(
    'DIALOG: an Unlock sent before the probe is answered stays pending: the ' +
      "probe's auth_required does not call it wrong and auth_ok stores it",
    async () => {
      const page = newPage();
      const {doc, sockets, timers, window} = page;
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_required'});
      sockets[0].fireClose(); // idle unauthenticated socket dropped
      const s = openNextSocket(sockets, timers, 0);
      s.fireOpen(); // the quiet reconnect sends the empty probe
      await unlock(doc, 'typed-pw'); // second auth of the same socket
      s.fireMessage({type: 'auth_required'}); // answers the probe
      assert.strictEqual(errorLine(doc), '');
      s.fireMessage({type: 'auth_ok'});
      assert.strictEqual(window.localStorage.getItem('sorcar-remote-pwd'), 'typed-pw');
      window.close();
    },
  );

  await test('DIALOG: a typed password answered with auth_locked is not sent again', async () => {
    const page = newPage();
    const {doc, sockets, timers} = page;
    sockets[0].fireOpen();
    sockets[0].fireMessage({type: 'auth_required'});
    await unlock(doc, 'typed-pw');
    sockets[0].fireMessage({type: 'auth_locked', retry_after: 30});
    sockets[0].fireClose();
    const s = openNextSocket(sockets, timers, 0); // the lockout is over
    s.fireOpen();
    assert.deepStrictEqual(frames(s), [{type: 'auth', password: '', parts: true}]);
    page.window.close();
  });

  report('transportAuthBatch_sweep1030');
}

main();
