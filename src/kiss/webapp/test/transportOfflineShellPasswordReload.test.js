'use strict';
// A page the service worker served from its offline cache (it carries
// the ``kiss-offline-shell`` meta) reloads itself on its first
// ``auth_ok``.  When that handshake was an interactive login, the
// accepted password must already be in localStorage when the reload
// fires: the fresh page probes with the stored password in its
// ``onopen`` and, without it, would ask the user again.
//
// Before the fix the ``auth_ok`` handler stored the password only
// after the reload branch had returned, so the reloaded page sent an
// empty auth frame and reopened the dialog.
const assert = require('assert');
const {
  newAuthDom,
  bootShim,
  installFakeWebSocket,
  frames,
  settle,
  unlock,
  ok,
  fail,
} = require('./transport_page_helpers');

const KEY = 'sorcar-remote-pwd';
const RELOADED_FLAG = 'sorcar-offline-reloaded';

/**
 * A tab on the real shim, carrying the meta the service worker's cached
 * page has.  *storage* shares localStorage and sessionStorage with an
 * earlier tab, as a reload of the same tab keeps both.  *reloads* counts
 * ``window.location.reload()`` calls: jsdom cannot navigate and reports
 * each as a "Not implemented: navigation" jsdomError.
 */
function newTab(storage, reloads) {
  const dom = newAuthDom('https://example.test/');
  const {window} = dom;
  const meta = window.document.createElement('meta');
  meta.setAttribute('name', 'kiss-offline-shell');
  meta.setAttribute('content', '1');
  window.document.head.appendChild(meta);
  if (storage) {
    Object.defineProperty(window, 'localStorage', {
      value: storage.local,
      configurable: true,
    });
    Object.defineProperty(window, 'sessionStorage', {
      value: storage.session,
      configurable: true,
    });
  }
  dom.virtualConsole.removeAllListeners('jsdomError');
  dom.virtualConsole.on('jsdomError', err => {
    if (/navigation/.test(String(err && err.message))) reloads.push(1);
    else console.error(err);
  });
  const sockets = [];
  installFakeWebSocket(window, sockets, {connecting: true});
  bootShim(window);
  return {window, sockets};
}

async function run() {
  const reloads = [];
  // 1. The cached page: the user types the password, the server
  //    accepts it, the page reloads with the password stored.
  const cached = newTab(null, reloads);
  const storage = {
    local: cached.window.localStorage,
    session: cached.window.sessionStorage,
  };
  cached.sockets[0].fireOpen();
  try {
    assert.deepStrictEqual(frames(cached.sockets[0]), [
      {type: 'auth', password: '', parts: true},
    ]);
    cached.sockets[0].fireMessage({type: 'auth_required'});
    await settle();
    await unlock(cached.window.document, 'right');
    assert.deepStrictEqual(frames(cached.sockets[0]).at(-1), {
      type: 'auth',
      password: 'right',
      parts: true,
    });
    assert.strictEqual(storage.local.getItem(KEY), null, 'still pending');
    cached.sockets[0].fireMessage({type: 'auth_ok'});
    await settle();
    assert.strictEqual(reloads.length, 1, 'the cached page reloads once');
    assert.strictEqual(storage.session.getItem(RELOADED_FLAG), '1');
    assert.strictEqual(
      storage.local.getItem(KEY),
      'right',
      'the accepted password is stored before the reload',
    );
    ok('an accepted interactive password is stored before the offline reload');
  } catch (err) {
    fail('password not stored before the offline-shell reload', err);
  }
  cached.window.close();

  // 2. The reloaded page (the worker may serve the cached copy again):
  //    its first auth frame carries the stored password, so the user
  //    is not asked twice, and the reload-once flag keeps it in place.
  const reloaded = newTab(storage, reloads);
  reloaded.sockets[0].fireOpen();
  try {
    assert.deepStrictEqual(frames(reloaded.sockets[0]), [
      {type: 'auth', password: 'right', parts: true},
    ]);
    reloaded.sockets[0].fireMessage({type: 'auth_ok'});
    await settle();
    assert.strictEqual(reloads.length, 1, 'the reloaded page stays');
    const modal = reloaded.window.document.getElementById('auth-modal');
    assert.ok(
      !modal.style.display || modal.style.display === 'none',
      'no password dialog after the reload',
    );
    ok('the reloaded page authenticates with the stored password');
  } catch (err) {
    fail('reloaded page asked for the password again', err);
  }
  reloaded.window.close();
}

run().catch(err => fail('unexpected error', err));
