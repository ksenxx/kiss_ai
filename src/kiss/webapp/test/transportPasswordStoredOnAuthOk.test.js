'use strict';
// A typed password reaches localStorage only once the server has
// accepted it (auth_ok).  Every tab of the origin probes with the
// stored password on each reconnect, and the server counts every
// non-empty wrong guess against the device's lockout bucket: a typo in
// one tab must not be replayed by the other tabs while it is still
// pending.  Two jsdom windows share one localStorage here, as two
// browser tabs of one origin do.
//
// Before the fix `_onPasswordEntered` stored the typed value at once.
const assert = require('assert');
const {
  newAuthDom,
  bootShim,
  installFakeWebSocket,
  installFakeTimers,
  openNextSocket,
  settle,
  frames,
  ok,
  fail,
} = require('./transport_page_helpers');

const KEY = 'sorcar-remote-pwd';

/**
 * A tab on the real shim whose socket is born CONNECTING; its timers
 * are recorded, so a reconnect backoff is fired by `reconnect`, never
 * slept through.
 */
function newTab(storage) {
  const {window} = newAuthDom('https://example.test/');
  if (storage) {
    Object.defineProperty(window, 'localStorage', {
      value: storage,
      configurable: true,
    });
  }
  const sockets = [];
  installFakeWebSocket(window, sockets, {connecting: true});
  const timers = installFakeTimers(window);
  bootShim(window);
  return {window, sockets, timers};
}

/** Fire the reconnect timer the tab armed, so socket *n + 1* opens. */
function reconnect(tab, n) {
  openNextSocket(tab.sockets, tab.timers, n);
}

/** Type *pwd* into the tab's dialog and press OK; wait for the frame. */
async function submitPassword(tab, pwd) {
  tab.window.document.getElementById('auth-modal-input').value = pwd;
  tab.window.document.getElementById('auth-modal-ok').click();
  await settle();
}

async function run() {
  // 1. Two tabs: the typo of tab A is not replayed by tab B's reconnect.
  {
    const a = newTab(null);
    const b = newTab(a.window.localStorage);
    const storage = a.window.localStorage;
    a.sockets[0].fireOpen();
    a.sockets[0].fireMessage({type: 'auth_required'});
    b.sockets[0].fireOpen();
    b.sockets[0].fireMessage({type: 'auth_required'});
    await submitPassword(a, 'typo');
    try {
      assert.deepStrictEqual(frames(a.sockets[0]).at(-1), {
        type: 'auth',
        password: 'typo',
      });
      assert.strictEqual(
        storage.getItem(KEY),
        null,
        'a password the server has not accepted is not stored',
      );
      // The server drops B's idle unauthenticated socket while A's
      // guess is pending; B reconnects underneath its open dialog.
      b.sockets[0].fireClose();
      reconnect(b, 0);
      b.sockets[1].fireOpen();
      assert.deepStrictEqual(
        frames(b.sockets[1]),
        [{type: 'auth', password: ''}],
        "tab B's reconnect probes with the uncounted empty password",
      );
      ok(
        "a pending guess in one tab is not replayed by another tab's reconnect",
      );
    } catch (err) {
      fail('typed password leaked to the other tab before auth_ok', err);
    }

    // The server refuses A's guess and closes; A's reconnect probes
    // bare, not with the refused guess.
    a.sockets[0].fireMessage({
      type: 'error',
      code: 'auth_failed',
      text: 'That password is not correct. Try again.',
    });
    a.sockets[0].fireClose();
    reconnect(a, 0);
    a.sockets[1].fireOpen();
    try {
      assert.deepStrictEqual(frames(a.sockets[1]), [
        {type: 'auth', password: ''},
      ]);
      assert.strictEqual(storage.getItem(KEY), null);
      ok('the refused guess is not re-sent by the reconnect');
    } catch (err) {
      fail('refused guess replayed', err);
    }

    // The right password: stored on auth_ok, and B's next reconnect
    // uses it.
    a.sockets[1].fireMessage({type: 'auth_required'});
    await submitPassword(a, 'right');
    try {
      assert.strictEqual(storage.getItem(KEY), null, 'still pending');
      a.sockets[1].fireMessage({type: 'auth_ok'});
      await settle();
      assert.strictEqual(
        storage.getItem(KEY),
        'right',
        'auth_ok stores the accepted password',
      );
      assert.ok(
        !a.window.document.getElementById('auth-modal').style.display ||
          a.window.document.getElementById('auth-modal').style.display ===
            'none',
        'the dialog closes on auth_ok',
      );
      b.sockets[1].fireClose();
      reconnect(b, 1);
      b.sockets[2].fireOpen();
      assert.deepStrictEqual(frames(b.sockets[2]), [
        {type: 'auth', password: 'right'},
      ]);
      ok(
        'the accepted password is stored on auth_ok and shared with the other tab',
      );
    } catch (err) {
      fail('accepted password not stored', err);
    }
    a.window.close();
    b.window.close();
  }

  // 2. The server dropped the socket while the user typed: the typed
  //    password still goes out in the reconnect's auth frame (it is
  //    not in storage yet) and is stored once accepted.
  {
    const tab = newTab(null);
    const storage = tab.window.localStorage;
    tab.sockets[0].fireOpen();
    tab.sockets[0].fireMessage({type: 'auth_required'});
    tab.sockets[0].fireClose();
    reconnect(tab, 0);
    // Still CONNECTING when the user presses OK: the typed password
    // waits for this socket's onopen.
    await submitPassword(tab, 'typed-late');
    try {
      assert.deepStrictEqual(tab.sockets[1].sent, [], 'nothing sent yet');
      assert.strictEqual(storage.getItem(KEY), null);
      tab.sockets[1].fireOpen();
      assert.deepStrictEqual(frames(tab.sockets[1]), [
        {type: 'auth', password: 'typed-late'},
      ]);
      tab.sockets[1].fireMessage({type: 'auth_ok'});
      await settle();
      assert.strictEqual(storage.getItem(KEY), 'typed-late');
      ok(
        'a password typed while the socket was down goes out on the reconnect',
      );
    } catch (err) {
      fail('typed-while-down password lost', err);
    }

    // The dialog was closed when the socket dropped: Enter reconnects
    // at once and the new socket carries the typed password.
    tab.sockets[1].fireMessage({
      type: 'error',
      code: 'auth_failed',
      text: 'Wrong password.',
    });
    tab.sockets[1].fireClose();
    reconnect(tab, 1);
    tab.sockets[2].fireOpen();
    tab.sockets[2].fireMessage({type: 'auth_required'});
    tab.sockets[2].fireClose();
    reconnect(tab, 2);
    tab.sockets[3].fireOpen();
    tab.sockets[3].fireClose();
    await submitPassword(tab, 'after-drop');
    try {
      assert.strictEqual(storage.getItem(KEY), null);
      assert.strictEqual(tab.sockets.length, 5, 'Enter reconnects at once');
      tab.sockets[4].fireOpen();
      assert.deepStrictEqual(frames(tab.sockets[4]), [
        {type: 'auth', password: 'after-drop'},
      ]);
      ok('Enter on a dead socket reconnects with the typed password');
    } catch (err) {
      fail('Enter on a dead socket', err);
    }
    tab.window.close();
  }
  process.exit(0);
}

run().catch(err => fail('unexpected error', err));
