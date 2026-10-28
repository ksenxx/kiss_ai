// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// The "Do not show this again" checkbox of a notification that offers it
// (`dontShowAgain: true`: the "VS Code extension removed" notice and the
// phone swipe hint, see swipeHintOnMobile.test.js):
//
// * The toast renders a checkbox labelled "Do not show this again" under its
//   message; a toast without the flag has none.
// * Ticking it posts `{type: 'dismissNotice', id}` to the daemon and
//   removes the toast at once.
// * A `close` for the id from the daemon (its answer, reaching every
//   window) removes the toast in a window that did not tick the box.
// * Replaying the notice (a reconnect) re-renders one toast with the
//   box unticked, not two.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

const NOTICE = {
  type: 'notification',
  id: 'vscode-retired',
  severity: 'warning',
  sticky: true,
  title: 'VS Code extension removed',
  message:
    'The KISS Sorcar VS Code extension has been removed and will never be ' +
    'updated. Use this web app instead: http://127.0.0.1:8787',
  dontShowAgain: true,
};

const TOAST = '.kiss-notification[data-notification-id="vscode-retired"]';

// The page renders and removes toasts synchronously, so `send` and a
// click are followed by plain assertions.
function runTests() {
  {
    const {win, posted} = makePage();
    try {
      send(win, NOTICE);
      const toast = win.document.querySelector(TOAST);
      assert.ok(toast, 'the notice must render a toast');
      const label = toast.querySelector('label.kiss-notification-opt-out');
      assert.ok(label, 'the toast must carry the opt-out label');
      assert.strictEqual(label.textContent.trim(), 'Do not show this again');
      const box = label.querySelector('input[type="checkbox"]');
      assert.ok(box, 'the label must wrap a checkbox');
      assert.strictEqual(box.checked, false, 'the box starts unticked');
      assert.ok(
        toast.querySelector('.kiss-notification-content').contains(label),
        'the box sits under the message, inside the content column',
      );
      assert.strictEqual(
        posted.filter(m => m && m.type === 'dismissNotice').length,
        0,
        'rendering alone must not dismiss anything',
      );

      // Replaying the notice (reconnect) keeps one toast, box unticked.
      send(win, NOTICE);
      assert.strictEqual(
        win.document.querySelectorAll(TOAST).length,
        1,
        'a replayed notice must not stack a second toast',
      );
      assert.strictEqual(
        win.document.querySelector(TOAST + ' input[type="checkbox"]').checked,
        false,
      );

      // Ticking the box tells the daemon and drops the toast.
      const liveBox = win.document.querySelector(TOAST + ' input[type="checkbox"]');
      liveBox.click();
      assert.strictEqual(liveBox.checked, true);
      const dismissals = posted.filter(m => m && m.type === 'dismissNotice');
      // Objects cross the jsdom realm: compare by value, not prototype.
      assert.strictEqual(
        JSON.stringify(dismissals),
        JSON.stringify([{type: 'dismissNotice', id: 'vscode-retired'}]),
      );
      assert.strictEqual(
        win.document.querySelector(TOAST),
        null,
        'the toast must leave once the box is ticked',
      );
    } finally {
      win.close();
    }
  }

  {
    // A sibling window: it never ticked the box, the daemon's close
    // removes its copy of the toast.
    const {win, posted} = makePage();
    try {
      send(win, NOTICE);
      assert.ok(win.document.querySelector(TOAST), 'toast renders');
      send(win, {type: 'notification', id: 'vscode-retired', close: true});
      assert.strictEqual(
        win.document.querySelector(TOAST),
        null,
        'the close must remove the toast',
      );
      assert.strictEqual(posted.filter(m => m && m.type === 'dismissNotice').length, 0);
    } finally {
      win.close();
    }
  }

  {
    // A notice without the flag has no checkbox.
    const {win} = makePage();
    try {
      send(win, {
        type: 'notification',
        id: 'plain',
        severity: 'info',
        message: 'Just a note',
      });
      const toast = win.document.querySelector(
        '.kiss-notification[data-notification-id="plain"]',
      );
      assert.ok(toast, 'plain toast renders');
      assert.strictEqual(toast.querySelector('.kiss-notification-opt-out'), null);
      assert.strictEqual(toast.querySelector('input[type="checkbox"]'), null);
    } finally {
      win.close();
    }
  }
}

try {
  runTests();
  console.log('\nAll tests passed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
