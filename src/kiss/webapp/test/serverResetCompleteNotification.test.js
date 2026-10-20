// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, waitFor} = require('./simplify2_harness');

const RESTARTING_MESSAGE = 'Restarting the KISS Sorcar web server…';
const COMPLETE_MESSAGE = 'KISS Sorcar web server restart complete.';

const SERVER_RESET_NOTIFICATIONS = [
  {id: 'server-reset-restarting', message: RESTARTING_MESSAGE},
  {id: 'server-reset-complete', message: COMPLETE_MESSAGE},
];

async function runTests() {
  for (const {id, message} of SERVER_RESET_NOTIFICATIONS) {
    const wv = makePage();
    try {
      send(wv.win, {type: 'notification', id, severity: 'info', message});

      const toast = await waitFor(
        () => wv.win.document.querySelector('.kiss-notification'),
        `"${id}" must render a .kiss-notification toast`,
      );

      assert.ok(
        toast.textContent.includes(message),
        `"${id}" toast must show the "${message}" message`,
      );
      assert.strictEqual(
        toast.getAttribute('data-notification-id'),
        id,
        `toast must be stamped with id="${id}" so a repeat broadcast does not stack duplicates`,
      );
      assert.ok(
        toast.classList.contains('kiss-notification-info'),
        `"${id}" toast must use the "info" severity styling`,
      );

      for (const el of wv.win.document.querySelectorAll('div.note')) {
        assert.ok(
          !el.textContent.includes('web server') &&
            !el.textContent.toLowerCase().includes('restart'),
          `"${id}" must NOT be rendered as a chat-output "Note: …" banner`,
        );
      }
    } finally {
      wv.win.close();
    }
  }

  {
    const wv = makePage();
    try {
      send(wv.win, {
        type: 'notification',
        id: 'server-reset-restarting',
        severity: 'info',
        message: RESTARTING_MESSAGE,
      });
      await waitFor(
        () =>
          wv.win.document.querySelector(
            '.kiss-notification[data-notification-id="server-reset-restarting"]',
          ),
        'pre-condition: the "restarting" toast must be visible first',
      );
      send(wv.win, {
        type: 'notification',
        id: 'server-reset-restarting',
        severity: 'info',
        message: RESTARTING_MESSAGE,
      });
      assert.strictEqual(
        wv.win.document.querySelectorAll(
          '.kiss-notification[data-notification-id="server-reset-restarting"]',
        ).length,
        1,
        'a repeat "server-reset-restarting" broadcast must not stack a duplicate toast',
      );

      send(wv.win, {
        type: 'notification',
        id: 'server-reset-complete',
        severity: 'info',
        message: COMPLETE_MESSAGE,
      });
      const completeToast = await waitFor(
        () =>
          wv.win.document.querySelector(
            '.kiss-notification[data-notification-id="server-reset-complete"]',
          ),
        'the "complete" toast must appear after the broadcast',
      );

      assert.ok(
        completeToast.textContent.includes(COMPLETE_MESSAGE),
        'the surviving toast must show the completion message',
      );

      const completeToasts = wv.win.document.querySelectorAll(
        '.kiss-notification[data-notification-id="server-reset-complete"]',
      );
      assert.strictEqual(
        completeToasts.length,
        1,
        'a repeat "server-reset-complete" broadcast must not stack a duplicate toast',
      );

      send(wv.win, {
        type: 'notification',
        id: 'server-reset-complete',
        severity: 'info',
        message: COMPLETE_MESSAGE,
      });
      assert.strictEqual(
        wv.win.document.querySelectorAll(
          '.kiss-notification[data-notification-id="server-reset-complete"]',
        ).length,
        1,
        're-broadcasting "server-reset-complete" must update the existing toast, not append a new one',
      );
    } finally {
      wv.win.close();
    }
  }

  {
    const wv = makePage();
    try {
      send(wv.win, {type: 'notice', text: COMPLETE_MESSAGE});
      const note = await waitFor(
        () =>
          Array.from(wv.win.document.querySelectorAll('div.note')).find(el =>
            el.textContent.includes(COMPLETE_MESSAGE),
          ),
        'control: a legacy `notice` event must still render as a chat-output note (proves Case A is not a false-positive)',
      );
      assert.ok(
        note.textContent.startsWith('Note:'),
        'control: legacy notice banner is prefixed with "Note:" — the surface the fix moves the post-restart message away from',
      );
      assert.strictEqual(
        wv.win.document.querySelectorAll('.kiss-notification').length,
        0,
        'control: legacy `notice` event must NOT raise a top-right notification toast',
      );
    } finally {
      wv.win.close();
    }
  }
}

runTests().then(
  () => {
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
