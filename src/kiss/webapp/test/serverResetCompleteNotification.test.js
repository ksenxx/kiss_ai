// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, waitFor} = require('./simplify2_harness');

const RESTARTING_MESSAGE = 'Restarting the KISS Sorcar web server…';
const COMPLETE_MESSAGE = 'KISS Sorcar web server restart complete.';

async function runTests() {
  {
    const wv = makePage();
    try {
      send(wv.win, {
        type: 'notification',
        id: 'server-reset-complete',
        severity: 'info',
        message: COMPLETE_MESSAGE,
      });

      const toast = await waitFor(
        () =>
          wv.win.document.querySelector(
            '.kiss-notification[data-notification-id="server-reset-complete"]',
          ),
        '"Server restart complete" must render a .kiss-notification toast',
      );

      assert.ok(
        toast.textContent.includes(COMPLETE_MESSAGE),
        'toast must show the "KISS Sorcar web server restart complete." message',
      );

      assert.strictEqual(
        toast.getAttribute('data-notification-id'),
        'server-reset-complete',
        'toast must be stamped with id="server-reset-complete" so a repeat broadcast does not stack duplicates',
      );
      assert.ok(
        toast.classList.contains('kiss-notification-info'),
        'toast must use the "info" severity styling',
      );

      const noteEls = wv.win.document.querySelectorAll('div.note');
      for (const el of noteEls) {
        assert.ok(
          !el.textContent.includes(COMPLETE_MESSAGE),
          '"Server restart complete" must NOT be rendered as a chat-output "Note: …" banner',
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
