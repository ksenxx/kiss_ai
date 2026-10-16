// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, waitFor} = require('./simplify2_harness');

const SERVER_RESET_MESSAGE = 'Restarting the KISS Sorcar web server…';

async function runTests() {
  {
    const wv = makePage();
    try {
      send(wv.win, {
        type: 'notification',
        id: 'server-reset-restarting',
        severity: 'info',
        message: SERVER_RESET_MESSAGE,
      });

      const toast = await waitFor(
        () => wv.win.document.querySelector('.kiss-notification'),
        'server-reset acknowledgement must render a .kiss-notification toast',
      );

      assert.ok(
        toast.textContent.includes(SERVER_RESET_MESSAGE),
        'toast must show the "Restarting the KISS Sorcar web server…" message',
      );

      assert.strictEqual(
        toast.getAttribute('data-notification-id'),
        'server-reset-restarting',
        'toast must be stamped with id="server-reset-restarting" so a repeat broadcast does not stack duplicates',
      );
      assert.ok(
        toast.classList.contains('kiss-notification-info'),
        'toast must use the "info" severity styling',
      );

      const noteEls = wv.win.document.querySelectorAll('div.note');
      for (const el of noteEls) {
        assert.ok(
          !el.textContent.includes('web server') &&
            !el.textContent.toLowerCase().includes('restart'),
          'server-reset message must NOT be rendered as a chat-output "Note: …" banner',
        );
      }
    } finally {
      wv.win.close();
    }
  }

  {
    const wv = makePage();
    try {
      send(wv.win, {type: 'notice', text: SERVER_RESET_MESSAGE});
      const note = await waitFor(
        () =>
          Array.from(wv.win.document.querySelectorAll('div.note')).find(el =>
            el.textContent.includes(SERVER_RESET_MESSAGE),
          ),
        'control: a legacy `notice` event must still render as a chat-output note (proves Case A is not a false-positive)',
      );
      assert.ok(
        note.textContent.startsWith('Note:'),
        'control: legacy notice banner is prefixed with "Note:" — the surface the fix moves the server-reset message away from',
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
