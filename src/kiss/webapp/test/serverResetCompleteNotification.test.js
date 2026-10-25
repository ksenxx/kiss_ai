// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The web server's restart messages reach the page as `notification`
// events (web_server.py), which raise a top-right toast and never a
// chat-output "Note:" banner; the legacy `notice` event is the other way
// round.  The toast path itself (id stamp, severity class, same-id
// updates in place) is notifications.test.js.

'use strict';

const assert = require('assert');
const {makePage, send, waitFor} = require('./simplify2_harness');

const COMPLETE_MESSAGE = 'KISS Sorcar web server restart complete.';

function notes(win) {
  return Array.from(win.document.querySelectorAll('div.note')).filter(el =>
    el.textContent.includes(COMPLETE_MESSAGE),
  );
}

async function main() {
  const {win} = makePage();
  send(win, {
    type: 'notification',
    id: 'server-reset-complete',
    severity: 'info',
    message: COMPLETE_MESSAGE,
  });
  const toast = await waitFor(
    () => win.document.querySelector('.kiss-notification'),
    'a server-reset notification must render a toast',
  );
  assert.ok(toast.textContent.includes(COMPLETE_MESSAGE));
  assert.strictEqual(
    notes(win).length,
    0,
    'a notification must NOT be rendered as a chat-output "Note:" banner',
  );
  win.close();

  const control = makePage();
  send(control.win, {type: 'notice', text: COMPLETE_MESSAGE});
  const note = await waitFor(
    () => notes(control.win)[0],
    'control: a legacy `notice` event still renders as a chat-output note',
  );
  assert.ok(note.textContent.startsWith('Note:'));
  assert.strictEqual(
    control.win.document.querySelectorAll('.kiss-notification').length,
    0,
    'control: a legacy `notice` event raises no toast',
  );
  control.win.close();
  console.log('serverResetCompleteNotification: all passed.');
}

main().catch(err => {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
});
