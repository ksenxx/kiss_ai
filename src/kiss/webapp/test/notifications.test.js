// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The daemon's `notification` events render as in-page toasts in
// media/main.js: one toast per id (a repeat updates it in place), a
// severity class, a close button that dismisses it, action buttons that
// dismiss it when clicked, and a `close: true` event that retires it.
// The old host-side half (VS Code's native notification API being
// bypassed while the webview was live) has no subject any more: the
// only surface is this page.

'use strict';

const assert = require('assert');
const {makePage, send, waitFor, click} = require('./simplify2_harness.js');
const {toast: toastById} = require('./ui_antipattern_harness');

async function main() {
  const {win} = makePage();

  send(win, {
    type: 'notification',
    id: 'choose-1',
    severity: 'warning',
    message: 'Choose an API key action.',
    actions: ['Enter Key', 'Skip'],
    sticky: true,
  });
  let toast = await waitFor(
    () => toastById(win, 'choose-1'),
    'notification toast was not rendered',
  );
  assert.ok(
    toast.classList.contains('kiss-notification-warning'),
    'the toast carries its severity class',
  );
  assert.ok(toast.textContent.includes('Choose an API key action.'));
  assert.strictEqual(
    toast.querySelectorAll('.kiss-notification-action').length,
    2,
    'one button per action',
  );
  const closeButton = toast.querySelector('.kiss-notification-close');
  assert.ok(closeButton, 'notification close button should exist');
  click(win, closeButton);
  await waitFor(
    () => !toastById(win, 'choose-1'),
    'the close button must dismiss the toast',
  );

  send(win, {
    type: 'notification',
    id: 'choose-2',
    severity: 'info',
    message: 'Choose another action.',
    actions: ['Apply'],
    sticky: true,
  });
  toast = await waitFor(
    () => toastById(win, 'choose-2'),
    'second notification toast was not rendered',
  );
  const actionButton = toast.querySelector('.kiss-notification-action');
  assert.ok(actionButton, 'notification action button should exist');
  assert.strictEqual(actionButton.textContent.trim(), 'Apply');
  click(win, actionButton);
  await waitFor(
    () => !toastById(win, 'choose-2'),
    'clicking an action button must dismiss the toast',
  );

  // A repeated id updates the toast in place instead of stacking.
  send(win, {
    type: 'notification',
    id: 'progress-1',
    severity: 'info',
    message: 'Installing (1/3)',
    progress: true,
  });
  send(win, {
    type: 'notification',
    id: 'progress-1',
    severity: 'info',
    message: 'Installing (2/3)',
    progress: true,
  });
  const progressToasts = win.document.querySelectorAll(
    '.kiss-notification[data-notification-id="progress-1"]',
  );
  assert.strictEqual(progressToasts.length, 1, 'same id: one toast');
  assert.ok(progressToasts[0].textContent.includes('Installing (2/3)'));
  assert.ok(
    progressToasts[0].querySelector('.kiss-notification-progress'),
    'a progress toast shows its progress bar',
  );
  send(win, {type: 'notification', id: 'progress-1', close: true});
  assert.strictEqual(
    toastById(win, 'progress-1'),
    null,
    'a close event retires the toast',
  );
  win.close();
  console.log('notifications: all passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
