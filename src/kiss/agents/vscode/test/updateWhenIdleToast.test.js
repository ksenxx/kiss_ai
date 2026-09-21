// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end webview tests for the "Update when idle" action on the
// daemon-driven sticky update toast, and for the toast being confined
// to chat webviews: the History (left) and Task Info (right) side
// panels of editor-tabs mode run the same main.js but must not show it.

'use strict';

const assert = require('assert');
const {makeWebview, send} = require('./simplify2_harness');

const TOAST_SELECTOR = '[data-notification-id="kiss-update-available"]';

const AVAILABLE = {
  type: 'update_available',
  available: true,
  latest: '9.9.9',
  current: '1.0.0',
  snoozed: false,
};

function actionLabels(toast) {
  return Array.from(toast.querySelectorAll('.kiss-notification-action')).map(
    btn => btn.textContent.trim(),
  );
}

function actionByLabel(toast, label) {
  return Array.from(toast.querySelectorAll('.kiss-notification-action')).find(
    btn => btn.textContent.trim() === label,
  );
}

function testUpdateWhenIdlePostsCommand() {
  const {win, posted} = makeWebview();
  const doc = win.document;

  send(win, AVAILABLE);
  const toast = doc.querySelector(TOAST_SELECTOR);
  assert.ok(toast, 'chat webview shows the sticky update toast');
  assert.deepStrictEqual(
    actionLabels(toast),
    ['Update', 'Update when idle', 'Remind me later'],
    'toast offers Update, Update when idle and Remind me later',
  );

  actionByLabel(toast, 'Update when idle').click();
  const armed = posted.filter(m => m.type === 'updateWhenIdle');
  assert.strictEqual(armed.length, 1, 'one updateWhenIdle posted');
  assert.strictEqual(armed[0].cancel, undefined, 'arming carries no cancel');
  assert.ok(
    !posted.some(m => m.type === 'runUpdate'),
    'Update when idle must not start the update right away',
  );
  assert.ok(
    !posted.some(m => m.type === 'snoozeUpdate'),
    'Update when idle must not snooze the notification',
  );

  win.close();
  console.log('  ok - Update when idle posts updateWhenIdle');
}

function testPendingIdleBroadcastShowsArmedToast() {
  const {win, posted} = makeWebview();
  const doc = win.document;

  send(win, AVAILABLE);
  send(win, Object.assign({}, AVAILABLE, {pendingIdle: true}));
  const toasts = doc.querySelectorAll(TOAST_SELECTOR);
  assert.strictEqual(
    toasts.length,
    1,
    'the rebroadcast updates the same toast',
  );
  const toast = toasts[0];
  assert.ok(
    toast.textContent.includes('KISS Sorcar 9.9.9 is available'),
    'armed toast still names the release',
  );
  assert.ok(
    toast.textContent.includes('installed when no task is running'),
    'armed toast explains when the update will run',
  );
  assert.deepStrictEqual(
    actionLabels(toast),
    ['Update now', 'Cancel'],
    'armed toast offers Update now and Cancel',
  );

  actionByLabel(toast, 'Cancel').click();
  const cancels = posted.filter(m => m.type === 'updateWhenIdle');
  assert.strictEqual(cancels.length, 1, 'one updateWhenIdle posted');
  assert.strictEqual(cancels[0].cancel, true, 'Cancel posts cancel: true');

  // The daemon's disarmed rebroadcast restores the default toast.
  send(win, Object.assign({}, AVAILABLE, {pendingIdle: false}));
  const restored = doc.querySelector(TOAST_SELECTOR);
  assert.ok(restored, 'disarmed rebroadcast shows the toast again');
  assert.deepStrictEqual(
    actionLabels(restored),
    ['Update', 'Update when idle', 'Remind me later'],
    'default actions are back after cancelling',
  );

  send(win, Object.assign({}, AVAILABLE, {pendingIdle: true}));
  posted.length = 0;
  actionByLabel(doc.querySelector(TOAST_SELECTOR), 'Update now').click();
  // In VS Code runUpdate is handled by the extension host and never
  // reaches the daemon, so the armed poller must be called off first.
  assert.deepStrictEqual(
    posted.map(m => m.type),
    ['updateWhenIdle', 'runUpdate'],
    'Update now on the armed toast cancels the idle update, then updates',
  );
  assert.strictEqual(posted[0].cancel, true, 'the cancel carries cancel: true');

  // The plain Update action (nothing armed) does not send a cancel.
  send(win, Object.assign({}, AVAILABLE, {pendingIdle: false}));
  posted.length = 0;
  actionByLabel(doc.querySelector(TOAST_SELECTOR), 'Update').click();
  assert.deepStrictEqual(
    posted.map(m => m.type),
    ['runUpdate'],
    'plain Update posts only runUpdate',
  );

  win.close();
  console.log('  ok - pendingIdle broadcast shows the armed toast');
}

function testPendingIdleOutlivesSnooze() {
  const {win} = makeWebview();
  const doc = win.document;

  send(win, Object.assign({}, AVAILABLE, {snoozed: true, pendingIdle: true}));
  assert.ok(
    doc.querySelector(TOAST_SELECTOR),
    'an armed idle update stays visible even while snoozed',
  );
  send(win, Object.assign({}, AVAILABLE, {snoozed: true, pendingIdle: false}));
  assert.strictEqual(
    doc.querySelector(TOAST_SELECTOR),
    null,
    'a plain snooze still hides the toast',
  );
  send(
    win,
    Object.assign({}, AVAILABLE, {available: false, pendingIdle: true}),
  );
  assert.strictEqual(
    doc.querySelector(TOAST_SELECTOR),
    null,
    'no toast without an available release',
  );

  win.close();
  console.log('  ok - pendingIdle outlives a snooze');
}

function testSidePanelsNeverShowTheToast() {
  ['history-panel-mode', 'meta-panel-mode'].forEach(mode => {
    const {win} = makeWebview({
      beforeScripts: w => {
        w.document.body.classList.add('editor-tab-mode', mode);
      },
    });
    const doc = win.document;
    send(win, AVAILABLE);
    assert.strictEqual(
      doc.querySelector(TOAST_SELECTOR),
      null,
      mode + ' must not show the update toast',
    );
    send(win, Object.assign({}, AVAILABLE, {pendingIdle: true}));
    assert.strictEqual(
      doc.querySelector(TOAST_SELECTOR),
      null,
      mode + ' must not show the armed update toast either',
    );
    win.close();
  });
  console.log('  ok - History and Task Info panels never show the toast');
}

function testSidebarChatStillShowsTheToast() {
  // The secondary-sidebar chat view (editor-tabs mode off) is a chat
  // webview: it keeps the toast.
  const {win} = makeWebview();
  send(win, AVAILABLE);
  assert.ok(
    win.document.querySelector(TOAST_SELECTOR),
    'the chat webview keeps the update toast',
  );
  win.close();
  console.log('  ok - chat webview keeps the toast');
}

console.log('updateWhenIdleToast');
testUpdateWhenIdlePostsCommand();
testPendingIdleBroadcastShowsArmedToast();
testPendingIdleOutlivesSnooze();
testSidePanelsNeverShowTheToast();
testSidebarChatStillShowsTheToast();
