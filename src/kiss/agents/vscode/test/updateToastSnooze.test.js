// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end webview tests for the "Remind me later" action on the
// daemon-driven sticky update toast: clicking it posts a
// `snoozeUpdate` command (VS Code: forwarded to the daemon over UDS;
// webapp: sent straight over WSS), and an `update_available` event
// carrying `snoozed: true` suppresses the toast while keeping the
// passive settings-button badge.

'use strict';

const assert = require('assert');
const {makeWebview, send} = require('./simplify2_harness');

const TOAST_SELECTOR = '[data-notification-id="kiss-update-available"]';

function actionByLabel(toast, label) {
  return Array.from(
    toast.querySelectorAll('.kiss-notification-action'),
  ).find(btn => btn.textContent.trim() === label);
}

function testRemindMeLaterPostsSnoozeAndDismisses() {
  const {win, posted} = makeWebview();
  const doc = win.document;

  send(win, {
    type: 'update_available',
    available: true,
    latest: '9.9.9',
    current: '1.0.0',
    snoozed: false,
  });
  const toast = doc.querySelector(TOAST_SELECTOR);
  assert.ok(toast, 'unsnoozed update shows the sticky toast');
  assert.ok(
    actionByLabel(toast, 'Update'),
    'toast keeps its Update action',
  );
  const remind = actionByLabel(toast, 'Remind me later');
  assert.ok(remind, 'toast offers a Remind me later action');

  remind.click();
  const snoozeMsgs = posted.filter(m => m.type === 'snoozeUpdate');
  assert.strictEqual(snoozeMsgs.length, 1, 'one snoozeUpdate posted');
  assert.strictEqual(
    snoozeMsgs[0].latest,
    '9.9.9',
    'snoozeUpdate names the snoozed release',
  );
  assert.strictEqual(
    doc.querySelector(TOAST_SELECTOR),
    null,
    'clicking Remind me later dismisses the toast locally',
  );
  assert.ok(
    !posted.some(m => m.type === 'runUpdate'),
    'Remind me later must not start an update',
  );

  win.close();
  console.log('  ok - Remind me later posts snoozeUpdate and dismisses');
}

function testSnoozedBroadcastSuppressesToastKeepsBadge() {
  const {win} = makeWebview();
  const doc = win.document;
  const btn = doc.getElementById('cfg-update-btn');

  // The daemon's post-snooze rebroadcast (or the rebroadcast a
  // reloaded window receives on connect) carries snoozed: true.
  send(win, {
    type: 'update_available',
    available: true,
    latest: '9.9.9',
    current: '1.0.0',
    snoozed: true,
  });
  assert.strictEqual(
    doc.querySelector(TOAST_SELECTOR),
    null,
    'a snoozed update must not show the sticky toast',
  );
  assert.ok(
    btn.classList.contains('has-update'),
    'the passive settings badge stays visible while snoozed',
  );

  // An already-visible toast disappears when the snooze arrives.
  send(win, {
    type: 'update_available',
    available: true,
    latest: '9.9.9',
    current: '1.0.0',
    snoozed: false,
  });
  assert.ok(doc.querySelector(TOAST_SELECTOR), 'toast shows unsnoozed');
  send(win, {
    type: 'update_available',
    available: true,
    latest: '9.9.9',
    current: '1.0.0',
    snoozed: true,
  });
  assert.strictEqual(
    doc.querySelector(TOAST_SELECTOR),
    null,
    'the snoozed rebroadcast removes the visible toast in every window',
  );
  assert.ok(
    btn.classList.contains('has-update'),
    'badge survives the snoozed rebroadcast',
  );

  // Snooze expiry: the next broadcast has snoozed: false again.
  send(win, {
    type: 'update_available',
    available: true,
    latest: '9.9.9',
    current: '1.0.0',
    snoozed: false,
  });
  assert.ok(
    doc.querySelector(TOAST_SELECTOR),
    'the toast returns once the snooze expires',
  );

  win.close();
  console.log('  ok - snoozed broadcast suppresses toast, keeps badge');
}

function testLegacyEventWithoutSnoozedFieldStillShowsToast() {
  // An older daemon does not send the `snoozed` field; the toast must
  // behave exactly as before.
  const {win} = makeWebview();
  const doc = win.document;
  send(win, {
    type: 'update_available',
    available: true,
    latest: '9.9.9',
    current: '1.0.0',
  });
  assert.ok(
    doc.querySelector(TOAST_SELECTOR),
    'missing snoozed field defaults to showing the toast',
  );
  win.close();
  console.log('  ok - legacy update_available event still shows the toast');
}

testRemindMeLaterPostsSnoozeAndDismisses();
testSnoozedBroadcastSuppressesToastKeepsBadge();
testLegacyEventWithoutSnoozedFieldStillShowsToast();
console.log('updateToastSnooze.test.js: all tests passed');
