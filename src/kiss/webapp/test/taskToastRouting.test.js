// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A task's outcome toast belongs to the task's chat, never to another one.
//
// The daemon's `notification` events carry the tab id of the task they
// report on (merge results, dropped submits); an untagged one is
// window-level (install progress, updates).  The page holds the whole
// tab registry, so media/main.js gates each toast by its tag:
//  - a tagged toast shows over its own chat only, and is dropped when the
//    page does not hold that tab at all;
//  - the same toast shows once its tab comes on screen;
//  - an untagged toast always shows;
//  - a tagged close retires a sticky progress toast even while another
//    tab is on screen (the close is never gated).

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness.js');

function toastText(win) {
  const container = win.document.getElementById('kiss-notification-container');
  return container ? container.textContent : '';
}

function main() {
  const {win} = makePage({beforeScripts: stubRaf});
  const api = win._testApi;
  const first = api.getActiveTabId();
  api.createNewTab();
  const second = api.getActiveTabId();
  assert.notStrictEqual(first, second);

  const commitToast = {
    type: 'notification',
    id: 'toast-a-commit',
    severity: 'info',
    message: 'Merged task A: TOAST_A_COMMIT',
    tabId: first,
  };
  send(win, commitToast);
  assert.ok(
    !toastText(win).includes('TOAST_A_COMMIT'),
    "a background tab's commit toast must not show over the tab on screen",
  );
  // A toast for a tab this page does not hold at all shows nowhere.
  send(win, {
    type: 'notification',
    id: 'toast-other',
    severity: 'error',
    message: 'TOAST_OTHER_TAB',
    tabId: 'tab-gone',
  });
  assert.ok(
    !toastText(win).includes('TOAST_OTHER_TAB'),
    'a toast tagged with an unknown tab is dropped by the page',
  );
  // Chats have no tab row of their own: switch as a Chats-panel pick
  // would.
  assert.ok(
    api.openTabs().some(t => t.id === first),
    'the first chat must still be open',
  );
  api.switchToTab(first);
  assert.strictEqual(api.getActiveTabId(), first);
  send(win, {...commitToast, id: commitToast.id + '-again'});
  assert.ok(
    toastText(win).includes('TOAST_A_COMMIT'),
    'the same toast shows once its own tab is on screen',
  );
  // A window-level toast shows whichever tab is on screen.
  send(win, {
    type: 'notification',
    id: 'toast-window',
    severity: 'info',
    message: 'TOAST_WINDOW_LEVEL',
  });
  assert.ok(
    toastText(win).includes('TOAST_WINDOW_LEVEL'),
    'an untagged toast is window-level and always shows',
  );

  // A sticky progress toast opened while its tab was on screen must still
  // close after the user switches tabs: the close is tagged with the
  // (now background) tab and shows nothing, so it is never gated.
  send(win, {
    type: 'notification',
    id: 'progress-first',
    tabId: first,
    severity: 'info',
    message: 'TOAST_FIRST_PROGRESS',
    progress: true,
    sticky: true,
  });
  assert.ok(toastText(win).includes('TOAST_FIRST_PROGRESS'));
  api.switchToTab(second);
  assert.strictEqual(api.getActiveTabId(), second);
  send(win, {
    type: 'notification',
    id: 'progress-first',
    tabId: first,
    close: true,
  });
  assert.ok(
    !toastText(win).includes('TOAST_FIRST_PROGRESS'),
    'a tagged close must retire the toast even while another tab is active',
  );
  win.close();
  console.log('taskToastRouting: OK');
}

main();
