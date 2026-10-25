// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, console */

// Split layout (desktop remote page): a file opened in the background
// ("Open to the Side", or a hidden chat's task opening a file) while the
// content pane is empty.  The pane opens for the first content tab either
// way, so its row and its body must agree: the new file is the pane's
// selected tab and shown, not "No open files." under a row of open files.
// The chat on screen is kept.  Once the pane shows a file, a background
// open only adds a row (the shown file stays).

const assert = require('assert');
const {makePage, send, fakeMatchMedia} = require('./simplify2_harness');

function stubDesktop(win) {
  win.Element.prototype.setPointerCapture = function () {};
  win.Element.prototype.releasePointerCapture = function () {};
  fakeMatchMedia(win, true);
}

function makeSplitPage() {
  const {win} = makePage({
    bodyClass: 'remote-chat',
    extraScripts: ['browserTab.js'],
    beforeScripts: stubDesktop,
  });
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 'a1', chatId: 'chat-1', title: 'a1', workDir: '/ws'},
      {tabId: 'a2', chatId: 'chat-2', title: 'a2', workDir: '/ws'},
    ],
  });
  send(win, {type: 'task_events', tabId: 'a1', task: 'the task', events: []});
  assert.ok(win.document.body.classList.contains('remote-desktop'));
  assert.strictEqual(win._testApi.getActiveTabId(), 'a1');
  return {win};
}

function paneRows(win) {
  return Array.from(
    win.document.querySelectorAll('#content-tab-list .chat-tab'),
  );
}

function activeRow(win) {
  return win.document.querySelector('#content-tab-list .chat-tab.active');
}

function fileTab(win, p) {
  return Array.from(win._testApi.openTabs()).find(
    t => t.isContentTab && t.contentPath === p,
  );
}

function viewShown(win, tab) {
  const view = win.document.querySelector(
    '.content-tab-view[data-tab-id="' + tab.id + '"]',
  );
  return !!view && view.style.display !== 'none';
}

function assertPaneAgrees(win, p, label) {
  const tab = fileTab(win, p);
  assert.ok(tab, label + ': the file tab exists');
  const row = activeRow(win);
  assert.ok(row, label + ': the pane has a selected row');
  assert.strictEqual(row.dataset.tabId, tab.id, label + ': it is the file');
  const empty = win.document.getElementById('content-pane-empty');
  assert.strictEqual(
    empty.style.display,
    'none',
    label + ': "No open files." is hidden',
  );
  assert.ok(viewShown(win, tab), label + ': the file view is shown');
  assert.ok(
    win.document.body.classList.contains('content-pane-open'),
    label + ': the pane is open',
  );
  return tab;
}

function testOpenToTheSideIntoEmptyPane() {
  const {win} = makeSplitPage();
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    background: true,
    path: '/ws/x.py',
    name: 'x.py',
    content: 'print(1)',
    version: 'v1',
  });
  assertPaneAgrees(win, '/ws/x.py', 'Open to the Side');
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    'a1',
    'the chat on screen is kept',
  );
  // A second background open while a file is shown only adds a row.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    background: true,
    path: '/ws/y.py',
    name: 'y.py',
    content: 'print(2)',
    version: 'v1',
  });
  assert.strictEqual(paneRows(win).length, 2, 'two rows');
  assertPaneAgrees(win, '/ws/x.py', 'second background open');
  assert.ok(!viewShown(win, fileTab(win, '/ws/y.py')), 'y.py is not shown');
  // The same file opened again in the background: still no switch.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    background: true,
    path: '/ws/y.py',
    name: 'y.py',
    content: 'print(2)',
    version: 'v1',
  });
  assert.strictEqual(paneRows(win).length, 2, 'still two rows');
  assertPaneAgrees(win, '/ws/x.py', 'background reopen');
  win.close();
  console.log('  ok - Open to the Side into an empty pane shows the file');
}

function testHiddenChatOpenIntoEmptyPane() {
  const {win} = makeSplitPage();
  send(win, {
    type: 'fileContent',
    tabId: 'a2',
    path: '/ws/y.py',
    name: 'y.py',
    content: 'print(2)',
    version: 'v1',
  });
  assertPaneAgrees(win, '/ws/y.py', "hidden chat's open");
  assert.strictEqual(win._testApi.getActiveTabId(), 'a1', 'a1 stays shown');
  win.close();
  console.log("  ok - a hidden chat's open into an empty pane shows the file");
}

function testBrowserTabIntoEmptyPane() {
  // A browser page announced without focus (the daemon's snapshot on a
  // reconnect) is the pane's first content tab: shown, not listed over
  // "No open files.".
  const {win} = makeSplitPage();
  send(win, {
    type: 'browserTabs',
    tabs: [{tab_id: 'bt1', url: 'https://x.test/', title: 'X', focus: false}],
  });
  const row = activeRow(win);
  assert.ok(row, 'the browser tab is the selected row');
  assert.strictEqual(row.dataset.tabId, 'bt1');
  assert.strictEqual(
    win.document.getElementById('content-pane-empty').style.display,
    'none',
  );
  assert.strictEqual(win._testApi.getActiveTabId(), 'a1', 'a1 stays shown');
  win.close();
  console.log('  ok - an unfocused browser tab into an empty pane is shown');
}

function testReopenAfterPaneEmptied() {
  // The pane was emptied by closing the shown file; a hidden chat then
  // reopens the file in the background: the pane shows it again.
  const {win} = makeSplitPage();
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/x.py',
    name: 'x.py',
    content: 'print(1)',
    version: 'v1',
  });
  assertPaneAgrees(win, '/ws/x.py', 'foreground open');
  activeRow(win)
    .querySelector('.chat-tab-close')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(paneRows(win).length, 0, 'the pane is empty');
  assert.ok(!activeRow(win));
  send(win, {
    type: 'fileContent',
    tabId: 'a2',
    path: '/ws/x.py',
    name: 'x.py',
    content: 'print(1)',
    version: 'v1',
  });
  assertPaneAgrees(win, '/ws/x.py', 'background reopen after close');
  win.close();
  console.log('  ok - a background reopen into an emptied pane shows the file');
}

testOpenToTheSideIntoEmptyPane();
testHiddenChatOpenIntoEmptyPane();
testBrowserTabIntoEmptyPane();
testReopenAfterPaneEmptied();
