// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the one-tab-per-chat invariant in the
// chat webview: for a given chat id, at most ONE tab may be open on a
// client.  The daemon's registry enforces uniqueness, and the webview
// keeps a deterministic keep-first backstop so a legacy or buggy
// `tabs_state` snapshot carrying duplicate chat bindings can never
// render two tabs for the same chat.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(initialState) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let state = initialState;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function tabBarIds(win) {
  return Array.from(win.document.querySelectorAll('.chat-tab'))
    .filter(el => !!el.dataset.tabId)
    .map(el => el.dataset.tabId);
}

function snapshotEntry(tabId, title, chatId) {
  return {
    tabId: tabId,
    chatId: chatId || '',
    title: title || 'new chat',
    workDir: '',
  };
}

function testDuplicateChatBindingsRenderOneTab() {
  // A snapshot binding one chat to two tabs must render exactly ONE
  // tab for that chat (keep-first), never two.
  const {win} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [
      snapshotEntry('t1', 'first', 'chat-A'),
      snapshotEntry('t2', 'dup of first', 'chat-A'),
      snapshotEntry('t3', 'other', 'chat-B'),
    ],
  });
  assert.deepStrictEqual(
    tabBarIds(win),
    ['t1', 't3'],
    'INVARIANT VIOLATED: chat-A is shown in two open tabs',
  );
}

function testDuplicateBindingArrivingLaterClosesExtraTab() {
  // A tab adopted earlier must be dropped when a later snapshot binds
  // its chat to an earlier-listed tab (the local tab bar converges to
  // one tab per chat).
  const {win} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [
      snapshotEntry('t1', 'one', 'chat-A'),
      snapshotEntry('t2', 'two', ''),
    ],
  });
  assert.deepStrictEqual(tabBarIds(win), ['t1', 't2']);
  send(win, {
    type: 'tabs_state',
    tabs: [
      snapshotEntry('t1', 'one', 'chat-A'),
      snapshotEntry('t2', 'two', 'chat-A'),
    ],
  });
  assert.deepStrictEqual(
    tabBarIds(win),
    ['t1'],
    'the duplicate binding must close the extra tab, not keep it',
  );
}

function testUnboundTabsAreNeverDeduped() {
  // Tabs without a chat binding ("new chat" tabs) may coexist freely.
  const {win} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [
      snapshotEntry('t1', 'one', ''),
      snapshotEntry('t2', 'two', ''),
      snapshotEntry('t3', 'three', ''),
    ],
  });
  assert.deepStrictEqual(tabBarIds(win), ['t1', 't2', 't3']);
}

function testPendingOpenDuplicateIsDroppedImmediately() {
  // A locally created tab (openTab still pending) whose id arrives in
  // a snapshot as a DUPLICATE chat binding must be dropped right away:
  // the daemon confirmed the id, so the pending shield must not keep
  // a second tab for the chat alive until the shield expires.
  const {win} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one', 'chat-A')],
  });
  win._testApi.createNewTab();
  const fresh = win._testApi.getActiveTabId();
  send(win, {
    type: 'tabs_state',
    tabs: [
      snapshotEntry('t1', 'one', 'chat-A'),
      snapshotEntry(fresh, 'dup', 'chat-A'),
    ],
  });
  assert.deepStrictEqual(
    tabBarIds(win),
    ['t1'],
    'a pending-open duplicate tab must not survive reconciliation',
  );
}

function testHistoryDisplacementSnapshotMovesChat() {
  // The daemon's displacement flow: chat-A moves from t1 to t9 (the
  // newest bind wins server-side).  The client must end up with ONE
  // tab for chat-A — the new one.
  const {win} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one', 'chat-A')],
  });
  assert.deepStrictEqual(tabBarIds(win), ['t1']);
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t9', 'one moved', 'chat-A')],
  });
  assert.deepStrictEqual(tabBarIds(win), ['t9']);
}

const tests = [
  testDuplicateChatBindingsRenderOneTab,
  testDuplicateBindingArrivingLaterClosesExtraTab,
  testUnboundTabsAreNeverDeduped,
  testPendingOpenDuplicateIsDroppedImmediately,
  testHistoryDisplacementSnapshotMovesChat,
];

let failures = 0;
for (const t of tests) {
  try {
    t();
    console.log('PASS', t.name);
  } catch (err) {
    failures += 1;
    console.error('FAIL', t.name);
    console.error(err && err.stack ? err.stack : err);
  }
}
if (failures > 0) {
  console.error(`${failures} test(s) failed`);
  process.exit(1);
}
console.log(`All ${tests.length} tabChatUniqueness tests passed`);
