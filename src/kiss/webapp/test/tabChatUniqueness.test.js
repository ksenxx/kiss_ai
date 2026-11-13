// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the one-tab-per-chat invariant in the
// chat page: for a given chat id, at most ONE tab may be open on a
// client.  The daemon's registry enforces uniqueness, and the page
// keeps a deterministic keep-first backstop so a legacy or buggy
// `tabs_state` snapshot carrying duplicate chat bindings can never
// render two tabs for the same chat.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  snapshotEntry,
  tabIds: tabBarIds,
} = require('./simplify2_harness');
const {runTests} = require('./workdir_helpers');

function testDuplicateChatBindingsRenderOneTab() {
  // A snapshot binding one chat to two tabs must render exactly ONE
  // tab for that chat (keep-first), never two.
  const {win} = makePage();
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
  const {win} = makePage();
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
  const {win} = makePage();
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
  const {win} = makePage();
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
  const {win} = makePage();
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

runTests(tests.map(fn => [fn.name, fn]));
