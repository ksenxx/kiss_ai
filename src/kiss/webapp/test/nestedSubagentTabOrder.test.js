// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) test for media/main.js: the order of nested
// sub-agent tabs is stable across registry snapshots.
//
// A sub-agent that itself runs a run_parallel opens a grandchild tab.
// Every `tabs_state` broadcast re-places each sub-agent tab after its
// parent's subtree; placing after the parent's *direct* children only
// made the strip alternate between `P, A, A1, B` and `P, A, B, A1` on
// every snapshot while the nested sub-agents ran.

'use strict';

const assert = require('assert');
const {makePage, send, readyTabId} = require('./simplify2_harness');
const {openSubTabIds, spawnSubagent} = require('./subagent_tab_helpers');

function tabOrder(win) {
  // Array.from: the page's arrays live in the JSDOM realm, whose Array
  // prototype deepStrictEqual would reject.
  return Array.from(win._testApi.openTabs(), t => String(t.id));
}

function registrySnapshot(win, rootId) {
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: rootId, chatId: 'chat-1', title: 'Root', workDir: ''}],
  });
}

function testNestedSubagentOrderIsStableAcrossSnapshots() {
  const {win, posted} = makePage();
  const rootId = readyTabId(posted);
  send(win, {type: 'status', running: true, tabId: rootId, taskId: 1});
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: rootId,
    extras: {tasks: JSON.stringify(['task-a', 'task-b'])},
  });
  const page = {win, posted};
  const a = spawnSubagent(page, rootId, 'task-a', 'sub a', 0);
  // Sub-agent A runs its own run_parallel: A1 hangs off A.
  send(win, {type: 'status', running: true, tabId: a, taskId: 2});
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: a,
    extras: {tasks: JSON.stringify(['task-a1'])},
  });
  const a1 = spawnSubagent(page, a, 'task-a1', 'sub a1', 0);
  const b = spawnSubagent(page, rootId, 'task-b', 'sub b', 1);

  const expected = [rootId, a, a1, b];
  assert.deepStrictEqual(
    tabOrder(win),
    expected,
    "a sub-agent's subtree stays contiguous: B follows A's whole subtree",
  );
  for (let i = 1; i <= 3; i++) {
    registrySnapshot(win, rootId);
    assert.deepStrictEqual(
      tabOrder(win),
      expected,
      'tab order must not change on registry snapshot #' + i,
    );
    assert.deepStrictEqual(
      openSubTabIds(win),
      [a, a1, b],
      'the strip must paint the same order on snapshot #' + i,
    );
  }
  win.close();
}

testNestedSubagentOrderIsStableAcrossSnapshots();
console.log('nestedSubagentTabOrder.test.js: ok');
