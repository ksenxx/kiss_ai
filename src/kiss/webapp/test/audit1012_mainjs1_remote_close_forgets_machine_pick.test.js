// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Audit 2026-10-12 (mainjs-1): a chat tab closed on ANOTHER surface
// (it vanishes from the daemon's `tabs_state` snapshot, reconcileTabs
// drops it) must forget its unsent machine pick exactly like a tab
// closed here does (closeTab deletes the tabMachinePicks entry).
// Otherwise every remotely closed tab leaves a pick behind in the
// persisted page state, forever: the map is written back on every
// persistTabState and re-read on every reload.
'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  pickMachine: pick,
} = require('./simplify2_harness');
const {MACHINES} = require('./settings_helpers');

function makePage(state) {
  const {win, posted, getState} = makeChatPage({
    bodyClass: 'remote-chat',
    initialState: state || undefined,
  });
  return {win, posted, state: getState};
}

function persistedPicks(state) {
  return JSON.parse(JSON.stringify(state().tabMachinePicks));
}

function testRemotelyClosedTabForgetsItsPick() {
  const {win, state} = makePage();
  const api = win._testApi;
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  const first = api.getActiveTabId();
  api.createNewTab();
  const second = api.getActiveTabId();
  // The second chat ran a task here; gpu-box is picked for its next.
  win.document.getElementById('task-input').value = 'two';
  win.document.getElementById('send-btn').click();
  send(win, {type: 'status', running: false, tabId: second});
  pick(win, 'gpu-box');
  // The registry lists both tabs: both are now mirrored tabs.
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: first, title: 'one'},
      {tabId: second, title: 'two'},
    ],
  });
  assert.deepStrictEqual(persistedPicks(state), {[second]: 'gpu-box'});
  // Another surface closes the second tab: the next snapshot omits it.
  send(win, {type: 'tabs_state', tabs: [{tabId: first, title: 'one'}]});
  assert.ok(!api.openTabs().some(t => t.id === second), 'the tab is gone');
  assert.deepStrictEqual(
    persistedPicks(state),
    {},
    'a tab closed on another surface forgets its machine pick',
  );
  win.close();
  console.log('  ok - a tab closed on another surface forgets its pick');
}

try {
  testRemotelyClosedTabForgetsItsPick();
  console.log('\n1 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
