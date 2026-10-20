// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the chat page's tab model
// (media/main.js): the daemon's whole tab registry is mirrored and
// chats are switched in place.
//
// Covered behavior:
//  - boot announces one new chat in `ready` (no `singleTabId`: the
//    daemon replays every bound tab);
//  - `tabs_state` reconciliation adopts every registry tab, and a
//    snapshot without a tab drops it (closed elsewhere) while the chat
//    on screen stays;
//  - the "+" button on a used chat opens a new chat in place;
//  - sub-agent tabs: an own chat's sub-agent gets an internal tab, a
//    foreign chat's sub-agent is ignored, a parentless spawn (a
//    run_agent sub-task) is adopted.
//
// The old host-side messages this suite also covered (`panelTitle`,
// `openChatFromHistory`, `openSettings`, the editor-tabs toggle) were
// the VS Code editor tab's and have no subject any more.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness.js');
const {ofType} = require('./simplify2_harness');

function testBootAnnouncesOneNewChat() {
  const {win, posted} = makePage();
  const readies = ofType(posted, 'ready');
  assert.strictEqual(readies.length, 1, 'exactly one ready');
  assert.strictEqual(readies[0].tabId, win._testApi.getActiveTabId());
  assert.strictEqual(
    readies[0].singleTabId,
    undefined,
    'the page mirrors the whole registry: no single tab',
  );
  assert.strictEqual(win._testApi.openTabs().length, 1);
  assert.strictEqual(win._testApi.openTabs()[0].title, 'new chat');
  win.close();
}

function testReconcileAdoptsEveryRegistryTab() {
  const {win} = makePage();
  const boot = win._testApi.getActiveTabId();
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: boot, chatId: 'chat-1', title: 'Renamed chat', workDir: ''},
      {tabId: 'other-tab', chatId: 'chat-2', title: 'Other', workDir: ''},
    ],
  });
  const ids = win._testApi.openTabs().map(t => String(t.id));
  assert.strictEqual(
    ids.sort().join(','),
    [boot, 'other-tab'].sort().join(','),
    "another client's tab is adopted",
  );
  const titleOf = id =>
    win._testApi.openTabs().find(t => String(t.id) === id).title;
  assert.strictEqual(
    titleOf(boot),
    'Renamed chat',
    'the registry title of the chat on screen is taken',
  );
  assert.strictEqual(titleOf('other-tab'), 'Other');

  // The status follows the tab it is addressed to.
  win._testApi.switchToTab('other-tab');
  send(win, {type: 'status', running: true, tabId: 'other-tab', taskId: 7});
  assert.strictEqual(
    win._testApi.openTabs().find(t => t.id === 'other-tab').isRunning,
    true,
  );
  win._testApi.switchToTab(boot);
  assert.notStrictEqual(
    win._testApi.openTabs().find(t => t.id === boot).isRunning,
    true,
  );

  // A snapshot without the other tab drops it (closed elsewhere); the
  // chat on screen stays.
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: boot, chatId: 'chat-1', title: 'Renamed chat', workDir: ''}],
  });
  assert.strictEqual(
    win._testApi
      .openTabs()
      .map(t => String(t.id))
      .join(','),
    boot,
  );
  assert.strictEqual(win._testApi.getActiveTabId(), boot);
  win.close();
}

function testNewChatOpensAChatInPlace() {
  const {win, posted} = makePage();
  const boot = win._testApi.getActiveTabId();
  send(win, {type: 'setTaskText', text: 'first task'});
  win.document
    .getElementById('new-chat-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.notStrictEqual(
    win._testApi.getActiveTabId(),
    boot,
    '"+" on a used chat opens a new chat in this page',
  );
  assert.ok(ofType(posted, 'newChat').length >= 1, 'the daemon is told');
  const active = win._testApi
    .openTabs()
    .find(t => t.id === win._testApi.getActiveTabId());
  assert.strictEqual(active.title, 'new chat');
  win.close();
}

function testSubagentTabs() {
  const {win, posted} = makePage();
  const boot = win._testApi.getActiveTabId();
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-1',
    parent_tab_id: 'not-in-this-page',
    description: 'foreign sub-agent',
    task_id: 'task-x',
  });
  assert.strictEqual(
    win._testApi.openTabs().length,
    1,
    "an unknown chat's sub-agent is ignored",
  );
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-2',
    parent_tab_id: boot,
    description: 'own sub-agent',
    task_id: 'task-y',
  });
  assert.strictEqual(
    win._testApi.openTabs().filter(t => t.isSubagentTab).length,
    1,
    "the chat's sub-agent gets an internal tab",
  );
  assert.strictEqual(
    win.document.getElementById('tab-bar').style.display,
    '',
    'tab bar appears when a sub-agent tab joins the chat',
  );
  // A parentless spawn (a run_agent sub-task) is the chat surface's
  // to adopt.
  send(win, {type: 'new_tab', task_id: 'task-orphan', parent_tab_id: ''});
  assert.strictEqual(
    win._testApi.openTabs().filter(t => t.isSubagentTab).length,
    2,
    'the chat surface adopts a parentless spawn',
  );
  assert.ok(ofType(posted, 'resumeSession').length >= 1);
  win.close();
}

function main() {
  const tests = [
    testBootAnnouncesOneNewChat,
    testReconcileAdoptsEveryRegistryTab,
    testNewChatOpensAChatInPlace,
    testSubagentTabs,
  ];
  for (const t of tests) {
    t();
    console.log(`  ok - ${t.name}`);
  }
  console.log('registryReconciliation: all tests passed');
}

main();
