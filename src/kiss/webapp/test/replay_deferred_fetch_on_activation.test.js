// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for `replay_deferred`: the daemon answers a
// `ready` with the transcript of the tab on screen only and offers the
// other registry tabs' transcripts instead.  The page must
//   - settle such a tab's state (statusKnown) without a transcript,
//   - fetch the transcript with ONE connection-scoped resumeSession the
//     first time the user opens the tab (and at once when the offer is
//     for the tab already on screen),
//   - carry the task the tab is pinned to,
//   - fetch the root chat's transcript when a sub-agent tab is opened.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {sendHistory, makeRow} = require('./history_helpers');

function scopedFetches(posted, tabId) {
  return posted.filter(
    m => m.type === 'resumeSession' && m.scoped === true && m.tabId === tabId,
  );
}

/** Adopt `tab-b` (bound to chat c-b) next to the boot tab and defer it. */
function adoptDeferredTab(win, taskId) {
  const bootTab = win._testApi.getActiveTabId();
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: bootTab, chatId: '', title: 'new chat', workDir: ''},
      {tabId: 'tab-b', chatId: 'c-b', title: 'B', workDir: ''},
    ],
  });
  assert.ok(
    win._testApi.openTabs().some(t => t.id === 'tab-b'),
    'the registry tab was adopted',
  );
  send(win, {type: 'status', running: false, tabId: 'tab-b'});
  send(win, {type: 'replay_deferred', tabId: 'tab-b', chatId: 'c-b', taskId});
  return bootTab;
}

function testFetchOnFirstActivationOnly() {
  const {win, posted} = makePage();
  const bootTab = adoptDeferredTab(win, '');
  assert.strictEqual(
    scopedFetches(posted, 'tab-b').length,
    0,
    'an offer for a tab not on screen is not fetched',
  );
  win._testApi.switchToTab('tab-b');
  const fetches = scopedFetches(posted, 'tab-b');
  assert.strictEqual(fetches.length, 1, 'opening the tab fetches once');
  assert.strictEqual(fetches[0].id, 'c-b');
  assert.strictEqual(fetches[0].taskId, undefined, 'no pinned task');
  win._testApi.switchToTab(bootTab);
  win._testApi.switchToTab('tab-b');
  assert.strictEqual(
    scopedFetches(posted, 'tab-b').length,
    1,
    'a second activation does not fetch again',
  );
  // The transcript arrives; a later offer (the next reconnect) for the
  // tab on screen is fetched at once.
  send(win, {
    type: 'task_events',
    tabId: 'tab-b',
    chat_id: 'c-b',
    task_id: 't1',
    task: 'B',
    extra: '',
    events: [{type: 'prompt', text: 'hello'}],
  });
  send(win, {
    type: 'replay_deferred',
    tabId: 'tab-b',
    chatId: 'c-b',
    taskId: 't9',
  });
  const again = scopedFetches(posted, 'tab-b');
  assert.strictEqual(
    again.length,
    2,
    'an offer for the active tab is fetched now',
  );
  assert.strictEqual(
    again[1].taskId,
    't9',
    'the pinned task travels with the fetch',
  );
  win.close();
  console.log('  ok - a deferred tab is fetched once, on activation');
}

function testPinnedTaskAndUnknownTab() {
  const {win, posted} = makePage();
  adoptDeferredTab(win, 't-old');
  // An offer for a tab this page does not have is ignored.
  send(win, {
    type: 'replay_deferred',
    tabId: 'nope',
    chatId: 'c-x',
    taskId: '',
  });
  win._testApi.switchToTab('tab-b');
  const fetches = scopedFetches(posted, 'tab-b');
  assert.strictEqual(fetches.length, 1);
  assert.strictEqual(fetches[0].taskId, 't-old');
  assert.strictEqual(scopedFetches(posted, 'nope').length, 0);
  win.close();
  console.log(
    '  ok - the fetch names the pinned task; unknown tabs are ignored',
  );
}

function testActiveSubagentTabFetchesItsOwnOffer() {
  const {win, posted} = makePage();
  adoptDeferredTab(win, '');
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'tab-b__sub_1',
    parent_tab_id: 'tab-b',
    description: 'child',
    task_id: 's1',
    isSubagentTab: true,
    isDone: false,
  });
  win._testApi.switchToTab('tab-b__sub_1');
  assert.strictEqual(win._testApi.getActiveTabId(), 'tab-b__sub_1');
  assert.strictEqual(
    scopedFetches(posted, 'tab-b').length,
    1,
    "opening the child fetches the root chat's deferred transcript",
  );
  assert.strictEqual(
    scopedFetches(posted, 'tab-b__sub_1').length,
    0,
    'the child itself is not fetched as a deferred tab',
  );
  // The daemon defers each child separately: the offer for the child
  // ON SCREEN is fetched at once, with the child's own chat/task.
  send(win, {
    type: 'replay_deferred',
    tabId: 'tab-b__sub_1',
    chatId: 'c-b',
    taskId: 's1',
  });
  const own = scopedFetches(posted, 'tab-b__sub_1');
  assert.strictEqual(own.length, 1, 'an active child fetches its own offer');
  assert.strictEqual(own[0].id, 'c-b');
  assert.strictEqual(own[0].taskId, 's1');
  // An offer for the parent while the child is on screen: fetched too.
  send(win, {
    type: 'replay_deferred',
    tabId: 'tab-b',
    chatId: 'c-b',
    taskId: '',
  });
  assert.strictEqual(scopedFetches(posted, 'tab-b').length, 2);
  win.close();
  console.log('  ok - an active sub-agent tab fetches its own offer');
}

function testHistoryClickOnDeferredTabResumesOnce() {
  const {win, posted} = makePage();
  adoptDeferredTab(win, 'old-task');
  // The Chats drawer lists a task of the deferred chat; its click is
  // one explicit (broadcast) resume at that task, no deferred fetch
  // first.
  win.document.getElementById('sidebar').classList.add('open');
  sendHistory(win, posted, [
    makeRow({id: 'c-b', task_id: 'history-task', title: 'B', has_events: true}),
  ]);
  const row = win.document.querySelector('#history-list .sidebar-item');
  assert.ok(row, 'the Chats drawer lists the task');
  row.click();
  const resumes = posted.filter(
    m => m.type === 'resumeSession' && m.tabId === 'tab-b',
  );
  assert.strictEqual(resumes.length, 1, JSON.stringify(resumes));
  assert.strictEqual(resumes[0].taskId, 'history-task');
  assert.strictEqual(
    resumes[0].scoped,
    undefined,
    'a history click is broadcast',
  );
  assert.strictEqual(win._testApi.getActiveTabId(), 'tab-b');
  win.close();
  console.log('  ok - a history click on a deferred tab resumes once');
}

function testNewRunDropsThePendingFetch() {
  const {win, posted} = makePage();
  const bootTab = adoptDeferredTab(win, 'old-task');
  // A run started on the tab from another surface: its live events are
  // the transcript; the withheld (older) one must not be fetched over
  // them when the user opens the tab.
  send(win, {type: 'clear', tabId: 'tab-b', chat_id: 'c-b'});
  send(win, {type: 'status', running: true, tabId: 'tab-b', startTs: 1});
  win._testApi.switchToTab('tab-b');
  assert.strictEqual(scopedFetches(posted, 'tab-b').length, 0);
  win._testApi.switchToTab(bootTab);
  win.close();
  console.log('  ok - a new run drops the pending fetch');
}

function main() {
  testFetchOnFirstActivationOnly();
  testPinnedTaskAndUnknownTab();
  testActiveSubagentTabFetchesItsOwnOffer();
  testHistoryClickOnDeferredTabResumesOnce();
  testNewRunDropsThePendingFetch();
  console.log('all replay_deferred tests passed');
}

main();
