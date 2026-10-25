// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the tab context menu's "Close All" and
// "Close Inactive" in media/main.js: the tab on screen is closed LAST,
// so no tab that is about to be closed anyway is brought on screen in
// between.  On the old code the active tab was closed first, in list
// order, and every successor it picked was restored, its deferred
// transcript fetched (a `resumeSession`), and then closed on the next
// iteration.
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');
const {tabEl} = require('./subagent_tab_helpers');

const {test, report} = h.makeRunner();

function scopedFetches(posted) {
  return h
    .ofType(posted, 'resumeSession')
    .filter(m => m.scoped === true && /^tab-[bc]$/.test(m.tabId));
}

// The boot chat (tabA, first in the list, on screen) with a sub-agent
// tab so the strip is shown, plus two registry chats whose transcripts
// the daemon deferred (they are fetched the first time they come on
// screen).
function pageWithDeferredTabs() {
  const {win, posted} = h.makePage();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.endLaunch();
  h.send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: tabA, chatId: '', title: 'new chat', workDir: ''},
      {tabId: 'tab-b', chatId: 'c-b', title: 'B', workDir: ''},
      {tabId: 'tab-c', chatId: 'c-c', title: 'C', workDir: ''},
    ],
  });
  for (const id of ['tab-b', 'tab-c']) {
    h.send(win, {type: 'status', running: false, tabId: id});
    h.send(win, {
      type: 'replay_deferred',
      tabId: id,
      chatId: id === 'tab-b' ? 'c-b' : 'c-c',
      taskId: '',
    });
  }
  h.send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-a',
    parent_tab_id: tabA,
    description: 'a sub-agent',
    task_id: 'sub-task-a',
    taskIndex: 0,
    isSubagentTab: true,
  });
  assert.strictEqual(win._testApi.getActiveTabId(), tabA, 'tabA is on screen');
  assert.strictEqual(
    scopedFetches(posted).length,
    0,
    'precondition: the deferred tabs were not fetched',
  );
  return {win, posted, tabA};
}

function pickMenuItem(win, tabA, label) {
  const tab = tabEl(win, tabA);
  tab.focus();
  h.rightClick(win, tab);
  const item = h
    .all(win, '#tab-context-menu .tab-ctx-item')
    .find(el => el.textContent === label);
  assert.ok(item, `the "${label}" item exists`);
  h.click(win, item);
}

async function main() {
  await test('Close All closes the active tab last, fetching nothing', () => {
    const {win, posted, tabA} = pageWithDeferredTabs();
    pickMenuItem(win, tabA, 'Close All');
    const ids = win._testApi.openTabs().map(t => t.id);
    assert.ok(!ids.includes(tabA) && !ids.includes('tab-b'), 'all closed');
    assert.ok(!ids.includes('tab-c') && !ids.includes('sub-a'), 'all closed');
    assert.strictEqual(ids.length, 1, 'one fresh chat replaces them');
    assert.deepStrictEqual(
      scopedFetches(posted),
      [],
      'no transcript was fetched for a tab closed by the same action',
    );
    win.close();
  });

  await test('Close Inactive closes the tab on screen last', () => {
    const {win, posted, tabA} = pageWithDeferredTabs();
    pickMenuItem(win, tabA, 'Close Inactive');
    assert.strictEqual(win._testApi.openTabs().length, 1, 'one fresh chat');
    assert.deepStrictEqual(scopedFetches(posted), []);
    win.close();
  });

  await test('Close Inactive keeps a running tab, fetched only once', () => {
    const {win, posted, tabA} = pageWithDeferredTabs();
    h.send(win, {type: 'status', running: true, tabId: 'tab-c'});
    pickMenuItem(win, tabA, 'Close Inactive');
    const ids = win._testApi.openTabs().map(t => t.id);
    assert.strictEqual(ids.join(','), 'tab-c', 'only the running tab survives');
    assert.strictEqual(win._testApi.getActiveTabId(), 'tab-c');
    const fetches = scopedFetches(posted);
    assert.strictEqual(fetches.length, 1, 'the survivor is fetched once');
    assert.strictEqual(fetches[0].tabId, 'tab-c');
    win.close();
  });

  // The active tab is the sub-agent under tabA: closing tabA (earlier in
  // the batch) takes sub-a along, so the active tab goes before B and C
  // unless the batch is ordered by what each close takes along.
  function pageWithSubagentOnScreen() {
    const page = pageWithDeferredTabs();
    h.click(page.win, tabEl(page.win, 'sub-a'));
    assert.strictEqual(
      page.win._testApi.getActiveTabId(),
      'sub-a',
      'the sub-agent tab is on screen',
    );
    return page;
  }

  await test('Close All from an active sub-agent tab fetches nothing', () => {
    const {win, posted} = pageWithSubagentOnScreen();
    pickMenuItem(win, 'sub-a', 'Close All');
    const ids = win._testApi.openTabs().map(t => t.id);
    assert.strictEqual(ids.length, 1, 'one fresh chat replaces them');
    for (const id of ['tab-b', 'tab-c', 'sub-a'])
      assert.ok(!ids.includes(id), `${id} closed`);
    assert.deepStrictEqual(
      scopedFetches(posted),
      [],
      'no transcript was fetched for a tab closed by the same action',
    );
    win.close();
  });

  await test('Close Inactive from an active sub-agent tab fetches nothing', () => {
    const {win, posted, tabA} = pageWithSubagentOnScreen();
    for (const id of [tabA, 'sub-a', 'tab-b', 'tab-c'])
      h.send(win, {type: 'status', running: false, tabId: id});
    pickMenuItem(win, 'sub-a', 'Close Inactive');
    const ids = win._testApi.openTabs().map(t => t.id);
    assert.strictEqual(ids.length, 1, 'one fresh chat replaces them');
    for (const id of ['tab-b', 'tab-c', 'sub-a'])
      assert.ok(!ids.includes(id), `${id} closed`);
    assert.deepStrictEqual(scopedFetches(posted), []);
    win.close();
  });

  await test('Close Inactive keeps the running root of the active sub-agent', () => {
    const {win, posted, tabA} = pageWithSubagentOnScreen();
    h.send(win, {type: 'status', running: true, tabId: tabA});
    h.send(win, {type: 'status', running: false, tabId: 'sub-a'});
    pickMenuItem(win, 'sub-a', 'Close Inactive');
    const ids = win._testApi.openTabs().map(t => t.id);
    assert.strictEqual(ids.join(','), tabA, 'only the running root survives');
    assert.strictEqual(win._testApi.getActiveTabId(), tabA);
    assert.deepStrictEqual(scopedFetches(posted), []);
    win.close();
  });

  // "Close Others" keeps the clicked tab.  From a sub-agent tab the
  // "others" include its parent chat, and closing a chat takes its
  // descendants along: the kept tab must therefore spare its ancestors.
  await test('Close Others from a sub-agent tab keeps that tab and its chat', () => {
    const {win, tabA} = pageWithSubagentOnScreen();
    h.send(win, {type: 'status', running: true, tabId: 'sub-a'});
    pickMenuItem(win, 'sub-a', 'Close Others');
    const ids = win._testApi.openTabs().map(t => t.id);
    assert.strictEqual(ids.join(','), tabA + ',sub-a', 'only tab-b and tab-c close');
    assert.strictEqual(win._testApi.getActiveTabId(), 'sub-a', 'still on screen');
    win.close();
  });

  await test('Close Others from a root chat closes every other tab', () => {
    const {win, tabA} = pageWithDeferredTabs();
    win._testApi.switchToTab('tab-b');
    pickMenuItem(win, 'tab-b', 'Close Others');
    const ids = win._testApi.openTabs().map(t => t.id);
    assert.strictEqual(ids.join(','), 'tab-b', `${tabA}, sub-a and tab-c close`);
    assert.strictEqual(win._testApi.getActiveTabId(), 'tab-b');
    win.close();
  });

  report('tabMenuCloseAllActiveLast');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
