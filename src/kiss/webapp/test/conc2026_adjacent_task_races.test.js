// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for adjacent-task / background-status races
// in media/main.js (audit candidates CAND-1..CAND-4). Each test asserts
// the CORRECT (post-fix) behavior:
//
//   CAND-1  After a neighbouring task W was spliced into tab A's
//           transcript, switching to tab B and back to A and then
//           delivering another adjacent_task_events for W must NOT
//           create a second .adjacent-task[data-task-id=W] container:
//           exactly one region per task id. (resetAdjacentState rewinds
//           the anchors on every tab switch while the spliced DOM
//           survives in the tab's fragment, and renderAdjacentTask
//           never dedups by task id.)
//
//   CAND-2  When a background tab's event swaps the visible tab
//           mid-event (a bg `summary` tool_call adopts a finished
//           run_parallel panel, collapses it, and closeTab() closes the
//           ACTIVE sub-agent tab, a finished one the user reopened),
//           processOutputEventForBgTab must NOT overwrite the newly
//           restored tab's status row texts / stepCount with the
//           pre-event values of the tab that was closed: the row keeps
//           the swapped-in tab's own values.
//
//   CAND-3  An #adjacent-loader that was detached into a tab's
//           fragment during a tab switch (its reply arrived addressed
//           to the switched-away tab and was dropped) is removed when
//           the tab is restored: no "Loading previous task…" row
//           survives its request.
//
//   CAND-4  A daemonStatus {connected:false} event clears
//           adjacentLoading and removes the loader, so a later
//           overscroll can fetch again instead of being blocked for
//           ever by a reply the outage swallowed.
//
// The harness mirrors adjacentTaskScroll.test.js (overscroll + adjacent
// replies) and bgRenderTabSwap_sweep1026.test.js (a hidden parent's
// `summary` swapping a reopened sub-agent tab off screen).

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');

const {send, clickTab, fakeGeometry, wheel, getAdjacent} = harness;
const {
  bootParallelRun,
  finishFanOut,
  togglePanel,
  subagentTabEls,
} = require('./subagent_tab_helpers');

function makePage() {
  const page = harness.makePage();
  page.win._testApi.endLaunch();
  page.win._testApi.hideWelcome();
  return page;
}

function statusRow(win) {
  const doc = win.document;
  return {
    tokens: doc.getElementById('status-tokens').textContent,
    budget: doc.getElementById('status-budget').textContent,
    steps: doc.getElementById('status-steps').textContent,
  };
}

const TS = 1767225600000;

// Tab with a finished history task ('42') on screen, geometry faked so
// edge overscroll can fire — same shape as adjacentTaskScroll.test.js.
function setupWithHistoryTask(win, posted) {
  const tabId = harness.readyTabId(posted);
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-abc',
    task_id: '42',
    task: 'My old task',
    events: [{type: 'system_output', text: 'hello\n'}],
  });
  return {tabId, O};
}

// Overscroll at the top until a getAdjacentTask request leaves, then
// answer it with task W = '41'.
function loadPrevTask41(win, posted, tabId, O) {
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  const adj = getAdjacent(posted);
  assert.ok(adj.length >= 1, 'overscroll at top must request the prev task');
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Older task',
    task_id: '41',
    events: [{type: 'system_output', text: 'older\n'}],
  });
  assert.strictEqual(
    O.querySelectorAll('.adjacent-task[data-task-id="41"]').length,
    1,
    'the first reply must splice exactly one container for task 41',
  );
}

// CAND-1: a tab switch must not forget which neighbours are already
// spliced into the transcript. Today restoreTab -> resetAdjacentState
// rewinds oldestLoadedTaskId to the tab's own task while the spliced
// container survives in the fragment, so the next overscroll re-fetches
// task 41 and renderAdjacentTask splices a SECOND container for it.
function testNoDuplicateAdjacentRegionAfterTabSwitch() {
  const {win, posted} = makePage();
  try {
    const {tabId, O} = setupWithHistoryTask(win, posted);
    loadPrevTask41(win, posted, tabId, O);

    // Away to tab B (transcript detaches into A's fragment) and back.
    win._testApi.createNewTab();
    const tabB = win._testApi.getActiveTabId();
    assert.notStrictEqual(tabB, tabId, 'createNewTab must open tab B');
    clickTab(win, tabId);
    assert.strictEqual(win._testApi.getActiveTabId(), tabId);
    assert.strictEqual(
      O.querySelectorAll('.adjacent-task[data-task-id="41"]').length,
      1,
      'the spliced container must come back with the restored transcript',
    );

    // The transcript still shows task 41, but the rewound anchors make
    // the next overscroll fetch it again; the reply must not splice a
    // second copy.
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    const adj = getAdjacent(posted);
    assert.ok(
      adj.length >= 2,
      'overscroll after the switch must still be able to request older ' +
        'tasks; got ' +
        JSON.stringify(adj),
    );
    send(win, {
      type: 'adjacent_task_events',
      tabId,
      direction: 'prev',
      task: 'Older task',
      task_id: '41',
      events: [{type: 'system_output', text: 'older\n'}],
    });
    assert.strictEqual(
      O.querySelectorAll('.adjacent-task[data-task-id="41"]').length,
      1,
      'a second adjacent_task_events for task 41 after a tab ' +
        'switch-and-return must NOT splice a second ' +
        '.adjacent-task[data-task-id="41"] container: one region per ' +
        'task id',
    );
    console.log('PASS no duplicate adjacent region after tab switch');
  } finally {
    win.close();
  }
}

// CAND-2: a background tab's event can swap the visible tab mid-event.
// The user reads a finished sub-agent's tab S, reopened by expanding
// the fan-out panel, while the hidden parent P gets a `summary`
// tool_call: the summary folds the fan-out, the fold closes the
// finished tab S, and closeTab puts P on screen (restoreTab repaints
// the status row from P's saved numbers). processOutputEventForBgTab
// must then leave the row as restoreTab painted it, not overwrite it
// with the closed tab's pre-event values.
function testBgEventTabSwapKeepsRestoredStatusRow() {
  const {win, parentId: tabP, panel, subTabIds} = bootParallelRun(
    1,
    makePage(),
  );
  try {
    // Parent tab P runs live with numbers of its own.
    send(win, {
      type: 'usage_info',
      text: '',
      total_tokens: 999,
      cost: '$9.99',
      total_steps: 9,
      tabId: tabP,
      ts: TS,
    });
    const parentRow = statusRow(win);
    assert.deepStrictEqual(parentRow, {
      tokens: 'Tokens: 999',
      budget: 'Cost: $9.99',
      steps: 'Steps: 9',
    });

    // The sub-agent finishes (its tab closes) and the user expands the
    // folded fan-out again: the finished sub-agent's tab reopens.
    finishFanOut(win, subTabIds);
    togglePanel(win, panel);
    const [tabS] = subagentTabEls(win).map(el => el.dataset.tabId);
    assert.ok(tabS, 'expanding the fan-out reopens the finished tab');

    // The reopened tab S is on screen with numbers of ITS own.
    clickTab(win, tabS);
    assert.strictEqual(win._testApi.getActiveTabId(), tabS);
    send(win, {
      type: 'task_events',
      tabId: tabS,
      task: 'sub 1',
      task_id: 'sub-task-1',
      events: [
        {
          type: 'usage_info',
          text: '',
          total_tokens: 555,
          cost: '$5.55',
          total_steps: 5,
          ts: TS,
        },
      ],
    });
    const subRow = statusRow(win);
    assert.deepStrictEqual(subRow, {
      tokens: 'Tokens: 555',
      budget: 'Cost: $5.55',
      steps: 'Steps: 5',
    });

    // The hidden parent's summary folds the fan-out mid-event.
    send(win, {
      type: 'tool_call',
      name: 'summary',
      tabId: tabP,
      extras: {description: 'folded the fan-out'},
      ts: TS,
    });
    assert.strictEqual(
      win._testApi.getActiveTabId(),
      tabP,
      'the fold must have closed sub-agent tab S and put parent tab P ' +
        'on screen (scenario precondition)',
    );
    const after = statusRow(win);
    assert.deepStrictEqual(
      after,
      parentRow,
      'after a background event swapped the visible tab, the status row ' +
        'must keep the swapped-in tab\u2019s restored values (' +
        JSON.stringify(parentRow) +
        '), not the closed tab\u2019s pre-event values (' +
        JSON.stringify(subRow) +
        ')',
    );
    console.log('PASS bg event tab swap keeps the restored status row');
  } finally {
    win.close();
  }
}

// CAND-3: a loader detached into the switched-away tab's fragment
// (its reply was dropped by the isForActiveTab guard) must be removed
// when the tab is restored.
function testDetachedLoaderRemovedOnRestore() {
  const {win, posted} = makePage();
  try {
    const {tabId, O} = setupWithHistoryTask(win, posted);
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    assert.ok(
      getAdjacent(posted).length >= 1,
      'overscroll at top must request the prev task',
    );
    assert.ok(
      O.querySelector('.adjacent-loader'),
      'the request must show the "Loading previous task…" row',
    );

    // Switch away while the request is in flight: the loader detaches
    // into tab A's fragment with the transcript.
    win._testApi.createNewTab();
    assert.notStrictEqual(win._testApi.getActiveTabId(), tabId);

    // The reply lands addressed to the switched-away tab and is dropped.
    send(win, {
      type: 'adjacent_task_events',
      tabId,
      direction: 'prev',
      task: 'Older task',
      task_id: '41',
      events: [{type: 'system_output', text: 'older\n'}],
    });

    clickTab(win, tabId);
    assert.strictEqual(win._testApi.getActiveTabId(), tabId);
    assert.strictEqual(
      O.querySelector('.adjacent-loader'),
      null,
      'restoring the tab must remove the stale "Loading previous ' +
        'task…" row whose request was answered while the tab was hidden',
    );
    console.log('PASS detached adjacent loader removed on tab restore');
  } finally {
    win.close();
  }
}

// CAND-4: an outage that swallows the in-flight getAdjacentTask reply
// must not leave adjacent loading stuck: daemonStatus {connected:false}
// clears adjacentLoading and removes the loader, so a later overscroll
// fetches again.
function testDaemonDisconnectClearsAdjacentLoading() {
  const {win, posted} = makePage();
  try {
    const {O} = setupWithHistoryTask(win, posted);
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    assert.strictEqual(
      getAdjacent(posted).length,
      1,
      'overscroll at top must post exactly one request',
    );
    assert.ok(
      O.querySelector('.adjacent-loader'),
      'the request must show the loader row',
    );

    // The daemon goes away; the reply never comes.
    send(win, {type: 'daemonStatus', connected: false});

    assert.strictEqual(
      O.querySelector('.adjacent-loader'),
      null,
      'a daemon disconnect must remove the adjacent loader: its ' +
        'request can never be answered',
    );
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    assert.strictEqual(
      getAdjacent(posted).length,
      2,
      'after the disconnect cleared the in-flight state, a later ' +
        'overscroll must be able to fetch the previous task again ' +
        '(adjacentLoading must not stay latched for ever)',
    );
    console.log('PASS daemon disconnect clears adjacent loading state');
  } finally {
    win.close();
  }
}

function main() {
  const tests = [
    testNoDuplicateAdjacentRegionAfterTabSwitch,
    testBgEventTabSwapKeepsRestoredStatusRow,
    testDetachedLoaderRemovedOnRestore,
    testDaemonDisconnectClearsAdjacentLoading,
  ];
  const failures = [];
  for (const t of tests) {
    try {
      t();
    } catch (e) {
      failures.push(t.name + ': ' + e.message);
      console.error('FAIL ' + t.name + '\n  ' + e.message);
    }
  }
  assert.strictEqual(
    failures.length,
    0,
    'conc2026_adjacent_task_races: ' +
      failures.length +
      ' test(s) failed:\n' +
      failures.join('\n'),
  );
  console.log('All conc2026_adjacent_task_races tests passed');
}

main();
