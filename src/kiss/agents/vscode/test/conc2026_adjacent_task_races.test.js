// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for adjacent-task / background-status races
// in media/main.js (audit candidates C CAND-1..CAND-4, see
// tmp/audit-media-cand-C.md). Each test asserts the CORRECT (post-fix)
// behavior:
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
//           ACTIVE sub-agent tab), processOutputEventForBgTab must NOT
//           overwrite the newly restored tab's status row texts /
//           stepCount with the pre-event values of the tab that was
//           closed: the row keeps the swapped-in tab's own values.
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
// replies) and bgTabStreamParity.test.js (multi-tab status row,
// clickTab, run_parallel sub-agent tabs).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
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
  let state;
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
  win._testApi.endLaunch();
  win._testApi.hideWelcome();
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function fakeGeometry(el) {
  Object.defineProperty(el, 'scrollWidth', {value: 2000, configurable: true});
  Object.defineProperty(el, 'clientWidth', {value: 400, configurable: true});
  Object.defineProperty(el, 'scrollHeight', {value: 3000, configurable: true});
  Object.defineProperty(el, 'clientHeight', {value: 500, configurable: true});
}

function wheel(win, O, deltaY, n) {
  for (let i = 0; i < n; i++) {
    O.dispatchEvent(
      new win.WheelEvent('wheel', {deltaY, bubbles: true, cancelable: true}),
    );
  }
}

function getAdjacent(posted) {
  return posted.filter(m => m.type === 'getAdjacentTask');
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
  const tabId = posted.find(m => m.type === 'ready').tabId;
  const O = win.document.getElementById('output');
  fakeGeometry(O);
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-abc',
    task_id: '42',
    task: 'My old task',
    events: [
      {type: 'task_start', task: 'My old task'},
      {type: 'system_output', text: 'hello\n'},
    ],
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
    events: [
      {type: 'task_start', task: 'Older task'},
      {type: 'system_output', text: 'older\n'},
    ],
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
  const {win, posted} = makeWebview();
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
        'tasks; got ' + JSON.stringify(adj),
    );
    send(win, {
      type: 'adjacent_task_events',
      tabId,
      direction: 'prev',
      task: 'Older task',
      task_id: '41',
      events: [
        {type: 'task_start', task: 'Older task'},
        {type: 'system_output', text: 'older\n'},
      ],
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

// CAND-2: a background tab's event can swap the visible tab mid-event
// (its `summary` tool_call adopts the finished run_parallel panel,
// collapses it, and the collapse closes the ACTIVE sub-agent tab;
// closeTab falls back to the parent and restoreTab repaints the status
// row from the parent's saved numbers). processOutputEventForBgTab must
// then leave the row as restoreTab painted it — not overwrite it with
// the closed tab's pre-event values.
function testBgEventTabSwapKeepsRestoredStatusRow() {
  const {win, posted} = makeWebview();
  try {
    const tabP = win._testApi.getActiveTabId();

    // Parent tab P runs live with numbers of its own...
    send(win, {type: 'status', running: true, tabId: tabP, startTs: TS});
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
    assert.strictEqual(parentRow.tokens, 'Tokens: 999');
    assert.strictEqual(parentRow.budget, 'Cost: $9.99');
    assert.strictEqual(parentRow.steps, 'Steps: 9');

    // ...fans out, and its sub-agent gets a tab.
    send(win, {
      type: 'tool_call',
      name: 'run_parallel',
      tabId: tabP,
      extras: {tasks: JSON.stringify(['sub 1'])},
      ts: TS,
    });
    const before = posted.length;
    send(win, {
      type: 'new_tab',
      task_id: 'sub-task-1',
      parent_tab_id: tabP,
      taskId: '',
    });
    const resume = posted
      .slice(before)
      .find(m => m.type === 'resumeSession' && m.taskId === 'sub-task-1');
    assert.ok(resume, 'new_tab must make the webview post resumeSession');
    const tabS = resume.tabId;
    send(win, {
      type: 'openSubagentTab',
      tab_id: tabS,
      parent_tab_id: tabP,
      description: 'sub 1',
      task_id: 'sub-task-1',
      taskIndex: 0,
    });

    // The sub-agent tab is on screen with numbers of ITS own.
    clickTab(win, tabS);
    assert.strictEqual(win._testApi.getActiveTabId(), tabS);
    send(win, {
      type: 'usage_info',
      text: '',
      total_tokens: 555,
      cost: '$5.55',
      total_steps: 5,
      tabId: tabS,
      ts: TS,
    });
    const subRow = statusRow(win);
    assert.strictEqual(subRow.tokens, 'Tokens: 555');
    assert.strictEqual(subRow.budget, 'Cost: $5.55');
    assert.strictEqual(subRow.steps, 'Steps: 5');

    // The hidden parent finishes its fan-out and calls `summary`: the
    // summary panel adopts the run_parallel panel, collapses it, and
    // the collapse closes tab S — the tab on screen — mid-event.
    send(win, {type: 'tool_result', content: 'sub done', tabId: tabP, ts: TS});
    send(win, {type: 'tool_call', name: 'summary', tabId: tabP, ts: TS});

    assert.strictEqual(
      win._testApi.getActiveTabId(),
      tabP,
      'the collapsed fan-out must have closed sub-agent tab S and put ' +
        'parent tab P on screen (scenario precondition)',
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
  const {win, posted} = makeWebview();
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
      events: [
        {type: 'task_start', task: 'Older task'},
        {type: 'system_output', text: 'older\n'},
      ],
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
  const {win, posted} = makeWebview();
  try {
    const {tabId, O} = setupWithHistoryTask(win, posted);
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
