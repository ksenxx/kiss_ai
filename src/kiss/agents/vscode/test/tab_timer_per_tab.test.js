// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the per-tab running/done timer in ``media/main.js``.
//
// Requirement: EVERY tab must show the running time as the actual time
// spent since the start of the task while it runs, and as the difference
// between the task's end timestamp and start timestamp once it has ended
// — regardless of tab switches and regardless of newer runs started in
// other windows (the daemon broadcasts tab-stamped events to every
// connected client).
//
// This test drives the real ``media/main.js`` (plus the real
// ``media/chat.html`` markup and ``media/panelCopy.js``) inside jsdom —
// no mocks of project code — exactly like ``bughunt2_status_timer.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/tab_timer_per_tab.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``panelCopy.js`` and
 * ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
 */
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
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function statusOf(win) {
  return win.document.getElementById('status-text').textContent;
}

/** Click the tab-bar tab whose ``data-tab-id`` matches ``tabId``. */
function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id="${tabId}"]`,
  );
  assert.ok(el, `no tab with id "${tabId}" in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

async function testForeignNewerRunDoesNotClobberActiveTimer() {
  const {win, posted} = makeWebview();
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const activeId = ready.tabId;

  // Active tab's agent started 65 s ago.
  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: Date.now() - 65_000,
  });
  assert.match(statusOf(win), /^Running 1m [0-9]s$/);

  // A NEWER run starts in ANOTHER window (unknown tab id, startTs ~now).
  send(win, {
    type: 'status',
    running: true,
    tabId: 'other-window-tab',
    startTs: Date.now(),
  });
  await sleep(1_300);
  assert.match(
    statusOf(win),
    /^Running 1m [0-9]+s$/,
    'BUG: a newer run in another window clobbered the active ' +
      `tab's running timer (header now "${statusOf(win)}")`,
  );

  // The other window's run finishes — tab-stamped task_done + status.
  send(win, {
    type: 'task_done',
    tabId: 'other-window-tab',
    startTs: Date.now() - 5_000,
    endTs: Date.now(),
  });
  send(win, {type: 'status', running: false, tabId: 'other-window-tab'});
  await sleep(1_300);
  assert.match(
    statusOf(win),
    /^Running 1m [0-9]+s$/,
    'BUG: another window finishing its run flipped the active ' +
      `tab's header (now "${statusOf(win)}")`,
  );
  win.close();
  console.log('  ok - newer runs in another window never touch the header');
}

async function testActiveTaskDoneShowsEndMinusStart() {
  const {win, posted} = makeWebview();
  const activeId = posted.find(m => m.type === 'ready').tabId;

  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: Date.now() - 5_000,
  });
  // The agent reports its true wall-clock: ran for exactly 2m 7s.
  const endTs = Date.now();
  send(win, {
    type: 'task_done',
    tabId: activeId,
    startTs: endTs - 127_000,
    endTs,
  });
  assert.strictEqual(
    statusOf(win),
    'Done (2m 7s)',
    'active task_done must show endTs - startTs',
  );
  win.close();
  console.log('  ok - active task_done shows endTs - startTs');
}

async function testBackgroundTaskDoneShowsDurationAfterSwitch() {
  const {win, posted} = makeWebview();
  const activeId = posted.find(m => m.type === 'ready').tabId;

  // Materialise a LOCAL background tab the way the backend does it.
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'bg-tab',
    description: 'background work',
  });
  // The user keeps working in the original (still active) tab.
  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: Date.now() - 3_000,
  });
  assert.match(statusOf(win), /^Running [2-6]s$/);

  // The background tab's agent runs: its status event must NOT
  // clobber the foreground tab's header.
  const bgEnd = Date.now();
  send(win, {
    type: 'status',
    running: true,
    tabId: 'bg-tab',
    startTs: bgEnd - 201_000,
  });
  assert.match(statusOf(win), /^Running [2-6]s$/);

  // The background tab finishes: 3m 21s of agent time.
  send(win, {
    type: 'task_done',
    tabId: 'bg-tab',
    startTs: bgEnd - 201_000,
    endTs: bgEnd,
  });
  send(win, {type: 'status', running: false, tabId: 'bg-tab'});

  // ``task_done`` for a LOCAL tab auto-switches focus to it (product
  // contract since ``focusFinishedTab``: the result panel must be
  // immediately visible) and the header MUST show the done duration
  // computed from the agent's own timestamps.
  const activeEl = win.document.querySelector('.chat-tab.active');
  assert.strictEqual(
    activeEl && activeEl.getAttribute('data-tab-id'),
    'bg-tab',
    'task_done in a local background tab must auto-focus that tab',
  );
  assert.strictEqual(
    statusOf(win),
    'Done (3m 21s)',
    'a tab whose task finished in the background must show ' +
      'endTs - startTs once focused',
  );

  // Switching back to the original tab restores its own running
  // timer untouched…
  clickTab(win, activeId);
  assert.match(
    statusOf(win),
    /^Running [2-6]s$/,
    "the original tab's running timer must survive the auto-switch",
  );
  // …and returning to the finished tab re-renders its done duration.
  clickTab(win, 'bg-tab');
  assert.strictEqual(
    statusOf(win),
    'Done (3m 21s)',
    'the done duration must re-render when switching back',
  );
  win.close();
  console.log(
    '  ok - background task_done auto-focuses its tab and shows duration',
  );
}

async function testDoneLabelSurvivesTabSwitches() {
  const {win, posted} = makeWebview();
  const activeId = posted.find(m => m.type === 'ready').tabId;

  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: Date.now() - 1_000,
  });
  const endTs = Date.now();
  send(win, {
    type: 'task_done',
    tabId: activeId,
    startTs: endTs - 45_000,
    endTs,
  });
  send(win, {type: 'status', running: false, tabId: activeId});
  assert.strictEqual(statusOf(win), 'Done (45s)');

  // Open a sub-agent tab (switches away), then come back.
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-x',
    description: 'sub work',
  });
  clickTab(win, 'sub-x');
  clickTab(win, activeId);
  assert.strictEqual(
    statusOf(win),
    'Done (45s)',
    'the done duration must survive switching away and back',
  );
  win.close();
  console.log('  ok - done duration survives tab switches');
}

async function testBackgroundReplayWithExtraTimestamps() {
  const {win, posted} = makeWebview();
  const activeId = posted.find(m => m.type === 'ready').tabId;
  void activeId;

  send(win, {
    type: 'openSubagentTab',
    tab_id: 'hist-tab',
    description: 'history task',
  });
  // Replay a FINISHED task into the background tab (history load):
  // extra carries the agent's persisted start/end timestamps.
  const end = Date.now() - 60_000;
  send(win, {
    type: 'task_events',
    tabId: 'hist-tab',
    task: 'history task',
    events: [],
    extra: JSON.stringify({startTs: end - 83_000, endTs: end}),
  });
  clickTab(win, 'hist-tab');
  assert.strictEqual(
    statusOf(win),
    'Done (1m 23s)',
    'a background tab replaying a finished task must show ' +
      'endTs - startTs after switching to it',
  );
  win.close();
  console.log('  ok - background replay with extra timestamps shows Done(…)');
}

async function testRunningTabKeepsAnchorAcrossSwitches() {
  const {win, posted} = makeWebview();
  const activeId = posted.find(m => m.type === 'ready').tabId;

  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: Date.now() - 125_000,
  });
  assert.match(statusOf(win), /^Running 2m [0-9]s$/);

  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-y',
    description: 'detour',
  });
  clickTab(win, 'sub-y');
  clickTab(win, activeId);
  await sleep(1_300);
  assert.match(
    statusOf(win),
    /^Running 2m [0-9]+s$/,
    'a running tab must stay anchored to its own startTs across ' +
      `tab switches (header now "${statusOf(win)}")`,
  );
  win.close();
  console.log('  ok - running tab keeps its anchor across tab switches');
}

async function runTests() {
  await testForeignNewerRunDoesNotClobberActiveTimer();
  await testActiveTaskDoneShowsEndMinusStart();
  await testBackgroundTaskDoneShowsDurationAfterSwitch();
  await testDoneLabelSurvivesTabSwitches();
  await testBackgroundReplayWithExtraTimestamps();
  await testRunningTabKeepsAnchorAcrossSwitches();
}

runTests().then(
  () => {
    console.log('\n6 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exit(1);
  },
);
