// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function statusOf(win) {
  return win.document.getElementById('status-text').textContent;
}

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

  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: Date.now() - 65_000,
  });
  assert.match(statusOf(win), /^Running 1m [0-9]s$/);

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

  send(win, {
    type: 'openSubagentTab',
    tab_id: 'bg-tab',
    parent_tab_id: activeId,
    description: 'background work',
  });
  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: Date.now() - 3_000,
  });
  assert.match(statusOf(win), /^Running [2-6]s$/);

  const bgEnd = Date.now();
  send(win, {
    type: 'status',
    running: true,
    tabId: 'bg-tab',
    startTs: bgEnd - 201_000,
  });
  assert.match(statusOf(win), /^Running [2-6]s$/);

  send(win, {
    type: 'task_done',
    tabId: 'bg-tab',
    startTs: bgEnd - 201_000,
    endTs: bgEnd,
  });
  send(win, {type: 'status', running: false, tabId: 'bg-tab'});

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

  clickTab(win, activeId);
  assert.match(
    statusOf(win),
    /^Running [2-6]s$/,
    "the original tab's running timer must survive the auto-switch",
  );
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

  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-x',
    parent_tab_id: activeId,
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
    parent_tab_id: activeId,
    description: 'history task',
  });
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
    parent_tab_id: activeId,
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
