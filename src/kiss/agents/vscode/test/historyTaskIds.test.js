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

const WS = '/Users/koushik/work/repo';

const SESSIONS_FIXTURE = [
  {
    id: 'chat-a',
    task_id: 'task-a',
    parent_task_id: 'parent-a',
    title: 'ids row A — chat + task + parent',
    timestamp: 1_700_000_000,
    preview: 'ids row A — chat + task + parent',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 100,
    cost: 0.01,
    steps: 1,
    is_favorite: false,
    work_dir: WS,
    model: 'gpt-5',
    is_worktree: true,
    is_parallel: false,
    auto_commit_mode: true,
    startTs: 1_700_000_000_000,
    endTs: 1_700_000_005_000,
  },
  {
    id: 'chat-b',
    task_id: 'task-b',
    title: 'ids row B — chat + task, no parent',
    timestamp: 1_700_000_100,
    preview: 'ids row B — chat + task, no parent',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 100,
    cost: 0.01,
    steps: 1,
    is_favorite: false,
    work_dir: WS,
    model: 'gpt-5',
    is_worktree: false,
    is_parallel: false,
    auto_commit_mode: false,
    startTs: 1_700_000_100_000,
    endTs: 1_700_000_105_000,
  },
  {
    id: 'chat-c',
    task_id: null,
    title: 'ids row C — chat only',
    timestamp: 1_700_000_200,
    preview: 'ids row C — chat only',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '',
    startTs: 1_700_000_200_000,
    endTs: 1_700_000_201_000,
  },
  {
    id: '',
    task_id: 'task-d',
    title: 'ids row D — task only',
    timestamp: 1_700_000_300,
    preview: 'ids row D — task only',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '',
    startTs: 1_700_000_300_000,
    endTs: 1_700_000_301_000,
  },
  {
    id: '',
    task_id: null,
    parent_task_id: 'parent-e',
    title: 'ids row E — parent only',
    timestamp: 1_700_000_400,
    preview: 'ids row E — parent only',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '',
    startTs: 1_700_000_400_000,
    endTs: 1_700_000_401_000,
  },
  {
    id: '',
    task_id: null,
    title: 'ids row F — neither chat, task, nor parent',
    timestamp: 1_700_000_500,
    preview: 'ids row F — neither chat, task, nor parent',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '',
    startTs: 1_700_000_500_000,
    endTs: 1_700_000_501_000,
  },
];

function rowsByTitle(win) {
  const list = win.document.getElementById('history-list');
  const rows = list.querySelectorAll('.sidebar-item');
  const map = {};
  rows.forEach(r => {
    const t = r.querySelector('.sidebar-item-text');
    if (!t) return;
    map[t.textContent] = r;
  });
  return map;
}

function disableWorkspaceFilter(win) {
  send(win, {
    type: 'configData',
    config: {work_dir: ''},
    apiKeys: {},
  });
  const ws = win.document.getElementById('hf-workspace');
  if (ws && ws.checked) {
    ws.checked = false;
    ws.dispatchEvent(new win.Event('change', {bubbles: true}));
  }
}

function idsSpan(row) {
  return row.querySelector('.running-item-ids');
}

function workspaceSpan(row) {
  return row.querySelector('.running-item-workspace');
}

function metricsSpan(row) {
  return row.querySelector('.running-item-metrics');
}

function infoBlock(row) {
  return row.querySelector('.running-item-info');
}

function testIdsLineRendersAllCombinations() {
  const {win} = makeWebview();
  disableWorkspaceFilter(win);

  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  const rows = rowsByTitle(win);
  const a = rows['ids row A — chat + task + parent'];
  const b = rows['ids row B — chat + task, no parent'];
  const c = rows['ids row C — chat only'];
  const d = rows['ids row D — task only'];
  const e = rows['ids row E — parent only'];
  const f = rows['ids row F — neither chat, task, nor parent'];
  assert.ok(a, 'row A must render');
  assert.ok(b, 'row B must render');
  assert.ok(c, 'row C must render');
  assert.ok(d, 'row D must render');
  assert.ok(e, 'row E must render');
  assert.ok(f, 'row F must render');

  const aIds = idsSpan(a);
  assert.ok(aIds, 'row A must render a .running-item-ids span');
  assert.strictEqual(
    aIds.textContent,
    'chat chat-a • task task-a • parent parent-a',
    `row A ids text mismatch; got: ${aIds.textContent}`,
  );

  const bIds = idsSpan(b);
  assert.ok(bIds, 'row B must render a .running-item-ids span');
  assert.strictEqual(
    bIds.textContent,
    'chat chat-b • task task-b',
    `row B ids text mismatch; got: ${bIds.textContent}`,
  );

  const cIds = idsSpan(c);
  assert.ok(cIds, 'row C must render a .running-item-ids span');
  assert.strictEqual(
    cIds.textContent,
    'chat chat-c',
    `row C ids text mismatch; got: ${cIds.textContent}`,
  );

  const dIds = idsSpan(d);
  assert.ok(dIds, 'row D must render a .running-item-ids span');
  assert.strictEqual(
    dIds.textContent,
    'task task-d',
    `row D ids text mismatch; got: ${dIds.textContent}`,
  );

  const eIds = idsSpan(e);
  assert.ok(eIds, 'row E must render a .running-item-ids span');
  assert.strictEqual(
    eIds.textContent,
    'parent parent-e',
    `row E ids text mismatch; got: ${eIds.textContent}`,
  );

  assert.strictEqual(
    idsSpan(f),
    null,
    'row F (no chat, no task, no parent) must NOT render a .running-item-ids span',
  );

  win.close();
  console.log('  ok - .running-item-ids renders the right text in every combination');
}

function testIdsLineOrderingAndContainer() {
  const {win} = makeWebview();
  disableWorkspaceFilter(win);

  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  const rows = rowsByTitle(win);
  const a = rows['ids row A — chat + task + parent'];
  const b = rows['ids row B — chat + task, no parent'];
  assert.ok(a, 'row A must render');
  assert.ok(b, 'row B must render');

  for (const [label, row] of [['A', a], ['B', b]]) {
    const info = infoBlock(row);
    assert.ok(
      info,
      `row ${label} must render a .running-item-info container ` +
        'wrapping metrics / workspace / ids',
    );
    const metrics = metricsSpan(row);
    const workspace = workspaceSpan(row);
    const ids = idsSpan(row);
    assert.ok(metrics, `row ${label} must render metrics`);
    assert.ok(workspace, `row ${label} must render workspace`);
    assert.ok(ids, `row ${label} must render ids`);
    assert.strictEqual(
      metrics.parentElement,
      info,
      `row ${label}: metrics must be a child of .running-item-info`,
    );
    assert.strictEqual(
      workspace.parentElement,
      info,
      `row ${label}: workspace must be a child of .running-item-info`,
    );
    assert.strictEqual(
      ids.parentElement,
      info,
      `row ${label}: ids must be a child of .running-item-info`,
    );

    assert.strictEqual(
      metrics.nextElementSibling,
      workspace,
      `row ${label}: workspace must come immediately after metrics`,
    );
    assert.strictEqual(
      workspace.nextElementSibling,
      ids,
      `row ${label}: ids must come immediately after workspace`,
    );

    assert.strictEqual(
      info,
      row.lastElementChild,
      `row ${label}: .running-item-info must be the last child of the row`,
    );
  }

  win.close();
  console.log(
    '  ok - metrics / workspace / ids stack inside .running-item-info in the right order',
  );
}

function testIdsLineLayoutEliminatesGap() {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');

  const infoRe = /\.running-item-info\s*\{([^}]*)\}/g;
  const infoMatch = infoRe.exec(css);
  assert.ok(
    infoMatch,
    'main.css must define a .running-item-info rule that wraps ' +
      'the metrics / workspace / ids lines',
  );
  const infoBody = infoMatch[1];
  assert.match(
    infoBody,
    /flex-basis\s*:\s*100%/,
    '.running-item-info must declare "flex-basis: 100%" so it ' +
      'drops onto its own line in the .sidebar-item flex container',
  );
  assert.match(
    infoBody,
    /flex-direction\s*:\s*column/,
    '.running-item-info must stack its children vertically ' +
      '(flex-direction: column)',
  );
  assert.match(
    infoBody,
    /(^|[^-])(?:row-)?gap\s*:\s*0(?:px)?\b/,
    '.running-item-info must declare gap: 0 (or row-gap: 0) so ' +
      'metrics / workspace / ids render flush with no visual gap',
  );

  const idsRe = /\.running-item-ids\s*\{([^}]*)\}/g;
  const idsMatch = idsRe.exec(css);
  assert.ok(
    idsMatch,
    'main.css must define a .running-item-ids rule for the new ' +
      'chat/task/parent ids line',
  );

  console.log('  ok - CSS eliminates the gap between metrics and workspace lines');
}

function main() {
  testIdsLineRendersAllCombinations();
  testIdsLineOrderingAndContainer();
  testIdsLineLayoutEliminatesGap();
  console.log('historyTaskIds.test.js: all assertions passed.');
}

main();
