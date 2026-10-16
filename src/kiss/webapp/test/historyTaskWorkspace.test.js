// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {makePage, send} = require('./simplify2_harness');

const MEDIA = path.join(__dirname, '..', 'media');

const WS_A = '/Users/koushik/work/repo-A';
const WS_B = 'C:\\Users\\koushik\\repo-B';

const SESSIONS_FIXTURE = [
  {
    id: 'chatA',
    task_id: 1,
    title: 'task with unix workspace',
    timestamp: 1_700_000_000,
    preview: 'task with unix workspace',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 1234,
    cost: 0.1234,
    steps: 3,
    is_favorite: false,
    work_dir: WS_A,
    startTs: 1_700_000_000_000,
    endTs: 1_700_000_010_000,
  },
  {
    id: 'chatB',
    task_id: 2,
    title: 'task with windows workspace',
    timestamp: 1_700_000_100,
    preview: 'task with windows workspace',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 100,
    cost: 0.01,
    steps: 1,
    is_favorite: false,
    work_dir: WS_B,
    startTs: 1_700_000_100_000,
    endTs: 1_700_000_165_000,
  },
  {
    id: 'chatC',
    task_id: 3,
    title: 'task with empty workspace string',
    timestamp: 1_700_000_200,
    preview: 'task with empty workspace string',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 7,
    cost: 0,
    steps: 1,
    is_favorite: false,
    work_dir: '',
    startTs: 1_700_000_200_000,
    endTs: 1_700_000_201_000,
  },
  {
    id: 'chatD',
    task_id: 4,
    title: 'task with missing workspace field',
    timestamp: 1_700_000_300,
    preview: 'task with missing workspace field',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    startTs: 1_700_000_300_000,
    endTs: 1_700_000_301_000,
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

function sendEmptyWorkDir(win) {
  send(win, {
    type: 'configData',
    config: {work_dir: ''},
    apiKeys: {},
  });
}

function workspaceSpan(row) {
  return row.querySelector('.running-item-workspace');
}

function metricsSpan(row) {
  return row.querySelector('.running-item-metrics');
}

function testWorkspaceRendersAfterMetrics() {
  const {win} = makePage();
  sendEmptyWorkDir(win);

  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  const rows = rowsByTitle(win);
  assert.ok(rows['task with unix workspace'], 'row A must render');
  assert.ok(rows['task with windows workspace'], 'row B must render');
  assert.ok(
    rows['task with empty workspace string'],
    'row C must render',
  );
  assert.ok(
    rows['task with missing workspace field'],
    'row D must render',
  );

  const a = rows['task with unix workspace'];
  const b = rows['task with windows workspace'];
  const c = rows['task with empty workspace string'];
  const d = rows['task with missing workspace field'];

  const aWs = workspaceSpan(a);
  const bWs = workspaceSpan(b);
  assert.ok(
    aWs,
    'row A must render a .running-item-workspace span for its work_dir',
  );
  assert.ok(
    bWs,
    'row B must render a .running-item-workspace span for its work_dir',
  );

  assert.strictEqual(
    aWs.textContent,
    WS_A,
    `row A workspace text must equal ${WS_A}; got: ${aWs.textContent}`,
  );
  assert.strictEqual(
    bWs.textContent,
    WS_B,
    `row B workspace text must equal ${WS_B}; got: ${bWs.textContent}`,
  );

  const aMetrics = metricsSpan(a);
  const bMetrics = metricsSpan(b);
  assert.ok(aMetrics, 'row A must keep its metrics span');
  assert.ok(bMetrics, 'row B must keep its metrics span');
  assert.strictEqual(
    aMetrics.nextElementSibling,
    aWs,
    'row A: workspace span must come immediately after metrics span',
  );
  assert.strictEqual(
    bMetrics.nextElementSibling,
    bWs,
    'row B: workspace span must come immediately after metrics span',
  );

  assert.strictEqual(
    workspaceSpan(c),
    null,
    'row C (empty work_dir) must NOT render a workspace span',
  );
  assert.strictEqual(
    workspaceSpan(d),
    null,
    'row D (missing work_dir) must NOT render a workspace span',
  );

  win.close();
  console.log(
    '  ok - workspace renders on its own line after metrics, ' +
      'omitted when work_dir is empty/missing',
  );
}

function testWorkspaceLineBreaksToOwnVisualLine() {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const re = /\.running-item-workspace\s*\{([^}]*)\}/g;
  let m;
  let found = false;
  while ((m = re.exec(css)) !== null) {
    const body = m[1];
    if (/flex-basis\s*:\s*100%/.test(body)) {
      found = true;
      break;
    }
  }
  assert.ok(
    found,
    '.running-item-workspace rule in main.css must declare ' +
      '"flex-basis: 100%" so the workspace span renders on its own ' +
      'line below the metrics row',
  );
  console.log('  ok - workspace span has flex-basis: 100%');
}

function main() {
  testWorkspaceRendersAfterMetrics();
  testWorkspaceLineBreaksToOwnVisualLine();
  console.log('historyTaskWorkspace.test.js: all assertions passed.');
}

main();
