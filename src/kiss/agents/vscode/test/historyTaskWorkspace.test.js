// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the "workspace" line in the History sidebar's
// per-task panel.
//
// Requirement driven by this test:
//
//   Every history row must render the task's workspace (the agent
//   ``work_dir``) on its OWN line, IMMEDIATELY after the
//   ``.running-item-metrics`` line (which itself follows the task
//   text and metrics-row content).  Visually:
//
//       <steps> steps • <tokens> tok • $<cost> • <hh:mm:ss>[ • <date>]
//       <work_dir>
//
//   * The workspace span has class ``running-item-workspace`` and
//     is the metrics row's immediate next sibling inside the
//     ``.sidebar-item`` row.
//   * Rows whose backend ``work_dir`` is empty/missing must NOT
//     render a workspace line (no blank line, no "(no workspace)").
//
// This test drives the production ``media/main.js`` (plus the real
// ``media/chat.html`` markup and ``media/panelCopy.js``) inside jsdom,
// exactly like ``historyTaskDuration.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/historyTaskWorkspace.test.js

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

// Fixture covering the four cases the workspace line must handle:
//
//   * row A: a normal task with a Unix-style workspace path.
//   * row B: a normal task with a Windows-style workspace path
//     (contains backslashes — must round-trip verbatim).
//   * row C: a task whose ``work_dir`` is an empty string — no
//     workspace line at all.
//   * row D: a task whose ``work_dir`` field is missing entirely
//     from the payload — no workspace line at all.
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
    // intentionally no ``work_dir`` key
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

// The History sidebar's workspace filter checkbox (#hf-workspace)
// hides every row whose ``work_dir`` does not match the currently
// configured workspace.  We always want every fixture row to render,
// so this helper clears the filter before sending the history event.
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

function workspaceSpan(row) {
  return row.querySelector('.running-item-workspace');
}

function metricsSpan(row) {
  return row.querySelector('.running-item-metrics');
}

function testWorkspaceRendersAfterMetrics() {
  const {win} = makeWebview();
  disableWorkspaceFilter(win);

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

  // Text must equal the original work_dir verbatim — no truncation,
  // no path-separator normalisation.
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

  // Position requirement: the workspace span must be the metrics
  // span's IMMEDIATE next sibling (so the workspace shows up on the
  // line right after the metrics line).
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

  // Empty / missing work_dir must NOT render a workspace line at
  // all — we don't want a blank line or a placeholder.
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
  // The workspace span must use ``flex-basis: 100%`` (same trick the
  // metrics row uses) so it drops onto its own visual line under the
  // metrics row inside the flex container ``.sidebar-item``.  jsdom
  // never loads the external ``main.css`` stylesheet that ``chat.html``
  // references via ``{{STYLE_HREF}}``, so we read the CSS file
  // directly and assert that a matching rule exists.  This is an
  // end-to-end check against the real production stylesheet.
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  // Extract every ``.running-item-workspace { ... }`` block.
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
