// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the per-task metadata in the History
// sidebar.
//
// Requirement driven by this test:
//
//   Every history row whose persisted ``extra`` carries a model
//   name must render the model name and the three run flags
//   appended to the workspace line, separated by ``•``.  Format:
//
//       <work_dir> • <model> • <wt|no-wt>
//         • <parallel|sequential> • <auto-commit|manual-commit>
//
//   The combined text lives inside the single
//   ``.running-item-workspace`` span (no separate "meta" span) so
//   it renders on the SAME visual line as the workspace path.
//
//   * Rows with workspace but no model render the workspace alone
//     (legacy behaviour).
//   * Rows with model but no workspace render only the metadata
//     part (no leading bullet, no placeholder).
//   * Rows with neither render no ``.running-item-workspace`` span
//     at all.
//   * The booleans default to ``false`` when missing, so a row
//     with model but no flags renders ``<model> • no-wt •
//     sequential • manual-commit``.
//
// This test drives the production ``media/main.js`` (plus the real
// ``media/chat.html`` markup and ``media/panelCopy.js``) inside jsdom,
// exactly like ``historyTaskWorkspace.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/historyTaskMeta.test.js

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

// Fixture covering every combination the workspace+meta line must
// handle:
//
//   * row A: all flags true, model + workspace present.
//   * row B: all flags false, model + workspace present.
//   * row C: model + flags present, but NO workspace — the
//     metadata still renders, alone, in the workspace span.
//   * row D: model missing — only the workspace renders.
//   * row E: model present, every boolean flag missing from the
//     payload — defaults to no-wt / sequential / manual-commit.
//   * row F: neither workspace nor model — NO span at all.
const WS_A = '/Users/koushik/work/repo-A';
const WS_B = 'C:\\Users\\koushik\\repo-B';
const WS_D = '/Users/koushik/work/repo-D';

const SESSIONS_FIXTURE = [
  {
    id: 'metaA',
    task_id: 101,
    title: 'meta row A — all flags on',
    timestamp: 1_700_000_000,
    preview: 'meta row A — all flags on',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 1234,
    cost: 0.1234,
    steps: 3,
    is_favorite: false,
    work_dir: WS_A,
    model: 'gpt-5',
    is_worktree: true,
    is_parallel: true,
    auto_commit_mode: true,
    startTs: 1_700_000_000_000,
    endTs: 1_700_000_010_000,
  },
  {
    id: 'metaB',
    task_id: 102,
    title: 'meta row B — all flags off',
    timestamp: 1_700_000_100,
    preview: 'meta row B — all flags off',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 100,
    cost: 0.01,
    steps: 1,
    is_favorite: false,
    work_dir: WS_B,
    model: 'claude-3.7-sonnet',
    is_worktree: false,
    is_parallel: false,
    auto_commit_mode: false,
    startTs: 1_700_000_100_000,
    endTs: 1_700_000_165_000,
  },
  {
    id: 'metaC',
    task_id: 103,
    title: 'meta row C — model without workspace',
    timestamp: 1_700_000_200,
    preview: 'meta row C — model without workspace',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 7,
    cost: 0,
    steps: 1,
    is_favorite: false,
    work_dir: '',
    model: 'gpt-5-mini',
    is_worktree: true,
    is_parallel: false,
    auto_commit_mode: true,
    startTs: 1_700_000_200_000,
    endTs: 1_700_000_201_000,
  },
  {
    id: 'metaD',
    task_id: 104,
    title: 'meta row D — workspace without model',
    timestamp: 1_700_000_300,
    preview: 'meta row D — workspace without model',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: WS_D,
    // intentionally NO model, NO is_worktree, NO is_parallel,
    // NO auto_commit_mode
    startTs: 1_700_000_300_000,
    endTs: 1_700_000_301_000,
  },
  {
    id: 'metaE',
    task_id: 105,
    title: 'meta row E — model only, flags missing',
    timestamp: 1_700_000_400,
    preview: 'meta row E — model only, flags missing',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 5,
    cost: 0,
    steps: 1,
    is_favorite: false,
    work_dir: '',
    model: 'legacy-model',
    // is_worktree / is_parallel / auto_commit_mode intentionally
    // omitted — must default to false → no-wt / sequential /
    // manual-commit.
    startTs: 1_700_000_400_000,
    endTs: 1_700_000_401_000,
  },
  {
    id: 'metaF',
    task_id: 106,
    title: 'meta row F — neither workspace nor model',
    timestamp: 1_700_000_500,
    preview: 'meta row F — neither workspace nor model',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '',
    // no model, no flags, no workspace
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

function testWorkspaceLineMergesMetadata() {
  const {win} = makeWebview();
  disableWorkspaceFilter(win);

  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  const rows = rowsByTitle(win);
  const a = rows['meta row A — all flags on'];
  const b = rows['meta row B — all flags off'];
  const c = rows['meta row C — model without workspace'];
  const d = rows['meta row D — workspace without model'];
  const e = rows['meta row E — model only, flags missing'];
  const f = rows['meta row F — neither workspace nor model'];
  assert.ok(a, 'row A must render');
  assert.ok(b, 'row B must render');
  assert.ok(c, 'row C must render');
  assert.ok(d, 'row D must render');
  assert.ok(e, 'row E must render');
  assert.ok(f, 'row F must render');

  const aWs = workspaceSpan(a);
  const bWs = workspaceSpan(b);
  const cWs = workspaceSpan(c);
  const dWs = workspaceSpan(d);
  const eWs = workspaceSpan(e);
  assert.ok(aWs, 'row A must render a .running-item-workspace span');
  assert.ok(bWs, 'row B must render a .running-item-workspace span');
  assert.ok(
    cWs,
    'row C (no workspace, has model) must still render a ' +
      '.running-item-workspace span carrying the metadata',
  );
  assert.ok(
    dWs,
    'row D (workspace, no model) must render a .running-item-workspace span',
  );
  assert.ok(
    eWs,
    'row E (no workspace, model only) must still render a ' +
      '.running-item-workspace span',
  );

  // Exact merged text: workspace, then model, then wt/no-wt,
  // parallel/sequential, auto-commit/manual-commit — joined with
  // the same bullet (``•``) the metrics row uses.
  assert.strictEqual(
    aWs.textContent,
    WS_A + ' • gpt-5 • wt • parallel • auto-commit',
    `row A workspace+meta text mismatch; got: ${aWs.textContent}`,
  );
  assert.strictEqual(
    bWs.textContent,
    WS_B + ' • claude-3.7-sonnet • no-wt • sequential • manual-commit',
    `row B workspace+meta text mismatch; got: ${bWs.textContent}`,
  );
  // No workspace → starts with the model, no leading bullet.
  assert.strictEqual(
    cWs.textContent,
    'gpt-5-mini • wt • sequential • auto-commit',
    `row C workspace+meta text mismatch; got: ${cWs.textContent}`,
  );
  // No model → workspace alone (legacy rows).
  assert.strictEqual(
    dWs.textContent,
    WS_D,
    `row D workspace text mismatch; got: ${dWs.textContent}`,
  );
  // Missing booleans default to false → no-wt / sequential /
  // manual-commit.
  assert.strictEqual(
    eWs.textContent,
    'legacy-model • no-wt • sequential • manual-commit',
    `row E workspace+meta text (default flags) mismatch; got: ${eWs.textContent}`,
  );

  // Row F: neither workspace nor model → NO span at all.
  assert.strictEqual(
    workspaceSpan(f),
    null,
    'row F (no workspace, no model) must NOT render a workspace span',
  );

  // Position requirement: the workspace+meta span must follow the
  // metrics span immediately so the metadata appears on the line
  // right below metrics.
  for (const [label, row] of [
    ['A', a],
    ['B', b],
    ['C', c],
    ['D', d],
    ['E', e],
  ]) {
    const metrics = metricsSpan(row);
    assert.ok(metrics, `row ${label} must keep its metrics span`);
    assert.strictEqual(
      metrics.nextElementSibling,
      workspaceSpan(row),
      `row ${label}: workspace span must come immediately after metrics span`,
    );
  }

  // No row should ever render a stray ``.running-item-meta`` span
  // any more — the metadata lives inside the workspace span now.
  for (const [label, row] of [
    ['A', a],
    ['B', b],
    ['C', c],
    ['D', d],
    ['E', e],
    ['F', f],
  ]) {
    assert.strictEqual(
      row.querySelector('.running-item-meta'),
      null,
      `row ${label} must NOT render a separate .running-item-meta span`,
    );
  }

  win.close();
  console.log(
    '  ok - workspace + metadata render on the same line, separated by " • "',
  );
}

function testWorkspaceLineBreaksToOwnVisualLine() {
  // The merged workspace+meta span must use ``flex-basis: 100%``
  // (same trick the metrics row uses) so it drops onto its own
  // visual line inside the flex container ``.sidebar-item``.
  // jsdom never loads the external ``main.css`` stylesheet that
  // ``chat.html`` references via ``{{STYLE_HREF}}``, so we read the
  // CSS file directly and assert that a matching rule exists.
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
      '"flex-basis: 100%" so the workspace+meta span renders on ' +
      'its own line below the metrics row',
  );
  console.log('  ok - workspace+meta span has flex-basis: 100%');
}

function main() {
  testWorkspaceLineMergesMetadata();
  testWorkspaceLineBreaksToOwnVisualLine();
  console.log('historyTaskMeta.test.js: all assertions passed.');
}

main();
