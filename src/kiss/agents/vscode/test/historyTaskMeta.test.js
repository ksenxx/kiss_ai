// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the per-task "meta" line in the History
// sidebar.
//
// Requirement driven by this test:
//
//   Every history row whose persisted ``extra`` carries a model
//   name must render a single dot-separated meta line below the
//   workspace line (or, if the row has no workspace, immediately
//   after the metrics line).  Format:
//
//       <model> • <wt|no-wt> • <parallel|sequential>
//         • <auto-commit|manual-commit>
//
//   * The meta span has class ``running-item-meta`` and is the
//     LAST child inside the ``.sidebar-item`` row (after the
//     optional workspace span and after the metrics span).
//   * Rows whose backend ``model`` is empty/missing must NOT
//     render a meta line at all (no blank line, no placeholder).
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

// Fixture covering every combination the meta line must handle:
//
//   * row A: all flags true, model + workspace present.
//   * row B: all flags false, model + workspace present.
//   * row C: model + flags present, but NO workspace — meta line
//     must still render, directly after the metrics span.
//   * row D: model missing — NO meta line at all.
//   * row E: model present, every boolean flag missing from the
//     payload — defaults to no-wt / sequential / manual-commit.
const WS_A = '/Users/koushik/work/repo-A';
const WS_B = 'C:\\Users\\koushik\\repo-B';

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
    title: 'meta row D — no model at all',
    timestamp: 1_700_000_300,
    preview: 'meta row D — no model at all',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: WS_A,
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

function metaSpan(row) {
  return row.querySelector('.running-item-meta');
}

function workspaceSpan(row) {
  return row.querySelector('.running-item-workspace');
}

function metricsSpan(row) {
  return row.querySelector('.running-item-metrics');
}

function testMetaRendersAfterWorkspaceOrMetrics() {
  const {win} = makeWebview();
  disableWorkspaceFilter(win);

  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  const rows = rowsByTitle(win);
  const a = rows['meta row A — all flags on'];
  const b = rows['meta row B — all flags off'];
  const c = rows['meta row C — model without workspace'];
  const d = rows['meta row D — no model at all'];
  const e = rows['meta row E — model only, flags missing'];
  assert.ok(a, 'row A must render');
  assert.ok(b, 'row B must render');
  assert.ok(c, 'row C must render');
  assert.ok(d, 'row D must render');
  assert.ok(e, 'row E must render');

  const aMeta = metaSpan(a);
  const bMeta = metaSpan(b);
  const cMeta = metaSpan(c);
  const eMeta = metaSpan(e);
  assert.ok(aMeta, 'row A must render a .running-item-meta span');
  assert.ok(bMeta, 'row B must render a .running-item-meta span');
  assert.ok(
    cMeta,
    'row C (no workspace) must still render a .running-item-meta span',
  );
  assert.ok(
    eMeta,
    'row E (flags missing) must still render a .running-item-meta span',
  );

  // Exact text — order: model, wt/no-wt, parallel/sequential,
  // auto-commit/manual-commit.  The separator is the same bullet
  // (``•``) used by the metrics row.
  assert.strictEqual(
    aMeta.textContent,
    'gpt-5 • wt • parallel • auto-commit',
    `row A meta text mismatch; got: ${aMeta.textContent}`,
  );
  assert.strictEqual(
    bMeta.textContent,
    'claude-3.7-sonnet • no-wt • sequential • manual-commit',
    `row B meta text mismatch; got: ${bMeta.textContent}`,
  );
  assert.strictEqual(
    cMeta.textContent,
    'gpt-5-mini • wt • sequential • auto-commit',
    `row C meta text mismatch; got: ${cMeta.textContent}`,
  );
  assert.strictEqual(
    eMeta.textContent,
    'legacy-model • no-wt • sequential • manual-commit',
    `row E meta text (default flags) mismatch; got: ${eMeta.textContent}`,
  );

  // Position requirement: when a workspace span is present, the
  // meta span must be the workspace span's IMMEDIATE next sibling
  // (so the meta line shows up directly below the workspace line).
  // When the workspace span is absent, the meta span must be the
  // metrics span's immediate next sibling instead.
  const aWs = workspaceSpan(a);
  const bWs = workspaceSpan(b);
  assert.ok(aWs, 'row A must keep its workspace span');
  assert.ok(bWs, 'row B must keep its workspace span');
  assert.strictEqual(
    aWs.nextElementSibling,
    aMeta,
    'row A: meta span must come immediately after workspace span',
  );
  assert.strictEqual(
    bWs.nextElementSibling,
    bMeta,
    'row B: meta span must come immediately after workspace span',
  );
  // Row C has no workspace — meta must follow metrics directly.
  const cMetrics = metricsSpan(c);
  assert.ok(cMetrics, 'row C must keep its metrics span');
  assert.strictEqual(
    workspaceSpan(c),
    null,
    'row C must NOT render a workspace span (empty work_dir)',
  );
  assert.strictEqual(
    cMetrics.nextElementSibling,
    cMeta,
    'row C: meta span must come immediately after metrics span ' +
      'when no workspace span is present',
  );

  // Row D: no model in payload → no meta line at all.
  assert.strictEqual(
    metaSpan(d),
    null,
    'row D (no model) must NOT render a meta span',
  );

  win.close();
  console.log(
    '  ok - meta renders on its own line in the documented order, ' +
      'omitted when the model field is missing',
  );
}

function testMetaLineBreaksToOwnVisualLine() {
  // The meta span must use ``flex-basis: 100%`` (same trick the
  // metrics and workspace rows use) so it drops onto its own
  // visual line inside the flex container ``.sidebar-item``.
  // jsdom never loads the external ``main.css`` stylesheet that
  // ``chat.html`` references via ``{{STYLE_HREF}}``, so we read the
  // CSS file directly and assert that a matching rule exists.
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const re = /\.running-item-meta\s*\{([^}]*)\}/g;
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
    '.running-item-meta rule in main.css must declare ' +
      '"flex-basis: 100%" so the meta span renders on its own ' +
      'line below the workspace / metrics row',
  );
  console.log('  ok - meta span has flex-basis: 100%');
}

function main() {
  testMetaRendersAfterWorkspaceOrMetrics();
  testMetaLineBreaksToOwnVisualLine();
  console.log('historyTaskMeta.test.js: all assertions passed.');
}

main();
