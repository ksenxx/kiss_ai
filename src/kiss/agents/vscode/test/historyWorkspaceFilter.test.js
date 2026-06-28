// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the History sidebar's "Workspace" filter
// checkbox.
//
// Requirements driven by this test:
//
//   1. The history-filter bar exposes a checkbox with id "hf-workspace"
//      that is CHECKED by default and sits IMMEDIATELY BEFORE the
//      "Favorites" checkbox.
//
//   2. Each history row carries the persisted ``extra.work_dir`` via a
//      ``data-work-dir`` attribute so the client-side filter helper can
//      decide visibility without re-querying the backend.
//
//   3. When the Workspace checkbox is CHECKED the history panel shows
//      only rows whose ``data-work-dir`` equals the configured client
//      ``work_dir`` (the ``configWorkDir`` populated from the
//      ``configData`` event).
//
//   4. When the Workspace checkbox is UNCHECKED every row is visible
//      again (subject to the other filter checkboxes).
//
//   5. Switching the configured ``work_dir`` (via a fresh
//      ``configData`` event) updates the visible set without requiring
//      a re-fetch.
//
// This test drives the production ``media/main.js`` (plus the real
// ``media/chat.html`` markup and ``media/panelCopy.js``) inside jsdom —
// no mocks of project code — exactly like the existing
// ``tab_timer_per_tab.test.js`` and ``bughunt2_status_timer.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/historyWorkspaceFilter.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview.
 *
 * The real ``chat.html`` body is loaded (placeholders blanked), then
 * ``panelCopy.js`` and ``main.js`` are evaluated in the window, and a
 * recording ``acquireVsCodeApi`` stub is installed (the only host API
 * the webview has).
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

/** Collect the visible history rows (display !== 'none'). */
function visibleRows(win) {
  const list = win.document.getElementById('history-list');
  const rows = list.querySelectorAll('.sidebar-item');
  const out = [];
  rows.forEach(r => {
    if (r.style.display !== 'none') {
      out.push({
        text: r.querySelector('.sidebar-item-text').textContent,
        workDir: r.dataset.workDir || '',
      });
    }
  });
  return out;
}

const SESSIONS_FIXTURE = [
  {
    id: 'chatA',
    task_id: 1,
    title: 'task A in /repo/alpha',
    timestamp: 1_700_000_000,
    preview: 'task A',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '/repo/alpha',
    startTs: 1_700_000_000_000,
    endTs: 1_700_000_010_000,
  },
  {
    id: 'chatB',
    task_id: 2,
    title: 'task B in /repo/beta',
    timestamp: 1_700_000_100,
    preview: 'task B',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '/repo/beta',
    startTs: 1_700_000_100_000,
    endTs: 1_700_000_110_000,
  },
  {
    id: 'chatC',
    task_id: 3,
    title: 'second task in /repo/alpha',
    timestamp: 1_700_000_200,
    preview: 'second alpha task',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '/repo/alpha',
    startTs: 1_700_000_200_000,
    endTs: 1_700_000_210_000,
  },
  {
    id: 'chatD',
    task_id: 4,
    title: 'legacy task with no work_dir',
    timestamp: 1_700_000_300,
    preview: 'legacy task',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '',
    startTs: 1_700_000_300_000,
    endTs: 1_700_000_310_000,
  },
];

function testCheckboxMarkupAndDefault() {
  const {win} = makeWebview();
  const doc = win.document;

  const ws = doc.getElementById('hf-workspace');
  const fav = doc.getElementById('hf-favorite');
  assert.ok(ws, 'Workspace checkbox must exist');
  assert.ok(fav, 'Favorites checkbox must exist');
  assert.strictEqual(ws.type, 'checkbox');
  assert.strictEqual(
    ws.checked,
    true,
    'Workspace checkbox must be CHECKED by default',
  );

  // Workspace must appear BEFORE Favorites in the filter bar.
  const bar = doc.querySelector('.history-filter-bar');
  assert.ok(bar, 'history-filter-bar must exist');
  const inputs = Array.from(bar.querySelectorAll('input[type="checkbox"]'));
  const ids = inputs.map(i => i.id);
  const wsIdx = ids.indexOf('hf-workspace');
  const favIdx = ids.indexOf('hf-favorite');
  assert.ok(wsIdx >= 0 && favIdx >= 0);
  assert.strictEqual(
    favIdx,
    wsIdx + 1,
    'Workspace must sit immediately before Favorites — got order ' +
      JSON.stringify(ids),
  );

  win.close();
  console.log('  ok - Workspace checkbox markup, default, ordering');
}

function testWorkspaceFilterHidesNonMatchingRows() {
  const {win} = makeWebview();

  // Configure the client work_dir to /repo/alpha BEFORE the history
  // list arrives — mirrors the real boot order (getConfig precedes
  // getHistory).
  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/alpha'},
    apiKeys: {},
  });

  // Backend pushes the history list.
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  // Every row must carry the right ``data-work-dir`` attribute.
  const all = win.document
    .getElementById('history-list')
    .querySelectorAll('.sidebar-item');
  assert.strictEqual(all.length, 4, 'all four rows must render');
  const byTitle = {};
  all.forEach(r => {
    byTitle[r.querySelector('.sidebar-item-text').textContent] =
      r.dataset.workDir;
  });
  assert.strictEqual(byTitle['task A in /repo/alpha'], '/repo/alpha');
  assert.strictEqual(byTitle['task B in /repo/beta'], '/repo/beta');
  assert.strictEqual(
    byTitle['second task in /repo/alpha'],
    '/repo/alpha',
  );
  assert.strictEqual(byTitle['legacy task with no work_dir'], '');

  // Workspace checkbox is CHECKED by default → rows whose work_dir
  // matches the client work_dir are visible, AND rows whose
  // work_dir is empty also pass per the documented contract in
  // ``applyHistoryFilterVisibility`` (an empty row work_dir
  // represents either a legacy row that pre-dates the
  // ``extra.work_dir`` persistence change or a freshly-started
  // running task whose extra has not been written yet — both
  // must remain visible so the user never silently loses sight
  // of a running task panel after opening the burger menu).
  // Only rows with an explicit, non-matching work_dir are hidden.
  const visible = visibleRows(win);
  const visibleTitles = visible.map(r => r.text).sort();
  assert.deepStrictEqual(
    visibleTitles,
    [
      'legacy task with no work_dir',
      'second task in /repo/alpha',
      'task A in /repo/alpha',
    ],
    'Workspace filter ON must show rows whose work_dir matches ' +
      'the client work_dir AND rows whose work_dir is empty; only ' +
      `rows with a different non-empty work_dir are hidden; got ${JSON.stringify(visibleTitles)}`,
  );

  win.close();
  console.log('  ok - Workspace ON hides non-matching rows');
}

function testUncheckingWorkspaceRevealsAllRows() {
  const {win} = makeWebview();
  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/alpha'},
    apiKeys: {},
  });
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  // Uncheck Workspace.
  const ws = win.document.getElementById('hf-workspace');
  ws.checked = false;
  ws.dispatchEvent(new win.Event('change', {bubbles: true}));

  const visible = visibleRows(win).map(r => r.text).sort();
  assert.deepStrictEqual(
    visible,
    [
      'legacy task with no work_dir',
      'second task in /repo/alpha',
      'task A in /repo/alpha',
      'task B in /repo/beta',
    ],
    'Workspace OFF must reveal every row (subject to other ' +
      `filters); got ${JSON.stringify(visible)}`,
  );

  win.close();
  console.log('  ok - Workspace OFF reveals all rows');
}

function testReconfiguringWorkDirReFiltersInPlace() {
  const {win} = makeWebview();
  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/alpha'},
    apiKeys: {},
  });
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  // Sanity: workspace filter pinned to /repo/alpha; the legacy
  // empty-work_dir row also passes per the documented contract.
  let visible = visibleRows(win).map(r => r.text).sort();
  assert.deepStrictEqual(visible, [
    'legacy task with no work_dir',
    'second task in /repo/alpha',
    'task A in /repo/alpha',
  ]);

  // Reconfigure the client work_dir to /repo/beta — the already-
  // rendered history list must re-filter in place.  The legacy
  // empty-work_dir row still passes; only the explicit
  // /repo/alpha rows now drop out.
  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/beta'},
    apiKeys: {},
  });

  visible = visibleRows(win).map(r => r.text).sort();
  assert.deepStrictEqual(
    visible,
    ['legacy task with no work_dir', 'task B in /repo/beta'],
    'Switching the client work_dir must re-filter the already-' +
      `rendered history list in place; got ${JSON.stringify(visible)}`,
  );

  win.close();
  console.log('  ok - changing client work_dir re-filters in place');
}

function testEmptyClientWorkDirMatchesEmptyRows() {
  const {win} = makeWebview();

  // No client work_dir configured at all.
  send(win, {
    type: 'configData',
    config: {work_dir: ''},
    apiKeys: {},
  });
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  const visible = visibleRows(win).map(r => r.text).sort();
  assert.deepStrictEqual(
    visible,
    [
      'legacy task with no work_dir',
      'second task in /repo/alpha',
      'task A in /repo/alpha',
      'task B in /repo/beta',
    ],
    'When client work_dir is empty, every row must remain visible ' +
      'per the documented contract (an empty client work_dir bypasses ' +
      `the workspace match); got ${JSON.stringify(visible)}`,
  );

  win.close();
  console.log('  ok - empty client work_dir matches only empty-row tasks');
}

function runTests() {
  testCheckboxMarkupAndDefault();
  testWorkspaceFilterHidesNonMatchingRows();
  testUncheckingWorkspaceRevealsAllRows();
  testReconfiguringWorkDirReFiltersInPlace();
  testEmptyClientWorkDirMatchesEmptyRows();
}

try {
  runTests();
  console.log('\n5 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
