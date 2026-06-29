// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the per-task ids line in the History sidebar.
//
// Requirement driven by this test:
//
//   Every history row renders a third dot-separated line — the
//   "ids" line — immediately under the workspace+meta line.  The
//   line shows the row's chat id, task id, and parent task id
//   (when each is set) using a single ``.running-item-ids`` span.
//   Format:
//
//       chat <chat_id> • task <task_id> • parent <parent_task_id>
//
//   * Missing fields are skipped without a leading bullet or
//     placeholder.
//   * When NONE of the three ids are set, the row renders no
//     ``.running-item-ids`` span at all.
//   * The ids span is the LAST child of the per-row info column
//     (after metrics and workspace+meta), so the three lines
//     appear in this fixed top-to-bottom order:
//
//         steps • tokens • cost • duration
//         workspace • model • flags
//         chat <id> • task <id> • parent <id>
//
//   The metrics, workspace+meta, and ids spans must sit inside a
//   single per-row info container (``.running-item-info``) that
//   eliminates the vertical gap between them — the three lines
//   must touch visually, with no empty row-gap between them.
//
//   The info container itself drops onto its own line below the
//   task text via ``flex-basis: 100%`` (the same trick the
//   metrics span used in the old layout) and uses a column flex
//   with ``gap: 0`` so the three lines stack flush.
//
// This test drives the production ``media/main.js`` (plus the real
// ``media/chat.html`` markup and ``media/panelCopy.js``) inside jsdom,
// exactly like ``historyTaskMeta.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/historyTaskIds.test.js

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

// Fixture covering every combination the ids line must handle:
//
//   * row A: all three ids set (chat + task + parent_task_id).
//   * row B: chat + task, no parent (regular non-sub-agent row).
//   * row C: chat only — task_id null, no parent.
//   * row D: task only — chat empty, no parent.
//   * row E: parent only — chat empty, task_id null.
//   * row F: none of the three — NO ids span at all.
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

  // Row A — all three ids — full line.
  const aIds = idsSpan(a);
  assert.ok(aIds, 'row A must render a .running-item-ids span');
  assert.strictEqual(
    aIds.textContent,
    'chat chat-a • task task-a • parent parent-a',
    `row A ids text mismatch; got: ${aIds.textContent}`,
  );

  // Row B — chat + task, no parent.
  const bIds = idsSpan(b);
  assert.ok(bIds, 'row B must render a .running-item-ids span');
  assert.strictEqual(
    bIds.textContent,
    'chat chat-b • task task-b',
    `row B ids text mismatch; got: ${bIds.textContent}`,
  );

  // Row C — chat only.
  const cIds = idsSpan(c);
  assert.ok(cIds, 'row C must render a .running-item-ids span');
  assert.strictEqual(
    cIds.textContent,
    'chat chat-c',
    `row C ids text mismatch; got: ${cIds.textContent}`,
  );

  // Row D — task only.
  const dIds = idsSpan(d);
  assert.ok(dIds, 'row D must render a .running-item-ids span');
  assert.strictEqual(
    dIds.textContent,
    'task task-d',
    `row D ids text mismatch; got: ${dIds.textContent}`,
  );

  // Row E — parent only.
  const eIds = idsSpan(e);
  assert.ok(eIds, 'row E must render a .running-item-ids span');
  assert.strictEqual(
    eIds.textContent,
    'parent parent-e',
    `row E ids text mismatch; got: ${eIds.textContent}`,
  );

  // Row F — none of the three ids set → no span at all.
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

  // The three rendered lines (metrics, workspace+meta, ids) must
  // all sit inside the per-row ``.running-item-info`` column.
  // Wrapping them eliminates the flex row-gap that previously
  // showed up between the metrics and workspace lines.
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

    // Fixed order: metrics → workspace → ids.
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

    // The info block sits at the end of the row, after the
    // running/failed dot, the task text, and the action column.
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
  // jsdom never loads the external ``main.css`` stylesheet
  // referenced via ``{{STYLE_HREF}}``, so we read the CSS file
  // directly and assert the layout rules required to make the
  // three lines render flush (no visual gap between metrics and
  // workspace, no visual gap between workspace and ids).
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');

  // The info container must drop onto its own line below the
  // task text and stack its children vertically with zero gap.
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
  // Either ``gap: 0`` or ``row-gap: 0`` must zero the vertical
  // spacing between metrics, workspace, and ids so they render
  // flush with no visual gap.
  assert.match(
    infoBody,
    /(^|[^-])(?:row-)?gap\s*:\s*0(?:px)?\b/,
    '.running-item-info must declare gap: 0 (or row-gap: 0) so ' +
      'metrics / workspace / ids render flush with no visual gap',
  );

  // The ids span itself needs its own rule so it inherits the
  // sidebar text colour, clips long content, and matches the
  // small font used by the metrics / workspace lines.
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
