// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  loadMainCss,
  makePage,
  send,
  sendEmptyWorkDir,
  rowsByTitle,
} = require('./simplify2_harness');
const {makeRow} = require('./history_helpers');

const WS_A = '/Users/koushik/work/repo-A';
const WS_B = 'C:\\Users\\koushik\\repo-B';
const WS_D = '/Users/koushik/work/repo-D';

/** A finished row titled `title` (its preview too), 1 s long. */
function metaRow(id, title, timestamp, fields) {
  return makeRow({
    id,
    task_id: timestamp,
    title,
    preview: title,
    timestamp,
    startTs: timestamp * 1000,
    endTs: timestamp * 1000 + 1000,
    ...fields,
  });
}

// No work_dir key at all (an older row), as opposed to row F's ''.
const ROW_H = metaRow(
  'metaH',
  'meta row H — workspace field missing',
  1_700_000_700,
);
delete ROW_H.work_dir;

const SESSIONS_FIXTURE = [
  metaRow('metaA', 'meta row A — all flags on', 1_700_000_000, {
    work_dir: WS_A,
    model: 'gpt-5',
    is_parallel: true,
  }),
  metaRow('metaB', 'meta row B — all flags off', 1_700_000_100, {
    work_dir: WS_B,
    model: 'claude-3.7-sonnet',
    is_parallel: false,
  }),
  metaRow('metaC', 'meta row C — model without workspace', 1_700_000_200, {
    model: 'gpt-5-mini',
    is_parallel: false,
  }),
  metaRow('metaD', 'meta row D — workspace without model', 1_700_000_300, {
    work_dir: WS_D,
  }),
  metaRow('metaE', 'meta row E — model only, flags missing', 1_700_000_400, {
    model: 'legacy-model',
  }),
  metaRow('metaF', 'meta row F — neither workspace nor model', 1_700_000_500),
  metaRow(
    'metaG',
    'meta row G — windows workspace without model',
    1_700_000_600,
    {work_dir: WS_B},
  ),
  ROW_H,
];

function workspaceSpan(row) {
  return row.querySelector('.running-item-workspace');
}

function metricsSpan(row) {
  return row.querySelector('.running-item-metrics');
}

function testWorkspaceLineMergesMetadata(win) {
  const rows = rowsByTitle(win);
  const a = rows['meta row A — all flags on'];
  const b = rows['meta row B — all flags off'];
  const c = rows['meta row C — model without workspace'];
  const d = rows['meta row D — workspace without model'];
  const e = rows['meta row E — model only, flags missing'];
  const f = rows['meta row F — neither workspace nor model'];
  const g = rows['meta row G — windows workspace without model'];
  const h = rows['meta row H — workspace field missing'];
  assert.ok(a, 'row A must render');
  assert.ok(b, 'row B must render');
  assert.ok(c, 'row C must render');
  assert.ok(d, 'row D must render');
  assert.ok(e, 'row E must render');
  assert.ok(f, 'row F must render');
  assert.ok(g, 'row G must render');
  assert.ok(h, 'row H must render');

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

  assert.strictEqual(
    aWs.textContent,
    WS_A + ' • gpt-5 • parallel',
    `row A workspace+meta text mismatch; got: ${aWs.textContent}`,
  );
  assert.strictEqual(
    bWs.textContent,
    WS_B + ' • claude-3.7-sonnet • sequential',
    `row B workspace+meta text mismatch; got: ${bWs.textContent}`,
  );
  assert.strictEqual(
    cWs.textContent,
    'gpt-5-mini • sequential',
    `row C workspace+meta text mismatch; got: ${cWs.textContent}`,
  );
  assert.strictEqual(
    dWs.textContent,
    WS_D,
    `row D workspace text mismatch; got: ${dWs.textContent}`,
  );
  assert.strictEqual(
    eWs.textContent,
    'legacy-model • sequential',
    `row E workspace+meta text (default flags) mismatch; got: ${eWs.textContent}`,
  );

  assert.strictEqual(
    workspaceSpan(f),
    null,
    'row F (no workspace, no model) must NOT render a workspace span',
  );
  assert.strictEqual(
    workspaceSpan(g).textContent,
    WS_B,
    'row G (windows workspace, no model) must render the raw path',
  );
  assert.strictEqual(
    workspaceSpan(h),
    null,
    'row H (work_dir key missing, no model) must NOT render a workspace span',
  );

  for (const [label, row] of [
    ['A', a],
    ['B', b],
    ['C', c],
    ['D', d],
    ['E', e],
    ['G', g],
  ]) {
    const metrics = metricsSpan(row);
    assert.ok(metrics, `row ${label} must keep its metrics span`);
    assert.strictEqual(
      metrics.nextElementSibling,
      workspaceSpan(row),
      `row ${label}: workspace span must come immediately after metrics span`,
    );
  }

  for (const [label, row] of [
    ['A', a],
    ['B', b],
    ['C', c],
    ['D', d],
    ['E', e],
    ['F', f],
    ['G', g],
    ['H', h],
  ]) {
    assert.strictEqual(
      row.querySelector('.running-item-meta'),
      null,
      `row ${label} must NOT render a separate .running-item-meta span`,
    );
  }

  console.log(
    '  ok - workspace + metadata render on the same line, separated by " • "',
  );
}

function testWorkspaceLineBreaksToOwnVisualLine(win) {
  const span = workspaceSpan(rowsByTitle(win)['meta row A — all flags on']);
  assert.strictEqual(
    win.getComputedStyle(span).flexBasis,
    '100%',
    'the workspace+meta span takes the whole row width, so it renders ' +
      'on its own line below the metrics row',
  );
  console.log('  ok - workspace+meta span has flex-basis: 100%');
}

function main() {
  // One styled page with the fixture rendered serves both checks.
  const {win} = makePage();
  loadMainCss(win);
  sendEmptyWorkDir(win);
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});
  testWorkspaceLineMergesMetadata(win);
  testWorkspaceLineBreaksToOwnVisualLine(win);
  win.close();
  console.log('historyTaskMeta.test.js: all assertions passed.');
}

main();
