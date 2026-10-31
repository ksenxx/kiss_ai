// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  sendEmptyWorkDir,
  rowsByTitle,
  loadMainCss,
} = require('./simplify2_harness');
const {makeRow} = require('./history_helpers');

const WS = '/Users/koushik/work/repo';

// Every combination of a present / missing chat id, task id and parent
// task id (rows C-F are id-less in one way or another).
const SESSIONS_FIXTURE = [
  makeRow({
    id: 'chat-a',
    task_id: 'task-a',
    parent_task_id: 'parent-a',
    title: 'ids row A — chat + task + parent',
    work_dir: WS,
    model: 'gpt-5',
  }),
  makeRow({
    id: 'chat-b',
    task_id: 'task-b',
    title: 'ids row B — chat + task, no parent',
    timestamp: 1_700_000_100,
    work_dir: WS,
    model: 'gpt-5',
  }),
  makeRow({id: 'chat-c', task_id: null, title: 'ids row C — chat only', timestamp: 1_700_000_200}),
  makeRow({id: '', task_id: 'task-d', title: 'ids row D — task only', timestamp: 1_700_000_300}),
  makeRow({
    id: '',
    task_id: null,
    parent_task_id: 'parent-e',
    title: 'ids row E — parent only',
    timestamp: 1_700_000_400,
  }),
  makeRow({
    id: '',
    task_id: null,
    title: 'ids row F — neither chat, task, nor parent',
    timestamp: 1_700_000_500,
  }),
];

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

/** Boot a page with main.css applied and the fixture rows rendered. */
function renderedRows() {
  const {win} = makePage();
  loadMainCss(win);
  sendEmptyWorkDir(win);
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});
  return {win, rows: rowsByTitle(win)};
}

function testIdsLineRendersAllCombinations(rows) {
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

  console.log(
    '  ok - .running-item-ids renders the right text in every combination',
  );
}

function testIdsLineOrderingAndContainer(win, rows) {
  const a = rows['ids row A — chat + task + parent'];
  const b = rows['ids row B — chat + task, no parent'];
  assert.ok(a, 'row A must render');
  assert.ok(b, 'row B must render');

  for (const [label, row] of [
    ['A', a],
    ['B', b],
  ]) {
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

    // main.css: the info block drops onto its own line of the
    // .sidebar-item flex row (remote-codex.css runs its lines together
    // as one " • "-separated text block).
    const style = win.getComputedStyle(info);
    assert.strictEqual(style.flexBasis, '100%', `row ${label}: own line`);
  }

  console.log(
    '  ok - metrics / workspace / ids share the .running-item-info line',
  );
}

function main() {
  const {win, rows} = renderedRows();
  testIdsLineRendersAllCombinations(rows);
  testIdsLineOrderingAndContainer(win, rows);
  win.close();
  console.log('historyTaskIds.test.js: all assertions passed.');
}

main();
