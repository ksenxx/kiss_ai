// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage: makeSharedPage,
  send,
  sendEmptyWorkDir,
  rowsByTitle: rowsByTaskId,
  sleep,
} = require('./simplify2_harness');
const {freezeClockAndStyle, makeRow} = require('./history_helpers');

const FROZEN_NOW_MS = 1_700_500_000_000;

function makePage() {
  const {win, posted} = makeSharedPage({beforeScripts: win => freezeClockAndStyle(win, FROZEN_NOW_MS)});
  return {win, posted};
}

/** Let main.js's end-of-tick history refetch coalescing settle. */
function tick() {
  return sleep(0);
}

function openSidebar(win) {
  const sidebar = win.document.getElementById('sidebar');
  if (sidebar && !sidebar.classList.contains('open')) {
    sidebar.classList.add('open');
  }
}

function dotOf(row) {
  return row.querySelector('.sidebar-item-running');
}

function testDotRendersForRunningRow() {
  const {win} = makePage();
  openSidebar(win);
  sendEmptyWorkDir(win);

  const SESSIONS = [
    makeRow({task_id: 1, title: 'finished task', is_running: false}),
    makeRow({task_id: 2, title: 'running task', is_running: true, endTs: 0}),
    makeRow({
      task_id: 3,
      title: 'failed task',
      is_running: false,
      failed: true,
    }),
  ];
  send(win, {type: 'history', sessions: SESSIONS, offset: 0});

  const rows = rowsByTaskId(win);
  const finished = rows['finished task'];
  const running = rows['running task'];
  const failed = rows['failed task'];
  assert.ok(finished && running && failed, 'all rows must render');

  assert.strictEqual(
    dotOf(finished),
    null,
    'finished row must NOT carry .sidebar-item-running',
  );
  assert.strictEqual(
    dotOf(failed),
    null,
    'failed row must NOT carry .sidebar-item-running',
  );

  // The row carrying `.sidebar-item-running` at all is
  // historyTaskRowIndicators'; this suite pins what the indicator is.
  const dot = dotOf(running);
  assert.strictEqual(
    running.firstElementChild,
    dot,
    'running spinner must be the first child of the row (left of text)',
  );
  assert.ok(
    dot.classList.contains('status-spinner'),
    'the running indicator must be the shared .status-spinner icon',
  );
  assert.strictEqual(
    dot.textContent,
    '',
    'the spinner is drawn by CSS, not by a text glyph',
  );

  const cs = win.getComputedStyle(dot);
  assert.strictEqual(
    cs.color,
    'var(--green)',
    `spinner colour must be the theme green (--status-ok -> var(--green)); got: ${cs.color}`,
  );
  const animName = cs.getPropertyValue('animation-name') || '';
  const animShort = cs.getPropertyValue('animation') || '';
  assert.ok(
    animName.indexOf('status-spin') >= 0 ||
      animShort.indexOf('status-spin') >= 0,
    `spinner must animate via 'status-spin'; got animation-name=` +
      `"${animName}" animation="${animShort}"`,
  );
  assert.ok(
    animName.indexOf('running-pulse') < 0 &&
      animShort.indexOf('running-pulse') < 0,
    'the running indicator must spin, not pulse',
  );

  assert.strictEqual(running.dataset.category, 'running');

  win.close();
  console.log('  ok - green spinner renders on is_running rows');
}

function testDotAppearsLiveOnStatusRunningTrue() {
  const {win, posted} = makePage();
  openSidebar(win);
  sendEmptyWorkDir(win);

  const initial = [
    makeRow({task_id: 7, title: 'live task', is_running: false}),
  ];
  send(win, {type: 'history', sessions: initial, offset: 0});

  let rows = rowsByTaskId(win);
  assert.ok(rows['live task'], 'row must render initially');
  assert.strictEqual(
    dotOf(rows['live task']),
    null,
    'no dot before the task starts running',
  );

  posted.length = 0;
  send(win, {
    type: 'status',
    running: true,
    startTs: FROZEN_NOW_MS - 5_000,
    tabId: undefined,
  });
  const sent = posted.find(m => m && m.type === 'getHistory');
  assert.ok(
    sent,
    'status running:true must trigger a getHistory refetch ' +
      'so the History panel can repaint with is_running:true; ' +
      'posted=' +
      JSON.stringify(posted),
  );

  const generation = sent.generation;
  const live = [
    makeRow({task_id: 7, title: 'live task', is_running: true, endTs: 0}),
  ];
  send(win, {
    type: 'history',
    sessions: live,
    offset: 0,
    generation: generation,
  });

  rows = rowsByTaskId(win);
  const row = rows['live task'];
  assert.ok(row, 'row must still render after the live refresh');
  const dot = dotOf(row);
  assert.ok(dot, 'spinner must appear on the row after status running:true');
  assert.strictEqual(
    row.firstElementChild,
    dot,
    'live-added spinner must still be the first child of the row',
  );

  win.close();
  console.log('  ok - dot appears live when status flips to running');
}

async function testDotDisappearsLiveOnStatusRunningFalse() {
  const {win, posted} = makePage();
  openSidebar(win);
  sendEmptyWorkDir(win);

  const initial = [
    makeRow({task_id: 9, title: 'ending task', is_running: true, endTs: 0}),
  ];
  send(win, {type: 'history', sessions: initial, offset: 0});

  let rows = rowsByTaskId(win);
  assert.ok(dotOf(rows['ending task']), 'dot must be present initially');

  // Every status refetches history (main.js `case 'status'` calls
  // refreshHistory unconditionally); the one that matters here is the
  // running:false that must drop the spinner.  A refresh while the
  // previous one's page is still in flight folds into it, so answer
  // that page before the status under test.
  send(win, {type: 'status', running: true, tabId: undefined});
  await tick();
  const pending = posted.find(m => m && m.type === 'getHistory');
  send(win, {
    type: 'history', sessions: initial, offset: 0, generation: pending.generation,
  });
  posted.length = 0;
  send(win, {type: 'status', running: false, tabId: undefined});
  const sent = posted.find(m => m && m.type === 'getHistory');
  assert.ok(
    sent,
    'status running:false must trigger a getHistory refetch ' +
      'so the History panel can drop the spinner',
  );

  const generation = sent.generation;
  const finished = [
    makeRow({
      task_id: 9,
      title: 'ending task',
      is_running: false,
      endTs: 1_700_000_010_000,
    }),
  ];
  send(win, {
    type: 'history',
    sessions: finished,
    offset: 0,
    generation: generation,
  });

  rows = rowsByTaskId(win);
  assert.strictEqual(
    dotOf(rows['ending task']),
    null,
    'spinner must be removed after status running:false reply',
  );

  win.close();
  console.log('  ok - dot disappears live when status flips to not-running');
}

async function main() {
  testDotRendersForRunningRow();
  testDotAppearsLiveOnStatusRunningTrue();
  await testDotDisappearsLiveOnStatusRunningFalse();
  console.log('historyRunningPulsingDot.test.js: all assertions passed.');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
