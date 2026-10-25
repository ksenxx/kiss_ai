// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage: makeHarnessPage,
  send,
  sendEmptyWorkDir,
  rowsByTitle,
} = require('./simplify2_harness');
const {freezeClockAndStyle, makeRow} = require('./history_helpers');

const FROZEN_NOW_MS = 1_700_500_000_000;

function makePage() {
  return makeHarnessPage({
    beforeScripts: win => freezeClockAndStyle(win, FROZEN_NOW_MS),
  });
}

function openBurgerMenu(win) {
  const btn = win.document.getElementById('menu-btn');
  assert.ok(btn, 'burger menu button (#menu-btn) must exist');
  btn.click();
}

function rows(win) {
  const list = win.document.getElementById('history-list');
  return list.querySelectorAll('.sidebar-item');
}

function indicatorOf(row) {
  return (
    row.querySelector('.sidebar-item-running') ||
    row.querySelector('.sidebar-item-completed') ||
    row.querySelector('.sidebar-item-failed')
  );
}

function testRunningTaskShowsAtTopAfterBurgerOpen() {
  const {win, posted} = makePage();

  openBurgerMenu(win);
  sendEmptyWorkDir(win);
  const getHist = posted.find(m => m && m.type === 'getHistory');
  assert.ok(
    getHist,
    'opening the burger menu must post getHistory; got: ' +
      JSON.stringify(posted),
  );

  const sessions = [
    makeRow({
      task_id: 101,
      title: 'NEW running task',
      is_running: true,
      timestamp: 1_700_100_000,
      startTs: 1_700_100_000_000,
      endTs: 0,
    }),
    makeRow({
      task_id: 100,
      title: 'OLD finished task',
      is_running: false,
      timestamp: 1_700_000_000,
    }),
  ];
  send(win, {
    type: 'history',
    sessions,
    offset: 0,
    generation: getHist.generation,
  });

  const r = rows(win);
  assert.strictEqual(r.length, 2, 'both rows must render');
  const firstText = r[0].querySelector('.sidebar-item-text');
  assert.ok(firstText, 'first row must have a text span');
  assert.strictEqual(
    firstText.textContent,
    'NEW running task',
    'the freshly-started running task MUST be the FIRST row in ' +
      'the History list when the burger menu is opened',
  );

  win.close();
  console.log(
    '  ok - running task is FIRST in History list when burger menu opens',
  );
}

function testFinishedTaskShowsSolidGreenCircle() {
  const {win, posted} = makePage();
  openBurgerMenu(win);
  sendEmptyWorkDir(win);
  const getHist = posted.find(m => m && m.type === 'getHistory');
  assert.ok(getHist, 'burger menu open must post getHistory');

  const initialSessions = [
    makeRow({task_id: 1, title: 'fresh completed task', is_running: false}),
    makeRow({
      task_id: 2,
      title: 'running task',
      is_running: true,
      endTs: 0,
    }),
    makeRow({
      task_id: 3,
      title: 'failed task',
      is_running: false,
      failed: true,
    }),
  ];
  send(win, {
    type: 'history',
    sessions: initialSessions,
    offset: 0,
    generation: getHist.generation,
  });

  let byTitle = rowsByTitle(win);

  const freshRow = byTitle['fresh completed task'];
  assert.ok(freshRow, 'fresh completed row must render');
  assert.strictEqual(
    freshRow.querySelector('.sidebar-item-completed'),
    null,
    'fresh history load of a completed task MUST NOT render a ' +
      'green tick — that is reserved for tasks the user ' +
      'just watched transition from running to completed',
  );
  assert.strictEqual(
    freshRow.querySelector('.sidebar-item-running'),
    null,
    'fresh completed row must not render a spinner either',
  );

  const runningRow = byTitle['running task'];
  assert.ok(
    runningRow.querySelector('.sidebar-item-running'),
    'running row must carry .sidebar-item-running',
  );

  const failedRow = byTitle['failed task'];
  assert.ok(
    failedRow.querySelector('.sidebar-item-failed'),
    'failed row must carry .sidebar-item-failed',
  );

  const finishedSessions = [
    makeRow({task_id: 1, title: 'fresh completed task', is_running: false}),
    makeRow({
      task_id: 2,
      title: 'running task',
      is_running: false,
      endTs: 1_700_000_010_000,
    }),
    makeRow({
      task_id: 3,
      title: 'failed task',
      is_running: false,
      failed: true,
    }),
  ];
  send(win, {
    type: 'history',
    sessions: finishedSessions,
    offset: 0,
    generation: getHist.generation,
  });
  byTitle = rowsByTitle(win);

  const transitionedRow = byTitle['running task'];
  const completedDot = transitionedRow.querySelector('.sidebar-item-completed');
  assert.ok(
    completedDot,
    'a row whose running→completed transition the session ' +
      'witnessed MUST render a .sidebar-item-completed green tick',
  );
  assert.strictEqual(
    transitionedRow.firstElementChild,
    completedDot,
    'green tick must be the FIRST child (middle-left) of the row',
  );
  assert.ok(
    completedDot.classList.contains('status-tick'),
    'the completed indicator must be the shared .status-tick icon',
  );
  assert.ok(
    !completedDot.classList.contains('status-spinner'),
    'the completed indicator must not be a spinner',
  );
  // Its colour and the absence of animation are pinned on a rendered
  // tick by historyBurgerMenuRunningTask.test.js.

  const stillFreshRow = byTitle['fresh completed task'];
  assert.strictEqual(
    stillFreshRow.querySelector('.sidebar-item-completed'),
    null,
    'an unrelated fresh-completed row MUST not inherit the solid ' +
      'green tick just because another row transitioned',
  );

  send(win, {
    type: 'history',
    sessions: finishedSessions,
    offset: 0,
    generation: getHist.generation,
  });
  const persisted = rows(win)[1];
  assert.ok(
    persisted.querySelector('.sidebar-item-completed'),
    'green tick MUST persist across subsequent ' +
      'history reloads once it has appeared',
  );

  win.close();
  console.log(
    '  ok - green tick only after witnessed running→completed transition',
  );
}

function testIndicatorsAreVerticallyCenteredInTaskPanels() {
  const {win, posted} = makePage();
  openBurgerMenu(win);
  sendEmptyWorkDir(win);
  const getHist = posted.find(m => m && m.type === 'getHistory');
  assert.ok(getHist, 'burger menu open must post getHistory');

  send(win, {
    type: 'history',
    sessions: [
      makeRow({
        task_id: 11,
        title: 'completed centered task',
        is_running: true,
        endTs: 0,
      }),
      makeRow({
        task_id: 12,
        title: 'running centered task',
        is_running: true,
        endTs: 0,
      }),
      makeRow({
        task_id: 13,
        title: 'failed centered task',
        failed: true,
      }),
    ],
    offset: 0,
    generation: getHist.generation,
  });

  send(win, {
    type: 'history',
    sessions: [
      makeRow({task_id: 11, title: 'completed centered task'}),
      makeRow({
        task_id: 12,
        title: 'running centered task',
        is_running: true,
        endTs: 0,
      }),
      makeRow({
        task_id: 13,
        title: 'failed centered task',
        failed: true,
      }),
    ],
    offset: 0,
    generation: getHist.generation,
  });

  rows(win).forEach(row => {
    const title = row.querySelector('.sidebar-item-text').textContent;
    const indicator = indicatorOf(row);
    assert.ok(indicator, `row ${title} must render a status indicator`);
    assert.strictEqual(
      row.firstElementChild,
      indicator,
      `row ${title} indicator must stay at the left edge as first child`,
    );
    const style = win.getComputedStyle(indicator);
    // The action buttons occupy a line of their own below the task
    // text, so the panel is taller than its title: the indicator is
    // centered on the first line of the text (the panel's padding-top
    // plus half a line box, --space-1-5 = 6px) instead of on the whole
    // panel.
    assert.strictEqual(
      style.top,
      'calc(0.5lh + 6px)',
      `row ${title} indicator must sit on the first line of the task ` +
        `text, not at the panel middle; got top=${style.top}`,
    );
    // The offset is the `translate` property, never `transform`: the
    // spinner's `rotate` animation is composed after `translate` but
    // before `transform`, so a `transform` offset would make the ring
    // orbit the point instead of turning in place.
    assert.strictEqual(
      style.translate,
      '0 -50%',
      `row ${title} indicator must translate by half its own height ` +
        `to center on that line; got translate=${style.translate}`,
    );
    assert.ok(
      style.transform === '' || style.transform === 'none',
      `row ${title} indicator must not be offset through transform ` +
        `(the spinner would revolve); got transform=${style.transform}`,
    );
  });

  win.close();
  console.log(
    '  ok - history task-panel indicators are centered at middle-left',
  );
}

function main() {
  testRunningTaskShowsAtTopAfterBurgerOpen();
  testFinishedTaskShowsSolidGreenCircle();
  testIndicatorsAreVerticallyCenteredInTaskPanels();
  console.log('historyTaskRowIndicators.test.js: all assertions passed.');
}

main();
