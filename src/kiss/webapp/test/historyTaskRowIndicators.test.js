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
const {
  freezeClockAndStyle,
  makeRow,
  historyRows,
  openChatsPanel,
  pressFilterOption,
} = require('./history_helpers');

const FROZEN_NOW_MS = 1_700_500_000_000;

function makePage() {
  return makeHarnessPage({
    beforeScripts: win => freezeClockAndStyle(win, FROZEN_NOW_MS),
  });
}

function indicatorOf(row) {
  return (
    row.querySelector('.sidebar-item-running') ||
    row.querySelector('.sidebar-item-completed') ||
    row.querySelector('.sidebar-item-failed')
  );
}

/** The states a row's status marks stand for (it must show exactly one). */
function marksOf(row) {
  return ['running', 'completed', 'failed'].filter(state =>
    row.querySelector('.sidebar-item-' + state),
  );
}

/** The titles of the rows `keep` accepts, sorted. */
function titlesWhere(win, keep) {
  return historyRows(win)
    .filter(keep)
    .map(r => r.querySelector('.sidebar-item-text').textContent)
    .sort();
}

function testRunningTaskShowsAtTopAfterBurgerOpen() {
  const {win, posted} = makePage();

  openChatsPanel(win);
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

  const r = historyRows(win);
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

function testFinishedTaskShowsGreenTick() {
  const {win, posted} = makePage();
  openChatsPanel(win);
  sendEmptyWorkDir(win);
  const getHist = posted.find(m => m && m.type === 'getHistory');
  assert.ok(getHist, 'burger menu open must post getHistory');

  const load = secondRuns =>
    send(win, {
      type: 'history',
      sessions: [
        makeRow({task_id: 1, title: 'settled task'}),
        makeRow(
          Object.assign(
            {task_id: 2, title: 'second task'},
            secondRuns ? {is_running: true, endTs: 0} : {},
          ),
        ),
        makeRow({task_id: 3, title: 'failed task', failed: true}),
      ],
      offset: 0,
      generation: getHist.generation,
    });

  // A task that finished successfully before the page loaded carries
  // the green tick, as a failed one carries its cross: every row shows
  // exactly one mark.
  load(true);
  let byTitle = rowsByTitle(win);
  assert.deepStrictEqual(marksOf(byTitle['settled task']), ['completed']);
  assert.deepStrictEqual(marksOf(byTitle['second task']), ['running']);
  assert.deepStrictEqual(marksOf(byTitle['failed task']), ['failed']);
  const tick = byTitle['settled task'].querySelector('.sidebar-item-completed');
  assert.strictEqual(
    byTitle['settled task'].firstElementChild,
    tick,
    'green tick must be the FIRST child (middle-left) of the row',
  );
  assert.ok(
    tick.classList.contains('status-tick'),
    'the completed indicator must be the shared .status-tick icon',
  );
  assert.ok(
    !tick.classList.contains('status-spinner'),
    'the completed indicator must not be a spinner',
  );
  assert.strictEqual(tick.getAttribute('aria-label'), 'Task completed');
  assert.strictEqual(tick.dataset.tooltip, 'Task completed');
  // Its colour and the absence of animation are pinned on a rendered
  // tick by historyBurgerMenuRunningTask.test.js.

  // The rows the tick marks are the rows the Succeeded filter counts:
  // with the filter off, exactly they go.
  const ticked = titlesWhere(win, r =>
    r.querySelector('.sidebar-item-completed'),
  );
  assert.deepStrictEqual(ticked, ['settled task']);
  pressFilterOption(win, 'status', 'Succeeded');
  assert.deepStrictEqual(
    titlesWhere(win, r => r.style.display === 'none'),
    ticked,
  );
  pressFilterOption(win, 'status', 'Succeeded');

  // The second task finishes: its spinner gives way to the tick, and
  // the other rows keep their marks.
  load(false);
  byTitle = rowsByTitle(win);
  assert.deepStrictEqual(marksOf(byTitle['settled task']), ['completed']);
  assert.deepStrictEqual(marksOf(byTitle['second task']), ['completed']);
  assert.deepStrictEqual(marksOf(byTitle['failed task']), ['failed']);

  win.close();
  console.log('  ok - every successfully finished task shows the green tick');
}

function testIndicatorsAreVerticallyCenteredInTaskPanels() {
  const {win, posted} = makePage();
  openChatsPanel(win);
  sendEmptyWorkDir(win);
  const getHist = posted.find(m => m && m.type === 'getHistory');
  assert.ok(getHist, 'burger menu open must post getHistory');

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

  // The panel's padding-top is the --space-1-5 token; the test reads
  // it from the page rather than pinning the designer's pixel value.
  const padTop = win
    .getComputedStyle(win.document.documentElement)
    .getPropertyValue('--space-1-5')
    .trim();
  assert.ok(padTop, '--space-1-5 is defined on the page');
  assert.strictEqual(historyRows(win).length, 3, 'the three task rows render');
  historyRows(win).forEach(row => {
    const title = row.querySelector('.sidebar-item-text').textContent;
    const indicator = indicatorOf(row);
    assert.ok(indicator, `row ${title} must render a status indicator`);
    assert.strictEqual(
      row.firstElementChild,
      indicator,
      `row ${title} indicator must stay at the left edge as first child`,
    );
    const style = win.getComputedStyle(indicator);
    // An unfolded panel shows its details below the task text, so it
    // may be taller than its title: the indicator is centered on the
    // first line of the text (the panel's padding-top plus half a line
    // box) instead of on the whole panel.
    assert.strictEqual(
      style.top,
      `calc(0.5lh + ${padTop})`,
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
  testFinishedTaskShowsGreenTick();
  testIndicatorsAreVerticallyCenteredInTaskPanels();
  console.log('historyTaskRowIndicators.test.js: all assertions passed.');
}

main();
