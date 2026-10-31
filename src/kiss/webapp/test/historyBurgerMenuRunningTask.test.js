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
} = require('./simplify2_harness');
const {
  freezeClockAndStyle,
  makeRow,
  rowByTitle,
  lastGetHistory,
  sendHistory,
} = require('./history_helpers');

const FROZEN_NOW_MS = 1_700_500_000_000;

function makePage() {
  return makeSharedPage({
    beforeScripts: win => freezeClockAndStyle(win, FROZEN_NOW_MS),
  });
}

function clickBurgerMenu(win) {
  const btn = win.document.getElementById('menu-btn');
  assert.ok(btn, 'burger menu button (#menu-btn) must exist');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function findDot(row) {
  return row.querySelector('.sidebar-item-running');
}

function findCompletedDot(row) {
  return row.querySelector('.sidebar-item-completed');
}

function testRunningTaskAppearsWhenBurgerOpened() {
  const {win, posted} = makePage();
  sendEmptyWorkDir(win);
  const sidebar = win.document.getElementById('sidebar');
  assert.ok(sidebar, 'sidebar element must exist');
  assert.ok(
    !sidebar.classList.contains('open'),
    'sidebar must start CLOSED to reproduce the regression flow',
  );

  posted.length = 0;
  send(win, {
    type: 'status',
    running: true,
    startTs: FROZEN_NOW_MS - 3_000,
    tabId: undefined,
  });
  assert.strictEqual(
    lastGetHistory(posted),
    undefined,
    'with sidebar closed, status running:true must NOT post getHistory; ' +
      'the burger click is what triggers the fetch',
  );

  posted.length = 0;
  clickBurgerMenu(win);
  assert.ok(
    sidebar.classList.contains('open'),
    'burger click must open the sidebar',
  );
  assert.ok(
    lastGetHistory(posted),
    'burger click must post a getHistory command so the History ' +
      'panel can list the freshly-started running task',
  );
  sendHistory(win, posted, [
    makeRow({
      task_id: 42,
      title: 'my running task',
      is_running: true,
      endTs: 0,
    }),
  ]);

  assert.ok(
    rowByTitle(win, 'my running task'),
    'the running task row MUST appear in the History list after ' +
      'the burger menu opens — this is the user-reported invariant',
  );
  // What the running row's indicator is and where it sits are pinned by
  // historyTaskRowIndicators.test.js and historyRunningSpinner.test.js.

  posted.length = 0;
  send(win, {type: 'status', running: false, tabId: undefined});
  assert.ok(
    lastGetHistory(posted),
    'status running:false with sidebar open MUST post getHistory ' +
      'so the row can swap pulsing for solid dot',
  );
  sendHistory(win, posted, [
    makeRow({
      task_id: 42,
      title: 'my running task',
      is_running: false,
      endTs: FROZEN_NOW_MS - 1_000,
    }),
  ]);

  const row2 = rowByTitle(win, 'my running task');
  assert.ok(row2, 'row must still exist after status running:false');
  assert.strictEqual(
    findDot(row2),
    null,
    'pulsing .sidebar-item-running dot MUST disappear once the task ends',
  );
  const completed = findCompletedDot(row2);
  assert.ok(
    completed,
    'finished task row MUST show a .sidebar-item-completed green tick ' +
      '(no animation)',
  );
  // The tick's place (first child) and icon (.status-tick) are pinned by
  // historyTaskRowIndicators.test.js; its colour and stillness here.

  const cs2 = win.getComputedStyle(completed);
  assert.strictEqual(
    cs2.color,
    'var(--green)',
    `tick colour must be the theme green (--status-ok -> var(--green)); got: ${cs2.color}`,
  );
  const anim2Name = cs2.getPropertyValue('animation-name') || '';
  const anim2Short = cs2.getPropertyValue('animation') || '';
  assert.ok(
    anim2Name.indexOf('status-spin') < 0 &&
      anim2Short.indexOf('status-spin') < 0,
    `tick must NOT animate; got animation-name="${anim2Name}" ` +
      `animation="${anim2Short}"`,
  );

  console.log(
    '  ok - running task appears via burger menu and swaps to a tick on finish',
  );
  win.close();
}

function main() {
  testRunningTaskAppearsWhenBurgerOpened();
  console.log('historyBurgerMenuRunningTask.test.js: all assertions passed.');
}

main();
